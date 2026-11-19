import argparse
from collections import Counter
from pathlib import Path
from analysis.common import study as c
from analysis.common import cached_retrieval as retrieval
from analysis.common import diagnose_replay as diagnosis
from analysis.common import prompt as prompt

BASE = c.BASE / "replay_review_20261009_v1"
SUMMARY = BASE / "diagnostic_summary.json"
STATUS = "retrieved_with_reviewed_cold_replay_differences"


def trace_changes(left, right):
    c.require(len(left) == len(right), "Trace operation count changed")
    result = []
    for i, (a, b) in enumerate(zip(left, right)):
        c.require(a["operation"] == b["operation"], "Trace operation order changed")
        if a == b:
            continue
        detail = dict(index=i, operation=a["operation"])
        if "hits" in a:
            aa = {r["id"]: r for r in a["hits"]}
            bb = {r["id"]: r for r in b["hits"]}
            detail.update(
                same_candidate_set=set(aa) == set(bb),
                same_order=[r["id"] for r in a["hits"]] == [r["id"] for r in b["hits"]],
                common_hit_values_equal=all(
                    (aa[k] == bb[k] for k in aa.keys() & bb.keys())
                ),
                only_left=sorted(aa.keys() - bb.keys()),
                only_right=sorted(bb.keys() - aa.keys()),
            )
        if "selected" in a:
            detail["same_selection"] = a["selected"] == b["selected"]
        result.append(detail)
    return result


def exclusion_roster(roster, keys):

    bykey = {r["key"]: r for r in roster}
    c.require(len(bykey) == len(roster), "Duplicate roster identity")
    c.require(set(keys) <= set(bykey), "Unknown sensitivity identity")
    return [
        dict(key=r["key"], population=r["phase"])
        for r in roster
        if r["key"] in set(keys)
    ]


def summarize():
    if SUMMARY.exists():
        saved = c.read(SUMMARY)
        c.verify(saved["inputs"])
        return saved
    p = c.read(diagnosis.PLAN)
    c.verify(p["inputs"])
    retrieval.validate()
    audit = c.read(diagnosis.BASE / "audit.json")
    roster = c.rows(c.PREP / "broad_roster.jsonl")
    bykey = {r["key"]: r for r in roster}
    inputs = dict(p["inputs"])
    inputs[c.rel(diagnosis.PLAN)] = c.sha(diagnosis.PLAN)
    for path in [
        Path(__file__),
        c.ROOT / "tests/common/test_replay_review.py",
        c.ROOT / "scripts/finish_comparison_preparation.ps1",
    ]:
        inputs[c.rel(path)] = c.sha(path)
    summary = {}
    diagnostic_keys = {}
    total = 0
    for pool in ("small", "full"):
        selected = p["selections"][pool]
        keys = selected["all"]
        known = {r["id"]: r for r in c.rows(c.CORE / f"kb/{pool}/records.jsonl")}
        originals = {
            r["key"]: r
            for r in diagnosis.stream(retrieval.BASE / pool / "prompts.jsonl")
            if r["key"] in keys
        }
        grouped = {k: [] for k in keys}
        cold_info = []
        message_changed = set()
        controls_changed = set()
        for number in (1, 2):
            path = diagnosis.BASE / pool / f"cold_{number}.json"
            doc = c.read(path)
            c.require(
                doc["plan_sha256"] == c.sha(diagnosis.PLAN)
                and doc["pool"] == pool
                and (doc["cold_pass"] == number),
                "Wrong diagnostic identity",
            )
            c.require(
                [r["key"] for r in doc["rows"]] == keys and doc["Qwen_calls"] == 0,
                "Incomplete diagnostic roster",
            )
            c.require(
                doc["index_vector_max_abs_difference"] <= 2e-07,
                "Diagnostic collection vector mismatch",
            )
            inputs[c.rel(path)] = c.sha(path)
            kinds = Counter()
            within = []
            for row in doc["rows"]:
                key = row["key"]
                c.require(len(row["repeats"]) == 2, "Missing diagnostic repeat")
                c.require(
                    row["affected_in_full_passes"] == (key in selected["affected"]),
                    "Incorrect control designation",
                )
                first = originals[key]
                for rep in row["repeats"]:
                    entry = rep["actual_entry"]
                    diagnosis.check_entry(
                        entry, bykey[key], known, pool, c.sha(retrieval.PLAN)
                    )
                    comp = diagnosis.compare(first, entry)
                    c.require(
                        comp == rep["comparison"], "Incorrect targeted comparison"
                    )
                    c.require(
                        set(comp["changed_fields"])
                        <= {
                            "context",
                            "context_sha256",
                            "messages",
                            "messages_sha256",
                            "retrieved",
                        },
                        "Changed diagnostic source identity",
                    )
                    c.require(
                        entry["messages"]
                        == prompt.render_messages(
                            bykey[key]["text"],
                            entry["lang"],
                            entry["context"],
                            "corrected",
                        ),
                        "Diagnostic render differs",
                    )
                    c.require(
                        set(comp["changed_buckets"]) <= {"Labelled examples"},
                        "Unexpected definition/guideline change",
                    )
                    c.require(
                        not comp["same_messages"]
                        or comp["context_equal_without_ranks"],
                        "Unexpected same-message content change",
                    )
                    if not comp["same_messages"]:
                        message_changed.add(key)
                    if comp["kind"] != "exact" and (not row["affected_in_full_passes"]):
                        controls_changed.add(key)
                    kinds[comp["kind"]] += 1
                    grouped[key].append(rep)
                    total += 1
                if row["repeats"][0] != row["repeats"][1]:
                    within.append(key)
            cold_info.append(
                dict(
                    cold_pass=number,
                    comparisons=dict(kinds),
                    within_process_changes=within,
                    index_vector_max_abs_difference=doc[
                        "index_vector_max_abs_difference"
                    ],
                    changed_physical_copy_files=[
                        k
                        for k in doc["physical_before"].keys()
                        | doc["physical_after"].keys()
                        if doc["physical_before"].get(k) != doc["physical_after"].get(k)
                    ],
                )
            )
        cross = []
        for key, values in grouped.items():
            changes = trace_changes(values[0]["trace"], values[2]["trace"])
            if changes:
                cross.append(
                    dict(
                        key=key,
                        operations=changes,
                        prompt_changed_between_processes=values[0]["actual_entry"][
                            "messages"
                        ]
                        != values[2]["actual_entry"]["messages"],
                    )
                )
        diagnostic_keys[pool] = sorted(message_changed)
        summary[pool] = dict(
            inputs=len(keys),
            affected_inputs=len(selected["affected"]),
            controls=len(selected["controls"]),
            cold_passes=cold_info,
            cross_process_trace_changes=cross,
            diagnostic_message_keys=sorted(message_changed),
            new_message_keys_beyond_full_passes=sorted(
                message_changed - set(audit["changes"][pool]["message_union"])
            ),
            controls_changed=sorted(controls_changed),
        )
    observed = {
        pool: sorted(
            set(audit["changes"][pool]["message_union"]) | set(diagnostic_keys[pool])
        )
        for pool in ("small", "full")
    }
    result = dict(
        status="targeted_diagnostics_verified",
        created_utc=c.utc(),
        inputs=inputs,
        diagnostic_retrievals=total,
        all_diagnostic_entries_source_and_render_verified=True,
        pools=summary,
        observed_message_keys_by_pool=observed,
        paired_sensitivity_roster=exclusion_roster(
            roster, set(observed["small"]) | set(observed["full"])
        ),
        mechanism="Full-pool fresh-process traces localize observed changes to dense candidate membership before MMR/RRF. Shared candidates retain exactly equal scores/vectors; lexical traces are stable in this targeted check. Small-pool full-pass differences were not reproduced here. A unique internal Chroma cause is not established.",
        limits="Targeted affected/control sample, not a new population instability estimate. No Qwen outcomes or accuracy effect have been measured.",
        actual_Qwen_calls=0,
    )
    c.verify(inputs)
    c.save(SUMMARY, result)
    print(
        "Targeted diagnostic review saved:",
        total,
        "retrievals; paired exclusion roster",
        len(result["paired_sensitivity_roster"]),
        flush=True,
    )
    return result


def finalize():
    done = retrieval.BASE / "complete.json"
    if done.exists():
        existing = c.read(done)
        c.require(
            existing["status"] == STATUS,
            "Unexpected pre-existing completion; preserve it",
        )
        c.verify(existing["outputs"])
        print("Reviewed completion already verified.")
        return
    c.require(
        not (c.BASE / "execution/broad/live").exists(),
        "Do not adopt a review after decoder execution began",
    )
    p = retrieval.validate()
    summary = summarize()
    audit = c.read(diagnosis.BASE / "audit.json")
    c.require(
        "First-pass prompts are authoritative" in p["index_policy"],
        "First-pass policy was not predeclared",
    )
    for pool in ("small", "full"):
        info = summary["pools"][pool]
        c.require(
            not info["controls_changed"]
            and (not info["new_message_keys_beyond_full_passes"]),
            "Additional scope of differences requires separate review",
        )
        c.require(
            all((not row["within_process_changes"] for row in info["cold_passes"])),
            "Within-process instability requires separate review",
        )
    c.require(
        audit["status"] == "all_six_saved_passes_verified_nonzero_differences"
        and audit["entries_checked"] == 43299
        and audit["original_indexes_unchanged"],
        "Saved-pass integrity audit required",
    )
    outputs = dict(summary["inputs"])
    outputs[c.rel(SUMMARY)] = c.sha(SUMMARY)
    sensitivity = BASE / "sensitivity_plan.json"
    if not sensitivity.exists():
        c.save(
            sensitivity,
            dict(
                created_utc=c.utc(),
                status="declared_before_broader_Qwen_outcomes",
                inputs={c.rel(SUMMARY): c.sha(SUMMARY)},
                policy="Keep all inputs and first-pass prompts in every primary comparison. After outcomes exist, repeat all planned broad paired system contrasts with these same observed prompt-changing keys excluded from every system, including zero-shot and encoders. Keep original full-roster endpoint label sets, null handling, whole-text clustering and family adjustments. Do not select results, replace prompts or tune retrieval from outcomes.",
                exclusion_roster=summary["paired_sensitivity_roster"],
                relationship_to_empty_input_sensitivity="Report separately from the predeclared German whitespace sensitivity; do not substitute one for the other.",
                limitations="This checks sensitivity to the observed changing cases only; it does not measure predictions under alternative retrievals or bound unobserved live retrieval variability.",
                execution_state="outcome_analysis_pending",
            ),
        )
    outputs[c.rel(sensitivity)] = c.sha(sensitivity)
    empty = {"definitions": [], "guidelines": [], "example_groups": []}
    zero = retrieval.BASE / "zero_shot.jsonl"
    c.save_rows(
        zero,
        [
            dict(
                key=r["key"],
                item_id=str(r["id"]),
                lang=r["lang"],
                population=r["phase"],
                condition="zero_shot",
                messages=(
                    m := prompt.render_messages(
                        r["text"], r["lang"], empty, "corrected"
                    )
                ),
                messages_sha256=c.signature(m),
            )
            for r in c.rows(c.PREP / "broad_roster.jsonl")
        ],
    )
    outputs[c.rel(zero)] = c.sha(zero)
    review = BASE / "review.json"
    if not review.exists():
        c.save(
            review,
            dict(
                created_utc=c.utc(),
                status="first_pass_snapshot_accepted_with_qualification",
                original_plan_sha256=c.sha(retrieval.PLAN),
                inputs=outputs,
                decision="Use the unchanged first-pass saved prompts under the original authority policy. The strict exact-replay gate remains failed and is not relabelled as passed. This additive review completes frozen-prompt preparation, conditional on the original independent prompt/token check and execution rehearsal.",
                cold_replay_exact=False,
                original_saved_prompts_changed=False,
                retrieval_algorithm_changed=False,
                observed_full_pass_message_changes={
                    pool: dict(
                        count=len(audit["changes"][pool]["message_union"]),
                        denominator=21627,
                        percent=100
                        * len(audit["changes"][pool]["message_union"])
                        / 21627,
                    )
                    for pool in ("small", "full")
                },
                interpretation="Snapshot reproducibility for subsequent decoder experiments; no claim of universal online retrieval determinism or unchanged F1. No new scientific effectiveness score.",
                remaining_gates=[
                    "independent broad render/token audit",
                    "broad failure/restart tests",
                    "execution plan and synthetic audit/restart rehearsal",
                    "future outcome sensitivity analysis",
                ],
                actual_Qwen_calls=0,
            ),
        )
    outputs[c.rel(review)] = c.sha(review)
    passes = {
        k: v for k, v in audit["passes"].items() if not k.endswith("/physical_copy")
    }
    c.verify(outputs)
    c.verify(p["original_index_hashes"])
    c.save(
        done,
        dict(
            status=STATUS,
            plan_sha256=c.sha(retrieval.PLAN),
            review=c.rel(review),
            review_sha256=c.sha(review),
            outputs=outputs,
            passes=passes,
            cold_replay_exact=False,
            first_pass_prompts_preserved=True,
            actual_Qwen_calls=0,
        ),
    )
    print(
        "Qualified first-pass snapshot prepared. Independent prompt checks and offline rehearsal remain.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["summarize", "finalize"])
    args = parser.parse_args()
    with c.offline():
        from analysis.common.api import item_lock

        with item_lock(retrieval.BASE / "launcher.lock"):
            if args.mode == "summarize":
                summarize()
            else:
                finalize()

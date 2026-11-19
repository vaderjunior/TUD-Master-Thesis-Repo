from collections import Counter, defaultdict
import itertools
import json
import math
import os
from pathlib import Path
import numpy as np
from analysis.feedback.reserved import review as io
from analysis.feedback.reserved import rates as rates
from analysis.feedback.reserved import support as support
from analysis.feedback.reserved import bounds as bounds


def prompt_diagnostics(prompts, choices, records, pools):

    choices = {(x["key"], x["condition"]): x for x in choices}
    pools = {p["condition"]: p for p in pools}
    by_example = {v["id"]: (k, v) for k, v in records.items()}
    references = {}
    meta = {}
    counts = defaultdict(Counter)
    tokens = defaultdict(list)
    for p in prompts:
        key, condition = p["key"], p["condition"]
        mode = condition.split("__")[0]
        examples = p["context"]["example_groups"]
        io.require(
            len(examples) == 1 and examples[0][0] == "Labelled examples",
            "Example grouping differs",
        )
        examples = examples[0][1]
        if condition == "reference5":
            references[key] = p
        io.require(
            key in references, "Reference must precede intervention in frozen catalog"
        )
        reference = references[key]
        original = reference["context"]["example_groups"][0][1]
        ids = [e["id"] for e in examples]
        delivered = [e for e in examples if e["id"] in by_example]
        selected = None
        source_gate = None
        if "__" in condition:
            pool = condition.split("__", 1)[1]
            choice = choices[key, pool]
            io.require(
                choice["source_id"] == p["source_id"]
                and choice["cosine"] == p["selected_correction_cosine"],
                "Selection attribution differs",
            )
            source_gate = choice["source_gate"]
            selected = records[p["source_id"]]
            io.require(
                selected["meta"]["gate"] == source_gate
                and all(
                    by_example[e["id"]][0] in pools[pool]["ids"] for e in delivered
                ),
                "Delivered source outside pool",
            )
            if mode in ("reserved5", "masked5", "extended6"):
                io.require(
                    [e["id"] for e in delivered] == [selected["id"]]
                    and examples[-1]["id"] == selected["id"],
                    "Reserved correction not delivered last",
                )
                io.require(
                    examples[-1]["meta"]["gate"]
                    == (None if mode == "masked5" else source_gate),
                    "Delivered gate differs",
                )
                io.require(
                    all(
                        examples[-1]["meta"].get(k) is None
                        for k in ("hate_types", "target_groups", "severity", "legal")
                    ),
                    "Invented correction fine labels",
                )
        removed = [e for e in original if e["id"] not in ids]
        fifth = original[4] if original[4]["id"] not in ids else None
        row = {
            k: p[k]
            for k in (
                "key",
                "condition",
                "population",
                "messages_sha256",
                "reference_input_tokens",
                "source_id",
                "selected_correction_cosine",
            )
        }
        row.update(
            source_gate=source_gate,
            delivered_correction_ids=[by_example[e["id"]][0] for e in delivered],
            delivered_annotations=[
                {"id": e["id"], "meta": e["meta"]} for e in delivered
            ],
            ordinary_ids=[e["id"] for e in examples if e["id"] not in by_example],
            removed_ordinary=[{"id": e["id"], "meta": e["meta"]} for e in removed],
            removed_fifth={"id": fifth["id"], "meta": fifth["meta"]} if fifth else {},
            example_count=len(examples),
            identical_to_reference=p["messages_sha256"] == reference["messages_sha256"],
            token_difference_vs_reference=p["reference_input_tokens"]
            - reference["reference_input_tokens"],
        )
        io.require((key, condition) not in meta, "Duplicate diagnostic row")
        meta[key, condition] = row
        ct = counts[condition]
        ct["query_pool_pairs"] += 1
        ct["delivered_pairs"] += bool(delivered)
        ct["delivered_records"] += len(delivered)
        ct["identical_messages"] += row["identical_to_reference"]
        ct["removed_fifth"] += fifth is not None
        tokens[condition].append(p["reference_input_tokens"])
        for e in delivered:
            ct["delivered_source:" + by_example[e["id"]][0]] += 1
            ct["delivered_annotation_gate:" + str(e["meta"].get("gate"))] += 1
    overlap = []
    for (n1, p1), (n2, p2) in itertools.combinations(sorted(pools.items()), 2):
        a, b = set(p1["ids"]), set(p2["ids"])
        overlap.append(
            {
                "a": n1,
                "b": n2,
                "intersection": len(a & b),
                "union": len(a | b),
                "jaccard": len(a & b) / len(a | b),
            }
        )
    summary = {
        "conditions": {
            k: {
                **dict(v),
                "local_tokens_min": min(tokens[k]),
                "local_tokens_max": max(tokens[k]),
                "local_tokens_mean": float(np.mean(tokens[k])),
            }
            for k, v in counts.items()
        },
        "pool_overlap": overlap,
        "unique_pool_sources": len({sid for p in pools.values() for sid in p["ids"]}),
        "unit": "query/pool combinations, dependent across pools; no outcome conditioning",
        "annotation_selection_cost": "Existing 1,024-source screening reused once; zero new A15 source labels or screening calls",
        "qualifications": [
            "Equal example count does not equal tokens or fine-label information.",
            "Extended6 changes both ordinary-neighbour retention and correction position.",
            "Masked labels are unknown, not non-hate. Selection and delivery are distinct.",
        ],
    }
    return meta, summary


def resources(predictions):
    groups = defaultdict(
        lambda: {
            "accounting": Counter(),
            "usage": Counter(),
            "ordinary_usage": Counter(),
            "repair_usage": Counter(),
            "durations": [],
            "unknown": 0,
            "backoff": 0.0,
            "predictions": 0,
            "nulls": 0,
        }
    )
    for (_, condition, repeat), row in predictions.items():
        for key in ("all", condition + "/r" + str(repeat)):
            g = groups[key]
            g["predictions"] += 1
            g["nulls"] += row["result"] is None
            g["accounting"].update(row["accounting"])
            g["usage"].update(row["observed_usage"])
            g["ordinary_usage"].update(row["ordinary_usage"])
            g["repair_usage"].update(row["repair_usage"])
            g["durations"].extend(row["known_request_durations_s"])
            g["unknown"] += row["administrative_unknown_requests"]
            g["backoff"] += row["retry_backoff_seconds_sum"]
    result = {}
    for name, g in groups.items():
        duration = g["durations"]
        result[name] = {
            "predictions": g["predictions"],
            "nulls": g["nulls"],
            "accounting": dict(g["accounting"]),
            "observed_usage": dict(g["usage"]),
            "ordinary_usage": dict(g["ordinary_usage"]),
            "repair_usage": dict(g["repair_usage"]),
            "known_request_duration_count": len(duration),
            "known_request_seconds_sum": math.fsum(duration),
            "known_request_seconds_mean": (
                float(np.mean(duration)) if duration else None
            ),
            "known_request_seconds_p50_p95": (
                np.quantile(duration, [0.5, 0.95]).tolist() if duration else None
            ),
            "retry_backoff_seconds_sum": g["backoff"],
            "administrative_unknown_requests": g["unknown"],
        }
        io.require(
            len(duration) + g["unknown"] == g["accounting"]["post_attempts"],
            "Timing/intent count differs",
        )
    return {
        "groups": result,
        "known_duration_unit": "saved POST attempt time; includes measured failures, excludes administrative sentinels",
        "timing_limits": "Summed request time is not elapsed wall time with eight workers. Shutdown delivery, token usage and duration remain unknown. Recovery timestamps are review times.",
        "billing": "Observed tokens only; no unverified currency conversion or invented usage for missing outcomes",
        "new_annotation_screening_calls": 0,
        "inherited_screened_sources": 1024,
    }


def save_array(path, values):
    path = io.logical(path)
    io.require(
        any(path.is_relative_to(p) for p in (io.PREP, io.OUTPUT)),
        "Array outside owned review",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with io.physical(path).open("xb") as f:
        np.save(f, values, allow_pickle=False)


def identifier(name):
    return name.replace("/", "__")


def analyze(
    roster, predictions, science, batch_by_key, meta, output, *, rehearsal=False
):
    io.require(
        rehearsal or (output == io.OUTPUT / "analysis"),
        "Production analysis destination differs",
    )
    if rehearsal:
        io.require(
            io.logical(output).is_relative_to(io.PREP),
            "Rehearsal outside preparation folder",
        )
    require_matrix = {
        (r["key"], c, rep)
        for r in roster
        for c in science["conditions"]
        for rep in range(3)
    }
    io.require(
        set(predictions) == require_matrix,
        "Missing/extra prediction in scientific assembly",
    )
    main_specs = rates.contrasts(science)
    secondary = support.supporting_specs(science)
    primary = {}
    secondary_results = {}
    source_results = {}
    omitted = {}
    verification = {}
    missing_bounds = {}
    masks = [
        ("ordinary_valid", False, False),
        ("all_three_votes", True, False),
        ("one_per_cluster", False, True),
    ]

    for cohort_name, excluded in [
        ("full", []),
        ("same_six_excluded", science["mandatory_excluded_keys"]),
    ]:
        for mask, strict, reps in masks:
            prefix = cohort_name + "/" + mask
            chosen_roster = support.selected_roster(roster, excluded, reps)
            runs = [
                support.metric_rows(chosen_roster, predictions, c, rep, strict=strict)
                for c in science["conditions"]
                for rep in range(3)
            ]
            io.write(output / prefix / "all_runs.json", runs)
            io.write(
                output / prefix / "pool_summaries.json", support.pool_summary(runs)
            )
            for spec in main_specs:
                name = prefix + "/" + spec["id"]
                c = rates.cohort(
                    roster,
                    predictions,
                    spec,
                    strict=strict,
                    representatives=reps,
                    excluded=excluded,
                )
                result, dist = rates.summarize(c, spec)
                primary[name] = result
                save_array(output / "draws" / (identifier(name) + ".npy"), dist)
                verification[name] = support.independent_rate_draws(
                    roster, predictions, spec, c, dist
                )
                omitted[name] = support.omission_points(
                    roster,
                    predictions,
                    spec,
                    batch_by_key,
                    strict=strict,
                    representatives=reps,
                    excluded=excluded,
                )
            print("Analysis completed " + prefix, flush=True)
        rows = support.selected_roster(roster, excluded)
        for spec in main_specs:
            name = cohort_name + "/" + spec["id"]
            c = rates.cohort(roster, predictions, spec, excluded=excluded)
            st = {**spec, "seed": support.seed_for("source_stratified/" + name)}
            summary, dist = support.source_bootstrap(c, roster, st)
            source_results[name + "/stratified"] = summary
            save_array(
                output / "draws" / (identifier(name + "/stratified") + ".npy"), dist
            )
            for source in sorted({r["source"] for r in roster}):
                sr = [r for r in rows if r["source"] == source]
                source_spec = {
                    **spec,
                    "confidence": 0.95,
                    "seed": support.seed_for("source_specific/" + name + "/" + source),
                }
                sub = rates.cohort(sr, predictions, source_spec)
                summary, dist = rates.summarize(sub, source_spec)
                summary["interpretation"] = (
                    "Supporting source-specific unadjusted 95% interval"
                )
                source_results[name + "/source/" + source] = summary
                save_array(
                    output
                    / "draws"
                    / (identifier(name + "/source/" + source) + ".npy"),
                    dist,
                )
        for spec in secondary:
            name = cohort_name + "/" + spec["id"]
            c = rates.cohort(roster, predictions, spec, excluded=excluded)
            summary, dist = rates.summarize(c, spec)
            secondary_results[name] = summary
            save_array(output / "draws" / (identifier(name) + ".npy"), dist)
            if spec["endpoint"] == "FPR":
                for fine in (False, True):
                    mname = name.rsplit("/", 1)[0] + ("/types" if fine else "/binary")
                    mspec = {**spec, "seed": support.seed_for(mname)}
                    summary, dist = support.paired_metrics(
                        rows, predictions, mspec, fine=fine
                    )
                    secondary_results[mname] = summary
                    save_array(output / "draws" / (identifier(mname) + ".npy"), dist)
                    for strict in (False, True):
                        missing_bounds[
                            mname
                            + ("/all_three_votes" if strict else "/ordinary_valid")
                        ] = bounds.metric_bounds(
                            rows,
                            predictions,
                            [spec["treatment_conditions"], spec["control_conditions"]],
                            fine=fine,
                            strict=strict,
                        )

        for spec in main_specs[2::2]:
            for fine in (False, True):
                name = (
                    cohort_name
                    + "/supporting_metrics/"
                    + spec["id"].rsplit("/", 1)[0]
                    + ("/types" if fine else "/binary")
                )
                mspec = {**spec, "seed": support.seed_for(name)}
                summary, dist = support.paired_metrics(
                    rows, predictions, mspec, fine=fine
                )
                secondary_results[name] = summary
                save_array(output / "draws" / (identifier(name) + ".npy"), dist)
                for strict in (False, True):
                    missing_bounds[
                        name + ("/all_three_votes" if strict else "/ordinary_valid")
                    ] = bounds.metric_bounds(
                        rows,
                        predictions,
                        [spec["treatment_conditions"], spec["control_conditions"]],
                        fine=fine,
                        strict=strict,
                    )
    io.write(output / "primary_mechanism.json", primary)
    io.write(output / "independent_bootstrap_check.json", verification)
    io.write(output / "supporting_effects.json", secondary_results)
    io.write(output / "supporting_missing_bounds.json", missing_bounds)
    io.write(output / "source_effects.json", source_results)
    io.write(output / "omission_points.json", omitted)
    pairs = [(c, "reference5") for c in science["conditions"] if c != "reference5"]
    for c in science["conditions"]:
        if c.startswith("reserved5__"):
            pairs.extend(
                (c, mode + "__" + c.split("__", 1)[1])
                for mode in ("natural5", "extended6", "masked5")
            )
    transition_summary = {}
    for cohort_name, excluded in [
        ("full", []),
        ("same_six_excluded", science["mandatory_excluded_keys"]),
    ]:
        rows = support.selected_roster(roster, excluded)
        for strict in (False, True):
            for treatment, control in pairs:
                name = (
                    cohort_name
                    + "/"
                    + ("all_three_votes" if strict else "ordinary_valid")
                    + "/"
                    + treatment
                    + "-vs-"
                    + control
                )
                value = support.transitions(
                    rows, predictions, treatment, control, meta, strict=strict
                )
                cases = value.pop("cases")
                value["changed_cases"] = len(cases)
                transition_summary[name] = value
                io.write(
                    output / "transitions" / (identifier(name) + ".jsonl"),
                    "".join(json.dumps(c, sort_keys=True) + "\n" for c in cases),
                )
    io.write(output / "transition_summary.json", transition_summary)
    io.write(output / "resources.json", resources(predictions))
    report = [
        "# Reserved correction study: generated review tables",
        "",
        (
            "SYNTHETIC REHEARSAL ONLY. No scientific finding."
            if rehearsal
            else "Computed from the sealed independently replayed assembly. Scientific interpretation and writer integration remain to be reviewed."
        ),
        "",
        "Rates are treatment minus control; negative means fewer errors. Intervals retain their declared primary/mechanism families. No null contrast establishes equivalence.",
        "",
        "| Roster / validity / comparison | Eligible / common texts | Difference (pp) | Adjusted interval (pp) | Missing-output bounds (pp) |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    fmt = lambda xs: (
        "not estimable"
        if xs is None
        else "[" + ", ".join(f"{100 * x:.4f}" for x in xs) + "]"
    )
    for name, result in primary.items():
        diff = (
            "not estimable"
            if "difference" not in result
            else f"{100 * result['difference']:.4f}"
        )
        report.append(
            f"| {name} | {result['original_n']} / {result['common_n']} | {diff} | {fmt(result.get('family_interval'))} | {fmt(result['missing_difference_bounds'])} |"
        )
    report.extend(
        [
            "",
            "Supporting results, all 222 real condition/repeat summaries per roster/mask, pool variation, source checks, omission estimates, every transition and exact saved bootstrap draws accompany this table.",
            "",
            "Interpretation must retain exposed DEV scope; fixed observed pools; binary-only corrections; unknown fine gold; token/information/position differences; dependent matched-repeat transition counts; and the separate eight unknown interrupted request outcomes.",
            "",
            "The pre-existing 1,024-source selection cost is inherited once. No new model call, retrieval, embedding, annotation, or training is performed by this report. Broader transfer and iterative protocols remain fixed regardless of these outcomes.",
            "",
            "Final review: inspect effect directions and missing bounds together, reconcile each headline with its cohort, retain all unfavorable/null outcomes, update the results journal and writer feed, and only then mark this experiment reviewed.",
        ]
    )
    io.write(output / "REPORT.md", "\n".join(report) + "\n")
    artifacts = {
        io.rel(p): io.sha(p) for p in io.physical(output).rglob("*") if p.is_file()
    }
    expected_transitions = 4 * len(pairs)
    io.require(
        sum("/transitions/" in p for p in artifacts) == expected_transitions,
        "Analysis manifest omits a transition file (including long Windows paths)",
    )
    io.write(
        output / "complete.json",
        {
            "status": (
                "synthetic_rehearsal_complete"
                if rehearsal
                else "scored_requires_scientific_interpretation"
            ),
            "artifacts": artifacts,
            "primary_mechanism_versions": len(primary),
            "supporting_versions": len(secondary_results),
            "independently_checked_rate_draws": sum(
                v["draws_checked"] for v in verification.values()
            ),
            "new_model_calls": 0,
            "analysis_source_hashes": io.reviewer_sources(),
        },
    )
    return io.read(output / "complete.json")


def main():
    with io.offline():
        io.completion_gate()
        assembly = io.read(io.OUTPUT / "assembly/complete.json")
        io.require(
            assembly["status"] == "complete_independently_replayed_unscored"
            and assembly["plan_sha256"] == io.PLAN_SHA,
            "Full independent review required before scoring",
        )
        io.verify(io.ROOT, assembly["artifacts"])
        io.verify(io.ROOT, assembly["analysis_source_hashes"])
        io.require(
            assembly["queue_complete_sha256"]
            == io.sha(io.EXECUTION / "live/complete.json")
            and assembly["scientific_plan_sha256"]
            == io.sha(io.STUDY / "scientific_plan_v1.json"),
            "Assembly provenance changed",
        )
        io.require(
            not (io.OUTPUT / "analysis").exists(),
            "Preserve existing analysis; inspect before publishing another version",
        )
        plan = io.read(io.EXECUTION / "plan.json")
        io.verify(io.ROOT, plan["source_hashes"])
        io.verify(io.ROOT, plan["input_hashes"])
        io.write(
            io.OUTPUT / "analysis_started.json",
            {
                "assembly_complete_sha256": io.sha(
                    io.OUTPUT / "assembly/complete.json"
                ),
                "analysis_source_hashes": io.reviewer_sources(),
                "source_inputs": plan["input_hashes"],
                "environment": io.environment(),
                "started_utc": io.utc(),
                "new_model_calls": 0,
            },
        )
        roster = list(io.rows(io.STUDY / "roster.jsonl"))
        science = io.read(io.STUDY / "scientific_plan_v1.json")
        predictions = {}
        for name in assembly["artifacts"]:
            if name.endswith(".jsonl"):
                for r in io.rows(io.ROOT / name):
                    key = r["key"], r["condition"], r["repeat"]
                    io.require(key not in predictions, "Duplicate assembly row")
                    predictions[key] = r
        selections = (
            io.ROOT
            / "experiments/astra/sq3_additions_results_20261006_v1/benign_selection"
        )
        meta, diagnostics = prompt_diagnostics(
            io.rows(io.STUDY / "prompts.jsonl"),
            io.rows(io.STUDY / "choices.jsonl"),
            io.read(selections / "gate_only_records.json"),
            io.read(selections / "actual_pools.json")["pools"],
        )
        io.write(io.OUTPUT / "prompt_diagnostics.json", diagnostics)
        io.write(
            io.OUTPUT / "prompt_metadata.jsonl",
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in meta.values()),
        )
        analyze(
            roster,
            predictions,
            science,
            io.read(io.OUTPUT / "assembly/batches.json"),
            meta,
            io.OUTPUT / "analysis",
        )
        io.verify(io.ROOT, plan["source_hashes"])
        io.verify(io.ROOT, plan["input_hashes"])
        io.verify(
            io.ROOT,
            io.read(io.OUTPUT / "analysis_started.json")["analysis_source_hashes"],
        )
        files = [
            io.OUTPUT / n
            for n in (
                "analysis_started.json",
                "assembly/complete.json",
                "analysis/complete.json",
                "prompt_diagnostics.json",
                "prompt_metadata.jsonl",
            )
        ]
        io.write(
            io.OUTPUT / "results_complete.json",
            {
                "status": "computed_requires_scientific_interpretation",
                "artifacts": {io.rel(p): io.sha(p) for p in files},
                "completed_utc": io.utc(),
                "original_plan_sha256": io.PLAN_SHA,
                "new_model_calls": 0,
                "remaining": "Scientific interpretation, final result report/journal and writer integration; no automatic next experiment",
            },
        )


if __name__ == "__main__":
    main()

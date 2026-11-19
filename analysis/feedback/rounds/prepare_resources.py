from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import io
import math
import unittest
import numpy as np
from analysis.feedback.rounds import resources_inputs as a
from analysis.feedback.rounds import resources_metrics as k
from analysis.feedback.reserved import resources_metrics as core


def chronology(reader, bindings):
    stages = []
    gets = []
    statuses = Counter()
    retries = []

    def bind(p, digest=None):
        if digest is not None:
            a.need(a.sha(p) == digest, "Chronology binding differs")
        bindings.update(a.bindings([p]))
        return a.read(p)

    for sr in reader["stage_receipts"]:
        receipt = bind(a.ROOT / sr["receipt"], sr["receipt_sha256"])
        a.check(receipt["inputs"])
        reportname = next(
            n
            for n in receipt["inputs"]
            if "/audit_" in n and n.endswith("/report.json")
        )
        report = bind(a.ROOT / reportname, receipt["inputs"][reportname])
        stagepath = (a.ROOT / reportname).parent.parent
        comp = bind(stagepath / "complete.json", report["stage_complete_sha256"])
        live = stagepath.parent.parent
        history = {n: bind(live / n, h) for n, h in report["attempt_hashes"].items()}
        folders = sorted(
            {
                str(Path(n).parent).replace("\\", "/")
                for n in history
                if Path(n).parent.name.startswith("attempt_")
            }
        )
        attempts = []
        for folder in folders:
            start = history[folder + "/start.json"]
            inv = history[folder + "/inventory.json"]
            intent = history.get(folder + "/model_check_intent.json")
            response = history.get(folder + "/model_check_response.json")
            ends = [
                name
                for name in ["summary.json", "interrupted.json"]
                if folder + "/" + name in history
            ]
            a.need(len(ends) <= 1, "Multiple attempt endpoints")
            end = history[folder + "/" + ends[0]] if ends else None
            row = dict(
                path=a.rel(live / folder),
                start=start["started_utc"],
                end=end["finished_utc"] if end else None,
                end_kind=ends[0] if ends else "unknown",
                completed_on_entry=inv["completed"],
                pending_on_entry=inv["pending"],
                effective_workers=128,
            )
            if intent:
                a.need(
                    response is not None
                    and response["status"] == 200
                    and intent["action"] == "GET PEASEC /models",
                    "Availability check differs",
                )
                row.update(
                    model_check_start=intent["started_utc"],
                    model_check_seconds=response["elapsed_s"],
                )
                gets.append(
                    dict(
                        phase=sr["phase"],
                        stage=sr["stage"],
                        attempt=row["path"],
                        started_utc=intent["started_utc"],
                        seconds=response["elapsed_s"],
                        status=200,
                    )
                )
            if ends and ends[0] == "summary.json":
                row.update(
                    saved_items=end["saved_items"],
                    submitted=end["submitted"],
                    peak_submitted=end["peak_submitted"],
                    stopped=end["stopped"],
                    errors=len(end.get("errors", [])),
                )
                a.need(end["peak_submitted"] <= 128, "Unexpected concurrency")
            attempts.append(row)
        attempts.sort(key=lambda x: core.timestamp(x["start"]))
        generation = comp["summary"]["finished_utc"]
        a.need(
            len(attempts) == report["attempt_count"]
            and attempts[-1]["end"] == generation
            and not comp["summary"]["stopped"],
            "Final generation/attempt mismatch",
        )
        partition = core.attempt_partition(attempts, generation)
        first = attempts[0]["start"]
        completed = comp["completed_utc"]
        audited = report["created_utc"]
        a.need(
            math.isclose(
                core.seconds(first, completed),
                report["wall_seconds_including_restarts"],
                abs_tol=1e-6,
            ),
            "Original stage wall differs",
        )
        stages.append(
            dict(
                phase=sr["phase"],
                stage=sr["stage"],
                role=sr["role"],
                first_start_utc=first,
                generation_summary_utc=generation,
                stage_completion_utc=completed,
                audit_report_utc=audited,
                predictions=report["counts"]["predictions"],
                attempts=attempts,
                partition=partition,
                generation_envelope_seconds=core.seconds(first, generation),
                completion_bookkeeping_seconds=core.seconds(generation, completed),
                audit_window_seconds=core.seconds(completed, audited),
                start_to_audit_seconds=core.seconds(first, audited),
                homogeneous_uninterrupted=len(attempts) == 1,
            )
        )
        statuses.update(report["http_statuses"])
        retries.extend(
            dict(phase=sr["phase"], stage=sr["stage"], **r)
            for r in report["retry_events"]
        )
    stages.sort(key=lambda r: core.timestamp(r["first_start_utc"]))
    parts = Counter()
    gaps = []
    for s in stages:
        parts.update(s["partition"]["seconds_by_kind"])
        parts["completion_bookkeeping"] += s["completion_bookkeeping_seconds"]
        parts["completion_to_audit"] += s["audit_window_seconds"]
    for left, right in zip(stages, stages[1:]):
        sec = core.seconds(left["audit_report_utc"], right["first_start_utc"])
        gaps.append(
            dict(
                after=left["phase"] + "/" + left["stage"],
                before=right["phase"] + "/" + right["stage"],
                seconds=sec,
            )
        )
        parts["between_batches_or_phases"] += sec
    wall = core.seconds(stages[0]["first_start_utc"], stages[-1]["audit_report_utc"])
    a.need(
        math.isclose(sum(parts.values()), wall, abs_tol=1e-4),
        "Chronology partition differs",
    )
    a.need(
        sum(statuses.values()) == reader["accounting"]["post_attempts"]
        and len(retries) == reader["accounting"]["retry_attempts"],
        "Audit transport inventory differs",
    )
    throughput = {}
    for role in ["source", "evaluation", "all"]:
        ss = [
            s
            for s in stages
            if s["homogeneous_uninterrupted"] and (role == "all" or s["role"] == role)
        ]
        n = sum(s["predictions"] for s in ss)
        sec = sum(s["generation_envelope_seconds"] for s in ss)
        aud = sum(s["start_to_audit_seconds"] for s in ss)
        throughput[role] = dict(
            stages=[s["phase"] + "/" + s["stage"] for s in ss],
            outputs=n,
            generation_seconds=sec,
            through_audit_seconds=aud,
            outputs_per_hour_generation=3600 * n / sec,
            outputs_per_hour_through_audit=3600 * n / aud,
        )
    return dict(
        stages=stages,
        availability_checks=gets,
        availability_seconds=math.fsum(r["seconds"] for r in gets),
        http_statuses=statuses,
        retry_events=retries,
        between_stages=gaps,
        partition_seconds=parts,
        start_utc=stages[0]["first_start_utc"],
        last_generation_utc=stages[-1]["generation_summary_utc"],
        last_audit_utc=stages[-1]["audit_report_utc"],
        start_to_last_generation_seconds=core.seconds(
            stages[0]["first_start_utc"], stages[-1]["generation_summary_utc"]
        ),
        start_to_last_audit_seconds=wall,
        throughput=throughput,
        finalization_time="Final study and continuation receipts have no internal completion timestamp. No finalizer-duration or filesystem proxy is invented.",
        qualification="Recorded client windows include preparation, waiting and interruptions. Summed POST time overlaps across 128 workers and is neither elapsed wall time nor GPU time. Uninterrupted batches have different texts and service states; throughput is observational.",
    )


def main():
    a.need(not a.OUT.exists(), "Preserve prior Part 8 work")
    checkpoint = a.read(a.REVIEW / "part7/final_verification.json")
    a.need(
        checkpoint["status"] == "iterative_part7_complete_with_final_readback"
        and checkpoint["complete_sha256"] == a.sha(a.REVIEW / "part7/complete.json"),
        "Verified Part 7 required",
    )
    reader = a.read(a.READER / "complete.json")
    a.check(reader["artifacts"])
    inputs = [
        a.READER / "complete.json",
        a.READER / "independent_verification.json",
        a.REVIEW / "review_plan.json",
        a.STUDY / "plan.json",
        a.STUDY / "complete.json",
        a.STUDY / "STUDY_PROTOCOL.md",
        a.REVIEW / "part7/complete.json",
        a.REVIEW / "part7/final_verification.json",
        a.REVIEW / "part7/metadata.jsonl",
        a.REVIEW / "part7/execution.json",
        a.REVIEW / "part5/metadata.jsonl",
        a.REVIEW / "part3/prompt_bindings.jsonl",
        a.REVIEW / "part3/deliveries.jsonl",
        a.REVIEW / "part3/selections.json",
        a.REVIEW / "part3/details.json",
        a.REVIEW / "part3/analysis.json",
        a.ROOT
        / "experiments/astra/sq3_transfer_part10_20261023_v1/partial_programme_ledger.json",
        a.LINEAR / "plan.json",
        a.LINEAR / "static_complete.json",
        a.LINEAR / "data.json",
        a.LINEAR / "STUDY_PROTOCOL.md",
    ]
    bindings = a.bindings(inputs)
    sourcepaths = [
        Path(__file__),
        Path(a.__file__),
        Path(k.__file__),
        Path(core.__file__),
        a.ROOT / "tests/feedback/rounds/test_resources.py",
    ]
    a.write(
        a.OUT / "analysis_contract.json",
        dict(
            status="part8_resource_and_synthesis_rules_recorded",
            created_utc=a.pure.utc(),
            input_bindings=bindings,
            source_hashes=a.bindings(sourcepaths),
            timing="After Parts 1-7 and before added resource comparisons. Original scientific outcomes and inference remain unchanged; resources are supporting descriptions.",
            accounting="Read all 259224 audit-bound compact outputs. Separate ordinary logical calls, repairs, extra transport attempts, successful replies and model-list GETs. Retain every null and partial output cost. Inspect only repaired/retried original result files for depth/failure attribution; no full raw response replay.",
            groups="Whole study, screening/evaluation, phases/batches, rounds, policies, populations, source/gold, HateCheck functions, decoder repeats, validity/repair/retry classes and common/strict included/excluded cohorts. Groups overlap; only stated disjoint partitions may be summed.",
            inference=dict(
                metrics=k.METRICS,
                columns=k.COLUMNS,
                populations=a.POPS,
                rounds=[1, 2, 3, 4],
                masks=["all_roster", "round_common", "all_round_common"],
                draws=10000,
                level=0.95,
                unit="Text-weighted whole inherited inference component",
                seed_rule="SHA256 first-eight little-endian of A17-part8-resources-v1|record",
                reference="Three actual reference outputs per round/text. Do not triple reference call costs for the three source streams.",
            ),
            chronology="Use original attempt starts/endpoints, completion and audit timestamps. Final study receipts lack internal timestamps; do not invent finalizer time. Partition explicit gaps, bookkeeping and audit windows. Observed request duration sums are overlapping client time, not GPU time.",
            marginal_value="Show incurred per-round and cumulative screening/evaluation expense alongside already verified fixed-cohort error and quality trajectories and context delivery. Initial screening of all 1024 occurs upfront. No outcome-selected stopping round, utility winner, deployment prevalence or causal dose-response claim.",
            classifier="Prepare exact cumulative ID-set aliases for the two frozen recipes, reuse 38 existing static fit receipts, and map the 2077 evaluation texts to the existing classifier roster. No fit, launcher, probability scoring, retuning or new primary family.",
            integration="Close only the successive-round experiment after all checks. Add one nonoverlapping scientific ledger row, retain earlier rows, update journal/status/register with writing held for cumulative comparison and combined SQ3 synthesis.",
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
            new_annotations=0,
        ),
    )
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "tests.feedback.rounds.test_resources"
    )
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    a.write(
        a.OUT / "tests.json",
        dict(
            passed=result.wasSuccessful(), tests_run=result.testsRun, log=log.getvalue()
        ),
    )
    a.need(result.wasSuccessful(), "Resource fixtures failed")
    meta = list(a.rows(a.REVIEW / "part7/metadata.jsonl"))
    index = {m["key"]: i for i, m in enumerate(meta)}
    function_meta = {m["key"]: m for m in a.rows(a.REVIEW / "part5/metadata.jsonl")}
    a.need(set(function_meta) == set(index), "Source/function metadata differs")
    prompts = {
        (r["phase"], r["key"], r["condition"]): r
        for r in a.rows(a.REVIEW / "part3/prompt_bindings.jsonl")
    }
    a.need(len(prompts) == 86408, "Prompt inventory differs")
    groups = defaultdict(core.Group)
    x = np.zeros((2077, 4, 10, 3, 8))
    seen = np.zeros((2077, 4, 10, 3), dtype="uint8")
    local = np.zeros((2077, 4, 10), dtype="int32")
    source_rows = []
    targeted = []
    usage_offset = Counter()
    promptkeys = set()
    localgroups = defaultdict(Counter)
    for st in reader["stage_receipts"]:
        receipt = a.read(a.ROOT / st["receipt"])
        a.need(
            a.sha(a.ROOT / st["receipt"]) == st["receipt_sha256"],
            "Compact receipt changed",
        )
        path = a.ROOT / st["predictions"]
        a.need(a.sha(path) == receipt["predictions_sha256"], "Compact outputs changed")
        bindings.update(a.bindings([a.ROOT / st["receipt"], path]))
        for r in a.rows(path):
            pk = (r["phase"], r["key"], r["condition"])
            p = prompts[pk]
            a.need(
                p["messages_sha256"] == r["messages_sha256"], "Actual prompt differs"
            )
            names = [
                "all",
                "role/" + r["role"],
                "phase/" + r["phase"],
                "stage/" + r["phase"] + "/" + r["stage"],
                "valid_votes/" + str(r["n_valid"]),
                "repair_used/" + str(bool(r["repairs"])),
                "retry_used/" + str(bool(r["accounting"]["retry_attempts"])),
                "population/" + r["population"],
                "repeat/" + str(r["repeat"]),
            ]
            r["resource_group_names"] = names
            if r["role"] == "evaluation":
                i = index[r["key"]]
                rr = r["round"] - 1
                c = a.CONDITIONS.index(r["condition"])
                d = r["repeat"]
                m = meta[i]
                a.need(not seen[i, rr, c, d], "Duplicate resource slot")
                seen[i, rr, c, d] = 1
                x[i, rr, c, d] = k.resource_row(r)
                local[i, rr, c] = p["local_token_estimate"]
                names += [
                    "policy/" + r["policy"],
                    "round/" + str(r["round"]),
                    "round_population_policy/"
                    + str(r["round"])
                    + "/"
                    + r["population"]
                    + "/"
                    + r["policy"],
                    "source_gold/" + m["source"] + "/" + str(int(m["gold_hate"])),
                    "population_policy/" + r["population"] + "/" + r["policy"],
                ]
                if function_meta[r["key"]].get("functionality") is not None:
                    names += ["function/" + function_meta[r["key"]]["functionality"]]
                for mask in ["round_common", "round_strict"]:
                    names += ["mask/" + mask + "/" + str(m["masks"][mask][rr])]
                names += [
                    "mask/all_round_common/" + str(m["masks"]["all_round_common"])
                ]
                usage_offset[r["population"] + "/" + r["policy"]] += (
                    r["ordinary_usage"]["prompt_tokens"] - 3 * p["local_token_estimate"]
                )
            else:
                source_rows.append(
                    {
                        key: r[key]
                        for key in [
                            "phase",
                            "round",
                            "stream",
                            "key",
                            "population",
                            "condition",
                            "repeat",
                            "accounting",
                            "observed_usage",
                            "ordinary_usage",
                            "repair_usage",
                            "n_valid",
                            "repairs",
                            "parse_failures",
                            "known_request_seconds_sum",
                            "retry_backoff_seconds_sum",
                        ]
                    }
                )
            if pk not in promptkeys:
                promptkeys.add(pk)
                g = localgroups[r["role"] + "/" + r["phase"] + "/" + r["condition"]]
                g.update(prompts=1, local_tokens=p["local_token_estimate"])
            for name in names:
                groups[name].add(r)
                groups[name].counts.update(
                    repair_used=int(bool(r["repairs"])),
                    retry_used=int(bool(r["accounting"]["retry_attempts"])),
                )
            if r["repairs"] or r["accounting"]["retry_attempts"]:
                targeted.append(
                    {
                        key: r[key]
                        for key in [
                            "phase",
                            "key",
                            "role",
                            "population",
                            "round",
                            "policy",
                            "stream",
                            "condition",
                            "repeat",
                            "item_receipt",
                            "item_receipt_sha256",
                            "result_sha256",
                            "n_valid",
                            "repairs",
                            "parse_failures",
                            "accounting",
                        ]
                    }
                )
        print("Resource census " + st["phase"] + "/" + st["stage"], flush=True)
    a.need(seen.all() and len(promptkeys) == 86408, "Resource matrix incomplete")
    values = {n: g.finish() for n, g in groups.items()}
    a.need(
        values["all"]["accounting"] == reader["accounting"]
        and values["all"]["observed_usage"] == reader["observed_usage"],
        "Resource totals differ",
    )
    with (a.OUT / "arrays.npz").open("xb") as f:
        np.savez_compressed(f, resources=x, local_prompt_tokens=local)
    with a.gzopen(a.OUT / "input_costs.jsonl.gz") as f:
        for i, m in enumerate(meta):
            a.emit(f, dict(**m, resources=x[i], local_prompt_tokens=local[i]))
    with a.gzopen(a.OUT / "screening_costs.jsonl.gz") as f:
        for r in source_rows:
            a.emit(f, r)
    a.write(a.OUT / "targeted_result_inventory.json", targeted)
    a.write(a.OUT / "resource_groups.json", values)
    a.write(
        a.OUT / "array_schema.json",
        dict(
            shape=list(x.shape),
            axes=[
                "2077 original inputs",
                "four rounds",
                "ten actual conditions",
                "three decoder repeats",
                "metric",
            ],
            metrics=k.METRICS,
            conditions=a.CONDITIONS,
            reference_replicated=False,
            null_costs_retained=True,
        ),
    )
    a.write(
        a.OUT / "local_prompt_costs.json",
        dict(
            groups=localgroups,
            ordinary_API_prompt_minus_three_local_tokens=usage_offset,
            unique_actual_prompts=len(promptkeys),
            qualification="Reuse independently checked complete local prompt estimates from Part 3. Ordinary provider prompt usage and local tokenizer estimates are distinct measurements; no retokenization or new encoding.",
        ),
    )
    history = chronology(reader, bindings)
    a.write(a.OUT / "chronology.json", history)
    files = [p for p in a.OUT.iterdir() if p.is_file()]
    a.write(
        a.OUT / "assembled.json",
        dict(
            status="part8_resource_census_complete",
            created_utc=a.pure.utc(),
            input_bindings=bindings,
            source_hashes=a.bindings(sourcepaths),
            artifacts=a.bindings(files),
            outputs=259224,
            evaluation_outputs=249240,
            screening_outputs=9984,
            resource_groups=len(values),
            targeted_results=len(targeted),
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        "Part 8 resource census and chronology complete; no new calls or fits.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()

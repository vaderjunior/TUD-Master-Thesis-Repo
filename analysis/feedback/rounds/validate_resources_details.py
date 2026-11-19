from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import hashlib
import json
import math
from analysis.feedback.rounds import resources_inputs as a


def seconds(start, end):
    x = datetime.fromisoformat(start)
    y = datetime.fromisoformat(end)
    a.need(
        x.tzinfo is not None and y.tzinfo is not None and y >= x,
        "Invalid chronological endpoint",
    )
    return (y - x).total_seconds()


def close(x, y, message):
    a.need(math.isclose(x, y, rel_tol=2e-13, abs_tol=1e-6), message)


def main():
    repair = a.read(a.OUT / "repair_depth.json")
    failure = a.read(a.OUT / "failure_costs.json")
    groups = a.read(a.OUT / "resource_groups.json")
    timeline = a.read(a.OUT / "chronology.json")
    handoff = a.read(a.OUT / "classifier_handoff.json")
    a.check(handoff["input_bindings"])
    a.check(a.read(a.OUT / "repairs_complete.json")["artifacts"])
    by_result = defaultdict(dict)
    summary = defaultdict(Counter)
    role = defaultdict(Counter)
    ledger_rows = 0
    for r in a.gzrows(a.OUT / "repair_vote_costs.jsonl.gz"):
        a.need(
            r["vote"] not in by_result[r["result_path"]],
            "Duplicate repaired vote in ledger",
        )
        by_result[r["result_path"]][r["vote"]] = r
        ledger_rows += 1
        u = r["usage_by_turn"]
        cat = r["category"]
        for turn, cost in enumerate(u):
            for group in ["all_turn_" + str(turn), cat + "/turn_" + str(turn)]:
                summary[group]["calls"] += 1
                for field, value in cost.items():
                    summary[group][field] += value
        summary["category/" + cat].update(
            votes=1,
            repair_calls=len(u) - 1,
            ordinary_tokens=u[0]["total_tokens"],
            repair_tokens=sum(v["total_tokens"] for v in u[1:]),
            total_tokens=sum(v["total_tokens"] for v in u),
        )
        role[r["role"] + "/" + cat].update(
            votes=1,
            repair_calls=len(u) - 1,
            repair_tokens=sum(v["total_tokens"] for v in u[1:]),
        )
    a.need(
        {k: dict(v) for k, v in summary.items()} == repair["groups"]
        and {k: dict(v) for k, v in role.items()} == repair["role_groups"],
        "Independent repair regroup differs",
    )
    inventory = a.read(a.OUT / "targeted_result_inventory.json")
    actual_failures = []
    checked = 0
    vote_checks = 0

    def inspect(row):
        path = a.ROOT / Path(row["item_receipt"]).with_name("result.json")
        name = a.rel(path)
        a.need(
            a.sha(path)
            == row["result_sha256"]
            == repair["original_result_bindings"][name],
            "Original repaired/retried result changed",
        )
        q = a.read(path)
        actual = {}
        fails = []
        for event in q["events"]:
            if event["action"] == "ok":
                actual.setdefault(event["vote"], []).append(event)
            else:
                fails.append(
                    dict(
                        phase=row["phase"],
                        key=row["key"],
                        condition=row["condition"],
                        repeat=row["repeat"],
                        vote=event["vote"],
                        turn=event["turn"],
                        response_class=event["response_class"],
                        request_seconds=event["elapsed_s"],
                        backoff_seconds=event["backoff_elapsed_s"],
                        result_path=name,
                        result_sha256=row["result_sha256"],
                    )
                )
        checks = 0
        for vote, events in actual.items():
            if len(events) == 1:
                a.need(vote not in by_result[name], "Spurious repaired vote")
                continue
            saved = by_result[name][vote]
            depth = len(events) - 1
            valid = q["run_predictions"][vote] is not None
            cat = f"recovered_after_{depth}" if valid else "terminal_invalid"
            a.need(
                saved["repair_depth"] == depth
                and saved["category"] == cat
                and saved["final_schema_valid"] == valid,
                "Independent repair outcome differs",
            )
            a.need(
                [e["turn"] for e in events] == list(range(len(events))),
                "Repair order differs",
            )
            for turn, event in enumerate(events):
                u = {
                    k: event["model_response_metadata"]["usage"][k]
                    for k in ["prompt_tokens", "completion_tokens", "total_tokens"]
                }
                a.need(saved["usage_by_turn"][turn] == u, "Repair-turn usage differs")
            checks += 1
        return fails, checks

    with ThreadPoolExecutor(max_workers=4) as executor:
        for failed, n in executor.map(inspect, inventory):
            actual_failures.extend(failed)
            vote_checks += n
            checked += 1
    a.need(
        actual_failures == failure["events"]
        and len(actual_failures) == 669
        and checked == 19676
        and vote_checks == ledger_rows == 29992,
        "Failure/repair result coverage differs",
    )
    a.need(
        sum(summary[f"all_turn_{t}"]["total_tokens"] for t in [1, 2])
        == groups["all"]["repair_usage"]["total_tokens"],
        "Repair tokens not a subset of incurred total",
    )
    a.need(
        repair["recovered_after_first"]
        + repair["recovered_after_second"]
        + repair["terminal_invalid_votes"]
        == repair["repaired_votes"],
        "Repair recovery count differs",
    )
    print(
        "Every targeted original result and all 29,992 repaired vote paths PASS.",
        flush=True,
    )
    stage_times = []
    availability = 0
    availability_seconds = 0.0
    attempts = 0
    pieces = 0
    bindings = {}
    for saved in timeline["stages"]:
        root = (
            a.STUDY
            / "phases"
            / saved["phase"]
            / "execution/live"
            / saved["role"]
            / "stages"
            / saved["stage"]
        )
        review = a.read(root / "review.json")
        report = a.read(root / review["report"])
        complete = a.read(root / "complete.json")
        live = root.parent.parent
        a.need(
            a.sha(root / "complete.json") == review["complete_sha256"]
            and a.sha(root / review["report"]) == review["artifacts"][review["report"]],
            "Chronology original audit binding differs",
        )
        bindings.update(
            a.bindings(
                [root / "review.json", root / "complete.json", root / review["report"]]
            )
        )
        starts = []
        for item in saved["attempts"]:
            path = a.ROOT / item["path"]
            start = a.read(path / "start.json")
            a.need(
                start["started_utc"] == item["start"], "Original attempt start differs"
            )
            starts.append(start["started_utc"])
            attempts += 1
            if item["end_kind"] != "unknown":
                a.need(
                    a.read(path / item["end_kind"])["finished_utc"] == item["end"],
                    "Original attempt end differs",
                )
            intent = a.read(path / "model_check_intent.json")
            response = a.read(path / "model_check_response.json")
            a.need(
                response["status"] == 200
                and response["elapsed_s"] == item["model_check_seconds"]
                and intent["started_utc"] == item["model_check_start"],
                "GET accounting differs",
            )
            availability += 1
            availability_seconds += response["elapsed_s"]
            for filename in [
                "start.json",
                "inventory.json",
                "model_check_intent.json",
                "model_check_response.json",
            ] + ([] if item["end_kind"] == "unknown" else [item["end_kind"]]):
                rel = (path / filename).relative_to(live).as_posix()
                a.need(
                    a.sha(path / filename) == report["attempt_hashes"][rel],
                    "Attempt evidence changed",
                )
        first = min(starts, key=datetime.fromisoformat)
        a.need(
            first == saved["first_start_utc"]
            and complete["summary"]["finished_utc"] == saved["generation_summary_utc"]
            and complete["completed_utc"] == saved["stage_completion_utc"]
            and report["created_utc"] == saved["audit_report_utc"],
            "Stage clock differs",
        )
        for field, end in [
            ("generation_envelope_seconds", saved["generation_summary_utc"]),
            ("start_to_audit_seconds", saved["audit_report_utc"]),
        ]:
            close(saved[field], seconds(first, end), "Stage duration differs")
        close(
            saved["audit_window_seconds"],
            seconds(saved["stage_completion_utc"], saved["audit_report_utc"]),
            "Audit window differs",
        )
        previous = first
        totals = Counter()
        for segment in saved["partition"]["pieces"]:
            a.need(
                datetime.fromisoformat(segment["start"])
                == datetime.fromisoformat(previous),
                "Chronology piece gap/overlap",
            )
            duration = seconds(segment["start"], segment["end"])
            close(duration, segment["seconds"], "Timing piece length differs")
            totals[segment["kind"]] += duration
            previous = segment["end"]
            pieces += 1
        a.need(
            datetime.fromisoformat(previous)
            == datetime.fromisoformat(saved["generation_summary_utc"]),
            "Timing partition end differs",
        )
        for kind, duration in totals.items():
            close(
                duration,
                saved["partition"]["seconds_by_kind"][kind],
                "Timing partition category differs",
            )
        stage_times.append(saved)
    a.need(
        len(stage_times) == 38
        and availability == len(timeline["availability_checks"]) == 40
        and attempts == 40,
        "Chronology coverage differs",
    )
    close(
        availability_seconds,
        timeline["availability_seconds"],
        "Total GET clock differs",
    )
    overall = seconds(
        stage_times[0]["first_start_utc"], stage_times[-1]["audit_report_utc"]
    )
    close(
        overall,
        timeline["start_to_last_audit_seconds"],
        "Timezone-aware whole duration differs",
    )
    close(
        sum(timeline["partition_seconds"].values()),
        overall,
        "Whole clock partition differs",
    )
    for i, gap in enumerate(timeline["between_stages"]):
        close(
            gap["seconds"],
            seconds(
                stage_times[i]["audit_report_utc"],
                stage_times[i + 1]["first_start_utc"],
            ),
            "Inter-stage gap differs",
        )
    print(
        "All 38 stages, 40 attempts/GETs and the timezone-aware wall partition PASS.",
        flush=True,
    )
    selections = a.read(a.REVIEW / "part3/selections.json")
    plan = a.read(a.LINEAR / "plan.json")
    data = a.read(a.LINEAR / "data.json")
    labels = dict(zip(data["candidate_ids"], data["candidate_labels"]))
    aliases = {}
    for variant in plan["variants"]:
        aliases[variant, "reference"] = ()
        for s in selections:
            for policy, ids in s["cumulative"].items():
                aliases[variant, f'r{s["round"]}_{policy}_s{s["stream"]}'] = tuple(
                    sorted(ids)
                )
    covered = {}
    fresh = 0
    old = 0
    for job in handoff["jobs"]:
        ident = job["identity"]
        canonical = (
            json.dumps(
                ident, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False
            )
            + "\n"
        ).encode()
        folder = (
            a.LINEAR
            / "jobs"
            / job["variant"]
            / hashlib.sha256(canonical).hexdigest()[:16]
        )
        a.need(
            a.rel(folder) == job["job_folder"],
            "Classifier deterministic identity differs",
        )
        ids = tuple(job["source_ids"])
        fit = data["partitions"][job["variant"]]
        pos = sum(r["target"] for r in fit) + sum(labels[i] for i in ids)
        n = len(fit) + len(ids)
        a.need(
            job["positive"] == pos
            and job["negative"] == n - pos
            and job["fit_rows"] == n,
            "Classifier labels/counts differ",
        )
        close(
            job["class_weight_balanced"]["positive"],
            n / (2 * pos),
            "Positive class weight differs",
        )
        close(
            job["class_weight_balanced"]["negative"],
            n / (2 * (n - pos)),
            "Negative class weight differs",
        )
        for alias in job["aliases"]:
            key = (job["variant"], alias)
            a.need(
                key not in covered and aliases[key] == ids,
                "Classifier pool alias differs",
            )
            covered[key] = ids
        if job["reused_existing"]:
            a.need(
                a.sha(folder / "complete.json") == job["existing_receipt_sha256"],
                "Reused classifier receipt changed",
            )
            old += 1
        else:
            a.need(
                not (folder / "complete.json").exists(), "Unexpected new classifier fit"
            )
            fresh += 1
    a.need(
        covered == aliases
        and len(aliases) == 74
        and fresh == 64
        and old == 2
        and handoff["actual_new_fits"] == 0,
        "Classifier exact-pool reuse coverage differs",
    )
    evals = list(a.rows(a.STUDY / "evaluation.jsonl"))
    roster = list(a.rows(a.LINEAR / "roster.jsonl"))
    meta = list(a.rows(a.REVIEW / "part7/metadata.jsonl"))
    for entry in handoff["evaluation_mapping"]:
        i, j = entry["iterative_index"], entry["classifier_index"]
        r = evals[i]
        q = roster[j]
        a.need(
            r["key"] == q["key"] == entry["key"]
            and r["text"] == q["text"]
            and r["gate"] == q["gate"] == entry["gold_hate"]
            and r["inference_cluster"]
            == q["inference_cluster"]
            == entry["inference_cluster"],
            "Classifier evaluation mapping differs",
        )
        a.need(
            entry["original_feature_index"] == data["evaluation_indices"][j]
            and entry["masks"] == meta[i]["masks"]
            and hashlib.sha256(r["text"].encode()).hexdigest()
            == entry["exact_text_sha256"],
            "Feature index/text/mask differs",
        )
    a.need(len(handoff["evaluation_mapping"]) == 2077, "Incomplete evaluation mapping")
    a.write(
        a.OUT / "details_independent_verification.json",
        dict(
            status="part8_details_independent_verification_pass",
            created_utc=a.pure.utc(),
            source_sha256=a.sha(Path(__file__)),
            inputs=a.bindings(
                [
                    a.OUT / f
                    for f in [
                        "assembled.json",
                        "repairs_complete.json",
                        "repair_depth.json",
                        "failure_costs.json",
                        "repair_vote_costs.jsonl.gz",
                        "classifier_handoff.json",
                        "chronology.json",
                    ]
                ]
            ),
            chronology_bindings=bindings,
            targeted_original_results=checked,
            repaired_vote_paths=vote_checks,
            retry_events=669,
            chronology_stages=38,
            attempts=40,
            availability_GETs=40,
            timing_pieces=pieces,
            classifier_logical_jobs=74,
            unique_jobs=66,
            reused_baselines=2,
            pending_new_fits=64,
            evaluation_mapping=2077,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        "Independent chronology, repairs and exact-pool handoff PASS; no fits started.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()

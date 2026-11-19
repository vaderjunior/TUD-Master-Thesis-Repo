import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import socket
from unittest.mock import patch
from analysis.common import batch as b
from analysis.common import execution_history as history

a = b.a
ROOT = b.ROOT
BASE = b.BASE


def forbidden(*args, **kwargs):
    raise AssertionError("No network in completed-stage audit")


def require_equal(left, right, message):
    a.require(left == right, message)


def independent_vote(predictions):

    valid = [x for x in predictions if x is not None]
    n = len(valid)
    if not n:
        return (None, False)
    yes = sum((x["hate"] is True for x in valid))
    hate = yes * 2 > n
    tie = yes * 2 == n
    result = {"hate": hate}
    for field in ("target_group", "hate_type", "legal"):
        count = Counter((label for p in valid for label in p[field]))
        result[field] = sorted((label for label, v in count.items() if v * 2 > n))
    order = ["low", "medium", "high"]
    sevs = sorted(
        (order.index(p["severity"]) for p in valid if p["severity"] is not None)
    )
    result["severity"] = order[sevs[(len(sevs) - 1) // 2]] if sevs else None
    if not hate:
        result.update(target_group=[], hate_type=[], severity=None)
    return (result, tie)


def audit(stage_id):
    a.require(
        stage_id == "stage_09_secondary", "Version 4 is scoped to reviewed stage 09"
    )
    from src.hsrag.parse import parse_response
    import numpy as np

    plan = b.validate()
    schedule = a.read(BASE / "schedule.json")
    cat = b.catalog()
    stage = next((x for x in schedule["stages"] if x["id"] == stage_id))
    jobs = stage["jobs"]
    target = a.storage_path(BASE / "stages" / stage_id)
    complete = target / "complete.json"
    record = a.read(complete)
    dest = a.storage_path(BASE / "audits" / f"{stage_id}_v4")
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = (
        ROOT
        / f"Thesis_Log/astra_result/evidence/core_api_{stage_id}_audit_{date}_v4.json"
    )
    a.require(
        not out.exists() and (not (dest / "complete.json").exists()),
        "Preserve completed audit",
    )
    a.require(
        not list(target.glob("stop_*.json")),
        "Stage contains a preserved stop requiring review",
    )
    require_equal(record["plan_sha256"], a.digest(b.PLAN), "Stage plan differs")
    require_equal(record["stage"], stage_id, "Wrong completed stage")
    a.require(
        not record["summary"]["errors"] and (not record["summary"]["stopped"]),
        "Stage did not complete cleanly",
    )
    live = a.storage_path(BASE / "live")
    expected = {b.rel(b.item_dir(live, j) / "complete.json") for j in jobs}
    require_equal(
        set(record["item_receipt_hashes"]), expected, "Missing/extra stage prediction"
    )
    inputs = {
        **plan["input_hashes"],
        **plan["source_hashes"],
        **plan["generated_hashes"],
        b.rel(b.PLAN): a.digest(b.PLAN),
        b.rel(complete): a.digest(complete),
    }
    ready = a.read(BASE / "ready.json")
    pilot_review = a.read(BASE / "live_review_complete_v1.json")
    require_equal(
        a.digest(BASE / "ready.json"),
        pilot_review["ready_sha256"],
        "Pilot readiness changed",
    )
    require_equal(
        a.digest(ROOT / pilot_review["evidence"]),
        pilot_review["evidence_sha256"],
        "Pilot audit changed",
    )
    for name, h in ready["pilot_item_receipt_hashes"].items():
        require_equal(a.digest(ROOT / name), h, "Pilot result changed")
    history_record = history.inspect(stage, record, plan, target)
    inputs.update(history_record["inputs"])
    start = history_record["start"]
    live_dirs = {p.name for p in (live / "items").iterdir() if p.is_dir()}
    known = {
        a.signature(b.identity(j)) for st in schedule["stages"] for j in st["jobs"]
    }
    a.require(live_dirs <= known, "Unscheduled live item directory")
    a.require(
        all(((live / "items" / name / "complete.json").exists() for name in live_dirs)),
        "Incomplete live item outside stage summary",
    )
    counts = Counter()
    accounting = Counter()
    http = Counter()
    models = Counter()
    finishes = Counter()
    failures = Counter()
    errors = Counter()
    first_errors = Counter()
    failed_vote_errors = Counter()
    raw_turns = Counter()
    by_group = defaultdict(Counter)
    usage_counts = Counter()
    usage_sum = Counter()
    latencies = []
    response_latencies = []
    backoff = 0.0
    hashes = {}
    repair_items = []
    failed_votes = []
    retry_rows = []
    ties = []
    request_ids = Counter()
    request_id_occurrences = 0
    null_details = []
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ):
        for number, job in enumerate(jobs, 1):
            d = b.item_dir(live, job)
            p = d / "complete.json"
            name = b.rel(p)
            require_equal(
                a.digest(p),
                record["item_receipt_hashes"][name],
                "Stage item receipt changed",
            )
            a.audit_item(
                live,
                b.identity(job),
                cat[job["key"], job["condition"]]["messages"],
                execution_mode="live",
            )
            receipt = a.read(p)
            result = a.read(d / "result.json")
            hashes[name] = a.digest(p)
            group = by_group[job["subset"] + "/" + job["condition"]]
            for c in (counts, group):
                c["items"] += 1
                c["valid_votes_" + str(result["n_valid"])] += 1
                c["null_items"] += result["result"] is None
                for field in (
                    "parse_failures",
                    "repairs",
                    "normalisations",
                    "gate_normalised",
                ):
                    c[field] += result[field]
                c["gate_ties"] += result["uncertain"]
                c["items_with_repairs"] += result["repairs"] > 0
            if result["result"] is None:
                null_details.append({"job": job, "valid_votes": result["n_valid"]})
            if result["repairs"]:
                repair_items.append(job)
            if result["uncertain"]:
                ties.append(job)
            accounting.update(result["accounting"])
            latencies.extend(result["latencies"])
            voted, tie = independent_vote(result["run_predictions"])
            require_equal(tie, result["uncertain"], "Independent gate tie differs")
            if voted is None:
                require_equal(result["result"], None, "Independent null differs")
            else:
                require_equal(
                    {k: result["result"][k] for k in voted},
                    voted,
                    "Independent voted fields differ",
                )
            for vote in range(3):
                saved = a.read(d / "votes" / f"{vote}.json")
                raw = saved["raw_outputs"]
                parsed = []
                group["votes"] += 1
                group["votes_needing_repair"] += len(raw) > 1
                counts["votes_needing_repair"] += len(raw) > 1
                raw_turns[str(len(raw))] += 1
                for turn, content in enumerate(raw):
                    val, _, err = parse_response(content)
                    parsed.append(val)
                    if val is None:
                        errors[err] += 1
                        if turn == 0:
                            first_errors[err] += 1
                require_equal(
                    len(raw) - 1,
                    saved["repairs"],
                    "Repair count differs from response turns",
                )
                if saved["parse_failure"]:
                    a.require(
                        len(raw) == 3 and all((x is None for x in parsed)),
                        "Invalid vote exceeded/stopped before repair cap",
                    )
                    failed_vote_errors[saved["error"]] += 1
                    failed_votes.append(
                        {"job": job, "vote": vote, "error": saved["error"]}
                    )
                else:
                    a.require(
                        parsed[-1] is not None
                        and all((x is None for x in parsed[:-1])),
                        "Repair continued after valid vote",
                    )
            physical = list(d.glob("calls/*/attempt_*/outcome.json"))
            intents = list(d.glob("calls/*/attempt_*/intent.json"))
            require_equal(
                len(physical),
                result["accounting"]["post_attempts"],
                "Response count differs",
            )
            require_equal(
                {p.parent for p in physical},
                {p.parent for p in intents},
                "Unresolved request intent",
            )
            events = {(e["vote"], e["turn"], e["attempt"]): e for e in result["events"]}
            for path in physical:
                outcome = a.read(path)
                intent = a.read(path.with_name("intent.json"))
                a.require(
                    "administrative_recovery" not in outcome,
                    "Administrative unknown outcome needs recovery-aware review",
                )
                event = events[intent["vote"], intent["turn"], intent["attempt"]]
                backoff += intent["backoff_elapsed_s"]
                response_latencies.append(outcome["elapsed_s"])
                group["physical_POSTs"] += 1
                group["transport_retries"] += intent["attempt"] > 0
                if event["action"] != "ok":
                    failures[event["response_class"]] += 1
                    retry_rows.append(
                        {
                            "job": job,
                            "vote": intent["vote"],
                            "turn": intent["turn"],
                            "attempt": intent["attempt"],
                            "class": event["response_class"],
                            "remote_completion_uncertain": outcome.get(
                                "remote_completion_uncertain", False
                            ),
                        }
                    )
                if outcome["kind"] == "transport_error":
                    http["transport_error:" + outcome["category"]] += 1
                    continue
                http[str(outcome["status"])] += 1
                if event["action"] != "ok":
                    continue
                data = json.loads(
                    base64.b64decode(outcome["body_base64"], validate=True)
                )
                choice = data["choices"][0]
                models[str(data.get("model"))] += 1
                finishes[str(choice.get("finish_reason"))] += 1
                if data.get("id"):
                    request_ids[str(data["id"])] += 1
                    request_id_occurrences += 1
                usage = data.get("usage")
                if isinstance(usage, dict):
                    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        value = usage.get(field)
                        if type(value) in (int, float) and a.finite_nonnegative(value):
                            usage_counts[field] += 1
                            usage_sum[field] += value
            for relative, h in receipt["hashes"].items():
                hashes[b.rel(live / relative)] = h
            if number % 250 == 0 or number == len(jobs):
                print(
                    f"Audited {number}/{len(jobs)} predictions from raw responses",
                    flush=True,
                )
    require_equal(counts["items"], stage["items"], "Final prediction count differs")
    require_equal(
        accounting["ordinary_logical_calls"],
        stage["ordinary_responses"],
        "Ordinary response count differs",
    )
    for k, v in record["summary"]["completed_item_accounting"].items():
        require_equal(accounting[k], v, "Stage accounting differs: " + k)
    require_equal(
        counts["null_items"], record["summary"]["null_items"], "Null summary differs"
    )
    require_equal(
        accounting["repair_logical_calls"],
        counts["repairs"],
        "Repair accounting differs",
    )
    require_equal(
        accounting["post_attempts"],
        sum(http.values()),
        "Physical attempt accounting differs",
    )
    require_equal(
        accounting["http_success_with_text"],
        sum(models.values()),
        "Successful response metadata count differs",
    )
    require_equal(
        accounting["post_attempts"],
        accounting["ordinary_logical_calls"]
        + accounting["repair_logical_calls"]
        + accounting["retry_attempts"],
        "Call decomposition differs",
    )
    require_equal(
        sum(errors.values()),
        counts["repairs"] + counts["parse_failures"],
        "Invalid-response accounting differs",
    )
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(b, "live_client", forbidden):
        b.run_stage(stage_id)
    require_equal(
        a.digest(complete),
        inputs[b.rel(complete)],
        "Stage receipt changed during restart check",
    )
    elapsed = (
        datetime.fromisoformat(record["completed_utc"])
        - datetime.fromisoformat(start["started_utc"])
    ).total_seconds()
    for p in [
        BASE / "ready.json",
        BASE / "live_review_complete_v1.json",
        ROOT / pilot_review["evidence"],
        ROOT / "analysis/common/validate_stage.py",
    ]:
        inputs[b.rel(p)] = a.digest(p)
    for name, h in inputs.items():
        require_equal(a.digest(ROOT / name), h, "Input changed during audit")
    stage_index = schedule["stages"].index(stage)
    next_stage = (
        schedule["stages"][stage_index + 1]
        if stage_index + 1 < len(schedule["stages"])
        else None
    )
    phase_stages = [x for x in schedule["stages"] if x["phase"] == stage["phase"]]
    payload_keys = {x["key"] for x in jobs}
    by_subset = Counter()
    for subset in sorted({x["subset"] for x in jobs}):
        by_subset[subset] = len({x["key"] for x in jobs if x["subset"] == subset})
    report = {
        "created_utc": a.utc(),
        "status": "completed_stage_verified_next_stage_ready",
        "stage": stage_id,
        "plan_sha256": a.digest(b.PLAN),
        "source_sha256": a.digest(ROOT / "analysis/common/validate_stage.py"),
        "stage_complete_sha256": a.digest(complete),
        "start_utc": start["started_utc"],
        "completed_utc": record["completed_utc"],
        "wall_seconds_including_final_save": elapsed,
        "execution_history": history_record["summary"],
        "unique_inputs": len(payload_keys),
        "inputs_by_subset": dict(by_subset),
        "null_prediction_details": null_details,
        "counts": dict(counts),
        "accounting": dict(accounting),
        "http_statuses": dict(http),
        "reported_models": dict(models),
        "finish_reasons": dict(finishes),
        "validation_errors_all_response_turns": dict(errors),
        "first_response_validation_errors": dict(first_errors),
        "terminal_invalid_vote_errors": dict(failed_vote_errors),
        "raw_responses_per_vote_distribution": dict(raw_turns),
        "transport_failure_classes": dict(failures),
        "by_subset_condition": {k: dict(v) for k, v in sorted(by_group.items())},
        "failed_votes": failed_votes,
        "retry_events": retry_rows,
        "gate_tie_jobs": ties,
        "observed_usage_field_counts": dict(usage_counts),
        "observed_usage_sums": dict(usage_sum),
        "response_metadata_id_occurrences": request_id_occurrences,
        "distinct_response_metadata_ids": len(request_ids),
        "duplicate_response_metadata_ids": {
            k: v for k, v in request_ids.items() if v > 1
        },
        "latency": {
            "vote_seconds_sum_including_retries": sum(latencies),
            "physical_attempt_seconds_sum": sum(response_latencies),
            "backoff_seconds_sum": backoff,
            "vote_seconds_percentiles": dict(
                zip(
                    ["p50", "p90", "p95", "p99", "max"],
                    np.percentile(latencies, [50, 90, 95, 99, 100]).tolist(),
                )
            ),
        },
        "semantic_replay_items": len(jobs),
        "independent_field_vote_checks": len(jobs),
        "completed_stage_restart_new_POSTs": 0,
        "verified_input_hashes": inputs,
        "verified_output_hashes": hashes,
        "actual_audit_API_calls": 0,
        "actual_gold_scoring": False,
        "live_item_directories_at_audit": len(live_dirs),
        "phase_complete": all(
            (
                (BASE / "stages" / x["id"] / "complete.json").exists()
                for x in phase_stages
            )
        ),
        "next_stage": (
            {k: v for k, v in next_stage.items() if k != "jobs"} if next_stage else None
        ),
        "next_stage_started": (
            (BASE / "stages" / next_stage["id"]).exists() if next_stage else False
        ),
        "limits": [
            "Operational integrity audit, no F1 or annotation-based comparison on this partial roster.",
            "Part-valid predictions remain in the study under the frozen vote/coverage rules; no favourable resampling.",
            "Reported model names do not prove backend checkpoint identity; API latencies exclude offline retrieval.",
            "Transport exceptions can leave remote completion uncertain even when a later retry succeeds; saved attempts remain visible.",
            "This report audits one scheduled batch; final endpoint comparisons require their complete frozen rosters. Do not tune or stop based on partial outcomes.",
        ],
    }
    a.save_new(out, report)
    a.save_new(
        dest / "complete.json",
        {
            "status": report["status"],
            "stage": stage_id,
            "evidence": b.rel(out),
            "evidence_sha256": a.digest(out),
            "stage_complete_sha256": a.digest(complete),
            "plan_sha256": a.digest(b.PLAN),
        },
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "stage",
                    "wall_seconds_including_final_save",
                    "null_prediction_details",
                    "counts",
                    "accounting",
                    "http_statuses",
                    "validation_errors_all_response_turns",
                    "first_response_validation_errors",
                    "terminal_invalid_vote_errors",
                    "transport_failure_classes",
                    "observed_usage_sums",
                    "latency",
                    "phase_complete",
                    "next_stage",
                    "next_stage_started",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    print("Evidence SHA256: " + a.digest(out), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True)
    args = parser.parse_args()
    audit(args.stage)

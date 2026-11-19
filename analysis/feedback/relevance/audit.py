import argparse
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import socket
from unittest.mock import patch
from analysis.feedback.relevance import run as b

a = b.a
ROOT = b.ROOT
BASE = b.BASE


def forbidden(*args, **kwargs):
    raise AssertionError("No network in completed-stage audit")


def require_equal(left, right, message):
    a.require(left == right, message)


def timestamp(value):
    result = datetime.fromisoformat(value)
    a.require(result.tzinfo is not None, "Timezone missing from saved time")
    return result


def model_roster_check(intent, response, start, end):
    require_equal(intent["action"], "GET PEASEC /models", "Wrong availability action")
    require_equal(intent["model"], a.MODEL, "Availability model differs")
    require_equal(response["status"], 200, "Availability HTTP failure")
    a.require(
        a.finite_nonnegative(response["elapsed_s"]), "Invalid availability timing"
    )
    when = timestamp(intent["started_utc"])
    a.require(start <= when <= end, "Availability request outside stage")
    raw = base64.b64decode(response["body_base64"], validate=True)
    require_equal(
        hashlib.sha256(raw).hexdigest(),
        response["body_sha256"],
        "Availability body hash differs",
    )
    data = json.loads(raw)
    models = [row["id"] for row in data["data"]]
    a.require(a.MODEL in models, "Pinned model absent from saved roster")
    return {
        "status": 200,
        "pinned_model_present": True,
        "model_count": len(models),
        "body_sha256": response["body_sha256"],
        "elapsed_s": response["elapsed_s"],
    }


def call_timing_check(intent, outcome, start, end, previous_end=None):
    began = timestamp(intent["started_utc"])
    finished = timestamp(outcome["finished_utc"])
    a.require(
        start <= began <= finished <= end,
        "Request chronology outside stage or reversed",
    )
    a.require(
        previous_end is None or previous_end <= began,
        "Sequential item calls overlap or reverse",
    )
    a.require(
        a.finite_nonnegative(outcome["elapsed_s"])
        and a.finite_nonnegative(intent["backoff_elapsed_s"]),
        "Invalid request timing",
    )
    if intent["attempt"] == 0:
        require_equal(intent["backoff_elapsed_s"], 0, "Initial call has retry backoff")
    else:
        a.require(1 <= intent["attempt"] <= 4, "Retry exceeds frozen budget")
        a.require(
            intent["backoff_elapsed_s"]
            >= a.CONTRACT["backoff_seconds"][intent["attempt"] - 1] - 0.1,
            "Saved retry wait below frozen budget",
        )
    return finished


def independent_vote(predictions):

    valid = [x for x in predictions if x is not None]
    n = len(valid)
    if not n:
        return None, False
    yes = sum(x["hate"] is True for x in valid)
    hate = yes * 2 > n
    tie = yes * 2 == n
    result = {"hate": hate}
    for field in ("target_group", "hate_type", "legal"):
        count = Counter(label for p in valid for label in p[field])
        result[field] = sorted(label for label, v in count.items() if v * 2 > n)
    order = ["low", "medium", "high"]
    sevs = sorted(
        order.index(p["severity"]) for p in valid if p["severity"] is not None
    )
    result["severity"] = order[sevs[(len(sevs) - 1) // 2]] if sevs else None
    if not hate:
        result.update(target_group=[], hate_type=[], severity=None)
    return result, tie


def audit(stage_id):
    from src.hsrag.parse import parse_response
    import numpy as np

    plan = b.validate()
    plan_sha = a.digest(b.PLAN)
    schedule = a.read(BASE / "schedule.json")
    cat = b.frozen_catalog(ROOT / plan["prompts"])
    stage = next(x for x in schedule["stages"] if x["id"] == stage_id)
    jobs = stage["jobs"]
    stage_index = schedule["stages"].index(stage)
    b.previous_review(stage_index, schedule, plan_sha)
    tests_path = BASE / "audit_development/stage_check_v1/tests_passed.json"
    tests = a.read(tests_path)
    a.require(
        tests["passed"] is True and tests["environment"] == b.core.environment(),
        "Passing audit tests required",
    )
    for name, digest in tests["source_hashes"].items():
        require_equal(a.digest(ROOT / name), digest, "Audit source changed after tests")
    target = a.storage_path(BASE / "live/stages" / stage_id)
    complete = target / "complete.json"
    record = a.read(complete)
    dest = a.storage_path(BASE / "audits" / f"{stage_id}_v1")
    date = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = (
        ROOT
        / f"Thesis_Log/astra_result/evidence/sq3_relevance_{stage_id}_audit_{date}_v1.json"
    )
    a.require(
        not out.exists() and not (dest / "complete.json").exists(),
        "Preserve completed audit",
    )
    a.require(
        not (BASE / "stage_reviews" / f"{stage_id}.json").exists(),
        "Preserve published stage review",
    )
    a.require(
        not list(target.glob("stop_*.json")),
        "Stage contains a preserved stop requiring review",
    )
    require_equal(record["plan_sha256"], a.digest(b.PLAN), "Stage plan differs")
    require_equal(record["stage"], stage_id, "Wrong completed stage")
    a.require(
        not record["summary"]["errors"] and not record["summary"]["stopped"],
        "Stage did not complete cleanly",
    )
    live = a.storage_path(BASE / "live")
    expected = {b.rel(b.item_dir(live, j, plan_sha) / "complete.json") for j in jobs}
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
    ready = a.read(BASE / "offline_check_complete.json")
    require_equal(ready["plan_sha256"], plan_sha, "Offline readiness plan differs")
    require_equal(
        ready["status"],
        "offline_checked_ready_for_user_relevance",
        "Offline readiness differs",
    )
    require_equal(
        a.digest(ROOT / ready["evidence"]),
        ready["evidence_sha256"],
        "Offline evidence changed",
    )
    attempts = sorted(target.glob("attempt_*"))
    a.require(
        len(attempts) == 1,
        "Unexpected multiple stage attempts; review execution history",
    )
    start = a.read(attempts[0] / "start.json")
    attempt_summary = a.read(attempts[0] / "summary.json")
    require_equal(
        attempt_summary, record["summary"], "Final stage summary differs from attempt"
    )
    require_equal(start["execution_mode"], "live", "Attempt is not live")
    require_equal(record["execution_mode"], "live", "Stage is not live")
    require_equal(start["plan_sha256"], a.digest(b.PLAN), "Attempt plan changed")
    require_equal(start["stage"], stage_id, "Attempt stage changed")
    require_equal(record["attempt"], b.rel(attempts[0]), "Attempt path differs")
    require_equal(
        record["summary_sha256"],
        a.digest(attempts[0] / "summary.json"),
        "Summary hash differs",
    )
    require_equal(
        {p.name for p in attempts[0].iterdir()},
        {
            "start.json",
            "summary.json",
            "model_check_intent.json",
            "model_check_response.json",
        },
        "Unexpected attempt history",
    )
    require_equal(
        {p.name for p in target.iterdir()},
        {attempts[0].name, "complete.json"},
        "Unexpected stage history",
    )
    started = timestamp(start["started_utc"])
    finished = timestamp(attempt_summary["finished_utc"])
    a.require(
        started <= finished <= timestamp(record["completed_utc"]),
        "Stage chronology differs",
    )
    availability_intent = a.read(attempts[0] / "model_check_intent.json")
    availability = model_roster_check(
        availability_intent,
        a.read(attempts[0] / "model_check_response.json"),
        started,
        finished,
    )
    require_equal(attempt_summary["saved_items"], len(jobs), "Saved item count differs")
    require_equal(
        attempt_summary["submitted"],
        len(jobs),
        "Uninterrupted submission count differs",
    )
    require_equal(attempt_summary["unsubmitted"], 0, "Unsubmitted jobs remain")
    require_equal(
        attempt_summary["stats"],
        {"complete": len(jobs), "null": attempt_summary["null_items"]},
        "Summary stats differ",
    )
    a.require(
        0 < attempt_summary["peak_submitted"] <= b.WORKERS, "Worker limit exceeded"
    )
    for p in (attempts[0] / "start.json", attempts[0] / "summary.json"):
        inputs[b.rel(p)] = a.digest(p)
    live_dirs = {p.name for p in (live / "items").iterdir() if p.is_dir()}
    known = {
        a.signature(b.identity(j, plan_sha))
        for st in schedule["stages"][: stage_index + 1]
        for j in st["jobs"]
    }
    require_equal(
        live_dirs, known, "Missing, extra or future-stage live item directory"
    )
    require_equal(
        {p.name for p in (live / "stages").iterdir()},
        {st["id"] for st in schedule["stages"][: stage_index + 1]},
        "Unexpected stage directory",
    )
    for st in schedule["stages"][:stage_index]:
        a.require(
            (live / "stages" / st["id"] / "complete.json").exists(),
            "Earlier stage incomplete",
        )
    a.require(
        all((live / "items" / name / "complete.json").exists() for name in live_dirs),
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
    timing_checks = 0
    usage_arithmetic_checks = 0
    expected_payloads = set()
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ):
        for number, job in enumerate(jobs, 1):
            d = b.item_dir(live, job, plan_sha)
            p = d / "complete.json"
            name = b.rel(p)
            require_equal(
                a.digest(p),
                record["item_receipt_hashes"][name],
                "Stage item receipt changed",
            )
            a.audit_item(
                live,
                b.identity(job, plan_sha),
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
                        len(raw) == 3 and all(x is None for x in parsed),
                        "Invalid vote exceeded/stopped before repair cap",
                    )
                    failed_vote_errors[saved["error"]] += 1
                    failed_votes.append(
                        {"job": job, "vote": vote, "error": saved["error"]}
                    )
                else:
                    a.require(
                        parsed[-1] is not None and all(x is None for x in parsed[:-1]),
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
            previous_end = None
            for event in result["events"]:
                call = (
                    d
                    / "calls"
                    / f"vote_{event['vote']}_turn_{event['turn']}"
                    / f"attempt_{event['attempt']}"
                )
                intent = a.read(call / "intent.json")
                outcome = a.read(call / "outcome.json")
                previous_end = call_timing_check(
                    intent,
                    outcome,
                    timestamp(availability_intent["started_utc"]),
                    finished,
                    previous_end,
                )
                timing_checks += 1
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
                    token_values = [
                        usage.get(field)
                        for field in (
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                        )
                    ]
                    if all(type(v) is int and v >= 0 for v in token_values):
                        require_equal(
                            token_values[0] + token_values[1],
                            token_values[2],
                            "Usage total differs from parts",
                        )
                        usage_arithmetic_checks += 1
                    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                        value = usage.get(field)
                        if type(value) in (int, float) and a.finite_nonnegative(value):
                            usage_counts[field] += 1
                            usage_sum[field] += value
            for relative, h in receipt["hashes"].items():
                hashes[b.rel(live / relative)] = h
                if (live / relative).parent == live / "payloads":
                    expected_payloads.add((live / relative).name)
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
    require_equal(
        timing_checks, accounting["post_attempts"], "Request timing check count differs"
    )

    for st in schedule["stages"][:stage_index]:
        for job in st["jobs"]:
            receipt = a.read(b.item_dir(live, job, plan_sha) / "complete.json")
            expected_payloads.update(
                (live / name).name
                for name in receipt["hashes"]
                if (live / name).parent == live / "payloads"
            )
    require_equal(
        {p.name for p in (live / "payloads").iterdir()},
        expected_payloads,
        "Unattributed/missing live payload",
    )
    print(
        "Raw response audit finished. Verifying complete-stage restart and source hashes; no new calls.",
        flush=True,
    )

    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(b.core, "live_client", forbidden):
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
        BASE / "offline_check_complete.json",
        ROOT / ready["evidence"],
        *attempts[0].glob("*.json"),
        ROOT / "analysis/feedback/relevance/audit.py",
        ROOT / "tests/feedback/relevance/test_audit.py",
        tests_path,
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
    endpoint_progress = b.endpoint_progress(schedule, stage_index)

    payload_keys = {x["key"] for x in jobs}
    by_subset = Counter()
    for subset in sorted({x["subset"] for x in jobs}):
        by_subset[subset] = len({x["key"] for x in jobs if x["subset"] == subset})
    report = {
        "created_utc": a.utc(),
        "status": "completed_relevance_stage_verified",
        "stage": stage_id,
        "plan_sha256": a.digest(b.PLAN),
        "source_sha256": a.digest(ROOT / "analysis/feedback/relevance/audit.py"),
        "stage_complete_sha256": a.digest(complete),
        "start_utc": start["started_utc"],
        "completed_utc": record["completed_utc"],
        "model_availability_check": availability,
        "request_chronology_checks": timing_checks,
        "usage_arithmetic_checks": usage_arithmetic_checks,
        "attributed_live_payload_files": len(expected_payloads),
        "auditor_tests": {
            "count": tests["tests_run"],
            "receipt": b.rel(tests_path),
            "sha256": a.digest(tests_path),
        },
        "wall_seconds_including_final_save": elapsed,
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
        "endpoint_progress": endpoint_progress,
        "next_stage": (
            {k: v for k, v in next_stage.items() if k != "jobs"} if next_stage else None
        ),
        "next_stage_started": (
            (BASE / "live/stages" / next_stage["id"]).exists() if next_stage else False
        ),
        "limits": [
            "Operational integrity audit only; no F1 here. Score a complete endpoint separately under its frozen analysis contract.",
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
    a.save_new(
        BASE / "stage_reviews" / f"{stage_id}.json",
        {
            "status": "audited",
            "plan_sha256": plan_sha,
            "stage": stage_id,
            "complete_sha256": a.digest(complete),
            "evidence": b.rel(out),
            "evidence_sha256": a.digest(out),
        },
    )
    if next_stage:
        b.previous_review(stage_index + 1, schedule, plan_sha)
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
                    "endpoint_progress",
                    "next_stage",
                    "next_stage_started",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    print("Evidence SHA256: " + a.digest(out), flush=True)


def tests():
    import io
    import time
    import unittest

    dest = BASE / "audit_development/stage_check_v1"
    a.require(
        not (dest / "tests_passed.json").exists(), "Preserve completed audit tests"
    )
    names = [
        "analysis/feedback/relevance/audit.py",
        "tests/feedback/relevance/test_audit.py",
        "analysis/knowledge/strategies/validate_stage.py",
        "analysis/feedback/relevance/run.py",
        *a.DEPENDENCIES,
    ]
    hashes = {name: a.digest(ROOT / name) for name in names}
    attempt = dest / ("attempt_" + str(time.time_ns()))
    a.save_new(
        attempt / "sources.json",
        {name: (ROOT / name).read_text(encoding="utf8") for name in names[:2]},
    )
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "tests.feedback.relevance.test_audit"
    )
    stream = io.StringIO()
    tick = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    receipt = {
        "created_utc": a.utc(),
        "passed": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "elapsed_s": time.perf_counter() - tick,
        "output": stream.getvalue(),
        "source_hashes": hashes,
        "environment": b.core.environment(),
        "actual_API_calls": 0,
        "attempt": b.rel(attempt),
    }
    a.save_new(attempt / "tests.json", receipt)
    print(stream.getvalue(), flush=True)
    a.require(
        result.wasSuccessful(), "Audit tests failed; preserved source and attempt"
    )
    for name, digest in hashes.items():
        require_equal(a.digest(ROOT / name), digest, "Source changed during tests")
    a.save_new(dest / "tests_passed.json", receipt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--stage")
    group.add_argument("--tests", action="store_true")
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(b.core, "live_client", forbidden):
        tests() if args.tests else audit(args.stage)

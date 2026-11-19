import base64
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
import socket
from unittest.mock import patch

from analysis.feedback.supplements import run as b

a = b.a


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    a.require(parsed.tzinfo is not None, "Timestamp missing timezone")
    return parsed


def timing(intent, outcome, start, end, previous=None, live=True):
    first = timestamp(intent["started_utc"])
    last = timestamp(outcome["finished_utc"])
    a.require(
        start <= first <= last <= end and (previous is None or previous <= first),
        "Request chronology differs",
    )
    a.require(
        a.finite_nonnegative(intent["backoff_elapsed_s"])
        and a.finite_nonnegative(outcome["elapsed_s"]),
        "Invalid request timing",
    )
    attempt = intent["attempt"]
    a.require(type(attempt) is int and 0 <= attempt <= 4, "Retry limit differs")
    if attempt == 0:
        a.require(intent["backoff_elapsed_s"] == 0, "Initial call has retry wait")
    elif live:
        a.require(
            intent["backoff_elapsed_s"]
            >= a.CONTRACT["backoff_seconds"][attempt - 1] - 0.1,
            "Retry wait below contract",
        )
    return last


def independent_vote(predictions):
    valid = [p for p in predictions if p is not None]
    if not valid:
        return None, False
    n = len(valid)
    yes = sum(p["hate"] for p in valid)
    out = {"hate": yes * 2 > n}
    for field in ["target_group", "hate_type", "legal"]:
        counts = Counter(l for p in valid for l in p[field])
        out[field] = sorted(l for l, c in counts.items() if c * 2 > n)
    order = ["low", "medium", "high"]
    ranked = sorted(
        order.index(p["severity"]) for p in valid if p["severity"] is not None
    )
    out["severity"] = order[ranked[(len(ranked) - 1) // 2]] if ranked else None
    if not out["hate"]:
        out.update(target_group=[], hate_type=[], severity=None)
    return out, yes * 2 == n


def model_check(folder, start, end):
    intent = a.read(folder / "model_check_intent.json")
    response = a.read(folder / "model_check_response.json")
    a.require(
        intent["action"] == "GET PEASEC /models"
        and intent["model"] == a.MODEL
        and start <= timestamp(intent["started_utc"]) <= end,
        "Model check attribution differs",
    )
    raw = base64.b64decode(response["body_base64"], validate=True)
    a.require(
        response["status"] == 200
        and hashlib.sha256(raw).hexdigest() == response["body_sha256"]
        and a.finite_nonnegative(response["elapsed_s"]),
        "Model availability response differs",
    )
    a.require(
        a.MODEL in {v["id"] for v in json.loads(raw)["data"]}, "Pinned model absent"
    )


def audit_stage(stage, cat, run_root, plan_sha, *, mode):
    root = b.checked_root(run_root, stage["study"], mode)
    target = root / "stages" / stage["id"]
    if (target / "review.json").exists():
        return b.verify_review(stage, root, plan_sha, mode)
    a.require(
        not list(target.glob("stop_*.json")), "Preserved execution stop blocks audit"
    )
    complete = a.read(target / "complete.json")
    backends = b.engines(cat)
    a.require(
        complete["plan_sha256"] == plan_sha
        and complete["stage"] == stage["id"]
        and complete["study"] == stage["study"]
        and complete["mode"] == mode,
        "Stage completion differs",
    )
    summary = complete["summary"]
    final_attempt = b.ROOT / complete["final_attempt"]
    a.require(
        not summary["errors"]
        and not summary["stopped"]
        and summary["saved_items"] == stage["items"],
        "Stage incomplete",
    )
    a.require(
        a.digest(final_attempt / "summary.json") == complete["summary_sha256"]
        and a.read(final_attempt / "summary.json") == summary,
        "Final summary binding differs",
    )
    expected = {
        b.item_dir(root, j, plan_sha).relative_to(root).as_posix() + "/complete.json"
        for j in stage["jobs"]
    }
    a.require(
        set(complete["item_receipt_hashes"]) == expected, "Wrong stage item roster"
    )
    attempts = sorted(target.glob("attempt_*"))
    end = timestamp(complete["completed_utc"])
    starts = []
    history = {}
    model_checks = 0
    a.require(bool(attempts), "No stage attempt history")
    for attempt in attempts:
        files = {p.name for p in attempt.iterdir()}
        allowed = {
            "start.json",
            "inventory.json",
            "summary.json",
            "interrupted.json",
            "model_check_intent.json",
            "model_check_response.json",
        }
        a.require(files <= allowed and "start.json" in files, "Unreviewed attempt file")
        start = a.read(attempt / "start.json")
        a.require(
            start["plan_sha256"] == plan_sha
            and start["stage"] == stage["id"]
            and start["study"] == stage["study"]
            and start["mode"] == mode,
            "Attempt identity differs",
        )
        began = timestamp(start["started_utc"])
        a.require(began <= end, "Attempt chronology differs")
        starts.append(began)
        if "inventory.json" in files:
            inventory = a.read(attempt / "inventory.json")
            pending = inventory["pending_identity_hashes"]
            a.require(
                inventory["completed"] + inventory["pending"] == stage["items"]
                and len(pending) == len(set(pending)) == inventory["pending"],
                "Attempt inventory differs",
            )
            a.require(
                set(pending)
                <= {a.signature(b.identity(j, plan_sha)) for j in stage["jobs"]},
                "Unknown pending prediction",
            )
        if "model_check_intent.json" in files:
            a.require(
                "model_check_response.json" in files, "Unresolved availability request"
            )
            model_check(attempt, began, end)
            model_checks += 1
        elif mode == "mock":
            a.require(
                "model_check_response.json" not in files,
                "Mock network availability check",
            )
        if "summary.json" in files:
            s = a.read(attempt / "summary.json")
            a.require(
                not s["errors"] and not s["stopped"],
                "Failed attempt preserved; requires review",
            )
        for p in attempt.iterdir():
            history[p.relative_to(root).as_posix()] = a.digest(p)
    if mode == "live":
        a.require(model_checks > 0, "Live work lacks saved model availability check")
    start = min(starts)
    counts = Counter()
    accounting = Counter()
    usage = Counter()
    statuses = Counter()
    models = Counter()
    finishes = Counter()
    per_group = defaultdict(Counter)
    errors = Counter()
    ids = []
    hashes = {}
    nulls = []
    retries = []
    latencies = []
    schemas = Counter()
    with patch.object(socket.socket, "connect", b.forbidden), patch.object(
        socket, "create_connection", b.forbidden
    ), patch.object(b.core, "live_client", b.forbidden):
        for number, j in enumerate(stage["jobs"], 1):
            folder = b.item_dir(root, j, plan_sha)
            receipt_path = folder / "complete.json"
            name = receipt_path.relative_to(root).as_posix()
            a.require(
                a.digest(receipt_path) == complete["item_receipt_hashes"][name],
                "Prediction receipt changed",
            )
            engine = backends[j["schema_path"]]
            entry = cat[j["key"], j["condition"]]
            engine.audit_item(
                root, b.identity(j, plan_sha), entry["messages"], execution_mode=mode
            )
            value = a.read(folder / "result.json")
            receipt = a.read(receipt_path)
            for k, h in receipt["hashes"].items():
                a.require(
                    k not in hashes or hashes[k] == h, "Conflicting shared payload hash"
                )
                hashes[k] = h
            hashes[name] = a.digest(receipt_path)
            independent, tie = independent_vote(value["run_predictions"])
            a.require(
                tie == value["uncertain"]
                and independent
                == (
                    {k: value["result"][k] for k in independent}
                    if independent
                    else None
                ),
                "Independent field vote differs",
            )
            group = per_group[j["population"] + "/" + j["condition"]]
            for c in (counts, group):
                c["predictions"] += 1
                c["nulls"] += value["result"] is None
                c["valid_votes_" + str(value["n_valid"])] += 1
                for field in [
                    "repairs",
                    "parse_failures",
                    "normalisations",
                    "gate_normalised",
                ]:
                    c[field] += value[field]
                c["gate_ties"] += value["uncertain"]
            if value["result"] is None:
                nulls.append(j)
            schemas[j["schema_path"]] += 1
            accounting.update(value["accounting"])
            latencies.extend(value["latencies"])
            for raw_run in value["raw_runs"]:
                for raw in raw_run:
                    _, _, error = engine.bundle.parser.parse_response(raw)
                    if error:
                        errors[error] += 1
            previous = None
            for event in value["events"]:
                call = (
                    folder
                    / "calls"
                    / f"vote_{event['vote']}_turn_{event['turn']}"
                    / f"attempt_{event['attempt']}"
                )
                intent = a.read(call / "intent.json")
                outcome = a.read(call / "outcome.json")
                previous = timing(
                    intent, outcome, start, end, previous, live=mode == "live"
                )
                if event["action"] != "ok":
                    retries.append(
                        {
                            "job": j,
                            "vote": event["vote"],
                            "turn": event["turn"],
                            "attempt": event["attempt"],
                            "class": event["response_class"],
                        }
                    )
                if outcome["kind"] == "transport_error":
                    statuses["transport_error:" + outcome["category"]] += 1
                    continue
                statuses[str(outcome["status"])] += 1
                if event["action"] != "ok":
                    continue
                remote = json.loads(
                    base64.b64decode(outcome["body_base64"], validate=True)
                )
                choice = remote["choices"][0]
                models[str(remote.get("model"))] += 1
                finishes[str(choice.get("finish_reason"))] += 1
                a.require(
                    isinstance(remote.get("id"), str) and bool(remote["id"]),
                    "Missing remote response ID: preserve outputs for review",
                )
                ids.append(remote["id"])
                u = remote.get("usage")
                a.require(
                    isinstance(u, dict)
                    and all(
                        type(u.get(k)) is int and u[k] >= 0
                        for k in ["prompt_tokens", "completion_tokens", "total_tokens"]
                    )
                    and u["prompt_tokens"] + u["completion_tokens"]
                    == u["total_tokens"],
                    "Missing/invalid observed usage",
                )
                usage.update(
                    {
                        k: u[k]
                        for k in ["prompt_tokens", "completion_tokens", "total_tokens"]
                    }
                )
            if number % 250 == 0 or number == len(stage["jobs"]):
                print(
                    f"{stage['id']}: audited {number}/{len(stage['jobs'])} saved predictions",
                    flush=True,
                )
    a.require(
        len(ids) == len(set(ids)) == accounting["http_success_with_text"],
        "Response ID reused within batch",
    )
    a.require(
        accounting["ordinary_logical_calls"] == stage["ordinary_calls"]
        and accounting["repair_logical_calls"] == counts["repairs"],
        "Logical calls differ",
    )
    a.require(
        accounting["post_attempts"]
        == sum(statuses.values())
        == accounting["ordinary_logical_calls"]
        + accounting["repair_logical_calls"]
        + accounting["retry_attempts"],
        "POST accounting differs",
    )
    a.require(
        sum(errors.values()) == counts["repairs"] + counts["parse_failures"],
        "Invalid-response accounting differs",
    )
    a.require(counts["nulls"] == summary["null_items"], "Null summary differs")
    for k, v in summary["completed_item_accounting"].items():
        a.require(v == accounting[k], "Stage accounting mismatch")
    a.require(counts["predictions"] == stage["items"], "Missing prediction audit")
    report = {
        "status": "schema_specific_stage_audited",
        "stage": stage["id"],
        "study": stage["study"],
        "mode": mode,
        "plan_sha256": plan_sha,
        "created_utc": a.utc(),
        "stage_complete_sha256": a.digest(target / "complete.json"),
        "attempt_count": len(attempts),
        "model_availability_checks": model_checks,
        "counts": dict(counts),
        "accounting": dict(accounting),
        "observed_usage": dict(usage),
        "http_statuses": dict(statuses),
        "reported_models": dict(models),
        "finish_reasons": dict(finishes),
        "by_population_condition": {k: dict(v) for k, v in per_group.items()},
        "schema_prediction_counts": dict(schemas),
        "validation_errors": dict(errors),
        "null_jobs": nulls,
        "retry_events": retries,
        "distinct_response_ids": len(ids),
        "raw_hash_bindings": len(hashes),
        "attempt_hashes": history,
        "vote_seconds_sum": sum(latencies),
        "wall_seconds_including_restarts": (end - start).total_seconds(),
        "source_hashes": {n: a.digest(b.ROOT / n) for n in b.SOURCE_FILES},
        "actual_audit_API_calls": 0,
        "gold_scoring": False,
        "limits": [
            "Reported model name is not proof of backend checkpoint identity.",
            "Completed invalid votes remain; no favorable resampling.",
            "Audit supports clean restarts; unresolved intents or terminal stops require review.",
        ],
    }

    audit_folder = target / f'audit_{__import__("time").time_ns()}'
    b.save(audit_folder / "report.json", report)
    b.save(audit_folder / "raw_hashes.json", hashes)
    b.save(audit_folder / "response_ids.json", ids)
    artifacts = {
        p.relative_to(target).as_posix(): a.digest(p) for p in audit_folder.iterdir()
    }
    review = {
        "status": "audited",
        "stage": stage["id"],
        "study": stage["study"],
        "mode": mode,
        "plan_sha256": plan_sha,
        "complete_sha256": a.digest(target / "complete.json"),
        "report": (audit_folder / "report.json").relative_to(target).as_posix(),
        "response_ids": (audit_folder / "response_ids.json")
        .relative_to(target)
        .as_posix(),
        "raw_hashes": (audit_folder / "raw_hashes.json").relative_to(target).as_posix(),
        "artifacts": artifacts,
    }
    b.save(target / "review.json", review)
    print(f"{stage['id']}: audit passed; next batch may proceed.", flush=True)
    return review


def finalize_queue(schedule, live_root, plan_sha, *, mode):
    live_root = a.storage_path(live_root)
    reports = []
    remote_ids = set()
    counts = Counter()
    accounting = Counter()
    usage = Counter()
    receipts = {}
    expected_items = defaultdict(set)
    payloads = defaultdict(set)
    expected_stages = defaultdict(set)
    for stage in schedule["stages"]:
        root = live_root / stage["study"]
        target = root / "stages" / stage["id"]
        review = b.verify_review(stage, root, plan_sha, mode)
        report = a.read(target / review["report"])
        ids = a.read(target / review["response_ids"])
        a.require(
            not remote_ids.intersection(ids),
            "Remote response ID reused across stages/studies",
        )
        remote_ids.update(ids)
        manifest = a.read(target / review["raw_hashes"])
        for number, (name, sha) in enumerate(manifest.items(), 1):
            a.require(
                a.digest(root / name) == sha, "Saved output changed after stage audit"
            )
            if name.startswith("payloads/"):
                payloads[stage["study"]].add(name.split("/", 1)[1])
            if number % 25000 == 0:
                print(
                    f"Final integrity check {stage['id']}: {number}/{len(manifest)} files",
                    flush=True,
                )
        for name, sha in report["attempt_hashes"].items():
            a.require(a.digest(root / name) == sha, "Saved attempt history changed")
        expected_items[stage["study"]].update(
            a.signature(b.identity(j, plan_sha)) for j in stage["jobs"]
        )
        expected_stages[stage["study"]].add(stage["id"])
        counts.update(report["counts"])
        accounting.update(report["accounting"])
        usage.update(report["observed_usage"])
        receipts[b.rel(target / "review.json")] = a.digest(target / "review.json")
        reports.append(
            {
                "study": stage["study"],
                "stage": stage["id"],
                "review": b.rel(target / "review.json"),
            }
        )
    for study, items in expected_items.items():
        root = live_root / study
        a.require(
            {p.name for p in (root / "items").iterdir() if p.is_dir()} == items,
            "Unexpected or missing live item directory",
        )
        a.require(
            {p.name for p in (root / "payloads").iterdir()} == payloads[study],
            "Unattributed live payload",
        )
        a.require(
            {p.name for p in (root / "stages").iterdir()} == expected_stages[study],
            "Unattributed stage",
        )
    result = {
        "status": "all_studies_executed_and_audited",
        "plan_sha256": plan_sha,
        "mode": mode,
        "counts": dict(counts),
        "accounting": dict(accounting),
        "observed_usage": dict(usage),
        "distinct_remote_ids": len(remote_ids),
        "stages": reports,
        "review_hashes": receipts,
        "actual_finalization_API_calls": 0,
        "gold_scoring": False,
    }
    b.save(live_root / "complete.json", result)
    return result

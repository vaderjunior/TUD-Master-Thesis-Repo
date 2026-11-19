import base64
from collections import Counter
from datetime import datetime
import hashlib
import json
import socket
from unittest.mock import patch
from analysis.common import batch as b

a = b.a
ROOT = b.ROOT
BASE = b.BASE
OUT = ROOT / "Thesis_Log/astra_result/evidence/core_api_live_preflight_20260915_v1.json"


def forbidden(*args, **kwargs):
    raise AssertionError("No network or live client construction in saved-pilot review")


def main():
    a.require(not OUT.exists(), "Preserve completed live review")
    plan = b.validate()
    schedule = a.read(BASE / "schedule.json")
    cat = b.catalog()
    ready = a.read(BASE / "ready.json")
    pilot = a.read(BASE / "live_preflight.json")
    a.require(
        ready["status"] == "live_pilot_audited_first_stage_ready",
        "Live audit prerequisite missing",
    )
    a.require(
        ready["plan_sha256"] == pilot["plan_sha256"] == a.digest(b.PLAN),
        "Pilot plan changed",
    )
    a.require(
        ready["live_preflight_sha256"] == a.digest(BASE / "live_preflight.json"),
        "Pilot receipt changed",
    )
    attempt = ROOT / pilot["attempt"]
    summary = a.read(attempt / "summary.json")
    for file, field in [
        ("summary.json", "summary_sha256"),
        ("roster_response.json", "roster_sha256"),
    ]:
        a.require(a.digest(attempt / file) == pilot[field], "Pilot attempt changed")
    roster = a.read(attempt / "roster_response.json")
    roster_raw = base64.b64decode(roster["body_base64"], validate=True)
    a.require(
        hashlib.sha256(roster_raw).hexdigest() == roster["body_sha256"],
        "Roster bytes changed",
    )
    roster_json = json.loads(roster_raw)
    a.require(
        roster["status"] == 200 and a.MODEL in {x["id"] for x in roster_json["data"]},
        "Requested model not in successful roster",
    )
    a.require(
        not summary["stopped"]
        and (not summary["errors"])
        and (summary["saved_items"] == 15),
        "Pilot did not complete cleanly",
    )
    root = a.storage_path(BASE / "live")
    jobs = schedule["pilot_jobs"]
    first = schedule["stages"][0]
    ids = lambda xs: {(x["key"], x["condition"], x["repeat"]) for x in xs}
    a.require(
        len(jobs) == 15 and ids(jobs) <= ids(first["jobs"]),
        "Pilot not included in first stage",
    )
    headers = Counter()
    models = Counter()
    finish = Counter()
    usage_fields = Counter()
    usage_sums = Counter()
    valid = Counter()
    conditions = Counter()
    normalisations = Counter()
    latencies = []
    hashes = {}
    for name, h in ready["pilot_item_receipt_hashes"].items():
        a.require(a.digest(ROOT / name) == h, "Item receipt changed")
        hashes[name] = h
    for job in jobs:
        d = b.item_dir(root, job)
        receipt = a.read(d / "complete.json")
        result = a.read(d / "result.json")
        expected_messages = cat[job["key"], job["condition"]]["messages"]
        expected_wire = {
            "model": a.MODEL,
            "temperature": 1.0,
            "max_tokens": 4096,
            "messages": [
                {"role": m["role"], "content": m["text"]} for m in expected_messages
            ],
        }
        a.require(
            result["execution_mode"] == "live" and result["active_model"] == a.MODEL,
            "Invalid live attribution",
        )
        a.require(
            result["n_valid"] == 3
            and result["parse_failures"] == result["repairs"] == 0,
            "Unexpected pilot parse/repair result",
        )
        valid[str(result["n_valid"])] += 1
        conditions[job["subset"] + "/" + job["condition"]] += 1
        normalisations.update(
            {k: result[k] for k in ("normalisations", "gate_normalised")}
        )
        for name, h in receipt["hashes"].items():
            p = root / name
            a.require(a.digest(p) == h, "Saved pilot artifact changed")
            hashes[b.rel(p)] = h
        for vote in range(3):
            folder = d / "calls" / f"vote_{vote}_turn_0/attempt_0"
            intent = a.read(folder / "intent.json")
            outcome = a.read(folder / "outcome.json")
            payload = root / "payloads" / f"{intent['body_sha256']}.json"
            a.require(
                a.read(payload) == expected_wire,
                "Sent payload differs from frozen prompt/settings",
            )
            a.require(
                a.digest(payload) == intent["body_sha256"],
                "Sent payload signature differs",
            )
            a.require(
                outcome["intent_sha256"] == a.signature(intent),
                "Response/intent binding differs",
            )
            raw = base64.b64decode(outcome["body_base64"], validate=True)
            a.require(
                hashlib.sha256(raw).hexdigest() == outcome["body_sha256"],
                "Raw response changed",
            )
            data = json.loads(raw)
            choice = data["choices"][0]
            a.require(
                outcome["status"] == 200
                and data.get("model") == a.MODEL
                and (choice.get("finish_reason") == "stop"),
                "Pilot response metadata differs",
            )
            headers[str(outcome["status"])] += 1
            models[data["model"]] += 1
            finish[choice["finish_reason"]] += 1
            latencies.append(outcome["elapsed_s"])
            usage = data.get("usage")
            if isinstance(usage, dict):
                for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                    value = usage.get(field)
                    if type(value) in (int, float) and a.finite_nonnegative(value):
                        usage_fields[field] += 1
                        usage_sums[field] += value
    a.require(
        len(list(root.rglob("intent.json")))
        == len(list(root.rglob("outcome.json")))
        == 45,
        "Extra/missing pilot attempts",
    )
    before = {
        b.rel(p): a.digest(p)
        for p in root.rglob("*")
        if p.is_file() and p.name != "item.lock"
    }
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ):
        restart = b.run_jobs(
            jobs,
            cat,
            root,
            a.storage_path(BASE / "live_review_v1/restart"),
            client_factory=forbidden,
            mode="live",
        )
    after = {
        b.rel(p): a.digest(p)
        for p in root.rglob("*")
        if p.is_file() and p.name != "item.lock"
    }
    a.require(before == after, "Live pilot changed during offline restart check")
    a.require(
        restart["submitted"] == 0
        and restart["saved_items"] == 15
        and (not restart["stopped"]),
        "Completed live items were resubmitted",
    )
    a.require(not (BASE / "stages").exists(), "Long stage unexpectedly started")
    start = a.read(attempt / "intent.json")["started_utc"]
    wall = (
        datetime.fromisoformat(summary["finished_utc"]) - datetime.fromisoformat(start)
    ).total_seconds()
    remaining_items = first["items"] - len(jobs)
    mean = sum(latencies) / len(latencies)
    for p in [
        b.PLAN,
        BASE / "ready.json",
        BASE / "live_preflight.json",
        BASE / "offline_check_complete.json",
        attempt / "intent.json",
        attempt / "summary.json",
        attempt / "roster_response.json",
        BASE / "live_review_v1/restart/summary.json",
        ROOT / "analysis/common/validate_pilot.py",
    ]:
        hashes[b.rel(p)] = a.digest(p)
    report = {
        "created_utc": a.utc(),
        "status": "live_pilot_independently_checked_first_stage_ready",
        "plan_sha256": a.digest(b.PLAN),
        "ready_sha256": a.digest(BASE / "ready.json"),
        "approval": "Ajay explicitly approved the planned input/KB-context transfer in the conversation after the recorded auto-review rejection. The approved pilot then ran through exec_command.",
        "pilot_items": 15,
        "unique_input_texts": 2,
        "saved_POSTs": 45,
        "ordinary_responses": 45,
        "repairs": 0,
        "retries": 0,
        "http_statuses": dict(headers),
        "reported_models": dict(models),
        "finish_reasons": dict(finish),
        "valid_vote_distribution": dict(valid),
        "condition_counts": dict(conditions),
        "normalisation_counts": dict(normalisations),
        "observed_usage_field_counts": dict(usage_fields),
        "observed_usage_sums": dict(usage_sums),
        "latency_seconds_sum": sum(latencies),
        "mean_response_seconds": mean,
        "pilot_wall_seconds_including_roster": wall,
        "offline_restart_new_POSTs": 0,
        "offline_restart_preserved_all_live_hashes": True,
        "verified_hashes": hashes,
        "first_stage": {
            "id": first["id"],
            "total_items": first["items"],
            "ordinary_responses": first["ordinary_responses"],
            "already_complete_pilot_items": 15,
            "remaining_items": remaining_items,
            "remaining_ordinary_responses": remaining_items * 3,
            "ideal_hours_at_pilot_mean_and_eight_workers": remaining_items
            * 3
            * mean
            / 8
            / 3600,
            "hours_extrapolating_total_pilot_wall_throughput": remaining_items
            / 15
            * wall
            / 3600,
        },
        "actual_gold_scoring": False,
        "long_stage_started": False,
        "review_actual_API_calls": 0,
        "limits": [
            "Two texts across all primary conditions are a workflow check, not an accuracy sample.",
            "Throughput estimates from a tiny pilot do not guarantee duration for different texts, prompt lengths or server load.",
            "Reported model name does not establish the underlying server checkpoint. Full-stage response audit remains required.",
            "Usage is reported only when present; absence is not zero usage. Pilot predictions count once in stage 01.",
        ],
    }
    a.save_new(OUT, report)
    a.save_new(
        BASE / "live_review_complete_v1.json",
        {
            "status": report["status"],
            "evidence": b.rel(OUT),
            "evidence_sha256": a.digest(OUT),
            "plan_sha256": a.digest(b.PLAN),
            "ready_sha256": a.digest(BASE / "ready.json"),
        },
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("verified_hashes", "condition_counts")
            },
            indent=2,
        )
    )
    print("Evidence SHA256: " + a.digest(OUT))


if __name__ == "__main__":
    main()

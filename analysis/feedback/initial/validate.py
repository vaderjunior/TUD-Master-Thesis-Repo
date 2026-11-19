from collections import Counter, defaultdict
import json
import os
import socket
import threading
import time
import uuid
from unittest.mock import patch

from analysis.feedback.initial import run as b
from analysis.feedback.initial import audit as auditor

a = b.a


def check_schedule(schedule, catalogs):

    seen = set()
    assignments = {}
    counts = Counter()
    schema_counts = Counter()
    populations = Counter()
    studies = Counter()
    for stage in schedule["stages"]:
        study = stage["study"]
        info = b.STUDIES[study]
        cat = catalogs[study]
        a.require(stage["id"] not in studies, "Repeated stage ID")
        studies[stage["id"]] += 1
        keys = stage["input_keys"]
        a.require(
            len(keys) == len(set(keys)) == info["inputs"] // info["batches"],
            "Batch roster size",
        )
        a.require(
            stage["items"] == len(stage["jobs"]) == len(keys) * info["conditions"] * 3,
            "Batch item count",
        )
        a.require(stage["ordinary_calls"] == stage["items"] * 3, "Batch request count")
        per_input = Counter()
        for j in stage["jobs"]:
            k = (study, j["key"], j["condition"], j["repeat"])
            a.require(
                k not in seen and j["study"] == study and j["repeat"] in (0, 1, 2),
                "Wrong/repeated execution identity",
            )
            seen.add(k)
            a.require(j["key"] in keys, "Job outside batch roster")
            per_input[j["key"]] += 1
            owner = (study, j["key"])
            a.require(
                assignments.setdefault(owner, stage["id"]) == stage["id"],
                "Input split across batches",
            )
            entry = cat[j["key"], j["condition"]]
            for field in [
                "messages_sha256",
                "schema_path",
                "schema_sha256",
                "population",
            ]:
                a.require(entry[field] == j[field], "Scheduled prompt/schema mismatch")
            a.require(
                a.signature(entry["messages"]) == j["messages_sha256"],
                "Wire input mismatch",
            )
            wire = a.body(entry["messages"])
            a.require(
                [m["content"] for m in wire["messages"]]
                == [m["text"] for m in entry["messages"]],
                "Wire conversion changed text",
            )
            counts[study] += 1
            schema_counts[j["schema_path"]] += 1
            populations[study + "/" + j["population"]] += 1
        a.require(
            set(per_input) == set(keys)
            and set(per_input.values()) == {info["conditions"] * 3},
            "Unbalanced condition/repeat matrix",
        )
    expected = {
        (study, key, cond, repeat)
        for study, cat in catalogs.items()
        for key, cond in cat
        for repeat in range(3)
    }
    a.require(
        seen == expected and len(seen) == 73359 and len(studies) == 8,
        "Missing/unplanned scientific comparison",
    )
    a.require(
        counts == Counter({k: v["predictions"] for k, v in b.STUDIES.items()}),
        "Study totals",
    )
    return {
        "identities": len(seen),
        "stages": len(studies),
        "predictions_by_study": dict(counts),
        "predictions_by_schema": dict(schema_counts),
        "predictions_by_population": dict(populations),
        "ordinary_calls": len(seen) * 3,
        "unique_study_inputs": len(assignments),
    }


class SyntheticTransport:
    """No socket or credentials; fake answers must never be used as findings."""

    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()
        self.prefix = uuid.uuid4().hex

    def __call__(self, payload):
        with self.lock:
            self.count += 1
            number = self.count
        content = {
            "reasoning": "OFFLINE EXECUTION REHEARSAL ONLY",
            "hate": False,
            "target_group": [],
            "hate_type": [],
            "severity": None,
            "legal": [],
        }
        reply = {
            "id": f"offline-{self.prefix}-{number}",
            "model": a.MODEL,
            "choices": [
                {"message": {"content": json.dumps(content)}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return a.WireReply(
            200, json.dumps(reply).encode(), {"Content-Type": "application/json"}
        )


def rehearse(schedule, catalogs, plan_sha):
    root = b.BASE / "offline_check" / f"rehearsal_{time.time_ns()}"
    selected = []
    for stage in schedule["stages"]:

        groups = defaultdict(list)
        for key in stage["input_keys"]:
            entry = next(
                v for (k, c), v in catalogs[stage["study"]].items() if k == key
            )
            groups[entry["population"]].append(key)
        a.require(len(groups) == 2, "Rehearsal requires both evaluation populations")
        keys = {sorted(v)[0] for v in groups.values()}
        jobs = [j for j in stage["jobs"] if j["key"] in keys]
        selected.append(
            {
                **stage,
                "input_keys": sorted(keys),
                "jobs": jobs,
                "items": len(jobs),
                "ordinary_calls": len(jobs) * 3,
            }
        )
    transport = SyntheticTransport()
    reduced = {**schedule, "stages": selected}

    def run(st, factory):
        b.execute_stage(
            st,
            catalogs[st["study"]],
            root / st["study"],
            plan_sha,
            mode="mock",
            client_factory=factory,
            sleep=lambda _: None,
        )

    def audit(st):
        auditor.audit_stage(
            st, catalogs[st["study"]], root / st["study"], plan_sha, mode="mock"
        )

    def reviewed(st):
        return (root / st["study"] / "stages" / st["id"] / "review.json").exists()

    def verify(st):
        return b.verify_review(st, root / st["study"], plan_sha, "mock")

    b.run_sequence(
        selected, lambda st: run(st, lambda: transport), audit, reviewed, verify
    )
    first = auditor.finalize_queue(reduced, root, plan_sha, mode="mock")
    before = transport.count
    b.run_sequence(selected, lambda st: run(st, b.forbidden), audit, reviewed, verify)
    second = auditor.finalize_queue(reduced, root, plan_sha, mode="mock")
    a.require(
        first == second
        and transport.count == before == 2934
        and first["counts"]["predictions"] == 978,
        "Restart/rehearsal budget differs",
    )
    return {
        "root": b.rel(root),
        "completion_sha256": a.digest(root / "complete.json"),
        "mock_predictions": 978,
        "simulated_ordinary_replies": before,
        "new_replies_on_completed_restart": 0,
        "actual_API_calls": 0,
        "all_eight_batches": True,
        "both_populations_per_batch": True,
        "all_condition_repeat_pairs_for_selected_inputs": True,
        "rehearsal_status": "mechanical only; synthetic classifications and usage are not research evidence",
    }


def main():
    os.chdir(b.ROOT)
    with patch.object(socket.socket, "connect", b.forbidden), patch.object(
        socket, "create_connection", b.forbidden
    ), patch.object(b.core, "live_client", b.forbidden):
        plan = b.validate()
        plan_sha = a.digest(b.PLAN)
        a.require(
            not (b.BASE / "live").exists(),
            "Offline readiness cannot rewrite an already-started live queue",
        )
        if (b.BASE / "ready.json").exists():
            b.validate(ready=True)
            print(
                "Existing readiness receipt verified; no simulation repeated.",
                flush=True,
            )
            return
        cats = {k: b.load_catalog(k) for k in b.STUDIES}
        schedule = a.read(b.BASE / "schedule.json")
        matrix = check_schedule(schedule, cats)
        print(
            "All 73,359 planned identities independently checked. Rehearsing all eight batches with fake responses.",
            flush=True,
        )
        rehearsal = rehearse(schedule, cats, plan_sha)
        b.validate()
        a.require(
            a.digest(b.PLAN) == plan_sha and not (b.BASE / "live").exists(),
            "Plan/live boundary changed",
        )
        path = b.BASE / "development" / f"readiness_evidence_{time.time_ns()}.json"
        evidence = {
            "status": "full_schedule_and_mock_queue_verified",
            "created_utc": a.utc(),
            "plan_sha256": plan_sha,
            "matrix": matrix,
            "rehearsal": rehearsal,
            "tests": b.rel(b.test_receipt()),
            "tests_sha256": a.digest(b.test_receipt()),
            "input_bindings_verified": len(plan["input_hashes"]),
            "source_hashes": plan["source_hashes"],
            "environment": b.core.environment(),
            "actual_API_calls": 0,
            "gold_scoring": False,
        }
        b.save(path, evidence)
        b.save(
            b.BASE / "ready.json",
            {
                "status": "offline_verified_ready_for_user",
                "created_utc": a.utc(),
                "plan_sha256": plan_sha,
                "evidence": b.rel(path),
                "evidence_sha256": a.digest(path),
                "actual_API_calls": 0,
            },
        )
        b.validate(ready=True)
        print(
            "COMBINED LAUNCHER READY FOR USER. All tests/rehearsals were offline. No Qwen call or CUDA use.",
            flush=True,
        )


if __name__ == "__main__":
    main()

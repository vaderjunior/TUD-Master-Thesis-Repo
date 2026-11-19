from collections import Counter, defaultdict
import json
import os
import socket
import threading
import time
import uuid
from unittest.mock import patch
from analysis.feedback.supplements import run as b
from analysis.feedback.supplements import audit as audit

a = b.a


def check_schedule(schedule, catalogs):
    seen = set()
    owners = {}
    counts = Counter()
    per_stage = {}
    for stage in schedule["stages"]:
        study = stage["study"]
        info = b.STUDIES[study]
        cat = catalogs[study]
        a.require(stage["id"] not in per_stage, "Duplicate stage")
        keys = set(stage["input_keys"])
        a.require(
            len(keys) == len(stage["input_keys"]) == info["inputs"] // info["batches"],
            "Stage input count",
        )
        a.require(
            stage["items"] == len(stage["jobs"]) == len(keys) * info["conditions"] * 3
            and stage["ordinary_calls"] == len(stage["jobs"]) * 3,
            "Stage budget",
        )
        local = Counter()
        for j in stage["jobs"]:
            ident = study, j["key"], j["condition"], j["repeat"]
            a.require(
                ident not in seen and j["study"] == study and j["repeat"] in (0, 1, 2),
                "Duplicate/wrong job identity",
            )
            seen.add(ident)
            a.require(j["key"] in keys, "Job outside stage roster")
            a.require(
                owners.setdefault((study, j["key"]), stage["id"]) == stage["id"],
                "Comparisons split across batches",
            )
            entry = cat[j["key"], j["condition"]]
            for field in [
                "messages_sha256",
                "schema_path",
                "schema_sha256",
                "population",
            ]:
                a.require(j[field] == entry[field], "Wire identity mismatch")
            a.require(
                a.signature(entry["messages"]) == j["messages_sha256"],
                "Changed message",
            )
            wire = a.body(entry["messages"])
            a.require(
                [m["content"] for m in wire["messages"]]
                == [m["text"] for m in entry["messages"]],
                "Wire conversion drift",
            )
            local[j["key"]] += 1
            counts[study] += 1
        a.require(
            set(local) == keys and set(local.values()) == {info["conditions"] * 3},
            "Incomplete local matrix",
        )
        per_stage[stage["id"]] = len(stage["jobs"])
    expected = {
        (study, key, cond, rep)
        for study, cat in catalogs.items()
        for key, cond in cat
        for rep in range(3)
    }
    a.require(
        seen == expected and len(seen) == 22536 and len(per_stage) == 4,
        "Queue matrix differs",
    )
    a.require(
        counts == Counter({k: v["predictions"] for k, v in b.STUDIES.items()}),
        "Study counts differ",
    )
    return {
        "identities": len(seen),
        "ordinary_calls": len(seen) * 3,
        "per_study": dict(counts),
        "stages": per_stage,
    }


class SyntheticTransport:
    def __init__(self):
        self.count = 0
        self.lock = threading.Lock()
        self.prefix = uuid.uuid4().hex

    def __call__(self, payload):
        with self.lock:
            self.count += 1
            number = self.count
        content = {
            "reasoning": "OFFLINE REHEARSAL ONLY",
            "hate": False,
            "hate_type": [],
            "target_group": [],
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
                v for (k, _), v in catalogs[stage["study"]].items() if k == key
            )
            groups[entry["population"]].append(key)
        a.require(len(groups) == 2, "Two populations required per batch")
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

    def run(stage, factory):
        b.execute_stage(
            stage,
            catalogs[stage["study"]],
            root / stage["study"],
            plan_sha,
            mode="mock",
            client_factory=factory,
            sleep=lambda _: None,
        )

    def audit_one(stage):
        audit.audit_stage(
            stage,
            catalogs[stage["study"]],
            root / stage["study"],
            plan_sha,
            mode="mock",
        )

    def reviewed(stage):
        return (root / stage["study"] / "stages" / stage["id"] / "review.json").exists()

    def verify(stage):
        return b.verify_review(stage, root / stage["study"], plan_sha, "mock")

    b.run_sequence(
        selected,
        lambda stage: run(stage, lambda: transport),
        audit_one,
        reviewed,
        verify,
    )
    first = audit.finalize_queue(reduced, root, plan_sha, mode="mock")
    before = transport.count
    b.run_sequence(
        selected, lambda stage: run(stage, b.forbidden), audit_one, reviewed, verify
    )
    second = audit.finalize_queue(reduced, root, plan_sha, mode="mock")
    a.require(
        first == second
        and transport.count == before == 450
        and first["counts"]["predictions"] == 150,
        "Rehearsal/restart mismatch",
    )
    return {
        "root": b.rel(root),
        "completion_sha256": a.digest(root / "complete.json"),
        "mock_predictions": 150,
        "mock_replies": 450,
        "new_replies_on_restart": 0,
        "actual_API_calls": 0,
        "all_batches_and_conditions": True,
        "synthetic_responses_are_research_evidence": False,
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
            "Offline readiness cannot rewrite live execution",
        )
        if (b.BASE / "ready.json").exists():
            b.validate(ready=True)
            print("Existing readiness verified.")
            return
        cats = {k: b.load_catalog(k) for k in b.STUDIES}
        schedule = a.read(b.BASE / "schedule.json")
        matrix = check_schedule(schedule, cats)
        print(
            "All 22,536 scheduled predictions verified. Rehearsing with synthetic replies only.",
            flush=True,
        )
        rehearsal = rehearse(schedule, cats, plan_sha)
        b.validate()
        evidence_path = b.BASE / "development" / f"readiness_{time.time_ns()}.json"
        evidence = {
            "status": "all_identities_and_restart_audit_rehearsal_passed",
            "plan_sha256": plan_sha,
            "matrix": matrix,
            "rehearsal": rehearsal,
            "tests": b.rel(b.test_receipt()),
            "tests_sha256": a.digest(b.test_receipt()),
            "source_hashes": plan["source_hashes"],
            "environment": b.core.environment(),
            "actual_API_calls": 0,
        }
        b.save(evidence_path, evidence)
        b.save(
            b.BASE / "ready.json",
            {
                "status": "offline_verified_ready_for_user",
                "plan_sha256": plan_sha,
                "evidence": b.rel(evidence_path),
                "evidence_sha256": a.digest(evidence_path),
                "actual_API_calls": 0,
            },
        )
        b.validate(ready=True)
        print(
            "SQ3 ADDITIONS LAUNCHER READY. No live model or CUDA calls occurred.",
            flush=True,
        )


if __name__ == "__main__":
    main()

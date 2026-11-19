import argparse
from collections import Counter
import io
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch
from analysis.feedback.source import updates as b

a = b.a
OUT = (
    b.ROOT
    / "Thesis_Log/astra_result/evidence/sq3_feedback_updates_readiness_20260929_v1.json"
)


def forbidden(*args, **kwargs):
    raise AssertionError("No network or credentials in offline update verification")


def tests():
    a.require(
        not b.PLAN.exists() and not (b.BASE / "development/tests_passed.json").exists(),
        "Preserve passed tests and frozen execution",
    )
    attempt = b.BASE / "development" / ("test_attempt_" + str(time.time_ns()))
    hashes = {n: a.digest(b.ROOT / n) for n in b.SOURCE_FILES}
    b.save(
        attempt / "sources.json",
        {
            n: (b.ROOT / n).read_text(encoding="utf8")
            for n in b.SOURCE_FILES
            if "feedback_updates" in n
        },
    )
    suite = unittest.defaultTestLoader.loadTestsFromNames(
        [
            "tests.feedback.source.test_updates",
            "tests.feedback.source.test_update_audit",
        ]
    )
    stream = io.StringIO()
    tick = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    receipt = {
        "created_utc": a.utc(),
        "passed": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "output": stream.getvalue(),
        "elapsed_s": time.perf_counter() - tick,
        "source_hashes": hashes,
        "environment": b.baseline.environment(),
        "actual_API_calls": 0,
        "attempt": b.rel(attempt),
    }
    b.save(attempt / "tests.json", receipt)
    print(stream.getvalue(), flush=True)
    a.require(
        result.wasSuccessful(),
        "Preserve failed attempt/source; do not freeze execution",
    )
    for n, h in hashes.items():
        a.require(a.digest(b.ROOT / n) == h, "Tested source changed")
    b.save(b.BASE / "development/tests_passed.json", receipt)


def check():
    a.require(
        not OUT.exists() and not (b.BASE / "offline_check_complete.json").exists(),
        "Preserve completed readiness",
    )
    plan = b.validate()
    plan_sha = a.digest(b.PLAN)
    schedule = a.read(b.BASE / "schedule.json")
    cat = b.frozen_catalog(b.ROOT / plan["prompts"])
    audit_tests_path = b.BASE / "audit_development/stage_check_v1/tests_passed.json"
    audit_tests = a.read(audit_tests_path)
    a.require(
        audit_tests["passed"] and audit_tests["environment"] == b.core.environment(),
        "Passing auditor tests required",
    )
    for n, h in audit_tests["source_hashes"].items():
        a.require(a.digest(b.ROOT / n) == h, "Auditor changed after tests")
    science = a.read(b.SCIENCE)
    roster = {
        r["key"]: r for r in b.core.rows(b.baseline.preparation.OUT / "roster.jsonl")
    }
    expected_keys = {"en/" + i for i in science["rosters"]["evaluation_phase"]}
    all_jobs = [j for s in schedule["stages"] for j in s["jobs"]]
    actual = {(j["key"], j["condition"], j["repeat"]) for j in all_jobs}
    expected = {
        (k, c, r) for k in expected_keys for c in b.CONDITIONS for r in range(3)
    }
    a.require(
        actual == expected and len(all_jobs) == len(actual) == 33552,
        "Independent complete update roster differs",
    )
    a.require(
        [s["id"] for s in schedule["stages"]] == ["stage_02", "stage_03", "stage_04"]
        and [s["items"] for s in schedule["stages"]] == [16776, 8388, 8388],
        "Batch matrix differs",
    )
    identities = set()
    baseline_jobs = a.read(b.baseline.BASE / "schedule.json")["stages"][0]["jobs"]
    baseline_ids = {
        a.signature(b.baseline.identity(j, a.digest(b.baseline.PLAN)))
        for j in baseline_jobs
    }
    per_setting = Counter()
    for stage in schedule["stages"]:
        a.require(
            stage["ordinary_responses"] == len(stage["jobs"]) * 3
            and stage["max_logical_calls"] == len(stage["jobs"]) * 9
            and stage["max_physical_POSTs"] == len(stage["jobs"]) * 45,
            "Batch hard caps differ",
        )
        conditions = [c for c in b.CONDITIONS if int(c[-2:]) in stage["levels"]]
        expected_order = [
            (
                key,
                conditions[(ordinal + (ordinal + block) % 3 + pos) % len(conditions)],
                (ordinal + block) % 3,
            )
            for ordinal, key in enumerate(sorted(expected_keys))
            for block in range(3)
            for pos in range(len(conditions))
        ]
        a.require(
            [(j["key"], j["condition"], j["repeat"]) for j in stage["jobs"]]
            == expected_order,
            "Independent rotating order differs",
        )
        for job in stage["jobs"]:
            entry = cat[job["key"], job["condition"]]
            source = roster[job["key"]]
            a.require(
                job["subset"]
                == ("candidate" if source["correction_eligible"] else source["role"])
                and job["strategy"] == "hybrid"
                and job["messages_sha256"] == entry["messages_sha256"],
                "Job scope/prompt differs",
            )
            for field in ("arm", "seed", "budget_level", "pool_size"):
                a.require(job[field] == entry[field], "Setting metadata differs")
            ident = a.signature(b.identity(job, plan_sha))
            a.require(
                ident not in identities and ident not in baseline_ids,
                "Reused response identity",
            )
            identities.add(ident)
            per_setting[job["condition"], job["repeat"]] += 1
    a.require(
        len(per_setting) == 72 and set(per_setting.values()) == {466},
        "Missing draw/budget/repeat input",
    )
    for entry in cat.values():
        wire = a.body(entry["messages"])
        a.require(
            wire
            == {
                "model": a.MODEL,
                "temperature": 1.0,
                "max_tokens": 4096,
                "messages": [
                    {"role": m["role"], "content": m["text"]} for m in entry["messages"]
                ],
            },
            "Wire payload differs",
        )
    picked = {
        next(
            k
            for k in sorted(expected_keys)
            if cat[k, b.CONDITIONS[0]]["subset"] == role
        )
        for role in b.ENDPOINT_ORDER
    }
    mini = []
    for stage in schedule["stages"]:
        jobs = [j for j in stage["jobs"] if j["key"] in picked]
        mini.append(
            {
                **stage,
                "jobs": jobs,
                "items": len(jobs),
                "distinct_inputs": 3,
                "ordinary_responses": len(jobs) * 3,
            }
        )
    a.require([s["items"] for s in mini] == [108, 54, 54], "Miniature coverage differs")
    attempt = a.storage_path(
        b.BASE / "offline_check" / ("combined_" + str(time.time_ns()))
    )
    root = attempt / "mock"
    counts = [0]
    mutex = threading.Lock()
    audit_calls = []
    stages = {s["id"]: s for s in mini}
    value = json.dumps(
        {
            "reasoning": "OFFLINE FIXTURE ONLY",
            "hate": False,
            "target_group": [],
            "hate_type": [],
            "severity": None,
            "legal": [],
        }
    )

    def factory():
        def send(payload):
            with mutex:
                counts[0] += 1
                n = counts[0]
            return a.WireReply(
                200,
                json.dumps(
                    {
                        "id": f"mock-{n}",
                        "model": a.MODEL,
                        "choices": [
                            {"message": {"content": value}, "finish_reason": "stop"}
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                    }
                ).encode(),
                {},
            )

        return send

    def run_one(stage_id):
        b.execute_stage(
            stages[stage_id], cat, root, plan_sha, client_factory=factory, mode="mock"
        )

    def audit_one(stage_id):
        for job in stages[stage_id]["jobs"]:
            a.audit_item(
                root,
                b.identity(job, plan_sha),
                cat[job["key"], job["condition"]]["messages"],
                execution_mode="mock",
            )
        b.save(
            attempt / "reviews" / f"{stage_id}.json",
            {
                "stage": stage_id,
                "complete_sha256": a.digest(
                    root / "stages" / stage_id / "complete.json"
                ),
            },
        )
        audit_calls.append(stage_id)

    def exists(stage_id):
        return (attempt / "reviews" / f"{stage_id}.json").exists()

    def verify(stage_id):
        reviewed = a.read(attempt / "reviews" / f"{stage_id}.json")
        a.require(
            reviewed
            == {
                "stage": stage_id,
                "complete_sha256": a.digest(
                    root / "stages" / stage_id / "complete.json"
                ),
            },
            "Mock audit receipt changed",
        )

    b.run_sequence(list(stages), run_one, audit_one, exists, verify)
    a.require(
        counts[0] == 648 and audit_calls == list(stages),
        "Whole three-batch mock chain differs",
    )
    print(
        "Three mock batches passed. Restarting the full chain with new clients/audits forbidden.",
        flush=True,
    )

    def completed(stage_id):
        b.execute_stage(
            stages[stage_id], cat, root, plan_sha, client_factory=forbidden, mode="mock"
        )

    b.run_sequence(list(stages), completed, forbidden, exists, verify)
    a.require(
        counts[0] == 648 and not (b.BASE / "live").exists(),
        "Restart sent calls or created live outputs",
    )
    inputs = {
        **plan["input_hashes"],
        **plan["source_hashes"],
        **plan["generated_hashes"],
        b.rel(b.PLAN): plan_sha,
        b.rel(audit_tests_path): a.digest(audit_tests_path),
    }
    for n, h in inputs.items():
        a.require(a.digest(b.ROOT / n) == h, "Source changed during rehearsal")
    outputs = {
        b.rel(p): a.digest(p)
        for p in a.storage_path(b.BASE / "offline_check").rglob("*")
        if p.is_file() and p.name not in ("item.lock", "launcher.lock")
    }
    report = {
        "created_utc": a.utc(),
        "status": "offline_checked_ready_for_user_updates",
        "plan_sha256": plan_sha,
        "programme": plan["programme"],
        "stages": [
            {k: v for k, v in s.items() if k != "jobs"} for s in schedule["stages"]
        ],
        "tests_run": a.read(b.BASE / "development/tests_passed.json")["tests_run"],
        "included_auditor_tests": audit_tests["tests_run"],
        "checked_payloads": len(cat),
        "checked_prediction_identities": len(all_jobs),
        "setting_repeats": 72,
        "inputs_per_setting_repeat": 466,
        "shared_identity_with_baseline": 0,
        "mock_texts": sorted(picked),
        "mock_predictions": 216,
        "mock_responses": 648,
        "mock_batches": 3,
        "mock_audits": audit_calls,
        "completed_chain_restart_new_calls": 0,
        "actual_API_calls": 0,
        "credentials_loaded": False,
        "CUDA_or_training": False,
        "verified_inputs": inputs,
        "verified_mock_outputs": outputs,
        "readiness_scope": "All33552 actual job identities and11184 payloads checked. Representative miniature execution covers every condition/repeat and three roles through all three internal batches; not a33552-prediction mock.",
        "next": "Ajay runs the combined update launcher once. It audits each batch and advances automatically only when all checks pass.",
    }
    a.save_new(OUT, report)
    b.save(
        b.BASE / "offline_check_complete.json",
        {
            "status": report["status"],
            "plan_sha256": plan_sha,
            "evidence": b.rel(OUT),
            "evidence_sha256": a.digest(OUT),
        },
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "tests_run",
                    "checked_payloads",
                    "checked_prediction_identities",
                    "mock_predictions",
                    "mock_responses",
                    "completed_chain_restart_new_calls",
                    "actual_API_calls",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    print("Evidence SHA256: " + a.digest(OUT), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tests", "check"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(b.core, "live_client", forbidden):
        tests() if args.mode == "tests" else check()

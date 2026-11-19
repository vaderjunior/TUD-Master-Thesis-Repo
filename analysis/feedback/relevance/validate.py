import argparse
from collections import Counter
import io
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch
from analysis.feedback.relevance import run as b

a = b.a
OUT = (
    b.ROOT / "Thesis_Log/astra_result/evidence/sq3_relevance_readiness_20260930_v1.json"
)


def forbidden(*args, **kwargs):
    raise AssertionError("No network or credentials in offline relevance checks")


def tests():
    a.require(
        not b.PLAN.exists() and not (b.BASE / "development/tests_passed.json").exists(),
        "Preserve existing tests and plan",
    )
    attempt = b.BASE / "development" / ("test_attempt_" + str(time.time_ns()))
    hashes = {n: a.digest(b.ROOT / n) for n in b.SOURCE_FILES}
    b.save(
        attempt / "sources.json",
        {
            n: (b.ROOT / n).read_text(encoding="utf8")
            for n in b.SOURCE_FILES
            if "sq3_relevance" in n
        },
    )
    suite = unittest.defaultTestLoader.loadTestsFromNames(
        ["tests.feedback.relevance.test_run", "tests.feedback.relevance.test_audit"]
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
    a.require(result.wasSuccessful(), "Tests failed; sources and attempt preserved")
    a.require(
        all(a.digest(b.ROOT / n) == h for n, h in hashes.items()),
        "Tested source changed",
    )
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
    science = a.read(b.SCIENCE)
    audit_path = b.BASE / "audit_development/stage_check_v1/tests_passed.json"
    audit_tests = a.read(audit_path)
    a.require(
        audit_tests["passed"] and audit_tests["environment"] == b.core.environment(),
        "Passing auditor tests required",
    )
    a.require(
        all(a.digest(b.ROOT / n) == h for n, h in audit_tests["source_hashes"].items()),
        "Auditor source changed",
    )
    jobs = [j for stage in schedule["stages"] for j in stage["jobs"]]
    expected = {
        (f"en/{i}", c, r)
        for i in science["rosters"]["evaluation"]
        for c in b.CONDITIONS
        for r in range(3)
    }
    a.require(
        {(j["key"], j["condition"], j["repeat"]) for j in jobs} == expected
        and len(jobs) == 16800,
        "Complete response matrix differs",
    )
    a.require(
        sum(j["condition"] == "reference" for j in jobs) == 1050,
        "Reference artificially repeated by pool",
    )
    counter = Counter((j["condition"], j["repeat"]) for j in jobs)
    a.require(
        len(counter) == 48 and set(counter.values()) == {350},
        "Incomplete setting/repeat roster",
    )
    identities = {a.signature(b.identity(j, plan_sha)) for j in jobs}
    a.require(len(identities) == 16800, "Duplicate response identity")
    old_plan = b.preparation.h.OLD / "update_api_v1/plan.json"
    a.require(plan_sha != a.digest(old_plan), "Old experiment identity reused")
    subsets = {
        r["key"]: r["role"]
        for r in b.core.rows(b.preparation.h.OLD / "feasibility_v1/roster.jsonl")
    }
    for stage_index, stage in enumerate(schedule["stages"]):
        ids = science["batching"]["input_batches"][stage_index]
        order = [
            (
                "en/" + iid,
                b.CONDITIONS[
                    (ordinal + (ordinal + block) % 3 + position) % len(b.CONDITIONS)
                ],
                (ordinal + block) % 3,
            )
            for ordinal, iid in enumerate(ids)
            for block in range(3)
            for position in range(len(b.CONDITIONS))
        ]
        a.require(
            [(j["key"], j["condition"], j["repeat"]) for j in stage["jobs"]] == order,
            "Independent order differs",
        )
        for job in stage["jobs"]:
            entry = cat[job["key"], job["condition"]]
            a.require(
                job["subset"] == subsets[job["key"]]
                and job["messages_sha256"] == entry["messages_sha256"],
                "Scope/payload differs",
            )
            a.require(
                job["policy"] == entry["policy"]
                and job["pool_seed"] == entry["pool_seed"],
                "Condition metadata differs",
            )
    for entry in cat.values():
        a.require(
            a.body(entry["messages"])
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
    mini = []
    picked = []
    for stage in schedule["stages"]:
        selected = {
            next(j["key"] for j in stage["jobs"] if j["subset"] == role)
            for role in b.ENDPOINT_ORDER
        }
        small = [j for j in stage["jobs"] if j["key"] in selected]
        mini.append(
            {
                **stage,
                "jobs": small,
                "items": len(small),
                "distinct_inputs": len(selected),
                "ordinary_responses": len(small) * 3,
            }
        )
        picked.extend(sorted(selected))
    a.require(
        [s["items"] for s in mini] == [96, 96, 96] and len(set(picked)) == 6,
        "Mock role/setting coverage differs",
    )
    attempt = a.storage_path(
        b.BASE / "offline_check" / ("combined_" + str(time.time_ns()))
    )
    root = attempt / "mock"
    counts = [0]
    mutex = threading.Lock()
    audits = []
    stages = {s["id"]: s for s in mini}
    output = json.dumps(
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
                        "id": f"relevance-mock-{n}",
                        "model": a.MODEL,
                        "choices": [
                            {"message": {"content": output}, "finish_reason": "stop"}
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

    def run_one(stage):
        b.execute_stage(
            stages[stage], cat, root, plan_sha, client_factory=factory, mode="mock"
        )

    def audit_one(stage):
        for job in stages[stage]["jobs"]:
            a.audit_item(
                root,
                b.identity(job, plan_sha),
                cat[job["key"], job["condition"]]["messages"],
                execution_mode="mock",
            )
        b.save(
            attempt / "reviews" / f"{stage}.json",
            {
                "stage": stage,
                "complete_sha256": a.digest(root / "stages" / stage / "complete.json"),
            },
        )
        audits.append(stage)

    def exists(stage):
        return (attempt / "reviews" / f"{stage}.json").exists()

    def verify(stage):
        a.require(
            a.read(attempt / "reviews" / f"{stage}.json")
            == {
                "stage": stage,
                "complete_sha256": a.digest(root / "stages" / stage / "complete.json"),
            },
            "Mock review altered",
        )

    b.run_sequence(list(stages), run_one, audit_one, exists, verify)
    a.require(counts[0] == 864 and audits == list(stages), "Mock chain count differs")
    print(
        "Mock execution passed. Checking complete-chain restart with all new clients/audits forbidden.",
        flush=True,
    )

    def completed(stage):
        b.execute_stage(
            stages[stage], cat, root, plan_sha, client_factory=forbidden, mode="mock"
        )

    b.run_sequence(list(stages), completed, forbidden, exists, verify)
    a.require(
        counts[0] == 864 and not (b.BASE / "live").exists(),
        "Restart sent calls or created live data",
    )
    inputs = {
        **plan["input_hashes"],
        **plan["source_hashes"],
        **plan["generated_hashes"],
        b.rel(b.PLAN): plan_sha,
        b.rel(audit_path): a.digest(audit_path),
    }
    a.require(
        all(a.digest(b.ROOT / n) == h for n, h in inputs.items()),
        "Source changed in rehearsal",
    )
    outputs = {
        b.rel(p): a.digest(p)
        for p in a.storage_path(b.BASE / "offline_check").rglob("*")
        if p.is_file() and p.name not in ("item.lock", "launcher.lock")
    }
    report = {
        "created_utc": a.utc(),
        "status": "offline_checked_ready_for_user_relevance",
        "plan_sha256": plan_sha,
        "programme": plan["programme"],
        "tests_run": a.read(b.BASE / "development/tests_passed.json")["tests_run"],
        "included_auditor_tests": audit_tests["tests_run"],
        "checked_payloads": len(cat),
        "checked_prediction_identities": len(jobs),
        "setting_repeats": 48,
        "inputs_per_setting_repeat": 350,
        "mock_texts": picked,
        "mock_predictions": 288,
        "mock_responses": 864,
        "mock_batches": 3,
        "mock_audits": audits,
        "completed_chain_restart_new_calls": 0,
        "actual_API_calls": 0,
        "credentials_loaded": False,
        "CUDA_or_training": False,
        "verified_inputs": inputs,
        "verified_mock_outputs": outputs,
        "readiness_scope": "All5600 payloads/16800 real job identities checked; representative288-prediction mock covers every condition/repeat and both roles in all3 batches.",
        "next": "Ajay runs the combined relevance command once. All three successful batches audit and advance automatically.",
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
                for k in [
                    "status",
                    "tests_run",
                    "checked_payloads",
                    "checked_prediction_identities",
                    "mock_predictions",
                    "mock_responses",
                    "completed_chain_restart_new_calls",
                    "actual_API_calls",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tests", "check"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(b.core, "live_client", forbidden):
        tests() if args.mode == "tests" else check()

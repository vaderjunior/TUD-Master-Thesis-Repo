import argparse
from collections import Counter, defaultdict
import io
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch
from analysis.knowledge.sampled_banks import run as b

a = b.a
OUT = (
    b.ROOT
    / "Thesis_Log/astra_result/evidence/sq1_pool_api_preparation_20260926_v1.json"
)


def forbidden(*args, **kwargs):
    raise AssertionError("No network or credentials during offline API review")


def tests():
    a.require(
        not b.PLAN.exists()
        and (not (b.BASE / "development/tests_passed.json").exists()),
        "Preserve frozen successful tests",
    )
    attempt = b.BASE / "development" / ("test_attempt_" + str(time.time_ns()))
    hashes = {name: a.digest(b.ROOT / name) for name in b.SOURCE_FILES}
    b.save(
        attempt / "source_snapshot.json",
        {
            name: (b.ROOT / name).read_text(encoding="utf8")
            for name in b.SOURCE_FILES
            if "pool_" in name
        },
    )
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(b.core, "live_client", forbidden):
        suite = unittest.defaultTestLoader.loadTestsFromNames(
            [
                "tests.common.test_api",
                "tests.common.test_batch",
                "tests.knowledge.sampled_banks.test_run",
                "tests.knowledge.sampled_banks.test_analysis_plan",
            ]
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
    b.save(attempt / "tests.json", receipt)
    print(stream.getvalue(), flush=True)
    a.require(
        result.wasSuccessful(), "Execution tests failed; preserve attempt and snapshots"
    )
    for name, digest in hashes.items():
        a.require(a.digest(b.ROOT / name) == digest, "Tested source changed")
    b.save(b.BASE / "development/tests_passed.json", receipt)


def check():
    a.require(
        not OUT.exists() and (not (b.BASE / "offline_check_complete.json").exists()),
        "Preserve completed offline review",
    )
    plan = b.validate()
    plan_sha = a.digest(b.PLAN)
    tests_receipt = a.read(b.BASE / "development/tests_passed.json")
    a.require(tests_receipt["passed"] is True, "Passing tests required")
    for name, digest in tests_receipt["source_hashes"].items():
        a.require(a.digest(b.ROOT / name) == digest, "Test binding differs")
    audit_tests_path = b.BASE / "audit_development/stage_check_v1/tests_passed.json"
    audit_tests = a.read(audit_tests_path)
    a.require(
        audit_tests["passed"] and audit_tests["environment"] == b.core.environment(),
        "Passing stage-audit tests required",
    )
    for name, digest in audit_tests["source_hashes"].items():
        a.require(a.digest(b.ROOT / name) == digest, "Stage-auditor source differs")
    schedule = a.read(b.BASE / "schedule.json")
    analysis = a.read(b.BASE / "analysis_plan.json")
    a.require(
        analysis
        == b.preparation.analysis_plan(
            a.read(b.STUDY / "scientific_plan.json"),
            a.read(b.preparation.CORE / "api_execution_v1/scoring_plan.json"),
            a.digest(b.preparation.CORE / "api_execution_v1/scoring_plan.json"),
            analysis["gold_input_hashes"],
        ),
        "Execution analysis differs from scientific contract",
    )
    cat = b.frozen_catalog(b.ROOT / plan["prompts"])
    dev = b.core.rows(b.preparation.CORE / "dev_inputs.jsonl")
    membership = {row["key"]: row["scopes"][0]["subset"] for row in dev}
    bysubset = {
        name: [row["key"] for row in dev if membership[row["key"]] == name]
        for name in b.ENDPOINT_ORDER
    }
    expected_keys = [
        bysubset[subset][i]
        for i in range(max(map(len, bysubset.values())))
        for subset in b.ENDPOINT_ORDER
        if i < len(bysubset[subset])
    ]
    jobs = [j for stage in schedule["stages"] for j in stage["jobs"]]
    ids = [(j["key"], j["condition"], j["repeat"]) for j in jobs]
    a.require(len(jobs) == len(set(ids)) == 42735, "Prediction roster size differs")
    a.require(
        set(ids) == {(key, cond, r) for key, cond in cat for r in range(3)},
        "Missing/extra condition/repeat",
    )
    a.require(
        [j["key"] for j in jobs[:: b.UNIT]] == expected_keys,
        "Independent endpoint order differs",
    )
    positions = [Counter() for _ in range(b.UNIT)]
    locations = defaultdict(set)
    endpoint_counts = Counter()
    for ordinal in range(1295):
        unit = jobs[ordinal * b.UNIT : ordinal * b.UNIT + b.UNIT]
        a.require(len({j["key"] for j in unit}) == 1, "Input comparison split")
        for index, job in enumerate(unit):
            block, position = divmod(index, 11)
            repeat = (ordinal + block) % 3
            condition = b.CONDITIONS[(ordinal + repeat + position) % 11]
            a.require(
                job["repeat"] == repeat and job["condition"] == condition,
                "Independent condition order differs",
            )
            a.require(
                job["strategy"] == "hybrid" and job["subset"] == membership[job["key"]],
                "Wrong scope/strategy",
            )
            a.require(
                job["messages_sha256"]
                == cat[job["key"], job["condition"]]["messages_sha256"],
                "Scheduled message differs",
            )
            positions[index][condition] += 1
            endpoint_counts[job["subset"]] += 1
    a.require(
        [s["items"] for s in schedule["stages"]] == [10692] * 3 + [10659],
        "Stage budget differs",
    )
    for stage in schedule["stages"]:
        a.require(
            stage["items"] == len(stage["jobs"])
            and stage["inputs"] == stage["items"] // b.UNIT,
            "Wrong stage counts",
        )
        a.require(
            stage["ordinary_responses"] == stage["items"] * 3
            and stage["max_logical_calls"] == stage["items"] * 9
            and (stage["max_physical_POSTs"] == stage["items"] * 45),
            "Wrong stage limits",
        )
        a.require(
            stage["endpoint_inputs"]
            == dict(Counter((j["subset"] for j in stage["jobs"][:: b.UNIT]))),
            "Stage endpoint count differs",
        )
        for job in stage["jobs"]:
            locations[job["key"]].add(stage["id"])
    a.require(
        all((len(v) == 1 for v in locations.values())),
        "Input comparison crosses stage boundary",
    )
    a.require(
        dict(endpoint_counts)
        == {name: n * b.UNIT for name, n in b.SCOPE_COUNTS.items()},
        "Wrong outcome totals",
    )
    for (key, condition), entry in cat.items():
        payload = a.body(entry["messages"])
        a.require(
            payload["model"] == a.MODEL
            and payload["temperature"] == 1.0
            and (payload["max_tokens"] == 4096),
            "Model settings differ",
        )
        a.require(
            payload["messages"]
            == [{"role": m["role"], "content": m["text"]} for m in entry["messages"]],
            "Wire payload differs",
        )
    attempt = a.storage_path(
        b.BASE / "offline_check" / ("rehearsal_" + str(time.time_ns()))
    )
    sample = jobs[:231]
    a.require(
        len({j["key"] for j in sample}) == 7
        and {j["subset"] for j in sample} == set(b.ENDPOINT_ORDER),
        "Rehearsal must cover seven rosters",
    )
    stage = {
        "id": "offline_seven_inputs",
        "jobs": sample,
        "items": 231,
        "ordinary_responses": 693,
    }
    count = [0]
    mutex = threading.Lock()
    valid = json.dumps(
        {
            "reasoning": "OFFLINE FIXTURE; not a model response",
            "hate": False,
            "target_group": [],
            "hate_type": [],
            "severity": None,
            "legal": ["insult_defamation"],
        }
    )

    def factory():

        def send(payload):
            with mutex:
                count[0] += 1
            return a.WireReply(
                200,
                json.dumps(
                    {
                        "model": a.MODEL,
                        "choices": [
                            {"message": {"content": valid}, "finish_reason": "stop"}
                        ],
                        "usage": {
                            "prompt_tokens": 10,
                            "completion_tokens": 5,
                            "total_tokens": 15,
                        },
                        "mock_only": True,
                    }
                ).encode(),
                {},
            )

        return send

    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(b.core, "live_client", forbidden):
        b.execute_stage(
            stage, cat, attempt / "mock", plan_sha, client_factory=factory, mode="mock"
        )
        a.require(count[0] == 693, "Rehearsal call count differs")
        b.execute_stage(
            stage,
            cat,
            attempt / "mock",
            plan_sha,
            client_factory=forbidden,
            mode="mock",
        )
        a.require(count[0] == 693, "Completed restart sent new request")
        for job in sample:
            a.audit_item(
                attempt / "mock",
                b.identity(job, plan_sha),
                cat[job["key"], job["condition"]]["messages"],
                execution_mode="mock",
            )
    a.require(not (b.BASE / "live").exists(), "Offline work created live output")
    a.require(
        all(
            (
                max(c.values()) - min(c.values()) <= 1 and set(c) == set(b.CONDITIONS)
                for c in positions
            )
        ),
        "Unbalanced arm positions",
    )
    inputs = {
        b.rel(audit_tests_path): a.digest(audit_tests_path),
        **plan["input_hashes"],
        **plan["source_hashes"],
        **plan["generated_hashes"],
        b.rel(b.PLAN): plan_sha,
    }
    for name, digest in inputs.items():
        a.require(a.digest(b.ROOT / name) == digest, "Input changed during rehearsal")
    outputs = {
        b.rel(p): a.digest(p)
        for p in a.storage_path(b.BASE / "offline_check").rglob("*")
        if p.is_file() and p.name not in ("item.lock", "launcher.lock")
    }
    report = {
        "created_utc": a.utc(),
        "status": "offline_checked_ready_for_user_stage_01",
        "plan_sha256": plan_sha,
        "programme": plan["programme"],
        "stages": [
            {k: v for k, v in s.items() if k != "jobs"} for s in schedule["stages"]
        ],
        "verified_prompt_payloads": len(cat),
        "unique_scheduled_predictions": len(jobs),
        "endpoint_predictions": dict(endpoint_counts),
        "position_counts": [dict(v) for v in positions],
        "scientific_preparation": a.read(
            b.STUDY / "retrieval_v1/review_v1/retrieval_acceptance.json"
        ),
        "stage_auditor_tests_run": audit_tests["tests_run"],
        "tests_run": tests_receipt["tests_run"],
        "test_attempt": tests_receipt["attempt"],
        "rehearsal_attempt": b.rel(attempt),
        "mock_predictions": 231,
        "mock_POSTs": 693,
        "new_POSTs_on_completed_stage_restart": 0,
        "actual_API_calls": 0,
        "live_credentials_loaded": False,
        "gold_scoring": False,
        "new_retrieval_or_training": False,
        "verified_input_hashes": inputs,
        "verified_output_hashes": outputs,
        "qualification": "Offline readiness; no live service probe. First incomplete stage checks model roster. Backend identity/latency cannot be guaranteed by mock execution.",
        "next": "Ajay runs stage_01 only. Audit saved responses before stage_02; no automatic next stage.",
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
                    "verified_prompt_payloads",
                    "unique_scheduled_predictions",
                    "mock_predictions",
                    "mock_POSTs",
                    "new_POSTs_on_completed_stage_restart",
                    "actual_API_calls",
                )
            },
            indent=2,
        )
    )
    print("Evidence SHA256: " + a.digest(OUT))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tests", "check"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(b.core, "live_client", forbidden):
        tests() if args.mode == "tests" else check()

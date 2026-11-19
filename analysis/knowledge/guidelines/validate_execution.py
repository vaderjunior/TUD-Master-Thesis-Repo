import argparse
from collections import Counter, defaultdict
import io
from itertools import combinations
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch
from analysis.knowledge.guidelines import run as b

a = b.a
OUT = (
    b.ROOT
    / "Thesis_Log/astra_result/evidence/sq1_guideline_api_preparation_20260925_v1.json"
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
            if "guideline_api" in name
        },
    )
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(b.core, "live_client", forbidden):
        suite = unittest.defaultTestLoader.loadTestsFromName(
            "tests.knowledge.guidelines.test_run"
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
    audit_tests_path = b.BASE / "audit_development/stage_check_v1/tests_passed.json"
    audit_tests = a.read(audit_tests_path)
    for receipt in (tests_receipt, audit_tests):
        a.require(
            receipt["passed"] is True
            and receipt["environment"] == b.core.environment(),
            "Passing current tests required",
        )
        for name, digest in receipt["source_hashes"].items():
            a.require(a.digest(b.ROOT / name) == digest, "Test source differs")
    schedule = a.read(b.BASE / "schedule.json")
    analysis = a.read(b.BASE / "analysis_plan.json")
    a.require(
        analysis == a.read(b.STUDY / "analysis_plan.json"),
        "Execution analysis differs from scientific contract",
    )
    cat = b.frozen_catalog(b.ROOT / plan["prompts"])
    inputs = b.core.rows(b.preparation.INVENTORY / "inputs.jsonl")
    expected_keys = [row["key"] for row in inputs]
    jobs = [j for stage in schedule["stages"] for j in stage["jobs"]]
    ids = [(j["key"], j["condition"], j["repeat"]) for j in jobs]
    a.require(len(jobs) == len(set(ids)) == 10395, "Prediction roster size differs")
    a.require(
        set(ids) == {(key, c, r) for key, c in cat for r in range(3)},
        "Missing/extra condition/repeat",
    )
    a.require(
        [j["key"] for j in jobs[::15]] == expected_keys, "Frozen input order differs"
    )
    positions = [Counter() for _ in range(15)]
    locations = defaultdict(set)
    endpoint_counts = Counter()
    for ordinal, key in enumerate(expected_keys):
        unit = jobs[ordinal * 15 : ordinal * 15 + 15]
        a.require({j["key"] for j in unit} == {key}, "Input comparison split")
        for index, job in enumerate(unit):
            block, position = divmod(index, 5)
            condition = b.CONDITIONS[(ordinal + 2 * block + position) % 5]
            a.require(
                job["repeat"] == (ordinal + block) % 3
                and job["condition"] == condition,
                "Independent condition order differs",
            )
            a.require(
                job["strategy"] == "hybrid" and job["subset"] == b.ENDPOINT,
                "Wrong scope/strategy",
            )
            a.require(
                job["messages_sha256"]
                == cat[job["key"], job["condition"]]["messages_sha256"],
                "Scheduled message differs",
            )
            positions[index][condition] += 1
            endpoint_counts[job["subset"]] += 1
    scientific = a.read(b.preparation.PLAN)
    a.require(
        [s["items"] for s in schedule["stages"]] == [2610] * 3 + [2565],
        "Stage budget differs",
    )
    for stage, expected in zip(schedule["stages"], scientific["planned_stages"]):
        a.require(
            stage["items"] == len(stage["jobs"])
            and stage["inputs"] == stage["items"] // 15,
            "Wrong stage counts",
        )
        a.require(
            stage["ordinary_responses"] == stage["items"] * 3
            and stage["max_logical_calls"] == stage["items"] * 9
            and (stage["max_physical_POSTs"] == stage["items"] * 45),
            "Wrong stage limits",
        )
        keys = [j["key"] for j in stage["jobs"][::15]]
        a.require(
            a.signature(keys) == expected["input_keys_sha256"],
            "Scientific stage input binding differs",
        )
        a.require(
            stage["endpoint_inputs"] == {b.ENDPOINT: len(keys)},
            "Stage endpoint count differs",
        )
        for job in stage["jobs"]:
            locations[job["key"]].add(stage["id"])
    a.require(
        all((len(v) == 1 for v in locations.values())),
        "Input comparison crosses stage boundary",
    )
    a.require(dict(endpoint_counts) == {b.ENDPOINT: 10395}, "Wrong outcome totals")
    for entry in cat.values():
        payload = a.body(entry["messages"])
        a.require(
            set(payload) == {"model", "temperature", "max_tokens", "messages"},
            "Unplanned wire metadata",
        )
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
    duplicates = []
    duplicate_counts = {}
    for left, right in combinations(b.CONDITIONS, 2):
        keys = [
            key
            for key in expected_keys
            if cat[key, left]["messages"] == cat[key, right]["messages"]
        ]
        duplicate_counts[left + " / " + right] = len(keys)
        duplicates.extend(({"key": key, "conditions": [left, right]} for key in keys))
    acceptance = a.read(b.STUDY / "retrieval_acceptance_v1.json")
    a.require(
        duplicate_counts == acceptance["identical_within_input_condition_payloads"],
        "Duplicate payload count differs",
    )
    sample_keys = set(expected_keys[:6]) | {row["key"] for row in duplicates}
    sample = [j for j in jobs if j["key"] in sample_keys]
    a.require(
        len({row["functionality"] for row in inputs[:6]}) == 6,
        "First six no longer cover all functionalities",
    )
    a.require(
        len(sample) == len(sample_keys) * 15 and len(duplicates) == 5,
        "Rehearsal coverage differs",
    )
    attempt = a.storage_path(
        b.BASE / "offline_check" / ("rehearsal_" + str(time.time_ns()))
    )
    stage = {
        "id": "offline_six_functions_and_duplicate_payloads",
        "jobs": sample,
        "items": len(sample),
        "ordinary_responses": len(sample) * 3,
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
        a.require(count[0] == len(sample) * 3, "Rehearsal call count differs")
        b.execute_stage(
            stage,
            cat,
            attempt / "mock",
            plan_sha,
            client_factory=forbidden,
            mode="mock",
        )
        a.require(count[0] == len(sample) * 3, "Completed restart sent new request")
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
    input_hashes = {
        b.rel(audit_tests_path): a.digest(audit_tests_path),
        **plan["input_hashes"],
        **plan["source_hashes"],
        **plan["generated_hashes"],
        b.rel(b.PLAN): plan_sha,
    }
    for name, digest in input_hashes.items():
        a.require(a.digest(b.ROOT / name) == digest, "Input changed during rehearsal")
    output_hashes = {
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
        "scientific_preparation": acceptance,
        "identical_within_input_condition_payloads": duplicate_counts,
        "rehearsed_duplicate_cases": duplicates,
        "stage_auditor_tests_run": audit_tests["tests_run"],
        "tests_run": tests_receipt["tests_run"],
        "test_attempt": tests_receipt["attempt"],
        "rehearsal_attempt": b.rel(attempt),
        "mock_input_keys": [key for key in expected_keys if key in sample_keys],
        "mock_predictions": len(sample),
        "mock_POSTs": count[0],
        "new_POSTs_on_completed_stage_restart": 0,
        "actual_API_calls": 0,
        "live_credentials_loaded": False,
        "gold_scoring": False,
        "new_retrieval_or_training": False,
        "verified_input_hashes": input_hashes,
        "verified_output_hashes": output_hashes,
        "qualification": "Offline readiness; no live service probe. First incomplete stage checks model roster. Mock execution does not establish backend identity, runtime, or scientific effects. All live responses remain pending.",
        "next": "Ajay runs stage_01 only. Review saved audit before stage_02; no automatic next stage.",
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
                    "stage_auditor_tests_run",
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
    print("Evidence SHA256: " + a.digest(OUT), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tests", "check"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(b.core, "live_client", forbidden):
        tests() if args.mode == "tests" else check()

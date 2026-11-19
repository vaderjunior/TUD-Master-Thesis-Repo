from collections import Counter, defaultdict
import argparse
import io
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch
from analysis.knowledge.strategies import run as b

a = b.a
OUT = (
    b.ROOT
    / "Thesis_Log/astra_result/evidence/sq1_strategy_api_preparation_20260920_v1.json"
)


def forbidden(*args, **kwargs):
    raise AssertionError("Network/credential loading forbidden in offline review")


def tests():
    a.require(
        not b.PLAN.exists(), "Run development tests before freezing the execution plan"
    )
    a.require(
        not (b.BASE / "development/tests_passed.json").exists(),
        "Preserve already passing test evidence",
    )
    attempt = a.storage_path(b.BASE / "development" / f"test_attempt_{time.time_ns()}")
    sources = {name: a.digest(b.ROOT / name) for name in b.SOURCE_FILES}
    b.save(
        attempt / "source_snapshot.json",
        {
            name: (b.ROOT / name).read_text(encoding="utf8")
            for name in b.SOURCE_FILES
            if "sq1_strategy_api" in name
        },
    )
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(b.core, "live_client", forbidden):
        suite = unittest.defaultTestLoader.loadTestsFromNames(
            [
                "tests.common.test_api",
                "tests.common.test_batch",
                "tests.knowledge.strategies.test_run",
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
        "source_hashes": sources,
        "environment": b.core.environment(),
        "actual_API_calls": 0,
        "attempt": b.rel(attempt),
    }
    b.save(attempt / "tests.json", receipt)
    print(stream.getvalue(), flush=True)
    a.require(
        result.wasSuccessful(),
        "Development tests failed; attempt and source snapshot retained",
    )
    for name, digest in sources.items():
        a.require(a.digest(b.ROOT / name) == digest, "Source changed during tests")
    b.save(b.BASE / "development/tests_passed.json", receipt)


def main():
    a.require(
        not OUT.exists() and (not (b.BASE / "offline_check_complete.json").exists()),
        "Preserve completed offline review",
    )
    plan = b.validate()
    plan_sha = a.digest(b.PLAN)
    tests_receipt = a.read(b.BASE / "development/tests_passed.json")
    a.require(tests_receipt["passed"] is True, "Passing development tests required")
    for name, digest in tests_receipt["source_hashes"].items():
        a.require(a.digest(b.ROOT / name) == digest, "Test/source binding differs")
    schedule = a.read(b.BASE / "schedule.json")
    analysis = a.read(b.BASE / "analysis_plan.json")
    study = a.read(b.STUDY / "plan.json")
    cat = b.frozen_catalog(b.ROOT / plan["prompts"])
    raw = b.core.rows(b.ROOT / plan["prompts"])
    dev = b.core.rows(b.retrieval.CORE / "dev_inputs.jsonl")
    membership = {r["key"]: r["scopes"][0]["subset"] for r in dev}
    keys_by_subset = {
        name: [r["key"] for r in dev if membership[r["key"]] == name]
        for name in b.ENDPOINT_ORDER
    }
    expected_keys = [
        keys_by_subset[subset][i]
        for i in range(max(map(len, keys_by_subset.values())))
        for subset in b.ENDPOINT_ORDER
        if i < len(keys_by_subset[subset])
    ]
    jobs = [j for stage in schedule["stages"] for j in stage["jobs"]]
    identities = [(j["key"], j["condition"], j["repeat"]) for j in jobs]
    a.require(
        len(jobs) == len(set(identities)) == 11655, "Wrong unique prediction roster"
    )
    a.require(
        set(identities)
        == {(key, condition, r) for key, condition in cat for r in range(3)},
        "Missing/extra comparison",
    )
    a.require(
        [j["key"] for j in jobs[::9]] == expected_keys,
        "Independent round-robin input order differs",
    )
    position_counts = [Counter() for _ in range(9)]
    location = defaultdict(set)
    endpoint_counts = Counter()
    for ordinal in range(1295):
        unit = jobs[ordinal * 9 : ordinal * 9 + 9]
        a.require(len({j["key"] for j in unit}) == 1, "Input unit split")
        for p, job in enumerate(unit):
            block, pos = divmod(p, 3)
            repeat = (ordinal + block) % 3
            strategy = ("dense", "bm25", "hybrid")[(ordinal + repeat + pos) % 3]
            a.require(
                job["repeat"] == repeat
                and job["strategy"] == strategy
                and (job["condition"] == "rag_" + strategy),
                "Independent rotation differs",
            )
            a.require(
                job["subset"] == membership[job["key"]], "Wrong scheduled endpoint"
            )
            a.require(
                job["messages_sha256"]
                == cat[job["key"], job["condition"]]["messages_sha256"],
                "Wrong scheduled message",
            )
            position_counts[p][strategy] += 1
            endpoint_counts[job["subset"]] += 1
    for stage in schedule["stages"]:
        a.require(
            stage["items"] == len(stage["jobs"]) <= 2700 and stage["items"] % 9 == 0,
            "Stage cap/unit",
        )
        a.require(
            stage["inputs"] == stage["items"] // 9
            and stage["ordinary_responses"] == stage["items"] * 3,
            "Stage arithmetic",
        )
        a.require(
            stage["max_logical_calls"] == stage["items"] * 9
            and stage["max_physical_POSTs"] == stage["items"] * 45,
            "Worst-case cap arithmetic",
        )
        a.require(
            stage["endpoint_inputs"]
            == dict(Counter((j["subset"] for j in stage["jobs"][::9]))),
            "Stage endpoint counts",
        )
        for j in stage["jobs"]:
            location[j["key"]].add(stage["id"])
    a.require(
        all((len(value) == 1 for value in location.values())),
        "Input comparison crosses stages",
    )
    a.require(
        sum((x["ordinary_responses"] for x in schedule["stages"])) == 34965,
        "Programme arithmetic",
    )
    a.require(
        all(
            (
                max(value.values()) - min(value.values()) <= 1
                for value in position_counts
            )
        ),
        "Strategy position imbalance",
    )
    for entry in raw:
        payload = a.body(cat[entry["key"], entry["condition"]]["messages"])
        a.require(
            payload
            == {
                "model": a.MODEL,
                "temperature": 1.0,
                "max_tokens": 4096,
                "messages": [
                    {"role": m["role"], "content": m["text"]} for m in entry["messages"]
                ],
            },
            "Wire body differs from first-pass prompt",
        )
    inherited = a.read(b.retrieval.CORE / "api_execution_v1/scoring_plan.json")[
        "endpoints"
    ]
    for subset, endpoint in analysis["endpoints"].items():
        expected = {
            **inherited[subset],
            "inherited_core_role": inherited[subset]["role"],
            "role": study["analysis_roles"][subset],
        }
        a.require(
            endpoint == expected
            and endpoint_counts[subset] == len(keys_by_subset[subset]) * 9,
            "Inherited labels/masks or role mapping differ",
        )
    a.require(
        analysis["analysis"] == study["analysis"]
        and analysis["focal_family_size"] == 3
        and (analysis["other_endpoint_contrasts"] == 18)
        and (analysis["analysis"]["bootstrap_seed"] == 20260920),
        "New analysis contract differs",
    )
    attempt = a.storage_path(b.BASE / "offline_check" / f"rehearsal_{time.time_ns()}")
    sample_jobs = jobs[:63]
    a.require(
        {j["subset"] for j in sample_jobs} == set(b.ENDPOINT_ORDER),
        "Mock rehearsal must cover all seven endpoints",
    )
    sample_stage = {
        "id": "offline_seven_inputs",
        "jobs": sample_jobs,
        "items": 63,
        "ordinary_responses": 189,
    }
    count = [0]
    mutex = threading.Lock()
    valid = json.dumps(
        {
            "reasoning": "OFFLINE FIXTURE, not a model response",
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
            sample_stage,
            cat,
            attempt / "mock",
            plan_sha,
            client_factory=factory,
            mode="mock",
        )
        a.require(count[0] == 189, "Mock run response count differs")
        b.execute_stage(
            sample_stage,
            cat,
            attempt / "mock",
            plan_sha,
            client_factory=forbidden,
            mode="mock",
        )
        a.require(count[0] == 189, "Completed-stage restart sent a new request")
        for job in sample_jobs:
            a.audit_item(
                attempt / "mock",
                b.identity(job, plan_sha),
                cat[job["key"], job["condition"]]["messages"],
                execution_mode="mock",
            )
    a.require(not (b.BASE / "live").exists(), "Offline review created live output")
    inputs = {
        **plan["input_hashes"],
        **plan["source_hashes"],
        **plan["generated_hashes"],
        b.rel(b.PLAN): plan_sha,
    }
    for name, digest in inputs.items():
        a.require(
            a.digest(b.ROOT / name) == digest, "Input changed during offline check"
        )
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
            {k: v for k, v in x.items() if k != "jobs"} for x in schedule["stages"]
        ],
        "input_prompt_payloads_verified": len(raw),
        "unique_scheduled_predictions": len(jobs),
        "endpoint_predictions": dict(endpoint_counts),
        "position_counts": [dict(c) for c in position_counts],
        "inherited_mask_and_label_checks": "All seven saved endpoint masks, supports and labels equal core; only explicit new analysis roles differ. No fresh gold/outcome scoring.",
        "tests_run": tests_receipt["tests_run"],
        "test_attempt": tests_receipt["attempt"],
        "rehearsal_attempt": b.rel(attempt),
        "mock_predictions": 63,
        "mock_POSTs": 189,
        "new_POSTs_on_completed_stage_restart": 0,
        "actual_API_calls": 0,
        "live_credentials_loaded": False,
        "actual_gold_scoring": False,
        "new_retrieval_or_training": False,
        "verified_input_hashes": inputs,
        "verified_output_hashes": outputs,
        "qualification": "Offline readiness only; no new live service probe. First incomplete live-stage attempt will save/check model availability and use the same reviewed core transport. Backend identity/latency cannot be guaranteed by this rehearsal.",
        "next": "Ajay runs the named stage_01 launcher. Inspect all saved responses afterward. No next stage starts automatically.",
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
                    "input_prompt_payloads_verified",
                    "unique_scheduled_predictions",
                    "mock_predictions",
                    "mock_POSTs",
                    "new_POSTs_on_completed_stage_restart",
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
    tests() if args.mode == "tests" else main()

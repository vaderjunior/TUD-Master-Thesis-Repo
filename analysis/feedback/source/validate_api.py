import argparse
from collections import Counter
import hashlib
import io
import json
import socket
import threading
import time
import unittest
from unittest.mock import patch
from analysis.feedback.source import run as b

a = b.a
OUT = (
    b.ROOT
    / "Thesis_Log/astra_result/evidence/sq3_feedback_baseline_preparation_20260928_v1.json"
)


def forbidden(*args, **kwargs):
    raise AssertionError(
        "No network or live credentials in offline baseline preparation"
    )


def tests():
    a.require(
        not b.PLAN.exists()
        and (not (b.BASE / "development/tests_passed.json").exists()),
        "Preserve frozen passing tests",
    )
    attempt = b.BASE / "development" / ("test_attempt_" + str(time.time_ns()))
    hashes = {name: a.digest(b.ROOT / name) for name in b.SOURCE_FILES}
    b.save(
        attempt / "source_snapshot.json",
        {
            name: (b.ROOT / name).read_text(encoding="utf8")
            for name in b.SOURCE_FILES
            if "sq3_feedback" in name
        },
    )
    suite = unittest.defaultTestLoader.loadTestsFromNames(
        [
            "tests.common.test_api",
            "tests.common.test_batch",
            "tests.feedback.source.test_run",
            "tests.feedback.source.test_audit",
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
        "environment": b.environment(),
        "actual_API_calls": 0,
        "attempt": b.rel(attempt),
    }
    b.save(attempt / "tests.json", receipt)
    print(stream.getvalue(), flush=True)
    a.require(
        result.wasSuccessful(),
        "Preparation test failed; preserve source snapshot and failure",
    )
    for name, h in hashes.items():
        a.require(a.digest(b.ROOT / name) == h, "Source changed during tests")
    b.save(b.BASE / "development/tests_passed.json", receipt)


def check():
    import numpy as np

    a.require(
        not OUT.exists() and (not (b.BASE / "offline_check_complete.json").exists()),
        "Preserve successful readiness evidence",
    )
    plan = b.validate()
    plan_sha = a.digest(b.PLAN)
    tests_receipt = a.read(b.BASE / "development/tests_passed.json")
    audit_tests_path = b.BASE / "audit_development/stage_check_v1/tests_passed.json"
    audit_tests = a.read(audit_tests_path)
    a.require(
        audit_tests["passed"] and audit_tests["environment"] == b.core.environment(),
        "Stage audit tests required",
    )
    for name, h in audit_tests["source_hashes"].items():
        a.require(a.digest(b.ROOT / name) == h, "Audit test binding differs")
    science = a.read(b.SCIENCE)
    roster = b.core.rows(b.preparation.OUT / "roster.jsonl")
    byid = {r["id"]: r for r in roster}
    eligible = {r["id"] for r in roster if r["correction_eligible"]}
    pool = {r["id"] for r in roster if r["role"] == "pool"}
    holdout = {r["id"] for r in roster if r["role"] == "held_out"}
    selection = eligible | pool
    reference = selection | holdout
    a.require(
        (len(eligible), len(pool), len(holdout), len(selection), len(reference))
        == (116, 229, 121, 345, 466),
        "Independent role counts differ",
    )
    a.require(
        selection == set(science["rosters"]["selection_phase"])
        and reference == set(science["rosters"]["evaluation_phase"]),
        "Independent phase reconstruction differs",
    )
    a.require(not eligible & (pool | holdout) and (not pool & holdout), "Role overlap")
    copied = b.core.rows(b.ROOT / plan["prompts"])
    originals = {
        r["item_id"]: r
        for r in b.core.rows(b.preparation.RET / "full/prompts.jsonl")
        if r["lang"] == "en"
    }
    records = {
        r["id"]: r for r in b.core.rows(b.preparation.CORE / "kb/full/records.jsonl")
    }
    annotated_hits = 0
    for row in copied:
        a.require(
            row == originals[row["item_id"]], "Copied entry differs from core original"
        )
        messages = b.renderer.render_messages(
            row["text"], row["lang"], row["context"], "corrected"
        )
        a.require(messages == row["messages"], "Independent baseline rerender differs")
        ctx = row["context"]
        for hit in (
            ctx["definitions"]
            + ctx["guidelines"]
            + [h for _, rr in ctx["example_groups"] for h in rr]
        ):
            rec = records[hit["id"]]
            meta = {
                **{k: rec[k] for k in ("kind", "lang", "source", "dimension", "label")},
                **rec["meta"],
            }
            a.require(
                hit["text"] == rec["text"]
                and all((hit["meta"].get(k) == v for k, v in meta.items()))
                and (set(hit["meta"]) <= set(meta) | {"_ranks"}),
                "Original annotation/text changed",
            )
            annotated_hits += 1
    cat = b.frozen_catalog(b.ROOT / plan["prompts"])
    schedule = a.read(b.BASE / "schedule.json")
    a.require(len(schedule["stages"]) == 1, "Update execution prematurely enabled")
    stage = schedule["stages"][0]
    jobs = stage["jobs"]
    ids = [(j["key"], j["condition"], j["repeat"]) for j in jobs]
    expected = {
        ("en/" + i, condition, k)
        for condition, items in zip(b.CONDITIONS, (selection, reference))
        for i in items
        for k in range(3)
    }
    a.require(
        set(ids) == expected and len(ids) == len(set(ids)) == 2433,
        "Independent job roster differs",
    )
    expected_order = [
        (condition, "en/" + i, (ordinal + block) % 3)
        for condition, items in zip(b.CONDITIONS, (selection, reference))
        for ordinal, i in enumerate(sorted(items))
        for block in range(3)
    ]
    a.require(
        [(j["condition"], j["key"], j["repeat"]) for j in jobs] == expected_order,
        "Independent submission order differs",
    )
    a.require(
        stage["phase_predictions"] == {"selection": 1035, "reference": 1398},
        "Phase budgets differ",
    )
    a.require(
        [
            stage[k]
            for k in (
                "items",
                "distinct_inputs",
                "ordinary_responses",
                "max_logical_calls",
                "max_physical_POSTs",
            )
        ]
        == [2433, 466, 7299, 21897, 109485],
        "Hard budget differs",
    )
    phase_ids = {"selection": set(), "reference": set()}
    for job in jobs:
        key, condition = (job["key"], job["condition"])
        entry = cat[key, condition]
        a.require(
            job["phase"] == condition.removesuffix("_baseline")
            and job["strategy"] == "hybrid"
            and (job["subset"] == entry["subset"]),
            "Job attribution differs",
        )
        a.require(
            job["messages_sha256"] == entry["messages_sha256"], "Job payload differs"
        )
        phase_ids[job["phase"]].add(a.signature(b.identity(job, plan_sha)))
        payload = a.body(entry["messages"])
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
            "Wire conversion differs",
        )
    a.require(
        not phase_ids["selection"] & phase_ids["reference"],
        "Fresh phases share result identity",
    )
    for iid in selection:
        a.require(
            cat["en/" + iid, b.CONDITIONS[0]] == cat["en/" + iid, b.CONDITIONS[1]],
            "Identical baseline source differs across phases",
        )
    dev = b.core.rows(b.preparation.CORE / "dev_inputs.jsonl")
    vector_source = b.core.rows(b.preparation.RET / "query_vector_sources.jsonl")
    vectors = np.load(b.preparation.RET / "query_vectors.npy", mmap_mode="r")
    a.require(
        vectors.shape == (1295, 1024) and len(vector_source) == len(dev) == 1295,
        "Cached vector roster differs",
    )
    index = {r["key"]: i for i, r in enumerate(dev)}
    for iid in reference:
        n = index["en/" + iid]
        original = dev[n]
        source = vector_source[n]
        a.require(
            original["text"] == byid[iid]["text"] and source["key"] == original["key"],
            "Vector/text provenance differs",
        )
        a.require(
            source["text_sha256"]
            == hashlib.sha256(original["text"].encode()).hexdigest(),
            "Vector source text hash differs",
        )
        a.require(
            np.isfinite(vectors[n]).all()
            and np.linalg.norm(vectors[n].astype(np.float64)) > 0,
            "Invalid cached vector",
        )
    print(
        "All467 baseline entries/source annotations,466 cached vectors and2,433 phase jobs verified. Starting full mock stage.",
        flush=True,
    )
    attempt = a.storage_path(
        b.BASE / "offline_check" / ("full_rehearsal_" + str(time.time_ns()))
    )
    count = [0]
    mutex = threading.Lock()
    valid = json.dumps(
        {
            "reasoning": "OFFLINE FIXTURE ONLY; not an accuracy result",
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
                number = count[0]
            return a.WireReply(
                200,
                json.dumps(
                    {
                        "id": f"mock-{number}",
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

    b.execute_stage(
        stage, cat, attempt / "mock", plan_sha, client_factory=factory, mode="mock"
    )
    a.require(count[0] == 7299, "Full mock response count differs")
    print(
        "Full mock complete. Reopening the completed stage with every client constructor blocked.",
        flush=True,
    )
    b.execute_stage(
        stage, cat, attempt / "mock", plan_sha, client_factory=forbidden, mode="mock"
    )
    a.require(count[0] == 7299, "Completed restart sent a request")
    a.require(
        not (b.BASE / "live").exists(), "Offline preparation created live results"
    )
    inputs = {
        **plan["input_hashes"],
        **plan["source_hashes"],
        **plan["generated_hashes"],
        b.rel(b.PLAN): plan_sha,
        b.rel(audit_tests_path): a.digest(audit_tests_path),
        plan["lexical_source"]["path"]: plan["lexical_source"]["sha256"],
    }
    for name, h in inputs.items():
        a.require(a.digest(b.ROOT / name) == h, "Input changed during rehearsal")
    print(
        "Restart sent zero requests. Hashing saved mock evidence for publication.",
        flush=True,
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
        "scientific_plan_sha256": b.STUDY_PLAN_SHA,
        "programme": plan["programme"],
        "stages": [{k: v for k, v in stage.items() if k != "jobs"}],
        "tests_run": tests_receipt["tests_run"],
        "stage_auditor_tests_run": audit_tests["tests_run"],
        "copied_baseline_entries_verified": len(copied),
        "original_context_hits_verified": annotated_hits,
        "phase_payloads": len(cat),
        "unique_scheduled_predictions": len(jobs),
        "cached_vectors_verified": len(reference),
        "distinct_selection_identities": len(phase_ids["selection"]),
        "distinct_reference_identities": len(phase_ids["reference"]),
        "shared_result_identities_between_phases": 0,
        "rehearsal_attempt": b.rel(attempt),
        "mock_predictions": 2433,
        "mock_POSTs": 7299,
        "new_POSTs_on_completed_stage_restart": 0,
        "actual_API_calls": 0,
        "live_credentials_loaded": False,
        "CUDA_or_training": False,
        "gold_scoring": False,
        "actual_update_pools_selected": False,
        "verified_input_hashes": inputs,
        "verified_output_hashes": outputs,
        "qualification": "Mock validation does not verify live backend availability, speed or checkpoint identity. Installed exact bucket is tested; actual selected update pools/payloads still need independent source/replay verification.",
        "next": "Ajay runs stage_01 baseline only, including offline raw-response audit. Astra reviews baseline before selecting real corrections and preparing remaining three batches.",
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
                    "copied_baseline_entries_verified",
                    "unique_scheduled_predictions",
                    "mock_predictions",
                    "mock_POSTs",
                    "new_POSTs_on_completed_stage_restart",
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

"""Tests for german guidelines run."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import socket
import threading
import time
import unittest
from unittest.mock import patch
import numpy as np
from analysis.knowledge.german_guidelines import common as c
from analysis.knowledge.german_guidelines import prepare as p
from analysis.knowledge.german_guidelines import run as r
from analysis.common import schema as isolated

DECISION = {
    "reasoning": "OFFLINE TEST ONLY",
    "hate": False,
    "target_group": [],
    "hate_type": [],
    "severity": None,
    "legal": [],
}
MESSAGES = [
    {"role": "system", "text": "Offline fixture"},
    {"role": "user", "text": "Offline input"},
]
TEST_ROOT = c.BASE / "offline_checks" / f"tests_{time.time_ns()}"


def reply(value=DECISION, status=200, model=None, raw=None):
    body = {
        "model": model or c.a.MODEL,
        "choices": [
            {
                "message": {"content": json.dumps(value) if raw is None else raw},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return c.a.WireReply(status, json.dumps(body).encode(), {"Content-Type": "application/json"})


class Transport:

    def __init__(self, responses=None, delay=0):
        self.calls = 0
        self.responses = responses
        self.delay = delay
        self.active = 0
        self.peak = 0
        self.lock = threading.Lock()

    def __call__(self, payload):
        with self.lock:
            number = self.calls
            self.calls += 1
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.delay:
                time.sleep(self.delay)
            value = self.responses[number] if self.responses is not None else reply()
            if isinstance(value, BaseException):
                raise value
            return value
        finally:
            with self.lock:
                self.active -= 1


class Checks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.backend = isolated.adapter(c.ROOT / "config/taxonomy.yaml")

    def setUp(self):
        self.root = TEST_ROOT / self._testMethodName
        self.root.mkdir(parents=True)
        self.ident = {
            "experiment_sha256": "offline-fixture",
            "condition": "mixed_2",
            "repeat": 0,
            "key": "de/fixture",
        }

    def execute(self, transport, **kwargs):
        return self.backend.execute_item(
            self.root, self.ident, MESSAGES, transport, sleep=lambda _: None, **kwargs
        )

    def audit(self):
        return self.backend.audit_item(self.root, self.ident, MESSAGES)

    def test_completed_item_resume_sends_nothing(self):
        transport = Transport()
        first = self.execute(transport)
        self.assertEqual(transport.calls, 3)
        self.assertEqual(self.execute(c.forbidden), first)
        self.assertEqual(self.audit()["status"], "verified_from_saved_responses")

    def test_known_network_failure_reuses_saved_response_and_retry_budget(self):
        transport = Transport([c.a.TransportFault("ConnectionError"), reply(), reply(), reply()])
        result = self.execute(transport)
        self.assertEqual(result["accounting"]["post_attempts"], 4)
        self.assertEqual(result["accounting"]["retry_attempts"], 1)
        self.execute(c.forbidden)
        self.audit()

    def test_unknown_request_is_never_resent(self):
        transport = Transport([RuntimeError("Simulated process death after intent")])
        with self.assertRaises(RuntimeError):
            self.execute(transport)
        with self.assertRaises(c.a.IncompleteAttempt):
            self.execute(c.forbidden)
        self.assertEqual(transport.calls, 1)

    def test_exhausted_retry_budget_is_not_reset(self):
        transport = Transport([c.a.TransportFault("ConnectionError")] * 5)
        with self.assertRaises(c.a.CallStopped):
            self.execute(transport)
        with self.assertRaises(c.a.CallStopped):
            self.execute(c.forbidden)
        self.assertEqual(transport.calls, 5)

    def test_saved_outcome_before_vote_commit_can_resume(self):
        transport = Transport()
        real_save = c.a.save_new
        failed = False

        def save(path, obj):
            nonlocal failed
            if Path(path).name == "0.json" and Path(path).parent.name == "votes" and (not failed):
                failed = True
                raise RuntimeError("Simulated interruption before vote publication")
            real_save(path, obj)

        with patch.dict(self.backend.execute_item.__globals__, save_new=save):
            with self.assertRaises(RuntimeError):
                self.execute(transport)
        self.assertEqual(transport.calls, 1)
        result = self.execute(transport)
        self.assertEqual(transport.calls, 3)
        self.assertEqual(result["n_valid"], 3)
        self.audit()

    def test_repairs_and_permanent_schema_failures(self):
        transport = Transport(
            [
                reply(raw="not JSON"),
                reply(),
                reply(raw="bad"),
                reply(raw="bad"),
                reply(raw="bad"),
                reply(),
            ]
        )
        result = self.execute(transport)
        self.assertEqual(result["n_valid"], 2)
        self.assertEqual(result["repairs"], 3)
        self.assertEqual(result["parse_failures"], 1)
        self.audit()

    def test_all_invalid_votes_remain_null(self):
        result = self.execute(Transport([reply(raw="invalid")] * 9))
        self.assertIsNone(result["result"])
        self.assertEqual(result["n_valid"], 0)
        self.assertIsNone(r.independent_decision(result["run_predictions"]))
        self.audit()

    def test_legal_is_independent_of_hate_gate(self):
        value = DECISION | {"legal": ["insult_defamation"]}
        result = self.execute(Transport([reply(value)] * 3))
        self.assertFalse(result["result"]["hate"])
        self.assertEqual(result["result"]["legal"], ["insult_defamation"])
        self.audit()

    def test_two_valid_votes_gate_tie_and_fine_normalization(self):
        positive = DECISION | {"hate": True, "hate_type": ["explicit"], "severity": "high"}
        result = self.execute(
            Transport(
                [reply(positive), reply(), reply(raw="bad"), reply(raw="bad"), reply(raw="bad")]
            )
        )
        self.assertEqual(result["n_valid"], 2)
        self.assertTrue(result["uncertain"])
        self.assertFalse(result["result"]["hate"])
        self.assertEqual(result["result"]["hate_type"], [])
        self.assertIsNone(result["result"]["severity"])
        self.audit()

    def test_authentication_and_wrong_model_stop(self):
        with self.assertRaises(c.a.CallStopped):
            self.execute(Transport([reply(status=401)]))
        self.ident = self.ident | {"key": "de/wrong-model"}
        with self.assertRaises(c.a.CallStopped):
            self.execute(Transport([reply(model="different-model")]))

    def test_saved_result_tampering_is_rejected(self):
        self.execute(Transport())
        path = c.a.storage_path(self.root) / "items" / c.signature(self.ident) / "result.json"
        value = c.read(path)
        value["n_valid"] = 2
        path.write_bytes(c.a.encoded(value))
        with self.assertRaises(c.a.IntegrityError):
            self.audit()

    def test_cross_condition_identity_never_reuses_an_answer(self):
        transport = Transport()
        self.execute(transport)
        self.ident = self.ident | {"condition": "general_2"}
        self.execute(transport)
        self.ident = self.ident | {"repeat": 1}
        self.execute(transport)
        self.assertEqual(transport.calls, 9)

    def test_partial_scan_blocks_exhausted_and_unknown_requests(self):
        job = {"key": self.ident["key"], "condition": self.ident["condition"], "repeat": 0}
        entry = {"messages": MESSAGES, "messages_sha256": c.signature(MESSAGES)}
        with self.assertRaises(c.a.CallStopped):
            self.execute(Transport([c.a.TransportFault("ConnectionError")] * 5))
        with self.assertRaises(c.a.CallStopped):
            r.scan_job(self.root, job, entry, "offline-fixture", self.backend, "mock")
        self.ident = self.ident | {"key": "de/unknown"}
        job = job | {"key": "de/unknown"}
        with self.assertRaises(RuntimeError):
            self.execute(Transport([RuntimeError("fixture interruption")]))
        with self.assertRaises(c.a.IncompleteAttempt):
            r.scan_job(self.root, job, entry, "offline-fixture", self.backend, "mock")

    def test_availability_receipt_with_mock_http_only(self):
        from types import SimpleNamespace
        import requests

        response = SimpleNamespace(
            status_code=200,
            content=json.dumps({"data": [{"id": c.a.MODEL}]}).encode(),
            json=lambda: {"data": [{"id": c.a.MODEL}]},
        )
        transport = SimpleNamespace(
            _endpoint="https://offline.invalid/api/v1/chat/completions", _token="dummy-fixture"
        )
        with patch.object(requests, "get", return_value=response) as fake:
            r.availability(self.root, transport)
        self.assertEqual(fake.call_count, 1)
        self.assertEqual(c.read(self.root / "model_check_response.json")["status"], 200)

    def test_live_mode_rejects_synthetic_transport(self):
        with self.assertRaises(c.a.IntegrityError):
            self.execute(Transport(), execution_mode="live")

    def test_stage_resume_reuses_previously_completed_items(self):
        job = {
            "key": self.ident["key"],
            "condition": "mixed_2",
            "repeat": 0,
            "messages_sha256": c.signature(MESSAGES),
        }
        second = job | {"repeat": 1}
        cat = {
            (job["key"], "mixed_2"): {
                "messages": MESSAGES,
                "messages_sha256": c.signature(MESSAGES),
            }
        }
        stage = {"id": "batch_partial", "jobs": [job, second], "items": 2, "ordinary_calls": 6}
        transport = Transport()
        self.execute(transport)
        r.execute_batch(
            self.root, stage, cat, "offline-fixture", self.backend, "mock", lambda: transport
        )
        r.audit_batch(self.root, stage, cat, "offline-fixture", self.backend, "mock")
        self.assertEqual(transport.calls, 6)

    def test_exclusive_queue_lock(self):
        path = c.a.storage_path(self.root / "queue.lock")
        with c.a.item_lock(path):
            with self.assertRaises(c.a.IntegrityError):
                with c.a.item_lock(path):
                    pass
        with c.a.item_lock(path):
            pass

    def test_offline_guard_blocks_network_and_credentials(self):
        import dotenv
        import requests

        with c.offline():
            for action in [
                lambda: requests.get("https://example.invalid"),
                lambda: socket.create_connection(("localhost", 1)),
                lambda: dotenv.dotenv_values(c.ROOT / ".env"),
                r.core.live_client,
            ]:
                with self.assertRaises(RuntimeError):
                    action()

    def test_exact_dense_bm25_rrf_on_small_fixture(self):
        recs = [
            {
                "id": key,
                "text": text,
                "kind": "guideline",
                "lang": "en",
                "source": "fixture",
                "dimension": "hate",
                "label": None,
                "meta": {},
            }
            for key, text in [("a", "one shared"), ("b", "two shared"), ("c", "three")]
        ]
        selector = p.Guidelines(recs, np.eye(3))
        hits, trace = selector.select("one", np.array([1.0, 0.0, 0.0]), 2)
        self.assertEqual(trace["dense_order"], ["a", "b", "c"])
        self.assertEqual(trace["bm25_order"], ["a"])
        self.assertEqual([h["id"] for h in hits], ["a", "b"])
        self.assertAlmostEqual(trace["fused_scores"]["a"], 2 / 61)
        self.assertEqual(p.tokenize("Größe, N-Wort!"), ["größe", "n", "wort"])

    def test_stop_drains_submitted_jobs_and_never_exceeds_128(self):
        lock = threading.Lock()
        active = 0
        peak = 0

        def work(job):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.1)
            with lock:
                active -= 1
            if job == 0:
                raise c.a.CallStopped("fixture stop")
            return {"result": DECISION}

        outcome = r.core.bounded_jobs(list(range(500)), work, workers=128)
        self.assertTrue(outcome["stopped"])
        self.assertLessEqual(peak, 128)
        self.assertGreater(peak, 100)
        self.assertEqual(active, 0)
        self.assertLess(outcome["submitted"], 500)

    def test_end_to_end_128_worker_batches_audit_and_restart(self):
        cat, jobs = ({}, [])
        for i in range(15):
            key = f"de/mock-{i}"
            for cond in c.CONDITIONS:
                cat[key, cond] = {"messages": MESSAGES, "messages_sha256": c.signature(MESSAGES)}
                for repeat in range(3):
                    jobs.append(
                        {
                            "key": key,
                            "condition": cond,
                            "repeat": repeat,
                            "population": "mock",
                            "messages_sha256": c.signature(MESSAGES),
                        }
                    )
        stage = {
            "id": "batch_mock",
            "jobs": jobs,
            "items": len(jobs),
            "ordinary_calls": len(jobs) * 3,
        }
        transport = Transport(delay=0.003)
        r.execute_batch(
            self.root,
            stage,
            cat,
            "fixture",
            self.backend,
            "mock",
            lambda: transport,
            sleep=lambda _: None,
        )
        r.audit_batch(self.root, stage, cat, "fixture", self.backend, "mock")
        done = r.finalize(self.root, {"stages": [stage]}, "fixture", "mock")
        r.execute_batch(self.root, stage, cat, "fixture", self.backend, "mock", c.forbidden)
        self.assertEqual(r.finalize(self.root, {"stages": [stage]}, "fixture", "mock"), done)
        self.assertEqual(transport.calls, len(jobs) * 3)
        self.assertLessEqual(transport.peak, 128)
        self.assertEqual(done["counts"]["predictions"], 135)


def main():
    with c.offline():
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        record = {
            "passed": result.wasSuccessful(),
            "tests_run": result.testsRun,
            "failures": [str(test) for test, _ in result.failures],
            "errors": [str(test) for test, _ in result.errors],
            "fixture_root": c.ref(TEST_ROOT),
            "source_hashes": c.bindings(c.source_files()),
            "actual_Qwen_calls": 0,
            "actual_model_list_GETs": 0,
            "network_and_credential_guard": True,
        }
        c.save(TEST_ROOT / "test_report.json", record)
        print(
            json.dumps(
                {"report": c.ref(TEST_ROOT / "test_report.json"), "passed": result.wasSuccessful()}
            ),
            flush=True,
        )
        if not result.wasSuccessful():
            raise SystemExit(1)


if __name__ == "__main__":
    main()

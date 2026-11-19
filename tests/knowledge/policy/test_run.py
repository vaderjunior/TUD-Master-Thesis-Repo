"""Tests for policy run."""

from copy import deepcopy
from datetime import timedelta
import json
from pathlib import Path
import socket
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from analysis.knowledge.policy import common as c
from analysis.knowledge.policy import engine as e
from analysis.knowledge.policy import run as r
from analysis.knowledge.policy import verify as verify
from analysis.common import schema as isolated

DECISION = {
    "reasoning": "OFFLINE FIXTURE",
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


def reply(value=DECISION, status=200, model=None, raw=None, finish="stop"):
    body = {
        "model": model or c.a.MODEL,
        "choices": [
            {
                "message": {"content": json.dumps(value) if raw is None else raw},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return c.a.WireReply(status, json.dumps(body).encode(), {"Content-Type": "application/json"})


class Transport:

    def __init__(self, responses=None, delay=0):
        self.responses = responses
        self.delay = delay
        self.calls = 0
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

    def setUp(self):
        self.root = TEST_ROOT / self._testMethodName
        self.root.mkdir(parents=True)
        self.guard = r.RequestGuard(
            0,
            clock=lambda: c.DEADLINE - timedelta(hours=24),
            resource_check=lambda: {},
            token_count=lambda _: 10,
        )
        self.backend = e.adapter(self.guard)
        self.ident = {
            "experiment_sha256": "offline-fixture",
            "condition": "retrieved_policy",
            "repeat": 0,
            "key": "en/fixture",
        }

    def execute(self, transport, **kw):
        return self.backend.execute_item(
            self.root, self.ident, MESSAGES, transport, sleep=lambda _: None, **kw
        )

    def audit(self):
        return self.backend.audit_item(self.root, self.ident, MESSAGES)

    def files(self, name):
        return list(c.a.storage_path(self.root).rglob(name))

    def job(self):
        return {
            "key": self.ident["key"],
            "condition": self.ident["condition"],
            "repeat": self.ident["repeat"],
            "messages_sha256": c.signature(MESSAGES),
        }

    def entry(self):
        return {"messages": MESSAGES, "messages_sha256": c.signature(MESSAGES)}

    def test_complete_restart_uses_no_transport_or_guards(self):
        t = Transport()
        first = self.execute(t)
        self.assertEqual(t.calls, 3)
        blocked = e.adapter(c.forbidden)
        self.assertEqual(blocked.execute_item(self.root, self.ident, MESSAGES, c.forbidden), first)
        self.assertEqual(self.audit()["status"], "verified_from_saved_responses")

    def test_saved_outcome_resumes_without_repeating_request(self):
        real = self.backend.execute_item.__globals__["save_new"]
        failed = False

        def save(path, obj):
            nonlocal failed
            if Path(path).name == "0.json" and Path(path).parent.name == "votes" and (not failed):
                failed = True
                raise RuntimeError("fixture stop before vote commit")
            return real(path, obj)

        t = Transport()
        with patch.dict(self.backend.execute_item.__globals__, save_new=save):
            with self.assertRaises(RuntimeError):
                self.execute(t)
        self.assertEqual(t.calls, 1)
        self.assertEqual(self.execute(t)["n_valid"], 3)
        self.assertEqual(t.calls, 3)
        self.audit()

    def test_unknown_response_stops_without_resending(self):
        t = Transport([RuntimeError("fixture process death")])
        with self.assertRaises(RuntimeError):
            self.execute(t)
        with self.assertRaises(c.a.IncompleteAttempt):
            self.execute(c.forbidden)
        with self.assertRaises(c.a.IncompleteAttempt):
            r.scan_job(self.root, self.job(), self.entry(), "offline-fixture", self.backend, "mock")
        self.assertEqual(t.calls, 1)

    def test_retry_budget_and_backoff_are_never_reset(self):
        t = Transport([c.a.TransportFault("ConnectionError")] * 5)
        waits = []
        with self.assertRaises(c.a.CallStopped):
            self.backend.execute_item(self.root, self.ident, MESSAGES, t, sleep=waits.append)
        self.assertEqual(waits, [2, 2, 4, 8])
        self.assertEqual(t.calls, 5)
        with self.assertRaises(c.a.CallStopped):
            self.execute(c.forbidden)
        with self.assertRaises(c.a.CallStopped):
            r.scan_job(self.root, self.job(), self.entry(), "offline-fixture", self.backend, "mock")

    def test_transient_errors_reuse_saved_attempts(self):
        t = Transport(
            [c.a.TransportFault("ReadTimeout"), reply(status=503), reply(), reply(), reply()]
        )
        out = self.execute(t)
        self.assertEqual(out["accounting"]["post_attempts"], 5)
        self.assertEqual(out["accounting"]["retry_attempts"], 2)
        self.audit()
        self.execute(c.forbidden)

    def test_deadline_guard_creates_no_uncertain_intent(self):
        self.backend = e.adapter(
            r.RequestGuard(
                0, clock=lambda: c.DEADLINE, resource_check=lambda: {}, token_count=lambda _: 10
            )
        )
        t = Transport()
        with self.assertRaises(e.PausedBeforeRequest):
            self.execute(t)
        self.assertEqual(t.calls, 0)
        self.assertFalse(self.files("intent.json"))
        self.assertIsNone(
            r.scan_job(self.root, self.job(), self.entry(), "offline-fixture", self.backend, "mock")
        )
        self.backend = e.adapter(self.guard)
        self.execute(t)
        self.assertEqual(t.calls, 3)
        self.audit()

    def test_deadline_rechecked_after_retry_backoff(self):
        current = [c.DEADLINE - timedelta(seconds=1)]
        self.backend = e.adapter(
            r.RequestGuard(
                0, clock=lambda: current[0], resource_check=lambda: {}, token_count=lambda _: 10
            )
        )
        t = Transport([c.a.TransportFault("ConnectionError"), reply(), reply(), reply()])

        def wait(_):
            current[0] = c.DEADLINE

        with self.assertRaises(e.PausedBeforeRequest):
            self.backend.execute_item(self.root, self.ident, MESSAGES, t, sleep=wait)
        self.assertEqual(t.calls, 1)
        self.assertEqual(len(self.files("intent.json")), 1)
        current[0] = c.DEADLINE - timedelta(hours=1)
        self.execute(t)
        self.assertEqual(t.calls, 4)
        self.audit()

    def test_ram_pause_preserves_completed_vote(self):
        available = [True]

        def resources():
            if not available[0]:
                raise RuntimeError("fixture low RAM")
            return {}

        guard = r.RequestGuard(
            0,
            clock=lambda: c.DEADLINE - timedelta(days=1),
            resource_check=resources,
            token_count=lambda _: 10,
        )
        self.backend = e.adapter(guard)
        t = Transport()

        def transport(payload):
            value = t(payload)
            available[0] = False
            guard.last_check = -float("inf")
            return value

        with self.assertRaises(e.PausedBeforeRequest):
            self.execute(transport)
        self.assertEqual(t.calls, 1)
        self.assertEqual(len(self.files("intent.json")), 1)
        available[0] = True
        self.execute(t)
        self.assertEqual(t.calls, 3)
        self.audit()

    def test_initial_input_limit_never_sends_or_truncates(self):
        self.backend = e.adapter(
            r.RequestGuard(
                0,
                clock=lambda: c.DEADLINE - timedelta(days=1),
                resource_check=lambda: {},
                token_count=lambda _: 8193,
            )
        )
        t = Transport()
        with self.assertRaises(e.PausedBeforeRequest):
            self.execute(t)
        self.assertEqual(t.calls, 0)
        self.assertFalse(self.files("intent.json"))

    def test_repair_input_limit_keeps_original_reply(self):
        guard = r.RequestGuard(
            0,
            clock=lambda: c.DEADLINE - timedelta(days=1),
            resource_check=lambda: {},
            token_count=lambda p: 9000 if len(p["messages"]) > 2 else 10,
        )
        self.backend = e.adapter(guard)
        t = Transport([reply(raw="not JSON")])
        with self.assertRaises(e.PausedBeforeRequest):
            self.execute(t)
        self.assertEqual(t.calls, 1)
        self.assertEqual(len(self.files("outcome.json")), 1)
        self.assertEqual(len(self.files("intent.json")), 1)

    def test_reported_context_limit_includes_output_allowance(self):
        self.guard.context_limit = 4096
        with self.assertRaises(e.PausedBeforeRequest):
            self.execute(Transport())
        self.assertFalse(self.files("intent.json"))

    def test_repair_conversation_tokenizer_matches_official_template(self):
        from analysis.knowledge.policy import setup as setup

        if not (setup.TOKENIZER / "tokenizer.json").exists():
            self.skipTest("Original tokenizer assets are not available")
        expected = setup.counter()
        actual = r.RequestGuard(0, resource_check=lambda: {})
        for msgs in (
            MESSAGES,
            MESSAGES
            + [
                {"role": "assistant", "text": "broken { answer ä ß"},
                {"role": "user", "text": "Please repair the JSON."},
            ],
            MESSAGES
            + [
                {"role": "assistant", "text": "<think>test</think>\n```json\n{}\n```"},
                {"role": "user", "text": "repair again"},
            ],
        ):
            self.assertEqual(actual.count_tokens(c.a.body(msgs)), expected(msgs))

    def test_guarded_parser_and_voter_match_unchanged_engine(self):
        original = isolated.adapter(c.ROOT / "config/taxonomy.yaml")
        positive = DECISION | {
            "hate": True,
            "hate_type": ["explicit"],
            "severity": "high",
            "legal": ["propaganda"],
        }
        fixtures = [
            reply(positive),
            reply(raw="bad"),
            reply(DECISION | {"legal": ["propaganda"]}),
            reply(positive),
        ]
        first = self.execute(Transport(fixtures))
        second = original.execute_item(
            self.root / "original", self.ident, MESSAGES, Transport(fixtures), sleep=lambda _: None
        )
        for field in (
            "result",
            "n_valid",
            "n_runs",
            "uncertain",
            "agreement",
            "raw_runs",
            "run_predictions",
            "parse_failures",
            "repairs",
            "normalisations",
            "gate_normalised",
            "accounting",
        ):
            self.assertEqual(first[field], second[field])
        self.audit()
        original.audit_item(self.root / "original", self.ident, MESSAGES)

    def test_all_invalid_votes_are_missing_not_negative(self):
        out = self.execute(Transport([reply(raw="bad")] * 9))
        self.assertIsNone(out["result"])
        self.assertEqual(out["n_valid"], 0)
        self.assertEqual(out["repairs"], 6)
        self.assertIsNone(r.independent_decision(out["run_predictions"]))
        self.audit()

    def test_one_valid_vote_retains_original_semantics(self):
        out = self.execute(
            Transport([reply(DECISION | {"legal": ["propaganda"]})] + [reply(raw="bad")] * 6)
        )
        self.assertEqual(out["n_valid"], 1)
        self.assertEqual(out["result"]["legal"], ["propaganda"])
        self.assertEqual(out["repairs"], 4)
        self.audit()

    def test_gate_and_legal_independence(self):
        value = DECISION | {
            "legal": ["insult_defamation"],
            "hate_type": ["explicit"],
            "severity": "high",
        }
        out = self.execute(Transport([reply(value)] * 3))
        self.assertFalse(out["result"]["hate"])
        self.assertEqual(out["result"]["hate_type"], [])
        self.assertIsNone(out["result"]["severity"])
        self.assertEqual(out["result"]["legal"], ["insult_defamation"])
        self.audit()

    def test_gate_tie_and_lower_median(self):
        positive = DECISION | {"hate": True, "hate_type": ["explicit"], "severity": "high"}
        out = self.execute(
            Transport(
                [reply(positive), reply(), reply(raw="bad"), reply(raw="bad"), reply(raw="bad")]
            )
        )
        self.assertEqual(out["n_valid"], 2)
        self.assertTrue(out["uncertain"])
        self.assertFalse(out["result"]["hate"])
        self.audit()
        self.ident = self.ident | {"key": "en/median"}
        out = self.execute(
            Transport(
                [
                    reply(positive | {"severity": "low"}),
                    reply(positive),
                    reply(raw="bad"),
                    reply(raw="bad"),
                    reply(raw="bad"),
                ]
            )
        )
        self.assertEqual(out["result"]["severity"], "low")
        self.audit()

    def test_historical_duplicate_label_semantics_are_retained(self):
        one = DECISION | {"legal": ["propaganda", "propaganda"]}
        out = self.execute(Transport([reply(one), reply(), reply()]))
        self.assertEqual(out["result"]["legal"], ["propaganda"])
        self.audit()

    def test_authentication_wrong_model_and_truncation_stop(self):
        for i, value in enumerate(
            (reply(status=401), reply(model="different-model"), reply(finish="length"))
        ):
            self.ident = self.ident | {"key": f"en/stop-{i}"}
            t = Transport([value])
            with self.assertRaises(c.a.CallStopped):
                self.execute(t)
            self.assertEqual(t.calls, 1)
            with self.assertRaises(c.a.CallStopped):
                self.execute(c.forbidden)

    def test_live_mode_rejects_mock_transport(self):
        with self.assertRaises(c.a.IntegrityError):
            self.execute(Transport(), execution_mode="live")

    def test_cross_condition_and_repeat_do_not_share_answers(self):
        t = Transport()
        self.execute(t)
        self.ident = self.ident | {"condition": "complete_policy"}
        self.execute(t)
        self.ident = self.ident | {"repeat": 1}
        self.execute(t)
        self.assertEqual(t.calls, 9)

    def test_result_and_raw_tampering_are_detected(self):
        self.execute(Transport())
        folder = r.item_dir(self.root, self.job(), "offline-fixture")
        p = folder / "result.json"
        data = c.read(p)
        data["n_valid"] = 2
        p.write_bytes(c.a.encoded(data))
        with self.assertRaises(c.a.IntegrityError):
            self.audit()
        self.ident = self.ident | {"key": "en/raw"}
        self.execute(Transport())
        p = next(r.item_dir(self.root, self.job(), "offline-fixture").rglob("outcome.json"))
        data = c.read(p)
        data["elapsed_s"] += 1
        p.write_bytes(c.a.encoded(data))
        with self.assertRaises(c.a.IntegrityError):
            self.audit()

    def test_existing_conflicting_file_is_preserved(self):
        path = self.root / "immutable.json"
        c.save(path, {"a": 1})
        before = path.read_bytes()
        with self.assertRaises(c.a.IntegrityError):
            c.save(path, {"a": 2})
        self.assertEqual(path.read_bytes(), before)

    def test_long_path_roundtrip(self):
        self.root = self.root / ("a" * 70) / ("b" * 70) / ("c" * 70)
        self.execute(Transport())
        self.audit()

    def test_global_and_study_locks_are_exclusive(self):
        with patch.object(c, "OUTPUT", self.root / "output"), patch.object(
            r.host, "SHARED", self.root / "shared"
        ):
            with r.locks():
                with self.assertRaises(c.a.IntegrityError):
                    with r.locks():
                        pass
            with r.locks():
                pass

    def test_latest_start_only_blocks_new_invocations(self):
        r.launch_window(False, c.LATEST_START)
        with self.assertRaises(c.a.IntegrityError):
            r.launch_window(False, c.LATEST_START + timedelta(seconds=1))
        r.launch_window(True, c.LATEST_START + timedelta(hours=1))
        with self.assertRaises(c.a.IntegrityError):
            r.launch_window(True, c.DEADLINE)

    def test_disk_guard_and_initial_vs_running_ram(self):
        with patch.object(r, "verify_storage"), patch.object(
            r.host, "resources", return_value={}
        ) as ram, patch.object(
            r.shutil, "disk_usage", return_value=SimpleNamespace(free=100 * 2**30)
        ):
            out = r.resources(62154, initial=True)
            self.assertAlmostEqual(out["D_required_GiB"], 68.697265625)
            ram.assert_called_with(2.0, 8.0)
            r.resources(1)
            ram.assert_called_with(0.75, 8.0)
        with patch.object(r, "verify_storage"), patch.object(
            r.host, "resources", return_value={}
        ), patch.object(r.shutil, "disk_usage", return_value=SimpleNamespace(free=8 * 2**30)):
            with self.assertRaises(c.a.IntegrityError):
                r.resources(1)

    def test_model_availability_uses_only_mock_http(self):
        import requests

        value = {"data": [{"id": c.a.MODEL, "max_model_len": 16384}]}
        response = SimpleNamespace(
            status_code=200, content=json.dumps(value).encode(), json=lambda: value
        )
        transport = SimpleNamespace(
            _endpoint="https://offline.invalid/api/v1/chat/completions", _token="dummy-fixture"
        )
        with patch.object(requests, "get", return_value=response) as get, patch.object(
            r, "launch_window"
        ):
            r.availability(self.root, transport, self.guard)
        self.assertEqual(get.call_count, 1)
        self.assertEqual(self.guard.context_limit, 16384)

    def test_unavailable_model_stops_without_fallback(self):
        import requests

        value = {"data": [{"id": "another-model"}]}
        response = SimpleNamespace(
            status_code=200, content=json.dumps(value).encode(), json=lambda: value
        )
        transport = SimpleNamespace(
            _endpoint="https://offline.invalid/api/v1/chat/completions", _token="dummy-fixture"
        )
        with patch.object(requests, "get", return_value=response), patch.object(r, "launch_window"):
            with self.assertRaises(c.a.IntegrityError):
                r.availability(self.root, transport, self.guard)
        self.assertFalse((self.root / "availability_ready.json").exists())

    def test_offline_guard_blocks_credentials_and_network(self):
        import requests, dotenv

        with c.offline():
            for call in (
                lambda: requests.get("https://offline.invalid"),
                lambda: socket.create_connection(("offline.invalid", 443)),
                lambda: dotenv.dotenv_values(c.ROOT / ".env"),
                r.core.live_client,
            ):
                with self.assertRaises(RuntimeError):
                    call()

    def make_stage(self, n=15):
        jobs = []
        cat = {}
        for i in range(n):
            key = f"en/mock-{i}"
            for rep in range(3):
                for cond in c.CONDITIONS:
                    cat[key, cond] = self.entry()
                    jobs.append(
                        {
                            "key": key,
                            "condition": cond,
                            "repeat": rep,
                            "population": "mock",
                            "messages_sha256": c.signature(MESSAGES),
                        }
                    )
        return (
            {
                "id": "batch_fixture",
                "input_keys": [f"en/mock-{i}" for i in range(n)],
                "jobs": jobs,
                "items": len(jobs),
                "ordinary_calls": len(jobs) * 3,
            },
            cat,
        )

    def test_128_worker_end_to_end_audit_final_check_and_restart(self):
        stage, cat = self.make_stage()
        t = Transport(delay=0.004)
        r.execute_batch(
            self.root, stage, cat, "fixture", self.backend, "mock", lambda: t, sleep=lambda _: None
        )
        r.audit_batch(self.root, stage, cat, "fixture", self.backend, "mock")
        done = r.finalize(self.root, {"stages": [stage]}, "fixture", "mock")
        r.execute_batch(self.root, stage, cat, "fixture", self.backend, "mock", c.forbidden)
        self.assertEqual(r.finalize(self.root, {"stages": [stage]}, "fixture", "mock"), done)
        self.assertEqual(t.calls, 405)
        self.assertLessEqual(t.peak, 128)
        self.assertGreater(t.peak, 1)
        self.assertEqual(done["counts"]["predictions"], 135)

    def test_partial_batch_reuses_saved_results(self):
        stage, cat = self.make_stage(n=1)
        t = Transport()
        job = stage["jobs"][0]
        self.backend.execute_item(
            self.root, r.identity(job, "fixture"), MESSAGES, t, sleep=lambda _: None
        )
        r.execute_batch(self.root, stage, cat, "fixture", self.backend, "mock", lambda: t)
        r.audit_batch(self.root, stage, cat, "fixture", self.backend, "mock")
        self.assertEqual(t.calls, 27)

    def test_final_check_detects_late_raw_changes(self):
        stage, cat = self.make_stage(n=1)
        t = Transport()
        r.execute_batch(self.root, stage, cat, "fixture", self.backend, "mock", lambda: t)
        r.audit_batch(self.root, stage, cat, "fixture", self.backend, "mock")
        raw = self.files("outcome.json")[0]
        value = c.read(raw)
        value["elapsed_s"] += 1
        raw.write_bytes(c.a.encoded(value))
        with self.assertRaises(c.a.IntegrityError):
            r.finalize(self.root, {"stages": [stage]}, "fixture", "mock")

    def test_final_check_detects_unplanned_items(self):
        stage, cat = self.make_stage(n=1)
        t = Transport()
        r.execute_batch(self.root, stage, cat, "fixture", self.backend, "mock", lambda: t)
        r.audit_batch(self.root, stage, cat, "fixture", self.backend, "mock")
        (self.root / "items/unplanned").mkdir()
        with self.assertRaises(c.a.IntegrityError):
            r.finalize(self.root, {"stages": [stage]}, "fixture", "mock")

    def test_process_quality_stop_is_sticky(self):
        stage, cat = self.make_stage(n=5)
        t = Transport([reply(raw="bad")] * 405)
        with self.assertRaises(c.a.IntegrityError):
            r.execute_batch(
                self.root,
                stage,
                cat,
                "fixture",
                self.backend,
                "mock",
                lambda: t,
                sleep=lambda _: None,
            )
        self.assertTrue((self.root / "batches/batch_fixture/review_required.json").exists())
        with self.assertRaises(c.a.IntegrityError):
            r.execute_batch(self.root, stage, cat, "fixture", self.backend, "mock", c.forbidden)

    def test_scheduler_stops_new_submissions_and_drains_128(self):
        active = 0
        peak = 0
        lock = threading.Lock()
        barrier = threading.Barrier(128, timeout=10)

        def work(job):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            if job < 128:
                barrier.wait()
            time.sleep(0.03)
            with lock:
                active -= 1
            if job == 0:
                raise c.a.CallStopped("offline fixture stop")
            return {"result": DECISION}

        outcome = r.core.bounded_jobs(list(range(1000)), work, workers=128)
        self.assertTrue(outcome["stopped"])
        self.assertEqual(peak, 128)
        self.assertEqual(active, 0)
        self.assertLess(outcome["submitted"], 1000)

    def test_schedule_tampering_is_detected(self):
        if not (c.BASE / "schedule.json").exists():
            self.skipTest("Original archived input is not available")
        schedule = c.read(c.BASE / "schedule.json")
        index = c.read(c.BASE / "prompt_index.json")
        verify.check_schedule(schedule, index)
        altered = deepcopy(schedule)
        altered["stages"][0]["jobs"][0]["repeat"] = 2
        with self.assertRaises(c.a.IntegrityError):
            verify.check_schedule(altered, index)
        altered = deepcopy(schedule)
        altered["stages"][0]["jobs"].reverse()
        with self.assertRaises(c.a.IntegrityError):
            verify.check_schedule(altered, index)


def main():
    with c.offline():
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
        )
        report = {
            "passed": result.wasSuccessful(),
            "tests_run": result.testsRun,
            "failures": [str(t) for t, _ in result.failures],
            "errors": [str(t) for t, _ in result.errors],
            "fixture_root": c.ref(TEST_ROOT),
            "source_hashes": c.bindings(c.source_files()),
            "actual_Qwen_calls": 0,
            "actual_model_list_GETs": 0,
            "credentials_read": False,
            "network_and_credential_guard": True,
        }
        c.save(TEST_ROOT / "test_report.json", report)
        print(
            json.dumps(
                {
                    "report": c.ref(TEST_ROOT / "test_report.json"),
                    "passed": report["passed"],
                    "tests": result.testsRun,
                }
            ),
            flush=True,
        )
        if not result.wasSuccessful():
            raise SystemExit(1)


if __name__ == "__main__":
    main()

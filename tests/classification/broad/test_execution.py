from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import socket
import threading
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch
from analysis.classification.broad import execution as r
from tests.common import test_execution as inherited

a, e, c = (r.a, r.e, r.c)
TEST_ROOT = r.BASE / "tests" / str(time.time_ns())
AMENDMENT = dict(
    path=r.rel(TEST_ROOT / "synthetic_manifest.json"), sha256="synthetic-only"
)


class RuntimeTests(unittest.TestCase):

    def test_private_stage_change_only_and_global_limit_unchanged(self):
        original = e.queue("matched")
        code = original.execute_stage.__code__
        bound = r.bind_stage(original, AMENDMENT)
        self.assertIs(bound.execute_stage.__code__, code)
        self.assertEqual(bound.execute_stage.__globals__["WORKERS"], 128)
        self.assertEqual(e.queue("matched").WORKERS, 8)
        self.assertEqual(e.template.WORKERS, 8)
        with self.assertRaises(a.IntegrityError):
            r.bind_stage(bound, AMENDMENT)

    def test_actual_128_workers_and_no_129th_submission(self):
        gate = threading.Barrier(128, timeout=40)
        lock = threading.Lock()
        active = 0
        peak = 0

        def work(number):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            gate.wait()
            with lock:
                active -= 1
            return {"result": {"hate": False}}

        result = e.core.bounded_jobs(list(range(256)), work, 128)
        self.assertEqual(
            (peak, result["peak_submitted"], result["submitted"]), (128, 128, 256)
        )
        self.assertFalse(result["errors"])
        self.assertEqual(len(result["results"]), 256)

    def test_first_error_drains_bounded_inflight_and_preserves_unsubmitted(self):
        gate = threading.Barrier(128, timeout=40)
        exited = []

        def work(number):
            gate.wait()
            if number == 0:
                raise a.CallStopped("synthetic stop")
            time.sleep(0.15)
            exited.append(number)
            return {"result": {"hate": False}}

        result = e.core.bounded_jobs(list(range(400)), work, 128)
        self.assertTrue(result["stopped"])
        self.assertEqual(result["submitted"], 128)
        self.assertEqual(result["unsubmitted"], 272)
        self.assertEqual(len(exited), 127)

    def test_stream_reader_exact_utf8_whitespace_and_empty_lines(self):
        path = TEST_ROOT / "rows.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        values = [{"text": " ", "unicode": "Grüße"}, {"text": "\n\t"}]
        path.write_text(
            "\n".join((json.dumps(v, ensure_ascii=False) for v in values)) + "\n\n",
            encoding="utf8",
        )
        self.assertEqual(list(r.stream_rows(path)), c.rows(path))

    def test_stream_catalogue_keeps_old_function_globals_intact(self):
        old_rows = c.rows
        self.assertIs(e.load_catalog.__globals__["c"], c)
        self.assertIs(c.rows, old_rows)

    def test_long_path_normalization_and_escape_rejection(self):
        path = TEST_ROOT / ("a" * 120) / ("b" * 120) / "receipt.json"
        r.save(path, {"fixture": True})
        self.assertEqual(r.rel(path), r.rel(a.storage_path(path)))
        self.assertEqual(a.read(path), {"fixture": True})
        with self.assertRaises(ValueError):
            r.rel(c.ROOT.parent / "not-in-workspace")
        with self.assertRaises(a.IntegrityError):
            r.save(c.ROOT / "must-not-be-written.json", {})

    def test_same_manifest_cannot_overwrite_old_output(self):
        path = TEST_ROOT / "immutable.json"
        r.save(path, {"value": 1})
        r.save(path, {"value": 1})
        with self.assertRaises(a.IntegrityError):
            r.save(path, {"value": 2})

    def test_shared_and_both_legacy_study_locks_are_required(self):
        paths = r.lock_paths()
        self.assertEqual(len(paths), 3)
        self.assertIn(r.resources.SHARED / "qwen_queue.lock", paths)
        self.assertEqual({p.parent.name for p in paths[1:]}, {"matched", "broad"})
        dummy = TEST_ROOT / "queue.lock"
        with patch.object(r, "lock_paths", return_value=[dummy]):
            with r.queue_locks():
                with self.assertRaises(a.IntegrityError), r.queue_locks():
                    pass
            with r.queue_locks():
                pass

    def test_lock_conflict_blocks_run_before_credentials(self):
        dummy = TEST_ROOT / "conflict.lock"
        with patch.object(r, "lock_paths", return_value=[dummy]), patch.object(
            r, "validate"
        ) as validation:
            with r.queue_locks():
                with self.assertRaises(a.IntegrityError):
                    r.run("matched")
            validation.assert_not_called()

    def test_validation_failure_precedes_catalogue_and_live_client(self):
        with patch.object(
            r, "lock_paths", return_value=[TEST_ROOT / "validation.lock"]
        ), patch.object(
            r, "validate", side_effect=RuntimeError("synthetic readiness failure")
        ), patch.object(
            r, "load_catalog"
        ) as catalogue:
            with self.assertRaises(RuntimeError):
                r.run("matched")
            catalogue.assert_not_called()

    def test_all_offline_network_paths_and_live_factory_blocked(self):
        import requests

        with r.offline():
            for invoke in [
                lambda: socket.create_connection(("example.invalid", 443)),
                lambda: socket.socket().connect_ex(("127.0.0.1", 1)),
                lambda: requests.get("https://example.invalid"),
                lambda: e.core.live_client(),
            ]:
                with self.assertRaises(RuntimeError):
                    invoke()

    def test_disk_allowance_counts_all_remaining_and_reserve(self):
        self.assertEqual(r.disk_needed(0), 8)
        self.assertAlmostEqual(r.disk_needed(231342), 36.239990234375)
        with self.assertRaises(a.IntegrityError):
            r.disk_needed(-1)

    def test_host_guards_keep_original_initial_ram_and_after_load_reserve(self):
        with patch.object(
            r, "DEADLINE", datetime.now().astimezone() + timedelta(hours=1)
        ), patch.object(r.resources, "resources", return_value={}) as check:
            r.host_check(100, initial=True)
            check.assert_called_with(2.5, r.disk_needed(100))
            r.host_check(80)
            check.assert_called_with(0.75, r.disk_needed(80))
        with patch.object(
            r, "DEADLINE", datetime.now().astimezone() + timedelta(hours=1)
        ), patch.object(
            r.resources, "resources", side_effect=RuntimeError("low memory")
        ):
            with self.assertRaises(RuntimeError):
                r.host_check(100)

    def test_original_plan_hashes_and_total_budget(self):
        if not all(path.is_file() for path in ARCHIVE_PLANS.values()):
            raise unittest.SkipTest("Original matched/broad execution plans are absent")
        for study, expected in r.PLANS.items():
            self.assertEqual(c.sha(ARCHIVE_PLANS[study]), expected)
        self.assertEqual(sum((size[2] * 3 for size in e.SIZES.values())), 694026)


ARCHIVE_PLANS = {study: e.queue(study).PLAN for study in r.PLANS}


def run_tests():
    suite = unittest.TestSuite()
    fixture_roots = []
    for study in r.PLANS:
        b = r.bind_stage(e.queue(study), AMENDMENT)
        tests, root = inherited.inherited_suite(b)
        suite.addTests(tests)
        fixture_roots.append(r.rel(root))
    suite.addTests(
        unittest.defaultTestLoader.loadTestsFromTestCase(inherited.ScienceTests)
    )
    suite.addTests(unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeTests))
    with r.offline():
        outcome = unittest.TextTestRunner(verbosity=2).run(suite)
    record = dict(
        passed=outcome.wasSuccessful(),
        tests_run=outcome.testsRun,
        errors=len(outcome.errors),
        failures=len(outcome.failures),
        source_hashes={r.rel(p): c.sha(p) for p in r.sources()},
        environment=e.core.environment(),
        preserved_fixtures=[r.rel(TEST_ROOT), *fixture_roots],
        actual_Qwen_calls=0,
        actual_model_list_GETs=0,
        training_fits=0,
    )
    r.save(TEST_ROOT / "result.json", record)
    if not outcome.wasSuccessful():
        raise SystemExit(1)
    r.save(r.BASE / "tests.json", record)


if __name__ == "__main__":
    run_tests()

"""Tests for common batch."""

from collections import Counter
import threading
import time
import unittest
from analysis.common import batch as b


class ScheduleTests(unittest.TestCase):

    def fixture(self):
        dev = []
        cat = {}
        for subset, lang, n, conditions in [
            ("en_dev_eval_main", "en", 8, 6),
            ("de_legal_dev_eval", "de", 7, 9),
        ]:
            for i in range(n):
                key = f"{lang}/{i}"
                dev.append({"key": key, "scopes": [{"subset": subset}]})
                for c in range(conditions):
                    cat[key, f"condition{c}"] = {}
        return (dev, cat)

    def test_budget_exact_coverage_and_each_comparison_stays_together(self):
        dev, cat = self.fixture()
        stages = b.make_schedule(dev, cat, cap=405)
        seen = {}
        identities = []
        for stage in stages:
            self.assertLessEqual(stage["ordinary_responses"], 405)
            self.assertEqual(stage["ordinary_responses"], stage["items"] * 3)
            for job in stage["jobs"]:
                seen.setdefault(job["key"], set()).add(stage["id"])
                identities.append((job["key"], job["condition"], job["repeat"]))
        self.assertTrue(all((len(x) == 1 for x in seen.values())))
        self.assertEqual(len(identities), len(set(identities)))
        self.assertEqual(
            set(identities), {(key, condition, r) for key, condition in cat for r in range(3)}
        )
        self.assertEqual(stages, b.make_schedule(dev, cat, cap=405))

    def test_cap_too_small_refused(self):
        dev, cat = self.fixture()
        with self.assertRaises(b.a.IntegrityError):
            b.make_schedule(dev, cat, cap=20)

    def test_missing_condition_refused(self):
        dev, cat = self.fixture()
        cat.pop(next(iter(cat)))
        with self.assertRaises(b.a.IntegrityError):
            b.make_schedule(dev, cat)

    def test_bounded_in_flight_and_full_completion(self):
        barrier = threading.Barrier(4)
        active = [0]
        peak = [0]
        lock = threading.Lock()

        def work(job):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            barrier.wait(timeout=10)
            with lock:
                active[0] -= 1
            return {"result": {"hate": False}}

        out = b.bounded_jobs(list(range(24)), work, workers=4)
        self.assertFalse(out["stopped"])
        self.assertEqual(out["submitted"], 24)
        self.assertEqual(out["peak_submitted"], 4)
        self.assertEqual(peak[0], 4)

    def test_first_failure_stops_new_submissions_and_drains(self):
        barrier = threading.Barrier(4)

        def work(i):
            barrier.wait(timeout=10)
            if i == 0:
                raise b.a.CallStopped("authentication")
            time.sleep(0.03)
            return {"result": {"hate": False}}

        out = b.bounded_jobs(list(range(100)), work, workers=4)
        self.assertTrue(out["stopped"])
        self.assertEqual(out["submitted"], 4)
        self.assertEqual(len(out["results"]), 3)
        self.assertEqual(len(out["errors"]), 1)

    def test_null_guard_preserves_null_results(self):
        out = b.bounded_jobs(list(range(100)), lambda _: {"result": None}, workers=1)
        self.assertEqual(out["submitted"], 20)
        self.assertTrue(out["stopped"])
        self.assertEqual(out["stats"]["null"], 20)
        self.assertEqual(len(out["results"]), 20)

    def test_guard_includes_prior_completed_items(self):
        out = b.bounded_jobs(
            list(range(100)),
            lambda _: {"result": None},
            workers=1,
            initial={"complete": 19, "null": 1},
        )
        self.assertEqual(out["submitted"], 1)
        self.assertTrue(out["stopped"])

    def test_no_hidden_next_stage(self):
        out = b.bounded_jobs([1, 2], lambda _: {"result": {}}, workers=2)
        self.assertEqual(out["submitted"], 2)
        self.assertEqual(out["unsubmitted"], 0)


if __name__ == "__main__":
    unittest.main()

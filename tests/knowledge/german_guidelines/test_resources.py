"""Tests for german guidelines resources."""

import json
import time
import unittest
import numpy as np
from analysis.knowledge.german_guidelines import resources as x


class Checks(unittest.TestCase):

    def test_missing_usage_not_zero(self):
        self.assertEqual(x.usage({"usage": None}), [-1, -1, -1])

    def test_partial_usage(self):
        self.assertEqual(x.usage({"usage": {"prompt_tokens": 7}}), [7, -1, -1])

    def test_usage_arithmetic(self):
        self.assertEqual(
            x.usage({"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}}),
            [7, 3, 10],
        )
        with self.assertRaises(ValueError):
            x.usage({"usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 11}})

    def test_bool_and_negative_usage(self):
        for v in (True, -1, 2.5):
            with self.assertRaises(ValueError):
                x.usage({"usage": {"prompt_tokens": v}})

    def test_empty_statistics(self):
        self.assertIsNone(x.stats([])["mean"])

    def test_quantiles(self):
        self.assertAlmostEqual(x.stats([1, 2, 3, 8])["p95"], 7.25)

    def test_bad_statistics(self):
        with self.assertRaises(ValueError):
            x.stats([float("nan")])

    def test_wall_and_gaps(self):
        s = x.wall_intervals([(20, 25), (0, 10)])
        self.assertEqual(s["attempt_seconds"], 15)
        self.assertEqual(s["between_attempt_seconds"], 10)

    def test_overlap_rejected(self):
        with self.assertRaises(ValueError):
            x.wall_intervals([(0, 10), (9, 20)])

    def test_failed_post_and_repair(self):
        meta = [
            {"n_valid": 2, "repairs": 1, "request_chain_s": 15.0, "vote_chain_s": [10.0, 3.0, 2.0]}
        ]
        a = np.array(
            [
                [0, 0, 0, 0, 2.0, 0.0, -1, -1, -1],
                [0, 0, 1, 1, 3.0, 1.0, 10, 2, 12],
                [0, 1, 1, 0, 4.0, 0.0, 15, 3, 18],
                [0, 0, 1, 0, 3.0, 0.0, 10, 2, 12],
                [0, 0, 1, 0, 2.0, 0.0, 10, 2, 12],
            ],
            float,
        )
        z = x.group_stats(meta, a, [0])
        self.assertEqual(z["physical_POST_attempts"], 5)
        self.assertEqual(z["repair_logical_calls"], 1)
        self.assertEqual(z["usage"]["all_success"]["total_tokens"]["sum"], 54)
        self.assertEqual(z["failed_attempts"], 1)
        self.assertEqual(z["backoff_seconds"], 1)
        self.assertEqual(z["successful_latency_s"]["n"], 4)

    def test_offline_guard(self):
        import socket

        with x.offline():
            with self.assertRaises(Exception):
                socket.create_connection(("example.com", 443))

    def test_write_guard(self):
        with self.assertRaises(ValueError):
            x.save(x.c.ROOT / "outside_part7.json", {})


if __name__ == "__main__":
    with x.offline():
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        path = x.BASE / "tests" / f"tests_{time.time_ns()}" / "report.json"
        x.save(
            path,
            {
                "passed": result.wasSuccessful(),
                "tests": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
                "sources": x.c.bindings(x.sources()),
                "actual_Qwen_calls": 0,
            },
        )
        print(json.dumps({"test_report": x.c.ref(path), "passed": result.wasSuccessful()}))
        if not result.wasSuccessful():
            raise SystemExit(1)

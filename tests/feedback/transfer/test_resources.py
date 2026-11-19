import unittest
import numpy as np
from analysis.feedback.transfer import resources_metrics as k
from analysis.feedback.reserved import resources_metrics as old


class Checks(unittest.TestCase):
    def test_reference_once(self):
        x = np.ones((2, 4, 3, 8))
        x[:, 1:] *= 2
        y = k.paired(x)
        np.testing.assert_array_equal(y[:, :, 0], 1)
        np.testing.assert_array_equal(y[:, :, 1], 2)
        np.testing.assert_array_equal(y[:, :, 2:], 1)

    def test_pool_repeat_weights(self):
        x = np.zeros((1, 4, 3, 8))
        x[:, 1, 0] = 9
        self.assertEqual(k.paired(x)[0, 0, 1], 1)
        self.assertEqual(k.paired(x)[0, 0, 3], 3)

    def test_bad_shape(self):
        with self.assertRaises(ValueError):
            k.paired(np.zeros((2, 4, 8)))

    def test_bad_negative(self):
        with self.assertRaises(ValueError):
            k.paired(-np.ones((1, 4, 3, 8)))

    def test_empty(self):
        with self.assertRaises(ValueError):
            k.paired(np.zeros((0, 4, 3, 8)))

    def test_cluster_kept(self):
        x = np.zeros((2, 8, 6))
        x[1] = 2
        np.testing.assert_array_equal(k.bootstrap(x, ["a", "a"], 3, 8), 1)

    def test_seed_stable_distinct(self):
        self.assertEqual(k.seed("a"), k.seed("a"))
        self.assertNotEqual(k.seed("a"), k.seed("b"))

    def test_prevalence_extremes(self):
        self.assertAlmostEqual(k.error_cost(0.03, -0.01, 0), 30)
        self.assertAlmostEqual(k.error_cost(0.03, -0.01, 1), -10)

    def test_break_even(self):
        ratio = k.break_even_ratio(0.03, -0.01, 0.25)
        self.assertEqual(ratio, 9)
        self.assertAlmostEqual(k.error_cost(0.03, -0.01, 0.25, ratio), 0)

    def test_invalid_scenario(self):
        with self.assertRaises(ValueError):
            k.error_cost(0.03, -0.01, 1.01)
        with self.assertRaises(ValueError):
            k.break_even_ratio(0.03, 0, 0.5)

    def test_attempts_not_repairs(self):
        old.accounting_check(
            dict(
                ordinary_logical_calls=3,
                repair_logical_calls=2,
                retry_attempts=1,
                post_attempts=6,
                http_success_with_text=5,
                retryable_failures=1,
            )
        )
        with self.assertRaises(ValueError):
            old.accounting_check(
                dict(
                    ordinary_logical_calls=3,
                    repair_logical_calls=2,
                    retry_attempts=1,
                    post_attempts=5,
                    http_success_with_text=5,
                    retryable_failures=1,
                )
            )

    def test_known_gap(self):
        q = old.attempt_partition(
            [
                dict(
                    start="2026-01-01T00:00:00+00:00", end="2026-01-01T00:00:02+00:00"
                ),
                dict(
                    start="2026-01-01T00:00:05+00:00", end="2026-01-01T00:00:08+00:00"
                ),
            ],
            "2026-01-01T00:00:08+00:00",
        )
        self.assertEqual(q["seconds_by_kind"]["explicit_between_attempt_gap"], 3)

    def test_unknown_not_measured_downtime(self):
        q = old.attempt_partition(
            [dict(start="2026-01-01T00:00:00+00:00")], "2026-01-01T00:00:08+00:00"
        )
        self.assertEqual(q["seconds_by_kind"], {"unallocated_interrupted_window": 8})

    def test_negative_time(self):
        with self.assertRaises(ValueError):
            old.seconds("2026-01-01T00:00:02+00:00", "2026-01-01T00:00:00+00:00")

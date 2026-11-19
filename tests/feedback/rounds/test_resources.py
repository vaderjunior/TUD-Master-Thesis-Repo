import unittest
import numpy as np
from analysis.feedback.rounds import resources_metrics as k
from analysis.feedback.reserved import resources_metrics as core


class Resources(unittest.TestCase):
    def test_shared_reference_not_tripled(self):
        x = np.zeros((2, 10, 3, 8))
        x[:, 0] = 9
        x[:, 1:] = 3
        y = k.paired(x)
        self.assertTrue(np.all(y[:, :, 0] == 9))
        self.assertTrue(np.all(y[:, :, 1] == 3))
        self.assertTrue(np.all(y[:, :, 6] == -6))

    def test_each_stream_weight(self):
        x = np.zeros((1, 10, 3, 8))
        x[:, 1] = 3
        x[:, 4] = 6
        x[:, 7] = 9
        self.assertTrue(np.all(k.paired(x)[:, :, 1] == 6))

    def test_paired_controls(self):
        x = np.ones((2, 10, 3, 8))
        x[:, [1, 4, 7]] = 7
        x[:, [2, 5, 8]] = 4
        x[:, [3, 6, 9]] = 2
        np.testing.assert_array_equal(k.paired(x)[0, 0], [1, 7, 4, 2, 3, 5, 6])

    def test_invalid_shape(self):
        with self.assertRaises(ValueError):
            k.paired(np.zeros((2, 4, 3, 8)))

    def test_negative_cost_rejected(self):
        x = np.zeros((1, 10, 3, 8))
        x[0, 0, 0, 0] = -1
        with self.assertRaises(ValueError):
            k.paired(x)

    def test_single_cluster_preserves_all_texts(self):
        f = np.zeros((2, 8, 7))
        f[1] = 4
        draw = k.bootstrap(f, ["a", "a"], 1, 10)
        self.assertTrue(np.all(draw == 2))

    def test_unequal_cluster_sizes(self):
        f = np.zeros((3, 8, 7))
        f[2] = 6
        draw = k.bootstrap(f, ["a", "a", "b"], 7, 100)
        self.assertEqual(set(draw[:, 0, 0]), {0, 2, 6})

    def test_empty_bootstrap_rejected(self):
        with self.assertRaises(ValueError):
            k.bootstrap(np.zeros((0, 8, 7)), [], 1)

    def test_deterministic_seed(self):
        self.assertEqual(k.seed("x"), k.seed("x"))
        self.assertNotEqual(k.seed("x"), k.seed("y"))

    def test_identical_pools_are_aliases(self):
        g = k.unique_jobs(
            [
                dict(condition="a", source_ids=["2", "1"]),
                dict(condition="b", source_ids=["1", "2"]),
            ],
            ["main", "support"],
        )
        self.assertEqual(len(g), 2)
        self.assertEqual(g["main", ("1", "2")], ["a", "b"])

    def test_duplicate_id_rejected(self):
        with self.assertRaises(ValueError):
            k.unique_jobs([dict(condition="a", source_ids=["1", "1"])], ["main"])

    def test_denominators_separate(self):
        self.assertEqual(
            k.error_counts_per_1000(0.1, 0.2),
            dict(false_alarms_per_1000_benign=100, misses_per_1000_hateful=200),
        )

    def test_attempt_slots_include_repair_and_retry(self):
        core.accounting_check(
            dict(
                ordinary_logical_calls=3,
                repair_logical_calls=2,
                retry_attempts=1,
                post_attempts=6,
                http_success_with_text=5,
                retryable_failures=1,
            )
        )

    def test_retry_is_not_successful_token_reply(self):
        with self.assertRaises(ValueError):
            core.accounting_check(
                dict(
                    ordinary_logical_calls=3,
                    repair_logical_calls=2,
                    retry_attempts=1,
                    post_attempts=6,
                    http_success_with_text=6,
                    retryable_failures=1,
                )
            )

    def test_elapsed_rejects_reversed_clock(self):
        with self.assertRaises(ValueError):
            core.seconds("2026-01-01T02:00:00Z", "2026-01-01T01:00:00Z")

    def test_interruption_partition(self):
        p = core.attempt_partition(
            [
                dict(start="2026-01-01T00:00:00Z", end="2026-01-01T00:01:00Z"),
                dict(start="2026-01-01T00:03:00Z", end="2026-01-01T00:05:00Z"),
            ],
            "2026-01-01T00:05:00Z",
        )
        self.assertEqual(sum(p["seconds_by_kind"].values()), 300)
        self.assertEqual(p["seconds_by_kind"]["explicit_between_attempt_gap"], 120)


if __name__ == "__main__":
    unittest.main()

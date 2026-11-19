import unittest
import numpy as np
from analysis.feedback.rounds import errors_kernel as k


class ErrorsTests(unittest.TestCase):
    def test_draw_column_order_survives_sorted_json(self):
        import json

        definitions = k.columns([1, 2, 3, 4], trajectories=True)
        record = json.loads(
            json.dumps(
                {"columns": definitions, "column_order": list(definitions)},
                sort_keys=True,
            )
        )
        self.assertNotEqual(list(record["columns"]), list(definitions))
        self.assertEqual(record["column_order"], list(definitions))
        count = np.arange(16).reshape((1, 4, 4))
        expected = k.transformed_counts(count, np.array([0]), definitions)
        restored = {c: record["columns"][c] for c in record["column_order"]}
        np.testing.assert_array_equal(
            k.transformed_counts(count, np.array([0]), restored), expected
        )

    def test_reference_has_three_actual_slots_not_nine_independent_references(self):
        error = np.zeros((2, 4, 10, 3), dtype=np.int8)
        error[0, :, 0, 0] = 1
        error[0, :, [1, 4, 7], 0] = 1
        counts, missing = k.policy_counts(error)
        self.assertTrue(np.all(counts[0, :, 0] == 3))
        self.assertTrue(np.all(counts[0, :, 1] == 3))
        self.assertEqual(missing.sum(), 0)

    def test_missing_is_separate_from_false_and_from_errors(self):
        error = np.zeros((1, 4, 10, 3), dtype=np.int8)
        error[0, 3, 1, 0] = -1
        counts, missing = k.policy_counts(error)
        self.assertEqual(counts.sum(), 0)
        self.assertEqual(missing[0, 3, 1], 1)
        result = k.bounds(counts, missing, np.array([0]), 4)
        self.assertEqual(result["contrasts"]["adaptive-static"], [0, 1 / 9])

    def test_common_mask_includes_reference_and_all_repeats(self):
        rows = [{"population": "en_test", "gold_hate": False}] * 3
        gate = np.zeros((3, 4, 10, 3), dtype=np.int8)
        votes = np.full_like(gate, 3)
        gate[0, 3, 0, 2] = -1
        gate[1, 0, 7, 0] = -1
        votes[2, 3, 5, 0] = 2
        self.assertEqual(
            k.cohort(rows, gate, votes, "en_test", "FPR", "round_common", 4).tolist(),
            [1, 2],
        )
        self.assertEqual(
            k.cohort(
                rows, gate, votes, "en_test", "FPR", "all_round_common", 4
            ).tolist(),
            [2],
        )

    def test_pairing_precedes_averaging_and_round_contrasts(self):
        count = np.zeros((1, 4, 4), dtype=np.int64)
        count[0, 0, :] = [3, 6, 4, 2]
        count[0, 3, :] = [6, 6, 3, 1]
        defs = k.columns([1, 2, 3, 4], trajectories=True)
        values = k.transformed_counts(count, np.array([0]), defs)[0]
        lookup = dict(zip(defs, values.tolist()))
        self.assertEqual(lookup["r4_adaptive-static"], 3)
        self.assertEqual(lookup["r4-r1_adaptive"], 0)
        self.assertEqual(lookup["r4-r1_adaptive-reference"], -3)

    def test_cluster_sizes_keep_text_weighting(self):
        values = np.array([[9], [9], [0]], dtype=np.int64)
        labels, sizes, sums = k.grouped(values, ["b", "b", "a"])
        self.assertEqual(labels, ["a", "b"])
        self.assertEqual(sizes.tolist(), [1, 2])
        self.assertEqual(sums.tolist(), [[0], [18]])
        self.assertEqual(values.sum() / (9 * len(values)), 2 / 3)

    def test_bootstrap_matches_explicit_text_multiplicities(self):
        values = np.array([[9, -9], [9, 0], [0, 9]], dtype=np.int64)
        actual, _, _ = k.bootstrap(values, ["b", "b", "a"], 123, draws=67)
        rng = np.random.Generator(np.random.PCG64(123))
        expected = []
        for _ in range(67):
            pick = rng.integers(0, 2, size=2).tolist()
            weights = np.array([pick.count(1), pick.count(1), pick.count(0)])
            expected.append(weights @ values / (9 * weights.sum()))
        np.testing.assert_array_equal(actual, expected)

    def test_undefined_not_zero(self):
        with self.assertRaises(ValueError):
            k.bootstrap(np.empty((0, 1), dtype=np.int64), [], 1)
        with self.assertRaises(ValueError):
            k.bounds(
                np.zeros((0, 4, 4)), np.zeros((0, 4, 4)), np.array([], dtype=int), 4
            )

    def test_full_roster_opposing_missing_bounds(self):
        counts = np.zeros((2, 4, 4), dtype=np.int64)
        missing = np.zeros_like(counts)
        counts[:, 3, 1] = [3, 6]
        counts[:, 3, 2] = [2, 3]
        missing[:, 3, 1] = [1, 0]
        missing[:, 3, 2] = [0, 2]
        result = k.bounds(counts, missing, np.array([0, 1]), 4)
        self.assertEqual(result["contrasts"]["adaptive-static"], [2 / 18, 5 / 18])


if __name__ == "__main__":
    unittest.main()

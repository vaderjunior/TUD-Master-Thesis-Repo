"""Tests for guidelines metrics."""

import unittest
import numpy as np
from analysis.knowledge.guidelines import metrics as m


class GuidelineMetricsTests(unittest.TestCase):

    def test_shared_mask_removes_one_item_from_all_fifteen_settings(self):
        data = np.zeros((3, 5, 3), dtype=np.int8)
        data[1, 4, 2] = -1
        self.assertEqual(m.common_mask(data).tolist(), [True, False, True])

    def test_null_is_unknown_and_bounds_not_zero_imputed(self):
        data = np.array([[[1, 1, 1], [0, 0, 0]], [[-1, 0, 0], [1, 1, 1]]])
        s = m.counts_and_rates(data, ["a", "b"])
        row = s["settings"][0]
        self.assertEqual((row["valid"], row["null"], row["false_positives"]), (1, 1, 1))
        self.assertEqual(row["own_valid_rate"], 1.0)
        self.assertEqual(row["full_roster_bounds"], [0.5, 1.0])
        self.assertEqual(s["common_items"], 1)
        self.assertEqual(s["conditions"]["b"]["common"]["mean"], 0.0)

    def test_sample_sd_uses_three_repeat_sample_denominator(self):
        s = m.scalar_summary([0.0, 0.5, 1.0])
        self.assertEqual(s["sample_sd"], 0.5)
        self.assertEqual(s["range"], 1.0)

    def test_item_weighted_not_equal_cluster_weighted(self):
        data = np.array([[[1] * 3]] * 3 + [[[0] * 3]])
        draws = np.array([[0, 0], [1, 1], [0, 1]])
        got = m.cluster_rates(data, [10, 10, 10, 20], draws)
        np.testing.assert_array_equal(got[:, 0], [1.0, 0.0, 0.75])

    def test_templates_draw_whole_cluster_with_multiplicity(self):
        data = np.array([[[1, 0, 1]], [[0, 1, 0]], [[1, 1, 1]]])
        got = m.cluster_rates(data, [1, 1, 2], np.array([[0, 1], [1, 1], [0, 0]]))
        np.testing.assert_allclose(got[:, 0], [2 / 3, 1.0, 0.5])

    def test_pairing_identical_arms_has_zero_difference(self):
        data = np.array([[[0, 1, 1], [0, 1, 1]], [[1, 1, 0], [1, 1, 0]]])
        got = m.cluster_rates(data, [1, 2], np.array([[0, 0], [0, 1], [1, 1]]))
        np.testing.assert_array_equal(got[:, 0] - got[:, 1], 0.0)

    def test_contrast_sign_family_and_missing_bounds(self):
        data = np.array([[[0, 0, 0], [1, 1, 1]]])
        s = m.counts_and_rates(data, ["a", "b"])
        samples = np.tile([0.0, 1.0], (20, 1))
        c = m.contrasts(
            s, samples, ["a", "b"], {"a_minus_b": {"a": 1, "b": -1}}, ["a_minus_b"], 0.9875, True
        )["a_minus_b"]
        self.assertEqual(c["mean_difference"], -1.0)
        self.assertEqual(c["primary_family_interval"], [-1.0, -1.0])
        self.assertTrue(c["family_interval_excludes_zero"])
        self.assertEqual(c["full_roster_missing_output_difference_bounds"], [-1.0, -1.0])
        d = m.contrasts(
            s, samples, ["a", "b"], {"a_minus_b": {"a": 1, "b": -1}}, ["a_minus_b"], 0.9875, False
        )["a_minus_b"]
        self.assertIsNone(d["primary_family_interval"])

    def test_percentile_interval_is_two_sided(self):
        np.testing.assert_allclose(m.interval(np.arange(101), 0.95), [2.5, 97.5])
        np.testing.assert_allclose(m.interval(np.arange(101), 0.9875), [0.625, 99.375])

    def test_empty_common_pool_unscorable_not_zero(self):
        data = np.full((2, 2, 3), -1)
        s = m.counts_and_rates(data, ["a", "b"])
        self.assertEqual(s["common_items"], 0)
        self.assertIsNone(s["conditions"]["a"]["common"])
        self.assertIsNone(s["conditions"]["a"]["own_valid"])
        self.assertEqual(s["conditions"]["a"]["full_roster_mean_bounds"], [0.0, 1.0])
        with self.assertRaises(ValueError):
            m.cluster_rates(data, [1, 2], np.array([[0, 1]]))

    def test_bad_inputs_rejected(self):
        with self.assertRaises(ValueError):
            m.common_mask(np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            m.common_mask(np.full((2, 5, 3), 2))
        with self.assertRaises(ValueError):
            m.counts_and_rates(np.zeros((0, 5, 3)), list("abcde"))
        with self.assertRaises(ValueError):
            m.cluster_rates(np.zeros((2, 1, 3)), [1, 2], np.array([[0, 2]]))


if __name__ == "__main__":
    unittest.main()

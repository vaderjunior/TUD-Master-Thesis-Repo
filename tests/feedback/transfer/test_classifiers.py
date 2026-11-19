import unittest
import numpy as np
from analysis.feedback.transfer import classifiers_metrics as k
from analysis.feedback.transfer import effects_metrics as masks


class Checks(unittest.TestCase):
    def arrays(self, n=2):
        return (
            np.zeros((n, 2, 2, 4), dtype=np.int8),
            np.zeros((n, 4, 3), dtype=np.int8),
            np.full((n, 4, 3), 3, dtype=np.int8),
        )

    def test_zero(self):
        l, q, v = self.arrays()
        self.assertTrue((k.expand(k.raw_rows(l, q).sum(axis=0), 2) == 0).all())

    def test_full_reference_weight(self):
        l, q, v = self.arrays()
        q[:, 0] = 1
        x = k.expand(k.raw_rows(l, q).sum(axis=0), 2)
        self.assertTrue((x[:, 2] == 1).all())
        self.assertTrue((x[:, 6] == 1).all())

    def test_one_pool_not_one_repeat(self):
        l, q, v = self.arrays()
        l[:, :, :, 1] = 1
        x = k.expand(k.raw_rows(l, q).sum(axis=0), 2)
        self.assertTrue((x[:, 1] == 1 / 3).all())
        self.assertTrue((x[:, 8] == 1).all())

    def test_qwen_pool_repeat_weight(self):
        l, q, v = self.arrays()
        q[:, 1, 0] = 1
        x = k.expand(k.raw_rows(l, q).sum(axis=0), 2)
        self.assertTrue((x[:, 3] == 1 / 9).all())
        self.assertTrue((x[:, 11] == 1 / 3).all())

    def test_difference_in_change_sign(self):
        l, q, v = self.arrays()
        l[:, :, :, 0] = 1
        q[:, 1:] = 1
        x = k.expand(k.raw_rows(l, q).sum(axis=0), 2)
        self.assertTrue((x[:, 6] == -2).all())
        self.assertTrue((x[:, 7] == -1).all())

    def test_settings_independent(self):
        l, q, v = self.arrays()
        l[:, 1, 0, 0] = 1
        x = k.expand(k.raw_rows(l, q).sum(axis=0), 2)
        self.assertEqual(x[:, 0].tolist(), [0, 0, 1, 0])

    def test_empty_denominator(self):
        with self.assertRaises(ValueError):
            k.expand(np.zeros(20, dtype=int), 0)

    def test_bad_shape(self):
        with self.assertRaises(ValueError):
            k.raw_rows(np.zeros((2, 4), dtype=int), np.zeros((2, 4, 3), dtype=int))

    def test_bad_code(self):
        l, q, v = self.arrays()
        l[0, 0, 0, 0] = -1
        with self.assertRaises(ValueError):
            k.raw_rows(l, q)

    def test_complete_bounds(self):
        l, q, v = self.arrays()
        q[:, 1:] = 1
        b = k.bounds(l, q, v, [0, 1])
        self.assertEqual(b[0]["difference_in_change"], [-1, -1])

    def test_null_reference_bound(self):
        l, q, v = self.arrays()
        q[:, 0] = -1
        v[:, 0] = 0
        b = k.bounds(l, q, v, [0, 1])
        self.assertEqual(b[0]["difference_in_change"], [0, 1])

    def test_null_updated_bound(self):
        l, q, v = self.arrays()
        q[:, 1:] = -1
        v[:, 1:] = 0
        b = k.bounds(l, q, v, [0, 1])
        self.assertEqual(b[0]["difference_in_change"], [-1, 0])

    def test_strict_bounds(self):
        l, q, v = self.arrays()
        v[:, 1:] = 2
        b = k.bounds(l, q, v, [0, 1], True)
        self.assertEqual(b[0]["difference_in_change"], [-1, 0])
        self.assertEqual(k.bounds(l, q, v, [0, 1])[0]["difference_in_change"], [0, 0])

    def test_empty_bounds(self):
        l, q, v = self.arrays()
        self.assertIsNone(k.bounds(l, q, v, []))

    def test_threshold_is_strict(self):
        self.assertEqual(
            (np.array([0.499, 0.5, 0.6, 0.601]) > 0.6).tolist(),
            [False, False, False, True],
        )

    def test_per_run_f1(self):
        c, s = k.binary_scores(np.array([0, 1]), np.array([[0, 0], [1, 0]]))
        self.assertEqual(c.tolist(), [[1, 0, 0, 1], [1, 0, 1, 0]])
        self.assertAlmostEqual(s["macro_f1"][1], 1 / 3)

    def test_seed_stable_distinct(self):
        self.assertEqual(k.seed("a"), k.seed("a"))
        self.assertNotEqual(k.seed("a"), k.seed("b"))

    def test_bootstrap_keeps_clusters_and_settings(self):
        l, q, v = self.arrays(3)
        l[:, :, :, 0] = 1
        raw = k.raw_rows(l, q)
        m = [
            dict(key=str(i), inference_cluster=str(i // 2), source="x")
            for i in range(3)
        ]
        x, info = k.bootstrap(m, raw, [0, 1, 2], [0, 1, 2], 43, draws=11, block=3)
        self.assertEqual(info["units"], 2)
        self.assertTrue((x[:, :, 6] == -1).all())

    def test_source_strata_do_not_split_cluster(self):
        l, q, v = self.arrays(3)
        m = [
            dict(key=str(i), inference_cluster=str(i // 2), source=("a", "b", "a")[i])
            for i in range(3)
        ]
        labels, c, n, st = k.grouping(
            m, k.raw_rows(l, q), [0, 1, 2], [0, 1, 2], "source_stratified"
        )
        self.assertEqual(st, {("a", "b"): [0], ("a",): [1]})
        self.assertEqual(n.tolist(), [2, 1])

    def test_common_mask_uses_every_repeat(self):
        l, q, v = self.arrays()
        q[1, 3, 2] = -1
        v[1, 3, 2] = 0
        m = [
            dict(
                key=str(i),
                population="en_test",
                gold_hate=False,
                source="x",
                inference_cluster=str(i),
            )
            for i in range(2)
        ]
        c = masks.cohort(m, q, v, "en_test", False)
        self.assertEqual(c["common"].tolist(), [0])


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
from sklearn.metrics import f1_score

from analysis.classification.supervised import metrics as m


class MainMetricsTests(unittest.TestCase):
    def setUp(self):
        self.y = np.array([[1, 0], [1, 1], [0, 1], [0, 0]], dtype=np.uint8)
        self.p = np.array(
            [[[1, 0], [0, 1], [1, 0], [0, 0]], [[1, 1], [1, 1], [0, 1], [1, 0]]],
            dtype=np.uint8,
        )

    def test_all_label_scores_against_sklearn(self):
        actual, counts = m.point_scores(self.y, self.p, [0, 1])
        expected = [
            np.mean([f1_score(self.y[:, j], p[:, j], zero_division=0) for j in (0, 1)])
            for p in self.p
        ]
        np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
        np.testing.assert_array_equal(
            counts["tp"] + counts["fn"], np.repeat(self.y.sum(0)[None, :], 2, axis=0)
        )

    def test_single_label_is_positive_f1(self):
        y = np.array([[1], [0], [0], [0]], dtype=np.uint8)
        p = np.zeros((1, 4, 1), dtype=np.uint8)
        self.assertEqual(m.point_scores(y, p, [0])[0][0], 0.0)
        self.assertGreater(
            f1_score(y[:, 0], p[0, :, 0], average="macro", zero_division=0), 0.0
        )

    def test_zero_denominator_is_zero(self):
        value, _ = m.point_scores(
            np.zeros((2, 1), dtype=np.uint8), np.zeros((1, 2, 1), dtype=np.uint8), [0]
        )
        self.assertEqual(value[0], 0.0)

    def test_unsupported_label_cannot_enter_macro(self):
        actual, _ = m.point_scores(self.y, self.p, [1])
        expected = [f1_score(self.y[:, 1], v[:, 1], zero_division=0) for v in self.p]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)

    def test_label_permutation_with_support(self):
        a, _ = m.point_scores(self.y, self.p, [0])
        b, _ = m.point_scores(self.y[:, ::-1], self.p[:, :, ::-1], [1])
        np.testing.assert_array_equal(a, b)

    def test_input_failures(self):
        cases = [
            (self.y, self.p[:, :3], [0]),
            (self.y, self.p, [0, 0]),
            (self.y, self.p, []),
            (self.y, self.p, [2]),
            (self.y, self.p.astype(float) + 0.1, [0]),
            (self.y[:0], self.p[:, :0], [0]),
            (self.y, self.p[:0], [0]),
        ]
        for y, p, s in cases:
            with self.subTest(shapes=(y.shape, p.shape), support=s), self.assertRaises(
                m.r.IntegrityError
            ):
                m.point_scores(y, p, s)

    def test_confusion_counts_hand_calculation(self):
        scores, counts = m.point_scores(
            np.array([[1], [1], [0], [0]]), np.array([[[1], [0], [1], [1]]]), [0]
        )
        self.assertEqual(counts["tp"][0, 0], 1)
        self.assertEqual(counts["fp"][0, 0], 2)
        self.assertEqual(counts["fn"][0, 0], 1)
        self.assertEqual(scores[0], 0.4)


class BootstrapTests(unittest.TestCase):
    setUp = MainMetricsTests.setUp

    def test_grouped_point_equals_raw_point(self):
        _, _, terms = m.group_terms(self.y, self.p, ["a", "a", "b", "c"], [0, 1])
        value = m.score_terms(terms.sum(0)[None, :], 2, 2)[0]
        np.testing.assert_allclose(
            value, m.point_scores(self.y, self.p, [0, 1])[0], atol=1e-15, rtol=0
        )

    def test_whole_cluster_bootstrap_against_direct_rows(self):
        clusters = np.array(["a", "a", "b", "c"])
        values, info = m.bootstrap(
            self.y, self.p, clusters.tolist(), [0, 1], draws=127, seed=73, batch=19
        )
        unique = np.unique(clusters)
        rng = np.random.default_rng(73)
        expected = []
        for _ in range(127):
            selected = rng.choice(unique, size=len(unique), replace=True)
            rows = np.concatenate(
                [np.flatnonzero(clusters == item) for item in selected]
            )
            expected.append(
                [
                    np.mean(
                        [
                            f1_score(self.y[rows, j], p[rows, j], zero_division=0)
                            for j in (0, 1)
                        ]
                    )
                    for p in self.p
                ]
            )
        np.testing.assert_allclose(values, expected, rtol=0, atol=1e-15)
        self.assertEqual(info["clusters"], 3)

    def test_batch_size_does_not_change_draws(self):
        a, aa = m.bootstrap(
            self.y, self.p, ["a", "a", "b", "c"], [0, 1], draws=101, seed=9, batch=1
        )
        b, bb = m.bootstrap(
            self.y, self.p, ["a", "a", "b", "c"], [0, 1], draws=101, seed=9, batch=64
        )
        np.testing.assert_array_equal(a, b)
        self.assertEqual(aa["picks_sha256"], bb["picks_sha256"])

    def test_pairing_identical_methods_is_exact_zero(self):
        values, _ = m.bootstrap(
            self.y,
            np.repeat(self.p[:1], 3, axis=0),
            ["a", "a", "b", "c"],
            [0, 1],
            draws=64,
        )
        np.testing.assert_array_equal(values[:, 0] - values[:, 2], np.zeros(64))

    def test_support_does_not_change_on_zero_positive_draw(self):
        y = np.eye(2, dtype=np.uint8)
        p = np.repeat(y[None, :, :], 2, axis=0)
        values, _ = m.bootstrap(y, p, ["a", "b"], [0, 1], draws=128, seed=1)
        self.assertEqual(set(np.unique(values)), {0.5, 1.0})

    def test_cluster_failures(self):
        for clusters in (["a"], ["a", "b", "", "d"], ["a", "b", None, "d"]):
            with self.subTest(clusters=clusters), self.assertRaises(m.r.IntegrityError):
                m.group_terms(self.y, self.p, clusters, [0, 1])

    def test_impossible_count_rejected(self):
        with self.assertRaises(m.r.IntegrityError):
            m.score_terms(np.array([[2.0, 1.0, 1.0]]), 1, 1)

    def test_fixed_rng_changes_with_seed(self):
        a, _ = m.bootstrap(
            self.y, self.p, ["a", "a", "b", "c"], [0, 1], draws=37, seed=1
        )
        b, _ = m.bootstrap(
            self.y, self.p, ["a", "a", "b", "c"], [0, 1], draws=37, seed=2
        )
        self.assertFalse(np.array_equal(a, b))


class AggregationTests(unittest.TestCase):
    def setUp(self):
        self.models = []
        for pool in m.r.POOLS:
            self.models.extend(
                m.model(pool, "tuned_selected", seed) for seed in m.r.SEEDS
            )
            self.models.extend(m.model(pool, method) for method in m.METHODS[1:])

    def test_mean_seeds_then_equal_pools(self):
        values = np.arange(63, dtype=float) / 100
        pools, budgets = m.aggregation(self.models, values)
        self.assertAlmostEqual(pools[m.r.POOLS[0], "tuned_selected"], 0.02)
        self.assertAlmostEqual(
            budgets[147, "tuned_selected"], np.mean([0.02, 0.09, 0.16])
        )
        self.assertAlmostEqual(
            budgets[147, "bge_selected_fold42"], np.mean([0.05, 0.12, 0.19])
        )

    def test_draw_axis_kept(self):
        values = np.stack([np.arange(63) / 100, np.arange(63) / 200])
        a, b = m.aggregation(self.models, values)
        self.assertEqual(b[147, "tuned_selected"].shape, (2,))
        np.testing.assert_allclose(b[147, "tuned_selected"], [0.09, 0.045])

    def test_missing_seed_fails(self):
        with self.assertRaises(m.r.IntegrityError):
            m.aggregation(self.models[1:], np.ones(62))

    def test_duplicate_seed_fails(self):
        self.models[1] = dict(self.models[0], id="different")
        with self.assertRaises(m.r.IntegrityError):
            m.aggregation(self.models, np.ones(63))

    def test_no_manufactured_linear_seed(self):
        self.models[5] = dict(self.models[5], seed=10)
        with self.assertRaises(m.r.IntegrityError):
            m.aggregation(self.models, np.ones(63))

    def test_score_axis_mismatch_fails(self):
        with self.assertRaises(m.r.IntegrityError):
            m.aggregation(self.models, np.ones(62))

    def test_extra_component_does_not_change_primary(self):
        a, b = m.aggregation(self.models, np.arange(63) / 100)
        extra = self.models + [m.model(m.r.POOLS[0], "supplement", 10)]
        c, d = m.aggregation(extra, np.r_[np.arange(63) / 100, 1.0])
        self.assertEqual(b, d)

    def test_budget_aliases(self):
        self.assertEqual(
            [m.budget(p) for p in m.r.POOLS], [147] * 3 + [1000] * 3 + [5000] * 3
        )


if __name__ == "__main__":
    unittest.main()

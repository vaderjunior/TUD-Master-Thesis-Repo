import unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from training.matched_annotations import results as s


class ReviewTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(51)
        y = rng.integers(0, 2, (12, 3)).astype(bool)
        x = rng.integers(0, 2, (5, 7, 12, 3)).astype(bool)
        return y, x, np.stack((x & y, x & ~y, ~x & y), axis=-1).astype(int)

    def test_metric_against_sklearn(self):
        y, x, counts = self.data()
        f = s.f1counts(counts.sum(2), [0, 1, 2])
        for pool in range(5):
            for fit in range(7):
                self.assertAlmostEqual(
                    f[pool, fit],
                    f1_score(y, x[pool, fit], average="macro", zero_division=0),
                    places=13,
                )

    def test_zero_counts(self):
        self.assertEqual(float(s.f1counts(np.zeros((3, 3)), [0, 1, 2])), 0)

    def test_pool_and_seed_estimand(self):
        _, _, counts = self.data()
        f = s.f1counts(counts.sum(2), [0, 1, 2])
        out = s.paired(
            counts, list(map(str, range(12))), ["a"] * 12, [0, 1, 2], draws=128
        )
        self.assertAlmostEqual(
            out["tuned_selected_minus_bge"]["difference"],
            float(np.mean(f[:, :5].mean(1) - f[:, 5])),
            places=13,
        )
        self.assertAlmostEqual(
            out["frozen_roberta_minus_bge"]["difference"],
            float(np.mean(f[:, 6] - f[:, 5])),
            places=13,
        )
        self.assertEqual(len(out["tuned_selected_minus_bge"]["leave_one_pool_out"]), 5)

    def test_identical_methods_exact_zero(self):
        _, _, counts = self.data()
        counts[:] = counts[:, :1]
        out = s.paired(counts, ["same"] * 12, ["a"] * 12, [0, 1, 2], draws=128)
        for value in out.values():
            self.assertAlmostEqual(value["difference"], 0, places=13)
            self.assertTrue(np.allclose(value["ci9875"], 0, rtol=0, atol=1e-13))

    def test_duplicate_rows_are_whole_clusters(self):
        _, _, counts = self.data()
        clusters = ["k" + str(i) for i in range(12)]
        a = s.paired(counts, clusters, ["a"] * 12, [0, 1, 2], draws=128)
        b = s.paired(
            np.repeat(counts, 2, axis=2),
            np.repeat(clusters, 2).tolist(),
            ["a"] * 24,
            [0, 1, 2],
            draws=128,
        )
        for k in a:
            for field in ("difference", "ci9875", "ci95", "joint_pool_text_ci95"):
                self.assertTrue(
                    np.allclose(a[k][field], b[k][field], atol=1e-13, rtol=0)
                )

    def test_strata_and_reproducibility(self):
        _, _, counts = self.data()
        clusters = list(map(str, range(12)))
        sources = ["a"] * 6 + ["b"] * 6
        a = s.paired(counts, clusters, sources, [0, 1, 2], draws=128, stratified=True)
        b = s.paired(counts, clusters, sources, [0, 1, 2], draws=128, stratified=True)
        self.assertEqual(a, b)
        self.assertEqual(a["tuned_selected_minus_bge"]["clusters"], 12)


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReviewTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    s.r.save(
        "statistics_tests.json",
        dict(
            status="pass",
            tests=result.testsRun,
            source_sha256=s.c.sha(Path(__file__)),
            statistics_source_sha256=s.c.sha(Path(s.__file__)),
        ),
    )

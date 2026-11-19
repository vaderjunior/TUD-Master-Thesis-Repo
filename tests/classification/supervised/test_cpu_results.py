from pathlib import Path
import unittest
import numpy as np
from analysis.classification.supervised import cpu_results as s

r = s.r
c = s.c


class ScientificChecks(unittest.TestCase):
    def test_one_fit_vs_three_is_mean_of_metrics_not_ensemble(self):
        y = np.array([[1], [1], [0], [0]], dtype=bool)
        xs = [y, y, ~y, y]
        counts = np.stack([np.stack((y & x, ~y & x, y & ~x), axis=-1) for x in xs])
        result = s.bootstrap(counts, ["a", "b", "c", "d"], ["src"] * 4, [0], draws=128)
        self.assertAlmostEqual(result["left"], 1.0)
        self.assertAlmostEqual(result["right"], 2 / 3)
        self.assertAlmostEqual(result["difference"], 1 / 3)
        self.assertEqual(
            (result["left_fixed_fits"], result["right_fixed_fits"]), (1, 3)
        )

    def test_fast_cluster_bootstrap_matches_independent_row_resampling(self):
        rng = np.random.default_rng(4)
        y = rng.integers(0, 2, (6, 3)).astype(bool)
        xs = rng.integers(0, 2, (4, 6, 3)).astype(bool)
        counts = np.stack([np.stack((y & x, ~y & x, y & ~x), axis=-1) for x in xs])
        clusters = ["a", "a", "b", "c", "d", "d"]
        sources = ["s", "t", "s", "u", "u", "u"]
        result = s.bootstrap(counts, clusters, sources, [0, 1, 2], draws=128)
        random = np.random.default_rng(20261012)
        values = []
        unique = sorted(set(clusters))
        for picks in random.choice(range(4), (128, 4), replace=True):
            rows = [i for j in picks for i, k in enumerate(clusters) if k == unique[j]]
            sums = counts[:, rows].sum(1)
            scores = []
            for model in sums:
                f = []
                for tp, fp, fn in model:
                    f.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
                scores.append(sum(f) / len(f))
            values.append(scores[0] - sum(scores[1:]) / 3)
        np.testing.assert_allclose(
            result["ci9875"],
            np.quantile(values, [0.00625, 0.99375]),
            atol=1e-12,
            rtol=0,
        )
        np.testing.assert_allclose(
            result["ci95"], np.quantile(values, [0.025, 0.975]), atol=1e-12, rtol=0
        )

    def test_paired_identical_predictions_have_zero_interval(self):
        counts = np.ones((4, 4, 3, 3), dtype=np.int64)
        for stratified in (False, True):
            v = s.bootstrap(
                counts,
                ["a", "a", "b", "c"],
                ["s", "t", "s", "t"],
                [0, 2],
                draws=64,
                stratified=stratified,
            )
            self.assertEqual(v["clusters"], 3)
            self.assertEqual(v["ci9875"], [0.0, 0.0])
            self.assertEqual(v["difference"], 0.0)

    def test_zero_support_macro_label_is_not_dropped(self):
        ep = dict(dimension="hate_type", all_labels=["a", "b"], macro_labels=["a", "b"])
        y = np.array([[1, 0], [1, 0]], dtype=bool)
        result, _ = s.summary(y, y.copy(), ep)
        self.assertEqual(result["macro_f1"], 0.5)
        self.assertEqual(result["exact_set_accuracy"], 1.0)

    def test_native_cutoff_binary_softmax_and_strict_gate_boundary(self):
        p = s.neural_probability("hate", [[1000.0, 1000.0], [1000.0, 1001.0]])
        self.assertEqual(
            r.bits("hate", p, 0.5).tolist(), [[True, False], [False, True]]
        )
        np.testing.assert_allclose(
            p[:, 0], [0.5, 1 / (1 + np.exp(-1))], rtol=0, atol=1e-15
        )
        self.assertEqual(r.bits("hate_type", np.array([[0.5]]), 0.5).tolist(), [[True]])
        self.assertEqual(
            r.bits("severity", np.array([[0.5, 0.5, 0.1]]), None).tolist(),
            [[True, False, False]],
        )

    def test_missing_annotations_and_gold_gate_mask(self):
        ep = dict(
            dimension="hate_type",
            all_labels=["a", "b"],
            macro_labels=["a", "b"],
            scorable_ids=["1", "3"],
            gold_support={"a": 1, "b": 0},
        )
        rows = [
            dict(id="1", gate=True, hate_types=["a"]),
            dict(id="2", gate=False, hate_types=["a"]),
            dict(id="3", gate=True, hate_types=[]),
            dict(id="4", gate=True, hate_types=None),
        ]
        predictions = {
            v["id"]: dict(
                hate=True, hate_type=["a"], target_group=[], severity=None, legal=[]
            )
            for v in rows
        }
        ids, y, x = s.previous.arrays(rows, predictions, ep)
        self.assertEqual(ids, ["1", "3"])
        self.assertAlmostEqual(s.summary(y, x, ep)[0]["macro_f1"], 1 / 3)

    def test_training_only_text_vocabulary_and_saved_transform(self):
        train = ["plain useful words", "more plain words"]
        held = ["novelonlytoken xyz"]
        x, state, idf = r.independent_features(train)
        self.assertNotIn("novelonlytoken", state["word"]["vocabulary"])
        same = r.independent_features(train, state, idf)[0]
        self.assertLess(r.matrix_check(same, x), 1e-12)
        prediction_features = r.independent_features(held, state, idf)[0]
        self.assertEqual(prediction_features.shape[1], x.shape[1])


if __name__ == "__main__":
    with c.offline():
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(ScientificChecks)
        )
    if not result.wasSuccessful():
        raise SystemExit(1)
    r.save(
        "statistics_tests.json",
        dict(
            status="pass",
            tests_run=result.testsRun,
            source_hashes={
                c.rel(p): c.sha(p)
                for p in [Path(__file__), Path(s.__file__), Path(r.__file__)]
            },
            Qwen_calls=0,
            CUDA_used=False,
        ),
    )

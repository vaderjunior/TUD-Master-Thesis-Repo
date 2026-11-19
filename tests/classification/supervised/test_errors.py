import unittest
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score
from analysis.classification.supervised import errors as e
from analysis.classification.supervised import archive as r


class ErrorTests(unittest.TestCase):
    def test_binary_orientation_and_metrics(self):
        y = np.array([1, 1, 0, 0, 0])
        p = np.array([[1, 0, 1, 0, 0], [1, 1, 1, 1, 1]])
        c = e.binary_counts(y, p)
        np.testing.assert_array_equal(c, [[1, 1, 1, 2], [2, 3, 0, 0]])
        self.assertAlmostEqual(e.binary_metrics(c)["fpr"][0], 1 / 3)
        for j in range(2):
            self.assertAlmostEqual(
                e.binary_metrics(c)["macro_f1"][j],
                f1_score(y, p[j], labels=[0, 1], average="macro"),
            )

    def test_binary_rejects_invalid_axes_and_values(self):
        for y, p in [
            (np.zeros((2, 1)), np.zeros((1, 2))),
            (np.zeros(2), np.array([[0, 2]])),
        ]:
            with self.assertRaises(r.IntegrityError):
                e.binary_counts(y, p)

    def test_false_alarm_and_missed_hate_fixes_do_not_mix(self):
        y = np.array([0, 0, 1, 1])
        a = np.array([[1, 0, 0, 1]])
        b = 1 - a
        events = e.paired_events(y, a, b)
        for key, index in [
            ("fp_fixed", 0),
            ("fp_introduced", 1),
            ("fn_fixed", 2),
            ("fn_introduced", 3),
        ]:
            np.testing.assert_array_equal(
                events[key], np.eye(4, dtype=bool)[index][None, :]
            )

    def test_counts_not_reinterpreted_as_positive_only_f1(self):
        counts = np.array([[2, 3, 0, 0]])
        self.assertAlmostEqual(e.binary_metrics(counts)["positive_f1"][0], 4 / 7)
        self.assertAlmostEqual(e.binary_metrics(counts)["macro_f1"][0], 2 / 7)

    def test_fine_missing_label_partition(self):
        y = np.array([[1, 1], [1, 1], [1, 0]])
        raw = np.array([[[1, 0], [1, 0], [0, 1]]])
        gate = np.array([[1, 0, 0]])
        f = e.fine_flow(y, raw, gate)
        np.testing.assert_array_equal(f["fn_head_pass"], [1])
        np.testing.assert_array_equal(f["fn_head_blocked"], [2])
        np.testing.assert_array_equal(f["fn_gate_only"], [1])
        np.testing.assert_array_equal(f["fn_total"], [4])
        np.testing.assert_array_equal(f["fp_removed_by_gate"], [1])

    def test_blocking_correct_raw_set_is_distinct_from_head_failure(self):
        y = np.array([[1, 0], [0, 1]])
        raw = np.array([[[1, 0], [1, 0]]])
        g = np.array([[0, 1]])
        f = e.fine_flow(y, raw, g)
        self.assertEqual(f["raw_exact_blocked"][0], 1)
        self.assertEqual(f["raw_wrong_pass"][0], 1)
        self.assertEqual(f["gate_aware_exact"][0], 0)

    def test_gold_empty_exception_does_not_credit_wrong_hate_gate(self):
        y = np.array([[0, 0], [1, 0]])
        raw = np.array([[[1, 0], [1, 0]]])
        g = np.array([[0, 1]])
        f = e.fine_flow(y, raw, g)
        self.assertEqual(f["gated_exact"][0], 2)
        self.assertEqual(f["gate_aware_exact"][0], 1)
        self.assertEqual(f["empty_gold_correct_only_after_block"][0], 1)

    def test_fine_flow_rejects_wrong_gate_shape(self):
        with self.assertRaises(r.IntegrityError):
            e.fine_flow(np.zeros((2, 2)), np.zeros((1, 2, 2)), np.zeros((2, 1)))

    def test_hybrid_pair_uses_one_baseline_fit_without_new_seeds(self):
        models = [
            dict(pool="p", method="tuned", variant="selected", seed=s) for s in r.SEEDS
        ] + [dict(pool="p", method="bge", variant="selected", seed=None)]
        pairs = e.pair_indices(models, "tuned/selected", "bge/selected", "p")
        self.assertEqual(pairs, [(j, 5) for j in range(5)])
        with self.assertRaises(r.IntegrityError):
            e.pair_indices(models[:-1], "tuned/selected", "bge/selected", "p")

    def test_seed_matching_for_two_neural_heads(self):
        models = [
            dict(pool="p", method=f, variant="selected", seed=s)
            for f in ("tuned", "fixed_balanced")
            for s in r.SEEDS
        ]
        self.assertEqual(
            e.pair_indices(models, "tuned/selected", "fixed_balanced/selected", "p"),
            [(j, j + 5) for j in range(5)],
        )

    def test_cluster_counts_match_direct_unequal_group_resampling(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([[0, 0, 1, 1], [1, 1, 1, 0]])
        clusters = ["a", "a", "b", "c"]
        actual, _ = e.count_resamples(y, p, clusters, draws=15, batch=4)
        rng = np.random.default_rng(20261020)
        groups = [np.array([0, 1]), np.array([2]), np.array([3])]
        for i in range(15):
            picks = rng.integers(3, size=3)
            rows = np.concatenate([groups[k] for k in picks])
            np.testing.assert_array_equal(
                actual[i], e.binary_counts(y[rows], p[:, rows])
            )

    def test_count_bootstrap_batch_invariance(self):
        y = np.array([0, 1, 1, 0])
        p = np.array([[0, 0, 1, 1]])
        a, ai = e.count_resamples(y, p, ["a", "a", "b", "c"], draws=17, batch=3)
        b, bi = e.count_resamples(y, p, ["a", "a", "b", "c"], draws=17, batch=8)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(ai, bi)

    def test_annotation_strata_use_saved_source_metadata(self):
        self.assertEqual(
            e.annotation_stratum(
                dict(
                    source="hatexplain",
                    raw={"votes": ["offensive", "normal", "offensive"]},
                )
            ),
            "hatexplain/offensive",
        )
        self.assertEqual(
            e.annotation_stratum(dict(source="mhs", raw={"score": 0.5})),
            "mhs/0<score<=0.5",
        )
        self.assertEqual(
            e.annotation_stratum(dict(source="mhs", raw={"score": 1.63})),
            "mhs/1.0<score<=1.63",
        )
        self.assertEqual(
            e.annotation_stratum(
                dict(
                    source="implicit_hate",
                    raw={"stage1_class": "implicit_hate (stg2-only)"},
                )
            ),
            "implicit_hate/implicit_hate (stg2-only)",
        )

    def test_majority_tie_is_not_arbitrarily_resolved(self):
        with self.assertRaises(r.IntegrityError):
            e.annotation_stratum(
                dict(
                    source="hatexplain",
                    raw={"votes": ["normal", "offensive", "hatespeech"]},
                )
            )


if __name__ == "__main__":
    unittest.main()

import unittest
from unittest.mock import patch
import numpy as np
from training.frozen_features import compact as m


class CompactTests(unittest.TestCase):
    def test_gate_tie_is_negative_and_fine_tie_positive(self):
        p = np.array([[0.5], [0.6]])
        np.testing.assert_array_equal(m.decisions("hate", p, 0.5), [[1, 0], [0, 1]])
        np.testing.assert_array_equal(m.decisions("hate_type", p, 0.5), [[1], [1]])

    def test_gate_clears_multilabel_and_severity(self):
        p = np.array([[0.9, 0.8, 0.1], [0.1, 0.2, 0.7]])
        np.testing.assert_array_equal(
            m.decisions("hate_type", p, 0.5, [False, True]), [[0, 0, 0], [0, 0, 1]]
        )
        np.testing.assert_array_equal(
            m.decisions("severity", p, None, [False, True]), [[0, 0, 0], [0, 0, 1]]
        )

    def test_ordinal_argmax_tie_uses_frozen_order(self):
        np.testing.assert_array_equal(
            m.decisions("severity", np.array([[0.7, 0.7, 0.2]])), [[1, 0, 0]]
        )

    def test_macro_is_mean_label_f1_not_micro(self):
        y = np.array([[1, 1], [1, 0], [0, 0]])
        p = np.array([[1, 0], [0, 0], [0, 0]])
        self.assertAlmostEqual(m.macro(y, p, [0, 1]), (2 / 3 + 0) / 2)
        self.assertAlmostEqual(m.macro(y, p, [0]), 2 / 3)

    def test_rare_constant_labels_are_explicit(self):
        x = np.zeros((5, 3))
        for v in [0, 1]:
            saved = m.binary_fit(x, np.full(5, v), 1, 42)
            self.assertEqual(saved, {"constant": v})
            np.testing.assert_array_equal(m.binary_predict(saved, x), np.full(5, v))

    def test_saved_linear_weights_reproduce_a_fitted_classifier(self):
        x = np.array([[-2.0, 0], [-1.0, 1], [1.0, 1], [2.0, 0]])
        saved = m.binary_fit(x, [0, 0, 1, 1], 1, 42)
        pred = m.binary_predict(saved, x)
        self.assertTrue(np.all(pred[:2] < 0.5) and np.all(pred[2:] > 0.5))

    def test_tie_rule_uses_regularization_then_cutoff(self):
        options = [
            {"score": 0.5, "C": c, "threshold": t}
            for c in [0.1, 1]
            for t in [0.2, 0.5, 0.8]
        ]
        self.assertEqual(m.choose(options), {"score": 0.5, "C": 0.1, "threshold": 0.5})
        self.assertEqual(
            m.choose([{"score": 0.5, "C": 1, "threshold": t} for t in [0.4, 0.6]])[
                "threshold"
            ],
            0.6,
        )

    def test_shared_train_groups_do_not_cross_folds(self):
        from sklearn.model_selection import StratifiedGroupKFold

        x = np.arange(80).reshape(40, 2)
        groups = np.repeat(np.arange(20), 2)
        y = np.repeat(np.arange(20) % 2, 2)
        seen = []
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=42).split(
            x, y, groups
        ):
            self.assertFalse(set(groups[tr]) & set(groups[va]))
            seen.extend(va)
        self.assertEqual(sorted(seen), list(range(40)))

    def test_missing_label_is_not_a_negative(self):
        with self.assertRaises((TypeError, ValueError, RuntimeError)):
            m.binary_fit(np.ones((3, 2)), [0, None, 1], 1, 42)

    def test_offline_guard_blocks_connections(self):
        import socket

        with m.c.offline(), self.assertRaises(RuntimeError):
            socket.create_connection(("example.com", 443))

    def test_feature_normalization_is_deterministic(self):
        self.assertEqual(m.norm("  A\tＢ  "), "a b")


if __name__ == "__main__":
    unittest.main()

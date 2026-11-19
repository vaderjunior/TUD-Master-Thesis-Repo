import unittest
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from analysis.classification.supervised.operating_points import independent_rank
from analysis.classification.supervised.operating_points import probabilities


class ConfidenceTests(unittest.TestCase):
    def test_ties_match_rank_metrics(self):
        y = np.array([1, 0, 1, 0, 0, 1])
        s = np.array([0.8, 0.8, 0.5, 0.5, 0.1, 0.1])
        ap, auc = independent_rank(y, s)
        self.assertAlmostEqual(ap, average_precision_score(y, s))
        self.assertAlmostEqual(auc, roc_auc_score(y, s))

    def test_no_positives_is_unavailable(self):
        self.assertEqual(independent_rank([0, 0], [0.2, 0.3]), (None, None))

    def test_all_positives_has_no_roc(self):
        self.assertEqual(independent_rank([1, 1], [0.2, 0.3]), (1.0, None))

    def test_constant_scores(self):
        ap, auc = independent_rank([1, 0, 0, 0], [0.5] * 4)
        self.assertEqual(ap, 0.25)
        self.assertEqual(auc, 0.5)

    def test_probabilities_numerically_stable(self):
        p = probabilities([[1000, 999], [-1000, -1001]], "binary")
        self.assertTrue(np.allclose(p.sum(axis=1), 1))
        self.assertTrue(np.isfinite(p).all())
        p = probabilities([[-1000, 0, 1000]], "multilabel")
        self.assertTrue(np.isfinite(p).all())
        self.assertEqual(p[0, 1], 0.5)

    def test_binary_tie_and_multilabel_tie_differ(self):
        p = probabilities([[0, 0]], "binary")
        self.assertFalse(bool(p[0, 1] > 0.5))
        p = probabilities([[0]], "multilabel")
        self.assertTrue(bool(p[0, 0] >= 0.5))


if __name__ == "__main__":
    unittest.main()

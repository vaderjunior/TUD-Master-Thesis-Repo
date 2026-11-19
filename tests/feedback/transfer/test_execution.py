import unittest
import numpy as np
from analysis.feedback.transfer import execution_metrics as k


class Checks(unittest.TestCase):
    def votes(self, values, gold=False):
        v = np.tile(np.array(values, dtype=np.int16), (1, 4, 3, 1))
        return k.vote_completion(v, [gold])

    def test_old_era(self):
        self.assertEqual(k.era(np.full((4, 3), 8)), "all_8")

    def test_new_era(self):
        self.assertEqual(k.era(np.full((4, 3), 128)), "all_128")

    def test_cross_era(self):
        self.assertEqual(k.era([[8, 128]]), "cross_boundary")

    def test_invalid_era(self):
        with self.assertRaises(ValueError):
            k.era([32])

    def test_complete_majority(self):
        lo, hi, m = self.votes([1, 1, 0])
        self.assertTrue((lo == 1).all())
        self.assertTrue(np.array_equal(lo, hi))
        self.assertEqual(m.sum(), 0)

    def test_two_agree_positive(self):
        lo, hi, m = self.votes([1, -99, 1])
        self.assertTrue((lo == 1).all())
        self.assertTrue(np.array_equal(lo, hi))

    def test_two_agree_negative(self):
        lo, hi, m = self.votes([0, 0, -99])
        self.assertTrue((hi == 0).all())

    def test_two_split(self):
        lo, hi, m = self.votes([1, 0, -99])
        self.assertTrue((lo == 0).all())
        self.assertTrue((hi == 1).all())

    def test_one_positive(self):
        lo, hi, m = self.votes([-99, 1, -99])
        self.assertTrue((lo == 0).all())
        self.assertTrue((hi == 1).all())

    def test_one_negative(self):
        lo, hi, m = self.votes([-99, 0, -99])
        self.assertTrue((lo == 0).all())
        self.assertTrue((hi == 1).all())

    def test_no_votes(self):
        lo, hi, m = self.votes([-99, -99, -99])
        self.assertTrue((lo == 0).all())
        self.assertTrue((hi == 1).all())
        self.assertEqual(m.sum(), 36)

    def test_gold_hateful_inverts(self):
        lo, hi, m = self.votes([1, -99, 1], True)
        self.assertTrue((hi == 0).all())

    def test_opposing_weights_reference_once(self):
        lo = np.zeros((2, 4, 3), dtype=np.int8)
        hi = lo.copy()
        hi[:, 0, :] = 1
        b = k.opposing(lo, hi)
        self.assertEqual(b["difference"], [-1, 0])
        self.assertEqual(b["variable_output_counts"], [6, 0, 0, 0])

    def test_empty_bounds(self):
        b = k.opposing(np.zeros((0, 4, 3)), np.zeros((0, 4, 3)))
        self.assertIsNone(b["difference"])

    def test_bad_vote_rejected(self):
        with self.assertRaises(ValueError):
            self.votes([0, 2, 1])

    def test_cluster_omission(self):
        sizes = np.array([2, 1])
        counts = np.array([[0, 3, 3, 3], [3, 0, 0, 0]])
        n, num, d = k.omission(counts.sum(axis=0), sizes, counts)
        self.assertEqual(n.tolist(), [1, 2])
        self.assertEqual(d.tolist(), [-1, 0.5])

    def test_single_cluster_undefined(self):
        n, num, d = k.omission([0, 3, 3, 3], [1], [[0, 3, 3, 3]])
        self.assertTrue(np.isnan(d[0]))

    def test_seeds_stable_and_distinct(self):
        self.assertEqual(k.seed("a"), k.seed("a"))
        self.assertNotEqual(k.seed("a"), k.seed("b"))


if __name__ == "__main__":
    unittest.main()

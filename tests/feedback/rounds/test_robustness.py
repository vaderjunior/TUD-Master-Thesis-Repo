import itertools
import unittest
import numpy as np
from analysis.feedback.rounds import robustness_metrics as k


class Robustness(unittest.TestCase):
    def test_reference_weight(self):
        x = np.zeros((2, 4, 10, 3), dtype=int)
        x[:, :, 0, 0] = 1
        x[:, :, 1, 0] = 1
        y = k.collapse(x)
        self.assertEqual(y[0, 3, 0], 3)
        self.assertEqual(y[0, 3, 1], 1)

    def test_contrasts(self):
        self.assertEqual(
            k.contrasts(np.array([[1, 2, 3, 4]])).tolist(), [[1, 2, 3, 4, -1, -2, 1]]
        )

    def test_cluster_weight_is_not_text_weight(self):
        v = np.array([[9], [0], [0]])
        labels = ["a", "b", "b"]
        self.assertAlmostEqual(k.bootstrap(v, labels, 7, draws=8)[0][0], 1 / 3)
        self.assertAlmostEqual(
            k.bootstrap(v, labels, 7, "cluster_equal", draws=8)[0][0], 0.5
        )

    def test_cluster_keeps_whole_members(self):
        v = np.array([[9], [0], [0]])
        _, draw, _, _ = k.bootstrap(v, ["a", "b", "b"], 4, draws=100)
        self.assertTrue(set(draw.ravel()) <= {0, 1 / 3, 1})

    def test_all_missing_majority(self):
        v = np.full((2, 4, 10, 3, 3), -1)
        lo, hi = k.majority_completion(v, [0, 1])
        self.assertTrue((lo == 0).all())
        self.assertTrue((hi == 1).all())

    def test_every_partial_vote_combination(self):
        for pattern in itertools.product([-1, 0, 1], repeat=3):
            v = np.tile(pattern, (2, 4, 10, 3, 1))
            lo, hi = k.majority_completion(v, [0, 1])
            free = [i for i, x in enumerate(pattern) if x == -1]
            gates = []
            for repl in itertools.product([0, 1], repeat=len(free)):
                x = list(pattern)
                for i, b in zip(free, repl):
                    x[i] = b
                gates.append(sum(x) >= 2)
            for truth in [0, 1]:
                errors = [x != truth for x in gates]
                self.assertEqual(lo[truth, 0, 0, 0], min(errors))
                self.assertEqual(hi[truth, 0, 0, 0], max(errors))

    def test_two_agreeing_votes_already_fix_gate(self):
        v = np.tile([1, 1, -1], (1, 4, 10, 3, 1))
        lo, hi = k.majority_completion(v, [1])
        self.assertTrue((lo == hi).all())
        self.assertTrue((lo == 0).all())

    def test_opposing_reference_bound(self):
        x = k.opposing(np.array([[0, 1, 2, 3]]), np.array([[3, 4, 5, 6]]))
        self.assertEqual(x["intervals"]["adaptive-reference"], [-2 / 9, 4 / 9])

    def test_omission_preserves_members(self):
        result = k.omissions(np.array([[9], [0], [0]]), ["a", "b", "b"])
        self.assertEqual(result[0]["remaining"], 2)
        self.assertEqual(result[1]["points"][0], 1)

    def test_single_cluster_omission_undefined(self):
        self.assertIsNone(k.omissions(np.array([[1], [2]]), ["a", "a"])[0]["points"])

    def test_masks_reference_and_pair_are_distinct(self):
        meta = [
            dict(
                population="p",
                gold_hate=False,
                text_cluster="t",
                key="a",
                masks=dict(replay_exclusion_member=False),
            )
        ]
        g = np.zeros((1, 4, 10, 3), dtype=int)
        v = np.full_like(g, 3)
        g[0, 3, 0, 0] = -1
        v[0, 3, 0, 0] = 0
        ex = {name: np.zeros_like(g) for name in ["ties", "repairs", "retries"]}
        m = k.masks(meta, g, v, ex)
        self.assertFalse(m["final"][0])
        self.assertTrue(m["policy_only"][0])
        self.assertTrue(m["paired_static"][0])
        self.assertFalse(m["paired_reference"][0])

    def test_representative_before_validity(self):
        meta = [
            dict(
                population="p",
                gold_hate=False,
                text_cluster="t",
                key=x,
                masks=dict(replay_exclusion_member=False),
            )
            for x in ["a", "b"]
        ]
        rep = next(iter(k.representatives(meta)))
        g = np.zeros((2, 4, 10, 3), dtype=int)
        v = np.full_like(g, 3)
        g[rep, 3, 0, 0] = -1
        v[rep, 3, 0, 0] = 0
        ex = {name: np.zeros_like(g) for name in ["ties", "repairs", "retries"]}
        self.assertEqual(k.masks(meta, g, v, ex)["representative"].sum(), 0)

    def test_opposite_gold_representatives_kept(self):
        meta = [
            dict(population="p", gold_hate=bool(i), text_cluster="t", key=str(i))
            for i in range(2)
        ]
        self.assertEqual(len(k.representatives(meta)), 2)

    def test_empty_bootstrap_rejected(self):
        with self.assertRaises(ValueError):
            k.bootstrap(np.zeros((0, 1), dtype=int), [], 1)

    def test_unknown_vote_rejected(self):
        with self.assertRaises(ValueError):
            k.majority_completion(np.full((1, 4, 10, 3, 3), 2), [0])

    def test_strict_mask_does_not_revote(self):
        meta = [
            dict(
                population="p",
                gold_hate=False,
                text_cluster="t",
                key="a",
                masks=dict(replay_exclusion_member=True),
            )
        ]
        g = np.zeros((1, 4, 10, 3), dtype=int)
        v = np.full_like(g, 3)
        v[0, 3, 1, 0] = 2
        ex = {name: np.zeros_like(g) for name in ["ties", "repairs", "retries"]}
        m = k.masks(meta, g, v, ex)
        self.assertTrue(m["final"][0])
        self.assertFalse(m["strict"][0])
        self.assertFalse(m["replay_excluded"][0])
        self.assertTrue((g == 0).all())


if __name__ == "__main__":
    unittest.main()

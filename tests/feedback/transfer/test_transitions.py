import unittest
from analysis.feedback.transfer import transitions_metrics as k


def meta(**kw):
    return dict(
        gold_hate=True,
        gold_hate_types=None,
        gold_target_groups=None,
        gold_severity=None,
        gold_legal=None,
        **kw
    )


def pred(hate=True, types=None, targets=None, severity="low", legal=None):
    return dict(
        hate=hate,
        hate_type=types or [],
        target_group=targets or [],
        severity=severity if hate else None,
        legal=legal or [],
    )


class Boundaries(unittest.TestCase):
    def test_missing_is_not_a_fix_or_error(self):
        self.assertEqual(k.transition(-1, 1), -1)
        self.assertEqual(k.transition(0, -1), -1)
        self.assertEqual(k.transition(-2, -1), -2)

    def test_unknown_and_known_empty_differ(self):
        m = meta()
        self.assertEqual(k.correctness(m, pred())[4], -2)
        m["gold_target_groups"] = []
        self.assertEqual(k.correctness(m, pred())[4], 1)

    def test_gated_known_empty_is_distinct(self):
        m = meta()
        m["gold_target_groups"] = []
        s = k.correctness(m, pred(False))
        self.assertEqual(s[4:6], [1, 0])
        self.assertEqual(s[1], 0)

    def test_unannotated_legal_change_is_not_scored_wrong(self):
        m = meta()
        x = pred()
        y = pred(legal=["insult_defamation"])
        self.assertNotEqual(k.code(x), k.code(y))
        self.assertEqual(k.correctness(m, x), k.correctness(m, y))

    def test_label_order_is_not_change(self):
        self.assertEqual(
            k.code(pred(types=["irony", "explicit"])),
            k.code(pred(types=["explicit", "irony"])),
        )

    def test_consistent_fix_requires_every_repeat_and_pool(self):
        self.assertEqual(
            k.profile([[0] * 3] + [[1] * 3] * 3)["stability"], "consistent_fix"
        )
        self.assertEqual(
            k.profile([[0] * 3, [1] * 3, [1] * 3, [1, 0, 1]])["stability"], "variable"
        )

    def test_equal_average_can_hide_both_changes(self):
        q = k.profile([[1, 0, 1]] + [[0, 1, 1]] * 3)
        self.assertEqual(q["direction"], "equal_average")
        self.assertTrue(q["offsetting"])

    def test_opposite_pools(self):
        q = k.profile([[1, 0, 0], [1, 1, 0], [0, 0, 0], [1, 0, 0]])
        self.assertTrue(q["opposing_pool_directions"])

    def test_repeat_alignment_changes_gross_not_net(self):
        ref = [0, 1, 0]
        trt = [1, 0, 0]
        gross = set()
        net = set()
        for perm in k.PERMUTATIONS:
            x = [k.transition(ref[r], trt[perm[r]]) for r in range(3)]
            gross.add(x.count(1))
            net.add(x.count(1) - x.count(2))
        self.assertEqual(net, {0})
        self.assertEqual(gross, {0, 1})

    def test_single_vote_does_not_prove_unanimity_of_three(self):
        q = k.vote_stats([pred(), None, None], pred())
        self.assertEqual(q["distinct"][-1], 1)
        self.assertEqual(q["available_pairs"], 0)
        self.assertEqual(q["n_valid"], 1)

    def test_two_to_one_gate_split_is_variation(self):
        q = k.vote_stats([pred(), pred(), pred(False)], pred())
        self.assertEqual(q["distinct"][0], 2)
        self.assertEqual(q["disagree_pairs"][0], 2)

    def test_composite_aggregate_can_be_unseen(self):
        votes = [
            pred(types=["explicit"], targets=["race"]),
            pred(types=["explicit"], targets=["religion"]),
            pred(types=["irony"], targets=["race"]),
        ]
        self.assertFalse(
            k.vote_stats(votes, pred(types=["explicit"], targets=["race"]))[
                "aggregate_not_observed"
            ]
        )
        votes[0] = pred(types=["explicit"], targets=["gender"])
        self.assertTrue(
            k.vote_stats(votes, pred(types=["explicit"]))["aggregate_not_observed"]
        )

    def test_missing_repeat_is_never_stable(self):
        self.assertEqual(
            k.profile([[1] * 3, [1] * 3, [1] * 3, [1, -1, 1]])["status"], "unavailable"
        )

    def test_case_order_is_deterministic(self):
        self.assertEqual(
            min(["a", "b"], key=lambda x: k.rank("slot", x)),
            min(["b", "a"], key=lambda x: k.rank("slot", x)),
        )


if __name__ == "__main__":
    unittest.main()

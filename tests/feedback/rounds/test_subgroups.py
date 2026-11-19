import unittest
import numpy as np
from analysis.feedback.rounds import subgroups_metrics as k
from analysis.feedback.rounds import errors_kernel as prior


class SubgroupBoundaries(unittest.TestCase):
    def meta(self):
        return [
            dict(
                gold_hate=g,
                masks=dict(
                    all_round_common=f,
                    round_common=[True, True, True, True],
                    round_strict=[False] * 4,
                ),
                inference_cluster=c,
                text_cluster=str(i),
                template_id=i,
            )
            for i, (g, f, c) in enumerate(
                [(False, True, "a"), (False, False, "a"), (True, True, "b")]
            )
        ]

    def test_missing_longitudinal_not_final(self):
        m = self.meta()
        self.assertEqual(k.cohort(m, [0, 1, 2], False, "fixed"), [0])
        self.assertEqual(k.cohort(m, [0, 1, 2], False, "final"), [0, 1])

    def test_wrong_class_undefined(self):
        self.assertEqual(k.support(self.meta(), [0, 1], True)["roster"], 0)

    def test_strict_empty_not_zero(self):
        self.assertEqual(k.cohort(self.meta(), [0, 1], False, "strict"), [])

    def test_support_clusters_not_texts(self):
        self.assertEqual(k.support(self.meta(), [0, 1], False)["inference_clusters"], 1)

    def test_unknown_mask(self):
        with self.assertRaises(ValueError):
            k.cohort([], [], False, "bad")

    def test_leave_one_weighting(self):
        v = k.influence([[0], [0], [9]], ["a", "a", "b"])
        self.assertAlmostEqual(v["full"][0], 1 / 3)
        self.assertAlmostEqual(v["equal_cluster_mean"][0], 0.5)
        self.assertEqual([r["values"][0] for r in v["leave_one_out"]], [1.0, 0.0])

    def test_one_cluster_not_uncertain_estimate(self):
        self.assertEqual(
            k.influence([[9], [0]], ["a", "a"])["status"], "undefined_one_cluster"
        )

    def test_empty_influence(self):
        self.assertEqual(
            k.influence(np.empty((0, 2), int), [])["status"], "undefined_no_texts"
        )

    def test_joint_pair_not_average_accuracy(self):
        e = np.zeros((2, 4, 10, 3), int)
        e[0, :, :, 0] = 1
        e[1, :, :, 1] = 1
        self.assertTrue(np.all(k.pair_counts(e, 0, 1) == 3))

    def test_missing_pair_rejected(self):
        e = np.zeros((2, 4, 10, 3), int)
        e[1, 0, 0, 0] = -1
        with self.assertRaises(ValueError):
            k.pair_counts(e, 0, 1)

    def test_reference_three_actual_repeats(self):
        e = np.zeros((1, 4, 10, 3), int)
        e[:, :, 0, 0] = 1
        c, _ = prior.policy_counts(e)
        self.assertTrue(np.all(c[:, :, 0] == 3))

    def test_retained_mean_can_have_errors(self):
        self.assertTrue(
            k.selected_category(
                np.ones((4, 9), int) * 0,
                np.ones((4, 3), int),
                "retained_average_benefit",
            )
        )

    def test_mixed_relative_requires_both_signs(self):
        e = np.zeros((4, 9), int)
        r = np.zeros((4, 3), int)
        e[3, :] = 1
        r[0, :] = 1
        self.assertTrue(k.selected_category(e, r, "mixed_relative_direction"))

    def test_persistent_wrong_distinct_from_baseline(self):
        self.assertTrue(
            k.selected_category(
                np.ones((4, 9), int), np.zeros((4, 3), int), "persistent_error"
            )
        )

    def test_missing_case_not_eligible(self):
        e = np.zeros((4, 9), int)
        e[0, 0] = -1
        self.assertFalse(
            k.selected_category(e, np.ones((4, 3), int), "retained_average_benefit")
        )

    def test_cluster_bootstrap_preserves_multiplicity(self):
        d, labels, sizes = prior.bootstrap(
            np.array([[0], [0], [9]]), ["a", "a", "b"], 7, 100
        )
        self.assertEqual(labels, ["a", "b"])
        self.assertEqual(sizes.tolist(), [2, 1])
        self.assertTrue(set(d[:, 0]) <= {0.0, 1 / 3, 1.0})


if __name__ == "__main__":
    unittest.main()

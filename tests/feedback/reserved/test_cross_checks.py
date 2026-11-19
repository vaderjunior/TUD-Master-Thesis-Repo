from fractions import Fraction as F
import unittest
import numpy as np
from analysis.feedback.reserved import cross_checks_metrics as c


class ScientificInvariants(unittest.TestCase):
    def test_nonlinear_average_is_not_pooled_f1(self):
        g = np.array([0, 0, 1, 1])
        runs = [np.array([0, 0, 0, 0]), np.array([0, 0, 1, 1])]
        average = c.mean([c.score(g, p, False)["binary_macro_f1"] for p in runs])
        pooled = c.score(np.tile(g, 2), np.concatenate(runs), False)["binary_macro_f1"]
        self.assertEqual(average, F(2, 3))
        self.assertNotEqual(average, pooled)

    def test_fixed_seven_labels_and_empty_label(self):
        self.assertEqual(
            c.score(np.array([1]), np.array([1]), True)["type_macro_f1_fixed7"], F(1, 7)
        )
        self.assertEqual(
            c.score(np.array([0]), np.array([0]), True)["type_macro_f1_fixed7"], 0
        )
        self.assertIsNone(
            c.score(np.array([], dtype=int), np.array([], dtype=int), True)[
                "type_exact_set"
            ]
        )

    def test_single_class_balanced_accuracy_undefined(self):
        self.assertIsNone(
            c.score(np.ones(3, dtype=int), np.ones(3, dtype=int), False)[
                "balanced_accuracy"
            ]
        )
        self.assertEqual(
            c.score(np.ones(3, dtype=int), np.ones(3, dtype=int), False)[
                "binary_macro_f1"
            ],
            F(1, 2),
        )

    def fixture(self):
        rows = [
            dict(key=k, cluster=cluster, gold=dict(hate=g, hate_type=types), source="x")
            for k, cluster, g, types in [
                ("a", "dup", False, None),
                ("b", "dup", False, None),
                ("c", "other", True, ["explicit"]),
            ]
        ]
        cond = ["t1", "t2", "t3", "ref"]
        records = []
        for row in rows:
            for cd in cond:
                for rep in range(3):
                    null = row["key"] == "a" and cd == "t1" and rep == 0
                    records.append(
                        dict(
                            key=row["key"],
                            condition=cd,
                            repeat=rep,
                            n_valid=0 if null else 3,
                            result=(
                                None
                                if null
                                else dict(
                                    hate=cd != "ref",
                                    hate_type=["explicit"] if cd != "ref" else [],
                                )
                            ),
                            stage="batch",
                            administrative_unknown_requests=0,
                        )
                    )
        return c.Data(rows, cond, records), dict(
            treatment_conditions=cond[:3], control_conditions=["ref"], endpoint="FPR"
        )

    def test_reference_multiplicity_and_full_roster_bounds(self):
        d, s = self.fixture()
        p = d.rate(s)
        self.assertEqual(p["outputs_per_input"], [9, 3])
        self.assertEqual(p["eligible_n"], 2)
        self.assertEqual(p["common_n"], 1)
        self.assertEqual(p["missing_real_outputs"], [1, 0])
        self.assertEqual(p["difference"], 1)
        self.assertEqual(p["bounds"], [F(17, 18), F(1)])

    def test_representative_is_selected_before_validity(self):
        d, s = self.fixture()
        chosen = c.representative_indices(np.array([0, 1]), d.roster)[0]
        d.valid[chosen, 0] = 0
        d.hate[chosen, 0] = -1
        p = d.rate(s, mask="one_per_cluster")
        self.assertEqual(p["eligible_n"], 1)
        self.assertEqual(p["common_n"], 0)

    def test_unknown_gold_is_not_empty_gold(self):
        d, s = self.fixture()
        p = d.quality(s, True)
        self.assertEqual(p["eligible_n"], 1)
        self.assertEqual(p["common_ids"], ["c"])

    def test_strict_is_valid_votes_not_unanimity(self):
        d, s = self.fixture()
        self.assertEqual(d.rate(s, mask="all_three_votes")["common_n"], 1)
        d.valid[1, 1] = 2
        self.assertEqual(d.rate(s, mask="ordinary_valid")["common_n"], 1)
        self.assertEqual(d.rate(s, mask="all_three_votes")["common_n"], 0)

    def test_exact_zero_and_tiny_positive_do_not_merge(self):
        x = [F(-1), F(0), F(1, 100000)]
        self.assertEqual(c.exact_quantile(x, F(1, 2)), 0)
        self.assertGreater(c.exact_quantile(x, F(3, 4)), 0)

    def test_cluster_item_weighting(self):

        numerators = np.array([2, 0])
        sizes = np.array([2, 1])
        self.assertEqual(F(int(numerators.sum()), int(sizes.sum())), F(2, 3))
        self.assertNotEqual(F(2, 3), (F(2, 2) + F(0, 1)) / 2)


if __name__ == "__main__":
    unittest.main()

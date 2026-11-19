import unittest
import numpy as np
from analysis.feedback.supplements import scores as s


def fixture(factor=False):
    d = object.__new__(s.Data)
    d.study = "factorial" if factor else "human_meaning"
    d.conditions = (
        ["E0D0", "E1D0", "E0D1", "E1D1"]
        if factor
        else [f"{p}_s{seed}" for seed in s.SEEDS for p in ["plain", "human", "generic"]]
    )
    d.keys = [(c, r) for c in d.conditions for r in range(3)]
    d.kpos = {k: n for n, k in enumerate(d.keys)}
    d.ids = [f"key-{i:02d}" for i in range(14)]
    d.pos = {i: n for n, i in enumerate(d.ids)}
    d.roster = {
        i: {
            "key": i,
            "original_key": i,
            "cluster": i,
            "gold": {
                "hate": True,
                "hate_type": [s.LABELS[n // 2]] if n % 2 == 0 else [],
            },
            "target_label": s.LABELS[n // 2],
        }
        for n, i in enumerate(d.ids)
    }
    d.y = np.array(
        [[l in d.roster[i]["gold"]["hate_type"] for l in s.LABELS] for i in d.ids],
        dtype=bool,
    )
    d.truth = np.ones(14, dtype=bool)
    d.valid = np.ones((len(d.keys), 14), dtype=bool)
    d.votes = np.full(d.valid.shape, 3)
    d.x = np.zeros((len(d.keys), 14, 7), dtype=bool)
    d.gate = np.ones(d.valid.shape, dtype=bool)
    for k, (c, r) in enumerate(d.keys):
        if (factor and c.startswith("E1")) or (not factor and c.startswith("human")):
            d.x[k] = d.y
    return d


class StatisticsTests(unittest.TestCase):
    def test_factorial_examples_only_effect_and_category_weighting(self):
        d = fixture(True)
        out = s.factorial(d, d.ids)
        self.assertAlmostEqual(
            out["effects"]["marginal_example_block_effect"]["difference"], 1
        )
        self.assertAlmostEqual(
            out["effects"]["marginal_definition_block_effect"]["difference"], 0
        )
        self.assertAlmostEqual(out["effects"]["interaction"]["difference"], 0)
        self.assertEqual(out["common_positive_support_by_category"], [1] * 7)
        self.assertTrue(out["primary_estimable"])

    def test_human_average_uses_nine_runs_and_seven_labels(self):
        d = fixture()
        groups = {p: [f"{p}_s{seed}" for seed in s.SEEDS] for p in ["plain", "human"]}
        out = s.human_compare(d, d.ids, groups, 7)
        self.assertEqual(len(out["settings"]), 18)
        self.assertAlmostEqual(out["contrasts"]["human_minus_plain"]["difference"], 1)
        self.assertEqual(out["groups"]["human"]["pool_means"], [1, 1, 1])

    def test_common_masks_do_not_drop_only_one_condition(self):
        d = fixture(True)
        d.valid[0, 0] = False
        d.votes[1, 1] = 2
        self.assertEqual(len(d.common(d.ids, list(range(12)))), 13)
        self.assertEqual(len(d.common(d.ids, list(range(12)), True)), 12)
        d.votes[0, 0] = 0
        self.assertEqual(len(d.common(d.ids, list(range(12)), True)), 12)

    def test_bounds_do_not_treat_missing_as_observed_negative(self):
        d = fixture()
        d.valid[0, 0] = False
        worst = s.run_metric(d, d.ids, 0, "worst")
        best = s.run_metric(d, d.ids, 0, "best")
        own = s.run_metric(d, d.ids, 0)
        self.assertEqual(worst["n"], 14)
        self.assertEqual(best["n"], 14)
        self.assertEqual(own["n"], 13)
        self.assertGreater(best["macro_f1"], worst["macro_f1"])

    def test_clusters_keep_pair_rows_together(self):
        d = fixture(True)
        d.roster[d.ids[1]]["cluster"] = d.roster[d.ids[0]]["cluster"]
        w, member, picks = s.weights(d.ids, d.roster, 9, 100)
        self.assertEqual(member[0], member[1])
        self.assertEqual(w.shape, (100, 13))
        self.assertTrue(np.all(w.sum(axis=1) == 13))

    def test_representative_keeps_every_pair_of_selected_text(self):
        d = fixture(True)
        for i in d.ids[:3]:
            d.roster[i]["cluster"] = "shared"
        for i in d.ids[:2]:
            d.roster[i]["original_key"] = "same-text"
        selected = s.representatives(d, d.ids)
        self.assertEqual(d.ids[0] in selected, d.ids[1] in selected)
        self.assertNotEqual(d.ids[0] in selected, d.ids[2] in selected)

    def test_zero_support_category_is_not_removed(self):
        d = fixture(True)
        d.valid[:, 0] = False
        out = s.factorial(d, d.ids)
        self.assertFalse(out["primary_estimable"])
        self.assertEqual(out["common_positive_support_by_category"][0], 0)
        self.assertEqual(len(out["groups"]["E1D1"]["per_category_f1"]), 7)

    def test_negative_coefficients_use_opposite_missing_bounds(self):
        z = s.effect(
            {"a": 0.7, "b": 0.3},
            {"a": np.array([0.7, 0.7]), "b": np.array([0.3, 0.3])},
            {"a": [0.6, 0.8], "b": [0.1, 0.4]},
            {"a": 1, "b": -1},
        )
        np.testing.assert_allclose(z["full_roster_missing_bounds"], [0.2, 0.7])


if __name__ == "__main__":
    unittest.main()

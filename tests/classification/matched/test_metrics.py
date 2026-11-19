import unittest
import numpy as np
from analysis.classification.matched import metrics as q


def pred(hate=True, types=None, legal=None):
    return dict(
        hate=hate,
        hate_type=(types or []) if hate else [],
        target_group=[],
        severity=None,
        legal=legal or [],
    )


def row(i, gate=True, types=None, legal=None):
    return dict(
        id=str(i),
        gate=gate,
        hate_types=types,
        target_groups=None,
        severity=None,
        legal=legal,
        source="a",
    )


class ReviewTests(unittest.TestCase):
    def setUp(self):
        self.rows = [row(i, i < 12, ["x"] if i < 12 else None, []) for i in range(24)]
        self.tax = {
            "dimensions": {
                "hate_type": {"labels": ["x", "y"]},
                "legal": {"labels": ["p"]},
            }
        }
        self.ep = q.s.endpoint(self.rows, "hate", self.tax)
        self.out = {r["id"]: pred(r["gate"]) for r in self.rows}

    def test_fixed_labels_require_ten(self):
        ep = q.s.endpoint(
            [row(i, True, ["x"] if i < 10 else ["y"]) for i in range(19)],
            "hate_type",
            self.tax,
        )
        self.assertEqual(ep["macro_labels"], ["x"])
        self.assertEqual(ep["rare_labels"], ["y"])

    def test_empty_not_missing(self):
        ep = q.s.endpoint(
            [row(0, None, legal=[]), row(1, None, legal=None)], "legal", self.tax
        )
        self.assertEqual(ep["scorable_ids"], ["0"])

    def test_fine_gold_gate(self):
        self.assertIsNone(q.s.original.truth(row(0, False, ["x"]), "hate_type"))
        self.assertEqual(q.s.original.truth(row(0, None, legal=["p"]), "legal"), ["p"])

    def test_invalid_is_not_negative(self):
        self.out["0"] = None
        v = q.s.contributions(self.rows, self.out, self.ep)
        self.assertFalse(v["valid"][0])
        self.assertEqual(v["counts"][0].sum(), 0)
        score = q.s.original.score(v)
        self.assertEqual(score["macro_f1_valid_only"], 1)
        self.assertLess(score["invalid_inclusive_bounds"][0], 1)
        self.assertEqual(score["invalid_inclusive_bounds"][1], 1)

    def test_reject_missing_identity(self):
        del self.out["0"]
        with self.assertRaises(ValueError):
            q.s.contributions(self.rows, self.out, self.ep)

    def test_reject_ungated_prediction(self):
        self.out["0"] = pred(False)
        self.out["0"]["hate_type"] = ["x"]
        with self.assertRaises(Exception):
            q.s.contributions(self.rows, self.out, self.ep)

    def test_known_exact_ignores_unknown_fields(self):
        r = row(0, True, types=None)
        p = pred(True, ["x"])
        p["target_group"] = ["unscored"]
        self.assertTrue(q.exact_known(r, p))
        self.assertIsNone(q.exact_known(r, None))
        self.assertFalse(q.exact_known(row(0, True, types=[]), p))

    def test_mask_zeros_counts_without_changing_labels(self):
        v = q.s.contributions(self.rows, self.out, self.ep)
        w = q.masked(v, [False] + [True] * 23)
        self.assertEqual(w["counts"][0].sum(), 0)
        self.assertEqual(v["counts"][0].sum(), 1)
        self.assertEqual(w["macro_indices"], v["macro_indices"])

    def test_bootstrap_identity_and_saved_draws(self):
        v = q.s.contributions(self.rows, self.out, self.ep)
        result, draws = q.captured_contrast(
            [v] * 3, [v] * 9, [str(i // 2) for i in range(24)], draws=257, family=6
        )
        self.assertTrue(np.array_equal(draws, np.zeros(257)))
        self.assertEqual(result["family_interval"], [0.0, 0.0])
        self.assertEqual(result["family_confidence"], 1 - 0.05 / 6)
        self.assertEqual(result["n_clusters"], 12)

    def test_stratified_repeated_cluster_not_split(self):
        v = q.s.contributions(self.rows, self.out, self.ep)
        result, _ = q.captured_contrast(
            [v],
            [v],
            [str(i // 2) for i in range(24)],
            sources=["a", "b"] * 12,
            stratify=True,
            draws=65,
        )
        self.assertEqual(result["cluster_strata"], {"a|b": 12})

    def test_common_mask_over_all_variants(self):
        left = q.s.contributions(self.rows, self.out, self.ep)
        self.out["0"] = None
        right = q.s.contributions(self.rows, self.out, self.ep)
        result, _ = q.captured_contrast(
            [left] * 3, [right] * 9, [str(i) for i in range(24)], draws=65
        )
        self.assertEqual(result["n_common_valid"], 23)
        self.assertEqual(result["excluded_ids"], ["0"])

    def test_hate_confusion_denominators(self):
        self.out["0"] = pred(False)
        self.out["12"] = pred(True)
        self.out["13"] = None
        d = q.diagnostic(self.rows, self.out)
        self.assertEqual(d["counts"]["fp"], 1)
        self.assertEqual(d["counts"]["fn"], 1)
        self.assertEqual(d["fpr"], 1 / 11)
        self.assertEqual(d["fnr"], 1 / 12)

    def test_vote_tie_retains_independent_legal(self):
        p = q.m.gold.independent_aggregate(
            [pred(True, ["x"], ["p"]), pred(False, legal=["p"]), None]
        )
        self.assertFalse(p["hate"])
        self.assertEqual(p["hate_type"], [])
        self.assertEqual(p["legal"], ["p"])

    def test_nested_sensitivity_is_not_main_stratification(self):
        from analysis.classification.matched.validate import stratified_record

        self.assertFalse(
            stratified_record({"source_stratified": {"source_stratified": True}})
        )
        self.assertTrue(stratified_record({"source_stratified": True}))
        self.assertFalse(stratified_record({"source_stratified": False}))

    def test_absent_field_is_unannotated(self):
        from analysis.classification.matched.validate import truth

        self.assertIsNone(truth({"gate": True}, "legal"))
        self.assertIsNone(truth({"gate": True}, "target_group"))


if __name__ == "__main__":
    unittest.main()

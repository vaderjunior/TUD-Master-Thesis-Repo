"""Tests for common scoring."""

import copy
import unittest
import numpy as np
from analysis.common import scoring as s


class ScoringTests(unittest.TestCase):

    def gate(self, predictions):
        rows = [{"id": str(i), "gate": y} for i, y in enumerate([True, True, False, False])]
        ep = {
            "all_labels": ["not_hate", "hate"],
            "macro_labels": ["not_hate", "hate"],
            "dimension": "hate",
            "scorable_ids": ["0", "1", "2", "3"],
        }
        outs = {
            str(i): {"result": None if p is None else {"hate": p}, "n_valid": 0 if p is None else 3}
            for i, p in enumerate(predictions)
        }
        return s.contributions(rows, outs, ep)

    def test_legal_missing_empty_and_gate_independence(self):
        self.assertIsNone(s.truth({"gate": False, "legal": None}, "legal"))
        self.assertEqual(s.truth({"gate": None, "legal": []}, "legal"), [])
        self.assertEqual(
            s.truth({"gate": False, "legal": ["insult_defamation"]}, "legal"), ["insult_defamation"]
        )
        self.assertIsNone(s.truth({"gate": False, "hate_types": ["explicit"]}, "hate_type"))
        self.assertEqual(s.truth({"gate": True, "hate_types": []}, "hate_type"), [])

    def test_numpy_and_nan_gate_mask(self):
        self.assertIsNone(s.truth({"gate": float("nan")}, "hate"))
        self.assertIs(s.truth({"gate": np.bool_(False)}, "hate"), False)
        with self.assertRaises(ValueError):
            s.truth({"gate": "False"}, "hate")

    def test_gate_confusion_arithmetic(self):
        c = self.gate([True, False, True, False])
        out = s.score(c)
        self.assertEqual(out["macro_f1_valid_only"], 0.5)
        self.assertEqual(out["per_label_counts"], [[1, 1, 1], [1, 1, 1]])

    def test_invalid_is_excluded_and_bounds_include_it(self):
        out = s.score(self.gate([True, None, False, False]))
        self.assertEqual(out["n_valid"], 3)
        self.assertEqual(out["coverage"], 0.75)
        self.assertEqual(out["macro_f1_valid_only"], 1)
        self.assertAlmostEqual(out["invalid_inclusive_bounds"][0], (2 / 3 + 4 / 5) / 2)
        self.assertEqual(out["invalid_inclusive_bounds"][1], 1)

    def test_all_invalid_has_no_valid_only_score(self):
        out = s.score(self.gate([None] * 4))
        self.assertIsNone(out["macro_f1_valid_only"])
        self.assertEqual(out["invalid_inclusive_bounds"], [0, 1])

    def test_partial_run_rejected(self):
        with self.assertRaisesRegex(ValueError, "Missing"):
            s.contributions(
                [{"id": "a", "gate": True}],
                {},
                {
                    "dimension": "hate",
                    "all_labels": ["not_hate", "hate"],
                    "macro_labels": ["hate"],
                    "scorable_ids": ["a"],
                },
            )

    def test_fixed_support_does_not_depend_on_predictions(self):
        tax = {"dimensions": {"hate_type": {"labels": {"explicit": {}, "irony": {}}}}}
        rows = [
            {"id": str(i), "gate": True, "hate_types": ["explicit"] if i < 10 else ["irony"]}
            for i in range(19)
        ]
        ep = s.roster("en_dev_eval_sq3_types", rows, tax)
        self.assertEqual(ep["macro_labels"], ["explicit"])
        self.assertEqual(ep["gold_support"], {"explicit": 10, "irony": 9})

    def test_missing_severity_is_error_on_valid_gold_item(self):
        ep = {
            "dimension": "severity",
            "all_labels": ["low", "high"],
            "macro_labels": ["low"],
            "scorable_ids": ["a"],
        }
        c = s.contributions(
            [{"id": "a", "gate": True, "severity": "low"}],
            {"a": {"result": {"severity": None}, "n_valid": 3}},
            ep,
        )
        self.assertEqual(s.score(c)["macro_f1_valid_only"], 0)

    def test_identical_settings_zero_paired_interval(self):
        c = self.gate([True, False, True, False])
        out = s.contrast([c] * 3, [c] * 9, draws=100, primary=True)
        self.assertEqual(out["mean_difference"], 0)
        self.assertEqual(out["ci95"], [0, 0])
        self.assertEqual(out["family_ci"], [0, 0])
        self.assertEqual(out["family_confidence"], 0.9875)

    def test_common_valid_is_shared_across_every_repeat(self):
        a = self.gate([True, None, False, False])
        b = self.gate([None, True, False, False])
        out = s.contrast([a] * 3, [b] * 9, draws=50)
        self.assertEqual(out["common_valid_ids"], ["2", "3"])
        self.assertEqual(out["mean_difference"], 0)

    def test_mean_setting_f1_is_not_pooled_votes(self):
        perfect = self.gate([True, True, False, False])
        bad = self.gate([False, False, True, True])
        out = s.contrast([perfect] * 3, [perfect, bad, bad], draws=50)
        self.assertAlmostEqual(out["mean_difference"], 2 / 3)

    def test_zero_denominator_fixed_label_has_explicit_zero(self):
        self.assertEqual(float(s.macro(np.array([[0, 0, 0], [3, 0, 0]]), [0, 1])), 0.5)


if __name__ == "__main__":
    unittest.main()

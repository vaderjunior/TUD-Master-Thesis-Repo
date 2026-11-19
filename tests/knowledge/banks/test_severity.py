"""Tests for banks severity."""

import unittest
import numpy as np
from analysis.knowledge.banks import severity as r


class SeverityTests(unittest.TestCase):

    def ep(self, ids):
        return {
            "dimension": "severity",
            "all_labels": r.LABELS,
            "macro_labels": r.LABELS,
            "scorable_ids": ids,
        }

    def output(self, value, gate=True):
        return {"result": {"hate": gate, "severity": value}, "n_valid": 3}

    def test_gate_suppression_is_missed_class_not_null(self):
        rows = [
            {"id": "a", "gate": True, "severity": "low"},
            {"id": "b", "gate": True, "severity": "medium"},
            {"id": "c", "gate": True, "severity": "high"},
        ]
        out = {"a": self.output(None, False), "b": self.output("medium"), "c": self.output("high")}
        _, score = r.checked_score(rows, out, self.ep(["a", "b", "c"]))
        self.assertEqual(score["macro_f1_valid_only"], 2 / 3)
        self.assertEqual(score["n_invalid"], 0)
        self.assertEqual(score["per_label_counts"], [[0, 0, 1], [1, 0, 0], [1, 0, 0]])
        self.assertEqual(score["severity_confusion"]["low"]["none_gate_false"], 1)

    def test_missing_severity_with_true_gate_is_distinct(self):
        rows = [{"id": "a", "gate": True, "severity": "low"}]
        _, score = r.checked_score(rows, {"a": self.output(None)}, self.ep(["a"]))
        self.assertEqual(score["n_valid"], 1)
        self.assertEqual(score["macro_f1_valid_only"], 0)
        self.assertEqual(score["severity_confusion"]["low"]["none_gate_true"], 1)

    def test_null_bounds_relax_severity_exclusivity(self):
        rows = [
            {"id": "a", "gate": True, "severity": "low"},
            {"id": "b", "gate": True, "severity": "medium"},
            {"id": "c", "gate": True, "severity": "high"},
        ]
        out = {
            "a": self.output("low"),
            "b": self.output("medium"),
            "c": {"result": None, "n_valid": 0},
        }
        _, score = r.checked_score(rows, out, self.ep(["a", "b", "c"]))
        self.assertEqual(score["n_invalid"], 1)
        self.assertEqual(score["macro_f1_valid_only"], 2 / 3)
        self.assertAlmostEqual(score["invalid_inclusive_bounds"][0], 4 / 9)
        self.assertEqual(score["invalid_inclusive_bounds"][1], 1)

    def test_gold_mask_and_numpy_gate(self):
        rows = [
            {"id": "a", "gate": np.bool_(True), "severity": "low"},
            {"id": "b", "gate": False, "severity": "high"},
            {"id": "c", "gate": True, "severity": None},
            {"id": "d", "gate": None, "severity": "medium"},
        ]
        out = {row["id"]: self.output("low") for row in rows}
        _, score = r.checked_score(rows, out, self.ep(["a"]))
        self.assertEqual(score["n_scorable"], 1)

    def test_contradictory_validity_and_gating_rejected(self):
        rows = [{"id": "a", "gate": True, "severity": "low"}]
        for output in [
            {"result": None, "n_valid": 1},
            {"result": {"hate": False, "severity": "low"}, "n_valid": 3},
        ]:
            with self.assertRaises(Exception):
                r.independent_counts(rows, {"a": output}, self.ep(["a"]))

    def test_missing_or_duplicate_identity_rejected(self):
        rows = [{"id": "a", "gate": True, "severity": "low"}]
        with self.assertRaises(Exception):
            r.independent_counts(rows, {}, self.ep(["a"]))
        with self.assertRaises(Exception):
            r.independent_counts(rows + rows, {"a": self.output("low")}, self.ep(["a"]))

    def test_condition_matrix_cannot_alias_eligible_uniform(self):
        names = ["rag_full", "rag_small", "zero_shot", "encoder_full", "encoder_kb"]
        names += ["fewshot_full_s" + str(i) for i in (42, 43, 44)] + [
            "fewshot_eligible_severity_s" + str(i) for i in (42, 43, 44)
        ]
        matrix = {(name, rep): {"a": self.output("low")} for name in names for rep in range(3)}
        r.validate_settings(matrix, ["a"])
        del matrix["fewshot_eligible_severity_s42", 0]
        with self.assertRaises(Exception):
            r.validate_settings(matrix, ["a"])


if __name__ == "__main__":
    unittest.main()

from copy import deepcopy
import unittest
from analysis.feedback.reserved import quality_diagnostics as d
from tests.feedback.reserved.test_support import fixture


class DiagnosticTests(unittest.TestCase):
    def test_exclusive_categories_unknown_and_null(self):
        row = {"key": "x", "gold": {"hate": True, "hate_type": ["irony", "grievance"]}}
        b = d.bucket()
        cases = [
            (["irony", "grievance"], "exact"),
            (["irony", "grievance", "explicit"], "only_extra"),
            (["irony"], "only_missing"),
            (["explicit"], "both_extra_and_missing"),
        ]
        for labels, expected in cases:
            value = d.outcome(
                row, {"n_valid": 3, "result": {"hate": True, "hate_type": labels}}
            )
            self.assertEqual(value["category"], expected)
            d.update(b, value)
        unknown = deepcopy(row)
        unknown["gold"]["hate_type"] = None
        d.update(
            b,
            d.outcome(
                unknown,
                {"n_valid": 3, "result": {"hate": True, "hate_type": ["explicit"]}},
            ),
        )
        d.update(b, d.outcome(row, {"n_valid": 0, "result": None}))
        result = d.finish(b)
        self.assertEqual(result["counts"]["known_type_outputs"], 4)
        self.assertEqual(result["counts"]["contains_gold"], 2)
        self.assertEqual(result["counts"]["unknown_type_gold"], 1)
        self.assertEqual(result["counts"]["nulls"], 1)

    def test_reconciles_original_run_counts(self):
        rows, pred, _ = fixture()
        pred["c", "t1", 0]["result"] = {"hate": False, "hate_type": []}
        b = d.bucket()
        for row in rows:
            d.update(b, d.outcome(row, pred[row["key"], "t1", 0]))
        d.verify_against_frozen(b, d.a.support.metric_rows(rows, pred, "t1", 0))
        result = d.finish(b)
        self.assertEqual(result["per_label"]["explicit"]["fn_gate_negative"], 1)
        self.assertEqual(result["counts"]["gate_negative"], 1)

    def test_matched_pair_gate_and_identity_partition(self):
        rows, pred, spec = fixture()
        for (key, c, rep), value in pred.items():
            value["messages_sha256"] = key
        pred["c", "t1", 0]["result"] = {"hate": False, "hate_type": []}
        pred["c", "t1", 0]["messages_sha256"] = "changed"
        result = d.conditional_pairs(rows, pred, {"spec": spec}, ["c", "d"])
        self.assertEqual(result["all"]["counts"]["pairs"], 18)
        self.assertEqual(result["both_gates_positive"]["counts"]["pairs"], 17)
        self.assertEqual(result["one_or_both_gates_negative"]["counts"]["pairs"], 1)
        self.assertEqual(result["changed_messages"]["counts"]["exact_regression"], 1)
        point = d.a.independent.paired_point(rows, pred, spec, True)
        self.assertAlmostEqual(
            result["all"]["exact_difference"],
            point["metrics"]["type_exact_set"]["difference"],
        )


if __name__ == "__main__":
    unittest.main()

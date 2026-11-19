import unittest
import numpy as np
from analysis.common import replay as c
from analysis.classification.inner_votes import accounting


def result(hate=True, types=None, severity="low", legal=None):
    return {
        "hate": hate,
        "target_group": [],
        "hate_type": types or [],
        "severity": severity if hate else None,
        "legal": legal or [],
    }


class OfflineTests(unittest.TestCase):

    def test_null_first_is_not_first_valid(self):
        slots = [None, result(types=["explicit"]), result(types=["explicit"])]
        self.assertIsNone(slots[0])
        self.assertEqual(c.independent_aggregate(slots)["hate_type"], ["explicit"])

    def test_all_null_stays_null(self):
        self.assertIsNone(c.independent_aggregate([None, None, None]))

    def test_tie_clears_fine_but_preserves_legal(self):
        x = c.independent_aggregate(
            [result(legal=["insult"]), result(False, legal=["insult"]), None]
        )
        self.assertFalse(x["hate"])
        self.assertEqual(x["legal"], ["insult"])
        self.assertIsNone(x["severity"])

    def test_strict_majority_lower_median(self):
        x = c.independent_aggregate(
            [
                result(types=["explicit"], severity="high"),
                result(types=[], severity="low"),
                None,
            ]
        )
        self.assertEqual(x["hate_type"], [])
        self.assertEqual(x["severity"], "low")

    def test_gold_empty_is_observed_missing_is_not(self):
        records = [
            {"id": "a", "gate": True, "hate_types": []},
            {"id": "b", "gate": True, "hate_types": None},
            {"id": "c", "gate": False, "hate_types": []},
        ]
        ep = {
            "dimension": "hate_type",
            "all_labels": ["explicit", "irony"],
            "macro_labels": ["explicit", "irony"],
            "scorable_ids": ["a"],
        }
        z = c.contributions(
            records, {r["id"]: result(types=["explicit"]) for r in records}, ep
        )
        self.assertEqual(z["ids"], ["a"])
        self.assertEqual(z["counts"][0, 0, 1], 1)

    def test_missing_not_negative_and_bounds(self):
        records = [{"id": "a", "gate": True}, {"id": "b", "gate": False}]
        ep = {
            "dimension": "hate",
            "all_labels": ["not_hate", "hate"],
            "macro_labels": ["not_hate", "hate"],
            "scorable_ids": ["a", "b"],
        }
        z = c.contributions(records, {"a": None, "b": result(False)}, ep)
        s = c.summarize(z)
        self.assertEqual(s["n_invalid"], 1)
        self.assertEqual(s["per_label_counts"], [[1, 0, 0], [0, 0, 0]])
        self.assertEqual(s["invalid_inclusive_bounds"][1], 1.0)
        self.assertLess(s["invalid_inclusive_bounds"][0], s["macro_f1_valid_only"])

    def test_legal_independent_of_gate(self):
        records = [{"id": "a", "gate": False, "legal": ["x"]}]
        ep = {
            "dimension": "legal",
            "all_labels": ["x", "y"],
            "macro_labels": ["x"],
            "scorable_ids": ["a"],
        }
        z = c.contributions(records, {"a": result(False, legal=["x"])}, ep)
        self.assertEqual(c.summarize(z)["macro_f1_valid_only"], 1.0)

    def test_cluster_bootstrap_and_identity_zero(self):
        records = [{"id": str(i), "gate": bool(i % 2)} for i in range(4)]
        ep = {
            "dimension": "hate",
            "all_labels": ["not_hate", "hate"],
            "macro_labels": ["not_hate", "hate"],
            "scorable_ids": [str(i) for i in range(4)],
        }
        z = c.contributions(records, {r["id"]: result(r["gate"]) for r in records}, ep)
        out = c.bootstrap([z], [z], ["a", "a", "b", "c"], draws=207, family_size=2)
        self.assertEqual(out["ci95"], [0.0, 0.0])
        self.assertEqual(out["n_text_clusters"], 3)
        self.assertEqual(out["direct_resample_checks"], 2)

    def test_agreement_absence_and_missing_explicit(self):
        a = c.pair_agreement(
            [
                (None, result()),
                (result(types=[]), result(types=[])),
                (result(types=["irony"]), result(types=[])),
            ],
            "hate_type",
            ["irony"],
        )
        self.assertEqual(a["possible_pairs"], 3)
        self.assertEqual(a["valid_pairs"], 2)
        self.assertEqual(a["endpoint_exact_agreement"], 0.5)
        self.assertEqual(a["per_label"]["irony"]["positive_on_either"], 1)

    def test_cost_counts_retry_and_repair_with_slot_filter(self):
        rows = []
        for vote, kind, attempt, action, tokens in [
            (0, "ordinary", 0, "retry", None),
            (0, "ordinary", 1, "ok", 10),
            (0, "repair", 0, "ok", 20),
            (1, "ordinary", 0, "ok", 30),
        ]:
            rows.append(
                {
                    "vote": vote,
                    "kind": kind,
                    "attempt": attempt,
                    "action": action,
                    "elapsed_s": 1.0,
                    "backoff_elapsed_s": 0.5,
                    "model_response_metadata": (
                        {"usage": {"total_tokens": tokens}} if tokens else None
                    ),
                }
            )
        a = accounting(rows, {0})
        self.assertEqual(a["post_attempts"], 3)
        self.assertEqual(a["retry_attempts"], 1)
        self.assertEqual(a["repair_attempts"], 1)
        self.assertEqual(a["total_tokens"], 30)


if __name__ == "__main__":
    unittest.main()

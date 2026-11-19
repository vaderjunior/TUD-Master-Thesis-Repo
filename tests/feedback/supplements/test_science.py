from copy import deepcopy
import unittest
from analysis.feedback.balanced import selection_pools as p
from analysis.common.prompt import corrected_gold
from analysis.feedback.supplements.prepare import interpretation_note
from analysis.feedback.supplements.prepare import add_note


def statuses(n=40):
    return {
        f"{gate}-{status}-{i}": {"hate": gate, "status": status}
        for gate in (False, True)
        for status in ("error", "correct")
        for i in range(n)
    }


class ScienceTests(unittest.TestCase):

    def test_binary_selection_ignores_type_disagreement(self):
        roster = [
            {"id": "a", "gold": {"hate": True}},
            {"id": "b", "gold": {"hate": False}},
        ]
        predictions = {
            (i, rep): {"hate": gate, "hate_type": ["wrong_type"]}
            for i, gate in [("a", True), ("b", False)]
            for rep in range(3)
        }
        result = p.classify_candidates(roster, predictions)
        self.assertEqual({r["status"] for r in result.values()}, {"correct"})

    def test_majority_and_null_exclusion(self):
        roster = [
            {"id": "a", "gold": {"hate": True}},
            {"id": "b", "gold": {"hate": False}},
        ]
        preds = {
            ("a", 0): {"hate": True},
            ("a", 1): {"hate": False},
            ("a", 2): {"hate": False},
            ("b", 0): None,
            ("b", 1): {"hate": True},
            ("b", 2): {"hate": True},
        }
        result = p.classify_candidates(roster, preds)
        self.assertEqual(result["a"]["status"], "error")
        self.assertEqual(result["b"]["status"], "invalid")

    def test_missing_response_matrix_fails(self):
        with self.assertRaises(AssertionError):
            p.classify_candidates([{"id": "a", "gold": {"hate": False}}], {})

    def test_balance_and_shared_positive_half(self):
        s = statuses()
        result = p.select_pools(s)
        self.assertEqual(result["budget"], 40)
        self.assertEqual(len(result["pools"]), 18)
        pools = {r["condition"]: r for r in result["pools"]}
        for seed in p.SEEDS:
            for arm in ("error", "correct", "random"):
                positive = pools[f"{arm}_positive_s{seed}"]["ids"]
                balanced = pools[f"{arm}_balanced_s{seed}"]["ids"]
                self.assertTrue(all((s[i]["hate"] for i in positive)))
                self.assertEqual(balanced[:20], positive[:20])
                self.assertEqual(sum((s[i]["hate"] for i in balanced)), 20)
                self.assertEqual(len(set(balanced)), 40)

    def test_twenty_budget_and_insufficient_supply(self):
        self.assertEqual(p.select_pools(statuses(39))["budget"], 20)
        self.assertEqual(p.select_pools(statuses(20))["budget"], 20)
        self.assertEqual(
            p.select_pools(statuses(19))["status"],
            "insufficient_predeclared_candidate_supply",
        )

    def test_invalids_never_enter_random_control(self):
        s = statuses(40)
        s["invalid-case"] = {"hate": False, "status": "invalid"}
        self.assertFalse(
            any(("invalid-case" in pool["ids"] for pool in p.select_pools(s)["pools"]))
        )

    def test_pool_selection_order_invariant_and_inputs_unchanged(self):
        s = statuses(45)
        before = deepcopy(s)
        self.assertEqual(
            p.select_pools(s), p.select_pools(dict(reversed(list(s.items()))))
        )
        self.assertEqual(s, before)

    def test_binary_feedback_does_not_impute_other_heads(self):
        for hate in (False, True):
            r = p.gate_only_record(
                {
                    "id": "x",
                    "source": "implicit_hate",
                    "text": "synthetic",
                    "gold": {"hate": hate, "hate_type": ["irony"]},
                }
            )
            self.assertEqual(
                r["meta"],
                {
                    "gate": hate,
                    "hate_types": None,
                    "target_groups": None,
                    "severity": None,
                    "legal": None,
                },
            )
            text = corrected_gold(r["meta"])
            self.assertIn("true" if hate else "false", text)
            self.assertNotIn("irony", text)
            self.assertNotIn("legal=", text)

    def test_interpretation_strings_are_attributed_and_escaped(self):
        note = interpretation_note(
            {"target": 'synthetic "target"', "implied_statement": "line one\nline two"}
        )
        self.assertIn("Annotator interpretation of the source:", note)
        self.assertNotIn("\n", note)
        self.assertIn("\\n", note)
        self.assertIn('\\"target\\"', note)

    def test_note_preserves_system_and_query_and_original(self):
        base = {
            "text": "synthetic query",
            "messages": [
                {"role": "system", "text": "fixed"},
                {
                    "role": "user",
                    "text": "original labelled source\n\n## Text to classify (language: en)\nsynthetic query",
                },
            ],
        }
        before = deepcopy(base)
        result = add_note(base, "synthetic note")
        self.assertEqual(base, before)
        self.assertEqual(result[0], base["messages"][0])
        self.assertTrue(
            result[1]["text"].endswith(
                "\n\n## Text to classify (language: en)\nsynthetic query"
            )
        )
        with self.assertRaises(AssertionError):
            add_note({**base, "text": "different query"}, "note")

"""Tests for german guidelines context."""

import json
import time
import unittest
import numpy as np
from analysis.knowledge.german_guidelines import context as x


class ContextChecks(unittest.TestCase):

    def test_case_gold_distinguishes_missing_from_empty(self):
        self.assertEqual(x.case_gold({"gate": False}), {"gold_hate": False, "gold_legal": None})
        self.assertEqual(
            x.case_gold({"gate": None, "legal": []}), {"gold_hate": None, "gold_legal": []}
        )
        self.assertEqual(
            x.case_gold({"legal": ["propaganda"]}),
            {"gold_hate": None, "gold_legal": ["propaganda"]},
        )

    def setUp(self):
        self.inv = {
            rid: {"dimension": "legal" if rid.startswith("L") else "hate"}
            for rid in ("G1", "G2", "G3", "L1", "L2", "L3")
        }

    def test_full_guideline_block_is_verified(self):
        hits = [{"text": "One."}, {"text": "Two."}]
        messages = [
            {"role": "system", "text": "Fixed."},
            {
                "role": "user",
                "text": "Before\n\n## Annotation guidelines\n- One.\n- Two.\n\n## Text\nQuery",
            },
        ]
        result = x.partition_messages(messages, hits)
        self.assertEqual(
            result, {"system": "Fixed.", "before": "Before\n\n", "after": "\n\n## Text\nQuery"}
        )
        with self.assertRaises(ValueError):
            x.partition_messages(messages, [{"text": "Altered."}, {"text": "Two."}])

    def test_query_contains_heading_but_remains_preserved(self):
        messages = [
            {"role": "system", "text": "s"},
            {
                "role": "user",
                "text": "## Annotation guidelines\n- G\n\n## Text\n## Annotation guidelines\n- Query text",
            },
        ]
        self.assertIn("Query text", x.partition_messages(messages, [{"text": "G"}])["after"])

    def test_zero_legal_same_reordered_and_replaced(self):
        self.assertEqual(
            x.replacement(["G1", "G2"], ["G1", "G2"], "L1", self.inv)["general_change"],
            "same_ordered_guidelines",
        )
        self.assertEqual(
            x.replacement(["G1", "G2"], ["G2", "G1"], "L1", self.inv)["general_change"],
            "same_guidelines_reordered",
        )
        v = x.replacement(["G1", "G2"], ["G2", "G3"], "L1", self.inv)
        self.assertEqual(v["removed_general"], ["G1"])
        self.assertEqual(v["introduced_general"], ["G3"])
        self.assertEqual(v["added_legal_status"], "no_original_legal")

    def test_one_legal_can_keep_or_remove_the_old_general(self):
        a = x.replacement(["L1", "G1"], ["G1", "G3"], "L1", self.inv)
        b = x.replacement(["L1", "G1"], ["G2", "G3"], "L2", self.inv)
        self.assertEqual(a["retained_general"], ["G1"])
        self.assertEqual(a["added_legal_original_slot"], 1)
        self.assertEqual(b["removed_general"], ["G1"])
        self.assertEqual(b["added_legal_status"], "new_legal_despite_original_legal")

    def test_two_legal_to_one_does_not_restore_both(self):
        a = x.replacement(["L1", "L2"], ["G1", "G2"], "L2", self.inv)
        b = x.replacement(["L1", "L2"], ["G1", "G2"], "L3", self.inv)
        self.assertEqual(a["removed_original_legal_in_separate"], ["L1"])
        self.assertEqual(b["removed_original_legal_in_separate"], ["L1", "L2"])

    def test_url_t_overlap_and_literal_t_distinguished(self):
        trace = {
            "candidates": ["g"],
            "bm25_scores": [1.0],
            "bm25_order": ["g"],
            "dense_order": ["g"],
        }
        rule = {"id": "g", "text": "Don't misread negation."}
        url = x.lexical_observation("Beispiel https://t.co/abc", rule, trace)
        literal = x.lexical_observation("t Beispiel https://t.co/abc", rule, trace)
        self.assertTrue(url["only_t_from_url"])
        self.assertEqual(url["shared_tokens"], ["t"])
        self.assertFalse(literal["only_t_from_url"])

    def test_dense_scores_fixed_but_saved_lists_can_change(self):
        old = {
            "candidates": ["G1", "L1", "G2"],
            "dense_scores": [0.9, 0.8, 0.7],
            "bm25_scores": [1.0, 2.0, 0.5],
            "dense_order": ["G1", "L1"],
            "bm25_order": ["L1", "G1", "G2"],
        }
        new = {
            "candidates": ["G1", "G2"],
            "dense_scores": [0.9, 0.7],
            "bm25_scores": [0.4, 0.8],
            "dense_order": ["G1", "G2"],
            "bm25_order": ["G2", "G1"],
        }
        v = x.trace_changes({"mixed_diagnostic": old, "general": new})
        self.assertTrue(v["dense_scores_same_to_1e12"])
        self.assertTrue(v["filtered_old_dense_top10_differs"])
        self.assertTrue(v["bm25_shared_hit_order_differs"])

    def test_hate_groups_recover_count_differences(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([[[0, 1, 1, 1]] * 3, [[1, 1, 1, 0]] * 3, [[0, 0, 1, 0]] * 3])
        a = {"gold": y, "predictions": p}
        full = x.score_group(a, list(range(4)), False)
        groups = [x.score_group(a, v, False) for v in ([0, 1], [2, 3])]
        for i in range(3):
            for field in ("fp", "fn"):
                self.assertAlmostEqual(
                    full["contrasts"][i]["differences"][field],
                    sum((g["contrasts"][i]["differences"][field] for g in groups)),
                )

    def test_small_legal_group_keeps_fixed_three_labels(self):
        y = np.array([[1, 0, 0], [0, 0, 0]])
        p = np.tile(y, (3, 3, 1, 1))
        v = x.score_group({"gold": y, "predictions": p}, [0, 1], True)
        self.assertAlmostEqual(v["common"]["mixed_2"]["mean"]["macro_f1"], 1 / 3)
        self.assertEqual(v["common"]["mixed_2"]["mean"]["exact_set_accuracy"], 1)

    def test_identical_prompt_reference_does_not_equate_answers(self):

        def pred(v):
            return {"hate": v, "legal": [], "target_group": [], "hate_type": [], "severity": None}

        values = {("a", cond, r): pred(False) for cond in x.CONDITIONS for r in range(3)}
        values["a", "general_2", 1] = pred(True)
        v = x.decision_variation(values, ["a"], "synthetic")
        self.assertAlmostEqual(v["fields"]["hate"]["mean_cross_count"], 1 / 3)
        self.assertAlmostEqual(v["fields"]["hate"]["within_mean_count"]["general_2"], 2 / 3)

    def test_live_and_retrieval_methods_blocked(self):
        with self.assertRaises(RuntimeError):
            x.c.a.RequestsTransport.__call__(None, {})
        with self.assertRaises(RuntimeError):
            x.preparation.Guidelines.select(None, None, None, None)
        with self.assertRaises(RuntimeError):
            x.preparation.load_vectors({})
        with self.assertRaises(ValueError):
            x.save(x.reader.LIVE / "forbidden.json", {})


def main():
    with x.offline():
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(ContextChecks)
        )
        path = x.BASE / "tests" / f"tests_{time.time_ns()}" / "report.json"
        x.save(
            path,
            {
                "passed": result.wasSuccessful(),
                "tests": result.testsRun,
                "failures": [str(t) for t, _ in result.failures],
                "errors": [str(t) for t, _ in result.errors],
                "sources": x.c.bindings(x.own_sources()),
                "actual_Qwen_calls": 0,
                "retrieval_reruns": 0,
            },
        )
        print(json.dumps({"passed": result.wasSuccessful(), "report": x.c.ref(path)}))
        if not result.wasSuccessful():
            raise SystemExit(1)


if __name__ == "__main__":
    main()

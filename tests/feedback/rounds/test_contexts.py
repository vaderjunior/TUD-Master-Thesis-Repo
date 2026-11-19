import unittest
import numpy as np
from copy import deepcopy
from analysis.feedback.rounds import contexts_inputs as a
from tests.feedback.transfer.test_contexts import Boundaries


class SelectionTests(unittest.TestCase):
    def test_invalid_repeat_excludes_even_with_two_votes_agreeing(self):
        self.assertIsNone(a.majority([True, True, None]))
        self.assertTrue(a.majority([True, False, True]))
        with self.assertRaises(ValueError):
            a.majority([True, False])

    def test_budget_and_hash_roles(self):
        rows = [{"id": str(i), "gold": {"hate": i >= 8}} for i in range(16)]
        initial = {r["id"]: [not r["gold"]["hate"]] * 3 for r in rows}
        previous = {p: [] for p in a.POLICIES}
        selected = a.selection(rows, initial, initial, previous, 2026101201, 1)
        self.assertEqual(selected["per_class_budget"], 5)
        self.assertEqual(
            selected["additions"]["adaptive"], selected["additions"]["static"]
        )
        self.assertTrue(all(len(v) == 10 for v in selected["additions"].values()))
        current = deepcopy(initial)
        for r in rows:
            if r["gold"]["hate"] and r["id"] not in ["8", "9"]:
                current[r["id"]] = [True] * 3
        selected = a.selection(rows, initial, current, previous, 2026101201, 2)
        self.assertEqual(selected["per_class_budget"], 2)
        self.assertTrue(all(len(v) == 4 for v in selected["additions"].values()))

    def test_missing_source_common_to_all_policies(self):
        rows = [{"id": str(i), "gold": {"hate": i >= 6}} for i in range(12)]
        initial = {r["id"]: [not r["gold"]["hate"]] * 3 for r in rows}
        initial["0"] = [True, None, True]
        selected = a.selection(
            rows, initial, initial, {p: [] for p in a.POLICIES}, 1, 1
        )
        self.assertEqual(selected["invalid_ids"], ["0"])
        self.assertTrue(all("0" not in v for v in selected["additions"].values()))

    def test_retriever_label_blind_and_sorted_identity(self):
        records = [
            {
                "id": f"r{i:02}",
                "text": f"common word token{i}",
                "meta": {"gate": bool(i % 2)},
            }
            for i in range(10)
        ]
        vec = np.eye(10)
        one = a.ReplayBucket(records, vec).retrieve("word token4", vec[4])
        altered = deepcopy(records)
        for r in altered:
            r["meta"]["gate"] = not r["meta"]["gate"]
        two = a.ReplayBucket(altered[::-1], vec[::-1]).retrieve("word token4", vec[4])
        self.assertEqual(one, two)
        self.assertEqual(one["selected_id"], "r04")
        a.check_trace(one, two)

    def test_trace_mismatch_is_not_ignored(self):
        trace = {
            "dense": [{"id": "a", "score": 0.5}],
            "bm25": [],
            "rrf": [{"id": "a", "score": 1 / 61}],
            "selected_id": "a",
        }
        with self.assertRaises(ValueError):
            a.check_trace(trace, dict(trace, selected_id="b"))


if __name__ == "__main__":
    unittest.main()

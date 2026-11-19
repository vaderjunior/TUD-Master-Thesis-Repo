from copy import deepcopy
import unittest
import numpy as np
from analysis.feedback.balanced import prepare as preparation
from analysis.common.prompt import corrected_gold
from analysis.feedback.source.retrieval import FeedbackBucket


def record(i="feedback-a", gate=True):
    return {
        "id": i,
        "kind": "feedback",
        "lang": "en",
        "text": "A synthetic example.",
        "meta": {
            "gate": gate,
            "hate_types": None,
            "target_groups": None,
            "severity": None,
            "legal": None,
        },
    }


def context():
    return {
        "definitions": [{"id": "d1"}, {"id": "d2"}],
        "guidelines": [{"id": "g1"}, {"id": "g2"}],
        "example_groups": [
            [
                "Labelled examples",
                [{"id": f"e{i}", "meta": {"gate": True}} for i in range(5)],
            ]
        ],
    }


class ScienceTests(unittest.TestCase):

    def test_both_binary_labels_render_without_invented_fine_heads(self):
        for gate in [False, True]:
            result = preparation.append_binary(context(), record(gate=gate))
            hit = result["example_groups"][-1][1][0]
            self.assertEqual(corrected_gold(hit["meta"]), "hate=" + str(gate).lower())
            self.assertTrue(
                all((v is None for k, v in hit["meta"].items() if k != "gate"))
            )

    def test_base_and_source_are_not_mutated_or_aliased(self):
        base = context()
        source = record()
        before = deepcopy((base, source))
        result = preparation.append_binary(base, source)
        self.assertEqual((base, source), before)
        result["definitions"][0]["id"] = "changed"
        result["example_groups"][-1][1][0]["meta"]["gate"] = False
        self.assertEqual((base, source), before)

    def test_any_annotated_fine_head_is_rejected(self):
        for field in ["hate_types", "target_groups", "severity", "legal"]:
            source = record()
            source["meta"][field] = "low" if field == "severity" else []
            with self.assertRaises(AssertionError):
                preparation.append_binary(context(), source)

    def test_double_slot_and_duplicate_base_record_are_rejected(self):
        added = preparation.append_binary(context(), record())
        with self.assertRaises(AssertionError):
            preparation.append_binary(added, record("new"))
        with self.assertRaises(AssertionError):
            preparation.append_binary(context(), record("e0"))

    def test_wrong_budget_and_non_boolean_gate_are_rejected(self):
        base = context()
        base["example_groups"][0][1].pop()
        with self.assertRaises(AssertionError):
            preparation.append_binary(base, record())
        for gate in [None, 0, 1, "false"]:
            with self.assertRaises(AssertionError):
                preparation.append_binary(context(), record(gate=gate))

    def test_retrieval_is_label_independent_and_order_invariant(self):
        records = [record("c"), record("a", False), record("b")]
        vectors = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        query = np.array([0.8, 0.2])
        first, trace = FeedbackBucket(records, vectors).retrieve("synthetic", query)
        changed = deepcopy(records)
        for r in changed:
            r["meta"]["gate"] = not r["meta"]["gate"]
        second, other = FeedbackBucket(list(reversed(changed)), vectors[::-1]).retrieve(
            "synthetic", query
        )
        self.assertEqual(first["id"], second["id"])
        self.assertEqual(trace, other)


if __name__ == "__main__":
    unittest.main()

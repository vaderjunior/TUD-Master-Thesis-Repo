"""Tests for sampled banks annotations."""

import unittest
from analysis.knowledge.sampled_banks import annotations as c


class CoverageSemantics(unittest.TestCase):

    def test_missing_false_and_empty_differ(self):
        self.assertEqual(c.annotation_state({}, "gate"), "unannotated")
        self.assertEqual(c.annotation_state({"gate": None}, "gate"), "unannotated")
        self.assertEqual(c.annotation_state({"gate": False}, "gate"), "annotated_value")
        self.assertEqual(c.annotation_state({"legal": []}, "legal"), "annotated_empty")
        self.assertEqual(c.occurrences({"gate": False}, "gate"), ["false"])

    def test_never_infer_fine_labels_from_negative_gate(self):
        self.assertEqual(c.annotation_state({"gate": False}, "target_groups"), "unannotated")
        self.assertFalse(c.eligible({"gate": False}, "target_groups"))

    def test_legal_is_independent_of_hate(self):
        for gate in (False, None, True):
            self.assertTrue(c.eligible({"gate": gate, "legal": []}, "legal"))
            self.assertTrue(c.eligible({"gate": gate, "legal": ["insult_defamation"]}, "legal"))

    def test_fine_head_control_requires_positive_gate(self):
        for gate in (False, None):
            self.assertFalse(c.eligible({"gate": gate, "hate_types": ["irony"]}, "hate_types"))
        self.assertTrue(c.eligible({"gate": True, "hate_types": []}, "hate_types"))

    def test_slot_repetition_is_not_unique_record_count(self):
        a = {"id": "a", "source": "x", "meta": {"gate": True, "legal": []}}
        b = {"id": "b", "source": "x", "meta": {"gate": False}}
        summary = c.describe([a, a, b])
        self.assertEqual(summary["records_or_slots"], 3)
        self.assertEqual(summary["unique_records"], 2)
        self.assertEqual(
            summary["fields"]["legal"]["states"],
            {"unannotated": 1, "annotated_empty": 2, "annotated_value": 0},
        )

    def test_multilabel_occurrences_can_exceed_slots(self):
        a = {
            "id": "a",
            "source": "x",
            "meta": {"gate": True, "target_groups": ["race", "religion"]},
        }
        summary = c.describe([a])["fields"]["target_groups"]
        self.assertEqual(summary["annotated"], 1)
        self.assertEqual(sum(summary["label_occurrences"].values()), 2)

    def test_invalid_shapes_and_values_fail(self):
        for meta, field in [
            ({"gate": 0}, "gate"),
            ({"gate": "false"}, "gate"),
            ({"hate_types": ["insult_defamation"]}, "hate_types"),
            ({"legal": ["propaganda", "propaganda"]}, "legal"),
            ({"severity": []}, "severity"),
            ({"target_groups": "race"}, "target_groups"),
        ]:
            with self.subTest(meta=meta), self.assertRaises(ValueError):
                c.annotation_state(meta, field)

    def test_annotations_with_unknown_gate_not_lost(self):
        a = {"id": "a", "source": "x", "meta": {"gate": None, "legal": ["propaganda"]}}
        summary = c.describe([a])
        self.assertEqual(summary["gate"]["unknown"], 1)
        self.assertEqual(summary["fields"]["legal"]["annotated_by_gate"]["unknown"], 1)
        self.assertEqual(summary["fields"]["gate"]["states"]["unannotated"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)

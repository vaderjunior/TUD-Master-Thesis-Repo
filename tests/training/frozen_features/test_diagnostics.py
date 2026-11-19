import unittest
from training.frozen_features import diagnostics as m


class SavedDecisions(unittest.TestCase):
    def setUp(self):
        self.heads = {
            "hate": {"selected": {"threshold": 0.7}},
            "hate_type": {"selected": {"threshold": 0.3}, "labels": ["a", "b"]},
            "target_group": {"selected": {"threshold": 0.4}, "labels": ["x", "y"]},
            "severity": {"labels": ["high", "low", "medium"]},
        }
        self.p = dict(
            hate=[0.7],
            hate_type=[0.3, 0.5],
            target_group=[0.4, 0.5],
            severity=[0.2, 0.8, 0.8],
        )

    def test_gate_tie_clears_all_fine_fields(self):
        r = m.decode(self.p, self.heads, "selected")
        self.assertFalse(r["hate"])
        self.assertEqual(r["hate_type"], [])
        self.assertIsNone(r["severity"])

    def test_fine_inclusive_and_severity_first_tie(self):
        r = m.decode(self.p, self.heads, "half_gate")
        self.assertEqual(r["hate_type"], ["a", "b"])
        self.assertEqual(r["target_group"], ["x", "y"])
        self.assertEqual(r["severity"], "low")

    def test_default_fine_changes_only_cutoff(self):
        p = {**self.p, "hate": [0.8]}
        r = m.decode(p, self.heads, "half_fine")
        self.assertTrue(r["hate"])
        self.assertEqual(r["hate_type"], ["b"])
        self.assertEqual(r["target_group"], ["y"])

    def test_oracle_does_not_change_reported_gate(self):
        r = m.decode(self.p, self.heads, "ungated_selected")
        self.assertFalse(r["hate"])
        self.assertEqual(r["hate_type"], ["a", "b"])

    def test_half_gate_tie_remains_false(self):
        p = {**self.p, "hate": [0.5]}
        self.assertFalse(m.decode(p, self.heads, "half_both")["hate"])

    def test_unknown_mode_rejected(self):
        with self.assertRaises(RuntimeError):
            m.decode(self.p, self.heads, "best_dev")


if __name__ == "__main__":
    unittest.main()

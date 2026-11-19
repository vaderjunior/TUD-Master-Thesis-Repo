"""Tests for banks primary."""

import unittest
from analysis.knowledge.banks import primary as r


class PrimaryTests(unittest.TestCase):

    def test_duplicate_join_rejected(self):
        with self.assertRaises(Exception):
            r.keyed([{"id": "a"}, {"id": "a"}])

    def test_missing_output_rejected(self):
        with self.assertRaises(Exception):
            r.independent_counts(
                [{"id": "a", "gate": True}],
                {},
                {
                    "all_labels": ["not_hate", "hate"],
                    "macro_labels": ["not_hate", "hate"],
                    "dimension": "hate",
                },
            )

    def test_independent_gate_counts_and_null_bounds(self):
        rows = [{"id": str(i), "gate": g} for i, g in enumerate([True, False, True])]
        outputs = {
            "0": {"result": {"hate": False}, "n_valid": 3},
            "1": {"result": {"hate": False}, "n_valid": 1},
            "2": {"result": None, "n_valid": 0},
        }
        ep = {
            "dimension": "hate",
            "all_labels": ["not_hate", "hate"],
            "macro_labels": ["not_hate", "hate"],
            "scorable_ids": ["0", "1", "2"],
        }
        c, score = r.checked_score(rows, outputs, ep)
        self.assertEqual(score["per_label_counts"], [[1, 1, 0], [0, 0, 1]])
        self.assertEqual(score["macro_f1_valid_only"], 1 / 3)
        self.assertEqual(score["invalid_inclusive_bounds"], [0.25, 2 / 3])

    def test_legal_independent_of_gate_and_empty_is_observed(self):
        rows = [
            {"id": "a", "gate": False, "legal": ["propaganda"]},
            {"id": "b", "gate": None, "legal": []},
        ]
        out = {
            "a": {"result": {"hate": False, "legal": ["propaganda"]}, "n_valid": 1},
            "b": {"result": {"hate": True, "legal": []}, "n_valid": 3},
        }
        ep = {
            "dimension": "legal",
            "all_labels": ["propaganda"],
            "macro_labels": ["propaganda"],
            "scorable_ids": ["a", "b"],
        }
        c, score = r.checked_score(rows, out, ep)
        self.assertEqual(score["macro_f1_valid_only"], 1)
        self.assertEqual(score["n_scorable"], 2)

    def test_example_set_dispersion_not_pooled_runs(self):
        means = [
            r.summarize_three(values, "decoder_repeat")["mean"]
            for values in ([0, 0, 1], [1, 1, 1], [0, 0, 0])
        ]
        summary = r.summarize_three(means, "example_set_mean")
        self.assertAlmostEqual(summary["mean"], 4 / 9)
        self.assertNotAlmostEqual(
            summary["sample_sd"], float(r.np.std([0, 0, 1, 1, 1, 1, 0, 0, 0], ddof=1))
        )


if __name__ == "__main__":
    unittest.main()

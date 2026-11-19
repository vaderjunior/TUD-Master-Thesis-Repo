"""Tests for policy labels."""

import unittest
import hashlib
import numpy as np
from analysis.knowledge.policy import labels as p


class Part4Tests(unittest.TestCase):

    def fixture(self):
        gold = np.array([[1, 0], [0, 1], [0, 1]], bool)
        predicted = np.tile(gold, (3, 3, 1, 1))
        predicted[0, :, 0] = [0, 1]
        predicted[2, :, 1] = [1, 0]
        correct = (predicted == gold).all(axis=-1)
        return dict(
            gold=gold,
            predicted=predicted,
            correct=correct,
            gates=predicted[:, :, :, 1],
            common=np.array([1, 1, 0], bool),
        )

    def test_subset_respects_common(self):
        out = p.subset(
            {"dimension": "hate", "all_labels": ["not_hate", "hate"]}, self.fixture(), [1, 1, 1]
        )
        self.assertEqual(out["n"], 2)
        self.assertEqual(out["exact"], [0.5, 1.0, 0.5])
        self.assertEqual(out["false_positive_rate"], [1.0, 0.0, 0.0])
        self.assertEqual(out["false_negative_rate"], [0.0, 0.0, 1.0])
        self.assertEqual(out["transitions"][0]["stable_fixes"], 1)
        self.assertEqual(out["transitions"][0]["stable_regressions"], 1)

    def test_single_class_denominator(self):
        out = p.subset(
            {"dimension": "hate", "all_labels": ["not_hate", "hate"]}, self.fixture(), [1, 0, 0]
        )
        self.assertEqual(out["false_negative_rate"], [None, None, None])
        self.assertEqual(out["labels_with_no_gold"], ["hate"])

    def test_empty_group(self):
        self.assertEqual(
            p.subset(
                {"dimension": "hate", "all_labels": ["not_hate", "hate"]}, self.fixture(), [0, 0, 1]
            ),
            {"n": 0},
        )

    def test_gate_aware_exact_separate_from_jaccard(self):
        arr = self.fixture()
        arr["gold"][:] = False
        arr["predicted"][:] = False
        arr["correct"][:] = False
        arr["gates"][:] = False
        out = p.subset({"dimension": "hate_type", "all_labels": ["a", "b"]}, arr, [1, 1, 1])
        self.assertEqual(out["exact"], [0.0, 0.0, 0.0])
        self.assertEqual(out["jaccard"], [1.0, 1.0, 1.0])

    def test_definition_coverage(self):
        meta = {"dimension": "target_group"}
        roster = {"x": {"target_groups": ["race", "religion"]}}
        for ds, expected in [
            ([], "none"),
            (["def-target_group-race"], "some"),
            (["def-target_group-race", "def-target_group-religion"], "all"),
        ]:
            self.assertEqual(
                p.definition_coverage(meta, roster, {"x": {"definitions_A": ds}}, "x"), expected
            )
        roster["x"]["target_groups"] = []
        self.assertEqual(
            p.definition_coverage(meta, roster, {"x": {"definitions_A": []}}, "x"), "empty_gold"
        )

    def test_severity_and_hate_definition(self):
        self.assertEqual(
            p.definition_coverage(
                {"dimension": "severity"},
                {"x": {"severity": "high"}},
                {"x": {"definitions_A": ["def-severity-high"]}},
                "x",
            ),
            "all",
        )
        self.assertEqual(
            p.definition_coverage(
                {"dimension": "hate"}, {"x": {}}, {"x": {"definitions_A": ["def-hate"]}}, "x"
            ),
            "all",
        )

    def test_case_direction_and_missing(self):
        arr = self.fixture()
        keys = ["x", "y", "z"]
        positions = {k: i for i, k in enumerate(keys)}
        self.assertEqual(p.choose_case(keys, arr, positions, 2, 0, "fix")["key"], "x")
        self.assertEqual(p.choose_case(keys, arr, positions, 2, 0, "regression")["key"], "y")
        self.assertIsNone(p.choose_case(["z"], arr, positions, 2, 0, "fix")["key"])

    def test_case_tie_is_text_independent(self):
        arr = self.fixture()
        arr["correct"][0, :, :2] = False
        arr["correct"][2, :, :2] = True
        expected = min(
            ["x", "y"], key=lambda k: hashlib.sha256((p.CASE_SEED + "|" + k).encode()).hexdigest()
        )
        for keys in (["x", "y"], ["y", "x"]):
            out = p.choose_case(keys, arr, {"x": 0, "y": 1}, 2, 0, "fix")
            self.assertEqual(out["key"], expected)
            self.assertEqual(out["stable_candidates"], 2)

    def test_fallback_and_zero_net_change(self):
        arr = self.fixture()
        arr["correct"][0, :, 0] = [0, 1, 0]
        arr["correct"][2, :, 0] = [1, 0, 1]
        out = p.choose_case(["x"], arr, {"x": 0}, 2, 0, "fix")
        self.assertEqual(out["net_strength"], 1)
        self.assertEqual(out["selection"], "largest_net_correct_repeat_change")
        arr["correct"][2, :, 0] = [1, 0, 0]
        self.assertIsNone(p.choose_case(["x"], arr, {"x": 0}, 2, 0, "fix")["key"])


if __name__ == "__main__":
    with p.c.offline():
        unittest.main()

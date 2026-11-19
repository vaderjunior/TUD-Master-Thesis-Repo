from collections import Counter
from fractions import Fraction
import itertools
import unittest
from analysis.feedback.reserved import transitions_metrics as c


def prediction(hate, types=(), votes=3):
    return {"result": {"hate": hate, "hate_type": list(types)}, "n_valid": votes}


class Transitions(unittest.TestCase):
    def test_all_binary_truth_tables(self):
        for gold, t, b in itertools.product((False, True), repeat=3):
            v = c.compare(
                {"hate": gold, "hate_type": None}, prediction(t), prediction(b)
            )
            e = (int(t != gold), int(b != gold))
            self.assertEqual(
                v["binary"],
                {
                    (0, 0): "unchanged_correct",
                    (0, 1): "fix",
                    (1, 0): "regression",
                    (1, 1): "unchanged_error",
                }[e],
            )
            self.assertNotIn("type_exact", v)

    def test_all_type_bitsets(self):
        labels = list("abcdefg")
        goldcode = 5
        gold = [labels[i] for i in range(7) if goldcode >> i & 1]
        for tc, bc in itertools.product(range(128), repeat=2):
            t = [labels[i] for i in range(7) if tc >> i & 1]
            b = [labels[i] for i in range(7) if bc >> i & 1]
            v = c.compare(
                {"hate": True, "hate_type": gold},
                prediction(True, t),
                prediction(True, b),
            )
            self.assertEqual(
                v["correct_labels_added"], (tc & ~bc & goldcode).bit_count()
            )
            self.assertEqual(
                v["extra_labels_removed"], (bc & ~tc & ~goldcode & 127).bit_count()
            )
            self.assertEqual(v["types_changed"], tc != bc)
            self.assertEqual(
                v["type_exact"],
                {
                    (True, True): "unchanged_correct",
                    (True, False): "fix",
                    (False, True): "regression",
                    (False, False): "unchanged_error",
                }[tc == goldcode, bc == goldcode],
            )

    def test_null_is_neither_error_nor_unknown_gold_imputation(self):
        null = {"result": None, "n_valid": 0}
        ct = Counter()
        g = {"hate": True, "hate_type": ["a"]}
        for t, b in [
            (null, null),
            (null, prediction(True, ["a"])),
            (prediction(False), null),
        ]:
            c.update(ct, c.compare(g, t, b))
        out = c.finish(ct)
        self.assertEqual(out["eligible_pairs"], 3)
        self.assertIsNone(out["binary_error_difference"])
        self.assertNotIn("fix", ct)
        v = c.compare(g, prediction(True, ["a"], 2), prediction(True, ["a"]))
        self.assertEqual(v["type_exact"], "unchanged_correct")
        self.assertFalse(v["strict_pair_valid"])

    def test_reference_reuse_matches_equal_arm_weighting(self):
        ts = ["a", "b", "c"]
        cs = ["ref"]
        gold = {"hate": True, "hate_type": ["a"]}
        data = {"a": [0, 0, 0], "b": [0, 1, 1], "c": [1, 1, 1], "ref": [0, 1, 0]}
        ct = Counter()
        for t, b in c.condition_pairs(ts, cs):
            for rep in range(3):
                c.update(
                    ct,
                    c.compare(
                        gold,
                        prediction(bool(data[t][rep])),
                        prediction(bool(data[b][rep])),
                    ),
                )
        independent = Fraction(sum(1 - x for t in ts for x in data[t]), 9) - Fraction(
            sum(1 - x for x in data["ref"]), 3
        )
        self.assertEqual(c.finish(ct)["binary_error_difference"], float(independent))
        self.assertEqual(ct["valid"], 9)

    def test_offsetting_changes_not_unchanged_predictions(self):
        ct = Counter()
        g = {"hate": False, "hate_type": None}
        for t, b in [(False, True), (True, False), (False, False)]:
            c.update(ct, c.compare(g, prediction(t), prediction(b)))
        self.assertEqual(c.direction(ct), "same_average_errors")
        self.assertEqual(ct["fix"] + ct["regression"], 2)

    def test_type_categories_and_gate_misses(self):
        g = {"hate": True, "hate_type": ["a", "b"]}
        for pred, expected in [
            (["a", "b"], "exact"),
            (["a", "b", "c"], "only_extra"),
            (["a"], "only_missing"),
            (["c"], "both_extra_and_missing"),
        ]:
            self.assertEqual(c.category(g["hate_type"], pred), expected)
        v = c.compare(g, prediction(False), prediction(True, ["a", "b"]))
        self.assertEqual(v["binary"], "regression")
        self.assertEqual(v["type_exact"], "regression")

    def test_case_selection_independent_of_input_order(self):
        keys = ["q1", "q2", "q3"]
        first = c.select_key("slot", keys, set())
        self.assertEqual(first, c.select_key("slot", reversed(keys), set()))
        self.assertNotEqual(first, c.select_key("slot", keys, {first}))
        self.assertIsNone(c.select_key("slot", keys, set(keys)))


if __name__ == "__main__":
    unittest.main()

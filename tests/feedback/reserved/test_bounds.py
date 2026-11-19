from copy import deepcopy
from itertools import product
import unittest
from analysis.feedback.reserved import bounds as b
from analysis.feedback.reserved import support as s


class BoundsTests(unittest.TestCase):
    def test_binary_bounds_match_exhaustive_completions(self):
        r = [
            {"key": k, "gold": {"hate": truth, "hate_type": None}}
            for k, truth in [("a", False), ("b", True), ("c", True)]
        ]
        p = {
            (row["key"], arm, rep): {
                "result": {"hate": row["gold"]["hate"], "hate_type": []},
                "n_valid": 3,
            }
            for row in r
            for arm in ("t", "c")
            for rep in range(3)
        }
        missing = [("a", "t", 0), ("b", "t", 0), ("c", "c", 1)]
        for key in missing:
            p[key] = {"result": None, "n_valid": 0}
        bounds = b.metric_bounds(r, p, [["t"], ["c"]])
        seen = {m: [] for m in bounds["metrics"]}
        for bits in product((False, True), repeat=3):
            complete = deepcopy(p)
            for key, bit in zip(missing, bits):
                complete[key] = {"result": {"hate": bit, "hate_type": []}, "n_valid": 3}
            for m in seen:
                arms = [
                    sum(
                        s.metric_rows(r, complete, arm, rep)["metrics"][m]
                        for rep in range(3)
                    )
                    / 3
                    for arm in ("t", "c")
                ]
                seen[m].append(arms[0] - arms[1])
        for m in seen:
            self.assertAlmostEqual(
                min(seen[m]), bounds["metrics"][m]["opposing_difference"][0]
            )
            self.assertAlmostEqual(
                max(seen[m]), bounds["metrics"][m]["opposing_difference"][1]
            )

    def test_fine_bounds_fixed_support_and_strict_votes(self):
        r = [{"key": "a", "gold": {"hate": True, "hate_type": ["explicit"]}}]
        p = {
            ("a", arm, rep): {
                "result": {"hate": True, "hate_type": ["explicit"]},
                "n_valid": 3,
            }
            for arm in ("t", "c")
            for rep in range(3)
        }
        p["a", "t", 0]["n_valid"] = 2
        ordinary = b.metric_bounds(r, p, [["t"], ["c"]], fine=True)
        strict = b.metric_bounds(r, p, [["t"], ["c"]], fine=True, strict=True)
        self.assertEqual(
            ordinary["metrics"]["type_exact_set"]["opposing_difference"], [0, 0]
        )
        self.assertAlmostEqual(
            strict["metrics"]["type_exact_set"]["opposing_difference"][0], -1 / 3
        )
        self.assertAlmostEqual(
            strict["metrics"]["type_macro_f1_fixed7"]["opposing_difference"][0], -1 / 21
        )
        self.assertEqual(strict["missing_real_outputs_by_arm"], [1, 0])


if __name__ == "__main__":
    unittest.main()

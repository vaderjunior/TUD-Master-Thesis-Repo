from copy import deepcopy
import unittest
from analysis.feedback.reserved import effects as a
from tests.feedback.reserved.test_rates import fixture


class Part2Tests(unittest.TestCase):
    def test_exact_counts_and_bounds_detect_corrupted_rate(self):
        roster, p, spec = fixture()
        p["a", "t1", 0]["result"]["hate"] = True
        p["b", "reference5", 2]["result"]["hate"] = True
        p["c", "t2", 1] = {"result": None, "n_valid": 0}
        cohort = a.rates.cohort(roster, p, spec)
        result, _ = a.rates.summarize(cohort, spec)
        value = a.exact_check(roster, p, spec, result)
        self.assertEqual(value["real_error_counts"], [1, 1])
        self.assertEqual(value["real_output_counts"], [18, 6])
        self.assertEqual(value["missing_real_outputs"], [1, 0])
        result["difference"] += 0.01
        with self.assertRaises(ValueError):
            a.exact_check(roster, p, spec, result)

    def test_strict_exclusion_and_representative_masks(self):
        roster, p, spec = fixture()
        p["a", "t1", 0]["n_valid"] = 2
        for strict in (False, True):
            for reps in (False, True):
                for excluded in ([], ["c"]):
                    cohort = a.rates.cohort(
                        roster,
                        p,
                        spec,
                        strict=strict,
                        representatives=reps,
                        excluded=excluded,
                    )
                    result, _ = a.rates.summarize(cohort, spec)
                    a.exact_check(roster, p, spec, result, strict, reps, excluded)

    def test_reference_count_is_not_tripled_and_transition_net_matches(self):
        roster, p, spec = fixture()
        for row in roster:
            row["source"] = "synthetic"
        for rep in range(3):
            p["a", "reference5", rep]["result"]["hate"] = True
        p["b", "t2", 1]["result"]["hate"] = True
        result, _ = a.rates.summarize(a.rates.cohort(roster, p, spec), spec)
        value = a.descriptive(roster, p, spec, result)
        ref = [r for r in value["runs"] if r["condition"] == "reference5"]
        self.assertEqual(len(ref), 3)
        self.assertEqual(sum(r["n"] for r in ref), 9)
        net = sum(
            r["matched_repeat_regressed"] - r["matched_repeat_fixed"]
            for r in value["pools"]
        )
        self.assertAlmostEqual(net / (3 * 9), result["difference"])
        self.assertEqual(sum(value["per_text_average_direction"].values()), 3)


if __name__ == "__main__":
    unittest.main()

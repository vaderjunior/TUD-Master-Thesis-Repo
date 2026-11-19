from copy import deepcopy
from fractions import Fraction
from itertools import product
import unittest
import numpy as np
from analysis.feedback.reserved import rates as s


def fixture():
    roster = [
        {"key": k, "gold": {"hate": False}, "cluster": cluster}
        for k, cluster in [("a", "c1"), ("b", "c1"), ("c", "c2")]
    ]
    spec = {
        "id": "synthetic",
        "endpoint": "FPR",
        "treatment_conditions": ["t1", "t2", "t3"],
        "control_conditions": ["reference5"],
        "seed": 42,
        "draws": 80,
        "confidence": 0.975,
    }
    p = {
        (k, arm, repeat): {"result": {"hate": False}, "n_valid": 3}
        for k in ("a", "b", "c")
        for arm in ("t1", "t2", "t3", "reference5")
        for repeat in range(3)
    }
    return roster, p, spec


class RatesTests(unittest.TestCase):
    def test_declared_families_and_control_major_seed_assignment(self):
        seeds = [2026100441, 2026100442, 2026100443]
        groups = ["reserved5", "natural5", "extended6", "masked5"]
        science = {
            "bootstrap_draws": 10000,
            "decoder_repeats": 3,
            "pool_seeds": seeds,
            "conditions": ["reference5"]
            + [g + "__error_balanced_s" + str(seed) for g in groups for seed in seeds],
            "primary": {
                "confidence": 0.975,
                "control": "reference5",
                "treatment": "reserved5__error_balanced",
                "endpoints": ["FPR", "FNR"],
                "bootstrap_seeds": [2026101201, 2026101202],
            },
            "mechanism": {
                "confidence": 1 - 0.05 / 6,
                "treatment": "reserved5__error_balanced",
                "controls": [g + "__error_balanced" for g in groups[1:]],
                "endpoints": ["FPR", "FNR"],
                "bootstrap_seeds": list(range(2026101210, 2026101216)),
            },
        }
        specs = s.contrasts(science)
        self.assertEqual(len(specs), 8)
        self.assertEqual(
            [x["seed"] for x in specs],
            [2026101201, 2026101202] + list(range(2026101210, 2026101216)),
        )
        self.assertEqual([x["endpoint"] for x in specs], ["FPR", "FNR"] * 4)
        self.assertTrue(
            all(x["control_conditions"] == ["reference5"] for x in specs[:2])
        )
        self.assertTrue(all(len(x["control_conditions"]) == 3 for x in specs[2:]))
        science["mechanism"]["confidence"] = 0.95
        with self.assertRaises(ValueError):
            s.contrasts(science)

    def test_reference_has_three_actual_outputs_and_equal_pool_weight(self):
        r, p, spec = fixture()
        for rep in range(3):
            p["a", "t1", rep]["result"]["hate"] = True
        c = s.cohort(r, p, spec)
        self.assertEqual(c["outputs_per_input"], [9, 3])
        self.assertEqual(c["errors"]["a"], [1 / 3, 0])
        result, _ = s.summarize(c, spec)
        self.assertAlmostEqual(result["difference"], 1 / 9)

    def test_common_mask_requires_every_contributing_pool_and_repeat(self):
        r, p, spec = fixture()
        p["a", "t3", 2] = {"result": None, "n_valid": 0}
        c = s.cohort(r, p, spec)
        self.assertEqual(c["common_ids"], ["b", "c"])
        self.assertEqual(c["bounds"]["a"][0], [0, 1 / 9])
        result, _ = s.summarize(c, spec)
        self.assertEqual(result["missing_difference_bounds"], [0, 1 / 27])

    def test_strict_votes_are_a_separate_mask_and_bounds(self):
        r, p, spec = fixture()
        p["a", "reference5", 1]["n_valid"] = 2
        self.assertEqual(len(s.cohort(r, p, spec)["common_ids"]), 3)
        strict = s.cohort(r, p, spec, strict=True)
        self.assertEqual(strict["common_ids"], ["b", "c"])
        self.assertEqual(strict["bounds"]["a"][1], [0, 1 / 3])

    def test_fnr_error_direction_and_missing_bounds(self):
        r, p, spec = fixture()
        spec["endpoint"] = "FNR"
        for row in r:
            row["gold"]["hate"] = True
        for key in p:
            if key[1] != "reference5":
                p[key]["result"]["hate"] = True
        result, draws = s.summarize(s.cohort(r, p, spec), spec)
        self.assertEqual(result["difference"], -1)
        self.assertEqual(result["missing_difference_bounds"], [-1, -1])
        np.testing.assert_array_equal(draws, -np.ones(spec["draws"]))

    def test_cluster_sampling_matches_expanded_items_with_unequal_sizes(self):
        r, p, spec = fixture()
        for arm in spec["treatment_conditions"]:
            for rep in range(3):
                p["c", arm, rep]["result"]["hate"] = True
        c = s.cohort(r, p, spec)
        labels, totals, sizes = s.cluster_totals(c)
        self.assertEqual(sizes, [2, 1])
        for indices in product(range(2), repeat=2):
            expanded = [
                k
                for i in indices
                for k in c["common_ids"]
                if c["clusters"][k] == labels[i]
            ]
            direct = sum(Fraction(int(k == "c"), 1) for k in expanded) / len(expanded)
            self.assertAlmostEqual(
                s.resampled_difference(totals, sizes, list(indices)), float(direct)
            )
        self.assertEqual(s.resampled_difference(totals, sizes, [0, 1]), 1 / 3)

    def test_pcg64_bootstrap_independently_matches_direct_expansion(self):
        r, p, spec = fixture()
        spec["draws"] = 511
        p["a", "t1", 0]["result"]["hate"] = True
        p["c", "reference5", 1]["result"]["hate"] = True
        c = s.cohort(r, p, spec)
        result, draws = s.summarize(c, spec)
        rng = np.random.Generator(np.random.PCG64(spec["seed"]))
        manual = []
        for _ in range(spec["draws"]):
            chosen = rng.integers(0, 2, size=2)
            keys = [
                k
                for i in chosen
                for k in c["common_ids"]
                if c["clusters"][k] == ("c1", "c2")[i]
            ]
            differences = []
            for k in keys:
                treatment = (
                    sum(
                        Fraction(p[k, arm, j]["result"]["hate"])
                        for arm in spec["treatment_conditions"]
                        for j in range(3)
                    )
                    / 9
                )
                reference = (
                    sum(
                        Fraction(p[k, "reference5", j]["result"]["hate"])
                        for j in range(3)
                    )
                    / 3
                )
                differences.append(treatment - reference)
            manual.append(float(sum(differences) / len(keys)))
        np.testing.assert_allclose(draws, manual, rtol=0, atol=1e-15)
        ordered = sorted(manual)

        def quantile(q):
            x = q * (len(ordered) - 1)
            lo = int(x)
            return ordered[lo] + (
                ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]
            ) * (x - lo)

        np.testing.assert_allclose(
            result["family_interval"],
            [quantile(0.0125), quantile(0.9875)],
            rtol=0,
            atol=1e-15,
        )

    def test_representative_selected_before_validity_no_replacement(self):
        r, p, spec = fixture()
        first = s.cohort(r, p, spec, representatives=True)
        chosen = next(k for k in first["common_ids"] if k != "c")
        p[chosen, "reference5", 0] = {"result": None, "n_valid": 0}
        after = s.cohort(r, p, spec, representatives=True)
        self.assertEqual(after["common_ids"], ["c"])
        self.assertEqual(after["original_ids"], first["original_ids"])

    def test_shared_exclusion_applied_to_both_arms(self):
        r, p, spec = fixture()
        c = s.cohort(r, p, spec, excluded=["a"])
        self.assertEqual(c["original_ids"], ["b", "c"])
        self.assertEqual(set(c["errors"]), {"b", "c"})
        with self.assertRaises(ValueError):
            s.cohort(r, p, spec, excluded=["unknown"])

    def test_no_common_valid_is_not_zero_error(self):
        r, p, spec = fixture()
        for row in r:
            p[row["key"], "reference5", 0] = {"result": None, "n_valid": 0}
        out, draws = s.summarize(s.cohort(r, p, spec), spec)
        self.assertEqual(out["common_n"], 0)
        self.assertNotIn("difference", out)
        self.assertEqual(len(draws), 0)

    def test_reject_corrupt_counts_gold_conditions_and_missing_outputs(self):
        r, p, spec = fixture()
        for record in (
            {"result": None, "n_valid": 1},
            {"result": {"hate": 1}, "n_valid": 3},
            {"result": {"hate": False}, "n_valid": True},
        ):
            with self.assertRaises(ValueError):
                s.valid_prediction(record, False)
        with self.assertRaises(ValueError):
            s.cohort(r + [r[0]], p, spec)
        wrong = deepcopy(spec)
        wrong["control_conditions"] *= 3
        with self.assertRaises(ValueError):
            s.cohort(r, p, wrong)
        del p["b", "t1", 0]
        with self.assertRaises(KeyError):
            s.cohort(r, p, spec)


if __name__ == "__main__":
    unittest.main()

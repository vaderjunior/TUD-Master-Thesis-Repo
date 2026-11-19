from collections import Counter
from copy import deepcopy
import unittest
from analysis.feedback.reserved import pools as a
from analysis.feedback.reserved import pools_context as context


class Part4(unittest.TestCase):
    def science(self):
        seeds = [2026100441, 2026100442, 2026100443]
        return {
            "pool_seeds": seeds,
            "bootstrap_draws": 10000,
            "mandatory_excluded_keys": ["k0"],
            "conditions": ["reference5", "reference4"]
            + [
                m + "__" + p + "_s" + str(s)
                for m in a.MODES
                for p in a.POLICIES
                for s in seeds
            ],
        }

    def fixture(self):
        science = self.science()
        roster = [
            {
                "key": "k" + str(i),
                "cluster": "g" + str(i // 2),
                "source": "source",
                "gold": {"hate": i > 2},
            }
            for i in range(6)
        ]
        pred = {}
        for i, row in enumerate(roster):
            for j, c in enumerate(science["conditions"]):
                for rep in range(3):
                    pred[row["key"], c, rep] = {
                        "result": {"hate": (i * 11 + j * 5 + rep * 2) % 7 < 2},
                        "n_valid": 3,
                    }
        return roster, pred, science

    def test_complete_direct_coverage_and_orientation(self):
        science = self.science()
        specs = a.direct_specs(science)
        self.assertEqual(
            Counter(s["dimension"] for s in specs), {"selection": 48, "composition": 24}
        )
        self.assertEqual(Counter(s["mode"] for s in specs), {m: 18 for m in a.MODES})
        self.assertEqual(len({s["seed"] for s in specs}), 72)
        for s in specs:
            self.assertEqual(s["confidence"], 0.95)
            self.assertTrue(
                set(s["treatment_conditions"] + s["control_conditions"])
                <= set(science["conditions"])
            )
            self.assertEqual(
                [c.rsplit("_s", 1)[1] for c in s["treatment_conditions"]],
                [c.rsplit("_s", 1)[1] for c in s["control_conditions"]],
            )
            if s["dimension"] == "composition":
                self.assertTrue(
                    s["treatment_policy"].endswith("_balanced")
                    and s["control_policy"].endswith("_positive")
                )

    def test_direct_masks_strict_exclusion_and_integer_draws(self):
        roster, pred, science = self.fixture()
        spec = a.direct_specs(science)[0]
        spec["draws"] = 1000
        pred["k1", spec["treatment_conditions"][0], 0] = {"result": None, "n_valid": 0}
        pred["k2", spec["control_conditions"][0], 0]["n_valid"] = 2
        for strict in (False, True):
            c = a.rates.cohort(roster, pred, spec, excluded=["k0"], strict=strict)
            point = a.point_result(c, spec)
            a.prior.prior.exact_check(roster, pred, spec, point, strict, False, ["k0"])
            self.assertEqual(point["common_n"], 0 if strict else 1)
            r, draws = a.rates.summarize(c, spec)
            check = a.prior.integer_draw_check(roster, pred, spec, c, draws)
            self.assertEqual(check["draws_checked"], 0 if strict else 1000)

    def test_pool_details_equal_weight_and_real_reference_outputs(self):
        roster, pred, science = self.fixture()
        spec = a.support.supporting_specs(science)[2]
        c = a.rates.cohort(roster, pred, spec)
        r = a.point_result(c, spec)
        d = a.comparison_details(roster, pred, spec, r)
        check = a.prior.prior.exact_check(roster, pred, spec, r)
        self.assertEqual(check["real_output_counts"], [27, 9])
        self.assertEqual(len(d["pools"]), 3)
        self.assertAlmostEqual(
            sum(p["difference"] for p in d["pools"]) / 3, r["difference"]
        )

    def test_common_mask_table_never_replicates_reference(self):
        roster, pred, science = self.fixture()
        own, tables, dependence = a.own_valid_and_common(roster, pred, science)
        table = tables["full/ordinary_valid/FPR"]
        self.assertEqual(table["common_n"], 3)
        self.assertEqual(table["groups"]["reference5"]["pools"][0]["actual_outputs"], 9)
        self.assertEqual(
            sum(
                p["actual_outputs"]
                for p in table["groups"]["reserved5__error_balanced"]["pools"]
            ),
            27,
        )
        self.assertEqual(len(own), 6)
        self.assertEqual(len(own["full/ordinary_valid"]), 222)
        self.assertEqual(len(dependence), 144)

    def pool_fixture(self):
        ids = ["s" + str(i) for i in range(40)]
        records = {
            sid: {
                "meta": {
                    "gate": i < 20,
                    "hate_types": None,
                    "target_groups": None,
                    "severity": None,
                    "legal": None,
                }
            }
            for i, sid in enumerate(ids)
        }
        statuses = {
            sid: {"status": "error" if i < 10 else "correct", "hate": i < 20}
            for i, sid in enumerate(ids)
        }
        pool = {
            "ids": ids,
            "budget": 40,
            "condition": "random_balanced_s1",
            "polarity": "balanced",
            "selection": "random",
            "seed": 1,
            "actual_status_counts": {"error": 10, "correct": 30},
        }
        return pool, records, statuses

    def test_random_mixture_is_verified_not_assumed_correct(self):
        pool, records, statuses = self.pool_fixture()
        result = context.check_pool(pool, records, statuses)
        self.assertEqual(result["previous_status_counts"], {"error": 10, "correct": 30})
        bad = deepcopy(pool)
        bad["selection"] = "correct"
        with self.assertRaises(ValueError):
            context.check_pool(bad, records, statuses)

    def test_source_gold_mismatch_and_duplicate_rejected(self):
        pool, records, statuses = self.pool_fixture()
        bad = deepcopy(records)
        bad["s0"]["meta"]["gate"] = False
        with self.assertRaises(ValueError):
            context.check_pool(pool, bad, statuses)
        badpool = deepcopy(pool)
        badpool["ids"][-1] = badpool["ids"][0]
        with self.assertRaises(ValueError):
            context.check_pool(badpool, records, statuses)


if __name__ == "__main__":
    unittest.main()

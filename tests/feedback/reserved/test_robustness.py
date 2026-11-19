from copy import deepcopy
from fractions import Fraction
import unittest
import numpy as np
from analysis.feedback.reserved import robustness_metrics as c
from analysis.feedback.reserved import rates as rates
from analysis.feedback.reserved import controls as p3
from tests.feedback.reserved.test_support import fixture


class RobustnessTests(unittest.TestCase):
    def test_zero_and_positive_tiny_endpoints_are_distinct(self):
        self.assertEqual(c.interval_state(["-1/30", "0"]), "includes_zero")
        self.assertEqual(c.interval_state(["-1/30", "1/30000"]), "includes_zero")
        self.assertEqual(c.interval_state(["-1/30", "-1/30000"]), "negative")
        self.assertEqual(c.sign("0"), "zero")
        self.assertEqual(c.interval_state(None), "not_estimable")

    def test_exact_null_strict_and_reference_denominators(self):
        rows, pred, spec = fixture()
        pred["a", "t1", 0] = {"result": None, "n_valid": 0}
        pred["b", "t2", 1]["n_valid"] = 2
        pred["b", "t3", 2]["result"]["hate"] = True
        ordinary = c.rate_point(rows, pred, spec)
        self.assertEqual(ordinary["common_ids"], ["b"])
        self.assertEqual(ordinary["outputs_per_input"], [9, 3])
        self.assertEqual(ordinary["difference_exact"], "1/9")
        strict = c.rate_point(rows, pred, spec, strict=True)
        self.assertEqual(strict["common_ids"], [])
        self.assertEqual(strict["missing_real_outputs"], [2, 0])
        for tight in (False, True):
            direct = c.rate_point(rows, pred, spec, strict=tight)
            frozen, _ = rates.summarize(
                rates.cohort(rows, pred, spec, strict=tight), spec
            )
            np.testing.assert_allclose(
                direct["missing_difference_bounds"], frozen["missing_difference_bounds"]
            )

    def test_representative_precedes_validity_and_follows_exclusion(self):
        rows, pred, spec = fixture()
        chosen = c.selected(rows, representatives=True)
        benign = next(r["key"] for r in chosen if not r["gold"]["hate"])
        pred[benign, "t1", 0] = {"result": None, "n_valid": 0}
        self.assertEqual(
            c.rate_point(rows, pred, spec, representatives=True)["common_n"], 0
        )
        after = c.rate_point(rows, pred, spec, representatives=True, excluded=[benign])
        self.assertEqual(after["common_n"], 1)

    def test_pool_omission_recomputes_common_and_keeps_real_reference(self):
        rows, pred, spec = fixture()
        names = {"t" + str(i): "t_s" + str(i) for i in range(1, 4)}
        pred = {
            (k, names.get(cond, cond), rep): v for (k, cond, rep), v in pred.items()
        }
        spec["treatment_conditions"] = list(names.values())
        pred["a", "t_s1", 0] = {"result": None, "n_valid": 0}
        self.assertEqual(c.rate_point(rows, pred, spec)["common_n"], 1)
        omitted = c.omit_pool(spec, 1)
        self.assertEqual(omitted["control_conditions"], ["reference5"])
        result = c.rate_point(rows, pred, omitted)
        self.assertEqual((result["common_n"], result["outputs_per_input"]), (2, [6, 3]))
        self.assertIsNone(c.omit_pool(spec, 99))

    def test_quality_unknown_and_single_class_are_not_imputed(self):
        rows, pred, spec = fixture()
        benign = [r for r in rows if not r["gold"]["hate"]]
        q = c.quality_point(benign, pred, spec)
        self.assertIsNone(q["metrics"]["balanced_accuracy"]["difference"])
        self.assertTrue(q["single_class_binary_slice"])
        fine = c.quality_point(benign, pred, spec, True)
        self.assertEqual(fine["common_n"], 0)
        self.assertIsNone(fine["metrics"]["type_exact_set"]["difference"])
        allfine = c.quality_point(rows, pred, spec, True)
        self.assertEqual(allfine["common_ids"], ["c", "d"])

    def test_quality_is_mean_of_actual_runs_and_null_omission_recovers(self):
        rows, pred, spec = fixture()
        for r in rows:
            pred[r["key"], "t1", 0]["result"]["hate"] = True
        q = c.quality_point(rows, pred, spec)
        runs = [
            c.support.metric_rows(rows, pred, cond, rep)["metrics"]["binary_macro_f1"]
            for cond in spec["treatment_conditions"]
            for rep in range(3)
        ]
        self.assertAlmostEqual(
            q["metrics"]["binary_macro_f1"]["treatment"], sum(runs) / 9
        )
        pred["c", "t3", 2] = {"result": None, "n_valid": 0}
        q = c.quality_point(rows, pred, spec, True)
        self.assertEqual(q["common_ids"], ["d"])
        q = c.quality_point(
            rows, pred, {**spec, "treatment_conditions": ["t1", "t2"]}, True
        )
        self.assertEqual(q["common_ids"], ["c", "d"])

    def test_source_strata_keep_mixed_source_clusters_whole_and_detect_corruption(self):
        rows, pred, spec = fixture()
        rows[1]["source"] = "z"
        rows.append({**deepcopy(rows[0]), "key": "e", "cluster": "c4"})
        for cond in spec["treatment_conditions"] + spec["control_conditions"]:
            for rep in range(3):
                pred["e", cond, rep] = deepcopy(pred["a", cond, rep])
        pred["b", "t1", 0]["result"]["hate"] = True
        cohort = rates.cohort(rows, pred, spec)
        result, draws = c.support.source_bootstrap(cohort, rows, spec)
        self.assertEqual(
            result["strata"],
            [
                {"source_signature": ["x"], "clusters": 1},
                {"source_signature": ["x", "z"], "clusters": 1},
            ],
        )
        checked = p3.integer_draw_check(
            rows, pred, spec, cohort, draws, stratified=True
        )
        self.assertEqual(checked["draws_checked"], 271)
        draws[0] += 0.01
        with self.assertRaisesRegex(ValueError, "Integer bootstrap differs"):
            p3.integer_draw_check(rows, pred, spec, cohort, draws, stratified=True)

    def test_spread_preserves_zero_and_undefined(self):
        self.assertEqual(
            c.spread([None, 0, -1, 2]),
            {
                "n": 4,
                "estimable": 3,
                "min": -1,
                "max": 2,
                "directions": {
                    "not_estimable": 1,
                    "zero": 1,
                    "negative": 1,
                    "positive": 1,
                },
            },
        )
        with self.assertRaisesRegex(ValueError, "Unknown exclusion"):
            c.selected(fixture()[0], ["missing"])


if __name__ == "__main__":
    unittest.main()

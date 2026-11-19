from pathlib import Path

from copy import deepcopy
import itertools
import unittest
import numpy as np
from analysis.feedback.reserved import quality as a
from analysis.feedback.reserved import quality_independent as check
from tests.feedback.reserved.test_support import fixture


class QualityTests(unittest.TestCase):
    def changed(self):
        rows, pred, spec = fixture()
        pred["a", "t1", 0]["result"] = {"hate": True, "hate_type": ["threatening"]}
        pred["c", "t2", 1]["result"] = {"hate": False, "hate_type": []}
        pred["d", "reference5", 2]["result"] = {
            "hate": True,
            "hate_type": ["irony", "explicit"],
        }
        return rows, pred, spec

    def test_unknown_gold_fixed_labels_and_extra_missing_counts(self):
        rows, pred, _ = self.changed()
        for c in ("t1", "t2", "t3", "reference5"):
            for rep in range(3):
                result = a.support.metric_rows(rows, pred, c, rep)
                check.run_check(rows, pred, c, rep, result)
        result = a.support.metric_rows(rows, pred, "t1", 0)
        self.assertEqual(result["counts"]["type_valid"], 2)
        self.assertEqual(result["per_label"]["threatening"]["fp"], 0)
        self.assertEqual(result["metrics"]["type_macro_f1_fixed7"], 3 / 7)

    def test_expanded_clusters_fractional_quantiles_and_undefined_draws(self):
        rows, pred, spec = self.changed()
        for fine in (False, True):
            result, dist = a.support.paired_metrics(rows, pred, spec, fine=fine)
            checked = check.bootstrap_check(rows, pred, spec, result, dist, fine)
            self.assertEqual(checked["values_checked"], 542)
            self.assertLess(checked["max_absolute_error"], 1e-12)
            if not fine:
                self.assertGreater(checked["undefined_draws"]["balanced_accuracy"], 0)
                self.assertNotIn("balanced_accuracy", checked["quantiles"])

    def test_draw_corruption_rejected(self):
        rows, pred, spec = self.changed()
        result, dist = a.support.paired_metrics(rows, pred, spec, fine=True)
        dist[-1, 0] += 0.001
        with self.assertRaisesRegex(ValueError, "expanded bootstrap differs"):
            check.bootstrap_check(rows, pred, spec, result, dist, True)

    def test_empty_and_strict_masks_and_unknown_excluded(self):
        rows, pred, spec = self.changed()
        for key in ("c", "d"):
            pred[key, "t3", 2] = {"result": None, "n_valid": 0}
        result, dist = a.support.paired_metrics(rows, pred, spec, fine=True)
        self.assertEqual(
            check.bootstrap_check(rows, pred, spec, result, dist, True)[
                "draws_checked"
            ],
            0,
        )
        rows, pred, spec = self.changed()
        pred["c", "t1", 0]["n_valid"] = 2
        point, runs = a.point_and_runs(rows, pred, spec, True, True)
        self.assertEqual(point["common_ids"], ["d"])
        for arm in runs:
            for r in arm:
                self.assertEqual(r["counts"]["type_valid"], 1)

    def test_non_linear_mean_is_not_pooled_f1(self):
        rows, pred, spec = fixture()
        for rep in (0, 1):
            for row in rows:
                pred[row["key"], "t1", rep]["result"]["hate"] = True
        runs = [a.support.metric_rows(rows, pred, "t1", r) for r in range(3)]
        mean = sum(r["metrics"]["binary_macro_f1"] for r in runs) / 3
        total = {
            k: sum(r["counts"].get(k, 0) for r in runs)
            for k in ("tp", "tn", "fp", "fn")
        }
        tp, tn, fp, fn = [total[k] for k in ("tp", "tn", "fp", "fn")]
        pooled = (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fp + fn)) / 2
        self.assertNotAlmostEqual(mean, pooled)
        value = check.fractional_scores(
            check.indicators(rows, pred, ["t1"]).sum(axis=0), False
        )[0]
        self.assertAlmostEqual(float(value), mean)

    def test_opposing_bounds_binary_exhaustive(self):
        rows, pred, spec = fixture()
        arms = [spec["treatment_conditions"], spec["control_conditions"]]
        missing = [("a", "t1", 0), ("c", "t1", 0)]
        for k in missing:
            pred[k] = {"result": None, "n_valid": 0}
        result = a.bounds.metric_bounds(rows, pred, arms)
        check.bounds_check(rows, pred, arms, result)
        scores = []
        for decisions in itertools.product((False, True), repeat=2):
            p = deepcopy(pred)
            for k, v in zip(missing, decisions):
                p[k] = {"result": {"hate": v, "hate_type": []}, "n_valid": 3}
            scores.append(
                check.paired_point(rows, p, spec)["metrics"]["binary_macro_f1"][
                    "treatment"
                ]
            )
        np.testing.assert_allclose(
            result["metrics"]["binary_macro_f1"]["treatment"],
            [min(scores), max(scores)],
            atol=1e-14,
        )

    def test_opposing_bounds_fine_all_128_sets(self):
        rows, pred, spec = fixture()
        arms = [spec["treatment_conditions"], spec["control_conditions"]]
        pred["c", "t1", 0] = {"result": None, "n_valid": 0}
        result = a.bounds.metric_bounds(rows, pred, arms, fine=True)
        check.bounds_check(rows, pred, arms, result, True)
        scores = []
        for bits in range(128):
            p = deepcopy(pred)
            labels = [l for i, l in enumerate(check.LABELS) if bits & (1 << i)]
            p["c", "t1", 0] = {
                "result": {"hate": bool(labels), "hate_type": labels},
                "n_valid": 3,
            }
            scores.append(
                check.paired_point(rows, p, spec, True)["metrics"][
                    "type_macro_f1_fixed7"
                ]["treatment"]
            )
        np.testing.assert_allclose(
            result["metrics"]["type_macro_f1_fixed7"]["treatment"],
            [min(scores), max(scores)],
            atol=1e-14,
        )

    def test_strict_missing_bound_and_count_corruption(self):
        rows, pred, spec = self.changed()
        pred["c", "t2", 1]["n_valid"] = 1
        arms = [spec["treatment_conditions"], spec["control_conditions"]]
        for fine in (False, True):
            result = a.bounds.metric_bounds(rows, pred, arms, fine=fine, strict=True)
            checked = check.bounds_check(rows, pred, arms, result, fine, True)
            self.assertEqual(checked["missing_real_outputs_by_arm"], [1, 0])
        run = a.support.metric_rows(rows, pred, "t1", 0)
        run["counts"]["type_extra_labels"] += 1
        with self.assertRaisesRegex(ValueError, "count mismatch"):
            check.run_check(rows, pred, "t1", 0, run)

    def test_original_metric_names_seeds_complete_coverage(self):
        if not (
            Path(__file__).resolve().parents[3]
            / "experiments/astra/sq3_reserved_slot_20261012_v1/scientific_plan_v1.json"
        ).is_file():
            self.skipTest("Original reserved-slot scientific plan is absent")
        science = a.io.read(a.io.STUDY / "scientific_plan_v1.json")
        specs = a.metric_specs(science)
        self.assertEqual(len(specs), 112)
        for s in specs:
            self.assertEqual(s["spec"]["seed"], a.support.seed_for(s["name"]))
        self.assertEqual(sum("/supporting_metrics/" in s["name"] for s in specs), 12)
        self.assertEqual(sum(s["fine"] for s in specs), 56)
        self.assertIn(
            "full/supporting/reserved5__error_balanced-reference5/types",
            {s["name"] for s in specs},
        )


if __name__ == "__main__":
    unittest.main()

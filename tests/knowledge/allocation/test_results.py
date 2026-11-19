"""Tests for allocation results."""

import copy
import unittest
import numpy as np
from tests.knowledge.fixtures import configuration, schedule_fixture
from analysis.knowledge.allocation import results as q


class AllocationScoringTests(unittest.TestCase):

    def setUp(self):
        self.config = configuration("allocation")
        self.config["bootstrap_draws"] = 80
        self.rows = [{"id": str(i), "gate": v} for i, v in enumerate([True, True, False, False])]
        self.ep = {
            "dimension": "hate",
            "all_labels": ["not_hate", "hate"],
            "macro_labels": ["not_hate", "hate"],
            "scorable_ids": [str(i) for i in range(4)],
        }

    def contrib(self, values):
        outs = {
            str(i): {"result": None if v is None else {"hate": v}, "n_valid": 0 if v is None else 3}
            for i, v in enumerate(values)
        }
        return q.s.contributions(self.rows, outs, self.ep)

    def all(self, values):
        return {key: self.contrib(values) for key in q.setting_keys(self.config)}

    def test_identical_conditions_use_six_contrast_family(self):
        result = q.compare(self.all([True, False, True, False]), self.config, focal=True)
        self.assertEqual(len(result["contrasts"]), 3)
        for value in result["contrasts"].values():
            self.assertEqual(value["mean_difference"], 0)
            self.assertEqual(value["ci95"], [0, 0])
            self.assertEqual(value["family_ci"], [0, 0])
            self.assertAlmostEqual(value["family_confidence"], 1 - 0.05 / 6)
            self.assertAlmostEqual(value["family_percentiles"][0], 5 / 12)

    def test_old_family_or_wrong_focal_roster_rejected(self):
        for key, value in [
            ("focal_family_size", 3),
            ("focal_family_confidence", 1 - 0.05 / 3),
            ("focal_endpoints", [q.FOCAL[1]]),
        ]:
            config = copy.deepcopy(self.config)
            config[key] = value
            with self.subTest(key=key), self.assertRaises(q.a.IntegrityError):
                q.compare(self.all([True] * 4), config, focal=True)

    def test_descriptive_endpoint_has_no_family_interval(self):
        result = q.compare(self.all([True, True, False, False]), self.config, focal=False)
        self.assertTrue(
            all(
                (
                    v["family_ci"] is None and v["family_confidence"] is None
                    for v in result["contrasts"].values()
                )
            )
        )

    def test_any_arm_repeat_failure_shrinks_every_contrast(self):
        for arm in q.CONDITIONS:
            cc = self.all([True, True, False, False])
            cc[arm, 2] = self.contrib([None, True, False, False])
            result = q.compare(cc, self.config, focal=False)
            self.assertEqual(result["excluded_from_common_valid"], ["0"])
            self.assertTrue(all((v["n_common_valid"] == 3 for v in result["contrasts"].values())))

    def test_mean_repeat_f1_and_sample_sd_not_pooling(self):
        cc = self.all([True, True, False, False])
        for rep in (1, 2):
            cc["flat2", rep] = self.contrib([False, False, True, True])
        result = q.compare(cc, self.config, focal=False)
        self.assertAlmostEqual(result["contrasts"]["flat7_minus_flat2"]["mean_difference"], 2 / 3)
        summary = result["conditions"]["flat2"]["common_valid_summary"]
        self.assertAlmostEqual(summary["sample_sd"], np.std([1, 0, 0], ddof=1))
        self.assertEqual(summary["range"], 1)

    def test_contrast_direction_and_equal_budget_contrast(self):
        cc = self.all([False, False, True, True])
        for rep in range(3):
            cc["proportional7", rep] = self.contrib([True, True, False, False])
        contrasts = q.compare(cc, self.config, focal=False)["contrasts"]
        self.assertEqual(contrasts["flat7_minus_flat2"]["mean_difference"], 0)
        self.assertEqual(contrasts["proportional_minus_flat2"]["mean_difference"], 1)
        self.assertEqual(contrasts["proportional_minus_flat7"]["mean_difference"], 1)

    def test_missing_and_unpaired_settings_rejected(self):
        cc = self.all([True, True, False, False])
        del cc["flat7", 0]
        with self.assertRaises(q.a.IntegrityError):
            q.compare(cc, self.config, focal=False)
        cc = self.all([True, True, False, False])
        cc["flat7", 0]["ids"] = ["3", "2", "1", "0"]
        with self.assertRaises(q.a.IntegrityError):
            q.compare(cc, self.config, focal=False)
        cc = self.all([True, True, False, False])
        cc["flat7", 0]["gold"][0, 0] = 1
        with self.assertRaises(q.a.IntegrityError):
            q.compare(cc, self.config, focal=False)

    def test_no_common_valid_requires_review(self):
        cc = self.all([True, True, False, False])
        cc["flat7", 0] = self.contrib([None] * 4)
        with self.assertRaises(q.a.IntegrityError):
            q.compare(cc, self.config, focal=False)

    def test_full_roster_bounds_preserve_failure(self):
        cc = self.all([True, True, False, False])
        cc["flat7", 0] = self.contrib([None, True, False, False])
        value = q.compare(cc, self.config, focal=False)["contrasts"]["flat7_minus_flat2"]
        self.assertEqual(value["mean_difference"], 0)
        self.assertLess(value["invalid_inclusive_difference_bounds"][0], 0)
        self.assertEqual(value["invalid_inclusive_difference_bounds"][1], 0)

    def test_partial_and_duplicate_roster_rejected(self):
        settings = {key: dict.fromkeys(["0", "1", "2", "3"]) for key in q.setting_keys(self.config)}
        q.check_settings(settings, ["0", "1", "2", "3"], self.config)
        with self.assertRaises(q.a.IntegrityError):
            q.check_settings(settings, ["0", "1", "2", "3", "3"], self.config)
        del settings["flat7", 2]["3"]
        with self.assertRaises(q.a.IntegrityError):
            q.check_settings(settings, ["0", "1", "2", "3"], self.config)

    def test_zero_support_label_stays_in_macro(self):
        counts = np.asarray([[[1, 0, 0], [0, 0, 0]]])
        self.assertEqual(float(q.s.macro(counts, [0, 1])[0]), 0.5)
        self.assertEqual(float(q.primary.independent_f1(counts, [0, 1])[0]), 0.5)

    def test_severity_gate_miss_is_valid_and_null_is_failure(self):
        ep = {
            "dimension": "severity",
            "all_labels": ["low", "medium", "high"],
            "macro_labels": ["low", "medium", "high"],
            "scorable_ids": ["a"],
        }
        rows = [{"id": "a", "gate": True, "severity": "low"}]
        _, metric = q.severity.checked_score(
            rows, {"a": {"result": {"hate": False, "severity": None}, "n_valid": 3}}, ep
        )
        self.assertEqual(metric["macro_f1_valid_only"], 0)
        self.assertEqual(metric["n_invalid"], 0)
        self.assertEqual(metric["severity_confusion"]["low"]["none_gate_false"], 1)
        _, metric = q.severity.checked_score(rows, {"a": {"result": None, "n_valid": 0}}, ep)
        self.assertIsNone(metric["macro_f1_valid_only"])
        self.assertEqual(metric["n_invalid"], 1)
        self.assertEqual(metric["severity_confusion"]["low"]["invalid"], 1)

    def test_type_scope_aliases(self):
        for subset, scope in [
            ("en_dev_eval_sq3_types", "hate_type_primary"),
            ("en_dev_eval_types", "hate_type_secondary"),
        ]:
            q.check_input_scope(
                {"scopes": [{"dimension": scope, "subset": subset}]}, subset, "hate_type"
            )
            with self.assertRaises(q.a.IntegrityError):
                q.check_input_scope(
                    {"scopes": [{"dimension": "hate_type", "subset": subset}]}, subset, "hate_type"
                )

    def test_legal_independent_of_hate_gate(self):
        labels = ["propaganda", "incitement_public_peace", "insult_defamation"]
        ep = {
            "dimension": "legal",
            "all_labels": labels,
            "macro_labels": labels,
            "scorable_ids": ["a", "b"],
        }
        rows = [
            {"id": "a", "gate": False, "legal": ["insult_defamation"]},
            {"id": "b", "gate": None, "legal": []},
        ]
        outs = {
            "a": {"result": {"hate": False, "legal": ["insult_defamation"]}, "n_valid": 3},
            "b": {"result": {"hate": False, "legal": []}, "n_valid": 3},
        }
        _, metric = q.primary.checked_score(rows, outs, ep)
        self.assertEqual(metric["per_label"]["insult_defamation"]["tp"], 1)
        self.assertAlmostEqual(metric["macro_f1_valid_only"], 1 / 3)

    def test_stage_selection_does_not_allow_partial_rosters(self):
        schedule = schedule_fixture(
            {
                "en_dev_eval_severity": 1,
                "en_dev_eval_targets": 2,
                "en_dev_eval_types": 2,
                "de_dev_eval": 3,
                "de_legal_dev_eval": 3,
                "en_dev_eval_main": 4,
                "en_dev_eval_sq3_types": 4,
            }
        )
        expected = {
            "en_dev_eval_severity": 1,
            "en_dev_eval_targets": 2,
            "en_dev_eval_types": 2,
            "de_dev_eval": 3,
            "de_legal_dev_eval": 3,
            "en_dev_eval_main": 4,
            "en_dev_eval_sq3_types": 4,
        }
        for subset, n in expected.items():
            self.assertEqual(
                [st["id"] for st in q.selected_stages(schedule, subset)],
                [f"stage_{i:02d}" for i in range(1, n + 1)],
            )
        with self.assertRaises(q.a.IntegrityError):
            q.selected_stages(schedule, "unknown")


if __name__ == "__main__":
    unittest.main()

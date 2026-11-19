"""Tests for components results."""

import unittest
import numpy as np
from tests.knowledge.fixtures import configuration, schedule_fixture
from analysis.knowledge.components import results as q


class ComponentScoringTests(unittest.TestCase):

    def setUp(self):
        self.config = configuration("components")
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

    def test_identical_conditions_and_fixed_family(self):
        result = q.compare(self.all([True, False, True, False]), self.config, focal=True)
        contrasts = [v for mask in result.values() for v in mask["contrasts"].values()]
        self.assertEqual(len(contrasts), 17)
        self.assertEqual(sum((v["family_ci"] is not None for v in contrasts)), 3)
        for value in contrasts:
            self.assertAlmostEqual(value["mean_difference"], 0, places=14)
            np.testing.assert_allclose(value["ci95"], [0, 0], rtol=0, atol=1e-14)
            if value["group"] == "focal_leave_one_out":
                self.assertEqual(value["family_ci"], [0, 0])
                self.assertAlmostEqual(value["family_percentiles"][0], 5 / 6)

    def test_reference_failure_cannot_shrink_factorial(self):
        cc = self.all([True, True, False, False])
        cc["reference_rag", 2] = self.contrib([None, True, False, False])
        result = q.compare(cc, self.config, focal=False)
        self.assertEqual(result["factorial"]["common_valid_ids"], ["0", "1", "2", "3"])
        self.assertEqual(result["bridge"]["common_valid_ids"], ["1", "2", "3"])
        for mask in result.values():
            for value in mask["contrasts"].values():
                self.assertIsNone(value["family_ci"])

    def test_other_arm_failure_cannot_shrink_bridge(self):
        cc = self.all([True, True, False, False])
        cc["g", 1] = self.contrib([None, True, False, False])
        result = q.compare(cc, self.config, focal=False)
        self.assertEqual(result["factorial"]["excluded_from_common_valid"], ["0"])
        self.assertEqual(result["bridge"]["excluded_from_common_valid"], [])

    def test_edg_failure_shrinks_both_masks(self):
        cc = self.all([True, True, False, False])
        cc["edg", 0] = self.contrib([None, True, False, False])
        result = q.compare(cc, self.config, focal=False)
        for mask in result.values():
            self.assertEqual(mask["excluded_from_common_valid"], ["0"])

    def test_mean_repeat_f1_not_pooling(self):
        cc = self.all([True, True, False, False])
        for rep in (1, 2):
            cc["none", rep] = self.contrib([False, False, True, True])
        result = q.compare(cc, self.config, focal=False)["factorial"]
        self.assertAlmostEqual(result["contrasts"]["all_context"]["mean_difference"], 2 / 3)
        summary = result["conditions"]["none"]["common_valid_summary"]
        self.assertAlmostEqual(summary["sample_sd"], np.std([1, 0, 0], ddof=1))
        self.assertEqual(summary["range"], 1)

    def test_main_effect_and_interactions_have_declared_sign_scale(self):
        cc = self.all([False, False, True, True])
        for key in cc:
            if key[0] in ("e", "eg", "ed", "edg"):
                cc[key] = self.contrib([True, True, False, False])
        result = q.compare(cc, self.config, focal=False)["factorial"]["contrasts"]
        for name in ("e", "examples_alone", "examples_given_definitions_guidelines"):
            self.assertEqual(result[name]["mean_difference"], 1)
        for name in (
            "d",
            "g",
            "e_x_d",
            "e_x_g",
            "d_x_g",
            "e_x_d_x_g",
            "definitions_given_examples_guidelines",
            "guidelines_given_examples_definitions",
        ):
            self.assertEqual(result[name]["mean_difference"], 0)

    def test_pairwise_interaction_detects_joint_requirement(self):
        cc = self.all([False, False, True, True])
        for name in ("ed", "edg"):
            for rep in range(3):
                cc[name, rep] = self.contrib([True, True, False, False])
        result = q.compare(cc, self.config, focal=False)["factorial"]["contrasts"]
        self.assertEqual(result["e_x_d"]["mean_difference"], 1)
        self.assertEqual(result["e"]["mean_difference"], 0.5)
        self.assertEqual(result["d"]["mean_difference"], 0.5)
        self.assertEqual(result["e_x_d_x_g"]["mean_difference"], 0)

    def test_missing_and_unpaired_setting_rejected(self):
        cc = self.all([True, True, False, False])
        del cc["d", 0]
        with self.assertRaises(q.a.IntegrityError):
            q.compare(cc, self.config, focal=False)
        cc = self.all([True, True, False, False])
        cc["d", 0]["ids"] = ["3", "2", "1", "0"]
        with self.assertRaises(q.a.IntegrityError):
            q.compare(cc, self.config, focal=False)
        cc = self.all([True, True, False, False])
        cc["d", 0]["gold"][0, 0] = 1
        with self.assertRaises(q.a.IntegrityError):
            q.compare(cc, self.config, focal=False)

    def test_no_common_valid_requires_review(self):
        cc = self.all([True, True, False, False])
        cc["d", 0] = self.contrib([None] * 4)
        with self.assertRaises(q.a.IntegrityError):
            q.compare(cc, self.config, focal=False)

    def test_full_roster_bounds_preserve_failure(self):
        cc = self.all([True, True, False, False])
        cc["d", 0] = self.contrib([None, True, False, False])
        result = q.compare(cc, self.config, focal=False)["factorial"]["contrasts"][
            "definitions_alone"
        ]
        self.assertEqual(result["mean_difference"], 0)
        self.assertLess(result["invalid_inclusive_difference_bounds"][0], 0)
        self.assertEqual(result["invalid_inclusive_difference_bounds"][1], 0)

    def test_partial_and_duplicate_roster_rejected(self):
        settings = {key: dict.fromkeys(["0", "1", "2", "3"]) for key in q.setting_keys(self.config)}
        q.check_settings(settings, ["0", "1", "2", "3"], self.config)
        with self.assertRaises(q.a.IntegrityError):
            q.check_settings(settings, ["0", "1", "2", "3", "3"], self.config)
        del settings["edg", 2]["3"]
        with self.assertRaises(q.a.IntegrityError):
            q.check_settings(settings, ["0", "1", "2", "3"], self.config)

    def test_zero_support_class_remains_in_macro(self):
        counts = np.asarray([[[1, 0, 0], [0, 0, 0]]])
        self.assertEqual(float(q.s.macro(counts, [0, 1])[0]), 0.5)
        self.assertEqual(float(q.primary.independent_f1(counts, [0, 1])[0]), 0.5)

    def test_severity_none_and_target_gate_suppression(self):
        ep = {
            "dimension": "severity",
            "all_labels": ["low", "medium", "high"],
            "macro_labels": ["low", "medium", "high"],
            "scorable_ids": ["a"],
        }
        _, metric = q.severity.checked_score(
            [{"id": "a", "gate": True, "severity": "low"}],
            {"a": {"result": {"hate": False, "severity": None}, "n_valid": 3}},
            ep,
        )
        self.assertEqual(metric["macro_f1_valid_only"], 0)
        self.assertEqual(metric["n_invalid"], 0)
        self.assertEqual(metric["severity_confusion"]["low"]["none_gate_false"], 1)
        _, metric = q.severity.checked_score(
            [{"id": "a", "gate": True, "severity": "low"}],
            {"a": {"result": None, "n_valid": 0}},
            ep,
        )
        self.assertIsNone(metric["macro_f1_valid_only"])
        self.assertEqual(metric["n_invalid"], 1)
        self.assertEqual(metric["severity_confusion"]["low"]["invalid"], 1)
        ep = {
            "dimension": "target_group",
            "all_labels": ["race"],
            "macro_labels": ["race"],
            "scorable_ids": ["a"],
        }
        _, metric = q.secondary.checked_score(
            [{"id": "a", "gate": True, "target_groups": ["race"]}],
            {"a": {"result": {"hate": False, "target_group": []}, "n_valid": 3}},
            ep,
        )
        self.assertEqual(metric["per_label"]["race"]["fn"], 1)
        self.assertEqual(metric["n_invalid"], 0)

    def test_type_scope_aliases_preserve_separate_rosters(self):
        for subset, scope in [
            ("en_dev_eval_sq3_types", "hate_type_primary"),
            ("en_dev_eval_types", "hate_type_secondary"),
        ]:
            q.check_input_scope(
                {"scopes": [{"dimension": scope, "subset": subset}]}, subset, "hate_type"
            )
            for wrong in [
                "hate_type",
                "hate_type_secondary" if scope == "hate_type_primary" else "hate_type_primary",
            ]:
                with self.subTest(subset=subset, wrong=wrong), self.assertRaises(
                    q.a.IntegrityError
                ):
                    q.check_input_scope(
                        {"scopes": [{"dimension": wrong, "subset": subset}]}, subset, "hate_type"
                    )

    def test_legal_scoring_stays_independent_of_hate_gate(self):
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
        outputs = {
            "a": {"result": {"hate": False, "legal": ["insult_defamation"]}, "n_valid": 3},
            "b": {"result": {"hate": False, "legal": []}, "n_valid": 3},
        }
        _, metric = q.primary.checked_score(rows, outputs, ep)
        self.assertEqual(metric["per_label"]["insult_defamation"]["tp"], 1)
        self.assertAlmostEqual(metric["macro_f1_valid_only"], 1 / 3)

    def test_stage_selection_requires_later_unfinished_chunks(self):
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
        self.assertEqual(
            [st["id"] for st in q.selected_stages(schedule, "en_dev_eval_severity")], ["stage_01"]
        )
        self.assertEqual(
            [st["id"] for st in q.selected_stages(schedule, "en_dev_eval_sq3_types")],
            ["stage_01", "stage_02", "stage_03", "stage_04"],
        )
        with self.assertRaises(q.a.IntegrityError):
            q.selected_stages(schedule, "unknown")


if __name__ == "__main__":
    unittest.main()

"""Tests for strategies results."""

import copy
import unittest
import numpy as np
from analysis.knowledge.strategies import results as q


class StrategyScoringTests(unittest.TestCase):

    def setUp(self):
        self.config = {
            "conditions": ["rag_dense", "rag_bm25", "rag_hybrid"],
            "repeats": [0, 1, 2],
            "analysis": {
                "contrasts": [["dense", "hybrid"], ["dense", "bm25"], ["hybrid", "bm25"]],
                "bootstrap_draws": 100,
                "bootstrap_seed": 20260920,
                "focal_interval_confidence": 1 - 0.05 / 3,
            },
        }
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

    def test_identical_strategies_zero_intervals(self):
        result = q.compare(self.all([True, False, True, False]), self.config, focal=True)
        for value in result["contrasts"].values():
            self.assertEqual(value["mean_difference"], 0)
            self.assertEqual(value["ci95"], [0, 0])
            self.assertEqual(value["family_ci"], [0, 0])
            self.assertAlmostEqual(value["family_confidence"], 1 - 0.05 / 3)
            self.assertAlmostEqual(value["family_percentiles"][0], 0.8333333333333334)

    def test_third_strategy_failure_also_masks_other_pair(self):
        cc = self.all([True, True, False, False])
        cc["rag_bm25", 2] = self.contrib([None, True, False, False])
        result = q.compare(cc, self.config, focal=False)
        self.assertEqual(result["common_valid_ids"], ["1", "2", "3"])
        self.assertEqual(result["excluded_from_common_valid"], ["0"])
        for value in result["contrasts"].values():
            self.assertEqual(value["n_common_valid"], 3)
            self.assertIsNone(value["family_ci"])

    def test_mean_repeat_f1_not_pooling(self):
        cc = self.all([True, True, False, False])
        for rep in (1, 2):
            cc["rag_bm25", rep] = self.contrib([False, False, True, True])
        result = q.compare(cc, self.config, focal=False)
        self.assertAlmostEqual(result["contrasts"]["dense_minus_bm25"]["mean_difference"], 2 / 3)
        self.assertAlmostEqual(
            result["conditions"]["rag_bm25"]["common_valid_summary"]["sample_sd"],
            np.std([1, 0, 0], ddof=1),
        )

    def test_missing_and_unpaired_setting_rejected(self):
        cc = self.all([True, True, False, False])
        del cc["rag_dense", 0]
        with self.assertRaises(q.a.IntegrityError):
            q.compare(cc, self.config, focal=False)
        cc = self.all([True, True, False, False])
        cc["rag_dense", 0]["ids"] = ["3", "2", "1", "0"]
        with self.assertRaises(q.a.IntegrityError):
            q.compare(cc, self.config, focal=False)

    def test_no_common_valid_requires_review(self):
        cc = self.all([True, True, False, False])
        cc["rag_dense", 0] = self.contrib([None] * 4)
        with self.assertRaises(q.a.IntegrityError):
            q.compare(cc, self.config, focal=False)

    def test_full_roster_bounds_preserve_failure(self):
        cc = self.all([True, True, False, False])
        cc["rag_bm25", 0] = self.contrib([None, True, False, False])
        result = q.compare(cc, self.config, focal=False)["contrasts"]["dense_minus_bm25"]
        self.assertEqual(result["mean_difference"], 0)
        self.assertEqual(result["invalid_inclusive_difference_bounds"][0], 0)
        self.assertGreater(result["invalid_inclusive_difference_bounds"][1], 0)

    def test_partial_roster_rejected(self):
        settings = {key: dict.fromkeys(["0", "1", "2", "3"]) for key in q.setting_keys(self.config)}
        q.check_settings(settings, ["0", "1", "2", "3"], self.config)
        del settings["rag_hybrid", 2]["3"]
        with self.assertRaises(q.a.IntegrityError):
            q.check_settings(settings, ["0", "1", "2", "3"], self.config)

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

    def test_scope_rejects_wrong_dimension_subset_and_multiple_scopes(self):
        valid = {"dimension": "hate", "subset": "de_dev_eval"}
        q.check_input_scope({"scopes": [valid]}, "de_dev_eval", "hate")
        for scope, subset, dimension in [
            ([valid], "de_dev_eval", "legal"),
            ([valid], "en_dev_eval_main", "hate"),
            ([valid, valid], "de_dev_eval", "hate"),
            ([], "de_dev_eval", "hate"),
        ]:
            with self.subTest(scope=scope, subset=subset, dimension=dimension), self.assertRaises(
                q.a.IntegrityError
            ):
                q.check_input_scope({"scopes": scope}, subset, dimension)
        for subset, dimension in [
            ("de_legal_dev_eval", "legal"),
            ("en_dev_eval_targets", "target_group"),
            ("en_dev_eval_severity", "severity"),
        ]:
            q.check_input_scope(
                {"scopes": [{"dimension": dimension, "subset": subset}]}, subset, dimension
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


if __name__ == "__main__":
    unittest.main()

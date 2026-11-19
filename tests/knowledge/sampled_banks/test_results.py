"""Tests for sampled banks results."""

import copy
import unittest
import numpy as np
from tests.knowledge.fixtures import configuration, schedule_fixture
from analysis.knowledge.sampled_banks import results as q


class PoolScoringTests(unittest.TestCase):

    def setUp(self):
        self.config = configuration("sampled_banks")
        self.rows = [{"id": str(i), "gate": v} for i, v in enumerate([True, True, False, False])]
        self.ep = {
            "dimension": "hate",
            "all_labels": ["not_hate", "hate"],
            "macro_labels": ["not_hate", "hate"],
            "scorable_ids": [str(i) for i in range(4)],
        }

    def contribution(self, values):
        out = {
            str(i): {"result": None if v is None else {"hate": v}, "n_valid": 0 if v is None else 3}
            for i, v in enumerate(values)
        }
        return q.s.contributions(self.rows, out, self.ep)

    def all(self, values):
        return {key: self.contribution(values) for key in q.keys(self.config)}

    def compare(self, cc, subset="en_dev_eval_main"):
        return q.compare(cc, self.config, subset, _test_draws=80)

    def test_identical_settings_exact_zero_primary_and_secondary(self):
        result = self.compare(self.all([True, False, True, False]))
        self.assertEqual(len(result["contrasts"]), 8)
        self.assertEqual(sum((v["primary_family_member"] for v in result["contrasts"].values())), 2)
        for row in result["contrasts"].values():
            self.assertEqual(row["mean_difference"], 0)
            self.assertEqual(row["ci95"], [0, 0])
            self.assertEqual(row["family_ci"], [0, 0] if row["primary_family_member"] else None)
            if row["primary_family_member"]:
                self.assertAlmostEqual(row["family_confidence"], 1 - 0.05 / 6)

    def test_secondary_endpoint_never_gets_focal_interval(self):
        result = self.compare(self.all([True, True, False, False]), "en_dev_eval_types")
        self.assertEqual(len(result["contrasts"]), 8)
        self.assertTrue(
            all(
                (
                    not x["primary_family_member"] and x["family_ci"] is None
                    for x in result["contrasts"].values()
                )
            )
        )

    def test_every_pool_and_repeat_participates_in_shared_mask(self):
        for key in q.keys(self.config):
            with self.subTest(key=key):
                cc = self.all([True, True, False, False])
                cc[key] = self.contribution([None, True, False, False])
                result = q.compare(cc, self.config, "en_dev_eval_targets", _test_draws=3)
                self.assertEqual(result["common_valid_ids"], ["1", "2", "3"])
                self.assertTrue(
                    all((x["n_common_valid"] == 3 for x in result["contrasts"].values()))
                )
                self.assertTrue(all((x["mean"] == 1 for x in result["groups"].values())))

    def test_equal_pool_weights_and_separate_decoder_variation(self):
        cc = self.all([False, False, True, True])
        for rep in (0, 1):
            cc["uniform_compact_s42", rep] = self.contribution([True, True, False, False])
        for rep in range(3):
            cc["uniform_compact_s43", rep] = self.contribution([True, True, False, False])
        result = self.compare(cc)
        self.assertEqual(
            result["conditions"]["uniform_compact_s42"]["common_valid_summary"]["values"], [1, 1, 0]
        )
        self.assertAlmostEqual(
            result["conditions"]["uniform_compact_s42"]["common_valid_summary"]["sample_sd"],
            np.std([1, 1, 0], ddof=1),
        )
        group = result["groups"]["uniform_compact"]
        self.assertEqual(len(group["pool_means"]), 3)
        self.assertAlmostEqual(group["mean"], 5 / 9)
        self.assertAlmostEqual(group["between_pool_sample_sd"], np.std([2 / 3, 1, 0], ddof=1))
        self.assertIsNone(result["groups"]["full_reference"]["between_pool_sample_sd"])
        self.assertAlmostEqual(
            result["contrasts"]["full_reference_minus_uniform_compact"]["mean_difference"], -5 / 9
        )

    def test_no_common_rows_has_undefined_scores_and_preserved_bounds(self):
        cc = self.all([True, True, False, False])
        cc["full_reference", 2] = self.contribution([None] * 4)
        result = self.compare(cc)
        self.assertFalse(result["defined"])
        self.assertEqual(result["n_common_valid"], 0)
        self.assertIsNone(result["groups"]["full_reference"]["mean"])
        self.assertIsNone(
            result["contrasts"]["full_reference_minus_curated_small"]["mean_difference"]
        )
        self.assertIsNone(result["contrasts"]["full_reference_minus_curated_small"]["ci95"])
        self.assertAlmostEqual(
            result["contrasts"]["full_reference_minus_curated_small"][
                "invalid_inclusive_difference_bounds"
            ][0],
            -1 / 3,
        )

    def test_failure_bounds_are_full_roster_and_directional(self):
        cc = self.all([True, True, False, False])
        cc["full_reference", 0] = self.contribution([None, True, False, False])
        result = self.compare(cc)
        row = result["contrasts"]["full_reference_minus_curated_small"]
        self.assertEqual(row["mean_difference"], 0)
        self.assertLess(row["invalid_inclusive_difference_bounds"][0], 0)
        self.assertEqual(row["invalid_inclusive_difference_bounds"][1], 0)

    def test_missing_unpaired_or_different_gold_rejected(self):
        cc = self.all([True, True, False, False])
        del cc["uniform_5000_s44", 2]
        with self.assertRaises(q.a.IntegrityError):
            self.compare(cc)
        cc = self.all([True, True, False, False])
        cc["curated_small", 0]["ids"].reverse()
        with self.assertRaises(q.a.IntegrityError):
            self.compare(cc)
        cc = self.all([True, True, False, False])
        cc["curated_small", 0]["gold"][0, 0] = 1
        with self.assertRaises(q.a.IntegrityError):
            self.compare(cc)

    def test_fixed_macro_label_stays_when_draw_has_no_support(self):
        counts = np.asarray([[2, 0, 0], [0, 0, 0]])
        self.assertEqual(float(q.s.macro(counts, [0, 1])), 0.5)
        self.assertEqual(float(q.primary.independent_f1(counts, [0, 1])), 0.5)

    def test_changed_weights_family_seed_or_contrasts_rejected(self):
        for change in ("weights", "family", "seed", "contrasts"):
            cfg = copy.deepcopy(self.config)
            if change == "weights":
                cfg["group_weights"]["uniform_compact"]["uniform_compact_s42"] = 0.5
            elif change == "family":
                cfg["contract"]["bootstrap"]["primary_family_size"] = 3
            elif change == "seed":
                cfg["contract"]["bootstrap"]["seed"] = 42
            else:
                cfg["secondary_contrasts"] = cfg["secondary_contrasts"][:-1]
            with self.subTest(change=change), self.assertRaises(q.a.IntegrityError):
                q.contract(cfg)

    def test_incomplete_endpoint_stage_and_output_rosters_rejected(self):
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
        q.complete_stages(schedule, "en_dev_eval_targets", ["stage_01", "stage_02"])
        with self.assertRaises(q.a.IntegrityError):
            q.complete_stages(schedule, "en_dev_eval_main", ["stage_01", "stage_02"])
        out = {key: dict.fromkeys(["0", "1", "2", "3"]) for key in q.keys(self.config)}
        del out["uniform_5000_s44", 2]["3"]
        with self.assertRaises(q.a.IntegrityError):
            q.m.check_settings(
                out,
                ["0", "1", "2", "3"],
                {"conditions": self.config["conditions"], "repeats": [0, 1, 2]},
            )

    def test_none_and_known_empty_gold_remain_distinct(self):
        rows = [
            {"id": "a", "gate": True, "target_groups": None},
            {"id": "b", "gate": True, "target_groups": []},
        ]
        ep = {
            "dimension": "target_group",
            "all_labels": ["race"],
            "macro_labels": ["race"],
            "scorable_ids": ["b"],
        }
        out = {
            row["id"]: {"result": {"hate": True, "target_group": ["race"]}, "n_valid": 3}
            for row in rows
        }
        c, metric = q.secondary.checked_score(rows, out, ep)
        self.assertEqual(c["ids"], ["b"])
        self.assertEqual(metric["per_label"]["race"]["fp"], 1)

    def test_severity_false_gate_valid_miss_null_invalid(self):
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

    def test_legal_independent_of_false_hate_gate(self):
        ep = {
            "dimension": "legal",
            "all_labels": ["insult_defamation"],
            "macro_labels": ["insult_defamation"],
            "scorable_ids": ["a"],
        }
        rows = [{"id": "a", "gate": False, "legal": ["insult_defamation"]}]
        _, metric = q.primary.checked_score(
            rows,
            {"a": {"result": {"hate": False, "legal": ["insult_defamation"]}, "n_valid": 3}},
            ep,
        )
        self.assertEqual(metric["macro_f1_valid_only"], 1)

    def test_type_scope_aliases_preserved(self):
        for subset, scope in [
            ("en_dev_eval_types", "hate_type_secondary"),
            ("en_dev_eval_sq3_types", "hate_type_primary"),
        ]:
            q.m.check_input_scope(
                {"scopes": [{"dimension": scope, "subset": subset}]}, subset, "hate_type"
            )
            with self.assertRaises(q.a.IntegrityError):
                q.m.check_input_scope(
                    {"scopes": [{"dimension": "hate_type", "subset": subset}]}, subset, "hate_type"
                )


if __name__ == "__main__":
    unittest.main()

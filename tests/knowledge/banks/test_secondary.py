"""Tests for banks secondary."""

import unittest
import numpy as np
from analysis.knowledge.banks import secondary as r


class SecondaryTests(unittest.TestCase):

    def ep(self, ids, dim="target_group", labels=None):
        labels = labels or ["race", "religion"]
        return {"dimension": dim, "all_labels": labels, "macro_labels": labels, "scorable_ids": ids}

    def out(self, labels, gate=True, dim="target_group"):
        return {"result": {"hate": gate, dim: labels}, "n_valid": 3}

    def test_multilabel_counts_not_single_class(self):
        rows = [
            {"id": "a", "gate": True, "target_groups": ["race", "religion"]},
            {"id": "b", "gate": True, "target_groups": ["religion"]},
        ]
        _, score = r.checked_score(
            rows,
            {"a": self.out(["race"]), "b": self.out(["race", "religion"])},
            self.ep(["a", "b"]),
        )
        self.assertEqual(score["per_label_counts"], [[1, 1, 0], [1, 0, 1]])
        self.assertEqual(score["macro_f1_valid_only"], 2 / 3)

    def test_empty_is_observed_missing_is_excluded(self):
        rows = [
            {"id": "a", "gate": np.bool_(True), "target_groups": np.array([])},
            {"id": "b", "gate": True, "target_groups": None},
            {"id": "c", "gate": False, "target_groups": ["race"]},
        ]
        _, score = r.checked_score(
            rows, {row["id"]: self.out(["race"]) for row in rows}, self.ep(["a"])
        )
        self.assertEqual(score["n_scorable"], 1)
        self.assertEqual(score["per_label"]["race"]["fp"], 1)

    def test_gate_suppression_is_fn_not_null(self):
        rows = [{"id": "a", "gate": True, "target_groups": ["race", "religion"]}]
        _, score = r.checked_score(rows, {"a": self.out([], False)}, self.ep(["a"]))
        self.assertEqual(score["n_invalid"], 0)
        self.assertEqual(score["per_label_counts"], [[0, 0, 1], [0, 0, 1]])
        self.assertEqual(score["missed_hate_on_gold_positive_scorable_inputs"], 1)

    def test_invalid_output_and_bounds(self):
        rows = [
            {"id": "a", "gate": True, "target_groups": ["race"]},
            {"id": "b", "gate": True, "target_groups": ["religion"]},
        ]
        out = {"a": self.out(["race"]), "b": {"result": None, "n_valid": 0}}
        _, score = r.checked_score(rows, out, self.ep(["a", "b"]))
        self.assertEqual(score["macro_f1_valid_only"], 0.5)
        self.assertEqual(score["n_invalid"], 1)
        self.assertAlmostEqual(score["invalid_inclusive_bounds"][0], 1 / 3)
        self.assertEqual(score["invalid_inclusive_bounds"][1], 1)

    def test_binary_gate_two_classes_and_missing_mask(self):
        rows = [{"id": "a", "gate": True}, {"id": "b", "gate": False}, {"id": "c", "gate": None}]
        out = {rid: {"result": {"hate": False}, "n_valid": 3} for rid in ["a", "b", "c"]}
        _, score = r.checked_score(rows, out, self.ep(["a", "b"], "hate", ["not_hate", "hate"]))
        self.assertEqual(score["macro_f1_valid_only"], 1 / 3)
        self.assertEqual(score["n_scorable"], 2)

    def test_hate_types_use_their_own_gold_field(self):
        rows = [{"id": "a", "gate": True, "hate_types": ["irony"], "target_groups": ["race"]}]
        _, score = r.checked_score(
            rows,
            {"a": self.out(["irony"], dim="hate_type")},
            self.ep(["a"], "hate_type", ["irony"]),
        )
        self.assertEqual(score["macro_f1_valid_only"], 1)

    def test_missing_duplicate_unknown_and_contradictory_rejected(self):
        rows = [{"id": "a", "gate": True, "target_groups": ["race"]}]
        for rr, outs in [
            (rows, {}),
            (rows + rows, {"a": self.out(["race"])}),
            (rows, {"a": self.out(["invented"])}),
            (rows, {"a": {"result": None, "n_valid": 1}}),
            (rows, {"a": self.out(["race"], False)}),
        ]:
            with self.assertRaises(Exception):
                r.independent_counts(rr, outs, self.ep(["a"]))

    def test_complete_matrix_requires_every_eligible_repeat(self):
        subset = "en_dev_eval_targets"
        matrix = {
            (name, rep): {"a": self.out(["race"])}
            for name in r.conditions(subset)
            for rep in range(3)
        }
        r.validate_settings(matrix, ["a"], subset)
        del matrix["fewshot_eligible_target_group_s42", 0]
        with self.assertRaises(Exception):
            r.validate_settings(matrix, ["a"], subset)

    def test_primary_and_completed_severity_scope_rejected(self):
        for subset in ["en_dev_eval_main", "de_legal_dev_eval", "en_dev_eval_severity", "../bad"]:
            with self.assertRaises(Exception):
                r.destination(subset)

    def test_fixed_macro_labels_can_exclude_rare_label(self):
        ep = self.ep(["a"])
        ep["macro_labels"] = ["race"]
        rows = [{"id": "a", "gate": True, "target_groups": ["race", "religion"]}]
        _, score = r.checked_score(rows, {"a": self.out(["race"])}, ep)
        self.assertEqual(score["macro_f1_valid_only"], 1)
        self.assertEqual(score["per_label"]["religion"]["f1"], 0)


class AuditAcceptanceTests(unittest.TestCase):

    def fixture(self, stage_id="stage_09_secondary"):
        stage = {"id": stage_id, "items": 2700}
        receipt = {"stage": stage_id, "status": "completed_stage_verified_next_stage_ready"}
        ev = {
            "stage": stage_id,
            "status": receipt["status"],
            "semantic_replay_items": 2700,
            "independent_field_vote_checks": 2700,
            "actual_audit_API_calls": 0,
            "actual_gold_scoring": False,
            "completed_stage_restart_new_POSTs": 0,
            "execution_history": {
                "status": "clean_two_attempt_history_verified",
                "completed_items_preserved": 8,
                "resumed_new_items": 2692,
                "administrative_unknown_outcomes": 0,
                "interruption_evidence_sha256": "f5e1ffd4bdafaf20fbf2c58bab7196b1fc69fb64d24a6ff8f5317490e2084c03",
            },
        }
        return (stage, receipt, ev)

    def test_reviewed_history_accepted(self):
        r.checked_audit_binding(*self.fixture())

    def test_missing_or_changed_history_rejected(self):
        for key, value in [
            ("completed_items_preserved", 7),
            ("resumed_new_items", 2700),
            ("administrative_unknown_outcomes", 1),
            ("interruption_evidence_sha256", "wrong"),
        ]:
            stage, receipt, ev = self.fixture()
            ev["execution_history"][key] = value
            with self.assertRaises(r.a.IntegrityError):
                r.checked_audit_binding(stage, receipt, ev)
        stage, receipt, ev = self.fixture()
        del ev["execution_history"]
        with self.assertRaises(r.a.IntegrityError):
            r.checked_audit_binding(stage, receipt, ev)

    def test_wrong_stage_incomplete_or_live_audit_rejected(self):
        for key, value in [
            ("stage", "stage_08_secondary"),
            ("semantic_replay_items", 2699),
            ("independent_field_vote_checks", 2699),
            ("actual_audit_API_calls", 1),
            ("actual_gold_scoring", True),
        ]:
            stage, receipt, ev = self.fixture()
            ev[key] = value
            with self.assertRaises(r.a.IntegrityError):
                r.checked_audit_binding(stage, receipt, ev)

    def test_uninterrupted_stage_still_supported(self):
        stage, receipt, ev = self.fixture("stage_08_secondary")
        del ev["execution_history"]
        r.checked_audit_binding(stage, receipt, ev)


class PreparedScopeTests(unittest.TestCase):

    def test_real_preparation_main_and_sensitivity_tags_accepted(self):
        r.checked_input_scope(
            {"scopes": [{"dimension": "hate_type_primary", "subset": "en_dev_eval_sq3_types"}]},
            "en_dev_eval_sq3_types",
        )
        r.checked_input_scope(
            {"scopes": [{"dimension": "hate_type_secondary", "subset": "en_dev_eval_types"}]},
            "en_dev_eval_types",
        )

    def test_bare_metric_dimension_does_not_identify_type_roster(self):
        for subset in ("en_dev_eval_sq3_types", "en_dev_eval_types"):
            with self.assertRaises(r.a.IntegrityError):
                r.checked_input_scope(
                    {"scopes": [{"dimension": "hate_type", "subset": subset}]}, subset
                )

    def test_swapped_type_tags_and_wrong_subsets_rejected(self):
        for scope in [
            {"dimension": "hate_type_secondary", "subset": "en_dev_eval_sq3_types"},
            {"dimension": "hate_type_primary", "subset": "en_dev_eval_types"},
        ]:
            with self.assertRaises(r.a.IntegrityError):
                r.checked_input_scope({"scopes": [scope]}, "en_dev_eval_sq3_types")

    def test_missing_duplicate_and_mixed_scopes_rejected(self):
        main = {"dimension": "hate_type_primary", "subset": "en_dev_eval_sq3_types"}
        other = {"dimension": "hate_type_secondary", "subset": "en_dev_eval_types"}
        for scopes in ([], [main, main], [main, other]):
            with self.assertRaises(r.a.IntegrityError):
                r.checked_input_scope({"scopes": scopes}, "en_dev_eval_sq3_types")

    def test_unchanged_target_and_german_scopes_accepted(self):
        for subset, dim in [("en_dev_eval_targets", "target_group"), ("de_dev_eval", "hate")]:
            r.checked_input_scope({"scopes": [{"dimension": dim, "subset": subset}]}, subset)

    def test_invented_scope_suffix_and_unknown_endpoint_rejected(self):
        with self.assertRaises(r.a.IntegrityError):
            r.checked_input_scope(
                {"scopes": [{"dimension": "hate_type_future", "subset": "en_dev_eval_sq3_types"}]},
                "en_dev_eval_sq3_types",
            )
        with self.assertRaises(r.a.IntegrityError):
            r.checked_input_scope({"scopes": []}, "en_test_eval_types")


if __name__ == "__main__":
    unittest.main()

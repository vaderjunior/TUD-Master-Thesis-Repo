"""Tests for sampled banks analysis plan."""

from tests.knowledge.fixtures import configuration, inherited_roster
from copy import deepcopy
import math
import unittest
from analysis.knowledge.sampled_banks import analysis_plan as d
from analysis.knowledge.sampled_banks import run as b
from tests.common.test_api import MESSAGES


class PoolAnalysisTests(unittest.TestCase):

    def plan(self):
        return configuration("sampled_banks")

    def test_original_gold_masks_labels_and_six_primary_comparisons(self):
        plan = self.plan()
        inherited = inherited_roster()
        for name, endpoint in plan["endpoints"].items():
            other = deepcopy(endpoint)
            other["role"] = other.pop("inherited_core_role")
            self.assertEqual(other, inherited["endpoints"][name])
        self.assertEqual(
            {(x["endpoint"], x["left"], x["right"]) for x in plan["primary_contrasts"]},
            {
                (endpoint, left, "uniform_compact")
                for endpoint in ("en_dev_eval_main", "en_dev_eval_sq3_types", "de_legal_dev_eval")
                for left in ("full_reference", "curated_small")
            },
        )
        self.assertEqual(len(plan["primary_contrasts"]), 6)
        self.assertEqual(len(plan["secondary_contrasts"]), 50)
        bootstrap = plan["contract"]["bootstrap"]
        self.assertEqual(bootstrap["draws"], 10000)
        self.assertEqual(bootstrap["seed"], 20260926)
        self.assertAlmostEqual(bootstrap["primary_confidence"], 119 / 120)
        self.assertEqual(bootstrap["secondary_confidence"], 0.95)
        self.assertEqual(plan["conditions"], list(d.CONDITIONS))

    def test_no_changed_roster_or_primary_family(self):
        study = d.a.read(d.STUDY / "scientific_plan.json")
        inherited = inherited_roster()
        for change in ("rows", "labels", "family", "conditions"):
            s, old = (deepcopy(study), deepcopy(inherited))
            if change == "rows":
                old["endpoints"]["en_dev_eval_main"]["rows"] -= 1
            elif change == "labels":
                old["endpoints"]["en_dev_eval_main"]["macro_labels"] = ["hate"]
            elif change == "family":
                s["evaluation"]["primary_family"]["contrasts"] = 4
            else:
                s["conditions"].pop()
            with self.subTest(change=change), self.assertRaises(d.a.IntegrityError):
                d.analysis_plan(s, old, "fixture", {})

    def test_pool_variation_is_not_nine_independent_decoder_settings(self):
        values = {c: [0.5, 0.5, 0.5] for c in d.CONDITIONS}
        values["uniform_compact_s42"] = [0, 0, 0]
        values["uniform_compact_s43"] = [0.5, 0.5, 0.5]
        values["uniform_compact_s44"] = [1, 1, 1]
        values["curated_small"] = [0, 0.5, 1]
        result = d.aggregate(values)
        compact = result["groups"]["uniform_compact"]
        self.assertEqual(compact["mean"], 0.5)
        self.assertEqual(compact["between_pool_sample_sd"], 0.5)
        self.assertEqual(compact["pool_means"], [0, 0.5, 1])
        self.assertTrue(
            all(
                (
                    result["per_pool"][c]["decoder_sample_sd"] == 0
                    for c in d.GROUPS["uniform_compact"]
                )
            )
        )
        self.assertEqual(result["per_pool"]["curated_small"]["decoder_sample_sd"], 0.5)
        self.assertIsNone(result["groups"]["curated_small"]["between_pool_sample_sd"])
        self.assertNotAlmostEqual(compact["between_pool_sample_sd"], math.sqrt(3 / 16))

    def test_all_33_shared_mask_keeps_failure_on_unrelated_size_visible(self):
        valid = {(c, r): [True] * 4 for c in d.CONDITIONS for r in range(3)}
        valid["uniform_5000_s44", 2][1] = False
        valid["curated_small", 0][2] = False
        self.assertEqual(
            d.shared_valid([True, True, True, False], valid), [True, False, False, False]
        )
        valid["full_reference", 1][0] = False
        self.assertEqual(d.shared_valid([True, True, True, False], valid), [False] * 4)
        missing = dict(valid)
        missing.pop(next(iter(missing)))
        with self.assertRaises(d.a.IntegrityError):
            d.shared_valid([True] * 4, missing)
        valid["full_reference", 1] = [True]
        with self.assertRaises(d.a.IntegrityError):
            d.shared_valid([True] * 4, valid)

    def test_aggregation_refuses_missing_or_null_scores(self):
        values = {c: [0.5] * 3 for c in d.CONDITIONS}
        for invalid in ([0.5] * 2, [0.5, None, 0.5], [0.5, float("nan"), 0.5], [1.1, 0.5, 0.5]):
            with self.subTest(invalid=invalid), self.assertRaises(d.a.IntegrityError):
                d.aggregate(values | {"curated_small": invalid})
        values.pop("full_reference")
        with self.assertRaises(d.a.IntegrityError):
            d.aggregate(values)

    def test_catalog_binds_original_first_pass_pool_and_scope(self):
        row = {
            "key": "en/fixture",
            "condition": "uniform_compact_s42",
            "pool_id": "p00",
            "strategy": "hybrid",
            "plan_sha256": b.RETRIEVAL_PLAN_SHA,
            "scopes": [{"subset": "en_dev_eval_main"}],
            "messages": deepcopy(MESSAGES),
            "context": "fixture",
        }
        row["messages_sha256"] = d.a.signature(row["messages"])
        row["context_sha256"] = d.a.signature(row["context"])
        checked = {k: row[k] for k in ("key", "condition", "messages_sha256", "context_sha256")}
        checked["entry_sha256"] = d.a.signature(row)
        self.assertEqual(b.catalog_entry(row, checked)["messages"], MESSAGES)
        for field, value in [
            ("pool_id", "p01"),
            ("strategy", "dense"),
            ("plan_sha256", "other"),
            ("scopes", [{"subset": "en_dev_eval_main"}, {"subset": "de_dev_eval"}]),
        ]:
            bad = row | {field: value}
            with self.subTest(field=field), self.assertRaises(d.a.IntegrityError):
                b.catalog_entry(bad, checked)
        replay = deepcopy(row)
        replay["messages"][0]["text"] += "different reopened result"
        replay["messages_sha256"] = d.a.signature(replay["messages"])
        with self.assertRaises(d.a.IntegrityError):
            b.catalog_entry(replay, checked)


if __name__ == "__main__":
    unittest.main()

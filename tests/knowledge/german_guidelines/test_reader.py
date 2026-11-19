"""Tests for german guidelines reader."""

from copy import deepcopy
import json
from pathlib import Path
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import yaml
from analysis.knowledge.german_guidelines import common as c
from analysis.knowledge.german_guidelines import reader as r
from analysis.common import schema as isolated

EMPTY = {
    "hate": False,
    "target_group": [],
    "hate_type": [],
    "severity": None,
    "legal": [],
    "reasoning": "SYNTHETIC READER TEST",
}


class ReaderChecks(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.taxonomy = yaml.safe_load((c.ROOT / "config/taxonomy.yaml").read_text(encoding="utf8"))
        cls.backend = isolated.adapter(c.ROOT / "config/taxonomy.yaml")
        cls.order = list(cls.taxonomy["dimensions"]["severity"]["labels"])

    def compare(self, votes):
        bundle = self.backend.bundle
        runs = [
            bundle.parser.RunResult(result=bundle.schema.Result(**v) if v is not None else None)
            for v in votes
        ]
        expected = bundle.voter.aggregate(runs)
        actual = r.independent_vote(votes, self.order)
        self.assertEqual(
            actual,
            {
                "result": expected.result.model_dump() if expected.result is not None else None,
                "n_valid": expected.n_valid,
                "uncertain": expected.uncertain,
                "agreement": expected.agreement,
            },
        )
        return actual

    def test_nulls_do_not_become_negative(self):
        out = self.compare([None, None, None])
        self.assertIsNone(out["result"])
        self.assertEqual(out["agreement"], {})

    def test_legal_survives_nonhate(self):
        out = self.compare(
            [
                EMPTY | {"legal": ["insult_defamation"]},
                EMPTY | {"legal": ["insult_defamation"]},
                EMPTY,
            ]
        )
        self.assertFalse(out["result"]["hate"])
        self.assertEqual(out["result"]["legal"], ["insult_defamation"])

    def test_one_valid_vote_not_three_independent_answers(self):
        out = self.compare([None, EMPTY | {"reasoning": "only usable reply"}, None])
        self.assertEqual(out["n_valid"], 1)
        self.assertEqual(out["result"]["reasoning"], "only usable reply")

    def test_gate_tie_and_rationale_origin(self):
        yes = EMPTY | {
            "hate": True,
            "hate_type": ["explicit"],
            "severity": "high",
            "reasoning": "yes",
        }
        no = EMPTY | {"reasoning": "no"}
        out = self.compare([yes, no, None])
        self.assertTrue(out["uncertain"])
        self.assertEqual(out["result"]["reasoning"], "no")
        self.assertEqual(out["result"]["hate_type"], [])

    def test_lower_median_and_severity_diagnostic(self):
        votes = [EMPTY | {"hate": True, "severity": s} for s in ("high", "medium")] + [None]
        out = self.compare(votes)
        self.assertEqual(out["result"]["severity"], "medium")
        self.assertEqual(out["agreement"]["severity"], 1.0)

    def test_duplicate_occurrence_semantics_preserved(self):
        one = EMPTY | {"hate": True, "hate_type": ["explicit", "explicit"]}
        votes = [one, EMPTY | {"hate": True}, EMPTY]
        actual = self.compare(votes)
        self.assertEqual(actual["result"]["hate_type"], ["explicit"])
        alternate = r.independent_vote(votes, self.order, deduplicate=True)
        self.assertEqual(alternate["result"]["hate_type"], [])

    def test_duplicate_votes_allowed_but_aggregate_duplicates_rejected(self):
        value = EMPTY | {"legal": ["propaganda", "propaganda"]}
        r.check_decision(value, self.taxonomy, duplicates=True)
        with self.assertRaises(ValueError):
            r.check_decision(value, self.taxonomy)

    def test_unknown_labels_and_inconsistent_fine_heads_rejected(self):
        for value in [
            EMPTY | {"legal": ["unknown"]},
            EMPTY | {"hate_type": ["explicit"]},
            EMPTY | {"hate": 0},
            EMPTY | {"reasoning": None},
        ]:
            with self.assertRaises(ValueError):
                r.check_decision(value, self.taxonomy)

    def test_audited_hash_tampering_rejected(self):
        path = TEST_ROOT / "fixture.json"
        r.save(path, {"fixture": 1})
        digest = c.sha(path)
        self.assertEqual(r.checked_json(path, digest), {"fixture": 1})
        path.write_text('{"fixture": 2}', encoding="utf8")
        with self.assertRaises(ValueError):
            r.checked_json(path, digest)

    def test_legal_gold_empty_is_observed_not_missing(self):
        self.assertEqual(r.scoring.original.truth({"gate": None, "legal": []}, "legal"), [])
        self.assertIsNone(r.scoring.original.truth({"gate": True, "legal": None}, "legal"))

    def test_common_mask_spans_all_conditions_and_repeats(self):
        keys = ["k1", "k2"]
        phase = "synthetic_legal_dev"
        validity = {
            (phase, cond, rep): {key: True for key in keys}
            for cond in c.CONDITIONS
            for rep in range(3)
        }
        validity[phase, "general_2", 2]["k2"] = False
        roster = {
            key: {
                "phase": phase,
                "legal": [] if key == "k1" else ["propaganda"],
                "source": "synthetic",
                "text_cluster": key,
            }
            for key in keys
        }
        endpoint = {
            "scorable_ids": keys,
            "dimension": "legal",
            "all_labels": ["propaganda"],
            "macro_labels": [],
            "gold_support": {"propaganda": 1},
        }
        with patch.object(c, "COUNTS", {phase: 2}):
            result = r.endpoint_setup(roster, {phase: endpoint}, validity)[phase]
        self.assertEqual(result["common_valid_keys"], ["k1"])
        self.assertEqual(result["common_excluded_keys"], ["k2"])
        self.assertEqual(result["cells"]["mixed_2/0"]["valid"], 2)
        self.assertEqual(result["cells"]["general_2/2"]["valid"], 1)
        self.assertEqual(result["gold"], [[], ["propaganda"]])
        self.assertEqual(result["observed_empty_gold_sets"], 1)

    def test_offline_guard_blocks_model_and_scoring(self):
        for action in [
            lambda: c.a.RequestsTransport.__call__(None, {}),
            lambda: r.scoring.original.score(None),
            lambda: r.scoring.contrast(None, None, None),
        ]:
            with self.assertRaises(RuntimeError):
                action()

    def test_timestamps_must_include_timezone(self):
        with self.assertRaises(ValueError):
            r.timestamp("2026-11-12T18:00:00")
        self.assertIsNotNone(r.timestamp("2026-11-12T18:00:00+01:00").tzinfo)


TEST_ROOT = r.BASE / "tests" / f"tests_{time.time_ns()}"


def main():
    with r.offline():
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(ReaderChecks)
        )
        report = {
            "passed": result.wasSuccessful(),
            "tests": result.testsRun,
            "failures": [str(x) for x, _ in result.failures],
            "errors": [str(x) for x, _ in result.errors],
            "source_hashes": c.bindings(r.own_sources()),
            "actual_Qwen_calls": 0,
            "performance_scores_computed": False,
        }
        r.save(TEST_ROOT / "test_report.json", report)
        print(
            json.dumps(
                {
                    "test_report": c.ref(TEST_ROOT / "test_report.json"),
                    "passed": result.wasSuccessful(),
                }
            ),
            flush=True,
        )
        if not result.wasSuccessful():
            raise SystemExit(1)


if __name__ == "__main__":
    main()

import ast
from unittest.mock import patch
import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import unittest
from analysis.knowledge.sampled_banks import validate_stage as c

ORIGINAL_BASE = c.BASE
ORIGINAL_PLAN = c.b.PLAN


class StageAuditTests(unittest.TestCase):

    def test_four_stage_pool_binding(self):
        if (
            not ORIGINAL_PLAN.is_file()
            or not (ORIGINAL_BASE / "schedule.json").is_file()
        ):
            self.skipTest("Original study execution plan and schedule are absent")
        with patch.object(c, "BASE", ORIGINAL_BASE), patch.object(
            c.b, "BASE", ORIGINAL_BASE
        ), patch.object(c.b, "PLAN", ORIGINAL_PLAN):
            self.assertEqual(c.BASE.name, "api_execution_v1")
            self.assertEqual(c.b.STUDY.name, "sq1_pool_size_composition_20260926_v1")
            self.assertEqual(c.b.UNIT, 33)
            self.assertEqual(
                [x["items"] for x in c.a.read(c.BASE / "schedule.json")["stages"]],
                [10692] * 3 + [10659],
            )
            self.assertEqual(c.b.validate()["study_plan_sha256"], c.b.STUDY_PLAN_SHA)

    def test_auditor_is_inherited_with_only_study_bindings_changed(self):
        original = (
            c.ROOT / "analysis/knowledge/allocation/validate_stage.py"
        ).read_text(encoding="utf8")
        expected = (
            original.replace("allocation_api", "pool_api")
            .replace("allocation_stage_check", "pool_stage_check")
            .replace("full-KB SQ1 allocation Qwen stage", "SQ1 sampled-pool Qwen stage")
        )
        actual = (
            c.ROOT / "analysis/knowledge/sampled_banks/validate_stage.py"
        ).read_text(encoding="utf8")
        expected = expected.replace(
            "analysis.knowledge.allocation", "analysis.knowledge.sampled_banks"
        )
        expected = expected.replace(
            "analysis/knowledge/allocation/validate_stage.py",
            "analysis/knowledge/sampled_banks/validate_stage.py",
        )
        expected = expected.replace(
            "tests/knowledge/allocation/", "tests/knowledge/sampled_banks/"
        )
        expected = expected.replace(
            "analysis/knowledge/allocation/", "analysis/knowledge/sampled_banks/"
        )
        expected = expected.replace(
            "tests.knowledge.allocation", "tests.knowledge.sampled_banks"
        )
        self.assertEqual(
            ast.dump(ast.parse(actual), include_attributes=False),
            ast.dump(ast.parse(expected), include_attributes=False),
        )

    def setUp(self):
        self.start = datetime(2026, 9, 20, 10, tzinfo=timezone.utc)
        self.end = datetime(2026, 9, 20, 11, tzinfo=timezone.utc)
        self.intent = {
            "action": "GET PEASEC /models",
            "model": c.a.MODEL,
            "started_utc": "2026-09-20T10:00:01+00:00",
        }
        raw = json.dumps({"data": [{"id": c.a.MODEL}]}).encode()
        self.response = {
            "status": 200,
            "body_base64": base64.b64encode(raw).decode(),
            "body_sha256": hashlib.sha256(raw).hexdigest(),
            "elapsed_s": 0.8,
        }
        self.call = {
            "started_utc": "2026-09-20T10:00:02+00:00",
            "attempt": 0,
            "backoff_elapsed_s": 0,
        }
        self.outcome = {"finished_utc": "2026-09-20T10:00:03+00:00", "elapsed_s": 1}

    def test_model_roster_valid(self):
        self.assertTrue(
            c.model_roster_check(self.intent, self.response, self.start, self.end)[
                "pinned_model_present"
            ]
        )

    def test_model_roster_corrupt_or_wrong(self):
        for field, value in [
            ("status", 401),
            ("body_sha256", "0" * 64),
            ("elapsed_s", float("nan")),
        ]:
            response = {**self.response, field: value}
            with self.subTest(field=field), self.assertRaises(c.a.IntegrityError):
                c.model_roster_check(self.intent, response, self.start, self.end)
        raw = json.dumps({"data": [{"id": "other-model"}]}).encode()
        response = {
            **self.response,
            "body_base64": base64.b64encode(raw).decode(),
            "body_sha256": hashlib.sha256(raw).hexdigest(),
        }
        with self.assertRaises(c.a.IntegrityError):
            c.model_roster_check(self.intent, response, self.start, self.end)

    def test_timing_accepts_initial_and_retry(self):
        c.call_timing_check(self.call, self.outcome, self.start, self.end)
        for attempt, wait in enumerate([2, 2, 4, 8], 1):
            c.call_timing_check(
                {**self.call, "attempt": attempt, "backoff_elapsed_s": wait},
                self.outcome,
                self.start,
                self.end,
            )

    def test_timing_rejects_invalid_order_and_budget(self):
        variants = [
            ({"backoff_elapsed_s": 1}, {}),
            ({"attempt": 1, "backoff_elapsed_s": 1}, {}),
            ({"attempt": 5, "backoff_elapsed_s": 8}, {}),
            ({}, {"elapsed_s": -1}),
            ({"started_utc": "2026-09-20T09:59:59+00:00"}, {}),
            ({}, {"finished_utc": "2026-09-20T10:00:01+00:00"}),
        ]
        for left, right in variants:
            with self.subTest(left=left, right=right), self.assertRaises(
                c.a.IntegrityError
            ):
                c.call_timing_check(
                    {**self.call, **left},
                    {**self.outcome, **right},
                    self.start,
                    self.end,
                )
        with self.assertRaises(c.a.IntegrityError):
            c.call_timing_check(
                self.call,
                self.outcome,
                self.start,
                self.end,
                c.timestamp(self.outcome["finished_utc"]),
            )

    def test_independent_vote_null_tie_and_legal_independence(self):
        self.assertEqual(c.independent_vote([None, None, None]), (None, False))
        p = {
            "hate": True,
            "target_group": ["x"],
            "hate_type": ["y"],
            "severity": "high",
            "legal": ["z"],
        }
        q = {**p, "hate": False, "target_group": [], "hate_type": [], "severity": None}
        result, tie = c.independent_vote([p, q, None])
        self.assertTrue(tie)
        self.assertEqual(
            result,
            {
                "hate": False,
                "target_group": [],
                "hate_type": [],
                "severity": None,
                "legal": ["z"],
            },
        )

    def test_independent_vote_strict_majority_and_lower_median(self):
        p = {
            "hate": True,
            "target_group": ["x"],
            "hate_type": ["y"],
            "severity": "high",
            "legal": [],
        }
        q = {**p, "target_group": ["q"], "severity": "low"}
        result, tie = c.independent_vote([deepcopy(p), q, None])
        self.assertFalse(tie)
        self.assertEqual(result["target_group"], [])
        self.assertEqual(result["hate_type"], ["y"])
        self.assertEqual(result["severity"], "low")


if __name__ == "__main__":
    unittest.main()

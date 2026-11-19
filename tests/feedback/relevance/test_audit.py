import base64
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import unittest
from analysis.feedback.relevance import audit as c


class StageAuditTests(unittest.TestCase):
    def test_study_binding(self):
        if not (c.b.SCIENCE).is_file():
            self.skipTest("Original relevance scientific plan is absent")
        self.assertEqual(c.BASE.name, "api_execution_v1")
        self.assertEqual(c.b.STUDY.name, "sq3_full_kb_relevance_20260930_v1")
        self.assertEqual(c.a.digest(c.b.SCIENCE), c.b.STUDY_PLAN_SHA)
        science = c.a.read(c.b.SCIENCE)
        self.assertEqual(len(science["rosters"]["evaluation"]), 350)
        self.assertEqual(len(science["rosters"]["primary_full_cover"]), 343)

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

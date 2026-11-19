import base64
from collections import Counter
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import unittest
import uuid
from unittest.mock import patch
from analysis.feedback.reserved import review_earlier as r
from analysis.common import api as a
from analysis.common import schema as schema
from analysis.feedback.reserved import recovery as recovery


class Reply:

    def __init__(self, answers=None, retry=False, duplicate=False):
        self.answers = list(answers or [])
        self.retry = retry
        self.duplicate = duplicate
        self.calls = 0
        self.prefix = uuid.uuid4().hex

    def __call__(self, payload):
        self.calls += 1
        if self.retry and self.calls == 1:
            return a.WireReply(503, b"busy", {})
        answer = (
            self.answers.pop(0)
            if self.answers
            else json.dumps(
                {
                    "hate": True,
                    "hate_type": ["explicit"],
                    "target_group": ["religion"],
                    "severity": "medium",
                    "legal": [],
                    "reasoning": "SYNTHETIC ONLY",
                }
            )
        )
        value = {
            "id": "synthetic-"
            + self.prefix
            + ("fixed" if self.duplicate else str(self.calls)),
            "model": a.MODEL,
            "choices": [{"message": {"content": answer}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        return a.WireReply(200, json.dumps(value).encode(), {})


class ReviewTests(unittest.TestCase):

    def setUp(self):
        r.PREP.mkdir(parents=True, exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(prefix="review_test_", dir=r.PREP)
        self.tmp.name = str(r.physical(self.tmp.name))
        self.root = r.logical(self.tmp.name)
        self.backend = schema.adapter(r.ROOT / "config/taxonomy.yaml")
        self.prompt = {
            "messages": [
                {"role": "system", "text": "SYNTHETIC TEST"},
                {"role": "user", "text": "Synthetic text"},
            ]
        }
        self.prompt["messages_sha256"] = a.signature(self.prompt["messages"])
        self.job = {
            "key": "synthetic/text",
            "condition": "reference5",
            "repeat": 0,
            "study": "reserved_slot",
            "messages_sha256": self.prompt["messages_sha256"],
            "schema_path": "config/taxonomy.yaml",
            "schema_sha256": self.backend.taxonomy_sha256,
        }
        self.plan_sha = "a" * 64
        self.begin = datetime.now(timezone.utc) - timedelta(seconds=1)

    def tearDown(self):
        self.assertTrue(self.root.resolve().is_relative_to(r.PREP.resolve()))
        self.tmp.cleanup()

    def execute(self, transport):
        with r.offline():
            self.backend.execute_item(
                self.root,
                r.identity(self.job, self.plan_sha),
                self.prompt["messages"],
                transport,
                sleep=lambda _: None,
                execution_mode="mock",
            )

    def audit(self, admin=()):
        with r.offline():
            return r.audit_item(
                self.root,
                self.job,
                self.plan_sha,
                self.prompt,
                self.backend,
                mode="mock",
                start=self.begin,
                end=datetime.now(timezone.utc) + timedelta(seconds=1),
                administrative_paths=set(admin),
            )

    def test_missing_final_receipt_stops_before_other_reads(self):
        with patch.object(
            r, "read", side_effect=AssertionError("No read before marker")
        ):
            with self.assertRaisesRegex(ValueError, "WAIT:"):
                r.completion_gate(self.root)

    def test_mock_cannot_be_mistaken_for_live_completion(self):
        (self.root / "live").mkdir()
        a.save_new(self.root / "plan.json", {"synthetic": True})
        a.save_new(
            self.root / "live/complete.json",
            {
                "status": "all_studies_executed_and_audited",
                "mode": "mock",
                "plan_sha256": a.digest(self.root / "plan.json"),
            },
        )
        with self.assertRaisesRegex(ValueError, "not completely audited"):
            r.completion_gate(self.root, a.digest(self.root / "plan.json"))

    def test_retry_repair_raw_reparse_and_no_writes(self):
        transport = Reply(["invalid JSON"], retry=True)
        self.execute(transport)
        before = {
            str(p): a.digest(p)
            for p in a.storage_path(self.root).rglob("*")
            if p.is_file()
        }
        record, ids, statuses, unknown = self.audit()
        self.assertEqual(transport.calls, 5)
        self.assertEqual(len(ids), 4)
        self.assertEqual(record["repairs"], 1)
        self.assertEqual(record["accounting"]["retry_attempts"], 1)
        self.assertEqual(statuses["http_503"], 1)
        self.assertEqual(record["n_valid"], 3)
        self.assertFalse(unknown)
        self.assertEqual(record["ordinary_usage"]["total_tokens"], 45)
        self.assertEqual(record["repair_usage"]["total_tokens"], 15)
        self.assertEqual(
            before,
            {
                str(p): a.digest(p)
                for p in a.storage_path(self.root).rglob("*")
                if p.is_file()
            },
        )

    def test_null_is_preserved_after_nine_invalid_replies(self):
        self.execute(Reply(["bad"] * 9))
        record, _, _, _ = self.audit()
        self.assertIsNone(record["result"])
        self.assertEqual(record["n_valid"], 0)
        self.assertEqual(record["repairs"], 6)
        self.assertEqual(record["parse_failures"], 3)

    def test_two_valid_votes_are_preserved(self):
        self.execute(Reply(["bad"] * 3))
        record, _, _, _ = self.audit()
        self.assertEqual(record["n_valid"], 2)

    def test_duplicate_remote_response_ids_rejected(self):
        self.execute(Reply(duplicate=True))
        with self.assertRaisesRegex(ValueError, "Duplicate/missing response ID"):
            self.audit()

    def test_tampered_saved_bytes_rejected(self):
        self.execute(Reply())
        folder = (
            a.storage_path(self.root)
            / "items"
            / a.signature(r.identity(self.job, self.plan_sha))
        )
        p = folder / "result.json"
        original = p.read_bytes()
        p.write_bytes(original + b" ")
        with self.assertRaises(Exception):
            self.audit()

    def test_unresolved_intent_stops_without_resending(self):

        def interrupted(payload):
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self.execute(interrupted)
        transport = Reply()
        with self.assertRaises(a.IncompleteAttempt):
            self.execute(transport)
        self.assertEqual(transport.calls, 0)

    def test_administrative_unknown_zero_excluded_from_latency(self):

        def interrupted(payload):
            raise KeyboardInterrupt()

        with self.assertRaises(KeyboardInterrupt):
            self.execute(interrupted)
        folder = (
            a.storage_path(self.root)
            / "items"
            / a.signature(r.identity(self.job, self.plan_sha))
        )
        call = folder / "calls/vote_0_turn_0/attempt_0"
        intent = a.read(call / "intent.json")
        a.save_new(
            call / "outcome.json", recovery.resolution(intent, "f" * 64, a.utc(), 1)
        )
        self.execute(Reply())
        name = r.rel(call / "outcome.json")
        record, ids, statuses, unknown = self.audit([name])
        self.assertEqual(record["administrative_unknown_requests"], 1)
        self.assertEqual(record["known_request_duration_count"], 3)
        self.assertEqual(record["accounting"]["post_attempts"], 4)
        self.assertEqual(statuses["administrative_unknown"], 1)
        self.assertEqual(unknown, [name])
        self.assertEqual(record["observed_usage"]["total_tokens"], 45)
        with self.assertRaisesRegex(ValueError, "Unregistered"):
            self.audit()

    def test_gate_tie_suppression_and_independent_legal(self):
        p = {
            "hate": True,
            "hate_type": ["explicit"],
            "target_group": ["religion"],
            "severity": "high",
            "legal": ["incitement"],
        }
        q = {**p, "hate": False, "severity": "low"}
        out, tie = r.independent_vote([p, q, None])
        self.assertTrue(tie)
        self.assertFalse(out["hate"])
        self.assertEqual(out["hate_type"], [])
        self.assertIsNone(out["severity"])
        self.assertEqual(out["legal"], ["incitement"])

    def test_windows_extended_path_round_trip_and_escape(self):
        p = self.root / "long" / ("x" * 100) / ("y" * 100) / "item.json"
        a.save_new(p, {"synthetic": True})
        self.assertEqual(r.rel(a.storage_path(p)), r.rel(p))
        self.assertEqual(r.read(p), {"synthetic": True})
        with self.assertRaises(ValueError):
            r.inside(self.root, "../../escaped")

    def test_network_guard_blocks_connect(self):
        import socket

        with r.offline():
            with self.assertRaises(AssertionError):
                socket.create_connection(("example.invalid", 443))


if __name__ == "__main__":
    unittest.main()

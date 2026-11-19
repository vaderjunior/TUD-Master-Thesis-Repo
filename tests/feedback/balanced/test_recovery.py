import json
from pathlib import Path
import unittest
import uuid
from analysis.common import vote_api as old
from analysis.feedback.balanced import recovery_support as r
from analysis.feedback.balanced import resume as new
from analysis.feedback.balanced import resume_audit as audit

a = old.a
GOOD = {
    "reasoning": "Offline mechanical fixture",
    "hate": False,
    "target_group": [],
    "hate_type": [],
    "severity": None,
    "legal": [],
}
MESSAGES = [{"role": "user", "text": "OFFLINE MECHANICAL RECOVERY FIXTURE"}]
REHEARSAL_RESULT = None


def never(*args, **kwargs):
    raise AssertionError("Unexpected request")


class Scripted:

    def __init__(self, values):
        self.values = iter(values)
        self.calls = []

    def __call__(self, payload):
        self.calls.append(payload)
        value = next(self.values)
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, a.WireReply):
            return value
        content = json.dumps(GOOD) if value is None else value
        return a.WireReply(
            200,
            json.dumps(
                {
                    "id": "mock-" + uuid.uuid4().hex,
                    "model": a.MODEL,
                    "choices": [
                        {"finish_reason": "stop", "message": {"content": content}}
                    ],
                    "usage": {
                        "prompt_tokens": 8,
                        "completion_tokens": 3,
                        "total_tokens": 11,
                    },
                }
            ).encode(),
            {},
        )


def fault():
    return a.TransportFault("ConnectionError")


def job(key):
    return {
        "study": "balanced_corrections",
        "key": key,
        "condition": "fixture",
        "repeat": 0,
        "schema_path": "config/taxonomy.yaml",
        "schema_sha256": a.digest(a.ROOT / "config/taxonomy.yaml"),
        "messages_sha256": a.signature(MESSAGES),
        "population": "synthetic",
    }


def catalog(jobs):
    return {(j["key"], j["condition"]): {**j, "messages": MESSAGES} for j in jobs}


def fixture_context(root, jobs, plan="mock-recovery", stage="balanced_corrections_01"):
    calls = []
    before = {}
    for j in jobs:
        d = old.item_dir(root, j, plan)
        files = {
            r.rel(p): a.digest(p)
            for p in d.rglob("*")
            if p.is_file() and p.name != "item.lock"
        }
        terminal = list(d.glob("calls/*/attempt_4/intent.json"))
        assert len(terminal) == 1
        i = a.read(terminal[0])
        calls.append(
            {
                "store": r.rel(root),
                "item": r.rel(d),
                "job": j,
                "vote": i["vote"],
                "turn": i["turn"],
                "prior_hashes": files,
                "body_sha256": i["body_sha256"],
            }
        )
        before.update(files)
    target = a.storage_path(root) / "stages" / stage
    stops = {r.rel(p): a.digest(p) for p in target.glob("stop_*.json")}
    failed = {
        r.rel(p): a.digest(p)
        for p in target.glob("attempt_*/summary.json")
        if a.read(p)["stopped"]
    }
    return {
        "calls": calls,
        "stage": stage,
        "preserved_hashes": before,
        "reviewed_stop_files": stops,
        "reviewed_failed_summaries": failed,
        "source_hashes": {},
        "extra_attempts_per_call": 5,
    }


class RecoveryTests(unittest.TestCase):

    def setUp(self):
        self.root = a.storage_path(
            old.BASE / "offline_check" / ("recovery_" + uuid.uuid4().hex)
        )
        self.j = job("fixture/one")
        self.ident = old.identity(self.j, "mock-recovery")
        self.original = old.isolated.adapter(a.ROOT / "config/taxonomy.yaml")

    def failed(self, prefix=0):
        first = Scripted([None] * prefix + [fault() for _ in range(5)])
        with self.assertRaises(a.CallStopped):
            self.original.execute_item(
                self.root, self.ident, MESSAGES, first, sleep=lambda _: None
            )
        self.ctx = fixture_context(self.root, [self.j])
        r.activate(self.ctx, "mock-amendment")
        self.engine = r.adapter(a.ROOT / "config/taxonomy.yaml")
        return first

    def execute(self, transport):
        return self.engine.execute_item(
            self.root, self.ident, MESSAGES, transport, sleep=lambda _: None
        )

    def test_saved_votes_retained_same_missing_payload_and_zero_completed_resend(self):
        first = self.failed(2)
        second = Scripted([None])
        out = self.execute(second)
        self.assertEqual(len(second.calls), 1)
        self.assertEqual(first.calls[-1], second.calls[0])
        self.assertEqual(out["accounting"]["post_attempts"], 8)
        self.assertEqual(out["accounting"]["retry_attempts"], 5)
        self.assertEqual(out["n_valid"], 3)
        r.verify_files(self.ctx["preserved_hashes"])
        self.engine.audit_item(self.root, self.ident, MESSAGES)
        self.assertEqual(self.execute(never), out)
        self.assertEqual(
            a.read(old.item_dir(self.root, self.j, "mock-recovery") / "complete.json")[
                "reviewed_transport_amendment_sha256"
            ],
            "mock-amendment",
        )

    def test_total_cap_ten_is_not_reset_by_restart(self):
        self.failed()
        second = Scripted([fault() for _ in range(5)])
        with self.assertRaises(a.CallStopped):
            self.execute(second)
        self.assertEqual(len(second.calls), 5)
        with self.assertRaises(a.CallStopped):
            self.execute(never)

    def test_other_vote_keeps_original_five_attempt_cap(self):
        self.failed()
        second = Scripted([None] + [fault() for _ in range(5)])
        with self.assertRaises(a.CallStopped):
            self.execute(second)
        self.assertEqual(len(second.calls), 6)
        item = old.item_dir(self.root, self.j, "mock-recovery")
        self.assertFalse((item / "calls/vote_1_turn_0/attempt_5").exists())

    def test_other_item_keeps_original_five_attempt_cap(self):
        self.failed()
        j = job("fixture/two")
        second = Scripted([fault() for _ in range(5)])
        with self.assertRaises(a.CallStopped):
            self.engine.execute_item(
                self.root,
                old.identity(j, "mock-recovery"),
                MESSAGES,
                second,
                sleep=lambda _: None,
            )
        self.assertEqual(len(second.calls), 5)

    def test_interrupted_extra_attempt_never_blindly_resends(self):
        self.failed()
        second = Scripted([KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            self.execute(second)
        with self.assertRaises(a.IncompleteAttempt):
            self.execute(never)
        self.assertEqual(len(second.calls), 1)

    def test_changed_preserved_outcome_blocks_before_network(self):
        self.failed()
        item = old.item_dir(self.root, self.j, "mock-recovery")
        (item / "calls/vote_0_turn_0/attempt_0/outcome.json").write_text(
            "{}", encoding="utf8"
        )
        with self.assertRaises(a.IntegrityError):
            self.execute(never)

    def test_authentication_failure_is_never_recovered_as_connection(self):
        self.failed()
        second = Scripted([a.WireReply(401, b"{}", {})])
        with self.assertRaises(a.CallStopped):
            self.execute(second)
        self.assertEqual(len(second.calls), 1)
        with self.assertRaises(a.CallStopped):
            self.execute(never)

    def test_completed_invalid_outputs_are_not_resampled(self):
        r.activate(
            {
                "calls": [],
                "stage": "unused",
                "source_hashes": {},
                "reviewed_stop_files": {},
                "reviewed_failed_summaries": {},
            },
            "mock",
        )
        self.engine = r.adapter(a.ROOT / "config/taxonomy.yaml")
        out = self.execute(Scripted(["not JSON"] * 9))
        self.assertIsNone(out["result"])
        self.assertEqual(self.execute(never), out)
        self.engine.audit_item(self.root, self.ident, MESSAGES)

    def test_unrelated_stop_is_blocked(self):
        self.failed()
        target = self.root / "stages/balanced_corrections_01"
        a.save_new(target / "stop_unreviewed.json", {"unreviewed": True})
        with self.assertRaises(a.IntegrityError):
            r.verify_stops(self.root, target)

    def test_original_budget_still_blocks_old_launcher(self):
        self.failed()
        with self.assertRaises(a.CallStopped):
            self.original.execute_item(
                self.root, self.ident, MESSAGES, never, sleep=lambda _: None
            )

    def test_repair_and_three_vote_rules_are_unchanged(self):
        self.failed()
        transport = Scripted(["not JSON", None, None, None])
        out = self.execute(transport)
        self.assertEqual(out["repairs"], 1)
        self.assertEqual(out["n_runs"], 3)
        self.assertEqual(out["n_valid"], 3)
        self.assertEqual(out["accounting"]["ordinary_logical_calls"], 3)
        self.assertEqual(out["accounting"]["repair_logical_calls"], 1)
        self.engine.audit_item(self.root, self.ident, MESSAGES)

    def test_gateway_then_ssl_then_connection_prefix_recovers_without_changes(self):
        first = Scripted(
            [
                a.WireReply(502, b"gateway unavailable", {}),
                a.TransportFault("SSLError"),
                fault(),
                fault(),
                fault(),
            ]
        )
        with self.assertRaises(a.CallStopped):
            self.original.execute_item(
                self.root, self.ident, MESSAGES, first, sleep=lambda _: None
            )
        ctx = fixture_context(self.root, [self.j])
        r.activate(ctx, "mock-mixed-outage")
        self.engine = r.adapter(a.ROOT / "config/taxonomy.yaml")
        second = Scripted([None] * 3)
        out = self.execute(second)
        self.assertEqual(len(second.calls), 3)
        self.assertEqual(out["accounting"]["post_attempts"], 8)
        self.assertEqual(out["accounting"]["retry_attempts"], 5)
        r.verify_files(ctx["preserved_hashes"])
        self.engine.audit_item(self.root, self.ident, MESSAGES)

    def test_new_unreviewed_stop_remains_blocked_after_success(self):
        self.failed()
        self.execute(Scripted([None] * 3))
        target = self.root / "stages/balanced_corrections_01"
        a.save_new(target / "stop_later.json", {"new_failure": True})
        with self.assertRaises(a.IntegrityError):
            r.verify_stops(self.root, target)

    def test_full_failed_stage_resume_audit_finalization_and_completed_restart(self):
        global REHEARSAL_RESULT
        self.root = self.root / "balanced_corrections"
        jobs = [job("fixture/" + str(i)) for i in range(8)]
        cat = catalog(jobs)
        stage = {
            "id": "balanced_corrections_01",
            "study": "balanced_corrections",
            "items": 8,
            "ordinary_calls": 24,
            "jobs": jobs,
        }
        with self.assertRaises(a.IntegrityError):
            old.execute_stage(
                stage,
                cat,
                self.root,
                "mock-recovery",
                mode="mock",
                client_factory=lambda: Scripted([fault() for _ in range(5)]),
                sleep=lambda _: None,
            )
        ctx = fixture_context(self.root, jobs)
        self.assertEqual(len(ctx["calls"]), 8)
        r.activate(ctx, "mock-full-amendment")
        new.execute_stage(
            stage,
            cat,
            self.root,
            "mock-recovery",
            mode="mock",
            client_factory=lambda: Scripted([None] * 24),
            sleep=lambda _: None,
        )
        review = audit.audit_stage(stage, cat, self.root, "mock-recovery", mode="mock")
        report = a.read(self.root / "stages/balanced_corrections_01" / review["report"])
        self.assertEqual(report["accounting"]["post_attempts"], 64)
        self.assertEqual(report["accounting"]["retry_attempts"], 40)
        self.assertEqual(report["attempt_count"], 2)
        self.assertEqual(report["counts"]["predictions"], 8)
        r.verify_files(ctx["preserved_hashes"])
        r.verify_files(ctx["reviewed_stop_files"])
        r.verify_files(ctx["reviewed_failed_summaries"])
        new.execute_stage(
            stage, cat, self.root, "mock-recovery", mode="mock", client_factory=never
        )
        new.verify_review(stage, self.root, "mock-recovery", "mock")
        final = audit.finalize_queue(
            {"stages": [stage]}, self.root.parent, "mock-recovery", mode="mock"
        )
        self.assertEqual(final["counts"]["predictions"], 8)
        self.assertEqual(final["accounting"]["post_attempts"], 64)
        for name, sha in a.read(
            self.root / "stages/balanced_corrections_01" / review["raw_hashes"]
        ).items():
            self.assertEqual(a.digest(self.root / name), sha)
        REHEARSAL_RESULT = {
            "passed": True,
            "root": r.rel(self.root),
            "review_sha256": a.digest(
                self.root / "stages/balanced_corrections_01/review.json"
            ),
            "queue_complete_sha256": a.digest(self.root.parent / "complete.json"),
            "mock_predictions": 8,
            "mock_successful_replies": 24,
            "mock_failed_attempts": 40,
            "old_artifacts_unchanged": len(ctx["preserved_hashes"]),
            "completed_restart_new_requests": 0,
            "actual_API_calls": 0,
        }


if __name__ == "__main__":
    unittest.main()

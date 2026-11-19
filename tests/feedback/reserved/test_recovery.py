from copy import deepcopy
from pathlib import Path
import unittest
import uuid

from analysis.feedback.reserved import recovery as r
from tests.feedback.summary.test_reserved import ReservedTests

a = r.a


class RecoveryTests(unittest.TestCase):
    def fixture(self):
        b, cat, stage, root = ReservedTests().one_job()
        stage["id"] = r.STAGE
        transport = r.original.checker.SyntheticTransport()
        calls = []

        def crash(payload):
            calls.append(deepcopy(payload))
            if len(calls) == 2:
                raise KeyboardInterrupt("simulated shutdown during second vote")
            return transport(payload)

        with self.assertRaises(KeyboardInterrupt):
            b.execute_stage(
                stage,
                cat,
                root,
                "mock-recovery",
                mode="mock",
                client_factory=lambda: crash,
                sleep=lambda _: None,
            )
        with self.assertRaises(a.IncompleteAttempt):
            b.execute_stage(
                stage,
                cat,
                root,
                "mock-recovery",
                mode="mock",
                client_factory=r.c.forbidden,
            )
        folder = b.item_dir(root, stage["jobs"][0], "mock-recovery")
        ip = folder / "calls/vote_1_turn_0/attempt_0/intent.json"
        target = a.storage_path(root) / "stages" / r.STAGE
        manifest = {
            "stage": r.STAGE,
            "store": r.rel(root),
            "manifest_path": r.rel(root.parent / "recovery_manifest.json"),
            "source_hashes": {},
            "stop_files": {r.rel(p): a.digest(p) for p in target.glob("stop_*.json")},
            "calls": [
                {
                    "intent": r.rel(ip),
                    "intent_sha256": a.digest(ip),
                    "outcome": r.rel(ip.with_name("outcome.json")),
                }
            ],
        }
        a.save_new(root.parent / "recovery_manifest.json", manifest)
        before = r.file_hashes(root)
        sha = a.digest(root.parent / "recovery_manifest.json")
        now = a.utc()
        out = r.resolution(a.read(ip), sha, now, 1)
        a.save_new(ip.with_name("outcome.json"), out)
        applied = {
            "reviewed_utc": now,
            "administrative_outcomes": {
                r.rel(ip.with_name("outcome.json")): a.digest(
                    ip.with_name("outcome.json")
                )
            },
        }
        scope = r.Scope(manifest, sha, applied, root)
        return scope, cat, stage, root, transport, calls, before

    def test_record_is_unknown_not_invented_response(self):
        out = r.resolution({"vote": 1, "turn": 0, "attempt": 0}, "fixture", a.utc(), 8)
        self.assertEqual(a.interpret(out), ("retry", r.CATEGORY, None))
        self.assertTrue(out["remote_completion_uncertain"])
        self.assertTrue(
            out["administrative_recovery"]["elapsed_s_is_unknown_zero_sentinel"]
        )
        self.assertFalse(out["administrative_recovery"]["http_response_observed"])
        self.assertFalse(out["administrative_recovery"]["network_exception_observed"])
        self.assertNotIn("body_base64", out)
        self.assertNotIn("usage", out)

    def test_end_to_end_preserves_votes_payload_caps_and_history(self):
        scope, cat, stage, root, transport, calls, before = self.fixture()
        b = r.wrapped_queue(scope)
        resumed = []

        def finish(payload):
            resumed.append(deepcopy(payload))
            return transport(payload)

        b.execute_stage(
            stage,
            cat,
            root,
            "mock-recovery",
            mode="mock",
            client_factory=lambda: finish,
            sleep=lambda _: None,
        )
        self.assertEqual(len(resumed), 2)
        self.assertEqual(resumed[0], calls[-1])
        for n, h in before.items():
            self.assertEqual(a.digest(r.ROOT / n), h)
        review = b.audit.audit_stage(stage, cat, root, "mock-recovery", mode="mock")
        report = a.read(a.storage_path(root) / "stages" / r.STAGE / review["report"])
        self.assertEqual(report["administrative_unknown_requests"], 1)
        self.assertEqual(report["accounting"]["post_attempts"], 4)
        self.assertEqual(report["accounting"]["ordinary_logical_calls"], 3)
        self.assertEqual(report["accounting"]["retry_attempts"], 1)
        self.assertTrue(
            set(scope.m["stop_files"]).issubset(
                {r.rel(a.storage_path(root) / n) for n in report["attempt_hashes"]}
            )
        )
        b.execute_stage(
            stage, cat, root, "mock-recovery", mode="mock", client_factory=r.c.forbidden
        )
        b.verify_review(stage, root, "mock-recovery", "mock")
        result = b.audit.finalize_queue(
            {"stages": [stage]}, root.parent, "mock-recovery", mode="mock"
        )
        self.assertEqual(result["administrative_unknown_requests"], 1)
        self.assertEqual(result["counts"]["predictions"], 1)
        self.assertEqual(transport.count, 3)

    def test_unreviewed_stop_is_rejected(self):
        scope, cat, stage, root, _, _, _ = self.fixture()
        b = r.wrapped_queue(scope)
        a.save_new(
            a.storage_path(root) / "stages" / r.STAGE / "stop_new.json",
            {"new": "unreviewed"},
        )
        with self.assertRaises(a.IntegrityError):
            b.execute_stage(
                stage,
                cat,
                root,
                "mock-recovery",
                mode="mock",
                client_factory=r.c.forbidden,
            )

    def test_deleted_stop_is_rejected_without_deleting_any_fixture(self):
        scope, _, _, root, _, _, _ = self.fixture()
        scope.m["stop_files"][
            r.rel(root / "stages" / r.STAGE / "stop_missing.json")
        ] = "not-present"
        with self.assertRaises(a.IntegrityError):
            scope.verify_stops(root / "stages" / r.STAGE)

    def test_modified_record_binding_is_rejected(self):
        scope, _, _, _, _, _, _ = self.fixture()
        name = next(iter(scope.applied["administrative_outcomes"]))
        scope.applied["administrative_outcomes"][name] = "wrong"
        with self.assertRaises(a.IntegrityError):
            scope.verify_records()

    def test_original_retry_limit_is_not_extended(self):
        scope, cat, stage, root, _, _, _ = self.fixture()
        b = r.wrapped_queue(scope)
        requests = []

        def fail(payload):
            requests.append(payload)
            raise a.TransportFault("ReadTimeout")

        with self.assertRaises(a.IntegrityError):
            b.execute_stage(
                stage,
                cat,
                root,
                "mock-recovery",
                mode="mock",
                client_factory=lambda: fail,
                sleep=lambda _: None,
            )
        self.assertEqual(len(requests), 4)
        folder = (
            b.item_dir(root, stage["jobs"][0], "mock-recovery") / "calls/vote_1_turn_0"
        )
        self.assertEqual(
            {p.name for p in folder.iterdir()}, {f"attempt_{n}" for n in range(5)}
        )
        with self.assertRaises(a.IntegrityError):
            scope.verify_stops(root / "stages" / r.STAGE)

    def test_future_unknown_intent_still_stops(self):
        scope, cat, stage, root, transport, _, _ = self.fixture()
        b = r.wrapped_queue(scope)

        def crash_again(payload):
            raise KeyboardInterrupt("new unresolved request")

        with self.assertRaises(KeyboardInterrupt):
            b.execute_stage(
                stage,
                cat,
                root,
                "mock-recovery",
                mode="mock",
                client_factory=lambda: crash_again,
                sleep=lambda _: None,
            )
        with self.assertRaises(a.IncompleteAttempt):
            b.scan_jobs(
                stage["jobs"], cat, root, "mock-recovery", "mock", b.engines(cat)
            )

    def test_unaffected_stage_runs_with_unchanged_engine(self):
        scope, _, _, _, _, _, _ = self.fixture()
        b = r.wrapped_queue(scope)
        old, cat, stage, root = ReservedTests().one_job()

        root = scope.root
        stage["id"] = "reserved_slot_05"
        stage["jobs"][0]["key"] = "other-fixture"
        entry = cat.pop(("fixture", "reference5"))
        cat[("other-fixture", "reference5")] = entry
        transport = r.original.checker.SyntheticTransport()
        b.execute_stage(
            stage,
            cat,
            root,
            "mock-recovery",
            mode="mock",
            client_factory=lambda: transport,
            sleep=lambda _: None,
        )
        review = b.audit.audit_stage(stage, cat, root, "mock-recovery", mode="mock")
        report = a.read(root / "stages" / stage["id"] / review["report"])
        self.assertEqual(report["administrative_unknown_requests"], 0)
        self.assertEqual(report["accounting"]["post_attempts"], 3)


if __name__ == "__main__":
    with r.c.offline():
        unittest.main()

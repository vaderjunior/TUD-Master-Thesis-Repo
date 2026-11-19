"""Tests for guidelines run."""

from collections import Counter
from copy import deepcopy
import time
import unittest
from unittest.mock import Mock, patch
from analysis.knowledge.guidelines import run as b
from tests.common.test_api import reply, QueueTransport, never, MESSAGES


def fixture(n=2, identical=False):
    inputs, cat = ([], {})
    for i in range(n):
        key = "en/synthetic-" + str(i)
        inputs.append({"key": key})
        for condition in b.CONDITIONS:
            messages = deepcopy(MESSAGES)
            if not identical:
                messages[-1]["text"] += " " + key + " " + condition
            cat[key, condition] = {
                "messages": messages,
                "messages_sha256": b.a.signature(messages),
                "strategy": "hybrid",
                "subset": b.ENDPOINT,
            }
    return (inputs, cat)


class GuidelineApiTests(unittest.TestCase):

    def root(self, purpose):
        return b.a.storage_path(
            b.BASE / "offline_check" / ("unit_" + purpose + "_" + str(time.time_ns()))
        )

    def test_inherited_engine_unchanged(self):
        old = (b.ROOT / "analysis/knowledge/allocation/run.py").read_text(encoding="utf8")
        new = (b.ROOT / "analysis/knowledge/guidelines/run.py").read_text(encoding="utf8")
        segment = lambda text: text[
            text.index("def identity(") : text.index("def run_stage(")
        ].strip()
        self.assertEqual(segment(old), segment(new))

    def test_schedule_preserves_roster_covers_matrix_and_balances_positions(self):
        inputs, cat = fixture(31)
        schedule = b.make_schedule(inputs, cat, cap=46)
        self.assertEqual(schedule, b.make_schedule(inputs, cat, cap=46))
        jobs = [j for s in schedule["stages"] for j in s["jobs"]]
        self.assertEqual([j["key"] for j in jobs[::15]], [row["key"] for row in inputs])
        self.assertTrue(
            all((s["items"] <= 46 and s["items"] % 15 == 0 for s in schedule["stages"]))
        )
        for i in range(31):
            unit = jobs[i * 15 : (i + 1) * 15]
            self.assertEqual(len({j["key"] for j in unit}), 1)
            self.assertEqual(
                {(j["condition"], j["repeat"]) for j in unit},
                {(c, r) for c in b.CONDITIONS for r in range(3)},
            )
            for c in b.CONDITIONS:
                self.assertEqual(len({k % 5 for k, j in enumerate(unit) if j["condition"] == c}), 3)
            for block in range(3):
                self.assertEqual(
                    [j["repeat"] for j in unit[block * 5 : block * 5 + 5]], [(i + block) % 3] * 5
                )
        for position in range(15):
            counts = Counter((j["condition"] for j in jobs[position::15]))
            self.assertEqual(set(counts), set(b.CONDITIONS))
            self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)

    def test_wrong_catalogue_roster_and_cap_refused(self):
        inputs, cat = fixture()
        missing = dict(cat)
        missing.pop(next(iter(missing)))
        extra = dict(cat)
        extra["en/extra", "natural"] = next(iter(cat.values()))
        wrong = deepcopy(cat)
        next(iter(wrong.values()))["subset"] = "wrong"
        for invalid in (missing, extra, wrong):
            with self.assertRaises(b.a.IntegrityError):
                b.make_schedule(inputs, invalid)
        with self.assertRaises(b.a.IntegrityError):
            b.make_schedule(inputs + [inputs[0]], cat)
        for cap in (14, True, 15.0):
            with self.assertRaises(b.a.IntegrityError):
                b.make_schedule(inputs, cat, cap=cap)

    def test_identical_messages_still_get_15_independent_jobs_and_restart_no_calls(self):
        inputs, cat = fixture(1, identical=True)
        stage = b.make_schedule(inputs, cat)["stages"][0]
        root = self.root("identical")
        calls = []

        def factory():

            def send(payload):
                calls.append(payload)
                return reply()

            return send

        b.execute_stage(stage, cat, root, "mock-plan", client_factory=factory, mode="mock")
        self.assertEqual(len(calls), 45)
        self.assertEqual(len(list((root / "items").iterdir())), 15)
        self.assertEqual(len(list((root / "payloads").iterdir())), 1)
        b.execute_stage(
            stage, cat, root, "mock-plan", client_factory=lambda: never({}), mode="mock"
        )
        self.assertEqual(len(calls), 45)
        self.assertFalse((root / "stages/stage_02").exists())
        first = stage["jobs"][0]
        changed = deepcopy(cat)
        changed[first["key"], first["condition"]]["messages"][0]["text"] += "corrupt"
        with self.assertRaises(b.a.IntegrityError):
            b.execute_stage(
                stage, changed, root, "mock-plan", client_factory=lambda: never({}), mode="mock"
            )
        self.assertNotEqual(b.identity(first, "mock-plan"), b.identity(first, "different-plan"))

    def test_partial_saved_reply_reused_without_resending(self):
        inputs, cat = fixture(1)
        stage = b.make_schedule(inputs, cat)["stages"][0]
        root = self.root("partial")
        job = stage["jobs"][0]
        with patch("src.hsrag.parse.parse_response", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                b.a.execute_item(
                    root,
                    b.identity(job, "mock-plan"),
                    cat[job["key"], job["condition"]]["messages"],
                    QueueTransport([reply()]),
                )
        existing, todo = b.scan_jobs(stage["jobs"], cat, root, "mock-plan", "mock")
        self.assertEqual((len(existing), len(todo)), (0, 15))
        calls = []

        def factory():

            def send(payload):
                calls.append(payload)
                return reply()

            return send

        b.execute_stage(stage, cat, root, "mock-plan", client_factory=factory, mode="mock")
        self.assertEqual(len(calls), 44)
        b.check_stage_complete(stage, cat, root, "mock-plan", "mock")

    def test_unresolved_request_blocks_every_new_job_and_keeps_stop(self):
        inputs, cat = fixture(1)
        stage = b.make_schedule(inputs, cat)["stages"][0]
        root = self.root("unknown")
        job = stage["jobs"][-1]
        with self.assertRaises(KeyboardInterrupt):
            b.a.execute_item(
                root,
                b.identity(job, "mock-plan"),
                cat[job["key"], job["condition"]]["messages"],
                QueueTransport([KeyboardInterrupt()]),
            )
        factory = Mock(side_effect=AssertionError("No client allowed"))
        with self.assertRaises(b.a.IncompleteAttempt):
            b.execute_stage(stage, cat, root, "mock-plan", client_factory=factory, mode="mock")
        factory.assert_not_called()
        self.assertEqual(len(list((root / "stages/stage_01").glob("stop_*.json"))), 1)
        with self.assertRaises(b.a.IntegrityError):
            b.execute_stage(stage, cat, root, "mock-plan", client_factory=factory, mode="mock")

    def test_null_after_repair_cap_is_complete_and_never_resampled(self):
        root = self.root("null")
        identity = {
            "experiment_sha256": "mock",
            "key": "en/mock",
            "condition": "natural",
            "repeat": 0,
        }
        transport = QueueTransport([reply("invalid")] * 9)
        result = b.a.execute_item(root, identity, MESSAGES, transport)
        self.assertIsNone(result["result"])
        self.assertEqual(result["repairs"], 6)
        self.assertEqual(result["parse_failures"], 3)
        self.assertEqual(result, b.a.execute_item(root, identity, MESSAGES, never))
        self.assertEqual(b.a.audit_item(root, identity, MESSAGES)["post_attempts"], 9)

    def test_retry_budget_never_resets_on_restart(self):
        root = self.root("budget")
        identity = {
            "experiment_sha256": "mock",
            "key": "en/mock",
            "condition": "natural",
            "repeat": 0,
        }
        transport = QueueTransport([reply(status=503)] * 5)
        with self.assertRaisesRegex(b.a.CallStopped, "exhausted"):
            b.a.execute_item(root, identity, MESSAGES, transport, sleep=lambda _: None)
        with self.assertRaisesRegex(b.a.CallStopped, "exhausted"):
            b.a.execute_item(root, identity, MESSAGES, never, sleep=lambda _: None)
        self.assertEqual(len(transport.payloads), 5)

    def test_transient_retry_and_schema_repair_are_separate(self):
        root = self.root("retry")
        identity = {
            "experiment_sha256": "mock",
            "key": "en/mock",
            "condition": "natural",
            "repeat": 0,
        }
        transport = QueueTransport([reply(status=429), reply("invalid"), reply(), reply(), reply()])
        result = b.a.execute_item(root, identity, MESSAGES, transport, sleep=lambda _: None)
        self.assertEqual(result["repairs"], 1)
        self.assertEqual(result["accounting"]["retry_attempts"], 1)
        self.assertEqual(result["accounting"]["post_attempts"], 5)
        b.a.audit_item(root, identity, MESSAGES)
        for path in root.rglob("outcome.json"):
            self.assertNotIn("DO_NOT_STORE", path.read_text())

    def test_mode_boundary_and_prior_review_content_binding(self):
        inputs, cat = fixture(2)
        schedule = b.make_schedule(inputs, cat, cap=15)
        with self.assertRaises(b.a.IntegrityError):
            b.execute_stage(
                schedule["stages"][0],
                cat,
                b.BASE / "live",
                "mock-plan",
                client_factory=lambda: never({}),
                mode="mock",
            )
        root = self.root("prior")
        with patch.object(b, "BASE", root):
            b.previous_review(0, schedule, "mock-plan")
            with self.assertRaises(b.a.IntegrityError):
                b.previous_review(1, schedule, "mock-plan")
            evidence = root / "evidence.json"
            complete = root / "live/stages/stage_01/complete.json"
            b.a.save_new(evidence, {"mock": True})
            b.a.save_new(complete, {"mock": True})
            b.a.save_new(
                root / "stage_reviews/stage_01.json",
                {
                    "status": "audited",
                    "stage": "stage_01",
                    "plan_sha256": "mock-plan",
                    "complete_sha256": b.a.digest(complete),
                    "evidence": b.rel(evidence),
                    "evidence_sha256": b.a.digest(evidence),
                },
            )
            b.previous_review(1, schedule, "mock-plan")
            with self.assertRaises(b.a.IntegrityError):
                b.previous_review(1, schedule, "changed-plan")
            with patch.object(b.a, "digest", return_value="changed"):
                with self.assertRaises(b.a.IntegrityError):
                    b.previous_review(1, schedule, "mock-plan")

    def test_model_unavailable_stops_without_post_or_credential_storage(self):
        import requests

        attempt = self.root("model")
        client = b.a.RequestsTransport("https://openwebui.srv.peasec.de/api/v1", "fixture-not-real")
        response = Mock(status_code=200, content=b'{"data":[]}')
        response.json.return_value = {"data": []}
        with patch.object(requests, "get", return_value=response) as get, patch.object(
            requests, "post", side_effect=AssertionError
        ) as post:
            with self.assertRaises(b.a.IntegrityError):
                b.live_availability(attempt, client)
        self.assertEqual(get.call_count, 1)
        post.assert_not_called()
        self.assertFalse(get.call_args.kwargs["allow_redirects"])
        self.assertNotIn("fixture-not-real", (attempt / "model_check_response.json").read_text())

    def test_frozen_catalogue_and_internal_wire_schema(self):
        if not (b.STUDY / "retrieval_acceptance_v1.json").exists():
            self.skipTest("Original archived input is not available")
        acceptance = b.a.read(b.STUDY / "retrieval_acceptance_v1.json")
        path = b.ROOT / acceptance["prompts"]
        cat = b.frozen_catalog(path)
        self.assertEqual(len(cat), 3465)
        row = deepcopy(b.core.rows(path)[0])
        row["input_sha256"] = "bad"
        original_rows = b.core.rows
        with patch.object(
            b.core, "rows", side_effect=lambda p: [row] if p == path else original_rows(p)
        ):
            with self.assertRaises(b.a.IntegrityError):
                b.frozen_catalog(path)
        with self.assertRaises(b.a.IntegrityError):
            b.a.body([{"role": "user", "text": "test", "gate": False}])


if __name__ == "__main__":
    unittest.main()

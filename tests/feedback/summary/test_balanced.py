import base64
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
import itertools
import json
import socket
import threading
import time
import unittest
import uuid
from unittest.mock import patch
from analysis.common import vote_api as b
from analysis.common import vote_audit as audit

a = b.a
GOOD = {
    "reasoning": "Synthetic test only",
    "hate": False,
    "target_group": [],
    "hate_type": [],
    "severity": None,
    "legal": ["insult_defamation"],
}
MESSAGES = [
    {"role": "system", "text": "Synthetic classification fixture."},
    {"role": "user", "text": "Synthetic input."},
]
COUNTER = itertools.count()
PREFIX = uuid.uuid4().hex
TEST_ROOT = b.BASE / "offline_check" / f"unit_{time.time_ns()}"


def reply(content=None, status=200, model=a.MODEL, finish="stop", response_id=None):
    value = {
        "id": response_id or f"mock-{PREFIX}-{next(COUNTER)}",
        "model": model,
        "choices": [
            {
                "message": {
                    "content": json.dumps(GOOD) if content is None else content
                },
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return a.WireReply(
        status,
        json.dumps(value).encode(),
        {"Content-Type": "application/json", "Authorization": "NEVER_STORE"},
    )


class Scripted:

    def __init__(self, values):
        self.values = list(values)
        self.calls = []

    def __call__(self, payload):
        self.calls.append(deepcopy(payload))
        assert self.values, "Unexpected new request"
        value = self.values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value if isinstance(value, a.WireReply) else reply(value)


def never(*args, **kwargs):
    raise AssertionError("No new request/client allowed")


def fixture_job(
    study="explanations",
    condition="reference",
    schema=None,
    key="en/synthetic",
    repeat=0,
):
    schema = schema or b.ROOT / "config/taxonomy.yaml"
    return {
        "study": study,
        "key": key,
        "condition": condition,
        "repeat": repeat,
        "schema_path": b.rel(schema),
        "schema_sha256": a.digest(schema),
        "messages_sha256": a.signature(MESSAGES),
        "population": "synthetic",
    }


def fixture_catalog(jobs):
    return {
        (j["key"], j["condition"]): {
            **{
                k: j[k]
                for k in [
                    "schema_path",
                    "schema_sha256",
                    "messages_sha256",
                    "population",
                ]
            },
            "messages": MESSAGES,
            "entry_sha256": "synthetic",
        }
        for j in jobs
    }


class CombinedTests(unittest.TestCase):

    def setUp(self):
        self.root = a.storage_path(TEST_ROOT / self._testMethodName)
        self.engine = b.isolated.adapter(b.ROOT / "config/taxonomy.yaml")
        self.ident = {
            "experiment_sha256": "offline-test",
            "condition": "synthetic",
            "repeat": 0,
            "key": "en/synthetic",
        }

    def run_item(self, transport, engine=None, ident=None, root=None):
        return (engine or self.engine).execute_item(
            root or self.root,
            ident or self.ident,
            MESSAGES,
            transport,
            sleep=lambda _: None,
        )

    def item(self):
        return self.root / "items" / a.signature(self.ident)

    def check_item(self):
        return self.engine.audit_item(self.root, self.ident, MESSAGES)

    def test_full_schema_exactly_matches_production(self):
        values = [
            {
                **GOOD,
                "hate": True,
                "target_group": ["religion"],
                "hate_type": ["grievance"],
                "severity": "high",
            },
            GOOD,
            {**GOOD, "hate": True, "hate_type": ["grievance"], "severity": "low"},
        ]
        isolated = self.run_item(Scripted([json.dumps(v) for v in values]))
        original = a.execute_item(
            self.root / "production",
            self.ident,
            MESSAGES,
            Scripted([json.dumps(v) for v in values]),
            sleep=lambda _: None,
        )
        for field in [
            "result",
            "n_valid",
            "uncertain",
            "agreement",
            "run_predictions",
            "parse_failures",
            "repairs",
        ]:
            self.assertEqual(isolated[field], original[field])
        self.check_item()

    def test_all_eight_taxonomies_have_isolated_validation(self):
        root = (
            b.ROOT / "experiments/astra/sq3_full_kb_taxonomy_20261001_v1" / "taxonomies"
        )
        labels = [
            "explicit",
            "incitement",
            "inferiority",
            "irony",
            "stereotyping",
            "threatening",
            "grievance",
        ]
        for schema in ["full"] + labels:
            engine = b.isolated.adapter(root / f"{schema}.yaml")
            for label in labels:
                content = json.dumps({**GOOD, "hate": True, "hate_type": [label]})
                result, _, _ = engine.bundle.parser.parse_response(content)
                self.assertEqual(result is None, schema == label)

    def test_parallel_schemas_do_not_leak(self):
        path = (
            b.ROOT
            / "experiments/astra/sq3_full_kb_taxonomy_20261001_v1"
            / "taxonomies/grievance.yaml"
        )
        removed = b.isolated.adapter(path)
        raw = json.dumps({**GOOD, "hate": True, "hate_type": ["grievance"]})
        jobs = [(self.engine, False), (removed, True)] * 8

        def run(pair):
            engine, invalid = pair
            ident = {**self.ident, "key": "en/" + uuid.uuid4().hex}
            value = self.run_item(
                Scripted([raw] * (9 if invalid else 3)), engine=engine, ident=ident
            )
            engine.audit_item(self.root, ident, MESSAGES)
            return (value["result"] is None, invalid)

        with ThreadPoolExecutor(max_workers=8) as ex:
            self.assertTrue(
                all((actual == expected for actual, expected in ex.map(run, jobs)))
            )
        self.assertIsNot(self.engine.bundle.schema.Result, removed.bundle.schema.Result)

    def test_removed_label_repair_uses_removed_schema(self):
        removed = b.isolated.adapter(
            b.ROOT
            / "experiments/astra/sq3_full_kb_taxonomy_20261001_v1"
            / "taxonomies/grievance.yaml"
        )
        invalid = json.dumps({**GOOD, "hate": True, "hate_type": ["grievance"]})
        t = Scripted([invalid, None, None, None])
        result = self.run_item(t, engine=removed)
        self.assertEqual(result["repairs"], 1)
        self.assertEqual(len(t.calls[1]["messages"]), 4)
        self.assertNotIn("'grievance'", t.calls[1]["messages"][-1]["content"])
        removed.audit_item(self.root, self.ident, MESSAGES)

    def test_schema_change_cannot_reuse_saved_item(self):
        self.run_item(Scripted([None] * 3))
        removed = b.isolated.adapter(
            b.ROOT
            / "experiments/astra/sq3_full_kb_taxonomy_20261001_v1"
            / "taxonomies/grievance.yaml"
        )
        with self.assertRaises(a.IntegrityError):
            self.run_item(never, engine=removed)

    def test_completed_item_restart_zero_requests(self):
        expected = self.run_item(Scripted([None] * 3))
        self.assertEqual(expected, self.run_item(never))
        self.check_item()
        self.assertEqual(expected["result"]["legal"], ["insult_defamation"])
        self.assertFalse(expected["result"]["hate"])

    def test_ambiguous_interrupted_intent_never_resends(self):
        first = Scripted([KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            self.run_item(first)
        with self.assertRaises(a.IncompleteAttempt):
            self.run_item(never)
        self.assertEqual(len(first.calls), 1)

    def test_saved_response_before_parser_resumes_two_remaining_calls(self):
        first = Scripted([None])
        with patch.object(
            self.engine.bundle.parser, "parse_response", side_effect=KeyboardInterrupt
        ):
            with self.assertRaises(KeyboardInterrupt):
                self.run_item(first)
        second = Scripted([None, None])
        value = self.run_item(second)
        self.assertEqual(len(second.calls), 2)
        self.assertEqual(value["accounting"]["post_attempts"], 3)
        self.check_item()

    def test_retry_and_repair_budgets_preserved(self):
        t = Scripted(
            [
                a.TransportFault("ReadTimeout"),
                reply(status=429),
                "not JSON",
                None,
                None,
                None,
            ]
        )
        value = self.run_item(t)
        self.assertEqual(value["accounting"]["retry_attempts"], 2)
        self.assertEqual(value["repairs"], 1)
        self.check_item()
        self.assertTrue(
            all(
                (
                    "authorization" not in a.read(p)["headers"]
                    for p in self.item().rglob("outcome.json")
                    if a.read(p)["kind"] == "http_response"
                )
            )
        )

    def test_exhausted_retry_does_not_get_new_budget(self):
        t = Scripted([a.TransportFault("ConnectionError")] * 5)
        with self.assertRaises(a.CallStopped):
            self.run_item(t)
        with self.assertRaises(a.CallStopped):
            self.run_item(never)
        self.assertEqual(len(t.calls), 5)

    def test_invalid_outputs_are_completed_nulls(self):
        value = self.run_item(Scripted(["bad JSON"] * 9))
        self.assertIsNone(value["result"])
        self.assertEqual((value["repairs"], value["parse_failures"]), (6, 3))
        self.check_item()
        self.assertEqual(self.run_item(never), value)

    def test_truncation_and_wrong_model_stop(self):
        for idx, wire in enumerate([reply(finish="length"), reply(model="wrong")]):
            with self.assertRaises(a.CallStopped):
                self.run_item(Scripted([wire]), root=self.root / str(idx))

    def test_changed_payload_is_rejected(self):
        self.run_item(Scripted([None] * 3))
        p = next(self.item().rglob("outcome.json"))
        p.write_text("{}", encoding="utf8")
        with self.assertRaises(a.IntegrityError):
            self.run_item(never)

    def test_live_transport_cannot_be_marked_mock(self):
        transport = a.RequestsTransport(
            "https://openwebui.srv.peasec.de/api/v1", "synthetic-not-a-secret"
        )
        with self.assertRaises(a.IntegrityError):
            self.run_item(transport)

    def test_completed_stage_restart_never_constructs_client(self):
        j = fixture_job()
        cat = fixture_catalog([j])
        stage = {
            "id": "synthetic_01",
            "study": "explanations",
            "items": 1,
            "ordinary_calls": 3,
            "jobs": [j],
        }
        b.execute_stage(
            stage,
            cat,
            self.root,
            "offline-test",
            mode="mock",
            client_factory=lambda: Scripted([None] * 3),
            sleep=lambda _: None,
        )
        audit.audit_stage(stage, cat, self.root, "offline-test", mode="mock")
        b.execute_stage(
            stage, cat, self.root, "offline-test", mode="mock", client_factory=never
        )
        b.verify_review(stage, self.root, "offline-test", "mock")

    def test_stage_restart_blocks_ambiguous_request_before_client_creation(self):
        j = fixture_job()
        cat = fixture_catalog([j])
        ident = b.identity(j, "offline-test")
        with self.assertRaises(KeyboardInterrupt):
            self.run_item(Scripted([KeyboardInterrupt()]), ident=ident)
        stage = {
            "id": "synthetic_01",
            "study": "explanations",
            "items": 1,
            "ordinary_calls": 3,
            "jobs": [j],
        }
        with self.assertRaises(a.IncompleteAttempt):
            b.execute_stage(
                stage, cat, self.root, "offline-test", mode="mock", client_factory=never
            )

    def test_clean_stage_interruption_resumes_only_two_unsent_calls(self):
        j = fixture_job()
        cat = fixture_catalog([j])
        stage = {
            "id": "synthetic_01",
            "study": "explanations",
            "items": 1,
            "ordinary_calls": 3,
            "jobs": [j],
        }
        first = Scripted([None])
        with patch.object(
            self.engine.bundle.parser, "parse_response", side_effect=KeyboardInterrupt
        ):
            with self.assertRaises(KeyboardInterrupt):
                b.execute_stage(
                    stage,
                    cat,
                    self.root,
                    "offline-test",
                    mode="mock",
                    client_factory=lambda: first,
                    sleep=lambda _: None,
                )
        second = Scripted([None, None])
        b.execute_stage(
            stage,
            cat,
            self.root,
            "offline-test",
            mode="mock",
            client_factory=lambda: second,
            sleep=lambda _: None,
        )
        review = audit.audit_stage(stage, cat, self.root, "offline-test", mode="mock")
        self.assertEqual((len(first.calls), len(second.calls)), (1, 2))
        self.assertEqual(
            a.read(self.root / "stages" / stage["id"] / review["report"])[
                "attempt_count"
            ],
            2,
        )

    def test_completed_stage_summary_change_stops_before_client(self):
        j = fixture_job()
        cat = fixture_catalog([j])
        stage = {
            "id": "synthetic_01",
            "study": "explanations",
            "items": 1,
            "ordinary_calls": 3,
            "jobs": [j],
        }
        b.execute_stage(
            stage,
            cat,
            self.root,
            "offline-test",
            mode="mock",
            client_factory=lambda: Scripted([None] * 3),
            sleep=lambda _: None,
        )
        receipt = a.read(self.root / "stages/synthetic_01/complete.json")
        a.storage_path(b.ROOT / receipt["final_attempt"] / "summary.json").write_text(
            "{}", encoding="utf8"
        )
        with self.assertRaises(a.IntegrityError):
            b.execute_stage(
                stage, cat, self.root, "offline-test", mode="mock", client_factory=never
            )

    def test_stage_audit_rejects_duplicate_response_ids(self):
        j = fixture_job()
        cat = fixture_catalog([j])
        stage = {
            "id": "synthetic_01",
            "study": "explanations",
            "items": 1,
            "ordinary_calls": 3,
            "jobs": [j],
        }
        b.execute_stage(
            stage,
            cat,
            self.root,
            "offline-test",
            mode="mock",
            client_factory=lambda: Scripted([reply(response_id="same")] * 3),
            sleep=lambda _: None,
        )
        with self.assertRaises(a.IntegrityError):
            audit.audit_stage(stage, cat, self.root, "offline-test", mode="mock")
        self.assertFalse((self.root / "stages/synthetic_01/review.json").exists())

    def test_failed_audit_prevents_next_stage(self):
        seen = []

        def fail(stage):
            raise a.IntegrityError("Synthetic audit failure")

        with self.assertRaises(a.IntegrityError):
            b.run_sequence(
                ["one", "two"], lambda x: seen.append(x), fail, lambda _: False, never
            )
        self.assertEqual(seen, ["one"])

    def test_readiness_gate_precedes_live_client_creation(self):
        with patch.object(
            b,
            "validate",
            side_effect=a.IntegrityError("Synthetic missing ready receipt"),
        ), patch.object(
            b.core, "live_client", side_effect=AssertionError("Credentials touched")
        ) as client:
            with self.assertRaises(a.IntegrityError):
                b.run_all()
            client.assert_not_called()

    def test_full_science_schedule_exact_counts_and_shared_references(self):
        if not all(
            (info["root"] / "scientific_plan_v1.json").is_file()
            for info in b.STUDIES.values()
        ):
            self.skipTest("Original study rosters and scientific plans are absent")
        cats = {}
        for study, info in b.STUDIES.items():
            science = a.read(info["root"] / "scientific_plan_v1.json")
            rr = b.core.rows(info["root"] / "roster.jsonl")
            cats[study] = {
                (r["key"], cond): {
                    "messages_sha256": "fixture",
                    "schema_path": "fixture",
                    "schema_sha256": "fixture",
                    "population": r["population"],
                }
                for r in rr
                for cond in science["conditions"]
            }
        schedule = b.make_schedule(cats)
        jobs = [j for st in schedule["stages"] for j in st["jobs"]]
        self.assertEqual(len(jobs), 38475)
        self.assertEqual(
            sum((st["ordinary_calls"] for st in schedule["stages"])), 115425
        )
        self.assertEqual(
            len({(j["study"], j["condition"], j["repeat"], j["key"]) for j in jobs}),
            38475,
        )
        self.assertEqual(sum((j["condition"] == "reference" for j in jobs)), 675 * 3)
        self.assertEqual(len(schedule["stages"]), 4)
        assignment = {}
        for st in schedule["stages"]:
            for j in st["jobs"]:
                key = (j["study"], j["key"])
                self.assertEqual(assignment.setdefault(key, st["id"]), st["id"])

    def test_timing_and_vote_boundary_checks(self):
        begin = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
        end = datetime(2026, 10, 1, 13, tzinfo=timezone.utc)
        intent = {
            "started_utc": "2026-10-01T12:01:00+00:00",
            "attempt": 1,
            "backoff_elapsed_s": 2.0,
        }
        outcome = {"finished_utc": "2026-10-01T12:01:01+00:00", "elapsed_s": 1.0}
        audit.timing(intent, outcome, begin, end)
        with self.assertRaises(a.IntegrityError):
            audit.timing({**intent, "backoff_elapsed_s": 0.5}, outcome, begin, end)
        with self.assertRaises(a.IntegrityError):
            audit.timing({**intent, "attempt": 5}, outcome, begin, end)
        p = {**GOOD, "hate": True, "hate_type": ["grievance"], "severity": "high"}
        out, tie = audit.independent_vote([p, GOOD, None])
        self.assertTrue(tie)
        self.assertFalse(out["hate"])
        self.assertEqual(out["legal"], ["insult_defamation"])


def main():
    with patch.object(socket.socket, "connect", never), patch.object(
        socket, "create_connection", never
    ):
        from tests.feedback.balanced.test_science import ScienceTests

        suite = unittest.TestSuite(
            [
                unittest.defaultTestLoader.loadTestsFromTestCase(CombinedTests),
                unittest.defaultTestLoader.loadTestsFromTestCase(ScienceTests),
            ]
        )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    b.save(
        b.test_receipt(),
        {
            "passed": True,
            "tests_run": result.testsRun,
            "created_utc": a.utc(),
            "environment": b.core.environment(),
            "source_hashes": {n: a.digest(b.ROOT / n) for n in b.SOURCE_FILES},
            "preserved_fixtures": b.rel(TEST_ROOT),
            "actual_API_calls": 0,
        },
    )


if __name__ == "__main__":
    main()

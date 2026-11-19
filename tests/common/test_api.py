"""Tests for common api."""

from concurrent.futures import ThreadPoolExecutor
import base64
import ast
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
from analysis.common import api as api

MESSAGES = [
    {"role": "system", "text": "Classify using the supplied labels."},
    {"role": "user", "text": "A synthetic input."},
]
IDENTITY = {
    "experiment_sha256": "offline-test-only",
    "condition": "synthetic",
    "repeat": 0,
    "key": "en/synthetic",
}
GOOD = {
    "reasoning": "Synthetic fixture",
    "hate": False,
    "target_group": [],
    "hate_type": [],
    "severity": None,
    "legal": ["insult_defamation"],
}


def reply(content=None, status=200, model=api.MODEL, finish="stop"):
    data = {
        "model": model,
        "choices": [
            {
                "message": {"content": json.dumps(GOOD) if content is None else content},
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    return api.WireReply(
        status,
        json.dumps(data).encode(),
        {
            "Content-Type": "application/json",
            "Authorization": "DO_NOT_STORE",
            "Set-Cookie": "DO_NOT_STORE",
        },
    )


class QueueTransport:

    def __init__(self, values):
        self.values = list(values)
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(copy.deepcopy(payload))
        if not self.values:
            raise AssertionError("Unexpected extra POST")
        value = self.values.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def never(payload):
    raise AssertionError("A completed or blocked job must not make a new POST")


class DurableTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

        def cleanup():
            target = api.storage_path(self.temp.name).resolve()
            self.assertEqual(target.parent, api.storage_path(tempfile.gettempdir()).resolve())
            self.assertTrue(target.name.startswith("tmp"))
            self.temp.name = str(target)
            self.temp.cleanup()

        self.addCleanup(cleanup)
        self.root = Path(self.temp.name)
        self.identity = dict(IDENTITY)

    def run_item(self, transport, messages=MESSAGES):
        return api.execute_item(self.root, self.identity, messages, transport, sleep=lambda _: None)

    def audit(self):
        return api.audit_item(self.root, self.identity, MESSAGES)

    def job(self):
        return self.root / "items" / api.signature(self.identity)

    def test_completed_restart_sends_nothing_and_retains_legal_when_not_hate(self):
        t = QueueTransport([reply()] * 3)
        out = self.run_item(t)
        self.assertEqual(out["result"]["legal"], ["insult_defamation"])
        self.assertFalse(out["result"]["hate"])
        self.assertEqual(out, self.run_item(never))
        self.assertEqual(self.audit()["post_attempts"], 3)
        self.assertEqual(len(list(self.root.glob("payloads/*.json"))), 1)
        for p in self.root.rglob("outcome.json"):
            self.assertNotIn("authorization", api.read(p)["headers"])
            self.assertNotIn("DO_NOT_STORE", p.read_text())

    def test_production_depth_paths_work_and_resume_without_resending(self):
        self.root = self.root / ("deep_experiment_" * 5) / ("nested_attempt_" * 5)
        self.assertGreater(len(str(self.job() / "calls/vote_0_turn_0/attempt_0/outcome.json")), 260)
        out = self.run_item(QueueTransport([reply()] * 3))
        self.assertEqual(out, self.run_item(never))
        self.audit()

    def test_repair_is_exact_original_multiturn_conversation(self):
        from src.hsrag.parse import parse_response, REPAIR_TEMPLATE

        broken = "not JSON"
        t = QueueTransport([reply(broken), reply(), reply(), reply()])
        out = self.run_item(t)
        err = parse_response(broken)[2]
        self.assertEqual(
            t.payloads[1]["messages"],
            api.body(MESSAGES)["messages"]
            + [
                {"role": "assistant", "content": broken},
                {"role": "user", "content": REPAIR_TEMPLATE.format(error=err)},
            ],
        )
        self.assertEqual(out["repairs"], 1)
        self.assertEqual(out["accounting"]["post_attempts"], 4)
        self.audit()

    def test_all_parse_failures_are_completed_null_and_never_resampled(self):
        t = QueueTransport([reply("invalid")] * 9)
        out = self.run_item(t)
        self.assertIsNone(out["result"])
        self.assertEqual(out["n_valid"], 0)
        self.assertEqual(out["repairs"], 6)
        self.assertEqual(out["parse_failures"], 3)
        self.assertEqual(out, self.run_item(never))
        self.audit()

    def test_two_valid_vote_tie_uses_original_gate_rule(self):
        hateful = {
            **GOOD,
            "hate": True,
            "target_group": ["gender"],
            "hate_type": ["explicit"],
            "severity": "high",
        }
        t = QueueTransport([reply("bad")] * 3 + [reply(json.dumps(hateful)), reply()])
        out = self.run_item(t)
        self.assertTrue(out["uncertain"])
        self.assertFalse(out["result"]["hate"])
        self.assertEqual(out["n_valid"], 2)
        self.assertEqual(out["result"]["legal"], GOOD["legal"])
        self.audit()

    def test_wire_retries_count_separately_from_model_repairs(self):
        t = QueueTransport(
            [reply(status=429), api.TransportFault("ReadTimeout"), reply(), reply(), reply()]
        )
        out = self.run_item(t)
        self.assertEqual(out["repairs"], 0)
        self.assertEqual(out["accounting"]["post_attempts"], 5)
        self.assertEqual(out["accounting"]["retry_attempts"], 2)
        self.audit()

    def test_transport_budget_cannot_reset_on_restart(self):
        t = QueueTransport([reply(status=503)] * 5)
        with self.assertRaisesRegex(api.CallStopped, "exhausted"):
            self.run_item(t)
        with self.assertRaisesRegex(api.CallStopped, "exhausted"):
            self.run_item(never)
        self.assertEqual(len(t.payloads), 5)
        self.assertFalse((self.job() / "complete.json").exists())

    def test_fatal_statuses_store_body_and_never_resend(self):
        for status in (400, 401, 403, 302):
            with self.subTest(status=status):
                self.identity = {**IDENTITY, "key": "en/status" + str(status)}
                t = QueueTransport([reply(status=status)])
                with self.assertRaises(api.CallStopped):
                    self.run_item(t)
                with self.assertRaises(api.CallStopped):
                    self.run_item(never)
                p = next(self.job().rglob("outcome.json"))
                self.assertEqual(api.read(p)["status"], status)

    def test_truncation_model_drift_and_malformed_envelope_stop(self):
        cases = [
            reply(finish="length"),
            reply(model="other-model"),
            api.WireReply(200, b'{"choices":[]}', {}),
        ]
        for i, wire in enumerate(cases):
            with self.subTest(i=i):
                self.identity = {**IDENTITY, "key": "en/terminal" + str(i)}
                with self.assertRaises(api.CallStopped):
                    self.run_item(QueueTransport([wire]))
                self.assertEqual(
                    base64.b64decode(
                        api.read(next(self.job().rglob("outcome.json")))["body_base64"]
                    ),
                    wire.content,
                )
                with self.assertRaises(api.CallStopped):
                    self.run_item(never)

    def test_invalid_gateway_json_is_saved_and_retried(self):
        t = QueueTransport([api.WireReply(200, b"not json", {}), reply(), reply(), reply()])
        out = self.run_item(t)
        self.assertEqual(out["accounting"]["retry_attempts"], 1)
        self.audit()

    def test_missing_reported_model_is_explicit_not_invented(self):
        out = self.run_item(QueueTransport([reply(model=None)] * 3))
        self.assertEqual(out["accounting"]["reported_model_present"], 0)
        self.audit()

    def test_interrupted_intent_blocks_without_a_new_post(self):
        t = QueueTransport([KeyboardInterrupt()])
        with self.assertRaises(KeyboardInterrupt):
            self.run_item(t)
        with self.assertRaises(api.IncompleteAttempt):
            self.run_item(never)
        self.assertEqual(len(t.payloads), 1)
        self.assertEqual(len(list(self.job().rglob("intent.json"))), 1)
        self.assertEqual(len(list(self.job().rglob("outcome.json"))), 0)

    def test_restart_after_saved_response_before_vote_uses_cached_response(self):
        from src.hsrag import parse

        real = parse.parse_response
        t = QueueTransport([reply()])
        with patch.object(parse, "parse_response", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_item(t)
        later = QueueTransport([reply(), reply()])
        out = self.run_item(later)
        self.assertEqual(len(later.payloads), 2)
        self.assertEqual(out["n_valid"], 3)
        self.audit()

    def test_changed_prompt_or_response_rejected(self):
        self.run_item(QueueTransport([reply()] * 3))
        changed = copy.deepcopy(MESSAGES)
        changed[1]["text"] = "Changed text"
        with self.assertRaises(api.IntegrityError):
            self.run_item(never, changed)
        p = next(self.job().rglob("outcome.json"))
        p.write_text("{}", encoding="utf8")
        with self.assertRaises(api.IntegrityError):
            self.run_item(never)

    def test_extra_attempt_file_is_not_silently_ignored(self):
        self.run_item(QueueTransport([reply()] * 3))
        p = self.job() / "calls/unreviewed/complete.json"
        p.parent.mkdir()
        p.write_text("{}")
        with self.assertRaises(api.IntegrityError):
            self.run_item(never)

    def test_concurrent_workers_share_payloads_without_lost_responses(self):
        barrier = threading.Barrier(8)
        mutex = threading.Lock()
        counter = [0]

        def work(i):
            calls = [0]

            def transport(payload):
                if calls[0] == 0:
                    barrier.wait(timeout=20)
                calls[0] += 1
                with mutex:
                    counter[0] += 1
                return reply()

            identity = {**IDENTITY, "key": "en/concurrent" + str(i)}
            api.execute_item(self.root, identity, MESSAGES, transport, sleep=lambda _: None)
            return api.audit_item(self.root, identity, MESSAGES)

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(work, range(32)))
        self.assertEqual(counter[0], 96)
        self.assertEqual(len(results), 32)
        self.assertEqual(len(list(self.root.rglob("outcome.json"))), 96)
        self.assertEqual(len(list(self.root.glob("payloads/*.json"))), 1)

    def test_five_attempt_cap_on_each_of_nine_calls_is_45(self):
        t = QueueTransport(([reply(status=500)] * 4 + [reply("bad")]) * 9)
        out = self.run_item(t)
        self.assertIsNone(out["result"])
        self.assertEqual(out["accounting"]["post_attempts"], 45)
        self.assertEqual(out["accounting"]["retry_attempts"], 36)
        self.assertEqual(out["repairs"], 6)
        self.audit()

    def test_explicit_adapter_rejects_other_hosts_and_url_credentials(self):
        with self.assertRaises(api.IntegrityError):
            api.RequestsTransport("https://example.com/api/v1", "test")
        with self.assertRaises(api.IntegrityError):
            api.RequestsTransport("https://user:pw@openwebui.srv.peasec.de/api/v1", "test")

    def test_mock_results_cannot_be_mistaken_for_live_results(self):
        out = self.run_item(QueueTransport([reply()] * 3))
        self.assertEqual(out["execution_mode"], "mock")
        self.assertIsNone(out["active_model"])
        with self.assertRaises(api.IntegrityError):
            api.audit_item(self.root, self.identity, MESSAGES, execution_mode="live")
        transport = api.RequestsTransport("https://openwebui.srv.peasec.de/api/v1", "fixture")
        with self.assertRaises(api.IntegrityError):
            self.run_item(transport)

    def test_adapter_has_no_hidden_retries_or_redirects_and_redacts_exceptions(self):
        import requests

        transport = api.RequestsTransport("https://openwebui.srv.peasec.de/api/v1", "fixture-token")
        with patch.object(
            requests,
            "post",
            side_effect=requests.exceptions.ReadTimeout("secret URL fixture-token"),
        ) as post:
            with self.assertRaises(api.TransportFault) as caught:
                transport(api.body(MESSAGES))
        self.assertEqual(str(caught.exception), "ReadTimeout")
        self.assertEqual(post.call_count, 1)
        self.assertIs(post.call_args.kwargs["allow_redirects"], False)

    def test_wire_body_matches_original_provider_method_without_importing_client(self):
        from types import SimpleNamespace

        tree = ast.parse((api.ROOT / "src/hsrag/client.py").read_text(encoding="utf8"))
        cls = next(
            (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "PEASECClient")
        )
        tree.body = [
            next(
                (n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_build_body")
            )
        ]
        namespace = {}
        exec(compile(tree, "frozen_original_provider_method", "exec"), namespace)
        original = namespace["_build_body"](
            SimpleNamespace(temperature=1.0, max_tokens=4096), api.MODEL, MESSAGES
        )
        self.assertEqual(original, api.body(MESSAGES))


if __name__ == "__main__":
    unittest.main()

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import socket
import unittest
from unittest.mock import patch
from analysis.common import batch as b

a = b.a
ROOT = b.ROOT
BASE = b.BASE
STAGE = "stage_02_primary"
HERE = BASE / "recovery_stage02_20260917_v1"
PLAN = HERE / "plan.json"
SOURCE = ROOT / "analysis/common/recover_interruption.py"


def forbidden(*args, **kwargs):
    raise AssertionError("No network in interruption recovery")


def files(folder):
    return {
        b.rel(p): a.digest(p)
        for p in a.storage_path(folder).rglob("*")
        if p.is_file() and p.name != "item.lock"
    }


def resolution(intent, manifest_sha, reviewed_utc):
    return {
        "kind": "transport_error",
        "category": "LocalProcessInterruptedOutcomeUnknown",
        "remote_completion_uncertain": True,
        "intent_sha256": a.signature(intent),
        "finished_utc": reviewed_utc,
        "elapsed_s": 0.0,
        "administrative_recovery": {
            "manifest_sha256": manifest_sha,
            "observation": "The local Python process is no longer running and this durable intent has no saved outcome.",
            "request_delivery": "unknown: an intent precedes POST; no assertion that the server received or completed it",
            "http_response_observed": False,
            "network_exception_observed": False,
            "elapsed_s_is_unknown_zero_sentinel": True,
            "finished_utc_is_review_time_not_request_end": True,
            "latency_interpretation": "Exclude this zero sentinel from measured-duration claims. Any summed API duration is a lower bound excluding the unknown elapsed time.",
            "retry_decision": "Consume this existing attempt, preserve the same payload and logical call, and use only remaining original retry slots.",
            "selection": "All eight interrupted outcomes treated alike without looking at gold or outcome quality.",
        },
    }


class RecoveryTests(unittest.TestCase):

    def test_unknown_is_explicit_and_retryable(self):
        x = resolution({"vote": 2, "turn": 0, "attempt": 0}, "fixture", "fixture")
        self.assertEqual(
            a.interpret(x), ("retry", "LocalProcessInterruptedOutcomeUnknown", None)
        )
        self.assertTrue(
            x["administrative_recovery"]["elapsed_s_is_unknown_zero_sentinel"]
        )
        self.assertFalse(x["administrative_recovery"]["http_response_observed"])
        self.assertNotIn("body_base64", x)

    def test_existing_reply_and_remaining_budget_preserved(self):
        root = a.storage_path(HERE / "offline_test_fixture")
        self.assertFalse(
            root.exists(),
            "Preserve earlier fixture; use a new version for a failed test",
        )
        identity = {
            "experiment_sha256": "recovery_test_only",
            "condition": "mock",
            "repeat": 0,
            "key": "mock/input",
        }
        messages = [{"role": "user", "text": "OFFLINE RECOVERY FIXTURE"}]
        good = json.dumps(
            {
                "reasoning": "fixture",
                "hate": False,
                "target_group": [],
                "hate_type": [],
                "severity": None,
                "legal": [],
            }
        )
        calls = []

        def initial(payload):
            calls.append(payload)
            if len(calls) == 3:
                raise KeyboardInterrupt("simulate local process loss")
            text = good if len(calls) == 1 else "{}"
            return a.WireReply(
                200,
                json.dumps(
                    {
                        "model": a.MODEL,
                        "choices": [
                            {"finish_reason": "stop", "message": {"content": text}}
                        ],
                    }
                ).encode(),
                {},
            )

        with self.assertRaises(KeyboardInterrupt):
            a.execute_item(
                root,
                identity,
                messages,
                initial,
                sleep=lambda s: None,
                execution_mode="mock",
            )
        job = root / "items" / a.signature(identity)
        before = files(root)
        intent_path = job / "calls/vote_1_turn_1/attempt_0/intent.json"
        intent = a.read(intent_path)
        with self.assertRaises(a.IncompleteAttempt):
            a.execute_item(
                root,
                identity,
                messages,
                forbidden,
                sleep=lambda s: None,
                execution_mode="mock",
            )
        a.save_new(
            intent_path.with_name("outcome.json"),
            resolution(intent, "fixture", "fixture"),
        )
        resumed = []

        def finish(payload):
            resumed.append(payload)
            return a.WireReply(
                200,
                json.dumps(
                    {
                        "model": a.MODEL,
                        "choices": [
                            {"finish_reason": "stop", "message": {"content": good}}
                        ],
                    }
                ).encode(),
                {},
            )

        result = a.execute_item(
            root,
            identity,
            messages,
            finish,
            sleep=lambda s: None,
            execution_mode="mock",
        )
        self.assertEqual(len(resumed), 2)
        self.assertEqual(resumed[0], calls[-1])
        self.assertEqual(result["accounting"]["ordinary_logical_calls"], 3)
        self.assertEqual(result["accounting"]["repair_logical_calls"], 1)
        self.assertEqual(result["accounting"]["retry_attempts"], 1)
        self.assertEqual(result["accounting"]["post_attempts"], 5)
        self.assertEqual(result["n_valid"], 3)
        for name, h in before.items():
            self.assertEqual(a.digest(ROOT / name), h)
        a.audit_item(root, identity, messages, execution_mode="mock")
        a.execute_item(
            root,
            identity,
            messages,
            forbidden,
            sleep=lambda s: None,
            execution_mode="mock",
        )
        self.assertTrue((job / "calls/vote_1_turn_1/attempt_1/outcome.json").exists())
        with self.assertRaises(a.IntegrityError):
            a.save_new(
                intent_path.with_name("outcome.json"), {"overwrite": "forbidden"}
            )


def selftest():
    a.require(not (HERE / "tests.json").exists(), "Preserve completed recovery tests")
    output = io.StringIO()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecoveryTests)
    result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    a.save_new(
        HERE / "tests.json",
        {
            "passed": result.wasSuccessful(),
            "tests": result.testsRun,
            "output": output.getvalue(),
            "source_sha256": a.digest(SOURCE),
            "actual_API_calls": 0,
        },
    )
    print(output.getvalue(), flush=True)
    a.require(result.wasSuccessful(), "Recovery fixture failed; do not prepare/apply")


def snapshot():
    plan = b.validate()
    schedule = a.read(BASE / "schedule.json")
    stage = next((x for x in schedule["stages"] if x["id"] == STAGE))
    target = a.storage_path(BASE / "stages" / STAGE)
    a.require(
        not (target / "complete.json").exists()
        and (not list(target.glob("stop_*.json"))),
        "Unexpected completed/stopped stage",
    )
    attempts = list(target.glob("attempt_*"))
    a.require(len(attempts) == 1, "Unexpected stage history")
    a.require(
        (attempts[0] / "start.json").exists()
        and (not (attempts[0] / "summary.json").exists()),
        "Interrupted attempt history differs",
    )
    existing = files(target)
    pending = []
    partial = []
    not_started = []
    completed = 0
    cat = b.catalog()
    for number, job in enumerate(stage["jobs"], 1):
        d = b.item_dir(BASE / "live", job)
        if not d.exists():
            not_started.append(job)
            continue
        if (d / "complete.json").exists():
            receipt = a.read(d / "complete.json")
            a.verify_inventory(a.storage_path(BASE / "live"), d, receipt["hashes"])
            completed += 1
        else:
            partial.append(job)
            stamp = a.read(d / "job.json")
            a.require(
                stamp
                == {
                    "identity": b.identity(job),
                    "messages_sha256": a.signature(
                        cat[job["key"], job["condition"]]["messages"]
                    ),
                    "contract": a.CONTRACT,
                    "implementation": a.implementation(),
                    "execution_mode": "live",
                },
                "Partial job attribution differs",
            )
            a.require(
                not list(d.rglob("*.partial_*")),
                "Unpublished partial file needs separate review",
            )
            missing = []
            for p in d.glob("calls/*/attempt_*/intent.json"):
                intent = a.read(p)
                out = p.with_name("outcome.json")
                if out.exists():
                    record = a.read(out)
                    a.require(
                        record["intent_sha256"] == a.signature(intent),
                        "Saved response binding differs",
                    )
                    a.interpret(record)
                else:
                    a.require(
                        intent["attempt"] == 0,
                        "Recovery scope is the eight observed first attempts only",
                    )
                    missing.append(
                        {
                            "job": job,
                            "intent": b.rel(p),
                            "intent_sha256": a.digest(p),
                            "outcome": b.rel(out),
                        }
                    )
            a.require(
                len(missing) == 1, "Unexpected unresolved attempt count in partial item"
            )
            pending.extend(missing)
        existing.update(files(d))
        if number % 500 == 0:
            print(
                f"Checked saved stage scope through scheduled item {number}/2700",
                flush=True,
            )
    a.require(
        completed == 2418
        and len(partial) == len(pending) == 8
        and (len(not_started) == 274),
        "Observed interruption scope changed; inspect before proceeding",
    )
    return {
        "completed_items": completed,
        "partial_items": partial,
        "not_started_items": not_started,
        "missing_outcomes": pending,
        "existing_hashes": existing,
        "stage_file_hashes": files(target),
    }


def prepare():
    a.require(not PLAN.exists(), "Preserve frozen recovery plan")
    tests = a.read(HERE / "tests.json")
    a.require(
        tests["passed"] and tests["source_sha256"] == a.digest(SOURCE),
        "Matching recovery tests required",
    )
    with a.item_lock(a.storage_path(BASE / "launcher.lock")):
        observed = snapshot()
        record = {
            "created_utc": a.utc(),
            "status": "reviewed_offline_interruption_recovery_prepared",
            "stage": STAGE,
            "execution_plan_sha256": a.digest(b.PLAN),
            "source_sha256": a.digest(SOURCE),
            "tests_sha256": a.digest(HERE / "tests.json"),
            "process_review": "Read-only Win32_Process check at 2026-09-17T10:48:50.9555968Z found zero matching thesis Python runners; launcher lock is acquired for this snapshot.",
            "reason": "User accidentally closed the VS Code terminal window and requested recovery. Completed output count was stable at 2418.",
            "policy": "Preserve all files and saved votes; add explicit unknown-interruption records for all eight missing outcomes; consume attempt 0 and resume within the existing five-attempt cap. No labels or gold used for selection.",
            "timing_qualification": "elapsed_s=0 in these eight administrative records is an explicitly marked unknown-duration sentinel, never a measured zero. Summed request latency after recovery is a lower bound; remote usage/delivery/completion are unknown.",
            "actual_API_calls": 0,
            **observed,
        }
        a.save_new(PLAN, record)
    print(
        json.dumps(
            {
                "status": record["status"],
                "saved": 2418,
                "partial": 8,
                "not_started": 274,
                "existing_files": len(observed["existing_hashes"]),
                "plan_sha256": a.digest(PLAN),
                "actual_API_calls": 0,
            },
            indent=2,
        ),
        flush=True,
    )


def apply():
    a.require(
        not (HERE / "complete.json").exists(),
        "Recovery already applied; inspect receipt, do not repeat",
    )
    p = a.read(PLAN)
    a.require(
        p["source_sha256"] == a.digest(SOURCE)
        and p["tests_sha256"] == a.digest(HERE / "tests.json"),
        "Recovery implementation changed",
    )
    a.require(
        p["execution_plan_sha256"] == a.digest(b.PLAN), "Original execution changed"
    )
    with a.item_lock(a.storage_path(BASE / "launcher.lock")):
        current = snapshot()
        for key in current:
            a.require(
                current[key] == p[key], "Interrupted files/scope changed since review"
            )
        added = {}
        reviewed = a.utc()
        for item in p["missing_outcomes"]:
            destination = ROOT / item["outcome"]
            a.require(
                not a.storage_path(destination).exists(),
                "Do not replace an existing response",
            )
            intent = a.read(ROOT / item["intent"])
            a.require(
                a.digest(ROOT / item["intent"]) == item["intent_sha256"],
                "Intent changed",
            )
            record = resolution(intent, a.digest(PLAN), reviewed)
            a.save_new(destination, record)
            added[item["outcome"]] = a.digest(destination)
        for name, h in p["existing_hashes"].items():
            a.require(a.digest(ROOT / name) == h, "Pre-interruption evidence changed")
        a.save_new(
            HERE / "complete.json",
            {
                "created_utc": a.utc(),
                "status": "interruption_records_added_original_stage_command_can_resume",
                "recovery_plan_sha256": a.digest(PLAN),
                "execution_plan_sha256": a.digest(b.PLAN),
                "source_sha256": a.digest(SOURCE),
                "preexisting_files_verified_unchanged": len(p["existing_hashes"]),
                "completed_items_preserved": 2418,
                "administrative_unknown_outcomes_added": added,
                "remaining_items": 282,
                "actual_API_calls": 0,
                "important": "These eight records are explicit administrative acknowledgements of unknown request outcomes, not observed network exceptions or HTTP responses. Unknown request time is excluded from measured-latency claims. Original parser/prompt/voting and retry cap are unchanged.",
                "resume": "Use original launcher for stage_02_primary. Completed predictions are skipped and partial votes/replies reused. Future stage audit must handle two execution attempts and this recovery ledger; the old single-attempt-only stage audit is insufficient.",
            },
        )
    print(
        json.dumps(
            {
                "status": "ready_to_resume_stage_02_primary",
                "completed_preserved": 2418,
                "partial_to_finish": 8,
                "not_started": 274,
                "new_API_calls": 0,
                "receipt_sha256": a.digest(HERE / "complete.json"),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["selftest", "prepare", "apply"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ):
        {"selftest": selftest, "prepare": prepare, "apply": apply}[args.mode]()

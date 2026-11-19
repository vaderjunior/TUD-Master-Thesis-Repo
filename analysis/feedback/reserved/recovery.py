from analysis.common.source import replace_source, insert_before_assignment

import argparse
import ast
from contextlib import ExitStack
from datetime import datetime
import gc
import hashlib
import inspect
import io
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from analysis.feedback.reserved import run as original

a = original.a
c = original.c
ROOT = c.ROOT
BASE = original.BASE
HERE = BASE / "recovery_20261014_v1"
MANIFEST = HERE / "manifest.json"
APPLIED = HERE / "applied.json"
READY = HERE / "ready.json"
STAGE = "reserved_slot_04"
CATEGORY = "LocalProcessInterruptedOutcomeUnknown"
SOURCES = [
    "analysis/feedback/reserved/recovery.py",
    "tests/feedback/reserved/test_recovery.py",
    "scripts/resume_reserved_feedback_earlier.ps1",
]


def rel(path):
    return original.template.isolated.rel(path)


def own(path):
    p = a.storage_path(path).resolve()
    a.require(
        p.is_relative_to(a.storage_path(BASE).resolve()),
        "Recovery write outside A15 execution",
    )
    return p


def save(path, value):
    a.save_new(own(path), value)


def file_hashes(folder):
    return {
        rel(p): a.digest(p)
        for p in a.storage_path(folder).rglob("*")
        if p.is_file() and p.name not in ("item.lock", "launcher.lock")
    }


def resolution(intent, manifest_sha, reviewed_utc, count):
    return {
        "kind": "transport_error",
        "category": CATEGORY,
        "remote_completion_uncertain": True,
        "intent_sha256": a.signature(intent),
        "finished_utc": reviewed_utc,
        "elapsed_s": 0.0,
        "administrative_recovery": {
            "manifest_sha256": manifest_sha,
            "observation": "User-reported Windows shutdown; durable intent exists but no saved outcome or unpublished response file was found.",
            "request_delivery": "unknown; intent is saved before POST, so delivery and remote completion are not asserted",
            "http_response_observed": False,
            "network_exception_observed": False,
            "elapsed_s_is_unknown_zero_sentinel": True,
            "finished_utc_is_review_time_not_request_end": True,
            "latency_interpretation": "Exclude this sentinel from measured latency; stored summed durations omit this unknown request time.",
            "usage_interpretation": "No response or token usage is invented; observed usage excludes possible remote processing of this intent.",
            "retry_decision": "Consume the existing attempt and reuse the identical payload only within remaining original attempt_0..4 slots.",
            "selection": f"All {count} missing outcomes receive the same rule without using gold labels or answer quality.",
        },
    }


class Scope:

    def __init__(self, manifest, manifest_sha, applied, root):
        self.m = manifest
        self.sha = manifest_sha
        self.applied = applied
        self.root = a.storage_path(root).resolve()

    def verify_stops(self, target):
        target = a.storage_path(target).resolve()
        a.require(target.is_relative_to(self.root), "Stop check outside reviewed store")
        expected = {
            n: h
            for n, h in self.m["stop_files"].items()
            if a.storage_path(ROOT / n).resolve().parent == target
        }
        actual = {rel(p): a.digest(p) for p in target.glob("stop_*.json")}
        a.require(
            actual == expected,
            "New, missing or changed stop requires a separate review",
        )

    def history(self, target):
        self.verify_stops(target)
        return {
            a.storage_path(ROOT / n).relative_to(self.root).as_posix(): h
            for n, h in self.m["stop_files"].items()
            if a.storage_path(ROOT / n).parent == a.storage_path(target)
        }

    def verify_records(self):
        expected = {x["outcome"] for x in self.m["calls"]}
        a.require(
            set(self.applied["administrative_outcomes"]) == expected,
            "Recovery outcome roster changed",
        )
        for entry in self.m["calls"]:
            intent = a.read(ROOT / entry["intent"])
            a.require(
                a.digest(ROOT / entry["intent"]) == entry["intent_sha256"],
                "Reviewed intent changed",
            )
            out = ROOT / entry["outcome"]
            a.require(
                a.digest(out)
                == self.applied["administrative_outcomes"][entry["outcome"]],
                "Administrative outcome changed",
            )
            a.require(
                a.read(out)
                == resolution(
                    intent, self.sha, self.applied["reviewed_utc"], len(expected)
                ),
                "Unknown-outcome record differs",
            )

    def provenance(self):
        return {
            "manifest": self.m["manifest_path"],
            "manifest_sha256": self.sha,
            "supplemental_source_hashes": self.m["source_hashes"],
            "affected_stage": self.m["stage"],
            "unknown_request_outcomes": len(self.m["calls"]),
            "retry_cap_changed": False,
            "scope": "Reviewed shutdown intents only; original votes, payloads, source plan and retry cap preserved.",
            "timing": f"{len(self.m['calls'])} administrative zeros represent unknown request duration, not measured zero latency; per-stage count records the affected scope.",
        }

    def enrich_report(self, report, stage, hashes):
        self.verify_records()
        affected = stage["id"] == self.m["stage"]
        expected = len(self.m["calls"]) if affected else 0
        observed = report["http_statuses"].get("transport_error:" + CATEGORY, 0)
        a.require(
            observed == expected,
            "Unreviewed or omitted administrative interruption event",
        )
        if affected:
            for entry in self.m["calls"]:
                key = (
                    a.storage_path(ROOT / entry["outcome"])
                    .relative_to(self.root)
                    .as_posix()
                )
                a.require(
                    hashes[key]
                    == self.applied["administrative_outcomes"][entry["outcome"]],
                    "Recovery response absent from raw audit manifest",
                )
        report["reviewed_interruption_recovery"] = self.provenance()
        report["administrative_unknown_requests"] = expected
        report["request_duration_unknown_count"] = expected
        report["accounted_post_intents"] = report["accounting"]["post_attempts"]
        report["physical_delivery_qualification"] = (
            "Accounted attempts include durable intents with unknown delivery; do not call all of them confirmed server requests."
        )
        if affected:
            report["limits"].append(
                "Eight interrupted request outcomes and durations are unknown; observed token usage and recorded request time omit any unknown remote work. Zero administrative durations are not latency measurements."
            )

    def enrich_queue(self, result):
        self.verify_records()
        result["reviewed_interruption_recovery"] = self.provenance()
        result["administrative_unknown_requests"] = len(self.m["calls"])
        result["accounted_post_intents"] = result["accounting"]["post_attempts"]
        result["observed_usage_qualification"] = (
            "Saved-response usage excludes possible unknown remote work for the reviewed shutdown intents."
        )


def replace_once(text, before, after, expected=1):
    try:
        return replace_source(text, before, after, expected)
    except ValueError as error:
        a.require(False, str(error))


def wrapped_queue(scope):

    b = original.queue()
    source = inspect.getsource(b.execute_stage)
    source = replace_once(
        source,
        "a.require(not list(target.glob('stop_*.json')), 'Preserved execution stop requires review')",
        "recovery.verify_stops(target)",
        2,
    )
    ns = dict(b.execute_stage.__globals__)
    ns["recovery"] = scope
    exec(compile(source, __file__ + "[stage wrapper]", "exec"), ns)
    b.execute_stage = ns["execute_stage"]
    unmodified_review = b.verify_review

    def verified_review(stage, root, plan_sha, mode):
        scope.verify_stops(a.storage_path(root) / "stages" / stage["id"])
        record = unmodified_review(stage, root, plan_sha, mode)
        if stage["id"] == scope.m["stage"]:
            report = a.read(
                a.storage_path(root) / "stages" / stage["id"] / record["report"]
            )
            a.require(
                report["reviewed_interruption_recovery"] == scope.provenance(),
                "Recovery audit provenance changed",
            )
            scope.verify_records()
        return record

    b.verify_review = verified_review
    source = (ROOT / "analysis/common/vote_audit.py").read_text(encoding="utf8")
    source = replace_once(source, "from analysis.common import vote_api as b", "")
    source = replace_once(
        source,
        "a.require(not list(target.glob('stop_*.json')), 'Preserved execution stop blocks audit')",
        "recovery.verify_stops(target)",
    )
    source = replace_once(
        source,
        "starts=[]; history={}; model_checks=0",
        "starts=[]; history=recovery.history(target); model_checks=0",
    )
    source = insert_before_assignment(
        source, "audit_folder", "recovery.enrich_report(report,stage,hashes)"
    )
    source = replace_once(
        source,
        "    b.save(live_root/'complete.json',result)",
        "    recovery.enrich_queue(result)\n    b.save(live_root/'complete.json',result)",
    )
    ns = {
        "__name__": __name__ + "_audit",
        "__file__": __file__,
        "__package__": __package__,
        "b": b,
        "recovery": scope,
    }
    exec(compile(source, __file__ + "[audit wrapper]", "exec"), ns)
    b.audit = SimpleNamespace(**ns)
    return b


def snapshot():
    b = original.queue()
    b.validate(ready=True)
    root = a.storage_path(BASE / "live" / original.STUDY)
    schedule = a.read(BASE / "schedule.json")
    plan_sha = a.digest(b.PLAN)
    stage = next((s for s in schedule["stages"] if s["id"] == STAGE))
    jobs = {
        a.signature(b.identity(j, plan_sha)): j
        for s in schedule["stages"]
        for j in s["jobs"]
    }
    target = root / "stages" / STAGE
    a.require(
        not (target / "complete.json").exists(), "Interrupted stage already complete"
    )
    stops = {rel(p): a.digest(p) for p in target.glob("stop_*.json")}
    a.require(len(stops) == 1, "Unexpected stop history; inspect again")
    for name in stops:
        stop = a.read(ROOT / name)
        a.require(
            stop["stage"] == STAGE
            and stop["plan_sha256"] == plan_sha
            and (stop["error_type"] == "IncompleteAttempt")
            and (
                stop["reason"] == "Saved intent without outcome: stop without resending"
            ),
            "Different stop reason",
        )
    preserved = file_hashes(target)
    completed = {}
    partial = []
    count_in_stage = 0
    stage_ids = {a.signature(b.identity(j, plan_sha)) for j in stage["jobs"]}
    for ent in os.scandir(root / "items"):
        if not ent.is_dir():
            continue
        a.require(ent.name in jobs, "Unexpected live prediction directory")
        p = Path(ent.path)
        a.require(
            (p / "job.json").exists(), "Job-less directory requires separate review"
        )
        stamp = a.read(p / "job.json")
        a.require(
            stamp["identity"] == b.identity(jobs[ent.name], plan_sha),
            "Saved job identity changed",
        )
        if (p / "complete.json").exists():
            record = a.read(p / "complete.json")
            a.require(
                record["status"] == "complete"
                and record["job_stamp"] == a.signature(stamp),
                "Completed stamp differs",
            )
            a.require(
                record["hashes"][
                    str((p / "job.json").relative_to(root)).replace("\\", "/")
                ]
                == a.digest(p / "job.json"),
                "Completed job binding differs",
            )
            completed[rel(p / "complete.json")] = a.digest(p / "complete.json")
            count_in_stage += ent.name in stage_ids
            if len(completed) % 5000 == 0:
                print(
                    f"Checked {len(completed):,} completed prediction receipts",
                    flush=True,
                )
        else:
            a.require(
                ent.name in stage_ids, "Incomplete item outside interrupted stage"
            )
            partial.append((p, jobs[ent.name], stamp))
    a.require(
        len(completed) == 31184 and count_in_stage == 2546 and (len(partial) == 8),
        "Observed interruption partition changed",
    )
    cat = b.load_catalog(original.STUDY)
    engines = b.engines(cat)
    calls = []
    saved_votes = 0
    for p, j, stamp in partial:
        entry = cat[j["key"], j["condition"]]
        engine = engines[j["schema_path"]]
        a.require(
            stamp
            == {
                "identity": b.identity(j, plan_sha),
                "messages_sha256": entry["messages_sha256"],
                "contract": a.CONTRACT,
                "implementation": engine.implementation(),
                "execution_mode": "live",
            },
            "Partial job attribution differs",
        )
        a.require(
            not list(p.rglob("*.partial_*")),
            "Unpublished file needs separate salvage review",
        )
        preserved.update(file_hashes(p))
        missing = []
        for ip in p.glob("calls/*/attempt_*/intent.json"):
            intent = a.read(ip)
            out = ip.with_name("outcome.json")
            a.require(
                intent["job_stamp"] == a.signature(stamp),
                "Partial request belongs to another job",
            )
            payload = root / "payloads" / f"{intent['body_sha256']}.json"
            a.require(
                a.digest(payload) == intent["body_sha256"]
                and a.read(payload) == a.body(entry["messages"]),
                "Unexpected original payload/repair",
            )
            preserved[rel(payload)] = a.digest(payload)
            if out.exists():
                outcome = a.read(out)
                a.require(
                    outcome["intent_sha256"] == a.signature(intent)
                    and a.interpret(outcome)[0] == "ok",
                    "Unexpected saved response state",
                )
            else:
                a.require(
                    intent["attempt"] == 0
                    and intent["turn"] == 0
                    and (intent["vote"] in (1, 2)),
                    "Unknown intent is outside reviewed first-attempt scope",
                )
                missing.append(
                    {
                        "job": j,
                        "item": rel(p),
                        "intent": rel(ip),
                        "intent_sha256": a.digest(ip),
                        "outcome": rel(out),
                        "vote": intent["vote"],
                        "turn": intent["turn"],
                        "attempt": intent["attempt"],
                    }
                )
        a.require(
            len(missing) == 1,
            "Expected exactly one missing outcome per interrupted item",
        )
        saved_votes += len(list((p / "votes").glob("*.json")))
        calls.extend(missing)
        a.require(
            all((q.with_name("intent.json").exists() for q in p.rglob("outcome.json"))),
            "Orphan saved outcome",
        )
    a.require(saved_votes == 13 and len(calls) == 8, "Partial saved-vote scope changed")
    stage_receipts = {}
    for s in schedule["stages"][:3]:
        review = b.verify_review(s, root, plan_sha, "live")
        folder = root / "stages" / s["id"]
        for name in ["complete.json", "review.json", *review["artifacts"]]:
            stage_receipts[rel(folder / name)] = a.digest(folder / name)
    del cat
    gc.collect()
    return {
        "plan_sha256": plan_sha,
        "stage": STAGE,
        "store": rel(root),
        "stop_files": stops,
        "calls": sorted(calls, key=lambda x: x["intent"]),
        "preserved_files": preserved,
        "completed_receipt_hashes": completed,
        "audited_stage_files": stage_receipts,
        "completed_items": 31184,
        "stage_completed_items": 2546,
        "partial_items": 8,
        "saved_partial_votes": 13,
        "stage_unstarted_items": 6770,
        "total_remaining_items": 118666,
        "no_unpublished_partial_files_in_interrupted_items": True,
    }


def selftest():
    from tests.feedback.reserved import test_recovery as tests

    buffer = io.StringIO()
    result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(tests.RecoveryTests)
    )
    print(buffer.getvalue(), flush=True)
    receipt = {
        "passed": result.wasSuccessful(),
        "tests": result.testsRun,
        "output": buffer.getvalue(),
        "source_hashes": {n: a.digest(ROOT / n) for n in SOURCES},
        "actual_API_calls": 0,
        "created_utc": a.utc(),
    }
    save(
        HERE / "development" / ("tests_" + str(__import__("time").time_ns()) + ".json"),
        receipt,
    )
    a.require(result.wasSuccessful(), "Offline recovery tests failed")
    save(HERE / "tests.json", receipt)


def prepare():
    a.require(
        not a.storage_path(MANIFEST).exists(),
        "Recovery already prepared; preserve its manifest",
    )
    tests = a.read(HERE / "tests.json")
    a.require(tests["passed"], "Passing recovery tests required")
    for n, h in tests["source_hashes"].items():
        a.require(a.digest(ROOT / n) == h, "Recovery source changed since tests")
    with ExitStack() as locks:
        locks.enter_context(
            a.item_lock(a.storage_path(BASE / "combined_launcher.lock"))
        )
        locks.enter_context(
            a.item_lock(
                a.storage_path(BASE / "live" / original.STUDY / "launcher.lock")
            )
        )
        observed = snapshot()
        record = {
            "status": "reviewed_shutdown_recovery",
            "created_utc": a.utc(),
            "manifest_path": rel(MANIFEST),
            "source_hashes": tests["source_hashes"],
            "tests_sha256": a.digest(HERE / "tests.json"),
            "reason": "User reported Windows turned off during A15; attempted resume stopped before new model access on an unresolved durable intent.",
            "process_review": "Read-only Win32_Process inspection found no matching A15 Python runner; both launcher OS locks acquired for preparation.",
            "policy": "All eight incomplete ordinary requests consume attempt 0; preserve all saved votes and use only remaining original attempts. No changes to scientific settings.",
            "actual_API_calls": 0,
            **observed,
        }
        save(MANIFEST, record)
    print(
        json.dumps(
            {
                k: record[k]
                for k in [
                    "status",
                    "completed_items",
                    "partial_items",
                    "saved_partial_votes",
                    "stage_unstarted_items",
                    "total_remaining_items",
                ]
            },
            indent=2,
        )
    )


def verify_bindings(m, *, prior_receipts=True):
    a.require(
        m["status"] == "reviewed_shutdown_recovery"
        and m["stage"] == STAGE
        and (len(m["calls"]) == 8),
        "Wrong recovery scope",
    )
    a.require(
        a.digest(BASE / "plan.json") == m["plan_sha256"],
        "Original execution plan changed",
    )
    a.require(
        a.digest(HERE / "tests.json") == m["tests_sha256"]
        and a.read(HERE / "tests.json")["passed"],
        "Recovery tests changed",
    )
    groups = ["source_hashes", "preserved_files", "audited_stage_files"]
    if prior_receipts:
        groups.append("completed_receipt_hashes")
    for group in groups:
        for name, h in m[group].items():
            a.require(a.digest(ROOT / name) == h, "Preserved artifact changed: " + name)


def apply():
    a.require(
        not a.storage_path(APPLIED).exists(),
        "Recovery already applied; use validate, never reapply",
    )
    m = a.read(MANIFEST)
    verify_bindings(m)
    manifest_sha = a.digest(MANIFEST)
    with ExitStack() as locks:
        locks.enter_context(
            a.item_lock(a.storage_path(BASE / "combined_launcher.lock"))
        )
        locks.enter_context(
            a.item_lock(a.storage_path(ROOT / m["store"] / "launcher.lock"))
        )
        now = snapshot()
        a.require(
            all((now[k] == m[k] for k in now)),
            "Live interruption evidence changed since preparation",
        )
        reviewed = a.utc()
        added = {}
        for entry in m["calls"]:
            locks.enter_context(
                a.item_lock(a.storage_path(ROOT / entry["item"] / "item.lock"))
            )
            p = ROOT / entry["outcome"]
            a.require(
                not a.storage_path(p).exists(),
                "Existing outcome must never be replaced",
            )
            save(
                p,
                resolution(
                    a.read(ROOT / entry["intent"]),
                    manifest_sha,
                    reviewed,
                    len(m["calls"]),
                ),
            )
            added[entry["outcome"]] = a.digest(p)
        verify_bindings(m)
        save(
            APPLIED,
            {
                "status": "administrative_outcomes_added",
                "manifest_sha256": manifest_sha,
                "reviewed_utc": reviewed,
                "administrative_outcomes": added,
                "completed_items_preserved": m["completed_items"],
                "saved_partial_votes_preserved": m["saved_partial_votes"],
                "actual_API_calls": 0,
                "request_delivery_completion_usage_duration": "unknown for the eight reviewed intents",
            },
        )
    print(
        "Eight explicitly unknown administrative outcomes added; no model request made.",
        flush=True,
    )


def scope_from_disk(*, ready=True):
    original.queue().validate(ready=True)
    m = a.read(MANIFEST)
    verify_bindings(m)
    applied = a.read(APPLIED)
    a.require(
        applied["manifest_sha256"] == a.digest(MANIFEST),
        "Applied receipt attribution changed",
    )
    scope = Scope(m, a.digest(MANIFEST), applied, ROOT / m["store"])
    scope.verify_records()
    scope.verify_stops(a.storage_path(ROOT / m["store"]) / "stages" / STAGE)
    if ready:
        r = a.read(READY)
        a.require(
            r["status"] == "offline_verified_ready_for_user"
            and r["manifest_sha256"] == a.digest(MANIFEST)
            and (r["applied_sha256"] == a.digest(APPLIED)),
            "Recovery readiness changed",
        )
    return scope


def check():
    scope = scope_from_disk(ready=False)
    b = wrapped_queue(scope)
    schedule = a.read(BASE / "schedule.json")
    stage = next((s for s in schedule["stages"] if s["id"] == STAGE))
    root = ROOT / scope.m["store"]
    cat = b.load_catalog(original.STUDY)
    existing, todo = b.scan_jobs(
        stage["jobs"], cat, root, scope.m["plan_sha256"], "live", b.engines(cat)
    )
    a.require(
        len(existing) == 2546 and len(todo) == 6778,
        "Unexpected resumable stage partition",
    )
    for s in schedule["stages"][:3]:
        b.verify_review(s, root, scope.m["plan_sha256"], "live")
    save(
        READY,
        {
            "status": "offline_verified_ready_for_user",
            "created_utc": a.utc(),
            "manifest_sha256": a.digest(MANIFEST),
            "applied_sha256": a.digest(APPLIED),
            "stage_completed_predictions_replayed": len(existing),
            "stage_pending_items": len(todo),
            "preserved_prior_completed_receipts": scope.m["completed_items"],
            "reviewed_unknown_outcomes": 8,
            "original_sources_and_settings_unchanged": True,
            "actual_API_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": "ready_to_resume",
                "completed_total": 31184,
                "stage04_verified": 2546,
                "stage04_pending": 6778,
                "unknown_requests_logged": 8,
                "new_API_calls": 0,
            },
            indent=2,
        ),
        flush=True,
    )


def run_all():
    with a.item_lock(a.storage_path(BASE / "combined_launcher.lock")):
        original.resource_guard(live=True)
        scope = scope_from_disk()
        b = wrapped_queue(scope)
        plan_sha = a.digest(b.PLAN)
        schedule = a.read(BASE / "schedule.json")
        cat = b.load_catalog(original.STUDY)
        root = BASE / "live" / original.STUDY

        def run(stage):
            b.validate(ready=True)
            a.require(
                original.shutil.disk_usage(ROOT).free / 2**30 >= 4,
                "Disk below 4 GiB; preserve outputs",
            )
            b.execute_stage(
                stage,
                cat,
                root,
                plan_sha,
                mode="live",
                client_factory=original.core.live_client,
            )

        b.run_sequence(
            schedule["stages"],
            run,
            lambda stage: b.audit.audit_stage(stage, cat, root, plan_sha, mode="live"),
            lambda stage: (
                a.storage_path(root) / "stages" / stage["id"] / "review.json"
            ).exists(),
            lambda stage: b.verify_review(stage, root, plan_sha, "live"),
        )
        del cat
        gc.collect()
        b.validate(ready=True)
        b.audit.finalize_queue(schedule, BASE / "live", plan_sha, mode="live")
    print(
        "ALL SIXTEEN A15 BATCHES AND AUDITS COMPLETE, with the reviewed shutdown ledger. Tell Astra to inspect.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["selftest", "prepare", "apply", "check", "validate", "run-all"]
    )
    args = parser.parse_args()
    if args.mode == "run-all":
        run_all()
        return
    with c.offline(), patch.object(original.core, "live_client", c.forbidden):
        if args.mode == "validate":
            scope_from_disk()
            print(
                "A15 reviewed shutdown recovery and original frozen plan verified. No API calls."
            )
        else:
            globals()[args.mode]()


if __name__ == "__main__":
    main()

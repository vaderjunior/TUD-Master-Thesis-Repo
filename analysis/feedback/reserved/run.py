import argparse
import ast
import gc
import shutil
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from analysis.feedback.reserved import common as c
from analysis.common import api as a
from analysis.common import batch as core
from analysis.common import vote_api as template
from analysis.feedback.insertion import run as binding
from analysis.feedback.reserved import validate as checker

BASE = c.BASE / "execution"
STUDY = "reserved_slot"
COPIED = binding.COPIED
SOURCE_FILES = list(
    dict.fromkeys(
        [
            *binding.SOURCE_FILES,
            "analysis/feedback/reserved/common.py",
            "analysis/feedback/reserved/prepare.py",
            "analysis/feedback/reserved/validate.py",
            "analysis/feedback/reserved/run.py",
            "tests/feedback/summary/test_reserved.py",
            "scripts/run_reserved_feedback.ps1",
        ]
    )
)


def queue():
    ns = {
        **vars(template),
        "BASE": BASE,
        "PLAN": BASE / "plan.json",
        "WORKERS": 8,
        "SOURCE_FILES": SOURCE_FILES,
        "STUDIES": {
            STUDY: {
                "root": c.BASE,
                "inputs": c.N_INPUTS,
                "conditions": c.N_CONDITIONS,
                "batches": c.N_BATCHES,
                "predictions": c.N_PREDICTIONS,
            }
        },
    }
    tree = ast.parse(Path(template.__file__).read_text(encoding="utf8"))
    funcs = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in COPIED
    ]
    c.require(
        {n.name for n in funcs} == set(COPIED), "Reviewed runner function missing"
    )
    exec(
        compile(ast.Module(body=funcs, type_ignores=[]), template.__file__, "exec"), ns
    )
    b = SimpleNamespace(
        **{
            k: ns[k]
            for k in [
                "ROOT",
                "BASE",
                "PLAN",
                "WORKERS",
                "SOURCE_FILES",
                "STUDIES",
                "a",
                "core",
                "isolated",
                *COPIED,
            ]
        }
    )
    b.audit = binding.rebound_module(
        c.ROOT / "analysis/common/vote_audit.py", {"b": b}, {"b"}
    )
    return b


def resource_guard(*, live=False):
    import ctypes

    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (k, ctypes.c_ulonglong)
            for k in (
                "totalPhys",
                "availPhys",
                "totalPage",
                "availPage",
                "totalVirtual",
                "availVirtual",
                "availExtended",
            )
        ]

    state = Memory()
    state.length = ctypes.sizeof(state)
    if hasattr(ctypes, "windll"):
        c.require(
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(state)) != 0,
            "Memory status unavailable",
        )
        available = state.availPhys / 2**30
        c.require(
            available >= 2.5,
            f"A15 needs at least 2.5 GiB available RAM before catalog loading; available {available:.2f}. Close unused apps, preserve run terminals.",
        )
    else:
        available = None
    free = shutil.disk_usage(c.ROOT).free / 2**30
    c.require(
        free >= (20 if live else 3), f"Insufficient disk for A15: {free:.2f} GiB free"
    )
    print(
        {
            "available_RAM_GiB": available,
            "disk_free_GiB": round(free, 2),
            "CUDA_required": False,
        },
        flush=True,
    )


def test():
    from tests.feedback.summary import test_reserved as tests

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests)
    )
    c.require(result.wasSuccessful(), "A15 tests failed")
    b = queue()
    b.save(
        b.test_receipt(),
        {
            "passed": True,
            "tests": result.testsRun,
            "source_hashes": {n: c.sha(c.ROOT / n) for n in SOURCE_FILES},
            "environment": core.environment(),
            "actual_API_calls": 0,
        },
    )


def prepare():
    b = queue()
    if b.PLAN.exists():
        b.validate()
        print("Existing A15 execution plan verified.")
        return
    c.require(not (BASE / "live").exists(), "Preserve live execution")
    tested = a.read(b.test_receipt())
    c.require(
        tested["passed"] and tested["environment"] == core.environment(),
        "Current tests required",
    )
    c.verify(tested["source_hashes"])
    inputs = b.study_inputs(STUDY)
    cat = b.load_catalog(STUDY)
    schedule = b.make_schedule({STUDY: cat})
    schedule["ordering"] = (
        "Sixteen automatically audited batches of 42/43 inputs; every condition/repeat for each input stays together; deterministic hashed input order and rotated condition/repeat order."
    )
    checker.check_schedule(schedule, {STUDY: cat}, b)
    b.save(BASE / "schedule.json", schedule)
    for path in (BASE / "schedule.json", b.test_receipt()):
        b.bind(inputs, c.rel(path), c.sha(path))
    b.save(
        b.PLAN,
        {
            "created_utc": c.utc(),
            "study": STUDY,
            "status": "frozen_requires_offline_rehearsal",
            "input_hashes": inputs,
            "source_hashes": {n: c.sha(c.ROOT / n) for n in SOURCE_FILES},
            "environment": core.environment(),
            "api_contract": a.CONTRACT,
            "workers": 8,
            "budget": {
                "stages": c.N_BATCHES,
                "prompts": c.N_PROMPTS,
                "predictions": c.N_PREDICTIONS,
                "ordinary_calls": c.N_CALLS,
                "max_logical_calls": c.N_CALLS * 3,
                "max_transport_POST_attempts": c.N_CALLS * 15,
            },
            "derivation": {
                "runner": c.rel(Path(template.__file__)),
                "unchanged_functions": COPIED,
                "auditor": "analysis/common/vote_audit.py",
            },
            "no_accuracy_based_stopping": True,
            "scope": "A15 only. No new retrieval, training, original-response reuse or automatic A16/A17 execution.",
        },
    )
    print(
        "A15 schedule frozen: 149,850 predictions / 449,550 ordinary calls.", flush=True
    )


def rehearse():
    b = queue()
    if (BASE / "ready.json").exists():
        b.validate(ready=True)
        print("Existing A15 readiness verified.")
        return
    b.validate()
    c.require(not (BASE / "live").exists(), "Readiness cannot rewrite live execution")
    cat = b.load_catalog(STUDY)
    schedule = a.read(BASE / "schedule.json")
    plan_sha = a.digest(b.PLAN)
    matrix = checker.check_schedule(schedule, {STUDY: cat}, b)
    rehearsal = checker.rehearse(b, schedule, {STUDY: cat}, plan_sha)
    b.validate()
    path = BASE / "development/readiness.json"
    b.save(
        path,
        {
            "status": "complete_matrix_and_restart_audit_verified",
            "matrix": matrix,
            "rehearsal": rehearsal,
            "plan_sha256": plan_sha,
            "actual_API_calls": 0,
            "environment": core.environment(),
        },
    )
    b.save(
        BASE / "ready.json",
        {
            "status": "offline_verified_ready_for_user",
            "plan_sha256": plan_sha,
            "evidence": c.rel(path),
            "evidence_sha256": c.sha(path),
            "actual_API_calls": 0,
        },
    )
    b.validate(ready=True)
    print("A15 READY. No live Qwen or CUDA calls made.", flush=True)


def run_all():
    b = queue()
    with a.item_lock(a.storage_path(BASE / "combined_launcher.lock")):
        resource_guard(live=True)
        b.validate(ready=True)
        plan_sha = a.digest(b.PLAN)
        schedule = a.read(BASE / "schedule.json")
        cat = b.load_catalog(STUDY)
        root = BASE / "live" / STUDY

        def run(stage):
            b.validate(ready=True)
            c.require(
                shutil.disk_usage(c.ROOT).free / 2**30 >= 4,
                "Disk below 4 GiB. Preserve completed work and free space before continuing.",
            )
            b.execute_stage(
                stage, cat, root, plan_sha, mode="live", client_factory=core.live_client
            )

        b.run_sequence(
            schedule["stages"],
            run,
            lambda stage: b.audit.audit_stage(stage, cat, root, plan_sha, mode="live"),
            lambda stage: (root / "stages" / stage["id"] / "review.json").exists(),
            lambda stage: b.verify_review(stage, root, plan_sha, "live"),
        )
        del cat
        gc.collect()
        b.validate(ready=True)
        b.audit.finalize_queue(schedule, BASE / "live", plan_sha, mode="live")
    print(
        "ALL SIXTEEN A15 RESERVED-SLOT BATCHES AND AUDITS COMPLETE. Tell Astra to inspect. No next study started.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "resources",
            "test",
            "prepare",
            "rehearse",
            "validate",
            "run-all",
            "status",
        ],
    )
    args = parser.parse_args()
    if args.mode == "run-all":
        run_all()
        return
    with c.offline(), patch.object(core, "live_client", c.forbidden):
        if args.mode == "resources":
            resource_guard()
        elif args.mode == "test":
            test()
        elif args.mode == "prepare":
            prepare()
        elif args.mode == "rehearse":
            rehearse()
        elif args.mode == "validate":
            queue().validate(ready=True)
            print("A15 frozen inputs and execution readiness verified.")
        else:
            print(
                {
                    "preparation_complete": (c.BASE / "complete.json").exists(),
                    "execution_ready": (BASE / "ready.json").exists(),
                    "execution_complete": (BASE / "live/complete.json").exists(),
                }
            )


if __name__ == "__main__":
    main()

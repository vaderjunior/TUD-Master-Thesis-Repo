import argparse
import ast
from pathlib import Path
import socket
import time
from types import SimpleNamespace
from unittest.mock import patch
from analysis.feedback.insertion import common as c
from analysis.common import api as a
from analysis.common import batch as core
from analysis.common import vote_api as template

BASE = c.BASE / "execution"
STUDY = "five_slot"
COPIED = [
    "rel",
    "save",
    "forbidden",
    "bind",
    "verify",
    "test_receipt",
    "engines",
    "identity",
    "item_dir",
    "scan_jobs",
    "availability",
    "checked_root",
    "execute_stage",
    "verify_review",
    "run_sequence",
    "study_inputs",
    "load_catalog",
    "make_schedule",
    "validate",
]
SOURCE_FILES = list(
    dict.fromkeys(
        [
            *core.SOURCE_FILES,
            "analysis/common/schema.py",
            "analysis/common/prompt_archive.py",
            "analysis/common/vote_api.py",
            "analysis/common/vote_audit.py",
            "analysis/feedback/balanced/validate.py",
            "analysis/feedback/insertion/common.py",
            "analysis/feedback/insertion/prepare.py",
            "analysis/feedback/insertion/validate.py",
            "analysis/feedback/insertion/run.py",
            "analysis/feedback/insertion/scores_earlier.py",
            "tests/feedback/summary/test_five_slot.py",
            "scripts/run_correction_insertion_earlier.ps1",
        ]
    )
)


def rebound_module(path, injected, aliases):
    tree = ast.parse(path.read_text(encoding="utf8"))
    tree.body = [
        n
        for n in tree.body
        if not (
            isinstance(n, ast.ImportFrom)
            and any((x.asname in aliases for x in n.names))
        )
    ]
    namespace = {
        "__name__": __name__ + "_bound_" + path.stem,
        "__package__": __package__,
        **injected,
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


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
                "inputs": 675,
                "conditions": 19,
                "batches": 4,
                "predictions": 38475,
            }
        },
    }
    tree = ast.parse(Path(template.__file__).read_text(encoding="utf8"))
    functions = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in COPIED
    ]
    c.require(
        {n.name for n in functions} == set(COPIED),
        "Reviewed execution functions missing",
    )
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), template.__file__, "exec"),
        ns,
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
    b.audit = rebound_module(c.ROOT / "analysis/common/vote_audit.py", {"b": b}, {"b"})
    b.check = rebound_module(
        c.ROOT / "analysis/feedback/balanced/validate.py",
        {"b": b, "audit": b.audit},
        {"b", "audit"},
    )
    return b


def test_receipt():
    import unittest
    from tests.feedback.summary import test_five_slot as tests

    b = queue()
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests)
    )
    c.require(result.wasSuccessful(), "A14 tests failed")
    hashes = {n: c.sha(c.ROOT / n) for n in SOURCE_FILES}
    b.save(
        b.test_receipt(),
        {
            "passed": True,
            "tests": result.testsRun,
            "environment": core.environment(),
            "source_hashes": hashes,
            "actual_API_calls": 0,
        },
    )


def prepare():
    b = queue()
    if b.PLAN.exists():
        b.validate()
        print("Existing A14 execution plan verified.")
        return
    c.require(not (BASE / "live").exists(), "Preserve live execution")
    tested = a.read(b.test_receipt())
    c.require(
        tested["passed"] and tested["environment"] == core.environment(),
        "Current passing tests required",
    )
    c.verify(tested["source_hashes"])
    inputs = b.study_inputs(STUDY)
    cat = b.load_catalog(STUDY)
    schedule = b.make_schedule({STUDY: cat})
    schedule["ordering"] = (
        "Four A14 batches of 169/169/169/168 inputs. Stable hashed input order using the reviewed scheduler; rotating conditions/repeats; all comparisons for a key stay together."
    )
    b.check.check_schedule(schedule, {STUDY: cat})
    b.save(BASE / "schedule.json", schedule)
    for path in [BASE / "schedule.json", b.test_receipt()]:
        b.bind(inputs, c.rel(path), c.sha(path))
    b.save(
        b.PLAN,
        {
            "created_utc": c.utc(),
            "study": STUDY,
            "status": "frozen_offline_rehearsal_required",
            "input_hashes": inputs,
            "source_hashes": {n: c.sha(c.ROOT / n) for n in SOURCE_FILES},
            "environment": core.environment(),
            "api_contract": a.CONTRACT,
            "workers": 8,
            "budget": {
                "stages": 4,
                "prompts": 12825,
                "predictions": 38475,
                "ordinary_calls": 115425,
                "max_logical_calls": 346275,
                "max_transport_POST_attempts": 1731375,
            },
            "derivation": {
                "template": c.rel(Path(template.__file__)),
                "unchanged_functions": COPIED,
                "auditor": "analysis/common/vote_audit.py",
                "checker": "analysis/feedback/balanced/validate.py",
                "only_change": "Owned queue namespace; functions unchanged",
            },
            "scope": "Fresh Qwen outputs for frozen five-example contexts. No retrieval/training/GPU or changing old inputs during live calls. No outcome-based stopping.",
        },
    )
    print("A14 execution plan frozen; synthetic rehearsal required.", flush=True)


def rehearse():
    b = queue()
    if (BASE / "ready.json").exists():
        b.validate(ready=True)
        print("A14 readiness already verified; preserved live work may resume.")
        return
    b.check.main()


def pilot_rehearsal():

    from tests.feedback.summary.test_five_slot import synthetic_catalog

    b = queue()
    path = BASE / "development/synthetic_queue_pilot.json"
    if path.exists():
        receipt = c.read(path)
        c.verify(receipt["source_hashes"])
        c.require(
            c.sha(c.ROOT / receipt["rehearsal"]["root"] / "complete.json")
            == receipt["rehearsal"]["completion_sha256"],
            "Pilot completion changed",
        )
        print("Synthetic queue pilot already verified.")
        return
    cat = synthetic_catalog()
    schedule = b.make_schedule({STUDY: cat})
    matrix = b.check.check_schedule(schedule, {STUDY: cat})
    rehearsal = b.check.rehearse(schedule, {STUDY: cat}, "A14-synthetic-queue-pilot")
    b.save(
        path,
        {
            "status": "synthetic_queue_pilot_passed",
            "matrix": matrix,
            "rehearsal": rehearsal,
            "source_hashes": {n: c.sha(c.ROOT / n) for n in SOURCE_FILES},
            "environment": core.environment(),
            "new_model_calls": 0,
        },
    )


def run_all():
    b = queue()
    with a.item_lock(a.storage_path(BASE / "combined_launcher.lock")):
        b.validate(ready=True)
        plan_sha = a.digest(b.PLAN)
        schedule = a.read(BASE / "schedule.json")
        cat = b.load_catalog(STUDY)

        def root(stage):
            return BASE / "live" / stage["study"]

        def run_one(stage):
            b.validate(ready=True)
            b.execute_stage(
                stage,
                cat,
                root(stage),
                plan_sha,
                mode="live",
                client_factory=core.live_client,
            )

        b.run_sequence(
            schedule["stages"],
            run_one,
            lambda stage: b.audit.audit_stage(
                stage, cat, root(stage), plan_sha, mode="live"
            ),
            lambda stage: (
                root(stage) / "stages" / stage["id"] / "review.json"
            ).exists(),
            lambda stage: b.verify_review(stage, root(stage), plan_sha, "live"),
        )
        b.validate(ready=True)
        b.audit.finalize_queue(schedule, BASE / "live", plan_sha, mode="live")
    print(
        "ALL FOUR FIVE-EXAMPLE CORRECTION BATCHES AND AUDITS COMPLETE. Tell Astra to inspect the results.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "test",
            "prepare",
            "rehearse",
            "pilot-rehearse",
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
        if args.mode == "test":
            test_receipt()
        elif args.mode == "prepare":
            prepare()
        elif args.mode == "rehearse":
            rehearse()
        elif args.mode == "pilot-rehearse":
            pilot_rehearsal()
        elif args.mode == "status":
            print(
                {
                    "preparation_complete": (c.BASE / "complete.json").exists(),
                    "execution_ready": (BASE / "ready.json").exists(),
                    "live_complete": (BASE / "live/complete.json").exists(),
                }
            )
        else:
            queue().validate(ready=True)
            print("A14 frozen inputs and live-readiness receipt verified.")


if __name__ == "__main__":
    main()

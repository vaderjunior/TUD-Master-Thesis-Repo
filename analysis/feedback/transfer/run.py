import argparse
import unittest
from unittest.mock import patch
from analysis.common import runtime as c
from analysis.feedback.common import queue as q

SOURCES = [
    "analysis/feedback/transfer/prepare.py",
    "analysis/feedback/transfer/run.py",
    "tests/feedback/common/test_queue.py",
    "scripts/run_feedback_transfer.ps1",
]


def queue():
    return q.queue(c.TRANSFER, "transfer", SOURCES)


def prepare():
    c.resources(2.5, 3.0)
    b = queue()
    if b.PLAN.exists():
        b.validate()
        q.rehearse(b)
        return
    from tests.feedback.common import test_queue as tests

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests)
    )
    c.require(result.wasSuccessful(), "Follow-up queue tests failed")
    record = {
        "passed": True,
        "tests": result.testsRun,
        "source_hashes": {n: c.sha(c.ROOT / n) for n in b.SOURCE_FILES},
        "environment": b.core.environment(),
        "actual_API_calls": 0,
    }
    q.prepare(b, record)
    q.rehearse(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "validate", "run-all", "status"])
    args = ap.parse_args()
    if args.mode == "run-all":
        b = queue()
        q.require_a15_finished()
        with b.a.item_lock(
            b.a.storage_path(c.SHARED / "qwen_queue.lock")
        ), b.a.item_lock(b.a.storage_path(b.BASE / "launcher.lock")):
            q.execute(b)
        print(
            "A16 BROADER TRANSFER AND ALL AUTOMATIC AUDITS COMPLETE. Tell Astra to review. No next study started.",
            flush=True,
        )
        return
    with c.offline():
        if args.mode == "prepare":
            prepare()
        elif args.mode == "validate":
            queue().validate(ready=True)
            print("A16 readiness and frozen bindings verified.")
        else:
            print(
                {
                    "offline_ready": (c.TRANSFER / "execution/ready.json").exists(),
                    "live_complete": (
                        c.TRANSFER / "execution/live/complete.json"
                    ).exists(),
                    "wait_for_A15": not (
                        c.A15 / "execution/live/complete.json"
                    ).exists(),
                }
            )


if __name__ == "__main__":
    main()

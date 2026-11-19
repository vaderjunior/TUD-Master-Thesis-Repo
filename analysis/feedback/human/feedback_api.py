import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch
from analysis.feedback.human import feedback_common as c
from analysis.feedback.human import feedback_score as scorer
from analysis.common import vote_api as template
from analysis.feedback.insertion import run as binding
from analysis.feedback.common import queue as generic
from analysis.feedback.balanced.validate import SyntheticTransport

STUDY = "human_feedback"
SOURCE_FILES = list(
    dict.fromkeys(
        [
            *binding.SOURCE_FILES,
            "analysis/common/runtime.py",
            "analysis/feedback/common/queue.py",
            "analysis/feedback/human/feedback_common.py",
            "analysis/feedback/human/feedback_prepare.py",
            "analysis/feedback/human/validate_feedback.py",
            "analysis/feedback/human/feedback_score.py",
            "analysis/feedback/human/feedback_api.py",
            "tests/feedback/human/test_feedback.py",
            "scripts/run_human_feedback_earlier.ps1",
            "analysis/feedback/human/feedback_api.py",
            "scripts/run_human_feedback.ps1",
        ]
    )
)


def queue():
    base = c.BASE / "execution_v2"
    ns = {
        **vars(template),
        "BASE": base,
        "PLAN": base / "plan.json",
        "WORKERS": 8,
        "SOURCE_FILES": SOURCE_FILES,
        "STUDIES": {
            STUDY: {
                "root": c.BASE,
                "inputs": 24,
                "conditions": 3,
                "batches": 1,
                "predictions": 216,
            }
        },
    }
    tree = ast.parse(Path(template.__file__).read_text(encoding="utf8"))
    funcs = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name in binding.COPIED
    ]
    c.require(
        {n.name for n in funcs} == set(binding.COPIED),
        "Reviewed engine function missing",
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
                *binding.COPIED,
            ]
        }
    )
    b.audit = binding.rebound_module(
        c.ROOT / "analysis/common/vote_audit.py", {"b": b}, {"b"}
    )
    return b


def test():
    from tests.feedback.human import test_feedback as tests

    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    c.require(result.wasSuccessful(), "Human-feedback tests failed")
    with patch.object(
        generic, "require_a15_finished", side_effect=RuntimeError("still running")
    ), patch.object(template.core, "live_client", c.forbidden):
        try:
            run_all()
        except RuntimeError as exc:
            c.require(str(exc) == "still running", "Wrong launch guard failure")
        else:
            raise AssertionError("Live launch did not refuse an incomplete queue")
    b = queue()
    record = {
        "passed": True,
        "tests": result.testsRun,
        "incomplete_queue_guard": "passed before lock/client",
        "source_hashes": c.bindings((c.ROOT / n for n in SOURCE_FILES)),
        "environment": b.core.environment(),
        "actual_API_calls": 0,
    }
    b.save(b.test_receipt(), record)
    return record


def extract(b, root, mode):

    done = c.read(root / "complete.json")
    plan_sha = c.sha(b.PLAN)
    c.require(
        done["status"] == "all_studies_executed_and_audited"
        and done["mode"] == mode
        and (done["plan_sha256"] == plan_sha),
        "Complete audited queue required",
    )
    c.verify(done["review_hashes"])
    schedule = c.read(b.BASE / "schedule.json")
    cat = b.load_catalog(STUDY)
    backends = b.engines(cat)
    out = {}
    dependencies = {c.rel(root / "complete.json"): c.sha(root / "complete.json")}
    for stage in schedule["stages"]:
        b.verify_review(stage, root / STUDY, plan_sha, mode)
        for job in stage["jobs"]:
            entry = cat[job["key"], job["condition"]]
            backend = backends[job["schema_path"]]
            backend.audit_item(
                root / STUDY,
                b.identity(job, plan_sha),
                entry["messages"],
                execution_mode=mode,
            )
            path = b.item_dir(root / STUDY, job, plan_sha) / "result.json"
            v = c.read(path)
            identity = (job["key"], job["condition"], job["repeat"])
            c.require(identity not in out, "Duplicate result identity")
            independent, tie = b.audit.independent_vote(v["run_predictions"])
            c.require(
                tie == v["uncertain"]
                and (
                    independent is None
                    if v["result"] is None
                    else independent == {k: v["result"][k] for k in independent}
                ),
                "Independent aggregation differs",
            )
            out[identity] = {"result": v["result"], "n_valid": v["n_valid"]}
            dependencies[b.rel(path)] = b.a.digest(path)
    c.require(len(out) == 216, "Human feedback result matrix incomplete")
    return (out, dependencies, done)


def rehearse():
    b = queue()
    b.validate()
    if (b.BASE / "ready.json").exists():
        b.validate(ready=True)
        print("Existing human-feedback readiness verified.")
        return
    schedule = c.read(b.BASE / "schedule.json")
    cat = b.load_catalog(STUDY)
    plan_sha = c.sha(b.PLAN)
    matrix = generic.validate_schedule(b, schedule, {STUDY: cat})
    root = b.BASE / "offline_check" / ("full_rehearsal_" + str(time.time_ns()))
    transport = SyntheticTransport()
    for stage in schedule["stages"]:
        b.execute_stage(
            stage,
            cat,
            root / STUDY,
            plan_sha,
            mode="mock",
            client_factory=lambda: transport,
            sleep=lambda _: None,
        )
        b.audit.audit_stage(stage, cat, root / STUDY, plan_sha, mode="mock")
    first = b.audit.finalize_queue(schedule, root, plan_sha, mode="mock")
    before = transport.count
    for stage in schedule["stages"]:
        b.execute_stage(
            stage, cat, root / STUDY, plan_sha, mode="mock", client_factory=c.forbidden
        )
        b.verify_review(stage, root / STUDY, plan_sha, "mock")
    second = b.audit.finalize_queue(schedule, root, plan_sha, mode="mock")
    c.require(
        first == second and before == transport.count == 648,
        "Restart repeated or lost requests",
    )
    predictions, dependencies, _ = extract(b, root, "mock")
    analyzed = scorer.analyze(list(c.rows(c.BASE / "roster.jsonl")), predictions)
    analyzed["status"] = "synthetic_pipeline_check_only_NOT_scientific_results"
    b.save(root / "synthetic_analysis.json", analyzed)
    path = b.BASE / "development/readiness.json"
    b.save(
        path,
        {
            "status": "all_216_predictions_mocked_audited_resumed_and_scored",
            "matrix": matrix,
            "mock_predictions": 216,
            "mock_replies": before,
            "new_calls_on_restart": 0,
            "actual_API_calls": 0,
            "rehearsal_root": c.rel(root),
            "completion_sha256": c.sha(root / "complete.json"),
            "synthetic_analysis_sha256": c.sha(root / "synthetic_analysis.json"),
            "plan_sha256": plan_sha,
            "audited_result_bindings": dependencies,
            "environment": b.core.environment(),
        },
    )
    b.save(
        b.BASE / "ready.json",
        {
            "status": "offline_verified_ready_for_user",
            "plan_sha256": plan_sha,
            "evidence": c.rel(path),
            "evidence_sha256": c.sha(path),
            "actual_API_calls": 0,
        },
    )
    b.validate(ready=True)
    print(
        "Human-feedback preparation, full synthetic queue, raw audit, zero-call restart and scoring passed. No real Qwen calls."
    )


def run_all():
    generic.require_a15_finished()
    b = queue()
    b.validate(ready=True)
    with b.a.item_lock(
        b.a.storage_path(c.shared.SHARED / "qwen_queue.lock")
    ), b.a.item_lock(b.a.storage_path(b.BASE / "launcher.lock")):
        generic.execute(b)
    print(
        "HUMAN FEEDBACK CALLS AND AUTOMATIC AUDIT COMPLETE. Independent scientific review remains. No next study started."
    )


def analyze_live():
    b = queue()
    b.validate(ready=True)
    predictions, deps, done = extract(b, b.BASE / "live", "live")
    report = scorer.analyze(list(c.rows(c.BASE / "roster.jsonl")), predictions)
    report["execution_accounting"] = {
        k: done.get(k)
        for k in ("counts", "accounting", "observed_usage", "stages", "review_hashes")
    }
    report["input_hashes"] = deps
    c.save(c.BASE / "results/descriptive_analysis.json", report)
    print(
        "Frozen descriptive analysis saved. Review all cases, raw audit and limitations before publishing findings."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "mode",
        choices=["test", "prepare", "rehearse", "validate", "run-all", "analyze"],
    )
    args = p.parse_args()
    if args.mode == "run-all":
        run_all()
        return
    with c.offline(), patch.object(template.core, "live_client", c.forbidden):
        if args.mode == "test":
            test()
        elif args.mode == "prepare":
            b = queue()
            generic.prepare(b, c.read(b.test_receipt()))
        elif args.mode == "rehearse":
            rehearse()
        elif args.mode == "analyze":
            analyze_live()
        else:
            queue().validate(ready=True)
            print(
                "Human-feedback frozen sources and offline readiness verified. Zero Qwen calls."
            )


if __name__ == "__main__":
    main()

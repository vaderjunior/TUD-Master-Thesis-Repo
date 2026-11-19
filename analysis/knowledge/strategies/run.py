"""Strategies run."""

import argparse
import base64
from collections import Counter
import json
import os
from pathlib import Path
import socket
import time
from unittest.mock import patch
from analysis.common import batch as core
from analysis.knowledge.strategies import prepare as retrieval

a = core.a
ROOT = a.ROOT
STUDY = retrieval.BASE
BASE = STUDY / "api_execution_v1"
PLAN = BASE / "plan.json"
STRATEGIES = ("dense", "bm25", "hybrid")
ENDPOINT_ORDER = (
    "en_dev_eval_sq3_types",
    "en_dev_eval_main",
    "en_dev_eval_targets",
    "en_dev_eval_severity",
    "de_dev_eval",
    "de_legal_dev_eval",
    "en_dev_eval_types",
)
WORKERS = 8
CAP = 2700
RETRIEVAL_PLAN_SHA = "d22c3018572515d19943c598422ee91a8c1fe42af631d85efa17dcf7eacb6ad1"
RETRIEVAL_AUDIT_SHA = "25befd0770a23e031a6b1b816328973e49fe3aeed665e18b7817afb8fdac7023"
PROMPTS_SHA = "691189131b734178a21b15294f764623b96212d6b5ac9d7f758402175775ad7d"
SOURCE_FILES = [
    "analysis/knowledge/strategies/run.py",
    "tests/knowledge/strategies/test_run.py",
    "analysis/knowledge/strategies/validate_execution.py",
    "scripts/run_retrieval_strategies.ps1",
    "analysis/common/batch.py",
    "analysis/common/scoring.py",
    "tests/common/test_api.py",
    "tests/common/test_batch.py",
    "analysis/knowledge/strategies/prepare.py",
    "analysis/common/retrieval.py",
    "analysis/common/prompt.py",
    *a.DEPENDENCIES,
]


def rel(path):
    return core.rel(path)


def save(path, value):
    path = a.storage_path(path).resolve()
    a.require(
        path.is_relative_to(a.storage_path(BASE).resolve()),
        "Write outside new strategy API experiment",
    )
    a.save_new(path, value)


def frozen_catalog(path):
    entries = core.rows(path)
    catalogue = {}
    for entry in entries:
        key = (entry["key"], entry["condition"])
        a.require(key not in catalogue, "Duplicate prompt identity")
        a.require(
            entry["strategy"] in STRATEGIES
            and entry["condition"] == "rag_" + entry["strategy"],
            "Prompt strategy differs",
        )
        a.require(
            entry["plan_sha256"] == RETRIEVAL_PLAN_SHA and entry["pool"] == "full",
            "Prompt provenance differs",
        )
        a.require(
            a.signature(entry["messages"]) == entry["messages_sha256"],
            "Frozen message signature mismatch",
        )
        a.body(entry["messages"])
        catalogue[key] = {
            "messages": entry["messages"],
            "messages_sha256": entry["messages_sha256"],
            "strategy": entry["strategy"],
            "subset": entry["scopes"][0]["subset"],
        }
    a.require(len(entries) == len(catalogue) == 3885, "Wrong strategy catalogue size")
    return catalogue


def make_schedule(dev, catalogue, cap=CAP):
    """Round-robin rosters; keep each rotated nine-prediction input unit intact."""
    a.require(
        type(cap) is int and cap >= 9, "Stage cap cannot hold one input comparison"
    )
    by_subset = {name: [] for name in ENDPOINT_ORDER}
    a.require(len({row["key"] for row in dev}) == len(dev), "Duplicate DEV identity")
    for row in dev:
        a.require(
            len(row["scopes"]) == 1 and row["scopes"][0]["subset"] in by_subset,
            "Unexpected input scope",
        )
        by_subset[row["scopes"][0]["subset"]].append(row["key"])
    expected = {
        (row["key"], "rag_" + strategy) for row in dev for strategy in STRATEGIES
    }
    a.require(set(catalogue) == expected, "Missing or extra strategy prompt")
    stages, pending = ([], [])

    def finish():
        nonlocal pending
        if not pending:
            return
        number = len(stages) + 1
        stages.append(
            {
                "id": f"stage_{number:02d}",
                "jobs": pending,
                "items": len(pending),
                "inputs": len(pending) // 9,
                "ordinary_responses": len(pending) * 3,
                "max_logical_calls": len(pending) * 9,
                "max_physical_POSTs": len(pending) * 45,
                "endpoint_inputs": dict(
                    Counter((job["subset"] for job in pending[::9]))
                ),
            }
        )
        pending = []

    ordinal = 0
    for index in range(max(map(len, by_subset.values()), default=0)):
        for subset in ENDPOINT_ORDER:
            if index >= len(by_subset[subset]):
                continue
            key = by_subset[subset][index]
            unit = []
            for offset in range(3):
                repeat = (ordinal + offset) % 3
                for position in range(3):
                    strategy = STRATEGIES[(ordinal + repeat + position) % 3]
                    condition = "rag_" + strategy
                    entry = catalogue[key, condition]
                    a.require(
                        entry["subset"] == subset and entry["strategy"] == strategy,
                        "Catalogue scope/strategy differs",
                    )
                    unit.append(
                        {
                            "key": key,
                            "subset": subset,
                            "condition": condition,
                            "strategy": strategy,
                            "repeat": repeat,
                            "messages_sha256": entry["messages_sha256"],
                        }
                    )
            if len(pending) + 9 > cap:
                finish()
            pending.extend(unit)
            ordinal += 1
    finish()
    jobs = [j for stage in stages for j in stage["jobs"]]
    ids = [(j["key"], j["condition"], j["repeat"]) for j in jobs]
    a.require(
        len(ids) == len(set(ids)) == len(catalogue) * 3,
        "Duplicate or omitted schedule prediction",
    )
    return {
        "stages": stages,
        "endpoint_round_robin_order": list(ENDPOINT_ORDER),
        "ordering": "Round-robin DEV rosters in saved row order. Input ordinal i: repeats (i+offset)%3; within repeat r, strategies[(i+r+position)%3]. Nine predictions for one input remain adjacent and in one stage. Submission order is fixed; asynchronous completion order can differ.",
    }


def analysis_plan(study_plan):
    original = study_plan["endpoints_inherited_masks_and_labels"]
    endpoints = {
        name: {
            **row,
            "inherited_core_role": row["role"],
            "role": study_plan["analysis_roles"][name],
        }
        for name, row in original.items()
    }
    return {
        "retrieval_plan_sha256": RETRIEVAL_PLAN_SHA,
        "endpoints": endpoints,
        "analysis": study_plan["analysis"],
        "conditions": ["rag_" + x for x in STRATEGIES],
        "repeats": [0, 1, 2],
        "focal_endpoint": "en_dev_eval_sq3_types",
        "focal_family_size": 3,
        "other_endpoint_contrasts": 18,
        "validity_mask": "One shared mask per endpoint, valid in all nine strategy/repeat settings; retain nulls, coverage and full-roster failure bounds.",
        "fixed_rules": {
            "gate_mapping": "strict",
            "minimum_full_roster_support": 10,
            "missing_annotations": "None is unannotated, [] observed empty; legal independent of gate.",
            "replicate_reporting": "Three individual repeat F1 scores, mean and sample SD. No pooled-vote F1.",
            "bootstrap": "10000 paired item draws seed20260920, fixed inherited labels. Three focal contrasts: 98.333...% family intervals plus95%; other18 descriptive95%.",
            "no_partial_scoring": True,
        },
        "limits": "Post-selection DEV extension. Retrieval is frozen to first-pass prompts; decoder repeats do not quantify observed ANN replay variation. Historical core hybrid is a separate descriptive reference, never extra repeats. No automatic strategy promotion.",
    }


def prepare():
    a.require(not PLAN.exists(), "API plan already frozen; use validate")
    tests_path = BASE / "development/tests_passed.json"
    tests = a.read(tests_path)
    a.require(
        tests["passed"] is True and tests["environment"] == core.environment(),
        "Passing execution tests required before freeze",
    )
    for name, digest in tests["source_hashes"].items():
        a.require(
            a.digest(ROOT / name) == digest, "Tested source changed before freeze"
        )
    acceptance_path = STUDY / "retrieval_acceptance_v1.json"
    acceptance = a.read(acceptance_path)
    a.require(
        acceptance["status"] == "accepted_first_pass_with_replay_qualification",
        "Accepted retrieval review required",
    )
    a.require(
        a.digest(STUDY / "plan.json")
        == acceptance["plan_sha256"]
        == RETRIEVAL_PLAN_SHA,
        "Retrieval plan changed",
    )
    a.require(
        a.digest(ROOT / acceptance["evidence"])
        == acceptance["evidence_sha256"]
        == RETRIEVAL_AUDIT_SHA,
        "Retrieval evidence changed",
    )
    a.require(
        a.digest(ROOT / acceptance["prompts"])
        == acceptance["prompts_sha256"]
        == PROMPTS_SHA,
        "First-pass prompts changed",
    )
    a.require(
        a.digest(ROOT / acceptance["report"]) == acceptance["report_sha256"],
        "Accepted retrieval report changed",
    )
    study_plan = a.read(STUDY / "plan.json")
    for name, digest in study_plan["input_hashes"].items():
        a.require(
            a.digest(ROOT / name) == digest, "Frozen retrieval input changed: " + name
        )
    prior_api = a.read(retrieval.CORE / "api_execution_v1/plan.json")
    a.require(
        prior_api["api_contract"] == a.CONTRACT,
        "Completed core transport contract differs",
    )
    for name in a.DEPENDENCIES:
        a.require(
            a.digest(ROOT / name) == prior_api["source_hashes"][name],
            "Reviewed transport/parser/vote source changed",
        )
    cat = frozen_catalog(ROOT / acceptance["prompts"])
    dev = core.rows(retrieval.CORE / "dev_inputs.jsonl")
    retrieval.roster_check(dev, study_plan["endpoints_inherited_masks_and_labels"])
    schedule = make_schedule(dev, cat)
    a.require(
        [x["items"] for x in schedule["stages"]] == [2700, 2700, 2700, 2700, 855],
        "Unexpected stage budget",
    )
    analysis = analysis_plan(study_plan)
    save(BASE / "schedule.json", schedule)
    save(BASE / "analysis_plan.json", analysis)
    paths = [
        tests_path,
        STUDY / "plan.json",
        STUDY / "PROTOCOL.md",
        acceptance_path,
        STUDY / "retrieval_review_v1.json",
        ROOT / acceptance["evidence"],
        ROOT / acceptance["report"],
        ROOT / acceptance["prompts"],
        STUDY / "runs/prepare/complete.json",
        STUDY / "runs/replay/complete.json",
        retrieval.CORE / "dev_inputs.jsonl",
        retrieval.CORE / "api_execution_v1/plan.json",
        retrieval.CORE / "api_execution_v1/ready.json",
        retrieval.CORE / "api_execution_v1/scoring_plan.json",
        ROOT
        / "Thesis_Log/astra_result/evidence/core_completion_summary_20260920_v1.json",
    ]
    for subset in ENDPOINT_ORDER:
        path = ROOT / f"data/processed/{subset}.parquet"
        a.require(
            a.digest(path) == prior_api["input_hashes"][rel(path)],
            "Inherited DEV file changed",
        )
        paths.append(path)
    plan = {
        "created_utc": a.utc(),
        "status": "prepared_offline_check_pending",
        "retrieval_plan_sha256": RETRIEVAL_PLAN_SHA,
        "retrieval_audit_sha256": RETRIEVAL_AUDIT_SHA,
        "prompts": acceptance["prompts"],
        "prompts_sha256": PROMPTS_SHA,
        "input_hashes": {rel(p): a.digest(p) for p in paths},
        "source_hashes": {name: a.digest(ROOT / name) for name in SOURCE_FILES},
        "generated_hashes": {
            rel(BASE / n): a.digest(BASE / n)
            for n in ("schedule.json", "analysis_plan.json")
        },
        "environment": core.environment(),
        "api_contract": a.CONTRACT,
        "workers": WORKERS,
        "programme": {
            "stages": 5,
            "distinct_inputs": 1295,
            "strategy_prompts": 3885,
            "predictions": 11655,
            "ordinary_responses": 34965,
            "max_logical_calls": 104895,
            "max_physical_POSTs": 524475,
        },
        "stage_cap_predictions": CAP,
        "model_check": "Before each incomplete live stage attempt, save GET /models and require the pinned model. Reuse the completed core transport/parser/voter; no extra sampled pilot or extra predictions. No provider fallback.",
        "resume": "Check all existing selected items before any new service call. Completed items are semantically audited and skipped. Partial saved replies are reused; unresolved intents, partial writes, fatal/exhausted calls or sticky stage stops require review. Never reset retry limits.",
        "stop": "One named stage. Eight workers. Stop new submissions on any error or >=10% null items after20 completed; drain in-flight jobs, preserve sticky stop. Never stop/select using F1.",
        "next_stage": "Requires a separately saved stage-review receipt bound to previous completion and evidence; no automatic next stage.",
        "scope": "Fresh classification of all three strategies from preserved first-pass prompts. No retrieval, training, TEST/benchmark execution, Priority3/adaptation, LaTeX or old core writes.",
    }
    save(PLAN, plan)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "plan_sha256": a.digest(PLAN),
                "programme": plan["programme"],
                "stages": [
                    {k: v for k, v in x.items() if k != "jobs"}
                    for x in schedule["stages"]
                ],
            },
            indent=2,
        ),
        flush=True,
    )


def validate():
    plan = a.read(PLAN)
    a.require(
        plan["environment"] == core.environment(), "API library environment changed"
    )
    a.require(
        plan["api_contract"] == a.CONTRACT
        and plan["workers"] == WORKERS
        and (plan["stage_cap_predictions"] == CAP),
        "Execution contract differs",
    )
    a.require(
        plan["prompts_sha256"] == PROMPTS_SHA
        and plan["retrieval_plan_sha256"] == RETRIEVAL_PLAN_SHA,
        "Input contract differs",
    )
    for group in ("input_hashes", "source_hashes", "generated_hashes"):
        for name, digest in plan[group].items():
            a.require(
                a.digest(ROOT / name) == digest, "Frozen API artifact changed: " + name
            )
    return plan


def identity(job, plan_sha):
    return {
        "experiment_sha256": plan_sha,
        "condition": job["condition"],
        "repeat": job["repeat"],
        "key": job["key"],
    }


def item_dir(root, job, plan_sha):
    return a.storage_path(root) / "items" / a.signature(identity(job, plan_sha))


def scan_jobs(jobs, cat, root, plan_sha, mode):
    existing, todo = ([], [])
    implementation = a.implementation()
    for job in jobs:
        entry = cat[job["key"], job["condition"]]
        a.require(
            entry["messages_sha256"] == job["messages_sha256"]
            and entry["strategy"] == job["strategy"]
            and (entry["subset"] == job["subset"]),
            "Scheduled prompt identity differs",
        )
        ident = identity(job, plan_sha)
        folder = item_dir(root, job, plan_sha)
        if (folder / "complete.json").exists():
            a.audit_item(root, ident, entry["messages"], execution_mode=mode)
            existing.append((job, a.read(folder / "result.json")))
            continue
        if folder.exists():
            a.require(
                not list(folder.rglob("*.partial_*")),
                "Partial atomic write requires review",
            )
            stamp = {
                "identity": ident,
                "messages_sha256": entry["messages_sha256"],
                "contract": a.CONTRACT,
                "implementation": implementation,
                "execution_mode": mode,
            }
            a.require(
                a.read(folder / "job.json") == stamp, "Partial item identity changed"
            )
            for path in folder.rglob("intent.json"):
                outcome_path = path.with_name("outcome.json")
                if not outcome_path.exists():
                    raise a.IncompleteAttempt(
                        "Saved request intent has no outcome; review before sending anything new"
                    )
                intent, outcome = (a.read(path), a.read(outcome_path))
                a.require(
                    intent["job_stamp"] == a.signature(stamp)
                    and outcome["intent_sha256"] == a.signature(intent),
                    "Partial request binding differs",
                )
                action, reason, _ = a.interpret(outcome)
                if action == "stop" or (action == "retry" and intent["attempt"] == 4):
                    raise a.CallStopped(
                        reason
                        if action == "stop"
                        else "transport_attempt_budget_exhausted"
                    )
            a.require(
                all(
                    (
                        p.with_name("intent.json").exists()
                        for p in folder.rglob("outcome.json")
                    )
                ),
                "Orphan response requires review",
            )
        todo.append(job)
    return (existing, todo)


def run_jobs(
    jobs,
    cat,
    root,
    attempt,
    plan_sha,
    *,
    client_factory,
    mode,
    existing=None,
    todo=None,
):
    import threading

    if existing is None or todo is None:
        existing, todo = scan_jobs(jobs, cat, root, plan_sha, mode)
    initial = {
        "complete": len(existing),
        "null": sum((out["result"] is None for _, out in existing)),
    }
    a.require(
        not (
            initial["complete"] >= 20 and initial["null"] / initial["complete"] >= 0.1
        ),
        "Existing null rate requires review",
    )
    local = threading.local()

    def work(job):
        if not hasattr(local, "client"):
            local.client = client_factory()
        return a.execute_item(
            root,
            identity(job, plan_sha),
            cat[job["key"], job["condition"]]["messages"],
            local.client,
            execution_mode=mode,
        )

    def progress(job, result, stats):
        if stats["complete"] % 50 == 0:
            print(
                f"Saved {stats['complete']}/{len(jobs)} predictions; {stats['null']} null results",
                flush=True,
            )

    result = core.bounded_jobs(todo, work, WORKERS, initial, progress)
    finished = existing + result.pop("results")
    result.update(
        saved_items=len(finished),
        null_items=sum((out["result"] is None for _, out in finished)),
        finished_utc=a.utc(),
        completed_item_accounting={
            field: sum((out["accounting"][field] for _, out in finished))
            for field in (
                "ordinary_logical_calls",
                "repair_logical_calls",
                "post_attempts",
                "retry_attempts",
            )
        },
    )
    save(attempt / "summary.json", result)
    return result


def live_availability(attempt, client):
    import requests

    save(
        attempt / "model_check_intent.json",
        {"started_utc": a.utc(), "action": "GET PEASEC /models", "model": a.MODEL},
    )
    tick = time.perf_counter()
    try:
        response = requests.get(
            client._endpoint.removesuffix("/chat/completions") + "/models",
            headers={"Authorization": "Bearer " + client._token},
            timeout=30,
            allow_redirects=False,
        )
    except requests.exceptions.RequestException as exc:
        save(
            attempt / "model_check_error.json",
            {"error_type": type(exc).__name__, "elapsed_s": time.perf_counter() - tick},
        )
        raise a.CallStopped(
            "Model roster request failed; saved category only"
        ) from None
    save(
        attempt / "model_check_response.json",
        {
            "status": response.status_code,
            "elapsed_s": time.perf_counter() - tick,
            "body_base64": base64.b64encode(response.content).decode("ascii"),
            "body_sha256": __import__("hashlib").sha256(response.content).hexdigest(),
        },
    )
    a.require(
        response.status_code == 200, "Model roster refused; no classification sent"
    )
    try:
        data = response.json()
        models = {m["id"] for m in data["data"]}
    except (ValueError, KeyError, TypeError):
        raise a.CallStopped("Malformed model roster; saved for review") from None
    a.require(
        a.MODEL in models, "Pinned Qwen unavailable; no fallback or classification"
    )


def check_stage_complete(stage, cat, root, plan_sha, mode):
    target = a.storage_path(root) / "stages" / stage["id"]
    record = a.read(target / "complete.json")
    a.require(
        record["plan_sha256"] == plan_sha
        and record["stage"] == stage["id"]
        and (record["execution_mode"] == mode),
        "Completed stage attribution differs",
    )
    expected = {
        rel(item_dir(root, j, plan_sha) / "complete.json") for j in stage["jobs"]
    }
    a.require(
        set(record["item_receipt_hashes"]) == expected,
        "Completed stage item roster differs",
    )
    for name, digest in record["item_receipt_hashes"].items():
        a.require(a.digest(ROOT / name) == digest, "Completed item receipt changed")
    existing, todo = scan_jobs(stage["jobs"], cat, root, plan_sha, mode)
    a.require(
        not todo and len(existing) == stage["items"], "Completed stage incomplete"
    )
    summary_path = ROOT / record["attempt"] / "summary.json"
    a.require(
        a.digest(summary_path) == record["summary_sha256"]
        and a.read(summary_path) == record["summary"],
        "Stage summary binding differs",
    )
    return record


def execute_stage(
    stage, cat, run_root, plan_sha, *, client_factory, mode, availability=None
):
    root = a.storage_path(run_root).resolve()
    a.require(mode in ("mock", "live"), "Explicit stage mode required")
    allowed = a.storage_path(
        BASE / ("live" if mode == "live" else "offline_check")
    ).resolve()
    a.require(
        root == allowed if mode == "live" else root.is_relative_to(allowed),
        "Stage output/mode boundary",
    )
    target = root / "stages" / stage["id"]
    with a.item_lock(root / "launcher.lock"):
        if (target / "complete.json").exists():
            check_stage_complete(stage, cat, root, plan_sha, mode)
            print(
                "THIS STAGE IS ALREADY COMPLETE. No network request sent; no next stage started.",
                flush=True,
            )
            return
        a.require(
            not list(target.glob("stop_*.json")),
            "Stage has a preserved stop; ask Astra to review before resuming",
        )
        attempt = target / f"attempt_{time.time_ns()}"
        save(
            attempt / "start.json",
            {
                "plan_sha256": plan_sha,
                "stage": stage["id"],
                "execution_mode": mode,
                "started_utc": a.utc(),
            },
        )
        try:
            existing, todo = scan_jobs(stage["jobs"], cat, root, plan_sha, mode)
            print(
                f"STAGE {stage['id']}: {stage['items']} predictions / {stage['ordinary_responses']} ordinary responses. Verified {len(existing)} already completed; {len(todo)} pending.",
                flush=True,
            )
            if todo and mode == "live":
                a.require(
                    availability is not None, "Live model availability check required"
                )
                availability(attempt, client_factory())
            result = run_jobs(
                stage["jobs"],
                cat,
                root,
                attempt,
                plan_sha,
                client_factory=client_factory,
                mode=mode,
                existing=existing,
                todo=todo,
            )
            a.require(
                not result["stopped"] and result["saved_items"] == stage["items"],
                "Stage stopped; preserve every attempt for review",
            )
        except Exception as exc:
            save(
                target / f"stop_{time.time_ns()}.json",
                {
                    "plan_sha256": plan_sha,
                    "stage": stage["id"],
                    "attempt": rel(attempt),
                    "error_type": type(exc).__name__,
                    "reason": (
                        str(exc)
                        if isinstance(
                            exc, (a.IntegrityError, a.CallStopped, a.IncompleteAttempt)
                        )
                        else "Inspect saved attempt; detail omitted to protect credentials"
                    ),
                },
            )
            raise
        hashes = {
            rel(item_dir(root, j, plan_sha) / "complete.json"): a.digest(
                item_dir(root, j, plan_sha) / "complete.json"
            )
            for j in stage["jobs"]
        }
        save(
            target / "complete.json",
            {
                "plan_sha256": plan_sha,
                "stage": stage["id"],
                "execution_mode": mode,
                "completed_utc": a.utc(),
                "attempt": rel(attempt),
                "summary_sha256": a.digest(attempt / "summary.json"),
                "summary": result,
                "item_receipt_hashes": hashes,
                "next": "Stop here; Astra reviews saved results before the next named stage.",
            },
        )
        print(
            f"STAGE COMPLETE: {stage['id']}. Tell Astra to inspect. No next stage started.",
            flush=True,
        )


def previous_review(stage_index, schedule, plan_sha):
    if stage_index == 0:
        return
    prior = schedule["stages"][stage_index - 1]["id"]
    path = BASE / "stage_reviews" / (prior + ".json")
    a.require(
        path.exists(), "Preceding stage must be audited by Astra before the next stage"
    )
    review = a.read(path)
    a.require(
        review["status"] == "audited"
        and review["plan_sha256"] == plan_sha
        and (review["stage"] == prior),
        "Prior review identity differs",
    )
    a.require(
        review["complete_sha256"]
        == a.digest(BASE / "live/stages" / prior / "complete.json"),
        "Prior stage completion changed",
    )
    evidence = (ROOT / review["evidence"]).resolve()
    a.require(
        evidence.is_relative_to(ROOT)
        and evidence.is_file()
        and (review["evidence_sha256"] == a.digest(evidence)),
        "Prior stage evidence changed",
    )


def run_stage(stage_id):
    plan = validate()
    plan_sha = a.digest(PLAN)
    ready = a.read(BASE / "offline_check_complete.json")
    a.require(
        ready["status"] == "offline_checked_ready_for_user_stage_01"
        and ready["plan_sha256"] == plan_sha,
        "Offline readiness required",
    )
    a.require(
        a.digest(ROOT / ready["evidence"]) == ready["evidence_sha256"],
        "Offline evidence changed",
    )
    schedule = a.read(BASE / "schedule.json")
    selected = next((x for x in schedule["stages"] if x["id"] == stage_id), None)
    a.require(selected is not None, "Unknown frozen stage")
    if not (BASE / "live/stages" / stage_id / "complete.json").exists():
        previous_review(schedule["stages"].index(selected), schedule, plan_sha)
    cat = frozen_catalog(ROOT / plan["prompts"])
    execute_stage(
        selected,
        cat,
        BASE / "live",
        plan_sha,
        client_factory=core.live_client,
        mode="live",
        availability=live_availability,
    )


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "validate", "status", "run"])
    parser.add_argument("--stage", default="stage_01")
    args = parser.parse_args()
    if args.mode == "run":
        run_stage(args.stage)
        return
    with patch.object(socket.socket, "connect", retrieval.forbidden), patch.object(
        socket, "create_connection", retrieval.forbidden
    ):
        if args.mode == "prepare":
            prepare()
        else:
            plan = validate()
            print(
                json.dumps(
                    {
                        "validated": True,
                        "no_API_calls": True,
                        "programme": plan["programme"],
                        "offline_ready": (
                            BASE / "offline_check_complete.json"
                        ).exists(),
                        "completed_stages": [
                            x.name
                            for x in (BASE / "live/stages").glob("*")
                            if (x / "complete.json").exists()
                        ],
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()

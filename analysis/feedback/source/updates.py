import argparse
import base64
from collections import Counter
import json
import os
import socket
import time
from unittest.mock import patch
from analysis.feedback.source import run as baseline
from analysis.feedback.source import payloads as preparation

core = baseline.core
a = baseline.a
ROOT = baseline.ROOT
STUDY = baseline.STUDY
BASE = STUDY / "update_api_v1"
PLAN = BASE / "plan.json"
SCIENCE = baseline.SCIENCE
STUDY_PLAN_SHA = baseline.STUDY_PLAN_SHA
PAYLOAD_PLAN_SHA = "118a90a2a198bc6bc5097ffdf8588750e12b3dd2cea96de5cf76edd24d497bfd"
BASELINE_REVIEW_SHA = "f49061bf53ccff2c3e856017d037c71111d95f268a4d2b150ccb5a1f2d686720"
WORKERS = 8
ENDPOINT_ORDER = baseline.ENDPOINT_ORDER
SEEDS = [202609281, 202609282, 202609283]
CONDITIONS = tuple(
    f"{arm}_s{seed}_b{level:02d}"
    for level in range(1, 5)
    for seed in SEEDS
    for arm in ("error", "correct")
)
SOURCE_FILES = list(
    dict.fromkeys(
        [
            *baseline.SOURCE_FILES,
            "analysis/feedback/source/baseline_review.py",
            "analysis/feedback/source/payloads.py",
            "analysis/feedback/source/updates.py",
            "analysis/feedback/source/updates_build.py",
            "analysis/feedback/source/validate_updates.py",
            "analysis/feedback/source/update_audit.py",
            "tests/feedback/source/test_updates.py",
            "tests/feedback/source/test_update_audit.py",
            "scripts/run_feedback_updates.ps1",
        ]
    )
)


def rel(path):
    return core.rel(path)


def save(path, value):
    path = a.storage_path(path).resolve()
    a.require(
        path.is_relative_to(a.storage_path(BASE).resolve()),
        "Write outside isolated feedback-update execution",
    )
    a.save_new(path, value)


def frozen_catalog(path):
    checked = {
        (r["key"], r["condition"]): r
        for r in core.rows(preparation.BASE / "prompt_inventory.jsonl")
    }
    a.require(len(checked) == 11184, "Wrong independently checked update inventory")
    cat = {}
    with a.storage_path(path).open(encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            key = row["key"], row["condition"]
            a.require(
                key in checked and key not in cat and row["condition"] in CONDITIONS,
                "Unreviewed/duplicate update prompt",
            )
            known = checked[key]
            a.require(
                a.signature(row) == known["entry_sha256"]
                and row["plan_sha256"] == PAYLOAD_PLAN_SHA,
                "Update prompt source differs",
            )
            a.require(
                a.signature(row["messages"])
                == row["messages_sha256"]
                == known["messages_sha256"]
                and a.signature(row["context"])
                == row["context_sha256"]
                == known["context_sha256"],
                "Reviewed context/messages differ",
            )
            a.require(
                row["strategy"] == "hybrid" and row["subset"] in ENDPOINT_ORDER,
                "Wrong retrieval/role",
            )
            a.body(row["messages"])
            cat[key] = {
                k: row[k]
                for k in (
                    "messages",
                    "messages_sha256",
                    "strategy",
                    "subset",
                    "arm",
                    "seed",
                    "budget_level",
                    "pool_size",
                )
            }
    a.require(set(cat) == set(checked), "Incomplete update catalog")
    return cat


def make_schedule(cat):
    keys = sorted({k for k, _ in cat})
    stages = []
    a.require(
        set(cat) == {(k, c) for k in keys for c in CONDITIONS},
        "Incomplete condition coverage",
    )
    for stage_id, levels in (
        ("stage_02", [1, 2]),
        ("stage_03", [3]),
        ("stage_04", [4]),
    ):
        conditions = [c for c in CONDITIONS if int(c[-2:]) in levels]
        jobs = []
        for ordinal, key in enumerate(keys):
            for block in range(3):
                repeat = (ordinal + block) % 3
                for position in range(len(conditions)):
                    condition = conditions[
                        (ordinal + repeat + position) % len(conditions)
                    ]
                    entry = cat[key, condition]
                    a.require(
                        entry["strategy"] == "hybrid"
                        and entry["subset"] in ENDPOINT_ORDER,
                        "Scope/strategy differs",
                    )
                    a.require(
                        entry["budget_level"] == int(condition[-2:])
                        and condition
                        == f"{entry['arm']}_s{entry['seed']}_b{entry['budget_level']:02d}"
                        and entry["pool_size"] == entry["budget_level"] * 10,
                        "Unplanned setting attributes",
                    )
                    jobs.append(
                        {
                            "key": key,
                            "condition": condition,
                            "repeat": repeat,
                            "messages_sha256": entry["messages_sha256"],
                            **{
                                k: entry[k]
                                for k in (
                                    "strategy",
                                    "subset",
                                    "arm",
                                    "seed",
                                    "budget_level",
                                    "pool_size",
                                )
                            },
                        }
                    )
        stages.append(
            {
                "id": stage_id,
                "levels": levels,
                "jobs": jobs,
                "items": len(jobs),
                "distinct_inputs": len(keys),
                "ordinary_responses": len(jobs) * 3,
                "max_logical_calls": len(jobs) * 9,
                "max_physical_POSTs": len(jobs) * 45,
            }
        )
    jobs = [j for s in stages for j in s["jobs"]]
    a.require(
        len(jobs)
        == len({(j["key"], j["condition"], j["repeat"]) for j in jobs})
        == len(cat) * 3,
        "Repeated/missing update jobs",
    )
    return {
        "stages": stages,
        "ordering": "All settings of a text within one batch remain together. Repeat=(input ordinal+block)%3; condition index=(input ordinal+repeat+position)%number of batch conditions. Error/control and pool draws rotate. Completion order asynchronous.",
        "baseline": "Separate completed baseline_api_v1; its predictions are not repeated by this runner.",
        "advance": "User requested one command. Run each fixed batch, audit offline, verify its receipt, then advance. Stop on any execution/audit failure. No outcome-based choice.",
    }


def prepare():
    a.require(
        not PLAN.exists() and not (BASE / "live").exists(), "Preserve frozen updates"
    )
    selection, science, _ = preparation.prerequisites()
    a.require(
        a.digest(preparation.PLAN) == PAYLOAD_PLAN_SHA
        and a.digest(preparation.review.EVIDENCE) == BASELINE_REVIEW_SHA,
        "Prerequisite identity differs",
    )
    completed = a.read(preparation.BASE / "complete.json")
    audit = a.read(preparation.EVIDENCE)
    a.require(
        completed["status"] == "all_update_payloads_verified"
        and a.digest(preparation.EVIDENCE) == completed["evidence_sha256"]
        and audit["entries"] == audit["complete_replay_exact_matches"] == 11184
        and audit["changed_base_contexts"] == 0,
        "All update payloads must pass review",
    )
    tests_path = BASE / "development/tests_passed.json"
    tested = a.read(tests_path)
    a.require(
        tested["passed"] and tested["environment"] == baseline.environment(),
        "Passing current update tests required",
    )
    for name, h in tested["source_hashes"].items():
        a.require(a.digest(ROOT / name) == h, "Tested source changed")
    prompts = preparation.BASE / "prompts.jsonl"
    cat = frozen_catalog(prompts)
    schedule = make_schedule(cat)
    a.require(
        [s["items"] for s in schedule["stages"]] == [16776, 8388, 8388],
        "Frozen update batch budget differs",
    )
    save(BASE / "schedule.json", schedule)
    inputs = {**audit["inputs"], **audit["outputs"]}
    paths = [
        preparation.BASE / "complete.json",
        preparation.EVIDENCE,
        preparation.review.BASE / "complete.json",
        preparation.review.EVIDENCE,
        baseline.PLAN,
        tests_path,
        BASE / "EXECUTION_PROTOCOL.md",
        BASE / "development/source_build_receipt_v1.json",
    ]
    for p in paths:
        inputs[rel(p)] = a.digest(p)
    for name, h in inputs.items():
        a.require(a.digest(ROOT / name) == h, "Frozen prerequisite changed")
    inherited = a.read(baseline.PLAN)
    plan = {
        "created_utc": a.utc(),
        "status": "prepared_offline_check_pending",
        "study_plan_sha256": STUDY_PLAN_SHA,
        "payload_plan_sha256": PAYLOAD_PLAN_SHA,
        "baseline_review_sha256": BASELINE_REVIEW_SHA,
        "prompts": rel(prompts),
        "prompts_sha256": a.digest(prompts),
        "input_hashes": inputs,
        "source_hashes": {name: a.digest(ROOT / name) for name in SOURCE_FILES},
        "generated_hashes": {
            rel(BASE / "schedule.json"): a.digest(BASE / "schedule.json")
        },
        "environment": baseline.environment(),
        "api_contract": a.CONTRACT,
        "lexical_source": inherited["lexical_source"],
        "workers": WORKERS,
        "programme": {
            "internal_batches": 3,
            "whole_study_batches": 4,
            "distinct_inputs": 466,
            "settings": 24,
            "prompts": 11184,
            "predictions": 33552,
            "ordinary_responses": 100656,
            "max_logical_calls": 301968,
            "max_physical_POSTs": 1509840,
        },
        "advance": "One user command executes/audits three fixed internal batches. Stop on any execution or audit failure. A valid evidence-bound audit receipt is mandatory before advancing.",
        "resume": "Exclusive combined-launcher lock; reuse validated completed items/batches and saved partial replies. Never reset exhausted budgets, delete failures or resend unknown intents. Interrupted histories require separate audit review.",
        "selection": "Actual pools and99 sentinels frozen using selection baseline only. No updated-output reselection or accuracy-based stopping.",
        "scoring": "Scientific plan retains one primary final-K error-control transfer contrast. No outcome scoring inside the live combined launcher.",
        "scope": "Saved prompts only; no baseline re-execution, retrieval, model training, CUDA, TEST, active KB or taxonomy changes.",
    }
    save(PLAN, plan)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "plan_sha256": a.digest(PLAN),
                "programme": plan["programme"],
            },
            indent=2,
        )
    )


def validate():
    plan = a.read(PLAN)
    a.require(
        plan["environment"] == baseline.environment()
        and plan["api_contract"] == a.CONTRACT
        and plan["workers"] == WORKERS,
        "Execution environment/contract differs",
    )
    a.require(
        plan["study_plan_sha256"] == STUDY_PLAN_SHA
        and plan["payload_plan_sha256"] == PAYLOAD_PLAN_SHA
        and plan["baseline_review_sha256"] == BASELINE_REVIEW_SHA,
        "Study binding changed",
    )
    for group in ("input_hashes", "source_hashes", "generated_hashes"):
        for name, h in plan[group].items():
            a.require(
                a.digest(ROOT / name) == h, "Frozen update artifact changed: " + name
            )
    a.require(
        a.digest(ROOT / plan["lexical_source"]["path"])
        == plan["lexical_source"]["sha256"],
        "Lexical source changed",
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
    existing, todo = [], []
    implementation = a.implementation()
    for job in jobs:
        entry = cat[job["key"], job["condition"]]
        a.require(
            entry["messages_sha256"] == job["messages_sha256"]
            and entry["strategy"] == job["strategy"]
            and entry["subset"] == job["subset"],
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
                intent, outcome = a.read(path), a.read(outcome_path)
                a.require(
                    intent["job_stamp"] == a.signature(stamp)
                    and outcome["intent_sha256"] == a.signature(intent),
                    "Partial request binding differs",
                )
                action, reason, _ = a.interpret(outcome)
                if action == "stop" or action == "retry" and intent["attempt"] == 4:
                    raise a.CallStopped(
                        reason
                        if action == "stop"
                        else "transport_attempt_budget_exhausted"
                    )
            a.require(
                all(
                    p.with_name("intent.json").exists()
                    for p in folder.rglob("outcome.json")
                ),
                "Orphan response requires review",
            )
        todo.append(job)
    return existing, todo


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
        "null": sum(out["result"] is None for _, out in existing),
    }
    a.require(
        not (
            initial["complete"] >= 20 and initial["null"] / initial["complete"] >= 0.10
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
        null_items=sum(out["result"] is None for _, out in finished),
        finished_utc=a.utc(),
        completed_item_accounting={
            field: sum(out["accounting"][field] for _, out in finished)
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
        and record["execution_mode"] == mode,
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
        and review["stage"] == prior,
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
        and review["evidence_sha256"] == a.digest(evidence),
        "Prior stage evidence changed",
    )


def run_stage(stage_id):
    plan = validate()
    plan_sha = a.digest(PLAN)
    ready = a.read(BASE / "offline_check_complete.json")
    a.require(
        ready["status"] == "offline_checked_ready_for_user_updates"
        and ready["plan_sha256"] == plan_sha,
        "Combined offline readiness required",
    )
    a.require(
        a.digest(ROOT / ready["evidence"]) == ready["evidence_sha256"],
        "Readiness evidence changed",
    )
    schedule = a.read(BASE / "schedule.json")
    stage = next((s for s in schedule["stages"] if s["id"] == stage_id), None)
    a.require(stage is not None, "Unknown frozen update stage")
    if not (BASE / "live/stages" / stage_id / "complete.json").exists():
        previous_review(schedule["stages"].index(stage), schedule, plan_sha)
    execute_stage(
        stage,
        frozen_catalog(ROOT / plan["prompts"]),
        BASE / "live",
        plan_sha,
        client_factory=core.live_client,
        mode="live",
        availability=live_availability,
    )


def run_sequence(stage_ids, run_one, audit_one, review_exists, verify_review):

    for stage_id in stage_ids:
        run_one(stage_id)
        if not review_exists(stage_id):
            print(
                f"{stage_id}: starting its offline audit. Leave this command running; it advances automatically only after verification.",
                flush=True,
            )
            audit_one(stage_id)
        verify_review(stage_id)
        print(f"{stage_id}: saved outputs and audit verified.", flush=True)


def endpoint_progress(schedule, index):
    result = {}
    for subset in ENDPOINT_ORDER:
        all_jobs = [
            j for s in schedule["stages"] for j in s["jobs"] if j["subset"] == subset
        ]
        done = [
            j
            for s in schedule["stages"][: index + 1]
            for j in s["jobs"]
            if j["subset"] == subset
        ]
        result[subset] = {
            "completed_inputs": len({j["key"] for j in done}),
            "total_inputs": len({j["key"] for j in all_jobs}),
            "completed_predictions": len(done),
            "total_predictions": len(all_jobs),
            "complete": len(done) == len(all_jobs),
        }
    return result


def run_all():
    from analysis.feedback.source import update_audit as auditor

    validate()
    schedule = a.read(BASE / "schedule.json")
    plan_sha = a.digest(PLAN)
    stage_ids = [s["id"] for s in schedule["stages"]]
    with a.item_lock(a.storage_path(BASE / "combined_launcher.lock")):
        run_sequence(
            stage_ids,
            run_stage,
            auditor.audit,
            lambda s: (BASE / "stage_reviews" / f"{s}.json").exists(),
            lambda s: previous_review(stage_ids.index(s) + 1, schedule, plan_sha),
        )
    print(
        "ALL THREE UPDATE BATCHES AND OFFLINE AUDITS COMPLETE. Tell Astra to inspect all results. No further experiment started.",
        flush=True,
    )


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "validate", "status", "run-all"])
    args = parser.parse_args()
    if args.mode == "run-all":
        run_all()
        return
    with patch.object(socket.socket, "connect", u_forbid), patch.object(
        socket, "create_connection", u_forbid
    ), patch.object(core, "live_client", u_forbid):
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
                            p.name
                            for p in (BASE / "live/stages").glob("*")
                            if (p / "complete.json").exists()
                        ],
                    },
                    indent=2,
                )
            )


def u_forbid(*args, **kwargs):
    raise AssertionError("No network/credentials during offline preparation")


if __name__ == "__main__":
    main()

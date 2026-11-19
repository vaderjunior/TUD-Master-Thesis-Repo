import argparse
import base64
from collections import Counter
import importlib.metadata
import json
import os
import socket
import time
from unittest.mock import patch
from analysis.common import batch as core
from analysis.feedback.source import eligibility as preparation
from analysis.common import prompt as renderer

a = core.a
ROOT = a.ROOT
STUDY = preparation.BASE
BASE = STUDY / "baseline_api_v1"
PLAN = BASE / "plan.json"
SCIENCE = STUDY / "scientific_plan_v1.json"
STUDY_PLAN_SHA = "087b00447f7fe6428f382f8693df191b66c16ef9f6d14044918853ad4ce9cae6"
WORKERS = 8
CONDITIONS = ("selection_baseline", "reference_baseline")
ENDPOINT_ORDER = ("candidate", "pool", "held_out")
SOURCE_FILES = list(
    dict.fromkeys(
        [
            "analysis/feedback/source/run.py",
            "analysis/feedback/source/audit_derivation.py",
            "analysis/feedback/source/validate_api.py",
            "analysis/feedback/source/audit.py",
            "tests/feedback/source/test_run.py",
            "tests/feedback/source/test_audit.py",
            "analysis/feedback/source/retrieval.py",
            "scripts/run_feedback_reference.ps1",
            "analysis/feedback/source/design.py",
            "analysis/feedback/source/eligibility.py",
            "analysis/feedback/summary/p3_audit.py",
            "analysis/common/prompt.py",
            "src/hsrag/prompt.py",
            "prompts/classify_v1.txt",
            "analysis/knowledge/sampled_banks/run.py",
            "analysis/knowledge/sampled_banks/validate_stage.py",
            "tests/knowledge/sampled_banks/test_stage_validation.py",
            "analysis/knowledge/strategies/run.py",
            "analysis/knowledge/strategies/validate_stage.py",
            "analysis/common/batch.py",
            "analysis/common/scoring.py",
            "tests/common/test_api.py",
            "tests/common/test_batch.py",
            *a.DEPENDENCIES,
        ]
    )
)


def rel(path):
    return core.rel(path)


def save(path, value):
    path = a.storage_path(path).resolve()
    a.require(
        path.is_relative_to(a.storage_path(BASE).resolve()),
        "Write outside isolated SQ3 baseline execution",
    )
    a.save_new(path, value)


def environment():
    return {**core.environment(), "rank_bm25": importlib.metadata.version("rank-bm25")}


def make_catalog(prompts, roster, science):
    byid = {r["id"]: r for r in roster}
    pmap = {r["item_id"]: r for r in prompts}
    a.require(
        len(byid) == len(roster) == len(pmap) == len(prompts),
        "Duplicate/missing baseline sources",
    )
    a.require(set(byid) == set(pmap), "Source roster differs")
    allowed = science["rosters"]
    eligible = set(allowed["eligible_candidates"])
    transfer = set(allowed["transfer"])
    a.require(
        set(allowed["evaluation_phase"]) == eligible | transfer
        and (not eligible & transfer),
        "Evaluation roles differ",
    )
    a.require(
        set(allowed["selection_phase"])
        == eligible | set(allowed["sentinel_selection_pool"]),
        "Selection roles differ",
    )
    a.require(
        not set(allowed["selection_phase"]) & set(allowed["holdout"]),
        "Holdout included in selection",
    )
    a.require(
        not set(allowed["excluded_candidate_ids"]) & set(allowed["evaluation_phase"]),
        "Quarantined ID included",
    )
    cat = {}
    for condition, role in zip(CONDITIONS, ("selection_phase", "evaluation_phase")):
        ids = allowed[role]
        a.require(len(ids) == len(set(ids)), "Duplicate phase input")
        for iid in ids:
            row = pmap[iid]
            source = byid[iid]
            a.require(
                row["text"] == source["text"]
                and row["key"] == source["key"] == "en/" + iid
                and (row["lang"] == "en"),
                "Source text/language differs",
            )
            a.require(
                row["pool"] == "full" and row["condition"] == "rag_full",
                "Wrong original condition",
            )
            ctx = row["context"]
            a.require(
                len(ctx["definitions"]) == len(ctx["guidelines"]) == 2
                and len(ctx["example_groups"]) == 1
                and (ctx["example_groups"][0][0] == "Labelled examples")
                and (len(ctx["example_groups"][0][1]) == 5),
                "Baseline context differs",
            )
            a.require(
                a.signature(ctx) == row["context_sha256"]
                and a.signature(row["messages"]) == row["messages_sha256"],
                "Payload digest differs",
            )
            a.body(row["messages"])
            subset = "candidate" if iid in eligible else source["role"]
            a.require(
                subset in ENDPOINT_ORDER
                and (source["correction_eligible"] is True) == (iid in eligible),
                "Wrong candidate role",
            )
            cat[row["key"], condition] = {
                "messages": row["messages"],
                "messages_sha256": row["messages_sha256"],
                "strategy": "hybrid",
                "subset": subset,
            }
    return cat


def frozen_catalog(path):
    a.require(a.digest(SCIENCE) == STUDY_PLAN_SHA, "Scientific plan changed")
    science = a.read(SCIENCE)
    for p in (path, preparation.OUT / "roster.jsonl"):
        a.require(a.digest(p) == science["inputs"][rel(p)], "Baseline source changed")
    return make_catalog(
        core.rows(path), core.rows(preparation.OUT / "roster.jsonl"), science
    )


def make_schedule(cat):
    jobs = []
    for condition in CONDITIONS:
        keys = sorted((key for key, c in cat if c == condition))
        for ordinal, key in enumerate(keys):
            entry = cat[key, condition]
            a.require(
                entry["subset"] in ENDPOINT_ORDER and entry["strategy"] == "hybrid",
                "Unplanned scope/strategy",
            )
            a.require(
                condition != "selection_baseline" or entry["subset"] != "held_out",
                "Selection must exclude holdout",
            )
            for block in range(3):
                jobs.append(
                    {
                        "key": key,
                        "condition": condition,
                        "phase": condition.removesuffix("_baseline"),
                        "subset": entry["subset"],
                        "strategy": "hybrid",
                        "repeat": (ordinal + block) % 3,
                        "messages_sha256": entry["messages_sha256"],
                    }
                )
    a.require(
        len(jobs) == len(cat) * 3
        and len({(j["key"], j["condition"], j["repeat"]) for j in jobs}) == len(jobs),
        "Missing or duplicate phase/condition",
    )
    stage = {
        "id": "stage_01",
        "scientific_batch": "baseline_selection_and_reference",
        "jobs": jobs,
        "items": len(jobs),
        "distinct_inputs": len({j["key"] for j in jobs}),
        "ordinary_responses": len(jobs) * 3,
        "max_logical_calls": len(jobs) * 9,
        "max_physical_POSTs": len(jobs) * 45,
        "phase_predictions": dict(Counter((j["phase"] for j in jobs))),
    }
    return {
        "stages": [stage],
        "ordering": "Selection phase then separate reference phase; sorted keys within phase, repeat=(input ordinal+block)%3. Submission order fixed, completion asynchronous.",
        "future_batches": ["budgets_01_02", "budget_03", "budget_04"],
        "future_status": "Disabled. Requires baseline review and a new frozen update execution plan.",
    }


def prepare():
    a.require(
        not PLAN.exists() and (not (BASE / "live").exists()),
        "Preserve frozen execution",
    )
    a.require(a.digest(SCIENCE) == STUDY_PLAN_SHA, "Scientific plan differs")
    science = a.read(SCIENCE)
    tests_path = BASE / "development/tests_passed.json"
    tests = a.read(tests_path)
    a.require(
        tests["passed"] and tests["environment"] == environment(),
        "Passing current offline tests required",
    )
    for name, h in tests["source_hashes"].items():
        a.require(a.digest(ROOT / name) == h, "Tested source changed")
    inherited_path = (
        ROOT
        / "experiments/astra/sq1_pool_size_composition_20260926_v1/api_execution_v1/plan.json"
    )
    inherited = a.read(inherited_path)
    a.require(
        inherited["api_contract"] == a.CONTRACT, "Inherited transport contract differs"
    )
    for name in a.DEPENDENCIES + [
        "analysis/common/batch.py",
        "analysis/knowledge/strategies/run.py",
    ]:
        a.require(
            a.digest(ROOT / name) == inherited["source_hashes"][name],
            "Inherited engine source changed",
        )
    prompts = preparation.OUT / "baseline_prompts.jsonl"
    cat = frozen_catalog(prompts)
    schedule = make_schedule(cat)
    a.require(
        schedule["stages"][0]["phase_predictions"]
        == {"selection": 1035, "reference": 1398},
        "Baseline counts differ",
    )
    save(BASE / "schedule.json", schedule)
    paths = [
        SCIENCE,
        inherited_path,
        tests_path,
        BASE / "EXECUTION_PROTOCOL.md",
        BASE / "development/source_build_receipt_v1.json",
        STUDY / "design_verification_v1.json",
        preparation.OUT / "plan.json",
        preparation.OUT / "complete.json",
    ]
    inputs = dict(science["inputs"])
    for group in (
        a.read(preparation.OUT / "plan.json")["input_hashes"],
        {rel(p): a.digest(p) for p in paths},
    ):
        for name, h in group.items():
            a.require(
                name not in inputs or inputs[name] == h, "Conflicting source binding"
            )
            inputs[name] = h
    import rank_bm25

    lexical = __import__("pathlib").Path(rank_bm25.__file__)
    for name, h in inputs.items():
        a.require(a.digest(ROOT / name) == h, "Frozen input changed: " + name)
    plan = {
        "created_utc": a.utc(),
        "status": "prepared_offline_check_pending",
        "study_plan_sha256": STUDY_PLAN_SHA,
        "prompts": rel(prompts),
        "prompts_sha256": a.digest(prompts),
        "input_hashes": inputs,
        "source_hashes": {name: a.digest(ROOT / name) for name in SOURCE_FILES},
        "generated_hashes": {
            rel(BASE / "schedule.json"): a.digest(BASE / "schedule.json")
        },
        "lexical_source": {"path": rel(lexical), "sha256": a.digest(lexical)},
        "environment": environment(),
        "api_contract": a.CONTRACT,
        "workers": WORKERS,
        "programme": {
            "enabled_stages": 1,
            "study_batches": 4,
            "distinct_inputs": 466,
            "phase_payloads": 811,
            "predictions": 2433,
            "ordinary_responses": 7299,
            "max_logical_calls": 21897,
            "max_physical_POSTs": 109485,
        },
        "identity": "Phase is encoded in condition: selection_baseline versus reference_baseline. Identical wire payload blobs may be shared; response records and job identities cannot be shared.",
        "stop": "One named baseline batch only. Eight workers; stop new submissions on error or >=10% whole-null after20 complete; drain pending work. No accuracy stopping.",
        "resume": "Verify saved results, reuse completed votes/partial replies, never resend unresolved intent or reset exhausted transport. Preserved stops require review. Baseline auditor requires one uninterrupted attempt; interrupted execution requires history-aware review.",
        "next": "Audit baseline, then freeze actual matched pools/K/sentinels and update prompts before the remaining three batches. This runner cannot launch updates.",
        "scope": "No real feedback selection, gold scoring, training, embeddings, index mutation, TEST or LaTeX. Fresh English DEV baseline only.",
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
        plan["environment"] == environment()
        and plan["api_contract"] == a.CONTRACT
        and (plan["workers"] == WORKERS),
        "Environment/transport differs",
    )
    a.require(plan["study_plan_sha256"] == STUDY_PLAN_SHA, "Scientific binding changed")
    for group in ("input_hashes", "source_hashes", "generated_hashes"):
        for name, h in plan[group].items():
            a.require(
                a.digest(ROOT / name) == h, "Frozen API artifact changed: " + name
            )
    lexical = plan["lexical_source"]
    a.require(
        a.digest(ROOT / lexical["path"]) == lexical["sha256"],
        "Installed BM25 source changed",
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
    a.require(
        stage_id == "stage_01",
        "Only the baseline is enabled; updates require separate reviewed preparation",
    )
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
    selected = a.read(BASE / "schedule.json")["stages"][0]
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
    with patch.object(socket.socket, "connect", preparation.u.forbid), patch.object(
        socket, "create_connection", preparation.u.forbid
    ), patch.object(core, "live_client", preparation.u.forbid):
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
                        "baseline_complete": (
                            BASE / "live/stages/stage_01/complete.json"
                        ).exists(),
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()

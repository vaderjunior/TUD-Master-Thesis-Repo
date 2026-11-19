import argparse
import base64
from collections import Counter
import json
import os
import socket
import time
from unittest.mock import patch
from analysis.feedback.source import run as baseline
from analysis.feedback.relevance import payloads as preparation

core = baseline.core
a = baseline.a
ROOT = baseline.ROOT
STUDY = preparation.STUDY
BASE = STUDY / "api_execution_v1"
PLAN = BASE / "plan.json"
SCIENCE = preparation.SCIENCE
STUDY_PLAN_SHA = "5665da9d37f4694b767003059ea086e0c4f60c9c499574fcb8129e9f57e6a785"
WORKERS = 8
ENDPOINT_ORDER = ("pool", "held_out")
CONDITIONS = tuple(
    f"{p}_s{s}" for p in preparation.design.POLICIES for s in preparation.h.SEEDS
) + ("reference",)
SOURCE_FILES = list(
    dict.fromkeys(
        [
            *baseline.SOURCE_FILES,
            "analysis/feedback/source/updates.py",
            "analysis/feedback/source/update_audit.py",
            "tests/feedback/source/test_update_audit.py",
            "analysis/feedback/relevance/plan.py",
            "analysis/feedback/relevance/payloads.py",
            "analysis/feedback/relevance/run.py",
            "analysis/feedback/relevance/build.py",
            "analysis/feedback/relevance/validate.py",
            "analysis/feedback/relevance/audit.py",
            "tests/feedback/relevance/test_run.py",
            "tests/feedback/relevance/test_audit.py",
            "scripts/run_feedback_relevance.ps1",
        ]
    )
)


def rel(path):
    return core.rel(path)


def save(path, value):
    path = a.storage_path(path).resolve()
    a.require(
        path.is_relative_to(a.storage_path(BASE).resolve()),
        "Write outside isolated relevance execution",
    )
    a.save_new(path, value)


def frozen_catalog(path):
    checked = {
        (r["key"], r["condition"]): r
        for r in core.rows(preparation.BASE / "inventory.jsonl")
    }
    a.require(len(checked) == 5600, "Wrong reviewed relevance inventory")
    cat = {}
    for row in core.rows(path):
        key = row["key"], row["condition"]
        a.require(
            key in checked and key not in cat and row["condition"] in CONDITIONS,
            "Unreviewed or duplicate prompt",
        )
        known = checked[key]
        a.require(
            a.signature(row) == known["entry_sha256"]
            and row["scientific_plan_sha256"] == STUDY_PLAN_SHA,
            "Payload source differs",
        )
        a.require(
            a.signature(row["messages"])
            == row["messages_sha256"]
            == known["messages_sha256"]
            and a.signature(row["context"])
            == row["context_sha256"]
            == known["context_sha256"],
            "Context/messages differ",
        )
        a.require(
            row["strategy"] == "hybrid" and row["subset"] in ENDPOINT_ORDER,
            "Wrong strategy or role",
        )
        a.body(row["messages"])
        cat[key] = {
            k: row[k]
            for k in (
                "messages",
                "messages_sha256",
                "strategy",
                "subset",
                "policy",
                "pool_seed",
            )
        }
    a.require(set(cat) == set(checked), "Incomplete relevance catalog")
    return cat


def make_schedule(cat):
    science = a.read(SCIENCE)
    stages = []
    expected = {
        ("en/" + i, c) for i in science["rosters"]["evaluation"] for c in CONDITIONS
    }
    a.require(set(cat) == expected, "Incomplete condition matrix")
    for index, ids in enumerate(science["batching"]["input_batches"], 1):
        jobs = []
        for ordinal, iid in enumerate(ids):
            key = "en/" + iid
            for block in range(3):
                repeat = (ordinal + block) % 3
                for position in range(len(CONDITIONS)):
                    condition = CONDITIONS[
                        (ordinal + repeat + position) % len(CONDITIONS)
                    ]
                    entry = cat[key, condition]
                    jobs.append(
                        {
                            "key": key,
                            "condition": condition,
                            "repeat": repeat,
                            "messages_sha256": entry["messages_sha256"],
                            **{
                                k: entry[k]
                                for k in ("strategy", "subset", "policy", "pool_seed")
                            },
                        }
                    )
        stages.append(
            {
                "id": f"stage_{index:02d}",
                "jobs": jobs,
                "items": len(jobs),
                "distinct_inputs": len(ids),
                "ordinary_responses": len(jobs) * 3,
                "max_logical_calls": len(jobs) * 9,
                "max_physical_POSTs": len(jobs) * 45,
            }
        )
    jobs = [j for s in stages for j in s["jobs"]]
    a.require(
        len(jobs)
        == len({(j["key"], j["condition"], j["repeat"]) for j in jobs})
        == 16800,
        "Repeated/missing job",
    )
    a.require(
        [s["items"] for s in stages] == [5616, 5616, 5568],
        "Scientific batch size differs",
    )
    return {
        "stages": stages,
        "ordering": "Frozen input partitions; all16 conditions and3 repeats for one input stay in the same batch. Conditions and repeats rotate by input ordinal; completion is asynchronous.",
        "reference": "Three actual fresh reference repeats per input, shared across pool comparisons; never triplicated.",
        "advance": "Automatic operational audit after every fixed batch. Stop on any failure; no outcome-based stopping.",
    }


def prepare():
    a.require(
        not PLAN.exists() and not (BASE / "live").exists(), "Preserve frozen execution"
    )
    a.require(a.digest(SCIENCE) == STUDY_PLAN_SHA, "Scientific plan changed")
    completed = a.read(preparation.BASE / "complete.json")
    a.require(
        completed["status"] == "all_relevance_payloads_verified"
        and completed["entries"] == 5600
        and completed["natural_exact_matches"] == 1050
        and completed["reference_exact_matches"] == 350,
        "Payload checks incomplete",
    )
    tested = a.read(BASE / "development/tests_passed.json")
    a.require(
        tested["passed"] and tested["environment"] == baseline.environment(),
        "Passing current tests required",
    )
    for name, h in tested["source_hashes"].items():
        a.require(a.digest(ROOT / name) == h, "Tested source changed")
    prompts = preparation.BASE / "prompts.jsonl"
    cat = frozen_catalog(prompts)
    schedule = make_schedule(cat)
    save(BASE / "schedule.json", schedule)
    inputs = {**completed["inputs"], **completed["outputs"]}
    for p in [
        SCIENCE,
        STUDY / "design_complete.json",
        preparation.BASE / "plan.json",
        preparation.BASE / "complete.json",
        BASE / "development/tests_passed.json",
        BASE / "development/source_build_receipt_v1.json",
        BASE / "EXECUTION_PROTOCOL.md",
    ]:
        inputs[rel(p)] = a.digest(p)
    for name, h in inputs.items():
        a.require(a.digest(ROOT / name) == h, "Frozen input changed")
    inherited = a.read(baseline.PLAN)
    save(
        PLAN,
        {
            "created_utc": a.utc(),
            "status": "prepared_offline_check_pending",
            "study_plan_sha256": STUDY_PLAN_SHA,
            "prompts": rel(prompts),
            "prompts_sha256": a.digest(prompts),
            "input_hashes": inputs,
            "source_hashes": {n: a.digest(ROOT / n) for n in SOURCE_FILES},
            "generated_hashes": {
                rel(BASE / "schedule.json"): a.digest(BASE / "schedule.json")
            },
            "environment": baseline.environment(),
            "api_contract": a.CONTRACT,
            "lexical_source": inherited["lexical_source"],
            "workers": WORKERS,
            "programme": {
                "internal_batches": 3,
                "distinct_inputs": 350,
                "settings": 16,
                "prompts": 5600,
                "predictions": 16800,
                "ordinary_responses": 50400,
                "max_logical_calls": 151200,
                "max_physical_POSTs": 756000,
            },
            "scope": "Fresh saved-prompt diagnostic only. No training, retrieval, CUDA, TEST or active configuration changes.",
            "resume": "Verify completed items and saved replies. Unknown request intents or preserved stops require review; no reset or deletion.",
            "advance": "One user command, automatic raw-response audits; integrity failures stop subsequent calls. No gold scoring in launcher.",
        },
    )
    print(
        json.dumps(
            {
                "status": "prepared_offline_check_pending",
                "plan_sha256": a.digest(PLAN),
                "programme": a.read(PLAN)["programme"],
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
        "Execution environment or contract differs",
    )
    a.require(
        plan["study_plan_sha256"] == STUDY_PLAN_SHA == a.digest(SCIENCE),
        "Scientific binding differs",
    )
    for group in ("input_hashes", "source_hashes", "generated_hashes"):
        for name, h in plan[group].items():
            a.require(a.digest(ROOT / name) == h, "Frozen artifact changed: " + name)
    a.require(
        a.digest(ROOT / plan["lexical_source"]["path"])
        == plan["lexical_source"]["sha256"],
        "Inherited package source changed",
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
                "next": "Automatic offline audit follows; advance only after verified success.",
            },
        )
        print(
            f"STAGE COMPLETE: {stage['id']}. Automatic offline audit follows before any next batch.",
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


def run_stage(stage_id):
    plan = validate()
    plan_sha = a.digest(PLAN)
    ready = a.read(BASE / "offline_check_complete.json")
    a.require(
        ready["status"] == "offline_checked_ready_for_user_relevance"
        and ready["plan_sha256"] == plan_sha,
        "Offline readiness required",
    )
    a.require(
        a.digest(ROOT / ready["evidence"]) == ready["evidence_sha256"],
        "Readiness evidence changed",
    )
    schedule = a.read(BASE / "schedule.json")
    stage = next((s for s in schedule["stages"] if s["id"] == stage_id), None)
    a.require(stage is not None, "Unknown frozen relevance stage")
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


def run_all():
    from analysis.feedback.relevance import audit as auditor

    validate()
    schedule = a.read(BASE / "schedule.json")
    plan_sha = a.digest(PLAN)
    ids = [s["id"] for s in schedule["stages"]]
    with a.item_lock(a.storage_path(BASE / "combined_launcher.lock")):
        run_sequence(
            ids,
            run_stage,
            auditor.audit,
            lambda s: (BASE / "stage_reviews" / f"{s}.json").exists(),
            lambda s: previous_review(ids.index(s) + 1, schedule, plan_sha),
        )
    print(
        "ALL THREE RELEVANCE BATCHES AND OFFLINE AUDITS COMPLETE. Tell Astra to inspect. No other SQ3 experiment started.",
        flush=True,
    )


def forbidden(*args, **kwargs):
    raise AssertionError("No network or credentials during offline relevance work")


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "validate", "status", "run-all"])
    args = parser.parse_args()
    if args.mode == "run-all":
        run_all()
        return
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(core, "live_client", forbidden):
        if args.mode == "prepare":
            prepare()
        else:
            plan = validate()
            print(
                json.dumps(
                    {
                        "validated": True,
                        "actual_API_calls": 0,
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


if __name__ == "__main__":
    main()

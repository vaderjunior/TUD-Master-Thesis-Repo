"""Components run."""

import argparse
import base64
from collections import Counter
import json
import os
import socket
import time
from unittest.mock import patch
from analysis.common import batch as core
from analysis.knowledge.components import prepare as preparation
from analysis.knowledge.components import execution as previous

a = core.a
ROOT = a.ROOT
STUDY = preparation.BASE
BASE = STUDY / "api_execution_v2"
PREVIOUS_BASE = previous.BASE
PLAN = BASE / "plan.json"
CONDITIONS = preparation.CONDITIONS
ENDPOINT_ORDER = preparation.ENDPOINT_ORDER
WORKERS = 8
CAP = 8748
UNIT = len(CONDITIONS) * 3
STUDY_PLAN_SHA = "599af3fa836f2b938755ec565ae2965428d6f1b8b2d523d33a016988155b0b3f"
PROMPTS_SHA = "30a35f5d43dc8f86d300eaf4702740f1ee031ca2b692a8af9cfceb06f2ffe68b"
SOURCE_FILES = [
    "analysis/knowledge/components/run.py",
    "tests/knowledge/components/test_run.py",
    "analysis/knowledge/components/validate_execution.py",
    "scripts/run_components.ps1",
    "analysis/knowledge/strategies/run.py",
    "tests/common/test_api.py",
    "tests/common/test_batch.py",
    *previous.SOURCE_FILES,
]


def previous_unstarted():
    a.require(
        not (PREVIOUS_BASE / "live").exists(),
        "Original 13-stage launcher has live artifacts; review before using the four-stage replacement",
    )


def rel(path):
    return core.rel(path)


def save(path, value):
    path = a.storage_path(path).resolve()
    a.require(
        path.is_relative_to(a.storage_path(BASE).resolve()),
        "Write outside new component API experiment",
    )
    a.save_new(path, value)


def frozen_catalog(path):
    entries = core.rows(path)
    cat = {}
    for row in entries:
        key = (row["key"], row["condition"])
        a.require(
            key not in cat and row["condition"] in CONDITIONS,
            "Duplicate/unplanned prompt identity",
        )
        a.require(
            row["strategy"] == "hybrid" and row["pool"] == "full",
            "Source retrieval differs",
        )
        a.require(
            a.signature(row["messages"]) == row["messages_sha256"],
            "Frozen message signature differs",
        )
        a.body(row["messages"])
        cat[key] = {
            "messages": row["messages"],
            "messages_sha256": row["messages_sha256"],
            "strategy": "hybrid",
            "subset": row["scopes"][0]["subset"],
        }
    a.require(len(entries) == len(cat) == 11655, "Wrong component catalogue size")
    return cat


def make_schedule(dev, cat, cap=CAP):
    a.require(
        type(cap) is int and cap >= UNIT,
        "Stage cap cannot hold complete input comparison",
    )
    by_subset = {name: [] for name in ENDPOINT_ORDER}
    a.require(len({row["key"] for row in dev}) == len(dev), "Duplicate input key")
    for row in dev:
        a.require(
            len(row["scopes"]) == 1 and row["scopes"][0]["subset"] in by_subset,
            "Unexpected scope",
        )
        by_subset[row["scopes"][0]["subset"]].append(row["key"])
    a.require(
        set(cat)
        == {(row["key"], condition) for row in dev for condition in CONDITIONS},
        "Prompt coverage differs",
    )
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
                "inputs": len(pending) // UNIT,
                "ordinary_responses": len(pending) * 3,
                "max_logical_calls": len(pending) * 9,
                "max_physical_POSTs": len(pending) * 45,
                "endpoint_inputs": dict(
                    Counter((j["subset"] for j in pending[::UNIT]))
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
            for block in range(3):
                repeat = (ordinal + block) % 3
                for position in range(len(CONDITIONS)):
                    condition = CONDITIONS[
                        (ordinal + 3 * repeat + position) % len(CONDITIONS)
                    ]
                    entry = cat[key, condition]
                    a.require(
                        entry["subset"] == subset and entry["strategy"] == "hybrid",
                        "Catalogue scope differs",
                    )
                    unit.append(
                        {
                            "key": key,
                            "subset": subset,
                            "condition": condition,
                            "strategy": "hybrid",
                            "repeat": repeat,
                            "messages_sha256": entry["messages_sha256"],
                        }
                    )
            if len(pending) + UNIT > cap:
                finish()
            pending.extend(unit)
            ordinal += 1
    finish()
    jobs = [job for stage in stages for job in stage["jobs"]]
    identities = {(j["key"], j["condition"], j["repeat"]) for j in jobs}
    a.require(
        len(jobs) == len(identities) == len(cat) * 3,
        "Duplicate or missing scheduled prediction",
    )
    a.require(
        identities == {(key, cond, r) for key, cond in cat for r in range(3)},
        "Wrong scheduled identity set",
    )
    return {
        "stages": stages,
        "endpoint_round_robin_order": list(ENDPOINT_ORDER),
        "predictions_per_input": UNIT,
        "ordering": "Round-robin saved DEV rosters; input ordinal i. Repeat=(i+block)%3; condition=CONDITIONS[(i+3*repeat+position)%9]. All27 predictions of one input remain adjacent in one stage. Submission order is fixed; completion order is asynchronous.",
    }


def prepare():
    a.require(not PLAN.exists(), "Execution already frozen; validate instead")
    previous_unstarted()
    prior = previous.validate()
    a.require(
        a.digest(previous.PLAN)
        == "8a72b6c0c2131d9440464d322d5bf396515d9312da4836b8cf0a34bd907c4ec2",
        "Original execution plan differs",
    )
    prior_ready = a.read(PREVIOUS_BASE / "offline_check_complete.json")
    a.require(
        prior_ready["plan_sha256"] == a.digest(previous.PLAN)
        and prior_ready["status"] == "offline_checked_ready_for_user_stage_01",
        "Original readiness differs",
    )
    a.require(
        a.digest(ROOT / prior_ready["evidence"])
        == prior_ready["evidence_sha256"]
        == "0e21d3c6789675ce3c084a4c0d2473fa0ff72fc2be6a876de3efe422115d178e",
        "Original offline evidence differs",
    )
    study = preparation.validate()
    a.require(
        a.digest(preparation.PLAN) == STUDY_PLAN_SHA
        and a.digest(preparation.PROMPTS) == PROMPTS_SHA,
        "Study inputs differ",
    )
    acceptance_path = STUDY / "preparation_checked.json"
    acceptance = a.read(acceptance_path)
    a.require(
        acceptance["status"] == "all_component_prompts_checked_offline"
        and acceptance["plan_sha256"] == STUDY_PLAN_SHA,
        "Independent component preparation acceptance required",
    )
    a.require(
        a.digest(ROOT / acceptance["evidence"]) == acceptance["evidence_sha256"]
        and acceptance["prompts_sha256"] == PROMPTS_SHA,
        "Preparation review binding differs",
    )
    tests_path = BASE / "development/tests_passed.json"
    tests = a.read(tests_path)
    a.require(
        tests["passed"] is True and tests["environment"] == core.environment(),
        "Passing API tests required",
    )
    for name, digest in tests["source_hashes"].items():
        a.require(a.digest(ROOT / name) == digest, "API source changed after tests")
    old_api = a.read(preparation.SOURCE / "api_execution_v1/plan.json")
    a.require(
        old_api["api_contract"] == a.CONTRACT, "Reviewed execution contract differs"
    )
    for name in a.DEPENDENCIES + [
        "analysis/common/batch.py",
        "analysis/knowledge/strategies/run.py",
    ]:
        a.require(
            a.digest(ROOT / name) == old_api["source_hashes"][name],
            "Inherited execution source changed",
        )
    cat = frozen_catalog(preparation.PROMPTS)
    schedule = make_schedule(core.rows(preparation.CORE / "dev_inputs.jsonl"), cat)
    a.require(
        [s["items"] for s in schedule["stages"]] == [8748] * 3 + [8721],
        "Unexpected stage budget",
    )
    prior_schedule = a.read(PREVIOUS_BASE / "schedule.json")
    original_jobs = [j for s in prior_schedule["stages"] for j in s["jobs"]]
    new_jobs = [j for s in schedule["stages"] for j in s["jobs"]]
    a.require(
        new_jobs == original_jobs,
        "Grouping must preserve every job and exact submission order",
    )
    amendment = {
        "created_utc": a.utc(),
        "status": "execution_only_amendment_before_any_live_output",
        "authorization": "Ajay asked to replace thirteen launches/review stops with four larger stages.",
        "previous_plan_sha256": a.digest(previous.PLAN),
        "scientific_plan_sha256": STUDY_PLAN_SHA,
        "prompts_sha256": PROMPTS_SHA,
        "previous_stages": 13,
        "stages": 4,
        "previous_cap": 2700,
        "cap": CAP,
        "stage_inputs": [324, 324, 324, 323],
        "stage_predictions": [8748, 8748, 8748, 8721],
        "identical_flat_job_order": True,
        "inherited_analysis_sha256": a.digest(STUDY / "analysis_plan.json"),
        "unchanged": "All 34965 job identities excluding the new execution-plan hash; exact messages, inputs, conditions, repeats, votes, worker count, transport/repair caps, parser, voting and statistical contract.",
        "operational_changes": "Four named stages, larger per-stage cumulative null-guard population and fewer manual review pauses. Per-prediction durable saves and fail-stop thresholds stay unchanged. New output root/plan identity; v1 has no live outputs to migrate.",
        "precedence": "Overrides only the thirteen-stage schedule/cap/checkpoint wording in original PROTOCOL.md, scientific plan budget and v1 execution handoff. Original scientific questions/contrasts and outcome rules remain frozen.",
        "preservation": "No original source, prompt, plan, result or receipt replaced; v1 remains historical. Use v2 only for this study execution.",
    }
    save(BASE / "scheduling_amendment.json", amendment)
    save(BASE / "schedule.json", schedule)
    save(BASE / "analysis_plan.json", a.read(STUDY / "analysis_plan.json"))
    a.require(
        a.digest(BASE / "analysis_plan.json") == study["analysis_sha256"],
        "Analysis copy differs",
    )
    paths = [
        tests_path,
        preparation.PLAN,
        preparation.PROMPTS,
        STUDY / "PROTOCOL.md",
        STUDY / "analysis_plan.json",
        STUDY / "render_complete.json",
        acceptance_path,
        ROOT / acceptance["evidence"],
        preparation.SOURCE / "api_execution_v1/plan.json",
        previous.PLAN,
        PREVIOUS_BASE / "schedule.json",
        PREVIOUS_BASE / "analysis_plan.json",
        PREVIOUS_BASE / "offline_check_complete.json",
        ROOT / prior_ready["evidence"],
        BASE / "SCHEDULING_AMENDMENT.md",
    ]
    inputs = {**study["input_hashes"], **{rel(p): a.digest(p) for p in paths}}
    budget = study["budget"]
    plan = {
        "created_utc": a.utc(),
        "status": "prepared_offline_check_pending",
        "study_plan_sha256": STUDY_PLAN_SHA,
        "prompts": rel(preparation.PROMPTS),
        "prompts_sha256": PROMPTS_SHA,
        "input_hashes": inputs,
        "source_hashes": {name: a.digest(ROOT / name) for name in SOURCE_FILES},
        "generated_hashes": {
            rel(BASE / name): a.digest(BASE / name)
            for name in (
                "schedule.json",
                "analysis_plan.json",
                "scheduling_amendment.json",
            )
        },
        "environment": core.environment(),
        "api_contract": a.CONTRACT,
        "workers": WORKERS,
        "programme": {
            "stages": 4,
            "distinct_inputs": 1295,
            "component_prompts": 11655,
            "predictions": 34965,
            "ordinary_responses": 104895,
            "max_logical_calls": budget["max_logical_calls"],
            "max_physical_POSTs": budget["max_physical_POSTs"],
        },
        "stage_cap_predictions": CAP,
        "predictions_per_input": UNIT,
        "model_check": "Before incomplete live-stage attempts, save GET /models and require pinned Qwen. No provider fallback or extra pilot samples.",
        "resume": "Audit every selected saved item before calls. Complete items skipped; partial replies reused. Unresolved intents, partial writes, fatal/exhausted budgets or sticky stop require review; never reset limits.",
        "stop": "One named stage. At most8 workers; stop new submissions on error or >=10% null predictions after20 complete, drain pending work. No accuracy-based stopping.",
        "next_stage": "Requires prior completion/evidence-bound review receipt. No automatic advance.",
        "scope": "Same nine fresh conditions and exact job order, grouped into four larger stages. No retrieval/training/gold scoring/TEST/benchmark/Priority3/LaTeX. Old studies unchanged.",
        "previous_execution_plan_sha256": a.digest(previous.PLAN),
        "scheduling_amendment_sha256": a.digest(BASE / "scheduling_amendment.json"),
    }
    save(PLAN, plan)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "plan_sha256": a.digest(PLAN),
                "programme": plan["programme"],
                "stage_predictions": [s["items"] for s in schedule["stages"]],
            },
            indent=2,
        )
    )


def validate():
    previous_unstarted()
    plan = a.read(PLAN)
    a.require(
        plan["environment"] == core.environment()
        and plan["api_contract"] == a.CONTRACT,
        "Execution environment/contract changed",
    )
    a.require(
        plan["workers"] == WORKERS
        and plan["stage_cap_predictions"] == CAP
        and (plan["predictions_per_input"] == UNIT),
        "Execution budget changed",
    )
    a.require(
        plan["study_plan_sha256"] == STUDY_PLAN_SHA
        and plan["prompts_sha256"] == PROMPTS_SHA,
        "Study binding changed",
    )
    a.require(
        plan["programme"]["stages"] == 4
        and plan["scheduling_amendment_sha256"]
        == a.digest(BASE / "scheduling_amendment.json"),
        "Four-stage amendment changed",
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
    selected = next((s for s in schedule["stages"] if s["id"] == stage_id), None)
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
    with patch.object(socket.socket, "connect", preparation.forbidden), patch.object(
        socket, "create_connection", preparation.forbidden
    ), patch.object(core, "live_client", preparation.forbidden):
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
                            s.name
                            for s in (BASE / "live/stages").glob("*")
                            if (s / "complete.json").exists()
                        ],
                    },
                    indent=2,
                )
            )


if __name__ == "__main__":
    main()

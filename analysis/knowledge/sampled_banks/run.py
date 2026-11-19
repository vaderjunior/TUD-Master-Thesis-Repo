"""Sampled banks run."""

import argparse
import base64
from collections import Counter
import json
import os
import socket
import time
from unittest.mock import patch
from analysis.common import batch as core
from analysis.knowledge.sampled_banks import analysis_plan as preparation

a = core.a
ROOT = a.ROOT
STUDY = preparation.STUDY
BASE = STUDY / "api_execution_v1"
PLAN = BASE / "plan.json"
CONDITIONS = preparation.CONDITIONS
ENDPOINT_ORDER = preparation.ENDPOINT_ORDER
WORKERS = 8
CAP = 10692
UNIT = len(CONDITIONS) * 3
STUDY_PLAN_SHA = "1b5bdced39a6e5f81a4a19122a321b9c61ddb7c881ef4563e3c034d41ec88a4e"
RETRIEVAL_PLAN_SHA = "e665e8427a9c59dd6ee63276ebd7ed6b07640722780c58c96dc6c2bd5020568c"
PROMPTS_SHA = "6d750f21b14202fa3aca8026806cc2c65a234585f6e0f9609d627265143f4c6b"
AUDIT_SHA = "678d3091f72834c6f219fecd3aa330cbf1f79d26805ab930e8b3dce4f2a3dcd8"
SCOPE_COUNTS = preparation.SCOPE_COUNTS
POOL_IDS = {c: c for c in CONDITIONS[:2]} | {
    c: f"p{i:02d}" for i, c in enumerate(CONDITIONS[2:])
}
SOURCE_FILES = list(
    dict.fromkeys(
        [
            "analysis/knowledge/sampled_banks/run.py",
            "analysis/knowledge/sampled_banks/analysis_plan.py",
            "analysis/knowledge/sampled_banks/validate_execution.py",
            "tests/knowledge/sampled_banks/test_run.py",
            "tests/knowledge/sampled_banks/test_analysis_plan.py",
            "analysis/knowledge/sampled_banks/validate_stage.py",
            "tests/knowledge/sampled_banks/test_stage_validation.py",
            "scripts/run_sampled_banks.ps1",
            "analysis/knowledge/allocation/run.py",
            "analysis/knowledge/allocation/validate_execution.py",
            "tests/knowledge/allocation/test_run.py",
            "analysis/knowledge/allocation/validate_stage.py",
            "tests/knowledge/allocation/test_stage_validation.py",
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
        "Write outside new sampled-pool API experiment",
    )
    a.save_new(path, value)


def catalog_entry(row, checked):
    key = (row["key"], row["condition"])
    a.require(
        row["condition"] in CONDITIONS and row["strategy"] == "hybrid",
        "Unplanned prompt/strategy",
    )
    a.require(
        row["pool_id"] == POOL_IDS[row["condition"]], "Wrong frozen candidate pool"
    )
    a.require(
        row["plan_sha256"] == RETRIEVAL_PLAN_SHA
        and len(row["scopes"]) == 1
        and (row["scopes"][0]["subset"] in ENDPOINT_ORDER),
        "Wrong retrieval identity/scope",
    )
    a.require(
        key == (checked["key"], checked["condition"])
        and a.signature(row) == checked["entry_sha256"],
        "Entry differs from independently reviewed original first pass",
    )
    a.require(
        a.signature(row["messages"])
        == row["messages_sha256"]
        == checked["messages_sha256"]
        and a.signature(row["context"])
        == row["context_sha256"]
        == checked["context_sha256"],
        "Frozen messages/context differ",
    )
    a.body(row["messages"])
    return {
        "messages": row["messages"],
        "messages_sha256": row["messages_sha256"],
        "strategy": "hybrid",
        "subset": row["scopes"][0]["subset"],
    }


def frozen_catalog(path):
    inventory = core.rows(STUDY / "retrieval_v1/review_v1/prompt_inventory.jsonl")
    checked = {(x["key"], x["condition"]): x for x in inventory}
    a.require(
        len(inventory) == len(checked) == 14245,
        "Wrong independently reviewed inventory",
    )
    cat = {}
    with a.storage_path(path).open(encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            key = (row["key"], row["condition"])
            a.require(
                key not in cat and key in checked, "Duplicate or unreviewed prompt"
            )
            cat[key] = catalog_entry(row, checked[key])
    a.require(set(cat) == set(checked), "Missing reviewed prompt")
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
                        (ordinal + repeat + position) % len(CONDITIONS)
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
        "ordering": "Round-robin saved DEV rosters; input ordinal i. Repeat=(i+block)%3; condition=CONDITIONS[(i+repeat+position)%11]. All33 predictions of one input remain adjacent in one stage. Submission order is fixed; completion order is asynchronous.",
    }


def prepare():
    a.require(
        not PLAN.exists() and (not (BASE / "live").exists()),
        "Preserve existing execution",
    )
    science_path = STUDY / "scientific_plan.json"
    study = a.read(science_path)
    retrieval = STUDY / "retrieval_v1"
    review = retrieval / "review_v1"
    a.require(
        a.digest(science_path) == STUDY_PLAN_SHA
        and a.digest(retrieval / "plan.json") == RETRIEVAL_PLAN_SHA,
        "Frozen scientific/retrieval plan differs",
    )
    acceptance_path = review / "retrieval_acceptance.json"
    acceptance = a.read(acceptance_path)
    a.require(
        acceptance["status"]
        == "accepted_original_first_pass_prompts_with_documented_replay_variation"
        and acceptance["entries"] == 14245
        and (acceptance["change_original_retrieval"] is False),
        "Qualified original-first-pass acceptance required",
    )
    prompts = ROOT / acceptance["prompts"]
    review_path = ROOT / acceptance["analysis"]
    a.require(
        a.digest(prompts) == acceptance["prompts_sha256"] == PROMPTS_SHA
        and a.digest(review_path) == acceptance["analysis_sha256"] == AUDIT_SHA,
        "Accepted prompts/review differ",
    )
    reviewed = a.read(review_path)
    a.require(
        reviewed["review_checks"]["streamed_first_pass_payloads"] == 14245
        and reviewed["replay"]["changed_messages"] == 13
        and (reviewed["replay"]["changed_entries"] == 62),
        "Qualified replay pattern differs",
    )
    tests_path = BASE / "development/tests_passed.json"
    tests = a.read(tests_path)
    a.require(
        tests["passed"] is True and tests["environment"] == core.environment(),
        "Passing API tests required",
    )
    for name, digest in tests["source_hashes"].items():
        a.require(a.digest(ROOT / name) == digest, "API source changed after tests")
    inherited_path = (
        ROOT
        / "experiments/astra/sq1_full_kb_allocation_20260924_v1/api_execution_v1/plan.json"
    )
    inherited = a.read(inherited_path)
    a.require(inherited["api_contract"] == a.CONTRACT, "Reviewed API contract differs")
    for name in a.DEPENDENCIES + [
        "analysis/common/batch.py",
        "analysis/knowledge/strategies/run.py",
    ]:
        a.require(
            a.digest(ROOT / name) == inherited["source_hashes"][name],
            "Inherited execution source changed",
        )
    scoring_path = preparation.CORE / "api_execution_v1/scoring_plan.json"
    original_plan = a.read(preparation.CORE / "plan.json")
    gold = {
        f"data/processed/{name}.parquet": original_plan["input_hashes"][
            f"data/processed/{name}.parquet"
        ]
        for name in ENDPOINT_ORDER
    }
    analysis = preparation.analysis_plan(
        study, a.read(scoring_path), a.digest(scoring_path), gold
    )
    cat = frozen_catalog(prompts)
    schedule = make_schedule(core.rows(preparation.CORE / "dev_inputs.jsonl"), cat)
    a.require(
        [s["items"] for s in schedule["stages"]] == [10692] * 3 + [10659],
        "Unexpected four-stage budget",
    )
    save(BASE / "schedule.json", schedule)
    save(BASE / "analysis_plan.json", analysis)
    paths = [
        tests_path,
        science_path,
        retrieval / "plan.json",
        prompts,
        acceptance_path,
        review_path,
        review / "prompt_inventory.jsonl",
        review / "replay_differences.jsonl",
        review / "publication_v1/complete.json",
        retrieval / "full_audit_v2.json",
        retrieval / "audit_amendment_v2.json",
        scoring_path,
        inherited_path,
        BASE / "EXECUTION_PROTOCOL.md",
        BASE / "development/source_build_receipt_v1.json",
    ]
    inputs = {}
    for group in (
        study["input_hashes"],
        reviewed["hashes"],
        gold,
        {rel(path): a.digest(path) for path in paths},
    ):
        for name, digest in group.items():
            a.require(
                name not in inputs or inputs[name] == digest,
                "Conflicting inherited hash",
            )
            inputs[name] = digest
    for name, digest in inputs.items():
        a.require(
            a.digest(ROOT / name) == digest, "Frozen prerequisite changed: " + name
        )
    plan = {
        "created_utc": a.utc(),
        "status": "prepared_offline_check_pending",
        "study_plan_sha256": STUDY_PLAN_SHA,
        "retrieval_plan_sha256": RETRIEVAL_PLAN_SHA,
        "prompts": rel(prompts),
        "prompts_sha256": PROMPTS_SHA,
        "input_hashes": inputs,
        "source_hashes": {name: a.digest(ROOT / name) for name in SOURCE_FILES},
        "generated_hashes": {
            rel(BASE / name): a.digest(BASE / name)
            for name in ("schedule.json", "analysis_plan.json")
        },
        "environment": core.environment(),
        "api_contract": a.CONTRACT,
        "workers": WORKERS,
        "programme": {
            "stages": 4,
            "distinct_inputs": 1295,
            "pool_prompts": 14245,
            "predictions": 42735,
            "ordinary_responses": 128205,
            "max_logical_calls": 384615,
            "max_physical_POSTs": 1923075,
        },
        "stage_cap_predictions": CAP,
        "predictions_per_input": UNIT,
        "accepted_preparation_evidence_sha256": AUDIT_SHA,
        "model_check": "Each incomplete live-stage attempt saves GET /models and requires pinned Qwen. No fallback or pilot responses.",
        "resume": "Verify all saved selected items; skip complete items and reuse saved partial replies. Unresolved intents, partial writes, fatal/exhausted budgets or saved stops require review, never automatic resending/reset. The inherited stage auditor accepts one uninterrupted attempt; multi-attempt history requires separate review.",
        "stop": "One named stage. Eight workers; stop new submissions on errors or >=10% null after20 complete, drain pending work. No accuracy stopping.",
        "next_stage": "Preceding completion/evidence-bound audit required. Launcher audits one stage then stops; no automatic advance.",
        "scope": "Eleven accepted pool conditions, three fresh repeats each, fixed original messages and abstract context. No old prediction reuse, retrieval, training, gold scoring, TEST/benchmark, Priority3 or LaTeX. Original studies preserved.",
    }
    a.require(
        plan["programme"]["predictions"] == study["budget"]["predictions"]
        and plan["programme"]["ordinary_responses"]
        == study["budget"]["ordinary_responses"],
        "Scientific budget differs",
    )
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
    with patch.object(socket.socket, "connect", preparation.forbid), patch.object(
        socket, "create_connection", preparation.forbid
    ), patch.object(core, "live_client", preparation.forbid):
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

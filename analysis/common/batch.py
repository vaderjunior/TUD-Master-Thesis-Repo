"""Schedule the core study from saved prompts."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time
from analysis.common import api as a
from analysis.common import scoring as s

ROOT = a.ROOT
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
BASE = CORE / "api_execution_v1"
PLAN = BASE / "plan.json"
MAX_ORDINARY = 8100
WORKERS = 8
ENV_PACKAGES = [
    "numpy",
    "pandas",
    "pyarrow",
    "pydantic",
    "PyYAML",
    "requests",
    "urllib3",
    "python-dotenv",
]
SOURCE_FILES = [
    "analysis/common/batch.py",
    "analysis/common/scoring.py",
    "analysis/common/validate_batch.py",
    "tests/common/test_batch.py",
    "tests/common/test_scoring.py",
    "scripts/run_core_comparison.ps1",
    *a.DEPENDENCIES,
]


def rel(p):
    return Path(str(p).removeprefix("\\\\?\\")).relative_to(ROOT).as_posix()


def environment():
    return {
        "python": sys.version.split()[0],
        "packages": {n: importlib.metadata.version(n) for n in ENV_PACKAGES},
    }


def rows(path):
    with a.storage_path(path).open(encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def catalog():
    result = {}
    for path in [
        CORE / "static_prompts.jsonl",
        CORE / "retrieval_v1/full/prompts.jsonl",
        CORE / "retrieval_v1/small/prompts.jsonl",
    ]:
        for entry in rows(path):
            key = (entry["key"], entry["condition"])
            a.require(key not in result, "Duplicate frozen prompt")
            a.require(
                a.signature(entry["messages"]) == entry["messages_sha256"],
                "Frozen prompt signature mismatch",
            )
            result[key] = {
                "messages": entry["messages"],
                "messages_sha256": entry["messages_sha256"],
            }
    a.require(len(result) == 10758, "Wrong prompt catalogue size")
    return result


def make_schedule(dev, cat, cap=MAX_ORDINARY):
    """Keep each input's complete condition/repeat comparison in one stage."""
    bysubset = defaultdict(list)
    available = defaultdict(list)
    for key, condition in cat:
        available[key].append(condition)
    for row in dev:
        a.require(
            len(row["scopes"]) == 1,
            "Current frozen rosters must be disjoint; amend schedule if that changes",
        )
        bysubset[row["scopes"][0]["subset"]].append(row["key"])
    phases = [
        ("primary", ["en_dev_eval_main", "de_legal_dev_eval"]),
        (
            "secondary",
            [
                "en_dev_eval_sq3_types",
                "en_dev_eval_targets",
                "en_dev_eval_severity",
                "de_dev_eval",
            ],
        ),
        ("sensitivity", ["en_dev_eval_types"]),
    ]
    stages = []
    pending = []
    budget = 0

    def finish(phase):
        nonlocal pending, budget
        if not pending:
            return
        number = len(stages) + 1
        random.Random(20260915 + number).shuffle(pending)
        stages.append(
            {
                "id": f"stage_{number:02d}_{phase}",
                "phase": phase,
                "jobs": pending,
                "items": len(pending),
                "ordinary_responses": budget,
                "max_logical_calls": len(pending) * 9,
                "max_physical_POSTs": len(pending) * 45,
                "inputs": len({j["key"] for j in pending}),
            }
        )
        pending = []
        budget = 0

    for phase, subsets in phases:
        for index in range(max(map(lambda name: len(bysubset[name]), subsets))):
            for subset in subsets:
                if index >= len(bysubset[subset]):
                    continue
                key = bysubset[subset][index]
                conditions = sorted(available[key])
                expected = 6 if subset == "en_dev_eval_main" else 9
                a.require(
                    len(conditions) == expected, "Wrong per-input condition count"
                )
                unit = [
                    {
                        "key": key,
                        "condition": condition,
                        "repeat": repeat,
                        "subset": subset,
                    }
                    for condition in conditions
                    for repeat in range(3)
                ]
                cost = len(unit) * 3
                a.require(cost <= cap, "Stage cap cannot hold one complete comparison")
                if pending and budget + cost > cap:
                    finish(phase)
                pending.extend(unit)
                budget += cost
        finish(phase)
    jobs = [j for stage in stages for j in stage["jobs"]]
    identities = {(j["key"], j["condition"], j["repeat"]) for j in jobs}
    a.require(
        len(jobs) == len(identities) == len(cat) * 3,
        "Schedule repeats or omits a prediction",
    )
    a.require(
        identities == {(key, cond, r) for key, cond in cat for r in range(3)},
        "Schedule does not cover frozen catalogue",
    )
    return stages


def bounded_jobs(jobs, work, workers=8, initial=None, on_result=None):
    """At most workers submitted; after first error/quality stop only drain."""
    stats = Counter(initial or {})
    results = []
    errors = []
    stop = False
    index = 0
    peak = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        active = {}
        while active or (index < len(jobs) and (not stop)):
            while not stop and index < len(jobs) and (len(active) < workers):
                job = jobs[index]
                index += 1
                active[executor.submit(work, job)] = job
                peak = max(peak, len(active))
            if not active:
                break
            done, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in done:
                job = active.pop(future)
                try:
                    out = future.result()
                    results.append((job, out))
                    stats["complete"] += 1
                    stats["null"] += out["result"] is None
                    if on_result:
                        on_result(job, out, stats)
                    if (
                        stats["complete"] >= 20
                        and stats["null"] / stats["complete"] >= 0.1
                    ):
                        stop = True
                except Exception as exc:
                    errors.append(
                        {
                            "job": job,
                            "error_type": type(exc).__name__,
                            "reason": (
                                str(exc)
                                if type(exc) in (a.CallStopped, a.IncompleteAttempt)
                                else "Inspect saved artifacts; exception detail not logged to avoid credential leakage"
                            ),
                        }
                    )
                    stop = True
    return {
        "results": results,
        "errors": errors,
        "stats": dict(stats),
        "submitted": index,
        "peak_submitted": peak,
        "stopped": stop,
        "unsubmitted": len(jobs) - index,
    }


def prepare():
    import pandas as pd
    import yaml

    a.require(not PLAN.exists(), "Preparation is frozen; use validate, never overwrite")
    core = a.read(CORE / "plan.json")
    dev = rows(CORE / "dev_inputs.jsonl")
    cat = catalog()
    a.require(
        a.digest(CORE / "plan.json")
        == "5c610f4fe320b5e0f185315933d47b388ab645b20b52eed738541a1579184e4e",
        "Core changed",
    )
    transport = CORE / "api_transport_v2/offline_complete.json"
    tr = a.read(transport)
    a.require(
        tr["evidence_sha256"]
        == "a51a90d3f5471b8bf4ae4b7e180bbf719fe8aa8c61551f0b362cada4f8a6f048"
        and a.digest(ROOT / tr["evidence"]) == tr["evidence_sha256"],
        "Transport preflight differs",
    )
    for name, h in a.read(ROOT / tr["evidence"])["verified_input_hashes"].items():
        a.require(a.digest(ROOT / name) == h, "Transport input/source changed")
    retrieval = CORE / "retrieval_v1/review_complete_v2.json"
    rr = a.read(retrieval)
    a.require(
        a.digest(ROOT / rr["evidence"]) == rr["evidence_sha256"],
        "Retrieval audit changed",
    )
    for pool, h in rr["prompt_hashes"].items():
        a.require(
            a.digest(CORE / f"retrieval_v1/{pool}/prompts.jsonl") == h,
            "RAG messages changed",
        )
    for name, h in core["generated_hashes"].items():
        a.require(a.digest(ROOT / name) == h, "Core generated data changed")
    paths = [
        CORE / "plan.json",
        CORE / "dev_inputs.jsonl",
        CORE / "static_prompts.jsonl",
        retrieval,
        ROOT / rr["evidence"],
        transport,
        ROOT / tr["evidence"],
        CORE / "retrieval_v1/full/prompts.jsonl",
        CORE / "retrieval_v1/small/prompts.jsonl",
        ROOT / "config/config.yaml",
    ]
    encoder_path = (
        ROOT / "Thesis_Log/astra_result/evidence/core_encoder_results_20260915_v5.json"
    )
    encoder = a.read(encoder_path)
    a.require(
        a.digest(encoder_path)
        == "0951038b8b96e06f2b52d00f8b58c79cf14a4472c9fd8a2144f3f98047ce1344",
        "Aligned encoder audit changed",
    )
    paths.append(encoder_path)
    for group in ("english_assembled", "retained_german_assembled"):
        for name, info in encoder[group].items():
            a.require(
                a.digest(ROOT / name) == info["sha256"],
                "Aligned encoder predictions changed",
            )
            paths.append(ROOT / name)
    taxonomy = yaml.safe_load(
        (ROOT / "config/taxonomy.yaml").read_text(encoding="utf8")
    )
    endpoints = {}
    for subset in s.ENDPOINTS:
        path = ROOT / f"data/processed/{subset}.parquet"
        df = pd.read_parquet(path)
        paths.append(path)
        a.require(len(df) == core["scope_counts"][subset], "Endpoint roster changed")
        endpoints[subset] = s.roster(subset, df.to_dict("records"), taxonomy)
        expected = [r["item_id"] for r in dev if r["scopes"][0]["subset"] == subset]
        a.require(
            list(df.id.astype(str)) == expected, "Endpoint order/identity differs"
        )
    stages = make_schedule(dev, cat)
    budget = sum((x["ordinary_responses"] for x in stages))
    a.require(
        budget == core["budget"]["ordinary_model_votes"] == 96822,
        "Programme budget changed",
    )
    pilot = []
    for subset in ("en_dev_eval_main", "de_legal_dev_eval"):
        key = next((r["key"] for r in dev if r["scopes"][0]["subset"] == subset))
        pilot.extend(
            (
                {"key": key, "condition": c, "repeat": 0, "subset": subset}
                for k, c in sorted(cat)
                if k == key
            )
        )
    a.require(
        len(pilot) == 15,
        "Pilot must cover the first primary EN/DE input across all conditions",
    )
    scoring = {"contract": s.contract(), "endpoints": endpoints}
    schedule = {
        "stages": stages,
        "pilot_jobs": pilot,
        "ordering": "Round-robin primary rosters, then secondary rosters, then sensitivity; complete input comparison kept in one stage; fixed within-stage shuffle seed 20260915 + stage number.",
    }
    a.save_new(BASE / "scoring_plan.json", scoring)
    a.save_new(BASE / "schedule.json", schedule)
    plan = {
        "created_utc": a.utc(),
        "status": "execution_prepared_live_pilot_and_review_pending",
        "core_sha256": a.digest(CORE / "plan.json"),
        "input_hashes": {rel(p): a.digest(p) for p in paths},
        "source_hashes": {name: a.digest(ROOT / name) for name in SOURCE_FILES},
        "generated_hashes": {
            rel(BASE / name): a.digest(BASE / name)
            for name in ("schedule.json", "scoring_plan.json")
        },
        "environment": environment(),
        "api_contract": a.CONTRACT,
        "workers": WORKERS,
        "stage_cap_ordinary": MAX_ORDINARY,
        "programme": {
            "stages": len(stages),
            "items": sum((x["items"] for x in stages)),
            "ordinary_responses": budget,
            "max_logical_calls": budget * 3,
            "max_physical_POSTs": budget * 15,
        },
        "pilot": {
            "items": 15,
            "ordinary_responses": 45,
            "max_logical_calls": 135,
            "max_physical_POSTs": 675,
            "reuse": "Actual planned repeat-0 predictions, included once in programme totals; selected before outputs, not resampled on invalidity.",
        },
        "stop_policy": "One named stage only. At most eight in flight. Stop new submissions on any transport/integrity error or >=10% all-null completed items after at least 20 completed in the stage; drain in-flight items. Preserve a sticky stop record for review. No F1-based stopping.",
        "resume_policy": "Verify completed item receipts and skip. Reuse saved replies for partial items. Unresolved intents and exhausted/fatal calls require explicit review; budgets never reset. Completed stage rerun is read-only and does not advance to the next stage.",
        "scoring_policy": s.contract(),
        "scope": "Core DEV programme only. No fresh retrieval, encoder training, outcome-driven tuning, Priority 3, TEST/benchmark work or LaTeX.",
    }
    a.save_new(PLAN, plan)
    print(
        json.dumps(
            {
                "prepared": rel(PLAN),
                "plan_sha256": a.digest(PLAN),
                "programme": plan["programme"],
                "stages": [{k: v for k, v in x.items() if k != "jobs"} for x in stages],
                "endpoints": {
                    k: {x: y for x, y in v.items() if x != "scorable_ids"}
                    for k, v in endpoints.items()
                },
            },
            indent=2,
        ),
        flush=True,
    )


def validate():
    plan = a.read(PLAN)
    a.require(
        plan["environment"] == environment(), "API/scoring library environment changed"
    )
    for group in ("input_hashes", "source_hashes", "generated_hashes"):
        for name, h in plan[group].items():
            a.require(a.digest(ROOT / name) == h, f"Frozen artifact changed: {name}")
    a.require(
        plan["api_contract"] == a.CONTRACT and plan["workers"] == 8,
        "Execution contract changed",
    )
    return plan


def identity(job):
    return {
        "experiment_sha256": a.digest(PLAN),
        "condition": job["condition"],
        "repeat": job["repeat"],
        "key": job["key"],
    }


def item_dir(root, job):
    return a.storage_path(root) / "items" / a.signature(identity(job))


def live_client():
    from dotenv import dotenv_values

    values = dotenv_values(ROOT / ".env")
    url = os.environ.get("PEASEC_URL") or values.get("PEASEC_URL")
    token = os.environ.get("PEASEC_TOKEN") or values.get("PEASEC_TOKEN")
    a.require(
        isinstance(url, str) and isinstance(token, str),
        "PEASEC credentials are unavailable",
    )
    return a.RequestsTransport(url, token)


def run_jobs(jobs, cat, root, attempt, client_factory=live_client, mode="live"):
    import threading

    local = threading.local()
    existing = []
    todo = []
    for job in jobs:
        if (item_dir(root, job) / "complete.json").exists():
            a.audit_item(
                root,
                identity(job),
                cat[job["key"], job["condition"]]["messages"],
                execution_mode=mode,
            )
            existing.append((job, a.read(item_dir(root, job) / "result.json")))
        else:
            todo.append(job)
    initial = {
        "complete": len(existing),
        "null": sum((x["result"] is None for _, x in existing)),
    }
    a.require(
        not (
            initial["complete"] >= 20 and initial["null"] / initial["complete"] >= 0.1
        ),
        "Existing null rate requires review",
    )

    def work(job):
        if not hasattr(local, "client"):
            local.client = client_factory()
        return a.execute_item(
            root,
            identity(job),
            cat[job["key"], job["condition"]]["messages"],
            local.client,
            execution_mode=mode,
        )

    def progress(job, out, stats):
        if stats["complete"] % 50 == 0:
            print(
                f"Saved {stats['complete']}/{len(jobs)} items; {stats['null']} null results",
                flush=True,
            )

    completed = bounded_jobs(todo, work, WORKERS, initial, progress)
    finished = existing + completed.pop("results")
    completed["saved_items"] = len(finished)
    completed["null_items"] = sum((x["result"] is None for _, x in finished))
    completed["completed_item_accounting"] = {
        k: sum((x["accounting"][k] for _, x in finished))
        for k in (
            "ordinary_logical_calls",
            "repair_logical_calls",
            "post_attempts",
            "retry_attempts",
        )
    }
    completed["finished_utc"] = a.utc()
    a.save_new(attempt / "summary.json", completed)
    return completed


def preflight():
    import base64
    import requests

    plan = validate()
    cat = catalog()
    root = a.storage_path(BASE / "live")
    schedule = a.read(BASE / "schedule.json")
    jobs = schedule["pilot_jobs"]
    offline = a.read(BASE / "offline_check_complete.json")
    a.require(
        offline["plan_sha256"] == a.digest(PLAN)
        and offline["status"] == "offline_batch_checked_live_pilot_pending",
        "Independent offline execution check is required",
    )
    with a.item_lock(a.storage_path(BASE / "launcher.lock")):
        a.require(
            not (BASE / "live_preflight.json").exists(),
            "Pilot already recorded; use audit-preflight",
        )
        attempt = a.storage_path(BASE / f"preflight_attempts/{time.time_ns()}")
        client = live_client()
        a.save_new(
            attempt / "intent.json",
            {
                "started_utc": a.utc(),
                "plan_sha256": a.digest(PLAN),
                "action": "GET PEASEC /models; then 15 planned repeat-0 items",
            },
        )
        try:
            response = requests.get(
                client._endpoint.removesuffix("/chat/completions") + "/models",
                headers={"Authorization": "Bearer " + client._token},
                timeout=30,
                allow_redirects=False,
            )
        except requests.exceptions.RequestException as exc:
            a.save_new(
                attempt / "roster_error.json", {"error_type": type(exc).__name__}
            )
            raise RuntimeError(
                "Model roster request failed; inspect saved attempt"
            ) from None
        a.save_new(
            attempt / "roster_response.json",
            {
                "status": response.status_code,
                "body_base64": base64.b64encode(response.content).decode("ascii"),
                "body_sha256": __import__("hashlib")
                .sha256(response.content)
                .hexdigest(),
            },
        )
        a.require(response.status_code == 200, "PEASEC model roster request refused")
        models = response.json()
        a.require(
            a.MODEL in {m["id"] for m in models.get("data", [])},
            "Pinned Qwen model unavailable; no fallback",
        )
        result = run_jobs(jobs, cat, root, attempt)
        a.require(
            not result["stopped"] and result["saved_items"] == 15,
            "Pilot stopped; review all saved attempts before continuing",
        )
        a.save_new(
            BASE / "live_preflight.json",
            {
                "plan_sha256": a.digest(PLAN),
                "attempt": rel(attempt),
                "summary_sha256": a.digest(attempt / "summary.json"),
                "roster_sha256": a.digest(attempt / "roster_response.json"),
                "status": "pilot_finished_independent_review_pending",
            },
        )
        print(
            json.dumps(
                {
                    "status": "pilot_finished_independent_review_pending",
                    "summary": result,
                },
                indent=2,
            ),
            flush=True,
        )


def audit_preflight():
    plan = validate()
    record = a.read(BASE / "live_preflight.json")
    cat = catalog()
    root = a.storage_path(BASE / "live")
    a.require(record["plan_sha256"] == a.digest(PLAN), "Pilot plan changed")
    attempt = ROOT / record["attempt"]
    a.require(
        a.digest(attempt / "summary.json") == record["summary_sha256"]
        and a.digest(attempt / "roster_response.json") == record["roster_sha256"],
        "Pilot receipt changed",
    )
    audits = []
    counts = Counter()
    latencies = []
    hashes = {}
    jobs = a.read(BASE / "schedule.json")["pilot_jobs"]
    for job in jobs:
        ident = identity(job)
        audits.append(
            a.audit_item(
                root,
                ident,
                cat[job["key"], job["condition"]]["messages"],
                execution_mode="live",
            )
        )
        d = item_dir(root, job)
        result = a.read(d / "result.json")
        a.require(
            result["n_valid"] > 0,
            "Pilot contains an all-null item; preserve for review, do not resample",
        )
        a.require(
            result["active_model"] == a.MODEL and result["execution_mode"] == "live",
            "Wrong pilot attribution",
        )
        latencies.extend(result["latencies"])
        counts.update(result["accounting"])
        a.require(
            len(list(d.rglob("intent.json")))
            == len(list(d.rglob("outcome.json")))
            == result["accounting"]["post_attempts"],
            "Pilot response accounting incomplete",
        )
        hashes[rel(d / "complete.json")] = a.digest(d / "complete.json")
    seal = {
        "created_utc": a.utc(),
        "status": "live_pilot_audited_first_stage_ready",
        "plan_sha256": a.digest(PLAN),
        "live_preflight_sha256": a.digest(BASE / "live_preflight.json"),
        "pilot_item_receipt_hashes": hashes,
        "audited_items": len(audits),
        "accounting": dict(counts),
        "saved_API_latency_seconds_sum": sum(latencies),
        "mean_vote_API_seconds": sum(latencies) / len(latencies),
        "actual_gold_scoring": False,
        "qualification": "Small live service/response check only; model name is requested and checked when returned, not a backend checkpoint guarantee. Full-run audit still required. No F1 was computed.",
    }
    a.save_new(BASE / "ready.json", seal)
    print(json.dumps(seal, indent=2), flush=True)


def run_stage(stage_id):
    plan = validate()
    schedule = a.read(BASE / "schedule.json")
    stages = schedule["stages"]
    selected = next((x for x in stages if x["id"] == stage_id), None)
    a.require(selected is not None, "Unknown frozen stage")
    root = a.storage_path(BASE / "live")
    target = a.storage_path(BASE / "stages" / stage_id)
    with a.item_lock(a.storage_path(BASE / "launcher.lock")):
        if (target / "complete.json").exists():
            saved = a.read(target / "complete.json")
            a.require(
                saved["plan_sha256"] == a.digest(PLAN), "Completed stage plan changed"
            )
            a.require(
                saved["stage"] == stage_id
                and set(saved["item_receipt_hashes"])
                == {rel(item_dir(root, j) / "complete.json") for j in selected["jobs"]},
                "Completed stage inventory differs",
            )
            for name, h in saved["item_receipt_hashes"].items():
                a.require(a.digest(ROOT / name) == h, "Completed item receipt changed")
                receipt = a.read(ROOT / name)
                d = a.storage_path(ROOT / name).parent
                a.verify_inventory(root, d, receipt["hashes"])
            print(
                "THIS STAGE IS ALREADY COMPLETE. No requests sent; next stage was not started.",
                flush=True,
            )
            return
        a.require(
            not list(target.glob("stop_*.json")),
            "This stage has a preserved stop; ask Astra to inspect before resuming",
        )
        ready = a.read(BASE / "ready.json")
        a.require(
            ready["plan_sha256"] == a.digest(PLAN)
            and ready["status"] == "live_pilot_audited_first_stage_ready",
            "Live readiness does not match execution plan",
        )
        a.require(
            ready["live_preflight_sha256"] == a.digest(BASE / "live_preflight.json"),
            "Pilot receipt changed after review",
        )
        for name, h in ready["pilot_item_receipt_hashes"].items():
            a.require(
                a.digest(ROOT / name) == h, "Pilot item receipt changed after review"
            )
        index = stages.index(selected)
        if index:
            a.require(
                (BASE / "stages" / stages[index - 1]["id"] / "complete.json").exists(),
                "Finish the preceding stage before this one",
            )
        cat = catalog()
        attempt = target / f"attempt_{time.time_ns()}"
        a.save_new(
            attempt / "start.json",
            {
                "plan_sha256": a.digest(PLAN),
                "stage": stage_id,
                "started_utc": a.utc(),
                "environment": environment(),
            },
        )
        print(
            f"STAGE {stage_id}: {selected['items']} planned items, {selected['ordinary_responses']} ordinary responses before repairs/retries. Completed items are checked and skipped.",
            flush=True,
        )
        result = run_jobs(selected["jobs"], cat, root, attempt)
        if result["stopped"] or result["saved_items"] != selected["items"]:
            a.save_new(
                target / f"stop_{time.time_ns()}.json",
                {
                    "plan_sha256": a.digest(PLAN),
                    "attempt": rel(attempt),
                    "summary": result,
                },
            )
            raise RuntimeError(
                "Stage stopped; preserve all files and ask Astra to inspect. No next stage started."
            )
        hashes = {
            rel(item_dir(root, j) / "complete.json"): a.digest(
                item_dir(root, j) / "complete.json"
            )
            for j in selected["jobs"]
        }
        a.save_new(
            target / "complete.json",
            {
                "plan_sha256": a.digest(PLAN),
                "stage": stage_id,
                "completed_utc": a.utc(),
                "item_receipt_hashes": hashes,
                "summary": result,
                "next": "Stop here and ask Astra to audit results before another named stage.",
            },
        )
        print(
            f"STAGE COMPLETE: {stage_id}. Tell Astra to inspect. No next stage was started.",
            flush=True,
        )


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "prepare",
            "validate",
            "status",
            "preflight",
            "audit-preflight",
            "run",
        ],
    )
    parser.add_argument("--stage", default="stage_01_primary")
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "status":
        p = a.read(PLAN)
        print(
            json.dumps(
                {
                    "programme": p["programme"],
                    "live_pilot": (BASE / "live_preflight.json").exists(),
                    "ready": (BASE / "ready.json").exists(),
                    "completed_stages": [
                        x.name
                        for x in (BASE / "stages").glob("*")
                        if (x / "complete.json").exists()
                    ],
                },
                indent=2,
            )
        )
    elif args.mode == "validate":
        p = validate()
        print(
            json.dumps(
                {"validated": True, "programme": p["programme"], "no_API_calls": True},
                indent=2,
            )
        )
    elif args.mode == "preflight":
        preflight()
    elif args.mode == "audit-preflight":
        audit_preflight()
    else:
        run_stage(args.stage)


if __name__ == "__main__":
    main()

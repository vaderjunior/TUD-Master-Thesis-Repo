import argparse
from collections import Counter
from datetime import datetime, timezone
import base64
import hashlib
import json
import os
from pathlib import Path
import socket
import threading
import time
from unittest.mock import patch
from analysis.common import api as a
from analysis.common import batch as core
from analysis.common import schema as isolated
from analysis.feedback.supplements import recovery_support as recovery

ROOT = a.ROOT
BASE = ROOT / "experiments/astra/sq3_additions_20261004_v1/execution"
PLAN = BASE / "plan.json"
WORKERS = 8
STUDIES = {
    "factorial": {
        "root": BASE.parent / "factorial",
        "inputs": 227,
        "conditions": 4,
        "batches": 1,
        "predictions": 2724,
    },
    "human_meaning": {
        "root": BASE.parent / "human_meaning_v2",
        "inputs": 558,
        "conditions": 10,
        "batches": 2,
        "predictions": 16740,
    },
    "benign_selection": {
        "root": BASE.parent / "benign_selection",
        "inputs": 1024,
        "conditions": 1,
        "batches": 1,
        "predictions": 3072,
    },
}
SOURCE_FILES = list(
    dict.fromkeys(
        [
            *core.SOURCE_FILES,
            "analysis/common/schema.py",
            "analysis/common/prompt_archive.py",
            "analysis/common/prompt.py",
            "analysis/feedback/supplements/run.py",
            "analysis/feedback/supplements/audit.py",
            "analysis/feedback/supplements/validate.py",
            "tests/feedback/summary/test_additions.py",
            "tests/feedback/supplements/test_science.py",
            "analysis/feedback/balanced/selection_pools.py",
            "scripts/run_feedback_supplements.ps1",
        ]
    )
)


def rel(path):
    return isolated.rel(path)


def save(path, value):
    p = a.storage_path(path).resolve()
    a.require(
        p.is_relative_to(a.storage_path(BASE).resolve()),
        "Write outside isolated execution",
    )
    a.save_new(p, value)


def forbidden(*args, **kwargs):
    raise AssertionError("Offline operation: no network or live client")


def bind(target, name, sha):
    a.require(
        name not in target or target[name] == sha, "Conflicting source binding: " + name
    )
    target[name] = sha


def verify(bindings):
    for name, expected in bindings.items():
        a.require(a.digest(ROOT / name) == expected, "Frozen file changed: " + name)


def test_receipt():
    hashes = {n: a.digest(ROOT / n) for n in SOURCE_FILES}
    return BASE / "development" / ("tests_" + a.signature(hashes) + ".json")


def engines(cat):
    result = {}
    for v in cat.values():
        if v["schema_path"] not in result:
            backend = recovery.adapter(ROOT / v["schema_path"])
            a.require(backend.taxonomy_sha256 == v["schema_sha256"], "Schema changed")
            result[v["schema_path"]] = backend
    return result


def identity(job, plan_sha):
    return {
        "experiment_sha256": plan_sha,
        "condition": job["study"] + "::" + job["condition"],
        "repeat": job["repeat"],
        "key": job["key"],
    }


def item_dir(root, job, plan_sha):
    return a.storage_path(root) / "items" / a.signature(identity(job, plan_sha))


def scan_jobs(jobs, cat, root, plan_sha, mode, backends):
    existing = []
    todo = []
    for j in jobs:
        entry = cat[j["key"], j["condition"]]
        backend = backends[j["schema_path"]]
        a.require(
            all(
                (
                    entry[k] == j[k]
                    for k in [
                        "messages_sha256",
                        "schema_path",
                        "schema_sha256",
                        "population",
                    ]
                )
            ),
            "Job/prompt/schema identity differs",
        )
        d = item_dir(root, j, plan_sha)
        ident = identity(j, plan_sha)
        if (d / "complete.json").exists():
            backend.audit_item(root, ident, entry["messages"], execution_mode=mode)
            existing.append((j, a.read(d / "result.json")))
            continue
        if d.exists():
            a.require(
                not list(d.rglob("*.partial_*")), "Partial atomic write requires review"
            )
            expected = {
                "identity": ident,
                "messages_sha256": entry["messages_sha256"],
                "contract": a.CONTRACT,
                "implementation": backend.implementation(),
                "execution_mode": mode,
            }
            a.require(a.read(d / "job.json") == expected, "Partial job stamp changed")
            for p in d.rglob("intent.json"):
                if not p.with_name("outcome.json").exists():
                    raise a.IncompleteAttempt(
                        "Saved intent without outcome: stop without resending"
                    )
                intent = a.read(p)
                out = a.read(p.with_name("outcome.json"))
                a.require(
                    intent["job_stamp"] == a.signature(expected)
                    and out["intent_sha256"] == a.signature(intent),
                    "Partial request attribution differs",
                )
                recovery.partial_outcome(root, d, intent, out)
            a.require(
                all(
                    (
                        p.with_name("intent.json").exists()
                        for p in d.rglob("outcome.json")
                    )
                ),
                "Orphan saved response",
            )
        todo.append(j)
    return (existing, todo)


def availability(attempt, client):
    import requests

    save(
        attempt / "model_check_intent.json",
        {"started_utc": a.utc(), "action": "GET PEASEC /models", "model": a.MODEL},
    )
    started = time.perf_counter()
    try:
        reply = requests.get(
            client._endpoint.removesuffix("/chat/completions") + "/models",
            headers={"Authorization": "Bearer " + client._token},
            timeout=30,
            allow_redirects=False,
        )
    except requests.exceptions.RequestException as exc:
        save(
            attempt / "model_check_error.json",
            {
                "error_type": type(exc).__name__,
                "elapsed_s": time.perf_counter() - started,
            },
        )
        raise a.CallStopped(
            "Model availability request failed; category saved"
        ) from None
    save(
        attempt / "model_check_response.json",
        {
            "status": reply.status_code,
            "elapsed_s": time.perf_counter() - started,
            "body_base64": base64.b64encode(reply.content).decode(),
            "body_sha256": hashlib.sha256(reply.content).hexdigest(),
        },
    )
    a.require(reply.status_code == 200, "Model roster refused; no classification sent")
    try:
        ids = {r["id"] for r in reply.json()["data"]}
    except (ValueError, KeyError, TypeError):
        raise a.CallStopped("Malformed model roster") from None
    a.require(a.MODEL in ids, "Pinned model unavailable; no fallback")


def checked_root(root, study, mode):
    root = a.storage_path(root).resolve()
    allowed = a.storage_path(
        BASE / ("live" if mode == "live" else "offline_check")
    ).resolve()
    a.require(
        mode in ("mock", "live")
        and (
            root == allowed / study if mode == "live" else root.is_relative_to(allowed)
        ),
        "Output/mode boundary",
    )
    return root


def execute_stage(
    stage,
    cat,
    root,
    plan_sha,
    *,
    mode,
    client_factory,
    sleep=time.sleep,
    availability_check=availability,
):
    root = checked_root(root, stage["study"], mode)
    target = root / "stages" / stage["id"]
    backends = engines(cat)
    with a.item_lock(root / "launcher.lock"):
        if (target / "complete.json").exists():
            record = a.read(target / "complete.json")
            recovery.verify_stops(root, target)
            a.require(
                record["plan_sha256"] == plan_sha
                and record["stage"] == stage["id"]
                and (record["study"] == stage["study"])
                and (record["mode"] == mode),
                "Stage completion attribution changed",
            )
            expected = {
                item_dir(root, j, plan_sha).relative_to(root).as_posix()
                + "/complete.json"
                for j in stage["jobs"]
            }
            a.require(
                set(record["item_receipt_hashes"]) == expected,
                "Completed stage roster changed",
            )
            final_summary = ROOT / record["final_attempt"] / "summary.json"
            a.require(
                a.digest(final_summary) == record["summary_sha256"]
                and a.read(final_summary) == record["summary"],
                "Completed attempt summary changed",
            )
            for name, expected in record["item_receipt_hashes"].items():
                a.require(
                    a.digest(root / name) == expected, "Item completion receipt changed"
                )
            existing, todo = scan_jobs(
                stage["jobs"], cat, root, plan_sha, mode, backends
            )
            a.require(
                not todo and len(existing) == stage["items"],
                "Completed stage incomplete",
            )
            print(f"VERIFIED COMPLETE {stage['id']}: no new request", flush=True)
            return
        recovery.verify_stops(root, target)
        attempt = target / f"attempt_{time.time_ns()}"
        save(
            attempt / "start.json",
            {
                "plan_sha256": plan_sha,
                "stage": stage["id"],
                "study": stage["study"],
                "mode": mode,
                "started_utc": a.utc(),
            },
        )
        try:
            existing, todo = scan_jobs(
                stage["jobs"], cat, root, plan_sha, mode, backends
            )
            save(
                attempt / "inventory.json",
                {
                    "completed": len(existing),
                    "pending": len(todo),
                    "pending_identity_hashes": [
                        a.signature(identity(j, plan_sha)) for j in todo
                    ],
                },
            )
            print(
                f"{stage['id']}: {len(existing)} verified, {len(todo)} pending; {stage['ordinary_calls']} ordinary calls planned in this batch.",
                flush=True,
            )
            if todo and mode == "live":
                availability_check(attempt, client_factory())
            initial = {
                "complete": len(existing),
                "null": sum((v["result"] is None for _, v in existing)),
            }
            a.require(
                not (
                    initial["complete"] >= 20
                    and initial["null"] / initial["complete"] >= 0.1
                ),
                "Existing invalid-output rate requires review",
            )
            local = threading.local()

            def work(j):
                if not hasattr(local, "client"):
                    local.client = client_factory()
                return backends[j["schema_path"]].execute_item(
                    root,
                    identity(j, plan_sha),
                    cat[j["key"], j["condition"]]["messages"],
                    local.client,
                    sleep=sleep,
                    execution_mode=mode,
                )

            def progress(j, value, stats):
                if stats["complete"] % 100 == 0:
                    print(
                        f"Saved {stats['complete']}/{stage['items']} predictions; {stats['null']} null",
                        flush=True,
                    )

            result = core.bounded_jobs(todo, work, WORKERS, initial, progress)
            completed = existing + result.pop("results")
            result.update(
                saved_items=len(completed),
                null_items=sum((v["result"] is None for _, v in completed)),
                finished_utc=a.utc(),
                completed_item_accounting={
                    k: sum((v["accounting"][k] for _, v in completed))
                    for k in [
                        "ordinary_logical_calls",
                        "repair_logical_calls",
                        "post_attempts",
                        "retry_attempts",
                    ]
                },
            )
            save(attempt / "summary.json", result)
            a.require(
                not result["stopped"]
                and (not result["errors"])
                and (result["saved_items"] == stage["items"]),
                "Stage stopped; preserve saved requests",
            )
        except KeyboardInterrupt:
            save(
                attempt / "interrupted.json",
                {
                    "finished_utc": a.utc(),
                    "reason": "KeyboardInterrupt; resumption must inspect all saved requests before any network access",
                },
            )
            raise
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
                        else "Inspect saved artifacts; exception details omitted to protect credentials"
                    ),
                },
            )
            raise
        hashes = {
            item_dir(root, j, plan_sha).relative_to(root).as_posix()
            + "/complete.json": a.digest(item_dir(root, j, plan_sha) / "complete.json")
            for j in stage["jobs"]
        }
        save(
            target / "complete.json",
            {
                "plan_sha256": plan_sha,
                "stage": stage["id"],
                "study": stage["study"],
                "mode": mode,
                "completed_utc": a.utc(),
                "final_attempt": rel(attempt),
                "summary_sha256": a.digest(attempt / "summary.json"),
                "summary": result,
                "item_receipt_hashes": hashes,
            },
        )
        print(
            f"{stage['id']} calls complete. Automatic offline audit follows; leave the launcher running.",
            flush=True,
        )


def _original_verify_review(stage, root, plan_sha, mode):
    target = a.storage_path(root) / "stages" / stage["id"]
    rp = target / "review.json"
    r = a.read(rp)
    a.require(
        r["status"] == "audited"
        and r["plan_sha256"] == plan_sha
        and (r["stage"] == stage["id"])
        and (r["mode"] == mode),
        "Audit identity differs",
    )
    a.require(
        r["complete_sha256"] == a.digest(target / "complete.json"),
        "Audited stage completion changed",
    )
    for name, h in r["artifacts"].items():
        a.require(a.digest(target / name) == h, "Stage audit artifact changed")
    return r


def verify_review(stage, root, plan_sha, mode):
    return recovery.verify_review(stage, root, plan_sha, mode, _original_verify_review)


def run_sequence(stages, run, audit, reviewed, verify_one):
    for stage in stages:
        run(stage)
        if not reviewed(stage):
            audit(stage)
        verify_one(stage)


def study_inputs(study):
    root = STUDIES[study]["root"]
    inputs = {}
    done = a.read(root / "complete.json")
    prepared = a.read(root / "prepared.json")
    a.require(
        done["status"] == "offline_preparation_independently_verified"
        and done["prepared_sha256"] == a.digest(root / "prepared.json"),
        "Study requires independent verification",
    )
    for name, h in a.read(root / "preparation_plan.json")["inputs"].items():
        bind(inputs, name, h)
    for name, h in prepared["outputs"].items():
        bind(inputs, name, h)
    for name in [
        "complete.json",
        "prepared.json",
        "preparation_plan.json",
        "scientific_plan_v1.json",
        "STUDY_PROTOCOL.md",
    ]:
        bind(inputs, rel(root / name), a.digest(root / name))
    a.require(
        done["verification_sha256"] == a.digest(ROOT / done["verification"]),
        "Preparation verification changed",
    )
    bind(inputs, done["verification"], done["verification_sha256"])
    verify(inputs)
    return inputs


def load_catalog(study):
    info = STUDIES[study]
    root = info["root"]
    science = a.read(root / "scientific_plan_v1.json")
    roster = {r["key"]: r for r in core.rows(root / "roster.jsonl")}
    conditions = science["conditions"]
    a.require(
        len(roster) == info["inputs"]
        and len(conditions) == len(set(conditions)) == info["conditions"],
        "Wrong scientific roster",
    )
    science_sha = a.digest(root / "scientific_plan_v1.json")
    schema = ROOT / "config/taxonomy.yaml"
    schema_sha = a.digest(schema)
    cat = {}
    for row in core.rows(root / "prompts.jsonl"):
        key = (row["key"], row["condition"])
        a.require(
            key not in cat and key[0] in roster and (key[1] in conditions),
            "Unexpected prompt",
        )
        a.require(
            row["scientific_plan_sha256"] == science_sha
            and a.signature(row["messages"]) == row["messages_sha256"]
            and (a.signature(row["context"]) == row["context_sha256"]),
            "Prompt attribution changed",
        )
        a.require(0 < row["reference_input_tokens"] <= 8192, "Unreviewed input tokens")
        a.require(
            [m["role"] for m in row["messages"]] == ["system", "user"],
            "Unexpected conversation",
        )
        a.body(row["messages"])
        cat[key] = {
            "messages": row["messages"],
            "messages_sha256": row["messages_sha256"],
            "schema_path": rel(schema),
            "schema_sha256": schema_sha,
            "population": roster[key[0]]["population"],
            "entry_sha256": a.signature(row),
        }
    a.require(
        set(cat) == {(k, cond) for k in roster for cond in conditions},
        "Incomplete scientific matrix",
    )
    return cat


def make_schedule(catalogs):
    stages = []
    for study, info in STUDIES.items():
        cat = catalogs[study]
        keys = sorted(
            {k for k, _ in cat},
            key=lambda k: hashlib.sha256(
                ("20261004-additions|" + study + "|" + k).encode()
            ).hexdigest(),
        )
        conditions = sorted({cond for _, cond in cat})
        a.require(len(keys) % info["batches"] == 0, "Unbalanced batch sizes")
        width = len(keys) // info["batches"]
        for batch in range(info["batches"]):
            group = keys[batch * width : (batch + 1) * width]
            jobs = []
            for ordinal, key in enumerate(group, batch * width):
                for block in range(3):
                    rep = (ordinal + block) % 3
                    for offset in range(len(conditions)):
                        condition = conditions[
                            (ordinal + rep + offset) % len(conditions)
                        ]
                        entry = cat[key, condition]
                        jobs.append(
                            {
                                "study": study,
                                "key": key,
                                "condition": condition,
                                "repeat": rep,
                                **{
                                    k: entry[k]
                                    for k in [
                                        "messages_sha256",
                                        "schema_path",
                                        "schema_sha256",
                                        "population",
                                    ]
                                },
                            }
                        )
            stages.append(
                {
                    "id": f"{study}_{batch + 1:02d}",
                    "study": study,
                    "input_keys": group,
                    "items": len(jobs),
                    "ordinary_calls": len(jobs) * 3,
                    "jobs": jobs,
                }
            )
    return {
        "ordering": "One factorial batch, two human-meaning batches, one source-selection batch. Stable hashed input order; rotating conditions/repeats; keep every comparison for a key together.",
        "stages": stages,
    }


def prepare():
    raise a.IntegrityError(
        "Use only the original frozen preparation; recovery cannot reprepare it"
    )
    a.require(
        not PLAN.exists() and (not (BASE / "live").exists()),
        "Preserve frozen execution",
    )
    tested = a.read(test_receipt())
    a.require(
        tested["passed"] and tested["environment"] == core.environment(),
        "Passing current tests required",
    )
    verify(tested["source_hashes"])
    inputs = {}
    catalogs = {}
    for study in STUDIES:
        for name, h in study_inputs(study).items():
            bind(inputs, name, h)
        catalogs[study] = load_catalog(study)
    schedule = make_schedule(catalogs)
    a.require(
        sum((s["items"] for s in schedule["stages"])) == 22536
        and sum((s["ordinary_calls"] for s in schedule["stages"])) == 67608,
        "Wrong queue budget",
    )
    save(BASE / "schedule.json", schedule)
    for name in ["EXECUTION_PROTOCOL.md", "schedule.json"]:
        bind(inputs, rel(BASE / name), a.digest(BASE / name))
    bind(inputs, rel(test_receipt()), a.digest(test_receipt()))
    save(
        PLAN,
        {
            "created_utc": a.utc(),
            "status": "prepared_offline_rehearsal_required",
            "studies": {
                k: {
                    "root": rel(v["root"]),
                    "science_sha": a.digest(v["root"] / "scientific_plan_v1.json"),
                    "inputs": v["inputs"],
                    "settings": v["conditions"],
                    "predictions": v["predictions"],
                }
                for k, v in STUDIES.items()
            },
            "input_hashes": inputs,
            "source_hashes": {n: a.digest(ROOT / n) for n in SOURCE_FILES},
            "environment": core.environment(),
            "api_contract": a.CONTRACT,
            "workers": WORKERS,
            "budget": {
                "stages": 4,
                "predictions": 22536,
                "ordinary_calls": 67608,
                "max_logical_calls": 202824,
                "max_transport_POST_attempts": 1014120,
            },
            "no_accuracy_based_stopping": True,
            "selection_stop": "Actual balanced correction pools must be audited and frozen after this queue; no update run starts here.",
            "scope": "Fresh frozen messages only; no retrieval, training, CUDA, TEST or source/schema edits during live execution.",
        },
    )
    print("Additions execution frozen; offline rehearsal required.", flush=True)


def validate(ready=False):
    recovery.validate()
    plan = a.read(PLAN)
    a.require(
        plan["environment"] == core.environment()
        and plan["workers"] == WORKERS
        and (plan["api_contract"] == a.CONTRACT),
        "Environment/contract changed",
    )
    verify(plan["input_hashes"])
    verify(plan["source_hashes"])
    if ready:
        receipt = a.read(BASE / "ready.json")
        a.require(
            receipt["status"] == "offline_verified_ready_for_user"
            and receipt["plan_sha256"] == a.digest(PLAN),
            "Execution rehearsal required",
        )
        a.require(
            a.digest(ROOT / receipt["evidence"]) == receipt["evidence_sha256"],
            "Readiness evidence changed",
        )
    return plan


def run_all():
    from analysis.feedback.supplements import resume_audit as auditor

    with a.item_lock(a.storage_path(BASE / "combined_launcher.lock")):
        validate(ready=True)
        plan_sha = a.digest(PLAN)
        schedule = a.read(BASE / "schedule.json")
        cats = {k: load_catalog(k) for k in STUDIES}

        def root(st):
            return BASE / "live" / st["study"]

        def run_one(st):
            validate(ready=True)
            execute_stage(
                st,
                cats[st["study"]],
                root(st),
                plan_sha,
                mode="live",
                client_factory=core.live_client,
            )

        run_sequence(
            schedule["stages"],
            run_one,
            lambda st: auditor.audit_stage(
                st, cats[st["study"]], root(st), plan_sha, mode="live"
            ),
            lambda st: (root(st) / "stages" / st["id"] / "review.json").exists(),
            lambda st: verify_review(st, root(st), plan_sha, "live"),
        )
        validate(ready=True)
        auditor.finalize_queue(schedule, BASE / "live", plan_sha, mode="live")
    print(
        "ALL FOUR BATCHES AND AUTOMATIC AUDITS COMPLETE. Tell Astra to inspect. Balanced-correction update runs have NOT started.",
        flush=True,
    )


def main():
    os.chdir(ROOT)
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["validate", "status", "run-all"])
    args = p.parse_args()
    if args.mode == "run-all":
        run_all()
        return
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(core, "live_client", forbidden):
        if args.mode == "prepare":
            prepare()
        elif args.mode == "status" and (not PLAN.exists()):
            print(
                json.dumps({"execution_ready": False, "actual_API_calls": 0}),
                flush=True,
            )
        else:
            plan = validate(ready=args.mode == "validate")
            print(
                json.dumps(
                    {
                        "validated": True,
                        "ready": (BASE / "ready.json").exists(),
                        "budget": plan["budget"],
                        "actual_API_calls": 0,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()

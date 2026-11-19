"""German guidelines run."""

import argparse
import base64
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import shutil
import time
from analysis.knowledge.german_guidelines import common as c
from analysis.common import batch as core
from analysis.common import schema as isolated
from analysis.common import runtime as host

a = c.a


def identity(job, plan_sha):
    return {
        "experiment_sha256": plan_sha,
        "condition": job["condition"],
        "repeat": job["repeat"],
        "key": job["key"],
    }


def item_dir(root, job, plan_sha):
    return a.storage_path(root) / "items" / a.signature(identity(job, plan_sha))


def catalogue():
    return {(r["key"], r["condition"]): r for r in c.rows(c.BASE / "prompts.jsonl")}


def storage_layout():
    c.require(
        c.OUTPUT.is_dir() and c.normal(c.OUTPUT) == c.OUTPUT,
        "D: output folder unavailable or redirected",
    )
    for path in (c.OUTPUT, *c.OUTPUT.parents):
        c.require(
            not path.is_symlink() and (not path.is_junction()),
            "Study output cannot use a junction or link",
        )
    return {
        "output_root": c.OUTPUT.as_posix(),
        "volume_device_id": c.OUTPUT.stat().st_dev,
        "workspace": c.ROOT.as_posix(),
        "plan_sha256": c.sha(c.PLAN),
        "old_outputs_moved": False,
    }


def init_storage():
    with c.offline():
        c.validate(ready=False)
        c.require(
            c.OUTPUT.drive == "D:" and c.OUTPUT.parent == Path("D:/Thesis_Experiments"),
            "Unexpected output path",
        )
        c.OUTPUT.mkdir(parents=True, exist_ok=True)
        c.save(c.OUTPUT / "storage_layout.json", storage_layout())
        print("New isolated D: output folder is ready; zero API calls.", flush=True)


def verify_storage():
    c.require(
        c.read(c.OUTPUT / "storage_layout.json") == storage_layout(),
        "Output drive identity changed",
    )


def locks():
    stack = ExitStack()
    try:
        for path in (host.SHARED / "qwen_queue.lock", c.OUTPUT / "queue.lock"):
            stack.enter_context(a.item_lock(a.storage_path(path)))
    except BaseException:
        stack.close()
        raise
    return stack


def host_check(remaining, initial=False):
    c.require(
        datetime.now().astimezone() <= c.DEADLINE,
        "15 November stop deadline reached; keep all evidence",
    )
    verify_storage()
    snapshot = host.resources(2.0 if initial else 0.75, 3.0)
    required = 8 * 2**30 + remaining * 512 * 1024
    free = shutil.disk_usage(c.OUTPUT).free
    c.require(
        free >= required,
        f"D: needs {required / 2 ** 30:.1f} GiB free for remaining output and reserve; currently {free / 2 ** 30:.1f}.",
    )
    return snapshot | {"D_free_GiB": free / 2**30, "D_required_GiB": required / 2**30}


def availability(folder, transport):
    import requests

    c.save(folder / "model_check_intent.json", {"started_utc": a.utc(), "model": a.MODEL})
    started = time.perf_counter()
    try:
        response = requests.get(
            transport._endpoint.removesuffix("/chat/completions") + "/models",
            headers={"Authorization": "Bearer " + transport._token},
            timeout=30,
            allow_redirects=False,
        )
    except requests.exceptions.RequestException as exc:
        c.save(folder / "model_check_error.json", {"category": type(exc).__name__})
        raise a.CallStopped(
            "Model availability check failed; no batch requests submitted"
        ) from None
    c.save(
        folder / "model_check_response.json",
        {
            "status": response.status_code,
            "body_base64": base64.b64encode(response.content).decode(),
            "body_sha256": hashlib.sha256(response.content).hexdigest(),
            "elapsed_s": time.perf_counter() - started,
        },
    )
    c.require(response.status_code == 200, "Model roster unavailable; no fallback")
    try:
        present = a.MODEL in {v["id"] for v in response.json()["data"]}
    except (ValueError, KeyError, TypeError):
        raise a.CallStopped("Malformed model roster") from None
    c.require(present, "Pinned Qwen model unavailable; no fallback")
    c.save(
        folder / "availability_ready.json",
        {"model": a.MODEL, "response_sha256": c.sha(folder / "model_check_response.json")},
    )


def scan_job(root, job, entry, plan_sha, backend, mode):
    folder = item_dir(root, job, plan_sha)
    if (folder / "complete.json").exists():
        backend.audit_item(root, identity(job, plan_sha), entry["messages"], execution_mode=mode)
        return c.read(folder / "result.json")
    if not folder.exists():
        return None
    c.require(
        not list(folder.rglob("*.partial_*")), "Partial atomic file requires review; preserve it"
    )
    stamp_path = folder / "job.json"
    if not stamp_path.exists():
        c.require(
            {p.name for p in folder.iterdir()} <= {"item.lock"},
            "Incomplete job attribution requires review",
        )
        return None
    stamp = c.read(stamp_path)
    c.require(
        stamp
        == {
            "identity": identity(job, plan_sha),
            "messages_sha256": entry["messages_sha256"],
            "contract": a.CONTRACT,
            "implementation": backend.implementation(),
            "execution_mode": mode,
        },
        "Partial job identity or implementation changed",
    )
    for intent_path in folder.rglob("intent.json"):
        out_path = intent_path.with_name("outcome.json")
        if not out_path.exists():
            raise a.IncompleteAttempt(
                "Saved request has no response: keep files and request review before resending"
            )
        intent, outcome = (c.read(intent_path), c.read(out_path))
        c.require(
            intent["job_stamp"] == a.signature(stamp)
            and outcome["intent_sha256"] == a.signature(intent),
            "Partial request attribution changed",
        )
        action, reason, _ = a.interpret(outcome)
        if action == "stop" or (action == "retry" and intent["attempt"] == 4):
            raise a.CallStopped("Saved request requires review: " + reason)
    c.require(
        all((p.with_name("intent.json").exists() for p in folder.rglob("outcome.json"))),
        "Response lacks request intent",
    )
    return None


def independent_decision(votes):
    valid = [v for v in votes if v is not None]
    if not valid:
        return None
    n = len(valid)
    result = {"hate": sum((v["hate"] for v in valid)) * 2 > n}
    for field in ("target_group", "hate_type", "legal"):
        counts = Counter((label for v in valid for label in v[field]))
        result[field] = sorted((k for k, count in counts.items() if 2 * count > n))
    order = ["low", "medium", "high"]
    severities = sorted((order.index(v["severity"]) for v in valid if v["severity"] is not None))
    result["severity"] = order[severities[(len(severities) - 1) // 2]] if severities else None
    if not result["hate"]:
        result.update(target_group=[], hate_type=[], severity=None)
    return result


def verify_review(root, stage, plan_sha, mode):
    root = a.storage_path(root)
    review = c.read(root / "batches" / stage["id"] / "review.json")
    c.require(
        review["plan_sha256"] == plan_sha
        and review["stage_sha256"] == a.signature(stage)
        and (review["mode"] == mode)
        and (review["status"] == "raw_responses_and_votes_verified"),
        "Batch review attribution changed",
    )
    expected = {a.signature(identity(j, plan_sha)) for j in stage["jobs"]}
    c.require(set(review["items"]) == expected, "Batch review roster changed")
    for stamp, hashes in review["items"].items():
        folder = root / "items" / stamp
        for name in ("complete.json", "result.json"):
            c.require(c.sha(folder / name) == hashes[name], "Audited result or receipt changed")
    c.verify(review["availability_hashes"])
    return review


def audit_batch(root, stage, cat, plan_sha, backend, mode):
    root = a.storage_path(root)
    path = root / "batches" / stage["id"] / "review.json"
    if path.exists():
        return verify_review(root, stage, plan_sha, mode)
    items, counts = ({}, Counter())
    for index, job in enumerate(stage["jobs"], 1):
        entry = cat[job["key"], job["condition"]]
        backend.audit_item(root, identity(job, plan_sha), entry["messages"], execution_mode=mode)
        folder = item_dir(root, job, plan_sha)
        result = c.read(folder / "result.json")
        decision = independent_decision(result["run_predictions"])
        delivered = result["result"]
        c.require(
            decision == (None if delivered is None else {k: delivered[k] for k in decision}),
            "Independent voter differs",
        )
        c.require(result["messages_sha256"] == job["messages_sha256"], "Audited message differs")
        counts["predictions"] += 1
        counts["null_results"] += delivered is None
        counts.update(result["accounting"])
        items[a.signature(identity(job, plan_sha))] = c.bindings([]) | {
            name: c.sha(folder / name) for name in ("complete.json", "result.json")
        }
        if index % 250 == 0:
            print(f"{stage['id']}: audited {index}/{stage['items']} predictions.", flush=True)
    availability_files = []
    for attempt in (root / "batches" / stage["id"]).glob("attempt_*"):
        if mode == "live":
            availability_files.extend((p for p in attempt.iterdir() if p.is_file()))
            response_path = attempt / "model_check_response.json"
            submitted = (attempt / "summary.json").exists() and c.read(
                attempt / "summary.json"
            ).get("submitted", 0)
            ready_path = attempt / "availability_ready.json"
            c.require(
                not submitted or ready_path.exists(),
                "Submitted batch lacks model availability receipt",
            )
            if ready_path.exists():
                intent = c.read(attempt / "model_check_intent.json")
                response = c.read(response_path)
                c.require(
                    c.read(ready_path)
                    == {"model": a.MODEL, "response_sha256": c.sha(response_path)},
                    "Availability approval receipt changed",
                )
                raw = base64.b64decode(response["body_base64"], validate=True)
                c.require(
                    intent["model"] == a.MODEL
                    and response["status"] == 200
                    and (hashlib.sha256(raw).hexdigest() == response["body_sha256"])
                    and (a.MODEL in {v["id"] for v in json.loads(raw)["data"]}),
                    "Model availability receipt differs",
                )
    c.require(
        counts["predictions"] == stage["items"]
        and counts["ordinary_logical_calls"] == stage["ordinary_calls"],
        "Batch call accounting differs",
    )
    review = {
        "status": "raw_responses_and_votes_verified",
        "plan_sha256": plan_sha,
        "stage_sha256": a.signature(stage),
        "mode": mode,
        "items": items,
        "counts": dict(counts),
        "availability_hashes": c.bindings(availability_files),
    }
    c.save(path, review)
    print(f"{stage['id']}: automatic audit passed.", flush=True)
    return review


def execute_batch(
    root,
    stage,
    cat,
    plan_sha,
    backend,
    mode,
    factory,
    sleep=time.sleep,
    availability_check=availability,
):
    root = a.storage_path(root)
    target = root / "batches" / stage["id"]
    if (target / "review.json").exists():
        verify_review(root, stage, plan_sha, mode)
        print(f"{stage['id']}: already audited; no new requests.", flush=True)
        return
    existing, pending = ([], [])
    for job in stage["jobs"]:
        value = scan_job(root, job, cat[job["key"], job["condition"]], plan_sha, backend, mode)
        (pending if value is None else existing).append(job if value is None else value)
    print(
        f"{stage['id']}: {len(existing)} complete, {len(pending)} pending; {3 * len(pending)} ordinary calls before saved-vote reuse.",
        flush=True,
    )
    if not pending:
        return
    nulls = sum((v["result"] is None for v in existing))
    c.require(
        len(existing) < 20 or nulls / len(existing) < 0.1,
        "Ten percent unavailable outputs: review saved replies before continuing",
    )
    attempt = target / f"attempt_{time.time_ns()}"
    c.save(
        attempt / "start.json",
        {
            "started_utc": a.utc(),
            "plan_sha256": plan_sha,
            "stage_sha256": a.signature(stage),
            "mode": mode,
            "workers": c.WORKERS,
            "existing_predictions": len(existing),
            "pending_predictions": len(pending),
        },
    )
    try:
        transport = factory()
        if mode == "live":
            availability_check(attempt, transport)

        def work(job):
            return backend.execute_item(
                root,
                identity(job, plan_sha),
                cat[job["key"], job["condition"]]["messages"],
                transport,
                sleep=sleep,
                execution_mode=mode,
            )

        def progress(job, result, stats):
            if stats["complete"] % 100 == 0 or stats["complete"] == stage["items"]:
                print(
                    f"{stage['id']}: saved {stats['complete']}/{stage['items']}; null outputs {stats['null']}.",
                    flush=True,
                )

        outcome = core.bounded_jobs(
            pending,
            work,
            workers=c.WORKERS,
            initial={"complete": len(existing), "null": nulls},
            on_result=progress,
        )
        summary = {k: v for k, v in outcome.items() if k != "results"}
        c.save(attempt / "summary.json", summary | {"finished_utc": a.utc()})
        c.require(
            not outcome["stopped"]
            and (not outcome["errors"])
            and (outcome["stats"]["complete"] == stage["items"]),
            "Queue stopped safely. Keep all files; send the final message for review. No retries were reset.",
        )
    except BaseException as exc:
        c.save(
            attempt / "stop.json",
            {
                "finished_utc": a.utc(),
                "error_type": type(exc).__name__,
                "reason": (
                    str(exc)[:400]
                    if type(exc) in (a.CallStopped, a.IncompleteAttempt, a.IntegrityError)
                    else "Inspect preserved evidence"
                ),
                "policy": "Completed replies stay saved. A graceful Ctrl+C can resume after pending requests drain; unknown or exhausted attempts require review.",
            },
        )
        raise


def finalize(root, schedule, plan_sha, mode):
    reviews = []
    counts = Counter()
    for stage in schedule["stages"]:
        review = verify_review(root, stage, plan_sha, mode)
        counts.update(review["counts"])
        reviews.append(a.storage_path(root) / "batches" / stage["id"] / "review.json")
    record = {
        "status": "all_three_conditions_executed_and_audited",
        "mode": mode,
        "plan_sha256": plan_sha,
        "counts": dict(counts),
        "review_hashes": c.bindings(reviews),
    }
    c.save(a.storage_path(root) / "complete.json", record)
    return record


def check():
    with c.offline():
        plan = c.validate()
        with locks():
            status = host_check(plan["budget"]["predictions"], initial=True)
        print(
            json.dumps(
                {
                    "ready": True,
                    "workers": c.WORKERS,
                    "budget": plan["budget"],
                    "output_root": str(c.OUTPUT),
                    "resources": status,
                    "actual_Qwen_calls": 0,
                    "actual_model_list_GETs": 0,
                }
            ),
            flush=True,
        )


def run():
    with locks():
        with c.offline():
            plan = c.validate()
            verify_storage()
            schedule = c.read(c.BASE / "schedule.json")
            cat = catalogue()
            plan_sha = c.sha(c.PLAN)
            backend = isolated.adapter(c.ROOT / "config/taxonomy.yaml")
            host_check(plan["budget"]["predictions"], initial=True)
            gc.collect()
        print(
            "German guidelines: one queue, 128 workers, all three conditions and three repeats.",
            flush=True,
        )
        print(
            "Press Ctrl+C once, then wait for active requests to finish before closing PowerShell.",
            flush=True,
        )
        print("All outputs go to " + str(c.OUTPUT), flush=True)
        completed = 0
        for stage in schedule["stages"]:
            with c.offline():
                c.verify(plan["source_hashes"])
                host_check(plan["budget"]["predictions"] - completed)
            execute_batch(
                c.OUTPUT / "live", stage, cat, plan_sha, backend, "live", core.live_client
            )
            with c.offline():
                audit_batch(c.OUTPUT / "live", stage, cat, plan_sha, backend, "live")
            completed += stage["items"]
            print(
                f"Overall audited: {completed}/{plan['budget']['predictions']} ({completed / plan['budget']['predictions']:.1%}).",
                flush=True,
            )
        with c.offline():
            result = finalize(c.OUTPUT / "live", schedule, plan_sha, "live")
            c.require(result["counts"]["predictions"] == 33093, "Final prediction count differs")
    print(
        "GERMAN GUIDELINE EXPERIMENT AND ALL AUDITS COMPLETE. Send this message for analysis. No other experiment started.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("init-storage", "check", "run"))
    args = parser.parse_args()
    {"init-storage": init_storage, "check": check, "run": run}[args.mode]()

from __future__ import annotations
from src.hsrag.paths import PROJECT_ROOT
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/sq2_stage_b20_20261021_v1"
RECOVERY = BASE / "recovery_memory_20261021_v1"
MANIFEST = RECOVERY / "manifest.json"
PLAN_SHA = "8e8e0742781af7df37e2b9d784c97920cc2dd7bef1ed2366017507bb98a43217"
MODULE = "training.text_budgets.extended.resume"
JOB_KEYS = ("id", "pool", "head", "config", "seed", "fold", "mode")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_new(path, value):
    path = Path(path)
    require(
        path.resolve().is_relative_to(RECOVERY.resolve()),
        "Recovery output escapes its directory",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex[:10] + ".writing")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    require(not path.exists(), "Never replace a recovery receipt")
    temporary.rename(path)


@contextmanager
def controller_lock():
    import msvcrt

    path = RECOVERY / "controller.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError(
                "The memory-resume controller is already running"
            ) from exc
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def verify_manifest():
    manifest = read(MANIFEST)
    require(
        manifest["status"] == "tested_ready" and manifest["plan_sha256"] == PLAN_SHA,
        "Memory recovery has not passed its checks",
    )
    require(sha(BASE / "plan.json") == PLAN_SHA, "Original sealed plan changed")
    for name, digest in manifest["bindings"].items():
        path = (ROOT / name).resolve()
        require(
            path.is_relative_to(ROOT) and sha(path) == digest,
            "Recovery binding changed: " + name,
        )
    return manifest


def final_jobs(b, plan_sha):
    selected = {}
    for pool in b.NAMES:
        r = b.c.read(b.c.BASE / "selections" / (pool + ".json"))
        b.c.require(
            r["plan_sha256"] == plan_sha and r["pool"] == pool and r["grid"] == b.GRID,
            "Selection identity changed",
        )
        selected[pool] = {family: r["families"][family] for family in b.c.FAMILIES}
    jobs = [j for pool in b.NAMES for j in b.final_jobs(pool, selected[pool])]
    b.c.require(
        len(jobs) == 360 and len({j["id"] for j in jobs}) == 360,
        "Final model inventory changed",
    )
    return selected, jobs


def perform(task, plan, plan_sha):
    "Called only inside a worker holding b.locks() and b.isolated()."
    from training.text_budgets.extended import common as b
    from training.text_budgets.extended import train as train
    from training.text_budgets.extended import audit as audit

    c = b.c
    op = task["op"]
    if op in ("search", "final"):
        job = task["job"]
        c.require(
            job
            == b.job(job["pool"], job["head"], job["config"], job["seed"], job["fold"]),
            "Malformed job",
        )
        if op == "search":
            c.require(job in b.jobs(job["pool"], plan["grid"]), "Undeclared search job")
            c.require(
                not any((c.BASE / "refits").glob("**/complete.json")),
                "Search requested after final fits",
            )
            remaining = None
        else:
            _, remaining = final_jobs(b, plan_sha)
            c.require(
                job in remaining and (c.BASE / "all_selections.json").exists(),
                "Final fit before complete selections",
            )
        b.store_guard(plan["stores"], remaining=remaining)
        data = b.data(job["pool"])
        fitted = train.execute(job, data, plan, plan_sha)
        predicted = None
        if op == "final":
            predicted = train.inference_job(
                job, data, plan_sha, c.read(c.BASE / "data/inference.json")
            )
        return dict(job=job, fit=fitted, inference=predicted)
    if op == "select":
        pool = task["pool"]
        c.require(pool in b.NAMES, "Unknown selection pool")
        audit.select(pool, plan["grid"], plan_sha)
        return dict(pool=pool, receipt=c.sha(c.BASE / "selections" / (pool + ".json")))
    if op == "seal_selections":
        selected, jobs = final_jobs(b, plan_sha)

        for pool in b.NAMES:
            c.verify(
                c.read(c.BASE / "selections" / (pool + ".json"))["search_proof"][
                    "receipts"
                ]
            )
        b.save(
            c.BASE / "all_selections.json",
            dict(
                plan_sha256=plan_sha,
                outputs=b.hashes(
                    c.BASE / "selections" / (pool + ".json") for pool in b.NAMES
                ),
                evaluation_gold_used=False,
            ),
        )
        return dict(jobs=jobs)
    if op == "audit":
        pool = task["pool"]
        c.require(pool in b.NAMES, "Unknown audit pool")
        selected, _ = final_jobs(b, plan_sha)
        a = audit.final_check(pool, selected[pool], plan_sha)
        z = audit.assemble(pool, selected[pool], plan_sha)
        return dict(pool=pool, final_models=a["final_models"], variants=z["variants"])
    if op == "complete":
        _, jobs = final_jobs(b, plan_sha)
        outputs = {}
        for pool in b.NAMES:
            for suffix in ("final", "assembly"):
                path = c.BASE / "audits" / f"{pool}_{suffix}.json"
                r = c.read(path)
                c.require(
                    r["status"] == "pass"
                    and r["pool"] == pool
                    and r["plan_sha256"] == plan_sha,
                    "Missing/mismatched final audit",
                )
                c.verify(r["outputs"])
                outputs.update(r["outputs"])
        for path in (
            c.BASE / "features/complete.json",
            c.BASE / "linear/complete.json",
            c.BASE / "all_selections.json",
        ):
            c.verify(c.read(path)["outputs"])
            outputs.update(b.hashes([path]))
            outputs.update(c.read(path)["outputs"])
        outputs.update(b.hashes((c.BASE / "audits").glob("*.json")))
        b.source_guard()
        b.save(
            c.BASE / "complete.json",
            dict(
                status="training_predictions_and_automatic_audits_complete",
                plan_sha256=plan_sha,
                outputs=outputs,
                pools=b.NAMES,
                distinct_final_models=len(jobs),
                Qwen_calls=0,
                scientific_review_pending=True,
            ),
        )
        return dict(complete=True, scientific_review_pending=True)
    raise RuntimeError("Unknown worker operation: " + op)


def preflight(plan, plan_sha):
    from training.text_budgets.extended import common as b
    from training.text_budgets.extended import prepare as prepare

    c = b.c
    prepare.validate()
    b.source_guard()
    done = c.BASE / "complete.json"
    if done.exists():
        record = c.read(done)
        c.require(
            record["plan_sha256"] == plan_sha
            and record["status"]
            == "training_predictions_and_automatic_audits_complete",
            "Completion identity",
        )
        c.verify(record["outputs"])
    for relative in ("features/complete.json", "linear/complete.json"):
        record = c.read(c.BASE / relative)
        c.require(
            record.get("plan_sha256", record.get("identity", {}).get("plan_sha256"))
            == plan_sha,
            "Earlier completed work belongs to another plan",
        )
        c.verify(record["outputs"])
    pending = {}
    counts = {}
    for pool in b.NAMES:
        pending[pool] = []
        for job in b.jobs(pool, plan["grid"]):
            if (c.job_dir(job) / "complete.json").exists():
                b.receipt(job, plan_sha)
            else:
                pending[pool].append(job)
        counts[pool] = len(b.jobs(pool)) - len(pending[pool])
        selected_path = c.BASE / "selections" / (pool + ".json")
        if selected_path.exists():
            r = c.read(selected_path)
            c.require(
                not pending[pool]
                and r["plan_sha256"] == plan_sha
                and r["pool"] == pool,
                "Selection before complete search",
            )
            c.verify(r["search_proof"]["receipts"])
    have_final = any((c.BASE / "refits").glob("**/complete.json"))
    all_selected = all(
        (c.BASE / "selections" / (p + ".json")).exists() for p in b.NAMES
    )
    c.require(not have_final or all_selected, "Final fit before all selections")
    remaining = final_jobs(b, plan_sha)[1] if all_selected else None
    b.validate_stores(plan["stores"])
    if not done.exists():

        clock = c.read(c.BASE / "execution_budget.json")
        c.require(clock["plan_sha256"] == plan_sha, "Execution clock identity")
        b.deadline_guard()
        b.ac_guard()
        b.store_guard(plan["stores"], remaining=remaining)
    return dict(
        complete=done.exists(),
        pools=b.NAMES,
        pending=pending,
        completed_search=counts,
        resources=b.resource.resources(),
    )


def worker(request_path):
    request_path = Path(request_path).resolve()
    require(
        request_path.is_relative_to((RECOVERY / "sessions").resolve())
        and request_path.name == "request.json",
        "Invalid worker request path",
    )
    verify_manifest()
    task = read(request_path)
    require(
        task["plan_sha256"] == PLAN_SHA and task["manifest_sha256"] == sha(MANIFEST),
        "Worker request binding",
    )
    from training.text_budgets.extended import common as b

    c = b.c
    plan = c.read(BASE / "plan.json")
    directory = request_path.parent
    started = time.perf_counter()
    try:
        with b.locks(), b.isolated(plan["grid"], plan["stores"]):

            for path in b.sources():
                c.require(
                    c.sha(path) == plan["inputs"][c.rel(path)], "Frozen source changed"
                )
            c.require(c.environment() == plan["environment"], "Environment changed")
            c.verify(plan["prepared"])
            before = b.resource.resources()
            if task["op"] == "preflight":
                result = preflight(plan, PLAN_SHA)
            else:
                b.deadline_guard()
                b.ac_guard()
                result = perform(task, plan, PLAN_SHA)
            write_new(
                directory / "complete.json",
                dict(
                    status="complete",
                    request_sha256=sha(request_path),
                    result=result,
                    resources_before=before,
                    resources_after=b.resource.resources(),
                    seconds=time.perf_counter() - started,
                    pid=os.getpid(),
                ),
            )
    except BaseException as exc:
        write_new(
            directory / "stopped.json",
            dict(
                type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
                request_sha256=sha(request_path),
                seconds=time.perf_counter() - started,
            ),
        )
        raise


def wait_child(process):
    try:
        return process.wait()
    except KeyboardInterrupt:

        previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            process.wait()
        finally:
            signal.signal(signal.SIGINT, previous)
        raise


def launch_worker(session, index, task):
    directory = session / f'{index:04d}_{task["op"]}'
    request = dict(task, plan_sha256=PLAN_SHA, manifest_sha256=sha(MANIFEST))
    path = directory / "request.json"
    write_new(path, request)
    command = [
        sys.executable,
        "-X",
        "utf8",
        "-B",
        "-u",
        "-m",
        MODULE,
        "worker",
        "--request",
        str(path),
    ]
    process = subprocess.Popen(command, cwd=ROOT)
    code = wait_child(process)
    require(
        code == 0,
        "Worker stopped. All saved work is retained; no automatic retry or guard bypass.",
    )
    receipt = read(directory / "complete.json")
    require(
        receipt["status"] == "complete" and receipt["request_sha256"] == sha(path),
        "Worker completion mismatch",
    )
    return receipt["result"]


def drive(call):
    state = call(dict(op="preflight"))
    if state["complete"]:
        print(
            "Verified completed comparison. Scientific review remains separate.",
            flush=True,
        )
        return
    for pool in state["pools"]:
        pending = state["pending"][pool]
        for index, job in enumerate(pending, 1):
            print(
                f'Search {pool}: remaining {index}/{len(pending)}; {job["id"]}',
                flush=True,
            )
            call(dict(op="search", job=job))
        call(dict(op="select", pool=pool))
    jobs = call(dict(op="seal_selections"))["jobs"]
    for index, job in enumerate(jobs, 1):
        print(
            f'Final model and predictions {index}/{len(jobs)}: {job["id"]}', flush=True
        )
        call(dict(op="final", job=job))
    for pool in state["pools"]:
        call(dict(op="audit", pool=pool))
    call(dict(op="complete"))
    print("EQUAL-BUDGET RUN COMPLETE. Ask Astra for scientific review.", flush=True)


def run():
    verify_manifest()
    with controller_lock():
        session = (
            RECOVERY
            / "sessions"
            / (
                datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                + "_"
                + uuid.uuid4().hex[:8]
            )
        )
        count = 0

        def call(task):
            nonlocal count
            count += 1
            return launch_worker(session, count, task)

        try:
            drive(call)
            write_new(
                session / "complete.json",
                dict(status="complete", workers=count, plan_sha256=PLAN_SHA),
            )
        except BaseException as exc:
            write_new(
                session / "stopped.json",
                dict(
                    type=type(exc).__name__, message=str(exc), workers_requested=count
                ),
            )
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["run", "worker"])
    parser.add_argument("--request")
    args = parser.parse_args()
    if args.mode == "run":
        require(args.request is None, "Run does not accept a worker request")
        run()
    else:
        require(args.request is not None, "Worker requires a bound request")
        worker(args.request)

from contextlib import ExitStack
from datetime import datetime, timezone, timedelta
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid
from training.full_data import common as c
from training.full_data import plan as p
from training.full_data import runtime as r
from training.full_data import study as study
from training.full_data import archive as archive


def used_seconds():
    total = 0.0
    for path in sorted((r.OUTPUT / "invocations").glob("*/started.json")):
        end = path.parent / "finished.json"
        c.require(
            end.exists(),
            "An unclosed controller invocation needs review: " + str(path.parent),
        )
        value = c.read(end)
        c.require(
            value["started_sha256"] == c.sha(path) and value["seconds"] >= 0,
            "Invalid elapsed-time record.",
        )
        total += value["seconds"]
    return total


def gpu_environment(count, total_mib):
    c.require(count in (1, 8), "No unreviewed worker count is allowed.")
    return dict(
        CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=str(80 // count),
        CUDA_MPS_PINNED_DEVICE_MEM_LIMIT=f"0={int(total_mib*.8-1024)//count}M",
    )


def cpu_preflight():
    path = r.OUTPUT / "cpu_test_certificate.json"
    package_sha = c.sha(p.PACKAGE / "manifest.json")
    if path.exists():
        value = c.read(path)
        c.require(
            value["package_sha256"] == package_sha,
            "CPU test certificate belongs to another package.",
        )
        r.verify_tree(r.OUTPUT, value["outputs"])
        return
    root = r.OUTPUT / "preflight_tests" / uuid.uuid4().hex
    root.mkdir(parents=True)
    with (root / "console.log").open("xb") as log:
        result = subprocess.run(
            [sys.executable, "-B", str(p.PACKAGE / "tests/test_preparation.py")],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
            env=dict(os.environ, SQ2_TEST_OUTPUT=str(root)),
            check=False,
        )
    c.require(
        result.returncode == 0,
        "CPU preparation tests failed on this server; no model was started.",
    )
    checked = c.read(root / "results.json")
    c.require(
        checked["success"]
        and checked["tests"] >= 40
        and checked["failures"] == checked["errors"] == checked["skipped"] == 0
        and checked["report_sha256"] == c.sha(root / "results.txt"),
        "Incomplete server CPU test receipt.",
    )
    c.save(
        path,
        dict(
            package_sha256=package_sha,
            tests=checked["tests"],
            new_model_fits=0,
            Qwen_calls=0,
            outputs={
                v.relative_to(r.OUTPUT).as_posix(): c.sha(v)
                for v in (
                    root / "results.json",
                    root / "results.txt",
                    root / "console.log",
                )
            },
        ),
    )


def backup_capacity():
    models = {
        pool: [
            j["id"]
            for j in p.final_jobs(
                pool, c.read(r.OUTPUT / "selections" / f"{pool}.json")["families"]
            )
        ]
        for pool in p.NAMES
    }
    count = sum(len(v) for v in models.values())
    required = (
        count * p.EXPORT_BYTES_MAX
        + r.settings()["retained_nonweight_allowance_bytes"]
        + r.settings()["backup_free_reserve_bytes"]
    )
    value = dict(
        plan_sha256=c.sha(p.PACKAGE / "plan.json"),
        models=models,
        models_total=count,
        model_export_bound_bytes=count * p.EXPORT_BYTES_MAX,
        required_initial_free_per_backup_volume_bytes=required,
    )
    c.save(r.OUTPUT / "final_export_plan.json", value)

    certificate = r.OUTPUT / "backup_capacity_accepted.json"
    if certificate.exists():
        saved = c.read(certificate)
        c.require(
            saved["export_plan_sha256"] == c.sha(r.OUTPUT / "final_export_plan.json"),
            "Backup capacity plan belongs to another model roster.",
        )
        r.verify_tree(r.OUTPUT, saved["outputs"])
        return
    path = Path("/inputs/backup_capacity.json")
    if not path.exists():
        raise r.Pause(
            "TRAIN search is complete. Read final_export_plan.json and provide the two-volume backup capacity receipt before final fitting."
        )
    review = c.read(path)
    volumes = review["volumes"]
    c.require(
        len(volumes) == 2 and volumes[0]["volume_id"] != volumes[1]["volume_id"],
        "Two distinct backup volumes are required.",
    )
    c.require(
        all(
            isinstance(v["volume_id"], str)
            and v["volume_id"]
            and isinstance(v["directory"], str)
            and v["directory"]
            for v in volumes
        ),
        "Missing backup volume identity.",
    )
    age = datetime.now(timezone.utc) - datetime.fromisoformat(review["checked_utc"])
    c.require(
        timedelta(0) <= age <= timedelta(days=7),
        "Refresh the backup-space measurement.",
    )
    if any(
        type(v["free_bytes"]) is not int or v["free_bytes"] < required for v in volumes
    ):
        raise r.Pause(
            f"The selected model roster needs {required/1e9:.2f} GB free on each backup volume, including reserves. Keep the completed search."
        )
    copy = r.OUTPUT / "backup_capacity_source.json"
    c.save(copy, review)
    c.save(
        certificate,
        dict(
            export_plan_sha256=c.sha(r.OUTPUT / "final_export_plan.json"),
            required_free_per_volume_bytes=required,
            outputs={copy.relative_to(r.OUTPUT).as_posix(): c.sha(copy)},
        ),
    )


def worker(assignment):
    r.configure()
    package_sha = r.validate_package()
    value = c.read(assignment)
    c.require(value["package_sha256"] == package_sha, "Worker package changed.")
    phase = value["phase"]
    count = 8 if phase == "search" else 1
    c.require(
        value["worker"] in range(count) and value["pool"] in p.NAMES,
        "Invalid worker assignment.",
    )
    expected = (
        p.allocation(value["pool"])[value["worker"]]
        if phase == "search"
        else p.final_jobs(
            value["pool"],
            c.read(r.OUTPUT / "selections" / f"{value['pool']}.json")["families"],
        )
    )
    c.require(
        value["jobs"] == expected, "Worker job list differs from the locked plan."
    )
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, r.request_stop)
    with r.lock(r.OUTPUT / "leases" / f"worker_{value['worker']}.lock"), r.offline():
        from training.full_data import train as train

        c.require(
            os.environ.get("CUDA_MPS_PIPE_DIRECTORY") == "/run/nvidia-mps",
            "Keep the assigned MPS pipe; do not start or alter a daemon.",
        )
        c.require(
            os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE") == str(80 // count),
            "Changed GPU client limit.",
        )
        torch = train.setup()
        props = torch.cuda.get_device_properties(0)
        c.require(
            torch.cuda.device_count() == 1
            and props.major == 12
            and props.multi_processor_count == (18 if count == 8 else 150),
            "GPU partition differs from the accepted benchmark.",
        )
        c.require(torch.version.cuda == "13.0", "CUDA runtime changed.")
        c.save(
            assignment.with_suffix(".device.json"),
            dict(
                name=props.name,
                sm_count=props.multi_processor_count,
                memory=props.total_memory,
                torch=torch.__version__,
                cuda=torch.version.cuda,
                profile=phase,
            ),
        )
        data = p.data(value["pool"])
        for job in value["jobs"]:
            r.deadline_guard(value["session"])
            study.execute(job, data, value["session"])
            if phase == "final":
                study.inference(job, data, value["session"])
            print("VERIFIED", job["id"], flush=True)
        c.save(
            assignment.with_suffix(".complete.json"),
            dict(assignment_sha256=c.sha(assignment), jobs=len(value["jobs"])),
        )


def run_workers(pool, phase, session, invocation):
    count = 8 if phase == "search" else 1
    groups = (
        p.allocation(pool)
        if phase == "search"
        else [
            p.final_jobs(
                pool, c.read(r.OUTPUT / "selections" / f"{pool}.json")["families"]
            )
        ]
    )
    before = r.gpu()
    c.require(
        before["uuid"] == r.settings()["gpu_uuid"], "Assigned GPU identity changed."
    )
    c.require(
        before["used_mib"] < 1024 and before["busy"] < 5,
        "Another GPU workload is present; leave it alone.",
    )

    with ExitStack() as stack:
        for i in range(8):
            stack.enter_context(r.lock(r.OUTPUT / "leases" / f"worker_{i}.lock"))
    work = invocation / (pool + "_" + phase)
    work.mkdir()
    children = []
    handles = []
    package_sha = c.sha(p.PACKAGE / "manifest.json")
    env = dict(os.environ, **gpu_environment(count, before["total_mib"]))
    try:
        for i, jobs in enumerate(groups):
            path = work / f"worker_{i}.json"
            c.save(
                path,
                dict(
                    package_sha256=package_sha,
                    pool=pool,
                    phase=phase,
                    worker=i,
                    jobs=jobs,
                    session=session,
                ),
            )
            log = (work / f"worker_{i}.log").open("xb")
            handles.append(log)
            children.append(
                subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        "-u",
                        "-m",
                        "training.full_data.run",
                        "worker",
                        str(path),
                    ],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env=env,
                    start_new_session=True,
                )
            )
        last_storage = 0.0
        with (work / "telemetry.jsonl").open("x", encoding="utf8") as telemetry:
            while any(child.poll() is None for child in children):
                r.deadline_guard(session)
                c.require(
                    all(child.poll() in (None, 0) for child in children),
                    "A worker stopped; preserve its log and attempts.",
                )
                sample = dict(
                    utc=c.utc(),
                    memory=r.memory(),
                    gpu=r.gpu(),
                    workers=sum(x.poll() is None for x in children),
                )
                if time.monotonic() - last_storage >= 60:
                    sample["storage"] = r.storage(10_000_000_000)
                    last_storage = time.monotonic()
                telemetry.write(json.dumps(sample, sort_keys=True) + "\n")
                telemetry.flush()
                time.sleep(2)
        c.require(
            all(child.wait(timeout=1) == 0 for child in children),
            "A worker did not finish successfully.",
        )
        for i in range(count):
            path = work / f"worker_{i}.json"
            c.require(
                c.read(path.with_suffix(".complete.json"))
                == dict(assignment_sha256=c.sha(path), jobs=len(groups[i])),
                "Worker completion differs.",
            )
    except BaseException:
        r.stop_children(children, session["stop_file"])
        raise
    finally:
        for handle in handles:
            handle.close()


def check_finished_pool(pool):
    value = c.read(r.OUTPUT / "audits" / f"{pool}_complete.json")
    r.verify_tree(r.OUTPUT, value["outputs"])
    chosen = c.read(r.OUTPUT / "selections" / f"{pool}.json")
    r.verify_tree(r.OUTPUT, chosen["search_receipts"])
    inputs = c.read(p.PACKAGE / "data/inference.json")
    data = p.data(pool)
    released = (archive.archive_root(pool) / "released.json").exists()
    for job in p.final_jobs(pool, chosen["families"]):
        saved, attempt = study.receipt(job, weights=not released)
        audit_model = None if released else saved["model_directory"]
        study.audit.metadata(job, data, attempt, model_directory=audit_model)
        study.check_inference(job, inputs, data["heads"][job["head"]]["labels"])
    archive.verify(pool)


def run(stage, storage_review):
    c.require(stage in ("search", "final", "all"), "Unknown stage.")
    r.configure()
    package_sha = r.validate_package()
    r.OUTPUT.mkdir(parents=True, exist_ok=True)
    with r.lock(Path("/results/worker.lock")), r.offline():
        r.check_external_storage(storage_review)
        r.memory()
        r.storage(10_000_000_000)
        remaining = r.settings()["active_seconds"] - used_seconds()
        c.require(
            remaining > 0,
            "The active execution budget is exhausted; preserve all work.",
        )
        now = datetime.now(timezone.utc)
        end = min(
            now + timedelta(seconds=remaining),
            datetime.fromisoformat(r.settings()["stop_utc"]),
        )
        c.require(end > now, "The experiment stop date has been reached.")
        import psutil

        invocation = r.OUTPUT / "invocations" / uuid.uuid4().hex
        session = dict(
            controller_pid=os.getpid(),
            controller_created=psutil.Process().create_time(),
            end_utc=end.isoformat(),
            stop_file=str(invocation / "STOP"),
        )
        c.save(
            invocation / "started.json",
            dict(
                utc=c.utc(),
                stage=stage,
                remaining_seconds=remaining,
                session=session,
                package_sha256=package_sha,
                storage_review_sha256=c.sha(storage_review),
            ),
        )
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, r.request_stop)
        begin = time.monotonic()
        status = "failed"
        detail = None
        try:
            cpu_preflight()
            if stage in ("search", "all"):
                for pool in p.NAMES:
                    r.deadline_guard(session)
                    if not (r.OUTPUT / "selections" / f"{pool}.json").exists():
                        run_workers(pool, "search", session, invocation)
                    study.select_pool(pool)
                    print("TRAIN SELECTION CHECKED", pool, flush=True)
                c.save(
                    r.OUTPUT / "selection_lock.json",
                    dict(
                        plan_sha256=c.sha(p.PACKAGE / "plan.json"),
                        selections={
                            f"selections/{pool}.json": c.sha(
                                r.OUTPUT / "selections" / f"{pool}.json"
                            )
                            for pool in p.NAMES
                        },
                    ),
                )
            if stage in ("final", "all"):
                locked = c.read(r.OUTPUT / "selection_lock.json")
                c.require(
                    locked["plan_sha256"] == c.sha(p.PACKAGE / "plan.json")
                    and set(locked["selections"])
                    == {f"selections/{pool}.json" for pool in p.NAMES},
                    "All nine TRAIN selections must be locked first.",
                )
                r.verify_tree(r.OUTPUT, locked["selections"])
                backup_capacity()
                for pool in p.NAMES:
                    r.deadline_guard(session)
                    if (archive.archive_root(pool) / "complete.json").exists():
                        check_finished_pool(pool)
                        continue
                    if len(archive.pending()) >= 2:
                        raise r.Pause(
                            "Two finished pools await verified backups and release. Download their archives before continuing."
                        )
                    if not (r.OUTPUT / "audits" / f"{pool}_complete.json").exists():
                        r.storage(65_000_000_000)
                        run_workers(pool, "final", session, invocation)
                        study.assemble(pool)
                    archive.build(pool, session)
                    print("POOL COMPLETE; ARCHIVE READY", pool, flush=True)
                for pool in p.NAMES:
                    check_finished_pool(pool)
                c.save(
                    r.OUTPUT / "complete.json",
                    dict(
                        status="neural_fits_and_text_only_predictions_complete",
                        plan_sha256=c.sha(p.PACKAGE / "plan.json"),
                        selection_lock_sha256=c.sha(r.OUTPUT / "selection_lock.json"),
                        pool_audits={
                            f"audits/{pool}_complete.json": c.sha(
                                r.OUTPUT / "audits" / f"{pool}_complete.json"
                            )
                            for pool in p.NAMES
                        },
                        scientific_scores_computed=False,
                        Qwen_calls=0,
                    ),
                )
            status = "complete"
        except BaseException as exc:
            status = (
                "paused" if isinstance(exc, (r.Pause, KeyboardInterrupt)) else "failed"
            )
            detail = dict(
                type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
            )
            raise
        finally:
            c.save(
                invocation / "finished.json",
                dict(
                    status=status,
                    utc=c.utc(),
                    seconds=time.monotonic() - begin,
                    started_sha256=c.sha(invocation / "started.json"),
                    detail=detail,
                ),
            )

import os

for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from src.hsrag.paths import PROJECT_ROOT
import signal
import subprocess
import sys
import time
import traceback
import uuid
import zipfile
import numpy as np
from training.type_update import checks

sys.dont_write_bytecode = True
HERE = Path(
    os.environ.get(
        "HSRAG_TYPE_UPDATE_PACKAGE",
        str(PROJECT_ROOT / "deploy/type_update"),
    )
)
ORIGINAL_PACKAGE = Path("/inputs/sq2_full")
ORIGINAL_RESULTS = Path("/results/sq2_full_20261029_v1")
OUTPUT = Path("/results/sq2_type_update_20261103_v1")


def original():
    sys.path.insert(0, str(ORIGINAL_PACKAGE))
    from training.full_data import common as c
    from training.full_data import runtime as r
    from training.full_data import plan as p
    from training.full_data import study as study
    from training.full_data import controller as controller

    c.require(
        p.PACKAGE.resolve() == ORIGINAL_PACKAGE.resolve(), "Wrong original package."
    )
    return c, r, p, study, controller


def validate(c, r, p):
    c.require(
        HERE == Path("/inputs/sq2_type_update"),
        "Use the separate prepared input directory.",
    )
    protocol = c.read(HERE / "protocol.json")
    checks.validate_plan(protocol)
    checks.separate_root(OUTPUT, ORIGINAL_RESULTS)
    manifest = c.read(HERE / "manifest.json")
    c.require(
        {x.relative_to(HERE).as_posix() for x in r.files(HERE)}
        == set(manifest["files"]) | {"manifest.json"},
        "Supplementary package inventory changed.",
    )
    r.verify_tree(HERE, manifest["files"])
    c.require(
        c.sha(p.PACKAGE / "manifest.json") == protocol["original_package_sha256"]
        and c.sha(p.PACKAGE / "plan.json") == protocol["original_plan_sha256"],
        "Original package identity changed.",
    )
    r.validate_package()
    r.verify_tree(ORIGINAL_RESULTS, protocol["preserved_sources"])
    data = p.data(checks.POOL)
    c.require(
        c.sha(p.PACKAGE / f"data/train/{checks.POOL}.json")
        == protocol["train_data_sha256"],
        "TRAIN data changed.",
    )
    c.require(
        p.config_map()[checks.CONFIG] == protocol["configuration"],
        "Original configuration changed.",
    )
    c.require(
        [x["id"] for x in data["heads"]["hate_type"]["rows"]] == protocol["fit_ids"]
        and data["heads"]["hate_type"]["labels"] == protocol["labels"],
        "TRAIN order or labels changed.",
    )
    inputs = c.read(p.PACKAGE / "data/inference.json")
    c.require(
        len(inputs) == protocol["inference_rows"]
        and c.signature(inputs) == protocol["inference_input_sha256"],
        "Text-only inference inputs changed.",
    )
    return protocol


def configure(c, r, p):

    c.require(r.OUTPUT == ORIGINAL_RESULTS, "Unexpected inherited output location.")
    r.configure()
    checks.separate_root(OUTPUT, ORIGINAL_RESULTS)
    r.OUTPUT = OUTPUT
    os.environ.update(
        HF_HOME=str(OUTPUT / "cache"), XDG_CACHE_HOME=str(OUTPUT / "cache")
    )


def capacity(c, r):
    external = r.check_external_storage(HERE / "storage_review.json")
    measured = r.storage(4_000_000_000)
    return checks.capacity(
        measured["inputs"],
        measured["outputs"],
        measured["free"],
        external["image_build_log_upper_bytes"],
        external["other_project_upper_bytes"],
    )


def time_left(c, r):
    records = []
    for started in sorted((OUTPUT / "invocations").glob("*/started.json")):
        finished = started.parent / "finished.json"
        value = c.read(finished) if finished.exists() else None
        if value:
            c.require(
                value["started_sha256"] == c.sha(started), "Invocation receipt changed."
            )
        records.append(value)
    return checks.time_budget(records, r.settings()["stop_utc"])


def idle(c, r):
    value = r.gpu()
    c.require(
        value["uuid"] == r.settings()["gpu_uuid"]
        and value["used_mib"] < 1024
        and value["busy"] < 5,
        "The assigned GPU is not idle. Leave other work alone.",
    )
    c.require(
        Path("/run/nvidia-mps/control").exists()
        and os.environ.get("CUDA_MPS_PIPE_DIRECTORY") == "/run/nvidia-mps",
        "The assigned MPS service is unavailable. Do not start another daemon.",
    )
    return value


def certificate(c, path):
    path = Path(path).resolve()
    c.require(
        path.parent.parent == OUTPUT / "preflight" and path.name == "passed.json",
        "Wrong preflight certificate path.",
    )
    value = c.read(path)
    c.require(
        value["status"] == "pass"
        and value["bundle_sha256"] == c.sha(HERE / "manifest.json")
        and value["protocol_sha256"] == c.sha(HERE / "protocol.json"),
        "Preflight identity differs.",
    )
    age = datetime.now(timezone.utc) - datetime.fromisoformat(value["utc"])
    c.require(
        timedelta(0) <= age <= timedelta(hours=1), "Refresh the preflight certificate."
    )
    for name, expected in value["test_files"].items():
        c.require(c.sha(path.parent / name) == expected, "Server test record changed.")
    return value


def preflight():
    c, r, p, study, controller = original()
    configure(c, r, p)
    with r.lock(Path("/results/worker.lock")), r.offline():
        protocol = validate(c, r, p)
        c.require(
            not (OUTPUT / "complete.json").exists(),
            "The supplement is already complete.",
        )
        remaining, end = time_left(c, r)
        gpu = idle(c, r)
        before = dict(memory=r.memory(), storage=capacity(c, r), gpu=gpu)
        root = OUTPUT / "preflight" / uuid.uuid4().hex
        root.mkdir(parents=True)
        for label, command, environment in (
            (
                "original",
                [sys.executable, "-B", str(p.PACKAGE / "tests/test_preparation.py")],
                dict(os.environ, SQ2_TEST_OUTPUT=str(root / "original")),
            ),
            (
                "supplement",
                [
                    sys.executable,
                    "-B",
                    str(HERE / "test_supplement.py"),
                    "--package",
                    str(p.PACKAGE),
                    "--output",
                    str(root / "supplement"),
                ],
                dict(os.environ),
            ),
        ):
            with (root / (label + ".log")).open("xb") as log:
                result = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=300,
                    env=environment,
                )
            c.require(
                result.returncode == 0,
                label + " CPU checks failed. No training started.",
            )
        a, b = c.read(root / "original/results.json"), c.read(
            root / "supplement/results.json"
        )
        c.require(
            a["success"]
            and a["tests"] >= 40
            and a["failures"] == a["errors"] == a["skipped"] == 0,
            "Original CPU checks incomplete.",
        )
        c.require(
            b["success"]
            and b["tests"] >= 20
            and b["failures"] == b["errors"] == b["skipped"] == 0,
            "Supplementary CPU checks incomplete.",
        )
        r.verify_tree(ORIGINAL_RESULTS, protocol["preserved_sources"])
        measured = dict(memory=r.memory(), storage=capacity(c, r), gpu=idle(c, r))
        paths = [
            root / "original.log",
            root / "supplement.log",
            root / "original/results.json",
            root / "original/results.txt",
            root / "supplement/results.json",
            root / "supplement/results.txt",
        ]
        value = dict(
            status="pass",
            utc=c.utc(),
            bundle_sha256=c.sha(HERE / "manifest.json"),
            protocol_sha256=c.sha(HERE / "protocol.json"),
            original_tests=a["tests"],
            supplementary_tests=b["tests"],
            remaining_active_seconds=remaining,
            stop_utc=end.isoformat(),
            before=before,
            after=measured,
            test_files={x.relative_to(root).as_posix(): c.sha(x) for x in paths},
            original_sources_unchanged=True,
            training_started=False,
            Qwen_calls=0,
        )
        c.save(root / "passed.json", value)
        print("PREFLIGHT PASSED", str(root / "passed.json"), flush=True)
        print(
            "Forecast GB:",
            measured["storage"]["conservative_project_forecast"] / 1e9,
            flush=True,
        )


def model_review(c, r, p, study, protocol, job):
    saved, attempt = study.receipt(job)
    meta = study.audit.metadata(
        job, p.data(checks.POOL), attempt, model_directory=saved["model_directory"]
    )
    checks.updated_model(meta, job, protocol["fit_ids"])
    inputs = c.read(p.PACKAGE / "data/inference.json")
    logits = study.check_inference(job, inputs, protocol["labels"])
    probabilities = c.probabilities("hate_type", logits)
    outputs = {}
    for policy in ("selected", "default"):
        relative = f'predictions/{checks.POOL}/tuned_{policy}_s{job["seed"]}.npz'
        source = ORIGINAL_RESULTS / relative
        c.require(
            c.sha(source) == protocol["preserved_sources"][relative],
            "Original matching predictions changed.",
        )
        with np.load(source, allow_pickle=False) as z:
            original_values = {name: z[name] for name in z.files}
        values = checks.replace_type(original_values, probabilities)
        path = OUTPUT / relative
        c.arrays(path, values)
        outputs[relative] = c.sha(path)
    for prefix in ("jobs", "inference"):
        name = f'{prefix}/{job["id"]}/complete.json'
        outputs[name] = c.sha(OUTPUT / name)
    result = dict(
        status="pass",
        job=job,
        protocol_sha256=c.sha(HERE / "protocol.json"),
        original_sources=protocol["preserved_sources"],
        outputs=outputs,
        positive_updates=sum(x["lr"] > 0 and not x["skipped"] for x in meta["steps"]),
        parameter_changes=meta["changes"],
        export_check=meta["export_check"],
        text_predictions=len(inputs),
        labels=protocol["labels"],
        cutoff=0.5,
        other_heads_reused=True,
        evaluation_gold_used=False,
    )
    c.save(OUTPUT / "audits" / f's{job["seed"]}.json', result)
    return result


def worker(assignment):
    c, r, p, study, controller = original()
    configure(c, r, p)
    protocol = validate(c, r, p)
    assignment = Path(assignment).resolve()
    c.require(
        assignment.parent.parent == OUTPUT / "invocations"
        and assignment.name == "worker.json",
        "Wrong worker assignment location.",
    )
    value = c.read(assignment)
    c.require(
        value["jobs"] == checks.jobs()
        and value["protocol_sha256"] == c.sha(HERE / "protocol.json")
        and value["bundle_sha256"] == c.sha(HERE / "manifest.json"),
        "Worker assignment changed.",
    )
    c.require(
        Path(value["session"]["stop_file"]) == assignment.parent / "STOP",
        "Worker stop-file scope changed.",
    )
    r.deadline_guard(value["session"])
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, r.request_stop)
    with r.lock(OUTPUT / "leases/worker.lock"), r.offline():
        from training.full_data import train as train

        c.require(
            os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE") == "80"
            and os.environ.get("CUDA_MPS_PIPE_DIRECTORY") == "/run/nvidia-mps",
            "Changed GPU allocation.",
        )
        c.require(
            os.environ.get("CUDA_MPS_PINNED_DEVICE_MEM_LIMIT")
            == controller.gpu_environment(1, r.gpu()["total_mib"])[
                "CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"
            ],
            "Changed GPU memory allocation.",
        )
        torch = train.setup()
        props = torch.cuda.get_device_properties(0)
        c.require(
            torch.cuda.device_count() == 1
            and props.major == 12
            and props.multi_processor_count == 150
            and torch.version.cuda == "13.0",
            "The frozen GPU profile differs.",
        )
        c.save(
            assignment.with_suffix(".device.json"),
            dict(
                name=props.name,
                sm_count=props.multi_processor_count,
                memory=props.total_memory,
                torch=torch.__version__,
                cuda=torch.version.cuda,
                percent=os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"],
                memory_limit=os.environ["CUDA_MPS_PINNED_DEVICE_MEM_LIMIT"],
            ),
        )
        data = p.data(checks.POOL)
        for job in value["jobs"]:
            r.deadline_guard(value["session"])
            r.memory()
            study.execute(job, data, value["session"])
            study.inference(job, data, value["session"])
            model_review(c, r, p, study, protocol, job)
            print("VERIFIED ADDITIONAL MODEL", job["id"], flush=True)
        c.save(
            assignment.with_suffix(".complete.json"),
            dict(assignment_sha256=c.sha(assignment), models=5, type_predictions=95410),
        )


def run(preflight_path):
    c, r, p, study, controller = original()
    configure(c, r, p)
    with r.lock(Path("/results/worker.lock")), r.offline():
        protocol = validate(c, r, p)
        certificate(c, preflight_path)
        c.require(
            not (OUTPUT / "complete.json").exists(),
            "Supplement already complete; do not train again.",
        )
        remaining, end = time_left(c, r)
        gpu = idle(c, r)
        before = dict(memory=r.memory(), storage=capacity(c, r), gpu=gpu)
        import psutil

        invocation = OUTPUT / "invocations" / uuid.uuid4().hex
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
                remaining_seconds=remaining,
                session=session,
                protocol_sha256=c.sha(HERE / "protocol.json"),
                preflight_sha256=c.sha(preflight_path),
                before=before,
            ),
        )
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, r.request_stop)
        begin = time.monotonic()
        status, detail, child = "failed", None, None
        try:
            assignment = invocation / "worker.json"
            c.save(
                assignment,
                dict(
                    jobs=checks.jobs(),
                    session=session,
                    protocol_sha256=c.sha(HERE / "protocol.json"),
                    bundle_sha256=c.sha(HERE / "manifest.json"),
                ),
            )
            env = dict(os.environ, **controller.gpu_environment(1, gpu["total_mib"]))
            with (invocation / "worker.log").open("xb") as log, (
                invocation / "telemetry.jsonl"
            ).open("x", encoding="utf8") as telemetry:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-B",
                        "-u",
                        str(HERE / "run.py"),
                        "worker",
                        str(assignment),
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=env,
                )
                c.save(
                    invocation / "worker_process.json",
                    dict(
                        pid=child.pid, created=psutil.Process(child.pid).create_time()
                    ),
                )
                last_storage = 0.0
                while child.poll() is None:
                    r.deadline_guard(session)
                    sample = dict(utc=c.utc(), memory=r.memory(), gpu=r.gpu())
                    if time.monotonic() - last_storage >= 60:
                        sample["storage"] = capacity(c, r)
                        last_storage = time.monotonic()
                    import json

                    telemetry.write(json.dumps(sample, sort_keys=True) + "\n")
                    telemetry.flush()
                    time.sleep(2)
                c.require(
                    child.returncode == 0,
                    "Worker stopped. Preserve all attempts; inspect worker.log.",
                )
            c.require(
                c.read(assignment.with_suffix(".complete.json"))
                == dict(
                    assignment_sha256=c.sha(assignment),
                    models=5,
                    type_predictions=95410,
                ),
                "Incomplete worker receipt.",
            )
            audits = {}
            for job in checks.jobs():
                r.deadline_guard(session)
                model_review(c, r, p, study, protocol, job)
                name = f'audits/s{job["seed"]}.json'
                audits[name] = c.sha(OUTPUT / name)
            r.verify_tree(ORIGINAL_RESULTS, protocol["preserved_sources"])
            final_storage = capacity(c, r)
            r.deadline_guard(session)
            c.save(
                OUTPUT / "complete.json",
                dict(
                    status="five_updated_models_and_predictions_complete",
                    utc=c.utc(),
                    models=5,
                    type_predictions=95410,
                    audits=audits,
                    protocol_sha256=c.sha(HERE / "protocol.json"),
                    bundle_sha256=c.sha(HERE / "manifest.json"),
                    original_sources_unchanged=True,
                    final_storage=final_storage,
                    original_results_replaced=False,
                    Qwen_calls=0,
                    scientific_analysis_complete=False,
                ),
            )
            status = "complete"
            print(
                "ALL FIVE TYPE MODELS AND PREDICTIONS VERIFIED. Original results retained.",
                flush=True,
            )
        except BaseException as exc:
            if child is not None and child.poll() is None:
                r.stop_children([child], session["stop_file"])
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


def launch(preflight_path):
    c, r, p, study, controller = original()
    configure(c, r, p)
    certificate(c, preflight_path)
    validate(c, r, p)
    c.require(not (OUTPUT / "complete.json").exists(), "Supplement already complete.")
    time_left(c, r)
    idle(c, r)
    capacity(c, r)
    with r.lock(Path("/results/worker.lock")):
        pass
    directory = OUTPUT / "launches" / uuid.uuid4().hex
    directory.mkdir(parents=True)
    with (directory / "controller.log").open("xb") as log:
        child = subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-u",
                str(HERE / "run.py"),
                "run",
                str(preflight_path),
            ],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=dict(os.environ),
        )
    c.save(
        directory / "launch.json",
        dict(
            pid=child.pid,
            utc=c.utc(),
            log=str(directory / "controller.log"),
            preflight_sha256=c.sha(preflight_path),
            bundle_sha256=c.sha(HERE / "manifest.json"),
        ),
    )
    time.sleep(2)
    c.require(
        child.poll() in (None, 0),
        "Controller stopped during startup; inspect its preserved log.",
    )
    print("Controller PID:", child.pid, flush=True)
    print("Log:", directory / "controller.log", flush=True)


def review_archive():
    c, r, p, study, controller = original()
    configure(c, r, p)
    with r.lock(Path("/results/worker.lock")), r.offline():
        protocol = validate(c, r, p)
        for started in (OUTPUT / "invocations").glob("*/started.json"):
            c.require(
                (started.parent / "finished.json").exists(),
                "Controller has not finished; do not package a live log.",
            )
        complete = c.read(OUTPUT / "complete.json")
        c.require(
            complete["status"] == "five_updated_models_and_predictions_complete",
            "Supplement is not complete.",
        )
        for job in checks.jobs():
            model_review(c, r, p, study, protocol, job)
        paths = [
            x
            for x in r.files(OUTPUT)
            if x.parts[len(OUTPUT.parts)]
            not in ("models", "cache", "downloads", "preflight", "leases", "launches")
        ]
        payloads = {x.relative_to(OUTPUT).as_posix(): x.read_bytes() for x in paths}
        payloads["input/protocol.json"] = (HERE / "protocol.json").read_bytes()
        payloads["input/manifest.json"] = (HERE / "manifest.json").read_bytes()
        c.require(
            sum(map(len, payloads.values())) < 50_000_000, "Unexpected review size."
        )
        import hashlib

        inventory = {
            name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()
        }
        path = OUTPUT / "downloads" / ("review_" + uuid.uuid4().hex + ".zip")
        path.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_DEFLATED) as z:
            for name, raw in sorted(payloads.items()):
                z.writestr(name, raw)
            z.writestr(
                "review_manifest.json",
                c.encoded(dict(files=inventory, weights_included=False)),
            )
        print("REVIEW ARCHIVE", path, c.sha(path), path.stat().st_size, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=("preflight", "launch", "run", "worker", "review")
    )
    parser.add_argument("path", nargs="?", type=Path)
    args = parser.parse_args()
    if args.mode in ("launch", "run", "worker"):
        (
            parser.error("A certificate/assignment path is required.")
            if args.path is None
            else None
        )
        {"launch": launch, "run": run, "worker": worker}[args.mode](args.path)
    elif args.mode == "preflight":
        preflight()
    else:
        review_archive()

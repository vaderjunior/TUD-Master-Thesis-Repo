from contextlib import contextmanager
import copy
from pathlib import Path
import subprocess
import sys
import time
import uuid
from unittest.mock import patch
import psutil
from threadpoolctl import threadpool_limits
from training.cpu import common as c
from training.cpu import run as run

ORIGINAL_BASE = c.ROOT / "experiments/astra/sq2_cpu_baselines_20261011_v1"
BASE = c.ROOT / "experiments/astra/sq2_cpu_baselines_20261012_v2"
PROBE = (
    c.ROOT
    / "experiments/astra/sq2_cpu_memory_20261012_v1/attempt_1791790332344015200_043657"
)
SOURCE_PLAN_SHA = "916e7266b2238661b7b098b2a19cd9b2457b140fa91a6ed47fc8cb6cf6dc0ac7"
ADDENDUM = c.ROOT / "Thesis_Log/astra_result/astra_sq2_cpu_memory_20261012_v1.md"
START_GIB = 3.5
RESERVE_GIB = 1.25
PRIVATE_LIMIT_GIB = 2.6
LEGACY_GUARD = c.resource_guard


def measured_guard(min_ram=START_GIB, min_disk=15.0):

    return LEGACY_GUARD(min_ram, min_disk)


@contextmanager
def execution_context():
    with patch.object(c, "BASE", BASE), patch.object(
        c, "resource_guard", measured_guard
    ):
        yield


def stop_reason(available, private, seconds, maximum_seconds=None):
    if available < RESERVE_GIB:
        return "available_RAM_below_1.25_GiB"
    if private > PRIVATE_LIMIT_GIB:
        return "CPU_worker_private_memory_above_2.6_GiB"
    if maximum_seconds is not None and seconds > maximum_seconds:
        return "technical_test_time_limit"
    return None


def terminate_owned(process, owner):
    "Only the disposable child we created and its current descendants."
    try:
        descendants = owner.children(recursive=True)
    except psutil.NoSuchProcess:
        descendants = []
    for descendant in reversed(descendants):
        try:
            descendant.terminate()
        except psutil.NoSuchProcess:
            pass
    if process.poll() is None:
        process.terminate()
    process.wait(timeout=30)


def monitored_command(command, folder, maximum_seconds=None, available_reader=None):

    available_reader = available_reader or (
        lambda: psutil.virtual_memory().available / 2**30
    )
    c.require(
        available_reader() >= START_GIB,
        "Need 3.5 GiB available before the CPU worker starts.",
    )
    folder.mkdir(parents=True, exist_ok=False)
    reason = None
    caught = None
    sampled = 0
    maximum_private = 0.0
    maximum_rss = 0.0
    minimum_available = float("inf")
    began = time.monotonic()
    identities = set()
    with (folder / "worker.log").open("x", encoding="utf8") as log:
        process = subprocess.Popen(
            command,
            cwd=c.ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        owner = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                available = available_reader()
                minimum_available = min(minimum_available, available)
                private = rss = 0
                try:
                    tree = [owner] + owner.children(recursive=True)
                except psutil.NoSuchProcess:
                    tree = []
                for member in tree:
                    try:
                        memory = member.memory_info()
                        private += memory.private
                        rss += memory.rss
                        identities.add(member.pid)
                    except psutil.NoSuchProcess:
                        pass
                maximum_private = max(maximum_private, private / 2**30)
                maximum_rss = max(maximum_rss, rss / 2**30)
                sampled += 1
                reason = stop_reason(
                    available,
                    private / 2**30,
                    time.monotonic() - began,
                    maximum_seconds,
                )
                if reason:
                    break
                time.sleep(0.5)
        except BaseException as exc:
            caught = exc
            reason = "monitor_interrupted_" + type(exc).__name__
        finally:
            if process.poll() is None:
                terminate_owned(process, owner)
    summary = dict(
        utc=c.utc(),
        exit_code=process.returncode,
        stop_reason=reason,
        seconds=time.monotonic() - began,
        sampled_min_available_GiB=minimum_available if sampled else None,
        sampled_peak_worker_tree_private_GiB=maximum_private,
        sampled_peak_worker_tree_working_set_GiB=maximum_rss,
        samples=sampled,
        worker_pids=sorted(identities),
        command=command,
        monitor_interval_seconds=0.5,
        originals_or_unrelated_processes_terminated=False,
    )
    c.save(folder / "resources.json", summary)
    if caught:
        raise caught
    c.require(
        reason is None and process.returncode == 0,
        f'CPU worker stopped ({reason or "worker error"}). Saved work is preserved. See {folder / "worker.log"}. '
        "The Qwen run was not stopped. Resume the same command when resources permit; do not bypass the check.",
    )
    return summary


def validate_probe():
    complete = c.read(PROBE / "complete.json")
    c.require(
        complete["status"] == "both_resource_probes_completed"
        and complete["source_plan_sha256"] == SOURCE_PLAN_SHA,
        "Completed resource evidence required",
    )
    source = c.ROOT / "training/cpu/memory_probe.py"
    c.require(
        c.sha(source) == complete["source_sha256"], "Resource probe source changed"
    )
    for kind in ("bge", "tfidf"):
        result = c.read(PROBE / (kind + "_child.json"))
        monitor = c.read(PROBE / (kind + "_monitor.json"))
        c.require(
            result["status"] == "resource_probe_completed"
            and result["train_rows"] == 55753
            and result["source_plan_sha256"] == SOURCE_PLAN_SHA
            and not result["F1_computed"],
            "Invalid TRAIN-only resource test",
        )
        c.require(
            monitor["exit_code"] == 0 and monitor["stopped_reason"] is None,
            "Resource test did not finish",
        )
        c.require(
            monitor["sampled_peak_tree_private_GiB"] + RESERVE_GIB + 0.35 < START_GIB,
            "Measured peak does not leave the declared reserve and launcher margin",
        )


def prepare():
    c.require(
        not (BASE / "plan.json").exists(),
        "V2 is already prepared; use preflight instead of overwriting.",
    )
    c.require(c.BASE == ORIGINAL_BASE, "Prepare starts from the original study binding")
    original = run.prep.validate()
    c.require(
        c.sha(ORIGINAL_BASE / "plan.json") == SOURCE_PLAN_SHA,
        "Original scientific plan changed",
    )
    c.require(
        not (ORIGINAL_BASE / "execution_started.json").exists(),
        "Original execution already started; review migration first",
    )
    validate_probe()
    original = copy.deepcopy(original)
    with execution_context():
        c.resource_guard(1.0, 15.0)
        prepared = {}
        for name, digest in original["prepared"].items():
            source = c.ROOT / name
            destination = BASE / source.relative_to(ORIGINAL_BASE)
            c.publish(destination, source.read_bytes())
            c.require(c.sha(destination) == digest, "Copied input differs")
            prepared[c.rel(destination)] = digest
        inputs = dict(original["inputs"])
        additional = [
            ORIGINAL_BASE / name
            for name in ("plan.json", "readiness.json", "qa/tests.json")
        ]
        additional += [
            Path(__file__),
            c.ROOT / "tests/training/cpu/test_memory.py",
            c.ROOT / "training/cpu/memory_probe.py",
            c.ROOT / "scripts/run_cpu_baselines_memory.ps1",
            ADDENDUM,
        ]
        additional += sorted(PROBE.glob("*.json"))
        for source in additional:
            inputs[c.rel(source)] = c.sha(source)
        original.update(
            utc=c.utc(),
            inputs=inputs,
            prepared=prepared,
            original_plan_sha256=SOURCE_PLAN_SHA,
            original_plan=c.rel(ORIGINAL_BASE / "plan.json"),
            operational_amendment=c.rel(ADDENDUM),
            scientific_procedures_changed=False,
            execution_module=__spec__.name,
            monitor_package_version=psutil.__version__,
            resource_policy=dict(
                min_available_RAM_GiB=START_GIB,
                min_disk_GiB=15,
                min_runtime_available_RAM_GiB=RESERVE_GIB,
                max_worker_tree_private_GiB=PRIVATE_LIMIT_GIB,
                monitor_interval_seconds=0.5,
                threads=1,
                no_parallel_CUDA=True,
                rationale="Completed largest TRAIN-only resource probes, with reserve and background-app watchdog.",
                supersedes_only_resource_and_output_location_sections_of_v1=True,
            ),
        )
        c.save(BASE / "plan.json", original)
        run.prep.validate()
        print(
            "V2 inputs copied and hashes verified. Scientific procedures unchanged. Tests and seal still required.",
            flush=True,
        )


def readiness():
    plan, plan_sha = run.readiness()
    c.verify(plan["active_source_hashes"])
    c.require(
        plan["original_plan_sha256"] == SOURCE_PLAN_SHA
        and not plan["scientific_procedures_changed"],
        "Unexpected scientific amendment",
    )
    c.require(
        plan["monitor_package_version"] == psutil.__version__,
        "Monitor package version changed",
    )
    c.require(
        plan["resource_policy"]["min_available_RAM_GiB"] == START_GIB
        and plan["resource_policy"]["min_runtime_available_RAM_GiB"] == RESERVE_GIB
        and plan["resource_policy"]["max_worker_tree_private_GiB"] == PRIVATE_LIMIT_GIB,
        "Execution resource policy changed",
    )
    return plan, plan_sha


def queue(pilot_only=False):
    _, plan_sha = readiness()

    with c.file_lock(ORIGINAL_BASE / "launcher.lock", existing=True), c.file_lock(
        BASE / "launcher.lock"
    ):
        if (BASE / "complete.json").exists():
            run.verify_complete(plan_sha)
            print("VERIFIED COMPLETE. No new fits.", flush=True)
            return
        c.current_cuda_guard()
        snapshot = c.resource_guard()
        attempt = (
            BASE / "execution" / (str(time.time_ns()) + "_" + uuid.uuid4().hex[:6])
        )
        attempt.mkdir(parents=True, exist_ok=False)
        c.save(
            attempt / "started.json",
            dict(
                utc=c.utc(),
                plan_sha256=plan_sha,
                original_plan_sha256=SOURCE_PLAN_SHA,
                resources=snapshot,
                pilot_only=pilot_only,
            ),
        )

        def child(*args):
            c.current_cuda_guard()
            c.resource_guard()
            name = "_".join(args).replace("--", "")
            print("Starting", " ".join(args), "(memory watched; saved log)", flush=True)
            result = monitored_command(
                [sys.executable, "-X", "utf8", "-B", "-u", "-m", __spec__.name, *args],
                attempt / name,
                maximum_seconds=600 if pilot_only else None,
            )
            print(
                "Completed",
                " ".join(args),
                "; peak process-tree RAM",
                round(result["sampled_peak_worker_tree_working_set_GiB"], 2),
                "GiB",
                flush=True,
            )

        child("pilot")
        if pilot_only:
            c.save(
                attempt / "complete.json",
                dict(
                    status="technical_pilot_only_completed",
                    plan_sha256=plan_sha,
                    scientific_fits=0,
                    Qwen_calls=0,
                    CUDA_used=False,
                ),
            )
            print(
                "Technical pilot only complete; full scientific queue has NOT started.",
                flush=True,
            )
            return
        if not (BASE / "execution_started.json").exists():
            c.save(
                BASE / "execution_started.json",
                dict(
                    plan_sha256=plan_sha,
                    original_plan_sha256=SOURCE_PLAN_SHA,
                    utc=c.utc(),
                    resources=snapshot,
                    Qwen_calls=0,
                    execution_module=__spec__.name,
                ),
            )
        child("neural")
        for stage, head, seed in run.group_roster():
            if run.verified_group(stage, head, seed, plan_sha) is not None:
                print("Verified saved head", stage, head, seed, flush=True)
                continue
            child("worker", "--stage", stage, "--head", head, "--seed", str(seed))
        child("assemble")
        outputs = {
            c.rel(run.group_path(stage, head, seed) / "result/complete.json"): c.sha(
                run.group_path(stage, head, seed) / "result/complete.json"
            )
            for stage, head, seed in run.group_roster()
        }
        for folder in ("pipelines", "neural_cutoffs"):
            outputs.update(
                {c.rel(p): c.sha(p) for p in (BASE / folder).glob("*/complete.json")}
            )
        path = BASE / "runtime_pilot/result/complete.json"
        outputs[c.rel(path)] = c.sha(path)
        run.prep.validate()
        run.verify_dependencies(outputs)
        c.save(
            BASE / "complete.json",
            dict(
                status="CPU_fitting_predictions_complete_review_pending",
                utc=c.utc(),
                plan_sha256=plan_sha,
                original_plan_sha256=SOURCE_PLAN_SHA,
                outputs=outputs,
                Qwen_calls=0,
                CUDA_used=False,
                scientific_review_required=True,
                operational_amendment=c.rel(ADDENDUM),
                remaining="Apply saved neural cutoffs to evaluation logits, independent scoring/paired analysis and scientific review.",
            ),
        )
        run.verify_complete(plan_sha)
        print(
            "CPU BASELINES COMPLETE. Tell Astra to inspect. Independent scientific review remains.",
            flush=True,
        )


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    if mode == "prepare":
        prepare()
        return
    with execution_context(), c.offline(), threadpool_limits(limits=1):
        if mode == "seal":
            run.prep.seal()
            readiness()
            return
        if mode == "run":
            queue()
            return
        if mode == "pilot-only":
            queue(pilot_only=True)
            return
        readiness()

        run.main()


if __name__ == "__main__":
    main()

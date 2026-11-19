import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime
import gc
import json
from pathlib import Path
import socket
from types import FunctionType, SimpleNamespace
from unittest.mock import patch
from analysis.common import execution as e
from analysis.common import runtime as resources

a = e.a
c = e.c
BASE = c.BASE / "concurrency_20261029_v1"
MANIFEST = BASE / "manifest.json"
PROTOCOL = (
    c.ROOT / "Thesis_Log/astra_result/astra_sq2_qwen_runtime_protocol_20261029_v1.md"
)
CAPACITY = (
    c.ROOT
    / "experiments/astra/sq3_transfer_20261012_v1/execution/concurrency_20261021_v3/manifest.json"
)
WORKERS = 128
PLANS = {
    "matched": "37df1b65b154826feaf7da65c09066cc111828d11cde7536b9a2cc4b01f9eb2d",
    "broad": "80e35aa16d18a8526697f17b4250b3edb15e6082f9ea5df98f9b9f153648a3cd",
}
BYTES_PER_PREDICTION = 128 * 1024
DISK_RESERVE_GIB = 8
DEADLINE = datetime.fromisoformat("2026-11-09T23:59:59+01:00")


def rel(path):
    return (
        a.storage_path(path)
        .resolve()
        .relative_to(a.storage_path(c.ROOT).resolve())
        .as_posix()
    )


def sources():
    return [
        Path(__file__),
        PROTOCOL,
        c.ROOT / "analysis/classification/broad/prepare_queue.py",
        c.ROOT / "tests/classification/broad/test_execution.py",
        c.ROOT / "scripts/run_broad_execution.ps1",
    ]


def save(path, value):
    a.require(
        a.storage_path(path).resolve().is_relative_to(a.storage_path(c.BASE).resolve()),
        "Output must stay in the later-evaluation directory",
    )
    a.save_new(path, value)


@contextmanager
def offline():
    "Fail before any socket connection, HTTP request or credential loading."
    import requests

    with c.offline(), patch.object(
        socket.socket, "connect_ex", c.forbidden
    ), patch.object(requests.sessions.Session, "request", c.forbidden), patch.object(
        e.core, "live_client", c.forbidden
    ):
        yield


def stream_rows(path):
    with a.storage_path(path).open(encoding="utf8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_catalog(study):
    helper = SimpleNamespace(**dict(vars(c), rows=stream_rows))
    namespace = dict(e.load_catalog.__globals__, c=helper)
    fn = FunctionType(e.load_catalog.__code__, namespace, e.load_catalog.__name__)
    return fn(study)


def provenance_path(b, root, stage, attempt):
    return (
        b.BASE
        / "concurrency_20261029_v1/provenance"
        / a.signature(rel(root))
        / stage
        / (attempt + ".json")
    )


def provenance(b, path, start, amendment):
    path = a.storage_path(path)
    return dict(
        original_attempt=rel(path.parent),
        original_start_sha256=a.digest(path),
        original_plan_sha256=start["plan_sha256"],
        stage=start["stage"],
        study=start["study"],
        mode=start["mode"],
        original_workers=8,
        effective_workers=WORKERS,
        amendment=amendment,
    )


def verify_provenance(b, stage, root, amendment):
    target = a.storage_path(root) / "stages" / stage["id"]
    expected = set()
    for attempt in target.glob("attempt_*"):
        start_path = attempt / "start.json"
        a.require(start_path.is_file(), "Attempt has no start receipt")
        marker = provenance_path(b, root, stage["id"], attempt.name)
        a.require(
            a.read(marker) == provenance(b, start_path, a.read(start_path), amendment),
            "Concurrency provenance changed or missing",
        )
        expected.add(marker.name)
    directory = provenance_path(b, root, stage["id"], "unused").parent
    actual = {p.name for p in directory.glob("*.json")}
    a.require(actual == expected, "Concurrency provenance roster differs")
    return len(expected)


def bind_stage(b, amendment):
    original = b.execute_stage
    a.require(
        b.WORKERS == original.__globals__["WORKERS"] == 8,
        "Original worker limit differs",
    )
    original_save = original.__globals__["save"]

    def save_with_provenance(path, value):
        original_save(path, value)
        path = a.storage_path(path)
        if path.name == "start.json" and path.parent.name.startswith("attempt_"):
            root = path.parent.parent.parent.parent
            marker = provenance_path(b, root, value["stage"], path.parent.name)
            original_save(marker, provenance(b, path, value, amendment))

    namespace = dict(original.__globals__, WORKERS=WORKERS, save=save_with_provenance)
    changed = {k for k in namespace if namespace[k] is not original.__globals__[k]}
    a.require(changed == {"WORKERS", "save"}, "Unexpected stage-engine change")
    rebound = FunctionType(
        original.__code__,
        namespace,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    rebound.__kwdefaults__ = dict(original.__kwdefaults__ or {})
    original_review = b.verify_review

    def review(stage, root, plan_sha, mode):
        receipt = original_review(stage, root, plan_sha, mode)
        verify_provenance(b, stage, root, amendment)
        return receipt

    b.execute_stage = rebound
    b.verify_review = review
    b.WORKERS = WORKERS
    b.runtime_amendment = amendment
    return b


def validate(study):
    a.require(study in PLANS, "Unknown SQ2 study")
    record = a.read(MANIFEST)
    a.require(
        record["status"] == "offline_verified_ready" and record["workers"] == WORKERS,
        "128-worker readiness is missing",
    )
    a.require(
        record["plans"] == PLANS and record["api_contract"] == a.CONTRACT,
        "Frozen generation contract changed",
    )
    a.require(
        record["environment"] == e.core.environment(), "Python environment changed"
    )
    mandatory = [
        *sources(),
        BASE / "tests.json",
        BASE / "launcher_check.json",
        BASE / "preflight.json",
        CAPACITY,
    ]
    a.require(
        all((record["bindings"].get(rel(p)) == c.sha(p) for p in mandatory)),
        "Missing or changed runtime binding",
    )
    c.verify(record["bindings"])
    plan = e.validate(study, ready=True)
    a.require(c.sha(e.queue(study).PLAN) == PLANS[study], "Original study plan changed")
    a.require(a.read(CAPACITY)["workers"] == WORKERS, "Capacity evidence differs")
    return (record, plan)


def lock_paths():
    return [
        resources.SHARED / "qwen_queue.lock",
        *(e.queue(study).BASE / "combined_launcher.lock" for study in PLANS),
    ]


@contextmanager
def queue_locks():
    with ExitStack() as stack:
        for path in lock_paths():
            stack.enter_context(a.item_lock(a.storage_path(path)))
        yield


def disk_needed(predictions):
    a.require(type(predictions) is int and predictions >= 0, "Invalid remaining count")
    return DISK_RESERVE_GIB + predictions * BYTES_PER_PREDICTION / 2**30


def host_check(remaining, initial=False):
    a.require(
        datetime.now().astimezone() <= DEADLINE,
        "Experiment deadline reached; preserve all work",
    )
    return resources.resources(2.5 if initial else 0.75, disk_needed(remaining))


def remaining_predictions(schedule, b, plan_sha):
    remaining = 0
    for stage in schedule["stages"]:
        root = b.BASE / "live" / stage["study"]
        path = a.storage_path(root) / "stages" / stage["id"] / "review.json"
        if path.exists():
            b.verify_review(stage, root, plan_sha, "live")
        else:
            remaining += stage["items"]
    return remaining


def execute_schedule(b, schedule, cat, root, plan_sha, *, mode, factory, guard=None):
    "The original stage, raw audit and finalizer remain responsible for evidence."

    def run(stage):
        store = root / stage["study"]
        verify_provenance(b, stage, store, b.runtime_amendment)
        if guard:
            guard(stage)
        b.execute_stage(stage, cat, store, plan_sha, mode=mode, client_factory=factory)

    b.run_sequence(
        schedule["stages"],
        run,
        lambda s: b.audit.audit_stage(s, cat, root / s["study"], plan_sha, mode=mode),
        lambda s: (
            a.storage_path(root) / s["study"] / "stages" / s["id"] / "review.json"
        ).exists(),
        lambda s: b.verify_review(s, root / s["study"], plan_sha, mode),
    )
    return b.audit.finalize_queue(schedule, root, plan_sha, mode=mode)


def runtime_completion(b, study, root, schedule, plan_sha, amendment):
    receipts = {}
    for stage in schedule["stages"]:
        store = root / study
        verify_provenance(b, stage, store, amendment)
        for attempt in (a.storage_path(store) / "stages" / stage["id"]).glob(
            "attempt_*"
        ):
            path = provenance_path(b, store, stage["id"], attempt.name)
            receipts[rel(path)] = c.sha(path)
    result = dict(
        status="128_worker_execution_and_original_audits_complete",
        study=study,
        original_plan_sha256=plan_sha,
        amendment=amendment,
        original_completion_sha256=c.sha(root / "complete.json"),
        workers=WORKERS,
        provenance_hashes=receipts,
        gold_scoring=False,
    )
    save(b.BASE / "concurrency_20261029_v1/live_complete.json", result)
    return result


def run(study):
    with queue_locks():
        with offline():
            _, plan = validate(study)
            amendment = dict(path=rel(MANIFEST), sha256=c.sha(MANIFEST))
            b = bind_stage(e.queue(study), amendment)
            schedule = c.read(b.BASE / "schedule.json")
            plan_sha = c.sha(b.PLAN)
            remaining = remaining_predictions(schedule, b, plan_sha)
            if remaining:
                print(
                    "Initial host check:",
                    host_check(remaining, initial=True),
                    flush=True,
                )
            cat = load_catalog(study)
            a.require(
                e.check_schedule(study, schedule, cat)["predictions"]
                == plan["budget"]["predictions"],
                "Complete frozen matrix differs",
            )
            gc.collect()
        print(
            f"{study.upper()}: up to 128 requests at once; all original prompts and saved work retained.",
            flush=True,
        )
        print(
            "One Ctrl+C drains submitted work. Keep the window open until the normal prompt returns.",
            flush=True,
        )
        print(
            "A preserved stop or unknown request requires review; never delete it or reset retries.",
            flush=True,
        )

        def guard(stage):
            with offline():
                validate(study)
                pending = remaining_predictions(schedule, b, plan_sha)
                if pending:
                    snapshot = host_check(pending)
                    print(
                        f"{stage['id']}: {pending:,} unaudited predictions; host {snapshot}",
                        flush=True,
                    )

        result = execute_schedule(
            b,
            schedule,
            cat,
            b.BASE / "live",
            plan_sha,
            mode="live",
            factory=e.core.live_client,
            guard=guard,
        )
        with offline():
            validate(study)
            runtime_completion(b, study, b.BASE / "live", schedule, plan_sha, amendment)
        a.require(
            result["counts"]["predictions"] == plan["budget"]["predictions"],
            "Final count differs",
        )
    print(
        f"{study.upper()} COMPLETE, INCLUDING ALL ORIGINAL AUDITS. Ask for result review. No next study started.",
        flush=True,
    )


def check(study):
    with offline():
        _, plan = validate(study)
        b = bind_stage(e.queue(study), dict(path=rel(MANIFEST), sha256=c.sha(MANIFEST)))
        schedule = c.read(b.BASE / "schedule.json")
        remaining = remaining_predictions(schedule, b, c.sha(b.PLAN))
        snapshot = (
            host_check(remaining, initial=True)
            if remaining
            else resources.resources(0, 0)
        )
        with queue_locks():
            pass
        print(
            json.dumps(
                dict(
                    study=study,
                    ready=True,
                    workers=WORKERS,
                    budget=plan["budget"],
                    remaining_unaudited=remaining,
                    host=snapshot,
                    actual_Qwen_calls=0,
                    actual_model_list_GETs=0,
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check", "run"])
    parser.add_argument("--study", choices=list(PLANS), required=True)
    args = parser.parse_args()
    (check if args.mode == "check" else run)(args.study)

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime
import gc
import json
from pathlib import Path
import shutil
from types import FunctionType, SimpleNamespace

from analysis.classification.broad import execution as original

a, e, c = original.a, original.e, original.c
STUDY = "broad"
SOURCE_BASE = c.BASE / "execution/broad"
BASE = Path("D:/Thesis_Experiments/sq2_broad_20261030_v1")
LOCAL = SOURCE_BASE / "storage_d_20261030_v1"
MANIFEST = LOCAL / "manifest.json"
LAYOUT = BASE / "storage_layout.json"
PLAN = SOURCE_BASE / "plan.json"
SCHEDULE = SOURCE_BASE / "schedule.json"
PROTOCOL = c.ROOT / "Thesis_Log/astra_result/astra_sq2_broad_d_protocol_20261030_v1.md"
WORKERS = 128
GPU_RESERVE_BYTES = 320_000_000_000
RUN_RESERVE_BYTES = 8 * 2**30
BYTES_PER_PREDICTION = 128 * 1024
C_RESERVE_GIB = 8


def normal(path):
    return Path(str(a.storage_path(path).resolve()).removeprefix("\\\\?\\"))


def reference(path):
    "Keep source references relative to C:; name D: outputs explicitly."
    path = normal(path)
    if path.is_relative_to(normal(BASE)):
        return path.as_posix()
    a.require(
        path.is_relative_to(normal(c.ROOT)), "Reference outside the two study roots"
    )
    return path.relative_to(normal(c.ROOT)).as_posix()


def resolve_reference(name):
    path = Path(name)
    if path.is_absolute():
        a.require(
            normal(path).is_relative_to(normal(BASE)),
            "Unexpected absolute output reference",
        )
    else:
        a.require(
            not path.drive and not path.root, "Ambiguous drive-relative reference"
        )
        path = c.ROOT / path
        a.require(
            normal(path).is_relative_to(normal(c.ROOT)),
            "Source reference escapes workspace",
        )
    a.require(reference(path) == name, "Noncanonical stored reference")
    return a.storage_path(path)


def verify(bindings):
    for name, expected in bindings.items():
        a.require(
            c.sha(resolve_reference(name)) == expected,
            "Changed storage binding: " + name,
        )


def save(path, value):
    path = normal(path)
    a.require(
        any(
            path.is_relative_to(normal(root)) and path != normal(root)
            for root in (LOCAL, BASE)
        ),
        "Write outside new storage preparation/output roots",
    )
    a.save_new(path, value)


def sources():
    return [
        Path(__file__),
        PROTOCOL,
        c.ROOT / "analysis/classification/broad/prepare_execution.py",
        c.ROOT / "tests/classification/broad/test_run.py",
        c.ROOT / "analysis/classification/broad/validate_execution.py",
        c.ROOT / "scripts/run_broad_comparison.ps1",
    ]


def clone(fn, namespace):
    result = FunctionType(
        fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__
    )
    result.__kwdefaults__ = dict(fn.__kwdefaults__ or {})
    return result


def runtime():

    namespace = dict(vars(original), rel=reference, save=save)
    names = [
        "provenance_path",
        "provenance",
        "verify_provenance",
        "bind_stage",
        "execute_schedule",
        "runtime_completion",
    ]
    for name in names:
        namespace[name] = clone(getattr(original, name), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in names})


def queue(amendment, base=None):
    if base is None:
        base = BASE
    a.require(
        normal(base).is_relative_to(normal(BASE)),
        "Queue root must stay in the D: study",
    )
    b = e.queue(STUDY)
    namespace = dict(b.execute_stage.__globals__, BASE=base, rel=reference)
    for name in e.COPIED:
        if name != "rel":
            namespace[name] = clone(getattr(b, name), namespace)

    namespace["execute_stage"].__kwdefaults__["availability_check"] = namespace[
        "availability"
    ]
    for name in e.COPIED:
        setattr(b, name, namespace[name])
    b.BASE = base
    a.require(b.PLAN == PLAN, "Scientific plan must remain the original C: file")

    return runtime().bind_stage(b, amendment)


def layout_record():
    a.require(
        BASE.is_dir(), "D: study folder is missing; do not substitute another drive"
    )
    a.require(
        normal(BASE) == BASE and normal(c.ROOT) == c.ROOT, "Study root redirected"
    )
    for path in [BASE, *BASE.parents]:
        a.require(
            not path.is_symlink() and not path.is_junction(),
            "Study storage must not use links/junctions",
        )
    return dict(
        version=1,
        study=STUDY,
        output_root=BASE.as_posix(),
        source_workspace=c.ROOT.as_posix(),
        source_plan=original.rel(PLAN),
        source_plan_sha256=original.PLANS[STUDY],
        volume_device_id=BASE.stat().st_dev,
        references="C: source-relative; canonical absolute D: output paths",
        old_outputs_moved=False,
    )


def check_layout():
    a.require(a.read(LAYOUT) == layout_record(), "D: storage identity changed")
    a.require(
        not (SOURCE_BASE / "live").exists(),
        "C: broad live work exists; inspect split storage before continuing",
    )


def required_d_bytes(remaining):
    a.require(
        type(remaining) is int and 0 <= remaining <= e.SIZES[STUDY][2],
        "Invalid pending prediction count",
    )
    return GPU_RESERVE_BYTES + RUN_RESERVE_BYTES + remaining * BYTES_PER_PREDICTION


def host_check(remaining, initial=False):
    a.require(
        datetime.now().astimezone() <= original.DEADLINE,
        "Experiment deadline reached; preserve work",
    )
    check_layout()
    host = original.resources.resources(2.5 if initial else 0.75, C_RESERVE_GIB)
    free = shutil.disk_usage(BASE).free
    needed = required_d_bytes(remaining)
    a.require(
        free >= needed,
        f"D: needs {needed / 1e9:.2f} GB free, including the GPU backup reserve; "
        f"currently {free / 1e9:.2f}. Preserve work and review storage; do not delete evidence.",
    )
    return dict(
        host,
        D_free_bytes=free,
        D_required_bytes=needed,
        D_GPU_backup_reserve_bytes=GPU_RESERVE_BYTES,
        D_remaining_output_allowance_bytes=remaining * BYTES_PER_PREDICTION,
        D_run_reserve_bytes=RUN_RESERVE_BYTES,
    )


@contextmanager
def queue_locks():
    with ExitStack() as stack:

        stack.enter_context(original.queue_locks())
        stack.enter_context(
            a.item_lock(a.storage_path(BASE / "combined_launcher.lock"))
        )
        yield


def validate():
    check_layout()
    _, plan = original.validate(STUDY)
    record = a.read(MANIFEST)
    a.require(
        record["status"] == "offline_verified_ready_on_D"
        and record["workers"] == WORKERS,
        "D: preparation is not complete",
    )
    a.require(
        record["output_root"] == BASE.as_posix()
        and record["source_plan_sha256"] == original.PLANS[STUDY]
        and record["api_contract"] == a.CONTRACT,
        "D: contract differs",
    )
    a.require(
        record["environment"] == e.core.environment(), "Python environment changed"
    )
    mandatory = [
        *sources(),
        LAYOUT,
        original.MANIFEST,
        LOCAL / "tests.json",
        LOCAL / "preflight.json",
        LOCAL / "launcher_check.json",
    ]
    a.require(
        all(record["bindings"].get(reference(p)) == c.sha(p) for p in mandatory),
        "Missing or changed storage readiness binding",
    )
    verify(record["bindings"])
    return record, plan


def amendment():
    return dict(path=reference(MANIFEST), sha256=c.sha(MANIFEST))


def check():
    with original.offline():
        _, plan = validate()
        with queue_locks():
            b = queue(amendment())
            schedule = a.read(SCHEDULE)
            pending = original.remaining_predictions(schedule, b, c.sha(PLAN))
            host = host_check(pending, initial=True)
        value = dict(
            ready=True,
            study=STUDY,
            output_root=BASE.as_posix(),
            workers=WORKERS,
            budget=plan["budget"],
            remaining_unaudited=pending,
            host=host,
            actual_Qwen_calls=0,
            actual_model_list_GETs=0,
        )
        print(json.dumps(value), flush=True)
        return value


def run():
    with original.offline():
        validate()
    with queue_locks():
        with original.offline():
            _, plan = validate()
            marker = amendment()
            b = queue(marker)
            schedule = a.read(SCHEDULE)
            plan_sha = c.sha(PLAN)
            pending = original.remaining_predictions(schedule, b, plan_sha)
            if pending:
                print(
                    "Initial host check:", host_check(pending, initial=True), flush=True
                )
            cat = original.load_catalog(STUDY)
            a.require(
                e.check_schedule(STUDY, schedule, cat)["predictions"]
                == plan["budget"]["predictions"],
                "Frozen matrix incomplete",
            )
            gc.collect()

        print(f"BROAD SQ2: 128 workers; saved outputs: {BASE}", flush=True)
        print(
            "Press Ctrl+C once to drain submitted requests; wait for the normal prompt.",
            flush=True,
        )
        print(
            "Unknown requests and preserved stops require review. Never delete evidence or reset retries.",
            flush=True,
        )

        def guard(stage):
            with original.offline():
                validate()
                pending = original.remaining_predictions(schedule, b, plan_sha)
                if pending:
                    print(
                        f'{stage["id"]}: {pending:,} unaudited predictions; host {host_check(pending)}',
                        flush=True,
                    )

        rt = runtime()
        result = rt.execute_schedule(
            b,
            schedule,
            cat,
            BASE / "live",
            plan_sha,
            mode="live",
            factory=e.core.live_client,
            guard=guard,
        )
        with original.offline():
            validate()
            rt.runtime_completion(b, STUDY, BASE / "live", schedule, plan_sha, marker)
            a.require(
                result["counts"]["predictions"] == plan["budget"]["predictions"],
                "Final count differs",
            )
            save(
                BASE / "storage_complete.json",
                dict(
                    status="broad_D_execution_and_all_audits_complete",
                    source_plan_sha256=plan_sha,
                    amendment=marker,
                    workers=WORKERS,
                    original_completion_sha256=c.sha(BASE / "live/complete.json"),
                    concurrency_completion_sha256=c.sha(
                        BASE / "concurrency_20261029_v1/live_complete.json"
                    ),
                    counts=result["counts"],
                    gold_scoring=False,
                ),
            )
    print(
        "BROAD SQ2 COMPLETE, INCLUDING ALL AUDITS. Ask for analysis. No next experiment started.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["check", "run"])
    args = parser.parse_args()
    (check if args.mode == "check" else run)()

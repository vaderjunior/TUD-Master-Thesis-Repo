import argparse
from pathlib import Path
from types import FunctionType, SimpleNamespace

from analysis.feedback.rounds import resume_earlier as previous

a, c, q = previous.a, previous.c, previous.q
original = previous.runtime.original
BASE = c.ITERATIVE / "execution/path_20261024_v2"
MANIFEST, READY = BASE / "manifest.json", BASE / "ready.json"
LAUNCHER = c.ROOT / "scripts/resume_feedback_rounds_paths.ps1"


def project_relative(path):

    return (
        a.storage_path(path)
        .resolve()
        .relative_to(a.storage_path(c.ROOT).resolve())
        .as_posix()
    )


def bind(fn, **changes):
    unknown = set(changes) - set(fn.__globals__)
    a.require(not unknown, "Unexpected function namespace change")
    result = FunctionType(
        fn.__code__,
        dict(fn.__globals__, **changes),
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )
    result.__kwdefaults__ = dict(fn.__kwdefaults__ or {})
    return result


def extractor():

    paths = SimpleNamespace(**dict(vars(c), rel=project_relative))
    return bind(original.extract, c=paths)


def extract(b):
    values, bindings = extractor()(b)
    if (
        a.storage_path(b.BASE).resolve()
        == a.storage_path(previous.PHASE / "execution").resolve()
    ):
        proof = a.read(BASE / "preflight.json")
        a.require(
            a.digest(b.BASE / "source_gates.json")
            == proof["expected_source_gates_sha256"]
            and a.digest(b.BASE / "source_gates_receipt.json")
            == proof["expected_source_receipt_sha256"],
            "Initial extraction differs from the checked saved responses",
        )
    marker = dict(
        status="source_extraction_with_normalized_project_paths",
        path_manifest_sha256=a.digest(MANIFEST),
        parent_outage_manifest_sha256=a.digest(previous.MANIFEST),
        phase=project_relative(b.BASE.parent),
        plan_sha256=a.digest(b.PLAN),
        source_phase_complete_sha256=a.digest(b.BASE / "live/complete.json"),
        source_gates_sha256=a.digest(b.BASE / "source_gates.json"),
        source_gates_receipt_sha256=a.digest(b.BASE / "source_gates_receipt.json"),
        source_inputs=len(values),
        actual_extraction_API_calls=0,
    )
    c.save(b.BASE / "path_20261024_v2/extract.json", marker)
    return values, bindings


def bound_controller():

    return bind(previous.bound_controller(), extract=extract)


def validate(*, ready=True):
    previous.validate()
    manifest = a.read(MANIFEST)
    a.require(
        manifest["status"] == "offline_verified_windows_path_continuation"
        and manifest["parent_outage_manifest_sha256"] == a.digest(previous.MANIFEST)
        and manifest["parent_runtime_manifest_sha256"]
        == a.digest(previous.runtime.MANIFEST)
        and manifest["parent_scientific_plan_sha256"]
        == a.digest(c.ITERATIVE / "plan.json"),
        "Path amendment identity differs",
    )
    a.require(
        manifest["effective_workers"] == 128
        and manifest["new_transport_allowance"] == 0
        and manifest["new_scientific_predictions"] == 0
        and manifest["api_contract"] == a.CONTRACT,
        "Path amendment scope differs",
    )
    previous.transport.verify_files(manifest["source_hashes"])
    previous.transport.verify_files(manifest["evidence_hashes"])
    previous.transport.verify_files(manifest["checkpoint_hashes"])
    a.require(
        a.read(BASE / "tests.json")["passed"]
        and a.read(BASE / "preflight.json")["passed"],
        "Path checks incomplete",
    )
    initial = q.queue(previous.PHASE, "source", original.SOURCES)
    previous.bind_for_phase(initial, previous.original_amendment())
    schedule = a.read(initial.BASE / "schedule.json")
    for stage in schedule["stages"]:
        initial.verify_review(
            stage, initial.BASE / "live/source", a.digest(initial.PLAN), "live"
        )
    complete = a.read(initial.BASE / "live/complete.json")
    a.require(
        complete["counts"]["predictions"] == 3072
        and complete["accounting"]["ordinary_logical_calls"] == 9216
        and complete["reviewed_transport_amendment"] == previous.provenance(),
        "Initial screening is not the reviewed completed checkpoint",
    )
    if ready:
        receipt = a.read(READY)
        a.require(
            receipt["status"] == "offline_verified_ready_for_user"
            and receipt["manifest_sha256"] == a.digest(MANIFEST),
            "Path continuation not sealed",
        )
        previous.transport.verify_files(receipt["verification_hashes"])
    return manifest


def final_check():
    previous.final_check()
    done = a.read(c.ITERATIVE / "complete.json")
    markers = {}
    counts = []
    for name in done["phase_receipts"]:
        phase = (c.ROOT / name).parent.parent.parent
        if phase.name != "source_initial" and "_source_" not in phase.name:
            continue
        execution = phase / "execution"
        path = execution / "path_20261024_v2/extract.json"
        marker = a.read(path)
        expected = dict(
            status="source_extraction_with_normalized_project_paths",
            path_manifest_sha256=a.digest(MANIFEST),
            parent_outage_manifest_sha256=a.digest(previous.MANIFEST),
            phase=project_relative(phase),
            plan_sha256=a.digest(execution / "plan.json"),
            source_phase_complete_sha256=a.digest(execution / "live/complete.json"),
            source_gates_sha256=a.digest(execution / "source_gates.json"),
            source_gates_receipt_sha256=a.digest(
                execution / "source_gates_receipt.json"
            ),
            source_inputs=1024 if phase.name == "source_initial" else 256,
            actual_extraction_API_calls=0,
        )
        a.require(marker == expected, "Source extraction provenance differs")
        c.verify(a.read(execution / "source_gates_receipt.json")["bindings"])
        markers[project_relative(path)] = a.digest(path)
        counts.append(marker["source_inputs"])
    a.require(
        len(markers) == 10 and sorted(counts) == [256] * 9 + [1024],
        "Ten source extractions required",
    )
    c.save(
        BASE / "live_path_runtime_complete.json",
        dict(
            status="all_rounds_outage_recovery_and_path_continuation_verified_scientific_review_pending",
            manifest_sha256=a.digest(MANIFEST),
            source_extraction_markers=markers,
            original_complete_sha256=a.digest(c.ITERATIVE / "complete.json"),
            runtime_complete_sha256=a.digest(
                previous.runtime.BASE / "live_runtime_complete.json"
            ),
            outage_complete_sha256=a.digest(
                previous.BASE / "live_recovery_complete.json"
            ),
            actual_finalization_API_calls=0,
        ),
    )
    print(
        "All feedback rounds and continuation checks complete. Ask Astra to review the saved results.",
        flush=True,
    )


def run():
    with a.item_lock(a.storage_path(previous.runtime.BASE / "runtime_launcher.lock")):
        validate()
        c.resources(2.5, 20.0)
        bound_controller()()
        final_check()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["validate", "run"], nargs="?", default="validate"
    )
    args = parser.parse_args()
    with original.threadpool_limits(limits=1):
        if args.mode == "run":
            run()
        else:
            with previous.inspection.offline():
                validate()
                c.resources(2.5, 20.0)
                print(
                    "Offline check passed: all 3,072 initial-screening predictions are complete and preserved. Path continuation ready; zero Qwen calls.",
                    flush=True,
                )

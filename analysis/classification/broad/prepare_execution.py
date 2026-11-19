import argparse
from collections import Counter
import gc
import math
from pathlib import Path
import time

from analysis.classification.broad import run as r
from analysis.classification.broad import prepare_queue as previous

a, e, c = r.a, r.e, r.c


def preservation():
    bindings = a.read(r.original.BASE / "preservation_before.json")
    matched = c.ROOT / "experiments/astra/sq2_matched_review_20261030_v1"
    bindings.update(a.read(matched / "preservation_inputs.json"))
    bindings.update(a.read(matched / "complete.json")["outputs"])
    for path in [
        matched / "complete.json",
        r.original.MANIFEST,
        r.original.BASE / "independent_final_review.json",
        e.queue("matched").BASE / "live/complete.json",
        e.queue("matched").BASE / "concurrency_20261029_v1/live_complete.json",
        c.ROOT
        / "experiments/astra/sq2_server_full_live_20261029_v1/running_checkpoint.json",
    ]:
        bindings[r.original.rel(path)] = c.sha(path)
    c.verify(bindings)
    return bindings


def initialize():
    with r.original.offline():
        r.original.validate("broad")
        a.require(not (r.SOURCE_BASE / "live").exists(), "C: broad work already exists")
        a.require(not (r.BASE / "live").exists(), "D: broad work already exists")
        r.save(r.LOCAL / "preservation_before.json", preservation())
        if not r.LAYOUT.exists():
            a.require(
                not r.BASE.exists() or not any(r.BASE.iterdir()),
                "Unrecognized D: folder; preserve it",
            )
            r.BASE.mkdir(parents=True, exist_ok=True)
            r.save(r.LAYOUT, r.layout_record())
        r.check_layout()
        print(
            "D: study location created and source evidence verified. No network calls.",
            flush=True,
        )


def rehearse(schedule, cat):
    marker = dict(
        path=r.reference(r.PROTOCOL), sha256=c.sha(r.PROTOCOL), synthetic=True
    )
    rt = r.runtime()
    queue_base = r.BASE / "offline_check" / ("matrix_" + str(time.time_ns()))
    b = r.queue(marker, base=queue_base)
    root = queue_base / "offline_check/run"
    stages = previous.selected_stages("broad", schedule)
    selected = dict(stages=stages)
    transport = e.SyntheticTransport()
    began = time.perf_counter()
    result = rt.execute_schedule(
        b, selected, cat, root, c.sha(r.PLAN), mode="mock", factory=lambda: transport
    )
    first_seconds = time.perf_counter() - began
    count = transport.count
    again = time.perf_counter()
    restarted = rt.execute_schedule(
        b, selected, cat, root, c.sha(r.PLAN), mode="mock", factory=c.forbidden
    )
    restart_seconds = time.perf_counter() - again
    a.require(
        restarted == result
        and count == transport.count == sum(s["ordinary_calls"] for s in stages),
        "Completed restart made a new synthetic request or changed completion",
    )
    rt.runtime_completion(b, "broad", root, selected, c.sha(r.PLAN), marker)
    runtime_receipt = queue_base / "concurrency_20261029_v1/live_complete.json"
    bindings = {
        r.reference(p): c.sha(p) for p in [root / "complete.json", runtime_receipt]
    }
    peak = 0
    counts = Counter()
    for stage in stages:
        target = a.storage_path(root / "broad/stages" / stage["id"])
        review = b.verify_review(stage, root / "broad", c.sha(r.PLAN), "mock")
        report = a.read(target / review["report"])
        for name in report["attempt_hashes"]:
            if name.endswith("/summary.json"):
                peak = max(
                    peak,
                    a.read(a.storage_path(root / "broad") / name)["peak_submitted"],
                )
        for attempt in target.glob("attempt_*"):
            path = rt.provenance_path(b, root / "broad", stage["id"], attempt.name)
            bindings[r.reference(path)] = c.sha(path)
        bindings[r.reference(target / "review.json")] = c.sha(target / "review.json")
    for path in a.storage_path(root).rglob("*"):
        if path.is_file():
            counts["files"] += 1
            counts["logical_bytes"] += path.stat().st_size
            counts["estimated_4KiB_allocated_bytes"] += max(
                4096, math.ceil(path.stat().st_size / 4096) * 4096
            )
    a.require(peak == 128, "D: rehearsal did not use 128 submitted jobs")
    return dict(
        root=r.reference(root),
        queue_base=r.reference(queue_base),
        stages=stages,
        predictions=sum(s["items"] for s in stages),
        synthetic_responses=count,
        additional_responses_on_restart=0,
        peak_submitted=peak,
        first_execution_and_audit_seconds=first_seconds,
        completed_restart_seconds=restart_seconds,
        storage=dict(counts),
        bindings=bindings,
        runtime_completion_checked=True,
    )


def prepare():
    with r.original.offline():
        r.check_layout()
        r.original.validate("broad")
        tests = a.read(r.LOCAL / "tests.json")
        a.require(tests["passed"], "Passing storage tests required")
        r.verify(tests["source_hashes"])
        a.require(
            not (r.BASE / "live").exists(), "Scientific generation already exists"
        )
        initial_host = r.host_check(e.SIZES["broad"][2], initial=True)
        cat = r.original.load_catalog("broad")
        schedule = a.read(r.SCHEDULE)
        fingerprint = previous.fingerprint(cat)
        old = a.read(r.original.BASE / "preflight.json")["studies"]["broad"]
        a.require(
            fingerprint == old["catalogue_sha256"],
            "Catalogue changed since original 128-worker review",
        )
        matrix = previous.independent_matrix("broad", schedule, cat)
        a.require(matrix == old["matrix"], "Frozen wire matrix changed")
        a.require(
            e.make_schedule("broad", cat) == schedule,
            "Original schedule no longer reconstructs",
        )
        a.require(
            e.check_schedule("broad", schedule, cat)["predictions"]
            == matrix["predictions"],
            "Matrix checks differ",
        )
        b = r.queue(
            dict(path=r.reference(r.PROTOCOL), sha256=c.sha(r.PROTOCOL), synthetic=True)
        )
        old_b = e.queue("broad")
        identities = set()
        plan_sha = c.sha(r.PLAN)
        for stage in schedule["stages"]:
            for job in stage["jobs"]:
                before = old_b.identity(job, plan_sha)
                a.require(
                    b.identity(job, plan_sha) == before,
                    "Storage changed scientific identity",
                )
                identities.add(a.signature(before))
        a.require(len(identities) == 194643, "D: output identity collision")
        gc.collect()
        print(
            "All 194,643 frozen identities and wire payloads match. Starting synthetic D: rehearsal.",
            flush=True,
        )
        rehearsal = rehearse(schedule, cat)
        c.verify(a.read(r.LOCAL / "preservation_before.json"))
        value = dict(
            status="D_full_matrix_and_synthetic_rehearsal_passed",
            catalogue_entries=len(cat),
            catalogue_sha256=fingerprint,
            matrix=matrix,
            unchanged_identity_count=len(identities),
            rehearsal=rehearsal,
            host_before_catalogue=initial_host,
            host_after_rehearsal=r.host_check(194643),
            source_hashes={r.reference(p): c.sha(p) for p in r.sources()},
            environment=e.core.environment(),
            original_evidence_unchanged=True,
            actual_Qwen_calls=0,
            actual_model_list_GETs=0,
            GPU_accesses=0,
            training_fits=0,
        )
        r.save(r.LOCAL / "preflight.json", value)
        print("D: offline preflight passed; no live queue started.", flush=True)


def seal():
    with r.original.offline(), r.queue_locks():
        r.check_layout()
        tests = a.read(r.LOCAL / "tests.json")
        preflight = a.read(r.LOCAL / "preflight.json")
        launcher = a.read(r.LOCAL / "launcher_check.json")
        a.require(
            tests["passed"]
            and preflight["status"] == "D_full_matrix_and_synthetic_rehearsal_passed",
            "Checks incomplete",
        )
        for record in (tests, preflight):
            r.verify(record["source_hashes"])
            a.require(
                record["environment"] == e.core.environment(), "Environment changed"
            )
            a.require(
                record["actual_Qwen_calls"] == record["actual_model_list_GETs"] == 0,
                "Offline boundary violated",
            )
        a.require(
            all(launcher[k] is True for k in ["parsed", "set_succeeded", "restored"]),
            "Launcher check failed",
        )
        a.require(
            launcher["launcher_sha256"] == c.sha(r.sources()[-1]),
            "Launcher changed since check",
        )
        c.verify(a.read(r.LOCAL / "preservation_before.json"))
        a.require(not (r.BASE / "live").exists(), "Do not seal over scientific output")
        bindings = {
            r.reference(p): c.sha(p)
            for p in [
                *r.sources(),
                r.LAYOUT,
                r.original.MANIFEST,
                r.LOCAL / "tests.json",
                r.LOCAL / "preflight.json",
                r.LOCAL / "launcher_check.json",
                r.LOCAL / "preservation_before.json",
            ]
        }
        bindings.update(preflight["rehearsal"]["bindings"])
        r.verify(bindings)
        r.save(
            r.MANIFEST,
            dict(
                status="offline_verified_ready_on_D",
                study="broad",
                workers=128,
                source_plan_sha256=r.original.PLANS["broad"],
                output_root=r.BASE.as_posix(),
                api_contract=a.CONTRACT,
                environment=e.core.environment(),
                bindings=bindings,
                original_concurrency_manifest_sha256=c.sha(r.original.MANIFEST),
                storage_only=True,
                old_outputs_moved=False,
                actual_Qwen_calls=0,
                actual_model_list_GETs=0,
                GPU_accesses=0,
                training_fits=0,
            ),
        )
        r.validate()
        print(
            "D: runtime sealed. Final separate review and exact CheckOnly command remain.",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["initialize", "prepare", "seal"])
    args = parser.parse_args()
    {"initialize": initialize, "prepare": prepare, "seal": seal}[args.mode]()

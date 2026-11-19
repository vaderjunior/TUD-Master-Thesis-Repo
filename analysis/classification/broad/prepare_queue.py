import argparse
from collections import Counter
import gc
import hashlib
import json
import math
from pathlib import Path
import time

from analysis.classification.broad import execution as r

a, e, c = r.a, r.e, r.c


def preservation():
    paths = {}
    for study, expected in r.PLANS.items():
        b = e.queue(study)
        plan = e.validate(study, True)
        a.require(c.sha(b.PLAN) == expected, "Frozen plan identity differs")
        paths.update(plan["input_hashes"])
        paths.update(plan["source_hashes"])
        ready = c.read(b.BASE / "ready.json")
        for path in [b.PLAN, b.BASE / "ready.json", c.ROOT / ready["evidence"]]:
            paths[r.rel(path)] = c.sha(path)
        a.require(
            not (b.BASE / "live").exists(),
            "Unexpected existing live SQ2 work; inspect before sealing",
        )
    gpu_before = (
        c.ROOT
        / "experiments/astra/sq2_server_full_live_20261029_v1/preservation_full_before.json"
    )
    paths.update(c.read(gpu_before))
    for path in [
        gpu_before,
        c.ROOT
        / "experiments/astra/sq2_server_full_live_20261029_v1/running_checkpoint.json",
        r.CAPACITY,
        c.BASE / "scoring_contract_v1.json",
        c.BASE / "replay_review_20261009_v1/sensitivity_plan.json",
    ]:
        paths[r.rel(path)] = c.sha(path)
    c.verify(paths)
    return paths


def fingerprint(cat):
    h = hashlib.sha256()
    for key, value in sorted(cat.items()):
        h.update(a.encoded([list(key), value]))
    return h.hexdigest()


def independent_matrix(study, schedule, cat):
    n, settings, predictions = e.SIZES[study]
    roster = Counter()
    stages = set()
    input_owner = {}
    wire = hashlib.sha256()
    conditions = {condition for key, condition in cat}
    populations = Counter()
    for stage in schedule["stages"]:
        a.require(stage["id"] not in stages, "Duplicate stage ID")
        stages.add(stage["id"])
        a.require(
            stage["study"] == study and stage["items"] == len(stage["jobs"]),
            "Stage identity/count differs",
        )
        a.require(
            stage["ordinary_calls"] == len(stage["jobs"]) * 3 <= 12000,
            "Batch cap differs",
        )
        a.require(
            set(stage["input_keys"]) == {job["key"] for job in stage["jobs"]},
            "Stage input list differs",
        )
        for job in stage["jobs"]:
            ident = job["key"], job["condition"], job["repeat"]
            roster[ident] += 1
            a.require(
                job["repeat"] in (0, 1, 2) and job["study"] == study,
                "Wrong repeat/study",
            )
            entry = cat[job["key"], job["condition"]]
            a.require(
                input_owner.setdefault(job["key"], stage["id"]) == stage["id"],
                "Input split across batches",
            )
            a.require(
                all(
                    job[k] == entry[k]
                    for k in [
                        "messages_sha256",
                        "schema_path",
                        "schema_sha256",
                        "population",
                    ]
                ),
                "Job differs from original saved prompt",
            )
            expected_body = dict(
                model="Qwen/Qwen3-VL-32B-Instruct",
                temperature=1.0,
                max_tokens=4096,
                messages=[
                    dict(role=m["role"], content=m["text"]) for m in entry["messages"]
                ],
            )
            a.require(
                a.body(entry["messages"]) == expected_body,
                "Wire transformation differs",
            )
            wire.update(a.encoded([list(ident), expected_body]))
            populations[job["population"]] += 1
    a.require(
        len(roster) == predictions and set(roster.values()) == {1},
        "Missing/duplicate prediction",
    )
    expected = {
        (key, condition, repeat) for key, condition in cat for repeat in range(3)
    }
    a.require(
        set(roster) == expected
        and len(input_owner) == n
        and len(conditions) == settings,
        "Incomplete matrix",
    )
    return dict(
        predictions=predictions,
        ordinary_calls=predictions * 3,
        batches=len(stages),
        conditions=sorted(conditions),
        population_predictions=dict(populations),
        independently_rebuilt_wire_matrix_sha256=wire.hexdigest(),
    )


def selected_stages(study, schedule):
    selected = []
    seen = set()
    for stage in schedule["stages"]:
        if stage["population"] in seen:
            continue
        seen.add(stage["population"])
        width = 4 if study == "matched" else 15
        keys = set(stage["input_keys"][:width])
        jobs = [j for j in stage["jobs"] if j["key"] in keys]
        selected.append(
            dict(
                stage,
                input_keys=sorted(keys),
                jobs=jobs,
                items=len(jobs),
                ordinary_calls=len(jobs) * 3,
            )
        )
    return selected


def rehearse(study, schedule, cat):
    b = e.queue(study)
    fixture = b.BASE / "offline_check" / ("workers128_" + str(time.time_ns()))
    amendment = dict(path=r.rel(r.PROTOCOL), sha256=c.sha(r.PROTOCOL), synthetic=True)
    b = r.bind_stage(b, amendment)
    stages = selected_stages(study, schedule)
    reduced = dict(stages=stages)
    transport = e.SyntheticTransport()
    started = time.perf_counter()
    result = r.execute_schedule(
        b, reduced, cat, fixture, c.sha(b.PLAN), mode="mock", factory=lambda: transport
    )
    count = transport.count
    restarted = r.execute_schedule(
        b, reduced, cat, fixture, c.sha(b.PLAN), mode="mock", factory=c.forbidden
    )
    a.require(
        result == restarted
        and count == transport.count == sum(s["ordinary_calls"] for s in stages),
        "Rehearsal restart changed results or sent another request",
    )
    bindings = {r.rel(fixture / "complete.json"): c.sha(fixture / "complete.json")}
    peak = 0
    file_count = 0
    allocated = 0
    for stage in stages:
        store = fixture / study
        target = a.storage_path(store) / "stages" / stage["id"]
        review = b.verify_review(stage, store, c.sha(b.PLAN), "mock")
        report = a.read(target / review["report"])
        bindings[r.rel(target / "review.json")] = c.sha(target / "review.json")
        for name in report["attempt_hashes"]:
            if name.endswith("/summary.json"):
                summary = a.read(a.storage_path(store) / name)
                peak = max(peak, summary["peak_submitted"])
        for attempt in target.glob("attempt_*"):
            marker = r.provenance_path(b, store, stage["id"], attempt.name)
            bindings[r.rel(marker)] = c.sha(marker)
    for path in a.storage_path(fixture).rglob("*"):
        if path.is_file():
            file_count += 1
            allocated += max(4096, math.ceil(path.stat().st_size / 4096) * 4096)
    predictions = sum(s["items"] for s in stages)
    a.require(peak == 128, "Rehearsal did not submit 128 actual concurrent worker jobs")
    return dict(
        root=r.rel(fixture),
        predictions=predictions,
        simulated_replies=count,
        new_calls_on_completed_restart=0,
        peak_submitted=peak,
        populations=sorted({s["population"] for s in stages}),
        estimated_4KiB_allocated_bytes=allocated,
        files=file_count,
        allocated_bytes_per_prediction=allocated / predictions,
        elapsed_seconds=time.perf_counter() - started,
        bindings=bindings,
    )


def prepare():
    with r.offline():
        before = preservation()
        r.save(r.BASE / "preservation_before.json", before)
        tests = a.read(r.BASE / "tests.json")
        a.require(
            tests["passed"]
            and tests["actual_Qwen_calls"] == tests["actual_model_list_GETs"] == 0,
            "Passing offline tests required",
        )
        c.verify(tests["source_hashes"])
        capacity = a.read(r.CAPACITY)
        a.require(
            capacity["status"] == "tested_ready"
            and capacity["workers"] == 128
            and capacity["api_contract"] == a.CONTRACT,
            "Capacity evidence changed",
        )
        c.verify(capacity["bindings"])
        out = {}
        for study in r.PLANS:
            print(
                study.upper() + ": comparing original and streamed catalogues",
                flush=True,
            )
            original = e.load_catalog(study)
            original_hash = fingerprint(original)
            del original
            gc.collect()
            cat = r.load_catalog(study)
            a.require(
                fingerprint(cat) == original_hash,
                "Streamed catalogue differs from original",
            )
            b = e.queue(study)
            schedule = c.read(b.BASE / "schedule.json")
            matrix = independent_matrix(study, schedule, cat)
            a.require(
                e.check_schedule(study, schedule, cat)["predictions"]
                == matrix["predictions"],
                "Schedule checks disagree",
            )
            a.require(
                e.make_schedule(study, cat) == schedule,
                "Original schedule reconstruction differs",
            )
            print(
                study.upper()
                + ": complete matrix verified; synthetic 128-worker rehearsal follows",
                flush=True,
            )
            rehearsal = rehearse(study, schedule, cat)
            out[study] = dict(
                catalogue_sha256=original_hash,
                catalogue_entries=len(cat),
                matrix=matrix,
                rehearsal=rehearsal,
                host_after_catalogue=r.resources.resources(0, 0),
            )
            del cat, schedule
            gc.collect()
        c.verify(before)
        r.save(
            r.BASE / "preservation_after.json",
            dict(
                unchanged=True,
                bindings=len(before),
                before_sha256=c.sha(r.BASE / "preservation_before.json"),
            ),
        )
        result = dict(
            status="full_matrix_and_128_worker_rehearsals_verified",
            studies=out,
            preservation_bindings=len(before),
            original_sources_unchanged=True,
            actual_Qwen_calls=0,
            actual_model_list_GETs=0,
            training_fits=0,
            server_accesses=0,
            source_hashes={r.rel(p): c.sha(p) for p in r.sources()},
            environment=e.core.environment(),
        )
        r.save(r.BASE / "preflight.json", result)
        print(
            "BOTH OFFLINE PREFLIGHTS COMPLETE. No live requests or GPU operations.",
            flush=True,
        )


def seal():
    with r.offline():
        tests = a.read(r.BASE / "tests.json")
        preflight = a.read(r.BASE / "preflight.json")
        awake = a.read(r.BASE / "launcher_check.json")
        a.require(
            tests["passed"]
            and preflight["status"] == "full_matrix_and_128_worker_rehearsals_verified",
            "Offline checks incomplete",
        )
        a.require(
            all(awake[k] is True for k in ["parsed", "set_succeeded", "restored"]),
            "Launcher check failed",
        )
        for value in (tests, preflight):
            c.verify(value["source_hashes"])
            a.require(
                value["environment"] == e.core.environment(),
                "Environment changed after tests",
            )
            a.require(
                value["actual_Qwen_calls"] == value["actual_model_list_GETs"] == 0,
                "Offline boundary violated",
            )
        bindings = {
            r.rel(p): c.sha(p)
            for p in [
                *r.sources(),
                r.CAPACITY,
                r.BASE / "tests.json",
                r.BASE / "preflight.json",
                r.BASE / "launcher_check.json",
                r.BASE / "preservation_before.json",
                r.BASE / "preservation_after.json",
            ]
        }
        a.require(
            awake["launcher_sha256"]
            == c.sha(c.ROOT / "scripts/run_broad_execution.ps1"),
            "Launcher changed",
        )
        for study in r.PLANS:
            e.validate(study, True)
            bindings.update(preflight["studies"][study]["rehearsal"]["bindings"])
            for path in [e.queue(study).PLAN, e.queue(study).BASE / "ready.json"]:
                bindings[r.rel(path)] = c.sha(path)
        c.verify(bindings)
        c.verify(c.read(r.BASE / "preservation_before.json"))
        with r.queue_locks():
            a.require(
                all(not (e.queue(s).BASE / "live").exists() for s in r.PLANS),
                "Live work already exists",
            )
            r.save(
                r.MANIFEST,
                dict(
                    status="offline_verified_ready",
                    workers=128,
                    original_workers=8,
                    plans=r.PLANS,
                    api_contract=a.CONTRACT,
                    environment=e.core.environment(),
                    bindings=bindings,
                    generation_scope_changed=False,
                    one_queue_at_a_time=True,
                    actual_Qwen_calls=0,
                    actual_model_list_GETs=0,
                ),
            )
        print(
            "SEALED: matched and broad ready for user launch; neither started.",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "seal"])
    args = parser.parse_args()
    (prepare if args.mode == "prepare" else seal)()

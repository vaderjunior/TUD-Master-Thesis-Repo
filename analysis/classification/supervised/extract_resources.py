from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import sys
import numpy as np
from analysis.classification.supervised import resources as b
from analysis.classification.supervised import archive as r
from analysis.classification.supervised import metrics as m

OUT = b.BASE / "extraction"


def utc(value):
    result = datetime.fromisoformat(value)
    r.require(result.tzinfo is not None, "Naive resource timestamp.")
    return result.timestamp()


def extract():
    for path, digest in r.read(b.BASE / "analysis_contract.json")["inputs"].items():
        r.checked_file(r.ROOT / path, digest)
    OUT.mkdir(exist_ok=False)
    study = r.GpuStudy()
    alljobs = []
    historical = []
    sources = {}
    inventories = {}
    assignments = []
    inferences = []
    bindings = {}
    small_duplicates = {}
    receipt = r.read(b.TRAIN / "complete.json")
    for pool in r.POOLS:
        for leaf in (
            "canonical_jobs.json",
            "historical_attempts.json",
            "selection_review.json",
        ):
            name = pool + "/" + leaf
            path = b.TRAIN / name
            r.checked_file(path, receipt["outputs"][name], sources)
        jobs = r.read(b.TRAIN / pool / "canonical_jobs.json")
        history = r.read(b.TRAIN / pool / "historical_attempts.json")
        alljobs += jobs
        historical += history
        with study.pool(pool) as view:
            lookup = {x["job"]["id"]: x for x in jobs}
            totals = defaultdict(lambda: dict(files=0, logical_bytes=0))
            weights = []
            extras = []
            for name, (part, spec) in view.store.entries.items():
                category = (
                    "weights"
                    if name.endswith("/model.safetensors")
                    else (
                        "model_ancillary"
                        if name.startswith("models/")
                        else name.split("/")[0]
                    )
                )
                totals[category]["files"] += 1
                totals[category]["logical_bytes"] += spec["bytes"]
                if category in ("weights", "model_ancillary"):
                    item = dict(
                        path=name,
                        bytes=spec["bytes"],
                        sha256=spec["sha256"],
                        zip_part=part,
                        job="/".join(name.split("/")[1:5]),
                    )
                    (weights if category == "weights" else extras).append(item)
                if category in ("selections", "audits"):
                    small_duplicates[name] = spec
            for family in r.FAMILIES:
                for seed in r.SEEDS:
                    heads = {}
                    for head in r.HEADS:
                        job = view.job(head, family, seed)
                        row = lookup[job["id"]]
                        match = [x for x in weights if x["job"] == job["id"]]
                        r.require(len(match) == 1, "Final model size identity differs.")
                        heads[head] = dict(
                            job=job["id"],
                            weight_bytes=match[0]["bytes"],
                            model_files_bytes=sum(
                                x["bytes"]
                                for x in extras + weights
                                if x["job"] == job["id"]
                            ),
                            state=row["state"],
                            total_seconds=row["total_seconds"],
                            train_seconds=row["train_seconds"],
                            steps=row["planned_steps"],
                        )
                    assignments.append(
                        dict(
                            pool=pool,
                            budget=m.budget(pool),
                            family=family,
                            seed=seed,
                            heads=heads,
                        )
                    )
            for job in view.jobs:
                row = lookup[job]
                name = "inference/" + job + "/complete.json"
                inf = view.store.json(name)
                r.require(
                    inf["binding"]["identity"]["job"]["id"] == job
                    and inf["binding"]["rows"] == 19082
                    and inf["binding"]["training_receipt_sha256"]
                    == row["receipt_sha256"],
                    "Inference identity differs.",
                )
                inferences.append(
                    dict(
                        job=job,
                        pool=pool,
                        head=row["job"]["head"],
                        rows=inf["binding"]["rows"],
                        receipt_sha256=view.store.expected(name),
                        timer_recorded=any("second" in k or "time" in k for k in inf),
                    )
                )

                training = view.store.json(row["attempt"] + "/training.json")
                r.require(
                    training["total_seconds"] == row["total_seconds"]
                    and training["train_seconds"] == row["train_seconds"],
                    "Canonical timing cache differs from fit record.",
                )
                row["peak_allocated_gib"] = training["peak_allocated_gib"]
                row["peak_reserved_gib"] = training["peak_reserved_gib"]

            sampled = []
            for head in r.HEADS:
                for config in study.configs:
                    matches = [
                        x
                        for x in jobs
                        if x["job"]["mode"] == "search"
                        and x["job"]["head"] == head
                        and x["job"]["config"] == config
                        and x["job"]["seed"] == 0
                        and x["state"] != "empty_held_fold"
                    ]
                    row = min(matches, key=lambda x: x["job"]["fold"])
                    raw = view.store.json(row["attempt"] + "/training.json")
                    r.require(
                        raw["total_seconds"] == row["total_seconds"]
                        and raw["train_seconds"] == row["train_seconds"],
                        "Sampled search timer differs.",
                    )
                    sampled.append(row["job"]["id"])
            inventories[pool] = dict(
                categories=totals,
                files=len(view.store.entries),
                logical_bytes=sum(x["logical_bytes"] for x in totals.values()),
                weights=weights,
                model_ancillary=extras,
                zip_payload_bytes=sum(x["bytes"] for x in view.store.parts.values()),
                direct_payload_bytes=sum(
                    spec["bytes"]
                    for part, spec in view.store.entries.values()
                    if part is None
                ),
                zip_parts=len(view.store.parts),
                sampled_search_timers=sampled,
            )
            bindings[pool] = view.store.read_bindings
        print(
            pool
            + ": timings, final assignments, inference receipts and storage inventory extracted.",
            flush=True,
        )
    r.require(
        len(alljobs) == 23210
        and len(historical) == 16
        and len(assignments) == 135
        and len(inferences) == 530,
        "Work roster differs.",
    )
    b.save(OUT / "jobs.json", alljobs)
    b.save(OUT / "historical_attempts.json", historical)
    b.save(OUT / "pipelines.json", assignments)
    b.save(OUT / "inferences.json", inferences)
    timeline = []
    phases = []
    devices = []
    workerassignments = []
    rawrecords = {}
    with study.store("study_records") as store:
        for name in sorted(store.entries):
            leaf = Path(name).name
            if leaf in ("started.json", "finished.json"):
                rawrecords[name] = store.json(name)
            if leaf.endswith(".device.json"):
                devices.append(dict(path=name, record=store.json(name)))
            if (
                name.startswith("invocations/")
                and leaf.startswith("worker_")
                and leaf.endswith(".json")
                and leaf.count(".") == 1
            ):
                x = store.json(name)
                done = name[:-5] + ".complete.json"
                record = dict(
                    path=name,
                    pool=x["pool"],
                    phase=x["phase"],
                    worker=x["worker"],
                    session_sha256=r.signature(x["session"]),
                    jobs=[z["id"] for z in x["jobs"]],
                    complete=done in store.entries,
                )
                if record["complete"]:
                    v = store.json(done)
                    r.require(
                        v
                        == dict(
                            assignment_sha256=store.expected(name), jobs=len(x["jobs"])
                        ),
                        "Worker completion differs.",
                    )
                workerassignments.append(record)
            if leaf == "telemetry.jsonl":
                raw = store.raw(name)
                rows = [r.parsed(x) for x in raw.splitlines()]
                phase = Path(name).parent.name.rsplit("_", 1)[-1]
                pool = Path(name).parent.name.rsplit("_", 1)[0]
                inv = Path(name).parts[1]
                times = [utc(x["utc"]) for x in rows]
                r.require(
                    all(b >= a for a, b in zip(times, times[1:])),
                    "Telemetry time went backwards.",
                )
                columns = [
                    "utc_seconds",
                    "gpu_busy_pct",
                    "gpu_used_mib",
                    "gpu_total_mib",
                    "temperature_c",
                    "ram_current",
                    "ram_limit",
                    "ram_available",
                    "workers",
                    "storage_inputs",
                    "storage_outputs",
                    "storage_reserve",
                    "storage_free",
                    "project_limit",
                ]
                values = []
                for x, t in zip(rows, times):
                    v = x.get("storage", {})
                    values.append(
                        [
                            t,
                            x["gpu"]["busy"],
                            x["gpu"]["used_mib"],
                            x["gpu"]["total_mib"],
                            x["gpu"]["temperature"],
                            x["memory"]["current"],
                            x["memory"]["limit"],
                            x["memory"]["available"],
                            x["workers"],
                        ]
                        + [
                            v.get(k, -1)
                            for k in (
                                "inputs",
                                "outputs",
                                "reserve",
                                "free",
                                "project_limit",
                            )
                        ]
                    )
                dest = f"telemetry_{len(phases):02d}.npz"
                m.array(OUT / dest, values=np.asarray(values, dtype=np.float64))
                phases.append(
                    dict(
                        path=name,
                        invocation=inv,
                        pool=pool,
                        phase=phase,
                        samples=len(rows),
                        columns=columns,
                        array=dest,
                        first_utc=rows[0]["utc"] if rows else None,
                        last_utc=rows[-1]["utc"] if rows else None,
                        gpu_ids=sorted({x["gpu"]["uuid"] for x in rows}),
                        gpu_names=sorted({x["gpu"]["name"] for x in rows}),
                        drivers=sorted({x["gpu"]["driver"] for x in rows}),
                    )
                )
        for name, start in rawrecords.items():
            if not name.endswith("/started.json"):
                continue
            endname = name.replace("/started.json", "/finished.json")
            r.require(endname in rawrecords, "Unclosed invocation.")
            end = rawrecords[endname]
            r.require(
                end["started_sha256"] == store.expected(name) and end["seconds"] >= 0,
                "Invocation duration binding differs.",
            )
            timeline.append(
                dict(
                    id=Path(name).parts[1],
                    start=start,
                    end=end,
                    wall_timestamp_seconds=utc(end["utc"]) - utc(start["utc"]),
                    start_sha256=store.expected(name),
                    end_sha256=store.expected(endname),
                    session_sha256=r.signature(start["session"]),
                )
            )
        totals = defaultdict(lambda: dict(files=0, logical_bytes=0))
        duplicate_bytes = 0
        duplicates = []
        for name, (part, spec) in store.entries.items():
            category = name.split("/")[0]
            totals[category]["files"] += 1
            totals[category]["logical_bytes"] += spec["bytes"]
            if name in small_duplicates:
                r.require(
                    spec == small_duplicates[name],
                    "Cross-archive duplicated metadata differs.",
                )
                duplicate_bytes += spec["bytes"]
                duplicates.append(name)
        inventories["study_records"] = dict(
            categories=totals,
            files=len(store.entries),
            logical_bytes=sum(x["logical_bytes"] for x in totals.values()),
            duplicate_metadata_bytes=duplicate_bytes,
            duplicated_files=duplicates,
            zip_payload_bytes=sum(x["bytes"] for x in store.parts.values()),
            direct_payload_bytes=0,
            zip_parts=len(store.parts),
        )
        bindings["study_records"] = store.read_bindings
    timeline.sort(key=lambda x: utc(x["start"]["utc"]))
    r.require(
        len(timeline) == 7 and len(phases) == 21, "Archived timeline coverage differs."
    )
    b.save(OUT / "timeline.json", timeline)
    b.save(OUT / "phases.json", phases)
    b.save(OUT / "devices.json", devices)
    b.save(OUT / "worker_assignments.json", workerassignments)
    b.save(OUT / "inventories.json", inventories)
    b.save(
        OUT / "source_bindings.json",
        dict(local={**sources, **study.bindings}, archives=bindings),
    )
    b.save(
        OUT / "complete.json",
        dict(
            status="complete",
            jobs=len(alljobs),
            pipelines=len(assignments),
            inference_models=len(inferences),
            telemetry_samples=sum(x["samples"] for x in phases),
            code_sha256=r.sha(Path(__file__)),
            outputs={p.name: r.sha(p) for p in OUT.iterdir() if p.is_file()},
            weights_read=0,
            network_calls=0,
            training_calls=0,
        ),
    )
    print("Resource extraction complete; no model or network call.")


if __name__ == "__main__":
    with r.offline():
        extract()

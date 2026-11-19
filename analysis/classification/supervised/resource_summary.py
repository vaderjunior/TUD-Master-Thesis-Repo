from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import math
import numpy as np
from analysis.classification.supervised import resources as b
from analysis.classification.supervised import archive as r
from analysis.classification.supervised import metrics as m

EX = b.BASE / "extraction"
OUT = b.BASE / "analysis"
MAX_GAP = 10.0


def stamp(value):
    v = datetime.fromisoformat(value)
    r.require(v.tzinfo is not None, "Timestamp requires timezone.")
    return v.timestamp()


def summarize(rows):
    timed = [x for x in rows if "total_seconds" in x]
    r.require(
        all(x["total_seconds"] >= x["train_seconds"] >= 0 for x in timed),
        "Invalid timer boundary.",
    )
    return dict(
        slots=len(rows),
        timed_fits=len(timed),
        untimed_slots=len(rows) - len(timed),
        states=dict(Counter(x["state"] for x in rows)),
        total_worker_seconds=math.fsum(x["total_seconds"] for x in timed),
        loop_worker_seconds=math.fsum(x["train_seconds"] for x in timed),
        outside_loop_worker_seconds=math.fsum(
            x["total_seconds"] - x["train_seconds"] for x in timed
        ),
        **{
            key: sum(x[key] for x in rows)
            for key in (
                "planned_steps",
                "successful_steps",
                "skipped_steps",
                "positive_updates",
                "accepted_zero_rate_steps",
            )
        }
    )


def groups(rows, key):
    out = defaultdict(list)
    for row in rows:
        out[str(key(row))].append(row)
    return {k: summarize(v) for k, v in sorted(out.items())}


def sequence(intervals):
    intervals = sorted(intervals)
    r.require(all(z >= a for a, z in intervals), "Negative interval.")
    r.require(
        all(a >= z for (_, z), (a, _) in zip(intervals, intervals[1:])),
        "Overlapping controller/phase intervals.",
    )
    active = math.fsum(z - a for a, z in intervals)
    calendar = intervals[-1][1] - intervals[0][0]
    gaps = [a - z for (_, z), (a, _) in zip(intervals, intervals[1:])]
    return dict(
        active_seconds=active,
        calendar_seconds=calendar,
        gap_seconds=math.fsum(gaps),
        gaps=gaps,
    )


def telemetry_stats(arrays):
    v = np.concatenate(arrays)
    weights = []
    samples = []
    allgaps = []
    windows = 0.0
    for x in arrays:
        dt = np.diff(x[:, 0])
        good = (dt > 0) & (dt <= MAX_GAP)
        weights.append(dt[good])
        samples.append(x[:-1][good])
        allgaps.extend(dt.tolist())
        windows += float(dt.sum())
    w = np.concatenate(weights)
    observed = np.concatenate(samples)
    covered = float(w.sum())
    g = np.asarray(allgaps)
    r.require(covered > 0 and len(g) > 0, "Missing telemetry coverage.")
    storage = v[v[:, 9] >= 0]
    result = dict(
        samples=len(v),
        window_seconds=windows,
        weighted_covered_seconds=covered,
        coverage_fraction=covered / windows,
        ignored_long_gap_seconds=float(g[g > MAX_GAP].sum()),
        maximum_gap_seconds=float(g.max()),
        median_gap_seconds=float(np.median(g)),
        weighted_gpu_busy_pct=float(np.average(observed[:, 1], weights=w)),
        unweighted_gpu_busy_pct=float(v[:, 1].mean()),
        gpu_busy_percentiles=dict(
            zip(("p05", "p50", "p95"), np.percentile(v[:, 1], [5, 50, 95]).tolist())
        ),
        max_gpu_mib=float(v[:, 2].max()),
        gpu_total_mib=sorted(set(v[:, 3].tolist())),
        max_temperature_c=float(v[:, 4].max()),
        max_ram_bytes=int(v[:, 5].max()),
        ram_limits=sorted(set(v[:, 6].astype(int).tolist())),
        minimum_ram_available_bytes=int(v[:, 7].min()),
        max_workers=int(v[:, 8].max()),
        ram_over_80pct_samples=int((v[:, 5] > 0.8 * v[:, 6]).sum()),
        gpu_memory_over_80pct_samples=int((v[:, 2] > 0.8 * v[:, 3]).sum()),
        storage_samples=len(storage),
    )
    if len(storage):
        result.update(
            max_inputs_bytes=int(storage[:, 9].max()),
            max_outputs_bytes=int(storage[:, 10].max()),
            max_inputs_outputs_bytes=int((storage[:, 9] + storage[:, 10]).max()),
            project_limits=sorted(set(storage[:, 13].astype(int).tolist())),
        )
    return result


def main():
    for name, digest in r.read(EX / "complete.json")["outputs"].items():
        r.checked_file(EX / name, digest)
    r.checked_file(
        r.ROOT / "analysis/classification/supervised/extract_resources.py",
        r.read(EX / "complete.json")["code_sha256"],
    )
    jobs = r.read(EX / "jobs.json")
    config = {x["id"]: x for x in r.read(r.PACKAGE / "data/configs.json")}
    source = {
        str(r.PACKAGE / "data/configs.json"): r.sha(r.PACKAGE / "data/configs.json")
    }
    search = [x for x in jobs if x["job"]["mode"] == "search"]
    final = [x for x in jobs if x["job"]["mode"] == "final"]
    r.require(
        len({x["job"]["id"] for x in jobs}) == len(jobs), "Duplicate incurred fit."
    )
    accounting = {}
    for phase, rows in [("search", search), ("final", final)]:
        accounting[phase] = dict(
            total=summarize(rows),
            by_budget=groups(rows, lambda x: m.budget(x["job"]["pool"])),
            by_pool=groups(rows, lambda x: x["job"]["pool"]),
            by_head=groups(rows, lambda x: x["job"]["head"]),
            by_rate=groups(rows, lambda x: config[x["job"]["config"]]["learning_rate"]),
            by_epochs=groups(rows, lambda x: config[x["job"]["config"]]["epochs"]),
            by_loss=groups(rows, lambda x: config[x["job"]["config"]]["loss"]),
        )
    subsets = {
        name: summarize([x for x in search if predicate(config[x["job"]["config"]])])
        for name, predicate in {
            "one_rate_all_durations_losses": lambda c: c["learning_rate"] == 2e-5,
            "both_fixed20_recipes": lambda c: c["learning_rate"] == 2e-5
            and c["epochs"] == 20,
            "fixed_balanced20_recipe": lambda c: c["learning_rate"] == 2e-5
            and c["epochs"] == 20
            and c["loss"] == "balanced",
            "extra_two_rates": lambda c: c["learning_rate"] != 2e-5,
            "sixty_epochs": lambda c: c["epochs"] == 60,
        }.items()
    }

    pipe = r.read(EX / "pipelines.json")
    pipe_summary = []
    assignments = Counter()
    for row in pipe:
        for head in row["heads"].values():
            assignments[head["job"]] += 1
    for budget in (147, 1000, 5000):
        for family in r.FAMILIES:
            rows = [x for x in pipe if x["budget"] == budget and x["family"] == family]
            times = np.array(
                [sum(h["total_seconds"] for h in x["heads"].values()) for x in rows]
            )
            loops = np.array(
                [sum(h["train_seconds"] for h in x["heads"].values()) for x in rows]
            )
            sizes = np.array(
                [sum(h["model_files_bytes"] for h in x["heads"].values()) for x in rows]
            )
            pipe_summary.append(
                dict(
                    budget=budget,
                    family=family,
                    pipelines=len(rows),
                    mean_fit_seconds=float(times.mean()),
                    min_fit_seconds=float(times.min()),
                    max_fit_seconds=float(times.max()),
                    mean_loop_seconds=float(loops.mean()),
                    mean_model_bytes=float(sizes.mean()),
                    min_model_bytes=int(sizes.min()),
                    max_model_bytes=int(sizes.max()),
                )
            )
    r.require(
        len(assignments) == len(final) == 530 and sum(assignments.values()) == 540,
        "Final assignment reuse differs.",
    )
    reused = [
        dict(
            job=k,
            assignments=v,
            total_seconds=next(
                x["total_seconds"] for x in final if x["job"]["id"] == k
            ),
        )
        for k, v in assignments.items()
        if v > 1
    ]
    history = r.read(EX / "historical_attempts.json")
    for x in history:
        x["attempt_envelope_seconds"] = stamp(x["stop"]["utc"]) - stamp(
            x["started_utc"]
        )
    timeline = r.read(EX / "timeline.json")
    intervals = [(stamp(x["start"]["utc"]), stamp(x["end"]["utc"])) for x in timeline]
    timing = sequence(intervals)
    timing.update(
        start=timeline[0]["start"]["utc"],
        end=timeline[-1]["end"]["utc"],
        monotonic_controller_seconds=math.fsum(x["end"]["seconds"] for x in timeline),
        timestamp_vs_monotonic_difference_seconds=math.fsum(
            x["wall_timestamp_seconds"] - x["end"]["seconds"] for x in timeline
        ),
    )
    workers = r.read(EX / "worker_assignments.json")
    index = defaultdict(list)
    for w in workers:
        for job in w["jobs"]:
            index[(w["session_sha256"], job)].append(w)
    assigned = Counter()
    incomplete_assigned = Counter()
    invocations = {x["session_sha256"]: x for x in timeline}
    for x in jobs:
        matches = index[(x["session_sha256"], x["job"]["id"])]
        r.require(
            len(matches) == 1,
            "Canonical fit not uniquely assigned to recorded worker/session.",
        )
        w = matches[0]
        assigned[x["job"]["mode"]] += 1
        if not w["complete"]:
            incomplete_assigned[x["job"]["mode"]] += 1
        inv = invocations[x["session_sha256"]]
        r.require(
            stamp(inv["start"]["utc"])
            <= stamp(x["started_utc"])
            <= stamp(inv["end"]["utc"]),
            "Canonical fit started outside invocation.",
        )
    phase_rows = r.read(EX / "phases.json")
    phasegroups = defaultdict(list)
    perphase = []
    for row in phase_rows:
        with np.load(EX / row["array"], allow_pickle=False) as z:
            v = z["values"]
        phasegroups[row["phase"]].append(v)
        stat = telemetry_stats([v])
        perphase.append({**row, "statistics": stat})
    sampled_windows = sequence(
        [(stamp(x["first_utc"]), stamp(x["last_utc"])) for x in phase_rows]
    )
    tele = {name: telemetry_stats(arrays) for name, arrays in phasegroups.items()}
    tele["all"] = telemetry_stats(
        [x for arrays in phasegroups.values() for x in arrays]
    )
    timing["telemetry_window_seconds"] = sampled_windows["active_seconds"]
    timing["controller_outside_sample_windows_seconds"] = (
        timing["active_seconds"] - sampled_windows["active_seconds"]
    )
    inventory = r.read(EX / "inventories.json")
    category = defaultdict(lambda: dict(files=0, logical_bytes=0))
    poolbytes = []
    for pool, v in inventory.items():
        for k, x in v["categories"].items():
            category[k]["files"] += x["files"]
            category[k]["logical_bytes"] += x["logical_bytes"]
        poolbytes.append(
            dict(
                pool=pool,
                logical_bytes=v["logical_bytes"],
                zip_payload_bytes=v["zip_payload_bytes"],
                direct_payload_bytes=v["direct_payload_bytes"],
                payload_bytes=v["zip_payload_bytes"] + v["direct_payload_bytes"],
                zip_parts=v["zip_parts"],
                files=v["files"],
            )
        )
    rawlogical = sum(x["logical_bytes"] for x in poolbytes)
    duplicates = inventory["study_records"]["duplicate_metadata_bytes"]
    storage = dict(
        categories=category,
        pools=poolbytes,
        manifest_entry_bytes=rawlogical,
        unique_logical_bytes=rawlogical - duplicates,
        duplicate_metadata_bytes=duplicates,
        total_payload_bytes=sum(x["payload_bytes"] for x in poolbytes),
        zip_parts=sum(x["zip_parts"] for x in poolbytes),
        files=sum(x["files"] for x in poolbytes),
        direct_weight_files=sum(
            sum(w["zip_part"] is None for w in v.get("weights", []))
            for v in inventory.values()
        ),
        weight_bytes=category["weights"]["logical_bytes"],
        nonweight_unique_logical_bytes=rawlogical
        - duplicates
        - category["weights"]["logical_bytes"],
        weight_share_of_unique_logical=category["weights"]["logical_bytes"]
        / (rawlogical - duplicates),
    )
    inference = r.read(EX / "inferences.json")
    r.require(
        not any(x["timer_recorded"] for x in inference),
        "Unexpected recorded inference timer; review boundary.",
    )
    OUT.mkdir(exist_ok=False)
    b.save(
        OUT / "fit_accounting.json",
        dict(
            phases=accounting,
            executed_search_subsets=subsets,
            pipeline_summary=pipe_summary,
            reused_assignments=reused,
            inference_model_passes=len(inference),
            inference_head_rows=sum(x["rows"] for x in inference),
            standalone_inference_seconds=None,
            worker_assignment_checks=dict(
                total=len(jobs),
                assigned=dict(assigned),
                canonical_fits_from_incomplete_worker_files=dict(incomplete_assigned),
            ),
            historical_attempts=history,
            historical_attempt_envelope_seconds=math.fsum(
                x["attempt_envelope_seconds"] for x in history
            ),
        ),
    )
    b.save(
        OUT / "timeline_summary.json",
        dict(
            summary=timing,
            invocations=timeline,
            phases=perphase,
            telemetry=tele,
            weighting=dict(
                maximum_gap_seconds=MAX_GAP,
                method="Left sample weighted by next interval, only positive intervals <=10 seconds; no interpolation over longer gaps.",
                statistics="Descriptive; observed maxima are lower bounds on true unsampled peaks.",
            ),
        ),
    )
    b.save(OUT / "storage_summary.json", storage)
    b.save(OUT / "source_bindings.json", source)
    b.save(
        OUT / "complete.json",
        dict(
            status="complete",
            code_sha256=r.sha(Path(__file__)),
            extraction_sha256=r.sha(EX / "complete.json"),
            outputs={p.name: r.sha(p) for p in OUT.iterdir() if p.is_file()},
        ),
    )
    print("Fit accounting:", {k: v["total"] for k, v in accounting.items()})
    print("Timeline:", timing)
    print("Telemetry:", tele)
    print("Pipelines:", pipe_summary)
    print(
        "Storage:",
        {k: v for k, v in storage.items() if k not in ("pools", "categories")},
    )


if __name__ == "__main__":
    with r.offline():
        main()

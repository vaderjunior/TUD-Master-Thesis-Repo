from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import sys
import numpy as np
from analysis.feedback.reserved import transitions as prior
from analysis.feedback.reserved import robustness_metrics as core

io, need, rates, support, reader = (
    prior.io,
    prior.need,
    prior.rates,
    prior.support,
    prior.reader,
)
P5, P4 = prior.prior, prior.P4
P3, P2 = P4.prior, P4.prior.prior
ROOT = io.ROOT
OUT = ROOT / "experiments/astra/sq3_reserved_part7_results_20261020_v1"
PREP = ROOT / "experiments/astra/sq3_reserved_part7_preparation_20261020_v1"
COHORTS = ("full", "same_six_excluded")
MASKS = P4.MASKS


def write(path, value):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside Part 7 outputs")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(
            value
            if isinstance(value, str)
            else json.dumps(
                value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        )


def save_draws(name, draws):
    path = OUT / "draws" / (name.replace("/", "__") + ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.save(f, draws, allow_pickle=False)
    return path


def check_number(x, y, message):
    need(y is None if x is None else y is not None and abs(x - y) < 1e-12, message)


def verify_previous():
    paths = []
    for folder, name in (
        (P2.OUT, "complete.json"),
        (P3.OUT, "complete.json"),
        (P4.OUT, "complete.json"),
        (P5.OUT, "complete_v2.json"),
        (prior.OUT, "complete.json"),
    ):
        path = folder / name
        d = io.read(path)
        for field in ("artifacts", "bindings", "report_binding"):
            if field in d:
                io.verify(ROOT, d[field])
        paths.append(path)
    for part in (P2, P3, P4, P5, prior):
        name = (
            "interpretation_check_v2.json"
            if part == P5
            else "interpretation_check.json"
        )
        paths.append(part.OUT / name)
    need(
        io.sha(reader.OUT / "complete.json")
        == "f6bbca18ca59dcfcf7a639399acd8367bcdbcc27cdb424143afe0254abaf7d03",
        "Audited-receipt assembly changed",
    )
    return paths


def specifications(science):
    specs = (
        rates.contrasts(science)
        + support.supporting_specs(science)
        + P4.direct_specs(science)
    )
    need(
        len(specs) == 130 and len({s["id"] for s in specs}) == 130,
        "Incomplete robustness scope",
    )
    return specs


def inputs():
    results = {}
    checks = {}
    origins = {}
    omissions = {}
    sources = {}
    source_origins = {}
    for module, filename in (
        (P2, "primary_and_sensitivities.json"),
        (P3, "mechanism_and_sensitivities.json"),
    ):
        for key, value in io.read(module.OUT / filename).items():
            results[key] = value
            origins[key] = {"path": io.rel(module.OUT / filename), "key": key}
        checks.update(io.read(module.OUT / "independent_checks.json"))
        omissions.update(io.read(module.OUT / "omission_points.json"))
        for key, value in io.read(module.OUT / "source_effects.json").items():
            sources[key] = value
            source_origins[key] = module.OUT
    ordinary = io.read(P4.OUT / "rate_comparisons.json")
    for key, value in ordinary.items():
        cohort, spec = key.split("/", 1)
        new = cohort + "/ordinary_valid/" + spec
        results[new] = value
        origins[new] = {"path": io.rel(P4.OUT / "rate_comparisons.json"), "key": key}
    for key, value in io.read(P4.OUT / "point_sensitivities.json").items():
        results[key] = value
        origins[key] = {"path": io.rel(P4.OUT / "point_sensitivities.json"), "key": key}
    checks4 = io.read(P4.OUT / "independent_checks.json")
    for key, value in io.read(P4.OUT / "omission_points.json").items():
        cohort, spec = key.split("/", 1)
        omissions[cohort + "/ordinary_valid/" + spec] = value
    need(
        len(results) == 780 and len(sources) == 64 and len(omissions) == 292,
        "Existing sensitivity inventory differs",
    )
    return results, checks, checks4, origins, omissions, sources, source_origins


def rates_and_omissions(roster, predictions, science, specs, batches, existing):
    results, checks, checks4, origins, omissions, _, _ = existing
    exact2 = io.read(P2.OUT / "integer_draw_verification.json")["results"]
    profiles = {
        s["id"]: {"specification": s, "views": {}, "ordinary_details": {}}
        for s in specs
    }
    checked_omissions = {}
    n_checked = 0
    for cohort in COHORTS:
        excluded = [] if cohort == "full" else science["mandatory_excluded_keys"]
        for s in specs:
            for mask, strict, reps in MASKS:
                key = cohort + "/" + mask + "/" + s["id"]
                result = results[key]
                direct = core.rate_point(
                    roster,
                    predictions,
                    s,
                    strict=strict,
                    representatives=reps,
                    excluded=excluded,
                )
                need(
                    direct["common_ids"] == result["common_ids"]
                    and direct["eligible_n"] == result["original_n"],
                    "Prior mask differs: " + key,
                )
                check_number(
                    direct["difference"],
                    result.get("difference"),
                    "Prior rate differs: " + key,
                )
                need(
                    all(
                        abs(x - y) < 1e-12
                        for x, y in zip(
                            direct["missing_difference_bounds"],
                            result["missing_difference_bounds"],
                        )
                    ),
                    "Prior missing bounds differ: " + key,
                )
                interval = None
                if "family_interval" in result:
                    interval = (
                        exact2[key]["interval_from_exact_integer_errors"]
                        if s["family"] == "primary"
                        else (
                            checks[key]["integer_bootstrap"]["exact_interval"]
                            if s["family"] == "mechanism"
                            else checks4[cohort + "/" + s["id"]]["integer_bootstrap"][
                                "exact_interval"
                            ]
                        )
                    )
                    need(
                        all(
                            abs(x - y) < 1e-12
                            for x, y in zip(interval, result["family_interval"])
                        ),
                        "Interval normalization differs: " + key,
                    )
                profiles[s["id"]]["views"][cohort + "/" + mask] = {
                    **{k: v for k, v in direct.items() if k != "common_ids"},
                    "common_ids_sha256": hashlib.sha256(
                        json.dumps(direct["common_ids"], separators=(",", ":")).encode()
                    ).hexdigest(),
                    "point_direction": core.sign(direct["difference_exact"]),
                    "exact_interval": interval,
                    "raw_interval": result.get("family_interval"),
                    "confidence": result.get("family_confidence"),
                    "interval_state": core.interval_state(interval),
                    "missing_bound_state": core.interval_state(
                        direct["missing_bounds_exact"]
                    ),
                    "origin": origins[key],
                }
                n_checked += 1
                if key not in omissions:
                    continue
                verified = {}
                for kind, entries in omissions[key].items():
                    checked = []
                    for entry in entries:
                        if kind == "pool":
                            chosen = core.omit_pool(s, entry["omitted_pool_seed"])
                            need(chosen is not None, "Unexpected pool omission")
                            remove = excluded
                        else:
                            chosen = s
                            remove = sorted(
                                set(excluded)
                                | {
                                    k
                                    for k, b in batches.items()
                                    if b == entry["omitted_batch"]
                                }
                            )
                        point = core.rate_point(
                            roster,
                            predictions,
                            chosen,
                            strict=strict,
                            representatives=reps,
                            excluded=remove,
                        )
                        need(
                            point["common_n"] == entry["common_n"],
                            "Omission mask count differs",
                        )
                        check_number(
                            point["difference"],
                            entry["difference"],
                            "Omission point differs",
                        )
                        checked.append(
                            {**entry, "difference_exact": point["difference_exact"]}
                        )
                    verified[kind] = {
                        "points": checked,
                        "spread": core.spread(v["difference"] for v in checked),
                    }
                checked_omissions[key] = verified
            base = results[cohort + "/ordinary_valid/" + s["id"]]
            detail = P4.comparison_details(roster, predictions, s, base)
            profiles[s["id"]]["ordinary_details"][cohort] = detail
        print("Rate masks and omissions checked: " + cohort, flush=True)
    need(n_checked == 780, "Incomplete rate mask checks")
    write(OUT / "rate_profiles.json", profiles)
    write(OUT / "omission_points_checked.json", checked_omissions)
    return profiles, {
        "rate_views": n_checked,
        "omission_packages": len(checked_omissions),
        "omission_points": sum(
            len(x["points"]) for v in checked_omissions.values() for x in v.values()
        ),
    }


def source_checks(roster, predictions, science, specs, existing):
    old, locations = existing[-2:]
    source_names = sorted({r["source"] for r in roster})
    results, checks = {}, {}
    new_objects = 0
    for cohort in COHORTS:
        excluded = [] if cohort == "full" else science["mandatory_excluded_keys"]
        eligible = core.selected(roster, excluded)
        for index, spec in enumerate(specs, 1):
            base = cohort + "/" + spec["id"]
            for source in [None, *source_names]:
                stratified = source is None
                key = base + ("/stratified" if stratified else "/source/" + source)
                ss = {
                    **spec,
                    "confidence": 0.95,
                    "seed": support.seed_for(
                        ("source_stratified/" + base)
                        if stratified
                        else "source_specific/" + base + "/" + source
                    ),
                }
                rows = (
                    roster
                    if stratified
                    else [r for r in eligible if r["source"] == source]
                )
                c = rates.cohort(
                    rows, predictions, ss, excluded=excluded if stratified else []
                )
                if key in old:
                    value = old[key]
                    path = locations[key] / "draws" / (key.replace("/", "__") + ".npy")
                    draws = np.load(path, allow_pickle=False)
                    origin = "reused_existing_primary_or_mechanism_source_check"
                else:
                    value, draws = (
                        support.source_bootstrap(c, rows, ss)
                        if stratified
                        else rates.summarize(c, ss)
                    )
                    path = save_draws(key, draws)
                    origin = "additional_part7_supporting_source_check"
                    new_objects += 1
                exact = P3.integer_draw_check(
                    rows, predictions, ss, c, draws, stratified=stratified
                )
                if c["common_ids"]:
                    interval = (
                        value["interval95"] if stratified else value["family_interval"]
                    )
                    need(
                        all(
                            abs(x - y) < 1e-12
                            for x, y in zip(interval, exact["exact_interval"])
                        ),
                        "Source interval differs: " + key,
                    )
                    direct = core.rate_point(
                        rows, predictions, ss, excluded=excluded if stratified else []
                    )
                    check_number(
                        direct["difference"],
                        value["difference"],
                        "Source point differs",
                    )
                    if not stratified:
                        need(
                            direct["common_ids"] == value["common_ids"],
                            "Source common mask differs",
                        )
                else:
                    direct = core.rate_point(
                        rows, predictions, ss, excluded=excluded if stratified else []
                    )
                    need(len(draws) == 0, "Empty source has draws")
                results[key] = {
                    "result": value,
                    "common_n": len(c["common_ids"]),
                    "eligible_n": len(c["original_ids"]),
                    "difference_exact": direct["difference_exact"],
                    "exact_interval": exact["exact_interval"],
                    "interval_state": core.interval_state(exact["exact_interval"]),
                    "point_direction": core.sign(direct["difference_exact"]),
                    "confidence": 0.95,
                    "origin": origin,
                    "draw_path": io.rel(path),
                    "draw_sha256": io.sha(path),
                }
                checks[key] = exact
            if index % 10 == 0 or index == len(specs):
                print(f"Source checks: {cohort} {index}/{len(specs)}", flush=True)
    need(len(results) == 1040 and new_objects == 976, "Incomplete source scope")
    write(OUT / "source_effects.json", results)
    write(OUT / "source_integer_checks.json", checks)
    return {
        "source_objects": len(results),
        "new_source_objects": new_objects,
        "reused_source_objects": len(old),
        "new_estimable_distributions": sum(
            v["common_n"] > 0 and v["origin"].startswith("additional")
            for v in results.values()
        ),
        "draw_values_checked": sum(c["draws_checked"] for c in checks.values()),
        "max_absolute_error": max(c["max_absolute_error"] for c in checks.values()),
    }


def quality_checks(roster, predictions, science, batches):
    specs = P5.metric_specs(science)
    prior_results = io.read(P5.OUT / "paired_quality.json")
    sensitivities = io.read(P5.OUT / "point_sensitivities.json")
    bounds = io.read(P5.OUT / "missing_output_bounds.json")
    profiles, diagnostics = {}, {}
    npoints = 0
    for index, entry in enumerate(specs, 1):
        name, spec, fine = entry["name"], entry["spec"], entry["fine"]
        rows = core.selected(roster, entry["excluded"])
        original = prior_results[name]
        full = core.quality_point(rows, predictions, spec, fine)
        need(full["common_ids"] == original["common_ids"], "Prior quality mask differs")
        for metric, value in full["metrics"].items():
            for field in ("treatment", "control", "difference"):
                check_number(
                    value[field],
                    original["metrics"][metric][field],
                    "Prior quality point differs",
                )
        profiles[name] = {
            "specification": entry,
            "ordinary": original,
            "ordinary_point_check": full,
            "point_sensitivities": {},
            "missing_bounds": {
                mask: bounds[name + "/" + mask]
                for mask in ("ordinary_valid", "all_three_votes")
            },
        }
        for mask, _, _ in MASKS[1:]:
            profiles[name]["point_sensitivities"][mask] = sensitivities[
                name + "/" + mask
            ]
        diag = {"sources": {}, "pool_omissions": [], "batch_omissions": []}
        for source in sorted({r["source"] for r in roster}):
            diag["sources"][source] = core.quality_point(
                [r for r in rows if r["source"] == source], predictions, spec, fine
            )
            npoints += 1
        for seed in science["pool_seeds"]:
            omitted = core.omit_pool(spec, seed)
            if omitted is not None:
                diag["pool_omissions"].append(
                    {
                        "omitted_seed": seed,
                        "result": core.quality_point(rows, predictions, omitted, fine),
                    }
                )
                npoints += 1
        for batch in sorted(set(batches.values())):
            subset = [r for r in rows if batches[r["key"]] != batch]
            diag["batch_omissions"].append(
                {
                    "omitted_batch": batch,
                    "result": core.quality_point(subset, predictions, spec, fine),
                }
            )
            npoints += 1
        diag["summary"] = {
            metric: {
                "source_points": {
                    s: v["metrics"][metric]["difference"]
                    for s, v in diag["sources"].items()
                },
                "pool_omission": core.spread(
                    v["result"]["metrics"][metric]["difference"]
                    for v in diag["pool_omissions"]
                ),
                "batch_omission": core.spread(
                    v["result"]["metrics"][metric]["difference"]
                    for v in diag["batch_omissions"]
                ),
            }
            for metric in full["metrics"]
        }
        diagnostics[name] = diag
        if index % 8 == 0 or index == len(specs):
            print(
                f"Quality robustness points checked: {index}/{len(specs)}", flush=True
            )
    need(
        len(profiles) == 112 and npoints == 2452, "Incomplete quality diagnostic scope"
    )
    write(OUT / "quality_profiles.json", profiles)
    write(OUT / "quality_omissions_and_sources.json", diagnostics)
    return {
        "original_quality_objects": len(profiles),
        "new_point_objects": npoints,
        "original_scalar_intervals": 224,
        "reused_scalar_point_sensitivities": 448,
        "new_scalar_intervals": 0,
    }


def validity(roster, predictions, science, batches):
    groups = defaultdict(list)
    for r in roster:
        groups[r["cluster"]].append(r)
    votes = Counter()
    nulls, reduced = [], []
    bycondition = defaultdict(Counter)
    for (key, condition, repeat), p in predictions.items():
        votes[p["n_valid"]] += 1
        bycondition[condition][p["n_valid"]] += 1
        record = {
            "key": key,
            "condition": condition,
            "repeat": repeat,
            "stage": p["stage"],
            "n_valid": p["n_valid"],
        }
        if p["result"] is None:
            nulls.append(record)
        elif p["n_valid"] < 3:
            reduced.append(record)
    masks = {}
    for cohort in COHORTS:
        excluded = [] if cohort == "full" else science["mandatory_excluded_keys"]
        for mask, strict, reps in MASKS:
            rows = core.selected(roster, excluded, reps)
            common = [
                r
                for r in rows
                if all(
                    p["result"] is not None and (not strict or p["n_valid"] == 3)
                    for c in science["conditions"]
                    for rep in range(3)
                    for p in [predictions[r["key"], c, rep]]
                )
            ]
            masks[cohort + "/" + mask] = {
                "eligible_by_polarity": dict(
                    Counter(str(r["gold"]["hate"]) for r in rows)
                ),
                "common_by_polarity": dict(
                    Counter(str(r["gold"]["hate"]) for r in common)
                ),
                "common_ids": [r["key"] for r in common],
                "clusters": len({r["cluster"] for r in common}),
            }
    duplicates = [
        {
            "cluster": label,
            "members": rows,
            "batches": sorted({batches[r["key"]] for r in rows}),
            "stable_representatives": [
                r["key"] for r in core.selected(rows, representatives=True)
            ],
        }
        for label, rows in groups.items()
        if len(rows) > 1
    ]
    data = {
        "predictions": len(predictions),
        "votes": {str(k): v for k, v in votes.items()},
        "null_count": len(nulls),
        "reduced_vote_valid_count": len(reduced),
        "null_by_text": dict(Counter(r["key"] for r in nulls)),
        "reduced_by_text": dict(Counter(r["key"] for r in reduced)),
        "by_condition": {
            c: {str(k): v for k, v in d.items()} for c, d in bycondition.items()
        },
        "source_polarity_counts": dict(
            Counter(r["source"] + "/" + str(r["gold"]["hate"]) for r in roster)
        ),
        "masks": masks,
        "duplicate_clusters": duplicates,
        "exclusions": [
            r for r in roster if r["key"] in science["mandatory_excluded_keys"]
        ],
        "batch_source_polarity": {
            b: dict(
                Counter(
                    r["source"] + "/" + str(r["gold"]["hate"])
                    for r in roster
                    if batches[r["key"]] == b
                )
            )
            for b in sorted(set(batches.values()))
        },
    }
    need(
        len(nulls) == 84 and len(roster) == 675 and len(groups) == 674,
        "Validity coverage differs",
    )
    old = io.read(P4.OUT / "common_all_condition_policy_rates.json")
    for name, mask in masks.items():
        for ep, flag in (("FPR", "False"), ("FNR", "True")):
            need(
                mask["common_by_polarity"].get(flag, 0)
                == old[name + "/" + ep]["common_n"],
                "All-condition mask differs",
            )
    write(OUT / "validity_and_dependence.json", data)
    write(
        OUT / "null_and_reduced_vote_outputs.json",
        {"nulls": nulls, "reduced_vote_valid_outputs": reduced},
    )
    return {
        "nulls": len(nulls),
        "reduced_vote_valid_outputs": len(reduced),
        "original_clusters": len(groups),
    }


def main():
    need(not OUT.exists(), "Preserve existing Part 7 output")
    preserved = verify_previous()
    roster, science, predictions, assembly = P5.load()
    specs = specifications(science)
    batches = io.read(reader.OUT / "batches.json")
    test = io.read(PREP / "implementation_checks.json")
    io.verify(ROOT, test["bindings"])
    paths = [
        Path(__file__),
        Path(core.__file__),
        Path(rates.__file__),
        Path(support.__file__),
        Path(P3.__file__),
        Path(P5.__file__),
        Path(P5.independent.__file__),
        io.STUDY / "scientific_plan_v1.json",
        io.STUDY / "STUDY_PROTOCOL.md",
        io.STUDY / "roster.jsonl",
        reader.OUT / "complete.json",
        PREP / "scope.md",
        PREP / "implementation_checks.json",
        *preserved,
    ]
    bindings = {io.rel(p): io.sha(p) for p in paths}
    write(
        OUT / "started.json",
        {
            "created_utc": io.utc(),
            "bindings": bindings,
            "rate_specifications": specs,
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "new_model_calls": 0,
            "provenance": assembly["verification_basis"],
        },
    )
    old = inputs()
    _, counts = rates_and_omissions(roster, predictions, science, specs, batches, old)
    counts.update(source_checks(roster, predictions, science, specs, old))
    counts.update(quality_checks(roster, predictions, science, batches))
    counts.update(validity(roster, predictions, science, batches))
    io.verify(ROOT, bindings)
    write(
        OUT / "statistics_complete.json",
        {
            "status": "part7_statistics_checked_interpretation_pending",
            "created_utc": io.utc(),
            "bindings": bindings,
            "artifacts": {io.rel(p): io.sha(p) for p in OUT.rglob("*") if p.is_file()},
            "counts": counts,
            "new_model_calls": 0,
            "new_training_fits": 0,
            "remaining_parts": [8, 9, 10],
        },
    )
    print(json.dumps(counts, indent=2), flush=True)


if __name__ == "__main__":
    with io.offline():
        main()

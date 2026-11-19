from collections import Counter
from pathlib import Path
import json
import platform
import sys
import numpy as np
from analysis.feedback.reserved import pools as previous
from analysis.feedback.reserved import bounds as bounds
from analysis.feedback.reserved import quality_independent as independent

io, rates, support, reader, need = (
    previous.io,
    previous.rates,
    previous.support,
    previous.reader,
    previous.need,
)
OUT = io.ROOT / "experiments/astra/sq3_reserved_part5_results_20261020_v1"
PREP = io.ROOT / "experiments/astra/sq3_reserved_part5_preparation_20261020_v1"
MASKS = previous.MASKS


def write(path, value):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside Part 5")
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


def save_draws(name, values):
    p = OUT / "draws" / (name.replace("/", "__") + ".npy")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        np.save(f, values, allow_pickle=False)


def load():
    io.completion_gate()
    assembly = io.read(reader.OUT / "complete.json")
    need(
        assembly["status"] == "complete_from_automatically_audited_receipts_unscored"
        and assembly["plan_sha256"] == io.PLAN_SHA,
        "Wrong assembly provenance",
    )
    io.verify(io.ROOT, assembly["artifacts"])
    io.verify(io.ROOT, assembly["input_bindings"])
    old = io.read(previous.OUT / "complete.json")
    io.verify(io.ROOT, old["artifacts"])
    io.verify(io.ROOT, old["report_binding"])
    roster = list(io.rows(io.STUDY / "roster.jsonl"))
    science = io.read(io.STUDY / "scientific_plan_v1.json")
    rates.contrasts(science)
    predictions = {}
    for name in assembly["artifacts"]:
        if not name.endswith(".jsonl"):
            continue
        for row in io.rows(io.ROOT / name):
            key = row["key"], row["condition"], row["repeat"]
            need(key not in predictions, "Duplicate assembly row")
            result = row["result"]
            if result is not None:
                need(
                    type(result["hate"]) is bool
                    and isinstance(result["hate_type"], list)
                    and set(result["hate_type"]) <= set(support.TYPES),
                    "Invalid saved quality labels",
                )
                need(
                    len(set(result["hate_type"])) == len(result["hate_type"]),
                    "Duplicate saved type label",
                )
                result = {
                    "hate": result["hate"],
                    "hate_type": list(result["hate_type"]),
                }
            predictions[key] = {
                "result": result,
                "n_valid": row["n_valid"],
                "stage": row["stage"],
                "messages_sha256": row["messages_sha256"],
            }
    need(
        set(predictions)
        == {
            (r["key"], c, rep)
            for r in roster
            for c in science["conditions"]
            for rep in range(3)
        },
        "Incomplete quality matrix",
    )
    return roster, science, predictions, assembly


def metric_specs(science):
    specs = []
    for cohort, excluded in [
        ("full", []),
        ("same_six_excluded", science["mandatory_excluded_keys"]),
    ]:
        for source in support.supporting_specs(science):
            if source["endpoint"] != "FPR":
                continue
            for fine in (False, True):
                name = (
                    cohort
                    + "/"
                    + source["id"].rsplit("/", 1)[0]
                    + ("/types" if fine else "/binary")
                )
                specs.append(
                    {
                        "name": name,
                        "cohort": cohort,
                        "excluded": excluded,
                        "fine": fine,
                        "spec": {**source, "seed": support.seed_for(name)},
                        "provenance": "Original supporting nonlinear group/reference comparison and deterministic seed",
                    }
                )
        for source in rates.contrasts(science)[2::2]:
            for fine in (False, True):
                name = (
                    cohort
                    + "/supporting_metrics/"
                    + source["id"].rsplit("/", 1)[0]
                    + ("/types" if fine else "/binary")
                )
                specs.append(
                    {
                        "name": name,
                        "cohort": cohort,
                        "excluded": excluded,
                        "fine": fine,
                        "spec": {**source, "seed": support.seed_for(name)},
                        "provenance": "Original supporting nonlinear mechanism comparison; 95%, not its rate-family confidence",
                    }
                )
    need(
        len(specs) == 112 and len({s["name"] for s in specs}) == 112,
        "Incomplete original quality comparison set",
    )
    return specs


def point_and_runs(rows, predictions, spec, fine=False, strict=False):
    eligible = [r for r in rows if not fine or r["gold"]["hate_type"] is not None]
    arms = [spec["treatment_conditions"], spec["control_conditions"]]
    common = support.paired_roster(eligible, predictions, arms, strict=strict)
    runs = [
        [
            support.metric_rows(common, predictions, c, rep, strict=strict)
            for c in arm
            for rep in range(3)
        ]
        for arm in arms
    ]
    for arm in runs:
        for r in arm:
            independent.run_check(
                common, predictions, r["condition"], r["repeat"], r, strict
            )
    names = support.METRICS[2:] if fine else support.METRICS[:2]
    point = {
        "original_n": len(eligible),
        "common_n": len(common),
        "common_ids": [r["key"] for r in common],
        "metrics": {},
    }
    for metric in names:
        points = [
            (
                sum(r["metrics"][metric] for r in arm) / len(arm)
                if all(r["metrics"][metric] is not None for r in arm)
                else None
            )
            for arm in runs
        ]
        point["metrics"][metric] = {
            "treatment": points[0],
            "control": points[1],
            "difference": points[0] - points[1] if None not in points else None,
        }
    checked = independent.paired_point(rows, predictions, spec, fine, strict)
    need(
        checked["common_ids"] == point["common_ids"]
        and checked["original_n"] == point["original_n"],
        "Independent point cohort differs",
    )
    for metric, values in point["metrics"].items():
        for field, v in values.items():
            w = checked["metrics"][metric][field]
            need(
                w is None if v is None else w is not None and abs(v - w) < 1e-13,
                "Independent point metric differs",
            )
    point["independent_exact"] = checked
    return point, runs


def all_runs(roster, predictions, science):
    previous_runs = io.read(previous.OUT / "all_own_valid_binary_runs.json")
    previous_common = io.read(previous.OUT / "common_all_condition_policy_rates.json")
    own = {}
    common = {}
    verified = 0
    for cohort, excluded in [
        ("full", []),
        ("same_six_excluded", science["mandatory_excluded_keys"]),
    ]:
        for mask, strict, reps in MASKS:
            name = cohort + "/" + mask
            rows = support.selected_roster(roster, excluded, reps)
            runs = [
                support.metric_rows(rows, predictions, c, rep, strict=strict)
                for c in science["conditions"]
                for rep in range(3)
            ]
            checked = []
            for run, old in zip(runs, previous_runs[name]):
                check = independent.run_check(
                    rows, predictions, run["condition"], run["repeat"], run, strict
                )
                verified += 1
                checked.append(check)
                need(
                    (run["condition"], run["repeat"])
                    == (old["condition"], old["repeat"]),
                    "Part 4 run order differs",
                )
                ct = run["counts"]
                oc = old["counts"]
                need(
                    ct.get("fp", 0) == oc.get("errors_FPR", 0)
                    and ct.get("fn", 0) == oc.get("errors_FNR", 0),
                    "Part 4 error counts differ",
                )
                need(
                    ct.get("tn", 0) + ct.get("fp", 0) == oc.get("valid_FPR", 0)
                    and ct.get("tp", 0) + ct.get("fn", 0) == oc.get("valid_FNR", 0),
                    "Part 4 denominators differ",
                )
            own[name] = {
                "runs": runs,
                "pool_summary": support.pool_summary(runs),
                "independent_checks": checked,
            }
            paired = support.paired_roster(
                rows, predictions, [science["conditions"]], strict=strict
            )
            ids = {r["key"] for r in paired}
            need(
                ids
                == set(previous_common[name + "/FPR"]["common_ids"])
                | set(previous_common[name + "/FNR"]["common_ids"]),
                "Part 4 shared mask differs",
            )
            cruns = [
                support.metric_rows(paired, predictions, c, rep, strict=strict)
                for c in science["conditions"]
                for rep in range(3)
            ]
            cchecks = []
            for run in cruns:
                cchecks.append(
                    independent.run_check(
                        paired,
                        predictions,
                        run["condition"],
                        run["repeat"],
                        run,
                        strict,
                    )
                )
                verified += 1
            common[name] = {
                "common_n": len(paired),
                "common_ids": [r["key"] for r in paired],
                "binary_gold": dict(Counter(str(r["gold"]["hate"]) for r in paired)),
                "known_type_n": sum(r["gold"]["hate_type"] is not None for r in paired),
                "runs": cruns,
                "pool_summary": support.pool_summary(cruns),
                "independent_checks": cchecks,
                "interpretation": "Descriptive single common mask across all 74 conditions; not a replacement for paired masks.",
            }
            print("Quality run tables checked: " + name, flush=True)
    write(OUT / "all_own_valid_quality_runs.json", own)
    write(OUT / "all_common_mask_quality_runs.json", common)
    return verified


def main():
    need(not OUT.exists(), "Preserve existing Part 5 analysis")
    roster, science, predictions, assembly = load()
    specs = metric_specs(science)
    sources = [
        Path(__file__),
        Path(independent.__file__),
        Path(support.__file__),
        Path(bounds.__file__),
        Path(rates.__file__),
        Path(reader.__file__),
        io.STUDY / "roster.jsonl",
        io.STUDY / "scientific_plan_v1.json",
        reader.OUT / "complete.json",
        previous.OUT / "complete.json",
        PREP / "scope.md",
        PREP / "implementation_checks.json",
    ]
    bindings = {io.rel(p): io.sha(p) for p in sources}
    write(
        OUT / "started.json",
        {
            "created_utc": io.utc(),
            "bindings": bindings,
            "metric_specifications": specs,
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "provenance": assembly["verification_basis"],
            "new_model_calls": 0,
        },
    )
    verified = all_runs(roster, predictions, science)
    results = {}
    checks = {}
    missing = {}
    points = {}
    details = {}
    for index, entry in enumerate(specs, 1):
        name = entry["name"]
        spec = entry["spec"]
        fine = entry["fine"]
        rows = support.selected_roster(roster, entry["excluded"])
        result, draws = support.paired_metrics(rows, predictions, spec, fine=fine)
        result["provenance"] = entry["provenance"]
        results[name] = result
        save_draws(name, draws)
        checks[name] = independent.bootstrap_check(
            rows, predictions, spec, result, draws, fine
        )
        point, runs = point_and_runs(rows, predictions, spec, fine)
        need(point["common_ids"] == result["common_ids"], "Run detail cohort differs")
        details[name] = {"point": point, "arms": runs}
        for strict in (False, True):
            key = name + ("/all_three_votes" if strict else "/ordinary_valid")
            value = bounds.metric_bounds(
                rows,
                predictions,
                [spec["treatment_conditions"], spec["control_conditions"]],
                fine=fine,
                strict=strict,
            )
            value["independent_exact"] = independent.bounds_check(
                rows,
                predictions,
                [spec["treatment_conditions"], spec["control_conditions"]],
                value,
                fine,
                strict,
            )
            missing[key] = value
        for mask, strict, reps in MASKS[1:]:
            chosen = support.selected_roster(roster, entry["excluded"], reps)
            point, _ = point_and_runs(chosen, predictions, spec, fine, strict)
            points[name + "/" + mask] = {
                **point,
                "status": "descriptive_point_only_no_extra_interval",
            }
        if index % 4 == 0 or index == len(specs):
            print(
                f"Quality paired comparisons checked: {index}/{len(specs)}", flush=True
            )
    for name, value in [
        ("paired_quality", results),
        ("independent_checks", checks),
        ("missing_output_bounds", missing),
        ("point_sensitivities", points),
        ("paired_run_details", details),
    ]:
        write(OUT / (name + ".json"), value)
    write(
        OUT / "gold_support.json",
        {
            "inputs": len(roster),
            "known_type_n": sum(r["gold"]["hate_type"] is not None for r in roster),
            "type_support": dict(
                Counter(l for r in roster for l in r["gold"]["hate_type"] or [])
            ),
            "cardinalities": dict(
                Counter(
                    (
                        str(len(r["gold"]["hate_type"]))
                        if r["gold"]["hate_type"] is not None
                        else "unknown"
                    )
                    for r in roster
                )
            ),
            "known_type_sets": dict(
                Counter(
                    "|".join(sorted(r["gold"]["hate_type"]))
                    for r in roster
                    if r["gold"]["hate_type"] is not None
                )
            ),
            "label_order": support.TYPES,
            "interpretation": "Unknown fine labels excluded, never scored as known empty sets.",
        },
    )
    io.verify(io.ROOT, bindings)
    write(
        OUT / "statistics_complete.json",
        {
            "status": "part5_statistics_checked_quality_diagnostics_and_interpretation_pending",
            "created_utc": io.utc(),
            "artifacts": {io.rel(p): io.sha(p) for p in OUT.rglob("*") if p.is_file()},
            "bindings": bindings,
            "predictions": len(predictions),
            "nulls": sum(p["result"] is None for p in predictions.values()),
            "independently_checked_run_summaries": verified,
            "paired_comparison_objects": len(results),
            "scalar_metric_comparisons": sum(
                len(r["metrics"]) for r in results.values()
            ),
            "draw_vectors_checked": sum(c["draws_checked"] for c in checks.values()),
            "scalar_draw_values_checked": sum(
                c["values_checked"] for c in checks.values()
            ),
            "point_sensitivity_objects": len(points),
            "missing_bound_objects": len(missing),
            "new_model_calls": 0,
            "remaining_parts": list(range(6, 11)),
        },
    )
    print(
        "Part 5 frozen quality statistics and independent checks complete.", flush=True
    )


if __name__ == "__main__":
    with io.offline():
        main()

from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
from analysis.feedback.reserved import receipt_reader as reader
from analysis.feedback.reserved import rates as rates
from analysis.feedback.reserved import support as support

io = reader.io
OUT = io.ROOT / "experiments/astra/sq3_reserved_part2_results_20261019_v1"
need = io.require


def write(path, value):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside Part 2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(
            value
            if isinstance(value, str)
            else json.dumps(
                value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
            )
            + "\n"
        )


def save_draws(name, values):
    path = OUT / "draws" / (name.replace("/", "__") + ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.save(f, values, allow_pickle=False)


def exact_check(
    roster, predictions, spec, result, strict=False, reps=False, excluded=()
):

    eligible = [
        r
        for r in roster
        if r["gold"]["hate"] == (spec["endpoint"] == "FNR") and r["key"] not in excluded
    ]
    if reps:
        groups = defaultdict(list)
        for r in eligible:
            groups[r["cluster"]].append(r)
        keep = {
            min(
                g,
                key=lambda r: hashlib.sha256(
                    ("A15-representative|" + r["key"]).encode()
                ).hexdigest(),
            )["key"]
            for g in groups.values()
        }
        eligible = [r for r in eligible if r["key"] in keep]
    common = []
    totals = [Fraction(0), Fraction(0)]
    bound = [Fraction(0), Fraction(0)]
    real_errors = [0, 0]
    real_outputs = [0, 0]
    missing = [0, 0]
    for row in eligible:
        arms = []
        for i, conditions in enumerate(
            (spec["treatment_conditions"], spec["control_conditions"])
        ):
            values = []
            for c in conditions:
                for rep in range(3):
                    p = predictions[row["key"], c, rep]
                    valid = p["result"] is not None and (
                        not strict or p["n_valid"] == 3
                    )
                    values.append(
                        int(p["result"]["hate"] != row["gold"]["hate"])
                        if valid
                        else None
                    )
            missing[i] += values.count(None)
            arms.append(values)
        lows = [
            Fraction(sum(x for x in arm if x is not None), len(arm)) for arm in arms
        ]
        highs = [
            lo + Fraction(arm.count(None), len(arm)) for lo, arm in zip(lows, arms)
        ]
        bound[0] += lows[0] - highs[1]
        bound[1] += highs[0] - lows[1]
        if all(None not in a for a in arms):
            common.append(row["key"])
            for i in (0, 1):
                totals[i] += lows[i]
                real_errors[i] += sum(arms[i])
                real_outputs[i] += len(arms[i])
    need(
        common == result.get("common_ids", [])
        and len(eligible) == result["original_n"],
        "Independent common mask mismatch",
    )
    expected_bounds = [float(b / len(eligible)) for b in bound] if eligible else None
    need(
        expected_bounds is None
        or all(
            abs(a - b) < 1e-12
            for a, b in zip(expected_bounds, result["missing_difference_bounds"])
        ),
        "Independent missing bounds mismatch",
    )
    if common:
        means = [v / len(common) for v in totals]
        need(
            all(
                abs(float(v) - result[k]) < 1e-12
                for v, k in zip(means, ["treatment_rate", "control_rate"])
            ),
            "Independent rates mismatch",
        )
        need(
            abs(float(means[0] - means[1]) - result["difference"]) < 1e-12,
            "Independent difference mismatch",
        )
    return {
        "method": "independent exact integer/Fraction count and mask reconstruction",
        "common_n": len(common),
        "eligible_n": len(eligible),
        "real_error_counts": real_errors,
        "real_output_counts": real_outputs,
        "missing_real_outputs": missing,
        "exact_rates": [str(v / len(common)) for v in totals] if common else None,
        "missing_difference_bounds": expected_bounds,
    }


def descriptive(roster, predictions, spec, result):

    bykey = {r["key"]: r for r in roster}
    common = result["common_ids"]
    runs = []
    pool = []
    per_text = []
    for condition in spec["control_conditions"] + spec["treatment_conditions"]:
        for rep in range(3):
            errors = sum(
                predictions[k, condition, rep]["result"]["hate"]
                != bykey[k]["gold"]["hate"]
                for k in common
            )
            runs.append(
                {
                    "condition": condition,
                    "repeat": rep,
                    "errors": errors,
                    "n": len(common),
                    "rate": errors / len(common),
                }
            )
    for c in spec["treatment_conditions"]:
        errors = sum(x["errors"] for x in runs if x["condition"] == c)
        fixes = regressions = 0
        for k in common:
            truth = bykey[k]["gold"]["hate"]
            for rep in range(3):
                ce = predictions[k, "reference5", rep]["result"]["hate"] != truth
                te = predictions[k, c, rep]["result"]["hate"] != truth
                fixes += ce and not te
                regressions += te and not ce
        pool.append(
            {
                "condition": c,
                "errors": errors,
                "real_outputs": len(common) * 3,
                "rate": errors / (len(common) * 3),
                "difference_vs_reference": errors / (len(common) * 3)
                - result["control_rate"],
                "matched_repeat_fixed": fixes,
                "matched_repeat_regressed": regressions,
                "note": "Descriptive paired-repeat counts; same reference reused across pools; no independent-output inference.",
            }
        )
    for row in roster:
        if row["gold"]["hate"] != (spec["endpoint"] == "FNR"):
            continue
        values = {}
        for c in spec["control_conditions"] + spec["treatment_conditions"]:
            vals = [predictions[row["key"], c, rep] for rep in range(3)]
            values[c] = {
                "errors": [
                    (
                        None
                        if p["result"] is None
                        else int(p["result"]["hate"] != row["gold"]["hate"])
                    )
                    for p in vals
                ],
                "valid_votes": [p["n_valid"] for p in vals],
            }
        per_text.append(
            {
                "key": row["key"],
                "source": row["source"],
                "cluster": row["cluster"],
                "gold_hate": row["gold"]["hate"],
                "in_primary_common_mask": row["key"] in common,
                "outputs": values,
            }
        )
    direction = Counter()
    for k in common:
        truth = bykey[k]["gold"]["hate"]
        t = sum(
            predictions[k, c, rep]["result"]["hate"] != truth
            for c in spec["treatment_conditions"]
            for rep in range(3)
        )
        c = sum(
            predictions[k, "reference5", rep]["result"]["hate"] != truth
            for rep in range(3)
        )
        direction[
            (
                "fewer_errors"
                if t < 3 * c
                else "more_errors" if t > 3 * c else "same_average_errors"
            )
        ] += 1
    return {
        "runs": runs,
        "pools": pool,
        "per_text_average_direction": dict(direction),
        "per_text": per_text,
    }


def main():
    io.completion_gate()
    need(not OUT.exists(), "Preserve earlier Part 2 outputs")
    assembly = io.read(reader.OUT / "complete.json")
    need(
        assembly["status"] == "complete_from_automatically_audited_receipts_unscored"
        and assembly["plan_sha256"] == io.PLAN_SHA,
        "Wrong assembly contract",
    )
    io.verify(io.ROOT, assembly["artifacts"])
    io.verify(io.ROOT, assembly["input_bindings"])
    roster = list(io.rows(io.STUDY / "roster.jsonl"))
    science = io.read(io.STUDY / "scientific_plan_v1.json")
    specs = [s for s in rates.contrasts(science) if s["family"] == "primary"]
    need(len(specs) == 2, "Wrong primary family")
    conditions = set(specs[0]["treatment_conditions"] + specs[0]["control_conditions"])
    predictions = {}
    seen = set()
    for name in assembly["artifacts"]:
        if not name.endswith(".jsonl"):
            continue
        for row in io.rows(io.ROOT / name):
            key = (row["key"], row["condition"], row["repeat"])
            need(key not in seen, "Duplicate assembly row")
            seen.add(key)
            if row["condition"] in conditions:
                predictions[key] = row
    need(
        seen
        == {
            (r["key"], c, rep)
            for r in roster
            for c in science["conditions"]
            for rep in range(3)
        },
        "Incomplete full assembly matrix",
    )
    need(len(predictions) == 675 * 4 * 3, "Incomplete primary matrix")
    batches = io.read(reader.OUT / "batches.json")
    source_files = [
        Path(__file__),
        Path(reader.__file__),
        Path(rates.__file__),
        Path(support.__file__),
        reader.OUT / "complete.json",
        io.STUDY / "roster.jsonl",
        io.STUDY / "scientific_plan_v1.json",
        io.ROOT
        / "experiments/astra/sq3_reserved_part2_preparation_20261019_v1/scope.md",
    ]
    bindings = {io.rel(p): io.sha(p) for p in source_files}
    write(
        OUT / "started.json",
        {
            "started_utc": io.utc(),
            "bindings": bindings,
            "specs": specs,
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "reader_basis": assembly["verification_basis"],
            "new_model_calls": 0,
            "parts_scored": [2],
        },
    )
    results = {}
    checks = {}
    omissions = {}
    descriptions = {}
    sources = {}
    for cohort_name, excluded in [
        ("full", []),
        ("same_six_excluded", science["mandatory_excluded_keys"]),
    ]:
        for mask, strict, reps in [
            ("ordinary_valid", False, False),
            ("all_three_votes", True, False),
            ("one_per_cluster", False, True),
        ]:
            for spec in specs:
                name = cohort_name + "/" + mask + "/" + spec["id"]
                cohort = rates.cohort(
                    roster,
                    predictions,
                    spec,
                    strict=strict,
                    representatives=reps,
                    excluded=excluded,
                )
                result, draws = rates.summarize(cohort, spec)
                results[name] = result
                save_draws(name, draws)
                checks[name] = {
                    "exact": exact_check(
                        roster, predictions, spec, result, strict, reps, excluded
                    ),
                    "bootstrap": support.independent_rate_draws(
                        roster, predictions, spec, cohort, draws
                    ),
                }
                omissions[name] = support.omission_points(
                    roster,
                    predictions,
                    spec,
                    batches,
                    strict=strict,
                    representatives=reps,
                    excluded=excluded,
                )
                if cohort_name == "full" and mask == "ordinary_valid":
                    descriptions[spec["endpoint"]] = descriptive(
                        roster, predictions, spec, result
                    )
                print(
                    f"Primary analysis checked: {name}; n={result['common_n']}",
                    flush=True,
                )
        eligible = [r for r in roster if r["key"] not in excluded]
        for spec in specs:
            name = cohort_name + "/" + spec["id"]
            cohort = rates.cohort(roster, predictions, spec, excluded=excluded)
            stratified_spec = {
                **spec,
                "seed": support.seed_for("source_stratified/" + name),
            }
            result, draws = support.source_bootstrap(cohort, roster, stratified_spec)
            sources[name + "/stratified"] = result
            save_draws(name + "/stratified", draws)
            for source in sorted({r["source"] for r in roster}):
                subset = [r for r in eligible if r["source"] == source]
                sub_spec = {
                    **spec,
                    "confidence": 0.95,
                    "seed": support.seed_for("source_specific/" + name + "/" + source),
                }
                cohort = rates.cohort(subset, predictions, sub_spec)
                result, draws = rates.summarize(cohort, sub_spec)
                result["interpretation"] = (
                    "Supporting source-specific unadjusted 95% interval; not a primary discovery."
                )
                ident = name + "/source/" + source
                sources[ident] = result
                save_draws(ident, draws)
                checks[ident] = {
                    "exact": exact_check(subset, predictions, sub_spec, result),
                    "bootstrap": support.independent_rate_draws(
                        subset, predictions, sub_spec, cohort, draws
                    ),
                }
    for endpoint, value in descriptions.items():
        per_text = value.pop("per_text")
        write(
            OUT / ("per_text_" + endpoint + ".jsonl"),
            "".join(json.dumps(r, sort_keys=True) + "\n" for r in per_text),
        )
    write(OUT / "primary_and_sensitivities.json", results)
    write(OUT / "independent_checks.json", checks)
    write(OUT / "source_effects.json", sources)
    write(OUT / "omission_points.json", omissions)
    write(OUT / "descriptive_counts.json", descriptions)
    nulls = [
        {
            "key": k,
            "condition": c,
            "repeat": rep,
            "stage": v["stage"],
            "n_valid": v["n_valid"],
        }
        for (k, c, rep), v in predictions.items()
        if v["result"] is None
    ]
    write(OUT / "primary_nulls.json", nulls)
    io.verify(io.ROOT, bindings)
    artifacts = {io.rel(p): io.sha(p) for p in OUT.rglob("*") if p.is_file()}
    write(
        OUT / "complete.json",
        {
            "status": "part2_primary_calculated_and_numerically_checked_interpretation_pending",
            "completed_utc": io.utc(),
            "artifacts": artifacts,
            "bindings": bindings,
            "new_model_calls": 0,
            "primary_predictions": len(predictions),
            "null_primary_predictions": len(nulls),
            "ordinary_rate_draws_independently_checked": sum(
                c["bootstrap"]["draws_checked"] for c in checks.values()
            ),
            "remaining_parts": list(range(3, 11)),
            "scope": "Two co-primary endpoints and their required robustness only; other conditions assembled but not scientifically scored.",
        },
    )
    for endpoint in ("FPR", "FNR"):
        result = results["full/ordinary_valid/primary/reference5/" + endpoint]
        print(
            json.dumps(
                {
                    k: result[k]
                    for k in (
                        "endpoint",
                        "original_n",
                        "common_n",
                        "treatment_rate",
                        "control_rate",
                        "difference",
                        "family_interval",
                        "missing_difference_bounds",
                    )
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    with io.offline():
        main()

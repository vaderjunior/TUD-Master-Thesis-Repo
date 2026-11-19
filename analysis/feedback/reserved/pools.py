from collections import Counter, defaultdict
from fractions import Fraction
import itertools
import json
import math
from pathlib import Path
import platform
import sys
import numpy as np
from analysis.feedback.reserved import controls as prior

io, rates, support, reader, need = (
    prior.io,
    prior.rates,
    prior.support,
    prior.reader,
    prior.need,
)
OUT = io.ROOT / "experiments/astra/sq3_reserved_part4_results_20261020_v1"
PREP = io.ROOT / "experiments/astra/sq3_reserved_part4_preparation_20261020_v1"
MODES = ("natural5", "reserved5", "extended6", "masked5")
SELECTORS = ("error", "correct", "random")
COMPOSITIONS = ("positive", "balanced")
POLICIES = [s + "_" + p for s in SELECTORS for p in COMPOSITIONS]
MASKS = [
    ("ordinary_valid", False, False),
    ("all_three_votes", True, False),
    ("one_per_cluster", False, True),
]


def write(path, value):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside Part 4")
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


def jsonlines(path, values):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside Part 4")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        for row in values:
            f.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )


def save_draws(name, values):
    path = OUT / "draws" / (name.replace("/", "__") + ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.save(f, values, allow_pickle=False)


def group_conditions(mode, policy, seeds):
    return [mode + "__" + policy + "_s" + str(s) for s in seeds]


def direct_specs(science):
    pairs = []
    for mode in MODES:
        for composition in COMPOSITIONS:
            for left, right in (
                ("error", "correct"),
                ("error", "random"),
                ("correct", "random"),
            ):
                pairs.append(
                    (
                        mode,
                        left + "_" + composition,
                        right + "_" + composition,
                        "selection",
                    )
                )
        for selector in SELECTORS:
            pairs.append(
                (mode, selector + "_balanced", selector + "_positive", "composition")
            )
    specs = []
    for mode, left, right, dimension in pairs:
        for endpoint in ("FPR", "FNR"):
            name = (
                "part4-direct-v1/"
                + mode
                + "/"
                + left
                + "-minus-"
                + right
                + "/"
                + endpoint
            )
            specs.append(
                {
                    "id": name,
                    "family": "additional_part4_supporting",
                    "dimension": dimension,
                    "mode": mode,
                    "treatment_policy": left,
                    "control_policy": right,
                    "endpoint": endpoint,
                    "treatment_conditions": group_conditions(
                        mode, left, science["pool_seeds"]
                    ),
                    "control_conditions": group_conditions(
                        mode, right, science["pool_seeds"]
                    ),
                    "confidence": 0.95,
                    "draws": science["bootstrap_draws"],
                    "seed": support.seed_for(name),
                    "provenance": "Complete direct pair set specified in Part 4 after Parts 2/3; not enumerated before execution or multiplicity-adjusted.",
                }
            )
    need(
        len(specs) == 72 and len({s["id"] for s in specs}) == 72,
        "Incomplete direct policy set",
    )
    return specs


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
    roster = list(io.rows(io.STUDY / "roster.jsonl"))
    science = io.read(io.STUDY / "scientific_plan_v1.json")
    rates.contrasts(science)
    predictions = {}
    for name in assembly["artifacts"]:
        if name.endswith(".jsonl"):
            for row in io.rows(io.ROOT / name):
                key = row["key"], row["condition"], row["repeat"]
                need(key not in predictions, "Duplicate prediction")
                predictions[key] = {
                    "result": (
                        None
                        if row["result"] is None
                        else {"hate": row["result"]["hate"]}
                    ),
                    "n_valid": row["n_valid"],
                    "messages_sha256": row["messages_sha256"],
                    "stage": row["stage"],
                }
    need(
        set(predictions)
        == {
            (r["key"], c, rep)
            for r in roster
            for c in science["conditions"]
            for rep in range(3)
        },
        "Incomplete 675x74x3 matrix",
    )
    return roster, science, predictions, assembly


def point_result(c, spec):
    keys = c["common_ids"]
    eligible = c["original_ids"]
    bounds = c["bounds"]
    out = {
        "contrast": spec["id"],
        "endpoint": spec["endpoint"],
        "original_n": len(eligible),
        "common_n": len(keys),
        "common_ids": keys,
        "clusters": len(set(c["clusters"].values())),
        "outputs_per_input": c["outputs_per_input"],
        "strict_all_three_votes": c["strict_all_three_votes"],
        "representatives_before_validity": c["representatives_before_validity"],
        "missing_difference_bounds": (
            None
            if not eligible
            else [
                math.fsum(bounds[k][0][0] - bounds[k][1][1] for k in eligible)
                / len(eligible),
                math.fsum(bounds[k][0][1] - bounds[k][1][0] for k in eligible)
                / len(eligible),
            ]
        ),
        "status": (
            "descriptive_point_only" if keys else "not_estimable_no_common_valid_inputs"
        ),
    }
    if keys:
        means = [math.fsum(c["errors"][k][i] for k in keys) / len(keys) for i in (0, 1)]
        out.update(
            treatment_rate=means[0],
            control_rate=means[1],
            difference=means[0] - means[1],
        )
    return out


def comparison_details(roster, predictions, spec, result):

    bykey = {r["key"]: r for r in roster}
    keys = result["common_ids"]
    ts = spec["treatment_conditions"]
    cs = spec["control_conditions"]
    need(len(ts) in (1, 3) and len(cs) in (1, 3), "Unexpected pool count")
    pairs = list(zip(ts, cs)) if len(ts) == len(cs) else [(t, cs[0]) for t in ts]
    runs = []
    pools = []
    direction = Counter()
    sources = {}
    for t, c in pairs:
        errors = [0, 0]
        for rep in range(3):
            e = [
                sum(
                    predictions[k, condition, rep]["result"]["hate"]
                    != bykey[k]["gold"]["hate"]
                    for k in keys
                )
                for condition in (t, c)
            ]
            for i in (0, 1):
                errors[i] += e[i]
            runs.append(
                {
                    "treatment": t,
                    "control": c,
                    "repeat": rep,
                    "common_n": len(keys),
                    "errors": e,
                    "difference": (e[0] - e[1]) / len(keys),
                }
            )
        pools.append(
            {
                "treatment": t,
                "control": c,
                "common_n": len(keys),
                "errors": errors,
                "actual_outputs_per_arm": 3 * len(keys),
                "difference": (errors[0] - errors[1]) / (3 * len(keys)),
            }
        )
    for k in keys:
        e = [
            Fraction(
                sum(
                    predictions[k, condition, rep]["result"]["hate"]
                    != bykey[k]["gold"]["hate"]
                    for condition in arm
                    for rep in range(3)
                ),
                3 * len(arm),
            )
            for arm in (ts, cs)
        ]
        direction[
            (
                "fewer_errors"
                if e[0] < e[1]
                else "more_errors" if e[0] > e[1] else "same_average_errors"
            )
        ] += 1
    for source in sorted({r["source"] for r in roster}):
        subset = [k for k in keys if bykey[k]["source"] == source]
        errors = [
            sum(
                predictions[k, c, rep]["result"]["hate"] != bykey[k]["gold"]["hate"]
                for k in subset
                for c in arm
                for rep in range(3)
            )
            for arm in (ts, cs)
        ]
        den = [len(subset) * 3 * len(arm) for arm in (ts, cs)]
        sources[source] = {
            "common_n": len(subset),
            "errors": errors,
            "actual_outputs": den,
            "difference": (
                float(Fraction(errors[0], den[0]) - Fraction(errors[1], den[1]))
                if subset
                else None
            ),
            "interpretation": "Source-specific point on the parent common mask; no interval or cross-source claim.",
        }
    return {
        "runs": runs,
        "pools": pools,
        "per_text_direction": dict(direction),
        "sources": sources,
        "reference_note": "Reference predictions reused descriptively in pool/run differences; do not sum them as independent reference outputs.",
    }


def own_valid_and_common(roster, predictions, science):
    own = {}
    common_tables = {}
    dependence = []
    groups = {"reference5": ["reference5"], "reference4": ["reference4"]}
    groups.update(
        {
            mode + "__" + policy: group_conditions(mode, policy, science["pool_seeds"])
            for mode in MODES
            for policy in POLICIES
        }
    )
    for cohort_name, excluded in [
        ("full", []),
        ("same_six_excluded", science["mandatory_excluded_keys"]),
    ]:
        for mask, strict, reps in MASKS:
            rows = support.selected_roster(roster, excluded, reps)
            prefix = cohort_name + "/" + mask
            run = []
            for condition in science["conditions"]:
                for rep in range(3):
                    counts = Counter()
                    votes = Counter()
                    for row in rows:
                        p = predictions[row["key"], condition, rep]
                        votes[str(p["n_valid"])] += 1
                        endpoint = "FNR" if row["gold"]["hate"] else "FPR"
                        counts["eligible_" + endpoint] += 1
                        if not rates.valid_prediction(p, strict):
                            counts["missing_" + endpoint] += 1
                            continue
                        counts["valid_" + endpoint] += 1
                        counts["errors_" + endpoint] += (
                            p["result"]["hate"] != row["gold"]["hate"]
                        )
                    run.append(
                        {
                            "condition": condition,
                            "repeat": rep,
                            "counts": dict(counts),
                            "valid_votes": dict(votes),
                            "rates": {
                                e: (
                                    counts["errors_" + e] / counts["valid_" + e]
                                    if counts["valid_" + e]
                                    else None
                                )
                                for e in ("FPR", "FNR")
                            },
                        }
                    )
            own[prefix] = run
            for endpoint in ("FPR", "FNR"):
                eligible = [r for r in rows if r["gold"]["hate"] == (endpoint == "FNR")]
                usable = [
                    r
                    for r in eligible
                    if all(
                        rates.valid_prediction(predictions[r["key"], c, rep], strict)
                        for c in science["conditions"]
                        for rep in range(3)
                    )
                ]
                result = {}
                for group, conditions in groups.items():
                    values = []
                    for c in conditions:
                        errors = sum(
                            predictions[r["key"], c, rep]["result"]["hate"]
                            != r["gold"]["hate"]
                            for r in usable
                            for rep in range(3)
                        )
                        den = 3 * len(usable)
                        values.append(
                            {
                                "condition": c,
                                "errors": errors,
                                "actual_outputs": den,
                                "rate": errors / den if den else None,
                            }
                        )
                    result[group] = {
                        "pools": values,
                        "rate": (
                            float(np.mean([v["rate"] for v in values]))
                            if usable
                            else None
                        ),
                        "pool_min": min(v["rate"] for v in values) if usable else None,
                        "pool_max": max(v["rate"] for v in values) if usable else None,
                        "pool_sd": (
                            float(np.std([v["rate"] for v in values], ddof=1))
                            if usable and len(values) > 1
                            else None
                        ),
                    }
                    if (
                        cohort_name == "full"
                        and mask == "ordinary_valid"
                        and len(conditions) == 3
                    ):
                        per_text = {
                            c: np.array(
                                [
                                    sum(
                                        predictions[r["key"], c, rep]["result"]["hate"]
                                        != r["gold"]["hate"]
                                        for rep in range(3)
                                    )
                                    / 3
                                    for r in usable
                                ]
                            )
                            for c in conditions
                        }
                        for left, right in itertools.combinations(conditions, 2):
                            x, y = per_text[left], per_text[right]
                            dependence.append(
                                {
                                    "group": group,
                                    "endpoint": endpoint,
                                    "left": left,
                                    "right": right,
                                    "common_n": len(usable),
                                    "correlation": (
                                        float(np.corrcoef(x, y)[0, 1])
                                        if np.std(x) > 0 and np.std(y) > 0
                                        else None
                                    ),
                                    "same_text_mean_count": int(np.sum(x == y)),
                                    "interpretation": "Observed dependence across fixed pools on shared texts, not an effective independent sample size.",
                                }
                            )
                common_tables[prefix + "/" + endpoint] = {
                    "eligible_n": len(eligible),
                    "common_n": len(usable),
                    "common_ids": [r["key"] for r in usable],
                    "groups": result,
                    "interpretation": "Descriptive shared mask across all 74 conditions; does not replace any comparison-specific frozen estimate.",
                }
    return own, common_tables, dependence


def main():
    need(not OUT.exists(), "Preserve earlier Part 4 outputs")
    roster, science, predictions, assembly = load()
    frozen = support.supporting_specs(science)
    direct = direct_specs(science)
    need(len(frozen) == 50, "Wrong frozen supporting family")
    specs = frozen + direct
    paths = [
        Path(__file__),
        Path(prior.__file__),
        Path(prior.prior.__file__),
        Path(rates.__file__),
        Path(support.__file__),
        Path(reader.__file__),
        reader.OUT / "complete.json",
        prior.OUT / "complete.json",
        io.STUDY / "scientific_plan_v1.json",
        io.STUDY / "roster.jsonl",
        PREP / "scope.md",
        PREP / "implementation_checks.json",
    ]
    bindings = {io.rel(p): io.sha(p) for p in paths}
    write(
        OUT / "started.json",
        {
            "created_utc": io.utc(),
            "bindings": bindings,
            "frozen_reference_specs": frozen,
            "additional_direct_specs": direct,
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "reader_basis": assembly["verification_basis"],
            "new_model_calls": 0,
            "scope": "Part 4 binary policy comparisons; Part 5 fine-label/F1/complete-answer review remains pending.",
        },
    )
    results = {}
    checks = {}
    sensitivities = {}
    omissions = {}
    details = {}
    batches = io.read(reader.OUT / "batches.json")
    for cohort_name, excluded in [
        ("full", []),
        ("same_six_excluded", science["mandatory_excluded_keys"]),
    ]:
        for index, spec in enumerate(specs, 1):
            name = cohort_name + "/" + spec["id"]
            c = rates.cohort(roster, predictions, spec, excluded=excluded)
            r, dist = rates.summarize(c, spec)
            r["interpretation"] = (
                "Pre-execution supporting 95% reference comparison"
                if spec["family"] == "supporting"
                else spec["provenance"]
            )
            results[name] = r
            save_draws(name, dist)
            integer = prior.integer_draw_check(roster, predictions, spec, c, dist)
            need(
                all(
                    abs(x - y) < 1e-12
                    for x, y in zip(integer["exact_interval"], r["family_interval"])
                ),
                "Exact quantile mismatch",
            )
            checks[name] = {
                "exact": prior.prior.exact_check(
                    roster, predictions, spec, r, excluded=excluded
                ),
                "integer_bootstrap": integer,
            }
            if spec["family"] != "supporting":
                checks[name]["expanded_bootstrap"] = support.independent_rate_draws(
                    roster, predictions, spec, c, dist
                )
            details[name] = comparison_details(roster, predictions, spec, r)
            omissions[name] = support.omission_points(
                roster, predictions, spec, batches, excluded=excluded
            )
            for mask, strict, reps in MASKS[1:]:
                c = rates.cohort(
                    roster,
                    predictions,
                    spec,
                    strict=strict,
                    representatives=reps,
                    excluded=excluded,
                )
                point = point_result(c, spec)
                checked = prior.prior.exact_check(
                    roster, predictions, spec, point, strict, reps, excluded
                )
                sensitivities[cohort_name + "/" + mask + "/" + spec["id"]] = {
                    **point,
                    "independent_exact_check": checked,
                }
            if index % 10 == 0 or index == len(specs):
                print(
                    f"Part 4 {cohort_name}: {index}/{len(specs)} comparisons independently checked",
                    flush=True,
                )
    own, common, dependence = own_valid_and_common(roster, predictions, science)
    old = io.read(prior.OUT / "supporting_reference_controls.json")
    reused = []
    for key, value in old.items():
        current = dict(results[key])
        current.pop("interpretation")
        need(current == value, "Previous supporting result changed: " + key)
        old_draw = prior.OUT / "draws" / (key.replace("/", "__") + ".npy")
        new_draw = OUT / "draws" / (key.replace("/", "__") + ".npy")
        need(io.sha(old_draw) == io.sha(new_draw), "Previous supporting draws changed")
        reused.append(key)
    nulls = [
        {"key": k, "condition": c, "repeat": rep, "stage": v["stage"]}
        for (k, c, rep), v in predictions.items()
        if v["result"] is None
    ]
    for name, value in [
        ("rate_comparisons", results),
        ("independent_checks", checks),
        ("point_sensitivities", sensitivities),
        ("omission_points", omissions),
        ("comparison_details", details),
        ("all_own_valid_binary_runs", own),
        ("common_all_condition_policy_rates", common),
        ("pool_dependence", dependence),
        ("nulls", nulls),
        (
            "part3_reproduction",
            {"identical_results_and_draws": reused, "count": len(reused)},
        ),
    ]:
        write(OUT / (name + ".json"), value)
    io.verify(io.ROOT, bindings)
    write(
        OUT / "statistics_complete.json",
        {
            "status": "part4_statistics_checked_context_and_interpretation_pending",
            "created_utc": io.utc(),
            "artifacts": {io.rel(p): io.sha(p) for p in OUT.rglob("*") if p.is_file()},
            "bindings": bindings,
            "predictions": len(predictions),
            "nulls": len(nulls),
            "reference_comparison_versions": 100,
            "additional_direct_comparison_versions": 144,
            "point_sensitivity_versions": len(sensitivities),
            "integer_draws_checked": sum(
                c["integer_bootstrap"]["draws_checked"] for c in checks.values()
            ),
            "additionally_expanded_draws_checked": sum(
                c.get("expanded_bootstrap", {}).get("draws_checked", 0)
                for c in checks.values()
            ),
            "new_model_calls": 0,
            "remaining_parts": list(range(5, 11)),
        },
    )
    print(
        "All Part 4 binary policy statistics saved and checked; scientific interpretation pending.",
        flush=True,
    )


if __name__ == "__main__":
    with io.offline():
        main()

from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import numpy as np
from analysis.feedback.reserved import effects as prior

reader, rates, support, io, need = (
    prior.reader,
    prior.rates,
    prior.support,
    prior.io,
    prior.need,
)
OUT = io.ROOT / "experiments/astra/sq3_reserved_part3_results_20261019_v2"
PREP = io.ROOT / "experiments/astra/sq3_reserved_part3_preparation_20261019_v1"
MODES = ("natural5", "reserved5", "extended6", "masked5")


def write(path, value):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside Part 3")
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


def jsonlines(path, values):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside Part 3")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        for row in values:
            f.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            )


def save_draws(name, values):
    path = OUT / "draws" / (name.replace("/", "__") + ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.save(f, values, allow_pickle=False)


def conditions_for(science):
    return ["reference5", "reference4"] + [
        mode + "__error_balanced_s" + str(s)
        for mode in MODES
        for s in science["pool_seeds"]
    ]


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
    conditions = conditions_for(science)
    predictions = {}
    seen = set()
    for name in assembly["artifacts"]:
        if not name.endswith(".jsonl"):
            continue
        for row in io.rows(io.ROOT / name):
            key = row["key"], row["condition"], row["repeat"]
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
        "Incomplete full assembly",
    )
    need(len(predictions) == 675 * 14 * 3, "Incomplete Part 3 matrix")
    return roster, science, predictions, assembly


def integer_draw_check(roster, predictions, spec, cohort, saved, *, stratified=False):

    if not cohort["common_ids"]:
        need(len(saved) == 0, "Nonempty draws for empty cohort")
        return {"draws_checked": 0, "max_absolute_error": 0.0, "exact_interval": None}
    bykey = {r["key"]: r for r in roster}
    groups = defaultdict(list)
    for k in cohort["common_ids"]:
        groups[bykey[k]["cluster"]].append(k)
    labels = sorted(groups)
    arms = [spec["treatment_conditions"], spec["control_conditions"]]
    counts = [3 * len(a) for a in arms]
    denominator_unit = math.lcm(*counts)
    nums, sizes = [], []
    for label in labels:
        numerator = 0
        for k in groups[label]:
            gold = bykey[k]["gold"]["hate"]
            error = [
                sum(
                    predictions[k, c, r]["result"]["hate"] != gold
                    for c in arm
                    for r in range(3)
                )
                for arm in arms
            ]
            numerator += error[0] * (denominator_unit // counts[0]) - error[1] * (
                denominator_unit // counts[1]
            )
        nums.append(numerator)
        sizes.append(len(groups[label]))
    nums = np.asarray(nums, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.int64)
    strata = defaultdict(list)
    for i, label in enumerate(labels):
        strata[tuple(sorted({bykey[k]["source"] for k in groups[label]}))].append(i)
    rng = np.random.Generator(np.random.PCG64(spec["seed"]))
    exact = []
    zeros = 0
    for start in range(0, spec["draws"], 250):
        length = min(250, spec["draws"] - start)
        if stratified:
            chosen = np.concatenate(
                [
                    np.asarray(v)[rng.integers(0, len(v), (length, len(v)))]
                    for _, v in sorted(strata.items())
                ],
                axis=1,
            )
        else:
            chosen = rng.integers(0, len(labels), (length, len(labels)))
        numerators = nums[chosen].sum(axis=1)
        denominators = denominator_unit * sizes[chosen].sum(axis=1)
        exact.extend(Fraction(int(n), int(d)) for n, d in zip(numerators, denominators))
        zeros += int((numerators == 0).sum())
    delta = float(np.max(np.abs(np.asarray([float(v) for v in exact]) - saved)))
    need(delta < 1e-12, "Integer bootstrap differs")
    confidence = (
        Fraction(19, 20)
        if stratified
        else Fraction(spec["confidence"]).limit_denominator(10000)
    )
    alpha = (1 - confidence) / 2
    ordered = sorted(exact)
    endpoints = []
    tails = []
    for q in (alpha, 1 - alpha):
        position = (len(ordered) - 1) * q
        lo = math.floor(position)
        hi = math.ceil(position)
        weight = position - lo
        value = (1 - weight) * ordered[lo] + weight * ordered[hi]
        endpoints.append(float(value))
        tails.append(
            {
                "q": str(q),
                "ranks": [lo, hi],
                "weight": str(weight),
                "adjacent_values": [str(ordered[lo]), str(ordered[hi])],
                "exact_quantile": str(value),
            }
        )
    return {
        "draws_checked": len(exact),
        "max_absolute_error": delta,
        "denominator_unit": denominator_unit,
        "outputs_per_input": counts,
        "confidence_exact": str(confidence),
        "exact_interval": endpoints,
        "tails": tails,
        "zero_draws": zeros,
        "method": "integer decisions and exact rational linear quantiles",
        "source_stratified": stratified,
    }


def describe(roster, predictions, spec, result):
    bykey = {r["key"]: r for r in roster}
    keys = result["common_ids"]
    arms = [spec["treatment_conditions"], spec["control_conditions"]]
    runs = []
    for arm, conditions in zip(("treatment", "control"), arms):
        for c in conditions:
            for rep in range(3):
                errors = sum(
                    predictions[k, c, rep]["result"]["hate"] != bykey[k]["gold"]["hate"]
                    for k in keys
                )
                runs.append(
                    {
                        "arm": arm,
                        "condition": c,
                        "repeat": rep,
                        "errors": errors,
                        "n": len(keys),
                        "rate": errors / len(keys),
                    }
                )
    pools = []
    for t, c in zip(*arms):
        te = sum(x["errors"] for x in runs if x["condition"] == t)
        ce = sum(x["errors"] for x in runs if x["condition"] == c)
        counts = Counter()
        for k in keys:
            truth = bykey[k]["gold"]["hate"]
            for rep in range(3):
                a = predictions[k, t, rep]["result"]["hate"] != truth
                b = predictions[k, c, rep]["result"]["hate"] != truth
                counts[
                    (
                        "fix"
                        if b and not a
                        else (
                            "regression"
                            if a and not b
                            else "unchanged_error" if a else "unchanged_correct"
                        )
                    )
                ] += 1
        pools.append(
            {
                "treatment": t,
                "control": c,
                "n_texts": len(keys),
                "real_outputs_per_arm": 3 * len(keys),
                "treatment_errors": te,
                "control_errors": ce,
                "difference": (te - ce) / (3 * len(keys)),
                "matched_repeat_counts": dict(counts),
            }
        )
    direction = Counter()
    per_text = []
    for k in keys:
        nums = [
            sum(
                predictions[k, c, r]["result"]["hate"] != bykey[k]["gold"]["hate"]
                for c in cs
                for r in range(3)
            )
            for cs in arms
        ]
        difference = Fraction(nums[0], 3 * len(arms[0])) - Fraction(
            nums[1], 3 * len(arms[1])
        )
        direction[
            (
                "fewer_errors"
                if difference < 0
                else "more_errors" if difference > 0 else "same_average_errors"
            )
        ] += 1
        per_text.append(
            {
                "key": k,
                "source": bykey[k]["source"],
                "cluster": bykey[k]["cluster"],
                "error_counts": nums,
                "outputs_per_arm": [3 * len(a) for a in arms],
                "difference_exact": str(difference),
            }
        )
    return {
        "runs": runs,
        "pools": pools,
        "per_text_direction": dict(direction),
        "per_text": per_text,
        "unit": "real outputs descriptive; inference uses original text clusters, not independent repeats",
    }


def common_context_decomposition(roster, predictions, science):

    conditions = conditions_for(science)
    groups = {"reference5": ["reference5"], "reference4": ["reference4"]}
    groups.update({m: [c for c in conditions if c.startswith(m + "__")] for m in MODES})
    result = {}
    for name, excluded in [
        ("full", []),
        ("same_six_excluded", science["mandatory_excluded_keys"]),
    ]:
        for endpoint in ("FPR", "FNR"):
            eligible = [
                r
                for r in roster
                if r["gold"]["hate"] == (endpoint == "FNR") and r["key"] not in excluded
            ]
            rows = [
                r
                for r in eligible
                if all(
                    rates.valid_prediction(predictions[r["key"], c, rep], False)
                    for c in conditions
                    for rep in range(3)
                )
            ]
            means = {}
            totals = {}
            for group, cs in groups.items():
                errors = sum(
                    predictions[r["key"], c, rep]["result"]["hate"] != r["gold"]["hate"]
                    for r in rows
                    for c in cs
                    for rep in range(3)
                )
                denom = len(rows) * 3 * len(cs)
                means[group] = Fraction(errors, denom)
                totals[group] = {
                    "errors": errors,
                    "real_outputs": denom,
                    "rate": float(means[group]),
                    "exact_rate": str(means[group]),
                }
            diff = {
                a + "-" + b: float(means[a] - means[b])
                for a, b in [
                    ("reference4", "reference5"),
                    ("masked5", "reference4"),
                    ("reserved5", "masked5"),
                    ("reserved5", "reference4"),
                    ("reserved5", "reference5"),
                    ("extended6", "reserved5"),
                    ("natural5", "reference5"),
                ]
            }
            need(
                (means["reference4"] - means["reference5"])
                + (means["masked5"] - means["reference4"])
                + (means["reserved5"] - means["masked5"])
                == means["reserved5"] - means["reference5"],
                "Decomposition identity failed",
            )
            result[name + "/" + endpoint] = {
                "eligible_n": len(eligible),
                "common_n": len(rows),
                "common_ids": [r["key"] for r in rows],
                "groups": totals,
                "differences": diff,
                "interpretation": "Additional descriptive common-all-14 mask; no new uncertainty family or mediation claim.",
            }
    return result


def main():
    need(not OUT.exists(), "Preserve earlier Part 3 output")
    roster, science, predictions, assembly = load()
    specs = [s for s in rates.contrasts(science) if s["family"] == "mechanism"]
    need(len(specs) == 6, "Wrong mechanism family")
    support_groups = {
        "reference4",
        "natural5__error_balanced",
        "extended6__error_balanced",
        "masked5__error_balanced",
    }
    secondary = [
        s
        for s in support.supporting_specs(science)
        if s["id"].split("/")[1].rsplit("-reference5", 1)[0] in support_groups
    ]
    need(len(secondary) == 8, "Wrong supporting comparisons")
    paths = [
        Path(__file__),
        Path(prior.__file__),
        Path(reader.__file__),
        Path(rates.__file__),
        Path(support.__file__),
        reader.OUT / "complete.json",
        io.STUDY / "roster.jsonl",
        io.STUDY / "scientific_plan_v1.json",
        PREP / "scope.md",
        PREP / "implementation_checks_v2.json",
        PREP / "repair_amendment_v2.json",
        prior.OUT / "complete.json",
        prior.OUT / "interpretation_check.json",
    ]
    bindings = {io.rel(p): io.sha(p) for p in paths}
    write(
        OUT / "started.json",
        {
            "created_utc": io.utc(),
            "bindings": bindings,
            "specs": specs,
            "supporting_specs": secondary,
            "environment": {
                "python": sys.version,
                "numpy": np.__version__,
                "platform": platform.platform(),
            },
            "reader_basis": assembly["verification_basis"],
            "new_model_calls": 0,
            "parts_scored": [3],
        },
    )
    results = {}
    checks = {}
    omissions = {}
    descriptions = {}
    sources = {}
    supporting = {}
    batches = io.read(reader.OUT / "batches.json")
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
                c = rates.cohort(
                    roster,
                    predictions,
                    spec,
                    strict=strict,
                    representatives=reps,
                    excluded=excluded,
                )
                r, draws = rates.summarize(c, spec)
                results[name] = r
                save_draws(name, draws)
                integer = integer_draw_check(roster, predictions, spec, c, draws)
                need(
                    all(
                        abs(x - y) < 1e-12
                        for x, y in zip(integer["exact_interval"], r["family_interval"])
                    ),
                    "Exact family tail differs",
                )
                checks[name] = {
                    "exact": prior.exact_check(
                        roster, predictions, spec, r, strict, reps, excluded
                    ),
                    "integer_bootstrap": integer,
                    "expanded_bootstrap": support.independent_rate_draws(
                        roster, predictions, spec, c, draws
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
                    d = describe(roster, predictions, spec, r)
                    jsonlines(
                        OUT / "per_text" / (spec["id"].replace("/", "__") + ".jsonl"),
                        d.pop("per_text"),
                    )
                    descriptions[spec["id"]] = d
                print(
                    "Mechanism checked: " + name + "; n=" + str(r["common_n"]),
                    flush=True,
                )
        eligible = [r for r in roster if r["key"] not in excluded]
        for spec in specs:
            name = cohort_name + "/" + spec["id"]
            c = rates.cohort(roster, predictions, spec, excluded=excluded)
            ss = {
                **spec,
                "confidence": 0.95,
                "seed": support.seed_for("source_stratified/" + name),
            }
            r, draws = support.source_bootstrap(c, roster, ss)
            ident = name + "/stratified"
            sources[ident] = r
            save_draws(ident, draws)
            integer = integer_draw_check(
                roster, predictions, ss, c, draws, stratified=True
            )
            need(
                all(
                    abs(x - y) < 1e-12
                    for x, y in zip(integer["exact_interval"], r["interval95"])
                ),
                "Stratified tails differ",
            )
            checks[ident] = {"integer_bootstrap": integer}
            for source in sorted({r["source"] for r in roster}):
                subset = [r for r in eligible if r["source"] == source]
                ss = {
                    **spec,
                    "confidence": 0.95,
                    "seed": support.seed_for("source_specific/" + name + "/" + source),
                }
                c = rates.cohort(subset, predictions, ss)
                r, draws = rates.summarize(c, ss)
                r["interpretation"] = (
                    "Supporting unadjusted source-specific 95% interval; not mechanism-family discovery."
                )
                ident = name + "/source/" + source
                sources[ident] = r
                save_draws(ident, draws)
                checks[ident] = {
                    "exact": prior.exact_check(subset, predictions, ss, r),
                    "integer_bootstrap": integer_draw_check(
                        subset, predictions, ss, c, draws
                    ),
                }
        for spec in secondary:
            name = cohort_name + "/" + spec["id"]
            c = rates.cohort(roster, predictions, spec, excluded=excluded)
            r, draws = rates.summarize(c, spec)
            supporting[name] = r
            save_draws(name, draws)
            checks[name] = {
                "exact": prior.exact_check(
                    roster, predictions, spec, r, excluded=excluded
                ),
                "integer_bootstrap": integer_draw_check(
                    roster, predictions, spec, c, draws
                ),
            }
        print("Supporting/source controls checked: " + cohort_name, flush=True)
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
    validity = {
        condition: dict(
            Counter(
                str(v["n_valid"])
                for (k, c, r), v in predictions.items()
                if c == condition
            )
        )
        for condition in conditions_for(science)
    }
    for filename, value in [
        ("mechanism_and_sensitivities", results),
        ("independent_checks", checks),
        ("supporting_reference_controls", supporting),
        ("source_effects", sources),
        ("omission_points", omissions),
        ("descriptive_counts", descriptions),
        ("nulls", nulls),
        ("validity_counts", validity),
        (
            "common_context_decomposition",
            common_context_decomposition(roster, predictions, science),
        ),
    ]:
        write(OUT / (filename + ".json"), value)
    io.verify(io.ROOT, bindings)
    artifacts = {io.rel(p): io.sha(p) for p in OUT.rglob("*") if p.is_file()}
    write(
        OUT / "statistics_complete.json",
        {
            "status": "part3_statistics_checked_context_and_interpretation_pending",
            "completed_utc": io.utc(),
            "artifacts": artifacts,
            "bindings": bindings,
            "predictions": len(predictions),
            "nulls": len(nulls),
            "new_model_calls": 0,
            "integer_draws_checked": sum(
                c["integer_bootstrap"]["draws_checked"] for c in checks.values()
            ),
            "additionally_expanded_draws_checked": sum(
                c.get("expanded_bootstrap", {}).get("draws_checked", 0)
                for c in checks.values()
            ),
            "remaining_parts": list(range(4, 11)),
        },
    )
    for name, r in results.items():
        if name.startswith("full/ordinary_valid/"):
            print(
                json.dumps(
                    {
                        k: r[k]
                        for k in (
                            "contrast",
                            "common_n",
                            "treatment_rate",
                            "control_rate",
                            "difference",
                            "family_interval",
                            "missing_difference_bounds",
                        )
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    with io.offline():
        main()

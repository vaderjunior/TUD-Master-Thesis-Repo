from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import math
import numpy as np
from analysis.feedback.reserved import support as support
from analysis.feedback.reserved import quality_independent as independent


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sign(value):
    if value is None:
        return "not_estimable"
    value = Fraction(value)
    return "negative" if value < 0 else "positive" if value > 0 else "zero"


def interval_state(values):
    if values is None:
        return "not_estimable"
    lo, hi = map(Fraction, values)
    need(lo <= hi, "Reversed interval")
    return "negative" if hi < 0 else "positive" if lo > 0 else "includes_zero"


def selected(rows, excluded=(), representatives=False):
    excluded = set(excluded)
    need(excluded <= {r["key"] for r in rows}, "Unknown exclusion")
    chosen = [r for r in rows if r["key"] not in excluded]
    if representatives:
        groups = defaultdict(list)
        for r in chosen:
            groups[r["gold"]["hate"], r["cluster"]].append(r["key"])
        keys = {
            min(
                g,
                key=lambda k: hashlib.sha256(
                    ("A15-representative|" + k).encode()
                ).hexdigest(),
            )
            for g in groups.values()
        }
        chosen = [r for r in chosen if r["key"] in keys]
    return chosen


def rate_point(
    rows, predictions, spec, *, strict=False, representatives=False, excluded=()
):

    rows = [
        r
        for r in selected(rows, excluded, representatives)
        if r["gold"]["hate"] == (spec["endpoint"] == "FNR")
    ]
    arms = [spec["treatment_conditions"], spec["control_conditions"]]
    need(all(arms), "Empty comparison arm")
    common = []
    errors = [0, 0]
    missing = [0, 0]
    bounds = [Fraction(0), Fraction(0)]
    for r in rows:
        values = [
            [
                (
                    None
                    if (p := predictions[r["key"], c, rep])["result"] is None
                    or (strict and p["n_valid"] != 3)
                    else int(p["result"]["hate"] != r["gold"]["hate"])
                )
                for c in arm
                for rep in range(3)
            ]
            for arm in arms
        ]
        lo = [
            Fraction(sum(v for v in arm if v is not None), len(arm)) for arm in values
        ]
        hi = [
            lo[i] + Fraction(arm.count(None), len(arm)) for i, arm in enumerate(values)
        ]
        bounds[0] += lo[0] - hi[1]
        bounds[1] += hi[0] - lo[1]
        for i in (0, 1):
            missing[i] += values[i].count(None)
        if all(None not in arm for arm in values):
            common.append(r["key"])
            for i in (0, 1):
                errors[i] += sum(values[i])
    rates = [
        Fraction(errors[i], len(common) * 3 * len(arms[i])) if common else None
        for i in (0, 1)
    ]
    diff = rates[0] - rates[1] if common else None
    return {
        "common_ids": common,
        "common_n": len(common),
        "eligible_n": len(rows),
        "real_error_counts": errors,
        "outputs_per_input": [3 * len(v) for v in arms],
        "missing_real_outputs": missing,
        "exact_rates": [str(v) if v is not None else None for v in rates],
        "difference": float(diff) if diff is not None else None,
        "difference_exact": str(diff) if diff is not None else None,
        "missing_difference_bounds": (
            [float(v / len(rows)) for v in bounds] if rows else None
        ),
        "missing_bounds_exact": [str(v / len(rows)) for v in bounds] if rows else None,
    }


def quality_point(rows, predictions, spec, fine=False):

    eligible = [r for r in rows if not fine or r["gold"]["hate_type"] is not None]
    arms = [spec["treatment_conditions"], spec["control_conditions"]]
    common = support.paired_roster(eligible, predictions, arms)
    checked = independent.paired_point(rows, predictions, spec, fine)
    need(
        checked["common_ids"] == [r["key"] for r in common],
        "Quality diagnostic common mask differs",
    )
    names = support.METRICS[2:] if fine else support.METRICS[:2]
    if common:
        arrays = [
            support._sufficient(common, predictions, arm, fine=fine) for arm in arms
        ]
        floats = [support._metrics_from_counts(arr.sum(axis=0), fine) for arr in arrays]
        exacts = [
            independent.fractional_scores(
                independent.indicators(common, predictions, arm, fine).sum(axis=0), fine
            )
            for arm in arms
        ]
    else:
        floats = [[None, None], [None, None]]
        exacts = [[None, None], [None, None]]
    metrics = {}
    for i, name in enumerate(names):
        point = {}
        exact_point = {}
        for j, arm in enumerate(("treatment", "control")):
            value = exacts[j][i]
            f = floats[j][i]
            need(
                (
                    (f is None or not math.isfinite(f))
                    if value is None
                    else f is not None and abs(float(value) - f) < 1e-12
                ),
                "Quality diagnostic metric differs",
            )
            point[arm] = float(value) if value is not None else None
            exact_point[arm] = str(value) if value is not None else None
        diff = (
            exacts[0][i] - exacts[1][i]
            if None not in (exacts[0][i], exacts[1][i])
            else None
        )
        point["difference"] = float(diff) if diff is not None else None
        exact_point["difference"] = str(diff) if diff is not None else None
        need(
            all(
                (
                    checked["metrics"][name][k] is None
                    if v is None
                    else abs(checked["metrics"][name][k] - v) < 1e-12
                )
                for k, v in point.items()
            ),
            "Independent exact quality point differs",
        )
        metrics[name] = {**point, "exact": exact_point}
    return {
        "eligible_n": len(eligible),
        "common_n": len(common),
        "common_ids": [r["key"] for r in common],
        "gold_polarities": dict(Counter(str(r["gold"]["hate"]) for r in common)),
        "metrics": metrics,
        "status": "descriptive_point_no_new_interval",
        "single_class_binary_slice": not fine
        and len({r["gold"]["hate"] for r in common}) == 1,
    }


def omit_pool(spec, seed):
    arms = {
        k: [c for c in spec[k] if not c.endswith("_s" + str(seed))]
        for k in ("treatment_conditions", "control_conditions")
    }
    if not all(arms.values()) or all(arms[k] == spec[k] for k in arms):
        return None
    return {**spec, **arms}


def spread(values):
    values = list(values)
    finite = [v for v in values if v is not None]
    return {
        "n": len(values),
        "estimable": len(finite),
        "min": min(finite) if finite else None,
        "max": max(finite) if finite else None,
        "directions": dict(Counter(sign(v) for v in values)),
    }

from collections import Counter
from fractions import Fraction
import math
import numpy as np

LABELS = (
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def usable(p, strict=False):
    return p["result"] is not None and (not strict or p["n_valid"] == 3)


def bitset(values):
    return sum(1 << LABELS.index(x) for x in set(values))


def indicators(rows, predictions, conditions, fine=False):

    width = 23 if fine else 4
    out = np.zeros((len(rows), 3 * len(conditions), width), dtype=np.float64)
    for i, row in enumerate(rows):
        for j, (c, rep) in enumerate((c, rep) for c in conditions for rep in range(3)):
            p = predictions[row["key"], c, rep]["result"]
            if fine:
                g = bitset(row["gold"]["hate_type"])
                v = bitset(p["hate_type"])
                for h in range(7):
                    mask = 1 << h
                    gb = bool(g & mask)
                    vb = bool(v & mask)
                    out[i, j, 3 * h : 3 * h + 3] = [
                        gb and vb,
                        not gb and vb,
                        gb and not vb,
                    ]
                out[i, j, 21:] = [g == v, 1]
            else:
                code = 2 * int(row["gold"]["hate"]) + int(p["hate"])
                out[i, j, {0: 1, 1: 2, 2: 3, 3: 0}[code]] = 1
    return out


def fractional_scores(counts, fine):
    values = []
    for row in counts:
        row = [int(x) for x in row]

        def f1(tp, fp, fn):
            return (
                Fraction(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else Fraction(0)
            )

        if fine:
            values.append(
                [
                    sum(f1(*row[h * 3 : h * 3 + 3]) for h in range(7)) / 7,
                    Fraction(row[21], row[22]) if row[22] else None,
                ]
            )
        else:
            tp, tn, fp, fn = row
            values.append(
                [
                    (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2,
                    (
                        (Fraction(tp, tp + fn) + Fraction(tn, tn + fp)) / 2
                        if tp + fn and tn + fp
                        else None
                    ),
                ]
            )
    return [
        (
            sum(v[i] for v in values) / len(values)
            if all(v[i] is not None for v in values)
            else None
        )
        for i in range(2)
    ]


def float_scores(counts, fine):

    answer = np.zeros((len(counts), 2))
    for j in range(counts.shape[1]):
        row = counts[:, j, :]
        if fine:
            for h in range(7):
                tp, fp, fn = row[:, 3 * h : 3 * h + 3].T
                den = 2 * tp + fp + fn
                answer[:, 0] += (
                    np.divide(2 * tp, den, out=np.zeros(len(row)), where=den != 0) / 7
                )
            answer[:, 1] += row[:, 21] / row[:, 22]
        else:
            tp, tn, fp, fn = row.T
            for correct in (tp, tn):
                den = 2 * correct + fp + fn
                answer[:, 0] += np.divide(
                    correct, den, out=np.zeros(len(row)), where=den != 0
                )
            with np.errstate(invalid="ignore", divide="ignore"):
                answer[:, 1] += (tp / (tp + fn) + tn / (tn + fp)) / 2
    return answer / counts.shape[1]


def run_check(rows, predictions, condition, repeat, result, strict=False):
    counts = Counter()
    per = {l: Counter() for l in LABELS}
    card = Counter()
    for row in rows:
        p = predictions[row["key"], condition, repeat]
        counts["eligible"] += 1
        counts["valid_votes_" + str(p["n_valid"])] += 1
        counts["nulls"] += p["result"] is None
        if not usable(p, strict):
            counts["unusable"] += 1
            continue
        p = p["result"]
        counts["valid"] += 1
        counts[
            ["tn", "fp", "fn", "tp"][2 * int(row["gold"]["hate"]) + int(p["hate"])]
        ] += 1
        if row["gold"]["hate_type"] is None:
            continue
        g = bitset(row["gold"]["hate_type"])
        v = bitset(p["hate_type"])
        both = g & v
        counts["type_valid"] += 1
        counts["type_exact"] += g == v
        counts["type_contains_gold"] += both == g
        counts["type_extra_labels"] += (v ^ both).bit_count()
        counts["type_missing_labels"] += (g ^ both).bit_count()
        counts["type_predicted_labels"] += v.bit_count()
        counts["type_gold_labels"] += g.bit_count()
        card["gold_" + str(g.bit_count())] += 1
        card["prediction_" + str(v.bit_count())] += 1
        for h, l in enumerate(LABELS):
            mask = 1 << h
            gb = bool(g & mask)
            vb = bool(v & mask)
            per[l].update(
                {
                    "support": int(gb),
                    "predicted": int(vb),
                    "tp": int(gb and vb),
                    "fp": int(not gb and vb),
                    "fn": int(gb and not vb),
                }
            )
    need(dict(counts) == result["counts"], "Independent run count mismatch")
    need(dict(card) == result["cardinality"], "Independent cardinalities mismatch")

    def divide(a, b):
        return Fraction(a, b) if b else Fraction(0)

    metrics = {}
    if counts["valid"]:
        tp, tn, fp, fn = (counts[k] for k in ("tp", "tn", "fp", "fn"))
        metrics["binary_macro_f1"] = (
            divide(2 * tp, 2 * tp + fp + fn) + divide(2 * tn, 2 * tn + fp + fn)
        ) / 2
        metrics["balanced_accuracy"] = (
            (divide(tp, tp + fn) + divide(tn, tn + fp)) / 2
            if tp + fn and tn + fp
            else None
        )
        metrics["FPR"] = Fraction(fp, fp + tn) if fp + tn else None
        metrics["FNR"] = Fraction(fn, fn + tp) if fn + tp else None
    else:
        metrics.update(
            {k: None for k in ("binary_macro_f1", "balanced_accuracy", "FPR", "FNR")}
        )
    for l, c in per.items():
        for k, v in c.items():
            need(
                v == result["per_label"][l].get(k, 0),
                "Independent per-label count mismatch",
            )
        for k, value in [
            ("f1", divide(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"])),
            ("precision", divide(c["tp"], c["tp"] + c["fp"])),
            ("recall", divide(c["tp"], c["tp"] + c["fn"])),
        ]:
            need(
                abs(float(value) - result["per_label"][l][k]) < 1e-14,
                "Independent label score mismatch",
            )
    n = counts["type_valid"]
    metrics["type_macro_f1_fixed7"] = (
        sum(divide(2 * c["tp"], 2 * c["tp"] + c["fp"] + c["fn"]) for c in per.values())
        / 7
        if n
        else None
    )
    for name, key in [
        ("type_exact_set", "type_exact"),
        ("type_contains_gold", "type_contains_gold"),
        ("type_mean_cardinality", "type_predicted_labels"),
        ("type_gold_mean_cardinality", "type_gold_labels"),
    ]:
        metrics[name] = Fraction(counts[key], n) if n else None
    for name, v in metrics.items():
        target = result["metrics"][name]
        need(
            (
                target is None
                if v is None
                else target is not None and abs(float(v) - target) < 1e-14
            ),
            "Independent metric mismatch " + name,
        )
    return {
        "metrics_exact": {
            k: str(v) if v is not None else None for k, v in metrics.items()
        },
        "valid": counts["valid"],
        "type_valid": n,
    }


def paired_point(rows, predictions, spec, fine=False, strict=False):
    eligible = [r for r in rows if not fine or r["gold"]["hate_type"] is not None]
    arms = [spec["treatment_conditions"], spec["control_conditions"]]
    common = [
        r
        for r in eligible
        if all(
            usable(predictions[r["key"], c, rep], strict)
            for arm in arms
            for c in arm
            for rep in range(3)
        )
    ]
    metrics = (
        ("type_macro_f1_fixed7", "type_exact_set")
        if fine
        else ("binary_macro_f1", "balanced_accuracy")
    )
    result = {
        "original_n": len(eligible),
        "common_n": len(common),
        "common_ids": [r["key"] for r in common],
        "metrics": {},
    }
    if not common:
        result["metrics"] = {
            m: {"treatment": None, "control": None, "difference": None} for m in metrics
        }
        return result
    points = [
        fractional_scores(indicators(common, predictions, arm, fine).sum(axis=0), fine)
        for arm in arms
    ]
    for i, m in enumerate(metrics):
        t, c = points[0][i], points[1][i]
        result["metrics"][m] = {
            k: float(v) if v is not None else None
            for k, v in [
                ("treatment", t),
                ("control", c),
                ("difference", t - c if t is not None and c is not None else None),
            ]
        }
        result["metrics"][m]["exact_fractions"] = [
            str(v) if v is not None else None
            for v in (t, c, t - c if t is not None and c is not None else None)
        ]
    return result


def bootstrap_check(rows, predictions, spec, result, distribution, fine=False):
    point = paired_point(rows, predictions, spec, fine)
    need(point["common_ids"] == result["common_ids"], "Independent paired mask differs")
    if not point["common_n"]:
        need(distribution.shape == (0, 2), "Empty cohort has draws")
        return {
            "draws_checked": 0,
            "values_checked": 0,
            "point": point,
            "max_absolute_error": 0,
        }
    for metric, p in point["metrics"].items():
        for field in ("treatment", "control", "difference"):
            target = result["metrics"][metric][field]
            value = p[field]
            need(
                (
                    target is None
                    if value is None
                    else target is not None and abs(target - value) < 1e-13
                ),
                "Independent fractional point differs",
            )
    keyset = set(point["common_ids"])
    common = [r for r in rows if r["key"] in keyset]
    clusters = sorted({r["cluster"] for r in common})
    lookup = {c: i for i, c in enumerate(clusters)}
    row_cluster = np.array([lookup[r["cluster"]] for r in common])
    arms = [spec["treatment_conditions"], spec["control_conditions"]]
    inputs = [indicators(common, predictions, arm, fine) for arm in arms]
    rng = np.random.Generator(np.random.PCG64(spec["seed"]))
    calculated = np.empty_like(distribution)

    for start in range(0, spec["draws"], 79):
        length = min(79, spec["draws"] - start)
        chosen = rng.integers(0, len(clusters), (length, len(clusters)))
        weights = np.array([np.bincount(v, minlength=len(clusters)) for v in chosen])[
            :, row_cluster
        ].astype(np.float64)
        ct = [
            (weights @ arr.reshape(len(common), -1)).reshape(length, *arr.shape[1:])
            for arr in inputs
        ]
        need(
            all(np.array_equal(v, np.rint(v)) for v in ct),
            "Noninteger expanded text counts",
        )
        calculated[start : start + length] = float_scores(ct[0], fine) - float_scores(
            ct[1], fine
        )
    need(
        np.array_equal(np.isfinite(calculated), np.isfinite(distribution)),
        "Undefined bootstrap draw mismatch",
    )
    finite = np.isfinite(calculated)
    error = (
        float(np.max(np.abs(calculated[finite] - distribution[finite])))
        if finite.any()
        else 0
    )
    need(error < 1e-12, "Independent expanded bootstrap differs")
    metrics = list(result["metrics"])
    tails = {}
    wanted = set()
    windows = {}
    for col, metric in enumerate(metrics):
        if not np.isfinite(calculated[:, col]).all():
            need(
                result["metrics"][metric]["interval95"] is None,
                "Undefined draws received interval",
            )
            continue
        ordered = np.sort(calculated[:, col])
        windows[col] = []
        for q in (Fraction(1, 40), Fraction(39, 40)):
            position = (spec["draws"] - 1) * q
            lo = position.numerator // position.denominator
            hi = math.ceil(position)
            lower, upper = ordered[lo] - 1e-11, ordered[hi] + 1e-11
            candidates = np.flatnonzero(
                (calculated[:, col] >= lower) & (calculated[:, col] <= upper)
            ).tolist()
            below = int(np.count_nonzero(calculated[:, col] < lower))
            wanted.update(candidates)
            windows[col].append((q, lo, hi, position - lo, candidates, below))

    exact = {}
    rng = np.random.Generator(np.random.PCG64(spec["seed"]))
    for start in range(0, spec["draws"], 79):
        length = min(79, spec["draws"] - start)
        chosen = rng.integers(0, len(clusters), (length, len(clusters)))
        selected = [i for i in range(length) if start + i in wanted]
        if not selected:
            continue
        weights = np.array(
            [np.bincount(chosen[i], minlength=len(clusters)) for i in selected]
        )[:, row_cluster].astype(np.float64)
        ct = [
            (weights @ arr.reshape(len(common), -1)).reshape(
                len(selected), *arr.shape[1:]
            )
            for arr in inputs
        ]
        for j, i in enumerate(selected):
            left, right = [fractional_scores(v[j], fine) for v in ct]
            exact[start + i] = [
                None if t is None or c is None else t - c for t, c in zip(left, right)
            ]
    for col, ws in windows.items():
        values = []
        rows_out = []
        for q, lo, hi, weight, candidates, below in ws:
            ordered = sorted(exact[i][col] for i in candidates)
            left, right = ordered[lo - below], ordered[hi - below]
            value = left * (1 - weight) + right * weight
            values.append(float(value))
            rows_out.append(
                {
                    "q": str(q),
                    "candidate_draws": len(candidates),
                    "ranks": [lo, hi],
                    "adjacent_exact": [str(left), str(right)],
                    "exact_quantile": str(value),
                    "value": float(value),
                }
            )
        metric = metrics[col]
        need(
            np.max(np.abs(np.array(values) - result["metrics"][metric]["interval95"]))
            < 1e-12,
            "Rational quantile differs",
        )
        tails[metric] = {"exact_interval": values, "tails": rows_out}
    return {
        "draws_checked": spec["draws"],
        "values_checked": int(distribution.size),
        "point": point,
        "max_absolute_error": error,
        "undefined_draws": {
            m: int(np.count_nonzero(~np.isfinite(calculated[:, i])))
            for i, m in enumerate(metrics)
        },
        "quantiles": tails,
        "fractional_tail_draws_checked": len(wanted),
        "method": "Independent binary-code/bitset counts; whole-cluster multiplicities expanded to original texts; per-output scores; exact Fraction points and quantile neighbourhoods",
    }


def bounds_check(rows, predictions, arms, result, fine=False, strict=False):

    eligible = [r for r in rows if not fine or r["gold"]["hate_type"] is not None]
    points = []
    missing = []
    for arm in arms:
        miss = sum(
            not usable(predictions[r["key"], c, rep], strict)
            for r in eligible
            for c in arm
            for rep in range(3)
        )
        missing.append(miss)
        endpoints = []
        for best in (False, True):
            patched = {}
            for r in eligible:
                for c in arm:
                    for rep in range(3):
                        p = predictions[r["key"], c, rep]
                        if usable(p, strict):
                            patched[r["key"], c, rep] = p
                            continue
                        if fine:
                            g = bitset(r["gold"]["hate_type"])
                            v = g if best else g ^ 127
                            value = {
                                "hate": bool(v),
                                "hate_type": [
                                    l for h, l in enumerate(LABELS) if v & (1 << h)
                                ],
                            }
                        else:
                            value = {
                                "hate": (
                                    r["gold"]["hate"] if best else not r["gold"]["hate"]
                                ),
                                "hate_type": [],
                            }
                        patched[r["key"], c, rep] = {"result": value, "n_valid": 3}
            endpoints.append(
                fractional_scores(
                    indicators(eligible, patched, arm, fine).sum(axis=0), fine
                )
                if eligible
                else [None, None]
            )
        points.append(endpoints)
    need(
        result["eligible_n"] == len(eligible)
        and result["missing_real_outputs_by_arm"] == missing,
        "Missing-bound counts differ",
    )
    names = (
        ("type_macro_f1_fixed7", "type_exact_set")
        if fine
        else ("binary_macro_f1", "balanced_accuracy")
    )
    exact = {}
    for i, name in enumerate(names):
        t = [x[i] for x in points[0]]
        c = [x[i] for x in points[1]]
        diff = [t[0] - c[1], t[1] - c[0]] if None not in t + c else None
        exact[name] = {}
        for field, values in [
            ("treatment", t),
            ("control", c),
            ("opposing_difference", diff),
        ]:
            target = result["metrics"][name][field]
            need(
                (
                    target is None
                    if values is None
                    else len(target) == len(values)
                    and all(
                        (
                            x is None
                            if v is None
                            else x is not None and abs(x - float(v)) < 1e-13
                        )
                        for x, v in zip(target, values)
                    )
                ),
                "Independent bound score differs",
            )
            exact[name][field] = (
                [None if v is None else str(v) for v in values]
                if values is not None
                else None
            )
    return {
        "eligible_n": len(eligible),
        "missing_real_outputs_by_arm": missing,
        "exact_fractions": exact,
    }

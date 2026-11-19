from collections import Counter, defaultdict
import hashlib
import math
import numpy as np
from analysis.feedback.reserved import rates as rates

TYPES = (
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
)
METRICS = (
    "binary_macro_f1",
    "balanced_accuracy",
    "type_macro_f1_fixed7",
    "type_exact_set",
)
require = rates.require


def seed_for(name):
    return int.from_bytes(
        hashlib.sha256(("A15-support-v1|" + name).encode()).digest()[:8], "little"
    )


def ratio(n, d, empty=None):
    return n / d if d else empty


def metric_rows(roster, predictions, condition, repeat, *, strict=False):
    counts = Counter()
    label = {x: Counter() for x in TYPES}
    sets = Counter()
    for row in roster:
        record = predictions[row["key"], condition, repeat]
        counts["eligible"] += 1
        counts["valid_votes_" + str(record["n_valid"])] += 1
        counts["nulls"] += record["result"] is None
        if not rates.valid_prediction(record, strict):
            counts["unusable"] += 1
            continue
        p = record["result"]
        truth = row["gold"]["hate"]
        counts["valid"] += 1
        counts[("t" if truth == p["hate"] else "f") + ("p" if p["hate"] else "n")] += 1
        if row["gold"]["hate_type"] is None:
            continue
        gold, pred = set(row["gold"]["hate_type"]), set(p["hate_type"])
        require(gold <= set(TYPES) and pred <= set(TYPES), "Unexpected fine label")
        counts["type_valid"] += 1
        counts["type_exact"] += pred == gold
        counts["type_contains_gold"] += gold <= pred
        counts["type_extra_labels"] += len(pred - gold)
        counts["type_missing_labels"] += len(gold - pred)
        counts["type_predicted_labels"] += len(pred)
        counts["type_gold_labels"] += len(gold)
        sets["gold_" + str(len(gold))] += 1
        sets["prediction_" + str(len(pred))] += 1
        for name, c in label.items():
            c["support"] += name in gold
            c["predicted"] += name in pred
            c["tp"] += name in gold and name in pred
            c["fp"] += name not in gold and name in pred
            c["fn"] += name in gold and name not in pred
    tp, tn, fp, fn = (counts[k] for k in ("tp", "tn", "fp", "fn"))
    f1 = lambda a, b, d: ratio(2 * a, 2 * a + b + d, 0.0)
    for c in label.values():
        c["f1"] = f1(c["tp"], c["fp"], c["fn"])
        c["precision"] = ratio(c["tp"], c["tp"] + c["fp"], 0.0)
        c["recall"] = ratio(c["tp"], c["tp"] + c["fn"], 0.0)
    metrics = {
        "FPR": ratio(fp, fp + tn),
        "FNR": ratio(fn, fn + tp),
        "binary_macro_f1": (
            (f1(tp, fp, fn) + f1(tn, fn, fp)) / 2 if counts["valid"] else None
        ),
        "balanced_accuracy": (
            (tp / (tp + fn) + tn / (tn + fp)) / 2 if tp + fn and tn + fp else None
        ),
        "type_macro_f1_fixed7": (
            sum(c["f1"] for c in label.values()) / 7 if counts["type_valid"] else None
        ),
        "type_exact_set": ratio(counts["type_exact"], counts["type_valid"]),
        "type_contains_gold": ratio(counts["type_contains_gold"], counts["type_valid"]),
        "type_mean_cardinality": ratio(
            counts["type_predicted_labels"], counts["type_valid"]
        ),
        "type_gold_mean_cardinality": ratio(
            counts["type_gold_labels"], counts["type_valid"]
        ),
    }
    return {
        "condition": condition,
        "repeat": repeat,
        "strict": strict,
        "counts": dict(counts),
        "metrics": metrics,
        "per_label": {k: dict(v) for k, v in label.items()},
        "cardinality": dict(sets),
        "mask": "own-valid; not a paired comparison",
    }


def supporting_specs(science):
    groups = sorted({c.rsplit("_s", 1)[0] for c in science["conditions"] if "__" in c})
    specs = []
    for group in ["reference4", *groups]:
        members = (
            [group]
            if group == "reference4"
            else [group + "_s" + str(s) for s in science["pool_seeds"]]
        )
        for endpoint in ("FPR", "FNR"):
            name = group + "-reference5/" + endpoint
            specs.append(
                {
                    "id": "supporting/" + name,
                    "family": "supporting",
                    "endpoint": endpoint,
                    "treatment_conditions": members,
                    "control_conditions": ["reference5"],
                    "draws": science["bootstrap_draws"],
                    "confidence": 0.95,
                    "seed": seed_for(name),
                }
            )
    return specs


def selected_roster(roster, excluded=(), representatives=False):
    eligible = [r for r in roster if r["key"] not in set(excluded)]
    if not representatives:
        return eligible

    groups = defaultdict(list)
    for r in eligible:
        groups[r["gold"]["hate"], r["cluster"]].append(r)
    selected = {
        min(
            rows,
            key=lambda r: hashlib.sha256(
                ("A15-representative|" + r["key"]).encode()
            ).hexdigest(),
        )["key"]
        for rows in groups.values()
    }
    return [r for r in eligible if r["key"] in selected]


def paired_roster(roster, predictions, arms, *, strict=False):
    return [
        r
        for r in roster
        if all(
            rates.valid_prediction(predictions[r["key"], c, rep], strict)
            for arm in arms
            for c in arm
            for rep in range(3)
        )
    ]


def pool_summary(runs):
    by_condition = defaultdict(list)
    for r in runs:
        by_condition[r["condition"]].append(r)
    pools = []
    for condition, records in by_condition.items():
        require(
            {r["repeat"] for r in records} == {0, 1, 2}, "Incomplete decoder repeats"
        )
        means = {
            k: (
                sum(r["metrics"][k] for r in records) / 3
                if all(r["metrics"][k] is not None for r in records)
                else None
            )
            for k in records[0]["metrics"]
        }
        pools.append({"condition": condition, "means": means})
    grouped = defaultdict(list)
    for p in pools:
        grouped[
            (
                p["condition"].rsplit("_s", 1)[0]
                if "__" in p["condition"]
                else p["condition"]
            )
        ].append(p)
    families = []
    for group, rows in grouped.items():
        result = {}
        for metric in rows[0]["means"]:
            values = [r["means"][metric] for r in rows]
            result[metric] = {
                "pool_values": values,
                "mean": None,
                "min": None,
                "max": None,
                "sd": None,
            }
            if all(v is not None for v in values):
                result[metric].update(
                    mean=float(np.mean(values)),
                    min=min(values),
                    max=max(values),
                    sd=float(np.std(values, ddof=1)) if len(values) > 1 else None,
                )
        families.append(
            {
                "family": group,
                "pools": len(rows),
                "metrics": result,
                "interpretation": "Observed pool spread; not an interval over future correction pools",
            }
        )
    return {"pools": pools, "families": families}


def point_rate(c):
    keys = c["common_ids"]
    return (
        sum(c["errors"][k][0] - c["errors"][k][1] for k in keys) / len(keys)
        if keys
        else None
    )


def source_bootstrap(c, roster, spec):

    bykey = {r["key"]: r for r in roster}
    labels, totals, sizes = rates.cluster_totals(c)
    if not labels:
        return {"status": "not_estimable_no_common_valid_inputs"}, np.empty(0)
    strata = defaultdict(list)
    for i, label in enumerate(labels):
        sources = tuple(
            sorted(
                {
                    bykey[k]["source"]
                    for k in c["common_ids"]
                    if c["clusters"][k] == label
                }
            )
        )
        strata[sources].append(i)
    rng = np.random.Generator(np.random.PCG64(spec["seed"]))
    values = np.empty(spec["draws"])
    t = np.asarray(totals)
    n = np.asarray(sizes)
    for start in range(0, len(values), 250):
        length = min(250, len(values) - start)
        selected = [
            np.asarray(v)[rng.integers(0, len(v), (length, len(v)))]
            for _, v in sorted(strata.items())
        ]
        indices = np.concatenate(selected, axis=1)
        values[start : start + length] = (t[indices, 0] - t[indices, 1]).sum(
            axis=1
        ) / n[indices].sum(axis=1)
    return {
        "difference": point_rate(c),
        "interval95": np.quantile(values, [0.025, 0.975]).tolist(),
        "strata": [
            {"source_signature": list(k), "clusters": len(v)}
            for k, v in sorted(strata.items())
        ],
        "confidence": 0.95,
        "seed": spec["seed"],
        "draws": len(values),
        "status": "supporting",
    }, values


def independent_rate_draws(roster, predictions, spec, cohort, distribution):

    if not cohort["common_ids"]:
        require(len(distribution) == 0, "Nonempty distribution for empty cohort")
        return {"draws_checked": 0, "max_absolute_error": 0.0}
    bykey = {r["key"]: r for r in roster}
    groups = defaultdict(list)
    for k in cohort["common_ids"]:
        groups[bykey[k]["cluster"]].append(k)
    ordered = [groups[k] for k in sorted(groups)]
    errors = {}
    for k in cohort["common_ids"]:
        gold = bykey[k]["gold"]["hate"]
        arms = [
            [
                int(predictions[k, c, rep]["result"]["hate"] != gold)
                for c in cs
                for rep in range(3)
            ]
            for cs in (spec["treatment_conditions"], spec["control_conditions"])
        ]
        errors[k] = sum(arms[0]) / len(arms[0]) - sum(arms[1]) / len(arms[1])
    rng = np.random.Generator(np.random.PCG64(spec["seed"]))
    largest = 0.0
    require(len(distribution) == spec["draws"], "Bootstrap count differs")
    for draw in range(spec["draws"]):
        sampled = rng.integers(0, len(ordered), len(ordered))
        expanded = [errors[k] for i in sampled for k in ordered[i]]
        direct = math.fsum(expanded) / len(expanded)
        largest = max(largest, abs(direct - float(distribution[draw])))
    require(largest < 1e-12, "Expanded-item bootstrap disagrees")
    return {
        "draws_checked": spec["draws"],
        "max_absolute_error": largest,
        "method": "independent expanded original texts",
    }


def omission_points(
    roster,
    predictions,
    spec,
    batch_by_key,
    *,
    strict=False,
    representatives=False,
    excluded=()
):
    out = {"pool": [], "execution_batch": []}
    for seed in (2026100441, 2026100442, 2026100443):
        arms = {
            field: [c for c in spec[field] if not c.endswith("_s" + str(seed))]
            for field in ("treatment_conditions", "control_conditions")
        }
        if not all(arms.values()) or all(arms[f] == spec[f] for f in arms):
            continue
        cohort = rates.cohort(
            roster,
            predictions,
            {**spec, **arms},
            strict=strict,
            representatives=representatives,
            excluded=excluded,
        )
        out["pool"].append(
            {
                "omitted_pool_seed": seed,
                "common_n": len(cohort["common_ids"]),
                "difference": point_rate(cohort),
            }
        )
    require(
        set(batch_by_key) == {r["key"] for r in roster}, "Incomplete batch assignments"
    )
    for batch in sorted(set(batch_by_key.values())):
        keys = set(excluded) | {k for k, b in batch_by_key.items() if b == batch}
        cohort = rates.cohort(
            roster,
            predictions,
            spec,
            strict=strict,
            representatives=representatives,
            excluded=sorted(keys),
        )
        out["execution_batch"].append(
            {
                "omitted_batch": batch,
                "common_n": len(cohort["common_ids"]),
                "difference": point_rate(cohort),
            }
        )
    return out


def _sufficient(roster, predictions, conditions, *, fine):

    width = 23 if fine else 4
    result = np.zeros((len(roster), len(conditions) * 3, width), dtype=np.float64)
    for i, row in enumerate(roster):
        for j, (c, rep) in enumerate((c, r) for c in conditions for r in range(3)):
            pred = predictions[row["key"], c, rep]["result"]
            if fine:
                gold, predicted = set(row["gold"]["hate_type"]), set(pred["hate_type"])
                for h, name in enumerate(TYPES):
                    result[i, j, h * 3 : h * 3 + 3] = (
                        name in gold and name in predicted,
                        name not in gold and name in predicted,
                        name in gold and name not in predicted,
                    )
                result[i, j, 21:] = (gold == predicted, 1)
            else:
                t, p = row["gold"]["hate"], pred["hate"]
                result[i, j] = (t and p, not t and not p, not t and p, t and not p)
    return result


def _metrics_from_counts(counts, fine):
    if fine:
        labels = counts[..., :21].reshape(*counts.shape[:-1], 7, 3)
        tp, fp, fn = (labels[..., i] for i in range(3))
        den = 2 * tp + fp + fn
        f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den != 0).mean(axis=-1)
        exact = counts[..., 21] / counts[..., 22]
        return np.stack((f1, exact), axis=-1).mean(axis=-2)
    tp, tn, fp, fn = (counts[..., i] for i in range(4))
    pos, neg = 2 * tp + fp + fn, 2 * tn + fp + fn
    f1 = (
        np.divide(2 * tp, pos, out=np.zeros_like(tp), where=pos != 0)
        + np.divide(2 * tn, neg, out=np.zeros_like(tn), where=neg != 0)
    ) / 2
    with np.errstate(divide="ignore", invalid="ignore"):
        balanced = (tp / (tp + fn) + tn / (tn + fp)) / 2
    return np.stack((f1, balanced), axis=-1).mean(axis=-2)


def paired_metrics(roster, predictions, spec, *, fine=False, strict=False):
    eligible = [r for r in roster if not fine or r["gold"]["hate_type"] is not None]
    arms = [spec["treatment_conditions"], spec["control_conditions"]]
    common = paired_roster(eligible, predictions, arms, strict=strict)
    names = METRICS[2:] if fine else METRICS[:2]
    result = {
        "original_n": len(eligible),
        "common_n": len(common),
        "common_ids": [r["key"] for r in common],
        "invalid_ids": sorted(
            {r["key"] for r in eligible} - {r["key"] for r in common}
        ),
        "metrics": {},
        "confidence": 0.95,
        "seed": spec["seed"],
        "draws": spec["draws"],
        "weighting": "mean metric over real repeats within pool, then mean over pools",
    }
    if not common:
        result["status"] = "not_estimable_no_common_valid_inputs"
        return result, np.empty((0, 2))
    arrays = [_sufficient(common, predictions, arm, fine=fine) for arm in arms]
    labels = sorted({r["cluster"] for r in common})
    loc = {label: i for i, label in enumerate(labels)}
    totals = [np.zeros((len(labels), *arr.shape[1:])) for arr in arrays]
    for arr, target in zip(arrays, totals):
        for row, data in zip(common, arr):
            target[loc[row["cluster"]]] += data
    point = [_metrics_from_counts(arr.sum(axis=0), fine) for arr in arrays]
    rng = np.random.Generator(np.random.PCG64(spec["seed"]))
    values = np.empty((spec["draws"], 2))
    for start in range(0, len(values), 64):
        length = min(64, len(values) - start)
        chosen = rng.integers(0, len(labels), (length, len(labels)))
        weights = np.zeros((length, len(labels)))
        np.add.at(weights, (np.arange(length)[:, None], chosen), 1)
        sampled = [
            (weights @ arr.reshape(len(labels), -1)).reshape(length, *arr.shape[1:])
            for arr in totals
        ]
        values[start : start + length] = _metrics_from_counts(
            sampled[0], fine
        ) - _metrics_from_counts(sampled[1], fine)
    result["status"] = "supporting"
    for i, name in enumerate(names):
        valid_draws = values[:, i][np.isfinite(values[:, i])]

        result["metrics"][name] = {
            "treatment": float(point[0][i]) if np.isfinite(point[0][i]) else None,
            "control": float(point[1][i]) if np.isfinite(point[1][i]) else None,
            "difference": (
                float(point[0][i] - point[1][i])
                if np.isfinite(point[0][i] - point[1][i])
                else None
            ),
            "undefined_draws": len(values) - len(valid_draws),
            "interval95": (
                np.quantile(valid_draws, [0.025, 0.975]).tolist()
                if len(valid_draws) == len(values)
                else None
            ),
        }
    return result, values


def transitions(roster, predictions, treatment, control, prompt_meta, *, strict=False):
    groups = defaultdict(Counter)
    cases = []
    for r in roster:
        key = r["key"]
        tm = prompt_meta[key, treatment]
        cm = prompt_meta[key, control]
        identical = tm["messages_sha256"] == cm["messages_sha256"]
        for rep in range(3):
            t, b = predictions[key, treatment, rep], predictions[key, control, rep]
            base = {
                "key": key,
                "repeat": rep,
                "treatment": treatment,
                "control": control,
                "identical_prompt": identical,
            }
            labels = [
                "all",
                "identical" if identical else "changed",
                "source:" + r["source"],
                "selected_gate:" + str(tm.get("source_gate")),
                "delivered:" + str(bool(tm.get("delivered_correction_ids"))),
            ]
            cosine = tm.get("selected_correction_cosine")
            labels.append(
                "cosine:"
                + (
                    "none"
                    if cosine is None
                    else (
                        "<0.4"
                        if cosine < 0.4
                        else "0.4_to_0.6" if cosine < 0.6 else ">=0.6"
                    )
                )
            )
            labels.append(
                "removed_fifth_gate:"
                + str(tm.get("removed_fifth", {}).get("meta", {}).get("gate"))
            )
            for name in labels:
                groups[name]["eligible_pairs"] += 1
            if not all(rates.valid_prediction(x, strict) for x in (t, b)):
                for name in labels:
                    groups[name]["missing_pairs"] += 1
                continue
            truth = r["gold"]["hate"]
            t, b = t["result"], b["result"]
            change = (
                ("fix" if t["hate"] == truth else "regression")
                if t["hate"] != b["hate"]
                else "unchanged_gate"
            )
            for name in labels:
                groups[name]["valid_pairs"] += 1
                groups[name][change] += 1
                groups[name][("benign_" if not truth else "hateful_") + change] += 1
            fine = None
            if r["gold"]["hate_type"] is not None:
                g, a, d = (
                    set(r["gold"]["hate_type"]),
                    set(t["hate_type"]),
                    set(b["hate_type"]),
                )
                fine = {
                    "correct_labels_added": sorted((a - d) & g),
                    "correct_labels_lost": sorted((d - a) & g),
                    "false_labels_added": sorted((a - d) - g),
                    "false_labels_removed": sorted((d - a) - g),
                    "both_gates_positive": t["hate"] and b["hate"],
                }
                for name in labels:
                    for field in (
                        "correct_labels_added",
                        "correct_labels_lost",
                        "false_labels_added",
                        "false_labels_removed",
                    ):
                        groups[name][field] += len(fine[field])
            if change != "unchanged_gate" or (
                fine and any(fine[k] for k in fine if k != "both_gates_positive")
            ):
                cases.append({**base, "change": change, "type_changes": fine})
    return {
        "groups": {k: dict(v) for k, v in groups.items()},
        "cases": cases,
        "unit": "dependent matched-repeat pairs; descriptive, not independent texts",
    }

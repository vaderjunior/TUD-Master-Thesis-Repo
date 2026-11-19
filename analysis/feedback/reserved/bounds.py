from collections import Counter
from analysis.feedback.reserved import rates as rates
from analysis.feedback.reserved.support import TYPES
from analysis.feedback.reserved.support import ratio


def metric_bounds(roster, predictions, arms, *, fine=False, strict=False):
    eligible = [r for r in roster if not fine or r["gold"]["hate_type"] is not None]
    metrics = (
        ("type_macro_f1_fixed7", "type_exact_set")
        if fine
        else ("binary_macro_f1", "balanced_accuracy")
    )
    all_bounds = []
    missing_counts = []
    for arm in arms:
        outputs = []
        missing = 0
        for condition in arm:
            for repeat in range(3):
                endpoints = []
                for best in (False, True):
                    counts = Counter()
                    labels = {name: Counter() for name in TYPES}
                    for row in eligible:
                        record = predictions[row["key"], condition, repeat]
                        valid = rates.valid_prediction(record, strict)
                        if not best:
                            missing += not valid
                        if fine:
                            gold = set(row["gold"]["hate_type"])
                            pred = (
                                set(record["result"]["hate_type"])
                                if valid
                                else gold if best else set(TYPES) - gold
                            )
                            counts["exact"] += gold == pred
                            for name, c in labels.items():
                                c["tp"] += name in gold and name in pred
                                c["fp"] += name not in gold and name in pred
                                c["fn"] += name in gold and name not in pred
                        else:
                            gold = row["gold"]["hate"]
                            pred = (
                                record["result"]["hate"]
                                if valid
                                else gold if best else not gold
                            )
                            counts[
                                ("t" if gold == pred else "f") + ("p" if pred else "n")
                            ] += 1
                    if not eligible:
                        endpoints.append([None, None])
                    elif fine:
                        endpoints.append(
                            [
                                sum(
                                    ratio(
                                        2 * c["tp"],
                                        2 * c["tp"] + c["fp"] + c["fn"],
                                        0.0,
                                    )
                                    for c in labels.values()
                                )
                                / 7,
                                counts["exact"] / len(eligible),
                            ]
                        )
                    else:
                        tp, tn, fp, fn = (counts[k] for k in ("tp", "tn", "fp", "fn"))
                        f1 = (
                            ratio(2 * tp, 2 * tp + fp + fn, 0.0)
                            + ratio(2 * tn, 2 * tn + fp + fn, 0.0)
                        ) / 2
                        balanced = (
                            (tp / (tp + fn) + tn / (tn + fp)) / 2
                            if tp + fn and tn + fp
                            else None
                        )
                        endpoints.append([f1, balanced])
                outputs.append(endpoints)
        all_bounds.append(
            [
                [
                    (
                        sum(x[b][m] for x in outputs) / len(outputs)
                        if all(x[b][m] is not None for x in outputs)
                        else None
                    )
                    for b in (0, 1)
                ]
                for m in (0, 1)
            ]
        )
        missing_counts.append(missing)
    result = {}
    for m, name in enumerate(metrics):
        t, c = all_bounds[0][m], all_bounds[1][m]
        result[name] = {
            "treatment": t,
            "control": c,
            "opposing_difference": (
                [t[0] - c[1], t[1] - c[0]] if None not in t + c else None
            ),
        }
    return {
        "eligible_n": len(eligible),
        "missing_real_outputs_by_arm": missing_counts,
        "strict_all_three_votes": strict,
        "metrics": result,
        "interpretation": "full-roster identification bounds, not sampling intervals",
    }

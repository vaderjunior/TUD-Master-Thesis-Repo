"""Gold masks, fixed-label F1 and paired item resampling."""

from collections import Counter
import math

ENDPOINTS = {
    "en_dev_eval_main": ("hate", "primary"),
    "de_legal_dev_eval": ("legal", "primary"),
    "en_dev_eval_sq3_types": ("hate_type", "secondary"),
    "en_dev_eval_targets": ("target_group", "secondary"),
    "en_dev_eval_severity": ("severity", "secondary"),
    "de_dev_eval": ("hate", "secondary"),
    "en_dev_eval_types": ("hate_type", "sensitivity"),
}
GOLD_FIELDS = {
    "target_group": "target_groups",
    "hate_type": "hate_types",
    "severity": "severity",
    "legal": "legal",
    "hate": "gate",
}


def clean(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value.tolist() if hasattr(value, "tolist") else value


def truth(row, dimension):
    gate = clean(row.get("gate"))
    if gate is not None and type(gate) is not bool:
        raise ValueError("Gold gate must be boolean or missing")
    value = clean(row.get(GOLD_FIELDS[dimension]))
    if dimension == "hate":
        return gate
    if dimension != "legal" and gate is not True:
        return None
    if value is None:
        return None
    if dimension in ("legal", "hate_type", "target_group"):
        if not isinstance(value, list):
            raise ValueError("Multilabel gold must be a list or missing")
        return sorted(set(value))
    return value


def roster(subset, rows, taxonomy):
    dimension, role = ENDPOINTS[subset]
    labels = (
        ["not_hate", "hate"]
        if dimension == "hate"
        else list(taxonomy["dimensions"][dimension]["labels"])
    )
    kept = []
    support = Counter()
    for row in rows:
        value = truth(row, dimension)
        if value is None:
            continue
        positive = (
            ["hate" if value else "not_hate"]
            if dimension == "hate"
            else [value] if dimension == "severity" else value
        )
        if not set(positive) <= set(labels):
            raise ValueError("Unknown gold label")
        support.update(positive)
        kept.append(str(row["id"]))
    fixed = labels if dimension == "hate" else [name for name in labels if support[name] >= 10]
    if role == "primary" and (not all((support[name] > 0 for name in labels)) or fixed != labels):
        raise ValueError("Primary endpoint lacks its declared supported classes")
    return {
        "subset": subset,
        "dimension": dimension,
        "role": role,
        "rows": len(rows),
        "scorable_rows": len(kept),
        "scorable_ids": kept,
        "all_labels": labels,
        "macro_labels": fixed,
        "gold_support": {name: support[name] for name in labels},
        "mask": (
            "gate present"
            if dimension == "hate"
            else (
                "legal present, independent of gate"
                if dimension == "legal"
                else "strict gold gate true and dimension annotated"
            )
        ),
    }


def bits(value, dimension, labels):
    positive = (
        {"hate" if value else "not_hate"}
        if dimension == "hate"
        else (
            {value}
            if dimension == "severity" and value is not None
            else set(value or []) if dimension != "severity" else set()
        )
    )
    if not positive <= set(labels):
        raise ValueError("Unknown prediction label")
    return [int(name in positive) for name in labels]


def contributions(rows, outputs, endpoint):
    """One-vs-rest TP/FP/FN per item; missing/invalid never becomes negative."""
    import numpy as np

    if set(outputs) != {str(r["id"]) for r in rows}:
        raise ValueError("Missing or extra completed item: do not score partial runs")
    labels = endpoint["all_labels"]
    dimension = endpoint["dimension"]
    ids = []
    gold = []
    pred = []
    valid = []
    for row in rows:
        value = truth(row, dimension)
        if value is None:
            continue
        rid = str(row["id"])
        out = outputs[rid]
        result = out["result"]
        ok = result is not None and out["n_valid"] > 0
        if (result is None) != (out["n_valid"] == 0):
            raise ValueError("Contradictory validity stamp")
        ids.append(rid)
        gold.append(bits(value, dimension, labels))
        valid.append(ok)
        pred.append(bits(result[dimension], dimension, labels) if ok else [0] * len(labels))
    if ids != endpoint["scorable_ids"]:
        raise ValueError("Gold mask changed")
    y = np.asarray(gold, dtype=np.int64).reshape(len(ids), len(labels))
    x = np.asarray(pred, dtype=np.int64).reshape(len(ids), len(labels))
    v = np.asarray(valid, dtype=bool)
    counts = np.stack([y * x, (1 - y) * x, y * (1 - x)], axis=-1)
    counts[~v] = 0
    return {
        "ids": ids,
        "gold": y,
        "counts": counts,
        "valid": v,
        "labels": labels,
        "macro_indices": [labels.index(l) for l in endpoint["macro_labels"]],
    }


def macro(counts, indices):
    import numpy as np

    counts = np.asarray(counts)
    tp, fp, fn = (counts[..., i] for i in range(3))
    denom = 2 * tp + fp + fn
    f = np.divide(2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0)
    return f[..., indices].mean(axis=-1) if indices else np.full(f.shape[:-1], np.nan)


def score(contrib):
    import numpy as np

    v = contrib["valid"]
    counts = contrib["counts"]
    idx = contrib["macro_indices"]
    y = contrib["gold"]
    base = counts.sum(axis=0)
    low = base.copy()
    high = base.copy()
    unknown = y[~v]
    low[:, 1] += (1 - unknown).sum(axis=0)
    low[:, 2] += unknown.sum(axis=0)
    high[:, 0] += unknown.sum(axis=0)
    scalar = lambda c: float(macro(c, idx)) if idx else None
    return {
        "macro_f1_valid_only": scalar(base) if v.any() else None,
        "n_scorable": len(v),
        "n_valid": int(v.sum()),
        "n_invalid": int((~v).sum()),
        "coverage": float(v.mean()) if len(v) else None,
        "invalid_inclusive_bounds": [scalar(low), scalar(high)] if len(v) else [None, None],
        "per_label_counts": base.tolist(),
        "macro_labels": [contrib["labels"][i] for i in idx],
        "bounds_note": "Unknown bits all incorrect/correct; conservative relaxation for multiclass severity. These are bounds, not imputed predictions.",
    }


def contrast(left, right, draws=10000, seed=20260915, primary=False):
    """Mean of setting-specific F1s, never pooled votes. Shared item bootstrap."""
    import numpy as np

    allc = left + right
    if not left or not right:
        raise ValueError("Both conditions required")
    if any(
        (
            x["ids"] != allc[0]["ids"]
            or x["labels"] != allc[0]["labels"]
            or x["macro_indices"] != allc[0]["macro_indices"]
            for x in allc
        )
    ):
        raise ValueError("Unpaired scoring inputs")
    common = np.logical_and.reduce([x["valid"] for x in allc])
    n = int(common.sum())
    idx = allc[0]["macro_indices"]
    lower = (
        float(
            np.mean([score(x)["invalid_inclusive_bounds"][0] for x in left])
            - np.mean([score(x)["invalid_inclusive_bounds"][1] for x in right])
        )
        if idx and len(common)
        else None
    )
    upper = (
        float(
            np.mean([score(x)["invalid_inclusive_bounds"][1] for x in left])
            - np.mean([score(x)["invalid_inclusive_bounds"][0] for x in right])
        )
        if idx and len(common)
        else None
    )
    if not n or not idx:
        return {
            "n_common_valid": n,
            "mean_difference": None,
            "ci95": None,
            "family_ci": None,
            "invalid_inclusive_difference_bounds": [lower, upper],
        }
    cc = np.stack([x["counts"][common] for x in allc])
    values = macro(cc.sum(axis=1), idx)
    difference = lambda v: float((v[: len(left), None] - v[None, len(left) :]).mean())
    point = difference(values)
    rng = np.random.default_rng(seed)
    delta = []
    for _ in range(draws):
        pick = rng.integers(0, n, n)
        v = macro(cc[:, pick].sum(axis=1), idx)
        delta.append(difference(v))
    return {
        "n_common_valid": n,
        "common_valid_ids": [rid for rid, keep in zip(allc[0]["ids"], common) if keep],
        "mean_difference": point,
        "ci95": np.percentile(delta, [2.5, 97.5]).tolist(),
        "family_ci": np.percentile(delta, [0.625, 99.375]).tolist() if primary else None,
        "family_confidence": 0.9875 if primary else None,
        "invalid_inclusive_difference_bounds": [lower, upper],
        "per_setting_f1": values.tolist(),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
    }


def contract():
    return {
        "gate_mapping": "strict",
        "minimum_full_roster_support": 10,
        "fixed_labels": "Full scorable-roster support >=10, frozen before predictions; gate both classes. Same labels in every condition/repeat and bootstrap draw, zero F1 for zero denominator.",
        "missing_annotations": "None is unannotated; [] is observed empty. Fine heads require strict gold gate true. Legal is independent of gate.",
        "invalid_outputs": "All-null votes are completed invalid items, never negative predictions or rerun. Report coverage and invalid-inclusive bounds. Transport-incomplete items prevent final endpoint scoring.",
        "primary_contrasts": [
            {"endpoint": ep, "contrast": c}
            for ep in ("en_dev_eval_main", "de_legal_dev_eval")
            for c in ("rag_full_minus_rag_small", "rag_full_minus_eligible_fewshot_mean")
        ],
        "primary_difference": "Mean of three RAG-repeat F1s minus mean of three small repeats or nine eligible-fewshot setting/repeat F1s; EN gate eligible aliases uniform sets. Pair on common-valid IDs across every setting in that contrast.",
        "uncertainty": "10000 paired item bootstrap draws, seed 20260915. Same item resample across all settings/repeats in a contrast. Pointwise 95% and four-primary-contrast Bonferroni 98.75% percentile intervals. Hold label set fixed, even if absent in a resample.",
        "descriptive_comparisons": "Zero-shot, uniform few-shot, aligned full/compact encoders, secondary/sensitivity endpoints: report complete tables and pointwise uncertainty; do not call them additional confirmatory primary tests.",
        "replicate_reporting": "Every RAG/ZS decoder repeat; every FS set x decoder-repeat value, within-set mean/sample SD and spread across three set means separately. Encoder training seeds are distinct from decoder repeats. No pooled-vote F1 or SD treating nine FS runs as independent draws of one setting.",
        "bounds": "Full-roster invalid-inclusive low/high F1 by treating unknown bits all wrong/all right, with separate condition-difference bounds. Severity uses a conservative multiclass relaxation. No gold-based imputation is called a prediction.",
        "diagnostics": "Per-label TP/FP/FN and support, gate per-source FP/FN, all-invalid/part-valid vote counts, repairs/retries, gate ties, metadata timing and observed token usage. Legal remains independent; do not call aggregate agreement fields concordance.",
        "limits": "Post-selection DEV evidence; intervals condition on observed decoder repeats, fixed three example sets and fitted encoder seeds. They do not cover all training/generation/selection variance, shared-text dependence, or prove causal retrieval effects. No F1-based stopping or threshold/KB tuning.",
    }

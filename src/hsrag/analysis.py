"""Agreement, paired bootstrap and diagnostic analysis helpers."""

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from src.hsrag.schema import SEVERITY_ORDER

DECISIVE_PATH = Path("config/decisive_guidelines.yaml")


def _alpha(units: list[list], ordinal: bool = False) -> dict:
    """Krippendorff's alpha over a list of units, each a list of coder values."""
    usable = [[v for v in u if v is not None] for u in units]
    usable = [u for u in usable if len(u) >= 2]
    if not usable:
        return {"alpha": None, "n_units": 0, "note": "no unit had 2+ values"}

    values = sorted({v for u in usable for v in u},
                    key=lambda v: SEVERITY_ORDER.get(v, v) if ordinal else str(v))
    idx = {v: i for i, v in enumerate(values)}
    k = len(values)

    o = np.zeros((k, k))
    for u in usable:
        m = len(u)
        for a in range(m):
            for b in range(m):
                if a != b:
                    o[idx[u[a]], idx[u[b]]] += 1.0 / (m - 1)

    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n < 2:
        return {"alpha": None, "n_units": len(usable), "note": "too few values"}

    if ordinal:

        delta = np.zeros((k, k))
        for c in range(k):
            for kk in range(k):
                lo, hi = min(c, kk), max(c, kk)
                s = n_c[lo:hi + 1].sum() - (n_c[c] + n_c[kk]) / 2
                delta[c, kk] = s ** 2
    else:
        delta = 1.0 - np.eye(k)

    d_o = (o * delta).sum() / n
    d_e = (np.outer(n_c, n_c) * delta).sum() / (n * (n - 1))

    if d_e == 0:
        return {"alpha": None, "n_units": len(usable), "n_values": int(n),
                "note": "perfect agreement (expected disagreement is zero, "
                        "so alpha is undefined)"}
    return {"alpha": 1.0 - d_o / d_e, "n_units": len(usable),
            "n_values": int(n), "categories": [str(v) for v in values]}


def consistency(rows: list, taxonomy_labels: dict) -> dict:
    """Krippendorff's alpha per dimension, across the stored raw runs."""
    out = {}

    out["gate"] = _alpha([[p["hate"] if p else None for p in r["run_predictions"]]
                          for r in rows])

    for field, labels in (("target_group", taxonomy_labels["target_group"]),
                          ("hate_type", taxonomy_labels["hate_type"]),
                          ("legal", taxonomy_labels["legal"])):
        units = []
        for r in rows:
            for label in labels:
                units.append([(label in p[field]) if p else None
                              for p in r["run_predictions"]])
        out[field] = _alpha(units)
        out[field]["unit"] = "(item, label) binary decision"

    out["severity"] = _alpha(
        [[p["severity"] if p and p["severity"] else None
          for p in r["run_predictions"]] for r in rows], ordinal=True)
    out["severity"]["unit"] = "item, ordinal"
    return out


def single_run_view(rows: list, run_index: int = 0) -> list:
    """Re-cast results as if n_votes had been 1 (decision Q3)."""
    out = []
    for r in rows:
        copy = dict(r)
        preds = r["run_predictions"]
        copy["result"] = preds[run_index] if run_index < len(preds) else None
        copy["uncertain"] = False
        copy["n_runs"] = 1
        copy["n_valid"] = int(copy["result"] is not None)
        out.append(copy)
    return out


def mcnemar(rows_a: list, rows_b: list, gold: dict, effective_gate,
            mapping: str = "strict") -> dict:
    """Exact McNemar on paired gate correctness."""
    from scipy.stats import binomtest

    a = {r["item_id"]: r for r in rows_a}
    b = {r["item_id"]: r for r in rows_b}
    n00 = n01 = n10 = n11 = 0

    for item_id in set(a) & set(b):
        g = gold.get(item_id)
        if g is None or a[item_id]["result"] is None or b[item_id]["result"] is None:
            continue
        truth = effective_gate(g, mapping)
        ok_a = bool(a[item_id]["result"]["hate"]) == truth
        ok_b = bool(b[item_id]["result"]["hate"]) == truth
        if ok_a and ok_b:
            n11 += 1
        elif ok_a and not ok_b:
            n10 += 1
        elif not ok_a and ok_b:
            n01 += 1
        else:
            n00 += 1

    disc = n10 + n01
    p = binomtest(n10, disc, 0.5).pvalue if disc else 1.0
    return {"both_correct": n11, "both_wrong": n00,
            "only_a_correct": n10, "only_b_correct": n01,
            "discordant": disc, "p_value": p,
            "note": ("no discordant pairs; the arms are indistinguishable "
                     "on this data" if not disc else
                     "exact binomial on discordant pairs only")}


def paired_bootstrap(rows_a: list, rows_b: list, gold: dict, score_fn,
                     b: int = 1000, seed: int = 42) -> dict:
    """Bootstrap CI for a metric delta, resampling ITEMS."""
    a = {r["item_id"]: r for r in rows_a}
    bb = {r["item_id"]: r for r in rows_b}
    ids = sorted(set(a) & set(bb) & set(gold))
    if not ids:
        return {"delta": None, "note": "no shared items"}

    obs_a = score_fn([a[i] for i in ids], gold)
    obs_b = score_fn([bb[i] for i in ids], gold)
    if obs_a is None or obs_b is None:
        return {"delta": None, "note": "metric undefined on the full sample"}

    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(b):
        pick = rng.choice(len(ids), size=len(ids), replace=True)
        sample = [ids[i] for i in pick]
        sa = score_fn([a[i] for i in sample], gold)
        sb = score_fn([bb[i] for i in sample], gold)
        if sa is not None and sb is not None:
            deltas.append(sa - sb)

    if not deltas:
        return {"delta": obs_a - obs_b, "note": "metric undefined in resamples"}

    d = np.array(deltas)
    n_a = int((d > 0).sum())
    n_tie = int((d == 0).sum())
    n_b = int((d < 0).sum())

    frac_le = (n_b + 0.5 * n_tie) / len(d)
    frac_ge = (n_a + 0.5 * n_tie) / len(d)
    p = min(1.0, 2 * min(frac_le, frac_ge))

    return {"score_a": obs_a, "score_b": obs_b, "delta": obs_a - obs_b,
            "ci_low": float(np.percentile(d, 2.5)),
            "ci_high": float(np.percentile(d, 97.5)),

            "sd": float(d.std(ddof=1)),
            "p_value": float(p), "n_items": len(ids),
            "n_resamples": len(deltas),
            "favour_a": n_a, "ties": n_tie, "favour_b": n_b}


def guideline_effect(rag_rows: list, zero_rows: list, gold: dict,
                     effective_gate, mapping: str = "strict") -> dict:
    """Does putting a decisive guideline in the prompt change the answer?"""
    decisive = yaml.safe_load(DECISIVE_PATH.read_text(encoding="utf-8"))["decisive"]
    zero = {r["item_id"]: r for r in zero_rows}
    out = {}

    for guide, implies in decisive.items():
        cells = {}
        for side, want in (("gold_hate", True), ("gold_not_hate", False)):
            sub = [r for r in rag_rows
                   if guide in r["retrieved"].get("guidelines", [])
                   and r["item_id"] in gold
                   and effective_gate(gold[r["item_id"]], mapping) is want]
            scored = [r for r in sub if r["result"] is not None]
            rag_ok = sum(1 for r in scored
                         if bool(r["result"]["hate"]) is want)

            paired = [r for r in scored if r["item_id"] in zero
                      and zero[r["item_id"]]["result"] is not None]
            zs_ok = sum(1 for r in paired
                        if bool(zero[r["item_id"]]["result"]["hate"]) is want)

            cells[side] = {
                "n": len(scored),
                "rag_correct": rag_ok,
                "rag_accuracy": rag_ok / len(scored) if scored else None,
                "n_paired": len(paired),
                "zero_shot_correct": zs_ok,
                "zero_shot_accuracy": zs_ok / len(paired) if paired else None,
                "delta": ((rag_ok / len(scored)) - (zs_ok / len(paired)))
                if scored and paired else None,
            }

        total = sum(c["n"] for c in cells.values())
        applies = cells["gold_hate" if implies else "gold_not_hate"]["n"]
        out[guide] = {
            "implies_hate": implies,
            "retrieved_on": total,
            "gold_agrees_with_implication": applies,
            "gold_contradicts": total - applies,
            "contradiction_rate": (total - applies) / total if total else None,
            **cells,
        }
    return out

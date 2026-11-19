"""Score observed annotations and report invalid-output coverage."""

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field

import pandas as pd

from src.hsrag.schema import SEVERITY_ORDER

MIN_SUPPORT = 10
ARMS = ("zero_shot", "few_shot", "rag")


def clean(val):
    """Parquet round-trip: NaN -> None, numpy array -> list."""
    if val is None:
        return None
    if isinstance(val, float) and pd.isna(val):
        return None
    if hasattr(val, "tolist"):
        return val.tolist()
    return val


def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def hx_original_class(raw_json: str) -> str | None:
    """HateXplain's own three-way label: normal | offensive | hatespeech."""
    try:
        votes = json.loads(raw_json).get("votes") or []
    except (TypeError, ValueError, AttributeError):
        return None
    return Counter(votes).most_common(1)[0][0] if votes else None


def effective_gate(row, mapping: str = "strict") -> bool:
    """Read the hate annotation under the strict or offensive-inclusive mapping."""

    if row.gate is None:
        return None
    if mapping == "strict" or row.source != "hatexplain":
        return bool(row.gate)
    return True if hx_original_class(row.raw) == "offensive" else bool(row.gate)


@dataclass
class LabelScore:
    label: str
    precision: float
    recall: float
    f1: float
    support: int
    predicted: int
    tp: int
    fp: int
    fn: int
    averaged: bool


@dataclass
class DimensionScore:
    dimension: str
    macro_f1: float | None
    micro_f1: float
    per_label: dict = field(default_factory=dict)
    labels_averaged: list = field(default_factory=list)
    labels_excluded: list = field(default_factory=list)
    n_items: int = 0
    n_unscored: int = 0
    subset_accuracy: float = 0.0
    hamming_loss: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class GateScore:
    macro_f1: float
    mapping: str
    per_class: dict = field(default_factory=dict)
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    per_source: dict = field(default_factory=dict)
    n_scored: int = 0
    n_unscored: int = 0
    n_uncertain: int = 0
    macro_f1_confident_only: float | None = None
    note: str | None = None


def score_gate(rows: list, gold: dict, mapping: str = "strict") -> GateScore:
    """Binary gate, macro-F1 over both classes, decomposed by source dataset."""
    out = GateScore(macro_f1=0.0, mapping=mapping)
    scored, confident = [], []

    for r in rows:
        g = gold.get(r["item_id"])
        if g is None:
            continue
        if r["result"] is None:
            out.n_unscored += 1
            continue
        truth = effective_gate(g, mapping)
        if truth is None:
            continue
        pred = bool(r["result"]["hate"])
        scored.append((truth, pred, g.source))
        if r.get("uncertain"):
            out.n_uncertain += 1
        else:
            confident.append((truth, pred))

    out.n_scored = len(scored)
    if not scored:
        return out

    for name, cls in (("hate", True), ("not_hate", False)):
        tp = sum(1 for t, p, _ in scored if t == cls and p == cls)
        fp = sum(1 for t, p, _ in scored if t != cls and p == cls)
        fn = sum(1 for t, p, _ in scored if t == cls and p != cls)
        out.per_class[name] = dict(zip(("precision", "recall", "f1"),
                                       _prf(tp, fp, fn)))
        out.per_class[name].update(tp=tp, fp=fp, fn=fn)

    out.macro_f1 = sum(c["f1"] for c in out.per_class.values()) / 2

    present = {t for t, _, _ in scored}
    if len(present) < 2:
        only = present.pop()
        out.macro_f1 = None
        out.note = (f"single-class subset: every gold item has gate={only}. "
                    f"Macro-F1 over both classes is undefined; recall on the "
                    f"present class is "
                    f"{out.per_class['hate' if only else 'not_hate']['recall']:.3f}.")
    out.tp = sum(1 for t, p, _ in scored if t and p)
    out.fp = sum(1 for t, p, _ in scored if not t and p)
    out.fn = sum(1 for t, p, _ in scored if t and not p)
    out.tn = sum(1 for t, p, _ in scored if not t and not p)

    for src in sorted({s for _, _, s in scored}):
        sub = [(t, p) for t, p, s in scored if s == src]
        benign = sum(1 for t, _ in sub if not t)
        hateful = sum(1 for t, _ in sub if t)
        fp = sum(1 for t, p in sub if not t and p)
        fn = sum(1 for t, p in sub if t and not p)
        out.per_source[src] = {
            "n": len(sub), "benign": benign, "hateful": hateful,
            "fp": fp, "fn": fn,
            "fp_rate": fp / benign if benign else None,
            "fn_rate": fn / hateful if hateful else None,
        }

    if confident and len(confident) < len(scored):
        f1s = []
        for cls in (True, False):
            tp = sum(1 for t, p in confident if t == cls and p == cls)
            fp = sum(1 for t, p in confident if t != cls and p == cls)
            fn = sum(1 for t, p in confident if t == cls and p != cls)
            f1s.append(_prf(tp, fp, fn)[2])
        out.macro_f1_confident_only = sum(f1s) / 2

    return out


def score_multilabel(rows: list, gold: dict, gold_col: str, pred_field: str,
                     min_support: int = MIN_SUPPORT,
                     fixed_labels: list | None = None) -> DimensionScore:
    """Multilabel dimension, scored on items whose gold for it is not None."""
    out = DimensionScore(dimension=pred_field, macro_f1=None, micro_f1=0.0)
    pairs = []

    for r in rows:
        g = gold.get(r["item_id"])
        if g is None:
            continue

        if pred_field != "legal" and not bool(g.gate):
            continue

        truth = clean(getattr(g, gold_col, None))
        if truth is None:
            continue
        if r["result"] is None:
            out.n_unscored += 1
            continue
        pairs.append((set(truth), set(r["result"][pred_field])))

    out.n_items = len(pairs)
    if not pairs:
        return out

    labels = sorted({l for t, _ in pairs for l in t} |
                    {l for _, p in pairs for l in p})
    tot_tp = tot_fp = tot_fn = 0

    for label in labels:
        tp = sum(1 for t, p in pairs if label in t and label in p)
        fp = sum(1 for t, p in pairs if label not in t and label in p)
        fn = sum(1 for t, p in pairs if label in t and label not in p)
        support = sum(1 for t, _ in pairs if label in t)
        pr, rc, f1 = _prf(tp, fp, fn)
        out.per_label[label] = asdict(LabelScore(
            label=label, precision=pr, recall=rc, f1=f1, support=support,
            predicted=tp + fp, tp=tp, fp=fp, fn=fn,
            averaged=support >= min_support))
        tot_tp, tot_fp, tot_fn = tot_tp + tp, tot_fp + fp, tot_fn + fn

    if fixed_labels is None:
        out.labels_averaged = [l for l in labels
                               if out.per_label[l]["averaged"]]
    else:
        out.labels_averaged = [l for l in fixed_labels if l in out.per_label]
    out.labels_excluded = [l for l in labels if l not in out.labels_averaged]

    if out.labels_averaged:
        out.macro_f1 = (sum(out.per_label[l]["f1"] for l in out.labels_averaged)
                        / len(out.labels_averaged))
    out.micro_f1 = _prf(tot_tp, tot_fp, tot_fn)[2]

    out.subset_accuracy = sum(1 for t, p in pairs if t == p) / len(pairs)
    out.hamming_loss = (sum(len(t ^ p) for t, p in pairs)
                        / (len(pairs) * max(len(labels), 1)))

    out.extra = {
        "mean_gold_labels": sum(len(t) for t, _ in pairs) / len(pairs),
        "mean_pred_labels": sum(len(p) for _, p in pairs) / len(pairs),
        "any_overlap_rate": sum(1 for t, p in pairs if t & p) / len(pairs),
        "gold_subset_of_pred": sum(1 for t, p in pairs if t <= p) / len(pairs),
    }
    return out


def score_severity(rows: list, gold: dict,
                   min_support: int = MIN_SUPPORT) -> DimensionScore:
    """Score observed severity classes and ordered band errors."""
    out = DimensionScore(dimension="severity", macro_f1=None, micro_f1=0.0)
    pairs = []

    for r in rows:
        g = gold.get(r["item_id"])
        if g is None or not bool(g.gate):
            continue
        truth = clean(g.severity)
        if truth is None:
            continue
        if r["result"] is None:
            out.n_unscored += 1
            continue
        pairs.append((truth, r["result"]["severity"]))

    out.n_items = len(pairs)
    if not pairs:
        return out

    labels = sorted({t for t, _ in pairs} | {p for _, p in pairs if p},
                    key=lambda l: SEVERITY_ORDER.get(l, 99))
    for label in labels:
        tp = sum(1 for t, p in pairs if t == label and p == label)
        fp = sum(1 for t, p in pairs if t != label and p == label)
        fn = sum(1 for t, p in pairs if t == label and p != label)
        support = sum(1 for t, _ in pairs if t == label)
        pr, rc, f1 = _prf(tp, fp, fn)
        out.per_label[label] = asdict(LabelScore(
            label=label, precision=pr, recall=rc, f1=f1, support=support,
            predicted=tp + fp, tp=tp, fp=fp, fn=fn,
            averaged=support >= min_support))

    out.labels_averaged = [l for l in labels if out.per_label[l]["averaged"]]
    out.labels_excluded = [l for l in labels if not out.per_label[l]["averaged"]]
    if out.labels_averaged:
        out.macro_f1 = (sum(out.per_label[l]["f1"] for l in out.labels_averaged)
                        / len(out.labels_averaged))

    exact = sum(1 for t, p in pairs if t == p)
    out.micro_f1 = exact / len(pairs)
    out.subset_accuracy = exact / len(pairs)

    ranked = [(SEVERITY_ORDER[t], SEVERITY_ORDER[p])
              for t, p in pairs if p in SEVERITY_ORDER]
    out.extra = {
        "n_predicted_null": sum(1 for _, p in pairs if p is None),
        "mean_abs_rank_error": (sum(abs(a - b) for a, b in ranked) / len(ranked)
                                if ranked else None),
        "within_one_accuracy": (sum(1 for a, b in ranked if abs(a - b) <= 1)
                                / len(ranked) if ranked else None),
    }
    return out


def calibration(rows: list, gold: dict, mapping: str = "strict") -> dict:
    """Expected calibration error on the gate, using self-consistency vote share as the confidence proxy."""
    buckets = defaultdict(lambda: {"n": 0, "correct": 0})
    for r in rows:
        g = gold.get(r["item_id"])
        if g is None or r["result"] is None:
            continue

        if effective_gate(g, mapping) is None:
            continue
        conf = r.get("agreement", {}).get("hate")
        if conf is None:
            continue
        b = buckets[round(float(conf), 3)]
        b["n"] += 1
        b["correct"] += int(bool(r["result"]["hate"]) == effective_gate(g, mapping))

    total = sum(b["n"] for b in buckets.values())
    if not total:
        return {"ece": None, "bins": {}, "n": 0}

    ece = 0.0
    bins = {}
    for conf, b in sorted(buckets.items()):
        acc = b["correct"] / b["n"]
        ece += b["n"] / total * abs(acc - conf)
        bins[conf] = {"n": b["n"], "accuracy": acc, "confidence": conf}
    return {"ece": ece, "bins": bins, "n": total,
            "note": "vote share proxy; n_votes=3 gives 2-3 distinct values"}


def honesty(rows: list) -> dict:
    """The numbers that say how much to trust everything above."""
    runs = sum(r["n_runs"] for r in rows)
    lat = [x for r in rows for x in r["latencies"]]
    return {
        "n_items": len(rows),
        "total_runs": runs,
        "parse_failures": sum(r["parse_failures"] for r in rows),
        "parse_failure_rate": sum(r["parse_failures"] for r in rows) / max(runs, 1),
        "repairs": sum(r["repairs"] for r in rows),
        "normalisations": sum(r["normalisations"] for r in rows),
        "gate_normalised": sum(r["gate_normalised"] for r in rows),
        "uncertain": sum(1 for r in rows if r["uncertain"]),
        "no_prediction": sum(1 for r in rows if r["result"] is None),
        "mean_latency_s": sum(lat) / max(len(lat), 1),
        "workers": sorted({r.get("workers", 1) for r in rows}),
        "models": dict(Counter(r["active_model"] for r in rows)),
        "kb_versions": sorted({r["kb_version"] for r in rows if r["kb_version"]}),
        "prompt_versions": sorted({r["prompt_version"] for r in rows}),
    }


def score_all(df: pd.DataFrame, results: list, gate_mapping: str = "both",
              min_support: int = MIN_SUPPORT) -> dict:
    """Score every arm present in `results` against gold in `df`."""
    gold = {str(r.id): r for r in df.itertuples(index=False)}
    by_arm = defaultdict(list)
    for r in results:
        if "_manifest" in r:
            continue
        by_arm[r["arm"]].append(r)

    mappings = (["strict", "lenient"] if gate_mapping == "both"
                else [gate_mapping])

    out = {}
    for arm, rows in by_arm.items():
        out[arm] = {
            "gate": {m: asdict(score_gate(rows, gold, m)) for m in mappings},
            "target_group": asdict(score_multilabel(
                rows, gold, "target_groups", "target_group", min_support)),
            "hate_type": asdict(score_multilabel(
                rows, gold, "hate_types", "hate_type", min_support)),
            "legal": asdict(score_multilabel(
                rows, gold, "legal", "legal", min_support)),
            "severity": asdict(score_severity(rows, gold, min_support)),
            "calibration": calibration(rows, gold, mappings[0]),
            "honesty": honesty(rows),
        }
    return out

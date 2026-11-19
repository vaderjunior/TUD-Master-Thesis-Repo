"""Common replay."""

from src.hsrag.paths import PROJECT_ROOT
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import socket
import unicodedata
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from analysis.common import scoring as original

ROOT = PROJECT_ROOT
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
BASE = ROOT / "experiments/astra/offline_closure_20261006_v1"
FIELDS = ("hate", "target_group", "hate_type", "severity", "legal")


def utc():
    return datetime.now(timezone.utc).isoformat()


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def encoded(x):
    return (
        json.dumps(x, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf8")


def signature(x):
    return hashlib.sha256(encoded(x)).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def rows(p):
    return [json.loads(x) for x in Path(p).read_text(encoding="utf8").splitlines() if x.strip()]


def save(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        f.write(encoded(x))


def save_rows(p, rr):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf8", newline="\n") as f:
        for row in rr:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def normalized(text):
    return re.sub("\\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def no_network(*args, **kwargs):
    raise AssertionError("Offline analysis: network forbidden")


def network_guard():
    return patch.object(socket.socket, "connect", no_network)


def verify(hashes):
    for name, expected in hashes.items():
        assert sha(ROOT / name) == expected, ("Changed input", name)


def clean(x):
    if x is None or x is pd.NA or (isinstance(x, float) and np.isnan(x)):
        return None
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def gold_rows(subset):
    return [
        {k: clean(v) for k, v in row.items()}
        for row in pd.read_parquet(ROOT / f"data/processed/{subset}.parquet").to_dict("records")
    ]


def decision(x):
    return (
        None if x is None else {k: sorted(x[k]) if isinstance(x[k], list) else x[k] for k in FIELDS}
    )


def independent_aggregate(slots):
    good = [x for x in slots if x is not None]
    if not good:
        return None
    n = len(good)
    out = {"hate": sum((x["hate"] for x in good)) > n / 2}
    for field in ("target_group", "hate_type", "legal"):
        count = Counter((v for x in good for v in set(x[field])))
        out[field] = sorted((k for k, v in count.items() if v > n / 2))
    sevs = sorted(
        (["low", "medium", "high"].index(x["severity"]) for x in good if x["severity"] is not None)
    )
    out["severity"] = ["low", "medium", "high"][sevs[(len(sevs) - 1) // 2]] if sevs else None
    if not out["hate"]:
        out.update(target_group=[], hate_type=[], severity=None)
    return out


def contributions(records, outputs, endpoint):
    """Use the frozen scorer and independently check bits/counts with direct sets."""
    out = {
        str(k): {"result": decision(v), "n_valid": int(v is not None)} for k, v in outputs.items()
    }
    c = original.contributions(records, out, endpoint)
    dim = endpoint["dimension"]
    labs = endpoint["all_labels"]
    ys = []
    xs = []
    ids = []
    valid = []
    for row in records:
        gate = row.get("gate")
        if dim == "hate":
            value = gate
        elif dim != "legal" and gate is not True:
            continue
        else:
            value = row.get(original.GOLD_FIELDS[dim])
        if value is None:
            continue
        positive = (
            {"hate" if value else "not_hate"}
            if dim == "hate"
            else {value} if dim == "severity" else set(value)
        )
        pred = outputs[str(row["id"])]
        values = set()
        if pred is not None:
            v = pred[dim]
            values = (
                {"hate" if v else "not_hate"}
                if dim == "hate"
                else (
                    {v}
                    if dim == "severity" and v is not None
                    else set(v or []) if dim != "severity" else set()
                )
            )
        assert positive <= set(labs) and values <= set(labs)
        ids.append(str(row["id"]))
        valid.append(pred is not None)
        ys.append([int(x in positive) for x in labs])
        xs.append([int(x in values) for x in labs])
    y = np.array(ys, dtype=np.int64)
    x = np.array(xs, dtype=np.int64)
    v = np.array(valid, dtype=bool)
    counts = np.stack([y * x, (1 - y) * x, y * (1 - x)], axis=-1)
    counts[~v] = 0
    assert (
        ids == c["ids"]
        and np.array_equal(y, c["gold"])
        and np.array_equal(v, c["valid"])
        and np.array_equal(counts, c["counts"])
    )
    c["pred"] = x
    return c


def summarize(c):
    out = original.score(c)
    idx = c["macro_indices"]
    ok = c["valid"]
    y = c["gold"]
    x = c["pred"]
    if ok.any():
        f = np.mean([f1_score(y[ok, k], x[ok, k], average="binary", zero_division=0) for k in idx])
        assert abs(f - out["macro_f1_valid_only"]) < 1e-12
        out.update(
            exact_set_accuracy=float((x[ok] == y[ok]).all(axis=1).mean()),
            mean_gold_label_count=float(y[ok].sum(axis=1).mean()),
            mean_predicted_label_count=float(x[ok].sum(axis=1).mean()),
            mean_extra_labels=float(((1 - y[ok]) * x[ok]).sum(axis=1).mean()),
            mean_missing_labels=float((y[ok] * (1 - x[ok])).sum(axis=1).mean()),
        )
    out["per_label"] = {
        lab: {
            "support": int(y[:, k].sum()),
            "valid_support": int(y[ok, k].sum()),
            "tp": int(c["counts"][:, k, 0].sum()),
            "fp": int(c["counts"][:, k, 1].sum()),
            "fn": int(c["counts"][:, k, 2].sum()),
        }
        for k, lab in enumerate(c["labels"])
    }
    return out


def bootstrap(left, right, clusters, seed=20261006, draws=10000, family_size=None):
    """Paired resampling of exact-normalized-text clusters; seeds/repeats fixed."""
    allc = left + right
    assert all(
        (
            x["ids"] == allc[0]["ids"] and x["macro_indices"] == allc[0]["macro_indices"]
            for x in allc
        )
    )
    mask = np.logical_and.reduce([x["valid"] for x in allc])
    indices = allc[0]["macro_indices"]
    kept = [clusters[k] for k in np.flatnonzero(mask)]
    unique = sorted(set(kept))
    group = {k: i for i, k in enumerate(unique)}
    assert unique and indices
    counts = np.stack([x["counts"][mask] for x in allc])
    n = counts.shape[1]
    grouped = np.zeros((len(allc), len(unique), counts.shape[2], 3), dtype=np.int64)
    for i, key in enumerate(kept):
        grouped[:, group[key]] += counts[:, i]

    def metric(total):
        return original.macro(total, indices)

    pointvalues = metric(counts.sum(axis=1))
    point = float(pointvalues[: len(left)].mean() - pointvalues[len(left) :].mean())
    values = []
    rng = np.random.default_rng(seed)
    checks = 0
    flat = grouped.reshape(len(allc), len(unique), -1)
    for start in range(0, draws, 200):
        picks = rng.integers(0, len(unique), (min(200, draws - start), len(unique)))
        weights = np.array([np.bincount(p, minlength=len(unique)) for p in picks])
        total = np.einsum("bg,sgk->bsk", weights, flat, optimize=True).reshape(
            len(picks), len(allc), counts.shape[2], 3
        )
        scores = metric(total)
        delta = scores[:, : len(left)].mean(axis=1) - scores[:, len(left) :].mean(axis=1)
        direct = grouped[:, picks[0]].sum(axis=1)
        assert np.array_equal(total[0], direct)
        checks += 1
        values.extend(delta.tolist())
    lo = np.mean([original.score(x)["invalid_inclusive_bounds"][0] for x in left]) - np.mean(
        [original.score(x)["invalid_inclusive_bounds"][1] for x in right]
    )
    hi = np.mean([original.score(x)["invalid_inclusive_bounds"][1] for x in left]) - np.mean(
        [original.score(x)["invalid_inclusive_bounds"][0] for x in right]
    )
    return {
        "n_common_valid": n,
        "n_text_clusters": len(unique),
        "mean_difference": point,
        "left_common_mean": float(pointvalues[: len(left)].mean()),
        "right_common_mean": float(pointvalues[len(left) :].mean()),
        "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        "family_interval": (
            np.quantile(values, [0.05 / (2 * family_size), 1 - 0.05 / (2 * family_size)]).tolist()
            if family_size
            else None
        ),
        "family_size": family_size,
        "full_roster_failure_difference_bounds": [float(lo), float(hi)],
        "draws": draws,
        "seed": seed,
        "direct_resample_checks": checks,
        "excluded_ids": [rid for rid, v in zip(allc[0]["ids"], mask) if not v],
    }


def means(runs):
    vals = [x["macro_f1_valid_only"] for x in runs]
    return {"values": vals, "mean": float(np.mean(vals)), "sample_sd": float(np.std(vals, ddof=1))}


def pair_agreement(pairs, dimension, labels):
    """Pairwise endpoint equality/Jaccard, explicitly conditioned on both valid."""
    counts = Counter()
    per = {x: Counter() for x in labels}
    j = []
    for one, two in pairs:
        counts["possible_pairs"] += 1
        if one is None or two is None:
            counts["unavailable_pairs"] += 1
            continue
        counts["valid_pairs"] += 1
        a = one[dimension]
        b = two[dimension]
        sa = (
            {"hate" if a else "not_hate"}
            if dimension == "hate"
            else (
                {a}
                if dimension == "severity" and a is not None
                else set(a or []) if dimension != "severity" else set()
            )
        )
        sb = (
            {"hate" if b else "not_hate"}
            if dimension == "hate"
            else (
                {b}
                if dimension == "severity" and b is not None
                else set(b or []) if dimension != "severity" else set()
            )
        )
        counts["equal_pairs"] += sa == sb
        counts["gate_equal"] += one["hate"] == two["hate"]
        j.append(len(sa & sb) / len(sa | sb) if sa | sb else 1.0)
        for lab in labels:
            per[lab]["disagreements"] += (lab in sa) != (lab in sb)
            per[lab]["positive_on_either"] += lab in sa or lab in sb
            per[lab]["positive_on_both"] += lab in sa and lab in sb
    n = counts["valid_pairs"]
    return {
        **dict(counts),
        "endpoint_exact_agreement": counts["equal_pairs"] / n if n else None,
        "gate_agreement": counts["gate_equal"] / n if n else None,
        "mean_label_jaccard": float(np.mean(j)) if j else None,
        "per_label": {k: dict(v) for k, v in per.items()},
        "missing_policy": "Both valid; unavailable pairs counted separately. Empty/empty Jaccard = 1.",
    }

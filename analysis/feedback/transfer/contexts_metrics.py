from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
import re
import numpy as np

FIELDS = ("hate_types", "target_groups", "severity", "legal")
EDGES = (-1.0, 0.4, 0.5, 0.6, 0.7, 1.0000001)
BIN_NAMES = ("<0.40", "0.40–0.50", "0.50–0.60", "0.60–0.70", ">=0.70")


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def sig(value):
    return hashlib.sha256(
        (
            json.dumps(
                value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False
            )
            + "\n"
        ).encode("utf8")
    ).hexdigest()


def flat(text):
    return " ".join(str(text).split())


def gold(meta):
    bits = []
    if meta.get("gate") is not None:
        need(type(meta["gate"]) is bool, "Boolean annotation gate")
        bits.append("hate=" + str(meta["gate"]).lower())
    for key, label in [("target_groups", "target_group"), ("hate_types", "hate_type")]:
        if meta.get(key) is not None:
            bits.append(label + "=[" + ", ".join(meta[key]) + "]")
    if meta.get("severity") is not None:
        bits.append("severity=" + meta["severity"])
    if meta.get("legal") is not None:
        bits.append("legal=[" + ", ".join(meta["legal"]) + "]")
    return ", ".join(bits)


def example(hit):
    return '- "' + flat(hit["text"]) + '"\n  labels: ' + gold(hit["meta"])


def render_user(text, context, caution):
    d = context["definitions"]
    g = context["guidelines"]
    groups = context["example_groups"]
    need(
        len(d) == len(g) == 2
        and len(groups) == 1
        and groups[0][0] == "Labelled examples",
        "Context budgets/heading",
    )
    hits = groups[0][1]
    need(
        len(hits) == 5 and all(not h["meta"].get("illustrative_only") for h in hits),
        "Five ordinary labelled examples",
    )
    parts = [
        "## Label definitions\n"
        + "\n".join(
            "- "
            + str(h["meta"].get("dimension"))
            + (" / " + h["meta"]["label"] if h["meta"].get("label") else "")
            + ": "
            + flat(h["text"])
            for h in d
        ),
        "## Annotation guidelines\n" + "\n".join("- " + h["text"] for h in g),
        "## Labelled examples\n"
        + caution
        + "\n\n"
        + "\n".join(example(h) for h in hits),
        "## Text to classify (language: en)\n" + text,
    ]
    return "\n\n".join(parts)


def replacement(reference, treatment, correction):
    need(
        reference["definitions"] == treatment["definitions"]
        and reference["guidelines"] == treatment["guidelines"],
        "Abstract contexts changed",
    )
    rh = reference["example_groups"]
    th = treatment["example_groups"]
    need(
        len(rh) == len(th) == 1 and rh[0][0] == th[0][0] == "Labelled examples",
        "Heading changed",
    )
    rb, tb = rh[0][1], th[0][1]
    need(
        len(rb) == len(tb) == 5 and rb[:4] == tb[:4], "First-four order/content changed"
    )
    need(
        tb[4] == correction and correction["id"] not in {h["id"] for h in rb},
        "Wrong or duplicate inserted example",
    )
    need(
        set(correction["meta"]) == {"gate", *FIELDS}
        and type(correction["meta"]["gate"]) is bool
        and all(correction["meta"][f] is None for f in FIELDS),
        "Correction leaks fine gold",
    )
    return rb[4]


def fuse(trace, pool_ids):
    scores = {}
    ranks = {}
    for channel in ("dense", "bm25"):
        seq = trace[channel]
        need(len(seq) == 10 if channel == "dense" else len(seq) <= 10, "Channel budget")
        ids = [v["id"] for v in seq]
        need(len(ids) == len(set(ids)) and set(ids) <= pool_ids, "Channel identity")
        need(seq == sorted(seq, key=lambda v: (-v["score"], v["id"])), "Channel order")
        need(
            all(
                math.isfinite(v["score"]) and (channel != "bm25" or v["score"] > 0)
                for v in seq
            ),
            "Channel score eligibility",
        )
        for j, row in enumerate(seq, 1):
            scores[row["id"]] = scores.get(row["id"], Fraction(0)) + Fraction(1, 60 + j)
            ranks[row["id"], channel] = j

    fs = {}
    for channel in ("dense", "bm25"):
        for j, row in enumerate(trace[channel], 1):
            fs[row["id"]] = fs.get(row["id"], 0.0) + 1.0 / (60 + j)
    order = sorted(fs, key=lambda key: (-fs[key], key))
    need([r["id"] for r in trace["rrf"]] == order, "RRF identities/order")
    need(
        all(
            abs(row["score"] - float(scores[row["id"]])) < 1e-14 for row in trace["rrf"]
        ),
        "RRF reciprocal-rank arithmetic",
    )
    need(trace["selected_id"] == order[0], "Selected correction is not rank one")
    chosen = order[0]
    return dict(
        dense_rank=ranks.get((chosen, "dense")),
        bm25_rank=ranks.get((chosen, "bm25")),
        rrf_score=fs[chosen],
        rrf_margin=fs[chosen] - fs[order[1]] if len(order) > 1 else None,
        dense_best=trace["dense"][0]["score"],
        dense_winner_same=chosen == trace["dense"][0]["id"],
        lexical_winner_same=bool(trace["bm25"]) and chosen == trace["bm25"][0]["id"],
        retrieval_channels=(
            "both"
            if (chosen, "dense") in ranks and (chosen, "bm25") in ranks
            else "dense_only" if (chosen, "dense") in ranks else "bm25_only"
        ),
    )


def cosine(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    need(
        x.shape == y.shape
        and x.ndim == 1
        and np.isfinite(x).all()
        and np.isfinite(y).all(),
        "Vector shape/finite",
    )
    norm = np.linalg.norm(x) * np.linalg.norm(y)
    need(norm > 0, "Zero vector")
    return float(x @ y / norm)


def similarity_bin(x):
    need(-1 <= x <= 1.00000001, "Cosine outside range")
    return int(np.searchsorted(EDGES, x, side="right") - 1)


def concentration(counts):
    values = sorted(counts, reverse=True)
    total = sum(values)
    if not total:
        return dict(
            total=0,
            used=0,
            max_share=None,
            top5_share=None,
            top10_share=None,
            inverse_simpson=None,
        )
    return dict(
        total=total,
        used=sum(x > 0 for x in values),
        max_share=values[0] / total,
        top5_share=sum(values[:5]) / total,
        top10_share=sum(values[:10]) / total,
        inverse_simpson=total * total / sum(x * x for x in values),
    )


def point(counts):
    n, ref, trt = map(int, counts)
    return dict(
        assignments=n,
        reference_errors=ref,
        treatment_errors=trt,
        reference=ref / (3 * n) if n else None,
        treatment=trt / (3 * n) if n else None,
        difference=(trt - ref) / (3 * n) if n else None,
    )


def boot(features, clusters, seed, draws=10000):
    units = sorted(set(clusters))
    ui = {u: i for i, u in enumerate(units)}
    aggregate = np.zeros((len(units), 3), dtype=np.int64)
    np.add.at(aggregate, [ui[u] for u in clusters], features)
    rng = np.random.Generator(np.random.PCG64(seed))
    out = np.empty((draws, 3))
    start = 0
    while start < draws:
        size = min(32, draws - start)
        selected = rng.integers(0, len(units), (size, len(units)), dtype=np.int64)
        weights = np.asarray([np.bincount(x, minlength=len(units)) for x in selected])
        counts = weights @ aggregate
        out[start : start + size] = np.stack(
            [
                counts[:, 1] / (3 * counts[:, 0]),
                counts[:, 2] / (3 * counts[:, 0]),
                (counts[:, 2] - counts[:, 1]) / (3 * counts[:, 0]),
            ],
            axis=1,
        )
        start += size
    return out

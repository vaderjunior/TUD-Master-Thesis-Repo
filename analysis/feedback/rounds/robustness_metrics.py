import hashlib
from collections import defaultdict
import numpy as np

POLICIES = ["reference", "adaptive", "static", "random"]
CONTRASTS = ["adaptive-static", "adaptive-random", "adaptive-reference"]
COLUMNS = POLICIES + CONTRASTS


def need(ok, message):
    if not ok:
        raise ValueError(message)


def collapse(value):
    x = np.asarray(value)
    need(x.ndim == 4 and x.shape[1:] == (4, 10, 3), "Wrong output axes")
    out = np.zeros((len(x), 4, 4), dtype=np.int64)
    out[:, :, 0] = x[:, :, 0].sum(axis=-1) * 3
    for p in range(1, 4):
        out[:, :, p] = x[:, :, [p, p + 3, p + 6]].sum(axis=(2, 3))
    return out


def contrasts(counts):
    x = np.asarray(counts)
    need(x.ndim == 2 and x.shape[1] == 4, "Wrong policy axes")
    return np.column_stack(
        [
            x,
            *[
                x[:, 1] - x[:, POLICIES.index(p)]
                for p in ["static", "random", "reference"]
            ],
        ]
    )


def representatives(meta):
    groups = defaultdict(list)
    for i, m in enumerate(meta):
        groups[m["population"], m["gold_hate"], m["text_cluster"]].append(i)
    return {
        min(
            ids,
            key=lambda i: hashlib.sha256(
                ("A17-part7-representative|" + meta[i]["key"]).encode()
            ).hexdigest(),
        )
        for ids in groups.values()
    }


def masks(meta, gates, votes, execution):
    final = (gates[:, 3] >= 0).all(axis=(1, 2))
    strict = (votes[:, 3] == 3).all(axis=(1, 2))
    n = len(meta)
    rep = representatives(meta)
    replay = np.array([m["masks"]["replay_exclusion_member"] for m in meta])
    result = dict(
        final=final,
        fixed=(gates >= 0).all(axis=(1, 2, 3)),
        strict=strict,
        strict_fixed=(votes == 3).all(axis=(1, 2, 3)),
        replay_excluded=final & ~replay,
        representative=final & np.array([i in rep for i in range(n)]),
        no_ties=final & ~execution["ties"][:, 3].any(axis=(1, 2)),
        no_repairs=final & ~(execution["repairs"][:, 3] > 0).any(axis=(1, 2)),
        no_transport_retry=final & ~(execution["retries"][:, 3] > 0).any(axis=(1, 2)),
        policy_only=(gates[:, 3, 1:] >= 0).all(axis=(1, 2)),
    )
    for p in ["static", "random", "reference"]:
        cols = [1, 4, 7] + (
            [0] if p == "reference" else [POLICIES.index(p) + j for j in [0, 3, 6]]
        )
        result["paired_" + p] = (gates[:, 3, cols] >= 0).all(axis=(1, 2))
    return result


def grouped(values, labels, mode):
    v = np.asarray(values)
    need(v.ndim == 2 and len(v) == len(labels) and len(v) > 0, "Empty or wrong group")
    names = sorted(set(labels))
    lookup = {s: i for i, s in enumerate(names)}
    member = np.array([lookup[x] for x in labels])
    sizes = np.bincount(member, minlength=len(names))
    sums = np.zeros((len(names), v.shape[1]), dtype=np.int64)
    np.add.at(sums, member, v)
    if mode == "cluster_equal":
        numerators = sums / (9 * sizes[:, None])
        denominators = np.ones(len(names))
    else:
        need(mode == "text", "Unknown weighting")
        numerators = sums
        denominators = 9 * sizes
    return names, sizes, numerators, denominators


def bootstrap(values, labels, seed, mode="text", draws=10000):
    names, sizes, num, den = grouped(values, labels, mode)
    rng = np.random.Generator(np.random.PCG64(seed))
    out = np.empty((draws, num.shape[1]))
    for start in range(0, draws, 32):
        chosen = rng.integers(0, len(names), size=(min(32, draws - start), len(names)))
        out[start : start + len(chosen)] = (
            num[chosen].sum(axis=1) / den[chosen].sum(axis=1)[:, None]
        )
    return num.sum(axis=0) / den.sum(), out, names, sizes


def majority_completion(vote_gates, gold):
    v = np.asarray(vote_gates)
    g = np.asarray(gold, dtype=bool)
    need(
        v.ndim == 5 and v.shape[1:] == (4, 10, 3, 3) and g.shape == (len(v),),
        "Wrong vote axes",
    )
    need(np.isin(v, [-1, 0, 1]).all(), "Unknown vote code")
    yes = (v == 1).sum(axis=-1)
    missing = (v < 0).sum(axis=-1)
    lower = yes >= 2
    upper = yes + missing >= 2
    lo = np.where(g[:, None, None, None], ~upper, lower).astype(np.int8)
    hi = np.where(g[:, None, None, None], ~lower, upper).astype(np.int8)
    return lo, hi


def opposing(low, high):
    lo = np.asarray(low)
    hi = np.asarray(high)
    need(
        lo.shape == hi.shape and lo.ndim == 2 and lo.shape[1] == 4 and len(lo) > 0,
        "Wrong bound arrays",
    )
    need((lo <= hi).all(), "Reversed bounds")
    lc = lo.sum(axis=0)
    hc = hi.sum(axis=0)
    den = 9 * len(lo)
    out = {p: [float(lc[i] / den), float(hc[i] / den)] for i, p in enumerate(POLICIES)}
    for p in ["static", "random", "reference"]:
        j = POLICIES.index(p)
        out["adaptive-" + p] = [
            float((lc[1] - hc[j]) / den),
            float((hc[1] - lc[j]) / den),
        ]
    return dict(
        inputs=len(lo), denominator=den, lower_counts=lc, upper_counts=hc, intervals=out
    )


def omissions(values, labels):
    names, sizes, num, den = grouped(values, labels, "text")
    n = len(values)
    total = num.sum(axis=0)
    return [
        dict(
            group=name,
            removed=int(sizes[j]),
            remaining=n - int(sizes[j]),
            points=((total - num[j]) / (9 * (n - sizes[j])) if n > sizes[j] else None),
        )
        for j, name in enumerate(names)
    ]

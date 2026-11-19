import os

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
from collections import defaultdict
import hashlib
import numpy as np

VARIANTS = ("all_permitted", "matched_gradient")
CUTOFFS = (0.6, 0.5)
SETTINGS = tuple((variant, cutoff) for variant in VARIANTS for cutoff in CUTOFFS)
COLUMNS = (
    "linear_baseline",
    "linear_updated",
    "qwen_reference",
    "qwen_updated",
    "linear_change",
    "qwen_change",
    "difference_in_change",
    "updated_absolute_difference",
    *[f"pool_{i}_linear_change" for i in (1, 2, 3)],
    *[f"pool_{i}_qwen_change" for i in (1, 2, 3)],
    *[f"pool_{i}_difference_in_change" for i in (1, 2, 3)],
    *[f"pool_{i}_updated_absolute_difference" for i in (1, 2, 3)],
)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def seed(name):
    return int.from_bytes(
        hashlib.sha256(("A18-transfer-part8-support-v1|" + name).encode()).digest()[:8],
        "little",
    )


def setting_name(s):
    return SETTINGS[s][0] + "/cutoff_" + str(SETTINGS[s][1]).replace(".", "")


def coefficients():

    result = np.zeros((20, 80), dtype=np.int64)
    for s in range(4):
        b = np.zeros((20, 20), dtype=np.int64)
        start = s * 4
        b[start, 0] = 9
        b[start + 1 : start + 4, 1] = 3
        b[16, 2] = 3
        b[17:20, 3] = 1
        b[:, 4] = b[:, 1] - b[:, 0]
        b[:, 5] = b[:, 3] - b[:, 2]
        b[:, 6] = b[:, 4] - b[:, 5]
        b[:, 7] = b[:, 1] - b[:, 3]
        for j in range(3):
            b[start + j + 1, 8 + j] = 9
            b[start, 8 + j] = -9
            b[17 + j, 11 + j] = 3
            b[16, 11 + j] = -3
            b[:, 14 + j] = b[:, 8 + j] - b[:, 11 + j]
            b[start + j + 1, 17 + j] = 9
            b[17 + j, 17 + j] = -3
        result[:, s * 20 : (s + 1) * 20] = b
    return result


COEFFICIENTS = coefficients()


def expand(totals, n):
    totals = np.asarray(totals, dtype=np.int64)
    n = np.asarray(n)
    need(
        totals.shape[-1] == 20 and np.all(n > 0),
        "Positive denominators and twenty raw totals required",
    )
    return ((totals @ COEFFICIENTS) / (9 * n[..., None])).reshape(
        totals.shape[:-1] + (4, 20)
    )


def raw_rows(linear_errors, qwen_errors):
    need(
        linear_errors.shape[1:] == (2, 2, 4)
        and qwen_errors.shape == (len(linear_errors), 4, 3),
        "Input axes",
    )
    need(
        np.isin(linear_errors, [0, 1]).all() and np.isin(qwen_errors, [-1, 0, 1]).all(),
        "Binary error codes",
    )
    return np.column_stack(
        (
            linear_errors.reshape(len(linear_errors), 16),
            np.where(qwen_errors < 0, 0, qwen_errors).sum(axis=2),
        )
    ).astype(np.int64)


def grouping(meta, raw, common, eligible, mode="cluster"):
    need(mode in ("cluster", "source_stratified", "item"), "Unknown resampling mode")
    groups = defaultdict(list)
    for i in common:
        groups[
            meta[i]["key"] if mode == "item" else meta[i]["inference_cluster"]
        ].append(int(i))
    labels = sorted(groups)
    sizes = np.asarray([len(groups[label]) for label in labels], dtype=np.int64)
    counts = np.asarray(
        [raw[groups[label]].sum(axis=0, dtype=np.int64) for label in labels],
        dtype=np.int64,
    ).reshape(-1, 20)
    strata = defaultdict(list)
    if mode == "source_stratified":
        membership = defaultdict(set)
        for i in eligible:
            membership[meta[i]["inference_cluster"]].add(meta[i]["source"])
        for j, label in enumerate(labels):
            strata[tuple(sorted(membership[label]))].append(j)
    return labels, counts, sizes, dict(strata)


def bootstrap(
    meta, raw, common, eligible, rng_seed, mode="cluster", draws=10000, block=32
):
    labels, counts, sizes, strata = grouping(meta, raw, common, eligible, mode)
    need(len(labels) > 0, "No resampling units")
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    streams = []
    info = []
    for signature, ids in sorted(strata.items()):
        sub = seed("stratum|" + str(rng_seed) + "|" + "|".join(signature))
        streams.append((np.asarray(ids), np.random.Generator(np.random.PCG64(sub))))
        info.append(dict(source_signature=list(signature), units=len(ids), seed=sub))
    out = np.empty((draws, 4, 20), dtype=np.float64)
    for first in range(0, draws, block):
        length = min(block, draws - first)
        chosen = (
            np.concatenate(
                [
                    ids[g.integers(0, len(ids), (length, len(ids)), dtype=np.int64)]
                    for ids, g in streams
                ],
                axis=1,
            )
            if streams
            else rng.integers(0, len(labels), (length, len(labels)), dtype=np.int64)
        )
        out[first : first + length] = expand(
            counts[chosen].sum(axis=1), sizes[chosen].sum(axis=1)
        )
    return out, dict(
        seed=rng_seed,
        draws=draws,
        generator="PCG64",
        units=len(labels),
        mode=mode,
        strata=info,
        block_size=block,
        unit_order="Sorted inference cluster IDs; sorted input keys for item sensitivity.",
        weighting="Whole clusters with replacement; preserve all settings, pool fits and real Qwen repeats; item-weighted rates.",
    )


def bounds(
    linear_errors,
    qwen_errors,
    valid_votes,
    eligible,
    strict=False,
    lower=None,
    upper=None,
):

    n = len(eligible)
    if not n:
        return None
    l = linear_errors[eligible].reshape(n, 4, 4).sum(axis=0, dtype=np.int64)
    if lower is None:
        q = qwen_errors[eligible]
        valid = q >= 0
        if strict:
            valid &= valid_votes[eligible] == 3
        lo = np.where(valid, q, 0).sum(axis=(0, 2), dtype=np.int64)
        hi = lo + (~valid).sum(axis=(0, 2), dtype=np.int64)
    else:
        lo = lower[eligible].sum(axis=(0, 2), dtype=np.int64)
        hi = upper[eligible].sum(axis=(0, 2), dtype=np.int64)
    ref = [lo[0] / (3 * n), hi[0] / (3 * n)]
    upd = [lo[1:].sum() / (9 * n), hi[1:].sum() / (9 * n)]
    change = [upd[0] - ref[1], upd[1] - ref[0]]
    result = []
    for s in range(4):
        before = l[s, 0] / n
        after = l[s, 1:].sum() / (3 * n)
        delta = (l[s, 1:].sum() - 3 * l[s, 0]) / (3 * n)
        result.append(
            dict(
                linear_baseline=before,
                linear_updated=after,
                linear_change=delta,
                qwen_reference=ref,
                qwen_updated=upd,
                qwen_change=change,
                difference_in_change=[delta - change[1], delta - change[0]],
                updated_absolute_difference=[after - upd[1], after - upd[0]],
            )
        )
    return result


def binary_scores(gold, pred):
    need(pred.ndim == 2 and len(gold) == len(pred), "Model prediction matrix")
    need(np.isin(pred, [0, 1]).all(), "No missing predictions in binary quality")
    counts = np.asarray(
        [
            ((gold[:, None] == g) & (pred == v)).sum(axis=0)
            for g, v in ((0, 0), (0, 1), (1, 0), (1, 1))
        ],
        dtype=np.int64,
    ).T
    n = counts.sum(axis=1)
    tn, fp, fn, tp = counts.T

    def div(x, y):
        return np.divide(x, y, out=np.zeros_like(x, dtype=np.float64), where=y != 0)

    f1 = (div(2 * tn, 2 * tn + fp + fn) + div(2 * tp, 2 * tp + fp + fn)) / 2
    balanced = (div(tn, tn + fp) + div(tp, tp + fn)) / 2
    return counts, dict(
        macro_f1=f1,
        balanced_accuracy=balanced,
        accuracy=div(tn + tp, n),
        hate_precision=div(tp, tp + fp),
        hate_recall=div(tp, tp + fn),
        predicted_hate_rate=div(fp + tp, n),
    )

"""Common micro f1."""

import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def count_array(counts):
    c = np.asarray(counts)
    require(
        c.ndim >= 2 and c.shape[-1] == 3 and (c.shape[-2] > 0),
        "Expected non-empty label axis and TP/FP/FN axis.",
    )
    require(
        np.issubdtype(c.dtype, np.number) and np.isfinite(c).all(), "Counts must be finite numbers."
    )
    require(
        (c >= 0).all() and np.equal(c, np.floor(c)).all(),
        "Counts must be non-negative integers within each setting.",
    )
    return c.astype(np.int64, copy=False)


def label_f1(counts):
    c = count_array(counts)
    tp, fp, fn = (c[..., i] for i in range(3))
    denominator = 2 * tp + fp + fn
    return np.divide(
        2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0
    )


def label_indices(indices, n):
    idx = list(indices)
    require(
        len(idx) == len(set(idx)) and all((type(x) is int and 0 <= x < n for x in idx)),
        "Invalid label indices.",
    )
    return idx


def macro(counts, indices):
    c = count_array(counts)
    idx = label_indices(indices, c.shape[-2])
    if not idx:
        return None
    return label_f1(c)[..., idx].mean(axis=-1)


def micro(counts, indices=None):
    c = count_array(counts)
    idx = list(range(c.shape[-2])) if indices is None else label_indices(indices, c.shape[-2])
    if not idx:
        return None
    pooled = c[..., idx, :].sum(axis=-2)
    tp, fp, fn = (pooled[..., i] for i in range(3))
    denominator = 2 * tp + fp + fn
    return np.divide(
        2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0
    )


def contributions(gold, predictions, valid):
    y, p, v = (np.asarray(gold), np.asarray(predictions), np.asarray(valid))
    require(
        y.ndim == 2 and p.ndim == 3 and (p.shape[1:] == y.shape),
        "Gold and predictions must share item and label axes.",
    )
    require(v.shape == p.shape[:2] and v.dtype == bool, "Validity has the wrong shape or type.")
    require(
        np.isin(y, [0, 1]).all() and np.isin(p, [0, 1]).all(),
        "Only binary label indicators are accepted.",
    )
    y, p = (y.astype(bool), p.astype(bool))
    c = np.stack((p & y, p & ~y, ~p & y), axis=-1).astype(np.int64)
    c[~v] = 0
    return c


def common_mask(valid, setting_positions):
    v = np.asarray(valid)
    require(v.ndim == 2 and v.dtype == bool, "Invalid validity matrix.")
    pos = label_indices(setting_positions, len(v))
    require(bool(pos), "Comparison has no settings.")
    return v[pos].all(axis=0)


def hierarchical_mean(values, pools):
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and np.isfinite(values).all(), "Invalid setting scores.")
    require(bool(pools) and all((pool for pool in pools)), "Empty group or pool.")
    used = [i for pool in pools for i in pool]
    require(len(set(used)) == len(used), "A reused reference must not become another fitted model.")
    label_indices(used, len(values))
    return float(np.mean([values[pool].mean() for pool in pools]))


def exact_sets(gold, predictions):
    y, p = (np.asarray(gold), np.asarray(predictions))
    require(y.shape == p.shape and y.ndim == 2 and (len(y) > 0), "Invalid exact-set inputs.")
    return float(np.all(y == p, axis=1).mean())


def run_tests():
    """Synthetic examples only; this does not score any experiment."""
    from analysis.common import micro_f1_files as c

    tests = []

    def check(name, ok):
        require(bool(ok), name)
        tests.append(name)

    def close(a, b):
        return bool(np.allclose(a, b, atol=1e-14, rtol=0))

    check(
        "micro higher than macro",
        close(macro([[100, 0, 0], [0, 0, 10]], [0, 1]), 0.5)
        and close(micro([[100, 0, 0], [0, 0, 10]]), 200 / 210),
    )
    check(
        "micro lower than macro",
        close(macro([[0, 0, 100], [1, 0, 0]], [0, 1]), 0.5)
        and close(micro([[0, 0, 100], [1, 0, 0]]), 2 / 102),
    )
    check("perfect", close(micro([[3, 0, 0], [1, 0, 0]]), 1))
    check("zero denominator", close(micro([[0, 0, 0], [0, 0, 0]]), 0))
    check("unsupported macro unavailable", macro([[1, 0, 0]], []) is None)
    check("unsupported restricted micro unavailable", micro([[1, 0, 0]], []) is None)
    check("extra zero-gold label counts", close(micro([[1, 0, 0], [0, 2, 0]]), 0.5))
    check("different label scopes", close(micro([[1, 0, 0], [0, 2, 0]], [0]), 1))
    y = np.array([[1, 0], [0, 0]])
    p = np.array([[[0, 0], [0, 1]], [[1, 0], [0, 0]]])
    v = np.array([[True, True], [True, False]])
    counts = contributions(y, p, v)
    check("gate-suppressed set retains false negative", counts[0, 0, 0, 2] == 1)
    check("observed empty gold retains false positive", counts[0, 1, 1, 1] == 1)
    check("invalid answer is excluded", counts[1, 1].sum() == 0)
    check("common validity", common_mask(v, [0, 1]).tolist() == [True, False])
    check(
        "unknown validity is not an empty prediction",
        counts[1, 1].sum() == 0
        and contributions(y, np.zeros((1, 2, 2), dtype=int), np.ones((1, 2), dtype=bool))[
            0, 0, 0, 2
        ]
        == 1,
    )
    per = np.array([[[1, 0, 0], [0, 0, 1]], [[100, 0, 0], [0, 0, 0]]])
    check(
        "mean repeats differs from pooled counts",
        abs(float(micro(per).mean() - micro(per.sum(0)))) > 0.1,
    )
    check("equal pool weighting", close(hierarchical_mean([0, 0, 1], [[0, 1], [2]]), 0.5))
    check("set exactness is separate", close(exact_sets([[1, 0], [0, 1]], [[1, 1], [0, 1]]), 0.5))
    z = np.array([[8, 2, 2], [1, 20, 0]])
    check(
        "not support-weighted macro",
        abs(float(micro(z) - np.average(label_f1(z), weights=z[:, 0] + z[:, 2]))) > 0.1,
    )
    denominator = 2 * z[:, 0] + z[:, 1] + z[:, 2]
    check(
        "label-weight decomposition", close(micro(z), np.average(label_f1(z), weights=denominator))
    )
    for name, fn in [
        ("negative count rejected", lambda: micro([[-1, 0, 0]])),
        ("fractional setting count rejected", lambda: macro([[0.5, 0, 0]], [0])),
        ("duplicate label rejected", lambda: macro([[1, 0, 0]], [0, 0])),
        ("duplicate reference rejected", lambda: hierarchical_mean([0.5], [[0], [0]])),
    ]:
        try:
            fn()
        except ValueError:
            check(name, True)
        else:
            check(name, False)
    rng = np.random.default_rng(20261113)
    a = rng.integers(0, 100, size=(40, 7, 3))
    scalar_micro = []
    scalar_macro = []
    for setting in a.tolist():
        labels = []
        for tp, fp, fn in setting:
            precision = tp / (tp + fp) if tp + fp else 0
            recall = tp / (tp + fn) if tp + fn else 0
            labels.append(
                2 * precision * recall / (precision + recall) if precision + recall else 0
            )
        scalar_macro.append(sum(labels) / 7)
        tp, fp, fn = (sum((row[i] for row in setting)) for i in range(3))
        pr = tp / (tp + fp) if tp + fp else 0
        rc = tp / (tp + fn) if tp + fn else 0
        scalar_micro.append(2 * pr * rc / (pr + rc) if pr + rc else 0)
    check("independent scalar random macro", close(macro(a, list(range(7))), scalar_macro))
    check("independent scalar random micro", close(micro(a), scalar_micro))
    c.save(
        "synthetic_tests.json",
        {
            "status": "passed",
            "tests": tests,
            "count": len(tests),
            "random_setting_checks": 80,
            "code_sha256": c.sha(__file__),
        },
    )
    print("Synthetic metric checks passed:", len(tests), flush=True)


if __name__ == "__main__":
    run_tests()

import hashlib
import numpy as np


def require(ok, message):
    if not ok:
        raise ValueError(message)


def era(workers):
    values = set(np.asarray(workers).reshape(-1).tolist())
    require(values and values <= {8, 128}, "Unknown or missing runtime policy")
    return (
        "all_8" if values == {8} else "all_128" if values == {128} else "cross_boundary"
    )


def seed(name):
    return int.from_bytes(
        hashlib.sha256(("A16-part7-support-v1|" + name).encode()).digest()[:8], "little"
    )


def vote_completion(vote_gates, gold):

    v = np.asarray(vote_gates)
    g = np.asarray(gold, dtype=bool)
    require(
        v.ndim == 4 and v.shape[1:] == (4, 3, 3) and g.shape == (len(v),),
        "Wrong vote axes",
    )
    require(np.all(np.isin(v, [-99, 0, 1])), "Unexpected saved binary vote")
    yes = (v == 1).sum(axis=-1)
    missing = (v == -99).sum(axis=-1)
    lower_gate = yes >= 2
    upper_gate = yes + missing >= 2
    lower_error = np.where(g[:, None, None], ~upper_gate, lower_gate).astype(np.int8)
    upper_error = np.where(g[:, None, None], ~lower_gate, upper_gate).astype(np.int8)
    return lower_error, upper_error, missing


def opposing(low, high):
    lo = np.asarray(low)
    hi = np.asarray(high)
    require(
        lo.shape == hi.shape and lo.ndim == 3 and lo.shape[1:] == (4, 3),
        "Wrong bounds axes",
    )
    require(np.all(lo <= hi), "Reversed per-output bounds")
    n = len(lo)
    lc = lo.sum(axis=(0, 2), dtype=np.int64)
    hc = hi.sum(axis=(0, 2), dtype=np.int64)
    return dict(
        inputs=n,
        lower_error_counts=lc.tolist(),
        upper_error_counts=hc.tolist(),
        variable_output_counts=(hi != lo).sum(axis=(0, 2), dtype=np.int64).tolist(),
        reference=[int(lc[0]) / (3 * n), int(hc[0]) / (3 * n)] if n else None,
        treatment=(
            [int(lc[1:].sum()) / (9 * n), int(hc[1:].sum()) / (9 * n)] if n else None
        ),
        difference=(
            [
                int(lc[1:].sum() - 3 * hc[0]) / (9 * n),
                int(hc[1:].sum() - 3 * lc[0]) / (9 * n),
            ]
            if n
            else None
        ),
    )


def omission(total, sizes, counts):

    total = np.asarray(total, dtype=np.int64)
    sizes = np.asarray(sizes, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    require(
        counts.shape == (len(sizes), 4) and total.shape == (4,), "Wrong grouped counts"
    )
    require(
        np.all(sizes > 0)
        and np.all(counts >= 0)
        and np.all(counts <= 3 * sizes[:, None]),
        "Invalid grouped errors",
    )
    require(np.array_equal(counts.sum(axis=0), total), "Groups do not partition cohort")
    n = int(sizes.sum())
    remain = n - sizes
    left = total[None, :] - counts
    numerator = left[:, 1:].sum(axis=1) - 3 * left[:, 0]
    delta = np.divide(
        numerator, 9 * remain, out=np.full(len(remain), np.nan), where=remain > 0
    )
    return remain, numerator, delta

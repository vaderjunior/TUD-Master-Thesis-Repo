import numpy as np
from analysis.feedback.rounds.robustness_metrics import collapse
from analysis.feedback.rounds.robustness_metrics import bootstrap
from analysis.feedback.rounds.robustness_metrics import majority_completion
from analysis.feedback.rounds.robustness_metrics import representatives

METRICS = [
    "linear_baseline",
    "linear_updated",
    "qwen_reference",
    "qwen_updated",
    "linear_change",
    "qwen_change",
    "difference_in_change",
    "updated_absolute_difference",
]


def raw_counts(linear_error, qwen_error):

    l = np.asarray(linear_error)
    q = np.asarray(qwen_error)
    if l.ndim != 4 or l.shape[1:] != (4, 4, 10) or q.shape != (len(l), 4, 10, 3):
        raise ValueError("Wrong axes")
    out = np.empty((len(l), 4, 20), dtype=np.int64)
    for s in range(4):
        out[:, :, s * 4] = l[:, s, :, 0] * 9
        for p in range(1, 4):
            out[:, :, s * 4 + p] = l[:, s][:, :, [p, p + 3, p + 6]].sum(axis=2) * 3
    out[:, :, 16:] = collapse(q)
    return out


def expand(x):
    a = np.asarray(x)
    l = a[..., :16].reshape(*a.shape[:-1], 4, 4)
    q = a[..., 16:]
    lb = np.broadcast_to(l[..., 0, None], l[..., 1:].shape)
    lu = l[..., 1:]
    qr = np.broadcast_to(q[..., 0, None, None], lu.shape)
    qu = np.broadcast_to(q[..., None, 1:], lu.shape)
    ld, qd = lu - lb, qu - qr
    return np.stack([lb, lu, qr, qu, ld, qd, ld - qd, lu - qu], axis=-1)


def intervals(point, draws):
    p, d = expand(point), expand(draws)
    return p, np.quantile(d, [0.025, 0.975], axis=0).transpose(*range(1, d.ndim), 0)


def binary_scores(gold, decisions):
    y = np.asarray(gold, dtype=bool)
    p = np.asarray(decisions, dtype=bool)
    if p.ndim != 2 or p.shape[0] != len(y) or not len(y):
        raise ValueError("Wrong binary cohort")
    counts = np.stack(
        [
            ((~y[:, None]) & (~p)).sum(0),
            ((~y[:, None]) & p).sum(0),
            (y[:, None] & (~p)).sum(0),
            (y[:, None] & p).sum(0),
        ],
        axis=-1,
    )
    tn, fp, fn, tp = counts.T

    def div(a, b):
        return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

    return counts, dict(
        macro_f1=(div(2 * tp, 2 * tp + fp + fn) + div(2 * tn, 2 * tn + fp + fn)) / 2,
        balanced_accuracy=(div(tp, tp + fn) + div(tn, tn + fp)) / 2,
        accuracy=(tp + tn) / len(y),
        precision=div(tp, tp + fp),
        recall=div(tp, tp + fn),
        predicted_hate=(tp + fp) / len(y),
    )


def bound_metrics(linear_raw, qlo, qhi):

    l = np.asarray(linear_raw)
    low = np.concatenate([l, np.asarray(qlo)], axis=-1)
    high = np.concatenate([l, np.asarray(qhi)], axis=-1)
    a, b = expand(low), expand(high)
    out = np.stack([a, b], axis=-1)
    out[..., 5, 0] = a[..., 3] - b[..., 2]
    out[..., 5, 1] = b[..., 3] - a[..., 2]
    out[..., 6, 0] = a[..., 4] - out[..., 5, 1]
    out[..., 6, 1] = a[..., 4] - out[..., 5, 0]
    out[..., 7, 0] = a[..., 1] - b[..., 3]
    out[..., 7, 1] = a[..., 1] - a[..., 3]
    return out


def path_counts(errors):
    e = np.asarray(errors, dtype=bool)
    if e.ndim != 2 or e.shape[1] != 5:
        raise ValueError("Baseline plus four rounds required")
    fixed = e[:, 0, None] & ~e[:, 1:]
    new = ~e[:, 0, None] & e[:, 1:]
    first_fixed = e[:, 0] & ~e[:, 1]
    first_new = ~e[:, 0] & e[:, 1]
    return dict(
        paths=len(e),
        baseline_errors=int(e[:, 0].sum()),
        round_errors=e[:, 1:].sum(0),
        baseline_fixes=fixed.sum(0),
        baseline_new_errors=new.sum(0),
        adjacent_fixes=(e[:, :-1] & ~e[:, 1:]).sum(0),
        adjacent_new_errors=(~e[:, :-1] & e[:, 1:]).sum(0),
        first_fixes=int(first_fixed.sum()),
        first_fixes_stay_correct=int((first_fixed & (~e[:, 2:]).all(1)).sum()),
        first_fixes_reverse_then_recover=int(
            (first_fixed & e[:, 2:4].any(1) & ~e[:, 4]).sum()
        ),
        first_fixes_end_wrong=int((first_fixed & e[:, 4]).sum()),
        first_new_errors=int(first_new.sum()),
        first_new_errors_stay_wrong=int((first_new & e[:, 2:].all(1)).sum()),
        any_switch=int((e[:, 1:] != e[:, :-1]).any(1).sum()),
    )

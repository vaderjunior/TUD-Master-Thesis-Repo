import numpy as np
from training.full_data.common import require
from training.full_data.plan import HEADS
from training.full_data.plan import CV_SEEDS
from training.full_data.plan import FINAL_SEEDS
from training.full_data.plan import THRESHOLDS
from training.full_data.plan import FAMILIES
from training.full_data.plan import config_map
from training.full_data.plan import protocol


def truth(head, rows, labels):
    y = np.asarray([r["target"] for r in rows], dtype=np.int64)
    return (
        y
        if head in ("hate_type", "target_group")
        else np.eye(len(labels), dtype=np.int64)[y]
    )


def probabilities(head, logits):
    logits = np.asarray(logits, dtype=np.float64)
    require(logits.ndim == 2 and np.isfinite(logits).all(), "Invalid logits")
    if head in ("hate_type", "target_group"):
        return np.exp(-np.logaddexp(0.0, -logits))
    z = np.exp(logits - logits.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def decisions(head, prob, cutoff=None, gate=None):
    if head == "hate":
        positive = prob[:, 1] > cutoff
        out = np.column_stack((~positive, positive)).astype(np.int64)
    elif head == "severity":
        out = np.eye(prob.shape[1], dtype=np.int64)[prob.argmax(axis=1)]
    else:
        out = (prob >= cutoff).astype(np.int64)
    if gate is not None:
        out[~np.asarray(gate, dtype=bool)] = 0
    return out


def macro(y, z, supported):
    if not len(supported):
        return None
    a, b = y[:, supported].astype(bool), z[:, supported].astype(bool)
    tp = (a & b).sum(0)
    fp = (~a & b).sum(0)
    fn = (a & ~b).sum(0)
    return float(
        np.divide(
            2 * tp,
            2 * tp + fp + fn,
            out=np.zeros(len(supported)),
            where=(2 * tp + fp + fn) > 0,
        ).mean()
    )


def class_weights(head, y, classes, balanced):
    y = np.asarray(y)
    require(len(y) > 0, "Empty training partition")
    if head in ("hate_type", "target_group"):
        pos = y.sum(0)
        neg = len(y) - pos
        counts = np.stack((neg, pos), axis=1)
        weights = np.ones(counts.shape, dtype=np.float64)
        valid = (pos > 0) & (neg > 0)
        if balanced:
            weights[valid] = len(y) / (2 * counts[valid])
    else:
        counts = np.bincount(y.astype(int), minlength=classes)
        weights = np.ones(classes, dtype=np.float64)
        if balanced:
            present = counts > 0
            weights[present] = len(y) / (present.sum() * counts[present])
    return dict(counts=counts.tolist(), weights=weights.tolist(), balanced=balanced)


def candidate_key(row):
    cfg = config_map()[row["config"]]
    t = 0.5 if row["threshold"] is None else row["threshold"]
    return (
        -row["mean_score"],
        cfg["epochs"],
        cfg["learning_rate"],
        cfg["loss"] != "unweighted",
        abs(t - 0.5),
        -t,
    )


def select(data, oof):
    "oof[head][config] has axes neural seed, annotated row, output label."
    result = {}
    for family in FAMILIES:
        gate = None
        heads = {}
        allowed = (
            [protocol()["fixed_configs"][family]]
            if family != "tuned"
            else list(config_map())
        )
        for head in HEADS:
            spec = data["heads"][head]
            yy = truth(head, spec["rows"], spec["labels"])
            supported = np.flatnonzero((yy.sum(0) >= 2) & ((1 - yy).sum(0) >= 2))
            require(
                len(supported) > 0, "No selection support in declared original pool"
            )
            ix = [r["feature_row"] for r in spec["rows"]]
            candidates = []
            for cfg in allowed:
                pp = oof[head][cfg]
                require(
                    pp.shape == (len(CV_SEEDS), len(yy), yy.shape[1])
                    and np.isfinite(pp).all(),
                    "OOF shape/finite mismatch",
                )
                for cutoff in (None,) if head == "severity" else THRESHOLDS:
                    values = [
                        macro(
                            yy,
                            decisions(
                                head,
                                pp[j],
                                cutoff,
                                None if gate is None else gate[j, ix],
                            ),
                            supported,
                        )
                        for j in range(len(CV_SEEDS))
                    ]
                    candidates.append(
                        dict(
                            config=cfg,
                            threshold=cutoff,
                            seed_scores=values,
                            mean_score=float(np.mean(values)),
                        )
                    )
            chosen = min(candidates, key=candidate_key)
            heads[head] = dict(
                selected=chosen,
                candidates=candidates,
                selection_indices=supported.tolist(),
            )
            if head == "hate":
                gate = oof[head][chosen["config"]][:, :, 1] > chosen["threshold"]
        result[family] = heads
    return result

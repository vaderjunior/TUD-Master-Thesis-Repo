import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
from collections import defaultdict
from functools import lru_cache
import hashlib
import numpy as np

LABELS = {
    "binary": ["not_hate", "hate"],
    "hate_type": [
        "explicit",
        "incitement",
        "inferiority",
        "irony",
        "stereotyping",
        "threatening",
        "grievance",
    ],
    "target_group": [
        "race",
        "religion",
        "gender",
        "sexual_orientation",
        "disability",
        "national_origin",
        "other",
    ],
    "severity": ["low", "medium", "high"],
    "known_answer": [],
    "legal": ["propaganda", "incitement_public_peace", "insult_defamation"],
}
GOLD = {
    "hate_type": "gold_hate_types",
    "target_group": "gold_target_groups",
    "severity": "gold_severity",
    "legal": "gold_legal",
}
VIEWS = {
    "main": (False, False, False),
    "strict": (True, False, False),
    "replay_excluded": (False, False, True),
    "strict_replay_excluded": (True, False, True),
    "representative": (False, True, False),
    "representative_replay_excluded": (False, True, True),
}
COLS = [
    "reference",
    "pool_2026100441",
    "pool_2026100442",
    "pool_2026100443",
    "treatment",
    "difference",
]


def require(ok, message):
    if not ok:
        raise ValueError(message)


def seed(name):
    return int.from_bytes(
        hashlib.sha256(("A16-part4-quality-v1|" + name).encode()).digest()[:8], "little"
    )


@lru_cache(None)
def names(head):
    if head == "binary":
        return ["n", "tp", "tn", "fp", "fn"]
    if head == "known_answer":
        return ["n", "exact", "gate_correct"]
    cols = [
        "n",
        *[f"{label}:{c}" for label in LABELS[head] for c in ("tp", "fp", "fn")],
        "exact",
        "gated_exact",
        "gate_positive",
    ]
    if head in ("hate_type", "target_group", "legal"):
        cols += [
            "contains_gold",
            "only_extra",
            "only_missing",
            "both_extra_missing",
            "extra_labels",
            "missing_labels",
            "predicted_labels",
            "gold_labels",
        ]
    else:
        cols += [
            "provided",
            "rank_absolute",
            "rank_signed",
            "over",
            "under",
            *[
                f"confusion:{g}:{p}"
                for g in LABELS[head]
                for p in ["none", *LABELS[head]]
            ],
        ]
    return cols


@lru_cache(None)
def metric_names(head):
    if head == "binary":
        return ["macro_f1", "balanced_accuracy", "accuracy"]
    if head == "known_answer":
        return ["known_answer_exact", "gate_accuracy"]
    common = ["macro_f1", "micro_f1", "exact_set", "gated_exact_set"]
    if head in ("hate_type", "target_group", "legal"):
        return common + [
            "contains_gold",
            "only_extra",
            "only_missing",
            "both_extra_missing",
            "mean_extra_labels",
            "mean_missing_labels",
            "mean_predicted_labels",
        ]
    return common + [
        "severity_provided",
        "ordinal_mae_given_provided",
        "ordinal_bias_given_provided",
        "over_given_provided",
        "under_given_provided",
    ]


def eligible(row, head):
    if head in ("binary", "known_answer"):
        return True
    return row[GOLD[head]] is not None and (head == "legal" or row["gold_hate"])


def features(row, result, head):

    out = np.zeros(len(names(head)), dtype=np.int16)
    if result is None or not eligible(row, head):
        return out
    out[0] = 1
    index = {name: i for i, name in enumerate(names(head))}
    if head == "binary":
        code = (
            "tp"
            if row["gold_hate"] and result["hate"]
            else "fn" if row["gold_hate"] else "fp" if result["hate"] else "tn"
        )
        out[index[code]] = 1
        return out
    if head == "known_answer":
        correct = result["hate"] == row["gold_hate"]
        out[index["gate_correct"]] = correct
        for field, goldfield in GOLD.items():
            if not eligible(row, field):
                continue
            g = row[goldfield]
            v = result[field]
            correct &= (v == g) if field == "severity" else set(v) == set(g)
        out[index["exact"]] = correct
        return out
    gold = row[GOLD[head]]
    prediction = result[head]
    g = {gold} if head == "severity" else set(gold)
    p = (
        ({prediction} if prediction is not None else set())
        if head == "severity"
        else set(prediction)
    )
    require(g <= set(LABELS[head]) and p <= set(LABELS[head]), "Unknown label")
    for label in LABELS[head]:
        out[index[label + ":tp"]] = int(label in g and label in p)
        out[index[label + ":fp"]] = int(label not in g and label in p)
        out[index[label + ":fn"]] = int(label in g and label not in p)
    out[index["exact"]] = g == p
    out[index["gated_exact"]] = g == p and result["hate"]
    out[index["gate_positive"]] = result["hate"]
    if head in ("hate_type", "target_group", "legal"):
        extra = p - g
        missing = g - p
        for name, value in dict(
            contains_gold=g <= p,
            only_extra=bool(extra) and not missing,
            only_missing=bool(missing) and not extra,
            both_extra_missing=bool(extra) and bool(missing),
            extra_labels=len(extra),
            missing_labels=len(missing),
            predicted_labels=len(p),
            gold_labels=len(g),
        ).items():
            out[index[name]] = value
    else:
        out[
            index[
                f'confusion:{gold}:{prediction if prediction is not None else "none"}'
            ]
        ] = 1
        if prediction is not None:
            change = LABELS[head].index(prediction) - LABELS[head].index(gold)
            for name, value in dict(
                provided=1,
                rank_absolute=abs(change),
                rank_signed=change,
                over=change > 0,
                under=change < 0,
            ).items():
                out[index[name]] = value
    return out


def divide(x, y, zero=np.nan):
    x, y = np.broadcast_arrays(
        np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    )
    return np.divide(x, y, out=np.full(x.shape, zero, dtype=np.float64), where=y != 0)


def score(counts, head, macro_indices):

    ix = {name: i for i, name in enumerate(names(head))}
    v = lambda name: counts[..., ix[name]]
    n = v("n")
    if head == "binary":
        tp, tn, fp, fn = [v(name) for name in ("tp", "tn", "fp", "fn")]
        f = (
            divide(2 * tp, 2 * tp + fp + fn, 0) + divide(2 * tn, 2 * tn + fp + fn, 0)
        ) / 2
        values = [
            np.where(n > 0, f, np.nan),
            (divide(tp, tp + fn) + divide(tn, tn + fp)) / 2,
            divide(tp + tn, n),
        ]
    elif head == "known_answer":
        values = [divide(v("exact"), n), divide(v("gate_correct"), n)]
    else:
        label_counts = counts[..., 1 : 1 + 3 * len(LABELS[head])].reshape(
            *counts.shape[:-1], len(LABELS[head]), 3
        )
        tp, fp, fn = [label_counts[..., j] for j in range(3)]
        per = divide(2 * tp, 2 * tp + fp + fn, 0)
        macro = (
            per[..., macro_indices].mean(axis=-1)
            if macro_indices
            else np.full(n.shape, np.nan)
        )
        micro = divide(2 * tp.sum(axis=-1), (2 * tp + fp + fn).sum(axis=-1), 0)
        values = [
            np.where(n > 0, macro, np.nan),
            np.where(n > 0, micro, np.nan),
            divide(v("exact"), n),
            divide(v("gated_exact"), n),
        ]
        if head in ("hate_type", "target_group", "legal"):
            values += [
                divide(v(name), n)
                for name in (
                    "contains_gold",
                    "only_extra",
                    "only_missing",
                    "both_extra_missing",
                    "extra_labels",
                    "missing_labels",
                    "predicted_labels",
                )
            ]
        else:
            values += [
                divide(v("provided"), n),
                *[
                    divide(v(name), v("provided"))
                    for name in ("rank_absolute", "rank_signed", "over", "under")
                ],
            ]
    return np.stack(values, axis=-1)


def summaries(run_scores):
    require(run_scores.shape[-3:-1] == (4, 3), "Condition/repeat axes differ")
    conditions = run_scores.mean(axis=-2)
    treatment = conditions[..., 1:, :].mean(axis=-2)

    delta = (run_scores[..., 1:, :, :] - run_scores[..., :1, :, :]).mean(axis=(-3, -2))
    return np.concatenate(
        [conditions, treatment[..., None, :], delta[..., None, :]], axis=-2
    )


def selected(meta, indices, head, view, exclusions, valid, votes, arms=(1, 2, 3)):
    strict, reps, ex = VIEWS[view]
    ids = [
        i
        for i in indices
        if eligible(meta[i], head) and (not ex or meta[i]["key"] not in exclusions)
    ]
    if reps:
        groups = defaultdict(list)
        for i in ids:
            groups[meta[i]["gold_hate"], meta[i]["inference_cluster"]].append(i)
        keep = {
            min(
                g,
                key=lambda i: hashlib.sha256(
                    ("A16-representative|" + meta[i]["key"]).encode()
                ).hexdigest(),
            )
            for g in groups.values()
        }
        ids = [i for i in ids if i in keep]
    ok = valid[ids].copy()
    if strict:
        ok &= votes[ids] == 3
    common = [
        i for i, yes in zip(ids, np.all(ok[:, (0, *arms), :], axis=(1, 2))) if yes
    ]
    return np.asarray(ids, dtype=np.int64), np.asarray(common, dtype=np.int64)


def missing_bounds(meta, data, ids, valid, votes, head, macro_indices, strict=False):
    available = valid[ids] & ((votes[ids] == 3) if strict else True)
    unknown = ~available
    missing = unknown.sum(axis=0, dtype=np.int64)
    observed = (data[ids] * available[..., None]).sum(axis=0, dtype=np.int64)
    lower = observed.copy()
    upper = observed.copy()
    ix = {name: i for i, name in enumerate(names(head))}
    for value in (lower, upper):
        value[..., ix["n"]] += missing
    if head == "binary":
        positive = np.asarray([meta[i]["gold_hate"] for i in ids], dtype=bool)
        pos = (unknown & positive[:, None, None]).sum(axis=0, dtype=np.int64)
        neg = missing - pos
        lower[..., ix["fn"]] += pos
        lower[..., ix["fp"]] += neg
        upper[..., ix["tp"]] += pos
        upper[..., ix["tn"]] += neg
        allowed = metric_names(head)
    elif head == "known_answer":
        upper[..., ix["exact"]] += missing
        upper[..., ix["gate_correct"]] += missing
        allowed = metric_names(head)
    else:
        for label in LABELS[head]:
            positive = np.asarray(
                [
                    (
                        meta[i][GOLD[head]] == label
                        if head == "severity"
                        else label in meta[i][GOLD[head]]
                    )
                    for i in ids
                ]
            )
            pos = (unknown & positive[:, None, None]).sum(axis=0, dtype=np.int64)
            neg = missing - pos
            lower[..., ix[label + ":fn"]] += pos
            lower[..., ix[label + ":fp"]] += neg
            upper[..., ix[label + ":tp"]] += pos
        upper[..., ix["exact"]] += missing
        upper[..., ix["gated_exact"]] += missing
        allowed = ["macro_f1", "micro_f1", "exact_set", "gated_exact_set"]
        if head == "severity":
            upper[..., ix["provided"]] += missing
            allowed += ["severity_provided"]
    low_score = score(lower, head, macro_indices)
    high_score = score(upper, head, macro_indices)
    low_mean = summaries(low_score)
    high_mean = summaries(high_score)
    result = {}
    for metric in allowed:
        mi = metric_names(head).index(metric)
        result[metric] = dict(
            reference=[low_mean[0, mi], high_mean[0, mi]],
            treatment=[low_mean[4, mi], high_mean[4, mi]],
            opposing_difference=[
                low_mean[4, mi] - high_mean[0, mi],
                high_mean[4, mi] - low_mean[0, mi],
            ],
            run_low=low_score[..., mi].tolist(),
            run_high=high_score[..., mi].tolist(),
        )
    return dict(
        eligible_n=len(ids),
        missing_outputs=missing.tolist(),
        metrics=result,
        convention="Whole eligible cohort, separately per real output. Unknown labels all wrong/all correct; conservative multiclass relaxation for severity. These are identification bounds, not predictions or confidence intervals. Ordinal conditional-error bounds are not claimed.",
    )


def bootstrap(
    meta,
    data,
    ids,
    eligible_ids,
    head,
    macro_indices,
    random_seed,
    mode="cluster",
    draws=10000,
    block=16,
):
    label = lambda i: meta[i]["key"] if mode == "item" else meta[i]["inference_cluster"]
    groups = defaultdict(list)
    for i in ids:
        groups[label(i)].append(int(i))
    labels = sorted(groups)
    require(labels, "No bootstrap units")
    totals = np.stack(
        [data[groups[g]].sum(axis=0, dtype=np.int64) for g in labels]
    ).astype(np.float64)
    flat = totals.reshape(len(labels), -1)
    strata = {}
    if mode == "source_stratified":
        membership = defaultdict(set)
        for i in eligible_ids:
            membership[meta[i]["inference_cluster"]].add(meta[i]["source"])
        s = defaultdict(list)
        for i, label_ in enumerate(labels):
            s[tuple(sorted(membership[label_]))].append(i)
        strata = dict(sorted(s.items()))
    rng = np.random.Generator(np.random.PCG64(random_seed))
    values = np.empty((draws, 6, len(metric_names(head))), dtype=np.float64)
    for start in range(0, draws, block):
        size = min(block, draws - start)
        chosen = (
            np.concatenate(
                [
                    np.asarray(g)[
                        rng.integers(0, len(g), (size, len(g)), dtype=np.int64)
                    ]
                    for g in strata.values()
                ],
                axis=1,
            )
            if strata
            else rng.integers(0, len(labels), (size, len(labels)), dtype=np.int64)
        )
        weights = np.stack(
            [np.bincount(c, minlength=len(labels)) for c in chosen]
        ).astype(np.float64)
        counts = (weights @ flat).reshape(size, 4, 3, data.shape[-1])
        values[start : start + size] = summaries(score(counts, head, macro_indices))
    return values, dict(
        draws=draws,
        seed=random_seed,
        generator="PCG64",
        units=len(labels),
        mode=mode,
        block=block,
        strata=[
            dict(sources=list(key), units=len(value)) for key, value in strata.items()
        ],
        unit_order="Sorted keys for item sensitivity, otherwise sorted frozen inference-cluster IDs.",
        weight_rule="Whole-cluster multiplicities shared by all twelve outputs; score each actual run, then equal means.",
    )

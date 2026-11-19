"""Policy metrics."""

from collections import defaultdict
import numpy as np

PAIRS = ((0, 1), (0, 2), (1, 2))
CONTRASTS = ((2, 0), (1, 0), (2, 1))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def divide(a, b):
    a, b = (np.asarray(a), np.asarray(b))
    return np.divide(
        a, b, out=np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=float), where=b != 0
    )


def contributions(gold, predicted):
    y, p = (np.asarray(gold, dtype=bool), np.asarray(predicted, dtype=bool))
    require(
        y.ndim == 2 and p.ndim == 4 and (p.shape[2:] == y.shape),
        "Expected condition/repeat/text/label predictions",
    )
    return np.stack((p & y, p & ~y, ~p & y), axis=-1).astype(np.int64)


def f1(counts):
    x = np.asarray(counts)
    tp, fp, fn = (x[..., i] for i in range(3))
    per_label = divide(2 * tp, 2 * tp + fp + fn)
    pooled = x.sum(axis=-2)
    micro = divide(2 * pooled[..., 0], 2 * pooled[..., 0] + pooled[..., 1] + pooled[..., 2])
    return (per_label.mean(axis=-1), micro, per_label)


def endpoint_arrays(meta, roster, index, conditions):
    keys = meta["keys"]
    field = meta["dimension"]
    labels = meta["all_labels"]
    n = len(keys)
    gold = np.zeros((n, len(labels)), bool)
    predicted = np.zeros((3, 3, n, len(labels)), bool)
    valid = np.zeros((3, 3, n), bool)
    gates = np.zeros_like(valid)
    nv = np.zeros((3, 3, n), np.uint8)
    repairs = np.zeros_like(nv)
    for i, key in enumerate(keys):
        row = roster[key]
        if field == "hate":
            values = ["hate" if row["gate"] else "not_hate"]
            require(type(row["gate"]) is bool, "Unknown hate gold")
        elif field == "severity":
            values = [row["severity"]]
        else:
            values = row[
                {"hate_type": "hate_types", "target_group": "target_groups", "legal": "legal"}[
                    field
                ]
            ]
        require(values is not None and set(values) <= set(labels), "Unknown gold is not a negative")
        for value in values:
            gold[i, labels.index(value)] = True
        for ci, condition in enumerate(conditions):
            for repeat in range(3):
                entry = index[key, condition, repeat]
                v = entry["result"]
                nv[ci, repeat, i] = entry["n_valid"]
                repairs[ci, repeat, i] = entry["repairs"]
                if v is None:
                    continue
                valid[ci, repeat, i] = True
                gates[ci, repeat, i] = v["hate"]
                if field == "hate":
                    values = ["hate" if v["hate"] else "not_hate"]
                elif field == "severity":
                    values = [] if v["severity"] is None else [v["severity"]]
                else:
                    values = v[field]
                for value in values:
                    predicted[ci, repeat, i, labels.index(value)] = True
    require(np.array_equal(valid, nv > 0), "Validity/inner response disagreement")
    require(
        {label: int(gold[:, i].sum()) for i, label in enumerate(labels)} == meta["gold_support"],
        "Frozen support differs",
    )
    fine = field not in ("hate", "legal")
    if fine:
        require(not np.any(predicted & ~gates[..., None]), "Fine labels survive closed gate")
        require(
            all((roster[k]["gate"] is True for k in keys)),
            "Fine endpoint contains non-positive gold gate",
        )
    states = predicted.astype(np.int64) @ 2 ** np.arange(len(labels), dtype=np.int64)
    if fine:
        states += gates * 2 ** len(labels)
    exact = (predicted == gold).all(axis=-1)
    if fine:
        exact &= gates
    exact &= valid
    common = valid.all(axis=(0, 1))
    return dict(
        gold=gold,
        predicted=predicted,
        valid=valid,
        gates=gates,
        n_valid=nv,
        repairs=repairs,
        states=states,
        correct=exact,
        common=common,
    )


def consistency(states, correct, gates=None, fine=False):
    require(states.shape == correct.shape and states.ndim == 3, "Wrong repetition shape")
    agree = np.stack([states[:, a] == states[:, b] for a, b in PAIRS], axis=1)
    correct_pairs = np.stack([correct[:, a] & correct[:, b] for a, b in PAIRS], axis=1)
    require(not np.any(correct_pairs & ~agree), "Correct states disagree")
    for j, (a, b) in enumerate(PAIRS):
        require(
            not np.any(agree[:, j] & (correct[:, a] != correct[:, b])),
            "Same state has different correctness",
        )
    same = (states[:, 0] == states[:, 1]) & (states[:, 0] == states[:, 2])
    always = correct.all(axis=1)
    never = ~correct.any(axis=1)
    values = dict(
        pairwise_agreement=agree.mean(axis=(1, 2)),
        pairwise_correct_agreement=correct_pairs.mean(axis=(1, 2)),
        pairwise_wrong_agreement=(agree & ~correct_pairs).mean(axis=(1, 2)),
        three_way_agreement=same.mean(axis=1),
        stable_correct=always.mean(axis=1),
        stable_wrong=(same & never).mean(axis=1),
        changing_but_always_wrong=(~same & never).mean(axis=1),
        variable_correctness=(~always & ~never).mean(axis=1),
        all_three_correct_count=always.sum(axis=1),
        all_three_wrong_count=never.sum(axis=1),
        correct_repeat_counts=np.stack(
            [(correct.sum(axis=1) == j).sum(axis=1) for j in range(4)], axis=1
        ),
    )
    if fine:
        closed = (~gates).all(axis=1)
        require(
            not np.any(closed & (~same | ~never)),
            "Repeated closed gate must be same incorrect fine state",
        )
        values["all_three_gate_closed"] = closed.mean(axis=1)
        values["all_three_gate_closed_count"] = closed.sum(axis=1)
        values["stable_wrong_open_or_mixed_gate"] = (same & never & ~closed).mean(axis=1)
    require(
        np.allclose(
            values["stable_correct"]
            + values["stable_wrong"]
            + values["changing_but_always_wrong"]
            + values["variable_correctness"],
            1,
        ),
        "Repeat partition differs",
    )
    return (values, agree.mean(axis=1))


def point_metrics(meta, arrays, mask=None):
    mask = arrays["common"] if mask is None else np.asarray(mask, dtype=bool)
    require(mask.any(), "Empty comparison")
    y = arrays["gold"][mask]
    p = arrays["predicted"][:, :, mask]
    g = arrays["gates"][:, :, mask]
    require(arrays["valid"][:, :, mask].all(), "Invalid output entered point calculation")
    correct = arrays["correct"][:, :, mask]
    states = arrays["states"][:, :, mask]
    field = meta["dimension"]
    counts = contributions(y, p).sum(axis=2)
    macro, micro, lf1 = f1(counts)
    consistency_values, pairwise = consistency(states, correct, g, field not in ("hate", "legal"))
    out = dict(
        n=int(mask.sum()),
        macro_f1=macro.mean(axis=1),
        macro_f1_repeats=macro,
        exact=correct.mean(axis=(1, 2)),
        exact_repeats=correct.mean(axis=2),
        counts=counts,
        per_label_f1=lf1.mean(axis=1),
        per_label_f1_repeats=lf1,
        precision=divide(counts[..., 0], counts[..., 0] + counts[..., 1]).mean(axis=1),
        recall=divide(counts[..., 0], counts[..., 0] + counts[..., 2]).mean(axis=1),
        gold_support=y.sum(axis=0),
        repeatability=consistency_values,
    )
    if field in ("hate_type", "target_group", "legal"):
        intersection = (p & y).sum(axis=-1)
        union = (p | y).sum(axis=-1)
        jaccard = np.where(union == 0, 1.0, divide(intersection, union))
        nonempty = y.any(axis=-1)
        out.update(
            micro_f1=micro.mean(axis=1),
            micro_f1_repeats=micro,
            jaccard=jaccard.mean(axis=(1, 2)),
            jaccard_repeats=jaccard.mean(axis=2),
            nonempty_gold_n=int(nonempty.sum()),
            nonempty_gold_jaccard=jaccard[:, :, nonempty].mean(axis=(1, 2)),
            empty_gold_n=int((~nonempty).sum()),
            empty_gold_exact=None if nonempty.all() else correct[:, :, ~nonempty].mean(axis=(1, 2)),
            extra_labels_per_text=(p & ~y).sum(axis=-1).mean(axis=(1, 2)),
            missing_labels_per_text=(~p & y).sum(axis=-1).mean(axis=(1, 2)),
            any_extra_rate=(p & ~y).any(axis=-1).mean(axis=(1, 2)),
            any_missing_rate=(~p & y).any(axis=-1).mean(axis=(1, 2)),
            predicted_labels_per_text=p.sum(axis=-1).mean(axis=(1, 2)),
            gold_labels_per_text=float(y.sum(axis=-1).mean()),
        )
    if field not in ("hate", "legal"):
        out["gate_closed_rate"] = (~g).mean(axis=(1, 2))
        out["gate_closed_counts"] = (~g).sum(axis=2)
    if field == "hate":
        yy = y[:, meta["all_labels"].index("hate")]
        pp = p[..., meta["all_labels"].index("hate")]
        confusion = dict(
            tn=(~pp & ~yy).sum(axis=2),
            fp=(pp & ~yy).sum(axis=2),
            fn=(~pp & yy).sum(axis=2),
            tp=(pp & yy).sum(axis=2),
        )
        out.update(
            confusion_repeats=confusion,
            confusion_mean={k: v.mean(axis=1) for k, v in confusion.items()},
            negative_n=int((~yy).sum()),
            positive_n=int(yy.sum()),
            false_positive_rate=confusion["fp"].mean(axis=1) / (~yy).sum(),
            false_negative_rate=confusion["fn"].mean(axis=1) / yy.sum(),
            positive_precision=divide(confusion["tp"], confusion["tp"] + confusion["fp"]).mean(
                axis=1
            ),
            positive_recall=divide(confusion["tp"], confusion["tp"] + confusion["fn"]).mean(axis=1),
        )
    if field == "severity":
        emitted = p.any(axis=-1)
        value = p.argmax(axis=-1)
        gold = y.argmax(axis=-1)
        difference = value - gold
        shared = emitted.all(axis=(0, 1))
        error_counts = {
            str(d): ((difference == d) & emitted).sum(axis=2) for d in (-2, -1, 0, 1, 2)
        }
        n_emitted = emitted.sum(axis=2)
        mae = divide((abs(difference) * emitted).sum(axis=2), n_emitted)
        require((n_emitted > 0).all() and shared.any(), "Severity conditional population empty")
        out.update(
            severity=dict(
                emitted_n_repeats=n_emitted,
                emitted_rate=emitted.mean(axis=(1, 2)),
                gate_open_missing_n_repeats=(g & ~emitted).sum(axis=2),
                signed_error_counts_repeats=error_counts,
                mae_emitted_repeats=mae,
                mae_emitted=mae.mean(axis=1),
                common_emitted_n=int(shared.sum()),
                common_emitted_mae_repeats=abs(difference[:, :, shared]).mean(axis=2),
                common_emitted_mae=abs(difference[:, :, shared]).mean(axis=(1, 2)),
            )
        )
    transitions = []
    for left, right in CONTRASTS:
        fixes = (correct[left] & ~correct[right]).sum(axis=1)
        regressions = (~correct[left] & correct[right]).sum(axis=1)
        record = dict(
            left=left,
            right=right,
            fixes_repeats=fixes,
            regressions_repeats=regressions,
            fixes_mean=float(fixes.mean()),
            regressions_mean=float(regressions.mean()),
            net_correct_mean=float((fixes - regressions).mean()),
            always_fixed=int((correct[left] & ~correct[right]).all(axis=0).sum()),
            always_regressed=int((~correct[left] & correct[right]).all(axis=0).sum()),
        )
        if field == "hate":
            for label, is_positive in (("false_negative", True), ("false_positive", False)):
                subset = yy == is_positive
                record[label + "_fixed_repeats"] = (
                    correct[left, :, subset] & ~correct[right, :, subset]
                ).sum(axis=0)
                record[label + "_new_repeats"] = (
                    ~correct[left, :, subset] & correct[right, :, subset]
                ).sum(axis=0)
        transitions.append(record)
    out["transitions"] = transitions
    return out


def cluster_features(meta, arrays, roster):
    keys = meta["keys"]
    mask = arrays["common"]
    full_membership = defaultdict(set)
    for key in keys:
        full_membership[roster[key]["text_cluster"]].add(roster[key]["source"])
    selected = [k for k, m in zip(keys, mask) if m]
    y = arrays["gold"][mask]
    p = arrays["predicted"][:, :, mask]
    counts = contributions(y, p)
    _, agreement = consistency(arrays["states"][:, :, mask], arrays["correct"][:, :, mask])
    rows = counts.transpose(2, 0, 1, 3, 4).reshape(len(selected), -1)
    rows = np.column_stack((rows, agreement.T, np.ones(len(selected))))
    grouped = {}
    for i, key in enumerate(selected):
        cluster = roster[key]["text_cluster"]
        if cluster not in grouped:
            grouped[cluster] = np.zeros(rows.shape[1], dtype=float)
        grouped[cluster] += rows[i]
    clusters = sorted(grouped)
    strata = defaultdict(list)
    for i, cluster in enumerate(clusters):
        strata[tuple(sorted(full_membership[cluster]))].append(i)
    return (
        np.array([grouped[k] for k in clusters]),
        [np.array(strata[s]) for s in sorted(strata)],
        strata,
    )


def bootstrap_from_features(features, strata, labels, draws, seed, chunk=256, progress=None):
    rng = np.random.default_rng(seed)
    count_columns = 3 * 3 * labels * 3
    output = {
        name: np.zeros((draws, 3))
        for name in ("macro_f1", "micro_f1_all_labels", "mean_pairwise_repeat_agreement")
    }
    for start in range(0, draws, chunk):
        size = min(chunk, draws - start)
        weights = np.zeros((size, len(features)), dtype=float)
        for indices in strata:
            weights[:, indices] = rng.multinomial(
                len(indices), np.ones(len(indices)) / len(indices), size=size
            )
        totals = weights @ features
        counts = totals[:, :count_columns].reshape(size, 3, 3, labels, 3)
        macro, micro, _ = f1(counts)
        output["macro_f1"][start : start + size] = macro.mean(axis=2)
        output["micro_f1_all_labels"][start : start + size] = micro.mean(axis=2)
        output["mean_pairwise_repeat_agreement"][start : start + size] = (
            totals[:, count_columns : count_columns + 3] / totals[:, -1, None]
        )
        if progress and (start // 5000 != (start + size) // 5000 or start + size == draws):
            progress(start + size)
    return output


def interval(values, confidence):
    tail = (1 - confidence) / 2
    return np.quantile(values, [tail, 1 - tail], method="linear").tolist()


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value

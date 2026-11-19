from collections import defaultdict
import hashlib
import numpy as np
from analysis.feedback.rounds import quality_inputs as a

k = a.quality


def selected(meta, members, head, view):
    eligible = [i for i in members if k.eligible(meta[i], head)]
    if view == "replay_excluded":
        eligible = [
            i for i in eligible if not meta[i]["masks"]["replay_exclusion_member"]
        ]
    if view == "representative":
        groups = defaultdict(list)
        for i in eligible:
            groups[meta[i]["gold_hate"], meta[i]["text_cluster"]].append(i)
        keep = {
            min(
                ids,
                key=lambda i: hashlib.sha256(
                    ("A17-part6-representative|" + meta[i]["key"]).encode()
                ).hexdigest(),
            )
            for ids in groups.values()
        }
        eligible = [i for i in eligible if i in keep]
    if view == "fixed":
        good = lambda m: m["masks"]["all_round_common"]
    elif view == "strict":
        good = lambda m: m["masks"]["round_strict"][3]
    elif view.startswith("round"):
        good = lambda m: m["masks"]["round_common"][int(view[-1]) - 1]
    else:
        good = lambda m: m["masks"]["round_common"][3]
    return eligible, [i for i in eligible if good(meta[i])]


def columns(rounds):
    cols = {}
    for rnd in rounds:
        for p in a.POLICIES:
            cols[f"r{rnd}_{p}"] = [(rnd - 1, p, 1)]
        for p in ["reference", "static", "random"]:
            cols[f"r{rnd}_adaptive-{p}"] = [(rnd - 1, "adaptive", 1), (rnd - 1, p, -1)]
    if rounds == [1, 2, 3, 4]:
        for p in a.POLICIES:
            cols["r4-r1_" + p] = [(3, p, 1), (0, p, -1)]
        for p in ["reference", "static", "random"]:
            cols["r4-r1_adaptive-" + p] = [
                (3, "adaptive", 1),
                (0, "adaptive", -1),
                (3, p, -1),
                (0, p, 1),
            ]
    return cols


def policy_means(scores):

    by_condition = scores.mean(axis=-2)
    return np.stack(
        [by_condition[..., 0, :]]
        + [
            by_condition[
                ..., [a.CONDITIONS.index(f"{p}_s{s}") for s in a.STREAMS], :
            ].mean(axis=-2)
            for p in a.POLICIES[1:]
        ],
        axis=-2,
    )


def summarize(scores, cols):
    means = policy_means(scores)
    return np.stack(
        [
            sum(
                coef * means[..., rnd, a.POLICIES.index(p), :] for rnd, p, coef in terms
            )
            for terms in cols.values()
        ],
        axis=-2,
    )


def boot(meta, data, ids, head, macro_indices, cols, random_seed, draws=10000):
    groups = defaultdict(list)
    for i in ids:
        groups[meta[i]["inference_cluster"]].append(i)
    labels = sorted(groups)
    totals = np.stack(
        [data[groups[g]].sum(axis=0, dtype=np.int64) for g in labels]
    ).astype(float)
    flat = totals.reshape(len(labels), -1)
    rng = np.random.Generator(np.random.PCG64(random_seed))
    out = np.empty((draws, len(cols), len(k.metric_names(head))), dtype=float)
    for start in range(0, draws, 16):
        n = min(16, draws - start)
        chosen = rng.integers(0, len(labels), (n, len(labels)), dtype=np.int64)
        weight = np.stack(
            [np.bincount(c, minlength=len(labels)) for c in chosen]
        ).astype(float)
        counts = (weight @ flat).reshape(n, 4, 10, 3, data.shape[-1])
        out[start : start + n] = summarize(k.score(counts, head, macro_indices), cols)
    return out, labels


def missing_bounds(meta, data, ids, valid, head, macro_indices):

    observed = data[ids, 3].sum(axis=0, dtype=np.int64)
    unknown = ~valid[ids, 3]
    missing = unknown.sum(axis=0, dtype=np.int64)
    low = observed.copy()
    high = observed.copy()
    ix = {n: i for i, n in enumerate(k.names(head))}
    for counts in [low, high]:
        counts[..., ix["n"]] += missing
    if head == "binary":
        truth = np.asarray([meta[i]["gold_hate"] for i in ids])
        pos = (unknown & truth[:, None, None]).sum(axis=0)
        neg = missing - pos
        low[..., ix["fn"]] += pos
        low[..., ix["fp"]] += neg
        high[..., ix["tp"]] += pos
        high[..., ix["tn"]] += neg
        allowed = k.metric_names(head)
    elif head == "known_answer":
        high[..., ix["exact"]] += missing
        high[..., ix["gate_correct"]] += missing
        allowed = k.metric_names(head)
    else:
        for label in k.LABELS[head]:
            truth = np.asarray(
                [
                    (
                        meta[i][k.GOLD[head]] == label
                        if head == "severity"
                        else label in meta[i][k.GOLD[head]]
                    )
                    for i in ids
                ]
            )
            pos = (unknown & truth[:, None, None]).sum(axis=0)
            neg = missing - pos
            low[..., ix[label + ":fn"]] += pos
            low[..., ix[label + ":fp"]] += neg
            high[..., ix[label + ":tp"]] += pos
        high[..., ix["exact"]] += missing
        high[..., ix["gated_exact"]] += missing
        allowed = ["macro_f1", "micro_f1", "exact_set", "gated_exact_set"]
        if head == "severity":
            high[..., ix["provided"]] += missing
            allowed += ["severity_provided"]
    lo = policy_means(k.score(low, head, macro_indices)[None, ...])[0]
    hi = policy_means(k.score(high, head, macro_indices)[None, ...])[0]
    result = {}
    for metric in allowed:
        j = k.metric_names(head).index(metric)
        result[metric] = {p: [lo[pi, j], hi[pi, j]] for pi, p in enumerate(a.POLICIES)}
        result[metric].update(
            {
                "adaptive-" + p: [lo[1, j] - hi[pi, j], hi[1, j] - lo[pi, j]]
                for pi, p in enumerate(a.POLICIES)
                if p != "adaptive"
            }
        )
    return dict(
        eligible=len(ids),
        missing_outputs=missing,
        metrics=result,
        interpretation="Opposing full-roster missing-output bounds, not confidence intervals. Severity F1 uses a conservative multiclass relaxation; ordinal conditional-error bounds are not asserted.",
    )

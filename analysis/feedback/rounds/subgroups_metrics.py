from collections import Counter
import numpy as np


def cohort(meta, indices, truth, view):
    if view not in ["fixed", "final", "strict", "round1", "round2", "round3", "round4"]:
        raise ValueError("Unknown view")
    result = []
    for i in indices:
        r = meta[i]
        if r["gold_hate"] != truth:
            continue
        m = r["masks"]
        valid = (
            m["all_round_common"]
            if view == "fixed"
            else (
                m["round_strict"][3]
                if view == "strict"
                else m["round_common"][3 if view == "final" else int(view[-1]) - 1]
            )
        )
        if valid:
            result.append(i)
    return result


def support(meta, indices, truth):
    eligible = [i for i in indices if meta[i]["gold_hate"] == truth]
    return dict(
        roster=len(eligible),
        fixed=len(cohort(meta, indices, truth, "fixed")),
        final=len(cohort(meta, indices, truth, "final")),
        strict=len(cohort(meta, indices, truth, "strict")),
        round_counts=[
            len(cohort(meta, indices, truth, "round" + str(r))) for r in range(1, 5)
        ],
        inference_clusters=len({meta[i]["inference_cluster"] for i in eligible}),
        text_clusters=len({meta[i]["text_cluster"] for i in eligible}),
        templates=len(
            {meta[i]["template_id"] for i in eligible if "template_id" in meta[i]}
        ),
        fixed_clusters=len(
            {
                meta[i]["inference_cluster"]
                for i in cohort(meta, indices, truth, "fixed")
            }
        ),
    )


def influence(values, cluster_ids):
    values = np.asarray(values, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(cluster_ids):
        raise ValueError("Influence dimensions")
    labels = sorted(set(cluster_ids))
    n = len(values)
    if not n:
        return dict(status="undefined_no_texts")
    totals = values.sum(axis=0)
    rows = []
    for label in labels:
        mask = np.array([v == label for v in cluster_ids])
        remaining = n - int(mask.sum())
        if remaining:
            rows.append(
                dict(
                    cluster=label,
                    removed_inputs=int(mask.sum()),
                    remaining_inputs=remaining,
                    values=(
                        (totals - values[mask].sum(axis=0)) / (9 * remaining)
                    ).tolist(),
                )
            )
    return dict(
        status="defined" if len(labels) > 1 else "undefined_one_cluster",
        inputs=n,
        clusters=len(labels),
        full=(totals / (9 * n)).tolist(),
        equal_cluster_mean=np.mean(
            [values[np.array(cluster_ids) == c].mean(axis=0) / 9 for c in labels],
            axis=0,
        ).tolist(),
        leave_one_out=rows,
    )


def pair_counts(error, left, right):

    x = np.asarray(error[left])
    y = np.asarray(error[right])
    if x.shape != (4, 10, 3) or y.shape != x.shape or np.any(x < 0) or np.any(y < 0):
        raise ValueError("Incomplete pair")
    both = ((x == 0) & (y == 0)).astype(np.int64)
    out = np.zeros((4, 4), dtype=np.int64)
    out[:, 0] = 3 * both[:, 0, :].sum(axis=1)
    for p in range(1, 4):
        out[:, p] = both[:, [p, p + 3, p + 6], :].sum(axis=(1, 2))
    return out


def selected_category(errors, reference, category):
    e = np.asarray(errors)
    r = np.asarray(reference)
    if e.shape != (4, 9) or r.shape != (4, 3) or np.any(e < 0) or np.any(r < 0):
        return False
    counts = e.sum(axis=1)
    ref = r.sum(axis=1)
    delta = counts - 3 * ref
    if category == "retained_average_benefit":
        return bool(np.all(delta < 0))
    if category == "final_regression":
        return bool(delta[-1] > 0)
    if category == "mixed_relative_direction":
        return bool(np.any(delta < 0) and np.any(delta > 0))
    if category == "persistent_error":
        return bool(np.all(e == 1))
    raise ValueError("Unknown case category")

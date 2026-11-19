from collections import defaultdict
import hashlib
import os

for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"
import numpy as np

REP_PREFIX = "A16-representative|"
DRAW_COLUMNS = [
    "reference",
    "pool_2026100441",
    "pool_2026100442",
    "pool_2026100443",
    "treatment",
    "difference",
]
VIEWS = {
    "main": (False, False, False),
    "strict": (True, False, False),
    "replay_excluded": (False, False, True),
    "strict_replay_excluded": (True, False, True),
    "representative": (False, True, False),
    "representative_replay_excluded": (False, True, True),
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def seed_for(name):
    return int.from_bytes(
        hashlib.sha256(("A16-part2-support-v1|" + name).encode()).digest()[:8], "little"
    )


def eligible_indices(
    meta, population, truth, *, excluded=(), representatives=False, source=None
):
    excluded = set(excluded)
    selected = [
        i
        for i, r in enumerate(meta)
        if r["population"] == population
        and r["gold_hate"] == truth
        and r["key"] not in excluded
        and (source is None or r["source"] == source)
    ]
    if representatives:
        groups = defaultdict(list)
        for i in selected:
            groups[meta[i]["inference_cluster"]].append(i)
        keep = {
            min(
                values,
                key=lambda i: hashlib.sha256(
                    (REP_PREFIX + meta[i]["key"]).encode()
                ).hexdigest(),
            )
            for values in groups.values()
        }
        selected = [i for i in selected if i in keep]
    return np.asarray(selected, dtype=np.int64)


def cohort(
    meta,
    error,
    valid_votes,
    population,
    truth,
    *,
    strict=False,
    excluded=(),
    representatives=False,
    source=None,
    arms=(1, 2, 3),
    eligible_override=None
):
    need(
        error.shape == valid_votes.shape
        and error.ndim == 3
        and error.shape[1:] == (4, 3),
        "Wrong prediction axes",
    )
    need(
        len(arms) == len(set(arms)) and set(arms) <= {1, 2, 3} and len(arms) > 0,
        "Invalid treatment arms",
    )
    ids = eligible_indices(
        meta,
        population,
        truth,
        excluded=excluded,
        representatives=representatives,
        source=source,
    )
    if eligible_override is not None:
        allowed = set(int(i) for i in eligible_override)
        ids = np.asarray([i for i in ids if int(i) in allowed], dtype=np.int64)
    available = error[ids] >= 0
    if strict:
        available &= valid_votes[ids] == 3
    keep = np.all(available[:, (0, *arms), :], axis=(1, 2))
    common = ids[keep]
    observed = np.where(available, error[ids], 0).sum(axis=(0, 2), dtype=np.int64)
    missing = (~available).sum(axis=(0, 2), dtype=np.int64)
    actual = error[common].sum(axis=(0, 2), dtype=np.int64)
    n = len(ids)
    lo_c = int(observed[0]) / (3 * n) if n else None
    hi_c = int(observed[0] + missing[0]) / (3 * n) if n else None
    lo_t = int(observed[list(arms)].sum()) / (3 * len(arms) * n) if n else None
    hi_t = (
        int((observed + missing)[list(arms)].sum()) / (3 * len(arms) * n) if n else None
    )
    cn = len(common)
    control = int(actual[0]) / (3 * cn) if cn else None
    treatment = int(actual[list(arms)].sum()) / (3 * len(arms) * cn) if cn else None
    numerator = int(actual[list(arms)].sum()) - len(arms) * int(actual[0])
    return {
        "eligible": ids,
        "common": common,
        "arms": tuple(arms),
        "strict": strict,
        "point": {
            "eligible_n": n,
            "common_n": cn,
            "excluded_by_availability": n - cn,
            "reference_rate": control,
            "treatment_rate": treatment,
            "difference": numerator / (3 * len(arms) * cn) if cn else None,
            "reference_error_count": int(actual[0]) if cn else 0,
            "treatment_error_count": int(actual[list(arms)].sum()) if cn else 0,
            "reference_output_count": 3 * cn,
            "treatment_output_count": 3 * len(arms) * cn,
            "pool_rates": [int(actual[j]) / (3 * cn) for j in arms] if cn else None,
            "observed_errors_by_condition_full_eligible": observed.tolist(),
            "missing_outputs_by_condition_full_eligible": missing.tolist(),
            "reference_missing_bounds": [lo_c, hi_c] if n else None,
            "treatment_missing_bounds": [lo_t, hi_t] if n else None,
            "difference_missing_bounds": [lo_t - hi_c, hi_t - lo_c] if n else None,
        },
    }


def grouped(meta, error, common, *, mode="cluster", eligible=None):
    need(mode in ("cluster", "source_stratified", "item"), "Unknown resampling mode")
    groups = defaultdict(list)
    for i in common:
        label = meta[i]["key"] if mode == "item" else meta[i]["inference_cluster"]
        groups[label].append(int(i))
    labels = sorted(groups)
    counts = np.zeros((len(labels), 4), dtype=np.int64)
    sizes = np.zeros(len(labels), dtype=np.int64)
    for j, label in enumerate(labels):
        ids = groups[label]
        counts[j] = error[ids].sum(axis=(0, 2), dtype=np.int64)
        sizes[j] = len(ids)
    strata = defaultdict(list)
    if mode == "source_stratified":
        membership = defaultdict(set)
        for i in eligible:
            membership[meta[i]["inference_cluster"]].add(meta[i]["source"])
        for j, label in enumerate(labels):
            strata[tuple(sorted(membership[label]))].append(j)
    return labels, counts, sizes, dict(strata)


def bootstrap(meta, error, c, seed, *, draws=10000, mode="cluster", block_size=32):
    need(
        c["arms"] == (1, 2, 3), "Bootstrap comparison must retain all three fixed pools"
    )
    labels, counts, sizes, strata = grouped(
        meta, error, c["common"], mode=mode, eligible=c["eligible"]
    )
    if not labels:
        return np.empty((0, 6)), {
            "units": 0,
            "mode": mode,
            "seed": seed,
            "draws": 0,
            "strata": [],
        }
    rng = np.random.Generator(np.random.PCG64(seed))
    stratum_info = []
    streams = []
    for signature, indices in sorted(strata.items()):
        s = seed_for("stratum|" + str(seed) + "|" + "|".join(signature))
        streams.append(
            (
                np.asarray(indices, dtype=np.int64),
                np.random.Generator(np.random.PCG64(s)),
            )
        )
        stratum_info.append(
            {"source_signature": list(signature), "units": len(indices), "seed": s}
        )
    values = np.empty((draws, 6), dtype=np.float64)
    for start in range(0, draws, block_size):
        length = min(block_size, draws - start)
        if streams:
            chosen = np.concatenate(
                [
                    indices[
                        g.integers(
                            0, len(indices), (length, len(indices)), dtype=np.int64
                        )
                    ]
                    for indices, g in streams
                ],
                axis=1,
            )
        else:
            chosen = rng.integers(0, len(labels), (length, len(labels)), dtype=np.int64)
        n = sizes[chosen].sum(axis=1)
        total = counts[chosen].sum(axis=1)
        values[start : start + length, :4] = total / (3 * n[:, None])
        values[start : start + length, 4] = total[:, 1:].sum(axis=1) / (9 * n)
        values[start : start + length, 5] = (
            total[:, 1:].sum(axis=1) - 3 * total[:, 0]
        ) / (9 * n)
    need(np.all(np.isfinite(values)), "Nonfinite resample")
    return values, {
        "units": len(labels),
        "mode": mode,
        "seed": seed,
        "draws": draws,
        "generator": "PCG64",
        "strata": stratum_info,
        "block_size": block_size,
        "weighting": "Sample complete clusters with replacement; item-weighted rate after cluster multiplicity. Preserve all repeats/pools together.",
        "stratum_membership": "Source signatures from endpoint-eligible inputs before output-validity filtering; never split a mixed-source cluster.",
    }


def intervals(draws):
    if not len(draws):
        return {
            "interval98_75": None,
            "interval95": None,
            "pool_difference_intervals95": None,
        }
    diff = draws[:, 5]
    return {
        "interval98_75": np.quantile(
            diff, [0.00625, 0.99375], method="linear"
        ).tolist(),
        "interval95": np.quantile(diff, [0.025, 0.975], method="linear").tolist(),
        "pool_difference_intervals95": np.quantile(
            draws[:, 1:4] - draws[:, :1], [0.025, 0.975], axis=0, method="linear"
        ).T.tolist(),
    }


def direction(value):
    return (
        "fewer_errors"
        if value < 0
        else "more_errors" if value > 0 else "same_average_errors"
    )


def description(meta, error, votes, c, conditions):
    ids = c["common"]
    need(len(ids) > 0, "No common inputs for description")
    rr = error[ids]
    reference = rr[:, 0, :]
    runs = []
    for j, condition in enumerate(conditions):
        for repeat in range(3):
            count = int(rr[:, j, repeat].sum())
            runs.append(
                {
                    "condition": condition,
                    "repeat": repeat,
                    "errors": count,
                    "n": len(ids),
                    "rate": count / len(ids),
                }
            )
    pools = []
    for j in range(1, 4):
        treatment = rr[:, j, :]
        fixed = int(((reference == 1) & (treatment == 0)).sum())
        new = int(((reference == 0) & (treatment == 1)).sum())
        persisted = int(((reference == 1) & (treatment == 1)).sum())
        stable_correct = int(((reference == 0) & (treatment == 0)).sum())
        need(
            new - fixed == int(treatment.sum() - reference.sum()),
            "Transition/rate identity failed",
        )
        pools.append(
            {
                "condition": conditions[j],
                "errors": int(treatment.sum()),
                "outputs": int(treatment.size),
                "rate": float(treatment.mean()),
                "difference": (new - fixed) / treatment.size,
                "matched_repeat_fixes": fixed,
                "matched_repeat_new_errors": new,
                "matched_repeat_persistent_errors": persisted,
                "matched_repeat_stable_correct": stable_correct,
            }
        )
    directions = defaultdict(int)
    variation = {
        "reference_disagreement_inputs": 0,
        "any_treatment_repeat_disagreement_inputs": 0,
        "pool_directions_disagree_inputs": 0,
        "all_twelve_correct": 0,
        "all_twelve_wrong": 0,
        "equal_mean_with_offsetting_repeat_changes": 0,
    }
    per_input = []
    for local, i in enumerate(ids):
        totals = rr[local].sum(axis=1)
        delta = int(totals[1:].sum() - 3 * totals[0])
        d = direction(delta)
        directions[d] += 1
        variation["reference_disagreement_inputs"] += int(0 < totals[0] < 3)
        variation["any_treatment_repeat_disagreement_inputs"] += int(
            any(0 < v < 3 for v in totals[1:])
        )
        variation["pool_directions_disagree_inputs"] += int(
            np.any(totals[1:] < totals[0]) and np.any(totals[1:] > totals[0])
        )
        variation["all_twelve_correct"] += int(totals.sum() == 0)
        variation["all_twelve_wrong"] += int(totals.sum() == 12)
        variation["equal_mean_with_offsetting_repeat_changes"] += int(
            delta == 0 and np.any(rr[local, 1:, :] != rr[local, :1, :])
        )
        per_input.append(
            {
                "key": meta[i]["key"],
                "population": meta[i]["population"],
                "source": meta[i]["source"],
                "inference_cluster": meta[i]["inference_cluster"],
                "stage": meta[i]["stage"],
                "gold_hate": meta[i]["gold_hate"],
                "condition_error_counts": totals.tolist(),
                "difference_ninths": delta,
                "direction": d,
            }
        )
    return {
        "runs": runs,
        "pools": pools,
        "per_text_average_directions": dict(directions),
        "variation": variation,
        "interpretation": "Repeated outputs and reused reference comparisons are dependent. Repeat-ID pairings are descriptive, not controlled common-random-seed interventions.",
    }, per_input

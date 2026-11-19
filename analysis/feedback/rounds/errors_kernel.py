import hashlib
import numpy as np

POLICIES = ["reference", "adaptive", "static", "random"]
PAIRS = [
    ("adaptive", "static"),
    ("adaptive", "random"),
    ("adaptive", "reference"),
    ("static", "reference"),
    ("random", "reference"),
]


def arrays(rows):
    gates = np.array(
        [
            [
                [
                    [-1 if v["result"] is None else int(v["result"]["hate"]) for v in c]
                    for c in rr
                ]
                for rr in r["predictions"]
            ]
            for r in rows
        ],
        dtype=np.int8,
    )
    votes = np.array(
        [
            [[[v["n_valid"] for v in c] for c in rr] for rr in r["predictions"]]
            for r in rows
        ],
        dtype=np.int8,
    )
    if gates.shape != (len(rows), 4, 10, 3):
        raise ValueError("Wrong four-round matrix shape")
    gold = np.array([r["gold_hate"] for r in rows], dtype=np.int8)
    error = np.where(
        gates < 0, -1, (gates != gold[:, None, None, None]).astype(np.int8)
    )
    return gates, votes, error


def policy_counts(error):
    observed = np.maximum(error, 0).astype(np.int64)
    missing = (error < 0).astype(np.int64)

    def collapse(value):
        result = np.empty((len(value), 4, 4), dtype=np.int64)
        result[:, :, 0] = value[:, :, 0, :].sum(axis=2) * 3
        for p in range(1, 4):
            result[:, :, p] = value[:, :, [p, p + 3, p + 6], :].sum(axis=(2, 3))
        return result

    return collapse(observed), collapse(missing)


def cohort(rows, gates, votes, population, endpoint, mask, round_no):
    gold = endpoint == "FNR"
    eligible = np.array(
        [r["population"] == population and r["gold_hate"] == gold for r in rows]
    )
    if mask == "round_common":
        valid = (gates[:, round_no - 1] >= 0).all(axis=(1, 2))
    elif mask == "all_round_common":
        valid = (gates >= 0).all(axis=(1, 2, 3))
    elif mask == "round_strict":
        valid = (votes[:, round_no - 1] == 3).all(axis=(1, 2))
    else:
        raise ValueError("Unknown cohort mask")
    return np.flatnonzero(eligible & valid)


def term(round_no, policy, weight=1):
    return {"round": round_no, "policy": policy, "weight": weight}


def columns(rounds, trajectories=False):
    result = {}
    for rnd in rounds:
        for p in POLICIES:
            result[f"r{rnd}_{p}"] = [term(rnd, p)]
        for left, right in PAIRS:
            result[f"r{rnd}_{left}-{right}"] = [term(rnd, left), term(rnd, right, -1)]
    if trajectories:
        for earlier, later in [(1, 4), (1, 2), (2, 3), (3, 4)]:
            for p in POLICIES:
                result[f"r{later}-r{earlier}_{p}"] = [
                    term(later, p),
                    term(earlier, p, -1),
                ]
            for p in POLICIES[1:]:
                result[f"r{later}-r{earlier}_{p}-reference"] = [
                    term(later, p),
                    term(later, "reference", -1),
                    term(earlier, p, -1),
                    term(earlier, "reference"),
                ]
    return result


def transformed_counts(counts, indices, definitions):
    values = np.zeros((len(indices), len(definitions)), dtype=np.int64)
    for j, terms in enumerate(definitions.values()):
        for t in terms:
            values[:, j] += (
                t["weight"]
                * counts[indices, t["round"] - 1, POLICIES.index(t["policy"])]
            )
    return values


def grouped(values, cluster_ids):
    labels = sorted(set(cluster_ids))
    if not labels or len(values) != len(cluster_ids):
        raise ValueError("Undefined endpoint or wrong cluster join")
    positions = {c: i for i, c in enumerate(labels)}
    membership = np.array([positions[c] for c in cluster_ids], dtype=np.int64)
    sizes = np.bincount(membership, minlength=len(labels))
    sums = np.zeros((len(labels), values.shape[1]), dtype=np.int64)
    np.add.at(sums, membership, values)
    return labels, sizes, sums


def bootstrap(values, cluster_ids, seed, draws=10000):
    labels, sizes, sums = grouped(values, cluster_ids)
    rng = np.random.Generator(np.random.PCG64(seed))
    result = np.empty((draws, values.shape[1]), dtype=np.float64)
    for start in range(0, draws, 32):
        stop = min(start + 32, draws)
        chosen = rng.integers(0, len(labels), size=(stop - start, len(labels)))
        numerator = sums[chosen].sum(axis=1)
        denominator = 9 * sizes[chosen].sum(axis=1)
        result[start:stop] = numerator / denominator[:, None]
    if not np.isfinite(result).all():
        raise ValueError("Undefined bootstrap replicate")
    return result, labels, sizes


def support_seed(name):
    return int.from_bytes(
        hashlib.sha256(("A17-part2-support-v1|" + name).encode()).digest()[:8], "little"
    )


def bounds(counts, missing, indices, round_no):
    if not len(indices):
        raise ValueError("Undefined full-roster endpoint")
    errors = counts[indices, round_no - 1].sum(axis=0)
    absent = missing[indices, round_no - 1].sum(axis=0)
    denominator = 9 * len(indices)
    result = {"inputs": len(indices), "policies": {}, "contrasts": {}}
    for i, policy in enumerate(POLICIES):
        observed = denominator - int(absent[i])
        result["policies"][policy] = {
            "error_numerator_over_nine": int(errors[i]),
            "missing_numerator_over_nine": int(absent[i]),
            "denominator_over_nine": denominator,
            "available_case_rate": float(errors[i] / observed) if observed else None,
            "bound": [
                float(errors[i] / denominator),
                float((errors[i] + absent[i]) / denominator),
            ],
        }
    for left, right in PAIRS:
        a, b = POLICIES.index(left), POLICIES.index(right)
        result["contrasts"][left + "-" + right] = [
            float((errors[a] - errors[b] - absent[b]) / denominator),
            float((errors[a] + absent[a] - errors[b]) / denominator),
        ]
    return result

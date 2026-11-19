from collections import defaultdict
import hashlib
import math


def require(ok, message):
    if not ok:
        raise ValueError(message)


def contrasts(science):

    require(
        science["bootstrap_draws"] == 10000 and science["decoder_repeats"] == 3,
        "Frozen draws/repeats differ",
    )
    require(
        science["primary"]["confidence"] == 0.975
        and science["mechanism"]["confidence"] == 1 - 0.05 / 6,
        "Frozen confidence families differ",
    )
    require(
        science["primary"]["control"] == "reference5"
        and science["primary"]["treatment"]
        == science["mechanism"]["treatment"]
        == "reserved5__error_balanced",
        "Frozen system comparison differs",
    )
    require(
        science["mechanism"]["controls"]
        == [
            "natural5__error_balanced",
            "extended6__error_balanced",
            "masked5__error_balanced",
        ],
        "Frozen mechanism controls differ",
    )
    require(
        science["primary"]["endpoints"]
        == science["mechanism"]["endpoints"]
        == ["FPR", "FNR"],
        "Frozen endpoint order differs",
    )
    require(
        science["primary"]["bootstrap_seeds"] == [2026101201, 2026101202]
        and science["mechanism"]["bootstrap_seeds"]
        == list(range(2026101210, 2026101216)),
        "Frozen seeds differ",
    )
    seeds = science["pool_seeds"]
    require(seeds == [2026100441, 2026100442, 2026100443], "Frozen pool seeds differ")
    result = []
    for family in ("primary", "mechanism"):
        spec = science[family]
        controls = [spec["control"]] if family == "primary" else spec["controls"]
        for i, control in enumerate(controls):
            for j, endpoint in enumerate(spec["endpoints"]):
                treatment_conditions = [
                    spec["treatment"] + "_s" + str(s) for s in seeds
                ]
                control_conditions = (
                    [control]
                    if control == "reference5"
                    else [control + "_s" + str(s) for s in seeds]
                )
                require(
                    set(treatment_conditions + control_conditions)
                    <= set(science["conditions"]),
                    "Frozen condition absent",
                )
                result.append(
                    {
                        "id": family + "/" + control + "/" + endpoint,
                        "family": family,
                        "endpoint": endpoint,
                        "treatment_conditions": treatment_conditions,
                        "control_conditions": control_conditions,
                        "confidence": spec["confidence"],
                        "seed": spec["bootstrap_seeds"][i * 2 + j],
                        "draws": science["bootstrap_draws"],
                    }
                )
    return result


def valid_prediction(record, strict):
    require(
        type(record["n_valid"]) is int and 0 <= record["n_valid"] <= 3,
        "Invalid valid-vote count",
    )
    if record["result"] is None:
        require(record["n_valid"] == 0, "Null output has usable votes")
        return False
    require(
        record["n_valid"] > 0 and type(record["result"]["hate"]) is bool,
        "Invalid final hate prediction",
    )
    return not strict or record["n_valid"] == 3


def cohort(
    roster, predictions, spec, *, strict=False, representatives=False, excluded=()
):

    require(spec["endpoint"] in ("FPR", "FNR"), "Unknown rate endpoint")
    truth = spec["endpoint"] == "FNR"
    bykey = {r["key"]: r for r in roster}
    require(len(bykey) == len(roster), "Duplicate roster identity")
    require(
        all(
            type(r["gold"]["hate"]) is bool
            and isinstance(r["cluster"], str)
            and r["cluster"]
            for r in roster
        ),
        "Unknown gold polarity or cluster",
    )
    arms = [spec["treatment_conditions"], spec["control_conditions"]]
    require(
        all(a and len(a) == len(set(a)) for a in arms)
        and not set(arms[0]) & set(arms[1]),
        "Duplicated or overlapping condition membership",
    )
    require(set(excluded) <= set(bykey), "Exclusion outside the original roster")
    keys = [
        r["key"]
        for r in roster
        if r["gold"]["hate"] == truth and r["key"] not in set(excluded)
    ]
    if representatives:
        groups = defaultdict(list)
        for k in keys:
            groups[bykey[k]["cluster"]].append(k)
        selected = {
            min(
                g,
                key=lambda k: hashlib.sha256(
                    ("A15-representative|" + k).encode()
                ).hexdigest(),
            )
            for g in groups.values()
        }
        keys = [k for k in keys if k in selected]
    common, errors, bounds = [], {}, {}
    for key in keys:
        data = [
            [
                predictions[key, condition, repeat]
                for condition in arm
                for repeat in range(3)
            ]
            for arm in arms
        ]
        validity = [[valid_prediction(r, strict) for r in arm] for arm in data]
        bounds[key] = [
            [
                sum(
                    (r["result"]["hate"] != truth) if ok else missing
                    for r, ok in zip(arm, mask)
                )
                / len(arm)
                for missing in (0, 1)
            ]
            for arm, mask in zip(data, validity)
        ]
        if all(all(mask) for mask in validity):
            common.append(key)
            errors[key] = [
                sum(r["result"]["hate"] != truth for r in arm) / len(arm)
                for arm in data
            ]
    return {
        "original_ids": keys,
        "common_ids": common,
        "errors": errors,
        "bounds": bounds,
        "clusters": {k: bykey[k]["cluster"] for k in common},
        "strict_all_three_votes": strict,
        "representatives_before_validity": representatives,
        "excluded_replay_ids": list(excluded),
        "outputs_per_input": [len(a) * 3 for a in arms],
    }


def cluster_totals(c):
    labels = sorted(set(c["clusters"].values()))
    totals = [[0.0, 0.0] for _ in labels]
    sizes = [0 for _ in labels]
    pos = {label: i for i, label in enumerate(labels)}
    for key in c["common_ids"]:
        i = pos[c["clusters"][key]]
        sizes[i] += 1
        for arm in (0, 1):
            totals[i][arm] += c["errors"][key][arm]
    return labels, totals, sizes


def resampled_difference(totals, sizes, indices):

    require(
        len(indices) > 0
        and all(type(i) is int and 0 <= i < len(sizes) for i in indices),
        "Invalid cluster sample",
    )
    denominator = sum(sizes[i] for i in indices)
    require(denominator > 0, "Empty cluster sample")
    return sum(totals[i][0] - totals[i][1] for i in indices) / denominator


def summarize(c, spec):

    import numpy as np

    require(
        type(spec["draws"]) is int and spec["draws"] > 0 and 0 < spec["confidence"] < 1,
        "Invalid uncertainty contract",
    )
    keys, common = c["original_ids"], c["common_ids"]
    out = {
        "status": "not_estimable_no_common_valid_inputs",
        "contrast": spec["id"],
        "endpoint": spec["endpoint"],
        "original_n": len(keys),
        "common_n": len(common),
        "common_ids": common,
        "invalid_ids": sorted(set(keys) - set(common)),
        "outputs_per_input": c["outputs_per_input"],
        "strict_all_three_votes": c["strict_all_three_votes"],
        "representatives_before_validity": c["representatives_before_validity"],
        "excluded_replay_ids": c["excluded_replay_ids"],
    }
    out["missing_difference_bounds"] = (
        [
            sum(c["bounds"][k][0][0] - c["bounds"][k][1][1] for k in keys) / len(keys),
            sum(c["bounds"][k][0][1] - c["bounds"][k][1][0] for k in keys) / len(keys),
        ]
        if keys
        else None
    )
    if not common:
        return out, np.empty((0,), dtype=float)
    labels, totals, sizes = cluster_totals(c)
    t, n = np.asarray(totals, dtype=float), np.asarray(sizes, dtype=float)
    rng = np.random.Generator(np.random.PCG64(spec["seed"]))
    distribution = np.empty(spec["draws"], dtype=float)
    for start in range(0, spec["draws"], 250):
        stop = min(start + 250, spec["draws"])
        chosen = rng.integers(0, len(labels), size=(stop - start, len(labels)))
        sampled = t[chosen].sum(axis=1) / n[chosen].sum(axis=1)[:, None]
        distribution[start:stop] = sampled[:, 0] - sampled[:, 1]
    point = [sum(c["errors"][k][i] for k in common) / len(common) for i in (0, 1)]
    alpha = (1 - spec["confidence"]) / 2
    out.update(
        status="scored_requires_full_review",
        treatment_rate=point[0],
        control_rate=point[1],
        difference=point[0] - point[1],
        family_confidence=spec["confidence"],
        family_interval=np.quantile(
            distribution, [alpha, 1 - alpha], method="linear"
        ).tolist(),
        descriptive_interval95=np.quantile(
            distribution, [0.025, 0.975], method="linear"
        ).tolist(),
        bootstrap={
            "seed": spec["seed"],
            "draws": spec["draws"],
            "generator": "PCG64",
            "clusters": len(labels),
            "unit": "whole original cluster; item-weighted rate",
            "pool_repeat_uncertainty": "conditional on observed pools and repetitions",
        },
    )
    require(all(math.isfinite(x) for x in distribution), "Nonfinite bootstrap result")
    return out, distribution

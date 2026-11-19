import hashlib
from collections import defaultdict
from analysis.feedback.classifiers import common as a
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.feedback.classifiers import kernel as k
from analysis.feedback.rounds import robustness_metrics as old


def seed(name):
    return int.from_bytes(
        hashlib.sha256(("A18-cumulative-review-v1|" + name).encode()).digest()[:8],
        "big",
    )


def main():
    a.contract()
    assembly = a.read(a.OUT / "assembled.json")
    a.c.verify(assembly["outputs"])
    z = a.load("predictions.npz")
    meta = list(a.rows(a.OUT / "metadata.jsonl"))
    le = z["linear_errors"]
    qe = z["qwen_errors"]
    raw = k.raw_counts(le, np.maximum(qe, 0))
    pop = np.array([m["population"] for m in meta])
    gold = z["gold"]
    with np.load(a.REVIEW / "part7/execution_arrays.npz") as f:
        ex = {x: f[x] for x in f.files}
    masks = old.masks(meta, z["qwen_gates"], z["qwen_votes"], ex)
    plans = []
    groups = []
    undefined = []
    draw_paths = []
    populations = sorted(set(pop))
    for population in populations:
        for endpoint in ["FPR", "FNR"]:
            eligible = (pop == population) & (gold == (endpoint == "FNR"))
            for r in range(4):
                plans.append(
                    (
                        f"round_{r+1}",
                        population,
                        endpoint,
                        np.flatnonzero(
                            eligible & (z["qwen_gates"][:, r] >= 0).all((1, 2))
                        ),
                        [r],
                        "text",
                    )
                )
            plans.append(
                (
                    "fixed",
                    population,
                    endpoint,
                    np.flatnonzero(eligible & masks["fixed"]),
                    list(range(4)),
                    "text",
                )
            )
            for name in [
                "strict",
                "strict_fixed",
                "replay_excluded",
                "representative",
                "no_ties",
                "no_repairs",
                "no_transport_retry",
            ]:
                plans.append(
                    (
                        name,
                        population,
                        endpoint,
                        np.flatnonzero(eligible & masks[name]),
                        [3],
                        "text",
                    )
                )
            plans.append(
                (
                    "cluster_equal",
                    population,
                    endpoint,
                    np.flatnonzero(eligible & masks["final"]),
                    [3],
                    "cluster_equal",
                )
            )

    for j, (cohort, population, endpoint, ids, rounds, mode) in enumerate(plans):
        name = f"{cohort}__{population}__{endpoint}"
        ident = dict(
            id=name,
            cohort=cohort,
            population=population,
            endpoint=endpoint,
            rounds=[r + 1 for r in rounds],
            mode=mode,
            indices=ids,
        )
        if not len(ids):
            undefined.append(ident)
            continue
        values = raw[ids][:, rounds].reshape(len(ids), -1)
        labels = [meta[i]["inference_cluster"] for i in ids]
        s = seed(name)
        point, draws, names, sizes = k.bootstrap(
            values, labels, s, mode=mode, draws=10000
        )
        point = point.reshape(len(rounds), 20)
        draws = draws.reshape(10000, len(rounds), 20)
        p, ci = k.intervals(point, draws)
        file = f"draws/{name}.npz"
        a.arrays(file, raw_point=point, raw_draws=draws)
        draw_paths.append(a.OUT / file)
        v = dict(
            **ident,
            inputs=len(ids),
            clusters=len(names),
            cluster_names=names,
            cluster_sizes=sizes,
            seed=s,
            draw_file=file,
            point=p,
            ci95=ci,
        )
        if cohort == "fixed":
            trajectory = []
            expanded = k.expand(draws)
            for before, after in [(0, 3), (0, 1), (1, 2), (2, 3)]:
                delta = expanded[:, after] - expanded[:, before]
                trajectory.append(
                    dict(
                        before=before + 1,
                        after=after + 1,
                        point=p[after] - p[before],
                        ci95=np.moveaxis(
                            np.quantile(delta, [0.025, 0.975], axis=0), 0, -1
                        ),
                    )
                )
            v["trajectories"] = trajectory
        groups.append(v)
        if (j + 1) % 10 == 0:
            print(f"Paired cluster groups: {j+1}/{len(plans)}", flush=True)
    descriptive = []
    for family, field in [
        ("source", "source"),
        ("function", "functionality"),
        ("template", "template_id"),
    ]:
        labels = sorted({str(m[field]) for m in meta if m[field] is not None})
        for label in labels:
            member = np.array(
                [str(m[field]) == label and m[field] is not None for m in meta]
            )
            for population in populations:
                for endpoint in ["FPR", "FNR"]:
                    eligible = (
                        member & (pop == population) & (gold == (endpoint == "FNR"))
                    )
                    if not eligible.any():
                        continue
                    for r in range(4):
                        ids = np.flatnonzero(
                            eligible & (z["qwen_gates"][:, r] >= 0).all((1, 2))
                        )
                        descriptive.append(
                            dict(
                                family=family,
                                label=label,
                                population=population,
                                endpoint=endpoint,
                                round=r + 1,
                                inputs=len(ids),
                                clusters=len(
                                    {meta[i]["inference_cluster"] for i in ids}
                                ),
                                point=(
                                    k.expand(raw[ids, r].mean(0) / 9)
                                    if len(ids)
                                    else None
                                ),
                            )
                        )

    views = []
    definitions = [("all", [0, 1, 2], [0, 1, 2])]
    definitions += [(f"stream_{s}", [s], [0, 1, 2]) for s in range(3)]
    definitions += [
        (f"omit_stream_{s}", [x for x in range(3) if x != s], [0, 1, 2])
        for s in range(3)
    ]
    definitions += [(f"repeat_{s}", [0, 1, 2], [s]) for s in range(3)]
    definitions += [
        (f"omit_repeat_{s}", [0, 1, 2], [x for x in range(3) if x != s])
        for s in range(3)
    ]
    for name, streams, repeats in definitions:
        x = np.empty((len(meta), 20))
        for setting in range(4):
            x[:, setting * 4] = le[:, setting, 3, 0]
            for p in range(1, 4):
                x[:, setting * 4 + p] = le[:, setting, 3][
                    :, [p + 3 * s for s in streams]
                ].mean(1)
        x[:, 16] = qe[:, 3, 0][:, repeats].mean(1)
        for p in range(1, 4):
            x[:, 16 + p] = qe[:, 3][:, [p + 3 * s for s in streams]][
                :, :, repeats
            ].mean((1, 2))
        for population in populations:
            for endpoint in ["FPR", "FNR"]:
                ids = np.flatnonzero(
                    (pop == population) & (gold == (endpoint == "FNR")) & masks["final"]
                )
                views.append(
                    dict(
                        view=name,
                        streams=streams,
                        repeats=repeats,
                        population=population,
                        endpoint=endpoint,
                        inputs=len(ids),
                        point=k.expand(x[ids].mean(0)),
                    )
                )
    omissions = []
    for population in populations:
        for endpoint in ["FPR", "FNR"]:
            ids = np.flatnonzero(
                (pop == population) & (gold == (endpoint == "FNR")) & masks["final"]
            )
            total = raw[ids, 3].sum(0)
            for field in ["inference_cluster", "source", "functionality"]:
                subsets = defaultdict(list)
                for i in ids:
                    if meta[i][field] is not None:
                        subsets[str(meta[i][field])].append(i)
                for label, ix in sorted(subsets.items()):
                    if len(ix) == len(ids):
                        continue
                    omissions.append(
                        dict(
                            population=population,
                            endpoint=endpoint,
                            field=field,
                            omitted=label,
                            removed=len(ix),
                            remaining=len(ids) - len(ix),
                            point=k.expand(
                                (total - raw[ix, 3].sum(0)) / (9 * (len(ids) - len(ix)))
                            ),
                        )
                    )

    lower, upper = k.majority_completion(ex["vote_gates"], gold)
    scenarios = {
        "terminal_null": (
            old.collapse(np.maximum(qe, 0)),
            old.collapse(np.where(qe < 0, 1, qe)),
        ),
        "any_partial_unknown": (
            old.collapse(np.where(z["qwen_votes"] < 3, 0, qe)),
            old.collapse(np.where(z["qwen_votes"] < 3, 1, qe)),
        ),
        "vote_completion": (old.collapse(lower), old.collapse(upper)),
    }
    bounds = []
    for name, (lo, hi) in scenarios.items():
        for population in populations:
            for endpoint in ["FPR", "FNR"]:
                ids = np.flatnonzero(
                    (pop == population) & (gold == (endpoint == "FNR"))
                )
                for r in range(4):
                    intervals = k.bound_metrics(
                        raw[ids, r, :16].mean(0) / 9,
                        lo[ids, r].mean(0) / 9,
                        hi[ids, r].mean(0) / 9,
                    )
                    bounds.append(
                        dict(
                            scenario=name,
                            population=population,
                            endpoint=endpoint,
                            round=r + 1,
                            inputs=len(ids),
                            intervals=intervals,
                        )
                    )
    quality = []
    for cohort in ["round_common", "fixed"]:
        for population in populations:
            for r in range(4):
                mask = (
                    (z["qwen_gates"][:, r] >= 0).all((1, 2))
                    if cohort == "round_common"
                    else masks["fixed"]
                )
                ids = np.flatnonzero((pop == population) & mask)
                for s, (variant, cutoff) in enumerate(a.SETTINGS):
                    for p in range(4):
                        cols = [0] if p == 0 else [p, p + 3, p + 6]
                        lp = z["linear_gates"][ids, s, r][:, cols]
                        qp = z["qwen_gates"][ids, r][:, cols].reshape(len(ids), -1)
                        lc, ls = k.binary_scores(gold[ids], lp)
                        qc, qs = k.binary_scores(gold[ids], qp)
                        quality.append(
                            dict(
                                cohort=cohort,
                                population=population,
                                round=r + 1,
                                setting=s,
                                policy=["reference"] + a.POLICIES,
                                policy_index=p,
                                inputs=len(ids),
                                linear_confusions=lc,
                                qwen_confusions=qc,
                                linear={n: float(v.mean()) for n, v in ls.items()},
                                qwen={n: float(v.mean()) for n, v in qs.items()},
                                linear_per_fit=ls,
                                qwen_per_output=qs,
                            )
                        )
    a.save(
        "analysis.json",
        dict(
            metrics=k.METRICS,
            settings=a.SETTINGS,
            policies=a.POLICIES,
            groups=groups,
            undefined=undefined,
            interval_scope="All supporting nominal 95%; no new primary test family.",
        ),
    )
    a.jsonlines("subgroups.jsonl", descriptive)
    a.jsonlines("stream_repeat_views.jsonl", views)
    a.jsonlines("omissions.jsonl", omissions)
    a.save("bounds.json", bounds)
    a.save("binary_quality.json", quality)
    outputs = [
        a.OUT / n
        for n in [
            "analysis.json",
            "subgroups.jsonl",
            "stream_repeat_views.jsonl",
            "omissions.jsonl",
            "bounds.json",
            "binary_quality.json",
        ]
    ] + draw_paths
    a.save(
        "scored.json",
        dict(
            status="supporting_estimates_complete",
            groups=len(groups),
            undefined=len(undefined),
            descriptive_groups=len(descriptive),
            views=len(views),
            omissions=len(omissions),
            draws_per_group=10000,
            outputs=a.bindings(outputs),
            inputs=a.bindings(
                [
                    a.OUT / n
                    for n in [
                        "assembled.json",
                        "analysis_contract.json",
                        "predictions.npz",
                        "metadata.jsonl",
                    ]
                ]
            ),
        ),
    )
    print(
        "Saved all paired estimates, subgroup views, bounds and binary metrics.",
        flush=True,
    )


if __name__ == "__main__":
    with a.c.offline(), threadpool_limits(limits=1):
        main()

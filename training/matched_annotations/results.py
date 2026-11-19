from training.matched_annotations import validate as r
from training.roberta import results as older
from collections import Counter, defaultdict
import argparse
import hashlib
from pathlib import Path
import socket
from unittest.mock import patch
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

c = r.c
m = r.m
previous = older.previous
METHODS = ["bge", "frozen_roberta"] + [
    f"{family}_{policy}" for family in r.FAMILIES for policy in ("selected", "default")
]
PRIMARY = ["tuned_selected", "bge", "frozen_roberta"]


def summary(y, x, ep):
    stats, counts = previous.summarize_bits(y, x, ep)
    stats.update(
        mean_extra_labels=float((x & ~y).sum(1).mean()),
        mean_missing_labels=float((~x & y).sum(1).mean()),
        empty_prediction_rate=float((~x.any(1)).mean()),
    )
    return stats, counts


def arrays(name, pool):
    d = m.data(pool)
    sel = c.read(r.BASE / f"selections/{pool}.json")["families"]
    linear = c.read(r.BASE / f"linear/{pool}/fits.json")
    if name in ("bge", "frozen_roberta"):
        key = "bge" if name == "bge" else "roberta"
        fit = linear["bge_imported" if key == "bge" else key]
        with np.load(
            r.BASE / f"linear/{pool}/probabilities.npz", allow_pickle=False
        ) as z:
            p = z[key]
        with np.load(r.BASE / "data/evaluation_features.npz", allow_pickle=False) as z:
            gate = z["gate"].astype(bool)
        raw = p >= fit["selected"]["threshold"]
        pred = raw.copy()
        pred[~gate] = False
        return [(None, p, raw, pred, fit["selected"])]
    family, policy = name.rsplit("_", 1)
    chosen = sel[family]["selected"]
    cfg = chosen["config"]
    out = []
    for seed in c.FINAL_SEEDS:
        p = c.probabilities(
            "hate_type",
            np.load(
                r.BASE / f"inference/{pool}/hate_type/{cfg}/s{seed}/logits.npy",
                allow_pickle=False,
            ),
        )
        with np.load(
            r.BASE / f"predictions/{pool}/{name}_s{seed}.npz", allow_pickle=False
        ) as z:
            raw = z["ungated"].astype(bool)
            pred = z["delivered"].astype(bool)
        out.append(
            (
                seed,
                p,
                raw,
                pred,
                dict(
                    config=cfg,
                    threshold=chosen["threshold"] if policy == "selected" else 0.5,
                ),
            )
        )
    return out


def means(rows):
    fields = [
        "macro_f1",
        "exact_set_accuracy",
        "mean_predicted_labels",
        "mean_gold_labels",
        "mean_extra_labels",
        "mean_missing_labels",
        "empty_prediction_rate",
    ]
    return {k: float(np.mean([v[k] for v in rows])) for k in fields}


def score():
    c.require(
        c.read(r.OUT / "integrity.json")["status"] == "pass"
        and c.read(r.OUT / "linear_audit.json")["status"] == "pass",
        "Raw/rebuilt fits must pass first",
    )
    inputrows = c.read(r.BASE / "data/evaluation_inputs.json")
    positions = {v["key"]: i for i, v in enumerate(inputrows)}
    endpoints = c.read(r.BASE / "data/endpoints.json")
    cells = {}
    cache = {}
    slices = {}
    bindings = {}
    reproduced = 0
    coretexts = {
        previous.norm(v["text"])
        for v in previous.rows(c.CORE / "dev_inputs.jsonl")
        if v["lang"] == "en"
    }
    prepared = c.read(m.feature.BASE / "plan.json")["inputs"]
    with np.load(r.BASE / "data/evaluation_features.npz", allow_pickle=False) as z:
        globalgate = z["gate"].astype(bool)
    for endpoint, rec in endpoints.items():
        ep = rec["endpoint"]
        scope, name = endpoint.split("/", 1)
        goldpath = (
            c.ROOT / f"data/processed/{name}.parquet"
            if scope == "core"
            else previous.PREP / "broad_roster.jsonl"
        )
        c.require(c.sha(goldpath) == prepared[c.rel(goldpath)], "Gold source changed")
        bindings[c.rel(goldpath)] = c.sha(goldpath)
        rr, kept, ids, y = older.gold(scope, name, ep)
        idx = [positions[k] for k in rec["keys"]]
        c.require(
            len(kept) == len(idx)
            and all(inputrows[j]["text"] == v["text"] for j, v in zip(idx, kept)),
            "Endpoint text alignment differs",
        )
        gate = globalgate[idx]
        runs = {}
        poolmeans = {}
        groups = {}
        rankings = {}
        masks = {
            f"source:{s}": np.asarray([v["source"] == s for v in kept])
            for s in sorted({v["source"] for v in kept})
        }
        if scope == "broad":
            masks["outside_original_core"] = np.asarray(
                [previous.norm(v["text"]) not in coretexts for v in kept]
            )
        lengths = np.asarray([len(v["text"].split()) for v in kept])
        masks.update(
            words_1_to_20=lengths <= 20,
            words_21_to_40=(lengths > 20) & (lengths <= 40),
            words_over_40=lengths > 40,
        )
        labelorder = m.data(m.POOL_NAMES[0])["heads"]["hate_type"]["labels"]
        order = [labelorder.index(l) for l in ep["all_labels"]]
        for pool in m.POOL_NAMES:
            runs[pool] = {}
            poolmeans[pool] = {}
            rankings[pool] = {}
            for method in METHODS:
                records = []
                rank = []
                for seed, prob, raw, pred, choice in arrays(method, pool):
                    x = pred[idx][:, order]
                    u = raw[idx][:, order]
                    prob = prob[idx][:, order]
                    stats, counts = summary(y, x, ep)
                    head, _ = summary(y, u, ep)
                    stats.update(seed=seed, head_only=head, choice=choice)
                    records.append(stats)
                    perlabel = {}
                    for j, label in enumerate(ep["all_labels"]):
                        if y[:, j].any() and (~y[:, j]).any():
                            perlabel[label] = dict(
                                auc=float(roc_auc_score(y[:, j], prob[:, j])),
                                ap=float(average_precision_score(y[:, j], prob[:, j])),
                            )
                    rank.append(
                        dict(
                            seed=seed,
                            macro_auc=float(
                                np.mean([v["auc"] for v in perlabel.values()])
                            ),
                            macro_ap=float(
                                np.mean([v["ap"] for v in perlabel.values()])
                            ),
                            per_label=perlabel,
                        )
                    )
                    if method in PRIMARY:
                        slices[f"{endpoint}|{pool}|{method}|{seed}"] = {
                            k: summary(y[mask], x[mask], ep)[0]
                            for k, mask in masks.items()
                            if mask.any()
                        }
                        if scope == "broad":
                            cache[f"{pool}|{method}|{seed}"] = counts
                    if method == "bge" and scope == "core":
                        old = c.read(
                            (c.ROOT / m.data(pool)["source_bge_fit"]).parent
                            / "scores.json"
                        )["endpoints"][name]
                        for new, oldkey in [
                            (stats, "result"),
                            (head, "ungated_diagnostic"),
                        ]:
                            want = old[oldkey]
                            c.require(
                                abs(new["macro_f1"] - want["macro_f1_valid_only"])
                                < 1e-12
                                and abs(
                                    new["exact_set_accuracy"]
                                    - want["exact_set_accuracy"]
                                )
                                < 1e-12
                                and abs(
                                    new["mean_predicted_labels"]
                                    - want["mean_predicted_label_count"]
                                )
                                < 1e-12,
                                "Historical BGE endpoint differs",
                            )
                            for label, v in new["per_label"].items():
                                c.require(
                                    all(
                                        v[k] == want["per_label"][label][k]
                                        for k in ("support", "tp", "fp", "fn")
                                    ),
                                    "Historical label confusion differs",
                                )
                            reproduced += 1
                runs[pool][method] = records
                poolmeans[pool][method] = means(records)
                poolmeans[pool][method].update(
                    seed_sd_f1=(
                        float(np.std([v["macro_f1"] for v in records], ddof=1))
                        if len(records) > 1
                        else None
                    ),
                    seed_min_f1=min(v["macro_f1"] for v in records),
                    seed_max_f1=max(v["macro_f1"] for v in records),
                    head_only_f1=float(
                        np.mean([v["head_only"]["macro_f1"] for v in records])
                    ),
                )
                rankings[pool][method] = rank
        for group, pools in r.GROUPS.items():
            groups[group] = {}
            for method in METHODS:
                vv = [poolmeans[p][method] for p in pools]
                groups[group][method] = means(vv)
                groups[group][method].update(
                    pool_sd_f1=float(np.std([v["macro_f1"] for v in vv], ddof=1)),
                    pool_min_f1=min(v["macro_f1"] for v in vv),
                    pool_max_f1=max(v["macro_f1"] for v in vv),
                    mean_seed_sd_f1=(
                        float(np.mean([v["seed_sd_f1"] for v in vv]))
                        if vv[0]["seed_sd_f1"] is not None
                        else None
                    ),
                    head_only_f1=float(np.mean([v["head_only_f1"] for v in vv])),
                    macro_auc=float(
                        np.mean(
                            [s["macro_auc"] for p in pools for s in rankings[p][method]]
                        )
                    ),
                    macro_ap=float(
                        np.mean(
                            [s["macro_ap"] for p in pools for s in rankings[p][method]]
                        )
                    ),
                )
        cells[endpoint] = dict(
            n=len(y),
            gold_support=ep["gold_support"],
            labels=ep["all_labels"],
            macro_labels=ep["macro_labels"],
            gate_negative_rows=int((~gate).sum()),
            gate_suppression_fraction=float((~gate).mean()),
            sources=dict(Counter(v["source"] for v in kept)),
            normalized_clusters=len({previous.norm(v["text"]) for v in kept}),
            slice_counts={k: int(mask.sum()) for k, mask in masks.items()},
            runs=runs,
            pool_means=poolmeans,
            group_means=groups,
            rankings=rankings,
        )
        print(
            endpoint,
            {
                g: {a: round(b["macro_f1"], 6) for a, b in v.items()}
                for g, v in groups.items()
            },
            flush=True,
        )
    target = r.OUT / "primary_counts.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as z:
            c.require(
                set(z.files) == set(cache)
                and all(np.array_equal(z[k], v) for k, v in cache.items()),
                "Prior counts differ",
            )
    else:
        np.savez_compressed(target, **cache)
    r.save(
        "scores.json",
        dict(
            status="scored",
            cells=cells,
            old_bge_core_scores_reproduced=reproduced,
            gold_bindings=bindings,
            source_sha256=c.sha(Path(__file__)),
            score_cells=sum(
                sum(
                    len(rs)
                    for methods in cell["runs"].values()
                    for rs in methods.values()
                )
                for cell in cells.values()
            ),
        ),
    )
    r.save(
        "slices.json",
        dict(exploratory=True, fixed_label_denominator=True, cells=slices),
    )


def f1counts(counts, indices):
    tp, fp, fn = np.moveaxis(np.asarray(counts), -1, 0)
    den = 2 * tp + fp + fn
    return np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den > 0)[
        ..., indices
    ].mean(-1)


def paired(
    counts, clusters, sources, indices, *, draws=10000, seed=20261012, stratified=False
):

    c.require(
        counts.ndim == 5
        and counts.shape[:2] == (5, 7)
        and counts.shape[2] == len(clusters),
        "Bootstrap axes differ",
    )
    names = sorted(set(clusters))
    lookup = {s: i for i, s in enumerate(names)}
    agg = np.zeros((len(names), 5, 7, counts.shape[-2], 3), np.int64)
    members = defaultdict(set)
    for i, k in enumerate(clusters):
        agg[lookup[k]] += counts[:, :, i]
        members[k].add(sources[i])
    strata = defaultdict(list)
    for k, j in lookup.items():
        strata[tuple(sorted(members[k])) if stratified else ("all",)].append(j)
    rng = np.random.default_rng(seed)
    rngpool = np.random.default_rng(seed + 1)
    flat = agg.reshape(len(names), -1).astype(float)
    values = []
    joint = []
    for start in range(0, draws, 64):
        n = min(64, draws - start)
        picks = np.concatenate(
            [
                rng.choice(g, (n, len(g)), replace=True)
                for _, g in sorted(strata.items())
            ],
            axis=1,
        )
        weights = np.asarray(
            [np.bincount(p, minlength=len(names)) for p in picks], dtype=float
        )
        totals = (weights @ flat).reshape(n, 5, 7, counts.shape[-2], 3)
        c.require(
            np.array_equal(totals[0], agg[picks[0]].sum(0)),
            "Direct cluster resample disagrees",
        )
        scores = f1counts(totals, indices)
        diff = np.stack(
            (
                scores[:, :, :5].mean(-1) - scores[:, :, 5],
                scores[:, :, 6] - scores[:, :, 5],
            ),
            axis=-1,
        )
        values.extend(diff.mean(1).tolist())
        poolpicks = rngpool.integers(0, 5, size=(n, 5))
        joint.extend(
            np.take_along_axis(diff, poolpicks[:, :, None], axis=1).mean(1).tolist()
        )
    observed = f1counts(counts.sum(2), indices)
    delta = np.stack(
        (observed[:, :5].mean(-1) - observed[:, 5], observed[:, 6] - observed[:, 5]),
        axis=-1,
    )
    values = np.asarray(values)
    joint = np.asarray(joint)
    out = {}
    for i, name in enumerate(("tuned_selected_minus_bge", "frozen_roberta_minus_bge")):
        out[name] = dict(
            difference=float(delta[:, i].mean()),
            by_pool=delta[:, i].tolist(),
            ci9875=np.quantile(values[:, i], [0.00625, 0.99375]).tolist(),
            ci95=np.quantile(values[:, i], [0.025, 0.975]).tolist(),
            joint_pool_text_ci95=np.quantile(joint[:, i], [0.025, 0.975]).tolist(),
            leave_one_pool_out=[
                float(np.delete(delta[:, i], j).mean()) for j in range(5)
            ],
            pools_positive=int((delta[:, i] > 0).sum()),
            draws=draws,
            seed=seed,
            clusters=len(names),
            rows=len(clusters),
            stratified=stratified,
        )
    return out


def statistics():
    c.require(
        c.read(r.OUT / "statistics_tests.json")["status"] == "pass",
        "Bootstrap unit checks required",
    )
    ep = c.read(r.BASE / "data/endpoints.json")["broad/en_test/hate_type"]["endpoint"]
    _, kept, ids, y = older.gold("broad", "en_test/hate_type", ep)
    scores = c.read(r.OUT / "scores.json")
    indices = [ep["all_labels"].index(v) for v in ep["macro_labels"]]
    clusters = [previous.norm(v["text"]) for v in kept]
    sources = [v["source"] for v in kept]
    out = {}
    policy = c.read(r.OUT / "analysis_plan.json")
    bs = policy["bootstrap"]
    with np.load(r.OUT / "primary_counts.npz", allow_pickle=False) as z:
        for group, pools in r.GROUPS.items():
            data = np.stack(
                [
                    np.stack(
                        [
                            z[f"{p}|{method}|{seed}"]
                            for method, seeds in [
                                ("tuned_selected", c.FINAL_SEEDS),
                                ("bge", [None]),
                                ("frozen_roberta", [None]),
                            ]
                            for seed in seeds
                        ]
                    )
                    for p in pools
                ]
            )
            ordinary = paired(
                data, clusters, sources, indices, draws=bs["draws"], seed=bs["seed"]
            )
            stratified = paired(
                data,
                clusters,
                sources,
                indices,
                draws=bs["draws"],
                seed=bs["seed"],
                stratified=True,
            )
            actual = scores["cells"]["broad/en_test/hate_type"]["group_means"][group]
            for k, item in ordinary.items():
                left = k.removesuffix("_minus_bge")
                c.require(
                    abs(
                        actual[left]["macro_f1"]
                        - actual["bge"]["macro_f1"]
                        - item["difference"]
                    )
                    < 1e-12,
                    "Bootstrap estimand differs from fit-average scores",
                )
            out[group] = dict(
                primary=ordinary, source_stratified_sensitivity=stratified
            )
            print("Paired uncertainty:", group, ordinary, flush=True)
    r.save(
        "statistics.json",
        dict(
            status="pass",
            primary_family_size=4,
            confidence=0.9875,
            conditional_on_fixed_fits_and_pools=True,
            exploratory_joint_pool_intervals=True,
            prior_evaluation_exposure=True,
            groups=out,
            source_sha256=c.sha(Path(__file__)),
        ),
    )


def diagnostics():
    training = c.read(r.OUT / "training_runs.json")
    scores = c.read(r.OUT / "scores.json")
    runs = training["runs"]
    curves = {}
    selections = {}
    for pool in m.POOL_NAMES:
        candidates = {}
        for cfg in c.config_map():
            fits = [
                v
                for v in runs
                if v["job"]["pool"] == pool
                and v["job"]["mode"] == "search"
                and v["job"]["config"] == cfg
            ]
            c.require(len(fits) == 15, "Curve fit inventory differs")
            candidates[cfg] = dict(
                mean_first_held_loss=float(
                    np.mean([v["first_epoch"]["held_unweighted_loss"] for v in fits])
                ),
                mean_last_held_loss=float(
                    np.mean([v["last_epoch"]["held_unweighted_loss"] for v in fits])
                ),
                mean_first_objective=float(
                    np.mean([v["first_epoch"]["training_objective"] for v in fits])
                ),
                mean_last_objective=float(
                    np.mean([v["last_epoch"]["training_objective"] for v in fits])
                ),
                last_worse_than_first=sum(
                    v["last_epoch"]["held_unweighted_loss"]
                    > v["first_epoch"]["held_unweighted_loss"]
                    for v in fits
                ),
                best_held_loss_epochs=[v["minimum_held_loss_epoch"] for v in fits],
                skipped_steps=sum(v["skipped_steps"] for v in fits),
            )
        curves[pool] = candidates
        sel = c.read(r.BASE / f"selections/{pool}.json")["families"]
        lin = c.read(r.BASE / f"linear/{pool}/fits.json")
        selections[pool] = {k: v["selected"] for k, v in sel.items()} | {
            k: lin[n]["selected"]
            for k, n in [("bge", "bge_imported"), ("frozen_roberta", "roberta")]
        }
    inventory = c.read(r.BASE / "data/inventory.json")
    support = {}
    for group, pools in r.GROUPS.items():
        support[group] = {
            lab: [inventory[p]["labels"][lab] for p in pools]
            for lab in inventory[pools[0]]["labels"]
        }
    budgets = {
        mode: dict(
            fits=sum(v["job"]["mode"] == mode for v in runs),
            train_seconds=sum(
                v["train_seconds"] for v in runs if v["job"]["mode"] == mode
            ),
            total_seconds=sum(
                v["total_seconds"] for v in runs if v["job"]["mode"] == mode
            ),
            steps=sum(v["attempted_steps"] for v in runs if v["job"]["mode"] == mode),
            skipped_steps=sum(
                v["skipped_steps"] for v in runs if v["job"]["mode"] == mode
            ),
        )
        for mode in ("search", "final")
    }
    r.save(
        "diagnostics.json",
        dict(
            status="pass",
            selections=selections,
            curves=curves,
            training_support=support,
            costs=budgets,
            peak_allocated_gib=max(v["peak_allocated_gib"] for v in runs),
            attempt_directories=sum(len(v["attempt_directories"]) for v in runs),
            nonselected_attempts=[
                a for v in runs for a in v["attempt_directories"] if not a["selected"]
            ],
            maximum_recorded_export_difference=max(
                v["export_check"]["max_abs_logit_difference"]
                for v in runs
                if v["job"]["mode"] == "final"
            ),
            reused_final_config_pools=[
                p for p, v in training["pools"].items() if v["final_fits"] < 15
            ],
            source_sha256=c.sha(Path(__file__)),
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["score", "statistics", "diagnostics"])
    args = ap.parse_args()
    with patch.object(socket.socket, "connect", r.forbid), patch.object(
        socket, "create_connection", r.forbid
    ):
        globals()[args.mode]()


if __name__ == "__main__":
    main()

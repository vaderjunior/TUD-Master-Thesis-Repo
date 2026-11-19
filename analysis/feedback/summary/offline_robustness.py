import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from analysis.common import replay as c
from analysis.classification.supervised.operating_points import independent_rank

BASE = c.BASE / "sq3_robustness"
OLD = c.ROOT / "experiments/astra/sq3_combined_results_20261004_v1"
EX = c.ROOT / "experiments/astra/sq3_full_kb_explanations_20261001_v1"
TAX = c.ROOT / "experiments/astra/sq3_full_kb_taxonomy_20261001_v1"
FEEDBACK = c.ROOT / "experiments/astra/sq3_full_kb_feedback_20260928_v1"
SEEDS = [202609281, 202609282, 202609283]
LABELS = [
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
]
BINS = [-1.0, 0.4, 0.5, 0.6, 0.64, 0.7, 1.0000001]


def prepare():
    paths = [
        Path(__file__).resolve(),
        Path(c.__file__).resolve(),
        c.ROOT / "analysis/classification/supervised/operating_points.py",
        OLD / "review/predictions.jsonl",
        OLD / "review/complete.json",
        OLD / "taxonomy/primary.json",
        OLD / "explanations/all_transfer.json",
        OLD / "explanations/benign.json",
        EX / "roster.jsonl",
        EX / "choices.jsonl",
        TAX / "roster.jsonl",
        FEEDBACK / "feasibility_v1/roster.jsonl",
        FEEDBACK / "update_payloads_v1/pools.json",
    ]
    c.verify(c.read(OLD / "review/complete.json")["outputs"])
    c.save(
        BASE / "plan.json",
        {
            "created_utc": c.utc(),
            "status": "post_result_descriptive_plan_before_new_calculation",
            "inputs": {c.rel(p): c.sha(p) for p in paths},
            "questions": [
                "Does restored-category F1 depend on one category or one decoder repeat?",
                "Does higher cosine similarity rank corrections whose annotated types are compatible with the query?",
                "How concentrated are correction exposure and error changes across the fixed correction sources?",
            ],
            "taxonomy": "Original 466 common-valid texts, original clusters, all seven labels and three repeats. Reproduce old primary, then report all seven leave-one-category and all three leave-one-repeat results. 10000 paired cluster resamples using original seed 2026100103; new 95% intervals descriptive only.",
            "corrections": "Original all-policy common 349 hateful and 203 benign inputs from the completed explanation study. Three fixed error-selected 40-source pools and all three decoder repeats. Compare plain_always with its own fresh reference; not the old error-minus-correct primary.",
            "compatibility": "On gold-hateful annotated queries only: any shared type, all source types compatible (source set subset of query set), and all query types covered. Per-pool AP/ROC of saved cosine, with prevalence and fixed bins [-1,.4,.5,.6,.64,.7,1]. Gold is diagnostic only, never a new deployment filter.",
            "source_influence": "Keep all 40 source records per pool, including unused. Report every source; top-five display by usage only. Leave-one-source-associated-query-out drops paired inputs receiving that source separately within each pool, then averages fixed pools/repeats. This is NOT removing a KB source and reretrieving or a counterfactual estimate of a better correction policy.",
            "limits": "Descriptive robustness, no new primary significance claim, no tuned cutoff or outcome-based source pruning. Shared inputs/pools are not independent annotation judgments. No partial balanced-correction outputs, no humans, no calls.",
        },
    )
    print("Additional SQ3 diagnostic plan frozen.", flush=True)


def bitcounts(y, x):
    return np.stack([y * x, (1 - y) * x, y * (1 - x)], axis=-1)


def f1(counts):
    den = 2 * counts[..., 0] + counts[..., 1] + counts[..., 2]
    return np.divide(
        2 * counts[..., 0], den, out=np.zeros_like(den, dtype=float), where=den != 0
    )


def inputs():
    predictions = {}
    allowed = {
        "taxonomy": {"rag_full", *[f"rag_schema_{l}" for l in LABELS]},
        "explanations": {"reference", *[f"plain_always_s{s}" for s in SEEDS]},
    }
    with (OLD / "review/predictions.jsonl").open(encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] in allowed[r["study"]]:
                key = (r["study"], r["condition"], r["repeat"], r["id"])
                assert key not in predictions
                predictions[key] = r["result"]
    roster = {
        name: {r["id"]: r for r in c.rows(path / "roster.jsonl")}
        for name, path in [("taxonomy", TAX), ("explanations", EX)]
    }
    return (predictions, roster)


def taxonomy(pred, roster):
    old = c.read(OLD / "taxonomy/primary.json")
    ids = old["common_valid_ids"]
    n = len(ids)
    assert n == 466
    arr = np.zeros((2, 7, 3, n, 3), dtype=np.int64)
    for lab, l in enumerate(LABELS):
        y = np.array([l in roster[i]["gold"]["hate_type"] for i in ids], int)
        for mode, cond in enumerate(["rag_full", "rag_schema_" + l]):
            for rep in range(3):
                pp = [pred["taxonomy", cond, rep, i] for i in ids]
                assert all((p is not None for p in pp))
                x = np.array([l in p["hate_type"] for p in pp], int)
                arr[mode, lab, rep] = bitcounts(y, x)
    points = f1(arr.sum(axis=3))
    delta = points[0] - points[1]
    primary = float(delta.mean())
    assert abs(primary - old["primary"]["difference"]) < 1e-12
    for lab, l in enumerate(LABELS):
        assert abs(delta[lab].mean() - old["contrasts"][l]["difference"]) < 1e-12
    clusters = sorted({roster[i]["cluster"] for i in ids})
    index = {x: n for n, x in enumerate(clusters)}
    cc = np.zeros((2, 7, 3, len(clusters), 3), dtype=np.int64)
    for j, i in enumerate(ids):
        cc[:, :, :, index[roster[i]["cluster"]]] += arr[:, :, :, j]
    rng = np.random.default_rng(2026100103)
    draws = []
    for _ in range(10000):
        chosen = rng.integers(0, len(clusters), len(clusters))
        score = f1(cc[:, :, :, chosen].sum(axis=3))
        draws.append(score[0] - score[1])
    draws = np.asarray(draws)
    assert np.allclose(
        np.quantile(draws.mean(axis=(1, 2)), [0.00625, 0.99375]),
        old["primary"]["ci9875"],
        rtol=0,
        atol=1e-12,
    )
    out = {
        "common_n": n,
        "clusters": len(clusters),
        "primary_reproduced": primary,
        "old_adjusted_ci_reproduced": old["primary"]["ci9875"],
        "per_category": {
            l: {
                "difference": float(delta[k].mean()),
                "repeat_differences": delta[k].tolist(),
            }
            for k, l in enumerate(LABELS)
        },
        "omit_category": {},
        "omit_repeat": {},
        "full_target_by_repeat": points[0].mean(axis=0).tolist(),
        "schema_target_by_repeat": points[1].mean(axis=0).tolist(),
    }
    for k, l in enumerate(LABELS):
        keep = [i for i in range(7) if i != k]
        out["omit_category"][l] = {
            "difference": float(delta[keep].mean()),
            "ci95_descriptive": np.quantile(
                draws[:, keep].mean(axis=(1, 2)), [0.025, 0.975]
            ).tolist(),
        }
    for rep in range(3):
        keep = [r for r in range(3) if r != rep]
        out["omit_repeat"][str(rep)] = {
            "difference": float(delta[:, keep].mean()),
            "ci95_descriptive": np.quantile(
                draws[:, :, keep].mean(axis=(1, 2)), [0.025, 0.975]
            ).tolist(),
        }
    c.save(BASE / "taxonomy_influence.json", out)
    return out


def corrections(pred, roster):
    old = c.read(OLD / "explanations/all_transfer.json")
    ids = old["common_valid_ids"]
    benign = c.read(OLD / "explanations/benign.json")["common_valid_ids"]
    assert len(ids) == 349 and len(benign) == 203
    choices = {(r["pool_seed"], r["query_id"]): r for r in c.rows(EX / "choices.jsonl")}
    sources = {r["id"]: r for r in c.rows(FEEDBACK / "feasibility_v1/roster.jsonl")}
    pools = {
        r["seed"]: r["ids"]
        for r in c.read(FEEDBACK / "update_payloads_v1/pools.json")["pools"]
        if r["arm"] == "error" and r["size"] == 40
    }
    assert set(pools) == set(SEEDS) and all((len(v) == 40 for v in pools.values()))
    y = np.array(
        [[l in roster[i]["gold"]["hate_type"] for l in LABELS] for i in ids], int
    )
    bypool = {}
    records = []
    benign_rows = []
    ranking = {}
    per_source = []
    for seed in SEEDS:
        cnd = f"plain_always_s{seed}"
        xx = []
        for cond in ["reference", cnd]:
            for rep in range(3):
                pp = [pred["explanations", cond, rep, i] for i in ids]
                assert all((p is not None for p in pp))
                xx.append(
                    np.array([[l in p["hate_type"] for l in LABELS] for p in pp], int)
                )
        counts = bitcounts(y[None], np.array(xx)).reshape(2, 3, len(ids), 7, 3)
        points = f1(counts.sum(axis=2)).mean(axis=-1)
        for mode, cond in enumerate(["reference", cnd]):
            for rep in range(3):
                assert (
                    abs(
                        points[mode, rep]
                        - old["settings"][f"{cond}_rep{rep}"]["common_valid"][
                            "macro_f1"
                        ]
                    )
                    < 1e-12
                )
        bypool[seed] = {
            "counts": counts,
            "sources": np.array([choices[seed, i]["source_id"] for i in ids]),
            "points": points,
        }
        poolrows = []
        for n, i in enumerate(ids):
            choice = choices[seed, i]
            sid = choice["source_id"]
            assert sid in pools[seed]
            sg = set(sources[sid]["gold"]["hate_type"])
            qg = set(roster[i]["gold"]["hate_type"])
            assert sg and qg
            delta = counts[1, :, n] - counts[0, :, n]
            row = {
                "query_id": i,
                "pool_seed": seed,
                "source_id": sid,
                "cosine": choice["cosine"],
                "selected_at_frozen_cutoff": choice["apply"],
                "any_shared_type": bool(sg & qg),
                "all_source_types_compatible": sg <= qg,
                "all_query_types_covered": qg <= sg,
                "source_types": sorted(sg),
                "query_types": sorted(qg),
                "correct_label_change_by_repeat": delta[:, :, 0].sum(axis=1).tolist(),
                "extra_label_change_by_repeat": delta[:, :, 1].sum(axis=1).tolist(),
                "missed_hate_change_by_repeat": [
                    int(not pred["explanations", cnd, r, i]["hate"])
                    - int(not pred["explanations", "reference", r, i]["hate"])
                    for r in range(3)
                ],
            }
            records.append(row)
            poolrows.append(row)
        rank = {}
        for label in [
            "any_shared_type",
            "all_source_types_compatible",
            "all_query_types_covered",
        ]:
            yy = np.array([r[label] for r in poolrows], int)
            ss = np.array([r["cosine"] for r in poolrows])
            ap, auc = independent_rank(yy, ss)
            assert (
                abs(ap - average_precision_score(yy, ss)) < 1e-12
                and abs(auc - roc_auc_score(yy, ss)) < 1e-12
            )
            rank[label] = {
                "positive": int(yy.sum()),
                "n": len(yy),
                "prevalence": float(yy.mean()),
                "average_precision": ap,
                "roc_auc": auc,
            }
        ranking[str(seed)] = rank
        for sid in pools[seed]:
            used = [r for r in poolrows if r["source_id"] == sid]
            per_source.append(
                {
                    "pool_seed": seed,
                    "source_id": sid,
                    "query_pool_assignments": len(used),
                    "source_types": sources[sid]["gold"]["hate_type"],
                    "compatible_assignments": sum(
                        (r["all_source_types_compatible"] for r in used)
                    ),
                    "sum_correct_change_mean_over_repeats": sum(
                        (np.mean(r["correct_label_change_by_repeat"]) for r in used)
                    ),
                    "sum_extra_change_mean_over_repeats": sum(
                        (np.mean(r["extra_label_change_by_repeat"]) for r in used)
                    ),
                }
            )
        for i in benign:
            choice = choices[seed, i]
            a = [pred["explanations", "reference", r, i] for r in range(3)]
            b = [pred["explanations", cnd, r, i] for r in range(3)]
            assert all((p is not None for p in a + b))
            benign_rows.append(
                {
                    "query_id": i,
                    "pool_seed": seed,
                    "source_id": choice["source_id"],
                    "cosine": choice["cosine"],
                    "false_positive_change_by_repeat": [
                        int(b[r]["hate"]) - int(a[r]["hate"]) for r in range(3)
                    ],
                }
            )
    source_ids = sorted(set((sid for p in pools.values() for sid in p)))
    assert len(source_ids) == 58
    mean_delta = float(
        np.mean(
            [v["points"][1].mean() - v["points"][0].mean() for v in bypool.values()]
        )
    )
    assert (
        abs(
            mean_delta
            - (
                old["groups"]["plain_always"]["macro_f1"]
                - old["groups"]["reference"]["macro_f1"]
            )
        )
        < 1e-12
    )
    leave = []
    for sid in source_ids:
        runs = []
        for seed in SEEDS:
            state = bypool[seed]
            keep = state["sources"] != sid
            scores = f1(state["counts"][:, :, keep].sum(axis=2)).mean(axis=-1)
            runs.append(
                {
                    "pool_seed": seed,
                    "retained": int(keep.sum()),
                    "excluded": int((~keep).sum()),
                    "reference_by_repeat": scores[0].tolist(),
                    "plain_by_repeat": scores[1].tolist(),
                    "difference": float(scores[1].mean() - scores[0].mean()),
                }
            )
        leave.append(
            {
                "source_id": sid,
                "difference": float(np.mean([r["difference"] for r in runs])),
                "pools": runs,
            }
        )
    bins = {}
    for name, rows, fields in [
        (
            "hateful",
            records,
            [
                "correct_label_change_by_repeat",
                "extra_label_change_by_repeat",
                "missed_hate_change_by_repeat",
            ],
        ),
        ("benign", benign_rows, ["false_positive_change_by_repeat"]),
    ]:
        binned = []
        for lo, hi in zip(BINS, BINS[1:]):
            take = [r for r in rows if lo <= r["cosine"] < hi]
            row = {
                "lower_inclusive": lo,
                "upper_exclusive": hi,
                "query_pool_pairs": len(take),
                "unique_queries": len({r["query_id"] for r in take}),
            }
            if name == "hateful":
                row["compatible_rate"] = (
                    float(np.mean([r["all_source_types_compatible"] for r in take]))
                    if take
                    else None
                )
            for f in fields:
                row[f.replace("_by_repeat", "_mean")] = (
                    float(np.mean([r[f] for r in take])) if take else None
                )
            binned.append(row)
        assert sum((r["query_pool_pairs"] for r in binned)) == len(rows)
        bins[name] = binned
    summary = {
        "hateful_n": len(ids),
        "benign_n": len(benign),
        "source_union": len(source_ids),
        "source_pool_records": len(per_source),
        "unused_source_pool_records": sum(
            (r["query_pool_assignments"] == 0 for r in per_source)
        ),
        "plain_minus_reference_reproduced": mean_delta,
        "ranking_by_pool": ranking,
        "ranking_mean": {
            label: {
                k: float(np.mean([v[label][k] for v in ranking.values()]))
                for k in ["prevalence", "average_precision", "roc_auc"]
            }
            for label in ranking[str(SEEDS[0])]
        },
        "leave_source_associated_queries_out": leave,
        "fixed_cosine_bins": bins,
        "per_source_pool": per_source,
        "leave_source_difference_range": [
            min((r["difference"] for r in leave)),
            max((r["difference"] for r in leave)),
        ],
        "interpretation": "Same known source pool and outcomes, association only. Dropped-query diagnostics do not predict source removal, reretrieval or new policy performance.",
    }
    c.save_rows(BASE / "correction_hateful_pairs.jsonl", records)
    c.save_rows(BASE / "correction_benign_pairs.jsonl", benign_rows)
    c.save(BASE / "correction_compatibility_influence.json", summary)
    return summary


def run():
    c.verify(c.read(BASE / "plan.json")["inputs"])
    pred, roster = inputs()
    tax = taxonomy(pred, roster["taxonomy"])
    co = corrections(pred, roster["explanations"])
    output = [
        "taxonomy_influence.json",
        "correction_compatibility_influence.json",
        "correction_hateful_pairs.jsonl",
        "correction_benign_pairs.jsonl",
    ]
    c.save(
        BASE / "complete.json",
        {
            "status": "offline_descriptive_analysis_complete",
            "created_utc": c.utc(),
            "plan_sha256": c.sha(BASE / "plan.json"),
            "outputs": {c.rel(BASE / p): c.sha(BASE / p) for p in output},
            "new_model_calls": 0,
        },
    )
    print(
        "Taxonomy leave-category range",
        min((x["difference"] for x in tax["omit_category"].values())),
        max((x["difference"] for x in tax["omit_category"].values())),
    )
    print("Compatibility ranking:", co["ranking_mean"])
    print("Source-associated-query omission range", co["leave_source_difference_range"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "run"])
    args = ap.parse_args()
    with c.network_guard():
        prepare() if args.mode == "prepare" else run()

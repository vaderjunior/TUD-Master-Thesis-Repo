import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import yaml

from analysis.classification.prototype.type_scores import METADATA
from analysis.classification.prototype.type_scores import clean
from analysis.classification.prototype.type_scores import require
from analysis.classification.prototype.type_scores import sha
from src.hsrag.metrics import MIN_SUPPORT, score_gate, score_multilabel

SRC = {
    "zero_shot": ("hatecheck_A_r{}", "zero_shot"),
    "few_shot": ("hatecheck_A_r{}", "few_shot"),
    "rag": ("hatecheck_A_r{}", "rag"),
    "guidelines_only": ("hatecheck_G_r{}", "rag"),
    "definitions_only": ("hatecheck_D_r{}", "rag"),
    "examples_only": ("hatecheck_E_r{}", "rag"),
    "abstract_only": ("hatecheck_AB_r{}", "rag"),
}
ARMS = list(SRC)
REPS = (1, 2, 3)
B, SEED = 3000, 42
LEGACY_GATE = [("rag", "few_shot"), ("rag", "zero_shot"), ("few_shot", "zero_shot")]
LEGACY_FP = [
    ("rag", "zero_shot"),
    ("examples_only", "zero_shot"),
    ("definitions_only", "zero_shot"),
    ("abstract_only", "zero_shot"),
    ("examples_only", "rag"),
    ("examples_only", "few_shot"),
]
PAIRS = [(a, "zero_shot") for a in ARMS if a != "zero_shot"] + [
    ("rag", "few_shot"),
    ("examples_only", "rag"),
    ("examples_only", "few_shot"),
    ("definitions_only", "rag"),
    ("abstract_only", "definitions_only"),
    ("abstract_only", "rag"),
]
GOLD_PATH = Path("data/processed/en_hatecheck_eval.parquet")


def load(stem):
    arms = defaultdict(dict)
    with (Path("experiments/results") / f"{stem}_live.jsonl").open(
        encoding="utf-8"
    ) as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            if "_manifest" in r:
                continue
            require(r["item_id"] not in arms[r["arm"]], f"Duplicate ID: {stem}")
            fields = (
                "item_id",
                "result",
                "text_hash",
                "uncertain",
                "n_valid",
                "retrieval_config",
                "retrieved",
                *METADATA,
            )
            arms[r["arm"]][r["item_id"]] = {k: r.get(k) for k in fields}
    return arms


def summary(draws):
    draws = np.asarray(draws)
    return {
        "ci_low": float(np.percentile(draws, 2.5)),
        "ci_high": float(np.percentile(draws, 97.5)),
        "sd": float(draws.std(ddof=1)),
        "p_value": float(min(1, 2 * min((draws <= 0).mean(), (draws >= 0).mean()))),
        "positive_draws": int((draws > 0).sum()),
        "zero_draws": int((draws == 0).sum()),
        "negative_draws": int((draws < 0).sum()),
        "ci_excludes_zero": bool(
            np.percentile(draws, 2.5) > 0 or np.percentile(draws, 97.5) < 0
        ),
    }


def gate_macro(totals):
    tp, fp, fn, tn = np.moveaxis(totals, -1, 0)
    require(
        bool(np.all(tp + fn > 0) and np.all(tn + fp > 0)), "Single-class gate resample"
    )
    return 0.5 * (2 * tp / (2 * tp + fp + fn) + 2 * tn / (2 * tn + fn + fp))


def rate(totals):
    require(bool(np.all(totals[..., 1] > 0)), "Empty rate denominator")
    return totals[..., 0] / totals[..., 1]


def target_macro(totals):
    shape = totals.shape[:-1] + (6, 3)
    tp, fp, fn = np.moveaxis(totals.reshape(shape), -1, 0)
    require(
        bool(np.all(tp + fn > 0)),
        "A fixed label lost all gold support in a cluster draw",
    )
    return (2 * tp / (2 * tp + fp + fn)).mean(axis=-1)


def clustered_counts(ids, gold, tmpl, predictions, metric, labels):
    by_t = defaultdict(list)
    for i in sorted(ids):
        by_t[tmpl[i]].append(i)
    tids = sorted(by_t)
    width = {"gate": 4, "fp": 2, "target_group": 18}[metric]
    counts = np.zeros((len(tids), 3, 7, width), dtype=np.int64)
    for ti, t in enumerate(tids):
        for i in by_t[t]:
            truth_hate = bool(gold[i].gate)
            for rep in range(3):
                for ai, arm in enumerate(ARMS):
                    prediction = predictions[arm][rep][i]["result"]
                    pred_hate = bool(prediction["hate"])
                    if metric == "gate":
                        index = (
                            0
                            if truth_hate and pred_hate
                            else 1 if pred_hate else 2 if truth_hate else 3
                        )
                        counts[ti, rep, ai, index] += 1
                    elif metric == "fp":
                        require(not truth_hate, "FP pool contains hateful gold")
                        counts[ti, rep, ai] += (int(pred_hate), 1)
                    else:
                        require(truth_hate, "Target pool contains non-hateful gold")
                        truth = set(clean(gold[i].target_groups))
                        pred = set(prediction["target_group"])
                        for li, label in enumerate(labels):
                            counts[ti, rep, ai, 3 * li : 3 * li + 3] += (
                                int(label in truth and label in pred),
                                int(label not in truth and label in pred),
                                int(label in truth and label not in pred),
                            )
    return counts, tids, by_t


def clustered_draws(counts, tids, by_t, score, rng, validate):
    draws = np.empty((B,) + counts.shape[1:-1])

    cloned = np.random.default_rng()
    cloned.bit_generator.state = rng.bit_generator.state
    expected_first = cloned.choice(tids, len(tids), replace=True).tolist()
    for k in range(B):
        pick = rng.choice(len(tids), len(tids), replace=True)
        if k == 0:
            require(
                [tids[j] for j in pick] == expected_first,
                "RNG template mapping mismatch",
            )
        draws[k] = score(counts[pick].sum(axis=0))
        if k in (0, 1, B - 1):
            sampled = [i for j in pick for i in by_t[tids[j]]]
            validate(sampled, draws[k])
    return draws


def contrast(arm_a, arm_b, points, draws):
    ai, bi = ARMS.index(arm_a), ARMS.index(arm_b)
    delta = float(points[:, ai].mean() - points[:, bi].mean())
    floor = float(max(np.ptp(points[:, ai]), np.ptp(points[:, bi])))
    out = summary((draws[:, :, ai] - draws[:, :, bi]).mean(axis=1))
    out.update(
        {
            "arm_a": arm_a,
            "arm_b": arm_b,
            "delta": delta,
            "floor": floor,
            "magnitude_over_floor": abs(delta) / floor if floor else None,
            "clears_floor": abs(delta) > floor,
            "complete_separation": bool(
                points[:, ai].max() < points[:, bi].min()
                or points[:, bi].max() < points[:, ai].min()
            ),
        }
    )
    return out


def show(c, prefix="  "):
    ratio = (
        f"{c['magnitude_over_floor']:.2f}"
        if c.get("magnitude_over_floor") is not None
        else "undefined"
    )
    print(
        f"{prefix}{c.get('arm_a', c.get('name'))} - {c.get('arm_b', '')}: delta={c['delta']:+.6f}; "
        f"floor={c.get('floor', 0):.6f}; ratio={ratio}; CI=[{c['ci_low']:+.6f}, {c['ci_high']:+.6f}]; "
        f"p={c['p_value']:.6f}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite {args.output}")
    paths = {
        Path(__file__),
        GOLD_PATH,
        Path("src/hsrag/metrics.py"),
        Path("src/hsrag/schema.py"),
        Path("analysis/classification/prototype/type_scores.py"),
        Path("analysis/classification/prototype/bootstrap_gate.py"),
        Path("analysis/classification/prototype/six_arms.py"),
        Path("analysis/classification/prototype/repeat_table.py"),
        Path("Thesis_Log/hatecheck_log.md"),
        Path("Thesis_Log/sq2_log.md"),
    }
    for pattern, arm in SRC.values():
        for rep in REPS:
            stem = pattern.format(rep)
            paths.update(
                (
                    Path("experiments/results") / f"{stem}_live.jsonl",
                    Path("experiments/manifests") / f"{stem}.yaml",
                )
            )
    before = {p: sha(p) for p in paths}
    df = pd.read_parquet(GOLD_PATH)
    gold = {str(r.id): r for r in df.itertuples(index=False)}
    require(len(gold) == len(df), "Duplicate gold IDs")
    meta = {i: json.loads(g.raw) for i, g in gold.items()}
    tmpl = {i: m["templ_id"] for i, m in meta.items()}
    labels = sorted(
        {l for g in gold.values() if bool(g.gate) for l in clean(g.target_groups)}
    )
    require(len(labels) == 6, "Unexpected target label set")
    cache, predictions, inventory = {}, {}, []
    for arm, (pattern, stored_arm) in SRC.items():
        predictions[arm] = []
        for rep in REPS:
            stem = pattern.format(rep)
            if stem not in cache:
                cache[stem] = load(stem)
            rows = cache[stem][stored_arm]
            require(set(rows) <= set(gold), f"Unknown result ID: {stem}")
            predictions[arm].append(rows)
            manifest = yaml.safe_load(
                (Path("experiments/manifests") / f"{stem}.yaml").read_text(
                    encoding="utf-8"
                )
            )
            require(
                manifest["subset"] == "en_hatecheck_eval", "Manifest subset mismatch"
            )
            entry = {
                "arm": arm,
                "stored_arm": stored_arm,
                "replicate": rep,
                "stem": stem,
                "rows": len(rows),
                "missing_ids": sorted(set(gold) - set(rows)),
                "unscored_ids": sorted(
                    i for i, r in rows.items() if r["result"] is None
                ),
                "metadata": {
                    k: sorted(
                        {json.dumps(r.get(k), sort_keys=True) for r in rows.values()}
                    )
                    for k in METADATA
                },
                "retrieval_configs": sorted(
                    {
                        json.dumps(r["retrieval_config"], sort_keys=True)
                        for r in rows.values()
                    }
                ),
                "manifest": {
                    k: v
                    for k, v in manifest.items()
                    if k not in ("name", "description")
                },
            }
            inventory.append(entry)
            print(
                f"Source {stem}/{arm}: rows={len(rows)}; missing={len(entry['missing_ids'])}; "
                f"unscored={len(entry['unscored_ids'])}; metadata={entry['metadata']}",
                flush=True,
            )

    def common_for(arms):
        out = set(gold)
        for arm in arms:
            for rows in predictions[arm]:
                out &= {i for i, r in rows.items() if r["result"] is not None}
        for i in out:
            hashes = {rows[i]["text_hash"] for arm in arms for rows in predictions[arm]}
            require(
                None not in hashes and len(hashes) == 1, "Mismatched input text hashes"
            )
        return sorted(out)

    common3, common7 = common_for(ARMS[:3]), common_for(ARMS)
    nonhate = [i for i in common7 if not bool(gold[i].gate)]
    hateful = [i for i in common7 if bool(gold[i].gate)]
    pools = {}
    for name, ids in [
        ("three_arm", common3),
        ("seven_arm", common7),
        ("non_hateful", nonhate),
        ("hateful", hateful),
    ]:
        pools[name] = {
            "n": len(ids),
            "templates": len({tmpl[i] for i in ids}),
            "item_ids": ids,
            "gold_hateful": sum(bool(gold[i].gate) for i in ids),
        }
        print(
            f"POOL {name}: n={len(ids)}, templates={pools[name]['templates']}, hateful={pools[name]['gold_hateful']}",
            flush=True,
        )
    pools["three_arm"]["excluded_ids"] = sorted(set(gold) - set(common3))
    pools["seven_arm"]["excluded_ids"] = sorted(set(gold) - set(common7))

    primary_predictions = {
        arm: predictions[arm] if arm in ARMS[:3] else predictions["zero_shot"]
        for arm in ARMS
    }
    pc, pt, pb = clustered_counts(
        common3, gold, tmpl, primary_predictions, "gate", labels
    )
    pp = gate_macro(pc.sum(axis=0))
    primary_scores = {}
    for ai, arm in enumerate(ARMS[:3]):
        for rep in range(3):
            actual = score_gate(
                [predictions[arm][rep][i] for i in common3], gold
            ).macro_f1
            require(abs(actual - pp[rep, ai]) < 1e-12, "Primary gate scorer mismatch")
        vals = pp[:, ai]
        primary_scores[arm] = {
            "replicates": vals.tolist(),
            "mean": float(vals.mean()),
            "spread": float(np.ptp(vals)),
        }
        print(
            f"PRIMARY gate {arm}: scores={vals.tolist()}; mean={vals.mean():.6f}; spread={np.ptp(vals):.6f}",
            flush=True,
        )
    rng, primary = np.random.default_rng(SEED), []
    for a, b in LEGACY_GATE:
        ai, bi = ARMS.index(a), ARMS.index(b)
        rows = []
        for rep in range(3):
            selected = pc[:, rep, [ai, bi], :]

            def validate(ids, values, rep=rep, a=a, b=b):
                for j, arm in enumerate((a, b)):
                    expected = score_gate(
                        [predictions[arm][rep][i] for i in ids], gold
                    ).macro_f1
                    require(
                        abs(expected - values[j]) < 1e-12,
                        "Item-expanded primary bootstrap mismatch",
                    )

            draws = clustered_draws(selected, pt, pb, gate_macro, rng, validate)
            result = summary(draws[:, 0] - draws[:, 1])
            result.update(
                {
                    "replicate_pair": rep + 1,
                    "arm_a": a,
                    "arm_b": b,
                    "delta": float(pp[rep, ai] - pp[rep, bi]),
                }
            )
            rows.append(result)
            show(result, f"PRIMARY pair {rep + 1}: ")
        floor = max(primary_scores[a]["spread"], primary_scores[b]["spread"])
        delta = float((pp[:, ai] - pp[:, bi]).mean())
        primary.append(
            {
                "arm_a": a,
                "arm_b": b,
                "delta": delta,
                "floor": floor,
                "magnitude_over_floor": abs(delta) / floor,
                "intervals_excluding_zero": sum(r["ci_excludes_zero"] for r in rows),
                "pairs": rows,
            }
        )
        print(
            f"PRIMARY mean {a}-{b}: delta={delta:+.6f}; floor={floor:.6f}; ratio={abs(delta)/floor:.2f}",
            flush=True,
        )

    scopes = [
        ("gate_all", common7, "gate"),
        ("fp_all", nonhate, "fp"),
        ("target_group", hateful, "target_group"),
    ] + [
        ("fp_" + block, [i for i in nonhate if meta[i]["block"] == block], "fp")
        for block in ("guideline_covered", "definition_covered", "uncovered")
    ]
    results, legacy_fp_state = {}, np.random.default_rng(SEED)
    legacy_fp_counts = {}
    for scope, ids, metric in scopes:
        counts, tids, by_t = clustered_counts(
            ids, gold, tmpl, predictions, metric, labels
        )
        scorer = {"gate": gate_macro, "fp": rate, "target_group": target_macro}[metric]
        points = scorer(counts.sum(axis=0))
        extras = {}
        for ai, arm in enumerate(ARMS):
            if metric == "gate":
                extras[arm] = [
                    asdict(score_gate([predictions[arm][rep][i] for i in ids], gold))
                    for rep in range(3)
                ]
                for rep, s in enumerate(extras[arm]):
                    require(
                        np.array_equal(
                            counts.sum(axis=0)[rep, ai],
                            [s[k] for k in ("tp", "fp", "fn", "tn")],
                        ),
                        "Gate counts mismatch",
                    )
                    require(
                        abs(s["macro_f1"] - points[rep, ai]) < 1e-12,
                        "Seven-arm gate mismatch",
                    )
            elif metric == "target_group":
                extras[arm] = [
                    asdict(
                        score_multilabel(
                            [predictions[arm][rep][i] for i in ids],
                            gold,
                            "target_groups",
                            "target_group",
                        )
                    )
                    for rep in range(3)
                ]
                for rep, s in enumerate(extras[arm]):
                    require(
                        s["labels_averaged"] == labels and s["n_items"] == len(ids),
                        "Target label/pool mismatch",
                    )
                    require(
                        abs(s["macro_f1"] - points[rep, ai]) < 1e-12,
                        "Target macro mismatch",
                    )
                    for li, label in enumerate(labels):
                        require(
                            np.array_equal(
                                counts.sum(axis=0)[rep, ai, 3 * li : 3 * li + 3],
                                [s["per_label"][label][k] for k in ("tp", "fp", "fn")],
                            ),
                            "Target TP/FP/FN mismatch",
                        )

        def validate(ids, values):

            for ai, arm in enumerate(ARMS):
                rows = [predictions[arm][0][i] for i in ids]
                if metric == "gate":
                    expected = score_gate(rows, gold).macro_f1
                elif metric == "fp":
                    expected = sum(bool(r["result"]["hate"]) for r in rows) / len(rows)
                else:
                    expected = score_multilabel(
                        rows, gold, "target_groups", "target_group", fixed_labels=labels
                    ).macro_f1
                require(
                    abs(expected - values[0, ai]) < 1e-12,
                    f"Item-expanded {scope} bootstrap mismatch",
                )

        draws = clustered_draws(
            counts, tids, by_t, scorer, np.random.default_rng(SEED), validate
        )
        record = {
            "metric": metric,
            "n_items": len(ids),
            "n_templates": len(tids),
            "item_ids": ids,
            "replicate_scores": {
                a: points[:, ai].tolist() for ai, a in enumerate(ARMS)
            },
            "full_sample_counts": counts.sum(axis=0).tolist(),
            "scores": extras,
            "contrasts": [contrast(a, b, points, draws) for a, b in PAIRS],
        }
        print(f"\nSEVEN-ARM {scope}: n={len(ids)}, templates={len(tids)}", flush=True)
        for ai, arm in enumerate(ARMS):
            extra = ""
            if metric == "gate":
                extra = f"; FP={[s['fp'] for s in extras[arm]]}; FN={[s['fn'] for s in extras[arm]]}"
            if metric == "target_group":
                extra = f"; predicted/cardinality={np.mean([s['extra']['mean_pred_labels'] for s in extras[arm]]):.6f}"
            print(
                f"  {arm}: scores={[round(x,6) for x in points[:,ai]]}; mean={points[:,ai].mean():.6f}; "
                f"spread={np.ptp(points[:,ai]):.6f}{extra}",
                flush=True,
            )
        for c in record["contrasts"]:
            show(c)
        record["composition_residuals"] = []
        for name, weights in [
            (
                "three_kind_predicted_minus_observed",
                {
                    "definitions_only": 1,
                    "guidelines_only": 1,
                    "examples_only": 1,
                    "rag": -1,
                    "zero_shot": -2,
                },
            ),
            (
                "abstract_pair_predicted_minus_observed",
                {
                    "definitions_only": 1,
                    "guidelines_only": 1,
                    "abstract_only": -1,
                    "zero_shot": -1,
                },
            ),
        ]:
            observed = sum(
                w * points[:, ARMS.index(a)].mean() for a, w in weights.items()
            )
            distribution = sum(
                w * draws[:, :, ARMS.index(a)].mean(axis=1) for a, w in weights.items()
            )
            floor = max(
                np.ptp(points[:, ARMS.index(a)])
                for a in (ARMS if len(weights) == 5 else weights)
            )
            residual = summary(distribution)
            residual.update(
                {
                    "name": name,
                    "delta": float(observed),
                    "floor": float(floor),
                    "magnitude_over_floor": float(abs(observed) / floor),
                    "weights": weights,
                }
            )
            record["composition_residuals"].append(residual)
            show(residual)
        if metric == "target_group":
            record["labels"] = labels
            record["gold_support"] = extras["zero_shot"][0]["per_label"]
            record["gold_cardinality"] = extras["zero_shot"][0]["extra"][
                "mean_gold_labels"
            ]
            per_label_delta = {
                l: float(
                    np.mean(
                        [s["per_label"][l]["f1"] for s in extras["definitions_only"]]
                    )
                    - np.mean([s["per_label"][l]["f1"] for s in extras["zero_shot"]])
                )
                for l in labels
            }
            record["definitions_minus_zero_shot_per_label_delta"] = per_label_delta
            record["sexual_orientation_share_of_definition_gain"] = per_label_delta[
                "sexual_orientation"
            ] / sum(per_label_delta.values())
            print(
                f"  Target labels={labels}; gold cardinality={record['gold_cardinality']}; "
                f"supports={ {l: record['gold_support'][l]['support'] for l in labels} }",
                flush=True,
            )
            print(
                f"  Definition per-label deltas={per_label_delta}; sexual_orientation share="
                f"{record['sexual_orientation_share_of_definition_gain']:.6f}",
                flush=True,
            )
        results[scope] = record
        if metric == "fp":
            legacy_fp_counts[scope] = (counts, tids, by_t, points)

    legacy_fp = []
    for scope in (
        "fp_all",
        "fp_guideline_covered",
        "fp_definition_covered",
        "fp_uncovered",
    ):
        counts, tids, by_t, points = legacy_fp_counts[scope]
        for a, b in LEGACY_FP:
            ai, bi = ARMS.index(a), ARMS.index(b)
            selected = counts[:, :, [ai, bi], :]

            def validate(ids, values, a=a, b=b):
                for j, arm in enumerate((a, b)):
                    for rep in range(3):
                        expected = sum(
                            bool(predictions[arm][rep][i]["result"]["hate"])
                            for i in ids
                        ) / len(ids)
                        require(
                            abs(expected - values[rep, j]) < 1e-12,
                            "Legacy FP item-expanded mismatch",
                        )

            draws = clustered_draws(
                selected, tids, by_t, rate, legacy_fp_state, validate
            )
            result = summary((draws[:, :, 0] - draws[:, :, 1]).mean(axis=1))
            result.update(
                {
                    "scope": scope,
                    "arm_a": a,
                    "arm_b": b,
                    "delta": float((points[:, ai] - points[:, bi]).mean()),
                }
            )
            legacy_fp.append(result)
            show(result, f"LEGACY {scope}: ")
    require(
        all(sha(p) == h for p, h in before.items()), "Input changed during execution"
    )
    print(
        f"\nPASS: {len(before)}/{len(before)} input hashes unchanged. All scorer/count and sampled-draw validations passed.",
        flush=True,
    )
    evidence = {
        "audit": "Astra P2 checkpoint 4: HateCheck gate and record-kind statistics",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "analysis_script": {
            "path": Path(__file__).as_posix(),
            "sha256": before[Path(__file__)],
        },
        "inputs": [
            {"path": p.as_posix(), "sha256": h, "bytes": p.stat().st_size}
            for p, h in sorted(before.items())
        ],
        "source_inventory": inventory,
        "pools": pools,
        "method": {
            "bootstrap_draws": B,
            "seed": SEED,
            "unit": "templ_id cluster, sampled uniformly with replacement; all rows in sampled cluster retained",
            "primary_gate_rng": "one continuous RNG across rag-few_shot, rag-zero_shot, few_shot-zero_shot; r1,r2,r3 within each",
            "legacy_fp_rng": "one continuous RNG across ALL, guideline_covered, definition_covered, uncovered and six original contrasts",
            "supplementary_rng": "reset seed 42 per scope; same draws across all arms/replicates/contrasts in a scope",
            "supplementary_estimator": "mean of three per-replicate metric differences; stored replicates fixed",
            "interval": "unadjusted 95% percentile",
            "p_value": "twice smaller inclusive sign-tail fraction, capped at one; legacy HateCheck method",
            "scorer_validation": "all full-sample counts checked; first, second and final draws checked by expanding sampled template items",
            "floor": "maximum observed within-arm range of compared arms; composition residual convention documented separately",
        },
        "primary_gate_scores": primary_scores,
        "primary_gate_comparisons": primary,
        "seven_arm_results": results,
        "legacy_fp_bootstrap": legacy_fp,
        "all_inputs_unchanged": True,
        "limitations": [
            "Supplementary intervals are exploratory, unadjusted, and conditional on three stored replicates",
            "Cluster sampling allows within-template dependence; it does not remove all possible dependence between templates",
            "Shared suite/configuration does not make non-hateful FP and hateful target-group strata the same items",
            "No equivalence margins or equivalence tests; non-detection is not proof of no effect",
            "Seven arms do not form a complete factorial design for three record kinds",
            "No encoder, model-cost, guideline-oracle rerun, or Priority 3 work in this execution",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(evidence, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"WROTE NEW EVIDENCE: {args.output}", flush=True)


if __name__ == "__main__":
    main()

import argparse
from collections import Counter
from pathlib import Path
import json
import socket
import time
import traceback
from unittest.mock import patch
import numpy as np
from analysis.feedback.source import updates_review as r

b = r.b
a = b.a
ROOT = b.ROOT
u = r.u
BASE = b.STUDY / "results_v1"
EVIDENCE = (
    ROOT / "Thesis_Log/astra_result/evidence/sq3_feedback_results_20260930_v1.json"
)
LABELS = list(u.LABELS)
SEEDS = list(b.SEEDS)
DRAWS = 10000
SEED = 20260928
METRICS = (
    "macro_f1",
    "contains",
    "exact",
    "predicted_labels",
    "missed_hate_rate",
    "flip_rate",
)


def condition(arm, seed, level):
    return f"{arm}_s{seed}_b{level:02d}"


def summarize(values):
    return u.describe([float(v) for v in values])


def q95(values):
    return np.quantile(values, [0.025, 0.975], method="linear").tolist()


def arrays(ids, gold, run):
    y = np.array([[label in gold[i] for label in LABELS] for i in ids], dtype=bool)
    valid = np.array([run[i]["result"] is not None for i in ids], dtype=bool)
    x = np.array(
        [
            [
                label in run[i]["result"]["hate_type"] if valid[k] else False
                for label in LABELS
            ]
            for k, i in enumerate(ids)
        ],
        dtype=bool,
    )
    gate = np.array(
        [
            bool(run[i]["result"]["hate"]) if valid[k] else False
            for k, i in enumerate(ids)
        ],
        dtype=bool,
    )
    a.require(not np.any(x[valid & ~gate]), "Predictions violate hate gate")
    counts = np.stack([y & x, ~y & x, y & ~x], axis=-1).astype(np.int32)
    rates = np.stack(
        [
            np.all(~y | x, axis=1),
            np.all(y == x, axis=1),
            x.sum(axis=1),
            ~gate,
            ~np.all(~y | x, axis=1),
        ],
        axis=-1,
    ).astype(float)
    return {
        "ids": list(ids),
        "gold": y,
        "valid": valid,
        "counts": counts,
        "rates": rates,
    }


def f1(counts):
    tp, fp, fn = [counts[..., k] for k in range(3)]
    denominator = 2 * tp + fp + fn
    return np.divide(
        2 * tp,
        denominator,
        out=np.zeros_like(denominator, dtype=float),
        where=denominator != 0,
    ).mean(axis=-1)


def checked_metric(ids, gold, run):
    score = u.metric(ids, gold, run)
    data = arrays(ids, gold, run)
    counts = data["counts"][data["valid"]].sum(axis=0)
    point = float(f1(counts))
    a.require(abs(point - score["macro_f1"]) < 1e-13, "Independent metric differs")
    rates = data["rates"][data["valid"]].mean(axis=0)
    a.require(
        np.allclose(
            rates[:3],
            [score["contains"], score["exact"], score["predicted_labels"]],
            rtol=0,
            atol=1e-13,
        ),
        "Rate metric differs",
    )
    score["missed_hate_rate"] = float(rates[3])
    score["flip_rate"] = float(rates[4])
    for label, entry in score["per_label"].items():
        tp, fp, fn = entry["tp"], entry["fp"], entry["fn"]
        entry["precision"] = tp / (tp + fp) if tp + fp else 0.0
        entry["recall"] = tp / (tp + fn) if tp + fn else 0.0
    return score


def bootstrap(counts, rates, draws, seed):

    settings, n, labels, _ = counts.shape
    a.require(
        n > 0 and labels == 7 and rates.shape == (settings, n, 5), "Bad bootstrap shape"
    )
    indices = np.random.default_rng(seed).integers(0, n, size=(draws, n))
    weights = np.stack([np.bincount(pick, minlength=n) for pick in indices]).astype(
        np.int32
    )
    scores = np.empty((draws, settings))
    values = np.empty((draws, settings, 5))
    for setting in range(settings):
        summed = (weights @ counts[setting].reshape(n, 21)).reshape(draws, 7, 3)
        scores[:, setting] = f1(summed)
        values[:, setting, :] = (weights @ rates[setting]) / n
        for begin in range(0, draws, 128):
            pick = indices[begin : begin + 128]
            expanded = counts[setting][pick].sum(axis=1)

            tp = expanded[:, :, 0]
            den = 2 * tp + expanded[:, :, 1] + expanded[:, :, 2]
            alt = np.zeros_like(den, dtype=float)
            np.divide(2 * tp, den, out=alt, where=den > 0)
            a.require(
                np.allclose(
                    scores[begin : begin + 128, setting],
                    alt.mean(axis=1),
                    rtol=0,
                    atol=1e-13,
                ),
                "Expanded F1 bootstrap differs",
            )
            alt_rates = rates[setting][pick].mean(axis=1)
            a.require(
                np.allclose(
                    values[begin : begin + 128, setting, :],
                    alt_rates,
                    rtol=0,
                    atol=1e-13,
                ),
                "Expanded rate bootstrap differs",
            )
    return scores, values


def compare(ids, gold, runs, groups, draws=DRAWS):
    conditions = [c for members in groups.values() for c in members]
    a.require(len(conditions) == len(set(conditions)), "Group membership repeated")
    keys = [(c, rep) for c in conditions for rep in range(3)]
    data = [arrays(ids, gold, runs[key]) for key in keys]
    common = np.logical_and.reduce([d["valid"] for d in data])
    shared = [iid for iid, ok in zip(ids, common) if ok]
    a.require(
        shared
        == [i for i in ids if all(runs[k][i]["result"] is not None for k in keys)],
        "Independent common mask differs",
    )
    a.require(shared, "No common-valid items")
    counts = np.stack([d["counts"][common] for d in data])
    rates = np.stack([d["rates"][common] for d in data])
    point = f1(counts.sum(axis=1))
    rate_point = rates.mean(axis=1)
    boot, rate_boot = bootstrap(counts, rates, draws, SEED)
    own = {}
    condition_scores = {}
    for c in conditions:
        pos = [keys.index((c, rep)) for rep in range(3)]
        scores = [checked_metric(ids, gold, runs[c, rep]) for rep in range(3)]
        common_scores = [checked_metric(shared, gold, runs[c, rep]) for rep in range(3)]
        a.require(
            np.allclose(
                point[pos], [s["macro_f1"] for s in common_scores], rtol=0, atol=1e-13
            ),
            "Common-mask run F1 differs",
        )
        own[c] = scores
        condition_scores[c] = {
            "common_valid_runs": common_scores,
            "macro_f1": summarize(point[pos]),
            "own_valid_runs": scores,
            "mean_full_roster_failure_bounds": np.mean(
                [s["full_roster_failure_bounds"] for s in scores], axis=0
            ).tolist(),
        }
    groups_out = {}
    group_boot = {}
    group_rate_boot = {}
    for group, members in groups.items():
        positions = [[keys.index((c, rep)) for rep in range(3)] for c in members]
        pool_means = [float(point[p].mean()) for p in positions]
        group_boot[group] = np.mean(
            [boot[:, p].mean(axis=1) for p in positions], axis=0
        )
        group_rate_boot[group] = np.mean(
            [rate_boot[:, p, :].mean(axis=1) for p in positions], axis=0
        )
        metrics = {"macro_f1": float(np.mean(pool_means))}
        metrics.update(
            {
                name: float(np.mean([rate_point[p, j].mean() for p in positions]))
                for j, name in enumerate(METRICS[1:])
            }
        )
        per_label = {
            label: {
                metric: float(
                    np.mean(
                        [
                            s["per_label"][label][metric]
                            for c in members
                            for s in condition_scores[c]["common_valid_runs"]
                        ]
                    )
                )
                for metric in ("precision", "recall", "f1")
            }
            for label in LABELS
        }
        groups_out[group] = {
            "conditions": members,
            "metrics": metrics,
            "macro_f1_ci95": q95(group_boot[group]),
            "pool_means": pool_means,
            "between_pool_sample_sd": (
                float(np.std(pool_means, ddof=1)) if len(members) > 1 else None
            ),
            "between_pool_range": (
                float(np.ptp(pool_means)) if len(members) > 1 else None
            ),
            "per_label_mean_metrics": per_label,
            "mean_full_roster_failure_bounds": np.mean(
                [
                    condition_scores[c]["mean_full_roster_failure_bounds"]
                    for c in members
                ],
                axis=0,
            ).tolist(),
            "reference_note": (
                "Three actual repeats only; no artificial pool replication."
                if group == "reference"
                else "Three decoder repeats per fixed pool, then equal average over pools."
            ),
        }
    contrasts = {}
    for left, right in (
        ("error", "correct"),
        ("error", "reference"),
        ("correct", "reference"),
    ):
        if left not in groups or right not in groups:
            continue
        diff = group_boot[left] - group_boot[right]
        rate_diff = group_rate_boot[left] - group_rate_boot[right]
        lo = groups_out[left]["mean_full_roster_failure_bounds"]
        hi = groups_out[right]["mean_full_roster_failure_bounds"]
        contrasts[left + "_minus_" + right] = {
            "left": left,
            "right": right,
            "mean_difference": groups_out[left]["metrics"]["macro_f1"]
            - groups_out[right]["metrics"]["macro_f1"],
            "macro_f1_ci95": q95(diff),
            "full_roster_failure_difference_bounds": [lo[0] - hi[1], lo[1] - hi[0]],
            "other_metrics": {
                name: {
                    "difference": groups_out[left]["metrics"][name]
                    - groups_out[right]["metrics"][name],
                    "ci95": q95(rate_diff[:, j]),
                }
                for j, name in enumerate(METRICS[1:])
            },
        }
    return {
        "original_n": len(ids),
        "common_valid_n": len(shared),
        "common_valid_ids": shared,
        "excluded_ids": sorted(set(ids) - set(shared)),
        "conditions": condition_scores,
        "groups": groups_out,
        "contrasts": contrasts,
        "bootstrap": {
            "draws": draws,
            "seed": SEED,
            "method": "Paired singleton clusters; same sampled texts in every setting, score each repeat then average within/across fixed pools.",
            "independently_verified_setting_draws": draws * len(keys),
            "confidence": 0.95,
            "percentile_method": "linear",
            "scope": "Conditional on selected fixed pools and observed decoder repeats; no resampling of repeats or pool seeds.",
        },
    }


def tests():
    def pred(types=None, hate=True):
        return {"result": None if types is None else {"hate": hate, "hate_type": types}}

    gold = {"a": {"explicit"}, "b": {"irony"}, "c": {"explicit"}}
    perfect = {i: pred(sorted(y)) for i, y in gold.items()}
    zero = {i: pred([], False) for i in gold}
    imperfect = {"a": pred(["explicit"]), "b": pred(["explicit"]), "c": pred(["irony"])}
    n = 0
    m = checked_metric(list(gold), gold, perfect)
    a.require(
        abs(m["macro_f1"] - 2 / 7) < 1e-13 and m["contains"] == m["exact"] == 1,
        "Perfect/fixed-label metric",
    )
    n += 1
    m = checked_metric(list(gold), gold, zero)
    a.require(m["macro_f1"] == 0 and m["missed_hate_rate"] == 1, "Gate false metric")
    n += 1
    null = dict(perfect)
    null["c"] = pred()
    m = checked_metric(list(gold), gold, null)
    a.require(
        m["valid"] == 2
        and m["null_ids"] == ["c"]
        and m["full_roster_failure_bounds"][0] < m["full_roster_failure_bounds"][1],
        "Null bounds",
    )
    n += 1
    extra = {i: pred(sorted(y | {"grievance"})) for i, y in gold.items()}
    m = checked_metric(list(gold), gold, extra)
    a.require(m["contains"] == 1 and m["exact"] == 0, "Contains allows extra labels")
    n += 1
    bad = {"a": pred(["explicit"], False)}
    try:
        arrays(["a"], gold, bad)
    except a.IntegrityError:
        pass
    else:
        raise AssertionError("Bad gate accepted")
    n += 1
    groups = {
        "error": ["e1", "e2", "e3"],
        "correct": ["c1", "c2", "c3"],
        "reference": ["reference"],
    }
    runs = {
        (c, rep): perfect
        for members in groups.values()
        for c in members
        for rep in range(3)
    }
    same = compare(list(gold), gold, runs, groups, 31)
    a.require(
        all(
            x["mean_difference"] == 0 and x["macro_f1_ci95"] == [0, 0]
            for x in same["contrasts"].values()
        ),
        "Identical paired systems",
    )
    n += 1
    for c in groups["correct"]:
        for rep in range(3):
            runs[c, rep] = zero
    positive = compare(list(gold), gold, runs, groups, 31)
    a.require(
        abs(positive["contrasts"]["error_minus_correct"]["mean_difference"] - 2 / 7)
        < 1e-13
        and positive["contrasts"]["error_minus_correct"]["macro_f1_ci95"][0] > 0,
        "Positive paired fixture",
    )
    n += 1
    a.require(
        len(positive["groups"]["reference"]["pool_means"]) == 1
        and positive["groups"]["reference"]["between_pool_sample_sd"] is None,
        "Baseline not triplicated",
    )
    n += 1
    runs["e1", 1] = null
    masked = compare(list(gold), gold, runs, groups, 31)
    a.require(
        masked["excluded_ids"] == ["c"] and masked["common_valid_n"] == 2,
        "Shared mask across settings",
    )
    n += 1
    runs["e1", 1] = zero
    runs["e1", 2] = zero
    uneven = compare(list(gold), gold, runs, groups, 31)
    a.require(
        abs(uneven["conditions"]["e1"]["macro_f1"]["mean"] - (2 / 7) / 3) < 1e-13,
        "Average per-repeat F1, not pooled confusion",
    )
    n += 1
    repeat = compare(list(gold), gold, runs, groups, 31)
    a.require(repeat == uneven, "Deterministic paired bootstrap")
    n += 1
    a.require(
        positive["groups"]["error"]["between_pool_sample_sd"] == 0
        and positive["conditions"]["e1"]["macro_f1"]["sample_sd"] == 0,
        "Separate pool/decoder spread",
    )
    n += 1
    rec = {
        "created_utc": a.utc(),
        "passed": True,
        "fixtures": n,
        "source_sha256": a.digest(__file__),
        "environment": b.baseline.environment(),
        "labels": LABELS,
        "actual_API_calls": 0,
    }
    path = BASE / "development" / ("tests_" + a.digest(__file__)[:16] + ".json")
    a.save_new(path, rec)
    print(json.dumps(rec, indent=2))
    return rec


def run():
    a.require(
        not EVIDENCE.exists() and not (BASE / "complete.json").exists(),
        "Preserve completed scores",
    )
    tested_path = BASE / "development" / ("tests_" + a.digest(__file__)[:16] + ".json")
    tested = a.read(tested_path)
    a.require(
        tested["passed"] and tested["source_sha256"] == a.digest(__file__),
        "Current tests required",
    )
    done = a.read(r.BASE / "complete.json")
    a.require(
        a.digest(r.EVIDENCE) == done["evidence_sha256"]
        and done["status"] == "all_feedback_updates_independently_reviewed",
        "Complete independent review required",
    )
    science = a.read(b.SCIENCE)
    a.require(
        science["scoring"]["labels"] == LABELS
        and science["selection"]["seeds"] == SEEDS,
        "Scientific labels/draws changed",
    )
    selection = a.read(b.preparation.review.BASE / "selection.json")
    a.require(
        selection["k"] == 40 and selection["levels"] == [10, 20, 30, 40],
        "Selected budgets changed",
    )
    inputs = {
        b.rel(__file__): a.digest(__file__),
        b.rel(tested_path): a.digest(tested_path),
        b.rel(r.EVIDENCE): a.digest(r.EVIDENCE),
        b.rel(r.BASE / "complete.json"): a.digest(r.BASE / "complete.json"),
        b.rel(b.SCIENCE): a.digest(b.SCIENCE),
        b.rel(b.preparation.review.BASE / "selection.json"): a.digest(
            b.preparation.review.BASE / "selection.json"
        ),
    }
    for name, sha in done["outputs"].items():
        inputs[name] = sha
    for p in [
        b.preparation.BASE / "pools.json",
        b.preparation.BASE / "added_records.jsonl",
        b.preparation.BASE / "prompts.jsonl",
        b.preparation.BASE / "complete.json",
        b.STUDY / "feasibility_v1/roster.jsonl",
        ROOT
        / "Thesis_Log/astra_result/evidence/sq3_feedback_feasibility_20260928_v1.json",
        ROOT / "analysis/feedback/summary/p3_audit.py",
        b.preparation.review.BASE / "complete.json",
    ]:
        inputs[b.rel(p)] = a.digest(p)
    baseline_done = a.read(b.preparation.review.BASE / "complete.json")
    for rep in range(3):
        p = b.preparation.review.BASE / "assembled" / f"reference_rep{rep}.jsonl"
        a.require(
            a.digest(p) == baseline_done["outputs"][b.rel(p)],
            "Reference assembly changed",
        )
        inputs[b.rel(p)] = a.digest(p)
    r.verify(inputs)
    attempt = BASE / "attempts" / str(time.time_ns())
    a.save_new(
        attempt / "plan.json",
        {
            "created_utc": a.utc(),
            "inputs": inputs,
            "tests": tested,
            "primary": "Final40 error-minus-correct macro-F1 on350 transfer texts, common validity across18 update settings.",
            "secondary_masks": "For each budget, common-valid across18 update settings plus3 actual reference repeats, separately for transfer, holdout and sentinels.",
            "bootstrap": {
                "draws": DRAWS,
                "seed": SEED,
                "labels": LABELS,
                "primary_confidence": 0.95,
                "secondary": "descriptive pointwise95",
            },
            "additional_diagnostics": "Own-valid all-run metrics; selected/not-selected candidate results; retrieved-example label relevance.",
            "scope": "Implements frozen scientific plan after outcomes generated. No pool/condition changes, tuning, or new API calls.",
        },
    )
    try:
        roster = {
            x["id"]: x for x in b.core.rows(b.STUDY / "feasibility_v1/roster.jsonl")
        }
        gold = {i: set(row["gold"]["hate_type"]) for i, row in roster.items()}
        runs = {}
        for name in done["outputs"]:
            rows = b.core.rows(ROOT / name)
            c = rows[0]["condition"]
            rep = rows[0]["repeat"]
            a.require(
                (c, rep) not in runs
                and len(rows) == 466
                and len({x["id"] for x in rows}) == 466,
                "Malformed assembled run",
            )
            a.require(
                all(x["condition"] == c and x["repeat"] == rep for x in rows),
                "Mixed assembled settings",
            )
            runs[c, rep] = {x["id"]: x for x in rows}
        for rep in range(3):
            runs["reference", rep] = {
                x["id"]: x
                for x in b.core.rows(
                    b.preparation.review.BASE
                    / "assembled"
                    / f"reference_rep{rep}.jsonl"
                )
            }
        a.require(
            len(runs) == 75
            and all(
                set(run) == set(science["rosters"]["evaluation_phase"])
                for run in runs.values()
            ),
            "Complete run matrix differs",
        )
        for run in runs.values():
            for row in run.values():
                u.check_prediction(row["result"], LABELS)
        feasibility = a.read(
            ROOT
            / "Thesis_Log/astra_result/evidence/sq3_feedback_feasibility_20260928_v1.json"
        )
        components = feasibility["components"]
        transfer = set(science["rosters"]["transfer"])
        clusters = [
            sorted(set(group) & transfer)
            for group in components
            if set(group) & transfer
        ]
        a.require(
            len(clusters) == 350
            and all(len(group) == 1 for group in clusters)
            and {i for group in clusters for i in group} == transfer,
            "Frozen transfer clusters are not350singletons",
        )
        endpoints = {
            "transfer": science["rosters"]["transfer"],
            "holdout": science["rosters"]["holdout"],
            "sentinel": selection["sentinel_ids"],
        }
        all_groups = {
            **endpoints,
            "original_transfer_pool": science["rosters"]["sentinel_selection_pool"],
            "eligible_candidates": science["rosters"]["eligible_candidates"],
            "entire_reference_roster": science["rosters"]["evaluation_phase"],
        }
        metrics = {}
        for name, ids in all_groups.items():
            metrics[name] = {
                f"{c}_rep{rep}": checked_metric(ids, gold, run)
                for (c, rep), run in sorted(runs.items())
            }
        own_path = BASE / "all_run_metrics.json"
        a.save_new(own_path, metrics)
        comparisons = {}
        comparison_paths = {}
        for endpoint, ids in endpoints.items():
            for level in range(1, 5):
                groups = {
                    arm: [condition(arm, seed, level) for seed in SEEDS]
                    for arm in ("error", "correct")
                }
                groups["reference"] = ["reference"]
                print(
                    f"Scoring {endpoint}, budget{level*10}:10000paired draws with independent check.",
                    flush=True,
                )
                result = compare(ids, gold, runs, groups)
                is_primary = endpoint == "transfer" and level == 4
                if is_primary:
                    main = [
                        i
                        for i in ids
                        if all(
                            runs[c, rep][i]["result"] is not None
                            for arm in ("error", "correct")
                            for c in groups[arm]
                            for rep in range(3)
                        )
                    ]
                    a.require(
                        main == result["common_valid_ids"],
                        "Reference changes the prespecified18-setting primary mask",
                    )
                result.update(
                    endpoint=endpoint,
                    budget_level=level,
                    pool_size=level * 10,
                    primary_contrast="error_minus_correct" if is_primary else None,
                    qualification="Only final40 transfer error-minus-correct is primary; all other contrasts are descriptive pointwise95.",
                )
                path = BASE / f"{endpoint}_b{level:02d}.json"
                a.save_new(path, result)
                comparisons[endpoint, level] = result
                comparison_paths[f"{endpoint}_b{level:02d}"] = {
                    "path": b.rel(path),
                    "sha256": a.digest(path),
                }
                brief = {
                    name: {k: res[k] for k in ("mean_difference", "macro_f1_ci95")}
                    for name, res in result["contrasts"].items()
                }
                print(
                    json.dumps(
                        {
                            "endpoint": endpoint,
                            "budget": level * 10,
                            "n": result["common_valid_n"],
                            "contrasts": brief,
                        },
                        indent=2,
                    ),
                    flush=True,
                )
        pools = a.read(b.preparation.BASE / "pools.json")["pools"]
        candidates = set(science["rosters"]["eligible_candidates"])
        selected_diagnostics = []
        for pool in pools:
            c = pool["condition"]
            for role, ids in (
                ("selected", pool["ids"]),
                ("not_selected", sorted(candidates - set(pool["ids"]))),
            ):
                keys = [(c, rep) for rep in range(3)] + [
                    ("reference", rep) for rep in range(3)
                ]
                shared = sorted(
                    i
                    for i in ids
                    if all(runs[k][i]["result"] is not None for k in keys)
                )
                scores = {
                    name: [
                        checked_metric(shared, gold, runs[cc, rep]) for rep in range(3)
                    ]
                    for name, cc in [("update", c), ("reference", "reference")]
                }
                selected_diagnostics.append(
                    {
                        "condition": c,
                        "arm": pool["arm"],
                        "seed": pool["seed"],
                        "pool_size": pool["size"],
                        "role": role,
                        "n": len(ids),
                        "shared_n": len(shared),
                        "excluded_ids": sorted(set(ids) - set(shared)),
                        "runs": scores,
                        "summary": {
                            name: {
                                metric: summarize([score[metric] for score in rows])
                                for metric in METRICS
                            }
                            for name, rows in scores.items()
                        },
                        "interpretation": "Selected texts may retrieve themselves; within-arm descriptive diagnostic, not transfer or matched between-arm estimate.",
                    }
                )
        selected_path = BASE / "candidate_diagnostics.json"
        a.save_new(selected_path, selected_diagnostics)
        added = {
            x["id"]: x for x in b.core.rows(b.preparation.BASE / "added_records.jsonl")
        }
        poolby = {x["condition"]: x for x in pools}
        retrieved = {}
        for entry in b.core.rows(b.preparation.BASE / "prompts.jsonl"):
            c = entry["condition"]
            iid = entry["item_id"]
            record_id = entry["feedback_trace"]["selected_id"]
            record = added[record_id]
            source_id = record["provenance"]["source_id"]
            a.require(
                source_id in poolby[c]["ids"] and record_id == "sq3fb-" + source_id,
                "Feedback retrieval outside frozen pool",
            )
            a.require(
                iid not in retrieved.setdefault(c, {}), "Duplicate retrieved input"
            )
            retrieved[c][iid] = {
                "source_id": source_id,
                "contains_gold_types": gold[iid] <= gold[source_id],
                "any_gold_type_overlap": bool(gold[iid] & gold[source_id]),
                "self_retrieval": iid == source_id,
                "source_type_labels": sorted(gold[source_id]),
            }
        retrieval = []
        for pool in pools:
            c = pool["condition"]
            for name, ids in {
                **all_groups,
                "selected_candidates": pool["ids"],
                "not_selected_candidates": sorted(candidates - set(pool["ids"])),
            }.items():
                rows = [retrieved[c][i] for i in ids]
                retrieval.append(
                    {
                        "condition": c,
                        "arm": pool["arm"],
                        "seed": pool["seed"],
                        "pool_size": pool["size"],
                        "group": name,
                        "n": len(ids),
                        "gold_types_contained_count": sum(
                            x["contains_gold_types"] for x in rows
                        ),
                        "any_type_overlap_count": sum(
                            x["any_gold_type_overlap"] for x in rows
                        ),
                        "self_retrieval_count": sum(x["self_retrieval"] for x in rows),
                        "gold_types_contained_rate": sum(
                            x["contains_gold_types"] for x in rows
                        )
                        / len(ids),
                        "source_label_counts": dict(
                            Counter(l for x in rows for l in x["source_type_labels"])
                        ),
                    }
                )
            a.require(
                all(retrieved[c][i]["self_retrieval"] for i in pool["ids"]),
                "Selected candidate self-delivery differs from preparation audit",
            )
        retrieval_path = BASE / "retrieval_diagnostics.json"
        a.save_new(
            retrieval_path,
            {
                "summary": retrieval,
                "per_item": retrieved,
                "interpretation": "Gold used only now to diagnose fixed saved retrieval; never to select the delivered example.",
            },
        )
        main = comparisons["transfer", 4]
        concise = {}
        for (endpoint, level), comparison in comparisons.items():
            concise[f"{endpoint}_b{level:02d}"] = {
                key: comparison[key]
                for key in (
                    "endpoint",
                    "pool_size",
                    "original_n",
                    "common_valid_n",
                    "excluded_ids",
                    "primary_contrast",
                    "groups",
                    "contrasts",
                    "bootstrap",
                )
            }
        artifacts = [own_path, selected_path, retrieval_path] + [
            ROOT[v["path"]] for v in comparison_paths.values()
        ]
        r.verify(inputs)
        report = {
            "created_utc": a.utc(),
            "status": "feedback_scoring_complete",
            "plan": b.rel(attempt / "plan.json"),
            "plan_sha256": a.digest(attempt / "plan.json"),
            "inputs": inputs,
            "tests": tested,
            "labels": LABELS,
            "pool_seeds": SEEDS,
            "decoder_repeats": [0, 1, 2],
            "transfer_clusters": clusters,
            "reference_run_count": 3,
            "update_run_count": 72,
            "all_run_metrics": b.rel(own_path),
            "comparisons": concise,
            "primary": {
                **main["contrasts"]["error_minus_correct"],
                "n": main["common_valid_n"],
                "error_mean": main["groups"]["error"]["metrics"]["macro_f1"],
                "correct_mean": main["groups"]["correct"]["metrics"]["macro_f1"],
                "reference_mean": main["groups"]["reference"]["metrics"]["macro_f1"],
                "confidence": 0.95,
                "multiplicity": "One declared primary comparison; secondary intervals descriptive pointwise95.",
            },
            "candidate_diagnostics": b.rel(selected_path),
            "retrieval_diagnostics": b.rel(retrieval_path),
            "outputs": {b.rel(p): a.digest(p) for p in artifacts},
            "actual_API_calls": 0,
            "limits": [
                "Post-selection DEV evidence; not untouched confirmation.",
                "Three fixed pool draws, each three decoder repeats; CIs condition on these observed pools/repeats.",
                "Dedicated sixth example in both updates; comparisons with five-example reference also change context size.",
                "Error selection uses contains, allowing extra predicted labels; it does not directly optimize macro-F1.",
                "Earlier budgets are dependent nested interventions; secondary intervals have no multiplicity correction.",
                "Historical small-KB adaptive rounds use different selection/control/delivery; no causal KB-size interaction from side-by-side scores.",
                "All inputs here are gold-hateful; missed-hate counts do not estimate overall binary discrimination.",
                "Selected-text repair can be retrieval of its own answer and is not transfer.",
                "Failed answers remain recorded; matched-valid scores and original-roster failure bounds are both saved.",
            ],
            "next": "Publish and independently verify all results; feedback model execution is finished.",
        }
        a.save_new(EVIDENCE, report)
        a.save_new(
            BASE / "complete.json",
            {
                "status": report["status"],
                "evidence": b.rel(EVIDENCE),
                "evidence_sha256": a.digest(EVIDENCE),
                "plan": b.rel(attempt / "plan.json"),
                "plan_sha256": a.digest(attempt / "plan.json"),
                "outputs": report["outputs"],
                "source_sha256": a.digest(__file__),
            },
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "primary": report["primary"],
                    "evidence_sha256": a.digest(EVIDENCE),
                },
                indent=2,
            ),
            flush=True,
        )
    except Exception:
        a.save_new(
            attempt / "failure.json",
            {
                "created_utc": a.utc(),
                "traceback": traceback.format_exc(),
                "source_sha256": a.digest(__file__),
                "actual_API_calls": 0,
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tests", "run"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", u.forbid), patch.object(
        socket, "create_connection", u.forbid
    ), patch.object(b.core, "live_client", u.forbid):
        tests() if args.mode == "tests" else run()

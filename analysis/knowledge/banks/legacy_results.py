"""Banks legacy results."""

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
from analysis.knowledge.banks import build as core
from analysis.knowledge.banks import resume as continuation
from src.hsrag.metrics import clean, effective_gate, score_all, score_gate, score_multilabel
from src.hsrag.schema import Result
from src.hsrag.vote import aggregate

OUT = core.ROOT / "Thesis_Log/astra_result/evidence/sq1_kb_results_20260912_v1.json"
B, SEED = (10000, 42)
POOLS = ("small", "all")


def require(ok, msg):
    core.require(ok, msg)


def stats(values):
    a = np.asarray(values, dtype=float)
    return dict(
        values=a.tolist(),
        mean=float(a.mean()),
        sd=float(a.std(ddof=1)),
        min=float(a.min()),
        max=float(a.max()),
        spread=float(np.ptp(a)),
    )


def interval(a):
    a = np.asarray(a)
    return {
        "ci95": np.percentile(a, [2.5, 97.5]).tolist(),
        "ci97_5_two_endpoint_bonferroni": np.percentile(a, [1.25, 98.75]).tolist(),
        "bootstrap_sd": float(a.std(ddof=1)),
        "fraction_above_zero": float((a > 0).mean()),
    }


def safe_ratio(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def measures(totals):
    tp, fp, fn = (totals[..., k] for k in range(3))
    require(np.all(tp + fn > 0), "A supported gold class disappeared in a resample")
    label_f1 = safe_ratio(2 * tp, 2 * tp + fp + fn)
    return dict(
        macro_f1=label_f1.mean(axis=-1),
        label_f1=label_f1,
        label_precision=safe_ratio(tp, tp + fp),
        label_recall=tp / (tp + fn),
        micro_f1=safe_ratio(2 * tp.sum(-1), (2 * tp + fp + fn).sum(-1)),
    )


def count_arrays(truth, pred):
    return np.stack([pred & truth, pred & ~truth, ~pred & truth], axis=-1).astype(np.int64)


def bootstrap(counts):
    n = counts.shape[2]
    flat = counts.transpose(2, 0, 1, 3, 4).reshape(n, -1)
    rng = np.random.default_rng(SEED)
    sums = []
    for start in range(0, B, 250):
        weights = rng.multinomial(n, np.full(n, 1 / n), size=min(250, B - start))
        sums.append((weights @ flat).reshape(-1, 3, 2, counts.shape[3], 3))
    return measures(np.concatenate(sums))


def contrast(points, draws):
    delta = points[:, 1] - points[:, 0]
    by_pool = {p: stats(points[:, pi]) for pi, p in enumerate(POOLS)}
    floor = max((v["spread"] for v in by_pool.values()))
    return {
        "pools": by_pool,
        "mean_delta_all_minus_small": float(delta.mean()),
        "mean_difference_interval": interval((draws[:, :, 1] - draws[:, :, 0]).mean(axis=1)),
        "replicate_deltas": delta.tolist(),
        "replicate_intervals": [interval(draws[:, ri, 1] - draws[:, ri, 0]) for ri in range(3)],
        "max_within_pool_replicate_range": floor,
        "absolute_mean_delta_over_range": float(abs(delta.mean()) / floor) if floor else None,
    }


def print_contrast(name, c):
    print(name, json.dumps(c), flush=True)


def audit_votes(rows):
    errors = Counter()
    for row in rows:
        attempts = []
        for pred, raws in zip(row["run_predictions"], row["raw_runs"]):
            require(1 <= len(raws) <= 3, "Unexpected raw attempt count")
            obj = Result.model_validate(pred) if pred is not None else None
            if obj is not None:
                require(obj.model_dump() == pred, "Saved vote changes under current schema")
            else:
                try:
                    Result.model_validate_json(raws[-1])
                except Exception as exc:
                    if hasattr(exc, "errors"):
                        for error in exc.errors():
                            errors[
                                "/".join((str(x) for x in error["loc"])) + ":" + error["type"]
                            ] += 1
                    else:
                        errors[type(exc).__name__] += 1
                else:
                    errors["schema-valid-last-raw-but-parser-rejected"] += 1
            attempts.append(SimpleNamespace(ok=obj is not None, result=obj))
        voted = aggregate(attempts)
        require(
            (voted.result.model_dump() if voted.result else None) == row["result"],
            f"Vote replay differs: {row['item_id']}",
        )
        for k in ("n_valid", "n_runs", "uncertain", "agreement"):
            require(getattr(voted, k) == row[k], f"Vote audit differs: {k}")
    return dict(errors)


def gate_failure_bounds(gold, data):
    ids = sorted(gold)
    entries = [
        (ri, pi, ii)
        for ri in range(3)
        for pi in range(2)
        for ii, i in enumerate(ids)
        if data[ri][pi][i]["result"] is None
    ]
    pred = np.array(
        [
            [
                [
                    bool(data[ri][pi][i]["result"]["hate"]) if data[ri][pi][i]["result"] else False
                    for i in ids
                ]
                for pi in range(2)
            ]
            for ri in range(3)
        ]
    )
    result = {"null_item_run_entries": entries, "assignments": 2 ** len(entries), "by_mapping": {}}
    for mapping in ("strict", "lenient"):
        y = np.array([effective_gate(gold[i], mapping) for i in ids], dtype=bool)
        deltas = []
        ranges = []
        for assignment in product((False, True), repeat=len(entries)):
            x = pred.copy()
            for entry, value in zip(entries, assignment):
                x[entry] = value
            point = measures(count_arrays(np.stack([y, ~y], -1), np.stack([x, ~x], -1)).sum(2))[
                "macro_f1"
            ]
            ranges.append(point.mean(0).tolist())
            deltas.append(float((point[:, 1] - point[:, 0]).mean()))
        result["by_mapping"][mapping] = {
            "delta_min": min(deltas),
            "delta_max": max(deltas),
            "pool_means_min": np.min(ranges, 0).tolist(),
            "pool_means_max": np.max(ranges, 0).tolist(),
        }
    result["meaning"] = (
        "Hypothetical binary completions of every saved null result, never actual recoding or resampling."
    )
    return result


def primary(scope, gold, data):
    ids = sorted(
        (
            i
            for i in gold
            if all((data[ri][pi][i]["result"] is not None for ri in range(3) for pi in range(2)))
        )
    )
    full_vote_ids = [
        i
        for i in ids
        if all((data[ri][pi][i]["n_valid"] == 3 for ri in range(3) for pi in range(2)))
    ]
    result = {
        "common_valid_ids": ids,
        "n_common_valid": len(ids),
        "excluded_ids": sorted(set(gold) - set(ids)),
        "all_three_votes_valid_ids": full_vote_ids,
        "n_all_three_votes_valid": len(full_vote_ids),
        "scores": {p: [] for p in POOLS},
        "all_three_votes_valid_sensitivity": {p: [] for p in POOLS},
    }
    for pi, pool in enumerate(POOLS):
        for ri in range(3):
            result["scores"][pool].append(
                score_all(
                    pd.DataFrame([gold[i]._asdict() for i in ids]), [data[ri][pi][i] for i in ids]
                )["rag"]
            )
            result["all_three_votes_valid_sensitivity"][pool].append(
                score_all(
                    pd.DataFrame([gold[i]._asdict() for i in full_vote_ids]),
                    [data[ri][pi][i] for i in full_vote_ids],
                )["rag"]
            )
    mappings = ("strict", "lenient") if scope == "gate" else ("hate_type",)
    result["contrasts"] = {}
    for mapping in mappings:
        if scope == "gate":
            labels = ["hate", "not_hate"]
            y = np.array([effective_gate(gold[i], mapping) for i in ids], dtype=bool)
            truth = np.stack([y, ~y], -1)
            x = np.array(
                [
                    [[data[ri][pi][i]["result"]["hate"] for i in ids] for pi in range(2)]
                    for ri in range(3)
                ]
            )
            pred = np.stack([x, ~x], -1)
            expected = np.array(
                [
                    [result["scores"][p][ri]["gate"][mapping]["macro_f1"] for p in POOLS]
                    for ri in range(3)
                ]
            )
        else:
            labels = result["scores"]["small"][0]["hate_type"]["labels_averaged"]
            require(
                all(
                    (
                        result["scores"][p][ri]["hate_type"]["labels_averaged"] == labels
                        for p in POOLS
                        for ri in range(3)
                    )
                ),
                "Different label rosters",
            )
            require(
                all((gold[i].gate and clean(gold[i].hate_types) is not None for i in ids)),
                "Unexpected hate-type gold mask",
            )
            truth = np.array(
                [[label in clean(gold[i].hate_types) for label in labels] for i in ids]
            )
            pred = np.array(
                [
                    [
                        [
                            [label in data[ri][pi][i]["result"]["hate_type"] for label in labels]
                            for i in ids
                        ]
                        for pi in range(2)
                    ]
                    for ri in range(3)
                ]
            )
            expected = np.array(
                [
                    [result["scores"][p][ri]["hate_type"]["macro_f1"] for p in POOLS]
                    for ri in range(3)
                ]
            )
        counts = count_arrays(truth, pred)
        points = measures(counts.sum(2))
        require(np.max(np.abs(points["macro_f1"] - expected)) < 1e-12, "Independent scorer differs")
        draws = bootstrap(counts)
        c = contrast(points["macro_f1"], draws["macro_f1"])
        c["labels"] = labels
        c["gold_support"] = truth.sum(0).tolist()
        c["per_label"] = {
            label: {
                metric: contrast(
                    points["label_" + metric][..., li], draws["label_" + metric][..., li]
                )
                for metric in ("f1", "precision", "recall")
            }
            for li, label in enumerate(labels)
        }
        c["micro_f1"] = contrast(points["micro_f1"], draws["micro_f1"])
        rng = np.random.default_rng(SEED + 1)
        for _ in range(20):
            pick = rng.choice(len(ids), len(ids), replace=True)
            expanded = measures(counts[:, :, pick].sum(2))["macro_f1"]
            for ri in range(3):
                for pi in range(2):
                    rows = [data[ri][pi][ids[k]] for k in pick]
                    score = (
                        score_gate(rows, gold, mapping)
                        if scope == "gate"
                        else score_multilabel(
                            rows, gold, "hate_types", "hate_type", fixed_labels=labels
                        )
                    )
                    require(
                        abs(expanded[ri, pi] - score.macro_f1) < 1e-12,
                        "Expanded bootstrap score differs",
                    )
        result["contrasts"][mapping] = c
        print_contrast(
            scope + "/" + mapping,
            {k: v for k, v in c.items() if k not in ("per_label", "micro_f1")},
        )
    if scope == "gate":
        result["failure_bounds"] = gate_failure_bounds(gold, data)
        print("FAILURE SENSITIVITY", json.dumps(result["failure_bounds"]), flush=True)
        result["per_source"] = {}
        for source in sorted({g.source for g in gold.values()}):
            selected = [i for i in ids if gold[i].source == source]
            result["per_source"][source] = {
                p: [
                    asdict(score_gate([data[ri][pi][i] for i in selected], gold)) for ri in range(3)
                ]
                for pi, p in enumerate(POOLS)
            }
        stable = Counter()
        changes = []
        for i in ids:
            small = [data[ri][0][i]["result"]["hate"] for ri in range(3)]
            full = [data[ri][1][i]["result"]["hate"] for ri in range(3)]
            if len(set(small)) == len(set(full)) == 1 and small[0] != full[0]:
                kind = (
                    "fixed_false_positive"
                    if not gold[i].gate and (not full[0])
                    else (
                        "new_false_positive"
                        if not gold[i].gate
                        else "fixed_false_negative" if full[0] else "new_false_negative"
                    )
                )
                stable[kind] += 1
                changes.append({"id": i, "source": gold[i].source, "kind": kind})
        result["stable_gate_changes"] = {"counts": dict(stable), "items": changes}
    else:
        result["hate_type_diagnostics"] = {p: [] for p in POOLS}
        for pi, p in enumerate(POOLS):
            for ri in range(3):
                rows = [data[ri][pi][i] for i in ids]
                result["hate_type_diagnostics"][p].append(
                    {
                        "gate_false": sum((not r["result"]["hate"] for r in rows)),
                        "empty_type_despite_positive_gate": sum(
                            (r["result"]["hate"] and (not r["result"]["hate_type"]) for r in rows)
                        ),
                        **result["scores"][p][ri]["hate_type"]["extra"],
                        "subset_accuracy": result["scores"][p][ri]["hate_type"]["subset_accuracy"],
                    }
                )
    print("MATCHED POOLS", scope, len(ids), "all-three-valid", len(full_vote_ids), flush=True)
    return result


def retrieval_audit(golds, contexts, records):
    require(set(contexts["small"]) == set(contexts["all"]), "Retrieval key mismatch")
    result = {}
    for scope, gold in golds.items():
        result[scope] = {}
        for pool in POOLS:
            examples = records[pool]
            per_item = []
            used, sources = (Counter(), Counter())
            for i, g in sorted(gold.items()):
                key = core.SCOPES[scope] + "/" + i
                a, b = (contexts["small"][key], contexts["all"][key])
                require(
                    all((a[k] == b[k] for k in ("definitions", "guidelines"))),
                    "Abstract context differs",
                )
                picked = contexts[pool][key]["Labelled examples"]
                require(len(picked) == len(set(picked)) == 5, "Wrong example budget")
                require(
                    all(
                        (
                            examples[j]["lang"] == "en" and examples[j]["kind"] == "example"
                            for j in picked
                        )
                    ),
                    "Wrong record kind/language",
                )
                used.update(picked)
                sources.update((examples[j]["source"] for j in picked))
                meta = [examples[j]["meta"] for j in picked]
                gold_types = set(clean(g.hate_types) or [])
                context_types = set((t for m in meta for t in m.get("hate_types") or []))
                per_item.append(
                    {
                        "id": i,
                        "gold_gate": bool(g.gate),
                        "source": g.source,
                        "hateful_examples": sum((m["gate"] is True for m in meta)),
                        "type_annotated_examples": sum(
                            (m.get("hate_types") is not None for m in meta)
                        ),
                        "gold_type_in_examples": (
                            bool(gold_types & context_types) if gold_types else None
                        ),
                        "shared_small_full_examples": len(
                            set(a["Labelled examples"]) & set(b["Labelled examples"])
                        ),
                    }
                )
            summary = {
                "n_inputs": len(gold),
                "total_example_slots": 5 * len(gold),
                "unique_examples_used": len(used),
                "source_counts": dict(sources),
                "most_reused_examples": used.most_common(10),
                "mean_hateful_of_five": float(np.mean([v["hateful_examples"] for v in per_item])),
                "mean_type_annotated_of_five": float(
                    np.mean([v["type_annotated_examples"] for v in per_item])
                ),
                "mean_shared_examples_of_five": float(
                    np.mean([v["shared_small_full_examples"] for v in per_item])
                ),
                "by_gold_gate": {
                    str(g): {
                        "n": sum((v["gold_gate"] == g for v in per_item)),
                        "mean_hateful_of_five": float(
                            np.mean(
                                [v["hateful_examples"] for v in per_item if v["gold_gate"] == g]
                            )
                        ),
                    }
                    for g in (False, True)
                    if any((v["gold_gate"] == g for v in per_item))
                },
            }
            covered = [
                v["gold_type_in_examples"]
                for v in per_item
                if v["gold_type_in_examples"] is not None
            ]
            summary["gold_type_coverage"] = float(np.mean(covered)) if covered else None
            if scope == "types":
                summary["coverage_by_type"] = {}
                for label in sorted({t for g in gold.values() for t in clean(g.hate_types)}):
                    subset = [v for v in per_item if label in clean(gold[v["id"]].hate_types)]
                    summary["coverage_by_type"][label] = {
                        "n": len(subset),
                        "coverage": float(np.mean([v["gold_type_in_examples"] for v in subset])),
                    }
            result[scope][pool] = {"summary": summary, "per_item": per_item}
            print("RETRIEVAL", scope, pool, json.dumps(summary), flush=True)
    return result


def historical_context(golds, endpoints, paths):
    specs = {
        "gate": ["main_base_r0", "main_base_r0_rep2", "main_base_r0_rep3", "main_base_r0_rep4"],
        "types": ["strat_hybrid_r1", "strat_hybrid_r2", "strat_hybrid_r3"],
    }
    out = {}
    for scope, stems in specs.items():
        entries = []
        saved = []
        for stem in stems:
            path = core.ROOT / "experiments/results" / (stem + "_live.jsonl")
            if not path.exists():
                entries.append({"stem": stem, "status": "file unavailable; no inferred score"})
                continue
            paths.add(path)
            rows = [r for r in core.read_lines(path) if r.get("arm") == "rag"]
            gold = golds[scope]
            mapped = {r["item_id"]: r for r in rows}
            require(
                len(mapped) == len(rows) and set(mapped) == set(gold),
                "Historical input pool differs",
            )
            require(
                all((r["text_hash"] == core.item_text_hash(gold[r["item_id"]].text) for r in rows)),
                "Historical input text differs",
            )
            saved.append((stem, rows, mapped))
        ids = [
            i
            for i in endpoints[scope]["common_valid_ids"]
            if all((mapped[i]["result"] is not None for _, _, mapped in saved))
        ]
        for stem, rows, mapped in saved:
            matched = [mapped[i] for i in ids]
            own = (
                asdict(score_gate(rows, gold))
                if scope == "gate"
                else asdict(score_multilabel(rows, gold, "hate_types", "hate_type"))
            )
            aligned = (
                asdict(score_gate(matched, gold))
                if scope == "gate"
                else asdict(score_multilabel(matched, gold, "hate_types", "hate_type"))
            )
            entries.append(
                {
                    "stem": stem,
                    "own_pool_score": own,
                    "historical_joint_pool_score": aligned,
                    "invalid_ids": [i for i, r in mapped.items() if r["result"] is None],
                    "stamps": {
                        k: sorted({r.get(k) for r in rows}, key=str)
                        for k in (
                            "code_version",
                            "prompt_version",
                            "taxonomy_version",
                            "kb_version",
                            "active_model",
                        )
                    },
                }
            )
        out[scope] = {
            "runs": entries,
            "historical_joint_pool_ids": ids,
            "new_runs_on_historical_joint_pool": {p: [] for p in POOLS},
        }
        plan = core.read_json(core.PLAN)
        for run in plan["runs"]:
            if run["scope"] != scope:
                continue
            rows = [
                r for r in core.read_lines(core.ROOT / run["output"]) if r.get("item_id") in ids
            ]
            score = (
                asdict(score_gate(rows, gold))
                if scope == "gate"
                else asdict(score_multilabel(rows, gold, "hate_types", "hate_type"))
            )
            out[scope]["new_runs_on_historical_joint_pool"][run["pool"]].append(
                {"replicate": run["replicate"], "score": score}
            )
        print(
            "HISTORICAL",
            scope,
            "joint n=",
            len(ids),
            [
                (
                    r["stem"],
                    r.get("own_pool_score", {}).get("macro_f1"),
                    r.get("historical_joint_pool_score", {}).get("macro_f1"),
                )
                for r in entries
            ],
            flush=True,
        )
    return out


def main():
    require(not OUT.exists(), f"Refusing to overwrite {OUT}")
    plan = continuation.validate()
    batch = core.read_json(core.BASE / "batch_complete.json")
    require(batch["plan_sha256"] == core.digest(core.PLAN), "Batch plan hash differs")
    require(
        batch["continuation_policy_sha256"] == core.digest(continuation.POLICY),
        "Batch continuation hash differs",
    )
    require(
        set(batch["result_hashes"]) == {r["output"] for r in plan["runs"]},
        "Batch output list differs",
    )
    paths = {Path(__file__), core.PLAN, continuation.POLICY, core.BASE / "batch_complete.json"}
    paths.update((core.ROOT / name for name in plan["input_hashes"]))
    paths.update((core.ROOT / name for name in plan["generated_hashes"]))
    golds = {
        scope: {
            str(g.id): g
            for g in pd.read_parquet(
                core.ROOT / "data/processed" / (subset + ".parquet")
            ).itertuples(index=False)
        }
        for scope, subset in core.SCOPES.items()
    }
    loaded = {scope: [[None for _ in POOLS] for _ in range(3)] for scope in core.SCOPES}
    run_audits = []
    for run in plan["runs"]:
        path = core.ROOT / run["output"]
        require(
            core.digest(path) == batch["result_hashes"][run["output"]],
            "Completed output hash differs",
        )
        info = core.check_output(plan, run)
        require(info["missing"] == 0, "Missing result item")
        rows = [r for r in core.read_lines(path) if "item_id" in r]
        quality_path = continuation.HERE / "quality" / (run["name"] + ".json")
        q = core.read_json(quality_path)
        require(q["result_sha256"] == core.digest(path), "Quality hash differs")
        require(all((q[k] == v for k, v in info.items())), "Quality counts differ")
        failed = [r["item_id"] for r in rows if r["result"] is None]
        require(q["invalid_item_ids"] == failed, "Quality failed IDs differ")
        errors = audit_votes(rows)
        df = pd.DataFrame([g._asdict() for g in golds[run["scope"]].values()])
        rescored = score_all(df, rows)
        metric_path = path.with_suffix(".metrics.json")
        require(
            json.loads(json.dumps(rescored)) == core.read_json(metric_path),
            "Saved metrics differ from offline rescore",
        )
        paths.update((path, metric_path, quality_path))
        loaded[run["scope"]][run["replicate"] - 1][POOLS.index(run["pool"])] = {
            r["item_id"]: r for r in rows
        }
        audit = {
            **run,
            **info,
            "invalid_item_ids": failed,
            "invalid_gold": {
                i: {
                    "gate": bool(golds[run["scope"]][i].gate),
                    "source": golds[run["scope"]][i].source,
                }
                for i in failed
            },
            "valid_vote_counts": dict(Counter((str(r["n_valid"]) for r in rows))),
            "last_raw_failed_vote_schema_errors": errors,
            "saved_raw_response_count": sum(
                (len(attempts) for r in rows for attempts in r["raw_runs"])
            ),
            "saved_prediction_timestamp_span_minutes": (
                max((r["timestamp"] for r in rows)) - min((r["timestamp"] for r in rows))
            )
            / 60,
            "rescored": rescored["rag"],
        }
        run_audits.append(audit)
        print(
            "VERIFIED",
            run["scope"],
            run["pool"],
            run["replicate"],
            json.dumps(
                {
                    k: audit[k]
                    for k in (
                        "written",
                        "invalid_item_ids",
                        "valid_vote_counts",
                        "last_raw_failed_vote_schema_errors",
                        "saved_raw_response_count",
                    )
                }
            ),
            flush=True,
        )
    before = {p: core.digest(p) for p in paths}
    endpoints = {scope: primary(scope, golds[scope], loaded[scope]) for scope in core.SCOPES}
    contexts = {p: core.read_json(core.BASE / ("retrieval_" + p + ".json")) for p in POOLS}
    records = {
        p: {r["id"]: r for r in core.read_lines(core.BASE / "kb" / p / "records.jsonl")}
        for p in POOLS
    }
    retrieved = retrieval_audit(golds, contexts, records)
    historical = historical_context(golds, endpoints, paths)
    paths.update((core.BASE / ("retrieval_" + p + ".json") for p in POOLS))
    require(all((core.digest(p) == h for p, h in before.items())), "Input changed during analysis")
    hashes = {core.rel(p): core.digest(p) for p in sorted(paths)}
    evidence = {
        "audit_id": "ASTRA-SQ1-RESULTS-20260912-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "method": {
            "bootstrap_resamples": B,
            "seed": SEED,
            "sampling": "multinomial item multiplicities; sorted common valid IDs; jointly paired across all six predictions",
            "estimand": "Mean of three replicate macro-F1 scores, full pool minus small pool. Not a pooled-vote ensemble.",
            "labels": "Full-sample supported labels frozen in every draw; no new support filtering.",
            "uncertainty": "Percentile item intervals conditional on these three realized runs; decoder/provider population uncertainty is not identified.",
            "multiplicity": "Report 95% intervals and 97.5% per-endpoint intervals for a Bonferroni sensitivity across the two declared primary endpoints. Other analyses exploratory.",
            "range": "Maximum within-pool replicate range is descriptive, not a statistical threshold.",
            "independent_verification": "All points plus 20 expanded item resamples per endpoint/mapping matched project scoring within 1e-12.",
        },
        "provenance": {
            "frozen_plan_and_continuation_valid": True,
            "all_12_result_hashes_match_completion": True,
            "all_4596_saved_aggregates_replayed": True,
            "all_12_metrics_exactly_reproduced": True,
            "all_contexts_equal_to_preflight": True,
            "abstract_context_equal_across_pools": True,
            "hashes": hashes,
        },
        "runs": run_audits,
        "endpoints": endpoints,
        "retrieval": retrieved,
        "historical_context": historical,
        "limitations": [
            "Post-hoc development sensitivity, not a new held-out test result.",
            "KB size, class balance, source mix and annotation richness change together.",
            "Small control has 148 eligible examples; historical KB has 149. Historical comparisons are descriptive across dates.",
            "Adjacent replicate pairing does not imply shared decoder seeds or identical service state.",
            "Failed predictions are retained; complete-case gate comparison may select a different population. Exhaustive binary bounds reported.",
            "Retrieved-label coverage is a diagnostic association, not causal mediation evidence.",
            "No fresh semantic-near-duplicate scan. Original saved exclusions and stricter normalized held-out overlaps were removed during preparation.",
            "No claim of improvement over encoders or of generalization to German, HateCheck or held-out test data.",
        ],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    print("AUDIT PASSED. NEW EVIDENCE:", core.rel(OUT), "SHA256:", core.digest(OUT), flush=True)


if __name__ == "__main__":
    main()

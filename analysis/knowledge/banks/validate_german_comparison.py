from collections import Counter
from datetime import datetime
from itertools import product
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from analysis.knowledge.banks import german_resume as run
from src.hsrag.metrics import clean, score_all
from src.hsrag.parse import parse_response, REPAIR_TEMPLATE
from src.hsrag.vote import aggregate

p, u, ROOT, BASE = run.p, run.u, run.ROOT, run.BASE
OUT = (
    ROOT
    / "Thesis_Log/astra_result/evidence/german_kb_comparison_results_20260914_v1.json"
)
TABLE = ROOT / "Thesis_Log/astra_result/astra_german_kb_tables_20260914_v1.md"
CONDS = list(p.CONDITIONS)
CONTRASTS = {"B-A": (0, 1), "C-B": (1, 2), "D-C": (2, 3), "D-A": (0, 3)}
N_BOOT, SEED = 10000, 42


def stats(x):
    a = np.asarray(x, dtype=float)
    return {
        "values": a.tolist(),
        "mean": float(a.mean()),
        "sd": float(a.std(ddof=1)),
        "min": float(a.min()),
        "max": float(a.max()),
        "range": float(np.ptp(a)),
    }


def divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)


def measures(counts):
    tp, fp, fn = (counts[..., k] for k in range(3))
    f = divide(2 * tp, 2 * tp + fp + fn)
    return {
        "macro_f1": f.mean(-1),
        "label_f1": f,
        "precision": divide(tp, tp + fp),
        "recall": divide(tp, tp + fn),
        "micro_f1": divide(2 * tp.sum(-1), (2 * tp + fp + fn).sum(-1)),
    }


def counts(y, x):
    return np.stack([x & y, x & ~y, ~x & y], -1).astype(np.int64)


def audit_raw(row, entry, call_shapes, errors):
    attempts, counter = [], Counter()
    for saved, raw_group in zip(row["run_predictions"], row["raw_runs"]):
        p.require(1 <= len(raw_group) <= 3, "Invalid repair attempt count")
        convo = list(entry["messages"])
        for k, raw in enumerate(raw_group):
            call_shapes[(sum(len(m["text"]) for m in convo), len(raw))] += 1
            parsed, normalisations, err = parse_response(raw)
            counter["normalisations"] += normalisations
            if k < len(raw_group) - 1:
                p.require(parsed is None, "A successful vote was resampled")
                convo += [
                    {"role": "assistant", "text": raw},
                    {"role": "user", "text": REPAIR_TEMPLATE.format(error=err)},
                ]
        p.require(
            (parsed.model_dump() if parsed else None) == saved,
            "Final raw parse differs from saved vote",
        )
        p.require(
            parsed is not None or len(raw_group) == 3,
            "Failure ended before repair limit",
        )
        counter["repairs"] += len(raw_group) - 1
        counter["parse_failures"] += parsed is None
        counter["gate_normalised"] += int(parsed.gate_normalised) if parsed else 0
        if parsed is None:
            errors[
                (
                    err.split(" at ")[-1]
                    if err.startswith("schema violation")
                    else err.split(":")[0]
                )
            ] += 1
        attempts.append(SimpleNamespace(ok=parsed is not None, result=parsed))
    for k in ("normalisations", "repairs", "parse_failures", "gate_normalised"):
        p.require(counter[k] == row[k], f"Raw accounting differs: {k}")
    voted = aggregate(attempts)
    p.require(
        (voted.result.model_dump() if voted.result else None) == row["result"],
        "Aggregate replay differs",
    )
    for k in ("n_valid", "n_runs", "uncertain", "agreement"):
        p.require(getattr(voted, k) == row[k], f"Vote accounting differs: {k}")


def read_api_log():
    records, anomalies = [], []
    for line_number, line in enumerate(
        (BASE / "api_log.jsonl").read_text(encoding="utf-8").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError as err:
            anomalies.append({"line": line_number, "raw": line, "error": str(err)})

    p.require(
        all(a["raw"] == "}" for a in anomalies),
        "Unexplained API log corruption needs review",
    )
    return records, anomalies


def matrix(scope, gold, data, ids, labels):
    if scope == "gate":
        y = np.array([bool(gold[i].gate) for i in ids])
        x = np.array(
            [
                [[data[r][c][i]["result"]["hate"] for i in ids] for c in range(4)]
                for r in range(3)
            ]
        )
        return np.stack([y, ~y], -1), np.stack([x, ~x], -1)
    y = np.array([[l in clean(gold[i].legal) for l in labels] for i in ids])
    x = np.array(
        [
            [
                [[l in data[r][c][i]["result"]["legal"] for l in labels] for i in ids]
                for c in range(4)
            ]
            for r in range(3)
        ]
    )
    return y, x


def bootstrap(cc):
    n = cc.shape[2]
    flat = cc.transpose(2, 0, 1, 3, 4).reshape(n, -1)
    rng = np.random.default_rng(SEED)
    totals = []
    for start in range(0, N_BOOT, 250):
        weights = rng.multinomial(n, np.full(n, 1 / n), size=min(250, N_BOOT - start))
        totals.append((weights @ flat).reshape(-1, 3, 4, cc.shape[3], 3))
    totals = np.concatenate(totals)
    p.require(
        np.all(totals[..., 0] + totals[..., 2] > 0),
        "A fixed gold class disappeared in bootstrap",
    )
    return measures(totals)["macro_f1"]


def contrasts(point, draws=None):
    out = {}
    for name, (a, b) in CONTRASTS.items():
        delta = point[:, b] - point[:, a]
        floor = max(float(np.ptp(point[:, a])), float(np.ptp(point[:, b])))
        row = {
            "mean_delta": float(delta.mean()),
            "replicate_deltas": delta.tolist(),
            "max_condition_repeat_range": floor,
            "abs_delta_over_range": float(abs(delta.mean()) / floor) if floor else None,
        }
        if draws is not None:
            d = (draws[:, :, b] - draws[:, :, a]).mean(1)
            row["paired_item_ci95"] = np.percentile(d, [2.5, 97.5]).tolist()
            if name != "D-A":
                row["bonferroni_six_planned_comparisons_ci"] = np.percentile(
                    d, [100 * 0.05 / 12, 100 * (1 - 0.05 / 12)]
                ).tolist()
            row["replicate_ci95"] = [
                np.percentile(draws[:, r, b] - draws[:, r, a], [2.5, 97.5]).tolist()
                for r in range(3)
            ]
        out[name] = row
    return out


def compare(scope, df, data):
    gold = {str(g.id): g for g in df.itertuples(index=False)}
    ids = sorted(
        i
        for i in gold
        if all(data[r][c][i]["result"] is not None for r in range(3) for c in range(4))
    )
    full_votes = [
        i
        for i in ids
        if all(data[r][c][i]["n_valid"] == 3 for r in range(3) for c in range(4))
    ]
    labels = (
        ["hate", "not_hate"]
        if scope == "gate"
        else sorted({l for i in gold for l in clean(gold[i].legal)})
    )
    p.require(scope != "legal" or len(labels) == 3, "Unexpected legal label roster")
    y, x = matrix(scope, gold, data, ids, labels)
    cc = counts(y, x)
    total = cc.sum(2)
    point = measures(total)
    draw = bootstrap(cc)
    report = {
        "labels": labels,
        "all_items": len(gold),
        "n_common_valid": len(ids),
        "common_valid_ids": ids,
        "excluded_ids": sorted(set(gold) - set(ids)),
        "gold_supports": y.sum(0).tolist(),
        "n_all_votes_valid": len(full_votes),
        "conditions": {},
        "contrasts": contrasts(point["macro_f1"], draw),
    }
    for c, condition in enumerate(CONDS):
        evidence = {
            "macro_f1": stats(point["macro_f1"][:, c]),
            "micro_f1": stats(point["micro_f1"][:, c]),
            "per_label": {},
        }
        for r in range(3):
            subset = df[df["id"].astype(str).isin(ids)]
            scored = score_all(
                subset, [data[r][c][i] for i in ids], gate_mapping="strict"
            )["rag"]
            scored = scored["gate"]["strict"] if scope == "gate" else scored["legal"]
            p.require(
                abs(scored["macro_f1"] - point["macro_f1"][r, c]) < 1e-12,
                "Independent F1 arithmetic differs",
            )
            if scope == "legal":
                p.require(
                    scored["labels_averaged"] == labels, "Different macro label roster"
                )
            for j, label in enumerate(labels):
                fields = (
                    scored["per_class"][label]
                    if scope == "gate"
                    else scored["per_label"][label]
                )
                p.require(
                    [fields[k] for k in ("tp", "fp", "fn")] == total[r, c, j].tolist(),
                    "Independent confusion counts differ",
                )
        for j, label in enumerate(labels):
            evidence["per_label"][label] = {
                k: stats(point[k][:, c, j]) for k in ("label_f1", "precision", "recall")
            }
            evidence["per_label"][label].update(
                {
                    k: stats(total[:, c, j, ki])
                    for ki, k in enumerate(("tp", "fp", "fn"))
                }
            )
        same = np.all(x[0, c] == x[1, c], -1) & np.all(x[0, c] == x[2, c], -1)
        evidence["items_varying_between_repeats"] = int((~same).sum())
        evidence["mean_label_count"] = (
            float(x[:, c].sum(-1).mean()) if scope == "legal" else None
        )
        if scope == "legal":
            evidence["exact_set_accuracy"] = stats(np.all(x[:, c] == y, -1).mean(-1))
            evidence["gold_empty_items"] = int((~y.any(-1)).sum())
            evidence["any_legal_false_positives"] = stats(
                (x[:, c].any(-1) & ~y.any(-1)).sum(-1)
            )
            evidence["missed_all_gold_legal"] = stats(
                (~x[:, c].any(-1) & y.any(-1)).sum(-1)
            )
        else:
            evidence["per_source"] = {}
            for source in sorted({gold[i].source for i in ids}):
                mask = np.array([gold[i].source == source for i in ids])
                st = cc[:, c, mask].sum(1)
                evidence["per_source"][source] = {
                    "n": int(mask.sum()),
                    "positive": int(y[mask, 0].sum()),
                    "macro_f1": stats(measures(st)["macro_f1"]),
                    "fp": stats(st[:, 0, 1]),
                    "fn": stats(st[:, 0, 2]),
                }
        report["conditions"][condition] = evidence
    fy, fx = matrix(scope, gold, data, full_votes, labels)
    fpoint = measures(counts(fy, fx).sum(2))["macro_f1"]
    report["all_three_votes_valid_sensitivity"] = {
        "n": len(full_votes),
        "conditions": {c: stats(fpoint[:, ci]) for ci, c in enumerate(CONDS)},
        "contrasts": contrasts(fpoint),
    }

    if scope == "legal":
        missing = [
            (r, c, i)
            for r in range(3)
            for c in range(4)
            for i in sorted(gold)
            if data[r][c][i]["result"] is None
        ]
        p.require(len(missing) <= 3, "Too many nulls for exact bounded sensitivity")
        all_ids = sorted(gold)
        ay = np.array([[l in clean(gold[i].legal) for l in labels] for i in all_ids])
        ax = np.array(
            [
                [
                    [
                        [
                            l in (data[r][c][i]["result"] or {"legal": []})["legal"]
                            for l in labels
                        ]
                        for i in all_ids
                    ]
                    for c in range(4)
                ]
                for r in range(3)
            ]
        )
        pts = []
        for assignment in product(range(2 ** len(labels)), repeat=len(missing)):
            xp = ax.copy()
            for (r, c, i), bits in zip(missing, assignment):
                xp[r, c, all_ids.index(i)] = [
                    bool(bits & (1 << j)) for j in range(len(labels))
                ]
            pts.append(measures(counts(ay, xp).sum(2))["macro_f1"])
        pts = np.asarray(pts)
        report["null_completion_bounds"] = {
            "entries": missing,
            "assignments": len(pts),
            "meaning": "All hypothetical legal label-set completions on all 175 items; no predictions altered or resampled.",
            "contrasts": {
                name: {
                    "min_delta": float((pts[:, :, b] - pts[:, :, a]).mean(-1).min()),
                    "max_delta": float((pts[:, :, b] - pts[:, :, a]).mean(-1).max()),
                }
                for name, (a, b) in CONTRASTS.items()
            },
        }
    return report


def main():
    p.require(
        not OUT.exists() and not TABLE.exists(),
        "Evidence already exists; do not overwrite",
    )
    plan = run.validate()
    p.require(run.complete_check(plan), "Batch not complete")
    frozen = {**plan["input_hashes"], **plan["generated_hashes"]}
    for receipt in ("execution_v2.json", "setup_complete.json", "batch_complete.json"):
        rr = u.read_json(BASE / receipt)
        frozen.update(rr.get("hashes", rr.get("input_hashes", {})))
    files = [
        BASE / x
        for x in (
            "plan.json",
            "execution_v2.json",
            "setup_complete.json",
            "setup_review.json",
            "batch_complete.json",
            "api_log.jsonl",
        )
    ]
    files += list((BASE / "logs").glob("*.log")) + list(
        (BASE / "runs").glob("*/items/*.json")
    )
    files += [
        Path(__file__),
        ROOT / "src/hsrag/parse.py",
        ROOT / "src/hsrag/vote.py",
        ROOT / "src/hsrag/metrics.py",
    ]
    frozen.update({u.rel(f): u.digest(f) for f in files})
    dfs = {
        scope: pd.read_parquet(ROOT / f"data/processed/{subset}.parquet")
        for scope, subset in p.SCOPES.items()
    }
    data = {scope: [[None] * 4 for _ in range(3)] for scope in p.SCOPES}
    summaries, raw_shapes, errors, failures = [], Counter(), Counter(), []
    manifests = {}
    first_t, last_t = [], []
    for spec in plan["runs"]:
        rows, missing = run.check_run(plan, spec)
        p.require(not missing, "Missing item files")
        rr = BASE / f"results/{spec['name']}_live.jsonl"
        saved = u.read_lines(rr)
        p.require(
            len(saved) == len(rows) + 1 and saved[0].keys() == {"_manifest"},
            "Unexpected result header",
        )
        p.require(saved[1:] == rows, "Results differ from individual saved items")
        manifest = saved[0]["_manifest"]
        for k, v in {
            **spec,
            **run.stamps(plan, spec),
            "arms": ["rag"],
            "provider": plan["provider"],
            "pinned_model": plan["model"],
            "temperature": 1.0,
            "n_votes": 3,
            "workers": 8,
            "retrieval": plan["retrieval"],
            "gate_mapping": "strict",
            "plan_sha256": u.digest(p.PLAN),
        }.items():
            p.require(manifest[k] == v, f"Manifest differs: {k}")
        manifests[spec["name"]] = manifest
        prompt = {e["item_id"]: e for e in run.entries(spec)}
        local_shapes, local_errors = Counter(), Counter()
        for row in rows:
            audit_raw(row, prompt[row["item_id"]], local_shapes, local_errors)
            p.require(
                len(row["latencies"]) == 3 and all(v >= 0 for v in row["latencies"]),
                "Invalid vote latencies",
            )
            if row["result"] is None:
                failures.append(
                    {
                        "run": spec["name"],
                        "id": row["item_id"],
                        "raw_attempts": sum(map(len, row["raw_runs"])),
                    }
                )
        raw_shapes.update(local_shapes)
        errors.update(local_errors)
        fresh = score_all(dfs[spec["scope"]], rows, gate_mapping="strict")

        p.require(
            json.loads(json.dumps(fresh))
            == u.read_json(BASE / f"results/{spec['name']}_live.metrics.json"),
            "Saved metrics differ from recomputation",
        )
        metric = (
            fresh["rag"]["gate"]["strict"]
            if spec["scope"] == "gate"
            else fresh["rag"]["legal"]
        )
        summaries.append(
            {
                **spec,
                "metric": metric,
                "honesty": fresh["rag"]["honesty"],
                "n_valid_counts": dict(Counter(r["n_valid"] for r in rows)),
                "null_ids": [r["item_id"] for r in rows if r["result"] is None],
                "raw_attempts": sum(local_shapes.values()),
                "final_parse_error_locations": dict(local_errors),
                "earliest_item_timestamp": min(r["timestamp"] for r in rows),
            }
        )
        first_t.extend(r["timestamp"] for r in rows)
        data[spec["scope"]][spec["replicate"] - 1][CONDS.index(spec["condition"])] = {
            r["item_id"]: r for r in rows
        }
        print(
            f"AUDITED {spec['name']}: {len(rows)} items, F1={metric['macro_f1']:.6f}",
            flush=True,
        )
    api, api_anomalies = read_api_log()
    p.require(
        all(
            r["model"] == plan["model"] and r["provider"] == plan["provider"]
            for r in api
        ),
        "API model/provider differs",
    )
    api_shapes = Counter((r["prompt_chars"], r["response_chars"]) for r in api)
    missing_shapes, extra_shapes = raw_shapes - api_shapes, api_shapes - raw_shapes
    p.require(
        not extra_shapes, "API log contains calls not accounted for by saved responses"
    )
    reconciliation = {
        "status": (
            "complete"
            if not missing_shapes and not api_anomalies
            else "incomplete_shared_log"
        ),
        "saved_response_calls": sum(raw_shapes.values()),
        "valid_api_log_entries": len(api),
        "missing_api_entries": sum(missing_shapes.values()),
        "extra_api_entries": sum(extra_shapes.values()),
        "missing_shapes": [
            {"prompt_chars": k[0], "response_chars": k[1], "count": v}
            for k, v in sorted(missing_shapes.items())
        ],
        "meaning": "All saved responses parse/vote/score and are individually hashed. Shared API log is supplementary and incomplete; use saved raw groups for call totals. No original files repaired.",
    }
    failure_files = list((BASE / "runs").glob("*/failures/*.json"))
    p.require(
        not failure_files, "Request failures need separate orphan-call accounting"
    )
    logs = sorted((BASE / "logs").glob("*.log"))
    p.require(
        len(logs) == 24, "More than one attempt per run; inspect execution history"
    )
    for spec in plan["runs"]:
        matching = [
            f
            for f in logs
            if f'RUN COMPLETE {spec["name"]}:' in f.read_text(encoding="utf-8")
        ]
        p.require(len(matching) == 1, "Run completion logs differ")
    report = {
        "id": "ASTRA-GERMAN-COMPARISON-RESULTS-20260914-v1",
        "created_utc": u.now(),
        "input_hashes": frozen,
        "runs": summaries,
        "null_results": failures,
        "execution": {
            "runs": len(summaries),
            "item_results": sum(r["n_items"] for r in summaries),
            "ordinary_votes": 11700,
            "saved_raw_responses": sum(raw_shapes.values()),
            "valid_api_log_entries": len(api),
            "api_trace_reconciliation": reconciliation,
            "repairs": sum(r["honesty"]["repairs"] for r in summaries),
            "failed_votes": sum(r["honesty"]["parse_failures"] for r in summaries),
            "null_items": len(failures),
            "uncertain_gate_ties": sum(r["honesty"]["uncertain"] for r in summaries),
            "final_parse_error_locations": dict(errors),
            "request_failure_files": len(failure_files),
            "run_logs": len(logs),
            "api_log_anomalies": api_anomalies,
            "elapsed_first_item_to_batch_complete_hours": (
                datetime.fromisoformat(
                    u.read_json(BASE / "batch_complete.json")["completed_utc"]
                ).timestamp()
                - min(first_t)
            )
            / 3600,
            "mean_recorded_api_latency_seconds_incomplete_log": float(
                np.mean([r["latency_s"] for r in api])
            ),
            "checks": [
                "Every saved item hash/provenance and delivered context",
                "Frozen data/code/environment/model cache",
                "All raw attempts re-parsed, bounded repairs and free normalisations replayed",
                "All votes re-aggregated",
                "All saved metrics reproduced exactly",
                "Independent F1/confusion arithmetic on common valid items",
                "Logged API models checked; saved original/repair call multiset reconciled with explicit shared-log shortfall",
                "24 successful logs, no item request-failure markers",
            ],
        },
        "analysis_method": {
            "bootstrap_draws": N_BOOT,
            "seed": SEED,
            "unit": "Same dev item resampled jointly across four conditions and three repeats; average per-repeat F1 differences.",
            "limitations": "Intervals condition on these three completed repeats and assume independent dev items; generation/server variation and text-family dependence are not fully captured. Exploratory dev evaluation. D-A is descriptive; six planned endpoint/adjacent-condition contrasts also receive Bonferroni intervals.",
        },
        "scopes": {
            scope: compare(scope, dfs[scope], data[scope]) for scope in p.SCOPES
        },
    }
    for name, h in frozen.items():
        p.require(u.digest(ROOT / name) == h, f"Input changed during analysis: {name}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("x", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    lines = [
        "# German KB comparison: audited dev results (2026-09-14)",
        "",
        "A = cleaned original small / legacy display; B = same small / fixed display; C = canonical small; D = canonical full.",
        "Raw scores retain each run's original denominator. Common-item scores use the same valid items across all four conditions and three repeats.",
    ]
    for scope, result in report["scopes"].items():
        lines += [
            "",
            f"## {scope}",
            "",
            "| Scope | Condition | Repeat 1 raw F1 | Repeat 2 | Repeat 3 | Raw mean ± sample SD | Common-item mean ± sample SD |",
            "|---|---|---:|---:|---:|---|---|",
        ]
        for c in CONDS:
            vals = [
                r["metric"]["macro_f1"]
                for r in summaries
                if r["scope"] == scope and r["condition"] == c
            ]
            st, common = stats(vals), result["conditions"][c]["macro_f1"]
            lines.append(
                f"| {scope} | {c} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} | {st['mean']:.6f} ± {st['sd']:.6f} | {common['mean']:.6f} ± {common['sd']:.6f} |"
            )
        lines += [
            "",
            f"Common-item {scope} n={result['n_common_valid']}; excluded IDs: {result['excluded_ids']}.",
            "",
            "| Contrast | Mean F1 difference | Paired item 95% interval | Six-comparison adjusted interval | Max repeat range |",
            "|---|---:|---|---|---:|",
        ]
        for name, row in result["contrasts"].items():
            fmt = lambda v: "[" + ", ".join(f"{x:+.6f}" for x in v) + "]"
            adj = row.get("bonferroni_six_planned_comparisons_ci")
            lines.append(
                f"| {name} | {row['mean_delta']:+.6f} | {fmt(row['paired_item_ci95'])} | {fmt(adj) if adj else 'descriptive'} | {row['max_condition_repeat_range']:.6f} |"
            )
        lines += [
            "",
            "| Condition | Label | Mean precision | Mean recall | Mean F1 | Mean FP | Mean FN |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for c in CONDS:
            for label, v in result["conditions"][c]["per_label"].items():
                lines.append(
                    f"| {c} | {label} | {v['precision']['mean']:.6f} | {v['recall']['mean']:.6f} | {v['label_f1']['mean']:.6f} | {v['fp']['mean']:.3f} | {v['fn']['mean']:.3f} |"
                )
        lines += [""]
    lines += [
        "All-repeat common-item comparisons are additional analyses, not replacements for the saved raw metrics. C-B includes annotation enrichment and presentation order. D-C includes class/source/annotation mix changes.",
        "",
        "Bootstrap intervals condition on these three repeats and assume independent items. They do not establish untouched test confirmation, a pure KB-size mechanism, or superiority to matched few-shot controls.",
        "",
        f"Full evidence: `evidence/{OUT.name}`, SHA-256 `{u.digest(OUT)}`.",
    ]
    with TABLE.open("x", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(
        json.dumps(
            {
                "evidence": u.rel(OUT),
                "sha256": u.digest(OUT),
                "input_hashes": len(frozen),
                "execution": report["execution"],
                "contrasts": {s: r["contrasts"] for s, r in report["scopes"].items()},
                "table": u.rel(TABLE),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

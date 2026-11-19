"""Banks legacy diagnostics."""

from src.hsrag.paths import PROJECT_ROOT
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean
import numpy as np
import pandas as pd

ROOT = PROJECT_ROOT
SOURCE = ROOT / "Thesis_Log/astra_result/evidence/sq1_kb_results_20260912_v1.json"
OUT = SOURCE.with_name("sq1_kb_diagnostics_20260912_v1.json")
TABLE = ROOT / "Thesis_Log/astra_result/astra_sq1_kb_tables_20260912_v1.md"
B, SEED = (10000, 42)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_new(path, text):
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)


def rate_change(ids, data, error):
    small = np.array([[error(data["small"][ri][i]) for i in ids] for ri in range(3)], dtype=float)
    full = np.array([[error(data["all"][ri][i]) for i in ids] for ri in range(3)], dtype=float)
    diff = (full - small).mean(0)
    rng = np.random.default_rng(SEED)
    draws = []
    for start in range(0, B, 250):
        weights = rng.multinomial(
            len(ids), np.full(len(ids), 1 / len(ids)), size=min(250, B - start)
        )
        draws.extend((weights @ diff / len(ids)).tolist())
    return {
        "n": len(ids),
        "small_counts": small.sum(1).astype(int).tolist(),
        "all_counts": full.sum(1).astype(int).tolist(),
        "small_mean_rate": float(small.mean()),
        "all_mean_rate": float(full.mean()),
        "mean_rate_change": float(diff.mean()),
        "ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
    }


def fmt(v):
    return f"{v:.6f}"


def main():
    assert not OUT.exists() and (not TABLE.exists()), "Existing evidence is preserved"
    evidence = json.loads(SOURCE.read_text(encoding="utf-8"))
    for p, expected in evidence["provenance"]["hashes"].items():
        assert digest(ROOT / p) == expected, p
    loaded = {scope: {p: [None] * 3 for p in ("small", "all")} for scope in ("gate", "types")}
    bad = Counter()
    for r in evidence["runs"]:
        rows = [
            json.loads(line)
            for line in (ROOT / r["output"]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows = [v for v in rows if "item_id" in v]
        loaded[r["scope"]][r["pool"]][r["replicate"] - 1] = {v["item_id"]: v for v in rows}
        for v in rows:
            for pred, raws in zip(v["run_predictions"], v["raw_runs"]):
                if pred is not None:
                    continue
                try:
                    raw = json.loads(raws[-1])
                except ValueError:
                    bad["invalid_JSON"] += 1
                    continue
                for label in raw.get("hate_type", []):
                    if (
                        label
                        not in evidence["endpoints"]["types"]["contrasts"]["hate_type"]["labels"]
                    ):
                        bad[label] += 1
    gold = {
        str(g.id): g
        for g in pd.read_parquet(ROOT / "data/processed/en_dev_eval_main.parquet").itertuples(
            index=False
        )
    }
    ids = evidence["endpoints"]["gate"]["common_valid_ids"]
    rates = {}
    for source in ("all", *sorted({g.source for g in gold.values()})):
        eligible = [i for i in ids if source == "all" or gold[i].source == source]
        rates[source] = {}
        for positive, name in ((False, "false_positive"), (True, "false_negative")):
            chosen = [i for i in eligible if bool(gold[i].gate) == positive]
            rates[source][name] = rate_change(
                chosen, loaded["gate"], lambda r: bool(r["result"]["hate"]) != positive
            )
    types_ids = evidence["endpoints"]["types"]["common_valid_ids"]
    type_gate = rate_change(types_ids, loaded["types"], lambda r: not r["result"]["hate"])
    totals = {
        p: {
            k: sum((r["rescored"]["honesty"][k] for r in evidence["runs"] if r["pool"] == p))
            for k in (
                "n_items",
                "total_runs",
                "parse_failures",
                "repairs",
                "uncertain",
                "no_prediction",
            )
        }
        for p in ("small", "all")
    }
    output = {
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": digest(SOURCE),
        "analysis_script": str(Path(__file__).relative_to(ROOT)),
        "analysis_script_sha256": digest(Path(__file__)),
        "method": "10,000 paired item multinomial resamples, seed 42; three realized replicates fixed; exploratory 95% percentile intervals, no subgroup multiplicity correction.",
        "gate_error_rates": rates,
        "types_gate_false_negative": type_gate,
        "failed_vote_last_raw_invalid_labels": dict(bad),
        "totals_by_pool": totals,
    }
    print("SUPPLEMENT", json.dumps(output, indent=2), flush=True)
    lines = [
        "# Astra SQ1 KB comparison: complete run and audit tables",
        "",
        "Source: `evidence/sq1_kb_results_20260912_v1.json`; error-rate supplement: `evidence/sq1_kb_diagnostics_20260912_v1.json`.",
        "",
        "All changes are full minus small. Scores range from 0 to 1. Main comparisons use common valid inputs across all six runs; raw scores retain their original per-run valid pools. Three repeats are repeated LLM sampling runs, not independent training seeds.",
        "",
        "## All twelve runs",
        "",
        "Every suffix below has prefix `astra_sq1_kb_size_20260911_v1_` and output suffix `_live.jsonl` under `experiments/astra/sq1_kb_size_20260911_v1/results/`.",
        "",
        "| Run suffix | Attempted | Valid | Votes valid: 3/2/1/0 | Raw primary F1 | Common-pool primary F1 | Invalid item IDs |",
        "|---|---:|---:|---|---:|---:|---|",
    ]
    for r in evidence["runs"]:
        scope, p, ri = (r["scope"], r["pool"], r["replicate"] - 1)
        raw, common = (r["rescored"], evidence["endpoints"][scope]["scores"][p][ri])
        score = lambda v: (
            v["gate"]["strict"]["macro_f1"] if scope == "gate" else v["hate_type"]["macro_f1"]
        )
        votes = "/".join((str(r["valid_vote_counts"].get(str(n), 0)) for n in (3, 2, 1, 0)))
        lines.append(
            f"| {scope}_{p}_r{ri + 1} | {r['written']} | {r['written'] - r['invalid_results']} | {votes} | {fmt(score(raw))} | {fmt(score(common))} | {', '.join(r['invalid_item_ids']) or 'none'} |"
        )
    lines += [
        "",
        "## Paired uncertainty",
        "",
        "10,000 paired item resamples, seed 42. Every draw uses the same item multiplicities across both pools and all three repeats. Labels are fixed. Mean intervals condition on these stored runs and do not cover all future decoder/provider variability.",
        "",
        "| Endpoint | Small mean +/- sample SD | Full mean +/- sample SD | Mean change | 95% interval | 97.5% interval | Change / largest run range |",
        "|---|---|---|---:|---|---|---:|",
    ]
    for scope, mapping in (("gate", "strict"), ("gate", "lenient"), ("types", "hate_type")):
        c = evidence["endpoints"][scope]["contrasts"][mapping]
        pools = c["pools"]
        v = c["mean_difference_interval"]
        ci = lambda a: "[" + ", ".join((fmt(x) for x in a)) + "]"
        lines.append(
            f"| {scope}/{mapping} | {fmt(pools['small']['mean'])} +/- {fmt(pools['small']['sd'])} | {fmt(pools['all']['mean'])} +/- {fmt(pools['all']['sd'])} | {fmt(c['mean_delta_all_minus_small'])} | {ci(v['ci95'])} | {ci(v['ci97_5_two_endpoint_bonferroni'])} | {c['absolute_mean_delta_over_range']:.2f} |"
        )
    lines += [
        "",
        "The wider intervals provide a Bonferroni sensitivity for the two declared primary endpoints, strict gate and hate type. Lenient gate is a separate sensitivity; its interval is not a correction covering all analyses. The range ratio is descriptive, not a significance test.",
        "",
        "| Endpoint | Repeat pair | Change | 95% interval |",
        "|---|---:|---:|---|",
    ]
    for scope, ep in evidence["endpoints"].items():
        for mapping, c in ep["contrasts"].items():
            for ri in range(3):
                lines.append(
                    f"| {scope}/{mapping} | {ri + 1} | {fmt(c['replicate_deltas'][ri])} | {ci(c['replicate_intervals'][ri]['ci95'])} |"
                )
    lines += [
        "",
        "## Strict gate counts and error rates",
        "",
        "| Pool | Repeat | TP | FP | FN | TN |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for p, scores in evidence["endpoints"]["gate"]["scores"].items():
        for ri, s in enumerate(scores):
            g = s["gate"]["strict"]
            lines.append(f"| {p} | {ri + 1} | {g['tp']} | {g['fp']} | {g['fn']} | {g['tn']} |")
    lines += [
        "",
        "| Source | Error | Gold class n | Small mean count | Full mean count | Small rate | Full rate | Change | 95% interval |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for source, group in rates.items():
        for name, d in group.items():
            lines.append(
                f"| {source} | {name} | {d['n']} | {mean(d['small_counts']):.3f} | {mean(d['all_counts']):.3f} | {fmt(d['small_mean_rate'])} | {fmt(d['all_mean_rate'])} | {fmt(d['mean_rate_change'])} | {ci(d['ci95'])} |"
            )
    lines += [
        "",
        "## Hate-type detail",
        "",
        "| Label | Gold support | Small mean F1 | Full mean F1 | Change | 95% interval |",
        "|---|---:|---:|---:|---:|---|",
    ]
    c = evidence["endpoints"]["types"]["contrasts"]["hate_type"]
    for label, support in zip(c["labels"], c["gold_support"]):
        d = c["per_label"][label]["f1"]
        lines.append(
            f"| {label} | {support} | {fmt(d['pools']['small']['mean'])} | {fmt(d['pools']['all']['mean'])} | {fmt(d['mean_delta_all_minus_small'])} | {ci(d['mean_difference_interval']['ci95'])} |"
        )
    lines += [
        "",
        "Per-label intervals are exploratory and unadjusted. The gold annotations contain 478 labels across 467 items, so some items are multilabel.",
        "",
        "| Pool | Repeat | Gold-hateful inputs called not hate | Mean predicted type labels | Any correct type overlap | Exact type set match |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for p, ds in evidence["endpoints"]["types"]["hate_type_diagnostics"].items():
        for ri, d in enumerate(ds):
            lines.append(
                f"| {p} | {ri + 1} | {d['gate_false']} | {fmt(d['mean_pred_labels'])} | {fmt(d['any_overlap_rate'])} | {fmt(d['subset_accuracy'])} |"
            )
    lines += [
        "",
        f"On the type set, the mean false-negative-rate change is {fmt(type_gate['mean_rate_change'])}, 95% interval {ci(type_gate['ci95'])}. This all-hateful subset cannot measure binary false positives or two-class gate macro-F1.",
        "",
        "## All-three-valid-vote sensitivity",
        "",
        "| Endpoint | Matched n | Small repeat F1s | Full repeat F1s | Mean change |",
        "|---|---:|---|---|---:|",
    ]
    for scope, ep in evidence["endpoints"].items():
        getter = lambda v: (
            v["gate"]["strict"]["macro_f1"] if scope == "gate" else v["hate_type"]["macro_f1"]
        )
        small = [getter(v) for v in ep["all_three_votes_valid_sensitivity"]["small"]]
        full = [getter(v) for v in ep["all_three_votes_valid_sensitivity"]["all"]]
        lines.append(
            f"| {scope} | {ep['n_all_three_votes_valid']} | {', '.join((fmt(x) for x in small))} | {', '.join((fmt(x) for x in full))} | {fmt(mean(full) - mean(small))} |"
        )
    lines += [
        "",
        "## Delivered examples",
        "",
        "Definitions and guidelines matched across pools on all 766 inputs, and every saved prediction matched the preflight context IDs. Each input received five English examples.",
        "",
        "| Scope | Pool | Unique examples used | Mean hateful examples / 5 | Mean type-annotated examples / 5 | Any gold type in examples | Mean shared examples / 5 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for scope, pools in evidence["retrieval"].items():
        for p, v in pools.items():
            s = v["summary"]
            coverage = fmt(s["gold_type_coverage"]) if scope == "types" else "not a main endpoint"
            lines.append(
                f"| {scope} | {p} | {s['unique_examples_used']} | {s['mean_hateful_of_five']:.3f} | {s['mean_type_annotated_of_five']:.3f} | {coverage} | {s['mean_shared_examples_of_five']:.5f} |"
            )
    lines += [
        "",
        "Coverage means any gold type label appears among the five retrieved examples. It is not a relevance judgment. In the source JSON, coverage_by_type groups items carrying that label but still measures any-gold overlap, not label-specific coverage for multilabel items. No claim of higher semantic relevance follows from these diagnostics.",
        "",
        "## Historical context",
        "",
        "Historical comparisons are descriptive. The primary estimate is the contemporaneous 148-example versus 56,109-example comparison. The old KB contains 149 English examples; dates, failures and historical stamps differ.",
        "",
        "| Scope | Historical joint valid n | Historical original mean F1 | New small mean on this joint set | New full mean on this joint set |",
        "|---|---:|---:|---:|---:|",
    ]
    for scope, v in evidence["historical_context"].items():
        old = mean((r["historical_joint_pool_score"]["macro_f1"] for r in v["runs"]))
        new = {
            p: mean((r["score"]["macro_f1"] for r in rows))
            for p, rows in v["new_runs_on_historical_joint_pool"].items()
        }
        lines.append(
            f"| {scope} | {len(v['historical_joint_pool_ids'])} | {fmt(old)} | {fmt(new['small'])} | {fmt(new['all'])} |"
        )
    lines += [
        "",
        "## Execution quality and cost accounting",
        "",
        "| Pool | Item-run attempts | No prediction | Planned sample calls | Saved parse-failed votes | Repairs | Gate ties |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for p, d in totals.items():
        lines.append(
            f"| {p} | {d['n_items']} | {d['no_prediction']} | {d['total_runs']} | {d['parse_failures']} | {d['repairs']} | {d['uncertain']} |"
        )
    lines += [
        "",
        "There are 13,788 ordinary sample attempts plus 614 saved repair responses, giving 14,402 stored raw responses. This is not an audited HTTP retry count, token bill or total wall-clock runtime. Indexing cost is not included in these response counts.",
        "",
        "The 112 final failed sample votes comprise 110 with insult_defamation in hate_type, one with insult in hate_type, and one invalid JSON response. Seven item-run aggregates remained null. No failed answer was manually salvaged or regenerated.",
        "",
        "## Evidence integrity",
        "",
        f"Main evidence SHA-256: `{digest(SOURCE)}`.",
        "",
        "All 12 prediction hashes match the completion receipt. Original plan, continuation, input/config/code/model-cache hashes validated; all 4,596 aggregates replayed; all 12 saved metric documents reproduced exactly. These are saved-file checks, not a fresh semantic leakage scan or provider-side reproducibility guarantee.",
        "",
    ]
    text = "\n".join(lines)
    print(text, flush=True)
    write_new(OUT, json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_new(TABLE, text)
    print("WROTE", str(OUT.relative_to(ROOT)), digest(OUT), flush=True)
    print("WROTE", str(TABLE.relative_to(ROOT)), digest(TABLE), flush=True)


if __name__ == "__main__":
    main()

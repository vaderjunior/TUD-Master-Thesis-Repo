import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS = Path("experiments/results")
DATA = Path("data/processed/en_hatecheck_eval.parquet")
RUNNER = Path("analysis/prototype/guideline_run.py")
REPS = (1, 2, 3)
SOURCES = {
    "ZS": [(f"hatecheck_A_r{r}", "zero_shot") for r in REPS],
    "FS": [(f"hatecheck_A_r{r}", "few_shot") for r in REPS],
    "RAG": [(f"hatecheck_A_r{r}", "rag") for r in REPS],
    "G": [(f"hatecheck_G_r{r}", "rag") for r in REPS],
    **{
        a: [(f"guideoracle_{a}" + ("" if r == 1 else f"_rep{r}"), "rag") for r in REPS]
        for a in ("A", "B", "N")
    },
}
EXPECTED = {
    "active_model": "Qwen/Qwen3-VL-32B-Instruct",
    "prompt_version": "classify_v1-8e588db7",
    "taxonomy_version": "taxonomy-02dabf77",
    "temperature": 1.0,
    "n_runs": 3,
    "workers": 8,
}
KEEP = [
    "item_id",
    "result",
    "retrieved",
    "retrieval_config",
    "kb_version",
    "text_hash",
    "lang",
    "code_version",
    "timestamp",
    "n_valid",
    "parse_failures",
    "repairs",
    *EXPECTED,
]


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_mapping():

    tree = ast.parse(RUNNER.read_text(encoding="utf-8-sig"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "MAPPING" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise ValueError("Original runner has no literal MAPPING")


def load_predictions(block):
    wanted = {}
    for name, specs in SOURCES.items():
        for rep, (stem, arm) in zip(REPS, specs):
            wanted.setdefault(stem, {})[arm] = (name, rep)
    predictions = {(a, r): {} for a in SOURCES for r in REPS}
    files = []
    for stem in sorted(wanted):
        path = RESULTS / f"{stem}_live.jsonl"
        files.append(path)
        with path.open(encoding="utf-8") as handle:
            for number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if "_manifest" in row or row.get("arm") not in wanted[stem]:
                    continue
                iid = row["item_id"]
                if iid not in block:
                    continue
                key = wanted[stem][row["arm"]]
                require(iid not in predictions[key], f"Duplicate item {path}:{number}")
                predictions[key][iid] = {k: row.get(k) for k in KEEP}
    return predictions, files


def common_ids(predictions, block, arms):
    return sorted(
        set(block).intersection(
            *(
                {i for i, row in predictions[a, r].items() if row["result"] is not None}
                for a in arms
                for r in REPS
            )
        )
    )


def timestamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def check_provenance(predictions, block, meta, kb_hash, mapping):
    rows_out = []
    for (arm, rep), rows in predictions.items():
        require(set(rows) == block, f"Missing source rows for {arm} rep {rep}")
        for iid, row in rows.items():
            for field, expected in EXPECTED.items():
                require(
                    row[field] == expected,
                    f"{arm} rep {rep} {iid}: unexpected {field}={row[field]}",
                )
            require(row["lang"] == "en", f"Unexpected language: {iid}")
            if arm != "ZS" and arm != "FS":
                require(
                    row["kb_version"] == kb_hash,
                    f"KB mismatch for {arm} rep {rep} {iid}",
                )
            if arm in ("A", "B", "N"):
                got = row["retrieved"]
                functionality = meta[iid]["functionality"]
                rule_a, rule_b, filler = mapping[functionality]
                expected = (
                    [] if arm == "N" else [rule_a if arm == "A" else rule_b, filler]
                )
                require(
                    got["guidelines"] == expected == got["injected"],
                    f"Oracle injection mismatch: {arm} rep {rep} {iid}",
                )
                require(
                    got["condition"] == arm and got["functionality"] == functionality,
                    f"Oracle condition/functionality mismatch: {iid}",
                )
            if row["result"] is not None:
                require(
                    type(row["result"]["hate"]) is bool,
                    f"Non-boolean hate result: {iid}",
                )
        times = [r["timestamp"] for r in rows.values()]
        rows_out.append(
            {
                "arm": arm,
                "replicate": rep,
                "stem": SOURCES[arm][rep - 1][0],
                "source_arm": SOURCES[arm][rep - 1][1],
                "source_rows_in_block": len(rows),
                "unscored_ids": sorted(
                    i for i, r in rows.items() if r["result"] is None
                ),
                "code_versions": sorted(
                    {str(r["code_version"]) for r in rows.values()}
                ),
                "kb_versions": sorted({str(r["kb_version"]) for r in rows.values()}),
                "n_valid_distribution": dict(
                    sorted(Counter(r["n_valid"] for r in rows.values()).items())
                ),
                "parse_failures_sum": sum(r["parse_failures"] for r in rows.values()),
                "repairs_sum": sum(r["repairs"] for r in rows.values()),
                "first_timestamp_utc": timestamp(min(times)),
                "last_timestamp_utc": timestamp(max(times)),
            }
        )
    return rows_out


def check_context(predictions, ids, arms, reference, kb_ids):
    counts = {a: 0 for a in arms}
    baseline = {i: predictions[reference, 1][i] for i in ids}
    for a in arms:
        for rep in REPS:
            for iid in ids:
                row, ref = predictions[a, rep][iid], baseline[iid]
                require(
                    row["text_hash"] == ref["text_hash"], f"Input text mismatch: {iid}"
                )
                for field, length in [("definitions", 2), ("Labelled examples", 5)]:
                    got = row["retrieved"].get(field)
                    expected = ref["retrieved"].get(field)
                    require(
                        isinstance(got, list) and len(got) == length,
                        f"Unexpected {field} count: {a} rep {rep} {iid}",
                    )
                    require(
                        got == expected,
                        f"Ordered {field} mismatch: {a} rep {rep} {iid}",
                    )
                    require(
                        all(k in kb_ids for k in got), f"Missing KB record for {iid}"
                    )

                cfg = {
                    k: v
                    for k, v in row["retrieval_config"].items()
                    if k != "k_guidelines"
                }
                ref_cfg = {
                    k: v
                    for k, v in ref["retrieval_config"].items()
                    if k != "k_guidelines"
                }
                require(
                    cfg == ref_cfg,
                    f"Other retrieval settings mismatch: {a} rep {rep} {iid}",
                )
                counts[a] += 1
    return {
        "items": len(ids),
        "reference": f"{reference} replicate 1",
        "item_replicate_matches_per_arm": counts,
        "checks": [
            "same input text hash",
            "same ordered two definition IDs",
            "same ordered five labelled example IDs",
            "all IDs in matching KB",
            "same retrieval config except k_guidelines",
        ],
    }


def analyse_pool(name, ids, arms, pairs, predictions, meta, draws, seed):
    require(bool(ids), f"Empty comparison pool {name}")
    tids = sorted({meta[i]["templ_id"] for i in ids})
    tindex = {t: k for k, t in enumerate(tids)}
    sizes = np.zeros(len(tids), dtype=np.int64)
    grouped = {a: np.zeros((len(tids), 3), dtype=np.int64) for a in arms}
    counts = {a: np.zeros(3, dtype=np.int64) for a in arms}
    discordant = {}
    for iid in ids:
        k = tindex[meta[iid]["templ_id"]]
        sizes[k] += 1
        for a in arms:
            vals = [int(predictions[a, r][iid]["result"]["hate"]) for r in REPS]
            grouped[a][k] += vals
            counts[a] += vals
    rng = np.random.default_rng(seed)
    boot = {a: np.empty(draws) for a in arms}

    for k in range(draws):
        selected = rng.choice(len(tids), len(tids), replace=True)
        denominator = sizes[selected].sum()
        for a in arms:
            boot[a][k] = grouped[a][selected].sum(axis=0).mean() / denominator
    stats = {}
    for a in arms:
        require(
            int(grouped[a].sum()) == int(counts[a].sum()), "Cluster count disagreement"
        )
        rates = counts[a] / len(ids)
        stats[a] = {
            "fp_counts": counts[a].tolist(),
            "fp_rates": rates.tolist(),
            "mean_fp_rate": float(rates.mean()),
            "spread": float(np.ptp(rates)),
        }
    contrasts = []
    for a, b in pairs:
        delta = stats[a]["mean_fp_rate"] - stats[b]["mean_fp_rate"]
        floor = max(stats[a]["spread"], stats[b]["spread"])
        ci = np.percentile(boot[a] - boot[b], [2.5, 97.5]).tolist()
        per_rep = (np.array(stats[a]["fp_rates"]) - stats[b]["fp_rates"]).tolist()
        diff = []
        for r in REPS:
            better = sum(
                not predictions[a, r][i]["result"]["hate"]
                and predictions[b, r][i]["result"]["hate"]
                for i in ids
            )
            worse = sum(
                predictions[a, r][i]["result"]["hate"]
                and not predictions[b, r][i]["result"]["hate"]
                for i in ids
            )
            diff.append(
                {
                    "replicate": r,
                    "a_better": better,
                    "a_worse": worse,
                    "tied": len(ids) - better - worse,
                }
            )
            require(
                counts[a][r - 1] - counts[b][r - 1] == worse - better,
                "Discordant counts do not reconcile",
            )
        discordant[f"{a}-{b}"] = diff
        contrasts.append(
            {
                "a": a,
                "b": b,
                "delta": delta,
                "per_replicate_delta": per_rep,
                "floor": floor,
                "magnitude_over_floor": abs(delta) / floor if floor else None,
                "ci_95": ci,
                "ci_excludes_zero": ci[0] > 0 or ci[1] < 0,
                "clears_floor": bool(abs(delta) > floor and floor > 0),
                "complete_separation": max(stats[a]["fp_rates"])
                < min(stats[b]["fp_rates"])
                or max(stats[b]["fp_rates"]) < min(stats[a]["fp_rates"]),
            }
        )
    return {
        "name": name,
        "n_items": len(ids),
        "n_templates": len(tids),
        "item_ids": ids,
        "arms": stats,
        "contrasts": contrasts,
        "discordant": discordant,
    }


def print_pool(pool):
    print(f"\n{pool['name']}: n={pool['n_items']}, templates={pool['n_templates']}")
    for a, s in pool["arms"].items():
        print(
            f"  {a:4} FP {str(s['fp_counts']):>16} mean {s['mean_fp_rate']:.6f} range {s['spread']:.6f}"
        )
    for c in pool["contrasts"]:
        ratio = (
            "undefined"
            if c["magnitude_over_floor"] is None
            else f"{c['magnitude_over_floor']:.2f}x"
        )
        print(
            f"  {c['a']}-{c['b']}: delta {c['delta']:+.6f}; floor {c['floor']:.6f}; {ratio}; "
            f"CI [{c['ci_95'][0]:+.6f}, {c['ci_95'][1]:+.6f}]; "
            f"clears floor={c['clears_floor']}; CI excludes zero={c['ci_excludes_zero']}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite {args.output}")
    require(args.draws >= 1000, "Use at least 1000 bootstrap draws")
    df = pd.read_parquet(DATA)
    require(not df.id.astype(str).duplicated().any(), "Duplicate dataset IDs")
    meta = {str(r.id): json.loads(r.raw) for r in df.itertuples(index=False)}
    block = {i for i, m in meta.items() if m["block"] == "guideline_covered"}
    require(len(block) == 693, "Unexpected guideline-covered block size")
    require(
        all(not bool(r.gate) for r in df.itertuples(index=False) if str(r.id) in block),
        "The covered block contains a gold-hateful item",
    )
    kb_path = Path("kb/records.jsonl")
    kb_hash = digest(kb_path)[:16]
    require(kb_hash == "b3e1a021777f7b85", "Current KB differs from audited snapshot")
    kb_ids = {
        json.loads(line)["id"]
        for line in kb_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    predictions, result_files = load_predictions(block)
    inventory = check_provenance(predictions, block, meta, kb_hash, load_mapping())
    oracle = common_ids(predictions, block, ["A", "B", "N"])
    matched = common_ids(predictions, block, ["RAG", "N"])
    all_arms = common_ids(predictions, block, list(SOURCES))
    context_ids = common_ids(predictions, block, ["A", "B", "N", "RAG"])
    context = check_context(
        predictions, context_ids, ["A", "B", "N", "RAG"], "RAG", kb_ids
    )
    print("Stored-prediction audit. No API calls, retrieval, or training.")
    print("Source inventory:")
    for row in inventory:
        print(
            f"  {row['arm']} r{row['replicate']}: {row['source_rows_in_block']} rows; "
            f"unscored={row['unscored_ids']}; code={row['code_versions']}"
        )
    print("Context checks:", json.dumps(context))
    pairs = [("A", "N"), ("B", "N"), ("A", "B")]
    pools = [
        analyse_pool(
            "oracle_only",
            oracle,
            ["A", "B", "N"],
            pairs,
            predictions,
            meta,
            args.draws,
            args.seed,
        ),
        analyse_pool(
            "matched_natural_guidelines_primary",
            matched,
            ["RAG", "N"],
            [("RAG", "N")],
            predictions,
            meta,
            args.draws,
            args.seed,
        ),
        analyse_pool(
            "all_arms_common_sensitivity",
            all_arms,
            list(SOURCES),
            pairs + [("RAG", "N"), ("A", "RAG"), ("G", "N"), ("G", "ZS"), ("A", "G")],
            predictions,
            meta,
            args.draws,
            args.seed,
        ),
    ]
    by_function = []
    for functionality in sorted({meta[i]["functionality"] for i in oracle}):
        ids = [i for i in oracle if meta[i]["functionality"] == functionality]
        by_function.append(
            analyse_pool(
                functionality,
                ids,
                ["A", "B", "N"],
                pairs,
                predictions,
                meta,
                args.draws,
                args.seed,
            )
        )
    for pool in pools + by_function:
        print_pool(pool)
    print(
        "\nMatched primary discordant counts:",
        json.dumps(pools[1]["discordant"]["RAG-N"]),
    )
    print(
        "\nInterpretation: G-N changes definitions and examples and is not a guideline-effect estimate."
    )
    print(
        "Limitation: oracle code_version is absent and its runner differs from ordinary classification."
    )
    print(
        "Bootstrap intervals condition on the three stored runs; a CI containing zero does not prove equivalence."
    )
    input_paths = [
        DATA,
        kb_path,
        RUNNER,
        Path("prompts/classify_v1.txt"),
        Path("config/taxonomy.yaml"),
        *result_files,
    ]
    evidence = {
        "audit": "Astra guideline oracle P1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script": {
            "path": Path(__file__).as_posix(),
            "sha256": digest(Path(__file__)),
        },
        "method": {
            "draws": args.draws,
            "seed": args.seed,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "replicates": list(REPS),
            "statistic": "mean false-positive rate across three fixed replicates",
            "resampling": "whole templ_id clusters with replacement, paired across arms",
            "plan": "one shared draw plan per pool; seed reset for each pool",
            "interval": "percentile 2.5 and 97.5; exploratory, unadjusted for multiple comparisons",
            "floor": "larger within-arm replicate range, unrounded",
        },
        "verified_stamps": {**EXPECTED, "kb_version": kb_hash},
        "inputs": [
            {"path": p.as_posix(), "sha256": digest(p), "bytes": p.stat().st_size}
            for p in input_paths
        ],
        "source_inventory": inventory,
        "context_verification": context,
        "excluded_from_all_arms": sorted(block - set(all_arms)),
        "excluded_from_matched_primary": sorted(block - set(matched)),
        "pools": pools,
        "per_functionality": by_function,
        "limitations": [
            "Oracle and ordinary runs use different execution paths; oracle code stamp absent",
            "Matching IDs and KB hash do not establish identical historical full prompt bytes",
            "Not simultaneous/randomised runs; provider-side temporal effects are not isolated",
            "Three saved replicates are fixed for item/template bootstrap inference",
            "Covered non-hateful English functionality stratum only",
            "No equivalence margin or equivalence test",
            "G-N is confounded",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(evidence, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"\nWROTE NEW EVIDENCE: {args.output}")


if __name__ == "__main__":
    main()

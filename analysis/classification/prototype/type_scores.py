import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd

from src.hsrag.metrics import MIN_SUPPORT, clean, score_multilabel

ROOT = Path("experiments/results")
SPEC = {
    "en_dev_eval_sq3_types": {
        "zero_shot": [f"sq3_types_sq2_r{i}" for i in (1, 2, 3)],
        "few_shot": [f"sq3_types_sq2_r{i}" for i in (1, 2, 3)],
        "rag": [f"sq3_round0_r{i}" for i in (1, 2, 3, 4)],
    },
    "en_test_eval_types": {
        a: [f"types_test_r{i}" for i in (1, 2, 3)]
        for a in ("zero_shot", "few_shot", "rag")
    },
}
METADATA = (
    "active_model",
    "prompt_version",
    "taxonomy_version",
    "kb_version",
    "code_version",
    "temperature",
    "n_runs",
    "workers",
)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(stem):
    by_arm = {}
    path = ROOT / f"{stem}_live.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            r = json.loads(line)
            if "_manifest" in r:
                continue
            out = by_arm.setdefault(r["arm"], {})
            require(r["item_id"] not in out, f"Duplicate ID in {stem}, arm {r['arm']}")
            fields = (
                "item_id",
                "result",
                "text_hash",
                "n_valid",
                "retrieval_config",
                *METADATA,
            )
            out[r["item_id"]] = {k: r.get(k) for k in fields}
    return by_arm


def independent_score(rows, gold, labels, expected):
    eligible = []
    for r in rows:
        g = gold.get(r["item_id"])
        if g is None or not bool(g.gate):
            continue
        truth = clean(getattr(g, "hate_types", None))
        if truth is not None and r["result"] is not None:
            eligible.append((set(truth), set(r["result"]["hate_type"])))
    values = []
    for label in labels:
        tp = sum(label in t and label in p for t, p in eligible)
        fp = sum(label not in t and label in p for t, p in eligible)
        fn = sum(label in t and label not in p for t, p in eligible)
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 0.0
        values.append(f1)
        for k, value in (("tp", tp), ("fp", fp), ("fn", fn), ("support", tp + fn)):
            require(
                expected.per_label[label][k] == value,
                f"Independent {label}/{k} mismatch",
            )
    score = sum(values) / len(values)
    require(abs(score - expected.macro_f1) < 1e-12, "Independent macro-F1 mismatch")
    require(len(eligible) == expected.n_items, "Scored denominator mismatch")
    return score


def mean(values):
    return sum(values) / len(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), f"Refusing to overwrite {args.output}")
    paths = {
        Path("src/hsrag/metrics.py"),
        Path("prompts/classify_v1.txt"),
        Path("config/taxonomy.yaml"),
    }
    cache = {}
    results = []
    for subset, sources in SPEC.items():
        path = Path("data/processed") / f"{subset}.parquet"
        paths.add(path)
        df = pd.read_parquet(path)
        require(
            not df.id.astype(str).duplicated().any(), f"Duplicate dataset IDs: {subset}"
        )
        gold = {str(r.id): r for r in df.itertuples(index=False)}
        eligible_gold = {
            i
            for i, g in gold.items()
            if bool(g.gate) and clean(getattr(g, "hate_types", None)) is not None
        }
        gold_support = Counter(
            label for i in eligible_gold for label in clean(gold[i].hate_types)
        )
        expected_labels = sorted(l for l, n in gold_support.items() if n >= MIN_SUPPORT)
        require(len(expected_labels) == 7, f"Unexpected supported label set: {subset}")
        by_arm, inventory = {}, []
        common = set(eligible_gold)
        text_hashes = {i: set() for i in gold}
        for arm, stems in sources.items():
            by_arm[arm] = []
            for rep, stem in enumerate(stems, 1):
                paths.add(ROOT / f"{stem}_live.jsonl")
                if stem not in cache:
                    cache[stem] = load(stem)
                rows_by_id = cache[stem][arm]
                require(
                    set(rows_by_id) == set(gold), f"Source ID mismatch: {stem}/{arm}"
                )
                rows = [rows_by_id[i] for i in sorted(rows_by_id)]
                score = score_multilabel(rows, gold, "hate_types", "hate_type")
                require(
                    score.labels_averaged == expected_labels,
                    f"Averaged label set mismatch: {stem}/{arm}",
                )
                independent_score(rows, gold, expected_labels, score)
                by_arm[arm].append(asdict(score))
                valid = {i for i, r in rows_by_id.items() if r["result"] is not None}
                common &= valid
                for i, r in rows_by_id.items():
                    require(
                        r["text_hash"] is not None,
                        f"Missing input hash: {stem}/{arm}/{i}",
                    )
                    text_hashes[i].add(r["text_hash"])
                inventory.append(
                    {
                        "arm": arm,
                        "replicate": rep,
                        "stem": stem,
                        "n_rows": len(rows),
                        "n_scored": score.n_items,
                        "unscored_ids": sorted(set(rows_by_id) - valid),
                        "metadata": {
                            k: sorted({json.dumps(r[k], sort_keys=True) for r in rows})
                            for k in METADATA
                        },
                        "retrieval_configs": sorted(
                            {
                                json.dumps(r["retrieval_config"], sort_keys=True)
                                for r in rows
                            }
                        ),
                        "valid_vote_counts": dict(
                            sorted(Counter(r["n_valid"] for r in rows).items())
                        ),
                    }
                )
        require(
            all(len(v) == 1 for v in text_hashes.values()),
            f"Input hashes vary across arms: {subset}",
        )
        require(
            common == eligible_gold,
            f"Unequal scoring pools in {subset}; reconcile before comparing",
        )
        summaries = {}
        for arm, scores in by_arm.items():
            vals = [s["macro_f1"] for s in scores]
            summaries[arm] = {
                "replicates": len(vals),
                "macro_f1": vals,
                "mean": mean(vals),
                "spread": max(vals) - min(vals),
            }
        n = min(summaries["rag"]["replicates"], summaries["few_shot"]["replicates"])
        deltas = [
            summaries["rag"]["macro_f1"][k] - summaries["few_shot"]["macro_f1"][k]
            for k in range(n)
        ]
        floor = max(summaries["rag"]["spread"], summaries["few_shot"]["spread"])
        paired = mean(deltas)
        arm_mean = summaries["rag"]["mean"] - summaries["few_shot"]["mean"]
        result = {
            "subset": subset,
            "dataset_rows": len(gold),
            "gold_hateful": sum(bool(r.gate) for r in gold.values()),
            "gold_hate_type_annotated": sum(
                clean(getattr(r, "hate_types", None)) is not None for r in gold.values()
            ),
            "common_scored_items": len(common),
            "gold_label_support": dict(sorted(gold_support.items())),
            "gold_mean_cardinality": sum(gold_support.values()) / len(eligible_gold),
            "sources_in_gold": dict(
                Counter(str(getattr(r, "source", "<missing>")) for r in gold.values())
            ),
            "labels_averaged": expected_labels,
            "min_support": MIN_SUPPORT,
            "input_hashes_match_across_arms": True,
            "independent_f1_check": "PASS",
            "inventory": inventory,
            "arm_summary": summaries,
            "dimension_scores": by_arm,
            "rag_minus_few_shot": {
                "paired_replicates": n,
                "per_replicate_delta": deltas,
                "paired_mean_delta": paired,
                "all_arm_means_delta": arm_mean,
                "floor_all_available_replicates": floor,
                "paired_mean_magnitude_over_floor": abs(paired) / floor,
                "bootstrap_status": "NOT RUN IN THIS CHECKPOINT",
            },
        }
        results.append(result)
        print(
            f"\n{subset}: rows={len(gold)}, gold hateful={result['gold_hateful']}, common scored={len(common)}"
        )
        print(
            "  Label support:",
            result["gold_label_support"],
            "gold cardinality:",
            result["gold_mean_cardinality"],
        )
        for arm, s in summaries.items():
            print(
                f"  {arm}: R={s['replicates']}; scores={[round(v,6) for v in s['macro_f1']]}; "
                f"mean={s['mean']:.6f}; spread={s['spread']:.6f}"
            )
        print(f"  RAG-FS paired deltas: {[round(v,6) for v in deltas]}")
        print(
            f"  RAG-FS paired mean={paired:+.6f}; all-arm means={arm_mean:+.6f}; "
            f"floor={floor:.6f}; paired mean / floor={abs(paired)/floor:.2f}"
        )
        print("  Independent TP/FP/FN/F1 arithmetic and matching input hashes: PASS")
        print("  Recorded provenance by source run:")
        for r in inventory:
            m = r["metadata"]
            print(
                f"    {r['arm']} r{r['replicate']} {r['stem']}: code={m['code_version']}; "
                f"taxonomy={m['taxonomy_version']}; prompt={m['prompt_version']}; "
                f"KB={m['kb_version']}; model={m['active_model']}; T={m['temperature']}; n_runs={m['n_runs']}"
            )
    print(
        "\nCHECKPOINT STOP: no bootstrap, pooling, secondary-dev, or other-dimension analysis performed."
    )
    evidence = {
        "audit": "Astra P2 checkpoint 1: hate_type scores and provenance",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_script": {
            "path": Path(__file__).as_posix(),
            "sha256": sha(Path(__file__)),
        },
        "inputs": [
            {"path": p.as_posix(), "sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(paths)
        ],
        "results": results,
        "status": "PARTIAL: bootstrap and statistical interpretation reconciliation pending",
        "limitations": [
            "Main dev and test hate_type only; not the 103-item secondary dev set",
            "Recorded stamps are inventoried, not proof of identical provider state",
            "Taxonomy hash normalisation and full configuration comparison pending",
            "Original confidence intervals are not recomputed here",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(evidence, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write("\n")
    print(f"WROTE NEW EVIDENCE: {args.output}")


if __name__ == "__main__":
    main()

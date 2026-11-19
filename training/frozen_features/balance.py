import argparse
from collections import Counter, defaultdict
from pathlib import Path
from unittest.mock import patch
import numpy as np
from training.frozen_features import pools as pools
from training.frozen_features import compact as old
from analysis.common import study as c
from analysis.common.api import save_new

BASE = c.ROOT / "experiments/astra/compact_balance_20261008_v1"
SEEDS = (111, 112, 113, 114, 115)


def prepare():
    pools.validate()
    template = c.read(old.BASE / "train.json")
    original = template["heads"]["hate_type"]
    records = c.rows(c.CORE / "kb/full/records.jsonl")
    labels = original["labels"]
    groups = defaultdict(list)
    for i, r in enumerate(records):
        meta = r.get("meta", {})
        if (
            r["kind"] != "example"
            or r["lang"] != "en"
            or meta.get("gate") is not True
            or (meta.get("hate_types") is None)
        ):
            continue
        target = [int(lab in meta["hate_types"]) for lab in labels]
        row = dict(
            id=r["id"],
            vector_index=i,
            cluster=c.signature(old.norm(r["text"])),
            target=target,
            source=r["source"],
        )
        groups[r["source"], tuple(target)].append(row)
    demand = Counter(((r["source"], tuple(r["target"])) for r in original["rows"]))
    c.require(sum(demand.values()) == 67, "Wrong curated annotation count")
    c.require(
        all((len(groups[k]) >= n for k, n in demand.items())),
        "Exact block matching infeasible; do not substitute",
    )
    fixed = {r["cluster"]: i for i, r in enumerate(template["heads"]["hate"]["rows"])}
    jobs = []
    prepared = {}
    inventories = {}
    for draw in SEEDS:
        rows = []
        for key, n in sorted(demand.items()):
            rows += sorted(
                groups[key],
                key=lambda r: pools.rank("astra-type-balanced67-v1", draw, r["id"]),
            )[:n]
        rows = sorted(rows, key=lambda r: r["id"])
        c.require(
            Counter(((r["source"], tuple(r["target"])) for r in rows)) == demand,
            "Annotation-block or source balance changed",
        )
        c.require(len({r["id"] for r in rows}) == 67, "Repeated training record")
        folds = {
            str(seed): [
                (
                    template["folds"][str(seed)][fixed[r["cluster"]]]
                    if r["cluster"] in fixed
                    else int(
                        pools.rank("astra-type-extra-fold-v1", seed, r["cluster"])[0],
                        16,
                    )
                    % 5
                )
                for r in rows
            ]
            for seed in old.SEEDS
        }
        name = f"balanced_type67_s{draw}"
        path = BASE / f"train/{name}.json"
        save_new(path, dict(labels=labels, rows=rows, folds=folds))
        prepared[c.rel(path)] = c.sha(path)
        inventories[name] = dict(
            type_annotations=67,
            gate_annotations=147,
            training_id_union=len(set(template["gate_ids"]) | {r["id"] for r in rows}),
            shared_curated_type_ids=len(
                {r["id"] for r in original["rows"]} & {r["id"] for r in rows}
            ),
            label_support={
                lab: sum((r["target"][j] for r in rows)) for j, lab in enumerate(labels)
            },
        )
        for seed in old.SEEDS:
            jobs.append(
                dict(
                    id=f"{name}_fold{seed}",
                    kind="balanced_type67",
                    pool=name,
                    seed=seed,
                    train=c.rel(path),
                )
            )
    original_plan = c.read(pools.BASE / "plan.json")
    hashes = original_plan["inputs"] | {
        c.rel(Path(__file__)): c.sha(Path(__file__)),
        c.rel(pools.BASE / "results.json"): c.sha(pools.BASE / "results.json"),
    }
    save_new(
        BASE / "plan.json",
        dict(
            created_utc=c.utc(),
            status="frozen_before_balanced_followup_fits",
            inputs=hashes,
            prepared=prepared,
            jobs=jobs,
            training_counts=inventories,
            endpoints=original_plan["endpoints"],
            environment=old.environment(),
            design="Five TRAIN-only draws matching the curated 67 type labels exactly by complete co-label block and source. Same original gate and head-training recipe as prior equal67 control. Every draw/fold retained. No new training supervision for gate.",
            reason="Prior equal-count result left a source/label-composition explanation: curated type supports 9/10 versus uneven uniform67 draws. This post-outcome extension is explanatory, not confirmatory.",
            additional_baseline="For each original BGE gate, predict all seven type labels whenever hate=true. No learned type head or tuning. Explain the positive F1 floor of an indiscriminate type predictor; not a uniform-random chance bound.",
            limitations="Matching labels and source does not isolate all text properties. The single curated pool is not replicated curation. Gate OOF handling follows the original equal67 policy, with different possible overlap with gate-training texts. No p-value, selected new model, outcome-based redrawing or unexposed TEST claim.",
        ),
    )
    print("Exact source/co-label matching feasible: five draws, 15 fits.", flush=True)


def run():
    from analysis.common import replay as s

    plan = c.read(BASE / "plan.json")
    c.verify(plan["inputs"])
    c.verify(plan["prepared"])
    c.require(
        plan["environment"] == old.environment(), "Changed scientific environment"
    )
    with patch.object(pools, "BASE", BASE):
        for job in plan["jobs"]:
            pools.work(job, plan)
    results = {}
    receipts = {}
    for job in plan["jobs"]:
        done = BASE / "jobs" / job["id"] / "complete.json"
        receipt = c.read(done)
        c.verify(receipt["outputs"])
        receipts[c.rel(done)] = c.sha(done)
        for name, value in c.read(BASE / "jobs" / job["id"] / "scores.json")[
            "endpoints"
        ].items():
            results.setdefault(job["pool"] + "/" + name, []).append(
                dict(
                    seed=job["seed"],
                    macro_f1=value["result"]["macro_f1_valid_only"],
                    ungated_macro_f1=value["ungated_diagnostic"]["macro_f1_valid_only"],
                )
            )
    baseline = {}
    labels = c.read(old.BASE / "train.json")["heads"]["hate_type"]["labels"]
    for name, ep in plan["endpoints"].items():
        if ep["dimension"] != "hate_type":
            continue
        gold = s.gold_rows(name)
        ids = [str(r["id"]) for r in gold]
        runs = []
        for seed in old.SEEDS:
            pred = {
                r["item_id"]: r["result"]
                for r in c.read(old.BASE / f"predictions_seed{seed}.json")
            }
            outputs = {
                key: pred[key]
                | {"hate_type": list(labels) if pred[key]["hate"] else []}
                for key in ids
            }
            runs.append(
                dict(seed=seed, **s.summarize(s.contributions(gold, outputs, ep)))
            )
        baseline[name] = dict(
            runs=runs, mean=float(np.mean([r["macro_f1_valid_only"] for r in runs]))
        )
    save_new(
        BASE / "results.json",
        dict(
            created_utc=c.utc(),
            status="all_15_balanced_type_fits_complete",
            plan_sha256=c.sha(BASE / "plan.json"),
            completion_receipts=receipts,
            groups={
                key: dict(runs=vals, mean=float(np.mean([v["macro_f1"] for v in vals])))
                for key, vals in results.items()
            },
            all_labels_with_original_gate=baseline,
            Qwen_calls=0,
            CUDA_used=False,
        ),
    )
    print("Balanced annotation controls and all-label baseline complete.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "run"])
    args = p.parse_args()
    with c.offline():
        globals()[args.mode]()

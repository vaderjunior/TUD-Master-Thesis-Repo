import argparse
from pathlib import Path
import time
import numpy as np
from training.frozen_features import fit as linear
from training.frozen_features import compact as old
from analysis.common import study as c
from analysis.common.api import save_new

BASE = c.ROOT / "experiments/astra/compact_controls_20261008_v1"
ARMS = {
    "fixedC_fixedT": (False, False),
    "fixedC_tunedT": (False, True),
    "tunedC_fixedT": (True, False),
    "tunedC_tunedT": (True, True),
}


def prepare():
    original = old.validate()
    files = [
        Path(__file__),
        Path(linear.__file__),
        Path(old.__file__),
        c.ROOT / "tests/training/frozen_features/test_fit.py",
        c.ROOT / "analysis/common/replay.py",
        c.ROOT / "analysis/common/scoring.py",
        old.BASE / "plan.json",
        old.BASE / "train.json",
        old.BASE / "train_vectors.npy",
        old.BASE / "evaluation_inputs.json",
        old.BASE / "evaluation_vectors.npy",
        old.BASE / "results.json",
    ]
    files += [old.BASE / f"fit_seed{s}.json" for s in old.SEEDS]
    files += [old.BASE / f"predictions_seed{s}.json" for s in old.SEEDS]
    files += [c.ROOT / f"data/processed/{ep}.parquet" for ep in original["endpoints"]]
    save_new(
        BASE / "plan.json",
        {
            "created_utc": c.utc(),
            "status": "frozen_before_control_fits",
            "inputs": {c.rel(p): c.sha(p) for p in files},
            "environment": old.environment(),
            "seeds": old.SEEDS,
            "arms": ARMS,
            "C_grid": old.CS,
            "threshold_grid": old.THRESHOLDS,
            "additional_arm": "tunedC_tunedT_original_fit",
            "endpoints": original["endpoints"],
            "policy": "Same features, group folds, annotation budget, class weights and learner. Cross fixed C=1/selected C with fixed .5/selected cutoff. Gate selected before gated fine heads within each recipe. Severity uses argmax.",
            "refit_policy": "Lock the original fully selected C/cutoffs and all tuning information; change only final weight-fitting rows from the full head annotation budget to the original internal fit partition. This does not isolate extra annotation access.",
            "comparison": "Descriptive conditional factorial on exposed DEV, all arms/seeds/labels reported. No selected best recipe, new deployment rule, new superiority tests or untouched confirmation.",
            "sparse_policy": "Retain missing annotation masks. Skip empty held folds; empty fitting folds predict zero scores, one-class fits are constants. If no label has two positive and two negative TRAIN observations, use C=1/default cutoff and record fallback. An entirely untrained fine head produces empty/None. No actual compact source task needs these fallbacks.",
            "scope": "New isolated CPU fitting only. No Qwen, CUDA, embedding generation or changes to original or running studies.",
        },
    )
    print("Control protocol frozen.", flush=True)


def validate():
    p = c.read(BASE / "plan.json")
    c.verify(p["inputs"])
    c.require(p["environment"] == old.environment(), "Environment changed")
    return p


def fit():
    plan = validate()
    data = c.read(old.BASE / "train.json")
    x = np.load(old.BASE / "train_vectors.npy", allow_pickle=False)
    receipts = {}
    candidate_checks = 0
    original_checks = 0
    start = time.perf_counter()
    for seed in old.SEEDS:
        cached, counts = linear.make_oof(data, x, seed)
        for arm, (tune_c, tune_t) in ARMS.items():
            heads = linear.fit_recipe(
                data, x, seed, cached, tune_c=tune_c, tune_threshold=tune_t
            )
            candidate_checks += linear.check_selection(data, seed, heads, cached)
            if arm == "tunedC_tunedT":
                previous = c.read(old.BASE / f"fit_seed{seed}.json")["heads"]
                for h in old.HEADS:
                    c.require(
                        heads[h]["selected"] == previous[h]["selected"],
                        "Original tuning selection differs",
                    )
                    c.require(
                        heads[h]["models"] == previous[h]["models"],
                        "Original coefficients differ",
                    )
                    c.require(
                        heads[h]["oof_probabilities"]
                        == previous[h]["oof_probabilities"],
                        "Original OOF probabilities differ",
                    )
                    original_checks += 1
            path = BASE / f"fits/{arm}_seed{seed}.json"
            save_new(
                path,
                {
                    "plan_sha256": c.sha(BASE / "plan.json"),
                    "seed": seed,
                    "arm": arm,
                    "heads": heads,
                    "oof_task_counts": counts,
                },
            )
            receipts[c.rel(path)] = c.sha(path)
            print("Fit and checked", arm, seed, flush=True)
        arm = "tunedC_tunedT_original_fit"
        locked = c.read(old.BASE / f"fit_seed{seed}.json")["heads"]
        heads = linear.fit_recipe(
            data, x, seed, cached, original_fit=True, locked_selection=locked
        )
        candidate_checks += linear.check_selection(data, seed, heads, cached)
        path = BASE / f"fits/{arm}_seed{seed}.json"
        save_new(
            path,
            {
                "plan_sha256": c.sha(BASE / "plan.json"),
                "seed": seed,
                "arm": arm,
                "heads": heads,
                "oof_task_counts": counts,
            },
        )
        receipts[c.rel(path)] = c.sha(path)
    save_new(
        BASE / "fit_complete.json",
        {
            "status": "all_control_fits_complete",
            "plan_sha256": c.sha(BASE / "plan.json"),
            "outputs": receipts,
            "candidate_score_checks": candidate_checks,
            "original_head_fits_exactly_reproduced": original_checks,
            "seconds_including_io_checks": time.perf_counter() - start,
        },
    )


def predict():
    validate()
    receipt = c.read(BASE / "fit_complete.json")
    c.verify(receipt["outputs"])
    inputs = c.read(old.BASE / "evaluation_inputs.json")
    x = np.load(old.BASE / "evaluation_vectors.npy", allow_pickle=False)
    files = {}
    reproduced = 0
    for name in receipt["outputs"]:
        fit = c.read(c.ROOT / name)
        probs = linear.probabilities(fit["heads"], x)
        delivered = linear.decode(fit["heads"], probs)
        ungated = linear.decode(fit["heads"], probs, ungated=True)
        rows = [
            row | {"result": r, "ungated_diagnostic": u}
            for row, r, u in zip(inputs, delivered, ungated)
        ]
        if fit["arm"] == "tunedC_tunedT":
            previous = c.read(old.BASE / f"predictions_seed{fit['seed']}.json")
            c.require(
                all((a["result"] == b["result"] for a, b in zip(rows, previous))),
                "Old delivered predictions differ",
            )
            reproduced += len(rows)
        path = BASE / f"predictions/{fit['arm']}_seed{fit['seed']}.json"
        save_new(
            path, {"fit_path": name, "fit_sha256": c.sha(c.ROOT / name), "rows": rows}
        )
        files[c.rel(path)] = c.sha(path)
    save_new(
        BASE / "predictions_complete.json",
        {"outputs": files, "original_predictions_reproduced": reproduced},
    )


def score():
    from analysis.common import replay as s

    plan = validate()
    receipt = c.read(BASE / "predictions_complete.json")
    c.verify(receipt["outputs"])
    predictions = {
        Path(name).stem: {r["item_id"]: r for r in c.read(c.ROOT / name)["rows"]}
        for name in receipt["outputs"]
    }
    result = {}
    count = 0
    for name, ep in plan["endpoints"].items():
        gold = s.gold_rows(name)
        ids = [str(r["id"]) for r in gold]
        arms = {}
        for arm in [*ARMS, "tunedC_tunedT_original_fit"]:
            runs = []
            for seed in old.SEEDS:
                pred = predictions[f"{arm}_seed{seed}"]
                full = s.summarize(
                    s.contributions(gold, {k: pred[k]["result"] for k in ids}, ep)
                )
                ungated = s.summarize(
                    s.contributions(
                        gold, {k: pred[k]["ungated_diagnostic"] for k in ids}, ep
                    )
                )
                runs.append(
                    {"seed": seed, "delivered": full, "ungated_diagnostic": ungated}
                )
                count += 2
            arms[arm] = {
                "runs": runs,
                "mean": float(
                    np.mean([r["delivered"]["macro_f1_valid_only"] for r in runs])
                ),
                "ungated_mean": float(
                    np.mean(
                        [r["ungated_diagnostic"]["macro_f1_valid_only"] for r in runs]
                    )
                ),
            }
        result[name] = {"endpoint": ep, "arms": arms}
        print(name, {a: round(v["mean"], 6) for a, v in arms.items()}, flush=True)
    c.verify(plan["inputs"])
    save_new(
        BASE / "results.json",
        {
            "created_utc": c.utc(),
            "status": "completed_exploratory_recipe_controls",
            "plan_sha256": c.sha(BASE / "plan.json"),
            "endpoints": result,
            "scored_arm_seed_policy_cells": count,
            "Qwen_calls": 0,
            "CUDA_used": False,
            "new_embeddings": 0,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "fit", "predict", "score"])
    args = p.parse_args()
    with c.offline():
        globals()[args.mode]()

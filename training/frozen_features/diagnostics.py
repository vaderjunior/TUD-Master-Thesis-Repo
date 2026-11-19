import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
from pathlib import Path
import numpy as np
from analysis.common import study as c
from analysis.common.api import save_new

BASE = c.ROOT / "experiments/astra/compact_followup_20261008_v1"
OLD = c.ROOT / "experiments/astra/compact_bge_20261008_v1"
DIAG = c.ROOT / "experiments/astra/offline_closure_20261006_v1/encoders/results.json"
MODES = ("selected", "half_gate", "half_fine", "half_both", "ungated_selected")
SEEDS = (42, 43, 44)


def save(path, obj):
    c.require(
        Path(path).resolve().is_relative_to(BASE.resolve()), "Outside follow-up folder"
    )
    save_new(path, obj)


def prepare():
    old = c.read(OLD / "plan.json")
    paths = [
        Path(__file__),
        Path(c.__file__),
        c.ROOT / "analysis/common/api.py",
        c.ROOT / "analysis/common/replay.py",
        c.ROOT / "analysis/common/scoring.py",
        c.ROOT / "tests/training/frozen_features/test_diagnostics.py",
        OLD / "plan.json",
        OLD / "results.json",
        OLD / "verification.json",
        OLD / "predictions_complete.json",
        DIAG,
    ]
    c.verify(c.read(OLD / "predictions_complete.json")["outputs"])
    c.verify(c.read(OLD / "fit_complete.json")["outputs"])
    paths += [
        OLD / f"{prefix}_seed{seed}.json"
        for seed in SEEDS
        for prefix in ("fit", "predictions")
    ]
    paths += [c.ROOT / f"data/processed/{name}.parquet" for name in old["endpoints"]]
    for name in old["endpoints"]:
        key = f"data/processed/{name}.parquet"
        c.require(c.sha(c.ROOT / key) == old["inputs"][key], "Original gold changed")
    save(
        BASE / "plan.json",
        {
            "created_utc": c.utc(),
            "status": "frozen_before_new_decision_policy_summaries",
            "inputs": {c.rel(p): c.sha(p) for p in paths},
            "seeds": SEEDS,
            "endpoints": old["endpoints"],
            "modes": MODES,
            "design": "2x2 decision-only intervention: chosen/0.5 gate cutoff by chosen/0.5 fine-label cutoff; severity always argmax. Also ungated selected heads as an oracle diagnostic on gold-hateful fine rows only.",
            "selection": "No fit, threshold search, C selection or chosen best outcome. Reuse all three fixed BGE fits.",
            "checks": "Reproduce every saved selected decision and all 15 BGE endpoint scores; independently reconstruct indicator matrices and label metrics. Read old cutoff/ranking diagnostics as previously exposed context.",
            "statistics": "Descriptive mechanism diagnosis, all five original endpoints and every fit/label retained. No new p-values, superiority claim, best-cell choice or conversion to deployment.",
            "limits": "C and fitted weights were selected under the original TRAIN-only tuned recipe. Setting a cutoff back to 0.5 is a conditional post-processing intervention, not a separately trained default baseline. Ungating is not deployable; score only already gold-hateful fine rows. No decomposition of the old-to-new gain into additive causal shares.",
            "follow_up": "Broader frozen-model extension and TRAIN-only training-recipe/pool experiments are separate pending studies. No change to running retrieval/SQ3 plans.",
        },
    )
    print("Saved-decision diagnostic protocol frozen.", flush=True)


def decode(prob, heads, mode):
    c.require(mode in MODES, "Unknown mode")
    half_gate = mode in ("half_gate", "half_both")
    half_fine = mode in ("half_fine", "half_both")
    gate = float(prob["hate"][0]) > (
        0.5 if half_gate else heads["hate"]["selected"]["threshold"]
    )
    result = dict(
        hate=bool(gate), hate_type=[], target_group=[], severity=None, legal=[]
    )
    if gate or mode == "ungated_selected":
        for head in ("hate_type", "target_group"):
            threshold = 0.5 if half_fine else heads[head]["selected"]["threshold"]
            result[head] = [
                lab
                for lab, p in zip(heads[head]["labels"], prob[head])
                if p >= threshold
            ]
        result["severity"] = heads["severity"]["labels"][
            int(np.argmax(prob["severity"]))
        ]
    return result


def direct_bits(records, ep, predictions, heads, mode):

    dimension = ep["dimension"]
    all_labels = ep["all_labels"]
    wanted = set(ep["scorable_ids"])
    result = []
    for row in records:
        rid = str(row["id"])
        if rid not in wanted:
            continue
        p = predictions[rid]["ungated_probabilities"]
        gate_cut = (
            0.5
            if mode in ("half_gate", "half_both")
            else heads["hate"]["selected"]["threshold"]
        )
        gate = float(p["hate"][0]) > gate_cut
        if dimension == "hate":
            selected = {"hate" if gate else "not_hate"}
        elif not gate and mode != "ungated_selected":
            selected = set()
        elif dimension == "severity":
            labels = heads[dimension]["labels"]
            scores = p[dimension]
            selected = {labels[max(range(len(scores)), key=lambda k: scores[k])]}
        else:
            cut = (
                0.5
                if mode in ("half_fine", "half_both")
                else heads[dimension]["selected"]["threshold"]
            )
            selected = {
                lab
                for k, lab in enumerate(heads[dimension]["labels"])
                if p[dimension][k] >= cut
            }
        result.append([int(lab in selected) for lab in all_labels])
    return np.asarray(result, dtype=np.int64)


def run():
    from analysis.common import replay as s
    from sklearn.metrics import average_precision_score, roc_auc_score

    plan = c.read(BASE / "plan.json")
    c.verify(plan["inputs"])
    old = c.read(OLD / "results.json")
    prior = c.read(DIAG)
    fits = {seed: c.read(OLD / f"fit_seed{seed}.json")["heads"] for seed in SEEDS}
    predictions = {
        seed: {r["item_id"]: r for r in c.read(OLD / f"predictions_seed{seed}.json")}
        for seed in SEEDS
    }
    replayed = 0
    bit_checks = 0
    arithmetic_checks = 0
    endpoints = {}
    for seed in SEEDS:
        c.require(len(predictions[seed]) == 970, "Changed prediction roster")
        for row in predictions[seed].values():
            restored = decode(row["ungated_probabilities"], fits[seed], "selected")
            c.require(
                s.decision(restored) == s.decision(row["result"]),
                "Selected prediction mismatch",
            )
            replayed += 1
    for name, ep in plan["endpoints"].items():
        gold = s.gold_rows(name)
        records = [r for r in gold if str(r["id"]) in set(ep["scorable_ids"])]
        c.require(
            [str(r["id"]) for r in records] == ep["scorable_ids"], "Gold/order mismatch"
        )
        modes = {}
        ranks = []
        for mode in MODES:
            runs = []
            for seed in SEEDS:
                heads = fits[seed]
                pp = predictions[seed]
                outputs = {
                    rid: decode(row["ungated_probabilities"], heads, mode)
                    for rid, row in pp.items()
                }
                con = s.contributions(gold, roster_outputs(gold, outputs), ep)
                bits = direct_bits(gold, ep, pp, heads, mode)
                c.require(
                    np.array_equal(bits, con["pred"]),
                    "Direct bit reconstruction differs",
                )
                bit_checks += int(bits.size)
                metric = s.summarize(con)
                metric["predicted_hate_false"] = sum(
                    (not outputs[str(r["id"])]["hate"] for r in records)
                )
                for lab, value in metric["per_label"].items():
                    tp, fp, fn = (value[k] for k in ("tp", "fp", "fn"))
                    value["f1"] = (
                        2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
                    )
                    value["precision"] = tp / (tp + fp) if tp + fp else 0.0
                    value["recall"] = tp / (tp + fn) if tp + fn else 0.0
                labels = [con["labels"][k] for k in con["macro_indices"]]
                direct_macro = np.mean(
                    [metric["per_label"][lab]["f1"] for lab in labels]
                )
                c.require(
                    abs(direct_macro - metric["macro_f1_valid_only"]) < 1e-12,
                    "Confusion arithmetic differs",
                )
                arithmetic_checks += 1
                if mode == "selected":
                    expected = old["endpoints"][name]["means"]["bge"]["values"][
                        SEEDS.index(seed)
                    ]
                    c.require(
                        abs(metric["macro_f1_valid_only"] - expected) < 1e-12,
                        "Original BGE score differs",
                    )
                    dimension = ep["dimension"]
                    order = ep["all_labels"]
                    raw = np.array(
                        [
                            pp[str(r["id"])]["ungated_probabilities"][dimension]
                            for r in records
                        ]
                    )
                    if dimension == "hate":
                        probs = np.column_stack((1 - raw[:, 0], raw[:, 0]))
                        c.require(
                            order == ["not_hate", "hate"], "Unexpected gate label order"
                        )
                    else:
                        source_labels = heads[dimension]["labels"]
                        probs = raw[:, [source_labels.index(label) for label in order]]
                    ranking = {}
                    for j in con["macro_indices"]:
                        truth = con["gold"][:, j]
                        if 0 < truth.sum() < len(truth):
                            ranking[order[j]] = {
                                "average_precision": float(
                                    average_precision_score(truth, probs[:, j])
                                ),
                                "roc_auc": float(roc_auc_score(truth, probs[:, j])),
                            }
                    ranks.append(
                        {
                            "seed": seed,
                            "per_label": ranking,
                            "macro_average_precision": float(
                                np.mean(
                                    [v["average_precision"] for v in ranking.values()]
                                )
                            ),
                            "macro_roc_auc": float(
                                np.mean([v["roc_auc"] for v in ranking.values()])
                            ),
                        }
                    )
                runs.append({"seed": seed, **metric})
            modes[mode] = {
                "runs": runs,
                "mean_macro_f1": float(
                    np.mean([r["macro_f1_valid_only"] for r in runs])
                ),
            }
        old_context = prior["evaluation"][name]["models"]["kb"]
        endpoints[name] = {
            "endpoint": ep,
            "policies": modes,
            "ungated_ranking": ranks,
            "old_compact_macro_f1": old_context["mean_assembled_macro_f1"],
            "old_compact_ungated_macro_ap": old_context["mean_head_average_precision"],
            "new_ungated_macro_ap": float(
                np.mean([r["macro_average_precision"] for r in ranks])
            ),
        }
        print(
            name,
            {m: round(v["mean_macro_f1"], 6) for m, v in modes.items()},
            flush=True,
        )
    c.verify(plan["inputs"])
    save(
        BASE / "results.json",
        {
            "created_utc": c.utc(),
            "plan_sha256": c.sha(BASE / "plan.json"),
            "status": "completed_exploratory_saved_decision_diagnostics",
            "endpoints": endpoints,
            "verification": {
                "selected_predictions_reproduced": replayed,
                "direct_indicator_bits_checked": bit_checks,
                "score_arithmetic_checks": arithmetic_checks,
                "original_scores_reproduced": 15,
            },
            "new_model_inference": 0,
            "training": 0,
            "Qwen_calls": 0,
            "CUDA_used": False,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "run"))
    args = parser.parse_args()
    with c.offline():
        globals()[args.mode]()


def roster_outputs(records, outputs):
    ids = [str(r["id"]) for r in records]
    c.require(len(ids) == len(set(ids)), "Duplicate endpoint identity")
    c.require(set(ids) <= set(outputs), "Missing requested prediction")
    return {rid: outputs[rid] for rid in ids}


if __name__ == "__main__":
    main()

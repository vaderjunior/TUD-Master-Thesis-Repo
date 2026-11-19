from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from training.roberta.short import results as sc
import numpy as np

r, c, s = sc.r, sc.c, sc.s


def main():
    from scipy.special import expit, softmax
    from sklearn.metrics import average_precision_score, roc_auc_score

    c.require(
        c.read(r.OUT / "integrity.json")["status"] == "pass", "Integrity required"
    )
    selections = {
        "expanded": c.read(r.BASE / "selection.json")["families"],
        "original": c.read(s.OLD / "selection.json")["families"],
    }
    configs = {v["id"]: v for v in s.configs()}
    profiles, counterfactual = {}, {}
    data = s.read_data()
    with np.load(r.OUT / "OOF_probabilities.npz", allow_pickle=False) as saved:
        old_gate = selections["original"]["tuned"]["hate"]["selected"]
        gates = saved["hate|" + old_gate["config"]][:, :, 1] > old_gate["threshold"]
        for head in c.HEADS:
            candidates = selections["expanded"]["tuned"][head]["candidates"]

            def key(row):
                cfg = configs[row["config"]]
                t = 0.5 if row["threshold"] is None else row["threshold"]
                return (
                    -row["mean_score"],
                    cfg["epochs"],
                    cfg["learning_rate"],
                    cfg["loss"] == "balanced",
                    abs(t - 0.5),
                    -t,
                )

            profiles[head] = {
                str(duration): min(
                    [
                        x
                        for x in candidates
                        if configs[x["config"]]["epochs"] == duration
                    ],
                    key=key,
                )
                for duration in s.DURATIONS
            }
            if head == "hate":
                continue
            spec = data["heads"][head]
            y = c.truth(head, spec["rows"], spec["labels"])
            supported = selections["expanded"]["tuned"][head]["selection_indices"]
            positions = [x["feature_row"] for x in spec["rows"]]
            values = []
            for row in candidates:
                probs = saved[head + "|" + row["config"]]
                per_seed = [
                    c.macro(
                        y,
                        c.decisions(
                            head, probs[i], row["threshold"], gates[i, positions]
                        ),
                        supported,
                    )
                    for i in range(3)
                ]
                values.append(
                    {
                        "config": row["config"],
                        "threshold": row["threshold"],
                        "seed_scores": per_seed,
                        "mean_score": float(np.mean(per_seed)),
                    }
                )
            old_candidates = selections["original"]["tuned"][head]["candidates"]
            lookup = {(x["config"], x["threshold"]): x for x in values}
            c.require(
                all(lookup[x["config"], x["threshold"]] == x for x in old_candidates),
                "Original-gate candidate reconstruction differs from prior TRAIN selection",
            )
            counterfactual[head] = {
                "best_with_original_gate": min(values, key=key),
                "best_per_duration": {
                    str(duration): min(
                        [
                            x
                            for x in values
                            if configs[x["config"]]["epochs"] == duration
                        ],
                        key=key,
                    )
                    for duration in s.DURATIONS
                },
                "all_candidate_scores": values,
            }
    r.save(
        "TRAIN_duration_profiles.json",
        {
            "description": "Best TRAIN OOF score within each predeclared duration, with the expanded selected gate for fine heads.",
            "not_unbiased_generalization_scores": True,
            "profiles": profiles,
            "original_gate_counterfactual": counterfactual,
            "counterfactual_not_a_new_selected_deployed_model": True,
            "different_schedule_horizons": True,
            "source_sha256": c.sha(Path(__file__)),
        },
    )
    routes = {
        x["job"]["id"]: x["origin"]
        for x in c.read(r.BASE / "final_routes.json")["routes"]
    }
    probability = {}
    for version in selections:
        for seed in c.FINAL_SEEDS:
            for head in c.HEADS:
                cfg = selections[version]["tuned"][head]["selected"]["config"]
                ident = f"{head}/{cfg}/s{seed}"
                base = (
                    r.BASE
                    if version == "expanded" and routes[ident] == "new_fit"
                    else s.OLD
                )
                z = np.load(
                    base / "inference" / ident / "logits.npy", allow_pickle=False
                ).astype(float)
                probability[version, seed, head] = (
                    expit(z)
                    if head in ("hate_type", "target_group")
                    else softmax(z, axis=1)
                )
    diagnostics, factorial, gate_mechanism = {}, {}, {}
    fine_probability_checks = 0
    for scope in ("core", "broad"):
        eps, positions = r.context(scope)
        for name, ep in eps.items():
            _, kept, ids, y = r.prior.gold(scope, name, ep)
            head = ep["dimension"]
            ix = [positions[k] for k in ids]
            labels = (
                data["heads"][head]["labels"]
                if head != "hate"
                else ["not_hate", "hate"]
            )
            order = [labels.index(label) for label in ep["all_labels"]]
            macro = [ep["all_labels"].index(label) for label in ep["macro_labels"]]
            for seed in c.FINAL_SEEDS:
                gates = {}
                for version, base in [("expanded", r.BASE), ("original", s.OLD)]:
                    heads = selections[version]["tuned"]
                    p = probability[version, seed, head][ix][:, order]
                    gp = probability[version, seed, "hate"][ix, 1]
                    gate = gp > heads["hate"]["selected"]["threshold"]
                    gates[version] = gate
                    per_label = {}
                    for j, label in enumerate(ep["all_labels"]):
                        truth = y[:, j]
                        per_label[label] = {
                            "average_precision": (
                                float(average_precision_score(truth, p[:, j]))
                                if truth.any()
                                else None
                            ),
                            "AUROC": (
                                float(roc_auc_score(truth, p[:, j]))
                                if truth.any() and (~truth).any()
                                else None
                            ),
                            "positive_probability_median": (
                                float(np.median(p[truth, j])) if truth.any() else None
                            ),
                            "negative_probability_median": (
                                float(np.median(p[~truth, j]))
                                if (~truth).any()
                                else None
                            ),
                        }
                    aps = [
                        per_label[ep["all_labels"][j]]["average_precision"]
                        for j in macro
                    ]
                    aucs = [per_label[ep["all_labels"][j]]["AUROC"] for j in macro]
                    diagnostics[f"{scope}/{name}/{version}/{seed}"] = {
                        "raw_per_label": per_label,
                        "macro_average_precision": (
                            float(np.mean(aps))
                            if all(v is not None for v in aps)
                            else None
                        ),
                        "macro_AUROC": (
                            float(np.mean(aucs))
                            if all(v is not None for v in aucs)
                            else None
                        ),
                        "gate_negative_fraction": float((~gate).mean()),
                    }
                    if head == "hate":
                        thresholds = sorted(
                            {
                                0.5,
                                selections["original"]["tuned"]["hate"]["selected"][
                                    "threshold"
                                ],
                                selections["expanded"]["tuned"]["hate"]["selected"][
                                    "threshold"
                                ],
                            }
                        )
                        for cutoff in thresholds:
                            out = gp > cutoff
                            stats = sc.metric(y, np.column_stack((~out, out)), ep)[0]
                            factorial[f"{scope}/{name}/{version}/{seed}/{cutoff}"] = (
                                stats
                            )
                if head != "hate":
                    c.require(
                        np.array_equal(
                            probability["expanded", seed, head],
                            probability["original", seed, head],
                        ),
                        "Fine-head probability changed despite exact reuse",
                    )
                    old = r.bits(
                        s.OLD, "tuned_selected", seed, head, ids, positions, ep
                    )
                    new = r.bits(
                        r.BASE, "tuned_selected", seed, head, ids, positions, ep
                    )
                    raw = r.bits(
                        r.BASE,
                        "tuned_selected",
                        seed,
                        head,
                        ids,
                        positions,
                        ep,
                        ungated=True,
                    )
                    c.require(
                        np.array_equal(old, raw & gates["original"][:, None])
                        and np.array_equal(new, raw & gates["expanded"][:, None]),
                        "Gate-only explanation failed",
                    )
                    opened, closed = (
                        ~gates["original"] & gates["expanded"],
                        gates["original"] & ~gates["expanded"],
                    )
                    c.require(
                        np.array_equal((new & ~old), raw & opened[:, None])
                        and np.array_equal((old & ~new), raw & closed[:, None]),
                        "Added/removed fine-label mechanism",
                    )
                    gate_mechanism[f"{scope}/{name}/{seed}"] = {
                        "gate_opened": int(opened.sum()),
                        "gate_closed": int(closed.sum()),
                        "fine_labels_added": int((new & ~old).sum()),
                        "fine_labels_removed": int((old & ~new).sum()),
                        "all_fine_label_changes_explained_by_gate": True,
                    }
                    fine_probability_checks += 1
            print("Gate and ranking explanation", scope, name, flush=True)
    r.save(
        "diagnostics.json",
        {
            "post_result_supporting": True,
            "raw_probability_cells": diagnostics,
            "gate_only_fine_changes": gate_mechanism,
            "fine_probability_equality_checks": fine_probability_checks,
            "new_gate_weight_cutoff_factorial": factorial,
            "new_thresholds_searched_on_evaluation": False,
            "Qwen_calls": 0,
            "source_sha256": c.sha(Path(__file__)),
        },
    )
    training = c.read(r.OUT / "training_runs.json")
    curves = {}
    for key, values in training["curves"].items():
        curves[key] = {
            "fits": len(values),
            "first_held_loss_mean": float(np.mean([x["first_loss"] for x in values])),
            "terminal_held_loss_mean": float(np.mean([x["last_loss"] for x in values])),
            "minimum_held_loss_mean": float(
                np.mean([x["minimum_loss"] for x in values])
            ),
            "earliest_minimum_epoch_median": float(
                np.median([x["earliest_minimum_epoch"] for x in values])
            ),
            "terminal_worse_than_first": sum(
                x["last_loss"] > x["first_loss"] for x in values
            ),
            "terminal_worse_than_minimum": sum(
                x["terminal_minus_best_loss"] > 0 for x in values
            ),
        }
    r.save(
        "curve_summary.json",
        {
            "post_result_description_only": True,
            "not_early_stopping": True,
            "cells": curves,
        },
    )
    integrity = c.read(r.OUT / "integrity.json")
    start = datetime.fromisoformat(integrity["execution_started"]["utc"])
    finish = datetime.fromisoformat(integrity["completion_filesystem_UTC_proxy"])
    new_runs = [x for x in training["runs"] if x["origin"].startswith("new_")]
    weights = []
    for row in new_runs:
        if row["origin"] == "new_final":
            path = c.ROOT / row["attempt"] / "model/model.safetensors"
            weights.append(
                {
                    "path": c.rel(path),
                    "bytes": path.stat().st_size,
                    "sha256": c.sha(path),
                }
            )
    r.save(
        "resources.json",
        {
            "costs_by_origin": training["costs_by_origin"],
            "wall_span_seconds_filesystem_end_proxy": (finish - start).total_seconds(),
            "start_utc": start.isoformat(),
            "end_utc_filesystem_proxy": finish.isoformat(),
            "new_search_candidates": 1800,
            "reused_search_candidates": 720,
            "new_final_models": 5,
            "reused_final_prediction_jobs": 55,
            "new_inference_head_rows": 5 * 19082,
            "decoded_complete_outputs": 30 * 19082,
            "new_weight_files": weights,
            "new_weight_bytes": sum(x["bytes"] for x in weights),
            "new_attempts": integrity["new_attempts"],
            "stopped_attempts": integrity["stopped_attempts"],
            "historical_cost_is_not_free_tuning": True,
            "training_metadata_sums_exclude_untimed_validation_and_inference": True,
            "no_energy_or_euro_cost_measured": True,
            "external_archives_rehashed": False,
            "Qwen_calls": 0,
            "review_training_or_model_inference": 0,
        },
    )


if __name__ == "__main__":
    with r.offline():
        main()

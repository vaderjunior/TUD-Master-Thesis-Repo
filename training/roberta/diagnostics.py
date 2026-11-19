from collections import defaultdict
from pathlib import Path
import socket
from unittest.mock import patch
import numpy as np
from training.roberta import results as r

c = r.c


def main():
    from scipy.special import logsumexp, expit, softmax
    from sklearn.metrics import average_precision_score, roc_auc_score

    c.require(
        c.read(r.OUT / "integrity.json")["status"] == "pass", "Audited outputs required"
    )
    data = c.read(c.BASE / "data/train.json")
    selection = c.read(c.BASE / "selection.json")["families"]
    curves = []
    max_loss_diff = 0.0
    for job in c.search_jobs():
        receipt = c.read(c.job_dir(job) / "complete.json")
        attempt = c.BASE / receipt["attempt"]
        meta = c.read(attempt / "training.json")
        spec = data["heads"][job["head"]]
        held = [
            v
            for v in spec["rows"]
            if data["folds"]["42"][v["feature_row"]] == job["fold"]
        ]
        y = np.asarray([v["target"] for v in held])
        z = np.load(attempt / "held_logits.npy", allow_pickle=False).astype(float)
        loss = (
            float(np.mean(np.logaddexp(0, z) - y * z))
            if spec["kind"] == "multilabel"
            else float(np.mean(logsumexp(z, axis=1) - z[np.arange(len(y)), y]))
        )
        error = abs(loss - meta["epochs"][-1]["held_unweighted_loss"])
        max_loss_diff = max(max_loss_diff, error)
        c.require(error <= 2e-6, "Independent terminal held loss differs")
        curves.append(
            {
                "job": job,
                "epochs": meta["epochs"],
                "last_held_loss_recomputed": loss,
                "loss_max_error": error,
            }
        )
    groups = defaultdict(list)
    for row in curves:
        groups[row["job"]["head"] + "/" + row["job"]["config"]].append(row)
    summaries = {}
    for name, values in groups.items():
        first = np.asarray([v["epochs"][0]["held_unweighted_loss"] for v in values])
        last = np.asarray([v["epochs"][-1]["held_unweighted_loss"] for v in values])
        summaries[name] = {
            "fits": len(values),
            "first_held_loss_mean": float(first.mean()),
            "last_held_loss_mean": float(last.mean()),
            "held_loss_increased": int((last > first).sum()),
            "training_objective_decreased": sum(
                v["epochs"][-1]["training_objective"]
                < v["epochs"][0]["training_objective"]
                for v in values
            ),
        }
    r.save(
        "learning_curves.json",
        {
            "post_result_description": True,
            "selection_did_not_use_early_stopping": True,
            "held_loss_max_recalculation_error": max_loss_diff,
            "summaries": summaries,
            "runs": curves,
            "caution": "TRAIN-fold held losses describe generalization/overfit, not a second model-selection step. Weighted training objectives and unweighted held losses are different quantities.",
        },
    )
    logits = {}
    for family, heads in selection.items():
        for seed in c.FINAL_SEEDS:
            for head in c.HEADS:
                cfg = heads[head]["selected"]["config"]
                path = c.BASE / "inference" / f"{head}/{cfg}/s{seed}/logits.npy"
                logits[family, seed, head] = np.load(path, allow_pickle=False)
    diagnostics = {}
    cutoffs = {}
    for scope in ["core", "broad"]:
        eps, positions = r.context(scope)
        for name, ep in eps.items():
            _, kept, ids, y = r.gold(scope, name, ep)
            head = ep["dimension"]
            ix = [positions[k] for k in ids]
            labels = (
                data["heads"][head]["labels"]
                if head != "hate"
                else ["not_hate", "hate"]
            )
            order = [labels.index(l) for l in ep["all_labels"]]
            macro = [ep["all_labels"].index(l) for l in ep["macro_labels"]]
            for family, heads in selection.items():
                for seed in c.FINAL_SEEDS:
                    z = logits[family, seed, head][ix].astype(float)
                    p = (
                        expit(z)
                        if head in ["hate_type", "target_group"]
                        else softmax(z, axis=1)
                    )
                    p = p[:, order]
                    gate_p = softmax(
                        logits[family, seed, "hate"][ix].astype(float), axis=1
                    )[:, 1]
                    selected_gate = gate_p > heads["hate"]["selected"]["threshold"]
                    threshold = heads[head]["selected"]["threshold"]
                    per_label = {}
                    for j, label in enumerate(ep["all_labels"]):
                        yy = y[:, j]
                        pp = p[:, j]
                        per_label[label] = {
                            "prevalence": float(yy.mean()),
                            "average_precision": (
                                float(average_precision_score(yy, pp))
                                if yy.any()
                                else None
                            ),
                            "auroc": (
                                float(roc_auc_score(yy, pp))
                                if yy.any() and (~yy).any()
                                else None
                            ),
                            "positive_probability_median": (
                                float(np.median(pp[yy])) if yy.any() else None
                            ),
                            "negative_probability_median": (
                                float(np.median(pp[~yy])) if (~yy).any() else None
                            ),
                        }
                    aps = [
                        per_label[ep["all_labels"][j]]["average_precision"]
                        for j in macro
                    ]
                    aurocs = [per_label[ep["all_labels"][j]]["auroc"] for j in macro]
                    if head == "hate":
                        raw = np.column_stack((~selected_gate, selected_gate))
                    elif head == "severity":
                        raw = np.eye(p.shape[1], dtype=bool)[p.argmax(1)]
                    else:
                        raw = p >= threshold
                    delivered = raw.copy()
                    if head != "hate":
                        delivered[~selected_gate] = False
                    expected = r.new_bits(
                        family + "_selected", seed, head, ids, positions, ep
                    )
                    c.require(
                        np.array_equal(delivered, expected),
                        "Diagnostic reconstruction differs",
                    )
                    diagnostics[f"{scope}/{name}/{family}/{seed}"] = {
                        "rows": len(ids),
                        "raw_per_label": per_label,
                        "macro_average_precision_ungated": (
                            float(np.mean(aps))
                            if all(a is not None for a in aps)
                            else None
                        ),
                        "macro_auroc_ungated": (
                            float(np.mean(aurocs))
                            if all(a is not None for a in aurocs)
                            else None
                        ),
                        "ungated": r.previous.summarize_bits(y, raw, ep)[0],
                        "delivered": r.previous.summarize_bits(y, delivered, ep)[0],
                        "gate_negative_fraction": float((~selected_gate).mean()),
                    }
                    if head == "hate":
                        continue
                    for gate_policy in ["default", "selected"]:
                        gate = gate_p > (
                            0.5
                            if gate_policy == "default"
                            else heads["hate"]["selected"]["threshold"]
                        )
                        for head_policy in (
                            ["selected"]
                            if head == "severity"
                            else ["default", "selected"]
                        ):
                            zz = (
                                np.eye(p.shape[1], dtype=bool)[p.argmax(1)]
                                if head == "severity"
                                else p
                                >= (0.5 if head_policy == "default" else threshold)
                            )
                            zz[~gate] = False
                            cutoffs[
                                f"{scope}/{name}/{family}/{seed}/gate_{gate_policy}/head_{head_policy}"
                            ] = r.previous.summarize_bits(y, zz, ep)[0]
            print("Explained gate/cutoff/ranking", scope, name, flush=True)
    r.save(
        "diagnostics.json",
        {
            "status": "pass",
            "post_result_descriptions": True,
            "cells": diagnostics,
            "ranking_scope": "Raw probabilities before delivery gate, on the same annotated evaluation masks; not another cutoff search or new model selection.",
        },
    )
    r.save(
        "cutoff_factorial.json",
        {
            "post_result_descriptive": True,
            "policy": "Only the already frozen default and TRAIN-selected cutoffs, crossed on fixed weights. Do not promote an evaluation-selected winner.",
            "cells": cutoffs,
        },
    )
    cpu_plan = c.read(r.OUT / "cpu_reload_plan.json")
    positions = cpu_plan["positions"]
    cpu = {}
    for job in cpu_plan["jobs"]:
        path = r.OUT / "cpu_reloads" / f"{job['head']}_{job['config']}_s10.npy"
        cpu[job["head"], job["config"]] = c.probabilities(
            job["head"], np.load(path, allow_pickle=False)
        )
    checks = 0
    for family, heads in selection.items():
        for policy in ["selected", "default"]:
            gate = None
            with np.load(
                c.BASE / "predictions" / f"{family}_{policy}_s10.npz",
                allow_pickle=False,
            ) as saved:
                for head in c.HEADS:
                    threshold = (
                        None
                        if head == "severity"
                        else (
                            0.5
                            if policy == "default"
                            else heads[head]["selected"]["threshold"]
                        )
                    )
                    z = c.decisions(
                        head,
                        cpu[head, heads[head]["selected"]["config"]],
                        threshold,
                        gate,
                    )
                    c.require(
                        np.array_equal(z, saved[head][positions]),
                        "CPU-probe delivery decisions differ",
                    )
                    if head == "hate":
                        gate = z[:, 1].astype(bool)
                    checks += len(z)
    r.save(
        "diagnostics_verification.json",
        {
            "status": "pass",
            "terminal_held_losses_recomputed": 720,
            "max_loss_difference": max_loss_diff,
            "CPU_probe_head_decisions_exact": checks,
            "diagnostic_cells": len(diagnostics),
            "cutoff_factorial_cells": len(cutoffs),
            "source_sha256": c.sha(Path(__file__)),
            "Qwen_calls": 0,
            "new_training_or_CUDA": False,
        },
    )


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", r.forbid), patch.object(
        socket, "create_connection", r.forbid
    ):
        main()

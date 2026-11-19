import argparse
from collections import defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from analysis.common import replay as c

ROOT = c.ROOT
CORE = c.CORE
BASE = c.BASE / "encoders"
SOURCE = Path(__file__).resolve()
INVENTORY = (
    ROOT / "experiments/astra/qwen_downtime_inventory_20261006_v1/inventory.json"
)
SUMMARY = (
    ROOT / "Thesis_Log/astra_result/evidence/core_completion_summary_20260920_v1.json"
)
GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def independent_rank(y, s):
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    pos = int(y.sum())
    neg = len(y) - pos
    if not pos:
        return (None, None)
    order = np.argsort(-s, kind="stable")
    yy = y[order]
    ss = s[order]
    ends = np.r_[np.flatnonzero(np.diff(ss) != 0), len(y) - 1]
    tp = np.cumsum(yy)[ends]
    fp = ends + 1 - tp
    recall = tp / pos
    precision = tp / (ends + 1)
    ap = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    auc = float(np.trapezoid(np.r_[0.0, recall], np.r_[0.0, fp / neg])) if neg else None
    return (ap, auc)


def ranking(y, p, labels, macro_labels):
    out = {}
    eligible = []
    for k, lab in enumerate(labels):
        yy = y[:, k]
        ss = p[:, k]
        ap, auc = independent_rank(yy, ss)
        if ap is not None:
            assert abs(ap - average_precision_score(yy, ss)) < 1e-12
        if auc is not None:
            assert abs(auc - roc_auc_score(yy, ss)) < 1e-12
        out[lab] = {
            "positive": int(yy.sum()),
            "negative": int(len(yy) - yy.sum()),
            "prevalence": float(yy.mean()),
            "average_precision": ap,
            "roc_auc": auc,
            "brier": float(np.mean((ss - yy) ** 2)),
            "positive_probability_quantiles": (
                np.quantile(ss[yy == 1], [0, 0.25, 0.5, 0.75, 1]).tolist()
                if yy.any()
                else None
            ),
            "negative_probability_quantiles": (
                np.quantile(ss[yy == 0], [0, 0.25, 0.5, 0.75, 1]).tolist()
                if (yy == 0).any()
                else None
            ),
            "fraction_within_005_of_half": float((np.abs(ss - 0.5) <= 0.05).mean()),
        }
        if lab in macro_labels and ap is not None and (auc is not None):
            eligible.append(lab)
    return {
        "per_label": out,
        "ranking_macro_labels": eligible,
        "macro_average_precision": (
            float(np.mean([out[l]["average_precision"] for l in eligible]))
            if eligible
            else None
        ),
        "macro_roc_auc": (
            float(np.mean([out[l]["roc_auc"] for l in eligible])) if eligible else None
        ),
        "mean_prevalence_of_ranking_labels": (
            float(np.mean([out[l]["prevalence"] for l in eligible]))
            if eligible
            else None
        ),
    }


def probabilities(logits, kind):
    z = np.asarray(logits, dtype=float)
    if kind == "multilabel":
        return 1 / (1 + np.exp(-np.clip(z, -700, 700)))
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def prepare():
    assert not (BASE / "plan.json").exists()
    inv = c.read(INVENTORY)
    assert inv["status"] == "availability_and_selected_file_integrity_pass"
    ep = c.read(CORE / "api_execution_v1/scoring_plan.json")["endpoints"]
    paths = [
        SOURCE,
        Path(c.__file__).resolve(),
        ROOT / "tests/classification/test_operating_points.py",
        INVENTORY,
        SUMMARY,
        CORE / "api_execution_v1/scoring_plan.json",
        CORE / "dev_inputs.jsonl",
        CORE / "encoder_plan.json",
    ]
    paths += [ROOT / f"data/processed/{k}.parquet" for k in ep]
    for job in inv["encoder_jobs"]:
        for value in job["files"].values():
            p = ROOT / value["path"]
            assert c.sha(p) == value["sha256"]
            paths.append(p)
    c.save(
        BASE / "plan.json",
        {
            "created_utc": c.utc(),
            "status": "frozen_before_new_logit_summaries",
            "endpoints": ep,
            "jobs": inv["encoder_jobs"],
            "threshold_grid": GRID,
            "inputs": {c.rel(p): c.sha(p) for p in paths},
            "questions": [
                "Does the saved head rank positives above negatives?",
                "What probability/support patterns accompany the fixed default decisions?",
                "How sensitive are decisions to a predeclared global threshold grid?",
            ],
            "metrics": "Per-label average precision and ROC AUC, both independently tie-aware reconstructed; positive prevalence, Brier and fixed quantiles. Head rankings are ungated. Macro ranking uses original macro labels with positives AND negatives.",
            "decisions": "Reconstruct the original decision and hate gate, and reproduce all 36 aligned core seed/head scores. Fine-label threshold grid keeps the original gate fixed; separate gate-grid behavior is a diagnostic, not a new assembled-system recipe.",
            "threshold_rule": "Grid 0.1..0.9 fixed before inspection. No best threshold/seed is selected, no new deployed/calibrated score. Binary hate uses > threshold (ties non-hate); multilabel uses >= threshold. Severity keeps argmax.",
            "validation": "Inspect existing internal-validation support/ranking; it was used in model selection and is not a new untouched validation set. Missing positive or negative classes yield absent ranking metrics, not zeros.",
            "uncertainty": "Descriptive fixed-seed diagnostic, all seeds and labels shown. No new superiority test, tuning, selected subset or retraining.",
            "scope": "Completed aligned English v5 / retained German v4 only. No Qwen, human or unfinished balanced-correction data.",
        },
    )
    print("Encoder confidence plan frozen.", flush=True)


def run():
    plan = c.read(BASE / "plan.json")
    c.verify(plan["inputs"])
    assert not (BASE / "results.json").exists()
    jobs = {j["job"]: j for j in plan["jobs"]}
    meta = c.read(CORE / "encoder_plan.json")["split_summaries"]
    raw = {}
    validation = {}
    rankchecks = 0
    for name, job in jobs.items():
        rows = c.rows(ROOT / job["files"]["dev_head_logits.jsonl"]["path"])
        v = c.rows(ROOT / job["files"]["internal_val_logits.jsonl"]["path"])
        arm, head, seed = name.split("/")
        m = meta[f"{arm}/{head}"]
        labels = m["labels"]
        kind = m["kind"]
        assert all((x["labels"] == labels for x in rows))
        raw[name] = {x["key"]: x for x in rows}
        assert len(raw[name]) == len(rows)
        logits = np.asarray([x["logits"] for x in v])
        p = probabilities(logits, kind)
        target = np.asarray([x["target"] for x in v], dtype=int)
        y = target if kind == "multilabel" else np.eye(len(labels), dtype=int)[target]
        expected = (
            (logits >= 0).astype(int) if kind == "multilabel" else logits.argmax(axis=1)
        )
        assert np.array_equal(expected, np.asarray([x["decision"] for x in v]))
        stats = ranking(y, p, labels, labels)
        rankchecks += sum(
            (x["average_precision"] is not None for x in stats["per_label"].values())
        ) + sum((x["roc_auc"] is not None for x in stats["per_label"].values()))
        validation[name] = {
            "train_rows": m["counts"]["train"],
            "validation_rows": len(v),
            "declared_validation_rows": m["counts"]["val"],
            "kind": kind,
            **stats,
        }
        assert len(v) == m["counts"]["val"]
    summary = c.read(SUMMARY)
    evaluations = {}
    oldchecks = 0
    for subset, ep in plan["endpoints"].items():
        rows = c.gold_rows(subset)
        ids = [str(x["id"]) for x in rows]
        assert ids == ep["scorable_ids"]
        head = ep["dimension"]
        lang = "en" if subset.startswith("en_") else "de"
        arms = ["full", "kb"] if lang == "en" else ["de"]
        groups = {}
        for arm in arms:
            runs = []
            for seed in range(3):
                name = f"{arm}/{head}/seed{seed}"
                h = raw[name]
                m = meta[f"{arm}/{head}"]
                kind = m["kind"]
                labels = m["labels"]
                hh = [h[lang + "/" + rid] for rid in ids]
                z = np.asarray([x["logits"] for x in hh])
                p = probabilities(z, kind)
                if head == "hate":
                    y = np.eye(2, dtype=int)[[int(x["gate"]) for x in rows]]
                elif head == "severity":
                    y = np.eye(3, dtype=int)[
                        [labels.index(x["severity"]) for x in rows]
                    ]
                else:
                    y = np.array(
                        [
                            [
                                int(lab in x[c.original.GOLD_FIELDS[head]])
                                for lab in labels
                            ]
                            for x in rows
                        ]
                    )
                ranking_labels = labels if head == "hate" else ep["macro_labels"]
                stats = ranking(y, p, labels, ranking_labels)
                rankchecks += sum(
                    (
                        x["average_precision"] is not None
                        for x in stats["per_label"].values()
                    )
                ) + sum((x["roc_auc"] is not None for x in stats["per_label"].values()))
                gate = np.asarray(
                    [
                        bool(
                            raw[f"{arm}/hate/seed{seed}"][lang + "/" + rid]["decision"]
                        )
                        for rid in ids
                    ]
                )
                pred = (
                    (z >= 0).astype(int) if kind == "multilabel" else z.argmax(axis=1)
                )
                assert np.array_equal(pred, np.asarray([x["decision"] for x in hh]))

                def outputs(threshold=None, ignore_gate=False):
                    out = {}
                    for k, rid in enumerate(ids):
                        res = {
                            "hate": bool(gate[k]),
                            "target_group": [],
                            "hate_type": [],
                            "severity": None,
                            "legal": [],
                        }
                        if head == "hate":
                            res["hate"] = (
                                bool(pred[k])
                                if threshold is None
                                else bool(p[k, 1] > threshold)
                            )
                        elif head == "severity":
                            res["severity"] = (
                                labels[int(pred[k])] if gate[k] or ignore_gate else None
                            )
                        else:
                            res[head] = [
                                lab
                                for n, lab in enumerate(labels)
                                if (
                                    bool(pred[k, n])
                                    if threshold is None
                                    else p[k, n] >= threshold
                                )
                                and (head == "legal" or gate[k] or ignore_gate)
                            ]
                        out[rid] = res
                    return out

                score = c.summarize(c.contributions(rows, outputs(), ep))
                orig = summary["endpoints"][subset]["conditions"][
                    "encoder_kb" if arm == "kb" else "encoder_full"
                ]["values"][seed]
                assert abs(score["macro_f1_valid_only"] - orig) < 1e-12
                oldchecks += 1
                grid = []
                if kind != "ordinal":
                    for threshold in GRID:
                        st = c.summarize(
                            c.contributions(rows, outputs(threshold=threshold), ep)
                        )
                        grid.append(
                            {
                                "threshold": threshold,
                                "macro_f1": st["macro_f1_valid_only"],
                                "mean_predicted_label_count": st[
                                    "mean_predicted_label_count"
                                ],
                                "per_label_counts": st["per_label_counts"],
                            }
                        )
                    assert (
                        abs(
                            next((x["macro_f1"] for x in grid if x["threshold"] == 0.5))
                            - orig
                        )
                        < 1e-12
                    )
                ungated = (
                    c.summarize(c.contributions(rows, outputs(ignore_gate=True), ep))
                    if head not in ("hate", "legal")
                    else score
                )
                record = {
                    "seed": seed,
                    "n": len(rows),
                    "kind": kind,
                    "assembled_score": score,
                    "ungated_score": ungated,
                    "head_ranking": stats,
                    "threshold_grid_diagnostic": grid,
                    "gate_negative": int((~gate).sum()),
                    "head_positive_decisions": (
                        int(np.count_nonzero(pred)) if kind == "multilabel" else None
                    ),
                    "all_head_labels_below_half_rows": (
                        int((p < 0.5).all(axis=1).sum())
                        if kind == "multilabel"
                        else None
                    ),
                }
                if head == "severity":
                    true = y.argmax(axis=1)
                    distance = np.abs(pred - true)
                    record["ordinal_head_confusions"] = {
                        "exact": int((distance == 0).sum()),
                        "adjacent_level": int((distance == 1).sum()),
                        "low_high_confusion": int((distance == 2).sum()),
                        "note": "Head before gating. Adjacent/two-level counts use ordering only; no numeric severity distance is assumed.",
                    }
                runs.append(record)
            groups[arm] = {
                "runs": runs,
                "mean_assembled_macro_f1": float(
                    np.mean([x["assembled_score"]["macro_f1_valid_only"] for x in runs])
                ),
                "mean_ungated_macro_f1": float(
                    np.mean([x["ungated_score"]["macro_f1_valid_only"] for x in runs])
                ),
                "mean_head_average_precision": float(
                    np.mean(
                        [x["head_ranking"]["macro_average_precision"] for x in runs]
                    )
                ),
                "mean_head_roc_auc": float(
                    np.mean([x["head_ranking"]["macro_roc_auc"] for x in runs])
                ),
                "mean_head_positive_prevalence": float(
                    np.mean(
                        [
                            x["head_ranking"]["mean_prevalence_of_ranking_labels"]
                            for x in runs
                        ]
                    )
                ),
            }
            print(
                f"Diagnosed {subset}/{arm}; original seed scores reproduce.", flush=True
            )
        evaluations[subset] = {"endpoint": ep, "models": groups}
    assert oldchecks == 36
    result = {
        "created_utc": c.utc(),
        "status": "complete",
        "plan_sha256": c.sha(BASE / "plan.json"),
        "validation": validation,
        "evaluation": evaluations,
        "original_encoder_seed_scores_verified": oldchecks,
        "independent_ranking_metric_checks": rankchecks,
        "actual_API_calls": 0,
        "new_fits": 0,
    }
    c.save(BASE / "results.json", result)
    c.save(
        BASE / "complete.json",
        {
            "status": "complete",
            "results_sha256": c.sha(BASE / "results.json"),
            "source_sha256": c.sha(SOURCE),
        },
    )


def report():
    q = c.read(BASE / "results.json")
    assert c.read(BASE / "complete.json")["results_sha256"] == c.sha(
        BASE / "results.json"
    )
    text = [
        "# Saved encoder confidence and label-ranking diagnostics",
        "",
        "6 October 2026. No fitting, new model prediction, threshold selection, Qwen call or CUDA. All 36 original aligned core encoder seed scores reproduce. Average-precision and ROC-area calculations independently reproduce with ties handled explicitly.",
        "",
        "The encoders are the trained English RoBERTa and German GBERT SQ2 baselines. These scores describe the saved classifier heads. The existing gate/head performance decomposition is reproduced as context, not a newly discovered effect.",
        "",
        "## Ranking and actual system performance",
        "",
        "AP (average precision) asks whether true labels tend to receive higher scores. Higher is better; positive prevalence supplies context, not a formal random-ranking significance bound. Ranking uses the ungated head; it is not the delivered system F1 or proof that a new threshold is valid.",
        "",
        "| Endpoint | Encoder arm | Assembled F1 | Head before gate F1 | Head macro AP | Mean positive prevalence | Head macro ROC AUC |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ep, x in q["evaluation"].items():
        for arm, z in x["models"].items():
            text.append(
                f"| {ep} | {arm} | {z['mean_assembled_macro_f1']:.6f} | {z['mean_ungated_macro_f1']:.6f} | {z['mean_head_average_precision']:.6f} | {z['mean_head_positive_prevalence']:.6f} | {z['mean_head_roc_auc']:.6f} |"
            )
    text += [
        "",
        "`full` and `kb` are full-data and compact-trained English models; `de` is the retained full-data German model. These are not RAG KB sizes. Binary gate ranking averages both classes; fine-label ranking follows the fixed supported label set.",
        "",
        "## Validation support",
        "",
        "The same training/validation split is shared across seeds. These existing internal-validation data participated in model selection. The compact sets are particularly small; ranking support does not create a reliable calibration sample.",
        "",
        "| Model/head | Training rows | Validation rows | Validation positives by label |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, x in q["validation"].items():
        if name.endswith("seed0"):
            text.append(
                f"| {name.rsplit('/', 1)[0]} | {x['train_rows']} | {x['validation_rows']} | "
                + ", ".join(
                    (f"{k}: {v['positive']}" for k, v in x["per_label"].items())
                )
                + " |"
            )
    text += [
        "",
        "## Fixed threshold grid: diagnostic only",
        "",
        "All nine thresholds were fixed before the new summaries. Each fine-head sweep retains its original predicted gate; the separate hate-gate sweep does not refit the fine heads or reassemble a newly optimized full model. Severity retains argmax. Do not choose the best DEV cell as a deployable improvement.",
        "",
        "| Endpoint | Arm | Threshold | Mean assembled macro-F1 |",
        "| --- | --- | ---: | ---: |",
    ]
    runs = [
        "# Every encoder seed: confidence diagnostics",
        "",
        "All values use saved logits; original assembled scores reproduce.",
        "",
        "| Endpoint | Arm | Seed | Assembled F1 | Head F1 | Head AP | Head ROC AUC | Gate negatives |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for ep, x in q["evaluation"].items():
        for arm, z in x["models"].items():
            for item in z["runs"]:
                runs.append(
                    f"| {ep} | {arm} | {item['seed']} | {item['assembled_score']['macro_f1_valid_only']:.6f} | {item['ungated_score']['macro_f1_valid_only']:.6f} | {item['head_ranking']['macro_average_precision']:.6f} | {item['head_ranking']['macro_roc_auc']:.6f} | {item['gate_negative']} |"
                )
            if z["runs"][0]["threshold_grid_diagnostic"]:
                for k, t in enumerate(GRID):
                    text.append(
                        f"| {ep} | {arm} | {t:.1f} | {np.mean([y['threshold_grid_diagnostic'][k]['macro_f1'] for y in z['runs']]):.6f} |"
                    )
    text += [
        "",
        "This is a descriptive exposed-DEV sensitivity. It cannot establish that encoders generally fail, that changing a threshold will generalize, or that retrieval is universally more label-efficient. Broad compact-model claims would require a stronger predeclared TRAIN-only validation design, and possibly additional fitting.",
        "",
        "Evidence: [frozen plan](../../experiments/astra/offline_closure_20261006_v1/encoders/plan.json), [every label/seed/threshold and confidence quantiles](../../experiments/astra/offline_closure_20261006_v1/encoders/results.json).",
    ]
    runs += [
        "",
        "Full per-label support, AP, ROC, Brier, score quantiles, every threshold confusion matrix and severity ordered-confusion counts are in [the machine result](../../experiments/astra/offline_closure_20261006_v1/encoders/results.json).",
    ]
    for name, lines in [
        ("astra_encoder_confidence_results_20261006_v1.md", text),
        ("astra_encoder_confidence_run_tables_20261006_v1.md", runs),
    ]:
        with (ROOT / "Thesis_Log/astra_result" / name).open(
            "x", encoding="utf8", newline="\n"
        ) as f:
            f.write("\n".join(lines) + "\n")
    print("Encoder reports written.", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "run", "report"])
    mode = p.parse_args().mode
    with c.network_guard():
        globals()[mode]()


if __name__ == "__main__":
    main()

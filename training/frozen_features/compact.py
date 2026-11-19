import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import argparse
from collections import Counter
import importlib.metadata
import io
from pathlib import Path
import platform
import re
import time
import unicodedata
import warnings
import numpy as np
from analysis.common import study as c
from analysis.common.api import save_new

BASE = c.ROOT / "experiments/astra/compact_bge_20261008_v1"
HEADS = ("hate", "hate_type", "target_group", "severity")
FIELDS = dict(
    hate="gate",
    hate_type="hate_types",
    target_group="target_groups",
    severity="severity",
)
SEEDS = (42, 43, 44)
CS = (0.1, 1.0, 10.0, 100.0)
THRESHOLDS = tuple((i / 10 for i in range(1, 10)))
PRIMARY = (
    "en_dev_eval_main",
    "en_dev_eval_sq3_types",
    "en_dev_eval_targets",
    "en_dev_eval_severity",
)


def save(path, obj):
    path = Path(path).resolve()
    c.require(path.is_relative_to(BASE.resolve()), "Outside compact experiment")
    save_new(path, obj)


def save_array(path, value):
    data = io.BytesIO()
    np.save(data, value, allow_pickle=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(data.getvalue())
        f.flush()
        os.fsync(f.fileno())


def norm(text):
    return re.sub("\\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def environment():
    return {
        "python": platform.python_version(),
        "packages": {
            n: importlib.metadata.version(n)
            for n in ("numpy", "pandas", "pyarrow", "scikit-learn", "scipy")
        },
    }


def validate():
    p = c.read(BASE / "plan.json")
    c.verify(p["inputs"])
    c.verify(p["prepared"])
    c.require(p["environment"] == environment(), "Compact environment changed")
    return p


def prepare():
    if (BASE / "plan.json").exists():
        validate()
        print("Existing compact plan verified; no changes.")
        return
    import pyarrow.dataset, pyarrow.parquet
    import pandas as pd
    from sklearn.model_selection import StratifiedGroupKFold

    old = c.read(c.CORE / "plan.json")
    rp = c.read(c.CORE / "retrieval_v1/plan.json")
    c.verify(rp["generated_hashes"])
    full = c.rows(c.CORE / "kb/full/records.jsonl")
    small = {
        r["id"]: r
        for r in c.rows(c.CORE / "kb/small/records.jsonl")
        if r["lang"] == "en" and r["kind"] == "example"
    }
    ix = {r["id"]: i for i, r in enumerate(full)}
    vectors = np.load(
        c.CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    train = {}
    input_paths = []
    for head in HEADS:
        meta_path = c.CORE / f"encoder_data/kb/{head}_meta.json"
        meta = c.read(meta_path)
        rr = []
        for part in ("train", "val"):
            path = c.CORE / f"encoder_data/kb/{head}_{part}.parquet"
            c.require(
                c.sha(path) == old["generated_hashes"][c.rel(path)],
                "Changed TRAIN partition",
            )
            input_paths.append(path)
            df = pd.read_parquet(path)
            c.require(len(df) == meta["counts"][part], "TRAIN count mismatch")
            for row in df.to_dict("records"):
                rid = str(row["id"])
                c.require(
                    rid in small
                    and row["text"] == small[rid]["text"] == full[ix[rid]]["text"],
                    "Training identity/text/vector mismatch",
                )
                value = row[FIELDS[head]]
                target = (
                    int(value)
                    if head == "hate"
                    else (
                        meta["labels"].index(value)
                        if head == "severity"
                        else [int(label in value) for label in meta["labels"]]
                    )
                )
                rr.append(
                    {
                        "id": rid,
                        "text_hash": c.signature(row["text"]),
                        "cluster": c.signature(norm(row["text"])),
                        "vector_index": ix[rid],
                        "target": target,
                        "original_partition": part,
                        "source": row["source"],
                    }
                )
        c.require(
            len(rr) == len({r["id"] for r in rr}), "Duplicate head training identity"
        )
        input_paths.append(meta_path)
        train[head] = {
            "labels": meta["labels"],
            "kind": meta["kind"],
            "rows": sorted(rr, key=lambda x: x["id"]),
        }
    gate = train["hate"]["rows"]
    ids = [r["id"] for r in gate]
    pos = {rid: i for i, rid in enumerate(ids)}
    c.require(
        len(gate) == 147 and set(ids) == set(small), "Training label budget changed"
    )
    for head in HEADS:
        c.require(
            all((r["id"] in pos for r in train[head]["rows"])),
            "Fine row missing from gate budget",
        )
        for r in train[head]["rows"]:
            r["feature_row"] = pos[r["id"]]
            if head != "hate":
                c.require(
                    gate[pos[r["id"]]]["target"] == 1,
                    "Fine training row not gold hateful",
                )
    x = np.asarray(vectors[[r["vector_index"] for r in gate]], dtype=np.float64)
    c.require(
        np.isfinite(x).all() and np.allclose(np.linalg.norm(x, axis=1), 1, atol=0.0002),
        "Bad cached features",
    )
    y = np.array([r["target"] for r in gate])
    groups = [r["cluster"] for r in gate]
    folds = {}
    for seed in SEEDS:
        assignments = np.full(len(gate), -1, dtype=int)
        for fold, (fit, val) in enumerate(
            StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(x, y, groups)
        ):
            c.require(
                not set(np.array(groups)[fit]) & set(np.array(groups)[val]),
                "TRAIN fold leakage",
            )
            assignments[val] = fold
        c.require((assignments >= 0).all(), "Missing TRAIN fold")
        folds[str(seed)] = assignments.tolist()
    core_dev = c.rows(c.CORE / "dev_inputs.jsonl")
    selected = [(i, r) for i, r in enumerate(core_dev) if r["lang"] == "en"]
    c.require(
        not {norm(small[r]["text"]) for r in ids}
        & {norm(r["text"]) for _, r in selected},
        "TRAIN/evaluation overlap",
    )
    query = np.load(c.CORE / "retrieval_v1/query_vectors.npy", allow_pickle=False)
    scoring_path = c.CORE / "api_execution_v1/scoring_plan.json"
    endpoints = {
        k: v
        for k, v in c.read(scoring_path)["endpoints"].items()
        if k.startswith("en_")
    }
    for name in endpoints:
        input_paths.append(c.ROOT / f"data/processed/{name}.parquet")
    expected = (
        c.ROOT
        / "Thesis_Log/astra_result/evidence/core_completion_summary_20260920_v1.json"
    )
    comparators = [
        c.CORE / f"encoder_results_v5/en_{arm}_seed{s}_assembled.jsonl"
        for arm in ("kb", "full")
        for s in range(3)
    ]
    input_paths += [
        Path(__file__),
        Path(c.__file__),
        c.ROOT / "tests/training/frozen_features/test_compact.py",
        c.ROOT / "analysis/common/api.py",
        c.ROOT / "analysis/common/replay.py",
        c.ROOT / "analysis/common/scoring.py",
        c.CORE / "plan.json",
        c.CORE / "encoder_plan.json",
        c.CORE / "retrieval_v1/plan.json",
        c.CORE / "kb/small/records.jsonl",
        c.CORE / "kb/full/records.jsonl",
        c.CORE / "dev_inputs.jsonl",
        scoring_path,
        expected,
    ] + comparators
    inputs = {c.rel(p): c.sha(p) for p in input_paths}
    inputs.update(rp["generated_hashes"])
    save(BASE / "train.json", {"heads": train, "folds": folds, "gate_ids": ids})
    save_array(BASE / "train_vectors.npy", x)
    save(
        BASE / "evaluation_inputs.json",
        [
            {"key": r["key"], "item_id": r["item_id"], "source_row": i}
            for i, r in selected
        ],
    )
    save_array(
        BASE / "evaluation_vectors.npy",
        np.asarray(query[[i for i, _ in selected]], dtype=np.float64),
    )
    prepared = {
        c.rel(BASE / n): c.sha(BASE / n)
        for n in (
            "train.json",
            "train_vectors.npy",
            "evaluation_inputs.json",
            "evaluation_vectors.npy",
        )
    }
    save(
        BASE / "plan.json",
        {
            "created_utc": c.utc(),
            "status": "frozen_before_any_new_fit_or_score",
            "inputs": inputs,
            "prepared": prepared,
            "environment": environment(),
            "heads": HEADS,
            "seeds": SEEDS,
            "C_grid": CS,
            "threshold_grid": THRESHOLDS,
            "endpoints": endpoints,
            "primary_contrasts": list(PRIMARY),
            "family_size": 4,
            "bootstrap_draws": 10000,
            "training_counts": {h: len(train[h]["rows"]) for h in HEADS},
            "features": "Existing text-only normalized FP32 BGE-M3 vectors, converted to FP64 for linear fitting. No re-encoding, scaling, label text, definitions or retrieval features.",
            "budget": "The same 147 cleaned English compact KB texts, combining their old internal train/validation partitions. Annotation counts are head-specific. No additional labels; original RoBERTa fits retained separate.",
            "folds": "Five group-exclusive folds shared across every head; stratify on gate target. Three declared fold assignments 42/43/44; they are not independent datasets or neural seeds.",
            "learner": "Independent L2 logistic regressions for binary outputs, solver=liblinear, class_weight=balanced, tol=1e-8, max_iter=2000; one-vs-rest severity with argmax. Constant 0 or 1 when a fold has only one class. No probability-calibration claim.",
            "selection": "Choose C and one global threshold per binary/multilabel head by mean macro-F1 on pooled TRAIN-only out-of-fold predictions. Gate selected first; fine scores include that out-of-fold gate. Severity selects C with gated argmax. Refit on all permitted head rows.",
            "selection_labels": "Every label with at least two TRAIN positives and two negatives. Other labels still fit and evaluate; record their unsupported selection status. Gate uses both classes; no evaluation support enters tuning.",
            "ties": "Highest selection macro-F1; then smaller C, threshold nearest .5, then larger threshold. Binary gate uses >; fine labels use >=; severity argmax uses original label order.",
            "evaluation": "All five original English endpoint rosters, masks and macro labels. Four planned BGE-minus-compact-RoBERTa contrasts, 98.75% paired text-cluster intervals, 10,000 draws. The smaller type roster is sensitivity only. Old full-data encoder and RAG means contextual, not extra primary claims.",
            "exposure": "This is a supplementary recipe motivated by exposed DEV diagnostics, not an untouched confirmation. TRAIN-only tuning removes direct DEV cutoff selection; it does not erase prior research exposure.",
            "limits": "Changes both representation and fitting/selection recipe. Does not isolate a repaired RoBERTa threshold, prove universal label efficiency, add a learning curve or establish new-category adaptation. No new Qwen, human judgments or neural training.",
        },
    )
    print(
        "Compact protocol and exact features frozen:",
        {h: len(train[h]["rows"]) for h in HEADS},
        flush=True,
    )


def binary_fit(x, y, regularization, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning

    y = np.asarray(y, dtype=int)
    c.require(len(y) and set(y) <= {0, 1}, "Invalid binary training target")
    if len(set(y)) == 1:
        return {"constant": int(y[0])}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model = LogisticRegression(
            C=regularization,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            tol=1e-08,
            random_state=seed,
        ).fit(x, y)
    saved = {
        "coef": model.coef_[0].tolist(),
        "intercept": float(model.intercept_[0]),
        "iterations": int(model.n_iter_[0]),
    }
    c.require(
        np.allclose(
            binary_predict(saved, x), model.predict_proba(x)[:, 1], rtol=0, atol=1e-12
        ),
        "Serialized model differs",
    )
    return saved


def binary_predict(model, x):
    if "constant" in model:
        return np.full(len(x), float(model["constant"]))
    z = np.asarray(x) @ np.array(model["coef"]) + model["intercept"]
    return 1 / (1 + np.exp(-np.clip(z, -700, 700)))


def target_bits(head, spec):
    values = np.array([r["target"] for r in spec["rows"]], dtype=int)
    if head == "hate":
        return values[:, None]
    if head == "severity":
        return np.eye(len(spec["labels"]), dtype=int)[values]
    return values


def decisions(head, probabilities, threshold=None, gate=None):
    if head == "hate":
        positive = probabilities[:, 0] > threshold
        return np.column_stack((~positive, positive)).astype(int)
    if head == "severity":
        result = np.eye(probabilities.shape[1], dtype=int)[probabilities.argmax(axis=1)]
    else:
        result = (probabilities >= threshold).astype(int)
    if gate is not None:
        result[~np.asarray(gate, dtype=bool)] = 0
    return result


def macro(y, pred, indices):
    tp = (y * pred).sum(axis=0)
    fp = ((1 - y) * pred).sum(axis=0)
    fn = (y * (1 - pred)).sum(axis=0)
    den = 2 * tp + fp + fn
    return float(
        np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0)[
            indices
        ].mean()
    )


def choose(candidates):
    return min(
        candidates,
        key=lambda r: (
            -r["score"],
            r["C"],
            abs((r["threshold"] if r["threshold"] is not None else 0.5) - 0.5),
            -(r["threshold"] or 0),
        ),
    )


def fit():
    plan = validate()
    data = c.read(BASE / "train.json")
    x = np.load(BASE / "train_vectors.npy", allow_pickle=False)
    for seed in plan["seeds"]:
        path = BASE / f"fit_seed{seed}.json"
        if path.exists():
            c.require(
                c.read(path)["plan_sha256"] == c.sha(BASE / "plan.json"),
                "Wrong saved fit",
            )
            print("Verified saved fit", seed, flush=True)
            continue
        start = time.perf_counter()
        fold = np.array(data["folds"][str(seed)])
        heads = {}
        gate_oof = None
        fits = 0
        for head in HEADS:
            spec = data["heads"][head]
            ix = np.array([r["feature_row"] for r in spec["rows"]])
            yy = target_bits(head, spec)
            truth = np.column_stack((1 - yy[:, 0], yy[:, 0])) if head == "hate" else yy
            selection_labels = np.flatnonzero(
                (truth.sum(axis=0) >= 2) & ((1 - truth).sum(axis=0) >= 2)
            )
            c.require(len(selection_labels), "No TRAIN-supported selection label")
            candidates = []
            cached = {}
            for strength in plan["C_grid"]:
                prob = np.full(yy.shape, np.nan, dtype=float)
                for k in range(5):
                    train_mask = fold[ix] != k
                    held = ~train_mask
                    c.require(train_mask.any() and held.any(), "Empty compact fold")
                    for label in range(yy.shape[1]):
                        model = binary_fit(
                            x[ix[train_mask]], yy[train_mask, label], strength, seed
                        )
                        prob[held, label] = binary_predict(model, x[ix[held]])
                        fits += "constant" not in model
                c.require(np.isfinite(prob).all(), "OOF predictions incomplete")
                cached[str(strength)] = prob
                for cutoff in [None] if head == "severity" else plan["threshold_grid"]:
                    pred = decisions(
                        head, prob, cutoff, None if head == "hate" else gate_oof[ix]
                    )
                    candidates.append(
                        {
                            "C": strength,
                            "threshold": cutoff,
                            "score": macro(truth, pred, selection_labels),
                        }
                    )
            selected = choose(candidates)
            oof = cached[str(selected["C"])]
            if head == "hate":
                gate_oof = decisions(head, oof, selected["threshold"])[:, 1].astype(
                    bool
                )
            models = []
            for label in range(yy.shape[1]):
                model = binary_fit(x[ix], yy[:, label], selected["C"], seed)
                models.append(model)
                fits += "constant" not in model
            heads[head] = {
                "selected": selected,
                "candidates": candidates,
                "models": models,
                "labels": spec["labels"],
                "selection_indices": selection_labels.tolist(),
                "training_ids": [r["id"] for r in spec["rows"]],
                "oof_probabilities": oof.tolist(),
            }
            print(
                "Fitted TRAIN-only",
                seed,
                head,
                "C",
                selected["C"],
                "threshold",
                selected["threshold"],
                flush=True,
            )
        save(
            path,
            {
                "plan_sha256": c.sha(BASE / "plan.json"),
                "seed": seed,
                "heads": heads,
                "nonconstant_logistic_fits": fits,
                "fit_seconds": time.perf_counter() - start,
                "selection_scores_are_not_unbiased_validation_estimates": True,
                "Qwen_calls": 0,
            },
        )
    save(
        BASE / "fit_complete.json",
        {
            "plan_sha256": c.sha(BASE / "plan.json"),
            "outputs": {
                c.rel(BASE / f"fit_seed{s}.json"): c.sha(BASE / f"fit_seed{s}.json")
                for s in plan["seeds"]
            },
        },
    )


def predict():
    plan = validate()
    c.verify(c.read(BASE / "fit_complete.json")["outputs"])
    inputs = c.read(BASE / "evaluation_inputs.json")
    x = np.load(BASE / "evaluation_vectors.npy", allow_pickle=False)
    outputs = {}
    for seed in plan["seeds"]:
        fit_record = c.read(BASE / f"fit_seed{seed}.json")
        values = {}
        probabilities = {}
        for head in HEADS:
            spec = fit_record["heads"][head]
            prob = np.column_stack([binary_predict(m, x) for m in spec["models"]])
            probabilities[head] = prob
            values[head] = decisions(
                head,
                prob,
                spec["selected"]["threshold"],
                None if head == "hate" else values["hate"][:, 1],
            )
        rows = []
        for i, row in enumerate(inputs):
            result = {
                "hate": bool(values["hate"][i, 1]),
                "hate_type": [],
                "target_group": [],
                "severity": None,
                "legal": [],
            }
            for head in HEADS[1:]:
                selected = [
                    l
                    for j, l in enumerate(fit_record["heads"][head]["labels"])
                    if values[head][i, j]
                ]
                result[head] = (
                    (selected[0] if selected else None)
                    if head == "severity"
                    else selected
                )
            rows.append(
                {
                    **row,
                    "seed": seed,
                    "result": result,
                    "ungated_probabilities": {
                        h: probabilities[h][i].tolist() for h in HEADS
                    },
                }
            )
        dest = BASE / f"predictions_seed{seed}.json"
        save(dest, rows)
        outputs[c.rel(dest)] = c.sha(dest)
    save(
        BASE / "predictions_complete.json",
        {
            "plan_sha256": c.sha(BASE / "plan.json"),
            "rows_per_seed": len(inputs),
            "outputs": outputs,
        },
    )
    print(
        "Predicted from frozen linear weights:",
        len(inputs),
        "English inputs per fit; no new embeddings.",
        flush=True,
    )


def score():
    plan = validate()
    c.verify(c.read(BASE / "predictions_complete.json")["outputs"])
    from analysis.common import replay as scoring

    summary = c.read(
        c.ROOT
        / "Thesis_Log/astra_result/evidence/core_completion_summary_20260920_v1.json"
    )
    new = {
        s: {
            r["item_id"]: r["result"]
            for r in c.read(BASE / f"predictions_seed{s}.json")
        }
        for s in plan["seeds"]
    }
    old = {
        arm: [
            {
                r["item_id"]: r["result"]
                for r in c.rows(
                    c.CORE / f"encoder_results_v5/en_{arm}_seed{s}_assembled.jsonl"
                )
            }
            for s in range(3)
        ]
        for arm in ("kb", "full")
    }
    result = {}
    reproduced = 0
    for name, ep in plan["endpoints"].items():
        gold = scoring.gold_rows(name)
        ids = [str(r["id"]) for r in gold]
        contributions = {}
        runs = {}
        for arm, maps in [
            ("bge", [new[s] for s in plan["seeds"]]),
            ("kb", old["kb"]),
            ("full", old["full"]),
        ]:
            contributions[arm] = [
                scoring.contributions(gold, {rid: m[rid] for rid in ids}, ep)
                for m in maps
            ]
            runs[arm] = [scoring.summarize(v) for v in contributions[arm]]
            if arm != "bge":
                expected = summary["endpoints"][name]["conditions"][
                    "encoder_" + ("kb" if arm == "kb" else "full")
                ]["values"]
                c.require(
                    all(
                        (
                            abs(r["macro_f1_valid_only"] - e) < 1e-12
                            for r, e in zip(runs[arm], expected)
                        )
                    ),
                    "Old score failed to reproduce",
                )
                reproduced += len(expected)
        paired = None
        if name in PRIMARY:
            cluster = [
                scoring.normalized(r["text"])
                for r in gold
                if str(r["id"]) in ep["scorable_ids"]
            ]
            paired = scoring.bootstrap(
                contributions["bge"],
                contributions["kb"],
                cluster,
                seed=20261008 + list(PRIMARY).index(name),
                draws=10000,
                family_size=4,
            )
        result[name] = {
            "endpoint": ep,
            "runs": runs,
            "means": {a: scoring.means(v) for a, v in runs.items()},
            "bge_minus_compact_roberta": paired,
            "historical_rag_context": {
                a: summary["endpoints"][name]["conditions"][a]
                for a in ("rag_small", "rag_full")
            },
        }
    save(
        BASE / "results.json",
        {
            "created_utc": c.utc(),
            "plan_sha256": c.sha(BASE / "plan.json"),
            "endpoints": result,
            "original_encoder_scores_reproduced": reproduced,
            "Qwen_calls": 0,
            "new_embeddings": 0,
            "CUDA_used": False,
            "status": "supplementary_exposed_dev_results",
        },
    )
    print("All original encoder scores reproduced:", reproduced, flush=True)
    for name, x in result.items():
        print(name, {arm: r["mean"] for arm, r in x["means"].items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=["prepare", "validate", "fit", "predict", "score"]
    )
    args = parser.parse_args()
    with c.offline():
        globals()[args.mode]()


if __name__ == "__main__":
    main()

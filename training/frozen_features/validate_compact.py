import os

os.environ.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
from pathlib import Path
import warnings
import pyarrow.dataset, pyarrow.parquet
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.exceptions import ConvergenceWarning
from analysis.common import study as c
from analysis.common.api import save_new

BASE = c.ROOT / "experiments/astra/compact_bge_20261008_v1"


def main():
    with c.offline():
        plan = c.read(BASE / "plan.json")
        c.verify(plan["inputs"])
        c.verify(plan["prepared"])
        training = c.read(BASE / "train.json")
        x = np.load(BASE / "train_vectors.npy", allow_pickle=False)
        ex = np.load(BASE / "evaluation_vectors.npy", allow_pickle=False)
        full = c.rows(c.CORE / "kb/full/records.jsonl")
        vectors = np.load(
            c.CORE / "retrieval_v1/full_vectors.npy", allow_pickle=False, mmap_mode="r"
        )
        np.testing.assert_array_equal(
            x,
            vectors[
                [r["vector_index"] for r in training["heads"]["hate"]["rows"]]
            ].astype(float),
        )
        fields = {
            "hate": "gate",
            "hate_type": "hate_types",
            "target_group": "target_groups",
            "severity": "severity",
        }
        for head, spec in training["heads"].items():
            source = pd.concat(
                [
                    pd.read_parquet(c.CORE / f"encoder_data/kb/{head}_{part}.parquet")
                    for part in ("train", "val")
                ]
            ).set_index("id")
            assert set(source.index) == {r["id"] for r in spec["rows"]}
            for r in spec["rows"]:
                source_row = source.loc[r["id"]]
                v = source_row[fields[head]]
                target = (
                    int(v)
                    if head == "hate"
                    else (
                        spec["labels"].index(v)
                        if head == "severity"
                        else [int(l in v) for l in spec["labels"]]
                    )
                )
                assert (
                    target == r["target"]
                    and source_row["text"] == full[r["vector_index"]]["text"]
                )
        cv_checks = prediction_checks = score_checks = 0

        def probability(fit_x, fit_y, test_x, C, seed):
            if len(set(fit_y)) == 1:
                return np.full(len(test_x), float(fit_y[0]))
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                model = LogisticRegression(
                    C=C,
                    class_weight="balanced",
                    solver="liblinear",
                    tol=1e-08,
                    max_iter=2000,
                    random_state=seed,
                ).fit(fit_x, fit_y)
            return model.predict_proba(test_x)[:, 1]

        def classify(head, p, t, gate):
            if head == "hate":
                return np.column_stack((p[:, 0] <= t, p[:, 0] > t)).astype(int)
            pred = (
                np.eye(p.shape[1], dtype=int)[np.argmax(p, axis=1)]
                if head == "severity"
                else (p >= t).astype(int)
            )
            pred[~gate] = 0
            return pred

        for seed in plan["seeds"]:
            record = c.read(BASE / f"fit_seed{seed}.json")
            predictions = c.read(BASE / f"predictions_seed{seed}.json")
            folds = np.array(training["folds"][str(seed)])
            gate = None
            eval_gate = None
            for head in plan["heads"]:
                spec = training["heads"][head]
                saved = record["heads"][head]
                ix = np.array([r["feature_row"] for r in spec["rows"]])
                raw = np.array([r["target"] for r in spec["rows"]])
                y = (
                    raw[:, None]
                    if head == "hate"
                    else (
                        np.eye(len(spec["labels"]), dtype=int)[raw]
                        if head == "severity"
                        else raw
                    )
                )
                truth = np.column_stack((1 - raw, raw)) if head == "hate" else y
                eligible = np.flatnonzero(
                    (truth.sum(axis=0) >= 2) & ((1 - truth).sum(axis=0) >= 2)
                )
                assert eligible.tolist() == saved["selection_indices"]
                recomputed = []
                for C in plan["C_grid"]:
                    p = np.zeros(y.shape, dtype=float)
                    for fold in range(5):
                        train = folds[ix] != fold
                        val = ~train
                        assert not set(ix[train]) & set(ix[val])
                        for label in range(y.shape[1]):
                            p[val, label] = probability(
                                x[ix[train]], y[train, label], x[ix[val]], C, seed
                            )
                    if C == saved["selected"]["C"]:
                        np.testing.assert_allclose(
                            p, saved["oof_probabilities"], atol=1e-12, rtol=0
                        )
                        selected_p = p.copy()
                    for t in [None] if head == "severity" else plan["threshold_grid"]:
                        decision = classify(
                            head, p, t, None if head == "hate" else gate[ix]
                        )
                        f = float(
                            np.mean(
                                [
                                    f1_score(
                                        truth[:, j], decision[:, j], zero_division=0
                                    )
                                    for j in eligible
                                ]
                            )
                        )
                        recomputed.append({"C": C, "threshold": t, "score": f})
                        cv_checks += 1
                assert len(recomputed) == len(saved["candidates"])
                for left, right in zip(recomputed, saved["candidates"]):
                    assert (
                        left["C"] == right["C"]
                        and left["threshold"] == right["threshold"]
                        and (abs(left["score"] - right["score"]) < 1e-12)
                    )
                selected = min(
                    recomputed,
                    key=lambda r: (
                        -r["score"],
                        r["C"],
                        abs(
                            (r["threshold"] if r["threshold"] is not None else 0.5)
                            - 0.5
                        ),
                        -(r["threshold"] or 0),
                    ),
                )
                assert (
                    selected["C"] == saved["selected"]["C"]
                    and selected["threshold"] == saved["selected"]["threshold"]
                )
                predp = np.column_stack(
                    [
                        probability(x[ix], y[:, j], ex, selected["C"], seed)
                        for j in range(y.shape[1])
                    ]
                )
                np.testing.assert_allclose(
                    predp,
                    [r["ungated_probabilities"][head] for r in predictions],
                    atol=1e-12,
                    rtol=0,
                )
                output = classify(head, predp, selected["threshold"], eval_gate)
                for i, row in enumerate(predictions):
                    got = row["result"][head]
                    expected = (
                        bool(output[i, 1])
                        if head == "hate"
                        else [l for j, l in enumerate(spec["labels"]) if output[i, j]]
                    )
                    if head == "severity":
                        expected = expected[0] if expected else None
                    assert got == expected
                    prediction_checks += 1
                if head == "hate":
                    gate = classify(head, selected_p, selected["threshold"], None)[
                        :, 1
                    ].astype(bool)
                    eval_gate = output[:, 1].astype(bool)
        results = c.read(BASE / "results.json")
        for endpoint, r in results["endpoints"].items():
            for arm, runs in r["runs"].items():
                for run in runs:
                    labels = run["per_label"]
                    mac = run["macro_labels"]
                    values = []
                    for lab in mac:
                        v = labels[lab]
                        den = 2 * v["tp"] + v["fp"] + v["fn"]
                        values.append(2 * v["tp"] / den if den else 0)
                    assert (
                        abs(sum(values) / len(values) - run["macro_f1_valid_only"])
                        < 1e-12
                    )
                    score_checks += 1
        paths = [
            BASE / "plan.json",
            BASE / "fit_complete.json",
            BASE / "predictions_complete.json",
            BASE / "results.json",
            Path(__file__),
        ]
        hashes = {c.rel(p): c.sha(p) for p in paths}
        hashes.update(c.read(BASE / "fit_complete.json")["outputs"])
        hashes.update(c.read(BASE / "predictions_complete.json")["outputs"])
        c.verify(hashes)
        c.verify(plan["inputs"])
        c.verify(plan["prepared"])
        out = {
            "created_utc": c.utc(),
            "status": "independent_training_selection_prediction_and_score_checks_pass",
            "candidate_score_checks": cv_checks,
            "all_head_prediction_checks": prediction_checks,
            "score_arithmetic_checks": score_checks,
            "new_linear_fits_for_verification_only": True,
            "actual_Qwen_calls": 0,
            "CUDA_used": False,
            "verified_hashes": hashes,
        }
        save_new(BASE / "verification.json", out)
        print({k: v for k, v in out.items() if k != "verified_hashes"})


if __name__ == "__main__":
    main()

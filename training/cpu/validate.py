import os

os.environ.update(
    CUDA_VISIBLE_DEVICES="",
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
)
import argparse
from collections import Counter
import gc
from pathlib import Path
import sys
import time
import warnings
from unittest.mock import patch
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
from training.cpu import memory as memory
from training.cpu import common as c

ROOT = c.ROOT
BASE = memory.BASE
OUT = ROOT / "experiments/astra/sq2_cpu_review_20261012_v1"
PRIMARY = [
    "en_test/hate",
    "en_test/hate_type",
    "larger_dev/target_group",
    "larger_dev/severity",
]
EXPECTED_PLAN = "3e18117fce09e114d8ff2cb49b40e2aaa13c0ac3d6c89e550030ce92da3d63a1"


def save(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = c.encoded(value)
    if path.exists():
        c.require(
            path.read_bytes() == payload, "Preserve previous review artifact: " + name
        )
    else:
        with path.open("xb") as stream:
            stream.write(payload)


def save_arrays(name, values):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            c.require(
                set(z.files) == set(values)
                and all(np.array_equal(z[k], v) for k, v in values.items()),
                "Review array differs",
            )
    else:
        with path.open("xb") as stream:
            np.savez_compressed(stream, **values)


def receipt(directory):
    path = Path(directory) / "complete.json"
    rec = c.read(path)
    c.require(rec["identity"]["plan_sha256"] == EXPECTED_PLAN, "Unexpected job plan")
    for name, digest in rec["outputs"].items():
        target = (path.parent / name).resolve()
        c.require(
            target.is_relative_to(path.parent.resolve()) and c.sha(target) == digest,
            "Changed job artifact",
        )
    return rec, path.parent / rec["attempt"]


def freeze():
    with memory.execution_context():
        plan, digest = memory.readiness()
        memory.run.verify_complete(digest)
    c.require(digest == EXPECTED_PLAN, "Wrong completed study")
    save(
        "analysis_plan.json",
        dict(
            status="fixed_before_new_evaluation_scores",
            study_plan_sha256=digest,
            primary=PRIMARY,
            left="full_bge_matched_selected",
            right="full_neural_selected",
            left_fits=1,
            right_seeds=[0, 1, 2],
            bootstrap=dict(
                draws=10000,
                seed=20261012,
                confidence=0.9875,
                resampling="Shared whole normalized-text clusters; mean of three per-seed neural F1 values. No ensemble or duplicated BGE fits.",
            ),
            sensitivity="Source-stratified paired whole-text clusters at 95%; exclude original-core texts descriptively. HateCheck uses its frozen functional/template clusters.",
            supporting="Default/selected policies, all-permitted refits, compact/full TF-IDF, compact BGE/tuned RoBERTa context, gate and exact-set diagnostics, labels, sources, ranking.",
            safeguards="No scientific recipe selected from evaluation. All populations already exposed. Missing annotations remain excluded; fixed original macro-label masks.",
            audit="All 292 linear candidate/final tasks refit from independently checked feature matrices; no new scientific search or neural inference.",
            Qwen_calls=0,
            CUDA_used=False,
        ),
    )
    print("Independent review plan frozen.", flush=True)


def inventory():
    c.require(
        c.read(OUT / "analysis_plan.json")["study_plan_sha256"]
        == c.sha(BASE / "plan.json"),
        "Plan changed",
    )
    bindings = {}
    types = Counter()
    attempts = []
    total_bytes = 0
    plan = c.read(BASE / "plan.json")
    for field in ("inputs", "prepared", "active_source_hashes"):
        for name, digest in plan[field].items():
            c.require(
                name not in bindings or bindings[name] == digest,
                "Conflicting source hash",
            )
            bindings[name] = digest
    for folder in ("groups", "pipelines", "neural_cutoffs", "runtime_pilot"):
        for path in sorted((BASE / folder).rglob("complete.json")):
            rec, attempt = receipt(path.parent)
            types[rec["identity"]["kind"]] += 1
            bindings[c.rel(path)] = c.sha(path)
            for name, digest in rec["outputs"].items():
                bindings[c.rel(path.parent / name)] = digest
            if (path.parent / "attempts").is_dir():
                for ap in sorted((path.parent / "attempts").iterdir()):
                    if ap.is_dir():
                        attempts.append(dict(path=c.rel(ap), selected=ap == attempt))
    for name, digest in bindings.items():
        c.require(c.sha(ROOT / name) == digest, "Frozen source changed: " + name)
        total_bytes += (ROOT / name).stat().st_size
    for kind, expected in [
        ("head_complete", 20),
        ("pipeline", 7),
        ("retained_neural_cutoff", 12),
        ("candidate", 272),
        ("all_permitted_refit", 8),
        ("final_compact", 12),
        ("features", 88),
        ("inference", 28),
    ]:
        c.require(
            types[kind] == expected,
            "Unexpected task count " + kind + ": " + str(types[kind]),
        )
    execution = [
        c.read(p) | dict(path=c.rel(p))
        for p in sorted((BASE / "execution").glob("*/*/resources.json"))
    ]
    c.require(
        all(v["exit_code"] == 0 and v["stop_reason"] is None for v in execution),
        "Execution stop/error requires review",
    )
    texts = {r["id"]: r["text"] for r in c.rows(BASE / "data/train_texts.jsonl")}
    inputs = c.read(BASE / "data/evaluation.json")
    scopes = c.read(BASE / "data/scopes.json")
    c.require(
        len(texts) == 55753 and len(inputs) == len({r["key"] for r in inputs}) == 19082,
        "Input counts",
    )
    c.require(
        not {c.cluster(v) for v in texts.values()}
        & {c.cluster(r["text"]) for r in inputs},
        "TRAIN/evaluation normalized overlap",
    )
    for name, rows in scopes.items():
        c.require(
            len({r["source_id"] for r in rows}) == len(rows), "Duplicate scope ID"
        )
        c.require(
            all(inputs[r["position"]]["key"] == r["key"] for r in rows),
            "Scope index mismatch",
        )
    save(
        "inventory.json",
        dict(
            status="pass",
            source_plan_sha256=EXPECTED_PLAN,
            files=len(bindings),
            bytes=total_bytes,
            bindings=bindings,
            task_counts=dict(types),
            attempts=attempts,
            unselected_attempts=sum(not r["selected"] for r in attempts),
            execution=execution,
            normalized_TRAIN_evaluation_overlap=0,
            scope_rows={k: len(v) for k, v in scopes.items()},
            Qwen_calls=0,
        ),
    )
    print("Inventory passed:", len(bindings), "bound files;", dict(types), flush=True)


def target(head, spec, rows):
    y = np.asarray([r["target"] for r in rows], dtype=np.int64)
    if head == "hate":
        return y.reshape(-1, 1)
    if head == "severity":
        return np.eye(len(spec["labels"]), dtype=np.int64)[y]
    c.require(y.shape == (len(rows), len(spec["labels"])), "Malformed target matrix")
    return y


def bits(head, prob, cutoff, gate=None):
    if head == "hate":
        positive = prob[:, 0] > cutoff
        value = np.column_stack((~positive, positive))
    elif head == "severity":
        value = np.eye(prob.shape[1], dtype=bool)[np.argmax(prob, axis=1)]
    else:
        value = prob >= cutoff
    if gate is not None:
        value[~np.asarray(gate, dtype=bool)] = False
    return value


def select(head, yy, probabilities, gate=None, compact=False):
    y = np.column_stack((1 - yy[:, 0], yy[:, 0])) if head == "hate" else yy
    support = y.sum(0)
    ix = (
        np.flatnonzero((support >= 2) & ((len(y) - support) >= 2))
        if compact
        else np.arange(y.shape[1])
    )
    c.require(len(ix) > 0, "Empty selection label set")
    values = []
    for strength, prob in probabilities.items():
        for cutoff in (
            [None] if head == "severity" else [i / 10 for i in range(1, 10)]
        ):
            x = bits(head, prob, cutoff, gate)
            tp = np.logical_and(y, x).sum(0)
            fp = np.logical_and(~y.astype(bool), x).sum(0)
            fn = np.logical_and(y, ~x).sum(0)
            den = 2 * tp + fp + fn
            scores = np.divide(2 * tp, den, out=np.zeros(len(tp)), where=den != 0)
            score = float(scores[ix].mean())
            c.require(
                abs(score - f1_score(y, x, average=None, zero_division=0)[ix].mean())
                < 1e-12,
                "Independent selection score mismatch",
            )
            values.append(dict(C=float(strength), threshold=cutoff, score=score))
    choice = sorted(
        values,
        key=lambda v: (
            -v["score"],
            v["C"],
            abs((v["threshold"] if v["threshold"] is not None else 0.5) - 0.5),
            -(v["threshold"] or 0),
        ),
    )[0]
    return dict(
        selected=choice,
        candidates=values,
        selection_indices=ix.tolist(),
        positive_support=support.tolist(),
        negative_support=(len(y) - support).tolist(),
    )


def check_selection(expected, actual):
    for field in (
        "selected",
        "candidates",
        "selection_indices",
        "positive_support",
        "negative_support",
    ):
        c.require(
            expected[field] == actual[field],
            "Independent TRAIN selection differs: " + field,
        )


def independent_features(texts, vocabulary=None, idf=None):
    channels = []
    states = {}
    vectors = {}
    for kind in ("word", "char"):
        supplied = vocabulary[kind]["vocabulary"] if vocabulary is not None else None
        if supplied is not None and not supplied:
            channels.append(sparse.csr_matrix((len(texts), 0), dtype=np.float64))
            continue
        v = TfidfVectorizer(
            analyzer="word" if kind == "word" else "char_wb",
            ngram_range=(1, 2) if kind == "word" else (3, 5),
            max_features=100000 if kind == "word" else 150000,
            lowercase=True,
            strip_accents=None,
            stop_words=None,
            min_df=1,
            max_df=1.0,
            sublinear_tf=True,
            smooth_idf=True,
            norm="l2",
            dtype=np.float64,
            vocabulary=supplied,
        )
        if vocabulary is not None:
            v.idf_ = idf[kind]
            x = v.transform(texts)
        else:
            try:
                x = v.fit_transform(texts)
            except ValueError as exc:
                if "empty vocabulary" not in str(exc):
                    raise
                x = sparse.csr_matrix((len(texts), 0), dtype=np.float64)
                v = None
            states[kind] = dict(
                vocabulary=(
                    {}
                    if v is None
                    else {str(k): int(j) for k, j in v.vocabulary_.items()}
                )
            )
            vectors[kind] = np.empty(0) if v is None else np.asarray(v.idf_)
        channels.append(x)
    x = normalize(sparse.hstack(channels, format="csr"), norm="l2", copy=False)
    return x, states, vectors


def matrix_check(expected, stored):
    c.require(
        expected.shape == stored.shape and stored.dtype == np.float64,
        "Feature shape/precision differs",
    )
    delta = (
        (expected - stored).data
        if sparse.issparse(expected)
        else np.asarray(expected) - stored
    )
    maximum = float(np.max(np.abs(delta), initial=0))
    c.require(maximum < 1e-12, "Features do not reconstruct from permitted texts")
    return maximum


def check_features(directory, representation, fit, held, texts):
    rec, path = receipt(directory)
    identity = rec["identity"]
    membership = c.read(path / "memberships.json")
    c.require(
        identity["fit_hash"] == c.signature(fit)
        and identity["held_hash"] == c.signature(held),
        "Feature partition identity differs",
    )
    c.require(
        membership
        == dict(fit_ids=[r["id"] for r in fit], held_ids=[r["id"] for r in held]),
        "Feature membership/order differs",
    )
    c.require(
        not {c.cluster(texts[r["id"]]) for r in fit}
        & {c.cluster(texts[r["id"]]) for r in held},
        "TRAIN/held text leakage",
    )
    maximum = 0
    if representation == "bge":
        cache = np.load(
            c.CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r", allow_pickle=False
        )
        for part, rows in [("fit", fit), ("held", held)]:
            stored = np.load(path / (part + ".npy"), mmap_mode="r", allow_pickle=False)
            c.require(
                stored.shape == (len(rows), 1024) and stored.dtype == np.float64,
                "BGE shape/precision",
            )
            for start in range(0, len(rows), 512):
                expected = np.asarray(
                    cache[[r["vector_index"] for r in rows[start : start + 512]]],
                    dtype=np.float64,
                )
                maximum = max(
                    maximum, matrix_check(expected, stored[start : start + 512])
                )
            del stored
        del cache
    else:
        expected, state, idf = independent_features([texts[r["id"]] for r in fit])
        saved_state = c.read(path / "vocabulary.json")
        saved_idf = c.load_arrays(path / "idf.npz")
        c.require(
            state == saved_state
            and all(np.array_equal(idf[k], saved_idf[k]) for k in idf),
            "Vocabulary/IDF not TRAIN fitted",
        )
        stored = sparse.load_npz(path / "fit.npz")
        maximum = matrix_check(expected, stored)
        width = stored.shape[1]
        del expected, stored, state, idf
        expected = (
            independent_features(
                [texts[r["id"]] for r in held], saved_state, saved_idf
            )[0]
            if held
            else sparse.csr_matrix((0, width), dtype=np.float64)
        )
        stored = sparse.load_npz(path / "held.npz")
        maximum = max(maximum, matrix_check(expected, stored))
        del expected, stored, saved_state, saved_idf
    gc.collect()
    return path, dict(
        path=c.rel(path),
        fit_rows=len(fit),
        held_rows=len(held),
        max_abs_difference=maximum,
        feature_seconds=rec["detail"]["feature_seconds"],
        resource_after=rec["detail"]["resource_after"],
    )


def independent_probability(model, x):
    columns = []
    for j, constant in enumerate(model["constants"]):
        if constant >= 0:
            columns.append(np.full(x.shape[0], constant, dtype=float))
        else:
            z = np.asarray(x @ model["coef"][j]).reshape(-1) + model["intercept"][j]
            columns.append(1 / (1 + np.exp(-np.clip(z, -700, 700))))
    return np.column_stack(columns)


def check_fit(directory, feature_path, representation, yy, strength, seed):
    rec, path = receipt(directory)
    ident = rec["identity"]
    model = c.load_arrays(path / "model.npz")
    c.require(
        ident["feature_receipt_sha256"]
        == c.sha(feature_path.parent.parent / "complete.json")
        and ident["target_sha256"] == c.signature(yy.tolist())
        and ident["C"] == strength
        and ident["seed"] == seed,
        "Fit provenance differs",
    )
    x = (
        np.load(feature_path / "fit.npy", allow_pickle=False)
        if representation == "bge"
        else sparse.load_npz(feature_path / "fit.npz")
    )
    held = (
        np.load(feature_path / "held.npy", allow_pickle=False)
        if representation == "bge"
        else sparse.load_npz(feature_path / "held.npz")
    )
    c.require(
        x.shape[0] == len(yy) and model["coef"].shape == (yy.shape[1], x.shape[1]),
        "Fitted shape differs",
    )
    maximum = 0.0
    solved = 0
    began = time.perf_counter()
    for j in range(yy.shape[1]):
        values, counts = np.unique(yy[:, j], return_counts=True)
        if len(values) == 1:
            c.require(
                model["constants"][j] == values[0]
                and np.all(model["coef"][j] == 0)
                and model["intercept"][j] == 0
                and model["iterations"][j] == 0,
                "Constant column differs",
            )
        else:
            c.require(
                model["constants"][j] == -1,
                "Nonconstant TRAIN target recorded constant",
            )
            weights = {
                int(value): len(yy) / (2 * int(n)) for value, n in zip(values, counts)
            }
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                fitted = LogisticRegression(
                    C=strength,
                    solver="liblinear",
                    class_weight=weights,
                    fit_intercept=True,
                    tol=1e-8,
                    max_iter=2000,
                    random_state=seed,
                ).fit(x, yy[:, j])
            error = max(
                float(np.max(np.abs(model["coef"][j] - fitted.coef_[0]), initial=0)),
                abs(float(model["intercept"][j] - fitted.intercept_[0])),
            )
            c.require(
                error < 1e-10 and model["iterations"][j] == fitted.n_iter_[0],
                "Independent refit differs",
            )
            maximum = max(maximum, error)
            solved += 1
    prob = independent_probability(model, held)
    saved = np.load(path / "held.npy", allow_pickle=False)
    c.require(
        np.allclose(prob, saved, rtol=0, atol=1e-12),
        "Held probabilities differ from saved coefficients",
    )
    c.require(
        rec["detail"]["positive_support"] == yy.sum(0).tolist()
        and rec["detail"]["iterations"] == model["iterations"].tolist(),
        "Fit metadata differs",
    )
    for cutoff in c.CUTOFFS:
        c.require(
            np.array_equal(prob >= cutoff, saved >= cutoff)
            and np.array_equal(prob > cutoff, saved > cutoff),
            "Tiny probability drift changes held cutoff",
        )
    c.require(np.array_equal(prob.argmax(1), saved.argmax(1)), "Held argmax differs")
    info = dict(
        path=c.rel(path),
        kind=ident["kind"],
        C=strength,
        seed=seed,
        fit_rows=len(yy),
        dimensions=x.shape[1],
        binary_solves=solved,
        constant_columns=yy.shape[1] - solved,
        iterations=model["iterations"].tolist(),
        coefficient_max_abs_difference=maximum,
        held_probability_max_abs_difference=float(
            np.max(np.abs(prob - saved), initial=0)
        ),
        original_fit_seconds=rec["detail"]["fit_seconds"],
        review_refit_seconds=time.perf_counter() - began,
        resource_after=rec["detail"]["resource_after"],
    )
    return saved, info


def audit_group(stage, head, seed):
    name = f"{stage}_s{seed}/{head}"
    output = OUT / "groups" / name / "audit.json"
    c.require(c.read(OUT / "inventory.json")["status"] == "pass", "Inventory required")
    if output.exists():
        c.require(
            c.read(output)["source_sha256"] == c.sha(Path(__file__)),
            "Review source changed after group audit",
        )
        print("Verified prior independent group", name, flush=True)
        return
    c.resource_guard(3.5, 5.0)
    directory = BASE / "groups" / name
    rec, resultpath = receipt(directory / "result")
    saved_result = c.read(resultpath / "head.json")
    texts = {r["id"]: r["text"] for r in c.rows(BASE / "data/train_texts.jsonl")}
    rep = "bge" if stage == "full_bge" else "tfidf"
    features = []
    fits = []
    gate = None
    probabilities = {}
    if stage == "compact_tfidf":
        data = c.read(BASE / "data/compact.json")
        spec = data["heads"][head]
        rows = spec["rows"]
        yy = target(head, spec, rows)
        assigned = np.asarray(data["folds"][str(seed)])[
            [r["feature_row"] for r in rows]
        ]
        probabilities = {v: np.full(yy.shape, np.nan) for v in c.CS}
        for fold in range(5):
            fi = np.flatnonzero(assigned != fold)
            he = np.flatnonzero(assigned == fold)
            c.require(len(fi) > 0, "Empty fitting fold")
            if not len(he):
                continue
            fp, info = check_features(
                directory / f"features/fold{fold}",
                rep,
                [rows[i] for i in fi],
                [rows[i] for i in he],
                texts,
            )
            features.append(info)
            for strength in c.CS:
                p, info = check_fit(
                    directory / f"search/fold{fold}/C{strength:g}",
                    fp,
                    rep,
                    yy[fi],
                    strength,
                    seed,
                )
                probabilities[strength][he] = p
                fits.append(info)
        if head != "hate":
            gr = c.read(OUT / f"groups/{stage}_s{seed}/hate/audit.json")
            gp = np.load(
                BASE / f"groups/{stage}_s{seed}/hate/selected_oof.npy",
                allow_pickle=False,
            )
            gate = bits("hate", gp, gr["selection"]["selected"]["threshold"])[:, 1][
                [r["feature_row"] for r in rows]
            ]
        choice = select(head, yy, probabilities, gate, True)
        c.require(
            all(np.isfinite(p).all() for p in probabilities.values()), "OOF holes"
        )
        c.require(
            np.array_equal(
                probabilities[choice["selected"]["C"]],
                np.load(directory / "selected_oof.npy", allow_pickle=False),
            ),
            "OOF assembly differs",
        )
        fp, info = check_features(directory / "features/final", rep, rows, [], texts)
        features.append(info)
        _, info = check_fit(
            directory / "final/compact", fp, rep, yy, choice["selected"]["C"], seed
        )
        fits.append(info)
    else:
        spec = c.read(BASE / "data/full_heads.json")[head]
        fit, held = spec["train"], spec["val"]
        yy = target(head, spec, fit)
        yv = target(head, spec, held)
        fp, info = check_features(directory / "features/matched", rep, fit, held, texts)
        features.append(info)
        for strength in c.CS:
            prob, info = check_fit(
                directory / f"search/C{strength:g}", fp, rep, yy, strength, seed
            )
            probabilities[strength] = prob
            fits.append(info)
        choice = select(head, yv, probabilities)
        fp, info = check_features(
            directory / "features/all_permitted", rep, fit + held, [], texts
        )
        features.append(info)
        _, info = check_fit(
            directory / "final/all_permitted",
            fp,
            rep,
            target(head, spec, fit + held),
            choice["selected"]["C"],
            seed,
        )
        fits.append(info)
    check_selection(saved_result["selection"], choice)
    c.require(
        c.read(directory / "selection.json")["selection"] == saved_result["selection"],
        "Saved selected model mismatch",
    )
    inference = []
    evaluation = c.read(BASE / "data/evaluation.json")
    for variant, saved in saved_result["variants"].items():
        featurepath = ROOT / saved["features"]
        modelpath = ROOT / saved["model"]
        model = c.load_arrays(modelpath)
        mr, mp = receipt(modelpath.parent.parent.parent)
        c.require(
            mr["identity"]["C"] == choice["selected"]["C"],
            "Inference uses wrong selected model",
        )
        if stage != "compact_tfidf":
            wanted = (
                "gradient_train" if variant == "matched_gradient" else "all_permitted"
            )
            c.require(
                c.read(featurepath.parent.parent / "complete.json")["identity"][
                    "partition"
                ]
                == wanted,
                "Budget variant confused",
            )
        actual = np.load(ROOT / saved["probabilities"], allow_pickle=False)
        maximum = 0.0
        zeros = 0
        if rep == "bge":
            cache = np.load(
                BASE / "data/evaluation_vectors.npy", mmap_mode="r", allow_pickle=False
            )
        else:
            state = c.read(featurepath / "vocabulary.json")
            idf = c.load_arrays(featurepath / "idf.npz")
        for start in range(0, len(evaluation), 256):
            if rep == "bge":
                x = np.asarray(cache[start : start + 256], dtype=np.float64)
            else:
                x = independent_features(
                    [r["text"] for r in evaluation[start : start + 256]], state, idf
                )[0]
                zeros += int(np.sum(x.getnnz(axis=1) == 0))
            p = independent_probability(model, x)
            savedpart = actual[start : start + len(p)]
            error = float(np.max(np.abs(p - savedpart), initial=0))
            maximum = max(maximum, error)
            c.require(error < 1e-12, "Evaluation probabilities differ")
            for cutoff in c.CUTOFFS:
                c.require(
                    np.array_equal(p >= cutoff, savedpart >= cutoff)
                    and np.array_equal(p > cutoff, savedpart > cutoff),
                    "Inference cutoff differs",
                )
            c.require(
                np.array_equal(p.argmax(1), savedpart.argmax(1)),
                "Inference argmax differs",
            )
        if rep == "bge":
            del cache
        infrec, _ = receipt(directory / "inference" / variant)
        c.require(
            zeros == infrec["detail"]["zero_feature_rows"],
            "Zero-feature accounting differs",
        )
        inference.append(
            dict(
                variant=variant,
                rows=len(actual),
                labels=actual.shape[1],
                max_abs_probability_difference=maximum,
                zero_feature_rows=zeros,
            )
        )
    save(
        "groups/" + name + "/audit.json",
        dict(
            status="pass",
            stage=stage,
            head=head,
            seed=seed,
            selection=choice,
            features=features,
            fits=fits,
            inference=inference,
            source_sha256=c.sha(Path(__file__)),
            original_result_sha256=c.sha(directory / "result/complete.json"),
        ),
    )
    print(
        "Independent features/refits/selection/every prediction PASS:",
        name,
        len(fits),
        "fits",
        flush=True,
    )


def groups():
    for stage, head, seed in memory.run.group_roster():
        path = OUT / f"groups/{stage}_s{seed}/{head}/audit.json"
        if path.exists():
            c.require(
                c.read(path)["source_sha256"] == c.sha(Path(__file__)),
                "Review source changed",
            )
            continue
        log = OUT / "execution" / (f"{stage}_{head}_{seed}_{time.time_ns()}")
        with patch.object(c, "BASE", OUT):
            memory.monitored_command(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-B",
                    "-u",
                    "-m",
                    __spec__.name,
                    "group",
                    "--stage",
                    stage,
                    "--head",
                    head,
                    "--seed",
                    str(seed),
                ],
                log,
                maximum_seconds=1500,
            )
        print("Reviewed", stage, head, seed, flush=True)


def assemble():
    reviews = [
        c.read(OUT / f"groups/{stage}_s{seed}/{head}/audit.json")
        for stage, head, seed in memory.run.group_roster()
    ]
    c.require(all(v["status"] == "pass" for v in reviews), "Every group audit required")
    bits_checked = 0
    pipelines = []
    for path in sorted((BASE / "pipelines").glob("*/complete.json")):
        rec, attempt = receipt(path.parent)
        contract = c.read(attempt / "contract.json")
        stored = c.load_arrays(attempt / "decisions.npz")
        pp = {
            head: np.load(ROOT / contract["probabilities"][head], allow_pickle=False)
            for head in c.HEADS
        }
        for policy in ("default", "selected"):
            cut = {
                h: (
                    contract["selected_cutoffs"][h]
                    if policy == "selected"
                    else (None if h == "severity" else 0.5)
                )
                for h in c.HEADS
            }
            gate = bits("hate", pp["hate"], cut["hate"])
            c.require(
                np.array_equal(gate, stored[policy + "_hate"]), "Pipeline gate differs"
            )
            bits_checked += gate.size
            for head in c.HEADS[1:]:
                raw = bits(head, pp[head], cut[head])
                delivered = raw.copy()
                delivered[~gate[:, 1]] = False
                c.require(
                    np.array_equal(raw, stored[policy + "_" + head + "_ungated"])
                    and np.array_equal(delivered, stored[policy + "_" + head]),
                    "Pipeline fine head differs",
                )
                bits_checked += 2 * raw.size
        pipelines.append(
            dict(
                name=path.parent.name,
                rows=pp["hate"].shape[0],
                contract_sha256=c.sha(attempt / "contract.json"),
            )
        )
    fits = [f for v in reviews for f in v["fits"]]
    features = [f for v in reviews for f in v["features"]]
    c.require(
        len(fits) == 292 and len(features) == 88 and len(pipelines) == 7,
        "Incomplete audit matrix",
    )
    save(
        "linear_audit.json",
        dict(
            status="pass",
            groups=len(reviews),
            candidate_final_models=len(fits),
            binary_solves=sum(v["binary_solves"] for v in fits),
            constant_columns=sum(v["constant_columns"] for v in fits),
            feature_partitions=len(features),
            pipeline_decision_bits=bits_checked,
            pipelines=pipelines,
            candidate_selection_cells=sum(
                len(v["selection"]["candidates"]) for v in reviews
            ),
            prediction_probabilities_checked=sum(
                i["rows"] * i["labels"] for v in reviews for i in v["inference"]
            ),
            maximum_coefficient_difference=max(
                v["coefficient_max_abs_difference"] for v in fits
            ),
            maximum_feature_difference=max(v["max_abs_difference"] for v in features),
            original_feature_seconds=sum(v["feature_seconds"] for v in features),
            original_fit_seconds=sum(v["original_fit_seconds"] for v in fits),
            review_refit_seconds=sum(v["review_refit_seconds"] for v in fits),
            Qwen_calls=0,
            CUDA_used=False,
            review_refits_are_excluded_from_scientific_run_count=True,
        ),
    )
    print(
        "All 292 fits, 88 feature partitions and seven pipelines independently checked.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["freeze", "inventory", "groups", "group", "assemble"]
    )
    parser.add_argument("--stage")
    parser.add_argument("--head")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    with c.offline(), threadpool_limits(limits=1):
        if args.mode == "group":
            audit_group(args.stage, args.head, args.seed)
        else:
            globals()[args.mode]()


if __name__ == "__main__":
    main()

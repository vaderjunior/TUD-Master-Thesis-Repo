from src.hsrag.paths import PROJECT_ROOT
import os

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    TOKENIZERS_PARALLELISM="false",
    HF_HUB_OFFLINE="1",
)
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import socket
import time
import unicodedata
from unittest.mock import patch
import warnings

import numpy as np

ROOT = PROJECT_ROOT
OUT = ROOT / "experiments/astra/sq2_cuda_review_20261010_v3"
EXP = ROOT / "experiments/astra"
CORE = EXP / "full_kb_core_20260915_v1"
ENC = EXP / "later_evaluation_20261008_v1/encoders"
ROB = EXP / "compact_roberta_probe_20261008_v1"
BGE = EXP / "compact_bge_20261008_v1"
BROAD = EXP / "compact_broad_20261008_v1"
PREP = EXP / "offline_closure_20261006_v1/evaluation_preparation"
HEADS = ("hate", "hate_type", "target_group", "severity")
FIELDS = dict(
    hate="gate",
    hate_type="hate_types",
    target_group="target_groups",
    severity="severity",
    legal="legal",
)


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def rows(path):
    with Path(path).open(encoding="utf8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def encode(value):
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf8")


def digest(value):
    return hashlib.sha256(encode(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path):
    path = Path(path).resolve()
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    payload = encode(value)
    if path.exists():
        need(path.read_bytes() == payload, "Review output already differs: " + name)
    else:
        with path.open("xb") as stream:
            stream.write(payload)


def norm(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def integrity():
    bindings = {}
    receipts = [
        ENC / "plan.json",
        ENC / "complete.json",
        ENC / "preflight.json",
        ENC / "runtime_complete.json",
        ENC / "runtime_preflight.json",
        ENC / "assembled/complete.json",
        ROB / "plan.json",
        ROB / "features_complete.json",
        ROB / "results.json",
        ROB / "preflight.json",
        BROAD / "plan.json",
        BROAD / "paired_plan.json",
        BROAD / "paired_results.json",
        BROAD / "predictions_complete.json",
        BGE / "plan.json",
        BGE / "fit_complete.json",
        BGE / "predictions_complete.json",
    ]
    ep = read(ENC / "plan.json")
    receipts += [ENC / "predictions" / j["job"] / "complete.json" for j in ep["jobs"]]
    receipts += list((ROB / "fits").glob("*_complete.json"))
    for receipt in receipts:
        for key in ("inputs", "outputs", "prepared", "source_sha256"):
            for name, value in read(receipt).get(key, {}).items():
                need(
                    name not in bindings or bindings[name] == value,
                    "Conflicting hash: " + name,
                )
                bindings[name] = value
        bindings[rel(receipt)] = sha(receipt)
    for name, value in read(ROB / "plan.json")["base_cache"]["files"].items():
        bindings[
            str(Path(read(ROB / "plan.json")["base_cache"]["snapshot"]) / name)
        ] = value
    total = 0
    for i, (name, value) in enumerate(sorted(bindings.items())):
        path = ROOT / name
        need(sha(path) == value, "Frozen file changed: " + name)
        total += path.stat().st_size
        if i % 250 == 0:
            print("Integrity hashes checked", i + 1, "/", len(bindings), flush=True)
    roster = rows(PREP / "broad_roster.jsonl")
    need(len(roster) == len({r["key"] for r in roster}) == 21627, "Broad identities")
    bylang = {lang: [r for r in roster if r["lang"] == lang] for lang in ("en", "de")}
    raw = {}
    jobs = []
    n = 0
    for job in ep["jobs"]:
        folder = ENC / "predictions" / job["job"]
        receipt = read(folder / "complete.json")
        saved = rows(folder / "heads.jsonl")
        expected = bylang[job["meta"]["lang"]]
        need(receipt["plan_sha256"] == sha(ENC / "plan.json"), "Head plan binding")
        need(
            receipt["job"] == job["job"]
            and receipt["status"] == "fixed_checkpoint_inference_complete",
            "Job receipt",
        )
        need(receipt["probe"]["decisions_exact"], "GPU reference decisions differ")
        chunk_rows = []
        for path in sorted((folder / "chunks").glob("chunk_*.json")):
            chunk = read(path)
            need(chunk["rows_sha256"] == digest(chunk["rows"]), "Chunk row digest")
            need(
                chunk["stamp"]
                == dict(plan_sha256=sha(ENC / "plan.json"), job=job["job"]),
                "Chunk identity",
            )
            chunk_rows.extend(chunk["rows"])
        need(chunk_rows == saved, "Chunks differ from saved head")
        need(
            [r["key"] for r in saved] == [r["key"] for r in expected], "Head row order"
        )
        need(len(saved) == job["rows"] == receipt["rows"], "Head row count")
        decisions = {}
        minimum_margin = float("inf")
        for row, source in zip(saved, expected):
            need(
                row["text_sha256"]
                == hashlib.sha256(source["text"].encode()).hexdigest(),
                "Head text hash",
            )
            need(
                row["job"] == job["job"]
                and row["checkpoint_sha256"] == job["checkpoint_sha256"],
                "Head checkpoint",
            )
            need(row["labels"] == job["meta"]["labels"], "Head label order")
            logits = np.asarray(row["logits"])
            need(
                logits.shape == (len(row["labels"]),) and np.isfinite(logits).all(),
                "Logits invalid",
            )
            if job["meta"]["kind"] == "multilabel":
                dec = (logits >= 0).astype(int).tolist()
                value = [label for label, bit in zip(row["labels"], dec) if bit]
                margin = float(np.abs(logits).min())
            else:
                dec = int(logits.argmax())
                value = row["labels"][dec]
                margin = float(np.sort(logits)[-1] - np.sort(logits)[-2])
            need(dec == row["decision"], "Stored decision differs from logits")
            decisions[row["key"]] = value
            minimum_margin = min(margin, minimum_margin)
        raw[job["arm"], job["head"], job["seed"]] = decisions
        jobs.append(
            dict(
                job=job["job"],
                rows=len(saved),
                minimum_decision_margin=minimum_margin,
                probe=receipt["probe"],
            )
        )
        n += len(saved)
        print("Every raw row checked:", job["job"], len(saved), flush=True)
    need(len(jobs) == 30 and n == 458712, "Incomplete head matrix")
    assembled = 0
    for arm, lang, system in [
        ("full", "en", "encoder_full"),
        ("kb", "en", "encoder_kb"),
        ("de", "de", "encoder_gbert"),
    ]:
        for seed in range(3):
            saved = rows(ENC / "assembled" / f"{system}_seed{seed}.jsonl")
            need(
                [r["key"] for r in saved] == [r["key"] for r in bylang[lang]],
                "Assembly order",
            )
            for row in saved:
                key = row["key"]
                gate = raw[arm, "hate", seed][key] == "true"
                value = dict(
                    hate=gate, hate_type=[], target_group=[], severity=None, legal=[]
                )
                if arm == "de":
                    value["legal"] = raw[arm, "legal", seed][key]
                elif gate:
                    for head in HEADS[1:]:
                        value[head] = raw[arm, head, seed][key]
                need(
                    value == row["result"]
                    and row["condition"] == system
                    and row["seed"] == seed,
                    "Assembly content",
                )
            assembled += len(saved)
    inputs = read(ROB / "feature_inputs.json")
    mapping = read(ROB / "scope_mapping.json")
    need(len(inputs) == len({r["key"] for r in inputs}) == 19229, "Feature inputs")
    lookup = {r["key"]: r for r in inputs}
    for row in inputs:
        need(
            hashlib.sha256(row["text"].encode()).hexdigest()
            == row["exact_text_sha256"],
            "Feature text hash",
        )
    train = read(BGE / "train.json")
    small = {r["id"]: r for r in rows(CORE / "kb/small/records.jsonl")}
    need(
        [r["source_id"] for r in mapping["train"]] == train["gate_ids"],
        "Feature TRAIN order",
    )
    for row in mapping["train"]:
        need(
            row["key"] == "train/" + row["source_id"]
            and lookup[row["key"]]["text"] == small[row["source_id"]]["text"],
            "TRAIN features identity",
        )
    for scope, expected in [
        ("core", [r for r in rows(CORE / "dev_inputs.jsonl") if r["lang"] == "en"]),
        ("broad", bylang["en"]),
    ]:
        need(
            [r["key"] for r in mapping[scope]] == [r["key"] for r in expected],
            "Feature eval scope",
        )
        for row, source in zip(mapping[scope], expected):
            need(lookup[row["key"]]["text"] == source["text"], "Feature eval text")
            need(
                row["source_id"] == source["item_id" if scope == "core" else "key"],
                "Feature score mapping",
            )
    train_texts = {norm(lookup[r["key"]]["text"]) for r in mapping["train"]}
    evaluation_texts = {
        norm(lookup[r["key"]]["text"])
        for scope in ("core", "broad")
        for r in mapping[scope]
    }
    need(not train_texts & evaluation_texts, "Normalized TRAIN/evaluation overlap")
    feature_info = {}
    for kind in ("mean", "cls"):
        matrix = np.load(
            ROB / f"features_{kind}.npy", allow_pickle=False, mmap_mode="r"
        )
        need(
            matrix.shape == (19229, 768)
            and matrix.dtype == np.float32
            and np.isfinite(matrix).all(),
            "Feature matrix",
        )
        norms = np.linalg.norm(matrix, axis=1)
        need(np.allclose(norms, 1, atol=2e-4), "Feature normalization")
        for start in range(0, len(inputs), 256):
            with np.load(
                ROB / "chunks" / f"chunk_{start:06d}.npz", allow_pickle=False
            ) as chunk:
                need(
                    str(chunk["plan_sha"]) == sha(ROB / "plan.json"),
                    "Feature chunk plan",
                )
                need(
                    str(chunk["rows_sha"]) == digest(inputs[start : start + 256]),
                    "Feature chunk rows",
                )
                need(
                    np.array_equal(chunk[kind], matrix[start : start + 256]),
                    "Feature concatenation",
                )
        feature_info[kind] = dict(
            shape=list(matrix.shape),
            dtype=str(matrix.dtype),
            norm_min=float(norms.min()),
            norm_max=float(norms.max()),
        )
    cp = read(ROB / "features_complete.json")["CPU_CUDA_probe_max_abs"]
    need(max(cp.values()) < 1e-4, "CPU/GPU feature probe")
    save(
        "integrity.json",
        dict(
            status="pass",
            hashes=bindings,
            files=len(bindings),
            bytes=total,
            head_predictions=n,
            jobs=jobs,
            assembled_predictions=assembled,
            feature_matrices=feature_info,
            feature_scope_counts={k: len(v) for k, v in mapping.items()},
            normalized_train_eval_overlap=0,
            CPU_CUDA_probe_max_abs=cp,
            Qwen_calls=0,
            CUDA_used=False,
        ),
    )
    print(
        "Integrity passed:", n, "head rows;", assembled, "assembled rows.", flush=True
    )


def logistic(x, target, strength, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.exceptions import ConvergenceWarning

    if len(np.unique(target)) == 1:
        return {"constant": int(target[0])}
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        fit = LogisticRegression(
            C=strength,
            solver="liblinear",
            class_weight="balanced",
            max_iter=2000,
            tol=1e-8,
            random_state=seed,
        ).fit(x, target)
    return dict(coef=fit.coef_[0], intercept=fit.intercept_[0])


def probabilities(models, x):
    out = []
    for model in models:
        if "constant" in model:
            out.append(np.full(len(x), model["constant"], dtype=float))
        else:
            z = np.einsum("ij,j->i", x, np.asarray(model["coef"])) + model["intercept"]
            out.append(1 / (1 + np.exp(-np.clip(z, -700, 700))))
    return np.column_stack(out)


def bit_f1(y, x, labels):
    tp = np.logical_and(y, x).sum(axis=0)
    fp = np.logical_and(~y, x).sum(axis=0)
    fn = np.logical_and(y, ~x).sum(axis=0)
    den = 2 * tp + fp + fn
    per = np.divide(2 * tp, den, out=np.zeros(len(tp), float), where=den != 0)
    return float(per[labels].mean())


def predict_bits(head, p, cutoff, gate=None):
    if head == "hate":
        bit = p[:, 0] > cutoff
        x = np.column_stack((~bit, bit))
    elif head == "severity":
        x = np.eye(p.shape[1], dtype=bool)[p.argmax(axis=1)]
    else:
        x = p >= cutoff
    if gate is not None:
        x[~gate] = False
    return x


def decode(heads, x, ungated=False):
    bits = {}
    probs = {}
    for head in HEADS:
        probs[head] = probabilities(heads[head]["models"], x)
        bits[head] = predict_bits(
            head,
            probs[head],
            heads[head]["selected"]["threshold"],
            None if head == "hate" or ungated else bits["hate"][:, 1],
        )
    out = []
    for i in range(len(x)):
        value = dict(
            hate=bool(bits["hate"][i, 1]),
            hate_type=[],
            target_group=[],
            severity=None,
            legal=[],
        )
        for head in HEADS[1:]:
            labels = [
                label for label, on in zip(heads[head]["labels"], bits[head][i]) if on
            ]
            value[head] = (
                (labels[0] if labels else None) if head == "severity" else labels
            )
        out.append(value)
    return out, probs


def fits():
    data = read(BGE / "train.json")
    inputs = read(ROB / "feature_inputs.json")
    mapping = read(ROB / "scope_mapping.json")
    pos = {r["key"]: i for i, r in enumerate(inputs)}
    info = {}
    prediction_checks = 0
    candidate_checks = 0
    for kind in ("mean", "cls"):
        allx = np.load(ROB / f"features_{kind}.npy", allow_pickle=False, mmap_mode="r")
        x = np.asarray(
            allx[[pos[r["key"]] for r in mapping["train"]]], dtype=np.float64
        )
        for seed in (42, 43, 44):
            fit = read(ROB / f"fits/{kind}_seed{seed}.json")
            gate = None
            heads = {}
            need(
                fit["plan_sha256"] == sha(ROB / "plan.json")
                and fit["seed"] == seed
                and fit["pooling"] == kind,
                "Fit identity",
            )
            folds = np.asarray(data["folds"][str(seed)])
            groups = defaultdict(set)
            for row, fold in zip(data["heads"]["hate"]["rows"], folds):
                groups[row["cluster"]].add(int(fold))
            need(
                all(len(v) == 1 for v in groups.values()),
                "Shared text crosses TRAIN folds",
            )
            for head in HEADS:
                spec = data["heads"][head]
                saved = fit["heads"][head]
                ix = np.asarray([r["feature_row"] for r in spec["rows"]])
                target = np.asarray([r["target"] for r in spec["rows"]])
                yy = (
                    target[:, None]
                    if head == "hate"
                    else (
                        np.eye(len(spec["labels"]), dtype=int)[target]
                        if head == "severity"
                        else target
                    )
                )
                truth = (
                    np.column_stack((1 - yy[:, 0], yy[:, 0])).astype(bool)
                    if head == "hate"
                    else yy.astype(bool)
                )
                labels = np.flatnonzero(
                    (truth.sum(0) >= 2) & ((~truth).sum(0) >= 2)
                ).tolist()
                need(
                    saved["training_ids"] == [r["id"] for r in spec["rows"]]
                    and saved["labels"] == spec["labels"],
                    "Fit annotation/order",
                )
                need(saved["selection_indices"] == labels, "TRAIN-supported label mask")
                oof = {}
                candidates = []
                for strength in (0.1, 1.0, 10.0, 100.0):
                    p = np.full(yy.shape, np.nan, float)
                    for fold in range(5):
                        trainmask = folds[ix] != fold
                        testmask = ~trainmask
                        if not testmask.any():
                            continue
                        models = [
                            logistic(x[ix[trainmask]], yy[trainmask, j], strength, seed)
                            for j in range(yy.shape[1])
                        ]
                        p[testmask] = probabilities(models, x[ix[testmask]])
                    need(np.isfinite(p).all(), "OOF incomplete")
                    oof[strength] = p
                    for cutoff in (
                        [None] if head == "severity" else [j / 10 for j in range(1, 10)]
                    ):
                        bits = predict_bits(
                            head, p, cutoff, None if head == "hate" else gate[ix]
                        )
                        candidates.append(
                            dict(
                                C=strength,
                                threshold=cutoff,
                                score=bit_f1(truth, bits, labels),
                            )
                        )
                need(len(saved["candidates"]) == len(candidates), "Candidate grid")
                for actual, expected in zip(saved["candidates"], candidates):
                    need(
                        actual["C"] == expected["C"]
                        and actual["threshold"] == expected["threshold"]
                        and abs(actual["score"] - expected["score"]) < 1e-12,
                        "OOF candidate mismatch",
                    )
                choice = min(
                    candidates,
                    key=lambda v: (
                        -v["score"],
                        v["C"],
                        abs(
                            (v["threshold"] if v["threshold"] is not None else 0.5)
                            - 0.5
                        ),
                        -(v["threshold"] or 0),
                    ),
                )
                need(
                    choice["C"] == saved["selected"]["C"]
                    and choice["threshold"] == saved["selected"]["threshold"],
                    "Selected TRAIN recipe differs",
                )
                need(
                    np.allclose(
                        oof[choice["C"]], saved["oof_probabilities"], rtol=0, atol=1e-10
                    ),
                    "OOF probability mismatch",
                )
                if head == "hate":
                    gate = oof[choice["C"]][:, 0] > choice["threshold"]
                refitted = [
                    logistic(x[ix], yy[:, j], choice["C"], seed)
                    for j in range(yy.shape[1])
                ]
                maximum = 0.0
                for replay, stored in zip(refitted, saved["models"]):
                    if "constant" in replay:
                        need(replay == stored, "Constant fit mismatch")
                    else:
                        error = max(
                            float(
                                np.abs(
                                    replay["coef"] - np.asarray(stored["coef"])
                                ).max()
                            ),
                            abs(float(replay["intercept"]) - stored["intercept"]),
                        )
                        maximum = max(maximum, error)
                        need(error < 1e-10, "Final fit coefficient mismatch")
                heads[head] = dict(
                    selected=saved["selected"],
                    training_rows=len(ix),
                    candidate_checks=len(candidates),
                    max_coefficient_error=maximum,
                )
                candidate_checks += len(candidates)
            for scope in ("core", "broad"):
                xx = np.asarray(
                    allx[[pos[r["key"]] for r in mapping[scope]]], dtype=np.float64
                )
                values, _ = decode(fit["heads"], xx)
                saved = rows(ROB / f"predictions/{kind}_{scope}_seed{seed}.jsonl")
                need(
                    [r["key"] for r in saved]
                    == [r["source_id"] for r in mapping[scope]],
                    "Linear prediction order",
                )
                need(
                    [r["result"] for r in saved] == values,
                    "Independent linear decisions differ",
                )
                need(
                    all(r["seed"] == seed and r["pooling"] == kind for r in saved),
                    "Prediction fit attribution",
                )
                prediction_checks += len(saved)
            info[f"{kind}/seed{seed}"] = heads
            print(
                "Independent OOF, selection, refit and predictions pass:",
                kind,
                seed,
                flush=True,
            )
    save(
        "fit_replay.json",
        dict(
            status="pass",
            fits=info,
            candidate_checks=candidate_checks,
            predictions_reconstructed=prediction_checks,
            selection_data="TRAIN only",
            new_neural_training_steps=0,
            Qwen_calls=0,
            CUDA_used=False,
            source_sha256=sha(__file__),
        ),
    )


def clean(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def gold_records(scope, name):
    if scope == "core":
        import pandas as pd

        return [
            {k: clean(v) for k, v in r.items()}
            for r in pd.read_parquet(ROOT / f"data/processed/{name}.parquet").to_dict(
                "records"
            )
        ]
    return [
        r | {"id": r["key"]}
        for r in rows(PREP / "broad_roster.jsonl")
        if r["phase"] == name.split("/")[0]
    ]


def as_set(value, dim):
    if dim == "hate":
        return {"hate" if value else "not_hate"}
    if dim == "severity":
        return {value} if value is not None else set()
    return set(value or [])


def arrays(records, pred, ep, ungated=False):
    dim = ep["dimension"]
    labs = ep["all_labels"]
    ids = []
    ys = []
    xs = []
    for row in records:
        value = clean(row.get(FIELDS[dim]))
        gate = clean(row.get("gate"))
        if value is None or (dim not in ("hate", "legal") and gate is not True):
            continue
        key = str(row["id"])
        out = pred[key]
        need(out is not None and type(out["hate"]) is bool, "Invalid classifier output")
        if not ungated and not out["hate"]:
            need(
                not out["hate_type"]
                and not out["target_group"]
                and out["severity"] is None,
                "Ungated result",
            )
        truth = as_set(value, dim)
        prediction = as_set(out[dim], dim)
        need(truth <= set(labs) and prediction <= set(labs), "Unknown label")
        ids.append(key)
        ys.append([label in truth for label in labs])
        xs.append([label in prediction for label in labs])
    need(ids == ep["scorable_ids"], "Independent eligibility mask differs")
    y = np.asarray(ys, dtype=bool).reshape(len(ids), len(labs))
    x = np.asarray(xs, dtype=bool).reshape(y.shape)
    need(
        y.sum(0).tolist() == [ep["gold_support"][label] for label in labs],
        "Gold support differs",
    )
    return ids, y, x


def summarize_bits(y, x, ep):
    from sklearn.metrics import f1_score

    labels = ep["all_labels"]
    idx = [labels.index(l) for l in ep["macro_labels"]]
    counts = np.stack((y & x, ~y & x, y & ~x), axis=-1).astype(np.int64)
    aggregate = counts.sum(0)
    tp, fp, fn = aggregate.T
    f = np.divide(
        2 * tp,
        2 * tp + fp + fn,
        out=np.zeros(len(tp), float),
        where=(2 * tp + fp + fn) > 0,
    )
    score = float(f[idx].mean()) if len(y) else None
    if len(y):
        need(
            abs(
                score
                - np.mean(
                    [
                        f1_score(y[:, j], x[:, j], average="binary", zero_division=0)
                        for j in idx
                    ]
                )
            )
            < 1e-12,
            "sklearn independent F1",
        )
    stats = dict(
        n=len(y),
        macro_f1=score,
        exact_set_accuracy=float(np.all(y == x, axis=1).mean()) if len(y) else None,
        mean_predicted_labels=float(x.sum(1).mean()) if len(y) else None,
        mean_gold_labels=float(y.sum(1).mean()) if len(y) else None,
        per_label={
            label: dict(
                support=int(y[:, i].sum()),
                tp=int(tp[i]),
                fp=int(fp[i]),
                fn=int(fn[i]),
                precision=float(tp[i] / (tp[i] + fp[i])) if tp[i] + fp[i] else 0.0,
                recall=float(tp[i] / (tp[i] + fn[i])) if tp[i] + fn[i] else 0.0,
                f1=float(f[i]),
            )
            for i, label in enumerate(labels)
        },
        macro_labels=ep["macro_labels"],
    )
    if ep["dimension"] == "hate":
        j = labels.index("hate")
        stats.update(
            false_positive_rate=(
                float(fp[j] / (~y[:, j]).sum()) if (~y[:, j]).any() else None
            ),
            false_negative_rate=float(fn[j] / y[:, j].sum()) if y[:, j].any() else None,
        )
    return stats, counts


def predictions(scope):
    out = {}
    for model, arm in [("encoder_kb", "kb"), ("encoder_full", "full")]:
        for seed in range(3):
            file = (
                CORE / f"encoder_results_v5/en_{arm}_seed{seed}_assembled.jsonl"
                if scope == "core"
                else ENC / f"assembled/{model}_seed{seed}.jsonl"
            )
            out[model, seed] = {
                r["item_id" if scope == "core" else "key"]: r["result"]
                for r in rows(file)
            }
    if scope == "broad":
        for seed in range(3):
            out["encoder_gbert", seed] = {
                r["key"]: r["result"]
                for r in rows(ENC / f"assembled/encoder_gbert_seed{seed}.jsonl")
            }
    for seed in (42, 43, 44):
        saved = (
            read(BGE / f"predictions_seed{seed}.json")
            if scope == "core"
            else rows(BROAD / f"predictions_seed{seed}.jsonl")
        )
        out["bge", seed] = {
            r["item_id" if scope == "core" else "key"]: r["result"] for r in saved
        }
        for kind in ("mean", "cls"):
            out["roberta_" + kind, seed] = {
                r["key"]: r["result"]
                for r in rows(ROB / f"predictions/{kind}_{scope}_seed{seed}.jsonl")
            }
    return out


def scores():
    import yaml

    rp = read(ROB / "results.json")["results"]
    bp = read(BROAD / "paired_results.json")["results"]
    old = read(BGE / "results.json")["endpoints"]
    cells = {}
    verified = 0
    more = {}
    coretexts = {
        norm(r["text"]) for r in rows(CORE / "dev_inputs.jsonl") if r["lang"] == "en"
    }
    for scope in ("core", "broad"):
        eps = read(BGE / "plan.json" if scope == "core" else BROAD / "plan.json")[
            "endpoints"
        ]
        maps = predictions(scope)
        if scope == "broad":
            tax = yaml.safe_load(
                (ROOT / "config/taxonomy.yaml").read_text(encoding="utf8")
            )

            german = [
                r | {"id": r["key"]}
                for r in rows(PREP / "broad_roster.jsonl")
                if r["lang"] == "de"
            ]
            for phase in sorted({r["phase"] for r in german}):
                rr = [r for r in german if r["phase"] == phase]
                for dim in sorted(
                    {d for r in rr for d in r["endpoints"]} & {"hate", "legal"}
                ):
                    labs = (
                        ["not_hate", "hate"]
                        if dim == "hate"
                        else list(tax["dimensions"][dim]["labels"])
                    )
                    support = Counter()
                    ids = []
                    for row in rr:
                        value = row.get(FIELDS[dim])
                        if value is None:
                            continue
                        ids.append(row["id"])
                        support.update(as_set(value, dim))
                    eps[phase + "/" + dim] = dict(
                        dimension=dim,
                        all_labels=labs,
                        macro_labels=[l for l in labs if support[l] >= 10],
                        scorable_ids=ids,
                        gold_support={l: support[l] for l in labs},
                    )
        for name, ep in eps.items():
            rr = gold_records(scope, name)
            byid = {str(r["id"]): r for r in rr}
            runs = defaultdict(list)
            for (model, seed), pred in maps.items():
                if (rr[0]["lang"] == "de") != (model == "encoder_gbert"):
                    continue
                ids, y, x = arrays(rr, pred, ep)
                stats, _ = summarize_bits(y, x, ep)
                stats["seed"] = seed
                runs[model].append(stats)
                existing = None
                if model.startswith("roberta_"):
                    existing = rp[f'{model.removeprefix("roberta_")}/seed{seed}'][
                        scope + "/" + name
                    ]
                elif scope == "broad" and name in bp:
                    existing = bp[name]["runs"][model][
                        (42, 43, 44).index(seed) if model == "bge" else seed
                    ]
                elif scope == "core":
                    existing = old[name]["runs"][
                        {"encoder_kb": "kb", "encoder_full": "full"}.get(model, model)
                    ][(42, 43, 44).index(seed) if model == "bge" else seed]
                if existing is not None:
                    need(
                        abs(stats["macro_f1"] - existing["macro_f1_valid_only"])
                        < 1e-12,
                        "Published score differs " + scope + "/" + name + "/" + model,
                    )
                    need(
                        stats["n"] == existing["n_scorable"]
                        and existing["n_invalid"] == 0,
                        "Invalid/row tally differs",
                    )
                    verified += 1

                groupstats = {}
                groups = {
                    src: np.asarray([byid[key]["source"] == src for key in ids])
                    for src in sorted({byid[key]["source"] for key in ids})
                }
                if scope == "broad":
                    groups["outside_original_core"] = np.asarray(
                        [norm(byid[key]["text"]) not in coretexts for key in ids]
                    )
                    groups["exclude_flagged_empty"] = np.asarray(
                        [not byid[key].get("empty_input", False) for key in ids]
                    )
                for group, mask in groups.items():
                    if mask.any():
                        groupstats[group] = summarize_bits(y[mask], x[mask], ep)[0]
                more[f"{scope}/{name}/{model}/{seed}"] = groupstats
            cells[scope + "/" + name] = dict(
                endpoint={k: v for k, v in ep.items() if k != "scorable_ids"},
                runs=dict(runs),
                means={
                    model: float(np.mean([r["macro_f1"] for r in values]))
                    for model, values in runs.items()
                },
                sample_sds={
                    model: float(np.std([r["macro_f1"] for r in values], ddof=1))
                    for model, values in runs.items()
                },
            )
            print(
                "Independent scores:",
                scope,
                name,
                {m: round(v, 6) for m, v in cells[scope + "/" + name]["means"].items()},
                flush=True,
            )
    save(
        "scores.json",
        dict(
            status="pass",
            published_score_cells_verified=verified,
            cells=cells,
            interpretation="Representation comparisons descriptive, mean primary and CLS supporting; all fitted seeds retained. Scores are macro-F1, not accuracy.",
        ),
    )
    save(
        "descriptive_subgroups.json",
        dict(
            post_result=True,
            fixed_full_roster_macro_labels=True,
            cells=more,
            caution="Overlapping subgroups are explanations/sensitivities, not independently tested new winners.",
        ),
    )


def independent_bootstrap(
    counts, clusters, sources, idx, family, stratify, draws=10000
):

    unique = sorted(set(clusters))
    lookup = {v: i for i, v in enumerate(unique)}
    aggregate = np.zeros((len(unique), counts.shape[0], counts.shape[2], 3), np.int64)
    membership = defaultdict(set)
    for i, key in enumerate(clusters):
        aggregate[lookup[key]] += counts[:, i]
        membership[key].add(sources[i])
    strata = defaultdict(list)
    for key, i in lookup.items():
        strata[tuple(sorted(membership[key])) if stratify else ("all",)].append(i)
    rng = np.random.default_rng(20261008)
    samples = []
    flat = aggregate.reshape(len(unique), -1).astype(float)
    for start in range(0, draws, 64):
        size = min(64, draws - start)
        picks = np.concatenate(
            [
                rng.choice(group, (size, len(group)), replace=True)
                for _, group in sorted(strata.items())
            ],
            axis=1,
        )
        weights = np.asarray(
            [np.bincount(pick, minlength=len(unique)) for pick in picks], dtype=float
        )
        total = (weights @ flat).reshape(size, counts.shape[0], counts.shape[2], 3)
        need(
            np.array_equal(total[0], aggregate[picks[0]].sum(0)),
            "Independent bootstrap direct draw differs",
        )
        tp, fp, fn = (total[..., i] for i in range(3))
        den = 2 * tp + fp + fn
        f = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den != 0)[
            ..., idx
        ].mean(-1)
        samples.extend((f[:, :3].mean(1) - f[:, 3:].mean(1)).tolist())
    return dict(
        ci95=np.quantile(samples, [0.025, 0.975]).tolist(),
        family_interval=(
            np.quantile(
                samples, [0.05 / (2 * family), 1 - 0.05 / (2 * family)]
            ).tolist()
            if family
            else None
        ),
        n_clusters=len(unique),
    )


def statistics():
    plans = read(BROAD / "plan.json")
    published = read(BROAD / "paired_results.json")
    maps = predictions("broad")
    out = {}
    for name, ep in plans["endpoints"].items():
        rr = gold_records("broad", name)
        lookup = {str(r["id"]): r for r in rr}
        ids = ep["scorable_ids"]
        functional = name.startswith("en_hatecheck_eval/")
        clusters = [
            lookup[key]["functional_cluster" if functional else "text_cluster"]
            for key in ids
        ]
        sources = [lookup[key]["source"] for key in ids]
        idx = [ep["all_labels"].index(l) for l in ep["macro_labels"]]
        for arm in ("encoder_kb", "encoder_full"):
            values = []
            for model, seeds in [("bge", (42, 43, 44)), (arm, range(3))]:
                for seed in seeds:
                    _, y, x = arrays(rr, maps[model, seed], ep)
                    values.append(summarize_bits(y, x, ep)[1])
            values = np.stack(values)
            family = (
                4
                if arm == "encoder_kb" and name in published["primary_family"]
                else None
            )
            previous = published["results"][name]["bge_minus"][arm]
            checks = {}
            for mode in ("primary", "sensitivity"):
                actual = independent_bootstrap(
                    values,
                    ids if functional and mode == "sensitivity" else clusters,
                    sources,
                    idx,
                    family if mode == "primary" else None,
                    mode == "sensitivity" and not functional,
                )
                expected = previous if mode == "primary" else previous["sensitivity"]
                for field in ("ci95", "family_interval"):
                    need(
                        (actual[field] is None and expected[field] is None)
                        or actual[field] is not None
                        and np.allclose(
                            actual[field], expected[field], atol=1e-12, rtol=0
                        ),
                        "Bootstrap interval differs " + name + "/" + arm + "/" + mode,
                    )
                need(actual["n_clusters"] == expected["n_clusters"], "Cluster tally")
                checks[mode] = actual
            out[name + "/" + arm] = checks
            print(
                "Independently reproduced both paired intervals:", name, arm, flush=True
            )
    save(
        "statistics.json",
        dict(
            status="pass",
            contrasts=out,
            bootstrap_draws_per_interval=10000,
            intervals_reproduced=2 * len(out),
            primary_family=published["primary_family"],
            primary_confidence=0.9875,
            caveat="Conditional intervals over sampled texts/template clusters, with the fitted settings fixed; no seed/pool population uncertainty or untouched-confirmation claim.",
        ),
    )


def self_tests():
    ep = dict(
        dimension="hate_type",
        all_labels=["a", "b"],
        macro_labels=["a", "b"],
        scorable_ids=["1", "3"],
        gold_support={"a": 1, "b": 0},
    )
    rr = [
        dict(id="1", gate=True, hate_types=["a"]),
        dict(id="2", gate=False, hate_types=["a"]),
        dict(id="3", gate=True, hate_types=[]),
        dict(id="4", gate=True, hate_types=None),
    ]
    values = {
        r["id"]: dict(
            hate=True, hate_type=["a"], target_group=[], severity=None, legal=[]
        )
        for r in rr
    }
    ids, y, x = arrays(rr, values, ep)
    need(
        ids == ["1", "3"]
        and abs(summarize_bits(y, x, ep)[0]["macro_f1"] - 1 / 3) < 1e-12,
        "Missing/empty/gate test",
    )
    p = np.array([[0.5], [0.50001]])
    need(
        predict_bits("hate", p, 0.5)[:, 1].tolist() == [False, True],
        "Gate boundary test",
    )
    need(
        predict_bits("hate_type", p, 0.5)[:, 0].tolist() == [True, True],
        "Fine boundary test",
    )
    need(
        predict_bits("severity", np.array([[0.5, 0.5]]), None).tolist()
        == [[True, False]],
        "Severity tie order test",
    )
    need(
        np.array_equal(
            predict_bits("hate_type", np.ones((1, 2)), 0.5, np.array([False])),
            [[False, False]],
        ),
        "Delivery gate test",
    )
    counts = np.ones((6, 3, 2, 3), dtype=np.int64)
    interval = independent_bootstrap(
        counts, ["x", "x", "y"], ["s", "t", "t"], [0, 1], 4, True, draws=128
    )
    need(
        interval["ci95"] == [0.0, 0.0] and interval["n_clusters"] == 2,
        "Cluster pairing test",
    )
    print("Six focused scientific fixtures passed.", flush=True)


def no_network(*args, **kwargs):
    raise RuntimeError("This review forbids network access")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["all", "integrity", "fits", "scores", "statistics", "tests"]
    )
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", no_network), patch.object(
        socket, "create_connection", no_network
    ):
        self_tests()
        if args.mode == "tests":
            return
        for name in (
            ("integrity", "fits", "scores", "statistics")
            if args.mode == "all"
            else (args.mode,)
        ):
            start = time.perf_counter()
            globals()[name]()
            print(
                name,
                "finished in",
                round(time.perf_counter() - start, 1),
                "seconds",
                flush=True,
            )


if __name__ == "__main__":
    main()

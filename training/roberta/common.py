from __future__ import annotations
from src.hsrag.paths import PROJECT_ROOT

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import sys
import uuid

for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_key] = "1"
os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    HF_HUB_DISABLE_TELEMETRY="1",
    TOKENIZERS_PARALLELISM="false",
    CUBLAS_WORKSPACE_CONFIG=":4096:8",
)

import numpy as np

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/compact_roberta_tuning_20261010_v1"
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
BGE = ROOT / "experiments/astra/compact_bge_20261008_v1"
FEATURES = ROOT / "experiments/astra/compact_roberta_probe_20261008_v1"
HEADS = ("hate", "hate_type", "target_group", "severity")
CV_SEEDS = (0, 1, 2)
FINAL_SEEDS = (10, 11, 12, 13, 14)
THRESHOLDS = tuple(i / 10 for i in range(1, 10))
FAMILIES = ("fixed_unweighted", "fixed_balanced", "tuned")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def ready(value):
    if isinstance(value, dict):
        return {str(k): ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [ready(v) for v in value]
    if isinstance(value, set):
        return sorted(ready(v) for v in value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def encoded(value):
    return (
        json.dumps(
            ready(value), sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False
        )
        + "\n"
    ).encode("utf8")


def signature(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def own(path):
    path = Path(path).resolve()
    require(
        path.is_relative_to(BASE.resolve()) and path != BASE.resolve(),
        "Output outside isolated tuning study",
    )
    return path


def publish_bytes(path, payload):

    path = own(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(
            path.read_bytes() == payload,
            "Refusing to replace different output: " + rel(path),
        )
        return
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex[:10] + ".writing")
    with temporary.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())

    require(not path.exists(), "Destination appeared during publication")
    temporary.rename(path)


def save(path, value):
    publish_bytes(path, encoded(value))


def array(path, value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    publish_bytes(path, stream.getvalue())


def arrays(path, values):
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    publish_bytes(path, stream.getvalue())


def verify(mapping):
    for name, digest in mapping.items():
        require(sha(ROOT / name) == digest, "Changed frozen input: " + name)


def environment():
    names = (
        "torch",
        "transformers",
        "tokenizers",
        "safetensors",
        "numpy",
        "pandas",
        "pyarrow",
        "scikit-learn",
        "scipy",
    )
    return {
        "python": sys.version.split()[0],
        "packages": {n: importlib.metadata.version(n) for n in names},
    }


def source_paths():
    paths = [
        ROOT / "training/roberta/common.py",
        ROOT / "training/roberta/prepare.py",
        ROOT / "training/roberta/train.py",
        ROOT / "training/roberta/run.py",
        ROOT / "training/roberta/audit.py",
    ]
    return paths + [
        ROOT / "tests/training/roberta/test_training.py",
        ROOT / "scripts/run_roberta_search.ps1",
        ROOT / "training/roberta/tokenizer.py",
        ROOT / "training/frozen_features/compact.py",
    ]


def source_hashes():
    return {rel(p): sha(p) for p in source_paths()}


def configs():
    return [
        dict(id=f"l{i}_e{epoch}_{loss[0]}", learning_rate=lr, epochs=epoch, loss=loss)
        for i, lr in enumerate((5e-6, 2e-5, 5e-5))
        for epoch in (20, 60)
        for loss in ("unweighted", "balanced")
    ]


def config_map():
    return {v["id"]: v for v in configs()}


def protocol():
    return dict(
        version=1,
        heads=HEADS,
        training_counts=dict(hate=147, hate_type=67, target_group=60, severity=37),
        fold_assignment=42,
        fold_count=5,
        cv_neural_seeds=CV_SEEDS,
        final_neural_seeds=FINAL_SEEDS,
        configurations=configs(),
        cutoffs=THRESHOLDS,
        families=FAMILIES,
        fixed_configs=dict(fixed_unweighted="l1_e20_u", fixed_balanced="l1_e20_b"),
        optimizer=dict(
            name="torch.optim.AdamW",
            fused=True,
            betas=[0.9, 0.999],
            eps=1e-8,
            batch_size=16,
            gradient_accumulation=1,
            weight_decay=0.01,
            max_grad_norm=1.0,
            warmup_ratio=0.1,
            schedule="linear; horizon equals the candidate total optimizer-step budget",
        ),
        precision="FP32 parameters; FP16 CUDA autocast/GradScaler for training; FP32 standalone inference; TF32 disabled",
        numerical="Deterministic algorithms required. Nonfinite forward loss/logits stop. Record skipped AMP steps; stop after >20 consecutive skips or no successful updates.",
        model="Pinned roberta-base; RobertaForSequenceClassification; original guarded tokenizer; max128; original two-logit gate, seven-logit fine heads, three-logit severity",
        loss="Unweighted CE for hate/severity, BCE for multi-label. Balanced losses multiply unreduced terms by TRAIN-fold inverse class frequencies, then ordinary mean. Missing classes use unit observed-class weights, never inverse-zero.",
        selection="Average the three per-neural-seed macro-F1 values on pooled held-fold TRAIN predictions, not predictions averaged as an ensemble. Select gate first in each family; select fine heads with that family/seed held-fold gate. Severity argmax. No per-label cutoff search.",
        support="TRAIN-only labels with >=2 positives and >=2 negatives; keep unsupported labels in fitting/evaluation. Require nonempty folds and both gate classes. No evaluation labels in preparation/training/selection.",
        ties="Higher mean score; then fewer epochs, smaller learning rate, unweighted loss, cutoff nearest .5, larger cutoff. Exact floating scores; deterministic order.",
        decisions="Gate positive iff p>cutoff; fine label iff p>=cutoff; severity argmax in recorded label order. Non-hate clears every fine output.",
        variants="Each family has selected-cutoff and default-.5 outputs from exactly the same final weights. Tuned/default is a fixed-weights diagnostic, not separately selected at .5.",
        final_refit="All permitted rows per head; five new initialization/order seeds 10..14. Reuse identical selected head/config/seed jobs across families; no best-seed selection.",
        evaluation="Core970 + broader18275 English input identities; union19082. Prediction inputs contain text/identity only. Gold scoring occurs after every selection/final fit is locked.",
        primary="Tuned/selected minus unchanged compact BGE: en_test hate/types, larger_dev targets/severity. Four-question family; 10000 paired text-cluster draws; 98.75% intervals conditional on final fits. Report neural seed spread separately.",
        comparators="Preserved BGE mean across its three fold settings; frozen-RoBERTa mean, archived original and fixed recipe/balancing/threshold controls supporting. Different randomization axes and selection costs remain explicit.",
        exposure="Exploratory follow-up motivated by earlier exposed DEV/TEST/HateCheck findings. TRAIN-only selection does not erase research exposure; selection OOF scores are not unbiased performance estimates.",
        resume="One launcher lock. Completed job receipts/hashes verified and skipped. Incomplete fitting attempts retained, then that job restarts from pretrained weights. Inference resumes verified 512-row chunks; no remote requests.",
        storage="CV models live in memory only; retain all CV logits/loss curves/receipts. Preserve up to60 unique final checkpoints (~30GiB), raw inference and source/selection hashes. Require45GiB free initially and8GiB before a fitting job.",
        scope="English original147 only. No Qwen, KB edit, new humans, German fitting, extra pools, full-data linear fit or TF-IDF execution in this launcher.",
    )


def search_jobs():
    return [
        dict(
            id=f'{h}/{cfg["id"]}/s{s}_f{k}',
            mode="search",
            head=h,
            config=cfg["id"],
            seed=s,
            fold=k,
        )
        for h in HEADS
        for cfg in configs()
        for s in CV_SEEDS
        for k in range(5)
    ]


def truth(head, rows, labels):
    y = np.asarray([r["target"] for r in rows], dtype=np.int64)
    return (
        y
        if head in ("hate_type", "target_group")
        else np.eye(len(labels), dtype=np.int64)[y]
    )


def probabilities(head, logits):
    logits = np.asarray(logits, dtype=np.float64)
    require(logits.ndim == 2 and np.isfinite(logits).all(), "Invalid logits")
    if head in ("hate_type", "target_group"):
        return np.exp(-np.logaddexp(0.0, -logits))
    z = np.exp(logits - logits.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def decisions(head, prob, cutoff=None, gate=None):
    if head == "hate":
        positive = prob[:, 1] > cutoff
        out = np.column_stack((~positive, positive)).astype(np.int64)
    elif head == "severity":
        out = np.eye(prob.shape[1], dtype=np.int64)[prob.argmax(axis=1)]
    else:
        out = (prob >= cutoff).astype(np.int64)
    if gate is not None:
        out[~np.asarray(gate, dtype=bool)] = 0
    return out


def macro(y, z, supported):
    if not len(supported):
        return None
    a, b = y[:, supported].astype(bool), z[:, supported].astype(bool)
    tp = (a & b).sum(0)
    fp = (~a & b).sum(0)
    fn = (a & ~b).sum(0)
    return float(
        np.divide(
            2 * tp,
            2 * tp + fp + fn,
            out=np.zeros(len(supported)),
            where=(2 * tp + fp + fn) > 0,
        ).mean()
    )


def class_weights(head, y, classes, balanced):
    y = np.asarray(y)
    require(len(y) > 0, "Empty training partition")
    if head in ("hate_type", "target_group"):
        pos = y.sum(0)
        neg = len(y) - pos
        counts = np.stack((neg, pos), axis=1)
        weights = np.ones(counts.shape, dtype=np.float64)
        valid = (pos > 0) & (neg > 0)
        if balanced:
            weights[valid] = len(y) / (2 * counts[valid])
    else:
        counts = np.bincount(y.astype(int), minlength=classes)
        weights = np.ones(classes, dtype=np.float64)
        if balanced:
            present = counts > 0
            weights[present] = len(y) / (present.sum() * counts[present])
    return dict(counts=counts.tolist(), weights=weights.tolist(), balanced=balanced)


def candidate_key(row):
    cfg = config_map()[row["config"]]
    t = 0.5 if row["threshold"] is None else row["threshold"]
    return (
        -row["mean_score"],
        cfg["epochs"],
        cfg["learning_rate"],
        cfg["loss"] != "unweighted",
        abs(t - 0.5),
        -t,
    )


def select(data, oof):
    "oof[head][config] has axes neural seed, annotated row, output label."
    result = {}
    for family in FAMILIES:
        gate = None
        heads = {}
        allowed = (
            [protocol()["fixed_configs"][family]]
            if family != "tuned"
            else list(config_map())
        )
        for head in HEADS:
            spec = data["heads"][head]
            yy = truth(head, spec["rows"], spec["labels"])
            supported = np.flatnonzero((yy.sum(0) >= 2) & ((1 - yy).sum(0) >= 2))
            require(
                len(supported) > 0, "No selection support in declared original pool"
            )
            ix = [r["feature_row"] for r in spec["rows"]]
            candidates = []
            for cfg in allowed:
                pp = oof[head][cfg]
                require(
                    pp.shape == (len(CV_SEEDS), len(yy), yy.shape[1])
                    and np.isfinite(pp).all(),
                    "OOF shape/finite mismatch",
                )
                for cutoff in (None,) if head == "severity" else THRESHOLDS:
                    values = [
                        macro(
                            yy,
                            decisions(
                                head,
                                pp[j],
                                cutoff,
                                None if gate is None else gate[j, ix],
                            ),
                            supported,
                        )
                        for j in range(len(CV_SEEDS))
                    ]
                    candidates.append(
                        dict(
                            config=cfg,
                            threshold=cutoff,
                            seed_scores=values,
                            mean_score=float(np.mean(values)),
                        )
                    )
            chosen = min(candidates, key=candidate_key)
            heads[head] = dict(
                selected=chosen,
                candidates=candidates,
                selection_indices=supported.tolist(),
            )
            if head == "hate":
                gate = oof[head][chosen["config"]][:, :, 1] > chosen["threshold"]
        result[family] = heads
    return result


def final_jobs(selection):
    jobs = {}
    for heads in selection.values():
        for head in HEADS:
            cfg = heads[head]["selected"]["config"]
            for seed in FINAL_SEEDS:
                key = f"{head}/{cfg}/s{seed}"
                jobs[key] = dict(
                    id=key, mode="final", head=head, config=cfg, seed=seed, fold=None
                )
    return list(jobs.values())


def job_dir(job):
    return BASE / ("search" if job["mode"] == "search" else "refits") / job["id"]


def verify_receipt(path, job, plan_sha):
    value = read(path)
    require(
        value["job"] == job
        and value["plan_sha256"] == plan_sha
        and value["status"] == "complete",
        "Completion receipt does not match job/plan",
    )
    attempt = own(BASE / value["attempt"])
    require(
        attempt.is_relative_to(job_dir(job).resolve()),
        "Receipt attempt belongs to another job",
    )
    for name, digest in value["products"].items():
        file = own(attempt / name)
        require(
            file.is_relative_to(attempt) and sha(file) == digest,
            "Completed job artifact differs: " + str(file),
        )
    return value, attempt


@contextmanager
def launcher_lock():
    BASE.mkdir(parents=True, exist_ok=True)
    path = BASE / "launcher.lock"
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                "Another RoBERTa tuning launcher is running; use only one copy."
            ) from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)

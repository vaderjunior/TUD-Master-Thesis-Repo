from src.hsrag.paths import PROJECT_ROOT
from contextlib import contextmanager
from datetime import datetime, timezone
import ctypes
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import time
import unicodedata
import uuid
import warnings
from unittest.mock import patch

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ.update(
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1"
)

import numpy as np
from scipy import sparse
from scipy.special import expit, softmax
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import normalize

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/sq2_cpu_baselines_20261011_v1"
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
COMPACT = ROOT / "experiments/astra/compact_bge_20261008_v1"
TUNING = ROOT / "experiments/astra/compact_roberta_tuning_20261010_v1"
BROAD = ROOT / "experiments/astra/compact_broad_20261008_v1"
LATER = ROOT / "experiments/astra/later_evaluation_20261008_v1"
HEADS = ("hate", "hate_type", "target_group", "severity")
FIELDS = dict(
    hate="gate",
    hate_type="hate_types",
    target_group="target_groups",
    severity="severity",
)
CS = (0.1, 1.0, 10.0, 100.0)
CUTOFFS = tuple(i / 10 for i in range(1, 10))
SEEDS = (42, 43, 44)
PROTOCOL = ROOT / "Thesis_Log/astra_result/astra_sq2_cpu_protocol_20261011_v1.md"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def utc():
    return datetime.now(timezone.utc).isoformat()


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf8")


def signature(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def cluster(text):
    return text_sha(" ".join(unicodedata.normalize("NFKC", text).casefold().split()))


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def rows(path):
    with Path(path).open(encoding="utf8") as stream:
        for line in stream:
            yield json.loads(line)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def verify(bindings):
    for path, expected in bindings.items():
        require(sha(ROOT / path) == expected, "Changed frozen file: " + path)


def publish(path, value):
    path = Path(path).resolve()
    require(
        path.is_relative_to(BASE.resolve()) and path != BASE.resolve(),
        "Output outside isolated CPU study",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(
            path.read_bytes() == value,
            "Refusing to replace different artifact: " + str(path),
        )
        return

    temporary = path.with_name(".w_" + uuid.uuid4().hex[:10])
    with temporary.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
    require(not path.exists(), "Output appeared during immutable publication")
    temporary.rename(path)


def save(path, obj):
    publish(path, encoded(obj))


def array(path, value):
    stream = io.BytesIO()
    np.save(stream, value, allow_pickle=False)
    publish(path, stream.getvalue())


def arrays(path, values):
    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    publish(path, stream.getvalue())


def sparse_save(path, value):
    stream = io.BytesIO()
    sparse.save_npz(stream, value, compressed=True)
    publish(path, stream.getvalue())


def environment():
    return dict(
        python=sys.version,
        packages={
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "pyarrow", "threadpoolctl")
        },
    )


def resources():
    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (name, ctypes.c_ulonglong)
            for name in (
                "total",
                "available",
                "page_total",
                "page_available",
                "virtual_total",
                "virtual_available",
                "extended",
            )
        ]

    m = Memory()
    m.length = ctypes.sizeof(m)
    require(os.name == "nt", "This resource guard is for the recorded Windows host")
    require(
        bool(ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))),
        "Cannot read available RAM",
    )

    class Counters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (name, ctypes.c_size_t)
            for name in (
                "PeakWorkingSetSize",
                "WorkingSetSize",
                "QuotaPeakPagedPoolUsage",
                "QuotaPagedPoolUsage",
                "QuotaPeakNonPagedPoolUsage",
                "QuotaNonPagedPoolUsage",
                "PagefileUsage",
                "PeakPagefileUsage",
            )
        ]

    p = Counters()
    p.cb = ctypes.sizeof(p)
    kernel = ctypes.windll.kernel32
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    query = ctypes.windll.psapi.GetProcessMemoryInfo
    query.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
    require(
        bool(query(kernel.GetCurrentProcess(), ctypes.byref(p), p.cb)),
        "Cannot read process memory",
    )
    return dict(
        available_GiB=m.available / 2**30,
        total_GiB=m.total / 2**30,
        free_disk_GiB=shutil.disk_usage(ROOT).free / 2**30,
        process_peak_working_set_GiB=p.PeakWorkingSetSize / 2**30,
        utc=utc(),
    )


def resource_guard(min_ram=6.0, min_disk=15.0):
    snapshot = resources()
    require(
        snapshot["available_GiB"] >= min_ram,
        f"Need {min_ram:g} GiB available RAM; currently {snapshot['available_GiB']:.2f}. Keep completed work and resume later.",
    )
    require(
        snapshot["free_disk_GiB"] >= min_disk,
        f"Need {min_disk:g} GiB free disk. Preserve outputs; do not delete artifacts automatically.",
    )
    return snapshot


@contextmanager
def file_lock(path, existing=False):
    path = Path(path)
    if existing and not path.exists():
        yield
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("r+b" if existing else "a+b") as stream:
        if not existing and stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        import msvcrt

        try:
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError("Launcher is already active: " + str(path)) from exc
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def current_cuda_guard():

    path = ROOT / "experiments/astra/matched_type_pools_20261011_v1/launcher.lock"
    with file_lock(path, existing=True):
        pass


@contextmanager
def offline():
    def forbidden(*args, **kwargs):
        raise RuntimeError("Network access is disabled for the CPU study")

    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ):
        yield


def inspect_task(directory, identity):
    directory = Path(directory)
    path = directory / "complete.json"
    if not path.exists():
        return None
    receipt = read(path)
    require(
        receipt["identity"] == identity,
        "Completed job has a different identity: " + str(directory),
    )
    for name, digest in receipt["outputs"].items():
        target = (directory / name).resolve()
        require(
            target.is_relative_to(directory.resolve()),
            "Receipt path escapes job directory",
        )
        require(sha(target) == digest, "Completed job artifact changed: " + str(target))
    return receipt


def task(directory, identity, callback):
    "Atomic completion; keep partial attempts, reuse verified completed tasks."
    directory = Path(directory)
    receipt = inspect_task(directory, identity)
    if receipt is not None:
        return receipt
    attempt = directory / "attempts" / uuid.uuid4().hex[:12]
    attempt.mkdir(parents=True, exist_ok=False)
    save(attempt / "started.json", dict(identity=identity, utc=utc()))
    start = time.perf_counter()
    detail = callback(attempt)
    files = sorted(
        p for p in attempt.rglob("*") if p.is_file() and not p.name.startswith(".w_")
    )
    require(files, "Task wrote no evidence")
    receipt = dict(
        identity=identity,
        attempt=attempt.relative_to(directory).as_posix(),
        seconds_this_attempt=time.perf_counter() - start,
        utc=utc(),
        detail=detail,
        outputs={p.relative_to(directory).as_posix(): sha(p) for p in files},
    )
    save(directory / "complete.json", receipt)
    return inspect_task(directory, identity)


def task_path(directory, receipt):
    return Path(directory) / receipt["attempt"]


def target_bits(head, spec, selected_rows=None):
    selected_rows = spec["rows"] if selected_rows is None else selected_rows
    values = np.asarray([r["target"] for r in selected_rows], dtype=int)
    if not len(selected_rows):
        return np.empty((0, 1 if head == "hate" else len(spec["labels"])), dtype=int)
    if head == "hate":
        return values[:, None]
    if head == "severity":
        return np.eye(len(spec["labels"]), dtype=int)[values]
    require(
        values.shape == (len(selected_rows), len(spec["labels"])),
        "Target label dimensions differ",
    )
    return values


def truth(head, bits):
    return np.column_stack((1 - bits[:, 0], bits[:, 0])) if head == "hate" else bits


def decisions(head, probability, cutoff, gate=None):
    probability = np.asarray(probability)
    require(
        probability.ndim == 2 and np.isfinite(probability).all(),
        "Invalid probability matrix",
    )
    if head == "hate":
        z = probability[:, 0] > cutoff
        return np.column_stack((~z, z)).astype(np.uint8)
    if head == "severity":
        out = np.eye(probability.shape[1], dtype=np.uint8)[probability.argmax(axis=1)]
    else:
        out = (probability >= cutoff).astype(np.uint8)
    if gate is not None:
        require(len(gate) == len(out), "Gate alignment differs")
        out[~np.asarray(gate, dtype=bool)] = 0
    return out


def choose(candidates):
    require(bool(candidates), "No selection candidates")
    return min(
        candidates,
        key=lambda r: (
            -r["score"],
            r["C"],
            abs((r["threshold"] if r["threshold"] is not None else 0.5) - 0.5),
            -(r["threshold"] or 0),
        ),
    )


def selection(head, yy, probabilities, gate=None, compact=False):
    actual = truth(head, yy)
    support = actual.sum(axis=0)
    indices = (
        np.flatnonzero((support >= 2) & ((len(actual) - support) >= 2))
        if compact
        else np.arange(actual.shape[1])
    )
    require(
        len(indices), "No TRAIN-supported selection label; stop for protocol review"
    )
    candidates = []
    for strength, p in probabilities.items():
        require(
            p.shape == yy.shape and np.isfinite(p).all(),
            "Incomplete selection probabilities",
        )
        for cutoff in ([None] if head == "severity" else CUTOFFS):
            pred = decisions(head, p, cutoff, gate)
            tp = (actual * pred).sum(axis=0)
            fp = ((1 - actual) * pred).sum(axis=0)
            fn = (actual * (1 - pred)).sum(axis=0)
            den = 2 * tp + fp + fn
            value = float(
                np.divide(2 * tp, den, out=np.zeros(len(tp)), where=den != 0)[
                    indices
                ].mean()
            )

            separate = f1_score(actual, pred, average=None, zero_division=0)[
                indices
            ].mean()
            require(abs(value - separate) < 1e-12, "Independent selection F1 differs")
            candidates.append(dict(C=float(strength), threshold=cutoff, score=value))
    return dict(
        selected=choose(candidates),
        candidates=candidates,
        selection_indices=indices.tolist(),
        positive_support=support.tolist(),
        negative_support=(len(actual) - support).tolist(),
        selection_is_development_not_unbiased_validation=True,
    )


def fit_linear(x, y, strength, seed):
    require(
        x.shape[0] == len(y) and len(y) > 0 and x.shape[1] > 0,
        "Empty or mismatched fitting data",
    )
    require(set(np.unique(y)) <= {0, 1}, "Invalid binary target")
    n, d = y.shape[1], x.shape[1]
    coef = np.zeros((n, d), dtype=np.float64)
    intercept = np.zeros(n)
    constants = np.full(n, -1, dtype=int)
    iterations = np.zeros(n, dtype=int)
    reference = np.empty((x.shape[0], n))
    for j in range(n):
        classes = np.unique(y[:, j])
        if len(classes) == 1:
            constants[j] = int(classes[0])
            reference[:, j] = classes[0]
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            fitted = LogisticRegression(
                C=strength,
                solver="liblinear",
                class_weight="balanced",
                max_iter=2000,
                tol=1e-8,
                random_state=seed,
            ).fit(x, y[:, j])
        coef[j] = fitted.coef_[0]
        intercept[j] = fitted.intercept_[0]
        iterations[j] = fitted.n_iter_[0]
        reference[:, j] = fitted.predict_proba(x)[:, 1]
    model = dict(
        coef=coef, intercept=intercept, constants=constants, iterations=iterations
    )
    require(
        np.allclose(predict_linear(model, x), reference, rtol=0, atol=1e-12),
        "Serialized coefficient predictions differ",
    )
    return model


def predict_linear(model, x):
    require(
        x.shape[1] == model["coef"].shape[1], "Feature dimension differs from model"
    )
    logits = np.asarray(x @ model["coef"].T) + model["intercept"]
    result = expit(logits)
    for j, value in enumerate(model["constants"]):
        if value >= 0:
            result[:, j] = value
    require(
        np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all(),
        "Invalid fitted probabilities",
    )
    return result


def load_arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def vectorizer(kind):
    return TfidfVectorizer(
        analyzer="word" if kind == "word" else "char_wb",
        ngram_range=(1, 2) if kind == "word" else (3, 5),
        max_features=100000 if kind == "word" else 150000,
        lowercase=True,
        min_df=1,
        max_df=1.0,
        strip_accents=None,
        stop_words=None,
        sublinear_tf=True,
        smooth_idf=True,
        norm="l2",
        dtype=np.float64,
    )


def fit_text_features(texts):
    matrices = []
    state = {}
    vectors = {}
    for kind in ("word", "char"):
        v = vectorizer(kind)
        try:
            x = v.fit_transform(texts).tocsr()
        except ValueError as exc:
            if "empty vocabulary" not in str(exc):
                raise
            x = sparse.csr_matrix((len(texts), 0), dtype=np.float64)
            v = None
        if v is None:
            state[kind] = dict(vocabulary={})
            vectors[kind] = np.empty(0, dtype=np.float64)
        else:
            state[kind] = dict(
                vocabulary={str(k): int(i) for k, i in v.vocabulary_.items()}
            )
            vectors[kind] = np.asarray(v.idf_, dtype=np.float64)
        matrices.append(x)
    out = sparse.hstack(matrices, format="csr")
    require(out.shape[1] > 0, "Completely empty text representation")
    out = normalize(out, copy=False)
    restored = transform_texts(texts, state, vectors)
    delta = (out - restored).data
    require(
        not len(delta) or np.max(np.abs(delta)) < 1e-12,
        "Vocabulary/IDF serialization changes features",
    )
    return out, state, vectors


def transform_texts(texts, state, idf):
    matrices = []
    for kind in ("word", "char"):
        vocab = state[kind]["vocabulary"]
        require(
            len(vocab) == len(idf[kind])
            and set(vocab.values()) == set(range(len(vocab))),
            "Malformed vocabulary/IDF",
        )
        if not vocab:
            matrices.append(sparse.csr_matrix((len(texts), 0), dtype=np.float64))
            continue
        v = vectorizer(kind)
        v.set_params(vocabulary=vocab)
        v.idf_ = idf[kind]
        matrices.append(v.transform(texts).tocsr())
    return normalize(sparse.hstack(matrices, format="csr"), copy=False)


def neural_probabilities(head, logits):
    logits = np.asarray(logits, dtype=np.float64)
    require(np.isfinite(logits).all(), "Nonfinite saved neural logits")
    if head == "hate":
        require(logits.shape[1] == 2, "Hate softmax requires two logits")
        return softmax(logits, axis=1)[:, 1:2]
    return softmax(logits, axis=1) if head == "severity" else expit(logits)

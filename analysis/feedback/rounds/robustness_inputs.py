import os

for name in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import hashlib
import math
import numpy as np
from analysis.feedback.rounds import reader as reader

ROOT, STUDY, REVIEW, READER = reader.ROOT, reader.STUDY, reader.REVIEW, reader.READER
OUT = REVIEW / "part7"
PART2 = REVIEW / "part2"
PART3 = REVIEW / "part3"
PART6 = REVIEW / "part6"
need, read, rows, sha, rel, pure, physical = (
    reader.need,
    reader.read,
    reader.rows,
    reader.sha,
    reader.rel,
    reader.pure,
    reader.physical,
)
STREAMS, CONDITIONS = reader.STREAMS, reader.CONDITIONS
POLICIES = ["reference", "adaptive", "static", "random"]
POPS = ["en_test", "en_hatecheck_eval"]


def native(v):
    if isinstance(v, np.ndarray):
        return native(v.tolist())
    if isinstance(v, np.generic):
        return native(v.item())
    if isinstance(v, float) and not math.isfinite(v):
        return None
    if isinstance(v, dict):
        return {str(k): native(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [native(x) for x in v]
    return v


def write(path, value):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside Part 7")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(
            value
            if isinstance(value, bytes)
            else pure.encoded(native(value), pretty=True)
        )


def lines(path, values):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside Part 7")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        for v in values:
            f.write(pure.encoded(native(v)))


def npz(path, **values):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside Part 7")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, **values)


def seed(name):
    return int.from_bytes(
        hashlib.sha256(("A17-part7-robustness-v1|" + name).encode()).digest()[:8],
        "little",
    )


def bindings(paths):
    return {rel(p): sha(p) for p in paths}


def check(mapping):
    for path, digest in mapping.items():
        need(sha(ROOT / path) == digest, "Bound evidence changed: " + path)

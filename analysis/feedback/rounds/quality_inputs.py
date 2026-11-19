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
from analysis.feedback.transfer import quality_metrics as quality

ROOT, STUDY, REVIEW, READER = reader.ROOT, reader.STUDY, reader.REVIEW, reader.READER
OUT = REVIEW / "part6"
PART5 = REVIEW / "part5"
PART3 = REVIEW / "part3"
PART2 = REVIEW / "part2"
need, read, rows, sha, rel, pure = (
    reader.need,
    reader.read,
    reader.rows,
    reader.sha,
    reader.rel,
    reader.pure,
)
STREAMS, CONDITIONS = reader.STREAMS, reader.CONDITIONS
POLICIES = ["reference", "adaptive", "static", "random"]
HEADS = ["binary", "hate_type", "target_group", "severity", "known_answer"]
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
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside new Part 6")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(
            value
            if isinstance(value, bytes)
            else pure.encoded(native(value), pretty=True)
        )


def lines(path, values):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside new Part 6")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        for v in values:
            f.write(pure.encoded(native(v)))


def seed(name):
    return int.from_bytes(
        hashlib.sha256(("A17-part6-quality-v1|" + name).encode()).digest()[:8], "little"
    )


def rank(group, category, key):
    return hashlib.sha256(
        f"A17-part6-case-v1|{group}|{category}|{key}".encode()
    ).hexdigest()


def metadata(row):
    m = {k: v for k, v in row.items() if k != "predictions"}
    fields = [h for h in quality.GOLD if quality.eligible(m, h)]
    m["gold_profile"] = ("hateful_" if m["gold_hate"] else "benign_") + (
        "+".join(fields) if fields else "gate_only"
    )
    m["any_fine_gold"] = bool(fields)
    return m

import os

for name in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[name] = "1"
from pathlib import Path
import numpy as np
from analysis.feedback.rounds import reader as previous

ROOT, STUDY, REVIEW, PART2, READER = (
    previous.ROOT,
    previous.STUDY,
    previous.REVIEW,
    previous.OUT,
    previous.READER,
)
PART3, OUT = REVIEW / "part3", REVIEW / "part4"
STREAMS, POLICIES, CONDITIONS = previous.STREAMS, previous.POLICIES, previous.CONDITIONS
need, read, rows, sha, rel, pure, physical = (
    previous.need,
    previous.read,
    previous.rows,
    previous.sha,
    previous.rel,
    previous.pure,
    previous.physical,
)
POPS = ["en_test", "en_hatecheck_eval"]


def native(value):
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def write(path, value):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside new Part 4")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(
            value
            if isinstance(value, bytes)
            else pure.encoded(native(value), pretty=True)
        )


def lines(path, values):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside new Part 4")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        for v in values:
            f.write(pure.encoded(native(v)))


def policy_array(error, policy):
    if policy == "reference":
        return error[:, :, 0, :]
    cols = [CONDITIONS.index(f"{policy}_s{s}") for s in STREAMS]
    return error[:, :, cols, :].reshape((len(error), 4, 9))

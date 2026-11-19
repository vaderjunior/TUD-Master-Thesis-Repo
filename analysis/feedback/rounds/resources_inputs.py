import os

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from pathlib import Path
import gzip
import io
import json
from analysis.feedback.rounds import robustness_inputs as prior

ROOT, STUDY, REVIEW, READER = prior.ROOT, prior.STUDY, prior.REVIEW, prior.READER
OUT = REVIEW / "part8"
LINEAR = ROOT / "experiments/astra/sq3_linear_update_20261012_v1"
need, read, rows, sha, rel, pure, physical = (
    prior.need,
    prior.read,
    prior.rows,
    prior.sha,
    prior.rel,
    prior.pure,
    prior.physical,
)
STREAMS, CONDITIONS, POPS = prior.STREAMS, prior.CONDITIONS, prior.POPS
native, bindings, check = prior.native, prior.bindings, prior.check


def write(path, value):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside Part 8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(
            value
            if isinstance(value, bytes)
            else pure.encoded(native(value), pretty=True)
        )


def gzopen(path):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside Part 8")
    path.parent.mkdir(parents=True, exist_ok=True)
    return io.TextIOWrapper(
        gzip.GzipFile(filename=str(path), mode="xb", mtime=0, compresslevel=5),
        encoding="utf8",
        newline="\n",
    )


def emit(f, value):
    f.write(
        json.dumps(
            native(value),
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )


def gzrows(path):
    with gzip.open(path, "rt", encoding="utf8") as f:
        for line in f:
            yield json.loads(line)

import os

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    CUDA_VISIBLE_DEVICES="",
)
from pathlib import Path
import json
import hashlib
import numpy as np
from analysis.classification.broad import execution as runtime
from analysis.common import result_reader as scoring
from analysis.common import replay as gold

a, c, e = (runtime.a, runtime.c, runtime.e)
ROOT = c.ROOT
BASE = ROOT / "experiments/astra/sq2_matched_review_20261030_v1"
PLAN = ROOT / "Thesis_Log/astra_result/astra_sq2_matched_review_plan_20261030_v1.md"
PHASES = list(scoring.MATCHED_DIMS)
DIMENSIONS = ["hate", "hate_type", "target_group", "severity", "legal"]
PAIRS = [
    "coverage_matched_",
    "permuted_",
    "fixed_",
    "retrieved_text_only",
    "retrieved_labels_only",
    "original_bridge",
]


def save(path, value):
    p = a.storage_path(path).resolve()
    a.require(
        p.is_relative_to(a.storage_path(BASE).resolve()), "Outside review directory"
    )
    a.save_new(p, value)


def jsonlines(path, values):
    path = a.storage_path(path)
    a.require(
        path.resolve().is_relative_to(a.storage_path(BASE).resolve()),
        "Outside review directory",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as handle:
        for value in values:
            handle.write(
                json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            )


def rows(path):
    return list(runtime.stream_rows(path))


def bindings(paths):
    return {runtime.rel(p): c.sha(p) for p in paths}


def load():
    done = a.read(BASE / "assembly.json")
    c.verify(done["outputs"])
    records = rows(BASE / "predictions.jsonl")
    by_run = {}
    metadata = {}
    for row in records:
        ident = (row["condition"], row["repeat"])
        by_run.setdefault(ident, {})[row["key"]] = row["result"]
        metadata[row["key"], *ident] = row
    groups, _ = scoring.scientific_rosters("matched")
    return (groups, by_run, metadata)


def names(runs, family):
    return (
        [k for k in sorted(runs) if k[0].startswith(family)]
        if family.endswith("_")
        else [k for k in sorted(runs) if k[0] == family]
    )


def slug(*parts):
    return "__".join(parts).replace("/", "_").rstrip("_")


def array(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.save(handle, np.asarray(value), allow_pickle=False)


def fraction_mean(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None

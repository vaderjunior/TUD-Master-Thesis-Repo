import io
import json
import os
from pathlib import Path
import uuid

os.environ.update(
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    CUDA_VISIBLE_DEVICES="",
)
import numpy as np
from analysis.common import runtime as c

ROOT = c.ROOT
BASE = c.LINEAR
REVIEW = ROOT / "experiments/astra/sq3_iterative_review_20261026_v1"
OUT = ROOT / "experiments/astra/sq3_cumulative_results_20261027_v1"
VARIANTS = ["all_permitted", "matched_gradient"]
SETTINGS = [(v, t) for v in VARIANTS for t in [0.6, 0.5]]
POLICIES = ["adaptive", "static", "random"]
STREAMS = [2026101201, 2026101202, 2026101203]
CONDITIONS = ["reference"] + [f"{p}_s{s}" for s in STREAMS for p in POLICIES]
read, rows, sha, rel, need = (c.read, c.rows, c.sha, c.rel, c.require)


def clean(v):
    if isinstance(v, np.ndarray):
        return clean(v.tolist())
    if isinstance(v, np.generic):
        return clean(v.item())
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    return v


def write(path, data):
    path = Path(path).resolve()
    need(
        path.is_relative_to(OUT.resolve()) and path != OUT.resolve(),
        "Review output only",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        need(
            path.read_bytes() == data, "Preserve previous review artifact: " + rel(path)
        )
        return
    temp = path.with_name(path.name + ".partial_" + uuid.uuid4().hex)
    with temp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.link(temp, path)
    temp.unlink()


def save(name, value):
    write(OUT / name, c.encoded(clean(value)))


def jsonlines(name, values):
    write(
        OUT / name,
        "".join(
            (
                json.dumps(
                    clean(v), sort_keys=True, ensure_ascii=False, allow_nan=False
                )
                + "\n"
                for v in values
            )
        ).encode("utf8"),
    )


def arrays(name, **values):
    buf = io.BytesIO()
    np.savez_compressed(buf, **values)
    write(OUT / name, buf.getvalue())


def load(name):
    with np.load(OUT / name, allow_pickle=False) as v:
        return {k: v[k] for k in v.files}


def bindings(paths):
    return {rel(p): sha(p) for p in paths}


def contract():
    name = OUT / "analysis_contract.json"
    if name.exists():
        value = read(name)
        c.verify(value["inputs"])
        return value
    files = [
        BASE / n
        for n in [
            "STUDY_PROTOCOL.md",
            "plan.json",
            "data.json",
            "roster.jsonl",
            "static_complete.json",
            "iterative_schedule.json",
            "iterative_complete.json",
        ]
    ]
    files += [
        REVIEW / n
        for n in [
            "part8/classifier_handoff.json",
            "part8/final_verification.json",
            "part2/reader_v3/complete.json",
            "part2/reader_v3/independent_verification.json",
            "part2/reader_v3/matrix.jsonl",
            "part5/metadata.jsonl",
            "part7/metadata.jsonl",
            "part7/execution_arrays.npz",
            "part3/selections.json",
            "part3/source_catalog.json",
            "part3/pool_memberships.jsonl",
        ]
    ]
    value = dict(
        created_utc=c.utc(),
        status="supporting_review_contract_after_fitting_before_comparison_scoring",
        scope="Completed cumulative binary classifier only; combined SQ3 synthesis remains deferred by user.",
        recipe="Original 12 October plan unchanged; no tuning or new fits, embeddings, Qwen calls or writer-owned edits.",
        main="All-permitted TRAIN; strict probability >0.6; supporting matched-gradient TRAIN and >0.5.",
        inference="Supporting nominal 95% paired whole-inference-cluster percentile intervals, 10000 PCG64 draws; no new primary family or multiplicity-adjusted discoveries.",
        weighting="Equal texts within each population/class; 3 feedback streams averaged within text; Qwen reference has 3 actual repeated answers shared across streams, never 9 independent answers.",
        contrasts=[
            "linear baseline",
            "linear updated",
            "Qwen reference",
            "Qwen updated",
            "linear change",
            "Qwen change",
            "linear change minus Qwen change",
            "updated linear minus updated Qwen",
        ],
        cohorts=[
            "round common for each round",
            "all-round common for trajectories",
            "final strict",
            "all-round strict",
            "replay-excluded",
            "one predeclared representative per exact-text cluster",
            "no ties",
            "no repairs",
            "no transport retries",
            "equal cluster weighting",
        ],
        diagnostics=[
            "source and every HateCheck function/template",
            "stream and decoder-repeat views",
            "group omissions",
            "null-output and vote-completion bounds",
            "whole prediction roster descriptive classifier changes",
            "saved-model source adoption (in-sample, not transfer)",
            "probability movement and cutoff crossings",
            "durable fixes, reversals and new errors",
            "binary quality per actual fit before averaging",
            "solver convergence, memory, storage and elapsed resources",
        ],
        cases="On the fixed common cohort, principal recipe/current-error policy: all changed texts plus maximum absolute probability shift, lasting fix, lasting regression, reversal, classifier-correct/Qwen-mostly-wrong and reverse. Maximum shift then SHA key breaks ties. Empty categories stay empty.",
        longer_run="Assess solver convergence and the scientific question; never launch another fit or select settings on TEST.",
        inputs=bindings(files),
    )
    save("analysis_contract.json", value)
    return value

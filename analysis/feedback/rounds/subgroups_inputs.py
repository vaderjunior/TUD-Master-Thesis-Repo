import os

for name in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[name] = "1"
from collections import Counter, defaultdict
import hashlib
import numpy as np
from analysis.feedback.rounds import reader as previous
from analysis.feedback.transfer.subgroups import FUNCTIONS
from analysis.feedback.transfer.subgroups import annotation

ROOT, STUDY, REVIEW, READER = (
    previous.ROOT,
    previous.STUDY,
    previous.REVIEW,
    previous.READER,
)
PART2, PART3, PART4 = previous.OUT, REVIEW / "part3", REVIEW / "part4"
OUT = REVIEW / "part5"
SCOPE = ROOT / "experiments/astra/sq3_followups_20261012_v1/scope"
need, read, rows, sha, rel, pure = (
    previous.need,
    previous.read,
    previous.rows,
    previous.sha,
    previous.rel,
    previous.pure,
)
STREAMS, CONDITIONS = previous.STREAMS, previous.CONDITIONS
POLICIES = ["reference", "adaptive", "static", "random"]
SOURCES = ["hatexplain", "implicit_hate", "mhs"]


def native(v):
    if isinstance(v, np.ndarray):
        return native(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {str(k): native(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [native(x) for x in v]
    return v


def write(path, value):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside new Part 5")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(
            value
            if isinstance(value, bytes)
            else pure.encoded(native(value), pretty=True)
        )


def lines(path, values):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside new Part 5")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        for value in values:
            f.write(pure.encoded(native(value)))


def seed(name):
    return int.from_bytes(
        hashlib.sha256(("A17-part5-support-v1|" + name).encode()).digest()[:8], "little"
    )


def case_rank(group, category, key):
    return hashlib.sha256(
        f"A17-part5-case-v1|{group}|{category}|{key}".encode()
    ).hexdigest()


def definitions(meta):
    groups = {}

    def add(name, ids, family, truths=(False, True)):
        need(name not in groups, "Duplicate group")
        groups[name] = dict(indices=ids, family=family, truths=list(truths))

    for source in SOURCES:
        add(
            "source/" + source,
            [i for i, r in enumerate(meta) if r["source"] == source],
            "source",
        )
    for fn in FUNCTIONS:
        add(
            "function/" + fn,
            [i for i, r in enumerate(meta) if r.get("functionality") == fn],
            "function",
        )
    for source, field in [
        ("hatexplain", "majority_label"),
        ("implicit_hate", "stage1_class"),
        ("mhs", "severity"),
    ]:
        strata = defaultdict(list)
        for i, r in enumerate(meta):
            if r["source"] != source:
                continue
            value = (
                r["majority_label"] + "/" + r["annotation_agreement"]
                if source == "hatexplain"
                else (
                    r["stage1_class"]
                    if source == "implicit_hate"
                    else r["fine_gold"]["severity"]
                )
            )
            if value is not None:
                strata[str(value)].append(i)
        for label, ids in sorted(strata.items()):
            truth = meta[ids[0]]["gold_hate"]
            need(
                all(meta[i]["gold_hate"] == truth for i in ids), "Mixed annotation cell"
            )
            add("annotation/" + source + "/" + label, ids, "annotation", (truth,))
    for field in ["block", "target_identity", "direction"]:
        strata = defaultdict(list)
        for i, r in enumerate(meta):
            if r["source"] == "hatecheck":
                strata[str(r[field])].append(i)
        for label, ids in sorted(strata.items()):
            add("metadata/" + field + "/" + label, ids, "metadata")
    return groups


def load():
    matrix = list(rows(READER / "matrix.jsonl"))
    roster = list(rows(STUDY / "evaluation.jsonl"))
    need(len(matrix) == len(roster) == 2077, "Evaluation roster length")
    meta = []
    for i, (r, m) in enumerate(zip(roster, matrix, strict=True)):
        v = annotation(r)
        for key in [
            "key",
            "population",
            "source",
            "inference_cluster",
            "functional_cluster",
            "text_cluster",
            "overlap_cluster",
            "gold_hate",
        ]:
            need(v[key] == m[key], "Original metadata differs: " + key)
        v.update(input_index=i, text=r["text"], masks=m["masks"])
        need(i == m["input_index"], "Matrix order")
        meta.append(v)
    return meta, matrix, roster

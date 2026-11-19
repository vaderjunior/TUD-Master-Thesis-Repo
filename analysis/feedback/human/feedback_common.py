from copy import deepcopy
from pathlib import Path
import os
import uuid
from analysis.common import runtime as shared

ROOT = shared.ROOT
BASE = ROOT / "experiments/astra/sq3_human_feedback_20261015_v1"
HUMAN = ROOT / "experiments/astra/sq3_human_review_results_20261010_v1"
CORE = shared.CORE
SCOPE = shared.SHARED / "scope"
A15 = shared.A15
CONDITIONS = ("reference5", "dataset_feedback5", "human_feedback5")
SOURCE_ALIASES = ("R014", "R035", "R059")
LABELS = (
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
)
read, rows, sha, sig, rel, utc, require, verify, bindings, norm, offline, forbidden = (
    shared.read,
    shared.rows,
    shared.sha,
    shared.sig,
    shared.rel,
    shared.utc,
    shared.require,
    shared.verify,
    shared.bindings,
    shared.norm,
    shared.offline,
    shared.forbidden,
)


def write(path, data):
    p = Path(path).resolve()
    require(p.is_relative_to(BASE.resolve()), "Write outside human-feedback study")
    data = data.encode("utf8") if isinstance(data, str) else data
    if p.exists():
        require(p.read_bytes() == data, "Preserve existing artifact: " + rel(p))
        return
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".partial_" + uuid.uuid4().hex)
    with tmp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.link(tmp, p)
    tmp.unlink()


def save(path, value):
    write(path, shared.encoded(value))


def jsonlines(path, values):
    import json

    write(
        path,
        "".join(
            (
                json.dumps(x, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
                for x in values
            )
        ),
    )


def agreed_correction(item):

    a, b = (item["judgments"][x + "_project"] for x in ("A", "B"))
    if a["binary_hate"] is None or a["binary_hate"] != b["binary_hate"]:
        return None
    hate = a["binary_hate"]
    types = (
        []
        if not hate
        else (
            sorted(a["types"])
            if a["type_state"] == b["type_state"] == "assigned"
            and set(a["types"]) == set(b["types"])
            else None
        )
    )
    old = item["model_repeat_0"]
    if old["hate"] == hate and (types is None or set(old["hate_type"]) == set(types)):
        return None
    return {
        "gate": hate,
        "hate_types": types,
        "target_groups": None,
        "severity": None,
        "legal": None,
    }


def correction_record(item, arm):
    agreed = agreed_correction(item)
    require(agreed is not None, "Not an agreed model correction")
    meta = deepcopy(agreed)
    if arm == "dataset_feedback5":
        meta["gate"] = item["gold"]["hate"]
        meta["hate_types"] = (
            sorted(item["gold"]["hate_type"])
            if agreed["hate_types"] is not None
            else None
        )
    else:
        require(arm == "human_feedback5", "Unknown correction arm")
    return {
        "id": "human-feedback-" + item["id"],
        "text": item["text"],
        "meta": meta,
        "via": ["frozen_human_feedback_memory"],
    }


def context(base, correction=None):
    out = deepcopy(base)
    groups = out["example_groups"]
    require(
        len(groups) == 1
        and groups[0][0] == "Labelled examples"
        and (len(groups[0][1]) == 5),
        "Expected five ordinary examples",
    )
    require(
        len(out["definitions"]) == len(out["guidelines"]) == 2,
        "Abstract budget differs",
    )
    if correction is not None:
        require(
            correction["id"] not in {x["id"] for x in groups[0][1]},
            "Repeated correction identity",
        )
        groups[0][1] = groups[0][1][:4] + [deepcopy(correction)]
    return out


def reference_target(item, reference):
    if reference == "dataset":
        return (item["gold"]["hate"], sorted(item["gold"]["hate_type"]))
    j = item["judgments"][reference + "_project"]
    gate = j["binary_hate"]
    types = (
        None
        if gate is None
        else (
            []
            if not gate
            else sorted(j["types"]) if j["type_state"] == "assigned" else None
        )
    )
    return (gate, types)


def active_preservation():

    p = read(A15 / "execution/plan.json")
    verify(p["source_hashes"])
    verify(p["input_hashes"])
    require(
        read(A15 / "execution/ready.json")["plan_sha256"]
        == sha(A15 / "execution/plan.json"),
        "Active plan changed",
    )
    recovery = A15 / "execution/recovery_20261014_v1"
    manifest, applied = (
        read(recovery / "manifest.json"),
        read(recovery / "applied.json"),
    )
    verify(manifest["source_hashes"])
    verify(manifest["stop_files"])
    verify(applied["administrative_outcomes"])
    require(len(applied["administrative_outcomes"]) == 8, "Unknown outcomes changed")
    return {
        "execution_plan_sha256": sha(A15 / "execution/plan.json"),
        "source_files": len(p["source_hashes"]),
        "input_files": len(p["input_hashes"]),
        "unknown_outcomes_preserved": 8,
        "new_Qwen_calls": 0,
        "active_prediction_records_read": 0,
    }

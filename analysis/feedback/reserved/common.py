from copy import deepcopy
from pathlib import Path
import hashlib
import json
import os
import uuid
from analysis.feedback.insertion import common as previous

ROOT = previous.ROOT
BASE = ROOT / "experiments/astra/sq3_reserved_slot_20261012_v1"
A14 = previous.BASE
A12 = previous.old.BASE
SELECTION = previous.SELECTION
TOK = previous.old.old.TOK
SEEDS = previous.SEEDS
MODES = ("natural5", "reserved5", "extended6", "masked5")
N_INPUTS, N_CONDITIONS, N_BATCHES = (675, 74, 16)
N_PROMPTS = N_INPUTS * N_CONDITIONS
N_PREDICTIONS = N_PROMPTS * 3
N_CALLS = N_PREDICTIONS * 3
read, rows, sha, sig, rel, utc, require, verify, bindings = (
    previous.read,
    previous.rows,
    previous.sha,
    previous.sig,
    previous.rel,
    previous.utc,
    previous.require,
    previous.verify,
    previous.bindings,
)
offline, forbidden = (previous.offline, previous.forbidden)


def own(path):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE.resolve()), "Write outside owned A15 study")
    return path


def save(path, value):
    previous.a.save_new(own(path), value)


def jsonlines(path, values):

    path = own(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial_" + uuid.uuid4().hex)
    with temp.open("xb") as f:
        for value in values:
            f.write(
                (
                    json.dumps(
                        value, ensure_ascii=False, sort_keys=True, allow_nan=False
                    )
                    + "\n"
                ).encode("utf8")
            )
        f.flush()
        os.fsync(f.fileno())
    if path.exists():
        require(sha(path) == sha(temp), "Existing A15 artifact differs: " + rel(path))
    else:
        os.link(temp, path)
    temp.unlink()


def stream(path):
    with Path(path).open(encoding="utf8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def examples(context):
    require(
        len(context["example_groups"]) == 1
        and context["example_groups"][0][0] == "Labelled examples",
        "Expected ordinary English example group",
    )
    return context["example_groups"][0][1]


def intervene(base, mode, correction=None):

    require(
        mode in ("reference5", "reference4", "reserved5", "extended6", "masked5"),
        "Unknown intervention",
    )
    require(len(examples(base)) == 5, "Base must have five examples")
    require(
        len(base["definitions"]) == len(base["guidelines"]) == 2,
        "Abstract budget changed",
    )
    out = deepcopy(base)
    if mode == "reference5":
        require(correction is None, "Unexpected reference correction")
        return out
    if mode == "reference4":
        require(correction is None, "Unexpected ablation correction")
        out["example_groups"][0][1] = out["example_groups"][0][1][:4]
        return out
    require(
        correction is not None and set(correction) == {"id", "text", "meta", "via"},
        "Correction shape changed",
    )
    meta = correction["meta"]
    require(
        set(meta) == {"gate", "hate_types", "target_groups", "severity", "legal"}
        and type(meta["gate"]) is bool,
        "Binary correction annotation required",
    )
    require(
        all((v is None for k, v in meta.items() if k != "gate")),
        "Fine correction annotation invented",
    )
    require(
        correction["id"] not in {x["id"] for x in examples(base)},
        "Correction duplicates baseline example",
    )
    new = deepcopy(correction)
    if mode == "masked5":
        new["meta"]["gate"] = None
    ordinary_count = 5 if mode == "extended6" else 4
    out["example_groups"][0][1] = out["example_groups"][0][1][:ordinary_count] + [new]
    return out


def condition_list(pools):
    names = [p["condition"] for p in pools]
    require(len(names) == len(set(names)) == 18, "Original 18 pools required")
    return ["reference5", "reference4"] + [
        mode + "__" + name for mode in MODES for name in names
    ]


def input_files():
    return [
        A14 / n
        for n in (
            "complete.json",
            "prepared.json",
            "scientific_plan_v1.json",
            "roster.jsonl",
            "prompts.jsonl",
        )
    ] + [
        A14 / "cold_replay_review_v1/qualification.json",
        A14 / "cold_replay_review_v1/sensitivity_plan.json",
        A12 / "complete.json",
        A12 / "prepared.json",
        A12 / "prompts.jsonl",
        A12 / "choices.jsonl",
        A12 / "independent_verification.json",
        SELECTION / "complete.json",
        SELECTION / "actual_pools.json",
        SELECTION / "gate_only_records.json",
        TOK / "complete.json",
        TOK / "tokenizer.json",
        ROOT / "config/taxonomy.yaml",
        ROOT / "prompts/classify_v1.txt",
        ROOT / "src/hsrag/prompt.py",
        ROOT / "analysis/common/prompt.py",
    ]


def verify_parent(root):
    done = read(root / "complete.json")
    prepared = read(root / "prepared.json")
    require(
        done["status"] == "offline_preparation_independently_verified",
        "Unreviewed parent preparation",
    )
    require(
        done["prepared_sha256"] == sha(root / "prepared.json"),
        "Parent prepared receipt changed",
    )
    require(
        done["verification_sha256"] == sha(ROOT / done["verification"]),
        "Parent verification changed",
    )
    for p in input_files():
        name = rel(p)
        if name in prepared["outputs"]:
            require(
                sha(p) == prepared["outputs"][name],
                "Consumed parent input changed: " + name,
            )


def load_sources():
    pools = read(SELECTION / "actual_pools.json")["pools"]
    records = read(SELECTION / "gate_only_records.json")
    roster = rows(A14 / "roster.jsonl")
    references = {}
    natural = {}
    for p in stream(A14 / "prompts.jsonl"):
        compact = {
            k: p[k]
            for k in (
                "key",
                "condition",
                "context",
                "messages",
                "messages_sha256",
                "text",
            )
        }
        if p["condition"] == "reference":
            references[p["key"]] = compact
        else:
            natural[p["key"], p["condition"]] = compact
    corrections = {}
    for p in stream(A12 / "prompts.jsonl"):
        if p["condition"] == "reference":
            continue
        require(
            len(p["context"]["example_groups"]) == 2, "Parent correction group changed"
        )
        name, group = p["context"]["example_groups"][1]
        require(
            name == "Additional labelled examples" and len(group) == 1,
            "Parent correction count changed",
        )
        corrections[p["key"], p["condition"]] = {
            "hit": group[0],
            "source_id": p["source_id"],
        }
    choices = {(p["key"], p["condition"]): p for p in stream(A12 / "choices.jsonl")}
    require(
        len(roster) == len(references) == N_INPUTS
        and len(natural) == len(corrections) == len(choices) == N_INPUTS * 18,
        "Incomplete parent prompt matrix",
    )
    return (roster, pools, records, references, natural, corrections, choices)

from __future__ import annotations

from src.hsrag.paths import PROJECT_ROOT

import hashlib

import json

from pathlib import Path

ROOT = PROJECT_ROOT

CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"

BASE = CORE / "encoders_v2"

PLAN = BASE / "execution.json"

SELF = Path(__file__).resolve()

CHECK = (
    ROOT / "Thesis_Log/astra_result/evidence/core_preparation_check_20260915_v1.json"
)

CORE_SHA = "5c610f4fe320b5e0f185315933d47b388ab645b20b52eed738541a1579184e4e"

CHECK_SHA = "00e8e15f5821aa9cc505f2f86ca43971d3705fddda81ce785336d66822fbe1c0"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def load_model(path, meta, base=False):
    import torch
    from transformers import (
        RobertaConfig,
        RobertaForSequenceClassification,
        RobertaTokenizerFast,
        BertConfig,
        BertForSequenceClassification,
        BertTokenizerFast,
    )

    if meta["lang"] == "de":
        C, M, T = (BertConfig, BertForSequenceClassification, BertTokenizerFast)
    else:
        C, M, T = (
            RobertaConfig,
            RobertaForSequenceClassification,
            RobertaTokenizerFast,
        )
    tok = T.from_pretrained(str(path), local_files_only=True)
    config = C.from_pretrained(str(path), local_files_only=True)
    if base:
        config.num_labels = len(meta["labels"])
        config.id2label = dict(enumerate(meta["labels"]))
        config.label2id = {label: i for i, label in enumerate(meta["labels"])}
        config.problem_type = (
            "multi_label_classification"
            if meta["kind"] == "multilabel"
            else "single_label_classification"
        )
    require(config.num_labels == len(meta["labels"]), "Checkpoint output size differs")
    model, info = M.from_pretrained(
        str(path),
        config=config,
        local_files_only=True,
        use_safetensors=True,
        attn_implementation="sdpa",
        dtype=torch.float32,
        output_loading_info=True,
    )
    require(
        not info.get("mismatched_keys") and (not info.get("error_msgs")),
        "Pretrained tensor mismatch",
    )
    if base:
        expected = (
            {"classifier.weight", "classifier.bias"}
            if meta["lang"] == "de"
            else {
                "classifier.dense.weight",
                "classifier.dense.bias",
                "classifier.out_proj.weight",
                "classifier.out_proj.bias",
            }
        )
        require(
            set(info.get("missing_keys", [])) == expected,
            f"Unexpected missing weights: {info}",
        )
        require(
            all(
                (
                    k.startswith(("cls.", "lm_head."))
                    or k.endswith(("position_ids", "token_type_ids"))
                    for k in info.get("unexpected_keys", [])
                )
            ),
            f"Unexpected base weights: {info}",
        )
    else:
        require(
            not info.get("missing_keys") and (not info.get("unexpected_keys")),
            f"Checkpoint did not reload cleanly: {info}",
        )
    require(
        all((p.dtype == torch.float32 for p in model.parameters())),
        "Parameters are not FP32",
    )
    return (tok, model, info)


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def own(p):
    p = Path(p).resolve()
    require(
        p.is_relative_to(BASE.resolve()) and p != BASE.resolve(),
        "Output outside isolated encoder experiment",
    )
    return p


def verify_hashes(hashes, output=False):
    for name, expected in hashes.items():
        p = ROOT / name
        if output:
            own(p)
        require(sha(p) == expected, f"File changed: {name}")


def validate_core():
    require(
        sha(CORE / "plan.json") == CORE_SHA and sha(CHECK) == CHECK_SHA,
        "Core preparation identity changed",
    )
    cp = read(CORE / "plan.json")
    verify_hashes(read(CHECK)["input_hashes"])
    from analysis.knowledge.banks.build import environment

    require(environment() == cp["environment"], "Project package environment changed")
    for cache in cp["base_caches"].values():
        for name, h in cache["files"].items():
            require(
                sha(Path(cache["snapshot"]) / name) == h,
                f"Cached base file changed: {name}",
            )
    return cp


def validate():
    validate_core()
    plan = read(PLAN)
    require(plan["core_plan_sha256"] == CORE_SHA, "Execution/core mismatch")
    verify_hashes(plan["input_hashes"])
    return plan


def completion(attempt):
    attempt = own(attempt)
    r = read(attempt / "complete.json")
    start = read(attempt / "start.json")
    require(
        r["status"] == "complete" and r["execution_sha256"] == sha(PLAN),
        "Wrong completion plan",
    )
    require(
        r["job"] == start["job"] and r["mode"] == start["mode"],
        "Completion/start mismatch",
    )
    require(
        start["execution_sha256"] == sha(PLAN)
        and sha(attempt / "source_snapshot.py") == sha(SELF),
        "Attempt source changed",
    )
    require(not (attempt / "failure.json").exists(), "Conflicting completion/failure")
    verify_hashes(r["files"], output=True)
    required = {
        "internal_val_logits.jsonl",
        "verification.json",
        "start.json",
        "source_snapshot.py",
    }
    if r["mode"] != "reuse":
        required.update(
            {
                "training.json",
                "selected_model/model.safetensors",
                "selected_model/meta.json",
            }
        )
    if r["mode"] != "smoke":
        required.add("dev_head_logits.jsonl")
    require(
        {rel(attempt / p) for p in required} <= set(r["files"]),
        "Missing completion artifacts",
    )
    return r

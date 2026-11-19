"Shared RoBERTa loading, loss and inference functions."

from __future__ import annotations
import gc
import math
import shutil
import time
import traceback
import uuid
from pathlib import Path
from training.roberta import common as c
from training.roberta import tokenizer as guard


def setup():
    import pyarrow.dataset, pyarrow.parquet, pandas
    import torch

    c.require(
        torch.cuda.is_available(),
        "CUDA unavailable. Enable the NVIDIA GPU and rerun; no CPU fallback.",
    )
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info()
    c.require(
        free >= 5 * 2**30, "Need at least 5 GiB free GPU memory; close other CUDA work."
    )
    return torch


def load(cache, spec, seed, device="cuda", trained=False):
    import torch
    from transformers import RobertaConfig, RobertaForSequenceClassification, set_seed
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()
    hf_logging.disable_progress_bar()
    set_seed(seed)
    tok = guard.load(cache)
    config = RobertaConfig.from_pretrained(str(cache), local_files_only=True)
    if not trained:
        config.num_labels = len(spec["labels"])
        config.id2label = dict(enumerate(spec["labels"]))
        config.label2id = {label: i for i, label in enumerate(spec["labels"])}
        config.problem_type = (
            "multi_label_classification"
            if spec["kind"] == "multilabel"
            else "single_label_classification"
        )
    c.require(config.num_labels == len(spec["labels"]), "Classifier dimension changed")
    c.require(
        config.vocab_size == 50265
        and (config.bos_token_id, config.pad_token_id, config.eos_token_id)
        == (0, 1, 2),
        "Model vocabulary/special IDs differ",
    )
    c.require(
        [config.id2label[i] for i in range(config.num_labels)] == spec["labels"],
        "Model label order differs",
    )
    model, info = RobertaForSequenceClassification.from_pretrained(
        str(cache),
        config=config,
        local_files_only=True,
        use_safetensors=True,
        attn_implementation="sdpa",
        dtype=torch.float32,
        output_loading_info=True,
    )
    c.require(
        not info.get("mismatched_keys") and not info.get("error_msgs"),
        "Checkpoint tensor mismatch",
    )
    expected = {
        "classifier.dense.weight",
        "classifier.dense.bias",
        "classifier.out_proj.weight",
        "classifier.out_proj.bias",
    }
    if trained:
        c.require(
            not info.get("missing_keys") and not info.get("unexpected_keys"),
            "Trained checkpoint does not reload cleanly",
        )
    else:
        c.require(
            set(info.get("missing_keys", [])) == expected,
            "Unexpected missing pretrained weights",
        )
        c.require(
            all(
                k.startswith("lm_head.")
                or k.endswith(("position_ids", "token_type_ids"))
                for k in info.get("unexpected_keys", [])
            ),
            "Unexpected pretrained weights",
        )
    model.to(device)
    c.require(
        all(p.dtype == torch.float32 for p in model.parameters()), "Non-FP32 parameters"
    )
    return tok, model, c.ready(info)


def loss_value(logits, labels, head, weight_info):
    import torch
    import torch.nn.functional as F

    z = logits.float()
    weights = torch.as_tensor(
        weight_info["weights"], dtype=torch.float32, device=z.device
    )
    if head in ("hate_type", "target_group"):
        y = labels.float()
        term = F.binary_cross_entropy_with_logits(z, y, reduction="none")
        return (term * torch.where(y.bool(), weights[:, 1], weights[:, 0])).mean()
    return (
        F.cross_entropy(z, labels.long(), reduction="none") * weights[labels.long()]
    ).mean()


def tokens(tok, texts):
    enc = guard.encode(tok, texts)
    return [
        dict(input_ids=ids, attention_mask=mask)
        for ids, mask in zip(enc["input_ids"], enc["attention_mask"])
    ]


def batch(rows, device):
    import torch

    length = max(len(r["input_ids"]) for r in rows)
    return dict(
        input_ids=torch.tensor(
            [r["input_ids"] + [1] * (length - len(r["input_ids"])) for r in rows],
            dtype=torch.long,
            device=device,
        ),
        attention_mask=torch.tensor(
            [r["attention_mask"] + [0] * (length - len(r["input_ids"])) for r in rows],
            dtype=torch.long,
            device=device,
        ),
    )


def infer(model, encoded, device="cuda", microbatch=32):
    import torch
    import numpy as np

    model.eval()
    values = []
    with torch.inference_mode():
        for start in range(0, len(encoded), microbatch):
            logits = model(
                **batch(encoded[start : start + microbatch], device)
            ).logits.float()
            c.require(bool(torch.isfinite(logits).all()), "Nonfinite inference logits")
            values.append(logits.cpu().numpy())
    return np.concatenate(values)


def split(data, job):
    spec = data["heads"][job["head"]]
    folds = data["folds"]["42"]
    fit = [
        r
        for r in spec["rows"]
        if job["fold"] is None or folds[r["feature_row"]] != job["fold"]
    ]
    held = [
        r
        for r in spec["rows"]
        if job["fold"] is not None and folds[r["feature_row"]] == job["fold"]
    ]
    c.require(fit and (held or job["mode"] == "final"), "Empty fitting/held partition")
    c.require(
        not {r["cluster"] for r in fit} & {r["cluster"] for r in held},
        "Held group in fit",
    )
    return spec, fit, held

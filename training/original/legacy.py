from __future__ import annotations
from src.hsrag.paths import PROJECT_ROOT
import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import uuid

ROOT = PROJECT_ROOT
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
BASE = CORE / "encoders_v4"
PLAN = BASE / "execution.json"
SELF = Path(__file__).resolve()
LAUNCHER = ROOT / "scripts/run_legacy_encoder.ps1"
TEST = ROOT / "tests/training/original/test_legacy.py"
CHECK = (
    ROOT / "Thesis_Log/astra_result/evidence/core_preparation_check_20260915_v1.json"
)
CORE_SHA = "5c610f4fe320b5e0f185315933d47b388ab645b20b52eed738541a1579184e4e"
CHECK_SHA = "00e8e15f5821aa9cc505f2f86ca43971d3705fddda81ce785336d66822fbe1c0"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


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


def now():
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def own(p):
    p = Path(p).resolve()
    require(
        p.is_relative_to(BASE.resolve()) and p != BASE.resolve(),
        "Output outside isolated encoder experiment",
    )
    return p


def json_ready(obj):

    if isinstance(obj, dict):
        return {k: json_ready(v) for k, v in obj.items()}
    if isinstance(obj, set):
        return [json_ready(v) for v in sorted(obj, key=repr)]
    if isinstance(obj, (tuple, list)):
        return [json_ready(v) for v in obj]
    return obj


def write(p, obj):
    p = own(p)
    data = (
        json.dumps(
            json_ready(obj),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf-8", newline="\n") as f:
        f.write(data)


def lines(p, rows):
    p = own(p)
    with p.open("x", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n")


def identity(job):
    return f"{job['arm']}/{job['head']}/seed{job['seed']}"


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


def prepare():
    if PLAN.exists():
        validate()
        print("Existing execution plan verified; no model work.")
        return
    cp = validate_core()
    ep = read(CORE / "encoder_plan.json")
    jobs = ep["train_jobs"]
    require(len(jobs) == 21 and len(ep["weight_reuse"]) == 9, "Unexpected scope")
    from training.original import validation as previous_runner

    previous_runner.validate()
    inherited = previous_runner.done_jobs()
    require(
        set(inherited) == {("smoke", "de/hate/seed0"), ("smoke", "de/legal/seed0")},
        "Expected passed v3 German checks",
    )
    inherited_rows = [
        dict(e, runner="v2") for e in read(previous_runner.PLAN)["inherited_smokes"]
    ]
    inherited_rows.append(
        {
            "mode": "smoke",
            "job": "de/legal/seed0",
            "runner": "v3",
            **inherited["smoke", "de/legal/seed0"],
        }
    )
    smoke = [
        j for j in jobs if j["seed"] == 0 and j["arm"] == "de" and (j["head"] == "hate")
    ]
    plan = {
        "created_utc": now(),
        "core_plan_sha256": CORE_SHA,
        "input_hashes": {
            rel(p): sha(p)
            for p in (SELF, TEST, LAUNCHER, CORE / "encoder_plan.json", CHECK)
        },
        "train_jobs": jobs,
        "reuse_jobs": ep["weight_reuse"],
        "smoke_jobs": smoke,
        "inherited_smokes": inherited_rows,
        "base_caches": cp["base_caches"],
        "precision": "FP16 AMP training with FP32 master/stored parameters; explicit FP32 standalone inference; SDPA attention.",
        "optimizer": "adamw_torch_fused, matching the installed original trainer default; clipping 1.0; no weighting/resampling.",
        "checkpoints": "Full/DE: every epoch, weights only, preserve all. Compact gate: fixed final epoch only; no checkpoint selection. No optimizer state or within-job resume.",
        "resume": "Verify completed jobs and skip. Restart interrupted jobs from pinned base and same seed in a fresh preserved attempt. One batch/worker at a time.",
        "storage": "Approximately 60 GiB for uninterrupted training plus all seven smoke checks; require 75 GiB free at batch start, 8 GiB per worker. Failed attempts add storage.",
        "smoke": "Retain five English v2 two-step checks and German legal v3 eight-step check. Repeat German gate eight-step diagnostic only to verify loss-scale telemetry. Same first 32 TRAIN/internal-val rows and padding 128. No full recipe/precision change or DEV inference.",
        "numerics": "Finite loss/logits every batch. Track both Accelerate flag and GradScaler scale decreases for AMP skips, because fused optimizer may execute its Python step while its kernel skips the update. Stop after more than 20 consecutive skips. Preserve the installed Trainer learning-rate schedule; no optimizer intervention. Finite final parameters, changed classifier and FP32 reload atol=rtol=1e-5 required.",
        "dev_outputs": "Raw per-head logits/decisions only on all language-matched frozen DEV texts. Reused weights are loaded read-only; full predictions with updated gates assembled and scored later.",
        "scope": "No TEST/HateCheck gold or evaluation, no API, no active KB edits. Priority 3 and no-grievance adaptation parked.",
        "historical_limits": "Original trainer source hash is unavailable. Preserve qualified fixed-epoch reuse and old validation reports. Fresh core results are post-selection development evidence.",
    }
    previous = CORE / "encoders_v3"
    require((previous / "execution.json").exists(), "Missing preserved v3 execution")
    plan["supersedes"] = (
        "All seven v3/v2 diagnostics passed, but Accelerate reported zero skipped steps despite fused-optimizer FP16 overflows. v4 adds loss-scale-decrease accounting and a complete scale trace. This fixes telemetry/abort guard without changing the optimizer, schedule, data, precision or selection recipe. Earlier versions and failures preserved."
    )
    plan["input_hashes"].update(
        {
            rel(p): sha(p)
            for p in previous.rglob("*")
            if p.is_file() and p.name not in (".batch.lock", ".gpu.lock")
        }
    )
    write(PLAN, plan)
    print(
        json.dumps(
            {
                "prepared": rel(PLAN),
                "sha256": sha(PLAN),
                "jobs": len(jobs),
                "reuse": 9,
                "smokes": len(smoke),
            },
            indent=2,
        )
    )


def validate():
    validate_core()
    plan = read(PLAN)
    require(plan["core_plan_sha256"] == CORE_SHA, "Execution/core mismatch")
    verify_hashes(plan["input_hashes"])
    from training.original import validation as previous_runner
    from training.original import loading as english_runner

    previous_runner.validate()
    for entry in plan["inherited_smokes"]:
        r = (english_runner if entry["runner"] == "v2" else previous_runner).completion(
            ROOT / entry["attempt"]
        )
        require(
            r["mode"] == "smoke"
            and r["job"] == entry["job"]
            and (
                sha(ROOT / entry["attempt"] / "complete.json")
                == entry["receipt_sha256"]
            ),
            "Inherited smoke changed",
        )
    return plan


@contextmanager
def lock(name):
    import msvcrt

    p = own(BASE / name)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a+b") as f:
        if p.stat().st_size == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError(
                "Another encoder batch/worker is active. Keep only one copy running."
            ) from exc
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def setup_cuda():
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
    )
    import torch

    require(
        torch.cuda.is_available(),
        "CUDA unavailable. Enable NVIDIA GPU and rerun; no CPU fallback.",
    )
    free, total = torch.cuda.mem_get_info()
    require(
        free >= 5 * 2**30,
        "Need at least 5 GiB free GPU memory. Close other GPU work and retry.",
    )
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    print(
        f"GPU: {torch.cuda.get_device_name(0)}, {free / 2 ** 30:.2f}/{total / 2 ** 30:.2f} GiB free",
        flush=True,
    )
    return torch


def data_for(job, smoke=False):
    import pandas as pd
    from training.encoder import GOLD_COL
    from training.encoder import clean
    from training.encoder import encode_targets
    from training.encoder import taxonomy_labels

    d = CORE / "encoder_data" / job["arm"]
    meta = read(d / f"{job['head']}_meta.json")
    require(
        meta["labels"] == taxonomy_labels(job["head"]), "Taxonomy label order changed"
    )
    frames, targets = ({}, {})
    for split in ("train", "val"):
        df = pd.read_parquet(d / f"{job['head']}_{split}.parquet")
        require(len(df) == meta["counts"][split], "Partition size mismatch")
        if smoke:
            df = df.iloc[:32].copy()
        values = [clean(v) for v in df[GOLD_COL[job["head"]]]]
        require(
            all((v is not None for v in values)),
            "Missing annotated target in encoder partition",
        )
        if meta["kind"] == "binary":
            require(all((v in (True, False) for v in values)), "Nonbinary gate target")
        frames[split] = df
        targets[split] = encode_targets(values, meta["kind"], meta["labels"])
    return (meta, frames, targets)


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


def infer(model, tok, texts, batch_size=32):
    import numpy as np
    import torch

    result = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(texts), batch_size):
            batch = tok(
                texts[i : i + batch_size],
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            )
            batch = {
                k: v.to("cuda")
                for k, v in batch.items()
                if k in ("input_ids", "attention_mask")
            }
            out = model(**batch).logits.float().cpu().numpy()
            require(np.isfinite(out).all(), "Nonfinite inference logits")
            result.append(out)
    return np.concatenate(result)


def decisions(logits, kind):
    return (logits >= 0).astype(int) if kind == "multilabel" else logits.argmax(-1)


def smoke_steps(job):
    return 8 if job["arm"] == "de" else 2


def overflow_skipped(scale_before, scale_after, accelerate_flag):
    require(
        math.isfinite(scale_before)
        and math.isfinite(scale_after)
        and (scale_before > 0)
        and (scale_after > 0),
        "Invalid AMP loss scale",
    )
    return bool(accelerate_flag or scale_after < scale_before)


def arguments(job, output, smoke=False):
    from transformers import TrainingArguments

    early = job["arm"] != "kb"
    return TrainingArguments(
        output_dir=str(output),
        seed=job["seed"],
        data_seed=job["seed"],
        num_train_epochs=job["epochs"],
        max_steps=smoke_steps(job) if smoke else -1,
        per_device_train_batch_size=job["batch_size"],
        per_device_eval_batch_size=job["eval_batch_size"],
        learning_rate=job["learning_rate"],
        warmup_ratio=job["warmup_ratio"],
        weight_decay=job["weight_decay"],
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        optim="adamw_torch_fused",
        fp16=True,
        bf16=False,
        tf32=False,
        eval_strategy="steps" if smoke else "epoch",
        eval_steps=smoke_steps(job) if smoke else None,
        save_strategy=("steps" if smoke else "epoch") if early else "no",
        save_steps=smoke_steps(job) if smoke else 500,
        load_best_model_at_end=early,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=None,
        save_only_model=True,
        logging_steps=1 if smoke else 50,
        logging_nan_inf_filter=False,
        report_to=[],
        disable_tqdm=True,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
        torch_compile=False,
    )


def train(attempt, job, smoke, cp):
    import numpy as np

    torch = setup_cuda()
    from transformers import (
        Trainer,
        TrainerCallback,
        DataCollatorWithPadding,
        EarlyStoppingCallback,
        set_seed,
    )
    from training.encoder import macro_f1

    meta, frames, targets = data_for(job, smoke)
    set_seed(job["seed"])
    cache = cp["base_caches"][job["base_model"]]
    tok, model, info = load_model(cache["snapshot"], meta, base=True)
    require(model.config._attn_implementation == "sdpa", "Attention backend changed")
    model.to("cuda")
    initial = {
        k: v.detach().cpu().clone() for k, v in model.classifier.named_parameters()
    }
    token_stats = {}

    class Dataset(torch.utils.data.Dataset):

        def __init__(self, split):
            self.enc = tok(
                frames[split].text.astype(str).tolist(),
                truncation=True,
                max_length=128,
                padding=False,
            )
            self.y = targets[split]
            token_stats[split] = {
                "rows": len(self.y),
                "at_max_length": sum((len(x) == 128 for x in self.enc["input_ids"])),
            }

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return {
                **{k: self.enc[k][i] for k in ("input_ids", "attention_mask")},
                "labels": self.y[i],
            }

    class CheckedTrainer(Trainer):

        def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
        ):
            loss, outputs = super().compute_loss(
                model,
                inputs,
                return_outputs=True,
                num_items_in_batch=num_items_in_batch,
            )
            require(
                bool(torch.isfinite(loss).all()), "Nonfinite training/evaluation loss"
            )
            require(
                bool(torch.isfinite(outputs.logits).all()),
                "Nonfinite training/evaluation logits",
            )
            return (loss, outputs) if return_outputs else loss

    class Monitor(TrainerCallback):

        def __init__(self):
            self.skipped = 0
            self.consecutive = 0
            self.grad_norm_nonfinite = 0
            self.scale_before = None
            self.scale_trace = []

        def on_log(self, args, state, control, logs=None, **kwargs):
            for k, v in (logs or {}).items():
                if isinstance(v, (float, int)) and (not math.isfinite(v)):
                    if k == "grad_norm":
                        self.grad_norm_nonfinite += 1
                    else:
                        raise RuntimeError(f"Nonfinite logged {k}")

        def on_pre_optimizer_step(self, args, state, control, **kwargs):
            self.scale_before = float(trainer.accelerator.scaler.get_scale())

        def on_optimizer_step(self, args, state, control, **kwargs):
            after = float(trainer.accelerator.scaler.get_scale())
            flag = bool(trainer.accelerator.optimizer_step_was_skipped)
            skipped = overflow_skipped(self.scale_before, after, flag)
            self.scale_trace.append(
                {
                    "step": state.global_step + 1,
                    "scale_before": self.scale_before,
                    "scale_after": after,
                    "accelerate_skip_flag": flag,
                    "overflow_skipped": skipped,
                }
            )
            self.skipped += int(skipped)
            self.consecutive = self.consecutive + 1 if skipped else 0
            require(
                self.consecutive <= 20, "More than 20 consecutive AMP-skipped updates"
            )

        def on_save(self, args, state, control, **kwargs):
            print(
                f"Saved epoch {state.epoch:.2f}; selected internal-val F1={state.best_metric}",
                flush=True,
            )

    monitor = Monitor()
    trainer = CheckedTrainer(
        model=model,
        args=arguments(job, own(attempt / "checkpoints"), smoke),
        train_dataset=Dataset("train"),
        eval_dataset=Dataset("val"),
        processing_class=tok,
        data_collator=DataCollatorWithPadding(
            tok,
            padding="max_length" if smoke else True,
            max_length=128 if smoke else None,
        ),
        compute_metrics=lambda p: {
            "macro_f1": macro_f1(
                p.label_ids,
                decisions(p.predictions, meta["kind"]),
                meta["kind"] == "multilabel",
                len(meta["labels"]),
            )
        },
        callbacks=[monitor]
        + (
            [EarlyStoppingCallback(early_stopping_patience=2)]
            if job["arm"] != "kb"
            else []
        ),
    )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    output = trainer.train()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    require(math.isfinite(output.training_loss), "Nonfinite final training loss")
    require(
        (
            trainer.state.global_step == smoke_steps(job)
            if smoke
            else trainer.state.global_step > 2
        ),
        "Wrong step count",
    )
    require(
        monitor.skipped < trainer.state.global_step, "No successful optimizer update"
    )
    model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
    require(not hasattr(model, "_original_forward"), "AMP forward wrapper remains")
    require(
        all((bool(torch.isfinite(p).all()) for p in model.parameters())),
        "Nonfinite final model",
    )
    changed = max(
        (
            float((p.detach().cpu() - initial[k]).abs().max())
            for k, p in model.classifier.named_parameters()
        )
    )
    require(changed > 0, "Selected classifier did not update")
    selected = own(attempt / "selected_model")
    trainer.save_model(str(selected))
    tok.save_pretrained(str(selected))
    state = (
        trainer.state.to_dict()
        if hasattr(trainer.state, "to_dict")
        else __import__("dataclasses").asdict(trainer.state)
    )
    state = json.loads(json.dumps(state), parse_constant=lambda s: s)
    if job["arm"] != "kb":
        best = own(Path(trainer.state.best_model_checkpoint))
        require(
            sha(best / "model.safetensors") == sha(selected / "model.safetensors"),
            "Selected export differs from best checkpoint",
        )
    else:
        require(
            trainer.state.best_model_checkpoint is None,
            "Compact head unexpectedly selected by validation",
        )
    probe_text = frames["val"].text.astype(str).tolist()[:8]
    probe = infer(model, tok, probe_text, batch_size=8)
    result = {
        "job": job,
        "diagnostic_only": smoke,
        "train_seconds": elapsed,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "gpu": torch.cuda.get_device_name(0),
        "model_class": type(model).__name__,
        "tokenizer_class": type(tok).__name__,
        "attention": model.config._attn_implementation,
        "loading_info": info,
        "token_stats": token_stats,
        "amp_skipped_steps": monitor.skipped,
        "amp_scale_trace": monitor.scale_trace,
        "nonfinite_logged_gradient_norms": monitor.grad_norm_nonfinite,
        "classifier_max_change": changed,
        "training_loss": output.training_loss,
        "trainer_state": state,
        "probe_ids": frames["val"].id.astype(str).tolist()[:8],
        "probe_logits": probe.tolist(),
    }
    model_meta = {
        **{k: meta[k] for k in ("arm", "head", "kind", "lang", "labels")},
        "seed": job["seed"],
        "max_seq_length": 128,
        "threshold": 0.5,
        "precision": "fp16 training; fp32 standalone inference",
        "base_model": job["base_model"],
        "revision": cache["revision"],
        "execution_sha256": sha(PLAN),
        "n_train": len(frames["train"]),
        "n_val": len(frames["val"]),
        "diagnostic_only": smoke,
    }
    write(selected / "meta.json", model_meta)
    write(attempt / "training.json", result)
    del trainer, model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return (meta, frames, targets, result)


def predict(attempt, job, mode, cp):
    import numpy as np

    torch = setup_cuda()
    from training.encoder import macro_f1

    if mode == "reuse":
        meta, frames, targets = data_for(job)
        selected = ROOT / job["path"]
        historical = read(selected / "meta.json")
        require(
            historical["labels"] == meta["labels"]
            and historical["kind"] == meta["kind"]
            and (historical["lang"] == meta["lang"]),
            "Reused checkpoint metadata mismatch",
        )
        require(
            historical["seed"] == job["seed"]
            and historical["epochs"] == 20
            and (historical["early_stopping"] is False),
            "Wrong reused recipe",
        )
        training = None
    else:
        meta, frames, targets, training = train(attempt, job, mode == "smoke", cp)
        selected = attempt / "selected_model"
    tok, model, info = load_model(selected, meta)
    model.to("cuda")
    t0 = time.perf_counter()
    probe = infer(model, tok, frames["val"].text.astype(str).tolist()[:8], 8)
    difference = None
    if training is not None:
        difference = float(np.abs(probe - np.asarray(training["probe_logits"])).max())
        require(
            np.allclose(probe, training["probe_logits"], atol=1e-05, rtol=1e-05),
            "FP32 save/reload probe differs",
        )
    logits = infer(model, tok, frames["val"].text.astype(str).tolist())
    pred = decisions(logits, meta["kind"])
    val_rows = [
        {
            "id": str(row.id),
            "logits": log.tolist(),
            "target": y.tolist(),
            "decision": d.tolist(),
        }
        for row, log, y, d in zip(
            frames["val"].itertuples(index=False), logits, targets["val"], pred
        )
    ]
    lines(attempt / "internal_val_logits.jsonl", val_rows)
    score = macro_f1(
        targets["val"], pred, meta["kind"] == "multilabel", len(meta["labels"])
    )
    dev_rows = []
    if mode != "smoke":
        inputs = [
            json.loads(x)
            for x in (CORE / "dev_inputs.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if x.strip()
        ]
        inputs = [x for x in inputs if x["lang"] == meta["lang"]]
        logits = infer(model, tok, [x["text"] for x in inputs])
        pred = decisions(logits, meta["kind"])
        checkpoint_hash = sha(selected / "model.safetensors")
        dev_rows = [
            {
                "key": r["key"],
                "text_sha256": hashlib.sha256(r["text"].encode("utf-8")).hexdigest(),
                "logits": log.tolist(),
                "decision": d.tolist(),
                "labels": meta["labels"],
                "job": identity(job),
                "execution_sha256": sha(PLAN),
                "checkpoint": rel(selected),
                "checkpoint_sha256": checkpoint_hash,
            }
            for r, log, d in zip(inputs, logits, pred)
        ]
        require(
            len(dev_rows) == (325 if meta["lang"] == "de" else 970), "Wrong DEV count"
        )
        lines(attempt / "dev_head_logits.jsonl", dev_rows)
    write(
        attempt / "verification.json",
        {
            "job": identity(job),
            "mode": mode,
            "reload_max_difference": difference,
            "reload_loading_info": info,
            "internal_val_macro_f1_fp32": score,
            "internal_val_rows": len(val_rows),
            "dev_rows": len(dev_rows),
            "inference_seconds": time.perf_counter() - t0,
            "checkpoint": rel(selected),
            "checkpoint_sha256": sha(selected / "model.safetensors"),
            "note": "Validation is TRAIN-internal. FP32 score may differ from AMP selection score. DEV raw logits are not scored here.",
        },
    )
    return len(dev_rows)


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


def done_jobs():
    done = {}
    for p in sorted((BASE / "attempts").glob("*/complete.json")):
        r = completion(p.parent)
        key = (r["mode"], r["job"])
        require(key not in done, "Duplicate successful job; inspect rather than select")
        done[key] = {"attempt": rel(p.parent), "receipt_sha256": sha(p)}
    return done


def worker(attempt):
    attempt = own(attempt)
    require(
        attempt.parent == BASE / "attempts", "Worker must target an attempt directory"
    )
    try:
        plan = validate()
        start = read(attempt / "start.json")
        mode = start["mode"]
        require(
            start["execution_sha256"] == sha(PLAN)
            and sha(attempt / "source_snapshot.py") == sha(SELF),
            "Worker provenance mismatch",
        )
        require(
            not (attempt / "complete.json").exists()
            and (not (attempt / "failure.json").exists()),
            "Existing attempt cannot be restarted",
        )
        group = {"smoke": "smoke_jobs", "full": "train_jobs", "reuse": "reuse_jobs"}[
            mode
        ]
        job = next((j for j in plan[group] if identity(j) == start["job"]))
        with lock(".gpu.lock"):
            require(
                shutil.disk_usage(BASE).free >= 8 * 2**30,
                "Need 8 GiB free disk for worker",
            )
            n = predict(attempt, job, mode, read(CORE / "plan.json"))
            validate()
            files = {
                rel(p): sha(p)
                for p in attempt.rglob("*")
                if p.is_file()
                and p.name not in ("console.log", "complete.json", "failure.json")
            }
            write(
                attempt / "complete.json",
                {
                    "status": "complete",
                    "job": identity(job),
                    "mode": mode,
                    "completed_utc": now(),
                    "execution_sha256": sha(PLAN),
                    "dev_rows": n,
                    "files": files,
                },
            )
            print(
                json.dumps(
                    {
                        "status": "complete",
                        "mode": mode,
                        "job": identity(job),
                        "dev_rows": n,
                        "attempt": rel(attempt),
                    }
                ),
                flush=True,
            )
    except BaseException as exc:
        if not (attempt / "complete.json").exists() and (
            not (attempt / "failure.json").exists()
        ):
            write(
                attempt / "failure.json",
                {
                    "failed_utc": now(),
                    "error": str(exc),
                    "type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                },
            )
        raise


def launch(mode, job):
    attempt = own(
        BASE
        / "attempts"
        / f"{time.time_ns()}_{mode}_{identity(job).replace('/', '_')}_{uuid.uuid4().hex[:6]}"
    )
    attempt.mkdir(parents=True, exist_ok=False)
    with (attempt / "source_snapshot.py").open("xb") as f:
        f.write(SELF.read_bytes())
    write(
        attempt / "start.json",
        {
            "mode": mode,
            "job": identity(job),
            "started_utc": now(),
            "execution_sha256": sha(PLAN),
        },
    )
    cmd = [
        sys.executable,
        "-X",
        "utf8",
        "-B",
        "-u",
        "-m",
        "training.original.legacy",
        "worker",
        "--attempt",
        str(attempt),
    ]
    print(
        f"Starting {mode} {identity(job)}; log {rel(attempt / 'console.log')}",
        flush=True,
    )
    with (attempt / "console.log").open("x", encoding="utf-8") as f:
        process = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        try:
            for line in process.stdout:
                f.write(line)
                f.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            process.terminate()
            process.wait()
            raise
    require(
        code == 0,
        f"Worker stopped; preserved {rel(attempt)}. Ask Astra to inspect before retry.",
    )
    completion(attempt)


def run(smoke_only=False):
    plan = validate()
    with lock(".batch.lock"):
        done = done_jobs()
        needed = [("smoke", j) for j in plan["smoke_jobs"]]
        if not smoke_only:
            needed += [("full", j) for j in plan["train_jobs"]] + [
                ("reuse", j) for j in plan["reuse_jobs"]
            ]
        remaining = [(m, j) for m, j in needed if (m, identity(j)) not in done]
        if remaining:
            require(
                shutil.disk_usage(BASE).free >= (15 if smoke_only else 75) * 2**30,
                "Insufficient free disk: need 15 GiB for smoke, 75 GiB for full batch",
            )
        for mode, job in needed:
            if (mode, identity(job)) in done:
                print(f"Verified and skipping {mode} {identity(job)}", flush=True)
            else:
                launch(mode, job)
        done = done_jobs()
        require(all(((m, identity(j)) in done for m, j in needed)), "Incomplete batch")
        write(
            BASE / "batch_receipts" / f"{time.time_ns()}.json",
            {
                "completed_utc": now(),
                "smoke_only": smoke_only,
                "execution_sha256": sha(PLAN),
                "jobs": [
                    {"mode": m, "job": identity(j), **done[m, identity(j)]}
                    for m, j in needed
                ],
            },
        )
        print(
            (
                "ALL SEVEN SMOKE CHECKS COMPLETE. No full training or DEV predictions."
                if smoke_only
                else "ALL 21 TRAINING JOBS AND 9 REUSED-HEAD CHECKS COMPLETE. Tell Astra the batch finished; outputs are ready for inspection."
            ),
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "mode", choices=["prepare", "validate", "status", "smoke", "run", "worker"]
    )
    ap.add_argument("--attempt", type=Path)
    a = ap.parse_args()
    if a.mode == "prepare":
        prepare()
    elif a.mode == "worker":
        require(a.attempt is not None, "Missing attempt")
        worker(a.attempt)
    elif a.mode in ("run", "smoke"):
        run(a.mode == "smoke")
    else:
        plan = validate()
        done = done_jobs()
        print(
            json.dumps(
                {
                    "inherited_smokes": plan["inherited_smokes"],
                    "verified_completions": [
                        {"mode": m, "job": j, **v} for (m, j), v in done.items()
                    ],
                    "full_jobs_total": 21,
                    "reuse_total": 9,
                    "smoke_total": 7,
                    "no_model_loaded": True,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

"Fit one model using the fixed training recipe."

from __future__ import annotations
from training.recovery.update_rules import (
    two_step_case as _two_step_case,
    verify as _verify_update,
)
import gc
import hashlib
import math
import time
from pathlib import Path
from training.full_data import common as c
from training.full_data import tokenizer as guard

b = None


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
        not info.get("mismatched_keys") and (not info.get("error_msgs")),
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
            not info.get("missing_keys") and (not info.get("unexpected_keys")),
            "Trained checkpoint does not reload cleanly",
        )
    else:
        c.require(
            set(info.get("missing_keys", [])) == expected,
            "Unexpected missing pretrained weights",
        )
        c.require(
            all(
                (
                    k.startswith("lm_head.")
                    or k.endswith(("position_ids", "token_type_ids"))
                    for k in info.get("unexpected_keys", [])
                )
            ),
            "Unexpected pretrained weights",
        )
    model.to(device)
    c.require(
        all((p.dtype == torch.float32 for p in model.parameters())),
        "Non-FP32 parameters",
    )
    return (tok, model, c.ready(info))


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

    length = max((len(r["input_ids"]) for r in rows))
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
    return b.split(data, job)


def parameter_fingerprint(model):
    import numpy as np

    digest = hashlib.sha256()
    canonical = hashlib.sha256()
    negative_zeros = 0
    for name, parameter in model.named_parameters():
        value = parameter.detach().cpu().contiguous().numpy().copy()
        header = name.encode("utf-8") + str(value.shape).encode("ascii")
        digest.update(header)
        canonical.update(header)
        digest.update(value.tobytes())
        negative_zeros += int(np.sum((value == 0) & np.signbit(value)))
        value[value == 0] = 0.0
        canonical.update(value.tobytes())
    return dict(
        raw_sha256=digest.hexdigest(),
        zero_canonical_sha256=canonical.hexdigest(),
        negative_zero_count=negative_zeros,
    )


def train_attempt(attempt, job, data, cache, *, pilot=False, save_model=False):
    import numpy as np

    torch = setup()
    start = time.perf_counter()
    cfg = c.config_map()[job["config"]]
    spec, fit, held = split(data, job)
    tok, model, loading = load(cache, spec, job["seed"])
    encoded = tokens(tok, [data["texts"][r["feature_row"]] for r in fit])
    held_tokens = (
        tokens(tok, [data["texts"][r["feature_row"]] for r in held]) if held else []
    )
    yy = np.asarray([r["target"] for r in fit], dtype=np.int64)
    weights = c.class_weights(
        job["head"], yy, len(spec["labels"]), cfg["loss"] == "balanced"
    )
    watch_names = (
        "roberta.encoder.layer.0.attention.self.query.weight",
        "roberta.encoder.layer.11.attention.self.query.weight",
        "classifier.out_proj.weight",
    )
    parameters = dict(model.named_parameters())
    before = {k: parameters[k].detach().cpu().clone() for k in watch_names}
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        (
            no_decay if name.endswith(".bias") or "LayerNorm.weight" in name else decay
        ).append(param)
    optimizer = torch.optim.AdamW(
        [
            dict(params=decay, weight_decay=0.01),
            dict(params=no_decay, weight_decay=0.0),
        ],
        lr=cfg["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-08,
        fused=True,
    )
    total = math.ceil(len(fit) / 16) * cfg["epochs"]
    warmup = math.ceil(0.1 * total)
    initialization_only = total == 1
    initial_fingerprint = parameter_fingerprint(model) if total in (1, 2) else None

    def multiplier(step):
        if step < warmup:
            return step / max(1, warmup)
        return max(0.0, (total - step) / max(1, total - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
    scaler = torch.amp.GradScaler("cuda")
    rng = np.random.default_rng(job["seed"])
    curve = []
    steps = []
    skipped = 0
    consecutive = 0
    max_steps = min(8, total) if pilot else total
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    train_start = time.perf_counter()
    try:
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            epoch_losses = []
            order = rng.permutation(len(fit))
            for offset in range(0, len(order), 16):
                b.deadline_guard()
                idx = order[offset : offset + 16]
                optimizer.zero_grad(set_to_none=True)
                labels = torch.as_tensor(yy[idx], device="cuda")
                scale_before = float(scaler.get_scale())
                lr = float(optimizer.param_groups[0]["lr"])
                with torch.autocast("cuda", dtype=torch.float16):
                    logits = model(**batch([encoded[i] for i in idx], "cuda")).logits
                    loss = loss_value(logits, labels, job["head"], weights)
                c.require(
                    bool(torch.isfinite(logits).all()) and bool(torch.isfinite(loss)),
                    "Nonfinite training forward",
                )
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                after = float(scaler.get_scale())
                skip = after < scale_before
                c.require(math.isfinite(after) and after > 0, "Invalid AMP scale")
                if not skip:
                    scheduler.step()
                skipped += int(skip)
                consecutive = consecutive + 1 if skip else 0
                c.require(
                    consecutive <= 20,
                    "More than20 consecutive skipped optimizer updates",
                )
                gn = float(grad_norm)
                c.require(
                    math.isfinite(gn) or skip,
                    "Nonfinite gradient without skipped update",
                )
                steps.append(
                    dict(
                        step=len(steps) + 1,
                        epoch=epoch,
                        lr=lr,
                        scale_before=scale_before,
                        scale_after=after,
                        skipped=skip,
                        grad_norm=gn if math.isfinite(gn) else str(gn),
                    )
                )
                epoch_losses.append((float(loss.detach()), len(idx)))
                if len(steps) >= max_steps:
                    break
            weighted_loss = sum((v * n for v, n in epoch_losses)) / sum(
                (n for _, n in epoch_losses)
            )
            item = dict(
                epoch=epoch,
                training_objective=weighted_loss,
                attempted_steps=len(steps),
                successful_steps=len(steps) - skipped,
            )
            if held:
                z = infer(model, held_tokens)
                yt = np.asarray([r["target"] for r in held], dtype=np.int64)
                c.array(attempt / "epoch_logits" / f"{epoch:03d}.npy", z)
                unit = c.class_weights(job["head"], yt, len(spec["labels"]), False)
                item["held_unweighted_loss"] = float(
                    loss_value(
                        torch.from_numpy(z), torch.from_numpy(yt), job["head"], unit
                    )
                )
            curve.append(item)
            if not pilot and (epoch == 1 or epoch % 10 == 0 or epoch == cfg["epochs"]):
                print(
                    f"  {job['id']} epoch {epoch}/{cfg['epochs']}; steps {len(steps)}/{total}; skipped {skipped}",
                    flush=True,
                )
            if len(steps) >= max_steps:
                break
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - train_start
        c.require(
            len(steps) == max_steps and (initialization_only or skipped < len(steps)),
            "Incomplete or ineffective training",
        )
        c.require(
            all((bool(torch.isfinite(p).all()) for p in model.parameters())),
            "Nonfinite trained weight",
        )
        changes = {
            k: float((parameters[k].detach().cpu() - before[k]).abs().max())
            for k in watch_names
        }
        amp_initialization_only = _two_step_case(steps, total)
        final_fingerprint = (
            parameter_fingerprint(model)
            if initialization_only or amp_initialization_only
            else None
        )
        if initialization_only or amp_initialization_only:
            c.save(
                attempt / "parameter_equivalence.json",
                dict(
                    before=initial_fingerprint, after=final_fingerprint, changes=changes
                ),
            )
            c.require(
                initial_fingerprint["zero_canonical_sha256"]
                == final_fingerprint["zero_canonical_sha256"]
                and all((v == 0 for v in changes.values())),
                "One-step initialization changed numerical weights",
            )
            c.require(
                all((step["lr"] == 0 or step["skipped"] for step in steps)),
                "Initialization-only outcome has an accepted positive-rate update",
            )
        else:
            c.require(
                all((v > 0 for v in changes.values()))
                and any((not step["skipped"] and step["lr"] > 0 for step in steps)),
                "Unexpected no-update training; preserve attempt and inspect",
            )
        final_held = infer(model, held_tokens) if held else None
        if held:
            c.array(attempt / "held_logits.npy", final_held)
        probe_encoded = (held_tokens or encoded)[:3]
        probe = infer(model, probe_encoded)
        c.array(attempt / "probe_logits.npy", probe)
        export_check = None
        if save_model:
            destination = b.model_path(job, attempt.name)
            c.require(not destination.exists(), "Preserve existing export")
            destination.mkdir(parents=True)
            model.save_pretrained(str(destination), safe_serialization=True)
            tok.save_pretrained(str(destination))
            with (destination / "study_meta.json").open("xb") as stream:
                stream.write(
                    c.encoded(
                        dict(
                            job=job,
                            labels=spec["labels"],
                            kind=spec["kind"],
                            config=cfg,
                            pilot_only=pilot,
                        )
                    )
                )
            c.require(
                sum((p.stat().st_size for p in destination.rglob("*") if p.is_file()))
                <= b.EXPORT_BYTES_MAX,
                "Export exceeds predeclared storage bound",
            )
            tok2, reloaded, info2 = load(
                destination, spec, job["seed"], device="cpu", trained=True
            )
            cpu_logits = infer(reloaded, probe_encoded, device="cpu")
            difference = float(np.max(np.abs(cpu_logits - probe)))
            c.require(difference <= 0.0001, "Reloaded CPU/GPU probe mismatch")
            export_check = dict(
                max_abs_logit_difference=difference, tolerance=0.0001, clean_reload=True
            )
            del reloaded, tok2
        result = dict(
            job=job,
            configuration=cfg,
            pilot_only=pilot,
            loading=loading,
            initialization_only=initialization_only,
            initial_parameter_fingerprint=initial_fingerprint,
            final_parameter_fingerprint=final_fingerprint,
            model_directory=str(destination) if save_model else None,
            fit_ids=[r["id"] for r in fit],
            held_ids=[r["id"] for r in held],
            class_weights=weights,
            planned_steps=total,
            attempted_steps=len(steps),
            successful_steps=len(steps) - skipped,
            skipped_steps=skipped,
            steps=steps,
            epochs=curve,
            changes=changes,
            train_seconds=train_seconds,
            total_seconds=time.perf_counter() - start,
            peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            gpu=torch.cuda.get_device_name(0),
            export_check=export_check,
            token_lengths=dict(
                fit_max=max((len(v["input_ids"]) for v in encoded)),
                fit_at128=sum((len(v["input_ids"]) == 128 for v in encoded)),
            ),
            optimizer="AdamW; correct bias correction; no bias/LayerNorm weight decay; linear warmup/decay",
            inference_precision="FP32; no autocast",
            training_precision="FP16 autocast, FP32 parameters",
        )
        result["update_state"] = (
            "initialization_only"
            if initialization_only
            else "amp_initialization_only" if amp_initialization_only else "trained"
        )
        result["accepted_positive_updates"] = sum(
            not s["skipped"] and s["lr"] > 0 for s in steps
        )
        _verify_update(
            result,
            (
                c.read(attempt / "parameter_equivalence.json")
                if initialization_only or amp_initialization_only
                else None
            ),
        )
        c.save(attempt / "training.json", result)
        return result
    finally:
        del model, tok, optimizer, scheduler, scaler, parameters, before
        gc.collect()
        torch.cuda.empty_cache()

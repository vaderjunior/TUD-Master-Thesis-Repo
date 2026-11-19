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
from training.roberta.kernels import (
    setup,
    load,
    loss_value,
    tokens,
    batch,
    infer,
    split,
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
        eps=1e-8,
        fused=True,
    )
    total = math.ceil(len(fit) / 16) * cfg["epochs"]
    warmup = math.ceil(0.1 * total)

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
    max_steps = 8 if pilot else total
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    train_start = time.perf_counter()
    try:
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            epoch_losses = []
            order = rng.permutation(len(fit))
            for offset in range(0, len(order), 16):
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
            weighted_loss = sum(v * n for v, n in epoch_losses) / sum(
                n for _, n in epoch_losses
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
            len(steps) == max_steps and skipped < len(steps),
            "Incomplete or ineffective training",
        )
        c.require(
            all(bool(torch.isfinite(p).all()) for p in model.parameters()),
            "Nonfinite trained weight",
        )
        changes = {
            k: float((parameters[k].detach().cpu() - before[k]).abs().max())
            for k in watch_names
        }
        c.require(
            all(v > 0 for v in changes.values()),
            "Classifier or encoder layers did not update",
        )
        final_held = infer(model, held_tokens) if held else None
        if held:
            c.array(attempt / "held_logits.npy", final_held)
        probe_encoded = (held_tokens or encoded)[:3]
        probe = infer(model, probe_encoded)
        c.array(attempt / "probe_logits.npy", probe)
        export_check = None
        if save_model:
            destination = c.own(attempt / "model")
            model.save_pretrained(str(destination), safe_serialization=True)
            tok.save_pretrained(str(destination))
            c.save(
                destination / "study_meta.json",
                dict(
                    job=job,
                    labels=spec["labels"],
                    kind=spec["kind"],
                    config=cfg,
                    pilot_only=pilot,
                ),
            )

            tok2, reloaded, info2 = load(
                destination, spec, job["seed"], device="cpu", trained=True
            )
            cpu_logits = infer(reloaded, probe_encoded, device="cpu")
            difference = float(np.max(np.abs(cpu_logits - probe)))
            c.require(difference <= 1e-4, "Reloaded CPU/GPU probe mismatch")
            export_check = dict(
                max_abs_logit_difference=difference, tolerance=1e-4, clean_reload=True
            )
            del reloaded, tok2
        result = dict(
            job=job,
            configuration=cfg,
            pilot_only=pilot,
            loading=loading,
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
                fit_max=max(len(v["input_ids"]) for v in encoded),
                fit_at128=sum(len(v["input_ids"]) == 128 for v in encoded),
            ),
            optimizer="AdamW; correct bias correction; no bias/LayerNorm weight decay; linear warmup/decay",
            inference_precision="FP32; no autocast",
            training_precision="FP16 autocast, FP32 parameters",
        )
        c.save(attempt / "training.json", result)
        return result
    finally:
        del model, tok, optimizer, scheduler, scaler, parameters, before
        gc.collect()
        torch.cuda.empty_cache()


def execute(job, data, plan, plan_sha):
    directory = c.job_dir(job)
    receipt = directory / "complete.json"
    if receipt.exists():
        c.verify_receipt(receipt, job, plan_sha)
        return "verified"
    c.require(
        shutil.disk_usage(c.ROOT).free >= 8 * 2**30,
        "Less than8GiB free; preserve outputs and free space before resuming",
    )
    attempt = c.own(directory / "attempts" / uuid.uuid4().hex[:16])
    attempt.mkdir(parents=True)
    c.save(
        attempt / "started.json",
        dict(job=job, plan_sha256=plan_sha, started_utc=c.utc()),
    )
    try:
        train_attempt(
            attempt,
            job,
            data,
            plan["base_cache"]["snapshot"],
            save_model=job["mode"] == "final",
        )
        products = {
            p.relative_to(attempt).as_posix(): c.sha(p)
            for p in attempt.rglob("*")
            if p.is_file()
        }
        c.save(
            receipt,
            dict(
                status="complete",
                job=job,
                plan_sha256=plan_sha,
                attempt=attempt.relative_to(c.BASE).as_posix(),
                products=products,
            ),
        )
        c.verify_receipt(receipt, job, plan_sha)
        return "new"
    except BaseException as exc:
        c.save(
            attempt / "stopped.json",
            dict(
                type=type(exc).__name__,
                message=str(exc),
                traceback=traceback.format_exc(),
                utc=c.utc(),
            ),
        )
        raise


def inference_job(job, data, plan_sha, inputs):
    import numpy as np

    torch = setup()
    source, attempt = c.verify_receipt(c.job_dir(job) / "complete.json", job, plan_sha)
    source_sha = c.sha(c.job_dir(job) / "complete.json")
    directory = c.BASE / "inference" / job["id"]
    done = directory / "complete.json"
    if done.exists():
        value = c.read(done)
        c.require(
            value["job"] == job
            and value["plan_sha256"] == plan_sha
            and value["training_receipt_sha256"] == source_sha,
            "Inference receipt changed",
        )
        c.verify(value["artifacts"])
        return "verified"
    spec = data["heads"][job["head"]]
    output = np.empty((len(inputs), len(spec["labels"])), dtype=np.float32)
    records = []
    pending = []
    for start in range(0, len(inputs), 512):
        stop = min(start + 512, len(inputs))
        chunk = directory / "chunks" / f"{start:06d}_{stop:06d}"
        identity = dict(
            job=job,
            plan_sha256=plan_sha,
            training_receipt_sha256=source_sha,
            start=start,
            stop=stop,
            input_sha256=c.signature(inputs[start:stop]),
        )
        receipt = chunk.with_suffix(".json")
        array = chunk.with_suffix(".npy")
        if receipt.exists():
            r = c.read(receipt)
            c.require(
                r["identity"] == identity and c.sha(array) == r["sha256"],
                "Inference chunk changed",
            )
            values = np.load(array, allow_pickle=False)
            c.require(
                values.shape == (stop - start, len(spec["labels"]))
                and np.isfinite(values).all(),
                "Bad saved inference chunk",
            )
            output[start:stop] = values
            records.append(receipt)
        else:

            pending.append((start, stop, chunk, identity))
    if pending:
        tok, model, _ = load(attempt / "model", spec, job["seed"], trained=True)
        try:
            for start, stop, chunk, identity in pending:
                values = infer(
                    model, tokens(tok, [r["text"] for r in inputs[start:stop]])
                )
                path = chunk.with_suffix(".npy")
                if path.exists():

                    old = np.load(path, allow_pickle=False)
                    c.require(
                        old.shape == values.shape
                        and np.allclose(old, values, rtol=0, atol=1e-4),
                        "Interrupted chunk differs; inspect before continuing",
                    )
                    values = old
                else:
                    c.array(path, values)
                receipt = chunk.with_suffix(".json")
                c.save(receipt, dict(identity=identity, sha256=c.sha(path)))
                output[start:stop] = values
                records.append(receipt)
                if stop % 4096 == 0 or stop == len(inputs):
                    print(f"  inference {job['id']}: {stop}/{len(inputs)}", flush=True)
        finally:
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()
    c.array(directory / "logits.npy", output)
    artifacts = {c.rel(p): c.sha(p) for p in records}
    artifacts.update(
        {c.rel(p.with_suffix(".npy")): c.sha(p.with_suffix(".npy")) for p in records}
    )
    artifacts[c.rel(directory / "logits.npy")] = c.sha(directory / "logits.npy")
    c.save(
        done,
        dict(
            job=job,
            plan_sha256=plan_sha,
            training_receipt_sha256=source_sha,
            rows=len(inputs),
            artifacts=artifacts,
        ),
    )
    return "new"

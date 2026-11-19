import gc
from pathlib import Path
import time
import numpy as np
from training.text_budgets.extended import common as b
from training.frozen_features import roberta as encoder

c = b.c


def validate_values(value, n):
    c.require(
        value.dtype == np.float32
        and value.shape == (n, 768)
        and np.isfinite(value).all()
        and np.allclose(np.linalg.norm(value, axis=1), 1, rtol=0, atol=2e-4),
        "Invalid frozen feature matrix",
    )


def commit_chunk(stem, identity, values=None):
    path = stem.with_suffix(".npy")
    receipt = stem.with_suffix(".json")
    n = len(identity["keys"])
    if receipt.exists():
        r = c.read(receipt)
        c.require(
            r["identity"] == identity and c.sha(path) == r["sha256"],
            "Changed feature chunk identity/content",
        )
        saved = np.load(path, allow_pickle=False)
        validate_values(saved, n)
        return saved
    c.require(
        values is not None,
        "Incomplete feature chunk must be recomputed before commitment",
    )
    validate_values(values, n)
    if path.exists():
        saved = np.load(path, allow_pickle=False)
        validate_values(saved, n)
        c.require(
            np.allclose(saved, values, rtol=0, atol=1e-4),
            "Interrupted feature chunk differs; preserve for inspection",
        )
        values = saved
    else:
        b.array(path, values)
    b.save(receipt, dict(identity=identity, sha256=c.sha(path)))
    return values


def run(plan, plan_sha):
    root = c.BASE / "features"
    done = root / "complete.json"
    rows = c.read(c.BASE / "data/TRAIN_features.json")
    inputs_sha = c.sha(c.BASE / "data/TRAIN_features.json")
    identity = dict(
        plan_sha256=plan_sha,
        input_sha256=inputs_sha,
        rows=len(rows),
        pooling="mean_all_nonpadding_L2_max128_FP32",
    )
    if done.exists():
        r = c.read(done)
        c.require(r["identity"] == identity, "Frozen features belong to different plan")
        c.verify(r["outputs"])
        value = np.load(root / "TRAIN_mean.npy", mmap_mode="r", allow_pickle=False)
        validate_values(value, len(rows))
        return r
    b.resource.resource_guard(3.5, 12)
    matrix = np.empty((len(rows), 768), dtype=np.float32)
    pending = []
    outputs = {}
    sources = c.read(c.BASE / "data/cache_sources.json")
    caches = {
        name: np.load(b.ROOT / entry["matrix"], mmap_mode="r", allow_pickle=False)
        for name, entry in sources.items()
    }
    for i, row in enumerate(rows):
        old = row["cached_feature"]
        if old is None:
            pending.append(i)
        else:
            matrix[i] = caches[old["source"]][old["row"]]
    missing = []
    for start in range(0, len(pending), 64):
        indices = pending[start : start + 64]
        rr = [rows[i] for i in indices]
        stamp = dict(
            plan_sha256=plan_sha,
            keys=[r["key"] for r in rr],
            input_sha256=c.signature(rr),
            indices=indices,
        )
        stem = root / "chunks" / f"{start:06d}"
        if stem.with_suffix(".json").exists():
            matrix[indices] = commit_chunk(stem, stamp)
        else:
            missing.append((stem, stamp, indices, rr))
        outputs.update(
            {c.rel(stem.with_suffix(ext)): None for ext in (".json", ".npy")}
        )
    if missing:
        from training.text_budgets.extended import train as train

        torch = train.setup()
        tok, model, info = encoder.load("cuda", plan)
        try:
            reference = c.read(b.ROOT / plan["pilot_receipt"])["feature_reference"]
            cpu = np.load(b.ROOT / reference["cpu_path"], allow_pickle=False)
            gpu = encoder.infer(tok, model, encoder.PROBES, "cuda", True)["mean"]
            c.require(
                np.allclose(cpu, gpu, rtol=0, atol=1e-4),
                "Frozen feature CPU/GPU reference changed",
            )
            for stem, stamp, indices, rr in missing:
                b.deadline_guard()
                batches = [
                    encoder.infer(
                        tok, model, [r["text"] for r in rr[i : i + 16]], "cuda"
                    )["mean"]
                    for i in range(0, len(rr), 16)
                ]
                matrix[indices] = commit_chunk(stem, stamp, np.concatenate(batches))
                print(
                    f"Frozen TRAIN features: chunk {stem.name}; {len(indices)} rows committed",
                    flush=True,
                )
        finally:
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()
    validate_values(matrix, len(rows))
    for i, row in enumerate(rows):
        old = row["cached_feature"]
        if old is not None:
            c.require(
                np.array_equal(matrix[i], caches[old["source"]][old["row"]]),
                "Cached feature row changed",
            )
    b.array(root / "TRAIN_mean.npy", matrix)
    outputs[c.rel(root / "TRAIN_mean.npy")] = None
    outputs = {path: c.sha(b.ROOT / path) for path in outputs}
    result = dict(
        identity=identity,
        outputs=outputs,
        reused_rows=len(rows) - len(pending),
        new_rows=len(pending),
        Qwen_calls=0,
        neural_training_steps=0,
    )
    b.save(done, result)
    return result

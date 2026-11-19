from training.matched_annotations import validate as r
import gc
from pathlib import Path
import socket
import time
from unittest.mock import patch
import numpy as np


def main():
    c = r.c
    m = r.m
    c.require(
        c.read(r.OUT / "integrity.json")["status"] == "pass",
        "Integrity review required",
    )
    policy = c.read(r.OUT / "analysis_plan.json")
    plan = c.read(r.BASE / "plan.json")
    inputs = c.read(r.BASE / "data/evaluation_inputs.json")
    pos = policy["cpu_probe"]["positions"]
    metadata = c.read(r.OUT / "training_runs.json")["runs"]
    truncated = 0
    for run in metadata:
        meta = c.read(c.ROOT / run["attempt"] / "training.json")
        info = meta["loading"]
        c.require(
            set(info["missing_keys"])
            == {
                "classifier.dense.bias",
                "classifier.dense.weight",
                "classifier.out_proj.bias",
                "classifier.out_proj.weight",
            }
            and not info["error_msgs"]
            and not info["mismatched_keys"]
            and all(k.startswith("lm_head.") for k in info["unexpected_keys"]),
            "Pretrained encoder was not loaded cleanly",
        )
        c.require(
            meta["optimizer"]
            == "AdamW; correct bias correction; no bias/LayerNorm weight decay; linear warmup/decay"
            and meta["training_precision"] == "FP16 autocast, FP32 parameters"
            and meta["inference_precision"] == "FP32; no autocast",
            "Training precision/optimizer differs",
        )
        c.require(
            0 < meta["train_seconds"] <= meta["total_seconds"]
            and 0 < meta["token_lengths"]["fit_max"] <= 128,
            "Invalid timing/token lengths",
        )
        truncated += meta["token_lengths"]["fit_at128"]
    import pyarrow.dataset, pyarrow.parquet, pandas
    import torch
    from transformers import RobertaForSequenceClassification
    from transformers.utils import logging

    logging.set_verbosity_error()
    logging.disable_progress_bar()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = m.feature.tokenizer.load(plan["base_cache"]["snapshot"])
    guard = m.feature.tokenizer.check(tok)
    rows = []
    begin = time.perf_counter()
    for job in policy["cpu_probe"]["jobs"]:
        rec = c.read(r.BASE / "refits" / job["id"] / "complete.json")
        modelpath = r.BASE / rec["attempt"] / "model"
        model, info = RobertaForSequenceClassification.from_pretrained(
            str(modelpath),
            local_files_only=True,
            use_safetensors=True,
            dtype=torch.float32,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        c.require(
            not any(
                info.get(k)
                for k in (
                    "missing_keys",
                    "unexpected_keys",
                    "mismatched_keys",
                    "error_msgs",
                )
            ),
            "CPU reload failed",
        )
        labels = m.data(job["pool"])["heads"]["hate_type"]["labels"]
        c.require(
            [model.config.id2label[i] for i in range(7)] == labels,
            "CPU label order changed",
        )
        c.require(
            all(
                p.dtype == torch.float32 and p.device.type == "cpu"
                for p in model.parameters()
            ),
            "CPU/FP32 required",
        )
        model.eval()
        out = []
        with torch.inference_mode():
            for start in range(0, len(pos), 4):
                batch = m.feature.tokenizer.encode(
                    tok,
                    [inputs[i]["text"] for i in pos[start : start + 4]],
                    padding=True,
                    return_tensors="pt",
                )
                out.append(
                    model(
                        **{
                            k: v
                            for k, v in batch.items()
                            if k in ("input_ids", "attention_mask")
                        }
                    ).logits.numpy()
                )
        fresh = np.concatenate(out)
        old = np.load(
            r.BASE / "inference" / job["id"] / "logits.npy", allow_pickle=False
        )[pos]
        drift = float(np.max(np.abs(old - fresh)))
        c.require(
            drift <= policy["cpu_probe"]["tolerance"],
            "CPU inference does not reproduce",
        )

        cuts = {0.5}
        for fam in c.read(r.BASE / f"selections/{job['pool']}.json")[
            "families"
        ].values():
            if fam["selected"]["config"] == job["config"]:
                cuts.add(fam["selected"]["threshold"])
        bits = {
            str(t): int(
                np.sum(
                    (c.probabilities("hate_type", old) >= t)
                    != (c.probabilities("hate_type", fresh) >= t)
                )
            )
            for t in sorted(cuts)
        }
        c.require(not any(bits.values()), "CPU changes a probe decision")
        name = f"cpu_reloads/{job['pool']}_{job['config']}_s{job['seed']}.npy"
        r.array(name, fresh)
        rows.append(
            dict(
                job=job,
                max_abs_logit_difference=drift,
                decision_differences=bits,
                array=name,
            )
        )
        del model
        gc.collect()
        print(
            "CPU model reload passed:", job["id"], f"max drift {drift:.3g}", flush=True
        )
    fi = c.read(r.BASE / "data/feature_inputs.json")
    saved = np.load(r.BASE / "features/mean.npy", allow_pickle=False)
    tok, model, info = m.feature.load("cpu", plan)
    parts = []
    for start in range(0, len(fi), 16):
        parts.append(
            m.feature.infer(
                tok,
                model,
                [v["text"] for v in fi[start : start + 16]],
                "cpu",
                check_numpy=True,
            )["mean"]
        )
        if (start // 16) % 8 == 0:
            print(
                "CPU independent feature pooling:",
                min(start + 16, len(fi)),
                "/",
                len(fi),
                flush=True,
            )
    fresh = np.concatenate(parts)
    drift = float(np.max(np.abs(saved - fresh)))
    c.require(drift <= 1e-4, "Frozen TRAIN features do not reproduce on CPU")
    r.array("cpu_training_features.npy", fresh)
    del model
    gc.collect()
    r.save(
        "cpu_reload_audit.json",
        dict(
            status="pass",
            models=rows,
            model_count=len(rows),
            probe_rows=len(pos),
            positions=pos,
            tokenizer_guard=guard,
            feature_rows=len(fi),
            feature_max_abs_difference=drift,
            independent_numpy_pooling=True,
            max_model_logit_difference=max(v["max_abs_logit_difference"] for v in rows),
            seconds=time.perf_counter() - begin,
            clean_pretrained_initialization_records=len(metadata),
            summed_fit_at128_counts=truncated,
            source_sha256=c.sha(Path(__file__)),
            Qwen_calls=0,
            CUDA_used=False,
            new_neural_training=0,
        ),
    )
    print("All CPU reloads and all631 frozen TRAIN features reproduced.", flush=True)


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", r.forbid), patch.object(
        socket, "create_connection", r.forbid
    ):
        main()

import os

os.environ.update(
    CUDA_VISIBLE_DEVICES="",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    TOKENIZERS_PARALLELISM="false",
)
import gc
import hashlib
from pathlib import Path
import socket
from unittest.mock import patch
import numpy as np
from training.roberta import common as c
from training.roberta import tokenizer as tokenizer
from training.roberta import results as review


def main():
    import pyarrow.dataset, pyarrow.parquet, pandas
    import torch
    from transformers import RobertaForSequenceClassification
    from transformers.utils import logging

    logging.set_verbosity_error()
    logging.disable_progress_bar()
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    torch.use_deterministic_algorithms(True)
    inputs = c.read(c.BASE / "data/inference.json")
    data = c.read(c.BASE / "data/train.json")
    selection = c.read(c.BASE / "selection.json")["families"]
    positions = sorted(
        sorted(
            range(len(inputs)),
            key=lambda i: hashlib.sha256(
                ("roberta-review-cpu|" + inputs[i]["key"]).encode()
            ).hexdigest(),
        )[:16]
    )
    jobs = [j for j in c.final_jobs(selection) if j["seed"] == 10]
    c.require(len(jobs) == 12, "Expected twelve fixed-seed/config/head checks")
    plan = {
        "selection": "Sixteen smallest identity hashes under roberta-review-cpu, sorted by original position; all twelve unique final head/config jobs at seed10.",
        "positions": positions,
        "keys": [inputs[i]["key"] for i in positions],
        "jobs": jobs,
        "tolerance": 1e-4,
        "new_training": False,
        "CUDA_used": False,
    }
    review.save("cpu_reload_plan.json", plan)
    results = []
    for job in jobs:
        receipt, attempt = c.verify_receipt(
            c.job_dir(job) / "complete.json", job, c.sha(c.BASE / "plan.json")
        )
        tok = tokenizer.load(c.read(c.BASE / "plan.json")["base_cache"]["snapshot"])
        model, info = RobertaForSequenceClassification.from_pretrained(
            str(attempt / "model"),
            local_files_only=True,
            use_safetensors=True,
            dtype=torch.float32,
            attn_implementation="sdpa",
            output_loading_info=True,
        )
        c.require(
            all(
                not info.get(k)
                for k in [
                    "missing_keys",
                    "unexpected_keys",
                    "mismatched_keys",
                    "error_msgs",
                ]
            ),
            "Independent checkpoint reload failed",
        )
        c.require(
            [model.config.id2label[i] for i in range(model.config.num_labels)]
            == data["heads"][job["head"]]["labels"],
            "Reloaded label order differs",
        )
        c.require(
            all(
                p.device.type == "cpu" and p.dtype == torch.float32
                for p in model.parameters()
            ),
            "CPU/precision contract",
        )
        model.eval()
        values = []
        with torch.inference_mode():
            for start in range(0, len(positions), 4):
                texts = [inputs[i]["text"] for i in positions[start : start + 4]]
                encoded = tokenizer.encode(
                    tok, texts, padding=True, return_tensors="pt"
                )

                values.append(
                    model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                    ).logits.numpy()
                )
        actual = np.concatenate(values)
        saved = np.load(
            c.BASE / "inference" / job["id"] / "logits.npy", allow_pickle=False
        )[positions]
        difference = float(np.max(np.abs(actual - saved)))
        c.require(
            np.isfinite(actual).all() and difference <= 1e-4,
            "Independent inference mismatch",
        )
        path = review.OUT / "cpu_reloads" / f"{job['head']}_{job['config']}_s10.npy"
        path.parent.mkdir(exist_ok=True)
        if path.exists():
            c.require(
                np.array_equal(np.load(path, allow_pickle=False), actual),
                "Preserve previous CPU probe",
            )
        else:
            np.save(path, actual, allow_pickle=False)
        result = {
            "job": job,
            "rows": len(positions),
            "max_abs_logit_difference": difference,
            "array_sha256": c.sha(path),
            "model_sha256": receipt["products"]["model/model.safetensors"],
        }
        results.append(result)
        print("Independent CPU reload", job["id"], difference, flush=True)
        del model, tok, actual, saved
        gc.collect()
    review.save(
        "cpu_reload.json",
        {
            "status": "pass",
            "plan_sha256": c.sha(review.OUT / "cpu_reload_plan.json"),
            "checks": results,
            "models": len(results),
            "head_rows": sum(r["rows"] for r in results),
            "maximum_difference": max(r["max_abs_logit_difference"] for r in results),
            "source_sha256": c.sha(Path(__file__)),
            "training_or_cuda_used": False,
            "Qwen_calls": 0,
        },
    )


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", review.forbid), patch.object(
        socket, "create_connection", review.forbid
    ):
        main()

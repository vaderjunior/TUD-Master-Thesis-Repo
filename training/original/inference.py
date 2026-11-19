import argparse
import gc
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from analysis.common import study as c
from analysis.common import checkpoints as chunks
from training.original import run as en
from training.original import legacy as de

BASE = c.BASE / "encoders"
PLAN = BASE / "plan.json"
AUDIT = (
    c.ROOT / "Thesis_Log/astra_result/evidence/core_encoder_results_20260915_v5.json"
)


def prepare():
    if PLAN.exists():
        validate()
        print("Fixed-checkpoint inference plan verified.")
        return
    audit = c.read(AUDIT)
    c.require(
        audit["status"] == "complete_english_v5_german_v4_retained",
        "Wrong retained encoder generation",
    )
    bindings = c.scientific_inputs()
    bindings[c.rel(AUDIT)] = c.sha(AUDIT)
    jobs = []
    entries = [
        {"job": j["job"], "attempt": j["attempt"]} for j in audit["jobs"]
    ] + c.read(en.PLAN)["inherited_full"]
    for entry in entries:
        attempt = c.ROOT / entry["attempt"]
        receipt = c.read(attempt / "complete.json")
        v = c.read(attempt / "verification.json")
        c.require(
            v["job"] == entry["job"] and receipt["status"] == "complete",
            "Incomplete/wrong saved checkpoint",
        )
        arm, head, seed = entry["job"].split("/")
        path = c.ROOT / v["checkpoint"]
        meta_path = c.CORE / f"encoder_data/{arm}/{head}_meta.json"
        meta = c.read(meta_path)
        c.require(
            c.sha(path / "model.safetensors") == v["checkpoint_sha256"],
            "Saved weights changed",
        )
        for name, h in receipt["files"].items():
            if (c.ROOT / name).parent == path or Path(name).name in (
                "verification.json",
                "internal_val_logits.jsonl",
            ):
                c.require(c.sha(c.ROOT / name) == h, "Checkpoint/probe source differs")
                bindings[name] = h
        for file in path.iterdir():
            if file.is_file():
                name = c.rel(file)
                expected = receipt["files"].get(
                    name, audit["verified_hashes"].get(name)
                )
                c.require(expected == c.sha(file), "Unbound checkpoint file: " + name)
                bindings[name] = expected
        for file in [
            attempt / "complete.json",
            attempt / "verification.json",
            attempt / "internal_val_logits.jsonl",
            meta_path,
            c.CORE / f"encoder_data/{arm}/{head}_val.parquet",
        ]:
            bindings[c.rel(file)] = c.sha(file)
        jobs.append(
            {
                "job": entry["job"],
                "arm": arm,
                "head": head,
                "seed": int(seed.removeprefix("seed")),
                "checkpoint": c.rel(path),
                "checkpoint_sha256": v["checkpoint_sha256"],
                "meta": meta,
                "reference_attempt": entry["attempt"],
                "rows": 18275 if meta["lang"] == "en" else 3352,
            }
        )
        print("Bound retained checkpoint:", entry["job"], flush=True)
    c.require(
        len(jobs) == 30 and len({j["job"] for j in jobs}) == 30,
        "Wrong retained checkpoint roster",
    )
    for file in [
        Path(__file__),
        Path(c.__file__),
        Path(chunks.__file__),
        Path(en.__file__),
        Path(de.__file__),
        Path(en.token_guard.__file__),
        en.token_guard.REFERENCE,
        c.CORE / "encoder_plan.json",
        en.PLAN,
    ]:
        bindings[c.rel(file)] = c.sha(file)
    from analysis.knowledge.banks.build import environment

    c.save(
        PLAN,
        {
            "created_utc": c.utc(),
            "status": "inference_only_preflight_required",
            "inputs": bindings,
            "jobs": jobs,
            "environment": environment(),
            "policy": "English corrected v5/full and compact heads, German retained v4 GBERT. All three fixed fitted seeds; original 128-token input semantics, FP32 inference, batch 32, default argmax or sigmoid>=0.5. No refitting, threshold sweep or new checkpoint selection.",
            "budget": {
                "head_jobs": 30,
                "raw_head_predictions": sum((j["rows"] for j in jobs)),
            },
            "preflight": "Every tokenizer checked against its serialized reference on every broad input; four existing internal-validation logit probes on CPU. All 30 checkpoints hash checked. Full GPU inference also performs a saved-reference probe for each head before predicting.",
            "device": "CUDA required only by explicit run mode. CPU preflight does not enable CUDA. No Qwen.",
        },
    )


def validate():
    p = c.read(PLAN)
    c.verify(p["inputs"])
    from analysis.knowledge.banks.build import environment

    c.require(environment() == p["environment"], "Encoder environment changed")
    return p


def tokenizer_check(job, roster):
    from transformers import BertTokenizerFast
    from tokenizers import Tokenizer

    path = c.ROOT / job["checkpoint"]
    texts = [r["text"] for r in roster if r["lang"] == job["meta"]["lang"]]
    if job["meta"]["lang"] == "en":
        tok = en.token_guard.load(path)
        enc = en.token_guard.encode(tok, texts)
        return {
            "rows": len(texts),
            "input_ids_sha256": c.signature(enc["input_ids"]),
            "reference": "historical pinned RoBERTa serialization",
            "semantic_sha256": en.token_guard.SEMANTIC_SHA,
        }
    tok = BertTokenizerFast.from_pretrained(str(path), local_files_only=True)
    ref = Tokenizer.from_file(str(path / "tokenizer.json"))
    ref.no_padding()
    ref.enable_truncation(max_length=128, direction="right")
    enc = tok(texts, truncation=True, max_length=128, padding=False)
    expected = ref.encode_batch(texts)
    c.require(
        enc["input_ids"] == [x.ids for x in expected]
        and enc["attention_mask"] == [x.attention_mask for x in expected],
        "German tokenizer differs from saved serialization",
    )
    c.require(
        tok.padding_side == tok.truncation_side == "right" and tok.vocab_size == 31102,
        "Unexpected GBERT tokenizer",
    )
    return {
        "rows": len(texts),
        "input_ids_sha256": c.signature(enc["input_ids"]),
        "reference_sha256": c.sha(path / "tokenizer.json"),
    }


def load(job, device):
    import torch

    tok, model, info = (en if job["meta"]["lang"] == "en" else de).load_model(
        c.ROOT / job["checkpoint"], job["meta"]
    )
    model.to(device)
    model.eval()
    c.require(
        all((p.dtype == torch.float32 for p in model.parameters())),
        "Wrong inference dtype",
    )
    return (tok, model)


def infer(tok, model, texts, lang, device):
    import torch

    with torch.inference_mode():
        batch = (
            en.token_guard.encode(tok, texts, padding=True, return_tensors="pt")
            if lang == "en"
            else tok(
                texts,
                truncation=True,
                max_length=128,
                padding=True,
                return_tensors="pt",
            )
        )
        batch = {
            k: v.to(device)
            for k, v in batch.items()
            if k in ("input_ids", "attention_mask")
        }
        out = model(**batch).logits.float().cpu().numpy()
    c.require(np.isfinite(out).all(), "Nonfinite inference")
    return out


def reference_probe(job, tok, model, device):
    import pandas as pd

    rr = c.rows(c.ROOT / job["reference_attempt"] / "internal_val_logits.jsonl")[:8]
    frame = pd.read_parquet(
        c.CORE / f"encoder_data/{job['arm']}/{job['head']}_val.parquet"
    ).iloc[: len(rr)]
    c.require(
        frame.id.astype(str).tolist() == [r["id"] for r in rr],
        "Internal-validation probe identity changed",
    )
    logits = infer(
        tok, model, frame.text.astype(str).tolist(), job["meta"]["lang"], device
    )
    expected = np.asarray([r["logits"] for r in rr])
    error = float(np.abs(logits - expected).max())
    c.require(
        np.allclose(logits, expected, atol=0.0003, rtol=0.0003),
        "Saved-reference FP32 logits differ",
    )
    c.require(
        np.array_equal(
            en.decisions(logits, job["meta"]["kind"]),
            np.asarray([r["decision"] for r in rr]),
        ),
        "Probe decision changed",
    )
    return {
        "job": job["job"],
        "rows": len(rr),
        "device": device,
        "max_abs_logit_difference": error,
        "decisions_exact": True,
        "ids": [r["id"] for r in rr],
    }


def preflight():
    p = validate()
    path = BASE / "preflight.json"
    if path.exists():
        c.require(c.read(path)["plan_sha256"] == c.sha(PLAN), "Wrong preflight")
        print("Encoder preflight already complete.")
        return
    roster = c.rows(c.PREP / "broad_roster.jsonl")
    token_checks = {}
    probes = []
    for job in p["jobs"]:
        token_checks[job["job"]] = tokenizer_check(job, roster)
        c.require(
            token_checks[job["job"]]["rows"] == job["rows"],
            "Expanded tokenizer roster differs",
        )
        print("Tokenizer reference matches:", job["job"], flush=True)
    import torch

    torch.set_num_threads(4)
    for name in [
        "full/hate/seed0",
        "kb/hate_type/seed0",
        "de/hate/seed0",
        "de/legal/seed0",
    ]:
        job = next((j for j in p["jobs"] if j["job"] == name))
        tok, model = load(job, "cpu")
        probes.append(reference_probe(job, tok, model, "cpu"))
        print("CPU saved-logit probe matches:", name, flush=True)
        del model, tok
        gc.collect()
    c.save(
        path,
        {
            "status": "preflight_pass",
            "plan_sha256": c.sha(PLAN),
            "tokenizer_checks": token_checks,
            "reference_probes": probes,
            "actual_Qwen_calls": 0,
            "training_steps": 0,
            "broad_inference_performed": False,
        },
    )


def run():
    p = validate()
    ready = c.read(BASE / "preflight.json")
    c.require(
        ready["status"] == "preflight_pass" and ready["plan_sha256"] == c.sha(PLAN),
        "Verified preflight required",
    )
    from analysis.common.api import item_lock

    with item_lock(BASE / "inference.lock"):
        torch = en.setup_cuda()
        roster = c.rows(c.PREP / "broad_roster.jsonl")
        outputs = {}
        for job in p["jobs"]:
            root = BASE / "predictions" / job["job"]
            done = root / "complete.json"
            if done.exists():
                saved = c.read(done)
                c.require(
                    saved["plan_sha256"] == c.sha(PLAN) and saved["job"] == job["job"],
                    "Wrong completed encoder job",
                )
                c.verify(saved["outputs"])
                outputs[c.rel(done)] = c.sha(done)
                continue
            rr = [r for r in roster if r["lang"] == job["meta"]["lang"]]
            tok, model = load(job, "cuda")
            probe = reference_probe(job, tok, model, "cuda")
            cache = {}

            def make(i, row):
                start = i // 32 * 32
                if cache.get("start") != start:
                    ll = infer(
                        tok,
                        model,
                        [r["text"] for r in rr[start : start + 32]],
                        job["meta"]["lang"],
                        "cuda",
                    )
                    cache.update(
                        start=start,
                        logits=ll,
                        decisions=en.decisions(ll, job["meta"]["kind"]),
                    )
                return {
                    "key": row["key"],
                    "text_sha256": hashlib.sha256(row["text"].encode()).hexdigest(),
                    "logits": cache["logits"][i - start].tolist(),
                    "decision": cache["decisions"][i - start].tolist(),
                    "labels": job["meta"]["labels"],
                    "job": job["job"],
                    "checkpoint_sha256": job["checkpoint_sha256"],
                }

            values, receipt = chunks.gather(
                root / "chunks",
                rr,
                make,
                {"plan_sha256": c.sha(PLAN), "job": job["job"]},
                chunk_size=512,
                progress=lambda n, total, reused: print(
                    f"{job['job']}: {n}/{total}, {reused} verified/reused", flush=True
                ),
            )
            c.save_rows(root / "heads.jsonl", values)
            c.save(
                done,
                {
                    "status": "fixed_checkpoint_inference_complete",
                    "plan_sha256": c.sha(PLAN),
                    "job": job["job"],
                    "rows": len(values),
                    "probe": probe,
                    "outputs": {
                        c.rel(root / "heads.jsonl"): c.sha(root / "heads.jsonl"),
                        **receipt["hashes"],
                    },
                },
            )
            outputs[c.rel(done)] = c.sha(done)
            del model, tok
            gc.collect()
            torch.cuda.empty_cache()
        c.save(
            BASE / "complete.json",
            {
                "status": "all_fixed_heads_complete",
                "plan_sha256": c.sha(PLAN),
                "jobs": 30,
                "head_predictions": sum((j["rows"] for j in p["jobs"])),
                "outputs": outputs,
                "new_training_steps": 0,
                "Qwen_calls": 0,
            },
        )
    print(
        "All fixed encoder heads complete. Offline result verification and assembly required before scoring."
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "preflight", "validate", "run"])
    args = p.parse_args()
    with c.offline():
        if args.mode == "prepare":
            prepare()
        elif args.mode == "preflight":
            preflight()
        elif args.mode == "validate":
            validate()
            print("Fixed-checkpoint inference plan validated.")
        else:
            run()

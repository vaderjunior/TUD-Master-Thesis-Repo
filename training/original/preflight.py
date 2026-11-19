import gc
import json
from pathlib import Path

import numpy as np

from training.original import run as m
from training.encoder import macro_f1

OUT = m.BASE / "preflight.json"


def main():
    m.require(not OUT.exists(), "Preserve completed preflight receipt")
    plan = m.validate()
    done = m.done_jobs()
    expected = {("smoke", m.identity(j)) for j in plan["smoke_jobs"]}
    m.require(
        set(done) == expected and len(expected) == 5,
        "Expected five fresh English smokes, no full/reuse jobs yet",
    )
    hashes = {
        m.rel(p): m.sha(p)
        for p in (
            m.PLAN,
            Path(__file__),
            m.TEST,
            m.GUARD,
            m.AUDIT,
            m.BASE / "tokenizer_preflight.json",
        )
    }
    results = []
    for job in plan["smoke_jobs"]:
        entry = done[("smoke", m.identity(job))]
        attempt = m.ROOT / entry["attempt"]
        receipt = m.completion(attempt)
        hashes[m.rel(attempt / "complete.json")] = m.sha(attempt / "complete.json")
        hashes.update(receipt["files"])
        training, verify = m.read(attempt / "training.json"), m.read(
            attempt / "verification.json"
        )
        meta, frames, targets = m.data_for(job, smoke=True)
        rows = [
            json.loads(x)
            for x in (attempt / "internal_val_logits.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        m.require(
            [r["id"] for r in rows] == frames["val"].id.astype(str).tolist(),
            "Wrong smoke validation identities",
        )
        logits = np.asarray([r["logits"] for r in rows])
        pred = np.asarray([r["decision"] for r in rows])
        m.require(
            logits.shape == (len(rows), len(meta["labels"]))
            and np.isfinite(logits).all(),
            "Bad smoke logits",
        )
        np.testing.assert_array_equal([r["target"] for r in rows], targets["val"])
        np.testing.assert_array_equal(pred, m.decisions(logits, meta["kind"]))
        score = macro_f1(
            targets["val"], pred, meta["kind"] == "multilabel", len(meta["labels"])
        )
        m.require(
            abs(score - verify["internal_val_macro_f1_fp32"]) < 1e-12,
            "Smoke metric differs",
        )
        state, trace = training["trainer_state"], training["amp_scale_trace"]
        m.require(
            state["global_step"] == 8 and training["classifier_max_change"] > 0,
            "Smoke did not train",
        )
        m.require(
            training["diagnostic_only"]
            and receipt["dev_rows"] == verify["dev_rows"] == 0,
            "Smoke left diagnostic scope",
        )
        m.require(
            not (attempt / "dev_head_logits.jsonl").exists(), "Unexpected DEV inference"
        )
        m.require(verify["reload_max_difference"] <= 1e-5, "Reload differs")
        for r in (training["tokenizer_guard"], verify["tokenizer_guard"]):
            m.require(
                r["status"] == "pass"
                and r["semantic_sha256"] == m.token_guard.SEMANTIC_SHA,
                "Semantic guard mismatch",
            )
        selected = attempt / "selected_model"
        tok = m.token_guard.load(selected)
        for split in ("train", "val"):
            enc = m.token_guard.encode(tok, frames[split].text.astype(str).tolist())
            m.require(
                training["token_stats"][split]
                == {
                    "rows": len(frames[split]),
                    "at_max_length": sum(len(x) == 128 for x in enc["input_ids"]),
                },
                "Stored token lengths differ from reference",
            )
        if job["arm"] != "kb":
            m.require(
                m.sha(Path(state["best_model_checkpoint"]) / "model.safetensors")
                == m.sha(selected / "model.safetensors"),
                "Wrong selected weights",
            )
        else:
            m.require(
                state["best_model_checkpoint"] is None, "Compact selection changed"
            )
        m.require(
            len(trace) == 8 and [r["step"] for r in trace] == list(range(1, 9)),
            "Missing scale trace",
        )
        for i, r in enumerate(trace):
            m.require(
                r["overflow_skipped"]
                == m.overflow_skipped(
                    r["scale_before"], r["scale_after"], r["accelerate_skip_flag"]
                ),
                "Wrong AMP skip",
            )
            if i:
                m.require(
                    r["scale_before"] == trace[i - 1]["scale_after"],
                    "Broken AMP scale continuity",
                )
        m.require(
            sum(r["overflow_skipped"] for r in trace) == training["amp_skipped_steps"],
            "Wrong skip total",
        )
        results.append(
            {
                "job": m.identity(job),
                "attempt": m.rel(attempt),
                "steps": 8,
                "classifier_max_change": training["classifier_max_change"],
                "reload_max_difference": verify["reload_max_difference"],
                "amp_skipped_steps": training["amp_skipped_steps"],
                "peak_allocated_gib": training["peak_allocated_gib"],
                "internal_val_rows": len(rows),
                "diagnostic_f1": score,
                "tokenizer_guard": verify["tokenizer_guard"],
            }
        )
    print(
        "Five corrected English smoke attempts verified against independent token IDs.",
        flush=True,
    )
    torch = m.setup_cuda()
    reused = []
    with m.lock(".batch.lock"), m.lock(".gpu.lock"):
        for job in plan["reuse_jobs"]:
            meta, frames, _ = m.data_for(job)
            path = m.ROOT / job["path"]
            old = m.read(path / "meta.json")
            m.require(
                old["labels"] == meta["labels"]
                and old["kind"] == meta["kind"]
                and old["lang"] == "en",
                "Wrong reused metadata",
            )
            m.require(
                old["seed"] == job["seed"]
                and old["epochs"] == 20
                and old["early_stopping"] is False,
                "Wrong reuse recipe",
            )
            tok, model, info = m.load_model(path, meta)
            model.to("cuda")
            texts = frames["val"].text.astype(str).tolist()[:8]
            logits = m.infer(model, tok, texts, 8)

            ref = m.token_guard.reference_encoding(texts, padding=True)
            batch = {
                k: torch.tensor(v, dtype=torch.long, device="cuda")
                for k, v in ref.items()
            }
            with torch.inference_mode():
                reference_logits = model(**batch).logits.float().cpu().numpy()
            m.require(
                np.isfinite(logits).all()
                and logits.shape == (len(texts), len(meta["labels"])),
                "Invalid reuse logits",
            )
            np.testing.assert_allclose(logits, reference_logits, atol=1e-5, rtol=1e-5)
            reused.append(
                {
                    "job": m.identity(job),
                    "checkpoint": m.rel(path),
                    "checkpoint_sha256": m.sha(path / "model.safetensors"),
                    "probe_ids": frames["val"].id.astype(str).tolist()[:8],
                    "probe_logits": logits.tolist(),
                    "reference_logits": reference_logits.tolist(),
                    "reference_max_difference": float(
                        np.abs(logits - reference_logits).max()
                    ),
                    "loading_info": m.json_ready(info),
                    "tokenizer_guard": m.token_guard.check(tok),
                    "scope": "First up to eight TRAIN-internal VAL examples; no DEV inference or performance comparison.",
                }
            )
            for p in path.iterdir():
                if p.is_file():
                    hashes[m.rel(p)] = m.sha(p)
            print(
                f"Reused head matches independent reference: {m.identity(job)}",
                flush=True,
            )
            del model, tok, batch
            gc.collect()
            torch.cuda.empty_cache()
    m.validate()
    m.verify_hashes(hashes)
    record = {
        "id": "ASTRA-CORE-ENCODER-PREFLIGHT-20260915-v5",
        "status": "pass",
        "created_utc": m.now(),
        "execution_sha256": m.sha(m.PLAN),
        "input_hashes": hashes,
        "verified_smokes": results,
        "reused_head_probes": reused,
        "retained_german": plan["inherited_full"],
        "scope": "Five eight-step English training diagnostics, independent stored-output checks and nine read-only original-head internal-validation probes. No full fitting or DEV/TEST inference/scoring.",
        "limitations": "Short diagnostics cannot guarantee full-training stability or accuracy; original trainer-source/fixed-epoch reuse qualification remains.",
    }
    m.write(OUT, record)
    m.require_preflight()
    print(
        json.dumps(
            {
                "status": "ready_for_user_batch",
                "receipt": m.rel(OUT),
                "sha256": m.sha(OUT),
                "hashes": len(hashes),
                "smokes": len(results),
                "reuse_probes": len(reused),
                "max_reload_difference": max(
                    r["reload_max_difference"] for r in results
                ),
                "max_reference_difference": max(
                    r["reference_max_difference"] for r in reused
                ),
                "max_peak_allocated_gib": max(r["peak_allocated_gib"] for r in results),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

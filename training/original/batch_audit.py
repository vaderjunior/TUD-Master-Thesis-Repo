from __future__ import annotations
from src.hsrag.paths import PROJECT_ROOT

import hashlib
import json
import math
import os
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev

os.environ.update(
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false"
)
ROOT = PROJECT_ROOT
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
BASE = CORE / "encoders_v4"
OUT = (
    ROOT / "Thesis_Log/astra_result/evidence/core_encoder_batch_audit_20260915_v1.json"
)
TABLE = ROOT / "Thesis_Log/astra_result/astra_core_encoder_audit_20260915_v1.md"
DERIVED = CORE / "encoder_audit_v1"
HASHES = {}


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def rows(p):
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def name(p):
    p = Path(p).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def sha(p):
    p = Path(p).resolve()
    key = name(p)
    if key not in HASHES:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda: f.read(1024 * 1024), b""):
                h.update(b)
        HASHES[key] = h.hexdigest()
    return HASHES[key]


def fingerprint(obj):
    return hashlib.sha256(
        json.dumps(
            obj, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def structural(tok):
    d = json.loads(tok.backend_tokenizer.to_str())
    d.pop("truncation", None)
    d.pop("padding", None)
    return {
        "class": type(tok).__name__,
        "vocab_size": len(tok),
        "backend_sha256": fingerprint(d),
        "bpe_merges": len(d["model"].get("merges", [])),
        "model_type": d["model"]["type"],
        "pre_tokenizer": d["pre_tokenizer"],
        "normalizer": d["normalizer"],
        "model_sha256": fingerprint(d["model"]),
        "post_processor": d["post_processor"],
    }


def token_ids(tok, texts):
    return tok(texts, truncation=True, max_length=128, padding=False)["input_ids"]


def compare(a, b):
    require(len(a) == len(b), "Tokenizer row count differs")
    mismatches = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    return {
        "rows": len(a),
        "different_rows": len(mismatches),
        "reference_ids_sha256": fingerprint(a),
        "candidate_ids_sha256": fingerprint(b),
        "reference_at_128": sum(len(x) == 128 for x in a),
        "candidate_at_128": sum(len(x) == 128 for x in b),
        "reference_mean_length": mean(map(len, a)),
        "candidate_mean_length": mean(map(len, b)),
    }


def tokenizer_audit(plan, entries):
    import pandas as pd
    from tokenizers import Tokenizer
    from transformers import (
        AutoTokenizer,
        RobertaTokenizer,
        RobertaTokenizerFast,
        BertTokenizer,
        BertTokenizerFast,
    )

    dev = rows(CORE / "dev_inputs.jsonl")
    result = {
        "backends": {},
        "partitions": {},
        "saved_models": {},
        "known_loading_failure": {},
    }
    refs, candidates = {}, {}
    for lang, model, cls, correct, arm in [
        ("en", "roberta-base", RobertaTokenizerFast, RobertaTokenizer, "full"),
        ("de", "deepset/gbert-base", BertTokenizerFast, BertTokenizer, "de"),
    ]:
        cache = Path(plan["base_caches"][model]["snapshot"])
        reference_path = ROOT / f"models/encoder/{arm}/hate/seed0"
        for p in [
            reference_path / "tokenizer.json",
            reference_path / "tokenizer_config.json",
        ]:
            sha(p)
        refs[lang] = AutoTokenizer.from_pretrained(
            reference_path, local_files_only=True
        )
        candidates[lang] = cls.from_pretrained(cache, local_files_only=True)
        intended = correct.from_pretrained(cache, local_files_only=True)
        result["backends"][lang] = {
            "reference": structural(refs[lang]),
            "v4_base": structural(candidates[lang]),
            "corrected_base": structural(intended),
        }
        try:
            AutoTokenizer.from_pretrained(cache, local_files_only=True)
        except Exception as e:
            result["known_loading_failure"][lang] = {
                "class": type(e).__name__,
                "message": str(e),
                "resolution": "Explicit architecture tokenizer class without Fast suffix loads locally; not an environment upgrade.",
            }
        scopes = [
            (name(p), pd.read_parquet(p).text.astype(str).tolist())
            for a in (["full", "kb"] if lang == "en" else ["de"])
            for p in sorted((CORE / "encoder_data" / a).glob("*.parquet"))
        ]
        scopes.append((f"{lang}/DEV", [x["text"] for x in dev if x["lang"] == lang]))

        scopes.append(
            (
                f"{lang}/neutral_probes",
                [
                    "This is a simple sentence.",
                    " leading  spaces\nnext\tline",
                    "Äpfel, Grüße und Flüchtlinge.",
                    "hello🙂 world",
                    "<mask> [CLS] <s> [UNK]",
                    "",
                    "x" * 600,
                ],
            )
        )
        backend = Tokenizer.from_file(str(reference_path / "tokenizer.json"))
        backend.no_padding()
        backend.enable_truncation(max_length=128)
        pinned = None
        if lang == "en":
            pinned = Tokenizer.from_file(str(cache / "tokenizer.json"))
            pinned.no_padding()
            pinned.enable_truncation(max_length=128)
        for scope, texts in scopes:
            if scope.startswith("experiments/"):
                sha(ROOT / scope)
            ref = token_ids(refs[lang], texts)
            original_direct = [x.ids for x in backend.encode_batch(texts)]
            require(
                ref == original_direct,
                "Auto historical tokenizer differs from its serialized backend",
            )
            require(
                ref == token_ids(intended, texts),
                f"Corrected tokenizer differs from historical on {scope}",
            )
            if pinned:
                require(
                    ref == [x.ids for x in pinned.encode_batch(texts)],
                    f"Pinned EN tokenizer differs on {scope}",
                )
            result["partitions"][scope] = compare(
                ref, token_ids(candidates[lang], texts)
            )
            result["partitions"][scope]["corrected_equals_reference"] = True
            if lang == "de":
                require(
                    result["partitions"][scope]["different_rows"] == 0,
                    "German tokenizer affected",
                )
        print(f"Tokenizer partitions checked: {lang}", flush=True)
    for entry in entries:
        if entry["mode"] == "smoke":
            continue
        a = ROOT / entry["attempt"]
        v = read(a / "verification.json")
        path = ROOT / v["checkpoint"]
        lang = "de" if entry["job"].startswith("de/") else "en"
        cls = BertTokenizerFast if lang == "de" else RobertaTokenizerFast
        loaded = cls.from_pretrained(path, local_files_only=True)
        texts = [x["text"] for x in dev if x["lang"] == lang]
        for p in [path / "tokenizer.json", path / "tokenizer_config.json"]:
            sha(p)
        saved = {
            "checkpoint": name(path),
            "backend": structural(loaded),
            "DEV_vs_original": compare(
                token_ids(refs[lang], texts), token_ids(loaded, texts)
            ),
            "DEV_vs_v4_base": compare(
                token_ids(candidates[lang], texts), token_ids(loaded, texts)
            ),
        }
        if entry["mode"] == "reuse":
            original = AutoTokenizer.from_pretrained(path, local_files_only=True)
            require(
                token_ids(original, texts) == token_ids(refs[lang], texts),
                "Original reused tokenizer differs",
            )
            saved["original_auto_loader_equals_reference"] = True
        if lang == "de":
            require(
                saved["DEV_vs_original"]["different_rows"] == 0,
                "Saved German tokenizer changed",
            )
        else:
            require(
                saved["DEV_vs_v4_base"]["different_rows"] == 0,
                "Affected tokenizer changed between train/reload",
            )
        result["saved_models"][entry["job"]] = saved
    return result


def training_audit(entries, execution):
    import numpy as np
    from training.original import legacy as runner
    from training.encoder import macro_f1

    dev = rows(CORE / "dev_inputs.jsonl")
    report, raw = [], {}
    for entry in entries:
        a = ROOT / entry["attempt"]
        receipt = read(a / "complete.json")
        require(
            sha(a / "complete.json") == entry["receipt_sha256"], "Batch receipt differs"
        )
        require(
            receipt["job"] == entry["job"] and receipt["mode"] == entry["mode"],
            "Wrong completed job",
        )
        for file, digest in receipt["files"].items():
            require(sha(ROOT / file) == digest, f"Artifact changed: {file}")
        require(
            sha(a / "source_snapshot.py") == sha(ROOT / "training/original/legacy.py"),
            "Worker source changed",
        )
        if entry["mode"] == "smoke":
            continue
        arm, head, seed = entry["job"].split("/")
        job = {"arm": arm, "head": head, "seed": int(seed.removeprefix("seed"))}
        meta, frames, targets = runner.data_for(job)
        v = read(a / "verification.json")
        saved = rows(a / "internal_val_logits.jsonl")
        require(
            [x["id"] for x in saved] == frames["val"].id.astype(str).tolist(),
            "Validation identities differ",
        )
        logits = np.asarray([x["logits"] for x in saved])
        require(
            logits.shape == (len(saved), len(meta["labels"]))
            and np.isfinite(logits).all(),
            "Invalid validation logits",
        )
        prediction = (
            (logits >= 0).astype(int)
            if meta["kind"] == "multilabel"
            else logits.argmax(-1)
        )
        require(
            np.array_equal([x["target"] for x in saved], targets["val"]),
            "Wrong validation targets",
        )
        require(
            np.array_equal([x["decision"] for x in saved], prediction),
            "Wrong validation decisions",
        )
        val = macro_f1(
            targets["val"],
            prediction,
            meta["kind"] == "multilabel",
            len(meta["labels"]),
        )
        require(
            abs(val - v["internal_val_macro_f1_fp32"]) < 1e-12,
            "Validation score differs",
        )
        checkpoint = ROOT / v["checkpoint"]
        require(
            sha(checkpoint / "model.safetensors") == v["checkpoint_sha256"],
            "Checkpoint hash differs",
        )
        rr = rows(a / "dev_head_logits.jsonl")
        expected = [x for x in dev if x["lang"] == meta["lang"]]
        require(
            [x["key"] for x in rr] == [x["key"] for x in expected],
            "DEV identity/order differs",
        )
        require(receipt["dev_rows"] == v["dev_rows"] == len(rr), "DEV counts differ")
        for x, source in zip(rr, expected):
            require(
                x["text_sha256"] == hashlib.sha256(source["text"].encode()).hexdigest(),
                "DEV text differs",
            )
            require(
                x["labels"] == meta["labels"] and x["job"] == entry["job"],
                "DEV label/job mismatch",
            )
            require(
                x["checkpoint_sha256"] == v["checkpoint_sha256"]
                and x["checkpoint"] == v["checkpoint"],
                "DEV weights differ",
            )
            require(
                x["execution_sha256"] == sha(BASE / "execution.json"),
                "DEV execution mismatch",
            )
            ll = np.asarray(x["logits"])
            require(
                ll.shape == (len(meta["labels"]),) and np.isfinite(ll).all(),
                "Invalid DEV logits",
            )
            dd = (
                (ll >= 0).astype(int).tolist()
                if meta["kind"] == "multilabel"
                else int(ll.argmax())
            )
            require(dd == x["decision"], "DEV decision differs from raw logits")
        raw[entry["job"]] = rr
        historical = ROOT / f"models/encoder/{arm}/{head}/{seed}/meta.json"
        sha(historical)
        summary = {
            "job": entry["job"],
            "mode": entry["mode"],
            "attempt": name(a),
            "internal_val_macro_f1_fp32": val,
            "old_internal_val_f1_historical": read(historical)["val_macro_f1"],
            "internal_val_rows": len(saved),
            "dev_rows": len(rr),
            "checkpoint": name(checkpoint),
            "reload_max_difference": v["reload_max_difference"],
            "status": (
                "retain"
                if arm == "de"
                else (
                    "regenerate_predictions_only"
                    if entry["mode"] == "reuse"
                    else "quarantine_retrain"
                )
            ),
        }
        if entry["mode"] == "full":
            t = read(a / "training.json")
            state, trace = t["trainer_state"], t["amp_scale_trace"]
            require(
                t["job"]["seed"] == job["seed"]
                and t["job"]["arm"] == arm
                and t["job"]["head"] == head,
                "Training job mismatch",
            )
            require(
                not t["diagnostic_only"] and len(trace) == state["global_step"],
                "Diagnostic/steps mismatch",
            )
            skipped, consecutive, longest = 0, 0, 0
            for i, step in enumerate(trace):
                require(step["step"] == i + 1, "AMP trace sequence differs")
                if i:
                    require(
                        step["scale_before"] == trace[i - 1]["scale_after"],
                        "AMP scale continuity differs",
                    )
                skip = bool(
                    step["accelerate_skip_flag"]
                    or step["scale_after"] < step["scale_before"]
                )
                require(
                    skip == step["overflow_skipped"], "AMP skip classification differs"
                )
                require(
                    all(
                        math.isfinite(step[k]) and step[k] > 0
                        for k in ("scale_before", "scale_after")
                    ),
                    "Invalid scale",
                )
                skipped += skip
                consecutive = consecutive + 1 if skip else 0
                longest = max(longest, consecutive)
            require(
                skipped == t["amp_skipped_steps"] and longest <= 20,
                "AMP skip total/guard differs",
            )
            require(
                v["reload_max_difference"] == 0.0
                and math.isfinite(t["training_loss"])
                and t["classifier_max_change"] > 0,
                "Numerical checks failed",
            )
            evaluations = [x for x in state["log_history"] if "eval_macro_f1" in x]
            if arm != "kb":
                best = max(evaluations, key=lambda x: x["eval_macro_f1"])
                require(
                    best["step"] == state["best_global_step"]
                    and best["eval_macro_f1"] == state["best_metric"],
                    "Wrong validation-best selection",
                )
                require(
                    Path(state["best_model_checkpoint"]).name
                    == f"checkpoint-{best['step']}",
                    "Selected checkpoint step differs",
                )
                require(
                    sha(Path(state["best_model_checkpoint"]) / "model.safetensors")
                    == v["checkpoint_sha256"],
                    "Selected model not best weights",
                )
                require(
                    state["epoch"] == t["job"]["epochs"]
                    or len(evaluations) - evaluations.index(best) - 1 == 2,
                    "Early stopping not patience two",
                )
            else:
                require(
                    state["epoch"] == 20 and state["best_model_checkpoint"] is None,
                    "Compact fixed-last selection changed",
                )
            summary.update(
                epochs=state["epoch"],
                steps=state["global_step"],
                selected_step=(
                    state["best_global_step"] if arm != "kb" else state["global_step"]
                ),
                selected_internal_val_f1_amp=(
                    state["best_metric"]
                    if arm != "kb"
                    else evaluations[-1]["eval_macro_f1"]
                ),
                amp_skipped_steps=skipped,
                max_consecutive_skips=longest,
                token_stats=t["token_stats"],
                train_seconds=t["train_seconds"],
                peak_allocated_gib=t["peak_allocated_gib"],
            )
        report.append(summary)
        print(f'Artifacts/selection/outputs verified: {entry["job"]}', flush=True)
    return report, raw


def score_german(raw):
    import numpy as np
    import pandas as pd
    from src.hsrag.metrics import score_gate, score_multilabel, clean
    from src.hsrag.schema import Result

    dev = [x for x in rows(CORE / "dev_inputs.jsonl") if x["lang"] == "de"]
    metrics = {}
    for subset, head in [("de_dev_eval", "hate"), ("de_legal_dev_eval", "legal")]:
        path = ROOT / f"data/processed/{subset}.parquet"
        sha(path)
        df = pd.read_parquet(path)
        gold = {str(r.id): r for r in df.itertuples(index=False)}
        roster = [x for x in dev if any(s["subset"] == subset for s in x["scopes"])]
        require(
            set(gold) == {x["item_id"] for x in roster}, "German DEV roster differs"
        )
        per_seed, all_old, all_new = [], [], []
        for seed in range(3):
            gates = {x["key"]: x for x in raw[f"de/hate/seed{seed}"]}
            legal = {x["key"]: x for x in raw[f"de/legal/seed{seed}"]}
            current = []
            for x in roster:
                g, l = gates[x["key"]], legal[x["key"]]
                result = Result(
                    reasoning=f"encoder de seed{seed}; v4 audit assembly",
                    hate=bool(g["decision"]),
                    legal=[v for v, on in zip(l["labels"], l["decision"]) if on],
                ).model_dump()
                current.append(
                    {
                        "item_id": x["item_id"],
                        "key": x["key"],
                        "subset": subset,
                        "seed": seed,
                        "result": result,
                        "uncertain": False,
                        "gate_normalization": "Result schema; legal independent",
                        "source_gate_sha256": sha(
                            ROOT / gates[x["key"]]["checkpoint"] / "model.safetensors"
                        ),
                        "source_legal_sha256": sha(
                            ROOT / legal[x["key"]]["checkpoint"] / "model.safetensors"
                        ),
                    }
                )
            oldpath = ROOT / f"experiments/results/encoder_de_seed{seed}_live.jsonl"
            sha(oldpath)
            old = [
                x
                for x in rows(oldpath)
                if x.get("encoder_meta", {}).get("subset") == subset
            ]
            require(
                len(old) == len(gold) and {x["item_id"] for x in old} == set(gold),
                "Historical German DEV differs",
            )
            old = {x["item_id"]: x for x in old}
            for x in roster:
                h = hashlib.sha256(" ".join(x["text"].split()).encode()).hexdigest()[
                    :16
                ]
                require(
                    old[x["item_id"]]["text_hash"] == h, "Historical DEV text differs"
                )
            old = [old[x["item_id"]] for x in roster]
            scorer = (
                (lambda rr: asdict(score_gate(rr, gold)))
                if head == "hate"
                else (lambda rr: asdict(score_multilabel(rr, gold, "legal", "legal")))
            )
            oldscore, newscore = scorer(old), scorer(current)
            require(
                (newscore["n_scored"] if head == "hate" else newscore["n_items"])
                == len(roster),
                "Unexpected German score mask",
            )
            require(
                newscore["n_unscored"] == oldscore["n_unscored"] == 0,
                "Missing German predictions",
            )
            per_seed.append(
                {
                    "seed": seed,
                    "old": oldscore,
                    "new": newscore,
                    "delta": newscore["macro_f1"] - oldscore["macro_f1"],
                }
            )
            output = DERIVED / f"de_seed{seed}_{subset}_assembled.jsonl"
            with output.open("x", encoding="utf-8", newline="\n") as f:
                for row in current:
                    f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            sha(output)
            if head == "hate":
                truth = np.asarray(
                    [bool(gold[x["item_id"]].gate) for x in roster], dtype=int
                )[:, None]
                new_array = np.asarray(
                    [x["result"]["hate"] for x in current], dtype=int
                )[:, None]
                old_array = np.asarray([x["result"]["hate"] for x in old], dtype=int)[
                    :, None
                ]
                truth = np.concatenate([1 - truth, truth], axis=1)
                new_array = np.concatenate([1 - new_array, new_array], axis=1)
                old_array = np.concatenate([1 - old_array, old_array], axis=1)
            else:
                labels = newscore["labels_averaged"]
                require(
                    labels == oldscore["labels_averaged"], "Macro label mask changed"
                )
                truth = np.asarray(
                    [
                        [v in clean(gold[x["item_id"]].legal) for v in labels]
                        for x in roster
                    ],
                    dtype=int,
                )
                new_array = np.asarray(
                    [[v in x["result"]["legal"] for v in labels] for x in current],
                    dtype=int,
                )
                old_array = np.asarray(
                    [[v in x["result"]["legal"] for v in labels] for x in old],
                    dtype=int,
                )
            all_old.append(old_array)
            all_new.append(new_array)

        def sample_f1(pred, idx):
            yy, pp = truth[idx], pred[:, idx, :]
            tp = (pp * yy).sum(axis=1)
            den = pp.sum(axis=1) + yy.sum(axis=0)
            return np.divide(
                2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0
            ).mean(axis=1)

        all_old, all_new = np.asarray(all_old), np.asarray(all_new)
        ix = np.arange(len(roster))
        require(
            np.allclose(
                sample_f1(all_new, ix),
                [x["new"]["macro_f1"] for x in per_seed],
                atol=1e-12,
            ),
            "Independent current score differs",
        )
        require(
            np.allclose(
                sample_f1(all_old, ix),
                [x["old"]["macro_f1"] for x in per_seed],
                atol=1e-12,
            ),
            "Independent historical score differs",
        )
        rng = np.random.default_rng(20260915)
        draws = []
        for _ in range(10000):
            idx = rng.integers(0, len(roster), len(roster))
            draws.append(
                float((sample_f1(all_new, idx) - sample_f1(all_old, idx)).mean())
            )
        metrics[subset] = {
            "head": head,
            "n": len(roster),
            "per_seed": per_seed,
            "old_mean": mean(x["old"]["macro_f1"] for x in per_seed),
            "old_sample_sd": stdev(x["old"]["macro_f1"] for x in per_seed),
            "new_mean": mean(x["new"]["macro_f1"] for x in per_seed),
            "new_sample_sd": stdev(x["new"]["macro_f1"] for x in per_seed),
            "delta_mean": mean(x["delta"] for x in per_seed),
            "paired_item_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
            "bootstrap": "10,000 paired item resamples shared across old/new and all three seeds; seed 20260915. Fixed original supported macro-label set, zero F1 for zero denominator. Conditional on these three fitted seeds; not training-seed or source-cluster uncertainty. No multiplicity adjustment.",
            "interpretation_limit": "Aligned cleaned-data refits versus historical models; different internal partitions and historical precision/source limitations. Post-selection DEV evidence, not an isolated data-removal causal effect.",
        }
    return metrics


def main():
    require(
        not OUT.exists() and not TABLE.exists() and not DERIVED.exists(),
        "Audit outputs already exist; preserve them",
    )
    os.chdir(ROOT)
    sha(Path(__file__))
    plan = read(CORE / "plan.json")
    execution = read(BASE / "execution.json")

    from training.original import legacy as runner

    runner.validate()
    for doc in [
        CORE / "plan.json",
        CORE / "encoder_plan.json",
        BASE / "execution.json",
        CORE / "dev_inputs.jsonl",
        ROOT
        / "Thesis_Log/astra_result/evidence/core_preparation_check_20260915_v1.json",
    ]:
        sha(doc)
    for digest_map in [
        plan["input_hashes"],
        plan["generated_hashes"],
        execution["input_hashes"],
    ]:
        for path, expected in digest_map.items():
            require(sha(ROOT / path) == expected, f"Frozen input changed: {path}")
    for cache in plan["base_caches"].values():
        for file, expected in cache["files"].items():
            require(
                sha(Path(cache["snapshot"]) / file) == expected, "Base cache changed"
            )
    receiptpath = BASE / "batch_receipts/1789468001939697400.json"
    batch = read(receiptpath)
    sha(receiptpath)
    entries = batch["jobs"]
    require(
        Counter(x["mode"] for x in entries) == {"full": 21, "reuse": 9, "smoke": 1},
        "Wrong batch counts",
    )
    require(len({(x["mode"], x["job"]) for x in entries}) == 31, "Duplicate jobs")
    require(
        not list((BASE / "attempts").glob("*/failure.json")),
        "Unexpected v4 failed attempt",
    )
    require(
        {x["job"] for x in entries if x["mode"] == "full"}
        == {runner.identity(x) for x in execution["train_jobs"]},
        "Training roster mismatch",
    )
    require(
        {x["job"] for x in entries if x["mode"] == "reuse"}
        == {runner.identity(x) for x in execution["reuse_jobs"]},
        "Reuse roster mismatch",
    )
    print("Frozen preparation/cache and batch roster verified.", flush=True)
    jobs, raw = training_audit(entries, execution)
    tokenizer = tokenizer_audit(plan, entries)
    DERIVED.mkdir(parents=True, exist_ok=False)
    metrics = score_german(raw)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "audit_complete_english_quarantined_german_retained",
        "mechanical_status": "21 training jobs + nine reuse predictions complete; all 25,230 raw DEV head rows verified",
        "batch_receipt": name(receiptpath),
        "batch_completed_utc": batch["completed_utc"],
        "scope": "Offline tokenizer construction, file hashing, stored-logit/selection checks and German DEV scoring only. No neural model loaded, training, API, TEST/HateCheck scoring, Priority 3 or original-file writes.",
        "defect": "Explicit RobertaTokenizerFast in this installed environment drops 50,000 BPE merges and ByteLevel pre-tokenizer. All 15 English refits used unintended token IDs; all nine reused English heads were also predicted with the defective loader. Same vocabulary size and successful FP32 reload did not detect this.",
        "responsibility": "Astra runner/preflight defect; not user execution, evidence of poor RoBERTa, or a negative full-KB result.",
        "jobs": jobs,
        "tokenizer": tokenizer,
        "german_dev": metrics,
        "next": {
            "retrain": [x["job"] for x in jobs if x["status"] == "quarantine_retrain"],
            "predict_only": [
                x["job"] for x in jobs if x["status"] == "regenerate_predictions_only"
            ],
            "retain": [x["job"] for x in jobs if x["status"] == "retain"],
            "guard": "New version only: explicit RobertaTokenizer (without Fast suffix) matches pinned tokenizer.json and historical AutoTokenizer on all prepared partitions and DEV texts. Require backend BPE/merge/pre-tokenizer identity and token IDs at base load, every saved reload and reused-head load. Keep German completed runs and original English reuse weights.",
            "not_done": "No corrected runner, CUDA preflight or rerun yet. Pause at completed diagnosis/audit; next bounded stage prepares the repair and one user batch.",
        },
        "verified_hashes": dict(sorted(HASHES.items())),
    }
    with OUT.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    doc = [
        "# Core encoder batch audit — 2026-09-15",
        "",
        "**Status: audit complete; English v4 results quarantined, German runs retained.**",
        "",
        "The user batch completed. The English runner used a defective tokenizer loader: 0 BPE merges instead of 50,000 and no ByteLevel pre-tokenizer. Vocabulary size remained 50,265, so dimensions/loading/finite-output checks passed. Training and save/reload repeated the same defect, explaining why those checks did not reveal it. This is an Astra implementation error, not a user execution error or evidence against the full-KB decision.",
        "",
        "Preserve all v4 attempts as engineering evidence. Fifteen new English models require refitting; nine old compact fine-head weights remain intact but their v4 predictions require regeneration. All six German models have matching tokenization and pass the stored-output audit. No long rerun was launched.",
        "",
        "## Verified scope",
        "",
        "- All 21 training and nine reuse jobs; original/prepared/cache hashes and every saved receipt artifact.",
        "- All 25,230 raw DEV head rows: exact order, text hashes, label order, finite logits, decisions and checkpoint provenance.",
        "- Internal-validation identities/targets/scores, best-checkpoint selection or compact fixed-final selection, full AMP traces and zero-difference reload probes.",
        "- Token IDs on every prepared TRAIN/internal-validation partition and both DEV text rosters; all 30 final loaded tokenizers checked.",
        "- Only unaffected German DEV predictions assembled and scored. English validation values below are diagnostic and cannot be used as thesis comparisons. TEST/HateCheck, RAG/API, Priority 3 and LaTeX remain parked.",
        "",
        "## Tokenizer evidence",
        "",
        "| Partition | Rows | Different token IDs | Correct at length 128 | v4 at length 128 |",
        "|---|---:|---:|---:|---:|",
    ]
    for scope, val in tokenizer["partitions"].items():
        if scope.endswith("neutral_probes"):
            continue
        doc.append(
            f"| {scope.replace('experiments/astra/full_kb_core_20260915_v1/encoder_data/', '')} | {val['rows']} | {val['different_rows']} | {val['reference_at_128']} | {val['candidate_at_128']} |"
        )
    doc += [
        "",
        "At-length counts include exact length 128; they are not all proven truncations. Corrected explicit `RobertaTokenizer` matches both the pinned serialized base and historical tokenizer on every checked English text. German explicit `BertTokenizerFast` resolves to the correct `BertTokenizer` and matches historical IDs on every checked German text. Generic AutoTokenizer still cannot load the bare German cache in this environment; do not apply a blanket AutoTokenizer substitution.",
        "",
        "## Per-job internal-validation diagnostics",
        "",
        "Old values come from saved historical metadata and use the original internal-validation partitions/precision; new values use cleaned partitions and FP32 standalone inference. They are not paired accuracy comparisons. English v4 rows are invalid for the intended baseline. Compact fixed-20 training has no validation selection.",
        "",
        "| Job | Status | Old internal F1 | v4 FP32 internal F1 | Epochs | Selected step | AMP skips |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for j in jobs:
        doc.append(
            f"| {j['job']} | {j['status']} | {j['old_internal_val_f1_historical']:.6f} | {j['internal_val_macro_f1_fp32']:.6f} | {j.get('epochs', 'reuse')} | {j.get('selected_step', '—')} | {j.get('amp_skipped_steps', '—')} |"
        )
    doc += [
        "",
        "## German DEV results",
        "",
        "Same frozen items and original scoring masks; legal is independent of the hate gate. All items have valid predictions. Means use seeds 0/1/2; SD is sample SD.",
        "",
        "| Head / DEV rows | Historical mean ± SD | Cleaned v4 mean ± SD | Difference | Paired item bootstrap 95% |",
        "|---|---:|---:|---:|---|",
    ]
    for scope, m in metrics.items():
        doc.append(
            f"| {m['head']} / {m['n']} | {m['old_mean']:.6f} ± {m['old_sample_sd']:.6f} | {m['new_mean']:.6f} ± {m['new_sample_sd']:.6f} | {m['delta_mean']:+.6f} | [{m['paired_item_bootstrap_95'][0]:+.6f}, {m['paired_item_bootstrap_95'][1]:+.6f}] |"
        )
    for scope, m in metrics.items():
        doc += [
            "",
            f"### {scope}",
            "",
            "| Seed | Old F1 | New F1 | Change |",
            "|---:|---:|---:|---:|",
        ]
        for s in m["per_seed"]:
            doc.append(
                f"| {s['seed']} | {s['old']['macro_f1']:.6f} | {s['new']['macro_f1']:.6f} | {s['delta']:+.6f} |"
            )
        if m["head"] == "hate":
            doc += ["", "| Seed | Old FP / FN | New FP / FN |", "|---:|---:|---:|"]
            for s in m["per_seed"]:
                doc.append(
                    f"| {s['seed']} | {s['old']['fp']} / {s['old']['fn']} | {s['new']['fp']} / {s['new']['fn']} |"
                )
        else:
            doc += [
                "",
                "| Legal label | Gold support | Old mean F1 | New mean F1 |",
                "|---|---:|---:|---:|",
            ]
            for label in m["per_seed"][0]["new"]["per_label"]:
                oo = [s["old"]["per_label"][label]["f1"] for s in m["per_seed"]]
                nn = [s["new"]["per_label"][label]["f1"] for s in m["per_seed"]]
                doc.append(
                    f"| {label} | {m['per_seed'][0]['new']['per_label'][label]['support']} | {mean(oo):.6f} | {mean(nn):.6f} |"
                )
        doc.append("")
    doc += [
        "Intervals use 10,000 paired item resamples, holding the three fitted seeds fixed and retaining the original supported label set. They exclude retraining/source-cluster uncertainty and are not multiplicity-adjusted. These are post-selection development results. Cleanup, fitting and historical inference/source limitations prevent claiming an isolated causal cleanup effect. The earlier GBERT/mmBERT comparison remains historical and is not replaced by these cleaned GBERT results.",
        "",
        "## Next bounded stage",
        "",
        "Prepare a separately versioned English-only repair (15 training jobs + nine prediction-only jobs), with semantic tokenizer checks against frozen references before training and after saving/loading. Retain all six German jobs. No hyperparameter tuning or data-policy change is justified by the defective scores. Run short correctness checks here; give the long repaired batch to Ajay as one resumable command. No corrected runner or rerun has been prepared/launched at this audit checkpoint.",
        "",
        f"Evidence: [core_encoder_batch_audit_20260915_v1.json](evidence/{OUT.name}). Receipt SHA-256: `{sha(OUT)}`. Command: `.\\.venv\\Scripts\\python.exe -X utf8 -B -u -m scratch.analysis.astra_core_encoder_batch_audit`.",
    ]
    with TABLE.open("x", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(doc) + "\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "jobs": len(jobs),
                "verified_hashes": len(report["verified_hashes"]),
                "evidence": name(OUT),
                "evidence_sha256": sha(OUT),
                "table": name(TABLE),
                "german": {
                    k: {
                        x: m[x]
                        for x in (
                            "old_mean",
                            "new_mean",
                            "delta_mean",
                            "paired_item_bootstrap_95",
                        )
                    }
                    for k, m in metrics.items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

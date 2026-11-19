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
BASE = CORE / "encoders_v5"
DERIVED = CORE / "encoder_results_v5"
OUT = ROOT / "Thesis_Log/astra_result/evidence/core_encoder_results_20260915_v5.json"
TABLE = ROOT / "Thesis_Log/astra_result/astra_core_encoder_results_20260915_v5.md"
BATCH = BASE / "batch_receipts/1789481983165297000.json"
HASHES = {}
TOKEN_PARTITIONS = {}
SCOPES = {
    "en_dev_eval_main": "hate",
    "en_dev_eval_targets": "target_group",
    "en_dev_eval_sq3_types": "hate_type",
    "en_dev_eval_severity": "severity",
    "en_dev_eval_types": "hate_type",
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def rows(p):
    with Path(p).open(encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


def name(p):
    p = Path(p).resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def sha(p):
    p = Path(p).resolve()
    key = name(p)
    if key not in HASHES:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for data in iter(lambda: f.read(1024 * 1024), b""):
                h.update(data)
        HASHES[key] = h.hexdigest()
    return HASHES[key]


def fingerprint(v):
    return hashlib.sha256(
        json.dumps(
            v, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def extra_token_checks(summary, entry, meta, frames, verification):
    from training.original import run as runner

    guard = runner.token_guard
    a = ROOT / entry["attempt"]
    tok = guard.load(ROOT / verification["checkpoint"])
    required = guard.check(tok)
    require(verification["tokenizer_guard"] == required, "Saved guard report differs")
    texts = [x["text"] for x in rows(CORE / "dev_inputs.jsonl") if x["lang"] == "en"]
    ids = guard.encode(tok, texts)["input_ids"]
    summary["dev_token_ids_sha256"] = fingerprint(ids)
    summary["tokenizer_guard"] = required
    if entry["mode"] == "full":
        t = read(a / "training.json")
        job = next(
            j
            for j in read(BASE / "execution.json")["train_jobs"]
            if runner.identity(j) == entry["job"]
        )
        require(t["job"] == job, "Training recipe changed from manifest")
        require(t["tokenizer_guard"] == required, "Training tokenizer guard differs")
        cfg = read(ROOT / verification["checkpoint"] / "config.json")
        require(
            [cfg["id2label"][str(i)] for i in range(len(meta["labels"]))]
            == meta["labels"],
            "Model label ordering differs",
        )
        for split in ("train", "val"):
            key = f"{job['arm']}/{job['head']}/{split}"
            if key not in TOKEN_PARTITIONS:
                enc = guard.encode(tok, frames[split].text.astype(str).tolist())
                TOKEN_PARTITIONS[key] = {
                    "rows": len(frames[split]),
                    "ids_sha256": fingerprint(enc["input_ids"]),
                    "at_length_128": sum(len(x) == 128 for x in enc["input_ids"]),
                }
            checked = TOKEN_PARTITIONS[key]
            require(
                t["token_stats"][split]
                == {"rows": checked["rows"], "at_max_length": checked["at_length_128"]},
                "Training token lengths differ",
            )
        summary["epoch_validation"] = [
            {k: x[k] for k in ("epoch", "step", "eval_macro_f1", "eval_loss")}
            for x in t["trainer_state"]["log_history"]
            if "eval_macro_f1" in x
        ]


def training_audit(entries, execution):
    import numpy as np
    from training.original import run as runner
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
            sha(a / "source_snapshot.py") == sha(ROOT / "training/original/run.py"),
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
            "status": "validated_v5",
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
        extra_token_checks(summary, entry, meta, frames, v)
        report.append(summary)
        print(f'Artifacts/selection/outputs verified: {entry["job"]}', flush=True)
    return report, raw


def assemble(raw):
    from src.hsrag.schema import Result

    outputs = {}
    for arm in ("full", "kb"):
        for seed in range(3):
            current = []
            heads = {
                h: {x["key"]: x for x in raw[f"{arm}/{h}/seed{seed}"]}
                for h in ("hate", "target_group", "hate_type", "severity")
            }
            for source in rows(CORE / "dev_inputs.jsonl"):
                if source["lang"] != "en":
                    continue
                rr = {h: values[source["key"]] for h, values in heads.items()}
                fields = {"hate": bool(rr["hate"]["decision"]), "legal": []}
                for h in ("target_group", "hate_type"):
                    fields[h] = [
                        label
                        for label, on in zip(rr[h]["labels"], rr[h]["decision"])
                        if on
                    ]
                fields["severity"] = rr["severity"]["labels"][
                    rr["severity"]["decision"]
                ]
                result = Result(
                    reasoning=f"encoder {arm} seed{seed}; aligned v5", **fields
                )
                actual = result.model_dump()
                require(
                    actual["hate"]
                    or (
                        actual["target_group"] == []
                        and actual["hate_type"] == []
                        and actual["severity"] is None
                    ),
                    "Gate assembly failed",
                )
                current.append(
                    {
                        "item_id": source["item_id"],
                        "key": source["key"],
                        "lang": "en",
                        "text_sha256": hashlib.sha256(
                            source["text"].encode()
                        ).hexdigest(),
                        "arm": arm,
                        "seed": seed,
                        "scopes": source["scopes"],
                        "raw_fields": fields,
                        "result": actual,
                        "uncertain": False,
                        "gate_normalised": result.gate_normalised,
                        "checkpoint_provenance": {
                            h: {
                                k: x[k]
                                for k in (
                                    "job",
                                    "checkpoint",
                                    "checkpoint_sha256",
                                    "execution_sha256",
                                )
                            }
                            for h, x in rr.items()
                        },
                    }
                )
            require(len(current) == 970, "Wrong assembled roster")
            outputs[f"en_{arm}_seed{seed}_assembled.jsonl"] = current
    return outputs


def interval(truth, old, new):
    "Paired item percentile intervals, conditional on the three fitted seeds."
    import numpy as np

    truth, old, new = [np.asarray(x, dtype=float) for x in (truth, old, new)]
    require(old.shape == new.shape == (3, *truth.shape), "Bootstrap array mismatch")
    n = len(truth)

    def scores(weights, pred):

        tp = np.einsum("bn,snl->bsl", weights, pred * truth, optimize=True)
        den = np.einsum("bn,snl->bsl", weights, pred + truth, optimize=True)
        return np.divide(2 * tp, den, out=np.zeros_like(tp), where=den != 0).mean(
            axis=2
        )

    original_old = scores(np.ones((1, n)), old)[0]
    original_new = scores(np.ones((1, n)), new)[0]
    rng = np.random.default_rng(20260915)
    draws = []
    for start in range(0, 10000, 250):
        ix = rng.integers(0, n, (min(250, 10000 - start), n))
        w = np.asarray([np.bincount(x, minlength=n) for x in ix])
        draws.extend((scores(w, new) - scores(w, old)).mean(axis=1).tolist())
    return {
        "old_check": original_old.tolist(),
        "new_check": original_new.tolist(),
        "paired_item_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "paired_item_bootstrap_bonferroni_8": np.quantile(
            draws, [0.05 / (2 * 8), 1 - 0.05 / (2 * 8)]
        ).tolist(),
    }


def score_english(outputs):
    import numpy as np
    import pandas as pd
    from src.hsrag.metrics import score_gate, score_multilabel, score_severity, clean

    goldcols = {
        "target_group": "target_groups",
        "hate_type": "hate_types",
        "severity": "severity",
        "hate": "gate",
    }
    metrics = {}
    dev = rows(CORE / "dev_inputs.jsonl")
    history = {}
    for arm in ("full", "kb"):
        for seed in range(3):
            path = ROOT / f"experiments/results/encoder_{arm}_seed{seed}_live.jsonl"
            sha(path)
            history[arm, seed] = rows(path)
        for subset, head in SCOPES.items():
            path = ROOT / f"data/processed/{subset}.parquet"
            sha(path)
            df = pd.read_parquet(path)
            gold = {str(x.id): x for x in df.itertuples(index=False)}
            roster = [x for x in dev if any(s["subset"] == subset for s in x["scopes"])]
            require(
                len(gold) == len(df) == len(roster)
                and set(gold) == {x["item_id"] for x in roster},
                "Gold roster mismatch",
            )
            if head == "hate":
                scorer = lambda r: asdict(score_gate(r, gold))
            elif head == "severity":
                scorer = lambda r: asdict(score_severity(r, gold))
            else:
                scorer = lambda r: asdict(
                    score_multilabel(r, gold, goldcols[head], head)
                )
            per_seed = []
            old_arrays = []
            new_arrays = []
            for seed in range(3):
                cc = {
                    x["item_id"]: x
                    for x in outputs[f"en_{arm}_seed{seed}_assembled.jsonl"]
                }
                current = [cc[x["item_id"]] for x in roster]
                oo = [
                    x
                    for x in history[arm, seed]
                    if x.get("encoder_meta", {}).get("subset") == subset
                ]
                require(
                    len(oo) == len(gold) and {x["item_id"] for x in oo} == set(gold),
                    "Historical roster mismatch",
                )
                oo = {x["item_id"]: x for x in oo}
                for x in roster:
                    h = hashlib.sha256(
                        " ".join(x["text"].split()).encode()
                    ).hexdigest()[:16]
                    require(
                        oo[x["item_id"]]["text_hash"] == h, "Historical text mismatch"
                    )
                old = [oo[x["item_id"]] for x in roster]
                oldscore, newscore = scorer(old), scorer(current)
                require(
                    oldscore["n_unscored"] == newscore["n_unscored"] == 0,
                    "Missing predictions",
                )
                nkey = "n_scored" if head == "hate" else "n_items"
                require(
                    oldscore[nkey] == newscore[nkey] == len(roster),
                    "Unexpected scoring mask",
                )
                detail = {
                    "seed": seed,
                    "old": oldscore,
                    "new": newscore,
                    "delta": newscore["macro_f1"] - oldscore["macro_f1"],
                }
                if head != "hate":
                    detail["new_head_without_gate"] = scorer(
                        [{**x, "result": x["raw_fields"]} for x in current]
                    )
                    detail["old_gate"] = asdict(score_gate(old, gold))
                    detail["new_gate"] = asdict(score_gate(current, gold))
                per_seed.append(detail)
                labels = (
                    ["not_hate", "hate"]
                    if head == "hate"
                    else newscore["labels_averaged"]
                )
                if head != "hate":
                    require(labels == oldscore["labels_averaged"], "Macro mask differs")

                def vector(value):
                    if head == "hate":
                        return [int(not value), int(bool(value))]
                    if head == "severity":
                        return [int(value == label) for label in labels]
                    return [int(label in (clean(value) or [])) for label in labels]

                truth = np.asarray(
                    [
                        vector(getattr(gold[x["item_id"]], goldcols[head]))
                        for x in roster
                    ]
                )
                old_arrays.append([vector(x["result"][head]) for x in old])
                new_arrays.append([vector(x["result"][head]) for x in current])
            ci = interval(truth, old_arrays, new_arrays)
            require(
                np.allclose(
                    ci.pop("old_check"),
                    [x["old"]["macro_f1"] for x in per_seed],
                    rtol=0,
                    atol=1e-12,
                ),
                "Independent old F1 differs",
            )
            require(
                np.allclose(
                    ci.pop("new_check"),
                    [x["new"]["macro_f1"] for x in per_seed],
                    rtol=0,
                    atol=1e-12,
                ),
                "Independent new F1 differs",
            )
            primary = subset != "en_dev_eval_types"
            if not primary:
                ci.pop("paired_item_bootstrap_bonferroni_8")
            metrics[f"{arm}/{subset}"] = {
                "arm": arm,
                "subset": subset,
                "head": head,
                "n": len(roster),
                "primary": primary,
                "labels_averaged": labels,
                "per_seed": per_seed,
                "old_mean": mean(x["old"]["macro_f1"] for x in per_seed),
                "old_sample_sd": stdev(x["old"]["macro_f1"] for x in per_seed),
                "new_mean": mean(x["new"]["macro_f1"] for x in per_seed),
                "new_sample_sd": stdev(x["new"]["macro_f1"] for x in per_seed),
                "delta_mean": mean(x["delta"] for x in per_seed),
                **ci,
            }
            print(
                f"Scored {arm}/{subset}: {metrics[f'{arm}/{subset}']['old_mean']:.6f} -> {metrics[f'{arm}/{subset}']['new_mean']:.6f}",
                flush=True,
            )
    return metrics


def markdown(report):
    doc = [
        "# Corrected core encoder results — 2026-09-15 (v5)",
        "",
        "**Status: completed and audited. English v5 is valid for the aligned DEV comparison; six German v4 jobs remain valid and retained. No new long run is needed for this encoder checkpoint.**",
        "",
        "This is a comparison of cleaned-data encoder baselines with the original encoders. It is not a new small/full RAG KB experiment, nor an isolated causal estimate of removing overlap. Original results and invalid English v4 artifacts remain preserved.",
        "",
        "## Verification",
        "",
        f"- Verified {report['verified_file_count']} file hashes, 15 full English jobs, nine original-head prediction-only jobs and five smoke receipts. Six German training receipts and their artifacts were rechecked.",
        "- Checked every saved English internal-validation target/logit/decision and all 23,280 DEV head rows; assembled six English system files containing 5,820 predictions. Six existing German system files contain another 975 predictions.",
        "- Independently compared tokenizer semantics and token IDs with the serialized historical reference on all new-training partitions, all 970 English DEV inputs and all 24 final tokenizer loads. The completed preflight also covers all reused-head partitions.",
        "- Rechecked selected validation-best checkpoints, patience-two stopping, compact fixed-20-epoch selection, every AMP loss-scale step, finite logits and exact FP32 save/reload agreement. No model, GPU training, API client or TEST scoring was used by this audit.",
        f"- AMP recorded {sum(x.get('amp_skipped_steps',0) for x in report['jobs'])} skipped updates in the 15 full jobs; longest consecutive sequence {max(x.get('max_consecutive_skips',0) for x in report['jobs'])}. These are handled mixed-precision overflows, not missing jobs; preserve them as recipe limitations.",
        "",
        "## Main English DEV results",
        "",
        "Means and sample SD use seeds 0/1/2. Fine-head scores include the existing hate-gate rule: a negative gate clears target/type/severity output. All compared predictions are valid and use identical DEV identities and scoring masks.",
        "",
        "| Encoder | Head / rows | Original mean ± SD | Corrected mean ± SD | Change | Paired 95% interval |",
        "|---|---|---:|---:|---:|---|",
    ]
    for m in report["english_dev"].values():
        if not m["primary"]:
            continue
        ci = m["paired_item_bootstrap_95"]
        doc.append(
            f"| {m['arm']} | {m['head']} / {m['n']} | {m['old_mean']:.6f} ± {m['old_sample_sd']:.6f} | {m['new_mean']:.6f} ± {m['new_sample_sd']:.6f} | {m['delta_mean']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |"
        )
    doc += [
        "",
        report["uncertainty_method"],
        "",
        "The 467-item type roster is a core type endpoint. Its historical filename contains “sq3”; using it here does not start Priority 3. The smaller 103-item type roster is secondary and reported separately below.",
        "",
        "## Every seed and gate diagnostics",
    ]
    for m in report["english_dev"].values():
        doc += [
            "",
            f"### {m['arm']} / {m['subset']} ({'primary' if m['primary'] else 'secondary'})",
            "",
            "| Seed | Original F1 | Corrected F1 | Change |",
            "|---:|---:|---:|---:|",
        ]
        for s in m["per_seed"]:
            doc.append(
                f"| {s['seed']} | {s['old']['macro_f1']:.6f} | {s['new']['macro_f1']:.6f} | {s['delta']:+.6f} |"
            )
        if m["primary"]:
            ci = m["paired_item_bootstrap_bonferroni_8"]
            doc += [
                "",
                f"Conservative interval across eight primary English contrasts: [{ci[0]:+.6f}, {ci[1]:+.6f}].",
            ]
        if m["head"] == "hate":
            doc += [
                "",
                "| Seed | Original FP / FN | Corrected FP / FN |",
                "|---:|---:|---:|",
            ]
            for s in m["per_seed"]:
                doc.append(
                    f"| {s['seed']} | {s['old']['fp']} / {s['old']['fn']} | {s['new']['fp']} / {s['new']['fn']} |"
                )
        else:
            doc += [
                "",
                "| Seed | Original gate misses | Corrected gate misses | Corrected head F1 before gate | Corrected assembled F1 |",
                "|---:|---:|---:|---:|---:|",
            ]
            for s in m["per_seed"]:
                doc.append(
                    f"| {s['seed']} | {s['old_gate']['fn']} | {s['new_gate']['fn']} | {s['new_head_without_gate']['macro_f1']:.6f} | {s['new']['macro_f1']:.6f} |"
                )
            doc += [
                "",
                "Fine rosters contain hateful gold items only. Gate macro-F1 is undefined on these single-class rosters; misses/recall are the appropriate diagnostics. Head-before-gate values are diagnostic and are not replacement system scores.",
                "",
                "| Label | Gold support | Original mean F1 | Corrected mean F1 | In macro |",
                "|---|---:|---:|---:|---|",
            ]
            for label in sorted(
                set().union(
                    *(
                        set(s["old"]["per_label"]) | set(s["new"]["per_label"])
                        for s in m["per_seed"]
                    )
                )
            ):
                vals = [
                    s["new"]["per_label"].get(label) or s["old"]["per_label"][label]
                    for s in m["per_seed"]
                ]
                oo = mean(
                    s["old"]["per_label"].get(label, {}).get("f1", 0)
                    for s in m["per_seed"]
                )
                nn = mean(
                    s["new"]["per_label"].get(label, {}).get("f1", 0)
                    for s in m["per_seed"]
                )
                doc.append(
                    f"| {label} | {vals[0]['support']} | {oo:.6f} | {nn:.6f} | {label in m['labels_averaged']} |"
                )
    doc += [
        "",
        "## Training diagnostics",
        "",
        "Internal validation differs from DEV and from the original partitions; historical internal scores are reference metadata, not paired evidence. Training checkpoint selection uses AMP validation; final stored predictions use FP32. Compact fine heads reuse unchanged original weights.",
        "",
        "| Job | Mode | Original internal F1 | Current internal F1 | Epochs | Selected step | AMP skips |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for j in report["jobs"]:
        doc.append(
            f"| {j['job']} | {j['mode']} | {j['old_internal_val_f1_historical']:.6f} | {j['internal_val_macro_f1_fp32']:.6f} | {j.get('epochs','reuse')} | {j.get('selected_step','—')} | {j.get('amp_skipped_steps','—')} |"
        )
    doc += [
        "",
        "## Retained German results",
        "",
        "| Head | Original mean ± SD | Cleaned mean ± SD | Change | Paired 95% interval |",
        "|---|---:|---:|---:|---|",
    ]
    for m in report["german_dev"].values():
        ci = m["paired_item_bootstrap_95"]
        doc.append(
            f"| {m['head']} | {m['old_mean']:.6f} ± {m['old_sample_sd']:.6f} | {m['new_mean']:.6f} ± {m['new_sample_sd']:.6f} | {m['delta_mean']:+.6f} | [{ci[0]:+.6f}, {ci[1]:+.6f}] |"
        )
    doc += [
        "",
        "German evidence is retained from the completed v4 audit with its unadjusted intervals; neither endpoint establishes a clear gain/loss. The earlier mmBERT/GBERT comparison remains a separate historical experiment.",
        "",
        "## Interpretation and next stage",
        "",
        report["interpretation_limit"],
        "",
        "Use these aligned encoder artifacts for the planned core comparison, keeping original small-KB studies as historical comparisons. Full-trained and compact-trained encoders differ in labelled training amount and recipe; their gap is not the causal effect of enlarging a retrieval pool. Do not select a single best seed.",
        "",
        "Pause here. The next bounded stage is preparation of combined KB indexes/RAG prompts and isolated concurrency-safe API logging, with an execution manifest and manageable batches before any large API run. Priority 3, adaptation and LaTeX remain parked.",
        "",
        f"Evidence: [audit JSON](evidence/{OUT.name}). Reproduction command (refuses existing outputs): `.\\.venv\\Scripts\\python.exe -X utf8 -B -u -m scratch.analysis.astra_core_encoder_results_v5`.",
    ]
    return "\n".join(doc) + "\n"


def main():
    require(
        not OUT.exists() and not TABLE.exists() and not DERIVED.exists(),
        "Preserve existing audit outputs",
    )
    os.chdir(ROOT)
    from training.original import run as runner

    execution = runner.validate()
    runner.require_preflight()
    plan = read(CORE / "plan.json")
    batch = read(BATCH)
    preflight = read(BASE / "preflight.json")
    for path in (
        Path(__file__),
        CORE / "plan.json",
        CORE / "encoder_plan.json",
        CORE / "dev_inputs.jsonl",
        BASE / "execution.json",
        BASE / "preflight.json",
        BASE / "tokenizer_preflight.json",
        BATCH,
    ):
        sha(path)
    for mapping in (
        plan["input_hashes"],
        plan["generated_hashes"],
        execution["input_hashes"],
        preflight["input_hashes"],
    ):
        for path, digest in mapping.items():
            require(sha(ROOT / path) == digest, f"Frozen file differs: {path}")
    for cache in plan["base_caches"].values():
        for file, digest in cache["files"].items():
            require(sha(Path(cache["snapshot"]) / file) == digest, "Base cache differs")
    require(
        batch["execution_sha256"] == sha(BASE / "execution.json")
        and batch["preflight_sha256"] == sha(BASE / "preflight.json"),
        "Batch preparation mismatch",
    )
    entries = batch["jobs"]
    require(
        not batch["smoke_only"]
        and Counter(x["mode"] for x in entries) == {"smoke": 5, "full": 15, "reuse": 9},
        "Wrong batch counts",
    )
    require(len({(x["mode"], x["job"]) for x in entries}) == 29, "Duplicate jobs")
    for mode, key in [("full", "train_jobs"), ("reuse", "reuse_jobs")]:
        require(
            {x["job"] for x in entries if x["mode"] == mode}
            == {runner.identity(j) for j in execution[key]},
            "Job roster differs",
        )
    require(
        not list((BASE / "attempts").glob("*/failure.json")),
        "Unexpected failed attempt",
    )
    require(
        all(
            (x / "complete.json").exists()
            for x in (BASE / "attempts").iterdir()
            if x.is_dir()
        ),
        "Incomplete attempt",
    )
    require(
        batch["retained_german"] == execution["inherited_full"],
        "Retained German roster differs",
    )
    for entry in batch["retained_german"]:
        a = ROOT / entry["attempt"]
        require(
            sha(a / "complete.json") == entry["receipt_sha256"],
            "German receipt differs",
        )
        for path, digest in read(a / "complete.json")["files"].items():
            require(sha(ROOT / path) == digest, "German artifact differs")
    print(
        "Frozen preparation, caches, readiness and complete batch roster verified.",
        flush=True,
    )
    jobs, raw = training_audit(entries, execution)
    require(
        len({j["dev_token_ids_sha256"] for j in jobs}) == 1,
        "Final tokenizer DEV IDs differ",
    )
    outputs = assemble(raw)
    metrics = score_english(outputs)
    old_audit = (
        ROOT
        / "Thesis_Log/astra_result/evidence/core_encoder_batch_audit_20260915_v1.json"
    )
    require(sha(old_audit) == runner.AUDIT_SHA, "German audit changed")
    inherited = read(old_audit)
    german_files = {}
    for path in sorted((CORE / "encoder_audit_v1").glob("de_*_assembled.jsonl")):
        require(
            sha(path) == inherited["verified_hashes"][name(path)],
            "German assembly changed",
        )
        german_files[name(path)] = {"sha256": sha(path), "rows": len(rows(path))}
    require(
        len(german_files) == 6 and sum(x["rows"] for x in german_files.values()) == 975,
        "German assembly incomplete",
    )
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_english_v5_german_v4_retained",
        "batch_receipt": name(BATCH),
        "batch_completed_utc": batch["completed_utc"],
        "execution_sha256": sha(BASE / "execution.json"),
        "jobs": jobs,
        "token_partitions_rechecked": TOKEN_PARTITIONS,
        "english_dev": metrics,
        "german_dev": inherited["german_dev"],
        "retained_german_assembled": german_files,
        "uncertainty_method": "10,000 paired item resamples (seed 20260915) shared across original/current and all three fitted seeds. Fixed supported label sets, zero F1 for zero denominator. Pointwise 95% percentile intervals; an additional 99.375% interval uses Bonferroni across the eight primary English arm/head contrasts. These are conditional on the fitted seeds, exclude retraining and source-cluster uncertainty, and do not undo prior DEV selection. Secondary type and retained German intervals are unadjusted.",
        "interpretation_limit": "These are post-selection DEV comparisons. Data exclusions, refitting, altered internal-validation membership and historical trainer/precision differences prevent isolating a causal cleanup effect. Small evaluation rosters and three seeds limit precision. No TEST, HateCheck, RAG/API, Priority 3 or LaTeX work was run.",
        "next": "Pause after audit; prepare combined indexing, prompts and isolated API logging in a later bounded stage.",
    }

    DERIVED.mkdir(parents=True, exist_ok=False)
    output_manifest = {}
    for filename, rr in outputs.items():
        path = DERIVED / filename
        with path.open("x", encoding="utf-8", newline="\n") as f:
            for x in rr:
                f.write(json.dumps(x, ensure_ascii=False, allow_nan=False) + "\n")
        output_manifest[name(path)] = {"sha256": sha(path), "rows": len(rr)}
    report["english_assembled"] = output_manifest
    report["verified_file_count"] = len(HASHES)
    report["verified_hashes"] = dict(sorted(HASHES.items()))
    doc = markdown(report)
    with OUT.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    with TABLE.open("x", encoding="utf-8", newline="\n") as f:
        f.write(doc)
    print(
        json.dumps(
            {
                "status": report["status"],
                "verified_files": len(HASHES),
                "evidence_sha256": sha(OUT),
                "evidence": name(OUT),
                "table": name(TABLE),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

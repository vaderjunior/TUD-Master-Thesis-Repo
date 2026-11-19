import argparse
import hashlib
import importlib.metadata
import io
import os

os.environ.update(
    OMP_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    TOKENIZERS_PARALLELISM="false",
)
from pathlib import Path
import time
import uuid
import numpy as np
from analysis.common import study as c
from training.frozen_features import compact as bge
from training.frozen_features import fit as linear
from training.roberta import tokenizer as tokenizer
from analysis.common.api import save_new, item_lock

BASE = c.ROOT / "experiments/astra/compact_roberta_probe_20261008_v1"
BASE_PLAN = c.CORE / "encoders_v5/execution.json"
KINDS = ("mean", "cls")
PROBES = [
    "This is a simple sentence.",
    " leading  spaces\nnext\tline",
    "Äpfel, Grüße und Flüchtlinge.",
    "hello🙂 world",
    "",
    "A longer sentence with several ordinary words about a book.",
    "word " * 180,
    "https://example.org/a?q=1",
]


def environment():
    return bge.environment() | {
        "neural_packages": {
            n: importlib.metadata.version(n)
            for n in ("torch", "transformers", "tokenizers", "safetensors")
        }
    }


def pool_numpy(hidden, mask):
    c.require(
        hidden.ndim == 3
        and mask.shape == hidden.shape[:2]
        and (mask.sum(axis=1) > 0).all(),
        "Invalid pooling mask",
    )
    mean = (hidden * mask[:, :, None]).sum(axis=1) / mask.sum(axis=1)[:, None]
    cls = hidden[:, 0]
    out = {}
    for name, x in [("mean", mean), ("cls", cls)]:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        c.require((norms > 1e-12).all(), "Zero pooled representation")
        out[name] = x / norms
    return out


def prepare():
    bge.validate()
    cache = c.read(BASE_PLAN)["base_caches"]["roberta-base"]
    for name, digest in cache["files"].items():
        c.require(
            c.sha(Path(cache["snapshot"]) / name) == digest,
            "Cached pretrained file changed",
        )
    template = c.read(bge.BASE / "train.json")
    records = {r["id"]: r for r in c.rows(c.CORE / "kb/small/records.jsonl")}
    inputs = {}
    mapping = {}

    def add(key, text, scope, source_id):
        if key in inputs:
            c.require(
                inputs[key]["text"] == text,
                "Same evaluation identity has different text",
            )
        else:
            inputs[key] = dict(
                key=key,
                text=text,
                exact_text_sha256=hashlib.sha256(text.encode("utf8")).hexdigest(),
            )
        mapping.setdefault(scope, []).append(dict(key=key, source_id=source_id))

    for rid in template["gate_ids"]:
        add("train/" + rid, records[rid]["text"], "train", rid)
    for row in c.rows(c.CORE / "dev_inputs.jsonl"):
        if row["lang"] == "en":
            add(row["key"], row["text"], "core", row["item_id"])
    for row in c.rows(c.PREP / "broad_roster.jsonl"):
        if row["lang"] == "en":
            add(row["key"], row["text"], "broad", row["key"])
    save_new(BASE / "feature_inputs.json", list(inputs.values()))
    save_new(BASE / "scope_mapping.json", mapping)
    files = [
        Path(__file__),
        Path(linear.__file__),
        Path(bge.__file__),
        Path(tokenizer.__file__),
        tokenizer.REFERENCE,
        c.ROOT / "tests/training/frozen_features/test_roberta.py",
        c.ROOT / "scripts/run_compact_roberta.ps1",
        c.ROOT / "training/frozen_features/compare.py",
        c.ROOT / "analysis/common/replay.py",
        c.ROOT / "analysis/common/result_reader.py",
        c.ROOT / "analysis/common/scoring.py",
        c.ROOT / "config/taxonomy.yaml",
        BASE_PLAN,
        bge.BASE / "plan.json",
        bge.BASE / "train.json",
        c.CORE / "kb/small/records.jsonl",
        c.CORE / "dev_inputs.jsonl",
        c.PREP / "broad_roster.jsonl",
        BASE / "feature_inputs.json",
        BASE / "scope_mapping.json",
        c.ROOT / "experiments/astra/compact_broad_20261008_v1/plan.json",
    ]
    files += [
        c.ROOT / f"data/processed/{name}.parquet"
        for name in c.read(bge.BASE / "plan.json")["endpoints"]
    ]
    save_new(
        BASE / "plan.json",
        {
            "created_utc": c.utc(),
            "status": "frozen_before_representation_extraction",
            "inputs": {c.rel(p): c.sha(p) for p in files},
            "base_cache": cache,
            "environment": environment(),
            "input_count": len(inputs),
            "scope_counts": {s: len(v) for s, v in mapping.items()},
            "features": "Pretrained roberta-base revision pinned by the original base-cache receipt. RobertaModel without pooling layer; no randomly initialized encoder weights accepted. FP32 SDPA, eval mode, max128, historical guarded tokenizer. TF32 disabled. Mean of all nonpadding final hidden states including special tokens, L2 normalized; fixed CLS/L2 sensitivity from same forward pass.",
            "model_comparison": "Same 147 permitted English labels, original shared TRAIN folds, balanced logistic learner, C/cutoff grids, gating and refit procedure as compact BGE. This compares representation packages; dimensions, pretraining and pooling differ. It does not isolate pretraining objective or repair the original fine-tuned RoBERTa.",
            "pooling_policy": "Mean is primary, CLS supporting; no best-pooling choice from evaluation outcomes.",
            "runtime": "CPU-only small preflight. Explicit run requires CUDA; 256-input durable NPZ chunks, microbatch32. Compare saved neutral CPU probes to CUDA within absolute1e-4 before extraction; norm tolerance2e-4. Replayed/completed chunks stay authoritative.",
            "fitting": "Three original fold seeds for each representation. All selection uses TRAIN only. Core/wider gold read by separate scoring step, not fitting.",
            "statistics": "Core and broad frozen-representation comparisons descriptive initially; publish all six fits, same populations and supported labels. No new superiority p-values or selected main system.",
            "limitations": "Original BGE and prior diagnostics already exposed DEV and later BGE absolute results. Wider TEST/HateCheck also have prior exposure. No untouched-data claim.",
        },
    )
    print(
        "Frozen representation plan:",
        len(inputs),
        "unique input identities; no model loaded.",
        flush=True,
    )


def validate():
    p = c.read(BASE / "plan.json")
    c.verify(p["inputs"])
    c.require(p["environment"] == environment(), "Representation environment changed")
    for name, digest in p["base_cache"]["files"].items():
        c.require(
            c.sha(Path(p["base_cache"]["snapshot"]) / name) == digest,
            "Pretrained cache changed",
        )
    return p


def load(device, plan):
    import pyarrow.dataset, pyarrow.parquet, pandas
    import torch
    from transformers import RobertaModel

    torch.set_num_threads(1)
    torch.manual_seed(20261008)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    tok = tokenizer.load(plan["base_cache"]["snapshot"])
    model, info = RobertaModel.from_pretrained(
        plan["base_cache"]["snapshot"],
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32,
        attn_implementation="sdpa",
        add_pooling_layer=False,
        output_loading_info=True,
    )
    c.require(
        not info.get("missing_keys")
        and (not info.get("mismatched_keys"))
        and (not info.get("error_msgs")),
        "Missing/mismatched pretrained encoder weights",
    )
    c.require(
        all(
            (
                k.startswith("lm_head.")
                or k.endswith(("position_ids", "token_type_ids"))
                for k in info.get("unexpected_keys", [])
            )
        ),
        "Unexpected checkpoint contents",
    )
    model.to(device)
    model.eval()
    c.require(
        all((p.dtype == torch.float32 for p in model.parameters())), "Non-FP32 encoder"
    )
    return (tok, model, info)


def infer(tok, model, texts, device, check_numpy=False):
    import torch

    with torch.inference_mode():
        batch = tokenizer.encode(tok, texts, padding=True, return_tensors="pt")
        batch = {
            k: v.to(device)
            for k, v in batch.items()
            if k in ("input_ids", "attention_mask")
        }
        hidden = model(**batch).last_hidden_state.float()
        mask = batch["attention_mask"].float()
        mean = (hidden * mask[:, :, None]).sum(1) / mask.sum(1)[:, None]
        out = {
            "mean": torch.nn.functional.normalize(mean, dim=1).cpu().numpy(),
            "cls": torch.nn.functional.normalize(hidden[:, 0], dim=1).cpu().numpy(),
        }
        if check_numpy:
            independent = pool_numpy(hidden.cpu().numpy(), mask.cpu().numpy())
            for name in KINDS:
                c.require(
                    np.allclose(out[name], independent[name], rtol=0, atol=2e-06),
                    "Pooling implementation mismatch",
                )
        for value in out.values():
            c.require(
                value.shape == (len(texts), 768)
                and np.isfinite(value).all()
                and np.allclose(np.linalg.norm(value, axis=1), 1, atol=0.0002),
                "Invalid representation",
            )
        return out


def preflight():
    plan = validate()
    tok, model, info = load("cpu", plan)
    start = time.perf_counter()
    full = infer(tok, model, PROBES, "cpu", True)
    split = {
        k: np.concatenate(
            [
                infer(tok, model, PROBES[i : i + 2], "cpu", True)[k]
                for i in range(0, len(PROBES), 2)
            ]
        )
        for k in KINDS
    }
    deltas = {k: float(np.max(np.abs(full[k] - split[k]))) for k in KINDS}
    c.require(max(deltas.values()) < 0.0001, "Batch/padding stability failure")
    outputs = {}
    for k in KINDS:
        path = BASE / f"preflight_{k}.npy"
        linear.save_array(path, full[k])
        outputs[c.rel(path)] = c.sha(path)
    save_new(
        BASE / "preflight.json",
        {
            "status": "cpu_preflight_pass",
            "plan_sha256": c.sha(BASE / "plan.json"),
            "outputs": outputs,
            "tokenizer_guard": tokenizer.check(tok),
            "pooling_numpy_check": True,
            "padding_max_abs_differences": deltas,
            "probe_count": len(PROBES),
            "seconds_including_repeat_probes": time.perf_counter() - start,
            "loading_info": info,
            "CUDA_used": False,
        },
    )
    print(
        "CPU representation preflight passed; full extraction remains an explicit CUDA job.",
        flush=True,
    )


def blob(path, payload):
    path = Path(path).resolve()
    c.require(path.is_relative_to(BASE.resolve()), "Outside representation folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        c.require(path.read_bytes() == payload, "Existing binary differs")
        return
    temp = path.with_name(path.name + ".partial_" + uuid.uuid4().hex)
    with temp.open("xb") as f:
        f.write(payload)
        f.flush()
        os.fsync(f.fileno())
    try:
        try:
            os.link(temp, path)
        except FileExistsError:
            c.require(path.read_bytes() == payload, "Concurrent binary differs")
    finally:
        temp.unlink()


def check_chunk(path, rows, plan_sha):
    with np.load(path, allow_pickle=False) as data:
        c.require(
            str(data["plan_sha"]) == plan_sha
            and str(data["rows_sha"]) == c.signature(rows),
            "Chunk identity changed",
        )
        values = {k: data[k].copy() for k in KINDS}
    for value in values.values():
        c.require(
            value.shape == (len(rows), 768)
            and np.isfinite(value).all()
            and np.allclose(np.linalg.norm(value, axis=1), 1, atol=0.0002),
            "Saved feature chunk invalid",
        )
    return values


def run():
    plan = validate()
    plan_sha = c.sha(BASE / "plan.json")
    pre = c.read(BASE / "preflight.json")
    c.require(
        pre["status"] == "cpu_preflight_pass" and pre["plan_sha256"] == plan_sha,
        "CPU preflight required",
    )
    c.verify(pre["outputs"])
    done = BASE / "features_complete.json"
    if done.exists():
        previous = c.read(done)
        c.require(
            previous["plan_sha256"] == plan_sha
            and previous["status"] == "frozen_representations_complete",
            "Completed feature identity changed",
        )
        c.verify(previous["outputs"])
        print("Verified complete feature extraction; skipped.", flush=True)
        return
    import pyarrow.dataset, pyarrow.parquet, pandas
    import torch

    c.require(
        torch.cuda.is_available(),
        "Enable CUDA for the explicit feature-extraction stage",
    )
    with item_lock(BASE / "inference.lock"):
        tok, model, info = load("cuda", plan)
        probe = infer(tok, model, PROBES, "cuda", True)
        drift = {
            k: float(
                np.max(
                    np.abs(
                        probe[k]
                        - np.load(BASE / f"preflight_{k}.npy", allow_pickle=False)
                    )
                )
            )
            for k in KINDS
        }
        c.require(
            max(drift.values()) < 0.0001,
            "CPU/CUDA representation mismatch; preserve before review",
        )
        rows = c.read(BASE / "feature_inputs.json")
        parts = {k: [] for k in KINDS}
        outputs = {}
        reused = 0
        for start in range(0, len(rows), 256):
            group = rows[start : start + 256]
            path = BASE / "chunks" / f"chunk_{start:06d}.npz"
            if path.exists():
                values = check_chunk(path, group, plan_sha)
                reused += len(group)
            else:
                batches = [
                    infer(tok, model, [r["text"] for r in group[i : i + 32]], "cuda")
                    for i in range(0, len(group), 32)
                ]
                values = {k: np.concatenate([v[k] for v in batches]) for k in KINDS}
                buffer = io.BytesIO()
                np.savez(
                    buffer,
                    **values,
                    plan_sha=np.array(plan_sha),
                    rows_sha=np.array(c.signature(group)),
                )
                blob(path, buffer.getvalue())
                values = check_chunk(path, group, plan_sha)
            for k in KINDS:
                parts[k].append(values[k])
            outputs[c.rel(path)] = c.sha(path)
            print(
                "Frozen RoBERTa features",
                min(start + 256, len(rows)),
                "/",
                len(rows),
                "reused",
                reused,
                flush=True,
            )
        for k in KINDS:
            path = BASE / f"features_{k}.npy"
            linear.save_array(path, np.concatenate(parts[k]))
            outputs[c.rel(path)] = c.sha(path)
        save_new(
            done,
            {
                "status": "frozen_representations_complete",
                "plan_sha256": plan_sha,
                "outputs": outputs,
                "rows": len(rows),
                "CPU_CUDA_probe_max_abs": drift,
                "device": torch.cuda.get_device_name(0),
                "torch_cuda": torch.version.cuda,
                "new_neural_training_steps": 0,
                "Qwen_calls": 0,
            },
        )


def fit_score():
    from analysis.common import replay as s
    from analysis.common import result_reader as broad

    plan = validate()
    receipt = c.read(BASE / "features_complete.json")
    c.verify(receipt["outputs"])
    rows = c.read(BASE / "feature_inputs.json")
    position = {r["key"]: i for i, r in enumerate(rows)}
    mapping = c.read(BASE / "scope_mapping.json")
    data = c.read(bge.BASE / "train.json")
    core_ep = c.read(bge.BASE / "plan.json")["endpoints"]
    broad_ep = c.read(c.ROOT / "experiments/astra/compact_broad_20261008_v1/plan.json")[
        "endpoints"
    ]
    groups, _ = broad.scientific_rosters("broad")
    results = {}
    outputs = {}
    for kind in KINDS:
        matrix = np.load(
            BASE / f"features_{kind}.npy", allow_pickle=False, mmap_mode="r"
        )
        x = np.asarray(
            matrix[[position[r["key"]] for r in mapping["train"]]], dtype=np.float64
        )
        c.require(
            [r["source_id"] for r in mapping["train"]] == data["gate_ids"],
            "TRAIN order differs",
        )
        for seed in bge.SEEDS:
            fit_path = BASE / f"fits/{kind}_seed{seed}.json"
            fit_receipt = BASE / f"fits/{kind}_seed{seed}_complete.json"
            if fit_path.exists():
                c.require(
                    fit_receipt.exists(),
                    "Interrupted fit publication; preserve for inspection",
                )
                c.verify(c.read(fit_receipt)["outputs"])
                fit = c.read(fit_path)
                c.require(
                    fit["plan_sha256"] == c.sha(BASE / "plan.json"),
                    "Saved feature fit differs",
                )
                heads = fit["heads"]
            else:
                cache, counts = linear.make_oof(data, x, seed)
                heads = linear.fit_recipe(data, x, seed, cache)
                checks = linear.check_selection(data, seed, heads, cache)
                save_new(
                    fit_path,
                    {
                        "plan_sha256": c.sha(BASE / "plan.json"),
                        "heads": heads,
                        "seed": seed,
                        "pooling": kind,
                        "candidate_checks": checks,
                    },
                )
                save_new(fit_receipt, {"outputs": {c.rel(fit_path): c.sha(fit_path)}})
            outputs[c.rel(fit_path)] = c.sha(fit_path)
            cells = {}
            for scope in ("core", "broad"):
                xx = np.asarray(
                    matrix[[position[r["key"]] for r in mapping[scope]]],
                    dtype=np.float64,
                )
                predicted = linear.decode(heads, linear.probabilities(heads, xx))
                records = [
                    dict(key=r["source_id"], result=v, seed=seed, pooling=kind)
                    for r, v in zip(mapping[scope], predicted)
                ]
                path = BASE / f"predictions/{kind}_{scope}_seed{seed}.jsonl"
                linear.save_rows(path, records)
                outputs[c.rel(path)] = c.sha(path)
                pred = {r["key"]: r["result"] for r in records}
                for name, ep in (core_ep if scope == "core" else broad_ep).items():
                    rr = (
                        s.gold_rows(name)
                        if scope == "core"
                        else groups[name.split("/")[0]]
                    )
                    keys = [str(r["id"]) for r in rr]
                    cells[scope + "/" + name] = s.summarize(
                        s.contributions(rr, {k: pred[k] for k in keys}, ep)
                    )
            results[f"{kind}/seed{seed}"] = cells
            print("Fitted/scored frozen representation", kind, seed, flush=True)
    save_new(
        BASE / "results.json",
        {
            "status": "six_representation_fits_and_scores_complete",
            "plan_sha256": c.sha(BASE / "plan.json"),
            "outputs": outputs,
            "results": results,
            "interpretation": "Mean primary; CLS supporting. Every fit and endpoint retained; no outcome-selected pooling or main-model change. Representation packages differ beyond pretraining objective.",
            "Qwen_calls": 0,
            "new_neural_training_steps": 0,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "preflight", "run", "fit-score"])
    args = p.parse_args()
    with c.offline():
        globals()[args.mode.replace("-", "_")]()

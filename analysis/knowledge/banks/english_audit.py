"""Banks english audit."""

import argparse
from collections import Counter
from contextlib import contextmanager, redirect_stdout, redirect_stderr
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
from analysis.knowledge.banks import build as u
from analysis.knowledge.banks import german_audit as numerical

ROOT = u.ROOT
RUN_ID = "english_kb_audit_20260914_v1"
BASE = ROOT / "experiments/astra" / RUN_ID
PLAN = BASE / "plan.json"
PREFLIGHT = ROOT / "experiments/astra/full_kb_english_preflight_20260914_v1"
require = u.require
js = numerical.js
jsonl = numerical.jsonl


def rel(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def owned(path):
    path = Path(path).resolve()
    require(
        path.is_relative_to(BASE.resolve()) and path != BASE.resolve(),
        "Write outside English audit refused",
    )
    return path


def write_once(path, data):
    path = owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, f"Existing audit artifact differs: {path}")
        return
    tmp = owned(path.with_name(path.name + f".partial_{time.time_ns()}"))
    with tmp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    require(not path.exists(), "Concurrent output appeared")
    tmp.rename(path)


@contextmanager
def lock():
    import msvcrt

    path = owned(BASE / "launcher.lock")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if path.stat().st_size == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError(
                "Another English audit is running. Use one terminal."
            ) from None
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def heldout_paths():
    return sorted(
        (
            p
            for p in (ROOT / "data/processed").glob("en_*.parquet")
            if "train" not in p.stem
        )
    )


def prepare():
    require(not PLAN.exists(), "Plan already frozen; use validate instead")
    prior = u.read_json(PREFLIGHT / "preflight.json")
    hashes = {**prior["input_hashes"], **prior["generated_hashes"]}
    for name, h in hashes.items():
        require(u.digest(ROOT / name) == h, f"Preflight input changed: {name}")
    summary = prior["summary"]
    require(
        summary["raw_english_train_rows"] == 56190
        and summary["heldout_distinct_nonempty_texts"] == 27810,
        "Preflight roster changed; review scope",
    )
    require(
        set(map(rel, heldout_paths())) == set(summary["heldout_file_counts"]),
        "Held-out files changed",
    )
    train = u.read_lines(PREFLIGHT / "train_texts.jsonl")
    held = u.read_lines(PREFLIGHT / "heldout_texts.jsonl")
    train_ids = {r["id"] for r in train}
    previous = ROOT / "experiments/astra/sq1_kb_size_20260911_v1"
    records = {
        pool: [
            r
            for r in u.read_lines(previous / f"kb/{pool}/records.jsonl")
            if r["kind"] == "example" and r["lang"] == "en"
        ]
        for pool in ("small", "all")
    }
    full_ids = {r["id"].removeprefix("ex-") for r in records["all"]}
    require(full_ids <= train_ids, "Old pool has non-TRAIN IDs")
    retained_ids = sorted((r["id"].removeprefix("ex-") for r in records["small"]))
    earlier_excluded = train_ids - full_ids
    cheap = (
        earlier_excluded
        | set(prior["normalized_overlap_ids"])
        | set(prior["shared_ids"])
    )
    require(
        len(earlier_excluded) == 81 and len(cheap) == 81,
        "Prior exclusion union changed",
    )
    known = [
        pair
        for case in prior["known_hatecheck_cases"]
        for pair in case["recorded_matches"]
    ]
    preparation = {
        "preflight_sha256": u.digest(PREFLIGHT / "preflight.json"),
        "summary": summary,
        "previous_full_excluded_ids": sorted(earlier_excluded),
        "cheap_excluded_ids": sorted(cheap),
        "previous_small_ids": retained_ids,
        "historical_pair_warnings": known,
        "policy": "Carry all 81 previous exclusions forward; add freshly computed cosine >=0.95 matches. Save [0.94,0.95) nearest pairs for review, without automatically excluding them. Recompute explicitly listed historical pairs even if a different held-out item is the global nearest. Final text-family/annotation policy and pool sizes await results review.",
    }
    write_once(BASE / "train_texts.jsonl", jsonl(train))
    write_once(BASE / "heldout_texts.jsonl", jsonl(held))
    write_once(BASE / "preparation.json", js(preparation))
    for name in (
        "analysis/knowledge/banks/english_audit.py",
        "tests/knowledge/banks/test_english_audit.py",
        "scripts/validate_english_bank.ps1",
        "analysis/knowledge/banks/german_audit.py",
    ):
        hashes[name] = u.digest(ROOT / name)
    hashes[rel(PREFLIGHT / "preflight.json")] = u.digest(PREFLIGHT / "preflight.json")
    plan = {
        "run_id": RUN_ID,
        "created_utc": u.now(),
        "input_hashes": hashes,
        "generated_hashes": {
            rel(BASE / n): u.digest(BASE / n)
            for n in ("train_texts.jsonl", "heldout_texts.jsonl", "preparation.json")
        },
        "environment": u.environment(),
        "embedding_cache": u.cache_info(),
        "heldout_files": list(summary["heldout_file_counts"]),
        "train_rows": len(train),
        "heldout_rows": len(held),
        "model": "BAAI/bge-m3",
        "precision": "float32",
        "batch_size": 4,
        "cache_chunk": 128,
        "train_similarity_chunk": 256,
        "eval_similarity_chunk": 512,
        "threshold": 0.95,
        "very_near": 0.98,
        "review_lower": 0.94,
        "seed": 42,
        "policy": preparation["policy"],
        "scope": "English TRAIN versus every full English dev/test and declared English evaluation text, including HateCheck. Only identity/text/language from held-out data; no held-out gold or performance scoring. Fresh local embeddings; no prior Chroma/vector reuse. No original data/KB/exclusion edits. Stop after audit. German pool remains a separate completed policy.",
    }
    require(
        all((u.digest(ROOT / n) == h for n, h in hashes.items())),
        "Input changed during preparation",
    )
    write_once(PLAN, js(plan))
    print(
        json.dumps(
            {
                "status": "prepared_not_run",
                "plan": rel(PLAN),
                "sha256": u.digest(PLAN),
                "train_rows": len(train),
                "heldout_rows": len(held),
                "embedding_chunks": sum(
                    ((n + 127) // 128 for n in (len(train), len(held)))
                ),
                "prior_excluded": len(cheap),
            },
            indent=2,
        ),
        flush=True,
    )


def validate(cache=True):
    plan = u.read_json(PLAN)
    require(plan["run_id"] == RUN_ID, "Wrong audit plan")
    for group in ("input_hashes", "generated_hashes"):
        for name, h in plan[group].items():
            require(u.digest(ROOT / name) == h, f"Frozen input changed: {name}")
    require(
        set(map(rel, heldout_paths())) == set(plan["heldout_files"]),
        "Held-out roster changed",
    )
    require(
        u.environment() == plan["environment"], "Python/package environment changed"
    )
    if cache:
        require(
            u.cache_info() == plan["embedding_cache"], "Pinned embedding cache changed"
        )
    return plan


def chunk_identity(plan, texts):
    return {
        "plan_sha256": u.digest(PLAN),
        "model": plan["model"],
        "revision": plan["embedding_cache"]["revision"],
        "precision": plan["precision"],
        "texts_sha256": hashlib.sha256(js(texts)).hexdigest(),
        "rows": len(texts),
    }


def save_chunk(path, identity, vectors):
    import numpy as np

    path = owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(not path.exists(), "Refusing to overwrite an embedding chunk")
    numerical.validate_vectors(vectors, identity["rows"])
    tmp = owned(path.with_name(path.name + f".partial_{time.time_ns()}"))
    meta = {**identity, "vectors_sha256": hashlib.sha256(vectors.tobytes()).hexdigest()}
    with tmp.open("xb") as f:
        np.savez(
            f, vectors=vectors, metadata=np.asarray(json.dumps(meta, sort_keys=True))
        )
        f.flush()
        os.fsync(f.fileno())
    require(not path.exists(), "Concurrent embedding chunk appeared")
    tmp.rename(path)


def encode_rows(plan, texts, tag, get_model):
    import numpy as np

    vectors, paths = ([], [])
    for start in range(0, len(texts), plan["cache_chunk"]):
        sub = texts[start : start + plan["cache_chunk"]]
        ident = chunk_identity(plan, sub)
        key = hashlib.sha256(js(ident)).hexdigest()
        path = BASE / f"embedding_chunks/{tag}_{start:06d}_{key[:16]}.npz"
        cached = path.exists()
        if not cached:
            vv = np.asarray(
                get_model().encode(
                    sub,
                    batch_size=plan["batch_size"],
                    normalize_embeddings=True,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                ),
                dtype=np.float32,
            )
            save_chunk(path, ident, vv)
        vectors.append(numerical.load_chunk(path, ident))
        paths.append(path)
        print(
            f"EMBED {tag}: {start + len(sub)}/{len(texts)} ({('verified cache' if cached else 'encoded')})",
            flush=True,
        )
    return (np.concatenate(vectors), paths)


def known_pair_scores(train, held, a, b, warnings):
    ti = {r["id"]: i for i, r in enumerate(train)}
    hi = {ref["id"]: i for i, r in enumerate(held) for ref in r["references"]}
    out = []
    for pair in warnings:
        rid, eid = (pair["train_id"], pair["eval_id"])
        require(rid in ti and eid in hi, "Historical pair missing from fresh roster")
        out.append(
            {
                "train_id": rid,
                "eval_id": eid,
                "subset": pair["subset"],
                "reported_historical_cosine": pair["cosine"],
                "fresh_cosine": float(a[ti[rid]] @ b[hi[eid]]),
                "train_text": train[ti[rid]]["text"],
                "heldout_text": held[hi[eid]]["text"],
                "in_previous_full": pair["in_previous_full"],
                "in_previous_small": pair["in_previous_small"],
            }
        )
    return out


def finalize(plan, train, held, mx, ix, paths, resources, known):
    survey = u.read_json(BASE / "preparation.json")
    near = {r["id"] for r, s in zip(train, mx) if float(s) >= plan["threshold"]}
    excluded = set(survey["cheap_excluded_ids"]) | near
    excluded_text = {u.canonical_text(r["text"]) for r in train if r["id"] in excluded}
    excluded |= {r["id"] for r in train if u.canonical_text(r["text"]) in excluded_text}
    keep = [r for r in train if r["id"] not in excluded]
    rows, pairs = ([], [])
    for r, s, index in zip(train, mx, ix):
        h = held[int(index)]
        item = {
            "train_id": r["id"],
            "source": r["source"],
            "max_cosine": float(s),
            "heldout_text_index": int(index),
            "nearest_heldout_references": h["references"],
            "threshold_excluded": r["id"] in near,
            "proposed_excluded": r["id"] in excluded,
        }
        rows.append(item)
        if float(s) >= plan["review_lower"] or r["id"] in excluded:
            pairs.append({**item, "train_text": r["text"], "heldout_text": h["text"]})
    report = {
        "status": "audit_complete_final_pool_pending_review",
        "plan_sha256": u.digest(PLAN),
        "train_rows": len(train),
        "heldout_texts": len(held),
        "near_train_rows": len(near),
        "very_near_train_rows": sum((float(s) >= plan["very_near"] for s in mx)),
        "review_band_rows": sum(
            (plan["review_lower"] <= float(s) < plan["threshold"] for s in mx)
        ),
        "boundary_within_0005": sum(
            (abs(float(s) - plan["threshold"]) <= 0.0005 for s in mx)
        ),
        "previous_excluded": len(survey["cheap_excluded_ids"]),
        "proposed_excluded_ids": sorted(excluded),
        "new_excluded_ids": sorted(excluded - set(survey["cheap_excluded_ids"])),
        "provisional_remaining_train_rows": len(keep),
        "provisional_remaining_texts": len({u.canonical_text(r["text"]) for r in keep}),
        "provisional_remaining_previous_small": len(
            set(survey["previous_small_ids"]) - excluded
        ),
        "remaining_sources": dict(Counter((r["source"] for r in keep))),
        "resources": resources,
        "limits": "Threshold exclusions are proposals. Review near/boundary/text-family cases and direct historical pairs. No proof of exhaustive semantic independence below screened range. English-only scan; no gold scoring or KB build.",
    }
    artifacts = {
        "nearest_train.jsonl": jsonl(rows),
        "review_pairs.jsonl": jsonl(pairs),
        "known_pair_checks.json": js(known),
        "report.json": js(report),
    }
    for name, data in artifacts.items():
        write_once(BASE / name, data)
    write_once(
        BASE / "complete.json",
        js(
            {
                "plan_sha256": u.digest(PLAN),
                "completed_utc": u.now(),
                "outputs": {rel(BASE / n): u.digest(BASE / n) for n in artifacts},
                "embedding_hashes": {rel(p): u.digest(p) for p in paths},
            }
        ),
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("proposed_excluded_ids", "new_excluded_ids")
            },
            indent=2,
        ),
        flush=True,
    )
    print(
        "ENGLISH OVERLAP AUDIT COMPLETE. Tell Astra to inspect the saved report. No classifier run has started.",
        flush=True,
    )


def check_complete():
    if not (BASE / "complete.json").exists():
        return False
    receipt = u.read_json(BASE / "complete.json")
    require(receipt["plan_sha256"] == u.digest(PLAN), "Completion plan differs")
    for group in ("outputs", "embedding_hashes"):
        for name, h in receipt[group].items():
            require(u.digest(ROOT / name) == h, f"Completed audit file changed: {name}")
    return True


def execute(plan):
    if check_complete():
        print("ENGLISH AUDIT ALREADY COMPLETE. Saved outputs verified; nothing rerun.")
        return
    require(
        not (BASE / "report.json").exists(),
        "Unreceipted report exists; preserve it for inspection",
    )
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_CACHE"] = plan["embedding_cache"]["hub"]
    os.environ["ANONYMIZED_TELEMETRY"] = "False"
    import torch

    require(
        torch.cuda.is_available(),
        "CUDA is unavailable. Enable CUDA and rerun the same command.",
    )
    require(shutil.disk_usage(BASE).free >= 2 * 2**30, "Need at least 2 GiB free space")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(plan["seed"])
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model_box = []
    print(
        f"CUDA: {torch.cuda.get_device_name(0)}; pinned BGE-M3 FP32, batch {plan['batch_size']}.",
        flush=True,
    )

    def model():
        if not model_box:
            from sentence_transformers import SentenceTransformer

            m = SentenceTransformer(
                plan["model"],
                revision=plan["embedding_cache"]["revision"],
                local_files_only=True,
                device="cuda",
            )
            m.float()
            m.eval()
            model_box.append(m)
            print(
                f"Loaded pinned model; max sequence length {m.max_seq_length}.",
                flush=True,
            )
        return model_box[0]

    train = u.read_lines(BASE / "train_texts.jsonl")
    held = u.read_lines(BASE / "heldout_texts.jsonl")
    a, ap = encode_rows(plan, [r["text"] for r in train], "train", model)
    b, bp = encode_rows(plan, [r["text"] for r in held], "heldout", model)
    encoded = time.perf_counter()
    peak = torch.cuda.max_memory_allocated() / 2**30
    model_box.clear()
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    mx, ix = numerical.nearest(
        a, b, plan["train_similarity_chunk"], plan["eval_similarity_chunk"]
    )
    known = known_pair_scores(
        train,
        held,
        a,
        b,
        u.read_json(BASE / "preparation.json")["historical_pair_warnings"],
    )
    compared = time.perf_counter()
    validate()
    finalize(
        plan,
        train,
        held,
        mx,
        ix,
        ap + bp,
        {
            "attempt_seconds_before_report": time.perf_counter() - started,
            "embedding_phase_seconds_including_cache_load": encoded - started,
            "comparison_phase_seconds": compared - encoded,
            "peak_allocated_gib": peak,
            "gpu": torch.cuda.get_device_name(0),
            "note": "Attempt timings include verified cache reuse if resumed; console log identifies encoded/reused chunks.",
        },
        known,
    )


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "validate", "status", "run"))
    args = parser.parse_args()
    if args.mode == "prepare":
        with lock():
            prepare()
    elif args.mode == "validate":
        validate()
        print(
            f"VALID: frozen English audit; completed={check_complete()}. No model calls."
        )
    elif args.mode == "status":
        print(
            json.dumps(
                {
                    "prepared": PLAN.exists(),
                    "complete": (BASE / "complete.json").exists(),
                    "cached_chunks": len(
                        list((BASE / "embedding_chunks").glob("*.npz"))
                    ),
                },
                indent=2,
            )
        )
    else:
        with lock():
            log = owned(BASE / f"logs/audit_{time.time_ns()}.log")
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("x", encoding="utf-8") as f, redirect_stdout(
                numerical.Tee(sys.stdout, f)
            ), redirect_stderr(numerical.Tee(sys.stderr, f)):
                try:
                    execute(validate())
                except Exception:
                    import traceback

                    traceback.print_exc()
                    print(
                        "STOPPED. Preserve log and cached chunks; no original files changed.",
                        flush=True,
                    )
                    raise


if __name__ == "__main__":
    main()

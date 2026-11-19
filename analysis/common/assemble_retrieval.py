import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np
from analysis.common import study as c
from analysis.common import retrieval as old
from analysis.common import prompt as prompt

BASE = c.BASE / "retrieval"
PLAN = BASE / "plan.json"


def snapshot(path):
    return {c.rel(p): c.sha(p) for p in sorted(path.rglob("*")) if p.is_file()}


def sources():
    return [
        Path(__file__),
        Path(c.__file__),
        Path(old.__file__),
        Path(prompt.__file__),
        c.ROOT / "src/hsrag/retrieve.py",
        c.ROOT / "src/hsrag/kb.py",
        c.ROOT / "src/hsrag/prompt.py",
        c.ROOT / "config/taxonomy.yaml",
        c.ROOT / "prompts/classify_v1.txt",
    ]


def prepare():
    if PLAN.exists():
        validate()
        print("Existing retrieval plan verified.")
        return
    bindings = c.scientific_inputs()
    receipt = c.read(c.CACHE)
    c.verify(receipt["source_sha256"])
    core = c.read(c.CORE / "plan.json")
    audit = c.read(
        c.ROOT
        / "Thesis_Log/astra_result/evidence/core_retrieval_check_20260915_v2.json"
    )
    verified = (
        audit["verified_hashes"]
        if "verified_hashes" in audit
        else audit.get("input_hashes", {})
    )
    index_hashes = {}
    for pool in ("small", "full"):
        for name, h in snapshot(c.CORE / f"retrieval_v1/{pool}/chroma").items():
            c.require(
                verified.get(name) == h,
                "Original index differs from audited final snapshot: " + name,
            )
            index_hashes[name] = h
    for p in [
        c.CACHE,
        c.CORE / "retrieval_v1/review_complete_v2.json",
        c.ROOT
        / "Thesis_Log/astra_result/evidence/core_retrieval_check_20260915_v2.json",
        *sources(),
    ]:
        bindings[c.rel(p)] = c.sha(p)
    cache = c.read(c.ROOT / "experiments/astra/german_kb_audit_20260913_v1/plan.json")[
        "embedding_cache"
    ]
    c.save(
        PLAN,
        {
            "created_utc": c.utc(),
            "status": "offline_preparation_no_Qwen",
            "inputs": bindings,
            "original_index_hashes": index_hashes,
            "retrieval": core["retrieval"],
            "embedding": {
                "model": old.MODEL,
                "revision": old.REVISION,
                "precision": "float32",
                "cache": cache,
            },
            "empty_input": {
                "key": c.EMPTY_KEY,
                "policy": "Retain the original empty text, encode it once with the pinned BGE on CPU in FP32, retrieve normally, and retain it in all full-roster denominators. Report an explicit empty-input exclusion sensitivity for supporting German TEST hate; never select handling by outcomes.",
            },
            "index_policy": "Copy reviewed physical indexes byte-for-byte into this owned experiment. Never open or mutate originals with Chroma. First-pass prompts are authoritative; two cold passes preserve/report mismatches. Nonzero differences require review, not automatic acceptance.",
            "query_policy": "Use exact text plus source-ID-bound audited cached FP32 vectors; no prompt/outcome/gold-based selection.",
            "counts": {
                "inputs": 21627,
                "rag_prompts": 43254,
                "zero_shot_prompts": 21627,
            },
            "environment": core["environment"],
        },
    )
    print("Retrieval preparation contract frozen; no model/API calls.")


def validate():
    p = c.read(PLAN)
    c.verify(p["inputs"])
    c.verify(p["original_index_hashes"])
    from analysis.knowledge.banks.build import environment

    c.require(environment() == p["environment"], "Retrieval environment changed")
    return p


def assemble():
    p = validate()
    target = BASE / "vectors"
    done = target / "complete.json"
    if done.exists():
        c.verify(c.read(done)["outputs"])
        print("Query vectors already assembled.")
        return
    roster = c.rows(c.PREP / "broad_roster.jsonl")
    receipt = c.read(c.CACHE)
    c.verify(receipt["source_sha256"])
    origins = {x["key"]: x for x in receipt["matched_vector_origins"]}
    grouped = defaultdict(list)
    matrix = np.zeros((len(roster), 1024), dtype=np.float32)
    used = []
    for i, row in enumerate(roster):
        if row["key"] in origins:
            origin = origins[row["key"]]
            c.require(
                hashlib.sha256(row["text"].encode()).hexdigest()
                == origin["exact_text_sha256"],
                "Cache text mismatch",
            )
            grouped[origin["file"]].append((i, origin["row"]))
            used.append(origin)
        else:
            c.require(
                row["key"] == c.EMPTY_KEY and row["text"] == "" and row["empty_input"],
                "Unexpected missing cached vector",
            )
    for name, pairs in grouped.items():
        with np.load(c.ROOT / name, allow_pickle=False) as z:
            values = z["vectors"]
            meta = json.loads(str(z["metadata"].item()))
        c.require(
            hashlib.sha256(values.tobytes()).hexdigest() == meta["vectors_sha256"],
            "Cached payload mismatch",
        )
        for i, j in pairs:
            matrix[i] = values[j]
    cache = p["embedding"]["cache"]
    source = Path(cache["snapshot"])
    for name, h in cache["files"].items():
        c.require(c.sha(source / name) == h, "Pinned BGE file changed: " + name)
    empty_file = target / "empty_embedding.json"
    if empty_file.exists():
        empty = c.read(empty_file)
        c.require(
            empty["plan_sha256"] == c.sha(PLAN),
            "Empty embedding belongs to different plan",
        )
        vector = np.asarray(empty["vector"], dtype=np.float32)
    else:
        print("Encoding the single empty input on CPU; no CUDA or Qwen.", flush=True)
        import torch
        from sentence_transformers import SentenceTransformer

        torch.set_num_threads(min(4, os.cpu_count() or 1))
        model = SentenceTransformer(str(source), device="cpu", local_files_only=True)
        model.float()
        with torch.inference_mode():
            vector = model.encode(
                [""],
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )[0]
        c.require(
            vector.dtype == np.float32 and np.isfinite(vector).all(),
            "Invalid empty embedding",
        )
        c.save(
            empty_file,
            {
                "plan_sha256": c.sha(PLAN),
                "key": c.EMPTY_KEY,
                "text": "",
                "device": "cpu",
                "vector": vector.tolist(),
                "new_embedding_texts": 1,
            },
        )
        del model
    for i, row in enumerate(roster):
        if row["key"] == c.EMPTY_KEY:
            matrix[i] = vector
    old.validate_vectors(matrix, len(roster))
    file = target / "query_vectors.npy"
    file.parent.mkdir(parents=True, exist_ok=True)
    if file.exists():
        c.require(
            np.array_equal(np.load(file, allow_pickle=False), matrix),
            "Partial matrix differs",
        )
    else:
        with file.open("xb") as f:
            np.save(f, matrix, allow_pickle=False)
    c.save_rows(
        target / "query_origins.jsonl",
        [
            origins.get(
                r["key"],
                {
                    "key": r["key"],
                    "file": c.rel(empty_file),
                    "row": 0,
                    "exact_text_sha256": hashlib.sha256(r["text"].encode()).hexdigest(),
                },
            )
            for r in roster
        ],
    )
    c.save(
        done,
        {
            "status": "query_vectors_assembled",
            "plan_sha256": c.sha(PLAN),
            "rows": len(roster),
            "cache_rows": len(origins),
            "new_embedding_texts": 1,
            "actual_Qwen_calls": 0,
            "outputs": {
                c.rel(x): c.sha(x)
                for x in [file, empty_file, target / "query_origins.jsonl"]
            },
        },
    )


def copy_index(pool):
    p = validate()
    source = c.CORE / f"retrieval_v1/{pool}/chroma"
    dest = BASE / pool / "chroma"
    dest.mkdir(parents=True, exist_ok=True)
    prefix = c.rel(source) + "/"
    expected = {
        name[len(prefix) :]: h
        for name, h in p["original_index_hashes"].items()
        if name.startswith(prefix)
    }
    receipt = BASE / pool / "copy.json"
    if receipt.exists():
        c.require(c.read(receipt)["source"] == expected, "Index copy origin differs")
        return
    c.require(
        {x.relative_to(dest).as_posix() for x in dest.rglob("*") if x.is_file()}
        <= set(expected),
        "Unexpected partial index file",
    )
    for name, h in expected.items():
        out = dest / name
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            shutil.copyfile(source / name, out)
        c.require(c.sha(out) == h, "Copied index not bitwise identical")
    c.save(
        receipt,
        {
            "plan_sha256": c.sha(PLAN),
            "source": expected,
            "policy": "Original files remain read-only; owned copy may undergo Chroma metadata maintenance.",
        },
    )


def worker(pool, mode, limit=0):
    p = validate()
    vd = c.read(BASE / "vectors/complete.json")
    c.verify(vd["outputs"])
    c.require(vd["plan_sha256"] == c.sha(PLAN), "Vector contract mismatch")
    copy_index(pool)
    records = c.rows(c.CORE / f"kb/{pool}/records.jsonl")
    full = c.rows(c.CORE / "kb/full/records.jsonl")
    ix = {r["id"]: i for i, r in enumerate(full)}
    matrix = np.load(
        c.CORE / "retrieval_v1/full_vectors.npy", allow_pickle=False, mmap_mode="r"
    )
    vectors = matrix[[ix[r["id"]] for r in records]]
    roster = c.rows(c.PREP / "broad_roster.jsonl")
    q = np.load(BASE / "vectors/query_vectors.npy", allow_pickle=False)
    encoder = old.CachedEncoder([], q)
    query_index = {r["key"]: i for i, r in enumerate(roster)}
    import chromadb

    client = chromadb.PersistentClient(path=str(BASE / pool / "chroma"))
    col = client.get_collection("hate_kb")
    origin = c.read(c.CORE / f"retrieval_v1/{pool}/origin.json")
    c.require(
        all((col.metadata.get(k) == v for k, v in origin.items())),
        "Copied collection metadata differs",
    )
    diff = old.verify_collection(col, records, vectors)
    retriever = old.make_retriever(
        col, c.CORE / f"kb/{pool}/records.jsonl", p["retrieval"], encoder
    )
    known = {r["id"]: r for r in records}
    path = BASE / pool / "prompts.jsonl"
    if mode == "probe":
        groups = defaultdict(list)
        for row in roster:
            groups[row["phase"]].append(row)
        selected = [
            r
            for phase in sorted(groups)
            for r in sorted(groups[phase], key=lambda x: x["key"])[: max(1, limit // 5)]
        ]
        selected += [r for r in roster if r["key"] == c.EMPTY_KEY and r not in selected]
        target = BASE / pool / f"probe_{time.time_ns()}"
    else:
        selected = roster
        target = BASE / pool / mode
    stored = c.rows(path) if path.exists() else []
    if mode == "freeze":
        if stored:
            c.require(
                len(stored) == len(roster),
                "Incomplete frozen prompt file; preserve for review",
            )
            c.verify(c.read(target / "complete.json")["outputs"])
            print(pool + " prompts already frozen.")
            return
    elif mode != "probe":
        c.require(
            [r["key"] for r in stored] == [r["key"] for r in roster],
            "Frozen prompt roster incomplete",
        )
    if (target / "complete.json").exists():
        c.verify(c.read(target / "complete.json")["outputs"])
        print(pool + "/" + mode + " already complete.")
        return
    actual = []
    comparisons = []
    counts = Counter()
    start = time.perf_counter()
    for n, row in enumerate(selected):
        encoder.lookup = {row["text"]: q[query_index[row["key"]]]}
        ctx = prompt.from_retrieval(retriever.retrieve(row["text"], row["lang"]))
        old.validate_context(ctx, known, row["lang"])
        messages = prompt.render_messages(row["text"], row["lang"], ctx, "corrected")
        out = {
            "key": row["key"],
            "item_id": str(row["id"]),
            "lang": row["lang"],
            "population": row["phase"],
            "condition": "rag_" + pool,
            "context": ctx,
            "context_sha256": c.signature(ctx),
            "messages": messages,
            "messages_sha256": c.signature(messages),
            "retrieved": prompt.hit_ids(ctx),
            "plan_sha256": c.sha(PLAN),
        }
        if mode.startswith("replay"):
            previous = stored[n]
            change = [
                k for k in previous.keys() | out.keys() if previous.get(k) != out.get(k)
            ]
            comparison = {
                "key": row["key"],
                "changed_fields": sorted(change),
                "same_messages": out["messages"] == previous["messages"],
                "same_retrieved_ids": out["retrieved"] == previous["retrieved"],
            }
            if change:
                comparison["actual_entry"] = out
            comparisons.append(comparison)
            counts["entry_differences"] += bool(change)
            counts["message_differences"] += not comparison["same_messages"]
            counts["retrieved_id_differences"] += not comparison["same_retrieved_ids"]
        else:
            actual.append(out)
        if (n + 1) % 100 == 0:
            print(
                f"{pool}/{mode}: {n + 1}/{len(selected)} ({time.perf_counter() - start:.1f}s)",
                flush=True,
            )
    seconds = time.perf_counter() - start
    dest = (
        path
        if mode == "freeze"
        else target
        / ("comparisons.jsonl" if mode.startswith("replay") else "prompts.jsonl")
    )
    c.save_rows(dest, comparisons if mode.startswith("replay") else actual)
    c.save(
        target / "complete.json",
        {
            "plan_sha256": c.sha(PLAN),
            "pool": pool,
            "mode": mode,
            "rows": len(selected),
            "seconds": seconds,
            "estimated_full_pass_minutes": seconds / len(selected) * len(roster) / 60,
            "differences": dict(counts),
            "index_vector_max_abs_difference": diff,
            "outputs": {c.rel(dest): c.sha(dest)},
            "original_index_hashes_after": {
                n: c.sha(c.ROOT / n) for n in p["original_index_hashes"]
            },
            "Qwen_calls": 0,
        },
    )
    c.verify(p["original_index_hashes"])
    print(
        json.dumps(
            c.read(target / "complete.json") | {"outputs": "saved"}, ensure_ascii=False
        ),
        flush=True,
    )


def finalize():
    validate()
    outputs = {}
    tally = {}
    for pool in ("small", "full"):
        for mode in ("freeze", "replay1", "replay2"):
            p = BASE / pool / mode / "complete.json"
            r = c.read(p)
            c.require(
                r["rows"] == 21627 and r["plan_sha256"] == c.sha(PLAN),
                "Incomplete retrieval pass",
            )
            c.verify(r["outputs"])
            outputs[c.rel(p)] = c.sha(p)
            outputs.update(r["outputs"])
            c.require(
                all((v == 0 for v in r["differences"].values())),
                "Cold replay differences require review; originals preserved",
            )
            tally[pool + "/" + mode] = {
                "rows": r["rows"],
                "seconds": r["seconds"],
                "differences": r["differences"],
            }
    empty = {"definitions": [], "guidelines": [], "example_groups": []}
    c.save_rows(
        BASE / "zero_shot.jsonl",
        [
            {
                "key": r["key"],
                "item_id": str(r["id"]),
                "lang": r["lang"],
                "population": r["phase"],
                "condition": "zero_shot",
                "messages": (
                    m := prompt.render_messages(
                        r["text"], r["lang"], empty, "corrected"
                    )
                ),
                "messages_sha256": c.signature(m),
            }
            for r in c.rows(c.PREP / "broad_roster.jsonl")
        ],
    )
    outputs[c.rel(BASE / "zero_shot.jsonl")] = c.sha(BASE / "zero_shot.jsonl")
    c.save(
        BASE / "complete.json",
        {
            "status": "retrieved_and_two_cold_passes_exact",
            "plan_sha256": c.sha(PLAN),
            "outputs": outputs,
            "passes": tally,
            "actual_Qwen_calls": 0,
        },
    )
    print(
        "Broad prompts prepared; independent prompt/token verification is still required."
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "mode",
        choices=[
            "prepare",
            "assemble",
            "probe",
            "freeze",
            "replay1",
            "replay2",
            "finalize",
            "run",
        ],
    )
    p.add_argument("--pool", choices=["small", "full"])
    p.add_argument("--limit", type=int, default=25)
    args = p.parse_args()
    os.chdir(c.ROOT)
    with c.offline():
        if args.mode == "prepare":
            prepare()
        elif args.mode == "assemble":
            assemble()
        elif args.mode == "finalize":
            finalize()
        elif args.mode == "run":
            prepare()
            assemble()
            for pool in ("small", "full"):
                for mode in ("freeze", "replay1", "replay2"):
                    subprocess.run(
                        [
                            sys.executable,
                            "-X",
                            "utf8",
                            "-B",
                            "-u",
                            "-m",
                            __spec__.name,
                            mode,
                            "--pool",
                            pool,
                        ],
                        cwd=c.ROOT,
                        check=True,
                    )
            finalize()
        else:
            c.require(args.pool, "Pool required")
            worker(args.pool, args.mode, args.limit)


if __name__ == "__main__":
    main()

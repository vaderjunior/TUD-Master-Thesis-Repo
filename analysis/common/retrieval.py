"""Retrieve study records from cached vectors."""

from __future__ import annotations
from src.hsrag.paths import PROJECT_ROOT
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    ANONYMIZED_TELEMETRY="False",
    TOKENIZERS_PARALLELISM="false",
)
ROOT = PROJECT_ROOT
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
BASE = CORE / "retrieval_v1"
PLAN = BASE / "plan.json"
CORE_SHA = "5c610f4fe320b5e0f185315933d47b388ab645b20b52eed738541a1579184e4e"
MODEL = "BAAI/bge-m3"
REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def require(ok, why):
    if not ok:
        raise RuntimeError(why)


def rel(p):
    return Path(p).resolve().relative_to(ROOT).as_posix()


def js(x):
    return (
        json.dumps(x, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf8")


def signature(x):
    return hashlib.sha256(js(x)).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def rows(p):
    return [
        json.loads(x)
        for x in Path(p).read_text(encoding="utf8").splitlines()
        if x.strip()
    ]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def owned(p):
    p = Path(p).resolve()
    require(
        p.is_relative_to(BASE.resolve()) and p != BASE.resolve(),
        "Write outside retrieval experiment",
    )
    return p


def put_bytes(p, b):
    p = owned(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        require(p.read_bytes() == b, f"Preserve differing artifact: {p}")
        return
    tmp = owned(p.with_name(p.name + f".partial_{time.time_ns()}"))
    with tmp.open("xb") as f:
        f.write(b)
        f.flush()
        os.fsync(f.fileno())
    tmp.rename(p)


def put(p, x):
    put_bytes(p, js(x))


def put_rows(p, rr):
    put_bytes(
        p,
        b"".join(
            (
                (
                    json.dumps(x, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    + "\n"
                ).encode("utf8")
                for x in rr
            )
        ),
    )


def verify_hashes(mapping):
    for path, digest in mapping.items():
        require(sha(ROOT / path) == digest, f"Changed artifact: {path}")


def validate_vectors(v, n):
    import numpy as np

    require(v.dtype == np.float32 and v.shape == (n, 1024), "Wrong vector dtype/shape")
    require(bool(np.isfinite(v).all()), "Nonfinite vector")
    require(
        bool(np.allclose(np.linalg.norm(v, axis=1), 1, atol=0.0002, rtol=0)),
        "Unnormalised vector",
    )


def load_chunk(path, identity):
    import numpy as np

    with np.load(path, allow_pickle=False) as z:
        meta = json.loads(str(z["metadata"].item()))
        v = z["vectors"]
    digest = meta.pop("vectors_sha256")
    require(meta == identity, "Cache identity differs")
    validate_vectors(v, identity["rows"])
    require(
        hashlib.sha256(v.tobytes()).hexdigest() == digest,
        "Cache payload checksum differs",
    )
    return v


def record_meta(r):
    return {
        **{k: r[k] for k in ("kind", "lang", "source", "dimension", "label")},
        **r.get("meta", {}),
    }


def validate_context(context, records, lang):
    groups = context["example_groups"]
    examples = [h for _, rr in groups for h in rr]
    require(
        len(examples) == len({h["id"] for h in examples}) == 5,
        "Five unique example slots required",
    )
    require(
        len(context["definitions"]) == 2 and len(context["guidelines"]) == 2,
        "Abstract budget differs",
    )
    for bucket, hh in [
        ("definition", context["definitions"]),
        ("guideline", context["guidelines"]),
        ("example", examples),
    ]:
        for h in hh:
            require(h["id"] in records, "Unknown retrieved ID")
            r = records[h["id"]]
            meta = record_meta(r)
            require(
                r["kind"] == bucket and r["text"] == h["text"],
                "Retrieved kind/text differs",
            )
            require(
                all((h["meta"].get(k) == v for k, v in meta.items())),
                "Retrieved annotation differs",
            )
            require(set(h["meta"]) <= set(meta) | {"_ranks"}, "Unexpected metadata")
            if bucket == "example":
                require(r["lang"] == lang, "Cross-language example")
    return examples


def prepare():
    import numpy as np
    from analysis.knowledge.banks import build as util

    require(not PLAN.exists(), "Preparation already frozen; use validate/run")
    require(sha(CORE / "plan.json") == CORE_SHA, "Core plan changed")
    core = read(CORE / "plan.json")
    require(util.environment() == core["environment"], "Environment changed")
    verify_hashes(core["generated_hashes"])
    verify_hashes(core["input_hashes"])
    full = rows(CORE / "kb/full/records.jsonl")
    small = rows(CORE / "kb/small/records.jsonl")
    dev = rows(CORE / "dev_inputs.jsonl")
    full_ids = {r["id"]: i for i, r in enumerate(full)}
    require(
        len(full_ids) == len(full) == 71131
        and len(small) == 340
        and (len(dev) == 1295),
        "Wrong core sizes",
    )
    require(
        all((full[full_ids[r["id"]]] == r for r in small)), "Small/full records differ"
    )
    require(len({r["key"] for r in dev}) == 1295, "Repeated DEV identity")
    frozen = {rel(CORE / "plan.json"): CORE_SHA, **core["generated_hashes"]}
    sources = {}
    query_sources = {}
    selected_chunks = {}
    cache_plans = {}
    duplicate_max = 0.0
    for lang, name in [
        ("en", "english_kb_audit_20260914_v1"),
        ("de", "german_kb_audit_20260913_v1"),
    ]:
        base = ROOT / "experiments/astra" / name
        p = read(base / "plan.json")
        done = read(base / "complete.json")
        require(
            done["plan_sha256"] == sha(base / "plan.json"),
            "Prior completion/plan mismatch",
        )
        require(
            p["model"] == MODEL
            and p["embedding_cache"]["revision"] == REVISION
            and (p["precision"] == "float32"),
            "Embedding contract changed",
        )
        require(p["environment"] == core["environment"], "Cache environment differs")
        cache_plans[lang] = p
        for f in [
            base / "plan.json",
            base / "complete.json",
            base / "train_texts.jsonl",
            base / "heldout_texts.jsonl",
        ]:
            frozen[rel(f)] = sha(f)
        for tag in ("train", "heldout"):
            path = base / f"{tag}_texts.jsonl"
            rr = rows(path)
            require(
                sha(path) == p["generated_hashes"][rel(path)],
                "Cache text roster changed",
            )
            wanted = {
                r["text"]
                for r in (full if tag == "train" else dev)
                if r["lang"] == lang
            }
            for start in range(0, len(rr), p["cache_chunk"]):
                block = rr[start : start + p["cache_chunk"]]
                picks = [i for i, r in enumerate(block) if r["text"] in wanted]
                if not picks:
                    continue
                ident = {
                    "plan_sha256": sha(base / "plan.json"),
                    "model": MODEL,
                    "revision": REVISION,
                    "precision": "float32",
                    "texts_sha256": signature([r["text"] for r in block]),
                    "rows": len(block),
                }
                f = (
                    base
                    / f"embedding_chunks/{tag}_{start:06d}_{signature(ident)[:16]}.npz"
                )
                require(
                    sha(f) == done["embedding_hashes"][rel(f)],
                    "Prior embedding file differs",
                )
                vv = load_chunk(f, ident)
                frozen[rel(f)] = sha(f)
                selected_chunks[rel(f)] = {"rows": len(block), "used_rows": len(picks)}
                dest = sources if tag == "train" else query_sources
                for i in picks:
                    r = block[i]
                    key = (lang, r["text"])
                    entry = (
                        vv[i].copy(),
                        {"file": rel(f), "row": i, "source_kind": tag},
                    )
                    if key in dest:
                        diff = float(np.max(np.abs(dest[key][0] - entry[0])))
                        duplicate_max = max(duplicate_max, diff)
                        require(
                            diff < 2e-05, "Duplicate exact-text embeddings disagree"
                        )
                    else:
                        dest[key] = entry
                    if tag == "heldout":
                        expected = [
                            x
                            for x in dev
                            if x["lang"] == lang and x["text"] == r["text"]
                        ]
                        refs = {(x["file"], x["id"]) for x in r["references"]}
                        for x in expected:
                            require(
                                all(
                                    (
                                        (
                                            f"data/processed/{s['subset']}.parquet",
                                            x["item_id"],
                                        )
                                        in refs
                                        for s in x["scopes"]
                                    )
                                ),
                                "DEV cache reference mismatch",
                            )
            print(f"Cache checked: {lang}/{tag}", flush=True)
    prior = ROOT / "experiments/astra/german_kb_comparison_20260913_v1"
    german_train = {
        r["text"]
        for r in rows(
            ROOT / "experiments/astra/german_kb_audit_20260913_v1/train_texts.jsonl"
        )
    }
    ordered = [
        r
        for pool in ("original_small", "canonical_small", "canonical_full")
        for r in rows(prior / f"kb/{pool}/records.jsonl")
    ]
    remaining = list(
        dict.fromkeys((r["text"] for r in ordered if r["text"] not in german_train))
    )
    bgpath = prior / "background_vectors.json"
    bg = read(bgpath)
    setup = read(prior / "setup_complete.json")
    require(
        bg["identity"]
        == {
            "plan_sha256": sha(prior / "plan.json"),
            "texts_sha256": signature(remaining),
        },
        "Background ordering/plan differs",
    )
    for f in [prior / "plan.json", prior / "setup_complete.json", bgpath] + [
        prior / f"kb/{p}/records.jsonl"
        for p in ("original_small", "canonical_small", "canonical_full")
    ]:
        frozen[rel(f)] = sha(f)
    require(sha(bgpath) == setup["hashes"][rel(bgpath)], "Background vectors changed")
    b = np.asarray(bg["vectors"], dtype=np.float32)
    validate_vectors(b, len(remaining))
    background = {
        text: (b[i], {"file": rel(bgpath), "row": i, "source_kind": "background"})
        for i, text in enumerate(remaining)
    }
    matrix = []
    provenance = []
    for r in full:
        v, origin = (
            sources[r["lang"], r["text"]]
            if r["kind"] == "example"
            else background[r["text"]]
        )
        matrix.append(v)
        provenance.append(
            {
                "id": r["id"],
                "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest(),
                **origin,
            }
        )
    qmatrix = []
    qprovenance = []
    for r in dev:
        v, origin = query_sources[r["lang"], r["text"]]
        qmatrix.append(v)
        qprovenance.append(
            {
                "key": r["key"],
                "text_sha256": hashlib.sha256(r["text"].encode()).hexdigest(),
                **origin,
            }
        )
    v = np.asarray(matrix, dtype=np.float32)
    q = np.asarray(qmatrix, dtype=np.float32)
    validate_vectors(v, len(full))
    validate_vectors(q, len(dev))
    BASE.mkdir(parents=True, exist_ok=True)
    for filename, arr in [("full_vectors.npy", v), ("query_vectors.npy", q)]:
        import io

        buffer = io.BytesIO()
        np.save(buffer, arr, allow_pickle=False)
        put_bytes(BASE / filename, buffer.getvalue())
    put_rows(BASE / "record_vector_sources.jsonl", provenance)
    put_rows(BASE / "query_vector_sources.jsonl", qprovenance)
    source_files = [
        Path(__file__),
        ROOT / "tests/common/test_retrieval.py",
        ROOT / "scripts/prepare_core_retrieval.ps1",
        ROOT / "analysis/common/prompt.py",
        ROOT / "src/hsrag/kb.py",
        ROOT / "src/hsrag/retrieve.py",
        ROOT / "src/hsrag/prompt.py",
        ROOT / "prompts/classify_v1.txt",
        ROOT / "config/taxonomy.yaml",
    ]
    frozen.update({rel(f): sha(f) for f in source_files})
    generated = {
        rel(BASE / n): sha(BASE / n)
        for n in (
            "full_vectors.npy",
            "query_vectors.npy",
            "record_vector_sources.jsonl",
            "query_vector_sources.jsonl",
        )
    }
    put(
        PLAN,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "core_sha256": CORE_SHA,
            "model": MODEL,
            "revision": REVISION,
            "environment": util.environment(),
            "retrieval": core["retrieval"],
            "input_hashes": frozen,
            "generated_hashes": generated,
            "counts": {
                "full": len(full),
                "small": len(small),
                "dev": len(dev),
                "rag_prompts": 2 * len(dev),
            },
            "selected_cache_chunks": selected_chunks,
            "duplicate_exact_text_vector_max_difference": duplicate_max,
            "query_policy": "Reuse exact-text FP32 normalized BGE-M3 vectors from completed text-only audits, batch size 4. No fresh query encoding. Both pools use the same query matrix. Historical single-query forward passes can differ numerically; do not claim historical prompt equivalence.",
            "scope": "Two isolated cosine Chroma indexes; original hybrid/MMR code, 5 examples + 2 definitions + 2 guidelines, frozen corrected renderer. No neural model, API calls or outcome scoring.",
            "index_policy": "Canonical record order, add batches of 128, original cosine HNSW defaults. Logical full-record/vector validation and post-process physical hashes. Separate process per pool; cold reload checks every stored prompt.",
            "next": "Run CPU-only preparation, then offline prompt/index audit. Safe API logger and execution contracts remain separate.",
        },
    )
    print(
        json.dumps(
            {
                "status": "prepared",
                "plan": rel(PLAN),
                "sha256": sha(PLAN),
                "cache_chunks": len(selected_chunks),
                "counts": read(PLAN)["counts"],
            },
            indent=2,
        ),
        flush=True,
    )


def validate():
    from analysis.knowledge.banks import build as util

    p = read(PLAN)
    require(
        p["core_sha256"] == CORE_SHA and sha(CORE / "plan.json") == CORE_SHA,
        "Core changed",
    )
    require(util.environment() == p["environment"], "Environment changed")
    verify_hashes(p["input_hashes"])
    verify_hashes(p["generated_hashes"])
    return p


class CachedEncoder:

    def __init__(self, texts, vectors):
        self.lookup = {t: vectors[i] for i, t in enumerate(texts)}

    def encode(self, texts, normalize_embeddings=True):
        import numpy as np

        require(normalize_embeddings is True, "Cached query normalization changed")
        require(
            all((t in self.lookup for t in texts)),
            "Uncached query; no model fallback allowed",
        )
        return np.stack([self.lookup[t] for t in texts])


def make_retriever(col, records_path, cfg, encoder):
    from src.hsrag.retrieve import Retriever

    require(col.metadata["embedding_model"] == MODEL, "Wrong collection model")
    return Retriever(
        BASE / "unused", records_path, MODEL, dict(cfg), encoder=encoder, collection=col
    )


def data(pool):
    import numpy as np

    full = rows(CORE / "kb/full/records.jsonl")
    records = rows(CORE / f"kb/{pool}/records.jsonl")
    indices = {r["id"]: i for i, r in enumerate(full)}
    vectors = np.load(BASE / "full_vectors.npy", allow_pickle=False, mmap_mode="r")
    selected = np.asarray(
        [vectors[indices[r["id"]]] for r in records], dtype=np.float32
    )
    dev = rows(CORE / "dev_inputs.jsonl")
    q = np.load(BASE / "query_vectors.npy", allow_pickle=False)
    validate_vectors(selected, len(records))
    validate_vectors(q, len(dev))
    return (records, selected, dev, q)


def verify_collection(col, records, vectors):
    import numpy as np
    from src.hsrag.kb import _flatten_meta

    require(
        col.count() == len(records)
        and set(col.get(include=[])["ids"]) == {r["id"] for r in records},
        "Wrong index roster",
    )
    maxdiff = 0.0
    for start in range(0, len(records), 256):
        rr = records[start : start + 256]
        got = col.get(
            ids=[r["id"] for r in rr], include=["documents", "metadatas", "embeddings"]
        )
        indexed = {
            rid: (text, meta, v)
            for rid, text, meta, v in zip(
                got["ids"], got["documents"], got["metadatas"], got["embeddings"]
            )
        }
        for i, r in enumerate(rr, start):
            text, meta, v = indexed[r["id"]]
            require(
                text == r["text"] and meta == _flatten_meta(r),
                "Index text/labels differ",
            )
            diff = float(np.max(np.abs(np.asarray(v, dtype=np.float32) - vectors[i])))
            maxdiff = max(maxdiff, diff)
            require(diff < 2e-07, "Index vector differs")
    return maxdiff


def worker(pool, check=False):
    import chromadb
    from src.hsrag.kb import _flatten_meta
    from analysis.common import prompt as prompt

    p = validate()
    records, vectors, dev, q = data(pool)
    root = BASE / pool
    identity = {
        "embedding_model": MODEL,
        "embedding_revision": REVISION,
        "kb_version": sha(CORE / f"kb/{pool}/records.jsonl")[:16],
        "plan_sha256": sha(PLAN),
        "hnsw:space": "cosine",
    }
    require(
        not (root / "chroma").exists() or (root / "origin.json").exists(),
        "Unowned index refused",
    )
    if check:
        require(read(root / "origin.json") == identity, "Wrong index origin")
    else:
        put(root / "origin.json", identity)
    client = chromadb.PersistentClient(path=str(owned(root / "chroma")))
    names = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    require(set(names) <= {"hate_kb"}, "Unexpected collection")
    require(not check or names == ["hate_kb"], "Missing completed collection")
    col = (
        client.get_collection("hate_kb")
        if names
        else client.create_collection("hate_kb", metadata=identity)
    )
    require(
        all((col.metadata.get(k) == v for k, v in identity.items())),
        "Wrong collection origin",
    )
    existing = set(col.get(include=[])["ids"])
    require(existing <= {r["id"] for r in records}, "Unexpected existing IDs")
    require(not check or len(existing) == len(records), "Incomplete check-only index")
    for start in range(0, len(records), 128):
        ix = [
            i
            for i in range(start, min(start + 128, len(records)))
            if records[i]["id"] not in existing
        ]
        if ix:
            col.add(
                ids=[records[i]["id"] for i in ix],
                documents=[records[i]["text"] for i in ix],
                metadatas=[_flatten_meta(records[i]) for i in ix],
                embeddings=vectors[ix].tolist(),
            )
        if start % 8192 == 0:
            print(f"INDEX {pool}: {col.count()}/{len(records)}", flush=True)
    maxdiff = verify_collection(col, records, vectors)
    retriever = make_retriever(
        col,
        CORE / f"kb/{pool}/records.jsonl",
        p["retrieval"],
        CachedEncoder([r["text"] for r in dev], q),
    )
    known = {r["id"]: r for r in records}
    path = root / "prompts.jsonl"
    stored = rows(path) if path.exists() else []
    require(not check or len(stored) == len(dev), "Completed prompts missing")
    if stored:
        require(
            [x["key"] for x in stored] == [x["key"] for x in dev],
            "Saved prompt roster differs",
        )
    collected = []
    stats = {}
    started = time.perf_counter()
    for i, r in enumerate(dev):
        ctx = prompt.from_retrieval(retriever.retrieve(r["text"], r["lang"]))
        examples = validate_context(ctx, known, r["lang"])
        messages = prompt.render_messages(r["text"], r["lang"], ctx, "corrected")
        entry = {
            **r,
            "pool": pool,
            "condition": f"rag_{pool}",
            "context": ctx,
            "context_sha256": signature(ctx),
            "messages": messages,
            "messages_sha256": signature(messages),
            "retrieved": prompt.hit_ids(ctx),
            "plan_sha256": sha(PLAN),
        }
        if stored:
            require(
                entry == stored[i],
                f"Cold/current retrieval differs at {pool}/{r['key']}",
            )
        collected.append(entry)
        for scope in r["scopes"]:
            count = stats.setdefault(scope["subset"], Counter())
            count["inputs"] += 1
            for h in examples:
                m = h["meta"]
                count["example_slots"] += 1
                count[f"gate_{m.get('gate')}"] += 1
                for field in ("target_groups", "hate_types", "severity", "legal"):
                    count[field + "_annotated"] += m.get(field) is not None
        if (i + 1) % 100 == 0 or i + 1 == len(dev):
            print(
                f"{('RECHECK' if check else 'RETRIEVE')} {pool}: {i + 1}/{len(dev)} ({time.perf_counter() - started:.0f}s)",
                flush=True,
            )
    put_rows(path, collected)
    result = {
        "plan_sha256": sha(PLAN),
        "pool": pool,
        "records": len(records),
        "prompts": len(collected),
        "index_vector_max_abs_difference": maxdiff,
        "summary": {k: dict(v) for k, v in stats.items()},
        "prompt_sha256": sha(path),
    }
    if check:
        prior = read(root / "complete.json")
        require(all((prior[k] == v for k, v in result.items())), "Cold summary differs")
        put(root / "cold_check.json", {**result, "all_prompts_retrieved_again": True})
    else:
        put(root / "complete.json", result)


@contextmanager
def lock():
    import msvcrt

    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / "launcher.lock").open("a+b") as f:
        if f.seek(0, 2) == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError("Another retrieval preparation is running") from None
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def run():
    with lock():
        p = validate()
        if (BASE / "complete.json").exists():
            done = read(BASE / "complete.json")
            verify_hashes(done["hashes"])
            print(
                "RETRIEVAL PREPARATION ALREADY COMPLETE. No API/classifier run.",
                flush=True,
            )
            return
        for pool in ("small", "full"):
            for mode in ("worker", "check"):
                label = f"{mode}_{pool}_{time.time_ns()}"
                out = owned(BASE / f"logs/{label}.txt")
                out.parent.mkdir(parents=True, exist_ok=True)
                with out.open("x", encoding="utf8") as log:
                    child = subprocess.Popen(
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
                        cwd=ROOT,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        encoding="utf8",
                    )
                    for line in child.stdout:
                        log.write(line)
                        log.flush()
                        print(line, end="", flush=True)
                    code = child.wait()
                require(code == 0, f"{mode}/{pool} failed; preserve log {rel(out)}")
        small = rows(BASE / "small/prompts.jsonl")
        full = rows(BASE / "full/prompts.jsonl")
        contrasts = Counter()
        for a, b in zip(small, full):
            require(a["key"] == b["key"], "Comparison roster differs")
            contrasts["inputs"] += 1
            for k in ("definitions", "guidelines"):
                contrasts[k + "_same_ordered_ids"] += [
                    r["id"] for r in a["context"][k]
                ] == [r["id"] for r in b["context"][k]]
            contrasts["messages_changed"] += a["messages"] != b["messages"]
        files = [
            f
            for pool in ("small", "full")
            for f in sorted((BASE / pool).rglob("*"))
            if f.is_file()
        ]
        hashes = {rel(f): sha(f) for f in files}
        receipt = {
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "status": "prepared_no_classifier_calls",
            "plan_sha256": sha(PLAN),
            "counts": p["counts"],
            "contrasts": dict(contrasts),
            "hashes": hashes,
            "next": "Independent preparation review; then isolated API logger and staged classifier execution manifest. No classifier/API batch is ready yet.",
        }
        put(BASE / "complete.json", receipt)
        print(
            json.dumps({k: v for k, v in receipt.items() if k != "hashes"}, indent=2),
            flush=True,
        )


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["prepare", "validate", "run", "worker", "check", "status"]
    )
    parser.add_argument("--pool", choices=["small", "full"])
    args = parser.parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "validate":
        p = validate()
        print(json.dumps({"status": "validated", "counts": p["counts"]}, indent=2))
    elif args.mode == "run":
        run()
    elif args.mode in ("worker", "check"):
        require(args.pool is not None, "Pool required")
        worker(args.pool, args.mode == "check")
    else:
        print(
            json.dumps(
                {
                    "plan": PLAN.exists(),
                    "complete": (BASE / "complete.json").exists(),
                    "pool_receipts": {
                        p: (BASE / p / "complete.json").exists()
                        for p in ("small", "full")
                    },
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

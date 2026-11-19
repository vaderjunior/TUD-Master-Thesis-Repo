"""Sampled banks retrieval."""

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import unittest
from unittest.mock import patch
from analysis.common import retrieval as r
from analysis.knowledge.strategies import prepare as trace_tools
from analysis.common import prompt as prompt

os.environ.update(
    CUDA_VISIBLE_DEVICES="",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    ANONYMIZED_TELEMETRY="False",
    TOKENIZERS_PARALLELISM="false",
)
ROOT = r.ROOT
CORE = r.CORE
STUDY = ROOT / "experiments/astra/sq1_pool_size_composition_20260926_v1"
BASE = STUDY / "retrieval_v1"
PLAN = BASE / "plan.json"
DESIGN_SHA = "1b5bdced39a6e5f81a4a19122a321b9c61ddb7c881ef4563e3c034d41ec88a4e"
MODULE = "analysis.knowledge.sampled_banks.retrieval"
KINDS = ("smoke", "smoke_replay", "prepare", "replay")
COMPARE_FIELDS = (
    "context",
    "messages",
    "retrieved",
    "raw_context",
    "trace",
    "final_hits",
)


def utc():
    return datetime.now(timezone.utc).isoformat()


def owned(path):
    p = Path(path).resolve()
    r.require(
        p.is_relative_to(BASE.resolve()) and p != BASE.resolve(),
        "Write outside isolated retrieval folder",
    )
    return p


def put_bytes(path, data):
    path = owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        r.require(
            path.read_bytes() == data, "Preserve conflicting artifact: " + r.rel(path)
        )
        return
    temporary = owned(path.with_name(path.name + ".partial_" + str(time.time_ns())))
    with temporary.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    temporary.rename(path)


def save(path, value):
    put_bytes(path, r.js(value))


def save_rows(path, values):
    put_bytes(
        path,
        b"".join(
            (
                (
                    json.dumps(v, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    + "\n"
                ).encode("utf8")
                for v in values
            )
        ),
    )


def no_network(*args, **kwargs):
    raise RuntimeError("Offline pool preparation: model/network fallback forbidden")


def tree_hashes(path):
    return {r.rel(p): r.sha(p) for p in sorted(path.rglob("*")) if p.is_file()}


def sources():
    return [
        Path(__file__),
        ROOT / "analysis/knowledge/sampled_banks/verify_retrieval.py",
        ROOT / "tests/knowledge/sampled_banks/test_retrieval.py",
        ROOT / "scripts/prepare_sampled_retrieval.ps1",
        ROOT / "analysis/common/retrieval.py",
        ROOT / "analysis/knowledge/strategies/prepare.py",
        ROOT / "analysis/knowledge/strategies/verify.py",
        ROOT / "analysis/knowledge/banks/build.py",
        ROOT / "analysis/common/prompt.py",
        ROOT / "src/hsrag/retrieve.py",
        ROOT / "src/hsrag/kb.py",
        ROOT / "src/hsrag/prompt.py",
        ROOT / "config/taxonomy.yaml",
        ROOT / "prompts/classify_v1.txt",
        ROOT / ".venv/Lib/site-packages/rank_bm25.py",
    ]


def test_sources():
    return {r.rel(p): r.sha(p) for p in sources()}


def tests():
    r.require(not PLAN.exists(), "Tests must pass before freezing a new retrieval plan")
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "tests.knowledge.sampled_banks.test_retrieval"
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    path = BASE / "development" / ("tests_" + str(time.time_ns()) + ".json")
    save(
        path,
        {
            "created_utc": utc(),
            "status": "passed" if result.wasSuccessful() else "failed",
            "tests_run": result.testsRun,
            "output": stream.getvalue(),
            "source_hashes": test_sources(),
        },
    )
    print(stream.getvalue(), flush=True)
    r.require(result.wasSuccessful(), "Tests failed; preserve attempt")


def selection(full, manifest):
    ids = list(manifest["abstract_ids"])
    for lang in ("en", "de"):
        sub = manifest["languages"][lang]
        r.require(
            len(sub["example_ids"]) == len(set(sub["example_ids"])) == sub["size"],
            "Duplicate/missing sample IDs",
        )
        ids += sub["example_ids"]
    r.require(len(ids) == len(set(ids)), "Cross-pool ID collision")
    wanted = set(ids)
    selected = [rec for rec in full if rec["id"] in wanted]
    r.require(len(selected) == len(ids), "Sample ID not in eligible full TRAIN")
    for lang in ("en", "de"):
        r.require(
            [
                rec["id"]
                for rec in selected
                if rec["kind"] == "example" and rec["lang"] == lang
            ]
            == manifest["languages"][lang]["example_ids"],
            "Sample language/order mismatch",
        )
    r.require(
        [rec["id"] for rec in selected if rec["kind"] != "example"]
        == manifest["abstract_ids"],
        "Abstract order/membership mismatch",
    )
    return selected


def verify_provenance(records, matrix, provenance, key_field, text_key):
    import hashlib

    r.validate_vectors(matrix, len(records))
    r.require(len(provenance) == len(records), "Vector provenance length")
    for row, prov in zip(records, provenance):
        r.require(
            row[key_field] == prov[key_field]
            and hashlib.sha256(row[text_key].encode("utf8")).hexdigest()
            == prov["text_sha256"],
            "Vector identity/text alignment",
        )


def freeze():
    import numpy as np
    from analysis.knowledge.banks.build import environment

    r.require(not PLAN.exists(), "Retrieval plan already frozen")
    r.require(
        r.sha(STUDY / "scientific_plan.json") == DESIGN_SHA, "Scientific design changed"
    )
    design = r.read(STUDY / "scientific_plan.json")
    verified = r.read(STUDY / "independent_check.json")
    r.require(
        verified["scientific_plan_sha256"] == DESIGN_SHA
        and verified["language_subsets"] == 18,
        "Pool design not independently checked",
    )
    r.verify_hashes(design["input_hashes"])
    candidates = sorted((BASE / "development").glob("tests_*.json"))
    r.require(candidates, "Run focused tests before freezing")
    test_path = candidates[-1]
    test = r.read(test_path)
    r.require(
        test["status"] == "passed" and test["tests_run"] >= 12,
        "Focused tests incomplete",
    )
    r.verify_hashes(test["source_hashes"])
    r.require(test["source_hashes"] == test_sources(), "Source changed after tests")
    original = r.read(r.PLAN)
    core = r.read(CORE / "plan.json")
    r.require(
        environment() == original["environment"] == core["environment"],
        "Environment differs from accepted original retrieval",
    )
    r.require(
        core["retrieval"]["strategy"] == "hybrid"
        and core["retrieval"]["k_examples"] == 5
        and (core["retrieval"]["k_examples_feedback"] == 0),
        "Unexpected baseline retrieval",
    )
    r.verify_hashes(original["generated_hashes"])
    full = r.rows(CORE / "kb/full/records.jsonl")
    dev = r.rows(CORE / "dev_inputs.jsonl")
    vectors = np.load(r.BASE / "full_vectors.npy", mmap_mode="r", allow_pickle=False)
    queries = np.load(r.BASE / "query_vectors.npy", allow_pickle=False)
    verify_provenance(
        full, vectors, r.rows(r.BASE / "record_vector_sources.jsonl"), "id", "text"
    )
    verify_provenance(
        dev, queries, r.rows(r.BASE / "query_vector_sources.jsonl"), "key", "text"
    )
    r.require(
        len(full) == 71131 and len(dev) == len({row["key"] for row in dev}) == 1295,
        "Unexpected source roster",
    )
    accepted = r.read(r.BASE / "review_complete_v2.json")
    review = r.read(ROOT / accepted["evidence"])
    r.require(
        r.sha(ROOT / accepted["evidence"])
        == accepted["evidence_sha256"]
        == trace_tools.REVIEW_SHA,
        "Retrieval acceptance changed",
    )
    original_index_hashes = {}
    for pool in ("small", "full"):
        hashes = tree_hashes(r.BASE / pool / "chroma")
        r.require(
            hashes == review["physical_index_hashes"][pool],
            "Original source index changed",
        )
        original_index_hashes.update(hashes)
    manifests = r.read(STUDY / "pool_manifests.json")
    jobs = []
    indices = {rec["id"]: i for i, rec in enumerate(full)}
    for i, manifest in enumerate(manifests):
        pool_id = f"p{i:02d}"
        root = BASE / "pools" / pool_id
        selected = selection(full, manifest)
        path = root / "records.jsonl"
        save_rows(path, selected)
        offsets = [indices[rec["id"]] for rec in selected]
        save(root / "full_vector_rows.json", offsets)
        jobs.append(
            {
                "pool_id": pool_id,
                "condition": manifest["condition"],
                "records": r.rel(path),
                "vector_rows": r.rel(root / "full_vector_rows.json"),
                "count": len(selected),
                "by_language": {
                    lang: manifest["languages"][lang]["size"] for lang in ("en", "de")
                },
            }
        )
    refs = {}
    for condition, path in design["reference_prompts"].items():
        rows = r.rows(ROOT / path)
        r.require(len(rows) == 1295, "Missing reference prompts")
        refs[condition] = {row["key"]: row for row in rows}
        r.require(
            len(refs[condition]) == 1295
            and r.sha(ROOT / path) == design["input_hashes"][path],
            "Reference identity/hash mismatch",
        )
    for row in dev:
        left = refs["curated_small"][row["key"]]
        right = refs["full_reference"][row["key"]]
        for key in ("definitions", "guidelines"):
            r.require(
                r.js(left["context"][key]) == r.js(right["context"][key]),
                "Full/small abstract metadata differ",
            )
    smoke_keys = [
        dev[0]["key"],
        next(
            (row["key"] for row in dev if row["scopes"][0]["subset"] == "de_dev_eval")
        ),
        next(
            (
                row["key"]
                for row in dev
                if row["scopes"][0]["subset"] == "de_legal_dev_eval"
            )
        ),
    ]
    files = sources() + [
        STUDY / "scientific_plan.json",
        STUDY / "pool_manifests.json",
        STUDY / "independent_check.json",
        CORE / "plan.json",
        CORE / "dev_inputs.jsonl",
        CORE / "kb/full/records.jsonl",
        CORE / "kb/small/records.jsonl",
        r.PLAN,
        r.BASE / "full_vectors.npy",
        r.BASE / "query_vectors.npy",
        r.BASE / "record_vector_sources.jsonl",
        r.BASE / "query_vector_sources.jsonl",
        r.BASE / "review_complete_v2.json",
        ROOT / accepted["evidence"],
        CORE / "api_execution_v1/scoring_plan.json",
        test_path,
    ]
    files += [ROOT / path for path in design["reference_prompts"].values()]
    files += [ROOT / job[key] for job in jobs for key in ("records", "vector_rows")]
    plan = {
        "status": "retrieval_implementation_frozen_smoke_pending",
        "created_utc": utc(),
        "scientific_plan_sha256": DESIGN_SHA,
        "environment": environment(),
        "retrieval": core["retrieval"],
        "input_hashes": {r.rel(p): r.sha(p) for p in files},
        "original_index_hashes": original_index_hashes,
        "jobs": jobs,
        "reference_prompts": design["reference_prompts"],
        "smoke_keys": smoke_keys,
        "tests": r.rel(test_path),
        "counts": {
            "inputs": 1295,
            "sampled_conditions": 9,
            "references": 2,
            "new_retrieval_prompts_per_pass": 11655,
            "final_prompts": 14245,
            "index_records_per_pass": sum((job["count"] for job in jobs)),
        },
        "index_policy": "Create cosine Chroma with original defaults, original full record order, batches of 128 and frozen FP32 vectors. No model encoding. Separate process per pool/pass. Build smoke indexes once; prepare copies a completed smoke index; replay copies completed prepare index. Original source indexes never opened by a client.",
        "context_policy": "Run unchanged full hybrid retrieval and record every raw hit/trace. Deliver its five example hits but replace both abstract buckets with the original per-query full reference, including order and metadata. Store raw and delivered contexts separately. Reference payloads remain exact.",
        "validation": "Every copied index verifies texts/annotations and vectors; export stored FP32 vectors for independent MMR arithmetic. Smoke compares traced/untraced retrieval on each query. Every completed process has post-exit physical hashes. Independent BM25 preserves numerical operation order.",
        "replay": "One complete cold replay per pool. Preserve first pass and every changed actual entry, including trace-only or rank-only changes. No retry-to-match, no automatic prompt substitution. Differences require later Astra review.",
        "resume": "Verify and skip complete pool/pass receipts. Refuse an incomplete attempt rather than erase or silently rerun it. Keep logs and ask Astra to inspect; never touches a completed prompt.",
        "matrix": "Nine new pool conditions plus byte-identical inherited small/full payload references, three fresh decoder repeats later. No classifier, API, gold score, TEST, Priority3 or LaTeX here.",
        "next": "Short smoke and cold reopening checks; independent preflight before user full preparation. Separate API runner after full retrieval review.",
    }
    save(PLAN, plan)
    print(
        json.dumps(
            {
                "status": plan["status"],
                "plan_sha256": r.sha(PLAN),
                "counts": plan["counts"],
                "smoke_keys": smoke_keys,
            },
            indent=2,
        )
    )


def validate(check_original=True):
    from analysis.knowledge.banks.build import environment

    plan = r.read(PLAN)
    r.require(
        r.sha(STUDY / "scientific_plan.json") == DESIGN_SHA, "Scientific design changed"
    )
    r.verify_hashes(plan["input_hashes"])
    r.require(
        plan["environment"] == environment(), "Pinned retrieval environment changed"
    )
    if check_original:
        observed = {}
        for pool in ("small", "full"):
            observed.update(tree_hashes(r.BASE / pool / "chroma"))
        r.require(
            observed == plan["original_index_hashes"], "Original source indexes changed"
        )
    return plan


def job_for(plan, pool_id):
    found = [j for j in plan["jobs"] if j["pool_id"] == pool_id]
    r.require(len(found) == 1, "Unplanned pool ID")
    return found[0]


def receipt_path(pool_id, kind):
    r.require(
        pool_id in {f"p{i:02d}" for i in range(9)} and kind in KINDS, "Unknown job"
    )
    return BASE / "runs" / pool_id / kind / "complete.json"


def receipt(pool_id, kind):
    path = receipt_path(pool_id, kind)
    if not path.exists():
        return None
    done = r.read(path)
    r.require(
        done["plan_sha256"] == r.sha(PLAN)
        and done["pool_id"] == pool_id
        and (done["kind"] == kind),
        "Receipt belongs to another job",
    )
    r.require(
        done["entries"] == (3 if kind.startswith("smoke") else 1295),
        "Incomplete receipt",
    )
    r.verify_hashes(done["hashes"])
    r.require(
        tree_hashes(ROOT / done["attempt"] / "chroma") == done["index_hashes"],
        "Completed index snapshot changed",
    )
    return done


def comparison(before, after):
    result = {key: r.js(before[key]) == r.js(after[key]) for key in COMPARE_FIELDS}
    result["entry"] = r.js(before) == r.js(after)
    return result


def delivered_context(raw, reference):
    r.require(
        len(raw["example_groups"]) == 1
        and raw["example_groups"][0][0] == "Labelled examples",
        "Unexpected retrieved example groups",
    )
    result = deepcopy(raw)
    for kind in ("definitions", "guidelines"):
        result[kind] = deepcopy(reference["context"][kind])
    return result


def make_entry(row, job, retriever, events, known, reference, plan_sha):
    events.clear()
    result = retriever.retrieve(row["text"], row["lang"])
    r.require(not result.feedback, "Feedback must remain empty")
    raw = prompt.from_retrieval(result)
    trace_tools.validate_context(raw, known, row["lang"], "hybrid")
    context = delivered_context(raw, reference)
    trace_tools.validate_context(context, known, row["lang"], "hybrid")
    messages = prompt.render_messages(row["text"], row["lang"], context, "corrected")
    return {
        **row,
        "condition": job["condition"],
        "pool_id": job["pool_id"],
        "strategy": "hybrid",
        "plan_sha256": plan_sha,
        "raw_context": raw,
        "context": context,
        "context_sha256": r.signature(context),
        "messages": messages,
        "messages_sha256": r.signature(messages),
        "retrieved": prompt.hit_ids(context),
        "trace": deepcopy(events),
        "final_hits": [trace_tools.hit_trace(h) for h in result.all_hits()],
    }


def data(job):
    import numpy as np

    records = r.rows(ROOT / job["records"])
    offsets = r.read(ROOT / job["vector_rows"])
    full = r.rows(CORE / "kb/full/records.jsonl")
    r.require(
        [full[i] for i in offsets] == records and len(records) == job["count"],
        "Indexed records/vector rows diverge",
    )
    source = np.load(r.BASE / "full_vectors.npy", mmap_mode="r", allow_pickle=False)
    matrix = np.asarray(source[offsets], dtype=np.float32)
    dev = r.rows(CORE / "dev_inputs.jsonl")
    queries = np.load(r.BASE / "query_vectors.npy", allow_pickle=False)
    r.validate_vectors(matrix, len(records))
    r.validate_vectors(queries, len(dev))
    return (records, matrix, dev, queries)


def copy_index(source_receipt, destination):
    r.require(not destination.exists(), "Never replace an index")
    src = ROOT / source_receipt["attempt"] / "chroma"
    r.require(
        tree_hashes(src) == source_receipt["index_hashes"], "Index changed before copy"
    )
    destination = owned(destination)
    destination.mkdir(parents=True)
    for name, sha in source_receipt["index_hashes"].items():
        original = ROOT / name
        target = owned(destination / original.relative_to(src))
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, target)
        r.require(r.sha(target) == sha, "Index copy differs")
    r.require(
        tree_hashes(src) == source_receipt["index_hashes"],
        "Source index changed during copy",
    )


def export_index(col, records, expected, path):
    import numpy as np

    maximum = r.verify_collection(col, records, expected)
    actual = []
    for start in range(0, len(records), 256):
        rr = records[start : start + 256]
        got = col.get(ids=[rec["id"] for rec in rr], include=["embeddings"])
        mapped = dict(zip(got["ids"], got["embeddings"]))
        actual.extend((mapped[rec["id"]] for rec in rr))
    actual = np.asarray(actual, dtype=np.float32)
    r.validate_vectors(actual, len(records))
    r.require(
        float(np.max(np.abs(actual - expected))) < 2e-07,
        "Exported index vectors differ",
    )
    buffer = io.BytesIO()
    np.save(buffer, actual, allow_pickle=False)
    put_bytes(path, buffer.getvalue())
    return maximum


def worker(pool_id, kind):
    import chromadb
    from src.hsrag.kb import _flatten_meta

    plan = validate()
    job = job_for(plan, pool_id)
    path = receipt_path(pool_id, kind)
    r.require(
        receipt(pool_id, kind) is None, "Completed worker must be skipped by parent"
    )
    root = path.parent
    r.require(
        not any(root.glob("attempt_*")),
        "Incomplete attempt preserved; ask Astra before continuing",
    )
    r.require(shutil.disk_usage(STUDY).free > 8 * 2**30, "At least 8 GiB free required")
    attempt = owned(root / ("attempt_" + str(time.time_ns())))
    attempt.mkdir(parents=True)
    wall_start = time.perf_counter()
    save(
        attempt / "start.json",
        {
            "created_utc": utc(),
            "pool_id": pool_id,
            "condition": job["condition"],
            "kind": kind,
            "plan_sha256": r.sha(PLAN),
        },
    )
    parent_kind = {
        "prepare": "smoke",
        "smoke_replay": "smoke",
        "replay": "prepare",
    }.get(kind)
    source = receipt(pool_id, parent_kind) if parent_kind else None
    if parent_kind:
        r.require(source is not None, "Prior completed index missing")
        copy_index(source, attempt / "chroma")
    records, matrix, dev, queries = data(job)
    all_dev = dev
    if kind.startswith("smoke"):
        by_key = {row["key"]: row for row in dev}
        dev = [by_key[k] for k in plan["smoke_keys"]]
    identity = {
        "embedding_model": r.MODEL,
        "embedding_revision": r.REVISION,
        "kb_version": r.sha(ROOT / job["records"])[:16],
        "pool_id": pool_id,
        "retrieval_plan_sha256": r.sha(PLAN),
        "hnsw:space": "cosine",
    }
    client = chromadb.PersistentClient(path=str(owned(attempt / "chroma")))
    if kind == "smoke":
        r.require(not client.list_collections(), "New index directory not empty")
        col = client.create_collection("hate_kb", metadata=identity)
        for start in range(0, len(records), 128):
            rr = records[start : start + 128]
            col.add(
                ids=[rec["id"] for rec in rr],
                documents=[rec["text"] for rec in rr],
                metadatas=[_flatten_meta(rec) for rec in rr],
                embeddings=matrix[start : start + 128].tolist(),
            )
    else:
        col = client.get_collection("hate_kb")
    r.require(
        all((col.metadata.get(k) == v for k, v in identity.items())),
        "Index origin changed",
    )
    maxdiff = export_index(col, records, matrix, attempt / "indexed_vectors.npy")
    encoder = r.CachedEncoder([row["text"] for row in all_dev], queries)
    retriever = r.make_retriever(col, ROOT / job["records"], plan["retrieval"], encoder)
    events = trace_tools.instrument(retriever)
    known = {rec["id"]: rec for rec in records}
    references = {
        row["key"]: row
        for row in r.rows(ROOT / plan["reference_prompts"]["full_reference"])
    }
    replay = kind in ("smoke_replay", "replay")
    prior = (
        {row["key"]: row for row in r.rows(ROOT / source["prompts"])} if replay else {}
    )
    output = attempt / ("comparisons.jsonl" if replay else "prompts.jsonl")
    differences = Counter()
    abstract_replacements = Counter()
    timings = []
    selected = []
    retrieve_start = time.perf_counter()
    with output.open("x", encoding="utf8", newline="\n") as f:
        for i, row in enumerate(dev):
            start = time.perf_counter()
            entry = make_entry(
                row, job, retriever, events, known, references[row["key"]], r.sha(PLAN)
            )
            timings.append({"key": row["key"], "seconds": time.perf_counter() - start})
            if kind == "smoke":
                plain = r.make_retriever(
                    col, ROOT / job["records"], plan["retrieval"], encoder
                )
                raw = plain.retrieve(row["text"], row["lang"])
                r.require(
                    r.js(prompt.from_retrieval(raw)) == r.js(entry["raw_context"])
                    and r.js([trace_tools.hit_trace(h) for h in raw.all_hits()])
                    == r.js(entry["final_hits"]),
                    "Traced/untraced retrieval differs",
                )
            for bucket in ("definitions", "guidelines"):
                abstract_replacements[bucket + "_raw_contexts_differ_from_fixed"] += (
                    entry["raw_context"][bucket] != entry["context"][bucket]
                )
            if replay:
                flags = comparison(prior[row["key"]], entry)
                out = {
                    "key": row["key"],
                    "condition": job["condition"],
                    "matches": flags,
                }
                for key, equal in flags.items():
                    differences[key] += not equal
                if not flags["entry"]:
                    out["actual_entry"] = entry
            else:
                out = entry
            f.write(
                json.dumps(out, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
            f.flush()
            selected.append(row["key"])
            if (i + 1) % 100 == 0 or i + 1 == len(dev):
                print(
                    f"{kind} {job['condition']}: {i + 1}/{len(dev)} ({time.perf_counter() - retrieve_start:.1f}s)",
                    flush=True,
                )
    save_rows(attempt / "timings.jsonl", timings)
    r.verify_collection(col, records, matrix)
    validate()
    done = {
        "status": "worker_finished_parent_sealing_pending",
        "completed_utc": utc(),
        "plan_sha256": r.sha(PLAN),
        "pool_id": pool_id,
        "condition": job["condition"],
        "kind": kind,
        "attempt": r.rel(attempt),
        "entries": len(selected),
        "keys": selected,
        "index_origin": identity,
        "index_vector_max_abs_difference": maxdiff,
        "indexed_vectors": r.rel(attempt / "indexed_vectors.npy"),
        "output": r.rel(output),
        "replay_differences": dict(differences),
        "fixed_abstract_diagnostics": dict(abstract_replacements),
        "retrieval_seconds": sum((t["seconds"] for t in timings)),
        "worker_seconds": time.perf_counter() - wall_start,
        "source_receipt": (
            r.rel(receipt_path(pool_id, parent_kind)) if parent_kind else None
        ),
        "source_receipt_sha256": (
            r.sha(receipt_path(pool_id, parent_kind)) if parent_kind else None
        ),
        "actual_API_calls": 0,
        "CUDA_used": False,
        "hashes": {
            r.rel(p): r.sha(p)
            for p in (
                attempt / "start.json",
                output,
                attempt / "timings.jsonl",
                attempt / "indexed_vectors.npy",
            )
        },
    }
    if not replay:
        done["prompts"] = r.rel(output)
    save(attempt / "worker_complete.json", done)


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
            raise RuntimeError("Another pool retrieval launcher is running") from None
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def child(args, label):
    log = owned(BASE / "logs" / (label + "_" + str(time.time_ns()) + ".txt"))
    log.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log.open("x", encoding="utf8") as f:
        process = subprocess.Popen(
            [sys.executable, "-X", "utf8", "-B", "-u", "-m", *args],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf8",
        )
        try:
            for line in process.stdout:
                f.write(line)
                f.flush()
                print(line, end="", flush=True)
            code = process.wait()
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            raise
    r.require(
        code == 0, "Preparation stopped; preserve log and ask Astra: " + r.rel(log)
    )
    return (log, time.perf_counter() - start)


def seal(pool_id, kind, log, seconds):
    path = receipt_path(pool_id, kind)
    roots = list(path.parent.glob("attempt_*"))
    r.require(
        len(roots) == 1 and (roots[0] / "worker_complete.json").exists(),
        "Missing/ambiguous worker completion",
    )
    attempt = roots[0]
    done = r.read(attempt / "worker_complete.json")
    r.require(
        done["pool_id"] == pool_id and done["kind"] == kind, "Worker receipt identity"
    )
    r.verify_hashes(done["hashes"])
    done["hashes"].update(
        {
            r.rel(attempt / "worker_complete.json"): r.sha(
                attempt / "worker_complete.json"
            ),
            r.rel(log): r.sha(log),
        }
    )
    done.update(
        status="completed_pool_pass_pending_independent_audit",
        index_hashes=tree_hashes(attempt / "chroma"),
        child_elapsed_seconds=seconds,
        sealed_utc=utc(),
    )
    save(path, done)
    r.require(receipt(pool_id, kind) is not None, "Seal failed")


def reference_entry(row, condition, reference, plan_sha):
    r.require(
        all((reference[k] == v for k, v in row.items())),
        "Reference target/scope differs",
    )
    r.require(
        r.signature(reference["messages"]) == reference["messages_sha256"],
        "Reference message signature",
    )
    return {
        **row,
        "condition": condition,
        "pool_id": condition,
        "strategy": "hybrid",
        "plan_sha256": plan_sha,
        "context": deepcopy(reference["context"]),
        "context_sha256": r.signature(reference["context"]),
        "messages": deepcopy(reference["messages"]),
        "messages_sha256": reference["messages_sha256"],
        "retrieved": deepcopy(reference["retrieved"]),
        "retrieval_source": "inherited_accepted_core_payload",
    }


def assemble(plan):
    path = BASE / "assembly.json"
    if path.exists():
        done = r.read(path)
        r.verify_hashes(done["hashes"])
        return
    dev = r.rows(CORE / "dev_inputs.jsonl")
    all_entries = []
    receipts = {}
    for condition, filename in plan["reference_prompts"].items():
        previous = {row["key"]: row for row in r.rows(ROOT / filename)}
        all_entries += [
            reference_entry(row, condition, previous[row["key"]], r.sha(PLAN))
            for row in dev
        ]
    for job in plan["jobs"]:
        for kind in ("prepare", "replay"):
            done = receipt(job["pool_id"], kind)
            r.require(done is not None, "Incomplete retrieval matrix")
            receipts[r.rel(receipt_path(job["pool_id"], kind))] = r.sha(
                receipt_path(job["pool_id"], kind)
            )
        done = receipt(job["pool_id"], "prepare")
        all_entries += r.rows(ROOT / done["prompts"])
    expected = {
        (row["key"], cond)
        for row in dev
        for cond in r.read(STUDY / "scientific_plan.json")["conditions"]
    }
    r.require(
        len(all_entries)
        == len({(x["key"], x["condition"]) for x in all_entries})
        == len(expected)
        == 14245,
        "Final matrix size",
    )
    r.require(
        {(x["key"], x["condition"]) for x in all_entries} == expected,
        "Final matrix membership",
    )
    save_rows(BASE / "prompts.jsonl", all_entries)
    save(
        path,
        {
            "status": "all_first_pass_prompts_assembled_independent_audit_pending",
            "plan_sha256": r.sha(PLAN),
            "entries": len(all_entries),
            "hashes": {
                r.rel(BASE / "prompts.jsonl"): r.sha(BASE / "prompts.jsonl"),
                **receipts,
            },
        },
    )


def run(mode):
    with lock():
        plan = validate()
        if mode == "run":
            ready = r.read(BASE / "preflight.json")
            r.require(
                ready["status"] == "ready_for_user_full_retrieval_not_API"
                and ready["plan_sha256"] == r.sha(PLAN),
                "Independent smoke review required",
            )
            r.verify_hashes(ready["hashes"])
        modes = ("smoke", "smoke_replay") if mode == "smoke" else ("prepare", "replay")
        for job in plan["jobs"]:
            for kind in modes:
                if receipt(job["pool_id"], kind):
                    print(
                        f"{kind} {job['condition']}: verified and skipped", flush=True
                    )
                    continue
                r.require(
                    not any(
                        receipt_path(job["pool_id"], kind).parent.glob("attempt_*")
                    ),
                    "Incomplete attempt: ask Astra to inspect before resuming",
                )
                log, seconds = child(
                    [MODULE, "worker", "--pool", job["pool_id"], "--kind", kind],
                    job["pool_id"] + "_" + kind,
                )
                seal(job["pool_id"], kind, log, seconds)
        if mode == "run":
            assemble(plan)
        child(
            [
                "analysis.knowledge.sampled_banks.verify_retrieval",
                "smoke" if mode == "smoke" else "full",
            ],
            mode + "_audit",
        )
        validate()
        print(
            (
                "SHORT POOL PREFLIGHT COMPLETE."
                if mode == "smoke"
                else "ALL POOL PROMPTS, COLD REPLAYS AND OFFLINE CHECKS COMPLETE. Tell Astra to inspect. No Qwen calls were made."
            ),
            flush=True,
        )


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["tests", "freeze", "validate", "smoke", "run", "worker"]
    )
    parser.add_argument("--pool")
    parser.add_argument("--kind", choices=KINDS)
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", no_network), patch.object(
        socket, "create_connection", no_network
    ):
        if args.mode == "tests":
            tests()
        elif args.mode == "freeze":
            freeze()
        elif args.mode == "validate":
            validate()
            print(
                "Frozen pool retrieval plan and original artifacts verified. No retrieval started."
            )
        elif args.mode == "worker":
            r.require(
                args.pool is not None and args.kind is not None,
                "Worker arguments required",
            )
            worker(args.pool, args.kind)
        else:
            run(args.mode)


if __name__ == "__main__":
    main()

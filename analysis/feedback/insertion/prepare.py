import os

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ.update(
    CUDA_VISIBLE_DEVICES="",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    TOKENIZERS_PARALLELISM="false",
    ANONYMIZED_TELEMETRY="False",
)
import argparse
import ast
from collections import Counter
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time
from types import FunctionType, MethodType, ModuleType, SimpleNamespace
from unittest.mock import patch
import numpy as np
from analysis.feedback.insertion import common as c
from analysis.common import retrieval as retrieval
from analysis.common import prompt as renderer

PLAN = c.BASE / "preparation_plan.json"
SCIENCE = c.BASE / "scientific_plan_v1.json"


@lru_cache(maxsize=1)
def runtime():

    def module(name, path, providers, excluded):
        tree = ast.parse(path.read_text(encoding="utf8"))
        tree.body = [
            n
            for n in tree.body
            if not (isinstance(n, ast.ImportFrom) and n.module in excluded)
        ]
        m = ModuleType(name)
        m.__dict__.update(providers)
        sys.modules[name] = m
        exec(compile(tree, str(path), "exec"), m.__dict__)
        return m

    kb = module(
        "astra_a14_cached_kb",
        c.ROOT / "src/hsrag/kb.py",
        {"SentenceTransformer": c.forbidden},
        {"sentence_transformers"},
    )
    rt = module(
        "astra_a14_cached_retrieve",
        c.ROOT / "src/hsrag/retrieve.py",
        {
            "SentenceTransformer": c.forbidden,
            "load_collection": c.forbidden,
            "unsentinel_meta": kb.unsentinel_meta,
        },
        {"sentence_transformers", "src.hsrag.kb"},
    )
    return SimpleNamespace(kb=kb, Retriever=rt.Retriever, mmr_select=rt.mmr_select)


def cached_retriever(col, path, cfg, encoder):
    rt = runtime()

    def model(name):
        c.require(name == retrieval.MODEL, "Unexpected cached model")
        return encoder

    def collection(_path, name):
        c.require(
            name == retrieval.MODEL and col.metadata["embedding_model"] == name,
            "Unexpected owned collection model",
        )
        return col

    init = FunctionType(
        rt.Retriever.__init__.__code__,
        {
            **rt.Retriever.__init__.__globals__,
            "SentenceTransformer": model,
            "load_collection": collection,
        },
    )
    result = rt.Retriever.__new__(rt.Retriever)
    init(result, path.parent / "chroma", path, retrieval.MODEL, dict(cfg))
    return result


def sources():
    return [
        Path(__file__),
        Path(c.__file__),
        Path(retrieval.__file__),
        Path(renderer.__file__),
        c.ROOT / "src/hsrag/retrieve.py",
        c.ROOT / "src/hsrag/kb.py",
        c.ROOT / "src/hsrag/prompt.py",
        c.ROOT / "config/taxonomy.yaml",
        c.ROOT / "prompts/classify_v1.txt",
        c.ROOT / "analysis/feedback/insertion/validate.py",
        c.ROOT / "analysis/feedback/insertion/scores_earlier.py",
        c.ROOT / "tests/feedback/summary/test_five_slot.py",
    ]


def snapshot(path):
    return c.bindings((p for p in sorted(path.rglob("*")) if p.is_file()))


def freeze():
    if PLAN.exists():
        validate()
        print("Five-slot scientific/input plan verified.")
        return
    old = c.read(c.old.BASE / "preparation_plan.json")["inputs"]
    c.verify(old)
    audited = c.read(
        c.ROOT
        / "Thesis_Log/astra_result/evidence/core_retrieval_check_20260915_v2.json"
    )
    reviewed = audited.get("verified_hashes", audited.get("input_hashes", {}))
    physical = snapshot(c.CORE / "retrieval_v1/full/chroma")
    c.require(
        physical and all((reviewed.get(n) == h for n, h in physical.items())),
        "Original physical index differs from reviewed snapshot",
    )
    paths = [
        c.CORE / n
        for n in [
            "plan.json",
            "kb/full/records.jsonl",
            "retrieval_v1/plan.json",
            "retrieval_v1/full_vectors.npy",
            "retrieval_v1/full/origin.json",
        ]
    ]
    paths += [
        c.old.BASE / n
        for n in [
            "preparation_plan.json",
            "independent_verification.json",
            "complete.json",
        ]
    ]
    inputs = {**old, **c.bindings(paths), **physical}
    pool = c.read(c.SELECTION / "actual_pools.json")
    conditions = ["reference"] + [p["condition"] for p in pool["pools"]]
    c.require(
        pool["budget"] == 40
        and pool["seeds"] == c.SEEDS
        and (len(set(conditions)) == 19),
        "Pool matrix differs",
    )
    roster = c.rows(c.TAXONOMY / "roster.jsonl")
    c.require(
        len(roster) == 675
        and Counter((r["population"] for r in roster))
        == {"benign": 208, "hateful_types": 467},
        "Roster changed",
    )
    c.jsonlines(c.BASE / "roster.jsonl", roster)
    core = c.read(c.CORE / "plan.json")
    c.require(
        core["retrieval"]["strategy"] == "hybrid"
        and core["retrieval"]["k_examples"] == 5
        and (core["retrieval"].get("rrf_k", 60) == 60)
        and (core["retrieval"].get("use_mmr") is True)
        and (core["retrieval"]["mmr_lambda"] == 0.7)
        and (core["retrieval"]["mmr_pool_factor"] == 3),
        "Original retrieval contract differs",
    )
    science = {
        "created_utc": c.utc(),
        "study": "five_slot",
        "conditions": conditions,
        "pool_seeds": c.SEEDS,
        "pool_budget": 40,
        "population_counts": {"hateful_types": 467, "benign": 208},
        "decoder_repeats": 3,
        "votes_per_prediction": 3,
        "primary": [
            {
                "name": "benign_false_positive_rate",
                "gold_hate": False,
                "left": "error_balanced",
                "right": "reference",
                "bootstrap_seed": 2026101014,
            },
            {
                "name": "hateful_false_negative_rate",
                "gold_hate": True,
                "left": "error_balanced",
                "right": "reference",
                "bootstrap_seed": 2026101015,
            },
        ],
        "primary_interval": 0.975,
        "bootstrap_draws": 10000,
        "bootstrap_generator": "PCG64",
        "bootstrap_unit": "Whole original text-overlap cluster; keep all pool seeds and decoder repeats together",
        "primary_mask": "Per population, all three balanced-error pools x three decoder repeats plus three reference repeats nonnull (12 outputs)",
        "averaging": "Equal decoder-repeat means inside pool, then equal pool means; shared reference has only three real repeats",
        "supporting": [
            "balanced versus positive-only errors",
            "already-correct and random source controls",
            "binary macro F1 and balanced accuracy at this artificial prevalence",
            "annotated hate-type regressions",
            "benign source strata",
            "correction exposure by count/polarity/rank",
            "no-delivery contexts changed by reranking",
            "pool overlap, tokens, calls and every run",
            "similarity and overwritten base example identity",
        ],
        "supporting_interval": 0.95,
        "sensitivities": [
            "all-three-valid-votes",
            "one stable representative per overlap cluster",
            "null-as-error/opposite bounds",
            "same union of cold-replay-changing identities excluded across all conditions",
            "leave-one-pool and leave-one-batch-out",
        ],
        "abstract_policy": "The same two definitions and two guidelines from each audited taxonomy rag_full input are held fixed across all 19 conditions. Only ordinary example retrieval is rerun.",
        "index_policy": "Each condition starts from its own byte-identical copy of the audited full index. Add 40 exact cached FP32 vectors in frozen pool order, in one batch. Reference adds zero. No original index opened.",
        "retrieval_policy": "Original Retriever hybrid/MMR example path: language-filtered Chroma top30, MMR lambda=.7 to dense top10 + original positive BM25 top10, RRF60 to five. No special feedback bucket or query gold.",
        "replay_policy": "First pass authoritative; two cold replays in fresh processes. Preserve all differences. Any changed ordered IDs or messages stops before live execution for explicit review. Metadata-only rank differences reported, not silently erased.",
        "scope": "New prospective family on previously exposed English DEV. Simulated existing binary labels, not expert feedback rounds. No clean TEST confirmation. Historical six-slot results contextual only.",
        "interpretation": "Report false alarms and missed hate jointly. Failure to detect recall loss is not non-inferiority. Conditional-on-delivery results descriptive, not randomized causal effects.",
        "failure_policy": "No outcome-based stopping, best-pool choice, replacement generation, deduplication of identical prompts across conditions, or invented fine-head labels.",
        "budget": {
            "texts": 675,
            "conditions": 19,
            "prompts": 12825,
            "predictions": 38475,
            "ordinary_calls": 115425,
            "automatic_audit_batches": 4,
        },
    }
    c.save(SCIENCE, science)
    c.save(
        PLAN,
        {
            "created_utc": c.utc(),
            "inputs": inputs,
            "original_index_hashes": physical,
            "source_hashes": c.bindings(sources()),
            "scientific_plan_sha256": c.sha(SCIENCE),
            "protocol_sha256": c.sha(c.BASE / "STUDY_PROTOCOL.md"),
            "roster_sha256": c.sha(c.BASE / "roster.jsonl"),
            "retrieval": core["retrieval"],
            "environment": core["environment"],
            "new_embeddings": 0,
            "cuda_jobs": 0,
        },
    )
    print(
        "A14 protocol and 19-condition matrix frozen before new retrieval/model outcomes.",
        flush=True,
    )


def validate():
    p = c.read(PLAN)
    c.verify(p["inputs"])
    c.verify(p["source_hashes"])
    from analysis.knowledge.banks.build import environment

    c.require(environment() == p["environment"], "Retrieval environment changed")
    c.require(
        c.sha(SCIENCE) == p["scientific_plan_sha256"]
        and c.sha(c.BASE / "STUDY_PROTOCOL.md") == p["protocol_sha256"]
        and (c.sha(c.BASE / "roster.jsonl") == p["roster_sha256"]),
        "Scientific protocol changed",
    )
    return p


def material(condition):
    science = c.read(SCIENCE)
    c.require(condition in science["conditions"], "Unknown condition")
    full = c.rows(c.CORE / "kb/full/records.jsonl")
    pools = {
        p["condition"]: p for p in c.read(c.SELECTION / "actual_pools.json")["pools"]
    }
    source_records = c.read(c.SELECTION / "gate_only_records.json")
    ids = [] if condition == "reference" else pools[condition]["ids"]
    added = [c.ordinary(source_records[i]) for i in ids]
    c.require(
        len({r["id"] for r in full + added}) == len(full) + len(added),
        "Inserted ID already in original KB",
    )
    full_vectors = np.load(
        c.CORE / "retrieval_v1/full_vectors.npy", allow_pickle=False, mmap_mode="r"
    )
    sources = c.rows(c.CANDIDATES / "roster.jsonl")
    si = {r["id"]: i for i, r in enumerate(sources)}
    source_vectors = np.load(
        c.CANDIDATES / "query_vectors.npy", allow_pickle=False, mmap_mode="r"
    )
    for source_id, record in zip(ids, added):
        c.require(
            sources[si[source_id]]["text"] == record["text"]
            and record["provenance"]["source_id"] == source_id,
            "Source vector text/ID differs",
        )
    extra_vectors = np.asarray(
        [source_vectors[si[i]] for i in ids], dtype=np.float32
    ).reshape(len(ids), 1024)
    return (full, added, full_vectors, extra_vectors)


def copy_index(condition, pilot=False):
    p = validate()
    root = c.BASE / ("pilot" if pilot else "retrieval") / condition
    source = c.CORE / "retrieval_v1/full/chroma"
    dest = root / "chroma"
    prefix = c.rel(source) + "/"
    expected = {n[len(prefix) :]: h for n, h in p["original_index_hashes"].items()}
    receipt = root / "copy.json"
    if receipt.exists():
        c.require(
            c.read(receipt)["source"] == expected
            and c.read(receipt)["plan_sha256"] == c.sha(PLAN),
            "Copied index provenance changed",
        )
        return root
    dest.mkdir(parents=True, exist_ok=True)
    c.require(
        {v.relative_to(dest).as_posix() for v in dest.rglob("*") if v.is_file()}
        <= set(expected),
        "Unexpected partial index content",
    )
    for name, h in expected.items():
        out = dest / name
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            shutil.copyfile(source / name, out)
        c.require(c.sha(out) == h, "Physical copy differs")
    c.save(
        receipt,
        {
            "source": expected,
            "plan_sha256": c.sha(PLAN),
            "originals_opened_with_chroma": False,
        },
    )
    return root


def compose(row, result, base, known, condition, token, source_ids):
    context = renderer.from_retrieval(result)
    c.require(
        not context["definitions"] and (not context["guidelines"]),
        "Example retrieval unexpectedly fetched abstract knowledge",
    )
    context["definitions"] = deepcopy(base["context"]["definitions"])
    context["guidelines"] = deepcopy(base["context"]["guidelines"])
    examples = retrieval.validate_context(context, known, "en")
    messages = renderer.render_messages(row["text"], "en", context, "corrected")
    c.require(messages[0] == base["messages"][0], "System/schema changed")
    delivered = [h["id"] for h in examples if h["id"] in source_ids]
    return {
        "key": row["key"],
        "item_id": row["id"],
        "lang": "en",
        "text": row["text"],
        "population": row["population"],
        "condition": condition,
        "context": context,
        "messages": messages,
        "context_sha256": c.sig(context),
        "messages_sha256": c.sig(messages),
        "retrieved": renderer.hit_ids(context),
        "inserted_ids_delivered": delivered,
        "delivered_gates": [known[i]["meta"]["gate"] for i in delivered],
        "reference_input_tokens": c.old.old.token_count(token, messages),
        "scientific_plan_sha256": c.sha(SCIENCE),
    }


def save_array(path, value):
    path = c.own(path)
    if path.exists():
        saved = np.load(path, allow_pickle=False, mmap_mode="r")
        c.require(
            saved.dtype == value.dtype and np.array_equal(saved, value),
            "Existing index-vector cache differs",
        )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial_" + __import__("uuid").uuid4().hex)
    with temp.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temp, path)
    finally:
        temp.unlink()


def metadata_identity(origin, records_sha, plan_sha, condition):
    return {
        **{k: v for k, v in origin.items() if k != "hnsw:space"},
        "kb_version": records_sha[:16],
        "plan_sha256": plan_sha,
        "astra_original_kb_version": origin["kb_version"],
        "astra_original_plan_sha256": origin["plan_sha256"],
        "astra_condition": condition,
    }


def persisted_schema(root):
    path = c.own(root / "chroma/chroma.sqlite3")
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as connection:
        rows = connection.execute(
            "SELECT schema_str FROM collections WHERE name=?", ("hate_kb",)
        ).fetchall()
    c.require(len(rows) == 1, "Ambiguous persistent collection schema")
    schema = __import__("json").loads(rows[0][0])
    c.require(
        schema["keys"]["#embedding"]["float_list"]["vector_index"]["config"]["space"]
        == "cosine",
        "Cosine index configuration changed",
    )
    return schema


def checked_index_vectors(col, records, source_vectors, full_count, root):

    _flatten_meta = runtime().kb._flatten_meta
    c.require(
        col.count() == len(records)
        and set(col.get(include=[])["ids"]) == {r["id"] for r in records},
        "Wrong index roster",
    )
    actual = np.empty((len(records), 1024), dtype=np.float32)
    maximum = 0.0
    for start in range(0, len(records), 256):
        block = records[start : start + 256]
        got = col.get(
            ids=[r["id"] for r in block],
            include=["documents", "metadatas", "embeddings"],
        )
        by_id = {
            i: (t, m, v)
            for i, t, m, v in zip(
                got["ids"], got["documents"], got["metadatas"], got["embeddings"]
            )
        }
        for i, record in enumerate(block, start):
            text, meta, vector = by_id[record["id"]]
            c.require(
                text == record["text"] and meta == _flatten_meta(record),
                "Indexed source text/annotation differs",
            )
            actual[i] = vector
        delta = float(
            np.max(
                np.abs(
                    actual[start : start + len(block)]
                    - source_vectors[start : start + len(block)]
                )
            )
        )
        maximum = max(maximum, delta)
        c.require(delta < 2e-07, "Indexed vector differs from reviewed source cache")
    full_path = c.BASE / "index_vectors/full.npy"
    inserted_path = root / "inserted_index_vectors.npy"
    save_array(full_path, actual[:full_count])
    save_array(inserted_path, actual[full_count:])
    return (actual, maximum, c.bindings([full_path, inserted_path]))


def worker(condition, mode, pilot=False):
    p = validate()
    root = c.BASE / ("pilot" if pilot else "retrieval") / condition
    finished = root / mode / "complete.json"
    if finished.exists():
        receipt = c.read(finished)
        c.verify(receipt["outputs"])
        c.require(
            receipt["plan_sha256"] == c.sha(PLAN)
            and receipt["condition"] == condition
            and (receipt["mode"] == mode)
            and (receipt["pilot"] == pilot),
            "Completed retrieval attribution differs",
        )
        print(condition + "/" + mode + " verified complete.", flush=True)
        return
    root = copy_index(condition, pilot)
    print(
        condition
        + "/"
        + mode
        + ": validating owned index and cached vectors; no neural libraries or model calls.",
        flush=True,
    )
    full, added, fv, av = material(condition)
    records = full + added
    c.jsonlines(root / "records.jsonl", records)
    import chromadb

    rt = runtime()
    _flatten_meta = rt.kb._flatten_meta
    with c.offline():
        client = chromadb.PersistentClient(path=str(root / "chroma"))
        col = client.get_collection("hate_kb")
        origin = c.read(c.CORE / "retrieval_v1/full/origin.json")
        identity = metadata_identity(
            origin, c.sha(root / "records.jsonl"), c.sha(PLAN), condition
        )
        before_schema = persisted_schema(root)
        c.require(
            col.metadata == identity or (mode == "freeze" and col.metadata == origin),
            "Copy model/index identity differs",
        )
        existing = set(col.get(include=[])["ids"])
        wanted = {r["id"] for r in records}
        original = {r["id"] for r in full}
        c.require(
            original <= existing <= wanted,
            "Index has missing originals or unexpected IDs",
        )
        if mode == "freeze":
            pending = [i for i, r in enumerate(added) if r["id"] not in existing]
            c.require(
                not pending or pending == list(range(len(added))),
                "Partly inserted index: preserve for review",
            )
            if pending:
                col.add(
                    ids=[r["id"] for r in added],
                    documents=[r["text"] for r in added],
                    metadatas=[_flatten_meta(r) for r in added],
                    embeddings=av.tolist(),
                )
            if col.metadata != identity:
                col.modify(metadata=identity)
        c.require(
            persisted_schema(root) == before_schema,
            "Metadata tagging changed the actual index configuration",
        )
        c.save(
            root / "index_identity.json",
            {
                "condition": condition,
                "plan_sha256": c.sha(PLAN),
                "metadata": identity,
                "schema_sha256": c.sig(before_schema),
                "space": "cosine",
                "records_sha256": c.sha(root / "records.jsonl"),
            },
        )
        c.require(col.count() == len(records), "Incomplete copied index")
        vectors = np.concatenate((fv, av), axis=0)
        indexed_vectors, maxdiff, index_vector_hashes = checked_index_vectors(
            col, records, vectors, len(full), root
        )
        vi = {r["id"]: i for i, r in enumerate(records)}
        roster = c.rows(c.BASE / "roster.jsonl")
        if pilot:
            roster = [
                r
                for pop in ["benign", "hateful_types"]
                for r in sorted(
                    (r for r in roster if r["population"] == pop),
                    key=lambda r: r["key"],
                )[:3]
            ]
        core = c.rows(c.CORE / "dev_inputs.jsonl")
        qi = {r["key"]: i for i, r in enumerate(core)}
        qv = np.load(
            c.CORE / "retrieval_v1/query_vectors.npy", allow_pickle=False, mmap_mode="r"
        )
        bases = {
            r["key"]: r
            for r in c.rows(c.TAXONOMY / "prompts.jsonl")
            if r["condition"] == "rag_full"
        }
        cfg = {
            **p["retrieval"],
            "k_definitions": 0,
            "k_guidelines": 0,
            "k_definitions_mode": "flat",
            "k_examples_feedback": 0,
        }
        encoder = retrieval.CachedEncoder([], qv)
        retriever = cached_retriever(col, root / "records.jsonl", cfg, encoder)
        trace = {}
        dense_query, lexical_query = (retriever._query_kind, retriever._bm25_kind)

        def dense(*args, **kwargs):
            hits = dense_query(*args, **kwargs)
            c.require(
                len(hits) == 30 and all((h.vec is not None for h in hits)),
                "Original MMR candidate path required",
            )
            for hit in hits:
                c.require(
                    np.array_equal(np.asarray(hit.vec), indexed_vectors[vi[hit.id]]),
                    "Query vectors differ from audited index vectors",
                )
            trace["dense_candidates"] = [{"id": h.id, "score": h.score} for h in hits]
            return hits

        def lexical(*args, **kwargs):
            hits = lexical_query(*args, **kwargs)
            trace["bm25"] = [{"id": h.id, "score": h.score} for h in hits]
            return hits

        retriever._query_kind, retriever._bm25_kind = (dense, lexical)
        Retriever, mmr_select = (rt.Retriever, rt.mmr_select)

        def recorded_mmr(*args, **kwargs):
            hits = mmr_select(*args, **kwargs)
            trace["dense"] = [{"id": h.id, "score": h.score} for h in hits]
            return hits

        wrapped = FunctionType(
            Retriever.retrieve.__code__,
            {**Retriever.retrieve.__globals__, "mmr_select": recorded_mmr},
        )
        retriever.retrieve = MethodType(wrapped, retriever)
        known = {r["id"]: r for r in records}
        source_ids = {r["id"] for r in added}
        token = c.old.old.tokenizer()
        output = root / mode
        finished = output / "complete.json"
        if finished.exists():
            receipt = c.read(finished)
            c.verify(receipt["outputs"])
            c.require(
                receipt["plan_sha256"] == c.sha(PLAN),
                "Completed retrieval has wrong plan",
            )
            print(condition + "/" + mode + " verified complete.", flush=True)
            return
        original_prompts = c.rows(root / "prompts.jsonl") if mode != "freeze" else []
        if mode != "freeze":
            c.require(
                [r["key"] for r in original_prompts] == [r["key"] for r in roster],
                "Frozen roster incomplete",
            )
        started = time.perf_counter()

        def make(i, row):
            c.require(
                core[qi[row["key"]]]["text"] == row["text"] == bases[row["key"]]["text"]
                and core[qi[row["key"]]]["item_id"] == row["id"],
                "Query identity differs",
            )
            encoder.lookup = {row["text"]: qv[qi[row["key"]]]}
            trace.clear()
            out = compose(
                row,
                retriever.retrieve(row["text"], "en"),
                bases[row["key"]],
                known,
                condition,
                token,
                source_ids,
            )
            c.require(
                set(trace) == {"dense_candidates", "dense", "bm25"},
                "Expected original MMR and two example channels",
            )
            out["dense_candidates"] = deepcopy(trace["dense_candidates"])
            out["channels"] = {
                name: deepcopy(trace[name]) for name in ["dense", "bm25"]
            }
            if mode == "freeze":
                return out
            prev = original_prompts[i]
            changed = sorted(
                (k for k in prev.keys() | out.keys() if prev.get(k) != out.get(k))
            )
            return {
                "key": row["key"],
                "changed_fields": changed,
                "same_messages": out["messages"] == prev["messages"],
                "same_retrieved_ids": out["retrieved"] == prev["retrieved"],
                **({"actual_entry": out} if changed else {}),
            }

        def progress(n, total, reused):
            print(
                f"{condition}/{mode}: {n}/{total}; {reused} durable rows reused; {time.perf_counter() - started:.1f}s",
                flush=True,
            )

        result, chunks = c.gather(
            output / "chunks",
            roster,
            make,
            {
                "condition": condition,
                "mode": mode,
                "pilot": pilot,
                "plan_sha256": c.sha(PLAN),
            },
            progress=progress,
        )
        path = (
            root / "prompts.jsonl" if mode == "freeze" else output / "comparisons.jsonl"
        )
        c.jsonlines(path, result)
        diffs = (
            {}
            if mode == "freeze"
            else {
                name: sum((test(r) for r in result))
                for name, test in [
                    ("entry", lambda r: bool(r["changed_fields"])),
                    ("messages", lambda r: not r["same_messages"]),
                    ("retrieved_ids", lambda r: not r["same_retrieved_ids"]),
                ]
            }
        )
        c.verify(p["original_index_hashes"])
        c.save(
            finished,
            {
                "plan_sha256": c.sha(PLAN),
                "condition": condition,
                "mode": mode,
                "pilot": pilot,
                "rows": len(result),
                "seconds": time.perf_counter() - started,
                "index_records": len(records),
                "vector_max_abs_difference": maxdiff,
                "differences": diffs,
                "checkpoint": chunks,
                "outputs": {
                    **chunks["hashes"],
                    **index_vector_hashes,
                    c.rel(path): c.sha(path),
                },
                "Qwen_calls": 0,
                "new_embeddings": 0,
            },
        )


def run(pilot=False):
    validate()
    science = c.read(SCIENCE)
    conditions = (
        ["reference", f"error_positive_s{c.SEEDS[0]}", f"error_balanced_s{c.SEEDS[0]}"]
        if pilot
        else science["conditions"]
    )
    with c.a.item_lock(c.BASE / ("pilot.lock" if pilot else "preparation.lock")):
        for condition in conditions:
            for mode in ["freeze", "replay1", "replay2"]:
                args = [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-B",
                    "-u",
                    "-m",
                    __spec__.name,
                    "worker",
                    "--condition",
                    condition,
                    "--pass-name",
                    mode,
                ]
                if pilot:
                    args += ["--pilot"]
                subprocess.run(args, cwd=c.ROOT, check=True)
    print(
        "Retrieval passes complete; independent offline verification follows.",
        flush=True,
    )


def main():
    os.chdir(c.ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["freeze", "validate", "pilot", "run", "worker"]
    )
    parser.add_argument("--condition")
    parser.add_argument("--pass-name", choices=["freeze", "replay1", "replay2"])
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    with c.offline():
        if args.mode == "freeze":
            freeze()
        elif args.mode == "validate":
            validate()
        elif args.mode in ("pilot", "run"):
            run(args.mode == "pilot")
        else:
            worker(args.condition, args.pass_name, args.pilot)


if __name__ == "__main__":
    main()

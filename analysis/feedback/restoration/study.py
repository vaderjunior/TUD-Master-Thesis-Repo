import os

for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_key] = "2"
os.environ.update(
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    ANONYMIZED_TELEMETRY="False",
    TOKENIZERS_PARALLELISM="false",
)
import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import shutil
import socket
import time
from types import FunctionType, MethodType
from unittest.mock import patch
import numpy as np
import pandas as pd
import yaml
from tokenizers import Tokenizer
from analysis.feedback.explanations import calibration as c
from analysis.common import prompt_archive as isolated
from analysis.common import prompt as renderer

ROOT = c.ROOT
BASE = ROOT / "experiments/astra/sq3_full_kb_taxonomy_20261001_v1"
CORE = c.CORE
RET = CORE / "retrieval_v1"
PACKET = c.PACKET
OLD = ROOT / "experiments/astra/sq3_full_kb_feedback_20260928_v1"
TOK = ROOT / "experiments/astra/qwen_tokenizer_assets_20261001_v1"
LABELS = [
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
]
POOLS = ["full"] + LABELS
EMPTY = {"definitions": [], "guidelines": [], "example_groups": []}


def sig(obj):
    return c.digest(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )


def put(name, obj):
    path = BASE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    c.write(path, obj)


def put_lines(name, rows):
    path = BASE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    c.jsonlines(path, rows)


def verify_plan():
    plan = c.read(BASE / "preparation_plan.json")
    for name, expected in plan["inputs"].items():
        c.require(c.sha(ROOT / name) == expected, "Changed frozen input: " + name)
    c.verify_active()
    return plan


def freeze():
    c.require(
        not (BASE / "preparation_plan.json").exists(), "Preserve frozen preparation"
    )
    c.require(
        c.read(BASE / "tests_passed.json")["successful"],
        "Current passing tests required",
    )
    c.verify_active()
    core, ret = (c.read(CORE / "plan.json"), c.read(RET / "plan.json"))
    for p in (CORE / "kb/full/records.jsonl", CORE / "dev_inputs.jsonl"):
        c.require(
            c.sha(p) == core["generated_hashes"][c.rel(p)], "Core artifact origin"
        )
    for name, expected in ret["generated_hashes"].items():
        c.require(c.sha(ROOT / name) == expected, "Cached vector origin")
    audit = c.read(ROOT / "experiments/astra/english_kb_audit_20260914_v1/plan.json")
    c.require(
        c.sha(ROOT / "data/processed/en_dev.parquet")
        == audit["input_hashes"]["data/processed/en_dev.parquet"],
        "DEV origin",
    )
    c.require(
        c.sha(RET / "full/prompts.jsonl")
        == c.read(RET / "full/complete.json")["prompt_sha256"],
        "Historical prompt origin",
    )
    paths = [
        BASE / "STUDY_PROTOCOL.md",
        BASE / "tests_passed.json",
        CORE / "plan.json",
        CORE / "dev_inputs.jsonl",
        CORE / "kb/full/records.jsonl",
        RET / "plan.json",
        RET / "full_vectors.npy",
        RET / "query_vectors.npy",
        RET / "record_vector_sources.jsonl",
        RET / "query_vector_sources.jsonl",
        RET / "full/prompts.jsonl",
        RET / "full/complete.json",
        RET / "review_complete_v2.json",
        ROOT / "data/processed/en_dev.parquet",
        OLD / "feasibility_v1/roster.jsonl",
        PACKET / "prepared.json",
        PACKET / "independent_check.json",
        PACKET / "taxonomy_inventory.json",
        TOK / "complete.json",
        c.ACTIVE,
        ROOT
        / "Thesis_Log/astra_result/astra_sq3_followups_accepted_plan_20260930_v1.md",
        ROOT / "experiments/astra/english_kb_audit_20260914_v1/plan.json",
    ]
    paths += [PACKET / f"taxonomy_manifests/{label}.json" for label in LABELS]
    paths += [
        ROOT / name
        for name in (
            "src/hsrag/schema.py",
            "src/hsrag/parse.py",
            "src/hsrag/vote.py",
            "src/hsrag/prompt.py",
            "src/hsrag/retrieve.py",
            "src/hsrag/kb.py",
            "config/taxonomy.yaml",
            "prompts/classify_v1.txt",
        )
    ]
    paths += [
        ROOT / name
        for name in (
            "analysis/feedback/restoration/study.py",
            "analysis/common/prompt_archive.py",
            "analysis/feedback/restoration/validate_earlier.py",
            "tests/feedback/summary/test_taxonomy.py",
            "analysis/feedback/explanations/calibration.py",
            "analysis/common/prompt.py",
            "analysis/common/retrieval.py",
            "analysis/knowledge/strategies/prepare.py",
            "analysis/knowledge/strategies/verify.py",
        )
    ]
    paths += [TOK / name for name in c.read(TOK / "complete.json")["files"]]
    index = sorted((p for p in (RET / "full/chroma").rglob("*") if p.is_file()))
    c.require(
        len(index) == 6
        and (not any((p.name.endswith(("-wal", "-shm")) for p in index))),
        "Source index is not a settled six-file snapshot",
    )
    paths += index
    put(
        "preparation_plan.json",
        {
            "status": "rules_frozen_before_scope_screen_and_prompt_delivery",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": {c.rel(p): c.sha(p) for p in paths},
            "initial_hateful": 467,
            "initial_benign": 208,
            "labels": LABELS,
            "near_threshold": 0.95,
            "borderline_margin": 0.0005,
            "retrieval": core["retrieval"],
            "input_token_guard": 8192,
            "primary_bootstrap_seed": 2026100103,
            "primary_interval": 0.9875,
            "primary_draws": 10000,
            "source_index_files": [c.rel(p) for p in index],
            "network_calls": 0,
            "logical_withdrawal": "eligible-ID pre-search filter on isolated copied full ANN index; filtered BM25 rebuilt",
            "retained_first_pass": True,
        },
    )
    print(
        json.dumps(
            {
                "status": "taxonomy_preparation_frozen",
                "sha256": c.sha(BASE / "preparation_plan.json"),
            }
        ),
        flush=True,
    )


def groups(n, edges):
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in edges:
        x, y = (find(a), find(b))
        parent[max(x, y)] = min(x, y)
    out = defaultdict(list)
    for i in range(n):
        out[find(i)].append(i)
    return list(out.values())


def scope():
    started = time.perf_counter()
    plan = verify_plan()
    c.require(not (BASE / "scope_complete.json").exists(), "Scope already prepared")
    core = c.lines(CORE / "dev_inputs.jsonl")
    qmap = {r["item_id"]: i for i, r in enumerate(core)}
    records = c.lines(CORE / "kb/full/records.jsonl")
    gold = {
        r["id"]: r
        for r in pd.read_parquet(ROOT / "data/processed/en_dev.parquet").to_dict(
            "records"
        )
    }
    old = c.lines(OLD / "feasibility_v1/roster.jsonl")
    hate_ids = {r["id"] for r in old}
    benign_ids = {
        r["item_id"]
        for r in core
        if r["lang"] == "en" and gold[r["item_id"]]["gate"] is False
    }
    c.require(
        len(hate_ids) == 467 and len(benign_ids) == 208 and (not hate_ids & benign_ids),
        "Original populations changed",
    )
    ids = sorted(hate_ids | benign_ids)
    qraw = np.load(RET / "query_vectors.npy", mmap_mode="r", allow_pickle=False)
    vraw = np.load(RET / "full_vectors.npy", mmap_mode="r", allow_pickle=False)
    qsrc, vsrc = (
        c.lines(RET / "query_vector_sources.jsonl"),
        c.lines(RET / "record_vector_sources.jsonl"),
    )
    for r, s in zip(core, qsrc):
        c.require(
            r["key"] == s["key"] and c.digest(r["text"]) == s["text_sha256"],
            "Query vector identity",
        )
    for r, s in zip(records, vsrc):
        c.require(
            r["id"] == s["id"] and c.digest(r["text"]) == s["text_sha256"],
            "Record vector identity",
        )
    for r in old:
        d = gold[r["id"]]
        c.require(
            r["text"] == d["text"]
            and d["gate"] is True
            and (set(r["gold"]["hate_type"]) == set(d["hate_types"])),
            "Old roster/gold mismatch",
        )
    q = c.unit(qraw[[qmap[i] for i in ids]])
    names = [c.normalize(gold[i]["text"]) for i in ids]
    similarity = q @ q.T
    edges = []
    evidence = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            score = float(similarity[i, j])
            same = names[i] == names[j]
            trigger = same or score >= 0.95
            if trigger:
                edges.append((i, j))
            if trigger or abs(score - 0.95) <= 0.0005:
                c.require(
                    abs(sum((float(a) * float(b) for a, b in zip(q[i], q[j]))) - score)
                    < 1e-12,
                    "Overlap dot product",
                )
                evidence.append(
                    {
                        "left": ids[i],
                        "right": ids[j],
                        "kind": "evaluation",
                        "cosine": score,
                        "normalized_equal": same,
                        "edge": trigger,
                    }
                )
    train_ix = [i for i, r in enumerate(records) if r["kind"] == "example"]
    c.require(len(train_ix) == 71091, "KB example census")
    normalized = defaultdict(list)
    for i in train_ix:
        normalized[c.normalize(records[i]["text"])].append(records[i]["id"])
    touched = set()
    nearest = np.full(len(ids), -np.inf)
    nearest_ids = [None] * len(ids)
    for i, name in enumerate(names):
        for rid in normalized[name]:
            touched.add(i)
            evidence.append(
                {"left": ids[i], "right": rid, "kind": "train_exact", "edge": True}
            )
    for start in range(0, len(train_ix), 1024):
        ix = train_ix[start : start + 1024]
        values = c.unit(vraw[ix])
        sims = q @ values.T
        for i in range(len(ids)):
            j = int(sims[i].argmax())
            score = float(sims[i, j])
            if score > nearest[i]:
                nearest[i], nearest_ids[i] = (score, records[ix[j]]["id"])
            for j in np.flatnonzero(sims[i] >= 0.9495):
                score = float(sims[i, j])
                c.require(
                    abs(
                        sum((float(a) * float(b) for a, b in zip(q[i], values[j])))
                        - score
                    )
                    < 1e-12,
                    "TRAIN dot product",
                )
                trigger = score >= 0.95
                if trigger:
                    touched.add(i)
                evidence.append(
                    {
                        "left": ids[i],
                        "right": records[ix[j]]["id"],
                        "kind": "train_cosine",
                        "cosine": score,
                        "edge": trigger,
                    }
                )
    components = []
    included = []
    excluded = []
    cluster_map = {}
    for members in groups(len(ids), edges):
        group_ids = [ids[i] for i in members]
        cluster = "taxonomy-" + sig(group_ids)[:16]
        representative = min(
            group_ids,
            key=lambda iid: (c.digest("20261001-taxonomy-representative|" + iid), iid),
        )
        keep = not bool(set(members) & touched)
        components.append(
            {
                "cluster": cluster,
                "ids": group_ids,
                "representative": representative,
                "included": keep,
                "reason": None if keep else "TRAIN_overlap_in_component",
            }
        )
        for iid in group_ids:
            cluster_map[iid] = cluster
            (included if keep else excluded).append(iid)
    roster = []
    for iid in sorted(included):
        row, d = (core[qmap[iid]], gold[iid])
        c.require(
            row["text"] == d["text"] and d["lang"] == "en", "DEV text/language mismatch"
        )
        types = list(d["hate_types"]) if iid in hate_ids else None
        if types is not None:
            c.require(
                types and len(types) == len(set(types)) and (set(types) <= set(LABELS)),
                "Type gold",
            )
        roster.append(
            {
                "id": iid,
                "key": "en/" + iid,
                "text": d["text"],
                "lang": "en",
                "source": d["source"],
                "population": "hateful_types" if iid in hate_ids else "benign",
                "gold": {"hate": bool(d["gate"]), "hate_type": types},
                "cluster": cluster_map[iid],
            }
        )
    c.require(
        all(
            (
                any(
                    (
                        r["gold"]["hate_type"] and label in r["gold"]["hate_type"]
                        for r in roster
                    )
                )
                for label in LABELS
            )
        ),
        "A category lost all support",
    )
    put_lines("roster.jsonl", roster)
    put("clusters.json", components)
    put_lines("overlap_evidence.jsonl", evidence)
    put("excluded_inputs.json", excluded)
    put_lines(
        "nearest_train.jsonl",
        [
            {"id": iid, "nearest_id": nearest_ids[i], "cosine": float(nearest[i])}
            for i, iid in enumerate(ids)
        ],
    )
    full_tax_bytes = (ROOT / "config/taxonomy.yaml").read_bytes()
    tax = yaml.safe_load(full_tax_bytes)
    c.require(
        list(tax["dimensions"]["hate_type"]["labels"]) == LABELS, "Type order changed"
    )
    (BASE / "taxonomies").mkdir()
    (BASE / "taxonomies/full.yaml").write_bytes(full_tax_bytes)
    raw_lines = (CORE / "kb/full/records.jsonl").read_bytes().splitlines(keepends=True)
    c.require(len(raw_lines) == len(records), "Unexpected blank record stream")
    inventory = {}
    for label in LABELS:
        manifest = c.read(PACKET / f"taxonomy_manifests/{label}.json")
        removed = set(
            manifest["removed_definition_ids"] + manifest["removed_example_ids"]
        )
        actual = {
            r["id"]
            for r in records
            if r["kind"] == "example"
            and label in (r["meta"].get("hate_types") or [])
            or (
                r["kind"] == "definition"
                and r["dimension"] == "hate_type"
                and (r["label"] == label)
            )
        }
        c.require(removed == actual, "Withdrawal manifest mismatch")
        retained = b"".join(
            (line for line, r in zip(raw_lines, records) if r["id"] not in removed)
        )
        c.require(
            hashlib.sha256(retained).hexdigest()
            == manifest["retained_original_line_stream_sha256"],
            "Retained raw stream differs",
        )
        path = BASE / f"kb/{label}/records.jsonl"
        path.parent.mkdir(parents=True)
        with path.open("xb") as f:
            f.write(retained)
        changed = deepcopy(tax)
        del changed["dimensions"]["hate_type"]["labels"][label]
        with (BASE / f"taxonomies/{label}.yaml").open(
            "x", encoding="utf8", newline="\n"
        ) as f:
            yaml.safe_dump(changed, f, sort_keys=False, allow_unicode=True)
        bundle = isolated.bundle(BASE / f"taxonomies/{label}.yaml")
        put(f"taxonomies/{label}.schema.json", bundle.schema.Result.model_json_schema())
        inventory[label] = {
            "removed_ids": sorted(removed),
            "retained_records": len(records) - len(removed),
            "records_sha256": c.sha(path),
            "taxonomy_sha256": bundle.taxonomy_sha256,
        }
    put(
        "taxonomies/full.schema.json",
        isolated.bundle(
            BASE / "taxonomies/full.yaml"
        ).schema.Result.model_json_schema(),
    )
    conditions = [
        {"condition": "rag_full", "pool": "full", "taxonomy": "full"},
        {"condition": "zs_full", "pool": None, "taxonomy": "full"},
    ]
    for label in LABELS:
        conditions += [
            {"condition": "rag_removed_" + label, "pool": label, "taxonomy": label},
            {"condition": "rag_schema_" + label, "pool": label, "taxonomy": "full"},
            {"condition": "zs_removed_" + label, "pool": None, "taxonomy": label},
        ]
    science = {
        "status": "frozen_before_classifier_outputs",
        "protocol_sha256": c.sha(BASE / "STUDY_PROTOCOL.md"),
        "preparation_plan_sha256": c.sha(BASE / "preparation_plan.json"),
        "conditions": conditions,
        "roster_sha256": c.sha(BASE / "roster.jsonl"),
        "clusters_sha256": c.sha(BASE / "clusters.json"),
        "labels": LABELS,
        "repeats": [0, 1, 2],
        "votes": 3,
        "primary_mask_conditions": ["rag_full"] + ["rag_schema_" + l for l in LABELS],
        "primary_interval": 0.9875,
        "bootstrap_draws": 10000,
        "primary_bootstrap_seed": 2026100103,
        "input_count": len(roster),
        "prompts": len(roster) * 23,
        "predictions": len(roster) * 69,
        "ordinary_calls": len(roster) * 207,
        "primary_population": "hateful_types",
        "benign_population": "benign",
        "fresh_shared_references": True,
        "no_accuracy_based_stopping": True,
        "historically_exposed_DEV": True,
    }
    put("scientific_plan_v1.json", science)
    put(
        "scope_summary.json",
        {
            "counts": dict(Counter((r["population"] for r in roster))),
            "benign_sources": dict(
                Counter((r["source"] for r in roster if r["population"] == "benign"))
            ),
            "type_support": dict(
                Counter((l for r in roster for l in r["gold"]["hate_type"] or []))
            ),
            "clusters": sum((x["included"] for x in components)),
            "non_singleton_clusters": [x for x in components if len(x["ids"]) > 1],
            "excluded_count": len(excluded),
            "overlap_evidence_rows": len(evidence),
            "inventory": inventory,
            "seconds": time.perf_counter() - started,
            "network_calls": 0,
        },
    )
    outputs = [
        p
        for p in BASE.rglob("*")
        if p.is_file()
        and (
            p.name
            in (
                "scientific_plan_v1.json",
                "roster.jsonl",
                "clusters.json",
                "overlap_evidence.jsonl",
                "excluded_inputs.json",
                "nearest_train.jsonl",
                "scope_summary.json",
            )
            or p.is_relative_to(BASE / "taxonomies")
            or p.is_relative_to(BASE / "kb")
        )
    ]
    put(
        "scope_complete.json",
        {
            "outputs": {p.relative_to(BASE).as_posix(): c.sha(p) for p in outputs},
            "active_bindings": c.verify_active(),
        },
    )
    print(
        json.dumps(
            {
                "status": "taxonomy_scope_frozen",
                "inputs": len(roster),
                "ordinary_calls": science["ordinary_calls"],
                "clusters": sum((x["included"] for x in components)),
                "excluded": len(excluded),
            },
            indent=2,
        ),
        flush=True,
    )


def verify_scope():
    for name, expected in c.read(BASE / "scope_complete.json")["outputs"].items():
        c.require(c.sha(BASE / name) == expected, "Scope artifact changed: " + name)


def flat_metadata(r):
    out = {k: r[k] for k in ("kind", "lang", "source")}
    out.update(dimension=r["dimension"] or "", label=r["label"] or "")
    for k, v in r.get("meta", {}).items():
        out[k] = (
            "__none__"
            if v is None
            else json.dumps(list(v)) if isinstance(v, (list, tuple)) else v
        )
    return out


def copy_index():
    plan = verify_plan()
    verify_scope()
    dest = BASE / "chroma_copy"
    c.require(not dest.exists(), "Copy already exists; preserve it")
    shutil.copytree(RET / "full/chroma", dest, copy_function=shutil.copy2)
    for name in plan["source_index_files"]:
        p = ROOT / name
        c.require(
            c.sha(dest / p.relative_to(RET / "full/chroma"))
            == plan["inputs"][name]
            == c.sha(p),
            "Index copy/source differs",
        )
    import chromadb

    col = chromadb.PersistentClient(path=str(dest)).get_collection("hate_kb")
    records = c.lines(CORE / "kb/full/records.jsonl")
    v = np.load(RET / "full_vectors.npy", mmap_mode="r", allow_pickle=False)
    c.require(
        col.count() == len(records)
        and set(col.get(include=[])["ids"]) == {r["id"] for r in records},
        "Index roster differs",
    )
    indexed = np.empty_like(v)
    maximum = 0.0
    for start in range(0, len(records), 256):
        rr = records[start : start + 256]
        result = col.get(
            ids=[r["id"] for r in rr], include=["documents", "metadatas", "embeddings"]
        )
        got = {
            rid: (text, meta, vec)
            for rid, text, meta, vec in zip(
                result["ids"],
                result["documents"],
                result["metadatas"],
                result["embeddings"],
            )
        }
        for i, r in enumerate(rr, start):
            text, meta, vec = got[r["id"]]
            c.require(
                text == r["text"] and meta == flat_metadata(r),
                "Index text/annotation changed",
            )
            indexed[i] = vec
            maximum = max(maximum, float(np.max(np.abs(indexed[i] - v[i]))))
    c.require(maximum < 2e-07, "Index vectors changed")
    b = io.BytesIO()
    np.save(b, indexed, allow_pickle=False)
    with (BASE / "indexed_vectors.npy").open("xb") as f:
        f.write(b.getvalue())
    put(
        "index_checked.json",
        {
            "records": len(records),
            "source_snapshot_verified": True,
            "max_vector_difference": maximum,
            "indexed_vectors_sha256": c.sha(BASE / "indexed_vectors.npy"),
        },
    )
    verify_plan()
    print(
        json.dumps(
            {
                "status": "isolated_index_checked",
                "records": len(records),
                "max_vector_difference": maximum,
            }
        ),
        flush=True,
    )


class FilteredCollection:
    """Apply logical withdrawal before ANN search; fail on any returned leak."""

    def __init__(self, collection, records, filtered):
        self.collection = collection
        self.metadata = collection.metadata
        self.allowed = {r["id"] for r in records}
        self.ids = [r["id"] for r in records]
        self.filtered = filtered

    def query(self, **kwargs):
        c.require("ids" not in kwargs, "Caller supplied a competing ID filter")
        if self.filtered:
            kwargs["ids"] = self.ids
        result = self.collection.query(**kwargs)
        c.require(
            all((set(group) <= self.allowed for group in result["ids"])),
            "Withdrawn or unknown ANN candidate",
        )
        return result


def make_retriever(collection, records, path, cfg, query_map, filtered):
    module = isolated.retrieval_module()

    class Encoder:

        def encode(self, texts, normalize_embeddings=True):
            c.require(
                normalize_embeddings and all((t in query_map for t in texts)),
                "Uncached query",
            )
            return np.stack([query_map[t] for t in texts])

    wrapper = FilteredCollection(collection, records, filtered)
    result = module.Retriever(
        BASE / "chroma_copy",
        path,
        "BAAI/bge-m3",
        dict(cfg),
        encoder=Encoder(),
        collection=wrapper,
    )
    return result


def hit_trace(hit):
    c.require(np.isfinite(hit.score), "Nonfinite hit score")
    return {
        "id": hit.id,
        "kind": hit.kind,
        "score": float(hit.score),
        "via": hit.via,
        "ranks": deepcopy(hit.meta.get("_ranks", {})),
    }


def instrument(retriever):
    m = isolated.retrieval_module()
    events = []
    dense0, bm0 = (retriever._query_kind, retriever._bm25_kind)

    def dense(qvec, where, k, via, want_vecs=False):
        hits = dense0(qvec, where, k, via, want_vecs)
        events.append(
            {
                "operation": "dense_query",
                "where": deepcopy(where),
                "k": k,
                "want_vecs": want_vecs,
                "hits": [hit_trace(h) for h in hits],
            }
        )
        return hits

    def lexical(query, bucket_key, k):
        hits = bm0(query, bucket_key, k)
        events.append(
            {
                "operation": "bm25_query",
                "bucket": bucket_key,
                "k": k,
                "hits": [hit_trace(h) for h in hits],
            }
        )
        return hits

    def mmr(qvec, hits, lam=0.7, top_n=5):
        before = [h.id for h in hits]
        out = m.mmr_select(qvec, hits, lam, top_n)
        events.append(
            {
                "operation": "mmr",
                "lambda": lam,
                "top_n": top_n,
                "input_ids": before,
                "hits": [hit_trace(h) for h in out],
            }
        )
        return out

    def fuse(rank_lists, k=60, top_n=5):
        channels = {
            name: [hit_trace(h) for h in hits] for name, hits in rank_lists.items()
        }
        out = m.rrf_fuse(rank_lists, k, top_n)
        events.append(
            {
                "operation": "rrf",
                "k": k,
                "top_n": top_n,
                "channels": channels,
                "hits": [hit_trace(h) for h in out],
            }
        )
        return out

    retriever._query_kind, retriever._bm25_kind = (dense, lexical)
    fn = FunctionType(
        m.Retriever.retrieve.__code__,
        {**m.Retriever.retrieve.__globals__, "mmr_select": mmr, "rrf_fuse": fuse},
        argdefs=m.Retriever.retrieve.__defaults__,
    )
    retriever.retrieve = MethodType(fn, retriever)
    return events


def retrieve(pool, replay=False):
    plan = verify_plan()
    verify_scope()
    started = time.perf_counter()
    c.require(
        c.sha(BASE / "indexed_vectors.npy")
        == c.read(BASE / "index_checked.json")["indexed_vectors_sha256"],
        "Indexed vector snapshot changed",
    )
    import chromadb

    collection = chromadb.PersistentClient(
        path=str(BASE / "chroma_copy")
    ).get_collection("hate_kb")
    path = (
        CORE / "kb/full/records.jsonl"
        if pool == "full"
        else BASE / f"kb/{pool}/records.jsonl"
    )
    records = c.lines(path)
    core = c.lines(CORE / "dev_inputs.jsonl")
    q = np.load(RET / "query_vectors.npy", mmap_mode="r", allow_pickle=False)
    query_map = {r["text"]: q[i] for i, r in enumerate(core)}
    r = make_retriever(
        collection, records, path, plan["retrieval"], query_map, pool != "full"
    )
    events = instrument(r)
    roster = c.lines(BASE / "roster.jsonl")
    old = c.lines(BASE / f"retrieval/{pool}.jsonl") if replay else None
    outpath = BASE / f"{('replay' if replay else 'retrieval')}/{pool}.jsonl"
    outpath.parent.mkdir(exist_ok=True)
    changed = Counter()
    with outpath.open("x", encoding="utf8", newline="\n") as f:
        for i, row in enumerate(roster):
            events.clear()
            result = r.retrieve(row["text"], "en")
            c.require(
                not result.feedback
                and len(result.examples) == 5
                and (len(result.definitions) == 2)
                and (len(result.guidelines) == 2),
                "Retrieval budget changed",
            )
            context = renderer.from_retrieval(result)
            entry = {
                "key": row["key"],
                "text": row["text"],
                "lang": "en",
                "pool": pool,
                "strategy": "hybrid",
                "context": context,
                "context_sha256": sig(context),
                "retrieved": renderer.hit_ids(context),
                "final_hits": [hit_trace(h) for h in result.all_hits()],
                "trace": deepcopy(events),
            }
            if replay:
                flags = {
                    k: entry[k] == old[i][k]
                    for k in ("context", "retrieved", "final_hits", "trace")
                }
                changed.update((k for k, v in flags.items() if not v))
                line = {"key": row["key"], "matches": flags}
                if not all(flags.values()):
                    line["actual_entry"] = entry
            else:
                line = entry
            f.write(
                json.dumps(line, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
            if (i + 1) % 100 == 0 or i + 1 == len(roster):
                f.flush()
                print(
                    f"{('REPLAY' if replay else 'RETRIEVE')} {pool}: {i + 1}/{len(roster)} ({time.perf_counter() - started:.1f}s)",
                    flush=True,
                )
    put(
        f"{('replay' if replay else 'retrieval')}/{pool}.complete.json",
        {
            "file_sha256": c.sha(outpath),
            "rows": len(roster),
            "seconds": time.perf_counter() - started,
            "differences": dict(changed),
            "retained_first_pass": True,
            "active_bindings": c.verify_active(),
        },
    )


def local_chat(messages):
    return (
        "".join(
            (
                "<|im_start|>" + m["role"] + "\n" + m["text"] + "<|im_end|>\n"
                for m in messages
            )
        )
        + "<|im_start|>assistant\n"
    )


def prompts():
    started = time.perf_counter()
    plan = verify_plan()
    verify_scope()
    science = c.read(BASE / "scientific_plan_v1.json")
    roster = c.lines(BASE / "roster.jsonl")
    contexts = {}
    for pool in POOLS:
        p = BASE / f"retrieval/{pool}.jsonl"
        c.require(
            c.sha(p) == c.read(BASE / f"retrieval/{pool}.complete.json")["file_sha256"],
            "Retrieval origin",
        )
        contexts[pool] = {r["key"]: r for r in c.lines(p)}
        replay = BASE / f"replay/{pool}.jsonl"
        c.require(
            c.sha(replay)
            == c.read(BASE / f"replay/{pool}.complete.json")["file_sha256"],
            "Cold replay origin",
        )
    bundles = {
        name: isolated.bundle(BASE / f"taxonomies/{name}.yaml") for name in POOLS
    }
    tokenizer = Tokenizer.from_file(str(TOK / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    original = {r["key"]: r for r in c.lines(RET / "full/prompts.jsonl")}
    stats = {condition["condition"]: Counter() for condition in science["conditions"]}
    rows = []
    changes = []
    lengths = []
    contents = set()
    delivery = defaultdict(Counter)
    for row in roster:
        contexts_for_row = {
            pool: contexts[pool][row["key"]]["context"] for pool in POOLS
        }
        historical = original[row["key"]]
        full_messages = bundles["full"].render(
            row["text"], "en", contexts_for_row["full"]
        )
        c.require(
            full_messages
            == renderer.render_messages(
                row["text"], "en", contexts_for_row["full"], "corrected"
            ),
            "Full renderer changed",
        )
        if full_messages != historical["messages"]:
            changes.append(
                {
                    "id": row["id"],
                    "old_messages_sha256": historical["messages_sha256"],
                    "new_messages_sha256": sig(full_messages),
                    "old_retrieved": historical["retrieved"],
                    "new_retrieved": renderer.hit_ids(contexts_for_row["full"]),
                }
            )
        for label in LABELS:
            ctx = contexts_for_row[label]
            removed = bundles[label].render(row["text"], "en", ctx)
            restored = bundles["full"].render(row["text"], "en", ctx)
            c.require(
                removed[1] == restored[1] and removed[0] != restored[0],
                "Schema/context intervention not isolated",
            )
            delivery[label][row["population"] + "_same_user_context_as_full"] += (
                removed[1] == full_messages[1]
            )
            full_ctx = contexts_for_row["full"]
            delivery[label][row["population"] + "_full_target_example_reached"] += any(
                (
                    label in (h["meta"].get("hate_types") or [])
                    for _, hh in full_ctx["example_groups"]
                    for h in hh
                )
            )
            delivery[label][
                row["population"] + "_full_target_definition_reached"
            ] += any(
                (h["id"] == "def-hate_type-" + label for h in full_ctx["definitions"])
            )
        for condition in science["conditions"]:
            ctx = contexts_for_row[condition["pool"]] if condition["pool"] else EMPTY
            b = bundles[condition["taxonomy"]]
            messages = b.render(row["text"], "en", ctx)
            token_count = len(
                tokenizer.encode(local_chat(messages), add_special_tokens=False).ids
            )
            c.require(
                token_count <= plan["input_token_guard"],
                "Prompt over input-token guard; no truncation",
            )
            entry = {
                "key": row["key"],
                "item_id": row["id"],
                "lang": "en",
                "text": row["text"],
                **condition,
                "taxonomy_sha256": b.taxonomy_sha256,
                "context": ctx,
                "context_sha256": sig(ctx),
                "messages": messages,
                "messages_sha256": sig(messages),
                "input_tokens_local": token_count,
                "scientific_plan_sha256": c.sha(BASE / "scientific_plan_v1.json"),
            }
            rows.append(entry)
            lengths.append(token_count)
            contents.add(entry["messages_sha256"])
            stats[condition["condition"]][row["population"]] += 1
    c.require(len(rows) == science["prompts"], "Prompt matrix incomplete")
    put_lines("prompts.jsonl", rows)
    put("historical_full_differences.json", changes)
    summary = {
        "inputs": len(roster),
        "prompts": len(rows),
        "unique_messages": len(contents),
        "predictions": science["predictions"],
        "ordinary_calls": science["ordinary_calls"],
        "local_input_tokens_min": min(lengths),
        "local_input_tokens_max": max(lengths),
        "historical_full_message_differences": len(changes),
        "conditions": {k: dict(v) for k, v in stats.items()},
        "delivery": {k: dict(v) for k, v in delivery.items()},
        "scope": c.read(BASE / "scope_summary.json"),
        "replay": {
            pool: c.read(BASE / f"replay/{pool}.complete.json") for pool in POOLS
        },
        "seconds": time.perf_counter() - started,
        "Qwen_calls": 0,
        "new_embeddings": 0,
        "CUDA": False,
    }
    put("summary.json", summary)
    files = [
        BASE / name
        for name in (
            "scientific_plan_v1.json",
            "scope_complete.json",
            "prompts.jsonl",
            "summary.json",
            "historical_full_differences.json",
            "index_checked.json",
            "indexed_vectors.npy",
        )
    ]
    files += [
        p
        for folder in ("retrieval", "replay")
        for p in (BASE / folder).iterdir()
        if p.is_file()
    ]
    put(
        "prepared.json",
        {
            "status": "prepared_awaiting_independent_check",
            "preparation_plan_sha256": c.sha(BASE / "preparation_plan.json"),
            "outputs": {p.relative_to(BASE).as_posix(): c.sha(p) for p in files},
            "active_bindings": c.verify_active(),
        },
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in summary.items()
                if k not in ("delivery", "scope", "conditions", "replay")
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    args = argparse.ArgumentParser()
    args.add_argument(
        "mode",
        choices=["freeze", "scope", "copy-index", "retrieve", "replay", "prompts"],
    )
    args.add_argument("--pool", choices=POOLS)
    opt = args.parse_args()
    with patch.object(
        socket.socket,
        "connect",
        side_effect=RuntimeError("Network forbidden in offline preparation"),
    ):
        if opt.mode == "freeze":
            freeze()
        elif opt.mode == "scope":
            scope()
        elif opt.mode == "copy-index":
            copy_index()
        elif opt.mode in ("retrieve", "replay"):
            c.require(opt.pool is not None, "Pool required")
            retrieve(opt.pool, opt.mode == "replay")
        else:
            prompts()


if __name__ == "__main__":
    main()

from src.hsrag.paths import PROJECT_ROOT
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import os

os.environ.update(
    HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", ANONYMIZED_TELEMETRY="False"
)
ROOT = PROJECT_ROOT
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
BASE = CORE / "retrieval_v1"
OUT = ROOT / "Thesis_Log/astra_result/evidence/core_retrieval_check_20260915_v2.json"
TABLE = ROOT / "Thesis_Log/astra_result/astra_core_retrieval_20260915_v2.md"
RECEIPT = BASE / "review_complete_v2.json"
HASHES = {}
ROSTERS = {}


def require(ok, msg):
    if not ok:
        raise RuntimeError(msg)


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    HASHES[rel(p)] = h.hexdigest()
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def rows(p):
    return [
        json.loads(x)
        for x in Path(p).read_text(encoding="utf8").splitlines()
        if x.strip()
    ]


def signature(x):
    return hashlib.sha256(
        (
            json.dumps(x, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        ).encode()
    ).hexdigest()


def verify(mapping):
    for name, digest in mapping.items():
        require(sha(ROOT / name) == digest, f"Changed source/output: {name}")


def cache_roster(base, tag):
    key = (str(base), tag)
    if key not in ROSTERS:
        p = read(base / "plan.json")
        rr = rows(base / f"{tag}_texts.jsonl")
        require(
            sha(base / f"{tag}_texts.jsonl")
            == p["generated_hashes"][rel(base / f"{tag}_texts.jsonl")],
            "Cache roster changed",
        )
        ROSTERS[key] = (p, rr)
    return ROSTERS[key]


def source_texts(name, meta):

    p = ROOT / name
    if p.suffix == ".npz":
        base = p.parent.parent
        tag, start, _ = p.stem.split("_", 2)
        start = int(start)
        plan, rr = cache_roster(base, tag)
        block = rr[start : start + plan["cache_chunk"]]
        identity = {
            "plan_sha256": sha(base / "plan.json"),
            "model": "BAAI/bge-m3",
            "revision": "5617a9f61b028005a4858fdac845db406aefb181",
            "precision": "float32",
            "texts_sha256": signature([x["text"] for x in block]),
            "rows": len(block),
        }
        require(
            {k: v for k, v in meta.items() if k != "vectors_sha256"} == identity,
            "Source cache identity differs",
        )
        require(
            p.name == f"{tag}_{start:06d}_{signature(identity)[:16]}.npz",
            "Cache filename/identity mismatch",
        )
        require(
            sha(p) == read(base / "complete.json")["embedding_hashes"][name],
            "Cache no longer matches prior completed audit",
        )
        return (block, tag, "en" if "english_kb" in str(base) else "de")
    prior = ROOT / "experiments/astra/german_kb_comparison_20260913_v1"
    require(p == prior / "background_vectors.json", "Unexpected non-NPZ vector source")
    _, train = cache_roster(
        ROOT / "experiments/astra/german_kb_audit_20260913_v1", "train"
    )
    excluded = {x["text"] for x in train}
    ordered = [
        r
        for pool in ("original_small", "canonical_small", "canonical_full")
        for r in rows(prior / f"kb/{pool}/records.jsonl")
    ]
    texts = list(
        dict.fromkeys((r["text"] for r in ordered if r["text"] not in excluded))
    )
    require(
        meta
        == {"plan_sha256": sha(prior / "plan.json"), "texts_sha256": signature(texts)},
        "Background row identities differ",
    )
    require(
        sha(p) == read(prior / "setup_complete.json")["hashes"][name],
        "Background source changed",
    )
    return ([{"text": t} for t in texts], "background", None)


def check_vectors(filename, sourcefile, identities, idkey):
    import numpy as np

    matrix = np.load(BASE / filename, allow_pickle=False, mmap_mode="r")
    source = rows(BASE / sourcefile)
    require(
        matrix.dtype == np.float32 and matrix.shape == (len(identities), 1024),
        "Wrong assembled matrix",
    )
    require(bool(np.isfinite(matrix).all()), "Nonfinite assembled vector")
    norm_error = 0.0
    for start in range(0, len(matrix), 1024):
        norm_error = max(
            norm_error,
            float(
                np.max(np.abs(np.linalg.norm(matrix[start : start + 1024], axis=1) - 1))
            ),
        )
    require(norm_error < 0.0002, "Unnormalised assembled vectors")
    require(
        [x[idkey] for x in source] == [x[idkey] for x in identities],
        "Vector mapping order differs",
    )
    byfile = defaultdict(list)
    for i, (origin, item) in enumerate(zip(source, identities)):
        require(
            origin["text_sha256"] == hashlib.sha256(item["text"].encode()).hexdigest(),
            "Vector text identity differs",
        )
        byfile[origin["file"]].append((i, origin["row"]))
    for name, pairs in byfile.items():
        if name.endswith(".npz"):
            with np.load(ROOT / name, allow_pickle=False) as z:
                vv = z["vectors"]
                meta = json.loads(str(z["metadata"].item()))
            require(
                hashlib.sha256(vv.tobytes()).hexdigest() == meta["vectors_sha256"],
                "Raw source vector checksum differs",
            )
        else:
            obj = read(ROOT / name)
            vv = np.asarray(obj["vectors"], dtype=np.float32)
            meta = obj["identity"]
        block, kind, lang = source_texts(name, meta)
        require(
            vv.shape == (len(block), 1024) and vv.dtype == np.float32,
            "Source vector shape/dtype differs",
        )
        for i, j in pairs:
            require(type(j) is int and 0 <= j < len(block), "Invalid cache row")
            item = identities[i]
            origin = source[i]
            require(
                block[j]["text"] == item["text"] and origin["source_kind"] == kind,
                "Cache row text/kind differs",
            )
            require(lang is None or item["lang"] == lang, "Wrong cache language")
            if idkey == "key":
                require(kind == "heldout", "Query vector must come from heldout cache")
                refs = {(v["file"], v["id"]) for v in block[j]["references"]}
                require(
                    all(
                        (
                            (f"data/processed/{s['subset']}.parquet", item["item_id"])
                            in refs
                            for s in item["scopes"]
                        )
                    ),
                    "Query cache reference mismatch",
                )
            else:
                require(
                    kind == ("train" if item["kind"] == "example" else "background"),
                    "Record vector from wrong split",
                )
        a, b = zip(*pairs)
        require(
            np.array_equal(matrix[list(a)], vv[list(b)]),
            "Copied vectors differ from original cache",
        )
    return {
        "rows": len(identities),
        "source_files": len(byfile),
        "bitwise_equal": True,
        "source_text_and_split_identity_verified": True,
        "max_unit_norm_error": norm_error,
    }


def check_log(log, expected):
    require(
        [x["key"] for x in log] == [x["key"] for x in expected],
        "Replay log omitted/reordered/duplicated input",
    )
    totals = {
        "items": len(log),
        "entry_differences": 0,
        "message_differences": 0,
        "context_differences": 0,
        "retrieved_id_differences": 0,
    }
    names = {
        "same_entry": "entry_differences",
        "same_messages": "message_differences",
        "same_context": "context_differences",
        "same_retrieved_ids": "retrieved_id_differences",
    }
    for row, old in zip(log, expected):
        actual = row.get("actual_entry", old)
        changed = sorted(
            (
                k
                for k in set(actual) | set(old)
                if k not in actual or k not in old or actual[k] != old[k]
            )
        )
        flags = {
            "same_entry": not changed,
            "same_messages": actual["messages"] == old["messages"],
            "same_context": actual["context"] == old["context"],
            "same_retrieved_ids": actual["retrieved"] == old["retrieved"],
        }
        require(
            set(row)
            == {
                "key",
                "changed_fields",
                "actual_context_sha256",
                "actual_messages_sha256",
                *names,
            }
            | ({"actual_entry"} if changed else set()),
            "Unexpected/missing replay fields",
        )
        require(row["changed_fields"] == changed, "Replay changed fields differ")
        for k, v in flags.items():
            require(
                type(row[k]) is bool and row[k] == v,
                "Replay flags differ from actual saved entry",
            )
            totals[names[k]] += not v
        for part in ("context", "messages"):
            require(
                row[f"actual_{part}_sha256"]
                == signature(actual[part])
                == actual[f"{part}_sha256"],
                "Replay actual content/signature differs",
            )
    return totals


def index_snapshot(pool):
    return {
        rel(p): sha(p)
        for p in sorted((BASE / pool / "chroma").rglob("*"))
        if p.is_file()
    }


def changed_files(a, b):
    return sorted((k for k in set(a) | set(b) if a.get(k) != b.get(k)))


def inspect_replays(dev):
    rp = BASE / "replay_v1/plan.json"
    plan = read(rp)
    require(
        sha(rp) == "691f8891a7d642a05f9610dd446ea910bf3da49ef126c0f6e458ac4efeac1dd9",
        "Replay plan differs",
    )
    verify(plan["input_hashes"])
    require(
        plan["original_plan_sha256"] == sha(BASE / "plan.json")
        and plan["passes"] == 2
        and (plan["rows_per_full_pass"] == 1295),
        "Wrong replay contract",
    )
    pre = (
        ROOT
        / "Thesis_Log/astra_result/evidence/core_retrieval_replay_preflight_20260915_v1.json"
    )
    require(
        sha(pre) == "12310b25450137662579de28d0074e097790d8a6990a3770fe71f539b8f1692b",
        "Preflight history changed",
    )
    original = rows(BASE / "full/prompts.jsonl")
    key = "de/detox-1389930895067467778"
    require(original[1033]["key"] == key, "Original failed input position differs")
    failure = BASE / "logs/check_full_1789485784975188200.txt"
    require(
        f"Cold/current retrieval differs at full/{key}"
        in failure.read_text(encoding="utf8"),
        "Original failure missing",
    )
    result = {}
    last = None
    for name, mode, number in [
        ("probe_0", "probe", 0),
        ("full_0", "full", 0),
        ("full_1", "full", 1),
    ]:
        target = BASE / "replay_v1" / name
        receipt = read(target / "complete.json")
        sha(target / "complete.json")
        require(
            receipt["plan_sha256"] == sha(rp)
            and receipt["mode"] == mode
            and (receipt["pass"] == number)
            and (receipt["status"] == "diagnostic_complete_not_clearance"),
            "Wrong diagnostic receipt",
        )
        verify(receipt["hashes"])
        require(len(receipt["hashes"]) == 1, "Unexpected diagnostic files")
        logpath = ROOT / next(iter(receipt["hashes"]))
        require(logpath.is_relative_to(target), "Log belongs to different replay")
        require(
            {rel(p) for p in target.rglob("*") if p.is_file()}
            == {rel(target / "complete.json"), rel(logpath)},
            "Unreviewed extra replay attempt/file",
        )
        expected = (
            original if mode == "full" else [original[0], original[1033], original[-1]]
        )
        log = rows(logpath)
        totals = check_log(log, expected)
        require(totals == receipt["totals"], "Recomputed replay counts differ")
        require(
            receipt["index_vector_max_abs_difference"] < 2e-07,
            "Stored index vector verification failed",
        )
        changes = changed_files(
            receipt["physical_index_before"], receipt["physical_index_after"]
        )
        require(
            changes == [rel(BASE / "full/chroma/chroma.sqlite3")],
            "Unexpected index file drift",
        )
        if last is not None:
            require(
                last == receipt["physical_index_before"],
                "Index changed between replay passes",
            )
        last = receipt["physical_index_after"]
        if mode == "full":
            require(
                totals
                == {
                    "items": 1295,
                    "entry_differences": 0,
                    "message_differences": 0,
                    "context_differences": 0,
                    "retrieved_id_differences": 0,
                },
                "Full replay has differences requiring review",
            )
        else:
            require(
                totals
                == {
                    "items": 3,
                    "entry_differences": 1,
                    "message_differences": 0,
                    "context_differences": 1,
                    "retrieved_id_differences": 0,
                },
                "Probe history differs",
            )
            actual = log[1]["actual_entry"]
            old = expected[1]
            require(
                log[1]["changed_fields"] == ["context", "context_sha256"],
                "Probe changed more than context metadata",
            )
            clean = json.loads(json.dumps(actual["context"]))
            hits = [h for _, rr in clean["example_groups"] for h in rr]
            require(
                hits[0]["id"] == "ex-detox-1389917486057988096"
                and hits[0]["meta"]["_ranks"] == {"bm25": 1, "dense": 10},
                "Probe rank observation differs",
            )
            hits[0]["meta"]["_ranks"] = {"bm25": 1}
            require(clean == old["context"], "Additional probe differences exist")
        result[name] = {
            "receipt_sha256": sha(target / "complete.json"),
            "log_sha256": sha(logpath),
            "totals": totals,
            "index_vector_max_abs_difference": receipt[
                "index_vector_max_abs_difference"
            ],
            "physical_files_changed_during_pass": changes,
        }
    require(last == index_snapshot("full"), "Current index changed after final replay")
    require(
        result["full_0"]["log_sha256"] == result["full_1"]["log_sha256"],
        "Two nominally exact replay logs differ",
    )
    require(
        not (BASE / "complete.json").exists()
        and (not (BASE / "full/cold_check.json").exists()),
        "Original strict failure history was overwritten",
    )
    result["interpretation"] = (
        "Two complete exact replays support the frozen saved prompts. Preserve earlier unrecorded strict failure and captured metadata-only probe. Not a guarantee of universally deterministic future live retrieval."
    )
    result["original_failure"] = {
        "key": key,
        "one_based_index": 1034,
        "first_1033_entries_equal_before_stop": True,
        "actual_changed_entry_not_saved": True,
        "log_sha256": sha(failure),
    }
    return result


def inspect_sqlite(pool, records):

    import ast, sqlite3, yaml

    path = BASE / pool / "chroma/chroma.sqlite3"
    before = index_snapshot(pool)
    require(
        not Path(str(path) + "-wal").exists(),
        "A live WAL requires separate review; immutable reads would ignore it",
    )
    tree = ast.parse((ROOT / "src/hsrag/kb.py").read_text(encoding="utf8"))
    tree.body = [
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_flatten_meta"
    ]
    taxonomy = yaml.safe_load(
        (ROOT / "config/taxonomy.yaml").read_text(encoding="utf8")
    )
    keys = [
        k
        for dim, spec in taxonomy["dimensions"].items()
        if spec.get("type") == "multilabel"
        for k in (dim, dim + "s")
    ]
    ns = {"json": json, "LIST_META_KEYS": keys}
    exec(compile(tree, "offline_frozen_flatten_meta", "exec"), ns)

    def scalar(values):
        present = [i for i, v in enumerate(values) if v is not None]
        require(len(present) == 1, "Invalid SQLite scalar")
        i = present[0]
        return bool(values[i]) if i == 3 else values[i]

    with sqlite3.connect(
        path.resolve().as_uri() + "?mode=ro&immutable=1", uri=True
    ) as con:
        require(
            con.execute("PRAGMA integrity_check").fetchall() == [("ok",)],
            "SQLite integrity check failed",
        )
        foreign = con.execute("PRAGMA foreign_key_check").fetchall()
        require(
            foreign
            == [("segments", 1, "collection", 0), ("segments", 2, "collection", 0)],
            "Unexpected SQLite foreign-key finding",
        )
        segment_schema = con.execute(
            "SELECT sql FROM sqlite_master WHERE name='segments'"
        ).fetchone()[0]
        require(
            "REFERENCES collection(id)" in segment_schema,
            "Expected Chroma schema qualification differs",
        )
        migration = (
            ROOT
            / ".venv/Lib/site-packages/chromadb/migrations/sysdb/00009-segment-collection-not-null.sqlite.sql"
        )
        require(
            con.execute(
                "SELECT sql FROM migrations WHERE filename=?", (migration.name,)
            ).fetchone()[0]
            == migration.read_text(encoding="utf8"),
            "Installed/stored Chroma migration differs",
        )
        sha(migration)
        require(
            con.execute(
                "SELECT count(*) FROM segments s LEFT JOIN collections c ON s.collection=c.id WHERE c.id IS NULL"
            ).fetchone()[0]
            == 0,
            "Orphan segment against actual collections table",
        )
        collections = con.execute(
            "SELECT id,name,dimension,schema_str FROM collections"
        ).fetchall()
        require(
            len(collections) == 1 and collections[0][1:3] == ("hate_kb", 1024),
            "Wrong stored collection",
        )
        cid = collections[0][0]
        schema = json.loads(collections[0][3])
        require(
            schema["keys"]["#embedding"]["float_list"]["vector_index"]["config"][
                "space"
            ]
            == "cosine",
            "Wrong distance space",
        )
        origin = read(BASE / pool / "origin.json")
        sha(BASE / pool / "origin.json")
        expected_origin = {
            "embedding_model": "BAAI/bge-m3",
            "embedding_revision": "5617a9f61b028005a4858fdac845db406aefb181",
            "hnsw:space": "cosine",
            "kb_version": sha(CORE / f"kb/{pool}/records.jsonl")[:16],
            "plan_sha256": sha(BASE / "plan.json"),
        }
        metadata = {
            k: scalar(v)
            for k, *v in con.execute(
                "SELECT key,str_value,int_value,float_value,bool_value FROM collection_metadata WHERE collection_id=?",
                (cid,),
            )
        }
        require(origin == metadata == expected_origin, "Stored index origin differs")
        stored = con.execute("SELECT id,embedding_id FROM embeddings").fetchall()
        segments = con.execute("SELECT id,scope,collection FROM segments").fetchall()
        require(
            len(segments) == 2
            and {s[1] for s in segments} == {"METADATA", "VECTOR"}
            and all((s[2] == cid for s in segments)),
            "Wrong index segments",
        )
        metadata_segment = next((s[0] for s in segments if s[1] == "METADATA"))
        require(
            con.execute("SELECT DISTINCT segment_id FROM embeddings").fetchall()
            == [(metadata_segment,)],
            "Stored embeddings belong to different segment",
        )
        require(
            len(stored) == len(records)
            and {x[1] for x in stored} == {r["id"] for r in records},
            "Stored record IDs differ",
        )
        known = {r["id"]: r for r in records}
        byid = {i: known[rid] for i, rid in stored}
        actual = {i: {} for i in byid}
        for i, k, *v in con.execute(
            "SELECT id,key,string_value,int_value,float_value,bool_value FROM embedding_metadata"
        ):
            require(i in actual, "Orphan stored metadata")
            actual[i][k] = scalar(v)
        require(
            con.execute("SELECT count(*) FROM embedding_metadata_array").fetchone()[0]
            == 0,
            "Unexpected array metadata",
        )
        for i, r in byid.items():
            require(
                actual[i] == {**ns["_flatten_meta"](r), "chroma:document": r["text"]},
                f"Current index text/annotation differs: {r['id']}",
            )
    require(before == index_snapshot(pool), "Read-only audit changed index files")
    return {
        "records": len(records),
        "all_documents_and_annotations_exact": True,
        "integrity_check": "ok",
        "origin_and_cosine_space_verified": True,
        "physical_hashes_unchanged_by_audit": True,
        "foreign_key_check_raw": foreign,
        "foreign_key_qualification": "Two segments reference nonexistent singular collection table in installed Chroma migration; actual collections references explicitly checked, zero orphans. No other foreign-key failures.",
        "segment_schema": segment_schema,
        "installed_migration_sha256": sha(migration),
    }


def main():
    require(
        not OUT.exists() and (not TABLE.exists()) and (not RECEIPT.exists()),
        "Preserve completed review",
    )
    os.chdir(ROOT)
    from analysis.common import prompt as prompt

    plan = read(BASE / "plan.json")
    core = read(CORE / "plan.json")
    require(
        sha(BASE / "plan.json")
        == "c86b65e3375fd8f32b11455a79c9bfb738f1de9ab938e3d726c66ae855480d35",
        "Retrieval plan differs",
    )
    require(plan["core_sha256"] == sha(CORE / "plan.json"), "Core plan differs")
    for mapping in (
        plan["input_hashes"],
        plan["generated_hashes"],
        core["input_hashes"],
        core["generated_hashes"],
    ):
        verify(mapping)
    for f in [
        Path(__file__),
        ROOT / "analysis/common/retrieval_reference.py",
        ROOT / "tests/common/test_retrieval_review.py",
        ROOT / "analysis/common/stopped_retrieval_review.py",
        ROOT
        / "Thesis_Log/astra_result/evidence/core_retrieval_check_attempt1_20260915_v2.json",
    ]:
        sha(f)
    print("Frozen plans, source and generated hashes verified.", flush=True)
    full = rows(CORE / "kb/full/records.jsonl")
    dev = rows(CORE / "dev_inputs.jsonl")
    small = rows(CORE / "kb/small/records.jsonl")
    byid = {r["id"]: r for r in full}
    require(
        len(full) == len(byid) == 71131
        and len(small) == len({r["id"] for r in small}) == 340,
        "Wrong KB counts/duplicate IDs",
    )
    require(
        all((byid[r["id"]] == r for r in small)), "Small is not an exact full subset"
    )
    require(
        [r for r in full if r["kind"] != "example"]
        == [r for r in small if r["kind"] != "example"],
        "Abstract contents/order differ",
    )
    require(
        len(dev) == len({r["key"] for r in dev}) == 1295,
        "Wrong DEV count/duplicate IDs",
    )
    require(
        len({r["text"] for r in dev}) == len(dev),
        "Duplicate query text would alias cached-provider entries",
    )
    replay = inspect_replays(dev)
    print(
        "Both full replay logs verified, including original failure and probe history.",
        flush=True,
    )
    vector_checks = {
        "records": check_vectors(
            "full_vectors.npy", "record_vector_sources.jsonl", full, "id"
        ),
        "queries": check_vectors(
            "query_vectors.npy", "query_vector_sources.jsonl", dev, "key"
        ),
    }
    print(
        "All 72,426 vector rows checked bitwise against identified source cache rows.",
        flush=True,
    )
    sqlite_checks = {
        pool: inspect_sqlite(pool, rr)
        for pool, rr in [("small", small), ("full", full)]
    }
    print(
        "Both current SQLite indexes verified read-only, every document and annotation.",
        flush=True,
    )
    summaries = {}
    saved = {}
    for pool in ("small", "full"):
        known = {r["id"]: r for r in rows(CORE / f"kb/{pool}/records.jsonl")}
        items = rows(BASE / pool / "prompts.jsonl")
        saved[pool] = {x["key"]: x for x in items}
        require(len(saved[pool]) == len(items) == 1295, "Wrong prompt counts")
        require(
            [x["key"] for x in items] == [x["key"] for x in dev], "Wrong prompt order"
        )
        first = read(BASE / pool / "complete.json")
        sha(BASE / pool / "complete.json")
        require(
            first["plan_sha256"] == sha(BASE / "plan.json")
            and first["pool"] == pool
            and (first["records"] == len(known))
            and (first["prompts"] == 1295),
            "Wrong first-pass receipt",
        )
        require(
            first["prompt_sha256"] == sha(BASE / pool / "prompts.jsonl"),
            "First-pass prompts changed",
        )
        if pool == "small":
            cold = read(BASE / pool / "cold_check.json")
            sha(BASE / pool / "cold_check.json")
            require(
                cold["all_prompts_retrieved_again"]
                and cold["prompt_sha256"] == sha(BASE / pool / "prompts.jsonl"),
                "Small cold replay incomplete",
            )
            require(
                {k: v for k, v in cold.items() if k != "all_prompts_retrieved_again"}
                == first,
                "Small cold replay receipt differs",
            )
        stats = {}
        used = defaultdict(set)
        lengths = defaultdict(list)
        for x, r in zip(items, dev):
            require(
                set(x)
                == set(r)
                | {
                    "pool",
                    "condition",
                    "context",
                    "context_sha256",
                    "messages",
                    "messages_sha256",
                    "retrieved",
                    "plan_sha256",
                },
                "Unexpected prompt fields",
            )
            require(
                x["pool"] == pool
                and x["condition"] == "rag_" + pool
                and (x["plan_sha256"] == sha(BASE / "plan.json")),
                "Wrong prompt condition/plan",
            )
            require(all((x[k] == r[k] for k in r)), "Input identity/text/scope differs")
            ctx = x["context"]
            examples = [h for _, rr in ctx["example_groups"] for h in rr]
            require(
                len(examples) == len({h["id"] for h in examples}) == 5,
                "Wrong examples count",
            )
            require(
                len(ctx["definitions"]) == len(ctx["guidelines"]) == 2,
                "Wrong abstract budget",
            )
            for bucket, rr in [
                ("definition", ctx["definitions"]),
                ("guideline", ctx["guidelines"]),
                ("example", examples),
            ]:
                for h in rr:
                    k = known[h["id"]]
                    expected = {
                        **{
                            name: k[name]
                            for name in ("kind", "lang", "source", "dimension", "label")
                        },
                        **k["meta"],
                    }
                    require(
                        k["text"] == h["text"] and k["kind"] == bucket,
                        "Delivered content differs",
                    )
                    require(
                        {
                            name: value
                            for name, value in h["meta"].items()
                            if name != "_ranks"
                        }
                        == expected,
                        "Delivered annotation differs",
                    )
                    if bucket == "example":
                        require(k["lang"] == r["lang"], "Wrong example language")
            rendered = prompt.render_messages(r["text"], r["lang"], ctx, "corrected")
            require(
                rendered == x["messages"]
                and signature(rendered) == x["messages_sha256"],
                "Messages do not reproduce",
            )
            require(
                signature(ctx) == x["context_sha256"]
                and prompt.hit_ids(ctx) == x["retrieved"],
                "Context stamps differ",
            )
            for scope in r["scopes"]:
                name = scope["subset"]
                c = stats.setdefault(name, Counter())
                c["inputs"] += 1
                lengths[name].append(sum((len(m["text"]) for m in rendered)))
                for h in examples:
                    m = h["meta"]
                    used[name].add(h["id"])
                    c["example_slots"] += 1
                    c[f"gate_{m.get('gate')}"] += 1
                    for f in ("target_groups", "hate_types", "severity", "legal"):
                        c[f + "_annotated"] += m.get(f) is not None
        summaries[pool] = {
            k: {
                **dict(v),
                "unique_examples": len(used[k]),
                "prompt_chars_min": min(lengths[k]),
                "prompt_chars_max": max(lengths[k]),
                "prompt_chars_mean": sum(lengths[k]) / len(lengths[k]),
            }
            for k, v in stats.items()
        }
        require(
            {k: dict(v) for k, v in stats.items()} == first["summary"],
            "Recomputed coverage differs from first-pass summary",
        )
    comparisons = Counter()
    changed_abstract = []
    for r in dev:
        a, b = (saved[p][r["key"]] for p in ("small", "full"))
        comparisons["inputs"] += 1
        comparisons["messages_equal"] += a["messages"] == b["messages"]
        for part in ("definitions", "guidelines"):
            same = a["context"][part] == b["context"][part]
            comparisons[part + "_same_exact_context"] += same
            if not same:
                changed_abstract.append({"key": r["key"], "kind": part})
        ids = lambda x: [
            h["id"] for _, rr in x["context"]["example_groups"] for h in rr
        ]
        comparisons["examples_same_ordered_ids"] += ids(a) == ids(b)
        comparisons["examples_same_id_set"] += set(ids(a)) == set(ids(b))
        comparisons["system_message_equal"] += a["messages"][0] == b["messages"][0]
    history = {}
    historical_changed = []
    prior = ROOT / "experiments/astra/german_kb_comparison_20260913_v1"
    for pool, oldname in [("small", "c_canonical"), ("full", "d_full")]:
        p = prior / f"prompts_{oldname}.jsonl"
        require(
            sha(p) == read(prior / "setup_complete.json")["hashes"][rel(p)],
            "Historical German prompt hash differs",
        )
        old = {x["item_id"]: x for x in rows(p)}
        c = Counter()
        for r in dev:
            if r["lang"] != "de":
                continue
            a = saved[pool][r["key"]]
            b = old[r["item_id"]]
            require(a["text"] == b["text"], "Historical German input text differs")
            c["inputs"] += 1
            c["same_messages"] += a["messages"] == b["messages"]
            c["same_context"] += a["context"] == b["context"]
            ids = lambda x: [
                h["id"] for _, rr in x["context"]["example_groups"] for h in rr
            ]
            c["same_example_set"] += set(ids(a)) == set(ids(b))
            c["same_example_order"] += ids(a) == ids(b)
            if a["messages"] != b["messages"]:
                historical_changed.append(
                    {
                        "pool": pool,
                        "key": r["key"],
                        "same_example_set": set(ids(a)) == set(ids(b)),
                        "same_example_order": ids(a) == ids(b),
                    }
                )
        oldpool = "canonical_small" if pool == "small" else "canonical_full"
        oldrecords = [
            x
            for x in rows(prior / f"kb/{oldpool}/records.jsonl")
            if x["kind"] == "example" and x["lang"] == "de"
        ]
        newrecords = [
            x
            for x in rows(CORE / f"kb/{pool}/records.jsonl")
            if x["kind"] == "example" and x["lang"] == "de"
        ]
        require(
            oldrecords == newrecords, "Historical German pool content/order changed"
        )
        history[pool] = {**dict(c), "same_german_record_order_and_content": True}
    physical = {pool: index_snapshot(pool) for pool in ("small", "full")}
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_retrieval_prompts_audited_api_execution_not_prepared",
        "plan_sha256": sha(BASE / "plan.json"),
        "vector_checks": vector_checks,
        "replay_review": replay,
        "current_sqlite_checks": sqlite_checks,
        "physical_index_hashes": physical,
        "acceptance_amendment": {
            "version": 2,
            "v1_checker_unexecuted_preserved": True,
            "original_overall_complete_marker_absent": True,
            "decision": "Accept the unchanged first-pass prompt files as frozen inputs for preparation of the next classifier execution stage.",
            "basis": "Two complete exact full replays, complete small cold replay, original-failure/probe review, independent vectors/source identity, current stored texts/labels and every rendered prompt.",
            "limits": "Does not certify universal exact live retrieval; does not authorize silent prompt regeneration or reuse of historical predictions. No classifier batch is ready yet.",
        },
        "audit_development_note": "First v2 offline attempt stopped on raw SQLite foreign_key_check. Source snapshot and failure note preserved; no final artifacts were emitted. Installed Chroma migration references singular collection rather than collections. Final audit records the two expected failures per index and explicitly checks actual references and all stored documents/annotations.",
        "index_storage_vector_max_abs_difference": {
            pool: read(BASE / pool / "complete.json")["index_vector_max_abs_difference"]
            for pool in ("small", "full")
        },
        "prompt_summaries": summaries,
        "pool_comparisons": dict(comparisons),
        "changed_abstract_contexts": changed_abstract,
        "historical_german_prompt_comparison": history,
        "historical_german_changed_prompts": historical_changed,
        "scope": "This offline audit reconstructed every vector mapping and saved RAG prompt and reviewed user-run replay evidence. Read all current SQLite documents and annotations using immutable read-only connections. No retrieval rerun, API/model forward pass, classifier predictions, outcome scoring, Priority 3 or LaTeX writing.",
        "caveats": "Cached query vectors use FP32 batch-4 outputs, not new individual forwards. Approximate Chroma indexes and combined pool can change rankings; preserve actual prompts. Identical historical German prompts alone do not authorize silent prediction reuse or establish contemporaneous service behaviour. API logger/execution protocol remain separate.",
        "verified_hashes": dict(sorted(HASHES.items())),
    }
    with OUT.open("x", encoding="utf8", newline="\n") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    doc = [
        "# Bilingual core retrieval audit — 2026-09-15 v2",
        "",
        "**Frozen RAG inputs audited; API execution preparation is next. No new classifier accuracy results.**",
        "",
        "RAG needs a searchable KB and a saved prompt for each input. This stage prepares those inputs for the next SQ2/baseline comparison, with relevance to the SQ1 KB-size design. It does not fine-tune an encoder or ask Qwen to classify.",
        "",
        "Two isolated indexes contain 71,131 full records and 340 small records, including the same 40 definitions/guidelines. All 71,131 record vectors and 1,295 query vectors equal identified source-cache rows bit for bit; original text/split/reference identities and unit norms were checked. All 2,590 RAG prompts use five distinct language-matched examples, two definitions and two guidelines; every delivered text, annotation, saved signature and rendered message reproduces. Small is an exact subset of full.",
        "",
        "## Replay and database findings",
        "",
        "Both user-run full replays contain all 1,295 inputs in the original order, with zero differences in complete entries, contexts, selected IDs or classifier messages. Their comparison files have identical SHA-256 hashes. The earlier small cold replay matched all 1,295 inputs. Receipt/source hashes, log flags, actual-content hashes, roster order and absence of additional replay attempts were checked independently.",
        "",
        "The original strict full replay still stopped at input 1,034 (`de/detox-1389930895067467778`), after 1,033 exact matches, without saving the differing entry. A subsequent three-input recorder probe captured a single rank-provenance change on this key: example `ex-detox-1389917486057988096` gained `dense:10` alongside `bm25:1`. Selected examples and actual messages remained identical. The earlier unrecorded failure cannot conclusively be attributed to this same change. Both observations remain part of the evidence.",
        "",
        "Only `chroma.sqlite3` changed physically during each probe/replay; all five vector-index files stayed identical. Snapshots join exactly between passes, and the current full index equals the final snapshot. Both current SQLite databases pass integrity checks; every stored document, scalar annotation, collection origin and cosine configuration matches the frozen source. The audit itself leaves both indexes unchanged. No pre-reopen SQLite copy was saved, so the exact internal database write is not reconstructed; logical content checks establish current record integrity. Stored vector maximum absolute differences were 1.49e-8 (small) and 2.98e-8 (full), within the 2e-7 cosine-storage tolerance.",
        "",
        "The first v2 offline attempt stopped on SQLite foreign-key checking before emitting any completed report. Investigation found two flags in each index caused by the installed Chroma migration referencing a nonexistent singular `collection` table, although the real table is `collections`. The source snapshot and failure note remain preserved. The final audit records these exact flags, checks the installed migration against database migration history, verifies both segment references against the real table (zero orphans), and rejects any other foreign-key finding. No Chroma schema or package was modified. Five reviewer corruption tests plus the seven preparation and four recorder tests passed (16 total).",
        "",
        "This v2 review accepts the unchanged first-pass prompt files for the next execution-preparation stage. It does not claim all future live retrieval is exactly deterministic. The unexecuted v1 review draft remains intact, and neither missing original `complete.json` nor `full/cold_check.json` is fabricated. A separate `review_complete_v2.json` identifies this qualified acceptance. Future classifier runs must consume the exact frozen messages and verify their hashes.",
        "",
        "No fresh embedding run, GPU training, classifier/API call or outcome scoring was needed. Original indexes and results remain intact.",
        "",
        "## Retrieved annotation coverage",
        "",
        "Counts are retrieved slots with observed labels, not unique training examples or classifier accuracy. Missing annotation is not the same as an annotated empty list.",
        "",
        "| Pool | DEV roster | Inputs | Unique examples | Gate true / false / missing slots | Target / type / severity / legal annotated slots |",
        "|---|---|---:|---:|---|---|",
    ]
    for pool, stats in summaries.items():
        for name, c in stats.items():
            doc.append(
                f"| {pool} | {name} | {c['inputs']} | {c['unique_examples']} | {c.get('gate_True', 0)} / {c.get('gate_False', 0)} / {c.get('gate_None', 0)} | {c.get('target_groups_annotated', 0)} / {c.get('hate_types_annotated', 0)} / {c.get('severity_annotated', 0)} / {c.get('legal_annotated', 0)} |"
            )
    doc += [
        "",
        "## Comparability checks",
        "",
        f"Across 1,295 paired inputs: `{dict(comparisons)}`. Changed abstract contexts: {len(changed_abstract)}. The evidence lists any changed keys.",
        "",
        "The full KB changes the example mix substantially: on the 299-input English gate roster, non-hate slots increase from 198/1,495 (13.2%) to 996/1,495 (66.6%). This could matter for model behaviour but is not a measured accuracy gain or causal explanation. Some dimensions have fewer explicitly labelled examples despite the larger search pool; annotation-eligible few-shot controls in the existing plan remain relevant. Do not tune retrieval in response to these descriptive counts.",
        "",
        f"Historical German message equivalence: `{history}`. These counts are a reuse diagnostic only. The frozen core plan still calls for new contemporaneous comparisons unless a later explicit, documented reuse amendment is made.",
        "",
        "The original hybrid/MMR ranking code and corrected renderer were preserved. A cached vector provider replaces neural query encoding; exact text matching is required and unknown inputs fail rather than invoking a model. Both pools use the same query vectors. Cached batched query arithmetic can differ slightly from old individual inference; this is recorded as an implementation/provenance qualification.",
        "",
        "## Next bounded stage",
        "",
        "Prepare isolated concurrency-safe API logging, explicit retry/repair caps, exact failure/scoring rules and staged resumable classifier execution. The 96,822-response programme remains a programme budget, not a ready batch. No user terminal action is required for this completed retrieval stage. Priority 3, adaptation and LaTeX stay parked.",
        "",
        f"Evidence: [review JSON](evidence/{OUT.name}); plan SHA-256 `{report['plan_sha256']}`; evidence SHA-256 `{sha(OUT)}`.",
    ]
    with TABLE.open("x", encoding="utf8", newline="\n") as f:
        f.write("\n".join(doc) + "\n")
    stamp = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": report["status"],
        "plan_sha256": report["plan_sha256"],
        "evidence": rel(OUT),
        "evidence_sha256": sha(OUT),
        "report": rel(TABLE),
        "report_sha256": sha(TABLE),
        "prompt_hashes": {
            pool: sha(BASE / pool / "prompts.jsonl") for pool in ("small", "full")
        },
        "acceptance_amendment": report["acceptance_amendment"],
        "next": "Prepare isolated safe API logging and a frozen staged execution/scoring/failure contract. No classifier batch ready; Priority 3 and LaTeX remain parked.",
    }
    with RECEIPT.open("x", encoding="utf8", newline="\n") as f:
        json.dump(stamp, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {
                "status": report["status"],
                "evidence_sha256": sha(OUT),
                "vectors": vector_checks,
                "comparisons": dict(comparisons),
                "historical_german": history,
                "verified_files": len(report["verified_hashes"]),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

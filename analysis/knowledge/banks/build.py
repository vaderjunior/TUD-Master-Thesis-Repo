"""Banks build."""

from __future__ import annotations
from src.hsrag.paths import PROJECT_ROOT
import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import unicodedata

ROOT = PROJECT_ROOT
RUN_ID = "sq1_kb_size_20260911_v1"
BASE = ROOT / "experiments" / "astra" / RUN_ID
PLAN = BASE / "plan.json"
SCOPES = {"gate": "en_dev_eval_main", "types": "en_dev_eval_sq3_types"}
POOLS = ("small", "all")
GOLD = ("gate", "target_groups", "hate_types", "severity", "legal")
PACKAGES = (
    "torch",
    "transformers",
    "sentence-transformers",
    "chromadb",
    "tokenizers",
    "numpy",
    "pandas",
    "pyarrow",
    "rank-bm25",
    "pydantic",
)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def owned(path):
    path = Path(path).resolve()
    require(
        path.is_relative_to(BASE.resolve()) and path != BASE.resolve(),
        f"Refusing a write outside this experiment: {path}",
    )
    return path


def json_bytes(value):
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def write_once(path, data):
    """Idempotent identical writes, never replace different frozen evidence."""
    path = owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(
            path.read_bytes() == data, f"Existing file differs; preserving it: {path}"
        )
    else:
        with path.open("xb") as f:
            f.write(data)


def status_write(value):
    """Only the live progress file is replaceable; evidence remains append-only."""
    path = owned(BASE / "batch_status.json")
    temp = owned(BASE / "batch_status.json.tmp")
    temp.write_bytes(json_bytes({"updated_utc": now(), **value}))
    temp.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_lines(path):
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def norm(text):
    s = unicodedata.normalize("NFKC", str(text)).casefold()
    return re.sub("\\s+", " ", s).strip(" \t\r\n\"'`«»“”‘’.,:;!?-–—…")


def canonical_text(text):
    return " ".join(unicodedata.normalize("NFKC", str(text)).casefold().split())


def clean(value):
    if value is None:
        return None
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, float) and value != value:
        return None
    return value


def example(row):
    return {
        "id": f"ex-{row['id']}",
        "kind": "example",
        "dimension": None,
        "label": None,
        "lang": "en",
        "text": row["text"],
        "source": f"{row['source']}-train",
        "meta": {
            "gate": clean(row.get("gate")),
            "target_groups": clean(row.get("target_groups")),
            "hate_types": clean(row.get("hate_types")),
            "severity": clean(row.get("severity")),
            "illustrative_only": False,
        },
    }


def assemble(original, rows, excluded, heldout):
    """Make nested English pools, preserving common records exactly."""
    old_en = {
        r["id"]: r for r in original if r["kind"] == "example" and r["lang"] == "en"
    }
    unchanged = [r for r in original if r["id"] not in old_en]
    ids, texts, kept, removed = (set(), set(), [], [])
    excluded = {str(i).removeprefix("ex-") for i in excluded}
    for row in rows:
        require(row["lang"] == "en", "English TRAIN contains another language")
        require(
            isinstance(row["text"], str) and norm(row["text"]),
            "Empty/non-text TRAIN row",
        )
        rid, nt = (str(row["id"]), norm(row["text"]))
        require(rid not in ids, f"Duplicate training ID: {rid}")
        ids.add(rid)
        if not any((clean(row.get(g)) is not None for g in GOLD)):
            removed.append({"id": rid, "reason": "no_gold"})
            continue
        if rid in excluded or nt in heldout:
            removed.append(
                {
                    "id": rid,
                    "reason": (
                        "known_near_duplicate"
                        if rid in excluded
                        else "normalized_heldout_overlap"
                    ),
                }
            )
            continue
        canonical = canonical_text(row["text"])
        require(
            canonical not in texts,
            f"TRAIN text collision needs an explicit policy: {rid}",
        )
        texts.add(canonical)
        rec = example(row)
        if rec["id"] in old_en:
            old = old_en[rec["id"]]
            require(
                rec == old, f"Original KB and TRAIN labels/text differ: {rec['id']}"
            )
            rec = old
        kept.append(rec)
    require(
        {i.removeprefix("ex-") for i in old_en} <= ids,
        "Original KB contains English IDs outside TRAIN",
    )
    keep_ids = {r["id"] for r in kept}
    small = [r for r in original if r["id"] not in old_en or r["id"] in keep_ids]
    full = unchanged + kept
    for pool in (small, full):
        require(len({r["id"] for r in pool}) == len(pool), "Duplicate output IDs")
        require(
            len({canonical_text(r["text"]) for r in pool}) == len(pool),
            "Duplicate normalized KB texts",
        )
    require(
        {r["id"] for r in small} <= {r["id"] for r in full}, "Small pool is not nested"
    )
    return ({"small": small, "all": full}, removed)


def pool_summary(records):
    en = [r for r in records if r["kind"] == "example" and r["lang"] == "en"]
    return {
        "total_records": len(records),
        "english_examples": len(en),
        "english_sources": dict(Counter((r["source"] for r in en))),
        "english_gate": dict(Counter((str(r["meta"].get("gate")) for r in en))),
        "english_annotated": {
            g: sum((r["meta"].get(g) is not None for r in en)) for g in GOLD
        },
    }


def environment():
    return {
        "python": sys.version.split()[0],
        "packages": {p: metadata.version(p) for p in PACKAGES},
    }


def cache_info():
    hub = Path(
        os.environ.get("HF_HUB_CACHE")
        or os.environ.get("HUGGINGFACE_HUB_CACHE")
        or str(
            Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface")))
            / "hub"
        )
    )
    repo = hub / "models--BAAI--bge-m3"
    ref = repo / "refs/main"
    require(
        ref.exists(), "BGE-M3 must already be available in the local Hugging Face cache"
    )
    revision = ref.read_text().strip()
    require(re.fullmatch("[a-f0-9]{40}", revision), "Unexpected BGE-M3 cache revision")
    snapshot = repo / "snapshots" / revision
    require(snapshot.is_dir(), "BGE-M3 snapshot is missing")
    files = {
        str(p.relative_to(snapshot)).replace("\\", "/"): digest(p)
        for p in sorted(snapshot.rglob("*"))
        if p.is_file()
    }
    require(
        any((n.endswith((".safetensors", ".bin")) for n in files)),
        "BGE-M3 cached weights missing",
    )
    return {
        "hub": str(hub.resolve()),
        "revision": revision,
        "snapshot": str(snapshot.resolve()),
        "files": files,
    }


def code_stamp():
    h = hashlib.sha256()
    for p in sorted(Path("src/hsrag").rglob("*.py")):
        h.update(p.as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:8]


def prepare():
    if PLAN.exists():
        validate_plan()
        print(f"Already prepared and unchanged: {PLAN}")
        return
    import pandas as pd
    import yaml

    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    require(cfg["kb"]["embedding_model"] == "BAAI/bge-m3", "Unexpected embedding model")
    retrieval = dict(cfg["retrieval"])
    require(
        all(
            (
                retrieval.get(k) == v
                for k, v in {
                    "strategy": "hybrid",
                    "k_definitions": 2,
                    "k_guidelines": 2,
                    "k_examples": 5,
                    "k_examples_feedback": 0,
                    "use_mmr": True,
                    "use_reranker": False,
                    "k_definitions_mode": "flat",
                }.items()
            )
        ),
        "Frozen retrieval settings differ",
    )
    originals = ROOT / "kb/records.jsonl"
    train = ROOT / "data/processed/en_train.parquet"
    exc_path = ROOT / "experiments/leakage_train_exclusions.json"
    report_path = ROOT / "experiments/leakage_report_en_train.json"
    exc, report = (read_json(exc_path), read_json(report_path))
    require(
        exc["threshold"] == 0.95 and report["near"] == 0.95 and (not report["smoke"]),
        "Expected the existing full 0.95 exclusion policy",
    )
    eval_files = sorted((ROOT / "data/processed").glob("*_eval*.parquet"))
    for name in SCOPES.values():
        n = len(
            pd.read_parquet(ROOT / "data/processed" / f"{name}.parquet", columns=["id"])
        )
        require(
            report["subsets"][name]["n_items"] == n,
            f"Saved leakage report coverage differs: {name}",
        )
    split_files = [
        ROOT / "data/processed" / f"{s}.parquet"
        for s in (
            "en_dev",
            "en_test",
            "de_dev",
            "de_test",
            "de_legal_dev",
            "de_legal_test",
        )
    ]
    held_files = sorted(set(eval_files + split_files))
    held = set()
    for p in held_files:
        held.update((norm(t) for t in pd.read_parquet(p, columns=["text"])["text"]))
    pools, removed = assemble(
        read_lines(originals),
        pd.read_parquet(train).to_dict("records"),
        exc["ids"]["en"],
        held,
    )
    summaries = {k: pool_summary(v) for k, v in pools.items()}
    require(
        summaries["small"]["english_examples"] == 148
        and summaries["all"]["english_examples"] == 56109,
        f"Eligibility changed from preparation audit; inspect before freezing: {summaries}",
    )
    manifests, runs, generated = ({}, [], [])
    for pool, recs in pools.items():
        path = BASE / "kb" / pool / "records.jsonl"
        content = "".join(
            (json.dumps(r, ensure_ascii=False) + "\n" for r in recs)
        ).encode("utf-8")
        write_once(path, content)
        generated.append(path)
    for rep in (1, 2, 3):
        for scope, subset in SCOPES.items():
            for pool in POOLS if rep % 2 else tuple(reversed(POOLS)):
                name = f"astra_{RUN_ID}_{scope}_{pool}_r{rep}"
                mf = {
                    "name": name,
                    "subset": subset,
                    "arms": ["rag"],
                    "provider": "peasec",
                    "pinned_model": "Qwen/Qwen3-VL-32B-Instruct",
                    "allow_fallback": False,
                    "temperature": 1.0,
                    "n_votes": 3,
                    "max_repair_retries": 2,
                    "workers": 8,
                    "limit": None,
                    "seed": 42,
                    "retrieval": retrieval,
                    "records_path": rel(BASE / "kb" / pool / "records.jsonl"),
                    "chroma_path": rel(BASE / "kb" / pool / "chroma"),
                    "gate_mapping": "both",
                    "hate_type_scoring": "multilabel",
                    "description": f"Post-hoc SQ1 KB policy sensitivity, {pool} eligible English TRAIN pool; fixed five examples, original hybrid settings, no feedback. Primary gate macro-F1/FP/FN on gate scope; hate-type macro-F1 on types scope. Three replicates per pool, paired item analysis; report both directions. No configuration selection or new test-set evaluation.",
                }
                path = BASE / "manifests" / f"{name}.yaml"
                write_once(
                    path,
                    yaml.safe_dump(mf, sort_keys=False, allow_unicode=True).encode(
                        "utf-8"
                    ),
                )
                generated.append(path)
                manifests[name] = mf
                n = len(
                    pd.read_parquet(
                        ROOT / "data/processed" / f"{subset}.parquet", columns=["id"]
                    )
                )
                runs.append(
                    {
                        "name": name,
                        "scope": scope,
                        "pool": pool,
                        "replicate": rep,
                        "n_items": n,
                        "manifest": rel(path),
                        "output": rel(BASE / "results" / f"{name}_live.jsonl"),
                    }
                )
    audit = {
        "pool_counts": summaries,
        "removed_train_ids": removed,
        "heldout_files": [rel(p) for p in held_files],
        "normalization": "NFKC/casefold/whitespace/edge punctuation",
        "within_train_deduplication": "NFKC/casefold/whitespace; retain punctuation variants unless held-out overlap",
        "removed_by_reason": dict(Counter((r["reason"] for r in removed))),
        "near_duplicate_policy": "Reuse saved 0.95 exclusions; no new embedding leakage scan",
        "limitations": "Saved near-duplicate report has no input content hashes. Coverage counts verified, current inputs pinned here; historical report identity cannot be fully proven.",
    }
    audit_path = BASE / "preparation.json"
    write_once(audit_path, json_bytes(audit))
    generated.append(audit_path)
    inputs = [
        originals,
        train,
        exc_path,
        report_path,
        ROOT / "config/config.yaml",
        ROOT / "config/taxonomy.yaml",
        ROOT / "prompts/classify_v1.txt",
        ROOT / "scripts/run_comparison.py",
        ROOT / "scripts/score_predictions.py",
        Path(__file__),
        ROOT / "tests/knowledge/banks/test_build.py",
        ROOT / "scripts/run_bank_comparison.ps1",
    ] + held_files
    inputs += list((ROOT / "src/hsrag").rglob("*.py"))
    plan = {
        "run_id": RUN_ID,
        "created_utc": now(),
        "input_hashes": {rel(p): digest(p) for p in sorted(set(inputs))},
        "generated_hashes": {rel(p): digest(p) for p in generated},
        "environment": environment(),
        "embedding_cache": cache_info(),
        "pool_counts": summaries,
        "runs": runs,
        "manifests": manifests,
        "planned_calls": sum((r["n_items"] * 3 for r in runs)),
        "code_version": code_stamp(),
        "prompt_version": "classify_v1-" + digest(ROOT / "prompts/classify_v1.txt")[:8],
        "taxonomy_version": "taxonomy-" + digest(ROOT / "config/taxonomy.yaml")[:8],
        "embedding_batch_size": 4,
        "ingestion_chunk_size": 128,
    }
    write_once(PLAN, json_bytes(plan))
    print(
        json.dumps(
            {
                "plan": str(PLAN),
                "pools": summaries,
                "runs": len(runs),
                "planned_calls": plan["planned_calls"],
                "removed": len(removed),
            },
            indent=2,
        )
    )
    print("PREPARED ONLY. No model load, embeddings, retrieval or API calls.")


def validate_plan(check_cache=True):
    plan = read_json(PLAN)
    require(plan["run_id"] == RUN_ID, "Wrong experiment plan")
    for category in ("input_hashes", "generated_hashes"):
        for name, expected in plan[category].items():
            require(
                digest(ROOT / name) == expected,
                f"Frozen file changed: {name}. Do not mix versions; ask for review.",
            )
    require(
        environment() == plan["environment"],
        "Python/package versions changed since preparation",
    )
    if check_cache:
        require(
            cache_info() == plan["embedding_cache"],
            "BGE-M3 cache changed since preparation",
        )
    return plan


@contextmanager
def run_lock():
    import msvcrt

    p = owned(BASE / "launcher.lock")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a+b") as f:
        if f.tell() == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as e:
            raise RuntimeError(
                "Another copy of this batch is running. Keep only one terminal launcher."
            ) from e
        try:
            yield
        except Exception:
            status_write(
                {
                    "status": "stopped",
                    "reason": "Execution stopped; inspect the terminal and latest log.",
                }
            )
            raise
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def offline_env(plan):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_CACHE"] = plan["embedding_cache"]["hub"]
    os.environ["ANONYMIZED_TELEMETRY"] = "False"


def require_cuda():
    import torch

    require(
        torch.cuda.is_available(),
        "CUDA is unavailable. Enable the NVIDIA GPU, then rerun the same launcher.",
    )
    free, total = torch.cuda.mem_get_info()
    print(
        f"CUDA: {torch.cuda.get_device_name(0)}; free {free / 2 ** 30:.2f} / {total / 2 ** 30:.2f} GiB",
        flush=True,
    )
    require(
        shutil.disk_usage(BASE).free >= 3 * 2**30,
        "Less than 3 GiB disk free; make space outside the frozen results first",
    )


def check_service(plan):
    """Read the model roster when the user launches; no generation request."""
    import yaml
    from src.hsrag.client import make_client

    cfg = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))
    mf = next(iter(plan["manifests"].values()))
    api = cfg["api"]["peasec"]
    try:
        client = make_client(
            "peasec",
            [mf["pinned_model"]],
            allow_fallback=False,
            url_env=api["url_env"],
            token_env=api["token_env"],
        )
        available = client.list_models()
    except Exception:
        raise RuntimeError(
            "PEASEC model-list check failed before indexing. Check connection/service credentials, then retry."
        ) from None
    require(
        mf["pinned_model"] in available,
        "Pinned PEASEC model unavailable; no substitute will be used",
    )
    print(
        "PEASEC connection and pinned model availability checked. No generation yet.",
        flush=True,
    )


def index_identity(plan, pool):
    records = BASE / "kb" / pool / "records.jsonl"
    return {
        "run_id": RUN_ID,
        "records_sha256": digest(records),
        "embedding_model": "BAAI/bge-m3",
        "embedding_revision": plan["embedding_cache"]["revision"],
        "plan_sha256": digest(PLAN),
    }


def open_index(plan, pool):
    import chromadb

    identity = index_identity(plan, pool)
    parent = owned(BASE / "kb" / pool)
    path = owned(parent / "chroma")
    marker = parent / "index_origin.json"
    require(not path.exists() or marker.exists(), f"Unowned index exists: {path}")
    write_once(marker, json_bytes(identity))
    client = chromadb.PersistentClient(path=str(path))
    names = [c if isinstance(c, str) else c.name for c in client.list_collections()]
    require(set(names) <= {"hate_kb"}, "Unexpected collection in isolated index")
    if "hate_kb" in names:
        col = client.get_collection("hate_kb")
        for key, val in identity.items():
            require(
                col.metadata.get(key) == val, f"Index identity differs: {pool}/{key}"
            )
    else:
        col = client.create_collection(
            "hate_kb",
            metadata={
                "hnsw:space": "cosine",
                **identity,
                "kb_version": identity["records_sha256"][:16],
            },
        )
    return col


def build_indexes(plan):
    from sentence_transformers import SentenceTransformer
    from src.hsrag.kb import _flatten_meta

    require_cuda()
    model = None
    for pool in POOLS:
        records = read_lines(BASE / "kb" / pool / "records.jsonl")
        col = open_index(plan, pool)
        existing = set(col.get(include=[])["ids"])
        wanted = {r["id"] for r in records}
        require(existing <= wanted, "Unexpected IDs in isolated index")
        todo = [r for r in records if r["id"] not in existing]
        print(
            f"INDEX {pool}: {len(existing)}/{len(records)} already present; {len(todo)} to encode",
            flush=True,
        )
        if todo and model is None:
            model = SentenceTransformer(
                "BAAI/bge-m3",
                revision=plan["embedding_cache"]["revision"],
                local_files_only=True,
                device="cuda",
            )
        chunk = min(plan["ingestion_chunk_size"], col._client.get_max_batch_size())
        for start in range(0, len(todo), chunk):
            rows = todo[start : start + chunk]
            vecs = model.encode(
                [r["text"] for r in rows],
                normalize_embeddings=True,
                batch_size=plan["embedding_batch_size"],
                show_progress_bar=False,
            )
            col.add(
                ids=[r["id"] for r in rows],
                embeddings=vecs.tolist(),
                documents=[r["text"] for r in rows],
                metadatas=[_flatten_meta(r) for r in rows],
            )
            print(
                f"INDEX {pool}: {col.count()}/{len(records)} records stored", flush=True
            )
        require(
            set(col.get(include=[])["ids"]) == wanted, "Index membership incomplete"
        )
        for start in range(0, len(records), 512):
            rows = records[start : start + 512]
            got = col.get(
                ids=[r["id"] for r in rows], include=["documents", "metadatas"]
            )
            byid = {
                i: (d, m)
                for i, d, m in zip(got["ids"], got["documents"], got["metadatas"])
            }
            for r in rows:
                require(
                    byid[r["id"]] == (r["text"], _flatten_meta(r)),
                    f"Index payload differs: {r['id']}",
                )
        write_once(
            BASE / "kb" / pool / "index_complete.json",
            json_bytes({**index_identity(plan, pool), "n_records": len(records)}),
        )


def retrieval_check(plan):
    import pandas as pd
    from src.hsrag.retrieve import Retriever
    from src.hsrag.classify import _hit_ids
    from src.hsrag.prompt import PromptContext

    require_cuda()
    reference = None
    for pool in POOLS:
        mf = next((v for k, v in plan["manifests"].items() if f"_{pool}_r" in k))
        retriever = Retriever(
            ROOT / mf["chroma_path"],
            ROOT / mf["records_path"],
            "BAAI/bge-m3",
            dict(mf["retrieval"]),
        )
        known = {r["id"]: r for r in read_lines(ROOT / mf["records_path"])}
        observed = {}
        for scope, subset in SCOPES.items():
            df = pd.read_parquet(ROOT / "data/processed" / f"{subset}.parquet")
            for i, row in enumerate(df.itertuples(index=False), 1):
                result = retriever.retrieve(row.text, row.lang)
                ids = _hit_ids(PromptContext.from_retrieval(result))
                require(
                    len(ids.get("Labelled examples", [])) == 5,
                    f"Expected exactly five examples: {row.id}",
                )
                require(
                    set(ids) == {"definitions", "guidelines", "Labelled examples"},
                    "Unexpected prompt group",
                )
                require(
                    len(ids["definitions"]) <= 2 and len(ids["guidelines"]) <= 2,
                    "Abstract budget exceeded",
                )
                for rid in ids["Labelled examples"]:
                    require(
                        known[rid]["lang"] == "en" and known[rid]["kind"] == "example",
                        "Wrong example kind/language",
                    )
                observed[f"{subset}/{row.id}"] = ids
                if i % 100 == 0 or i == len(df):
                    print(f"RETRIEVAL {pool} {scope}: {i}/{len(df)}", flush=True)
        if reference is None:
            reference = observed
        else:
            mismatches = [
                key
                for key in observed
                if any(
                    (
                        observed[key][b] != reference[key][b]
                        for b in ("definitions", "guidelines")
                    )
                )
            ]
            if mismatches:
                write_once(
                    BASE / "retrieval_abstract_mismatches.json", json_bytes(mismatches)
                )
            require(
                not mismatches,
                "Abstract retrieval changed across pools. Stopped before API calls; ask for review.",
            )
        write_once(BASE / f"retrieval_{pool}.json", json_bytes(observed))
        del retriever
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()
    write_once(
        BASE / "retrieval_complete.json",
        json_bytes(
            {
                "plan_sha256": digest(PLAN),
                "hashes": {
                    pool: digest(BASE / f"retrieval_{pool}.json") for pool in POOLS
                },
                "abstract_context_equal": True,
                "n_inputs": len(reference),
            }
        ),
    )


def item_text_hash(text):
    return hashlib.sha256(" ".join(str(text).split()).encode()).hexdigest()[:16]


def verify_result_rows(rows, mf, expected, context, stamps):
    headers = [r for r in rows if "_manifest" in r]
    require(len(headers) == 1, "Result needs exactly one manifest header")
    header = headers[0]["_manifest"]
    for key in (
        "name",
        "subset",
        "arms",
        "provider",
        "pinned_model",
        "temperature",
        "n_votes",
        "workers",
        "max_repair_retries",
        "retrieval",
        "records_path",
        "chroma_path",
        "gate_mapping",
        "hate_type_scoring",
        "seed",
        "limit",
        "allow_fallback",
    ):
        require(header.get(key) == mf.get(key), f"Result manifest differs: {key}")
    require(
        header.get("kb_version") == stamps["kb_version"], "Result header KB differs"
    )
    seen, invalid, partial = (set(), 0, 0)
    for r in rows:
        if "_manifest" in r:
            continue
        key = r.get("item_id")
        require(
            key in expected and key not in seen, f"Unknown/duplicate result ID: {key}"
        )
        seen.add(key)
        require(
            r.get("text_hash") == expected[key] and r.get("lang") == "en",
            f"Input changed: {key}",
        )
        for field, val in {
            "arm": "rag",
            "active_model": mf["pinned_model"],
            "temperature": mf["temperature"],
            "workers": mf["workers"],
            "n_runs": mf["n_votes"],
            "retrieval_config": mf["retrieval"],
            **stamps,
        }.items():
            require(r.get(field) == val, f"Result provenance differs: {key}/{field}")
        require(
            r.get("retrieved") == context[f"{mf['subset']}/{key}"],
            f"Delivered context drifted: {key}",
        )
        require(
            len(r.get("raw_runs", [])) == 3 and len(r.get("run_predictions", [])) == 3,
            f"Raw votes incomplete: {key}",
        )
        require(
            r.get("n_valid") == sum((v is not None for v in r["run_predictions"])),
            f"Valid-vote count differs: {key}",
        )
        invalid += int(not r.get("result") or r.get("n_valid", 0) == 0)
        partial += int(r.get("n_valid", 0) < 3)
    return {
        "written": len(seen),
        "expected": len(expected),
        "missing": len(set(expected) - seen),
        "invalid_results": invalid,
        "items_with_fewer_than_three_valid_votes": partial,
    }


def check_output(plan, run):
    import pandas as pd

    out = ROOT / run["output"]
    if not out.exists():
        return {"written": 0, "expected": run["n_items"], "missing": run["n_items"]}
    mf = plan["manifests"][run["name"]]
    df = pd.read_parquet(ROOT / "data/processed" / f"{mf['subset']}.parquet")
    expected = {str(r.id): item_text_hash(r.text) for r in df.itertuples(index=False)}
    stamps = {
        k: plan[k] for k in ("code_version", "prompt_version", "taxonomy_version")
    }
    stamps["kb_version"] = digest(ROOT / mf["records_path"])[:16]
    return verify_result_rows(
        read_lines(out),
        mf,
        expected,
        read_json(BASE / f"retrieval_{run['pool']}.json"),
        stamps,
    )


def subprocess_logged(arguments, label):
    log = owned(BASE / "logs" / f"{label}_{time.time_ns()}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    command = [sys.executable, "-B", "-u", *arguments]
    print("RUN:", subprocess.list2cmdline(command), flush=True)
    with log.open("x", encoding="utf-8") as f:
        proc = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
        )
        try:
            for line in proc.stdout:
                print(line, end="", flush=True)
                f.write(line)
                f.flush()
            code = proc.wait()
        except BaseException:
            proc.terminate()
            proc.wait()
            raise
    require(
        code == 0,
        f"Process failed ({code}). Log: {log}. Resume with the same launcher after review.",
    )


def run_batch():
    with run_lock():
        plan = validate_plan()
        offline_env(plan)
        if (BASE / "batch_complete.json").exists():
            complete = read_json(BASE / "batch_complete.json")
            require(complete["plan_sha256"] == digest(PLAN), "Completion plan changed")
            for name, expected_hash in complete["result_hashes"].items():
                require(
                    digest(ROOT / name) == expected_hash,
                    f"Completed result changed: {name}",
                )
            print("This batch is already complete. No new API calls.")
            return
        require_cuda()
        check_service(plan)
        status_write({"status": "indexing", "planned_calls": plan["planned_calls"]})
        subprocess_logged(["-m", "analysis.knowledge.banks.build", "index"], "index")
        status_write({"status": "retrieval_checks"})
        if not (BASE / "retrieval_complete.json").exists():
            subprocess_logged(
                ["-m", "analysis.knowledge.banks.build", "retrieval"], "retrieval"
            )
        receipt = read_json(BASE / "retrieval_complete.json")
        require(
            receipt["plan_sha256"] == digest(PLAN)
            and receipt["abstract_context_equal"],
            "Retrieval receipt differs",
        )
        for pool in POOLS:
            require(
                digest(BASE / f"retrieval_{pool}.json") == receipt["hashes"][pool],
                "Saved retrieval changed",
            )
        completed = []
        for index, run in enumerate(plan["runs"], 1):
            validate_plan(check_cache=False)
            info = check_output(plan, run)
            print(
                f"BATCH {index}/{len(plan['runs'])}: {run['name']}; missing {info['missing']}",
                flush=True,
            )
            for attempt in range(2):
                if not info["missing"]:
                    break
                status_write(
                    {
                        "status": "classifying",
                        "run": run["name"],
                        "run_number": index,
                        "total_runs": len(plan["runs"]),
                        "attempt": attempt + 1,
                        "completed": completed,
                    }
                )
                subprocess_logged(
                    [
                        "-m",
                        "scripts.run_comparison",
                        "--manifest",
                        run["manifest"],
                        "--out",
                        run["output"],
                        "--yes",
                    ],
                    run["name"],
                )
                info = check_output(plan, run)
            require(
                info["missing"] == 0,
                f"Missing predictions remain in {run['name']}; batch paused. Rerun same launcher later.",
            )
            require(
                info["invalid_results"] == 0,
                f"All-invalid votes recorded in {run['name']}; preserved for review, batch paused.",
            )
            completed.append({"run": run["name"], **info})
        validate_plan()
        result_hashes = {
            run["output"]: digest(ROOT / run["output"]) for run in plan["runs"]
        }
        write_once(
            BASE / "batch_complete.json",
            json_bytes(
                {
                    "plan_sha256": digest(PLAN),
                    "completed_utc": now(),
                    "runs": completed,
                    "result_hashes": result_hashes,
                }
            ),
        )
        status_write({"status": "complete", "completed": completed})
        print(
            f"ALL {len(completed)} RUNS COMPLETE. Results: {BASE / 'results'}",
            flush=True,
        )
        print(
            "Tell Astra the batch finished. Statistical interpretation is the next separate step."
        )


def main():
    os.chdir(ROOT)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "stage", choices=("prepare", "validate", "status", "run", "index", "retrieval")
    )
    args = ap.parse_args()
    if args.stage == "prepare":
        prepare()
    elif args.stage == "validate":
        plan = validate_plan()
        print(
            f"Frozen preparation valid: {len(plan['runs'])} runs, {plan['planned_calls']} planned calls; no model/API work."
        )
    elif args.stage == "status":
        p = BASE / "batch_status.json"
        print(
            json.dumps(read_json(p), indent=2)
            if p.exists()
            else "Prepared; batch not started."
        )
    elif args.stage == "run":
        run_batch()
    else:
        plan = validate_plan()
        offline_env(plan)
        (build_indexes if args.stage == "index" else retrieval_check)(plan)


if __name__ == "__main__":
    main()

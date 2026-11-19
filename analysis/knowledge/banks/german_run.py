import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import FunctionType, SimpleNamespace
from analysis.knowledge.banks import german_prepare as p
from analysis.common import prompt as rendering

ROOT, BASE, PLAN = (p.ROOT, p.BASE, p.PLAN)
u = p.u


def validate(cache=True):
    plan = u.read_json(PLAN)
    p.require(plan["run_id"] == BASE.name, "Wrong experiment plan")
    for group in ("input_hashes", "generated_hashes"):
        for name, h in plan[group].items():
            p.require(u.digest(ROOT / name) == h, f"Frozen input changed: {name}")
    p.require(u.environment() == plan["environment"], "Environment changed")
    if cache:
        p.require(u.cache_info() == plan["embedding_cache"], "Embedding cache changed")
    return plan


def signature(obj):
    return hashlib.sha256(p.js(obj)).hexdigest()


def offline(plan):
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_CACHE"] = plan["embedding_cache"]["hub"]
    os.environ["ANONYMIZED_TELEMETRY"] = "False"


@contextmanager
def lock():
    import msvcrt

    path = p.owned(BASE / "launcher.lock")
    with path.open("a+b") as f:
        if path.stat().st_size == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError(
                "Another German comparison is running. Use one terminal."
            ) from None
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def check_setup():
    path = BASE / "setup_complete.json"
    if not path.exists():
        return False
    receipt = u.read_json(path)
    p.require(receipt["plan_sha256"] == u.digest(PLAN), "Setup plan changed")
    for name, h in receipt["hashes"].items():
        p.require(u.digest(ROOT / name) == h, f"Setup artifact changed: {name}")
    return True


def setup(plan):
    if check_setup():
        print("SETUP ALREADY COMPLETE: frozen contexts/prompts verified.", flush=True)
        return
    import gc
    import numpy as np
    import pandas as pd
    import torch
    import chromadb
    from sentence_transformers import SentenceTransformer
    from src.hsrag.kb import _flatten_meta
    from src.hsrag.retrieve import Retriever
    from analysis.knowledge.banks import german_audit as prior

    p.require(torch.cuda.is_available(), "Enable CUDA before local retrieval setup")
    offline(plan)
    torch.backends.cuda.matmul.allow_tf32 = False
    pools = {name: u.read_lines(BASE / f"kb/{name}/records.jsonl") for name in p.POOLS}
    prior_plan = u.read_json(prior.PLAN)
    train = u.read_lines(prior.BASE / "train_texts.jsonl")
    vectors = {}
    for start in range(0, len(train), prior_plan["cache_chunk"]):
        rows = train[start : start + prior_plan["cache_chunk"]]
        ident = prior.chunk_identity(prior_plan, [r["text"] for r in rows])
        key = hashlib.sha256(prior.js(ident)).hexdigest()
        vv = prior.load_chunk(
            prior.BASE / f"embedding_chunks/train_{start:06d}_{key[:16]}.npz", ident
        )
        for r, v in zip(rows, vv):
            vectors.setdefault(r["text"], v)
    remaining = list(
        dict.fromkeys(
            (r["text"] for rr in pools.values() for r in rr if r["text"] not in vectors)
        )
    )
    bgpath = BASE / "background_vectors.json"
    identity = {"plan_sha256": u.digest(PLAN), "texts_sha256": signature(remaining)}
    if bgpath.exists():
        bg = u.read_json(bgpath)
        p.require(bg["identity"] == identity, "Background vector identity changed")
        vv = np.asarray(bg["vectors"], dtype=np.float32)
    else:
        print(
            f"Encoding {len(remaining)} background texts; reusing verified German audit vectors.",
            flush=True,
        )
        model = SentenceTransformer(
            "BAAI/bge-m3",
            revision=plan["embedding_cache"]["revision"],
            local_files_only=True,
            device="cuda",
        )
        model.float()
        model.eval()
        vv = model.encode(
            remaining, normalize_embeddings=True, batch_size=4, show_progress_bar=False
        )
        p.write(bgpath, p.js({"identity": identity, "vectors": vv.tolist()}))
        del model
        gc.collect()
        torch.cuda.empty_cache()
    prior.validate_vectors(vv, len(remaining))
    vectors.update(zip(remaining, vv))
    files = [bgpath]
    for pool, records in pools.items():
        root = BASE / f"kb/{pool}"
        ident = {
            "embedding_model": "BAAI/bge-m3",
            "embedding_revision": plan["embedding_cache"]["revision"],
            "kb_version": u.digest(root / "records.jsonl")[:16],
            "plan_sha256": u.digest(PLAN),
        }
        p.require(
            not (root / "chroma").exists() or (root / "index_origin.json").exists(),
            "Unowned index exists",
        )
        p.write(root / "index_origin.json", p.js(ident))
        client = chromadb.PersistentClient(path=str(root / "chroma"))
        names = [c if isinstance(c, str) else c.name for c in client.list_collections()]
        p.require(set(names) <= {"hate_kb"}, "Unexpected index collection")
        col = (
            client.get_collection("hate_kb")
            if names
            else client.create_collection(
                "hate_kb", metadata={"hnsw:space": "cosine", **ident}
            )
        )
        p.require(
            all((col.metadata.get(k) == v for k, v in ident.items())),
            "Index metadata changed",
        )
        existing = set(col.get(include=[])["ids"])
        wanted = {r["id"] for r in records}
        p.require(existing <= wanted, "Unexpected index IDs")
        todo = [r for r in records if r["id"] not in existing]
        for start in range(0, len(todo), 128):
            rows = todo[start : start + 128]
            col.add(
                ids=[r["id"] for r in rows],
                documents=[r["text"] for r in rows],
                metadatas=[_flatten_meta(r) for r in rows],
                embeddings=[vectors[r["text"]].tolist() for r in rows],
            )
            if start % 1024 == 0:
                print(f"INDEX {pool}: {col.count()}/{len(records)}", flush=True)
        p.require(set(col.get(include=[])["ids"]) == wanted, "Incomplete index")
        for start in range(0, len(records), 256):
            rows = records[start : start + 256]
            got = col.get(
                ids=[r["id"] for r in rows],
                include=["documents", "metadatas", "embeddings"],
            )
            byid = {
                rid: (doc, meta, vec)
                for rid, doc, meta, vec in zip(
                    got["ids"], got["documents"], got["metadatas"], got["embeddings"]
                )
            }
            for r in rows:
                doc, meta, vec = byid[r["id"]]
                p.require(
                    doc == r["text"] and meta == _flatten_meta(r),
                    "Stored index text/labels differ",
                )
                p.require(
                    np.array_equal(
                        np.asarray(vec, dtype=np.float32), vectors[r["text"]]
                    ),
                    "Stored index vector differs",
                )
        marker = root / "index_complete.json"
        p.write(marker, p.js({**ident, "records": len(records)}))
        files.extend([root / "index_origin.json", marker])
    del vectors
    gc.collect()
    reference = None
    all_contexts = {}
    for pool, records in pools.items():
        path = BASE / f"contexts_{pool}.jsonl"
        if path.exists():
            collected = u.read_lines(path)
        else:
            retriever = Retriever(
                BASE / f"kb/{pool}/chroma",
                BASE / f"kb/{pool}/records.jsonl",
                "BAAI/bge-m3",
                dict(plan["retrieval"]),
            )
            collected = []
            for scope, subset in p.SCOPES.items():
                df = pd.read_parquet(
                    ROOT / f"data/processed/{subset}.parquet",
                    columns=["id", "text", "lang"],
                )
                for i, row in enumerate(df.itertuples(index=False), 1):
                    ctx = rendering.from_retrieval(
                        retriever.retrieve(row.text, row.lang)
                    )
                    collected.append(
                        {
                            "item_id": str(row.id),
                            "text": row.text,
                            "lang": row.lang,
                            "subset": subset,
                            "context": ctx,
                        }
                    )
                    if i % 50 == 0 or i == len(df):
                        print(f"RETRIEVE {pool}/{scope}: {i}/{len(df)}", flush=True)
            del retriever
            gc.collect()
            torch.cuda.empty_cache()
            p.write(path, p.lines(collected))
        known = {r["id"]: r for r in records}
        keys = []
        for item in collected:
            ctx = item["context"]
            examples = [r for name, rr in ctx["example_groups"] for r in rr]
            p.require(
                len(examples) == 5 and len({r["id"] for r in examples}) == 5,
                "Five unique example slots required",
            )
            p.require(
                all(
                    (
                        known[r["id"]]["lang"] == "de"
                        and known[r["id"]]["kind"] == "example"
                        for r in examples
                    )
                ),
                "Wrong example language/kind",
            )
            p.require(
                len(ctx["definitions"]) <= 2 and len(ctx["guidelines"]) <= 2,
                "Abstract budget exceeded",
            )
            keys.append((item["subset"], item["item_id"]))
        expected = [
            (subset, str(rid))
            for subset in p.SCOPES.values()
            for rid in pd.read_parquet(
                ROOT / f"data/processed/{subset}.parquet", columns=["id"]
            ).id
        ]
        p.require(keys == expected, "Context input roster differs")
        abstracts = [
            [r["context"][k] for k in ("definitions", "guidelines")] for r in collected
        ]
        if reference is None:
            reference = abstracts
        else:
            p.require(
                abstracts == reference,
                "Abstract retrieval changed across pools; stopped before API calls",
            )
        all_contexts[pool] = collected
        files.append(path)
    for condition, (pool, renderer) in p.CONDITIONS.items():
        entries = []
        for item in all_contexts[pool]:
            messages = rendering.render_messages(
                item["text"], item["lang"], item["context"], renderer
            )
            entries.append(
                {
                    **item,
                    "messages": messages,
                    "messages_sha256": signature(messages),
                    "context_sha256": signature(item["context"]),
                    "retrieved": rendering.hit_ids(item["context"]),
                }
            )
        path = BASE / f"prompts_{condition}.jsonl"
        p.write(path, p.lines(entries))
        files.append(path)
    validate()
    p.write(
        BASE / "setup_complete.json",
        p.js(
            {
                "plan_sha256": u.digest(PLAN),
                "n_inputs": len(reference),
                "abstract_context_equal": True,
                "hashes": {u.rel(f): u.digest(f) for f in files},
            }
        ),
    )
    print(
        "SETUP COMPLETE. Index vectors/text/labels and frozen prompts verified. No API calls.",
        flush=True,
    )


def item_path(run, entry):
    ident = hashlib.sha256(
        (entry["subset"] + "/" + entry["item_id"]).encode()
    ).hexdigest()
    return p.owned(BASE / f"runs/{run['name']}/items/{ident}.json")


def provenance(plan, run, entry):
    return {
        "plan_sha256": u.digest(PLAN),
        "run": run["name"],
        "condition": run["condition"],
        "pool": run["pool"],
        "renderer": run["renderer"],
        "messages_sha256": entry["messages_sha256"],
        "context_sha256": entry["context_sha256"],
    }


def stamps(plan, run):
    composite = signature(
        {
            "base": plan["code_version"],
            "runner": u.digest(Path(__file__)),
            "renderer": plan["renderer_sha256"],
            "mode": run["renderer"],
        }
    )[:16]
    return {
        "code_version": "astra-de-" + composite,
        "prompt_version": plan["prompt_version"]
        + "-astra-"
        + run["renderer"]
        + "-"
        + plan["renderer_sha256"][:8],
        "taxonomy_version": plan["taxonomy_version"],
        "kb_version": u.digest(BASE / f"kb/{run['pool']}/records.jsonl")[:16],
    }


def check_item(row, plan, run, entry):
    p.require(
        row["item_id"] == entry["item_id"]
        and row["lang"] == "de"
        and (row["arm"] == "rag"),
        "Wrong saved item",
    )
    p.require(
        row["text_hash"]
        == hashlib.sha256(" ".join(entry["text"].split()).encode()).hexdigest()[:16],
        "Input text changed",
    )
    p.require(
        row["astra"] == provenance(plan, run, entry), "Saved item provenance changed"
    )
    for key, value in {
        **stamps(plan, run),
        "active_model": plan["model"],
        "n_runs": 3,
        "workers": 8,
        "temperature": 1.0,
        "retrieval_config": plan["retrieval"],
    }.items():
        p.require(row.get(key) == value, f"Saved item stamp differs: {key}")
    p.require(row["retrieved"] == entry["retrieved"], "Delivered context differs")
    p.require(
        len(row["run_predictions"]) == len(row["raw_runs"]) == 3,
        "Incomplete saved votes",
    )
    p.require(
        row["n_valid"] == sum((v is not None for v in row["run_predictions"])),
        "Valid vote count differs",
    )


def entries(run):
    return [
        r
        for r in u.read_lines(BASE / f"prompts_{run['condition']}.jsonl")
        if r["subset"] == run["subset"]
    ]


def check_run(plan, run):
    rows, missing = ([], [])
    ee = entries(run)
    p.require(len(ee) == run["n_items"], "Wrong prompt count")
    expected = {item_path(run, e) for e in ee}
    actual = set((BASE / f"runs/{run['name']}/items").glob("*.json"))
    p.require(actual <= expected, "Unexpected completed item files")
    for entry in ee:
        path = item_path(run, entry)
        if path.exists():
            row = u.read_json(path)
            check_item(row, plan, run, entry)
            rows.append(row)
        else:
            missing.append(entry)
    return (rows, missing)


def client_factory(plan):
    import yaml
    import src.hsrag.client as clients

    clients.API_LOG = p.owned(BASE / "api_log.jsonl")
    api = yaml.safe_load((ROOT / "config/config.yaml").read_text(encoding="utf-8"))[
        "api"
    ]
    cfg = api["peasec"]
    return clients.make_client(
        "peasec",
        [plan["model"]],
        allow_fallback=False,
        temperature=plan["temperature"],
        max_tokens=api["max_tokens"],
        timeout=api["timeout_seconds"],
        url_env=cfg["url_env"],
        token_env=cfg["token_env"],
    )


def infer(plan, run, entry, client):
    from src.hsrag.classify import classify as original

    stamp = stamps(plan, run)
    messages = entry["messages"]
    p.require(
        signature(messages) == entry["messages_sha256"], "Frozen messages changed"
    )
    ctx = entry["context"]
    hit = lambda r: SimpleNamespace(**r)
    context = SimpleNamespace(
        definitions=[hit(r) for r in ctx["definitions"]],
        guidelines=[hit(r) for r in ctx["guidelines"]],
        example_groups=[(n, [hit(r) for r in rr]) for n, rr in ctx["example_groups"]],
    )
    arms = SimpleNamespace(
        retriever=SimpleNamespace(cfg=dict(plan["retrieval"])),
        context=lambda *args: context,
    )

    def frozen_prompt(text, lang, unused):
        p.require(
            text == entry["text"] and lang == entry["lang"],
            "Wrong input supplied to frozen prompt",
        )
        return json.loads(json.dumps(messages))

    globals_copy = {
        **original.__globals__,
        "build_prompt": frozen_prompt,
        "prompt_version": lambda: stamp["prompt_version"],
        "code_version": lambda: stamp["code_version"],
    }
    fn = FunctionType(
        original.__code__,
        globals_copy,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    result = fn(
        item_id=entry["item_id"],
        text=entry["text"],
        lang="de",
        arm="rag",
        client=client,
        arms=arms,
        n_votes=3,
        max_repairs=2,
        pinned_model=plan["model"],
        kb_version=stamp["kb_version"],
        workers=8,
        temperature=1.0,
    )
    row = json.loads(result.to_json())
    row["astra"] = provenance(plan, run, entry)
    check_item(row, plan, run, entry)
    return row


def predict_run(plan, name):
    p.require(check_setup(), "Retrieval setup missing")
    run = next((r for r in plan["runs"] if r["name"] == name))
    rows, todo = check_run(plan, run)
    print(f"RUN {name}: {len(rows)} complete, {len(todo)} remaining", flush=True)
    local = threading.local()

    def work(entry):
        if not hasattr(local, "client"):
            local.client = client_factory(plan)
        return infer(plan, run, entry, local.client)

    errors = []
    with ThreadPoolExecutor(max_workers=plan["workers"]) as executor:
        futures = {executor.submit(work, e): e for e in todo}
        for n, future in enumerate(as_completed(futures), 1):
            e = futures[future]
            try:
                row = future.result()
                p.write(item_path(run, e), p.js(row))
            except Exception as err:
                marker = {
                    "item_id": e["item_id"],
                    "error_type": type(err).__name__,
                    "run": name,
                    "time": u.now(),
                }
                errors.append(marker)
                p.write(
                    BASE / f"runs/{name}/failures/{time.time_ns()}.json", p.js(marker)
                )
                print(
                    f"Item {e['item_id']} failed ({type(err).__name__}); other completed items are preserved.",
                    flush=True,
                )
            if n % 10 == 0 or n == len(todo):
                print(
                    f"SAVED {name}: {len(rows) + n - len(errors)}/{run['n_items']}; {len(errors)} failed requests",
                    flush=True,
                )
    p.require(
        not errors,
        "Some requests failed. Resume the same launcher; only missing items will be retried.",
    )
    rows, missing = check_run(plan, run)
    p.require(not missing, "Missing results")
    from src.hsrag.metrics import score_all
    import pandas as pd

    scores = score_all(
        pd.read_parquet(ROOT / f"data/processed/{run['subset']}.parquet"),
        rows,
        gate_mapping="strict",
    )
    p.write(
        BASE / f"results/{name}_live.jsonl",
        p.lines(
            [
                {
                    "_manifest": {
                        **run,
                        **stamps(plan, run),
                        "arms": ["rag"],
                        "provider": plan["provider"],
                        "pinned_model": plan["model"],
                        "temperature": 1.0,
                        "n_votes": 3,
                        "workers": 8,
                        "retrieval": plan["retrieval"],
                        "records_path": u.rel(BASE / f"kb/{run['pool']}/records.jsonl"),
                        "chroma_path": u.rel(BASE / f"kb/{run['pool']}/chroma"),
                        "gate_mapping": "strict",
                        "plan_sha256": u.digest(PLAN),
                    }
                },
                *rows,
            ]
        ),
    )
    p.write(BASE / f"results/{name}_live.metrics.json", p.js(scores))
    invalid = sum((r["result"] is None or r["n_valid"] == 0 for r in rows))
    p.write(
        BASE / f"runs/{name}/complete.json",
        p.js(
            {
                "items": len(rows),
                "invalid": invalid,
                "item_hashes": {
                    u.rel(item_path(run, e)): u.digest(item_path(run, e))
                    for e in entries(run)
                },
                "result_sha256": u.digest(BASE / f"results/{name}_live.jsonl"),
                "metrics_sha256": u.digest(BASE / f"results/{name}_live.metrics.json"),
            }
        ),
    )
    p.require(
        invalid < len(rows),
        "Entire run has invalid predictions. Preserved for review; batch stopped.",
    )
    print(
        f"RUN COMPLETE {name}: {len(rows)} items; {invalid} null/invalid results preserved.",
        flush=True,
    )


def child(mode, *extra):
    log = p.owned(BASE / f"logs/{mode}_{time.time_ns()}.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-X",
        "utf8",
        "-B",
        "-u",
        "-m",
        "analysis.knowledge.banks.german_run",
        mode,
        *extra,
    ]
    print("COMMAND:", subprocess.list2cmdline(cmd), flush=True)
    with log.open("x", encoding="utf-8") as f:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
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
    p.require(
        code == 0,
        f"Stopped. Preserved log: {log}. Resume the same launcher after checking the error.",
    )


def complete_check(plan):
    path = BASE / "batch_complete.json"
    if not path.exists():
        return False
    receipt = u.read_json(path)
    p.require(receipt["plan_sha256"] == u.digest(PLAN), "Completed plan differs")
    p.require(check_setup(), "Missing setup receipt")
    for name, h in receipt["hashes"].items():
        p.require(u.digest(ROOT / name) == h, f"Completed output changed: {name}")
    for run in plan["runs"]:
        rows, missing = check_run(plan, run)
        p.require(not missing, "Missing saved item")
    return True


def main():
    os.chdir(ROOT)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=["validate", "setup", "run", "predict", "status"])
    ap.add_argument("--name")
    args = ap.parse_args()
    if args.mode == "status":
        print(
            json.dumps(
                {
                    "prepared": PLAN.exists(),
                    "setup_complete": (BASE / "setup_complete.json").exists(),
                    "batch_complete": (BASE / "batch_complete.json").exists(),
                },
                indent=2,
            )
        )
        return
    plan = validate()
    if args.mode == "validate":
        ready = check_setup()
        done = complete_check(plan)
        print(
            f"VALID: {len(plan['runs'])} runs, {plan['planned_calls']} ordinary samples; setup={ready}, batch complete={done}"
        )
    elif args.mode == "setup":
        with lock():
            setup(plan)
    elif args.mode == "predict":
        p.require(args.name in {r["name"] for r in plan["runs"]}, "Unknown run name")
        predict_run(plan, args.name)
    else:
        with lock():
            if complete_check(plan):
                print("ALL GERMAN RUNS ALREADY COMPLETE. Nothing rerun.")
                return
            if not check_setup():
                setup(plan)
            try:
                client = client_factory(plan)
                p.require(
                    plan["model"] in client.list_models(),
                    "Pinned model is unavailable; no substitution",
                )
            except Exception:
                raise RuntimeError(
                    "PEASEC model check failed. Check service/credentials and resume the same command."
                ) from None
            for run in plan["runs"]:
                validate(cache=False)
                rows, missing = check_run(plan, run)
                marker = BASE / f"runs/{run['name']}/complete.json"
                if missing or not marker.exists():
                    child("predict", "--name", run["name"])
                receipt = u.read_json(marker)
                p.require(
                    receipt["invalid"] < receipt["items"],
                    "A wholly invalid run requires review",
                )
                for name, h in receipt["item_hashes"].items():
                    p.require(u.digest(ROOT / name) == h, "Completed item changed")
                p.require(
                    u.digest(BASE / f"results/{run['name']}_live.jsonl")
                    == receipt["result_sha256"],
                    "Result changed",
                )
                p.require(
                    u.digest(BASE / f"results/{run['name']}_live.metrics.json")
                    == receipt["metrics_sha256"],
                    "Metrics changed",
                )
            validate()
            outputs = list((BASE / "results").glob("*")) + list(
                (BASE / "runs").glob("*/complete.json")
            )
            p.write(
                BASE / "batch_complete.json",
                p.js(
                    {
                        "plan_sha256": u.digest(PLAN),
                        "completed_utc": u.now(),
                        "hashes": {u.rel(f): u.digest(f) for f in outputs},
                    }
                ),
            )
            print(
                "ALL 24 GERMAN RUNS COMPLETE. Tell Astra to inspect the saved results.",
                flush=True,
            )


if __name__ == "__main__":
    main()

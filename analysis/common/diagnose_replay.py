import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import FunctionType, MethodType
from analysis.common import study as c
from analysis.common import cached_retrieval as retrieval
from analysis.common import prompt as prompt

BASE = c.BASE / "replay_diagnosis_20261009_v1"
PLAN = BASE / "plan.json"


def stream(path):
    with Path(path).open(encoding="utf8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def compare(first, actual):
    fields = sorted(
        (k for k in first.keys() | actual.keys() if first.get(k) != actual.get(k))
    )
    same_messages = first["messages"] == actual["messages"]
    same_ids = first["retrieved"] == actual["retrieved"]
    kinds = [
        k
        for k in first["retrieved"].keys() | actual["retrieved"].keys()
        if first["retrieved"].get(k) != actual["retrieved"].get(k)
    ]
    same_sets = all(
        (
            set(first["retrieved"].get(k, [])) == set(actual["retrieved"].get(k, []))
            for k in first["retrieved"].keys() | actual["retrieved"].keys()
        )
    )
    clean = lambda x: (
        {k: clean(v) for k, v in x.items() if k != "_ranks"}
        if isinstance(x, dict)
        else [clean(v) for v in x] if isinstance(x, list) else x
    )
    return dict(
        changed_fields=fields,
        same_messages=same_messages,
        same_retrieved_ids=same_ids,
        changed_buckets=kinds,
        kind=(
            "exact"
            if not fields
            else (
                "metadata_only"
                if same_messages and same_ids
                else "order" if same_sets else "membership"
            )
        ),
        context_equal_without_ranks=clean(first["context"]) == clean(actual["context"]),
    )


def check_entry(row, source, known, pool, plan_sha):
    expected = dict(
        key=source["key"],
        item_id=str(source["id"]),
        lang=source["lang"],
        population=source["phase"],
        condition="rag_" + pool,
        plan_sha256=plan_sha,
    )
    c.require(
        all((row[k] == v for k, v in expected.items())), "Changed query/plan identity"
    )
    c.require(
        row["context_sha256"] == c.signature(row["context"])
        and row["messages_sha256"] == c.signature(row["messages"]),
        "Invalid entry signature",
    )
    c.require(
        row["retrieved"] == prompt.hit_ids(row["context"]),
        "Invalid retrieved ID signature",
    )
    retrieval.old.validate_context(row["context"], known, row["lang"])


def audit():
    p = retrieval.validate()
    plan_sha = c.sha(retrieval.PLAN)
    roster = c.rows(c.PREP / "broad_roster.jsonl")
    keys = [r["key"] for r in roster]
    gold = {r["key"]: r for r in roster}
    c.require(len(keys) == len(set(keys)) == 21627, "Invalid original roster")
    inputs = {
        c.rel(retrieval.PLAN): plan_sha,
        **p["inputs"],
        **p["original_index_hashes"],
    }
    counts = {}
    changes = {}
    selections = {}
    variants = []
    entries_checked = 0
    rendered = 0
    for pool in ("small", "full"):
        known = {r["id"]: r for r in c.rows(c.CORE / f"kb/{pool}/records.jsonl")}
        frozen_path = retrieval.BASE / pool / "prompts.jsonl"
        originals = {r["key"]: r for r in stream(frozen_path)}
        c.require(list(originals) == keys, "Frozen prompt order/identity differs")
        for row in originals.values():
            check_entry(row, gold[row["key"]], known, pool, plan_sha)
            entries_checked += 1
        pool_changed = set()
        message_changed = set()
        info = []
        for mode in ("freeze", "replay1", "replay2"):
            path = retrieval.BASE / pool / mode / "complete.json"
            receipt = c.read(path)
            c.require(
                receipt["rows"] == 21627
                and receipt["plan_sha256"] == plan_sha
                and (receipt["pool"] == pool)
                and (receipt["mode"] == mode),
                "Wrong pass completion",
            )
            c.verify(receipt["outputs"])
            c.require(
                receipt["original_index_hashes_after"] == p["original_index_hashes"],
                "Original indexes changed during pass",
            )
            inputs[c.rel(path)] = c.sha(path)
            inputs.update(receipt["outputs"])
            chunk_rows = []
            for chunk_path in sorted(
                (retrieval.BASE / pool / mode / "chunks").glob("chunk_*.json")
            ):
                chunk = c.read(chunk_path)
                c.require(
                    chunk["stamp"] == dict(plan_sha256=plan_sha, pool=pool, mode=mode),
                    "Wrong checkpoint stamp",
                )
                c.require(
                    chunk["keys"] == [r["key"] for r in chunk["rows"]]
                    and chunk["rows_sha256"] == c.signature(chunk["rows"]),
                    "Invalid durable chunk",
                )
                chunk_rows.extend(chunk["rows"])
            c.require(
                [r["key"] for r in chunk_rows] == keys
                and receipt["checkpoint_resume"]["chunks"] == 217,
                "Incomplete chunk roster",
            )
            if mode == "freeze":
                c.require(
                    chunk_rows == list(originals.values()),
                    "Frozen file/chunks disagree",
                )
                observed = {}
            else:
                saved = c.rows(retrieval.BASE / pool / mode / "comparisons.jsonl")
                c.require(saved == chunk_rows, "Comparison/chunks disagree")
                observed = Counter()
                for r in saved:
                    first = originals[r["key"]]
                    if r["changed_fields"]:
                        actual = r["actual_entry"]
                        check_entry(actual, gold[r["key"]], known, pool, plan_sha)
                        entries_checked += 1
                        checked = compare(first, actual)
                        for k in (
                            "changed_fields",
                            "same_messages",
                            "same_retrieved_ids",
                        ):
                            c.require(
                                checked[k] == r[k], "Incorrect replay comparison flag"
                            )
                        c.require(
                            set(checked["changed_fields"])
                            <= {
                                "context",
                                "context_sha256",
                                "messages",
                                "messages_sha256",
                                "retrieved",
                            },
                            "Unexpected changed identity/source",
                        )
                        c.require(
                            not checked["same_messages"]
                            or checked["context_equal_without_ranks"],
                            "Unexplained same-message content change",
                        )
                        for value in (first, actual):
                            c.require(
                                value["messages"]
                                == prompt.render_messages(
                                    gold[r["key"]]["text"],
                                    value["lang"],
                                    value["context"],
                                    "corrected",
                                ),
                                "Affected message rendering differs",
                            )
                            rendered += 1
                        row = dict(
                            pool=pool,
                            mode=mode,
                            key=r["key"],
                            population=first["population"],
                            **checked,
                        )
                        info.append(row)
                        variants.append(
                            dict(
                                pool=pool,
                                mode=mode,
                                key=r["key"],
                                entry=actual,
                                entry_sha256=c.signature(actual),
                            )
                        )
                        pool_changed.add(r["key"])
                        if not checked["same_messages"]:
                            message_changed.add(r["key"])
                    else:
                        c.require(
                            r["same_messages"]
                            and r["same_retrieved_ids"]
                            and ("actual_entry" not in r),
                            "Inconsistent exact replay flag",
                        )
                    observed["entry_differences"] += bool(r["changed_fields"])
                    observed["message_differences"] += not r["same_messages"]
                    observed["retrieved_id_differences"] += not r["same_retrieved_ids"]
                c.require(
                    dict(observed) == receipt["differences"],
                    "Receipt/recomputed count differs",
                )
            counts[pool + "/" + mode] = dict(
                rows=receipt["rows"],
                differences=dict(observed),
                seconds_recorded=receipt["seconds"],
                checkpoint_reused_rows=receipt["checkpoint_resume"]["reused_rows"],
            )
            del chunk_rows
        controls = []
        for phase in sorted({r["phase"] for r in roster}):
            eligible = sorted(
                [
                    r["key"]
                    for r in roster
                    if r["phase"] == phase and r["key"] not in pool_changed
                ]
            )
            controls += eligible[:2]
        if c.EMPTY_KEY not in pool_changed and c.EMPTY_KEY not in controls:
            controls.append(c.EMPTY_KEY)
        selections[pool] = dict(
            affected=sorted(pool_changed),
            controls=controls,
            all=sorted(pool_changed | set(controls)),
        )
        changes[pool] = dict(
            entry_union=sorted(pool_changed),
            message_union=sorted(message_changed),
            message_by_population=dict(
                Counter((gold[k]["phase"] for k in message_changed))
            ),
            cases=info,
        )
        copyreceipt = c.read(retrieval.BASE / pool / "copy.json")
        current = {
            f.relative_to(retrieval.BASE / pool / "chroma").as_posix(): c.sha(f)
            for f in (retrieval.BASE / pool / "chroma").rglob("*")
            if f.is_file()
        }
        counts[pool + "/physical_copy"] = {
            "changed_from_copy_origin": [
                n for n, h in current.items() if copyreceipt["source"].get(n) != h
            ],
            "current_hashes": current,
            "source_hashes": copyreceipt["source"],
        }
        print(
            pool,
            "all passes/entries verified;",
            len(pool_changed),
            "changed input entries,",
            len(message_changed),
            "changed prompts",
            flush=True,
        )
    c.verify(p["original_index_hashes"])
    inputs.update(
        {
            c.rel(Path(__file__)): c.sha(Path(__file__)),
            c.rel(c.ROOT / "tests/common/test_replay_diagnosis.py"): c.sha(
                c.ROOT / "tests/common/test_replay_diagnosis.py"
            ),
        }
    )
    c.save(
        BASE / "audit.json",
        dict(
            created_utc=c.utc(),
            status="all_six_saved_passes_verified_nonzero_differences",
            inputs=inputs,
            passes=counts,
            changes=changes,
            entries_checked=entries_checked,
            affected_renders_checked=rendered,
            original_indexes_unchanged=True,
            Qwen_calls=0,
            limits="Complete first-pass/chunk/source identity and annotation checks. Exact online determinism failed. No classifier effect inferred. Full independent token/render check remains downstream.",
        ),
    )
    c.save_rows(BASE / "observed_variants.jsonl", variants)
    c.save(
        PLAN,
        dict(
            created_utc=c.utc(),
            status="frozen_before_targeted_diagnostics",
            inputs=inputs
            | {
                c.rel(BASE / "audit.json"): c.sha(BASE / "audit.json"),
                c.rel(BASE / "observed_variants.jsonl"): c.sha(
                    BASE / "observed_variants.jsonl"
                ),
            },
            selections=selections,
            diagnostic="Two fresh processes per pool on newly owned source-index copies. Query each affected/control input twice. Trace dense candidates, BM25 and unmodified MMR outputs. No replacement or outcome selection; source indexes only read/copied.",
            limits="Targeted diagnostics do not replace the full passes or prove a unique backend cause. Original first-pass snapshots remain authoritative under the already declared plan; any acceptance needs an explicit additive review.",
        ),
    )


def worker(pool, number):
    import numpy as np

    p = c.read(PLAN)
    c.verify(p["inputs"])
    dest = BASE / pool / f"cold_{number}.json"
    c.require(number in (1, 2) and (not dest.exists()), "Preserve prior diagnostic run")
    source = c.CORE / f"retrieval_v1/{pool}/chroma"
    copydir = BASE / pool / "chroma"
    if not copydir.exists():
        shutil.copytree(source, copydir)
    before = retrieval.snapshot(copydir)
    import chromadb

    col = chromadb.PersistentClient(path=str(copydir)).get_collection("hate_kb")
    records = c.rows(c.CORE / f"kb/{pool}/records.jsonl")
    full = c.rows(c.CORE / "kb/full/records.jsonl")
    ix = {r["id"]: i for i, r in enumerate(full)}
    matrix = np.load(
        c.CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    maxdiff = retrieval.old.verify_collection(
        col, records, np.asarray(matrix[[ix[r["id"]] for r in records]])
    )
    roster = c.rows(c.PREP / "broad_roster.jsonl")
    byid = {r["key"]: (i, r) for i, r in enumerate(roster)}
    q = np.load(
        retrieval.BASE / "vectors/query_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    encoder = retrieval.old.CachedEncoder([], q)
    ret = retrieval.old.make_retriever(
        col,
        c.CORE / f"kb/{pool}/records.jsonl",
        c.read(retrieval.PLAN)["retrieval"],
        encoder,
    )
    from src.hsrag import retrieve as original

    trace = []

    def hits(rows):
        return [
            dict(
                id=h.id,
                score=h.score,
                via=h.via,
                vec_sha256=(
                    hashlib.sha256(
                        np.asarray(h.vec, dtype=np.float64).tobytes()
                    ).hexdigest()
                    if h.vec is not None
                    else None
                ),
            )
            for h in rows
        ]

    dense = ret._query_kind
    lexical = ret._bm25_kind

    def dense_trace(*args, **kwargs):
        result = dense(*args, **kwargs)
        trace.append(
            dict(operation="dense", where=args[1], k=args[2], hits=hits(result))
        )
        return result

    def lexical_trace(*args, **kwargs):
        result = lexical(*args, **kwargs)
        trace.append(dict(operation="bm25", bucket=args[1], hits=hits(result)))
        return result

    def mmr_trace(qvec, rows, lam, top_n):
        candidate = hits(rows)
        result = original.mmr_select(qvec, rows, lam, top_n)
        trace.append(
            dict(operation="mmr", input=candidate, selected=[h.id for h in result])
        )
        return result

    ret._query_kind = dense_trace
    ret._bm25_kind = lexical_trace
    method = FunctionType(
        original.Retriever.retrieve.__code__, vars(original) | {"mmr_select": mmr_trace}
    )
    ret.retrieve = MethodType(method, ret)
    selected = p["selections"][pool]["all"]
    frozen = {
        r["key"]: r
        for r in stream(retrieval.BASE / pool / "prompts.jsonl")
        if r["key"] in selected
    }
    output = []
    for j, key in enumerate(selected):
        idx, row = byid[key]
        encoder.lookup = {row["text"]: q[idx]}
        repeats = []
        for rep in range(2):
            trace.clear()
            ctx = prompt.from_retrieval(ret.retrieve(row["text"], row["lang"]))
            msg = prompt.render_messages(row["text"], row["lang"], ctx, "corrected")
            actual = frozen[key] | dict(
                context=ctx,
                context_sha256=c.signature(ctx),
                messages=msg,
                messages_sha256=c.signature(msg),
                retrieved=prompt.hit_ids(ctx),
            )
            repeats.append(
                dict(
                    comparison=compare(frozen[key], actual),
                    actual_entry=actual,
                    trace=copy.deepcopy(trace),
                )
            )
        output.append(
            dict(
                key=key,
                affected_in_full_passes=key in p["selections"][pool]["affected"],
                repeats=repeats,
            )
        )
        if (j + 1) % 5 == 0:
            print(pool, "diagnostic", number, j + 1, "/", len(selected), flush=True)
    after = retrieval.snapshot(copydir)
    c.verify(c.read(retrieval.PLAN)["original_index_hashes"])
    c.save(
        dest,
        dict(
            plan_sha256=c.sha(PLAN),
            pool=pool,
            cold_pass=number,
            rows=output,
            index_vector_max_abs_difference=maxdiff,
            physical_before=before,
            physical_after=after,
            Qwen_calls=0,
        ),
    )
    print(
        "Saved targeted diagnosis",
        pool,
        number,
        len(output),
        "inputs x two repeats",
        flush=True,
    )


def run():
    for pool in ("small", "full"):
        for number in (1, 2):
            path = BASE / pool / f"cold_{number}.json"
            if path.exists():
                c.require(
                    c.read(path)["plan_sha256"] == c.sha(PLAN),
                    "Changed saved diagnostic",
                )
                continue
            subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-B",
                    "-u",
                    "-m",
                    __spec__.name,
                    "worker",
                    "--pool",
                    pool,
                    "--number",
                    str(number),
                ],
                cwd=c.ROOT,
                check=True,
            )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["audit", "run", "worker"])
    p.add_argument("--pool", choices=["small", "full"])
    p.add_argument("--number", type=int)
    args = p.parse_args()
    with c.offline():
        if args.mode == "worker":
            worker(args.pool, args.number)
        elif args.mode == "audit":
            from analysis.common.api import item_lock

            with item_lock(retrieval.BASE / "launcher.lock"):
                audit()
        else:
            run()

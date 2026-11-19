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
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import gc
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch
import numpy as np
from analysis.feedback.restoration import study as s
from analysis.knowledge.strategies import verify as audit
from analysis.feedback.restoration import filter as filtered

BASE = s.BASE
QUEUE = BASE / "preparation_queue_v1"


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    s.c.write(path, obj)


def prefix(path, keys):
    if not path.exists():
        return []
    raw = path.read_bytes()
    s.c.require(
        not raw or raw.endswith(b"\n"),
        "Interrupted final JSONL line preserved. Ask Astra to inspect; no automatic truncation: "
        + str(path),
    )
    rows = [json.loads(line) for line in raw.decode("utf8").splitlines()]
    s.c.require(
        len(rows) <= len(keys) and [r["key"] for r in rows] == keys[: len(rows)],
        "Incomplete file does not have the exact ordered input prefix",
    )
    return rows


def completed(path, receipt, keys):
    if not receipt.exists():
        return False
    done = s.c.read(receipt)
    s.c.require(
        path.exists()
        and s.c.sha(path) == done["file_sha256"]
        and (done["rows"] == len(keys)),
        "Completed preparation view changed",
    )
    s.c.require(len(prefix(path, keys)) == len(keys), "Incomplete completed view")
    return True


@contextmanager
def lock(path):
    import msvcrt

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
                "This offline taxonomy preparation is already running"
            ) from None
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def verify_queue():
    plan = s.c.read(QUEUE / "plan.json")
    for name, expected in plan["inputs"].items():
        s.c.require(
            s.c.sha(s.ROOT / name) == expected,
            "Offline queue dependency changed: " + name,
        )
    s.verify_plan()
    s.verify_scope()
    return plan


def fixtures():
    root = QUEUE / f"fixture_evidence_{time.time_ns()}"
    root.mkdir(parents=True, exist_ok=False)
    keys = ["a", "b", "c"]
    tested = []
    path = root / "prefix.jsonl"
    assert prefix(path, keys) == []
    tested.append("missing_prefix_starts_at_zero")
    path.write_text('{"key":"a"}\n', encoding="utf8")
    assert prefix(path, keys) == [{"key": "a"}]
    tested.append("complete_prefix_retained")
    assert not completed(path, root / "absent.json", keys)
    tested.append("no_receipt_requires_remaining_work")
    bad = root / "partial.jsonl"
    bad.write_bytes(b'{"key":"a"}\n{"key":')
    try:
        prefix(bad, keys)
    except AssertionError:
        pass
    else:
        raise AssertionError("Truncated tail accepted")
    assert bad.read_bytes() == b'{"key":"a"}\n{"key":'
    tested.append("truncated_tail_stops_without_modification")
    wrong = root / "wrong.jsonl"
    wrong.write_text('{"key":"b"}\n', encoding="utf8")
    try:
        prefix(wrong, keys)
    except AssertionError:
        pass
    else:
        raise AssertionError("Wrong prefix accepted")
    tested.append("wrong_order_refused")
    repeated = root / "duplicate.jsonl"
    repeated.write_text('{"key":"a"}\n{"key":"a"}\n', encoding="utf8")
    try:
        prefix(repeated, keys)
    except AssertionError:
        pass
    else:
        raise AssertionError("Duplicate prefix accepted")
    tested.append("duplicate_prefix_refused")
    full = root / "complete.jsonl"
    full.write_text(
        "".join((json.dumps({"key": k}) + "\n" for k in keys)), encoding="utf8"
    )
    receipt = root / "receipt.json"
    put(receipt, {"file_sha256": s.c.sha(full), "rows": 3})
    assert completed(full, receipt, keys)
    tested.append("completed_restart_needs_no_retrieval")
    corrupt = root / "corrupt_receipt.json"
    put(corrupt, {"file_sha256": "0" * 64, "rows": 3})
    try:
        completed(full, corrupt, keys)
    except AssertionError:
        pass
    else:
        raise AssertionError("Tampered completion accepted")
    tested.append("changed_completion_refused")
    return tested


def preflight():
    start = time.perf_counter()
    s.verify_plan()
    s.verify_scope()
    s.c.require(not (QUEUE / "preflight.json").exists(), "Preflight already recorded")
    tested = fixtures()
    import chromadb

    col = chromadb.PersistentClient(path=str(BASE / "chroma_copy")).get_collection(
        "hate_kb"
    )
    all_records = s.c.lines(s.CORE / "kb/full/records.jsonl")
    ix = {r["id"]: i for i, r in enumerate(all_records)}
    vectors = np.load(BASE / "indexed_vectors.npy", allow_pickle=False, mmap_mode="r")
    core = s.c.lines(s.CORE / "dev_inputs.jsonl")
    q = np.load(s.RET / "query_vectors.npy", allow_pickle=False, mmap_mode="r")
    qm = {r["key"]: q[i] for i, r in enumerate(core)}
    textmap = {r["text"]: q[i] for i, r in enumerate(core)}
    roster = s.c.lines(BASE / "roster.jsonl")
    cfg = s.c.read(BASE / "preparation_plan.json")["retrieval"]
    from analysis.feedback.restoration import validate_earlier as scope_check

    raw_vectors = np.load(s.RET / "full_vectors.npy", allow_pickle=False, mmap_mode="r")
    independent_roster, independent_clusters = scope_check.check_scope(
        all_records, core, q, raw_vectors
    )
    s.c.require(independent_roster == roster, "Independent roster differs")
    full_path = BASE / "retrieval/full.jsonl"
    s.c.require(
        completed(
            full_path, BASE / "retrieval/full.complete.json", [r["key"] for r in roster]
        ),
        "Initial full view must finish before preflight",
    )
    full_checker = audit.TraceAudit(all_records, vectors, qm)
    historical = {r["key"]: r for r in s.c.lines(s.RET / "full/prompts.jsonl")}
    known_full = {r["id"]: r for r in all_records}
    historical_changes = []
    for entry, row in zip(s.c.lines(full_path), roster):
        s.c.require(
            entry["text"] == row["text"] and entry["pool"] == "full",
            "Full first-pass row differs",
        )
        full_checker.entry(entry)
        context = audit.source_context(entry["final_hits"], known_full)
        s.c.require(
            s.sig(context) == s.sig(entry["context"]) == entry["context_sha256"],
            "Full context differs",
        )
        messages = s.isolated.bundle(BASE / "taxonomies/full.yaml").render(
            row["text"], "en", context
        )
        s.c.require(
            messages
            == s.renderer.render_messages(row["text"], "en", context, "corrected"),
            "Full renderer differs",
        )
        if messages != historical[row["key"]]["messages"]:
            historical_changes.append(
                {
                    "key": row["key"],
                    "old_retrieved": historical[row["key"]]["retrieved"],
                    "new_retrieved": s.renderer.hit_ids(context),
                    "new_messages_sha256": s.sig(messages),
                }
            )
    full_counts = dict(full_checker.counts)
    full_error = full_checker.max_dense_score_error
    del full_checker
    gc.collect()
    traces = []
    totals = Counter()
    for pool in s.POOLS:
        path = (
            s.CORE / "kb/full/records.jsonl"
            if pool == "full"
            else BASE / f"kb/{pool}/records.jsonl"
        )
        records = all_records if pool == "full" else s.c.lines(path)
        row = (
            roster[0]
            if pool == "full"
            else next((r for r in roster if pool in (r["gold"]["hate_type"] or [])))
        )
        retriever = filtered.make_retriever(
            col, records, path, cfg, textmap, pool != "full"
        )
        events = s.instrument(retriever)
        started = time.perf_counter()
        result = retriever.retrieve(row["text"], "en")
        elapsed = time.perf_counter() - started
        entry = {
            "key": row["key"],
            "text": row["text"],
            "lang": "en",
            "pool": pool,
            "strategy": "hybrid",
            "context": s.renderer.from_retrieval(result),
            "trace": deepcopy(events),
            "final_hits": [s.hit_trace(h) for h in result.all_hits()],
        }
        checker = audit.TraceAudit(records, vectors[[ix[r["id"]] for r in records]], qm)
        checker.entry(entry)
        context = audit.source_context(
            entry["final_hits"], {r["id"]: r for r in records}
        )
        s.c.require(
            s.sig(context) == s.sig(entry["context"]),
            "Preflight context source mismatch",
        )
        if pool != "full":
            a = s.isolated.bundle(BASE / f"taxonomies/{pool}.yaml").render(
                row["text"], "en", context
            )
            z = s.isolated.bundle(BASE / "taxonomies/full.yaml").render(
                row["text"], "en", context
            )
            s.c.require(a[1] == z[1] and a[0] != z[0], "Preflight schema intervention")
        traces.append(
            {
                "entry": entry,
                "seconds": elapsed,
                "independent_trace_counts": dict(checker.counts),
            }
        )
        totals.update(checker.counts)
        print(
            f"PREFLIGHT {pool}: original ranking, allowed records and schema context checked",
            flush=True,
        )
        del retriever, checker, records
        gc.collect()
    from jinja2.sandbox import ImmutableSandboxedEnvironment
    from tokenizers import Tokenizer
    from analysis.feedback.restoration import validate as checker_v2

    amendment = s.c.read(BASE / "checker_amendment_v1.json")
    s.c.require(
        amendment["new_checker_sha256"] == s.c.sha(Path(checker_v2.__file__)),
        "Checker amendment mismatch",
    )
    chat = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    ).from_string(s.c.read(s.TOK / "chat_template.json")["chat_template"])
    tok = Tokenizer.from_file(str(s.TOK / "tokenizer.json"))
    tok.no_padding()
    tok.no_truncation()
    messages = s.isolated.bundle(BASE / "taxonomies/full.yaml").render(
        roster[0]["text"], "en", traces[0]["entry"]["context"]
    )
    official = chat.render(
        messages=[{"role": m["role"], "content": m["text"]} for m in messages],
        tools=None,
        add_generation_prompt=True,
        add_vision_id=False,
    )
    assert official == s.local_chat(messages)
    put(
        QUEUE / "preflight.json",
        {
            "status": "offline_queue_preflight_passed",
            "fixtures": tested,
            "real_index_views": len(traces),
            "compact_filter_fixtures": filtered.synthetic_checks(),
            "exact_retained_membership_checked_for_seven_views": True,
            "independent_scope_inputs": len(independent_roster),
            "independent_scope_clusters": sum(
                (g["included"] for g in independent_clusters)
            ),
            "initial_full_view_trace_checks": full_counts,
            "initial_full_view_dense_max_error": full_error,
            "historical_full_message_differences": historical_changes,
            "trace_counts": dict(totals),
            "traces": traces,
            "tokenizer_smoke_tokens": len(
                tok.encode(official, add_special_tokens=False).ids
            ),
            "seconds": time.perf_counter() - start,
            "source_hashes": {s.c.rel(Path(__file__)): s.c.sha(Path(__file__))},
            "active_bindings": s.c.verify_active(),
            "Qwen_calls": 0,
            "new_embeddings": 0,
            "CUDA": False,
        },
    )
    print(
        json.dumps(
            {
                "status": "offline_queue_preflight_passed",
                "fixtures": len(tested),
                "views": len(traces),
                "seconds": time.perf_counter() - start,
            }
        ),
        flush=True,
    )


def freeze():
    s.verify_plan()
    s.verify_scope()
    preflight = s.c.read(QUEUE / "preflight.json")
    s.c.require(
        preflight["status"] == "offline_queue_preflight_passed", "Preflight not passed"
    )
    s.c.require(
        preflight["source_hashes"][s.c.rel(Path(__file__))] == s.c.sha(Path(__file__)),
        "Queue changed after preflight",
    )
    roster = s.c.lines(BASE / "roster.jsonl")
    keys = [r["key"] for r in roster]
    s.c.require(
        completed(
            BASE / "retrieval/full.jsonl", BASE / "retrieval/full.complete.json", keys
        ),
        "Finish the initial agent-run full view before handing over",
    )
    inputs = [
        Path(__file__),
        s.ROOT / "scripts/prepare_restoration.ps1",
        QUEUE / "preflight.json",
        BASE / "preparation_plan.json",
        BASE / "scope_complete.json",
        BASE / "scientific_plan_v1.json",
        BASE / "checker_amendment_v1.json",
        BASE / "retrieval/full.jsonl",
        BASE / "retrieval/full.complete.json",
        s.ROOT / "analysis/feedback/restoration/validate.py",
        s.ROOT / "analysis/feedback/restoration/filter.py",
        BASE / "filter_execution_amendment_v1.json",
    ]
    stages = [
        {"mode": mode, "pool": pool}
        for mode in ("retrieve", "replay")
        for pool in s.POOLS
    ]
    put(
        QUEUE / "plan.json",
        {
            "status": "frozen_offline_preparation_only",
            "inputs": {s.c.rel(p): s.c.sha(p) for p in inputs},
            "stages": stages,
            "initial_full_view_complete": True,
            "input_count": len(keys),
            "network_calls": 0,
            "new_embeddings": 0,
            "CUDA": False,
            "API_concurrency": 0,
            "resume": "Verify and skip completed views. Independently audit complete JSONL prefixes, retain them exactly and append missing rows. Stop on truncated final line or conflicting identity.",
            "after_retrieval": "Render every frozen condition, run independent checker v2, stop for Astra review. No Qwen job is launched.",
        },
    )
    print(
        json.dumps(
            {
                "status": "offline_taxonomy_command_ready",
                "queue_plan_sha256": s.c.sha(QUEUE / "plan.json"),
            }
        ),
        flush=True,
    )


def worker(mode, pool):
    verify_queue()
    started = time.perf_counter()
    roster = s.c.lines(BASE / "roster.jsonl")
    keys = [r["key"] for r in roster]
    folder = "retrieval" if mode == "retrieve" else "replay"
    path = BASE / f"{folder}/{pool}.jsonl"
    receipt = path.with_suffix(".complete.json")
    with lock(QUEUE / f"locks/{mode}_{pool}.lock"):
        if completed(path, receipt, keys):
            print(f"VERIFIED SKIP {mode}/{pool}", flush=True)
            return
        existing = prefix(path, keys)
        if existing:
            print(
                f"Auditing {len(existing)} saved {mode}/{pool} rows before resuming",
                flush=True,
            )
        import chromadb

        col = chromadb.PersistentClient(path=str(BASE / "chroma_copy")).get_collection(
            "hate_kb"
        )
        records_path = (
            s.CORE / "kb/full/records.jsonl"
            if pool == "full"
            else BASE / f"kb/{pool}/records.jsonl"
        )
        records = s.c.lines(records_path)
        core = s.c.lines(s.CORE / "dev_inputs.jsonl")
        q = np.load(s.RET / "query_vectors.npy", allow_pickle=False, mmap_mode="r")
        textmap = {r["text"]: q[i] for i, r in enumerate(core)}
        retriever = filtered.make_retriever(
            col,
            records,
            records_path,
            s.c.read(BASE / "preparation_plan.json")["retrieval"],
            textmap,
            pool != "full",
        )
        events = s.instrument(retriever)
        first = (
            s.c.lines(BASE / f"retrieval/{pool}.jsonl") if mode == "replay" else None
        )
        changes = Counter()
        if existing:
            all_records = s.c.lines(s.CORE / "kb/full/records.jsonl")
            ix = {r["id"]: i for i, r in enumerate(all_records)}
            v = np.load(BASE / "indexed_vectors.npy", allow_pickle=False, mmap_mode="r")
            checker = audit.TraceAudit(
                records,
                v[[ix[r["id"]] for r in records]],
                {r["key"]: q[i] for i, r in enumerate(core)},
            )
            known = {r["id"]: r for r in records}
            for i, stored in enumerate(existing):
                if mode == "retrieve":
                    entry = stored
                else:
                    flags = stored["matches"]
                    assert set(flags) == {"context", "retrieved", "final_hits", "trace"}
                    assert all((type(v) is bool for v in flags.values()))
                    entry = first[i] if all(flags.values()) else stored["actual_entry"]
                    assert flags == {k: entry[k] == first[i][k] for k in flags}
                    changes.update((k for k, v in flags.items() if not v))
                assert (
                    entry["key"] == keys[i]
                    and entry["text"] == roster[i]["text"]
                    and (entry["pool"] == pool)
                    and (entry["lang"] == "en")
                )
                checker.entry(entry)
                assert (
                    s.sig(audit.source_context(entry["final_hits"], known))
                    == s.sig(entry["context"])
                    == entry["context_sha256"]
                )
                assert entry["retrieved"] == s.renderer.hit_ids(entry["context"])
            put(
                QUEUE / f"resumptions/{time.time_ns()}_{mode}_{pool}.json",
                {
                    "rows_preserved": len(existing),
                    "prefix_sha256": s.c.sha(path),
                    "queue_plan_sha256": s.c.sha(QUEUE / "plan.json"),
                    "independent_counts": dict(checker.counts),
                },
            )
            del checker
            gc.collect()
        path.parent.mkdir(exist_ok=True)
        with path.open(
            "a" if path.exists() else "x", encoding="utf8", newline="\n"
        ) as f:
            for i in range(len(existing), len(roster)):
                row = roster[i]
                events.clear()
                result = retriever.retrieve(row["text"], "en")
                s.c.require(
                    not result.feedback
                    and len(result.examples) == 5
                    and (len(result.definitions) == 2)
                    and (len(result.guidelines) == 2),
                    "Retrieval budget",
                )
                context = s.renderer.from_retrieval(result)
                entry = {
                    "key": row["key"],
                    "text": row["text"],
                    "lang": "en",
                    "pool": pool,
                    "strategy": "hybrid",
                    "context": context,
                    "context_sha256": s.sig(context),
                    "retrieved": s.renderer.hit_ids(context),
                    "final_hits": [s.hit_trace(h) for h in result.all_hits()],
                    "trace": deepcopy(events),
                }
                if mode == "replay":
                    flags = {
                        k: entry[k] == first[i][k]
                        for k in ("context", "retrieved", "final_hits", "trace")
                    }
                    changes.update((k for k, v in flags.items() if not v))
                    line = {"key": row["key"], "matches": flags}
                    if not all(flags.values()):
                        line["actual_entry"] = entry
                else:
                    line = entry
                f.write(
                    json.dumps(
                        line, ensure_ascii=False, sort_keys=True, allow_nan=False
                    )
                    + "\n"
                )
                f.flush()
                if (i + 1) % 50 == 0 or i + 1 == len(roster):
                    print(
                        f"{mode.upper()} {pool}: {i + 1}/{len(roster)} ({time.perf_counter() - started:.1f}s this attempt)",
                        flush=True,
                    )
        put(
            receipt,
            {
                "file_sha256": s.c.sha(path),
                "rows": len(roster),
                "seconds": time.perf_counter() - started,
                "resumed_rows": len(existing),
                "differences": dict(changes),
                "retained_first_pass": True,
                "queue_plan_sha256": s.c.sha(QUEUE / "plan.json"),
                "active_bindings": s.c.verify_active(),
            },
        )


def child(module, args, attempt):
    name = "_".join(args) or "check"
    log = attempt / (name + ".txt")
    with log.open("x", encoding="utf8") as f:
        process = subprocess.Popen(
            [sys.executable, "-X", "utf8", "-B", "-u", "-m", module, *args],
            cwd=s.ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf8",
        )
        for line in process.stdout:
            f.write(line)
            f.flush()
            print(line, end="", flush=True)
        code = process.wait()
    s.c.require(
        code == 0,
        "Offline step stopped; artifacts preserved. Send Astra the terminal output and "
        + str(log),
    )


def run():
    with lock(QUEUE / "launcher.lock"):
        plan = verify_queue()
        roster = s.c.lines(BASE / "roster.jsonl")
        keys = [r["key"] for r in roster]
        if (QUEUE / "complete.json").exists():
            done = s.c.read(QUEUE / "complete.json")
            s.c.require(
                done["queue_plan_sha256"] == s.c.sha(QUEUE / "plan.json")
                and done["independent_check_sha256"] == s.c.sha(BASE / "complete.json"),
                "Final preparation receipt changed",
            )
            for name, expected in s.c.read(BASE / "prepared.json")["outputs"].items():
                s.c.require(s.c.sha(BASE / name) == expected, "Prepared output changed")
            print(
                "OFFLINE PREPARATION ALREADY COMPLETE. No retrieval or model calls repeated.",
                flush=True,
            )
            return
        attempt = QUEUE / f"attempts/{time.time_ns()}"
        attempt.mkdir(parents=True)
        for stage in plan["stages"]:
            folder = "retrieval" if stage["mode"] == "retrieve" else "replay"
            path = BASE / f"{folder}/{stage['pool']}.jsonl"
            if completed(path, path.with_suffix(".complete.json"), keys):
                print(f"VERIFIED SKIP {stage['mode']}/{stage['pool']}", flush=True)
                continue
            child(
                "analysis.feedback.restoration.run",
                ["worker", stage["mode"], stage["pool"]],
                attempt,
            )
        if not (BASE / "prepared.json").exists():
            s.c.require(
                not (BASE / "prompts.jsonl").exists(),
                "Interrupted rendering preserved; ask Astra to inspect before recovery",
            )
            child("analysis.feedback.restoration.study", ["prompts"], attempt)
        if not (BASE / "complete.json").exists():
            child("analysis.feedback.restoration.validate", [], attempt)
        check = s.c.read(BASE / "complete.json")
        s.c.require(
            check["status"]
            == "taxonomy_prompts_independently_verified_no_classifier_calls",
            "Independent review did not pass",
        )
        s.c.require(
            check["offline_queue_plan_sha256"] == s.c.sha(QUEUE / "plan.json")
            and check["prepared_sha256"] == s.c.sha(BASE / "prepared.json"),
            "Review identity mismatch",
        )
        put(
            QUEUE / "complete.json",
            {
                "status": "offline_preparation_finished_no_model_calls",
                "queue_plan_sha256": s.c.sha(QUEUE / "plan.json"),
                "independent_check_sha256": s.c.sha(BASE / "complete.json"),
                "Qwen_calls": 0,
                "CUDA": False,
                "next": "Tell Astra preparation finished. Combined Qwen execution is a separate later command.",
            },
        )
        print(
            "ALL OFFLINE TAXONOMY PREPARATION AND CHECKS COMPLETE. Tell Astra. No Qwen run was started.",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["preflight", "freeze", "run", "worker"])
    parser.add_argument("operation", nargs="?", choices=["retrieve", "replay"])
    parser.add_argument("pool", nargs="?", choices=s.POOLS)
    args = parser.parse_args()
    with patch.object(
        socket.socket,
        "connect",
        side_effect=RuntimeError("Network forbidden in offline preparation queue"),
    ):
        if args.mode == "preflight":
            preflight()
        elif args.mode == "freeze":
            freeze()
        elif args.mode == "worker":
            assert args.operation and args.pool
            worker(args.operation, args.pool)
        else:
            run()


if __name__ == "__main__":
    main()

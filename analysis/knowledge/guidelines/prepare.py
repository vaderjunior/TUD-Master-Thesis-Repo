"""Guidelines prepare."""

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from unittest.mock import patch

os.environ.update(
    CUDA_VISIBLE_DEVICES="-1",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    ANONYMIZED_TELEMETRY="False",
    TOKENIZERS_PARALLELISM="false",
)
from analysis.knowledge.strategies import prepare as s
from analysis.knowledge.banks.build import environment

r = s.r
prompt = s.prompt
ROOT = r.ROOT
BASE = ROOT / "experiments/astra/sq1_full_kb_guideline_oracle_20260925_v1"
PLAN = BASE / "plan.json"
PLAN_SHA = "5f1183a8c1aee8b428b331f26b851d7865dfbe4982af85ad9a6f0d5bdf88abb7"
INVENTORY = BASE / "input_inventory_v1"
IMPL = BASE / "preparation_v1"
MANIFEST = IMPL / "implementation.json"
TESTS = IMPL / "tests_passed.json"
CONDITIONS = (
    "no_guidelines",
    "natural",
    "matched_rule",
    "alternative_rule",
    "filler_only",
)
IDENTITY = ("key", "item_id", "lang", "text")
SOURCE_FILES = (
    "analysis/knowledge/guidelines/prepare.py",
    "analysis/knowledge/guidelines/verify_preparation.py",
    "tests/knowledge/guidelines/test_prepare.py",
    "scripts/prepare_guideline_selection.ps1",
    "analysis/common/retrieval.py",
    "analysis/common/prompt.py",
    "analysis/knowledge/banks/build.py",
    "analysis/knowledge/strategies/prepare.py",
    "analysis/knowledge/strategies/verify.py",
    "src/hsrag/retrieve.py",
    "src/hsrag/kb.py",
    "src/hsrag/prompt.py",
    "prompts/classify_v1.txt",
    "config/taxonomy.yaml",
    ".venv/Lib/site-packages/rank_bm25.py",
)


def owned(path):
    path = Path(path).resolve()
    r.require(
        path.is_relative_to(IMPL.resolve()) and path != IMPL.resolve(),
        "Write outside guideline preparation",
    )
    return path


def save(path, value):
    path = owned(path)
    r.require(not path.exists(), "Preserve existing artifact: " + str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = owned(path.with_name(path.name + ".partial_" + str(time.time_ns())))
    with temporary.open("xb") as handle:
        handle.write(r.js(value))
        handle.flush()
        os.fsync(handle.fileno())
    temporary.rename(path)


def forbid(*args, **kwargs):
    raise RuntimeError(
        "Network/model construction or encoding forbidden in offline guideline preparation"
    )


@contextmanager
def offline():
    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket, "create_connection", forbid
    ):
        yield


@contextmanager
def lock():
    import msvcrt

    IMPL.mkdir(parents=True, exist_ok=True)
    with (IMPL / "launcher.lock").open("a+b") as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError("Another guideline preparation is running") from None
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def check_design():
    r.require(r.sha(PLAN) == PLAN_SHA, "Scientific plan changed")
    plan = r.read(PLAN)
    r.verify_hashes(plan["input_hashes"])
    r.require(
        r.sha(BASE / "analysis_plan.json") == plan["analysis_plan_sha256"],
        "Analysis changed",
    )
    r.require(
        s.files_under(ROOT / plan["source_index"]) == plan["source_index_hashes"],
        "Original index changed",
    )
    r.require(plan["environment"] == environment(), "Pinned environment changed")
    r.require(plan["conditions"] == list(CONDITIONS), "Condition order changed")
    return plan


def validate():
    plan = check_design()
    manifest = r.read(MANIFEST)
    r.require(
        manifest["plan_sha256"] == PLAN_SHA
        and manifest["tests_sha256"] == r.sha(TESTS),
        "Implementation/test binding differs",
    )
    r.verify_hashes(manifest["source_hashes"])
    return plan


def freeze():
    r.require(not MANIFEST.exists(), "Implementation already frozen")
    plan = check_design()
    tests = r.read(TESTS)
    r.require(
        tests["passed"] and tests["environment"] == plan["environment"],
        "Passing current tests required",
    )
    r.verify_hashes(tests["source_hashes"])
    save(
        MANIFEST,
        {
            "created_utc": s.utc(),
            "plan_sha256": PLAN_SHA,
            "tests_sha256": r.sha(TESTS),
            "source_hashes": {name: r.sha(ROOT / name) for name in SOURCE_FILES},
            "execution": "six-input traced/untraced smoke; 693-input first pass; one cold replay; independent audit",
            "no_network": True,
            "no_neural_encoding": True,
            "no_CUDA": True,
            "no_gold_scoring": True,
        },
    )
    print(
        json.dumps(
            {"status": "implementation_frozen", "sha256": r.sha(MANIFEST)}, indent=2
        ),
        flush=True,
    )


def load_inputs(plan):
    import numpy as np

    rows = r.rows(INVENTORY / "inputs.jsonl")
    queries = np.load(INVENTORY / "query_vectors.npy", allow_pickle=False)
    records = r.rows(r.CORE / "kb/full/records.jsonl")
    vectors = np.load(r.BASE / "full_vectors.npy", allow_pickle=False, mmap_mode="r")
    r.validate_vectors(queries, len(rows))
    r.validate_vectors(vectors, len(records))
    r.require(
        len(rows) == 693
        and len({x["key"] for x in rows}) == 693
        and (len({x["text"] for x in rows}) == 693),
        "Input roster differs",
    )
    r.require(
        len(records) == len({x["id"] for x in records}) == 71131, "KB roster differs"
    )
    r.require(
        all((x["lang"] == "en" and x["gate"] is False for x in rows)),
        "Wrong selected stratum",
    )
    return (rows, queries, records, vectors)


def rules_for(row, condition, mapping):
    r.require(condition in CONDITIONS, "Unknown condition")
    r.require(row["functionality"] in mapping, "Unmapped functionality")
    rules = mapping[row["functionality"]]
    r.require(
        set(rules) == {"matched", "alternative", "filler"}
        and len(set(rules.values())) == 3,
        "Invalid rule mapping",
    )
    return {
        "no_guidelines": [],
        "natural": None,
        "matched_rule": [rules["matched"], rules["filler"]],
        "alternative_rule": [rules["alternative"], rules["filler"]],
        "filler_only": [rules["filler"]],
    }[condition]


def forced_hit(rid, records):
    r.require(rid in records, "Missing forced rule")
    record = records[rid]
    r.require(
        record["kind"] == "guideline" and record["lang"] == "en",
        "Forced rule kind/language differs",
    )
    return {
        "id": rid,
        "text": record["text"],
        "meta": r.record_meta(record),
        "via": "oracle",
    }


def background(row, retriever, events, records):
    events.clear()
    result = retriever.retrieve(row["text"], row["lang"])
    r.require(not result.feedback, "Feedback unexpectedly retrieved")
    context = prompt.from_retrieval(result)
    r.validate_context(context, records, row["lang"])
    messages = prompt.render_messages(row["text"], row["lang"], context, "corrected")
    return {
        **{k: row[k] for k in IDENTITY},
        "pool": "full",
        "strategy": "hybrid",
        "plan_sha256": PLAN_SHA,
        "context": context,
        "context_sha256": r.signature(context),
        "messages": messages,
        "messages_sha256": r.signature(messages),
        "retrieved": prompt.hit_ids(context),
        "final_hits": [s.hit_trace(h) for h in result.all_hits()],
        "trace": deepcopy(events),
    }


def validate_arm(row, base, entry, mapping, records):
    condition = entry["condition"]
    context = entry["context"]
    r.require(
        set(context) == {"definitions", "guidelines", "example_groups"},
        "Context schema differs",
    )
    for key in ("definitions", "example_groups"):
        r.require(
            r.js(context[key]) == r.js(base["context"][key]),
            "Background changed: " + key,
        )
    ids = rules_for(row, condition, mapping)
    expected = (
        base["context"]["guidelines"]
        if ids is None
        else [forced_hit(rid, records) for rid in ids]
    )
    r.require(
        r.js(context["guidelines"]) == r.js(expected),
        "Rule text/annotation/order differs",
    )
    r.require(len({x["id"] for x in expected}) == len(expected), "Duplicate guideline")
    r.require(all((entry[k] == row[k] for k in IDENTITY)), "Input identity differs")
    if condition == "natural":
        r.require(
            r.js(entry["messages"]) == r.js(base["messages"]),
            "Natural prompt differs from original renderer",
        )


def make_entry(row, base, condition, mapping, records):
    context = deepcopy(base["context"])
    ids = rules_for(row, condition, mapping)
    if ids is not None:
        context["guidelines"] = [forced_hit(rid, records) for rid in ids]
    messages = prompt.render_messages(row["text"], row["lang"], context, "corrected")
    entry = {
        **{k: row[k] for k in IDENTITY},
        "condition": condition,
        "pool": "full",
        "strategy": "hybrid",
        "plan_sha256": PLAN_SHA,
        "input_sha256": r.signature(row),
        "background_sha256": r.signature(base),
        "context": context,
        "context_sha256": r.signature(context),
        "messages": messages,
        "messages_sha256": r.signature(messages),
        "retrieved": prompt.hit_ids(context),
    }
    validate_arm(row, base, entry, mapping, records)
    return entry


def compare(old, new, fields):
    return {name: r.js(old[name]) == r.js(new[name]) for name in fields} | {
        "entry": r.js(old) == r.js(new)
    }


def receipt(kind):
    path = IMPL / "runs" / kind / "complete.json"
    if not path.exists():
        return None
    value = r.read(path)
    r.require(
        value["plan_sha256"] == PLAN_SHA
        and value["implementation_sha256"] == r.sha(MANIFEST)
        and (value["kind"] == kind),
        "Wrong preparation receipt",
    )
    n = 6 if kind == "smoke" else 693
    r.require(
        value["inputs"] == n and value["entries"] == n * 5,
        "Incomplete preparation receipt",
    )
    r.verify_hashes(value["hashes"])
    return value


def copy_index(target, plan):
    target = owned(target)
    source_index = ROOT / plan["source_index"]
    r.require(not target.exists(), "Preserve existing index copy")
    r.require(
        shutil.disk_usage(BASE).free > 3 * 2**30,
        "At least 3 GiB free disk space required",
    )
    r.require(
        s.files_under(source_index) == plan["source_index_hashes"],
        "Source changed before copying",
    )
    target.mkdir(parents=True)
    for name, digest in plan["source_index_hashes"].items():
        source = ROOT / name
        dest = target / source.relative_to(source_index)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        r.require(r.sha(dest) == digest, "Copied index differs")
    r.require(
        s.files_under(source_index) == plan["source_index_hashes"],
        "Source changed during copying",
    )


def emit(handle, value):
    handle.write(
        json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    )
    handle.flush()


def worker(kind):
    plan = validate()
    if receipt(kind):
        print(kind + " verified and skipped", flush=True)
        return
    if kind == "prepare":
        r.require(
            receipt("smoke") is not None
            and (IMPL / "smoke_review/complete.json").exists(),
            "Audited smoke required",
        )
        r.verify_hashes(r.read(IMPL / "smoke_review/complete.json")["hashes"])
    prior = receipt("prepare") if kind == "replay" else None
    r.require(kind != "replay" or prior is not None, "First pass required")
    root = IMPL / "runs" / kind
    r.require(
        not root.exists(),
        "Prior incomplete run exists; preserve it and ask Astra to inspect",
    )
    attempt = root / ("attempt_" + str(time.time_ns()))
    attempt.mkdir(parents=True)
    start = time.perf_counter()
    save(
        attempt / "start.json",
        {
            "created_utc": s.utc(),
            "kind": kind,
            "plan_sha256": PLAN_SHA,
            "implementation_sha256": r.sha(MANIFEST),
        },
    )
    try:
        copy_index(attempt / "chroma", plan)
        rows, queries, records, vectors = load_inputs(plan)
        known = {x["id"]: x for x in records}
        mapping = r.read(INVENTORY / "rule_mapping.json")
        import chromadb
        from sentence_transformers import SentenceTransformer

        with patch.object(SentenceTransformer, "__init__", forbid), patch.object(
            SentenceTransformer, "encode", forbid
        ):
            col = chromadb.PersistentClient(
                path=str(attempt / "chroma")
            ).get_collection("hate_kb")
            origin = r.read(r.BASE / "full/origin.json")
            r.require(
                all((col.metadata.get(k) == v for k, v in origin.items())),
                "Copied collection origin differs",
            )
            maxdiff = r.verify_collection(col, records, vectors)
            encoder = r.CachedEncoder([x["text"] for x in rows], queries)
            retriever = r.make_retriever(
                col, r.CORE / "kb/full/records.jsonl", plan["retrieval"], encoder
            )
            events = s.instrument(retriever)
            plain = (
                r.make_retriever(
                    col, r.CORE / "kb/full/records.jsonl", plan["retrieval"], encoder
                )
                if kind == "smoke"
                else None
            )
            if kind == "smoke":
                by_key = {x["key"]: x for x in rows}
                rows = [by_key[k] for k in plan["smoke_keys"]]
            originals = (
                {(x["key"], x["condition"]): x for x in r.rows(ROOT / prior["prompts"])}
                if prior
                else {}
            )
            old_bases = (
                {x["key"]: x for x in r.rows(ROOT / prior["backgrounds"])}
                if prior
                else {}
            )
            backgrounds = attempt / "backgrounds.jsonl"
            prompts = attempt / "prompts.jsonl"
            timings = attempt / "timings.jsonl"
            comparisons = attempt / "comparisons.jsonl"
            bdiff = Counter()
            pdiff = Counter()
            loop_start = time.perf_counter()
            with backgrounds.open(
                "x", encoding="utf8", newline="\n"
            ) as bf, prompts.open(
                "x", encoding="utf8", newline="\n"
            ) as pf, timings.open(
                "x", encoding="utf8", newline="\n"
            ) as tf, comparisons.open(
                "x", encoding="utf8", newline="\n"
            ) as cf:
                for i, row in enumerate(rows):
                    tick = time.perf_counter()
                    base = background(row, retriever, events, known)
                    if plain is not None:
                        result = plain.retrieve(row["text"], row["lang"])
                        r.require(
                            r.js(prompt.from_retrieval(result)) == r.js(base["context"])
                            and r.js([s.hit_trace(h) for h in result.all_hits()])
                            == r.js(base["final_hits"]),
                            "Traced/untraced smoke differs; preserve attempt",
                        )
                    emit(bf, base)
                    if prior:
                        flags = compare(
                            old_bases[row["key"]],
                            base,
                            ("context", "messages", "retrieved", "trace", "final_hits"),
                        )
                        bdiff.update((k for k, equal in flags.items() if not equal))
                        emit(
                            cf,
                            {"key": row["key"], "kind": "background", "matches": flags},
                        )
                    for condition in CONDITIONS:
                        entry = make_entry(row, base, condition, mapping, known)
                        emit(pf, entry)
                        if prior:
                            flags = compare(
                                originals[row["key"], condition],
                                entry,
                                ("context", "messages", "retrieved"),
                            )
                            pdiff.update((k for k, equal in flags.items() if not equal))
                            emit(
                                cf,
                                {
                                    "key": row["key"],
                                    "condition": condition,
                                    "kind": "prompt",
                                    "matches": flags,
                                },
                            )
                    emit(tf, {"key": row["key"], "seconds": time.perf_counter() - tick})
                    if (i + 1) % 100 == 0 or i + 1 == len(rows):
                        print(
                            f"{kind.upper()}: {i + 1}/{len(rows)} texts, {(i + 1) * 5} prompts ({time.perf_counter() - loop_start:.1f}s)",
                            flush=True,
                        )
        validate()
        files = [attempt / "start.json", backgrounds, prompts, timings, comparisons]
        value = {
            "created_utc": s.utc(),
            "kind": kind,
            "status": "offline_preparation_complete_pending_independent_audit",
            "plan_sha256": PLAN_SHA,
            "implementation_sha256": r.sha(MANIFEST),
            "attempt": r.rel(attempt),
            "inputs": len(rows),
            "entries": len(rows) * 5,
            "backgrounds": r.rel(backgrounds),
            "prompts": r.rel(prompts),
            "elapsed_s": time.perf_counter() - start,
            "retrieval_loop_s": time.perf_counter() - loop_start,
            "background_replay_differences": {
                k: bdiff[k]
                for k in (
                    "context",
                    "messages",
                    "retrieved",
                    "trace",
                    "final_hits",
                    "entry",
                )
            },
            "prompt_replay_differences": {
                k: pdiff[k] for k in ("context", "messages", "retrieved", "entry")
            },
            "index_vector_max_abs_difference": maxdiff,
            "copied_index_before": plan["source_index_hashes"],
            "copied_index_after_in_process": s.files_under(attempt / "chroma"),
            "source_index_unchanged": True,
            "hashes": {r.rel(f): r.sha(f) for f in files},
            "actual_API_calls": 0,
            "new_embeddings": 0,
            "gold_scoring": False,
            "next": "Audit saved first pass and replay; retain first pass and disclose any differences, never rerun to match.",
        }
        save(root / "complete.json", value)
        print(
            json.dumps(
                {
                    k: value[k]
                    for k in (
                        "kind",
                        "inputs",
                        "entries",
                        "elapsed_s",
                        "background_replay_differences",
                        "prompt_replay_differences",
                    )
                },
                indent=2,
            ),
            flush=True,
        )
    except BaseException as exc:
        save(
            attempt / "failure.json",
            {
                "created_utc": s.utc(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "elapsed_s": time.perf_counter() - start,
            },
        )
        raise


def run(mode):
    with lock():
        validate()
        for kind in ("smoke",) if mode == "smoke" else ("prepare", "replay"):
            if receipt(kind):
                print(kind + " verified and skipped", flush=True)
                continue
            log = owned(IMPL / "logs" / (kind + "_" + str(time.time_ns()) + ".txt"))
            log.parent.mkdir(parents=True, exist_ok=True)
            with log.open("x", encoding="utf8") as handle:
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-X",
                        "utf8",
                        "-B",
                        "-u",
                        "-m",
                        __spec__.name,
                        "worker",
                        "--kind",
                        kind,
                    ],
                    cwd=ROOT,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf8",
                )
                for line in child.stdout:
                    handle.write(line)
                    handle.flush()
                    print(line, end="", flush=True)
                code = child.wait()
            r.require(
                code == 0 and receipt(kind) is not None,
                "Stopped; preserve log and ask Astra to inspect: " + r.rel(log),
            )
        print(
            "OFFLINE PROMPT PREPARATION COMPLETE. No Qwen calls; independent review remains.",
            flush=True,
        )


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["freeze", "validate", "smoke", "run", "worker"]
    )
    parser.add_argument("--kind", choices=["smoke", "prepare", "replay"])
    args = parser.parse_args()
    with offline():
        if args.mode == "freeze":
            freeze()
        elif args.mode == "validate":
            validate()
            print("Scientific and implementation plans verified; no retrieval/API.")
        elif args.mode == "worker":
            r.require(args.kind is not None, "Worker kind required")
            worker(args.kind)
        else:
            run(args.mode)


if __name__ == "__main__":
    main()

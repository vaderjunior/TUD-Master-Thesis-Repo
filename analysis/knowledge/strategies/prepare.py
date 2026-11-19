"""Strategies prepare."""

import argparse
from collections import Counter
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from types import FunctionType, MethodType
from unittest.mock import patch
from analysis.common import retrieval as r
from analysis.common import prompt as prompt

ROOT = r.ROOT
CORE = r.CORE
BASE = ROOT / "experiments/astra/sq1_full_kb_strategy_20260920_v1"
PLAN = BASE / "plan.json"
STRATEGIES = ("dense", "bm25", "hybrid")
PAIRS = (("dense", "hybrid"), ("dense", "bm25"), ("hybrid", "bm25"))
REVIEW_SHA = "48c6191f369b814f1a23b3bbf79d1b7af561e1ed603e7ba175b4dc938eda49d5"
SCOPE_COUNTS = {
    "en_dev_eval_main": 299,
    "en_dev_eval_targets": 56,
    "en_dev_eval_severity": 45,
    "en_dev_eval_sq3_types": 467,
    "en_dev_eval_types": 103,
    "de_dev_eval": 150,
    "de_legal_dev_eval": 175,
}
SOURCE_INDEX = r.BASE / "full/chroma"


def utc():
    return datetime.now(timezone.utc).isoformat()


def owned(path):
    path = Path(path).resolve()
    r.require(
        path.is_relative_to(BASE.resolve()) and path != BASE.resolve(),
        "Write outside strategy experiment",
    )
    return path


def save(path, value):
    path = owned(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = r.js(value)
    if path.exists():
        r.require(
            path.read_bytes() == payload, "Preserve existing artifact: " + str(path)
        )
        return
    temporary = owned(path.with_name(path.name + ".partial_" + str(time.time_ns())))
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.rename(path)


def forbidden(*args, **kwargs):
    raise RuntimeError(
        "Network/model fallback forbidden in offline strategy preparation"
    )


def files_under(directory):
    return {r.rel(p): r.sha(p) for p in sorted(directory.rglob("*")) if p.is_file()}


def check_sources(plan):
    r.verify_hashes(plan["input_hashes"])
    r.require(
        files_under(SOURCE_INDEX) == plan["source_index_hashes"],
        "Original index snapshot changed",
    )


def roster_check(dev, endpoints):
    r.require(len(dev) == len({x["key"] for x in dev}) == 1295, "DEV key roster")
    counts = Counter()
    for row in dev:
        r.require(len(row["scopes"]) == 1, "Unexpected multiple scopes")
        scope = row["scopes"][0]
        subset = scope["subset"]
        counts[subset] += 1
        r.require(
            subset in endpoints and row["item_id"] in endpoints[subset]["scorable_ids"],
            "Unplanned input",
        )
        expected = {
            "en_dev_eval_sq3_types": "hate_type_primary",
            "en_dev_eval_types": "hate_type_secondary",
        }.get(subset, endpoints[subset]["dimension"])
        r.require(
            scope["dimension"] == expected
            and row["key"] == row["lang"] + "/" + row["item_id"],
            "Input identity/scope",
        )
    r.require(dict(counts) == SCOPE_COUNTS, "DEV scope counts")


def prepare():
    from analysis.knowledge.banks.build import environment

    r.require(not PLAN.exists(), "Plan already frozen; validate or run")
    r.require(r.sha(CORE / "plan.json") == r.CORE_SHA, "Core plan changed")
    core = r.read(CORE / "plan.json")
    review_receipt = r.read(r.BASE / "review_complete_v2.json")
    evidence_path = ROOT / review_receipt["evidence"]
    r.require(
        r.sha(evidence_path) == review_receipt["evidence_sha256"] == REVIEW_SHA,
        "Prior review binding",
    )
    review = r.read(evidence_path)
    source_hashes = review["physical_index_hashes"]["full"]
    r.require(files_under(SOURCE_INDEX) == source_hashes, "Audited full index changed")
    endpoints = r.read(CORE / "api_execution_v1/scoring_plan.json")["endpoints"]
    dev = r.rows(CORE / "dev_inputs.jsonl")
    roster_check(dev, endpoints)
    required = [
        BASE / "PROTOCOL.md",
        CORE / "plan.json",
        CORE / "dev_inputs.jsonl",
        CORE / "kb/full/records.jsonl",
        r.PLAN,
        r.BASE / "full_vectors.npy",
        r.BASE / "query_vectors.npy",
        r.BASE / "full/origin.json",
        r.BASE / "full/prompts.jsonl",
        r.BASE / "record_vector_sources.jsonl",
        r.BASE / "query_vector_sources.jsonl",
        r.BASE / "review_complete_v2.json",
        evidence_path,
        CORE / "api_execution_v1/scoring_plan.json",
        ROOT
        / "Thesis_Log/astra_result/evidence/core_completion_summary_20260920_v1.json",
        ROOT / "analysis/common/retrieval.py",
        ROOT / "analysis/common/prompt.py",
        ROOT / "analysis/knowledge/banks/build.py",
        ROOT / "analysis/common/scoring.py",
        ROOT / "src/hsrag/retrieve.py",
        ROOT / "src/hsrag/kb.py",
        ROOT / "src/hsrag/prompt.py",
        ROOT / "prompts/classify_v1.txt",
        ROOT / "config/taxonomy.yaml",
        Path(__file__),
        ROOT / "tests/knowledge/strategies/test_prepare.py",
        ROOT / "scripts/prepare_retrieval_strategies.ps1",
    ]
    hashes = {r.rel(p): r.sha(p) for p in required}
    for name in (
        r.rel(CORE / "dev_inputs.jsonl"),
        r.rel(CORE / "kb/full/records.jsonl"),
        r.rel(r.BASE / "full_vectors.npy"),
        r.rel(r.BASE / "query_vectors.npy"),
    ):
        r.require(
            hashes[name] == review["verified_hashes"][name],
            "Inherited audited data changed",
        )
    r.require(environment() == core["environment"], "Environment changed")
    protocol = {
        "status": "protocol_frozen_no_new_strategy_outputs",
        "created_utc": utc(),
        "input_hashes": hashes,
        "source_index_hashes": source_hashes,
        "environment": environment(),
        "core_plan_sha256": r.CORE_SHA,
        "retrieval": core["retrieval"],
        "strategies": list(STRATEGIES),
        "scope_counts": SCOPE_COUNTS,
        "endpoints_inherited_masks_and_labels": endpoints,
        "analysis_roles": {
            name: (
                "focal"
                if name == "en_dev_eval_sq3_types"
                else "sensitivity" if name == "en_dev_eval_types" else "secondary"
            )
            for name in endpoints
        },
        "question": "Compare the existing dense, BM25 and hybrid retrieval packages on the cleaned full TRAIN pool, extending the original English-type experiment to both languages and all seven DEV rosters.",
        "budget": {
            "distinct_inputs": 1295,
            "prepared_prompts": 3885,
            "decoder_repeats": 3,
            "votes_per_prediction": 3,
            "future_fresh_predictions": 11655,
            "future_ordinary_responses": 34965,
            "max_repairs_per_vote": 2,
            "max_transport_attempts_per_logical_call": 5,
        },
        "future_classifier": {
            "provider": core["provider"],
            "model": core["model"],
            "temperature": core["temperature"],
            "fallback": False,
            "workers": 8,
            "all_three_strategies_fresh": True,
            "scheduling": "Interleave all three strategies and repeats within each input; alternate strategy order deterministically by input/repeat. Stage at most 2700 predictions. Freeze live runner/schedule separately after retrieval audit.",
        },
        "analysis": {
            "contrasts": [list(pair) for pair in PAIRS],
            "focal_family": "Three pairwise strategy contrasts on 467 main English type texts; two-sided paired percentile bootstrap, Bonferroni family confidence 1-0.05/3 per contrast.",
            "focal_interval_confidence": 1 - 0.05 / 3,
            "bootstrap_draws": 10000,
            "bootstrap_seed": 20260920,
            "other_endpoints": "Same three contrasts per remaining roster, descriptive pointwise 95% intervals. Separate 103-text sensitivity; no pooled overall F1 or outcome-selected endpoint.",
            "scoring": "Inherit fixed full-roster support>=10 labels, strict gate, missing vs empty rules and mean-of-three-run F1. All pairwise contrasts on an endpoint share the mask valid in all nine strategy/repeat settings. Keep every null and invalid-inclusive bounds; no favourable resampling.",
            "historical": "Original small-KB strategy results and completed core hybrid runs are historical/descriptive context, not extra repeats in this primary comparison.",
            "interpretation": "A predeclared post-selection DEV extension, not untouched confirmation or an automatic change of the main hybrid configuration. No single-bucket, MMR-only, fusion-only or pure ranking attribution.",
        },
        "implementation_qualifications": [
            "Budgets are maxima 2 definitions, 2 guidelines, 5 examples, zero feedback. BM25 retains positive-score-only behavior, may underfill and has no padding/fallback.",
            "Existing dense examples: 15 ANN candidates, MMR to 5. Existing hybrid: 30 ANN candidates, MMR to 10, BM25 depth 10, RRF k=60 to 5. Abstract dense depth 2 versus hybrid channel depth 10. These are strategy-package differences, not just adding a lexical channel.",
            "use_mmr=true applies to the dense example path, not BM25-only or abstract records. Keep source behavior unchanged.",
            "Original Chroma index never opened by a client: every attempt copies the audited physical files into its own directory and verifies hashes. Cached FP32 BGE-M3 record/query vectors; no model encoding or fresh index build.",
            "Chroma approximate searches can vary on reopening. Keep first-pass prompts, save actual entries for every replay mismatch, and require review before any live execution. Do not retry to obtain matching retrieval.",
            "Inherited Chroma singular-collection foreign-key migration qualification remains recorded in the original v2 audit; no schema/package edits.",
        ],
        "diagnostics": [
            "Exact texts, TRAIN annotations, IDs, language, budgets, rendering and signatures; preserve unannotated vs negative/empty.",
            "Per-channel candidate IDs/scores, pre-fusion MMR selection, RRF input/output ranks; tracing must reproduce untraced retrieval in tests/smoke.",
            "Per subset/strategy delivered counts, shortfalls, annotation/source/language/label mix, overlap and pairwise example-vector redundancy. No gold-based retuning.",
            "Full cold replay comparisons: entries, prompts, contexts, IDs, traces. Current hybrid messages versus original core hybrid are descriptive provenance checks.",
        ],
        "smoke_keys": list(
            dict.fromkeys(
                [dev[0]["key"], "de/detox-1389930895067467778", dev[-1]["key"]]
            )
        ),
        "preservation": "New directory only; old predictions, models, source, indexes and failures remain intact. No classifier/API calls, new gold scoring, training, TEST/benchmarks, adaptation/Priority3 or LaTeX in this preparation.",
        "next": "Focused tests and short smoke; user runs CPU preparation and one complete cold replay. Independent review precedes separate API preparation.",
    }
    save(PLAN, protocol)
    print(
        json.dumps(
            {
                "status": protocol["status"],
                "plan": r.rel(PLAN),
                "sha256": r.sha(PLAN),
                "budget": protocol["budget"],
            },
            indent=2,
        )
    )


def validate():
    from analysis.knowledge.banks.build import environment

    plan = r.read(PLAN)
    check_sources(plan)
    r.require(plan["environment"] == environment(), "Pinned environment changed")
    r.require(plan["strategies"] == list(STRATEGIES), "Strategy contract changed")
    return plan


def validate_context(context, records, language, strategy):
    r.require(strategy in STRATEGIES, "Unknown strategy")
    examples = [hit for _, group in context["example_groups"] for hit in group]
    all_ids = []
    for kind, hits, budget in [
        ("definition", context["definitions"], 2),
        ("guideline", context["guidelines"], 2),
        ("example", examples, 5),
    ]:
        r.require(
            len(hits) <= budget and (strategy == "bm25" or len(hits) == budget),
            "Retrieval budget violation",
        )
        for hit in hits:
            r.require(hit["id"] in records, "Unknown retrieved record")
            record = records[hit["id"]]
            expected = r.record_meta(record)
            r.require(
                record["kind"] == kind and hit["text"] == record["text"],
                "Retrieved kind/text differs",
            )
            actual = {k: v for k, v in hit["meta"].items() if k != "_ranks"}
            r.require(actual == expected, "Retrieved annotation differs")
            r.require(
                kind != "example" or record["lang"] == language,
                "Cross-language example",
            )
            all_ids.append(hit["id"])
    r.require(len(all_ids) == len(set(all_ids)), "Duplicate retrieved record")
    return examples


def hit_trace(hit):
    r.require(math.isfinite(float(hit.score)), "Nonfinite retrieval score")
    return {
        "id": hit.id,
        "kind": hit.kind,
        "score": float(hit.score),
        "via": hit.via,
        "ranks": deepcopy(hit.meta.get("_ranks", {})),
    }


def instrument(retriever):
    """Trace existing functions without changing candidate selection or their globals."""
    from src.hsrag.retrieve import Retriever, mmr_select, rrf_fuse

    events = []
    original_dense, original_bm25 = (retriever._query_kind, retriever._bm25_kind)

    def dense(qvec, where, k, via, want_vecs=False):
        hits = original_dense(qvec, where, k, via, want_vecs)
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
        hits = original_bm25(query, bucket_key, k)
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
        result = mmr_select(qvec, hits, lam, top_n)
        events.append(
            {
                "operation": "mmr",
                "lambda": lam,
                "top_n": top_n,
                "input_ids": before,
                "hits": [hit_trace(h) for h in result],
            }
        )
        return result

    def fuse(rank_lists, k=60, top_n=5):
        inputs = {
            name: [hit_trace(h) for h in hits] for name, hits in rank_lists.items()
        }
        result = rrf_fuse(rank_lists, k, top_n)
        events.append(
            {
                "operation": "rrf",
                "k": k,
                "top_n": top_n,
                "channels": inputs,
                "hits": [hit_trace(h) for h in result],
            }
        )
        return result

    retriever._query_kind, retriever._bm25_kind = (dense, lexical)
    cloned = FunctionType(
        Retriever.retrieve.__code__,
        {**Retriever.retrieve.__globals__, "mmr_select": mmr, "rrf_fuse": fuse},
    )
    retriever.retrieve = MethodType(cloned, retriever)
    return events


def make_entry(row, strategy, retriever, events, known):
    retriever.cfg["strategy"] = strategy
    events.clear()
    result = retriever.retrieve(row["text"], row["lang"])
    r.require(not result.feedback, "Feedback unexpectedly retrieved")
    context = prompt.from_retrieval(result)
    validate_context(context, known, row["lang"], strategy)
    messages = prompt.render_messages(row["text"], row["lang"], context, "corrected")
    return {
        **row,
        "pool": "full",
        "strategy": strategy,
        "condition": "rag_" + strategy,
        "plan_sha256": r.sha(PLAN),
        "context": context,
        "context_sha256": r.signature(context),
        "messages": messages,
        "messages_sha256": r.signature(messages),
        "retrieved": prompt.hit_ids(context),
        "final_hits": [hit_trace(h) for h in result.all_hits()],
        "trace": deepcopy(events),
    }


def comparison(old, new):
    return {
        key: old[key] == new[key]
        for key in ("context", "messages", "retrieved", "trace", "final_hits")
    } | {"entry": old == new}


def verified_receipt(kind):
    path = BASE / "runs" / kind / "complete.json"
    if not path.exists():
        return None
    receipt = r.read(path)
    r.require(
        receipt["plan_sha256"] == r.sha(PLAN) and receipt["kind"] == kind,
        "Wrong completion receipt",
    )
    expected = 9 if kind == "smoke" else 3885
    r.require(receipt["entries"] == expected, "Incomplete preparation receipt")
    r.verify_hashes(receipt["hashes"])
    return receipt


def copy_index(target, plan):
    target = owned(target)
    r.require(not target.exists(), "Preserve existing index copy")
    r.require(
        files_under(SOURCE_INDEX) == plan["source_index_hashes"],
        "Source index changed before copy",
    )
    r.require(
        shutil.disk_usage(BASE).free > 3 * 2**30, "At least 3 GiB free space required"
    )
    target.mkdir(parents=True)
    for name, expected in plan["source_index_hashes"].items():
        source = ROOT / name
        destination = target / source.relative_to(SOURCE_INDEX)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        r.require(r.sha(destination) == expected, "Index copy differs")
    r.require(
        files_under(SOURCE_INDEX) == plan["source_index_hashes"],
        "Source index changed while copying",
    )


def worker(kind):
    import chromadb

    plan = validate()
    if verified_receipt(kind):
        print(kind + " already complete; verified and skipped", flush=True)
        return
    if kind == "prepare":
        r.require(verified_receipt("smoke") is not None, "Smoke first")
    previous = verified_receipt("prepare") if kind == "replay" else None
    if kind == "replay":
        r.require(previous is not None, "First pass missing")
    root = BASE / "runs" / kind
    attempt = owned(root / ("attempt_" + str(time.time_ns())))
    attempt.mkdir(parents=True)
    save(
        attempt / "start.json",
        {"created_utc": utc(), "kind": kind, "plan_sha256": r.sha(PLAN)},
    )
    copy_index(attempt / "chroma", plan)
    records, vectors, dev, queries = r.data("full")
    roster_check(dev, plan["endpoints_inherited_masks_and_labels"])
    all_dev = dev
    if kind == "smoke":
        by_key = {x["key"]: x for x in dev}
        dev = [by_key[k] for k in plan["smoke_keys"]]
    col = chromadb.PersistentClient(path=str(attempt / "chroma")).get_collection(
        "hate_kb"
    )
    origin = r.read(r.BASE / "full/origin.json")
    r.require(
        all((col.metadata.get(k) == v for k, v in origin.items())),
        "Copied collection origin differs",
    )
    maxdiff = r.verify_collection(col, records, vectors)
    encoder = r.CachedEncoder([x["text"] for x in all_dev], queries)
    retriever = r.make_retriever(
        col, CORE / "kb/full/records.jsonl", plan["retrieval"], encoder
    )
    events = instrument(retriever)
    known = {x["id"]: x for x in records}
    reference = (
        {(x["key"], x["strategy"]): x for x in r.rows(ROOT / previous["prompts"])}
        if previous
        else {}
    )
    historical = {x["key"]: x for x in r.rows(r.BASE / "full/prompts.jsonl")}
    output = attempt / ("comparisons.jsonl" if kind == "replay" else "prompts.jsonl")
    history_output = attempt / "core_hybrid_comparisons.jsonl"
    timing_output = attempt / "timings.jsonl"
    stats, mismatches, history, count = ({}, Counter(), Counter(), 0)
    started = time.perf_counter()
    with output.open("x", encoding="utf8", newline="\n") as handle, history_output.open(
        "x", encoding="utf8", newline="\n"
    ) as history_handle, timing_output.open(
        "x", encoding="utf8", newline="\n"
    ) as timing_handle:

        def emit(stream, value):
            stream.write(
                json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
            stream.flush()

        for i, row in enumerate(dev):
            order = STRATEGIES[i % 3 :] + STRATEGIES[: i % 3]
            for strategy in order:
                tick = time.perf_counter()
                entry = make_entry(row, strategy, retriever, events, known)
                emit(
                    timing_handle,
                    {
                        "key": row["key"],
                        "strategy": strategy,
                        "seconds": time.perf_counter() - tick,
                    },
                )
                if kind == "smoke":
                    plain = r.make_retriever(
                        col,
                        CORE / "kb/full/records.jsonl",
                        {**plan["retrieval"], "strategy": strategy},
                        encoder,
                    )
                    actual = plain.retrieve(row["text"], row["lang"])
                    r.require(
                        prompt.from_retrieval(actual) == entry["context"]
                        and [hit_trace(h) for h in actual.all_hits()]
                        == entry["final_hits"],
                        "Traced/untraced smoke differs; preserve attempt",
                    )
                if kind == "replay":
                    check = comparison(reference[row["key"], strategy], entry)
                    record = {"key": row["key"], "strategy": strategy, "matches": check}
                    for key, equal in check.items():
                        mismatches[key] += not equal
                    if not check["entry"]:
                        record["actual_entry"] = entry
                    emit(handle, record)
                else:
                    emit(handle, entry)
                if strategy == "hybrid":
                    old = historical[row["key"]]
                    flags = {
                        k: old[k] == entry[k]
                        for k in ("context", "messages", "retrieved")
                    }
                    for key, equal in flags.items():
                        history[key] += not equal
                    emit(
                        history_handle,
                        {
                            "key": row["key"],
                            "matches": flags,
                            "old_messages_sha256": old["messages_sha256"],
                            "new_messages_sha256": entry["messages_sha256"],
                        },
                    )
                bucket = stats.setdefault(
                    row["scopes"][0]["subset"] + "/" + strategy, Counter()
                )
                bucket["inputs"] += 1
                for name, hits in [
                    ("definitions", entry["context"]["definitions"]),
                    ("guidelines", entry["context"]["guidelines"]),
                    (
                        "examples",
                        [
                            h
                            for _, group in entry["context"]["example_groups"]
                            for h in group
                        ],
                    ),
                ]:
                    bucket[name + "_slots"] += len(hits)
                    bucket[name + "_empty_inputs"] += not hits
                count += 1
            if (i + 1) % 100 == 0 or i + 1 == len(dev):
                print(
                    f"{kind.upper()}: {i + 1}/{len(dev)} texts, {count} entries ({time.perf_counter() - started:.0f}s)",
                    flush=True,
                )
    r.require(count == len(dev) * 3, "Incomplete output matrix")
    check_sources(plan)
    receipt = {
        "completed_utc": utc(),
        "status": "retrieval_diagnostics_complete_pending_review",
        "kind": kind,
        "plan_sha256": r.sha(PLAN),
        "attempt": r.rel(attempt),
        "entries": count,
        "index_vector_max_abs_difference": maxdiff,
        "copied_index_before": plan["source_index_hashes"],
        "copied_index_after_in_process": files_under(attempt / "chroma"),
        "source_index_unchanged": True,
        "replay_differences": dict(mismatches),
        "historical_hybrid_differences": dict(history),
        "bucket_counts": {k: dict(v) for k, v in stats.items()},
        "actual_API_calls": 0,
        "gold_scoring": False,
        "hashes": {
            r.rel(p): r.sha(p)
            for p in [attempt / "start.json", output, history_output, timing_output]
        },
        "next": "Independent review; no automatic API execution. Preserve all first-pass prompts and every replay mismatch.",
    }
    if kind != "replay":
        receipt["prompts"] = r.rel(output)
    save(root / "complete.json", receipt)
    print(
        json.dumps(
            {
                "kind": kind,
                "entries": count,
                "replay_differences": dict(mismatches),
                "historical_hybrid_differences": dict(history),
            },
            indent=2,
        ),
        flush=True,
    )


@contextmanager
def lock():
    import msvcrt

    BASE.mkdir(parents=True, exist_ok=True)
    with (BASE / "launcher.lock").open("a+b") as handle:
        if handle.seek(0, 2) == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError("Another strategy preparation is running") from None
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def run(mode):
    with lock():
        validate()
        for kind in ("smoke",) if mode == "smoke" else ("prepare", "replay"):
            if verified_receipt(kind):
                print(kind + " verified and skipped", flush=True)
                continue
            log = owned(BASE / "logs" / (kind + "_" + str(time.time_ns()) + ".txt"))
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
                for text in child.stdout:
                    handle.write(text)
                    handle.flush()
                    print(text, end="", flush=True)
                code = child.wait()
            r.require(
                code == 0,
                "Stopped; keep attempt/log and ask Astra to inspect: " + r.rel(log),
            )
            r.require(verified_receipt(kind) is not None, "Missing child receipt")
        print(
            (
                "SHORT SMOKE COMPLETE."
                if mode == "smoke"
                else "ALL STRATEGY PROMPTS AND COLD REPLAY COMPLETE. Tell Astra to inspect; no Qwen calls were made."
            ),
            flush=True,
        )


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["prepare", "validate", "smoke", "run", "worker"]
    )
    parser.add_argument("--kind", choices=["smoke", "prepare", "replay"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ):
        if args.mode == "prepare":
            prepare()
        elif args.mode == "validate":
            validate()
            print(
                "Frozen strategy plan and source index verified. No retrieval/API calls."
            )
        elif args.mode == "worker":
            r.require(args.kind is not None, "Worker kind required")
            worker(args.kind)
        else:
            run(args.mode)


if __name__ == "__main__":
    main()

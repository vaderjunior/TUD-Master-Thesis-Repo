"""Allocation prepare."""

import argparse
from collections import Counter
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
from analysis.knowledge.components import prepare as previous
from analysis.knowledge.strategies import prepare as strategy

a = previous.a
r = strategy.r
ROOT = previous.ROOT
CORE = previous.CORE
BASE = ROOT / "experiments/astra/sq1_full_kb_allocation_20260924_v1"
PLAN = BASE / "plan.json"
CONDITIONS = ("flat2", "flat7", "proportional7")
ENDPOINT_ORDER = previous.ENDPOINT_ORDER
QUOTAS = {"hate": 1, "target_group": 2, "hate_type": 2, "severity": 1, "legal": 1}
COMPLETE = (
    ROOT / "Thesis_Log/astra_result/evidence/sq1_components_completion_20260924_v1.json"
)
COMPLETE_SHA = "521f9f604efdce8f9383766aa5c68b87edc6de8988d6111ed6e9845c9679d973"
SOURCE_FILES = [
    "analysis/knowledge/allocation/prepare.py",
    "analysis/knowledge/allocation/verify_legacy_preparation.py",
    "tests/knowledge/allocation/test_prepare.py",
    "scripts/prepare_definition_allocation.ps1",
    "analysis/knowledge/strategies/prepare.py",
    "analysis/knowledge/strategies/verify.py",
    "analysis/common/retrieval.py",
    "analysis/common/prompt.py",
    "src/hsrag/retrieve.py",
    "src/hsrag/kb.py",
    "src/hsrag/prompt.py",
    "prompts/classify_v1.txt",
    "config/taxonomy.yaml",
]
SOURCE_FILES = list(dict.fromkeys(SOURCE_FILES + previous.SOURCE_FILES))


def owned(path):
    path = Path(path).resolve()
    a.require(
        path.is_relative_to(BASE.resolve()) and path != BASE.resolve(),
        "Write outside allocation study",
    )
    return path


def save(path, value):
    path = owned(path)
    a.save_new(path, value)


def forbid(*args, **kwargs):
    raise RuntimeError("Offline allocation preparation only")


def definitions_from_trace(trace, records, k):
    """Use the original ranker, with an independent implementation in the audit."""
    from src.hsrag.retrieve import Hit, rrf_fuse

    a.require(
        k in (2, 7)
        and trace[0]["operation"] == "dense_query"
        and (trace[0]["where"] == {"kind": "definition"})
        and (trace[0]["k"] == 10)
        and (trace[1]["operation"] == "bm25_query")
        and (trace[1]["bucket"] == "definition")
        and (trace[1]["k"] == 10)
        and (trace[2]["operation"] == "rrf"),
        "Wrong flat source trace",
    )
    channels = {}
    for name, event in zip(("dense", "bm25"), trace[:2]):
        channels[name] = [
            Hit(
                id=h["id"],
                kind="definition",
                text=records[h["id"]]["text"],
                meta=r.record_meta(records[h["id"]]),
                score=h["score"],
                via=name,
            )
            for h in event["hits"]
        ]
    hits = rrf_fuse(channels, k=60, top_n=k)
    return [{"id": h.id, "text": h.text, "meta": h.meta, "via": h.via} for h in hits]


def validate_context(source, context, condition, records):
    a.require(condition in CONDITIONS, "Unknown allocation condition")
    a.require(
        a.signature(context["example_groups"])
        == a.signature(source["context"]["example_groups"])
        and a.signature(context["guidelines"])
        == a.signature(source["context"]["guidelines"]),
        "Background changed",
    )
    defs = context["definitions"]
    n = 2 if condition == "flat2" else 7
    a.require(
        len(defs) == len({h["id"] for h in defs}) == n, "Definition count/duplicate"
    )
    for hit in defs:
        record = records[hit["id"]]
        a.require(
            record["kind"] == "definition"
            and record["text"] == hit["text"]
            and (
                a.signature({k: v for k, v in hit["meta"].items() if k != "_ranks"})
                == a.signature(r.record_meta(record))
            ),
            "Definition source changed",
        )
    if condition == "flat2":
        a.require(context == source["context"], "Flat2 context differs")
    if condition == "flat7":
        a.require(
            defs[:2] == source["context"]["definitions"], "Flat7 does not extend flat2"
        )
    if condition == "proportional7":
        dims = [h["meta"]["dimension"] for h in defs]
        a.require(
            Counter(dims) == QUOTAS
            and dims == [dim for dim, n in QUOTAS.items() for _ in range(n)],
            "Proportional quotas/order differ",
        )


def entry(source, condition, defs, trace, records):
    context = deepcopy(source["context"])
    context["definitions"] = deepcopy(defs)
    validate_context(source, context, condition, records)
    messages = previous.prompt.render_messages(
        source["text"], source["lang"], context, "corrected"
    )
    if condition == "flat2":
        a.require(messages == source["messages"], "Flat2 original prompt differs")
    a.body(messages)
    return {
        **{k: source[k] for k in ("key", "item_id", "lang", "text", "scopes")},
        "condition": condition,
        "pool": "full",
        "strategy": "hybrid",
        "source_entry_sha256": a.signature(source),
        "source_messages_sha256": source["messages_sha256"],
        "context": context,
        "context_sha256": a.signature(context),
        "messages": messages,
        "messages_sha256": a.signature(messages),
        "retrieved": previous.prompt.hit_ids(context),
        "definition_trace": deepcopy(trace),
        "plan_sha256": a.digest(PLAN),
    }


def analysis_plan(old):
    endpoints = {}
    focal = ["en_dev_eval_main", "en_dev_eval_sq3_types"]
    for name, ep in old["endpoints_inherited_masks_and_labels"].items():
        endpoints[name] = {
            **ep,
            "inherited_core_role": ep["role"],
            "role": (
                "focal"
                if name in focal
                else "sensitivity" if name == "en_dev_eval_types" else "secondary"
            ),
        }
    return {
        "endpoints": endpoints,
        "conditions": list(CONDITIONS),
        "repeats": [0, 1, 2],
        "contrasts": {
            "proportional_minus_flat2": {"proportional7": 1, "flat2": -1},
            "flat7_minus_flat2": {"flat7": 1, "flat2": -1},
            "proportional_minus_flat7": {"proportional7": 1, "flat7": -1},
        },
        "focal_endpoints": focal,
        "focal_family_size": 6,
        "focal_family_confidence": 1 - 0.05 / 6,
        "descriptive_confidence": 0.95,
        "bootstrap_draws": 10000,
        "bootstrap_seed": 20260924,
        "validity": "Shared nine-setting valid mask per endpoint; own-valid results, null counts, full-roster failure bounds retained.",
        "estimator": "Mean of three repeat macro-F1 scores, sample SD/range/all repeats; fixed inherited full-roster supported labels; no repeat pooling.",
        "bootstrap": "Paired item draws shared across all conditions/repeats. Independently verify each setting/draw using frequency weighting; labels stay fixed in zero-support draws.",
        "no_partial_scoring": True,
        "limits": "Previously inspected DEV. Conditional on fixed contexts/three decoder repeats; no equivalence, automatic promotion or pure guaranteed-hate-slot mechanism claim.",
    }


def prepare():
    a.require(not PLAN.exists(), "Allocation plan already frozen")
    tests_path = BASE / "development/tests_passed.json"
    tests = a.read(tests_path)
    a.require(
        tests["passed"] and tests["environment"] == previous.core.environment(),
        "Passing preparation tests required",
    )
    r.verify_hashes(tests["source_hashes"])
    old, dev, sources, paths = previous.sources()
    a.require(a.digest(COMPLETE) == COMPLETE_SHA, "Component completion changed")
    records = r.rows(CORE / "kb/full/records.jsonl")
    counts = Counter((x["dimension"] for x in records if x["kind"] == "definition"))
    a.require(
        list(counts) == list(QUOTAS)
        and {d: max(1, round(n * 0.3)) for d, n in counts.items()} == QUOTAS,
        "Inherited dimension budgets differ",
    )
    index_hashes = old["source_index_hashes"]
    a.require(
        strategy.files_under(strategy.SOURCE_INDEX) == index_hashes,
        "Original index changed",
    )
    vector_plan = a.read(r.PLAN)
    for name, h in vector_plan["generated_hashes"].items():
        a.require(a.digest(ROOT / name) == h, "Cached vector artifact changed")
    save(BASE / "analysis_plan.json", analysis_plan(old))
    paths += [
        BASE / "PROTOCOL.md",
        tests_path,
        COMPLETE,
        r.PLAN,
        r.BASE / "full/origin.json",
        *[ROOT / name for name in vector_plan["generated_hashes"]],
        ROOT / "Thesis_Log/astra_result/evidence/guideline_oracle_20260909_v1.json",
        ROOT / "experiments/manifests/defprop_r1.yaml",
        ROOT / "experiments/manifests/defflat7_main_r1.yaml",
    ]
    plan = {
        "created_utc": a.utc(),
        "status": "allocation_protocol_frozen_no_model_calls",
        "source_hashes": {name: a.digest(ROOT / name) for name in SOURCE_FILES},
        "input_hashes": {r.rel(path): a.digest(path) for path in paths},
        "source_index_hashes": index_hashes,
        "environment": previous.core.environment(),
        "analysis_sha256": a.digest(BASE / "analysis_plan.json"),
        "retrieval": old["retrieval"],
        "definition_counts": dict(counts),
        "quotas": QUOTAS,
        "scope_counts": previous.SCOPE_COUNTS,
        "smoke_keys": [
            next((x["key"] for x in dev if x["scopes"][0]["subset"] == name))
            for name in ENDPOINT_ORDER
        ],
        "source_prompts": a.read(previous.SOURCE / "retrieval_acceptance_v1.json")[
            "prompts"
        ],
        "budget": {
            "inputs": 1295,
            "conditions": 3,
            "prompt_payloads": 3885,
            "decoder_repeats": 3,
            "votes": 3,
            "fresh_predictions": 11655,
            "ordinary_responses": 34965,
        },
        "qualifications": [
            "Frozen examples/guidelines; flat2 exact old prompt, flat7 expanded from saved depth10 ranks, proportional definitions queried in a copied original index.",
            "Proportional per-dimension BM25 and order are existing package semantics; same record count does not isolate length/order/content or a single rule.",
            "Preserve all first-pass prompts and replay differences. Earlier implicit_hate-4207 ANN qualification retained.",
            "No training, model API, credentials, new gold scoring, TEST/HateCheck execution, Priority3 or LaTeX.",
        ],
        "next": "Smoke, complete preparation and cold replay, then independent audit. Separate API preparation only afterward.",
    }
    save(PLAN, plan)
    print(
        json.dumps(
            {"plan_sha256": a.digest(PLAN), "budget": plan["budget"], "quotas": QUOTAS},
            indent=2,
        ),
        flush=True,
    )


def validate():
    plan = a.read(PLAN)
    r.verify_hashes(plan["source_hashes"])
    r.verify_hashes(plan["input_hashes"])
    a.require(
        plan["environment"] == previous.core.environment()
        and a.digest(BASE / "analysis_plan.json") == plan["analysis_sha256"],
        "Environment/analysis changed",
    )
    a.require(
        strategy.files_under(strategy.SOURCE_INDEX) == plan["source_index_hashes"],
        "Source physical index changed",
    )
    return plan


def receipt(kind):
    path = BASE / "runs" / kind / "complete.json"
    if not path.exists():
        return None
    got = a.read(path)
    a.require(
        got["kind"] == kind and got["plan_sha256"] == a.digest(PLAN),
        "Receipt identity differs",
    )
    r.verify_hashes(got["hashes"])
    a.require(got["entries"] == (21 if kind == "smoke" else 3885), "Receipt incomplete")
    return got


def copy_index(target, plan):
    target = owned(target)
    a.require(not target.exists(), "Preserve index attempt")
    a.require(shutil.disk_usage(BASE).free > 3 * 2**30, "Need 3 GiB free")
    target.mkdir(parents=True)
    for name, h in plan["source_index_hashes"].items():
        src = ROOT / name
        dst = target / src.relative_to(strategy.SOURCE_INDEX)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        a.require(a.digest(dst) == h, "Copied index differs")


def worker(kind):
    import chromadb

    plan = validate()
    a.require(receipt(kind) is None, "Completed attempt must not be rerun")
    if kind != "smoke":
        a.require(receipt("smoke") is not None, "Smoke first")
        reviewed = a.read(BASE / "smoke_review_v1/complete.json")
        a.require(
            reviewed["status"] == "allocation_smoke_independently_checked"
            and a.digest(ROOT / reviewed["evidence"]) == reviewed["evidence_sha256"],
            "Smoke audit required",
        )
    prior = receipt("prepare") if kind == "replay" else None
    if kind == "replay":
        a.require(prior is not None, "First pass required")
    attempt = owned(BASE / "runs" / kind / f"attempt_{time.time_ns()}")
    attempt.mkdir(parents=True)
    save(
        attempt / "start.json",
        {"created_utc": a.utc(), "kind": kind, "plan_sha256": a.digest(PLAN)},
    )
    copy_index(attempt / "chroma", plan)
    recs, vectors, dev, queries = r.data("full")
    records = {x["id"]: x for x in recs}
    all_dev = dev
    sources = {
        x["key"]: x
        for x in r.rows(ROOT / plan["source_prompts"])
        if x["strategy"] == "hybrid"
    }
    if kind == "smoke":
        dev = [x for x in dev if x["key"] in plan["smoke_keys"]]
    col = chromadb.PersistentClient(path=str(attempt / "chroma")).get_collection(
        "hate_kb"
    )
    a.require(
        all(
            (
                col.metadata.get(k) == v
                for k, v in a.read(r.BASE / "full/origin.json").items()
            )
        ),
        "Wrong copied index identity",
    )
    maxdiff = r.verify_collection(col, recs, vectors)
    cfg = {
        **plan["retrieval"],
        "strategy": "hybrid",
        "k_definitions_mode": "proportional",
        "k_definitions_per_label": 0.3,
        "k_definitions_per_dimension": 1,
        "k_examples": 0,
        "k_guidelines": 0,
        "k_examples_feedback": 0,
    }
    encoder = r.CachedEncoder([x["text"] for x in all_dev], queries)
    retriever = r.make_retriever(col, CORE / "kb/full/records.jsonl", cfg, encoder)
    events = strategy.instrument(retriever)
    plain = (
        r.make_retriever(col, CORE / "kb/full/records.jsonl", cfg, encoder)
        if kind == "smoke"
        else None
    )
    reference = (
        {(x["key"], x["condition"]): x for x in r.rows(ROOT / prior["prompts"])}
        if prior
        else {}
    )
    out = attempt / ("comparisons.jsonl" if prior else "prompts.jsonl")
    timing = attempt / "timings.jsonl"
    n = 0
    differences = Counter()
    started = time.perf_counter()
    with out.open("x", encoding="utf8", newline="\n") as stream, timing.open(
        "x", encoding="utf8", newline="\n"
    ) as times:

        def emit(handle, value):
            handle.write(
                json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
            handle.flush()

        for index, row in enumerate(dev):
            tick = time.perf_counter()
            source = sources[row["key"]]
            flat2 = definitions_from_trace(source["trace"], records, 2)
            flat7 = definitions_from_trace(source["trace"], records, 7)
            a.require(
                flat2 == source["context"]["definitions"],
                "Source flat ranks do not reproduce",
            )
            events.clear()
            result = retriever.retrieve(row["text"], row["lang"])
            prop = previous.prompt.from_retrieval(result)["definitions"]
            a.require(
                not result.examples
                and (not result.guidelines)
                and (not result.feedback),
                "Unintended retrieval bucket",
            )
            if plain:
                actual = previous.prompt.from_retrieval(
                    plain.retrieve(row["text"], row["lang"])
                )["definitions"]
                a.require(
                    prop == actual,
                    "Traced/untraced proportional mismatch; preserve smoke",
                )
            for condition, defs, trace in [
                ("flat2", flat2, source["trace"][:3]),
                ("flat7", flat7, source["trace"][:3]),
                ("proportional7", prop, events),
            ]:
                got = entry(source, condition, defs, trace, records)
                if prior:
                    old = reference[got["key"], condition]
                    flags = {
                        k: old[k] == got[k]
                        for k in (
                            "context",
                            "messages",
                            "retrieved",
                            "definition_trace",
                        )
                    }
                    flags["entry"] = old == got
                    saved = {
                        "key": got["key"],
                        "condition": condition,
                        "matches": flags,
                    }
                    for key, same in flags.items():
                        differences[key] += int(not same)
                    if not flags["entry"]:
                        saved["actual_entry"] = got
                    emit(stream, saved)
                else:
                    emit(stream, got)
                n += 1
            emit(times, {"key": row["key"], "seconds": time.perf_counter() - tick})
            if (index + 1) % 100 == 0 or index + 1 == len(dev):
                print(
                    f"{kind}: {index + 1}/{len(dev)} inputs ({time.perf_counter() - started:.1f}s)",
                    flush=True,
                )
    a.require(n == len(dev) * 3, "Missing preparation entry")
    validate()
    made = {
        "created_utc": a.utc(),
        "kind": kind,
        "status": "allocation_prepared_pending_independent_review",
        "plan_sha256": a.digest(PLAN),
        "entries": n,
        "inputs": len(dev),
        "attempt": r.rel(attempt),
        "index_vector_max_abs_difference": maxdiff,
        "wall_seconds_after_index_validation": time.perf_counter() - started,
        "replay_differences": dict(differences),
        "copied_index_after_in_process": strategy.files_under(attempt / "chroma"),
        "source_index_unchanged": True,
        "actual_API_calls": 0,
        "new_gold_scoring": False,
        "hashes": {
            r.rel(path): a.digest(path)
            for path in (attempt / "start.json", out, timing)
        },
    }
    if not prior:
        made["prompts"] = r.rel(out)
    save(BASE / "runs" / kind / "complete.json", made)
    print(
        json.dumps(
            {
                k: made[k]
                for k in (
                    "kind",
                    "inputs",
                    "entries",
                    "wall_seconds_after_index_validation",
                    "replay_differences",
                )
            },
            indent=2,
        ),
        flush=True,
    )


def run(mode):
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
            raise RuntimeError("Allocation preparation already running") from None
        try:
            validate()
            for kind in ("smoke",) if mode == "smoke" else ("prepare", "replay"):
                if receipt(kind):
                    print(kind + " verified and skipped", flush=True)
                    continue
                log = owned(BASE / "logs" / f"{kind}_{time.time_ns()}.txt")
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("x", encoding="utf8") as output:
                    process = subprocess.Popen(
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
                    for line in process.stdout:
                        output.write(line)
                        output.flush()
                        print(line, end="", flush=True)
                    a.require(
                        process.wait() == 0,
                        "Preparation stopped; retain attempt/log for review",
                    )
                a.require(receipt(kind) is not None, "Missing completion receipt")
            print(
                "ALLOCATION PREPARATION FINISHED. No Qwen calls; saved outputs require independent review.",
                flush=True,
            )
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["prepare", "validate", "smoke", "run", "worker"]
    )
    parser.add_argument("--kind", choices=["smoke", "prepare", "replay"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket, "create_connection", forbid
    ), patch.object(previous.core, "live_client", forbid):
        if args.mode == "prepare":
            prepare()
        elif args.mode == "validate":
            validate()
            print("Allocation frozen inputs/index verified; no calls.")
        elif args.mode == "worker":
            a.require(args.kind is not None, "Worker kind required")
            worker(args.kind)
        else:
            run(args.mode)


if __name__ == "__main__":
    main()

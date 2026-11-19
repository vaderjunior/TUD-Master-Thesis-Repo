"""Allocation verify legacy preparation."""

import argparse
from collections import Counter, defaultdict
import io
import json
import math
from pathlib import Path
import socket
import time
import unittest
from unittest.mock import patch
import numpy as np
from analysis.knowledge.allocation import prepare as p
from analysis.knowledge.strategies import verify as independent

a = p.a
r = p.r
ROOT = p.ROOT


def tests():
    a.require(
        not p.PLAN.exists() and (not (p.BASE / "development/tests_passed.json").exists()),
        "Preserve frozen tests",
    )
    attempt = p.BASE / "development" / f"test_attempt_{time.time_ns()}"
    hashes = {name: a.digest(ROOT / name) for name in p.SOURCE_FILES}
    p.save(
        attempt / "sources.json",
        {
            name: (ROOT / name).read_text(encoding="utf8")
            for name in p.SOURCE_FILES
            if "allocation" in name
        },
    )
    stream = io.StringIO()
    tick = time.perf_counter()
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.knowledge.allocation.test_prepare")
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = {
        "created_utc": a.utc(),
        "passed": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "output": stream.getvalue(),
        "elapsed_s": time.perf_counter() - tick,
        "source_hashes": hashes,
        "environment": p.previous.core.environment(),
        "actual_API_calls": 0,
    }
    p.save(attempt / "tests.json", report)
    print(stream.getvalue(), flush=True)
    a.require(result.wasSuccessful(), "Allocation tests failed; preserve pre-freeze snapshot")
    r.verify_hashes(hashes)
    p.save(p.BASE / "development/tests_passed.json", report)


def identity(row):
    return (row["key"], row["condition"])


def replay_entry(original, saved):
    a.require(identity(original) == identity(saved), "Replay identity differs")
    fields = ("context", "messages", "retrieved", "definition_trace", "entry")
    flags = saved["matches"]
    a.require(
        set(flags) == set(fields) and all((type(v) is bool for v in flags.values())),
        "Replay flags differ",
    )
    if flags["entry"]:
        a.require(all(flags.values()) and "actual_entry" not in saved, "Hidden replay change")
        return original
    a.require("actual_entry" in saved, "Changed replay lacks actual entry")
    actual = saved["actual_entry"]
    a.require(identity(actual) == identity(original), "Changed replay identity")
    a.require(
        flags
        == {**{k: original[k] == actual[k] for k in fields[:-1]}, "entry": original == actual},
        "Incorrect replay flags",
    )
    return actual


def audit_trace(entry, source, records, vectors, queries, lexicons, counts):
    isprop = entry["condition"] == "proportional7"
    trace = entry["definition_trace"]
    out = []
    specs = (
        [(dim, k) for dim, k in p.QUOTAS.items()]
        if isprop
        else [(None, 7 if entry["condition"] == "flat7" else 2)]
    )
    a.require(len(trace) == 3 * len(specs), "Wrong definition trace count")
    if not isprop:
        a.require(trace == source["trace"][:3], "Flat candidate lists changed")
    q = np.asarray(queries[entry["key"]], dtype=np.float64)
    text = entry["text"]
    for i, (dim, k) in enumerate(specs):
        dense, lex, fuse = trace[i * 3 : i * 3 + 3]
        where = (
            {"kind": "definition"}
            if dim is None
            else {"$and": [{"kind": "definition"}, {"dimension": dim}]}
        )
        bucket = "definition" + (":" + dim if dim else "")
        a.require(
            {x: v for x, v in dense.items() if x != "hits"}
            == {"operation": "dense_query", "where": where, "k": 10, "want_vecs": False},
            "Dense request differs",
        )
        eligible = {
            rid
            for rid, x in records.items()
            if x["kind"] == "definition" and (dim is None or x["dimension"] == dim)
        }
        hits = dense["hits"]
        ids = [h["id"] for h in hits]
        a.require(
            len(ids) == len(set(ids)) == min(10, len(eligible)) and set(ids) <= eligible,
            "Dense membership/count differs",
        )
        if dim is not None:
            a.require(set(ids) == eligible, "Proportional bucket not exhaustively returned")
        for j, hit in enumerate(hits):
            v = np.asarray(vectors[hit["id"]], dtype=np.float64)
            expected = float(v @ q / (np.linalg.norm(v) * np.linalg.norm(q)))
            independent.compare_hits(
                [hit],
                [
                    {
                        "id": hit["id"],
                        "kind": "definition",
                        "score": expected,
                        "via": "dense",
                        "ranks": {},
                    }
                ],
                3e-06,
            )
            if j:
                a.require(hits[j - 1]["score"] >= hit["score"], "Dense ordering reversed")
        a.require(
            {x: v for x, v in lex.items() if x != "hits"}
            == {"operation": "bm25_query", "bucket": bucket, "k": 10},
            "Lexical request differs",
        )
        independent.compare_hits(lex["hits"], lexicons[bucket].top(text, 10), 1e-10)
        channels = {"dense": hits, "bm25": lex["hits"]}
        a.require(
            fuse["operation"] == "rrf"
            and fuse["channels"] == channels
            and (fuse["k"] == 60)
            and (fuse["top_n"] == (k if isprop else 2)),
            "RRF source linkage differs",
        )
        independent.compare_hits(
            fuse["hits"], independent.independent_rrf(channels, fuse["top_n"]), 1e-14
        )
        selected = independent.independent_rrf(channels, k)
        out.extend(selected)
        counts["independent_definition_buckets"] += 1
    expected = [
        {
            "id": h["id"],
            "text": records[h["id"]]["text"],
            "meta": {**r.record_meta(records[h["id"]]), "_ranks": h["ranks"]},
            "via": "rrf",
        }
        for h in out
    ]
    a.require(
        a.signature(expected) == a.signature(entry["context"]["definitions"]),
        "Independently selected definitions differ",
    )


def check(mode):
    dest = p.BASE / ("smoke_review_v1" if mode == "smoke" else "preparation_review_v1")
    evidence = ROOT / f"Thesis_Log/astra_result/evidence/sq1_allocation_{mode}_20260924_v1.json"
    a.require(
        not (dest / "complete.json").exists() and (not evidence.exists()),
        "Preserve completed review",
    )
    plan = p.validate()
    first = p.receipt("smoke" if mode == "smoke" else "prepare")
    replay = p.receipt("replay") if mode != "smoke" else None
    a.require(
        first is not None and (mode == "smoke" or replay is not None),
        "Required preparation incomplete",
    )
    allrecs = r.rows(p.CORE / "kb/full/records.jsonl")
    records = {x["id"]: x for x in allrecs}
    defs = [x for x in allrecs if x["kind"] == "definition"]
    matrix = np.load(r.BASE / "full_vectors.npy", mmap_mode="r", allow_pickle=False)
    vectors = {x["id"]: matrix[i] for i, x in enumerate(allrecs) if x["kind"] == "definition"}
    dev = r.rows(p.CORE / "dev_inputs.jsonl")
    query_matrix = np.load(r.BASE / "query_vectors.npy", allow_pickle=False)
    queries = {x["key"]: query_matrix[i] for i, x in enumerate(dev)}
    originals = {
        x["key"]: x for x in r.rows(ROOT / plan["source_prompts"]) if x["strategy"] == "hybrid"
    }
    if mode == "smoke":
        dev = [x for x in dev if x["key"] in plan["smoke_keys"]]
    lexicons = {
        "definition": independent.Lexicon(defs),
        **{
            "definition:" + dim: independent.Lexicon([x for x in defs if x["dimension"] == dim])
            for dim in p.QUOTAS
        },
    }
    entries = r.rows(ROOT / first["prompts"])
    expected = [(x["key"], cond) for x in dev for cond in p.CONDITIONS]
    a.require([identity(e) for e in entries] == expected, "First-pass order/roster differs")
    copies = [independent.check_copy(first, plan, records)]
    bindings = {**plan["source_hashes"], **plan["input_hashes"], **first["hashes"]}
    if replay:
        changes = r.rows(ROOT / replay["attempt"] / "comparisons.jsonl")
        a.require([identity(x) for x in changes] == expected, "Replay order/roster differs")
        actual = [replay_entry(old, new) for old, new in zip(entries, changes)]
        copies.append(independent.check_copy(replay, plan, records))
        bindings.update(replay["hashes"])
        diffs = {
            k: sum((not x["matches"][k] for x in changes))
            for k in ("context", "messages", "retrieved", "definition_trace", "entry")
        }
        a.require(diffs == replay["replay_differences"], "Replay total differs")
    else:
        actual = []
        diffs = {}
    counters = Counter()
    diagnostics = {}
    for number, got in enumerate(entries + actual):
        source = originals[got["key"]]
        for field in ("key", "item_id", "lang", "text", "scopes"):
            a.require(got[field] == source[field], "Input field changed")
        a.require(
            got["plan_sha256"] == a.digest(p.PLAN)
            and got["pool"] == "full"
            and (got["strategy"] == "hybrid")
            and (got["source_entry_sha256"] == a.signature(source))
            and (got["source_messages_sha256"] == source["messages_sha256"]),
            "Entry source binding differs",
        )
        p.validate_context(source, got["context"], got["condition"], records)
        audit_trace(got, source, {x["id"]: x for x in defs}, vectors, queries, lexicons, counters)
        messages = p.previous.prompt.render_messages(
            got["text"], got["lang"], got["context"], "corrected"
        )
        a.require(
            got["messages"] == messages
            and got["messages_sha256"] == a.signature(messages)
            and (got["context_sha256"] == a.signature(got["context"]))
            and (got["retrieved"] == p.previous.prompt.hit_ids(got["context"])),
            "Rendering/signature differs",
        )
        a.require(
            messages[0] == source["messages"][0]
            and messages[1]["text"].split("## Annotation guidelines", 1)[1]
            == source["messages"][1]["text"].split("## Annotation guidelines", 1)[1],
            "Nondefinition prompt content changed",
        )
        if got["condition"] == "flat2":
            a.require(messages == source["messages"], "Flat2 byte equivalence failed")
        counters["checked_entries"] += 1
        if number < len(entries):
            name = got["scopes"][0]["subset"] + "/" + got["condition"]
            d = diagnostics.setdefault(
                name,
                {
                    "inputs": 0,
                    "definition_slots": 0,
                    "dimensions": Counter(),
                    "ids": Counter(),
                    "has_hate_definition": 0,
                    "characters": [],
                },
            )
            d["inputs"] += 1
            d["definition_slots"] += len(got["context"]["definitions"])
            d["dimensions"].update((x["meta"]["dimension"] for x in got["context"]["definitions"]))
            d["ids"].update((x["id"] for x in got["context"]["definitions"]))
            d["has_hate_definition"] += int(
                any((x["id"] == "def-hate" for x in got["context"]["definitions"]))
            )
            d["characters"].append(sum((len(m["text"]) for m in messages)))
        if (number + 1) % 1000 == 0:
            print(
                f"Independently checked {number + 1}/{len(entries) + len(actual)} allocation entries",
                flush=True,
            )
    for d in diagnostics.values():
        values = d.pop("characters")
        d["prompt_characters"] = {
            "min": min(values),
            "mean": sum(values) / len(values),
            "max": max(values),
        }
    p.validate()
    for path in [
        p.PLAN,
        p.BASE / "analysis_plan.json",
        p.BASE / "runs" / first["kind"] / "complete.json",
    ] + ([p.BASE / "runs/replay/complete.json"] if replay else []):
        bindings[r.rel(path)] = a.digest(path)
    report = {
        "created_utc": a.utc(),
        "status": "allocation_" + mode + "_independently_checked",
        "plan_sha256": a.digest(p.PLAN),
        "verified_hashes": bindings,
        "checks": dict(counters),
        "prompt_entries": len(entries),
        "replay_differences": diffs,
        "copied_index_checks": copies,
        "delivery_diagnostics": diagnostics,
        "actual_API_calls": 0,
        "new_gold_scoring": False,
        "preservation": "Original full index, sources, prompts and results unchanged. Old backgrounds reused exactly; every new retrieval mismatch retained.",
        "next": "Separate staged API preparation after accepting first-pass contexts and any replay qualifications.",
    }
    a.save_new(evidence, report)
    p.save(
        dest / "complete.json",
        {
            "status": report["status"],
            "evidence": r.rel(evidence),
            "evidence_sha256": a.digest(evidence),
        },
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "checks": report["checks"],
                "replay_differences": diffs,
                "evidence_sha256": a.digest(evidence),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tests", "smoke", "full"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", p.forbid), patch.object(
        socket, "create_connection", p.forbid
    ), patch.object(p.previous.core, "live_client", p.forbid):
        tests() if args.mode == "tests" else check(args.mode)

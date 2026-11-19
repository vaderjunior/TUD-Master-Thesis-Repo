"""Allocation verify preparation."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import io
import json
import math
from pathlib import Path
import re
import socket
import time
import unittest
from unittest.mock import patch
import numpy as np
from analysis.knowledge.allocation import verify_legacy_preparation as old

p = old.p
a = p.a
r = p.r
ROOT = p.ROOT
BASE = p.BASE / "audit_v2"
TESTS = BASE / "tests_passed.json"
DIAG = BASE / "lexical_disagreement.json"
EVIDENCE = ROOT / "Thesis_Log/astra_result/evidence/sq1_allocation_full_20260924_v2.json"
RANK_SOURCE = ROOT / ".venv/Lib/site-packages/rank_bm25.py"


class OrderedLexicon:

    def __init__(self, records):
        self.records = records
        posting = defaultdict(list)
        lengths = []
        for i, record in enumerate(records):
            words = re.findall("\\w+", record["text"].lower())
            lengths.append(len(words))
            for token, count in Counter(words).items():
                posting[token].append((i, count))
        self.lengths = np.array(lengths)
        self.average_length = sum(lengths) / len(records)
        idfs = {}
        total = 0.0
        for token, values in posting.items():
            value = math.log(len(records) - len(values) + 0.5) - math.log(len(values) + 0.5)
            idfs[token] = value
            total += value
        floor = 0.25 * (total / len(idfs))
        self.idf = {token: floor if value < 0 else value for token, value in idfs.items()}
        self.posting = {
            token: (np.array([i for i, _ in values]), np.array([n for _, n in values]))
            for token, values in posting.items()
        }

    def top(self, text, k):
        scores = np.zeros(len(self.records))
        for token in re.findall("\\w+", text.lower()):
            if token not in self.posting:
                continue
            indices, frequencies = self.posting[token]
            denominator = frequencies + 1.5 * (
                0.25 + 0.75 * self.lengths[indices] / self.average_length
            )
            scores[indices] += self.idf[token] * (frequencies * 2.5 / denominator)
        order = np.argsort(-scores, kind="stable")[:k]
        return [
            {
                "id": self.records[i]["id"],
                "kind": self.records[i]["kind"],
                "score": float(scores[i]),
                "via": "bm25",
                "ranks": {},
            }
            for i in order
            if scores[i] > 0
        ]


def source_hashes():
    return {r.rel(path): a.digest(path) for path in (Path(__file__), RANK_SOURCE)}


def lexicons(defs, klass):
    return {
        "definition": klass(defs),
        **{
            "definition:" + dim: klass([x for x in defs if x["dimension"] == dim])
            for dim in p.QUOTAS
        },
    }


def diagnose():
    p.validate()
    defs = [x for x in r.rows(p.CORE / "kb/full/records.jsonl") if x["kind"] == "definition"]
    before = lexicons(defs, old.independent.Lexicon)
    after = lexicons(defs, OrderedLexicon)
    entries = r.rows(ROOT / p.receipt("prepare")["prompts"])
    differences = []
    n = 0
    max_error = 0.0
    for entry in entries:
        for event in entry["definition_trace"]:
            if event["operation"] != "bm25_query":
                continue
            actual = event["hits"]
            first = before[event["bucket"]].top(entry["text"], 10)
            second = after[event["bucket"]].top(entry["text"], 10)
            old.independent.compare_hits(actual, second, 1e-14)
            n += 1
            max_error = max(
                max_error,
                max((abs(x["score"] - y["score"]) for x, y in zip(actual, second)), default=0.0),
            )
            if [x["id"] for x in actual] != [x["id"] for x in first]:
                a.require(
                    {x["id"] for x in actual} == {x["id"] for x in first},
                    "Non-rounding candidate membership difference",
                )
                old_by_id = {x["id"]: x for x in first}
                error = max((abs(x["score"] - old_by_id[x["id"]]["score"]) for x in actual))
                a.require(error < 1e-12, "Disagreement larger than rounding")
                differences.append(
                    {
                        "key": entry["key"],
                        "condition": entry["condition"],
                        "bucket": event["bucket"],
                        "saved": actual,
                        "original_checker": first,
                        "amended_checker": second,
                        "max_original_score_difference": error,
                    }
                )
    a.require(differences, "Expected first failed audit disagreement missing")
    report = {
        "created_utc": a.utc(),
        "status": "lexical_disagreement_explained_without_changing_prompts",
        "first_full_audit_failure": {
            "module": "analysis.knowledge.allocation.verify_legacy_preparation",
            "exception": "RuntimeError: Ranked hit identity/provenance",
            "location": "audit_trace / BM25 independent.compare_hits",
            "first_key": differences[0]["key"],
            "first_bucket": differences[0]["bucket"],
            "source_sha256": a.digest(
                ROOT / "analysis/knowledge/allocation/verify_legacy_preparation.py"
            ),
        },
        "checked_lexical_events": n,
        "rank_disagreements": len(differences),
        "details": differences,
        "amended_max_abs_score_difference": max_error,
        "source_hashes": source_hashes(),
        "prompts_sha256": a.digest(ROOT / p.receipt("prepare")["prompts"]),
        "cause": "Original independent checker reassociated idf*frequency*2.5/denominator and used compensated sum for average IDF; original retrieval uses idf*(frequency*2.5/denominator) and sequential IDF accumulation. Preserve exact rank order rather than treating near ties as arbitrary.",
        "experiment_mutations": 0,
        "actual_API_calls": 0,
    }
    p.save(DIAG, report)
    return report


class NumericalAuditTests(unittest.TestCase):

    def fixture(self):
        return [
            {"id": rid, "kind": "definition", "text": text}
            for rid, text in [
                ("b", "rare same"),
                ("a", "rare same"),
                ("c", "third unique"),
                ("d", "fourth token"),
                ("e", "fifth extra"),
            ]
        ]

    def test_hand_computed_rare_token_and_stable_tie(self):
        hits = OrderedLexicon(self.fixture()).top("rare", 10)
        self.assertEqual([x["id"] for x in hits], ["b", "a"])
        for hit in hits:
            self.assertAlmostEqual(hit["score"], math.log(3.5) - math.log(2.5), places=14)

    def test_no_overlap_and_single_document_no_padding(self):
        self.assertEqual(OrderedLexicon(self.fixture()).top("absent", 10), [])
        self.assertEqual(
            OrderedLexicon([{"id": "x", "kind": "definition", "text": "word"}]).top("word", 10), []
        )

    def test_audit_does_not_permit_changed_rank_or_score(self):
        hits = OrderedLexicon(self.fixture()).top("rare", 10)
        with self.assertRaises(RuntimeError):
            old.independent.compare_hits(list(reversed(hits)), hits, 1e-10)
        wrong = deepcopy(hits)
        wrong[0]["score"] += 0.001
        with self.assertRaises(RuntimeError):
            old.independent.compare_hits(wrong, hits, 1e-10)

    def test_all_saved_lexical_traces_and_preserved_failure(self):
        report = diagnose()
        self.assertEqual(report["checked_lexical_events"], 9065)
        self.assertEqual(
            report["first_full_audit_failure"]["first_key"], "en/hatexplain-19930942_gab"
        )
        self.assertLessEqual(report["amended_max_abs_score_difference"], 1e-14)


def tests():
    a.require(not TESTS.exists() and (not DIAG.exists()), "Preserve numerical audit tests")
    attempt = BASE / f"test_attempt_{time.time_ns()}"
    hashes = source_hashes()
    p.save(
        attempt / "sources.json",
        {name: (ROOT / name).read_text(encoding="utf8") for name in hashes},
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(NumericalAuditTests)
    )
    report = {
        "created_utc": a.utc(),
        "passed": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "source_hashes": hashes,
        "output": stream.getvalue(),
        "actual_API_calls": 0,
    }
    p.save(attempt / "tests.json", report)
    print(stream.getvalue(), flush=True)
    a.require(result.wasSuccessful(), "Numerical audit tests failed; preserve snapshot")
    r.verify_hashes(hashes)
    p.save(TESTS, report)


def check():
    dest = p.BASE / "preparation_review_v2"
    a.require(
        not EVIDENCE.exists() and (not (dest / "complete.json").exists()),
        "Preserve completed audit",
    )
    tests = a.read(TESTS)
    a.require(tests["passed"] and tests["tests_run"] == 4, "Numerical audit tests required")
    r.verify_hashes(tests["source_hashes"])
    diagnosis = a.read(DIAG)
    r.verify_hashes(diagnosis["source_hashes"])
    plan = p.validate()
    first = p.receipt("prepare")
    replay = p.receipt("replay")
    a.require(first is not None and replay is not None, "Complete preparation/replay required")
    allrecs = r.rows(p.CORE / "kb/full/records.jsonl")
    records = {x["id"]: x for x in allrecs}
    defs = [x for x in allrecs if x["kind"] == "definition"]
    def_records = {x["id"]: x for x in defs}
    matrix = np.load(r.BASE / "full_vectors.npy", mmap_mode="r", allow_pickle=False)
    vectors = {x["id"]: matrix[i] for i, x in enumerate(allrecs) if x["kind"] == "definition"}
    dev = r.rows(p.CORE / "dev_inputs.jsonl")
    query_matrix = np.load(r.BASE / "query_vectors.npy", allow_pickle=False)
    queries = {x["key"]: query_matrix[i] for i, x in enumerate(dev)}
    originals = {
        x["key"]: x for x in r.rows(ROOT / plan["source_prompts"]) if x["strategy"] == "hybrid"
    }
    lex = lexicons(defs, OrderedLexicon)
    entries = r.rows(ROOT / first["prompts"])
    expected = [(x["key"], cond) for x in dev for cond in p.CONDITIONS]
    a.require([old.identity(e) for e in entries] == expected, "First-pass order/roster differs")
    changes = r.rows(ROOT / replay["attempt"] / "comparisons.jsonl")
    a.require([old.identity(x) for x in changes] == expected, "Replay order/roster differs")
    actual = [old.replay_entry(before, after) for before, after in zip(entries, changes)]
    diffs = {
        k: sum((not x["matches"][k] for x in changes))
        for k in ("context", "messages", "retrieved", "definition_trace", "entry")
    }
    a.require(diffs == replay["replay_differences"], "Replay summary differs")
    copies = [old.independent.check_copy(receipt, plan, records) for receipt in (first, replay)]
    counters = Counter()
    diagnostics = {}
    plan_sha = a.digest(p.PLAN)
    for number, got in enumerate(entries + actual):
        source = originals[got["key"]]
        for field in ("key", "item_id", "lang", "text", "scopes"):
            a.require(got[field] == source[field], "Input field changed")
        a.require(
            got["plan_sha256"] == plan_sha
            and got["pool"] == "full"
            and (got["strategy"] == "hybrid")
            and (got["source_entry_sha256"] == a.signature(source))
            and (got["source_messages_sha256"] == source["messages_sha256"]),
            "Entry source binding differs",
        )
        p.validate_context(source, got["context"], got["condition"], records)
        old.audit_trace(got, source, def_records, vectors, queries, lex, counters)
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
                f"Independently checked {number + 1}/{len(entries) + len(actual)} allocation entries (v2)",
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
    r.verify_hashes(tests["source_hashes"])
    bindings = {
        **plan["source_hashes"],
        **plan["input_hashes"],
        **first["hashes"],
        **replay["hashes"],
        **source_hashes(),
    }
    for path in [
        p.PLAN,
        p.BASE / "analysis_plan.json",
        p.BASE / "runs/prepare/complete.json",
        p.BASE / "runs/replay/complete.json",
        TESTS,
        DIAG,
    ]:
        bindings[r.rel(path)] = a.digest(path)
    report = {
        "created_utc": a.utc(),
        "status": "allocation_full_independently_checked",
        "audit_version": 2,
        "plan_sha256": plan_sha,
        "verified_hashes": bindings,
        "checks": dict(counters),
        "prompt_entries": len(entries),
        "replay_differences": diffs,
        "copied_index_checks": copies,
        "delivery_diagnostics": diagnostics,
        "actual_API_calls": 0,
        "new_gold_scoring": False,
        "auditor_amendment": {
            "diagnosis": r.rel(DIAG),
            "diagnosis_sha256": a.digest(DIAG),
            "tests": r.rel(TESTS),
            "tests_sha256": a.digest(TESTS),
            "old_rank_disagreements": diagnosis["rank_disagreements"],
            "amended_max_abs_score_difference": diagnosis["amended_max_abs_score_difference"],
            "change": "Independent lexical arithmetic/order only. No production ranker import, score-tolerance relaxation, prompt changes, selection changes, new retrieval or experiment rerun.",
        },
        "preservation": "Original source/index, frozen preparation, original v1 audit, all first-pass/replay outputs and initial failure preserved.",
        "next": "Separate staged API preparation after accepting first-pass contexts.",
    }
    a.save_new(EVIDENCE, report)
    p.save(
        dest / "complete.json",
        {
            "status": report["status"],
            "evidence": r.rel(EVIDENCE),
            "evidence_sha256": a.digest(EVIDENCE),
        },
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "audit_version": 2,
                "checks": report["checks"],
                "replay_differences": diffs,
                "old_rank_disagreements": diagnosis["rank_disagreements"],
                "amended_max_abs_score_difference": diagnosis["amended_max_abs_score_difference"],
                "evidence_sha256": a.digest(EVIDENCE),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tests", "full"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", p.forbid), patch.object(
        socket, "create_connection", p.forbid
    ), patch.object(p.previous.core, "live_client", p.forbid):
        tests() if args.mode == "tests" else check()

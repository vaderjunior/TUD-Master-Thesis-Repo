"""Sampled banks verify retrieval."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import socket
import sqlite3
from unittest.mock import patch
import numpy as np
from analysis.knowledge.sampled_banks import retrieval as p
from analysis.knowledge.strategies import verify as independent

r = p.r


class OrderedLexicon:

    def __init__(self, records):
        self.records = records
        posting = defaultdict(list)
        lengths = []
        for i, rec in enumerate(records):
            words = re.findall("\\w+", rec["text"].lower())
            lengths.append(len(words))
            for word, count in Counter(words).items():
                posting[word].append((i, count))
        self.lengths = np.asarray(lengths)
        self.average_length = sum(lengths) / len(records)
        total = 0.0
        idfs = {}
        for token, values in posting.items():
            value = math.log(len(records) - len(values) + 0.5) - math.log(len(values) + 0.5)
            idfs[token] = value
            total += value
        floor = 0.25 * (total / len(idfs))
        self.idf = {token: floor if value < 0 else value for token, value in idfs.items()}
        self.posting = {
            token: (np.asarray([i for i, _ in values]), np.asarray([n for _, n in values]))
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


class TraceAudit(independent.TraceAudit):

    def __init__(self, records, vectors, queries):
        self.records = {rec["id"]: rec for rec in records}
        self.ix = {rec["id"]: i for i, rec in enumerate(records)}
        self.vectors = vectors
        self.queries = queries
        buckets = defaultdict(list)
        for rec in records:
            bucket = rec["kind"] + (":" + rec["lang"] if rec["kind"] == "example" else "")
            buckets[bucket].append(rec)
        self.lexicons = {key: OrderedLexicon(rr) for key, rr in buckets.items()}
        self.cache = {}
        self.counts = Counter()
        self.max_dense_score_error = 0.0
        self.min_mmr_margin = 1.0


def database_policy(integrity, foreign, orphans):
    r.require(
        integrity == [("ok",)] and orphans == 0, "Corrupt index or genuine collection orphans"
    )
    known = [("segments", 1, "collection", 0), ("segments", 2, "collection", 0)]
    r.require(foreign in ([], known), "Unexpected SQLite foreign-key finding")


def database_check(done, records):
    directory = p.ROOT / done["attempt"] / "chroma"
    before = p.tree_hashes(directory)
    r.require(before == done["index_hashes"], "Post-exit index hash changed")
    known = {rec["id"]: rec for rec in records}
    with sqlite3.connect((directory / "chroma.sqlite3").as_uri() + "?mode=ro", uri=True) as conn:
        integrity = conn.execute("PRAGMA integrity_check").fetchall()
        foreign = conn.execute("PRAGMA foreign_key_check").fetchall()
        orphans = conn.execute(
            "SELECT COUNT(*) FROM segments s LEFT JOIN collections c ON s.collection=c.id WHERE c.id IS NULL"
        ).fetchone()[0]
        database_policy(integrity, foreign, orphans)
        ids = conn.execute("SELECT id, embedding_id FROM embeddings").fetchall()
        r.require(
            len(ids) == len(known) and {rid for _, rid in ids} == set(known), "SQLite record roster"
        )
        metadata = {i: {} for i, _ in ids}
        for i, key, string, integer, floating, boolean in conn.execute(
            "SELECT * FROM embedding_metadata"
        ):
            values = [v for v in (string, integer, floating, boolean) if v is not None]
            r.require(
                i in metadata and len(values) == 1 and (key not in metadata[i]),
                "SQLite field shape",
            )
            value = bool(boolean) if boolean is not None else values[0]
            if key != "chroma:document":
                if value in ("", "__none__"):
                    value = None
                elif isinstance(value, str) and value.startswith("["):
                    value = json.loads(value)
            metadata[i][key] = value
        for i, rid in ids:
            rec = known[rid]
            expected = {
                **{k: rec[k] for k in ("kind", "lang", "source", "dimension", "label")},
                **rec["meta"],
                "chroma:document": rec["text"],
            }
            r.require(r.js(metadata[i]) == r.js(expected), "SQLite text/annotations differ")
    r.require(p.tree_hashes(directory) == before, "Read-only SQLite inspection changed index")
    parent_diff = []
    if done["source_receipt"]:
        parent = r.read(p.ROOT / done["source_receipt"])
        r.require(
            r.sha(p.ROOT / done["source_receipt"]) == done["source_receipt_sha256"],
            "Copied-index receipt changed",
        )
        source = p.ROOT / parent["attempt"] / "chroma"
        names = {
            str((p.ROOT / path).relative_to(source)): sha
            for path, sha in parent["index_hashes"].items()
        }
        child = {str((p.ROOT / path).relative_to(directory)): sha for path, sha in before.items()}
        r.require(names.keys() == child.keys(), "Copied index file roster changed")
        parent_diff = [name for name in names if child[name] != names[name]]
        r.require(
            set(parent_diff) <= {"chroma.sqlite3"}, "Copied vector graph changed after queries"
        )
    return {
        "records": len(ids),
        "integrity": "ok",
        "foreign_key_flags": foreign,
        "actual_orphans": orphans,
        "changed_from_parent_index_files": parent_diff,
    }


def gold_line(meta):
    pieces = []
    if meta.get("gate") is not None:
        r.require(type(meta["gate"]) is bool, "Gate must remain boolean")
        pieces.append("hate=" + ("true" if meta["gate"] else "false"))
    for field, output in (("target_groups", "target_group"), ("hate_types", "hate_type")):
        if meta.get(field) is not None:
            pieces.append(output + "=[" + ", ".join(meta[field]) + "]")
    if meta.get("severity") is not None:
        pieces.append("severity=" + meta["severity"])
    if meta.get("legal") is not None:
        pieces.append("legal=[" + ", ".join(meta["legal"]) + "]")
    value = ", ".join(pieces)
    if meta.get("astra_aux_stgb"):
        value += (
            "\n  Auxiliary DeTox paragraph flags (illustrative, not legal gold): §"
            + meta["astra_aux_stgb"]
        )
    return value


def audit_entry(entry, row, job, known, reference, checker):
    r.require(
        all((entry[k] == value for k, value in row.items())), "Input text/identity/scope changed"
    )
    r.require(
        entry["condition"] == job["condition"]
        and entry["pool_id"] == job["pool_id"]
        and (entry["plan_sha256"] == r.sha(p.PLAN))
        and (entry["strategy"] == "hybrid"),
        "Wrong condition/plan",
    )
    checker.entry(entry)
    raw = independent.source_context(entry["final_hits"], known)
    r.require(r.js(raw) == r.js(entry["raw_context"]), "Raw retrieved source/annotation mismatch")
    context = {
        "definitions": reference["context"]["definitions"],
        "guidelines": reference["context"]["guidelines"],
        "example_groups": raw["example_groups"],
    }
    r.require(
        r.js(context) == r.js(entry["context"]), "Fixed abstract/selected example context differs"
    )
    p.trace_tools.validate_context(context, known, row["lang"], "hybrid")
    r.require(
        p.prompt.hit_ids(context) == entry["retrieved"]
        and r.signature(context) == entry["context_sha256"],
        "Context IDs/signature",
    )
    messages = p.prompt.render_messages(row["text"], row["lang"], context, "corrected")
    r.require(
        messages == entry["messages"] and r.signature(messages) == entry["messages_sha256"],
        "Prompt renderer/signature",
    )
    user = messages[-1]["text"]
    suffix = "## Text to classify (language: " + row["lang"] + ")\n" + row["text"]
    r.require(user.endswith(suffix), "Query text changed")
    example_section = user[: -len(suffix)].split("## Labelled examples\n", 1)[1]
    hits = context["example_groups"][0][1]
    lines = re.findall("^  labels: (.*)$", example_section, flags=re.MULTILINE)
    r.require(
        lines == [gold_line(known[hit["id"]]["meta"]).split("\n", 1)[0] for hit in hits],
        "Rendered annotation lines differ",
    )
    for hit in hits:
        rec = known[hit["id"]]
        fragment = '- "' + " ".join(rec["text"].split()) + '"\n  labels: ' + gold_line(rec["meta"])
        r.require(fragment in example_section, "Example text/label fragment differs")


def replay_entry(before, saved):
    flags = saved["matches"]
    r.require(
        set(flags) == set(p.COMPARE_FIELDS) | {"entry"}
        and all((type(v) is bool for v in flags.values())),
        "Replay schema",
    )
    if flags["entry"]:
        r.require(
            all(flags.values()) and "actual_entry" not in saved, "Invalid equal-replay record"
        )
        return before
    r.require("actual_entry" in saved, "Changed replay has no saved alternative")
    actual = saved["actual_entry"]
    r.require(
        actual["key"] == before["key"] == saved["key"]
        and actual["condition"] == before["condition"] == saved["condition"],
        "Replay identity mismatch",
    )
    expected = {field: r.js(before[field]) == r.js(actual[field]) for field in p.COMPARE_FIELDS}
    expected["entry"] = r.js(before) == r.js(actual)
    r.require(flags == expected, "Replay differences incorrectly reported")
    return actual


def main(mode):
    destination = p.BASE / ("preflight.json" if mode == "smoke" else "full_audit.json")
    if destination.exists():
        report = r.read(destination)
        r.require(report["plan_sha256"] == r.sha(p.PLAN), "Old audit plan changed")
        r.verify_hashes(report["hashes"])
        print("Completed independent audit verified and skipped.")
        return
    plan = p.validate()
    dev = r.rows(p.CORE / "dev_inputs.jsonl")
    dev_map = {row["key"]: row for row in dev}
    references = {
        cond: {row["key"]: row for row in r.rows(p.ROOT / path)}
        for cond, path in plan["reference_prompts"].items()
    }
    first, second = ("smoke", "smoke_replay") if mode == "smoke" else ("prepare", "replay")
    expected_keys = plan["smoke_keys"] if mode == "smoke" else [row["key"] for row in dev]
    hashes = {**plan["input_hashes"], **plan["original_index_hashes"], r.rel(p.PLAN): r.sha(p.PLAN)}
    checks = Counter()
    differences = Counter()
    by_pool = []
    annotation = {}
    first_entries = []
    projected_retrieval = 0.0
    overhead = 0.0
    for job in plan["jobs"]:
        a, b = (p.receipt(job["pool_id"], first), p.receipt(job["pool_id"], second))
        r.require(a is not None and b is not None, "Missing complete first/replay pair")
        records, source_vectors, _, queries = p.data(job)
        known = {rec["id"]: rec for rec in records}
        qmap = {row["key"]: queries[i] for i, row in enumerate(dev)}
        entries = r.rows(p.ROOT / a["prompts"])
        replay = r.rows(p.ROOT / b["output"])
        r.require(
            [e["key"] for e in entries] == [e["key"] for e in replay] == expected_keys,
            "Pair roster/order incomplete",
        )
        per_pool = Counter()
        pair_checks = []
        for done, label in ((a, "first"), (b, "replay")):
            hashes[r.rel(p.receipt_path(job["pool_id"], done["kind"]))] = r.sha(
                p.receipt_path(job["pool_id"], done["kind"])
            )
            hashes.update(done["hashes"])
            hashes.update(done["index_hashes"])
            db = database_check(done, records)
            vec = np.load(p.ROOT / done["indexed_vectors"], allow_pickle=False)
            r.validate_vectors(vec, len(records))
            error = float(np.max(np.abs(vec - source_vectors)))
            r.require(
                error < 2e-07 and error <= done["index_vector_max_abs_difference"] + 1e-12,
                "Stored vector export mismatch",
            )
            checker = TraceAudit(records, vec, qmap)
            for i, entry in enumerate(entries):
                actual = entry if label == "first" else replay_entry(entry, replay[i])
                audit_entry(
                    actual,
                    dev_map[actual["key"]],
                    job,
                    known,
                    references["full_reference"][actual["key"]],
                    checker,
                )
                checks["rendered_prompts"] += 1
                checks["rendered_example_label_blocks"] += 5
                if label == "first":
                    name = actual["scopes"][0]["subset"] + "/" + job["condition"]
                    bucket = annotation.setdefault(name, Counter())
                    bucket["inputs"] += 1
                    for hit in actual["context"]["example_groups"][0][1]:
                        bucket["example_slots"] += 1
                        for field in ("gate", "target_groups", "hate_types", "severity", "legal"):
                            value = hit["meta"].get(field)
                            bucket[field + "_annotated"] += value is not None
                            bucket[field + "_empty_lists"] += type(value) is list and (not value)
                else:
                    for key, equal in replay[i]["matches"].items():
                        per_pool[key] += not equal
            checks.update(checker.counts)
            checks["indexes"] += 1
            checks["index_records"] += len(records)
            pair_checks.append(
                {
                    "pass": label,
                    "database": db,
                    "index_vector_max_error": error,
                    "max_dense_score_error": checker.max_dense_score_error,
                    "minimum_MMR_margin": checker.min_mmr_margin,
                    "trace_checks": dict(checker.counts),
                }
            )
            projected_retrieval += done["retrieval_seconds"] * 1295 / done["entries"]
            overhead += max(0.0, done["child_elapsed_seconds"] - done["retrieval_seconds"])
        r.require(dict(per_pool) == b["replay_differences"], "Replay counter mismatch")
        differences.update(per_pool)
        by_pool.append(
            {
                "condition": job["condition"],
                "records": len(records),
                "replay_differences": dict(per_pool),
                "checks": pair_checks,
                "first_retrieval_seconds": a["retrieval_seconds"],
                "replay_retrieval_seconds": b["retrieval_seconds"],
            }
        )
        first_entries += entries
    if mode == "full":
        assembly = r.read(p.BASE / "assembly.json")
        r.verify_hashes(assembly["hashes"])
        assembled = r.rows(p.BASE / "prompts.jsonl")
        expected = []
        for condition in plan["reference_prompts"]:
            for row in dev:
                old = references[condition][row["key"]]
                current = assembled[len(expected)]
                r.require(
                    current["messages"] == old["messages"]
                    and current["context"] == old["context"]
                    and (current["retrieved"] == old["retrieved"])
                    and (current["messages_sha256"] == old["messages_sha256"]),
                    "Reference payload changed",
                )
                expected.append((row["key"], condition))
                checks["inherited_reference_payloads"] += 1
        expected += [(entry["key"], entry["condition"]) for entry in first_entries]
        r.require(
            len(assembled) == len(expected) == 14245
            and [(row["key"], row["condition"]) for row in assembled] == expected,
            "Final assembled roster differs",
        )
        r.require(
            r.js(assembled[2590:]) == r.js(first_entries),
            "Final matrix does not retain original first pass",
        )
        hashes.update(assembly["hashes"])
        hashes[r.rel(p.BASE / "assembly.json")] = r.sha(p.BASE / "assembly.json")
    else:
        r.require(
            not any(
                (p.receipt_path(job["pool_id"], "prepare").parent.exists() for job in plan["jobs"])
            ),
            "Full preparation already started during preflight",
        )
    p.validate()
    report = {
        "status": (
            "ready_for_user_full_retrieval_not_API"
            if mode == "smoke"
            else "full_pool_retrieval_checked_Astra_review_pending_not_API"
        ),
        "created_utc": p.utc(),
        "plan_sha256": r.sha(p.PLAN),
        "mode": mode,
        "checks": dict(checks),
        "replay_differences": dict(differences),
        "by_pool": by_pool,
        "delivered_annotation_counts": {k: dict(v) for k, v in annotation.items()},
        "runtime_projection_seconds": {
            "sampled_retrieval_extrapolation": projected_retrieval,
            "observed_worker_overhead": overhead,
            "rough_central": projected_retrieval + overhead,
            "qualification": "Only three smoke texts per pool, not a runtime guarantee. Audit and input-dependent costs can differ; allow considerably longer.",
        },
        "interpretation": "No classifier outputs or quality scores. ANN candidates are observed, not certified exact nearest neighbours. All saved first-pass prompts remain authoritative; any replay differences require Astra review, never substitution or retry-to-match.",
        "hashes": hashes,
        "actual_API_calls": 0,
        "CUDA_used": False,
        "gold_scoring": False,
    }
    p.save(destination, report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in ("status", "checks", "replay_differences", "runtime_projection_seconds")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["smoke", "full"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", p.no_network), patch.object(
        socket, "create_connection", p.no_network
    ):
        main(args.mode)

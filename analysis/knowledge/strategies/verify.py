"""Strategies verify."""

from collections import Counter, defaultdict
from datetime import datetime
from itertools import combinations
import json
import math
from pathlib import Path
import re
import sqlite3
import numpy as np
from analysis.knowledge.strategies import prepare as s

r = s.r
PLAN_SHA = "d22c3018572515d19943c598422ee91a8c1fe42af631d85efa17dcf7eacb6ad1"
EVIDENCE = s.ROOT / "Thesis_Log/astra_result/evidence/sq1_strategy_full_audit_20260920_v1.json"
FIELDS = ("context", "messages", "retrieved", "trace", "final_hits", "entry")
ANNOTATIONS = ("gate", "target_groups", "hate_types", "severity", "legal")


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def ident(entry):
    return (entry["key"], entry["strategy"])


def matrix_check(entries, expected):
    need(
        [ident(x) for x in entries] == expected, "Missing, duplicate or reordered strategy entries"
    )


def compare_replay(old, saved):
    flags = saved["matches"]
    need(
        set(flags) == set(FIELDS) and all((type(x) is bool for x in flags.values())),
        "Replay flag schema",
    )
    if flags["entry"]:
        need(all(flags.values()) and "actual_entry" not in saved, "Inconsistent unchanged replay")
        return old
    need("actual_entry" in saved, "Changed replay lacks actual entry")
    actual = saved["actual_entry"]
    need(ident(actual) == ident(saved) == ident(old), "Changed replay identity")
    expected = {k: old[k] == actual[k] for k in FIELDS if k != "entry"}
    expected["entry"] = old == actual
    need(flags == expected, "Replay flags do not match saved alternative")
    return actual


def plain_hits(hits):
    return [h["id"] for h in hits]


def compare_hits(actual, expected, tolerance=0.0):
    need(len(actual) == len(expected), "Ranked hit count")
    for a, b in zip(actual, expected):
        need(set(a) == set(b) == {"id", "kind", "score", "via", "ranks"}, "Trace hit schema")
        need(
            all((a[k] == b[k] for k in ("id", "kind", "via", "ranks"))),
            "Ranked hit identity/provenance",
        )
        need(
            math.isfinite(a["score"]) and abs(a["score"] - b["score"]) <= tolerance,
            "Ranked hit score",
        )


def source_context(final_hits, records):
    groups = {"definition": [], "guideline": [], "example": []}
    order = []
    for hit in final_hits:
        record = records[hit["id"]]
        need(hit["kind"] == record["kind"] in groups, "Final hit kind")
        meta = r.record_meta(record)
        if hit["via"] == "rrf":
            meta = {**meta, "_ranks": hit["ranks"]}
        else:
            need(not hit["ranks"], "Non-fused rank metadata")
        groups[record["kind"]].append(
            {"id": record["id"], "text": record["text"], "meta": meta, "via": hit["via"]}
        )
        order.append(record["kind"])
    need(
        order == sorted(order, key=("definition", "guideline", "example").index),
        "Final bucket order",
    )
    examples = groups["example"]
    regular = [h for h in examples if not h["meta"].get("illustrative_only")]
    illustrations = [h for h in examples if h["meta"].get("illustrative_only")]
    return {
        "definitions": groups["definition"],
        "guidelines": groups["guideline"],
        "example_groups": ([["Labelled examples", regular]] if regular else [])
        + ([["Legal context illustrations", illustrations]] if illustrations else []),
    }


def audit_context(entry, row, records):
    need(
        set(entry)
        == set(row)
        | {
            "condition",
            "context",
            "context_sha256",
            "final_hits",
            "messages",
            "messages_sha256",
            "plan_sha256",
            "pool",
            "retrieved",
            "strategy",
            "trace",
        },
        "Entry schema",
    )
    need(all((entry[k] == v for k, v in row.items())), "Input text/language/identity/scope differs")
    need(
        entry["plan_sha256"] == PLAN_SHA
        and entry["pool"] == "full"
        and (entry["condition"] == "rag_" + entry["strategy"]),
        "Entry contract",
    )
    need(entry["strategy"] in s.STRATEGIES, "Unknown strategy")
    context = source_context(entry["final_hits"], records)
    need(r.js(context) == r.js(entry["context"]), "Context source annotation/group/order mismatch")
    s.validate_context(context, records, row["lang"], entry["strategy"])
    need(r.signature(context) == entry["context_sha256"], "Context signature")
    need(s.prompt.hit_ids(context) == entry["retrieved"], "Ordered context IDs")
    messages = s.prompt.render_messages(row["text"], row["lang"], context, "corrected")
    need(
        messages == entry["messages"] and r.signature(messages) == entry["messages_sha256"],
        "Prompt rendering/signature",
    )


class Lexicon:
    """Sparse independent BM25Okapi arithmetic; no production ranker import."""

    def __init__(self, records):
        self.records = records
        posting = defaultdict(list)
        lengths = []
        for i, record in enumerate(records):
            tokens = re.findall("\\w+", record["text"].lower())
            lengths.append(len(tokens))
            for term, count in Counter(tokens).items():
                posting[term].append((i, count))
        self.lengths = np.array(lengths)
        self.average_length = sum(lengths) / len(records)
        idf = {
            term: math.log(len(records) - len(values) + 0.5) - math.log(len(values) + 0.5)
            for term, values in posting.items()
        }
        floor = 0.25 * sum(idf.values()) / len(idf)
        self.idf = {term: floor if value < 0 else value for term, value in idf.items()}
        self.posting = {
            term: (np.array([i for i, _ in values]), np.array([n for _, n in values]))
            for term, values in posting.items()
        }

    def top(self, text, k):
        scores = np.zeros(len(self.records))
        for term in re.findall("\\w+", text.lower()):
            if term not in self.posting:
                continue
            ix, freq = self.posting[term]
            scores[ix] += (
                self.idf[term]
                * freq
                * 2.5
                / (freq + 1.5 * (0.25 + 0.75 * self.lengths[ix] / self.average_length))
            )
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


def independent_mmr(query, vectors, top_n, lam=0.7):
    q = np.asarray(query, dtype=np.float64).ravel()
    q /= np.linalg.norm(q) or 1.0
    v = np.asarray(vectors, dtype=np.float64)
    v = v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-12)
    relevance, similarity = (v @ q, v @ v.T)
    chosen = [int(np.argmax(relevance))]
    margins = []
    while len(chosen) < top_n:
        value = lam * relevance - (1 - lam) * similarity[:, chosen].max(axis=1)
        value[chosen] = -np.inf
        best = int(np.argmax(value))
        ordered = np.sort(value[np.isfinite(value)])
        if len(ordered) > 1:
            margins.append(float(ordered[-1] - ordered[-2]))
        chosen.append(best)
    return (chosen, min(margins, default=None))


def independent_rrf(channels, top_n, k=60):
    scores, ranks, kinds = ({}, {}, {})
    for name, hits in channels.items():
        need(len(set(plain_hits(hits))) == len(hits), "Duplicate RRF channel candidate")
        for rank, hit in enumerate(hits, 1):
            rid = hit["id"]
            scores[rid] = scores.get(rid, 0.0) + 1 / (k + rank)
            ranks.setdefault(rid, {})[name] = rank
            kinds[rid] = hit["kind"]
    order = sorted(scores, key=lambda rid: (-scores[rid], rid))[:top_n]
    return [
        {"id": rid, "kind": kinds[rid], "score": scores[rid], "via": "rrf", "ranks": ranks[rid]}
        for rid in order
    ]


class TraceAudit:

    def __init__(self, records, vectors, queries):
        self.records = {x["id"]: x for x in records}
        self.ix = {x["id"]: i for i, x in enumerate(records)}
        self.vectors, self.queries = (vectors, queries)
        buckets = defaultdict(list)
        for record in records:
            key = record["kind"] + (":" + record["lang"] if record["kind"] == "example" else "")
            buckets[key].append(record)
        self.lexicons = {key: Lexicon(values) for key, values in buckets.items()}
        self.cache = {}
        self.counts = Counter()
        self.max_dense_score_error = 0.0
        self.min_mmr_margin = 1.0

    def lexical(self, event, bucket, k, text):
        need(
            {x: event[x] for x in event if x != "hits"}
            == {"operation": "bm25_query", "bucket": bucket, "k": k},
            "BM25 event contract",
        )
        identity = (bucket, text)
        if identity not in self.cache:
            self.cache[identity] = self.lexicons[bucket].top(text, 10) if k else []
        expected = self.cache[identity][:k]
        compare_hits(event["hits"], expected, 1e-10)
        self.counts["bm25_query"] += 1
        return event["hits"]

    def dense(self, event, kind, k, lang, query, want=False):
        where = (
            {"$and": [{"kind": kind}, {"lang": lang}]}
            if kind in ("example", "feedback")
            else {"kind": kind}
        )
        need(
            {x: event[x] for x in event if x != "hits"}
            == {"operation": "dense_query", "where": where, "k": k, "want_vecs": want},
            "Dense event contract",
        )
        hits = event["hits"]
        need(len(hits) == k and len(set(plain_hits(hits))) == k, "Dense count or duplicate")
        need(
            all((h["score"] >= n["score"] for h, n in zip(hits, hits[1:]))), "Dense candidate order"
        )
        for h in hits:
            record = self.records[h["id"]]
            need(
                record["kind"] == kind and (kind != "example" or record["lang"] == lang),
                "Dense candidate source/language",
            )
            vector = np.asarray(self.vectors[self.ix[h["id"]]], dtype=np.float64)
            q = np.asarray(query, dtype=np.float64)
            cosine = float(vector @ q / (np.linalg.norm(vector) * np.linalg.norm(q)))
            error = abs(h["score"] - cosine)
            self.max_dense_score_error = max(self.max_dense_score_error, error)
            compare_hits(
                [h],
                [{"id": record["id"], "kind": kind, "via": "dense", "ranks": {}, "score": cosine}],
                3e-06,
            )
        self.counts["dense_query"] += 1
        return hits

    def mmr(self, event, candidates, query, top_n):
        need(
            {x: event[x] for x in event if x != "hits"}
            == {
                "operation": "mmr",
                "input_ids": plain_hits(candidates),
                "lambda": 0.7,
                "top_n": top_n,
            },
            "MMR candidate linkage/config",
        )
        vectors = self.vectors[[self.ix[h["id"]] for h in candidates]]
        chosen, margin = independent_mmr(query, vectors, top_n)
        compare_hits(event["hits"], [candidates[i] for i in chosen])
        if margin is not None:
            self.min_mmr_margin = min(self.min_mmr_margin, margin)
        self.counts["mmr"] += 1
        return event["hits"]

    def rrf(self, event, dense, lexical, k):
        channels = {"dense": dense, "bm25": lexical}
        need(
            {x: event[x] for x in event if x != "hits"}
            == {"operation": "rrf", "channels": channels, "k": 60, "top_n": k},
            "RRF channel linkage/config",
        )
        compare_hits(event["hits"], independent_rrf(channels, k), 1e-14)
        self.counts["rrf"] += 1
        return event["hits"]

    def entry(self, entry):
        strategy, trace, lang = (entry["strategy"], entry["trace"], entry["lang"])
        q, text = (self.queries[entry["key"]], entry["text"])
        if strategy == "dense":
            need(len(trace) == 5, "Dense trace length")
            candidates = self.dense(trace[0], "example", 15, lang, q, True)
            examples = self.mmr(trace[1], candidates, q, 5)
            definitions = self.dense(trace[2], "definition", 2, lang, q)
            guidelines = self.dense(trace[3], "guideline", 2, lang, q)
            self.dense(trace[4], "feedback", 0, lang, q)
        elif strategy == "bm25":
            need(len(trace) == 4, "BM25 trace length")
            definitions = self.lexical(trace[0], "definition", 2, text)
            guidelines = self.lexical(trace[1], "guideline", 2, text)
            examples = self.lexical(trace[2], "example:" + lang, 5, text)
            self.lexical(trace[3], "feedback:" + lang, 0, text)
        else:
            need(strategy == "hybrid" and len(trace) == 10, "Hybrid trace shape")
            definitions = self.rrf(
                trace[2],
                self.dense(trace[0], "definition", 10, lang, q),
                self.lexical(trace[1], "definition", 10, text),
                2,
            )
            guidelines = self.rrf(
                trace[5],
                self.dense(trace[3], "guideline", 10, lang, q),
                self.lexical(trace[4], "guideline", 10, text),
                2,
            )
            dense = self.mmr(trace[7], self.dense(trace[6], "example", 30, lang, q, True), q, 10)
            examples = self.rrf(
                trace[9], dense, self.lexical(trace[8], "example:" + lang, 10, text), 5
            )
        compare_hits(entry["final_hits"], definitions + guidelines + examples)
        self.counts["entries"] += 1


def buckets(entry):
    return {
        "definitions": entry["context"]["definitions"],
        "guidelines": entry["context"]["guidelines"],
        "examples": [h for _, hits in entry["context"]["example_groups"] for h in hits],
    }


def counts(entries):
    summary = {}
    for entry in entries:
        name = entry["scopes"][0]["subset"] + "/" + entry["strategy"]
        value = summary.setdefault(name, Counter())
        value["inputs"] += 1
        for bucket, hits in buckets(entry).items():
            value[bucket + "_slots"] += len(hits)
            value[bucket + "_empty_inputs"] += not hits
    return {k: dict(v) for k, v in summary.items()}


def annotation_state(value):
    if value is None:
        return "unannotated"
    if value is False or value == []:
        return "annotated_negative_or_empty"
    return "annotated_value"


def diagnostics(entries, checker):
    summary, examples_by_entry, shortfalls = ({}, {}, [])
    for entry in entries:
        name = entry["scopes"][0]["subset"] + "/" + entry["strategy"]
        value = summary.setdefault(
            name,
            {
                "inputs": 0,
                "buckets": {},
                "example_annotations": {a: Counter() for a in ANNOTATIONS},
                "example_labels": {a: Counter() for a in ANNOTATIONS},
                "redundancies": [],
            },
        )
        value["inputs"] += 1
        for bucket, hits in buckets(entry).items():
            stat = value["buckets"].setdefault(
                bucket,
                {
                    "counts": Counter(),
                    "sources": Counter(),
                    "languages": Counter(),
                    "unique_ids": set(),
                },
            )
            stat["counts"][len(hits)] += 1
            for hit in hits:
                stat["sources"][hit["meta"]["source"]] += 1
                stat["languages"][hit["meta"]["lang"]] += 1
                stat["unique_ids"].add(hit["id"])
            if len(hits) < (5 if bucket == "examples" else 2):
                shortfalls.append(
                    {
                        "key": entry["key"],
                        "strategy": entry["strategy"],
                        "bucket": bucket,
                        "delivered": len(hits),
                    }
                )
        examples = buckets(entry)["examples"]
        examples_by_entry[ident(entry)] = plain_hits(examples)
        for hit in examples:
            for annotation in ANNOTATIONS:
                v = hit["meta"].get(annotation)
                value["example_annotations"][annotation][annotation_state(v)] += 1
                if v is not None:
                    for label in v if isinstance(v, list) else [v]:
                        value["example_labels"][annotation][
                            json.dumps(label, ensure_ascii=False)
                        ] += 1
        if len(examples) >= 2:
            vectors = np.asarray(
                checker.vectors[[checker.ix[h["id"]] for h in examples]], dtype=np.float64
            )
            vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
            cosines = (vectors @ vectors.T)[np.triu_indices(len(examples), 1)]
            value["redundancies"].append(float(cosines.mean()))
    for value in summary.values():
        redundancy = value.pop("redundancies")
        value["example_pairwise_cosine"] = {
            "inputs_with_two_or_more": len(redundancy),
            "mean_of_input_means": float(np.mean(redundancy)) if redundancy else None,
            "min_input_mean": min(redundancy, default=None),
            "max_input_mean": max(redundancy, default=None),
        }
        for stat in value["buckets"].values():
            stat["unique_records"] = len(stat.pop("unique_ids"))
    overlaps = {}
    by_id = {ident(e): e for e in entries}
    for key in dict.fromkeys((e["key"] for e in entries)):
        subset = by_id[key, "dense"]["scopes"][0]["subset"]
        for left, right in s.PAIRS:
            for bucket in ("definitions", "guidelines", "examples"):
                a = plain_hits(buckets(by_id[key, left])[bucket])
                b = plain_hits(buckets(by_id[key, right])[bucket])
                stat = overlaps.setdefault(
                    subset + "/" + left + "-" + right + "/" + bucket,
                    {
                        "inputs": 0,
                        "ordered_equal": 0,
                        "set_equal": 0,
                        "intersection_slots": 0,
                        "jaccards": [],
                    },
                )
                stat["inputs"] += 1
                stat["ordered_equal"] += a == b
                stat["set_equal"] += set(a) == set(b)
                intersection, union = (len(set(a) & set(b)), len(set(a) | set(b)))
                stat["intersection_slots"] += intersection
                stat["jaccards"].append(intersection / union if union else 1.0)
    for stat in overlaps.values():
        stat["mean_jaccard"] = float(np.mean(stat.pop("jaccards")))
    return {
        "by_endpoint_strategy": summary,
        "pairwise_bucket_overlaps": overlaps,
        "shortfall_records": shortfalls,
        "definition": "Annotation/label/source counts are delivered TRAIN record occurrences, not DEV gold. Redundancy uses cached record cosine, input-weighted; descriptive, not relevance or quality.",
    }


def database_findings(integrity, foreign, orphans):
    need(integrity == [("ok",)], "SQLite corruption")
    need(
        foreign == [("segments", 1, "collection", 0), ("segments", 2, "collection", 0)]
        and orphans == 0,
        "Unexpected SQLite foreign-key finding",
    )


def check_copy(receipt, plan, records):
    copy = s.ROOT / receipt["attempt"] / "chroma"
    current = s.files_under(copy)
    source = {
        (s.ROOT / name).relative_to(s.SOURCE_INDEX).as_posix(): digest
        for name, digest in plan["source_index_hashes"].items()
    }
    local = {
        (s.ROOT / name).relative_to(copy).as_posix(): digest for name, digest in current.items()
    }
    need(set(local) == set(source), "Copied index file roster")
    changed = [name for name in local if local[name] != source[name]]
    need(set(changed) <= {"chroma.sqlite3"}, "Copied vector graph changed")
    in_process = {
        (s.ROOT / name).relative_to(copy).as_posix(): digest
        for name, digest in receipt["copied_index_after_in_process"].items()
    }
    need(set(in_process) == set(local), "In-process index file roster")
    need(
        all((in_process[n] == source[n] for n in source if n != "chroma.sqlite3")),
        "In-process vector graph changed",
    )
    with sqlite3.connect((copy / "chroma.sqlite3").as_uri() + "?mode=ro", uri=True) as connection:
        integrity = connection.execute("PRAGMA integrity_check").fetchall()
        foreign = connection.execute("PRAGMA foreign_key_check").fetchall()
        orphans = connection.execute(
            "SELECT COUNT(*) FROM segments s LEFT JOIN collections c ON s.collection=c.id WHERE c.id IS NULL"
        ).fetchone()[0]
        database_findings(integrity, foreign, orphans)
        ids = connection.execute("SELECT id, embedding_id FROM embeddings").fetchall()
        need(
            len(ids) == len(records) and {rid for _, rid in ids} == set(records),
            "SQLite record roster",
        )
        metadata = {i: {} for i, _ in ids}
        for i, key, string, integer, floating, boolean in connection.execute(
            "SELECT * FROM embedding_metadata"
        ):
            values = [v for v in (string, integer, floating, boolean) if v is not None]
            need(
                i in metadata and len(values) == 1 and (key not in metadata[i]),
                "SQLite metadata schema",
            )
            value = bool(boolean) if boolean is not None else values[0]
            if key != "chroma:document":
                if value in ("", "__none__"):
                    value = None
                elif isinstance(value, str) and value.startswith("["):
                    value = json.loads(value)
            metadata[i][key] = value
        for i, rid in ids:
            record = records[rid]
            need(
                r.js(metadata[i])
                == r.js({**r.record_meta(record), "chroma:document": record["text"]}),
                "SQLite text/metadata differs: " + rid,
            )
    need(current == s.files_under(copy), "Read-only SQLite audit mutated files")
    return {
        "current_hashes": current,
        "changed_vs_source": changed,
        "changed_after_process_exit": [n for n in local if local[n] != in_process[n]],
        "integrity": "ok",
        "inherited_foreign_key_flags": foreign,
        "actual_orphans": orphans,
        "independently_verified_record_texts_and_annotations": len(ids),
    }


def difference_detail(before, after):
    changes = []
    for i, (old, new) in enumerate(zip(before["trace"], after["trace"])):
        if old != new:
            a, b = (plain_hits(old["hits"]), plain_hits(new["hits"]))
            changes.append(
                {
                    "event": i,
                    "operation": old["operation"],
                    "old_ids": a,
                    "new_ids": b,
                    "removed": [x for x in a if x not in b],
                    "added": [x for x in b if x not in a],
                }
            )
    return {
        "key": before["key"],
        "strategy": before["strategy"],
        "scope": before["scopes"][0],
        "old_messages_sha256": before["messages_sha256"],
        "new_messages_sha256": after["messages_sha256"],
        "old_retrieved": before["retrieved"],
        "new_retrieved": after["retrieved"],
        "trace_changes": changes,
        "changed_context": before["context"] != after["context"],
        "changed_messages": before["messages"] != after["messages"],
    }


def main():
    need(
        not EVIDENCE.exists() and (not (s.BASE / "retrieval_review_v1.json").exists()),
        "Preserve completed full audit",
    )
    need(r.sha(s.PLAN) == PLAN_SHA, "Frozen plan changed")
    plan = s.validate()
    tests_path = s.BASE / "development/full_audit_tests_v1.json"
    tests = r.read(tests_path)
    need(tests["status"] == "passed", "Audit tests must pass first")
    r.verify_hashes(tests["source_hashes"])
    receipts = {kind: s.verified_receipt(kind) for kind in ("smoke", "prepare", "replay")}
    need(all(receipts.values()), "All preparation receipts required")
    dev = r.rows(s.CORE / "dev_inputs.jsonl")
    s.roster_check(dev, plan["endpoints_inherited_masks_and_labels"])
    rows = {x["key"]: x for x in dev}
    records = r.rows(s.CORE / "kb/full/records.jsonl")
    known = {x["id"]: x for x in records}
    need(len(known) == len(records) == 71131, "Source record roster")
    vectors = np.load(r.BASE / "full_vectors.npy", mmap_mode="r", allow_pickle=False)
    query = np.load(r.BASE / "query_vectors.npy", allow_pickle=False)
    r.validate_vectors(vectors, len(records))
    r.validate_vectors(query, len(dev))
    checker = TraceAudit(records, vectors, {row["key"]: query[i] for i, row in enumerate(dev)})
    expected = [
        (row["key"], strategy)
        for i, row in enumerate(dev)
        for strategy in s.STRATEGIES[i % 3 :] + s.STRATEGIES[: i % 3]
    ]
    entries = r.rows(s.ROOT / receipts["prepare"]["prompts"])
    matrix_check(entries, expected)
    first = {ident(e): e for e in entries}
    for i, entry in enumerate(entries):
        audit_context(entry, rows[entry["key"]], known)
        checker.entry(entry)
        if (i + 1) % 600 == 0:
            print(
                f"Audited {i + 1}/3885 first-pass prompts and independent ranking calculations",
                flush=True,
            )
    first_counts = dict(checker.counts)
    replay_log = r.rows(s.ROOT / receipts["replay"]["attempt"] / "comparisons.jsonl")
    matrix_check(replay_log, expected)
    replay, differences, changed = ([], Counter(), [])
    for saved in replay_log:
        actual = compare_replay(first[ident(saved)], saved)
        replay.append(actual)
        for field, equal in saved["matches"].items():
            differences[field] += not equal
        if not saved["matches"]["entry"]:
            audit_context(actual, rows[actual["key"]], known)
            checker.entry(actual)
            changed.append(difference_detail(first[ident(saved)], actual))
    need(dict(differences) == receipts["replay"]["replay_differences"], "Replay summary differs")
    need(receipts["prepare"]["replay_differences"] == {}, "Unexpected first-pass replay summary")
    historical = {x["key"]: x for x in r.rows(r.BASE / "full/prompts.jsonl")}
    history, copies, runtimes, hashes = (
        {},
        {},
        {},
        {**plan["input_hashes"], **plan["source_index_hashes"]},
    )
    for kind, selected in [("prepare", entries), ("replay", replay)]:
        receipt = receipts[kind]
        attempt = s.ROOT / receipt["attempt"]
        need(
            list((s.BASE / "runs" / kind).glob("attempt_*")) == [attempt],
            "Additional attempt must be reviewed",
        )
        need(
            receipt["actual_API_calls"] == 0
            and receipt["gold_scoring"] is False
            and (receipt["source_index_unchanged"] is True),
            "Execution scope",
        )
        need(
            receipt["copied_index_before"] == plan["source_index_hashes"]
            and receipt["index_vector_max_abs_difference"] < 2e-07,
            "Copied source/vector contract",
        )
        need(counts(selected) == receipt["bucket_counts"], "Independent bucket summary differs")
        hrows = r.rows(attempt / "core_hybrid_comparisons.jsonl")
        hybrids = [x for x in selected if x["strategy"] == "hybrid"]
        need(
            [h["key"] for h in hrows] == [x["key"] for x in hybrids],
            "Historical comparison roster/order",
        )
        hcounts, hchanges = (Counter(), [])
        for h, e in zip(hrows, hybrids):
            old = historical[e["key"]]
            flags = {k: old[k] == e[k] for k in ("context", "messages", "retrieved")}
            need(
                h
                == {
                    "key": e["key"],
                    "matches": flags,
                    "old_messages_sha256": old["messages_sha256"],
                    "new_messages_sha256": e["messages_sha256"],
                },
                "Historical record mismatch",
            )
            for k, value in flags.items():
                hcounts[k] += not value
            if not all(flags.values()):
                hchanges.append(h)
        need(
            dict(hcounts) == receipt["historical_hybrid_differences"], "Historical summary differs"
        )
        history[kind] = {"differences": dict(hcounts), "changed_records": hchanges}
        timing = r.rows(attempt / "timings.jsonl")
        matrix_check(timing, expected)
        need(
            all((math.isfinite(x["seconds"]) and x["seconds"] >= 0 for x in timing)),
            "Invalid timing",
        )
        start = r.read(attempt / "start.json")
        need(start["kind"] == kind and start["plan_sha256"] == PLAN_SHA, "Start receipt binding")
        seconds = (
            datetime.fromisoformat(receipt["completed_utc"])
            - datetime.fromisoformat(start["created_utc"])
        ).total_seconds()
        measured = sum((x["seconds"] for x in timing))
        need(seconds > measured > 0, "Invalid run duration")
        runtimes[kind] = {
            "wall_seconds": seconds,
            "retrieval_entry_seconds": measured,
            "by_strategy_seconds": {
                st: sum((x["seconds"] for x in timing if x["strategy"] == st))
                for st in s.STRATEGIES
            },
        }
        copies[kind] = check_copy(receipt, plan, known)
        hashes.update(copies[kind]["current_hashes"])
        print(
            f"{kind}: receipt, history, replay and all 71,131 copied SQL records checked",
            flush=True,
        )
    for kind, receipt in receipts.items():
        hashes.update(receipt["hashes"])
        path = s.BASE / "runs" / kind / "complete.json"
        hashes[r.rel(path)] = r.sha(path)
    for path in [
        s.PLAN,
        Path(__file__),
        Path(__file__).with_name("tests/knowledge/strategies/test_verify.py"),
        tests_path,
        s.BASE / "preflight.json",
        s.ROOT / "Thesis_Log/astra_result/evidence/sq1_strategy_preflight_20260920_v1.json",
        s.ROOT
        / "Thesis_Log/astra_result/evidence/sq1_strategy_metadata_roundtrip_20260920_v1.json",
        s.ROOT / ".venv/Lib/site-packages/rank_bm25.py",
    ] + sorted((s.BASE / "logs").glob("*.txt")):
        hashes[r.rel(path)] = r.sha(path)
    diagnostic = diagnostics(entries, checker)
    r.verify_hashes(hashes)
    s.check_sources(plan)
    need(
        not any(("api" in p.name.lower() for p in s.BASE.iterdir())),
        "Unreviewed API execution directory",
    )
    report = {
        "created_utc": s.utc(),
        "status": "saved_retrieval_verified_with_documented_replay_variation",
        "plan_sha256": PLAN_SHA,
        "first_pass_entries": len(entries),
        "audit_tests": tests["tests_run"],
        "first_pass_trace_checks": first_counts,
        "all_trace_checks_including_seven_changed_replay_entries": dict(checker.counts),
        "independent_dense_score_max_error": checker.max_dense_score_error,
        "minimum_mmr_greedy_selection_margin": checker.min_mmr_margin,
        "replay_differences": dict(differences),
        "replay_changed_entries": changed,
        "historical_hybrid": history,
        "runtime": runtimes,
        "copy_checks": copies,
        "retrieval_diagnostics": diagnostic,
        "source_record_counts": dict(Counter((x["kind"] + "/" + x["lang"] for x in records))),
        "verified_hash_count": len(hashes),
        "verified_hashes": hashes,
        "original_index_unchanged": True,
        "actual_API_calls": 0,
        "actual_gold_scoring": False,
        "new_retrieval_calls": 0,
        "qualifications": [
            "First-pass prompts remain the prospective classifier inputs, including both later-different prompts. No favorable selection, replay substitution or retry-until-match.",
            "Seven changed replay entries are independently checked using saved actual alternatives. For unchanged replay entries only comparison flags were saved: exact equality relies on the pinned worker and hash-bound comparison log, not a second full prompt file.",
            "MMR is independently recomputed from audited cached FP32 vectors and every selected ID agrees. Workers checked copied-index vectors within 2.98e-8; this audit does not reopen Chroma or reproduce its internal float operations.",
            "Dense candidate scores agree with independent cosine arithmetic. Approximate candidate lists vary; exact internal nondeterminism cause and full-pool nearest-neighbour recall are not established.",
            "Classifier repeats will condition on frozen retrieval; they do not measure retrieval-run variance. Historical hybrid comparisons are provenance, not new accuracy or extra repeats.",
            "TRAIN annotation/label coverage and cosine redundancy are descriptive. No DEV gold rescoring, relevance judgment or outcome-driven tuning.",
        ],
        "next": "Record qualified acceptance and freeze a separate interleaved staged API runner using the first-pass prompt file. Do not rerun retrieval or launch Qwen in this audit.",
    }
    EVIDENCE.parent.mkdir(parents=True, exist_ok=True)
    with EVIDENCE.open("x", encoding="utf8", newline="\n") as handle:
        handle.write(r.js(report).decode("utf8"))
    s.save(
        s.BASE / "retrieval_review_v1.json",
        {
            "status": report["status"],
            "evidence": r.rel(EVIDENCE),
            "evidence_sha256": r.sha(EVIDENCE),
            "plan_sha256": PLAN_SHA,
            "prompts": receipts["prepare"]["prompts"],
            "prompts_sha256": r.sha(s.ROOT / receipts["prepare"]["prompts"]),
            "API_execution_ready": False,
        },
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "first_pass_entries",
                    "first_pass_trace_checks",
                    "replay_differences",
                    "independent_dense_score_max_error",
                    "minimum_mmr_greedy_selection_margin",
                    "verified_hash_count",
                )
            },
            indent=2,
        )
    )
    print("Evidence SHA256: " + r.sha(EVIDENCE))


if __name__ == "__main__":
    with s.patch.object(s.socket.socket, "connect", s.forbidden), s.patch.object(
        s.socket, "create_connection", s.forbidden
    ):
        main()

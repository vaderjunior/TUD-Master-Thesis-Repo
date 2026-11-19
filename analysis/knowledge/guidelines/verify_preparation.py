"""Guidelines verify preparation."""

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import io
import json
import math
import os
import re
import time
import unittest
import numpy as np
import yaml
from analysis.knowledge.guidelines import prepare as p
from analysis.knowledge.strategies import verify as a

r = p.r


class OrderedLexicon:
    """Sparse independent BM25; arithmetic grouping from frozen allocation v2."""

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


class TraceAudit(a.TraceAudit):

    def __init__(self, records, vectors, queries):
        self.records = {x["id"]: x for x in records}
        self.ix = {x["id"]: i for i, x in enumerate(records)}
        self.vectors, self.queries = (vectors, queries)
        buckets = defaultdict(list)
        for record in records:
            key = record["kind"] + (":" + record["lang"] if record["kind"] == "example" else "")
            buckets[key].append(record)
        self.lexicons = {key: OrderedLexicon(values) for key, values in buckets.items()}
        self.cache = {}
        self.counts = Counter()
        self.max_dense_score_error = 0.0
        self.min_mmr_margin = 1.0


def render_independently(text, language, context):
    sections = {}
    name = None
    for line in (p.ROOT / "prompts/classify_v1.txt").read_text(encoding="utf8").splitlines():
        if line.startswith("=== ") and line.endswith(" ==="):
            name = line[4:-4]
            sections[name] = []
        elif name is not None:
            sections[name].append(line)
    tpl = {name: "\n".join(lines).strip() for name, lines in sections.items()}
    taxonomy = yaml.safe_load((p.ROOT / "config/taxonomy.yaml").read_text(encoding="utf8"))
    label_lines = []
    for name, info in taxonomy["dimensions"].items():
        labels = "true or false" if info["type"] == "binary" else ", ".join(info["labels"])
        label_lines.append("- " + name + " (" + info["type"] + "): " + labels)
    system = tpl["SYSTEM"].format(label_space="\n".join(label_lines))
    flatten = lambda value: " ".join(str(value).split())
    parts = []
    if context["definitions"]:
        definitions = []
        for hit in context["definitions"]:
            label = " / " + hit["meta"]["label"] if hit["meta"].get("label") else ""
            definitions.append(
                "- " + str(hit["meta"].get("dimension")) + label + ": " + flatten(hit["text"])
            )
        parts.append("## Label definitions\n" + "\n".join(definitions))
    if context["guidelines"]:
        parts.append(
            "## Annotation guidelines\n"
            + "\n".join(("- " + h["text"] for h in context["guidelines"]))
        )
    for title, hits in context["example_groups"]:
        if not hits:
            continue
        key = (
            "ILLUSTRATIONS_CAUTION"
            if any((h["meta"].get("illustrative_only") for h in hits))
            else (
                "STATIC_EXAMPLES_CAUTION"
                if any((h["via"] == "static" for h in hits))
                else "EXAMPLES_CAUTION"
            )
        )
        examples = []
        for hit in hits:
            meta = hit["meta"]
            prefix = '- "' + flatten(hit["text"]) + '"\n  '
            if meta.get("illustrative_only"):
                examples.append(
                    prefix + "flagged under StGB §" + str(meta.get("stgb", "unspecified"))
                )
                continue
            annotations = []
            if meta.get("gate") is not None:
                r.require(type(meta["gate"]) is bool, "Nonboolean gate annotation")
                annotations.append("hate=" + ("true" if meta["gate"] else "false"))
            for source, label in [("target_groups", "target_group"), ("hate_types", "hate_type")]:
                if meta.get(source) is not None:
                    annotations.append(label + "=[" + ", ".join(meta[source]) + "]")
            if meta.get("severity") is not None:
                annotations.append("severity=" + str(meta["severity"]))
            if meta.get("legal") is not None:
                annotations.append("legal=[" + ", ".join(meta["legal"]) + "]")
            rendered = ", ".join(annotations)
            if meta.get("astra_aux_stgb"):
                rendered += (
                    "\n  Auxiliary DeTox paragraph flags (illustrative, not legal gold): §"
                    + meta["astra_aux_stgb"]
                )
            examples.append(prefix + "labels: " + rendered)
        parts.append("## " + title + "\n" + tpl[key] + "\n\n" + "\n".join(examples))
    parts.append("## Text to classify (language: " + language + ")\n" + text)
    return [{"role": "system", "text": system}, {"role": "user", "text": "\n\n".join(parts)}]


def expected_context(row, base, condition, mapping, records):
    rule = mapping[row["functionality"]]
    if condition == "no_guidelines":
        ids = []
    elif condition == "matched_rule":
        ids = [rule["matched"], rule["filler"]]
    elif condition == "alternative_rule":
        ids = [rule["alternative"], rule["filler"]]
    elif condition == "filler_only":
        ids = [rule["filler"]]
    else:
        r.require(condition == "natural", "Unknown audit condition")
        ids = None
    context = deepcopy(base["context"])
    if ids is not None:
        context["guidelines"] = []
        for rid in ids:
            record = records[rid]
            r.require(
                record["kind"] == "guideline" and record["lang"] == "en",
                "Forced source rule differs",
            )
            metadata = {k: record[k] for k in ("kind", "lang", "source", "dimension", "label")}
            metadata.update(record.get("meta", {}))
            context["guidelines"].append(
                {"id": rid, "text": record["text"], "meta": metadata, "via": "oracle"}
            )
    return context


def audit_arm(row, base, entry, mapping, records):
    r.require(
        set(entry)
        == set(p.IDENTITY)
        | {
            "condition",
            "pool",
            "strategy",
            "plan_sha256",
            "input_sha256",
            "background_sha256",
            "context",
            "context_sha256",
            "messages",
            "messages_sha256",
            "retrieved",
        },
        "Arm entry schema",
    )
    r.require(
        all((entry[k] == row[k] for k in p.IDENTITY)) and entry["input_sha256"] == r.signature(row),
        "Arm input binding",
    )
    r.require(
        entry["plan_sha256"] == p.PLAN_SHA
        and entry["pool"] == "full"
        and (entry["strategy"] == "hybrid"),
        "Arm contract",
    )
    r.require(entry["background_sha256"] == r.signature(base), "Arm background binding")
    context = expected_context(row, base, entry["condition"], mapping, records)
    r.require(
        r.js(context) == r.js(entry["context"]) and entry["context_sha256"] == r.signature(context),
        "Arm context mismatch",
    )
    expected_ids = {
        "definitions": [h["id"] for h in context["definitions"]],
        "guidelines": [h["id"] for h in context["guidelines"]],
    }
    expected_ids.update({name: [h["id"] for h in hits] for name, hits in context["example_groups"]})
    r.require(entry["retrieved"] == expected_ids, "Arm retrieval IDs")
    messages = render_independently(row["text"], row["lang"], context)
    r.require(
        r.js(messages) == r.js(entry["messages"])
        and entry["messages_sha256"] == r.signature(messages),
        "Independent arm rendering differs",
    )
    if entry["condition"] == "natural":
        r.require(r.js(messages) == r.js(base["messages"]), "Natural baseline message differs")

    def without_guidelines(value, ctx):
        copy = deepcopy(value)
        if ctx["guidelines"]:
            block = "\n\n## Annotation guidelines\n" + "\n".join(
                ("- " + h["text"] for h in ctx["guidelines"])
            )
            r.require(copy[1]["text"].count(block) == 1, "Ambiguous guideline section")
            copy[1]["text"] = copy[1]["text"].replace(block, "", 1)
        return copy

    r.require(
        r.js(without_guidelines(messages, context))
        == r.js(without_guidelines(base["messages"], base["context"])),
        "Other prompt wording changed",
    )


def audit_background(row, entry, records, trace_checker):
    r.require(
        set(entry)
        == set(p.IDENTITY)
        | {
            "pool",
            "strategy",
            "plan_sha256",
            "context",
            "context_sha256",
            "messages",
            "messages_sha256",
            "retrieved",
            "final_hits",
            "trace",
        },
        "Background schema",
    )
    r.require(all((entry[k] == row[k] for k in p.IDENTITY)), "Background input differs")
    r.require(
        entry["plan_sha256"] == p.PLAN_SHA
        and entry["pool"] == "full"
        and (entry["strategy"] == "hybrid"),
        "Background contract",
    )
    trace_checker.entry(entry)
    context = a.source_context(entry["final_hits"], records)
    r.require(
        r.js(context) == r.js(entry["context"]) and entry["context_sha256"] == r.signature(context),
        "Natural source context differs",
    )
    r.validate_context(context, records, row["lang"])
    ids = {
        "definitions": [h["id"] for h in context["definitions"]],
        "guidelines": [h["id"] for h in context["guidelines"]],
    }
    ids.update({name: [h["id"] for h in hits] for name, hits in context["example_groups"]})
    r.require(ids == entry["retrieved"], "Natural source IDs differ")
    messages = render_independently(row["text"], row["lang"], context)
    r.require(
        r.js(messages) == r.js(entry["messages"])
        and entry["messages_sha256"] == r.signature(messages),
        "Independent natural rendering differs",
    )


def assert_matrix(rows, backgrounds, entries):
    r.require(
        [x["key"] for x in backgrounds] == [x["key"] for x in rows],
        "Missing/duplicate/reordered background",
    )
    r.require(
        [(x["key"], x["condition"]) for x in entries]
        == [(x["key"], c) for x in rows for c in p.CONDITIONS],
        "Missing/duplicate/reordered prompt",
    )


def replay_flags(old, new, fields):
    result = {name: r.signature(old[name]) == r.signature(new[name]) for name in fields}
    result["entry"] = r.signature(old) == r.signature(new)
    return result


def tests():
    from tests.knowledge.guidelines import test_prepare as module

    r.require(
        not p.TESTS.exists() and (not p.MANIFEST.exists()), "Preserve frozen tests/implementation"
    )
    attempt = p.IMPL / "development" / ("tests_" + str(time.time_ns()))
    hashes = {name: r.sha(p.ROOT / name) for name in p.SOURCE_FILES}
    p.save(
        attempt / "sources.json",
        {
            name: (p.ROOT / name).read_text(encoding="utf8")
            for name in p.SOURCE_FILES
            if "guideline" in name
        },
    )
    stream = io.StringIO()
    start = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(module)
    )
    print(stream.getvalue(), flush=True)
    value = {
        "created_utc": p.s.utc(),
        "passed": result.wasSuccessful(),
        "tests": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "log": stream.getvalue(),
        "source_hashes": hashes,
        "environment": p.environment(),
        "seconds": time.perf_counter() - start,
        "actual_API_calls": 0,
        "new_retrieval_calls": 0,
        "new_embeddings": 0,
    }
    p.save(attempt / "result.json", value)
    r.verify_hashes(hashes)
    r.require(result.wasSuccessful(), "Guideline preparation tests failed; preserve this attempt")
    p.save(p.TESTS, value)


def check(mode):
    dest = p.IMPL / (mode + "_review")
    r.require(not (dest / "complete.json").exists(), "Preserve completed review")
    plan = p.validate()
    started = time.perf_counter()
    first = p.receipt("smoke" if mode == "smoke" else "prepare")
    replay = p.receipt("replay") if mode == "full" else None
    r.require(
        first is not None and (mode == "smoke" or replay is not None),
        "Required preparation incomplete",
    )
    rows, queries, all_records, vectors = p.load_inputs(plan)
    records = {x["id"]: x for x in all_records}
    mapping = r.read(p.INVENTORY / "rule_mapping.json")
    checker = TraceAudit(all_records, vectors, {x["key"]: queries[i] for i, x in enumerate(rows)})
    if mode == "smoke":
        by_key = {x["key"]: x for x in rows}
        rows = [by_key[k] for k in plan["smoke_keys"]]
    source_hashes = {}
    copies = []
    all_bases = []
    all_entries = []
    counts = Counter()
    for receipt in [first, replay] if replay else [first]:
        bases = r.rows(p.ROOT / receipt["backgrounds"])
        entries = r.rows(p.ROOT / receipt["prompts"])
        assert_matrix(rows, bases, entries)
        copies.append(a.check_copy(receipt, plan, records))
        source_hashes.update(receipt["hashes"])
        source_hashes[r.rel(p.IMPL / "runs" / receipt["kind"] / "complete.json")] = r.sha(
            p.IMPL / "runs" / receipt["kind"] / "complete.json"
        )
        all_bases.append(bases)
        all_entries.append(entries)
        for i, (row, base) in enumerate(zip(rows, bases)):
            audit_background(row, base, records, checker)
            counts["natural_backgrounds"] += 1
            for entry in entries[i * 5 : i * 5 + 5]:
                audit_arm(row, base, entry, mapping, records)
                counts["prompt_entries"] += 1
            if (i + 1) % 150 == 0 or i + 1 == len(rows):
                print(
                    f"AUDIT {receipt['kind']}: {i + 1}/{len(rows)} inputs ({time.perf_counter() - started:.1f}s)",
                    flush=True,
                )
    bdiff = Counter()
    pdiff = Counter()
    changes = []
    if replay:
        expected_checks = []
        for i, row in enumerate(rows):
            flags = replay_flags(
                all_bases[0][i],
                all_bases[1][i],
                ("context", "messages", "retrieved", "trace", "final_hits"),
            )
            bdiff.update((k for k, equal in flags.items() if not equal))
            expected_checks.append({"key": row["key"], "kind": "background", "matches": flags})
            if not flags["entry"]:
                changes.append({"key": row["key"], "kind": "background", "matches": flags})
            for j, condition in enumerate(p.CONDITIONS):
                flags = replay_flags(
                    all_entries[0][i * 5 + j],
                    all_entries[1][i * 5 + j],
                    ("context", "messages", "retrieved"),
                )
                pdiff.update((k for k, equal in flags.items() if not equal))
                expected_checks.append(
                    {"key": row["key"], "condition": condition, "kind": "prompt", "matches": flags}
                )
                if not flags["entry"]:
                    changes.append(
                        {
                            "key": row["key"],
                            "condition": condition,
                            "kind": "prompt",
                            "matches": flags,
                        }
                    )
        r.require(
            r.rows(p.ROOT / replay["attempt"] / "comparisons.jsonl") == expected_checks,
            "Saved replay flags differ from independent comparison",
        )
        r.require(
            replay["background_replay_differences"]
            == {
                k: bdiff[k]
                for k in ("context", "messages", "retrieved", "trace", "final_hits", "entry")
            },
            "Background replay totals differ",
        )
        r.require(
            replay["prompt_replay_differences"]
            == {k: pdiff[k] for k in ("context", "messages", "retrieved", "entry")},
            "Prompt replay totals differ",
        )
    coverage = {}
    languages = Counter()
    natural_ids = Counter()
    lengths = defaultdict(list)
    matched_examples = Counter()
    for i, (row, base) in enumerate(zip(rows, all_bases[0])):
        rules = mapping[row["functionality"]]
        hits = base["context"]["guidelines"]
        ids = {h["id"] for h in hits}
        group = coverage.setdefault(row["functionality"], Counter())
        group["inputs"] += 1
        for name in ("matched", "alternative", "filler"):
            group["natural_contains_" + name] += rules[name] in ids
        group["natural_has_german_guideline"] += any((h["meta"]["lang"] == "de" for h in hits))
        languages.update((h["meta"]["lang"] for h in hits))
        natural_ids.update((h["id"] for h in hits))
        examples = [h for _, group_hits in base["context"]["example_groups"] for h in group_hits]
        matched_examples["retrieved_example_slots"] += len(examples)
        matched_examples["non_hate_example_slots"] += sum(
            (h["meta"].get("gate") is False for h in examples)
        )
        matched_examples["hate_example_slots"] += sum(
            (h["meta"].get("gate") is True for h in examples)
        )
        matched_examples["unknown_gate_example_slots"] += sum(
            (h["meta"].get("gate") is None for h in examples)
        )
        matched_examples["inputs_all_five_examples_hateful"] += all(
            (h["meta"].get("gate") is True for h in examples)
        )
        for entry in all_entries[0][i * 5 : i * 5 + 5]:
            lengths[entry["condition"]].append(sum((len(m["text"]) for m in entry["messages"])))
    source_hashes.update(
        {
            r.rel(path): r.sha(path)
            for path in (p.PLAN, p.MANIFEST, p.TESTS, p.BASE / "analysis_plan.json")
        }
    )
    source_hashes.update(r.read(p.MANIFEST)["source_hashes"])
    source_hashes.update(plan["input_hashes"])
    p.validate()
    r.verify_hashes(source_hashes)
    evidence = {
        "created_utc": p.s.utc(),
        "status": "guideline_" + mode + "_prompts_independently_checked",
        "plan_sha256": p.PLAN_SHA,
        "implementation_sha256": r.sha(p.MANIFEST),
        "mode": mode,
        "inputs": len(rows),
        "first_pass_prompts": len(all_entries[0]),
        "audited_counts": dict(counts),
        "independent_retrieval_checks": dict(checker.counts),
        "max_dense_score_error": checker.max_dense_score_error,
        "min_mmr_margin": checker.min_mmr_margin,
        "copied_index_checks": copies,
        "background_replay_differences": {
            k: bdiff[k]
            for k in ("context", "messages", "retrieved", "trace", "final_hits", "entry")
        },
        "prompt_replay_differences": {
            k: pdiff[k] for k in ("context", "messages", "retrieved", "entry")
        },
        "replay_changes": changes,
        "verified_hashes": source_hashes,
        "natural_guideline_coverage": {k: dict(v) for k, v in coverage.items()},
        "natural_guideline_languages": dict(languages),
        "natural_guideline_ids": dict(natural_ids),
        "retrieved_example_annotations": dict(matched_examples),
        "prompt_characters": {
            name: {"min": min(v), "mean": sum(v) / len(v), "max": max(v)}
            for name, v in lengths.items()
        },
        "elapsed_s": time.perf_counter() - started,
        "actual_API_calls": 0,
        "new_embeddings": 0,
        "new_gold_scoring": False,
        "limitations": [
            "All 693 benchmark cases are previously inspected non-hate cases; future scoring is false-alarm analysis only.",
            "ANN membership is observed, not exhaustive nearest-neighbour truth. First-pass contexts are retained regardless of replay.",
            "Natural abstract retrieval is bilingual. Forced guidelines are unchanged English rules.",
            "Alternative/filler may apply too. Lead addition changes length/position; no perfect relevance or mechanism claim.",
            "Inherited Chroma legacy foreign-key flags retained; physical vector graph unchanged and actual orphans zero.",
        ],
        "next": "Publish preparation review. Separate tested API readiness before any named live stage; no automatic API execution.",
    }
    p.save(dest / "evidence.json", evidence)
    p.save(
        dest / "complete.json",
        {
            "status": evidence["status"],
            "hashes": {r.rel(dest / "evidence.json"): r.sha(dest / "evidence.json")},
        },
    )
    print(
        json.dumps(
            {
                k: evidence[k]
                for k in (
                    "status",
                    "inputs",
                    "first_pass_prompts",
                    "audited_counts",
                    "independent_retrieval_checks",
                    "background_replay_differences",
                    "prompt_replay_differences",
                    "elapsed_s",
                )
            },
            indent=2,
        ),
        flush=True,
    )


def main():
    os.chdir(p.ROOT)
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tests", "smoke", "full"])
    args = parser.parse_args()
    with p.offline():
        if args.mode == "tests":
            tests()
        else:
            check(args.mode)


if __name__ == "__main__":
    main()

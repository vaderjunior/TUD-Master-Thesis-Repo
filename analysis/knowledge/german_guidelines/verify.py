"""German guidelines verify."""

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import math
import re
import time
import numpy as np
from analysis.knowledge.german_guidelines import common as c


def rank_independently(records, vectors, query, text, count):
    ids = [r["id"] for r in records]
    tokens = [re.findall("\\w+", r["text"].lower()) for r in records]
    frequencies = [Counter(t) for t in tokens]
    n = len(records)
    lengths = [len(t) for t in tokens]
    avg = sum(lengths) / n
    df = Counter((word for freq in frequencies for word in freq))
    idf = {word: math.log(n - num + 0.5) - math.log(num + 0.5) for word, num in df.items()}
    floor = 0.25 * sum(idf.values()) / len(idf)
    idf = {word: value if value >= 0 else floor for word, value in idf.items()}
    lexical = [
        sum(
            (
                idf.get(word, 0)
                * freq[word]
                * 2.5
                / (freq[word] + 1.5 * (0.25 + 0.75 * length / avg))
                for word in re.findall("\\w+", text.lower())
            )
        )
        for freq, length in zip(frequencies, lengths)
    ]
    q = np.asarray(query, dtype=np.float64)
    dense = [
        float(
            np.dot(np.asarray(v, dtype=np.float64), q)
            / (np.linalg.norm(np.asarray(v, dtype=np.float64)) * np.linalg.norm(q))
        )
        for v in vectors
    ]
    dense_order = sorted(range(n), key=lambda i: (-dense[i], ids[i]))[:10]
    lexical_order = [i for i in sorted(range(n), key=lambda i: -lexical[i])[:10] if lexical[i] > 0]
    scores = defaultdict(float)
    for order in (dense_order, lexical_order):
        for position, i in enumerate(order, 1):
            scores[i] += 1 / (60 + position)
    fused = sorted(scores, key=lambda i: (-scores[i], ids[i]))
    return {
        "selected": [ids[i] for i in fused[:count]],
        "dense_order": [ids[i] for i in dense_order],
        "bm25_order": [ids[i] for i in lexical_order],
        "fused_order": [ids[i] for i in fused],
        "dense_scores": dense,
        "bm25_scores": lexical,
        "fused_scores": {ids[i]: scores[i] for i in fused},
    }


def check_schedule(schedule, cat):
    seen, owners = (set(), {})
    for stage in schedule["stages"]:
        c.require(
            stage["items"] == len(stage["jobs"])
            and stage["ordinary_calls"] == 3 * stage["items"] <= 4050,
            "Stage budget incorrect",
        )
        c.require(
            set(stage["input_keys"]) == {j["key"] for j in stage["jobs"]},
            "Stage input roster differs",
        )
        for j in stage["jobs"]:
            key = (j["key"], j["condition"], j["repeat"])
            c.require(
                key not in seen and j["repeat"] in (0, 1, 2),
                "Repeated or invalid scheduled identity",
            )
            seen.add(key)
            entry = cat[j["key"], j["condition"]]
            c.require(
                j["population"] == entry["population"] == stage["population"]
                and j["messages_sha256"] == entry["messages_sha256"],
                "Schedule/prompt identity differs",
            )
            c.require(
                owners.setdefault(j["key"], stage["id"]) == stage["id"],
                "Input split across audit batches",
            )
    c.require(
        seen == {(key, cond, rep) for key, cond in cat for rep in range(3)} and len(seen) == 33093,
        "Incomplete condition/repeat matrix",
    )
    return {
        "predictions": len(seen),
        "ordinary_calls": 3 * len(seen),
        "batches": len(schedule["stages"]),
    }


def verify():
    plan = c.validate(ready=False)
    originals = {}
    for path in (
        c.CORE / "retrieval_v1/full/prompts.jsonl",
        c.BROAD / "retrieval_v3/full/prompts.jsonl",
    ):
        for row in c.rows(path):
            phase = row.get("population") or row["scopes"][0]["subset"]
            if phase in c.COUNTS:
                c.require(row["key"] not in originals, "Repeated source key")
                originals[row["key"]] = row | {"population": phase}
    prompts = list(c.rows(c.BASE / "prompts.jsonl"))
    cat = {(r["key"], r["condition"]): r for r in prompts}
    c.require(
        len(prompts) == len(cat) == 11031
        and set(cat) == {(k, cond) for k in originals for cond in c.CONDITIONS},
        "Prompt cross-product differs",
    )
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(c.TOKENIZER / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    for row in prompts:
        c.require(
            [m["role"] for m in row["messages"]] == ["system", "user"],
            "Unexpected message role sequence",
        )
        serialized = (
            "".join(
                (
                    "<|im_start|>" + m["role"] + "\n" + m["text"] + "<|im_end|>\n"
                    for m in row["messages"]
                )
            )
            + "<|im_start|>assistant\n"
        )
        n = len(tokenizer.encode(serialized, add_special_tokens=False).ids)
        c.require(
            n == row["reference_input_tokens"] and 0 < n <= 8192,
            "Independent token accounting differs",
        )
    records = list(c.rows(c.BASE / "guideline_records.jsonl"))
    vectors = np.load(c.BASE / "guideline_vectors.npy", allow_pickle=False)
    queries = np.load(c.BASE / "query_vectors.npy", allow_pickle=False)
    order = c.read(c.BASE / "query_order.json")
    c.require(
        order == sorted(originals)
        and vectors.shape == (19, 1024)
        and (queries.shape == (3677, 1024)),
        "Prepared vector identity or dimensions differ",
    )
    for matrix in (vectors, queries):
        c.require(
            np.isfinite(matrix).all()
            and np.all(np.abs(np.linalg.norm(matrix, axis=1) - 1) < 0.0001),
            "Invalid cached vectors",
        )
    source_records = list(c.rows(c.CORE / "kb/full/records.jsonl"))
    source_vectors = np.load(c.CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r")
    selected = [i for i, row in enumerate(source_records) if row["kind"] == "guideline"]
    c.require(
        records == [source_records[i] for i in selected]
        and np.array_equal(vectors, source_vectors[selected]),
        "Guideline text/vector copy differs",
    )
    source_queries = {}
    for matrix_path, origin_path in [
        (
            c.CORE / "retrieval_v1/query_vectors.npy",
            c.CORE / "retrieval_v1/query_vector_sources.jsonl",
        ),
        (
            c.BROAD / "retrieval_v3/vectors/query_vectors.npy",
            c.BROAD / "retrieval_v3/vectors/query_origins.jsonl",
        ),
    ]:
        matrix = np.load(matrix_path, mmap_mode="r")
        for i, row in enumerate(c.rows(origin_path)):
            if row["key"] in originals:
                source_queries[row["key"]] = np.array(matrix[i], copy=True)
    c.require(
        all((np.array_equal(queries[i], source_queries[key]) for i, key in enumerate(order))),
        "Prepared query vectors differ from original rows",
    )
    roster = {r["key"]: r for r in c.rows(c.BASE / "roster.jsonl")}
    traces = {r["key"]: r for r in c.rows(c.BASE / "retrieval_traces.jsonl")}
    c.require(set(roster) == set(traces) == set(originals), "Audit/input roster differs")
    groups = {
        "general": [i for i, row in enumerate(records) if row["dimension"] == "hate"],
        "legal": [i for i, row in enumerate(records) if row["dimension"] == "legal"],
        "mixed_diagnostic": list(range(19)),
    }
    counts = {"general": 2, "legal": 1, "mixed_diagnostic": 2}
    mismatches, identical_general = ([], 0)
    input_counts = Counter()
    for i, key in enumerate(order):
        old, gold, trace = (originals[key], roster[key], traces[key])
        input_counts[gold["phase"]] += 1
        c.require(
            old["messages"][-1]["text"].endswith(
                "## Text to classify (language: de)\n" + gold["text"]
            ),
            "Query characters/whitespace changed",
        )
        c.require(
            trace["query_text_sha256"] == hashlib.sha256(gold["text"].encode()).hexdigest(),
            "Text hash differs",
        )
        checked = {}
        for name, idx in groups.items():
            calculation = rank_independently(
                [records[j] for j in idx], vectors[idx], queries[i], gold["text"], counts[name]
            )
            for field in ("selected", "dense_order", "bm25_order", "fused_order", "fused_scores"):
                c.require(
                    calculation[field] == trace[name][field],
                    "Independent ranking differs: " + field,
                )
            for field in ("dense_scores", "bm25_scores"):
                c.require(
                    np.allclose(calculation[field], trace[name][field], rtol=1e-12, atol=1e-12),
                    "Independent scores differ",
                )
            checked[name] = calculation["selected"]
        old_guides = old["context"]["guidelines"]
        old_ids = [h["id"] for h in old_guides]
        if checked["mixed_diagnostic"] != old_ids:
            mismatches.append(key)
        old_block = "## Annotation guidelines\n" + "\n".join(("- " + h["text"] for h in old_guides))
        c.require(
            old["messages"][-1]["text"].count(old_block) == 1, "Ambiguous original guideline block"
        )
        for cond in c.CONDITIONS:
            row = cat[key, cond]
            c.require(
                row["context_sha256"] == c.signature(row["context"])
                and row["messages_sha256"] == c.signature(row["messages"]),
                "Prepared hashes differ",
            )
            c.require(
                row["context"]["definitions"] == old["context"]["definitions"]
                and row["context"]["example_groups"] == old["context"]["example_groups"],
                "Non-guideline context changed",
            )
            hits = row["context"]["guidelines"]
            ids = [h["id"] for h in hits]
            expected_ids = (
                old_ids
                if cond == "mixed_2"
                else checked["general"] + (checked["legal"] if cond == "general_2_legal_1" else [])
            )
            c.require(
                ids == expected_ids and len(ids) == len(set(ids)), "Guideline allocation differs"
            )
            expected = deepcopy(old["messages"])
            block = "## Annotation guidelines\n" + "\n".join(("- " + h["text"] for h in hits))
            expected[-1]["text"] = expected[-1]["text"].replace(old_block, block, 1)
            c.require(row["messages"] == expected, "Messages changed beyond guideline bullets")
            if cond == "mixed_2":
                c.require(
                    row["messages"] == old["messages"] and row["context"] == old["context"],
                    "Baseline changed",
                )
            if cond == "general_2":
                identical_general += row["messages"] == old["messages"]
            c.a.body(row["messages"])
        if (i + 1) % 500 == 0:
            print(f"Independent ranking and prompt audit: {i + 1}/3677.", flush=True)
    c.require(input_counts == c.COUNTS, "Population counts differ")
    c.require(
        [
            r["key"]
            for r in c.read(c.BASE / "preparation_summary.json")[
                "exact_mixed_vs_saved_ANN_guideline_differences"
            ]
        ]
        == mismatches,
        "ANN diagnostic accounting differs",
    )
    schedule = check_schedule(c.read(c.BASE / "schedule.json"), cat)
    endpoints = c.read(c.BASE / "endpoints.json")
    for phase, endpoint in endpoints.items():
        rr = [r for r in roster.values() if r["phase"] == phase]
        dimension = endpoint["dimension"]
        counts = Counter()
        for row in rr:
            value = row["gate"] if dimension == "hate" else row["legal"]
            c.require(value is not None, "Missing gold on frozen scored endpoint")
            counts.update(["hate" if value else "not_hate"] if dimension == "hate" else value)
        c.require(
            endpoint["gold_support"] == {label: counts[label] for label in endpoint["all_labels"]}
            and set(endpoint["scorable_ids"]) == {r["key"] for r in rr},
            "Independent gold support differs",
        )
    c.require(roster["de/gahd-f4c7eaf2"]["text"] == " ", "Original whitespace-only text lost")
    c.validate(ready=False)
    report = {
        "status": "independent_full_matrix_verification_pass",
        "plan_sha256": c.sha(c.PLAN),
        "source_hashes": c.bindings(c.source_files()),
        "inputs": 3677,
        "prompts": len(cat),
        "schedule": schedule,
        "independently_recomputed_rankings": 3677 * 3,
        "original_baselines_retained": 3677,
        "unchanged_general_only_prompts": identical_general,
        "exact_mixed_vs_original_ANN_different_inputs": mismatches,
        "actual_Qwen_calls": 0,
        "actual_model_list_GETs": 0,
        "new_embeddings": 0,
        "old_sources_preserved": True,
        "network_and_credential_guard": True,
    }
    path = c.BASE / "offline_checks" / f"matrix_{time.time_ns()}.json"
    c.save(path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": c.ref(path),
                "matrix": schedule,
                "mixed_ANN_differences": len(mismatches),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    with c.offline():
        verify()

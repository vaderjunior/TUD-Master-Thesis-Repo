"""German guidelines prepare."""

from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import io
import json
import re
import unicodedata
import numpy as np
from rank_bm25 import BM25Okapi
import yaml
from analysis.knowledge.german_guidelines import common as c
from analysis.common import prompt as prompt
from analysis.common import result_reader as scoring


def tokenize(text):
    return re.findall("\\w+", text.lower())


def normalized(text):
    return re.sub("\\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


class Guidelines:

    def __init__(self, records, vectors):
        self.records = records
        self.ids = [r["id"] for r in records]
        self.vectors = np.asarray(vectors, dtype=np.float64)
        self.vectors /= np.linalg.norm(self.vectors, axis=1, keepdims=True)
        self.bm25 = BM25Okapi([tokenize(r["text"]) for r in records])

    def select(self, text, query, count):
        q = np.asarray(query, dtype=np.float64)
        dense_scores = self.vectors @ (q / np.linalg.norm(q))
        lexical_scores = self.bm25.get_scores(tokenize(text))
        dense = sorted(range(len(self.ids)), key=lambda i: (-dense_scores[i], self.ids[i]))[:10]
        lexical = [
            i
            for i in sorted(range(len(self.ids)), key=lambda i: -lexical_scores[i])[:10]
            if lexical_scores[i] > 0
        ]
        scores = defaultdict(float)
        ranks = defaultdict(dict)
        for name, order in [("dense", dense), ("bm25", lexical)]:
            for rank, i in enumerate(order, 1):
                scores[i] += 1 / (60 + rank)
                ranks[i][name] = rank
        fused = sorted(scores, key=lambda i: (-scores[i], self.ids[i]))
        chosen = fused[:count]
        hits = []
        for i in chosen:
            r = self.records[i]
            meta = {k: r[k] for k in ("kind", "lang", "source", "dimension", "label")}
            meta.update(r.get("meta", {}))
            meta["_ranks"] = ranks[i]
            hits.append({"id": r["id"], "text": r["text"], "meta": meta, "via": "rrf"})
        trace = {
            "candidates": self.ids,
            "dense_order": [self.ids[i] for i in dense],
            "bm25_order": [self.ids[i] for i in lexical],
            "dense_scores": dense_scores.tolist(),
            "bm25_scores": lexical_scores.tolist(),
            "fused_order": [self.ids[i] for i in fused],
            "fused_scores": {self.ids[i]: scores[i] for i in fused},
            "selected": [self.ids[i] for i in chosen],
        }
        return (hits, trace)


def inputs():
    paths = [
        c.CORE / "kb/full/records.jsonl",
        c.CORE / "dev_inputs.jsonl",
        c.CORE / "plan.json",
        c.CORE / "retrieval_v1/plan.json",
        c.CORE / "retrieval_v1/full/complete.json",
        c.CORE / "retrieval_v1/full/prompts.jsonl",
        c.CORE / "retrieval_v1/full_vectors.npy",
        c.CORE / "retrieval_v1/record_vector_sources.jsonl",
        c.CORE / "retrieval_v1/query_vectors.npy",
        c.CORE / "retrieval_v1/query_vector_sources.jsonl",
        c.BROAD / "retrieval_v3/full/prompts.jsonl",
        c.BROAD / "retrieval_v3/complete.json",
        c.BROAD / "retrieval_v3/vectors/complete.json",
        c.BROAD / "retrieval_v3/vectors/query_vectors.npy",
        c.BROAD / "retrieval_v3/vectors/query_origins.jsonl",
        c.BROAD / "replay_review_20261009_v1/sensitivity_plan.json",
        c.PREP / "broad_roster.jsonl",
        c.ROOT / "data/processed/de_dev_eval.parquet",
        c.ROOT / "data/processed/de_legal_dev_eval.parquet",
        c.ROOT / "experiments/astra/german_guideline_review_20261112_v1/inspection.json",
        c.ROOT
        / "experiments/astra/german_guideline_review_20261112_v1/verification_and_scope.json",
        c.TOKENIZER / "complete.json",
        c.TOKENIZER / "tokenizer.json",
        c.TOKENIZER / "chat_template.json",
    ]
    return paths


def anchor_sources():
    expected = {
        c.CORE
        / "kb/full/records.jsonl": "ae63f259519455154c1dcf617b8e649908416f2117b171fc27b0b1c37c3d3961",
        c.CORE
        / "retrieval_v1/full/prompts.jsonl": "ab851c042a1b33dc460c0d2fd60bb7839feb24b3a1d3d4779c5805c3528cc221",
        c.BROAD
        / "retrieval_v3/full/prompts.jsonl": "1205598424561eff4d456fde0940846cb3483e142dfe6e05a610f7e550f6cace",
    }
    for path, digest in expected.items():
        c.require(c.sha(path) == digest, "Historical source differs: " + c.ref(path))
    for receipt, field in [
        (c.CORE / "retrieval_v1/plan.json", "generated_hashes"),
        (c.BROAD / "retrieval_v3/vectors/complete.json", "outputs"),
    ]:
        hashes = c.read(receipt)[field]
        for path in inputs():
            if c.ref(path) in hashes:
                c.require(c.sha(path) == hashes[c.ref(path)], "Previously frozen vector changed")
    assets = c.read(c.TOKENIZER / "complete.json")
    for name in ("tokenizer.json", "chat_template.json"):
        c.require(
            c.sha(c.TOKENIZER / name) == assets["files"][name]["sha256"], "Tokenizer asset changed"
        )


def original_rows():
    result = {}
    for path in (
        c.CORE / "retrieval_v1/full/prompts.jsonl",
        c.BROAD / "retrieval_v3/full/prompts.jsonl",
    ):
        for row in c.rows(path):
            phase = row.get("population") or row["scopes"][0]["subset"]
            if phase not in c.COUNTS:
                continue
            key = row["key"]
            c.require(key not in result and row["lang"] == "de", "Duplicate/non-German input")
            c.require(
                c.signature(row["messages"]) == row["messages_sha256"]
                and c.signature(row["context"]) == row["context_sha256"],
                "Saved prompt hash differs",
            )
            row["population"] = phase
            row["source_prompt_file"] = c.ref(path)
            result[key] = row
    c.require(
        Counter((r["population"] for r in result.values())) == c.COUNTS, "Input population differs"
    )
    return result


def gold_rows():
    import pandas as pd

    result = {}
    for phase in ("de_dev_eval", "de_legal_dev_eval"):
        frame = pd.read_parquet(c.ROOT / f"data/processed/{phase}.parquet")
        for row in json.loads(frame.to_json(orient="records", force_ascii=False)):
            key = row["lang"] + "/" + str(row["id"])
            result[key] = {
                **row,
                "key": key,
                "id": key,
                "phase": phase,
                "text_cluster": hashlib.sha256(normalized(row["text"]).encode()).hexdigest(),
            }
    for row in c.rows(c.PREP / "broad_roster.jsonl"):
        if row["phase"] in c.COUNTS:
            c.require(row["key"] not in result, "DEV/TEST key collision")
            result[row["key"]] = {**row, "id": row["key"]}
    return result


def token_counter():
    from tokenizers import Tokenizer
    from jinja2.sandbox import ImmutableSandboxedEnvironment

    tokenizer = Tokenizer.from_file(str(c.TOKENIZER / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    template = c.read(c.TOKENIZER / "chat_template.json")["chat_template"]
    env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

    def fail(message):
        raise ValueError(message)

    env.globals["raise_exception"] = fail
    compiled = env.from_string(template)

    def count(messages):
        rendered = compiled.render(
            messages=[{"role": m["role"], "content": m["text"]} for m in messages],
            add_generation_prompt=True,
            tools=None,
        )
        return len(tokenizer.encode(rendered, add_special_tokens=False).ids)

    return count


def load_vectors(originals):
    records = list(c.rows(c.CORE / "kb/full/records.jsonl"))
    origins = list(c.rows(c.CORE / "retrieval_v1/record_vector_sources.jsonl"))
    all_vectors = np.load(c.CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r")
    c.require(
        len(records) == len(origins) == len(all_vectors) == 71131, "Record vector alignment differs"
    )
    selected = [i for i, r in enumerate(records) if r["kind"] == "guideline"]
    guides = [records[i] for i in selected]
    for i in selected:
        c.require(
            origins[i]["id"] == records[i]["id"]
            and origins[i]["text_sha256"]
            == hashlib.sha256(records[i]["text"].encode()).hexdigest(),
            "Guideline vector identity differs",
        )
    c.require(
        Counter(((r["lang"], r["dimension"]) for r in guides))
        == {("en", "hate"): 11, ("de", "legal"): 8},
        "Guideline inventory differs",
    )
    guide_vectors = np.array(all_vectors[selected], copy=True)
    queries = {}
    for vector_path, origin_path, hash_field in [
        (
            c.CORE / "retrieval_v1/query_vectors.npy",
            c.CORE / "retrieval_v1/query_vector_sources.jsonl",
            "text_sha256",
        ),
        (
            c.BROAD / "retrieval_v3/vectors/query_vectors.npy",
            c.BROAD / "retrieval_v3/vectors/query_origins.jsonl",
            "exact_text_sha256",
        ),
    ]:
        matrix = np.load(vector_path, mmap_mode="r")
        origins = list(c.rows(origin_path))
        c.require(len(origins) == len(matrix), "Query vector count differs")
        for i, origin in enumerate(origins):
            if origin["key"] in originals:
                c.require(origin["key"] not in queries, "Repeated query vector")
                queries[origin["key"]] = {
                    "vector": np.array(matrix[i], copy=True),
                    "text_sha256": origin[hash_field],
                    "source": c.ref(vector_path),
                    "row": i,
                }
    c.require(set(queries) == set(originals), "Missing frozen query vector")
    return (guides, guide_vectors, queries)


def schedule(prompts):
    by_key = defaultdict(dict)
    for row in prompts:
        by_key[row["key"]][row["condition"]] = row
    stages = []
    for phase in c.COUNTS:
        keys = [k for k, r in by_key.items() if r[c.CONDITIONS[0]]["population"] == phase]
        keys.sort(key=lambda k: hashlib.sha256(("20261112-guidelines|" + k).encode()).hexdigest())
        for start in range(0, len(keys), 150):
            group = keys[start : start + 150]
            jobs = []
            for ordinal, key in enumerate(group, start):
                for block in range(3):
                    repeat = (ordinal + block) % 3
                    for offset in range(3):
                        cond = c.CONDITIONS[(ordinal + repeat + offset) % 3]
                        jobs.append(
                            {
                                "key": key,
                                "condition": cond,
                                "repeat": repeat,
                                "population": phase,
                                "messages_sha256": by_key[key][cond]["messages_sha256"],
                            }
                        )
            stages.append(
                {
                    "id": f"batch_{len(stages) + 1:02d}",
                    "population": phase,
                    "input_keys": group,
                    "jobs": jobs,
                    "items": len(jobs),
                    "ordinary_calls": 3 * len(jobs),
                }
            )
    return {
        "ordering": "Hashed input order, rotating condition/repeat order, all nine predictions per input together.",
        "stages": stages,
    }


def prepare():
    c.require(not c.PLAN.exists(), "Plan already frozen; validate it instead of overwriting")
    anchor_sources()
    before = c.bindings(inputs())
    originals = original_rows()
    gold = gold_rows()
    c.require(set(originals) == set(gold), "Gold and prompt rosters differ")
    guides, guide_vectors, queries = load_vectors(originals)
    general_idx = [i for i, r in enumerate(guides) if r["dimension"] == "hate"]
    legal_idx = [i for i, r in enumerate(guides) if r["dimension"] == "legal"]
    general = Guidelines([guides[i] for i in general_idx], guide_vectors[general_idx])
    legal = Guidelines([guides[i] for i in legal_idx], guide_vectors[legal_idx])
    mixed = Guidelines(guides, guide_vectors)
    count_tokens = token_counter()
    outputs, traces, roster, diagnostics = ([], [], [], [])
    for number, key in enumerate(sorted(originals), 1):
        old = originals[key]
        text = gold[key]["text"]
        q = queries[key]
        c.require(
            hashlib.sha256(text.encode()).hexdigest() == q["text_sha256"],
            "Query text/vector mismatch",
        )
        context = old["context"]
        c.require(
            len(context["definitions"]) == 2
            and len(context["guidelines"]) == 2
            and (sum((len(rr) for _, rr in context["example_groups"])) == 5),
            "Original context budget differs",
        )
        c.require(
            prompt.render_messages(text, "de", context, "corrected") == old["messages"],
            "Original message cannot be reproduced exactly",
        )
        g, gt = general.select(text, q["vector"], 2)
        l, lt = legal.select(text, q["vector"], 1)
        m, mt = mixed.select(text, q["vector"], 2)
        if [h["id"] for h in m] != [h["id"] for h in context["guidelines"]]:
            diagnostics.append(
                {
                    "key": key,
                    "original": [h["id"] for h in context["guidelines"]],
                    "exact_mixed_diagnostic": [h["id"] for h in m],
                }
            )
        traces.append(
            {
                "key": key,
                "general": gt,
                "legal": lt,
                "mixed_diagnostic": mt,
                "query_vector_source": q["source"],
                "query_vector_row": q["row"],
                "query_text_sha256": q["text_sha256"],
            }
        )
        for cond, hits in zip(c.CONDITIONS, [context["guidelines"], g, g + l]):
            ctx = deepcopy(context)
            ctx["guidelines"] = hits
            messages = (
                old["messages"]
                if cond == "mixed_2"
                else prompt.render_messages(text, "de", ctx, "corrected")
            )
            tokens = count_tokens(messages)
            c.require(0 < tokens <= 8192, "Prompt exceeds the fixed reference input-token guard")
            outputs.append(
                {
                    "key": key,
                    "population": old["population"],
                    "lang": "de",
                    "condition": cond,
                    "context": ctx,
                    "context_sha256": c.signature(ctx),
                    "messages": messages,
                    "messages_sha256": c.signature(messages),
                    "reference_input_tokens": tokens,
                    "source_prompt_file": old["source_prompt_file"],
                    "original_messages_sha256": old["messages_sha256"],
                }
            )
        roster.append(
            gold[key]
            | {
                "original_messages_sha256": old["messages_sha256"],
                "original_legal_guidelines": sum(
                    (h["meta"]["dimension"] == "legal" for h in context["guidelines"])
                ),
            }
        )
        if number % 500 == 0:
            print(f"Prepared and checked {number}/3677 German inputs offline.", flush=True)
    c.jsonlines(c.BASE / "prompts.jsonl", outputs)
    c.jsonlines(c.BASE / "retrieval_traces.jsonl", traces)
    c.jsonlines(c.BASE / "roster.jsonl", roster)
    c.jsonlines(c.BASE / "guideline_records.jsonl", guides)
    for name, matrix in [
        ("guideline_vectors.npy", guide_vectors),
        ("query_vectors.npy", np.stack([queries[k]["vector"] for k in sorted(originals)])),
    ]:
        buffer = io.BytesIO()
        np.save(buffer, matrix, allow_pickle=False)
        c.write_bytes(c.BASE / name, buffer.getvalue())
    c.save(c.BASE / "query_order.json", sorted(originals))
    c.save(c.BASE / "schedule.json", schedule(outputs))
    taxonomy = yaml.safe_load((c.ROOT / "config/taxonomy.yaml").read_text(encoding="utf8"))
    endpoints = {}
    for phase in c.COUNTS:
        dimension = "legal" if "legal" in phase else "hate"
        rr = [r for r in roster if r["phase"] == phase]
        endpoints[phase] = scoring.endpoint(rr, dimension, taxonomy)
        c.require(
            len(endpoints[phase]["scorable_ids"]) == c.COUNTS[phase], "Endpoint lost annotated rows"
        )
    c.save(c.BASE / "endpoints.json", endpoints)
    token_stats = {}
    for cond in c.CONDITIONS:
        values = [r["reference_input_tokens"] for r in outputs if r["condition"] == cond]
        token_stats[cond] = {
            "min": min(values),
            "mean": float(np.mean(values)),
            "max": max(values),
            "p95": float(np.quantile(values, 0.95)),
        }
    c.save(
        c.BASE / "preparation_summary.json",
        {
            "inputs": len(roster),
            "prompts": len(outputs),
            "reference_tokens": token_stats,
            "exact_mixed_vs_saved_ANN_guideline_differences": diagnostics,
            "legal_guideline_delivery_by_population": {
                phase: dict(
                    Counter((r["original_legal_guidelines"] for r in roster if r["phase"] == phase))
                )
                for phase in c.COUNTS
            },
            "original_prompt_bytes_retained": True,
            "new_embeddings": 0,
            "actual_Qwen_calls": 0,
            "actual_model_list_GETs": 0,
            "original_files_changed": False,
        },
    )
    c.verify(before)
    prepared = [
        c.BASE / n
        for n in (
            "prompts.jsonl",
            "retrieval_traces.jsonl",
            "roster.jsonl",
            "guideline_records.jsonl",
            "guideline_vectors.npy",
            "query_vectors.npy",
            "query_order.json",
            "schedule.json",
            "endpoints.json",
            "preparation_summary.json",
        )
    ]
    c.save(
        c.PLAN,
        {
            "created_utc": c.a.utc(),
            "status": "frozen_offline_checks_required",
            "conditions": list(c.CONDITIONS),
            "populations": c.COUNTS,
            "workers": c.WORKERS,
            "output_root": c.OUTPUT.as_posix(),
            "deadline": c.DEADLINE.isoformat(),
            "environment": c.environment(),
            "api_contract": c.a.CONTRACT,
            "budget": {
                "inputs": 3677,
                "prompts": 11031,
                "outer_repeats": 3,
                "inner_votes": 3,
                "predictions": 33093,
                "ordinary_calls": 99279,
                "max_logical_calls": 297837,
                "max_transport_POST_attempts": 1489185,
            },
            "scientific_status": "Late exploratory supplement on already exposed DEV and TEST; frozen before new outcomes.",
            "input_hashes": before,
            "prepared_hashes": c.bindings(prepared),
            "source_hashes": c.bindings(c.source_files()),
            "no_accuracy_based_stopping": True,
        },
    )
    print(
        "Prompt matrix frozen. Independent verification and mock runs still required.", flush=True
    )


if __name__ == "__main__":
    with c.offline():
        prepare()

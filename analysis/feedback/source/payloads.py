import argparse
from collections import Counter
import hashlib
import json
import math
import re
import socket
from unittest.mock import patch
import numpy as np
from analysis.feedback.source import baseline_review as review
from analysis.feedback.source import retrieval as retrieval

b = review.b
a = b.a
u = review.u
ROOT = b.ROOT
STUDY = b.STUDY
BASE = STUDY / "update_payloads_v1"
PLAN = BASE / "plan.json"
EVIDENCE = (
    ROOT
    / "Thesis_Log/astra_result/evidence/sq3_feedback_update_payloads_20260929_v1.json"
)
SOURCE_FILES = [
    "analysis/feedback/source/payloads.py",
    "analysis/feedback/source/retrieval.py",
    "analysis/feedback/source/baseline_review.py",
    "analysis/feedback/source/design.py",
    "analysis/common/prompt.py",
    "src/hsrag/prompt.py",
    "prompts/classify_v1.txt",
    "config/taxonomy.yaml",
]


def condition(pool):
    return f"{pool['arm']}_s{pool['seed']}_b{pool['level']:02d}"


def independent_rank(records, vectors, text, query):

    ids = [r["id"] for r in records]
    x = np.array(vectors, dtype=np.float64)
    q = np.array(query, dtype=np.float64)
    x = x / np.sqrt(np.sum(x * x, axis=1))[:, None]
    q = q / math.sqrt(float(np.sum(q * q)))
    dense = np.sum(x * q[None, :], axis=1)
    corpus = [re.findall("\\w+", r["text"].lower()) for r in records]
    terms = {t for row in corpus for t in row}
    n = len(corpus)
    frequencies = [Counter(row) for row in corpus]
    average = sum(map(len, corpus)) / n
    idf = {
        t: math.log(n - sum((t in row for row in corpus)) + 0.5)
        - math.log(sum((t in row for row in corpus)) + 0.5)
        for t in terms
    }
    eps = 0.25 * sum(idf.values()) / len(idf)
    idf = {t: eps if v < 0 else v for t, v in idf.items()}
    lexical = np.zeros(n, dtype=np.float64)
    lengths = np.array(list(map(len, corpus)), dtype=np.float64)
    for term in re.findall("\\w+", text.lower()):
        freq = np.array([f[term] for f in frequencies], dtype=np.float64)
        lexical += idf.get(term, 0.0) * (
            freq * 2.5 / (freq + 1.5 * (0.25 + 0.75 * lengths / average))
        )
    dense_order = sorted(range(n), key=lambda i: (-dense[i], ids[i]))[:10]
    lexical_order = [
        i
        for i in sorted(range(n), key=lambda i: (-lexical[i], ids[i]))[:10]
        if lexical[i] > 0
    ]
    fused = {
        i: (1 / (61 + dense_order.index(i)) if i in dense_order else 0)
        + (1 / (61 + lexical_order.index(i)) if i in lexical_order else 0)
        for i in set(dense_order) | set(lexical_order)
    }
    final = sorted(fused, key=lambda i: (-fused[i], ids[i]))
    return {
        "dense": [{"id": ids[i], "score": float(dense[i])} for i in dense_order],
        "bm25": [{"id": ids[i], "score": float(lexical[i])} for i in lexical_order],
        "rrf": [{"id": ids[i], "score": fused[i]} for i in final],
        "selected_id": ids[final[0]],
    }


def compare_trace(left, right):
    a.require(
        left["selected_id"] == right["selected_id"],
        "Independent delivered record differs",
    )
    for channel in ("dense", "bm25", "rrf"):
        a.require(
            [r["id"] for r in left[channel]] == [r["id"] for r in right[channel]],
            "Independent channel order differs: " + channel,
        )
        for x, y in zip(left[channel], right[channel]):
            a.require(
                math.isclose(x["score"], y["score"], rel_tol=1e-11, abs_tol=1e-12),
                "Independent channel score differs",
            )


def checks():
    rng = np.random.default_rng(20260929)
    records = [
        {"id": f"x{i:02d}", "text": f"common word{i % 7} extra{i}"} for i in range(40)
    ]
    vectors = rng.normal(size=(40, 17))
    queries = [vectors[2], rng.normal(size=17)]
    n = 0
    for size in (10, 20, 30, 40):
        bucket = retrieval.FeedbackBucket(records[:size], vectors[:size])
        for q in queries:
            for text in ("common common word3", "no_lexical_match"):
                _, trace = bucket.retrieve(text, q)
                compare_trace(
                    trace, independent_rank(records[:size], vectors[:size], text, q)
                )
                n += 1
    return {"passed": True, "ranking_cases": n, "source_sha256": a.digest(__file__)}


def prerequisites():
    b.validate()
    receipt = a.read(review.BASE / "complete.json")
    evidence = a.read(review.EVIDENCE)
    a.require(
        a.digest(review.EVIDENCE) == receipt["evidence_sha256"]
        and receipt["status"] == "baseline_review_complete_selection_frozen",
        "Baseline review required",
    )
    for name, h in receipt["outputs"].items():
        a.require(a.digest(ROOT / name) == h, "Baseline selection/export changed")
    a.require(
        a.digest(review.BASE / "selection.json") == receipt["selection_sha256"],
        "Selection binding differs",
    )
    selection = a.read(review.BASE / "selection.json")
    science = a.read(b.SCIENCE)
    a.require(
        selection["k"] == 40
        and selection["levels"] == [10, 20, 30, 40]
        and (len(selection["sets"]) == 24),
        "Reviewed actual selection differs",
    )
    return (selection, science, evidence)


def sources():
    roster = {r["id"]: r for r in b.core.rows(b.preparation.OUT / "roster.jsonl")}
    prompts = {
        r["item_id"]: r
        for r in b.core.rows(b.preparation.OUT / "baseline_prompts.jsonl")
    }
    dev = b.core.rows(b.preparation.CORE / "dev_inputs.jsonl")
    order = {r["key"]: i for i, r in enumerate(dev)}
    provenance = b.core.rows(b.preparation.RET / "query_vector_sources.jsonl")
    vectors = np.load(b.preparation.RET / "query_vectors.npy", mmap_mode="r")
    return (roster, prompts, order, provenance, vectors)


def prepare():
    a.require(not PLAN.exists(), "Preserve prepared payloads")
    tested = checks()
    selection, science, _ = prerequisites()
    roster, prompts, order, provenance, vectors = sources()
    selected = sorted({i for pool in selection["sets"] for i in pool["ids"]})
    a.require(
        set(selected) <= set(science["rosters"]["eligible_candidates"])
        and (not set(selected) & set(science["rosters"]["transfer"])),
        "Selected source leak",
    )
    paths = [
        review.BASE / "complete.json",
        review.BASE / "selection.json",
        review.BASE / "sentinels.json",
        review.EVIDENCE,
        b.SCIENCE,
        b.PLAN,
        b.preparation.OUT / "roster.jsonl",
        b.preparation.OUT / "baseline_prompts.jsonl",
        b.preparation.CORE / "dev_inputs.jsonl",
        b.preparation.RET / "query_vectors.npy",
        b.preparation.RET / "query_vector_sources.jsonl",
    ]
    inputs = {b.rel(p): a.digest(p) for p in paths}
    inputs.update({name: a.digest(ROOT / name) for name in SOURCE_FILES})
    a.save_new(
        PLAN,
        {
            "created_utc": a.utc(),
            "status": "payload_preparation_planned",
            "scientific_plan_sha256": b.STUDY_PLAN_SHA,
            "baseline_selection_sha256": a.digest(review.BASE / "selection.json"),
            "inputs": inputs,
            "environment": b.environment(),
            "tests": tested,
            "settings": 24,
            "texts_per_setting": 466,
            "prompt_entries": 11184,
            "predictions": 33552,
            "ordinary_responses": 100656,
            "record_identity": "sq3fb- plus original candidate ID, identical across pool draws/budgets; selection arm never rendered.",
            "source_annotations": "Only hate=True and original hate_type gold; targets/severity/legal missing.",
            "ranking": science["retrieval"],
            "independent_check": "All channel rankings/scores independently recomputed; complete helper replay; exact base context and rendering retained.",
            "execution": "Not enabled here. User update runner requires this independent payload audit and separate execution readiness.",
        },
    )
    records = []
    for iid in selected:
        source = roster[iid]
        idx = order[source["key"]]
        prov = provenance[idx]
        a.require(
            source["correction_eligible"]
            and source["gold"]["hate"] is True
            and (prov["key"] == source["key"])
            and (
                prov["text_sha256"]
                == hashlib.sha256(source["text"].encode()).hexdigest()
            ),
            "Added-record provenance differs",
        )
        records.append(
            {
                "id": "sq3fb-" + iid,
                "kind": "feedback",
                "lang": "en",
                "source": "implicit_hate",
                "text": source["text"],
                "meta": {
                    "gate": True,
                    "hate_types": source["gold"]["hate_type"],
                    "target_groups": None,
                    "severity": None,
                    "legal": None,
                },
                "provenance": {
                    "source_id": iid,
                    "source_roster": b.rel(b.preparation.OUT / "roster.jsonl"),
                    "source_roster_sha256": a.digest(
                        b.preparation.OUT / "roster.jsonl"
                    ),
                    "text_sha256": prov["text_sha256"],
                    "vector_cache": b.rel(b.preparation.RET / "query_vectors.npy"),
                    "vector_row": idx,
                    "vector_sha256": hashlib.sha256(vectors[idx].tobytes()).hexdigest(),
                    "original_vector_source": prov,
                    "selection_sha256": a.digest(review.BASE / "selection.json"),
                },
            }
        )
    record_path = BASE / "added_records.jsonl"
    review.write_rows(record_path, records)
    rmap = {r["provenance"]["source_id"]: r for r in records}
    pools = [
        {
            **pool,
            "condition": condition(pool),
            "record_ids": ["sq3fb-" + i for i in pool["ids"]],
        }
        for pool in selection["sets"]
    ]
    a.save_new(
        BASE / "pools.json",
        {"selection_sha256": a.digest(review.BASE / "selection.json"), "pools": pools},
    )
    prompt_path = a.storage_path(BASE / "prompts.jsonl")
    prompt_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with prompt_path.open("x", encoding="utf8", newline="\n") as stream:
        for pool in pools:
            rr = [rmap[i] for i in pool["ids"]]
            vv = np.array([vectors[order["en/" + i]] for i in pool["ids"]])
            bucket = retrieval.FeedbackBucket(rr, vv)
            for iid in science["rosters"]["evaluation_phase"]:
                base = prompts[iid]
                query_row = order[base["key"]]
                result = retrieval.updated_prompt(base, bucket, vectors[query_row])
                row = {
                    "key": base["key"],
                    "item_id": iid,
                    "text": base["text"],
                    "lang": "en",
                    "condition": pool["condition"],
                    "arm": pool["arm"],
                    "seed": pool["seed"],
                    "budget_level": pool["level"],
                    "pool_size": pool["size"],
                    "strategy": "hybrid",
                    "subset": (
                        "candidate"
                        if roster[iid]["correction_eligible"]
                        else roster[iid]["role"]
                    ),
                    "context": result["context"],
                    "context_sha256": a.signature(result["context"]),
                    "messages": result["messages"],
                    "messages_sha256": a.signature(result["messages"]),
                    "feedback_trace": result["retrieval"],
                    "base_entry_sha256": a.signature(base),
                    "base_messages_sha256": base["messages_sha256"],
                    "query_vector_row": query_row,
                    "plan_sha256": a.digest(PLAN),
                }
                stream.write(
                    json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
                    + "\n"
                )
                count += 1
            print(f"Prepared {count}/11184 update prompts.", flush=True)
    a.require(count == 11184, "Payload count differs")
    for name, h in inputs.items():
        a.require(a.digest(ROOT / name) == h, "Frozen input changed during preparation")
    a.save_new(
        BASE / "prepared.json",
        {
            "created_utc": a.utc(),
            "status": "prepared_independent_check_pending",
            "plan_sha256": a.digest(PLAN),
            "outputs": {
                b.rel(p): a.digest(p)
                for p in (record_path, BASE / "pools.json", BASE / "prompts.jsonl")
            },
            "entries": count,
            "unique_added_records": len(records),
        },
    )
    print(
        json.dumps(
            {
                "status": "prepared_independent_check_pending",
                "entries": count,
                "unique_added_records": len(records),
            },
            indent=2,
        )
    )


def check():
    a.require(
        not EVIDENCE.exists() and (not (BASE / "complete.json").exists()),
        "Preserve checked payloads",
    )
    plan = a.read(PLAN)
    prepared = a.read(BASE / "prepared.json")
    selection, science, _ = prerequisites()
    a.require(
        prepared["plan_sha256"] == a.digest(PLAN)
        and plan["environment"] == b.environment(),
        "Frozen retrieval plan/environment differs",
    )
    for group in (plan["inputs"], prepared["outputs"]):
        for name, h in group.items():
            a.require(
                a.digest(ROOT / name) == h,
                "Input/output changed before independent audit",
            )
    roster, baseline, order, provenance, vectors = sources()
    records = b.core.rows(BASE / "added_records.jsonl")
    rmap = {r["id"]: r for r in records}
    pools = a.read(BASE / "pools.json")["pools"]
    pmap = {p["condition"]: p for p in pools}
    a.require(
        len(pmap) == len(pools) == 24 and len(rmap) == len(records),
        "Duplicate settings/record IDs",
    )
    for actual, expected in zip(pools, selection["sets"]):
        a.require(
            {k: actual[k] for k in expected} == expected
            and actual["condition"] == condition(expected)
            and (actual["record_ids"] == ["sq3fb-" + i for i in expected["ids"]]),
            "Pool selection changed",
        )
    for row in records:
        iid = row["provenance"]["source_id"]
        source = roster[iid]
        idx = order[source["key"]]
        a.require(
            iid in science["rosters"]["eligible_candidates"]
            and row["id"] == "sq3fb-" + iid
            and (row["text"] == source["text"]),
            "Added record not original eligible source",
        )
        a.require(
            row["kind"] == "feedback"
            and row["lang"] == "en"
            and (row["source"] == source["source"]),
            "Added kind/language/source differs",
        )
        a.require(
            row["meta"]
            == {
                "gate": True,
                "hate_types": source["gold"]["hate_type"],
                "target_groups": None,
                "severity": None,
                "legal": None,
            },
            "Source labels changed or missing heads imputed",
        )
        prov = row["provenance"]
        a.require(
            prov["vector_row"] == idx
            and prov["original_vector_source"] == provenance[idx]
            and (
                prov["text_sha256"]
                == hashlib.sha256(source["text"].encode()).hexdigest()
            )
            and (
                prov["vector_sha256"]
                == hashlib.sha256(vectors[idx].tobytes()).hexdigest()
            )
            and (prov["selection_sha256"] == a.digest(review.BASE / "selection.json"))
            and (
                prov["source_roster_sha256"]
                == a.digest(b.preparation.OUT / "roster.jsonl")
            ),
            "Added vector/selection provenance differs",
        )
    buckets = {}
    inputs_by_pool = {}
    for pool in pools:
        rr = [rmap[i] for i in pool["record_ids"]]
        vv = np.array([vectors[order["en/" + i]] for i in pool["ids"]])
        buckets[pool["condition"]] = retrieval.FeedbackBucket(rr, vv)
        inputs_by_pool[pool["condition"]] = (rr, vv)
    seen = set()
    retrieval_counts = Counter()
    self_hits = Counter()
    hashes = []
    token_chars = Counter()
    with a.storage_path(BASE / "prompts.jsonl").open(encoding="utf8") as stream:
        for number, line in enumerate(stream, 1):
            row = json.loads(line)
            key = (row["key"], row["condition"])
            pool = pmap[row["condition"]]
            iid = row["item_id"]
            base = baseline[iid]
            a.require(
                key not in seen
                and iid in science["rosters"]["evaluation_phase"]
                and (row["key"] == base["key"]),
                "Wrong/duplicate update query",
            )
            seen.add(key)
            a.require(
                row["arm"] == pool["arm"]
                and row["seed"] == pool["seed"]
                and (row["budget_level"] == pool["level"])
                and (row["pool_size"] == pool["size"]),
                "Setting attribution differs",
            )
            a.require(
                row["plan_sha256"] == a.digest(PLAN)
                and row["base_entry_sha256"] == a.signature(base)
                and (row["base_messages_sha256"] == base["messages_sha256"])
                and (row["query_vector_row"] == order[row["key"]]),
                "Base/query provenance differs",
            )
            a.require(
                row["text"] == base["text"]
                and row["lang"] == "en"
                and (row["strategy"] == "hybrid")
                and (
                    row["subset"]
                    == (
                        "candidate"
                        if roster[iid]["correction_eligible"]
                        else roster[iid]["role"]
                    )
                ),
                "Query text/scope differs",
            )
            rr, vv = inputs_by_pool[row["condition"]]
            q = vectors[order[row["key"]]]
            independent = independent_rank(rr, vv, row["text"], q)
            compare_trace(row["feedback_trace"], independent)
            selected = rmap[independent["selected_id"]]
            expected_context = json.loads(json.dumps(base["context"]))
            expected_context["example_groups"].append(
                [
                    "Additional labelled examples",
                    [
                        {
                            "id": selected["id"],
                            "text": selected["text"],
                            "meta": selected["meta"],
                            "via": "rrf",
                        }
                    ],
                ]
            )
            a.require(
                row["context"] == expected_context
                and row["context_sha256"] == a.signature(expected_context),
                "Base context or added slot differs",
            )
            expected_messages = b.renderer.render_messages(
                base["text"], "en", expected_context, "corrected"
            )
            a.require(
                row["messages"] == expected_messages
                and row["messages_sha256"] == a.signature(expected_messages)
                and (row["messages"][0] == base["messages"][0]),
                "Rendered update messages differ",
            )
            replay = retrieval.updated_prompt(base, buckets[row["condition"]], q)
            a.require(
                replay["context"] == row["context"]
                and replay["messages"] == row["messages"]
                and (replay["retrieval"] == row["feedback_trace"]),
                "Complete retrieval replay differs",
            )
            payload = a.body(row["messages"])
            a.require(
                payload["model"] == a.MODEL and payload["temperature"] == 1.0,
                "Wire contract differs",
            )
            retrieval_counts[row["condition"]] += 1
            self_hits[row["condition"]] += selected["provenance"]["source_id"] == iid
            token_chars[row["condition"]] += sum(
                (len(m["text"]) for m in row["messages"])
            )
            hashes.append(
                {
                    "key": row["key"],
                    "condition": row["condition"],
                    "entry_sha256": a.signature(row),
                    "messages_sha256": row["messages_sha256"],
                    "context_sha256": row["context_sha256"],
                }
            )
            if number % 1000 == 0:
                print(
                    f"Independently verified and replayed {number}/11184 prompts.",
                    flush=True,
                )
    expected = {
        ("en/" + i, p["condition"])
        for i in science["rosters"]["evaluation_phase"]
        for p in pools
    }
    a.require(
        seen == expected
        and len(seen) == 11184
        and all((n == 466 for n in retrieval_counts.values())),
        "Incomplete retrieval coverage",
    )
    for pool in pools:
        a.require(
            0 <= self_hits[pool["condition"]] <= pool["size"],
            "Self retrieval outside selected candidate membership",
        )
    inventory = BASE / "prompt_inventory.jsonl"
    review.write_rows(inventory, hashes)
    for name, h in plan["inputs"].items():
        a.require(a.digest(ROOT / name) == h, "Source changed during audit")
    outputs = {**prepared["outputs"], b.rel(inventory): a.digest(inventory)}
    evidence = {
        "created_utc": a.utc(),
        "status": "all_update_payloads_verified",
        "plan_sha256": a.digest(PLAN),
        "inputs": {
            **plan["inputs"],
            b.rel(PLAN): a.digest(PLAN),
            b.rel(BASE / "prepared.json"): a.digest(BASE / "prepared.json"),
        },
        "outputs": outputs,
        "entries": 11184,
        "settings": 24,
        "source_records": len(records),
        "source_annotation_and_vector_checks": len(records),
        "independent_dense_bm25_rrf_checks": 11184,
        "complete_replay_exact_matches": 11184,
        "changed_base_contexts": 0,
        "settings_counts": dict(retrieval_counts),
        "self_retrieval_on_selected_candidates": dict(self_hits),
        "mean_prompt_characters": {c: n / 466 for c, n in token_chars.items()},
        "tests": plan["tests"],
        "actual_API_calls": 0,
        "new_embeddings_or_training": False,
        "holdout_gold_used_for_ranking": False,
        "limitation": "Selected candidates may retrieve their own answer; their repair is diagnostic, never transfer. Real update outcomes remain ungenerated.",
        "next": "Prepare and mock-test the combined resumable update launcher using these immutable payloads.",
    }
    a.save_new(EVIDENCE, evidence)
    a.save_new(
        BASE / "complete.json",
        {
            "status": evidence["status"],
            "plan_sha256": a.digest(PLAN),
            "evidence": b.rel(EVIDENCE),
            "evidence_sha256": a.digest(EVIDENCE),
            "outputs": outputs,
        },
    )
    print(
        json.dumps(
            {
                "status": evidence["status"],
                "entries": 11184,
                "source_records": len(records),
                "replay_differences": 0,
                "evidence_sha256": a.digest(EVIDENCE),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tests", "prepare", "check"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", u.forbid), patch.object(
        socket, "create_connection", u.forbid
    ), patch.object(b.core, "live_client", u.forbid):
        if args.mode == "tests":
            print(json.dumps(checks(), indent=2))
        elif args.mode == "prepare":
            prepare()
        else:
            check()

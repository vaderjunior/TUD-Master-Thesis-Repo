from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
import re
import time
import numpy as np
from analysis.feedback.insertion import common as c
from analysis.common import prompt as renderer


def fused_ids(channels):
    c.require(set(channels) == {"dense", "bm25"}, "Wrong channels")
    scores = defaultdict(float)
    ranks = defaultdict(dict)
    for channel in ["dense", "bm25"]:
        hits = channels[channel]
        c.require(
            len(hits) <= 10 and len({h["id"] for h in hits}) == len(hits),
            "Duplicate/excess channel hits",
        )
        c.require(
            all((math.isfinite(h["score"]) for h in hits)), "Nonfinite channel score"
        )
        if channel == "bm25":
            c.require(
                all(
                    (
                        hits[i]["score"] >= hits[i + 1]["score"]
                        for i in range(len(hits) - 1)
                    )
                ),
                "Unordered lexical channel",
            )
            c.require(all((h["score"] > 0 for h in hits)), "Nonpositive lexical hit")
        for rank, hit in enumerate(hits, 1):
            scores[hit["id"]] += 1.0 / (60 + rank)
            ranks[hit["id"]][channel] = rank
    return (sorted(scores, key=lambda rid: (-scores[rid], rid))[:5], dict(ranks))


def mmr_ids(query, candidates, vectors):

    ids = [h["id"] for h in candidates]
    c.require(
        len(ids) == len(set(ids)) == 30, "MMR requires thirty distinct dense candidates"
    )
    q = np.asarray(query, dtype=float).ravel()
    q = q / (np.linalg.norm(q) or 1.0)
    matrix = np.asarray([vectors[rid] for rid in ids], dtype=float)
    matrix = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-12)
    relevance = matrix @ q
    similarities = matrix @ matrix.T
    selected = [int(np.argmax(relevance))]
    while len(selected) < 10:
        remaining = np.asarray([i for i in range(len(ids)) if i not in selected])
        redundancy = similarities[np.ix_(remaining, selected)].max(axis=1)
        value = 0.7 * relevance[remaining] - (1.0 - 0.7) * redundancy
        selected.append(int(remaining[int(np.argmax(value))]))
    return [ids[i] for i in selected]


def check_prompt(row, gold_row, base, known, inserted, token):
    c.require(
        row["key"] == gold_row["key"]
        and row["item_id"] == gold_row["id"]
        and (row["text"] == gold_row["text"])
        and (row["population"] == gold_row["population"])
        and (row["lang"] == "en"),
        "Query attribution differs",
    )
    ctx = row["context"]
    c.require(
        ctx["definitions"] == base["context"]["definitions"]
        and ctx["guidelines"] == base["context"]["guidelines"],
        "Abstract intervention was not authorized",
    )
    c.require(
        len(ctx["definitions"]) == len(ctx["guidelines"]) == 2, "Wrong abstract budget"
    )
    c.require(
        [name for name, _ in ctx["example_groups"]] == ["Labelled examples"],
        "Wrong example heading or special correction bucket",
    )
    examples = [h for _, values in ctx["example_groups"] for h in values]
    c.require(
        len(examples) == len({h["id"] for h in examples}) == 5,
        "Exactly five distinct examples required",
    )
    ordered, ranks = fused_ids(row["channels"])
    c.require(
        [h["id"] for h in examples] == ordered, "Independent RRF reconstruction differs"
    )
    for hit in examples:
        record = known[hit["id"]]
        c.require(
            record["kind"] == "example"
            and record["lang"] == "en"
            and (hit["text"] == record["text"])
            and (hit["via"] == "rrf"),
            "Wrong source text/kind/language",
        )
        expected = {
            k: record[k] for k in ["kind", "lang", "source", "dimension", "label"]
        }
        expected.update(record["meta"])
        expected["_ranks"] = ranks[hit["id"]]
        c.require(hit["meta"] == expected, "Source annotation or channel ranks changed")
        if hit["id"] in inserted:
            c.require(
                type(hit["meta"]["gate"]) is bool
                and all(
                    (
                        hit["meta"][k] is None
                        for k in ["hate_types", "target_groups", "severity", "legal"]
                    )
                ),
                "Missing fine annotations fabricated",
            )
            c.require(
                renderer.corrected_gold(hit["meta"]) in ["hate=true", "hate=false"],
                "Gate-only rendering changed",
            )
    delivered = [h["id"] for h in examples if h["id"] in inserted]
    c.require(
        row["inserted_ids_delivered"] == delivered
        and row["delivered_gates"] == [known[i]["meta"]["gate"] for i in delivered],
        "Delivery audit differs",
    )
    c.require(renderer.hit_ids(ctx) == row["retrieved"], "Stored ordered IDs differ")
    rebuilt = renderer.render_messages(gold_row["text"], "en", ctx, "corrected")
    c.require(
        rebuilt == row["messages"] and rebuilt[0] == base["messages"][0],
        "Prompt payload differs",
    )
    c.require(
        row["context_sha256"] == c.sig(ctx)
        and row["messages_sha256"] == c.sig(rebuilt),
        "Prompt hashes differ",
    )
    c.require(
        c.old.old.token_count(token, rebuilt) == row["reference_input_tokens"],
        "Token accounting differs",
    )


def main(pilot=False):
    from analysis.feedback.insertion import prepare as prep

    p = prep.validate()
    science = c.read(prep.SCIENCE)
    root = c.BASE / ("pilot" if pilot else "retrieval")
    conditions = (
        ["reference", f"error_positive_s{c.SEEDS[0]}", f"error_balanced_s{c.SEEDS[0]}"]
        if pilot
        else science["conditions"]
    )
    roster = c.rows(c.BASE / "roster.jsonl")
    if pilot:
        roster = [
            r
            for pop in ["benign", "hateful_types"]
            for r in sorted(
                (r for r in roster if r["population"] == pop), key=lambda r: r["key"]
            )[:3]
        ]
    by_key = {r["key"]: r for r in roster}
    bases = {
        r["key"]: r
        for r in c.rows(c.TAXONOMY / "prompts.jsonl")
        if r["condition"] == "rag_full"
    }
    full = c.rows(c.CORE / "kb/full/records.jsonl")
    source_records = c.read(c.SELECTION / "gate_only_records.json")
    pools = {
        r["condition"]: r for r in c.read(c.SELECTION / "actual_pools.json")["pools"]
    }
    reference = {r["key"]: r for r in c.rows(root / "reference/prompts.jsonl")}
    token = c.old.old.tokenizer()
    all_prompts = []
    delivery = {}
    differences = {}
    changed_keys = set()
    outputs = {}
    core_rows = c.rows(c.CORE / "dev_inputs.jsonl")
    qi = {r["key"]: i for i, r in enumerate(core_rows)}
    qv = np.load(
        c.CORE / "retrieval_v1/query_vectors.npy", allow_pickle=False, mmap_mode="r"
    )
    full_v = np.load(
        c.BASE / "index_vectors/full.npy", allow_pickle=False, mmap_mode="r"
    )
    candidate_rows = c.rows(c.CANDIDATES / "roster.jsonl")
    si = {r["id"]: i for i, r in enumerate(candidate_rows)}
    sv = np.load(c.CANDIDATES / "query_vectors.npy", allow_pickle=False, mmap_mode="r")
    sentinel_checks = 0
    cosine_checks = 0
    for condition in conditions:
        records = c.rows(root / condition / "records.jsonl")
        ids = [] if condition == "reference" else pools[condition]["ids"]
        c.require(
            records[: len(full)] == full and len(records) == len(full) + len(ids),
            "Original records changed or insertion count differs",
        )
        expected_added = []
        for sid in ids:
            r = deepcopy(source_records[sid])
            r["kind"] = "example"
            r["dimension"] = None
            r["label"] = None
            expected_added.append(r)
        c.require(
            records[len(full) :] == expected_added,
            "Insertions not exact frozen records/order",
        )
        c.require(
            not set(ids).intersection((r["id"] for r in roster))
            and (not set(ids).intersection((r["id"] for r in full))),
            "Source ID leakage",
        )
        c.require(
            not {r["text"] for r in expected_added}.intersection(
                (r["text"] for r in roster)
            ),
            "Source text leakage",
        )
        known = {r["id"]: r for r in records}
        inserted = {r["id"] for r in expected_added}
        vector_by_id = {r["id"]: full_v[i] for i, r in enumerate(full)}
        inserted_v = np.load(
            root / condition / "inserted_index_vectors.npy",
            allow_pickle=False,
            mmap_mode="r",
        )
        c.require(
            inserted_v.shape == (len(ids), 1024), "Inserted index-vector roster differs"
        )
        vector_by_id.update(
            {r["id"]: inserted_v[i] for i, r in enumerate(expected_added)}
        )
        prompts = c.rows(root / condition / "prompts.jsonl")
        c.require(
            [r["key"] for r in prompts] == list(by_key), "Prompt roster/order differs"
        )
        from rank_bm25 import BM25Okapi

        lexical_records = [
            r for r in records if r["kind"] == "example" and r["lang"] == "en"
        ]
        lex = BM25Okapi(
            [re.findall("\\w+", r["text"].lower()) for r in lexical_records]
        )
        sentinel = {
            r["key"]
            for pop in ["benign", "hateful_types"]
            for r in sorted(
                (r for r in roster if r["population"] == pop),
                key=lambda r: hashlib.sha256(
                    ("A14-bm25|" + r["key"]).encode()
                ).hexdigest(),
            )[:6]
        }
        counts = Counter()
        for row in prompts:
            c.require(
                row["condition"] == condition
                and row["scientific_plan_sha256"] == c.sha(prep.SCIENCE),
                "Scientific condition/hash differs",
            )
            check_prompt(
                row, by_key[row["key"]], bases[row["key"]], known, inserted, token
            )
            query = np.asarray(qv[qi[row["key"]]], dtype=np.float64)
            candidates = row["dense_candidates"]
            c.require(
                all(
                    (
                        candidates[i]["score"] >= candidates[i + 1]["score"] - 1e-07
                        for i in range(len(candidates) - 1)
                    )
                ),
                "Dense candidate order changed",
            )
            c.require(
                [h["id"] for h in row["channels"]["dense"]]
                == mmr_ids(query, candidates, vector_by_id),
                "Independent MMR reconstruction differs",
            )
            for hit in candidates:
                r = known[hit["id"]]
                c.require(
                    r["kind"] == "example" and r["lang"] == "en", "Dense filter changed"
                )
                vector = np.asarray(vector_by_id[hit["id"]], dtype=np.float64)
                cosine = float(
                    query @ vector / (np.linalg.norm(query) * np.linalg.norm(vector))
                )
                c.require(
                    abs(cosine - hit["score"]) < 0.0001,
                    "Dense hit score inconsistent with cached vectors",
                )
                cosine_checks += 1
            if row["key"] in sentinel:
                scores = lex.get_scores(re.findall("\\w+", row["text"].lower()))
                ordering = sorted(
                    range(len(lexical_records)), key=lambda i: -scores[i]
                )[:10]
                expected = [
                    {"id": lexical_records[i]["id"], "score": float(scores[i])}
                    for i in ordering
                    if scores[i] > 0
                ]
                c.require(
                    row["channels"]["bm25"] == expected,
                    "Independent sentinel BM25 ranks/scores differ",
                )
                sentinel_checks += 1
            delivered = len(row["inserted_ids_delivered"])
            counts["texts"] += 1
            counts["with_correction"] += delivered > 0
            counts[f"corrections_{delivered}"] += 1
            counts["hateful_insertions_delivered"] += sum(row["delivered_gates"])
            counts["benign_insertions_delivered"] += delivered - sum(
                row["delivered_gates"]
            )
            counts["same_messages_as_reference"] += (
                row["messages"] == reference[row["key"]]["messages"]
            )
            counts["no_correction_but_changed_messages"] += (
                delivered == 0 and row["messages"] != reference[row["key"]]["messages"]
            )
            all_prompts.append(row)
        delivery[condition] = dict(counts)
        differences[condition] = {}
        for mode in ["freeze", "replay1", "replay2"]:
            receipt_path = root / condition / mode / "complete.json"
            receipt = c.read(receipt_path)
            c.require(
                receipt["plan_sha256"] == c.sha(prep.PLAN)
                and receipt["condition"] == condition
                and (receipt["mode"] == mode)
                and (receipt["pilot"] == pilot)
                and (receipt["rows"] == len(roster)),
                "Retrieval receipt attribution differs",
            )
            c.verify(receipt["outputs"])
            outputs[c.rel(receipt_path)] = c.sha(receipt_path)
            outputs.update(receipt["outputs"])
            if mode == "freeze":
                continue
            replay = c.rows(root / condition / mode / "comparisons.jsonl")
            c.require(
                [r["key"] for r in replay] == list(by_key), "Replay roster differs"
            )
            previous = {r["key"]: r for r in prompts}
            recomputed = Counter()
            for row in replay:
                old = previous[row["key"]]
                new = row.get("actual_entry", old)
                fields = sorted(
                    (k for k in old.keys() | new.keys() if old.get(k) != new.get(k))
                )
                c.require(
                    fields == row["changed_fields"]
                    and row["same_messages"] == (old["messages"] == new["messages"])
                    and (
                        row["same_retrieved_ids"]
                        == (old["retrieved"] == new["retrieved"])
                    ),
                    "Replay difference report inconsistent",
                )
                if fields:
                    check_prompt(
                        new,
                        by_key[row["key"]],
                        bases[row["key"]],
                        known,
                        inserted,
                        token,
                    )
                    c.require(
                        [h["id"] for h in new["channels"]["dense"]]
                        == mmr_ids(
                            qv[qi[row["key"]]], new["dense_candidates"], vector_by_id
                        ),
                        "Replay MMR differs",
                    )
                recomputed["entry"] += bool(fields)
                recomputed["messages"] += not row["same_messages"]
                recomputed["retrieved_ids"] += not row["same_retrieved_ids"]
                if not row["same_messages"] or not row["same_retrieved_ids"]:
                    changed_keys.add(row["key"])
            c.require(
                dict(recomputed) == receipt["differences"], "Replay totals differ"
            )
            differences[condition][mode] = dict(recomputed)
        print(
            f"{condition}: independently checked {len(prompts)} five-slot prompts and both replays.",
            flush=True,
        )
    c.verify(p["original_index_hashes"])
    report = {
        "status": "verified" if not changed_keys else "cold_replay_requires_review",
        "pilot": pilot,
        "plan_sha256": c.sha(prep.PLAN),
        "prompts": len(all_prompts),
        "conditions": len(conditions),
        "texts": len(roster),
        "delivery": delivery,
        "cold_replays": differences,
        "changed_identity_union": sorted(changed_keys),
        "all_prompt_rrf_checks": len(all_prompts),
        "sentinel_bm25_checks": sentinel_checks,
        "dense_cosine_checks": cosine_checks,
        "source_hashes": c.bindings(prep.sources()),
        "inputs": outputs,
        "Qwen_calls": 0,
        "CUDA_used": False,
    }
    report_path = (
        c.BASE / "pilot" if pilot else c.BASE
    ) / "independent_verification.json"
    c.save(report_path, report)
    c.require(
        not changed_keys,
        "Cold-replay message/ID differences require review; preserved first-pass prompts are unchanged. No Qwen execution authorized by this check.",
    )
    if pilot:
        print("CPU pilot verified, no model calls.", flush=True)
        return
    c.require(len(all_prompts) == 12825, "Full prompt matrix incomplete")
    c.jsonlines(c.BASE / "prompts.jsonl", all_prompts)
    prepared_outputs = {
        **outputs,
        **c.bindings(
            [
                c.BASE / n
                for n in [
                    "roster.jsonl",
                    "prompts.jsonl",
                    "scientific_plan_v1.json",
                    "STUDY_PROTOCOL.md",
                    "independent_verification.json",
                ]
            ]
        ),
    }
    c.save(
        c.BASE / "prepared.json",
        {"status": "independently_checked", "outputs": prepared_outputs},
    )
    c.save(
        c.BASE / "complete.json",
        {
            "status": "offline_preparation_independently_verified",
            "prepared_sha256": c.sha(c.BASE / "prepared.json"),
            "verification": c.rel(report_path),
            "verification_sha256": c.sha(report_path),
            "new_model_calls": 0,
        },
    )
    print(
        "All 12,825 prompts and cold replays independently verified. Execution rehearsal can proceed.",
        flush=True,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()
    os.chdir(c.ROOT)
    with c.offline():
        main(args.pilot)

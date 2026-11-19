import os

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
from collections import Counter
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from analysis.feedback.transfer import archive as a

OUT = a.ROOT / "experiments/astra/sq3_transfer_part5_20261023_v1"
P4 = a.ROOT / "experiments/astra/sq3_transfer_part4_20261023_v1"
SCOPE = a.ROOT / "experiments/astra/sq3_followups_20261012_v1/scope"
SEL = a.ROOT / "experiments/astra/sq3_additions_results_20261006_v1/benign_selection"
CAND = a.ROOT / "experiments/astra/sq3_additions_20261004_v1/benign_selection"
CORE = a.ROOT / "experiments/astra/full_kb_core_20260915_v1"
FIELDS = ("hate_types", "target_groups", "severity", "legal")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def signature(x):
    return hashlib.sha256(
        (
            json.dumps(x, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
            + "\n"
        ).encode()
    ).hexdigest()


def numeric(x, y, message):
    if isinstance(y, np.ndarray):
        y = y.tolist()
    if isinstance(y, dict):
        check(set(x) == set(y), message + " fields")
        for key in y:
            numeric(x[key], y[key], message + "/" + str(key))
    elif isinstance(y, (list, tuple)):
        check(len(x) == len(y), message + " length")
        for i, z in enumerate(y):
            numeric(x[i], z, message + "/" + str(i))
    elif isinstance(y, (bool, str)) or y is None:
        check(x == y, message)
    else:
        check(x is not None and abs(float(x) - float(y)) < 2e-12, message)


def cos(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y)))


def labels(m):
    pieces = []
    if m.get("gate") is not None:
        pieces.append("hate=" + ("true" if m["gate"] else "false"))
    for field, label in [
        ("target_groups", "target_group"),
        ("hate_types", "hate_type"),
        ("legal", "legal"),
    ]:
        if m.get(field) is not None:
            pieces.append(label + "=[" + ", ".join(m[field]) + "]")

    if m.get("severity") is not None:
        pieces.insert(
            len(pieces) - (m.get("legal") is not None), "severity=" + m["severity"]
        )
    return ", ".join(pieces)


def fragment(hit):
    return '- "' + " ".join(hit["text"].split()) + '"\n  labels: ' + labels(hit["meta"])


def main():
    assembled = a.read(OUT / "assembled.json")
    for name, digest in {
        **assembled["input_bindings"],
        **assembled["analysis_sources"],
        **assembled["artifacts"],
    }.items():
        check(a.sha(a.ROOT / name) == digest, "Bound file changed " + name)
    conditions = a.read(a.STUDY / "scientific_plan_v1.json")["conditions"]
    pools = [
        p
        for p in a.read(SEL / "actual_pools.json")["pools"]
        if p["condition"] in conditions
    ]
    rec = a.read(SEL / "gate_only_records.json")
    status = a.read(SEL / "candidate_statuses.json")
    source_catalog = a.read(OUT / "source_catalog.json")
    original_sources = list(a.rows(CAND / "roster.jsonl"))
    si = {r["id"]: i for i, r in enumerate(original_sources)}
    members = Counter(s for p in pools for s in p["ids"])
    check(
        set(source_catalog) == set(members) and len(members) == 98,
        "All source identities",
    )
    for s, stored in source_catalog.items():
        row = rec[s]
        candidate = original_sources[si[s]]
        screen = status[s]
        valid = [x for x in screen["decisions"] if x is not None]
        check(
            bool(2 * sum(valid) > len(valid))
            == screen["majority_hate"]
            != candidate["gold"]["hate"],
            "Independent earlier-error qualification",
        )
        check(
            row["meta"]["gate"] == candidate["gold"]["hate"] == screen["hate"]
            and row["text"] == candidate["text"],
            "Source text/gold",
        )
        numeric(
            stored,
            dict(
                record=row,
                source_vector_row=si[s],
                original_candidate_gold=candidate["gold"],
                screening=screen,
                pool_memberships=[p["condition"] for p in pools if s in p["ids"]],
            ),
            "Source catalog",
        )
    query = list(a.rows(SCOPE / "evaluation.jsonl"))
    qby = {r["key"]: r for r in query}
    meta = list(a.rows(P4 / "metadata.jsonl"))
    vec = np.load(SCOPE / "vectors.npy", mmap_mode="r", allow_pickle=False)
    sv = np.load(CAND / "query_vectors.npy", mmap_mode="r", allow_pickle=False)
    kv = np.load(
        CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    kb = {r["id"]: (i, r) for i, r in enumerate(a.rows(CORE / "kb/full/records.jsonl"))}
    ordinary = a.read(OUT / "ordinary_example_catalog.json")
    for sid, row in ordinary.items():
        check(
            row == dict(vector_row=kb[sid][0], record=kb[sid][1]),
            "Ordinary catalog identity",
        )
    prompts = a.rows(a.STUDY / "prompts.jsonl")
    choices = a.rows(a.STUDY / "choices.jsonl")
    contexts = a.rows(OUT / "query_contexts.jsonl")
    ledger = a.rows(OUT / "delivery.jsonl")
    seen = set()
    prompt_shas = {}
    counts = Counter()
    occurrences = Counter()
    cos_error = 0.0
    cases = {}
    oldref = {}
    systems = set()
    for i, m in enumerate(meta):
        ref = next(prompts)
        stored = next(contexts)
        row = qby[m["key"]]
        hits = ref["context"]["example_groups"][0][1]
        fifth = hits[-1]
        q = vec[row["scope_node_index"]]
        check(
            ref["key"] == m["key"]
            and ref["condition"] == "reference"
            and ref["text"] == row["text"],
            "Original reference identity",
        )
        check(
            stored["ordinary_hits"] == hits and stored["text"] == row["text"],
            "All per-query hit ranks/content retained",
        )
        similarities = [cos(q, kv[kb[h["id"]][0]]) for h in hits]
        cos_error = max(
            cos_error,
            max(abs(x - y) for x, y in zip(stored["ordinary_cosines"], similarities)),
        )
        known = {f: fifth["meta"].get(f) is not None for f in FIELDS}
        remaining = {
            f: sum(h["meta"].get(f) is not None for h in hits[:4]) for f in FIELDS
        }
        expected = dict(
            input_index=i,
            key=m["key"],
            population=m["population"],
            source=m["source"],
            gold_hate=m["gold_hate"],
            inference_cluster=m["inference_cluster"],
            functionality=m["functionality"],
            text=row["text"],
            query_vector_row=row["scope_node_index"],
            reference_messages_sha256=ref["messages_sha256"],
            reference_context_sha256=ref["context_sha256"],
            reference_input_tokens=ref["reference_input_tokens"],
            definitions_ids=[h["id"] for h in ref["context"]["definitions"]],
            guidelines_ids=[h["id"] for h in ref["context"]["guidelines"]],
            ordinary_hits=hits,
            ordinary_cosines=similarities,
            reference_hateful_examples=sum(bool(h["meta"]["gate"]) for h in hits),
            removed_id=fifth["id"],
            removed_source=fifth["meta"]["source"],
            removed_gate=fifth["meta"]["gate"],
            removed_known_fine=known,
            reference_known_field_counts={
                f: remaining[f] + int(known[f]) for f in FIELDS
            },
            remaining_known_field_counts=remaining,
        )
        numeric(stored, expected, "Every query-context field")
        occurrences.update(h["id"] for h in hits)
        prompt_shas[ref["key"], ref["condition"]] = signature(ref["messages"])
        check(
            signature(ref["context"]) == ref["context_sha256"]
            and prompt_shas[ref["key"], ref["condition"]] == ref["messages_sha256"],
            "Reference checksum",
        )
        oldref[ref["key"]] = (ref["messages_sha256"], ref["context_sha256"])
        systems.add(signature(ref["messages"][0]))

        rh = "\n".join(fragment(h) for h in hits)
        query_suffix = "\n\n## Text to classify (language: en)\n" + row["text"]
        check(
            ref["messages"][1]["text"].endswith(rh + query_suffix),
            "Reference labels/text render",
        )
        prefix = ref["messages"][1]["text"][: -len(rh + query_suffix)]
        for c in (1, 2, 3):
            target = next(prompts)
            choice = next(choices)
            d = next(ledger)
            sid = target["source_id"]
            record = rec[sid]
            insert = dict(
                id=record["id"], text=record["text"], meta=record["meta"], via="rrf"
            )
            check(
                target["key"] == ref["key"]
                and target["condition"] == conditions[c]
                and choice["key"] == ref["key"]
                and choice["condition"] == conditions[c],
                "Fixed matrix identity",
            )
            check(
                sid in pools[c - 1]["ids"]
                and choice["source_id"] == sid
                and choice["source_gate"] == record["meta"]["gate"],
                "Chosen pool member/label",
            )
            expected_context = dict(
                ref["context"],
                example_groups=[["Labelled examples", [*hits[:4], insert]]],
            )
            check(
                target["context"] == expected_context
                and target["messages"]
                == [
                    ref["messages"][0],
                    dict(
                        role="user",
                        text=prefix
                        + "\n".join(fragment(h) for h in [*hits[:4], insert])
                        + query_suffix,
                    ),
                ],
                "Only fifth rendered example changes",
            )
            check(
                target["context_sha256"] == signature(expected_context)
                and target["messages_sha256"] == signature(target["messages"]),
                "Treatment hash",
            )
            prompt_shas[target["key"], target["condition"]] = target["messages_sha256"]
            trace = choice["retrieval"]
            ranks = {
                ch: {z["id"]: j for j, z in enumerate(trace[ch], 1)}
                for ch in ("dense", "bm25")
            }
            all_ids = set(ranks["dense"]) | set(ranks["bm25"])
            check(
                len(ranks["dense"]) == 10
                and len(ranks["bm25"]) <= 10
                and all(z["score"] > 0 for z in trace["bm25"]),
                "Channel inclusion",
            )
            check(
                all_ids <= {rec[x]["id"] for x in pools[c - 1]["ids"]},
                "Channel pool membership",
            )
            exact = {
                s: sum(
                    (Fraction(1, 60 + ranks[ch][s]) for ch in ranks if s in ranks[ch]),
                    Fraction(0),
                )
                for s in all_ids
            }
            sums = {
                s: sum(
                    1.0 / (60 + ranks[ch][s])
                    for ch in ("dense", "bm25")
                    if s in ranks[ch]
                )
                for s in all_ids
            }
            order = sorted(sums, key=lambda s: (-sums[s], s))
            check(
                [z["id"] for z in trace["rrf"]] == order
                and trace["selected_id"] == record["id"] == order[0],
                "Independent RRF selection",
            )
            check(
                all(
                    abs(z["score"] - float(exact[z["id"]])) < 1e-14
                    for z in trace["rrf"]
                ),
                "Exact reciprocal-rank fractions",
            )
            sc = cos(q, sv[si[sid]])
            diff = sc - similarities[4]
            cos_error = max(cos_error, abs(sc - d["cosine"]))
            check(
                sc < 0.9495 and abs(sc - choice["cosine"]) < 2e-12,
                "Source-query separation/cosine",
            )
            ch = (
                "both"
                if record["id"] in ranks["dense"] and record["id"] in ranks["bm25"]
                else "dense_only" if record["id"] in ranks["dense"] else "bm25_only"
            )
            expected_d = dict(
                input_index=i,
                key=m["key"],
                population=m["population"],
                source=m["source"],
                gold_hate=m["gold_hate"],
                inference_cluster=m["inference_cluster"],
                functionality=m["functionality"],
                condition=conditions[c],
                pool_index=c,
                source_id=sid,
                source_gate=record["meta"]["gate"],
                source_pool_memberships=members[sid],
                cosine=sc,
                cosine_bin=sum(sc >= edge for edge in (0.4, 0.5, 0.6, 0.7)),
                removed_id=fifth["id"],
                removed_gate=fifth["meta"]["gate"],
                removed_cosine=similarities[4],
                cosine_difference=diff,
                similarity_relation=(
                    "higher" if diff > 1e-12 else "lower" if diff < -1e-12 else "tie"
                ),
                removed_source=fifth["meta"]["source"],
                removed_known_fine=known,
                remaining_known_field_counts=remaining,
                reference_hateful_examples=stored["reference_hateful_examples"],
                feedback_hateful_examples=stored["reference_hateful_examples"]
                - int(fifth["meta"]["gate"])
                + int(record["meta"]["gate"]),
                hateful_count_change=int(record["meta"]["gate"])
                - int(fifth["meta"]["gate"]),
                messages_sha256=target["messages_sha256"],
                context_sha256=target["context_sha256"],
                reference_input_tokens=ref["reference_input_tokens"],
                feedback_input_tokens=target["reference_input_tokens"],
                token_difference=target["reference_input_tokens"]
                - ref["reference_input_tokens"],
                retrieval_sha256=signature(trace),
                dense_cosine_regret=trace["dense"][0]["score"] - sc,
                dense_rank=ranks["dense"].get(record["id"]),
                bm25_rank=ranks["bm25"].get(record["id"]),
                rrf_score=sums[record["id"]],
                rrf_margin=sums[order[0]] - sums[order[1]],
                dense_best=trace["dense"][0]["score"],
                dense_winner_same=trace["dense"][0]["id"] == record["id"],
                lexical_winner_same=bool(trace["bm25"])
                and trace["bm25"][0]["id"] == record["id"],
                retrieval_channels=ch,
            )
            numeric(d, expected_d, "Every delivery field")
            counts[conditions[c] + "/" + str(record["meta"]["gate"])] += 1
            case_key = "/".join(
                map(
                    str,
                    (m["population"], fifth["meta"]["gate"], record["meta"]["gate"]),
                )
            )
            rank = hashlib.sha256(
                ("A16-part5-context-example|" + m["key"] + "|" + conditions[c]).encode()
            ).hexdigest()
            if case_key not in cases or rank < cases[case_key]["selection_hash"]:
                cases[case_key] = dict(
                    selection_hash=rank,
                    delivery=d,
                    query_context=stored,
                    reference_messages=ref["messages"],
                    feedback_messages=target["messages"],
                    correction_record=source_catalog[sid],
                    removed_example=fifth,
                    note="Selected using context identities/polarities only; no prediction outcome used.",
                )
        if (i + 1) % 4000 == 0:
            print(f"Independent context verification {i+1}/{len(meta)}", flush=True)
    check(
        all(next(it, None) is None for it in (prompts, choices, contexts, ledger)),
        "No extra records",
    )
    matched = 0
    for old in a.rows(
        a.ROOT
        / "experiments/astra/later_evaluation_20261008_v1/retrieval_v3/full/prompts.jsonl"
    ):
        if old["key"] in oldref:
            check(
                oldref[old["key"]] == (old["messages_sha256"], old["context_sha256"]),
                "Original qualified reference",
            )
            matched += 1
    check(
        matched == 15758 and len(systems) == 1,
        "All original references and common system text",
    )
    numeric(
        a.read(OUT / "context_examples.json"),
        cases,
        "Outcome-independent case selection",
    )
    inv = a.read(OUT / "pool_inventory.json")
    check(
        inv["ordinary_occurrences"] == dict(occurrences)
        and inv["ordinary_examples_used"] == len(occurrences),
        "Ordinary-use inventory",
    )
    check(
        inv["pools"] == pools
        and inv["unique_sources"] == len(members)
        and inv["membership_counts"]
        == {str(n): c for n, c in Counter(members.values()).items()},
        "Pool totals",
    )
    expected_pairs = [
        dict(
            first=p["condition"],
            second=q["condition"],
            intersection=sorted(set(p["ids"]) & set(q["ids"])),
            union_n=len(set(p["ids"]) | set(q["ids"])),
        )
        for j, p in enumerate(pools)
        for q in pools[j + 1 :]
    ]
    check(
        inv["pairwise"] == expected_pairs
        and inv["triple_intersection"]
        == sorted(set.intersection(*(set(p["ids"]) for p in pools))),
        "Pool overlap",
    )
    seen = set()
    per = Counter()
    nulls = validvotes = 0
    for path in sorted((a.OUT / "assembly").glob("*.jsonl")):
        for row in a.rows(path):
            identity = row["key"], row["condition"], row["repeat"]
            check(
                identity not in seen
                and row["messages_sha256"] == prompt_shas[identity[:2]],
                "Every actual prompt/result link",
            )
            seen.add(identity)
            per[row["condition"]] += 1
            nulls += row["result"] is None
            validvotes += row["n_valid"]
    check(len(seen) == 189096 and len(prompt_shas) == 63032, "Execution binding count")
    b = a.read(OUT / "execution_binding.json")
    check(
        b["delivery_totals"] == dict(counts)
        and b["by_condition"] == dict(per)
        and b["missing_aggregates"] == nulls == 276
        and b["valid_votes"] == validvotes,
        "Execution aggregate",
    )
    check(
        b["source_assignments"] == 47274
        and b["source_prediction_slots"] == 141822
        and b["source_ordinary_initial_vote_slots"] == 425466,
        "Exposure multiplication",
    )
    check(cos_error < 2e-12, "Independent cosine precision")
    receipt = dict(
        status="part5_original_contexts_and_delivery_verified",
        checked_utc=a.utc(),
        queries=15758,
        distinct_prompts=63032,
        assignments=47274,
        prediction_bindings=189096,
        correction_sources=98,
        pool_memberships=120,
        ordinary_examples=len(ordinary),
        ordinary_occurrences=sum(occurrences.values()),
        case_contexts=len(cases),
        max_cosine_difference=cos_error,
        all_rendered_replacements_and_original_metadata_checked=True,
        all_source_and_ranking_fields_checked=True,
        assembled_sha256=a.sha(OUT / "assembled.json"),
        verifier_sha256=a.sha(Path(__file__)),
        new_Qwen_calls=0,
        new_fits=0,
        new_embeddings=0,
    )
    with (OUT / "independent_context_check.json").open("xb") as f:
        f.write(a.encoded(receipt, pretty=True))
    print(
        "All original contexts, labels, selected corrections, similarities and output links independently verified.",
        flush=True,
    )


if __name__ == "__main__":
    with a.offline():
        main()

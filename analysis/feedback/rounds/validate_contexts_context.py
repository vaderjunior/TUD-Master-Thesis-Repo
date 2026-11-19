from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import numpy as np
from analysis.feedback.rounds import contexts_inputs as io


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def digest(value):
    return hashlib.sha256(
        (
            json.dumps(
                value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False
            )
            + "\n"
        ).encode()
    ).hexdigest()


def equal(x, y, where):
    if isinstance(y, dict):
        check(set(x) == set(y), where + " keys")
        for k in y:
            equal(x[k], y[k], where + "/" + str(k))
    elif isinstance(y, (list, tuple)):
        check(len(x) == len(y), where + " length")
        for i, value in enumerate(y):
            equal(x[i], value, where + "/" + str(i))
    elif isinstance(y, float):
        check(x is not None and abs(x - y) < 2e-12, where)
    else:
        check(x == y, where)


def cosine(x, y):
    x, y = np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")
    return float(np.sum(x * y) / np.sqrt(np.sum(x * x) * np.sum(y * y)))


def example(hit):
    m = hit["meta"]
    labels = ["hate=" + str(m["gate"]).lower()]
    for field, name in [
        ("target_groups", "target_group"),
        ("hate_types", "hate_type"),
        ("severity", "severity"),
        ("legal", "legal"),
    ]:
        if m.get(field) is not None:
            labels.append(
                name
                + "="
                + (m[field] if field == "severity" else "[" + ", ".join(m[field]) + "]")
            )
    return '- "' + " ".join(hit["text"].split()) + '"\n  labels: ' + ", ".join(labels)


def hash_rank(seed, rnd, role, gold, sid):
    return hashlib.sha256(
        "|".join(map(str, ["A17", seed, rnd, role, int(gold), sid])).encode()
    ).hexdigest()


def majority(values):
    check(
        len(values) == 3 and all(type(v) is bool or v is None for v in values),
        "Source repeat type",
    )
    return None if any(v is None for v in values) else values.count(True) > 1


def main():
    assembled = io.read(io.OUT / "assembled.json")
    contract = io.read(io.OUT / "analysis_contract.json")
    bindings = {
        **assembled["input_bindings"],
        **assembled["artifacts"],
        **contract["source_hashes"],
    }
    for name, sha in bindings.items():
        check(io.sha(io.ROOT / name) == sha, "Changed input: " + name)
    candidates = list(io.rows(io.CAND / "roster.jsonl"))
    by_id = {r["id"]: r for r in candidates}
    si = {r["id"]: i for i, r in enumerate(candidates)}
    key_to_id = {r["key"]: r["id"] for r in candidates}
    evaluation = list(io.rows(io.STUDY / "evaluation.jsonl"))
    ei = {r["key"]: i for i, r in enumerate(evaluation)}
    all_queries = {r["key"]: r for r in candidates + evaluation}
    records = io.read(io.STUDY / "gate_only_records.json")
    blocks = io.read(io.STUDY / "source_blocks.json")
    source_gates = defaultdict(dict)
    reader = io.read(io.READER / "complete.json")
    for stage in reader["stage_receipts"]:
        if stage["role"] != "source":
            continue
        for row in io.rows(io.ROOT / stage["predictions"]):
            sid = key_to_id[row["key"]]
            values = source_gates[row["phase"]].setdefault(sid, {})
            check(row["repeat"] not in values, "Duplicate source repeat")
            values[row["repeat"]] = (
                None if row["result"] is None else row["result"]["hate"]
            )
    gates = {
        p: {sid: [v[k] for k in range(3)] for sid, v in rows.items()}
        for p, rows in source_gates.items()
    }
    for phase, values in gates.items():
        equal(
            values,
            io.read(io.STUDY / "phases" / phase / "execution/source_gates.json"),
            "Source gates",
        )
    initial = gates["source_initial"]
    ledgers = {
        (r["round"], r["stream"], r["source_id"]): r
        for r in io.rows(io.OUT / "candidate_selection_ledger.jsonl")
    }
    selected_rows = {
        (r["round"], r["stream"]): r for r in io.read(io.OUT / "selections.json")
    }
    pools, births, reconstructed_members = {}, {}, []
    for stream in io.STREAMS:
        split = [[] for _ in range(4)]
        for gold in (False, True):
            ordered = sorted(
                [s for s, r in by_id.items() if r["gold"]["hate"] == gold],
                key=lambda s: hash_rank(stream, 0, "source", gold, s),
            )
            for i in range(4):
                split[i].extend(ordered[i * 128 : (i + 1) * 128])
        equal(
            [sorted(x) for x in split],
            blocks["blocks"][str(stream)],
            "Independent source partition",
        )
        previous = {p: [] for p in io.POLICIES}
        for rnd, ids in enumerate(split, 1):
            current = initial if rnd == 1 else gates[f"r{rnd}_source_s{stream}"]
            valid = [
                sid
                for sid in ids
                if majority(initial[sid]) is not None
                and majority(current[sid]) is not None
            ]
            grouped = {
                (p, g): [
                    sid
                    for sid in valid
                    if by_id[sid]["gold"]["hate"] == g
                    and (
                        p == "random"
                        or majority((current if p == "adaptive" else initial)[sid]) != g
                    )
                ]
                for p in io.POLICIES
                for g in (False, True)
            }
            k = min(
                [5]
                + [
                    len(grouped[p, g])
                    for p in ("adaptive", "static")
                    for g in (False, True)
                ]
            )
            additions = {
                p: sorted(
                    sid
                    for g in (False, True)
                    for sid in sorted(
                        grouped[p, g],
                        key=lambda sid: hash_rank(
                            stream,
                            rnd,
                            "random" if p == "random" else "mistake",
                            g,
                            sid,
                        ),
                    )[:k]
                )
                for p in io.POLICIES
            }
            cumulative = {
                p: sorted(set(previous[p]) | set(additions[p])) for p in io.POLICIES
            }
            expected = dict(
                per_class_budget=k,
                eligible_ids=sorted(valid),
                invalid_ids=sorted(set(ids) - set(valid)),
                eligible_counts={
                    p: {str(g): len(grouped[p, g]) for g in (False, True)}
                    for p in io.POLICIES
                },
                additions=additions,
                cumulative=cumulative,
            )
            frozen = io.read(io.STUDY / f"rounds/r{rnd}/selection_s{stream}.json")
            check(
                frozen["evaluation_outcomes_read"] is False
                and frozen["screened_ids"] == sorted(ids),
                "Selection scope",
            )
            for key, value in expected.items():
                equal(frozen[key], value, "Frozen selection/" + key)
            transitions = Counter()
            for sid in ids:
                r, old, new = by_id[sid], majority(initial[sid]), majority(current[sid])
                gold, valid_row = r["gold"]["hate"], sid in valid
                state = (
                    "invalid"
                    if not valid_row
                    else ("error" if old != gold else "correct")
                    + "->"
                    + ("error" if new != gold else "correct")
                )
                transitions[state] += 1
                expected_row = dict(
                    round=rnd,
                    stream=stream,
                    source_id=sid,
                    source_key=r["key"],
                    gold_hate=gold,
                    candidate_cluster=r["cluster"],
                    initial_repeats=initial[sid],
                    current_repeats=current[sid],
                    initial_majority=old,
                    current_majority=new,
                    common_valid=valid_row,
                    error_transition=state,
                    selected={p: sid in additions[p] for p in io.POLICIES},
                    hash_mistake=hash_rank(stream, rnd, "mistake", gold, sid),
                    hash_random=hash_rank(stream, rnd, "random", gold, sid),
                )
                equal(ledgers.pop((rnd, stream, sid)), expected_row, "Candidate ledger")
            equal(
                selected_rows[rnd, stream],
                dict(
                    round=rnd,
                    stream=stream,
                    **expected,
                    source_error_transitions=dict(transitions),
                ),
                "Selected sets",
            )
            for p in io.POLICIES:
                pools[rnd, stream, p] = cumulative[p]
                for sid in additions[p]:
                    births[stream, p, sid] = rnd
                for sid in cumulative[p]:
                    reconstructed_members.append(
                        dict(
                            round=rnd,
                            stream=stream,
                            policy=p,
                            source_id=sid,
                            source_gate=by_id[sid]["gold"]["hate"],
                            added_round=births[stream, p, sid],
                        )
                    )
            previous = cumulative
    check(not ledgers, "Extra candidate entries")
    sort_member = lambda r: (r["round"], r["stream"], r["policy"], r["source_id"])
    equal(
        sorted(io.rows(io.OUT / "pool_memberships.jsonl"), key=sort_member),
        sorted(reconstructed_members, key=sort_member),
        "All cumulative memberships",
    )
    catalog = io.read(io.OUT / "source_catalog.json")
    check(
        set(catalog) == {r["source_id"] for r in reconstructed_members},
        "Catalog coverage",
    )
    for sid, stored in catalog.items():
        equal(
            stored,
            dict(record=records[sid], candidate=by_id[sid], vector_row=si[sid]),
            "Selected source catalog",
        )
        record = records[sid]
        check(
            record["text"] == by_id[sid]["text"]
            and record["meta"]
            == dict(gate=by_id[sid]["gold"]["hate"], **{f: None for f in io.FIELDS}),
            "Binary-only source annotation",
        )
    cluster_counts = Counter(str(r["cluster"]) for r in candidates)
    cross_block = {}
    for stream in io.STREAMS:
        cluster_rounds = defaultdict(set)
        for rnd, ids in enumerate(blocks["blocks"][str(stream)], 1):
            for sid in ids:
                cluster_rounds[str(by_id[sid]["cluster"])].add(rnd)
        cross_block[str(stream)] = {
            c: sorted(v) for c, v in cluster_rounds.items() if len(v) > 1
        }
    equal(
        io.read(io.OUT / "source_cluster_inventory.json"),
        dict(
            frozen_multi_candidate_components=blocks[
                "source_clusters_with_multiple_candidates"
            ],
            candidate_cluster_counts=dict(cluster_counts),
            cross_block_candidate_clusters=cross_block,
        ),
        "Source clusters",
    )
    print(
        "Independent source-selection, budget, chronology and cluster checks passed.",
        flush=True,
    )

    bases = {p["key"]: p for p in io.rows(io.CAND / "prompts.jsonl")}
    bases.update(
        {
            p["key"]: p
            for p in io.rows(io.LATER / "retrieval_v3/full/prompts.jsonl")
            if p["key"] in ei
        }
    )
    contexts = {r["key"]: r for r in io.rows(io.OUT / "query_contexts.jsonl")}
    kb = {
        r["id"]: (i, r)
        for i, r in enumerate(io.rows(io.CORE / "kb/full/records.jsonl"))
    }
    kv = np.load(
        io.CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    sv = np.load(io.CAND / "query_vectors.npy", mmap_mode="r", allow_pickle=False)
    ev = np.load(io.SCOPE / "vectors.npy", mmap_mode="r", allow_pickle=False)
    expected_ordinary = {}
    for key, row in all_queries.items():
        base = bases[key]
        ctx = base["context"]
        hits = ctx["example_groups"][0][1]
        check(
            len(hits) == 5
            and digest(ctx) == base["context_sha256"]
            and digest(base["messages"]) == base["messages_sha256"],
            "Original context hash",
        )
        for hit in [*ctx["definitions"], *ctx["guidelines"], *hits]:
            i, rec = kb[hit["id"]]
            check(
                hit["text"] == rec["text"]
                and {k: v for k, v in hit["meta"].items() if k != "_ranks"}
                == {
                    **rec["meta"],
                    **{
                        f: rec[f]
                        for f in ["dimension", "kind", "label", "lang", "source"]
                    },
                },
                "Ordinary source metadata",
            )
        for h in hits:
            expected_ordinary[h["id"]] = dict(
                record=kb[h["id"]][1], vector_row=kb[h["id"]][0]
            )
        q = sv[si[row["id"]]] if key in key_to_id else ev[row["scope_node_index"]]
        cosines = [cosine(q, kv[kb[h["id"]][0]]) for h in hits]
        fifth = hits[-1]
        expected = dict(
            key=key,
            id=row["id"],
            text=row["text"],
            source=row["source"],
            population=row["population"],
            context=ctx,
            messages=base["messages"],
            messages_sha256=base["messages_sha256"],
            context_sha256=base["context_sha256"],
            ordinary_cosines=cosines,
            removed_id=fifth["id"],
            removed_gate=fifth["meta"]["gate"],
            removed_source=fifth["meta"]["source"],
            removed_known_fine={f: fifth["meta"].get(f) is not None for f in io.FIELDS},
            known_fine_counts={
                f: sum(h["meta"].get(f) is not None for h in hits) for f in io.FIELDS
            },
            remaining_known_fine_counts={
                f: sum(h["meta"].get(f) is not None for h in hits[:4])
                for f in io.FIELDS
            },
            reference_hateful_examples=sum(h["meta"]["gate"] for h in hits),
            fifth_is_lowest_cosine=cosines[-1] <= min(cosines) + 1e-12,
            fifth_cosine_rank=1 + sum(v > cosines[-1] + 1e-12 for v in cosines),
        )
        equal(contexts[key], expected, "Every baseline context field")
    equal(
        io.read(io.OUT / "ordinary_example_catalog.json"),
        expected_ordinary,
        "Ordinary example catalog",
    )
    prompts = {
        (p["phase"], p["key"], p["condition"]): p
        for p in io.rows(io.OUT / "prompt_bindings.jsonl")
    }
    ledger = {
        (d["phase"], d["key"], d["condition"]): d
        for d in io.rows(io.OUT / "deliveries.jsonl")
    }
    check(
        len(prompts) == 86408 and len(ledger) == 77076, "Prompt and delivery identities"
    )
    case_minima, seen, source_counts, phase_counts = {}, set(), Counter(), []
    for phase in io.read(io.REVIEW / "part1/readiness.json")["phases"]:
        name = phase["phase"]
        root = io.STUDY / "phases" / name
        science = io.read(root / "scientific_plan_v1.json")
        source = phase["source"]
        rnd = 0 if name == "source_initial" else int(name[1])
        stream = int(name.split("_s")[-1]) if "_source_s" in name else None
        expected_pools = (
            {"reference": []}
            if rnd == 0
            else (
                {"adaptive": pools[rnd - 1, stream, "adaptive"]}
                if source
                else {
                    "reference": [],
                    **{
                        f"{p}_s{s}": pools[rnd, s, p]
                        for s in io.STREAMS
                        for p in io.POLICIES
                    },
                }
            )
        )
        equal(science["pools"], expected_pools, "Cumulative pools actually executed")
        delivery_counts, count = Counter(), 0
        for line, p in enumerate(io.rows(root / "prompts.jsonl"), 1):
            key, cond = p["key"], p["condition"]
            identity = (name, key, cond)
            check(identity not in seen, "Duplicated original prompt")
            seen.add(identity)
            count += 1
            base, ctx, row = bases[key], contexts[key], all_queries[key]
            stored = prompts[identity]
            equal(
                stored,
                dict(
                    phase=name,
                    key=key,
                    condition=cond,
                    messages_sha256=digest(p["messages"]),
                    context_sha256=digest(p["context"]),
                    source_id=p["source_id"],
                    line=line,
                    local_token_estimate=p["reference_input_tokens"],
                    reference_token_estimate=None,
                ),
                "Prompt binding",
            )
            check(
                p["messages_sha256"] == stored["messages_sha256"]
                and p["context_sha256"] == stored["context_sha256"]
                and p["scientific_plan_sha256"]
                == io.sha(root / "scientific_plan_v1.json"),
                "Original prompt hashes",
            )
            ids = expected_pools[cond]
            if not ids:
                check(
                    p["messages"] == base["messages"]
                    and p["context"] == base["context"]
                    and p["source_id"] is None
                    and p["retrieval_trace"] is None,
                    "Reference fallback",
                )
                continue
            policy, this_stream = (
                (cond, stream)
                if source
                else (cond.split("_s")[0], int(cond.split("_s")[1]))
            )
            sid = p["source_id"]
            record = records[sid]
            check(
                sid in ids and row["id"] not in ids,
                "Source membership and no self-feedback",
            )
            hits = base["context"]["example_groups"][0][1]
            insert = dict(
                id=record["id"], text=record["text"], meta=record["meta"], via="rrf"
            )
            expected_context = dict(
                base["context"],
                example_groups=[["Labelled examples", hits[:4] + [insert]]],
            )
            tail = (
                "\n".join(example(h) for h in hits)
                + "\n\n## Text to classify (language: en)\n"
                + row["text"]
            )
            check(
                base["messages"][1]["text"].endswith(tail),
                "Original independently rendered examples",
            )
            user_text = (
                base["messages"][1]["text"][: -len(tail)]
                + "\n".join(example(h) for h in hits[:4] + [insert])
                + "\n\n## Text to classify (language: en)\n"
                + row["text"]
            )
            equal(p["context"], expected_context, "Exact fifth replacement")
            equal(
                p["messages"],
                [base["messages"][0], dict(role="user", text=user_text)],
                "Independent replacement rendering",
            )
            trace = p["retrieval_trace"]
            ranks = {
                ch: {h["id"]: n for n, h in enumerate(trace[ch], 1)}
                for ch in ("dense", "bm25")
            }
            allowed = {records[x]["id"] for x in ids}
            all_ids = set(ranks["dense"]) | set(ranks["bm25"])
            check(
                len(ranks["dense"]) == 10
                and len(ranks["bm25"]) <= 10
                and all_ids <= allowed
                and all(h["score"] > 0 for h in trace["bm25"]),
                "Retrieval ranks and pool",
            )
            exact = {
                s: sum(
                    (Fraction(1, 60 + ranks[ch][s]) for ch in ranks if s in ranks[ch]),
                    Fraction(0),
                )
                for s in all_ids
            }
            scores = {
                s: sum(
                    1 / (60 + ranks[ch][s])
                    for ch in ("dense", "bm25")
                    if s in ranks[ch]
                )
                for s in all_ids
            }
            order = sorted(scores, key=lambda s: (-scores[s], s))
            check(
                [h["id"] for h in trace["rrf"]] == order
                and trace["selected_id"] == record["id"] == order[0],
                "Independent RRF ordering and winner",
            )
            check(
                all(
                    abs(h["score"] - float(exact[h["id"]])) < 1e-14
                    for h in trace["rrf"]
                ),
                "Reciprocal-rank arithmetic",
            )
            q = sv[si[row["id"]]] if source else ev[row["scope_node_index"]]
            sim = cosine(q, sv[si[sid]])
            check(source or sim < 0.9495, "Feedback-evaluation overlap cutoff")
            birth = births[this_stream, policy, sid]
            available = rnd - int(source)
            check(birth <= available, "No future correction used")
            d = ledger[identity]
            rid = record["id"]
            reference_tokens = prompts[
                ("source_initial" if source else name, key, "reference")
            ]["local_token_estimate"]
            derived = dict(
                phase=name,
                role="source" if source else "evaluation",
                round=rnd,
                available_round=available,
                stream=this_stream,
                policy=policy,
                condition=cond,
                key=key,
                input_index=ei.get(key),
                population=row["population"],
                gold_hate=row["gold"]["hate"] if source else row["gate"],
                query_source=row["source"],
                inference_cluster=None if source else row["inference_cluster"],
                source_id=sid,
                source_gate=record["meta"]["gate"],
                pool_size=len(ids),
                added_round=birth,
                newest_available_round=birth == available,
                removed_id=hits[-1]["id"],
                removed_gate=hits[-1]["meta"]["gate"],
                removed_source=hits[-1]["meta"]["source"],
                removed_known_fine=ctx["removed_known_fine"],
                remaining_known_fine_counts=ctx["remaining_known_fine_counts"],
                reference_hateful_examples=ctx["reference_hateful_examples"],
                feedback_hateful_examples=sum(h["meta"]["gate"] for h in hits[:4])
                + int(record["meta"]["gate"]),
                cosine=sim,
                removed_cosine=ctx["ordinary_cosines"][4],
                cosine_difference=sim - ctx["ordinary_cosines"][4],
                fifth_is_lowest_cosine=ctx["fifth_is_lowest_cosine"],
                local_token_estimate=p["reference_input_tokens"],
                reference_token_estimate=reference_tokens,
                local_token_difference=p["reference_input_tokens"] - reference_tokens,
                messages_sha256=p["messages_sha256"],
                context_sha256=p["context_sha256"],
                retrieval_sha256=digest(trace),
                prompt_line=line,
                dense_rank=ranks["dense"].get(rid),
                bm25_rank=ranks["bm25"].get(rid),
                rrf_score=scores[rid],
                rrf_margin=(
                    scores[order[0]] - scores[order[1]] if len(order) > 1 else None
                ),
                dense_best=trace["dense"][0]["score"],
                dense_winner_same=trace["dense"][0]["id"] == rid,
                lexical_winner_same=bool(trace["bm25"])
                and trace["bm25"][0]["id"] == rid,
                retrieval_channels=(
                    "both"
                    if rid in ranks["dense"] and rid in ranks["bm25"]
                    else "dense_only" if rid in ranks["dense"] else "bm25_only"
                ),
            )
            equal(d, derived, "Every delivered-context field")
            delivery_counts[cond + "/" + str(record["meta"]["gate"])] += 1
            if not source:
                cell = "/".join(
                    map(
                        str,
                        [
                            row["population"],
                            hits[-1]["meta"]["gate"],
                            record["meta"]["gate"],
                        ],
                    )
                )
                rank = hashlib.sha256(
                    f"A17-part3-case-v1|{name}|{key}|{cond}".encode()
                ).hexdigest()
                if (
                    cell not in case_minima
                    or rank < case_minima[cell]["selection_hash"]
                ):
                    case_minima[cell] = dict(
                        selection_hash=rank,
                        delivery=d,
                        query=row,
                        reference_messages=base["messages"],
                        feedback_messages=p["messages"],
                        reference_context=base["context"],
                        feedback_context=p["context"],
                        correction_record=record,
                        replaced_example=hits[-1],
                        retrieval_trace=trace,
                        selection_rule="Minimum fixed hash within population and polarity swap; outcomes not used.",
                    )
        proof = io.read(root / "independent_verification.json")
        check(
            proof["prompts"] == count and proof["delivery"] == dict(delivery_counts),
            "Existing generation-time independent proof",
        )
        phase_counts.append(
            dict(
                phase=name,
                prompts=count,
                correction_assignments=sum(delivery_counts.values()),
            )
        )
        print(
            f"Independent context readback passed: {name}, {count:,} prompts",
            flush=True,
        )
    check(seen == set(prompts), "Complete original prompt coverage")
    equal(
        io.read(io.OUT / "context_cases.json"),
        case_minima,
        "All eight outcome-blind case cards",
    )
    executed = Counter()
    nulls = 0
    for stage in reader["stage_receipts"]:
        for row in io.rows(io.ROOT / stage["predictions"]):
            identity = (row["phase"], row["key"], row["condition"])
            check(
                row["messages_sha256"] == prompts[identity]["messages_sha256"],
                "Actually executed prompt differs",
            )
            executed[(*identity, row["repeat"])] += 1
            nulls += row["result"] is None
    check(
        len(executed) == 259224 and set(executed.values()) == {1} and nulls == 242,
        "All executed repeats and nulls",
    )
    check(
        {x[:3] for x in executed} == set(prompts)
        and {x[3] for x in executed} == {0, 1, 2},
        "Executed prompt coverage",
    )
    io.write(
        io.OUT / "context_independent_verification.json",
        dict(
            status="passed",
            created_utc=io.pure.utc(),
            verifier_source_sha256=io.sha(Path(__file__)),
            bound_files=len(bindings),
            candidate_opportunities=3072,
            selections=12,
            cumulative_memberships=900,
            prompt_assignments=86408,
            correction_assignments=77076,
            executed_predictions=259224,
            preserved_nulls=nulls,
            outcome_blind_cases=len(case_minima),
            full_cached_retrieval_replay="Assembly independently replayed all 77076 traces; this readback checks all source identities, RRF fractions, winning ranks and selected-vector cosines against original prompts.",
            phases=phase_counts,
            inputs={
                io.rel(p): io.sha(p)
                for p in [io.OUT / "assembled.json", io.OUT / "analysis_contract.json"]
            },
            new_Qwen_calls=0,
        ),
    )
    print(
        "All independent source, context, case and execution checks passed.", flush=True
    )


if __name__ == "__main__":
    with io.pure.offline():
        main()

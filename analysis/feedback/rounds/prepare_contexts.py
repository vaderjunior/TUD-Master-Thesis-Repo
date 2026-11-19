from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import io
from pathlib import Path
import re
import time
import unittest
import numpy as np
from analysis.feedback.rounds import contexts_inputs as a


def build():
    start = time.monotonic()
    a.need(not a.OUT.exists(), "Preserve an existing Part 3 build")
    parent = a.read(a.PART2 / "complete.json")
    a.need(parent["status"] == "iterative_part2_complete_reviewed", "Part 2 required")
    readiness = a.read(a.REVIEW / "part1/readiness.json")
    plan = a.read(a.STUDY / "plan.json")
    reader = a.read(a.READER / "complete.json")
    authoritative = {
        **readiness["source_bindings"],
        **plan["inputs"],
        **plan["outputs"],
        **plan["source_hashes"],
        **reader["input_bindings"],
        **reader["artifacts"],
        **parent["components"],
    }
    kb_plan = a.read(a.CORE / "retrieval_v1/plan.json")
    authoritative.update(kb_plan["generated_hashes"])
    binding = {}

    def bind(path):
        name, digest = a.rel(path), a.sha(path)
        if name in authoritative:
            a.need(
                digest == authoritative[name], "Changed authoritative input: " + name
            )
        binding[name] = digest
        return digest

    input_paths = [
        a.PART2 / "complete.json",
        a.READER / "complete.json",
        a.READER / "independent_verification.json",
        a.PART2 / "errors_v1/results.json",
        a.PART2 / "errors_v1/independent_verification.json",
        a.STUDY / "plan.json",
        a.STUDY / "source_blocks.json",
        a.STUDY / "gate_only_records.json",
        a.STUDY / "evaluation.jsonl",
        a.CAND / "roster.jsonl",
        a.CAND / "prompts.jsonl",
        a.CAND / "query_vectors.npy",
        a.SCOPE / "vectors.npy",
        a.SCOPE / "components.json",
        a.CORE / "kb/full/records.jsonl",
        a.CORE / "retrieval_v1/plan.json",
        a.CORE / "retrieval_v1/full_vectors.npy",
        a.CORE / "retrieval_v1/record_vector_sources.jsonl",
        a.LATER / "retrieval_v3/full/prompts.jsonl",
        a.ROOT / "prompts/classify_v1.txt",
        a.READER / "matrix.jsonl",
    ]
    for phase in readiness["phases"]:
        root = a.STUDY / "phases" / phase["phase"]
        input_paths += [
            root / f
            for f in [
                "scientific_plan_v1.json",
                "roster.jsonl",
                "prompts.jsonl",
                "prepared.json",
                "independent_verification.json",
                "complete.json",
                "execution/schedule.json",
            ]
        ]
        if phase["source"]:
            input_paths.append(root / "execution/source_gates.json")
    for rnd in range(1, 5):
        input_paths += [
            a.STUDY / f"rounds/r{rnd}/selection_s{s}.json" for s in a.STREAMS
        ]
    for path in input_paths:
        bind(path)
    from tests.feedback.rounds import test_contexts as tests

    test_log = io.StringIO()
    tested = unittest.TextTestRunner(stream=test_log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests)
    )
    a.need(tested.wasSuccessful(), test_log.getvalue())
    contract = {
        "status": "context_analysis_conventions_before_context_outcome_summaries",
        "created_utc": a.pure.utc(),
        "timing": "After Part 2 main outcomes are known; before this part computes context-conditional outcomes. No new prospective claim.",
        "selection": "Reconstruct all 12 shared candidate masks, hash-ranked additions and cumulative pools from audited source-only predictions. Keep every candidate, including missing and unselected ones.",
        "delivery": "One assignment per phase/query/condition, not per decoder or ordinary vote. Count all saved prompts regardless of null predictions. Separate evaluation and later source-screening delivery.",
        "replay": "Check every prompt rendering, background and first-four retention, exact fifth replacement, cached-vector dense/BM25/RRF replay and reciprocal-rank arithmetic. Reuse completed raw audits and independent generation-time retrieval receipts.",
        "pool_dynamics": "Report selected-set overlap, exact delivered identity overlap, zero-use records, concentration, age/new-record uptake, stable contexts and switches to new or old records. Pools are balanced; delivered labels need not be.",
        "displacement": "Record original fifth identity, source, gate, known fine annotations, last-known-field loss, cached cosine rank and saved local token estimate. Fifth means prompt position, not lowest dense similarity.",
        "outcome_associations": "Use the Part 2 fixed all-round valid cohorts. For each round/population/endpoint/removed-to-inserted polarity stratum, weight query-policy-stream assignments equally, average three decoder repeats, and pair with the contemporaneous reference. Reference reuse is kept within input/cluster, never independent replication.",
        "supporting_intervals": "All 64 round x population x endpoint x polarity-swap cells receive nominal 95% cluster bootstrap intervals when defined: 10000 PCG64 draws, seed first 8 SHA256 bytes little endian of A17-part3-swap-v1| plus record name. Resample inherited clusters carrying all included text assignments. Empty strata stay undefined.",
        "other_outcomes": "Per-policy and delivery-change strata are descriptive only; no new primary family or winning-source/round selection.",
        "case_selection": "Eight population x removed/inserted-polarity cards, minimum SHA256(A17-part3-case-v1|phase|key|condition), using no prediction outcomes. Preserve exact original and substituted context, source record and provenance.",
        "causal_limit": "Source identity, labels, displaced text, fine-annotation coverage and length change together. Context strata are associations, not randomized mechanism effects. Detailed per-input durability remains Part 4.",
        "input_bindings": binding,
        "source_hashes": {
            a.rel(Path(p)): a.sha(Path(p))
            for p in [__file__, a.__file__, a.context.__file__, tests.__file__]
        },
        "new_Qwen_calls": 0,
        "new_fits": 0,
        "new_embeddings": 0,
    }
    a.write(a.OUT / "analysis_contract.json", contract)
    a.write(
        a.OUT / "tests.json",
        {"passed": True, "tests_run": tested.testsRun, "output": test_log.getvalue()},
    )
    candidates = list(a.rows(a.CAND / "roster.jsonl"))
    by_id = {r["id"]: r for r in candidates}
    source_index = {r["id"]: i for i, r in enumerate(candidates)}
    id_by_key = {r["key"]: r["id"] for r in candidates}
    evaluation = list(a.rows(a.STUDY / "evaluation.jsonl"))
    evaluation_index = {r["key"]: i for i, r in enumerate(evaluation)}
    records = a.read(a.STUDY / "gate_only_records.json")
    blocks = a.read(a.STUDY / "source_blocks.json")
    a.need(len(candidates) == len(records) == 1024, "Candidate scope")
    for seed in a.STREAMS:
        expected = [[] for _ in range(4)]
        for gold in [False, True]:
            ordered = sorted(
                [r["id"] for r in candidates if r["gold"]["hate"] == gold],
                key=lambda sid: a.rank(seed, 0, "source", gold, sid),
            )
            a.need(len(ordered) == 512, "Wrong source class balance")
            for rnd in range(4):
                expected[rnd] += ordered[rnd * 128 : (rnd + 1) * 128]
        a.need(
            [sorted(v) for v in expected] == blocks["blocks"][str(seed)],
            "Source partition hashes differ",
        )
    for sid, record in records.items():
        original = by_id[sid]
        a.need(
            record["text"] == original["text"]
            and record["meta"]["gate"] == original["gold"]["hate"]
            and record["provenance"]["source_id"] == sid
            and all(record["meta"][f] is None for f in a.FIELDS),
            "Feedback provenance or unknown fine labels changed",
        )
    gates = defaultdict(dict)
    for stage in reader["stage_receipts"]:
        if stage["role"] != "source":
            continue
        receipt = a.read(a.ROOT / stage["receipt"])
        a.need(
            a.sha(a.ROOT / stage["receipt"]) == stage["receipt_sha256"],
            "Reader receipt changed",
        )
        path = a.ROOT / stage["predictions"]
        a.need(bind(path) == receipt["predictions_sha256"], "Source assembly changed")
        for saved in a.rows(path):
            sid = id_by_key[saved["key"]]
            values = gates[stage["phase"]].setdefault(sid, [None] * 3)
            values[saved["repeat"]] = (
                None if saved["result"] is None else saved["result"]["hate"]
            )
    for phase, values in gates.items():
        a.need(
            values
            == a.read(a.STUDY / "phases" / phase / "execution/source_gates.json"),
            "Saved source gate extraction differs",
        )
    initial = gates["source_initial"]
    selection_rows, selections, memberships, birth = [], [], [], {}
    prior = {s: {p: [] for p in a.POLICIES} for s in a.STREAMS}
    for rnd in range(1, 5):
        for seed in a.STREAMS:
            candidate_ids = blocks["blocks"][str(seed)][rnd - 1]
            current = initial if rnd == 1 else gates[f"r{rnd}_source_s{seed}"]
            subset = [by_id[s] for s in candidate_ids]
            reconstructed = a.selection(
                subset, initial, current, prior[seed], seed, rnd
            )
            saved = a.read(a.STUDY / f"rounds/r{rnd}/selection_s{seed}.json")
            a.need(
                all(saved[k] == v for k, v in reconstructed.items())
                and saved["evaluation_outcomes_read"] is False,
                "Frozen source selection differs",
            )
            a.need(
                saved["screened_ids"] == sorted(candidate_ids)
                and saved["annotation_opportunities"] == 256,
                "Source block scope differs",
            )
            transitions = Counter()
            for row in subset:
                sid, gold = row["id"], row["gold"]["hate"]
                old, new = a.majority(initial[sid]), a.majority(current[sid])
                valid = old is not None and new is not None
                state = (
                    "invalid"
                    if not valid
                    else ("error" if old != gold else "correct")
                    + "->"
                    + ("error" if new != gold else "correct")
                )
                transitions[state] += 1
                selection_rows.append(
                    {
                        "round": rnd,
                        "stream": seed,
                        "source_id": sid,
                        "source_key": row["key"],
                        "gold_hate": gold,
                        "candidate_cluster": row["cluster"],
                        "initial_repeats": initial[sid],
                        "current_repeats": current[sid],
                        "initial_majority": old,
                        "current_majority": new,
                        "common_valid": valid,
                        "error_transition": state,
                        "selected": {
                            p: sid in reconstructed["additions"][p] for p in a.POLICIES
                        },
                        "hash_mistake": a.rank(seed, rnd, "mistake", gold, sid),
                        "hash_random": a.rank(seed, rnd, "random", gold, sid),
                    }
                )
            for policy in a.POLICIES:
                for sid in reconstructed["additions"][policy]:
                    birth[seed, policy, sid] = rnd
                for sid in reconstructed["cumulative"][policy]:
                    memberships.append(
                        {
                            "round": rnd,
                            "stream": seed,
                            "policy": policy,
                            "source_id": sid,
                            "source_gate": records[sid]["meta"]["gate"],
                            "added_round": birth[seed, policy, sid],
                        }
                    )
            selected = dict(
                round=rnd,
                stream=seed,
                **reconstructed,
                source_error_transitions=dict(transitions),
            )
            selections.append(selected)
            prior[seed] = reconstructed["cumulative"]
            print(f"Source choices verified: round {rnd}, stream {seed}", flush=True)
    a.lines(a.OUT / "candidate_selection_ledger.jsonl", selection_rows)
    a.write(a.OUT / "selections.json", selections)
    a.lines(a.OUT / "pool_memberships.jsonl", memberships)
    a.write(
        a.OUT / "source_catalog.json",
        {
            sid: {
                "record": records[sid],
                "candidate": by_id[sid],
                "vector_row": source_index[sid],
            }
            for sid in sorted({v["source_id"] for v in memberships})
        },
    )
    a.write(
        a.OUT / "source_cluster_inventory.json",
        {
            "frozen_multi_candidate_components": blocks[
                "source_clusters_with_multiple_candidates"
            ],
            "candidate_cluster_counts": dict(
                Counter(str(r["cluster"]) for r in candidates)
            ),
            "cross_block_candidate_clusters": {
                str(seed): {
                    cluster: sorted(
                        {
                            rnd + 1
                            for rnd, ids in enumerate(blocks["blocks"][str(seed)])
                            if any(str(by_id[sid]["cluster"]) == cluster for sid in ids)
                        }
                    )
                    for cluster in sorted({str(r["cluster"]) for r in candidates})
                    if len(
                        {
                            rnd
                            for rnd, ids in enumerate(blocks["blocks"][str(seed)])
                            if any(str(by_id[sid]["cluster"]) == cluster for sid in ids)
                        }
                    )
                    > 1
                }
                for seed in a.STREAMS
            },
        },
    )
    bases = {p["key"]: p for p in a.rows(a.CAND / "prompts.jsonl")}
    bases.update(
        {
            p["key"]: p
            for p in a.rows(a.LATER / "retrieval_v3/full/prompts.jsonl")
            if p["key"] in evaluation_index
        }
    )
    sv = np.load(a.CAND / "query_vectors.npy", mmap_mode="r", allow_pickle=False)
    ev = np.load(a.SCOPE / "vectors.npy", mmap_mode="r", allow_pickle=False)
    kb = {
        r["id"]: (i, r) for i, r in enumerate(a.rows(a.CORE / "kb/full/records.jsonl"))
    }
    kv = np.load(
        a.CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    contexts, ordinary = {}, {}
    for source in candidates + evaluation:
        q = (
            sv[source_index[source["id"]]]
            if source["key"] in id_by_key
            else ev[source["scope_node_index"]]
        )
        contexts[source["key"]] = a.input_context(
            bases[source["key"]], source, q, kb, kv
        )
        for hit in bases[source["key"]]["context"]["example_groups"][0][1]:
            ordinary[hit["id"]] = {
                "record": kb[hit["id"]][1],
                "vector_row": kb[hit["id"]][0],
            }
    a.lines(a.OUT / "query_contexts.jsonl", contexts.values())
    a.write(a.OUT / "ordinary_example_catalog.json", ordinary)
    template = (a.ROOT / "prompts/classify_v1.txt").read_text(encoding="utf8")
    sections = re.split(r"^=== (\w+) ===$", template, flags=re.MULTILINE)
    caution = {
        sections[i]: sections[i + 1].strip() for i in range(1, len(sections), 2)
    }["EXAMPLES_CAUTION"]
    prompts, deliveries, cases, phase_checks = {}, [], {}, []
    for phase in readiness["phases"]:
        name = phase["phase"]
        root = a.STUDY / "phases" / name
        science = a.read(root / "scientific_plan_v1.json")
        science_sha = a.sha(root / "scientific_plan_v1.json")
        source_phase = phase["source"]
        rnd = 0 if name == "source_initial" else int(name[1])
        phase_stream = int(name.split("_s")[-1]) if "_source_s" in name else None
        if name == "source_initial":
            expected_pools = {"reference": []}
        elif source_phase:
            expected_pools = {
                "adaptive": next(
                    s["cumulative"]["adaptive"]
                    for s in selections
                    if s["round"] == rnd - 1 and s["stream"] == phase_stream
                )
            }
        else:
            expected_pools = {
                "reference": [],
                **{
                    f'{p}_s{s["stream"]}': s["cumulative"][p]
                    for s in selections
                    if s["round"] == rnd
                    for p in a.POLICIES
                },
            }
        a.need(
            science["pools"] == expected_pools,
            "Executed phase pool differs from the selected cumulative pool",
        )
        query_rows = {r["key"]: r for r in a.rows(root / "roster.jsonl")}
        buckets = {
            c: a.ReplayBucket(
                [records[s] for s in ids], sv[[source_index[s] for s in ids]]
            )
            for c, ids in science["pools"].items()
            if ids
        }
        count, delivery_counts, systems = 0, Counter(), set()
        for line, prompt in enumerate(a.rows(root / "prompts.jsonl"), 1):
            key, condition = prompt["key"], prompt["condition"]
            row, base, ctx = query_rows[key], bases[key], contexts[key]
            ids = science["pools"][condition]
            policy, stream = (
                (condition, phase_stream)
                if source_phase
                else (
                    ("reference", None)
                    if condition == "reference"
                    else (condition.split("_s")[0], int(condition.split("_s")[1]))
                )
            )
            a.need(
                prompt["text"] == row["text"]
                and prompt["population"] == row["population"],
                "Prompt query/role differs",
            )
            a.need(
                prompt["scientific_plan_sha256"] == science_sha,
                "Prompt science binding differs",
            )
            a.need(
                prompt["context_sha256"] == a.context.sig(prompt["context"])
                and prompt["messages_sha256"] == a.context.sig(prompt["messages"]),
                "Prompt hash differs",
            )
            a.need(
                prompt["messages"][0] == base["messages"][0]
                and prompt["messages"][1]["text"]
                == a.context.render_user(row["text"], prompt["context"], caution),
                "Prompt independent rendering differs",
            )
            systems.add(a.context.sig(prompt["messages"][0]))
            identity = (name, key, condition)
            a.need(identity not in prompts, "Duplicate prompt assignment")
            prompts[identity] = {
                "phase": name,
                "key": key,
                "condition": condition,
                "messages_sha256": prompt["messages_sha256"],
                "context_sha256": prompt["context_sha256"],
                "source_id": prompt["source_id"],
                "line": line,
                "local_token_estimate": prompt["reference_input_tokens"],
                "reference_token_estimate": None,
            }
            count += 1
            if not ids:
                a.need(
                    prompt["context"] == base["context"]
                    and prompt["messages"] == base["messages"]
                    and prompt["source_id"] is None
                    and prompt["retrieval_trace"] is None,
                    "Empty-pool fallback differs",
                )
                continue
            sid = prompt["source_id"]
            a.need(
                sid in ids and row["id"] not in ids,
                "Wrong selected pool or self-correction",
            )
            record = records[sid]
            correction = {
                "id": record["id"],
                "text": record["text"],
                "meta": record["meta"],
                "via": "rrf",
            }
            fifth = a.context.replacement(
                base["context"], prompt["context"], correction
            )
            q = (
                sv[source_index[row["id"]]]
                if source_phase
                else ev[row["scope_node_index"]]
            )
            replay = buckets[condition].retrieve(row["text"], q)
            a.check_trace(prompt["retrieval_trace"], replay)
            trace = a.context.fuse(
                prompt["retrieval_trace"], {records[s]["id"] for s in ids}
            )
            cos = a.context.cosine(q, sv[source_index[sid]])
            a.need(
                source_phase or cos < 0.9495, "Evaluation/source overlap cutoff differs"
            )
            added = birth[stream, policy, sid]
            available_round = rnd - 1 if source_phase else rnd
            a.need(added <= available_round, "Future correction shown too early")
            selected = {
                "phase": name,
                "role": "source" if source_phase else "evaluation",
                "round": rnd,
                "available_round": available_round,
                "stream": stream,
                "policy": policy,
                "condition": condition,
                "key": key,
                "input_index": evaluation_index.get(key),
                "population": row["population"],
                "gold_hate": row["gold"]["hate"] if source_phase else row["gate"],
                "query_source": row["source"],
                "inference_cluster": None if source_phase else row["inference_cluster"],
                "source_id": sid,
                "source_gate": record["meta"]["gate"],
                "pool_size": len(ids),
                "added_round": added,
                "newest_available_round": added == available_round,
                "removed_id": fifth["id"],
                "removed_gate": fifth["meta"]["gate"],
                "removed_source": fifth["meta"]["source"],
                "removed_known_fine": ctx["removed_known_fine"],
                "remaining_known_fine_counts": ctx["remaining_known_fine_counts"],
                "reference_hateful_examples": ctx["reference_hateful_examples"],
                "feedback_hateful_examples": ctx["reference_hateful_examples"]
                - int(fifth["meta"]["gate"])
                + int(record["meta"]["gate"]),
                "cosine": cos,
                "removed_cosine": ctx["ordinary_cosines"][4],
                "cosine_difference": cos - ctx["ordinary_cosines"][4],
                "fifth_is_lowest_cosine": ctx["fifth_is_lowest_cosine"],
                "local_token_estimate": prompt["reference_input_tokens"],
                "messages_sha256": prompt["messages_sha256"],
                "context_sha256": prompt["context_sha256"],
                "retrieval_sha256": a.context.sig(prompt["retrieval_trace"]),
                "prompt_line": line,
                **trace,
            }
            deliveries.append(selected)
            delivery_counts[condition + "/" + str(record["meta"]["gate"])] += 1
            if not source_phase:
                cell = (
                    row["population"],
                    fifth["meta"]["gate"],
                    record["meta"]["gate"],
                )
                rank = hashlib.sha256(
                    f"A17-part3-case-v1|{name}|{key}|{condition}".encode()
                ).hexdigest()
                if cell not in cases or rank < cases[cell]["selection_hash"]:
                    cases[cell] = {
                        "selection_hash": rank,
                        "delivery": selected,
                        "query": row,
                        "reference_messages": base["messages"],
                        "feedback_messages": prompt["messages"],
                        "reference_context": base["context"],
                        "feedback_context": prompt["context"],
                        "correction_record": record,
                        "replaced_example": fifth,
                        "retrieval_trace": prompt["retrieval_trace"],
                        "selection_rule": "Minimum fixed hash within population and polarity swap; outcomes not used.",
                    }
        proof = a.read(root / "independent_verification.json")
        a.need(
            count == len(query_rows) * len(science["conditions"]) == proof["prompts"]
            and dict(delivery_counts) == proof["delivery"]
            and len(systems) == 1,
            "Generation-time context proof differs",
        )
        phase_checks.append(
            {
                "phase": name,
                "prompts": count,
                "correction_assignments": sum(delivery_counts.values()),
                "generation_delivery_counts": dict(delivery_counts),
                "cached_retrieval_replay_pass": True,
            }
        )
        print(f"All actual contexts verified: {name}, {count:,} prompts", flush=True)

    for d in deliveries:
        if d["role"] == "evaluation":
            reference = prompts[d["phase"], d["key"], "reference"][
                "local_token_estimate"
            ]
        else:
            reference = prompts["source_initial", d["key"], "reference"][
                "local_token_estimate"
            ]
        d["reference_token_estimate"] = reference
        d["local_token_difference"] = d["local_token_estimate"] - reference
    executed = Counter()
    nulls, nvalid = 0, 0
    for stage in reader["stage_receipts"]:
        receipt = a.read(a.ROOT / stage["receipt"])
        a.need(
            a.sha(a.ROOT / stage["receipt"]) == stage["receipt_sha256"],
            "Reader stage receipt changed",
        )
        path = a.ROOT / stage["predictions"]
        a.need(bind(path) == receipt["predictions_sha256"], "Bound assembly changed")
        slots = set()
        for saved in a.rows(path):
            identity = (saved["phase"], saved["key"], saved["condition"])
            a.need(
                prompts[identity]["messages_sha256"] == saved["messages_sha256"],
                "Actually executed prompt differs",
            )
            slot = (*identity, saved["repeat"])
            a.need(
                slot not in slots and saved["repeat"] in [0, 1, 2],
                "Duplicate/invalid decoder slot",
            )
            slots.add(slot)
            executed[identity] += 1
            nulls += saved["result"] is None
            nvalid += saved["n_valid"]
    a.need(
        len(prompts) == len(executed) == 86408
        and set(executed.values()) == {3}
        and nulls == 242,
        "Full actual execution binding differs",
    )
    a.need(
        sum(d["role"] == "evaluation" for d in deliveries) == 74772
        and len(deliveries) == 77076,
        "Delivery matrix differs",
    )
    a.lines(a.OUT / "deliveries.jsonl", deliveries)
    a.lines(a.OUT / "prompt_bindings.jsonl", prompts.values())
    a.write(
        a.OUT / "context_cases.json",
        {"/".join(map(str, key)): value for key, value in sorted(cases.items())},
    )
    a.write(
        a.OUT / "execution_binding.json",
        {
            "status": "all_86408_prompts_matched_to_259224_audited_outputs",
            "phases": phase_checks,
            "prompt_assignments": len(prompts),
            "prediction_slots": sum(executed.values()),
            "evaluation_correction_assignments": 74772,
            "source_correction_assignments": 2304,
            "missing_predictions": nulls,
            "valid_votes": nvalid,
            "new_Qwen_calls": 0,
        },
    )
    for name, digest in binding.items():
        a.need(a.sha(a.ROOT / name) == digest, "Input changed during context build")
    a.write(
        a.OUT / "assembled.json",
        {
            "status": "selection_and_context_assembled_independent_review_pending",
            "created_utc": a.pure.utc(),
            "input_bindings": binding,
            "artifacts": {a.rel(p): a.sha(p) for p in a.OUT.iterdir() if p.is_file()},
            "elapsed_seconds": time.monotonic() - start,
            "new_Qwen_calls": 0,
            "new_fits": 0,
            "new_embeddings": 0,
        },
    )
    print(
        "All selection, delivery and replacement ledgers saved; no new model calls.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        build()

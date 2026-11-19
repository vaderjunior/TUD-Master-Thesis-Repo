from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
import numpy as np
from analysis.feedback.reserved import pools as a
from analysis.feedback.reserved import controls_context as previous
from analysis.feedback.reserved import scoring as report


def check_pool(pool, records, statuses):
    ids = pool["ids"]
    a.need(
        len(ids) == len(set(ids)) == pool["budget"] == 40, "Pool size/duplicate error"
    )
    gates = Counter(str(records[s]["meta"]["gate"]) for s in ids)
    actual = Counter(statuses[s]["status"] for s in ids)
    a.need(
        dict(actual) == pool["actual_status_counts"],
        "Earlier source status counts differ",
    )
    a.need(
        (
            gates == Counter({"True": 40})
            if pool["polarity"] == "positive"
            else gates == Counter({"True": 20, "False": 20})
        ),
        "Pool polarity differs",
    )
    if pool["selection"] != "random":
        a.need(
            actual == Counter({pool["selection"]: 40}),
            "Status-conditioned pool changed",
        )
    for sid in ids:
        a.need(
            records[sid]["meta"]["gate"] == statuses[sid]["hate"],
            "Source annotation differs from screened gold",
        )
        a.need(
            all(
                records[sid]["meta"].get(k) is None
                for k in ("hate_types", "target_groups", "severity", "legal")
            ),
            "Invented source fine labels",
        )
    return {
        "condition": pool["condition"],
        "selection": pool["selection"],
        "polarity": pool["polarity"],
        "seed": pool["seed"],
        "size": 40,
        "gate_counts": dict(gates),
        "previous_status_counts": dict(actual),
    }


def new_bucket():
    return {
        "counts": Counter(),
        "sources": Counter(),
        "selected_sources": Counter(),
        "tokens": [],
        "cosines": [],
        "keys": set(),
    }


def summarise_bucket(v):
    return {
        "counts": dict(v["counts"]),
        "unique_texts": len(v["keys"]),
        "unique_selected_sources": len(v["selected_sources"]),
        "selected_source_use": dict(v["selected_sources"]),
        "delivered_source_use": dict(v["sources"]),
        "token_count": previous.summary_numbers(v["tokens"]),
        "cosine": previous.summary_numbers(v["cosines"]) if v["cosines"] else None,
    }


def main():
    done = a.io.read(a.OUT / "statistics_complete.json")
    a.io.verify(a.io.ROOT, done["artifacts"])
    a.io.verify(a.io.ROOT, done["bindings"])
    a.need(
        not (a.OUT / "context_complete.json").exists(),
        "Preserve existing context analysis",
    )
    roster, science, predictions, assembly = a.load()
    bykey = {r["key"]: r for r in roster}
    selection = (
        a.io.ROOT
        / "experiments/astra/sq3_additions_results_20261006_v1/benign_selection"
    )
    pools = a.io.read(selection / "actual_pools.json")["pools"]
    records = a.io.read(selection / "gate_only_records.json")
    statuses = a.io.read(selection / "candidate_statuses.json")
    choices = list(a.io.rows(a.io.STUDY / "choices.jsonl"))
    a.need(len(pools) == 18 and len(choices) == 12150, "Incomplete source selection")
    paths = [
        Path(__file__),
        Path(previous.__file__),
        Path(report.__file__),
        a.io.STUDY / "prompts.jsonl",
        a.io.STUDY / "choices.jsonl",
        *[
            selection / n
            for n in (
                "actual_pools.json",
                "gate_only_records.json",
                "candidate_statuses.json",
                "complete.json",
            )
        ],
        a.prior.OUT / "context_complete.json",
        a.prior.OUT / "prompt_metadata.jsonl",
    ]
    bindings = {a.io.rel(p): a.io.sha(p) for p in paths}
    a.write(
        a.OUT / "context_started.json",
        {"created_utc": a.io.utc(), "bindings": bindings, "new_model_calls": 0},
    )
    status_counts = Counter()
    discordant = 0
    for sid, source in statuses.items():
        decisions = source["decisions"]
        a.need(
            len(decisions) == 3 and all(type(v) is bool for v in decisions),
            "Unknown screening decision",
        )
        majority = sum(decisions) >= 2
        status = "correct" if majority == source["hate"] else "error"
        a.need(
            source["majority_hate"] == majority and source["status"] == status,
            "Earlier source majority/status changed",
        )
        status_counts[status] += 1
        discordant += len(set(decisions)) > 1
    pool_info = [check_pool(p, records, statuses) for p in pools]
    overlap = []
    policy_sources = {}
    for p, q in itertools.combinations(pools, 2):
        u, v = set(p["ids"]), set(q["ids"])
        overlap.append(
            {
                "left": p["condition"],
                "right": q["condition"],
                "intersection": len(u & v),
                "union": len(u | v),
                "jaccard": len(u & v) / len(u | v),
            }
        )
    for policy in a.POLICIES:
        selected = [p for p in pools if p["condition"].rsplit("_s", 1)[0] == policy]
        ids = [set(p["ids"]) for p in selected]
        policy_sources[policy] = {
            "memberships": 120,
            "unique_sources": len(set.union(*ids)),
            "three_way_intersection": len(set.intersection(*ids)),
            "pair_intersections": [
                len(x & y) for x, y in itertools.combinations(ids, 2)
            ],
        }
    a.write(
        a.OUT / "source_pool_audit.json",
        {
            "candidate_status_counts": dict(status_counts),
            "candidate_count": len(statuses),
            "nonunanimous_screening_majorities": discordant,
            "pools": pool_info,
            "all_153_pool_overlaps": overlap,
            "policy_source_summary": policy_sources,
            "unique_all_pool_sources": len({sid for p in pools for sid in p["ids"]}),
            "new_source_labels_or_screening_calls": 0,
            "interpretation": "Error/correct is previous majority binary correctness on source texts, not evaluation outcome. Random controls can contain both. Fixed overlapping pools are not independent samples from future feedback.",
        },
    )
    meta = {}
    ledgers = []
    seen = set()
    counts = Counter()
    aggregates = defaultdict(new_bucket)
    for block in previous.prompt_blocks(set(science["conditions"])):
        a.need(
            set(block) == set(science["conditions"]), "Incomplete frozen prompt block"
        )
        ref = block["reference5"]
        key = ref["key"]
        a.need(key not in seen, "Duplicate input")
        seen.add(key)
        a.need(ref["text"] == bykey[key]["text"], "Roster text differs")
        previous.check_intervention(ref, block["reference4"], "reference4")
        for condition, p in block.items():
            a.need(
                previous.sig(p["messages"]) == p["messages_sha256"]
                and previous.sig(p["context"]) == p["context_sha256"],
                "Frozen prompt hash differs",
            )
            a.need(
                all(
                    predictions[key, condition, rep]["messages_sha256"]
                    == p["messages_sha256"]
                    for rep in range(3)
                ),
                "Prediction message identity differs",
            )
            a.need(
                p["context"]["definitions"] == ref["context"]["definitions"]
                and p["context"]["guidelines"] == ref["context"]["guidelines"]
                and p["text"] == ref["text"],
                "Abstracts/query changed",
            )
            counts["prompts_rehashed_and_prediction_bound"] += 1
        current, _ = report.prompt_diagnostics(block.values(), choices, records, pools)
        meta.update(current)
        for pool in pools:
            name = pool["condition"]
            policy = name.rsplit("_s", 1)[0]
            ps = {mode: block[mode + "__" + name] for mode in a.MODES}
            r = ps["reserved5"]
            source = records[r["source_id"]]
            hit = previous.examples(r)[-1]
            a.need(
                hit["id"] == source["id"]
                and hit["text"] == source["text"]
                and hit["meta"] == source["meta"],
                "Correction source content differs",
            )
            a.need(
                hit["id"] not in {e["id"] for e in previous.examples(ref)}
                and " ".join(hit["text"].split()) != " ".join(ref["text"].split()),
                "Correction duplicates query/ordinary example",
            )
            for mode in ("reserved5", "extended6", "masked5"):
                previous.check_intervention(ref, ps[mode], mode, hit)
            previous.check_masked_message(r, ps["masked5"])
            counts["label_only_pairs"] += 1
            counts["retention_position_pairs"] += 1
            a.need(
                all(
                    ps[m]["source_id"] == r["source_id"]
                    and ps[m]["selected_correction_cosine"]
                    == r["selected_correction_cosine"]
                    for m in a.MODES
                ),
                "Selected correction differs across modes",
            )
            n = current[key, "natural5__" + name]
            gate = source["meta"]["gate"]
            endpoint = "FNR" if bykey[key]["gold"]["hate"] else "FPR"
            ledger = {
                "key": key,
                "source": bykey[key]["source"],
                "gold_hate": bykey[key]["gold"]["hate"],
                "pool": name,
                "policy": policy,
                "seed": pool["seed"],
                "selected_source": r["source_id"],
                "selected_gate": gate,
                "selected_previous_status": statuses[r["source_id"]]["status"],
                "selected_cosine": r["selected_correction_cosine"],
                "natural_delivered_ids": n["delivered_correction_ids"],
                "natural_delivered_gates": [
                    v["meta"]["gate"] for v in n["delivered_annotations"]
                ],
                "natural_delivers_selected": r["source_id"]
                in n["delivered_correction_ids"],
                "natural_identical_reference": n["identical_to_reference"],
                "natural_removed_ordinary_count": len(n["removed_ordinary"]),
                "tokens": {m: ps[m]["reference_input_tokens"] for m in a.MODES},
                "messages_sha256": {m: ps[m]["messages_sha256"] for m in a.MODES},
            }
            ledgers.append(ledger)
            for mode in a.MODES:
                diag = current[key, mode + "__" + name]
                for cohort in ("all", endpoint):
                    bucket = aggregates[mode + "__" + policy + "/" + cohort]
                    ct = bucket["counts"]
                    ct["query_pool_pairs"] += 1
                    bucket["keys"].add(key)
                    ct["selected_gate_" + str(gate)] += 1
                    ct["selected_status_" + statuses[r["source_id"]]["status"]] += 1
                    ct["delivered_pairs"] += bool(diag["delivered_correction_ids"])
                    ct["delivered_records"] += len(diag["delivered_correction_ids"])
                    ct["delivers_selected"] += (
                        r["source_id"] in diag["delivered_correction_ids"]
                    )
                    ct["identical_reference"] += diag["identical_to_reference"]
                    ct["removed_fifth"] += bool(diag["removed_fifth"])
                    ct["removed_ordinary_total"] += len(diag["removed_ordinary"])
                    for d in diag["delivered_annotations"]:
                        ct["delivered_gate_" + str(d["meta"]["gate"])] += 1
                    bucket["sources"].update(diag["delivered_correction_ids"])
                    bucket["selected_sources"][r["source_id"]] += 1
                    bucket["tokens"].append(ps[mode]["reference_input_tokens"])
                    bucket["cosines"].append(r["selected_correction_cosine"])
        if len(seen) % 100 == 0:
            print(f"Policy contexts verified: {len(seen)}/675", flush=True)
    a.need(
        len(seen) == 675 and len(meta) == 49950 and len(ledgers) == 12150,
        "Incomplete policy context matrix",
    )
    reproduced = 0
    for row in a.io.rows(a.prior.OUT / "prompt_metadata.jsonl"):
        a.need(
            meta[row["key"], row["condition"]] == row, "Part 3 context metadata changed"
        )
        reproduced += 1
    a.need(reproduced == 9450, "Part 3 subset incomplete")
    a.jsonlines(a.OUT / "prompt_metadata.jsonl", meta.values())
    a.jsonlines(a.OUT / "query_pool_contexts.jsonl", ledgers)
    a.write(
        a.OUT / "policy_context_summary.json",
        {
            "groups": {k: summarise_bucket(v) for k, v in aggregates.items()},
            "checked": dict(counts),
            "part3_metadata_reproduced": reproduced,
            "interpretation": "Query/pool pairs are dependent. Selected label is not necessarily delivered. Masked annotations are unknown, not non-hateful. Token counts are frozen local estimates.",
        },
    )
    identity_diagnostics(roster, science, predictions, meta, ledgers)
    a.io.verify(a.io.ROOT, bindings)
    files = [
        a.OUT / n
        for n in (
            "context_started.json",
            "source_pool_audit.json",
            "prompt_metadata.jsonl",
            "query_pool_contexts.jsonl",
            "policy_context_summary.json",
            "direct_policy_identity_summary.json",
            "direct_policy_changed_pairs.jsonl",
        )
    ]
    a.write(
        a.OUT / "context_complete.json",
        {
            "status": "part4_all_policy_contexts_pools_and_identity_checks_complete",
            "created_utc": a.io.utc(),
            "bindings": bindings,
            "artifacts": {a.io.rel(p): a.io.sha(p) for p in files},
            "checked": dict(counts),
            "new_model_calls": 0,
        },
    )
    print(
        "All 18 source pools, 49,950 prompts and direct-policy identity diagnostics checked.",
        flush=True,
    )


def identity_diagnostics(roster, science, predictions, meta, ledgers):
    index = {(x["key"], x["pool"]): x for x in ledgers}
    results = a.io.read(a.OUT / "rate_comparisons.json")
    groups = defaultdict(lambda: Counter(keys=set()))
    contexts = {}
    changed = []
    for spec in a.direct_specs(science):
        result = results["full/" + spec["id"]]
        common = set(result["common_ids"])
        ct = Counter()
        unique_same = set()
        unique_changed = set()
        eligible = [
            r for r in roster if r["gold"]["hate"] == (spec["endpoint"] == "FNR")
        ]
        for row in eligible:
            key = row["key"]
            for t, c in zip(spec["treatment_conditions"], spec["control_conditions"]):
                tx = index[key, t.split("__", 1)[1]]
                cx = index[key, c.split("__", 1)[1]]
                same = (
                    meta[key, t]["messages_sha256"] == meta[key, c]["messages_sha256"]
                )
                ct["query_pool_pairs"] += 1
                ct["same_selected_source"] += (
                    tx["selected_source"] == cx["selected_source"]
                )
                ct["same_selected_label"] += tx["selected_gate"] == cx["selected_gate"]
                ct["identical_messages"] += same
                (unique_same if same else unique_changed).add(key)
                if key not in common:
                    continue
                for rep in range(3):
                    th = predictions[key, t, rep]["result"]["hate"]
                    ch = predictions[key, c, rep]["result"]["hate"]
                    for suffix in ("all", "identical" if same else "changed"):
                        previous.pair_update(
                            groups[spec["id"] + "/" + suffix],
                            key,
                            th,
                            ch,
                            row["gold"]["hate"],
                        )
                    if th != ch:
                        changed.append(
                            {
                                "contrast": spec["id"],
                                "key": key,
                                "treatment": t,
                                "control": c,
                                "repeat": rep,
                                "identical_messages": same,
                                "gold_hate": row["gold"]["hate"],
                                "treatment_hate": th,
                                "control_hate": ch,
                                "selected_sources": [
                                    tx["selected_source"],
                                    cx["selected_source"],
                                ],
                            }
                        )
        contexts[spec["id"]] = {
            "counts": dict(ct),
            "unique_texts_with_identical_context": len(unique_same),
            "unique_texts_with_changed_context": len(unique_changed),
            "endpoint_eligible_texts": len(eligible),
        }
    final = previous.finish_groups(groups)
    for spec in a.direct_specs(science):
        a.need(
            abs(
                final[spec["id"] + "/all"]["difference"]
                - results["full/" + spec["id"]]["difference"]
            )
            < 1e-12,
            "Policy identity transition totals disagree with paired rate",
        )
    a.write(
        a.OUT / "direct_policy_identity_summary.json",
        {
            "contexts": contexts,
            "outcome_groups": final,
            "interpretation": "Context counts use the full endpoint roster; outcome pairs use the direct contrast common mask. All pairs are dependent. Identical-message flips are generation variation, not a content effect.",
        },
    )
    a.jsonlines(a.OUT / "direct_policy_changed_pairs.jsonl", changed)


if __name__ == "__main__":
    with a.io.offline():
        main()

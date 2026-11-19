from collections import Counter, defaultdict
import hashlib
from pathlib import Path
import numpy as np
from analysis.feedback.rounds import contexts_inputs as a


def distribution(values):
    x = np.asarray(list(values), dtype=np.float64)
    return {
        "n": len(x),
        "mean": float(x.mean()) if len(x) else None,
        "quantiles": (
            np.quantile(x, [0, 0.25, 0.5, 0.75, 1]).tolist() if len(x) else None
        ),
    }


def exposure(deliveries, member_ids):
    counts = Counter(d["source_id"] for d in deliveries)
    n = len(deliveries)
    a.need(n > 0, "Empty exposure population")
    return {
        "assignments": n,
        "hateful_correction": sum(d["source_gate"] for d in deliveries),
        "hateful_removed": sum(d["removed_gate"] for d in deliveries),
        "newest_available_round": sum(d["newest_available_round"] for d in deliveries),
        "added_round_counts": dict(Counter(d["added_round"] for d in deliveries)),
        "source_counts": {sid: counts[sid] for sid in member_ids},
        "unused_source_ids": [sid for sid in member_ids if counts[sid] == 0],
        "concentration": a.context.concentration([counts[sid] for sid in member_ids]),
        "swaps": {
            f"{int(old)}-{int(new)}": sum(
                d["removed_gate"] == old and d["source_gate"] == new for d in deliveries
            )
            for old in [False, True]
            for new in [False, True]
        },
        "known_fine_removed": {
            f: sum(d["removed_known_fine"][f] for d in deliveries) for f in a.FIELDS
        },
        "last_known_example_removed": {
            f: sum(
                d["removed_known_fine"][f] and d["remaining_known_fine_counts"][f] == 0
                for d in deliveries
            )
            for f in a.FIELDS
        },
        "any_known_fine_removed": sum(
            any(d["removed_known_fine"].values()) for d in deliveries
        ),
        "hateful_example_change": dict(
            Counter(
                d["feedback_hateful_examples"] - d["reference_hateful_examples"]
                for d in deliveries
            )
        ),
        "retrieval_channels": dict(
            Counter(d["retrieval_channels"] for d in deliveries)
        ),
        "dense_winner_selected": sum(d["dense_winner_same"] for d in deliveries),
        "lexical_winner_selected": sum(d["lexical_winner_same"] for d in deliveries),
        "rrf_exact_top_tie": sum(d["rrf_margin"] == 0 for d in deliveries),
        "cosine": distribution(d["cosine"] for d in deliveries),
        "cosine_difference": distribution(d["cosine_difference"] for d in deliveries),
        "correction_higher_cosine_than_fifth": sum(
            d["cosine_difference"] > 1e-12 for d in deliveries
        ),
        "local_token_difference": distribution(
            d["local_token_difference"] for d in deliveries
        ),
    }


def feature_rows(
    deliveries, matrix, round_no, population, endpoint, old, new, policy=None
):
    grouped = defaultdict(lambda: [0, 0, 0])
    for d in deliveries:
        i = d["input_index"]
        row = matrix[i]
        if (
            d["round"] != round_no
            or d["population"] != population
            or row["gold_hate"] != (endpoint == "FNR")
        ):
            continue
        if (
            d["removed_gate"] != old
            or d["source_gate"] != new
            or (policy is not None and d["policy"] != policy)
        ):
            continue
        if not row["masks"]["all_round_common"]:
            continue
        ref = row["predictions"][round_no - 1][0]
        trt = row["predictions"][round_no - 1][a.CONDITIONS.index(d["condition"])]
        a.need(
            all(v["result"] is not None for v in ref + trt),
            "Null admitted into context association",
        )
        grouped[i][0] += 1
        grouped[i][1] += sum(v["result"]["hate"] != row["gold_hate"] for v in ref)
        grouped[i][2] += sum(v["result"]["hate"] != row["gold_hate"] for v in trt)
    indices = sorted(grouped)
    return indices, np.asarray([grouped[i] for i in indices], dtype=np.int64).reshape(
        (-1, 3)
    )


def analyse():
    assembled = a.read(a.OUT / "assembled.json")
    for name, digest in assembled["artifacts"].items():
        a.need(a.sha(a.ROOT / name) == digest, "Assembly changed")
    selections = a.read(a.OUT / "selections.json")
    candidates = list(a.rows(a.OUT / "candidate_selection_ledger.jsonl"))
    members = list(a.rows(a.OUT / "pool_memberships.jsonl"))
    all_deliveries = list(a.rows(a.OUT / "deliveries.jsonl"))
    delivered = [d for d in all_deliveries if d["role"] == "evaluation"]
    source_delivered = [d for d in all_deliveries if d["role"] == "source"]
    matrix = list(a.rows(a.READER / "matrix.jsonl"))
    primary = a.read(a.PART2 / "errors_v1/results.json")["records"]
    pools = {
        (s["round"], s["stream"], p): s["cumulative"][p]
        for s in selections
        for p in a.POLICIES
    }
    overlaps, source_changes = [], []
    for s in selections:
        for first, second in [
            ("adaptive", "static"),
            ("adaptive", "random"),
            ("static", "random"),
        ]:
            for kind in ["additions", "cumulative"]:
                left, right = set(s[kind][first]), set(s[kind][second])
                overlaps.append(
                    {
                        "round": s["round"],
                        "stream": s["stream"],
                        "kind": kind,
                        "first": first,
                        "second": second,
                        "first_n": len(left),
                        "second_n": len(right),
                        "intersection_n": len(left & right),
                        "union_n": len(left | right),
                        "jaccard": len(left & right) / len(left | right),
                        "same_set": left == right,
                        "first_only": sorted(left - right),
                        "second_only": sorted(right - left),
                    }
                )
        for gold in [False, True]:
            rows = [
                r
                for r in candidates
                if r["round"] == s["round"]
                and r["stream"] == s["stream"]
                and r["gold_hate"] == gold
            ]
            valid = [r for r in rows if r["common_valid"]]
            source_changes.append(
                {
                    "round": s["round"],
                    "stream": s["stream"],
                    "gold_hate": gold,
                    "opportunities": len(rows),
                    "eligible": len(valid),
                    "excluded": len(rows) - len(valid),
                    "initial_errors": sum(r["initial_majority"] != gold for r in valid),
                    "current_errors": sum(r["current_majority"] != gold for r in valid),
                    "transitions": dict(Counter(r["error_transition"] for r in rows)),
                    "random_selected_initial_errors": sum(
                        r["selected"]["random"] and r["initial_majority"] != gold
                        for r in valid
                    ),
                    "random_selected_current_errors": sum(
                        r["selected"]["random"] and r["current_majority"] != gold
                        for r in valid
                    ),
                }
            )
    grouped = defaultdict(list)
    for d in delivered:
        grouped[d["round"], d["stream"], d["policy"], d["population"]].append(d)
    exposure_rows = []
    for (rnd, stream, policy, population), values in sorted(grouped.items()):
        exposure_rows.append(
            {
                "round": rnd,
                "stream": stream,
                "policy": policy,
                "population": population,
                "pool_size": len(pools[rnd, stream, policy]),
                **exposure(values, pools[rnd, stream, policy]),
            }
        )
    source_exposure = []
    for rnd in range(2, 5):
        for stream in a.STREAMS:
            values = [
                d
                for d in source_delivered
                if d["round"] == rnd and d["stream"] == stream
            ]
            source_exposure.append(
                {
                    "round": rnd,
                    "stream": stream,
                    "available_round": rnd - 1,
                    **exposure(values, pools[rnd - 1, stream, "adaptive"]),
                }
            )
    index = {(d["round"], d["stream"], d["policy"], d["key"]): d for d in delivered}
    policy_delivery, transitions = [], []
    for population in ["en_test", "en_hatecheck_eval"]:
        keys = [r["key"] for r in matrix if r["population"] == population]
        for rnd in range(1, 5):
            for stream in a.STREAMS:
                for left, right in [
                    ("adaptive", "static"),
                    ("adaptive", "random"),
                    ("static", "random"),
                ]:
                    pairs = [
                        (index[rnd, stream, left, key], index[rnd, stream, right, key])
                        for key in keys
                    ]
                    policy_delivery.append(
                        {
                            "round": rnd,
                            "stream": stream,
                            "population": population,
                            "first": left,
                            "second": right,
                            "inputs": len(keys),
                            "same_source": sum(
                                x["source_id"] == y["source_id"] for x, y in pairs
                            ),
                            "same_prompt": sum(
                                x["messages_sha256"] == y["messages_sha256"]
                                for x, y in pairs
                            ),
                            "same_gate": sum(
                                x["source_gate"] == y["source_gate"] for x, y in pairs
                            ),
                            "cumulative_intersection": len(
                                set(pools[rnd, stream, left])
                                & set(pools[rnd, stream, right])
                            ),
                        }
                    )
                if rnd == 1:
                    continue
                for policy in a.POLICIES:
                    states = Counter()
                    for key in keys:
                        old, new = (
                            index[rnd - 1, stream, policy, key],
                            index[rnd, stream, policy, key],
                        )
                        same = old["source_id"] == new["source_id"]
                        if same:
                            a.need(
                                old["messages_sha256"] == new["messages_sha256"],
                                "Same correction changed prompt",
                            )
                        status = (
                            "same_source"
                            if same
                            else (
                                "switch_to_new"
                                if new["added_round"] == rnd
                                else "switch_to_old"
                            )
                        )
                        states[status] += 1
                        states["gate_changed"] += (
                            old["source_gate"] != new["source_gate"]
                        )
                    transitions.append(
                        {
                            "round": rnd,
                            "stream": stream,
                            "policy": policy,
                            "population": population,
                            "inputs": len(keys),
                            "counts": dict(states),
                        }
                    )
    population_totals = {}
    for population in ["en_test", "en_hatecheck_eval"]:
        population_totals[population] = exposure(
            [d for d in delivered if d["population"] == population],
            sorted(
                {d["source_id"] for d in delivered if d["population"] == population}
            ),
        )

    associations, per_policy = {}, []
    for rnd in range(1, 5):
        for population in ["en_test", "en_hatecheck_eval"]:
            for endpoint in ["FPR", "FNR"]:
                mixture = np.zeros(3, dtype=np.int64)
                for old in [False, True]:
                    for new in [False, True]:
                        name = f"r{rnd}_{population}_{endpoint}_{int(old)}-{int(new)}"
                        indices, features = feature_rows(
                            delivered, matrix, rnd, population, endpoint, old, new
                        )
                        sums = features.sum(axis=0)
                        mixture += sums
                        point = a.context.point(sums)
                        seed = int.from_bytes(
                            hashlib.sha256(
                                ("A17-part3-swap-v1|" + name).encode()
                            ).digest()[:8],
                            "little",
                        )
                        clusters = [matrix[i]["inference_cluster"] for i in indices]
                        record = {
                            "round": rnd,
                            "population": population,
                            "endpoint": endpoint,
                            "removed_gate": old,
                            "inserted_gate": new,
                            **point,
                            "unique_inputs": len(indices),
                            "inference_clusters": len(set(clusters)),
                            "input_indices": indices,
                            "features": features.tolist(),
                            "seed": seed,
                            "interval_level": 0.95,
                            "columns": ["reference", "treatment", "difference"],
                            "status": (
                                "defined" if len(indices) else "undefined_empty_stratum"
                            ),
                        }
                        if len(indices):
                            draws = a.context.boot(
                                features, clusters, seed, draws=10000
                            )
                            path = a.OUT / "association_draws" / (name + ".npy")
                            path.parent.mkdir(parents=True, exist_ok=True)
                            with path.open("xb") as f:
                                np.save(f, draws, allow_pickle=False)
                            record.update(
                                draw_file=a.rel(path),
                                draw_sha256=a.sha(path),
                                intervals=np.quantile(
                                    draws, [0.025, 0.975], axis=0, method="linear"
                                ).T.tolist(),
                            )
                        associations[name] = record
                        for policy in a.POLICIES:
                            pi, pf = feature_rows(
                                delivered,
                                matrix,
                                rnd,
                                population,
                                endpoint,
                                old,
                                new,
                                policy,
                            )
                            per_policy.append(
                                {
                                    "name": name,
                                    "policy": policy,
                                    "unique_inputs": len(pi),
                                    **a.context.point(pf.sum(axis=0)),
                                }
                            )
                reference = primary[f"trajectory_{population}_{endpoint}"]
                expected_treatment = (
                    sum(reference["points"][f"r{rnd}_{p}"] for p in a.POLICIES) / 3
                )
                mixed = a.context.point(mixture)
                a.need(
                    abs(mixed["reference"] - reference["points"][f"r{rnd}_reference"])
                    < 1e-13
                    and abs(mixed["treatment"] - expected_treatment) < 1e-13,
                    "Context strata do not reconstruct Part 2 fixed-cohort rates",
                )
                print(
                    f"Context associations saved: round {rnd}, {population}, {endpoint}",
                    flush=True,
                )

    transition_rates = []
    for rnd in range(2, 5):
        for population in ["en_test", "en_hatecheck_eval"]:
            for endpoint in ["FPR", "FNR"]:
                totals = defaultdict(lambda: [0, 0, 0, 0, 0])
                for d in delivered:
                    if d["round"] != rnd or d["population"] != population:
                        continue
                    row = matrix[d["input_index"]]
                    if (
                        row["gold_hate"] != (endpoint == "FNR")
                        or not row["masks"]["all_round_common"]
                    ):
                        continue
                    old = index[rnd - 1, d["stream"], d["policy"], d["key"]]
                    category = (
                        "same_source"
                        if old["source_id"] == d["source_id"]
                        else (
                            "switch_to_new"
                            if d["added_round"] == rnd
                            else "switch_to_old"
                        )
                    )
                    col = a.CONDITIONS.index(d["condition"])
                    values = [
                        sum(
                            v["result"]["hate"] != row["gold_hate"]
                            for v in row["predictions"][r][c]
                        )
                        for r, c in [
                            (rnd - 2, col),
                            (rnd - 1, col),
                            (rnd - 2, 0),
                            (rnd - 1, 0),
                        ]
                    ]
                    totals[category] = [
                        x + y for x, y in zip(totals[category], [1, *values])
                    ]
                for category, counts in totals.items():
                    n, old, new, oldref, newref = counts
                    transition_rates.append(
                        {
                            "round": rnd,
                            "population": population,
                            "endpoint": endpoint,
                            "category": category,
                            "counts": counts,
                            "previous_rate": old / (3 * n),
                            "current_rate": new / (3 * n),
                            "change": (new - old) / (3 * n),
                            "reference_adjusted_change": (new - old - newref + oldref)
                            / (3 * n),
                        }
                    )
    summary = {
        "status": "part3_context_summaries_scored_independent_verification_pending",
        "created_utc": a.pure.utc(),
        "selection": {
            "round_streams": len(selections),
            "candidate_opportunities": len(candidates),
            "unique_candidates": len({r["source_id"] for r in candidates}),
            "common_eligible_opportunities": sum(r["common_valid"] for r in candidates),
            "selected_addition_memberships": sum(
                len(s["additions"][p]) for s in selections for p in a.POLICIES
            ),
            "cumulative_memberships": len(members),
            "unique_selected_ids": len({m["source_id"] for m in members}),
            "distinct_cumulative_sets": len({tuple(p) for p in pools.values()}),
            "all_per_class_budgets": [s["per_class_budget"] for s in selections],
            "overlaps": overlaps,
            "source_changes": source_changes,
        },
        "exposure": exposure_rows,
        "source_exposure": source_exposure,
        "population_totals": population_totals,
        "policy_delivery_overlap": policy_delivery,
        "delivery_transitions": transitions,
        "transition_error_associations": transition_rates,
        "polarity_associations": associations,
        "per_policy_associations": per_policy,
        "inputs": {
            a.rel(p): a.sha(p)
            for p in [
                a.OUT / "assembled.json",
                a.OUT / "analysis_contract.json",
                a.PART2 / "errors_v1/results.json",
                a.READER / "matrix.jsonl",
            ]
        },
        "analysis_source_sha256": a.sha(Path(__file__)),
        "new_Qwen_calls": 0,
        "new_fits": 0,
        "new_embeddings": 0,
    }
    a.write(a.OUT / "analysis.json", summary)
    print(
        "All 64 planned polarity-swap cells and all delivery summaries saved.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        analyse()

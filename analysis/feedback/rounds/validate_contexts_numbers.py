from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
from pathlib import Path
import numpy as np
from analysis.feedback.rounds import contexts_inputs as io
from analysis.feedback.rounds.validate_contexts_context import check
from analysis.feedback.rounds.validate_contexts_context import equal


def percentile(values, q):
    x = sorted(values)
    at = (len(x) - 1) * q
    left = int(at)
    right = min(left + 1, len(x) - 1)
    return x[left] + (x[right] - x[left]) * (at - left)


def distribution(values):
    x = list(values)
    return dict(
        n=len(x),
        mean=sum(x) / len(x) if x else None,
        quantiles=[percentile(x, q) for q in [0, 0.25, 0.5, 0.75, 1]] if x else None,
    )


def exposure(rows, ids):
    counts = {s: sum(d["source_id"] == s for d in rows) for s in ids}
    values = sorted(counts.values(), reverse=True)
    n = len(rows)
    return dict(
        assignments=n,
        hateful_correction=sum(d["source_gate"] is True for d in rows),
        hateful_removed=sum(d["removed_gate"] is True for d in rows),
        newest_available_round=sum(
            d["added_round"] == d["available_round"] for d in rows
        ),
        added_round_counts={
            str(k): v for k, v in Counter(d["added_round"] for d in rows).items()
        },
        source_counts=counts,
        unused_source_ids=[s for s in ids if not counts[s]],
        concentration=dict(
            total=n,
            used=sum(v != 0 for v in values),
            max_share=values[0] / n,
            top5_share=sum(values[:5]) / n,
            top10_share=sum(values[:10]) / n,
            inverse_simpson=float(Fraction(n * n, sum(v * v for v in values))),
        ),
        swaps={
            f"{o}-{v}": sum(
                int(d["removed_gate"]) == o and int(d["source_gate"]) == v for d in rows
            )
            for o in (0, 1)
            for v in (0, 1)
        },
        known_fine_removed={
            f: sum(d["removed_known_fine"][f] for d in rows) for f in io.FIELDS
        },
        last_known_example_removed={
            f: sum(
                d["removed_known_fine"][f] and not d["remaining_known_fine_counts"][f]
                for d in rows
            )
            for f in io.FIELDS
        },
        any_known_fine_removed=sum(
            any(d["removed_known_fine"][f] for f in io.FIELDS) for d in rows
        ),
        hateful_example_change={
            str(k): v
            for k, v in Counter(
                int(d["source_gate"]) - int(d["removed_gate"]) for d in rows
            ).items()
        },
        retrieval_channels=dict(Counter(d["retrieval_channels"] for d in rows)),
        dense_winner_selected=sum(d["dense_winner_same"] for d in rows),
        lexical_winner_selected=sum(d["lexical_winner_same"] for d in rows),
        rrf_exact_top_tie=sum(d["rrf_margin"] == 0 for d in rows),
        cosine=distribution(d["cosine"] for d in rows),
        cosine_difference=distribution(d["cosine"] - d["removed_cosine"] for d in rows),
        correction_higher_cosine_than_fifth=sum(
            d["cosine"] - d["removed_cosine"] > 1e-12 for d in rows
        ),
        local_token_difference=distribution(
            d["local_token_estimate"] - d["reference_token_estimate"] for d in rows
        ),
    )


def point(counts):
    n, ref, trt = map(int, counts)
    return dict(
        assignments=n,
        reference_errors=ref,
        treatment_errors=trt,
        reference=float(Fraction(ref, 3 * n)) if n else None,
        treatment=float(Fraction(trt, 3 * n)) if n else None,
        difference=float(Fraction(trt - ref, 3 * n)) if n else None,
    )


def main():
    result = io.read(io.OUT / "analysis.json")
    context_proof = io.read(io.OUT / "context_independent_verification.json")
    check(context_proof["status"] == "passed", "Original context proof missing")
    for p, sha in result["inputs"].items():
        check(io.sha(io.ROOT / p) == sha, "Analysis input changed")
    check(
        io.sha(Path(__file__).with_name("astra_sq3_iterative_part3_analyse_v1.py"))
        == result["analysis_source_sha256"],
        "Analysis implementation changed",
    )
    selections = io.read(io.OUT / "selections.json")
    members = list(io.rows(io.OUT / "pool_memberships.jsonl"))
    opportunities = list(io.rows(io.OUT / "candidate_selection_ledger.jsonl"))
    pools = {
        (s["round"], s["stream"], p): s["cumulative"][p]
        for s in selections
        for p in io.POLICIES
    }
    expected_selection = dict(
        round_streams=12,
        candidate_opportunities=len(opportunities),
        unique_candidates=len({r["source_id"] for r in opportunities}),
        common_eligible_opportunities=sum(r["common_valid"] for r in opportunities),
        selected_addition_memberships=sum(
            sum(r["selected"].values()) for r in opportunities
        ),
        cumulative_memberships=len(members),
        unique_selected_ids=len({r["source_id"] for r in members}),
        distinct_cumulative_sets=len({frozenset(ids) for ids in pools.values()}),
        all_per_class_budgets=[s["per_class_budget"] for s in selections],
    )
    for k, value in expected_selection.items():
        equal(result["selection"][k], value, "Selection totals/" + k)
    selections_by_id = {(s["round"], s["stream"]): s for s in selections}
    for row in result["selection"]["overlaps"]:
        s = selections_by_id[row["round"], row["stream"]]
        x = set(s[row["kind"]][row["first"]])
        y = set(s[row["kind"]][row["second"]])
        equal(
            row,
            dict(
                round=s["round"],
                stream=s["stream"],
                kind=row["kind"],
                first=row["first"],
                second=row["second"],
                first_n=len(x),
                second_n=len(y),
                intersection_n=len(x & y),
                union_n=len(x | y),
                jaccard=len(x & y) / len(x | y),
                same_set=x == y,
                first_only=sorted(x - y),
                second_only=sorted(y - x),
            ),
            "Selected-set overlap",
        )
    for row in result["selection"]["source_changes"]:
        g = row["gold_hate"]
        subset = [
            r
            for r in opportunities
            if (r["round"], r["stream"], r["gold_hate"])
            == (row["round"], row["stream"], g)
        ]
        valid = [r for r in subset if r["common_valid"]]
        equal(
            row,
            dict(
                round=row["round"],
                stream=row["stream"],
                gold_hate=g,
                opportunities=len(subset),
                eligible=len(valid),
                excluded=len(subset) - len(valid),
                initial_errors=sum(r["initial_majority"] != g for r in valid),
                current_errors=sum(r["current_majority"] != g for r in valid),
                transitions=dict(Counter(r["error_transition"] for r in subset)),
                random_selected_initial_errors=sum(
                    r["selected"]["random"] and r["initial_majority"] != g
                    for r in valid
                ),
                random_selected_current_errors=sum(
                    r["selected"]["random"] and r["current_majority"] != g
                    for r in valid
                ),
            ),
            "Source changes",
        )
    all_d = list(io.rows(io.OUT / "deliveries.jsonl"))
    data = [d for d in all_d if d["role"] == "evaluation"]
    matrix = list(io.rows(io.READER / "matrix.jsonl"))
    for row in result["exposure"]:
        rnd, stream, p, pop = [
            row[k] for k in ("round", "stream", "policy", "population")
        ]
        subset = [
            d
            for d in data
            if (d["round"], d["stream"], d["policy"], d["population"])
            == (rnd, stream, p, pop)
        ]
        equal(
            row,
            dict(
                round=rnd,
                stream=stream,
                policy=p,
                population=pop,
                pool_size=len(pools[rnd, stream, p]),
                **exposure(subset, pools[rnd, stream, p]),
            ),
            "Exposure",
        )
    for row in result["source_exposure"]:
        rnd, stream = row["round"], row["stream"]
        subset = [
            d
            for d in all_d
            if d["role"] == "source" and (d["round"], d["stream"]) == (rnd, stream)
        ]
        equal(
            row,
            dict(
                round=rnd,
                stream=stream,
                available_round=rnd - 1,
                **exposure(subset, pools[rnd - 1, stream, "adaptive"]),
            ),
            "Source exposure",
        )
    for population, row in result["population_totals"].items():
        subset = [d for d in data if d["population"] == population]
        equal(
            row,
            exposure(subset, sorted({d["source_id"] for d in subset})),
            "Population exposure",
        )
    index = {(d["round"], d["stream"], d["policy"], d["key"]): d for d in data}
    for row in result["policy_delivery_overlap"]:
        rnd, stream, pop, p, q = [
            row[k] for k in ("round", "stream", "population", "first", "second")
        ]
        keys = [r["key"] for r in matrix if r["population"] == pop]
        pairs = [
            (index[rnd, stream, p, key], index[rnd, stream, q, key]) for key in keys
        ]
        equal(
            row,
            dict(
                round=rnd,
                stream=stream,
                population=pop,
                first=p,
                second=q,
                inputs=len(keys),
                same_source=sum(x["source_id"] == y["source_id"] for x, y in pairs),
                same_prompt=sum(
                    x["messages_sha256"] == y["messages_sha256"] for x, y in pairs
                ),
                same_gate=sum(x["source_gate"] == y["source_gate"] for x, y in pairs),
                cumulative_intersection=len(
                    set(pools[rnd, stream, p]) & set(pools[rnd, stream, q])
                ),
            ),
            "Policy delivery overlap",
        )
    for row in result["delivery_transitions"]:
        rnd, stream, p, pop = [
            row[k] for k in ("round", "stream", "policy", "population")
        ]
        counts = Counter()
        keys = [r["key"] for r in matrix if r["population"] == pop]
        for key in keys:
            x, y = index[rnd - 1, stream, p, key], index[rnd, stream, p, key]
            cat = (
                "same_source"
                if x["source_id"] == y["source_id"]
                else "switch_to_new" if y["added_round"] == rnd else "switch_to_old"
            )
            counts[cat] += 1
            counts["gate_changed"] += x["source_gate"] != y["source_gate"]
        equal(
            row,
            dict(
                round=rnd,
                stream=stream,
                policy=p,
                population=pop,
                inputs=len(keys),
                counts=dict(counts),
            ),
            "Delivery changes",
        )
    print(
        "All selection, concentration, exposure, displacement and delivery-change summaries agree.",
        flush=True,
    )

    cells = defaultdict(lambda: defaultdict(lambda: np.zeros(3, dtype=np.int64)))
    policy_cells = defaultdict(lambda: np.zeros(3, dtype=np.int64))
    policy_inputs = defaultdict(set)
    transition_cells = defaultdict(lambda: np.zeros(5, dtype=np.int64))
    for d in data:
        i = d["input_index"]
        r = matrix[i]
        if not r["masks"]["all_round_common"]:
            continue
        rnd, p, pop = d["round"], d["policy"], d["population"]
        endpoint = "FNR" if r["gold_hate"] else "FPR"
        name = (
            f'r{rnd}_{pop}_{endpoint}_{int(d["removed_gate"])}-{int(d["source_gate"])}'
        )
        col = io.CONDITIONS.index(d["condition"])
        ref = sum(
            x["result"]["hate"] != r["gold_hate"] for x in r["predictions"][rnd - 1][0]
        )
        trt = sum(
            x["result"]["hate"] != r["gold_hate"]
            for x in r["predictions"][rnd - 1][col]
        )
        cells[name][i] += [1, ref, trt]
        policy_cells[name, p] += [1, ref, trt]
        policy_inputs[name, p].add(i)
        if rnd > 1:
            prev = index[rnd - 1, d["stream"], p, d["key"]]
            cat = (
                "same_source"
                if prev["source_id"] == d["source_id"]
                else "switch_to_new" if d["added_round"] == rnd else "switch_to_old"
            )
            older = sum(
                x["result"]["hate"] != r["gold_hate"]
                for x in r["predictions"][rnd - 2][col]
            )
            older_ref = sum(
                x["result"]["hate"] != r["gold_hate"]
                for x in r["predictions"][rnd - 2][0]
            )
            transition_cells[rnd, pop, endpoint, cat] += [1, older, trt, older_ref, ref]
    max_error, scalar_count, defined = 0.0, 0, 0
    for name, row in result["polarity_associations"].items():
        indices = sorted(cells[name])
        features = np.asarray(
            [cells[name][i] for i in indices], dtype=np.int64
        ).reshape((-1, 3))
        points = point(features.sum(axis=0))
        for k, v in points.items():
            equal(row[k], v, name + "/" + k)
        check(
            row["input_indices"] == indices and row["features"] == features.tolist(),
            "Every conditional text numerator",
        )
        clusters = [matrix[i]["inference_cluster"] for i in indices]
        units = sorted(set(clusters))
        check(
            row["unique_inputs"] == len(indices)
            and row["inference_clusters"] == len(units),
            "Conditional input and cluster counts",
        )
        seed = int.from_bytes(
            hashlib.sha256(("A17-part3-swap-v1|" + name).encode()).digest()[:8],
            "little",
        )
        check(
            row["seed"] == seed
            and row["columns"] == ["reference", "treatment", "difference"],
            "Seed and column order",
        )
        if not indices:
            check(
                row["status"] == "undefined_empty_stratum" and "draw_file" not in row,
                "Empty cell preserved",
            )
            continue
        defined += 1
        path = io.ROOT / row["draw_file"]
        check(io.sha(path) == row["draw_sha256"], "Bootstrap artifact hash")
        observed = np.load(path, allow_pickle=False)
        check(observed.shape == (10000, 3), "Draw matrix dimensions")
        membership = np.asarray([units.index(c) for c in clusters])
        rng = np.random.Generator(np.random.PCG64(seed))
        replay = np.empty_like(observed)
        for start in range(0, 10000, 32):
            draws = rng.integers(
                0, len(units), size=(min(32, 10000 - start), len(units)), dtype=np.int64
            )

            weights = np.asarray(
                [np.bincount(d, minlength=len(units))[membership] for d in draws]
            )
            totals = weights @ features
            replay[start : start + len(draws), 0] = totals[:, 1] / (3 * totals[:, 0])
            replay[start : start + len(draws), 1] = totals[:, 2] / (3 * totals[:, 0])
            replay[start : start + len(draws), 2] = (totals[:, 2] - totals[:, 1]) / (
                3 * totals[:, 0]
            )
        err = float(np.max(np.abs(observed - replay)))
        max_error = max(max_error, err)
        scalar_count += observed.size
        check(err < 1e-13, "All independently weighted bootstrap draws")
        equal(
            row["intervals"],
            [[percentile(replay[:, c], q) for q in [0.025, 0.975]] for c in range(3)],
            "Independent sorted linear intervals",
        )
    for row in result["per_policy_associations"]:
        name, p = row["name"], row["policy"]
        equal(
            row,
            dict(
                name=name,
                policy=p,
                unique_inputs=len(policy_inputs[name, p]),
                **point(policy_cells[name, p]),
            ),
            "Per-policy context association",
        )
    for row in result["transition_error_associations"]:
        key = tuple(row[k] for k in ["round", "population", "endpoint", "category"])
        n, old, new, oldref, newref = map(int, transition_cells[key])
        equal(
            row,
            dict(
                zip(["round", "population", "endpoint", "category"], key),
                counts=[n, old, new, oldref, newref],
                previous_rate=old / (3 * n),
                current_rate=new / (3 * n),
                change=(new - old) / (3 * n),
                reference_adjusted_change=(new - old - newref + oldref) / (3 * n),
            ),
            "Transition context association",
        )
    primary = io.read(io.PART2 / "errors_v1/results.json")["records"]
    for rnd in range(1, 5):
        for pop in result["population_totals"]:
            for endpoint in ["FPR", "FNR"]:
                counts = np.zeros(3, dtype=np.int64)
                for old in [0, 1]:
                    for new in [0, 1]:
                        for f in cells[f"r{rnd}_{pop}_{endpoint}_{old}-{new}"].values():
                            counts += f
                p = point(counts)
                truth = primary[f"trajectory_{pop}_{endpoint}"]["points"]
                equal(
                    p["reference"],
                    truth[f"r{rnd}_reference"],
                    "Fixed reference marginal",
                )
                equal(
                    p["treatment"],
                    sum(truth[f"r{rnd}_{pol}"] for pol in io.POLICIES) / 3,
                    "Fixed treatment marginal",
                )
    io.write(
        io.OUT / "numerical_independent_verification.json",
        dict(
            status="passed",
            created_utc=io.pure.utc(),
            verifier_source_sha256=io.sha(Path(__file__)),
            selection_overlap_rows=len(result["selection"]["overlaps"]),
            exposure_rows=len(result["exposure"]),
            source_exposure_rows=len(result["source_exposure"]),
            policy_overlap_rows=len(result["policy_delivery_overlap"]),
            delivery_transition_rows=len(result["delivery_transitions"]),
            planned_association_cells=len(result["polarity_associations"]),
            defined_association_cells=defined,
            per_policy_associations=len(result["per_policy_associations"]),
            transition_associations=len(result["transition_error_associations"]),
            bootstrap_scalars=scalar_count,
            max_bootstrap_absolute_difference=max_error,
            inference="All supporting intervals are nominal 95%, conditional on fixed complete cohorts, streams and model outputs; no new primary family.",
            inputs={
                io.rel(p): io.sha(p)
                for p in [
                    io.OUT / "analysis.json",
                    io.OUT / "context_independent_verification.json",
                    io.READER / "matrix.jsonl",
                ]
            },
            new_Qwen_calls=0,
        ),
    )
    print(
        f"Independent verification passed: {defined} defined cells, {scalar_count:,} bootstrap scalars; max difference {max_error:g}.",
        flush=True,
    )


if __name__ == "__main__":
    with io.pure.offline():
        main()

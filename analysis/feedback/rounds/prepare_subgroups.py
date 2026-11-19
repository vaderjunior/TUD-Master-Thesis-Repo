from collections import Counter, defaultdict
from pathlib import Path
import io
import unittest
import numpy as np
from analysis.feedback.rounds import subgroups_inputs as a
from analysis.feedback.rounds import subgroups_metrics as k
from analysis.feedback.rounds import errors_kernel as rates


def freeze():
    a.need(not a.OUT.exists(), "Preserve existing Part 5 evidence")
    paths = [
        a.STUDY / "evaluation.jsonl",
        a.STUDY / "STUDY_PROTOCOL.md",
        a.STUDY / "plan.json",
        a.SCOPE / "evaluation.jsonl",
        a.SCOPE / "iteration_evaluation.json",
        a.SCOPE / "complete.json",
        a.READER / "matrix.jsonl",
        a.READER / "complete.json",
        a.READER / "independent_verification.json",
        a.PART2 / "errors_v1/results.json",
        a.PART2 / "errors_v1/independent_verification.json",
        a.PART3 / "prompt_bindings.jsonl",
        a.PART3 / "assembled.json",
        a.PART3 / "deliveries.jsonl",
        a.PART4 / "final_verification.json",
        a.PART4 / "complete.json",
        a.PART4 / "publication_v1/readback.json",
        a.PART4 / "input_trajectories.jsonl",
        a.PART4 / "analysis.json",
        a.REVIEW / "review_plan.json",
    ]
    paths += [
        a.ROOT / p
        for p in [
            "src/hsrag/data/hatexplain.py",
            "src/hsrag/data/implicit_hate.py",
            "src/hsrag/data/mhs.py",
            "data/mappings/mhs_thresholds.md",
            "scripts/make_hatecheck_subset.py",
        ]
    ]
    p4 = a.read(a.PART4 / "final_verification.json")
    a.need(
        p4["status"] == "iterative_part4_complete_with_final_readback"
        and p4["complete_sha256"] == a.sha(a.PART4 / "complete.json")
        and p4["readback_sha256"] == a.sha(a.PART4 / "publication_v1/readback.json"),
        "Part 4 checkpoint changed",
    )
    reader = a.read(a.READER / "complete.json")
    a.need(
        reader["artifacts"][a.rel(a.READER / "matrix.jsonl")]
        == a.sha(a.READER / "matrix.jsonl"),
        "Audited matrix changed",
    )
    from tests.feedback.rounds import test_subgroups as tests

    output = io.StringIO()
    passed = unittest.TextTestRunner(stream=output, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests)
    )
    a.need(passed.wasSuccessful(), "Subgroup tests failed: " + output.getvalue())
    sources = [
        Path(__file__),
        Path(a.__file__),
        Path(k.__file__),
        Path(rates.__file__),
        Path(tests.__file__),
        a.ROOT / "analysis/feedback/transfer/subgroups.py",
    ]
    contract = dict(
        status="part5_conventions_before_new_subgroup_scoring",
        created_utc=a.pure.utc(),
        timing="After Parts 2–4 results and earlier broad-transfer subgroups are known; before new iterative subgroup scoring. Supporting, not a new prospective experiment.",
        input_bindings={a.rel(p): a.sha(p) for p in paths},
        source_hashes={a.rel(p): a.sha(p) for p in sources},
        functions=a.FUNCTIONS,
        primary_unchanged="Part 2 final English four-contrast 98.75% family remains authoritative; no new subgroup primary family.",
        main="All three natural sources and all 29 HateCheck functions, both endpoints. Opposite-polarity function cells are explicitly undefined. Fixed all-round all-arm common mask for trajectories.",
        views="Fixed trajectories: all 64 existing Part 2 rate/contrast/change columns. Final common and final strict: nine final-rate/contrast columns. Per-round common masks: point tables. Annotation subtypes: final-common nominal intervals and fixed-cohort final point tables. Metadata block/identity/direction: final-common points only.",
        bootstrap=dict(
            draws=10000,
            generator="PCG64",
            level=0.95,
            seed_prefix="A17-part5-support-v1|",
            unit="Sorted inherited inference clusters within subgroup and endpoint; all eligible texts travel with cluster multiplicity. Text-weighted ratios. Repeat/stream means within input first.",
            small_cells="Fewer than ten clusters explicitly flagged descriptive_small_cluster_support. One-cluster intervals suppressed. Degenerate intervals flagged, never evidence of perfect generalisation.",
            quantile="NumPy linear percentile; independent checker uses ordered interpolation.",
        ),
        annotation="Frozen HateXplain majority label crossed with unanimity; Implicit Hate stage1 class; existing MHS hateful severity bands. Mentioned HateCheck identity is not always a hate target. Unknown fine fields stay unknown; no fine prediction scoring.",
        influence="Every function: fixed-cohort final adaptive-reference/static/random and R4-R1 adaptive. Full text-weighted point, equal-cluster point and every leave-one-inference-cluster-out point. Ranges are not confidence intervals.",
        linked_pairs="Retain every HateCheck ref_case_id edge, including missing-parent and missing-output exclusions. Joint both-cases-correct for same condition/repeat across all four rounds, grouped by child function and same/opposite recorded polarity. Dependent descriptive pairs, not new independent texts.",
        cases="For each natural source and binary class, minimum SHA256(A17-part5-case-v1|group|category|key) among fixed-cohort retained-average-benefit, final-regression, mixed-relative-direction and all-feedback-persistent-error texts. Keep empty categories. For every HateCheck function choose the greatest absolute R4 adaptive-reference difference, then greatest R4 adaptive error count, then minimum hash. These function cases are selected extremes illustrating the recorded result, not representative prevalence or causal proof.",
        preservation="No source labels, predictions, prompts, protocols, writer files or previous analysis versions changed. No generation, embeddings or fitting.",
        new_Qwen_calls=0,
        new_fits=0,
        new_embeddings=0,
    )
    a.write(a.OUT / "analysis_contract.json", contract)
    a.write(
        a.OUT / "tests.json",
        dict(passed=True, tests_run=passed.testsRun, log=output.getvalue()),
    )
    return contract


def main():
    contract = freeze()
    meta, matrix, roster = a.load()
    gates, votes, error = rates.arrays(matrix)
    counts, missing = rates.policy_counts(error)
    groups = a.definitions(meta)
    a.write(a.OUT / "groups.json", groups)
    a.lines(a.OUT / "metadata.jsonl", meta)
    broad = {r["key"]: r for r in a.rows(a.SCOPE / "evaluation.jsonl")}
    a.need(
        set(r["key"] for r in meta)
        == set(a.read(a.SCOPE / "iteration_evaluation.json")["keys"]),
        "Fixed sample differs",
    )
    for r in roster:
        other = broad[r["key"]]
        for field in [
            "gate",
            "source",
            "raw",
            "text",
            "inference_cluster",
            "functional_cluster",
            "text_cluster",
        ]:
            a.need(r[field] == other[field], "Broad-population metadata differs")
    sample_support = []
    for source in a.SOURCES + ["hatecheck"]:
        for truth in [False, True]:
            eligible = [
                r
                for r in broad.values()
                if r["source"] == source and r["gate"] == truth
            ]
            selected = [
                r for r in meta if r["source"] == source and r["gold_hate"] == truth
            ]
            sample_support.append(
                dict(
                    source=source,
                    gold_hate=truth,
                    broad_inputs=len(eligible),
                    sample_inputs=len(selected),
                    broad_clusters=len({r["inference_cluster"] for r in eligible}),
                    sample_clusters=len({r["inference_cluster"] for r in selected}),
                )
            )
    records = {}
    supports = []
    bounds = []
    fine_support = []

    def score(group, truth, view, definitions, boot):
        ids = k.cohort(meta, groups[group]["indices"], truth, view)
        endpoint = "FNR" if truth else "FPR"
        name = f"{group}|{endpoint}|{view}"
        a.need(name not in records, "Duplicate analysis record")
        rec = dict(
            group=group,
            family=groups[group]["family"],
            endpoint=endpoint,
            view=view,
            input_indices=ids,
            inputs=len(ids),
            columns=definitions,
            column_order=list(definitions),
            supporting=True,
        )
        records[name] = rec
        if not ids:
            rec.update(
                status="undefined_no_valid_class_support", points=None, intervals=None
            )
            return
        values = rates.transformed_counts(counts, ids, definitions)
        labels = sorted({meta[i]["inference_cluster"] for i in ids})
        rec.update(
            status="defined",
            clusters=len(labels),
            cluster_labels=labels,
            cluster_sizes=[
                sum(meta[i]["inference_cluster"] == c for i in ids) for c in labels
            ],
            integer_numerators=dict(zip(definitions, values.sum(axis=0).tolist())),
            denominator=9 * len(ids),
            points=dict(
                zip(definitions, (values.sum(axis=0) / (9 * len(ids))).tolist())
            ),
            intervals=None,
            qualification=(
                "descriptive_small_cluster_support"
                if len(labels) < 10
                else "nominal_supporting_not_multiplicity_adjusted"
            ),
        )
        if boot and len(labels) > 1:
            name_seed = a.seed(name)
            draws, _, _ = rates.bootstrap(
                values, [meta[i]["inference_cluster"] for i in ids], name_seed
            )
            limits = np.quantile(draws, [0.025, 0.975], axis=0)
            buffer = io.BytesIO()
            np.savez_compressed(buffer, draws=draws)
            path = a.OUT / "draws" / f"{a.seed(name):016x}.npz"
            a.write(path, buffer.getvalue())
            rec.update(
                seed=name_seed,
                draws=10000,
                confidence=0.95,
                intervals={
                    col: limits[:, j].tolist() for j, col in enumerate(definitions)
                },
                degenerate_columns=[
                    col
                    for j, col in enumerate(definitions)
                    if np.all(draws[:, j] == draws[0, j])
                ],
                draw_file=a.rel(path),
                draw_sha256=a.sha(path),
            )
        elif boot:
            rec["interval_undefined_reason"] = "Only one represented inference cluster"

    for group, g in groups.items():
        for truth in g["truths"]:
            endpoint = "FNR" if truth else "FPR"
            support = k.support(meta, g["indices"], truth)
            supports.append(
                dict(group=group, family=g["family"], endpoint=endpoint, **support)
            )
            if g["family"] in ["source", "function"]:
                score(
                    group,
                    truth,
                    "fixed",
                    rates.columns([1, 2, 3, 4], trajectories=True),
                    True,
                )
                for view in ["final", "strict"]:
                    score(group, truth, view, rates.columns([4]), True)
                for rnd in range(1, 5):
                    score(group, truth, "round" + str(rnd), rates.columns([rnd]), False)
            elif g["family"] == "annotation":
                score(group, truth, "final", rates.columns([4]), True)
                score(group, truth, "fixed", rates.columns([4]), False)
            else:
                score(group, truth, "final", rates.columns([4]), False)
            ids = [i for i in g["indices"] if meta[i]["gold_hate"] == truth]
            if g["family"] in ["source", "function"]:
                bounds.append(
                    dict(
                        group=group,
                        endpoint=endpoint,
                        values=(
                            rates.bounds(counts, missing, np.array(ids, dtype=int), 4)
                            if ids
                            else None
                        ),
                    )
                )
            for view in ["roster", "fixed", "final", "strict"]:
                selected = (
                    ids
                    if view == "roster"
                    else k.cohort(meta, g["indices"], truth, view)
                )
                fine_support.append(
                    dict(
                        group=group,
                        endpoint=endpoint,
                        view=view,
                        inputs=len(selected),
                        fields={
                            field: dict(
                                known=sum(
                                    meta[i]["fine_gold"][field] is not None
                                    for i in selected
                                ),
                                unknown=sum(
                                    meta[i]["fine_gold"][field] is None
                                    for i in selected
                                ),
                                known_empty=sum(
                                    meta[i]["fine_gold"][field] == [] for i in selected
                                ),
                                labels=dict(
                                    Counter(
                                        v
                                        for i in selected
                                        for v in (meta[i]["fine_gold"][field] or [])
                                        if isinstance(meta[i]["fine_gold"][field], list)
                                    )
                                ),
                            )
                            for field in [
                                "hate_types",
                                "target_groups",
                                "severity",
                                "legal",
                            ]
                        },
                    )
                )
        print("Subgroups scored: " + group, flush=True)

    prior = a.read(a.PART2 / "errors_v1/results.json")["records"]
    partition = []
    for pop, family in [("en_test", "source"), ("en_hatecheck_eval", "function")]:
        for ep in ["FPR", "FNR"]:
            for view, priorname in [
                ("fixed", f"trajectory_{pop}_{ep}"),
                ("final", f"round_{pop}_{ep}_r4"),
                ("strict", f"strict_{pop}_{ep}_r4"),
            ]:
                parts = [
                    r
                    for r in records.values()
                    if r["family"] == family
                    and r["endpoint"] == ep
                    and r["view"] == view
                    and r["status"] == "defined"
                ]
                p = prior[priorname]
                a.need(
                    sum(r["inputs"] for r in parts) == p["inputs"],
                    "Subgroup population denominator differs",
                )
                for col in parts[0]["columns"]:
                    total = sum(r["integer_numerators"][col] for r in parts)
                    a.need(
                        total == p["integer_error_numerators"][col],
                        "Subgroup contribution does not reconcile",
                    )
                partition.append(
                    dict(
                        population=pop,
                        endpoint=ep,
                        view=view,
                        inputs=p["inputs"],
                        columns_checked=len(parts[0]["columns"]),
                    )
                )

    profiles = {
        r["input_index"]: r for r in a.rows(a.PART4 / "input_trajectories.jsonl")
    }
    durability = []
    flags = [
        "all_saved_correct",
        "all_saved_wrong",
        "persistent_error",
        "sustained_new_error",
        "recovered_new_error",
        "delayed_clean_suffix",
    ]
    for group, g in groups.items():
        if g["family"] not in ["source", "function"]:
            continue
        for truth in g["truths"]:
            ids = k.cohort(meta, g["indices"], truth, "fixed")
            for policy in a.POLICIES[1:]:
                pp = [profiles[i]["policies"][policy]["profile"] for i in ids]
                durability.append(
                    dict(
                        group=group,
                        endpoint="FNR" if truth else "FPR",
                        policy=policy,
                        inputs=len(ids),
                        **{f: sum(p[f] for p in pp) for f in flags},
                        retained_mean_advantage=sum(
                            p["contemporaneous"]["pattern"] == "BBBB" for p in pp
                        ),
                        mixed_mean_direction=sum(
                            p["contemporaneous"]["mixed_direction"] for p in pp
                        ),
                        r4_vs_r1_improved=sum(
                            p["errors"][3] < p["errors"][0] for p in pp
                        ),
                        r4_vs_r1_worsened=sum(
                            p["errors"][3] > p["errors"][0] for p in pp
                        ),
                    )
                )
    influence = []
    templates = []
    idefs = {
        c: v
        for c, v in rates.columns([1, 4], trajectories=True).items()
        if c
        in [
            "r4_adaptive-reference",
            "r4_adaptive-static",
            "r4_adaptive-random",
            "r4-r1_adaptive",
        ]
    }
    for fn in a.FUNCTIONS:
        group = "function/" + fn
        truth = fn.endswith("_h")
        ids = k.cohort(meta, groups[group]["indices"], truth, "fixed")
        values = rates.transformed_counts(counts, ids, idefs)
        influence.append(
            dict(
                function=fn,
                columns=list(idefs),
                **k.influence(values, [meta[i]["inference_cluster"] for i in ids]),
            )
        )
        tt = defaultdict(list)
        for i in groups[group]["indices"]:
            tt[meta[i]["template_id"]].append(i)
        for templ, allids in sorted(tt.items()):
            valid = [i for i in allids if meta[i]["masks"]["all_round_common"]]
            templates.append(
                dict(
                    function=fn,
                    template=templ,
                    roster=len(allids),
                    inputs=len(valid),
                    input_indices=valid,
                    inference_clusters=sorted(
                        {meta[i]["inference_cluster"] for i in allids}
                    ),
                    error_numerators=counts[valid].sum(axis=0).tolist(),
                    denominator=9 * len(valid),
                )
            )

    caseids = {
        r["case_id"]: i for i, r in enumerate(meta) if r["source"] == "hatecheck"
    }
    edges = []
    pairs = defaultdict(list)
    for i, r in enumerate(meta):
        if r["source"] != "hatecheck" or r["ref_case_id"] is None:
            continue
        parent = caseids.get(int(r["ref_case_id"]))
        status = (
            "parent_outside_fixed_roster"
            if parent is None
            else (
                "missing_output"
                if not (
                    r["masks"]["all_round_common"]
                    and meta[parent]["masks"]["all_round_common"]
                )
                else "usable"
            )
        )
        edge = dict(
            child=i,
            child_key=r["key"],
            reference_case_id=r["ref_case_id"],
            parent=parent,
            status=status,
            function=r["functionality"],
        )
        if parent is not None:
            edge.update(
                parent_key=meta[parent]["key"],
                same_gold=r["gold_hate"] == meta[parent]["gold_hate"],
                same_inference_cluster=r["inference_cluster"]
                == meta[parent]["inference_cluster"],
            )
        if status == "usable":
            edge["joint_correct_over_nine"] = k.pair_counts(error, i, parent).tolist()
            pairs[(r["functionality"], edge["same_gold"])].append(edge)
        edges.append(edge)
    linked = [
        dict(
            function=fn,
            same_gold=same,
            edges=len(es),
            unique_texts=len({i for e in es for i in [e["child"], e["parent"]]}),
            joint_correct_numerators=np.array(
                [e["joint_correct_over_nine"] for e in es]
            )
            .sum(axis=0)
            .tolist(),
            denominator=9 * len(es),
        )
        for (fn, same), es in sorted(pairs.items())
    ]

    selection = []
    for source in a.SOURCES:
        for truth in [False, True]:
            group = "source/" + source
            ids = k.cohort(meta, groups[group]["indices"], truth, "fixed")
            ep = "FNR" if truth else "FPR"
            for cat in [
                "retained_average_benefit",
                "final_regression",
                "mixed_relative_direction",
                "persistent_error",
            ]:
                eligible = [
                    i
                    for i in ids
                    if k.selected_category(
                        error[i, :, [1, 4, 7], :].transpose(1, 0, 2).reshape(4, 9),
                        error[i, :, 0, :],
                        cat,
                    )
                ]

                chosen = (
                    min(
                        eligible,
                        key=lambda i: a.case_rank(
                            group + "/" + ep, cat, meta[i]["key"]
                        ),
                    )
                    if eligible
                    else None
                )
                selection.append(
                    dict(
                        group=group,
                        endpoint=ep,
                        category=cat,
                        candidates=len(eligible),
                        status="selected" if chosen is not None else "empty",
                        input_index=chosen,
                        key=meta[chosen]["key"] if chosen is not None else None,
                    )
                )
    for fn in a.FUNCTIONS:
        group = "function/" + fn
        truth = fn.endswith("_h")
        ids = k.cohort(meta, groups[group]["indices"], truth, "fixed")
        chosen = (
            min(
                ids,
                key=lambda i: (
                    -abs(int(counts[i, 3, 1] - counts[i, 3, 0])),
                    -int(counts[i, 3, 1]),
                    a.case_rank(group, "function_extreme", meta[i]["key"]),
                ),
            )
            if ids
            else None
        )
        selection.append(
            dict(
                group=group,
                endpoint="FNR" if truth else "FPR",
                category="function_extreme",
                candidates=len(ids),
                status="selected" if chosen is not None else "empty",
                input_index=chosen,
                key=meta[chosen]["key"] if chosen is not None else None,
            )
        )
    a.write(a.OUT / "case_selection.json", selection)
    a.lines(a.OUT / "linked_edges.jsonl", edges)
    a.write(
        a.OUT / "analysis.json",
        dict(
            status="part5_scored_pending_independent_verification",
            created_utc=a.pure.utc(),
            records=records,
            supports=supports,
            sample_support=sample_support,
            final_missing_bounds=bounds,
            fine_support=fine_support,
            durability=durability,
            function_influence=influence,
            template_points=templates,
            linked_pairs=linked,
            linked_edge_statuses=dict(Counter(e["status"] for e in edges)),
            partition_checks=partition,
            evaluation_inputs=len(meta),
            evaluation_nulls=int((gates < 0).sum()),
            fixed_inputs=sum(r["masks"]["all_round_common"] for r in meta),
            case_slots=len(selection),
            selected_slots=sum(s["status"] == "selected" for s in selection),
            unique_case_inputs=len(
                {s["input_index"] for s in selection if s["status"] == "selected"}
            ),
            new_Qwen_calls=0,
            new_embeddings=0,
            new_fits=0,
        ),
    )
    files = [
        "analysis_contract.json",
        "tests.json",
        "groups.json",
        "metadata.jsonl",
        "case_selection.json",
        "linked_edges.jsonl",
        "analysis.json",
    ]
    a.write(
        a.OUT / "assembled.json",
        dict(
            status="part5_analysis_assembled_awaiting_independent_check",
            created_utc=a.pure.utc(),
            input_bindings=contract["input_bindings"],
            artifacts={a.rel(a.OUT / f): a.sha(a.OUT / f) for f in files},
            new_Qwen_calls=0,
        ),
    )
    print(
        f"Part 5 assembled: {len(records)} views, {len(selection)} declared slots, independent verification next.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()

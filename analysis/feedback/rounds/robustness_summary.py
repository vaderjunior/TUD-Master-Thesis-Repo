from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from analysis.feedback.rounds import robustness_inputs as a
from analysis.feedback.rounds import robustness_metrics as k
from analysis.feedback.rounds import errors_kernel as old


def main():
    assembled = a.read(a.OUT / "assembled.json")
    a.check(assembled["input_bindings"])
    a.check(assembled["source_hashes"])
    a.check(assembled["artifacts"])
    matrix = list(a.rows(a.READER / "matrix.jsonl"))
    meta = list(a.rows(a.OUT / "metadata.jsonl"))
    gates, votes, error = old.arrays(matrix)
    ex = dict(np.load(a.OUT / "execution_arrays.npz", allow_pickle=False))
    masks = k.masks(meta, gates, votes, ex)
    counts = k.collapse(np.maximum(error, 0))
    missing = k.collapse(error < 0)
    columns = k.COLUMNS
    prior = a.read(a.PART2 / "errors_v1/results.json")
    records = {}
    drawfiles = []
    cohort_rows = []
    omission_rows = []
    variation = []
    reference = []
    bounds = []
    composition = []
    batch_points = []
    gold = np.array([m["gold_hate"] for m in meta])
    completion_low, completion_high = k.majority_completion(ex["vote_gates"], gold)
    vote_lo = k.collapse(completion_low)
    vote_hi = k.collapse(completion_high)
    roster = list(a.rows(a.STUDY / "evaluation.jsonl"))
    functions = {r["key"]: r.get("functionality") for r in roster}

    from analysis.feedback.rounds.subgroups_inputs import annotation

    functions = {r["key"]: annotation(r).get("functionality") for r in roster}

    def add(
        pop, endpoint, view, ids, mode="text", cluster="inference_cluster", only=None
    ):
        name = f"{pop}|{endpoint}|{view}|{mode}|{cluster}"
        ids = np.asarray(ids, dtype=np.int64)
        selected = columns if only is None else only
        rec = dict(
            population=pop,
            endpoint=endpoint,
            view=view,
            weighting=mode,
            bootstrap_unit=cluster,
            indices=ids,
            n=len(ids),
            columns=selected,
        )
        if not len(ids):
            rec.update(status="undefined_empty", points=None, intervals=None)
            records[name] = rec
            return
        vals = k.contrasts(counts[ids, 3])[:, [columns.index(c) for c in selected]]
        labels = [str(i) if cluster == "row" else meta[i][cluster] for i in ids]
        inherited = None
        if (
            view in ["final", "strict", "fixed"]
            and mode == "text"
            and cluster == "inference_cluster"
        ):
            oldname = (
                f"round_{pop}_{endpoint}_r4"
                if view == "final"
                else (
                    f"strict_{pop}_{endpoint}_r4"
                    if view == "strict"
                    else f"trajectory_{pop}_{endpoint}"
                )
            )
            inherited = prior["records"][oldname]
            a.need(inherited["input_indices"] == ids.tolist(), "Inherited mask differs")
            saved = np.load(a.ROOT / inherited["draw_file"], allow_pickle=False)
            draws = saved[
                :, [inherited["column_order"].index("r4_" + c) for c in selected]
            ]
            point = np.array([inherited["points"]["r4_" + c] for c in selected])
            names = inherited["cluster_labels"]
            sizes = np.array(inherited["cluster_sizes"])
            seed = inherited["seed"]
            rec.update(
                inherited_record=oldname,
                draw_file=inherited["draw_file"],
                draw_sha256=inherited["draw_sha256"],
                draw_columns=[
                    inherited["column_order"].index("r4_" + c) for c in selected
                ],
                draw_format="npy",
            )
        else:
            seed = a.seed(name)
            point, draws, names, sizes = k.bootstrap(vals, labels, seed, mode)
            path = a.OUT / "draws" / (f"{seed:016x}.npz")
            a.npz(path, draws=draws)
            drawfiles.append(path)
            rec.update(
                draw_file=a.rel(path), draw_sha256=a.sha(path), draw_format="npz"
            )
        ci = np.quantile(draws, [0.025, 0.975], axis=0, method="linear").T
        family = (
            np.quantile(draws, [0.00625, 0.99375], axis=0, method="linear").T
            if pop == "en_test"
            else None
        )
        rec.update(
            status="defined",
            points={c: point[j] for j, c in enumerate(selected)},
            intervals={c: ci[j] for j, c in enumerate(selected)},
            family_level_sensitivity=(
                {
                    c: family[j]
                    for j, c in enumerate(selected)
                    if c in ["adaptive-static", "adaptive-random"]
                }
                if family is not None
                else None
            ),
            seed=seed,
            clusters=len(names),
            cluster_labels=names,
            cluster_sizes=sizes,
            integer_numerators=vals.sum(axis=0) if mode == "text" else None,
            denominator=9 * len(ids) if mode == "text" else None,
            source_counts=Counter(meta[i]["source"] for i in ids),
            interval_level=0.95,
            cluster_mass_effective_n=float(sizes.sum() ** 2 / (sizes * sizes).sum()),
            largest_cluster=int(sizes.max()),
            largest_cluster_fraction=float(sizes.max() / sizes.sum()),
        )
        records[name] = rec

    for pop in a.POPS:
        for endpoint in ["FPR", "FNR"]:
            eligible = np.array(
                [
                    m["population"] == pop and m["gold_hate"] == (endpoint == "FNR")
                    for m in meta
                ]
            )
            allids = np.flatnonzero(eligible)
            for view, mask in masks.items():
                ids = np.flatnonzero(eligible & mask)
                selected = (
                    [
                        "adaptive",
                        "static",
                        "random",
                        "adaptive-static",
                        "adaptive-random",
                    ]
                    if view == "policy_only"
                    else (
                        [
                            "adaptive",
                            view.removeprefix("paired_"),
                            "adaptive-" + view.removeprefix("paired_"),
                        ]
                        if view.startswith("paired_")
                        else None
                    )
                )
                add(pop, endpoint, view, ids, only=selected)
                cohort_rows.append(
                    dict(
                        population=pop,
                        endpoint=endpoint,
                        view=view,
                        eligible=len(allids),
                        retained=len(ids),
                        excluded=len(allids) - len(ids),
                        source_counts=Counter(meta[i]["source"] for i in ids),
                    )
                )
            for view in ["final", "strict"]:
                ids = np.flatnonzero(eligible & masks[view])
                add(pop, endpoint, view, ids, "cluster_equal")
                add(pop, endpoint, view, ids, cluster="row")
                vals = k.contrasts(counts[ids, 3])
                mainpoint = vals.sum(axis=0) / (9 * len(ids))
                groupings = {
                    "inference_cluster": [meta[i]["inference_cluster"] for i in ids],
                    "source": [meta[i]["source"] for i in ids],
                    "batch": [
                        f'evaluation_{int(ex["stage"][i,3,0,0]):02}' for i in ids
                    ],
                }
                if pop == "en_hatecheck_eval":
                    groupings["function"] = [functions[meta[i]["key"]] for i in ids]
                for grouping, labels in groupings.items():
                    for r in k.omissions(vals, labels):
                        omission_rows.append(
                            dict(
                                population=pop,
                                endpoint=endpoint,
                                view=view,
                                grouping=grouping,
                                columns=columns,
                                full_points=mainpoint,
                                **r,
                            )
                        )
                for stage in sorted(set(ex["stage"][ids, 3, 0, 0])):
                    sub = ids[ex["stage"][ids, 3, 0, 0] == stage]
                    v = k.contrasts(counts[sub, 3])
                    batch_points.append(
                        dict(
                            population=pop,
                            endpoint=endpoint,
                            view=view,
                            stage=int(stage),
                            inputs=len(sub),
                            points={
                                c: float(v[:, j].sum() / (9 * len(sub)))
                                for j, c in enumerate(columns)
                            },
                        )
                    )
            for rnd in range(4):
                b = k.opposing(
                    counts[allids, rnd], counts[allids, rnd] + missing[allids, rnd]
                )
                oldb = prior["full_roster_missing_bounds"][f"{pop}_{endpoint}_r{rnd+1}"]
                for c in k.CONTRASTS:
                    a.need(
                        np.array_equal(b["intervals"][c], oldb["contrasts"][c]),
                        "Inherited null bounds differ",
                    )
                vb = k.opposing(vote_lo[allids, rnd], vote_hi[allids, rnd])
                bounds.append(
                    dict(
                        population=pop,
                        endpoint=endpoint,
                        round=rnd + 1,
                        aggregate_null=b,
                        vote_completion=vb,
                        outputs=len(allids) * 30,
                        null_outputs=int((gates[allids, rnd] < 0).sum()),
                        partial_outputs=int(
                            ((votes[allids, rnd] > 0) & (votes[allids, rnd] < 3)).sum()
                        ),
                        variable_under_vote_completion=int(
                            (
                                completion_low[allids, rnd]
                                != completion_high[allids, rnd]
                            ).sum()
                        ),
                    )
                )
            fixed = np.flatnonzero(eligible & masks["fixed"])
            raw = error[fixed, 3]
            a.need((raw >= 0).all(), "Invalid variation cohort")
            for omit_kind in ["stream", "repeat"]:
                for omit in range(3):
                    policies = {}
                    for p, pol in enumerate(k.POLICIES):
                        cs = (
                            [0]
                            if p == 0
                            else [
                                p + 3 * j
                                for j in range(3)
                                if omit_kind != "stream" or j != omit
                            ]
                        )
                        ds = [j for j in range(3) if omit_kind != "repeat" or j != omit]
                        policies[pol] = float(raw[:, cs][:, :, ds].mean())
                    variation.append(
                        dict(
                            population=pop,
                            endpoint=endpoint,
                            omitted_kind=omit_kind,
                            omitted_index=omit,
                            n=len(fixed),
                            rates=policies,
                            contrasts={
                                c: policies["adaptive"] - policies[c.split("-")[1]]
                                for c in k.CONTRASTS
                            },
                        )
                    )
            for stream in range(3):
                for repeat in range(3):
                    rates = {
                        p: float(raw[:, 0 if j == 0 else j + 3 * stream, repeat].mean())
                        for j, p in enumerate(k.POLICIES)
                    }
                    variation.append(
                        dict(
                            population=pop,
                            endpoint=endpoint,
                            omitted_kind="single_cell",
                            stream=stream,
                            repeat=repeat,
                            n=len(fixed),
                            rates=rates,
                            contrasts={
                                c: rates["adaptive"] - rates[c.split("-")[1]]
                                for c in k.CONTRASTS
                            },
                        )
                    )
            trajectory = prior["records"][f"trajectory_{pop}_{endpoint}"]
            for rnd in range(1, 5):
                rr = counts[fixed, rnd - 1].mean(axis=0) / 9
                first = counts[fixed, 0, 0].mean() / 9
                reference.append(
                    dict(
                        population=pop,
                        endpoint=endpoint,
                        round=rnd,
                        n=len(fixed),
                        reference_rate=rr[0],
                        adaptive_rate=rr[1],
                        contemporary_difference=rr[1] - rr[0],
                        fixed_first_reference_difference=rr[1] - first,
                        reference_drift=rr[0] - first,
                        trajectory_record=f"trajectory_{pop}_{endpoint}",
                        r4_minus_r1_absolute=trajectory["points"]["r4-r1_adaptive"],
                        r4_minus_r1_adjusted=trajectory["points"][
                            "r4-r1_adaptive-reference"
                        ],
                    )
                )
            for view in ["strict", "no_ties", "no_repairs", "no_transport_retry"]:
                for keep in [True, False]:
                    ids = np.flatnonzero(
                        eligible
                        & masks["final"]
                        & (masks[view] if keep else ~masks[view])
                    )
                    vals = k.contrasts(counts[ids, 3])
                    composition.append(
                        dict(
                            population=pop,
                            endpoint=endpoint,
                            filter=view,
                            kept=keep,
                            n=len(ids),
                            source_counts=Counter(meta[i]["source"] for i in ids),
                            points=(
                                {
                                    c: float(vals[:, j].sum() / (9 * len(ids)))
                                    for j, c in enumerate(columns)
                                }
                                if len(ids)
                                else None
                            ),
                        )
                    )
            print("Robustness scored: " + pop + "/" + endpoint, flush=True)

    primary = {
        name: r
        for name, r in prior["records"].items()
        if r["family"] == "four_contrast_primary"
    }
    a.need(len(primary) == 4, "Primary family differs")
    primary_checks = []
    for name, r in primary.items():
        source = a.ROOT / r["draw_file"]
        a.need(a.sha(source) == r["draw_sha256"], "Primary draws changed")
        primary_checks.append(
            dict(
                name=name,
                inputs=r["inputs"],
                clusters=r["inference_clusters"],
                points=r["points"],
                intervals=r["intervals"],
                interval_level=r["interval_level"],
                seed=r["seed"],
                draw_file=r["draw_file"],
                draw_sha256=r["draw_sha256"],
            )
        )

    quality = a.read(a.PART6 / "analysis.json")["records"]
    quality_views = []
    for name, r in quality.items():
        if r["kind"] == "population" and r["view"] in [
            "final",
            "strict",
            "fixed",
            "replay_excluded",
            "representative",
        ]:
            quality_views.append(
                dict(
                    name=name,
                    head=r["head"],
                    population=r["population"],
                    view=r["view"],
                    status=r["status"],
                    n=r["n"],
                    macro_labels=r["macro_labels"],
                    metrics=r.get("metrics"),
                    column_order=r.get("column_order"),
                    estimates=r.get("estimates"),
                    intervals=r.get("intervals"),
                    all_taxonomy_macro_descriptive=r.get(
                        "all_taxonomy_macro_descriptive"
                    ),
                )
            )
    source_cluster = a.read(a.PART3 / "source_cluster_inventory.json")
    selection = a.read(a.PART3 / "analysis.json")["selection"]
    dependency = dict(
        source_inventory=source_cluster,
        selection_summary={
            key: selection[key]
            for key in [
                "candidate_opportunities",
                "common_eligible_opportunities",
                "unique_candidates",
                "unique_selected_ids",
                "selected_addition_memberships",
                "cumulative_memberships",
                "distinct_cumulative_sets",
            ]
        },
        source_overlap="The three streams reuse the same candidate universe. Repeats and shared-reference contrasts are dependent; fixed-stream intervals do not cover new feedback pools or independent annotators.",
        same_prompt_control=a.read(a.PART2 / "tables_v1/control_diagnostics.json"),
        annotation_boundaries=a.read(
            a.REVIEW / "part5/annotation_boundary_review.json"
        ),
    )
    a.write(
        a.OUT / "analysis.json",
        dict(
            status="part7_added_sensitivities_scored",
            created_utc=a.pure.utc(),
            records=records,
            primary_unchanged=primary_checks,
            cohorts=cohort_rows,
            bounds=bounds,
            filter_composition=composition,
            stream_repeat_variation=variation,
            reference_comparisons=reference,
            batch_points=batch_points,
            quality_views=quality_views,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    a.lines(a.OUT / "omissions.jsonl", omission_rows)
    a.write(a.OUT / "dependence.json", dependency)
    a.write(
        a.OUT / "scored.json",
        dict(
            status="part7_scored_pending_independent_check",
            created_utc=a.pure.utc(),
            input_bindings=a.bindings(
                [a.OUT / "assembled.json", a.OUT / "analysis_contract.json"]
            ),
            source_hashes=a.bindings([Path(__file__), Path(k.__file__)]),
            artifacts=a.bindings(
                [
                    a.OUT / "analysis.json",
                    a.OUT / "omissions.jsonl",
                    a.OUT / "dependence.json",
                    *drawfiles,
                ]
            ),
            new_draw_sets=len(drawfiles),
            records=len(records),
            omissions=len(omission_rows),
            new_Qwen_calls=0,
        ),
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()

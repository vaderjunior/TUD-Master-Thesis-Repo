from pathlib import Path
import numpy as np
from analysis.feedback.rounds import resources_inputs as a
from analysis.feedback.rounds import resources_metrics as k


def main():
    assembled = a.read(a.OUT / "assembled.json")
    a.check(assembled["artifacts"])
    a.check(assembled["source_hashes"])
    meta = list(a.rows(a.REVIEW / "part7/metadata.jsonl"))
    groups = a.read(a.OUT / "resource_groups.json")
    with np.load(a.OUT / "arrays.npz", allow_pickle=False) as saved:
        x = saved["resources"]
        local = saved["local_prompt_tokens"]
    records = {}
    a.OUT.joinpath("draws").mkdir()
    for rr in range(4):
        features = k.paired(x[:, rr])
        for pop in a.POPS:
            for mask in ["all_roster", "round_common", "all_round_common"]:
                ids = [
                    i
                    for i, m in enumerate(meta)
                    if m["population"] == pop
                    and (
                        mask == "all_roster"
                        or (
                            m["masks"]["round_common"][rr]
                            if mask == "round_common"
                            else m["masks"]["all_round_common"]
                        )
                    )
                ]
                clusters = [meta[i]["inference_cluster"] for i in ids]
                name = f"{pop}|r{rr+1}|{mask}"
                seed = k.seed(name)
                draw = k.bootstrap(features[ids], clusters, seed)
                p = a.OUT / "draws" / (name.replace("|", "__") + ".npz")
                with p.open("xb") as f:
                    np.savez_compressed(f, draws=draw)
                records[name] = dict(
                    population=pop,
                    round=rr + 1,
                    mask=mask,
                    n=len(ids),
                    indices=ids,
                    clusters=len(set(clusters)),
                    cluster_labels=sorted(set(clusters)),
                    metrics=k.METRICS,
                    columns=k.COLUMNS,
                    points=features[ids].mean(axis=0),
                    intervals=np.quantile(draw, [0.025, 0.975], axis=0).transpose(
                        1, 2, 0
                    ),
                    interval_level=0.95,
                    draws=10000,
                    seed=seed,
                    draw_file=a.rel(p),
                    draw_sha256=a.sha(p),
                )
                print("Paired costs " + name, flush=True)
    source_paths = [
        a.REVIEW / "part2/errors_v1/results.json",
        a.REVIEW / "part3/analysis.json",
        a.REVIEW / "part3/selections.json",
        a.REVIEW / "part6/analysis.json",
    ]
    scientific, context, selections, quality = [a.read(p) for p in source_paths]
    curves = []
    quality_curves = []
    for pop in a.POPS:
        for end in ["FPR", "FNR"]:
            r = scientific["records"][f"trajectory_{pop}_{end}"]
            a.need(r["mask"] == "all_round_common", "Trajectory mask differs")
            for rr in range(1, 5):
                columns = [
                    f"r{rr}_{p}" for p in ["reference", "adaptive", "static", "random"]
                ] + [f"r{rr}_adaptive-{p}" for p in ["static", "random", "reference"]]
                curves.append(
                    dict(
                        population=pop,
                        endpoint=end,
                        round=rr,
                        n=r["inputs"],
                        clusters=r["inference_clusters"],
                        source_record=f"trajectory_{pop}_{end}",
                        points={col: r["points"][col] for col in columns},
                        intervals={col: r["intervals"][col] for col in columns},
                        supporting_level=0.95,
                        r4_r1_reference_adjusted=dict(
                            point=r["points"]["r4-r1_adaptive-reference"],
                            interval=r["intervals"]["r4-r1_adaptive-reference"],
                        ),
                    )
                )
        q = quality["records"][f"population/{pop}|known_answer|fixed"]
        metric = q["metrics"].index("known_answer_exact")
        for rr in range(1, 5):
            columns = [
                f"r{rr}_{p}" for p in ["reference", "adaptive", "static", "random"]
            ] + [f"r{rr}_adaptive-reference"]
            quality_curves.append(
                dict(
                    population=pop,
                    round=rr,
                    n=q["n"],
                    head="known_answer",
                    mask="fixed",
                    points={
                        col: q["estimates"][q["column_order"].index(col)][metric]
                        for col in columns
                    },
                    intervals={
                        col: q["intervals"][q["column_order"].index(col)][metric]
                        for col in columns
                    },
                    supporting_level=0.95,
                )
            )
    increments = []
    cumulative_tokens = cumulative_calls = cumulative_outputs = 0
    for rr in range(1, 5):
        source_names = (
            ["source_initial"] if rr == 1 else [f"r{rr}_source_s{s}" for s in a.STREAMS]
        )
        screen = [groups["phase/" + name] for name in source_names]
        ev = groups[f"phase/r{rr}_evaluation"]
        screen_tokens = sum(g["observed_usage"]["total_tokens"] for g in screen)
        screen_calls = sum(g["accounting"]["ordinary_logical_calls"] for g in screen)
        total = screen_tokens + ev["observed_usage"]["total_tokens"]
        calls = screen_calls + ev["accounting"]["ordinary_logical_calls"]
        outputs = (
            sum(g["counts"]["predictions"] for g in screen)
            + ev["counts"]["predictions"]
        )
        cumulative_tokens += total
        cumulative_calls += calls
        cumulative_outputs += outputs
        selected = [s for s in selections if s["round"] == rr]
        unique_ids = set(
            i for s in selected for ids in s["additions"].values() for i in ids
        )
        exposure = [r for r in context["exposure"] if r["round"] == rr]
        transition = [r for r in context["delivery_transitions"] if r["round"] == rr]
        increments.append(
            dict(
                round=rr,
                retained_per_pool=10 * rr,
                screening_phases=source_names,
                screening_outputs=sum(g["counts"]["predictions"] for g in screen),
                screening_tokens=screen_tokens,
                evaluation_outputs=ev["counts"]["predictions"],
                evaluation_tokens=ev["observed_usage"]["total_tokens"],
                round_outputs=outputs,
                round_tokens=total,
                ordinary_calls=calls,
                cumulative_outputs=cumulative_outputs,
                cumulative_tokens=cumulative_tokens,
                cumulative_ordinary_calls=cumulative_calls,
                selected_memberships=sum(
                    len(ids) for s in selected for ids in s["additions"].values()
                ),
                unique_selected_this_round=len(unique_ids),
                context_assignments=sum(r["assignments"] for r in exposure),
                newest_correction_assignments=sum(
                    r["newest_available_round"] for r in exposure
                ),
                same_source_as_previous=(
                    sum(r["counts"]["same_source"] for r in transition)
                    if transition
                    else None
                ),
                switch_to_new=(
                    sum(r["counts"]["switch_to_new"] for r in transition)
                    if transition
                    else None
                ),
                switch_to_old=(
                    sum(r["counts"]["switch_to_old"] for r in transition)
                    if transition
                    else None
                ),
                interpretation="Initial screening for all 1024 candidates is charged upfront in round 1; later rows contain the three new adaptive screening blocks. These are research expenses, not per-user serving prices.",
            )
        )
    a.need(
        cumulative_tokens == groups["all"]["observed_usage"]["total_tokens"]
        and cumulative_outputs == 259224
        and cumulative_calls == 777672,
        "Marginal/cumulative expense does not reconcile",
    )
    local_means = []
    for pop in a.POPS:
        ids = [i for i, m in enumerate(meta) if m["population"] == pop]
        for rr in range(4):
            for policy, cs in [
                ("reference", [0]),
                ("adaptive", [1, 4, 7]),
                ("static", [2, 5, 8]),
                ("random", [3, 6, 9]),
            ]:
                values = local[ids, rr][:, cs]
                local_means.append(
                    dict(
                        population=pop,
                        round=rr + 1,
                        policy=policy,
                        prompts=values.size,
                        sum=int(values.sum()),
                        mean=float(values.mean()),
                        minimum=int(values.min()),
                        maximum=int(values.max()),
                    )
                )
    a.write(
        a.OUT / "analysis.json",
        dict(
            status="part8_supporting_costs_and_marginal_value_scored",
            created_utc=a.pure.utc(),
            source_sha256=a.sha(Path(__file__)),
            records=records,
            resource_metrics=k.METRICS,
            resource_columns=k.COLUMNS,
            round_increments=increments,
            fixed_error_curves=curves,
            fixed_quality_curves=quality_curves,
            local_prompt_means=local_means,
            original_primary=[
                {
                    field: r[field]
                    for field in [
                        "name",
                        "inputs",
                        "inference_clusters",
                        "points",
                        "intervals",
                        "seed",
                        "interval_level",
                        "draw_file",
                        "draw_sha256",
                    ]
                }
                for name, r in scientific["records"].items()
                if name.startswith("primary_")
            ],
            qualifiers=[
                "Cost intervals are supporting 95% and condition on these fixed texts, pools and saved outputs; no equivalence claim.",
                "No round was selected or stopped using evaluation outcomes. More retained corrections does not mean more shown per prompt.",
                "Cost per improvement is not reported when improvements are uncertain or depend on cohort/endpoint. False alarms and misses have different denominators.",
                "Research-matrix totals and illustrative serving means must not be conflated. No price, energy, GPU-hour or deployment-prevalence assumption.",
            ],
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    a.write(
        a.OUT / "scored.json",
        dict(
            status="part8_added_resource_analysis_complete",
            created_utc=a.pure.utc(),
            inputs=a.bindings(
                [
                    a.OUT / "assembled.json",
                    a.OUT / "arrays.npz",
                    a.OUT / "resource_groups.json",
                    *source_paths,
                ]
            ),
            source_hashes=a.bindings([Path(__file__), Path(k.__file__)]),
            artifacts=a.bindings(
                [a.OUT / "analysis.json", *sorted((a.OUT / "draws").glob("*.npz"))]
            ),
            resource_records=len(records),
            exact_resource_points=len(records) * 56,
            bootstrap_scalars=len(records) * 560000,
            new_Qwen_calls=0,
            new_fits=0,
        ),
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()

import hashlib
from collections import Counter
from datetime import datetime, timedelta
from analysis.feedback.classifiers import common as a
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.feedback.classifiers import kernel as k
from analysis.feedback.classifiers.scores import seed


def quantiles(value):
    x = np.asarray(value).ravel()
    return (
        dict(
            n=len(x),
            quantiles=dict(
                zip(
                    ["min", "p01", "p05", "p25", "median", "p75", "p95", "p99", "max"],
                    np.quantile(x, [0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1]),
                )
            ),
            mean=float(x.mean()),
        )
        if len(x)
        else None
    )


def main():
    a.contract()
    a.c.verify(a.read(a.OUT / "scored.json")["outputs"])
    z = a.load("predictions.npz")
    saved = a.load("saved_model_replay.npz")
    meta = list(a.rows(a.OUT / "metadata.jsonl"))
    execution = a.read(a.OUT / "execution.json")
    aliases = execution["aliases"]
    data = a.read(a.BASE / "data.json")
    selections = a.read(a.REVIEW / "part3/selections.json")
    pop = np.array([m["population"] for m in meta])
    gold = z["gold"]
    fixed = (z["qwen_gates"] >= 0).all((1, 2, 3))
    final = (z["qwen_gates"][:, 3] >= 0).all((1, 2))
    raw = k.raw_counts(z["linear_errors"], np.maximum(z["qwen_errors"], 0))
    source_intervals = []
    extra_draws = []
    for source in sorted({m["source"] for m in meta if m["population"] == "en_test"}):
        for endpoint in ["FPR", "FNR"]:
            ids = np.flatnonzero(
                (pop == "en_test")
                & (gold == (endpoint == "FNR"))
                & final
                & np.array([m["source"] == source for m in meta])
            )
            if not len(ids):
                continue
            name = f"source_{source}__en_test__{endpoint}"
            s = seed(name)
            point, draws, names, sizes = k.bootstrap(
                raw[ids, 3], [meta[i]["inference_cluster"] for i in ids], s
            )
            p, ci = k.intervals(point[None], draws[:, None])
            file = f"draws/{name}.npz"
            a.arrays(file, raw_point=point[None], raw_draws=draws[:, None])
            extra_draws.append(a.OUT / file)
            source_intervals.append(
                dict(
                    id=name,
                    cohort="source_" + source,
                    population="en_test",
                    endpoint=endpoint,
                    rounds=[4],
                    mode="text",
                    indices=ids,
                    inputs=len(ids),
                    clusters=len(names),
                    cluster_names=names,
                    cluster_sizes=sizes,
                    seed=s,
                    draw_file=file,
                    point=p,
                    ci95=ci,
                )
            )
    probability = []
    paths = []
    transitions = []
    quality_filter = []
    for s, (variant, cutoff) in enumerate(a.SETTINGS):
        v = a.VARIANTS.index(variant)
        base = z["probabilities"][:, v, 0, 0]
        for population in sorted(set(pop)):
            for endpoint in ["FPR", "FNR"]:
                eligible = (pop == population) & (gold == (endpoint == "FNR"))
                for maskname, mask in [
                    ("fixed", fixed),
                    ("full_sample", np.ones(len(meta), bool)),
                ]:
                    ids = np.flatnonzero(eligible & mask)
                    for p, policy in enumerate(a.POLICIES, 1):
                        cols = [p, p + 3, p + 6]
                        err = z["linear_errors"][ids, s][:, :, cols]
                        baseline = z["linear_errors"][ids, s, 0, 0]
                        paths_array = (
                            np.concatenate(
                                [
                                    np.broadcast_to(
                                        baseline[:, None, None], (len(ids), 1, 3)
                                    ),
                                    err,
                                ],
                                axis=1,
                            )
                            .transpose(0, 2, 1)
                            .reshape(-1, 5)
                        )
                        paths.append(
                            dict(
                                setting=s,
                                population=population,
                                endpoint=endpoint,
                                cohort=maskname,
                                policy=policy,
                                inputs=len(ids),
                                **k.path_counts(paths_array),
                                pattern_counts=dict(
                                    sorted(
                                        Counter(
                                            "".join(str(int(t)) for t in row)
                                            for row in paths_array
                                        ).items()
                                    )
                                ),
                            )
                        )
                        for r in range(4):
                            prob = z["probabilities"][ids, v, r][:, cols]
                            shift = prob - base[ids, None]
                            before = (
                                base[ids, None]
                                if r == 0
                                else z["probabilities"][ids, v, r - 1][:, cols]
                            )
                            prev = np.broadcast_to(before > cutoff, prob.shape)
                            now = prob > cutoff
                            truth = gold[ids, None]
                            transitions.append(
                                dict(
                                    setting=s,
                                    population=population,
                                    endpoint=endpoint,
                                    cohort=maskname,
                                    policy=policy,
                                    round=r + 1,
                                    inputs=len(ids),
                                    decisions=len(ids) * 3,
                                    fixes=int(((prev != truth) & (now == truth)).sum()),
                                    new_errors=int(
                                        ((prev == truth) & (now != truth)).sum()
                                    ),
                                    per_stream_fixes=(
                                        (prev != truth) & (now == truth)
                                    ).sum(0),
                                    per_stream_new_errors=(
                                        (prev == truth) & (now != truth)
                                    ).sum(0),
                                )
                            )
                            probability.append(
                                dict(
                                    setting=s,
                                    population=population,
                                    endpoint=endpoint,
                                    cohort=maskname,
                                    policy=policy,
                                    round=r + 1,
                                    shift_from_baseline=quantiles(shift),
                                    absolute_shift=quantiles(abs(shift)),
                                    changed_decisions=int(
                                        (
                                            (prob > cutoff)
                                            != (base[ids, None] > cutoff)
                                        ).sum()
                                    ),
                                    unique_changed_texts=int(
                                        ((prob > cutoff) != (base[ids, None] > cutoff))
                                        .any(1)
                                        .sum()
                                    ),
                                    baseline_distance_to_cutoff=quantiles(
                                        abs(base[ids] - cutoff)
                                    ),
                                )
                            )
                for name, mask in [
                    ("final_common", final),
                    ("excluded_final", ~final),
                    ("fixed", fixed),
                    ("excluded_fixed", ~fixed),
                    ("final_strict", (z["qwen_votes"][:, 3] == 3).all((1, 2))),
                    ("excluded_strict", (z["qwen_votes"][:, 3] != 3).any((1, 2))),
                ]:
                    ids = np.flatnonzero(eligible & mask)
                    if not len(ids):
                        continue
                    quality_filter.append(
                        dict(
                            setting=s,
                            population=population,
                            endpoint=endpoint,
                            cohort=name,
                            inputs=len(ids),
                            baseline_error=float(
                                z["linear_errors"][ids, s, 0, 0].mean()
                            ),
                            current_error=float(
                                z["linear_errors"][ids, s, 3][:, [1, 4, 7]].mean()
                            ),
                        )
                    )

    changed = []
    for i, m in enumerate(meta):
        base = z["linear_gates"][i, :, 0, 0]
        if (z["linear_gates"][i] != base[:, None, None]).any():
            changed.append(
                dict(
                    **m,
                    probabilities=z["probabilities"][i],
                    linear_decisions=z["linear_gates"][i],
                    qwen_decisions=z["qwen_gates"][i],
                )
            )
    candidate_index = {x: i for i, x in enumerate(data["candidate_ids"])}
    source_gold = np.asarray(data["candidate_labels"], dtype=bool)
    source = saved["candidate_probabilities"]
    adoption = []
    source_ledger = []
    for selection in selections:
        r = selection["round"]
        stream = selection["stream"]
        for p in a.POLICIES:
            for s, (variant, cutoff) in enumerate(a.SETTINGS):
                baseline = aliases[variant + "/reference"]
                current = aliases[f"{variant}/r{r}_{p}_s{stream}"]
                previous = (
                    baseline if r == 1 else aliases[f"{variant}/r{r-1}_{p}_s{stream}"]
                )
                for pool, source_ids in [
                    ("additions", selection["additions"][p]),
                    ("cumulative", selection["cumulative"][p]),
                ]:
                    ix = np.array([candidate_index[x] for x in source_ids])
                    yb = source_gold[ix]
                    pb = source[baseline, ix]
                    pp = source[previous, ix]
                    pu = source[current, ix]
                    eb = (pb > cutoff) != yb
                    ep = (pp > cutoff) != yb
                    eu = (pu > cutoff) != yb
                    adoption.append(
                        dict(
                            setting=s,
                            round=r,
                            stream=stream,
                            policy=p,
                            pool=pool,
                            records=len(ix),
                            baseline_errors=int(eb.sum()),
                            before_errors=int(ep.sum()),
                            after_errors=int(eu.sum()),
                            fixes=int((ep & ~eu).sum()),
                            new_errors=int((~ep & eu).sum()),
                            mean_true_class_probability_change=float(
                                np.mean(np.where(yb, pu - pp, pp - pu))
                            ),
                            scope="Selected TRAIN-added texts; in-sample diagnostic, never transfer evidence.",
                        )
                    )
                    if pool == "additions":
                        for j, sid in enumerate(source_ids):
                            source_ledger.append(
                                dict(
                                    setting=s,
                                    round=r,
                                    stream=stream,
                                    policy=p,
                                    source_id=sid,
                                    gold=bool(yb[j]),
                                    baseline_probability=pb[j],
                                    before_probability=pp[j],
                                    after_probability=pu[j],
                                    baseline_error=bool(eb[j]),
                                    before_error=bool(ep[j]),
                                    after_error=bool(eu[j]),
                                )
                            )
    full_roster = list(a.rows(a.BASE / "roster.jsonl"))
    full_pop = np.array([x["population"] for x in full_roster])
    full_gold = np.array([x["gate"] for x in full_roster], bool)
    full = []
    model_motion = []
    for job in execution["jobs"]:
        if not job["new"]:
            continue
        j = job["index"]
        v = job["variant"]
        b = aliases[v + "/reference"]
        delta = saved["full_probabilities"][j] - saved["full_probabilities"][b]
        model_motion.append(
            dict(
                job_index=j,
                source_count=len(job["source_ids"]),
                variant=v,
                coef_l2_change=float(
                    np.linalg.norm(saved["coef"][j] - saved["coef"][b])
                ),
                baseline_coef_l2=float(np.linalg.norm(saved["coef"][b])),
                intercept_change=float(saved["intercept"][j] - saved["intercept"][b]),
                absolute_probability_shift=quantiles(abs(delta)),
            )
        )
        for cutoff in [0.6, 0.5]:
            for population in sorted(set(full_pop)):
                for endpoint in ["FPR", "FNR"]:
                    ix = np.flatnonzero(
                        (full_pop == population) & (full_gold == (endpoint == "FNR"))
                    )
                    if not len(ix):
                        continue
                    eb = (saved["full_probabilities"][b, ix] > cutoff) != full_gold[ix]
                    eu = (saved["full_probabilities"][j, ix] > cutoff) != full_gold[ix]
                    full.append(
                        dict(
                            job_index=j,
                            aliases=job["aliases"],
                            variant=v,
                            cutoff=cutoff,
                            population=population,
                            endpoint=endpoint,
                            inputs=len(ix),
                            baseline_errors=int(eb.sum()),
                            updated_errors=int(eu.sum()),
                            fixes=int((eb & ~eu).sum()),
                            new_errors=int((~eb & eu).sum()),
                            absolute_probability_shift=quantiles(abs(delta[ix])),
                        )
                    )
    cases = []
    for population in sorted(set(pop)):
        for endpoint in ["FPR", "FNR"]:
            ids = np.flatnonzero(
                (pop == population) & (gold == (endpoint == "FNR")) & fixed
            )
            e = z["linear_errors"][:, 0, :][:, :, [1, 4, 7]]
            b = z["linear_errors"][:, 0, 0, 0]
            shift = abs(
                z["probabilities"][:, 0][:, :, [1, 4, 7]]
                - z["probabilities"][:, 0, 0, 0, None, None]
            ).max((1, 2))
            allpath = np.concatenate(
                [np.broadcast_to(b[:, None, None], (len(meta), 1, 3)), e], axis=1
            )
            choice = {
                "maximum_probability_shift": np.ones(len(meta), bool),
                "lasting_fix": b.astype(bool) & (~e.astype(bool)).all(1).any(1),
                "lasting_regression": ~b.astype(bool) & e.astype(bool).all(1).any(1),
                "reversal": ((allpath[:, 1:] != allpath[:, :-1]).sum(1) >= 2).any(1),
                "classifier_correct_qwen_mostly_wrong": (e[:, 3].mean(1) == 0)
                & (z["qwen_errors"][:, 3][:, [1, 4, 7]].mean((1, 2)) > 0.5),
                "qwen_correct_classifier_mostly_wrong": (e[:, 3].mean(1) > 0.5)
                & (z["qwen_errors"][:, 3][:, [1, 4, 7]].mean((1, 2)) == 0),
            }
            for category, eligible in choice.items():
                pool = [i for i in ids if eligible[i]]
                if not pool:
                    cases.append(
                        dict(
                            population=population,
                            endpoint=endpoint,
                            category=category,
                            eligible=0,
                            case=None,
                        )
                    )
                    continue
                i = min(
                    pool,
                    key=lambda i: (
                        -shift[i],
                        hashlib.sha256(meta[i]["key"].encode()).hexdigest(),
                    ),
                )
                cases.append(
                    dict(
                        population=population,
                        endpoint=endpoint,
                        category=category,
                        eligible=len(pool),
                        case=dict(
                            **meta[i],
                            max_absolute_probability_shift=shift[i],
                            probabilities=z["probabilities"][i, 0],
                            classifier_errors=z["linear_errors"][i, 0],
                            qwen_errors=z["qwen_errors"][i],
                        ),
                    )
                )
    jobs = [j for j in execution["jobs"] if j["new"]]
    mon = execution["monitors"]
    finishes = [datetime.fromisoformat(x["utc"]) for x in mon]
    starts = [f - timedelta(seconds=m["seconds"]) for f, m in zip(finishes, mon)]
    resources = dict(
        new_fits=64,
        reused_baselines=2,
        unique_cumulative_fits=66,
        logical_aliases=74,
        all_distinct_classifier_fits_static_plus_cumulative=102,
        new_fit_seconds=sum(x["attribution"]["fit_seconds"] for x in jobs),
        new_feature_seconds=sum(x["attribution"]["feature_seconds"] for x in jobs),
        new_inference_and_check_seconds=sum(
            x["attribution"]["inference_and_verification_seconds"] for x in jobs
        ),
        new_worker_attempt_seconds=sum(x["seconds_this_attempt"] for x in jobs),
        new_monitored_seconds=sum(x["seconds"] for x in mon),
        monitor_interval_start_derived_utc=min(starts).isoformat(),
        monitor_interval_end_recorded_utc=max(finishes).isoformat(),
        monitor_span_seconds=(max(finishes) - min(starts)).total_seconds(),
        span_note="Start derived from monitor finish UTC minus its monotonic duration; includes small inter-worker gaps, not setup before first worker.",
        fit_seconds=quantiles([x["attribution"]["fit_seconds"] for x in jobs]),
        iterations=quantiles([x["attribution"]["iterations"][0] for x in jobs]),
        max_iter=2000,
        tol=1e-8,
        convergence_warnings="Promoted to errors by frozen worker; all jobs completed.",
        min_sampled_available_GiB=min(x["sampled_min_available_GiB"] for x in mon),
        max_sampled_worker_private_GiB=max(
            x["sampled_peak_worker_tree_private_GiB"] for x in mon
        ),
        max_sampled_worker_working_set_GiB=max(
            x["sampled_peak_worker_tree_working_set_GiB"] for x in mon
        ),
        monitor_interval_seconds=0.5,
        new_fit_artifact_bytes=sum(x["artifact_bytes"] for x in jobs),
        new_model_bytes=sum(
            (a.ROOT / x["receipt"])
            .parent.joinpath(a.read(a.ROOT / x["receipt"])["attempt"], "model.npz")
            .stat()
            .st_size
            for x in jobs
        ),
        new_fit_directories_total_bytes=sum(
            sum(
                p.stat().st_size
                for p in (a.ROOT / x["receipt"]).parent.rglob("*")
                if p.is_file()
            )
            for x in jobs
        ),
        new_monitor_total_bytes=sum(
            sum(
                p.stat().st_size
                for p in (a.ROOT / x["path"]).parent.iterdir()
                if p.is_file()
            )
            for x in mon
        ),
        new_Qwen_calls=0,
        new_embeddings=0,
        CUDA=False,
        main_train_rows=55753,
        supporting_train_rows=50173,
        final_added_rows=40,
        original_training_cost_included=False,
        cached_embedding_creation_cost_included=False,
        source_screening_cost_included=False,
        longer_training_assessment="All new fits converge in 10–11 solver iterations under max_iter 2000, tol 1e-8; raising the cap alone does not change the fitted objective. Reweighting feedback, retuning or learning representations would be a different experiment, not completion of this fit.",
    )
    a.save("source_intervals.json", source_intervals)
    a.save("probability_movement.json", probability)
    a.save("durability.json", paths)
    a.save("transitions.json", transitions)
    a.save("cohort_filter.json", quality_filter)
    a.jsonlines("changed_evaluation_texts.jsonl", changed)
    a.save("source_adoption.json", adoption)
    a.jsonlines("source_adoption_ledger.jsonl", source_ledger)
    a.save("full_roster_diagnostics.json", full)
    a.save("model_movement.json", model_motion)
    a.save("cases.json", cases)
    a.save("resources.json", resources)
    names = [
        "source_intervals.json",
        "probability_movement.json",
        "durability.json",
        "transitions.json",
        "cohort_filter.json",
        "changed_evaluation_texts.jsonl",
        "source_adoption.json",
        "source_adoption_ledger.jsonl",
        "full_roster_diagnostics.json",
        "model_movement.json",
        "cases.json",
        "resources.json",
    ]
    a.save(
        "details_complete.json",
        dict(
            status="details_complete",
            outputs=a.bindings([a.OUT / n for n in names] + extra_draws),
            changed_evaluation_texts=len(changed),
            source_addition_memberships=len(source_ledger),
            case_slots=len(cases),
            empty_case_slots=sum(c["case"] is None for c in cases),
            source_intervals=len(source_intervals),
            analysis_adjustment="Natural-source interval supplement uses the observed en_test population identifier. The 52 main/sensitivity groups and their values remain unchanged.",
        ),
    )
    print(
        "Completed probability, source-adoption, durability, subgroup, case and resource diagnostics.",
        flush=True,
    )


if __name__ == "__main__":
    with a.c.offline(), threadpool_limits(limits=1):
        main()

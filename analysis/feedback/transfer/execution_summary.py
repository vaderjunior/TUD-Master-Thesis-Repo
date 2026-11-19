from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import numpy as np
from analysis.feedback.transfer import execution as p
from analysis.feedback.transfer import execution_metrics as k
from analysis.feedback.transfer import effects_metrics as rate
from analysis.feedback.transfer import quality_metrics as quality
from analysis.feedback.transfer.quality import summary as quality_summary

a = p.a
OUT = p.OUT
ENDPOINTS = (
    "en_test_FPR",
    "en_test_FNR",
    "en_hatecheck_eval_FPR",
    "en_hatecheck_eval_FNR",
)
HEADS = ("binary", "known_answer", "hate_type", "target_group", "severity", "legal")


def interval_kind(interval):
    if interval is None:
        return "not_estimated"
    return (
        "above_zero"
        if interval[0] > 0
        else "below_zero" if interval[1] < 0 else "includes_zero"
    )


def main():
    a.need(
        not (OUT / "calculated.json").exists()
        and not (OUT / "era_rates.json").exists(),
        "Preserve existing sensitivity results",
    )
    assembled = a.read(OUT / "assembled.json")
    for name, digest in assembled["artifacts"].items():
        a.need(a.sha(a.ROOT / name) == digest, "Changed Part 7 assembly")
    meta = list(a.rows(p.P4 / "metadata.jsonl"))
    N = len(meta)
    input_era = list(p.gzrows(OUT / "input_execution.jsonl.gz"))
    a.need(
        len(input_era) == N == 15758
        and all(m["key"] == input_era[i]["key"] for i, m in enumerate(meta)),
        "Input identity",
    )
    era = np.array([q["actual_execution_era"] for q in input_era])
    gold = np.array([m["gold_hate"] for m in meta], dtype=bool)
    with np.load(p.P6 / "arrays.npz", allow_pickle=False) as f:
        codes = f["codes"]
        votes = f["n_valid"]
        raw = f["votes"][..., 0]
        scores = f["scores"]
    valid = votes > 0
    errors = np.where(valid, codes[..., 0] != gold[:, None, None], -1).astype(np.int8)
    with np.load(p.P4 / "features.npz", allow_pickle=False) as f:
        data = {n: f[n] for n in f.files}
    labels = a.read(p.P4 / "analysis_contract.json")["population_macro_labels"]
    science = a.read(a.STUDY / "scientific_plan_v1.json")
    exclusions = science["mandatory_replay_keys"]
    oldrates = a.read(p.P2 / "rates.json")
    oldquality = a.read(p.P4 / "quality_results.json")
    rates = {}
    qresults = {}
    drawsets = 0
    global_cohorts = {}
    for endpoint in ENDPOINTS:
        pop, ep = endpoint.rsplit("_", 1)
        for view, (strict, reps, ex) in rate.VIEWS.items():
            c = rate.cohort(
                meta,
                errors,
                votes,
                pop,
                ep == "FNR",
                strict=strict,
                representatives=reps,
                excluded=exclusions if ex else [],
            )
            old = oldrates[f"{view}/{endpoint}"]
            a.need(
                c["common"].tolist() == old["common_indices"]
                and c["eligible"].tolist() == old["eligible_indices"],
                "Inherited endpoint mask differs",
            )
            for name, value in c["point"].items():
                if isinstance(value, float):
                    a.need(
                        abs(value - old[name]) < 1e-14,
                        "Inherited point differs " + name,
                    )
                else:
                    a.need(value == old[name], "Inherited count/bounds differs " + name)
            global_cohorts[endpoint, view] = c

    def evaluate(name, endpoint, view, members, bootstrap=False):
        nonlocal drawsets
        pop, ep = endpoint.rsplit("_", 1)
        strict, reps, ex = rate.VIEWS[view]

        c = rate.cohort(
            meta,
            errors,
            votes,
            pop,
            ep == "FNR",
            strict=strict,
            representatives=reps,
            excluded=exclusions if ex else [],
            eligible_override=members,
        )
        common = c["common"]
        row = dict(
            id=name,
            endpoint=endpoint,
            view=view,
            eligible_indices=c["eligible"],
            common_indices=common,
            **c["point"],
            clusters=len({meta[i]["inference_cluster"] for i in common}),
            interval95=None,
            pool_difference_intervals95=None,
            interval_role=(
                "Supporting 95% within-era paired cluster interval; not a concurrency effect or new primary family."
                if bootstrap
                else "Descriptive point; no new interval."
            ),
            per_run_errors=errors[common].sum(axis=0, dtype=np.int64),
            source_counts=Counter(meta[i]["source"] for i in common),
        )
        if bootstrap and len(common):
            draws, record = rate.bootstrap(meta, errors, c, k.seed(name))
            path = (
                OUT
                / "draws"
                / (hashlib.sha256(name.encode()).hexdigest()[:24] + ".npy")
            )
            path.parent.mkdir(exist_ok=True)
            with path.open("xb") as f:
                np.save(f, draws, allow_pickle=False)
            row.update(
                bootstrap=record,
                draw_file=a.rel(path),
                interval95=np.quantile(draws[:, 5], [0.025, 0.975]).tolist(),
                pool_difference_intervals95=np.quantile(
                    draws[:, 1:4] - draws[:, :1], [0.025, 0.975], axis=0
                ).T.tolist(),
            )
            drawsets += 1
        rates[name] = p.native(row)
        return row

    for label in ("all_8", "all_128"):
        members = np.flatnonzero(era == label)
        for endpoint in ENDPOINTS:
            for view in rate.VIEWS:
                evaluate(
                    f"era/{label}/{endpoint}/{view}",
                    endpoint,
                    view,
                    members,
                    view in ("main", "strict"),
                )
        for source in sorted(
            {m["source"] for m in meta if m["population"] == "en_test"}
        ):
            ids = [i for i in members if meta[i]["source"] == source]
            for ep in ("FPR", "FNR"):
                for view in ("main", "strict"):
                    evaluate(
                        f"source_era/{source}/{label}/en_test_{ep}/{view}",
                        "en_test_" + ep,
                        view,
                        ids,
                    )
        print("Calculated homogeneous runtime-era errors: " + label, flush=True)
    for endpoint in ENDPOINTS:
        evaluate(
            f"cross_boundary/{endpoint}/main",
            endpoint,
            "main",
            np.flatnonzero(era == "cross_boundary"),
        )
        for view in rate.VIEWS:
            evaluate(
                f"boundary_excluded/{endpoint}/{view}",
                endpoint,
                view,
                np.flatnonzero(era != "cross_boundary"),
            )
        for stage_period in ("original_8", "mixed_resume_8_128", "later_128"):
            ids = [
                i for i, m in enumerate(meta) if m["execution_period"] == stage_period
            ]
            for view in ("main", "strict"):
                evaluate(
                    f"coarse_period/{stage_period}/{endpoint}/{view}",
                    endpoint,
                    view,
                    ids,
                )
    a.need(
        drawsets == 16 and len(rates) == 124, "Complete runtime sensitivity inventory"
    )
    p.save("era_rates.json", rates)

    for group in ("all_8", "all_128", "boundary_excluded"):
        allowed = (
            era != "cross_boundary" if group == "boundary_excluded" else era == group
        )
        for pop in p.POPS:
            members = [
                i for i, m in enumerate(meta) if m["population"] == pop and allowed[i]
            ]
            for head in HEADS:
                for view in ("main", "strict"):
                    name = f"{group}/{pop}/{head}/{view}"
                    mi = [quality.LABELS[head].index(lab) for lab in labels[pop][head]]
                    eligible, common = quality.selected(
                        meta, members, head, view, exclusions, valid, votes
                    )
                    row = dict(
                        id=name,
                        group=group,
                        population=pop,
                        head=head,
                        view=view,
                        eligible_indices=eligible,
                        common_indices=common,
                        eligible_n=len(eligible),
                        common_n=len(common),
                        macro_indices=mi,
                        population_macro_labels=labels[pop][head],
                        interval_role="Descriptive existing metrics; no new interval.",
                    )
                    if head not in data or not len(common):
                        row.update(
                            status=(
                                "unscorable_no_known_gold"
                                if not len(eligible)
                                else "undefined_no_common_outputs"
                            ),
                            metrics={},
                        )
                    else:
                        row.update(
                            status="defined",
                            **quality_summary(
                                data[head][common].sum(axis=0, dtype=np.int64), head, mi
                            ),
                        )
                        row["missing_bounds"] = quality.missing_bounds(
                            meta,
                            data[head],
                            eligible,
                            valid,
                            votes,
                            head,
                            mi,
                            view == "strict",
                        )
                    qresults[name] = p.native(row)
    p.save("era_quality.json", qresults)

    standard = {}
    for endpoint in ("en_test_FPR", "en_test_FNR"):
        base = global_cohorts[endpoint, "main"]["common"]
        cnt = Counter(meta[i]["source"] for i in base)
        weights = {key: v / len(base) for key, v in sorted(cnt.items())}
        eras = {}
        for label in ("all_8", "all_128"):
            rows = {
                source: rates[f"source_era/{source}/{label}/{endpoint}/main"]
                for source in weights
            }
            a.need(
                all(q["common_n"] for q in rows.values()),
                "Missing source for descriptive standardization",
            )
            eras[label] = {
                metric: sum(weights[source] * q[metric] for source, q in rows.items())
                for metric in ("reference_rate", "treatment_rate", "difference")
            }
        standard[endpoint] = dict(
            weights=weights,
            source_common_counts=cnt,
            eras=eras,
            qualification="Descriptive reweighting across three observed sources only. It does not balance within-source text difficulty or isolate concurrency/backend effects.",
        )
    p.save("source_standardization.json", standard)

    lower, upper, n_missing = k.vote_completion(raw, gold)
    a.need(np.array_equal(n_missing, 3 - votes), "Raw vote/missing count")
    a.need(
        np.all((~valid) | (lower <= errors)) and np.all((~valid) | (errors <= upper)),
        "Saved aggregate outside possible vote completion",
    )
    bounds = {}
    for endpoint in ENDPOINTS:
        pop, ep = endpoint.rsplit("_", 1)
        base = [
            i
            for i, m in enumerate(meta)
            if m["population"] == pop and m["gold_hate"] == (ep == "FNR")
        ]
        sets = {
            "full_roster": base,
            **{
                label: [i for i in base if era[i] == label]
                for label in ("all_8", "all_128", "cross_boundary")
            },
        }
        if pop == "en_test":
            sets.update(
                {
                    f"source/{source}": [i for i in base if meta[i]["source"] == source]
                    for source in sorted({meta[i]["source"] for i in base})
                }
            )
        for group, ids in sets.items():
            ids = np.asarray(ids, dtype=int)
            err = errors[ids]
            nv = votes[ids]
            hyp = k.opposing(lower[ids], upper[ids])
            null = k.opposing(np.where(nv > 0, err, 0), np.where(nv > 0, err, 1))
            strict = k.opposing(np.where(nv == 3, err, 0), np.where(nv == 3, err, 1))
            uncertain = lower[ids] != upper[ids]
            under = nv < 3
            bounds[f"{endpoint}/{group}"] = dict(
                endpoint=endpoint,
                group=group,
                indices=ids,
                null_only=null,
                all_under_three_unknown=strict,
                invalid_vote_completion=hyp,
                fewer_than_three_outputs=int(under.sum()),
                already_determined_despite_invalid_votes=int(
                    (under & ~uncertain).sum()
                ),
                binary_undetermined_outputs=int(uncertain.sum()),
                inputs_with_any_under_three=int(under.any(axis=(1, 2)).sum()),
                inputs_with_any_undetermined=int(uncertain.any(axis=(1, 2)).sum()),
            )
            if group == "full_roster":
                a.need(
                    np.allclose(
                        null["difference"],
                        oldrates[f"main/{endpoint}"]["difference_missing_bounds"],
                        atol=1e-15,
                        rtol=0,
                    ),
                    "Null bounds mismatch",
                )
                a.need(
                    np.allclose(
                        strict["difference"],
                        oldrates[f"strict/{endpoint}"]["difference_missing_bounds"],
                        atol=1e-15,
                        rtol=0,
                    ),
                    "Strict bounds mismatch",
                )
    p.save("vote_completion_bounds.json", bounds)
    with (OUT / "vote_completion_arrays.npz").open("xb") as f:
        np.savez_compressed(f, lower_error=lower, upper_error=upper)

    groups = {"all": list(range(N))}
    for field in ("population", "source", "stage"):
        for value in sorted({m[field] for m in meta}):
            groups[f"{field}/{value}"] = [
                i for i, m in enumerate(meta) if m[field] == value
            ]
    for label in ("all_8", "all_128", "cross_boundary"):
        groups[f"era/{label}"] = np.flatnonzero(era == label).tolist()
        for pop in p.POPS:
            for truth in (False, True):
                groups[f"{pop}/gold_{int(truth)}/{label}"] = [
                    i
                    for i, m in enumerate(meta)
                    if m["population"] == pop
                    and m["gold_hate"] == truth
                    and era[i] == label
                ]
        for source in sorted({m["source"] for m in meta}):
            groups[f"source/{source}/{label}"] = [
                i
                for i, m in enumerate(meta)
                if m["source"] == source and era[i] == label
            ]
    availability = {}
    for name, ids in groups.items():
        vv = votes[ids]
        null = vv == 0
        under = vv < 3
        availability[name] = dict(
            inputs=len(ids),
            slots=len(ids) * 12,
            inputs_any_null=int(null.any(axis=(1, 2)).sum()),
            inputs_any_under_three=int(under.any(axis=(1, 2)).sum()),
            common_inputs=int((~null.any(axis=(1, 2))).sum()),
            strict_inputs=int((~under.any(axis=(1, 2))).sum()),
            valid_vote_histogram=np.bincount(vv.reshape(-1), minlength=4),
            per_condition_repeat_vote_counts=np.stack(
                [(vv == n).sum(axis=0, dtype=np.int64) for n in range(4)]
            ),
            affected_input_source_gold=dict(
                Counter(
                    f"{meta[i]['source']}/gold_{int(meta[i]['gold_hate'])}"
                    for i in ids
                    if (votes[i] == 0).any()
                )
            ),
            null_outputs_per_input_histogram=dict(
                Counter(map(int, null.sum(axis=(1, 2))))
            ),
            under_three_per_input_histogram=dict(
                Counter(map(int, under.sum(axis=(1, 2))))
            ),
        )
    p.save("availability.json", availability)

    influences = {}
    rows = 0
    with p.gzopen("group_omissions.jsonl.gz") as f:
        for endpoint in ENDPOINTS:
            pop, ep = endpoint.rsplit("_", 1)
            for view in ("main", "strict"):
                ids = global_cohorts[endpoint, view]["common"]
                base = global_cohorts[endpoint, view]["point"]["difference"]
                for field in (
                    ("inference_cluster", "functionality")
                    if pop == "en_hatecheck_eval"
                    else ("inference_cluster",)
                ):
                    grouped = defaultdict(list)
                    for i in ids:
                        grouped[meta[i][field]].append(int(i))
                    names = sorted(grouped)
                    sizes = np.array([len(grouped[n]) for n in names])
                    ct = np.array(
                        [
                            errors[grouped[n]].sum(axis=(0, 2), dtype=np.int64)
                            for n in names
                        ]
                    )
                    remaining, num, diff = k.omission(ct.sum(axis=0), sizes, ct)
                    name = f"{endpoint}/{view}/{field}"
                    items = []
                    for j, label in enumerate(names):
                        row = dict(
                            view_id=name,
                            omitted=label,
                            omitted_indices=grouped[label],
                            omitted_inputs=int(sizes[j]),
                            omitted_errors=ct[j],
                            remaining_inputs=int(remaining[j]),
                            difference_numerator=int(num[j]),
                            difference=diff[j],
                            base_difference=base,
                        )
                        p.emit(f, row)
                        items.append(row)
                        rows += 1
                    finite = [q for q in items if np.isfinite(q["difference"])]
                    influences[name] = dict(
                        groups=len(items),
                        inputs=len(ids),
                        base_difference=base,
                        minimum=min(q["difference"] for q in finite),
                        maximum=max(q["difference"] for q in finite),
                        opposite_sign_omissions=sum(
                            q["difference"] * base < 0 for q in finite
                        ),
                        exact_zero_omissions=sum(q["difference"] == 0 for q in finite),
                        undefined_omissions=len(items) - len(finite),
                        largest_effect_change=sorted(
                            finite,
                            key=lambda q: (-abs(q["difference"] - base), q["omitted"]),
                        )[:10],
                    )
    p.save("group_influence_summary.json", influences)

    reused = dict(
        primary_rates={},
        quality={},
        source_rate_evidence=dict(
            results=a.rel(p.P3 / "rates.json"),
            verification=a.rel(p.P3 / "independent_check.json"),
        ),
        error_omissions=a.read(p.P2 / "omissions.json"),
        quality_omissions=a.read(p.P4 / "omissions.json"),
    )
    for name, q in oldrates.items():
        reused["primary_rates"][name] = {
            f: q[f]
            for f in (
                "endpoint",
                "common_n",
                "eligible_n",
                "reference_rate",
                "treatment_rate",
                "difference",
                "interval95",
                "interval98_75",
                "interval_confidence_for_this_view",
                "difference_missing_bounds",
                "resampling_mode",
                "pool_rates",
                "pool_difference_intervals95",
                "draw_file",
            )
        }
        reused["primary_rates"][name]["interval98_75_direction"] = interval_kind(
            q["interval98_75"]
        )
    for name, q in oldquality.items():
        out = {
            f: q[f]
            for f in ("population", "head", "view", "common_n", "eligible_n", "status")
        }
        out["metrics"] = {}
        for metric, v in q["metrics"].items():
            out["metrics"][metric] = {
                f: v[f]
                for f in (
                    "reference",
                    "treatment",
                    "difference",
                    "pool_values",
                    "pool_differences",
                )
                if f in v
            } | dict(
                interval95=v.get("interval95"),
                interval95_direction=interval_kind(v.get("interval95")),
            )
        out["missing_bounds"] = {
            metric: v["opposing_difference"]
            for metric, v in q.get("missing_bounds", {}).get("metrics", {}).items()
        }
        reused["quality"][name] = out
    p.save("reused_robustness.json", reused)

    roster = list(a.rows(a.OUT / "roster.jsonl"))
    composition = {}
    for pop in p.POPS:
        for label in ("all_8", "all_128", "cross_boundary"):
            ids = [
                i
                for i, m in enumerate(meta)
                if m["population"] == pop and era[i] == label
            ]
            composition[f"{pop}/{label}"] = dict(
                inputs=len(ids),
                gold_counts=Counter(str(int(meta[i]["gold_hate"])) for i in ids),
                source_counts=Counter(meta[i]["source"] for i in ids),
                function_counts=Counter(
                    meta[i]["functionality"] for i in ids if meta[i]["functionality"]
                ),
                known_gold_profiles=Counter(meta[i]["gold_profile"] for i in ids),
                text_characters={
                    q: (
                        float(np.quantile([len(roster[i]["text"]) for i in ids], v))
                        if ids
                        else None
                    )
                    for q, v in [("median", 0.5), ("p10", 0.1), ("p90", 0.9)]
                },
            )
    clusters = defaultdict(set)
    for i, m in enumerate(meta):
        clusters[m["population"], m["gold_hate"], m["inference_cluster"]].add(era[i])
    p.save(
        "era_composition.json",
        dict(
            groups=composition,
            endpoint_cluster_era_membership=Counter(
                "|".join(sorted(v)) for v in clusters.values()
            ),
            warning="Some related inputs span eras; era estimates are not independent populations or a randomized concurrency experiment. Source-standardized rates leave within-source text, timing, prompt and backend differences uncontrolled.",
        ),
    )
    files = [path for path in OUT.rglob("*") if path.is_file()]
    p.save(
        "calculated.json",
        dict(
            status="part7_robustness_calculated_verification_pending",
            created_utc=a.utc(),
            assembled_sha256=a.sha(OUT / "assembled.json"),
            script_sha256=a.sha(Path(__file__)),
            artifacts={a.rel(f): a.sha(f) for f in files},
            new_rate_views=len(rates),
            new_supporting_bootstrap_sets=drawsets,
            draws_per_set=10000,
            columns_per_draw=6,
            new_quality_views=len(qresults),
            vote_completion_bound_views=len(bounds),
            availability_views=len(availability),
            group_omission_rows=rows,
            group_influence_views=len(influences),
            reused_rate_views=len(oldrates),
            reused_quality_views=len(oldquality),
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
            new_primary_families=0,
        ),
    )
    print(
        "Calculated all Part 7 robustness views; independent verification follows.",
        flush=True,
    )


if __name__ == "__main__":
    with a.offline():
        main()

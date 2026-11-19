from pathlib import Path
import math
import numpy as np
from analysis.feedback.transfer import resources as p

a = p.a
OUT = p.OUT
k = p.k


def main():
    a.need(
        (OUT / "assembled.json").exists() and not (OUT / "calculated.json").exists(),
        "Complete new resource assembly",
    )
    assembled = a.read(OUT / "assembled.json")
    for name, digest in assembled["artifacts"].items():
        a.need(a.sha(a.ROOT / name) == digest, "Changed assembled resource " + name)
    p.save(
        "analysis_implementation.json",
        dict(
            created_utc=a.utc(),
            script_sha256=a.sha(Path(__file__)),
            cost_families=[
                "en_test/all_roster",
                "en_test/main_common",
                "en_hatecheck_eval/all_roster",
                "en_hatecheck_eval/main_common",
            ],
            scenario_prevalences=[0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7],
            scenario_cost_ratios=[1, 2, 5, 10, 20],
            scenario_comparisons=[
                "Qwen feedback minus Qwen reference",
                "Qwen feedback minus principal classifier feedback",
            ],
            scenario_role="Illustrative point algebra using recorded English rates; not measured deployment conditions, new decision tuning, prevalence transport validation, confidence intervals or a price estimate.",
        ),
    )
    with np.load(OUT / "arrays.npz", allow_pickle=False) as f:
        x = f["resources"]
        valid = f["valid_votes"]
        local = f["local_prompt_tokens"]
    meta = list(a.rows(p.P3 / "metadata.jsonl"))
    matrix = list(a.rows(a.OUT / "matrix.jsonl"))
    features = k.paired(x)
    comparisons = {}
    drawdir = OUT / "draws"
    drawdir.mkdir()
    for pop in p.POPS:
        for cohort in ("all_roster", "main_common"):
            name = pop + "/" + cohort
            idx = [
                i
                for i, m in enumerate(meta)
                if m["population"] == pop
                and (cohort == "all_roster" or matrix[i]["masks"]["common_valid"])
            ]
            clusters = [meta[i]["inference_cluster"] for i in idx]
            values = features[idx]
            seed = k.seed(name)
            draws = k.bootstrap(values, clusters, seed)
            path = drawdir / (name.replace("/", "__") + ".npy")
            with path.open("xb") as f:
                np.save(f, draws, allow_pickle=False)
            estimates = values.mean(axis=0)
            quantiles = np.quantile(draws, [0.025, 0.975], axis=0)
            metrics = {
                metric: {
                    column: dict(
                        estimate=float(estimates[j, c]),
                        interval95=[float(v) for v in quantiles[:, j, c]],
                    )
                    for c, column in enumerate(k.COLUMNS)
                }
                for j, metric in enumerate(k.METRICS)
            }
            comparisons[name] = dict(
                population=pop,
                cohort=cohort,
                input_indices=idx,
                inputs=len(idx),
                inference_clusters=len(set(clusters)),
                seed=seed,
                draws=a.rel(path),
                metrics=metrics,
                role="Supporting paired whole-cluster interval; four fixed condition arms and actual three decoder repeats. All-roster totals retain null costs. Not a new primary family or causal runtime comparison.",
            )
            print("Paired resource comparison: " + name, flush=True)
    p.save("paired_costs.json", comparisons)
    groups = a.read(OUT / "resource_groups.json")
    total = groups["all"]
    ac = total["accounting"]
    u = total["observed_usage"]
    fail = a.read(OUT / "failure_resources.json")
    timeline = a.read(OUT / "timeline.json")
    profiles = []
    for i, m in enumerate(meta):
        row = x[i].sum(axis=(0, 1))
        profiles.append(
            dict(
                input_index=i,
                key=m["key"],
                population=m["population"],
                source=m["source"],
                gold_hate=m["gold_hate"],
                total_tokens=int(row[0]),
                repair_tokens=int(row[2]),
                repairs=int(row[3]),
                extra_attempts=int(row[4]),
                request_seconds=float(row[6]),
                null_outputs=int((valid[i] == 0).sum()),
                under_three_outputs=int((valid[i] < 3).sum()),
            )
        )
    ranking = sorted(profiles, key=lambda q: (-q["repairs"], q["key"]))
    concentration = {
        str(n): dict(
            inputs=n,
            repairs=sum(q["repairs"] for q in ranking[:n]),
            share=sum(q["repairs"] for q in ranking[:n]) / ac["repair_logical_calls"],
        )
        for n in (1, 10, 50, 100, 500)
    }
    p.save(
        "repair_concentration.json",
        dict(
            top_20=ranking[:20],
            top_n=concentration,
            inputs_with_repairs=sum(q["repairs"] > 0 for q in profiles),
            inputs_with_extra_attempts=sum(q["extra_attempts"] > 0 for q in profiles),
            all_input_profiles=a.rel(OUT / "input_costs.jsonl.gz"),
            selection="Descending actual repairs across every condition/repeat; ties use original key. Descriptive concentration, not a causal difficult-text or intervention claim.",
        ),
    )
    summary = dict(
        counts=total["counts"],
        accounting=ac,
        ordinary_usage=total["ordinary_usage"],
        repair_usage=total["repair_usage"],
        observed_usage=u,
        reported_tokens_per_planned_prediction=u["total_tokens"] / 189096,
        reported_tokens_per_nonnull_prediction_including_all_spend=u["total_tokens"]
        / (189096 - total["counts"]["nulls"]),
        repair_calls_per_100_ordinary_calls=100
        * ac["repair_logical_calls"]
        / ac["ordinary_logical_calls"],
        retry_attempts_per_100_logical_calls=100
        * ac["retry_attempts"]
        / ac["http_success_with_text"],
        attempt_overhead_percent=100
        * (ac["repair_logical_calls"] + ac["retry_attempts"])
        / ac["ordinary_logical_calls"],
        repair_token_share=total["repair_usage"]["total_tokens"] / u["total_tokens"],
        known_POST_seconds=total["known_request_seconds_sum"],
        recorded_retry_wait_seconds=total["retry_backoff_seconds_sum"],
        known_POST_duration_quantiles={
            name: total["known_request_seconds_" + name]
            for name in ("p50", "p95", "p99", "min", "max")
        },
        failed_POST_seconds=fail["groups"]["all"]["request_seconds"],
        successful_POST_seconds=total["known_request_seconds_sum"]
        - fail["groups"]["all"]["request_seconds"],
        failure_time_share=fail["groups"]["all"]["request_seconds"]
        / total["known_request_seconds_sum"],
        availability_GETs=len(timeline["availability_checks"]),
        availability_seconds=timeline["availability_seconds"],
        calendar_hours_generation=timeline["start_to_last_generation_seconds"] / 3600,
        calendar_hours_audits=timeline["start_to_last_audit_seconds"] / 3600,
        calendar_hours_final_proxy=timeline["start_to_final_file_proxy_seconds"] / 3600,
        resource_groups=len(groups),
        paired_cost_families=len(comparisons),
        new_Qwen_calls=0,
        new_fits=0,
        new_embeddings=0,
        remote_GPU_hours=None,
        remote_energy=None,
        money_cost=None,
        whole_thesis_cost=False,
    )
    p.save("resource_summary.json", summary)

    rates = a.read(p.P8 / "rates.json")
    fp = rates["all_permitted/cutoff_06/cohort/main/en_test_FPR"]
    fn = rates["all_permitted/cutoff_06/cohort/main/en_test_FNR"]
    prevalence = fn["common_n"] / (fp["common_n"] + fn["common_n"])
    scenarios = {}
    contrasts = {
        "Qwen_feedback_vs_reference": (
            fp["estimates"]["qwen_change"],
            fn["estimates"]["qwen_change"],
        ),
        "Qwen_feedback_vs_classifier_feedback": (
            -fp["estimates"]["updated_absolute_difference"],
            -fn["estimates"]["updated_absolute_difference"],
        ),
    }
    for name, (dfp, dfn) in contrasts.items():
        rows = []
        for pv in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7):
            rows.append(
                dict(
                    hypothetical_hate_prevalence=pv,
                    extra_false_alarms_per_1000=1000 * (1 - pv) * dfp,
                    missed_hate_change_per_1000=1000 * pv * dfn,
                    miss_to_false_alarm_break_even_cost=k.break_even_ratio(
                        dfp, dfn, pv
                    ),
                    weighted_changes={
                        str(r): k.error_cost(dfp, dfn, pv, r) for r in (1, 2, 5, 10, 20)
                    },
                )
            )
        scenarios[name] = dict(
            delta_FPR=dfp,
            delta_FNR=dfn,
            extra_false_alarms_per_1000_benign=1000 * dfp,
            missed_hate_change_per_1000_hateful=1000 * dfn,
            equal_error_cost_break_even_prevalence=dfp / (dfp - dfn),
            recorded_sample_prevalence=prevalence,
            sample_mean_extra_false_alarms=fp["common_n"] * dfp,
            sample_mean_missed_hate_change=fn["common_n"] * dfn,
            sample_ratio_break_even=k.break_even_ratio(dfp, dfn, prevalence),
            sample_error_change_per_1000=k.error_cost(dfp, dfn, prevalence),
            rows=rows,
        )
    p.save(
        "practical_scenarios.json",
        dict(
            population="English main common only",
            common_benign=fp["common_n"],
            common_hateful=fn["common_n"],
            comparisons=scenarios,
            formula="Per 1000: 1000*((1-p)*delta_FPR + p*r*delta_FNR), where r is missed-hate cost / false-alarm cost. Positive means more weighted errors for the first system.",
            limitations=[
                "The grid supplies hypothetical p and r, not measured prevalence or real harm valuations.",
                "Conditional rates and source mix are held fixed for illustration; transfer to a changed population is untested.",
                "Points only: original error intervals, missing-output bounds and repeat variation still apply.",
                "No utility estimate is derived from HateCheck prevalence.",
                "These binary calculations omit fine-label quality and compute/annotation costs.",
                "No policy, cutoff or next experiment is chosen using a scenario threshold.",
            ],
        ),
    )

    oldpath = (
        a.ROOT
        / "experiments/astra/sq3_human_feedback_results_20261020_v1/partial_programme_ledger.json"
    )
    old = a.read(oldpath)
    fields = {
        "ordinary_slots": "ordinary_logical_calls",
        "repair_slots": "repair_logical_calls",
        "retry_or_recovered_failure_attempts": "retry_attempts",
        "successful_responses": "http_success_with_text",
        "attempt_slots": "post_attempts",
    }
    row = dict(
        programme="SQ3 broader correction transfer",
        source=a.rel(OUT / "resource_summary.json"),
        source_sha256=a.sha(OUT / "resource_summary.json"),
        **{name: ac[key] for name, key in fields.items()},
        reported_tokens=u["total_tokens"]
    )
    a.need(
        len(old["programmes"]) == 11
        and all(
            old["totals"][key] == sum(q[key] for q in old["programmes"])
            for key in old["totals"]
        ),
        "Previous exact eleven-programme totals",
    )
    p.save(
        "resource_addition.json",
        dict(
            status="reviewed_part9_row_staged_for_part10_global_integration",
            previous_ledger=a.rel(oldpath),
            previous_ledger_sha256=a.sha(oldpath),
            preserved_previous_programmes=old["programmes"],
            addition=row,
            prospective_twelve_programme_totals={
                key: value + row[key] for key, value in old["totals"].items()
            },
            capacity_diagnostics_excluded=True,
            inherited_screening_excluded_from_new_row=True,
            classifier_nested_subsets_not_added_as_remote_costs=True,
            whole_thesis_cost=False,
            global_ledger_changed=False,
        ),
    )
    files = [
        OUT / name
        for name in (
            "analysis_implementation.json",
            "paired_costs.json",
            "repair_concentration.json",
            "resource_summary.json",
            "practical_scenarios.json",
            "resource_addition.json",
        )
    ] + list(drawdir.glob("*.npy"))
    p.save(
        "calculated.json",
        dict(
            status="part9_resource_and_practical_calculations_complete",
            created_utc=a.utc(),
            assembled_sha256=a.sha(OUT / "assembled.json"),
            sources={
                a.rel(Path(__file__)): a.sha(Path(__file__)),
                a.rel(Path(k.__file__)): a.sha(Path(k.__file__)),
                a.rel(p.P8 / "rates.json"): a.sha(p.P8 / "rates.json"),
            },
            artifacts={a.rel(path): a.sha(path) for path in files},
            paired_bootstrap_sets=4,
            bootstrap_scalars=1920000,
            resource_groups=len(groups),
            inputs=15758,
            outputs=189096,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
            new_primary_families=0,
        ),
    )
    print(
        "Resource comparisons, practical scenarios and staged ledger row complete.",
        flush=True,
    )


if __name__ == "__main__":
    with a.offline():
        main()

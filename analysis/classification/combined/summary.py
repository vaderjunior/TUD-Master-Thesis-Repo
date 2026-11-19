import argparse
from copy import deepcopy
from itertools import product
from pathlib import Path
import math
import statistics
from analysis.classification.combined import prepare as a

ROOT = a.ROOT
BASE = ROOT / "experiments/astra/sq2_combined_final_20261104_v1"
R = a.REPORTS
E = ROOT / "experiments/astra"
P = E / "sq2_combined_performance_20261104_v1"
Q = E / "sq2_combined_quality_mechanisms_20261104_v1"
BR = E / "sq2_broad_robustness_review_20261104_v1"
BC = E / "sq2_broad_resources_review_20261104_v1"
GR = E / "sq2_gpu_robustness_review_20261104_v1"
GC = E / "sq2_gpu_resources_review_20261104_v1"
GM = E / "sq2_gpu_main_review_20261103_v1/runs/9deace7a4235"
M = E / "sq2_matched_review_20261030_v1"
S = E / "roberta_short_review_20261020_v1"
CPU = E / "sq2_cpu_review_20261012_v1"
read = a.read


def save(name, value):
    path = BASE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(a.g.encoded(value))


def relation(interval):
    if interval is None:
        return "unsupported"
    lo, hi = interval
    a.require(math.isfinite(lo) and math.isfinite(hi) and lo <= hi, "Invalid interval")
    return (
        "above_zero"
        if lo > 0
        else "below_zero" if hi < 0 else "crosses_or_touches_zero"
    )


def original_interval(v):
    return v.get("family_interval") or v.get("ci95")


def contract():
    a.require(not BASE.exists(), "Preserve existing Part4")
    pp = [
        a.BASE / "part1_complete.json",
        P / "part2_complete.json",
        Q / "part3_complete.json",
        a.BASE / "inference_contracts.json",
        a.BASE / "study_inventory.json",
        a.BASE / "comparison_map.json",
        a.BASE / "training_information.json",
        a.BASE / "evaluation_alignment.json",
        P / "original_qwen_contrasts.json",
        Q / "original_quality_contrasts.json",
        Q / "matched_mechanism_connections.json",
        Q / "mechanism_evidence_map.json",
        Q / "aggregation_and_rationales.json",
        Q / "joint_quality.json",
        GM / "primary_comparisons.json",
        GR / "analysis/points.json",
        GR / "analysis/bootstrap_results.json",
        GR / "analysis/source_sensitivity.json",
        GR / "analysis/inventory.json",
        GR / "part7_complete.json",
        GR / "nesting_correction/review.json",
        GC / "part8_complete.json",
        GC / "analysis/fit_accounting.json",
        GC / "analysis/timeline_summary.json",
        GC / "analysis/storage_summary.json",
        GC / "context/baselines.json",
        GC / "context/historical_laptop.json",
        GC / "context/technical_and_supplement.json",
        M / "partial_programme_ledger.json",
        M / "metrics.json",
        M / "complete.json",
        M / "resources.json",
        S / "statistics.json",
        S / "resources.json",
        S / "complete.json",
        CPU / "statistics.json",
        CPU / "complete.json",
        BR / "sensitivities.json",
        BR / "part7_complete.json",
        BR / "original_coverage_and_bounds.json",
        BR / "repeat_comparisons.json",
        BR / "derived_findings.json",
        BR / "validity_inventory.json",
        BR / "exposure_review.json",
        BC / "ledger_entry_for_combined_synthesis.json",
        BC / "resource_summary.json",
        BC / "preparation_and_storage.json",
        BC / "part8_complete.json",
        E / "sq2_broad_synthesis_20261104_v1/part9_complete.json",
        E / "sq2_gpu_synthesis_20261104_v1/part9_complete.json",
        E / "sq3_final_synthesis_20261027_v1/resource_reconciliation.json",
        E / "sq3_final_synthesis_20261027_v1/publication_complete.json",
        ROOT
        / "Thesis_Log/astra_result/evidence/astra_master_research_journal_20261030_v21.json",
        ROOT / "Thesis_Log/astra_journal_assets/figure_sources_v21.json",
        ROOT / "scripts/astra_build_research_journal_v21.mjs",
        Path(__file__),
    ]
    pp += list(P.glob("endpoints/*/scores.json")) + list(Q.glob("endpoints/*.json"))
    names = [
        "astra_sq2_combined_synthesis_plan_20261104_v1.md",
        "astra_remaining_work_before_writing_20261104_v155.md",
        "astra_sq2_combined_comparison_setup_20261104_v1.md",
        "astra_sq2_combined_performance_errors_20261104_v2.md",
        "astra_sq2_combined_quality_mechanisms_20261104_v2.md",
        "astra_sq2_gpu_robustness_limitations_20261104_v1.md",
        "astra_sq2_gpu_resources_practical_value_20261104_v1.md",
        "astra_sq2_broad_robustness_20261104_v2.md",
        "astra_sq2_broad_resources_20261104_v1.md",
        "astra_sq2_matched_results_20261030_v1.md",
        "astra_sq2_type_update_results_20261103_v1.md",
        "astra_resource_evidence_20261030_v10.md",
        "astra_resource_evidence_20261027_v9.md",
        "astra_sq3_final_combined_synthesis_20261027_v1.md",
        "astra_sq1_pool_final_summary_20260928_v1.md",
        "astra_sq1_components_final_summary_20260924_v1.md",
    ]
    pp += [R / n for n in names]
    pp += [ROOT / x["source"] for x in read(a.BASE / "study_inventory.json")]
    sources = a.bind(sorted(set(pp)))
    BASE.mkdir()
    snapshots = {}
    for name in [
        "astra_master_research_journal.md",
        "astra_master_research_journal.html",
        "astra_session.md",
        "astra_review.md",
    ]:
        src = ROOT / "Thesis_Log" / name
        dst = BASE / "preimages" / name
        dst.parent.mkdir(exist_ok=True)
        with dst.open("xb") as f:
            f.write(src.read_bytes())
        snapshots[a.rel(src)] = {
            "sha256": a.g.sha(src),
            "copy": a.rel(dst),
            "bytes": src.stat().st_size,
        }
    save(
        "analysis_contract.json",
        dict(
            created_utc=a.utc(),
            scope="Combined SQ2 Part4: confidence, resources, final scientific answer and global reconciliation. Retrospective synthesis after Parts1-3; not preregistration.",
            sources=sources,
            preimages=snapshots,
            preserve=[
                "All original roles, families, masks, fixed macro labels, gates, missing outputs, repeats, pools, selected recipes and amendments.",
                "Original five initialization-only heads and separate five-update sensitivity.",
                "All earlier reports and navigation preimages.",
                "Fourteen remote programme row objects unchanged; append broader once, preserve omitted work and unknown outcomes.",
            ],
            questions=[
                "Which findings survive the available checks, and which uncertainty remains?",
                "Which practical costs are measured, with what non-additive boundaries?",
                "What bounded answer follows for the original SQ2 and the thesis aim?",
                "Are all accepted scientific tasks complete, separately from writing, backups and server release?",
            ],
            prohibited=[
                "Qwen or other model calls",
                "training, inference or weight loading",
                "new retrieval or embeddings",
                "new bootstrap draws, primary families or meta-analysis",
                "new model/cutoff/seed selection",
                "server actions, deletion or external messages",
                "writer-owned edits or Part5 writing handover",
            ],
            verification="Check sealed immutable outputs, independently reconcile interval signs/points, repeat summaries, resources, publication numbers and local links. No full raw-tree re-audit or archive rehash.",
            closure="Record scientific closure only after final review passes; Part5 handover and preservation/writing remain open.",
        ),
    )
    print("Part4 scope and preimages saved; original science unchanged.")


def confidence():
    broad = []
    for row in read(P / "original_qwen_contrasts.json"):
        v = row["original"]
        interval = original_interval(v)
        broad.append(
            dict(
                endpoint=row["endpoint"],
                left=v["left"],
                right=v["right"],
                role=v.get("role", "supporting"),
                difference=v.get("mean_difference"),
                interval=interval,
                confidence=(
                    v.get("family_confidence", 0.95)
                    if v.get("family_interval")
                    else 0.95
                ),
                relation=relation(interval),
                n=v.get("n_common_valid"),
                failure_bounds=v.get("full_roster_failure_difference_bounds"),
                original=v,
            )
        )
    sensitivity = []
    for v in read(BR / "sensitivities.json"):
        z = v["result"]
        ci = original_interval(z)
        sensitivity.append(
            dict(
                endpoint=v["endpoint"],
                left=v["left"],
                right=v["right"],
                role=v["role"],
                scenario=v["scenario"],
                difference=z.get("mean_difference"),
                interval=ci,
                relation=relation(ci),
                original_relation=relation(v["original_interval"]),
                classification_changed=relation(ci) != relation(v["original_interval"]),
                n=z.get("n_common_valid"),
                original_n=v["original_common_valid"],
                point_change=v["difference_change"],
                original=v,
            )
        )
    matched = []
    matched_robust = []
    for ep, v in read(M / "metrics.json")["results"].items():
        for control, z in v["contrasts"].items():
            if z.get("mean_difference") is None:
                continue
            ci = original_interval(z)
            primary = z.get("family_size") == 6
            matched.append(
                dict(
                    endpoint=ep,
                    control=control,
                    role="primary" if primary else "supporting",
                    difference=z["mean_difference"],
                    interval=ci,
                    confidence=(
                        z.get("family_confidence", 0.95)
                        if z.get("family_interval")
                        else 0.95
                    ),
                    relation=relation(ci),
                    n=z["n_common_valid"],
                    failure_bounds=z.get("full_roster_failure_difference_bounds"),
                    original=z,
                )
            )
            if control in v["robustness"]:
                rob = v["robustness"][control]
                strict = rob.get("three_valid_votes")
                matched_robust.append(
                    dict(
                        endpoint=ep,
                        control=control,
                        role="primary" if primary else "supporting",
                        original_relation=relation(ci),
                        strict_relation=(
                            relation(original_interval(strict)) if strict else None
                        ),
                        strict=strict,
                        finite_checks={
                            k: val for k, val in rob.items() if k != "three_valid_votes"
                        },
                    )
                )
    gpu = read(GM / "primary_comparisons.json")
    gpu_rows = []
    for v in gpu["comparisons"]:
        gpu_rows.append(
            dict(**v, role="primary", relation=relation(v["family_interval"]))
        )
    gpu_sens = []
    for ep, v in read(GR / "analysis/bootstrap_results.json").items():
        for scenario in ["alternative_fold", "core_excluded"]:
            for z in v[scenario]:
                gpu_sens.append(
                    dict(
                        endpoint=ep,
                        scenario=scenario,
                        **z,
                        relation=relation([z["low"], z["high"]]),
                        confidence=0.95,
                        role="supporting sensitivity"
                    )
                )
    finite = []
    for ep, v in read(GR / "analysis/points.json").items():
        for z in v["contrasts"]:
            finite.append(
                dict(
                    endpoint=ep,
                    **deepcopy(z),
                    interpretation="Finite saved-seed/pool summaries; not population CIs or probabilities."
                )
            )
    earlier = []
    for ep, v in read(S / "statistics.json")["contrasts"].items():
        if "primary" not in v:
            continue
        z = v["primary"]["unstratified"]
        earlier.append(
            dict(
                study="expanded_shorter",
                endpoint=ep,
                contrast="expanded_minus_original",
                difference=z["difference"],
                interval=z["ci9875"],
                confidence=0.9875,
                relation=relation(z["ci9875"]),
                role="primary",
                original=v,
            )
        )
    for ep, v in read(CPU / "statistics.json")["primary"].items():
        z = v["primary"]
        earlier.append(
            dict(
                study="native_full_cpu",
                endpoint=ep,
                contrast="full_BGE_refit_minus_retained_full_neural_selected",
                difference=z["difference"],
                interval=z["ci9875"],
                confidence=0.9875,
                relation=relation(z["ci9875"]),
                role="primary",
                original=v,
            )
        )
    save(
        "confidence_inventory.json",
        dict(
            broad=broad,
            matched=matched,
            gpu=gpu_rows,
            earlier=earlier,
            original_family_contracts=read(a.BASE / "inference_contracts.json"),
            other_study_routes=read(a.BASE / "study_inventory.json"),
            cross_study_points="Descriptive common-nine means; no new inferential family, pooled evidence, overall win count, or equivalence inference.",
        ),
    )
    save(
        "sensitivity_inventory.json",
        dict(
            broad=sensitivity,
            matched=matched_robust,
            gpu=gpu_sens,
            gpu_finite=finite,
            warning="Selected vote/repair cohorts and alternative folds are sensitivities. Preserve original families and comparators.",
        ),
    )
    changes = [
        {
            k: v[k]
            for k in [
                "scenario",
                "endpoint",
                "left",
                "right",
                "role",
                "original_relation",
                "relation",
                "n",
                "original_n",
                "difference",
                "interval",
            ]
        }
        for v in sensitivity
        if v["classification_changed"]
    ]
    save(
        "confidence_findings.json",
        dict(
            broad_original_rows=len(broad),
            broad_scorable=sum(x["difference"] is not None for x in broad),
            broad_sensitivity_rows=len(sensitivity),
            broad_classification_changes=changes,
            broad_required_changes=[
                x
                for x in changes
                if x["scenario"]
                in [
                    "required_sampling",
                    "exclude_replay",
                    "exclude_whitespace",
                    "exclude_core",
                ]
            ],
            broad_full_zero_changes=[x for x in changes if x["right"] == "zero_shot"],
            gpu_primary_rows=len(gpu_rows),
            matched_primary_rows=sum(x["role"] == "primary" for x in matched),
            no_new_draws=True,
            no_new_predictions=True,
        ),
    )


def resources():
    prior = read(M / "partial_programme_ledger.json")
    entry = read(BC / "ledger_entry_for_combined_synthesis.json")
    acc = entry["generated_accounting"]["accounting"]
    usage = entry["generated_accounting"]["usage"]
    extra = dict(
        programme="SQ2 broader full/small/no retrieval",
        ordinary_slots=acc["ordinary_logical_calls"],
        repair_slots=acc["repair_logical_calls"],
        retry_or_recovered_failure_attempts=acc["retry_attempts"],
        attempt_slots=acc["post_attempts"],
        successful_responses=acc["http_success_with_text"],
        reported_tokens=usage["total_tokens"],
        source="Thesis_Log/astra_result/astra_sq2_broad_resources_20261104_v1.md",
        unique_receipt=entry["unique_final_receipt"],
        receipt_sha256=entry["receipt_sha256"],
    )
    a.require(
        all(x["programme"] != extra["programme"] for x in prior["programmes"]),
        "Broader already present",
    )
    programmes = deepcopy(prior["programmes"]) + [extra]
    totals = {k: sum(x[k] for x in programmes) for k in prior["totals"]}
    save(
        "partial_programme_ledger.json",
        dict(
            status="fifteen_programme_partial_ledger",
            programmes=programmes,
            totals=totals,
            previous_ledger=a.rel(M / "partial_programme_ledger.json"),
            previous_sha256=a.g.sha(M / "partial_programme_ledger.json"),
            inherited_fourteen_programme_objects_unchanged=programmes[:-1]
            == prior["programmes"],
            added_once=extra["programme"],
            administrative_unknown_outcomes_in_inherited_reserved_row=8,
            scope=prior["scope"],
            new_Qwen_calls=0,
            new_training_fits=0,
        ),
    )
    sq3 = read(E / "sq3_final_synthesis_20261027_v1/resource_reconciliation.json")
    fit = read(GC / "analysis/fit_accounting.json")
    timeline = read(GC / "analysis/timeline_summary.json")
    storage = read(GC / "analysis/storage_summary.json")
    groups = read(BC / "resource_summary.json")["groups"]
    condition = [v for v in groups if v["group"].startswith("condition/")]
    gp = {k: v["total"] for k, v in fit["phases"].items()}
    derived = dict(
        search_worker_hours=gp["search"]["total_worker_seconds"] / 3600,
        final_fit_worker_hours=gp["final"]["total_worker_seconds"] / 3600,
        search_share_of_canonical_fit_seconds=gp["search"]["total_worker_seconds"]
        / (gp["search"]["total_worker_seconds"] + gp["final"]["total_worker_seconds"]),
        elapsed_days=timeline["summary"]["calendar_seconds"] / 86400,
        active_hours=timeline["summary"]["active_seconds"] / 3600,
        gap_hours=timeline["summary"]["gap_seconds"] / 3600,
        total_payload_decimal_GB=storage["total_payload_bytes"] / 1e9,
        unique_logical_decimal_GB=storage["unique_logical_bytes"] / 1e9,
        new_two_sq2_campaigns_reported_tokens=prior["programmes"][-1]["reported_tokens"]
        + extra["reported_tokens"],
    )
    save(
        "resource_reconciliation.json",
        dict(
            remote_totals=totals,
            broad_entry=entry,
            broad_condition_costs=condition,
            historical_SQ3=deepcopy(sq3),
            gpu_main=dict(
                fit_totals=gp,
                executed_search_subsets=fit["executed_search_subsets"],
                timeline=timeline["summary"],
                telemetry=timeline["telemetry"],
                weighting=timeline["weighting"],
                storage=storage,
                final_four_head_pipeline=fit["pipeline_summary"],
                inference_model_passes=fit["inference_model_passes"],
                inference_head_rows=fit["inference_head_rows"],
                unique_final_models=gp["final"]["timed_fits"],
                family_seed_head_assignments=gp["final"]["timed_fits"]
                + len(fit["reused_assignments"]),
            ),
            gpu_technical_and_supplement=read(
                GC / "context/technical_and_supplement.json"
            ),
            earlier_laptop=read(GC / "context/historical_laptop.json"),
            baseline_context=read(GC / "context/baselines.json"),
            earlier_short=read(S / "resources.json"),
            derived=derived,
            concurrent_work_warning="Wall spans, overlapping client/request/worker timers and preparation clocks are not additive or controlled deployment latency.",
            storage_warning="271 GB cumulative offloaded payload is not simultaneous server usage. 250 decimal GB is a soft project policy; no current server check.",
            missing_measurements=[
                "Failed-attempt backend tokens/work",
                "Provider weight/config digest",
                "Power, energy, emissions and invoice",
                "Complete early fitting and preparation history",
                "Complete human/author labour",
                "Matched end-to-end deployment latency and future service load",
            ],
            no_new_experiment_cost=True,
        ),
    )


def analyze():
    c = read(BASE / "analysis_contract.json")
    a.verify(c["sources"])
    confidence()
    resources()

    focus = [
        "qwen/rag_full",
        "qwen/rag_small",
        "qwen/zero_shot",
        "gpu/5000/tuned/selected",
        "gpu/5000/fixed_balanced/selected",
        "gpu/5000/bge/selected",
        "cpu/full_neural/selected",
    ]
    points = {}
    for file in sorted(P.glob("endpoints/*/scores.json")):
        v = read(file)
        groups = v["common_nine"]["groups"]
        points[file.parent.name] = {k: groups[k] for k in focus if k in groups}
    quality = {}
    for file in sorted(Q.glob("endpoints/*.json")):
        v = read(file)
        if "common_nine" not in v:
            continue
        quality[v["endpoint"]] = {
            k: v["common_nine"]["groups"][k]
            for k in focus
            if k in v["common_nine"]["groups"]
        }
    joint = {
        ep: dict(
            n=v["common_nine_n"],
            eligible=v["eligible"],
            fields=v["fields"],
            groups={
                k: v["common_nine"]["groups"][k]
                for k in focus
                if k in v["common_nine"]["groups"]
            },
        )
        for ep, v in read(Q / "joint_quality.json").items()
    }
    save(
        "integrated_result_views.json",
        dict(
            performance=points,
            quality=quality,
            joint=joint,
            role="Previously checked descriptive common-nine views, with original full-roster/formal comparisons retained in Parts2/3. No cross-study CI or ranking claim.",
        ),
    )
    a.verify(c["sources"])
    save(
        "analysis_complete.json",
        dict(
            created_utc=a.utc(),
            outputs=a.bind(
                sorted(
                    p
                    for p in BASE.glob("*.json")
                    if p.name
                    not in ["analysis_contract.json", "analysis_complete.json"]
                )
            ),
            source_count=len(c["sources"]),
            new_draws=0,
            new_model_calls=0,
        ),
    )
    print("Confidence/resource integration saved; independent verification follows.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["freeze", "analyze"])
    args = ap.parse_args()
    a.g.offline()
    if args.mode == "freeze":
        contract()
    else:
        analyze()

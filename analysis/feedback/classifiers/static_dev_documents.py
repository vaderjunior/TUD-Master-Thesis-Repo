from collections import Counter
from pathlib import Path
import json

from analysis.feedback.classifiers import static_dev_results as a

REPORT = a.ROOT / "Thesis_Log/astra_result/astra_sq3_linear_dev_results_20261020_v1.md"
HANDOFF = (
    a.ROOT / "Thesis_Log/astra_result/astra_sq3_available_synthesis_20261020_v1.md"
)


def save(path, value):
    path = Path(path).resolve()
    a.need(
        path.is_relative_to(a.OUT) or path in (REPORT, HANDOFF),
        "Unexpected document target",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        f.write(
            value
            if isinstance(value, str)
            else json.dumps(
                value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False
            )
            + "\n"
        )


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join("---" for _ in headers) + "|",
        ]
        + ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    )


def pct(x, sign=False):
    return f"{100*x:+.3f}" if sign else f"{100*x:.2f}"


def interval(x):
    return "[" + ", ".join(pct(v, True) for v in x) + "]"


def main():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from threadpoolctl import threadpool_limits

    with a.offline(), threadpool_limits(limits=1):
        checked = a.read(a.OUT / "independent_review.json")
        a.need(not checked["unresolved_discrepancies"], "Unresolved analysis")
        a.verify(checked["reviewed_sources"])
        a.verify(checked["code_bindings"])
        results = a.read(a.OUT / "analysis.json")["results"]
        index = {
            (
                r["mode"],
                r["group"],
                r["view"],
                r["endpoint"],
                r["variant"],
                r["cutoff"],
            ): r
            for r in results
        }

        def get(
            group="error_balanced",
            endpoint="FPR",
            view="common",
            variant="all_permitted",
            cutoff=0.6,
            mode="reserved5",
        ):
            return index[mode, group, view, endpoint, variant, cutoff]

        mainrows = []
        for endpoint in ("FPR", "FNR"):
            r = get(endpoint=endpoint)
            e = r["estimates"]
            mainrows.append(
                [
                    endpoint,
                    r["common_n"],
                    pct(e["linear_baseline"]) + " → " + pct(e["linear_updated"]),
                    pct(e["rag_reference"]) + " → " + pct(e["rag_updated"]),
                    pct(e["linear_change"], True),
                    pct(e["rag_change"], True),
                    pct(e["difference_in_change"], True)
                    + " "
                    + interval(r["interval95"]["difference_in_change"]),
                ]
            )
        headline = table(
            [
                "Endpoint",
                "Common texts",
                "Classifier before → after (%)",
                "Qwen before → after (%)",
                "Classifier change (pp)",
                "Qwen change (pp)",
                "Difference in change (pp; supporting 95%)",
            ],
            mainrows,
        )
        policyrows = []
        for group in a.GROUPS:
            f, n = get(group=group), get(group=group, endpoint="FNR")
            policyrows.append(
                [
                    group.replace("_", " "),
                    f"{f['common_n']}/{n['common_n']}",
                    pct(f["estimates"]["linear_change"], True),
                    pct(f["estimates"]["rag_change"], True),
                    pct(n["estimates"]["linear_change"], True),
                    pct(n["estimates"]["rag_change"], True),
                ]
            )
        policy = table(
            [
                "40-label pool",
                "Benign/hateful common N",
                "Classifier ΔFPR",
                "Qwen ΔFPR",
                "Classifier ΔFNR",
                "Qwen ΔFNR",
            ],
            policyrows,
        )
        sensrows = []
        for view in a.VIEWS:
            for endpoint in ("FPR", "FNR"):
                r = get(endpoint=endpoint, view=view)
                sensrows.append(
                    [
                        view.replace("_", " "),
                        endpoint,
                        r["common_n"],
                        pct(r["estimates"]["difference_in_change"], True),
                        interval(r["interval95"]["difference_in_change"]),
                    ]
                )
        sensitivity = table(
            [
                "View",
                "Endpoint",
                "Common N",
                "Difference in change (pp)",
                "Supporting 95%",
            ],
            sensrows,
        )
        variantrows = []
        for v in a.VARIANTS:
            for cutoff in a.CUTOFFS:
                f, n = get(variant=v, cutoff=cutoff), get(
                    endpoint="FNR", variant=v, cutoff=cutoff
                )
                variantrows.append(
                    [
                        v,
                        f">{cutoff}",
                        pct(f["estimates"]["linear_updated"]),
                        pct(n["estimates"]["linear_updated"]),
                        pct(f["estimates"]["difference_in_change"], True),
                        pct(n["estimates"]["difference_in_change"], True),
                        interval(n["interval95"]["difference_in_change"]),
                    ]
                )
        variants = table(
            [
                "TRAIN variant",
                "Fixed cutoff",
                "Updated FPR %",
                "Updated FNR %",
                "ΔΔ FPR pp",
                "ΔΔ FNR pp",
                "ΔΔ FNR supporting 95%",
            ],
            variantrows,
        )
        boundsrows = []
        for view in ("common", "strict", "excluded", "strict_excluded"):
            for endpoint in ("FPR", "FNR"):
                r = get(endpoint=endpoint, view=view)
                boundsrows.append(
                    [
                        view,
                        endpoint,
                        r["eligible_n"],
                        "/".join(map(str, r["missing_bounds"]["missing_real_outputs"])),
                        interval(r["missing_bounds"]["difference_in_change"]),
                    ]
                )
        bounds = table(
            [
                "View",
                "Endpoint",
                "Full eligible N",
                "Missing reference/update outputs",
                "Opposing missing-output ΔΔ bounds (pp)",
            ],
            boundsrows,
        )
        cases = list(a.rows(a.OUT / "all_DEV_cases.jsonl"))
        shifts = []
        changes = []
        for v in a.VARIANTS:
            for group in a.GROUPS:
                for seed in a.SEEDS:
                    pool = group + "_s" + str(seed)
                    b = np.array([r["linear"][v]["reference"] for r in cases])
                    u = np.array([r["linear"][v][pool] for r in cases])
                    shift = {
                        "variant": v,
                        "pool": pool,
                        "texts": 675,
                        "median_absolute_probability_change": float(
                            np.median(abs(u - b))
                        ),
                        "maximum_absolute_probability_change": float(
                            np.max(abs(u - b))
                        ),
                        "mean_signed_probability_change": float(np.mean(u - b)),
                        "cutoffs": {},
                    }
                    for c in a.CUTOFFS:
                        changed = [i for i in range(675) if (b[i] > c) != (u[i] > c)]
                        fixed = [
                            i for i in changed if (u[i] > c) == cases[i]["gold"]["hate"]
                        ]
                        shift["cutoffs"][str(c)] = {
                            "changed": len(changed),
                            "fixed": len(fixed),
                            "regressed": len(changed) - len(fixed),
                        }
                        changes.extend(
                            {
                                "key": cases[i]["key"],
                                "variant": v,
                                "pool": pool,
                                "cutoff": c,
                                "source": cases[i]["source"],
                                "gold_hate": cases[i]["gold"]["hate"],
                                "before_probability": float(b[i]),
                                "after_probability": float(u[i]),
                                "fixed": i in fixed,
                            }
                            for i in changed
                        )
                    shifts.append(shift)
        save(
            a.OUT / "probability_shift_description.json",
            {
                "status": "additional_descriptive_summary_no_new_inference_or_policy_selection",
                "timing": "summarized after paired outcomes; all 36 saved updates, both cutoffs, all 675 DEV texts",
                "rows": shifts,
                "all_changed_decision_occurrences": changes,
            },
        )

        for row in shifts:
            v, pool = row["variant"], row["pool"]
            diffs = sorted(
                abs(c["linear"][v][pool] - c["linear"][v]["reference"]) for c in cases
            )
            a.need(
                diffs[337] == row["median_absolute_probability_change"]
                and max(diffs) == row["maximum_absolute_probability_change"],
                "Probability summary",
            )
            for cut, counts in row["cutoffs"].items():
                c = float(cut)
                xx = [
                    r
                    for r in cases
                    if (r["linear"][v][pool] > c) != (r["linear"][v]["reference"] > c)
                ]
                a.need(
                    len(xx) == counts["changed"]
                    and sum((r["linear"][v][pool] > c) == r["gold"]["hate"] for r in xx)
                    == counts["fixed"],
                    "Independent probability transition census",
                )
        principal_shifts = [
            s
            for s in shifts
            if s["variant"] == "all_permitted"
            and s["pool"].startswith("error_balanced")
        ]
        shifts_table = table(
            [
                "Balanced error pool",
                "Median |Δ probability|",
                "Max |Δ probability|",
                "Changed / fixed / regressed at >0.6; all 675",
            ],
            [
                [
                    s["pool"],
                    f"{s['median_absolute_probability_change']:.6f}",
                    f"{s['maximum_absolute_probability_change']:.6f}",
                    " / ".join(
                        str(s["cutoffs"]["0.6"][k])
                        for k in ("changed", "fixed", "regressed")
                    ),
                ]
                for s in principal_shifts
            ],
        )

        complete_rows = []
        for r in results:
            e = r["estimates"]
            complete_rows.append(
                [
                    r["mode"],
                    r["group"],
                    r["view"],
                    r["variant"],
                    r["cutoff"],
                    r["endpoint"],
                    f"{r['common_n']}/{r['eligible_n']}",
                    *[pct(e[k]) for k in a.ARMS],
                    *[
                        pct(e[k], True)
                        for k in ("linear_change", "rag_change", "difference_in_change")
                    ],
                    (
                        interval(r["interval95"]["difference_in_change"])
                        if "interval95" in r
                        else "descriptive only"
                    ),
                ]
            )
        save(
            a.OUT / "all_rate_tables.md",
            "# All supporting DEV rate comparisons\n\n"
            "Percentages for absolute rates; percentage points for changes. Every 95% interval is "
            "supporting and unadjusted. Different policy masks can change reference rates; use paired "
            "within-row comparisons. Masked RAG deliberately lacks correction labels.\n\n"
            + table(
                [
                    "Mode",
                    "Pool",
                    "View",
                    "TRAIN",
                    "Cutoff >",
                    "Rate",
                    "Common/eligible",
                    "Linear base %",
                    "Linear update %",
                    "Qwen base %",
                    "Qwen update %",
                    "Linear Δ pp",
                    "Qwen Δ pp",
                    "ΔΔ pp",
                    "ΔΔ supporting 95%",
                ],
                complete_rows,
            )
            + "\n",
        )
        figdir = a.OUT / "figures"
        figdir.mkdir(exist_ok=False)
        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "svg.fonttype": "none",
            }
        )
        fig, axes = plt.subplots(1, 2, figsize=(11, 5.4), constrained_layout=True)
        for ax, endpoint, title in zip(
            axes,
            ("FPR", "FNR"),
            ("Benign texts flagged as hate", "Hateful texts missed"),
        ):
            r = get(endpoint=endpoint)
            names = [
                "Classifier\nbefore",
                "Classifier\nupdated",
                "Qwen\nbefore",
                "Qwen\nupdated",
            ]
            vals = np.array([100 * r["estimates"][k] for k in a.ARMS])
            ci = 100 * np.array([r["interval95"][k] for k in a.ARMS])
            bars = ax.bar(
                range(4),
                vals,
                color=["#79a7c9", "#245b84", "#e9bd91", "#b96124"],
                width=0.66,
            )
            ax.errorbar(
                range(4),
                vals,
                yerr=np.maximum(
                    0, np.column_stack([vals - ci[:, 0], ci[:, 1] - vals]).T
                ),
                fmt="none",
                color="#263746",
                capsize=4,
                lw=1.1,
            )
            ax.bar_label(
                bars, labels=[f"{x:.2f}%" for x in vals], padding=7, fontsize=10
            )
            ax.set_xticks(range(4), names)
            ax.set_ylim(0, 56)
            ax.set_ylabel("Error rate (%) · lower is better")
            ax.set_title(f"{title}\nN = {r['common_n']} common texts", pad=14)
            ax.grid(axis="y", alpha=0.15)
            ax.set_axisbelow(True)
        fig.suptitle(
            "Same 40-label feedback pools: different error trade-offs\n"
            "Saved BGE linear head (>0.6) and reserved-slot Qwen · exposed DEV · supporting 95% intervals",
            fontsize=13,
        )
        for ext in ("png", "svg"):
            fig.savefig(
                figdir / f"paired_error_rates.{ext}", dpi=180, bbox_inches="tight"
            )
        plt.close(fig)
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.3), constrained_layout=True)
        labels = [
            g.replace("error", "Earlier errors")
            .replace("correct", "Already correct")
            .replace("random", "Random")
            .replace("_positive", "\n40 hateful")
            .replace("_balanced", "\n20 hateful / 20 benign")
            for g in a.GROUPS
        ]
        for ax, endpoint in zip(axes, ("FPR", "FNR")):
            for key, color, offset, label in (
                ("linear_change", "#245b84", -0.13, "Classifier update"),
                ("rag_change", "#b96124", 0.13, "Qwen update"),
            ):
                rr = [get(group=g, endpoint=endpoint) for g in a.GROUPS]
                x = np.array([100 * r["estimates"][key] for r in rr])
                ci = 100 * np.array([r["interval95"][key] for r in rr])
                yy = np.arange(6) + offset
                ax.errorbar(
                    x,
                    yy,
                    xerr=np.maximum(0, np.column_stack([x - ci[:, 0], ci[:, 1] - x]).T),
                    fmt="o",
                    color=color,
                    label=label,
                    capsize=3,
                )
            ax.axvline(0, color="#60717e", lw=0.8)
            ax.set_yticks(range(6), labels)
            ax.invert_yaxis()
            ax.set_xlabel("Change in error rate (percentage points)")
            ax.set_title(
                "False-positive rate" if endpoint == "FPR" else "False-negative rate"
            )
            ax.grid(axis="x", alpha=0.12)
            ax.legend(loc="lower right", fontsize=9)
        fig.suptitle(
            "All fixed feedback policies · all-permitted TRAIN · strict probability >0.6\n"
            "Supporting 95% cluster intervals; fixed pools/repeats are dependent",
            fontsize=13,
        )
        for ext in ("png", "svg"):
            fig.savefig(figdir / f"policy_changes.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)
        save(
            a.OUT / "display_tables.json",
            {
                "headline": headline,
                "policies": policy,
                "sensitivity": sensitivity,
                "variants": variants,
                "bounds": bounds,
                "probability_shifts": shifts_table,
            },
        )
        resource = a.read(a.OUT / "fit_resources.json")
        jobs = resource["jobs"]
        baselines = [j for j in jobs.values() if j["baseline_reproduced"]]
        updates = [j for j in jobs.values() if not j["baseline_reproduced"]]
        resource_table = table(
            ["Saved computation", "Unique fits", "Recorded fit minutes"],
            [
                [
                    "Baseline reproduction",
                    2,
                    f"{sum(j['timing_seconds']['fit_seconds'] for j in baselines)/60:.3f}",
                ],
                [
                    "Updates",
                    36,
                    f"{sum(j['timing_seconds']['fit_seconds'] for j in updates)/60:.3f}",
                ],
                [
                    "Total",
                    38,
                    f"{resource['recorded_resources']['fit_seconds']/60:.3f}",
                ],
            ],
        )
        report = f"""# SQ3: using the same feedback to update a small classifier — completed DEV analysis

20 October 2026. Supporting analysis complete; broader transfer and cumulative-round comparisons remain pending. All original evidence is preserved. No model call, fit, embedding or new annotation was made.

The saved classifier and Qwen make different errors on these 675 exposed English DEV texts. The classifier has fewer false positives but substantially more missed hate. Appending the same forty source labels changes its decisions very little. Qwen's reserved correction changes errors more, increasing false positives while reducing missed hate. These results do not establish one overall adaptation winner.

## Exact main comparison

Principal saved setting: BGE features frozen; binary logistic head trained on all 55,753 permitted English TRAIN texts; strict probability >0.6. Each update appends one of three fixed balanced earlier-error pools of forty labels. Qwen uses four ordinary examples plus one correction selected from that same pool, compared with a fresh five-example reference. Pool seeds are 2026100441, 2026100442 and 2026100443.

{headline}

Changes are after minus before. Difference in change is **(classifier update − classifier baseline) − (Qwen update − Qwen reference)**. Negative means the classifier's change is more favourable for that error rate. A favourable relative change can arise because the other method worsens; it does not mean the classifier improved. In particular, its mean FPR change here is exactly zero.

The difference-in-change FPR interval excludes zero in this supporting comparison. The FNR interval includes zero: +1.451 pp [−0.0238, +2.9266]. Do not round its lower endpoint into an established difference. Updated absolute differences are −23.802 pp FPR and +22.436 pp FNR. The large opposing absolute gaps make a single-method superiority statement inappropriate.

![Paired absolute error rates](../../experiments/astra/sq3_linear_dev_results_20261020_v1/figures/paired_error_rates.png)

## What was compared and verified

The original classifier protocol was fixed on 12 October, independently of the later Qwen outcomes. Its 38 saved fits (two reproduced baselines and 36 updates) had already passed numerical verification on 14 October: 624,454 probabilities reconstructed and 1,248,908 saved decision bits checked. This analysis rechecked the bound receipts and saved arrays, aligned exact text, ID, gold and cluster identity for the 675 DEV rows, and reused the completed, automatically audited 149,850-record Qwen assembly. The ongoing 15,758-input transfer outcomes were not opened or scored.

This supporting DEV implementation was specified on 20 October **after the completed Qwen findings were known**, before joint classifier/Qwen scoring. It is not a new prospective confirmatory analysis. All 1,152 rate views are published: six policy/composition groups, four Qwen context modes, two TRAIN variants, two cutoffs, six validity/overlap views and two endpoints. Reserved-slot comparisons receive 10,000-draw paired original-cluster 95% intervals; other modes are descriptive. These intervals are unadjusted, conditional on these pools/repeats, and cannot replace the original broader-study four-endpoint primary family.

The independent review reconstructs every point, common mask, opposing missing-output bound, 36,864 diagnostic scalars and **11,520,000 bootstrap scalars**. It uses separate Fraction arithmetic, cluster-multiplicity dot products and manual percentile interpolation. No unresolved numerical discrepancy remains. The original extra full raw replay stays intentionally incomplete; prior final audits and completed reviews are reused.

## All feedback policies

Every row averages three fixed source pools and three Qwen decoder repeats within each text. The classifier is deterministic; three pool fits are not decoder repetitions. The reference has three real Qwen outputs, not nine independent copies. Different groups can have slightly different common-valid masks, so reference rates must not be subtracted across rows.

{policy}

![All policy changes](../../experiments/astra/sq3_linear_dev_results_20261020_v1/figures/policy_changes.png)

At the principal cutoff, classifier changes across these six pools are small (within about a quarter percentage point for both endpoints); Qwen changes are larger. Hateful-only Qwen corrections show a larger false-positive increase and missed-hate decrease than the main balanced pool. This supports the already documented trade-off; it selects no new winning policy and does not change the running transfer design. Already-correct and random controls remain visible.

All natural-insertion, six-example and label-masked context results are retained in the complete tables. The label-masked Qwen condition withholds labels while the classifier receives them: it is a context diagnostic, not an equal-information competition. The ordinary four-example reference remains covered by the original context analysis and is not a classifier update.

## Robustness, thresholds and source dependence

{sensitivity}

For the principal TRAIN/cutoff setting, all six FPR difference-in-change intervals remain negative. All six FNR intervals include zero. Strict voting removes inputs unless every relevant Qwen output has three valid votes. Replay exclusions remove the same six predeclared texts before validity. The one-per-text-cluster representative is chosen by the inherited fixed hash before validity; 675 inputs belong to 674 clusters.

{variants}

Lowering the already saved cutoff and using the matched-gradient TRAIN partition can change the apparent FNR difference-in-change inference. These are supporting sensitivity settings, not newly selected thresholds. The main comparison remains all-permitted TRAIN and >0.6. All hateful DEV inputs come from implicit_hate; this study cannot establish source-general hate recall. Benign-source, individual-pool, leave-pool and leave-original-batch points are retained, with empty source/endpoint cells explicitly undefined.

{bounds}

These are adverse assignments of missing outputs over the full eligible set, **not confidence intervals**. Missing reference outputs have their actual three-repeat denominator; updated Qwen outputs have nine across three pools. Boundaries do not reinterpret valid false predictions as missing. Pool and execution-batch omission ranges are likewise descriptive diagnostics, not new inferential intervals.

## Why the classifier barely changes

{shifts_table}

This complete descriptive census uses all 675 saved DEV texts, independent of each Qwen common-valid mask. It includes fixes and regressions even where a net rate change is zero. Probability changes and threshold crossings are directly observed; no claim about an internal causal mechanism is inferred.

Forty appended examples increase the principal TRAIN set from 55,753 to 55,793 (about 0.072% more rows), while one of five Qwen context positions is explicitly reserved for feedback. The models therefore use the same retained label budget in very different ways. The predeclared classifier uses an ordinary full refit with recomputed balanced class weights; it does not upweight the correction records or retune on their outcomes. No new classifier optimization is warranted by this result.

The feedback pool is selected using the RAG system's errors. The experiment asks how both systems use **that same RAG-selected information**, not how two independently optimized active-learning strategies compare. Retained labels, 1,024 screened opportunities, prior TRAIN labels, cached representations and online serving are separate quantities. Overlapping pools do not provide eighteen independent datasets.

## Recorded computation and preservation

{resource_table}

These fits already existed. Their monitored worker total is {resource['recorded_resources']['monitored_seconds']/60:.3f} minutes, including work beyond solver fitting; this is concurrent laptop timing, not a hardware-normalized deployment benchmark. This offline analysis adds zero fits and zero Qwen tokens. Prior embedding work and inherited source screening are not free; the eleven-programme remote ledger stays numerically unchanged.

The original Qwen assembly, 38 fit receipts/outputs, original protocols and running transfer dependencies passed readback. No active launcher, queue lock or live prediction was modified. The principal broad TEST/HateCheck difference-in-change family waits for the current transfer run. Cumulative classifier fits wait for the actual audited successive-round source pools. This closes only the supporting DEV analysis, not the whole classifier experiment or all of SQ3.

## Evidence

- [Implementation plan](../../experiments/astra/sq3_linear_dev_results_20261020_v1/analysis_plan.json), [all rate tables](../../experiments/astra/sq3_linear_dev_results_20261020_v1/all_rate_tables.md), [machine-readable estimates](../../experiments/astra/sq3_linear_dev_results_20261020_v1/analysis.json).
- [Independent numerical review](../../experiments/astra/sq3_linear_dev_results_20261020_v1/independent_review.json), [exact rational checks](../../experiments/astra/sq3_linear_dev_results_20261020_v1/exact_point_checks.json).
- [Source/pool/batch diagnostics](../../experiments/astra/sq3_linear_dev_results_20261020_v1/diagnostics.json), [all 675 cases](../../experiments/astra/sq3_linear_dev_results_20261020_v1/all_DEV_cases.jsonl), [all probability/decision changes](../../experiments/astra/sq3_linear_dev_results_20261020_v1/probability_shift_description.json).
- [Saved fit/resource attribution](../../experiments/astra/sq3_linear_dev_results_20261020_v1/fit_resources.json), [original classifier protocol](../../experiments/astra/sq3_linear_update_20261012_v1/STUDY_PROTOCOL.md), [original Qwen synthesis](astra_sq3_reserved_final_results_20261020_v1.md).
"""
        save(REPORT, report)
        handoff = f"""# Available SQ3 synthesis and staged writing handover

20 October 2026. The completed SQ3 evidence can now be integrated without waiting for the broader-transfer queue. This is an interim synthesis of available results, not final closure of all SQ3 experiments. The user sets a maximum twenty-day remaining experiment window, through **9 November 2026**, superseding 15 November. No new call, fit, annotation, writer-owned edit or acknowledgement was made.

## Connected answer

The system supports adaptation by changing external knowledge while Qwen's weights stay fixed. Earlier category-restoration experiments provide positive evidence within familiar categories. Correction experiments show why that technical capability does not guarantee useful transfer: an inserted correction must be retrieved, a delivered correction can shift the false-positive/false-negative balance, and adopting a correction on its own source does not show improved classification of other texts.

The two-person supplement adds real novice judgments and label disagreement. Human-derived corrections were adopted on the selected sources, but the main complete-answer agreement measure did not improve on the other reviewed texts. The same-label classifier comparison now shows that an inexpensive full linear refit can respond differently to the same information: very small marginal changes, fewer false alarms, but substantially more missed hate on this exposed DEV cohort. None of these studies establishes expert deployment over successive rounds.

## Claim map for writing now

| Claim | Evidence to read | Necessary limit |
|---|---|---|
| External category knowledge can restore performance without updating Qwen weights | Existing journal sections 8.7–8.8 and their original result reports | Familiar-category restoration; not proof of learning genuinely unseen categories |
| Ordinary insertion often fails to deliver feedback | [Five-slot report](astra_sq3_five_slot_results_20261012_v2.md) | Main delivery 13/2,025 pool–input combinations; 98.27% identical messages; exposed DEV |
| Reserving a correction slot changes the error trade-off | [Completed reserved-study synthesis](astra_sq3_reserved_final_results_20261020_v1.md), [its full writing handover](astra_sq3_reserved_writing_handoff_20261020_v1.md) | Original FPR/FNR family uses 97.5%; smaller FNR benefit reaches zero under strict/excluded sensitivity; no main complete-quality gain |
| Human-derived feedback is operationally usable | [Human result report](astra_sq3_human_feedback_results_20261020_v1.md), [all cases](astra_sq3_human_feedback_cases_20261020_v1.md) | Two novices, selected 24 texts; three sources are uptake only; seven different-message and fourteen identical-message transfer cases |
| Same labels can affect classifier and retrieval systems differently | [New saved-fit DEV comparison]({REPORT.name}) | Same RAG-selected forty labels, binary head only, fixed thresholds; supporting 95% comparisons; no overall winner |
| Wider transfer and successive-round durability remain unresolved | Current remaining-work roadmap and frozen protocols | Current transfer is still running; iterative results and cumulative fits do not exist |

## New compact classifier result

{headline}

This table is supporting DEV evidence. Keep the difference-in-change formula and absolute rates together. Do not replace the original Qwen interval with this newly computed 95% interval. Forty labels are only about 0.072% of the classifier's existing TRAIN set; correction upweighting or new tuning was not part of the fixed comparison.

## Concrete writing actions

Replace the stale pending reserved-study and human-feedback passages in `Results_SQ3.tex` with their reviewed findings. Add a concise saved-fit comparator paragraph and send complete policy/source/sensitivity tables to the supplement. Use one main trade-off visual, with the second policy figure as backup. Update methodology with the two distinct adaptation mechanisms (external retrieval update and frozen-feature head refit), while preserving their original information budgets and cohort limits.

Update Discussion, Conclusion and the presentation consistently: adaptation is technically demonstrated and its effectiveness depends on retrieval delivery, labels, task and error cost. Source uptake is not transfer; novices are not experts; fixed or simulated rounds do not establish longitudinal expert use. Preserve the positive category-restoration finding alongside the negative/mixed correction evidence.

The writing chat owns LaTeX/PPT editing, claim-source mapping, compile/render review and acknowledgement. The latest inspected writing status reported a tight word budget, so consolidate and replace provisional material instead of appending every experiment's detail. There is no need to wait for the running queue. Its later reviewed broad results, and any completed successive-round study, should receive separate additions.

## What still prevents final expanded SQ3 closure

1. Finish and review the running broader correction-transfer study, including its primary paired classifier analysis from the existing 38 fits.
2. Resolve a deadline-compatible prospective successive-round scope/serving plan, execute it only after its documented prerequisites, review retention/regression and compare the matching cumulative classifier fits.
3. Incorporate those reviewed results into a final SQ3 synthesis and resource/evidence handover. Do not mark unavailable results as finished.

Independent SQ2 preparation/training and writing may proceed after their own readiness/resource checks while remote Qwen runs. There is no blanket wait for all SQ3. The separate broad full/small/zero and matched-information campaigns are additional accepted evaluation work, not additional human annotation.

All original protocols and historical logs are preserved. The twenty-day limit changes planning priority; no future study has been silently reduced, cancelled or launched.
"""
        save(HANDOFF, handoff)
        save(
            a.OUT / "documents_prepared.json",
            {
                "created_utc": a.utc(),
                "documents": {a.rel(p): a.sha(p) for p in (REPORT, HANDOFF)},
                "figures": {a.rel(p): a.sha(p) for p in sorted(figdir.iterdir())},
                "display_rows": len(complete_rows),
                "figure_pairs": 2,
                "independent_probability_description_check": True,
                "visual_review_pending": True,
                "new_model_calls": 0,
                "new_fits": 0,
            },
        )
        print(
            json.dumps(
                {
                    "reports": [a.rel(REPORT), a.rel(HANDOFF)],
                    "tables": 1152,
                    "figures": 2,
                }
            )
        )


if __name__ == "__main__":
    main()

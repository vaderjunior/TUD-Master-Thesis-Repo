from pathlib import Path
import json
from analysis.knowledge.policy import common as c
from analysis.knowledge.policy import reader as reader

NAMES = {
    "en_hate": "English hate",
    "en_hate_type": "English hate types",
    "de_hate": "German hate",
    "de_legal": "German legal labels",
    "en_target_group": "English targets",
    "en_severity": "English severity",
}
LETTERS = dict(zip(c.CONDITIONS, "ABC"))


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        + ["| " + " | ".join((str(x) for x in row)) + " |" for row in rows]
    )


def f(x):
    return f"{x:.3f}"


def percent(x):
    return f"{100 * x:.2f}%"


def ci(x):
    return "[" + ", ".join((f"{100 * v:+.2f}" for v in x)) + "]"


def rep(x):
    return "/".join((str(int(v)) for v in x))


def main():
    base = reader.BASE
    latest = c.read(base / "analysis_complete.json")
    root = c.ROOT / latest["run"]
    complete = c.read(root / "complete.json")
    c.verify(complete["artifacts"])
    endpoints = {name: c.read(root / name / "results.json") for name in NAMES}
    contrasts = c.read(root / "contrasts.json")
    joint = c.read(root / "joint_correctness.json")
    chunks = [
        "# Complete-policy Part 3 — checked numerical appendix",
        "15 November 2026. Derived only from the saved, audited live study. This is a research analysis artifact, not an instruction to edit the thesis.",
        "A = original retrieved two definitions/two guidelines; B = all 21 definitions/original two guidelines; C = all 21 definitions/all 19 guidelines. Every condition retains the same five original examples. All result tables use the common-nine endpoint mask. Scores are calculated separately within each outer repeat and then averaged. A slash separates repeats 0/1/2. Counts with decimals are means across repeats, not fractional individual texts.",
        "## Populations and headline scores",
    ]
    chunks.append(
        table(
            [
                "Endpoint",
                "Eligible",
                "Common nine",
                "Macro A",
                "Macro B",
                "Macro C",
                "Micro A",
                "Micro B",
                "Micro C",
            ],
            [
                [
                    NAMES[name],
                    v["meta"]["n"],
                    v["common_nine"],
                    *[f(x) for x in v["point"]["macro_f1"]],
                    *(
                        [f(x) for x in v["point"]["micro_f1"]]
                        if "micro_f1" in v["point"]
                        else ["—"] * 3
                    ),
                ]
                for name, v in endpoints.items()
            ],
        )
    )
    chunks += [
        "Micro is reported for the three multilabel endpoints only. The fixed label scopes are identical to macro. English hate excludes the additional positive fine-label cases.",
        "## Every frozen comparison",
        "Differences and interval bounds below are percentage points (100 times the score difference). Ordinary 95% intervals are descriptive. Wider intervals are the frozen within-family intervals; separate families do not provide one error guarantee over the whole table. Supporting rows with no family keep only their descriptive 95% interval.",
    ]
    chunks.append(
        table(
            [
                "Endpoint",
                "Metric",
                "Contrast",
                "Difference (pp)",
                "95% interval (pp)",
                "Family",
                "Family interval (pp)",
            ],
            [
                [
                    NAMES[v["endpoint"]],
                    v["metric"],
                    LETTERS[v["left"]] + "−" + LETTERS[v["right"]],
                    f"{100 * v['estimate']:+.2f}",
                    ci(v["interval95"]),
                    v["family"] or "descriptive",
                    (
                        ci(v["family_interval"])
                        if v["family_interval"] is not None
                        else "—"
                    ),
                ]
                for v in contrasts
            ],
        )
    )
    chunks += [
        "## Hate decisions: false alarms and missed hate",
        "FPR = false positives / gold non-hate texts. FNR = false negatives / gold hateful texts. Positive precision and recall refer to the hateful class.",
    ]
    rows = []
    for endpoint in ("en_hate", "de_hate"):
        p = endpoints[endpoint]["point"]
        for i, label in enumerate("ABC"):
            q = p["confusion_repeats"]
            rows.append(
                [
                    NAMES[endpoint],
                    label,
                    p["negative_n"],
                    p["positive_n"],
                    rep(q["tn"][i]),
                    rep(q["fp"][i]),
                    rep(q["fn"][i]),
                    rep(q["tp"][i]),
                    percent(p["false_positive_rate"][i]),
                    percent(p["false_negative_rate"][i]),
                    f(p["positive_precision"][i]),
                    f(p["positive_recall"][i]),
                ]
            )
    chunks.append(
        table(
            [
                "Endpoint",
                "Method",
                "Negative N",
                "Positive N",
                "TN 0/1/2",
                "FP 0/1/2",
                "FN 0/1/2",
                "TP 0/1/2",
                "FPR",
                "FNR",
                "Precision",
                "Recall",
            ],
            rows,
        )
    )
    chunks += [
        "## Complete label sets and partial overlap",
        "Exactness for hate types and targets includes the hate gate. Jaccard compares the delivered label sets; empty/empty equals one, even if the gate is closed. The nonempty-gold column makes that convention visible. Legal scoring is independent of the hate gate. Extra and missing labels are mismatches against dataset annotations, not automatic proof of semantic error.",
    ]
    rows = []
    for endpoint in ("en_hate_type", "en_target_group", "de_legal"):
        p = endpoints[endpoint]["point"]
        for i, label in enumerate("ABC"):
            rows.append(
                [
                    NAMES[endpoint],
                    label,
                    p["n"],
                    f(p["exact"][i]),
                    f(p["jaccard"][i]),
                    p["nonempty_gold_n"],
                    f(p["nonempty_gold_jaccard"][i]),
                    f(p["extra_labels_per_text"][i]),
                    f(p["missing_labels_per_text"][i]),
                    f(p["predicted_labels_per_text"][i]),
                    f(p["gold_labels_per_text"]),
                    (
                        "—"
                        if "gate_closed_rate" not in p
                        else percent(p["gate_closed_rate"][i])
                    ),
                ]
            )
    chunks.append(
        table(
            [
                "Endpoint",
                "Method",
                "N",
                "Exact",
                "Jaccard",
                "Nonempty N",
                "Nonempty Jaccard",
                "Extra/text",
                "Missing/text",
                "Predicted/text",
                "Gold/text",
                "Gate closed",
            ],
            rows,
        )
    )
    chunks += [
        "## Severity distance and coverage",
        "A negative signed error is an underestimate. The emitted-subset MAE excludes absent severity and uses each setting’s own emitted texts; it must be read with coverage. Common-emitted MAE uses the same texts across all nine settings. Both are conditional measures, not substitutes for full-population accuracy.",
    ]
    p = endpoints["en_severity"]["point"]
    s = p["severity"]
    rows = []
    for i, label in enumerate("ABC"):
        rows.append(
            [
                label,
                p["n"],
                f(p["macro_f1"][i]),
                f(p["exact"][i]),
                percent(p["gate_closed_rate"][i]),
                rep(s["emitted_n_repeats"][i]),
                rep(s["gate_open_missing_n_repeats"][i]),
                f(s["mae_emitted"][i]),
                s["common_emitted_n"],
                f(s["common_emitted_mae"][i]),
            ]
        )
    chunks.append(
        table(
            [
                "Method",
                "N",
                "Macro",
                "Exact",
                "Gate closed",
                "Emitted N 0/1/2",
                "Open gate but absent 0/1/2",
                "Own-emitted MAE",
                "Common-emitted N",
                "Common-emitted MAE",
            ],
            rows,
        )
    )
    chunks.append(
        table(
            [
                "Method",
                "−2 bands 0/1/2",
                "−1 band 0/1/2",
                "Correct band 0/1/2",
                "+1 band 0/1/2",
                "+2 bands 0/1/2",
                "Gate closed 0/1/2",
            ],
            [
                [
                    label,
                    *[
                        rep(s["signed_error_counts_repeats"][str(d)][i])
                        for d in (-2, -1, 0, 1, 2)
                    ],
                    rep(p["gate_closed_counts"][i]),
                ]
                for i, label in enumerate("ABC")
            ],
        )
    )
    chunks += [
        "## Joint correctness on named annotation cohorts",
        "These cohorts overlap. They are not extra independent samples. Hate+types and hate+targets coincide with the corresponding gate-aware exact-set scores. No cohort supplies complete gold for all five output dimensions.",
    ]
    chunks.append(
        table(
            ["Cohort", "Eligible", "Common nine", "Exact A", "Exact B", "Exact C"],
            [
                [name, v["eligible"], v["n"], *[f(x) for x in v["exact"]]]
                for name, v in joint.items()
            ],
        )
    )
    chunks += [
        "## Agreement across the three outer repeats",
        "Agreement compares the predicted endpoint labels and, for fine fields, the hate gate. It ignores rationale wording and list order. The three repeat pairs are averaged within each text. Missing/missing is never counted as agreement.",
        "Stable correct means all three answers are correct. Stable wrong means all three produce the same incorrect endpoint state. Changing but always wrong means three incorrect answers that do not all agree. Variable correctness means one or two correct repeats. These four categories sum to one.",
    ]
    rows = []
    for endpoint, v in endpoints.items():
        p = v["point"]
        a = p["repeatability"]
        for i, label in enumerate("ABC"):
            rows.append(
                [
                    NAMES[endpoint],
                    label,
                    p["n"],
                    percent(a["pairwise_agreement"][i]),
                    percent(a["pairwise_correct_agreement"][i]),
                    percent(a["pairwise_wrong_agreement"][i]),
                    percent(a["three_way_agreement"][i]),
                    percent(a["stable_correct"][i]),
                    percent(a["stable_wrong"][i]),
                    percent(a["changing_but_always_wrong"][i]),
                    percent(a["variable_correctness"][i]),
                    (
                        "—"
                        if "all_three_gate_closed" not in a
                        else percent(a["all_three_gate_closed"][i])
                    ),
                ]
            )
    chunks.append(
        table(
            [
                "Endpoint",
                "Method",
                "N",
                "Pair agreement",
                "Correct pair",
                "Wrong pair",
                "Three-way",
                "Stable correct",
                "Stable wrong",
                "Changing/always wrong",
                "Variable correctness",
                "All gates closed",
            ],
            rows,
        )
    )
    chunks += [
        "## Individual fixes and new errors",
        "Each contrast pairs the same outer repeat index and the same text. These counts describe changes in gold correctness, not necessarily identical erroneous labels. Always-fixed/regressed requires that change in all three paired repeats.",
    ]
    rows = []
    for endpoint, v in endpoints.items():
        for x in v["point"]["transitions"]:
            rows.append(
                [
                    NAMES[endpoint],
                    "ABC"[x["left"]] + "−" + "ABC"[x["right"]],
                    rep(x["fixes_repeats"]),
                    rep(x["regressions_repeats"]),
                    f(x["net_correct_mean"]),
                    x["always_fixed"],
                    x["always_regressed"],
                ]
            )
    chunks.append(
        table(
            [
                "Endpoint",
                "Contrast",
                "Fixes 0/1/2",
                "New errors 0/1/2",
                "Net correct (mean)",
                "Always fixed",
                "Always regressed",
            ],
            rows,
        )
    )
    chunks += [
        "## Per-label descriptive results",
        "Support is the number of gold-positive texts on this endpoint’s common-nine population. Precision, recall and F1 are calculated per repeat and then averaged. Keep every frozen label; do not choose labels because their effects are attractive. Detailed interpretation belongs to Part 4.",
    ]
    rows = []
    for endpoint, v in endpoints.items():
        p = v["point"]
        for j, label in enumerate(v["meta"]["all_labels"]):
            for i, condition in enumerate("ABC"):
                rows.append(
                    [
                        NAMES[endpoint],
                        label,
                        condition,
                        p["gold_support"][j],
                        f(p["precision"][i][j]),
                        f(p["recall"][i][j]),
                        f(p["per_label_f1"][i][j]),
                    ]
                )
    chunks.append(
        table(
            ["Endpoint", "Label", "Method", "Support", "Precision", "Recall", "F1"],
            rows,
        )
    )
    chunks += [
        "## Source-level descriptive scores",
        "These are descriptions of the already selected common-nine texts. No source subgroup has been promoted to an additional primary comparison. Fine-only examples remain outside the main hate sample.",
    ]
    rows = []
    for endpoint, v in endpoints.items():
        for source, p in v["sources"].items():
            rows.append(
                [
                    NAMES[endpoint],
                    source,
                    p["n"],
                    *[f(x) for x in p["macro_f1"]],
                    *([f(x) for x in p["micro_f1"]] if "micro_f1" in p else ["—"] * 3),
                ]
            )
    chunks.append(
        table(
            [
                "Endpoint",
                "Source",
                "N",
                "Macro A",
                "Macro B",
                "Macro C",
                "Micro A",
                "Micro B",
                "Micro C",
            ],
            rows,
        )
    )
    chunks += [
        "## Evidence and limits",
        f"Numerical source: `{c.ref(root)}`. The JSON and NumPy artifacts retain unrounded values, masks, encoded predictions and every bootstrap draw. Source hashes and independent checks are retained there.",
        "Part 5 still needs the predeclared strict-inner-valid, no-repair, own-valid, failure-bound, ANN and historical-drift checks. This appendix is not a claim that those sensitivities are finished. Part 6 will account for resources, Part 7 will integrate the scientific conclusion, and Part 8 will prepare the separate writer handover. No thesis edit or new model request is part of this export.",
    ]
    target = base / "numerical_appendix.md"
    with target.open("x", encoding="utf8", newline="\n") as handle:
        handle.write("\n\n".join(chunks) + "\n")
    reader.save(
        base / "table_export.json",
        dict(
            status="exported_from_verified_unrounded_results",
            source_completion_sha256=c.sha(root / "complete.json"),
            appendix_sha256=c.sha(target),
            exporter_sha256=c.sha(Path(__file__)),
        ),
    )
    print("Numerical appendix exported:", target, flush=True)


if __name__ == "__main__":
    with c.offline():
        main()

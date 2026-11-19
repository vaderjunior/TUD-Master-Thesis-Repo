from analysis.micro_f1.combined import prepare as p

c = p.c


def main():
    rows = c.read(p.BASE / "comparison_ledger.json")
    main = [r for r in rows if r["question_id"]]
    existing = c.read(p.ROOTS[3] / "diagnostics/results.json")["matched_confidence"]
    result = dict(
        status="passed",
        matched_confidence_source="Part 3 already verified stored-draw diagnostics",
        supervised_same_confidence=existing,
        main_full_vs_restricted_adjusted_changes=[
            r["id"]
            for r in main
            if r["metrics"]["micro"]["adjusted_class"]
            != r["metrics"]["micro_macro_labels"]["adjusted_class"]
        ],
        all_full_vs_restricted_nominal_changes=[
            r["id"] for r in rows if r["full_vs_restricted_class95_changed"]
        ],
        main_nominal_metric_classification_changes=[
            r["id"]
            for r in main
            if r["metrics"]["macro"]["class95"] != r["metrics"]["micro"]["class95"]
        ],
        main_uncertain_point_reversals=[
            r["id"]
            for r in main
            if r["matched95_interpretation"] == "uncertain_point_reversal"
        ],
        main_supported_opposite_directions=[
            r["id"]
            for r in main
            if r["matched95_interpretation"] == "supported_opposite_directions"
        ],
        all_supported_opposite_directions=[
            r["id"]
            for r in rows
            if r["matched95_interpretation"] == "supported_opposite_directions"
        ],
        source_sha256=c.BINDINGS,
        caution="Different interval classifications are not a test that the two metrics have statistically different effects. All original family roles remain. Do not select a metric or interval after seeing results.",
    )
    c.require(
        len(result["main_full_vs_restricted_adjusted_changes"]) == 1,
        "Review extra main scope changes",
    )
    c.require(
        len(result["all_full_vs_restricted_nominal_changes"]) == 3,
        "Review extra nominal scope changes",
    )
    c.require(
        len(result["main_nominal_metric_classification_changes"]) == 9,
        "Review changed metric inventory",
    )
    c.require(
        len(result["main_uncertain_point_reversals"]) == 9
        and not result["main_supported_opposite_directions"],
        "Review main direction inventory",
    )
    c.require(
        len(result["all_supported_opposite_directions"]) == 1,
        "Review supporting direction inventory",
    )
    p.save("scope_confidence_audit.json", result)
    print(
        "Scope and confidence audit passed:",
        len(existing),
        "existing equal-confidence diagnostics; one main adjusted scope change; three supporting nominal scope changes.",
    )


if __name__ == "__main__":
    main()

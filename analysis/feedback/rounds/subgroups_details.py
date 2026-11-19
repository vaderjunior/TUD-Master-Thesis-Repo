from collections import Counter, defaultdict
from pathlib import Path
from analysis.feedback.rounds import subgroups_inputs as a


def main():
    proof = a.read(a.OUT / "independent_verification.json")
    a.need(
        proof["status"] == "part5_independent_verification_pass",
        "Independent analysis required",
    )
    result = a.read(a.OUT / "analysis.json")
    meta = list(a.rows(a.OUT / "metadata.jsonl"))
    records = result["records"]
    contributions = []
    for family in ["source", "function"]:
        for endpoint in ["FPR", "FNR"]:
            cells = [
                r
                for r in records.values()
                if r["family"] == family
                and r["endpoint"] == endpoint
                and r["view"] == "final"
                and r["status"] == "defined"
            ]
            n = sum(r["inputs"] for r in cells)
            errors = sum(r["integer_numerators"]["r4_adaptive"] for r in cells)
            for r in cells:
                num = (
                    r["integer_numerators"]["r4_adaptive"]
                    - r["integer_numerators"]["r4_reference"]
                )
                contributions.append(
                    dict(
                        group=r["group"],
                        endpoint=endpoint,
                        inputs=r["inputs"],
                        population_inputs=n,
                        difference_numerator=num,
                        contribution_pp=100 * num / (9 * n),
                        adaptive_error_numerator=r["integer_numerators"]["r4_adaptive"],
                        population_error_numerator=errors,
                        error_share=(
                            r["integer_numerators"]["r4_adaptive"] / errors
                            if errors
                            else None
                        ),
                    )
                )
    function_summary = []
    for fn in a.FUNCTIONS:
        ep = "FNR" if fn.endswith("_h") else "FPR"
        main = records[f"function/{fn}|{ep}|final"]
        fixed = records[f"function/{fn}|{ep}|fixed"]
        strict = records[f"function/{fn}|{ep}|strict"]
        influence = next(r for r in result["function_influence"] if r["function"] == fn)
        j = influence["columns"].index("r4_adaptive-reference")
        values = [r["values"][j] for r in influence["leave_one_out"]]
        function_summary.append(
            dict(
                function=fn,
                endpoint=ep,
                final_inputs=main["inputs"],
                clusters=main["clusters"],
                final_difference=main["points"]["r4_adaptive-reference"],
                final_interval=main["intervals"]["r4_adaptive-reference"],
                fixed_difference=fixed["points"]["r4_adaptive-reference"],
                strict_difference=strict["points"]["r4_adaptive-reference"],
                strict_interval=strict["intervals"]["r4_adaptive-reference"],
                all_policies_final_zero=all(
                    main["integer_numerators"]["r4_" + p] == 0 for p in a.POLICIES
                ),
                adaptive_final_zero=main["integer_numerators"]["r4_adaptive"] == 0,
                adaptive_all_rounds_zero=all(
                    fixed["integer_numerators"][f"r{r}_adaptive"] == 0
                    for r in range(1, 5)
                ),
                leave_one_cluster_range=[min(values), max(values)] if values else None,
                fixed_equal_cluster_difference=influence["equal_cluster_mean"][j],
            )
        )
    cluster_functions = defaultdict(set)
    for r in meta:
        if r["source"] == "hatecheck":
            cluster_functions[r["inference_cluster"]].add(r["functionality"])
    edges = list(a.rows(a.OUT / "linked_edges.jsonl"))
    usable = [r for r in edges if r["status"] == "usable"]
    annotation_fields = {}
    for source in a.SOURCES:
        ids = [r for r in meta if r["source"] == source]
        annotation_fields[source] = dict(
            roster=len(ids),
            fixed=sum(r["masks"]["all_round_common"] for r in ids),
            final=sum(r["masks"]["round_common"][3] for r in ids),
            strict=sum(r["masks"]["round_strict"][3] for r in ids),
        )
    annotated_source_counts = {}
    for r in meta:
        if r["source"] == "hatexplain":
            key = r["majority_label"] + "/" + r["annotation_agreement"]
        elif r["source"] == "implicit_hate":
            key = r["stage1_class"]
        elif r["source"] == "mhs":
            key = str(r["fine_gold"]["severity"])
        else:
            continue
        annotated_source_counts.setdefault(r["source"], Counter())[key] += 1
    a.write(
        a.OUT / "details.json",
        dict(
            created_utc=a.pure.utc(),
            contributions=contributions,
            function_summary=function_summary,
            natural_source_counts=annotation_fields,
            annotation_roster_counts=annotated_source_counts,
            hatecheck_clusters=len(cluster_functions),
            cross_function_clusters={
                c: sorted(v) for c, v in cluster_functions.items() if len(v) > 1
            },
            linked=dict(
                all_edges=len(edges),
                usable=len(usable),
                excluded=dict(
                    Counter(e["status"] for e in edges if e["status"] != "usable")
                ),
                unique_usable_texts=len(
                    {i for e in usable for i in [e["child"], e["parent"]]}
                ),
                same_cluster_edges=sum(e["same_inference_cluster"] for e in usable),
                same_gold_edges=sum(e["same_gold"] for e in usable),
                opposite_gold_edges=sum(not e["same_gold"] for e in usable),
            ),
            new_Qwen_calls=0,
        ),
    )
    a.write(
        a.OUT / "details_complete.json",
        dict(
            status="derived_reporting_quantities_pending_readback",
            inputs={
                a.rel(a.OUT / "analysis.json"): a.sha(a.OUT / "analysis.json"),
                a.rel(a.OUT / "metadata.jsonl"): a.sha(a.OUT / "metadata.jsonl"),
                a.rel(a.OUT / "linked_edges.jsonl"): a.sha(
                    a.OUT / "linked_edges.jsonl"
                ),
            },
            details_sha256=a.sha(a.OUT / "details.json"),
            source_sha256=a.sha(Path(__file__)),
        ),
    )
    print(
        "Part 5 reporting contributions, function summaries and linked-pair quantities saved.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()

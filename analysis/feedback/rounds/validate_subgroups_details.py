from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from analysis.feedback.rounds import subgroups_inputs as a


def digest(value):
    return hashlib.sha256(
        (
            json.dumps(
                value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False
            )
            + "\n"
        ).encode()
    ).hexdigest()


def near(left, right):
    a.need(abs(left - right) < 1e-12, "Derived numerical readback differs")


def main():
    result = a.read(a.OUT / "analysis.json")
    details = a.read(a.OUT / "details.json")
    proof = a.read(a.OUT / "details_complete.json")
    for path, sha in proof["inputs"].items():
        a.need(a.sha(a.ROOT / path) == sha, "Details input changed")
    a.need(a.sha(a.OUT / "details.json") == proof["details_sha256"], "Details changed")
    original = list(a.rows(a.STUDY / "evaluation.jsonl"))
    broad = list(a.rows(a.SCOPE / "evaluation.jsonl"))
    matrix = list(a.rows(a.READER / "matrix.jsonl"))
    meta = list(a.rows(a.OUT / "metadata.jsonl"))
    records = result["records"]
    selections = a.read(a.OUT / "case_selection.json")
    for r in result["sample_support"]:
        before = [
            x
            for x in broad
            if x["source"] == r["source"] and x["gate"] == r["gold_hate"]
        ]
        selected = [
            x
            for x in original
            if x["source"] == r["source"] and x["gate"] == r["gold_hate"]
        ]
        a.need(
            r["broad_inputs"] == len(before)
            and r["sample_inputs"] == len(selected)
            and r["broad_clusters"] == len({x["inference_cluster"] for x in before})
            and r["sample_clusters"] == len({x["inference_cluster"] for x in selected}),
            "Conditional sample accounting differs",
        )
    for r in details["contributions"]:
        family = r["group"].split("/")[0]
        parts = [
            v
            for v in records.values()
            if v["family"] == family
            and v["endpoint"] == r["endpoint"]
            and v["view"] == "final"
            and v["status"] == "defined"
        ]
        m = records[r["group"] + "|" + r["endpoint"] + "|final"]
        den = sum(p["inputs"] for p in parts)
        num = (
            m["integer_numerators"]["r4_adaptive"]
            - m["integer_numerators"]["r4_reference"]
        )
        errors = sum(p["integer_numerators"]["r4_adaptive"] for p in parts)
        a.need(
            r["difference_numerator"] == num
            and r["population_inputs"] == den
            and r["population_error_numerator"] == errors,
            "Contribution components",
        )
        near(r["contribution_pp"], float(Fraction(100 * num, 9 * den)))
        if errors:
            near(
                r["error_share"],
                float(Fraction(m["integer_numerators"]["r4_adaptive"], errors)),
            )
    for r in details["function_summary"]:
        for view, field in [
            ("final", "final"),
            ("fixed", "fixed"),
            ("strict", "strict"),
        ]:
            m = records[f'function/{r["function"]}|{r["endpoint"]}|{view}']
            near(r[field + "_difference"], m["points"]["r4_adaptive-reference"])
            if field != "fixed":
                a.need(
                    r[field + "_interval"] == m["intervals"]["r4_adaptive-reference"],
                    "Function interval copy",
                )
        m = records[f'function/{r["function"]}|{r["endpoint"]}|final']
        a.need(
            r["all_policies_final_zero"]
            == all(m["points"]["r4_" + p] == 0 for p in a.POLICIES),
            "Zero-observed-error summary",
        )
        m = records[f'function/{r["function"]}|{r["endpoint"]}|fixed']
        a.need(
            r["adaptive_all_rounds_zero"]
            == all(m["points"][f"r{rnd}_adaptive"] == 0 for rnd in range(1, 5)),
            "All-round zero summary",
        )
        inf = next(
            x for x in result["function_influence"] if x["function"] == r["function"]
        )
        j = inf["columns"].index("r4_adaptive-reference")
        vals = [x["values"][j] for x in inf["leave_one_out"]]
        a.need(
            r["leave_one_cluster_range"] == ([min(vals), max(vals)] if vals else None),
            "Influence range copy",
        )
        near(r["fixed_equal_cluster_difference"], inf["equal_cluster_mean"][j])
    mapping = defaultdict(set)
    for r in original:
        if r["source"] == "hatecheck":
            mapping[r["inference_cluster"]].add(json.loads(r["raw"])["functionality"])
    a.need(
        details["hatecheck_clusters"] == len(mapping)
        and details["cross_function_clusters"]
        == {c: sorted(v) for c, v in mapping.items() if len(v) > 1},
        "Cross-function cluster accounting",
    )
    edges = list(a.rows(a.OUT / "linked_edges.jsonl"))
    usable = [e for e in edges if e["status"] == "usable"]
    link = details["linked"]
    a.need(
        link["all_edges"] == len(edges)
        and link["usable"] == len(usable)
        and link["unique_usable_texts"]
        == len({v for e in usable for v in [e["child"], e["parent"]]})
        and link["same_cluster_edges"]
        == sum(e["same_inference_cluster"] for e in usable)
        and link["same_gold_edges"] == sum(e["same_gold"] for e in usable)
        and link["opposite_gold_edges"] == sum(not e["same_gold"] for e in usable),
        "Linked summary units",
    )

    cases = a.read(a.OUT / "cases_v1/complete.json")
    bindings = {
        (r["phase"], r["key"], r["condition"]): r
        for r in a.rows(a.PART3 / "prompt_bindings.jsonl")
        if r["phase"].endswith("_evaluation")
    }
    source_records = a.read(a.STUDY / "gate_only_records.json")
    prompts = predictions = 0
    for path, sha in cases["input_bindings"].items():
        a.need(a.sha(a.ROOT / path) == sha, "Original case evidence changed")
    for entry in cases["catalog"]:
        path = a.ROOT / entry["file"]
        a.need(a.sha(path) == entry["sha256"], "Case copy changed")
        case = a.read(path)
        i = case["input_index"]
        a.need(
            case["gold"] == original[i]
            and case["metadata"] == meta[i]
            and case["selection_slots"]
            == [s for s in selections if s["input_index"] == i],
            "Case gold/selection/metadata",
        )
        for frame in case["frames"]:
            rnd = frame["round"]
            a.need(
                [v["condition"] for v in frame["conditions"]] == a.CONDITIONS,
                "Complete condition axes",
            )
            for ci, v in enumerate(frame["conditions"]):
                p = v["prompt"]
                bound = bindings[f"r{rnd}_evaluation", case["key"], v["condition"]]
                a.need(
                    digest(p["messages"])
                    == bound["messages_sha256"]
                    == p["messages_sha256"]
                    and digest(p["context"])
                    == bound["context_sha256"]
                    == p["context_sha256"],
                    "Independent rendered prompt identity",
                )
                a.need(
                    v["correction_record"]
                    == (source_records[p["source_id"]] if p["source_id"] else None),
                    "Correction original record",
                )
                prompts += 1
                for rep, out in enumerate(v["outputs"]):
                    expected = matrix[i]["predictions"][rnd - 1][ci][rep]
                    a.need(
                        out["repeat"] == rep
                        and out["condition"] == v["condition"]
                        and out["key"] == case["key"]
                        and out["phase"] == f"r{rnd}_evaluation",
                        "Prediction identity",
                    )
                    a.need(
                        all(
                            out[f] == expected[f]
                            for f in ["result", "n_valid", "uncertain"]
                        )
                        and out["messages_sha256"] == bound["messages_sha256"],
                        "Prediction/result binding",
                    )
                    predictions += 1
        for policy, trajectory in case["trajectory"]["policies"].items():
            errors = []
            for frame in case["frames"]:
                errors.append(
                    [
                        int(out["result"]["hate"] != case["gold"]["gate"])
                        for stream in a.STREAMS
                        for v in frame["conditions"]
                        if v["condition"] == f"{policy}_s{stream}"
                        for out in v["outputs"]
                    ]
                )
            a.need(errors == trajectory["errors_by_repeat"], "Case error path")
    a.need(
        len(cases["catalog"]) == 52
        and prompts == 2080
        and predictions == 6240
        and len(cases["empty_slots"]) == 1,
        "Complete case coverage",
    )
    inputs = {
        a.rel(a.OUT / f): a.sha(a.OUT / f)
        for f in [
            "analysis.json",
            "details_complete.json",
            "details.json",
            "cases_v1/complete.json",
            "independent_verification.json",
        ]
    }
    a.write(
        a.OUT / "details_independent_verification.json",
        dict(
            status="passed",
            created_utc=a.pure.utc(),
            inputs=inputs,
            source_sha256=a.sha(Path(__file__)),
            sample_support_cells=len(result["sample_support"]),
            function_summaries=29,
            contributions=len(details["contributions"]),
            case_inputs=52,
            case_slots=53,
            empty_slots=1,
            prompts=prompts,
            predictions=predictions,
            new_Qwen_calls=0,
        ),
    )
    print(
        "Independent sample, reporting quantities and all 52 complete cases PASS.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()

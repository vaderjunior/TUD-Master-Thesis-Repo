import json
from analysis.feedback.human import agreements as a


def main():
    s = a.read(a.BASE / "result_summary.json")
    key = {x["packet_id"]: x for x in s["items"]}
    source_ids = {x["id"]: x["packet_id"] for x in s["items"]}
    data = {
        (r, f): {x["packet_id"]: x["judgments"][r + "_" + f] for x in s["items"]}
        for r in ["A", "B"]
        for f in ["source", "project"]
    }
    path = (
        a.ROOT
        / "experiments/astra/sq3_combined_results_20261004_v1/review/predictions.jsonl"
    )
    saved = {}
    with path.open(encoding="utf8") as stream:
        for line in stream:
            p = json.loads(line)
            if (
                p["study"] == "taxonomy"
                and p["condition"] == "rag_full"
                and p["population"] == "hateful_types"
                and p["id"] in source_ids
            ):
                k = p["id"], p["repeat"]
                a.require(k not in saved, "Duplicate saved-repeat identity")
                saved[k] = p
    a.require(len(saved) == 72, "Expected 24 texts across three existing repeats")
    sources = {}
    repetitions = {}
    for rep in range(3):
        rk = {}
        for iid, pid in source_ids.items():
            prediction = saved[iid, rep]
            a.require(prediction["result"] is not None, "Selected repeat invalid")
            source = a.ROOT / prediction["source"]
            a.require(
                a.sha(source) == prediction["source_sha256"],
                "Saved model source changed",
            )
            record = a.read(source)
            a.require(
                record["identity"]["repeat"] == rep
                and record["identity"]["key"] == "en/" + iid,
                "Wrong repeat identity",
            )
            a.require(
                {k: record["result"][k] for k in prediction["result"]}
                == prediction["result"],
                "Saved summary differs from raw aggregate",
            )
            sources[a.rel(source)] = a.sha(source)
            rk[pid] = {**key[pid], "model_repeat_0": prediction["result"]}
            if rep == 0:
                a.require(
                    prediction["result"] == key[pid]["model_repeat_0"],
                    "Primary repeat differs",
                )
        repetitions[str(rep)] = {
            name: a.occurrence_review(
                rk, data["A", "project"], data["B", "project"], name
            )
            for name in ["model_extra", "model_omitted", "model_and_gold"]
        }
    for name, value in repetitions["0"].items():
        a.require(
            value == s["model_occurrence_review_project_guide"][name],
            "Repeat-0 summary mismatch",
        )
    flagged = {
        i
        for i in key
        if any(
            data[r, f][i]["difficulty"]
            for r in ["A", "B"]
            for f in ["source", "project"]
        )
    }
    cohorts = {
        "all_four_binary_resolved": s["common_resolved_all_four"],
        "all_four_hateful": s["common_hateful_all_four"],
        "no_reported_difficulty_in_any_form": sorted(set(key) - flagged),
    }
    cohort_result = {}
    for name, ids in cohorts.items():
        subset = {
            (r, f): {i: data[r, f][i] for i in ids}
            for r in ["A", "B"]
            for f in ["source", "project"]
        }
        sk = {i: key[i] for i in ids}
        cohort_result[name] = {
            "n": len(ids),
            "ids": ids,
            "between_reviewers": {
                f: a.compare_reviewers(subset["A", f], subset["B", f], f)
                for f in ["source", "project"]
            },
            "within_reviewers": {
                r: a.compare_guides(subset[r, "source"], subset[r, "project"])
                for r in ["A", "B"]
            },
            "project_extra": a.occurrence_review(
                sk, subset["A", "project"], subset["B", "project"], "model_extra"
            ),
            "mean_complete_response_label_count": {
                r
                + "_"
                + f: sum(
                    len(a.annotation_set(subset[r, f][i]))
                    for i in ids
                    if a.annotation_set(subset[r, f][i]) is not None
                )
                / sum(a.annotation_set(subset[r, f][i]) is not None for i in ids)
                for r in ["A", "B"]
                for f in ["source", "project"]
            },
        }
    summary = {
        "status": "exploratory_saved_output_checks_complete",
        "same_frozen_texts": 24,
        "saved_predictions_checked": 72,
        "repeat_results": repetitions,
        "cohort_results": cohort_result,
        "any_difficulty_ids": sorted(flagged),
        "qualification": "B Part 1 reports discussion or AI/search; clarification pending. Neither repeated outputs nor label occurrences are independent texts.",
        "sources": {
            a.rel(path): a.sha(path),
            a.rel(a.BASE / "EXPLORATORY_CHECKS.md"): a.sha(
                a.BASE / "EXPLORATORY_CHECKS.md"
            ),
            a.rel(a.BASE / "result_summary.json"): a.sha(
                a.BASE / "result_summary.json"
            ),
            a.rel(__file__): a.sha(__file__),
            **sources,
        },
        "new_model_calls": 0,
        "human_judgments_added": 0,
    }
    a.save(a.BASE / "supplementary_checks.json", summary)
    print(
        json.dumps(
            {
                "repeats": {
                    rep: {
                        name: {"n": x["occurrences"], "counts": x["counts"]}
                        for name, x in value.items()
                    }
                    for rep, value in repetitions.items()
                },
                "common_cohorts": {k: v["n"] for k, v in cohort_result.items()},
                "flagged_texts": sorted(flagged),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

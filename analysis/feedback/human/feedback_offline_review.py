from collections import Counter
from copy import deepcopy
from fractions import Fraction
import math
from analysis.feedback.human import feedback_common as c
from analysis.feedback.human import feedback_score as scorer
from analysis.feedback.human import feedback_api as api


def expected_target(r, ref):
    if ref == "dataset":
        return r["gold"]["hate"], set(r["gold"]["hate_type"])
    j = r["judgments"][ref + "_project"]
    gate = j["binary_hate"]
    if gate is None:
        return None, None
    if not gate:
        return False, set()
    return True, set(j["types"]) if j["type_state"] == "assigned" else None


def independent_arithmetic(roster):

    predictions = {}
    for i, r in enumerate(roster):
        for j, arm in enumerate(c.CONDITIONS):
            for rep in range(3):
                n = (
                    0
                    if (i * 3 + j * 7 + rep) % 19 == 0
                    else 2 if (i + j + rep) % 11 == 0 else 3
                )
                gate = (i + 2 * j + rep) % 4 != 0
                types = (
                    [x for k, x in enumerate(c.LABELS) if (i + j + rep + k) % 3 == 0]
                    if gate
                    else []
                )
                predictions[r["key"], arm, rep] = {
                    "result": {"hate": gate, "hate_type": types} if n else None,
                    "n_valid": n,
                }
    report = scorer.analyze(roster, predictions)
    checks = 0
    rows = [r for r in roster if r["transfer_eligible"]]
    for strict in (False, True):
        section = report["transfer"]["all_three_votes_valid" if strict else "other_all"]
        for ref in ("A", "B", "dataset"):
            for ep in ("gate", "complete", "jaccard"):
                eligible = []
                values = {}
                for r in rows:
                    g, t = expected_target(r, ref)
                    if (
                        g is None
                        or (ep != "gate" and t is None)
                        or (ep == "jaccard" and not g)
                    ):
                        continue
                    eligible.append(r)
                    for arm in c.CONDITIONS:
                        for rep in range(3):
                            v = predictions[r["key"], arm, rep]
                            p = v["result"]
                            if p is None or strict and v["n_valid"] != 3:
                                z = None
                            elif ep == "gate":
                                z = Fraction(int(p["hate"] == g))
                            elif ep == "complete":
                                z = Fraction(
                                    int(p["hate"] == g and set(p["hate_type"]) == t)
                                )
                            else:
                                ps = set(p["hate_type"]) if p["hate"] else set()
                                z = (
                                    Fraction(len(ps & t), len(ps | t))
                                    if ps | t
                                    else Fraction(1)
                                )
                            values[r["key"], arm, rep] = z
                common = [
                    r
                    for r in eligible
                    if all(
                        values[r["key"], a, rep] is not None
                        for a in c.CONDITIONS
                        for rep in range(3)
                    )
                ]
                d = section[ref][ep]
                c.require(
                    d["annotation_eligible_texts"] == len(eligible)
                    and d["common_valid_texts"] == len(common),
                    "Independent eligibility/missing mask differs",
                )
                checks += 2
                for arm in c.CONDITIONS:
                    mean = (
                        float(
                            sum(
                                values[r["key"], arm, rep]
                                for r in common
                                for rep in range(3)
                            )
                            / (len(common) * 3)
                        )
                        if common
                        else None
                    )
                    actual = d["arms"][arm]["mean_text_agreement"]
                    c.require(
                        (
                            actual is None
                            if mean is None
                            else math.isclose(actual, mean, abs_tol=1e-12)
                        ),
                        "Exact arithmetic agreement differs",
                    )
                    checks += 1
                for treatment, control in scorer.CONTRASTS:
                    result = d["contrasts"][treatment + "_minus_" + control]
                    low = high = Fraction(0)
                    for r in eligible:
                        for rep in range(3):
                            av, bv = (
                                values[r["key"], treatment, rep],
                                values[r["key"], control, rep],
                            )
                            low += (Fraction(0) if av is None else av) - (
                                Fraction(1) if bv is None else bv
                            )
                            high += (Fraction(1) if av is None else av) - (
                                Fraction(0) if bv is None else bv
                            )
                    bounds = (
                        [
                            float(low / (3 * len(eligible))),
                            float(high / (3 * len(eligible))),
                        ]
                        if eligible
                        else None
                    )
                    actual = result["missing_output_bounds_full_eligible"]
                    c.require(
                        (
                            actual is None
                            if bounds is None
                            else all(
                                math.isclose(a, b, abs_tol=1e-12)
                                for a, b in zip(actual, bounds)
                            )
                        ),
                        "Missing-output bounds differ",
                    )
                    checks += 1
                    for rep in range(3):
                        nfix = sum(
                            values[r["key"], treatment, rep]
                            > values[r["key"], control, rep]
                            for r in common
                        )
                        nreg = sum(
                            values[r["key"], treatment, rep]
                            < values[r["key"], control, rep]
                            for r in common
                        )
                        c.require(
                            result["paired_transitions_by_repeat"][rep]
                            == {
                                "improved": nfix,
                                "regressed": nreg,
                                "unchanged": len(common) - nfix - nreg,
                            },
                            "Independent transition count differs",
                        )
                        checks += 1
    c.require(
        report["cohort_sizes"]["other_all"] == 21
        and report["cohort_sizes"]["other_changed_messages"] == 7
        and report["cohort_sizes"]["other_identical_messages"] == 14
        and len(report["source_uptake"]) == 3,
        "Scope accounting differs",
    )
    checks += 4
    return {
        "status": "passed",
        "exact_arithmetic_checks": checks,
        "synthetic_prediction_records": len(predictions),
        "has_nulls": any(v["result"] is None for v in predictions.values()),
        "has_two_valid_votes": any(v["n_valid"] == 2 for v in predictions.values()),
        "new_Qwen_calls": 0,
        "synthetic_NOT_scientific_results": True,
    }


def eligibility_reason(r):
    a, b = (r["judgments"][x + "_project"] for x in ("A", "B"))
    if a["binary_hate"] is None or b["binary_hate"] is None:
        return "At least one unresolved human gate"
    if a["binary_hate"] != b["binary_hate"]:
        return "Human gate disagreement"
    if r["is_correction_source"]:
        return (
            "Selected: agreed complete types"
            if r["packet_id"] != "R059"
            else "Selected: agreed gate only; types unresolved between people"
        )
    if a["binary_hate"] is False:
        return "Both agree non-hate; model already agrees"
    if a["type_state"] != "assigned" or b["type_state"] != "assigned":
        return "No complete agreed type annotation"
    if set(a["types"]) != set(b["types"]):
        return "Different type sets; partial agreement is not a complete target"
    return "Model already matches agreed answer"


def main():
    with c.offline():
        b = api.queue()
        b.validate(ready=True)
        roster = list(c.rows(c.BASE / "roster.jsonl"))
        verification = c.read(c.BASE / "independent_verification.json")
        readiness = c.read(b.BASE / "development/readiness.json")
        c.require(
            verification["project_cards_redecoded"] == 48
            and readiness["mock_predictions"] == 216
            and readiness["new_calls_on_restart"] == 0,
            "Preparation checks incomplete",
        )
        arithmetic = independent_arithmetic(roster)
        c.save(
            c.BASE / "analysis_validation.json",
            dict(arithmetic, source_hashes=c.bindings([__file__, scorer.__file__])),
        )
        census = []
        for r in roster:
            census.append(
                {
                    "alias": r["packet_id"],
                    "source_id": r["id"],
                    "stratum": r["stratum"],
                    "reason": eligibility_reason(r),
                    "correction_source": r["is_correction_source"],
                    "transfer_eligible": r["transfer_eligible"],
                    "selected_correction": r["selected_correction"],
                    "project_A": r["judgments"]["A_project"],
                    "project_B": r["judgments"]["B_project"],
                    "original_model": r["model_repeat_0"],
                    "dataset": r["gold"],
                }
            )
        c.save(c.BASE / "all_case_census.json", census)
        summary = c.read(c.BASE / "summary.json")
        c.require(
            summary["reviewer_coverage_other"]
            == {"A": {"gate": 15, "complete": 15}, "B": {"gate": 20, "complete": 20}},
            "Frozen coverage differs",
        )
        expected = {r["packet_id"] for r in roster if r["is_correction_source"]}
        c.require(expected == {"R014", "R035", "R059"}, "Source roster differs")
        c.require(
            not (b.BASE / "live").exists() and not (c.BASE / "execution/live").exists(),
            "Unexpected live human-feedback work",
        )
        paths = [
            c.BASE / x
            for x in (
                "STUDY_PROTOCOL.md",
                "preparation_plan.json",
                "prepared.json",
                "complete.json",
                "independent_verification.json",
                "summary.json",
                "analysis_validation.json",
                "all_case_census.json",
                "development/windows_path_reader_issue.json",
                "current_queue_checkpoint.json",
            )
        ]
        paths += [b.PLAN, b.BASE / "ready.json", b.BASE / "development/readiness.json"]
        c.save(
            c.BASE / "offline_review.json",
            {
                "status": "offline_ready_live_results_pending",
                "created_utc": c.utc(),
                "bindings": c.bindings(paths),
                "active_queue_preservation": c.active_preservation(),
                "unit_tests": 17,
                "independent_arithmetic_checks": arithmetic["exact_arithmetic_checks"],
                "raw_project_cards_redecoded": 48,
                "frozen_prompts": 72,
                "mock_predictions": 216,
                "mock_replies": 648,
                "zero_new_calls_on_restart": True,
                "actual_Qwen_calls": 0,
                "new_embeddings": 0,
                "training_fits": 0,
                "new_human_judgments": 0,
                "live_execution_path": c.rel(b.BASE),
                "live_launch_default": "validate",
                "limitations": [
                    "Preparation only; no human-feedback effectiveness outcomes",
                    "Only seven transfer texts receive changed labels",
                    "No expert/adjudicated gold or real successive human rounds",
                    "Other accepted SQ3 studies still pending",
                ],
            },
        )
        print(
            {
                "status": "offline_ready_live_results_pending",
                "unit_tests": 17,
                **arithmetic,
            }
        )


if __name__ == "__main__":
    main()

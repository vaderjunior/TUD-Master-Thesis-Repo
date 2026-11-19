from collections import Counter
from analysis.feedback.human import feedback_common as c

CONTRASTS = (
    ("human_feedback5", "dataset_feedback5"),
    ("human_feedback5", "reference5"),
    ("dataset_feedback5", "reference5"),
)


def metric(prediction, gate, types, endpoint):
    if prediction is None:
        return None
    pg = prediction["hate"]
    pt = set(prediction["hate_type"]) if pg else set()
    if endpoint == "gate":
        return float(pg == gate)
    if endpoint == "complete":
        return float(pg == gate and pt == set(types))
    c.require(
        endpoint == "jaccard" and gate is True and types is not None,
        "Invalid Jaccard target",
    )
    target = set(types)
    return len(pt & target) / len(pt | target) if pt | target else 1.0


def value(predictions, key, arm, repeat, strict=False):
    row = predictions[(key, arm, repeat)]
    return None if strict and row["n_valid"] != 3 else row["result"]


def endpoint(rows, predictions, reference, name, strict=False):
    eligible = []
    for r in rows:
        gate, types = c.reference_target(r, reference)
        if (
            gate is None
            or (name != "gate" and types is None)
            or (name == "jaccard" and gate is not True)
        ):
            continue
        eligible.append((r, gate, types))
    scores = {}
    for r, gate, types in eligible:
        scores[r["key"]] = {
            arm: [
                metric(
                    value(predictions, r["key"], arm, rep, strict), gate, types, name
                )
                for rep in range(3)
            ]
            for arm in c.CONDITIONS
        }
    common = [
        r
        for r, _, _ in eligible
        if all(v is not None for vs in scores[r["key"]].values() for v in vs)
    ]
    arm_stats = {}
    for arm in c.CONDITIONS:
        per_repeat = [
            (
                sum(scores[r["key"]][arm][rep] for r in common) / len(common)
                if common
                else None
            )
            for rep in range(3)
        ]
        arm_stats[arm] = {
            "mean_text_agreement": sum(per_repeat) / 3 if common else None,
            "per_repeat": per_repeat,
        }
    contrasts = {}
    for treatment, control in CONTRASTS:
        diffs = [
            [
                scores[r["key"]][treatment][rep] - scores[r["key"]][control][rep]
                for r in common
            ]
            for rep in range(3)
        ]
        transition = []
        for rep in range(3):
            counts = Counter()
            for r in common:
                a, b = scores[r["key"]][treatment][rep], scores[r["key"]][control][rep]
                counts[
                    "improved" if a > b else "regressed" if a < b else "unchanged"
                ] += 1
            transition.append(
                {k: counts[k] for k in ("improved", "regressed", "unchanged")}
            )
        low = high = 0.0
        for r, _, _ in eligible:
            for rep in range(3):
                a, b = scores[r["key"]][treatment][rep], scores[r["key"]][control][rep]
                low += (0 if a is None else a) - (1 if b is None else b)
                high += (1 if a is None else a) - (0 if b is None else b)
        denom = len(eligible) * 3
        contrasts[treatment + "_minus_" + control] = {
            "mean_difference": (
                sum(map(sum, diffs)) / (len(common) * 3) if common else None
            ),
            "per_repeat_difference": [
                sum(x) / len(common) if common else None for x in diffs
            ],
            "paired_transitions_by_repeat": transition,
            "mean_text_transition_counts": {
                k: sum(d[k] for d in transition) / 3
                for k in ("improved", "regressed", "unchanged")
            },
            "missing_output_bounds_full_eligible": (
                [low / denom, high / denom] if denom else None
            ),
        }
    return {
        "annotation_eligible_texts": len(eligible),
        "common_valid_texts": len(common),
        "eligible_aliases": [r["packet_id"] for r, _, _ in eligible],
        "common_aliases": [r["packet_id"] for r in common],
        "arms": arm_stats,
        "contrasts": contrasts,
        "per_text": [
            {"alias": r["packet_id"], "key": r["key"], "scores": scores[r["key"]]}
            for r, _, _ in eligible
        ],
    }


def label_statistics(rows, predictions, reference, strict=False):
    eligible = [
        r
        for r in rows
        if c.reference_target(r, reference)[0] is True
        and c.reference_target(r, reference)[1] is not None
    ]
    valid = [
        r
        for r in eligible
        if all(
            value(predictions, r["key"], arm, rep, strict) is not None
            for arm in c.CONDITIONS
            for rep in range(3)
        )
    ]
    outputs = {}
    for arm in c.CONDITIONS:
        repeats = []
        for rep in range(3):
            labels = {}
            for label in c.LABELS:
                counts = Counter()
                for r in valid:
                    target = label in c.reference_target(r, reference)[1]
                    p = value(predictions, r["key"], arm, rep, strict)
                    positive = bool(p["hate"]) and label in p["hate_type"]
                    counts[
                        (
                            "tp"
                            if target and positive
                            else "fn" if target else "fp" if positive else "tn"
                        )
                    ] += 1
                d = {k: counts[k] for k in ("tp", "fp", "fn", "tn")}
                denom = 2 * d["tp"] + d["fp"] + d["fn"]
                d.update(
                    support=d["tp"] + d["fn"], f1=2 * d["tp"] / denom if denom else 0.0
                )
                labels[label] = d
            repeats.append(
                {
                    "labels": labels,
                    "macro_f1_fixed_seven": (
                        sum(v["f1"] for v in labels.values()) / 7 if valid else None
                    ),
                }
            )
        outputs[arm] = {
            "per_repeat": repeats,
            "mean_macro_f1": (
                sum(x["macro_f1_fixed_seven"] for x in repeats) / 3 if valid else None
            ),
        }
    return {
        "annotation_eligible_texts": len(eligible),
        "common_valid_texts": len(valid),
        "arms": outputs,
        "definition": "Reviewer/dataset hateful and type-assigned mask; predicted gate retained; seven labels; absent-support F1=0; mean of three repeat F1s.",
    }


def analyze(roster, predictions):
    expected = {
        (r["key"], arm, rep) for r in roster for arm in c.CONDITIONS for rep in range(3)
    }
    c.require(
        set(predictions) == expected, "Missing, extra or duplicate prediction identity"
    )
    for p in predictions.values():
        c.require(
            type(p["n_valid"]) is int and 0 <= p["n_valid"] <= 3, "Invalid vote count"
        )
        c.require(
            (p["result"] is None) == (p["n_valid"] == 0), "Null/vote count mismatch"
        )
        if p["result"] is not None:
            c.require(
                type(p["result"]["hate"]) is bool
                and set(p["result"]["hate_type"]) <= set(c.LABELS),
                "Invalid result schema",
            )
    transfer = [r for r in roster if r["transfer_eligible"]]
    clusters = {}
    for r in sorted(transfer, key=lambda x: x["packet_id"]):
        clusters.setdefault(r["cluster"], r)
    cohorts = {
        "other_all": transfer,
        "other_changed_messages": [
            r for r in transfer if r["selected_correction"] != "R059"
        ],
        "other_identical_messages": [
            r for r in transfer if r["selected_correction"] == "R059"
        ],
        "both_reviewers_resolved": [
            r
            for r in transfer
            if all(c.reference_target(r, v)[0] is not None for v in ("A", "B"))
        ],
        "exclude_retrieval_replay": [
            r for r in transfer if not r["retrieval_replay_flag"]
        ],
        "exclude_training_overlap": [r for r in transfer if not r["touches_training"]],
        "one_per_overlap_cluster": list(clusters.values()),
    }
    report = {}
    for name, cohort in cohorts.items():
        report[name] = {
            v: {
                **{
                    ep: endpoint(cohort, predictions, v, ep)
                    for ep in ("gate", "complete", "jaccard")
                },
                "labels": label_statistics(cohort, predictions, v),
            }
            for v in ("A", "B", "dataset")
        }
    report["all_three_votes_valid"] = {
        v: {
            **{
                ep: endpoint(transfer, predictions, v, ep, True)
                for ep in ("gate", "complete", "jaccard")
            },
            "labels": label_statistics(transfer, predictions, v, True),
        }
        for v in ("A", "B", "dataset")
    }
    source = []
    for r in roster:
        if not r["is_correction_source"]:
            continue
        correction = c.agreed_correction(r)
        arms = {}
        for arm in c.CONDITIONS:
            arms[arm] = []
            for rep in range(3):
                p = value(predictions, r["key"], arm, rep)
                arms[arm].append(
                    {
                        "repeat": rep,
                        "prediction": p,
                        "n_valid": predictions[r["key"], arm, rep]["n_valid"],
                        "gate_uptake": (
                            None if p is None else p["hate"] == correction["gate"]
                        ),
                        "complete_uptake": (
                            None
                            if p is None or correction["hate_types"] is None
                            else bool(p["hate"])
                            and set(p["hate_type"]) == set(correction["hate_types"])
                        ),
                        "explicit_present": (
                            None
                            if p is None or correction["hate_types"] is None
                            else bool(p["hate"]) and "explicit" in p["hate_type"]
                        ),
                    }
                )
        source.append(
            {
                "alias": r["packet_id"],
                "agreement_target": correction,
                "arms": arms,
                "transfer": False,
            }
        )
    return {
        "status": "descriptive_analysis_requires_scientific_review",
        "source_uptake": source,
        "transfer": report,
        "null_predictions": [
            {"key": k, "condition": arm, "repeat": rep}
            for (k, arm, rep), p in predictions.items()
            if p["result"] is None
        ],
        "vote_coverage": dict(Counter(str(p["n_valid"]) for p in predictions.values())),
        "cohort_sizes": {k: len(v) for k, v in cohorts.items()},
        "independent_texts": len(roster),
        "new_human_judgments": 0,
        "new_model_calls_from_analysis": 0,
        "limitations": [
            "Selected, exposed, all-dataset-hateful sample",
            "Two novices; no adjudicated gold",
            "Three source cases are uptake only",
            "Only seven other texts have different human/dataset messages",
            "One update; no expert or successive-human-round result",
            "No population or causal-generalization inference",
        ],
    }

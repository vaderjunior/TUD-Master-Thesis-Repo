from collections import Counter, defaultdict
from fractions import Fraction as F
import base64
import json
import math
from unittest.mock import patch
from analysis.feedback.human import results as a

CONTRASTS = (
    ("human_feedback5", "dataset_feedback5"),
    ("human_feedback5", "reference5"),
    ("dataset_feedback5", "reference5"),
)
CHECKS = Counter()


def compare(expected, actual, path):
    if isinstance(expected, dict):
        a.need(set(expected) == set(actual), "Keys differ: " + path)
        for k, v in expected.items():
            compare(v, actual[k], path + "/" + k)
    elif isinstance(expected, list):
        a.need(len(expected) == len(actual), "Length differs: " + path)
        for i, (x, y) in enumerate(zip(expected, actual)):
            compare(x, y, path + "/" + str(i))
    elif isinstance(expected, F):
        a.need(
            isinstance(actual, (int, float))
            and math.isclose(float(expected), actual, rel_tol=1e-12, abs_tol=1e-12),
            "Exact arithmetic differs: "
            + path
            + " expected "
            + str(expected)
            + " actual "
            + str(actual),
        )
        CHECKS["exact_fraction_values"] += 1
    else:
        a.need(expected == actual, "Value differs: " + path)
        CHECKS["identities_counts_masks_and_nulls"] += 1


def target(r, ref):
    if ref == "dataset":
        return r["gold"]["hate"], frozenset(r["gold"]["hate_type"])
    v = r["judgments"][ref + "_project"]
    if v["binary_hate"] is None:
        return None, None
    if v["binary_hate"] is False:
        return False, frozenset()
    return True, frozenset(v["types"]) if v["type_state"] == "assigned" else None


def score(p, g, t, ep):
    if p is None:
        return None
    yes = p["hate"]
    labels = frozenset(p["hate_type"]) if yes else frozenset()
    if ep == "gate":
        return F(int(yes == g))
    if ep == "complete":
        return F(int(yes == g and labels == t))
    return F(len(labels & t), len(labels | t)) if labels | t else F(1)


def mean(values):
    return sum(values, F(0)) / len(values) if values else None


def endpoint(rows, predictions, ref, ep, strict=False):
    eligible = []
    vals = {}
    for r in rows:
        g, t = target(r, ref)
        if (
            g is None
            or (ep != "gate" and t is None)
            or (ep == "jaccard" and g is not True)
        ):
            continue
        eligible.append(r)
        vals[r["key"]] = {}
        for arm in a.ARMS:
            v = []
            for rep in range(3):
                record = predictions[r["key"], arm, rep]
                pred = (
                    record["prediction"]
                    if not strict or record["n_valid"] == 3
                    else None
                )
                v.append(score(pred, g, t, ep))
            vals[r["key"]][arm] = v
    common = [
        r
        for r in eligible
        if all(z is not None for v in vals[r["key"]].values() for z in v)
    ]
    arms = {
        arm: {
            "mean_text_agreement": mean([mean(vals[r["key"]][arm]) for r in common]),
            "per_repeat": [
                mean([vals[r["key"]][arm][i] for r in common]) for i in range(3)
            ],
        }
        for arm in a.ARMS
    }
    contrasts = {}
    for treated, baseline in CONTRASTS:
        paired = []
        by_rep = []
        for rep in range(3):
            d = [
                vals[r["key"]][treated][rep] - vals[r["key"]][baseline][rep]
                for r in common
            ]
            by_rep.append(mean(d))
            paired.append(
                {
                    "improved": sum(x > 0 for x in d),
                    "regressed": sum(x < 0 for x in d),
                    "unchanged": sum(x == 0 for x in d),
                }
            )
        low, high = [], []
        for r in eligible:
            for rep in range(3):
                x, y = vals[r["key"]][treated][rep], vals[r["key"]][baseline][rep]
                low.append((F(0) if x is None else x) - (F(1) if y is None else y))
                high.append((F(1) if x is None else x) - (F(0) if y is None else y))
        contrasts[treated + "_minus_" + baseline] = {
            "mean_difference": mean(by_rep) if common else None,
            "per_repeat_difference": by_rep,
            "paired_transitions_by_repeat": paired,
            "mean_text_transition_counts": {
                k: F(sum(x[k] for x in paired), 3)
                for k in ("improved", "regressed", "unchanged")
            },
            "missing_output_bounds_full_eligible": (
                [mean(low), mean(high)] if eligible else None
            ),
        }
    return {
        "annotation_eligible_texts": len(eligible),
        "common_valid_texts": len(common),
        "eligible_aliases": [r["packet_id"] for r in eligible],
        "common_aliases": [r["packet_id"] for r in common],
        "arms": arms,
        "contrasts": contrasts,
        "per_text": [
            {"alias": r["packet_id"], "key": r["key"], "scores": vals[r["key"]]}
            for r in eligible
        ],
    }


def labels(rows, predictions, ref, strict=False):
    eligible = [
        r for r in rows if target(r, ref)[0] is True and target(r, ref)[1] is not None
    ]
    valid = [
        r
        for r in eligible
        if all(
            predictions[r["key"], arm, rep]["prediction"] is not None
            and (not strict or predictions[r["key"], arm, rep]["n_valid"] == 3)
            for arm in a.ARMS
            for rep in range(3)
        )
    ]
    arms = {}
    for arm in a.ARMS:
        repeats = []
        for rep in range(3):
            ls = {}
            for label in a.LABELS:
                yy = [
                    (
                        label in target(r, ref)[1],
                        predictions[r["key"], arm, rep]["prediction"]["hate"]
                        and label
                        in predictions[r["key"], arm, rep]["prediction"]["hate_type"],
                    )
                    for r in valid
                ]
                tp = sum(t and p for t, p in yy)
                fp = sum(not t and p for t, p in yy)
                fn = sum(t and not p for t, p in yy)
                tn = sum(not t and not p for t, p in yy)
                ls[label] = {
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "tn": tn,
                    "support": tp + fn,
                    "f1": F(2 * tp, 2 * tp + fp + fn) if 2 * tp + fp + fn else F(0),
                }
            repeats.append(
                {
                    "labels": ls,
                    "macro_f1_fixed_seven": (
                        mean([z["f1"] for z in ls.values()]) if valid else None
                    ),
                }
            )
        arms[arm] = {
            "per_repeat": repeats,
            "mean_macro_f1": (
                mean([r["macro_f1_fixed_seven"] for r in repeats]) if valid else None
            ),
        }
    return {
        "annotation_eligible_texts": len(eligible),
        "common_valid_texts": len(valid),
        "arms": arms,
    }


def jsonable(value):
    if isinstance(value, F):
        return float(value)
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [jsonable(x) for x in value]
    return value


def main():
    a.need(
        not (a.OUT / "independent_verification.json").exists(),
        "Preserve prior verification",
    )
    raw = a.read(a.OUT / "raw_review.json")
    a.verify(raw["artifacts"])
    rows = a.rows(a.BASE / "roster.jsonl")
    records = a.read(a.OUT / "predictions.json")
    pred = {(p["key"], p["condition"], p["repeat"]): p for p in records}
    a.need(len(records) == len(pred) == 216, "Duplicate/missing identity")
    frozen = a.read(a.OUT / "frozen_descriptive_analysis.json")
    sources = []
    for r in rows:
        ga, ta = target(r, "A")
        gb, tb = target(r, "B")
        old = r["model_repeat_0"]
        if ga is not None and ga == gb:
            complete = ta if ta is not None and ta == tb else None
            if old["hate"] != ga or (
                complete is not None and frozenset(old["hate_type"]) != complete
            ):
                sources.append(r["packet_id"])
    compare(["R014", "R035", "R059"], sorted(sources), "source selection")
    transfer = [r for r in rows if r["transfer_eligible"]]
    prompts = {(p["key"], p["condition"]): p for p in a.rows(a.BASE / "prompts.jsonl")}
    changed = [
        r
        for r in transfer
        if prompts[r["key"], a.ARMS[1]]["messages"]
        != prompts[r["key"], a.ARMS[2]]["messages"]
    ]
    identical = [r for r in transfer if r not in changed]
    clusters = {}
    for r in sorted(transfer, key=lambda x: x["packet_id"]):
        clusters.setdefault(r["cluster"], r)
    cohorts = {
        "other_all": transfer,
        "other_changed_messages": changed,
        "other_identical_messages": identical,
        "both_reviewers_resolved": [
            r
            for r in transfer
            if target(r, "A")[0] is not None and target(r, "B")[0] is not None
        ],
        "exclude_retrieval_replay": [
            r for r in transfer if not r["retrieval_replay_flag"]
        ],
        "exclude_training_overlap": [r for r in transfer if not r["touches_training"]],
        "one_per_overlap_cluster": list(clusters.values()),
        "all_three_votes_valid": transfer,
    }
    compare(
        {k: len(v) for k, v in cohorts.items() if k != "all_three_votes_valid"},
        frozen["cohort_sizes"],
        "cohort sizes",
    )
    expected_endpoint = {}
    for name, cohort in cohorts.items():
        for ref in ("A", "B", "dataset"):
            for ep in ("gate", "complete", "jaccard"):
                result = endpoint(
                    cohort, pred, ref, ep, name == "all_three_votes_valid"
                )
                compare(
                    result,
                    frozen["transfer"][name][ref][ep],
                    name + "/" + ref + "/" + ep,
                )
                expected_endpoint[name, ref, ep] = result
            stat = labels(cohort, pred, ref, name == "all_three_votes_valid")
            compare(
                stat,
                {
                    k: v
                    for k, v in frozen["transfer"][name][ref]["labels"].items()
                    if k != "definition"
                },
                name + "/" + ref + "/labels",
            )
    source_checks = []
    for entry in frozen["source_uptake"]:
        r = next(r for r in rows if r["packet_id"] == entry["alias"])
        g, t = target(r, "A")
        _, other_t = target(r, "B")
        t = t if t == other_t else None
        for arm in a.ARMS:
            for rep in range(3):
                p = pred[r["key"], arm, rep]["prediction"]
                s = entry["arms"][arm][rep]
                compare(p, s["prediction"], "source prediction")
                compare(p["hate"] == g, s["gate_uptake"], "source gate")
                compare(
                    None if t is None else p["hate"] and frozenset(p["hate_type"]) == t,
                    s["complete_uptake"],
                    "source complete",
                )
                compare(
                    None if t is None else p["hate"] and "explicit" in p["hate_type"],
                    s["explicit_present"],
                    "source explicit",
                )
        source_checks.append(
            {
                "alias": entry["alias"],
                "prior_model_gate": r["model_repeat_0"]["hate"],
                "prior_model_types": r["model_repeat_0"]["hate_type"],
                "gate_successes": {
                    arm: sum(s["gate_uptake"] for s in entry["arms"][arm])
                    for arm in a.ARMS
                },
                "complete_successes": {
                    arm: (
                        None
                        if t is None
                        else sum(s["complete_uptake"] for s in entry["arms"][arm])
                    )
                    for arm in a.ARMS
                },
                "explicit_successes": {
                    arm: (
                        None
                        if t is None
                        else sum(s["explicit_present"] for s in entry["arms"][arm])
                    )
                    for arm in a.ARMS
                },
            }
        )

    transition_rows, transition_summaries = [], []
    for name, cohort in cohorts.items():
        for ref in ("A", "B", "dataset"):
            for ep in ("gate", "complete", "jaccard"):
                d = expected_endpoint[name, ref, ep]
                for treatment, control in CONTRASTS:
                    directions = Counter()
                    aliases = defaultdict(list)
                    for text in d["per_text"]:
                        x, y = text["scores"][treatment], text["scores"][control]
                        if any(v is None for v in x + y):
                            continue
                        diff = [s - t for s, t in zip(x, y)]
                        avg = mean(diff)
                        category = (
                            "improved"
                            if avg > 0
                            else (
                                "regressed"
                                if avg < 0
                                else (
                                    "equal_mean_opposing_changes"
                                    if any(v != 0 for v in diff)
                                    else "unchanged_each_repeat"
                                )
                            )
                        )
                        directions[category] += 1
                        aliases[category].append(text["alias"])
                        transition_rows.append(
                            {
                                "cohort": name,
                                "reference": ref,
                                "endpoint": ep,
                                "treatment": treatment,
                                "control": control,
                                "alias": text["alias"],
                                "treatment_scores": x,
                                "control_scores": y,
                                "mean_difference": avg,
                                "category": category,
                            }
                        )
                    transition_summaries.append(
                        {
                            "cohort": name,
                            "reference": ref,
                            "endpoint": ep,
                            "treatment": treatment,
                            "control": control,
                            "directions": dict(directions),
                            "aliases": dict(aliases),
                        }
                    )
    label_changes = []
    for r in rows:
        for treat, control in CONTRASTS:
            for rep in range(3):
                x, y = (
                    pred[r["key"], treat, rep]["prediction"],
                    pred[r["key"], control, rep]["prediction"],
                )
                added = set(x["hate_type"]) - set(y["hate_type"])
                removed = set(y["hate_type"]) - set(x["hate_type"])
                label_changes.append(
                    {
                        "alias": r["packet_id"],
                        "source": r["is_correction_source"],
                        "selected_correction": r["selected_correction"],
                        "treatment": treat,
                        "control": control,
                        "repeat": rep,
                        "control_gate": y["hate"],
                        "treatment_gate": x["hate"],
                        "added_labels": sorted(added),
                        "removed_labels": sorted(removed),
                        "same_gate_and_types": x["hate"] == y["hate"]
                        and set(x["hate_type"]) == set(y["hate_type"]),
                    }
                )
    variation = {}
    for name, cohort in (
        ("changed_transfer", changed),
        ("identical_transfer", identical),
        ("all_transfer", transfer),
    ):
        subset = {r["packet_id"] for r in cohort}
        pairs = [
            r
            for r in label_changes
            if r["alias"] in subset
            and r["treatment"] == "human_feedback5"
            and r["control"] == "dataset_feedback5"
        ]
        varying = [r for r in pairs if not r["same_gate_and_types"]]
        variation[name] = {
            "texts": len(cohort),
            "paired_repeat_occurrences": len(pairs),
            "same_gate_and_type_set": sum(r["same_gate_and_types"] for r in pairs),
            "different_gate_and_or_type_set": len(varying),
            "gate_changes": sum(
                r["control_gate"] != r["treatment_gate"] for r in pairs
            ),
            "aliases_with_output_variation": sorted({r["alias"] for r in varying}),
            "gained_labels": dict(
                Counter(label for r in pairs for label in r["added_labels"])
            ),
            "lost_labels": dict(
                Counter(label for r in pairs for label in r["removed_labels"])
            ),
        }
    strata = {}
    for stratum in sorted({r["stratum"] for r in rows}):
        cohort = [r for r in transfer if r["stratum"] == stratum]
        strata[stratum] = {
            "n": len(cohort),
            "aliases": [r["packet_id"] for r in cohort],
            "results": {
                ref: {
                    ep: endpoint(cohort, pred, ref, ep)
                    for ep in ("gate", "complete", "jaccard")
                }
                for ref in ("A", "B", "dataset")
            },
        }
    repeat_variation = []
    for r in rows:
        for arm in a.ARMS:
            pp = [pred[r["key"], arm, i]["prediction"] for i in range(3)]
            repeat_variation.append(
                {
                    "alias": r["packet_id"],
                    "condition": arm,
                    "distinct_gate_and_type_outputs": len(
                        {(p["hate"], tuple(sorted(p["hate_type"]))) for p in pp}
                    ),
                    "distinct_gates": len({p["hate"] for p in pp}),
                    "type_cardinalities": [len(p["hate_type"]) for p in pp],
                }
            )

    calls = a.read(a.OUT / "all_calls.json")
    usage = {arm: Counter() for arm in a.ARMS}
    errors = Counter()
    times = defaultdict(list)
    waits = defaultdict(list)
    for call in calls:
        path = a.ROOT / call["outcome_path"]
        a.need(a.sha(path) == call["outcome_sha256"], "Raw outcome changed")
        raw = a.read(path)
        arm = call["condition"]
        times[arm].append(raw["elapsed_s"])
        waits[arm].append(call["backoff_elapsed_s"])
        usage[arm]["post_attempts"] += 1
        if raw["kind"] == "transport_error":
            errors[raw["category"]] += 1
            continue
        body = json.loads(base64.b64decode(raw["body_base64"], validate=True))
        compare(
            {
                k: body["usage"][k]
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
            {
                k: call["usage"][k]
                for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            },
            "raw usage",
        )
        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
            usage[arm][k] += body["usage"][k]
            usage[arm][call["kind"] + "_" + k] += body["usage"][k]
        usage[arm]["http_success_with_text"] += 1
    resources = a.read(a.OUT / "resources.json")
    for arm, result in usage.items():
        compare(
            dict(result),
            {k: resources["by_condition"][arm][k] for k in result},
            "independent resources",
        )
        for field, values in (
            ("POST_seconds_sum", times[arm]),
            ("retry_wait_seconds_sum", waits[arm]),
        ):
            a.need(
                math.isclose(
                    math.fsum(values),
                    resources["by_condition"][arm][field],
                    abs_tol=1e-8,
                ),
                "Timing sum differs",
            )
    compare(
        dict(errors), resources["transport_error_categories"], "transport categories"
    )
    a.new(a.OUT / "text_transitions.json", jsonable(transition_rows))
    a.new(a.OUT / "transition_summaries.json", transition_summaries)
    a.new(a.OUT / "label_changes.json", label_changes)
    a.new(a.OUT / "variation.json", variation)
    a.new(a.OUT / "original_strata.json", jsonable(strata))
    a.new(a.OUT / "repeat_variation.json", repeat_variation)
    a.new(a.OUT / "source_uptake_summary.json", source_checks)
    a.new(
        a.OUT / "cohort_aliases.json",
        {k: [r["packet_id"] for r in v] for k, v in cohorts.items()},
    )
    prior = a.read(a.OUT / "before.json")
    a.verify(prior["live_files"])
    a.verify(prior["frozen_bindings"])
    a.verify(prior["receipt_bindings"])
    a.new(
        a.OUT / "independent_verification.json",
        {
            "status": "all_frozen_endpoints_labels_transitions_bounds_and_resources_match",
            "created_utc": a.utc(),
            "checks": dict(CHECKS),
            "cohorts": len(cohorts),
            "references": 3,
            "endpoint_objects": len(expected_endpoint),
            "per_label_tables": len(cohorts) * 3 * 3 * 3 * 7,
            "source_predictions": 27,
            "raw_attempt_resources_checked": len(calls),
            "transition_records": len(transition_rows),
            "originals_unchanged": True,
            "new_model_calls": 0,
            "checker_sha256": a.sha(__file__),
            "artifacts": {
                a.rel(p): a.sha(p)
                for p in [
                    a.OUT / "text_transitions.json",
                    a.OUT / "transition_summaries.json",
                    a.OUT / "label_changes.json",
                    a.OUT / "variation.json",
                    a.OUT / "original_strata.json",
                    a.OUT / "repeat_variation.json",
                    a.OUT / "source_uptake_summary.json",
                    a.OUT / "cohort_aliases.json",
                ]
            },
        },
    )
    print(
        json.dumps(
            {
                "checks": dict(CHECKS),
                "source_uptake": source_checks,
                "variation": variation,
                "main_endpoints": {
                    ref: {
                        ep: {
                            "n": frozen["transfer"]["other_all"][ref][ep][
                                "common_valid_texts"
                            ],
                            "arms": frozen["transfer"]["other_all"][ref][ep]["arms"],
                        }
                        for ep in ("gate", "complete", "jaccard")
                    }
                    for ref in ("A", "B", "dataset")
                },
                "macro_f1": {
                    ref: frozen["transfer"]["other_all"][ref]["labels"]["arms"]
                    for ref in ()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    with patch("socket.socket", side_effect=RuntimeError("Offline review only")), patch(
        "socket.create_connection", side_effect=RuntimeError("Offline review only")
    ):
        main()

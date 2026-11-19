from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
import socket
from unittest.mock import patch
from analysis.feedback.balanced import archive as r

POLICIES = [
    "reference",
    "error_positive",
    "error_balanced",
    "correct_positive",
    "correct_balanced",
    "random_positive",
    "random_balanced",
]


def close(a, b):
    r.require(abs(a - b) < 2e-12, "Independent diagnostic number differs")


def main():
    dest = r.DEST / "deep_check"
    done = r.read(r.DEST / "diagnostics/complete.json")
    r.verify(done["inputs"])
    r.verify(done["outputs"])
    r.require(not (dest / "complete.json").exists(), "Preserve existing deep review")
    inputs = {
        r.rel(p): r.sha(p)
        for p in [
            Path(__file__),
            r.DEST / "diagnostics/complete.json",
            r.REVIEW / "predictions.jsonl",
            r.EXEC.parent / "choices.jsonl",
            r.EXEC.parent / "roster.jsonl",
            r.EXEC / "recovery_20261006_v1/manifest.json",
        ]
    }
    r.save(
        dest / "plan.json",
        dict(
            created_utc=r.utc(),
            inputs=inputs,
            verification="Independently reconstruct transitions, delivered polarity and costs; verify remaining per-label/exact-match/coverage fields and reference-relative transitions.",
            exploratory="Added after primary results were seen: remove all inputs with any primary retry, remove the eight-call amendment input union, and remove all inputs receiving each source in either primary arm. Apply each exclusion identically to both arms/pools/repeats; retain original primary common masks otherwise. All-source influence range, not selected favourable source. Report point estimates only, no new discovery tests.",
            limits="Operational and source-conditioned deletion is nonrandom and alters the population; not bias correction or replacement primary. Exact identical-prompt cases reveal response variation, not hidden checkpoint equality.",
        ),
    )
    records = {
        (x["key"], x["condition"], x["repeat"]): x
        for x in r.rows(r.REVIEW / "predictions.jsonl")
    }
    roster = {x["key"]: x for x in r.rows(r.EXEC.parent / "roster.jsonl")}
    choices = {
        (x["key"], x["condition"]): x for x in r.rows(r.EXEC.parent / "choices.jsonl")
    }
    saved = r.read(r.DEST / "diagnostics/transitions.json")
    cases = r.read(r.DEST / "diagnostics/transition_cases.json")
    expected = {}
    seen = set()
    counts = Counter()
    for selection in ("error", "correct", "random"):
        for pop, truth in [("benign", False), ("hateful", True)]:
            score = r.read(
                r.DEST
                / pop
                / ("primary.json" if selection == "error" else "all_policies.json")
            )
            categories = defaultdict(Counter)
            members = defaultdict(set)
            bygate = defaultdict(Counter)
            bycos = defaultdict(Counter)
            for key in score["common_ids"]:
                for seed in r.SEEDS:
                    a = f"{selection}_positive_s{seed}"
                    b = f"{selection}_balanced_s{seed}"
                    ca = choices[key, a]
                    cb = choices[key, b]
                    category = (
                        "same_source"
                        if ca["source_id"] == cb["source_id"]
                        else "different_hateful" if cb["source_gate"] else "new_benign"
                    )
                    cosbin = (
                        "<0.4"
                        if cb["cosine"] < 0.4
                        else (
                            "0.4-0.5"
                            if cb["cosine"] < 0.5
                            else "0.5-0.6" if cb["cosine"] < 0.6 else ">=0.6"
                        )
                    )
                    members[category].add(key)
                    for rep in range(3):
                        pa = records[key, a, rep]
                        pb = records[key, b, rep]
                        x = pa["result"]
                        y = pb["result"]
                        r.require(
                            (pa["messages_sha256"] == pb["messages_sha256"])
                            == (category == "same_source"),
                            "Same-source prompt mismatch",
                        )
                        before = x["hate"] != truth
                        after = y["hate"] != truth
                        identity = (selection, pop, key, seed, rep)
                        expected[identity] = dict(
                            category=category,
                            positive_source=ca["source_id"],
                            balanced_source=cb["source_id"],
                            balanced_source_hate=cb["source_gate"],
                            positive_cosine=ca["cosine"],
                            balanced_cosine=cb["cosine"],
                            positive_error=before,
                            balanced_error=after,
                            positive_gate=x["hate"],
                            balanced_gate=y["hate"],
                            stage=pa["stage"],
                        )
                        for cell in [categories[category], bycos[cosbin]]:
                            cell.update(
                                pairs=1,
                                positive_errors=int(before),
                                balanced_errors=int(after),
                                fixed=int(before and not after),
                                harmed=int(after and not before),
                            )
                        if truth:
                            cell = bygate[f'{int(x["hate"])}->{int(y["hate"])}']
                            cell["pairs"] += 1
                            gold = set(roster[key]["gold"]["hate_type"])

                            for label in r.LABELS:
                                old = label in x["hate_type"]
                                new = label in y["hate_type"]
                                tag = "correct" if label in gold else "wrong"
                                cell[tag + "_labels_gained"] += int(new and not old)
                                cell[tag + "_labels_lost"] += int(old and not new)
            target = saved[selection + "/" + pop]
            r.require(
                target["pool_repeat_pairs"] == 9 * len(score["common_ids"]),
                "Transition denominator differs",
            )
            for category, v in categories.items():
                w = target["categories"][category]
                for k, n in v.items():
                    r.require(w[k] == n, "Transition counts differ")
                r.require(
                    w["unique_inputs"] == len(members[category]),
                    "Transition unique input count differs",
                )
                close(
                    w["conditional_difference"],
                    (v["balanced_errors"] - v["positive_errors"]) / v["pairs"],
                )
                close(
                    w["contribution_to_overall_difference"],
                    (v["balanced_errors"] - v["positive_errors"])
                    / target["pool_repeat_pairs"],
                )
            r.require(
                {k: dict(v) for k, v in bygate.items()}
                == target["fine_labels_by_gate_transition"],
                "Fine-label gate transitions differ",
            )
            r.require(
                {k: dict(v) for k, v in bycos.items()}
                == target["by_balanced_source_cosine"],
                "Cosine-bin transitions differ",
            )
            counts["transition_tables"] += 1
    for case in cases:
        identity = tuple(
            case[k] for k in ["selection", "population", "key", "seed", "repeat"]
        )
        r.require(
            identity in expected and identity not in seen,
            "Transition missing/duplicate identity",
        )
        r.require(
            all(case[k] == v for k, v in expected[identity].items()),
            "Transition row differs",
        )
        seen.add(identity)
    r.require(seen == set(expected), "Incomplete transition case table")
    counts["transition_rows"] = len(seen)

    fine = r.read(r.DEST / "types/all_policies.json")
    fine_summary = {}
    type_ids = fine["common_ids"]
    for policy, conditions in fine["condition_groups"].items():
        runs = []
        for condition in conditions:
            for rep in range(3):
                ids = type_ids
                ps = [records[k, condition, rep]["result"] for k in ids]
                gs = [set(roster[k]["gold"]["hate_type"]) for k in ids]
                saved_run = fine["settings"][f"{condition}_rep{rep}"]["common_valid"]
                per = {}
                support = []
                label_count = 0
                exact = 0
                contains = 0
                gate_positive = 0
                for label in r.LABELS:
                    tp = sum(
                        label in g and label in p["hate_type"] for g, p in zip(gs, ps)
                    )
                    fp = sum(
                        label not in g and label in p["hate_type"]
                        for g, p in zip(gs, ps)
                    )
                    fn = sum(
                        label in g and label not in p["hate_type"]
                        for g, p in zip(gs, ps)
                    )
                    per[label] = dict(
                        tp=tp,
                        fp=fp,
                        fn=fn,
                        support=tp + fn,
                        f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
                        precision=tp / (tp + fp) if tp + fp else 0,
                        recall=tp / (tp + fn) if tp + fn else 0,
                    )
                for g, p in zip(gs, ps):
                    exact += g == set(p["hate_type"])
                    contains += g <= set(p["hate_type"])
                    label_count += len(p["hate_type"])
                    gate_positive += p["hate"]
                for field, key in [
                    ("per_label_f1", "f1"),
                    ("precision", "precision"),
                    ("recall", "recall"),
                    ("support", "support"),
                ]:
                    for i, label in enumerate(r.LABELS):
                        close(saved_run[field][i], per[label][key])
                for key, number in [
                    ("exact_match", exact),
                    ("contains_all_gold", contains),
                    ("predicted_labels", label_count),
                ]:
                    close(saved_run[key], number / len(ids))
                runs.append(
                    dict(
                        condition=condition,
                        repeat=rep,
                        n=len(ids),
                        exact_match=exact / len(ids),
                        contains_all_gold=contains / len(ids),
                        predicted_labels=label_count / len(ids),
                        hate_positive=gate_positive / len(ids),
                        per_label=per,
                    )
                )
                counts["fine_label_runs"] += 1
        fine_summary[policy] = dict(
            runs=runs,
            means={
                k: mean(v[k] for v in runs)
                for k in [
                    "exact_match",
                    "contains_all_gold",
                    "predicted_labels",
                    "hate_positive",
                ]
            },
            per_label={
                label: {
                    key: mean(v["per_label"][label][key] for v in runs)
                    for key in [
                        "tp",
                        "fp",
                        "fn",
                        "support",
                        "f1",
                        "precision",
                        "recall",
                    ]
                }
                for label in r.LABELS
            },
        )

    ref_trans = {}
    for pop, truth in [("benign", False), ("hateful", True)]:
        score = r.read(r.DEST / pop / "all_policies.json")
        ids = score["common_ids"]
        for policy, conditions in score["condition_groups"].items():
            if policy == "reference":
                continue
            values = Counter()
            for condition in conditions:
                for rep in range(3):
                    for key in ids:
                        before = (
                            records[key, "reference", rep]["result"]["hate"] != truth
                        )
                        after = records[key, condition, rep]["result"]["hate"] != truth
                        values["pairs"] += 1
                        values["fixed"] += before and not after
                        values["harmed"] += after and not before
                        values["both_correct"] += not before and not after
                        values["both_wrong"] += before and after
            close(
                (values["harmed"] - values["fixed"]) / values["pairs"],
                score["contrasts"][policy + "_minus_reference"]["difference"],
            )
            ref_trans[pop + "/" + policy] = dict(values) | dict(
                n=len(ids),
                difference=(values["harmed"] - values["fixed"]) / values["pairs"],
            )
    operational = r.read(r.DEST / "diagnostics/operational.json")
    costs = defaultdict(Counter)
    nulls = Counter()
    invalid = Counter()
    votes = Counter()
    for v in records.values():
        policy = v["condition"].split("_s202610")[0]
        costs[policy].update(v["costs"])
        costs[policy]["predictions"] += 1
        nulls[policy] += v["result"] is None
        invalid[policy] += v["n_valid"] < 3
        votes[v["n_valid"]] += 1
    for policy, v in costs.items():
        for key, value in v.items():
            close(operational["policies"][policy][key], value)
        r.require(
            operational["policies"][policy]["nulls"] == nulls[policy]
            and operational["policies"][policy]["predictions_with_any_invalid_vote"]
            == invalid[policy],
            "Operational policy counts differ",
        )
    for n, value in votes.items():
        r.require(
            operational["counts"]["valid_votes_" + str(n)] == value,
            "Vote validity count differs",
        )
    for field in operational["accounting"]:
        r.require(
            sum(v[field] for v in costs.values()) == operational["accounting"][field],
            "Aggregate request count differs",
        )
    for field in operational["observed_usage"]:
        r.require(
            sum(v[field] for v in costs.values())
            == operational["observed_usage"][field],
            "Token cost differs",
        )
    summary = r.read(r.REVIEW / "summary.json")
    r.require(
        all(v["recovered"] for v in summary["retry_events"]), "Unrecovered request"
    )
    r.require(
        sum(v["post_attempts"] for v in costs.values())
        == sum(v["http_success_with_text"] for v in costs.values())
        + len(summary["retry_events"]),
        "Physical request accounting differs",
    )
    counts["policies_cost_checked"] = len(costs)
    delivery = r.read(r.DEST / "diagnostics/delivery.json")
    delivery_aggregates = {}
    for policy in POLICIES[1:]:
        allc = [
            v for (key, cond), v in choices.items() if cond.startswith(policy + "_s")
        ]
        popvalues = {}
        for pop, truth in [("benign", False), ("hateful", True)]:
            these = [v for v in allc if roster[v["key"]]["gold"]["hate"] == truth]
            popvalues[pop] = dict(
                input_pool_choices=len(these),
                benign_sources=sum(not v["source_gate"] for v in these),
                benign_share=mean(not v["source_gate"] for v in these),
            )
        for seed in r.SEEDS:
            condition = f"{policy}_s{seed}"
            these = [v for v in allc if v["condition"] == condition]
            uses = Counter(v["source_id"] for v in these)
            saved_delivery = delivery["conditions"][condition]
            r.require(
                saved_delivery["source_use_counts"] == dict(uses),
                "Source use identity/count differs",
            )
            for pop, truth in [("benign", False), ("hateful", True)]:
                rows = [v for v in these if roster[v["key"]]["gold"]["hate"] == truth]
                w = saved_delivery["population"][pop]
                r.require(
                    w["inputs"] == len(rows)
                    and w["delivered_benign"]
                    == sum(not v["source_gate"] for v in rows),
                    "Delivered polarity differs",
                )
            counts["delivery_settings"] += 1
        delivery_aggregates[policy] = popvalues

    amendment = r.read(r.EXEC / "recovery_20261006_v1/manifest.json")
    amended = {x["job"]["key"] for x in amendment["calls"]}
    retry_keys = {
        v["key"]
        for v in records.values()
        if v["condition"].startswith(("error_positive_", "error_balanced_"))
        and v["costs"]["retry_attempts"] > 0
    }
    influence = {}
    sensitivity = {}
    source_inputs = defaultdict(set)
    for (key, condition), v in choices.items():
        if condition.startswith(("error_positive_", "error_balanced_")):
            source_inputs[v["source_id"]].add(key)
    for pop, truth in [("benign", False), ("hateful", True)]:
        primary = r.read(r.DEST / pop / "primary.json")
        ids = primary["common_ids"]

        def points(exclusions):
            selected = [k for k in ids if k not in exclusions]
            means = {}
            per_pool = []
            for policy in ["error_positive", "error_balanced"]:
                means[policy] = mean(
                    records[k, f"{policy}_s{s}", rep]["result"]["hate"] != truth
                    for k in selected
                    for s in r.SEEDS
                    for rep in range(3)
                )
            return dict(
                n=len(selected),
                excluded_ids=sorted(set(ids) - set(selected)),
                rates=means,
                difference=means["error_balanced"] - means["error_positive"],
            )

        sensitivity[pop] = {
            name: points(keys)
            for name, keys in [
                ("drop_all_primary_retry_inputs", retry_keys),
                ("drop_amended_call_inputs", amended),
            ]
        }
        source_cases = [
            dict(source=source, **points(keys))
            for source, keys in sorted(source_inputs.items())
            if set(ids) & keys
        ]
        influence[pop] = dict(
            cases=source_cases,
            source_count=len(source_cases),
            original_n=len(ids),
            difference_range=[
                min(v["difference"] for v in source_cases),
                max(v["difference"] for v in source_cases),
            ],
            excluded_n_range=[
                min(len(v["excluded_ids"]) for v in source_cases),
                max(len(v["excluded_ids"]) for v in source_cases),
            ],
            interpretation="Deletion uses the union of source-receiving inputs over both arms/all seeds, then removes those inputs everywhere. It does not rerun retrieval after removing a KB source.",
        )
    outputs = {}
    for name, value in [
        ("fine_details", fine_summary),
        ("reference_transitions", ref_trans),
        ("delivery_aggregates", delivery_aggregates),
        ("source_influence", influence),
        (
            "operational_exclusion_sensitivity",
            dict(
                amended_input_ids=sorted(amended),
                retry_input_ids=sorted(retry_keys),
                populations=sensitivity,
            ),
        ),
        ("verification", dict(counts)),
    ]:
        p = dest / (name + ".json")
        r.save(p, value)
        outputs[r.rel(p)] = r.sha(p)
    r.verify(inputs)
    r.save(
        dest / "complete.json",
        dict(
            status="independent_supporting_check_and_descriptive_sensitivities_complete",
            created_utc=r.utc(),
            inputs=inputs,
            outputs=outputs,
            new_model_calls=0,
        ),
    )
    print(dict(counts), flush=True)
    print("Operational exclusions", sensitivity, flush=True)
    print(
        "All-source deletion ranges",
        {
            k: {
                n: v[n]
                for n in ["source_count", "difference_range", "excluded_n_range"]
            }
            for k, v in influence.items()
        },
        flush=True,
    )


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", r.forbid), patch.object(
        socket, "create_connection", r.forbid
    ):
        main()

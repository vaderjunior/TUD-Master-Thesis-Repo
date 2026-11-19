from collections import Counter, defaultdict
import json
from pathlib import Path
import socket
from statistics import mean
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


def metric(rows, records, condition, rep):
    chosen = [(row, records[row["key"], condition, rep]["result"]) for row in rows]
    r.require(all(p is not None for _, p in chosen), "Continuity common mask missing")
    counts = Counter()
    types = {label: Counter() for label in r.LABELS}
    for row, p in chosen:
        hate = row["gold"]["hate"]
        counts[
            (
                "tp"
                if hate and p["hate"]
                else "fn" if hate else "fp" if p["hate"] else "tn"
            )
        ] += 1
        if hate:
            gold = set(row["gold"]["hate_type"])
            pred = set(p["hate_type"])
            for label, v in types.items():
                v["tp"] += label in gold and label in pred
                v["fp"] += label not in gold and label in pred
                v["fn"] += label in gold and label not in pred
    return dict(
        counts=dict(counts),
        fpr=counts["fp"] / (counts["fp"] + counts["tn"]),
        fnr=counts["fn"] / (counts["fn"] + counts["tp"]),
        type_f1=mean(
            (
                2 * v["tp"] / (2 * v["tp"] + v["fp"] + v["fn"])
                if 2 * v["tp"] + v["fp"] + v["fn"]
                else 0
            )
            for v in types.values()
        ),
        per_type={k: dict(v) for k, v in types.items()},
    )


def continuity(new, roster):
    previous = r.ROOT / "experiments/astra/sq3_combined_results_20261004_v1/review"
    path = previous / "predictions.jsonl"
    done = r.read(previous / "complete.json")
    r.require(done["outputs"][r.rel(path)] == r.sha(path), "Old audited export changed")
    old = {
        (v["key"], v["condition"], v["repeat"]): v
        for v in r.rows(path)
        if v["study"] == "taxonomy" and v["condition"] == "rag_full"
    }
    r.require(len(old) == 2025, "Historical reference incomplete")
    shared = []
    checks = 0
    for key, row in roster.items():
        valid = True
        for rep in range(3):
            a = old[key, "rag_full", rep]
            b = new[key, "reference", rep]
            r.require(
                a["messages_sha256"] == b["messages_sha256"]
                and a["gold"] == row["gold"] == b["gold"],
                "Historical reference identity changed",
            )
            checks += 1
            valid &= a["result"] is not None and b["result"] is not None
        if valid:
            shared.append(row)
    runs = {
        phase: [metric(shared, recs, condition, rep) for rep in range(3)]
        for phase, recs, condition in [
            ("old", old, "rag_full"),
            ("fresh", new, "reference"),
        ]
    }
    return dict(
        status="descriptive_identical_prompt_continuity",
        message_gold_checks=checks,
        original_n=675,
        common_n=len(shared),
        excluded_ids=sorted(set(roster) - {v["key"] for v in shared}),
        runs=runs,
        means={
            phase: {
                name: mean(v[name] for v in values)
                for name in ["fpr", "fnr", "type_f1"]
            }
            for phase, values in runs.items()
        },
        inputs={
            r.rel(path): r.sha(path),
            r.rel(previous / "complete.json"): r.sha(previous / "complete.json"),
        },
        limitation="Same exposed evaluation population and identical messages, fresh responses in a later phase. Descriptive continuity, not a correction treatment effect or proof of backend checkpoint equivalence.",
    )


def transitions(records, roster, choices):
    summaries = {}
    cases = []
    for selection in ["error", "correct", "random"]:
        for truth, pop in [(False, "benign"), (True, "hateful")]:
            scored = r.read(
                r.DEST
                / pop
                / ("primary.json" if selection == "error" else "all_policies.json")
            )
            ids = scored["common_ids"]
            categories = defaultdict(Counter)
            members = defaultdict(set)
            by_gate = defaultdict(Counter)
            by_similarity = defaultdict(Counter)
            for seed in r.SEEDS:
                positive = f"{selection}_positive_s{seed}"
                balanced = f"{selection}_balanced_s{seed}"
                for key in ids:
                    ca = choices[key, positive]
                    cb = choices[key, balanced]
                    category = (
                        "same_source"
                        if ca["source_id"] == cb["source_id"]
                        else (
                            "new_benign"
                            if not cb["source_gate"]
                            else "different_hateful"
                        )
                    )
                    bin_name = (
                        "<0.4"
                        if cb["cosine"] < 0.4
                        else (
                            "0.4-0.5"
                            if cb["cosine"] < 0.5
                            else "0.5-0.6" if cb["cosine"] < 0.6 else ">=0.6"
                        )
                    )
                    gold = set(roster[key]["gold"]["hate_type"] or [])
                    for rep in range(3):
                        a = records[key, positive, rep]
                        b = records[key, balanced, rep]
                        p = a["result"]
                        q = b["result"]
                        r.require(
                            p is not None and q is not None,
                            "Invalid transition entered common mask",
                        )
                        r.require(
                            (a["messages_sha256"] == b["messages_sha256"])
                            == (category == "same_source"),
                            "Matched-source message identity differs",
                        )
                        before = p["hate"] != truth
                        after = q["hate"] != truth
                        for v in [categories[category], by_similarity[bin_name]]:
                            v["pairs"] += 1
                            v["positive_errors"] += before
                            v["balanced_errors"] += after
                            v["fixed"] += before and not after
                            v["harmed"] += not before and after
                        members[category].add(key)
                        if truth:
                            gate = f'{int(p["hate"])}->{int(q["hate"])}'
                            v = by_gate[gate]
                            v["pairs"] += 1
                            gained = set(q["hate_type"]) - set(p["hate_type"])
                            lost = set(p["hate_type"]) - set(q["hate_type"])
                            v["correct_labels_gained"] += len(gained & gold)
                            v["correct_labels_lost"] += len(lost & gold)
                            v["wrong_labels_gained"] += len(gained - gold)
                            v["wrong_labels_lost"] += len(lost - gold)
                        cases.append(
                            dict(
                                selection=selection,
                                population=pop,
                                key=key,
                                seed=seed,
                                repeat=rep,
                                category=category,
                                positive_source=ca["source_id"],
                                balanced_source=cb["source_id"],
                                balanced_source_hate=cb["source_gate"],
                                positive_cosine=ca["cosine"],
                                balanced_cosine=cb["cosine"],
                                positive_error=before,
                                balanced_error=after,
                                positive_gate=p["hate"],
                                balanced_gate=q["hate"],
                                stage=a["stage"],
                            )
                        )
            total = 9 * len(ids)
            net = sum(
                v["balanced_errors"] - v["positive_errors"] for v in categories.values()
            )
            expected = scored["contrasts"][
                selection + "_balanced_minus_" + selection + "_positive"
            ]["difference"]
            r.require(
                sum(v["pairs"] for v in categories.values()) == total
                and abs(net / total - expected) < 1e-12,
                "Transition decomposition differs from scored contrast",
            )
            summaries[selection + "/" + pop] = dict(
                common_n=len(ids),
                pool_repeat_pairs=total,
                difference=net / total,
                categories={
                    k: dict(v)
                    | dict(
                        unique_inputs=len(members[k]),
                        conditional_difference=(
                            v["balanced_errors"] - v["positive_errors"]
                        )
                        / v["pairs"],
                        contribution_to_overall_difference=(
                            v["balanced_errors"] - v["positive_errors"]
                        )
                        / total,
                    )
                    for k, v in categories.items()
                },
                by_balanced_source_cosine={
                    k: dict(v) for k, v in by_similarity.items()
                },
                fine_labels_by_gate_transition={k: dict(v) for k, v in by_gate.items()},
                limitation="Conditioning on selected source/polarity/cosine is observational. Source text and label composition change together. Counts across pools/repeats are dependent, not independent sample size. Label-count transitions do not additively decompose macro-F1.",
            )
    return summaries, cases


def delivery(roster, choices):
    selection = (
        r.ROOT / "experiments/astra/sq3_additions_results_20261006_v1/benign_selection"
    )
    pools = r.read(selection / "actual_pools.json")
    statuses = r.read(selection / "candidate_statuses.json")
    out = {}
    pairs = []
    mapping = {p["condition"]: p for p in pools["pools"]}
    for condition, pool in mapping.items():
        selected = [c for (key, cond), c in choices.items() if cond == condition]
        r.require(
            len(selected) == 675
            and all(v["source_id"] in pool["ids"] for v in selected),
            "Delivery/pool identity differs",
        )
        uses = Counter(v["source_id"] for v in selected)
        pop = {}
        for truth, label in [(False, "benign"), (True, "hateful")]:
            rows = [v for v in selected if roster[v["key"]]["gold"]["hate"] == truth]
            pop[label] = dict(
                inputs=len(rows),
                delivered_benign=sum(not v["source_gate"] for v in rows),
                delivered_hateful=sum(v["source_gate"] for v in rows),
                delivered_prior_errors=sum(
                    statuses[v["source_id"]]["status"] == "error" for v in rows
                ),
                cosine_mean=mean(v["cosine"] for v in rows),
                cosine_min=min(v["cosine"] for v in rows),
                cosine_max=max(v["cosine"] for v in rows),
            )
        out[condition] = dict(
            pool_actual_status_counts=pool["actual_status_counts"],
            population=pop,
            unique_delivered_sources=len(uses),
            unused_pool_sources=sorted(set(pool["ids"]) - set(uses)),
            source_use_counts=dict(uses),
            top_source_share=max(uses.values()) / 675,
        )
    for selection_name in ["error", "correct", "random"]:
        for seed in r.SEEDS:
            a = mapping[f"{selection_name}_positive_s{seed}"]
            b = mapping[f"{selection_name}_balanced_s{seed}"]
            r.require(
                a["ids"][:20] == b["ids"][:20]
                and len(set(a["ids"]) & set(b["ids"])) == 20,
                "Balanced matched-half rule differs",
            )
            pairs.append(
                dict(selection=selection_name, seed=seed, shared=20, jaccard=20 / 60)
            )
    return dict(
        conditions=out,
        matched_halves=pairs,
        actual_pool_overlaps=r.read(selection / "summary.json")["pool_overlaps"],
        labels_inspected=1024,
        pool_budget=40,
        distinct_sources=pools["distinct_sources"],
    )


def operational(records):
    review = r.read(r.REVIEW / "summary.json")
    costs = defaultdict(Counter)
    null_by_policy = Counter()
    invalid_by_policy = Counter()
    null_keys = Counter()
    error_fields = Counter()
    for row in records.values():
        policy = row["condition"].split("_s202610")[0]
        costs[policy].update(row["costs"])
        costs[policy]["predictions"] += 1
        if row["result"] is None:
            null_by_policy[policy] += 1
            null_keys[row["key"]] += 1
        if row["n_valid"] < 3:
            invalid_by_policy[policy] += 1
    for report in review["stage_operational"].values():
        for message, n in report["validation_errors"].items():
            field = (
                "JSON"
                if message.startswith("invalid JSON")
                else next(
                    (
                        k
                        for k in [
                            "hate_type",
                            "target_group",
                            "severity",
                            "legal",
                            "hate",
                        ]
                        if "('" + k + "'" in message
                    ),
                    "other",
                )
            )
            error_fields[field] += n
    return dict(
        counts=review["counts"],
        accounting=review["operational"]["accounting"],
        observed_usage=review["operational"]["observed_usage"],
        policies={
            k: dict(v)
            | dict(
                nulls=null_by_policy[k],
                predictions_with_any_invalid_vote=invalid_by_policy[k],
                mean_total_tokens=v["total_tokens"] / v["predictions"],
            )
            for k, v in costs.items()
        },
        null_inputs=dict(null_keys),
        raw_validation_failures_by_field=dict(error_fields),
        retry_categories=dict(
            Counter(
                v["category"] or v["response_class"] for v in review["retry_events"]
            )
        ),
        stages={
            stage: {
                key: report[key]
                for key in [
                    "counts",
                    "accounting",
                    "observed_usage",
                    "reported_models",
                    "finish_reasons",
                    "wall_seconds_including_restarts",
                    "vote_seconds_sum",
                    "attempt_count",
                ]
            }
            for stage, report in review["stage_operational"].items()
        },
        limitation="Request/vote time excludes outage/restart downtime according to the preserved transport amendment; elapsed wall time includes it. Remote work behind failed transport is unknown. Reported model alias does not identify a hidden checkpoint. Source selection additionally required 1024 inspected labels and its separately reported 3072 baseline predictions.",
    )


def main():
    check = r.read(r.DEST / "scoring_independent_check.json")
    r.verify(check["inputs"])
    r.require(
        not (r.DEST / "diagnostics/complete.json").exists(),
        "Preserve completed diagnostics",
    )
    inputs = {
        r.rel(r.DEST / "analysis_plan.json"): r.sha(r.DEST / "analysis_plan.json"),
        r.rel(r.DEST / "scoring_independent_check.json"): r.sha(
            r.DEST / "scoring_independent_check.json"
        ),
        r.rel(r.REVIEW / "predictions.jsonl"): r.sha(r.REVIEW / "predictions.jsonl"),
        r.rel(r.EXEC.parent / "choices.jsonl"): r.sha(r.EXEC.parent / "choices.jsonl"),
        r.rel(Path(__file__)): r.sha(Path(__file__)),
    }
    r.save(
        r.DEST / "diagnostics/plan.json",
        dict(
            created_utc=r.utc(),
            inputs=inputs,
            scope="Supporting diagnostics specified in analysis plan; not additional primary tests. Delivery categories: same source, changed to benign, changed to hateful. Fixed cosine bins <.4, .4-.5, .5-.6, >=.6. Whole common masks and every pool/repeat retained.",
        ),
    )
    rows = list(r.rows(r.REVIEW / "predictions.jsonl"))
    records = {(v["key"], v["condition"], v["repeat"]): v for v in rows}
    roster = {v["key"]: v for v in r.rows(r.EXEC.parent / "roster.jsonl")}
    choices = {
        (v["key"], v["condition"]): v for v in r.rows(r.EXEC.parent / "choices.jsonl")
    }
    summary, cases = transitions(records, roster, choices)
    outputs = {}
    for name, value in [
        ("transitions", summary),
        ("transition_cases", cases),
        ("delivery", delivery(roster, choices)),
        ("historical_continuity", continuity(records, roster)),
        ("operational", operational(records)),
    ]:
        path = r.DEST / "diagnostics" / (name + ".json")
        r.save(path, value)
        outputs[r.rel(path)] = r.sha(path)
        print("Saved diagnostic", name, flush=True)
    r.verify(inputs)
    r.save(
        r.DEST / "diagnostics/complete.json",
        dict(
            status="supporting_diagnostics_complete",
            created_utc=r.utc(),
            inputs=inputs,
            outputs=outputs,
            new_model_calls=0,
        ),
    )


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", r.forbid), patch.object(
        socket, "create_connection", r.forbid
    ):
        main()

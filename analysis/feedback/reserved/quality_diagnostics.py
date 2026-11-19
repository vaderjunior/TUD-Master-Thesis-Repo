from collections import Counter, defaultdict
from pathlib import Path
import json
from analysis.feedback.reserved import quality as a


def outcome(row, p):
    base = {
        "key": row["key"],
        "known_type_gold": row["gold"]["hate_type"] is not None,
        "n_valid": p["n_valid"],
    }
    if p["result"] is None:
        return {**base, "status": "null"}
    pred = p["result"]
    out = {
        **base,
        "status": "valid",
        "gold_hate": row["gold"]["hate"],
        "predicted_hate": pred["hate"],
        "binary_correct": row["gold"]["hate"] == pred["hate"],
        "predicted_types": pred["hate_type"],
        "predicted_cardinality": len(pred["hate_type"]),
        "negative_gate_with_types": not pred["hate"] and bool(pred["hate_type"]),
    }
    if row["gold"]["hate_type"] is None:
        return out
    gold = set(row["gold"]["hate_type"])
    labels = set(pred["hate_type"])
    extra = labels - gold
    missing = gold - labels
    category = (
        "both_extra_and_missing"
        if extra and missing
        else "only_extra" if extra else "only_missing" if missing else "exact"
    )
    return {
        **out,
        "gold_types": sorted(gold),
        "gold_cardinality": len(gold),
        "extra": sorted(extra),
        "missing": sorted(missing),
        "category": category,
        "contains_gold": not missing,
        "gold_set": "|".join(sorted(gold)),
    }


def bucket():
    return {
        "counts": Counter(),
        "gold_cardinality": Counter(),
        "predicted_cardinality": Counter(),
        "gold_sets": defaultdict(Counter),
        "per_label": {l: Counter() for l in a.support.TYPES},
    }


def update(b, value):
    ct = b["counts"]
    ct["eligible_outputs"] += 1
    if value["status"] == "null":
        ct["nulls"] += 1
        return
    ct["valid_outputs"] += 1
    ct["binary_correct"] += value["binary_correct"]
    ct["unknown_type_gold"] += not value["known_type_gold"]
    ct["negative_gate_with_types"] += value["negative_gate_with_types"]
    if not value["known_type_gold"]:
        return
    ct["known_type_outputs"] += 1
    ct[value["category"]] += 1
    ct["contains_gold"] += value["contains_gold"]
    ct["extra_labels"] += len(value["extra"])
    ct["missing_labels"] += len(value["missing"])
    ct["predicted_labels"] += value["predicted_cardinality"]
    ct["gold_labels"] += value["gold_cardinality"]
    ct["gate_positive"] += value["predicted_hate"]
    ct["gate_negative"] += not value["predicted_hate"]
    ct["positive_gate_empty_type"] += (
        value["predicted_hate"] and value["predicted_cardinality"] == 0
    )
    ct["gate_negative_exact"] += (
        not value["predicted_hate"] and value["category"] == "exact"
    )
    ct["exact_given_positive_count"] += (
        value["predicted_hate"] and value["category"] == "exact"
    )
    b["gold_cardinality"][str(value["gold_cardinality"])] += 1
    b["predicted_cardinality"][str(value["predicted_cardinality"])] += 1
    gs = b["gold_sets"][value["gold_set"]]
    gs["n"] += 1
    gs[value["category"]] += 1
    gs["gate_negative"] += not value["predicted_hate"]
    gold = set(value["gold_types"])
    pred = set(value["predicted_types"])
    for l, c in b["per_label"].items():
        c["support"] += l in gold
        c["predicted"] += l in pred
        c["tp"] += l in gold and l in pred
        c["fp"] += l not in gold and l in pred
        c["fn"] += l in gold and l not in pred
        c["fn_gate_negative"] += (
            l in gold and l not in pred and not value["predicted_hate"]
        )
        c["fn_gate_positive"] += l in gold and l not in pred and value["predicted_hate"]


def finish(b):
    c = b["counts"]
    n = c["known_type_outputs"]
    a.need(
        sum(
            c[k]
            for k in ("exact", "only_extra", "only_missing", "both_extra_and_missing")
        )
        == n,
        "Nonexclusive type categories",
    )
    a.need(
        c["exact"] + c["only_extra"] == c["contains_gold"],
        "Containment does not equal exact plus only-extra",
    )
    a.need(
        sum(v["fp"] for v in b["per_label"].values()) == c["extra_labels"]
        and sum(v["fn"] for v in b["per_label"].values()) == c["missing_labels"],
        "Per-label/set error counts differ",
    )
    return {
        "counts": dict(c),
        "gold_cardinality": dict(b["gold_cardinality"]),
        "predicted_cardinality": dict(b["predicted_cardinality"]),
        "gold_sets": {k: dict(v) for k, v in b["gold_sets"].items()},
        "per_label": {k: dict(v) for k, v in b["per_label"].items()},
        "rates": {
            k: c[k] / n if n else None
            for k in (
                "exact",
                "only_extra",
                "only_missing",
                "both_extra_and_missing",
                "contains_gold",
                "extra_labels",
                "missing_labels",
                "predicted_labels",
                "gate_negative",
            )
        },
        "exact_given_positive": (
            c["exact_given_positive_count"] / c["gate_positive"]
            if c["gate_positive"]
            else None
        ),
        "interpretation": "Descriptive actual output counts on known type gold. Conditional positive-gate subsets are selected; not independent or causal.",
    }


def verify_against_frozen(b, run):
    c = b["counts"]
    r = run["counts"]
    fields = {
        "known_type_outputs": "type_valid",
        "exact": "type_exact",
        "contains_gold": "type_contains_gold",
        "extra_labels": "type_extra_labels",
        "missing_labels": "type_missing_labels",
        "predicted_labels": "type_predicted_labels",
        "gold_labels": "type_gold_labels",
    }
    a.need(
        all(c.get(k, 0) == r.get(v, 0) for k, v in fields.items()),
        "Set decomposition differs from frozen metric counts",
    )
    for l, values in b["per_label"].items():
        a.need(
            all(
                values[k] == run["per_label"][l][k]
                for k in ("support", "predicted", "tp", "fp", "fn")
            ),
            "Type diagnosis label counts differ",
        )


def conditional_pairs(rows, predictions, entry, common_ids):
    common = set(common_ids)
    spec = entry["spec"]
    arms = [spec["treatment_conditions"], spec["control_conditions"]]
    pairs = [
        (t, arms[1][0] if len(arms[1]) == 1 else arms[1][i])
        for i, t in enumerate(arms[0])
    ]
    groups = {}

    def get(name):
        if name not in groups:
            groups[name] = {
                "counts": Counter(),
                "treatment": bucket(),
                "control": bucket(),
                "keys": set(),
            }
        return groups[name]

    for row in rows:
        if row["key"] not in common:
            continue
        for t, c in pairs:
            for rep in range(3):
                tp, cp = [predictions[row["key"], v, rep] for v in (t, c)]
                tv, cv = outcome(row, tp), outcome(row, cp)
                a.need(
                    tv["status"] == cv["status"] == "valid",
                    "Conditional pair unexpectedly missing",
                )
                both = tp["result"]["hate"] and cp["result"]["hate"]
                same = tp["messages_sha256"] == cp["messages_sha256"]
                for name in (
                    "all",
                    "both_gates_positive" if both else "one_or_both_gates_negative",
                    "identical_messages" if same else "changed_messages",
                ):
                    g = get(name)
                    ct = g["counts"]
                    g["keys"].add(row["key"])
                    ct["pairs"] += 1
                    te = tv["category"] == "exact"
                    ce = cv["category"] == "exact"
                    ct["exact_fix"] += te and not ce
                    ct["exact_regression"] += ce and not te
                    ct["both_exact"] += te and ce
                    ct["neither_exact"] += not te and not ce
                    ts, cs = set(tv["predicted_types"]), set(cv["predicted_types"])
                    gold = set(tv["gold_types"])
                    ct["correct_labels_added"] += len((ts - cs) & gold)
                    ct["correct_labels_lost"] += len((cs - ts) & gold)
                    ct["false_labels_added"] += len((ts - cs) - gold)
                    ct["false_labels_removed"] += len((cs - ts) - gold)
                    update(g["treatment"], tv)
                    update(g["control"], cv)
    final = {}
    for name, g in groups.items():
        t, c = finish(g["treatment"]), finish(g["control"])
        ct = g["counts"]
        a.need(
            ct["pairs"]
            == sum(
                ct[k]
                for k in (
                    "exact_fix",
                    "exact_regression",
                    "both_exact",
                    "neither_exact",
                )
            ),
            "Incomplete set transitions",
        )
        final[name] = {
            "counts": dict(ct),
            "unique_texts": len(g["keys"]),
            "treatment": t,
            "control": c,
            "exact_difference": (ct["exact_fix"] - ct["exact_regression"])
            / ct["pairs"],
            "unit": "Dependent matched condition/pool/repeat pairs; a shared reference is reused across pool pairs here, not extra real reference outputs.",
        }
    return final


def main():
    done = a.io.read(a.OUT / "statistics_complete.json")
    a.io.verify(a.io.ROOT, done["artifacts"])
    a.io.verify(a.io.ROOT, done["bindings"])
    a.need(not (a.OUT / "diagnostics_complete.json").exists(), "Preserve diagnostics")
    roster, science, predictions, assembly = a.load()
    bykey = {r["key"]: r for r in roster}
    bindings = {
        a.io.rel(p): a.io.sha(p)
        for p in [
            Path(__file__),
            a.OUT / "statistics_complete.json",
            a.OUT / "all_common_mask_quality_runs.json",
            a.OUT / "paired_quality.json",
        ]
    }
    a.write(
        a.OUT / "diagnostics_started.json",
        {"created_utc": a.io.utc(), "bindings": bindings, "new_model_calls": 0},
    )
    ledger = a.OUT / "quality_outcomes.jsonl"
    count = 0
    negative_types = 0
    with ledger.open("x", encoding="utf-8", newline="\n") as f:
        for (key, condition, rep), p in predictions.items():
            value = {
                **outcome(bykey[key], p),
                "condition": condition,
                "repeat": rep,
                "stage": p["stage"],
                "messages_sha256": p["messages_sha256"],
            }
            f.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
            negative_types += value.get("negative_gate_with_types", False)
    common = a.io.read(a.OUT / "all_common_mask_quality_runs.json")
    summaries = {}
    verified = 0
    for name, data in common.items():
        rows = [bykey[k] for k in data["common_ids"]]
        groups = defaultdict(bucket)
        for run in data["runs"]:
            condition, rep = run["condition"], run["repeat"]
            b = bucket()
            group = condition.rsplit("_s", 1)[0] if "__" in condition else condition
            for row in rows:
                value = outcome(row, predictions[row["key"], condition, rep])
                update(b, value)
                update(groups[group], value)
            verify_against_frozen(b, run)
            verified += 1
        summaries[name] = {
            "common_n": len(rows),
            "known_type_n": data["known_type_n"],
            "groups": {k: finish(v) for k, v in groups.items()},
        }
        print("Type-set decomposition checked: " + name, flush=True)
    a.write(a.OUT / "common_mask_type_diagnostics.json", summaries)
    paired = a.io.read(a.OUT / "paired_quality.json")
    pairs = {}
    for entry in a.metric_specs(science):
        if not entry["fine"]:
            continue
        result = paired[entry["name"]]
        value = conditional_pairs(roster, predictions, entry, result["common_ids"])
        pairs[entry["name"]] = value
        a.need(
            abs(
                value["all"]["exact_difference"]
                - result["metrics"]["type_exact_set"]["difference"]
            )
            < 1e-12,
            "Matched set-change total differs from frozen exact-set point",
        )
    a.write(a.OUT / "paired_type_diagnostics.json", pairs)
    a.io.verify(a.io.ROOT, bindings)
    files = [
        ledger,
        *[
            a.OUT / n
            for n in (
                "diagnostics_started.json",
                "common_mask_type_diagnostics.json",
                "paired_type_diagnostics.json",
            )
        ],
    ]
    a.write(
        a.OUT / "diagnostics_complete.json",
        {
            "status": "part5_quality_diagnostics_checked_interpretation_pending",
            "created_utc": a.io.utc(),
            "bindings": bindings,
            "artifacts": {a.io.rel(p): a.io.sha(p) for p in files},
            "quality_ledger_rows": count,
            "negative_gate_nonempty_types": negative_types,
            "run_decompositions_checked": verified,
            "paired_type_comparisons": len(pairs),
            "new_model_calls": 0,
        },
    )
    print(
        "All quality outcomes and type-set/gate diagnostics retained and reconciled.",
        flush=True,
    )


if __name__ == "__main__":
    with a.io.offline():
        main()

from collections import Counter
from fractions import Fraction
from pathlib import Path
import numpy as np
from analysis.feedback.transfer import validate_quality as v

a = v.a


def mean_scores(scores):
    means = [
        [sum(scores[c][t][j] for t in range(3)) / 3 for j in range(3)] for c in range(4)
    ]
    trt = [sum(means[c][j] for c in (1, 2, 3)) / 3 for j in range(3)]
    return means + [trt, [trt[j] - means[0][j] for j in range(3)]]


def main():
    receipt = a.read(v.OUT / "independent_numerical_check.json")
    v.check(
        receipt["status"]
        == "part4_quality_numerics_pass_diagnostic_verification_pending",
        "Numerical verification required",
    )
    contract = a.read(v.OUT / "analysis_contract.json")
    schema = a.read(v.OUT / "feature_schema.json")
    meta, data, codes = v.rebuild(contract, schema)
    results = a.read(v.OUT / "quality_results.json")
    per = a.read(v.OUT / "per_label_quality.json")
    shapes = a.read(v.OUT / "set_and_severity_diagnostics.json")
    pairs = a.read(v.OUT / "conditional_gate_pairs.json")
    defined = {
        name
        for name, r in results.items()
        if r["head"] in ("hate_type", "target_group", "severity") and r["common_n"]
    }
    v.check(
        set(per) == set(shapes) == defined and len(defined) == 76,
        "All fine-head diagnostics present",
    )
    expected_pairs = {
        name
        for name in defined
        if results[name]["view"] == "main"
        and results[name]["group_kind"] in ("global", "source")
    }
    v.check(
        set(pairs) == expected_pairs and len(pairs) == 8, "All gate-pair views present"
    )
    label_cells = transition_pairs = 0
    for name in sorted(defined):
        r = results[name]
        h = r["head"]
        labels = contract["labels"][h]
        ids = r["common_indices"]
        n = len(ids)
        gold = [
            {meta[i][v.FIELDS[h]]} if h == "severity" else set(meta[i][v.FIELDS[h]])
            for i in ids
        ]
        pred = [
            [
                [
                    (
                        (
                            {labels[int(codes[h][i, c, t])]}
                            if codes[h][i, c, t] >= 0
                            else set()
                        )
                        if h == "severity"
                        else {
                            label
                            for j, label in enumerate(labels)
                            if codes[h][i, c, t] & (1 << j)
                        }
                    )
                    for t in range(3)
                ]
                for c in range(4)
            ]
            for i in ids
        ]
        v.check(set(per[name]) == set(labels), "All taxonomy labels retained")
        members = v.groups(name, meta, codes)
        for label in labels:
            counts = np.zeros((4, 3, 3), dtype=np.int64)
            for g, p in zip(gold, pred):
                for c in range(4):
                    for t in range(3):
                        hit = label in p[c][t]
                        true = label in g
                        counts[c, t, 0] += hit and true
                        counts[c, t, 1] += hit and not true
                        counts[c, t, 2] += true and not hit
            divide = lambda x, y: Fraction(int(x), int(y)) if y else Fraction(0)
            scores = [
                [
                    [
                        divide(tp, tp + fp),
                        divide(tp, tp + fn),
                        divide(2 * tp, 2 * tp + fp + fn),
                    ]
                    for tp, fp, fn in counts[c]
                ]
                for c in range(4)
            ]
            full_support = sum(
                (
                    meta[i][v.FIELDS[h]] == label
                    if h == "severity"
                    else label in meta[i][v.FIELDS[h]]
                )
                for i in members
                if v.known(meta[i], h)
            )
            expected = dict(
                in_population_macro=label
                in contract["population_macro_labels"][r["population"]][h],
                full_group_positive_support=full_support,
                common_positive_support=sum(label in g for g in gold),
                counts_tp_fp_fn=counts,
                run_precision_recall_f1=scores,
                means_precision_recall_f1=mean_scores(scores),
            )
            v.near(per[name][label], expected, "Raw-set per-label precision/recall/F1")
            label_cells += 12
        shape = shapes[name]
        if h != "severity":
            goldhist = [sum(len(g) == j for g in gold) for j in range(len(labels) + 1)]
            phist = [
                [
                    [
                        sum(len(p[c][t]) == j for p in pred)
                        for j in range(len(labels) + 1)
                    ]
                    for t in range(3)
                ]
                for c in range(4)
            ]
            categories = {
                key: np.zeros((4, 3), dtype=np.int64)
                for key in (
                    "exact",
                    "only_extra",
                    "only_missing",
                    "both_extra_missing",
                    "contains_gold",
                )
            }
            for g, p in zip(gold, pred):
                for c in range(4):
                    for t in range(3):
                        extra = bool(p[c][t] - g)
                        missing = bool(g - p[c][t])
                        for key, yes in dict(
                            exact=g == p[c][t],
                            only_extra=extra and not missing,
                            only_missing=missing and not extra,
                            both_extra_missing=extra and missing,
                            contains_gold=not missing,
                        ).items():
                            categories[key][c, t] += yes
            gold_sets = dict(
                Counter(str(sum(1 << labels.index(l) for l in g)) for g in gold)
            )
            expected = dict(
                gold_cardinality=goldhist,
                predicted_cardinality=phist,
                gold_sets=gold_sets,
                categories=categories,
                mean_gold_labels=Fraction(sum(len(g) for g in gold), n),
                known_empty_gold=sum(not g for g in gold),
            )
            v.near(
                {key: shape[key] for key in expected},
                expected,
                "Raw-set cardinality and partitions",
            )
            v.check(
                all(
                    sum(
                        categories[k][c, t]
                        for k in (
                            "exact",
                            "only_extra",
                            "only_missing",
                            "both_extra_missing",
                        )
                    )
                    == n
                    for c in range(4)
                    for t in range(3)
                ),
                "Disjoint set-quality categories cover all texts",
            )
        else:
            confusion = np.zeros((4, 3, 3, 4), dtype=np.int64)
            no_category = np.zeros((4, 3), dtype=np.int64)
            for local, i in enumerate(ids):
                g = labels.index(meta[i]["gold_severity"])
                for c in range(4):
                    for t in range(3):
                        actual = pred[local][c][t]
                        j = 0 if not actual else labels.index(next(iter(actual))) + 1
                        confusion[c, t, g, j] += 1
                        no_category[c, t] += not actual and codes["gates"][i, c, t] == 1
            v.near(shape["confusion"], confusion, "Original severity confusion")
            v.near(
                shape["no_category_despite_positive_gate"],
                no_category,
                "Missing class despite positive gate",
            )
            v.check(
                shape["gold_order"] == labels
                and shape["prediction_order"] == ["none", *labels],
                "Severity order",
            )
        if name not in pairs:
            continue
        groups = {
            part: dict(
                pairs=0,
                keys=set(),
                by_pool={str(c): 0 for c in (1, 2, 3)},
                exact_fix=0,
                exact_regression=0,
                both_exact=0,
                neither_exact=0,
                gated_exact_fix=0,
                gated_exact_regression=0,
                correct_labels_added=0,
                correct_labels_removed=0,
                extra_labels_added=0,
                extra_labels_removed=0,
            )
            for part in (
                "all",
                "both_positive",
                "gained_positive_gate",
                "lost_positive_gate",
                "both_negative",
            )
        }
        for local, i in enumerate(ids):
            g = gold[local]
            for c in (1, 2, 3):
                for t in range(3):
                    rset = pred[local][0][t]
                    tset = pred[local][c][t]
                    rg = codes["gates"][i, 0, t] == 1
                    tg = codes["gates"][i, c, t] == 1
                    part = {
                        (True, True): "both_positive",
                        (False, True): "gained_positive_gate",
                        (True, False): "lost_positive_gate",
                        (False, False): "both_negative",
                    }[bool(rg), bool(tg)]
                    re = rset == g
                    te = tset == g
                    added = tset - rset
                    removed = rset - tset
                    for key in ("all", part):
                        group = groups[key]
                        group["pairs"] += 1
                        group["keys"].add(meta[i]["key"])
                        group["by_pool"][str(c)] += 1
                        events = dict(
                            exact_fix=not re and te,
                            exact_regression=re and not te,
                            both_exact=re and te,
                            neither_exact=not re and not te,
                            gated_exact_fix=not (re and rg) and te and tg,
                            gated_exact_regression=re and rg and not (te and tg),
                            correct_labels_added=len(added & g),
                            correct_labels_removed=len(removed & g),
                            extra_labels_added=len(added - g),
                            extra_labels_removed=len(removed - g),
                        )
                        for event, value in events.items():
                            group[event] += int(value)
        for part, group in groups.items():
            group["unique_texts"] = len(group.pop("keys"))
            group["exact_difference"] = (
                Fraction(group["exact_fix"] - group["exact_regression"], group["pairs"])
                if group["pairs"]
                else None
            )
        v.near(
            pairs[name]["partitions"], groups, "Every original gate/label transition"
        )
        for key in (
            "pairs",
            "exact_fix",
            "exact_regression",
            "gated_exact_fix",
            "gated_exact_regression",
            "correct_labels_added",
            "correct_labels_removed",
            "extra_labels_added",
            "extra_labels_removed",
        ):
            v.check(
                groups["all"][key]
                == sum(groups[part][key] for part in groups if part != "all"),
                "Gate partitions reconcile",
            )
        transition_pairs += groups["all"]["pairs"]
    source_paths = [
        v.OUT / name
        for name in (
            "quality_results.json",
            "per_label_quality.json",
            "set_and_severity_diagnostics.json",
            "conditional_gate_pairs.json",
            "independent_numerical_check.json",
        )
    ]
    receipt = dict(
        status="part4_numerics_and_diagnostics_verified_interpretation_pending",
        checked_utc=a.utc(),
        label_and_shape_views=len(defined),
        per_label_run_cells=label_cells,
        conditional_pair_views=len(pairs),
        dependent_transition_pairs=transition_pairs,
        all_actual_prediction_sets_checked=True,
        known_empty_gold_retained=True,
        no_category_not_imputed_to_low=True,
        sources={a.rel(path): a.sha(path) for path in source_paths},
        verifier_sha256=a.sha(Path(__file__)),
        new_Qwen_calls=0,
        new_fits=0,
    )
    with (v.OUT / "verified.json").open("xb") as f:
        f.write(a.encoded(receipt, pretty=True))
    print(
        "Independent label, set-shape, severity and gate-transition checks passed.",
        flush=True,
    )


if __name__ == "__main__":
    with a.offline():
        main()

"""German guidelines verify legal."""

from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from sklearn.metrics import f1_score, multilabel_confusion_matrix
from analysis.knowledge.german_guidelines import legal as l
from analysis.common import scoring as original


def same(actual, expected):
    if isinstance(expected, np.ndarray):
        same(actual, expected.tolist())
    elif isinstance(expected, dict):
        for key, value in expected.items():
            same(actual[key], value)
    elif isinstance(expected, (list, tuple)):
        l.require(len(actual) == len(expected), "Independent sequence length differs")
        for a, b in zip(actual, expected):
            same(a, b)
    elif expected is None or (
        isinstance(expected, (float, np.floating)) and (not np.isfinite(expected))
    ):
        l.require(actual is None, "Undefined ratio converted to value")
    else:
        l.require(
            np.allclose(actual, expected, rtol=0, atol=3e-12), "Independent numeric value differs"
        )


def feature_for_sets(gold, pred):
    extra, missed = (len(pred - gold), len(gold - pred))
    return [
        1,
        len(gold),
        len(pred),
        len(gold & pred),
        extra,
        missed,
        int(gold == pred),
        int(extra > 0),
        int(missed > 0),
        int(extra > 0 and missed == 0),
        int(extra == 0 and missed > 0),
        int(extra > 0 and missed > 0),
        int(not gold),
        int(bool(gold)),
        int(not pred),
        int(not gold and (not pred)),
        int(not gold and bool(pred)),
        int(bool(gold) and gold == pred),
        int(bool(gold) and (not pred)),
        *(int(len(pred) == k) for k in range(4)),
    ]


def reference_counts(y, p):
    z = np.zeros((3, 4), dtype=np.int64)
    features = np.zeros(len(l.FEATURES), dtype=np.int64)
    for yy, pp in zip(y, p):
        gold = {i for i, value in enumerate(yy) if value}
        pred = {i for i, value in enumerate(pp) if value}
        features += feature_for_sets(gold, pred)
        for i in range(3):
            z[i, 2 * int(i in gold) + int(i in pred)] += 1
    return (z, features)


def reference_metrics(z, f, idx):

    def ratio(a, b, default=np.nan):
        a, b = np.broadcast_arrays(np.asarray(a, float), np.asarray(b, float))
        out = np.full(a.shape, default)
        np.divide(a, b, out=out, where=b != 0)
        return out

    a = {name: np.asarray(f)[..., i] for i, name in enumerate(l.FEATURES)}
    tn, fp, fn, tp = [np.asarray(z)[..., i] for i in range(4)]
    per = dict(
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        support=tp + fn,
        predicted=tp + fp,
        f1=ratio(2 * tp, 2 * tp + fp + fn, 0.0),
        precision=ratio(tp, tp + fp),
        recall=ratio(tp, tp + fn),
        fpr=ratio(fp, fp + tn),
        fnr=ratio(fn, fn + tp),
    )
    hit, over, under = (v.sum(axis=-1) for v in (tp, fp, fn))
    n = a["n"]
    a.update(
        macro_f1=np.where(n > 0, per["f1"][..., idx].mean(axis=-1), np.nan),
        micro_f1=ratio(2 * hit, 2 * hit + over + under, 0.0),
        micro_precision=ratio(hit, hit + over),
        micro_recall=ratio(hit, hit + under),
        exact_set_accuracy=ratio(a["exact_correct"], n),
        exact_wrong=n - a["exact_correct"],
        mean_predicted_labels=ratio(a["predicted_labels"], n),
        mean_gold_labels=ratio(a["gold_labels"], n),
        label_errors=over + under,
        hamming_accuracy=1 - ratio(over + under, 3 * n),
        empty_gold_false_positive_rate=ratio(a["empty_gold_false_labels"], a["gold_empty"]),
        nonempty_gold_exact_accuracy=ratio(a["nonempty_gold_exact"], a["gold_nonempty"]),
        nonempty_gold_empty_rate=ratio(a["nonempty_gold_pred_empty"], a["gold_nonempty"]),
        empty_prediction_rate=ratio(a["pred_empty"], n),
    )
    return (a, per)


def reference(y, p, idx):
    return reference_metrics(*reference_counts(y, p), idx)


def check_summary(saved, values):
    for i, value in enumerate(values):
        same(saved["repeats"][i], value)
    for name in values[0]:
        v = np.asarray([row[name] for row in values])
        same(saved["mean"][name], v.mean(axis=0))
        same(saved["sample_sd"][name], v.std(axis=0, ddof=1))
        same(saved["range"][name], [v.min(axis=0), v.max(axis=0)])


def check_interval(saved, values, family=None):
    invalid = int((~np.isfinite(values)).sum())
    same(saved["undefined_draws"], invalid)
    same(saved["ci95"], None if invalid else np.percentile(values, [2.5, 97.5]).tolist())
    same(saved["family_size"], family)
    same(saved["family_confidence"], 0.9875 if family else None)
    same(
        saved["family_interval"],
        np.percentile(values, [0.625, 99.375]).tolist() if family and (not invalid) else None,
    )


def verify(path):
    with l.offline():
        c = l.c
        c.validate()
        contract = c.read(l.BASE / "analysis_contract.json")
        c.verify(contract["inputs"])
        manifest = c.read(path)
        l.require(
            manifest["status"] == "part3_scored_pending_independent_check", "Incomplete analysis"
        )
        c.verify(manifest["sources"])
        c.verify(manifest["outputs"])
        out = c.ROOT / manifest["run"]
        _, cache = l.reader.read_verified()
        endpoints = c.read(cache / "endpoint_setup.json")
        roster = {row["key"]: row for row in c.rows(c.BASE / "roster.jsonl")}
        outputs = defaultdict(dict)
        for row in c.rows(cache / "predictions.jsonl"):
            if row["phase"] in l.PHASES:
                outputs[row["condition"], row["repeat"]][row["key"]] = row["result"]
        checks = Counter()
        for phase in l.PHASES:
            ep = endpoints[phase]
            keys, labels = (ep["keys"], ep["all_labels"])
            idx = [labels.index(x) for x in ep["macro_labels"]]
            result = c.read(out / phase / "results.json")
            y = np.asarray(
                [[int(label in roster[key]["legal"]) for label in labels] for key in keys],
                dtype=np.int8,
            )
            p = np.asarray(
                [
                    [
                        [
                            (
                                [-1] * 3
                                if outputs[cond, r][key] is None
                                else [
                                    int(label in outputs[cond, r][key]["legal"]) for label in labels
                                ]
                            )
                            for key in keys
                        ]
                        for r in range(3)
                    ]
                    for cond in l.CONDITIONS
                ],
                dtype=np.int8,
            )
            gates = np.asarray(
                [
                    [
                        [
                            (
                                -1
                                if outputs[cond, r][key] is None
                                else int(outputs[cond, r][key]["hate"])
                            )
                            for key in keys
                        ]
                        for r in range(3)
                    ]
                    for cond in l.CONDITIONS
                ],
                dtype=np.int8,
            )
            common = (p >= 0).all(axis=(0, 1, 3))
            yy, pp = (y[common], p[:, :, common])
            l.require(
                result["common_keys"]
                == [key for key, on in zip(keys, common) if on]
                == ep["common_valid_keys"],
                "Legal mask differs",
            )
            same(result["common_inputs"], len(yy))
            same(result["common_gold_support"], dict(zip(labels, yy.sum(axis=0))))
            same(result["common_observed_empty_gold"], int((yy.sum(axis=1) == 0).sum()))
            with np.load(out / phase / "arrays.npz", allow_pickle=False) as handle:
                stored = {k: handle[k] for k in handle.files}
            for name, value in [
                ("gold", y),
                ("predictions", p),
                ("predicted_hate", gates),
                ("common_mask", common),
            ]:
                l.require(
                    np.array_equal(stored[name], value), "Legal cached array differs: " + name
                )
            for ci, cond in enumerate(l.CONDITIONS):
                values, label_values, own, own_labels = ([], [], [], [])
                matrices = []
                for r in range(3):
                    z, f = reference_counts(yy, pp[ci, r])
                    l.require(
                        np.array_equal(z, stored["confusion_counts"][ci, r])
                        and np.array_equal(f, stored["set_features"][ci, r]),
                        "Point totals differ",
                    )
                    l.require(
                        np.array_equal(z, multilabel_confusion_matrix(yy, pp[ci, r]).reshape(3, 4)),
                        "Scikit-learn confusion differs",
                    )
                    agg, per = reference_metrics(z, f, idx)
                    same(
                        float(agg["macro_f1"]),
                        f1_score(yy, pp[ci, r], labels=idx, average="macro", zero_division=0),
                    )
                    same(
                        float(agg["micro_f1"]),
                        f1_score(yy, pp[ci, r], average="micro", zero_division=0),
                    )
                    values.append(agg)
                    label_values.append(per)
                    v = (p[ci, r] >= 0).all(axis=1)
                    aa, ll = reference(y[v], p[ci, r, v], idx)
                    own.append(aa)
                    own_labels.append(ll)
                    rows = [
                        {"id": key, "gate": roster[key]["gate"], "legal": roster[key]["legal"]}
                        for key in keys
                    ]
                    outs = {
                        key: {
                            "result": outputs[cond, r][key],
                            "n_valid": int(outputs[cond, r][key] is not None),
                        }
                        for key in keys
                    }
                    endpoint = dict(
                        all_labels=labels,
                        macro_labels=ep["macro_labels"],
                        dimension="legal",
                        scorable_ids=keys,
                    )
                    original_value = original.score(original.contributions(rows, outs, endpoint))
                    same(aa["macro_f1"], original_value["macro_f1_valid_only"])
                    same(
                        result["full_roster_bounds"][cond]["bounds"]["macro_f1"]["by_repeat"][r],
                        original_value["invalid_inclusive_bounds"],
                    )
                    checks["original_contract_scores_and_bounds"] += 1
                    missing = np.flatnonzero(~v)
                    candidates = []
                    for assignments in product(range(8), repeat=len(missing)):
                        row = p[ci, r].copy()
                        for k, code in zip(missing, assignments):
                            row[k] = [code >> j & 1 for j in range(3)]
                        candidates.append(reference(y, row, idx)[0])
                        checks["exact_missing_set_scenarios"] += 1
                    for name in l.BOUND_METRICS:
                        same(
                            result["full_roster_bounds"][cond]["bounds"][name]["by_repeat"][r],
                            [
                                min((v[name] for v in candidates)),
                                max((v[name] for v in candidates)),
                            ],
                        )
                    actual_gate = result["predicted_gate_diagnostic"][cond]["by_repeat"][r]
                    g = gates[ci, r, common]
                    same(
                        actual_gate,
                        {
                            "predicted_nonhate": int((g == 0).sum()),
                            "nonempty_legal_despite_predicted_nonhate": int(
                                ((g == 0) & (pp[ci, r].sum(axis=-1) > 0)).sum()
                            ),
                        },
                    )
                    matrix = np.zeros((8, 8), int)
                    for gt, pr in zip(yy, pp[ci, r]):
                        matrix[
                            sum((int(x) << j for j, x in enumerate(gt))),
                            sum((int(x) << j for j, x in enumerate(pr))),
                        ] += 1
                    l.require(
                        np.array_equal(matrix, result["set_confusions"][cond]["repeats"][r]),
                        "Set confusion differs",
                    )
                    matrices.append(matrix)
                    checks["point_repeats_and_full_sets"] += 1
                same(result["set_confusions"][cond]["mean"], np.mean(matrices, axis=0).tolist())
                for saved, expected, per in [
                    (result["common"][cond], values, label_values),
                    (result["own_valid"][cond], own, own_labels),
                ]:
                    check_summary(saved, expected)
                    for j, label in enumerate(labels):
                        check_summary(
                            saved["per_label"][label],
                            [{k: v[j] for k, v in row.items()} for row in per],
                        )
                for name in l.BOUND_METRICS:
                    b = result["full_roster_bounds"][cond]["bounds"][name]
                    same(b["mean"], np.asarray(b["by_repeat"]).mean(axis=0).tolist())
            design = c.read(out / phase / "bootstrap_design.json")
            clusters = [roster[key]["text_cluster"] for key, on in zip(keys, common) if on]
            l.require(
                len(clusters) == len(set(clusters)) and design["clusters"] == sorted(clusters),
                "Unexpected cluster structure",
            )
            l.require(
                design["seed"] == l.SEEDS[phase] and design["draws"] == 10000,
                "Wrong bootstrap settings",
            )
            map_rows = np.asarray([clusters.index(x) for x in design["clusters"]])
            expected_strata = defaultdict(list)
            for i, rowidx in enumerate(map_rows):
                expected_strata[roster[result["common_keys"][rowidx]]["source"],].append(i)
            strata = [
                {"sources": list(k), "cluster_indices": v}
                for k, v in sorted(expected_strata.items())
            ]
            l.require(design["strata"] == strata, "Source strata changed")
            flat = pp.reshape(9, len(yy), 3)
            category_codes = 2 * yy + flat
            feature_table = np.asarray(
                [
                    feature_for_sets(
                        {j for j in range(3) if a & 1 << j}, {j for j in range(3) if b & 1 << j}
                    )
                    for a in range(8)
                    for b in range(8)
                ]
            )
            setcodes = yy @ [1, 2, 4] * 8 + flat @ [1, 2, 4]
            independently_derived_features = feature_table[setcodes]
            rng, digest = (np.random.default_rng(l.SEEDS[phase]), hashlib.sha256())
            for start in range(0, 10000, 64):
                size = min(64, 10000 - start)
                picks = np.concatenate(
                    [
                        rng.choice(
                            s["cluster_indices"], (size, len(s["cluster_indices"])), replace=True
                        )
                        for s in strata
                    ],
                    axis=1,
                )
                digest.update(picks.astype("<i4").tobytes())
                selected = map_rows[picks]
                direct = (
                    np.stack(
                        [(category_codes[:, selected] == k).sum(axis=2) for k in range(4)], axis=-1
                    )
                    .transpose(1, 0, 2, 3)
                    .reshape(size, 3, 3, 3, 4)
                )
                feat = (
                    independently_derived_features[:, selected]
                    .sum(axis=2)
                    .transpose(1, 0, 2)
                    .reshape(size, 3, 3, len(l.FEATURES))
                )
                l.require(
                    np.array_equal(direct, stored["bootstrap_confusions"][start : start + size]),
                    "Independent bootstrap categories differ",
                )
                l.require(
                    np.array_equal(feat, stored["bootstrap_features"][start : start + size]),
                    "Independent bootstrap sets differ",
                )
                checks["independent_bootstrap_draws"] += size
            l.require(digest.hexdigest() == design["pick_stream_sha256"], "Pick stream differs")
            bm, bl = reference_metrics(
                stored["bootstrap_confusions"], stored["bootstrap_features"], idx
            )
            bm, bl = ({k: v.mean(axis=2) for k, v in group.items()} for group in (bm, bl))
            for ci, cond in enumerate(l.CONDITIONS):
                for name in l.BOOT_METRICS:
                    check_interval(
                        result["condition_descriptive_intervals"][cond][name], bm[name][:, ci]
                    )
                    checks["aggregate_intervals"] += 1
                for j, label in enumerate(labels):
                    for name in l.LABEL_METRICS:
                        check_interval(
                            result["per_label_descriptive_intervals"][cond][label][name],
                            bl[name][:, ci, j],
                        )
                        checks["per_label_intervals"] += 1
            for contrast in result["contrasts"]:
                li, ri = (
                    l.CONDITIONS.index(contrast["left"]),
                    l.CONDITIONS.index(contrast["right"]),
                )
                for name in l.BOOT_METRICS:
                    family = (
                        4 if phase == "de_legal_test" and ri == 0 and (name == "macro_f1") else None
                    )
                    check_interval(
                        contrast["metrics"][name], bm[name][:, li] - bm[name][:, ri], family
                    )
                    same(
                        contrast["metrics"][name]["difference"],
                        result["common"][l.CONDITIONS[li]]["mean"][name]
                        - result["common"][l.CONDITIONS[ri]]["mean"][name],
                    )
                    same(
                        contrast["metrics"][name]["by_repeat"],
                        [
                            result["common"][l.CONDITIONS[li]]["repeats"][r][name]
                            - result["common"][l.CONDITIONS[ri]]["repeats"][r][name]
                            for r in range(3)
                        ],
                    )
                    checks["aggregate_intervals"] += 1
                for j, label in enumerate(labels):
                    flow = np.zeros((3, 3, 8), int)
                    for r in range(3):
                        for s in range(3):
                            for truth, a, b in zip(yy[:, j], pp[li, r, :, j], pp[ri, s, :, j]):
                                if truth == 0:
                                    event = (
                                        0
                                        if b == 1 and a == 0
                                        else 1 if b == 0 and a == 1 else 4 if b == a == 1 else 6
                                    )
                                else:
                                    event = (
                                        2
                                        if b == 0 and a == 1
                                        else 3 if b == 1 and a == 0 else 5 if b == a == 0 else 7
                                    )
                                flow[r, s, event] += 1
                    savedflow = contrast["transitions"]["per_label"][label]
                    l.require(
                        np.array_equal(savedflow["all_nine_pair_counts"], flow),
                        "Per-label error flow differs",
                    )
                    same(
                        savedflow["mean_events"],
                        {name: flow[:, :, i].mean() for i, name in enumerate(l.h.EVENTS)},
                    )
                    for name in l.LABEL_METRICS:
                        check_interval(
                            contrast["per_label"][label][name],
                            bl[name][:, li, j] - bl[name][:, ri, j],
                        )
                        a, b = (
                            result["common"][l.CONDITIONS[k]]["per_label"][label] for k in (li, ri)
                        )
                        same(
                            contrast["per_label"][label][name]["difference"],
                            a["mean"][name] - b["mean"][name],
                        )
                        checks["per_label_intervals"] += 1
                exact = np.zeros((3, 3, 4), int)
                for r in range(3):
                    for s in range(3):
                        for gold, a, b in zip(yy, pp[li, r], pp[ri, s]):
                            left, right = (np.array_equal(a, gold), np.array_equal(b, gold))
                            exact[
                                r,
                                s,
                                (
                                    0
                                    if left and (not right)
                                    else 1 if right and (not left) else 2 if left and right else 3
                                ),
                            ] += 1
                l.require(
                    np.array_equal(contrast["transitions"]["exact_all_nine_pair_counts"], exact),
                    "Exact-set transition differs",
                )
                same(
                    contrast["transitions"]["mean_exact_events"],
                    {
                        name: exact[:, :, j].mean()
                        for j, name in enumerate(contrast["transitions"]["exact_event_order"])
                    },
                )
                for name in l.BOUND_METRICS:
                    a, b = [
                        result["full_roster_bounds"][l.CONDITIONS[k]]["bounds"][name]["mean"]
                        for k in (li, ri)
                    ]
                    same(
                        contrast["full_roster_failure_difference_bounds"][name],
                        [a[0] - b[1], a[1] - b[0]],
                    )
                checks["all_pair_error_flow_checks"] += 1
            print(
                phase
                + ": original contract, exact sets, all label scores and 10,000 paired draws verified.",
                flush=True,
            )
        c.verify(manifest["outputs"])
        c.verify(contract["inputs"])
        c.verify(c.read(l.reader.BASE / "analysis_contract.json")["source_bindings"])
        report = {
            "status": "independent_legal_analysis_pass",
            "checks": dict(checks),
            "actual_Qwen_calls": 0,
            "D_writes": 0,
            "latex_edits": 0,
        }
        l.save(out / "independent_check.json", report)
        l.save(
            l.BASE / "complete.json",
            {
                "status": "part3_numerically_verified",
                "created_utc": c.a.utc(),
                "manifest": c.ref(path),
                "bindings": c.bindings(
                    [
                        path,
                        out / "independent_check.json",
                        l.BASE / "analysis_contract.json",
                        *l.own_sources(),
                    ]
                ),
                "actual_Qwen_calls": 0,
                "D_writes": 0,
            },
        )
        print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    verify(Path(parser.parse_args().manifest))

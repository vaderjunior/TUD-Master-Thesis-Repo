"""German guidelines verify hate."""

from collections import Counter, defaultdict
from itertools import product
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from analysis.knowledge.german_guidelines import hate as h
from analysis.common import scoring as original


def rates(y, pred):
    counts = Counter(zip(map(int, y), map(int, pred)))
    tn, fp, fn, tp = [counts[key] for key in ((0, 0), (0, 1), (1, 0), (1, 1))]
    n = len(y)
    div = lambda a, b: a / b if b else None
    f1 = lambda a, b: a / b if b else 0.0
    rec, spec = (div(tp, tp + fn), div(tn, tn + fp))
    return dict(
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        n=n,
        positive_gold=tp + fn,
        negative_gold=tn + fp,
        predicted_positive=tp + fp,
        total_errors=fp + fn,
        f1_hate=f1(2 * tp, 2 * tp + fp + fn),
        f1_not_hate=f1(2 * tn, 2 * tn + fp + fn),
        macro_f1=(f1(2 * tp, 2 * tp + fp + fn) + f1(2 * tn, 2 * tn + fp + fn)) / 2 if n else None,
        precision=div(tp, tp + fp),
        recall=rec,
        specificity=spec,
        precision_not_hate=div(tn, tn + fn),
        fpr=div(fp, fp + tn),
        fnr=div(fn, fn + tp),
        accuracy=div(tp + tn, n),
        error_rate=div(fp + fn, n),
        predicted_hate=div(tp + fp, n),
        gold_hate=div(tp + fn, n),
        balanced_accuracy=(rec + spec) / 2 if rec is not None and spec is not None else None,
    )


def check_summary(saved, y, pred, checks):
    expected = [rates(y, row) for row in pred]
    for i, result in enumerate(expected):
        same(saved["repeats"][i], result)
        yy, pp = (np.asarray(y), np.asarray(pred[i]))
        same(
            [result[k] for k in ("tn", "fp", "fn", "tp")],
            confusion_matrix(yy, pp, labels=[0, 1]).ravel().tolist(),
        )
        same(result["macro_f1"], f1_score(yy, pp, labels=[0, 1], average="macro", zero_division=0))
        same(result["precision"], precision_score(yy, pp, zero_division=np.nan))
        same(result["recall"], recall_score(yy, pp, zero_division=np.nan))
        checks["sklearn_point_summaries"] += 1
    for name in expected[0]:
        values = [v[name] for v in expected]
        same(saved["mean"][name], np.mean(values))
        same(saved["sample_sd"][name], np.std(values, ddof=1))
        same(saved["range"][name], [min(values), max(values)])


def same(actual, expected):
    if isinstance(expected, dict):
        for k, v in expected.items():
            same(actual[k], v)
    elif isinstance(expected, (list, tuple)):
        h.require(len(actual) == len(expected), "Independent list length differs")
        for a, b in zip(actual, expected):
            same(a, b)
    elif expected is None or (
        isinstance(expected, (float, np.floating)) and (not np.isfinite(expected))
    ):
        h.require(actual is None, "Undefined metric converted to number")
    else:
        h.require(
            np.allclose(actual, expected, rtol=0, atol=3e-12), "Independent numerical value differs"
        )


def bootstrap_metrics(z):
    tn, fp, fn, tp = np.moveaxis(z.astype(float), -1, 0)
    n = tn + fp + fn + tp

    def div(a, b, zero=np.nan):
        out = np.full_like(a, zero, dtype=float)
        np.divide(a, b, where=b != 0, out=out)
        return out

    pos, neg = (div(2 * tp, 2 * tp + fp + fn, 0.0), div(2 * tn, 2 * tn + fp + fn, 0.0))
    return {
        name: value.mean(axis=2)
        for name, value in {
            "macro_f1": (pos + neg) / 2,
            "f1_hate": pos,
            "f1_not_hate": neg,
            "precision": div(tp, tp + fp),
            "recall": div(tp, tp + fn),
            "fpr": div(fp, fp + tn),
            "fnr": div(fn, fn + tp),
            "accuracy": div(tp + tn, n),
            "error_rate": div(fp + fn, n),
            "predicted_hate": div(tp + fp, n),
            "balanced_accuracy": (div(tp, tp + fn) + div(tn, tn + fp)) / 2,
            "fp": fp,
            "fn": fn,
            "total_errors": fp + fn,
        }.items()
    }


def check_interval(saved, values, family):
    invalid = int((~np.isfinite(values)).sum())
    same(saved["undefined_draws"], invalid)
    same(saved["ci95"], None if invalid else np.percentile(values, [2.5, 97.5]).tolist())
    same(saved["family_size"], family)
    same(saved["family_confidence"], 1 - 0.05 / family if family else None)
    same(
        saved["family_interval"],
        np.percentile(values, [0.625, 99.375]).tolist() if family and (not invalid) else None,
    )


def verify(path):
    with h.offline():
        c = h.c
        c.validate()
        contract = c.read(h.BASE / "analysis_contract.json")
        c.verify(contract["inputs"])
        h.require(
            contract["analysis_dependencies"] == h.dependencies(), "Analysis environment changed"
        )
        manifest = c.read(path)
        h.require(
            manifest["status"] == "part2_scored_pending_independent_check",
            "Incomplete Part 2 output",
        )
        c.verify(manifest["outputs"])
        c.verify(manifest["sources"])
        out = c.ROOT / manifest["run"]
        _, cache = h.reader.read_verified()
        endpoints = c.read(cache / "endpoint_setup.json")
        roster = {r["key"]: r for r in c.rows(c.BASE / "roster.jsonl")}
        outputs = defaultdict(dict)
        for row in c.rows(cache / "predictions.jsonl"):
            if row["phase"] in h.PHASES:
                outputs[row["condition"], row["repeat"]][row["key"]] = row["result"]
        checks = Counter()
        for phase in h.PHASES:
            result = c.read(out / phase / "results.json")
            meta = c.read(out / phase / "input_roster.json")
            keys = endpoints[phase]["keys"]
            h.require(
                meta["keys"] == keys
                and result["excluded_keys"] == endpoints[phase]["common_excluded_keys"],
                "Input/mask roster changed",
            )
            y = np.asarray([roster[key]["gate"] for key in keys], dtype=np.int8)
            p = np.asarray(
                [
                    [
                        [
                            (
                                -1
                                if outputs[cond, rep][key] is None
                                else int(outputs[cond, rep][key]["hate"])
                            )
                            for key in keys
                        ]
                        for rep in range(3)
                    ]
                    for cond in h.CONDITIONS
                ],
                dtype=np.int8,
            )
            common = (p >= 0).all(axis=(0, 1))
            yy, pp = (y[common], p[:, :, common])
            same(result["common_inputs"], len(yy))
            same(
                result["common_gold_support"],
                {"not_hate": int((yy == 0).sum()), "hate": int((yy == 1).sum())},
            )
            h.require(
                result["common_keys"] == [key for key, keep in zip(keys, common) if keep],
                "Main comparison IDs differ",
            )
            with np.load(out / phase / "arrays.npz", allow_pickle=False) as f:
                stored = {k: f[k] for k in f.files}
            for name, value in [("gold", y), ("predictions", p), ("common_mask", common)]:
                h.require(
                    np.array_equal(stored[name], value), "Cached binary array changed: " + name
                )
            expected_counts = np.zeros((3, 3, 4), dtype=int)
            for ci, cond in enumerate(h.CONDITIONS):
                check_summary(result["common"][cond], yy, pp[ci], checks)
                for rep in range(3):
                    expected_counts[ci, rep] = confusion_matrix(
                        yy, pp[ci, rep], labels=[0, 1]
                    ).ravel()
                    valid = p[ci, rep] >= 0
                    own = rates(y[valid], p[ci, rep, valid])
                    same(result["own_valid"][cond]["repeats"][rep], own)
                    rows = [{"id": key, "gate": bool(g)} for key, g in zip(keys, y)]
                    outs = {
                        key: {
                            "result": None if value < 0 else {"hate": bool(value)},
                            "n_valid": int(value >= 0),
                        }
                        for key, value in zip(keys, p[ci, rep])
                    }
                    ep = {
                        "all_labels": ["not_hate", "hate"],
                        "macro_labels": ["not_hate", "hate"],
                        "dimension": "hate",
                        "scorable_ids": keys,
                    }
                    original_score = original.score(original.contributions(rows, outs, ep))
                    same(own["macro_f1"], original_score["macro_f1_valid_only"])
                    same(
                        result["full_roster_bounds"][cond]["bounds"]["macro_f1"]["by_repeat"][rep],
                        original_score["invalid_inclusive_bounds"],
                    )
                    missing = np.flatnonzero(~valid)
                    h.require(
                        len(missing) <= 3,
                        "Unexpectedly many missing answers for exact bound enumeration",
                    )
                    scenarios = []
                    for bits in product([0, 1], repeat=len(missing)):
                        possible = p[ci, rep].copy()
                        possible[missing] = bits
                        scenarios.append(rates(y, possible))
                        checks["exact_missing_answer_scenarios"] += 1
                    for name in h.BOUND_METRICS:
                        same(
                            result["full_roster_bounds"][cond]["bounds"][name]["by_repeat"][rep],
                            [min((v[name] for v in scenarios)), max((v[name] for v in scenarios))],
                        )
                    checks["original_contract_point_and_bounds"] += 1
                own_values = result["own_valid"][cond]["repeats"]
                for name in own_values[0]:
                    values = [v[name] for v in own_values]
                    same(result["own_valid"][cond]["mean"][name], np.mean(values))
                    same(result["own_valid"][cond]["sample_sd"][name], np.std(values, ddof=1))
                for name in h.BOUND_METRICS:
                    b = result["full_roster_bounds"][cond]["bounds"][name]
                    same(b["mean"], np.mean(b["by_repeat"], axis=0).tolist())
            h.require(
                np.array_equal(expected_counts, stored["confusion_counts"]),
                "Point confusion array differs",
            )
            design = c.read(out / phase / "bootstrap_design.json")
            clusters = [roster[key]["text_cluster"] for key, on in zip(keys, common) if on]
            h.require(
                len(set(clusters)) == len(clusters),
                "Independent actual-data bootstrap requires the verified singleton clusters",
            )
            unique = sorted(clusters)
            h.require(
                design["clusters"] == unique
                and design["draws"] == 10000
                and (design["seed"] == h.SEEDS[phase]),
                "Bootstrap design differs",
            )
            cluster_to_row = np.asarray([clusters.index(name) for name in unique])
            expected_strata = defaultdict(list)
            for i, rowidx in enumerate(cluster_to_row):
                source = roster[result["common_keys"][rowidx]]["source"]
                expected_strata[source,].append(i)
            strata = [
                {"sources": list(k), "cluster_indices": v}
                for k, v in sorted(expected_strata.items())
            ]
            h.require(design["strata"] == strata, "Source strata differ")
            rng = np.random.default_rng(h.SEEDS[phase])
            digest = hashlib.sha256()
            codes = (2 * yy + pp).reshape(9, len(yy))
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
                selected_codes = codes[:, cluster_to_row[picks]]
                direct = (
                    np.stack([(selected_codes == k).sum(axis=-1) for k in range(4)], axis=-1)
                    .transpose(1, 0, 2)
                    .reshape(size, 3, 3, 4)
                )
                h.require(
                    np.array_equal(direct, stored["bootstrap_confusions"][start : start + size]),
                    "Independent bootstrap confusion counts differ",
                )
                checks["independent_bootstrap_draws"] += size
            h.require(
                digest.hexdigest() == design["pick_stream_sha256"], "Bootstrap pick stream differs"
            )
            bm = bootstrap_metrics(stored["bootstrap_confusions"])
            for ci, cond in enumerate(h.CONDITIONS):
                for name in h.BOOT_METRICS:
                    check_interval(
                        result["condition_descriptive_intervals"][cond][name], bm[name][:, ci], None
                    )
                    checks["condition_intervals"] += 1
            for value in result["contrasts"]:
                left, right = (value["left"], value["right"])
                li, ri = (h.CONDITIONS.index(left), h.CONDITIONS.index(right))
                for name in h.BOOT_METRICS:
                    family = (
                        4
                        if phase == "de_test" and right == "mixed_2" and (name == "macro_f1")
                        else None
                    )
                    check_interval(
                        value["metrics"][name], bm[name][:, li] - bm[name][:, ri], family
                    )
                    same(
                        value["metrics"][name]["difference"],
                        result["common"][left]["mean"][name]
                        - result["common"][right]["mean"][name],
                    )
                    same(
                        value["metrics"][name]["by_repeat"],
                        [
                            result["common"][left]["repeats"][j][name]
                            - result["common"][right]["repeats"][j][name]
                            for j in range(3)
                        ],
                    )
                    checks["contrast_intervals"] += 1
                flow = np.zeros((3, 3, 8), dtype=int)
                for i in range(3):
                    for j in range(3):
                        for truth, a, b in zip(yy, pp[li, i], pp[ri, j]):
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
                            flow[i, j, event] += 1
                h.require(
                    np.array_equal(value["transitions"]["all_nine_pair_counts"], flow),
                    "Independent error-flow partition differs",
                )
                same(
                    value["transitions"]["mean_events"],
                    {name: flow[:, :, i].mean() for i, name in enumerate(h.EVENTS)},
                )
                for name in h.BOUND_METRICS:
                    lo = result["full_roster_bounds"][left]["bounds"][name]["mean"]
                    hi = result["full_roster_bounds"][right]["bounds"][name]["mean"]
                    same(
                        value["full_roster_failure_difference_bounds"][name],
                        [lo[0] - hi[1], lo[1] - hi[0]],
                    )
                checks["nine_pair_transition_partitions"] += 1
            print(
                phase
                + ": independent metrics, exact missing-answer bounds and all 10,000 draws verified.",
                flush=True,
            )
        c.verify(manifest["outputs"])
        c.verify(contract["inputs"])
        c.verify(c.read(h.reader.BASE / "analysis_contract.json")["source_bindings"])
        report = {
            "status": "independent_binary_analysis_check_pass",
            "checks": dict(checks),
            "dependencies": h.dependencies(),
            "actual_Qwen_calls": 0,
            "D_writes": 0,
            "legal_effectiveness_computed": False,
        }
        h.save(out / "independent_check.json", report)
        h.save(
            h.BASE / "complete.json",
            {
                "status": "part2_numerically_verified",
                "created_utc": c.a.utc(),
                "manifest": c.ref(path),
                "bindings": c.bindings(
                    [
                        path,
                        out / "independent_check.json",
                        h.BASE / "analysis_contract.json",
                        *h.own_sources(),
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
    args = parser.parse_args()
    verify(Path(args.manifest))

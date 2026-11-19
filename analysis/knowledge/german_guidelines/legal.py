"""German guidelines legal."""

from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from analysis.knowledge.german_guidelines import hate as h

c, reader, require = (h.c, h.reader, h.require)
offline, safe, divide, interval = (h.offline, h.safe, h.divide, h.interval)
BASE = c.ROOT / "experiments/astra/german_guidelines_legal_analysis_20261113_v1"
PHASES = ("de_legal_test", "de_legal_dev_eval")
CONDITIONS, PAIRS = (h.CONDITIONS, h.PAIRS)
LABELS = ("propaganda", "incitement_public_peace", "insult_defamation")
DRAWS = 10000
SEEDS = {"de_legal_test": 2026111303, "de_legal_dev_eval": 2026111304}
FEATURES = (
    "n",
    "gold_labels",
    "predicted_labels",
    "correct_labels",
    "extra_labels",
    "missed_labels",
    "exact_correct",
    "texts_with_extra_labels",
    "texts_with_missed_labels",
    "only_extra",
    "only_missing",
    "mixed_error",
    "gold_empty",
    "gold_nonempty",
    "pred_empty",
    "empty_gold_correct",
    "empty_gold_false_labels",
    "nonempty_gold_exact",
    "nonempty_gold_pred_empty",
    "pred_cardinality_0",
    "pred_cardinality_1",
    "pred_cardinality_2",
    "pred_cardinality_3",
)
BOOT_METRICS = (
    "macro_f1",
    "micro_f1",
    "micro_precision",
    "micro_recall",
    "exact_set_accuracy",
    "exact_correct",
    "mean_predicted_labels",
    "extra_labels",
    "missed_labels",
    "label_errors",
    "texts_with_extra_labels",
    "texts_with_missed_labels",
    "empty_gold_false_positive_rate",
    "nonempty_gold_exact_accuracy",
    "nonempty_gold_empty_rate",
    "empty_prediction_rate",
    "hamming_accuracy",
)
LABEL_METRICS = (
    "f1",
    "precision",
    "recall",
    "fpr",
    "fnr",
    "fp",
    "fn",
    "tp",
    "tn",
    "support",
    "predicted",
)
BOUND_METRICS = (
    "macro_f1",
    "micro_f1",
    "exact_set_accuracy",
    "exact_correct",
    "extra_labels",
    "missed_labels",
    "label_errors",
    "mean_predicted_labels",
    "hamming_accuracy",
)


def save(path, value):
    require(c.normal(path).is_relative_to(BASE), "Part 3 write outside new folder")
    c.save(path, safe(value))


def arrays(path, **values):
    require(c.normal(path).is_relative_to(BASE), "Part 3 arrays outside new folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **values)


def bits(values, labels=LABELS):
    require(type(values) is list and set(values) <= set(labels), "Missing or unknown legal labels")
    return [int(label in values) for label in labels]


def item_values(y, p):
    y, p = (np.asarray(y), np.asarray(p))
    require(y.ndim == 2 and y.shape[-1] == 3 and (p.shape[-2:] == y.shape), "Wrong legal axes")
    require(
        np.isin(y, [0, 1]).all() and np.isin(p, [0, 1]).all(),
        "Missing legal answer must not be scored as empty",
    )
    y, p = (y.astype(bool), p.astype(bool))
    counts = np.stack((~y & ~p, ~y & p, y & ~p, y & p), axis=-1).astype(np.int64)
    gold = np.broadcast_to(y.sum(axis=-1), p.shape[:-1])
    pred = p.sum(axis=-1)
    extra, missed = ((p & ~y).sum(axis=-1), (~p & y).sum(axis=-1))
    exact = extra + missed == 0
    feature = np.stack(
        (
            np.ones_like(pred),
            gold,
            pred,
            (p & y).sum(axis=-1),
            extra,
            missed,
            exact,
            extra > 0,
            missed > 0,
            (extra > 0) & (missed == 0),
            (extra == 0) & (missed > 0),
            (extra > 0) & (missed > 0),
            gold == 0,
            gold > 0,
            pred == 0,
            (gold == 0) & (pred == 0),
            (gold == 0) & (pred > 0),
            (gold > 0) & exact,
            (gold > 0) & (pred == 0),
            pred == 0,
            pred == 1,
            pred == 2,
            pred == 3,
        ),
        axis=-1,
    ).astype(np.int64)
    return (counts, feature)


def totals(y, p):
    counts, feature = item_values(y, p)
    return (counts.sum(axis=-3), feature.sum(axis=-2))


def metrics(counts, features, macro_indices=(0, 1, 2)):
    z, f = (np.asarray(counts), np.asarray(features))
    require(z.shape[-2:] == (3, 4) and f.shape == z.shape[:-2] + (len(FEATURES),), "Wrong totals")
    require((z >= 0).all() and (f >= 0).all() and (len(macro_indices) > 0), "Invalid legal totals")
    tn, fp, fn, tp = np.moveaxis(z, -1, 0)
    x = {name: f[..., i] for i, name in enumerate(FEATURES)}
    n = x["n"]
    require(
        np.array_equal(z.sum(axis=-1), np.broadcast_to(n[..., None], z.shape[:-1])),
        "Label support differs from set denominator",
    )
    per = dict(
        tn=tn,
        fp=fp,
        fn=fn,
        tp=tp,
        support=tp + fn,
        predicted=tp + fp,
        f1=divide(2 * tp, 2 * tp + fp + fn, 0.0),
        precision=divide(tp, tp + fp),
        recall=divide(tp, tp + fn),
        fpr=divide(fp, fp + tn),
        fnr=divide(fn, fn + tp),
    )
    tp_all, fp_all, fn_all = (value.sum(axis=-1) for value in (tp, fp, fn))
    out = dict(
        x,
        macro_f1=np.where(n > 0, per["f1"][..., list(macro_indices)].mean(axis=-1), np.nan),
        micro_f1=divide(2 * tp_all, 2 * tp_all + fp_all + fn_all, 0.0),
        micro_precision=divide(tp_all, tp_all + fp_all),
        micro_recall=divide(tp_all, tp_all + fn_all),
        exact_set_accuracy=divide(x["exact_correct"], n),
        exact_wrong=n - x["exact_correct"],
        mean_predicted_labels=divide(x["predicted_labels"], n),
        mean_gold_labels=divide(x["gold_labels"], n),
        label_errors=fp_all + fn_all,
        hamming_accuracy=1 - divide(fp_all + fn_all, 3 * n),
        empty_gold_false_positive_rate=divide(x["empty_gold_false_labels"], x["gold_empty"]),
        nonempty_gold_exact_accuracy=divide(x["nonempty_gold_exact"], x["gold_nonempty"]),
        nonempty_gold_empty_rate=divide(x["nonempty_gold_pred_empty"], x["gold_nonempty"]),
        empty_prediction_rate=divide(x["pred_empty"], n),
    )
    require(
        np.array_equal(out["extra_labels"], fp_all)
        and np.array_equal(out["missed_labels"], fn_all),
        "Set and label errors differ",
    )
    require(
        np.array_equal(
            x["only_extra"] + x["only_missing"] + x["mixed_error"] + x["exact_correct"], n
        ),
        "Set errors do not partition texts",
    )
    return (out, per)


def summarize(counts, features, macro_indices):
    require(
        counts.shape == (3, 3, 4) and features.shape == (3, len(FEATURES)), "Three repeats required"
    )
    aggregate, per = metrics(counts, features, macro_indices)

    def summaries(values):
        return {
            "repeats": [{name: value[i] for name, value in values.items()} for i in range(3)],
            "mean": {name: value.mean(axis=0) for name, value in values.items()},
            "sample_sd": {name: value.std(axis=0, ddof=1) for name, value in values.items()},
            "range": {
                name: [value.min(axis=0), value.max(axis=0)] for name, value in values.items()
            },
        }

    return safe(
        dict(
            summaries(aggregate),
            per_label={
                label: summaries({k: v[:, j] for k, v in per.items()})
                for j, label in enumerate(LABELS)
            },
        )
    )


def failure_bounds(y, pred, macro_indices):
    require(pred.shape == (3, len(y), 3), "Wrong missing legal axes")
    result, counts = ([], [])
    for p in pred:
        require(
            np.isin(p, [-1, 0, 1]).all()
            and np.array_equal((p < 0).any(axis=1), (p < 0).all(axis=1)),
            "Partial null legal vector",
        )
        missing = np.flatnonzero((p < 0).all(axis=1))
        require(
            len(missing) <= 3,
            "Exact bound enumeration needs review for more than three unknown sets",
        )
        scenarios = []
        for values in product((0, 1), repeat=3 * len(missing)):
            candidate = p.copy()
            if len(missing):
                candidate[missing] = np.asarray(values).reshape(-1, 3)
            m, _ = metrics(*totals(y, candidate), macro_indices)
            scenarios.append(m)
        result.append(
            {
                name: [min((v[name] for v in scenarios)), max((v[name] for v in scenarios))]
                for name in BOUND_METRICS
            }
        )
        counts.append(len(scenarios))
    return safe(
        {
            "unknown_by_repeat": (pred < 0).all(axis=-1).sum(axis=-1),
            "enumerated_sets_by_repeat": counts,
            "bounds": {
                name: {
                    "by_repeat": [v[name] for v in result],
                    "mean": np.mean([v[name] for v in result], axis=0),
                }
                for name in BOUND_METRICS
            },
            "meaning": "Exact full-roster extrema over all eight legal label subsets for each missing answer; not imputed outputs or confidence intervals.",
        }
    )


def bootstrap(y, p, clusters, full_memberships, seed, draws=DRAWS):
    require(p.shape == (3, 3, len(y), 3) and len(clusters) == len(y), "Bootstrap legal axes differ")
    count, feat = item_values(y, p.reshape(9, len(y), 3))
    joined = np.concatenate((count.reshape(9, len(y), 12), feat), axis=-1)
    unique = sorted(set(clusters))
    lookup = {cl: i for i, cl in enumerate(unique)}
    grouped = np.zeros((9, len(unique), joined.shape[-1]), dtype=np.int64)
    for i, cl in enumerate(clusters):
        grouped[:, lookup[cl]] += joined[:, i]
    strata = defaultdict(list)
    for i, cl in enumerate(unique):
        strata[tuple(sorted(full_memberships[cl]))].append(i)
    ordered = [group for _, group in sorted(strata.items())]
    rng, digest = (np.random.default_rng(seed), hashlib.sha256())
    combined = np.empty((draws, 9, joined.shape[-1]), dtype=np.int64)
    checks = 0
    for start in range(0, draws, 64):
        size = min(64, draws - start)
        picks = np.concatenate(
            [rng.choice(group, (size, len(group)), replace=True) for group in ordered], axis=1
        )
        digest.update(picks.astype("<i4").tobytes())
        weights = np.stack([np.bincount(row, minlength=len(unique)) for row in picks])
        block = np.einsum("bg,sgk->bsk", weights, grouped, optimize=True)
        require(
            np.array_equal(block[0], grouped[:, picks[0]].sum(axis=1)),
            "Direct bootstrap block check failed",
        )
        combined[start : start + size] = block
        checks += 1
    design = {
        "draws": draws,
        "seed": seed,
        "block_size": 64,
        "clusters": unique,
        "cluster_rows": [[i for i, name in enumerate(clusters) if name == cl] for cl in unique],
        "strata": [{"sources": list(k), "cluster_indices": v} for k, v in sorted(strata.items())],
        "pick_stream_sha256": digest.hexdigest(),
        "direct_block_checks": checks,
        "unit": "Whole normalized-text clusters with full-roster source memberships; common draws across all nine settings and all legal labels.",
    }
    return (
        combined[..., :12].reshape(draws, 3, 3, 3, 4),
        combined[..., 12:].reshape(draws, 3, 3, len(FEATURES)),
        design,
    )


def transitions(y, left, right):
    by_label = {
        label: h.transition_summary(y[:, j], left[:, :, j], right[:, :, j])
        for j, label in enumerate(LABELS)
    }
    exact_left, exact_right = ((left == y).all(axis=-1), (right == y).all(axis=-1))
    a, b = (exact_left[:, None], exact_right[None, :])
    values = np.stack((a & ~b, ~a & b, a & b, ~a & ~b), axis=-1).sum(axis=2)
    names = ("exact_fixed", "exact_lost", "both_exact", "both_wrong")
    return safe(
        {
            "per_label": by_label,
            "mean_label_events": {
                name: sum((v["mean_events"][name] for v in by_label.values())) for name in h.EVENTS
            },
            "exact_all_nine_pair_counts": values,
            "exact_event_order": names,
            "mean_exact_events": {name: values[:, :, i].mean() for i, name in enumerate(names)},
            "pairing": "All nine cross-repeat pairs equally weighted; not independent observations, chronological corrections or stable per-text fixes.",
        }
    )


def set_matrix(y, pred):
    codes = np.asarray([1, 2, 4])
    gold = y @ codes
    rows = []
    for p in pred:
        matrix = np.zeros((8, 8), dtype=np.int64)
        np.add.at(matrix, (gold, p @ codes), 1)
        rows.append(matrix)
    return safe(
        {
            "set_order": [
                [label for i, label in enumerate(LABELS) if code & 1 << i] for code in range(8)
            ],
            "rows_gold_columns_prediction": True,
            "repeats": rows,
            "mean": np.mean(rows, axis=0),
        }
    )


def own_sources():
    return [
        Path(__file__),
        c.ROOT / "tests/knowledge/german_guidelines/test_legal.py",
        c.ROOT / "analysis/knowledge/german_guidelines/verify_legal.py",
        Path(h.__file__),
        c.ROOT / "analysis/common/scoring.py",
    ]


def load():
    c.validate()
    c.verify(c.read(reader.BASE / "analysis_contract.json")["source_bindings"])
    _, cache = reader.read_verified()
    endpoints = c.read(cache / "endpoint_setup.json")
    data = {}
    for phase in PHASES:
        ep = endpoints[phase]
        require(
            ep["dimension"] == "legal" and ep["all_labels"] == list(LABELS),
            "Wrong legal label universe",
        )
        require(
            ep["macro_labels"] == [label for label in LABELS if ep["gold_support"][label] >= 10],
            "Full-roster macro labels changed",
        )
        data[phase] = {
            "endpoint": ep,
            "gold": np.asarray([bits(x) for x in ep["gold"]], dtype=np.int8),
            "index": {key: i for i, key in enumerate(ep["keys"])},
            "pred": np.full((3, 3, len(ep["keys"]), 3), -9, dtype=np.int8),
            "hate": np.full((3, 3, len(ep["keys"])), -9, dtype=np.int8),
        }
    count = 0
    for row in c.rows(cache / "predictions.jsonl"):
        count += 1
        if row["phase"] not in data:
            continue
        d = data[row["phase"]]
        i, ci, rep = (d["index"][row["key"]], CONDITIONS.index(row["condition"]), row["repeat"])
        require((d["pred"][ci, rep, i] == -9).all(), "Duplicate legal item")
        if row["result"] is None:
            require(row["n_valid"] == 0, "Contradictory null")
            d["pred"][ci, rep, i] = -1
            d["hate"][ci, rep, i] = -1
        else:
            require(
                row["n_valid"] > 0 and type(row["result"]["hate"]) is bool,
                "Contradictory valid item",
            )
            d["pred"][ci, rep, i] = bits(row["result"]["legal"])
            d["hate"][ci, rep, i] = int(row["result"]["hate"])
    require(count == 33093, "Verified cache incomplete")
    for d in data.values():
        require(np.isin(d["pred"], [-1, 0, 1]).all(), "Missing prediction")
        d["common"] = (d["pred"] >= 0).all(axis=(0, 1, 3))
        require(
            [k for k, on in zip(d["endpoint"]["keys"], d["common"]) if on]
            == d["endpoint"]["common_valid_keys"],
            "Part 1 mask changed",
        )
    return (data, cache)


def prepare():
    c.validate()
    _, cache = reader.read_verified()
    h.verified()
    tests = sorted((BASE / "tests").glob("tests_*/report.json"))
    require(tests and c.read(tests[-1])["passed"], "Passing legal tests required")
    require(
        c.read(tests[-1])["sources"] == c.bindings(own_sources()), "Tests do not cover current code"
    )
    save(
        BASE / "analysis_contract.json",
        {
            "status": "part3_frozen_before_legal_effectiveness_calculation",
            "created_utc": c.a.utc(),
            "inputs": c.bindings(
                [
                    c.PLAN,
                    c.PROTOCOL,
                    reader.ANALYSIS_PLAN,
                    reader.BASE / "complete.json",
                    reader.BASE / "analysis_contract.json",
                    cache / "manifest.json",
                    h.BASE / "complete.json",
                    tests[-1],
                    *own_sources(),
                ]
            ),
            "analysis_dependencies": h.dependencies(),
            "phases": PHASES,
            "conditions": CONDITIONS,
            "pairs": PAIRS,
            "labels": LABELS,
            "mask": "Unchanged Part 1 all-condition/all-repeat common-valid masks; legal gold independent of hate; observed empty is valid, null stays missing.",
            "metrics": "Mean of three separate repeat scores. Fixed full-roster supported legal labels. Per-label F1/precision/recall/errors, exact-set match, extra/missed labels, cardinality, empty/nonempty gold groups. No hate gate applied.",
            "uncertainty": {
                "draws": DRAWS,
                "seeds": SEEDS,
                "block_size": 64,
                "method": "Paired whole normalized-text clusters in full-roster source-membership strata; shared across all conditions, repeats and labels.",
                "test_family_size": 4,
                "family_role": "Only two legal TEST baseline macro-F1 contrasts receive 98.75% intervals; the two hate contrasts already belong to this same family.",
                "supporting": "Added-legal comparison, DEV and all other metrics/categories use descriptive 95%. Undefined ratios/draws are explicit, not silently removed.",
                "limits": "Conditional on the fixed three decoder repeats and exposed cohort; not all generation variance or equivalence evidence.",
            },
            "missing": "Own-valid coverage/scores plus exact full-roster enumeration of all eight label subsets per missing answer. Bounds are separate from uncertainty intervals.",
            "error_flows": "All nine cross-repeat pairs equally weighted, per label and exact set; no individual case/stability attribution.",
            "gate_diagnostic": "Count nonempty legal predictions despite a false predicted hate gate, without erasing them or interpreting them as errors.",
            "protected": [
                "original preparation and raw D evidence",
                "Part 1 and Part 2",
                "thesis/presentation/learning guide",
            ],
            "next": "Actual context changes Part 4; detailed datasets/cases/repeat stability Part 5; remaining robustness Part 6; synthesis Part 7; writer handover Part 8.",
            "new_Qwen_calls": 0,
        },
    )
    print("Part 3 contract frozen before legal effectiveness scoring.", flush=True)


def run():
    contract = c.read(BASE / "analysis_contract.json")
    c.verify(contract["inputs"])
    require(contract["analysis_dependencies"] == h.dependencies(), "Analysis environment changed")
    data, cache = load()
    out = BASE / "runs" / f"run_{time.time_ns()}"
    out.mkdir(parents=True)
    save(
        out / "start.json",
        {"created_utc": c.a.utc(), "contract_sha256": c.sha(BASE / "analysis_contract.json")},
    )
    for phase in PHASES:
        d = data[phase]
        ep, y, p, mask = (d["endpoint"], d["gold"], d["pred"], d["common"])
        idx = [list(LABELS).index(label) for label in ep["macro_labels"]]
        yy, pp = (y[mask], p[:, :, mask])
        counts, features = totals(yy, pp)
        shared, own, bounds, sets, gates = ({}, {}, {}, {}, {})
        for ci, cond in enumerate(CONDITIONS):
            shared[cond] = summarize(counts[ci], features[ci], idx)
            valid = (p[ci] >= 0).all(axis=-1)
            own_totals = [totals(y[v], row[v]) for row, v in zip(p[ci], valid)]
            own[cond] = summarize(
                np.stack([v[0] for v in own_totals]), np.stack([v[1] for v in own_totals]), idx
            )
            bounds[cond] = failure_bounds(y, p[ci], idx)
            sets[cond] = set_matrix(yy, pp[ci])
            pg = d["hate"][ci, :, mask].T
            require(pg.shape == (3, len(yy)), "Predicted hate axes differ")
            gates[cond] = {
                "by_repeat": [
                    {
                        "predicted_nonhate": int((g == 0).sum()),
                        "nonempty_legal_despite_predicted_nonhate": int(
                            ((g == 0) & (v.sum(axis=-1) > 0)).sum()
                        ),
                    }
                    for g, v in zip(pg, pp[ci])
                ],
                "meaning": "These legal answers remain scored; this combination is permitted, not an automatic error.",
            }
        boot_counts, boot_features, design = bootstrap(
            yy,
            pp,
            [cl for cl, on in zip(ep["clusters"], mask) if on],
            ep["source_memberships_by_cluster"],
            SEEDS[phase],
        )
        bm, bl = metrics(boot_counts, boot_features, idx)
        bm, bl = ({k: v.mean(axis=2) for k, v in group.items()} for group in (bm, bl))
        condition_ci = {
            cond: {name: interval(bm[name][:, i]) for name in BOOT_METRICS}
            for i, cond in enumerate(CONDITIONS)
        }
        label_ci = {
            cond: {
                label: {name: interval(bl[name][:, i, j]) for name in LABEL_METRICS}
                for j, label in enumerate(LABELS)
            }
            for i, cond in enumerate(CONDITIONS)
        }
        contrasts = []
        for left, right in PAIRS:
            li, ri = (CONDITIONS.index(left), CONDITIONS.index(right))
            main = phase == "de_legal_test" and right == "mixed_2"
            aggregate, perlabel = ({}, {})
            for name in BOOT_METRICS:
                aggregate[name] = dict(
                    interval(
                        bm[name][:, li] - bm[name][:, ri],
                        4 if main and name == "macro_f1" else None,
                    ),
                    difference=shared[left]["mean"][name] - shared[right]["mean"][name],
                    by_repeat=[
                        shared[left]["repeats"][i][name] - shared[right]["repeats"][i][name]
                        for i in range(3)
                    ],
                )
            for j, label in enumerate(LABELS):
                perlabel[label] = {}
                for name in LABEL_METRICS:
                    a, b = (shared[left]["per_label"][label], shared[right]["per_label"][label])
                    perlabel[label][name] = dict(
                        interval(bl[name][:, li, j] - bl[name][:, ri, j]),
                        difference=a["mean"][name] - b["mean"][name],
                        by_repeat=[a["repeats"][i][name] - b["repeats"][i][name] for i in range(3)],
                    )
            flow = transitions(yy, pp[li], pp[ri])
            for metric, fixed, added in [
                ("extra_labels", "fp_fixed", "fp_introduced"),
                ("missed_labels", "fn_fixed", "fn_introduced"),
            ]:
                require(
                    abs(
                        flow["mean_label_events"][added]
                        - flow["mean_label_events"][fixed]
                        - aggregate[metric]["difference"]
                    )
                    < 1e-10,
                    "Legal net/gross errors differ",
                )
            require(
                abs(
                    flow["mean_exact_events"]["exact_fixed"]
                    - flow["mean_exact_events"]["exact_lost"]
                    - aggregate["exact_correct"]["difference"]
                )
                < 1e-10,
                "Exact-set net/gross changes differ",
            )
            contrasts.append(
                {
                    "left": left,
                    "right": right,
                    "role": (
                        "main TEST macro-F1 contrast in four-comparison family"
                        if main
                        else "supporting descriptive contrast"
                    ),
                    "metrics": aggregate,
                    "per_label": perlabel,
                    "transitions": flow,
                    "own_valid_macro_difference": own[left]["mean"]["macro_f1"]
                    - own[right]["mean"]["macro_f1"],
                    "full_roster_failure_difference_bounds": {
                        name: [
                            bounds[left]["bounds"][name]["mean"][0]
                            - bounds[right]["bounds"][name]["mean"][1],
                            bounds[left]["bounds"][name]["mean"][1]
                            - bounds[right]["bounds"][name]["mean"][0],
                        ]
                        for name in BOUND_METRICS
                    },
                }
            )
        result = {
            "population": phase,
            "full_inputs": len(y),
            "common_inputs": len(yy),
            "common_keys": [key for key, on in zip(ep["keys"], mask) if on],
            "excluded_keys": ep["common_excluded_keys"],
            "labels": LABELS,
            "macro_labels": ep["macro_labels"],
            "full_gold_support": ep["gold_support"],
            "common_gold_support": dict(zip(LABELS, yy.sum(axis=0))),
            "full_observed_empty_gold": int((y.sum(axis=1) == 0).sum()),
            "common_observed_empty_gold": int((yy.sum(axis=1) == 0).sum()),
            "common_gold_cardinality": dict(Counter(map(int, yy.sum(axis=1)))),
            "common_source_counts": dict(Counter((s for s, on in zip(ep["sources"], mask) if on))),
            "common": shared,
            "own_valid": own,
            "full_roster_bounds": bounds,
            "condition_descriptive_intervals": condition_ci,
            "per_label_descriptive_intervals": label_ci,
            "contrasts": contrasts,
            "set_confusions": sets,
            "predicted_gate_diagnostic": gates,
            "legal_independent_of_hate_gate": True,
            "repeats_not_independent_texts": True,
        }
        save(out / phase / "results.json", result)
        save(
            out / phase / "input_roster.json",
            {
                "keys": ep["keys"],
                "sources": ep["sources"],
                "clusters": ep["clusters"],
                "conditions": CONDITIONS,
                "labels": LABELS,
                "repeat_indices": [0, 1, 2],
                "confusion_order": ["tn", "fp", "fn", "tp"],
                "feature_order": FEATURES,
            },
        )
        save(out / phase / "bootstrap_design.json", design)
        arrays(
            out / phase / "arrays.npz",
            gold=y,
            predictions=p,
            predicted_hate=d["hate"],
            common_mask=mask,
            confusion_counts=counts,
            set_features=features,
            bootstrap_confusions=boot_counts,
            bootstrap_features=boot_features,
        )
        print(
            phase + ": legal categories, full label sets and 10,000 paired draws saved.", flush=True
        )
    c.verify(contract["inputs"])
    c.verify(c.read(reader.BASE / "analysis_contract.json")["source_bindings"])
    c.validate()
    save(
        out / "source_and_preservation_check.json",
        {
            "status": "bound_inputs_and_previous_parts_unchanged",
            "hashes": contract["inputs"],
            "actual_Qwen_calls": 0,
            "D_writes": 0,
            "latex_edits": 0,
            "raw_full_tree_reaudited": False,
        },
    )
    save(
        out / "manifest.json",
        {
            "status": "part3_scored_pending_independent_check",
            "run": c.ref(out),
            "created_utc": c.a.utc(),
            "contract_sha256": c.sha(BASE / "analysis_contract.json"),
            "outputs": c.bindings((p for p in out.rglob("*") if p.is_file())),
            "sources": c.bindings(own_sources()),
            "actual_Qwen_calls": 0,
            "D_writes": 0,
        },
    )
    print(json.dumps({"manifest": c.ref(out / "manifest.json")}), flush=True)


def verified():
    done = c.read(BASE / "complete.json")
    require(
        done["status"] == "part3_numerically_verified", "Independent Part 3 verification missing"
    )
    c.verify(done["bindings"])
    manifest = c.read(c.ROOT / done["manifest"])
    c.verify(manifest["outputs"])
    return (manifest, c.ROOT / manifest["run"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run"))
    args = parser.parse_args()
    with offline():
        prepare() if args.mode == "prepare" else run()

from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import uuid
import zipfile

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits

from analysis.classification.supervised import archive as r
from analysis.classification.supervised import metrics as m
from analysis.classification.supervised.prepare_type_update import BASE
from analysis.classification.supervised.prepare_type_update import EXECUTION
from analysis.classification.supervised.prepare_type_update import MAIN

POOL = "uniform_compact_s42"
METHOD = "tuned_selected"
TYPE_ENDPOINTS = [
    ("broad", "en_test/hate_type"),
    ("core", "en_dev_eval_types"),
    ("core", "en_dev_eval_sq3_types"),
]


def native(v):
    if isinstance(v, dict):
        return {str(k): native(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [native(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def save(path, value):
    m.save(path, native(value))


def quality(y, predictions, gates):
    "All columns remain in the macro score, including zero-positive columns."
    y = np.asarray(y, dtype=bool)
    p = np.asarray(predictions, dtype=bool)
    g = np.asarray(gates, dtype=bool)
    r.require(
        y.ndim == 2 and p.ndim == 3 and p.shape[1:] == y.shape and len(y) > 0,
        "Wrong quality shapes or empty population.",
    )
    r.require(g.shape == p.shape[:2], "Wrong gate axis.")
    scores, counts = m.point_scores(y, p, list(range(y.shape[1])))
    tp, fp, fn = (counts[k].sum(axis=1) for k in ("tp", "fp", "fn"))
    divide = lambda a, b: np.divide(
        a, b, out=np.zeros_like(a, dtype=float), where=b != 0
    )
    exact = (p == y).all(axis=2)
    return dict(
        macro_f1=scores,
        micro_f1=divide(2 * tp, 2 * tp + fp + fn),
        micro_precision=divide(tp, tp + fp),
        micro_recall=divide(tp, tp + fn),
        exact_set=exact.mean(axis=1),
        hate_and_type_exact=(exact & g).mean(axis=1),
        hamming_error=(p != y).mean(axis=(1, 2)),
        predicted_labels=p.sum(axis=2).mean(axis=1),
        gold_labels=float(y.sum(axis=1).mean()),
        empty_rate=(p.sum(axis=2) == 0).mean(axis=1),
        predicted_hate=g.mean(axis=1),
        counts=counts,
        label_predicted_count=p.sum(axis=1),
        label_gold_count=y.sum(axis=0),
        label_precision=divide(counts["tp"], counts["tp"] + counts["fp"]),
        label_recall=divide(counts["tp"], counts["tp"] + counts["fn"]),
    )


def transitions(y, before, after):
    y = np.asarray(y, bool)
    old, new = np.asarray(before, bool), np.asarray(after, bool)
    r.require(
        old.shape == new.shape and old.ndim == 3 and old.shape[1:] == y.shape,
        "Wrong paired transition shapes.",
    )
    added, removed = new & ~old, old & ~new
    fixed = (added & y) | (removed & ~y)
    harmed = (added & ~y) | (removed & y)
    old_exact, new_exact = (old == y).all(2), (new == y).all(2)
    changed = (old != new).any(2)
    row_gain = fixed.sum(2) - harmed.sum(2)
    return dict(
        changed_rows=changed.sum(1),
        changed_labels=(added | removed).sum((1, 2)),
        added_tp=(added & y).sum(1),
        added_fp=(added & ~y).sum(1),
        removed_tp=(removed & y).sum(1),
        removed_fp=(removed & ~y).sum(1),
        label_fixes=fixed.sum((1, 2)),
        label_new_errors=harmed.sum((1, 2)),
        exact_fixes=(~old_exact & new_exact).sum(1),
        exact_harms=(old_exact & ~new_exact).sum(1),
        changed_but_inexact=(changed & ~old_exact & ~new_exact).sum(1),
        fewer_label_errors=(row_gain > 0).sum(1),
        more_label_errors=(row_gain < 0).sum(1),
        equal_label_errors_changed=((row_gain == 0) & changed).sum(1),
        unique_changed_rows=int(changed.any(0).sum()),
        changed_seed_histogram=dict(Counter(changed.sum(0).tolist())),
        exact_fixes_seed_histogram=dict(
            Counter((~old_exact & new_exact).sum(0).tolist())
        ),
        exact_harms_seed_histogram=dict(
            Counter((old_exact & ~new_exact).sum(0).tolist())
        ),
    )


def row_change_draws(values, clusters, draws=10000, seed=20261020):
    "Bootstrap means with whole-cluster multiplicity, including unequal clusters."
    values = np.asarray(values, dtype=float)
    r.require(
        values.ndim == 2 and len(clusters) == values.shape[1] and values.shape[1] > 0,
        "Wrong per-row change dimensions.",
    )
    unique, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    terms = np.stack(
        [np.bincount(inverse, weights=v, minlength=len(unique)) for v in values], axis=1
    )
    sizes = np.bincount(inverse, minlength=len(unique))
    rng = np.random.default_rng(seed)
    out = np.empty((draws, len(values)))
    for first in range(0, draws, 64):
        count = min(64, draws - first)
        picks = rng.integers(0, len(unique), size=(count, len(unique)), dtype=np.int64)
        weights = np.array([np.bincount(x, minlength=len(unique)) for x in picks])
        out[first : first + count] = (weights @ terms) / (weights @ sizes)[:, None]
    return out


def interval(values, alpha=0.05):
    return np.quantile(values, [alpha / 2, 1 - alpha / 2]).tolist()


def compare_arrays(before, after, labels, inputs):
    r.validate_decisions(before, labels, len(inputs))
    r.validate_decisions(after, labels, len(inputs))
    r.require(set(before) == set(after), "Prediction fields changed.")
    unchanged = [k for k in before if k not in ("hate_type", "hate_type_ungated")]
    for key in unchanged:
        r.require(np.array_equal(before[key], after[key]), "Other head changed: " + key)
    for values in (before, after):
        raw = values["hate_type_ungated"].copy()
        raw[values["hate"][:, 1] == 0] = 0
        r.require(np.array_equal(raw, values["hate_type"]), "Gated type mismatch.")
    return len(unchanged)


def read_sources(run):
    contract = r.read(BASE / "analysis_contract.json")
    for name, expected in contract["sources"].items():
        r.checked_file(name, expected)
    for name, expected in r.read(MAIN / "complete.json")["outputs"].items():
        r.checked_file(MAIN / name, expected)
    study = r.GpuStudy()
    linear = r.LinearStudy(study)
    context = r.SharedContext(study, linear)
    pairs = {policy: {"old": [], "new": []} for policy in r.POLICIES}
    old_logits, new_logits = [], []
    arrays_checked = 0
    with zipfile.ZipFile(EXECUTION / "execution_review.zip") as z, study.pool(
        POOL
    ) as saved:
        manifest = r.parsed(z.read("review_manifest.json"))["files"]
        r.require(
            set(z.namelist()) == set(manifest) | {"review_manifest.json"},
            "Supplement member roster changed.",
        )

        def raw(name):
            value = z.read(name)
            r.require(
                r.digest(value) == manifest[name], "Supplement hash differs: " + name
            )
            return value

        protocol = r.parsed(raw("input/protocol.json"))
        labels = saved.labels["hate_type"]
        r.require(
            labels == protocol["labels"]
            and protocol["fit_ids"]
            == [x["id"] for x in saved.data["heads"]["hate_type"]["rows"]],
            "TRAIN/label roster differs.",
        )
        train_rows = saved.data["heads"]["hate_type"]["rows"]
        train_y = np.array([x["target"] for x in train_rows])
        training = dict(
            ids=[x["id"] for x in train_rows],
            examples=len(train_rows),
            label_counts=dict(zip(labels, train_y.sum(0).tolist())),
            TRAIN_selection_labels=[
                labels[j]
                for j in range(len(labels))
                if train_y[:, j].sum() >= 2 and (1 - train_y[:, j]).sum() >= 2
            ],
            absent_labels=[
                labels[j] for j in range(len(labels)) if not train_y[:, j].any()
            ],
            configuration=protocol["configuration"],
            cutoff=protocol["cutoff"],
        )
        for seed in r.SEEDS:
            old_job = saved.job("hate_type", "tuned", seed)
            old_prob = saved.final_probabilities(old_job)
            oz = saved.store.array("inference/" + old_job["id"] + "/logits.npy")
            job = next(x for x in protocol["jobs"] if x["seed"] == seed)
            nz = np.load(
                io.BytesIO(raw("inference/" + job["id"] + "/logits.npy")),
                allow_pickle=False,
            )
            r.require(
                nz.shape == oz.shape == (19082, 7) and np.isfinite(nz).all(),
                "Invalid paired logits.",
            )
            new_prob = expit(nz.astype(np.float64))
            r.require(
                np.allclose(expit(oz.astype(np.float64)), old_prob, rtol=0, atol=1e-15),
                "Old probabilities differ.",
            )
            old_logits.append(oz)
            new_logits.append(nz)
            for policy in r.POLICIES:
                name = f"predictions/{POOL}/tuned_{policy}_s{seed}.npz"
                before = saved.predictions("tuned", policy, seed)
                after = r.arrays_from_bytes(raw(name))
                r.require(
                    saved.store.expected(name) == protocol["preserved_sources"][name],
                    "Original paired binding differs.",
                )
                arrays_checked += compare_arrays(
                    before, after, saved.labels, study.inputs
                )
                r.require(
                    np.array_equal(after["hate_type_ungated"], new_prob >= 0.5),
                    "New type threshold differs.",
                )
                pairs[policy]["old"].append(before)
                pairs[policy]["new"].append(after)
        save(run / "archive_bindings.json", saved.store.read_bindings)
    save(
        run / "source_bindings.json",
        study.bindings | linear.bindings | context.bindings,
    )
    save(run / "training_scope.json", training)
    save(
        run / "reader_review.json",
        dict(
            status="pass",
            unchanged_other_arrays=arrays_checked,
            seeds=list(r.SEEDS),
            settings=10,
            prediction_rows=19082,
            archive_sha256=r.sha(EXECUTION / "execution_review.zip"),
        ),
    )
    m.array(
        run / "paired_type_logits.npz",
        old=np.stack(old_logits),
        new=np.stack(new_logits),
    )
    m.array(
        run / "paired_full_predictions.npz",
        **{
            f"{policy}__{version}__{key}": np.stack([x[key] for x in values])
            for policy, versions in pairs.items()
            for version, values in versions.items()
            for key in values[0]
        },
    )
    return study, context, pairs, np.stack(old_logits), np.stack(new_logits), training


def main_comparison(run, context, pairs):
    meta = r.read(MAIN / "en_test__hate_type_metadata.json")
    old_result = r.read(MAIN / "en_test__hate_type_results.json")
    with np.load(MAIN / "en_test__hate_type_predictions.npz", allow_pickle=False) as z:
        y, old_matrix = z["truth"], z["predictions"]
    with np.load(MAIN / "en_test__hate_type_point_counts.npz", allow_pickle=False) as z:
        old_points = z["scores"]
    with np.load(MAIN / "en_test__hate_type_bootstrap.npz", allow_pickle=False) as z:
        old_draws = z["scores"]
    view = context.views["broad", "en_test/hate_type"]
    r.require(
        meta["positions"] == view["positions"]
        and np.array_equal(y, view["truth"])
        and meta["clusters"] == view["clusters"],
        "Original TEST cache alignment differs.",
    )
    indices = [
        next(
            i
            for i, row in enumerate(meta["models"])
            if row["pool"] == POOL and row["method"] == METHOD and row["seed"] == seed
        )
        for seed in r.SEEDS
    ]
    old = np.stack(
        [
            context.aligned(x, "broad", "en_test/hate_type")
            for x in pairs["selected"]["old"]
        ]
    )
    new = np.stack(
        [
            context.aligned(x, "broad", "en_test/hate_type")
            for x in pairs["selected"]["new"]
        ]
    )
    r.require(
        np.array_equal(old, old_matrix[indices]),
        "Original Part 2 prediction pairing differs.",
    )
    new_points, new_counts = m.point_scores(y, new, meta["supported_indices"])
    new_draws, bootstrap = m.bootstrap(
        y,
        new,
        meta["clusters"],
        meta["supported_indices"],
        progress=lambda n: print(
            "Supplement paired bootstrap", n, "/10000", flush=True
        ),
    )
    r.require(
        bootstrap["picks_sha256"] == old_result["bootstrap"]["picks_sha256"],
        "Original bootstrap stream not reproduced.",
    )
    replace_points, replace_draws = old_points.copy(), old_draws.copy()
    replace_points[indices] = new_points
    replace_draws[:, indices] = new_draws
    old_pool, old_budget = m.aggregation(meta["models"], old_points)
    new_pool, new_budget = m.aggregation(meta["models"], replace_points)
    old_pd, old_bd = m.aggregation(meta["models"], old_draws)
    new_pd, new_bd = m.aggregation(meta["models"], replace_draws)
    delta_pool = new_points.mean() - old_points[indices].mean()
    delta_budget = new_budget[147, METHOD] - old_budget[147, METHOD]
    r.require(abs(delta_budget - delta_pool / 3) < 1e-15, "Wrong three-pool dilution.")
    new_comparisons = []
    for row in r.read(MAIN / "primary_comparisons.json")["comparisons"]:
        updated = dict(row)
        changed = (
            row["endpoint"] == "en_test/hate_type"
            and row["budget"] == 147
            and row["contrast"].startswith("tuned_")
        )
        if changed:
            diff = new_bd[147, METHOD] - new_bd[147, m.METHODS[1]]
            ci = interval(diff, 0.05 / 24)
            updated.update(
                left=float(new_budget[147, METHOD]),
                difference=float(
                    new_budget[147, METHOD] - new_budget[147, m.METHODS[1]]
                ),
                family_interval=ci,
                ci95_descriptive=interval(diff),
                excludes_zero=ci[0] > 0 or ci[1] < 0,
                pool_differences=[
                    float(new_pool[p, METHOD] - new_pool[p, m.METHODS[1]])
                    for p in r.POOLS[:3]
                ],
            )
        new_comparisons.append(
            dict(original=row, supplementary=updated, affected=changed)
        )
    r.require(
        sum(x["affected"] for x in new_comparisons) == 1,
        "More than one primary contrast affected.",
    )
    per_seed = [
        dict(
            seed=seed,
            original=float(old_points[i]),
            updated=float(new_points[j]),
            difference=float(new_points[j] - old_points[i]),
        )
        for j, (seed, i) in enumerate(zip(r.SEEDS, indices))
    ]
    comparisons = []
    for method in m.METHODS[1:]:
        comparisons.append(
            dict(
                comparator=method,
                original=float(old_budget[147, METHOD] - old_budget[147, method]),
                updated=float(new_budget[147, METHOD] - old_budget[147, method]),
                updated_ci95=interval(new_bd[147, METHOD] - old_bd[147, method]),
            )
        )
    result = dict(
        rows=len(y),
        labels=meta["labels"],
        supported_indices=meta["supported_indices"],
        seeds=per_seed,
        changed_pool=dict(
            original=float(old_points[indices].mean()),
            updated=float(new_points.mean()),
            original_range=[
                float(old_points[indices].min()),
                float(old_points[indices].max()),
            ],
            updated_range=[float(new_points.min()), float(new_points.max())],
            original_sd=float(old_points[indices].std(ddof=1)),
            updated_sd=float(new_points.std(ddof=1)),
            change=float(delta_pool),
            change_ci95=interval(new_draws.mean(1) - old_draws[:, indices].mean(1)),
        ),
        three_pool=dict(
            original=float(old_budget[147, METHOD]),
            updated=float(new_budget[147, METHOD]),
            change=float(delta_budget),
            change_ci95=interval(new_bd[147, METHOD] - old_bd[147, METHOD]),
            updated_absolute_ci95=interval(new_bd[147, METHOD]),
            BGE=float(old_budget[147, m.METHODS[1]]),
            frozen_RoBERTa=float(old_budget[147, m.METHODS[2]]),
        ),
        per_pool=[
            dict(
                pool=p,
                original=float(old_pool[p, METHOD]),
                updated=float(new_pool[p, METHOD]),
                BGE=float(old_pool[p, m.METHODS[1]]),
                frozen_RoBERTa=float(old_pool[p, m.METHODS[2]]),
            )
            for p in r.POOLS[:3]
        ],
        comparators=comparisons,
        bootstrap=bootstrap,
        original_model_indices=indices,
        original_comparisons_retained=True,
        all_24_comparisons=new_comparisons,
    )
    save(run / "main_comparison.json", result)
    m.array(
        run / "primary_paired.npz",
        truth=y,
        old=old,
        new=new,
        old_scores=old_points[indices],
        new_scores=new_points,
        new_draws=new_draws,
        old_draws=old_draws[:, indices],
        **{"new_" + k: v for k, v in new_counts.items()},
    )
    save(
        run / "primary_metadata.json",
        dict(
            labels=meta["labels"],
            ids=meta["ids"],
            clusters=meta["clusters"],
            positions=meta["positions"],
            records=[
                {
                    k: x.get(k)
                    for k in (
                        "id",
                        "text",
                        "source",
                        "gate",
                        "hate_types",
                        "target_groups",
                        "severity",
                    )
                }
                for x in view["records"]
            ],
        ),
    )
    return result


def diagnostic_views(run, context, pairs):
    reports, primary_arrays = {}, {}
    for scope, endpoint in TYPE_ENDPOINTS:
        view = context.views[scope, endpoint]
        y = view["truth"]
        reports[scope + "/" + endpoint] = {}
        for policy in r.POLICIES:
            gates = np.stack(
                [x["hate"][view["positions"], 1] for x in pairs[policy]["old"]]
            ).astype(bool)
            for mode in ("gated", "ungated"):
                pp = {
                    v: np.stack(
                        [
                            context.aligned(
                                x, scope, endpoint, ungated=mode == "ungated"
                            )
                            for x in pairs[policy][v]
                        ]
                    )
                    for v in ("old", "new")
                }
                qq = {v: quality(y, pp[v], gates) for v in pp}
                tt = transitions(y, pp["old"], pp["new"])
                reports[scope + "/" + endpoint][policy + "/" + mode] = dict(
                    rows=len(y),
                    gold_support=y.sum(0),
                    labels=view["endpoint"]["all_labels"],
                    old=qq["old"],
                    new=qq["new"],
                    changes=tt,
                )
                if scope == "broad":
                    for version in pp:
                        primary_arrays[policy + "__" + mode + "__" + version] = pp[
                            version
                        ]
                    primary_arrays[policy + "__gate"] = gates
        if scope == "broad":
            g = primary_arrays["selected__gate"]
            before = primary_arrays["selected__gated__old"]
            after = primary_arrays["selected__gated__new"]
            old_exact, new_exact = (before == y).all(2), (after == y).all(2)
            row_changes = np.stack(
                [
                    (new_exact.astype(float) - old_exact).mean(0),
                    ((new_exact & g).astype(float) - (old_exact & g)).mean(0),
                    ((after != y).mean(2) - (before != y).mean(2)).mean(0),
                ]
            )
            draws = row_change_draws(row_changes, view["clusters"])
            save(
                run / "quality_change_intervals.json",
                dict(
                    names=["exact_set", "hate_and_type_exact", "hamming_error"],
                    point_changes=row_changes.mean(1),
                    descriptive_ci95=[interval(draws[:, i]) for i in range(3)],
                    conditional_on_fixed_models=True,
                ),
            )
            m.array(
                run / "quality_change_bootstrap.npz",
                row_changes=row_changes,
                draws=draws,
            )
    save(run / "endpoint_diagnostics.json", reports)
    m.array(run / "diagnostic_primary_arrays.npz", **primary_arrays)
    return reports


def changes_and_cases(run, study, context, pairs, old_z, new_z):
    old_prob, new_prob = expit(old_z.astype(np.float64)), expit(
        new_z.astype(np.float64)
    )
    probability_shift = new_prob - old_prob
    drift = []
    primary = context.views["broad", "en_test/hate_type"]
    primary_mask = np.zeros(len(study.inputs), bool)
    primary_mask[primary["positions"]] = True
    for j, seed in enumerate(r.SEEDS):
        raw_changed = (new_prob[j] >= 0.5) != (old_prob[j] >= 0.5)
        policy = {}
        for name in r.POLICIES:
            a, b = pairs[name]["old"][j], pairs[name]["new"][j]
            changed = a["hate_type"] != b["hate_type"]
            suppressed = raw_changed & (a["hate"][:, 1] == 0)[:, None]
            r.require(
                np.array_equal(changed | suppressed, raw_changed),
                "Gate change accounting differs.",
            )
            policy[name] = dict(
                gated_changed_labels=int(changed.sum()),
                gated_changed_texts=int(changed.any(1).sum()),
                suppressed_changed_labels=int(suppressed.sum()),
                changed_labels_in_primary=int(changed[primary_mask].sum()),
                changes_outside_primary=int(changed[~primary_mask].sum()),
            )
        drift.append(
            dict(
                seed=seed,
                raw_changed_labels=int(raw_changed.sum()),
                raw_changed_texts=int(raw_changed.any(1).sum()),
                mean_abs_logit_shift=float(abs(new_z[j] - old_z[j]).mean()),
                max_abs_logit_shift=float(abs(new_z[j] - old_z[j]).max()),
                mean_abs_probability_shift=float(abs(probability_shift[j]).mean()),
                max_abs_probability_shift=float(abs(probability_shift[j]).max()),
                per_label_mean_probability_shift=probability_shift[j].mean(0),
                old_probability_range=[
                    float(old_prob[j].min()),
                    float(old_prob[j].max()),
                ],
                new_probability_range=[
                    float(new_prob[j].min()),
                    float(new_prob[j].max()),
                ],
                policies=policy,
            )
        )
    save(
        run / "prediction_changes.json",
        dict(
            all_input_rows=len(study.inputs),
            labels=context.labels["hate_type"],
            per_seed=drift,
        ),
    )
    records, source_rows = [], []
    for j, seed in enumerate(r.SEEDS):
        before = context.aligned(
            pairs["selected"]["old"][j], "broad", "en_test/hate_type"
        ).astype(bool)
        after = context.aligned(
            pairs["selected"]["new"][j], "broad", "en_test/hate_type"
        ).astype(bool)
        for i in np.flatnonzero((before != after).any(1)):
            y = primary["truth"][i]
            old_exact, new_exact = bool((before[i] == y).all()), bool(
                (after[i] == y).all()
            )
            category = (
                "exact_fix"
                if new_exact and not old_exact
                else "exact_harm" if old_exact and not new_exact else "still_inexact"
            )
            fixed = (before[i] != y) & (after[i] == y)
            harm = (before[i] == y) & (after[i] != y)
            position = primary["positions"][i]
            record = primary["records"][i]
            labels = primary["endpoint"]["all_labels"]
            names = lambda mask: [label for label, v in zip(labels, mask) if v]
            records.append(
                dict(
                    seed=seed,
                    id=str(record["id"]),
                    source=record["source"],
                    text=record["text"],
                    category=category,
                    gold=names(y),
                    old=names(before[i]),
                    new=names(after[i]),
                    fixed_labels=names(fixed),
                    harmed_labels=names(harm),
                    old_probabilities=old_prob[j, position],
                    new_probabilities=new_prob[j, position],
                    old_label_errors=int((before[i] != y).sum()),
                    new_label_errors=int((after[i] != y).sum()),
                )
            )
    records.sort(key=lambda x: (x["id"], x["seed"]))
    with (run / "changed_cases.jsonl").open("xb") as f:
        for row in records:
            f.write(
                (
                    json.dumps(
                        native(row), sort_keys=True, ensure_ascii=False, allow_nan=False
                    )
                    + "\n"
                ).encode("utf8")
            )
    examples = []
    for category in ("exact_fix", "exact_harm", "still_inexact"):
        selected = [x for x in records if x["category"] == category]
        if selected:
            examples.append(selected[0])
    for tag, predicate in [
        (
            "label_error_reduction",
            lambda x: x["new_label_errors"] < x["old_label_errors"],
        ),
        (
            "label_error_increase",
            lambda x: x["new_label_errors"] > x["old_label_errors"],
        ),
        (
            "mixed_changes",
            lambda x: bool(x["fixed_labels"]) and bool(x["harmed_labels"]),
        ),
    ]:
        selected = [x for x in records if predicate(x)]
        if selected:
            examples.append(dict(selected[0], sampling_category=tag))
    save(
        run / "representative_cases.json",
        dict(
            rule="First changed id then seed within each predeclared category; examples are descriptive.",
            total_changed_seed_text_pairs=len(records),
            examples=examples,
        ),
    )
    before = np.stack(
        [
            context.aligned(x, "broad", "en_test/hate_type")
            for x in pairs["selected"]["old"]
        ]
    )
    after = np.stack(
        [
            context.aligned(x, "broad", "en_test/hate_type")
            for x in pairs["selected"]["new"]
        ]
    )
    gates = np.stack(
        [x["hate"][primary["positions"], 1] for x in pairs["selected"]["old"]]
    )
    for source in sorted({x["source"] for x in primary["records"]}):
        ix = [i for i, x in enumerate(primary["records"]) if x["source"] == source]
        source_rows.append(
            dict(
                source=source,
                rows=len(ix),
                old=quality(primary["truth"][ix], before[:, ix], gates[:, ix]),
                new=quality(primary["truth"][ix], after[:, ix], gates[:, ix]),
                changes=transitions(primary["truth"][ix], before[:, ix], after[:, ix]),
            )
        )
    save(run / "source_diagnostics.json", source_rows)


def full_answer(run, study, context, pairs):
    rows = context.records["broad", "en_test/hate"]
    heads = ("hate_type", "target_group", "severity")
    kept = [
        x
        for x in rows
        if x["gate"] is True and all(r.gold_labels(x, h) is not None for h in heads)
    ]
    result = dict(
        population="English TEST gold-hate with all three fine annotations observed",
        rows=len(kept),
        all_English_TEST_rows=len(rows),
        not_a_whole_TEST_accuracy=True,
        includes_legal_head=False,
        ids=[str(x["id"]) for x in kept],
    )
    if kept:
        pos = [context.positions["broad"][str(x["id"])] for x in kept]
        truths = {
            h: np.array(
                [
                    [label in r.gold_labels(row, h) for label in context.labels[h]]
                    for row in kept
                ],
                bool,
            )
            for h in heads
        }
        results = {}
        for policy in r.POLICIES:
            exact = {}
            for version in ("old", "new"):
                exact[version] = np.stack(
                    [
                        values["hate"][pos, 1].astype(bool)
                        & np.logical_and.reduce(
                            [(values[h][pos] == truths[h]).all(1) for h in heads]
                        )
                        for values in pairs[policy][version]
                    ]
                )
            results[policy] = dict(
                old_count=exact["old"].sum(1),
                new_count=exact["new"].sum(1),
                fixes=(~exact["old"] & exact["new"]).sum(1),
                harms=(exact["old"] & ~exact["new"]).sum(1),
            )
        result["results"] = results
    else:
        result["result"] = None
        result["reason"] = (
            "No fully annotated four-head TEST rows; type exactness cannot establish a complete four-head answer gain."
        )
    save(run / "four_head_exactness.json", result)


def main():
    run = BASE / "runs" / uuid.uuid4().hex[:12]
    run.mkdir(parents=True)
    started = time.perf_counter()
    sources = [
        Path(__file__),
        BASE / "analysis_contract.json",
        Path(m.__file__),
        Path(r.__file__),
    ]
    save(
        run / "started.json",
        dict(
            utc=datetime.now(timezone.utc).isoformat(),
            sources={str(p): r.sha(p) for p in sources},
            no_new_training=True,
            no_Qwen=True,
            no_weight_loading=True,
        ),
    )
    try:
        with r.offline(), threadpool_limits(limits=2):
            study, context, pairs, old_z, new_z, training = read_sources(run)
            print(
                "Affected reader verified; original gates and all other arrays unchanged.",
                flush=True,
            )
            result = main_comparison(run, context, pairs)
            diagnostic_views(run, context, pairs)
            changes_and_cases(run, study, context, pairs, old_z, new_z)
            full_answer(run, study, context, pairs)
            for p in sources:
                r.require(
                    r.sha(p) == r.read(run / "started.json")["sources"][str(p)],
                    "Analysis source changed.",
                )
            r.require("torch" not in sys.modules, "Unexpected model runtime import.")
            outputs = {
                str(p.relative_to(run)): r.sha(p) for p in run.iterdir() if p.is_file()
            }
            save(
                run / "complete.json",
                dict(
                    status="supplementary_analysis_complete",
                    utc=datetime.now(timezone.utc).isoformat(),
                    seconds=time.perf_counter() - started,
                    outputs=outputs,
                    independent_review_pending=True,
                    training_calls=0,
                    Qwen_calls=0,
                    server_actions=0,
                    model_weights_loaded=0,
                    original_results_changed=False,
                ),
            )
            print(
                "SUPPLEMENT ANALYSIS",
                str(run),
                native(result["changed_pool"]),
                native(result["three_pool"]),
                flush=True,
            )
    except BaseException as exc:
        save(
            run / "stopped.json",
            dict(
                type=type(exc).__name__,
                message=str(exc),
                utc=datetime.now(timezone.utc).isoformat(),
            ),
        )
        raise


if __name__ == "__main__":
    main()

"""German guidelines hate."""

from analysis.knowledge.german_guidelines import reader as reader
from analysis.knowledge.german_guidelines import common as c
from collections import Counter, defaultdict
from contextlib import contextmanager
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
from unittest.mock import patch
import numpy as np

BASE = c.ROOT / "experiments/astra/german_guidelines_hate_analysis_20261112_v1"
PHASES = ("de_test", "de_dev_eval")
CONDITIONS = tuple(c.CONDITIONS)
PAIRS = (
    ("general_2", "mixed_2"),
    ("general_2_legal_1", "mixed_2"),
    ("general_2_legal_1", "general_2"),
)
DRAWS = 10000
SEEDS = {"de_test": 2026111202, "de_dev_eval": 2026111203}
BOOT_METRICS = (
    "macro_f1",
    "f1_hate",
    "f1_not_hate",
    "precision",
    "recall",
    "fpr",
    "fnr",
    "accuracy",
    "error_rate",
    "predicted_hate",
    "balanced_accuracy",
    "fp",
    "fn",
    "total_errors",
)
ERROR_METRICS = {"fp", "fn", "total_errors", "fpr", "fnr", "error_rate"}
BOUND_METRICS = ("macro_f1", "accuracy", "fp", "fn", "total_errors", "fpr", "fnr", "error_rate")
EVENTS = (
    "fp_fixed",
    "fp_introduced",
    "fn_fixed",
    "fn_introduced",
    "fp_persistent",
    "fn_persistent",
    "both_correct_negative",
    "both_correct_positive",
)
require = reader.require


@contextmanager
def offline():
    with c.offline(), patch.object(c.a.RequestsTransport, "__call__", c.forbidden):
        yield


def safe(value):
    if isinstance(value, dict):
        return {str(k): safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def save(path, value):
    require(c.normal(path).is_relative_to(BASE), "Part 2 write outside new analysis folder")
    c.save(path, safe(value))


def arrays(path, **values):
    require(c.normal(path).is_relative_to(BASE), "Part 2 arrays outside new analysis folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        np.savez_compressed(handle, **values)


def divide(a, b, zero=np.nan):
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.full(a.shape, zero, dtype=float), where=b != 0)


def contributions(y, pred):
    y, pred = (np.asarray(y), np.asarray(pred))
    require(y.ndim == 1 and pred.ndim >= 1 and (pred.shape[-1] == len(y)), "Wrong binary axes")
    require(
        np.isin(y, [0, 1]).all() and np.isin(pred, [0, 1]).all(),
        "Nonbinary or missing value passed to confusion counts",
    )
    y, pred = (y.astype(bool), pred.astype(bool))
    return np.stack((~y & ~pred, ~y & pred, y & ~pred, y & pred), axis=-1).astype(np.int64)


def metrics(counts):
    z = np.asarray(counts)
    require(z.shape[-1] == 4 and (z >= 0).all(), "Bad TN/FP/FN/TP counts")
    tn, fp, fn, tp = np.moveaxis(z, -1, 0)
    n = z.sum(axis=-1)
    positive = divide(2 * tp, 2 * tp + fp + fn, 0.0)
    negative = divide(2 * tn, 2 * tn + fp + fn, 0.0)
    recall, specificity = (divide(tp, tp + fn), divide(tn, tn + fp))
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
        f1_hate=positive,
        f1_not_hate=negative,
        macro_f1=np.where(n > 0, (positive + negative) / 2, np.nan),
        precision=divide(tp, tp + fp),
        recall=recall,
        specificity=specificity,
        precision_not_hate=divide(tn, tn + fn),
        fpr=divide(fp, fp + tn),
        fnr=divide(fn, fn + tp),
        accuracy=divide(tp + tn, n),
        error_rate=divide(fp + fn, n),
        predicted_hate=divide(tp + fp, n),
        gold_hate=divide(tp + fn, n),
        balanced_accuracy=(recall + specificity) / 2,
    )


def summarize(counts):
    require(np.asarray(counts).shape == (3, 4), "Three outer-repeat confusion matrices required")
    values = metrics(counts)
    return safe(
        {
            "repeats": [{name: value[i] for name, value in values.items()} for i in range(3)],
            "mean": {name: value.mean() for name, value in values.items()},
            "sample_sd": {name: value.std(ddof=1) for name, value in values.items()},
            "range": {name: [value.min(), value.max()] for name, value in values.items()},
        }
    )


def failure_bounds(y, pred):
    y, pred = (np.asarray(y), np.asarray(pred))
    require(
        pred.shape == (3, len(y)) and np.isin(pred, [-1, 0, 1]).all(), "Bad missing-answer matrix"
    )
    valid = pred >= 0
    worst = np.where(valid, pred, 1 - y)
    best = np.where(valid, pred, y)
    lo = metrics(contributions(y, worst).sum(axis=-2))
    hi = metrics(contributions(y, best).sum(axis=-2))
    bounds = {}
    for name in BOUND_METRICS:
        lower, upper = (hi[name], lo[name]) if name in ERROR_METRICS else (lo[name], hi[name])
        require(
            np.all((lower <= upper) | ~np.isfinite(lower) | ~np.isfinite(upper)),
            "Reversed missing-answer bound",
        )
        bounds[name] = {
            "by_repeat": np.stack((lower, upper), axis=-1),
            "mean": [lower.mean(), upper.mean()],
        }
    return safe(
        {
            "unknown_by_repeat": (~valid).sum(axis=1),
            "unknown_negative_by_repeat": (~valid & (y == 0)).sum(axis=1),
            "unknown_positive_by_repeat": (~valid & (y == 1)).sum(axis=1),
            "bounds": bounds,
            "meaning": "Full-roster extremes for missing predictions; not confidence intervals or imputed answers.",
        }
    )


def transition_summary(y, left, right):
    y = np.asarray(y, dtype=bool)
    require(left.shape == right.shape == (3, len(y)), "Wrong transition axes")
    require(
        np.isin(left, [0, 1]).all() and np.isin(right, [0, 1]).all(),
        "Transitions require the common-valid mask",
    )
    a, b = (left.astype(bool)[:, None, :], right.astype(bool)[None, :, :])
    events = np.stack(
        (
            ~a & b & ~y,
            a & ~b & ~y,
            a & ~b & y,
            ~a & b & y,
            a & b & ~y,
            ~a & ~b & y,
            ~a & ~b & ~y,
            a & b & y,
        ),
        axis=-1,
    )
    require(
        events.shape == (3, 3, len(y), 8) and (events.sum(axis=-1) == 1).all(),
        "Transitions do not partition inputs",
    )
    counts = events.sum(axis=2)
    return safe(
        {
            "all_nine_pair_counts": counts,
            "event_order": EVENTS,
            "mean_events": {name: counts[:, :, i].mean() for i, name in enumerate(EVENTS)},
            "same_repeat_events": [
                {name: counts[j, j, i] for i, name in enumerate(EVENTS)} for j in range(3)
            ],
            "pairing": "All nine left/right outer-repeat pairings, equally weighted. No shared random seed or temporal correction is implied; nine pairs are not independent texts.",
        }
    )


def bootstrap(y, pred, clusters, full_memberships, seed, draws=DRAWS):
    require(pred.shape == (3, 3, len(y)) and len(clusters) == len(y), "Bootstrap matrix differs")
    cells = pred.reshape(9, len(y))
    item = contributions(y, cells)
    unique = sorted(set(clusters))
    lookup = {name: i for i, name in enumerate(unique)}
    grouped = np.zeros((9, len(unique), 4), dtype=np.int64)
    for i, name in enumerate(clusters):
        grouped[:, lookup[name]] += item[:, i]
    strata = defaultdict(list)
    for i, name in enumerate(unique):
        strata[tuple(sorted(full_memberships[name]))].append(i)
    ordered = [group for _, group in sorted(strata.items())]
    rng = np.random.default_rng(seed)
    totals = np.empty((draws, 9, 4), dtype=np.int64)
    digest = hashlib.sha256()
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
            "Direct bootstrap count check failed",
        )
        checks += 1
        totals[start : start + size] = block
    design = {
        "draws": draws,
        "seed": seed,
        "block_size": 64,
        "clusters": unique,
        "cluster_rows": [
            [i for i, name in enumerate(clusters) if name == cluster] for cluster in unique
        ],
        "strata": [
            {"sources": list(key), "cluster_indices": group}
            for key, group in sorted(strata.items())
        ],
        "pick_stream_sha256": digest.hexdigest(),
        "direct_block_checks": checks,
        "unit": "Whole normalized-text clusters; full-roster source-membership strata; one shared draw for all nine prediction settings.",
    }
    return (totals.reshape(draws, 3, 3, 4), design)


def interval(values, family=None):
    values = np.asarray(values)
    invalid = int((~np.isfinite(values)).sum())
    return safe(
        {
            "ci95": np.quantile(values, [0.025, 0.975]) if not invalid else None,
            "family_interval": (
                np.quantile(values, [0.05 / (2 * family), 1 - 0.05 / (2 * family)])
                if family and (not invalid)
                else None
            ),
            "family_size": family,
            "family_confidence": 1 - 0.05 / family if family else None,
            "undefined_draws": invalid,
        }
    )


def own_sources():
    return [
        Path(__file__),
        c.ROOT / "tests/knowledge/german_guidelines/test_hate.py",
        c.ROOT / "analysis/knowledge/german_guidelines/verify_hate.py",
    ]


def dependencies():
    return {name: importlib.metadata.version(name) for name in ("numpy", "scikit-learn")}


def load():
    c.validate()
    c.verify(c.read(reader.BASE / "analysis_contract.json")["source_bindings"])
    manifest, path = reader.read_verified()
    endpoints = c.read(path / "endpoint_setup.json")
    data = {}
    for phase in PHASES:
        ep = endpoints[phase]
        require(
            ep["dimension"] == "hate"
            and ep["all_labels"] == ep["macro_labels"] == ["not_hate", "hate"],
            "Frozen binary labels changed",
        )
        require(all((type(x) is bool for x in ep["gold"])), "Nonbinary gold")
        keys = ep["keys"]
        data[phase] = dict(
            endpoint=ep,
            index={key: i for i, key in enumerate(keys)},
            gold=np.asarray(ep["gold"], dtype=np.int8),
            pred=np.full((3, 3, len(keys)), -9, dtype=np.int8),
        )
    count = 0
    for row in c.rows(path / "predictions.jsonl"):
        count += 1
        if row["phase"] not in data:
            continue
        d = data[row["phase"]]
        ci, repeat, i = (CONDITIONS.index(row["condition"]), row["repeat"], d["index"][row["key"]])
        require(d["pred"][ci, repeat, i] == -9, "Duplicate hate prediction")
        d["pred"][ci, repeat, i] = -1 if row["result"] is None else int(row["result"]["hate"])
    require(count == 33093, "Incomplete verified cache")
    for phase, d in data.items():
        require(np.isin(d["pred"], [-1, 0, 1]).all(), "Missing cached hate slot")
        d["common"] = (d["pred"] >= 0).all(axis=(0, 1))
        require(
            [key for key, on in zip(d["endpoint"]["keys"], d["common"]) if on]
            == d["endpoint"]["common_valid_keys"],
            "Part 1 common mask changed",
        )
    return (data, path)


def prepare():
    c.validate()
    _, path = reader.read_verified()
    tests = sorted((BASE / "tests").glob("tests_*/report.json"))
    require(tests and c.read(tests[-1])["passed"], "Passing tests required")
    require(
        c.read(tests[-1])["sources"] == c.bindings(own_sources()), "Tests do not cover current code"
    )
    save(
        BASE / "analysis_contract.json",
        {
            "status": "part2_frozen_before_effectiveness_calculation",
            "created_utc": c.a.utc(),
            "inputs": c.bindings(
                [
                    c.PLAN,
                    c.PROTOCOL,
                    reader.BASE / "complete.json",
                    reader.ANALYSIS_PLAN,
                    reader.BASE / "analysis_contract.json",
                    path / "manifest.json",
                    tests[-1],
                    *own_sources(),
                ]
            ),
            "phases": PHASES,
            "conditions": CONDITIONS,
            "pairs": PAIRS,
            "confusion_order": ["tn", "fp", "fn", "tp"],
            "analysis_dependencies": dependencies(),
            "mask": "The same Part 1 all-three-conditions/all-three-repeats valid mask for every main comparison.",
            "point": "Calculate every repeat separately and average its metrics; no pooled predictions or majority across outer repeats.",
            "uncertainty": {
                "draws": DRAWS,
                "seeds": SEEDS,
                "block_size": 64,
                "method": "Source-stratified whole normalized-text clusters, full-roster memberships, shared draws across all nine settings.",
                "test_family_size": 4,
                "family_role": "98.75% intervals only for the two TEST baseline macro-F1 contrasts in this part; two legal contrasts remain for Part 3.",
                "diagnostics": "All other intervals, including FP/FN and the added-legal-rule contrast, are descriptive 95%. Undefined draws are counted and suppress an interval; never silently dropped.",
                "limits": "Conditional on exposed cohort, fixed prompts and these three decoder repeats; not fresh-test confirmation, equivalence testing or all model/retrieval/selection uncertainty.",
            },
            "coverage": "Own-valid repeat scores plus full-roster exact binary best/worst missing-answer bounds; missing predictions stay -1.",
            "transitions": "FP/FN fixed/introduced/persistent and both-correct partition for all nine cross-repeat pairings, equally weighted. Same-repeat counts also shown. No per-text stability interpretation or case selection yet.",
            "deferred": [
                "legal effectiveness Part 3",
                "guideline context/mechanism Part 4",
                "source-level effects, cases and individual repeat stability Part 5",
                "declared exclusion and other robustness checks Part 6",
                "resources and synthesis Part 7",
                "writer handover Part 8",
            ],
            "preservation": "New derived outputs only. No Qwen calls, training, retrieval, D: writes, original source edits or thesis edits.",
        },
    )
    print("Part 2 analysis contract frozen; no effectiveness scores calculated yet.", flush=True)


def run():
    contract = c.read(BASE / "analysis_contract.json")
    c.verify(contract["inputs"])
    require(
        contract["analysis_dependencies"] == dependencies(), "Part 2 analysis dependencies changed"
    )
    data, cache = load()
    out = BASE / "runs" / f"run_{time.time_ns()}"
    out.mkdir(parents=True)
    save(
        out / "start.json",
        {"created_utc": c.a.utc(), "contract_sha256": c.sha(BASE / "analysis_contract.json")},
    )
    results = {}
    for phase in PHASES:
        d = data[phase]
        ep, y, p, mask = (d["endpoint"], d["gold"], d["pred"], d["common"])
        yy, pp = (y[mask], p[:, :, mask])
        keys = [key for key, on in zip(ep["keys"], mask) if on]
        clusters = [cl for cl, on in zip(ep["clusters"], mask) if on]
        counts = contributions(yy, pp).sum(axis=-2)
        shared, own, bounds = ({}, {}, {})
        for ci, cond in enumerate(CONDITIONS):
            shared[cond] = summarize(counts[ci])
            own_counts = np.stack(
                [
                    contributions(y[p[ci, rep] >= 0], p[ci, rep, p[ci, rep] >= 0]).sum(axis=-2)
                    for rep in range(3)
                ]
            )
            own[cond] = summarize(own_counts)
            bounds[cond] = failure_bounds(y, p[ci])
        boot, design = bootstrap(
            yy, pp, clusters, ep["source_memberships_by_cluster"], SEEDS[phase]
        )
        bm = {name: values.mean(axis=2) for name, values in metrics(boot).items()}
        condition_ci = {
            cond: {name: interval(bm[name][:, ci]) for name in BOOT_METRICS}
            for ci, cond in enumerate(CONDITIONS)
        }
        contrasts = []
        for left, right in PAIRS:
            li, ri = (CONDITIONS.index(left), CONDITIONS.index(right))
            main = phase == "de_test" and right == "mixed_2"
            diagnostic = {}
            for name in BOOT_METRICS:
                diagnostic[name] = interval(
                    bm[name][:, li] - bm[name][:, ri], 4 if main and name == "macro_f1" else None
                )
                diagnostic[name].update(
                    difference=shared[left]["mean"][name] - shared[right]["mean"][name],
                    by_repeat=[
                        shared[left]["repeats"][i][name] - shared[right]["repeats"][i][name]
                        for i in range(3)
                    ],
                    direction=(
                        "Positive means more errors"
                        if name in ERROR_METRICS
                        else "Left minus right"
                    ),
                )
            flow = transition_summary(yy, pp[li], pp[ri])
            for field, fixed, added in [
                ("fp", "fp_fixed", "fp_introduced"),
                ("fn", "fn_fixed", "fn_introduced"),
            ]:
                require(
                    abs(
                        flow["mean_events"][added]
                        - flow["mean_events"][fixed]
                        - diagnostic[field]["difference"]
                    )
                    < 1e-10,
                    "Gross and net error changes differ",
                )
            bound_diffs = {
                name: [
                    bounds[left]["bounds"][name]["mean"][0]
                    - bounds[right]["bounds"][name]["mean"][1],
                    bounds[left]["bounds"][name]["mean"][1]
                    - bounds[right]["bounds"][name]["mean"][0],
                ]
                for name in BOUND_METRICS
            }
            contrasts.append(
                {
                    "left": left,
                    "right": right,
                    "role": (
                        "main TEST baseline contrast in four-comparison family"
                        if main
                        else "supporting descriptive contrast"
                    ),
                    "metrics": diagnostic,
                    "transitions": flow,
                    "full_roster_failure_difference_bounds": bound_diffs,
                    "own_valid_macro_difference": own[left]["mean"]["macro_f1"]
                    - own[right]["mean"]["macro_f1"],
                }
            )
        result = {
            "population": phase,
            "full_inputs": len(y),
            "common_inputs": len(yy),
            "common_keys": keys,
            "excluded_keys": ep["common_excluded_keys"],
            "full_gold_support": ep["gold_support"],
            "common_gold_support": {"not_hate": int((yy == 0).sum()), "hate": int(yy.sum())},
            "common_source_counts": dict(Counter((s for s, on in zip(ep["sources"], mask) if on))),
            "common": shared,
            "own_valid": own,
            "full_roster_bounds": bounds,
            "condition_descriptive_intervals": condition_ci,
            "contrasts": contrasts,
            "repeats_are_not_independent_texts": True,
        }
        save(out / phase / "results.json", result)
        save(
            out / phase / "input_roster.json",
            {
                "keys": ep["keys"],
                "sources": ep["sources"],
                "clusters": ep["clusters"],
                "conditions": CONDITIONS,
                "repeat_indices": [0, 1, 2],
                "confusion_order": ["tn", "fp", "fn", "tp"],
            },
        )
        save(out / phase / "bootstrap_design.json", design)
        arrays(
            out / phase / "arrays.npz",
            gold=y,
            predictions=p,
            common_mask=mask,
            confusion_counts=counts,
            bootstrap_confusions=boot,
        )
        results[phase] = result
        print(
            phase
            + ": three conditions scored, 10,000 paired cluster draws and all error-flow checks complete.",
            flush=True,
        )
    c.verify(contract["inputs"])
    c.verify(c.read(reader.BASE / "analysis_contract.json")["source_bindings"])
    c.validate()
    save(
        out / "source_and_preservation_check.json",
        {
            "status": "bound_inputs_and_original_preparation_unchanged",
            "hashes": contract["inputs"],
            "actual_Qwen_calls": 0,
            "D_writes": 0,
            "latex_edits": 0,
            "legal_effectiveness_computed": False,
            "raw_full_tree_reaudited": False,
        },
    )
    save(
        out / "manifest.json",
        {
            "status": "part2_scored_pending_independent_check",
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
    require(done["status"] == "part2_numerically_verified", "Independent Part 2 check missing")
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

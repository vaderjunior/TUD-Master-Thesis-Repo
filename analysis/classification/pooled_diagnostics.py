from src.hsrag.paths import PROJECT_ROOT
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import time
import traceback
import unicodedata
from unittest.mock import patch

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score

ROOT = PROJECT_ROOT
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
BASE = ROOT / "experiments/astra/pooled_fineheads_20260930_v1"
SOURCE = Path(__file__).resolve()
CONDITIONS = ["rag_full", "rag_small", "zero_shot", "encoder_full", "encoder_kb"]
LABELS = {
    "target_group": [
        "race",
        "religion",
        "gender",
        "sexual_orientation",
        "disability",
        "national_origin",
        "other",
    ],
    "severity": ["low", "medium", "high"],
}
FOLDERS = {
    "en_dev_eval_main": CORE / "primary_results_v1",
    "en_dev_eval_targets": CORE / "secondary_results_v2/en_dev_eval_targets",
    "en_dev_eval_severity": CORE / "severity_results_v1",
}
INPUTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bind(path, expected=None):
    path = Path(path)
    key = path.relative_to(ROOT).as_posix()
    actual = sha(path)
    assert expected is None or actual == expected, ("Hash differs", key)
    assert key not in INPUTS or INPUTS[key] == actual
    INPUTS[key] = actual
    return path


def read(path, expected=None):
    return json.loads(bind(path, expected).read_text(encoding="utf-8-sig"))


def lines(path, expected=None):
    with bind(path, expected).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def clean(value):
    if value is None or value is pd.NA or isinstance(value, float) and np.isnan(value):
        return None
    return value.tolist() if isinstance(value, np.ndarray) else value


def normalized(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def assert_inputs():
    for path, expected in INPUTS.items():
        assert sha(ROOT / path) == expected, ("Changed after read", path)


def prepare():
    assert not (BASE / "plan.json").exists(), "Preserve frozen plan"
    bind(SOURCE)
    previous = read(
        ROOT
        / "experiments/astra/supervisor_reverification_20260930_v1/attempts/1790801822389529500/coverage_and_reuse.json"
    )
    core = {
        row["item_id"]: row
        for row in lines(CORE / "dev_inputs.jsonl")
        if row["lang"] == "en"
    }
    assert len(core) == 970
    dev = pd.read_parquet(bind(ROOT / "data/processed/en_dev.parquet")).to_dict(
        "records"
    )
    assert len(dev) == len({str(row["id"]) for row in dev})
    endpoints = {}
    rosters = {}
    for dimension, field, count, subset in [
        ("target_group", "target_groups", 163, "en_dev_eval_targets"),
        ("severity", "severity", 116, "en_dev_eval_severity"),
    ]:
        receipt = read(FOLDERS[subset] / "complete.json")
        original = read(FOLDERS[subset] / "results.json", receipt["results_sha256"])[
            "endpoint"
        ]
        assert original["all_labels"] == original["macro_labels"] == LABELS[dimension]
        eligible = []
        for row in dev:
            iid = str(row["id"])
            value, gate = clean(row[field]), clean(row["gate"])
            if iid not in core or gate is not True or value is None:
                continue
            assert row["text"] == core[iid]["text"]
            gold = [value] if dimension == "severity" else value
            assert type(gold) is list and set(gold) <= set(LABELS[dimension])
            source = row.get("source", row.get("dataset"))
            if not isinstance(source, str) or not source:
                source = iid.split("-", 1)[0]
            origin = core[iid]["scopes"][0]["subset"]
            assert origin in FOLDERS
            eligible.append(
                {
                    "id": iid,
                    "key": core[iid]["key"],
                    "source": source,
                    "origin": origin,
                    "text_sha256": hashlib.sha256(
                        row["text"].encode("utf-8")
                    ).hexdigest(),
                    "cluster": hashlib.sha256(
                        normalized(row["text"]).encode("utf-8")
                    ).hexdigest(),
                    "gold": gold,
                    "original_roster": iid in original["scorable_ids"],
                }
            )
        eligible.sort(key=lambda row: row["id"])
        assert len(eligible) == count
        assert {row["id"] for row in eligible if row["original_roster"]} == set(
            original["scorable_ids"]
        )
        rosters[dimension] = eligible
        endpoints[dimension] = {
            "rows": count,
            "labels": LABELS[dimension],
            "original_subset": subset,
            "original_rows": len(original["scorable_ids"]),
            "original_ids": original["scorable_ids"],
            "gold_support": {
                label: sum(label in row["gold"] for row in eligible)
                for label in LABELS[dimension]
            },
            "by_source": dict(Counter(row["source"] for row in eligible)),
            "by_original_roster": dict(Counter(row["origin"] for row in eligible)),
            "clusters": len({row["cluster"] for row in eligible}),
        }
    exports = {}
    for subset, folder in FOLDERS.items():
        receipt = read(folder / "complete.json")

        read(folder / "results.json", receipt["results_sha256"])
        for condition in CONDITIONS[:3]:
            for repeat in range(3):
                path = (
                    folder
                    / "assembled"
                    / (subset if subset == "en_dev_eval_main" else "")
                    / f"{condition}_rep{repeat}.jsonl"
                )
                key = path.relative_to(ROOT).as_posix()
                bind(path, receipt["assembled"][key]["sha256"])
                exports[f"{subset}/{condition}/{repeat}"] = key
    bound_previous = read(
        ROOT
        / "experiments/astra/supervisor_reverification_20260930_v1/attempts/1790801822389529500/inputs.json"
    )
    for arm in ["full", "kb"]:
        for seed in range(3):
            path = CORE / f"encoder_results_v5/en_{arm}_seed{seed}_assembled.jsonl"
            key = path.relative_to(ROOT).as_posix()

            core_plan = read(CORE / "primary_results_v1/plan.json")
            bind(path, core_plan["hashes"][key])
            if key in bound_previous:
                assert INPUTS[key] == bound_previous[key]
            exports[f"encoder_{arm}/{seed}"] = key
    assert_inputs()
    BASE.mkdir(parents=True, exist_ok=True)
    (BASE / "prepared_source.py").write_bytes(SOURCE.read_bytes())
    save(BASE / "rosters.json", rosters)
    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_new_endpoint_scores",
        "source_sha256": sha(SOURCE),
        "input_hashes": INPUTS.copy(),
        "rosters_sha256": sha(BASE / "rosters.json"),
        "conditions": CONDITIONS,
        "repeats_or_training_seeds": [0, 1, 2],
        "endpoints": endpoints,
        "exports": exports,
        "purpose": "Exploratory reanalysis of all gold-eligible English core rows already predicted. No fresh evaluation or configuration tuning.",
        "eligibility": "Strict gold hate true, dimension annotated (empty target set is observed and retained), ID in original 970 English core; exact text identity verified.",
        "scoring": "Original seven target labels and three severity labels fixed in every stratum/resample; zero-denominator F1=0. Score each decoder repeat/training seed, then average; no pooled repeats as independent items.",
        "invalid": "Retain all nulls. Own-valid scores and full-roster all-wrong/all-right bounds. Pairwise common-valid masks across six outputs per contrast; shared all-15 mask also reported.",
        "gate": "False predicted gate implies empty fine predictions; null response is not non-hate. Count gate misses separately on observed outputs.",
        "strata": "Report original dedicated and added texts separately and each source/original-roster composition; fixed scoring labels remain unchanged in strata. No old/new absolute F1 improvement claim.",
        "primary_exploratory_family": {
            "contrasts": [
                "target_group:rag_full-rag_small",
                "severity:rag_full-rag_small",
            ],
            "individual_confidence": 0.975,
            "family_error": 0.05,
            "qualification": "Prospective within this exploratory reanalysis, not a new confirmatory family erasing prior DEV exposure.",
        },
        "secondary_contrasts": [
            "rag_full-zero_shot",
            "rag_full-encoder_full",
            "rag_full-encoder_kb",
        ],
        "secondary_confidence": 0.95,
        "bootstrap": {
            "draws": 10000,
            "seed": 20260930,
            "unit": "Normalized exact-text clusters (NFKC, casefold, collapsed whitespace); sorted clusters, shared draws within each contrast.",
            "method": "Resample clusters with replacement, retaining all members. Recompute each repeat/seed macro then average, paired differences. Percentile linear.",
            "qualification": "Conditional on observed repeats, fitted models and selected exposed roster. No claim of near-duplicate independence or representative deployment prevalence.",
        },
        "extra_metrics": "Per-label TP/FP/FN/precision/recall/F1; gold/predicted cardinality; exact-set accuracy; gate-miss rate. Severity confusion includes no-class output; ordinal MAE conditional on a valid nonempty severity, with coverage.",
        "original_check": "Dedicated original subsets must reconstruct each published own-valid per-repeat macro before pooled results are released.",
        "stopping": "Complete both endpoints and all five systems once. No threshold/prompt/KB selection or accuracy-based continuation. Keep all results.",
        "no_live_calls": True,
        "no_TEST": True,
    }
    save(BASE / "plan.json", plan)
    print(
        {
            "status": plan["status"],
            "plan_sha256": sha(BASE / "plan.json"),
            "endpoints": endpoints,
        },
        flush=True,
    )


def load_settings(plan, rosters):
    for name, expected in plan["input_hashes"].items():
        bind(ROOT / name, expected)
    assert sha(SOURCE) == plan["source_sha256"]
    files = {}
    for name, path in plan["exports"].items():
        rows = lines(ROOT / path)
        indexed = {row.get("id", row.get("item_id")): row for row in rows}
        assert len(indexed) == len(rows)
        files[name] = indexed
    settings = {}
    for dimension, roster in rosters.items():
        result = {}
        for condition in CONDITIONS:
            for repeat in range(3):
                outputs = []
                for row in roster:
                    key = (
                        f"{condition}/{repeat}"
                        if condition.startswith("encoder")
                        else f"{row['origin']}/{condition}/{repeat}"
                    )
                    output = files[key][row["id"]]
                    if condition.startswith("encoder"):
                        assert output["key"] == row["key"] and output["seed"] == repeat
                        assert output["text_sha256"] == row["text_sha256"]
                        assert output["arm"] == condition.replace("encoder_", "")
                        assert output["result"] is not None
                    else:
                        assert (
                            output["condition"] == condition
                            and output["replicate"] == repeat
                        )
                        assert (output["result"] is not None) == (output["n_valid"] > 0)
                    prediction = output["result"]
                    labels = []
                    gate = None
                    if prediction is not None:
                        gate = prediction["hate"]
                        assert type(gate) is bool
                        field = prediction[dimension]
                        labels = (
                            ([] if field is None else [field])
                            if dimension == "severity"
                            else field
                        )
                        assert type(labels) is list and set(labels) <= set(
                            LABELS[dimension]
                        )
                        assert gate or not labels
                    outputs.append(
                        {
                            "id": row["id"],
                            "valid": prediction is not None,
                            "gate": gate,
                            "labels": labels,
                        }
                    )
                result[condition, repeat] = outputs
        settings[dimension] = result
    return settings


def f1_counts(counts):
    numerator = 2 * counts[..., 0]
    denominator = numerator + counts[..., 1] + counts[..., 2]
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    ).mean(axis=-1)


def data_arrays(roster, settings, dimension):
    labels = LABELS[dimension]
    truth = np.asarray(
        [[label in row["gold"] for label in labels] for row in roster], dtype=np.int64
    )
    arrays = {}
    for key, rows in settings.items():
        prediction = np.asarray(
            [[label in row["labels"] for label in labels] for row in rows],
            dtype=np.int64,
        )
        valid = np.asarray([row["valid"] for row in rows], dtype=bool)
        counts = np.stack(
            (truth * prediction, (1 - truth) * prediction, truth * (1 - prediction)),
            axis=-1,
        )
        counts[~valid] = 0
        arrays[key] = {
            "prediction": prediction,
            "valid": valid,
            "counts": counts,
            "gate_missed": np.array(
                [row["valid"] and row["gate"] is False for row in rows], dtype=bool
            ),
        }
    return truth, arrays


def metric(truth, array, indices, dimension):
    y = truth[indices]
    valid = array["valid"][indices]
    x = array["prediction"][indices]
    counts = array["counts"][indices].sum(axis=0)
    assert valid.any()
    f1 = float(f1_counts(counts))
    independent = f1_score(y[valid], x[valid], average="macro", zero_division=0)
    assert abs(f1 - independent) < 1e-12
    label_f1 = f1_score(y[valid], x[valid], average=None, zero_division=0)
    label_precision = precision_score(y[valid], x[valid], average=None, zero_division=0)
    label_recall = recall_score(y[valid], x[valid], average=None, zero_division=0)
    low, high = counts.copy(), counts.copy()
    low[:, 1] += (1 - y[~valid]).sum(axis=0)
    low[:, 2] += y[~valid].sum(axis=0)
    high[:, 0] += y[~valid].sum(axis=0)
    out = {
        "rows": len(indices),
        "valid": int(valid.sum()),
        "null": int((~valid).sum()),
        "macro_f1": f1,
        "full_roster_failure_bounds": [float(f1_counts(low)), float(f1_counts(high))],
        "gate_missed": int(array["gate_missed"][indices].sum()),
        "gate_miss_rate_valid": float(array["gate_missed"][indices][valid].mean()),
        "exact_set_accuracy": float((y[valid] == x[valid]).all(axis=1).mean()),
        "predicted_labels_mean": float(x[valid].sum(axis=1).mean()),
        "per_label": {
            label: {
                "tp": int(counts[i, 0]),
                "fp": int(counts[i, 1]),
                "fn": int(counts[i, 2]),
                "gold_support": int(y[:, i].sum()),
                "gold_support_valid": int(y[valid, i].sum()),
                "f1": float(label_f1[i]),
                "precision": float(label_precision[i]),
                "recall": float(label_recall[i]),
            }
            for i, label in enumerate(LABELS[dimension])
        },
    }
    if dimension == "severity":
        confusion = np.zeros((3, 4), dtype=int)
        chosen = valid & (x.sum(axis=1) == 1)
        for yy, xx in zip(y[valid], x[valid]):
            confusion[int(yy.argmax()), int(xx.argmax()) if xx.sum() else 3] += 1
        out["severity_confusion"] = {
            "gold_rows": LABELS[dimension],
            "predicted_columns": LABELS[dimension] + ["no_class"],
            "counts": confusion.tolist(),
        }
        out["ordinal_mae_on_classified"] = (
            float(np.abs(y[chosen].argmax(axis=1) - x[chosen].argmax(axis=1)).mean())
            if chosen.any()
            else None
        )
        out["ordinal_classified_coverage_of_valid"] = float(chosen.sum() / valid.sum())
    return out


def contrast(roster, truth, arrays, left, right, family, seed, draws):
    keys = [(condition, repeat) for condition in [left, right] for repeat in range(3)]
    valid = np.stack([arrays[key]["valid"] for key in keys]).all(axis=0)
    chosen = np.flatnonzero(valid)
    clusters = sorted({roster[i]["cluster"] for i in chosen})
    groups = [
        np.array([i for i in chosen if roster[i]["cluster"] == cluster])
        for cluster in clusters
    ]
    data = np.stack([arrays[key]["counts"] for key in keys], axis=1)
    grouped = np.stack([data[group].sum(axis=0) for group in groups])
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, len(groups), size=(draws, len(groups)))
    weights = np.array(
        [np.bincount(row, minlength=len(groups)) for row in indexes], dtype=np.int64
    )
    totals = (weights @ grouped.reshape(len(groups), -1)).reshape(
        draws, 6, len(truth[0]), 3
    )
    sample = f1_counts(totals)

    max_error = 0.0
    for i in list(range(20)) + list(range(draws - 20, draws)):
        expanded = np.concatenate([groups[g] for g in indexes[i]])
        direct = []
        for key in keys:
            yy, xx = truth[expanded], arrays[key]["prediction"][expanded]
            tp = (yy * xx).sum(axis=0)
            fp = ((1 - yy) * xx).sum(axis=0)
            fn = (yy * (1 - xx)).sum(axis=0)
            den = 2 * tp + fp + fn
            direct.append(
                float(
                    np.divide(2 * tp, den, out=np.zeros(len(tp)), where=den != 0).mean()
                )
            )
        error = float(np.max(np.abs(np.asarray(direct) - sample[i])))
        assert error < 1e-12
        max_error = max(max_error, error)
    delta = sample[:, :3].mean(axis=1) - sample[:, 3:].mean(axis=1)
    points = f1_counts(data[chosen].sum(axis=0))
    bounds = {}
    for condition in [left, right]:
        bounds[condition] = np.mean(
            [
                metric(
                    truth,
                    arrays[condition, r],
                    np.arange(len(roster)),
                    "severity" if truth.shape[1] == 3 else "target_group",
                )["full_roster_failure_bounds"]
                for r in range(3)
            ],
            axis=0,
        )
    return {
        "left": left,
        "right": right,
        "common_valid_n": int(valid.sum()),
        "clusters": len(clusters),
        "excluded_ids": [row["id"] for row, ok in zip(roster, valid) if not ok],
        "left_values": points[:3].tolist(),
        "right_values": points[3:].tolist(),
        "mean_difference": float(points[:3].mean() - points[3:].mean()),
        "ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
        "family_ci975": (
            np.quantile(delta, [0.0125, 0.9875]).tolist() if family else None
        ),
        "full_roster_missing_output_delta_bounds": [
            float(bounds[left][0] - bounds[right][1]),
            float(bounds[left][1] - bounds[right][0]),
        ],
        "seed": seed,
        "draws": draws,
        "direct_expansion_checks": 40,
        "max_direct_error": max_error,
    }, delta


def run():
    plan = read(BASE / "plan.json")
    rosters = read(BASE / "rosters.json", plan["rosters_sha256"])
    assert not (BASE / "complete.json").exists(), "Preserve completed analysis"
    attempt = BASE / "attempts" / str(time.time_ns())
    attempt.mkdir(parents=True, exist_ok=False)
    (attempt / "source.py").write_bytes(SOURCE.read_bytes())
    save(
        attempt / "start.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "plan_sha256": sha(BASE / "plan.json"),
        },
    )
    try:
        settings = load_settings(plan, rosters)
        results = {}
        samples = {}
        published = []
        for dimension, roster in rosters.items():
            truth, arrays = data_arrays(roster, settings[dimension], dimension)
            all_indices = np.arange(len(roster))
            common = np.stack([array["valid"] for array in arrays.values()]).all(axis=0)
            strata = {
                "original": np.array(
                    [i for i, row in enumerate(roster) if row["original_roster"]]
                ),
                "additional": np.array(
                    [i for i, row in enumerate(roster) if not row["original_roster"]]
                ),
            }
            strata.update(
                {
                    "source:"
                    + source: np.array(
                        [i for i, row in enumerate(roster) if row["source"] == source]
                    )
                    for source in sorted({row["source"] for row in roster})
                }
            )
            strata.update(
                {
                    "origin:"
                    + origin: np.array(
                        [i for i, row in enumerate(roster) if row["origin"] == origin]
                    )
                    for origin in sorted({row["origin"] for row in roster})
                }
            )
            result = {
                "roster": plan["endpoints"][dimension],
                "conditions": {},
                "contrasts": {},
                "shared_all15_valid_n": int(common.sum()),
                "shared_all15_excluded_ids": [
                    row["id"] for row, ok in zip(roster, common) if not ok
                ],
            }
            original = read(
                FOLDERS[plan["endpoints"][dimension]["original_subset"]]
                / "results.json"
            )
            for condition in CONDITIONS:
                runs = []
                for repeat in range(3):
                    array = arrays[condition, repeat]
                    own = metric(truth, array, all_indices, dimension)
                    shared = metric(truth, array, np.flatnonzero(common), dimension)
                    sub = {
                        name: metric(truth, array, indices, dimension)
                        for name, indices in strata.items()
                    }
                    reference = original["conditions"][condition]["runs"][repeat][
                        "score"
                    ]
                    assert (
                        abs(
                            sub["original"]["macro_f1"]
                            - reference["macro_f1_valid_only"]
                        )
                        < 1e-12
                    )
                    assert sub["original"]["valid"] == reference["n_valid"]
                    assert sub["original"]["rows"] == reference["n_scorable"]
                    runs.append(
                        {
                            "repeat_or_seed": repeat,
                            "own_valid": own,
                            "shared_valid": shared,
                            "strata": sub,
                        }
                    )
                    for gold, output in zip(
                        roster, settings[dimension][condition, repeat]
                    ):
                        published.append(
                            {
                                "dimension": dimension,
                                "condition": condition,
                                "repeat_or_seed": repeat,
                                **gold,
                                "prediction": output,
                            }
                        )
                values = [row["own_valid"]["macro_f1"] for row in runs]
                shared_values = [row["shared_valid"]["macro_f1"] for row in runs]
                result["conditions"][condition] = {
                    "runs": runs,
                    "own_valid_mean": float(np.mean(values)),
                    "own_valid_values": values,
                    "sample_sd": float(np.std(values, ddof=1)),
                    "shared_valid_mean": float(np.mean(shared_values)),
                    "shared_valid_values": shared_values,
                    "null_prediction_slots": sum(
                        row["own_valid"]["null"] for row in runs
                    ),
                    "mean_failure_bounds": np.mean(
                        [
                            row["own_valid"]["full_roster_failure_bounds"]
                            for row in runs
                        ],
                        axis=0,
                    ).tolist(),
                    "mean_gate_miss_rate": float(
                        np.mean(
                            [row["own_valid"]["gate_miss_rate_valid"] for row in runs]
                        )
                    ),
                }
            for other in CONDITIONS[1:]:
                name = "rag_full_minus_" + other
                out, boot = contrast(
                    roster,
                    truth,
                    arrays,
                    "rag_full",
                    other,
                    other == "rag_small",
                    plan["bootstrap"]["seed"],
                    plan["bootstrap"]["draws"],
                )
                result["contrasts"][name] = out
                samples[dimension + "__" + name] = boot
            results[dimension] = result
            print(
                {
                    "endpoint": dimension,
                    "n": len(roster),
                    "means": {
                        key: value["own_valid_mean"]
                        for key, value in result["conditions"].items()
                    },
                },
                flush=True,
            )
        save(
            attempt / "results.json",
            {
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "plan_sha256": sha(BASE / "plan.json"),
                "endpoints": results,
                "new_model_calls": 0,
                "original_stratum_checks": 30,
                "interpretation": "Exploratory pooled exposed-DEV reanalysis; original dedicated strata reproduced, not replaced.",
            },
        )
        with (attempt / "predictions.jsonl").open(
            "x", encoding="utf-8", newline="\n"
        ) as stream:
            for row in published:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        np.savez_compressed(attempt / "bootstrap_differences.npz", **samples)
        assert_inputs()
        save(attempt / "inputs.json", INPUTS)
        receipt = {
            "status": "pooled_fineheads_scored",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "attempt": attempt.relative_to(ROOT).as_posix(),
            "plan_sha256": sha(BASE / "plan.json"),
            "source_sha256": sha(SOURCE),
            "exported_rows": len(published),
            "original_stratum_checks": 30,
            "paired_bootstrap_contrasts": 8,
            "independent_expanded_setting_draw_checks": 8 * 40 * 6,
            "input_hashes_verified": len(INPUTS),
            "new_model_calls": 0,
            "outputs": {
                path.name: sha(path) for path in attempt.iterdir() if path.is_file()
            },
        }
        save(attempt / "complete.json", receipt)
        save(
            BASE / "complete.json",
            {
                "status": receipt["status"],
                "attempt": receipt["attempt"],
                "receipt_sha256": sha(attempt / "complete.json"),
            },
        )
        print(
            {
                "status": receipt["status"],
                "attempt": receipt["attempt"],
                "input_hashes_verified": len(INPUTS),
            },
            flush=True,
        )
    except Exception:
        save(
            attempt / "failure.json",
            {
                "traceback": traceback.format_exc(),
                "plan_sha256": sha(BASE / "plan.json"),
            },
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["prepare", "run"])
    args = parser.parse_args()
    with patch.object(
        socket.socket, "connect", side_effect=RuntimeError("Offline pooled analysis")
    ):
        prepare() if args.mode == "prepare" else run()

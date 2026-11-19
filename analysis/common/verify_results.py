from src.hsrag.paths import PROJECT_ROOT

from src.hsrag.paths import evaluation_path
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
import traceback
from unittest.mock import patch

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd

ROOT = PROJECT_ROOT
STUDY = ROOT / "experiments/astra/supervisor_reverification_20260930_v1"
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
HASHES = {}
CACHE = {}
FIELDS = {
    "hate": "gate",
    "target_group": "target_groups",
    "hate_type": "hate_types",
    "severity": "severity",
    "legal": "legal",
}
SUBSETS = [
    "en_dev_eval_main",
    "de_legal_dev_eval",
    "en_dev_eval_targets",
    "en_dev_eval_severity",
    "de_dev_eval",
    "en_dev_eval_sq3_types",
    "en_dev_eval_types",
]


def sha(path):
    path = evaluation_path(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bind(path, expected=None):
    path = Path(path)
    name = path.relative_to(ROOT).as_posix()
    value = sha(path)
    if expected is not None:
        assert value == expected, ("Changed published input", name)
    if name in HASHES:
        assert HASHES[name] == value, ("Input changed during audit", name)
    HASHES[name] = value
    return path


def read(path):
    path = evaluation_path(path)
    return json.loads(bind(path).read_text(encoding="utf-8-sig"))


def lines(path, expected=None):
    path = bind(path, expected)
    with path.open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(
            value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        stream.write("\n")


def clean(value):
    if (
        value is None
        or value is pd.NA
        or (isinstance(value, float) and math.isnan(value))
    ):
        return None
    return value.tolist() if isinstance(value, np.ndarray) else value


def gold_value(row, dimension):
    gate = clean(row.get("gate"))
    if dimension == "hate":
        return gate
    if dimension != "legal" and gate is not True:
        return None
    return clean(row.get(FIELDS[dimension]))


def bits(value, dimension, labels):
    if dimension == "hate":
        assert type(value) is bool
        selected = {"hate" if value else "not_hate"}
    elif dimension == "severity":
        selected = set() if value is None else {value}
    else:
        selected = set(value or [])
    assert selected <= set(labels), (dimension, selected, labels)
    return [int(label in selected) for label in labels]


def f1(counts, indices):
    counts = np.asarray(counts)
    numerator = counts[..., 0] * 2
    denominator = numerator + counts[..., 1] + counts[..., 2]
    values = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator != 0,
    )
    return values[..., indices].mean(axis=-1)


def close(actual, expected):
    assert np.allclose(actual, expected, rtol=0, atol=1e-11), (actual, expected)


def arrays(rows, outputs, endpoint):
    assert {str(row["id"]) for row in rows} == set(outputs)
    kept = [row for row in rows if gold_value(row, endpoint["dimension"]) is not None]
    ids = [str(row["id"]) for row in kept]
    assert ids == endpoint["scorable_ids"]
    labels = endpoint["all_labels"]
    truth = np.array(
        [
            bits(gold_value(row, endpoint["dimension"]), endpoint["dimension"], labels)
            for row in kept
        ],
        dtype=np.int64,
    )
    valid, prediction = ([], [])
    for iid in ids:
        record = outputs[iid]
        out = record["result"]
        assert (out is None) == (record["n_valid"] == 0)
        valid.append(out is not None)
        prediction.append(
            [0] * len(labels)
            if out is None
            else bits(out[endpoint["dimension"]], endpoint["dimension"], labels)
        )
    prediction = np.array(prediction, dtype=np.int64)
    valid = np.array(valid, dtype=bool)
    counts = np.stack(
        (truth * prediction, (1 - truth) * prediction, truth * (1 - prediction)),
        axis=-1,
    )
    counts[~valid] = 0
    indices = [labels.index(label) for label in endpoint["macro_labels"]]
    assert {
        label: int(truth[:, i].sum()) for i, label in enumerate(labels)
    } == endpoint["gold_support"]
    return {
        "ids": ids,
        "gold": truth,
        "valid": valid,
        "counts": counts,
        "indices": indices,
    }


def verify_score(arr, published):
    total = arr["counts"].sum(axis=0)
    assert total.tolist() == published["per_label_counts"]
    close(float(f1(total, arr["indices"])), published["macro_f1_valid_only"])
    assert int(arr["valid"].sum()) == published["n_valid"]
    assert len(arr["ids"]) == published["n_scorable"]
    missing_gold = arr["gold"][~arr["valid"]]
    lower, upper = (total.copy(), total.copy())
    lower[:, 1] += (1 - missing_gold).sum(axis=0)
    lower[:, 2] += missing_gold.sum(axis=0)
    upper[:, 0] += missing_gold.sum(axis=0)
    close(
        [float(f1(lower, arr["indices"])), float(f1(upper, arr["indices"]))],
        published["invalid_inclusive_bounds"],
    )
    return float(f1(total, arr["indices"]))


def audit_endpoint(study_name, directory, subset, primary=False):
    receipt = read(directory / "complete.json")
    result = read(directory / "results.json")
    bind(directory / "results.json", receipt["results_sha256"])
    bind(directory / "plan.json", receipt["plan_sha256"])
    obj = result["endpoints"][subset] if primary else result
    endpoint = obj["endpoint"]
    rows = pd.read_parquet(bind(ROOT / f"data/processed/{subset}.parquet")).to_dict(
        "records"
    )
    published_rows = 0
    settings = {}
    outputs_by_setting = {}
    scores = {}
    for condition, group in obj["conditions"].items():
        values = []
        for run in group["runs"]:
            rep = run["replicate"]
            export = (
                directory
                / "assembled"
                / (subset if primary else "")
                / f"{condition}_rep{rep}.jsonl"
            )
            relative = export.relative_to(ROOT).as_posix()
            record = receipt["assembled"][relative]
            exported = lines(export, record["sha256"])
            assert len(exported) == record["rows"]
            out = {str(row["id"]): row for row in exported}
            assert len(out) == len(exported)
            assert all(
                (
                    row["condition"] == condition and row["replicate"] == rep
                    for row in exported
                )
            )
            arr = arrays(rows, out, endpoint)
            value = verify_score(arr, run["score"])
            values.append(value)
            settings[condition, rep] = arr
            outputs_by_setting[condition, rep] = out
            published_rows += len(exported)
        summary = group.get("summary", group.get("own_valid_summary"))
        if summary:
            close(float(np.mean(values)), summary["mean"])
            close(values, summary["values"])
        scores[condition] = values
    CACHE[study_name, subset] = {"settings": settings, "obj": obj, "endpoint": endpoint}
    if study_name == "core":
        CACHE[study_name, subset]["outputs"] = outputs_by_setting
    return {
        "study": study_name,
        "subset": subset,
        "dimension": endpoint["dimension"],
        "macro_labels": endpoint["macro_labels"],
        "texts": len(rows),
        "settings_verified": len(settings),
        "exported_rows_verified": published_rows,
        "own_valid_f1": scores,
    }


def bootstrap(study, subset, comparison, weights, saved, seed, draws=10000):
    cached = CACHE[study, subset]
    settings = cached["settings"]
    first = next(iter(settings.values()))
    wanted_ids = comparison["common_valid_ids"]
    selected = [first["ids"].index(iid) for iid in wanted_ids]
    keys = [(condition, rep) for condition in weights for rep in range(3)]
    for key in keys:
        assert settings[key]["valid"][selected].all(), key
    data = np.stack([settings[key]["counts"][selected] for key in keys], axis=1)
    n, runs, labels, _ = data.shape
    rng = np.random.default_rng(seed)
    frequencies = np.stack(
        [np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(draws)]
    ).astype(float)
    totals = (frequencies @ data.reshape(n, -1)).reshape(draws, runs, labels, 3)
    samples = f1(totals, first["indices"])
    points = f1(data.sum(axis=0), first["indices"])
    delta = np.zeros(draws)
    point = 0.0
    for condition, weight in weights.items():
        pos = [i for i, key in enumerate(keys) if key[0] == condition]
        delta += weight * samples[:, pos].mean(axis=1)
        point += weight * points[pos].mean()
    close(point, saved["mean_difference"])
    close(np.quantile(delta, [0.025, 0.975]), saved["ci95"])
    if saved.get("family_ci") is not None:
        confidence = saved["family_confidence"]
        close(
            np.quantile(delta, [(1 - confidence) / 2, (1 + confidence) / 2]),
            saved["family_ci"],
        )
    return {
        "study": study,
        "subset": subset,
        "n_common": n,
        "weights": weights,
        "difference": float(point),
        "ci95": saved["ci95"],
        "family_ci": saved.get("family_ci"),
        "draws_independently_recomputed": draws,
        "seed": seed,
    }


def bootstrap_checks():
    checks = []
    for subset in SUBSETS:
        obj = CACHE["core", subset]["obj"]
        saved = obj["contrasts"]["rag_full_minus_rag_small"]
        checks.append(
            bootstrap(
                "core",
                subset,
                saved,
                {"rag_full": 1, "rag_small": -1},
                saved,
                saved["bootstrap_seed"],
            )
        )
    selections = [
        ("components", "en_dev_eval_sq3_types"),
        ("strategy", "en_dev_eval_sq3_types"),
        ("allocation", "en_dev_eval_main"),
        ("allocation", "en_dev_eval_sq3_types"),
        ("pool", "en_dev_eval_main"),
        ("pool", "de_legal_dev_eval"),
        ("pool", "en_dev_eval_sq3_types"),
    ]
    for study, subset in selections:
        obj = CACHE[study, subset]["obj"]
        comparison = obj["paired_comparison"]
        if study == "components":
            comparison = comparison["factorial"]
        for name, saved in comparison["contrasts"].items():
            if saved.get("family_ci") is None:
                continue
            if "weights" in saved:
                weights = saved["weights"]
            elif study == "pool":
                weights = {}
                for group, sign in ((saved["left"], 1), (saved["right"], -1)):
                    members = [
                        c
                        for c in obj["conditions"]
                        if c == group or c.startswith(group + "_s")
                    ]
                    assert len(members) in (1, 3)
                    weights.update({c: sign / len(members) for c in members})
            else:
                weights = {saved["left"]: 1, saved["right"]: -1}
            checked = bootstrap(
                study, subset, comparison, weights, saved, comparison["bootstrap_seed"]
            )
            checked["contrast"] = name
            checks.append(checked)
        print(f"Bootstrap verified: {study}/{subset}", flush=True)
    return checks


def census_and_reuse():
    previous = read(
        ROOT
        / "experiments/astra/thesis_depth_review_20260930_v1/inventory_v1/inventory.json"
    )
    for name, expected in previous["inputs"].items():
        bind(ROOT / name, expected)
    frame = pd.read_parquet(bind(ROOT / "data/processed/en_dev.parquet"))
    core = lines(CORE / "dev_inputs.jsonl")
    en_core = {row["item_id"]: row for row in core if row["lang"] == "en"}
    targets = frame["gate"].eq(True) & frame["target_groups"].notna()
    severity = frame["gate"].eq(True) & frame["severity"].notna()
    inside = frame["id"].isin(en_core)
    assert int(targets.sum()) == 2506 and int(severity.sum()) == 1606
    assert int((severity & ~targets).sum()) == 0
    reuse = {}
    for name, mask, expected in [
        ("targets", targets, 163),
        ("severity", severity, 116),
    ]:
        chosen = frame[mask & inside].to_dict("records")
        assert len(chosen) == expected
        statuses = Counter()
        origins = Counter()
        for row in chosen:
            iid = str(row["id"])
            entry = en_core[iid]
            assert entry["text"] == row["text"]
            subset = entry["scopes"][0]["subset"]
            origins[subset] += 1
            for condition in ("rag_full", "rag_small", "zero_shot"):
                for rep in range(3):
                    export = CACHE["core", subset]["outputs"][condition, rep][iid]
                    result = export["result"]
                    if result is not None:
                        assert all(
                            (
                                field in result
                                for field in (
                                    "hate",
                                    "hate_type",
                                    "target_group",
                                    "severity",
                                    "legal",
                                )
                            )
                        )
                    statuses["null" if result is None else "available"] += 1
        reuse[name] = {
            "eligible_core_rows": len(chosen),
            "by_original_roster": dict(origins),
            "existing_full_small_zero_prediction_slots": dict(statuses),
            "new_endpoint_scores_computed": False,
        }
    original_targets = set(
        CACHE["core", "en_dev_eval_targets"]["endpoint"]["scorable_ids"]
    )
    original_severity = set(
        CACHE["core", "en_dev_eval_severity"]["endpoint"]["scorable_ids"]
    )
    reuse["new_over_original"] = {
        "targets": 163 - len(original_targets),
        "severity": 116 - len(original_severity),
    }
    return {
        "english_total_targets": 2506,
        "english_total_severity": 1606,
        "all_severity_in_targets": True,
        "saved_prediction_reuse": reuse,
        "unique_explanation_sources": previous["SQ3_explanation_source_inventory"][
            "unique_sources"
        ],
        "compact_validation": [
            x
            for x in previous["encoder_validation_assets"]
            if x["job"].startswith("kb/") and x["job"].endswith("seed0")
        ],
    }


def raw_annotation_counts():
    report = {}
    prompts_by_pool = {}
    for pool in ("small", "full"):
        records = lines(CORE / f"kb/{pool}/records.jsonl")
        examples = {row["id"]: row for row in records if row["kind"] == "example"}
        english = [row for row in examples.values() if row["lang"] == "en"]
        receipt = read(CORE / f"retrieval_v1/{pool}/complete.json")
        prompts = lines(
            CORE / f"retrieval_v1/{pool}/prompts.jsonl", receipt["prompt_sha256"]
        )
        prompts_by_pool[pool] = {row["key"]: row for row in prompts}
        gate_counts = Counter()
        type_slots = Counter()
        type_missing = 0
        for prompt in prompts:
            ids = prompt["retrieved"]["Labelled examples"]
            assert len(ids) == 5
            hit_records = [examples[iid] for iid in ids]
            subset = prompt["scopes"][0]["subset"]
            if subset == "en_dev_eval_main":
                gate_counts.update(
                    (str(row["meta"].get("gate")) for row in hit_records)
                )
            if subset == "en_dev_eval_sq3_types":
                count = sum(
                    (row["meta"].get("hate_types") is not None for row in hit_records)
                )
                type_slots["annotated"] += count
                type_slots["all"] += len(hit_records)
                type_missing += count == 0
        report[pool] = {
            "english_examples": len(english),
            "english_hateful": sum(
                (row["meta"].get("gate") is True for row in english)
            ),
            "english_type_annotated": sum(
                (row["meta"].get("hate_types") is not None for row in english)
            ),
            "english_hate_roster_example_gate_slots": dict(gate_counts),
            "main_type_annotated_slots": dict(type_slots),
            "main_type_inputs_without_annotation": type_missing,
        }
    assert (
        report["small"]["english_hateful"] == 127
        and report["full"]["english_hateful"] == 16960
    )
    assert report["small"]["english_hate_roster_example_gate_slots"] == {
        "True": 1297,
        "False": 198,
    }
    assert report["full"]["english_hate_roster_example_gate_slots"] == {
        "True": 499,
        "False": 996,
    }
    assert report["small"]["main_type_annotated_slots"]["annotated"] == 1428
    assert report["full"]["main_type_annotated_slots"]["annotated"] == 795
    assert report["small"]["main_type_inputs_without_annotation"] == 12
    assert report["full"]["main_type_inputs_without_annotation"] == 106
    assert len(prompts_by_pool["small"]) == len(prompts_by_pool["full"]) == 1295
    for key, small in prompts_by_pool["small"].items():
        full = prompts_by_pool["full"][key]
        for field in ("definitions", "guidelines"):
            assert small["retrieved"][field] == full["retrieved"][field]
    report["all_1295_abstract_id_lists_match"] = True
    return report


def main():
    attempt = STUDY / "attempts" / str(time.time_ns())
    attempt.mkdir(parents=True)
    start = time.perf_counter()
    bind(Path(__file__).resolve())
    (attempt / "source.py").write_bytes(Path(__file__).resolve().read_bytes())
    save(
        attempt / "plan.json",
        {
            "purpose": __doc__,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "Recompute completed core and all four SQ1 matrix-study metrics and selected frozen intervals. Reverify census/reuse feasibility; no new endpoint scores.",
            "source_sha256": sha(Path(__file__).resolve()),
            "new_model_calls": 0,
        },
    )
    try:
        summaries = []
        primary_dir = CORE / "primary_results_v1"
        for subset in SUBSETS[:2]:
            summaries.append(audit_endpoint("core", primary_dir, subset, primary=True))
        directories = {
            "en_dev_eval_targets": CORE / "secondary_results_v2/en_dev_eval_targets",
            "en_dev_eval_severity": CORE / "severity_results_v1",
            "de_dev_eval": CORE / "secondary_results_v3/de_dev_eval",
            "en_dev_eval_sq3_types": CORE
            / "secondary_results_v4/en_dev_eval_sq3_types",
            "en_dev_eval_types": CORE / "secondary_results_v4/en_dev_eval_types",
        }
        for subset, directory in directories.items():
            summaries.append(audit_endpoint("core", directory, subset))
        print("Independent core score/count/bound checks complete.", flush=True)
        studies = {
            "components": "sq1_full_kb_components_20260921_v1/component_results_v1",
            "pool": "sq1_pool_size_composition_20260926_v1/pool_results_v1",
            "strategy": "sq1_full_kb_strategy_20260920_v1/strategy_results_v2",
            "allocation": "sq1_full_kb_allocation_20260924_v1/allocation_results_v1",
        }
        for name, relative in studies.items():
            for subset in SUBSETS:
                directory = ROOT / "experiments/astra" / relative / subset
                if name == "strategy" and (not (directory / "complete.json").exists()):
                    directory = (
                        ROOT
                        / "experiments/astra"
                        / relative.replace("v2", "v1")
                        / subset
                    )
                summaries.append(audit_endpoint(name, directory, subset))
            print(
                f"Independent metric/count/bound checks complete: {name}.", flush=True
            )
        save(attempt / "metric_checks.json", summaries)
        intervals = bootstrap_checks()
        save(attempt / "bootstrap_checks.json", intervals)
        census = census_and_reuse()
        save(attempt / "coverage_and_reuse.json", census)
        annotation = raw_annotation_counts()
        save(attempt / "annotation_counts.json", annotation)
        for name, expected in HASHES.items():
            assert sha(ROOT / name) == expected, name
        save(attempt / "inputs.json", HASHES)
        complete = {
            "status": "independent_numeric_and_census_checks_passed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "new_API_calls": 0,
            "endpoint_study_tables": len(summaries),
            "settings_verified": sum((x["settings_verified"] for x in summaries)),
            "exported_prediction_rows_verified": sum(
                (x["exported_rows_verified"] for x in summaries)
            ),
            "paired_bootstrap_intervals_recomputed": len(intervals),
            "source_hashes_verified_before_after": len(HASHES),
            "elapsed_seconds": time.perf_counter() - start,
            "scope_limit": "Published prediction exports and existing audit bindings; not a new replay of every raw provider response or all historical side studies.",
            "outputs": {p.name: sha(p) for p in attempt.iterdir() if p.is_file()},
        }
        save(attempt / "complete.json", complete)
        print(
            json.dumps(
                {
                    **{k: v for k, v in complete.items() if k != "outputs"},
                    "attempt": attempt.relative_to(ROOT).as_posix(),
                    "saved_prediction_reuse": census["saved_prediction_reuse"],
                },
                indent=2,
            ),
            flush=True,
        )
    except BaseException:
        save(
            attempt / "failure.json",
            {
                "status": "preserved_failure_no_scientific_output_published",
                "traceback": traceback.format_exc(),
                "source_sha256": sha(Path(__file__).resolve()),
            },
        )
        raise


if __name__ == "__main__":
    with patch.object(
        socket.socket,
        "connect",
        side_effect=RuntimeError("Offline audit; network forbidden"),
    ):
        main()

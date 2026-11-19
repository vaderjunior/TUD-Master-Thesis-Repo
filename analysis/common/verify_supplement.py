from datetime import datetime, timezone
from pathlib import Path
import socket
import time
import traceback
from unittest.mock import patch

from analysis.common.verify_results import ROOT
from analysis.common.verify_results import STUDY
from analysis.common.verify_results import HASHES
from analysis.common.verify_results import bind
from analysis.common.verify_results import read
from analysis.common.verify_results import lines
from analysis.common.verify_results import save
from analysis.common.verify_results import sha
from analysis.common.verify_results import close
from analysis.common.verify_results import f1
from analysis.common.verify_results import np


def guideline():
    base = (
        ROOT
        / "experiments/astra/sq1_full_kb_guideline_oracle_20260925_v1/analysis_results_v1"
    )
    receipt = read(base / "complete.json")
    for path, expected in receipt["published_hashes"].items():
        bind(ROOT / path, expected)
    matrix = read(base / "gate_matrix.json")
    evidence = read(
        ROOT
        / "Thesis_Log/astra_result/evidence/sq1_guideline_final_results_20260926_v1.json"
    )
    published = evidence["scopes"]["all"]
    conditions = matrix["conditions"]
    ids = [item["item_id"] for item in matrix["items"]]
    assert len(ids) == len(set(ids)) == 693
    assert all(item["gate"] is False for item in matrix["items"])
    observed = np.full((693, 5, 3), -2, dtype=np.int16)
    lookup = {item: i for i, item in enumerate(ids)}
    rows = lines(base / "predictions.jsonl")
    for row in rows:
        idx = (
            lookup[row["item_id"]],
            conditions.index(row["condition"]),
            row["repeat"],
        )
        assert observed[idx] == -2
        assert row["hate"] is None or type(row["hate"]) is bool
        observed[idx] = -1 if row["hate"] is None else int(row["hate"])
    assert np.array_equal(observed, np.asarray(matrix["values"]))
    common = (observed >= 0).all(axis=(1, 2))
    assert int(common.sum()) == published["common_items"] == 692
    for setting in published["settings"]:
        ci, repeat = conditions.index(setting["condition"]), setting["repeat"]
        own = observed[:, ci, repeat]
        valid = own >= 0
        fp, null = int(own[valid].sum()), int((~valid).sum())
        assert fp == setting["false_positives"]
        assert null == setting["null"]
        assert int(valid.sum()) == setting["valid"]
        assert int(own[common].sum()) == setting["common_false_positives"]
        close(own[common].mean(), setting["common_rate"])
        close(own[valid].mean(), setting["own_valid_rate"])
        close([fp / 693, (fp + null) / 693], setting["full_roster_bounds"])
    templates = np.array([item["templ_id"] for item in matrix["items"]])[common]
    unique = sorted(set(templates.tolist()))
    assert unique == published["common_templates"] and len(unique) == 302
    sums = np.array(
        [observed[common][templates == tid].sum(axis=(0, 2)) for tid in unique]
    )
    sizes = np.array([(templates == tid).sum() for tid in unique])
    draws = np.random.default_rng(20260925).integers(
        0, 302, size=(10000, 302), dtype=np.int32
    )
    sampled = sums[draws].sum(axis=1) / (sizes[draws].sum(axis=1)[:, None] * 3)
    means = observed[common].mean(axis=(0, 2))
    checks = {}
    for name, contrast in published["contrasts"].items():
        left, right = conditions.index(contrast["left"]), conditions.index(
            contrast["right"]
        )
        delta = sampled[:, left] - sampled[:, right]
        close(means[left] - means[right], contrast["mean_difference"])
        close(np.quantile(delta, [0.025, 0.975]), contrast["descriptive_95_interval"])
        if contrast["primary_family_interval"] is not None:
            close(
                np.quantile(delta, [0.00625, 0.99375]),
                contrast["primary_family_interval"],
            )
        checks[name] = contrast
    return {
        "exported_rows": len(rows),
        "settings": len(published["settings"]),
        "common_items": 692,
        "templates": 302,
        "bootstrap_contrasts": len(checks),
        "common_false_positive_rates": dict(zip(conditions, means.tolist())),
        "comparisons": checks,
    }


def contributions(ids, rows, gold, labels):
    truth = np.array(
        [[label in gold[item]["hate_type"] for label in labels] for item in ids],
        dtype=bool,
    )
    valid = np.array([rows[item]["result"] is not None for item in ids])
    predictions = []
    gates = []
    for item in ids:
        result = rows[item]["result"]
        if result is None:
            selected, gate = set(), False
        else:
            gate = result["hate"]
            assert type(gate) is bool
            selected = set(result["hate_type"]) if gate else set()
            assert selected <= set(labels)
        predictions.append([label in selected for label in labels])
        gates.append(gate)
    predictions = np.asarray(predictions, dtype=bool)
    counts = np.stack(
        (truth & predictions, ~truth & predictions, truth & ~predictions), axis=-1
    ).astype(np.int64)
    return truth, predictions, np.asarray(gates), valid, counts


def score(ids, rows, gold, labels, published):
    truth, predictions, gates, valid, counts = contributions(ids, rows, gold, labels)
    counts = counts[valid].sum(axis=0)
    assert int(valid.sum()) == published["valid"]
    assert len(ids) == published["n"]
    assert sorted([item for item, ok in zip(ids, valid) if not ok]) == sorted(
        published["null_ids"]
    )
    contains = (~truth[valid] | predictions[valid]).all(axis=1).mean()
    out = {
        "macro_f1": float(f1(counts, list(range(len(labels))))),
        "contains": float(contains),
        "flip_rate": float(1 - contains),
        "exact": float((truth[valid] == predictions[valid]).all(axis=1).mean()),
        "predicted_labels": float(predictions[valid].sum(axis=1).mean()),
        "missed_hate_rate": float((~gates[valid]).mean()),
    }
    for metric, value in out.items():
        close(value, published[metric])
    assert int((~gates[valid]).sum()) == published["gate_false"]
    for label, (tp, fp, fn) in zip(labels, counts):
        current = published["per_label"][label]
        assert [int(tp), int(fp), int(fn)] == [
            current["tp"],
            current["fp"],
            current["fn"],
        ]
        close(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0, current["f1"])

    lower = counts.copy()
    lower[:, 1] += (~truth[~valid]).sum(axis=0)
    lower[:, 2] += truth[~valid].sum(axis=0)
    upper = counts.copy()
    upper[:, 0] += truth[~valid].sum(axis=0)
    close(
        [f1(lower, list(range(len(labels)))), f1(upper, list(range(len(labels))))],
        published["full_roster_failure_bounds"],
    )
    return out


def feedback():
    base = ROOT / "experiments/astra/sq3_full_kb_feedback_20260928_v1"
    plan = read(base / "scientific_plan_v1.json")
    labels = sorted(plan["scoring"]["labels"])
    gold = {
        row["id"]: row["gold"] for row in lines(base / "feasibility_v1/roster.jsonl")
    }
    saved = {}
    count = 0
    for receipt_name in ["updates_review_v1", "baseline_review_v1"]:
        receipt = read(base / receipt_name / "complete.json")
        for path, expected in receipt["outputs"].items():
            if (
                not path.endswith(".jsonl")
                or receipt_name == "baseline_review_v1"
                and "/reference_rep" not in path
            ):
                continue
            rows = lines(ROOT / path, expected)
            assert len(rows) == len({row["id"] for row in rows}) == 466
            condition = rows[0].get("condition", "reference")
            if receipt_name == "baseline_review_v1":
                condition = "reference"
            repeat = rows[0].get("repeat", rows[0].get("replicate"))
            assert repeat in [0, 1, 2]
            assert (condition, repeat) not in saved
            saved[condition, repeat] = {row["id"]: row for row in rows}
            count += len(rows)
    assert len(saved) == 75
    receipt = read(base / "results_v1/complete.json")
    for path, expected in receipt["outputs"].items():
        bind(ROOT / path, expected)
    selection = read(base / "baseline_review_v1/selection.json")
    endpoints = {
        "transfer": plan["rosters"]["transfer"],
        "holdout": plan["rosters"]["holdout"],
        "sentinel": selection["sentinel_ids"],
    }
    metrics_checked = 0
    comparisons = {}
    final_transfer = None
    for endpoint, ids in endpoints.items():
        for budget in range(1, 5):
            published = read(base / f"results_v1/{endpoint}_b{budget:02d}.json")
            settings = list(published["conditions"])
            common = [
                item
                for item in ids
                if all(
                    saved[c, r][item]["result"] is not None
                    for c in settings
                    for r in range(3)
                )
            ]
            assert common == published["common_valid_ids"]
            assert len(common) == published["common_valid_n"]
            derived = {}
            for condition, details in published["conditions"].items():
                for repeat in range(3):
                    rowmap = saved[condition, repeat]
                    derived[condition, repeat] = score(
                        common,
                        rowmap,
                        gold,
                        labels,
                        details["common_valid_runs"][repeat],
                    )
                    score(ids, rowmap, gold, labels, details["own_valid_runs"][repeat])
                    metrics_checked += 2
            group_values = {}
            for group, details in published["groups"].items():
                group_values[group] = {
                    metric: float(
                        np.mean(
                            [
                                derived[c, r][metric]
                                for c in details["conditions"]
                                for r in range(3)
                            ]
                        )
                    )
                    for metric in details["metrics"]
                }
                for metric, value in group_values[group].items():
                    close(value, details["metrics"][metric])
            for details in published["contrasts"].values():
                close(
                    group_values[details["left"]]["macro_f1"]
                    - group_values[details["right"]]["macro_f1"],
                    details["mean_difference"],
                )
            comparisons[f"{endpoint}_b{budget:02d}"] = {
                "common_n": len(common),
                "groups": group_values,
            }
            if endpoint == "transfer" and budget == 4:
                final_transfer = published

    ids = final_transfer["common_valid_ids"]
    n = len(ids)
    draws = np.random.default_rng(20260928).integers(0, n, size=(10000, n))
    weights = np.zeros((10000, n), dtype=np.int64)
    for index, sample in enumerate(draws):
        weights[index] = np.bincount(sample, minlength=n)
    boot_groups = {}
    for group, details in final_transfer["groups"].items():
        setting_scores = []
        for condition in details["conditions"]:
            for repeat in range(3):
                _, _, _, valid, counts = contributions(
                    ids, saved[condition, repeat], gold, labels
                )
                assert valid.all()
                totals = (weights @ counts.reshape(n, -1)).reshape(
                    10000, len(labels), 3
                )
                setting_scores.append(f1(totals, list(range(len(labels)))))
        boot_groups[group] = np.mean(setting_scores, axis=0)
        close(np.quantile(boot_groups[group], [0.025, 0.975]), details["macro_f1_ci95"])
    for details in final_transfer["contrasts"].values():
        delta = boot_groups[details["left"]] - boot_groups[details["right"]]
        close(np.quantile(delta, [0.025, 0.975]), details["macro_f1_ci95"])
    pools = {
        pool["condition"]: pool["ids"]
        for pool in read(base / "update_payloads_v1/pools.json")["pools"]
    }
    candidate_summaries = []
    for candidate in read(base / "results_v1/candidate_diagnostics.json"):
        if (
            candidate["arm"] != "error"
            or candidate["pool_size"] != 40
            or candidate["role"] != "selected"
        ):
            continue
        condition = candidate["condition"]
        ids = [
            item for item in pools[condition] if item not in candidate["excluded_ids"]
        ]
        results = {}
        for role, actual in [("reference", "reference"), ("update", condition)]:
            runs = [
                score(
                    ids,
                    saved[actual, repeat],
                    gold,
                    labels,
                    candidate["runs"][role][repeat],
                )
                for repeat in range(3)
            ]
            metrics_checked += 3
            results[role] = {
                metric: float(np.mean([row[metric] for row in runs]))
                for metric in runs[0]
            }
            for metric, value in results[role].items():
                close(value, candidate["summary"][role][metric]["mean"])
        candidate_summaries.append(results)
    assert len(candidate_summaries) == 3
    direct = {
        role: {
            metric: float(np.mean([row[role][metric] for row in candidate_summaries]))
            for metric in candidate_summaries[0][role]
        }
        for role in ["reference", "update"]
    }
    return {
        "exported_rows": count,
        "files": len(saved),
        "metric_evaluations": metrics_checked,
        "endpoint_budget_tables": len(comparisons),
        "tables": comparisons,
        "final_transfer_macro_bootstrap_contrasts": 3,
        "final_transfer_contrasts": final_transfer["contrasts"],
        "selected_error40": direct,
    }


def mmbert():
    value = read(
        ROOT / "Thesis_Log/astra_result/evidence/mmbert_results_20260913_v1.json"
    )
    bind(ROOT / value["experiment_plan"], value["plan_sha256"])
    items = value["item_predictions"]
    assert len(items) == len({row["id"] for row in items}) == 150
    output = {}
    for model in ["gbert", "mmbert"]:
        scores = []
        for repeat in range(3):
            tp = sum(row["gold"] and row[model][repeat] for row in items)
            fp = sum(not row["gold"] and row[model][repeat] for row in items)
            fn = sum(row["gold"] and not row[model][repeat] for row in items)
            tn = sum(not row["gold"] and not row[model][repeat] for row in items)
            score_value = float(f1([[tp, fp, fn], [tn, fn, fp]], [0, 1]))
            reference = value["runs"][repeat][model]
            assert [tp, fp, fn, tn] == [reference[k] for k in ["tp", "fp", "fn", "tn"]]
            close(score_value, reference["macro_f1"])
            scores.append(score_value)
        close(scores, value["summary"][model]["macro_f1"]["values"])
        close(np.mean(scores), value["summary"][model]["macro_f1"]["mean"])
        close(np.std(scores, ddof=1), value["summary"][model]["macro_f1"]["sample_sd"])
        output[model] = {
            "scores": scores,
            "mean": float(np.mean(scores)),
            "sd": float(np.std(scores, ddof=1)),
        }
    return {
        "scope": "Historical matched German 150-text DEV gate comparison; metric reconstruction from evidence item vectors, not weight or raw-log replay.",
        "item_model_seed_predictions": 900,
        "settings": 6,
        "models": output,
    }


def active_bindings():
    value = read(
        ROOT
        / "experiments/astra/sq3_full_kb_relevance_20260930_v1/api_execution_v1/plan.json"
    )
    checked = {}
    for section in ["input_hashes", "source_hashes", "generated_hashes"]:
        for path, expected in value[section].items():
            bind(ROOT / path, expected)
        checked[section] = len(value[section])
    return {
        "bindings": checked,
        "active_results_read": False,
        "source_modifications": False,
    }


def main():
    attempt = STUDY / "supplement_v1" / str(time.time_ns())
    attempt.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    (attempt / "source.py").write_bytes(Path(__file__).read_bytes())
    bind(Path(__file__))
    bind(ROOT / "analysis/common/verify_results.py")
    save(
        attempt / "plan.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "scope": __doc__,
            "source_sha256": sha(Path(__file__)),
            "tasks": [
                "global guideline metrics/template bootstrap",
                "feedback twelve tables/final transfer bootstrap/direct uptake",
                "historical mmBERT metric reconstruction",
                "active-run frozen source hashes only",
            ],
        },
    )
    try:
        with patch.object(
            socket.socket,
            "connect",
            side_effect=RuntimeError("Network forbidden during offline audit"),
        ):
            result = {
                "guideline": guideline(),
                "feedback": feedback(),
                "mmbert": mmbert(),
                "active_bindings": active_bindings(),
            }
        save(attempt / "checks.json", result)
        for path, expected in list(HASHES.items()):
            assert sha(ROOT / path) == expected, ("Input changed after audit", path)
        save(attempt / "inputs.json", HASHES)
        receipt = {
            "status": "independent_supplement_verified",
            "attempt": attempt.relative_to(ROOT).as_posix(),
            "verified_input_hashes": len(HASHES),
            "seconds": time.monotonic() - started,
            "guideline_rows": result["guideline"]["exported_rows"],
            "feedback_rows": result["feedback"]["exported_rows"],
            "historical_encoder_prediction_slots": 900,
            "feedback_metric_evaluations": result["feedback"]["metric_evaluations"],
            "bootstrap_contrasts": 13,
            "api_calls": 0,
            "cuda": False,
            "output_hashes": {
                file.name: sha(file) for file in attempt.iterdir() if file.is_file()
            },
        }
        save(attempt / "complete.json", receipt)
        print(receipt, flush=True)
    except Exception:
        save(
            attempt / "failure.json",
            {"traceback": traceback.format_exc(), "inputs": HASHES},
        )
        raise


if __name__ == "__main__":
    main()

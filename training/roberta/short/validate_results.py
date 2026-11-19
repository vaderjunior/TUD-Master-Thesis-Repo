from collections import Counter
from fractions import Fraction
import math
from pathlib import Path
import statistics
from training.roberta.short import validate as r
import numpy as np

c = r.c


def close(a, b, message):
    c.require(math.isclose(a, b, rel_tol=0, abs_tol=2e-12), message)


def quantile(values, q):
    ordered = sorted(map(float, values))
    position = (len(ordered) - 1) * q
    low = math.floor(position)
    high = math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def main():
    analysis = c.read(r.OUT / "analysis_plan.json")
    r.verify_analysis(analysis)
    integrity = c.read(r.OUT / "integrity.json")
    c.require(integrity["status"] == "pass", "Integrity missing")
    c.verify(integrity["fresh_training_artifact_bindings"])
    c.require(
        r.guard() == integrity["running_Qwen_sources_unchanged"], "Qwen source changed"
    )
    scores = c.read(r.OUT / "scores.json")
    cells = scores["cells"]
    checked, labels = 0, 0
    for cell in cells.values():
        for model, runs in cell["runs"].items():
            for run in runs:
                f1 = {}
                for label, row in run["per_label"].items():
                    tp, fp, fn = [row[k] for k in ("tp", "fp", "fn")]
                    c.require(
                        all(isinstance(v, int) and v >= 0 for v in (tp, fp, fn)),
                        "Invalid count",
                    )
                    c.require(tp + fn == row["support"], "Support arithmetic")
                    ratio = lambda a, b: float(Fraction(a, b)) if b else 0.0
                    f1[label] = ratio(2 * tp, 2 * tp + fp + fn)
                    close(f1[label], row["f1"], "Label F1")
                    close(ratio(tp, tp + fp), row["precision"], "Precision")
                    close(ratio(tp, tp + fn), row["recall"], "Recall")
                    labels += 1
                close(
                    statistics.mean(f1[l] for l in run["macro_labels"]),
                    run["macro_f1"],
                    "Macro F1",
                )
                close(
                    sum(x["tp"] + x["fp"] for x in run["per_label"].values())
                    / run["n"],
                    run["mean_predicted_labels"],
                    "Predicted cardinality",
                )
                close(
                    sum(x["support"] for x in run["per_label"].values()) / run["n"],
                    run["mean_gold_labels"],
                    "Gold cardinality",
                )
                if "hate" in run["per_label"] and "not_hate" in run["per_label"]:
                    h, nh = run["per_label"]["hate"], run["per_label"]["not_hate"]
                    close(h["fn"] / h["support"], run["false_negative_rate"], "FNR")
                    close(h["fp"] / nh["support"], run["false_positive_rate"], "FPR")
                    close(
                        (h["tp"] + nh["tp"]) / run["n"],
                        run["exact_set_accuracy"],
                        "Hate accuracy",
                    )
                checked += 1
            close(
                statistics.mean(x["macro_f1"] for x in runs),
                cell["means"][model],
                "Mean over seeds",
            )
            close(
                statistics.stdev(x["macro_f1"] for x in runs),
                cell["sample_sds"][model],
                "Sample SD",
            )
    c.require(
        checked == 975
        and len(cells) == 13
        and scores["old_reference_cells_reproduced"] == 585
        and scores["fixed_control_cells_reproduced"] == 260,
        "Expected complete result matrix",
    )
    stats = c.read(r.OUT / "statistics.json")
    intervals = 0
    with np.load(r.OUT / "bootstrap_draws.npz", allow_pickle=False) as saved:
        c.require(len(saved.files) == 12, "Bootstrap matrix inventory")
        for endpoint in r.PRIMARY:
            cell = cells["broad/" + endpoint]
            contrasts = stats["contrasts"][endpoint]
            for family in ("primary", "supporting_BGE"):
                right = "original_tuned_selected" if family == "primary" else "bge"
                for strat, result in contrasts[family].items():
                    draws = saved[endpoint + "|" + family + "|" + strat]
                    c.require(
                        len(draws) == 10000 and np.isfinite(draws).all(),
                        "Bootstrap draws",
                    )
                    close(
                        cell["means"]["expanded_tuned_selected"] - cell["means"][right],
                        result["difference"],
                        "Contrast",
                    )
                    for field, probabilities in [
                        ("ci9875", [0.00625, 0.99375]),
                        ("ci95", [0.025, 0.975]),
                    ]:
                        for got, q in zip(result[field], probabilities):
                            close(got, quantile(draws, q), "Independent quantile")
                            intervals += 1
    transitions = c.read(r.OUT / "transitions.json")["cells"]
    changed = c.read(r.OUT / "changed_cases.json")["rows"]
    counts = Counter((x["endpoint"], x["seed"]) for x in changed)
    for endpoint, runs in transitions.items():
        cell = cells["broad/" + endpoint]
        for row in runs:
            c.require(
                row["changed"]
                == row["exact_fixes"]
                + row["exact_regressions"]
                + row["changed_both_wrong"]
                == counts[endpoint, row["seed"]],
                "All changed-case accounting",
            )
            a, b = [
                next(x for x in cell["runs"][model] if x["seed"] == row["seed"])
                for model in ("expanded_tuned_selected", "original_tuned_selected")
            ]
            close(
                (row["exact_fixes"] - row["exact_regressions"]) / cell["n"],
                a["exact_set_accuracy"] - b["exact_set_accuracy"],
                "Exact-set transition arithmetic",
            )
            close(
                (row["labels_added"] - row["labels_removed"]) / cell["n"],
                a["mean_predicted_labels"] - b["mean_predicted_labels"],
                "Cardinality transition arithmetic",
            )
    diagnostic = c.read(r.OUT / "diagnostics.json")
    expected_fine = 5 * sum(
        cell["endpoint"]["dimension"] != "hate" for cell in cells.values()
    )
    c.require(
        diagnostic["fine_probability_equality_checks"] == expected_fine == 50
        and all(
            x["all_fine_label_changes_explained_by_gate"]
            for x in diagnostic["gate_only_fine_changes"].values()
        ),
        "Fine-head gate mechanism inventory",
    )
    training = c.read(r.OUT / "training_runs.json")
    resources = c.read(r.OUT / "resources.json")
    c.require(
        len(training["runs"]) == 2580
        and resources["new_search_candidates"] == 1800
        and resources["new_final_models"] == 5
        and resources["stopped_attempts"] == 0,
        "Fit inventory",
    )
    for origin, costs in training["costs_by_origin"].items():
        runs = [x for x in training["runs"] if x["origin"] == origin]
        c.require(len(runs) == costs["fits"], "Cost fits")
        for key in (
            "attempted_steps",
            "successful_steps",
            "skipped_steps",
            "positive_LR_updates",
            "train_seconds",
            "total_seconds",
        ):
            close(sum(x[key] for x in runs), costs[key], "Cost summation")
        c.require(
            costs["successful_steps"] + costs["skipped_steps"]
            == costs["attempted_steps"],
            "AMP accounting",
        )
    files = [
        p
        for p in r.OUT.iterdir()
        if p.is_file() and p.name not in ("verification.json", "complete.json")
    ]
    files += [
        Path(__file__),
        Path(r.__file__),
        r.ROOT / "training/roberta/short/results.py",
        r.ROOT / "training/roberta/short/diagnostics.py",
    ]
    r.save(
        "verification.json",
        {
            "status": "pass",
            "utc": c.utc(),
            "endpoint_seed_scores": checked,
            "verifier_inventory_fix": "First verifier reached the gate inventory check after all score/interval/case checks, then rejected an incorrect hard-coded 45. Deriving the count from 13 endpoints gives 50 fine-endpoint/seed checks (three hate endpoints). Prior verifier source retained; no scientific data changed.",
            "per_label_integer_arithmetic": labels,
            "independent_interval_endpoints": intervals,
            "case_rows": len(changed),
            "complete_case_accounting": True,
            "all_seeds_and_original_results_preserved": True,
            "fine_head_gate_only_verified": True,
            "frozen_sources_and_original_evidence_verified": True,
            "bindings": r.s.hashes(files),
            "Qwen_calls": 0,
            "new_fits_or_model_inference": 0,
            "source_sha256": c.sha(Path(__file__)),
        },
    )
    print(
        "Final independent arithmetic/evidence verification passed.",
        checked,
        "scores;",
        len(changed),
        "changed case rows.",
    )


if __name__ == "__main__":
    with r.offline():
        main()

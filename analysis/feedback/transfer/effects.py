import os

for _variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_variable] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from collections import Counter
import io
import json
from pathlib import Path
import platform
import sys
import unittest
import numpy as np
from analysis.feedback.transfer import archive as a
from analysis.feedback.transfer import effects_metrics as k
from tests.feedback.transfer import test_effects as tests

OUT = a.ROOT / "experiments/astra/sq3_transfer_part2_20261023_v1"
ENDPOINTS = [
    "en_test_FPR",
    "en_test_FNR",
    "en_hatecheck_eval_FPR",
    "en_hatecheck_eval_FNR",
]


def write(path, value):
    a.need(path.resolve().is_relative_to(OUT), "Write outside new Part 2 folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(
            value if isinstance(value, bytes) else a.encoded(value, pretty=True)
        )


def save_draws(name, values):
    path = OUT / "draws" / (name.replace("/", "__") + ".npy")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.save(stream, values, allow_pickle=False)
    return a.rel(path)


def load_matrix():
    meta = []
    predicted = np.full((15758, 4, 3), -1, dtype=np.int8)
    votes = np.zeros_like(predicted)
    for i, row in enumerate(a.rows(a.OUT / "matrix.jsonl")):
        a.need(row["input_index"] == i and i < 15758, "Matrix order/size differs")
        meta.append(
            {
                key: row[key]
                for key in (
                    "key",
                    "population",
                    "source",
                    "inference_cluster",
                    "functional_cluster",
                    "text_cluster",
                    "stage",
                    "execution_period",
                    "gold_hate",
                )
            }
        )
        for c in range(4):
            for repeat in range(3):
                result = row["predictions"][c][repeat]
                votes[i, c, repeat] = result["n_valid"]
                if result["result"] is not None:
                    a.need(type(result["result"]["hate"]) is bool, "Nonboolean gate")
                    predicted[i, c, repeat] = int(result["result"]["hate"])
        a.need(
            row["masks"]["common_valid"] == bool(np.all(predicted[i] >= 0))
            and row["masks"]["strict_all_votes"] == bool(np.all(votes[i] == 3)),
            "Part 1 availability mask differs",
        )
    a.need(
        len(meta) == len({r["key"] for r in meta}) == 15758,
        "Wrong matrix identity count",
    )
    a.need(np.array_equal(predicted < 0, votes == 0), "Null/vote mismatch")
    gold = np.asarray([r["gold_hate"] for r in meta], dtype=np.int8)
    errors = np.where(predicted < 0, -1, predicted != gold[:, None, None]).astype(
        np.int8
    )
    return meta, predicted, errors, votes


def prepare():
    a.need(not OUT.exists(), "Preserve any existing Part 2 output")
    done = a.read(a.OUT / "complete.json")
    a.need(
        done["status"] == "transfer_part1_complete_ready_for_part2",
        "Part 1 is not complete",
    )
    bindings = {a.rel(a.OUT / "complete.json"): a.sha(a.OUT / "complete.json")}
    for name in (
        "matrix.jsonl",
        "analysis_ready.json",
        "independent_matrix_check.json",
    ):
        path = a.OUT / name
        expected = done["artifacts"][a.rel(path)]
        a.need(a.sha(path) == expected, "Consumed Part 1 artifact changed")
        bindings[a.rel(path)] = expected
    for path in (a.STUDY / "scientific_plan_v1.json", a.STUDY / "STUDY_PROTOCOL.md"):
        expected = done["input_bindings"][a.rel(path)]
        a.need(a.sha(path) == expected, "Scientific contract changed")
        bindings[a.rel(path)] = expected
    for path in (
        Path(__file__),
        Path(k.__file__),
        Path(tests.__file__),
        Path(a.__file__),
    ):
        bindings[a.rel(path)] = a.sha(path)
    science = a.read(a.STUDY / "scientific_plan_v1.json")
    a.need(
        science["primary"]
        == {
            "endpoints": ENDPOINTS,
            "bootstrap_draws": 10000,
            "bootstrap_seeds": [2026101216, 2026101217, 2026101218, 2026101219],
            "confidence": 0.9875,
        },
        "Frozen primary contract differs",
    )
    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests)
    )
    a.need(result.wasSuccessful(), "Rate-kernel fixtures failed: " + log.getvalue())
    contract = {
        "status": "part2_implementation_conventions_recorded_before_scoring",
        "created_utc": a.utc(),
        "bindings": bindings,
        "primary": science["primary"],
        "conditions": science["conditions"],
        "planned_views": list(k.VIEWS),
        "replay_exclusions": science["mandatory_replay_keys"],
        "representatives": {
            "prefix": k.REP_PREFIX,
            "selection": "Minimum SHA256(prefix + key) within endpoint-specific eligible inference cluster, after any predefined replay exclusions and before checking output validity.",
        },
        "bootstrap": {
            "generator": "PCG64",
            "draws": 10000,
            "unit_order": "Sorted inference-cluster IDs (sorted keys for item sensitivity).",
            "draw_columns": k.DRAW_COLUMNS,
            "quantiles": "NumPy linear percentile interpolation",
            "weighting": "Each resample uses complete clusters and item-weighted rates; all 12 saved outputs remain together. Mean of three repeats within each pool, then equal mean of the three pools; reference counted once.",
            "support_seeds": "First 8 SHA256 bytes, little endian, prefix A16-part2-support-v1| plus named comparison; separate deterministic streams for source signatures.",
            "source_strata": "Whole clusters stratified by source-membership signature among endpoint-eligible rows before validity. Mixed-source components remain intact.",
        },
        "additional_draw_sets": "Natural source-stratified and HateCheck item-level sensitivity on main, strict and replay-excluded views; six natural-source supporting views on primary common masks.",
        "interval_roles": "Four primary 98.75% intervals. Six mask views retain 98.75% for comparison. Alternative resampling and source/pool summaries use supporting 95%, with 98.75% diagnostics also saved. No new discovery family.",
        "missing_bounds": "Oppose unknown treatment/reference errors on the whole eligible cohort; strict sensitivity treats outputs with fewer than three valid votes as unknown. These bounds are not confidence intervals.",
        "omissions": "Leave each pool out both on the fixed primary mask and with validity recomputed for retained arms; leave each execution batch out. Point estimates, not intervals.",
        "scope": "Four error-rate endpoints, their mandatory sensitivity checks, natural-source overview, run/pool and descriptive error-transition counts. Detailed HateCheck functions, complete-answer quality, context, cases, broader robustness, classifier comparison and synthesis remain later parts.",
        "outcomes_used_to_choose_conventions": False,
        "new_Qwen_calls": 0,
        "new_fits": 0,
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
        },
    }
    write(OUT / "analysis_contract.json", contract)
    write(
        OUT / "focused_tests.json",
        {"passed": True, "tests_run": result.testsRun, "log": log.getvalue()},
    )
    return science, contract


def calculate():
    science, contract = prepare()
    meta, predicted, error, votes = load_matrix()
    ready = a.read(a.OUT / "analysis_ready.json")
    conditions = science["conditions"]
    all_results, descriptions, omissions, groups = {}, {}, {}, {}
    records = []

    def record(
        name, endpoint, c, *, view, seed, mode="cluster", source=None, confidence=0.9875
    ):
        population, label = endpoint.rsplit("_", 1)
        strict, reps, excluded = k.VIEWS[view]
        values, boot = k.bootstrap(meta, error, c, seed, mode=mode)
        selected = {
            "population": population,
            "gold_hate": label == "FNR",
            "view": view,
            "strict": strict,
            "representatives": reps,
            "excluded_keys": science["mandatory_replay_keys"] if excluded else [],
            "source_filter": source,
        }
        result = {
            "id": name,
            "endpoint": endpoint,
            "selection": selected,
            "resampling_mode": mode,
            **c["point"],
            **k.intervals(values),
            "interval_confidence_for_this_view": confidence,
            "bootstrap": boot,
            "draw_file": save_draws(name, values),
            "eligible_indices": c["eligible"].tolist(),
            "common_indices": c["common"].tolist(),
            "status": "calculated_independent_verification_pending",
        }
        all_results[name] = result
        records.append(
            {
                "id": name,
                "endpoint": endpoint,
                "view": view,
                "mode": mode,
                "source": source,
            }
        )
        print(
            f"Calculated {name}: n={len(c['common']):,}, units={boot['units']:,}",
            flush=True,
        )
        return result

    for ei, endpoint in enumerate(ENDPOINTS):
        population, label = endpoint.rsplit("_", 1)
        truth = label == "FNR"
        for view, (strict, representatives, excluded) in k.VIEWS.items():
            kwargs = {
                "strict": strict,
                "representatives": representatives,
                "excluded": science["mandatory_replay_keys"] if excluded else [],
            }
            c = k.cohort(meta, error, votes, population, truth, **kwargs)
            if view in ("main", "strict", "replay_excluded", "strict_replay_excluded"):
                mask_name = {
                    "main": "common_valid",
                    "strict": "strict_all_votes",
                    "replay_excluded": "common_valid_exclude_replay",
                    "strict_replay_excluded": "strict_exclude_replay",
                }[view]
                a.need(
                    len(c["common"]) == ready["endpoint_masks"][endpoint][mask_name],
                    "Prechecked endpoint mask differs",
                )
            record(
                view + "/" + endpoint,
                endpoint,
                c,
                view=view,
                seed=science["primary"]["bootstrap_seeds"][ei],
            )
            omitted = {
                "pools_fixed_mask": [],
                "pools_recomputed_mask": [],
                "execution_batches": [],
            }
            for pool in range(1, 4):
                arms = tuple(i for i in range(1, 4) if i != pool)
                for fixed, field in (
                    (True, "pools_fixed_mask"),
                    (False, "pools_recomputed_mask"),
                ):
                    cp = k.cohort(
                        meta,
                        error,
                        votes,
                        population,
                        truth,
                        arms=arms,
                        eligible_override=c["common"] if fixed else None,
                        **kwargs,
                    )
                    omitted[field].append(
                        {"omitted_condition": conditions[pool], **cp["point"]}
                    )
            for batch in sorted({meta[i]["stage"] for i in c["eligible"]}):

                remaining = [i for i in c["eligible"] if meta[i]["stage"] != batch]
                cp = k.cohort(
                    meta,
                    error,
                    votes,
                    population,
                    truth,
                    eligible_override=remaining,
                    **kwargs,
                )
                omitted["execution_batches"].append(
                    {"omitted_batch": batch, **cp["point"]}
                )
            omissions[view + "/" + endpoint] = omitted
            if view in ("main", "strict", "replay_excluded"):
                mode = "source_stratified" if population == "en_test" else "item"
                name = mode + "/" + view + "/" + endpoint
                record(
                    name,
                    endpoint,
                    c,
                    view=view,
                    seed=k.seed_for(name),
                    mode=mode,
                    confidence=0.95,
                )
            if view == "main":
                descriptive, _ = k.description(meta, error, votes, c, conditions)
                descriptions[endpoint] = descriptive
                breakdown = {
                    "by_source": [],
                    "by_batch": [],
                    "by_execution_period": [],
                    "own_valid_runs": [],
                }
                for field, label_name in [
                    ("source", "by_source"),
                    ("stage", "by_batch"),
                    ("execution_period", "by_execution_period"),
                ]:
                    for value in sorted({meta[i][field] for i in c["eligible"]}):
                        ids = [i for i in c["eligible"] if meta[i][field] == value]
                        cg = k.cohort(
                            meta, error, votes, population, truth, eligible_override=ids
                        )
                        breakdown[label_name].append({"group": value, **cg["point"]})
                        if field == "source" and population == "en_test":
                            name = "source/" + value + "/" + endpoint
                            record(
                                name,
                                endpoint,
                                cg,
                                view="main",
                                seed=k.seed_for(name),
                                source=value,
                                confidence=0.95,
                            )
                for j, condition in enumerate(conditions):
                    for repeat in range(3):
                        e = error[c["eligible"], j, repeat]
                        observed = e[e >= 0]
                        breakdown["own_valid_runs"].append(
                            {
                                "condition": condition,
                                "repeat": repeat,
                                "eligible": len(e),
                                "valid": len(observed),
                                "null": int((e < 0).sum()),
                                "errors": int(observed.sum()),
                                "rate": (
                                    float(observed.mean()) if len(observed) else None
                                ),
                            }
                        )
                groups[endpoint] = breakdown
    a.need(len(all_results) == 42, "Incomplete planned Part 2 result inventory")
    path = OUT / "per_input_error_ledger.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        for i, r in enumerate(meta):
            common = bool(np.all(error[i] >= 0))
            sums = error[i].sum(axis=1)
            delta = int(sums[1:].sum() - 3 * sums[0]) if common else None
            stream.write(
                a.encoded(
                    {
                        **r,
                        "input_index": i,
                        "predicted_hate": predicted[i].tolist(),
                        "errors": error[i].tolist(),
                        "valid_votes": votes[i].tolist(),
                        "in_primary_common_mask": common,
                        "strict_all_votes": bool(np.all(votes[i] == 3)),
                        "difference_ninths": delta,
                        "direction": k.direction(delta) if common else "missing_output",
                        "missing_sentinel": -1,
                    }
                )
            )
    write(OUT / "rates.json", all_results)
    write(OUT / "descriptive_errors.json", descriptions)
    write(OUT / "omissions.json", omissions)
    write(OUT / "group_rates.json", groups)
    write(OUT / "comparison_inventory.json", records)
    for name, expected in contract["bindings"].items():
        a.need(
            a.sha(a.inside(a.ROOT, name)) == expected,
            "Consumed evidence/source changed while scoring",
        )
    artifacts = {a.rel(p): a.sha(p) for p in OUT.rglob("*") if p.is_file()}
    write(
        OUT / "calculated.json",
        {
            "status": "part2_calculated_independent_verification_pending",
            "completed_utc": a.utc(),
            "bindings": contract["bindings"],
            "artifacts": artifacts,
            "bootstrap_sets": len(all_results),
            "draws_per_set": 10000,
            "columns_per_draw": 6,
            "new_Qwen_calls": 0,
            "new_fits": 0,
            "scope": "Only the four error-rate endpoints and their planned sensitivity/descriptive checks; later review parts remain open.",
        },
    )
    for endpoint in ENDPOINTS:
        r = all_results["main/" + endpoint]
        print(
            json.dumps(
                {
                    key: r[key]
                    for key in (
                        "endpoint",
                        "common_n",
                        "reference_rate",
                        "treatment_rate",
                        "difference",
                        "interval98_75",
                        "difference_missing_bounds",
                    )
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    with a.offline():
        calculate()

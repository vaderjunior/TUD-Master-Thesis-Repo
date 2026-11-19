from src.hsrag.paths import PROJECT_ROOT
import os

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/sq3_linear_update_20261012_v1"
OUT = ROOT / "experiments/astra/sq3_linear_static_review_20261014_v1"
CPU = ROOT / "experiments/astra/sq2_cpu_baselines_20261012_v2"
CANDIDATES = ROOT / "experiments/astra/sq3_additions_20261004_v1/benign_selection"
A15 = ROOT / "experiments/astra/sq3_reserved_slot_20261012_v1"


def require(value, message):
    if not value:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def child(root, name):
    path = (root / name).resolve()
    require(path.is_relative_to(root.resolve()), "Escaping evidence path")
    return path


def rows(path):
    with Path(path).open(encoding="utf8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def model(path):
    with np.load(path, allow_pickle=False) as data:
        result = {name: data[name] for name in data.files}
    require(
        set(result) == {"coef", "intercept", "constants", "iterations"},
        "Model fields differ",
    )
    require(result["coef"].shape == (1, 1024), "Coefficient shape differs")
    require(
        all(result[k].shape == (1,) for k in ("intercept", "constants", "iterations")),
        "Head shape differs",
    )
    require(all(np.isfinite(v).all() for v in result.values()), "Nonfinite model")
    require(result["constants"].tolist() == [-1], "Unexpected constant model")
    require(
        0 < int(result["iterations"][0]) < 2000,
        "Solver did not finish before iteration limit",
    )
    return result


def review():
    require(not (OUT / "review.json").exists(), "Preserve existing review receipt")
    begun = datetime.now(timezone.utc).isoformat()
    bindings = {}

    def bind(path):
        digest = sha(path)
        bindings[rel(path)] = digest
        return digest

    def verify(mapping, root=ROOT):
        for name, expected in mapping.items():
            require(bind(child(root, name)) == expected, "Changed evidence: " + name)

    plan = read(BASE / "plan.json")
    plan_hash = bind(BASE / "plan.json")
    require(
        read(BASE / "ready.json")["plan_sha256"] == plan_hash,
        "Readiness binding differs",
    )
    bind(BASE / "ready.json")
    for name in ("inputs", "outputs", "source_hashes"):
        verify(plan[name])
    environment = {
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "scipy", "scikit-learn", "pyarrow", "threadpoolctl")
        },
    }
    require(environment == plan["environment"], "Numeric environment differs")
    require(
        (plan["C"], plan["cutoff"], plan["seed"], plan["cutoff_operator"])
        == (10.0, 0.6, 42, ">"),
        "Recipe differs",
    )
    require(plan["principal_variant"] == "all_permitted", "Principal variant differs")
    active_plan = read(A15 / "execution/plan.json")
    verify(active_plan["source_hashes"])
    require(
        bind(A15 / "execution/plan.json")
        == read(A15 / "execution/ready.json")["plan_sha256"],
        "A15 plan differs",
    )
    bind(A15 / "execution/ready.json")

    done = read(BASE / "static_complete.json")
    bind(BASE / "static_complete.json")
    require(
        done["status"] == "classifier_updates_verified_scientific_review_pending",
        "Unexpected completion status",
    )
    require(
        done["plan_sha256"] == plan_hash
        and done["new_Qwen_calls"] == 0
        and done["CUDA"] is False,
        "Completion identity differs",
    )
    require(
        done["source_bindings"] == {}, "Unexpected dynamic dependency in static block"
    )
    require(
        bind(BASE / "static_schedule.json") == done["schedule_sha256"],
        "Schedule changed",
    )
    schedule = read(BASE / "static_schedule.json")
    data = read(BASE / "data.json")
    entries = [{"condition": "reference", "ids": []}] + [
        {"condition": p["condition"], "ids": p["ids"]} for p in data["static_pools"]
    ]
    require(
        schedule
        == {
            "mode": "static",
            "entries": entries,
            "source_bindings": {},
            "plan_sha256": plan_hash,
        },
        "Static schedule differs",
    )
    frozen_pools = read(
        ROOT
        / "experiments/astra/sq3_additions_results_20261006_v1/benign_selection/actual_pools.json"
    )["pools"]
    require(
        data["static_pools"] == frozen_pools and len(frozen_pools) == 18,
        "Correction pools differ",
    )
    expected_names = {
        v + "/" + e["condition"]
        for v in ("all_permitted", "matched_gradient")
        for e in entries
    }
    require(
        len(expected_names) == 38
        and set(done["jobs"]) == expected_names
        and done["logical_jobs"] == 38,
        "Incomplete job family",
    )

    roster = rows(BASE / "roster.jsonl")
    require(
        len(roster) == len({r["key"] for r in roster}) == 16433,
        "Evaluation roster differs",
    )
    old_rows = read(CPU / "data/evaluation.json")
    indices = data["evaluation_indices"]
    require(
        len(indices) == len(set(indices)) == len(roster), "Evaluation indices differ"
    )
    for record, index in zip(roster, indices):
        old = old_rows[index]
        require(
            record["key"] == old["key"] and record["text"] == old["text"],
            "Evaluation row identity differs",
        )
        require(
            hashlib.sha256(record["text"].encode("utf8")).hexdigest()
            == old["exact_text_sha256"],
            "Evaluation text differs",
        )
    del old_rows
    vectors = np.load(
        CPU / "data/evaluation_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    candidates = rows(CANDIDATES / "roster.jsonl")
    labels = {r["id"]: int(r["gold"]["hate"]) for r in candidates}
    require(
        data["candidate_ids"] == [r["id"] for r in candidates],
        "Candidate order differs",
    )
    require(
        data["candidate_labels"] == [labels[r["id"]] for r in candidates],
        "Candidate targets differ",
    )
    verified = {}
    aliases = {}
    for variant in ("all_permitted", "matched_gradient"):
        fit = data["partitions"][variant]
        require(
            len(fit) == (55753 if variant == "all_permitted" else 50173),
            "Training partition size differs",
        )
        train_ids = {r["source_id"] for r in fit}
        base_positive = sum(r["target"] for r in fit)
        for entry in entries:
            name = variant + "/" + entry["condition"]
            source_ids = sorted(entry["ids"])
            require(
                len(source_ids)
                == len(set(source_ids))
                == (0 if entry["condition"] == "reference" else 40),
                "Update budget differs",
            )
            require(
                set(source_ids) <= set(labels) and not (set(source_ids) & train_ids),
                "Invalid correction membership",
            )
            job = done["jobs"][name]
            require(job["source_ids"] == source_ids, "Completion source IDs differ")
            receipt_path = child(ROOT, job["receipt"])
            require(
                receipt_path.is_relative_to((BASE / "jobs" / variant).resolve()),
                "Wrong job root",
            )
            require(bind(receipt_path) == job["sha256"], "Job receipt changed")
            identity = {
                "plan_sha256": plan_hash,
                "variant": variant,
                "source_ids": source_ids,
                "C": 10.0,
                "seed": 42,
            }
            identity_hash = hashlib.sha256(
                (
                    json.dumps(
                        identity,
                        sort_keys=True,
                        ensure_ascii=False,
                        indent=2,
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf8")
            ).hexdigest()
            require(
                receipt_path.parent.name == identity_hash[:16],
                "Job path identity differs",
            )
            aliases.setdefault(rel(receipt_path), []).append(name)
            if rel(receipt_path) in verified:
                continue
            receipt = read(receipt_path)
            require(receipt["identity"] == identity, "Job identity differs")
            expected_detail = {
                "rows": 16433,
                "source_ids": source_ids,
                "baseline_reproduced": not source_ids,
                "no_Qwen_or_CUDA": True,
            }
            require(
                receipt["detail"] == expected_detail, "Job completion detail differs"
            )
            attempt = child(receipt_path.parent, receipt["attempt"])
            expected_files = {
                "started.json",
                "attribution.json",
                "model.npz",
                "probabilities.npy",
                "decisions_05.npy",
                "decisions_06.npy",
            }
            require(
                set(receipt["outputs"])
                == {receipt["attempt"] + "/" + f for f in expected_files},
                "Unexpected job artifact inventory",
            )
            verify(receipt["outputs"], receipt_path.parent)
            require(
                read(attempt / "started.json")["identity"] == identity,
                "Start identity differs",
            )
            attribution = read(attempt / "attribution.json")
            n = len(fit) + len(source_ids)
            positives = base_positive + sum(labels[i] for i in source_ids)
            require(
                attribution["identity"] == identity
                and attribution["source_ids"] == source_ids,
                "Attribution identity differs",
            )
            require(
                (
                    attribution["fit_rows"],
                    attribution["positive"],
                    attribution["negative"],
                )
                == (n, positives, n - positives),
                "Training class counts differ",
            )
            require(
                attribution["class_weight_recomputed"] is True
                and attribution["fine_heads_changed"] is False,
                "Update policy differs",
            )
            require(
                attribution["baseline_reproduced"] == (not source_ids)
                and attribution["probability_rows"] == 16433,
                "Prediction attribution differs",
            )
            weights = model(attempt / "model.npz")
            require(
                attribution["iterations"] == weights["iterations"].tolist(),
                "Iteration attribution differs",
            )
            prob = np.load(attempt / "probabilities.npy", allow_pickle=False)
            require(
                prob.shape == (16433, 1) and prob.dtype == np.float64,
                "Probability shape/type differs",
            )
            require(
                np.isfinite(prob).all() and np.all((prob >= 0) & (prob <= 1)),
                "Invalid probabilities",
            )
            for cutoff, suffix in ((0.5, "05"), (0.6, "06")):
                decision = np.load(
                    attempt / ("decisions_" + suffix + ".npy"), allow_pickle=False
                )
                require(
                    decision.dtype == np.uint8
                    and np.array_equal(decision, (prob > cutoff).astype(np.uint8)),
                    "Saved cutoff decisions differ",
                )
            error = 0.0
            for start in range(0, len(indices), 128):
                xx = np.asarray(vectors[indices[start : start + 128]], dtype=np.float64)

                z = (
                    np.add.reduce(xx * weights["coef"][0], axis=1)
                    + weights["intercept"][0]
                )
                result = np.empty_like(z)
                positive = z >= 0
                result[positive] = 1 / (1 + np.exp(-z[positive]))
                exponent = np.exp(z[~positive])
                result[~positive] = exponent / (1 + exponent)
                saved = prob[start : start + len(result), 0]
                error = max(error, float(np.max(np.abs(result - saved))))
                require(
                    np.allclose(result, saved, rtol=0, atol=1e-12),
                    "Independent probability reconstruction differs",
                )
                require(
                    all(np.array_equal(result > c, saved > c) for c in (0.5, 0.6)),
                    "Independent decisions differ",
                )
            baseline_error = None
            if not source_ids:
                old = data["baseline_head"]["variants"][variant]
                prior = model(ROOT / old["model"])
                require(
                    all(
                        np.allclose(weights[k], prior[k], rtol=0, atol=1e-10)
                        for k in weights
                    ),
                    "Original baseline weights differ",
                )
                saved = np.load(
                    ROOT / old["probabilities"], mmap_mode="r", allow_pickle=False
                )[indices]
                baseline_error = float(np.max(np.abs(prob - saved)))
                require(
                    np.allclose(prob, saved, rtol=0, atol=1e-12),
                    "Original baseline probabilities differ",
                )
                require(
                    all(np.array_equal(prob > c, saved > c) for c in (0.5, 0.6)),
                    "Original baseline decisions differ",
                )
            times = {
                k: attribution[k]
                for k in (
                    "feature_seconds",
                    "fit_seconds",
                    "inference_and_verification_seconds",
                )
            }
            require(
                all(np.isfinite(v) and v >= 0 for v in times.values()),
                "Invalid recorded duration",
            )
            verified[rel(receipt_path)] = {
                "variant": variant,
                "source_ids": source_ids,
                "fit_rows": n,
                "positive": positives,
                "negative": n - positives,
                "class_weights": {
                    "negative": n / (2 * (n - positives)),
                    "positive": n / (2 * positives),
                },
                "iterations": int(weights["iterations"][0]),
                "probability_rows": len(prob),
                "independent_max_abs_difference": error,
                "baseline_max_abs_difference": baseline_error,
                "baseline_reproduced": not source_ids,
                "timing_seconds": times,
                "seconds_this_attempt": receipt["seconds_this_attempt"],
                "completed_utc": receipt["utc"],
            }
            print(
                "Independently verified A18 static job", len(verified), name, flush=True
            )
    require(len(verified) == done["unique_fits"], "Unique fit count differs")
    resources = []
    resource_identities = []
    for path in sorted((BASE / "resource_checks").glob("*/resources.json")):
        resource = read(path)
        bind(path)
        bind(path.parent / "worker.log")
        require(
            resource["exit_code"] == 0 and resource["stop_reason"] is None,
            "Resource-monitored job failed; review retained evidence",
        )
        require(
            resource["originals_or_unrelated_processes_terminated"] is False,
            "Unrelated process intervention reported",
        )
        require(
            resource["sampled_min_available_GiB"] >= 1.25
            and resource["sampled_peak_worker_tree_private_GiB"] <= 2.6,
            "Recorded resource guard exceeded",
        )
        command = resource["command"]
        require(
            "scratch.analysis.astra_sq3_linear_update_v1" in command
            and "worker" in command,
            "Unexpected monitored command",
        )
        v = command[command.index("--variant") + 1]
        ids = command[command.index("--ids") + 1]
        resource_identities.append((v, ids))
        resources.append(resource)
    expected_resource_ids = [
        (v["variant"], ",".join(v["source_ids"])) for v in verified.values()
    ]
    require(
        sorted(resource_identities) == sorted(expected_resource_ids),
        "Monitored worker/unique-fit inventory differs",
    )
    bind(Path(__file__))
    result = {
        "status": "static_numerical_outputs_independently_verified_paired_scientific_analysis_pending",
        "started_utc": begun,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "plan_sha256": plan_hash,
        "logical_jobs": 38,
        "unique_fits": len(verified),
        "baseline_reproductions": sum(
            v["baseline_reproduced"] for v in verified.values()
        ),
        "updated_fits": sum(not v["baseline_reproduced"] for v in verified.values()),
        "probabilities_independently_reconstructed": len(verified) * 16433,
        "saved_decisions_independently_reconstructed": len(verified) * 16433 * 2,
        "max_abs_probability_difference": max(
            v["independent_max_abs_difference"] for v in verified.values()
        ),
        "numeric_environment": environment,
        "jobs": verified,
        "aliases": aliases,
        "resources": {
            "workers": len(resources),
            "min_available_GiB": min(r["sampled_min_available_GiB"] for r in resources),
            "peak_worker_tree_private_GiB": max(
                r["sampled_peak_worker_tree_private_GiB"] for r in resources
            ),
            "monitored_seconds": sum(r["seconds"] for r in resources),
            "fit_seconds": sum(
                v["timing_seconds"]["fit_seconds"] for v in verified.values()
            ),
            "sampled_only": True,
            "unrelated_processes_terminated": False,
        },
        "bindings": bindings,
        "fits_performed_by_review": 0,
        "API_calls_by_review": 0,
        "A15_prediction_records_read": 0,
        "outcome_scores_computed": False,
        "limits": [
            "This reviews static model/output integrity and inference arithmetic, not a second refit.",
            "A16/A17 RAG outcomes and A17 cumulative classifier updates remain required.",
            "No classifier/RAG effectiveness or cost-superiority conclusion is made.",
            "Recorded concurrent laptop timing and sampled resources are not universal performance estimates.",
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "review.json").open("x", encoding="utf8", newline="\n") as stream:
        json.dump(
            result,
            stream,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        stream.write("\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "status",
                    "logical_jobs",
                    "unique_fits",
                    "probabilities_independently_reconstructed",
                    "saved_decisions_independently_reconstructed",
                    "max_abs_probability_difference",
                    "resources",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        review()

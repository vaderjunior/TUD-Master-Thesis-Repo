from collections import Counter
from fractions import Fraction
import hashlib
import math
import os
from pathlib import Path
import re
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

from analysis.feedback.rounds import reader as io

SCORE = io.OUT / "errors_v1"
CONTRACT = io.OUT / "analysis_contract_v3.json"
POLICIES = ["reference", "adaptive", "static", "random"]


def near(actual, expected, message):
    io.need(math.isfinite(actual) and abs(actual - expected) < 1e-13, message)


def eligible_indices(rows, rec):
    result = []
    for i, row in enumerate(rows):
        if row["population"] != rec["population"] or row["gold_hate"] != (
            rec["endpoint"] == "FNR"
        ):
            continue
        rounds = (
            row["predictions"]
            if rec["mask"] == "all_round_common"
            else [row["predictions"][rec["round"] - 1]]
        )
        valid = all(p["result"] is not None for rr in rounds for arm in rr for p in arm)
        if rec["mask"] == "round_strict":
            valid = valid and all(
                p["n_valid"] == 3 for rr in rounds for arm in rr for p in arm
            )
        if valid:
            result.append(i)
    return result


def define(name):

    rounds, policy = name.split("_", 1)
    rr = [int(x.removeprefix("r")) for x in rounds.split("-")]
    pp = policy.split("-")
    return [
        {
            "round": rnd,
            "policy": p,
            "weight": (1 if ri == 0 else -1) * (1 if pi == 0 else -1),
        }
        for ri, rnd in enumerate(rr)
        for pi, p in enumerate(pp)
    ]


def per_input(row, terms):
    total = 0
    for t in terms:
        selected = (
            [0]
            if t["policy"] == "reference"
            else [
                i
                for i, c in enumerate(io.CONDITIONS)
                if c.startswith(t["policy"] + "_s")
            ]
        )
        predictions = [
            v["result"] for i in selected for v in row["predictions"][t["round"] - 1][i]
        ]
        io.need(
            all(p is not None for p in predictions),
            "Contrast admits a missing prediction",
        )
        errors = sum(p["hate"] != row["gold_hate"] for p in predictions)
        total += t["weight"] * errors * (3 if t["policy"] == "reference" else 1)
    return total


def linear_quantile(values, q):
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    fraction = index - lower
    return (
        ordered[lower] * (1 - fraction)
        + ordered[min(lower + 1, len(ordered) - 1)] * fraction
    )


def verify():
    started = time.monotonic()
    complete = io.read(SCORE / "complete.json")
    io.need(
        io.sha(SCORE / "results.json") == complete["results_sha256"],
        "Scored results changed",
    )
    result = io.read(SCORE / "results.json")
    for name, digest in result["inputs"].items():
        io.need(io.sha(io.ROOT / name) == digest, "Scoring input changed")
    contract = io.read(CONTRACT)
    for name, digest in contract["source_hashes"].items():
        io.need(
            io.sha(io.ROOT / name) == digest, "Source changed after scoring contract"
        )
    rows = list(io.rows(io.READER / "matrix.jsonl"))
    primary_names = [
        f"primary_en_test_{e}_adaptive-{p}"
        for p in ["static", "random"]
        for e in ["FPR", "FNR"]
    ]
    expected_names = set(primary_names)
    for pop in ["en_test", "en_hatecheck_eval"]:
        for endpoint in ["FPR", "FNR"]:
            expected_names.update(f"round_{pop}_{endpoint}_r{r}" for r in range(1, 5))
            expected_names.add(f"trajectory_{pop}_{endpoint}")
            expected_names.add(f"strict_{pop}_{endpoint}_r4")
    io.need(
        set(result["records"]) == expected_names and len(expected_names) == 28,
        "Wrong inferential record family",
    )
    scalar_checks, exact_points = 0, 0
    max_error = 0.0
    for name, rec in result["records"].items():
        indices = eligible_indices(rows, rec)
        io.need(
            indices == rec["input_indices"] and len(indices) == rec["inputs"],
            "Independent cohort differs",
        )
        if name in primary_names:
            io.need(
                rec["seed"] == 2026101220 + primary_names.index(name)
                and rec["interval_level"] == 0.9875,
                "Wrong primary seed or family interval",
            )
        else:
            seed = int.from_bytes(
                hashlib.sha256(("A17-part2-support-v1|" + name).encode()).digest()[:8],
                "little",
            )
            io.need(
                rec["seed"] == seed and rec["interval_level"] == 0.95,
                "Wrong supporting seed or interval",
            )
        for col, terms in rec["columns"].items():
            io.need(
                terms == define(col), "Contrast specification differs from column name"
            )
        order = rec["column_order"]
        io.need(
            len(order) == len(set(order)) and set(order) == set(rec["columns"]),
            "Wrong saved draw column order",
        )
        values = np.array(
            [
                [per_input(rows[i], rec["columns"][col]) for col in order]
                for i in indices
            ],
            dtype=np.int64,
        )
        for j, col in enumerate(order):
            numerator = sum(int(v) for v in values[:, j])
            near(
                rec["points"][col],
                float(Fraction(numerator, 9 * len(indices))),
                "Exact point differs",
            )
            io.need(
                rec["integer_error_numerators"][col] == numerator
                and rec["integer_denominator"] == 9 * len(indices),
                "Integer estimand differs",
            )
            exact_points += 1
        cluster_ids = [rows[i]["inference_cluster"] for i in indices]
        labels = sorted(set(cluster_ids))
        lookup = {c: i for i, c in enumerate(labels)}
        membership = np.array([lookup[c] for c in cluster_ids], dtype=np.int64)
        io.need(
            rec["cluster_labels"] == labels
            and rec["inference_clusters"] == len(labels),
            "Cluster set differs",
        )
        io.need(
            rec["cluster_sizes"] == [cluster_ids.count(c) for c in labels],
            "Cluster sizes differ",
        )
        io.need(
            io.sha(io.ROOT / rec["draw_file"]) == rec["draw_sha256"],
            "Draw file changed",
        )
        saved = np.load(io.ROOT / rec["draw_file"], allow_pickle=False)
        io.need(saved.shape == (10000, len(rec["columns"])), "Wrong draw matrix")
        rng = np.random.Generator(np.random.PCG64(rec["seed"]))
        for first in range(0, 10000, 32):
            last = min(first + 32, 10000)
            chosen = rng.integers(0, len(labels), size=(last - first, len(labels)))
            row_weights = np.stack(
                [
                    np.bincount(draw, minlength=len(labels))[membership]
                    for draw in chosen
                ]
            )
            independently_weighted = (row_weights @ values) / (
                9 * row_weights.sum(axis=1)
            )[:, None]
            error = float(np.max(np.abs(independently_weighted - saved[first:last])))
            io.need(error < 1e-13, "Independent text-weighted bootstrap differs")
            max_error = max(max_error, error)
            scalar_checks += independently_weighted.size
        alpha = (1 - rec["interval_level"]) / 2
        for j, col in enumerate(order):
            for side, q in enumerate([alpha, 1 - alpha]):
                near(
                    rec["intervals"][col][side],
                    linear_quantile(saved[:, j].tolist(), q),
                    "Independent percentile differs",
                )
        print("Independent numeric check passed: " + name, flush=True)
    for name, bound in result["full_roster_missing_bounds"].items():
        population, endpoint, rnd = name.rsplit("_", 2)
        round_index = int(rnd[1:]) - 1
        eligible = [
            r
            for r in rows
            if r["population"] == population and r["gold_hate"] == (endpoint == "FNR")
        ]
        endpoints = {}
        for policy in POLICIES:
            selected = (
                [0]
                if policy == "reference"
                else [
                    i
                    for i, c in enumerate(io.CONDITIONS)
                    if c.startswith(policy + "_s")
                ]
            )
            predictions = [
                v["result"]
                for r in eligible
                for c in selected
                for v in r["predictions"][round_index][c]
            ]
            errors, absent = 0, 0
            for r in eligible:
                for c in selected:
                    for v in r["predictions"][round_index][c]:
                        absent += v["result"] is None
                        errors += (
                            v["result"] is not None
                            and v["result"]["hate"] != r["gold_hate"]
                        )
            denominator = len(predictions)
            lower, upper = Fraction(errors, denominator), Fraction(
                errors + absent, denominator
            )
            endpoints[policy] = lower, upper
            item = bound["policies"][policy]
            factor = 3 if policy == "reference" else 1
            io.need(
                item["error_numerator_over_nine"] == errors * factor
                and item["missing_numerator_over_nine"] == absent * factor
                and item["denominator_over_nine"] == denominator * factor,
                "Full-roster raw counts differ",
            )
            near(
                item["available_case_rate"],
                errors / (denominator - absent),
                "Available-case denominator differs",
            )
            for actual, expect in zip(item["bound"], [lower, upper]):
                near(actual, float(expect), "Policy missing bound differs")
        for name, actual in bound["contrasts"].items():
            left, right = name.split("-")
            for x, y in zip(
                actual,
                [
                    endpoints[left][0] - endpoints[right][1],
                    endpoints[left][1] - endpoints[right][0],
                ],
            ):
                near(x, float(y), "Opposing missing bound differs")
    io.need(
        len(result["stream_repeat_cells"]) == 480 and len(result["missingness"]) == 160,
        "Wrong descriptive axes",
    )
    for cell in result["stream_repeat_cells"]:
        indices = eligible_indices(rows, dict(cell, mask="all_round_common"))
        condition = io.CONDITIONS.index(cell["condition"])
        count = sum(
            rows[i]["predictions"][cell["round"] - 1][condition][
                cell["decoder_repeat"]
            ]["result"]["hate"]
            != rows[i]["gold_hate"]
            for i in indices
        )
        io.need(
            cell["errors"] == count and cell["inputs"] == len(indices),
            "Stream/repeat count differs",
        )
        near(cell["rate"], count / len(indices), "Stream/repeat rate differs")
    for item in result["missingness"]:
        selected = [
            r
            for r in rows
            if r["population"] == item["population"]
            and r["gold_hate"] == (item["endpoint"] == "FNR")
        ]
        c = io.CONDITIONS.index(item["condition"])
        vote_counts = Counter(
            v["n_valid"]
            for r in selected
            for v in r["predictions"][item["round"] - 1][c]
        )
        io.need(
            item["inputs"] == len(selected)
            and item["prediction_slots"] == 3 * len(selected),
            "Missingness denominator differs",
        )
        for n, field in enumerate(["nulls", "one_vote", "two_votes", "three_votes"]):
            io.need(item[field] == vote_counts[n], "Missingness count differs")
    verification = {
        "status": "independent_four_round_numerical_verification_pass",
        "created_utc": io.pure.utc(),
        "results_sha256": io.sha(SCORE / "results.json"),
        "contract_sha256": io.sha(CONTRACT),
        "verifier_sha256": io.sha(Path(__file__)),
        "records": len(result["records"]),
        "exact_fraction_points": exact_points,
        "bootstrap_scalars_recomputed": scalar_checks,
        "maximum_bootstrap_difference": max_error,
        "full_roster_bound_records": len(result["full_roster_missing_bounds"]),
        "stream_repeat_cells": 480,
        "missingness_cells": 160,
        "elapsed_seconds": time.monotonic() - started,
        "new_Qwen_calls": 0,
        "new_fits": 0,
    }
    io.write(SCORE / "independent_verification.json", verification)
    print(verification, flush=True)


if __name__ == "__main__":
    with io.pure.offline():
        verify()

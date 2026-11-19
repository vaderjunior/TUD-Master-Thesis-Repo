from __future__ import annotations
import argparse
from collections import Counter
import os
from pathlib import Path
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np

from analysis.feedback.rounds import reader as io
from analysis.feedback.rounds import errors_kernel as k

SCORE = io.OUT / "errors_v1"
CONTRACT = io.OUT / "analysis_contract_v3.json"
PRIMARY = [
    ("static", "FPR", 2026101220),
    ("static", "FNR", 2026101221),
    ("random", "FPR", 2026101222),
    ("random", "FNR", 2026101223),
]


def freeze():
    files = [
        Path(__file__),
        Path(k.__file__),
        Path(io.__file__),
        io.ROOT / "analysis/feedback/rounds/validate_reader.py",
        io.ROOT / "analysis/feedback/rounds/validate_effects.py",
        io.ROOT / "tests/feedback/rounds/test_errors.py",
        io.ROOT / "tests/feedback/rounds/test_reader.py",
        io.REVIEW / "review_plan.json",
        io.STUDY / "STUDY_PROTOCOL.md",
        io.STUDY / "plan.json",
    ]
    value = {
        "status": "implementation_contract_before_effect_scoring",
        "created_utc": io.pure.utc(),
        "timing": "After generation; before any Part 2 scientific effect scoring. Implements the frozen protocol and Part 1 clarifications.",
        "primary": [
            {
                "round": 4,
                "population": "en_test",
                "left": "adaptive",
                "right": right,
                "endpoint": endpoint,
                "seed": seed,
                "interval_level": 0.9875,
            }
            for right, endpoint, seed in PRIMARY
        ],
        "draws": 10000,
        "rng": "PCG64",
        "quantiles": "NumPy linear interpolation",
        "main_mask": "All ten conditions including the one reference, all three decoder repeats, in the stated round.",
        "trajectory_mask": "All four rounds, ten conditions and three decoder repeats valid on each retained input.",
        "strict_mask": "All thirty final-round predictions have all three valid votes. Supporting, not a replacement primary.",
        "estimand": "Text-weighted FPR among benign inputs or FNR among hateful inputs; nine equally weighted stream/repeat outcomes for each policy and three for the reference. Average contrasts within input first.",
        "bootstrap_unit": "Within each population and endpoint cohort, resample the inherited inference clusters represented in that cohort, with replacement, preserving all eligible same-class member inputs and their multiplicities. Ratio of weighted error sums to weighted input counts; not an unweighted mean of cluster means.",
        "reference": "Three actual saved reference predictions per round/input, shared when forming stream contrasts. Multiplication by three in integer arithmetic changes no statistical unit.",
        "supporting": "Per-round common-mask rates/contrasts, fixed-cohort four-round trajectories and strict-final-vote rates/contrasts use nominal 95% intervals. No additional confirmatory family or post-hoc winner selection.",
        "trajectory_change": "R4-R1 and adjacent rounds, both absolute and relative to the fresh reference; differences of paired within-input contrasts. Decoder-repeat IDs are a descriptive pairing of fresh stochastic calls.",
        "supporting_seed": "First eight SHA256 bytes, little endian, of A17-part2-support-v1| plus the exact record name.",
        "missing_bounds": "On every original class/population roster, assign missing policy outputs opposite error extremes. Identification bounds only, not confidence intervals. Available-case rates keep their own denominators and are descriptive.",
        "variation": "All-round common-cohort rates for every stream/decoder repeat; descriptive dependent cells, never independent sample replication.",
        "undefined": "Raise on an empty cohort or nonfinite bootstrap draw; never replace by zero or omit favourable draws.",
        "no_changes": [
            "saved predictions",
            "source selections",
            "generation code",
            "hypothesis family",
            "frozen source/evaluation rosters",
        ],
        "source_hashes": {io.rel(p): io.sha(p) for p in files},
        "new_Qwen_calls": 0,
        "new_fits": 0,
    }
    io.write(CONTRACT, value)
    print("Analysis implementation contract saved before effect scoring.", flush=True)


def score():
    started = time.monotonic()
    contract = io.read(CONTRACT)
    for name, digest in contract["source_hashes"].items():
        io.need(
            io.sha(io.ROOT / name) == digest,
            "Scoring source changed after contract: " + name,
        )
    checked = io.read(io.READER / "independent_verification.json")
    io.need(
        checked["status"] == "independent_reader_verification_pass"
        and checked["reader_complete_sha256"] == io.sha(io.READER / "complete.json"),
        "Reader verification required",
    )
    assembly = io.read(io.READER / "complete.json")
    for name, digest in assembly["artifacts"].items():
        io.need(io.sha(io.ROOT / name) == digest, "Reader artifact changed")
    rows = list(io.rows(io.READER / "matrix.jsonl"))
    gates, votes, error = k.arrays(rows)
    counts, missing = k.policy_counts(error)
    io.need(int((gates < 0).sum()) == 241, "Evaluation null count differs")
    records = {}

    def add(
        name, population, endpoint, mask, round_no, definitions, seed, level, family
    ):
        indices = k.cohort(rows, gates, votes, population, endpoint, mask, round_no)
        io.need(len(indices) > 0, "Undefined endpoint")
        values = k.transformed_counts(counts, indices, definitions)
        cluster_ids = [rows[i]["inference_cluster"] for i in indices]
        draws, labels, sizes = k.bootstrap(values, cluster_ids, seed, draws=10000)
        point = values.sum(axis=0) / (9 * len(indices))
        alpha = (1 - level) / 2
        limits = np.quantile(draws, [alpha, 1 - alpha], axis=0, method="linear")
        path = SCORE / "draws" / (name + ".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            np.save(stream, draws, allow_pickle=False)
        record = {
            "name": name,
            "family": family,
            "population": population,
            "endpoint": endpoint,
            "mask": mask,
            "round": round_no,
            "input_indices": indices.tolist(),
            "inputs": len(indices),
            "inference_clusters": len(labels),
            "cluster_labels": labels,
            "cluster_sizes": sizes.tolist(),
            "source_counts": dict(Counter(rows[i]["source"] for i in indices)),
            "seed": seed,
            "interval_level": level,
            "draws": 10000,
            "columns": definitions,
            "column_order": list(definitions),
            "points": {col: float(point[j]) for j, col in enumerate(definitions)},
            "intervals": {
                col: [float(limits[0, j]), float(limits[1, j])]
                for j, col in enumerate(definitions)
            },
            "integer_error_numerators": {
                col: int(values[:, j].sum()) for j, col in enumerate(definitions)
            },
            "integer_denominator": 9 * len(indices),
            "draw_file": io.rel(path),
            "draw_sha256": io.sha(path),
        }
        records[name] = record
        print(
            f"Scored {name}: {len(indices)} inputs, {len(labels)} clusters, {len(definitions)} paired endpoints",
            flush=True,
        )

    for right, endpoint, seed in PRIMARY:
        name = f"primary_en_test_{endpoint}_adaptive-{right}"
        defs = {f"r4_adaptive-{right}": [k.term(4, "adaptive"), k.term(4, right, -1)]}
        add(
            name,
            "en_test",
            endpoint,
            "round_common",
            4,
            defs,
            seed,
            0.9875,
            "four_contrast_primary",
        )
    for population in ["en_test", "en_hatecheck_eval"]:
        for endpoint in ["FPR", "FNR"]:
            for rnd in range(1, 5):
                name = f"round_{population}_{endpoint}_r{rnd}"
                add(
                    name,
                    population,
                    endpoint,
                    "round_common",
                    rnd,
                    k.columns([rnd]),
                    k.support_seed(name),
                    0.95,
                    "supporting",
                )
            name = f"trajectory_{population}_{endpoint}"
            add(
                name,
                population,
                endpoint,
                "all_round_common",
                4,
                k.columns([1, 2, 3, 4], trajectories=True),
                k.support_seed(name),
                0.95,
                "supporting",
            )
            name = f"strict_{population}_{endpoint}_r4"
            add(
                name,
                population,
                endpoint,
                "round_strict",
                4,
                k.columns([4]),
                k.support_seed(name),
                0.95,
                "supporting",
            )
    full_bounds, cells, missingness = {}, [], []
    for population in ["en_test", "en_hatecheck_eval"]:
        for endpoint in ["FPR", "FNR"]:
            eligible = np.array(
                [
                    i
                    for i, row in enumerate(rows)
                    if row["population"] == population
                    and row["gold_hate"] == (endpoint == "FNR")
                ],
                dtype=np.int64,
            )
            common = k.cohort(
                rows, gates, votes, population, endpoint, "all_round_common", 4
            )
            for rnd in range(1, 5):
                name = f"{population}_{endpoint}_r{rnd}"
                full_bounds[name] = k.bounds(counts, missing, eligible, rnd)
                for c, condition in enumerate(io.CONDITIONS):
                    missingness.append(
                        {
                            "population": population,
                            "endpoint": endpoint,
                            "round": rnd,
                            "condition": condition,
                            "inputs": len(eligible),
                            "prediction_slots": len(eligible) * 3,
                            "nulls": int((gates[eligible, rnd - 1, c] < 0).sum()),
                            "one_vote": int((votes[eligible, rnd - 1, c] == 1).sum()),
                            "two_votes": int((votes[eligible, rnd - 1, c] == 2).sum()),
                            "three_votes": int(
                                (votes[eligible, rnd - 1, c] == 3).sum()
                            ),
                        }
                    )
                    for d in range(3):
                        numerator = int(error[common, rnd - 1, c, d].sum())
                        cells.append(
                            {
                                "population": population,
                                "endpoint": endpoint,
                                "round": rnd,
                                "condition": condition,
                                "decoder_repeat": d,
                                "inputs": len(common),
                                "errors": numerator,
                                "rate": numerator / len(common),
                                "mask": "all_round_common",
                            }
                        )
    result = {
        "status": "four_round_error_analysis_scored",
        "created_utc": io.pure.utc(),
        "records": records,
        "full_roster_missing_bounds": full_bounds,
        "stream_repeat_cells": cells,
        "missingness": missingness,
        "inputs": {
            io.rel(p): io.sha(p)
            for p in [
                CONTRACT,
                io.READER / "complete.json",
                io.READER / "independent_verification.json",
                io.READER / "matrix.jsonl",
            ]
        },
        "elapsed_seconds": time.monotonic() - started,
        "new_Qwen_calls": 0,
        "new_fits": 0,
    }
    io.write(SCORE / "results.json", result)
    io.write(
        SCORE / "complete.json",
        {
            "status": "scored_awaiting_independent_numeric_check",
            "results_sha256": io.sha(SCORE / "results.json"),
            "records": len(records),
            "new_Qwen_calls": 0,
            "new_fits": 0,
        },
    )
    print(
        "All planned error-rate calculations saved; independent numerical verification is next.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["freeze", "score"])
    args = parser.parse_args()
    with io.pure.offline():
        (freeze if args.mode == "freeze" else score)()

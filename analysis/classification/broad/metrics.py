import argparse
from collections import defaultdict
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
from threadpoolctl import threadpool_limits
from analysis.classification.broad import reader as reader

c, r, scoring = reader.c, reader.r, reader.scoring
ROOT = reader.ROOT
BASE = ROOT / "experiments/astra/sq2_broad_main_review_20261104_v1"
CONDITIONS = reader.CONDITIONS
DRAWS = 10000
SEED = 20261008
require = reader.require


def save(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE.resolve()), "Output outside Part 2 directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(c.encoded(value))


def array(path, **values):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)


def bindings(paths):
    return reader.bindings(paths)


def slug(name):
    return name.replace("/", "__")


def mean(values):
    return (
        float(np.mean(values))
        if values and all(v is not None for v in values)
        else None
    )


def prepare():
    save(
        BASE / "analysis_contract.json",
        {
            "status": "part2_scope_recorded_before_effectiveness_calculation",
            "created_utc": c.utc(),
            "sources": bindings(
                [
                    ROOT
                    / "Thesis_Log/astra_result/astra_sq2_broad_analysis_plan_20261104_v1.md",
                    reader.BASE / "part1_complete.json",
                    reader.BASE / "final_review.json",
                    reader.BASE / "setup_complete.json",
                    reader.BASE / "comparison_map.json",
                    reader.BASE / "endpoint_setup.json",
                    Path(reader.__file__),
                    Path(scoring.__file__),
                    Path(scoring.original.__file__),
                ]
            ),
            "question": "Does full retrieval improve on small retrieval, and how do the saved retrieval systems compare with no retrieval on the frozen populations?",
            "primary": [list(pair) for pair in sorted(scoring.PRIMARY_BROAD)],
            "primary_contrast": "rag_full minus rag_small; four-question family, 98.75% paired percentile intervals",
            "supporting": "All original full-minus-zero and available historical encoder contrasts; other full-minus-small endpoints. Pointwise 95% intervals, no new primary family.",
            "scope": "All 40 original contrast entries; 38 have supported macro labels. Keep German target coverage without manufacturing a macro score.",
            "absolute_scores": "Own-valid means and all three individual F1 values, descriptive sample SD and full-roster failure bounds. Also an all-nine-Qwen common-valid table with per-repeat values. No pooled ensemble.",
            "comparability": "Pairwise intervals use all six compared settings. The three-condition display uses all nine Qwen settings. Never substitute one population for the other.",
            "small_minus_zero": "Descriptive difference only on the shared three-condition population; no additional inferential contrast or primary claim.",
            "bootstrap": dict(
                draws=DRAWS,
                seed=SEED,
                clusters="original normalized-text groups; HateCheck connected components",
                implementation="Unchanged frozen contrast function with captured draws. Exact integer-weighted sums may use checked float64 BLAS; the sampler and metric are unchanged.",
                confidence={"primary": 0.9875, "pointwise": 0.95},
            ),
            "fixed_labels_and_failures": scoring.contract(),
            "independent_check": "Rebuild encoded gold/predictions from saved decisions; recompute own/common scores and failure bounds; replay every saved bootstrap value with a separate TP/predicted/actual formulation.",
            "outputs": "Source-bound point and interval tables, per-repeat values, retained draws, primary-contrast figure, independent numerical/publication review, report and roadmap.",
            "deferred": [
                "FP/FN interpretation and cases (Part 3)",
                "fine-label/exact/joint answer review (Part 4)",
                "source/function breakdown (Part 5)",
                "context interpretation (Part 6)",
                "source/item/replay/whitespace/partial-vote robustness (Part 7)",
                "resources (Part 8)",
                "standalone and combined SQ2 synthesis",
            ],
            "limits": "Exposed data and fixed models, pools and repeats; conditional uncertainty. No equivalence inference from zero-crossing intervals, no endpoint-win-count test, no faithful-rationale claim.",
            "new_generation": False,
            "new_training": False,
            "new_inference": False,
        },
    )


def capture_contrast(
    left, right, clusters, *, draws=DRAWS, seed=SEED, family=None, accelerate=True
):

    captured = []
    cached = {}
    calls = 0

    def quantile(values, q, *args, **kwargs):
        values = np.asarray(values, dtype=np.float64)
        if captured:
            require(
                np.array_equal(values, captured[0]),
                "Different draw vectors for the two intervals",
            )
        else:
            captured.append(values.copy())
        return np.quantile(values, q, *args, **kwargs)

    def weighted(expression, weights, terms, **kwargs):
        nonlocal calls
        require(
            expression == "bg,sgk->bsk" and weights.dtype == terms.dtype == np.int64,
            "Unexpected frozen sum",
        )
        require(
            weights.ndim == 2
            and terms.ndim == 3
            and weights.shape[1] == terms.shape[1],
            "Wrong weighted shapes",
        )
        require(weights.min() >= 0 and terms.min() >= 0, "Negative count")
        require(
            int(weights.sum(axis=1).max()) * int(terms.max()) < 2**52,
            "Exact float sum bound exceeded",
        )
        if cached.get("terms") is not terms:
            cached.update(terms=terms, floating=terms.astype(np.float64))
        result = np.einsum(
            expression, weights.astype(np.float64), cached["floating"], optimize=True
        )
        require(
            np.array_equal(result, np.rint(result)),
            "Weighted counts are not exact integers",
        )
        calls += 1
        return result.astype(np.int64)

    proxy = SimpleNamespace(**dict(vars(np), quantile=quantile))
    if accelerate:
        proxy.einsum = weighted
    fn = reader.clone(scoring.contrast, np=proxy)
    result = fn(left, right, clusters, draws=draws, seed=seed, family=family)
    values = captured[0] if captured else np.empty(0, dtype=np.float64)
    require(len(values) in (0, draws), "Incomplete bootstrap draw capture")
    result.update(
        actual_draws=len(values),
        integer_sum_acceleration=accelerate,
        exact_integer_sum_blocks=calls,
    )
    return result, values


def f1_on_mask(contrib, mask):
    require(
        np.asarray(mask).dtype == np.bool_ and len(mask) == len(contrib["ids"]),
        "Wrong scoring mask",
    )
    require(
        not np.any(mask & ~contrib["valid"]), "Invalid answer in common-valid score"
    )
    if not mask.any() or not contrib["macro_indices"]:
        return None
    return float(
        scoring.original.macro(
            contrib["counts"][mask].sum(axis=0), contrib["macro_indices"]
        )
    )


def summarize_runs(values):
    scores = [scoring.original.score(value) for value in values]
    points = [score["macro_f1_valid_only"] for score in scores]
    return {
        "repeats": scores,
        "mean_macro_f1": mean(points),
        "sample_sd": (
            float(np.std(points, ddof=1))
            if all(v is not None for v in points)
            else None
        ),
        "mean_full_roster_failure_bounds": [
            mean([score["invalid_inclusive_bounds"][i] for score in scores])
            for i in (0, 1)
        ],
        "repeat_meaning": "fixed decoder repeats or fixed fitted seeds, not independent texts",
    }


def run():
    plan = c.read(BASE / "analysis_contract.json")
    r.verify(plan["sources"])
    tests = c.read(BASE / "tests.json")
    r.verify(tests["sources"])
    require(tests["status"] == "passed", "Scientific tests required")
    groups, dimensions, predictions, endpoints, comparisons = reader.load()
    before = bindings(
        [
            Path(__file__),
            Path(reader.__file__),
            Path(scoring.__file__),
            Path(scoring.original.__file__),
            BASE / "analysis_contract.json",
            BASE / "tests.json",
            reader.BASE / "part1_complete.json",
            reader.BASE / "final_review.json",
            reader.BASE / "assembly.json",
            reader.BASE / "setup_complete.json",
        ]
    )
    save(BASE / "preservation_before.json", before)
    outputs = {}
    index = []
    bootstrap_count = 0
    own_count = 0
    start = time.perf_counter()
    for name, ep in sorted(endpoints.items()):
        phase, dimension = name.split("/")
        rr = groups[phase]
        keys = {row["key"] for row in rr}
        expected = scoring.endpoint(
            rr,
            dimension,
            __import__("yaml").safe_load(
                (ROOT / "config/taxonomy.yaml").read_text(encoding="utf8")
            ),
        )
        require(all(ep[k] == v for k, v in expected.items()), "Frozen endpoint changed")
        settings = [
            (cond, rep) for cond in ep["available_conditions"] for rep in range(3)
        ]
        terms = {
            ident: scoring.contributions(
                rr, {key: predictions[ident][key] for key in keys}, ep
            )
            for ident in settings
        }
        own = {
            cond: summarize_runs([terms[cond, rep] for rep in range(3)])
            for cond in ep["available_conditions"]
        }
        own_count += len(settings)
        for cond in ep["available_conditions"]:
            for rep in range(3):
                actual = [
                    key
                    for key, on in zip(ep["scorable_ids"], terms[cond, rep]["valid"])
                    if not on
                ]
                require(
                    actual == ep["own_invalid_ids"][f"{cond}/{rep}"],
                    "Own-valid mask changed",
                )
        common = np.logical_and.reduce(
            [terms[cond, rep]["valid"] for cond in CONDITIONS for rep in range(3)]
        )
        common_excluded = [key for key, on in zip(ep["scorable_ids"], common) if not on]
        require(
            common_excluded == ep["three_Qwen_conditions_common_invalid_ids"],
            "Three-condition mask changed",
        )
        absolute = {}
        for cond in CONDITIONS:
            values = [f1_on_mask(terms[cond, rep], common) for rep in range(3)]
            absolute[cond] = {
                "repeat_macro_f1": values,
                "mean_macro_f1": mean(values),
                "sample_sd": (
                    float(np.std(values, ddof=1))
                    if all(v is not None for v in values)
                    else None
                ),
            }
        shared = {
            "n_eligible": len(common),
            "n_common_valid": int(common.sum()),
            "excluded_ids": common_excluded,
            "per_condition": absolute,
            "interpretation": "Three-condition descriptive display only; original paired contrasts use their own six-setting common mask.",
        }
        shared["small_minus_zero_descriptive"] = (
            None
            if not ep["macro_labels"]
            else absolute["rag_small"]["mean_macro_f1"]
            - absolute["zero_shot"]["mean_macro_f1"]
        )
        dest = BASE / "endpoints" / slug(name)
        dest.mkdir(parents=True, exist_ok=False)
        array(
            dest / "encoded.npz",
            gold=next(iter(terms.values()))["gold"].astype(np.int8),
            predicted=np.stack([terms[ident]["pred"] for ident in settings]).astype(
                np.int8
            ),
            valid=np.stack([terms[ident]["valid"] for ident in settings]),
            counts=np.stack([terms[ident]["counts"] for ident in settings]).astype(
                np.int8
            ),
        )
        paired = []
        for comp in [
            row
            for row in comparisons
            if row["population"] == phase and row["dimension"] == dimension
        ]:
            left, right = comp["left"], comp["right"]
            summary, draws = capture_contrast(
                [terms[left, rep] for rep in range(3)],
                [terms[right, rep] for rep in range(3)],
                ep["row_clusters"],
                family=comp["family_size"],
            )
            require(
                summary["excluded_ids"] == comp["common_invalid_ids"]
                and summary["n_common_valid"] == comp["common_valid_rows"],
                "Paired mask changed",
            )
            summary.update(
                left=left,
                right=right,
                role=comp["role"],
                population=phase,
                dimension=dimension,
            )
            if len(draws):
                path = dest / (left + "_minus_" + right + "_draws.npy")
                with path.open("xb") as stream:
                    np.save(stream, draws, allow_pickle=False)
                summary["draws_path"] = c.rel(path)
                summary["draws_sha256"] = c.sha(path)
                outputs[c.rel(path)] = summary["draws_sha256"]
            chosen = (
                summary["family_interval"]
                if comp["role"] == "primary"
                else summary["ci95"]
            )
            summary["interval_relation"] = (
                "not_scorable"
                if chosen is None
                else (
                    "above_zero"
                    if chosen[0] > 0
                    else "below_zero" if chosen[1] < 0 else "crosses_or_touches_zero"
                )
            )
            bounds = summary["full_roster_failure_difference_bounds"]
            summary["failure_bound_relation"] = (
                "not_scorable"
                if bounds[0] is None
                else (
                    "above_zero"
                    if bounds[0] > 0
                    else "below_zero" if bounds[1] < 0 else "crosses_or_touches_zero"
                )
            )
            bootstrap_count += len(draws)
            paired.append(summary)
            point = (
                "not scorable"
                if summary["mean_difference"] is None
                else f"{summary['mean_difference']*100:+.3f} F1 points"
            )
            print(name, left, "minus", right, point, flush=True)
        result = {
            "endpoint": ep,
            "settings": [list(ident) for ident in settings],
            "own_valid": own,
            "three_condition_common": shared,
            "contrasts": paired,
            "sources": before,
            "encoded_path": c.rel(dest / "encoded.npz"),
            "encoded_sha256": c.sha(dest / "encoded.npz"),
        }
        save(dest / "results.json", result)
        outputs.update(bindings([dest / "encoded.npz", dest / "results.json"]))
        index.append(
            {
                "endpoint": name,
                "path": c.rel(dest / "results.json"),
                "sha256": c.sha(dest / "results.json"),
            }
        )
    require(
        bootstrap_count == 380000 and len(index) == 11 and own_count == 153,
        "Unexpected comparison/setting/draw inventory",
    )
    r.verify(before)
    save(
        BASE / "scores_complete.json",
        {
            "status": "frozen_broader_main_comparisons_scored_review_pending",
            "endpoints": index,
            "own_valid_setting_endpoint_scores": own_count,
            "contrast_entries": 40,
            "scorable_contrasts": 38,
            "bootstrap_draws": bootstrap_count,
            "outputs": outputs,
            "sources": before,
            "elapsed_seconds": time.perf_counter() - start,
            "new_Qwen_calls": 0,
            "new_training": 0,
            "new_inference": 0,
            "sensitivities": "Original Part 7 sensitivities remain pending; no scientific closure is implied.",
        },
    )
    print(
        "Main comparison computation complete; independent review is next.", flush=True
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["prepare", "run"], required=True)
    args = parser.parse_args()
    with r.original.offline(), threadpool_limits(limits=2):
        {"prepare": prepare, "run": run}[args.mode]()

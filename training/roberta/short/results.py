from collections import defaultdict
import gc
from pathlib import Path
from training.roberta.short import validate as r
import numpy as np

c, s, prior = r.c, r.s, r.prior
previous = prior.previous
REFS = ["encoder_kb", "encoder_full", "bge", "roberta_mean", "roberta_cls"]


def metric(y, x, ep):
    stats, counts = previous.summarize_bits(y, x, ep)

    a, b = y.astype(int), x.astype(int)
    mismatch = np.abs(a - b).sum(1)
    c.require(
        stats["exact_set_accuracy"] == float((mismatch == 0).mean()),
        "Exact-set arithmetic",
    )
    c.require(
        stats["mean_predicted_labels"] == float(sum(map(sum, b)) / len(b)),
        "Cardinality arithmetic",
    )
    for j, label in enumerate(ep["all_labels"]):
        truth, prediction = set(np.flatnonzero(y[:, j])), set(np.flatnonzero(x[:, j]))
        got = stats["per_label"][label]
        c.require(
            (got["tp"], got["fp"], got["fn"])
            == (
                len(truth & prediction),
                len(prediction - truth),
                len(truth - prediction),
            ),
            "Set-based confusion count differs",
        )
    return stats, counts


def scores():
    integrity = c.read(r.OUT / "integrity.json")
    c.require(integrity["status"] == "pass", "Integrity required before scoring")
    before = r.guard()
    r.save(
        "scoring_plan.json",
        {
            "source_sha256": c.sha(Path(__file__)),
            "analysis_plan_sha256": c.sha(r.OUT / "analysis_plan.json"),
            "primary_family_unchanged": r.PRIMARY,
            "evaluation_based_selection": False,
            "supporting_gate_factorial": "Original/new selected gate cutoffs and 0.5, only on original/new already selected gate weights.",
            "zero_division": 0,
            "all_seeds_retained": True,
            "Qwen_calls": 0,
        },
    )
    old_scores = c.read(prior.OUT / "scores.json")["cells"]
    coretexts = {
        previous.norm(x["text"])
        for x in previous.rows(previous.CORE / "dev_inputs.jsonl")
        if x["lang"] == "en"
    }
    cells, groups, cache, transitions, cases, ungated = {}, {}, {}, {}, [], {}
    old_verified = fixed_verified = 0
    for scope in ("core", "broad"):
        eps, positions = r.context(scope)
        maps = previous.predictions(scope)
        for name, ep in eps.items():
            _, kept, ids, y = prior.gold(scope, name, ep)
            head = ep["dimension"]
            runs = defaultdict(list)
            model_specs = [
                (prefix + variant, list(c.FINAL_SEEDS), base, variant)
                for prefix, base in [("expanded_", r.BASE), ("original_", s.OLD)]
                for variant in r.VARIANTS
            ]
            model_specs += [
                (
                    model,
                    [0, 1, 2] if model.startswith("encoder_") else [42, 43, 44],
                    None,
                    None,
                )
                for model in REFS
            ]
            masks = {
                source: np.asarray([x["source"] == source for x in kept])
                for source in sorted({x["source"] for x in kept})
            }
            if scope == "broad":
                masks["outside_original_core"] = np.asarray(
                    [previous.norm(x["text"]) not in coretexts for x in kept]
                )
            for model, seeds, base, variant in model_specs:
                for seed in seeds:
                    if base is not None:
                        x = r.bits(base, variant, seed, head, ids, positions, ep)
                    else:
                        found, yy, x = previous.arrays(kept, maps[model, seed], ep)
                        c.require(
                            found == ids and np.array_equal(yy, y),
                            "Reference eligibility differs",
                        )
                    stats, counts = metric(y, x, ep)
                    stats["seed"] = seed
                    runs[model].append(stats)
                    if not model.startswith("expanded_"):
                        key = variant if model.startswith("original_") else model
                        expected = next(
                            v
                            for v in old_scores[scope + "/" + name]["runs"][key]
                            if v["seed"] == seed
                        )
                        c.require(
                            stats == expected,
                            "Earlier published result failed readback",
                        )
                        old_verified += 1
                    elif not variant.startswith("tuned"):
                        expected = next(
                            v
                            for v in old_scores[scope + "/" + name]["runs"][variant]
                            if v["seed"] == seed
                        )
                        c.require(
                            stats == expected, "Historical fixed-control result changed"
                        )
                        fixed_verified += 1
                    groups[f"{scope}/{name}/{model}/{seed}"] = {
                        key: metric(y[mask], x[mask], ep)[0]
                        for key, mask in masks.items()
                        if mask.any()
                    }
                    if (
                        scope == "broad"
                        and name in r.PRIMARY
                        and model
                        in ("expanded_tuned_selected", "original_tuned_selected", "bge")
                    ):
                        cache[f"{name}|{model}|{seed}"] = counts
                    if (
                        base is not None
                        and head != "hate"
                        and variant == "tuned_selected"
                    ):
                        raw = r.bits(
                            base, variant, seed, head, ids, positions, ep, ungated=True
                        )
                        ungated[f"{scope}/{name}/{model}/{seed}"] = metric(y, raw, ep)[
                            0
                        ]
            cells[scope + "/" + name] = {
                "n": len(ids),
                "endpoint": {k: v for k, v in ep.items() if k != "scorable_ids"},
                "runs": dict(runs),
                "means": {
                    m: float(np.mean([x["macro_f1"] for x in values]))
                    for m, values in runs.items()
                },
                "sample_sds": {
                    m: float(np.std([x["macro_f1"] for x in values], ddof=1))
                    for m, values in runs.items()
                },
            }
            if scope == "broad" and name in r.PRIMARY:
                transition_runs = []
                for seed in c.FINAL_SEEDS:
                    left = r.bits(
                        r.BASE, "tuned_selected", seed, head, ids, positions, ep
                    )
                    right = r.bits(
                        s.OLD, "tuned_selected", seed, head, ids, positions, ep
                    )
                    old_correct, new_correct = (right == y).all(1), (left == y).all(1)
                    changed = (left != right).any(1)
                    row = {
                        "seed": seed,
                        "rows": len(ids),
                        "changed": int(changed.sum()),
                        "exact_fixes": int((~old_correct & new_correct).sum()),
                        "exact_regressions": int((old_correct & ~new_correct).sum()),
                        "changed_both_wrong": int(
                            (changed & ~old_correct & ~new_correct).sum()
                        ),
                        "labels_added": int((left & ~right).sum()),
                        "labels_removed": int((right & ~left).sum()),
                    }
                    if head == "hate":
                        j = ep["all_labels"].index("hate")
                        row.update(
                            nonhate_to_hate=int((~right[:, j] & left[:, j]).sum()),
                            hate_to_nonhate=int((right[:, j] & ~left[:, j]).sum()),
                        )
                    else:
                        a = ungated[f"{scope}/{name}/expanded_tuned_selected/{seed}"]
                        b = ungated[f"{scope}/{name}/original_tuned_selected/{seed}"]
                        c.require(a == b, "Changed ungated fine-head performance")
                    transition_runs.append(row)
                    for i in np.flatnonzero(changed):
                        cases.append(
                            {
                                "endpoint": name,
                                "id": ids[i],
                                "source": kept[i]["source"],
                                "seed": seed,
                                "gold": [
                                    l for l, v in zip(ep["all_labels"], y[i]) if v
                                ],
                                "original": [
                                    l for l, v in zip(ep["all_labels"], right[i]) if v
                                ],
                                "expanded": [
                                    l for l, v in zip(ep["all_labels"], left[i]) if v
                                ],
                                "old_exact": bool(old_correct[i]),
                                "new_exact": bool(new_correct[i]),
                            }
                        )
                transitions[name] = transition_runs
            print(
                scope,
                name,
                {
                    m: round(cells[scope + "/" + name]["means"][m], 6)
                    for m in (
                        "expanded_tuned_selected",
                        "original_tuned_selected",
                        "bge",
                    )
                },
                flush=True,
            )
        del maps
        gc.collect()
    r.save(
        "scores.json",
        {
            "status": "scored",
            "cells": cells,
            "old_reference_cells_reproduced": old_verified,
            "fixed_control_cells_reproduced": fixed_verified,
            "scoring_source_sha256": c.sha(Path(__file__)),
        },
    )
    r.save(
        "subgroups.json",
        {
            "post_result_descriptions": True,
            "macro_labels_unchanged": True,
            "cells": groups,
        },
    )
    r.save("ungated.json", {"cells": ungated, "same_fine_heads_and_thresholds": True})
    r.save("transitions.json", {"paired_by_saved_seed": True, "cells": transitions})
    r.save(
        "changed_cases.json",
        {
            "selection": "Every changed primary-endpoint prediction in every final seed; no accuracy-based sampling.",
            "rows": cases,
        },
    )
    r.save("paired_counts.npz", r.npz_bytes(cache))
    c.require(r.guard() == before, "Active Qwen sources changed")


def bootstrap(counts, clusters, sources, indices, stratified=False, draws=10000):
    unique, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    aggregate = np.zeros((len(unique), counts.shape[0], counts.shape[2], 3), np.int64)
    np.add.at(aggregate, inverse, counts.transpose(1, 0, 2, 3))
    members = defaultdict(set)
    for i, src in zip(inverse, sources):
        members[int(i)].add(src)
    strata = defaultdict(list)
    for i in range(len(unique)):
        strata[tuple(sorted(members[i])) if stratified else ("all",)].append(i)
    rng = np.random.default_rng(20261008)
    flat = aggregate.reshape(len(unique), -1).astype(float)
    values, direct_checks = [], 0

    def score(total):
        tp, fp, fn = total[..., 0], total[..., 1], total[..., 2]
        den = 2 * tp + fp + fn
        return np.divide(
            2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0
        )[..., indices].mean(-1)

    for start in range(0, draws, 64):
        n = min(64, draws - start)
        picks = np.concatenate(
            [
                rng.choice(group, (n, len(group)), replace=True)
                for _, group in sorted(strata.items())
            ],
            axis=1,
        )
        weights = np.asarray(
            [np.bincount(p, minlength=len(unique)) for p in picks], dtype=float
        )
        total = (weights @ flat).reshape(n, counts.shape[0], counts.shape[2], 3)
        c.require(
            np.array_equal(total[0], aggregate[picks[0]].sum(0)),
            "Independent integer bootstrap sum",
        )
        direct_checks += 1
        f1 = score(total)
        values.extend((f1[:, :5].mean(1) - f1[:, 5:].mean(1)).tolist())
    values = np.asarray(values)
    original = score(counts.sum(1))
    summary = {
        "left": float(original[:5].mean()),
        "right": float(original[5:].mean()),
        "difference": float(original[:5].mean() - original[5:].mean()),
        "ci9875": np.quantile(values, [0.00625, 0.99375]).tolist(),
        "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
        "n": len(clusters),
        "clusters": len(unique),
        "draws": draws,
        "seed": 20261008,
        "stratified": stratified,
        "direct_integer_draw_checks": direct_checks,
    }
    return summary, values


def tests():
    counts = np.zeros((10, 3, 2, 3), np.int64)
    counts[:5, :, :, 0] = 1
    counts[5:, :, :, 2] = 1
    a, _ = bootstrap(counts, ["a", "a", "b"], ["s", "s", "t"], [0, 1], draws=128)
    c.require(
        a["difference"] == 1 and a["ci9875"] == [1.0, 1.0] and a["clusters"] == 2,
        "Five/five perfect-versus-wrong fixture",
    )
    counts[:] = 1
    a, draws = bootstrap(
        counts, ["a", "a", "b"], ["s", "s", "t"], [0, 1], stratified=True, draws=128
    )
    c.require(a["difference"] == 0 and not draws.any(), "Identical-paired fixture")
    rng = np.random.default_rng(181)
    counts = rng.integers(0, 2, (8, 7, 3, 3))
    cl, src = ["a", "b", "c", "b", "d", "e", "a"], ["x", "y", "x", "y", "x", "z", "x"]
    checks = 0
    for stratified in (False, True):
        a, _ = bootstrap(counts, cl, src, [0, 2], stratified=stratified, draws=128)
        b = prior.bootstrap(
            counts, cl, src, [0, 2], left=5, draws=128, stratified=stratified
        )
        c.require(
            all(a[k] == v for k, v in b.items()),
            "Five/three independent legacy-kernel parity",
        )
        checks += 1
    r.save(
        "statistics_tests.json",
        {
            "passed": True,
            "perfect_and_identical_cases": 2,
            "unequal_fit_count_legacy_parity": checks,
            "source_sha256": c.sha(Path(__file__)),
            "new_fits": 0,
            "Qwen_calls": 0,
        },
    )


def statistics():
    tests()
    cells = c.read(r.OUT / "scores.json")["cells"]
    eps, _ = r.context("broad")
    results, draws_out, reference_checks = {}, {}, 0
    with np.load(r.OUT / "paired_counts.npz", allow_pickle=False) as saved:
        for name in r.PRIMARY:
            ep = eps[name]
            _, kept, ids, _ = prior.gold("broad", name, ep)
            clusters, sources = [x["text_cluster"] for x in kept], [
                x["source"] for x in kept
            ]
            indices = [ep["all_labels"].index(l) for l in ep["macro_labels"]]
            comparisons = {}
            for right, seeds, label in [
                ("original_tuned_selected", c.FINAL_SEEDS, "primary"),
                ("bge", [42, 43, 44], "supporting_BGE"),
            ]:
                counts = np.stack(
                    [
                        saved[f"{name}|{model}|{seed}"]
                        for model, ss in [
                            ("expanded_tuned_selected", c.FINAL_SEEDS),
                            (right, seeds),
                        ]
                        for seed in ss
                    ]
                )
                comparisons[label] = {}
                for stratified in (False, True) if label == "primary" else (False,):
                    key = "source_stratified" if stratified else "unstratified"
                    result, draws = bootstrap(
                        counts, clusters, sources, indices, stratified=stratified
                    )
                    expected = (
                        cells["broad/" + name]["means"]["expanded_tuned_selected"]
                        - cells["broad/" + name]["means"][right]
                    )
                    c.require(
                        abs(result["difference"] - expected) < 1e-12,
                        "Bootstrap point/scoring mismatch",
                    )

                    if label == "primary":
                        legacy = prior.bootstrap(
                            counts,
                            clusters,
                            sources,
                            indices,
                            left=5,
                            stratified=stratified,
                        )
                        c.require(
                            all(result[k] == v for k, v in legacy.items()),
                            "Primary interval cross-check differs",
                        )
                        reference_checks += 1
                    comparisons[label][key] = result
                    draws_out[name + "|" + label + "|" + key] = draws
                    print("Paired contrast", name, label, key, result, flush=True)
            runs = cells["broad/" + name]["runs"]
            comparisons["seed_differences"] = [
                {
                    "seed": seed,
                    "difference": next(
                        x["macro_f1"]
                        for x in runs["expanded_tuned_selected"]
                        if x["seed"] == seed
                    )
                    - next(
                        x["macro_f1"]
                        for x in runs["original_tuned_selected"]
                        if x["seed"] == seed
                    ),
                }
                for seed in c.FINAL_SEEDS
            ]
            results[name] = comparisons
    r.save("bootstrap_draws.npz", r.npz_bytes(draws_out))
    r.save(
        "statistics.json",
        {
            "primary_family": r.PRIMARY,
            "confidence": 0.9875,
            "draws": 10000,
            "contrasts": results,
            "independent_legacy_primary_cross_checks": reference_checks,
            "bootstrap_scalar_values_saved": sum(v.size for v in draws_out.values()),
            "supporting_confidence": 0.95,
            "conditional_on_fitted_models": True,
            "prior_evaluation_exposure_retained": True,
            "source_sha256": c.sha(Path(__file__)),
        },
    )


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["scores", "statistics", "tests"])
    args = ap.parse_args()
    with r.offline():
        globals()[args.mode]()


if __name__ == "__main__":
    main()

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from analysis.classification.supervised import archive as r
from analysis.classification.supervised import training_audit as tr

BASE = r.ROOT / "experiments/astra/sq2_gpu_main_review_20261103_v1"
TRAIN = (
    r.ROOT / "experiments/astra/sq2_gpu_training_review_20261103_v1/runs/7b1f17c8f8b9"
)
METHODS = ("tuned_selected", "bge_selected_fold42", "frozen_roberta_selected_fold42")
BUDGETS = (147, 1000, 5000)
DRAWS = 10000
SEED = 20261020
ALPHA = 0.05 / 24
require = r.require


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(r.encoded(value))


def array(path, **values):
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)


def slug(endpoint):
    return endpoint.replace("/", "__")


def budget(pool):
    size = pool.split("_")[1]
    return 147 if size == "compact" else int(size)


def model(pool, method, seed=None):
    return dict(
        id=f"{pool}/{method}/" + ("fold42" if seed is None else "s" + str(seed)),
        pool=pool,
        budget=budget(pool),
        method=method,
        seed=seed,
    )


def validate_binary(y, predictions, support):
    require(
        y.ndim == 2
        and predictions.ndim == 3
        and predictions.shape[1:] == y.shape
        and len(y) > 0
        and len(predictions) > 0,
        "Wrong metric dimensions.",
    )
    require(
        ((y == 0) | (y == 1)).all() and ((predictions == 0) | (predictions == 1)).all(),
        "Nonbinary metric input.",
    )
    require(
        len(support) > 0
        and len(set(support)) == len(support)
        and all(type(j) is int and 0 <= j < y.shape[1] for j in support),
        "Invalid fixed macro label columns.",
    )


def point_scores(y, predictions, support):
    validate_binary(y, predictions, support)
    yy = np.asarray(y, dtype=bool)
    pp = np.asarray(predictions, dtype=bool)
    tp = (pp & yy).sum(axis=1, dtype=np.int64)
    predicted = pp.sum(axis=1, dtype=np.int64)
    actual = yy.sum(axis=0, dtype=np.int64)
    denominator = predicted + actual
    f1 = np.divide(
        2.0 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0
    )
    return f1[:, support].mean(axis=1), dict(
        tp=tp, fp=predicted - tp, fn=actual - tp, per_label_f1=f1
    )


def group_terms(y, predictions, clusters, support):
    validate_binary(y, predictions, support)
    require(
        len(clusters) == len(y) and all(isinstance(x, str) and x for x in clusters),
        "Invalid resampling identities.",
    )
    unique, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    yy = np.asarray(y[:, support], dtype=bool)
    pp = np.asarray(predictions[:, :, support], dtype=bool)
    models, _, labels = pp.shape
    terms = np.zeros((len(unique), 2 * models * labels + labels), dtype=np.float64)
    for m in range(models):
        for j in range(labels):
            terms[:, m * labels + j] = np.bincount(
                inverse, weights=pp[m, :, j] & yy[:, j], minlength=len(unique)
            )
            terms[:, models * labels + m * labels + j] = np.bincount(
                inverse, weights=pp[m, :, j], minlength=len(unique)
            )
    for j in range(labels):
        terms[:, 2 * models * labels + j] = np.bincount(
            inverse, weights=yy[:, j], minlength=len(unique)
        )
    require(np.array_equal(terms, terms.astype(np.int64)), "Noninteger cluster counts.")
    return unique, inverse, terms


def score_terms(total, models, labels):
    require(
        total.ndim == 2
        and total.shape[1] == 2 * models * labels + labels
        and (total >= 0).all(),
        "Invalid resampled counts.",
    )
    tp = total[:, : models * labels].reshape(-1, models, labels)
    predicted = total[:, models * labels : 2 * models * labels].reshape(
        -1, models, labels
    )
    actual = total[:, 2 * models * labels :][:, None, :]
    require(
        (tp <= predicted).all() and (tp <= actual).all(),
        "Impossible resampled true-positive count.",
    )
    den = predicted + actual
    return np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0).mean(axis=-1)


def bootstrap(
    y,
    predictions,
    clusters,
    support,
    *,
    draws=DRAWS,
    seed=SEED,
    batch=64,
    progress=None,
):
    unique, inverse, terms = group_terms(y, predictions, clusters, support)
    models, labels = len(predictions), len(support)
    rng = np.random.default_rng(seed)
    out = np.empty((draws, models), dtype=float)
    picks_hash = hashlib.sha256()
    direct_checks = 0
    for start in range(0, draws, batch):
        n = min(batch, draws - start)
        picks = rng.integers(0, len(unique), size=(n, len(unique)), dtype=np.int64)
        picks_hash.update(picks.astype("<i8", copy=False).tobytes())
        weights = np.asarray(
            [np.bincount(row, minlength=len(unique)) for row in picks], dtype=float
        )
        total = weights @ terms
        require(
            np.array_equal(total[0], terms[picks[0]].sum(axis=0)),
            "Direct cluster sum differs.",
        )
        direct_checks += 1
        out[start : start + n] = score_terms(total, models, labels)
        if progress and (start + n == draws or (start + n) % 2048 == 0):
            progress(start + n)
    require(
        np.isfinite(out).all() and ((0 <= out) & (out <= 1)).all(),
        "Nonfinite bootstrap score.",
    )
    return out, dict(
        draws=draws,
        seed=seed,
        clusters=len(unique),
        cluster_order_sha256=r.signature(unique.tolist()),
        picks_sha256=picks_hash.hexdigest(),
        direct_cluster_sum_checks=direct_checks,
        paired_across_all_methods_pools_and_seeds=True,
        fixed_supported_columns=support,
        conditional_on_fixed_fits=True,
    )


def aggregation(models, scores):
    require(
        scores.ndim in (1, 2) and scores.shape[-1] == len(models),
        "Wrong model score axis.",
    )
    pools = {}
    for pool in r.POOLS:
        for method in METHODS:
            indices = [
                i
                for i, x in enumerate(models)
                if x["pool"] == pool and x["method"] == method
            ]
            expected = 5 if method == "tuned_selected" else 1
            require(len(indices) == expected, "Missing/repeated model in pool average.")
            seeds = {models[i]["seed"] for i in indices}
            require(
                seeds == (set(r.SEEDS) if expected == 5 else {None}),
                "Wrong seed identity in pool average.",
            )
            pools[pool, method] = scores[..., indices].mean(axis=-1)
    budgets = {}
    for size in BUDGETS:
        pp = [p for p in r.POOLS if budget(p) == size]
        require(len(pp) == 3, "Budget does not have three pools.")
        for method in METHODS:
            budgets[size, method] = np.mean([pools[p, method] for p in pp], axis=0)
    return pools, budgets


def train_substitutions():
    folder = TRAIN / "uniform_compact_s42"
    heads = r.read(folder / "selection_review.json")["heads"]
    original = {x["head"]: x for x in heads if x["family"] == "tuned"}
    candidates = next(
        x["candidates"]
        for x in r.read(folder / "candidate_scores.json")
        if x["family"] == "tuned" and x["head"] == "hate_type"
    )
    configs = {x["id"]: x for x in r.read(r.PACKAGE / "data/configs.json")}
    result = []
    for family, cfg in [
        ("fixed_unweighted", "l1_e20_u"),
        ("fixed_balanced", "l1_e20_b"),
    ]:
        chosen = min(
            (x for x in candidates if x["config"] == cfg),
            key=lambda x: tr.neural_rank(x, configs),
        )
        result.append(
            dict(
                family=family,
                configuration=cfg,
                selected=chosen,
                gate=original["hate"]["selected"],
                selection_scope="Saved TRAIN OOF under original tuned hate gate",
                evaluation_scores_used=False,
            )
        )
    return result


def runtime_estimate():
    jobs = r.read(TRAIN / "uniform_compact_s42/canonical_jobs.json")
    final = sorted(
        (x for x in jobs if x["job"]["mode"] == "final"),
        key=lambda x: datetime.fromisoformat(x["started_utc"]),
    )
    gaps = []
    for a, b in zip(final, final[1:]):
        if (
            a["job"]["head"] != "hate_type"
            or a["session_sha256"] != b["session_sha256"]
        ):
            continue
        seconds = (
            datetime.fromisoformat(b["started_utc"])
            - datetime.fromisoformat(a["started_utc"])
        ).total_seconds()
        require(
            seconds >= a["total_seconds"],
            "Sequential final-job timing is inconsistent.",
        )
        gaps.append(
            dict(
                job=a["job"]["id"],
                next_job=b["job"]["id"],
                seconds_until_next_job=seconds,
                recorded_fit_and_export_seconds=a["total_seconds"],
                residual_including_inference_and_checks=seconds - a["total_seconds"],
            )
        )
    require(len(gaps) >= 10, "Insufficient same-session type-job timing evidence.")
    ordinary = [
        x
        for x in jobs
        if x["job"]["mode"] == "search"
        and x["job"]["head"] == "hate_type"
        and x["job"]["config"] == "l0_e2_u"
        and x["state"] == "trained"
    ]
    require(len(ordinary) == 9, "Two-epoch nonempty CV fit coverage differs.")
    return dict(
        estimate_only=True,
        server_availability_checked=False,
        historical_type_final_job_intervals=gaps,
        median_seconds_per_type_final_job=float(
            np.median([x["seconds_until_next_job"] for x in gaps])
        ),
        largest_seconds_per_type_final_job=max(
            x["seconds_until_next_job"] for x in gaps
        ),
        five_jobs_at_historical_median_seconds=5
        * float(np.median([x["seconds_until_next_job"] for x in gaps])),
        five_jobs_at_historical_maximum_seconds=5
        * max(x["seconds_until_next_job"] for x in gaps),
        two_epoch_CV_fit_seconds=[x["total_seconds"] for x in ordinary],
        bound="Intervals include the previous fit/export, full 19082-input type inference, checks and transition overhead. Different controller startup and new helper validation are additional; this is not a current live-server benchmark.",
        proposed_fits=5,
        inference_rows_per_fit=19082,
        retained_weights_approx_GB=2.5,
        preserve_originals=True,
        same_250_decimal_GB_cap=True,
        no_new_execution=True,
    )


def main():
    run = BASE / "runs" / uuid.uuid4().hex[:12]
    run.mkdir(parents=True, exist_ok=False)
    before = time.perf_counter()
    sources = [
        Path(__file__),
        Path(r.__file__),
        Path(tr.__file__),
        BASE / "analysis_contract.json",
        BASE / "update_required_TRAIN_proposal.json",
    ]
    save(
        run / "started.json",
        dict(
            utc=datetime.now(timezone.utc).isoformat(),
            sources={str(x): r.sha(x) for x in sources},
            scope="Part 2 primary equal-budget evaluation and sparse-pool descriptions",
            new_training=False,
            new_Qwen_calls=False,
        ),
    )
    try:
        with r.offline(), threadpool_limits(limits=2):
            contract = r.read(BASE / "analysis_contract.json")
            require(
                contract["primary"]["family_size"] == 24
                and contract["primary"]["bootstrap_draws"] == DRAWS
                and contract["primary"]["bootstrap_rng"] == SEED
                and contract["primary"]["confidence"] == 1 - ALPHA,
                "Changed primary comparison contract.",
            )
            for name, expected in contract["source_bindings"].items():
                r.checked_file(name, expected)
            substitutions = train_substitutions()
            save(run / "component_substitution_TRAIN_choices.json", substitutions)
            save(run / "retraining_runtime_estimate.json", runtime_estimate())
            study = r.GpuStudy()
            linear = r.LinearStudy(study)
            context = r.SharedContext(study, linear)
            matrices = {e: [] for e in r.PRIMARY}
            rosters = {e: [] for e in r.PRIMARY}
            extra = []
            sources_read = {}
            for pool in r.POOLS:
                print("Reading saved primary predictions:", pool, flush=True)
                with study.pool(pool) as saved:
                    for seed in r.SEEDS:
                        values = saved.predictions("tuned", "selected", seed)
                        for endpoint in r.PRIMARY:
                            matrices[endpoint].append(
                                context.aligned(values, "broad", endpoint)
                            )
                            rosters[endpoint].append(
                                model(pool, "tuned_selected", seed)
                            )
                        if pool == "uniform_compact_s42":
                            for spec in substitutions:
                                probs = saved.final_probabilities(
                                    saved.job("hate_type", spec["family"], seed)
                                )
                                raw = (probs >= spec["selected"]["threshold"]).astype(
                                    np.uint8
                                )
                                gated = raw.copy()
                                gated[values["hate"][:, 1] == 0] = 0
                                composite = dict(
                                    values, hate_type=gated, hate_type_ungated=raw
                                )
                                extra.append(
                                    (
                                        model(
                                            pool,
                                            "type_"
                                            + spec["family"]
                                            + "_with_tuned_gate",
                                            seed,
                                        ),
                                        context.aligned(
                                            composite, "broad", "en_test/hate_type"
                                        ),
                                    )
                                )
                    sources_read[pool] = saved.store.read_bindings
                    save(
                        run / "archive_bindings" / (pool + ".json"), sources_read[pool]
                    )
                for method, name in [
                    ("bge", "bge_selected_fold42"),
                    ("frozen_roberta", "frozen_roberta_selected_fold42"),
                ]:
                    values = linear.predictions(pool, method, 42, "selected")
                    for endpoint in r.PRIMARY:
                        matrices[endpoint].append(
                            context.aligned(values, "broad", endpoint)
                        )
                        rosters[endpoint].append(model(pool, name))
                linear.release_pool()
            for ident, pred in extra:
                rosters["en_test/hate_type"].append(ident)
                matrices["en_test/hate_type"].append(pred)
            endpoint_reports = {}
            primary = []
            all_sources = study.bindings | linear.bindings | context.bindings
            save(run / "source_bindings.json", all_sources)
            save(run / "context.json", context.summary())
            for endpoint in r.PRIMARY:
                name = slug(endpoint)
                view = context.views["broad", endpoint]
                y = view["truth"]
                support = [
                    view["endpoint"]["all_labels"].index(x)
                    for x in view["endpoint"]["macro_labels"]
                ]
                preds = np.stack(matrices[endpoint]).astype(np.uint8)
                models = rosters[endpoint]
                require(
                    len(models)
                    == len({x["id"] for x in models})
                    == (73 if endpoint == "en_test/hate_type" else 63),
                    "Primary model matrix incomplete.",
                )
                metadata = dict(
                    endpoint=endpoint,
                    models=models,
                    labels=view["endpoint"]["all_labels"],
                    macro_labels=view["endpoint"]["macro_labels"],
                    supported_indices=support,
                    clusters=view["clusters"],
                    rows=len(y),
                    ids=[str(x["id"]) for x in view["records"]],
                    positions=view["positions"],
                    gold_support=view["endpoint"]["gold_support"],
                )
                save(run / (name + "_metadata.json"), metadata)
                array(
                    run / (name + "_predictions.npz"),
                    truth=y.astype(np.uint8),
                    predictions=preds,
                )
                scores, counts = point_scores(y, preds, support)
                for i, pred in enumerate(preds):
                    reference = np.mean(
                        [
                            f1_score(y[:, j], pred[:, j], zero_division=0)
                            for j in support
                        ]
                    )
                    require(
                        abs(reference - scores[i]) < 1e-12,
                        "Independent sklearn point score differs.",
                    )
                array(run / (name + "_point_counts.npz"), scores=scores, **counts)
                draw, info = bootstrap(
                    y,
                    preds,
                    view["clusters"],
                    support,
                    progress=lambda n: print(
                        endpoint, n, "/", DRAWS, "paired cluster draws", flush=True
                    ),
                )
                array(run / (name + "_bootstrap.npz"), scores=draw)
                pool_points, absolute = aggregation(models, scores)
                pool_draws, absolute_draws = aggregation(models, draw)
                point_rows = []
                absolute_rows = []
                loo = []
                for pool in r.POOLS:
                    for method in METHODS:
                        ix = [
                            i
                            for i, m in enumerate(models)
                            if m["pool"] == pool and m["method"] == method
                        ]
                        values = scores[ix]
                        point_rows.append(
                            dict(
                                pool=pool,
                                budget=budget(pool),
                                method=method,
                                mean=float(pool_points[pool, method]),
                                individual=[
                                    dict(seed=models[i]["seed"], f1=float(scores[i]))
                                    for i in ix
                                ],
                                minimum=float(values.min()),
                                maximum=float(values.max()),
                                seed_sd=(
                                    None
                                    if len(values) == 1
                                    else float(values.std(ddof=1))
                                ),
                            )
                        )
                for size in BUDGETS:
                    pools = [p for p in r.POOLS if budget(p) == size]
                    for method in METHODS:
                        values = [float(pool_points[p, method]) for p in pools]
                        absolute_rows.append(
                            dict(
                                budget=size,
                                method=method,
                                mean=float(absolute[size, method]),
                                descriptive_ci95=np.quantile(
                                    absolute_draws[size, method], [0.025, 0.975]
                                ).tolist(),
                                pool_values=values,
                                pool_min=min(values),
                                pool_max=max(values),
                            )
                        )
                    for method, contrast in [
                        ("tuned_selected", "tuned_selected_minus_BGE_fold42"),
                        (
                            "frozen_roberta_selected_fold42",
                            "frozen_RoBERTa_fold42_minus_BGE_fold42",
                        ),
                    ]:
                        delta = (
                            absolute_draws[size, method]
                            - absolute_draws[size, METHODS[1]]
                        )
                        ci = np.quantile(delta, [ALPHA / 2, 1 - ALPHA / 2]).tolist()
                        primary.append(
                            dict(
                                endpoint=endpoint,
                                budget=size,
                                contrast=contrast,
                                left=float(absolute[size, method]),
                                right=float(absolute[size, METHODS[1]]),
                                difference=float(
                                    absolute[size, method] - absolute[size, METHODS[1]]
                                ),
                                family_interval=ci,
                                confidence=1 - ALPHA,
                                ci95_descriptive=np.quantile(
                                    delta, [0.025, 0.975]
                                ).tolist(),
                                excludes_zero=bool(ci[0] > 0 or ci[1] < 0),
                                pool_differences=[
                                    float(
                                        pool_points[p, method]
                                        - pool_points[p, METHODS[1]]
                                    )
                                    for p in pools
                                ],
                            )
                        )
                        for omitted in pools:
                            kept = [p for p in pools if p != omitted]
                            left = np.mean([pool_points[p, method] for p in kept])
                            right = np.mean([pool_points[p, METHODS[1]] for p in kept])
                            loo.append(
                                dict(
                                    budget=size,
                                    contrast=contrast,
                                    omitted_pool=omitted,
                                    left=float(left),
                                    right=float(right),
                                    difference=float(left - right),
                                    original_difference=float(
                                        absolute[size, method]
                                        - absolute[size, METHODS[1]]
                                    ),
                                    change_from_three_pool_difference=float(
                                        left
                                        - right
                                        - (
                                            absolute[size, method]
                                            - absolute[size, METHODS[1]]
                                        )
                                    ),
                                    exploratory=True,
                                    primary_replaced=False,
                                )
                            )
                component = []
                if endpoint == "en_test/hate_type":
                    original = pool_points["uniform_compact_s42", "tuned_selected"]
                    for spec in substitutions:
                        method = "type_" + spec["family"] + "_with_tuned_gate"
                        ix = [i for i, x in enumerate(models) if x["method"] == method]
                        require(
                            len(ix) == 5
                            and {models[i]["seed"] for i in ix} == set(r.SEEDS),
                            "Missing component substitution seed.",
                        )
                        point = float(scores[ix].mean())
                        changed_budget = float(
                            absolute[147, "tuned_selected"] + (point - original) / 3
                        )
                        change_draws = (
                            draw[:, ix].mean(axis=1)
                            - pool_draws["uniform_compact_s42", "tuned_selected"]
                        ) / 3
                        component.append(
                            dict(
                                method=method,
                                TRAIN_choice=spec,
                                per_seed=scores[ix].tolist(),
                                pool_mean=point,
                                original_pool_mean=float(original),
                                pool_difference=float(point - original),
                                three_pool_tuned_mean_if_substituted=changed_budget,
                                change_to_three_pool_mean=float((point - original) / 3),
                                changed_contrast_to_BGE=float(
                                    changed_budget - absolute[147, METHODS[1]]
                                ),
                                descriptive_ci95_of_budget_change=np.quantile(
                                    change_draws, [0.025, 0.975]
                                ).tolist(),
                                interpretation="Descriptive existing-weights substitution with unchanged tuned gate; not the proposed new two-epoch model and not a replacement primary result.",
                            )
                        )
                endpoint_reports[endpoint] = dict(
                    rows=len(y),
                    clusters=info["clusters"],
                    labels=metadata["labels"],
                    macro_labels=metadata["macro_labels"],
                    per_pool=point_rows,
                    absolute=absolute_rows,
                    leave_one_pool_out=loo,
                    component_substitutions=component,
                    bootstrap=info,
                )
                save(run / (name + "_results.json"), endpoint_reports[endpoint])
                print(
                    endpoint,
                    "PRIMARY SCORING COMPLETE",
                    [
                        (x["budget"], x["method"], round(x["mean"], 6))
                        for x in absolute_rows
                    ],
                    flush=True,
                )
            require(
                len(primary) == 24
                and {(x["budget"], x["endpoint"], x["contrast"]) for x in primary}
                == {
                    (x["budget"], x["endpoint"], x["contrast"])
                    for x in contract["primary"]["comparisons"]
                },
                "Primary family changed.",
            )
            save(
                run / "primary_comparisons.json",
                dict(
                    family_size=24,
                    comparisons=primary,
                    draws=DRAWS,
                    seed=SEED,
                    confidence=1 - ALPHA,
                    conditional_on_fixed_fits_and_pools=True,
                ),
            )
            for name, expected in r.read(run / "started.json")["sources"].items():
                require(
                    r.sha(name) == expected, "Analysis source changed during execution."
                )
            require(
                "torch" not in sys.modules,
                "Model runtime imported by offline analysis.",
            )
            outputs = {
                str(p.relative_to(run)): r.sha(p) for p in run.rglob("*") if p.is_file()
            }
            save(
                run / "complete.json",
                dict(
                    status="primary_equal_budget_analysis_complete",
                    utc=datetime.now(timezone.utc).isoformat(),
                    seconds=time.perf_counter() - before,
                    outputs=outputs,
                    primary_contrasts=24,
                    point_models_checked=262,
                    bootstrap_model_scores=262 * DRAWS,
                    training_calls=0,
                    Qwen_calls=0,
                    model_weights_loaded=0,
                    original_evidence_changed=False,
                    independent_final_review_pending=True,
                ),
            )
            print("PART 2 MAIN ANALYSIS COMPLETE", str(run), flush=True)
    except BaseException as exc:
        save(
            run / "stopped.json",
            dict(
                type=type(exc).__name__,
                message=str(exc),
                utc=datetime.now(timezone.utc).isoformat(),
            ),
        )
        raise


if __name__ == "__main__":
    main()

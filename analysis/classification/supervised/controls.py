from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits

from analysis.classification.supervised import archive as r
from analysis.classification.supervised import metrics as m
from analysis.classification.supervised import type_update as s

BASE = r.ROOT / "experiments/astra/sq2_gpu_controls_review_20261103_v1"
MAIN = m.BASE / "runs/9deace7a4235"
TRAIN = m.TRAIN
FAMILIES = r.FAMILIES
METHODS = FAMILIES + ("bge", "frozen_roberta")
SCALARS = (
    "macro_f1",
    "micro_f1",
    "micro_precision",
    "micro_recall",
    "exact_set",
    "hamming_error",
    "predicted_labels",
    "empty_rate",
    "predicted_hate",
)
CONTRASTS = tuple(
    (a, b)
    for v in ("default", "selected")
    for a, b in (
        (f"fixed_balanced/{v}", f"fixed_unweighted/{v}"),
        (f"tuned/{v}", f"fixed_unweighted/{v}"),
        (f"tuned/{v}", f"fixed_balanced/{v}"),
    )
) + tuple((f"{f}/selected", f"{f}/default") for f in METHODS)
save = s.save


def variants(head, neural=True):
    if not neural or head == "hate":
        return ("default", "selected")
    if head == "severity":
        return ("default", "selected", "ungated")
    return (
        "default",
        "gate_only",
        "head_only",
        "selected",
        "ungated_default",
        "ungated_selected",
    )


def factorial(default, selected, head):

    r.require(
        head in r.HEADS and set(default) == set(selected), "Wrong factorial inputs."
    )
    a, b = default[head], selected[head]
    out = {
        "default": (a, default["hate"][:, 1]),
        "selected": (b, selected["hate"][:, 1]),
    }
    if head == "hate":
        return out
    gd, gs = default["hate"][:, 1].astype(bool), selected["hate"][:, 1].astype(bool)
    ud, us = default[head + "_ungated"], selected[head + "_ungated"]
    if head == "severity":
        r.require(np.array_equal(ud, us), "Severity cannot change its argmax cutoff.")
        out["ungated"] = (ud, np.ones(len(gd), dtype=np.uint8))
    else:
        out["gate_only"] = (ud * gs[:, None], gs)
        out["head_only"] = (us * gd[:, None], gd)
        out["ungated_default"] = (ud, np.ones(len(gd), dtype=np.uint8))
        out["ungated_selected"] = (us, np.ones(len(gd), dtype=np.uint8))
    return {v: out[v] for v in variants(head)}


def aggregate(models, values):
    "Average five fitted seeds, then the three equally weighted fixed pools."
    r.require(
        values.ndim in (1, 2) and values.shape[-1] == len(models), "Wrong score axis."
    )
    r.require(len({x["id"] for x in models}) == len(models), "Duplicate model-view.")
    keys = sorted({x["method"] + "/" + x["variant"] for x in models})
    pool_values, budget_values = {}, {}
    for key in keys:
        method, variant = key.split("/")
        r.require(method in METHODS, "Unknown method.")
        for pool in r.POOLS:
            ix = [
                i
                for i, x in enumerate(models)
                if x["pool"] == pool
                and x["method"] == method
                and x["variant"] == variant
            ]
            expected = set(r.SEEDS) if method in FAMILIES else {None}
            r.require(
                len(ix) == len(expected)
                and {models[i]["seed"] for i in ix} == expected,
                "Missing, duplicate or wrong seed in control mean.",
            )
            pool_values[pool + "/" + key] = values[..., ix].mean(axis=-1)
        for budget in m.BUDGETS:
            pp = [p for p in r.POOLS if m.budget(p) == budget]
            r.require(len(pp) == 3, "Wrong budget roster.")
            budget_values[str(budget) + "/" + key] = np.mean(
                [pool_values[p + "/" + key] for p in pp], axis=0
            )
    return pool_values, budget_values


def decomposition(a, g, h, z):

    head = ((h - a) + (z - g)) / 2
    gate = ((g - a) + (z - h)) / 2
    r.require(
        np.allclose(head + gate, z - a, rtol=0, atol=1e-14),
        "Decomposition does not sum.",
    )
    return dict(
        total=z - a,
        head=head,
        gate=gate,
        interaction=z - h - g + a,
        head_first=h - a,
        head_second=z - g,
        gate_first=g - a,
        gate_second=z - h,
    )


def interval(value, draws):
    return dict(
        value=float(value),
        low=float(np.quantile(draws, 0.025)),
        high=float(np.quantile(draws, 0.975)),
    )


def freeze():
    r.offline()
    BASE.mkdir(parents=True, exist_ok=True)
    contract = dict(
        part=3,
        title="What training choices and cutoffs change",
        utc=datetime.now(timezone.utc).isoformat(),
        known_prior_results="Parts 1/2 and the separate five-type-update sensitivity have been read; these supporting analyses do not reset prior exposure.",
        endpoints=list(r.PRIMARY),
        families=list(FAMILIES),
        methods=list(METHODS),
        variants={h: list(variants(h)) for h in r.HEADS},
        comparisons=[list(x) for x in CONTRASTS],
        uncertainty=dict(
            draws=10000,
            seed=20261020,
            interval=0.95,
            paired_normalized_text_clusters=True,
            fixed_endpoint_label_support=True,
            conditional_on_fixed_fits_and_pools=True,
            supporting_descriptive_not_multiplicity_adjusted=True,
            original_24_primary_family_unchanged=True,
        ),
        planned_diagnostics=[
            "All default/selected family controls and fold42 linear references",
            "Two-by-two fine cutoff and matching hate-cutoff decomposition, both orders and ungated scores",
            "Macro/micro F1, precision/recall, exact sets, label counts, emptiness and gate rates",
            "All family/pool/seed variation; previously flagged uniform_1000_s43 hate probabilities, ranking and actual updates",
            "TRAIN-only best candidates by learning rate, duration and loss; no held-out attribution to one hyperparameter",
            "Cross-check all original tuned/linear primary scores and draws; preserve supplement separately",
        ],
        exclusions=[
            "No test cutoff search",
            "No seed/pool exclusion",
            "No new fit, inference, Qwen, server action or weight loading",
            "No final SQ2 synthesis or replacement of original primary evidence",
        ],
        inputs={
            str(p.relative_to(r.ROOT)): r.sha(p)
            for p in [
                m.BASE / "part2_complete.json",
                MAIN / "complete.json",
                r.ROOT
                / "experiments/astra/sq2_gpu_training_review_20261103_v1/part1_complete.json",
                r.ROOT
                / "experiments/astra/sq2_type_sensitivity_20261103_v1/supplement_analysis_complete.json",
                r.ROOT
                / "Thesis_Log/astra_result/astra_sq2_stage_b_protocol_20261020_v1.md",
                Path(__file__),
                Path(m.__file__),
                Path(r.__file__),
                Path(s.__file__),
            ]
        },
    )
    save(BASE / "analysis_contract.json", contract)
    print("Part 3 contract saved; no new outcomes calculated.", flush=True)


def training_summary(run):
    output = []
    bindings = {}
    for pool in r.POOLS:
        for name in (
            "selection_review.json",
            "candidate_scores.json",
            "canonical_jobs.json",
        ):
            p = TRAIN / pool / name
            bindings[str(p.relative_to(r.ROOT))] = r.sha(p)
        chosen = r.read(TRAIN / pool / "selection_review.json")["heads"]
        candidates = r.read(TRAIN / pool / "candidate_scores.json")
        for item in chosen:
            entry = dict(pool=pool, budget=m.budget(pool), **item)
            if item["family"] == "tuned":
                cc = next(
                    x["candidates"]
                    for x in candidates
                    if x["head"] == item["head"] and x["family"] == "tuned"
                )
                configs = {x["id"]: x for x in r.read(r.PACKAGE / "data/configs.json")}
                profiles = {}
                for key in ("learning_rate", "epochs", "loss"):
                    values = sorted({configs[x["config"]][key] for x in cc})
                    profiles[key] = [
                        dict(
                            value=v,
                            best_mean_score=max(
                                x["mean_score"]
                                for x in cc
                                if configs[x["config"]][key] == v
                            ),
                        )
                        for v in values
                    ]
                entry["TRAIN_profiles_conditioned_on_same_family_selected_gate"] = (
                    profiles
                )
            output.append(entry)
    save(run / "training_choices.json", output)
    return bindings


def run():
    r.offline()
    plan = r.read(BASE / "analysis_contract.json")
    for name, digest in plan["inputs"].items():
        r.checked_file(r.ROOT / name, digest)
    folder = BASE / "runs" / uuid.uuid4().hex[:12]
    save(
        folder / "started.json",
        dict(
            utc=datetime.now(timezone.utc).isoformat(),
            contract_sha256=r.sha(BASE / "analysis_contract.json"),
        ),
    )
    print("RUN " + str(folder), flush=True)
    started = time.monotonic()
    try:
        with threadpool_limits(limits=2):
            study = r.GpuStudy()
            linear = r.LinearStudy(study)
            context = r.SharedContext(study, linear)
            storage = {
                e: dict(
                    models=[],
                    predictions=[],
                    gates=[],
                    raw_models=[],
                    probabilities=[],
                    gate_probabilities=[],
                )
                for e in r.PRIMARY
            }
            choices, ranking = [], []
            train_bindings = training_summary(folder)
            for pool in r.POOLS:
                with study.pool(pool) as saved:
                    for method in METHODS:
                        neural = method in FAMILIES
                        for seed in (r.SEEDS if neural else (None,)):
                            if neural:
                                default, selected = (
                                    saved.predictions(method, v, seed)
                                    for v in r.POLICIES
                                )
                                settings = saved.families[method]
                                probs = {
                                    h: saved.final_probabilities(
                                        saved.job(h, method, seed)
                                    )
                                    for h in r.HEADS
                                }
                            else:
                                default, selected = (
                                    linear.predictions(pool, method, 42, v)
                                    for v in r.POLICIES
                                )
                                settings, probs, _ = linear.setting(pool, method, 42)
                            raw_id = f"{pool}/{method}/" + (
                                "fold42" if seed is None else f"s{seed}"
                            )
                            identity = dict(
                                id=raw_id,
                                pool=pool,
                                budget=m.budget(pool),
                                method=method,
                                seed=seed,
                            )
                            if seed in (None, 10):
                                choices.append(
                                    dict(pool=pool, method=method, settings=settings)
                                )
                            for endpoint in r.PRIMARY:
                                view = context.views["broad", endpoint]
                                head = view["endpoint"]["dimension"]
                                ix, cols = view["positions"], view["label_indices"]
                                st = storage[endpoint]
                                raw_index = len(st["raw_models"])
                                st["raw_models"].append(
                                    identity
                                    | dict(
                                        head_cutoff=(
                                            None
                                            if head == "severity"
                                            else settings[head]["selected"]["threshold"]
                                        ),
                                        hate_cutoff=settings["hate"]["selected"][
                                            "threshold"
                                        ],
                                    )
                                )
                                st["probabilities"].append(probs[head][ix][:, cols])
                                st["gate_probabilities"].append(probs["hate"][ix, 1])
                                fv = factorial(default, selected, head)
                                for variant in variants(head, neural):
                                    p, gate = fv[variant]
                                    st["models"].append(
                                        identity
                                        | dict(
                                            id=raw_id + "/" + variant,
                                            variant=variant,
                                            raw_index=raw_index,
                                        )
                                    )
                                    st["predictions"].append(
                                        p[ix][:, cols].astype(np.uint8)
                                    )
                                    st["gates"].append(gate[ix].astype(np.uint8))
                                if head == "hate":
                                    y = view["truth"][:, 1]
                                    p = probs["hate"][ix, 1]
                                    ranking.append(
                                        identity
                                        | dict(
                                            roc_auc=roc_auc_score(y, p),
                                            average_precision=average_precision_score(
                                                y, p
                                            ),
                                            brier=float(np.mean((p - y) ** 2)),
                                            unique_probabilities=len(np.unique(p)),
                                            probability_quantiles={
                                                g: np.quantile(
                                                    p[y == b],
                                                    [0, 0.01, 0.1, 0.5, 0.9, 0.99, 1],
                                                ).tolist()
                                                for g, b in [
                                                    ("negative", 0),
                                                    ("positive", 1),
                                                ]
                                            },
                                            cutoff=settings["hate"]["selected"][
                                                "threshold"
                                            ],
                                        )
                                    )
                    save(
                        folder / "archive_bindings" / f"{pool}.json",
                        saved.store.read_bindings,
                    )
                    save(folder / "final_jobs" / f"{pool}.json", saved.job_records)
                linear.release_pool()
                print(
                    pool + ": all saved families and cutoffs reconstructed.", flush=True
                )
            save(folder / "settings.json", choices)
            save(folder / "hate_probability_diagnostics.json", ranking)
            save(
                folder / "source_bindings.json",
                dict(
                    study=study.bindings,
                    linear=linear.bindings,
                    context=context.bindings,
                    training=train_bindings,
                ),
            )
            all_results = {}
            for endpoint in r.PRIMARY:
                st, view = storage.pop(endpoint), context.views["broad", endpoint]
                head = view["endpoint"]["dimension"]
                labels = view["endpoint"]["all_labels"]
                support = [labels.index(x) for x in view["endpoint"]["macro_labels"]]
                r.require(
                    support == list(range(len(labels))),
                    "Quality macro support requires review.",
                )
                y = view["truth"].astype(np.uint8)
                pred = np.asarray(st.pop("predictions"), dtype=np.uint8)
                gates = np.asarray(st.pop("gates"), dtype=np.uint8)
                probabilities = np.asarray(st.pop("probabilities"))
                gp = np.asarray(st.pop("gate_probabilities"))
                slug = m.slug(endpoint)
                metadata = dict(
                    endpoint=endpoint,
                    head=head,
                    labels=labels,
                    supported_indices=support,
                    models=st["models"],
                    raw_models=st["raw_models"],
                    rows=len(y),
                    ids=[str(x["id"]) for x in view["records"]],
                    positions=view["positions"],
                    clusters=view["clusters"],
                    gold_support=view["endpoint"]["gold_support"],
                )
                save(folder / f"{slug}_metadata.json", metadata)
                m.array(
                    folder / f"{slug}_arrays.npz",
                    truth=y,
                    predictions=pred,
                    gates=gates,
                    probabilities=probabilities,
                    gate_probabilities=gp,
                )
                q = s.quality(y, pred, gates)
                q.pop("hate_and_type_exact")
                save(folder / f"{slug}_quality.json", q)
                draws, info = m.bootstrap(
                    y,
                    pred,
                    view["clusters"],
                    support,
                    progress=lambda n: print(
                        endpoint + f": paired resamples {n}/10000", flush=True
                    ),
                )
                m.array(folder / f"{slug}_bootstrap.npz", scores=draws)
                pp, bp = aggregate(st["models"], q["macro_f1"])
                pd, bd = aggregate(st["models"], draws)
                pq, bq = {}, {}
                for key in SCALARS:
                    a, b = aggregate(st["models"], q[key])
                    pq[key], bq[key] = a, b
                comparisons, components = [], []
                for budget in m.BUDGETS:
                    for a, b in CONTRASTS:
                        ka, kb = f"{budget}/{a}", f"{budget}/{b}"
                        comparisons.append(
                            dict(
                                budget=budget,
                                after=a,
                                before=b,
                                **interval(bp[ka] - bp[kb], bd[ka] - bd[kb]),
                                pool_differences={
                                    p: pp[f"{p}/{a}"] - pp[f"{p}/{b}"]
                                    for p in r.POOLS
                                    if m.budget(p) == budget
                                },
                            )
                        )
                    if head in ("hate_type", "target_group"):
                        for family in FAMILIES:
                            keys = [
                                f"{budget}/{family}/{v}"
                                for v in (
                                    "default",
                                    "gate_only",
                                    "head_only",
                                    "selected",
                                )
                            ]
                            point = decomposition(*(bp[k] for k in keys))
                            boot = decomposition(*(bd[k] for k in keys))
                            components.append(
                                dict(
                                    budget=budget,
                                    family=family,
                                    **{k: interval(point[k], boot[k]) for k in point},
                                )
                            )

                old_meta = r.read(MAIN / f"{slug}_metadata.json")
                old_p = r.arrays_from_bytes(
                    (MAIN / f"{slug}_predictions.npz").read_bytes()
                )
                old_q = r.arrays_from_bytes(
                    (MAIN / f"{slug}_point_counts.npz").read_bytes()
                )
                old_b = r.arrays_from_bytes(
                    (MAIN / f"{slug}_bootstrap.npz").read_bytes()
                )
                matches = []
                for i, x in enumerate(old_meta["models"]):
                    method = {
                        "tuned_selected": "tuned",
                        "bge_selected_fold42": "bge",
                        "frozen_roberta_selected_fold42": "frozen_roberta",
                    }.get(x["method"])
                    if method is None:
                        continue
                    found = [
                        j
                        for j, z in enumerate(st["models"])
                        if z["pool"] == x["pool"]
                        and z["method"] == method
                        and z["seed"] == x["seed"]
                        and z["variant"] == "selected"
                    ]
                    r.require(len(found) == 1, "Old primary model missing.")
                    j = found[0]
                    r.require(
                        np.array_equal(y, old_p["truth"])
                        and np.array_equal(pred[j], old_p["predictions"][i])
                        and q["macro_f1"][j] == old_q["scores"][i]
                        and np.array_equal(draws[:, j], old_b["scores"][:, i]),
                        "Original primary evidence changed.",
                    )
                    matches.append(x["id"])
                r.require(
                    len(matches) == 63, "Not all original main methods were checked."
                )
                result = dict(
                    endpoint=endpoint,
                    bootstrap=info,
                    absolute={k: interval(v, bd[k]) for k, v in bp.items()},
                    pool_scores=pp,
                    quality_budget=bq,
                    quality_pool=pq,
                    comparisons=comparisons,
                    cutoff_components=components,
                    original_primary_exact_matches=matches,
                )
                save(folder / f"{slug}_results.json", result)
                all_results[endpoint] = dict(
                    model_views=len(pred),
                    raw_models=len(st["raw_models"]),
                    rows=len(y),
                    comparisons=len(comparisons),
                    components=len(components),
                )
                print(
                    endpoint
                    + ": complete; all 63 original primary model results unchanged.",
                    flush=True,
                )
                del pred, gates, probabilities, gp, draws, old_b, old_p, old_q, st
            r.require(
                "torch" not in sys.modules,
                "Weight/training library unexpectedly loaded.",
            )
            outputs = {
                str(p.relative_to(folder)): r.sha(p)
                for p in folder.rglob("*")
                if p.is_file()
            }
            save(
                folder / "complete.json",
                dict(
                    status="complete",
                    utc=datetime.now(timezone.utc).isoformat(),
                    seconds=time.monotonic() - started,
                    endpoints=all_results,
                    outputs=outputs,
                    offline=True,
                    training_calls=0,
                    Qwen_calls=0,
                    server_operations=0,
                    weights_loaded=0,
                ),
            )
            print("COMPLETE " + str(folder), flush=True)
    except BaseException as exc:
        save(
            folder / "stopped.json",
            dict(
                type=type(exc).__name__,
                message=str(exc),
                utc=datetime.now(timezone.utc).isoformat(),
            ),
        )
        raise


if __name__ == "__main__":
    if sys.argv[1:] == ["freeze"]:
        freeze()
    elif sys.argv[1:] == ["run"]:
        run()
    else:
        raise SystemExit("Use freeze or run.")

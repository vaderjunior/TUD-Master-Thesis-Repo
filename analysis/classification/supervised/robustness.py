from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.classification.supervised import archive as r
from analysis.classification.supervised import sources as s
from analysis.classification.supervised import metrics as m
from analysis.classification.supervised import controls as c

BASE = r.ROOT / "experiments/astra/sq2_gpu_robustness_review_20261104_v1"
MAIN = m.BASE / "runs/9deace7a4235"
METHODS = c.METHODS + ("bge_f43", "bge_f44")
save = s.save


def freeze():
    paths = [
        Path(__file__),
        Path(r.__file__),
        Path(s.__file__),
        Path(m.__file__),
        Path(c.__file__),
        s.BASE / "part6_complete.json",
        s.BASE / "extraction/complete.json",
        s.BASE / "analysis/complete.json",
        m.BASE / "part2_complete.json",
        MAIN / "complete.json",
        m.TRAIN / "complete.json",
        r.ROOT / "Thesis_Log/astra_result/astra_sq2_stage_b_protocol_20261020_v1.md",
        r.ROOT
        / "Thesis_Log/astra_result/astra_sq2_gpu_training_selection_20261103_v1.md",
        r.ROOT
        / "Thesis_Log/astra_result/astra_sq2_gpu_training_cutoffs_20261103_v1.md",
        r.ROOT / "Thesis_Log/astra_result/astra_sq2_type_update_results_20261103_v1.md",
        r.ROOT
        / "experiments/astra/sq2_type_sensitivity_20261103_v1/supplement_analysis_complete.json",
        r.LINEAR / "data/scopes.json",
        r.LINEAR / "data/inference.json",
    ]
    save(
        BASE / "analysis_contract.json",
        dict(
            part=7,
            utc=datetime.now(timezone.utc).isoformat(),
            scope="Offline robustness and limitations of the completed English GPU study. Prior primary results and supporting Parts 1–6 have already been examined. The following is supporting analysis-time scope, not new preregistration.",
            preserved=dict(
                primary_family=24,
                original_adjusted_confidence=1 - 0.05 / 24,
                original_results=True,
                pools=list(r.POOLS),
                seeds=list(r.SEEDS),
                policies=list(r.POLICIES),
            ),
            analyses=[
                "All eight broad endpoint scores: original five methods and default/selected policies; seeds within each pool, three-pool ranges and symmetric leave-one-pool/seed diagnostics.",
                "Tuned versus BGE fold42, frozen RoBERTa versus BGE fold42, and tuned versus fixed balanced. Report every pool and matched seed, five common-seed summaries, all 125 choices of one neural seed per pool, all 15 within-pool single-seed removals, without selecting a seed or replacing any average.",
                "Read the existing BGE fold assignments 43 and 44, preserving each fit, matching hate gate and original TRAIN-selected C/cutoff. Report each alternative and the equal mean of all three fold-assignment scores, not a prediction ensemble or independent extra pools.",
                "For all four primary endpoints, pair saved primary bootstrap draws with new alternative-fold draws after confirming identical text-cluster resampling. Supporting 95% intervals for changed baseline and comparison differences; the original 24 adjusted contrasts remain unchanged.",
                "Identify all original-core identities and normalized texts in the 19082-input union. Exclude a broad endpoint row if its identity OR normalized text occurs in that core. Report overlap and remaining rows, preserve original endpoint macro columns, and compute supporting primary contrasts on the remainder. Reuse unchanged draws when no row is removed.",
                "Natural-source sensitivity: leave each source out in turn where another source remains; equal-source mean of per-source macro F1 is a separate descriptive estimand. Reuse prior source scores when identical. Single-source fine endpoints cannot establish cross-source generalization.",
                "Reconstruct all 36 pairwise pool overlaps by identity and normalized text, shared memberships, source/head-specific overlaps, core/broad and cross-evaluation-phase overlap, and normalized TRAIN/evaluation intersections. Exact normalization does not establish absence of semantic duplicates or pretraining contamination.",
                "Inventory annotation missingness/known-empty values and per-head eligibility by source/phase. Preserve endpoint label support and contrast it with each TRAIN head's observed and selection-supported labels; do not impute missing fields or compute an unsupported four-field score.",
                "Integrate earlier evidence on prior evaluation exposure, reused pretraining, unequal search costs, unstable TRAIN selection, one-step initialization-only heads, the separate five-fit supplement and the retained weak trained seeds. No result-based exclusion, refit or new cutoff.",
            ],
            uncertainty=dict(
                new_draws=10000,
                seed=20261020,
                level=0.95,
                role="Supporting unadjusted and conditional on saved fits/pools. Finite seed/pool/fold enumerations are descriptive, not population confidence intervals.",
                pool_generalization="Only three overlapping non-nested pools per budget; do not claim a reliable population-of-training-pools interval or treat 15 fits as independent dataset replications.",
                fixed_support="Original endpoint macro columns stay fixed in exclusions and all resamples. Missing annotations remain excluded.",
            ),
            scope_boundaries=[
                "No fresh core endpoint benchmark or new gold labels",
                "No new semantic-neighbor search or training-set redesign",
                "No new sensitivity method selected from favorable outcomes",
                "No German/legal/Qwen analysis",
                "No training, inference, weight loading, server action or full backup audit",
            ],
            inputs={str(p.relative_to(r.ROOT)): r.sha(p) for p in paths},
        ),
    )
    print("Part 7 supporting scope frozen before new robustness outcomes.")


def prepare():
    for path, digest in r.read(BASE / "analysis_contract.json")["inputs"].items():
        r.checked_file(r.ROOT / path, digest)
    out = BASE / "prepared"
    out.mkdir(exist_ok=False)
    study = r.GpuStudy()
    linear = r.LinearStudy(study)
    old_models = r.read(s.BASE / "extraction/models.json")
    extra = []
    matrices = {}
    metadata = {}
    for name in r.read(s.BASE / "analysis/endpoints.json"):
        meta, z = s.load(name)
        metadata[name] = meta
        matrices[name] = dict(gated=[], raw=[], gate=[])
    choices = []
    for pool in r.POOLS:
        for fold in (43, 44):
            heads, probs, receipt = linear.setting(pool, "bge", fold)
            for head in r.HEADS:
                choices.append(
                    dict(
                        pool=pool,
                        fold=fold,
                        head=head,
                        selected=heads[head]["selected"],
                    )
                )
            for policy in r.POLICIES:
                method = "bge_f" + str(fold)
                extra.append(
                    dict(
                        id=f"{pool}/{method}/fold{fold}/{policy}",
                        method=method,
                        pool=pool,
                        budget=m.budget(pool),
                        variant=policy,
                        seed=None,
                        fold_assignment=fold,
                    )
                )
                values = linear.predictions(pool, "bge", fold, policy)
                for name, meta in metadata.items():
                    pos = meta["positions"]
                    head = meta["head"]
                    cols = (
                        [heads[head]["labels"].index(x) for x in meta["labels"]]
                        if head != "hate"
                        else [0, 1]
                    )
                    matrices[name]["gated"].append(values[head][pos][:, cols])
                    matrices[name]["raw"].append(
                        values[head + "_ungated"][pos][:, cols]
                    )
                    matrices[name]["gate"].append(values["hate"][pos, 1])
        linear.release_pool()
        print(pool + ": existing BGE fold43/fold44 predictions validated.", flush=True)
    r.require(len(extra) == 36 and len(old_models) == 306, "Wrong robustness roster.")
    models = old_models + extra
    r.require(len({x["id"] for x in models}) == len(models), "Duplicate model view.")
    save(out / "models.json", models)
    save(out / "alternative_choices.json", choices)
    for name, values in matrices.items():
        m.array(
            out / (m.slug(name) + "_extra.npz"),
            **{k: np.stack(v).astype(np.uint8) for k, v in values.items()},
        )
    save(
        out / "source_bindings.json", dict(study=study.bindings, linear=linear.bindings)
    )
    save(
        out / "complete.json",
        dict(
            status="complete",
            code_sha256=r.sha(Path(__file__)),
            old_model_views=306,
            extra_model_views=36,
            outputs={
                str(p.relative_to(out)): r.sha(p) for p in out.iterdir() if p.is_file()
            },
            training_calls=0,
            qwen_calls=0,
            weights_loaded=0,
        ),
    )
    print("Part 7 saved-output preparation complete; no model was run.")


def load(name):
    meta, z = s.load(name)
    path = BASE / "prepared" / (m.slug(name) + "_extra.npz")
    r.checked_file(path, r.read(BASE / "prepared/complete.json")["outputs"][path.name])
    extra = r.arrays_from_bytes(path.read_bytes())
    return meta, dict(
        truth=z["truth"],
        **{
            key: np.concatenate([z[key], extra[key]], axis=0)
            for key in ("gated", "raw", "gate")
        },
    )


def aggregate(models, values):
    values = np.asarray(values)
    r.require(values.shape[-1] == len(models), "Wrong model axis.")
    pools = {}
    budgets = {}
    for method, policy in sorted({(x["method"], x["variant"]) for x in models}):
        for pool in r.POOLS:
            ii = [
                j
                for j, x in enumerate(models)
                if (x["method"], x["variant"], x["pool"]) == (method, policy, pool)
            ]
            seeds = {x["seed"] for x in (models[j] for j in ii)}
            expected = set(r.SEEDS) if method in r.FAMILIES else {None}
            r.require(
                len(ii) == len(expected) and seeds == expected,
                "Incomplete seed/pool roster.",
            )
            pools[pool + "/" + method + "/" + policy] = values[..., ii].mean(axis=-1)
        for n in m.BUDGETS:
            budgets[f"{n}/{method}/{policy}"] = np.mean(
                [
                    pools[p + "/" + method + "/" + policy]
                    for p in r.POOLS
                    if m.budget(p) == n
                ],
                axis=0,
            )
    for pool in r.POOLS:
        for policy in r.POLICIES:
            if all(
                pool + "/" + method + "/" + policy in pools
                for method in ("bge", "bge_f43", "bge_f44")
            ):
                pools[pool + "/bge_fold_mean/" + policy] = np.mean(
                    [
                        pools[pool + "/" + method + "/" + policy]
                        for method in ("bge", "bge_f43", "bge_f44")
                    ],
                    axis=0,
                )
    for n in m.BUDGETS:
        for policy in r.POLICIES:
            if all(
                f"{n}/{method}/{policy}" in budgets
                for method in ("bge", "bge_f43", "bge_f44")
            ):
                budgets[f"{n}/bge_fold_mean/{policy}"] = np.mean(
                    [
                        budgets[f"{n}/{method}/{policy}"]
                        for method in ("bge", "bge_f43", "bge_f44")
                    ],
                    axis=0,
                )
    return pools, budgets


if __name__ == "__main__":
    with r.offline(), threadpool_limits(limits=2):
        if sys.argv[1:] == ["freeze"]:
            freeze()
        elif sys.argv[1:] == ["prepare"]:
            prepare()
        else:
            raise SystemExit("Use freeze or prepare.")

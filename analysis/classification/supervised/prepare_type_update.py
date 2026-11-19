from datetime import datetime, timezone
from pathlib import Path

from analysis.classification.supervised import archive as r
from analysis.classification.supervised.metrics import save

BASE = r.ROOT / "experiments/astra/sq2_type_sensitivity_20261103_v1"
EXECUTION = r.ROOT / "experiments/astra/sq2_type_update_20261103_v1"
MAIN = r.ROOT / "experiments/astra/sq2_gpu_main_review_20261103_v1/runs/9deace7a4235"


def main():
    sources = [
        Path(__file__),
        EXECUTION / "execution_complete.json",
        EXECUTION / "execution_independent_review.json",
        EXECUTION / "execution_review.zip",
        MAIN / "complete.json",
        MAIN.parent.parent / "analysis_contract.json",
        MAIN.parent.parent / "part2_complete.json",
        MAIN.parent.parent / "update_required_TRAIN_proposal.json",
        r.ROOT / "deploy/type_update/protocol.json",
        Path(r.__file__),
    ]
    save(
        BASE / "analysis_contract.json",
        dict(
            status="prospective_supplementary_analysis",
            utc=datetime.now(timezone.utc).isoformat(),
            sources={str(p): r.sha(p) for p in sources},
            seeds=list(r.SEEDS),
            changed_pool="uniform_compact_s42",
            changed_head="hate_type",
            primary_endpoint="broad/en_test/hate_type",
            training_budget=147,
            annotated_training_examples=5,
            estimand="Update-required TRAIN procedure versus original procedure, averaging five seed-level F1 scores within the changed pool; replacement affects only one of three equally weighted pools.",
            primary_retained=True,
            no_new_recipe_or_cutoff_selection=True,
            uncertainty=dict(
                draws=10000,
                seed=20261020,
                clusters="Same sorted normalized-text clusters and full-endpoint macro-label support as original Part 2",
                adjusted_sensitivity="Re-express the original 24-contrast family alongside the originals at 1-.05/24 confidence; exactly one contrast can change. This remains supplementary, not a replacement primary result.",
                change_interval="One new-versus-original type change with descriptive conditional 95% interval; pool and three-pool budget changes differ exactly by a factor of three.",
                frozen_comparison="Report points and descriptive conditional 95% interval; no additional confirmatory claim.",
                scope="Conditional on these fixed fitted models, five seeds and fixed pools, not arbitrary training samples.",
            ),
            required_views=[
                "All five seeds, changed-pool mean/range/SD, original and supplementary three-pool means, unchanged BGE and frozen RoBERTa comparators.",
                "All 24 original comparison rows beside supplementary values; exact accounting of unchanged heads, pools and gates.",
                "Gated and ungated type F1, exact sets, micro F1, precision/recall, label counts, Hamming errors, and type-plus-hate exactness on the same 1135 eligible English TEST cases.",
                "Per-label TP/FP/FN and F1; observed five-example TRAIN support; label changes, fixes/new errors and seed consistency; no unsupported annotation is treated as negative.",
                "All-input logit/probability changes and gated/ungated label flips; gate suppression; source-specific primary contributions.",
                "Full four-head exactness only on English TEST gold-hate rows with all three fine annotations present; report if no such rows exist, no zero-score invention.",
                "Default cutoff with its original matching gate; the two pre-existing core type endpoints as separate descriptive sensitivities, without pooling overlapping endpoints.",
                "A deterministic changed-case inventory and representative fixes/harm/mixed or unchanged-exactness examples; no best-seed selection.",
            ],
            exclusions=[
                "No training or inference",
                "No Qwen calls or output edits",
                "No server operation",
                "No model-weight loading",
                "No new hyperparameter search",
                "No removal of original sparse pool or poor seed",
                "No changes to frozen primary tables",
                "No automatically treating repeated seed-text pairs as independent people/texts",
            ],
            preservation="Read the small verified supplementary archive and existing original prediction archives/caches; write only a new analysis directory and versioned logs.",
            verification="Meaningful synthetic tests plus independent score/count/resampling reconstruction, output/source bindings and publication checks.",
            result_not_prejudged=True,
        ),
    )
    print("Supplement analysis contract frozen; no evaluation scores computed.")


if __name__ == "__main__":
    with r.offline():
        main()

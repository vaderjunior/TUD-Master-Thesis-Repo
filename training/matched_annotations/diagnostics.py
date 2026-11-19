from training.matched_annotations import results as s
from pathlib import Path
import numpy as np


def main():
    r = s.r
    c = s.c
    score = c.read(r.OUT / "scores.json")
    cell = score["cells"]["broad/en_test/hate_type"]
    rec = c.read(r.BASE / "data/endpoints.json")["broad/en_test/hate_type"]
    ep = rec["endpoint"]
    _, kept, ids, y = s.older.gold("broad", "en_test/hate_type", ep)
    inputs = c.read(r.BASE / "data/evaluation_inputs.json")
    positions = {v["key"]: i for i, v in enumerate(inputs)}
    with np.load(r.BASE / "data/evaluation_features.npz", allow_pickle=False) as z:
        gate = z["gate"][[positions[k] for k in rec["keys"]]].astype(bool)
    perfect = y.copy()
    perfect[~gate] = False
    upper = s.summary(y, perfect, ep)[0]
    labels = {
        label: dict(
            reference_positive=int(y[:, j].sum()),
            suppressed_by_shared_gate=int((y[:, j] & ~gate).sum()),
        )
        for j, label in enumerate(ep["all_labels"])
    }
    effects = {}
    for group, pools in r.GROUPS.items():
        changes = []
        cutoffs = []
        rank = []
        for pool in pools:
            for seed in s.c.FINAL_SEEDS:

                def find(method):
                    return next(
                        v for v in cell["runs"][pool][method] if v["seed"] == seed
                    )

                base = find("fixed_unweighted_default")
                balanced = find("fixed_balanced_default")
                cut = find("fixed_unweighted_selected")
                changes.append(balanced["macro_f1"] - base["macro_f1"])
                cutoffs.append(cut["macro_f1"] - base["macro_f1"])
                a = next(
                    v
                    for v in cell["rankings"][pool]["fixed_unweighted_default"]
                    if v["seed"] == seed
                )
                b = next(
                    v
                    for v in cell["rankings"][pool]["fixed_balanced_default"]
                    if v["seed"] == seed
                )
                rank.append(
                    dict(
                        auc=b["macro_auc"] - a["macro_auc"],
                        ap=b["macro_ap"] - a["macro_ap"],
                    )
                )
        effects[group] = dict(
            balanced_loss_default_cutoff=dict(
                mean_difference=float(np.mean(changes)),
                min_difference=min(changes),
                positive_fits=sum(v > 0 for v in changes),
                fits=len(changes),
            ),
            cutoff_only_same_unweighted_weights=dict(
                mean_difference=float(np.mean(cutoffs)),
                min_difference=min(cutoffs),
                positive_fits=sum(v > 0 for v in cutoffs),
                fits=len(cutoffs),
            ),
            balanced_minus_unweighted_ranking=dict(
                mean_auc=float(np.mean([v["auc"] for v in rank])),
                mean_ap=float(np.mean([v["ap"] for v in rank])),
                auc_positive_fits=sum(v["auc"] > 0 for v in rank),
                ap_positive_fits=sum(v["ap"] > 0 for v in rank),
            ),
            label_mean_scores={
                label: {
                    method: {
                        metric: float(
                            np.mean(
                                [
                                    fit["per_label"][label][metric]
                                    for p in pools
                                    for fit in cell["runs"][p][method]
                                ]
                            )
                        )
                        for metric in ("precision", "recall", "f1")
                    }
                    for method in s.METHODS
                }
                for label in ep["all_labels"]
            },
        )
    slices = c.read(r.OUT / "slices.json")["cells"]
    bygroup = {}
    for group, pools in r.GROUPS.items():
        bygroup[group] = {}
        for subset in cell["slice_counts"]:
            methods = {}
            for method in s.PRIMARY:
                values = []
                for pool in pools:
                    fits = [
                        slices[f"broad/en_test/hate_type|{pool}|{method}|{seed}"][
                            subset
                        ]
                        for seed in (
                            s.c.FINAL_SEEDS if method == "tuned_selected" else [None]
                        )
                    ]
                    values.append(s.means(fits))
                methods[method] = s.means(values)
            bygroup[group][subset] = methods
    r.save(
        "supporting_explanations.json",
        dict(
            status="pass",
            descriptive_not_new_primary_tests=True,
            shared_gate=dict(
                negative_rows=int((~gate).sum()),
                all_rows=len(y),
                ideal_type_head_under_actual_gate=upper,
                label_suppression=labels,
            ),
            controlled_changes=effects,
            slice_group_means=bygroup,
            source_sha256=c.sha(Path(__file__)),
            limitation="Perfect type-head ceiling uses known evaluation labels only as a diagnostic; no oracle classifier is fitted or deployed. Type extras/misses are relative to recorded references, not adjudicated semantic truth.",
        ),
    )
    print(
        "Saved controlled-loss/cutoff, ranking, gate-ceiling and length explanations.",
        flush=True,
    )


if __name__ == "__main__":
    main()

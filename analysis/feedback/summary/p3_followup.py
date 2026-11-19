from fractions import Fraction
from statistics import mean
from pathlib import Path
import json
import socket
from unittest.mock import patch

import pandas as pd
from scipy.stats import mannwhitneyu

from analysis.feedback.summary import p3_audit as a

OUT = (
    a.ROOT
    / "Thesis_Log/astra_result/evidence/sq3_p3_historical_followup_20260928_v1.json"
)
DIR = a.STUDY / "followup_v1"


def main():
    a.require(
        not OUT.exists() and not (DIR / "plan.json").exists(),
        "Preserve earlier follow-up",
    )
    audit = a.read(a.EVIDENCE)
    hashes = dict(audit["input_hashes"])
    hashes.update(
        {a.rel(a.EVIDENCE): a.digest(a.EVIDENCE), a.rel(__file__): a.digest(__file__)}
    )
    a.verify(hashes)
    a.save(
        DIR / "plan.json",
        {
            "created_utc": a.utc(),
            "input_hashes": hashes,
            "reason": "The original taxonomy paired counts differ from all-available-repeat contains means; the original oracle retraction does not match reconstructed item counts.",
            "checks": [
                "Reconstruct taxonomy per-item contains means from the first two restored and both removed repeats; keep the all-repeat main audit unchanged.",
                "Record single-repeat target counts and all-repeat per-label changes/cardinality to qualify mechanical-contribution claims.",
                "Recompute feedback/control with the known prefix-duplicate holdout item omitted as a post-hoc sensitivity, not a replacement.",
                "Reconstruct the exact unpaired 3-versus-3 macro rank test as one-sided and two-sided, clearly separate from item-level uncertainty.",
                "Score earlier single-repeat rounds on original fixed/heldout sets and retain their trajectory interpretation.",
            ],
            "new_API_calls": 0,
            "new_TEST_scoring": False,
        },
    )
    df = pd.read_parquet(a.ROOT / "data/processed/en_dev_eval_sq3_types.parquet")
    rows = {str(r.id): r for r in df.itertuples(index=False)}
    gold = {i: set(r.hate_types) for i, r in rows.items()}
    selected = {}
    for group, keys in a.GROUPS.items():
        selected[group] = []
        for stem, arm in keys:
            run = {
                r["item_id"]: r
                for r in a.jsonl(a.ROOT / f"experiments/results/{stem}_live.jsonl")
                if r.get("arm") == arm
            }
            a.require(
                all(r["result"] is not None for r in run.values()), "Unexpected null"
            )
            selected[group].append(run)
    ids = list(rows)
    target = [i for i in ids if "grievance" in gold[i]]
    taxonomy = {}
    for name, left_name, right_name in [
        ("zero_shot", "restored_zero_shot", "removed_zero_shot"),
        ("rag", "baseline_rag", "removed_rag"),
    ]:
        left = selected[left_name]
        right = selected[right_name]
        early = a.sign_counts(
            a.contains_vector(ids, gold, left[:2]), a.contains_vector(ids, gold, right)
        )
        early_target = a.sign_counts(
            a.contains_vector(target, gold, left[:2]),
            a.contains_vector(target, gold, right),
        )
        labels = {
            label: {
                "removed_mean": mean(
                    m["per_label"][label]["f1"]
                    for m in audit["group_scores"][right_name]["all"]["runs"]
                ),
                "restored_mean": mean(
                    m["per_label"][label]["f1"]
                    for m in audit["group_scores"][left_name]["all"]["runs"]
                ),
            }
            for label in a.LABELS
        }
        for rec in labels.values():
            rec["difference"] = rec["restored_mean"] - rec["removed_mean"]
        contribution = labels["grievance"]["difference"] / 7
        residual = (
            sum(labels[l]["difference"] for l in a.LABELS if l != "grievance") / 7
        )
        a.require(
            abs(
                contribution
                + residual
                - audit["comparisons"]["taxonomy_" + name]["macro_difference"]
            )
            < 1e-13,
            "Contribution arithmetic differs",
        )
        metrics = {
            state: [
                {
                    "contains_correct": sum(
                        gold[i] <= set(r[i]["result"]["hate_type"]) for i in target
                    ),
                    "predicted_grievance": sum(
                        "grievance" in r[i]["result"]["hate_type"] for i in target
                    ),
                    "mean_predicted_label_count": mean(
                        len(r[i]["result"]["hate_type"]) for i in target
                    ),
                    "predicted_label_counts": {
                        l: sum(l in r[i]["result"]["hate_type"] for i in target)
                        for l in a.LABELS
                    },
                }
                for r in runs
            ]
            for state, runs in [("removed", right), ("restored", left)]
        }
        taxonomy[name] = {
            "first_two_restored_vs_both_removed": early,
            "first_two_target": early_target,
            "all_repeat_per_label": labels,
            "new_label_contribution": contribution,
            "other_six_contribution": residual,
            "target_per_repeat": metrics,
            "interpretation": "A reproducible historical two-versus-two snapshot, not an all-available-replicate test and not the mean-macro-F1 endpoint. The original originating analysis was not recovered.",
        }
    a.require(
        tuple(
            taxonomy["zero_shot"]["first_two_restored_vs_both_removed"][k]
            for k in ("better", "worse", "tied")
        )
        == (77, 40, 350),
        "ZS old table not reproduced",
    )
    a.require(
        tuple(
            taxonomy["rag"]["first_two_restored_vs_both_removed"][k]
            for k in ("better", "worse", "tied")
        )
        == (71, 52, 344),
        "RAG old table not reproduced",
    )
    a.require(
        taxonomy["zero_shot"]["first_two_target"]["better"] == 33
        and taxonomy["rag"]["first_two_target"]["better"] == 43,
        "Target old table not reproduced",
    )
    overlaps = sorted(
        {r["fixed_item"] for r in audit["known_prefix_normalized_overlaps"]}
    )
    sens = {}
    for stratum in ("fixed", "held_out"):
        old_ids = audit["strata"][stratum]["ids"]
        keep = [i for i in old_ids if i not in overlaps]
        left = selected["feedback"]
        right = selected["control"]
        lm = [a.metric(keep, gold, r) for r in left]
        rm = [a.metric(keep, gold, r) for r in right]
        sens[stratum] = {
            "removed_ids": sorted(set(old_ids) - set(keep)),
            "n": len(keep),
            "left_macro": [m["macro_f1"] for m in lm],
            "right_macro": [m["macro_f1"] for m in rm],
            "macro_difference": mean(m["macro_f1"] for m in lm)
            - mean(m["macro_f1"] for m in rm),
            "macro_bootstrap": a.paired_bootstrap(keep, gold, left, right),
            "contains_sign": a.sign_counts(
                a.contains_vector(keep, gold, left),
                a.contains_vector(keep, gold, right),
            ),
        }
    trajectory = []
    for stem, arm in a.EXTRA:
        run = {
            r["item_id"]: r
            for r in a.jsonl(a.ROOT / f"experiments/results/{stem}_live.jsonl")
            if r.get("arm") == arm
        }
        trajectory.append(
            {
                "stem": stem,
                "arm": arm,
                "scores": {
                    s: a.metric(audit["strata"][s]["ids"], gold, run)
                    for s in ("fixed", "held_out")
                },
            }
        )
    av = audit["group_scores"]["oracle_A"]["held_out"]["macro"]["values"]
    bv = audit["group_scores"]["oracle_B"]["held_out"]["macro"]["values"]
    ranks = {
        alt: float(mannwhitneyu(av, bv, alternative=alt, method="exact").pvalue)
        for alt in ("greater", "two-sided")
    }
    result = {
        "status": "historical_paired_provenance_reconciled",
        "created_utc": a.utc(),
        "input_hashes": hashes,
        "plan": a.rel(DIR / "plan.json"),
        "plan_sha256": a.digest(DIR / "plan.json"),
        "taxonomy": taxonomy,
        "known_overlap_sensitivity": sens,
        "earlier_rounds": trajectory,
        "oracle_unpaired_decoder_macro_rank_p": ranks,
        "rank_test_note": "Three decoder repeats per arm on the same fixed items; does not replace paired item uncertainty. The original p=0.05 is one-sided; two-sided is 0.10.",
        "new_API_calls": 0,
        "new_retrieval": False,
        "new_training": False,
        "new_TEST_scoring": False,
    }
    a.verify(hashes)
    a.save(OUT, result)
    a.save(
        DIR / "complete.json",
        {"evidence": a.rel(OUT), "evidence_sha256": a.digest(OUT)},
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "taxonomy": {
                    n: {
                        k: r[k]
                        for k in (
                            "first_two_restored_vs_both_removed",
                            "first_two_target",
                            "new_label_contribution",
                            "other_six_contribution",
                        )
                    }
                    for n, r in taxonomy.items()
                },
                "sensitivity": {
                    s: {k: r[k] for k in ("n", "macro_difference", "macro_bootstrap")}
                    for s, r in sens.items()
                },
                "rank_p": ranks,
                "evidence_sha256": a.digest(OUT),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", a.forbid), patch.object(
        socket, "create_connection", a.forbid
    ):
        main()

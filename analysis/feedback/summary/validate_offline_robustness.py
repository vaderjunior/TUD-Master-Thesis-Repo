from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score
from analysis.common import replay as c

BASE = c.BASE / "sq3_robustness"
OLD = c.ROOT / "experiments/astra/sq3_combined_results_20261004_v1"
SEEDS = [202609281, 202609282, 202609283]
LABELS = [
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
]


def main():
    c.verify(c.read(BASE / "plan.json")["inputs"])
    c.verify(c.read(BASE / "complete.json")["outputs"])
    wanted = {
        "taxonomy": {"rag_full", *[f"rag_schema_{l}" for l in LABELS]},
        "explanations": {"reference", *[f"plain_always_s{s}" for s in SEEDS]},
    }
    saved = {}
    with (OLD / "review/predictions.jsonl").open(encoding="utf8") as f:
        for line in f:
            r = json.loads(line)
            if r["condition"] in wanted[r["study"]]:
                saved[r["study"], r["condition"], r["repeat"], r["id"]] = r["result"]
    tax = c.read(BASE / "taxonomy_influence.json")
    old = c.read(OLD / "taxonomy/primary.json")
    ids = old["common_valid_ids"]
    roster = {
        r["id"]: r
        for r in c.rows(
            c.ROOT / "experiments/astra/sq3_full_kb_taxonomy_20261001_v1/roster.jsonl"
        )
    }
    clusters = sorted({roster[i]["cluster"] for i in ids})
    ci = {v: i for i, v in enumerate(clusters)}
    totals = np.zeros((len(clusters), 2, 7, 3, 3), int)
    for rid in ids:
        gold = set(roster[rid]["gold"]["hate_type"])
        for k, l in enumerate(LABELS):
            for mode, cond in enumerate(["rag_full", "rag_schema_" + l]):
                for rep in range(3):
                    labels = set(saved["taxonomy", cond, rep, rid]["hate_type"])
                    y = l in gold
                    x = l in labels
                    totals[ci[roster[rid]["cluster"]], mode, k, rep] += [
                        int(y and x),
                        int(x and (not y)),
                        int(y and (not x)),
                    ]

    def metric(t):
        tp = t[..., 0]
        den = 2 * tp + t[..., 1] + t[..., 2]
        return np.divide(
            2 * tp, den, out=np.zeros_like(den, dtype=float), where=den != 0
        )

    point = metric(totals.sum(axis=0))
    delta = point[0] - point[1]
    assert abs(delta.mean() - tax["primary_reproduced"]) < 1e-12
    rng = np.random.default_rng(2026100103)
    draws = []
    for start in range(0, 10000, 100):
        chosen = rng.integers(0, len(clusters), size=(100, len(clusters)))
        weights = np.array([np.bincount(x, minlength=len(clusters)) for x in chosen])
        samples = (weights @ totals.reshape(len(clusters), -1)).reshape(100, 2, 7, 3, 3)
        scores = metric(samples)
        draws.extend(scores[:, 0] - scores[:, 1])
    draws = np.asarray(draws)
    for k, l in enumerate(LABELS):
        keep = np.arange(7) != k
        x = tax["omit_category"][l]
        assert abs(delta[keep].mean() - x["difference"]) < 1e-12
        assert np.allclose(
            np.quantile(draws[:, keep].mean(axis=(1, 2)), [0.025, 0.975]),
            x["ci95_descriptive"],
            rtol=0,
            atol=1e-12,
        )
    for rep in range(3):
        keep = np.arange(3) != rep
        x = tax["omit_repeat"][str(rep)]
        assert abs(delta[:, keep].mean() - x["difference"]) < 1e-12
        assert np.allclose(
            np.quantile(draws[:, :, keep].mean(axis=(1, 2)), [0.025, 0.975]),
            x["ci95_descriptive"],
            rtol=0,
            atol=1e-12,
        )
    rr = c.rows(BASE / "correction_hateful_pairs.jsonl")
    br = c.rows(BASE / "correction_benign_pairs.jsonl")
    summary = c.read(BASE / "correction_compatibility_influence.json")
    for r in rr:
        gold = set(r["query_types"])
        source = set(r["source_types"])
        seed = r["pool_seed"]
        rid = r["query_id"]
        assert r["all_source_types_compatible"] == (source <= gold) and r[
            "any_shared_type"
        ] == bool(source & gold)
        for rep in range(3):
            a = saved["explanations", "reference", rep, rid]
            b = saved["explanations", f"plain_always_s{seed}", rep, rid]
            aa, bb = (set(a["hate_type"]), set(b["hate_type"]))
            assert (
                len(bb & gold) - len(aa & gold)
                == r["correct_label_change_by_repeat"][rep]
            )
            assert (
                len(bb - gold) - len(aa - gold)
                == r["extra_label_change_by_repeat"][rep]
            )
            assert (
                int(not b["hate"]) - int(not a["hate"])
                == r["missed_hate_change_by_repeat"][rep]
            )
    for r in br:
        for rep in range(3):
            a = saved["explanations", "reference", rep, r["query_id"]]
            b = saved[
                "explanations", f"plain_always_s{r['pool_seed']}", rep, r["query_id"]
            ]
            assert (
                int(b["hate"]) - int(a["hate"])
                == r["false_positive_change_by_repeat"][rep]
            )
    fitchecks = 0
    for omit in summary["leave_source_associated_queries_out"]:
        differences = []
        for expected in omit["pools"]:
            seed = expected["pool_seed"]
            keep = [
                r
                for r in rr
                if r["pool_seed"] == seed and r["source_id"] != omit["source_id"]
            ]
            assert len(keep) == expected["retained"]
            y = np.array([[l in r["query_types"] for l in LABELS] for r in keep], int)
            means = []
            for mode, cond in [
                ("reference", "reference"),
                ("plain", f"plain_always_s{seed}"),
            ]:
                scores = []
                for rep in range(3):
                    x = np.array(
                        [
                            [
                                l
                                in saved["explanations", cond, rep, r["query_id"]][
                                    "hate_type"
                                ]
                                for l in LABELS
                            ]
                            for r in keep
                        ],
                        int,
                    )
                    value = f1_score(y, x, average="macro", zero_division=0)
                    scores.append(value)
                    assert abs(value - expected[mode + "_by_repeat"][rep]) < 1e-12
                    fitchecks += 1
                means.append(np.mean(scores))
            differences.append(means[1] - means[0])
            assert abs(differences[-1] - expected["difference"]) < 1e-12
        assert abs(np.mean(differences) - omit["difference"]) < 1e-12
    for s in summary["per_source_pool"]:
        subset = [
            r
            for r in rr
            if (r["pool_seed"], r["source_id"]) == (s["pool_seed"], s["source_id"])
        ]
        assert len(subset) == s["query_pool_assignments"]
        assert (
            abs(
                sum((sum(r["correct_label_change_by_repeat"]) / 3 for r in subset))
                - s["sum_correct_change_mean_over_repeats"]
            )
            < 1e-12
        )
        assert (
            abs(
                sum((sum(r["extra_label_change_by_repeat"]) / 3 for r in subset))
                - s["sum_extra_change_mean_over_repeats"]
            )
            < 1e-12
        )
    c.save(
        BASE / "verification.json",
        {
            "status": "independent_diagnostics_pass",
            "created_utc": c.utc(),
            "checker_sha256": c.sha(Path(__file__)),
            "taxonomy_all_10000_bootstrap_draws_reconstructed": True,
            "all_seven_category_and_three_repeat_omissions_checked": True,
            "source_omission_sklearn_run_scores_checked": fitchecks,
            "hateful_pair_transition_checks": len(rr) * 3,
            "benign_pair_gate_checks": len(br) * 3,
            "all_120_source_pool_records_reconciled": True,
            "new_model_calls": 0,
        },
    )
    print(
        "Independent SQ3 checks passed: 10000 bootstrap draws;",
        fitchecks,
        "source-omission run scores;",
        len(rr) * 3,
        "hateful and",
        len(br) * 3,
        "benign paired repeat checks.",
    )


if __name__ == "__main__":
    with c.network_guard():
        main()

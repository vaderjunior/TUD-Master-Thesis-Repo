from analysis.micro_f1.combined import prepare as p
import numpy as np
from collections import Counter

c = p.c


def equal(a, b, message):
    c.require(np.allclose(a, b, rtol=0, atol=1e-12), message)


def main():
    rows = c.read(p.BASE / "comparison_ledger.json")
    cohorts = c.read(p.BASE / "cohorts.json")
    opposite = [
        r
        for r in rows
        if r["matched95_interpretation"] == "supported_opposite_directions"
    ]
    c.require(
        len(opposite) == 1, "Review unexpected additional opposite supported comparison"
    )
    r = opposite[0]
    source = c.read(c.ROOT / r["source"])
    groups = {}
    checks = 0
    for method in ["fixed_balanced", "tuned"]:
        fits = [
            f
            for f in source["fit_rows"]
            if f["model"]["budget"] == 147
            and f["model"]["method"] == method
            and f["model"]["variant"] == "default"
        ]
        c.require(
            len(fits) == 15 and len({f["model"]["pool"] for f in fits}) == 3,
            "Expected three pools and five seeds",
        )
        c.require(
            set(Counter(f["model"]["pool"] for f in fits).values()) == {5},
            "Unbalanced fit hierarchy",
        )
        for f in fits:
            z = np.array(f["tp_fp_fn"])
            tp, fp, fn = z.T
            den = 2 * tp + fp + fn
            f1 = np.divide(2 * tp, den, out=np.zeros(len(tp), float), where=den != 0)
            weights = den / den.sum()
            equal(
                f1, f["per_label_f1"], "Stored label score differs from confusion count"
            )
            equal(
                weights,
                f["micro_weights"],
                "Stored micro weight differs from confusion count",
            )
            equal(f1.mean(), f["scores"]["macro"], "Macro reconstruction")
            equal((weights * f1).sum(), f["scores"]["micro"], "Micro reconstruction")
            equal((tp + fn).sum() / source["n"], f["mean_gold"], "Gold cardinality")
            equal(
                (tp + fp).sum() / source["n"],
                f["mean_predicted"],
                "Prediction cardinality",
            )
            checks += 6
        mean = lambda field: np.mean(
            np.array([f[field] for f in fits]), axis=0
        ).tolist()
        scores = {
            m: float(np.mean([f["scores"][m] for f in fits]))
            for m in ["macro", "micro"]
        }
        original = next(
            x
            for x in source["absolute"]
            if x["budget"] == 147 and x["method"] == method + "/default"
        )
        for m in scores:
            equal(
                scores[m],
                original["metrics"][m]["value"],
                "Fit mean changed original hierarchy",
            )
        groups[method] = dict(
            n_fits=len(fits),
            n_pools=3,
            scores=scores,
            mean_tp_fp_fn=mean("tp_fp_fn"),
            mean_label_f1=mean("per_label_f1"),
            mean_micro_weights=mean("micro_weights"),
            mean_micro_contribution=np.mean(
                [np.array(f["micro_weights"]) * f["per_label_f1"] for f in fits], axis=0
            ).tolist(),
            **{
                k: mean(k)
                for k in [
                    "precision",
                    "recall",
                    "set_exact",
                    "mean_predicted",
                    "mean_gold",
                    "complete_field",
                    "gate_miss",
                ]
            },
        )
    for m in ["macro", "micro"]:
        equal(
            groups["tuned"]["scores"][m] - groups["fixed_balanced"]["scores"][m],
            r["metrics"][m]["value"],
            "Opposite contrast reconstruction",
        )
    labels = []
    for i, label in enumerate(source["labels"]):
        a = groups["fixed_balanced"]
        b = groups["tuned"]
        labels.append(
            dict(
                label=label,
                gold_support=a["mean_tp_fp_fn"][i][0] + a["mean_tp_fp_fn"][i][2],
                fixed_label_f1=a["mean_label_f1"][i],
                tuned_label_f1=b["mean_label_f1"][i],
                delta_label_f1=b["mean_label_f1"][i] - a["mean_label_f1"][i],
                fixed_weight=a["mean_micro_weights"][i],
                tuned_weight=b["mean_micro_weights"][i],
                fixed_tp_fp_fn=a["mean_tp_fp_fn"][i],
                tuned_tp_fp_fn=b["mean_tp_fp_fn"][i],
                delta_micro_contribution=b["mean_micro_contribution"][i]
                - a["mean_micro_contribution"][i],
            )
        )
    equal(
        sum(l["delta_micro_contribution"] for l in labels),
        r["metrics"]["micro"]["value"],
        "Contribution identity",
    )
    equal(
        np.mean([l["delta_label_f1"] for l in labels]),
        r["metrics"]["macro"]["value"],
        "Equal-label identity",
    )
    control = dict(
        row_id=r["id"],
        n=source["n"],
        status="verified saved-count explanation",
        metrics=r["metrics"],
        groups=groups,
        labels=labels,
        independent_identities=checks + 8,
        limitation="A supporting default-cutoff comparison among 66 controls, on larger DEV; not a prelisted main micro question or a new selection rule. Contributions are arithmetic identities, not causal effects. Mean fit F1 is not F1 of pooled fit counts.",
    )

    def one(**attributes):
        found = [r for r in rows if all(r[k] == v for k, v in attributes.items())]
        c.require(len(found) == 1, "Ambiguous overlap row: " + str(attributes))
        return found[0]

    core = one(
        part=2,
        study="core",
        endpoint="en_dev_eval_sq3_types",
        contrast="rag_full minus rag_small",
    )
    pools = one(
        part=5,
        study="pools",
        endpoint="en_dev_eval_sq3_types",
        contrast="curated_small_minus_uniform_compact",
    )
    matched = one(
        part=4,
        study="matched",
        endpoint="en_dev_eval_sq3_types/hate_type",
        contrast="coverage_matched_",
    )
    reserved = one(
        part=7,
        study="reserved",
        view="reserved/hate_type/full/supporting/reserved5__error_balanced-reference5/types",
        contrast="treatment-control",
    )
    pairs = [
        ("Earlier core and SQ1 pool study", core, pools),
        ("SQ1 pools and matched information", pools, matched),
        ("Matched information and reserved feedback", matched, reserved),
    ]
    for population, dimension in [
        ("en_test", "hate_type"),
        ("en_test", "target_group"),
        ("en_hatecheck_eval", "target_group"),
    ]:
        broad = one(
            part=2,
            study="broad",
            endpoint=population + "__" + dimension,
            contrast="rag_full minus zero_shot",
        )
        transfer = one(
            part=7,
            study="transfer",
            view=f"transfer/population/{population}/{dimension}/main",
            contrast="feedback-reference",
        )
        rounds = one(
            part=7,
            study="rounds",
            view=f"rounds/population/{population}|{dimension}|final",
            contrast="r4_adaptive-reference",
        )
        pairs.extend(
            [
                (
                    population
                    + "/"
                    + dimension
                    + ": broader retrieval and feedback transfer",
                    broad,
                    transfer,
                ),
                (
                    population
                    + "/"
                    + dimension
                    + ": broader retrieval and final four-round study",
                    broad,
                    rounds,
                ),
                (
                    population
                    + "/"
                    + dimension
                    + ": feedback transfer and final four-round study",
                    transfer,
                    rounds,
                ),
            ]
        )
    overlaps = []
    for title, left, right in pairs:

        def normalized(row):
            ids = cohorts[row["cohort_sha256"]]["ids"]
            out = []
            for key in ids:
                if key.startswith("en/"):
                    c.require(key.count("/") == 1, "Unexpected ID namespace")
                    out.append(key)
                else:
                    c.require(
                        row["part"] in [2, 5] and key.startswith("implicit_hate-"),
                        "Unverified bare-ID namespace",
                    )
                    out.append("en/" + key)
            c.require(len(out) == len(set(out)), "Normalization merges IDs")
            return set(out)

        a = normalized(left)
        b = normalized(right)
        shared = a & b
        overlaps.append(
            dict(
                description=title,
                left_row=left["id"],
                right_row=right["id"],
                n_left=len(a),
                n_right=len(b),
                shared=len(shared),
                left_only=len(a - b),
                right_only=len(b - a),
                same_id_set=a == b,
                namespace="Existing en/<dataset>-<record> keys; add en/ only for earlier implicit_hate IDs. No text deduplication, prediction pooling or inference.",
            )
        )
    p.save(
        "targeted_diagnostics.json",
        dict(
            status="passed",
            cutoff_control=control,
            cohort_overlaps=overlaps,
            source_sha256=c.BINDINGS,
            interpretation="These are overlapping query populations and different interventions/runs; neither identical evidence nor independent replications. The ledger also excludes Part 5 copies of 23 Part 2 core contrasts.",
        ),
    )
    print("Supporting cutoff control:")
    for name, g in groups.items():
        print(
            name,
            {k: round(v * 100, 3) for k, v in g["scores"].items()},
            "precision/recall",
            round(g["precision"] * 100, 3),
            round(g["recall"] * 100, 3),
            "labels",
            round(g["mean_predicted"], 4),
        )
    for x in labels:
        print(
            x["label"],
            "gold",
            x["gold_support"],
            "F1 change pp",
            round(x["delta_label_f1"] * 100, 3),
            "micro contribution change pp",
            round(x["delta_micro_contribution"] * 100, 3),
        )
    for x in overlaps:
        print(x["description"], x["n_left"], x["n_right"], "shared", x["shared"])


if __name__ == "__main__":
    main()

from analysis.classification.micro_f1 import supervised as r
from itertools import product
import numpy as np

b = r.b
c = b.c


def relation(x):
    return "positive" if x[0] > 0 else "negative" if x[1] < 0 else "unresolved"


def load(study):
    return [c.read(p) for p in r.complete_for(study)["outputs"]]


def main():
    out = b.BASE / "diagnostics"
    out.mkdir(exist_ok=True)
    primary = []
    nominal = []
    rank = []
    labelrows = []
    variation = []
    matched_confidence = []
    for d in load("gpu"):
        base = (
            b.ROOT
            / next(
                p
                for p in r.complete_for("gpu")["outputs"]
                if c.read(p)["endpoint"] == d["endpoint"]
            )
        ).parent
        dr = b.arrays(base / "draws.npz")["scores"]
        meta = c.read(base / "metadata.json")
        _, bd = b.aggregate(meta["models"], dr)
        for z in d["contrasts"]:
            size, left, right = z["budget"], z["left"], z["right"]
            m = z["metrics"]
            key = f"gpu/{d['endpoint']}|{size}|" + (
                "tuned_selected_minus_BGE_fold42"
                if left == "tuned_selected"
                else "frozen_RoBERTa_fold42_minus_BGE_fold42"
            )
            primary.append(
                dict(
                    question_id=key,
                    study="gpu",
                    endpoint=d["endpoint"],
                    budget=size,
                    left=left,
                    right=right,
                    **{k: m[k] for k in ("macro", "micro")},
                )
            )
            dd = bd[size, left] - bd[size, right]
            matched_confidence.append(
                dict(
                    id=key,
                    macro_at_micro_confidence=b.ci(dd[:, 0], 1 - 0.05 / 12),
                    micro_at_original_macro_confidence=b.ci(dd[:, 1], 1 - 0.05 / 24),
                )
            )
            pairrows = {
                method: [
                    x
                    for x in d["fit_rows"]
                    if x["model"]["budget"] == size and x["model"]["method"] == method
                ]
                for method in (left, right)
            }

            def means(method, key):
                rr = pairrows[method]
                pools = sorted({x["model"]["pool"] for x in rr})
                return np.mean(
                    [
                        np.mean(
                            [np.asarray(x[key]) for x in rr if x["model"]["pool"] == p],
                            0,
                        )
                        for p in pools
                    ],
                    0,
                )

            def contribution(method):
                rr = pairrows[method]
                pools = sorted({x["model"]["pool"] for x in rr})
                return np.mean(
                    [
                        np.mean(
                            [
                                np.asarray(x["per_label_f1"]) * x["micro_weights"]
                                for x in rr
                                if x["model"]["pool"] == p
                            ],
                            0,
                        )
                        for p in pools
                    ],
                    0,
                )

            lf, rf = means(left, "per_label_f1"), means(right, "per_label_f1")
            lc, rc = means(left, "tp_fp_fn"), means(right, "tp_fp_fn")
            lw, rw = means(left, "micro_weights"), means(right, "micro_weights")
            con = contribution(left) - contribution(right)
            b.equal(
                con.sum(),
                m["micro"]["value"],
                "label contribution sums to micro difference",
            )
            for j, label in enumerate(d["labels"]):
                labelrows.append(
                    dict(
                        endpoint=d["endpoint"],
                        budget=size,
                        left=left,
                        right=right,
                        label=label,
                        gold=int(lc[j, 0] + lc[j, 2]),
                        left_f1=lf[j],
                        right_f1=rf[j],
                        macro_difference_contribution=(lf[j] - rf[j])
                        / len(d["labels"]),
                        micro_difference_contribution=con[j],
                        left_weight=lw[j],
                        right_weight=rw[j],
                        left_counts=lc[j],
                        right_counts=rc[j],
                    )
                )
            pp = [
                x
                for x in d["seed_comparisons"]
                if x["budget"] == size and x["method"] == left
            ]
            values = np.array(
                [
                    [x["difference_to_matched_pool_BGE"][k] for k in b.METRICS]
                    for x in pp
                ]
            )
            groups = sorted({x["pool"] for x in pp})
            combo = np.array(
                [
                    np.mean(v, 0)
                    for v in product(
                        *[
                            [
                                [
                                    x["difference_to_matched_pool_BGE"][k]
                                    for k in b.METRICS
                                ]
                                for x in pp
                                if x["pool"] == p
                            ]
                            for p in groups
                        ]
                    )
                ]
            )
            variation.append(
                dict(
                    endpoint=d["endpoint"],
                    budget=size,
                    method=left,
                    fits=len(values),
                    fit_positive={
                        k: int((values[:, j] > 0).sum())
                        for j, k in enumerate(b.METRICS)
                    },
                    fit_range={
                        k: [float(values[:, j].min()), float(values[:, j].max())]
                        for j, k in enumerate(b.METRICS)
                    },
                    combinations=len(combo),
                    combination_positive={
                        k: int((combo[:, j] > 0).sum()) for j, k in enumerate(b.METRICS)
                    },
                    combination_range={
                        k: [float(combo[:, j].min()), float(combo[:, j].max())]
                        for j, k in enumerate(b.METRICS)
                    },
                    omissions=[
                        x
                        for x in d["omissions"]
                        if x["budget"] == size and x["left"] == left
                    ],
                )
            )
    for s in ("compact_tuning", "shorter", "matched_type", "native_cpu"):
        for d in load(s):
            for z in d["contrasts"]:
                if not z.get("question_id"):
                    continue
                m = z["variations"]["unstratified"]["metrics"]
                primary.append(
                    dict(
                        question_id=z["question_id"],
                        study=s,
                        endpoint=z["endpoint"],
                        group=z.get("group"),
                        left=z["left"],
                        right=z["right"],
                        **{k: m[k] for k in ("macro", "micro")},
                    )
                )
                if s == "shorter":
                    dd = b.arrays(b.ROOT / z["variations"]["unstratified"]["draws"])[
                        "difference"
                    ]
                    matched_confidence.append(
                        dict(
                            id=z["question_id"],
                            macro_at_micro_confidence=b.ci(dd[:, 0], 0.975),
                            micro_at_original_macro_confidence=b.ci(dd[:, 1], 0.9875),
                        )
                    )
    expected = {
        x["id"]
        for x in b.MANIFEST["rows"]
        if x["family"]
        in ("gpu", "compact_tuning", "shorter", "matched_type", "native_cpu")
    }
    actual = {x["question_id"] for x in primary}
    c.require(
        expected == actual, "22-question inventory differs: " + str(expected ^ actual)
    )
    for z in primary:
        z["nominal_macro_relation"] = relation(z["macro"]["ci95"])
        z["nominal_micro_relation"] = relation(z["micro"]["ci95"])
        z["micro_family_relation"] = relation(z["micro"]["adjusted_interval"])
        z["original_macro_family_relation"] = relation(
            z["macro"]["original_adjusted_interval"]
        )
        z["direction_reversed"] = z["macro"]["value"] * z["micro"]["value"] < 0
        if z["nominal_macro_relation"] != z["nominal_micro_relation"]:
            nominal.append(z)
        if z["direction_reversed"]:
            rank.append(z)
    controls = []
    for d in load("controls"):
        for z in d["contrasts"]:
            m = z["metrics"]
            entry = dict(
                endpoint=d["endpoint"],
                budget=z["budget"],
                left=z["left"],
                right=z["right"],
                macro=m["macro"],
                micro=m["micro"],
                nominal_macro_relation=relation(m["macro"]["ci95"]),
                nominal_micro_relation=relation(m["micro"]["ci95"]),
                direction_reversed=m["macro"]["value"] * m["micro"]["value"] < 0,
            )
            controls.append(entry)
    f = load("five")[0]
    source = (b.ROOT / r.units("five_fit_update")[0]["source"]).parent
    old = c.read(source / "endpoint_diagnostics.json")
    diag = []
    raw = b.arrays(source / "diagnostic_primary_arrays.npz")
    g = b.arrays(source / "primary_paired.npz")["truth"]
    for endpoint, policies in old.items():
        for policy, v in policies.items():
            entries = {}
            for side in ("old", "new"):
                x = v[side]
                z = np.stack([x["counts"][k] for k in ("tp", "fp", "fn")], -1)
                scores = b.score(z, list(range(len(v["labels"]))))
                b.equal(
                    scores[:, 0], x["macro_f1"], "five-fit diagnostic original macro"
                )
                b.equal(
                    scores[:, 1],
                    x["micro_f1"],
                    "five-fit diagnostic existing micro independently verified",
                )
                if endpoint.startswith("broad/"):
                    pp = raw[policy.replace("/", "__") + "__" + side]
                    b.equal(
                        b.counts(g, pp).sum(1),
                        z,
                        "five-fit diagnostic raw/count agreement",
                    )
                entries[side] = dict(
                    scores=dict(zip(b.METRICS, scores.mean(0))),
                    per_seed_scores=scores,
                    counts=z,
                    exact_set=float(np.mean(x["exact_set"])),
                    complete_field=float(np.mean(x["hate_and_type_exact"])),
                    predicted_labels=float(np.mean(x["predicted_labels"])),
                )
            diag.append(
                dict(
                    endpoint=endpoint,
                    policy=policy,
                    n=v["rows"],
                    old=entries["old"],
                    new=entries["new"],
                    changes=v["changes"],
                )
            )
    quality = c.read(source / "quality_change_intervals.json")
    matrix = []
    for z in primary:
        if z["study"] != "gpu":
            continue
        update = next(
            (
                x
                for x in f["contrasts"]
                if x["affected"]
                and z["endpoint"] == "en_test/hate_type"
                and z["budget"] == x["budget"]
                and z["left"] == x["left"]
            ),
            None,
        )
        new = update["supplementary"]["micro"] if update else z["micro"]
        c.require(
            relation(new["adjusted_interval"]) == z["micro_family_relation"],
            "Micro five-fit family interpretation changed",
        )
        matrix.append(
            dict(
                question_id=z["question_id"],
                affected=update is not None,
                old=z["micro"],
                new=new,
                relation=relation(new["adjusted_interval"]),
            )
        )
    b.save(
        out / "results.json",
        dict(
            primary=primary,
            nominal_changes=nominal,
            direction_reversals=rank,
            label_contributions=labelrows,
            variation=variation,
            matched_confidence=matched_confidence,
            controls=controls,
            five_diagnostics=diag,
            five_quality=quality,
            five_micro_family_matrix=matrix,
            interpretation="Per-fit F1 then original equal-pool averaging. Label contributions decompose realized scores, not causal label effects. Fixed-seed combinations and omissions are descriptive. All five-fit diagnostics are separate from the primary original fits.",
        ),
    )
    b.save(
        out / "complete.json",
        dict(
            status="passed",
            sources=c.BINDINGS,
            checks=dict(b.CHECKS),
            errors=dict(b.ERRORS),
        ),
    )
    print(
        "DIAGNOSTICS COMPLETE; 22 questions,",
        len(nominal),
        "matched-95 interval changes",
    )


if __name__ == "__main__":
    main()

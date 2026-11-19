from analysis.classification import reporting_inputs as p
from collections import defaultdict
import numpy as np
from sklearn.metrics import (
    precision_recall_fscore_support,
    jaccard_score,
    mean_absolute_error,
)

c = p.c
METHODS = {
    "qwen/rag_full": "Qwen: full RAG",
    "qwen/rag_small": "Qwen: compact RAG",
    "qwen/zero_shot": "Qwen: no retrieval",
    "gpu/5000/bge/selected": "BGE: 5,000 texts",
    "gpu/5000/tuned/selected": "Tuned RoBERTa: 5,000 texts",
    "cpu/full_bge_matched/selected": "BGE: full TRAIN",
    "cpu/full_tfidf_matched/selected": "TF-IDF: full TRAIN",
    "cpu/full_neural/selected": "RoBERTa: full TRAIN",
}
LEGAL = {
    **{k: v for k, v in METHODS.items() if k.startswith("qwen/")},
    "historical_gbert/default": "Historical German GBERT",
}
ERRORS = [
    "correct",
    "under_one",
    "under_two",
    "over_one",
    "over_two",
    "gate_closed",
    "gate_open_missing",
]
CHECKS = []


def equal(label, actual, expected):
    actual = np.asarray(actual, dtype=float)
    expected = np.asarray(expected, dtype=float)
    c.require(
        actual.shape == expected.shape
        and np.allclose(actual, expected, rtol=0, atol=2e-12, equal_nan=False),
        "Numerical check failed: " + label,
    )
    CHECKS.append(
        dict(
            label=label,
            values=int(actual.size),
            max_difference=(
                float(np.max(np.abs(actual - expected))) if actual.size else 0.0
            ),
        )
    )


def scalar_divide(n, d):
    return np.divide(
        n, d, out=np.zeros_like(np.asarray(n), dtype=float), where=np.asarray(d) != 0
    )


def label_metrics(y, x):
    tp = np.count_nonzero(y & x, axis=0)
    fp = np.count_nonzero(~y & x, axis=0)
    fn = np.count_nonzero(y & ~x, axis=0)
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        support=y.sum(0),
        precision=scalar_divide(tp, tp + fp),
        recall=scalar_divide(tp, tp + fn),
        f1=scalar_divide(2 * tp, 2 * tp + fp + fn),
    )


def overlap(y, x, gate, legal=False):
    intersection = np.count_nonzero(y & x, axis=1)
    union = np.count_nonzero(y | x, axis=1)
    j = np.divide(
        intersection, union, out=np.ones(len(y), dtype=float), where=union != 0
    )
    exact = np.all(y == x, axis=1)
    return dict(
        sample_jaccard=float(j.mean()),
        gate_aware_exact=float((exact if legal else exact & gate).mean()),
    )


def ordinal(y, x, gate):
    c.require(np.all(y.sum(1) == 1), "Severity gold is not exactly one class")
    c.require(
        np.all(x.sum(1) <= 1) and not np.any(x[~gate]), "Invalid delivered severity"
    )
    gold = y.argmax(1)
    emitted = x.sum(1) == 1
    guess = x.argmax(1)
    delta = guess.astype(int) - gold.astype(int)
    counts = dict(
        correct=int(np.count_nonzero(emitted & (delta == 0))),
        under_one=int(np.count_nonzero(emitted & (delta == -1))),
        under_two=int(np.count_nonzero(emitted & (delta == -2))),
        over_one=int(np.count_nonzero(emitted & (delta == 1))),
        over_two=int(np.count_nonzero(emitted & (delta == 2))),
        gate_closed=int(np.count_nonzero(~gate)),
        gate_open_missing=int(np.count_nonzero(gate & ~emitted)),
    )
    c.require(
        sum(counts.values()) == len(y), "Severity partition does not sum to population"
    )
    confusion = np.zeros((3, 5), dtype=int)
    for truth, pred, is_open, has_class in zip(gold, guess, gate, emitted):
        col = 0 if not is_open else 1 if not has_class else 2 + int(pred)
        confusion[int(truth), col] += 1
    mae = float(np.abs(delta[emitted]).mean()) if emitted.any() else None
    return dict(
        counts=counts,
        rates={k: v / len(y) for k, v in counts.items()},
        confusion=confusion.tolist(),
        emitted=int(emitted.sum()),
        conditional_ordinal_mae=mae,
    )


def hierarchy(rows, key):
    pools = defaultdict(list)
    for row in rows:
        pools[row["pool"]].append(row[key])
    return np.mean([np.mean(v, axis=0) for v in pools.values()], axis=0)


def fixtures():
    y = np.array([[0, 0], [1, 0], [1, 1], [1, 0]], dtype=bool)
    x = np.array([[0, 0], [1, 1], [0, 1], [0, 0]], dtype=bool)
    gate = np.array([0, 1, 1, 0], dtype=bool)
    equal(
        "Jaccard includes known empty match", overlap(y, x, gate)["sample_jaccard"], 0.5
    )
    equal(
        "Gate-aware exact excludes empty wrong gate",
        overlap(y, x, gate)["gate_aware_exact"],
        0.0,
    )
    equal(
        "Independent legal empty match",
        overlap(y, x, gate, True)["gate_aware_exact"],
        0.25,
    )
    yy = np.eye(3, dtype=bool)[[0, 1, 2, 2, 1, 0, 2]]
    xx = np.eye(3, dtype=bool)[[2, 2, 0, 1, 1, 0, 2]]
    gg = np.array([1, 1, 1, 1, 1, 0, 1], dtype=bool)
    xx[5:7] = 0
    r = ordinal(yy, xx, gg)
    equal(
        "Every severity partition including open missing",
        [r["counts"][k] for k in ERRORS],
        [1, 1, 1, 1, 1, 1, 1],
    )
    equal(
        "Ordinal MAE excludes two missing outcomes", r["conditional_ordinal_mae"], 1.2
    )


def main():
    c.require(
        not (p.BASE / "reporting_values.json").exists(),
        "Preserve earlier derived results",
    )
    p.save(
        "selected_reporting_scope.json",
        dict(
            methods=METHODS,
            legal_methods=LEGAL,
            multilabel_endpoints=[
                "en_test/hate_type",
                "en_test/target_group",
                "de_legal_test/legal",
            ],
            severity_endpoint="en_test/severity",
            populations="Original common-nine masks, not new intersection or primary pair-specific masks.",
            reason="Use all eight procedures already in the final English context table; keep the German reference separate. No method or label selected by its result.",
            main_reporting="Supporting tables and a short main-text explanation. No new hypothesis test, retraining or replacement of primary scores.",
            supplemental_severity="All eight current English procedures retained; the original three Qwen rows are independently checked against the earlier severity report.",
        ),
    )
    fixtures()
    values = {}
    fit_values = {}
    figure = []
    display = c.read(p.READER / "evidence/jaccard/display_values.json")
    for name, hash_value in display["source_bindings"].items():
        c.require(c.sha(c.ROOT / name) == hash_value, "Jaccard source changed")
        c.bind(c.ROOT / name)
    for ep in [
        "en_test__hate_type",
        "en_test__target_group",
        "en_test__severity",
        "de_legal_test__legal",
    ]:
        root = p.PERFORMANCE / "endpoints" / ep
        meta = c.read(root / "metadata.json")
        scores = c.read(root / "scores.json")
        quality = c.read(p.QUALITY / "endpoints" / f"{ep}.json")
        with np.load(c.bind(root / "aligned.npz"), allow_pickle=False) as a:
            y0 = a["gold"].astype(bool)
            x0 = a["predictions"].astype(bool)
            gate0 = a["gate"].astype(bool)
            valid = a["valid"]
            mask = a["common"]
        c.require(
            len(meta["models"]) == len(x0) and len(meta["ids"]) == len(y0),
            "Alignment metadata mismatch",
        )
        qix = [
            i for i, m in enumerate(meta["models"]) if m["group"].startswith("qwen/")
        ]
        c.require(
            len(qix) == 9 and np.array_equal(mask, valid[qix].all(axis=0)),
            "Original common-nine mask changed",
        )
        y = y0[mask]
        n = len(y)
        methods = LEGAL if ep.startswith("de_legal") else METHODS
        c.require(
            n
            == {
                "en_test__hate_type": 1135,
                "en_test__target_group": 2326,
                "en_test__severity": 1465,
                "de_legal_test__legal": 173,
            }[ep],
            "Population mismatch",
        )
        endpoint = dict(
            endpoint=meta["endpoint"],
            n=n,
            labels=meta["labels"],
            macro_labels=meta["macro_labels"],
            ids=[key for key, on in zip(meta["ids"], mask) if on],
            gold_support=y.sum(0).tolist(),
            groups={},
        )
        for group, title in methods.items():
            ix = [i for i, m in enumerate(meta["models"]) if m["group"] == group]
            c.require(
                ix and valid[ix][:, mask].all(), "Incomplete saved prediction group"
            )
            rows = []
            for i in ix:
                model = meta["models"][i]
                x = x0[i, mask]
                gate = gate0[i, mask]
                if not ep.startswith("de_legal"):
                    c.require(
                        not x[~gate].any(),
                        "Predictions are not delivered gated outputs",
                    )
                result = label_metrics(y, x)
                pr, rc, f1, support = precision_recall_fscore_support(
                    y, x, average=None, zero_division=0
                )
                for key, expected in [
                    ("precision", pr),
                    ("recall", rc),
                    ("f1", f1),
                    ("support", support),
                ]:
                    equal(f"{ep}/{i}/sklearn/{key}", result[key], expected)
                original = scores["common_nine"]["runs"][i]
                for key, source_key in [
                    ("precision", "label_precision"),
                    ("recall", "label_recall"),
                    ("f1", "label_f1"),
                    ("support", "gold_support"),
                    ("tp", "tp"),
                    ("fp", "fp"),
                    ("fn", "fn"),
                ]:
                    equal(f"{ep}/{i}/saved/{key}", result[key], original[source_key])
                macro = float(result["f1"][meta["macro_indices"]].mean())
                equal(f"{ep}/{i}/saved_macro", macro, original["macro_f1"])
                overlap_result = overlap(y, x, gate, ep.startswith("de_legal"))
                equal(
                    f"{ep}/{i}/sklearn_jaccard",
                    overlap_result["sample_jaccard"],
                    jaccard_score(y, x, average="samples", zero_division=1),
                )
                row = dict(
                    model_index=i,
                    pool=model["pool"],
                    seed=model["seed"],
                    **{k: v.tolist() for k, v in result.items()},
                    macro_f1=macro,
                    **overlap_result,
                )
                if ep.endswith("severity"):
                    detail = ordinal(y, x, gate)
                    row["severity"] = detail
                    emitted = x.sum(1) == 1
                    if emitted.any():
                        equal(
                            f"{ep}/{i}/sklearn_mae",
                            detail["conditional_ordinal_mae"],
                            mean_absolute_error(
                                y.argmax(1)[emitted], x.argmax(1)[emitted]
                            ),
                        )
                    for key in ERRORS:
                        row[key] = detail["counts"][key]
                    row["ordinal_mae"] = detail["conditional_ordinal_mae"]
                    row["confusion"] = detail["confusion"]
                    row["emitted"] = detail["emitted"]
                rows.append(row)
            output = dict(
                method=title,
                fits=len(rows),
                pools=len({r["pool"] for r in rows}),
                label_rows=[],
            )
            means = {
                key: hierarchy(rows, key)
                for key in [
                    "tp",
                    "fp",
                    "fn",
                    "support",
                    "precision",
                    "recall",
                    "f1",
                    "macro_f1",
                    "sample_jaccard",
                    "gate_aware_exact",
                ]
            }
            saved = scores["common_nine"]["groups"][group]["mean"]
            for key, source_key in [
                ("precision", "label_precision"),
                ("recall", "label_recall"),
                ("f1", "label_f1"),
                ("support", "gold_support"),
                ("tp", "tp"),
                ("fp", "fp"),
                ("fn", "fn"),
                ("macro_f1", "macro_f1"),
            ]:
                equal(f"{ep}/{group}/aggregate/{key}", means[key], saved[source_key])
            for key in ["sample_jaccard", "gate_aware_exact"]:
                equal(
                    f"{ep}/{group}/saved_quality/{key}",
                    means[key],
                    quality["common_nine"]["groups"][group]["mean"][key],
                )
                output[key] = float(means[key])
            output["macro_f1"] = float(means["macro_f1"])
            for j, label in enumerate(meta["labels"]):
                output["label_rows"].append(
                    dict(
                        label=label,
                        support=int(means["support"][j]),
                        **{
                            k: float(means[k][j])
                            for k in ["precision", "recall", "f1", "tp", "fp", "fn"]
                        },
                    )
                )
            if ep.endswith("severity"):
                c.require(
                    all(r["ordinal_mae"] is not None for r in rows),
                    "Undefined fit MAE needs explicit handling",
                )
                counts = {key: float(hierarchy(rows, key)) for key in ERRORS}
                output["severity"] = dict(
                    counts=counts,
                    rates={k: v / n for k, v in counts.items()},
                    mean_emitted=float(hierarchy(rows, "emitted")),
                    conditional_ordinal_mae=float(hierarchy(rows, "ordinal_mae")),
                    confusion=hierarchy(rows, "confusion").tolist(),
                    confusion_columns=[
                        "gate_closed",
                        "gate_open_missing",
                        "low",
                        "medium",
                        "high",
                    ],
                )
                equal(f"{ep}/{group}/partition", sum(counts.values()), n)
                equal(
                    f"{ep}/{group}/exactness",
                    counts["correct"] / n,
                    output["gate_aware_exact"],
                )
            endpoint["groups"][group] = output
            fit_values[ep + "|" + group] = rows
        values[ep] = endpoint
        print("Checked", ep, "N", n, "methods", len(methods), flush=True)

    for row in display["figure"]:
        value = values["en_test__hate_type"]["groups"][row["group"]]
        equal(
            "Figure11/" + row["group"],
            [
                value["macro_f1"],
                100 * value["sample_jaccard"],
                100 * value["gate_aware_exact"],
            ],
            [row["macro_f1"], row["mean_jaccard_percent"], row["exact_percent"]],
        )
        figure.append(
            dict(
                group=row["group"],
                method=row["method"],
                macro_f1=value["macro_f1"],
                mean_jaccard_percent=100 * value["sample_jaccard"],
                exact_percent=100 * value["gate_aware_exact"],
            )
        )
    coordinates = c.read(p.READER / "evidence/jaccard/plot_coordinates.json")
    for panel in coordinates:
        c.require(
            panel["methods"] == [r["group"] for r in figure],
            "Plotted method order changed",
        )
        equal(
            "Plot coordinates " + panel["metric"],
            panel["values"],
            [r[panel["metric"]] for r in figure],
        )

    broad = c.read(p.BROAD / "endpoints/en_test__severity/results.json")
    with np.load(
        c.bind(p.BROAD / "endpoints/en_test__severity/encoded.npz"), allow_pickle=False
    ) as a:
        mask = a["common_nine"]
        gold = a["gold"][mask].astype(bool)
        c.require(
            [x for x, on in zip(broad["ids"], mask) if on]
            == values["en_test__severity"]["ids"],
            "Severity IDs differ between studies",
        )
        for condition in ["rag_full", "rag_small", "zero_shot"]:
            ix = [i for i, s in enumerate(broad["settings"]) if s[0] == condition]
            for repeat, i in enumerate(ix):
                result = ordinal(
                    gold,
                    a["predicted"][i, mask].astype(bool),
                    a["gate"][i, mask].astype(bool),
                )
                previous = broad["common_nine"][condition]["detail"][repeat]
                equal(
                    "Broad severity confusion " + condition + str(repeat),
                    result["confusion"],
                    previous["confusion"],
                )
                equal(
                    "Broad severity counts " + condition + str(repeat),
                    [result["counts"][k] for k in ERRORS],
                    [previous["counts"][k] for k in ERRORS],
                )
                equal(
                    "Broad severity MAE " + condition + str(repeat),
                    result["conditional_ordinal_mae"],
                    previous["conditional_ordinal_mae"],
                )
                aligned = fit_values["en_test__severity|qwen/" + condition][repeat][
                    "severity"
                ]
                c.require(
                    result == aligned, "Independent severity representations disagree"
                )
    p.save(
        "reporting_values.json",
        dict(
            status="verified descriptive values",
            endpoints=values,
            figure11=figure,
            metrics="Per-label scores are per-model, then mean within pool and equally across pools. Support is the shared gold count, not multiplied by repeats. Severity rates use all 1465 texts; MAE is conditional on emission separately per fit.",
            new_significance_tests=0,
            source_sha256=c.BINDINGS,
        ),
    )
    p.save("fit_level_checks.json", dict(fits=fit_values, source_sha256=c.BINDINGS))
    p.save(
        "numerical_verification.json",
        dict(
            status="passed",
            checks=CHECKS,
            numerical_values=sum(r["values"] for r in CHECKS),
            methods="Direct Boolean counts and per-text intersection/union; independent sklearn P/R/F1/support/Jaccard/MAE; saved per-fit and pool means; separate original severity encoding.",
            source_sha256=c.BINDINGS,
        ),
    )
    print(
        "Verified",
        len(CHECKS),
        "checks and",
        sum(r["values"] for r in CHECKS),
        "numerical values. No model calls.",
        flush=True,
    )


if __name__ == "__main__":
    main()

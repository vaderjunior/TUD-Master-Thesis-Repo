from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from analysis.feedback.transfer import archive as a
from analysis.feedback.transfer import quality_metrics as k
from analysis.feedback.transfer import quality as p


def main():
    done = a.read(p.OUT / "calculated.json")
    for name, digest in {**done["bindings"], **done["artifacts"]}.items():
        a.need(a.sha(a.ROOT / name) == digest, "Changed analysis " + name)
    a.need(
        not (p.OUT / "diagnostics_complete.json").exists(),
        "Preserve previous diagnostics",
    )
    results = a.read(p.OUT / "quality_results.json")
    meta = list(a.rows(p.OUT / "metadata.jsonl"))
    with np.load(p.OUT / "prediction_codes.npz", allow_pickle=False) as f:
        codes = {key: f[key] for key in f.files}
    per_label = {}
    shapes = {}
    pairs = {}
    lut = np.asarray([i.bit_count() for i in range(128)], dtype=np.int8)
    for name, r in results.items():
        head = r["head"]
        if head not in ("hate_type", "target_group", "severity") or not r["common_n"]:
            continue
        ids = r["common_indices"]
        counts = np.asarray(r["counts"])
        labels = k.LABELS[head]
        ix = {v: i for i, v in enumerate(k.names(head))}
        each = {}
        for label in labels:
            tp, fp, fn = [counts[..., ix[label + ":" + s]] for s in ("tp", "fp", "fn")]
            score = np.stack(
                [
                    k.divide(tp, tp + fp, 0),
                    k.divide(tp, tp + fn, 0),
                    k.divide(2 * tp, 2 * tp + fp + fn, 0),
                ],
                axis=-1,
            )
            each[label] = dict(
                in_population_macro=label in r["population_macro_labels"],
                full_group_positive_support=r["group_full_gold_support"][label],
                common_positive_support=int(tp[0, 0] + fn[0, 0]),
                counts_tp_fp_fn=np.stack([tp, fp, fn], axis=-1),
                run_precision_recall_f1=score,
                means_precision_recall_f1=k.summaries(score),
            )
        per_label[name] = p.native(each)
        if head in ("hate_type", "target_group"):
            gold = np.asarray(
                [
                    sum(1 << labels.index(lab) for lab in meta[i][k.GOLD[head]])
                    for i in ids
                ],
                dtype=np.int16,
            )
            pred = codes[head][ids]
            goldhist = np.bincount(lut[gold], minlength=len(labels) + 1)
            predhist = np.stack(
                [
                    np.bincount(lut[pred[:, c, t]], minlength=len(labels) + 1)
                    for c in range(4)
                    for t in range(3)
                ]
            ).reshape(4, 3, len(labels) + 1)
            extra = lut[pred & (~gold[:, None, None] & 127)]
            missing = lut[gold[:, None, None] & (~pred & 127)]
            a.need(
                np.array_equal(extra.sum(axis=0), counts[..., ix["extra_labels"]])
                and np.array_equal(
                    missing.sum(axis=0), counts[..., ix["missing_labels"]]
                ),
                "Independent bitset/set counts differ",
            )
            a.need(
                np.array_equal(
                    (pred == gold[:, None, None]).sum(axis=0), counts[..., ix["exact"]]
                ),
                "Independent exact count differs",
            )
            shapes[name] = p.native(
                dict(
                    gold_cardinality=goldhist,
                    predicted_cardinality=predhist,
                    gold_sets=dict(Counter(str(v) for v in gold.tolist())),
                    categories={
                        c: counts[..., ix[c]]
                        for c in (
                            "exact",
                            "only_extra",
                            "only_missing",
                            "both_extra_missing",
                            "contains_gold",
                        )
                    },
                    mean_gold_labels=float(lut[gold].mean()),
                    known_empty_gold=int((gold == 0).sum()),
                    note="Observed taxonomy bit sets. Containment is automatically true for known-empty gold; it does not require an exact answer.",
                )
            )
        else:
            gold = np.asarray(
                [labels.index(meta[i]["gold_severity"]) for i in ids], dtype=np.int16
            )
            pred = codes[head][ids]
            confusion = np.stack(
                [
                    counts[..., ix[f"confusion:{g}:{h}"]]
                    for g in labels
                    for h in ["none", *labels]
                ],
                axis=-1,
            ).reshape(4, 3, 3, 4)
            a.need(
                np.all(confusion.sum(axis=(-2, -1)) == len(ids)),
                "Severity confusion denominator",
            )
            missing_category_positive = ((pred < 0) & (codes["gates"][ids] == 1)).sum(
                axis=0
            )
            shapes[name] = p.native(
                dict(
                    gold_order=labels,
                    prediction_order=["none", *labels],
                    confusion=confusion,
                    no_category_despite_positive_gate=missing_category_positive,
                    note="No-category is an error on known severity gold. Ordinal distances are undefined for no-category outputs; retain their coverage separately.",
                )
            )
        if r["view"] != "main" or r["group_kind"] not in ("global", "source"):
            continue
        if head == "severity":
            gb = 1 << gold
            pb = np.where(pred >= 0, 1 << np.maximum(pred, 0), 0).astype(np.int16)
        else:
            gb = gold
            pb = pred
        groups = {}
        for condition in range(1, 4):
            for repeat in range(3):
                ref_gate = codes["gates"][ids, 0, repeat].astype(bool)
                trt_gate = codes["gates"][ids, condition, repeat].astype(bool)
                ref = pb[:, 0, repeat]
                trt = pb[:, condition, repeat]
                rex = ref == gb
                tex = trt == gb
                partitions = {
                    "all": np.ones(len(ids), dtype=bool),
                    "both_positive": ref_gate & trt_gate,
                    "gained_positive_gate": ~ref_gate & trt_gate,
                    "lost_positive_gate": ref_gate & ~trt_gate,
                    "both_negative": ~ref_gate & ~trt_gate,
                }
                for part, mask in partitions.items():
                    group = groups.setdefault(
                        part,
                        dict(
                            pairs=0,
                            keys=set(),
                            by_pool=Counter(),
                            exact_fix=0,
                            exact_regression=0,
                            both_exact=0,
                            neither_exact=0,
                            gated_exact_fix=0,
                            gated_exact_regression=0,
                            correct_labels_added=0,
                            correct_labels_removed=0,
                            extra_labels_added=0,
                            extra_labels_removed=0,
                        ),
                    )
                    group["pairs"] += int(mask.sum())
                    group["keys"].update(meta[i]["key"] for i, m in zip(ids, mask) if m)
                    group["by_pool"][str(condition)] += int(mask.sum())
                    rg = rex & ref_gate
                    tg = tex & trt_gate
                    for key, value in dict(
                        exact_fix=~rex & tex,
                        exact_regression=rex & ~tex,
                        both_exact=rex & tex,
                        neither_exact=~rex & ~tex,
                        gated_exact_fix=~rg & tg,
                        gated_exact_regression=rg & ~tg,
                    ).items():
                        group[key] += int((value & mask).sum())
                    added = trt & (~ref & 127)
                    removed = ref & (~trt & 127)
                    for key, value in dict(
                        correct_labels_added=lut[added & gb],
                        correct_labels_removed=lut[removed & gb],
                        extra_labels_added=lut[added & (~gb & 127)],
                        extra_labels_removed=lut[removed & (~gb & 127)],
                    ).items():
                        group[key] += int(value[mask].sum())
        for part, group in groups.items():
            group["unique_texts"] = len(group.pop("keys"))
            group["by_pool"] = dict(group["by_pool"])
            a.need(
                group["pairs"]
                == sum(
                    group[x]
                    for x in (
                        "exact_fix",
                        "exact_regression",
                        "both_exact",
                        "neither_exact",
                    )
                ),
                "Incomplete transition partition",
            )
            group["exact_difference"] = (
                (group["exact_fix"] - group["exact_regression"]) / group["pairs"]
                if group["pairs"]
                else None
            )
        all_ = groups["all"]
        a.need(
            abs(all_["exact_difference"] - r["metrics"]["exact_set"]["difference"])
            < 1e-13,
            "Matched exact transition identity",
        )
        a.need(
            abs(
                (all_["gated_exact_fix"] - all_["gated_exact_regression"])
                / all_["pairs"]
                - r["metrics"]["gated_exact_set"]["difference"]
            )
            < 1e-13,
            "Gated exact transition identity",
        )
        pairs[name] = dict(
            partitions=groups,
            note="Dependent matched-repeat pairs. Reference outputs are reused across three pool comparisons here; not additional actual reference runs or causal mediation. Conditional partitions are selected on observed gates.",
        )
    p.save("per_label_quality.json", per_label)
    p.save("set_and_severity_diagnostics.json", shapes)
    p.save("conditional_gate_pairs.json", pairs)
    files = [
        p.OUT / name
        for name in (
            "per_label_quality.json",
            "set_and_severity_diagnostics.json",
            "conditional_gate_pairs.json",
        )
    ]
    p.save(
        "diagnostics_complete.json",
        dict(
            status="part4_diagnostics_calculated_verification_pending",
            created_utc=a.utc(),
            calculated_sha256=a.sha(p.OUT / "calculated.json"),
            artifacts={a.rel(f): a.sha(f) for f in files},
            sources={a.rel(Path(__file__)): a.sha(Path(__file__))},
            label_and_shape_views=len(per_label),
            conditional_pair_views=len(pairs),
            new_Qwen_calls=0,
            new_fits=0,
        ),
    )
    print(
        f"Saved {len(per_label)} label/shape views and {len(pairs)} gate-pair decompositions."
    )


if __name__ == "__main__":
    with a.offline():
        main()

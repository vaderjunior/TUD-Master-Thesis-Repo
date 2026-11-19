from collections import Counter, defaultdict
from pathlib import Path
from types import FunctionType, SimpleNamespace
from analysis.classification.matched import common as m
import numpy as np
import yaml

s = m.scoring


def masked(value, keep):
    keep = np.asarray(keep, dtype=bool)
    valid = value["valid"] & keep
    counts = value["counts"].copy()
    counts[~valid] = 0
    return value | {"valid": valid, "counts": counts}


def point(left, right):
    allc = left + right
    common = np.logical_and.reduce([v["valid"] for v in allc])
    if not common.any() or not allc[0]["macro_indices"]:
        return dict(n=int(common.sum()), delta=None)
    scores = [
        float(s.original.macro(v["counts"][common].sum(axis=0), v["macro_indices"]))
        for v in allc
    ]
    return dict(
        n=int(common.sum()),
        delta=float(np.mean(scores[: len(left)]) - np.mean(scores[len(left) :])),
        left=float(np.mean(scores[: len(left)])),
        right=float(np.mean(scores[len(left) :])),
    )


def captured_contrast(left, right, clusters, **kwargs):
    captured = []

    def quantile(values, q, *args, **kw):
        captured.append(np.asarray(values, dtype=float))
        return np.quantile(values, q, *args, **kw)

    proxy = SimpleNamespace(**(vars(np) | {"quantile": quantile}))
    function = FunctionType(
        s.contrast.__code__, vars(s) | {"np": proxy}, argdefs=s.contrast.__defaults__
    )
    function.__kwdefaults__ = s.contrast.__kwdefaults__
    result = function(left, right, clusters, **kwargs)
    m.a.require(
        captured and all(np.array_equal(captured[0], v) for v in captured),
        "Bootstrap capture differs",
    )
    return result, captured[0]


def known_values(row):
    return {d: v for d in m.DIMENSIONS if (v := s.original.truth(row, d)) is not None}


def exact_known(row, output):
    values = known_values(row)
    if output is None or not values:
        return None
    return all(output[k] == v for k, v in values.items())


def diagnostic(rows, output):
    count = Counter()
    per_field = defaultdict(Counter)
    severity = Counter()
    signatures = Counter()
    for row in rows:
        value = output[row["id"]]
        known = known_values(row)
        count["eligible"] += 1
        signatures["+".join(known)] += 1
        if value is None:
            count["invalid"] += 1
            continue
        count["valid"] += 1
        count["exact_known"] += int(exact_known(row, value))
        if row["gate"] is not None:
            truth, pred = row["gate"], value["hate"]
            count["hateful" if truth else "benign"] += 1
            count[
                "tp" if truth and pred else "fn" if truth else "fp" if pred else "tn"
            ] += 1
        for dim, truth in known.items():
            g = per_field[dim]
            pred = value[dim]
            g["n"] += 1
            g["exact"] += truth == pred
            if isinstance(truth, list):
                ys, xs = set(truth), set(pred)
                g["gold_labels"] += len(ys)
                g["predicted_labels"] += len(xs)
                g["extra_labels"] += len(xs - ys)
                g["missing_labels"] += len(ys - xs)
                g["empty_prediction"] += not pred
                g["gate_false"] += not value["hate"]
            if dim == "severity":
                severity[str(truth) + " -> " + str(pred)] += 1
                g["gate_false"] += not value["hate"]
    return dict(
        counts=dict(count),
        known_field_signatures=dict(signatures),
        exact_known_valid=(
            count["exact_known"] / count["valid"] if count["valid"] else None
        ),
        exact_known_full_bounds=(
            [
                count["exact_known"] / count["eligible"],
                (count["exact_known"] + count["invalid"]) / count["eligible"],
            ]
            if count["eligible"]
            else [None, None]
        ),
        fpr=count["fp"] / count["benign"] if count["benign"] else None,
        fnr=count["fn"] / count["hateful"] if count["hateful"] else None,
        fields=dict(per_field),
        severity_confusion=dict(severity),
    )


def summary(value, records, dimension):
    result = s.summarize(value, records, dimension)
    ok = value["valid"]
    y = value["gold"][ok]
    x = value["pred"][ok]
    result.update(
        mean_gold_label_count=float(y.sum(axis=1).mean()) if len(y) else None,
        mean_predicted_label_count=float(x.sum(axis=1).mean()) if len(y) else None,
        mean_extra_labels=float(((1 - y) * x).sum(axis=1).mean()) if len(y) else None,
        mean_missing_labels=float((y * (1 - x)).sum(axis=1).mean()) if len(y) else None,
    )
    return result


def run():
    with m.runtime.offline():
        groups, predictions, metadata = m.load()
        taxonomy = yaml.safe_load(
            (m.ROOT / "config/taxonomy.yaml").read_text(encoding="utf8")
        )
        results = {}
        diagnostics = {}
        inference_files = {}
        contribution_checks = 0
        for phase, rr in groups.items():
            keys = {r["id"] for r in rr}
            reference = {r["id"]: r for r in rr}
            outputs = {
                ident: {k: v[k] for k in keys} for ident, v in predictions.items()
            }
            common_keys = {
                k for k in keys if all(v[k] is not None for v in outputs.values())
            }
            diagnostics[phase] = {
                "per_run": {},
                "all_conditions_common_ids": sorted(common_keys),
                "paired": {},
            }
            for ident, output in sorted(outputs.items()):
                diagnostics[phase]["per_run"][ident[0] + "/repeat" + str(ident[1])] = (
                    dict(
                        own=diagnostic(rr, output),
                        common_all=diagnostic(
                            [r for r in rr if r["id"] in common_keys], output
                        ),
                        by_source={
                            src: diagnostic(
                                [r for r in rr if r["source"] == src], output
                            )
                            for src in sorted({r["source"] for r in rr})
                        },
                    )
                )
            for family in m.PAIRS:
                lhs = m.names(outputs, "retrieved_complete")
                rhs = m.names(outputs, family)
                eligible = [
                    r
                    for r in rr
                    if all(outputs[k][r["id"]] is not None for k in lhs + rhs)
                ]
                d = [diagnostic(eligible, outputs[k]) for k in lhs + rhs]
                diagnostics[phase]["paired"][family] = {
                    "n_common": len(eligible),
                    "left_settings": lhs,
                    "right_settings": rhs,
                    "left_mean": {
                        field: m.fraction_mean(v[field] for v in d[: len(lhs)])
                        for field in ["exact_known_valid", "fpr", "fnr"]
                    },
                    "right_mean": {
                        field: m.fraction_mean(v[field] for v in d[len(lhs) :])
                        for field in ["exact_known_valid", "fpr", "fnr"]
                    },
                    "left_mean_counts": {
                        field: float(
                            np.mean([v["counts"].get(field, 0) for v in d[: len(lhs)]])
                        )
                        for field in ["fp", "fn", "tp", "tn", "exact_known"]
                    },
                    "right_mean_counts": {
                        field: float(
                            np.mean([v["counts"].get(field, 0) for v in d[len(lhs) :]])
                        )
                        for field in ["fp", "fn", "tp", "tn", "exact_known"]
                    },
                }
            for dim in m.DIMENSIONS:
                ep = s.endpoint(rr, dim, taxonomy)
                if not ep["scorable_ids"]:
                    continue
                runs = {}
                for ident, output in sorted(outputs.items()):
                    value = s.contributions(rr, output, ep)
                    independently = m.gold.contributions(rr, output, ep)
                    for field in ["gold", "pred", "counts", "valid"]:
                        m.a.require(
                            np.array_equal(value[field], independently[field]),
                            "Independent contribution mismatch",
                        )
                    runs[ident] = value
                    contribution_checks += 1
                first = next(iter(runs.values()))
                ids = first["ids"]
                clusters = [reference[k]["text_cluster"] for k in ids]
                sources = [reference[k]["source"] for k in ids]
                summaries = {
                    cond + "/repeat" + str(rep): summary(v, rr, dim)
                    for (cond, rep), v in runs.items()
                }
                replicate = {}
                for cond in sorted({k[0] for k in runs}):
                    vals = [
                        summaries[cond + "/repeat" + str(rep)]["macro_f1_valid_only"]
                        for rep in range(3)
                    ]
                    exact = [
                        summaries[cond + "/repeat" + str(rep)]["exact_set_accuracy"]
                        for rep in range(3)
                    ]
                    replicate[cond] = {
                        "values": vals,
                        "mean": m.fraction_mean(vals),
                        "sample_sd": (
                            float(np.std(vals, ddof=1))
                            if all(v is not None for v in vals)
                            else None
                        ),
                        "exact_values": exact,
                        "exact_mean": m.fraction_mean(exact),
                    }
                contrasts = {}
                robustness = {}
                if dim == s.MATCHED_DIMS[phase]:
                    all_mask = np.logical_and.reduce(
                        [v["valid"] for v in runs.values()]
                    )
                    for family in m.PAIRS:
                        lhs = m.names(runs, "retrieved_complete")
                        rhs = m.names(runs, family)
                        left = [runs[k] for k in lhs]
                        right = [runs[k] for k in rhs]
                        primary = family in m.PAIRS[:2]
                        value, draws = captured_contrast(
                            left,
                            right,
                            clusters,
                            sources=sources,
                            family=6 if primary else None,
                        )
                        p = (
                            m.BASE
                            / "bootstrap"
                            / (m.slug(phase, dim, family, "main") + ".npy")
                        )
                        m.array(p, draws)
                        inference_files[m.runtime.rel(p)] = m.c.sha(p)
                        value["bootstrap_file"] = m.runtime.rel(p)
                        value["left_settings"] = lhs
                        value["right_settings"] = rhs
                        sens, draws = captured_contrast(
                            left, right, clusters, sources=sources, stratify=True
                        )
                        p = (
                            m.BASE
                            / "bootstrap"
                            / (m.slug(phase, dim, family, "stratified") + ".npy")
                        )
                        m.array(p, draws)
                        inference_files[m.runtime.rel(p)] = m.c.sha(p)
                        sens["bootstrap_file"] = m.runtime.rel(p)
                        value["source_stratified"] = sens
                        contrasts[family] = value
                        robust = {
                            "common_all": point(
                                [masked(v, all_mask) for v in left],
                                [masked(v, all_mask) for v in right],
                            ),
                            "by_variant": {
                                name: point(
                                    left, [runs[k] for k in rhs if k[0] == name]
                                )
                                for name in sorted({k[0] for k in rhs})
                            },
                            "by_repeat": {
                                str(rep): point(
                                    [runs[k] for k in lhs if k[1] == rep],
                                    [runs[k] for k in rhs if k[1] == rep],
                                )
                                for rep in range(3)
                            },
                            "leave_one_repeat_out": {
                                str(rep): point(
                                    [runs[k] for k in lhs if k[1] != rep],
                                    [runs[k] for k in rhs if k[1] != rep],
                                )
                                for rep in range(3)
                            },
                            "by_source": {
                                src: point(
                                    [masked(v, np.array(sources) == src) for v in left],
                                    [
                                        masked(v, np.array(sources) == src)
                                        for v in right
                                    ],
                                )
                                for src in sorted(set(sources))
                            },
                            "leave_one_source_out": {
                                src: point(
                                    [masked(v, np.array(sources) != src) for v in left],
                                    [
                                        masked(v, np.array(sources) != src)
                                        for v in right
                                    ],
                                )
                                for src in sorted(set(sources))
                                if len(set(sources)) > 1
                            },
                        }
                        variants = sorted({k[0] for k in rhs})
                        if len(variants) > 1:
                            robust["leave_one_variant_out"] = {
                                name: point(
                                    left, [runs[k] for k in rhs if k[0] != name]
                                )
                                for name in variants
                            }
                        if primary:
                            strict = [
                                masked(
                                    runs[k],
                                    [
                                        metadata[item, *k]["n_valid"] == 3
                                        for item in ids
                                    ],
                                )
                                for k in lhs + rhs
                            ]
                            strict_result, draws = captured_contrast(
                                strict[: len(lhs)],
                                strict[len(lhs) :],
                                clusters,
                                sources=sources,
                            )
                            p = (
                                m.BASE
                                / "bootstrap"
                                / (m.slug(phase, dim, family, "strict_votes") + ".npy")
                            )
                            m.array(p, draws)
                            inference_files[m.runtime.rel(p)] = m.c.sha(p)
                            strict_result["bootstrap_file"] = m.runtime.rel(p)
                            robust["three_valid_votes"] = strict_result
                        robustness[family] = robust
                        print(
                            "Scored",
                            phase,
                            dim,
                            family,
                            round(100 * value["mean_difference"], 3),
                            "percentage points",
                            flush=True,
                        )
                results[phase + "/" + dim] = {
                    "endpoint": ep,
                    "per_condition_repeat": summaries,
                    "replicate_summary": replicate,
                    "contrasts": contrasts,
                    "robustness": robustness,
                }
        m.save(
            m.BASE / "metrics.json",
            {
                "status": "frozen_kernel_scored_independent_bits_checked",
                "contract": s.contract(),
                "results": results,
                "contribution_checks": contribution_checks,
                "bootstrap_bindings": inference_files,
            },
        )
        m.save(m.BASE / "diagnostics.json", diagnostics)
        m.save(
            m.BASE / "metrics_receipt.json",
            {
                "inputs": m.bindings(
                    [
                        m.BASE / "assembly.json",
                        m.BASE / "predictions.jsonl",
                        m.BASE / "gold_rosters.jsonl",
                    ]
                ),
                "outputs": m.bindings(
                    [m.BASE / "metrics.json", m.BASE / "diagnostics.json"]
                )
                | inference_files,
                "sources": m.bindings(
                    [Path(__file__), Path(s.__file__), Path(s.original.__file__)]
                ),
                "Qwen_calls": 0,
                "training_fits": 0,
            },
        )
        print("Scientific metrics complete.", flush=True)


if __name__ == "__main__":
    run()

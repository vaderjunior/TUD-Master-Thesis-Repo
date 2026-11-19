from collections import Counter
import hashlib
from pathlib import Path
import time
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.classification.broad import errors as e


def rates(y, p):
    z = Counter((int(a), int(b)) for a, b in zip(y, p))
    tn, fp, fn, tp = [z[x] for x in ((0, 0), (0, 1), (1, 0), (1, 1))]
    n = len(y)
    d = lambda a, b: a / b if b else None
    mf = lambda a, b: a / b if b else 0
    recall = d(tp, tp + fn)
    specificity = d(tn, tn + fp)
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        n=n,
        positive_gold=tp + fn,
        negative_gold=tn + fp,
        fpr=d(fp, tn + fp),
        fnr=d(fn, tp + fn),
        precision=d(tp, tp + fp),
        recall=recall,
        error_rate=d(fp + fn, n),
        accuracy=d(tp + tn, n),
        predicted_hate=d(tp + fp, n),
        balanced_accuracy=(
            (recall + specificity) / 2
            if recall is not None and specificity is not None
            else None
        ),
        macro_f1=(
            (mf(2 * tp, 2 * tp + fp + fn) + mf(2 * tn, 2 * tn + fp + fn)) / 2
            if n
            else None
        ),
    )


def summaries(y, p):
    values = [rates(y, row) for row in p]
    means = {
        key: (
            sum(row[key] for row in values) / len(values)
            if all(row[key] is not None for row in values)
            else None
        )
        for key in values[0]
    }
    return dict(repeats=values, mean=means)


def events(y, a, b):
    matrices = np.zeros((3, 3, 8), dtype=np.int64)
    numerators = np.zeros((len(y), 8), dtype=np.int64)
    stable = []
    for i, truth in enumerate(y):
        for j in range(3):
            for k in range(3):
                new, old = bool(a[k, i]), bool(b[j, i])
                truth = bool(truth)
                if not truth:
                    code = (
                        0
                        if old and not new
                        else 1 if new and not old else 4 if old and new else 6
                    )
                else:
                    code = (
                        2
                        if new and not old
                        else 3 if old and not new else 5 if not old and not new else 7
                    )
                matrices[j, k, code] += 1
                numerators[i, code] += 1
        new_correct = [bool(value) == bool(truth) for value in a[:, i]]
        old_correct = [bool(value) == bool(truth) for value in b[:, i]]
        stable.append(
            0
            if all(new_correct) and all(old_correct)
            else (
                1
                if all(new_correct) and not any(old_correct)
                else (
                    2
                    if not any(new_correct) and all(old_correct)
                    else 3 if not any(new_correct) and not any(old_correct) else 4
                )
            )
        )
    return matrices, numerators, np.asarray(stable, dtype=np.int8)


def propagation(rr, output, dim):
    fine = dim != "hate"
    field = {
        "hate_type": "hate_types",
        "target_group": "target_groups",
        "severity": "severity",
    }.get(dim)
    count = Counter(eligible=len(rr))
    count["valid"] = 0
    count["invalid"] = 0
    if fine:
        names = [
            "gold_positive_information",
            "gate_false",
            "positive_gold_gated_off",
            "positive_gold_bits_gated_off",
            "empty_gold_gate_false",
        ]
    else:
        names = [
            "valid_negative",
            "false_positive",
            "fp_type",
            "fp_target",
            "fp_severity",
            "fp_any_fine",
            "fp_all_fine",
            "fp_legal_present",
        ]
    for name in names:
        count[name] = 0
    for row in rr:
        out = output[row["key"]]
        if out is None:
            count["invalid"] += 1
            continue
        count["valid"] += 1
        if fine:
            truth = row[field]
            positive = (
                int(truth is not None) if dim == "severity" else int(len(truth) > 0)
            )
            count["gold_positive_information"] += positive
            if out["hate"] is False:
                count["gate_false"] += 1
                count["positive_gold_gated_off"] += positive
                count["positive_gold_bits_gated_off"] += (
                    positive if dim == "severity" else len(set(truth))
                )
                count["empty_gold_gate_false"] += 1 - positive
        elif row["gate"] is False:
            count["valid_negative"] += 1
            if out["hate"] is True:
                count["false_positive"] += 1
                bits = [
                    len(out["hate_type"]) > 0,
                    len(out["target_group"]) > 0,
                    out["severity"] is not None,
                ]
                for name, bit in zip(("fp_type", "fp_target", "fp_severity"), bits):
                    count[name] += bit
                count["fp_any_fine"] += any(bits)
                count["fp_all_fine"] += all(bits)
                count["fp_legal_present"] += len(out["legal"]) > 0
    return dict(count)


def run():
    start = time.perf_counter()
    receipt = e.c.read(e.BASE / "analysis_complete.json")
    e.r.verify(receipt["sources"])
    e.r.verify(receipt["outputs"])
    groups, _, runs, endpoints, _ = e.reader.load()
    results = e.c.read(e.BASE / "binary_results.json")
    checks = Counter()
    maxdiff = 0.0

    def same(actual, expected, label):
        nonlocal maxdiff
        if isinstance(expected, dict):
            for key, value in expected.items():
                same(actual[key], value, label + "/" + key)
        elif isinstance(expected, list):
            e.require(len(actual) == len(expected), "Length differs " + label)
            for a, b in zip(actual, expected):
                same(a, b, label)
        elif expected is None:
            e.require(actual is None, "Undefined value differs " + label)
        else:
            a = np.asarray(actual)
            b = np.asarray(expected)
            e.require(a.shape == b.shape, "Shape differs " + label)
            error = float(np.max(np.abs(a - b))) if a.size else 0.0
            maxdiff = max(maxdiff, error)
            e.require(
                np.allclose(a, b, rtol=0, atol=2e-12, equal_nan=True),
                "Independent value differs " + label,
            )
            checks["numeric_values"] += int(a.size)

    expected_cases = []
    for phase in e.PHASES:
        data = results[phase]
        rr = groups[phase]
        ids = [row["key"] for row in rr]
        y = np.array([row["gate"] for row in rr], dtype=bool)
        ep = endpoints[phase + "/hate"]
        settings = [tuple(x) for x in data["settings"]]
        p = np.array(
            [
                [
                    False if runs[setting][key] is None else runs[setting][key]["hate"]
                    for key in ids
                ]
                for setting in settings
            ],
            dtype=bool,
        )
        v = np.array(
            [[runs[setting][key] is not None for key in ids] for setting in settings],
            dtype=bool,
        )
        for cond, values in data["own_valid"].items():
            for rep, value in enumerate(values):
                i = settings.index((cond, rep))
                mask = v[i]
                expected = rates(y[mask], p[i, mask]) | dict(
                    n_invalid=int((~mask).sum()),
                    invalid_negative=int((~mask & ~y).sum()),
                    invalid_positive=int((~mask & y).sum()),
                )
                same(value, expected, phase + "/own")
                checks["own_setting_summaries"] += 1
        nine = [
            settings.index((cond, rep)) for cond in e.CONDITIONS for rep in range(3)
        ]
        common = v[nine].all(axis=0)
        same(data["common_nine_texts"], int(common.sum()), "nine denominator")
        for cond, summary in data["common_nine"].items():
            ix = [settings.index((cond, rep)) for rep in range(3)]
            same(summary, summaries(y[common], p[ix][:, common]), "nine summary")
            checks["common_condition_summaries"] += 1
        for pair in data["paired"]:
            right = pair["reference"]
            ix = [
                settings.index((cond, rep))
                for cond in ("rag_full", right)
                for rep in range(3)
            ]
            common = v[ix].all(axis=0)
            yy = y[common]
            pp = p[ix][:, common]
            same(pair["full"], summaries(yy, pp[:3]), "paired full")
            same(pair["reference_summary"], summaries(yy, pp[3:]), "paired reference")
            same(
                [pair["n_common"], pair["n_positive"], pair["n_negative"]],
                [len(yy), int(yy.sum()), int((~yy).sum())],
                "paired denominators",
            )
            matrix, numerator, stable = events(yy, pp[:3], pp[3:])
            with np.load(e.ROOT / pair["arrays"], allow_pickle=False) as saved:
                for name, expected in [
                    ("common_indices", np.flatnonzero(common)),
                    ("pair_counts", matrix),
                    ("event_numerators", numerator),
                    ("stable", stable),
                ]:
                    e.require(
                        np.array_equal(saved[name], expected),
                        "Transition array differs " + name,
                    )
                draws = saved["metric_draws"].copy()
            same(
                pair["transitions"]["mean_events"],
                {
                    name: float(matrix[:, :, i].sum() / 9)
                    for i, name in enumerate(e.EVENTS)
                },
                "event means",
            )
            for gold in (0, 1):
                same(
                    pair["transitions"]["stable_by_gold"][str(gold)],
                    {
                        name: int(sum(s == i and g == gold for s, g in zip(stable, yy)))
                        for i, name in enumerate(e.STABLE)
                    },
                    "stable categories",
                )
            cl = [cluster for cluster, on in zip(ep["row_clusters"], common) if on]
            unique = sorted(set(cl))
            lookup = {key: i for i, key in enumerate(unique)}
            index = np.array([lookup[key] for key in cl])
            grouped = []
            for pred in pp:
                code = yy.astype(int) * 2 + pred.astype(int)
                grouped.append(
                    np.bincount(index * 4 + code, minlength=len(unique) * 4).reshape(
                        len(unique), 4
                    )
                )
            grouped = np.asarray(grouped, dtype=float)
            rng = np.random.default_rng(e.SEED)
            independent = []
            for first in range(0, e.DRAWS, 64):
                size = min(64, e.DRAWS - first)
                picks = rng.integers(0, len(unique), (size, len(unique)))
                weights = np.array(
                    [np.bincount(row, minlength=len(unique)) for row in picks],
                    dtype=float,
                )
                total = np.stack([weights @ matrix for matrix in grouped], axis=1)
                tn, fp, fn, tp = np.moveaxis(total, -1, 0)

                def div(a, b):
                    return np.divide(a, b, out=np.full(a.shape, np.nan), where=b > 0)

                rates3 = np.stack(
                    (
                        div(fp, tn + fp),
                        div(fn, fn + tp),
                        div(fp + fn, tn + fp + fn + tp),
                    ),
                    axis=-1,
                )
                independent.extend(
                    (rates3[:, :3].mean(axis=1) - rates3[:, 3:].mean(axis=1)).tolist()
                )
            independent = np.asarray(independent)
            same(draws, independent, "every diagnostic bootstrap value")
            for j, name in enumerate(e.METRICS):
                value = pair["diagnostic_intervals"]["metrics"][name]
                diff = (
                    pair["full"]["mean"][name] - pair["reference_summary"]["mean"][name]
                )
                same(value["difference"], diff, "diagnostic point")
                finite = np.isfinite(independent[:, j])
                same(value["undefined_draws"], int((~finite).sum()), "undefined count")
                if finite.all():
                    same(
                        value["ci95"],
                        np.quantile(independent[:, j], [0.025, 0.975]),
                        "diagnostic interval",
                    )
                else:
                    e.require(
                        value["ci95"] is None, "Undefined samples silently removed"
                    )
            checks["bootstrap_values_reproduced"] += draws.size
            checks["paired_comparisons"] += 1
            if right in ("rag_small", "zero_shot"):
                kept = [key for key, on in zip(ids, common) if on]
                for gold in (0, 1):
                    for category, code in [
                        ("fixed", 1),
                        ("introduced", 2),
                        ("persistent_error", 3),
                    ]:
                        label = ("fp_" if not gold else "fn_") + category
                        candidates = []
                        for key, gold_value, stable_value in zip(kept, yy, stable):
                            if gold_value != gold or stable_value != code:
                                continue
                            digest = hashlib.sha256(
                                "|".join(
                                    ["sq2-broad-part3-v1", phase, right, label, key]
                                ).encode()
                            ).hexdigest()
                            candidates.append((digest, key))
                        winner = min(candidates) if candidates else (None, None)
                        expected_cases.append(
                            dict(
                                population=phase,
                                reference=right,
                                category=label,
                                candidates=len(candidates),
                                key=winner[1],
                                selection_sha256=winner[0],
                            )
                        )
        print("Independent binary review passed:", phase, flush=True)
    delivery = e.c.read(e.BASE / "gate_delivery.json")
    for value in delivery["rows"]:
        phase, dim, cond, rep = (
            value["population"],
            value["dimension"],
            value["condition"],
            value["repeat"],
        )
        ids = endpoints[phase + "/" + dim]["scorable_ids"]
        lookup = {row["key"]: row for row in groups[phase]}
        rr = [lookup[key] for key in ids]
        if value["mask"] == "common_nine":
            rr = [
                row
                for row in rr
                if all(
                    runs[k, j][row["key"]] is not None
                    for k in e.CONDITIONS
                    for j in range(3)
                )
            ]
        expected = propagation(rr, runs[cond, rep], dim)
        same(value, expected, "delivered gate counts")
        checks["gate_delivery_views"] += 1
    for item in delivery["legal_independence"]:
        pred = [
            runs[item["condition"], item["repeat"]][row["key"]]
            for row in groups["de_legal_test"]
        ]
        valid = [value for value in pred if value is not None]
        same(
            item,
            dict(
                eligible=len(pred),
                valid=len(valid),
                predicted_nonhate=len([value for value in valid if not value["hate"]]),
                nonhate_with_legal=len(
                    [
                        value
                        for value in valid
                        if not value["hate"] and len(value["legal"]) > 0
                    ]
                ),
            ),
            "independent legal branch",
        )
    selected = e.c.read(e.BASE / "case_selection.json")["slots"]
    cases = e.c.read(e.BASE / "cases.json")
    gold = {row["key"]: row for rr in groups.values() for row in rr}
    e.require(
        len(expected_cases) == len(selected) == len(cases) == 36,
        "Selection slot mismatch",
    )
    for index, (expected, item, case) in enumerate(
        zip(expected_cases, selected, cases), 1
    ):
        expected["case_id"] = f"C{index:02d}"
        e.require(expected == item, "Deterministic example selection differs")
        e.require(
            all(case[key] == value for key, value in item.items()),
            "Case identity differs",
        )
        if item["key"] is None:
            continue
        e.require(case["gold"] == gold[item["key"]], "Case gold or input differs")
        for pred in case["predictions"]:
            e.require(
                pred["result"] == runs[pred["condition"], pred["repeat"]][item["key"]],
                "Case delivered result differs",
            )
            checks["case_predictions_reconciled"] += 1
    e.r.verify(e.c.read(e.BASE / "focused_source_bindings.json"))
    e.r.verify(receipt["outputs"])
    e.r.verify(receipt["sources"])
    e.require(
        checks["paired_comparisons"] == 11
        and checks["bootstrap_values_reproduced"] == 330000
        and checks["gate_delivery_views"] == 180,
        "Independent review incomplete",
    )
    e.save(
        e.BASE / "independent_review.json",
        dict(
            status="binary_counts_rates_transitions_draws_gate_delivery_and_selection_independently_verified",
            checks=dict(checks),
            maximum_absolute_difference=maxdiff,
            elapsed_seconds=time.perf_counter() - start,
            sources=e.bindings([Path(__file__), e.BASE / "analysis_complete.json"]),
            method="Fresh saved decisions; Python confusion counters and all-nine transition loops; alternative TN/FP/FN/TP grouped sampling; all gate counts and SHA256 selection independently reconstructed.",
            new_Qwen_calls=0,
            new_training=0,
            new_inference=0,
        ),
    )
    print(dict(checks), flush=True)


if __name__ == "__main__":
    with e.r.original.offline(), threadpool_limits(limits=2):
        run()

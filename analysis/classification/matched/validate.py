from analysis.classification.matched import common as m
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def truth(row, dim):
    if dim == "hate":
        return row["gate"]
    if dim != "legal" and row["gate"] is not True:
        return None
    return row.get(
        {
            "hate_type": "hate_types",
            "target_group": "target_groups",
            "severity": "severity",
            "legal": "legal",
        }[dim]
    )


def positive(value, dim):
    if dim == "hate":
        return {"hate" if value else "not_hate"}
    if dim == "severity":
        return {value} if value is not None else set()
    return set(value)


def arithmetic(rows, outputs, ep):
    labels = ep["all_labels"]
    dim = ep["dimension"]
    ids = []
    ys = []
    xs = []
    valid = []
    for row in rows:
        actual = truth(row, dim)
        if actual is None:
            continue
        gold = positive(actual, dim)
        pred = outputs[row["id"]]
        chosen = positive(pred[dim], dim) if pred is not None else set()
        ys.append([int(l in gold) for l in labels])
        xs.append([int(l in chosen) for l in labels])
        ids.append(row["id"])
        valid.append(pred is not None)
    y = np.asarray(ys)
    x = np.asarray(xs)
    v = np.asarray(valid)
    counts = np.stack(
        [((y == 1) & (x == 1)), ((y == 0) & (x == 1)), ((y == 1) & (x == 0))], axis=2
    ).astype(np.int64)
    counts[~v] = 0
    m.a.require(ids == ep["scorable_ids"], "Independent eligibility differs")
    m.a.require(
        [labels[i] for i, n in enumerate(y.sum(axis=0)) if n >= 10]
        == ep["macro_labels"],
        "Fixed label support differs",
    )
    return dict(
        ids=ids,
        labels=labels,
        macro_indices=[labels.index(l) for l in ep["macro_labels"]],
        gold=y,
        pred=x,
        valid=v,
        counts=counts,
    )


def macro(total, indices):
    numerator = total[..., 0] * 2.0
    denom = numerator + total[..., 1] + total[..., 2]
    scores = np.zeros_like(numerator, dtype=float)
    np.divide(numerator, denom, out=scores, where=denom > 0)
    return scores[..., indices].mean(axis=-1)


def stratified_record(expected):
    flag = expected["source_stratified"]

    if isinstance(flag, dict):
        m.a.require(
            flag.get("source_stratified") is True, "Malformed nested sensitivity"
        )
        return False
    m.a.require(type(flag) is bool, "Invalid resampling flag")
    return flag


def compare(left, right, records, expected, strict=False):
    allc = left + right
    ids = allc[0]["ids"]
    idx = allc[0]["macro_indices"]
    mask = np.logical_and.reduce([v["valid"] for v in allc])
    reference = {r["id"]: r for r in records}
    kept = [key for key, on in zip(ids, mask) if on]
    m.a.require(kept == expected["common_valid_ids"], "Independent paired mask differs")
    clusters = sorted({reference[k]["text_cluster"] for k in kept})
    loc = {v: i for i, v in enumerate(clusters)}
    grouped = np.zeros(
        (len(allc), len(clusters), len(allc[0]["labels"]), 3), dtype=np.int64
    )
    for item, key in enumerate(ids):
        if not mask[item]:
            continue
        for setting, v in enumerate(allc):
            grouped[setting, loc[reference[key]["text_cluster"]]] += v["counts"][item]
    score = macro(grouped.sum(axis=1), idx)
    delta = float(score[: len(left)].mean() - score[len(left) :].mean())
    m.a.require(
        abs(delta - expected["mean_difference"]) < 1e-12,
        "Independent F1 difference differs",
    )
    membership = defaultdict(set)
    for key in ids:
        membership[reference[key]["text_cluster"]].add(reference[key]["source"])
    strata = defaultdict(list)
    for i, cluster in enumerate(clusters):
        strata[
            (
                tuple(sorted(membership[cluster]))
                if stratified_record(expected)
                else ("all",)
            )
        ].append(i)
    generator = np.random.default_rng(expected["seed"])
    actual = []

    matrix = grouped.transpose(1, 0, 2, 3).reshape(len(clusters), -1).astype(float)
    for begin in range(0, expected["draws"], 64):
        size = min(64, expected["draws"] - begin)
        sample = np.concatenate(
            [
                generator.choice(group, (size, len(group)), replace=True)
                for _, group in sorted(strata.items())
            ],
            axis=1,
        )
        weights = np.array(
            [np.bincount(r, minlength=len(clusters)) for r in sample], dtype=float
        )
        totals = (weights @ matrix).reshape(size, len(allc), len(allc[0]["labels"]), 3)
        m.a.require(
            np.array_equal(totals[0], grouped[:, sample[0]].sum(axis=1)),
            "Independent direct bootstrap count mismatch",
        )
        values = macro(totals, idx)
        actual.extend(
            (
                values[:, : len(left)].mean(axis=1)
                - values[:, len(left) :].mean(axis=1)
            ).tolist()
        )
    saved = np.load(m.ROOT / expected["bootstrap_file"], allow_pickle=False)
    actual = np.asarray(actual)
    error = float(np.abs(saved - actual).max())
    m.a.require(
        saved.shape == (10000,) and error < 1e-12,
        "Saved bootstrap draws differ from independent calculation",
    )
    m.a.require(
        np.allclose(
            np.quantile(actual, [0.025, 0.975]), expected["ci95"], rtol=0, atol=1e-12
        ),
        "Independent confidence interval differs",
    )
    if expected["family_size"]:
        alpha = 0.05 / (2 * expected["family_size"])
        m.a.require(
            np.allclose(
                np.quantile(actual, [alpha, 1 - alpha]),
                expected["family_interval"],
                rtol=0,
                atol=1e-12,
            ),
            "Independent family interval differs",
        )
    return dict(draws=len(saved), maximum_absolute_difference=error, common=len(kept))


def run():
    with m.runtime.offline():
        groups, outputs, metadata = m.load()
        for path in ["metrics_receipt.json", "context_receipt.json"]:
            receipt = m.a.read(m.BASE / path)
            m.c.verify(receipt["outputs"])
            m.c.verify(receipt["sources"])
        result = m.a.read(m.BASE / "metrics.json")
        diagnostics = m.a.read(m.BASE / "diagnostics.json")
        numerical_checks = 0
        resamples = {}
        independent_runs = {}
        for name, endpoint in result["results"].items():
            phase, dim = name.split("/")
            rr = groups[phase]
            ep = endpoint["endpoint"]
            runs = {}
            for ident, output in sorted(outputs.items()):
                v = arithmetic(rr, output, ep)
                runs[ident] = v
                reported = endpoint["per_condition_repeat"][
                    ident[0] + "/repeat" + str(ident[1])
                ]
                total = v["counts"].sum(axis=0)
                valid = v["valid"]
                idx = v["macro_indices"]
                y = v["gold"]
                if idx:
                    m.a.require(
                        abs(float(macro(total, idx)) - reported["macro_f1_valid_only"])
                        < 1e-12,
                        "Independent point score differs",
                    )
                m.a.require(
                    total.tolist() == reported["per_label_counts"],
                    "Independent label confusion differs",
                )
                m.a.require(
                    abs(
                        float((v["pred"][valid] == y[valid]).all(axis=1).mean())
                        - reported["exact_set_accuracy"]
                    )
                    < 1e-12,
                    "Independent exact-set score differs",
                )
                low = total.copy()
                high = total.copy()
                missing = y[~valid]
                low[:, 1] += (1 - missing).sum(axis=0)
                low[:, 2] += missing.sum(axis=0)
                high[:, 0] += missing.sum(axis=0)
                if idx:
                    m.a.require(
                        np.allclose(
                            [float(macro(low, idx)), float(macro(high, idx))],
                            reported["invalid_inclusive_bounds"],
                            rtol=0,
                            atol=1e-12,
                        ),
                        "Failure bounds differ",
                    )
                numerical_checks += 1
            independent_runs[name] = runs
            for family, contrast in endpoint["contrasts"].items():
                lhs = [tuple(x) for x in contrast["left_settings"]]
                rhs = [tuple(x) for x in contrast["right_settings"]]
                left = [runs[k] for k in lhs]
                right = [runs[k] for k in rhs]
                resamples[name + "/" + family + "/main"] = compare(
                    left, right, rr, contrast
                )
                resamples[name + "/" + family + "/stratified"] = compare(
                    left, right, rr, contrast["source_stratified"]
                )
                if family in m.PAIRS[:2]:
                    strict = []
                    for k in lhs + rhs:
                        v = runs[k]
                        valid = v["valid"] & np.array(
                            [metadata[item, *k]["n_valid"] == 3 for item in v["ids"]]
                        )
                        counts = v["counts"].copy()
                        counts[~valid] = 0
                        strict.append(v | {"valid": valid, "counts": counts})
                    resamples[name + "/" + family + "/strict"] = compare(
                        strict[: len(lhs)],
                        strict[len(lhs) :],
                        rr,
                        endpoint["robustness"][family]["three_valid_votes"],
                    )
            print("Independently checked", name, flush=True)
        diagnostic_checks = 0
        for phase, rr in groups.items():
            for ident, output in sorted(outputs.items()):
                valid = [r for r in rr if output[r["id"]] is not None]
                exact = 0
                fp = fn = 0
                for r in valid:
                    fields = {
                        d: v for d in m.DIMENSIONS if (v := truth(r, d)) is not None
                    }
                    exact += all(
                        output[r["id"]][d] == (sorted(v) if isinstance(v, list) else v)
                        for d, v in fields.items()
                    )
                    fp += r["gate"] is False and output[r["id"]]["hate"] is True
                    fn += r["gate"] is True and output[r["id"]]["hate"] is False
                reported = diagnostics[phase]["per_run"][
                    ident[0] + "/repeat" + str(ident[1])
                ]["own"]
                m.a.require(
                    exact == reported["counts"].get("exact_known", 0)
                    and fp == reported["counts"].get("fp", 0)
                    and fn == reported["counts"].get("fn", 0),
                    "Independent answer/confusion counts differ",
                )
                diagnostic_checks += 1
        resources = m.a.read(m.BASE / "resources.json")
        tokens = Counter()
        accounts = Counter()
        for v in resources["by_population_condition"].values():
            tokens.update(v["usage"])
            accounts.update(v["accounting"])
        m.a.require(
            dict(tokens) == resources["observed_totals"]
            and dict(accounts) == resources["accounting"],
            "Grouped resource sums differ",
        )
        cases = m.a.read(m.BASE / "cases.json")["cases"]
        case_outcomes = 0
        for case in cases:
            if case["key"] is None:
                continue
            for out in case["outcomes"]:
                saved = metadata[case["key"], out["condition"], out["repeat"]]
                m.a.require(
                    out["prediction"] == saved["result"]
                    and out["result_sha256"] == saved["result_sha256"],
                    "Case disagrees with verified evidence",
                )
                m.a.require(
                    m.c.sha(m.ROOT / out["result_path"]) == out["result_sha256"],
                    "Case original changed",
                )
                case_outcomes += 1
        m.c.verify(m.a.read(m.BASE / "preservation_inputs.json"))
        m.save(
            m.BASE / "independent_verification.json",
            dict(
                status="PASS",
                point_score_and_label_confusion_checks=numerical_checks,
                diagnostic_checks=diagnostic_checks,
                bootstrap_checks=resamples,
                total_bootstrap_values=sum(v["draws"] for v in resamples.values()),
                case_outcomes_verified=case_outcomes,
                original_frozen_bindings_unchanged=True,
                evidence=m.bindings(
                    [
                        m.BASE / p
                        for p in [
                            "assembly.json",
                            "metrics_receipt.json",
                            "context_receipt.json",
                        ]
                    ]
                ),
                sources=m.bindings([Path(__file__)]),
                Qwen_calls=0,
                training_fits=0,
            ),
        )
        print(
            "Independent numerical, resource and preservation review PASS.", flush=True
        )


if __name__ == "__main__":
    run()

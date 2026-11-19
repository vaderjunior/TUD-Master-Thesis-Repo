from collections import Counter
from pathlib import Path
import time
import numpy as np
import yaml
from threadpoolctl import threadpool_limits
from analysis.classification.broad import metrics as m


def f1(tp, predicted, actual, indices):
    denominator = predicted + actual
    values = np.divide(
        2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator > 0
    )
    return values[..., indices].mean(axis=-1)


def percentiles(values, probabilities):
    ordered = sorted(float(value) for value in values)
    out = []
    for p in probabilities:
        position = (len(ordered) - 1) * p
        i = int(position)
        fraction = position - i
        out.append(
            ordered[i] + fraction * (ordered[min(i + 1, len(ordered) - 1)] - ordered[i])
        )
    return out


def encoded(rr, runs, ep, settings, taxonomy):
    dim = ep["dimension"]
    field = dict(
        hate="gate",
        hate_type="hate_types",
        target_group="target_groups",
        severity="severity",
        legal="legal",
    )[dim]
    labels = (
        ["not_hate", "hate"]
        if dim == "hate"
        else list(taxonomy["dimensions"][dim]["labels"])
    )
    eligible = [
        row
        for row in rr
        if row.get(field) is not None
        and (dim in ("hate", "legal") or row.get("gate") is True)
    ]
    ids = [row["key"] for row in eligible]
    m.require(
        ids == ep["scorable_ids"] and labels == ep["all_labels"],
        "Independent gold population/labels differ",
    )

    def positive(value):
        if dim == "hate":
            return {"hate" if value else "not_hate"}
        if dim == "severity":
            return set() if value is None else {value}
        return set(value)

    y = np.array(
        [[int(label in positive(row[field])) for label in labels] for row in eligible],
        dtype=np.int64,
    )
    p = np.zeros((len(settings), len(ids), len(labels)), dtype=np.int64)
    v = np.zeros((len(settings), len(ids)), dtype=bool)
    for s, ident in enumerate(settings):
        for i, key in enumerate(ids):
            answer = runs[ident][key]
            if answer is not None:
                v[s, i] = True
                p[s, i] = [int(label in positive(answer[dim])) for label in labels]
    fixed = np.flatnonzero(y.sum(axis=0) >= 10).tolist()
    m.require(
        [labels[i] for i in fixed] == ep["macro_labels"],
        "Independent fixed labels differ",
    )
    m.require(
        dict(zip(labels, y.sum(axis=0).tolist())) == ep["gold_support"],
        "Independent supports differ",
    )
    clusters = [
        (
            row["functional_cluster"]
            if "functional_cluster" in row and row["phase"] == "en_hatecheck_eval"
            else row["text_cluster"]
        )
        for row in eligible
    ]
    m.require(clusters == ep["row_clusters"], "Independent clusters differ")
    return y, p, v, fixed, clusters


def run():
    start = time.perf_counter()
    receipt = m.c.read(m.BASE / "scores_complete.json")
    m.r.verify(receipt["outputs"])
    m.r.verify(receipt["sources"])
    before = m.bindings(
        [m.BASE / "scores_complete.json", Path(__file__), Path(m.__file__)]
    )
    groups, _, runs, _, _ = m.reader.load()
    taxonomy = yaml.safe_load(
        (m.ROOT / "config/taxonomy.yaml").read_text(encoding="utf8")
    )
    checks = Counter()
    review = []
    max_error = 0.0

    def same(actual, expected, label):
        nonlocal max_error
        if actual is None or expected is None:
            m.require(actual is expected, "Null differs: " + label)
        else:
            a = np.asarray(actual)
            b = np.asarray(expected)
            m.require(a.shape == b.shape, "Shape differs: " + label)
            error = float(np.max(np.abs(a - b))) if a.size else 0.0
            max_error = max(max_error, error)
            m.require(error <= 2e-14, "Numerical difference: " + label)
        checks["numerical_checks"] += 1

    for item in receipt["endpoints"]:
        data = m.c.read(m.ROOT / item["path"])
        ep = data["endpoint"]
        settings = [tuple(v) for v in data["settings"]]
        phase, dim = item["endpoint"].split("/")
        y, p, v, idx, clusters = encoded(groups[phase], runs, ep, settings, taxonomy)
        with np.load(m.ROOT / data["encoded_path"], allow_pickle=False) as saved:
            m.require(
                np.array_equal(saved["gold"], y)
                and np.array_equal(saved["predicted"], p)
                and np.array_equal(saved["valid"], v),
                "Encoded arrays differ",
            )
            counts = np.stack(
                (
                    np.logical_and(y, p),
                    np.logical_and(1 - y, p),
                    np.logical_and(y, 1 - p),
                ),
                axis=-1,
            ).astype(np.int64)
            counts[~v] = 0
            m.require(
                np.array_equal(counts, saved["counts"]), "Item contributions differ"
            )
        checks["reconstructed_setting_endpoint_arrays"] += len(settings)
        own = {}
        condition_order = ep["available_conditions"]
        for cond in condition_order:
            own[cond] = []
            for rep in range(3):
                s = settings.index((cond, rep))
                observed = v[s]
                x = p[s, observed]
                actual = y[observed]
                tp = np.logical_and(actual, x).sum(axis=0)
                pred = x.sum(axis=0)
                act = actual.sum(axis=0)
                point = (
                    float(f1(tp, pred, act, idx)) if idx and observed.any() else None
                )
                unknown = y[~observed]
                all_actual = y.sum(axis=0)
                bounds = (
                    [
                        float(
                            f1(tp, pred + (1 - unknown).sum(axis=0), all_actual, idx)
                        ),
                        float(
                            f1(
                                tp + unknown.sum(axis=0),
                                pred + unknown.sum(axis=0),
                                all_actual,
                                idx,
                            )
                        ),
                    ]
                    if idx
                    else [None, None]
                )
                score = data["own_valid"][cond]["repeats"][rep]
                same(score["macro_f1_valid_only"], point, "own point")
                for i in (0, 1):
                    same(score["invalid_inclusive_bounds"][i], bounds[i], "own bound")
                same(score["coverage"], float(observed.mean()), "coverage")
                m.require(
                    score["n_scorable"] == len(y)
                    and score["n_valid"] == int(observed.sum())
                    and score["n_invalid"] == int((~observed).sum()),
                    "Coverage denominator differs",
                )
                own[cond].append(dict(point=point, bounds=bounds))
                checks["own_scores"] += 1
            values = [z["point"] for z in own[cond]]
            same(
                data["own_valid"][cond]["mean_macro_f1"],
                sum(values) / 3 if idx else None,
                "own repeat mean",
            )
            same(
                data["own_valid"][cond]["sample_sd"],
                float(np.std(values, ddof=1)) if idx else None,
                "own repeat SD",
            )
            for i in (0, 1):
                same(
                    data["own_valid"][cond]["mean_full_roster_failure_bounds"][i],
                    sum(z["bounds"][i] for z in own[cond]) / 3 if idx else None,
                    "mean bounds",
                )
        nine = [
            settings.index((cond, rep)) for cond in m.CONDITIONS for rep in range(3)
        ]
        common = v[nine].all(axis=0)
        shared = data["three_condition_common"]
        m.require(
            int(common.sum()) == shared["n_common_valid"]
            and [key for key, on in zip(ep["scorable_ids"], common) if not on]
            == shared["excluded_ids"],
            "Nine-setting mask differs",
        )
        for cond in m.CONDITIONS:
            values = []
            for rep in range(3):
                x = p[settings.index((cond, rep)), common]
                actual = y[common]
                values.append(
                    float(
                        f1(
                            np.logical_and(actual, x).sum(axis=0),
                            x.sum(axis=0),
                            actual.sum(axis=0),
                            idx,
                        )
                    )
                    if idx
                    else None
                )
            s = shared["per_condition"][cond]
            for i in range(3):
                same(s["repeat_macro_f1"][i], values[i], "nine-common repeat")
            same(
                s["mean_macro_f1"], sum(values) / 3 if idx else None, "nine-common mean"
            )
            same(
                s["sample_sd"],
                float(np.std(values, ddof=1)) if idx else None,
                "nine-common SD",
            )
        if idx:
            same(
                shared["small_minus_zero_descriptive"],
                shared["per_condition"]["rag_small"]["mean_macro_f1"]
                - shared["per_condition"]["zero_shot"]["mean_macro_f1"],
                "descriptive difference",
            )
        for comp in data["contrasts"]:
            selected = [
                settings.index((cond, rep))
                for cond in (comp["left"], comp["right"])
                for rep in range(3)
            ]
            common = v[selected].all(axis=0)
            m.require(
                comp["n_common_valid"] == int(common.sum())
                and comp["common_valid_ids"]
                == [key for key, on in zip(ep["scorable_ids"], common) if on],
                "Six-setting mask differs",
            )
            checks["paired_masks"] += 1
            if not idx:
                m.require(
                    comp["mean_difference"] is None
                    and comp["actual_draws"] == 0
                    and "draws_path" not in comp,
                    "Unsupported macro reported",
                )
                continue
            subset = p[selected][:, common]
            actual = y[common]
            tp = np.logical_and(subset, actual).astype(np.int64)
            stats = np.stack(
                (tp, subset, np.broadcast_to(actual, subset.shape)), axis=-1
            )
            total = stats.sum(axis=1)
            points = f1(total[..., 0], total[..., 1], total[..., 2], idx)
            same(comp["common_f1_by_setting"], points, "six repeat points")
            same(
                comp["mean_difference"],
                float(points[:3].mean() - points[3:].mean()),
                "paired point",
            )
            left, right = own[comp["left"]], own[comp["right"]]
            bounds = [
                sum(a["bounds"][0] for a in left) / 3
                - sum(a["bounds"][1] for a in right) / 3,
                sum(a["bounds"][1] for a in left) / 3
                - sum(a["bounds"][0] for a in right) / 3,
            ]
            same(
                comp["full_roster_failure_difference_bounds"],
                bounds,
                "difference bounds",
            )
            kept = np.array(clusters)[common].tolist()
            unique = sorted(set(kept))
            lookup = {key: i for i, key in enumerate(unique)}
            gidx = np.array([lookup[key] for key in kept])
            grouped = np.zeros(
                (len(unique), 6, len(ep["all_labels"]), 3), dtype=np.int64
            )
            np.add.at(grouped, gidx, stats.transpose(1, 0, 2, 3))
            flat = grouped.reshape(len(unique), -1).astype(np.float64)
            rng = np.random.default_rng(m.SEED)
            values = []
            for first in range(0, m.DRAWS, 64):
                size = min(64, m.DRAWS - first)
                picks = rng.integers(0, len(unique), (size, len(unique)))
                weights = np.array(
                    [np.bincount(row, minlength=len(unique)) for row in picks],
                    dtype=np.float64,
                )
                total = (weights @ flat).reshape(size, 6, len(ep["all_labels"]), 3)
                m.require(
                    np.array_equal(total, np.rint(total)),
                    "Independent sums not integer",
                )
                scores = f1(total[..., 0], total[..., 1], total[..., 2], idx)
                values.extend(
                    (scores[:, :3].mean(axis=1) - scores[:, 3:].mean(axis=1)).tolist()
                )
            saved = np.load(m.ROOT / comp["draws_path"], allow_pickle=False)
            same(saved, values, "every paired cluster draw")
            same(
                comp["ci95"], percentiles(values, [0.025, 0.975]), "pointwise interval"
            )
            if comp["family_size"]:
                same(
                    comp["family_interval"],
                    percentiles(
                        values,
                        [
                            0.05 / (2 * comp["family_size"]),
                            1 - 0.05 / (2 * comp["family_size"]),
                        ],
                    ),
                    "family interval",
                )
            m.require(
                comp["n_clusters"] == len(unique) and len(saved) == m.DRAWS,
                "Cluster/draw counts differ",
            )
            checks["bootstrap_values_replayed"] += len(saved)
            checks["scorable_contrasts"] += 1
        review.append(
            dict(
                endpoint=item["endpoint"],
                setting_scores=len(settings),
                comparisons=len(data["contrasts"]),
                status="passed",
            )
        )
        print("Independent review passed:", item["endpoint"], flush=True)
    m.require(
        checks["own_scores"] == 153
        and checks["bootstrap_values_replayed"] == 380000
        and checks["paired_masks"] == 40,
        "Incomplete independent review",
    )
    m.r.verify(receipt["outputs"])
    m.r.verify(receipt["sources"])
    m.r.verify(before)
    m.save(
        m.BASE / "independent_review.json",
        dict(
            status="all_arrays_points_masks_bounds_and_bootstrap_values_independently_reproduced",
            checks=dict(checks),
            endpoints=review,
            max_absolute_numerical_difference=max_error,
            tolerance=2e-14,
            elapsed_seconds=time.perf_counter() - start,
            sources=before,
            new_Qwen_calls=0,
            new_training=0,
            new_inference=0,
            method="Independent eligibility/bit encoding and TP/predicted/actual F1; grouped add.at, integer-index resampling and matrix multiply; sorted linear percentiles. No frozen scoring function invoked for review numerics.",
        ),
    )


if __name__ == "__main__":
    with m.r.original.offline(), threadpool_limits(limits=2):
        run()

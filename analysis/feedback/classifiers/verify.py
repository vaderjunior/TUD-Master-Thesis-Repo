from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
from analysis.feedback.classifiers import common as a
import numpy as np
from threadpoolctl import threadpool_limits


def direct_raw(z):
    l = z["linear_errors"]
    q = z["qwen_errors"]
    v = np.zeros((len(l), 4, 20), dtype=np.int64)
    for r in range(4):
        for s in range(4):
            v[:, r, s * 4] = 9 * l[:, s, r, 0]
            for p in range(1, 4):
                for stream in range(3):
                    v[:, r, s * 4 + p] += 3 * l[:, s, r, p + 3 * stream]
        for repeat in range(3):
            v[:, r, 16] += 3 * np.maximum(q[:, r, 0, repeat], 0)
        for p in range(1, 4):
            for stream in range(3):
                for repeat in range(3):
                    v[:, r, 16 + p] += np.maximum(q[:, r, p + 3 * stream, repeat], 0)
    return v


def transform(x):
    y = np.asarray(x)
    out = np.empty((*y.shape[:-1], 4, 3, 8), float)
    for s in range(4):
        for p in range(1, 4):
            b, u, qr, qu = y[..., s * 4], y[..., s * 4 + p], y[..., 16], y[..., 16 + p]
            for m, v in enumerate(
                [b, u, qr, qu, u - b, qu - qr, (u - b) - (qu - qr), u - qu]
            ):
                out[..., s, p - 1, m] = v
    return out


def main():
    a.contract()
    plan = a.read(a.BASE / "plan.json")
    for name in ["inputs", "outputs", "source_hashes"]:
        a.c.verify(plan[name])
    for name in ["assembled.json", "scored.json", "details_complete.json"]:
        value = a.read(a.OUT / name)
        a.c.verify(value["outputs"])
        if "inputs" in value:
            a.c.verify(value["inputs"])
    z = a.load("predictions.npz")
    raw = direct_raw(z)
    meta = list(a.rows(a.OUT / "metadata.jsonl"))
    analysis = a.read(a.OUT / "analysis.json")
    groups = analysis["groups"] + a.read(a.OUT / "source_intervals.json")
    scalars = 0
    exact = 0
    maxdiff = 0.0
    trajectory_count = 0
    for gi, g in enumerate(groups):
        ids = np.asarray(g["indices"])
        rounds = [r - 1 for r in g["rounds"]]
        vals = raw[ids][:, rounds].reshape(len(ids), -1)
        labels = sorted({meta[i]["inference_cluster"] for i in ids})
        size = []
        sums = []
        for label in labels:
            local = [
                j for j, i in enumerate(ids) if meta[i]["inference_cluster"] == label
            ]
            size.append(len(local))
            sums.append(vals[local].sum(0))
        sums = np.asarray(sums)
        size = np.asarray(size)
        a.need(
            labels == g["cluster_names"] and size.tolist() == g["cluster_sizes"],
            "Cluster membership mismatch",
        )
        if g["mode"] == "cluster_equal":
            num = sums / (9 * size[:, None])
            den = np.ones(len(size))
            point = num.sum(0) / len(size)
        else:
            num = sums
            den = 9 * size
            total = sums.sum(0)
            point = np.array([float(Fraction(int(x), 9 * len(ids))) for x in total])
        check = transform(point.reshape(len(rounds), 20))
        a.need(
            np.allclose(check, g["point"], rtol=0, atol=2e-15),
            "Independent point differs",
        )
        exact += check.size
        with np.load(a.OUT / g["draw_file"]) as d:
            saved = d["raw_draws"]
            a.need(
                np.allclose(
                    point.reshape(len(rounds), 20), d["raw_point"], rtol=0, atol=2e-15
                ),
                "Saved raw point differs",
            )
        rng = np.random.Generator(np.random.PCG64(g["seed"]))
        rebuilt = np.empty_like(saved)

        for first in range(0, 10000, 32):
            chosen = rng.integers(
                0, len(labels), size=(min(32, 10000 - first), len(labels))
            )
            for offset, draw in enumerate(chosen):
                counts = np.bincount(draw, minlength=len(labels))
                rebuilt[first + offset] = (counts @ num / (counts @ den)).reshape(
                    len(rounds), 20
                )
        diff = float(np.max(abs(saved - rebuilt)))
        maxdiff = max(maxdiff, diff)
        a.need(diff < 2e-15, "Independent resampling differs")
        scalars += saved.size
        expanded = transform(rebuilt)
        ci = np.moveaxis(np.quantile(expanded, [0.025, 0.975], axis=0), 0, -1)
        a.need(
            np.allclose(ci, g["ci95"], rtol=0, atol=3e-15), "Interval endpoints differ"
        )
        for trajectory in g.get("trajectories", []):
            before, after = trajectory["before"] - 1, trajectory["after"] - 1
            a.need(
                np.allclose(
                    check[after] - check[before],
                    trajectory["point"],
                    rtol=0,
                    atol=2e-15,
                ),
                "Trajectory point differs",
            )
            interval = np.moveaxis(
                np.quantile(
                    expanded[:, after] - expanded[:, before], [0.025, 0.975], axis=0
                ),
                0,
                -1,
            )
            a.need(
                np.allclose(interval, trajectory["ci95"], rtol=0, atol=3e-15),
                "Trajectory interval differs",
            )
            trajectory_count += np.asarray(trajectory["point"]).size
        if (gi + 1) % 10 == 0:
            print(f"Independent bootstrap readback: {gi+1}/{len(groups)}", flush=True)

    subgroup_count = 0
    for v in a.rows(a.OUT / "subgroups.jsonl"):
        field = {
            "source": "source",
            "function": "functionality",
            "template": "template_id",
        }[v["family"]]
        r = v["round"] - 1
        ids = [
            i
            for i, m in enumerate(meta)
            if m[field] is not None
            and str(m[field]) == v["label"]
            and m["population"] == v["population"]
            and m["gold_hate"] == (v["endpoint"] == "FNR")
            and (z["qwen_gates"][i, r] >= 0).all()
        ]
        a.need(len(ids) == v["inputs"], "Subgroup cohort differs")
        if ids:
            a.need(
                np.allclose(
                    transform(raw[ids, r].mean(0) / 9), v["point"], rtol=0, atol=2e-15
                ),
                "Subgroup point differs",
            )
            subgroup_count += np.asarray(v["point"]).size
        else:
            a.need(v["point"] is None, "Empty group reported as zero")

    quality = a.read(a.OUT / "binary_quality.json")
    quality_count = 0
    for v in quality:
        r = v["round"] - 1
        s = v["setting"]
        p = v["policy_index"]
        cols = [0] if p == 0 else [p, p + 3, p + 6]
        ids = [
            i
            for i, m in enumerate(meta)
            if m["population"] == v["population"]
            and (
                (z["qwen_gates"][i] >= 0).all()
                if v["cohort"] == "fixed"
                else (z["qwen_gates"][i, r] >= 0).all()
            )
        ]
        for method, arr, expected in [
            ("linear", z["linear_gates"][ids, s, r][:, cols], v["linear_confusions"]),
            (
                "qwen",
                z["qwen_gates"][ids, r][:, cols].reshape(len(ids), -1),
                v["qwen_confusions"],
            ),
        ]:
            metrics = []
            for col in range(arr.shape[1]):
                counts = Counter(
                    (bool(z["gold"][i]), bool(arr[j, col])) for j, i in enumerate(ids)
                )
                tn, fp, fn, tp = [
                    counts[t]
                    for t in [
                        (False, False),
                        (False, True),
                        (True, False),
                        (True, True),
                    ]
                ]
                a.need([tn, fp, fn, tp] == expected[col], "Binary confusion differs")
                div = lambda n, d: n / d if d else 0.0
                metrics.append(
                    dict(
                        macro_f1=(
                            div(2 * tp, 2 * tp + fp + fn)
                            + div(2 * tn, 2 * tn + fp + fn)
                        )
                        / 2,
                        balanced_accuracy=(div(tp, tp + fn) + div(tn, tn + fp)) / 2,
                        accuracy=(tp + tn) / len(ids),
                        precision=div(tp, tp + fp),
                        recall=div(tp, tp + fn),
                        predicted_hate=(tp + fp) / len(ids),
                    )
                )
            for key, value in v[method].items():
                a.need(
                    abs(sum(x[key] for x in metrics) / len(metrics) - value) < 1e-14,
                    "Binary averaging differs",
                )
                quality_count += 1

    execution = a.read(a.OUT / "execution.json")
    aliases = execution["aliases"]
    data = a.read(a.BASE / "data.json")
    saved = a.load("saved_model_replay.npz")
    source_index = {s: i for i, s in enumerate(data["candidate_ids"])}
    ledger = list(a.rows(a.OUT / "source_adoption_ledger.jsonl"))
    for v in ledger:
        variant, t = a.SETTINGS[v["setting"]]
        r = v["round"]
        p = v["policy"]
        stream = v["stream"]
        ix = source_index[v["source_id"]]
        keys = [
            variant + "/reference",
            variant + "/" + ("reference" if r == 1 else f"r{r-1}_{p}_s{stream}"),
            f"{variant}/r{r}_{p}_s{stream}",
        ]
        for label, key in zip(["baseline", "before", "after"], keys):
            value = float(saved["candidate_probabilities"][aliases[key], ix])
            a.need(
                value == v[label + "_probability"]
                and ((value > t) != v["gold"]) == v[label + "_error"],
                "Source adoption reconstruction differs",
            )
    fixed = (z["qwen_gates"] >= 0).all((1, 2, 3))
    for v in a.read(a.OUT / "durability.json"):
        ids = [
            i
            for i, m in enumerate(meta)
            if m["population"] == v["population"]
            and m["gold_hate"] == (v["endpoint"] == "FNR")
            and (v["cohort"] == "full_sample" or fixed[i])
        ]
        p = a.POLICIES.index(v["policy"]) + 1
        s = v["setting"]
        patterns = Counter()
        for i in ids:
            for stream in range(3):
                patterns[
                    str(int(z["linear_errors"][i, s, 0, 0]))
                    + "".join(
                        str(int(z["linear_errors"][i, s, r, p + 3 * stream]))
                        for r in range(4)
                    )
                ] += 1
        a.need(dict(patterns) == v["pattern_counts"], "Path patterns differ")
        for r in range(4):
            a.need(
                sum(
                    n
                    for path, n in patterns.items()
                    if path[r] == "1" and path[r + 1] == "0"
                )
                == v["adjacent_fixes"][r],
                "Fix partition differs",
            )
            a.need(
                sum(
                    n
                    for path, n in patterns.items()
                    if path[r] == "0" and path[r + 1] == "1"
                )
                == v["adjacent_new_errors"][r],
                "New-error partition differs",
            )

    with np.load(a.REVIEW / "part7/execution_arrays.npz") as ex:
        vg = ex["vote_gates"]
    for v in a.read(a.OUT / "bounds.json"):
        ids = [
            i
            for i, m in enumerate(meta)
            if m["population"] == v["population"]
            and m["gold_hate"] == (v["endpoint"] == "FNR")
        ]
        r = v["round"] - 1
        bounds = np.asarray(v["intervals"])
        for s in range(4):
            b = float(z["linear_errors"][ids, s, r, 0].mean())
            for p in range(1, 4):
                u = float(z["linear_errors"][ids, s, r][:, [p, p + 3, p + 6]].mean())
                qbound = []
                for cols in [[0], [p, p + 3, p + 6]]:
                    low = high = 0
                    n = len(ids) * len(cols) * 3
                    for i in ids:
                        for col in cols:
                            for repeat in range(3):
                                e = int(z["qwen_errors"][i, r, col, repeat])
                                valid = int(z["qwen_votes"][i, r, col, repeat])
                                if v["scenario"] == "terminal_null":
                                    lo, hi = (0, 1) if e < 0 else (e, e)
                                elif v["scenario"] == "any_partial_unknown":
                                    lo, hi = (0, 1) if valid < 3 else (e, e)
                                else:
                                    votes = vg[i, r, col, repeat]
                                    yes = int((votes == 1).sum())
                                    unknown = int((votes < 0).sum())
                                    possible = {
                                        (yes + j) >= 2 for j in range(unknown + 1)
                                    }
                                    errors = {
                                        int(x != bool(z["gold"][i])) for x in possible
                                    }
                                    lo, hi = min(errors), max(errors)
                                low += lo
                                high += hi
                    qbound.append((low / n, high / n))
                (rl, rh), (ul, uh) = qbound
                delta = u - b
                expected = np.array(
                    [
                        [b, b],
                        [u, u],
                        [rl, rh],
                        [ul, uh],
                        [delta, delta],
                        [ul - rh, uh - rl],
                        [delta - (uh - rl), delta - (ul - rh)],
                        [u - uh, u - ul],
                    ]
                )
                a.need(
                    np.allclose(expected, bounds[s, p - 1], rtol=0, atol=2e-15),
                    "Missingness bound differs",
                )
    resources = a.read(a.OUT / "resources.json")
    new = [j for j in execution["jobs"] if j["new"]]
    a.need(
        resources["new_fits"] == len(new) == 64
        and resources["new_fit_seconds"]
        == sum(j["attribution"]["fit_seconds"] for j in new),
        "Resource sum differs",
    )
    a.c.verify(a.read(a.OUT / "analysis_contract.json")["inputs"])
    a.save(
        "independent_verification.json",
        dict(
            status="passed",
            groups=len(groups),
            exact_point_scalars=exact,
            bootstrap_raw_scalars=scalars,
            maximum_resampling_discrepancy=maxdiff,
            trajectory_scalars=trajectory_count,
            subgroup_scalars=subgroup_count,
            binary_metric_means=quality_count,
            source_memberships_checked=len(ledger),
            missing_bound_groups_checked=len(a.read(a.OUT / "bounds.json")),
            all_model_probability_replay=a.read(a.OUT / "execution.json")[
                "independently_replayed_probabilities"
            ],
            original_protocol_and_sources_preserved=True,
            inputs=a.bindings(
                [
                    a.OUT / n
                    for n in [
                        "scored.json",
                        "details_complete.json",
                        "assembled.json",
                        "tests.json",
                    ]
                ]
            ),
            new_fits=0,
            new_Qwen_calls=0,
        ),
    )
    print(
        "Independent numerical, bound, source, path and resource verification passed.",
        flush=True,
    )


if __name__ == "__main__":
    with a.c.offline(), threadpool_limits(limits=1):
        main()

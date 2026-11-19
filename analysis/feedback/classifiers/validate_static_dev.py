from collections import defaultdict
from fractions import Fraction
import hashlib
import json
import math

from analysis.feedback.classifiers import static_dev_results as io


def estimates(a):
    a, b, c, d = a
    return dict(zip(io.ESTIMATES, [a, b, c, d, b - a, d - c, b - a - d + c, b - d]))


def close(a, b):
    io.need(abs(float(a) - float(b)) < 1e-12, f"Numerical mismatch: {a} vs {b}")


def independently_select(cases, result, excluded):
    truth = result["endpoint"] == "FNR"
    use_exclusions = "excluded" in result["view"]
    eligible = [
        r
        for r in cases
        if r["gold"]["hate"] == truth
        and (not use_exclusions or r["key"] not in excluded)
    ]
    if result["view"].startswith("representative"):
        groups = defaultdict(list)
        for r in eligible:
            groups[r["cluster"]].append(r)
        chosen = {
            min(
                g,
                key=lambda r: hashlib.sha256(
                    ("A15-representative|" + r["key"]).encode()
                ).hexdigest(),
            )["key"]
            for g in groups.values()
        }
        eligible = [r for r in eligible if r["key"] in chosen]
    strict = result["view"].startswith("strict")
    pools = [result["group"] + "_s" + str(s) for s in io.SEEDS]
    conditions = ["reference5"] + [result["mode"] + "__" + p for p in pools]
    per_input = {}
    bounds = [
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
        Fraction(0),
    ]
    missing = [0, 0]
    for r in eligible:
        v, cut = result["variant"], result["cutoff"]
        lb = int((r["linear"][v]["reference"] > cut) != truth)
        lu = Fraction(sum((r["linear"][v][p] > cut) != truth for p in pools), 3)
        rr = []
        for condition in conditions:
            p = r["rag"][condition]
            rr.append(
                [
                    None if h is None or (strict and n != 3) else int(h != truth)
                    for h, n in zip(p["hate"], p["n_valid"])
                ]
            )
        arms = [rr[0], rr[1] + rr[2] + rr[3]]
        lows, highs = [], []
        for j, arm in enumerate(arms):
            low = Fraction(sum(x for x in arm if x is not None), len(arm))
            high = low + Fraction(arm.count(None), len(arm))
            missing[j] += arm.count(None)
            lows.append(low)
            highs.append(high)
        bounds[0] += lows[1] - highs[0]
        bounds[1] += highs[1] - lows[0]
        bounds[2] += lu - lb - highs[1] + lows[0]
        bounds[3] += lu - lb - lows[1] + highs[0]
        bounds[4] += lu - highs[1]
        bounds[5] += lu - lows[1]
        if all(None not in arm for arm in arms):
            per_input[r["key"]] = [Fraction(lb), lu, lows[0], lows[1]]
    return (
        eligible,
        per_input,
        [b / len(eligible) for b in bounds],
        missing,
        pools,
        conditions,
    )


def main():
    import numpy as np
    from threadpoolctl import threadpool_limits

    io.need(
        not (io.OUT / "independent_review.json").exists(),
        "Preserve completed independent review",
    )
    with io.offline(), threadpool_limits(limits=1):
        original = io.load()
        cases = list(io.rows(io.OUT / "all_DEV_cases.jsonl"))
        result = io.read(io.OUT / "analysis.json")["results"]
        diag = {r["id"]: r for r in io.read(io.OUT / "diagnostics.json")}
        bykey = {r["key"]: r for r in cases}
        checks = 0
        for i, c in enumerate(cases):
            for field in original["roster"][i]:
                io.need(c[field] == original["roster"][i][field], "Case roster changed")
            for v, dd in c["linear"].items():
                for pool, value in dd.items():
                    io.need(
                        value == original["linear"][v][pool][i],
                        "Case probability changed",
                    )
                    checks += 1
            for j, condition in enumerate(original["conditions"]):
                p = c["rag"][condition]
                io.need(
                    p["hate"]
                    == [None if h < 0 else bool(h) for h in original["rag"][i, j]],
                    "Case RAG decision differs",
                )
                io.need(
                    p["n_valid"] == original["votes"][i, j].tolist()
                    and p["stage"] == original["stages"][i, j].tolist(),
                    "Case metadata differs",
                )
                checks += 3
        cache = {}
        records = []
        bootstrap_inputs = defaultdict(dict)
        diagnostic_checks = 0
        for r in result:
            eligible, points, bounds, miss, pools, conditions = independently_select(
                cases, r, set(original["science"]["mandatory_excluded_keys"])
            )
            keys = list(points)
            io.need(
                keys == r["common_ids"] and len(eligible) == r["eligible_n"],
                "Independent cohort",
            )
            io.need(
                [x["key"] for x in eligible if x["key"] not in points]
                == r["invalid_ids"],
                "Independent invalid set",
            )
            io.need(
                len({bykey[k]["cluster"] for k in keys}) == r["clusters"],
                "Cluster denominator",
            )
            totals = [
                sum((p[j] for p in points.values()), Fraction(0)) for j in range(4)
            ]
            io.need(
                [int(t * 9) for t in totals] == r["integer_arm_totals_ninths"],
                "Integer arm counts",
            )
            ex = estimates([t / len(keys) for t in totals])
            for k in io.ESTIMATES:
                close(ex[k], r["estimates"][k])
            for j, k in enumerate(
                ("rag_change", "difference_in_change", "updated_absolute_difference")
            ):
                for b, saved in zip(bounds[2 * j : 2 * j + 2], r["missing_bounds"][k]):
                    close(b, saved)
            io.need(
                miss == r["missing_bounds"]["missing_real_outputs"],
                "Real output missingness",
            )
            checks += 8 + 6 + 5
            cache[r["id"]] = points
            records.append(
                {
                    "id": r["id"],
                    "common_n": len(keys),
                    "exact_estimates": {k: str(v) for k, v in ex.items()},
                }
            )
            if "draws_path" in r:
                bootstrap_inputs[r["draws_path"]][r["draws_column_start"]] = (r, points)
            if r["id"] in diag:
                saved = diag[r["id"]]
                for source, s in saved["sources"].items():
                    kk = [k for k in keys if bykey[k]["source"] == source]
                    io.need(s["n"] == len(kk), "Source denominator")
                    if not kk:
                        io.need(s["estimates"] is None, "Undefined empty source")
                    else:
                        e = estimates(
                            [
                                sum((points[k][j] for k in kk), Fraction(0)) / len(kk)
                                for j in range(4)
                            ]
                        )
                        for field in io.ESTIMATES:
                            close(e[field], s["estimates"][field])
                            diagnostic_checks += 1
                independently_pooled = []
                truth = r["endpoint"] == "FNR"
                v, cut = r["variant"], r["cutoff"]
                for j, saved_pool in enumerate(saved["pools"]):
                    sums = [0, 0, 0, 0]
                    fixes = regressions = 0
                    for k in keys:
                        c = bykey[k]
                        lb = (c["linear"][v]["reference"] > cut) != truth
                        lu = (c["linear"][v][pools[j]] > cut) != truth
                        sums[0] += lb
                        sums[1] += lu
                        sums[2] += sum(
                            h != truth for h in c["rag"]["reference5"]["hate"]
                        )
                        sums[3] += sum(
                            h != truth for h in c["rag"][conditions[j + 1]]["hate"]
                        )
                        fixes += lb and not lu
                        regressions += not lb and lu
                    e = estimates(
                        [
                            Fraction(sums[0], len(keys)),
                            Fraction(sums[1], len(keys)),
                            Fraction(sums[2], 3 * len(keys)),
                            Fraction(sums[3], 3 * len(keys)),
                        ]
                    )
                    independently_pooled.append(e)
                    io.need(
                        fixes == saved_pool["classifier_fixes"]
                        and regressions == saved_pool["classifier_regressions"],
                        "Fix/regression counts",
                    )
                    for field in io.ESTIMATES:
                        close(e[field], saved_pool["estimates"][field])
                        diagnostic_checks += 1
                for j, s in enumerate(saved["leave_one_pool"]):
                    for field in io.ESTIMATES:
                        close(
                            sum(
                                e[field]
                                for i, e in enumerate(independently_pooled)
                                if i != j
                            )
                            / 2,
                            s["estimates"][field],
                        )
                        diagnostic_checks += 1
                for s in saved["leave_one_batch"]:
                    kk = [
                        k
                        for k in keys
                        if all(
                            s["omitted_batch"] not in bykey[k]["rag"][c]["stage"]
                            for c in conditions
                        )
                    ]
                    io.need(s["n"] == len(kk), "Omitted-batch denominator")
                    if not kk:
                        io.need(
                            s["estimates"] is None, "Empty omission must be undefined"
                        )
                    else:
                        e = estimates(
                            [
                                sum((points[k][j] for k in kk), Fraction(0)) / len(kk)
                                for j in range(4)
                            ]
                        )
                        for field in io.ESTIMATES:
                            close(e[field], s["estimates"][field])
                            diagnostic_checks += 1
        bootstrap_checks = 0
        for path, entries in bootstrap_inputs.items():
            ordered = [entries[i] for i in (0, 4, 8, 12)]
            r, points = ordered[0]
            keys = list(points)
            io.need(
                all(list(p) == keys for _, p in ordered), "Settings use different masks"
            )
            labels = sorted({bykey[k]["cluster"] for k in keys})
            grouped = [[k for k in keys if bykey[k]["cluster"] == c] for c in labels]

            integer = np.array(
                [
                    [
                        int(sum((p[k][j] * 9 for k in g), Fraction(0)))
                        for _, p in ordered
                        for j in range(4)
                    ]
                    for g in grouped
                ],
                dtype=np.int64,
            )
            sizes = np.array([len(g) for g in grouped], dtype=np.int64)
            raw = np.load(io.physical(io.ROOT / path), allow_pickle=False)
            io.need(raw.shape == (10000, 16), "Bootstrap matrix shape")
            name = "/".join((r["mode"], r["group"], r["view"], r["endpoint"]))
            seed = int(
                hashlib.sha256(("A18-DEV-v1|" + name).encode()).hexdigest()[:16], 16
            )
            io.need(seed == r["seed"], "Independent seed derivation")
            rng = np.random.Generator(np.random.PCG64(seed))
            check = np.empty_like(raw)
            for start in range(0, 10000, 100):
                sampled = rng.integers(0, len(labels), size=(100, len(labels)))
                multiplicities = np.array(
                    [np.bincount(row, minlength=len(labels)) for row in sampled]
                )
                check[start : start + 100] = (multiplicities @ integer) / (
                    9 * (multiplicities @ sizes)[:, None]
                )
            io.need(np.array_equal(raw, check), "Independent bootstrap differs")
            bootstrap_checks += raw.size
            for offset, (r, _) in enumerate(ordered):
                columns = check[:, offset * 4 : offset * 4 + 4]
                independent = np.column_stack(
                    [
                        columns,
                        columns[:, 1] - columns[:, 0],
                        columns[:, 3] - columns[:, 2],
                        (columns[:, 1] - columns[:, 0])
                        - (columns[:, 3] - columns[:, 2]),
                        columns[:, 1] - columns[:, 3],
                    ]
                )
                sorted_columns = np.sort(independent, axis=0)
                for field, col in zip(io.ESTIMATES, sorted_columns.T):
                    qq = []
                    for q in (0.025, 0.975):
                        position = (len(col) - 1) * q
                        left = math.floor(position)
                        qq.append(
                            col[left]
                            + (position - left)
                            * (col[min(left + 1, len(col) - 1)] - col[left])
                        )
                    for actual, expected in zip(qq, r["interval95"][field]):
                        close(actual, expected)
                        checks += 1
            print("Independent cluster draws verified", name, flush=True)

        io.verify(io.read(io.OUT / "evidence_bindings.json"))
        io.need(
            io.running_plan()
            == io.read(io.OUT / "analysis_plan.json")["running_dependencies_before"],
            "Running source preservation",
        )
        io.new(io.OUT / "exact_point_checks.json", records)
        io.new(
            io.OUT / "independent_review.json",
            {
                "status": "all_supporting_DEV_points_bounds_diagnostics_and_draws_independently_verified",
                "finished_utc": io.utc(),
                "results": len(result),
                "original_DEV_cases": len(cases),
                "point_alignment_interval_checks": checks,
                "diagnostic_scalar_checks": diagnostic_checks,
                "bootstrap_files": len(bootstrap_inputs),
                "bootstrap_scalars_reconstructed": bootstrap_checks,
                "method": "Original bound arrays/assembly matched to case ledger; independent Fraction counts "
                "and masks, multiplicity-dot bootstrap, manual percentile interpolation.",
                "unresolved_discrepancies": [],
                "new_calls": 0,
                "new_fits": 0,
                "ongoing_transfer_predictions_read": 0,
                "running_sources_unchanged": True,
                "reviewed_sources": {
                    io.rel(p): io.sha(p)
                    for p in (
                        io.OUT / "analysis_plan.json",
                        io.OUT / "analysis.json",
                        io.OUT / "diagnostics.json",
                        io.OUT / "all_DEV_cases.jsonl",
                        io.OUT / "fit_resources.json",
                        io.OUT / "evidence_bindings.json",
                    )
                },
                "code_bindings": {
                    io.rel(__file__): io.sha(__file__),
                    io.rel(io.__file__): io.sha(io.__file__),
                },
            },
        )
        print(
            json.dumps(
                {
                    "status": "independent_review_passed",
                    "results": len(result),
                    "bootstrap_scalars": bootstrap_checks,
                    "diagnostic_checks": diagnostic_checks,
                }
            )
        )


if __name__ == "__main__":
    main()

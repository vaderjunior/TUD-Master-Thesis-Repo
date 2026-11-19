from collections import defaultdict
import hashlib
from pathlib import Path
import socket
from unittest.mock import patch
import numpy as np
from analysis.feedback.balanced import archive as r


def equal(a, b):
    if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
        r.require(
            np.allclose(a, b, rtol=0, atol=2e-12, equal_nan=False),
            "Numeric list mismatch",
        )
    elif a is None or b is None:
        r.require(a is b, "Unavailable value mismatch")
    else:
        r.require(abs(a - b) < 2e-12, "Numeric value mismatch")


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def divide(a, b):
    return np.divide(
        a,
        b,
        out=np.zeros(np.broadcast_shapes(np.shape(a), np.shape(b)), dtype=float),
        where=b != 0,
    )


def primitives(ids, run, records, roster, kind, missing=None):
    counts = np.zeros((2 if kind == "binary" else 7, 3), dtype=np.int64)
    errors = 0
    nulls = []
    n = 0
    gates = [0, 0, 0, 0]
    for key in ids:
        gold = roster[key]["gold"]
        p = records[key, *run]["result"]
        if p is None:
            nulls.append(key)
            if missing is None:
                continue
        n += 1
        if kind in ("gate", "binary"):
            answer = (
                p["hate"]
                if p is not None
                else gold["hate"] if missing == "best" else not gold["hate"]
            )
            errors += answer != gold["hate"]
            if gold["hate"] and answer:
                gates[0] += 1
            elif not gold["hate"] and answer:
                gates[1] += 1
            elif gold["hate"] and not answer:
                gates[2] += 1
            else:
                gates[3] += 1
            yy = {int(gold["hate"])}
            xx = {int(answer)}
        else:
            yy = {r.LABELS.index(x) for x in gold["hate_type"]}
            xx = (
                {r.LABELS.index(x) for x in p["hate_type"]}
                if p is not None
                else yy if missing == "best" else set(range(7)) - yy
            )
        if kind != "gate":
            for label in range(len(counts)):
                counts[label, 0] += label in yy and label in xx
                counts[label, 1] += label not in yy and label in xx
                counts[label, 2] += label in yy and label not in xx
    score = float(np.mean([f1(*cell) for cell in counts]))
    tp, fp, fn, tn = gates
    ba = (tp / (tp + fn) + tn / (tn + fp)) / 2 if tp + fn and tn + fp else None
    return dict(
        n=n,
        nulls=nulls,
        counts=counts,
        errors=errors,
        error_rate=errors / n if n else None,
        macro_f1=score,
        balanced_accuracy=ba,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


def cluster_samples(ids, roster, seed, stratified):
    groups = defaultdict(list)
    for key in ids:
        groups[roster[key]["cluster"]].append(key)
    names = sorted(groups)
    rng = np.random.Generator(np.random.PCG64(seed))
    n = len(names)
    if stratified:
        bins = defaultdict(list)
        for i, name in enumerate(names):
            bins[tuple(sorted({roster[k]["source"] for k in groups[name]}))].append(i)
        selected = []
        for source, positions in sorted(bins.items()):
            positions = np.array(positions)
            selected.append(
                positions[rng.integers(0, len(positions), size=(10000, len(positions)))]
            )
        picks = np.concatenate(selected, axis=1)
    else:
        picks = rng.integers(0, n, size=(10000, n))
    multiplicities = np.zeros((10000, n), dtype=float)
    for row, sample in enumerate(picks):
        values, counts = np.unique(sample, return_counts=True)
        multiplicities[row, values] = counts
    return [groups[name] for name in names], multiplicities, picks


def main():
    done = r.read(r.DEST / "scoring_complete.json")
    r.verify(done["inputs"])
    r.verify(done["outputs"])
    r.require(
        not (r.DEST / "scoring_independent_check.json").exists(),
        "Preserve completed independent check",
    )
    roster = {v["key"]: v for v in r.rows(r.EXEC.parent / "roster.jsonl")}
    records = {
        (v["key"], v["condition"], v["repeat"]): v
        for v in r.rows(r.REVIEW / "predictions.jsonl")
    }
    r.require(len(records) == 38475, "Incomplete result identity table")
    count = defaultdict(int)
    all_ids = sorted(roster)
    representatives = {}
    for key in all_ids:
        cluster = roster[key]["cluster"]
        h = hashlib.sha256(
            ("20261010-balanced-representative|" + key).encode()
        ).hexdigest()
        if cluster not in representatives or h < representatives[cluster][0]:
            representatives[cluster] = (h, key)
    reps = {key for _, key in representatives.values()}
    for name in sorted(done["outputs"]):
        path = r.ROOT / name
        value = r.read(path)
        if path.name == "supporting_deletions.json":
            continue
        kind = (
            "gate"
            if value["endpoint"] in ("false_positive_rate", "false_negative_rate")
            else "binary" if value["endpoint"] == "combined_binary" else "types"
        )
        groups = value["condition_groups"]
        runs = [(c, rep) for cs in groups.values() for c in cs for rep in range(3)]
        original = value["original_ids"]
        expected = all_ids
        if kind == "types" or value["endpoint"] == "false_negative_rate":
            expected = [k for k in expected if roster[k]["gold"]["hate"]]
        elif kind == "gate":
            expected = [k for k in expected if not roster[k]["gold"]["hate"]]
        if path.stem.startswith("source_") and path.stem != "source_stratified":
            expected = [
                k
                for k in expected
                if roster[k]["source"] == path.stem.removeprefix("source_")
            ]
        if path.stem == "representatives":
            expected = [k for k in expected if k in reps]
        r.require(original == expected, "Original scoring cohort differs: " + name)
        strict = value["mask"] == "all-three-valid-votes"
        common = [
            key
            for key in original
            if all(
                records[key, *run]["result"] is not None
                and (not strict or records[key, *run]["n_valid"] == 3)
                for run in runs
            )
        ]
        r.require(
            common == value["common_ids"]
            and sorted(set(original) - set(common)) == value["excluded_ids"],
            "Incorrect common-mask membership",
        )
        equal(len(common), value["common_n"])
        equal(len(original), value["original_n"])
        count["common_masks"] += 1
        own = {}
        points_by_run = {}
        bounds_by_run = {}
        for run in runs:
            key = f"{run[0]}_rep{run[1]}"
            saved = value["settings"][key]
            observed = {
                scope: primitives(ids, run, records, roster, kind, missing)
                for scope, ids, missing in [
                    ("own_valid", original, None),
                    ("common_valid", common, None),
                    ("full_roster_worst", original, "worst"),
                    ("full_roster_best", original, "best"),
                ]
            }
            if kind == "gate":
                r.require(
                    saved["null_ids"] == observed["own_valid"]["nulls"],
                    "Gate null roster differs",
                )
                for a, b in [
                    ("own_valid_n", observed["own_valid"]["n"]),
                    ("own_errors", observed["own_valid"]["errors"]),
                    ("own_error_rate", observed["own_valid"]["error_rate"]),
                    ("common_n", len(common)),
                    ("common_errors", observed["common_valid"]["errors"]),
                    ("common_error_rate", observed["common_valid"]["error_rate"]),
                ]:
                    equal(saved[a], b)
                equal(
                    saved["full_roster_bounds"],
                    [
                        observed["full_roster_best"]["error_rate"],
                        observed["full_roster_worst"]["error_rate"],
                    ],
                )
            else:
                for scope, computed in observed.items():
                    saved_scope = saved[scope]
                    equal(saved_scope["n"], computed["n"])
                    equal(saved_scope["macro_f1"], computed["macro_f1"])
                    r.require(
                        saved_scope["null_ids"] == computed["nulls"],
                        "Metric null roster differs",
                    )
                    if kind == "types":
                        equal(
                            saved_scope["counts_tp_fp_fn"], computed["counts"].tolist()
                        )
                    else:
                        for k in ["tp", "fp", "fn", "tn", "balanced_accuracy"]:
                            equal(saved_scope[k], computed[k])
            points_by_run[run] = observed["common_valid"]
            bounds_by_run[run] = observed
            count["independent_run_metrics"] += 4
        clustered, weights, picks = cluster_samples(
            common,
            roster,
            value["bootstrap"]["seed"],
            value["bootstrap"].get("stratification") == "source-set",
        )
        equal(len(clustered), value["cluster_count"])
        if kind == "gate":
            sums = np.array(
                [
                    [
                        primitives(ids, run, records, roster, "gate")["errors"]
                        for run in runs
                    ]
                    for ids in clustered
                ]
            )
            den = weights @ np.array([len(ids) for ids in clustered])
            samples = {"error_rate": (weights @ sums) / den[:, None]}
        else:
            sums = np.array(
                [
                    [
                        primitives(ids, run, records, roster, kind)["counts"]
                        for run in runs
                    ]
                    for ids in clustered
                ]
            )
            sample_counts = np.einsum("bc,cklt->bklt", weights, sums, optimize=True)
            samples = {
                "macro_f1": divide(
                    2 * sample_counts[..., 0],
                    2 * sample_counts[..., 0]
                    + sample_counts[..., 1]
                    + sample_counts[..., 2],
                ).mean(axis=-1)
            }
            if kind == "binary":
                samples["balanced_accuracy"] = divide(
                    sample_counts[..., 0], sample_counts[..., 0] + sample_counts[..., 2]
                ).mean(axis=-1)
        for metric, draws in samples.items():
            saved_metric = value["endpoints"][metric] if kind == "binary" else value
            points = {}
            boots = {}
            bounds = {}
            for policy, conditions in groups.items():
                these = [run for run in runs if run[0] in conditions]
                indices = [runs.index(run) for run in these]
                points[policy] = float(
                    np.mean([points_by_run[run][metric] for run in these])
                )
                boots[policy] = draws[:, indices].mean(axis=1)
                order = ["best", "worst"] if kind == "gate" else ["worst", "best"]
                bounds[policy] = [
                    float(
                        np.mean(
                            [
                                bounds_by_run[run]["full_roster_" + m][metric]
                                for run in these
                            ]
                        )
                    )
                    for m in order
                ]
                row = saved_metric["groups"][policy]
                point_name = (
                    "error_rate"
                    if kind == "gate"
                    else "macro_f1" if kind == "types" else "mean"
                )
                equal(row[point_name], points[policy])
                if "ci95" in row:
                    equal(
                        row["ci95"], np.quantile(boots[policy], [0.025, 0.975]).tolist()
                    )
                bound_name = (
                    "full_roster_bounds"
                    if kind == "gate"
                    else "full_roster_missing_bounds"
                )
                equal(row[bound_name], bounds[policy])
                vals = [
                    [points_by_run[c, rep][metric] for rep in range(3)]
                    for c in conditions
                ]
                equal(row["decoder_values"], vals)
                count["policy_aggregates"] += 1
            for key, comp in saved_metric["contrasts"].items():
                left, right = key.split("_minus_")
                diff = boots[left] - boots[right]
                equal(comp["difference"], points[left] - points[right])
                equal(comp["ci95"], np.quantile(diff, [0.025, 0.975]).tolist())
                alpha = (1 - comp["interval_coverage"]) / 2
                equal(
                    comp["family_interval"],
                    np.quantile(diff, [alpha, 1 - alpha]).tolist(),
                )
                equal(
                    comp["full_roster_missing_bounds"],
                    [
                        bounds[left][0] - bounds[right][1],
                        bounds[left][1] - bounds[right][0],
                    ],
                )
                count["contrasts_and_intervals"] += 1
        print("Independently checked", path.relative_to(r.DEST), flush=True)
    deletions = r.read(r.DEST / "supporting_deletions.json")
    for endpoint, sections in deletions.items():
        primary = r.read(r.DEST / endpoint / "primary.json")
        ids = primary["common_ids"]
        for kind, rows in sections.items():
            if kind == "scope":
                continue
            for item in rows:
                selected = [
                    key
                    for key in ids
                    if "omitted_stage" not in item
                    or records[key, "reference", 0]["stage"] != item["omitted_stage"]
                ]
                for policy, conditions in primary["condition_groups"].items():
                    conditions = [
                        c
                        for c in conditions
                        if "omitted_pool_seed" not in item
                        or not c.endswith(str(item["omitted_pool_seed"]))
                    ]
                    rates = [
                        primitives(selected, (c, rep), records, roster, "gate")[
                            "error_rate"
                        ]
                        for c in conditions
                        for rep in range(3)
                    ]
                    equal(item["rates"][policy], float(np.mean(rates)))
                equal(
                    item["difference"],
                    item["rates"]["error_balanced"] - item["rates"]["error_positive"],
                )
                count["deletion_cases"] += 1
    r.verify(done["outputs"])
    r.save(
        r.DEST / "scoring_independent_check.json",
        dict(
            status="all_counts_masks_aggregates_and_intervals_independently_verified",
            created_utc=r.utc(),
            inputs={
                r.rel(r.DEST / "scoring_complete.json"): r.sha(
                    r.DEST / "scoring_complete.json"
                ),
                **done["outputs"],
                r.rel(Path(__file__)): r.sha(Path(__file__)),
            },
            counts=dict(count),
            new_model_calls=0,
            scope="Separate explicit confusion/set reconstruction; all original/cohort masks and full-roster bounds; independent cluster-draw assembly and every reported contrast interval.",
        ),
    )
    print(dict(count), flush=True)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    with (
        patch.object(socket.socket, "connect", r.forbid),
        patch.object(socket, "create_connection", r.forbid),
        threadpool_limits(limits=2),
    ):
        main()

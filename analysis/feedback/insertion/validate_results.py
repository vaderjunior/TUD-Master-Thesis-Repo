from collections import defaultdict
import hashlib
from pathlib import Path
import socket
from unittest.mock import patch
import numpy as np
from analysis.feedback.insertion import archive as r
from analysis.feedback.balanced import validate_scores as old

equal, primitives, cluster_samples, divide = (
    old.equal,
    old.primitives,
    old.cluster_samples,
    old.divide,
)


def primary_check(value, roster, records, science, excluded):
    n = 0
    for endpoint in science["primary"]:
        truth = endpoint["gold_hate"]
        eligible = [
            k
            for k, v in roster.items()
            if v["gold"]["hate"] == truth and k not in excluded
        ]
        for suffix, strict, representative in [
            ("primary", False, False),
            ("all_three_votes", True, False),
            ("one_per_cluster", False, True),
        ]:
            saved = value["primary"][endpoint["name"] + "/" + suffix]
            keys = list(eligible)
            if representative:
                clusters = defaultdict(list)
                for k in keys:
                    clusters[roster[k]["cluster"]].append(k)
                selected = {
                    min(
                        v,
                        key=lambda k: hashlib.sha256(
                            ("A14-representative|" + k).encode()
                        ).hexdigest(),
                    )
                    for v in clusters.values()
                }
                keys = [k for k in keys if k in selected]
            runs = [
                [(f"error_balanced_s{s}", rep) for s in r.SEEDS for rep in range(3)],
                [("reference", rep) for rep in range(3)],
            ]
            common = []
            means = []
            bounds = []
            for key in keys:
                flags = []
                errors = []
                limits = []
                for arm in runs:
                    values = [records[key, *run] for run in arm]
                    valid = [
                        v["result"] is not None and (not strict or v["n_valid"] == 3)
                        for v in values
                    ]
                    e = [
                        int(v["result"]["hate"] != truth) if ok else 0
                        for v, ok in zip(values, valid)
                    ]
                    flags.extend(valid)
                    errors.append(sum(e) / len(arm))
                    limits.append(
                        [sum(e) / len(arm), (sum(e) + valid.count(False)) / len(arm)]
                    )
                bounds.append(limits)
                if all(flags):
                    common.append(key)
                    means.append(errors)
            r.require(
                saved["common_ids"] == common
                and saved["excluded_ids"] == sorted(set(keys) - set(common)),
                "Frozen primary mask differs",
            )
            equal(saved["original_n"], len(keys))
            equal(saved["common_n"], len(common))
            groups, mult, picks = cluster_samples(
                common, roster, endpoint["bootstrap_seed"], False
            )
            by_key = dict(zip(common, means))
            sums = np.asarray(
                [
                    [sum(by_key[k][a] for k in group) for a in range(2)]
                    for group in groups
                ]
            )
            rates = (mult @ sums) / (
                mult @ np.asarray([len(group) for group in groups])
            )[:, None]
            point = np.mean(means, axis=0)
            difference = rates[:, 0] - rates[:, 1]
            equal(saved["balanced_error_rate"], point[0])
            equal(saved["reference_error_rate"], point[1])
            equal(saved["difference"], point[0] - point[1])
            equal(saved["ci975"], np.quantile(difference, [0.0125, 0.9875]))
            b = np.mean(bounds, axis=0)
            equal(
                saved["full_roster_difference_missing_bounds"],
                [b[0, 0] - b[1, 1], b[0, 1] - b[1, 0]],
            )
            equal(saved["clusters"], len(groups))
            r.require(
                saved["shared_reference_repeats"] == 3, "Reference pseudoreplicated"
            )
            n += 1
    for item in value["all_runs"]:
        ids = [k for k in roster if k not in excluded]
        run = (item["condition"], item["repeat"])
        observed = primitives(ids, run, records, roster, "binary")
        for name in ["tp", "fp", "fn", "tn"]:
            equal(item[name], observed[name])
        equal(item["inputs"], len(ids))
        equal(item["nulls"], len(observed["nulls"]))
        equal(item["macro_f1_own_valid"], observed["macro_f1"])
        equal(
            item["false_positive_rate_own_valid"],
            observed["fp"] / (observed["fp"] + observed["tn"]),
        )
        equal(
            item["false_negative_rate_own_valid"],
            observed["fn"] / (observed["fn"] + observed["tp"]),
        )
    return n


def main():
    done = r.read(r.DEST / "scoring_complete.json")
    r.verify(done["inputs"])
    r.verify(done["outputs"])
    target = r.DEST / "scoring_independent_check.json"
    if target.exists():
        r.verify(r.read(target)["inputs"])
        print("Existing independent score check verified.")
        return
    roster = {v["key"]: v for v in r.rows(r.EXEC.parent / "roster.jsonl")}
    records = {
        (v["key"], v["condition"], v["repeat"]): v
        for v in r.rows(r.REVIEW / "predictions.jsonl")
    }
    r.require(len(records) == 38475, "Incomplete scientific matrix")
    science = r.read(r.EXEC.parent / "scientific_plan_v1.json")
    excluded = r.read(r.EXEC.parent / "cold_replay_review_v1/sensitivity_plan.json")[
        "excluded_keys"
    ]
    count = defaultdict(int)
    for scope, omit in [("full", []), ("exclude_replay", excluded)]:
        v = r.read(r.DEST / scope / "primary.json")
        count["primary_masks_rates_bounds_intervals"] += primary_check(
            v, roster, records, science, omit
        )
    for name in sorted(done["outputs"]):
        path = r.ROOT / name
        if path.name in ["primary.json", "supporting_deletions.json"]:
            continue
        value = r.read(path)
        scope = path.relative_to(r.DEST).parts[0]
        kind = (
            "gate"
            if value["endpoint"] in ("false_positive_rate", "false_negative_rate")
            else "binary" if value["endpoint"] == "combined_binary" else "types"
        )
        groups = value["condition_groups"]
        runs = [(c, rep) for cs in groups.values() for c in cs for rep in range(3)]
        expected = [k for k in sorted(roster) if scope == "full" or k not in excluded]
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
            clusters = defaultdict(list)
            for k in expected:
                clusters[roster[k]["cluster"]].append(k)
            reps = {
                min(
                    v,
                    key=lambda k: hashlib.sha256(
                        ("A14-representative|" + k).encode()
                    ).hexdigest(),
                )
                for v in clusters.values()
            }
            expected = [k for k in expected if k in reps]
        original = value["original_ids"]
        r.require(original == expected, "Original cohort differs: " + name)
        strict = value["mask"] == "all-three-valid-votes"
        common = [
            k
            for k in original
            if all(
                records[k, *run]["result"] is not None
                and (not strict or records[k, *run]["n_valid"] == 3)
                for run in runs
            )
        ]
        r.require(
            common == value["common_ids"]
            and sorted(set(original) - set(common)) == value["excluded_ids"],
            "Common mask differs",
        )
        equal(len(common), value["common_n"])
        equal(len(original), value["original_n"])
        count["supporting_common_masks"] += 1
        points_by_run = {}
        bounds_by_run = {}
        for run in runs:
            saved = value["settings"][f"{run[0]}_rep{run[1]}"]
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
                for label, computed in observed.items():
                    sv = saved[label]
                    equal(sv["n"], computed["n"])
                    equal(sv["macro_f1"], computed["macro_f1"])
                    r.require(
                        sv["null_ids"] == computed["nulls"],
                        "Metric null roster differs",
                    )
                    if kind == "types":
                        counts = computed["counts"]
                        equal(sv["counts_tp_fp_fn"], counts.tolist())
                        equal(
                            sv["per_label_f1"],
                            divide(
                                2 * counts[:, 0],
                                2 * counts[:, 0] + counts[:, 1] + counts[:, 2],
                            ),
                        )
                        equal(
                            sv["precision"],
                            divide(counts[:, 0], counts[:, 0] + counts[:, 1]),
                        )
                        equal(
                            sv["recall"],
                            divide(counts[:, 0], counts[:, 0] + counts[:, 2]),
                        )
                        equal(sv["support"], counts[:, 0] + counts[:, 2])
                        chosen = common if label == "common_valid" else original
                        sets = []
                        for case in chosen:
                            gold = set(roster[case]["gold"]["hate_type"])
                            pred = records[case, *run]["result"]
                            if pred is None and label in ["own_valid", "common_valid"]:
                                continue
                            answer = (
                                set(pred["hate_type"])
                                if pred is not None
                                else (
                                    gold
                                    if label == "full_roster_best"
                                    else set(r.LABELS) - gold
                                )
                            )
                            sets.append((gold, answer))
                        equal(
                            sv["exact_match"],
                            sum(gold == pred for gold, pred in sets) / len(sets),
                        )
                        equal(
                            sv["contains_all_gold"],
                            sum(gold <= pred for gold, pred in sets) / len(sets),
                        )
                        equal(
                            sv["predicted_labels"],
                            sum(len(pred) for gold, pred in sets) / len(sets),
                        )
                    else:
                        for k in ["tp", "fp", "fn", "tn", "balanced_accuracy"]:
                            equal(sv[k], computed[k])
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
            samples = {
                "error_rate": (weights @ sums)
                / (weights @ np.array([len(ids) for ids in clustered]))[:, None]
            }
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
            cc = np.einsum("bc,cklt->bklt", weights, sums, optimize=True)
            samples = {
                "macro_f1": divide(
                    2 * cc[..., 0], 2 * cc[..., 0] + cc[..., 1] + cc[..., 2]
                ).mean(axis=-1)
            }
            if kind == "binary":
                samples["balanced_accuracy"] = divide(
                    cc[..., 0], cc[..., 0] + cc[..., 2]
                ).mean(axis=-1)
        for metric, draws in samples.items():
            saved_metric = value["endpoints"][metric] if kind == "binary" else value
            points = {}
            boots = {}
            bounds = {}
            for policy, conditions in groups.items():
                these = [run for run in runs if run[0] in conditions]
                ix = [runs.index(run) for run in these]
                points[policy] = float(
                    np.mean([points_by_run[run][metric] for run in these])
                )
                boots[policy] = draws[:, ix].mean(axis=1)
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
                    equal(row["ci95"], np.quantile(boots[policy], [0.025, 0.975]))
                equal(
                    row[
                        (
                            "full_roster_bounds"
                            if kind == "gate"
                            else "full_roster_missing_bounds"
                        )
                    ],
                    bounds[policy],
                )
                equal(
                    row["decoder_values"],
                    [
                        [points_by_run[c, rep][metric] for rep in range(3)]
                        for c in conditions
                    ],
                )
                count["policy_aggregates"] += 1
            for key, comp in saved_metric["contrasts"].items():
                left, right = key.split("_minus_")
                diff = boots[left] - boots[right]
                equal(comp["difference"], points[left] - points[right])
                equal(comp["ci95"], np.quantile(diff, [0.025, 0.975]))
                alpha = (1 - comp["interval_coverage"]) / 2
                equal(comp["family_interval"], np.quantile(diff, [alpha, 1 - alpha]))
                equal(
                    comp["full_roster_missing_bounds"],
                    [
                        bounds[left][0] - bounds[right][1],
                        bounds[left][1] - bounds[right][0],
                    ],
                )
                count["contrasts_and_intervals"] += 1
        print("Independently checked", path.relative_to(r.DEST), flush=True)
    for scope in ["full", "exclude_replay"]:
        prim = r.read(r.DEST / scope / "primary.json")["primary"]
        for endpoint, sections in r.read(
            r.DEST / scope / "supporting_deletions.json"
        ).items():
            ids = prim[endpoint + "/primary"]["common_ids"]
            for kind, items in sections.items():
                if kind == "scope":
                    continue
                for item in items:
                    selected = [
                        k
                        for k in ids
                        if "omitted_stage" not in item
                        or records[k, "reference", 0]["stage"] != item["omitted_stage"]
                    ]
                    seeds = [s for s in r.SEEDS if s != item.get("omitted_pool_seed")]
                    for policy, conditions in {
                        "error_balanced": [f"error_balanced_s{s}" for s in seeds],
                        "reference": ["reference"],
                    }.items():
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
                        item["rates"]["error_balanced"] - item["rates"]["reference"],
                    )
                    count["deletion_cases"] += 1
    r.verify(done["outputs"])
    r.save(
        target,
        dict(
            status="all_counts_masks_aggregates_and_intervals_independently_verified",
            created_utc=r.utc(),
            inputs={
                r.rel(r.DEST / "scoring_complete.json"): r.sha(
                    r.DEST / "scoring_complete.json"
                ),
                **done["outputs"],
                r.rel(Path(__file__)): r.sha(Path(__file__)),
                r.rel(Path(old.__file__)): r.sha(Path(old.__file__)),
            },
            counts=dict(count),
            new_model_calls=0,
        ),
    )
    print(dict(count), flush=True)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    with (
        patch.object(socket.socket, "connect", r.forbid),
        patch.object(socket, "create_connection", r.forbid),
        threadpool_limits(limits=1),
    ):
        main()

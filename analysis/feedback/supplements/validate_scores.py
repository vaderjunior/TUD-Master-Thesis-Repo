from collections import defaultdict
import hashlib
import statistics
import socket
from unittest.mock import patch
import numpy as np
from analysis.feedback.supplements.archive import ROOT
from analysis.feedback.supplements.archive import DEST
from analysis.feedback.supplements.archive import REVIEW
from analysis.feedback.supplements.archive import STUDIES
from analysis.feedback.supplements.archive import LABELS
from analysis.feedback.supplements.archive import SEEDS
from analysis.feedback.supplements.archive import read
from analysis.feedback.supplements.archive import rows
from analysis.feedback.supplements.archive import sha
from analysis.feedback.supplements.archive import rel
from analysis.feedback.supplements.archive import save
from analysis.feedback.supplements.archive import verify
from analysis.feedback.supplements.archive import require
from analysis.feedback.supplements.archive import utc
from analysis.feedback.supplements.archive import forbid


def eq(a, b):
    if a is None or b is None:
        require(a is b, "None/value discrepancy")
    else:
        require(
            np.allclose(a, b, rtol=0, atol=2e-12),
            "Independent numerical discrepancy: " + str(a) + " vs " + str(b),
        )


def rat(n, d):
    return n / d if d else 0.0


def main():
    receipt = read(DEST / "scoring_complete.json")
    verify(receipt["inputs"])
    verify(receipt["outputs"])
    roster = {
        s: {r["key"]: r for r in rows(p / "roster.jsonl")} for s, p in STUDIES.items()
    }
    records = {
        (r["study"], r["condition"], r["repeat"], r["key"]): r
        for r in rows(REVIEW / "predictions.jsonl")
    }
    checks = 0
    masks = 0
    interval_checks = 0
    verified = {}

    def single(study, c, rep, ids, missing=None, mode="types"):
        stats = [[0, 0, 0] for _ in LABELS]
        other = [[[0, 0, 0] for _ in LABELS] for _ in LABELS]
        total = contains = exact = length = 0
        null = []
        for i in ids:
            gold = set(roster[study][i]["gold"]["hate_type"] or [])
            v = records[study, c, rep, i]["result"]
            if v is None:
                null.append(i)
                if missing is None:
                    continue
                predicted = (set(LABELS) - gold) if missing == "worst" else gold
            else:
                predicted = set(v["hate_type"])
            total += 1
            contains += gold <= predicted
            exact += gold == predicted
            length += len(predicted)
            target = roster[study][i].get("target_label")
            for n, l in enumerate(LABELS):
                if mode != "target" or l == target:
                    stats[n][0] += l in gold and l in predicted
                    stats[n][1] += l in gold
                    stats[n][2] += l in predicted
                if mode == "target":
                    v = other[LABELS.index(target)][n]
                    v[0] += l in gold and l in predicted
                    v[1] += l in gold
                    v[2] += l in predicted
        per = [rat(2 * t, g + p) for t, g, p in stats]
        out = {
            "n": total,
            "null_ids": null,
            "counts_tp_fp_fn": [[t, p - t, g - t] for t, g, p in stats],
            "precision": [rat(t, p) for t, g, p in stats],
            "recall": [rat(t, g) for t, g, p in stats],
            "support": [g for t, g, p in stats],
        }
        if mode == "target":
            other_f1 = [
                statistics.mean(
                    rat(2 * t, g + p) for k, (t, g, p) in enumerate(v) if k != n
                )
                for n, v in enumerate(other)
            ]
            out.update(
                macro_target_f1=statistics.mean(per),
                per_category_f1=per,
                other_six_f1_by_category=other_f1,
                other_six_macro_f1=statistics.mean(other_f1),
            )
        else:
            out.update(
                macro_f1=statistics.mean(per),
                per_label_f1=per,
                predicted_labels=length / total if total else None,
                contains_all_gold=contains / total if total else None,
                exact_match=exact / total if total else None,
            )
        return out

    for name, h in receipt["outputs"].items():
        value = read(ROOT / name)
        study = value["study"]
        endpoint = (ROOT / name).stem
        is_gate = value["endpoint"].startswith("false_")
        hateful = not endpoint.startswith("benign")
        expected = [i for i, r in roster[study].items() if r["gold"]["hate"] == hateful]
        if endpoint.startswith("benign_"):
            expected = [
                i for i in expected if roster[study][i]["source"] == endpoint[7:]
            ]
        if endpoint == "representatives":
            cluster = defaultdict(set)
            for i in expected:
                cluster[roster[study][i]["cluster"]].add(
                    roster[study][i].get("original_key", i)
                )
            chosen = {
                min(
                    v,
                    key=lambda x: hashlib.sha256(
                        ("20261006-additions-representative|" + x).encode()
                    ).hexdigest(),
                )
                for v in cluster.values()
            }
            expected = [
                i for i in expected if roster[study][i].get("original_key", i) in chosen
            ]
        expected = sorted(expected)
        require(value["original_ids"] == expected, "Endpoint population changed")
        if study == "factorial":
            groups = {c: [c] for c in ["E0D0", "E1D0", "E0D1", "E1D1"]}
        elif endpoint in ["primary", "all_three_valid", "representatives"]:
            groups = {p: [f"{p}_s{s}" for s in SEEDS] for p in ["plain", "human"]}
        else:
            groups = {
                "reference": ["reference"],
                **{
                    p: [f"{p}_s{s}" for s in SEEDS]
                    for p in ["plain", "human", "generic"]
                },
            }
        keys = [(c, r) for cs in groups.values() for c in cs for r in range(3)]
        require(
            set(value["settings"]) == {f"{c}_rep{r}" for c, r in keys},
            "Setting matrix mismatch",
        )
        strict = endpoint == "all_three_valid"
        common = [
            i
            for i in expected
            if all(
                records[study, c, r, i]["result"] is not None
                and (not strict or records[study, c, r, i]["n_valid"] == 3)
                for c, r in keys
            )
        ]
        require(
            common == value["common_ids"]
            and len(common) == value["common_n"]
            and len(expected) == value["original_n"],
            "Common mask mismatch",
        )
        masks += 1
        points = {}
        bounds = {}
        for setting, v in value["settings"].items():
            c, rep = setting.rsplit("_rep", 1)
            rep = int(rep)
            if is_gate:
                valid = [
                    i
                    for i in expected
                    if records[study, c, rep, i]["result"] is not None
                ]
                errors = sum(
                    records[study, c, rep, i]["result"]["hate"]
                    != roster[study][i]["gold"]["hate"]
                    for i in valid
                )
                matched = sum(
                    records[study, c, rep, i]["result"]["hate"]
                    != roster[study][i]["gold"]["hate"]
                    for i in common
                )
                eq(v["own_valid_n"], len(valid))
                eq(v["own_errors"], errors)
                eq(v["own_error_rate"], errors / len(valid) if valid else None)
                eq(v["common_n"], len(common))
                eq(v["common_errors"], matched)
                eq(v["common_error_rate"], matched / len(common))
                eq(
                    v["full_roster_bounds"],
                    [
                        errors / len(expected),
                        (errors + len(expected) - len(valid)) / len(expected),
                    ],
                )
                require(
                    v["null_ids"] == [i for i in expected if i not in valid],
                    "Null identity mismatch",
                )
                checks += 1
            else:
                for kind, ids, impute in [
                    ("own_valid", expected, None),
                    ("common_valid", common, None),
                    ("full_roster_worst", expected, "worst"),
                    ("full_roster_best", expected, "best"),
                ]:
                    independently = single(
                        study,
                        c,
                        rep,
                        ids,
                        impute,
                        "target" if study == "factorial" else "types",
                    )
                    require(
                        set(independently) == set(v[kind]), "Unexpected metric fields"
                    )
                    for k, z in independently.items():
                        if k == "null_ids":
                            require(z == v[kind][k], "Null identities changed")
                        else:
                            eq(z, v[kind][k])
                    checks += 1
        scalar = "macro_target_f1" if study == "factorial" else "macro_f1"
        for group, cs in groups.items():
            ss = [value["settings"][f"{c}_rep{r}"] for c in cs for r in range(3)]
            points[group] = statistics.mean(
                v["common_error_rate"] if is_gate else v["common_valid"][scalar]
                for v in ss
            )
            bounds[group] = [
                statistics.mean(
                    (
                        v["full_roster_bounds"][b]
                        if is_gate
                        else v["full_roster_" + ["worst", "best"][b]][scalar]
                    )
                    for v in ss
                )
                for b in range(2)
            ]
            eq(
                value["groups"][group]["error_rate" if is_gate else scalar],
                points[group],
            )
            eq(
                value["groups"][group][
                    "full_roster_bounds" if is_gate else "full_roster_missing_bounds"
                ],
                bounds[group],
            )
            if not is_gate:
                field = "per_category_f1" if study == "factorial" else "per_label_f1"
                eq(
                    value["groups"][group][field],
                    np.mean([v["common_valid"][field] for v in ss], axis=0),
                )
                if study != "factorial":
                    pools = [
                        statistics.mean(
                            value["settings"][f"{c}_rep{r}"]["common_valid"][scalar]
                            for r in range(3)
                        )
                        for c in cs
                    ]
                    eq(value["groups"][group]["pool_means"], pools)
                    eq(
                        value["groups"][group]["pool_sample_sd"],
                        statistics.stdev(pools) if len(pools) > 1 else None,
                    )

        clusters = sorted({roster[study][i]["cluster"] for i in common})
        index = {c: n for n, c in enumerate(clusters)}
        seed = value["bootstrap"]["seed"]
        draws = 10000
        picks = np.random.Generator(np.random.PCG64(seed)).integers(
            0, len(clusters), size=(draws, len(clusters))
        )
        multiplicities = np.array(
            [np.bincount(p, minlength=len(clusters)) for p in picks], dtype=float
        )
        if endpoint in ["primary", "all_three_valid", "representatives"]:
            require(
                seed == (202610041 if study == "factorial" else 202610042),
                "Frozen primary seed changed",
            )
        group_draws = {}
        for group, cs in groups.items():
            run_draws = []
            for c in cs:
                for repeat in range(3):
                    count = np.zeros((len(clusters), 1 if is_gate else 7, 2), dtype=int)
                    for i in common:
                        row = records[study, c, repeat, i]["result"]
                        where = index[roster[study][i]["cluster"]]
                        if is_gate:
                            count[where, 0, 0] += (
                                row["hate"] != roster[study][i]["gold"]["hate"]
                            )
                            count[where, 0, 1] += 1
                        else:
                            labels = (
                                [roster[study][i]["target_label"]]
                                if study == "factorial"
                                else LABELS
                            )
                            gold = set(roster[study][i]["gold"]["hate_type"] or [])
                            predicted = set(row["hate_type"])
                            for l in labels:
                                k = LABELS.index(l)
                                count[where, k, 0] += l in gold and l in predicted
                                count[where, k, 1] += (l in gold) + (l in predicted)
                    summed = (
                        multiplicities @ count.reshape(len(clusters), -1)
                    ).reshape(draws, count.shape[1], 2)
                    den = summed[:, :, 1]
                    fraction = np.divide(
                        (1 if is_gate else 2) * summed[:, :, 0],
                        den,
                        out=np.zeros(den.shape),
                        where=den != 0,
                    )
                    run_draws.append(fraction.mean(axis=1))
            group_draws[group] = np.mean(run_draws, axis=0)
        if study == "factorial" and not is_gate:
            terms = {
                "marginal_example_block_effect": {
                    "E1D0": 0.5,
                    "E0D0": -0.5,
                    "E1D1": 0.5,
                    "E0D1": -0.5,
                },
                "marginal_definition_block_effect": {
                    "E0D1": 0.5,
                    "E0D0": -0.5,
                    "E1D1": 0.5,
                    "E1D0": -0.5,
                },
                "joint_restoration": {"E1D1": 1, "E0D0": -1},
                "interaction": {"E1D1": 1, "E1D0": -1, "E0D1": -1, "E0D0": 1},
                "examples_when_definitions_withdrawn": {"E1D0": 1, "E0D0": -1},
                "examples_when_definitions_full": {"E1D1": 1, "E0D1": -1},
                "definitions_when_examples_withdrawn": {"E0D1": 1, "E0D0": -1},
                "definitions_when_examples_full": {"E1D1": 1, "E1D0": -1},
            }
            observed = value["effects"]
        else:
            terms = {
                l + "_minus_" + r: {l: 1, r: -1}
                for l in groups
                for r in groups
                if l != r
            }
            observed = value["contrasts"]
        for contrast, t in terms.items():
            out = observed[contrast]
            point = sum(weight * points[group] for group, weight in t.items())
            eq(out["difference"], point)
            bs = sum(weight * group_draws[group] for group, weight in t.items())
            eq(out["ci95"], np.quantile(bs, [0.025, 0.975]))
            coverage = 0.975 if contrast.startswith("marginal_") else 0.95
            eq(out["interval_coverage"], coverage)
            eq(
                out["family_interval"],
                np.quantile(bs, [(1 - coverage) / 2, 1 - (1 - coverage) / 2]),
            )
            eq(
                out["full_roster_missing_bounds"],
                [
                    sum(
                        weight * bounds[group][0 if weight > 0 else 1]
                        for group, weight in t.items()
                    ),
                    sum(
                        weight * bounds[group][1 if weight > 0 else 0]
                        for group, weight in t.items()
                    ),
                ],
            )
            interval_checks += 1
        if is_gate:
            for g in groups:
                eq(
                    value["groups"][g]["ci95"],
                    np.quantile(group_draws[g], [0.025, 0.975]),
                )
                interval_checks += 1
        verified[name] = {
            "common_n": len(common),
            "clusters": len(clusters),
            "draws": 10000,
            "seed": seed,
        }
        print("Independent endpoint verified: " + study + "/" + endpoint, flush=True)
    save(
        DEST / "independent_verification.json",
        {
            "status": "passed",
            "created_utc": utc(),
            "scoring_complete_sha256": sha(DEST / "scoring_complete.json"),
            "source_sha256": sha(__file__),
            "run_metric_checks": checks,
            "common_masks_checked": masks,
            "interval_checks": interval_checks,
            "endpoints": verified,
            "method": "Independent Python sets/counts and support-plus-prediction F1; all saved masks, metrics, means, null bounds and 10000-draw intervals rebuilt without importing the scorer.",
            "new_model_calls": 0,
        },
    )


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket, "create_connection", forbid
    ), threadpool_limits(limits=2):
        main()

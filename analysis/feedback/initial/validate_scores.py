from collections import Counter, defaultdict
import hashlib
import statistics
import socket
from unittest.mock import patch
import numpy as np
from analysis.feedback.initial.archive import ROOT
from analysis.feedback.initial.archive import DEST
from analysis.feedback.initial.archive import REVIEW
from analysis.feedback.initial.archive import STUDIES
from analysis.feedback.initial.archive import LABELS
from analysis.feedback.initial.archive import read
from analysis.feedback.initial.archive import rows
from analysis.feedback.initial.archive import sha
from analysis.feedback.initial.archive import save
from analysis.feedback.initial.archive import rel
from analysis.feedback.initial.archive import verify
from analysis.feedback.initial.archive import require
from analysis.feedback.initial.archive import utc
from analysis.feedback.initial.archive import forbid


def eq(a, b):
    require(np.allclose(a, b, rtol=0, atol=2e-12), f"Independent mismatch: {a} vs {b}")


def main():
    done = read(DEST / "scoring_complete.json")
    verify(done["inputs"])
    verify(done["outputs"])
    runs = defaultdict(dict)
    rosters = {
        s: {r["id"]: r for r in rows(p / "roster.jsonl")} for s, p in STUDIES.items()
    }
    for r in rows(REVIEW / "predictions.jsonl"):
        key = (r["study"], r["condition"], r["repeat"])
        require(r["id"] not in runs[key], "Duplicate export")
        require(
            r["gold"] == rosters[r["study"]][r["id"]]["gold"], "Export gold differs"
        )
        runs[key][r["id"]] = r
    metric_checks = 0
    mask_checks = 0
    primary_bootstraps = {}

    def run_metric(study, condition, rep, ids, missing=None):
        run = runs[study, condition, rep]
        labels = set(LABELS)
        allowed = (
            labels - {condition.split("_removed_")[1]}
            if "_removed_" in condition
            else labels
        )
        counters = {l: Counter() for l in LABELS}
        n = contains = exact = misses = length = 0
        null = []
        for i in ids:
            gold = set(rosters[study][i]["gold"]["hate_type"])
            p = run[i]["result"]
            if p is None:
                null.append(i)
                if missing is None:
                    continue
                predicted = ((labels - gold) if missing == "worst" else gold) & allowed
            else:
                predicted = set(p["hate_type"])
            n += 1
            contains += gold <= predicted
            exact += gold == predicted
            length += len(predicted)
            if p:
                misses += not p["hate"]
            for l in LABELS:
                counters[l]["support"] += l in gold
                counters[l]["predicted"] += l in predicted
                counters[l]["tp"] += l in gold and l in predicted
        per = []
        c = []
        prec = []
        rec = []
        for l in LABELS:
            v = counters[l]
            den = v["support"] + v["predicted"]
            per.append(2 * v["tp"] / den if den else 0.0)
            c.append([v["tp"], v["predicted"] - v["tp"], v["support"] - v["tp"]])
            prec.append(v["tp"] / v["predicted"] if v["predicted"] else 0.0)
            rec.append(v["tp"] / v["support"] if v["support"] else 0.0)
        return {
            "valid": n,
            "null_ids": null,
            "macro_f1": statistics.mean(per),
            "per_label_f1": per,
            "label_counts": c,
            "precision": prec,
            "recall": rec,
            "support": [counters[l]["support"] for l in LABELS],
            "contains": contains / n,
            "exact": exact / n,
            "predicted_labels": length / n,
            "missed_hate_rate": misses / n,
        }

    def members(group, settings):
        cs = sorted({k.rsplit("_rep", 1)[0] for k in settings})
        return [c for c in cs if c == group or c.startswith(group + "_s20260928")]

    for path in done["outputs"]:
        value = read(ROOT / path)
        study = (ROOT / path).parent.name
        roster = rosters[study]
        endpoint = (ROOT / path).stem
        settings = value["settings"]
        keys = [(n.rsplit("_rep", 1)[0], int(n.rsplit("_rep", 1)[1])) for n in settings]
        original = sorted(set(value["common_valid_ids"]) | set(value["excluded_ids"]))
        if study == "explanations":
            science = read(STUDIES[study] / "scientific_plan_v1.json")
            name = (
                "benign"
                if endpoint.startswith("benign")
                else (
                    endpoint
                    if endpoint in ["original_holdout", "historical_sentinels"]
                    else "transfer"
                )
            )
            expected_ids = science["rosters"][name]
            if endpoint.startswith("benign_"):
                expected_ids = [
                    i
                    for i in expected_ids
                    if roster[i]["source"] == endpoint[len("benign_") :]
                ]
            if endpoint in ["primary_explanation", "primary_application"]:
                policies = (
                    ["specific_always", "plain_always"]
                    if endpoint == "primary_explanation"
                    else ["plain_selective", "plain_always"]
                )
                expected_keys = {
                    (f"{p}_s{s}", r)
                    for p in policies
                    for s in science["conditions"]["pool_seeds"]
                    for r in range(3)
                }
                require(
                    set(keys) == expected_keys and len(keys) == 18,
                    "Explanation primary setting mask differs",
                )
                require(
                    value["bootstrap"]["seed"]
                    == (20261001 if endpoint == "primary_explanation" else 20261002),
                    "Explanation primary seed differs",
                )
        else:
            expected_ids = [
                i
                for i, v in roster.items()
                if v["population"]
                == ("benign" if endpoint.startswith("benign") else "hateful_types")
            ]
            if endpoint.startswith("benign_"):
                expected_ids = [
                    i
                    for i in expected_ids
                    if roster[i]["source"] == endpoint[len("benign_") :]
                ]
            if endpoint == "primary_representatives":
                by_cluster = defaultdict(list)
                for i in expected_ids:
                    by_cluster[roster[i]["cluster"]].append(i)
                expected_ids = [
                    min(
                        v,
                        key=lambda i: hashlib.sha256(
                            ("20261001-taxonomy-representative|" + i).encode()
                        ).hexdigest(),
                    )
                    for v in by_cluster.values()
                ]
            if endpoint.startswith("primary"):
                expected_keys = {
                    (c, r)
                    for c in ["rag_full", *["rag_schema_" + l for l in LABELS]]
                    for r in range(3)
                }
                require(
                    set(keys) == expected_keys
                    and len(keys) == 24
                    and value["bootstrap"]["seed"] == 2026100103,
                    "Taxonomy primary contract differs",
                )
        require(
            original == sorted(expected_ids),
            "Scored population differs from frozen scientific roster",
        )
        common = [
            i
            for i in original
            if all(runs[study, c, r][i]["result"] is not None for c, r in keys)
        ]
        require(
            common == value["common_valid_ids"]
            and len(common) == value["common_valid_n"]
            and len(original) == value["original_n"],
            "Independent mask differs",
        )
        mask_checks += 1
        if "/benign" in path:
            for setting, z in settings.items():
                c, r = setting.rsplit("_rep", 1)
                run = runs[study, c, int(r)]
                good = [i for i in original if run[i]["result"] is not None]
                fp = sum(run[i]["result"]["hate"] for i in good)
                require(
                    len(good) == z["own_valid_n"] and fp == z["own_fp"],
                    "Benign own counts",
                )
                require(
                    [i for i in original if run[i]["result"] is None] == z["null_ids"],
                    "Benign null IDs",
                )
                eq(z["own_fp_rate"], fp / len(good))
                eq(
                    z["full_roster_bounds"],
                    [
                        fp / len(original),
                        (fp + len(original) - len(good)) / len(original),
                    ],
                )
                eq(
                    z["common_fp_rate"],
                    sum(run[i]["result"]["hate"] for i in common) / len(common),
                )
                metric_checks += 1
            for group, z in value["groups"].items():
                pts = [
                    settings[f"{c}_rep{r}"]
                    for c in members(group, settings)
                    for r in range(3)
                ]
                eq(z["fp_rate"], statistics.mean(v["common_fp_rate"] for v in pts))
                eq(
                    z["own_valid_fp_rate"],
                    statistics.mean(v["own_fp_rate"] for v in pts),
                )
                eq(
                    z["full_roster_bounds"],
                    np.mean([v["full_roster_bounds"] for v in pts], axis=0),
                )
            for name, z in value["contrasts"].items():
                a, b = name.split("_minus_")
                eq(
                    z["difference"],
                    value["groups"][a]["fp_rate"] - value["groups"][b]["fp_rate"],
                )
                eq(
                    z["difference"],
                    z["became_false_positive_rate"] - z["ceased_false_positive_rate"],
                )
            continue
        for setting, z in settings.items():
            c, r = setting.rsplit("_rep", 1)
            r = int(r)
            for kind, ids in [("own_valid", original), ("common_valid", common)]:
                m = run_metric(study, c, r, ids)
                for k, v in m.items():
                    if k == "null_ids":
                        require(v == z[kind][k], "Null IDs differ")
                    else:
                        eq(v, z[kind][k])
                worst = run_metric(study, c, r, ids, "worst")
                best = run_metric(study, c, r, ids, "best")
                eq(
                    z[kind]["full_roster_macro_bounds"],
                    [worst["macro_f1"], best["macro_f1"]],
                )
                eq(
                    z[kind]["full_roster_per_label_bounds"],
                    np.array([worst["per_label_f1"], best["per_label_f1"]]).T,
                )
                metric_checks += 1
        for group, z in value["groups"].items():
            ms = members(group, settings)
            pts = [
                settings[f"{c}_rep{r}"]["common_valid"] for c in ms for r in range(3)
            ]
            own = [settings[f"{c}_rep{r}"]["own_valid"] for c in ms for r in range(3)]
            for k in [
                "macro_f1",
                "per_label_f1",
                "contains",
                "exact",
                "predicted_labels",
                "missed_hate_rate",
            ]:
                eq(z[k], np.mean([p[k] for p in pts], axis=0))
            eq(
                z["pool_means"],
                [
                    statistics.mean(
                        settings[f"{c}_rep{r}"]["common_valid"]["macro_f1"]
                        for r in range(3)
                    )
                    for c in ms
                ],
            )
            eq(
                z["full_roster_per_label_bounds"],
                np.mean([p["full_roster_per_label_bounds"] for p in own], axis=0),
            )
        for kind in ["contrasts", "target_contrasts", "other_six_contrasts"]:
            for name, z in value.get(kind, {}).items():
                if name == "explanation_by_selectivity_interaction":
                    eq(
                        z["difference"],
                        value["groups"]["specific_selective"]["macro_f1"]
                        - value["groups"]["plain_selective"]["macro_f1"]
                        - value["groups"]["specific_always"]["macro_f1"]
                        + value["groups"]["plain_always"]["macro_f1"],
                    )
                    continue
                if name in LABELS:
                    a, b = "rag_full", "rag_schema_" + name
                    ind = [LABELS.index(name)]
                else:
                    a, b = name.split("_minus_")
                    ind = list(range(7))
                    if kind == "target_contrasts":
                        ind = [LABELS.index(value["target"])]
                    if kind == "other_six_contrasts":
                        ind = [n for n, l in enumerate(LABELS) if l != value["target"]]
                ga = value["groups"][a]
                gb = value["groups"][b]
                eq(
                    z["difference"],
                    np.mean(np.array(ga["per_label_f1"])[ind])
                    - np.mean(np.array(gb["per_label_f1"])[ind]),
                )
                ab = np.array(ga["full_roster_per_label_bounds"])[ind].mean(axis=0)
                bb = np.array(gb["full_roster_per_label_bounds"])[ind].mean(axis=0)
                eq(
                    z["full_roster_missing_difference_bounds"],
                    [ab[0] - bb[1], ab[1] - bb[0]],
                )
        if "primary" in value:
            full = value["groups"]["rag_full"]["per_label_f1"]
            schema = [
                value["groups"]["rag_schema_" + l]["per_label_f1"][n]
                for n, l in enumerate(LABELS)
            ]
            eq(
                value["primary"]["difference"],
                statistics.mean(a - b for a, b in zip(full, schema)),
            )

    specs = [
        ("explanations", "primary_explanation", "specific_always", "plain_always"),
        ("explanations", "primary_application", "plain_selective", "plain_always"),
        ("taxonomy", "primary", None, None),
        ("taxonomy", "primary_representatives", None, None),
    ]
    for study, name, left, right in specs:
        v = read(DEST / study / (name + ".json"))
        ids = v["common_valid_ids"]
        clusters = sorted({rosters[study][i]["cluster"] for i in ids})
        cp = {c: n for n, c in enumerate(clusters)}
        index = np.array([cp[rosters[study][i]["cluster"]] for i in ids])
        rng = np.random.Generator(np.random.PCG64(v["bootstrap"]["seed"]))
        picks = rng.integers(0, len(clusters), (10000, len(clusters)))
        weights = np.zeros((10000, len(clusters)), dtype=np.int32)
        np.add.at(weights, (np.arange(10000)[:, None], picks), 1)
        groups = {}
        for g in v["groups"]:
            ss = []
            for c in members(g, v["settings"]):
                for rep in range(3):
                    cc = np.zeros((len(clusters), 7, 2), dtype=np.int32)
                    for n, i in enumerate(ids):
                        gold = set(rosters[study][i]["gold"]["hate_type"])
                        p = set(runs[study, c, rep][i]["result"]["hate_type"])
                        for k, l in enumerate(LABELS):
                            cc[index[n], k, 0] += int(l in gold and l in p)
                            cc[index[n], k, 1] += int(l in gold) + int(l in p)
                    total = (weights @ cc.reshape(len(clusters), -1)).reshape(
                        10000, 7, 2
                    )
                    den = total[:, :, 1]
                    ss.append(
                        np.divide(
                            2 * total[:, :, 0],
                            den,
                            out=np.zeros_like(den, dtype=float),
                            where=den > 0,
                        )
                    )
            groups[g] = np.mean(ss, axis=0)
        if study == "explanations":
            diff = (groups[left] - groups[right]).mean(axis=1)
            expected = v["contrasts"][left + "_minus_" + right]
        else:
            diff = np.mean(
                [
                    groups["rag_full"][:, k] - groups["rag_schema_" + l][:, k]
                    for k, l in enumerate(LABELS)
                ],
                axis=0,
            )
            expected = v["primary"]
        interval = np.quantile(diff, [0.00625, 0.99375])
        eq(interval, expected["ci9875"])
        primary_bootstraps[study + "/" + name] = {
            "draws": 10000,
            "ci9875": interval.tolist(),
        }
        print("Independently reconstructed bootstrap:", study, name, flush=True)
    save(
        DEST / "independent_verification.json",
        {
            "status": "passed",
            "created_utc": utc(),
            "scoring_complete_sha256": sha(DEST / "scoring_complete.json"),
            "source_sha256": sha(__file__),
            "verified_run_metrics": metric_checks,
            "verified_common_masks": mask_checks,
            "primary_bootstraps": primary_bootstraps,
            "method": "Independent set counts, support+prediction F1 denominator, every run and mask, group/pool averages, benign rates/bounds/flips, and all 40000 primary/sensitivity bootstrap draws. Does not import scoring functions.",
            "new_model_calls": 0,
        },
    )
    print(
        "Independent verification passed:",
        metric_checks,
        "metrics;",
        mask_checks,
        "masks.",
        flush=True,
    )


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket, "create_connection", forbid
    ), threadpool_limits(limits=2):
        main()

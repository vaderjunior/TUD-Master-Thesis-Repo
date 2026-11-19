import argparse
from collections import defaultdict
import itertools
import socket
from unittest.mock import patch
import numpy as np

from analysis.feedback.initial.archive import ROOT
from analysis.feedback.initial.archive import DEST
from analysis.feedback.initial.archive import REVIEW
from analysis.feedback.initial.archive import STUDIES
from analysis.feedback.initial.archive import LABELS
from analysis.feedback.initial.archive import SEEDS
from analysis.feedback.initial.archive import read
from analysis.feedback.initial.archive import rows
from analysis.feedback.initial.archive import sha
from analysis.feedback.initial.archive import rel
from analysis.feedback.initial.archive import save
from analysis.feedback.initial.archive import verify
from analysis.feedback.initial.archive import require
from analysis.feedback.initial.archive import utc
from analysis.feedback.initial.archive import forbid

POLICIES = [
    "reference",
    "plain_always",
    "specific_always",
    "generic_always",
    "plain_selective",
    "specific_selective",
]
EX_CONTRASTS = [
    ("specific_always", "plain_always"),
    ("plain_selective", "plain_always"),
    ("specific_always", "generic_always"),
    ("generic_always", "plain_always"),
    ("specific_selective", "plain_selective"),
    ("specific_selective", "specific_always"),
    *[(p, "reference") for p in POLICIES if p != "reference"],
]


def safe_ratio(num, den):
    return np.divide(
        num,
        den,
        out=np.zeros(np.broadcast_shapes(np.shape(num), np.shape(den)), dtype=float),
        where=den != 0,
    )


def f1(c):
    return safe_ratio(2 * c[..., 0], 2 * c[..., 0] + c[..., 1] + c[..., 2])


def counts(y, x):
    return np.stack([y & x, ~y & x, y & ~x], axis=-1).astype(np.int64)


def ex_groups(policies=POLICIES):
    return {
        p: [p] if p == "reference" else [f"{p}_s{s}" for s in SEEDS] for p in policies
    }


class Data:
    def __init__(self, study, exported):
        self.study = study
        self.roster = {r["id"]: r for r in rows(STUDIES[study] / "roster.jsonl")}
        self.ids = sorted(self.roster)
        self.pos = {i: k for k, i in enumerate(self.ids)}
        self.runs = defaultdict(dict)
        for r in exported:
            if r["study"] != study:
                continue
            key = r["condition"], r["repeat"]
            require(r["id"] not in self.runs[key], "Duplicate prediction")
            require(
                r["gold"] == self.roster[r["id"]]["gold"]
                and r["cluster"] == self.roster[r["id"]]["cluster"],
                "Gold/cluster mismatch",
            )
            self.runs[key][r["id"]] = r
        self.keys = sorted(self.runs)
        self.kpos = {k: i for i, k in enumerate(self.keys)}
        require(
            len(self.keys) == (48 if study == "explanations" else 69),
            "Wrong setting count",
        )
        require(
            all(set(v) == set(self.ids) for v in self.runs.values()),
            "Incomplete setting roster",
        )
        self.y = np.array(
            [
                [l in (self.roster[i]["gold"]["hate_type"] or []) for l in LABELS]
                for i in self.ids
            ],
            dtype=bool,
        )
        self.valid = np.array(
            [
                [self.runs[k][i]["result"] is not None for i in self.ids]
                for k in self.keys
            ]
        )
        self.x = np.array(
            [
                [
                    [
                        (
                            l in self.runs[k][i]["result"]["hate_type"]
                            if self.valid[n, m]
                            else False
                        )
                        for l in LABELS
                    ]
                    for m, i in enumerate(self.ids)
                ]
                for n, k in enumerate(self.keys)
            ],
            dtype=bool,
        )
        self.gate = np.array(
            [
                [
                    self.runs[k][i]["result"]["hate"] if self.valid[n, m] else False
                    for m, i in enumerate(self.ids)
                ]
                for n, k in enumerate(self.keys)
            ],
            dtype=bool,
        )
        self.allowed = np.ones((len(self.keys), 7), dtype=bool)
        for n, (condition, _) in enumerate(self.keys):
            if "_removed_" in condition:
                self.allowed[n, LABELS.index(condition.split("_removed_")[1])] = False
        require(
            not np.any(self.x & ~self.allowed[:, None, :]),
            "Forbidden label in voted prediction",
        )
        require(
            not np.any(self.x & ~self.gate[:, :, None]),
            "Fine labels despite negative gate",
        )

    def positions(self, ids):
        return np.array([self.pos[i] for i in ids], dtype=int)

    def choose(self, groups):
        return [
            self.kpos[c, r]
            for members in groups.values()
            for c in members
            for r in range(3)
        ]

    def common(self, ids, keys):
        pos = self.positions(ids)
        ok = self.valid[np.ix_(keys, pos)].all(axis=0)
        return [i for i, v in zip(ids, ok) if v]

    def metric(self, ids, key):
        pos = self.positions(ids)
        ok = self.valid[key, pos]
        y = self.y[pos]
        x = self.x[key, pos]
        gate = self.gate[key, pos]
        c = counts(y[ok], x[ok]).sum(axis=0)
        per = f1(c)
        n = int(ok.sum())
        bounds = []
        for impute in [~y, y]:
            xx = x.copy()
            xx[~ok] = impute[~ok] & self.allowed[key]
            bounds.append(f1(counts(y, xx).sum(axis=0)).tolist())
        return {
            "n": len(ids),
            "valid": n,
            "null_ids": [i for i, v in zip(ids, ok) if not v],
            "macro_f1": float(per.mean()),
            "per_label_f1": per.tolist(),
            "label_counts": c.tolist(),
            "precision": safe_ratio(c[:, 0], c[:, 0] + c[:, 1]).tolist(),
            "recall": safe_ratio(c[:, 0], c[:, 0] + c[:, 2]).tolist(),
            "support": y[ok].sum(axis=0).tolist(),
            "contains": float(np.all(~y[ok] | x[ok], axis=1).mean()) if n else None,
            "exact": float(np.all(y[ok] == x[ok], axis=1).mean()) if n else None,
            "predicted_labels": float(x[ok].sum(axis=1).mean()) if n else None,
            "missed_hate_rate": float((~gate[ok]).mean()) if n else None,
            "full_roster_per_label_bounds": np.array(bounds).T.tolist(),
            "full_roster_macro_bounds": np.array(bounds).mean(axis=1).tolist(),
        }


def cluster_weights(ids, roster, seed, draws):
    clusters = sorted({roster[i]["cluster"] for i in ids})
    cp = {c: k for k, c in enumerate(clusters)}
    membership = np.array([cp[roster[i]["cluster"]] for i in ids], dtype=int)
    picks = np.random.Generator(np.random.PCG64(seed)).integers(
        0, len(clusters), size=(draws, len(clusters))
    )

    weights = np.zeros((draws, len(clusters)), dtype=np.float64)
    for b, pick in enumerate(picks):
        weights[b] = np.bincount(pick, minlength=len(clusters))
    return weights, membership, picks


def compare(data, ids, groups, seed, contrasts=(), draws=10000):
    ids = sorted(ids)
    keys = data.choose(groups)
    common = data.common(ids, keys)
    require(common, "Empty common mask")
    pos = data.positions(common)
    w, mem, picks = cluster_weights(common, data.roster, seed, draws)
    run_boot = {}
    settings = {}
    run_points = {}
    for key in keys:
        y = data.y[pos]
        x = data.x[key, pos]
        c = counts(y, x)
        cc = np.zeros((w.shape[1], 7, 3), dtype=np.int64)
        np.add.at(cc, mem, c)
        boot = f1((w @ cc.reshape(w.shape[1], -1)).reshape(draws, 7, 3))
        run_boot[key] = boot

        for b in sorted(set([0, draws - 1, *range(min(draws, 17))])):
            expanded = np.concatenate(
                [np.flatnonzero(mem == cluster) for cluster in picks[b]]
            )
            require(
                np.allclose(
                    boot[b],
                    f1(counts(y[expanded], x[expanded]).sum(axis=0)),
                    rtol=0,
                    atol=1e-13,
                ),
                "Bootstrap cluster expansion differs",
            )
        own = data.metric(ids, key)
        matched = data.metric(common, key)
        run_points[key] = matched
        cnd, rep = data.keys[key]
        settings[f"{cnd}_rep{rep}"] = {"own_valid": own, "common_valid": matched}
    group_boot = {}
    results = {}
    for name, members in groups.items():
        kk = [data.kpos[c, r] for c in members for r in range(3)]
        pts = [run_points[k] for k in kk]
        own = [
            settings[f"{data.keys[k][0]}_rep{data.keys[k][1]}"]["own_valid"] for k in kk
        ]
        group_boot[name] = np.mean([run_boot[k] for k in kk], axis=0)
        means = [
            np.mean([run_points[data.kpos[c, r]]["macro_f1"] for r in range(3)])
            for c in members
        ]
        results[name] = {
            m: float(np.mean([p[m] for p in pts]))
            for m in [
                "macro_f1",
                "contains",
                "exact",
                "predicted_labels",
                "missed_hate_rate",
            ]
        }
        results[name].update(
            per_label_f1=np.mean([p["per_label_f1"] for p in pts], axis=0).tolist(),
            own_valid_macro_f1=float(np.mean([p["macro_f1"] for p in own])),
            own_valid_per_label_f1=np.mean(
                [p["per_label_f1"] for p in own], axis=0
            ).tolist(),
            full_roster_macro_bounds=np.mean(
                [p["full_roster_macro_bounds"] for p in own], axis=0
            ).tolist(),
            full_roster_per_label_bounds=np.mean(
                [p["full_roster_per_label_bounds"] for p in own], axis=0
            ).tolist(),
            pool_means=[float(x) for x in means],
            pool_sample_sd=float(np.std(means, ddof=1)) if len(means) > 1 else None,
            decoder_sample_sd_by_pool=[
                float(
                    np.std(
                        [run_points[data.kpos[c, r]]["macro_f1"] for r in range(3)],
                        ddof=1,
                    )
                )
                for c in members
            ],
        )
    contrast_results = {}
    for left, right in contrasts:
        if left in groups and right in groups:
            contrast_results[left + "_minus_" + right] = contrast(
                results, group_boot, left, right
            )
    if all(
        p in groups
        for p in [
            "plain_always",
            "specific_always",
            "plain_selective",
            "specific_selective",
        ]
    ):
        terms = [
            ("specific_selective", 1),
            ("plain_selective", -1),
            ("specific_always", -1),
            ("plain_always", 1),
        ]
        diff = sum(s * group_boot[p].mean(axis=1) for p, s in terms)
        contrast_results["explanation_by_selectivity_interaction"] = {
            "difference": sum(s * results[p]["macro_f1"] for p, s in terms),
            "ci95": np.quantile(diff, [0.025, 0.975]).tolist(),
        }
    out = {
        "original_n": len(ids),
        "common_valid_n": len(common),
        "common_valid_ids": common,
        "excluded_ids": sorted(set(ids) - set(common)),
        "cluster_count": w.shape[1],
        "labels": LABELS,
        "groups": results,
        "settings": settings,
        "contrasts": contrast_results,
        "bootstrap": {
            "draws": draws,
            "seed": seed,
            "generator": "PCG64",
            "sampling": "paired whole input clusters; fixed conditions/pools/repeats",
            "independently_expanded_draws_per_setting": min(draws, 17) + (draws > 17),
        },
    }
    return out, group_boot


def contrast(results, boot, left, right, indices=None):
    idx = np.arange(7) if indices is None else np.array(indices)
    a = results[left]
    b = results[right]
    pt = float(
        np.array(a["per_label_f1"])[idx].mean()
        - np.array(b["per_label_f1"])[idx].mean()
    )
    diff = boot[left][:, idx].mean(axis=1) - boot[right][:, idx].mean(axis=1)
    ab = np.array(a["full_roster_per_label_bounds"])[idx].mean(axis=0)
    bb = np.array(b["full_roster_per_label_bounds"])[idx].mean(axis=0)
    return {
        "difference": pt,
        "ci95": np.quantile(diff, [0.025, 0.975]).tolist(),
        "ci9875": np.quantile(diff, [0.00625, 0.99375]).tolist(),
        "own_valid_difference": float(
            np.array(a["own_valid_per_label_f1"])[idx].mean()
            - np.array(b["own_valid_per_label_f1"])[idx].mean()
        ),
        "full_roster_missing_difference_bounds": [
            float(ab[0] - bb[1]),
            float(ab[1] - bb[0]),
        ],
    }


def benign(data, ids, groups, seed, contrasts, draws=10000):
    ids = sorted(ids)
    keys = data.choose(groups)
    common = data.common(ids, keys)
    require(common, "Empty benign mask")
    require(
        all(
            not data.roster[i]["gold"]["hate"]
            and data.roster[i]["gold"]["hate_type"] is None
            for i in ids
        ),
        "Benign gold imputed",
    )
    pos = data.positions(common)
    original = data.positions(ids)
    w, mem, _ = cluster_weights(common, data.roster, seed, draws)
    denom = w @ np.bincount(mem, minlength=w.shape[1])
    result = {}
    boots = {}
    settings = {}
    for k in keys:
        valid = data.valid[k, original]
        fp = int(data.gate[k, original][valid].sum())
        c, r = data.keys[k]
        settings[f"{c}_rep{r}"] = {
            "original_n": len(ids),
            "own_valid_n": int(valid.sum()),
            "own_fp": fp,
            "own_fp_rate": fp / int(valid.sum()),
            "null_ids": [i for i, v in zip(ids, valid) if not v],
            "full_roster_bounds": [
                fp / len(ids),
                (fp + int((~valid).sum())) / len(ids),
            ],
            "common_valid_n": len(common),
            "common_fp": int(data.gate[k, pos].sum()),
            "common_fp_rate": float(data.gate[k, pos].mean()),
        }
    for name, members in groups.items():
        kk = [data.kpos[c, r] for c in members for r in range(3)]
        x = data.gate[np.ix_(kk, pos)].mean(axis=0)
        cc = np.zeros(w.shape[1])
        np.add.at(cc, mem, x)
        boots[name] = (w @ cc) / denom
        own = [settings[f"{c}_rep{r}"] for c in members for r in range(3)]
        result[name] = {
            "fp_rate": float(x.mean()),
            "ci95": np.quantile(boots[name], [0.025, 0.975]).tolist(),
            "own_valid_fp_rate": float(np.mean([v["own_fp_rate"] for v in own])),
            "full_roster_bounds": np.mean(
                [v["full_roster_bounds"] for v in own], axis=0
            ).tolist(),
            "repeat_rates": [
                [settings[f"{c}_rep{r}"]["common_fp_rate"] for r in range(3)]
                for c in members
            ],
        }
    diffs = {}
    for left, right in contrasts:
        if left not in groups or right not in groups:
            continue
        diff = boots[left] - boots[right]
        a = result[left]
        b = result[right]
        pairs = []
        lm = groups[left]
        rm = groups[right]
        require(len(lm) == len(rm) or len(lm) == 1 or len(rm) == 1, "Unmatched groups")
        for n in range(max(len(lm), len(rm))):
            for rep in range(3):
                pairs.append(
                    (data.kpos[lm[n % len(lm)], rep], data.kpos[rm[n % len(rm)], rep])
                )
        up = np.mean([data.gate[a, pos] & ~data.gate[b, pos] for a, b in pairs], axis=0)
        down = np.mean(
            [~data.gate[a, pos] & data.gate[b, pos] for a, b in pairs], axis=0
        )
        diffs[left + "_minus_" + right] = {
            "difference": a["fp_rate"] - b["fp_rate"],
            "ci95": np.quantile(diff, [0.025, 0.975]).tolist(),
            "own_valid_difference": a["own_valid_fp_rate"] - b["own_valid_fp_rate"],
            "full_roster_missing_difference_bounds": [
                a["full_roster_bounds"][0] - b["full_roster_bounds"][1],
                a["full_roster_bounds"][1] - b["full_roster_bounds"][0],
            ],
            "became_false_positive_rate": float(up.mean()),
            "ceased_false_positive_rate": float(down.mean()),
        }
    return {
        "original_n": len(ids),
        "common_valid_n": len(common),
        "common_valid_ids": common,
        "excluded_ids": sorted(set(ids) - set(common)),
        "cluster_count": w.shape[1],
        "groups": result,
        "settings": settings,
        "contrasts": diffs,
        "bootstrap": {
            "draws": draws,
            "seed": seed,
            "level": 0.95,
            "supporting_only": True,
        },
    }


def taxonomy_primary(data, ids, seed):
    groups = {c: [c] for c in ["rag_full", *[f"rag_schema_{l}" for l in LABELS]]}
    result, boot = compare(data, ids, groups, seed)
    point = []
    draws = []
    own = []
    bounds = []
    for n, l in enumerate(LABELS):
        ct = contrast(result["groups"], boot, "rag_full", f"rag_schema_{l}", [n])
        point.append(ct["difference"])
        own.append(ct["own_valid_difference"])
        bounds.append(ct["full_roster_missing_difference_bounds"])
        draws.append(boot["rag_full"][:, n] - boot[f"rag_schema_{l}"][:, n])
        result["contrasts"][l] = ct
    b = np.mean(draws, axis=0)
    result["primary"] = {
        "difference": float(np.mean(point)),
        "full_target_f1": float(np.mean(result["groups"]["rag_full"]["per_label_f1"])),
        "schema_only_target_f1": float(
            np.mean(
                [
                    result["groups"][f"rag_schema_{l}"]["per_label_f1"][n]
                    for n, l in enumerate(LABELS)
                ]
            )
        ),
        "ci9875": np.quantile(b, [0.00625, 0.99375]).tolist(),
        "ci95": np.quantile(b, [0.025, 0.975]).tolist(),
        "own_valid_difference": float(np.mean(own)),
        "full_roster_missing_difference_bounds": np.mean(bounds, axis=0).tolist(),
    }
    return result


def tests():
    require(np.all(f1(np.zeros((7, 3), int)) == 0), "Zero denominator")
    y = np.array([[1, 0], [0, 1]], bool)
    require(np.all(f1(counts(y, y).sum(axis=0)) == 1), "Perfect F1")
    require(np.all(f1(counts(y, ~y).sum(axis=0)) == 0), "Complement F1")

    yy = np.array([[1, 0], [0, 1], [1, 1]], bool)
    xx = np.array([[1, 1], [0, 0], [0, 0]], bool)
    for allowed in [np.array([1, 1], bool), np.array([1, 0], bool)]:
        all_values = []
        for bits in itertools.product([False, True], repeat=4):
            z = xx.copy()
            z[1:] = np.array(bits).reshape(2, 2) & allowed
            z[0] &= allowed
            all_values.append(f1(counts(yy, z).sum(axis=0)))
        lo = xx.copy()
        hi = xx.copy()
        lo[0] &= allowed
        hi[0] &= allowed
        lo[1:] = ~yy[1:] & allowed
        hi[1:] = yy[1:] & allowed
        require(
            np.allclose(f1(counts(yy, lo).sum(axis=0)), np.min(all_values, axis=0)),
            "Worst bound",
        )
        require(
            np.allclose(f1(counts(yy, hi).sum(axis=0)), np.max(all_values, axis=0)),
            "Best bound",
        )
    roster = {"a": {"cluster": "one"}, "b": {"cluster": "one"}, "c": {"cluster": "two"}}
    w, mem, picks = cluster_weights(list(roster), roster, 17, 100)
    require(
        np.all(w.sum(axis=1) == 2)
        and np.all((w[:, mem].sum(axis=1) >= 2) & (w[:, mem].sum(axis=1) <= 4)),
        "Whole-cluster size",
    )
    for b in range(100):
        require(
            np.array_equal(w[b], np.bincount(picks[b], minlength=2)),
            "Paired cluster draws",
        )
    require(
        len(ex_groups()["reference"]) == 1 and len(ex_groups()["plain_always"]) == 3,
        "Shared reference/pool weights",
    )
    fixture = Data.__new__(Data)
    fixture.ids = list(roster)
    fixture.pos = {i: k for k, i in enumerate(fixture.ids)}
    fixture.roster = roster
    fixture.keys = [
        (c, r) for members in ex_groups().values() for c in members for r in range(3)
    ]
    fixture.kpos = {k: n for n, k in enumerate(fixture.keys)}
    fixture.y = np.array(
        [[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]], bool
    )
    fixture.x = np.broadcast_to(fixture.y, (48, 3, 7)).copy()
    fixture.valid = np.ones((48, 3), bool)
    fixture.gate = np.ones((48, 3), bool)
    fixture.allowed = np.ones((48, 7), bool)
    result, _ = compare(fixture, list(roster), ex_groups(), 19, EX_CONTRASTS, draws=31)
    require(
        all(
            abs(c["difference"]) < 1e-13 and np.allclose(c["ci95"], [0, 0])
            for c in result["contrasts"].values()
        ),
        "Identical paired systems",
    )
    require(
        len(result["groups"]["reference"]["pool_means"]) == 1, "Reference duplicated"
    )
    for c in ex_groups()["plain_always"]:
        for r in range(3):
            fixture.x[fixture.kpos[c, r]] = False
            fixture.gate[fixture.kpos[c, r]] = False
    result, _ = compare(
        fixture,
        list(roster),
        ex_groups(["specific_always", "plain_always"]),
        19,
        [("specific_always", "plain_always")],
        draws=31,
    )
    require(
        abs(
            result["contrasts"]["specific_always_minus_plain_always"]["difference"]
            - 2 / 7
        )
        < 1e-13,
        "Fixed-label scoring and averaging",
    )
    fixture.valid[fixture.kpos["plain_always_s202609281", 0], 0] = False
    result, _ = compare(
        fixture,
        list(roster),
        ex_groups(["specific_always", "plain_always"]),
        19,
        [("specific_always", "plain_always")],
        draws=31,
    )
    require(
        result["excluded_ids"] == ["a"]
        and result["common_valid_n"] == 2
        and result["cluster_count"] == 2,
        "Common mask retains surviving cluster member",
    )
    save(
        DEST / "development" / f"scoring_tests_{sha(__file__)[:16]}.json",
        {
            "passed": True,
            "source_sha256": sha(__file__),
            "created_utc": utc(),
            "checks": "Zero/perfect/complement F1; exhaustive missing-output bounds with forbidden label; unequal cluster sizes, draw expansion and shared reference weights. Scientific tables separately checked using independent set/count formulas.",
            "new_model_calls": 0,
        },
    )
    print("Scoring fixtures passed.", flush=True)


def score():
    tested = DEST / "development" / f"scoring_tests_{sha(__file__)[:16]}.json"
    require(read(tested)["passed"], "Tests required")
    comp = read(REVIEW / "complete.json")
    verify(comp["outputs"])
    plan = read(DEST / "analysis_plan.json")
    verify(plan["inputs"])
    exported = list(rows(REVIEW / "predictions.jsonl"))
    outputs = []

    def put(study, name, value):
        path = DEST / study / (name + ".json")
        save(path, value)
        outputs.append(path)
        if "primary" in value:
            print(study, name, json_message(value["primary"]), flush=True)
        elif name.startswith("primary"):
            print(study, name, json_message(value["contrasts"]), flush=True)
        else:
            print(
                study,
                name,
                "saved; n=" + str(value.get("common_valid_n", "")),
                flush=True,
            )

    data = Data("explanations", exported)
    science = read(STUDIES["explanations"] / "scientific_plan_v1.json")
    hateful = science["rosters"]["transfer"]
    for name, left, right, seed in [
        ("primary_explanation", "specific_always", "plain_always", 20261001),
        ("primary_application", "plain_selective", "plain_always", 20261002),
    ]:
        result, _ = compare(
            data, hateful, ex_groups([left, right]), seed, [(left, right)]
        )
        put("explanations", name, result)
    for name, ids in [
        ("all_transfer", hateful),
        ("original_holdout", science["rosters"]["original_holdout"]),
        ("historical_sentinels", science["rosters"]["historical_sentinels"]),
    ]:
        result, _ = compare(data, ids, ex_groups(), 2026100402, EX_CONTRASTS)
        put("explanations", name, result)
    b_ids = science["rosters"]["benign"]
    b = benign(data, b_ids, ex_groups(), 2026100401, EX_CONTRASTS)
    put("explanations", "benign", b)

    for source in sorted({data.roster[i]["source"] for i in b_ids}):
        ss = [i for i in b_ids if data.roster[i]["source"] == source]
        put(
            "explanations",
            "benign_" + source,
            benign(data, ss, ex_groups(), 2026100401, EX_CONTRASTS),
        )
    data = Data("taxonomy", exported)
    science = read(STUDIES["taxonomy"] / "scientific_plan_v1.json")
    hateful = [i for i in data.ids if data.roster[i]["population"] == "hateful_types"]
    b_ids = [i for i in data.ids if data.roster[i]["population"] == "benign"]
    put("taxonomy", "primary", taxonomy_primary(data, hateful, 2026100103))
    clusters = read(STUDIES["taxonomy"] / "clusters.json")
    rep_ids = [
        r["representative"]
        for r in clusters
        if r["included"] and r["representative"] in hateful
    ]
    import hashlib

    for r in clusters:
        if r["included"]:
            require(
                r["representative"]
                == min(
                    r["ids"],
                    key=lambda i: hashlib.sha256(
                        ("20261001-taxonomy-representative|" + i).encode()
                    ).hexdigest(),
                ),
                "Representative choice differs",
            )
    put(
        "taxonomy",
        "primary_representatives",
        taxonomy_primary(data, rep_ids, 2026100103),
    )
    for n, l in enumerate(LABELS):
        for arm, conditions, contrasts_ in [
            (
                "rag",
                [f"rag_removed_{l}", f"rag_schema_{l}", "rag_full"],
                [
                    (f"rag_schema_{l}", f"rag_removed_{l}"),
                    ("rag_full", f"rag_schema_{l}"),
                    ("rag_full", f"rag_removed_{l}"),
                ],
            ),
            ("zs", [f"zs_removed_{l}", "zs_full"], [("zs_full", f"zs_removed_{l}")]),
        ]:
            result, boots = compare(
                data, hateful, {c: [c] for c in conditions}, 2026100105, contrasts_
            )
            result["target"] = l
            result["target_contrasts"] = {
                a + "_minus_" + b: contrast(result["groups"], boots, a, b, [n])
                for a, b in contrasts_
            }
            result["other_six_contrasts"] = {
                a
                + "_minus_"
                + b: contrast(
                    result["groups"], boots, a, b, [j for j in range(7) if j != n]
                )
                for a, b in contrasts_
            }
            common = result["common_valid_ids"]
            pos = data.positions(common)
            flips = {}
            for a, b in contrasts_:
                aa = np.stack([data.x[data.kpos[a, r], pos] for r in range(3)])
                bb = np.stack([data.x[data.kpos[b, r], pos] for r in range(3)])
                flips[a + "_minus_" + b] = {
                    "any_type_set_changed_rate": float(
                        np.any(aa != bb, axis=-1).mean()
                    ),
                    "other_six_set_changed_rate": float(
                        np.any(
                            (aa != bb)[..., [j for j in range(7) if j != n]], axis=-1
                        ).mean()
                    ),
                }
            result["type_set_changes"] = flips
            put("taxonomy", f"{arm}_{l}", result)
    groups = {c: [c] for c in sorted({c for c, r in data.keys})}
    bcontrasts = [
        (c, "zs_full" if c.startswith("zs_") else "rag_full")
        for c in groups
        if c not in ["rag_full", "zs_full"]
    ] + [("rag_full", "zs_full")]
    bcontrasts += [(f"rag_schema_{l}", f"rag_removed_{l}") for l in LABELS]
    put("taxonomy", "benign", benign(data, b_ids, groups, 2026100104, bcontrasts))
    for source in sorted({data.roster[i]["source"] for i in b_ids}):
        put(
            "taxonomy",
            "benign_" + source,
            benign(
                data,
                [i for i in b_ids if data.roster[i]["source"] == source],
                groups,
                2026100104,
                bcontrasts,
            ),
        )
    inputs = {
        **comp["outputs"],
        rel(REVIEW / "complete.json"): sha(REVIEW / "complete.json"),
        rel(DEST / "analysis_plan.json"): sha(DEST / "analysis_plan.json"),
        rel(__file__): sha(__file__),
        rel(tested): sha(tested),
    }
    verify(inputs)
    save(
        DEST / "scoring_complete.json",
        {
            "status": "all_frozen_primary_and_supporting_endpoints_scored",
            "created_utc": utc(),
            "inputs": inputs,
            "outputs": {rel(p): sha(p) for p in outputs},
            "new_model_calls": 0,
        },
    )


def json_message(v):
    import json

    return json.dumps(v, sort_keys=True)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["tests", "score"])
    args = p.parse_args()
    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket, "create_connection", forbid
    ), threadpool_limits(limits=2):
        {"tests": tests, "score": score}[args.mode]()

from collections import defaultdict
import hashlib
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

EFFECTS = {
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


def ratio(n, d):
    return np.divide(
        n,
        d,
        out=np.zeros(np.broadcast_shapes(np.shape(n), np.shape(d)), dtype=float),
        where=d != 0,
    )


def f1(count):
    return ratio(2 * count[..., 0], 2 * count[..., 0] + count[..., 1] + count[..., 2])


def counts(y, x):
    return np.stack([y & x, ~y & x, y & ~x], axis=-1).astype(np.int64)


def intervals(x, coverage=0.95):
    return np.quantile(x, [(1 - coverage) / 2, 1 - (1 - coverage) / 2]).tolist()


def groups_human():
    return {
        "reference": ["reference"],
        **{p: [f"{p}_s{s}" for s in SEEDS] for p in ["plain", "human", "generic"]},
    }


class Data:
    def __init__(self, study, exported):
        self.study = study
        self.roster = {r["key"]: r for r in rows(STUDIES[study] / "roster.jsonl")}
        self.ids = sorted(self.roster)
        self.pos = {v: i for i, v in enumerate(self.ids)}
        self.conditions = read(STUDIES[study] / "scientific_plan_v1.json")["conditions"]
        self.keys = [(c, r) for c in self.conditions for r in range(3)]
        self.kpos = {k: i for i, k in enumerate(self.keys)}
        self.records = {
            (r["key"], r["condition"], r["repeat"]): r
            for r in exported
            if r["study"] == study
        }
        require(
            set(self.records) == {(i, c, r) for i in self.ids for c, r in self.keys},
            "Incomplete scientific matrix",
        )
        self.y = np.array(
            [
                [l in (self.roster[i]["gold"]["hate_type"] or []) for l in LABELS]
                for i in self.ids
            ],
            dtype=bool,
        )
        self.truth = np.array(
            [self.roster[i]["gold"]["hate"] for i in self.ids], dtype=bool
        )
        self.valid = np.array(
            [
                [self.records[i, c, r]["result"] is not None for i in self.ids]
                for c, r in self.keys
            ]
        )
        self.votes = np.array(
            [[self.records[i, c, r]["n_valid"] for i in self.ids] for c, r in self.keys]
        )
        self.x = np.zeros((len(self.keys), len(self.ids), 7), dtype=bool)
        self.gate = np.zeros(self.valid.shape, dtype=bool)
        for k, (c, r) in enumerate(self.keys):
            for n, i in enumerate(self.ids):
                p = self.records[i, c, r]["result"]
                if p is not None:
                    self.x[k, n] = [l in p["hate_type"] for l in LABELS]
                    self.gate[k, n] = p["hate"]
        require(
            not np.any(self.x & ~self.gate[:, :, None]),
            "Negative hate gate has type labels",
        )

    def indices(self, ids):
        return np.array([self.pos[i] for i in ids], dtype=int)

    def chosen(self, groups):
        return [
            self.kpos[c, r]
            for members in groups.values()
            for c in members
            for r in range(3)
        ]

    def common(self, ids, keys, strict=False):
        p = self.indices(ids)
        v = (self.valid & (self.votes == 3)) if strict else self.valid
        return [i for i, ok in zip(ids, v[np.ix_(keys, p)].all(axis=0)) if ok]

    def population(self, hateful):
        return [i for i in self.ids if self.roster[i]["gold"]["hate"] == hateful]


def representatives(data, ids):
    clustered = defaultdict(set)
    for i in ids:
        clustered[data.roster[i]["cluster"]].add(data.roster[i].get("original_key", i))
    chosen = {
        min(
            v,
            key=lambda x: hashlib.sha256(
                ("20261006-additions-representative|" + x).encode()
            ).hexdigest(),
        )
        for v in clustered.values()
    }
    return [i for i in ids if data.roster[i].get("original_key", i) in chosen]


def weights(ids, roster, seed, draws=10000):
    clusters = sorted({roster[i]["cluster"] for i in ids})
    require(bool(clusters), "Empty bootstrap")
    index = {v: k for k, v in enumerate(clusters)}
    member = np.array([index[roster[i]["cluster"]] for i in ids])
    picks = np.random.Generator(np.random.PCG64(seed)).integers(
        0, len(clusters), size=(draws, len(clusters))
    )
    w = np.zeros((draws, len(clusters)), dtype=np.float64)
    for n, p in enumerate(picks):
        w[n] = np.bincount(p, minlength=len(clusters))
    return w, member, picks


def bootstrap_counts(raw, w, member):
    totals = np.zeros((w.shape[1], *raw.shape[1:]), dtype=np.int64)
    np.add.at(totals, member, raw)
    return (w @ totals.reshape(w.shape[1], -1)).reshape(w.shape[0], *raw.shape[1:])


def run_metric(data, ids, k, missing=None):
    p = data.indices(ids)
    ok = data.valid[k, p]
    y = data.y[p]
    x = data.x[k, p].copy()
    if missing is None:
        y = y[ok]
        x = x[ok]
    else:
        x[~ok] = (~y if missing == "worst" else y)[~ok]
    c = counts(y, x).sum(axis=0)
    scores = f1(c)
    n = len(y)
    return {
        "n": n,
        "null_ids": [i for i, v in zip(ids, ok) if not v],
        "macro_f1": float(scores.mean()),
        "per_label_f1": scores.tolist(),
        "counts_tp_fp_fn": c.tolist(),
        "precision": ratio(c[:, 0], c[:, 0] + c[:, 1]).tolist(),
        "recall": ratio(c[:, 0], c[:, 0] + c[:, 2]).tolist(),
        "support": (c[:, 0] + c[:, 2]).tolist(),
        "predicted_labels": float(x.sum(axis=1).mean()) if n else None,
        "contains_all_gold": float((~y | x).all(axis=1).mean()) if n else None,
        "exact_match": float((y == x).all(axis=1).mean()) if n else None,
    }


def factor_run(data, ids, k, missing=None):
    p = data.indices(ids)
    ok = data.valid[k, p]
    cat = np.array([LABELS.index(data.roster[i]["target_label"]) for i in ids])
    y = data.y[p, np.arange(7)[cat]]
    x = data.x[k, p, cat].copy()
    if missing is not None:
        x[~ok] = (~y if missing == "worst" else y)[~ok]
    keep = ok if missing is None else np.ones(len(p), dtype=bool)
    totals = np.zeros((7, 3), dtype=np.int64)
    np.add.at(totals, cat[keep], counts(y[keep], x[keep]))
    other = []
    for l in range(7):
        chosen = [i for i, j, t in zip(ids, cat, keep) if j == l and t]
        met = run_metric(data, chosen, k, missing=missing)
        other.append(float(np.delete(met["per_label_f1"], l).mean()))
    return {
        "n": int(keep.sum()),
        "null_ids": [i for i, v in zip(ids, ok) if not v],
        "macro_target_f1": float(f1(totals).mean()),
        "per_category_f1": f1(totals).tolist(),
        "counts_tp_fp_fn": totals.tolist(),
        "precision": ratio(totals[:, 0], totals[:, 0] + totals[:, 1]).tolist(),
        "recall": ratio(totals[:, 0], totals[:, 0] + totals[:, 2]).tolist(),
        "support": (totals[:, 0] + totals[:, 2]).tolist(),
        "other_six_f1_by_category": other,
        "other_six_macro_f1": float(np.mean(other)),
    }


def effect(point, boot, bounds, terms, coverage=0.95):
    value = sum(c * point[n] for n, c in terms.items())
    draw = sum(c * boot[n] for n, c in terms.items())
    low = sum(c * bounds[n][0 if c > 0 else 1] for n, c in terms.items())
    high = sum(c * bounds[n][1 if c > 0 else 0] for n, c in terms.items())
    return {
        "difference": float(value),
        "ci95": intervals(draw),
        "interval_coverage": coverage,
        "family_interval": intervals(draw, coverage),
        "full_roster_missing_bounds": [float(low), float(high)],
    }


def factorial(data, ids, strict=False, seed=202610041):
    kk = list(range(len(data.keys)))
    common = data.common(ids, kk, strict)
    p = data.indices(common)
    require(len(common) > 0, "No retained factorial pairs")
    cat = np.array([LABELS.index(data.roster[i]["target_label"]) for i in common])
    y = data.y[p, cat]
    x = data.x[:, p, :][:, np.arange(len(p)), cat].T
    raw = np.zeros((len(p), len(kk), 7, 3), dtype=np.int64)
    for n, l in enumerate(cat):
        raw[n, :, l, :] = counts(np.full(len(kk), y[n]), x[n])
    w, member, picks = weights(common, data.roster, seed)
    bs = f1(bootstrap_counts(raw, w, member))
    for b in [0, 1, 7, 9999]:
        expanded = np.concatenate([np.flatnonzero(member == c) for c in picks[b]])
        require(
            np.allclose(bs[b], f1(raw[expanded].sum(axis=0)), rtol=0, atol=1e-13),
            "Expanded factorial draw differs",
        )
    settings = {}
    boots = {}
    points = {}
    bounds = {}
    groups = {}
    for k, (condition, rep) in enumerate(data.keys):
        settings[f"{condition}_rep{rep}"] = {
            "own_valid": factor_run(data, ids, k),
            "common_valid": factor_run(data, common, k),
            "full_roster_worst": factor_run(data, ids, k, "worst"),
            "full_roster_best": factor_run(data, ids, k, "best"),
        }
    for condition in data.conditions:
        indices = [data.kpos[condition, r] for r in range(3)]
        runs = [settings[f"{condition}_rep{r}"] for r in range(3)]
        scores = [v["common_valid"]["macro_target_f1"] for v in runs]
        points[condition] = float(np.mean(scores))
        boots[condition] = bs[:, indices].mean(axis=(1, 2))
        bounds[condition] = [
            float(np.mean([v["full_roster_" + m]["macro_target_f1"] for v in runs]))
            for m in ["worst", "best"]
        ]
        groups[condition] = {
            "macro_target_f1": points[condition],
            "decoder_values": scores,
            "decoder_sample_sd": float(np.std(scores, ddof=1)),
            "per_category_f1": np.mean(
                [v["common_valid"]["per_category_f1"] for v in runs], axis=0
            ).tolist(),
            "other_six_macro_f1": float(
                np.mean([v["common_valid"]["other_six_macro_f1"] for v in runs])
            ),
            "full_roster_missing_bounds": bounds[condition],
        }
    supports = raw.sum(axis=(0, 1))[:, [0, 2]].sum(axis=1) / len(kk)
    effects = {
        n: effect(
            points, boots, bounds, t, 0.975 if n.startswith("marginal_") else 0.95
        )
        for n, t in EFFECTS.items()
    }
    return {
        "study": "factorial",
        "endpoint": "target_f1",
        "labels": LABELS,
        "original_ids": ids,
        "common_ids": common,
        "excluded_ids": sorted(set(ids) - set(common)),
        "original_n": len(ids),
        "common_n": len(common),
        "cluster_count": w.shape[1],
        "common_positive_support_by_category": supports.astype(int).tolist(),
        "primary_estimable": bool(np.all(supports > 0)),
        "mask": "all-three-valid-votes" if strict else "all-12-nonnull",
        "settings": settings,
        "groups": groups,
        "effects": effects,
        "bootstrap": {
            "seed": seed,
            "draws": 10000,
            "generator": "PCG64",
            "unit": "original-text overlap cluster; category pairs stay together",
        },
    }


def human_compare(data, ids, groups, seed, strict=False):
    kk = data.chosen(groups)
    common = data.common(ids, kk, strict)
    p = data.indices(common)
    require(len(p) > 0, "No common human texts")
    y = data.y[p]
    x = data.x[np.ix_(kk, p, np.arange(7))].transpose(1, 0, 2)
    raw = counts(y[:, None, :], x)
    w, member, picks = weights(common, data.roster, seed)
    bs = f1(bootstrap_counts(raw, w, member))
    for b in [0, 1, 7, 9999]:
        expanded = np.concatenate([np.flatnonzero(member == c) for c in picks[b]])
        require(
            np.allclose(bs[b], f1(raw[expanded].sum(axis=0)), rtol=0, atol=1e-13),
            "Expanded human draw differs",
        )
    settings = {}
    boots = {}
    points = {}
    bounds = {}
    out = {}
    kp = {k: n for n, k in enumerate(kk)}
    for k in kk:
        c, r = data.keys[k]
        settings[f"{c}_rep{r}"] = {
            "own_valid": run_metric(data, ids, k),
            "common_valid": run_metric(data, common, k),
            "full_roster_worst": run_metric(data, ids, k, "worst"),
            "full_roster_best": run_metric(data, ids, k, "best"),
        }
    for name, conditions in groups.items():
        indices = [data.kpos[c, r] for c in conditions for r in range(3)]
        zs = [settings[f"{c}_rep{r}"] for c in conditions for r in range(3)]
        points[name] = float(np.mean([v["common_valid"]["macro_f1"] for v in zs]))
        boots[name] = bs[:, [kp[k] for k in indices]].mean(axis=(1, 2))
        bounds[name] = [
            float(np.mean([v["full_roster_" + m]["macro_f1"] for v in zs]))
            for m in ["worst", "best"]
        ]
        pool_means = [
            float(
                np.mean(
                    [
                        settings[f"{c}_rep{r}"]["common_valid"]["macro_f1"]
                        for r in range(3)
                    ]
                )
            )
            for c in conditions
        ]
        out[name] = {
            "macro_f1": points[name],
            "per_label_f1": np.mean(
                [v["common_valid"]["per_label_f1"] for v in zs], axis=0
            ).tolist(),
            "pool_means": pool_means,
            "pool_sample_sd": (
                float(np.std(pool_means, ddof=1)) if len(pool_means) > 1 else None
            ),
            "decoder_values": [
                [settings[f"{c}_rep{r}"]["common_valid"]["macro_f1"] for r in range(3)]
                for c in conditions
            ],
            "decoder_sample_sd_by_pool": [
                float(
                    np.std(
                        [
                            settings[f"{c}_rep{r}"]["common_valid"]["macro_f1"]
                            for r in range(3)
                        ],
                        ddof=1,
                    )
                )
                for c in conditions
            ],
            "full_roster_missing_bounds": bounds[name],
        }
    contrasts = {
        l + "_minus_" + r: effect(points, boots, bounds, {l: 1, r: -1})
        for l in groups
        for r in groups
        if l != r
    }
    return {
        "study": data.study,
        "endpoint": "seven_type_macro_f1",
        "labels": LABELS,
        "original_ids": ids,
        "common_ids": common,
        "excluded_ids": sorted(set(ids) - set(common)),
        "original_n": len(ids),
        "common_n": len(common),
        "cluster_count": w.shape[1],
        "mask": "all-three-valid-votes" if strict else "all-selected-outputs-nonnull",
        "condition_groups": groups,
        "settings": settings,
        "groups": out,
        "contrasts": contrasts,
        "bootstrap": {
            "seed": seed,
            "draws": 10000,
            "generator": "PCG64",
            "unit": "original-text overlap cluster",
        },
    }


def gate_compare(data, ids, groups, seed, strict=False):
    kk = data.chosen(groups)
    common = data.common(ids, kk, strict)
    p = data.indices(common)
    original = data.indices(ids)
    require(
        len(p) > 0 and len(set(data.truth[p])) == 1,
        "Gate diagnostic needs one nonempty gold polarity",
    )
    truth = bool(data.truth[p][0])
    w, member, picks = weights(common, data.roster, seed)
    denominator = w @ np.bincount(member, minlength=w.shape[1])
    points = {}
    boots = {}
    bounds = {}
    out = {}
    settings = {}
    for k in kk:
        c, r = data.keys[k]
        ok = data.valid[k, original]
        errors = data.gate[k, original] != truth
        count = int(errors[ok].sum())
        n = int(ok.sum())
        masked = int((data.gate[k, p] != truth).sum())
        settings[f"{c}_rep{r}"] = {
            "original_n": len(ids),
            "own_valid_n": n,
            "own_errors": count,
            "own_error_rate": count / n if n else None,
            "common_n": len(p),
            "common_errors": masked,
            "common_error_rate": masked / len(p),
            "null_ids": [i for i, v in zip(ids, ok) if not v],
            "full_roster_bounds": [count / len(ids), (count + len(ids) - n) / len(ids)],
        }
    for name, conditions in groups.items():
        indices = [data.kpos[c, r] for c in conditions for r in range(3)]
        zs = [settings[f"{c}_rep{r}"] for c in conditions for r in range(3)]
        error = (data.gate[np.ix_(indices, p)] != truth).mean(axis=0)
        points[name] = float(error.mean())
        clustered = np.zeros(w.shape[1])
        np.add.at(clustered, member, error)
        boots[name] = (w @ clustered) / denominator
        bounds[name] = np.mean([v["full_roster_bounds"] for v in zs], axis=0).tolist()
        out[name] = {
            "error_rate": points[name],
            "ci95": intervals(boots[name]),
            "full_roster_bounds": bounds[name],
            "decoder_values": [
                [settings[f"{c}_rep{r}"]["common_error_rate"] for r in range(3)]
                for c in conditions
            ],
        }
    contrasts = {
        l + "_minus_" + r: effect(points, boots, bounds, {l: 1, r: -1})
        for l in groups
        for r in groups
        if l != r
    }
    return {
        "study": data.study,
        "endpoint": "false_negative_rate" if truth else "false_positive_rate",
        "original_ids": ids,
        "common_ids": common,
        "excluded_ids": sorted(set(ids) - set(common)),
        "original_n": len(ids),
        "common_n": len(common),
        "cluster_count": w.shape[1],
        "mask": "all-three-valid-votes" if strict else "all-selected-outputs-nonnull",
        "condition_groups": groups,
        "settings": settings,
        "groups": out,
        "contrasts": contrasts,
        "bootstrap": {
            "seed": seed,
            "draws": 10000,
            "generator": "PCG64",
            "unit": "original-text overlap cluster",
        },
    }


def main():
    completed = read(REVIEW / "complete.json")
    verify(completed["inputs"])
    verify(completed["outputs"])
    require(not (DEST / "scoring_complete.json").exists(), "Preserve completed scoring")
    exported = list(rows(REVIEW / "predictions.jsonl"))
    outputs = {}

    def put(name, value):
        path = DEST / (name + ".json")
        save(path, value)
        outputs[rel(path)] = sha(path)
        print(
            "Scored "
            + name
            + f": {value['common_n']}/{value['original_n']} common inputs",
            flush=True,
        )

    factor = Data("factorial", exported)
    hate = factor.population(True)
    for name, ids, strict in [
        ("primary", hate, False),
        ("all_three_valid", hate, True),
        ("representatives", representatives(factor, hate), False),
    ]:
        put("factorial/" + name, factorial(factor, ids, strict))
    groups = {c: [c] for c in factor.conditions}
    put(
        "factorial/benign",
        gate_compare(factor, factor.population(False), groups, 202610061),
    )
    human = Data("human_meaning", exported)
    hate = human.population(True)
    benign = human.population(False)
    primary = {p: [f"{p}_s{s}" for s in SEEDS] for p in ["plain", "human"]}
    for name, ids, strict in [
        ("primary", hate, False),
        ("all_three_valid", hate, True),
        ("representatives", representatives(human, hate), False),
    ]:
        put(
            "human_meaning/" + name,
            human_compare(human, ids, primary, 202610042, strict),
        )
    all_groups = groups_human()
    put(
        "human_meaning/all_policy_hateful",
        human_compare(human, hate, all_groups, 202610062),
    )
    put("human_meaning/hateful_gate", gate_compare(human, hate, all_groups, 202610062))
    put("human_meaning/benign", gate_compare(human, benign, all_groups, 202610063))
    for n, source in enumerate(sorted({human.roster[i]["source"] for i in benign})):
        put(
            "human_meaning/benign_" + source,
            gate_compare(
                human,
                [i for i in benign if human.roster[i]["source"] == source],
                all_groups,
                202610064 + n,
            ),
        )
    inputs = {
        rel(REVIEW / "complete.json"): sha(REVIEW / "complete.json"),
        **completed["outputs"],
        rel(__file__): sha(__file__),
        rel(DEST / "analysis_plan.json"): sha(DEST / "analysis_plan.json"),
    }
    for base in STUDIES.values():
        for n in ["roster.jsonl", "scientific_plan_v1.json", "STUDY_PROTOCOL.md"]:
            inputs[rel(base / n)] = sha(base / n)
    save(
        DEST / "scoring_complete.json",
        {
            "status": "scored_requires_independent_check",
            "created_utc": utc(),
            "inputs": inputs,
            "outputs": outputs,
            "new_model_calls": 0,
        },
    )


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket, "create_connection", forbid
    ), threadpool_limits(limits=2):
        main()

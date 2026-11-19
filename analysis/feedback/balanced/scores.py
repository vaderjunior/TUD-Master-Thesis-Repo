from collections import defaultdict
import hashlib
from pathlib import Path
import socket
from types import FunctionType
from unittest.mock import patch
import numpy as np
from analysis.feedback.balanced import archive as r
from analysis.feedback.supplements import scores as base

POLICIES = [
    "reference",
    "error_positive",
    "error_balanced",
    "correct_positive",
    "correct_balanced",
    "random_positive",
    "random_balanced",
]
PRIMARY = {
    p: [f"{p}_s{s}" for s in r.SEEDS] for p in ["error_positive", "error_balanced"]
}
GROUPS = {
    "reference": ["reference"],
    **{p: [f"{p}_s{s}" for s in r.SEEDS] for p in POLICIES[1:]},
}


class Data(base.Data):
    __init__ = FunctionType(
        base.Data.__init__.__code__, vars(base) | {"STUDIES": r.STUDIES}
    )


def representatives(data):
    clusters = defaultdict(list)
    for key in data.ids:
        clusters[data.roster[key]["cluster"]].append(key)
    selected = {
        min(
            v,
            key=lambda k: hashlib.sha256(
                ("20261010-balanced-representative|" + k).encode()
            ).hexdigest(),
        )
        for v in clusters.values()
    }
    return [k for k in data.ids if k in selected]


def source_weights(ids, roster, seed, draws=10000):

    clusters = sorted({roster[k]["cluster"] for k in ids})
    ix = {k: i for i, k in enumerate(clusters)}
    member = np.array([ix[roster[k]["cluster"]] for k in ids])
    sources = defaultdict(set)
    for k in ids:
        sources[roster[k]["cluster"]].add(roster[k]["source"])
    strata = defaultdict(list)
    for key in clusters:
        strata[tuple(sorted(sources[key]))].append(ix[key])
    rng = np.random.Generator(np.random.PCG64(seed))
    picks = []
    for key, positions in sorted(strata.items()):
        p = np.asarray(positions)
        picks.append(p[rng.integers(0, len(p), size=(draws, len(p)))])
    picks = np.concatenate(picks, axis=1)
    w = np.zeros((draws, len(clusters)), dtype=float)
    for n, selected in enumerate(picks):
        w[n] = np.bincount(selected, minlength=len(clusters))
    return w, member, picks


def gate(data, ids, groups, seed, strict=False, coverage=0.95, stratified=False):
    def effect(point, boot, bounds, terms):
        return base.effect(point, boot, bounds, terms, coverage)

    namespace = vars(base) | {"effect": effect}
    if stratified:
        namespace["weights"] = source_weights
    result = FunctionType(base.gate_compare.__code__, namespace)(
        data, ids, groups, seed, strict
    )
    result["bootstrap"]["stratification"] = "source-set" if stratified else "none"
    result["family_interval_coverage"] = coverage
    return result


def binary_run(data, ids, k, missing=None):
    pos = data.indices(ids)
    ok = data.valid[k, pos]
    gold = data.truth[pos]
    pred = data.gate[k, pos].copy()
    if missing is None:
        gold = gold[ok]
        pred = pred[ok]
    else:
        pred[~ok] = (gold if missing == "best" else ~gold)[~ok]
    tp = int(np.sum(gold & pred))
    fp = int(np.sum(~gold & pred))
    fn = int(np.sum(gold & ~pred))
    tn = int(np.sum(~gold & ~pred))
    fpr = fp / (fp + tn) if fp + tn else None
    fnr = fn / (fn + tp) if fn + tp else None
    f = base.f1(np.array([[tn, fn, fp], [tp, fp, fn]]))
    return dict(
        n=len(gold),
        null_ids=[key for key, v in zip(ids, ok) if not v],
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        macro_f1=float(f.mean()),
        balanced_accuracy=(
            1 - (fpr + fnr) / 2 if fpr is not None and fnr is not None else None
        ),
    )


def binary_compare(data, ids, groups, seed, strict=False):
    kk = data.chosen(groups)
    common = data.common(ids, kk, strict)
    pos = data.indices(common)
    r.require(
        len(pos) > 0 and len(set(data.truth[pos])) == 2,
        "Binary metric needs both classes",
    )
    w, member, picks = base.weights(common, data.roster, seed)
    truth = data.truth[pos]
    pred = data.gate[np.ix_(kk, pos)].T
    y = np.stack([~truth, truth], axis=-1)[:, None, :]
    x = np.stack([~pred, pred], axis=-1)
    raw = base.counts(y, x)
    bs = base.bootstrap_counts(raw, w, member)
    f1 = base.f1(bs).mean(axis=-1)

    ba = base.ratio(bs[..., 0], bs[..., 0] + bs[..., 2]).mean(axis=-1)
    for z in [0, 1, 7, 9999]:
        expanded = np.concatenate([np.flatnonzero(member == c) for c in picks[z]])
        r.require(
            np.allclose(
                f1[z],
                base.f1(raw[expanded].sum(axis=0)).mean(axis=-1),
                atol=1e-13,
                rtol=0,
            ),
            "Expanded binary bootstrap differs",
        )
    settings = {}
    kp = {k: i for i, k in enumerate(kk)}
    for k in kk:
        condition, rep = data.keys[k]
        settings[f"{condition}_rep{rep}"] = dict(
            own_valid=binary_run(data, ids, k),
            common_valid=binary_run(data, common, k),
            full_roster_worst=binary_run(data, ids, k, "worst"),
            full_roster_best=binary_run(data, ids, k, "best"),
        )
    endpoints = {}
    for metric, draws in [("macro_f1", f1), ("balanced_accuracy", ba)]:
        points = {}
        boots = {}
        bounds = {}
        out = {}
        for policy, conditions in groups.items():
            ks = [data.kpos[c, rep] for c in conditions for rep in range(3)]
            values = [settings[f"{c}_rep{rep}"] for c in conditions for rep in range(3)]
            points[policy] = float(np.mean([v["common_valid"][metric] for v in values]))
            boots[policy] = draws[:, [kp[k] for k in ks]].mean(axis=1)
            bounds[policy] = [
                float(np.mean([v["full_roster_" + bound][metric] for v in values]))
                for bound in ["worst", "best"]
            ]
            out[policy] = dict(
                mean=points[policy],
                ci95=base.intervals(boots[policy]),
                full_roster_missing_bounds=bounds[policy],
                decoder_values=[
                    [
                        settings[f"{c}_rep{rep}"]["common_valid"][metric]
                        for rep in range(3)
                    ]
                    for c in conditions
                ],
            )
        contrasts = {
            l + "_minus_" + rr: base.effect(points, boots, bounds, {l: 1, rr: -1})
            for l in groups
            for rr in groups
            if l != rr
        }
        endpoints[metric] = dict(groups=out, contrasts=contrasts)
    return dict(
        study=data.study,
        endpoint="combined_binary",
        original_ids=ids,
        common_ids=common,
        excluded_ids=sorted(set(ids) - set(common)),
        original_n=len(ids),
        common_n=len(common),
        cluster_count=w.shape[1],
        condition_groups=groups,
        settings=settings,
        endpoints=endpoints,
        mask="all-three-valid-votes" if strict else "all-57-nonnull",
        bootstrap=dict(
            seed=seed,
            draws=10000,
            generator="PCG64",
            unit="whole original text cluster",
        ),
    )


def deletion_diagnostics(data, primary):

    out = {}
    for endpoint, scored in primary.items():
        ids = scored["common_ids"]
        pos = data.indices(ids)
        truth = bool(data.truth[pos][0])
        cases = []
        for seed in r.SEEDS:
            kept = [s for s in r.SEEDS if s != seed]
            values = {}
            for policy in PRIMARY:
                kk = [
                    data.kpos[f"{policy}_s{s}", rep] for s in kept for rep in range(3)
                ]
                values[policy] = float((data.gate[np.ix_(kk, pos)] != truth).mean())
            cases.append(
                dict(
                    omitted_pool_seed=seed,
                    common_n=len(ids),
                    rates=values,
                    difference=values["error_balanced"] - values["error_positive"],
                )
            )
        stages = sorted({data.records[i, "reference", 0]["stage"] for i in ids})
        batches = []
        for stage in stages:
            selected = [
                i for i in ids if data.records[i, "reference", 0]["stage"] != stage
            ]
            p = data.indices(selected)
            values = {}
            for policy, conditions in PRIMARY.items():
                kk = [data.kpos[c, rep] for c in conditions for rep in range(3)]
                values[policy] = float((data.gate[np.ix_(kk, p)] != truth).mean())
            batches.append(
                dict(
                    omitted_stage=stage,
                    common_n=len(selected),
                    rates=values,
                    difference=values["error_balanced"] - values["error_positive"],
                )
            )
        out[endpoint] = dict(
            leave_one_pool_out=cases,
            leave_one_batch_out=batches,
            scope="Supporting point estimates; not new primary tests or evidence of independent pools.",
        )
    return out


def main():
    done = r.read(r.REVIEW / "complete.json")
    r.verify(done["inputs"])
    r.verify(done["outputs"])
    r.require(
        not (r.DEST / "scoring_complete.json").exists(), "Preserve completed scores"
    )
    sources = {
        r.rel(Path(__file__)): r.sha(Path(__file__)),
        r.rel(r.ROOT / "tests/feedback/balanced/test_scores.py"): r.sha(
            r.ROOT / "tests/feedback/balanced/test_scores.py"
        ),
    }
    r.save(
        r.DEST / "scoring_plan.json",
        dict(
            created_utc=r.utc(),
            analysis_plan_sha256=r.sha(r.DEST / "analysis_plan.json"),
            inputs=sources,
            scope="Implement existing two primary outcomes and declared supporting/sensitivity specifications. No selected best-pool/arm or new calls.",
        ),
    )
    data = Data("balanced_corrections", list(r.rows(r.REVIEW / "predictions.jsonl")))
    outputs = {}

    def put(name, value):
        path = r.DEST / (name + ".json")
        r.save(path, value)
        outputs[r.rel(path)] = r.sha(path)
        print(
            "Scored",
            name,
            (
                f"{value['common_n']}/{value['original_n']} common"
                if "common_n" in value
                else ""
            ),
            flush=True,
        )

    selected = set(representatives(data))
    primaries = {}
    for truth, name, seed in [
        (False, "benign", 202610044),
        (True, "hateful", 202610045),
    ]:
        ids = data.population(truth)
        for suffix, roster, strict in [
            ("primary", ids, False),
            ("all_three_valid", ids, True),
            ("representatives", [k for k in ids if k in selected], False),
        ]:
            result = gate(data, roster, PRIMARY, seed, strict, coverage=0.975)
            put(name + "/" + suffix, result)
            if suffix == "primary":
                primaries[name] = result
        put(name + "/all_policies", gate(data, ids, GROUPS, 202610101 + truth))
        put(
            name + "/all_policies_strict",
            gate(data, ids, GROUPS, 202610101 + truth, True),
        )
    benign = data.population(False)
    put(
        "benign/source_stratified",
        gate(data, benign, GROUPS, 202610103, stratified=True),
    )
    for index, source in enumerate(sorted({data.roster[k]["source"] for k in benign})):
        ids = [k for k in benign if data.roster[k]["source"] == source]
        put("benign/source_" + source, gate(data, ids, GROUPS, 202610110 + index))
    put("binary/all_policies", binary_compare(data, data.ids, GROUPS, 202610104))
    put(
        "binary/all_policies_strict",
        binary_compare(data, data.ids, GROUPS, 202610104, True),
    )
    for suffix, ids, strict in [
        ("all_policies", data.population(True), False),
        ("all_policies_strict", data.population(True), True),
        ("representatives", [k for k in data.population(True) if k in selected], False),
    ]:
        put("types/" + suffix, base.human_compare(data, ids, GROUPS, 202610105, strict))
    put("supporting_deletions", deletion_diagnostics(data, primaries))
    r.verify(sources)
    r.save(
        r.DEST / "scoring_complete.json",
        dict(
            created_utc=r.utc(),
            status="scored_requires_independent_check",
            inputs={
                **sources,
                r.rel(r.REVIEW / "complete.json"): r.sha(r.REVIEW / "complete.json"),
                **done["outputs"],
                r.rel(r.DEST / "analysis_plan.json"): r.sha(
                    r.DEST / "analysis_plan.json"
                ),
                r.rel(r.DEST / "scoring_plan.json"): r.sha(
                    r.DEST / "scoring_plan.json"
                ),
            },
            outputs=outputs,
            new_model_calls=0,
        ),
    )


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    with patch.object(socket.socket, "connect", r.forbid), patch.object(
        socket, "create_connection", r.forbid
    ), threadpool_limits(limits=2):
        main()

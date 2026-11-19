from analysis.classification.micro_f1 import supervised as r
from collections import defaultdict
from datetime import datetime, timezone
import argparse
import re
import traceback
import unicodedata
import numpy as np

b = r.b
c = b.c
ROOT = b.ROOT
OLD = r.OLD


def fitting(row, labels, macro):
    z = np.array(
        [[row["per_label"][label][k] for k in ("tp", "fp", "fn")] for label in labels],
        np.int64,
    )
    idx = [labels.index(x) for x in row.get("macro_labels", macro)]
    s = b.score(z, idx)
    b.equal(s[0], row["macro_f1"], "earlier saved per-fit macro")
    den = 2 * z[:, 0] + z[:, 1] + z[:, 2]
    weights = den / den.sum() if den.sum() else np.zeros(len(labels))
    out = dict(
        seed=row.get("seed"),
        scores=dict(zip(b.METRICS, s)),
        tp_fp_fn=z,
        per_label_f1=b.metric.label_f1(z),
        micro_weights=weights,
        precision=(
            float(z[:, 0].sum() / (z[:, 0] + z[:, 1]).sum())
            if (z[:, 0] + z[:, 1]).sum()
            else 0.0
        ),
        recall=(
            float(z[:, 0].sum() / (z[:, 0] + z[:, 2]).sum())
            if (z[:, 0] + z[:, 2]).sum()
            else 0.0
        ),
        quality={
            k: row[k]
            for k in (
                "exact_set_accuracy",
                "empty_prediction_rate",
                "mean_predicted_labels",
                "mean_gold_labels",
                "mean_extra_labels",
                "mean_missing_labels",
            )
            if k in row
        },
    )
    if "head_only" in row:
        out["head_only"] = fitting(row["head_only"], labels, macro)
    return out


def points(study):
    result = []
    for u in r.units(study):
        old = c.read(ROOT / u["source"])["cells"][u["endpoint"]]
        labels = u["labels"]
        macro = u["macro_labels"]
        rows = []
        means = {}
        pools = {}
        if study == "matched_type":
            for pool, methods in old["runs"].items():
                pools[pool] = {}
                for method, rr in methods.items():
                    fits = [
                        dict(pool=pool, method=method, **fitting(v, labels, macro))
                        for v in rr
                    ]
                    rows.extend(fits)
                    pools[pool][method] = {
                        k: float(np.mean([v["scores"][k] for v in fits]))
                        for k in b.METRICS
                    }
                    b.equal(
                        pools[pool][method]["macro"],
                        old["pool_means"][pool][method]["macro_f1"],
                        "earlier matched pool mean",
                    )
            for group, mm in old["group_means"].items():
                pp = [
                    p
                    for p in pools
                    if p.startswith("balanced_" if group == "balanced67" else "equal_")
                ]
                c.require(len(pp) == 5, "Matched pool count")
                for method, v in mm.items():
                    means[group + "/" + method] = {
                        k: float(np.mean([pools[p][method][k] for p in pp]))
                        for k in b.METRICS
                    }
                    b.equal(
                        means[group + "/" + method]["macro"],
                        v["macro_f1"],
                        "earlier matched group mean",
                    )
        else:
            for method, rr in old["runs"].items():
                fits = [dict(method=method, **fitting(v, labels, macro)) for v in rr]
                rows.extend(fits)
                means[method] = {
                    k: float(np.mean([v["scores"][k] for v in fits])) for k in b.METRICS
                }
                b.equal(
                    means[method]["macro"], old["means"][method], "earlier method mean"
                )
        result.append(
            dict(
                study=study,
                endpoint=u["endpoint"],
                n=old["n"],
                labels=labels,
                macro_labels=macro,
                fit_rows=rows,
                means=means,
                pools=pools,
            )
        )
    return result


def gold(name):
    plan = c.read(OLD / "compact_broad_20261008_v1/plan.json")["endpoints"][name]
    roster = c.lines(
        OLD / "offline_closure_20261006_v1/evaluation_preparation/broad_roster.jsonl"
    )
    field = {"hate_type": "hate_types", "target_group": "target_groups"}[
        plan["dimension"]
    ]
    rows = [
        v
        for v in roster
        if v["phase"] == name.split("/")[0]
        and v.get("gate") is True
        and v.get(field) is not None
    ]
    c.require(
        [v["key"] for v in rows] == plan["scorable_ids"],
        "Earlier gold identities/order",
    )
    g = np.asarray(
        [[label in v[field] for label in plan["all_labels"]] for v in rows], np.uint8
    )
    return plan, rows, g


def verify_cube(cube, g, labels, rows, models):
    b.equal(
        cube[..., 0] + cube[..., 2],
        np.broadcast_to(g, cube.shape[:-1]),
        "earlier per-item gold",
    )
    for i, (method, seed) in enumerate(models):
        fit = next(x for x in rows if x["method"] == method and x["seed"] == seed)
        b.equal(cube[i].sum(0), fit["tp_fp_fn"], "earlier raw/count totals")


def simple(study, folder):
    cells = points(study)
    root = (ROOT / r.units(study)[0]["source"]).parent
    stats = c.read(root / "statistics.json")
    raw = b.arrays(root / "paired_counts.npz")
    comparisons = []
    drawrefs = b.arrays(root / "bootstrap_draws.npz") if study == "shorter" else {}
    for name in ("en_test/hate_type", "larger_dev/target_group"):
        ep, rr, g = gold(name)
        idx = [ep["all_labels"].index(l) for l in ep["macro_labels"]]
        cell = next(x for x in cells if x["endpoint"] == "broad/" + name)
        if study == "compact_tuning":
            pairs = [
                (
                    "tuned_selected",
                    list(range(10, 15)),
                    "bge",
                    [42, 43, 44],
                    "primary",
                    2,
                    20261008,
                )
            ]
        elif study == "shorter":
            pairs = [
                (
                    "expanded_tuned_selected",
                    list(range(10, 15)),
                    "original_tuned_selected",
                    list(range(10, 15)),
                    "primary",
                    2,
                    20261008,
                ),
                (
                    "expanded_tuned_selected",
                    list(range(10, 15)),
                    "bge",
                    [42, 43, 44],
                    "supporting_BGE",
                    None,
                    20261008,
                ),
            ]
        else:
            pairs = [
                (
                    "full_bge_matched_selected",
                    [42],
                    "full_neural_selected",
                    [0, 1, 2],
                    "primary",
                    2,
                    20261012,
                )
            ]
        for left, ls, right, rs, label, family, seed in pairs:
            models = [(left, s) for s in ls] + [(right, s) for s in rs]
            cube = np.stack([raw[f"{name}|{m}|{s}"] for m, s in models])
            verify_cube(cube, g, ep["all_labels"], cell["fit_rows"], models)
            point = b.score(cube.sum(1), idx)
            delta = point[: len(ls)].mean(0) - point[len(ls) :].mean(0)
            variations = {}
            for stratified in (False, True) if label == "primary" else (False,):
                tag = "source_stratified" if stratified else "unstratified"
                draw, info = b.bootstrap(
                    cube,
                    [v["text_cluster"] for v in rr],
                    idx,
                    seed,
                    sources=[v["source"] for v in rr] if stratified else None,
                )
                dd = draw[:, : len(ls)].mean(1) - draw[:, len(ls) :].mean(1)
                if study == "compact_tuning":
                    old = stats["contrasts"][name][
                        "source_stratified" if stratified else "primary"
                    ]
                elif study == "shorter":
                    old = stats["contrasts"][name][label][tag]
                    b.equal(
                        dd[:, 0],
                        drawrefs[name + "|" + label + "|" + tag],
                        "earlier shorter saved macro draws",
                    )
                else:
                    old = stats["primary"][name][
                        "source_stratified" if stratified else "primary"
                    ]
                output = r.compare(delta, dd, old, family if not stratified else None)
                fname = (
                    name.replace("/", "__") + "__" + label + "__" + tag + "_draws.npz"
                )
                b.array(folder / fname, scores=draw, difference=dd)
                variations[tag] = dict(
                    metrics=output, bootstrap=info, draws=c.rel(folder / fname)
                )
            comparisons.append(
                dict(
                    study=study,
                    endpoint=name,
                    left=left,
                    right=right,
                    role=label,
                    family=family,
                    question_id=study + "/" + name if family else None,
                    left_scores=dict(zip(b.METRICS, point[: len(ls)].mean(0))),
                    right_scores=dict(zip(b.METRICS, point[len(ls) :].mean(0))),
                    variations=variations,
                )
            )
    if study == "native_cpu":
        name = "en_hatecheck_eval/target_group"
        ep, rr, g = gold(name)
        idx = [ep["all_labels"].index(l) for l in ep["macro_labels"]]
        cell = next(x for x in cells if x["endpoint"] == "broad/" + name)
        models = [("full_bge_matched_selected", 42)] + [
            ("full_neural_selected", i) for i in range(3)
        ]
        cube = np.stack([raw[f"{name}|{m}|{s}"] for m, s in models])
        verify_cube(cube, g, ep["all_labels"], cell["fit_rows"], models)
        pt = b.score(cube.sum(1), idx)
        delta = pt[0] - pt[1:].mean(0)
        vv = {}
        for tag, clusters in [
            ("functional", [x["functional_cluster"] for x in rr]),
            ("item_sensitivity", [x["key"] for x in rr]),
        ]:
            dd, info = b.bootstrap(cube, clusters, idx, 20261012)
            diff = dd[:, 0] - dd[:, 1:].mean(1)
            output = r.compare(delta, diff, stats["exploratory_functional"][name][tag])
            fn = folder / ("hatecheck_" + tag + "_draws.npz")
            b.array(fn, scores=dd, difference=diff)
            vv[tag] = dict(metrics=output, bootstrap=info, draws=c.rel(fn))
        comparisons.append(
            dict(
                study=study,
                endpoint=name,
                left=models[0][0],
                right=models[1][0],
                role="supporting_functional",
                family=None,
                question_id=None,
                left_scores=dict(zip(b.METRICS, pt[0])),
                right_scores=dict(zip(b.METRICS, pt[1:].mean(0))),
                variations=vv,
            )
        )
    b.save(
        folder / "results.json", dict(study=study, cells=cells, contrasts=comparisons)
    )
    return [c.rel(folder / "results.json")]


def matched(folder):
    cells = points("matched_type")
    cell = cells[0]
    root = (ROOT / r.units("matched_type")[0]["source"]).parent
    stats = c.read(root / "statistics.json")
    raw = b.arrays(root / "primary_counts.npz")
    ep, rr, g = gold("en_test/hate_type")
    idx = [ep["all_labels"].index(v) for v in ep["macro_labels"]]
    clusters = [
        re.sub(r"\s+", " ", unicodedata.normalize("NFKC", x["text"]).casefold()).strip()
        for x in rr
    ]
    sources = [x["source"] for x in rr]
    contrasts = []
    for group in ("balanced67", "equal67"):
        pools = sorted(
            p
            for p in cell["pools"]
            if p.startswith("balanced_" if group == "balanced67" else "equal_")
        )
        models = [
            (p, m, s)
            for p in pools
            for m, ss in [
                ("tuned_selected", range(10, 15)),
                ("bge", [None]),
                ("frozen_roberta", [None]),
            ]
            for s in ss
        ]
        cube = np.stack([raw[f"{p}|{m}|{s}"] for p, m, s in models])
        b.equal(
            cube[..., 0] + cube[..., 2],
            np.broadcast_to(g, cube.shape[:-1]),
            "matched raw gold",
        )
        for i, (p, m, s) in enumerate(models):
            b.equal(
                cube[i].sum(0),
                next(
                    x["tp_fp_fn"]
                    for x in cell["fit_rows"]
                    if (x["pool"], x["method"], x["seed"]) == (p, m, s)
                ),
                "matched raw count totals",
            )
        pt = b.score(cube.sum(1), idx).reshape(5, 7, 3)
        pd = np.stack([pt[:, :5].mean(1) - pt[:, 5], pt[:, 6] - pt[:, 5]], 1)
        entries = [
            dict(
                study="matched_type",
                endpoint="en_test/hate_type",
                group=group,
                left=m,
                right="bge",
                family=4,
                question_id="matched_type/en_test/hate_type|"
                + group
                + "|"
                + m
                + "_minus_bge",
                left_scores=dict(
                    zip(
                        b.METRICS,
                        pt[:, :5].mean((0, 1)) if i == 0 else pt[:, 6].mean(0),
                    )
                ),
                right_scores=dict(zip(b.METRICS, pt[:, 5].mean(0))),
                per_pool=[
                    dict(pool=p, difference=dict(zip(b.METRICS, pd[j, i])))
                    for j, p in enumerate(pools)
                ],
                variations={},
            )
            for i, m in enumerate(("tuned_selected", "frozen_roberta"))
        ]
        for stratified in (False, True):
            tag = "source_stratified" if stratified else "unstratified"
            draw, info = b.bootstrap(
                cube, clusters, idx, 20261012, sources=sources if stratified else None
            )
            z = draw.reshape(10000, 5, 7, 3)
            diff = np.stack(
                [z[:, :, :5].mean(2) - z[:, :, 5], z[:, :, 6] - z[:, :, 5]], 2
            )
            dd = diff.mean(1)
            picks = np.random.default_rng(20261013).integers(0, 5, size=(10000, 5))
            joint = np.take_along_axis(diff, picks[:, :, None, None], axis=1).mean(1)
            fn = folder / (group + "_" + tag + "_draws.npz")
            b.array(fn, scores=draw, difference=dd, joint=joint)
            for i, entry in enumerate(entries):
                key = entry["left"] + "_minus_bge"
                old = stats["groups"][group][
                    "source_stratified_sensitivity" if stratified else "primary"
                ][key]
                val = r.compare(
                    pd[:, i].mean(0), dd[:, i], old, 4 if not stratified else None
                )
                b.equal(pd[:, i, 0], old["by_pool"], "matched per-pool original delta")
                b.equal(
                    b.ci(joint[:, i, 0]),
                    old["joint_pool_text_ci95"],
                    "matched original joint-pool interval",
                )
                entry["variations"][tag] = dict(
                    metrics=val,
                    bootstrap=info,
                    joint_pool_text_ci95={
                        k: b.ci(joint[:, i, j]).tolist()
                        for j, k in enumerate(b.METRICS)
                    },
                    draws=c.rel(fn),
                )
        contrasts.extend(entries)
    b.save(
        folder / "results.json",
        dict(study="matched_type", cells=cells, contrasts=contrasts),
    )
    return [c.rel(folder / "results.json")]


def combined(folder):
    outputs = []
    for u in r.units("combined_context"):
        if u["dimension"] == "legal" or u["endpoint"].startswith("de_"):
            continue
        meta = c.read(ROOT / u["metadata_path"])
        old = c.read(ROOT / u["source"])
        a = b.arrays(ROOT / u["prediction_path"])
        g = a["gold"]
        p = a["predictions"]
        v = a["valid"]
        idx = meta["macro_indices"]
        models = meta["models"]
        data = {}
        c.require(idx, "English combined macro labels missing")
        for population in ("own_valid", "common_nine"):
            fits = []
            groups = {}
            poolrows = {}
            for i, m in enumerate(models):
                mask = v[i].astype(bool) & (
                    a["common"].astype(bool) if population == "common_nine" else True
                )
                z = b.counts(g[mask], p[i : i + 1, mask]).sum(1)[0]
                s = b.score(z, idx)
                prior = old[population]["runs"][i]
                b.equal(s[0], prior["macro_f1"], "combined original per-fit macro")
                for j, k in enumerate(("tp", "fp", "fn")):
                    b.equal(z[:, j], prior[k], "combined per-label count")
                row = dict(
                    model=m,
                    n=int(mask.sum()),
                    scores=dict(zip(b.METRICS, s)),
                    tp_fp_fn=z,
                    per_label_f1=b.metric.label_f1(z),
                    set_exact=float((p[i, mask] == g[mask]).all(-1).mean()),
                    complete_field=(
                        float(
                            (
                                (p[i, mask] == g[mask]).all(-1)
                                & a["gate"][i, mask].astype(bool)
                            ).mean()
                        )
                        if m["gate_available"]
                        else None
                    ),
                )
                fits.append(row)
            for group in sorted({m["group"] for m in models}):
                pp = sorted({m["pool"] for m in models if m["group"] == group})
                pv = {}
                for pool in pp:
                    ix = [
                        i
                        for i, m in enumerate(models)
                        if m["group"] == group and m["pool"] == pool
                    ]
                    pv[pool] = {
                        k: float(np.mean([fits[i]["scores"][k] for i in ix]))
                        for k in b.METRICS
                    }
                sv = {
                    k: float(np.mean([pv[pool][k] for pool in pp])) for k in b.METRICS
                }
                b.equal(
                    sv["macro"],
                    old[population]["groups"][group]["mean"]["macro_f1"],
                    "combined original equal-pool mean",
                )
                groups[group] = dict(
                    scores=sv,
                    pools=pv,
                    pool_count=len(pp),
                    fits=sum(m["group"] == group for m in models),
                )
            data[population] = dict(fit_rows=fits, groups=groups)
        path = folder / (u["endpoint"].replace("/", "__") + "_results.json")
        b.save(
            path,
            dict(
                study="combined_context",
                endpoint=u["endpoint"],
                labels=meta["labels"],
                macro_labels=meta["macro_labels"],
                common_nine_n=int(a["common"].sum()),
                **data,
            ),
        )
        outputs.append(c.rel(path))
    return outputs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "study",
        choices=[
            "compact_tuning",
            "shorter",
            "native_cpu",
            "matched_type",
            "combined_context",
        ],
    )
    args = ap.parse_args()
    folder = (
        b.BASE
        / "attempts"
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "_" + args.study)
    )
    folder.mkdir(parents=True)
    try:
        output = (
            matched(folder)
            if args.study == "matched_type"
            else (
                combined(folder)
                if args.study == "combined_context"
                else simple(args.study, folder)
            )
        )
        b.save(
            folder / "complete.json",
            dict(
                status="passed",
                study=args.study,
                outputs=output,
                sources=c.BINDINGS,
                checks=dict(b.CHECKS),
                errors=dict(b.ERRORS),
            ),
        )
        print("SAVED", c.rel(folder), flush=True)
    except Exception:
        b.save(
            folder / "failure.json",
            dict(
                traceback=traceback.format_exc(),
                sources=c.BINDINGS,
                checks=dict(b.CHECKS),
            ),
        )
        raise


if __name__ == "__main__":
    main()

from analysis.classification.micro_f1 import common as b
from datetime import datetime, timezone
import argparse
import traceback
import numpy as np

c = b.c
ROOT = b.ROOT
OLD = ROOT / "experiments/astra"


def units(study):
    return [u for u in b.REG["units"] if u["study"] == study]


def fit_summaries(g, p, models, idx, gates=None):
    cube = b.counts(g, p)
    z = cube.sum(1)
    scores = b.score(z, idx)
    tp, fp, fn = z.transpose(2, 0, 1)
    den = 2 * tp + fp + fn
    weights = np.divide(
        den,
        den.sum(1, keepdims=True),
        out=np.zeros_like(den, float),
        where=den.sum(1, keepdims=True) > 0,
    )
    pp = (tp + fp).sum(1)
    gg = (tp + fn).sum(1)
    pr = np.divide(tp.sum(1), pp, out=np.zeros(len(p)), where=pp > 0)
    rc = np.divide(tp.sum(1), gg, out=np.zeros(len(p)), where=gg > 0)
    exact = (p == g).all(-1)
    quality = dict(
        precision=pr,
        recall=rc,
        set_exact=exact.mean(-1),
        mean_predicted=p.sum(-1).mean(-1),
        mean_gold=np.repeat(g.sum(-1).mean(), len(p)),
    )
    if gates is not None:
        quality.update(
            complete_field=(exact & gates.astype(bool)).mean(-1),
            gate_miss=(~gates.astype(bool)).mean(-1),
        )
    rows = [
        dict(
            model=m,
            scores=dict(zip(b.METRICS, scores[i])),
            tp_fp_fn=z[i],
            per_label_f1=b.metric.label_f1(z[i]),
            micro_weights=weights[i],
            **{k: float(v[i]) for k, v in quality.items()}
        )
        for i, m in enumerate(models)
    ]
    return scores, rows, quality, cube


def compare(point, draw, old=None, family=None):
    out = b.interval(point, draw, family)
    if old:
        val = old.get("difference", old.get("value"))
        b.equal(point[0], val, "old contrast point")
        ci = old.get(
            "ci95_descriptive", old.get("ci95", [old.get("low"), old.get("high")])
        )
        b.equal(out["macro"]["ci95"], ci, "old contrast 95 interval")
        confidence = old.get("confidence", 0.9875 if "ci9875" in old else None)
        adjusted = old.get("family_interval", old.get("ci9875"))
        if adjusted is not None and confidence is not None:
            b.equal(
                b.ci(draw[:, 0], confidence), adjusted, "old adjusted macro interval"
            )
        out["macro"]["original_adjusted_interval"] = adjusted
        out["macro"]["original_adjusted_confidence"] = confidence
    return out


def gpu(folder):
    results = []
    primary = c.read(
        ROOT / units("gpu")[0]["source"].rsplit("/", 1)[0] / "primary_comparisons.json"
    )
    for u in units("gpu"):
        meta = c.read(ROOT / u["metadata_path"])
        old = c.read(ROOT / u["source"])
        a = b.arrays(ROOT / u["prediction_path"])
        p = a["predictions"]
        g = a["truth"]
        idx = meta["supported_indices"]
        models = meta["models"]
        slug = u["endpoint"].replace("/", "__")
        out = folder / slug
        point, rows, quality, cube = fit_summaries(g, p, models, idx)
        oldq = b.arrays(ROOT / u["count_path"])
        b.equal(point[:, 0], oldq["scores"], "GPU original model scores")
        for j, k in enumerate(("tp", "fp", "fn")):
            b.equal(cube.sum(1)[..., j], oldq[k], "GPU original label counts")
        draw, info = b.bootstrap(cube, meta["clusters"], idx, old["bootstrap"]["seed"])
        del cube
        orig = b.arrays((ROOT / u["source"]).parent / (slug + "_bootstrap.npz"))[
            "scores"
        ]
        b.equal(draw[..., 0], orig, "all GPU original macro draws")
        c.require(
            info["picks_sha256"] == old["bootstrap"]["picks_sha256"],
            "GPU RNG stream differs",
        )
        pp, bp = b.aggregate(models, point)
        pd, bd = b.aggregate(models, draw)
        contrasts = []
        absolute = []
        poolrows = []
        omissions = []
        seedrows = []
        for row in old["absolute"]:
            key = row["budget"], row["method"]
            b.equal(bp[key][0], row["mean"], "GPU original budget mean")
            b.equal(
                b.ci(bd[key][:, 0]),
                row["descriptive_ci95"],
                "GPU original absolute interval",
            )
            absolute.append(
                dict(budget=key[0], method=key[1], metrics=b.interval(bp[key], bd[key]))
            )
        for key, values in pp.items():
            poolrows.append(
                dict(pool=key[0], method=key[1], scores=dict(zip(b.METRICS, values)))
            )
        for oldpair in primary["comparisons"]:
            if oldpair["endpoint"] != u["endpoint"]:
                continue
            size = oldpair["budget"]
            left = (
                "tuned_selected"
                if oldpair["contrast"].startswith("tuned")
                else "frozen_roberta_selected_fold42"
            )
            right = "bge_selected_fold42"
            delta = bp[size, left] - bp[size, right]
            dd = bd[size, left] - bd[size, right]
            contrasts.append(
                dict(
                    endpoint=u["endpoint"],
                    budget=size,
                    left=left,
                    right=right,
                    metrics=compare(delta, dd, oldpair, 12),
                    left_scores=dict(zip(b.METRICS, bp[size, left])),
                    right_scores=dict(zip(b.METRICS, bp[size, right])),
                )
            )
            pools = sorted(
                {
                    m["pool"]
                    for m in models
                    if m["budget"] == size and m["method"] == left
                }
            )
            for omitted in pools:
                kept = [p for p in pools if p != omitted]
                lo = np.mean([pp[p, left] - pp[p, right] for p in kept], 0)
                omissions.append(
                    dict(
                        budget=size,
                        left=left,
                        right=right,
                        omitted=omitted,
                        difference=dict(zip(b.METRICS, lo)),
                    )
                )
            for i, m in enumerate(models):
                if m["budget"] == size and m["method"] == left:
                    seedrows.append(
                        dict(
                            **m,
                            difference_to_matched_pool_BGE=dict(
                                zip(b.METRICS, point[i] - pp[m["pool"], right])
                            )
                        )
                    )
        qmeans = {}
        for k, v in quality.items():
            _, ag = b.aggregate(models, v[:, None])
            qmeans[k] = {
                str(size) + "/" + method: float(q[0])
                for (size, method), q in ag.items()
            }
        result = dict(
            study="gpu",
            endpoint=u["endpoint"],
            labels=meta["labels"],
            macro_labels=meta["macro_labels"],
            n=len(g),
            fit_rows=rows,
            absolute=absolute,
            per_pool=poolrows,
            contrasts=contrasts,
            omissions=omissions,
            seed_comparisons=seedrows,
            quality_budget=qmeans,
            bootstrap=info,
        )
        b.save(out / "results.json", result)
        b.array(out / "draws.npz", scores=draw)
        b.save(out / "metadata.json", meta)
        b.array(out / "predictions.npz", truth=g, predictions=p)
        results.append(c.rel(out / "results.json"))
        print("GPU COMPLETE", u["endpoint"], flush=True)
    return results


def controls(folder):
    results = []
    for u in units("gpu_controls"):
        meta = c.read(ROOT / u["metadata_path"])
        old = c.read(ROOT / u["source"])
        a = b.arrays(ROOT / u["prediction_path"])
        g = a["truth"]
        p = a["predictions"]
        models = meta["models"]
        idx = meta["supported_indices"]
        slug = u["endpoint"].replace("/", "__")
        out = folder / slug
        point, rows, quality, cube = fit_summaries(g, p, models, idx, a["gates"])
        draw, info = b.bootstrap(cube, meta["clusters"], idx, old["bootstrap"]["seed"])
        del cube
        orig = b.arrays((ROOT / u["source"]).parent / (slug + "_bootstrap.npz"))[
            "scores"
        ]
        b.equal(draw[..., 0], orig, "all cutoff-control original macro draws")
        c.require(
            info["picks_sha256"] == old["bootstrap"]["picks_sha256"],
            "Control RNG differs",
        )
        pp, bp = b.aggregate(models, point, True)
        pd, bd = b.aggregate(models, draw, True)
        absolute = []
        contrasts = []
        components = []
        for key, oldvalue in old["absolute"].items():
            size, method = key.split("/", 1)
            k = int(size), method
            b.equal(bp[k][0], oldvalue["value"], "control original absolute")
            b.equal(
                b.ci(bd[k][:, 0]),
                [oldvalue["low"], oldvalue["high"]],
                "control original absolute interval",
            )
            absolute.append(
                dict(budget=int(size), method=method, metrics=b.interval(bp[k], bd[k]))
            )
        for x in old["comparisons"]:
            ka, kb = (x["budget"], x["after"]), (x["budget"], x["before"])
            contrasts.append(
                dict(
                    endpoint=u["endpoint"],
                    budget=x["budget"],
                    left=x["after"],
                    right=x["before"],
                    metrics=compare(bp[ka] - bp[kb], bd[ka] - bd[kb], x),
                    left_scores=dict(zip(b.METRICS, bp[ka])),
                    right_scores=dict(zip(b.METRICS, bp[kb])),
                )
            )

        def decompose(a, g, h, z):
            return dict(
                total=z - a,
                head=((h - a) + (z - g)) / 2,
                gate=((g - a) + (z - h)) / 2,
                interaction=z - h - g + a,
                head_first=h - a,
                head_second=z - g,
                gate_first=g - a,
                gate_second=z - h,
            )

        for oldcomp in old["cutoff_components"]:
            size, family = oldcomp["budget"], oldcomp["family"]
            keys = [
                (size, family + "/" + v)
                for v in ("default", "gate_only", "head_only", "selected")
            ]
            value = decompose(*(bp[k] for k in keys))
            boot = decompose(*(bd[k] for k in keys))
            entry = dict(budget=size, family=family)
            for key in value:
                entry[key] = compare(value[key], boot[key], oldcomp[key])
            components.append(entry)
        qb = {}
        for key, values in quality.items():
            _, vv = b.aggregate(models, values[:, None], True)
            qb[key] = {
                str(size) + "/" + method: float(v[0])
                for (size, method), v in vv.items()
            }
        b.save(
            out / "results.json",
            dict(
                study="gpu_controls",
                endpoint=u["endpoint"],
                n=len(g),
                labels=meta["labels"],
                macro_labels=[meta["labels"][i] for i in idx],
                fit_rows=rows,
                absolute=absolute,
                contrasts=contrasts,
                cutoff_components=components,
                per_pool=[
                    dict(pool=k[0], method=k[1], scores=dict(zip(b.METRICS, v)))
                    for k, v in pp.items()
                ],
                quality_budget=qb,
                bootstrap=info,
            ),
        )
        b.array(out / "draws.npz", scores=draw)
        b.save(out / "metadata.json", meta)
        results.append(c.rel(out / "results.json"))
        print("CONTROLS COMPLETE", u["endpoint"], flush=True)
    return results


def complete_for(study):
    paths = []
    for p in b.BASE.glob("attempts/*/complete.json"):
        d = c.read(p)
        if d["study"] == study:
            paths.append(d)
    c.require(len(paths) == 1, "Need one completed " + study)
    return paths[0]


def five(folder):
    u = units("five_fit_update")[0]
    old = c.read(ROOT / u["source"])
    meta = c.read(ROOT / u["metadata_path"])
    a = b.arrays(ROOT / u["prediction_path"])
    g = a["truth"]
    idx = old["supported_indices"]
    p = np.concatenate([a["old"], a["new"]])
    models = [
        dict(method=side, seed=s) for side in ("old", "new") for s in range(10, 15)
    ]
    point, rows, q, cube = fit_summaries(g, p, models, idx)
    draw, info = b.bootstrap(cube, meta["clusters"], idx, old["bootstrap"]["seed"])
    del cube
    b.equal(draw[:, :5, 0], a["old_draws"], "all five-fit old macro draws")
    b.equal(draw[:, 5:, 0], a["new_draws"], "all five-fit new macro draws")
    b.equal(point[:5, 0], a["old_scores"], "five old scores")
    b.equal(point[5:, 0], a["new_scores"], "five new scores")
    original = point[:5].mean(0)
    updated = point[5:].mean(0)
    change = updated - original
    dd = draw[:, 5:].mean(1) - draw[:, :5].mean(1)
    b.equal(
        change[0], old["changed_pool"]["change"], "original affected-pool macro change"
    )
    b.equal(
        b.ci(dd[:, 0]),
        old["changed_pool"]["change_ci95"],
        "original affected-pool interval",
    )
    path = next(p for p in complete_for("gpu")["outputs"] if "en_test__hate_type" in p)
    base = (ROOT / path).parent
    main = c.read(path)
    mm = c.read(base / "metadata.json")
    md = b.arrays(base / "draws.npz")["scores"]
    mp = np.array([[r["scores"][k] for k in b.METRICS] for r in main["fit_rows"]])
    replaced = mp.copy()
    rd = md.copy()
    indices = old["original_model_indices"]
    b.equal(mp[indices], point[:5], "five original/main point agreement")
    b.equal(md[:, indices], draw[:, :5], "five original/main draw agreement")
    replaced[indices] = point[5:]
    rd[:, indices] = draw[:, 5:]
    _, ob = b.aggregate(mm["models"], mp)
    _, nb = b.aggregate(mm["models"], replaced)
    _, od = b.aggregate(mm["models"], md)
    _, nd = b.aggregate(mm["models"], rd)
    b.equal(
        nb[147, "tuned_selected"] - ob[147, "tuned_selected"],
        change / 3,
        "exact one-third sensitivity contribution",
    )
    contrasts = []
    for pair in main["contrasts"]:
        size = pair["budget"]
        left = pair["left"]
        right = pair["right"]
        entry = dict(
            budget=size,
            left=left,
            right=right,
            original=pair["metrics"],
            supplementary=b.interval(
                nb[size, left] - nb[size, right], nd[size, left] - nd[size, right], 12
            ),
        )
        entry["affected"] = size == 147 and left == "tuned_selected"
        contrasts.append(entry)
        if not entry["affected"]:
            b.equal(
                nb[size, left] - nb[size, right],
                ob[size, left] - ob[size, right],
                "unaffected primary score",
            )
    for x in old["all_24_comparisons"]:
        if x["affected"]:
            metrics = compare(
                nb[147, "tuned_selected"] - nb[147, "bge_selected_fold42"],
                nd[147, "tuned_selected"] - nd[147, "bge_selected_fold42"],
                x["supplementary"],
                12,
            )
    result = dict(
        study="five_fit_update",
        endpoint=u["endpoint"],
        n=len(g),
        labels=old["labels"],
        fit_rows=rows,
        affected_pool=dict(
            original=dict(zip(b.METRICS, original)),
            updated=dict(zip(b.METRICS, updated)),
            change=b.interval(change, dd),
        ),
        aggregate147=dict(
            original=dict(zip(b.METRICS, ob[147, "tuned_selected"])),
            updated=dict(zip(b.METRICS, nb[147, "tuned_selected"])),
            change=b.interval(change / 3, dd / 3),
            updated_minus_BGE=metrics,
        ),
        contrasts=contrasts,
        bootstrap=info,
        per_seed=[
            dict(
                seed=s,
                original=rows[i],
                updated=rows[i + 5],
                change=dict(zip(b.METRICS, point[i + 5] - point[i])),
            )
            for i, s in enumerate(range(10, 15))
        ],
    )
    b.save(folder / "results.json", result)
    b.array(folder / "draws.npz", scores=draw)
    b.array(folder / "predictions.npz", truth=g, predictions=p)
    b.save(folder / "metadata.json", meta)
    print("FIVE FITS COMPLETE", flush=True)
    return [c.rel(folder / "results.json")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("study", choices=["gpu", "controls", "five"])
    args = ap.parse_args()
    folder = (
        b.BASE
        / "attempts"
        / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f") + "_" + args.study)
    )
    folder.mkdir(parents=True)
    try:
        output = globals()[args.study](folder)
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

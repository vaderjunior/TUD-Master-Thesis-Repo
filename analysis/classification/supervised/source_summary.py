from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.classification.supervised import sources as a
from analysis.classification.supervised import archive as r
from analysis.classification.supervised import metrics as m
from analysis.classification.supervised import controls as c
from analysis.classification.supervised import errors as e
from analysis.classification.supervised import quality as q


def aggregate_points(models, point):
    pool = {}
    budget = {}
    for key, value in point.items():
        x = np.asarray(value)
        if x.shape == (len(models),):
            pool[key], budget[key] = c.aggregate(models, x)
        elif x.ndim == 2 and x.shape[0] == len(models):
            pp, bb = c.aggregate(models, x.T)
            pool[key] = {k: v.tolist() for k, v in pp.items()}
            budget[key] = {k: v.tolist() for k, v in bb.items()}
    return pool, budget


def intervals(point, draws, keys):
    absolute = {key: c.interval(point[key], draws[:, j]) for j, key in enumerate(keys)}
    contrasts = []
    for budget in m.BUDGETS:
        for after, before in q.PAIRS:
            ka, kb = f"{budget}/{after}", f"{budget}/{before}"
            contrasts.append(
                dict(
                    budget=budget,
                    after=after,
                    before=before,
                    **c.interval(
                        point[ka] - point[kb],
                        draws[:, keys.index(ka)] - draws[:, keys.index(kb)],
                    ),
                )
            )
    return dict(absolute=absolute, contrasts=contrasts)


def binary_group(models, meta, z, indices, bootstrap=False):
    ix = np.asarray(indices)
    y = z["truth"][ix, 1]
    p = z["gated"][:, ix, 1]
    metrics, undefined = a.binary(y, p)
    metrics["error_rate"] = 1 - metrics["accuracy"]
    pool, budget = aggregate_points(models, metrics)
    out = dict(
        kind="binary",
        rows=len(ix),
        indices=ix,
        positive=int(y.sum()),
        negative=int((1 - y).sum()),
        clusters=len({meta["clusters"][i] for i in ix}),
        undefined=undefined,
        model=metrics,
        pool=pool,
        budget=budget,
        uncertainty={},
    )
    draws = {}
    if bootstrap:
        draws, info = a.bootstrap_binary(
            models, y, p, [meta["clusters"][i] for i in ix]
        )
        out["bootstrap"] = info
        for metric, d in draws.items():
            out["uncertainty"][metric] = intervals(budget[metric], d, info["keys"])

    out["transitions"] = []
    for n in m.BUDGETS:
        pairs = []
        for poolname in r.POOLS:
            if m.budget(poolname) == n:
                pairs += e.pair_indices(
                    models, "tuned/selected", "bge/selected", poolname
                )
        events = e.paired_events(
            y, p[[b for b0, b in pairs]], p[[b0 for b0, b in pairs]]
        )
        out["transitions"].append(
            dict(budget=n, **{k: float(v.sum(1).mean()) for k, v in events.items()})
        )
    return out, draws


def fine_group(models, meta, z, indices, bootstrap=False):
    ix = np.asarray(indices)
    y = z["truth"][ix]
    p = z["gated"][:, ix]
    g = z["gate"][:, ix]
    raw = z["raw"][:, ix]
    point, row = q.quality(y, p, g, raw)
    point["macro_f1"] = m.point_scores(y, p, meta["support"])[0]
    point["raw_macro_f1"] = m.point_scores(y, raw, meta["support"])[0]
    unsupported = [j for j in range(len(meta["labels"])) if j not in meta["support"]]
    point["unsupported_label_emissions"] = (
        p[:, :, unsupported].sum((1, 2)) if unsupported else np.zeros(len(models), int)
    )
    pool, budget = aggregate_points(models, point)
    out = dict(
        kind="fine",
        rows=len(ix),
        indices=ix,
        labels=meta["labels"],
        fixed_macro_indices=meta["support"],
        clusters=len({meta["clusters"][i] for i in ix}),
        gold_support=y.sum(0),
        known_empty=int((y.sum(1) == 0).sum()),
        model=point,
        pool=pool,
        budget=budget,
        uncertainty={},
    )
    draws = {}
    if bootstrap:
        dd, info = m.bootstrap(y, p, [meta["clusters"][i] for i in ix], meta["support"])
        _, bb = c.aggregate(models, dd)
        keys = sorted(bb)
        draws["macro_f1"] = np.stack([bb[k] for k in keys], axis=1)
        out["uncertainty"]["macro_f1"] = intervals(
            budget["macro_f1"], draws["macro_f1"], keys
        )
        keys2, prof = q.profiles(
            models, {k: row[k] for k in ("gate_aware_exact", "sample_jaccard")}
        )
        rr, info2 = q.row_bootstrap(prof, [meta["clusters"][i] for i in ix])
        r.require(set(keys) == set(keys2), "Quality bootstrap keys differ.")
        for j, metric in enumerate(("gate_aware_exact", "sample_jaccard")):
            draws[metric] = rr[:, [keys2.index(k) for k in keys], j]
            out["uncertainty"][metric] = intervals(budget[metric], draws[metric], keys)
        r.require(
            info["picks_sha256"] == info2["picks_sha256"],
            "Paired fine resamples differ.",
        )
        out["bootstrap"] = info | dict(
            keys=keys, conditional_on_fixed_fits_and_pools=True
        )
    return out, draws


def rate_group(models, rows, values, indices, bootstrap=False):
    ix = np.asarray(indices)
    p = values[:, ix]
    point = p.mean(1)
    pool, budget = c.aggregate(models, point)
    cc = [rows[i]["functional_cluster"] for i in ix]
    out = dict(
        rows=len(ix),
        indices=ix,
        clusters=len(set(cc)),
        templates=len({json.loads(rows[i]["raw"])["templ_id"] for i in ix}),
        model=point,
        pool=pool,
        budget=budget,
        uncertainty={},
    )
    draws = {}
    if bootstrap:
        keys, profile = q.profiles(models, {"rate": p})
        d, info = q.row_bootstrap(profile, cc)
        draws = {"rate": d[:, :, 0]}
        out["bootstrap"] = info | dict(keys=keys)
        out["uncertainty"] = intervals(budget, draws["rate"], keys)
    return out, draws


def structure(rows):
    groups = {}
    links = []
    bycase = {json.loads(x["raw"])["case_id"]: i for i, x in enumerate(rows)}
    r.require(len(bycase) == len(rows), "Duplicate HateCheck case ID.")
    for field in ("template", "functional_cluster"):
        d = defaultdict(list)
        for i, x in enumerate(rows):
            d[
                (
                    str(json.loads(x["raw"])["templ_id"])
                    if field == "template"
                    else x[field]
                )
            ].append(i)
        groups[field] = {
            key: dict(
                indices=ii,
                functions=sorted(
                    {json.loads(rows[i]["raw"])["functionality"] for i in ii}
                ),
                gold_counts=dict(Counter(str(rows[i]["gate"]) for i in ii)),
            )
            for key, ii in sorted(d.items())
        }
    for i, x in enumerate(rows):
        ref = json.loads(x["raw"])["ref_case_id"]
        if ref is None:
            continue
        r.require(ref in bycase, "HateCheck reference missing.")
        j = bycase[ref]
        r.require(
            rows[i]["functional_cluster"] == rows[j]["functional_cluster"],
            "Reference crosses a frozen component.",
        )
        links.append(
            dict(
                child=i,
                parent=j,
                child_key=x["key"],
                parent_key=rows[j]["key"],
                function=json.loads(x["raw"])["functionality"],
                same_gold=x["gate"] == rows[j]["gate"],
                cluster=x["functional_cluster"],
            )
        )
    return dict(
        groups=groups,
        reference_links=links,
        counts={
            field: dict(
                groups=len(d),
                largest=max(len(x["indices"]) for x in d.values()),
                crosses_function=sum(len(x["functions"]) > 1 for x in d.values()),
                crosses_gold=sum(len(x["gold_counts"]) > 1 for x in d.values()),
            )
            for field, d in groups.items()
        },
    )


def main():
    base = a.BASE
    out = base / "analysis"
    out.mkdir(exist_ok=False)
    models = r.read(base / "extraction/models.json")
    endpoint_names = list(r.read(base / "extraction/context.json")["endpoints"])
    endpoints = [x[len("broad/") :] for x in endpoint_names if x.startswith("broad/")]
    summary = {}
    groups = {}
    frozen = {}
    save = a.save
    save(
        out / "started.json",
        dict(
            extraction_sha256=r.sha(base / "extraction/complete.json"),
            code_sha256=r.sha(Path(__file__)),
            scope_sha256=r.sha(base / "analysis_contract.json"),
        ),
    )
    for name in endpoints:
        meta, z = a.load(name)
        frozen[name] = (meta, z)
        method = binary_group if meta["head"] == "hate" else fine_group
        result, draws = method(
            models,
            meta,
            z,
            list(range(len(meta["records"]))),
            bootstrap=name not in r.PRIMARY,
        )
        summary[name] = result
        if draws:
            m.array(out / (m.slug(name) + "_draws.npz"), **draws)
        print(
            name
            + ": whole-endpoint points"
            + (
                " and supporting intervals"
                if draws
                else "; original primary inference retained"
            )
            + ".",
            flush=True,
        )
    save(out / "endpoints.json", summary)
    meta, z = frozen["en_test/hate"]
    rows = meta["records"]
    for source in a.SOURCES:
        ix = [i for i, x in enumerate(rows) if x["source"] == source]
        key = "source/" + source
        groups[key], draws = binary_group(models, meta, z, ix, True)
        m.array(out / (key.replace("/", "__") + "_draws.npz"), **draws)
    for name in sorted({e.annotation_stratum(x) for x in rows}):
        ix = [i for i, x in enumerate(rows) if e.annotation_stratum(x) == name]
        groups["annotation/" + name], _ = binary_group(models, meta, z, ix)
    for agreement in ("unanimous", "split"):
        ix = [
            i
            for i, x in enumerate(rows)
            if x["source"] == "hatexplain"
            and (len(set(json.loads(x["raw"])["votes"])) == 1)
            == (agreement == "unanimous")
        ]
        groups["agreement/" + agreement], _ = binary_group(models, meta, z, ix)
    save(out / "natural_binary.json", groups)
    print(
        "Natural sources, annotation boundaries and vote agreement complete.",
        flush=True,
    )
    fine = {}
    for name in ("en_test/target_group", "larger_dev/target_group"):
        meta, z = frozen[name]
        for source in ("hatexplain", "mhs"):
            key = name + "/" + source
            ix = [i for i, x in enumerate(meta["records"]) if x["source"] == source]
            fine[key], draws = fine_group(models, meta, z, ix, True)
            m.array(out / (key.replace("/", "__") + "_draws.npz"), **draws)
    save(out / "target_sources.json", fine)
    print(
        "Targets by natural source complete; annotation masks and label support fixed.",
        flush=True,
    )
    meta, z = frozen["en_hatecheck_eval/hate"]
    rows = meta["records"]
    y = z["truth"][:, 1]
    p = z["gated"][:, :, 1]
    errors = p != y
    st = structure(rows)
    save(out / "hatecheck_structure.json", st)
    functions = {}
    identities = {}
    units = {}
    for fn, title in a.FUNCTIONS.items():
        ix = [
            i for i, x in enumerate(rows) if json.loads(x["raw"])["functionality"] == fn
        ]
        r.require(
            ix and all(bool(y[i]) == fn.endswith("_h") for i in ix),
            "Function/polarity mismatch.",
        )
        functions[fn], dd = rate_group(models, rows, errors, ix, True)
        functions[fn].update(title=title, positive=fn.endswith("_h"))
        m.array(out / ("function__" + fn + "_draws.npz"), **dd)
    for target in sorted({str(json.loads(x["raw"])["target_ident"]) for x in rows}):
        for gold in (False, True):
            ix = [
                i
                for i, x in enumerate(rows)
                if str(json.loads(x["raw"])["target_ident"]) == target
                and bool(y[i]) == gold
            ]
            if ix:
                identities[f"{target}/{gold}"], _ = rate_group(models, rows, errors, ix)
    for field, gg in st["groups"].items():
        units[field] = {}
        for ident, spec in gg.items():
            units[field][ident], _ = rate_group(models, rows, errors, spec["indices"])

    unit_means = {
        field: {
            key: float(np.mean([x["budget"][key] for x in entries.values()]))
            for key in next(iter(entries.values()))["budget"]
        }
        for field, entries in units.items()
    }
    function_mean = {
        key: float(np.mean([x["budget"][key] for x in functions.values()]))
        for key in next(iter(functions.values()))["budget"]
    }
    save(out / "hatecheck_functions.json", functions)
    save(out / "hatecheck_identities.json", identities)
    save(out / "hatecheck_units.json", units)
    save(
        out / "hatecheck_weighting.json",
        dict(
            equal_function=function_mean,
            **{"equal_" + k: v for k, v in unit_means.items()},
        ),
    )
    links = st["reference_links"]
    linked = {}
    li = np.asarray([x["child"] for x in links])
    ri = np.asarray([x["parent"] for x in links])
    both = (~errors[:, li]) & (~errors[:, ri])
    relation = (p[:, li] == p[:, ri]) == np.asarray([x["same_gold"] for x in links])
    linkrows = [rows[x["child"]] for x in links]
    for name, ix in [
        ("same_gold", [i for i, x in enumerate(links) if x["same_gold"]]),
        ("changed_gold", [i for i, x in enumerate(links) if not x["same_gold"]]),
    ] + [
        ("function/" + fn, [i for i, x in enumerate(links) if x["function"] == fn])
        for fn in a.FUNCTIONS
        if any(x["function"] == fn for x in links)
    ]:
        linked[name] = {}
        for metric, values in [("both_correct", both), ("expected_relation", relation)]:
            linked[name][metric], dd = rate_group(
                models, linkrows, values, ix, bootstrap="/" not in name
            )
            if dd:
                m.array(out / ("links__" + name + "__" + metric + "_draws.npz"), **dd)
    save(out / "hatecheck_linked_results.json", linked)

    contributions = []
    for fn, x in functions.items():
        den = int((y == x["positive"]).sum())
        for n in m.BUDGETS:
            ka, kb = f"{n}/tuned/selected", f"{n}/bge/selected"
            contributions.append(
                dict(
                    function=fn,
                    budget=n,
                    positive=x["positive"],
                    rows=x["rows"],
                    denominator=den,
                    tuned_mean_errors=x["budget"][ka] * x["rows"],
                    bge_mean_errors=x["budget"][kb] * x["rows"],
                    difference_count=(x["budget"][ka] - x["budget"][kb]) * x["rows"],
                    contribution_to_error_rate=(x["budget"][ka] - x["budget"][kb])
                    * x["rows"]
                    / den,
                )
            )
    save(out / "hatecheck_contributions.json", contributions)
    train = r.read(a.QUALITY / "training_support.json")
    save(
        out / "training_source_support.json",
        {
            pool: {
                head: {k: row[k] for k in ("rows", "sources", "labels", "positives")}
                for head, row in heads.items()
            }
            for pool, heads in train.items()
        },
    )
    r.require("torch" not in sys.modules, "Unexpected training module loaded.")
    save(
        out / "complete.json",
        dict(
            status="complete",
            outputs={
                str(x.relative_to(out)): r.sha(x) for x in out.rglob("*") if x.is_file()
            },
            endpoints=len(summary),
            binary_groups=len(groups),
            fine_source_groups=len(fine),
            functions=len(functions),
            identity_groups=len(identities),
            linked_groups=len(linked),
            code_sha256=r.sha(Path(__file__)),
            training_calls=0,
            qwen_calls=0,
            weights_loaded=0,
            server_actions=0,
        ),
    )
    print(
        "Part 6 source/function calculations complete; all original choices and evidence preserved.",
        flush=True,
    )


if __name__ == "__main__":
    with r.offline(), threadpool_limits(limits=2):
        main()

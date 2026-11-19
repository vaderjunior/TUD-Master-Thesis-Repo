from collections import Counter, defaultdict
from itertools import combinations, product
from pathlib import Path
import json
import sys
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.classification.supervised import robustness as b
from analysis.classification.supervised import archive as r
from analysis.classification.supervised import sources as s
from analysis.classification.supervised import metrics as m
from analysis.classification.supervised import controls as c
from analysis.classification.supervised import training_audit as tr

OUT = b.BASE / "analysis"
PAIRS = (("tuned", "bge"), ("frozen_roberta", "bge"), ("tuned", "fixed_balanced"))


def source_rows():
    scopes = r.read(r.LINEAR / "data/scopes.json")
    inputs = r.read(r.PACKAGE / "data/inference.json")
    core = [inputs[x["position"]] for x in scopes["core"]]
    r.require(
        all(
            inputs[x["position"]]["key"] == x["key"]
            for rows in scopes.values()
            for x in rows
        ),
        "Scope order changed.",
    )
    return scopes, inputs, core


def inventory():
    OUT.mkdir(exist_ok=True)
    scopes, inputs, core = source_rows()
    norm = r.normalized
    evaltexts = {norm(x["text"]) for x in inputs}
    evalids = {x["key"] for x in inputs}
    coretexts = {norm(x["text"]) for x in core}
    coreids = {x["key"] for x in core}
    pools = {}
    details = {}
    bound = {}
    for pool in r.POOLS:
        path = r.PACKAGE / "data/train" / f"{pool}.json"
        bound[str(path)] = r.sha(path)
        data = r.read(path)
        ids = data["gate_ids"]
        texts = data["texts"]
        r.require(
            all(x.startswith("ex-") for x in ids), "Unrecognized TRAIN ID namespace."
        )
        canonical = {"en/" + x[3:] for x in ids}
        normalized = {norm(text) for text in texts}
        overlap_ids = canonical & evalids
        overlap_texts = normalized & evaltexts
        r.require(
            not overlap_ids and not overlap_texts,
            "TRAIN/evaluation overlap found; stop for review.",
        )
        folds = {}
        for assignment, labels in data["folds"].items():
            groups = defaultdict(set)
            for text, fold in zip(texts, labels):
                groups[norm(text)].add(fold)
            r.require(
                len(labels) == len(texts)
                and all(len(group) == 1 for group in groups.values()),
                "A normalized group crosses TRAIN folds.",
            )
            folds[assignment] = dict(
                counts=[labels.count(j) for j in range(5)], unique_clusters=len(groups)
            )
        heads = {}
        for head, spec in data["heads"].items():
            y = tr.truth(head, spec)
            support = tr.supported_columns(y)
            heads[head] = dict(
                rows=len(y),
                labels=spec["labels"],
                positive=y.sum(0),
                negative=len(y) - y.sum(0),
                selection_labels=[spec["labels"][j] for j in support],
                known_empty=int((y.sum(1) == 0).sum()),
            )
        source = Counter(x["source"] for x in data["heads"]["hate"]["rows"])
        details[pool] = dict(
            texts=len(texts),
            unique_ids=len(set(ids)),
            unique_normalized_texts=len(normalized),
            sources=source,
            folds=folds,
            heads=heads,
            train_eval_id_overlap=0,
            train_eval_normalized_overlap=0,
        )
        pools[pool] = dict(data=data, ids=set(ids), texts=normalized)
    overlaps = []
    for left, right in combinations(r.POOLS, 2):
        x, y = pools[left], pools[right]
        shared = x["ids"] & y["ids"]
        sharedtext = x["texts"] & y["texts"]
        heads = {}
        sources = {}
        for head in r.HEADS:
            heads[head] = len(
                {v["id"] for v in x["data"]["heads"][head]["rows"]}
                & {v["id"] for v in y["data"]["heads"][head]["rows"]}
            )
        for row in x["data"]["heads"]["hate"]["rows"]:
            if row["id"] in shared:
                sources[row["source"]] = sources.get(row["source"], 0) + 1
        overlaps.append(
            dict(
                left=left,
                right=right,
                same_budget=m.budget(left) == m.budget(right),
                left_rows=len(x["ids"]),
                right_rows=len(y["ids"]),
                shared_ids=len(shared),
                shared_texts=len(sharedtext),
                identity_jaccard=len(shared) / len(x["ids"] | y["ids"]),
                text_jaccard=len(sharedtext) / len(x["texts"] | y["texts"]),
                head_intersections=heads,
                source_intersections=sources,
                shared_ids_list=sorted(shared),
            )
        )
    memberships = {}
    for n in m.BUDGETS:
        pp = [p for p in r.POOLS if m.budget(p) == n]
        count = Counter(key for pool in pp for key in pools[pool]["ids"])
        memberships[str(n)] = dict(
            unique_union=len(count),
            total_permitted=sum(count.values()),
            multiplicity=dict(Counter(count.values())),
        )
    broadkeys = {x["key"] for x in scopes["broad"]}
    broads = [
        x
        for x in map(json.loads, s.e.ROSTER.read_text(encoding="utf8").splitlines())
        if x["key"] in broadkeys
    ]
    r.require(len(broads) == len(broadkeys) == 18275, "Broad roster coverage differs.")
    coverage = {}
    for phase, source in sorted({(x["phase"], x["source"]) for x in broads}):
        rows = [x for x in broads if x["phase"] == phase and x["source"] == source]
        heads = {}
        for head in r.HEADS[1:]:
            field = r.FIELDS[head]
            positive = [x for x in rows if x["gate"] is True]
            eligible = [x for x in positive if x[field] is not None]
            heads[head] = dict(
                observed_all=sum(x[field] is not None for x in rows),
                eligible_positive=len(eligible),
                missing_positive=sum(x[field] is None for x in positive),
                known_empty_positive=sum(x[field] == [] for x in eligible),
            )
        masks = Counter(
            "".join("1" if x[r.FIELDS[h]] is not None else "0" for h in r.HEADS[1:])
            for x in rows
            if x["gate"] is True
        )
        coverage[phase + "/" + source] = dict(
            rows=len(rows),
            hate_positive=sum(x["gate"] is True for x in rows),
            hate_negative=sum(x["gate"] is False for x in rows),
            hate_missing=sum(x["gate"] is None for x in rows),
            heads=heads,
            positive_fine_masks=masks,
        )
    phasepairs = []
    for left, right in combinations(sorted({x["phase"] for x in broads}), 2):
        xx = [x for x in broads if x["phase"] == left]
        yy = [x for x in broads if x["phase"] == right]
        phasepairs.append(
            dict(
                left=left,
                right=right,
                shared_ids=len({x["key"] for x in xx} & {x["key"] for x in yy}),
                shared_normalized_texts=len(
                    {norm(x["text"]) for x in xx} & {norm(x["text"]) for x in yy}
                ),
            )
        )
    exclusions = {}
    for name in r.read(s.BASE / "analysis/endpoints.json"):
        meta, _ = s.load(name)
        records = meta["records"]
        identity = [i for i, x in enumerate(records) if x["key"] in coreids]
        text = [i for i, x in enumerate(records) if norm(x["text"]) in coretexts]
        omit = sorted(set(identity) | set(text))
        exclusions[name] = dict(
            rows=len(records),
            identity_overlap=len(identity),
            normalized_text_overlap=len(text),
            excluded_indices=omit,
            excluded_ids=[records[i]["key"] for i in omit],
            remaining_indices=[i for i in range(len(records)) if i not in set(omit)],
            macro_columns=meta["support"],
        )
    joint = sum(
        all(x[r.FIELDS[h]] is not None for h in r.HEADS[1:])
        for x in broads
        if x["gate"] is True
    )
    b.save(
        OUT / "inventory.json",
        dict(
            pools=details,
            pool_overlaps=overlaps,
            membership=memberships,
            core=dict(
                rows=len(core),
                unique_normalized_texts=len(coretexts),
                broad_identity_overlap=len(coreids & broadkeys),
                broad_normalized_text_overlap=len(
                    coretexts & {norm(x["text"]) for x in broads}
                ),
            ),
            phase_overlaps=phasepairs,
            coverage=coverage,
            exclusions=exclusions,
            fully_annotated_positive_rows=joint,
            source_files=bound,
            normalization="NFKC, casefold, collapse whitespace, strip; not semantic duplicate or pretraining detection.",
            id_normalization="TRAIN ex- prefix replaced by en/ to compare source identity namespaces.",
            code_sha256=r.sha(Path(__file__)),
        ),
    )
    print(
        "Inventory complete: pool/core overlap, annotation coverage and fixed exclusion indices.",
        flush=True,
    )


def stats(values):
    values = np.asarray(values, float)
    return dict(
        n=len(values),
        mean=float(values.mean()),
        minimum=float(values.min()),
        maximum=float(values.max()),
        sd=float(values.std(ddof=1)) if len(values) > 1 else None,
        positive=int((values > 1e-12).sum()),
        negative=int((values < -1e-12).sum()),
        zero=int((np.abs(values) <= 1e-12).sum()),
    )


def stability(models, scores):
    pools, budgets = b.aggregate(models, scores)
    absolute = []
    contrasts = []
    for n in m.BUDGETS:
        pp = [p for p in r.POOLS if m.budget(p) == n]
        for method in b.METHODS:
            for policy in r.POLICIES:
                poolviews = []
                for pool in pp:
                    ii = [
                        i
                        for i, x in enumerate(models)
                        if (x["pool"], x["method"], x["variant"])
                        == (pool, method, policy)
                    ]
                    values = scores[ii]
                    poolviews.append(
                        dict(
                            pool=pool,
                            **stats(values),
                            seeds=[
                                dict(seed=models[i]["seed"], score=scores[i])
                                for i in ii
                            ],
                        )
                    )
                absolute.append(
                    dict(
                        budget=n,
                        method=method,
                        policy=policy,
                        pool_values=poolviews,
                        across_pool_means=stats([x["mean"] for x in poolviews]),
                        mean_within_pool_sd=(
                            None
                            if method not in r.FAMILIES
                            else float(np.mean([x["sd"] for x in poolviews]))
                        ),
                    )
                )
        for left, right in PAIRS:
            policy = "selected"
            rows = []
            for pool in pp:
                li = sorted(
                    [
                        i
                        for i, x in enumerate(models)
                        if (x["pool"], x["method"], x["variant"])
                        == (pool, left, policy)
                    ],
                    key=lambda i: (
                        -1 if models[i]["seed"] is None else models[i]["seed"]
                    ),
                )
                ri = [
                    i
                    for i, x in enumerate(models)
                    if (x["pool"], x["method"], x["variant"]) == (pool, right, policy)
                ]
                matches = [
                    next(
                        j
                        for j in ri
                        if models[j]["seed"]
                        == (models[i]["seed"] if right in r.FAMILIES else None)
                    )
                    for i in li
                ]
                diffs = np.array([scores[i] - scores[j] for i, j in zip(li, matches)])
                rows.append(
                    dict(
                        pool=pool,
                        left_values=scores[li],
                        right_values=scores[matches],
                        seeds=[models[i]["seed"] for i in li],
                        differences=diffs,
                        **stats(diffs),
                    )
                )
            means = np.array([x["mean"] for x in rows])
            point = float(means.mean())
            loo = [
                dict(
                    omitted_pool=pp[j],
                    difference=float(np.delete(means, j).mean()),
                    change=float(np.delete(means, j).mean() - point),
                )
                for j in range(3)
            ]
            row = dict(
                budget=n,
                left=left,
                right=right,
                policy=policy,
                difference=point,
                pools=rows,
                pool_mean_statistics=stats(means),
                leave_one_pool_out=loo,
            )
            if left in r.FAMILIES:
                matrix = np.stack([x["differences"] for x in rows])
                r.require(matrix.shape == (3, 5), "Seed comparison grid differs.")
                combos = [
                    dict(
                        seeds=list(seeds),
                        difference=float(
                            np.mean(
                                [
                                    matrix[j, r.SEEDS.index(seed)]
                                    for j, seed in enumerate(seeds)
                                ]
                            )
                        ),
                    )
                    for seeds in product(r.SEEDS, repeat=3)
                ]
                removed = [
                    dict(
                        pool=pp[j],
                        seed=seed,
                        difference=float(
                            (means.sum() - means[j] + np.delete(matrix[j], k).mean())
                            / 3
                        ),
                        change=float((np.delete(matrix[j], k).mean() - means[j]) / 3),
                    )
                    for j in range(3)
                    for k, seed in enumerate(r.SEEDS)
                ]
                common = [
                    dict(seed=seed, difference=float(matrix[:, k].mean()))
                    for k, seed in enumerate(r.SEEDS)
                ]
                leavecommon = [
                    dict(
                        omitted_seed=seed,
                        difference=float(np.delete(matrix, k, axis=1).mean()),
                    )
                    for k, seed in enumerate(r.SEEDS)
                ]
                row.update(
                    matched_fit_statistics=stats(matrix.ravel()),
                    common_seed=common,
                    one_seed_per_pool=combos,
                    combination_statistics=stats([x["difference"] for x in combos]),
                    single_seed_omissions=removed,
                    common_seed_omissions=leavecommon,
                    maximum_single_seed_influence=max(
                        abs(x["change"]) for x in removed
                    ),
                    maximum_single_pool_influence=max(abs(x["change"]) for x in loo),
                )
            contrasts.append(row)
    return dict(absolute=absolute, contrasts=contrasts, pool=pools, budget=budgets)


def points():
    inv = r.read(OUT / "inventory.json")
    models = r.read(b.BASE / "prepared/models.json")
    summary = {}
    sourcegroups = {}
    for name in inv["exclusions"]:
        meta, z = b.load(name)
        scores = m.point_scores(z["truth"], z["gated"], meta["support"])[0]
        old = r.read(s.BASE / "analysis/endpoints.json")[name]["model"]["macro_f1"]
        r.require(
            np.allclose(scores[:306], old, rtol=0, atol=1e-14),
            "Original endpoint scores changed.",
        )
        ss = stability(models, scores)
        ix = inv["exclusions"][name]["remaining_indices"]
        r.require(len(ix) > 0, "Core exclusion emptied endpoint.")
        remaining = m.point_scores(z["truth"][ix], z["gated"][:, ix], meta["support"])[
            0
        ]
        rp, rb = b.aggregate(models, remaining)
        ss.update(
            rows=len(z["truth"]),
            labels=meta["labels"],
            support=meta["support"],
            model=scores,
            core_excluded=dict(
                rows=len(ix),
                removed=len(z["truth"]) - len(ix),
                gold_support=z["truth"][ix].sum(0),
                model=remaining,
                pool=rp,
                budget=rb,
            ),
        )
        m.array(
            OUT / (m.slug(name) + "_points.npz"),
            scores=scores,
            core_excluded_scores=remaining,
        )
        summary[name] = ss
        if name in r.PRIMARY:
            sources = sorted({x["source"] for x in meta["records"]})
            groups = {}
            per_source = {}
            for source in sources:
                si = [j for j, x in enumerate(meta["records"]) if x["source"] == source]
                sc = m.point_scores(z["truth"][si], z["gated"][:, si], meta["support"])[
                    0
                ]
                pp, bb = b.aggregate(models, sc)
                per_source[source] = dict(rows=len(si), model=sc, pool=pp, budget=bb)
                if len(sources) > 1:
                    keep = [
                        j
                        for j, x in enumerate(meta["records"])
                        if x["source"] != source
                    ]
                    co = m.point_scores(
                        z["truth"][keep], z["gated"][:, keep], meta["support"]
                    )[0]
                    cp, cb = b.aggregate(models, co)
                    groups[source] = dict(rows=len(keep), model=co, pool=cp, budget=cb)
            equal = np.mean([x["model"] for x in per_source.values()], axis=0)
            ep, eb = b.aggregate(models, equal)
            sourcegroups[name] = dict(
                sources=sources,
                per_source=per_source,
                leave_one_source_out=groups,
                equal_source_mean=dict(model=equal, pool=ep, budget=eb),
                no_cross_source_sensitivity=len(sources) == 1,
            )
        print(
            name + ": seed/pool, alternative-fold and core-exclusion points checked.",
            flush=True,
        )
    b.save(OUT / "points.json", summary)
    b.save(OUT / "source_sensitivity.json", sourcegroups)

    changes = []
    for pool in r.POOLS:
        heads = r.read(m.TRAIN / pool / "linear_selection_review.json")["heads"]
        entries = {(x["fold"], x["head"]): x for x in heads if x["method"] == "bge"}
        for head in r.HEADS:
            selected = {str(f): entries[f, head]["selected"] for f in (42, 43, 44)}
            r.require(
                all(
                    selected[str(x["fold"])] == x["selected"]
                    for x in r.read(b.BASE / "prepared/alternative_choices.json")
                    if x["pool"] == pool and x["head"] == head
                ),
                "Alternative TRAIN choice differs.",
            )
            signatures = {(x["C"], x["threshold"]) for x in selected.values()}
            changes.append(
                dict(
                    pool=pool,
                    head=head,
                    selected=selected,
                    distinct_C_cutoff=len(signatures),
                )
            )
    b.save(OUT / "fold_choices.json", changes)


def primary_saved(name, models):
    slug = m.slug(name)
    meta = r.read(b.MAIN / (slug + "_metadata.json"))
    receipt = r.read(b.MAIN / "complete.json")
    for suffix in (
        "_metadata.json",
        "_bootstrap.npz",
        "_point_counts.npz",
        "_results.json",
    ):
        path = b.MAIN / (slug + suffix)
        r.checked_file(path, receipt["outputs"][path.name])
    old = r.arrays_from_bytes((b.MAIN / (slug + "_bootstrap.npz")).read_bytes())[
        "scores"
    ]
    points = r.arrays_from_bytes((b.MAIN / (slug + "_point_counts.npz")).read_bytes())[
        "scores"
    ]
    alias = {
        "tuned": "tuned_selected",
        "bge": "bge_selected_fold42",
        "frozen_roberta": "frozen_roberta_selected_fold42",
    }
    indices = [
        i
        for i, x in enumerate(models)
        if x["variant"] == "selected" and x["method"] in alias
    ]
    mapping = []
    for i in indices:
        x = models[i]
        match = [
            j
            for j, z in enumerate(meta["models"])
            if (z["pool"], z["method"], z["seed"])
            == (x["pool"], alias[x["method"]], x["seed"])
        ]
        r.require(len(match) == 1, "Saved primary model mapping differs.")
        mapping.append(match[0])
    r.require(len(indices) == 63, "Wrong saved primary roster.")
    return (
        indices,
        old[:, mapping],
        points[mapping],
        r.read(b.MAIN / (slug + "_results.json"))["bootstrap"],
    )


def ci(value, draws):
    return dict(
        value=float(value),
        low=float(np.quantile(draws, 0.025)),
        high=float(np.quantile(draws, 0.975)),
    )


def bootstrap():
    models = r.read(b.BASE / "prepared/models.json")
    points = r.read(OUT / "points.json")
    inv = r.read(OUT / "inventory.json")
    result = {}
    for name in r.PRIMARY:
        meta, z = b.load(name)
        ii, old, oldpoint, info = primary_saved(name, models)
        r.require(
            np.allclose(
                np.asarray(points[name]["model"])[ii], oldpoint, rtol=0, atol=1e-14
            ),
            "Reused primary point mapping differs.",
        )
        extra = [
            j
            for j, x in enumerate(models)
            if x["variant"] == "selected" and x["method"] in ("bge_f43", "bge_f44")
        ]
        draw, newinfo = m.bootstrap(
            z["truth"], z["gated"][extra], meta["clusters"], meta["support"]
        )
        r.require(
            newinfo["picks_sha256"] == info["picks_sha256"],
            "Alternative folds do not share primary resampling.",
        )
        allidx = ii + extra
        allmodels = [models[j] for j in allidx]
        allvalues = np.concatenate([old, draw], axis=1)
        point = np.asarray(points[name]["model"])[allidx]
        _, bb = b.aggregate(allmodels, point)
        _, bd = b.aggregate(allmodels, allvalues)
        fold = []
        for n in m.BUDGETS:
            for left, right in [
                (method, "bge") for method in ("bge_f43", "bge_f44", "bge_fold_mean")
            ] + [
                (method, baseline)
                for method in ("tuned", "frozen_roberta")
                for baseline in ("bge_f43", "bge_f44", "bge_fold_mean")
            ]:
                ka, kb = f"{n}/{left}/selected", f"{n}/{right}/selected"
                fold.append(
                    dict(
                        budget=n,
                        left=left,
                        right=right,
                        **ci(bb[ka] - bb[kb], bd[ka] - bd[kb]),
                    )
                )
        keep = inv["exclusions"][name]["remaining_indices"]
        if len(keep) == len(z["truth"]):
            filtered = old
            filteredinfo = info
            reused = True
        else:
            filtered, filteredinfo = m.bootstrap(
                z["truth"][keep],
                z["gated"][ii][:, keep],
                [meta["clusters"][i] for i in keep],
                meta["support"],
            )
            reused = False
        coremodels = [models[i] for i in ii]
        corepoint = np.asarray(points[name]["core_excluded"]["model"])[ii]
        _, cb = b.aggregate(coremodels, corepoint)
        _, cd = b.aggregate(coremodels, filtered)
        core = []
        for n in m.BUDGETS:
            for method in ("tuned", "frozen_roberta"):
                ka, kb = f"{n}/{method}/selected", f"{n}/bge/selected"
                original = points[name]["budget"][ka] - points[name]["budget"][kb]
                core.append(
                    dict(
                        budget=n,
                        left=method,
                        right="bge",
                        original_difference=original,
                        point_change=cb[ka] - cb[kb] - original,
                        **ci(cb[ka] - cb[kb], cd[ka] - cd[kb]),
                    )
                )
        m.array(
            OUT / (m.slug(name) + "_alternative_draws.npz"),
            scores=draw,
            model_indices=np.asarray(extra),
        )
        if not reused:
            m.array(
                OUT / (m.slug(name) + "_core_excluded_draws.npz"),
                scores=filtered,
                model_indices=np.asarray(ii),
            )
        result[name] = dict(
            alternative_fold=fold,
            core_excluded=core,
            extra_model_indices=extra,
            original_model_indices=ii,
            alternative_info=newinfo,
            core_info=filteredinfo,
            core_draws_reused=reused,
        )
        print(
            name
            + ": supporting alternative-fold/core intervals complete; original adjusted family unchanged.",
            flush=True,
        )
    b.save(OUT / "bootstrap_results.json", result)
    b.save(
        OUT / "complete.json",
        dict(
            status="complete",
            outputs={
                str(p.relative_to(OUT)): r.sha(p) for p in OUT.iterdir() if p.is_file()
            },
            code_sha256=r.sha(Path(__file__)),
            training_calls=0,
            qwen_calls=0,
            weights_loaded=0,
            server_actions=0,
        ),
    )


if __name__ == "__main__":
    with r.offline(), threadpool_limits(limits=2):
        if sys.argv[1:] == ["inventory"]:
            inventory()
        elif sys.argv[1:] == ["points"]:
            points()
        elif sys.argv[1:] == ["bootstrap"]:
            bootstrap()
        else:
            raise SystemExit("Use inventory, points or bootstrap.")

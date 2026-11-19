from analysis.common import micro_f1_resampling as b
from analysis.common import micro_f1_reader as reader
import argparse
from datetime import datetime, timezone
import traceback
import numpy as np

c = b.c
ROOT = b.ROOT
OLD = ROOT / "experiments/astra"
REG = None
FAMILIES = None


def load_registry():
    global REG, FAMILIES
    if REG is None or FAMILIES is None:
        REG = c.read(b.SETUP / "comparison_registry_v2.json")
        FAMILIES = c.read(b.SETUP / "paired_contrast_manifest.json")["rows"]


def unit(study, endpoint):
    load_registry()
    candidates = [
        u for u in REG["units"] if u["study"] == study and u["endpoint"] == endpoint
    ]
    c.require(len(candidates) == 1, "Ambiguous registry unit")
    return candidates[0]


def grouped(g, p, v, settings, labels, macro, mask=None, gates=None):
    runs = []
    for i, key in enumerate(settings):
        vv = v[i] if mask is None else mask
        c.require(
            np.all(~vv | v[i]), "Common population includes an invalid prediction"
        )
        s = b.setting_summary(
            g, p[i], vv, labels, macro, None if gates is None else gates[i]
        )
        runs.append(dict(setting=key, **s))
    groups = {}
    for name in sorted({s[0] for s in settings}):
        ss = [r for r in runs if r["setting"][0] == name]
        groups[name] = b.average(ss)
    return dict(runs=runs, groups=groups)


def register_pair(study, ep, left, right):
    load_registry()
    rows = [
        r
        for r in FAMILIES
        if r["family"] == study
        and r["unit_id"] == study + "/" + ep
        and (r["left"] == left)
        and (r["right"] == right)
    ]
    c.require(len(rows) <= 1, "Repeated adjusted question")
    return rows[0] if rows else None


def compare(
    folder,
    g,
    p,
    v,
    settings,
    labels,
    macro,
    ids,
    clusters,
    left,
    right,
    seed,
    old=None,
    family=None,
    item_order=False,
    strata=None,
    mode="pairs",
    tag=None,
):
    leftkeys = [k for x in left for k in settings if k[0] == x]
    rightkeys = [k for x in right for k in settings if k[0] == x]
    keys = leftkeys + rightkeys
    ix = [settings.index(k) for k in keys]
    indices = [labels.index(x) for x in macro]
    boot = b.bootstrap(
        g,
        p[ix],
        v[ix],
        indices,
        clusters,
        len(leftkeys),
        seed,
        item_order=item_order,
        strata_sources=strata,
        delta_mode=mode,
    )
    out = b.contrast_summary(boot, leftkeys, rightkeys, family)
    out["common_ids"] = [x for x, yes in zip(ids, boot["mask"]) if yes]
    if old is not None:
        if "common_valid_ids" in old:
            c.require(
                out["common_ids"] == old["common_valid_ids"],
                "Original paired IDs differ",
            )
        for key in ("common_f1_by_setting", "per_setting_f1"):
            if key in old:
                b.equal(boot["point"][:, 0], old[key], "original paired setting macro")
        b.equal(
            out["metrics"]["macro"]["difference"],
            old.get("mean_difference", old.get("difference")),
            "original paired macro point",
        )
        b.equal(
            out["metrics"]["macro"]["ci95"],
            old["ci95"],
            "original paired macro 95 interval",
        )
        original_adjusted = old.get(
            "family_interval", old.get("family_ci", old.get("family_ci975"))
        )
        confidence = old.get(
            "family_confidence",
            (
                1 - 0.05 / old["family_size"]
                if old.get("family_size")
                else 0.975 if "family_ci975" in old else None
            ),
        )
        if original_adjusted is not None and confidence is not None:
            b.equal(
                b.interval(boot["draws"][:, 2, 0], confidence),
                original_adjusted,
                "original adjusted macro interval",
            )
        out["original_macro_role"] = old.get(
            "role", "primary" if old.get("primary") else "supporting"
        )
        out["original_macro_adjusted_interval"] = original_adjusted
        out["original_macro_adjusted_confidence"] = confidence
        if old.get("draws_path"):
            path = ROOT / old["draws_path"]
            c.require(c.sha(path) == old["draws_sha256"], "Original draw hash changed")
            original = np.load(c.bind(path), allow_pickle=False)
            b.equal(
                boot["draws"][:, 2, 0],
                original,
                "all saved original macro bootstrap draws",
            )
    out["label_scope"] = dict(full=labels, macro=macro)
    own = grouped(g, p[ix], v[ix], keys, labels, macro)
    bounds = {}
    for name in b.METHODS:
        lo = [
            r["full_roster_missing_bounds"][name] for r in own["runs"][: len(leftkeys)]
        ]
        ro = [
            r["full_roster_missing_bounds"][name] for r in own["runs"][len(leftkeys) :]
        ]
        bounds[name] = (
            [
                float(np.mean([x[0] for x in lo]) - np.mean([x[1] for x in ro])),
                float(np.mean([x[1] for x in lo]) - np.mean([x[0] for x in ro])),
            ]
            if macro or name == "micro"
            else None
        )
    out["full_roster_missing_difference_bounds"] = bounds
    out["paired_runs"] = grouped(g, p[ix], v[ix], keys, labels, macro, boot["mask"])
    name = tag or "_".join(left) + "_minus_" + "_".join(right)
    out["draw_file"] = c.rel(folder / (name + ".npz"))
    b.save_arrays(folder / (name + ".npz"), draws=boot["draws"], mask=boot["mask"])
    return out


def synthetic(folder):
    rng = np.random.default_rng(5879)
    checks = 0
    for _ in range(80):
        g = rng.integers(0, 2, (17, 4)).astype(bool)
        p = rng.integers(0, 2, (3, 17, 4)).astype(bool)
        v = rng.random((3, 17)) > 0.1
        for j in range(3):
            s = b.setting_summary(g, p[j], v[j], list("abcd"), list("abc"))
            gg, pp = (g[v[j]], p[j, v[j]])
            tp = np.count_nonzero(gg & pp)
            fp = np.count_nonzero(~gg & pp)
            fn = np.count_nonzero(gg & ~pp)
            b.equal(
                s["scores"]["micro"],
                2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
                "independent flattened micro",
            )
            checks += 1
            for optimistic, k in ((True, 1), (False, 0)):
                pp = p[j].copy()
                pp[~v[j]] = g[~v[j]] if optimistic else ~g[~v[j]]
                count = reader.counts(g, pp)
                b.equal(
                    b.values(count, [0, 1, 2]),
                    [s["full_roster_missing_bounds"][x][k] for x in b.METHODS],
                    "missing bound direct completion",
                )
    g = rng.integers(0, 2, (23, 4)).astype(bool)
    p = rng.integers(0, 2, (4, 23, 4)).astype(bool)
    v = np.ones((4, 23), bool)
    boot = b.bootstrap(
        g, p, v, [0, 1, 2], list(range(23)), 2, 34, draws=131, item_order=True
    )
    r = np.random.default_rng(34)
    ref = []
    for _ in range(131):
        ix = r.integers(0, 23, 23)
        score = np.stack(
            [b.values(reader.counts(g[ix], pp[ix]), [0, 1, 2]) for pp in p]
        )
        ref.append((score[:2, None] - score[None, 2:]).mean((0, 1)))
    b.equal(boot["draws"][:, 2], ref, "item bootstrap independent expansion")
    b.save(
        folder / "synthetic.json",
        dict(
            passed=True,
            flattened_checks=checks,
            item_draws=131,
            checks=dict(b.CHECKS),
            max_errors=dict(b.ERRORS),
        ),
    )


def broad(folder):
    load_registry()
    records = []
    for u in [u for u in REG["units"] if u["study"] == "broad"]:
        d = c.read(ROOT / u["source"])
        ep = d["endpoint"]
        name = ep["phase"] + "__" + ep["dimension"]
        outdir = folder / name
        a = b.read_arrays(ROOT / d["encoded_path"])
        c.require(
            c.sha(ROOT / d["encoded_path"]) == d["encoded_sha256"],
            "Encoded source changed",
        )
        g = a["gold"].astype(bool)
        p = a["predicted"].astype(bool)
        v = a["valid"].astype(bool)
        settings = [tuple(k) for k in d["settings"]]
        labels = ep["all_labels"]
        macro = ep["macro_labels"]
        ids = ep["scorable_ids"]
        clusters = ep["row_clusters"]
        combined = OLD / "sq2_combined_performance_20261104_v1/endpoints" / name
        meta = c.read(combined / "metadata.json")
        ca = b.read_arrays(combined / "aligned.npz")
        c.require(meta["ids"] == ids, "Combined IDs differ")
        b.equal(g, ca["gold"], "combined gold alignment")
        gate = []
        for i, k in enumerate(settings):
            found = [
                j
                for j, m in enumerate(meta["models"])
                if m["original_method"] == k[0]
                and m["seed"] == k[1]
                and (
                    m["group"] == "qwen/" + k[0] or m["group"].startswith("historical_")
                )
            ]
            c.require(len(found) == 1, "Broad gate model ambiguous")
            j = found[0]
            b.equal(p[i], ca["predictions"][j], "combined prediction alignment")
            b.equal(v[i], ca["valid"][j], "combined validity alignment")
            gate.append(ca["gate"][j])
        gates = np.array(gate) if ep["dimension"] != "legal" else None
        b.equal(b.encode(g, p, v), a["counts"], "broad per-item contributions")
        own = grouped(g, p, v, settings, labels, macro, gates=gates)
        for r in own["runs"]:
            cond, rep = r["setting"]
            old = d["own_valid"][cond]["repeats"][rep]
            b.equal(r["scores"]["macro"], old["macro_f1_valid_only"], "broad own macro")
            b.equal(r["tp_fp_fn"], old["per_label_counts"], "broad own counts")
        nine = [
            i
            for i, k in enumerate(settings)
            if k[0] in ("rag_full", "rag_small", "zero_shot")
        ]
        mask = v[nine].all(0)
        b.equal(
            int(mask.sum()),
            d["three_condition_common"]["n_common_valid"],
            "common-nine sample size",
        )
        common = grouped(
            g,
            p[nine],
            v[nine],
            [settings[i] for i in nine],
            labels,
            macro,
            mask,
            None if gates is None else gates[nine],
        )
        for cond, group in common["groups"].items():
            b.equal(
                group["scores"]["macro"],
                d["three_condition_common"]["per_condition"][cond]["mean_macro_f1"],
                "common-nine macro",
            )
        comparisons = []
        if macro:
            for old in d["contrasts"]:
                pair = register_pair("broad", u["endpoint"], old["left"], old["right"])
                family = pair["planned_family_size"] if pair else None
                res = compare(
                    outdir,
                    g,
                    p,
                    v,
                    settings,
                    labels,
                    macro,
                    ids,
                    clusters,
                    [old["left"]],
                    [old["right"]],
                    old["seed"],
                    old,
                    family,
                )
                res["question_id"] = pair["id"] if pair else None
                if family:
                    kind = (
                        "item"
                        if ep["phase"] == "en_hatecheck_eval"
                        else "source_stratified"
                    )
                    sens = compare(
                        outdir,
                        g,
                        p,
                        v,
                        settings,
                        labels,
                        macro,
                        ids,
                        clusters,
                        [old["left"]],
                        [old["right"]],
                        old["seed"],
                        family=family,
                        item_order=kind == "item",
                        strata=(
                            ep["row_sources"] if kind == "source_stratified" else None
                        ),
                        tag=old["left"] + "_minus_" + old["right"] + "_" + kind,
                    )
                    res["resampling_sensitivity"] = dict(kind=kind, **sens)
                comparisons.append(res)
        selected = [
            i
            for i, m in enumerate(meta["models"])
            if m["group"].startswith(("qwen/", "historical_"))
            or m["group"]
            in (
                "cpu/full_bge_matched/selected",
                "cpu/full_tfidf_matched/selected",
                "cpu/full_neural/selected",
            )
        ]
        cg = grouped(
            g,
            ca["predictions"][selected],
            ca["valid"][selected],
            [(meta["models"][i]["group"], meta["models"][i]["seed"]) for i in selected],
            labels,
            macro,
            mask,
            ca["gate"][selected] if ep["dimension"] != "legal" else None,
        )
        oldscores = c.read(combined / "scores.json")["common_nine"]["runs"]
        for r, i in zip(cg["runs"], selected):
            b.equal(
                r["scores"]["macro"],
                oldscores[i]["macro_f1"],
                "native reference/common macro",
            )
        result = dict(
            study="broad",
            endpoint=name,
            dimension=ep["dimension"],
            labels=labels,
            macro_labels=macro,
            ids=ids,
            own_valid=own,
            common_nine=common,
            common_nine_ids=[x for x, yes in zip(ids, mask) if yes],
            contrasts=comparisons,
            common_nine_context=cg,
            diagnostic_only=not bool(macro),
            interpretation="Pair-specific inference; common-nine display and native references descriptive.",
        )
        b.save(outdir / "results.json", result)
        b.save_arrays(
            outdir / "analysis_arrays.npz",
            gold=g,
            predictions=p,
            valid=v,
            gates=np.array(gate),
            common_nine=mask,
        )
        records.append(c.rel(outdir / "results.json"))
        print("Broad complete", name, flush=True)
    return records


def core(folder):
    load_registry()
    records = []
    for u in [u for u in REG["units"] if u["study"] == "core"]:
        d, ep, ids, g, prepared = reader.legacy_data(u)
        labels = ep["all_labels"]
        macro = ep["macro_labels"]
        settings = sorted(prepared)
        p = np.array([prepared[k][0] for k in settings])
        v = np.array([prepared[k][1] for k in settings])
        outdir = folder / u["endpoint"]
        own = grouped(g, p, v, settings, labels, macro)
        for r in own["runs"]:
            cond, rep = r["setting"]
            old = next(
                (x for x in d["conditions"][cond]["runs"] if x["replicate"] == rep)
            )["score"]
            b.equal(r["scores"]["macro"], old["macro_f1_valid_only"], "core own macro")
            b.equal(r["tp_fp_fn"], old["per_label_counts"], "core own counts")

        def members(name):
            return (
                [name]
                if name in d["conditions"]
                else d["fewshot_groups"][name]["conditions"]
            )

        comparisons = []
        for name, old in d["contrasts"].items():
            family = register_pair("core", u["endpoint"], old["left"], old["right"])
            res = compare(
                outdir,
                g,
                p,
                v,
                settings,
                labels,
                macro,
                ids,
                ids,
                members(old["left"]),
                members(old["right"]),
                old["bootstrap_seed"],
                old,
                family["planned_family_size"] if family else None,
                item_order=True,
                tag=name,
            )
            res["name"] = name
            res["question_id"] = family["id"] if family else None
            comparisons.append(res)
        b.save(
            outdir / "results.json",
            dict(
                study="core",
                endpoint=u["endpoint"],
                labels=labels,
                macro_labels=macro,
                ids=ids,
                settings=settings,
                own_valid=own,
                contrasts=comparisons,
            ),
        )
        b.save_arrays(outdir / "analysis_arrays.npz", gold=g, predictions=p, valid=v)
        records.append(c.rel(outdir / "results.json"))
        print("Core complete", u["endpoint"], flush=True)
    return records


def label_vote(folder):
    load_registry()
    base = OLD / "sq2_retrieved_label_vote_20261001_v1"
    f = c.read(base / "frozen_inputs.json")
    pred = c.read(base / "predictions.json")
    old = c.read(base / "results.json")
    records = []
    for u in [u for u in REG["units"] if u["study"] == "label_vote"]:
        subset = u["endpoint"]
        d, ep, ids, g, prepared = reader.legacy_data(unit("core", subset))
        labels = ep["all_labels"]
        macro = ep["macro_labels"]
        dim = ep["dimension"]
        roster = f["endpoints"][subset]["rows"]
        byid = {str(r["id"]): r for r in roster}
        keys = [byid[i]["key"] for i in ids]
        clusters = [f["clusters"][k] for k in keys]
        b.equal(
            g, [reader.bits(byid[i]["gold"], labels) for i in ids], "label vote gold"
        )
        for pool in ("full", "small"):
            rag = "rag_" + pool
            settings = [(rag, j) for j in range(3)] + [("label_vote", 0)]
            baseline = [pred[pool][k][dim] for k in keys]
            bp = np.asarray(
                [
                    (
                        reader.bits(x["value"], labels)
                        if x["valid"]
                        else [False] * len(labels)
                    )
                    for x in baseline
                ],
                bool,
            )
            bv = np.asarray([x["valid"] for x in baseline], bool)
            p = np.array([prepared[rag, j][0] for j in range(3)] + [bp])
            v = np.array([prepared[rag, j][1] for j in range(3)] + [bv])
            own = grouped(g, p, v, settings, labels, macro)
            previous = old["endpoints"][subset]["pools"][pool]
            b.equal(
                own["runs"][-1]["scores"]["macro"],
                previous["baseline"]["macro_f1_valid_only"],
                "label vote baseline macro",
            )
            pair = previous["common_valid_rag_minus_baseline"]
            outdir = folder / (subset + "__" + pool)
            result = compare(
                outdir,
                g,
                p,
                v,
                settings,
                labels,
                macro,
                ids,
                clusters,
                [rag],
                ["label_vote"],
                pair["seed"],
                pair,
                mode="means",
            )
            b.equal(result["n_common"], pair["rows"], "label vote common rows")
            b.equal(result["n_clusters"], pair["clusters"], "label vote clusters")
            b.save(
                outdir / "results.json",
                dict(
                    study="label_vote",
                    endpoint=subset,
                    pool=pool,
                    labels=labels,
                    macro_labels=macro,
                    ids=ids,
                    settings=settings,
                    own_valid=own,
                    contrasts=[result],
                ),
            )
            b.save_arrays(
                outdir / "analysis_arrays.npz", gold=g, predictions=p, valid=v
            )
            records.append(c.rel(outdir / "results.json"))
            print("Label vote complete", subset, pool, flush=True)
    return records


def response_vote(folder):
    load_registry()
    base = OLD / "offline_closure_20261006_v1/votes"
    raw = list(c.lines(base / "records.jsonl"))
    old = c.read(base / "results.json")
    records = []
    for u in [u for u in REG["units"] if u["study"] == "response_vote"]:
        subset = u["endpoint"]
        d, ep, ids, g, prepared = reader.legacy_data(unit("core", subset))
        labels = ep["all_labels"]
        macro = ep["macro_labels"]
        dim = ep["dimension"]
        rows = [r for r in raw if r["subset"] == subset]
        bykey = {(r["condition"], r["repeat"], str(r["item_id"])): r for r in rows}
        c.require(len(bykey) == len(rows), "Repeated vote record")
        for cond in ("rag_full", "rag_small"):
            settings = [(mode, j) for mode in ("voted", "slot0") for j in range(3)]
            p = np.zeros((6, len(g), len(labels)), bool)
            v = np.zeros((6, len(g)), bool)
            gate = v.copy()
            for j, (mode, rep) in enumerate(settings):
                for i, key in enumerate(ids):
                    r = bykey[cond, rep, key]
                    ans = r["voted"] if mode == "voted" else r["slots"][0]
                    if ans is not None and ans.get(dim) is not None:
                        v[j, i] = True
                        p[j, i] = reader.bits(ans[dim], labels)
                        gate[j, i] = ans["hate"] is True
                        if dim != "legal" and ans["hate"] is False:
                            c.require(not p[j, i].any(), "Uncleared vote labels")
            clusters = [bykey[cond, 0, key]["normalized_text_cluster"] for key in ids]
            own = grouped(
                g, p, v, settings, labels, macro, gates=gate if dim != "legal" else None
            )
            previous = old["endpoints"][subset]["conditions"][cond]
            for run in own["runs"]:
                mode, rep = run["setting"]
                orr = previous["modes"][mode]["runs"][rep]
                b.equal(
                    run["scores"]["macro"],
                    orr["macro_f1_valid_only"],
                    "response vote own macro",
                )
            pair = previous["voted_minus_slot0"]
            outdir = folder / (subset + "__" + cond)
            result = compare(
                outdir,
                g,
                p,
                v,
                settings,
                labels,
                macro,
                ids,
                clusters,
                ["voted"],
                ["slot0"],
                pair["seed"],
                pair,
                mode="means",
            )
            b.equal(
                result["n_common"], pair["n_common_valid"], "response vote common rows"
            )
            b.equal(
                result["n_clusters"], pair["n_text_clusters"], "response vote clusters"
            )
            b.save(
                outdir / "results.json",
                dict(
                    study="response_vote",
                    endpoint=subset,
                    condition=cond,
                    labels=labels,
                    macro_labels=macro,
                    ids=ids,
                    settings=settings,
                    own_valid=own,
                    contrasts=[result],
                ),
            )
            b.save_arrays(
                outdir / "analysis_arrays.npz",
                gold=g,
                predictions=p,
                valid=v,
                gates=gate,
            )
            records.append(c.rel(outdir / "results.json"))
            print("Response vote complete", subset, cond, flush=True)
    return records


def pooled(folder):
    base = OLD / "pooled_fineheads_20260930_v1"
    attempt = base / "attempts/1790803999301263800"
    old = c.read(attempt / "results.json")["endpoints"]["target_group"]
    rows = [
        r
        for r in c.lines(attempt / "predictions.jsonl")
        if r["dimension"] == "target_group"
    ]
    roster = c.read(base / "rosters.json")["target_group"]
    labels = old["roster"]["labels"]
    macro = labels
    ids = [r["key"] for r in roster]
    clusters = [r["cluster"] for r in roster]
    g = np.array([reader.bits(r["gold"], labels) for r in roster], bool)
    settings = sorted({(r["condition"], r["repeat_or_seed"]) for r in rows})
    lookup = {(r["condition"], r["repeat_or_seed"], r["key"]): r for r in rows}
    c.require(len(lookup) == len(rows), "Duplicate pooled row")
    p = np.zeros((len(settings), len(g), len(labels)), bool)
    v = np.zeros((len(settings), len(g)), bool)
    gate = v.copy()
    for j, (cond, rep) in enumerate(settings):
        for i, key in enumerate(ids):
            r = lookup[cond, rep, key]
            a = r["prediction"]
            b.equal(g[i], reader.bits(r["gold"], labels), "pooled per-answer gold")
            if a["valid"]:
                v[j, i] = True
                p[j, i] = reader.bits(a["labels"], labels)
                gate[j, i] = a["gate"] is True
    own = grouped(g, p, v, settings, labels, macro, gates=gate)
    for run in own["runs"]:
        cond, rep = run["setting"]
        orr = next(
            (x for x in old["conditions"][cond]["runs"] if x["repeat_or_seed"] == rep)
        )["own_valid"]
        b.equal(run["scores"]["macro"], orr["macro_f1"], "pooled own macro")
    draws = b.read_arrays(attempt / "bootstrap_differences.npz")
    contrasts = []
    for name, pair in old["contrasts"].items():
        result = compare(
            folder,
            g,
            p,
            v,
            settings,
            labels,
            macro,
            ids,
            clusters,
            [pair["left"]],
            [pair["right"]],
            pair["seed"],
            pair,
            mode="means",
            tag=name,
        )
        b.equal(result["n_common"], pair["common_valid_n"], "pooled common rows")
        new = b.read_arrays(ROOT / result["draw_file"])["draws"][:, 2, 0]
        b.equal(
            new,
            draws["target_group__" + name],
            "all pooled original macro bootstrap draws",
        )
        contrasts.append(result)
    b.save(
        folder / "results.json",
        dict(
            study="pooled",
            endpoint="target_group",
            labels=labels,
            macro_labels=macro,
            ids=ids,
            settings=settings,
            own_valid=own,
            contrasts=contrasts,
        ),
    )
    b.save_arrays(
        folder / "analysis_arrays.npz", gold=g, predictions=p, valid=v, gates=gate
    )
    print("Pooled target complete", flush=True)
    return [c.rel(folder / "results.json")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "study",
        choices=["synthetic", "broad", "core", "label_vote", "response_vote", "pooled"],
    )
    args = parser.parse_args()
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
                max_errors=dict(b.ERRORS),
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
                max_errors=dict(b.ERRORS),
            ),
        )
        raise


if __name__ == "__main__":
    main()

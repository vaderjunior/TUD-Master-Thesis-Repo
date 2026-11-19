from analysis.common import micro_f1_files as c
from analysis.common import micro_f1_resampling as h
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import json
import numpy as np

BASE = c.ROOT / "experiments/astra/micro_f1_sq2_matched_20261113_v1"
ANALYSIS = BASE / "analysis_v2"
OLD = c.ROOT / "experiments/astra/sq2_matched_review_20261030_v1"
PRIOR = c.ROOT / "experiments/astra/micro_f1_sq2_supervised_20261113_v1"
FIELDS = {"hate_type": "hate_types", "target_group": "target_groups", "legal": "legal"}


def save(path, value):
    path = Path(path).resolve()
    c.require(path.is_relative_to(BASE.resolve()), "Output outside new folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(
            value, f, indent=2, ensure_ascii=False, allow_nan=False, default=h.default
        )
        f.write("\n")


def arrays(path, **values):
    path = Path(path).resolve()
    c.require(path.is_relative_to(BASE.resolve()), "Array outside new folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, **values)


def freeze():
    c.require(not BASE.exists(), "Preserve earlier Part 4 output")
    reg = c.read(c.BASE / "comparison_registry_v2.json")
    manifest = c.read(c.BASE / "paired_contrast_manifest.json")
    seal = c.read(c.BASE / "complete.json")
    for name in [
        "comparison_registry_v2.json",
        "paired_contrast_manifest.json",
        "scoring_contract.json",
    ]:
        p = c.BASE / name
        c.require(c.sha(p) == seal["artifact_sha256"][c.rel(p)], "Frozen setup changed")
    previous = c.read(PRIOR / "source_preservation.json")
    for name, digest in previous["source_sha256"].items():
        c.require(c.sha(c.ROOT / name) == digest, "Earlier source changed: " + name)
    BASE.mkdir(parents=True)
    save(
        BASE / "preservation_before.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            old_sources=previous["source_sha256"],
        ),
    )
    save(
        BASE / "analysis_plan.json",
        dict(
            status="Part 4 scope fixed before new scores",
            part=4,
            units=[u for u in reg["units"] if u["study"] == "matched"],
            family=reg["new_micro_families"]["matched"],
            main_pairs=[u for u in manifest["rows"] if u["family"] == "matched"],
            scope="Four registered multilabel endpoints. Original twelve paired type/legal contrasts; all 39 setting scores. Small main-DEV 27-type and 62-target populations descriptive only; unsupported macro remains unavailable.",
            inference="10000 original paired text-cluster draws and original seed. Replay all twelve original main and source-stratified macro draws and four strict-three-vote sensitivities. Four supplementary micro questions use 98.75%; original six-question macro family stays 99.1667%. Eight remaining contrasts use nominal95.",
            averaging="Compute F1 separately per setting, then average three repeats equally within each variant and variants equally. Never pool counts across settings. Bootstrap varies texts conditional on saved variants and repeats.",
            robustness="Reproduce original common-all, by-variant, by-repeat, leave-one-repeat, by-source, leave-one-source and leave-one-variant point checks. Compare full-roster pessimistic/optimistic missing-output bounds. Strict-three-valid-vote intervals remain nominal95 sensitivities.",
            context_diagnostics="Retain all prompts in main comparisons. Reproduce changed/unchanged annotation-block strata for all three permutations, as descriptive points. Additionally describe zero versus at-least-one relevant annotated example in the reference prompt, with no new interval or subgroup selection. Label counts, extra/missed labels, gate errors and exact sets remain distinct.",
            verification="Cached vote aggregation and delivery-hash crosscheck; original gold, confusion, macro, exactness, bootstrap reproduction; independent metric arithmetic in every draw and separate sklearn verification. Check every frozen source and thesis hash before/after.",
            original_primary_metrics_unchanged=True,
            new_calls=0,
            new_training=0,
            latex_edits=0,
        ),
    )
    print("Part 4 scope frozen; earlier sources and thesis preserved.", flush=True)


def aggregate(votes):
    good = [v for v in votes if v is not None]
    if not good:
        return None
    out = {"hate": sum((v["hate"] for v in good)) > len(good) / 2}
    for dim in ("hate_type", "target_group", "legal"):
        labs = set().union(*(set(v[dim]) for v in good))
        out[dim] = sorted(
            (
                label
                for label in labs
                if sum((label in v[dim] for v in good)) > len(good) / 2
            )
        )
    sevs = sorted(
        (
            ["low", "medium", "high"].index(v["severity"])
            for v in good
            if v["severity"] is not None
        )
    )
    out["severity"] = (
        ["low", "medium", "high"][sevs[(len(sevs) - 1) // 2]] if sevs else None
    )
    if not out["hate"]:
        out.update(hate_type=[], target_group=[], severity=None)
    return out


def load():
    plan = c.read(BASE / "analysis_plan.json")
    old = c.read(OLD / "metrics.json")
    assembly = c.read(OLD / "assembly.json")
    for name in ["predictions.jsonl", "gold_rosters.jsonl"]:
        p = OLD / name
        c.require(c.sha(p) == assembly["outputs"][c.rel(p)], "Assembly cache changed")
    for receipt_name in ["metrics_receipt.json", "context_receipt.json"]:
        receipt = c.read(OLD / receipt_name)
        for name, digest in receipt["outputs"].items():
            c.require(c.sha(c.ROOT / name) == digest, "Old receipt changed: " + name)
    gold = list(c.lines(OLD / "gold_rosters.jsonl"))
    raw = list(c.lines(OLD / "predictions.jsonl"))
    delivery = {}
    for r in c.lines(OLD / "prompt_delivery.jsonl"):
        key = (r["key"], r["condition"])
        c.require(key not in delivery, "Duplicate prompt")
        delivery[key] = r
    lookup = {}
    settings = set()
    votes = Counter()
    nulls = 0
    for r in raw:
        k = (r["key"], r["condition"], r["repeat"])
        c.require(k not in lookup, "Duplicate saved answer")
        lookup[k] = r
        settings.add((r["condition"], r["repeat"]))
        votes[r["n_valid"]] += 1
        nulls += int(r["result"] is None)
        c.require(
            sum((v is not None for v in r["votes"])) == r["n_valid"],
            "Vote count differs",
        )
        c.require(
            aggregate(r["votes"]) == r["result"], "Independent aggregation differs"
        )
        c.require(
            r["messages_sha256"]
            == delivery[r["key"], r["condition"]]["messages_sha256"],
            "Prompt binding differs",
        )
    settings = sorted(settings)
    c.require(len(settings) == 39 and len(gold) == 941, "Unexpected full design")
    c.require(
        len(lookup) == len(gold) * len(settings) == 36699, "Answer matrix incomplete"
    )
    for row in gold:
        for cond, rep in settings:
            c.require(
                lookup[row["key"], cond, rep]["phase"] == row["phase"],
                "Phase or coverage mismatch",
            )
    datasets = {}
    for u in plan["units"]:
        e = old["results"][u["endpoint"]]
        ep = e["endpoint"]
        phase, dim = u["endpoint"].split("/")
        rows = [
            r
            for r in gold
            if r["phase"] == phase
            and r.get(FIELDS[dim]) is not None
            and (dim == "legal" or r["gate"] is True)
        ]
        ids = [r["key"] for r in rows]
        labels = ep["all_labels"]
        scope = ep["macro_labels"]
        c.require(ids == ep["scorable_ids"], "Gold eligibility/order mismatch")
        g = np.asarray(
            [[label in row[FIELDS[dim]] for label in labels] for row in rows], bool
        )
        p = np.zeros((len(settings), *g.shape), bool)
        v = np.zeros(p.shape[:2], bool)
        gate = np.zeros_like(v)
        nvalid = np.zeros(v.shape, np.int8)
        for j, (cond, rep) in enumerate(settings):
            for i, key in enumerate(ids):
                r = lookup[key, cond, rep]
                ans = r["result"]
                nvalid[j, i] = r["n_valid"]
                if ans is not None:
                    c.require(set(ans[dim]) <= set(labels), "Unknown prediction label")
                    v[j, i] = True
                    gate[j, i] = ans["hate"]
                    p[j, i] = [label in ans[dim] for label in labels]
        h.equal(g.sum(0), [ep["gold_support"][l] for l in labels], "gold label support")
        c.require(
            [l for l, n in zip(labels, g.sum(0)) if n >= 10] == scope,
            "Macro scope changed",
        )
        datasets[u["endpoint"]] = dict(
            unit=u,
            old=e,
            rows=rows,
            ids=ids,
            labels=labels,
            macro_labels=scope,
            indices=[labels.index(l) for l in scope],
            gold=g,
            pred=p,
            valid=v,
            gate=gate,
            nvalid=nvalid,
            clusters=[r["text_cluster"] for r in rows],
            sources=[r["source"] for r in rows],
            settings=settings,
            dimension=dim,
            phase=phase,
        )
    return (
        datasets,
        delivery,
        dict(
            saved_answers=len(raw),
            independent_vote_aggregations=len(raw),
            prompt_hash_checks=len(raw),
            full_input_count=len(gold),
            full_prompt_count=len(delivery),
            votes=dict(votes),
            invalid_answers=nulls,
        ),
    )


def summaries(d, mask=None, indices=None, valid=None):
    if indices is None:
        indices = list(range(len(d["settings"])))
    out = []
    for j in indices:
        keep = d["valid"][j] if valid is None else valid[j]
        if mask is not None:
            keep = keep & mask
        out.append(
            h.setting_summary(
                d["gold"],
                d["pred"][j],
                keep,
                d["labels"],
                d["macro_labels"],
                gate=d["gate"][j] if d["dimension"] != "legal" else None,
            )
        )
    return out


def point(d, left, right, mask=None):
    use = left + right
    keep = d["valid"][use].all(0)
    if mask is not None:
        keep = keep & mask
    c.require(keep.any(), "Empty diagnostic population")
    score = h.values(
        h.encode(
            d["gold"], d["pred"][use], np.broadcast_to(keep, (len(use), len(keep)))
        ).sum(1),
        d["indices"],
    )
    metrics = {
        name: dict(
            left=h.scalar_value(score[: len(left), j].mean()),
            right=h.scalar_value(score[len(left) :, j].mean()),
            difference=h.scalar_value(
                score[: len(left), j].mean() - score[len(left) :, j].mean()
            ),
        )
        for j, name in enumerate(h.METHODS)
    }
    return dict(n_common=int(keep.sum()), metrics=metrics)


def check_point(p, old):
    h.equal(p["n_common"], old["n"], "Original diagnostic mask")
    for new, prior in [("left", "left"), ("right", "right"), ("difference", "delta")]:
        h.equal(p["metrics"]["macro"][new], old[prior], "Original diagnostic " + new)


def missing_bounds(left, right):
    out = {}
    for method in h.METHODS:
        ll = [x["full_roster_missing_bounds"][method] for x in left]
        rr = [x["full_roster_missing_bounds"][method] for x in right]
        out[method] = (
            [
                float(np.mean([v[0] for v in ll]) - np.mean([v[1] for v in rr])),
                float(np.mean([v[1] for v in ll]) - np.mean([v[0] for v in rr])),
            ]
            if ll[0][0] is not None
            else None
        )
    return out


def bootstrap(d, left, right, old, tag, family=None, strict=False, stratified=False):
    use = left + right
    v = d["valid"][use].copy()
    if strict:
        v &= d["nvalid"][use] == 3
    boot = h.bootstrap(
        d["gold"],
        d["pred"][use],
        v,
        d["indices"],
        d["clusters"],
        len(left),
        old["seed"],
        draws=old["draws"],
        strata_sources=d["sources"] if stratified else None,
    )
    c.require(
        [i for i, on in zip(d["ids"], boot["mask"]) if on] == old["common_valid_ids"],
        "Original contrast mask changed",
    )
    h.equal(
        boot["point"][:, 0], old["common_f1_by_setting"], "Original paired macro scores"
    )
    h.equal(boot["delta"][0], old["mean_difference"], "Original macro difference")
    draws = np.load(c.bind(c.ROOT / old["bootstrap_file"]), allow_pickle=False)
    h.equal(boot["draws"][:, 2, 0], draws, "Saved macro draw replay")
    h.equal(
        h.interval(boot["draws"][:, 2, 0]),
        old["ci95"],
        "Original nominal macro interval",
    )
    if old["family_size"]:
        h.equal(
            h.interval(boot["draws"][:, 2, 0], old["family_confidence"]),
            old["family_interval"],
            "Original family macro interval",
        )
    result = h.contrast_summary(
        boot,
        [d["settings"][j] for j in left],
        [d["settings"][j] for j in right],
        family,
    )
    result["common_valid_ids"] = [i for i, on in zip(d["ids"], boot["mask"]) if on]
    result["metrics"]["macro"].update(
        original_adjusted_interval=old["family_interval"],
        original_adjusted_confidence=old["family_confidence"],
    )
    if family:
        result["metrics"]["micro"]["original_family_confidence_diagnostic"] = (
            h.interval(boot["draws"][:, 2, 1], old["family_confidence"])
        )
    ss = summaries(d, mask=boot["mask"], indices=use)
    result["left_summary"] = h.average(ss[: len(left)])
    result["right_summary"] = h.average(ss[len(left) :])
    vv = d["valid"].copy()
    if strict:
        vv &= d["nvalid"] == 3
    bounds = missing_bounds(
        summaries(d, indices=left, valid=vv), summaries(d, indices=right, valid=vv)
    )
    h.equal(
        bounds["macro"],
        old["full_roster_failure_difference_bounds"],
        "Original failure bounds",
    )
    result["full_roster_failure_difference_bounds"] = bounds
    result["paired_setting_summaries"] = dict(
        zip(
            [d["settings"][j][0] + "/repeat" + str(d["settings"][j][1]) for j in use],
            ss,
        )
    )
    path = ANALYSIS / "draws" / (h.slug(d["unit"]["endpoint"] + "|" + tag) + ".npz")
    arrays(path, draws=boot["draws"], point=boot["point"], mask=boot["mask"])
    result["draw_file"] = c.rel(path)
    result["original_macro_draws_verified"] = len(draws)
    return result


def analysis():
    plan = c.read(BASE / "analysis_plan.json")
    data, delivery, reader = load()
    context = c.read(OLD / "context_review.json")
    results = {}
    for endpoint, d in data.items():
        own = summaries(d)
        all_mask = d["valid"].all(0)
        common = summaries(d, mask=all_mask)
        snames = [a + "/repeat" + str(b) for a, b in d["settings"]]
        slookup = {v: i for i, v in enumerate(d["settings"])}
        for sn, new in zip(snames, own):
            old = d["old"]["per_condition_repeat"][sn]
            h.equal(
                new["tp_fp_fn"], old["per_label_counts"], "Original confusion counts"
            )
            h.equal(
                new["scores"]["macro"], old["macro_f1_valid_only"], "Original own macro"
            )
            h.equal(
                new["set_only_exact"], old["exact_set_accuracy"], "Original exact sets"
            )
            if d["indices"]:
                h.equal(
                    new["full_roster_missing_bounds"]["macro"],
                    old["invalid_inclusive_bounds"],
                    "Original own failure bounds",
                )
            else:
                c.require(
                    new["full_roster_missing_bounds"]["macro"]
                    == old["invalid_inclusive_bounds"]
                    == [None, None],
                    "Unsupported macro bounds changed",
                )
        grouped = {}
        for name in sorted({x[0] for x in d["settings"]}):
            ix = [j for j, k in enumerate(d["settings"]) if k[0] == name]
            grouped[name] = dict(
                own_valid=h.average([own[j] for j in ix]),
                common_all=h.average([common[j] for j in ix]),
            )
        result = dict(
            unit=d["unit"],
            ids=d["ids"],
            labels=d["labels"],
            macro_labels=d["macro_labels"],
            settings=d["settings"],
            n=len(d["gold"]),
            n_common_all=int(all_mask.sum()),
            gold_support=d["gold"].sum(0),
            own_setting_summaries=dict(zip(snames, own)),
            common_all_setting_summaries=dict(zip(snames, common)),
            condition_means=grouped,
            contrasts={},
            context_diagnostics={},
        )
        arr = ANALYSIS / "arrays" / (h.slug(endpoint) + ".npz")
        arrays(
            arr,
            gold=d["gold"],
            pred=d["pred"],
            valid=d["valid"],
            gate=d["gate"],
            nvalid=d["nvalid"],
            ids=np.asarray(d["ids"]),
            clusters=np.asarray(d["clusters"]),
            sources=np.asarray(d["sources"]),
            settings=np.asarray(snames),
            labels=np.asarray(d["labels"]),
            macro_labels=np.asarray(d["macro_labels"]),
        )
        result["array_file"] = c.rel(arr)
        for family, old in d["old"]["contrasts"].items():
            lhs = [slookup[tuple(x)] for x in old["left_settings"]]
            rhs = [slookup[tuple(x)] for x in old["right_settings"]]
            primary = "matched/" + endpoint + "|" + family in plan["family"]["members"]
            main = bootstrap(
                d,
                lhs,
                rhs,
                old,
                family + "main",
                plan["family"]["m"] if primary else None,
            )
            main["source_stratified"] = bootstrap(
                d,
                lhs,
                rhs,
                old["source_stratified"],
                family + "stratified",
                stratified=True,
            )
            robust = d["old"]["robustness"][family]
            new = {}
            for kind, previous in robust.items():
                if kind == "three_valid_votes":
                    new[kind] = bootstrap(
                        d, lhs, rhs, previous, family + "strict", strict=True
                    )
                    continue
                if kind == "common_all":
                    val = point(d, lhs, rhs, all_mask)
                    check_point(val, previous)
                    new[kind] = val
                    continue
                new[kind] = {}
                for key, expected in previous.items():
                    ll, rr, mask = (lhs, rhs, None)
                    if kind == "by_variant":
                        rr = [j for j in rhs if d["settings"][j][0] == key]
                    elif kind == "leave_one_variant_out":
                        rr = [j for j in rhs if d["settings"][j][0] != key]
                    elif kind == "by_repeat":
                        ll = [j for j in lhs if d["settings"][j][1] == int(key)]
                        rr = [j for j in rhs if d["settings"][j][1] == int(key)]
                    elif kind == "leave_one_repeat_out":
                        ll = [j for j in lhs if d["settings"][j][1] != int(key)]
                        rr = [j for j in rhs if d["settings"][j][1] != int(key)]
                    elif kind == "by_source":
                        mask = np.asarray(d["sources"]) == key
                    elif kind == "leave_one_source_out":
                        mask = np.asarray(d["sources"]) != key
                    else:
                        raise ValueError("Unknown diagnostic " + kind)
                    val = point(d, ll, rr, mask)
                    check_point(val, expected)
                    new[kind][key] = val
            main["robustness"] = new
            result["contrasts"][family] = main
            print(
                endpoint,
                family,
                "macro/micro pp",
                *(
                    round(main["metrics"][x]["difference"] * 100, 5)
                    for x in ["macro", "micro"]
                ),
                flush=True
            )
        if d["old"]["contrasts"]:
            lhs = [
                j for j, k in enumerate(d["settings"]) if k[0] == "retrieved_complete"
            ]
            for cond in ["permuted_0", "permuted_1", "permuted_2"]:
                rhs = [j for j, k in enumerate(d["settings"]) if k[0] == cond]
                changed = np.asarray(
                    [
                        delivery[key, cond]["changed_annotation_slots"] > 0
                        for key in d["ids"]
                    ]
                )
                for name, mask in [("changed", changed), ("unchanged", ~changed)]:
                    val = point(d, lhs, rhs, mask)
                    check_point(val, context["mutation_strata"][d["phase"]][cond][name])
                    result["context_diagnostics"][cond + "/" + name] = val
            annotated = np.asarray(
                [
                    delivery[key, "retrieved_complete"]["annotated_slots"].get(
                        FIELDS[d["dimension"]], 0
                    )
                    > 0
                    for key in d["ids"]
                ]
            )
            if d["dimension"] != "legal":
                annotated = np.asarray(
                    [
                        delivery[key, "retrieved_complete"]["annotated_slots"].get(
                            d["dimension"], 0
                        )
                        > 0
                        for key in d["ids"]
                    ]
                )
            for family in ["coverage_matched_", "permuted_"]:
                rhs = [
                    j for j, k in enumerate(d["settings"]) if k[0].startswith(family)
                ]
                for name, mask in [
                    ("no_relevant_annotation", ~annotated),
                    ("some_relevant_annotation", annotated),
                ]:
                    if mask.any():
                        result["context_diagnostics"][family + "/" + name] = point(
                            d, lhs, rhs, mask
                        )
        save(ANALYSIS / "endpoints" / (h.slug(endpoint) + ".json"), result)
        results[endpoint] = dict(
            path=c.rel(ANALYSIS / "endpoints" / (h.slug(endpoint) + ".json")),
            arrays=c.rel(arr),
        )
    for path in [
        Path(__file__),
        Path(h.__file__),
        Path(c.__file__),
        c.ROOT / "analysis/common/micro_f1.py",
    ]:
        c.bind(path)
    save(
        ANALYSIS / "analysis_index.json",
        dict(
            status="Part 4 numerical analysis complete; independent review pending",
            reader=reader,
            endpoints=results,
            checks=dict(h.CHECKS),
            max_errors=dict(h.ERRORS),
            source_sha256=c.BINDINGS,
            new_calls=0,
            new_training=0,
            latex_edits=0,
        ),
    )


if __name__ == "__main__":
    import sys

    {"freeze": freeze, "analyse": analysis}[sys.argv[1]]()

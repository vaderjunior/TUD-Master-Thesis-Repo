"""Micro f1 banks strategies."""

from analysis.common import micro_f1_files as c
from analysis.common import micro_f1_resampling as h
from analysis.common import micro_f1_reader as reader
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import re
import unicodedata
import numpy as np

BASE = c.ROOT / "experiments/astra/micro_f1_sq1_banks_strategies_20261113_v1"
PRIOR = c.ROOT / "experiments/astra/micro_f1_sq2_matched_20261113_v1"
REUSE = c.ROOT / "experiments/astra/micro_f1_sq2_retrieval_20261113_v1"
CONFIGS = {
    "strategy": "sq1_full_kb_strategy_20260920_v1",
    "pools": "sq1_pool_size_composition_20260926_v1",
}


def save(path, value):
    path = Path(path).resolve()
    c.require(path.is_relative_to(BASE.resolve()), "Write outside Part 5 folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(
            value, f, ensure_ascii=False, indent=2, allow_nan=False, default=h.default
        )
        f.write("\n")


def arrays(path, **values):
    path = Path(path).resolve()
    c.require(path.is_relative_to(BASE.resolve()), "Array outside Part 5 folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, **values)


def freeze():
    c.require(not BASE.exists(), "Preserve previous Part 5 work")
    registry = c.read(c.BASE / "comparison_registry_v2.json")
    manifest = c.read(c.BASE / "paired_contrast_manifest.json")
    seal = c.read(c.BASE / "complete.json")
    for name in [
        "comparison_registry_v2.json",
        "paired_contrast_manifest.json",
        "scoring_contract.json",
    ]:
        path = c.BASE / name
        c.require(
            c.sha(path) == seal["artifact_sha256"][c.rel(path)], "Frozen setup changed"
        )
    old = c.read(PRIOR / "source_preservation.json")
    for name, digest in old["source_sha256"].items():
        c.require(c.sha(c.ROOT / name) == digest, "Earlier source changed: " + name)
    changes = dict()
    BASE.mkdir(parents=True)
    save(
        BASE / "preservation_before.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            old_sources=old["source_sha256"],
        ),
    )
    save(
        BASE / "analysis_plan.json",
        dict(
            status="Part 5 scope fixed before new scores",
            part=5,
            units=[u for u in registry["units"] if u["study"] in ["strategy", "pools"]],
            reuse_units=[u for u in registry["units"] if u["study"] == "core"],
            new_families={
                k: registry["new_micro_families"][k] for k in ["strategy", "pools"]
            },
            main_pairs=[
                r for r in manifest["rows"] if r["family"] in ["strategy", "pools"]
            ],
            scope="Eight new multilabel endpoint views: four strategy and four pool-composition views. Reuse the four accepted Part 2 core views and their 23 comparisons without new scoring or counting them as new evidence. Binary hate and ordinal severity unchanged.",
            inference="Reproduce all 44 original multilabel macro contrasts and available common-setting/group intervals from original 10000 item draws and seeds. Strategy mask valid in all nine settings; pools valid in all 33 settings. New micro families m=3 strategy focal types, m=4 pools type/legal. Original macro families remain m=3 and m=6. Other contrasts nominal95.",
            averaging="F1 per decoder setting; equal three-repeat means within condition, then equal weights over the three sampled pools. Shared full/curated references have three real repeats, never nine. Keep conditional text uncertainty separate from pool/repeat variation.",
            diagnostics="Own-valid and all-setting-common scores, exact sets, gate omissions, per-label counts/weights/contributions, full-roster missing-output bounds. By-repeat/leave-one-repeat and by-pool/leave-one-pool points use the original global common mask. All original contrasts retained, including adverse/null ones.",
            sensitivity="Strict-three-valid-vote global mask per endpoint with nominal95 intervals, not replacement primary scoring. Inspect normalized-text duplicates; cluster resampling only if duplicates occur, otherwise document identity. Source-stratified text resampling only if multiple sources occur; otherwise document identity. Same 10000 draws/seed. These sensitivities are supporting, not new adjusted families.",
            context="Reuse documented annotation census and delivery/retrieval diagnostics, including 13 changed reopened pool prompts. No new retrieval. Relate micro to count/label weighting without treating annotation availability as randomised or selecting an optimal pool/strategy.",
            verification="Raw assembled answers and original parquet gold; original macro replay; independent sklearn and separately expanded bootstrap arithmetic. Source/thesis preservation and additive writer/session logs.",
            original_primary_metrics_unchanged=True,
            new_calls=0,
            new_training=0,
            latex_edits=0,
        ),
    )
    print(
        "Part 5 frozen; source hashes passed. Thesis changes since Part 4:",
        changes,
        flush=True,
    )


def normalized(text):
    return re.sub(
        "\\s+", " ", unicodedata.normalize("NFKC", str(text)).casefold()
    ).strip()


def load(u):
    import pyarrow.parquet as pq

    old, ep, ids, g, prepared = reader.legacy_data(u)
    config = c.read(
        c.ROOT
        / "experiments/astra"
        / CONFIGS[u["study"]]
        / "api_execution_v1/analysis_plan.json"
    )
    conditions = config["conditions"]
    settings = [(cond, j) for cond in conditions for j in range(3)]
    c.require(set(settings) == set(prepared), "Setting inventory differs")
    pred = np.array([prepared[k][0] for k in settings])
    valid = np.array([prepared[k][1] for k in settings])
    gate = np.zeros_like(valid)
    nvalid = np.zeros(valid.shape, np.int8)
    raw = {}
    seen = 0
    for path in u["assembled"]:
        rr = list(c.lines(c.ROOT / path))
        key = (rr[0]["condition"], rr[0]["replicate"])
        j = settings.index(key)
        lookup = {str(r["id"]): r for r in rr}
        for i, rid in enumerate(ids):
            row = lookup[rid]
            nvalid[j, i] = row["n_valid"]
            ans = row["result"]
            raw[key, rid] = row
            seen += 1
            if ans is not None:
                gate[j, i] = ans["hate"]
            c.require((ans is not None) == bool(valid[j, i]), "Answer validity differs")
            c.require((nvalid[j, i] > 0) == bool(valid[j, i]), "Vote validity differs")
    rows = pq.read_table(c.bind(c.ROOT / u["gold_path"])).to_pylist()
    byid = {str(r["id"]): r for r in rows}
    rows = [byid[i] for i in ids]
    clusters = [
        hashlib.sha256(normalized(r["text"]).encode("utf8")).hexdigest() for r in rows
    ]
    sources = [str(r.get("source", "unknown")) for r in rows]
    groups = (
        config["groups"]
        if u["study"] == "pools"
        else {cond: [cond] for cond in conditions}
    )
    mask = valid.all(0)
    pair = old["paired_comparison"]
    c.require(
        [rid for rid, on in zip(ids, mask) if on] == pair["common_valid_ids"],
        "Original global common mask changed",
    )
    c.require(
        list(groups) == list(pair.get("groups", groups)), "Original group order differs"
    )
    return dict(
        unit=u,
        old=old,
        ep=ep,
        ids=ids,
        gold=g,
        pred=pred,
        valid=valid,
        gate=gate,
        nvalid=nvalid,
        labels=ep["all_labels"],
        macro_labels=ep["macro_labels"],
        indices=[ep["all_labels"].index(l) for l in ep["macro_labels"]],
        settings=settings,
        conditions=conditions,
        groups=groups,
        mask=mask,
        clusters=clusters,
        sources=sources,
        raw_records_checked=seen,
    )


def setting_summaries(d, mask=None):
    return [
        h.setting_summary(
            d["gold"],
            d["pred"][j],
            d["valid"][j] if mask is None else d["valid"][j] & mask,
            d["labels"],
            d["macro_labels"],
            d["gate"][j] if d["unit"]["dimension"] != "legal" else None,
        )
        for j in range(len(d["settings"]))
    ]


def group_positions(d, name, repeat=None, omit_repeat=None, pool=None, omit_pool=None):
    members = d["groups"][name]
    return [
        i
        for i, (cond, rep) in enumerate(d["settings"])
        if cond in members
        and (repeat is None or rep == repeat)
        and (omit_repeat is None or rep != omit_repeat)
        and (pool is None or cond == pool)
        and (omit_pool is None or cond != omit_pool)
    ]


def group_summary(d, summaries, group):
    positions = group_positions(d, group)
    ss = [summaries[i] for i in positions]
    mean = h.average(ss)
    pools = {
        name: h.average(
            [summaries[i] for i, k in enumerate(d["settings"]) if k[0] == name]
        )
        for name in d["groups"][group]
    }
    for metric in h.METHODS:
        h.equal(
            mean["scores"][metric],
            np.mean([x["scores"][metric] for x in pools.values()]),
            "Hierarchical group mean",
        )
    return dict(
        mean=mean,
        pools=pools,
        between_pool_sd={
            k: (
                float(np.std([x["scores"][k] for x in pools.values()], ddof=1))
                if len(pools) > 1
                else None
            )
            for k in h.METHODS
        },
        between_pool_range={
            k: [
                min((x["scores"][k] for x in pools.values())),
                max((x["scores"][k] for x in pools.values())),
            ]
            for k in h.METHODS
        },
    )


def boot(d, mask, mode="item"):
    seed = d["old"]["paired_comparison"]["bootstrap_seed"]
    draws = d["old"]["paired_comparison"]["bootstrap_draws"]
    counts = h.encode(d["gold"], d["pred"], d["valid"])[:, mask]
    if mode == "cluster":
        names = sorted(set(np.array(d["clusters"])[mask]))
        ix = {k: j for j, k in enumerate(names)}
        grouped = np.zeros(
            (len(d["settings"]), len(names), len(d["labels"]), 3), np.int64
        )
        for i, k in enumerate(np.array(d["clusters"])[mask]):
            grouped[:, ix[k]] += counts[:, i]
    else:
        grouped = counts
        names = list(range(int(mask.sum())))
    groups = defaultdict(list)
    if mode == "source_stratified":
        for j, src in enumerate(np.array(d["sources"])[mask]):
            groups[src].append(j)
    else:
        groups["all"] = list(range(len(names)))
    flat = grouped.transpose(1, 0, 2, 3).reshape(len(names), -1).astype(float)
    alt = np.stack(
        [
            grouped[..., 0],
            grouped[..., 0] + grouped[..., 1],
            grouped[..., 0] + grouped[..., 2],
        ],
        axis=-1,
    )
    altflat = alt.transpose(1, 0, 2, 3).reshape(len(names), -1).astype(float)
    rng = np.random.default_rng(seed)
    saved = []
    direct = 0
    error = 0.0
    for start in range(0, draws, 64):
        size = min(64, draws - start)
        picks = (
            rng.integers(0, len(names), (size, len(names)))
            if mode == "item"
            else np.concatenate(
                [
                    rng.choice(v, (size, len(v)), replace=True)
                    for _, v in sorted(groups.items())
                ],
                axis=1,
            )
        )
        freq = np.array([np.bincount(x, minlength=len(names)) for x in picks])
        total = (
            (freq @ flat)
            .reshape(size, len(d["settings"]), len(d["labels"]), 3)
            .astype(np.int64)
        )
        c.require(
            np.array_equal(total[0], grouped[:, picks[0]].sum(1)),
            "Direct draw counts differ",
        )
        direct += 1
        scores = h.values(total, d["indices"])
        alternate = (freq @ altflat).reshape(
            size, len(d["settings"]), len(d["labels"]), 3
        )
        tp, pp, aa = np.moveaxis(alternate, -1, 0)
        den = pp + aa
        lf = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den != 0)

        def micro(ix):
            dd = den[..., ix].sum(-1)
            num = 2 * tp[..., ix].sum(-1)
            return np.divide(num, dd, out=np.zeros_like(num), where=dd != 0)

        other = np.stack(
            [
                lf[..., d["indices"]].mean(-1),
                micro(list(range(len(d["labels"])))),
                micro(d["indices"]),
            ],
            -1,
        )
        error = max(error, float(np.max(np.abs(scores - other))))
        c.require(error < 1e-14, "Independent draw arithmetic differs")
        saved.append(scores)
    return dict(
        scores=np.concatenate(saved),
        mask=mask,
        seed=seed,
        draws=draws,
        n_units=len(names),
        mode=mode,
        checks=dict(
            direct_expansions=direct,
            independent_setting_metric_values=draws * len(d["settings"]) * 3,
            maximum_error=error,
        ),
    )


def difference(d, score, left, right, **filters):
    li = group_positions(d, left, **filters)
    ri = group_positions(d, right, **filters)
    if d["unit"]["study"] == "pools":
        return (score[..., li, None, :] - score[..., None, ri, :]).mean(axis=(-3, -2))
    return score[..., li, :].mean(-2) - score[..., ri, :].mean(-2)


def missing_bounds(d, summaries, left, right):
    out = {}
    for metric in h.METHODS:
        ll = np.mean(
            [
                summaries[j]["full_roster_missing_bounds"][metric]
                for j in group_positions(d, left)
            ],
            0,
        )
        rr = np.mean(
            [
                summaries[j]["full_roster_missing_bounds"][metric]
                for j in group_positions(d, right)
            ],
            0,
        )
        out[metric] = [float(ll[0] - rr[1]), float(ll[1] - rr[0])]
    return out


def contrast(d, old, point, bootstraps, own, family):
    left, right = (old["left"], old["right"])
    delta = difference(d, point, left, right)
    bi = bootstraps["item"]
    bd = difference(d, bi["scores"], left, right)
    h.equal(delta[0], old["mean_difference"], "Original paired macro point")
    h.equal(h.interval(bd[:, 0]), old["ci95"], "Original paired macro95 interval")
    if old["family_ci"]:
        h.equal(
            h.interval(bd[:, 0], old["family_confidence"]),
            old["family_ci"],
            "Original macro adjusted interval",
        )
    bounds = missing_bounds(d, own, left, right)
    h.equal(
        bounds["macro"],
        old["invalid_inclusive_difference_bounds"],
        "Original full-roster failure bounds",
    )
    result = dict(
        left=left,
        right=right,
        n_common=int(d["mask"].sum()),
        family_size=family,
        metrics={},
        full_roster_missing_bounds=bounds,
        by_repeat={},
        leave_one_repeat_out={},
        by_pool={},
        leave_one_pool_out={},
        sensitivities={},
    )
    for j, metric in enumerate(h.METHODS):
        result["metrics"][metric] = dict(
            left=float(point[group_positions(d, left), j].mean()),
            right=float(point[group_positions(d, right), j].mean()),
            difference=float(delta[j]),
            ci95=h.interval(bd[:, j]),
            adjusted_interval=(
                h.interval(bd[:, j], 1 - 0.05 / family)
                if family and metric != "macro"
                else old["family_ci"] if metric == "macro" else None
            ),
            adjusted_confidence=(
                1 - 0.05 / family
                if family and metric != "macro"
                else old["family_confidence"] if metric == "macro" else None
            ),
            original_confidence_diagnostic=(
                h.interval(bd[:, j], old["family_confidence"])
                if old["family_confidence"] and metric != "macro"
                else None
            ),
        )
    for rep in range(3):
        result["by_repeat"][str(rep)] = dict(
            zip(h.METHODS, difference(d, point, left, right, repeat=rep))
        )
        result["leave_one_repeat_out"][str(rep)] = dict(
            zip(h.METHODS, difference(d, point, left, right, omit_repeat=rep))
        )
    if d["unit"]["study"] == "pools":
        for seed in [42, 43, 44]:

            def select(name, omit=False):
                members = d["groups"][name]
                if len(members) == 1:
                    return group_positions(d, name)
                return [
                    j
                    for j in group_positions(d, name)
                    if d["settings"][j][0].endswith("_s" + str(seed)) != omit
                ]

            for key, omit in [("by_pool", False), ("leave_one_pool_out", True)]:
                li, ri = (select(left, omit), select(right, omit))
                dd = point[li].mean(0) - point[ri].mean(0)
                result[key][str(seed)] = dict(zip(h.METHODS, dd))
    for mode, b in bootstraps.items():
        if mode == "item":
            continue
        pp = h.values(
            h.encode(
                d["gold"], d["pred"], np.broadcast_to(b["mask"], d["valid"].shape)
            ).sum(1),
            d["indices"],
        )
        dd = difference(d, pp, left, right)
        sample = difference(d, b["scores"], left, right)
        result["sensitivities"][mode] = dict(
            n_common=int(b["mask"].sum()),
            n_units=b["n_units"],
            metrics={
                m: dict(difference=float(dd[j]), ci95=h.interval(sample[:, j]))
                for j, m in enumerate(h.METHODS)
            },
        )
    return result


def analyse():
    plan = c.read(BASE / "analysis_plan.json")
    outputs = []
    total_raw = 0
    total_bootstrap = 0
    reused = c.read(REUSE / "accepted_analysis_index.json")["core"]
    reuse_seal = c.read(REUSE / "complete.json")
    c.require(reused["status"] == "passed", "Earlier core not accepted")
    references = []
    for name in reused["outputs"]:
        path = c.ROOT / name
        c.require(
            c.sha(path) == reuse_seal["artifact_sha256"][name],
            "Earlier core output changed",
        )
        old = c.read(path)
        references.append(
            dict(
                endpoint=old["endpoint"],
                source=name,
                sha256=c.sha(path),
                comparisons=len(old["contrasts"]),
                already_completed_part=2,
            )
        )
    save(
        BASE / "reused_core_index.json",
        dict(
            status="Existing accepted results reused without recalculation",
            records=references,
            already_counted_in_part2=True,
            source_index=c.rel(REUSE / "accepted_analysis_index.json"),
        ),
    )
    for u in plan["units"]:
        d = load(u)
        own = setting_summaries(d)
        common = setting_summaries(d, d["mask"])
        total_raw += d["raw_records_checked"]
        point = h.values(
            h.encode(
                d["gold"], d["pred"], np.broadcast_to(d["mask"], d["valid"].shape)
            ).sum(1),
            d["indices"],
        )
        old = d["old"]
        pair = old["paired_comparison"]
        for j, (cond, rep) in enumerate(d["settings"]):
            previous = next(
                (
                    x["score"]
                    for x in old["conditions"][cond]["runs"]
                    if x["replicate"] == rep
                )
            )
            h.equal(
                own[j]["scores"]["macro"],
                previous["macro_f1_valid_only"],
                "Original own macro",
            )
            h.equal(
                own[j]["tp_fp_fn"], previous["per_label_counts"], "Original own counts"
            )
            h.equal(own[j]["n_valid"], previous["n_valid"], "Original valid count")
            h.equal(
                own[j]["full_roster_missing_bounds"]["macro"],
                previous["invalid_inclusive_bounds"],
                "Original own failure bounds",
            )
            h.equal(
                point[j, 0],
                pair["conditions"][cond]["common_valid_summary"]["values"][rep],
                "Original global common macro",
            )
            if "common_runs" in old["conditions"][cond]:
                expected = next(
                    (
                        x["score"]
                        for x in old["conditions"][cond]["common_runs"]
                        if x["replicate"] == rep
                    )
                )
                h.equal(
                    common[j]["set_only_exact"],
                    expected["exact_match_rate"],
                    "Original common exact sets",
                )
        bs = {"item": boot(d, d["mask"])}
        strict = d["mask"] & (d["nvalid"] == 3).all(0)
        bs["strict_three_votes"] = boot(d, strict)
        if len(set(np.array(d["clusters"])[d["mask"]])) < int(d["mask"].sum()):
            bs["cluster"] = boot(d, d["mask"], "cluster")
        if len(set(np.array(d["sources"])[d["mask"]])) > 1:
            bs["source_stratified"] = boot(d, d["mask"], "source_stratified")
        folder = BASE / u["study"] / u["endpoint"]
        files = {}
        for mode, b in bs.items():
            path = folder / (mode + "_bootstrap.npz")
            arrays(path, scores=b["scores"], mask=b["mask"])
            total_bootstrap += b["scores"].size
            files[mode] = dict(
                path=c.rel(path),
                seed=b["seed"],
                draws=b["draws"],
                n_units=b["n_units"],
                checks=b["checks"],
            )
        groups = {}
        for group in d["groups"]:
            positions = group_positions(d, group)
            gs = group_summary(d, common, group)
            gs["interval95"] = {
                m: h.interval(bs["item"]["scores"][:, positions, j].mean(1))
                for j, m in enumerate(h.METHODS)
            }
            if u["study"] == "pools":
                prior = pair["groups"][group]
                h.equal(
                    gs["mean"]["scores"]["macro"], prior["mean"], "Original group mean"
                )
                h.equal(
                    gs["between_pool_sd"]["macro"],
                    prior["between_pool_sample_sd"],
                    "Original between-pool SD",
                )
                h.equal(
                    gs["interval95"]["macro"], prior["ci95"], "Original group interval"
                )
            groups[group] = dict(common=gs, own_valid=group_summary(d, own, group))
        if u["study"] == "pools":
            for cond, previous in pair["conditions"].items():
                ix = [i for i, k in enumerate(d["settings"]) if k[0] == cond]
                h.equal(
                    [h.interval(bs["item"]["scores"][:, i, 0]) for i in ix],
                    previous["decoder_ci95"],
                    "Original per-repeat interval",
                )
                h.equal(
                    h.interval(bs["item"]["scores"][:, ix, 0].mean(1)),
                    previous["mean_ci95"],
                    "Original pool interval",
                )
        comparisons = {}
        for name, oldc in pair["contrasts"].items():
            qid = u["id"] + "|paired|" + name
            family = (
                plan["new_families"][u["study"]]["m"]
                if qid in plan["new_families"][u["study"]]["members"]
                else None
            )
            comparisons[name] = contrast(d, oldc, point, bs, own, family)
            comparisons[name]["question_id"] = qid if family else None
        arr = folder / "arrays.npz"
        arrays(
            arr,
            gold=d["gold"],
            pred=d["pred"],
            valid=d["valid"],
            gate=d["gate"],
            nvalid=d["nvalid"],
            ids=np.asarray(d["ids"]),
            settings=np.asarray(d["settings"]),
            labels=np.asarray(d["labels"]),
            macro_labels=np.asarray(d["macro_labels"]),
            common=d["mask"],
            clusters=np.asarray(d["clusters"]),
            sources=np.asarray(d["sources"]),
        )
        result = dict(
            unit=u,
            settings=d["settings"],
            group_members=d["groups"],
            labels=d["labels"],
            macro_labels=d["macro_labels"],
            ids=d["ids"],
            common_ids=[i for i, on in zip(d["ids"], d["mask"]) if on],
            n_eligible=len(d["ids"]),
            n_common=int(d["mask"].sum()),
            source_counts=dict(Counter(d["sources"])),
            unique_common_clusters=len(set(np.array(d["clusters"])[d["mask"]])),
            own=own,
            common=common,
            groups=groups,
            contrasts=comparisons,
            arrays=c.rel(arr),
            bootstrap_files=files,
            validity_histogram=dict(Counter(d["nvalid"].ravel().tolist())),
            strict_common=int(strict.sum()),
            new_calls=0,
            new_training=0,
        )
        save(folder / "results.json", result)
        outputs.append(c.rel(folder / "results.json"))
        print(
            u["id"],
            "complete:",
            d["mask"].sum(),
            "shared; macro/micro",
            {
                k: [
                    round(v["common"]["mean"]["scores"][m] * 100, 4)
                    for m in ["macro", "micro"]
                ]
                for k, v in groups.items()
            },
            flush=True,
        )
    for path in [
        Path(__file__),
        Path(reader.__file__),
        Path(h.__file__),
        Path(c.__file__),
        c.ROOT / "analysis/common/micro_f1.py",
    ]:
        c.bind(path)
    save(
        BASE / "analysis_index.json",
        dict(
            status="Part 5 numerical analysis complete; independent review pending",
            outputs=outputs,
            reused_core=c.rel(BASE / "reused_core_index.json"),
            raw_eligible_answer_records=total_raw,
            stored_bootstrap_metric_values=total_bootstrap,
            checks=dict(h.CHECKS),
            max_errors=dict(h.ERRORS),
            reader_checks=dict(reader.CHECKS),
            source_sha256=c.BINDINGS,
            new_calls=0,
            new_training=0,
            latex_edits=0,
        ),
    )


if __name__ == "__main__":
    import sys

    {"freeze": freeze, "analyse": analyse}[sys.argv[1]]()

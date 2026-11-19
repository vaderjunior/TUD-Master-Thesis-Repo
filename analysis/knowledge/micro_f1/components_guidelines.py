"""Micro f1 components guidelines."""

from analysis.common import micro_f1_files as c
from analysis.common import micro_f1_resampling as h
from analysis.common import micro_f1_reader as reader
from analysis.knowledge.micro_f1 import banks_strategies as previous
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np

BASE = c.ROOT / "experiments/astra/micro_f1_sq1_components_guidelines_20261113_v1"
PRIOR = previous.BASE
STUDIES = ["components", "allocation", "german_guidelines"]
GERMAN_READER = c.ROOT / "experiments/astra/german_guidelines_reader_review_20261112_v1"
GERMAN_ROBUST = (
    c.ROOT / "experiments/astra/german_guidelines_robustness_analysis_20261113_v1"
)
CONFIGS = {
    "components": "sq1_full_kb_components_20260921_v1",
    "allocation": "sq1_full_kb_allocation_20260924_v1",
}


def save(path, value):
    path = Path(path).resolve()
    c.require(path.is_relative_to(BASE.resolve()), "Write outside Part 6")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(
            value, f, indent=2, ensure_ascii=False, allow_nan=False, default=h.default
        )
        f.write("\n")


def arrays(path, **values):
    path = Path(path).resolve()
    c.require(path.is_relative_to(BASE.resolve()), "Array outside Part 6")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, **values)


def freeze():
    c.require(not BASE.exists(), "Preserve existing Part 6 work")
    registry = c.read(c.BASE / "comparison_registry_v2.json")
    manifest = c.read(c.BASE / "paired_contrast_manifest.json")
    seal = c.read(c.BASE / "complete.json")
    for name in [
        "comparison_registry_v2.json",
        "paired_contrast_manifest.json",
        "scoring_contract.json",
    ]:
        p = c.BASE / name
        c.require(c.sha(p) == seal["artifact_sha256"][c.rel(p)], "Frozen setup changed")
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
            status="Part 6 scope fixed before new scores",
            part=6,
            units=[u for u in registry["units"] if u["study"] in STUDIES],
            new_families={k: registry["new_micro_families"][k] for k in STUDIES},
            main_pairs=[r for r in manifest["rows"] if r["family"] in STUDIES],
            scope="All 68 component, 12 allocation and six German legal contrasts; ten registered units and fourteen distinct paired views. Binary hate, ordinal severity and benign-only deliberate-guideline FPR remain unchanged.",
            masks="Components: 24-setting eight-condition factorial mask and separate six-setting wording bridge. Allocation: all nine settings. German: original all-nine common mask, legal independent of hate. Keep own-valid means separate.",
            inference="Original 10000 draws and seeds; preserve every original contrast weight including factorial interactions. Three supplementary micro families of three questions each. Original macro families unchanged (component 3, allocation 6 including hate, German 4 including hate). Other intervals nominal95. Full and macro-label micro identical for all ten units.",
            german_reuse="Verify existing saved micro points, intervals and missing-output bounds, and replay saved bootstrap confusions. Add the prelisted micro-only TEST family. No claim that existing micro scores are a new experiment or independent replication.",
            diagnostics="Own/common scores, label counts and arithmetic contributions, exact sets, fine-field gate misses, signed full-roster missing bounds, per-repeat and leave-one-repeat points. No selection or tuning with micro.",
            sensitivity="Strict-three-valid-vote symmetric masks per view with supporting 95% intervals; source-stratified resampling when multiple sources, cluster resampling only if normalized duplicate texts. German no-repair/no-hate-tie masks retain descriptive point-only role from original robustness analysis. No outcome-selected exclusions.",
            verification="Raw saved answers and gold, original macro points and intervals, independent sklearn and frequency arithmetic, seeded draw reproduction, synthetic weighted-contrast/mask/bound fixtures, preservation of prior evidence and current thesis.",
            writing="Append plain-language findings and exact placement instructions to the existing handover. Keep primary macro figures and original confidence roles. No LaTeX editing and no unnecessary new plots.",
            new_calls=0,
            new_training=0,
            new_retrieval=0,
            latex_edits=0,
        ),
    )


def verified_run(root):
    complete = c.read(root / "complete.json")
    mp = c.ROOT / complete["manifest"]
    manifest = c.read(mp)
    if c.rel(mp) in complete["bindings"]:
        c.require(c.sha(mp) == complete["bindings"][c.rel(mp)], "Manifest seal changed")
    for name, digest in manifest["outputs"].items():
        p = c.bind(c.ROOT / name)
        c.require(c.sha(p) == digest, "Cache artifact changed: " + name)
    return mp.parent


def legacy(u):
    import pyarrow.parquet as pq

    old, ep, ids, gold, prepared = reader.legacy_data(u)
    config = c.read(
        c.ROOT
        / "experiments/astra"
        / CONFIGS[u["study"]]
        / "api_execution_v1/analysis_plan.json"
    )
    conditions = config["conditions"]
    settings = [(name, r) for name in conditions for r in range(3)]
    c.require(set(settings) == set(prepared), "Settings differ")
    pred = np.array([prepared[k][0] for k in settings])
    valid = np.array([prepared[k][1] for k in settings])
    gate = np.zeros_like(valid)
    nvalid = np.zeros(valid.shape, np.int8)
    for path in u["assembled"]:
        rows = list(c.lines(c.ROOT / path))
        j = settings.index((rows[0]["condition"], rows[0]["replicate"]))
        lookup = {str(r["id"]): r for r in rows}
        for i, rid in enumerate(ids):
            row = lookup[rid]
            nvalid[j, i] = row["n_valid"]
            ans = row["result"]
            if ans is not None:
                gate[j, i] = ans["hate"]
            c.require(
                (ans is not None) == bool(valid[j, i]) == (nvalid[j, i] > 0),
                "Invalid vote/answer mismatch",
            )
    raw = pq.read_table(c.bind(c.ROOT / u["gold_path"])).to_pylist()
    lookup = {str(r["id"]): r for r in raw}
    rows = [lookup[i] for i in ids]
    labels = ep["all_labels"]
    macro = ep["macro_labels"]
    c.require(labels == macro, "Unexpected label-scope difference")
    return dict(
        unit=u,
        old=old,
        ids=ids,
        gold=gold,
        pred=pred,
        valid=valid,
        gate=gate,
        nvalid=nvalid,
        settings=settings,
        conditions=conditions,
        labels=labels,
        macro_labels=macro,
        indices=list(range(len(labels))),
        sources=[str(r.get("source", "unknown")) for r in rows],
        clusters=[
            hashlib.sha256(previous.normalized(r["text"]).encode("utf8")).hexdigest()
            for r in rows
        ],
        raw_records_checked=len(ids) * len(settings),
    )


def german(u, cache, robust, raw):
    old = c.read(c.ROOT / u["source"])
    meta = c.read(c.ROOT / u["metadata_path"])
    design = c.read(c.ROOT / u["resampling_source"])
    a = h.read_arrays(c.ROOT / u["prediction_path"])
    phase = old["population"]
    ep = c.read(cache / "endpoint_setup.json")[phase]
    info = h.read_arrays(robust / phase / "metadata.npz")
    conditions = meta["conditions"]
    settings = [(x, r) for x in conditions for r in range(3)]
    n = len(meta["keys"])
    pred = a["predictions"].reshape(9, n, 3)
    valid = (pred >= 0).all(-1)
    gate = a["predicted_hate"].reshape(9, n) > 0
    nvalid = info["n_valid"].reshape(9, n)
    repairs = info["repairs"].reshape(9, n)
    ties = info["ties"].reshape(9, n)
    h.equal(
        a["gold"],
        [[label in row for label in meta["labels"]] for row in ep["gold"]],
        "German cache gold",
    )
    c.require(ep["keys"] == meta["keys"], "German cache key order")
    for j, (cond, rep) in enumerate(settings):
        for i, key in enumerate(meta["keys"]):
            row = raw[phase, cond, rep, key]
            ans = row["result"]
            c.require(bool(valid[j, i]) == (ans is not None), "German raw validity")
            h.equal(
                [nvalid[j, i], repairs[j, i], ties[j, i]],
                [row["n_valid"], row["repairs"], row["uncertain"]],
                "German raw metadata",
            )
            if ans is not None:
                h.equal(
                    pred[j, i],
                    [label in ans["legal"] for label in meta["labels"]],
                    "German raw answer",
                )
                c.require(gate[j, i] == ans["hate"], "German raw gate")
    h.equal(valid.all(0), a["common_mask"], "German common mask")
    c.require(old["labels"] == old["macro_labels"], "German label-scope difference")
    return dict(
        unit=u,
        old=old,
        ids=meta["keys"],
        gold=a["gold"].astype(bool),
        pred=pred > 0,
        valid=valid,
        gate=gate,
        nvalid=nvalid,
        repairs=repairs,
        ties=ties,
        settings=settings,
        conditions=conditions,
        labels=meta["labels"],
        macro_labels=old["macro_labels"],
        indices=list(range(3)),
        sources=meta["sources"],
        clusters=meta["clusters"],
        raw_records_checked=9 * n,
        original_arrays=a,
        design=design,
        quality=c.read(robust / "quality_subset_scores.json")[phase],
    )


def view(d, names, pair):
    ix = [j for j, k in enumerate(d["settings"]) if k[0] in names]
    v = dict(d)
    for k in ["pred", "valid", "gate", "nvalid", "repairs", "ties"]:
        if k in v:
            v[k] = v[k][ix]
    v["settings"] = [d["settings"][j] for j in ix]
    v["conditions"] = [x for x in d["conditions"] if x in names]
    v["mask"] = v["valid"].all(0)
    v["old"] = {"paired_comparison": pair}
    h.equal(
        v["mask"],
        [rid in pair["common_valid_ids"] for rid in d["ids"]],
        "Original paired view mask",
    )
    return v


def positions(d, name, repeat=None, omit=None):
    return [
        i
        for i, (cond, r) in enumerate(d["settings"])
        if cond == name
        and (repeat is None or r == repeat)
        and (omit is None or r != omit)
    ]


def weighted(d, scores, weights, repeat=None, omit=None):
    return sum(
        (
            w * scores[..., positions(d, name, repeat, omit), :].mean(-2)
            for name, w in weights.items()
        )
    )


def bounds(d, summaries, weights):
    out = {}
    for metric in h.METHODS:
        value = np.zeros(2)
        for name, w in weights.items():
            pair = np.mean(
                [
                    summaries[i]["full_roster_missing_bounds"][metric]
                    for i in positions(d, name)
                ],
                0,
            )
            value += w * (pair if w > 0 else pair[::-1])
        out[metric] = value.tolist()
    return out


def replay_german(d):
    v = d["original_arrays"]
    raw = v["bootstrap_confusions"]
    counts = raw[..., [3, 1, 2]].reshape(len(raw), 9, 3, 3)
    scores = h.values(counts, d["indices"])
    design = d["design"]
    mask = d["mask"]
    common = h.encode(d["gold"], d["pred"], d["valid"])[:, mask]
    clusters = design["clusters"]
    c.require(len(design["cluster_rows"]) == len(clusters), "Cluster-row alignment")
    grouped = np.stack([common[:, rows].sum(1) for rows in design["cluster_rows"]], 1)
    flat = grouped.transpose(1, 0, 2, 3).reshape(len(clusters), -1)
    rng = np.random.default_rng(design["seed"])
    digest = hashlib.sha256()
    checks = 0
    for start in range(0, design["draws"], design["block_size"]):
        size = min(design["block_size"], design["draws"] - start)
        picks = np.concatenate(
            [
                rng.choice(
                    s["cluster_indices"],
                    (size, len(s["cluster_indices"])),
                    replace=True,
                )
                for s in design["strata"]
            ],
            1,
        )
        digest.update(picks.astype("<i4").tobytes())
        freq = np.array([np.bincount(x, minlength=len(clusters)) for x in picks])
        total = (freq @ flat).reshape(size, 9, 3, 3)
        h.equal(total, counts[start : start + size], "Saved German bootstrap counts")
        h.equal(total[0], grouped[:, picks[0]].sum(1), "German direct expansion")
        checks += 1
    c.require(
        digest.hexdigest() == design["pick_stream_sha256"], "German pick digest differs"
    )
    return dict(
        scores=scores,
        mask=mask,
        seed=design["seed"],
        draws=design["draws"],
        n_units=len(clusters),
        mode="saved_cluster",
        checks=dict(
            saved_confusion_values=int(raw.size),
            direct_expansions=checks,
            pick_stream_sha256=digest.hexdigest(),
        ),
    )


def contrast(d, old, point, bs, own, family, qid):
    w = old["weights"]
    delta = weighted(d, point, w)
    dd = weighted(d, bs["item"]["scores"], w)
    h.equal(delta[0], old["mean_difference"], "Original macro point")
    h.equal(h.interval(dd[:, 0]), old["ci95"], "Original macro95")
    if old["family_ci"] is not None:
        h.equal(
            h.interval(dd[:, 0], old["family_confidence"]),
            old["family_ci"],
            "Original adjusted macro",
        )
    mb = bounds(d, own, w)
    h.equal(
        mb["macro"],
        old["invalid_inclusive_difference_bounds"],
        "Original missing macro bounds",
    )
    out = dict(
        weights=w,
        n_common=int(d["mask"].sum()),
        family_size=family,
        question_id=qid if family else None,
        metrics={},
        full_roster_missing_bounds=mb,
        by_repeat={},
        leave_one_repeat_out={},
        sensitivities={},
    )
    for j, m in enumerate(h.METHODS):
        out["metrics"][m] = dict(
            difference=float(delta[j]),
            ci95=h.interval(dd[:, j]),
            adjusted_interval=(
                h.interval(dd[:, j], 1 - 0.05 / family)
                if family and m != "macro"
                else old["family_ci"] if m == "macro" else None
            ),
            adjusted_confidence=(
                1 - 0.05 / family
                if family and m != "macro"
                else old["family_confidence"] if m == "macro" else None
            ),
            original_confidence_diagnostic=(
                h.interval(dd[:, j], old["family_confidence"])
                if old["family_confidence"] and m != "macro"
                else None
            ),
        )
    for r in range(3):
        out["by_repeat"][str(r)] = dict(zip(h.METHODS, weighted(d, point, w, repeat=r)))
        out["leave_one_repeat_out"][str(r)] = dict(
            zip(h.METHODS, weighted(d, point, w, omit=r))
        )
    for mode, b in bs.items():
        if mode == "item":
            continue
        pp = h.values(
            h.encode(
                d["gold"], d["pred"], np.broadcast_to(b["mask"], d["valid"].shape)
            ).sum(1),
            d["indices"],
        )
        dif = weighted(d, pp, w)
        sample = weighted(d, b["scores"], w)
        out["sensitivities"][mode] = dict(
            n_common=int(b["mask"].sum()),
            n_units=b["n_units"],
            metrics={
                m: dict(difference=float(dif[j]), ci95=h.interval(sample[:, j]))
                for j, m in enumerate(h.METHODS)
            },
        )
    return out


def analyse():
    plan = c.read(BASE / "analysis_plan.json")
    outputs = []
    totalraw = 0
    totalboot = 0
    cache = verified_run(GERMAN_READER)
    robust = verified_run(GERMAN_ROBUST)
    raw = {}
    for row in c.lines(cache / "predictions.jsonl"):
        if row["phase"] in ["de_legal_test", "de_legal_dev_eval"]:
            raw[row["phase"], row["condition"], row["repeat"], row["key"]] = row
    c.require(len(raw) == 3168, "German legal cache incomplete")
    for u in plan["units"]:
        isg = u["study"] == "german_guidelines"
        base = german(u, cache, robust, raw) if isg else legacy(u)
        totalraw += base["raw_records_checked"]
        old = base["old"]
        if isg:
            pairs = {}
            for i, x in enumerate(old["contrasts"]):
                m = x["metrics"]["macro_f1"]
                pairs["pair" + str(i + 1)] = dict(
                    weights={x["left"]: 1, x["right"]: -1},
                    mean_difference=m["difference"],
                    ci95=m["ci95"],
                    family_ci=m["family_interval"],
                    family_confidence=m["family_confidence"],
                    invalid_inclusive_difference_bounds=x[
                        "full_roster_failure_difference_bounds"
                    ]["macro_f1"],
                )
            views = {
                "paired": dict(
                    conditions={x: {} for x in base["conditions"]},
                    common_valid_ids=old["common_keys"],
                    bootstrap_seed=base["design"]["seed"],
                    bootstrap_draws=10000,
                    contrasts=pairs,
                )
            }
        else:
            views = (
                old["paired_comparison"]
                if u["study"] == "components"
                else {"paired": old["paired_comparison"]}
            )
        for name, pair in views.items():
            d = view(base, list(pair["conditions"]), pair)
            own = previous.setting_summaries(d)
            common = previous.setting_summaries(d, d["mask"])
            point = np.array([[x["scores"][m] for m in h.METHODS] for x in common])
            for j, (cond, rep) in enumerate(d["settings"]):
                if isg:
                    for mode, ss in [("own_valid", own), ("common", common)]:
                        prior = old[mode][cond]["repeats"][rep]
                        for m in ["macro", "micro"]:
                            h.equal(
                                ss[j]["scores"][m],
                                prior[m + "_f1"],
                                "Saved German " + mode + " " + m,
                            )
                        h.equal(
                            ss[j]["set_only_exact"],
                            prior["exact_set_accuracy"],
                            "German exact sets",
                        )
                else:
                    prior = next(
                        (
                            x["score"]
                            for x in old["conditions"][cond]["runs"]
                            if x["replicate"] == rep
                        )
                    )
                    h.equal(
                        own[j]["scores"]["macro"],
                        prior["macro_f1_valid_only"],
                        "Original own macro",
                    )
                    h.equal(
                        own[j]["tp_fp_fn"],
                        prior["per_label_counts"],
                        "Original own counts",
                    )
                    h.equal(own[j]["n_valid"], prior["n_valid"], "Original own n")
                    h.equal(
                        own[j]["full_roster_missing_bounds"]["macro"],
                        prior["invalid_inclusive_bounds"],
                        "Original own bounds",
                    )
                    h.equal(
                        point[j, 0],
                        pair["conditions"][cond]["common_valid_summary"]["values"][rep],
                        "Original common setting macro",
                    )
            bs = {"item": replay_german(d) if isg else previous.boot(d, d["mask"])}
            strict = d["mask"] & (d["nvalid"] == 3).all(0)
            bs["strict_three_votes"] = previous.boot(d, strict)
            if not isg and len(set(np.array(d["sources"])[d["mask"]])) > 1:
                bs["source_stratified"] = previous.boot(
                    d, d["mask"], "source_stratified"
                )
            if not isg and len(set(np.array(d["clusters"])[d["mask"]])) < int(
                d["mask"].sum()
            ):
                bs["cluster"] = previous.boot(d, d["mask"], "cluster")
            folder = BASE / u["study"] / u["endpoint"] / name
            files = {}
            for mode, b in bs.items():
                path = folder / (mode + "_bootstrap.npz")
                arrays(path, scores=b["scores"], mask=b["mask"])
                totalboot += b["scores"].size
                files[mode] = dict(
                    path=c.rel(path),
                    seed=b["seed"],
                    draws=b["draws"],
                    n_units=b["n_units"],
                    mode=b["mode"],
                    checks=b["checks"],
                )
            groups = {}
            for cond in d["conditions"]:
                ix = positions(d, cond)
                groups[cond] = dict(
                    common=h.average([common[j] for j in ix]),
                    own_valid=h.average([own[j] for j in ix]),
                    interval95={
                        m: h.interval(bs["item"]["scores"][:, ix, j].mean(1))
                        for j, m in enumerate(h.METHODS)
                    },
                )
            contrasts = {}
            for key, x in pair["contrasts"].items():
                qid = u["id"] + "|" + key if isg else u["id"] + "|" + name + "|" + key
                fam = (
                    plan["new_families"][u["study"]]["m"]
                    if qid in plan["new_families"][u["study"]]["members"]
                    else None
                )
                contrasts[key] = contrast(d, x, point, bs, own, fam, qid)
                if isg:
                    prior = old["contrasts"][int(key[-1]) - 1]
                    h.equal(
                        contrasts[key]["metrics"]["micro"]["difference"],
                        prior["metrics"]["micro_f1"]["difference"],
                        "Saved German micro point",
                    )
                    h.equal(
                        contrasts[key]["metrics"]["micro"]["ci95"],
                        prior["metrics"]["micro_f1"]["ci95"],
                        "Saved German micro95",
                    )
                    h.equal(
                        contrasts[key]["full_roster_missing_bounds"]["micro"],
                        prior["full_roster_failure_difference_bounds"]["micro_f1"],
                        "Saved German micro bounds",
                    )
            quality = {}
            if isg:
                for label, mask in [
                    ("three_valid_inner_votes", strict),
                    ("no_repairs", d["mask"] & (d["repairs"] == 0).all(0)),
                    ("no_hate_ties", d["mask"] & (~d["ties"]).all(0)),
                ]:
                    ss = previous.setting_summaries(d, mask)
                    means = {
                        cond: h.average([ss[j] for j in positions(d, cond)])
                        for cond in d["conditions"]
                    }
                    quality[label] = dict(
                        n=int(mask.sum()),
                        gold_support=d["gold"][mask].sum(0),
                        groups=means,
                        contrasts={
                            k: dict(
                                zip(
                                    h.METHODS,
                                    weighted(
                                        d,
                                        np.array(
                                            [
                                                [x["scores"][m] for m in h.METHODS]
                                                for x in ss
                                            ]
                                        ),
                                        x["weights"],
                                    ),
                                )
                            )
                            for k, x in pair["contrasts"].items()
                        },
                    )
            arr = folder / "arrays.npz"
            extra = {k: d[k] for k in ["repairs", "ties"] if k in d}
            arrays(
                arr,
                gold=d["gold"],
                pred=d["pred"],
                valid=d["valid"],
                gate=d["gate"],
                nvalid=d["nvalid"],
                ids=np.array(d["ids"]),
                settings=np.array(d["settings"]),
                labels=np.array(d["labels"]),
                macro_labels=np.array(d["macro_labels"]),
                common=d["mask"],
                clusters=np.array(d["clusters"]),
                sources=np.array(d["sources"]),
                **extra
            )
            result = dict(
                unit=u,
                view=name,
                settings=d["settings"],
                conditions=d["conditions"],
                labels=d["labels"],
                macro_labels=d["macro_labels"],
                ids=d["ids"],
                common_ids=[rid for rid, on in zip(d["ids"], d["mask"]) if on],
                n_eligible=len(d["ids"]),
                n_common=int(d["mask"].sum()),
                source_counts=dict(Counter(d["sources"])),
                unique_common_clusters=len(set(np.array(d["clusters"])[d["mask"]])),
                own=own,
                common=common,
                groups=groups,
                contrasts=contrasts,
                quality=quality,
                arrays=c.rel(arr),
                bootstrap_files=files,
                validity_histogram=dict(Counter(d["nvalid"].ravel().tolist())),
                strict_common=int(strict.sum()),
                new_calls=0,
                new_training=0,
                original_micro_reused=isg,
                original_bootstrap_arrays_available=isg,
            )
            save(folder / "results.json", result)
            outputs.append(c.rel(folder / "results.json"))
            print(u["id"], name, "n", result["n_common"], "done", flush=True)
    for path in [
        Path(__file__),
        Path(reader.__file__),
        Path(h.__file__),
        Path(c.__file__),
        Path(previous.__file__),
        c.ROOT / "analysis/common/micro_f1.py",
    ]:
        c.bind(path)
    save(
        BASE / "analysis_index.json",
        dict(
            status="Part 6 numerical analysis complete; independent verification pending",
            outputs=outputs,
            raw_eligible_answer_records=totalraw,
            stored_bootstrap_metric_values=totalboot,
            checks=dict(h.CHECKS),
            max_errors=dict(h.ERRORS),
            reader_checks=dict(reader.CHECKS),
            source_sha256=c.BINDINGS,
            new_calls=0,
            new_training=0,
            latex_edits=0,
        ),
    )


def run():
    global BASE
    root = BASE
    plan = c.read(root / "analysis_plan.json")
    c.bind(Path(__file__))
    BASE = root / "analysis_v2"
    c.require(not BASE.exists(), "Preserve earlier corrected attempt")
    save(BASE / "analysis_plan.json", plan)
    save(
        BASE / "adapter_note.json",
        dict(
            reason="Original German cluster_rows is an ordered list aligned with clusters, not a mapping. Correct only this reader representation.",
            first_attempt_preserved=True,
            original_sources_changed=False,
            other_scores_unchanged=True,
        ),
    )
    analyse()


if __name__ == "__main__":
    run()

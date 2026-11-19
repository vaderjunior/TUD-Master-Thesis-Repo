"""Policy robustness."""

import os

os.environ.update(
    PYTHONDONTWRITEBYTECODE="1",
    OPENBLAS_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    OMP_NUM_THREADS="2",
    CUDA_VISIBLE_DEVICES="",
)
from collections import Counter
from itertools import product
from pathlib import Path
import argparse
import json
import numpy as np
from analysis.knowledge.policy import common as c
from analysis.knowledge.policy import metrics as m
from analysis.knowledge.policy import reader as reader

BASE = c.ROOT / "experiments/astra/complete_policy_part5_20261115_v1"
P3 = reader.BASE
P4 = c.ROOT / "experiments/astra/complete_policy_part4_20261115_v1"
OLD = c.ROOT / "experiments/astra/sq2_broad_reader_review_20261104_v1"
OLD_ROSTER = (
    c.ROOT
    / "experiments/astra/offline_closure_20261006_v1/evaluation_preparation/broad_roster.jsonl"
)
RUN = None
ENDPOINTS = None
REGISTRY = None
NAMES = c.CONDITIONS


def load_registry():
    global REGISTRY
    if REGISTRY is None:
        REGISTRY = c.read(c.SETUP / "comparison_registry.json")
    return REGISTRY


def load_endpoints():
    global ENDPOINTS
    if ENDPOINTS is None:
        ENDPOINTS = c.read(c.SETUP / "endpoints.json")
    return ENDPOINTS


def load_run():
    global RUN
    if RUN is None:
        RUN = c.ROOT / c.read(P3 / "analysis_complete.json")["run"]
    return RUN


def save(name, value):
    path = BASE / name
    c.require(c.normal(path).is_relative_to(BASE), "Output outside new review folder")
    c.save(path, m.serial(value))


def rowsave(name, values):
    path = BASE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        for v in values:
            f.write(
                json.dumps(m.serial(v), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )


def npzsave(name, values):
    path = BASE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, **values)


def load():
    roster = {r["key"]: r for r in c.rows(c.SETUP / "roster.jsonl")}
    arrays = {
        e: dict(np.load(load_run() / e / "encoded.npz", allow_pickle=False))
        for e in load_endpoints()
    }
    rows = [
        r
        for path in c.read(P3 / "reader_complete.json")["reader_files"]
        for r in c.read(c.ROOT / path)
    ]
    return (roster, arrays, rows)


def prepare():
    if (BASE / "analysis_contract.json").exists():
        c.verify(c.read(BASE / "analysis_contract.json")["inputs"])
        return
    c.require(c.sha(c.PLAN) == reader.PLAN_SHA, "Wrong frozen experiment")
    c.verify(c.read(P3 / "reader_complete.json")["reader_files"])
    c.verify(c.read(load_run() / "complete.json")["artifacts"])
    assembly = c.read(OLD / "assembly.json")
    c.require(
        assembly["status"] == "verified_D_reader_cache_complete", "Unverified historical reader"
    )
    c.verify(assembly["outputs"])
    paths = [
        c.PLAN,
        c.SETUP / "endpoints.json",
        c.SETUP / "roster.jsonl",
        c.SETUP / "sensitivity_rosters.json",
        c.SETUP / "comparison_registry.json",
        c.SETUP / "supporting_length_groups.json",
        P3 / "part3_complete.json",
        P3 / "reader_complete.json",
        P3 / "analysis_complete.json",
        load_run() / "complete.json",
        P4 / "part4_complete.json",
        P4 / "context_followup_checks.json",
        P4 / "context_features.jsonl",
        P4 / "groups_and_sources.json",
        OLD / "assembly.json",
        OLD / "predictions.jsonl",
        OLD / "item_bindings.json",
        OLD_ROSTER,
        c.ROOT / "Thesis_Log/astra_result/complete_policy_part1_protocol_20261114_v1.md",
        c.ROOT / "Thesis_Log/astra_result/complete_policy_work_plan_20261114_v2.md",
        Path(m.__file__),
        Path(reader.__file__),
    ]
    bound = c.bindings(paths)
    bound.update(c.read(P3 / "reader_complete.json")["reader_files"])
    bound.update(c.read(load_run() / "complete.json")["artifacts"])
    save(
        "analysis_contract.json",
        dict(
            created_utc=c.a.utc(),
            status="scope_recorded_before_part5_scores",
            inputs=bound,
            prior_exposure="Parts 3 and 4 are known. This is the frozen robustness work plus explicitly labelled explanatory diagnostics, not a new preregistration.",
            planned_masks=[
                "original_common_nine",
                "all_27_inner_valid",
                "no_repairs_in_any_of_nine",
                "fixed_ANN_exclusions",
            ],
            no_repair_definition="All nine final answers valid and zero repairs in every prediction. Check whether this implies all 27 initial votes valid; do not silently impose that additional filter.",
            additional_diagnostics=[
                "Exclude the 97 Part 4 GAHD inputs with a directly linked partner shown, equally in all settings. Exploratory, not an exhaustive semantic-overlap screen.",
                "Per-repeat, leave-one-repeat-out and all-nine cross-repeat directions; source, frozen length and label-support summaries.",
                "Leave one legal text out for sensitivity to an individual observation; descriptive, not 177 hypothesis tests.",
                "Equal-source means of source-specific macro scores and leave-one-source-out scores; different descriptive estimands, not replacement pooled macro.",
            ],
            uncertainty=dict(
                draws=50000,
                seed=20261114,
                method=load_registry()["method"],
                families=load_registry()["families"],
                fixed_labels=True,
                source_membership="From the full frozen endpoint before removal.",
                reuse="Identical masks reuse original saved draws. No seed or confidence-level changes after results.",
                limits="Families reused within each sensitivity are not an overall correction across analyses. Three repeats are fixed; no inference over all future generations.",
            ),
            missing="Own-valid per repeat; full-roster hypothetical best/worst fills. Binary labels remain one-hot, multilabel worst fill is the complement, severity worst macro enumerates all wrong-band allocations. Bounds are not imputations or confidence intervals.",
            historical="Only rag_full predictions on exact matching key/query/gold/initial prompt. Common six old/new A outputs. Descriptive scores and disagreements; no treatment or server-change attribution. Verify deterministic raw samples and saved request contracts.",
            preservation="Read-only original results, D and both thesis copies; new derived artifacts only. No model, embedding, training, server or Git action.",
            thesis_before=reader.snapshot_thesis(),
        ),
    )
    with (BASE / "session_before.md").open("xb") as f:
        f.write((c.ROOT / "Thesis_Log/astra_session.md").read_bytes())
    print("Part 5 scope and preservation snapshot saved.", flush=True)


def mask_set(endpoint, ar, linked):
    keys = load_endpoints()[endpoint]["keys"]
    common = ar["common"]
    ann = set(c.read(c.SETUP / "sensitivity_rosters.json")["ann_replay_exclusion_keys"])
    masks = {
        "original_common_nine": common,
        "all_27_inner_valid": common & (ar["n_valid"] == 3).all(axis=(0, 1)),
        "no_repairs_in_any_of_nine": common & (ar["repairs"] == 0).all(axis=(0, 1)),
        "fixed_ANN_exclusions": common & np.array([k not in ann for k in keys]),
    }
    if endpoint == "de_hate":
        masks["exploratory_no_direct_GAHD_partner"] = common & np.array(
            [k not in linked for k in keys]
        )
    return masks


def composition(meta, ar, mask, roster):
    keys = meta["keys"]
    selected = [k for k, on in zip(keys, mask) if on]
    return dict(
        n=len(selected),
        sources=dict(Counter((roster[k]["source"] for k in selected))),
        gold_support=dict(zip(meta["all_labels"], ar["gold"][mask].sum(axis=0).tolist())),
        gate_counts=dict(Counter((str(roster[k]["gate"]) for k in selected))),
        empty_gold=int((~ar["gold"][mask].any(axis=1)).sum()),
        keys=selected,
        mask_sha256=c.signature(selected),
    )


def contrasts(endpoint, point, boot, mask_name):
    values = {
        "macro_f1": point["macro_f1"],
        "mean_pairwise_repeat_agreement": point["repeatability"]["pairwise_agreement"],
    }
    if "micro_f1" in point:
        values["micro_f1_all_labels"] = point["micro_f1"]
    out = []
    for row in load_registry()["rows"]:
        if row["endpoint"] != endpoint:
            continue
        metric = row["metric"]
        a = NAMES.index(row["left"])
        b = NAMES.index(row["right"])
        confidence = (
            load_registry()["families"][row["family"]]["confidence"] if row["family"] else 0.95
        )
        samples = boot[metric][:, a] - boot[metric][:, b]
        ci = m.interval(samples, confidence)
        out.append(
            dict(
                row,
                mask=mask_name,
                n=point["n"],
                left_value=values[metric][a],
                right_value=values[metric][b],
                estimate=values[metric][a] - values[metric][b],
                family_confidence=confidence,
                family_interval=ci,
                interval95=m.interval(samples, 0.95),
                direction="positive" if ci[0] > 0 else "negative" if ci[1] < 0 else "unresolved",
            )
        )
    return out


def sensitivity():
    roster, arrays, _ = load()
    linked = {
        r["key"]
        for r in c.read(P4 / "context_followup_checks.json")["gahd_direct_contrastive_links"][
            "cases"
        ]
    }
    c.require(len(linked) == 97, "GAHD link roster changed")
    index = []
    all_contrasts = []
    for endpoint, ar in arrays.items():
        meta = load_endpoints()[endpoint]
        seen = {}
        for name, mask in mask_set(endpoint, ar, linked).items():
            folder = f"sensitivities/{endpoint}/{name}"
            if (BASE / folder / "results.json").exists():
                prior = c.read(BASE / folder / "results.json")
                index.append(
                    dict(endpoint=endpoint, mask=name, path=c.ref(BASE / folder / "results.json"))
                )
                all_contrasts.extend(prior["contrasts"])
                seen[mask.tobytes()] = folder
                continue
            point = m.point_metrics(meta, ar, mask)
            comp = composition(meta, ar, mask, roster)
            signature = mask.tobytes()
            same = seen.get(signature)
            if np.array_equal(mask, ar["common"]):
                path = load_run() / endpoint / "bootstrap.npz"
                boot = dict(np.load(path, allow_pickle=False))
                reused = c.ref(path)
                bootstrap_source = path
            elif same:
                prev = c.read(BASE / same / "results.json")
                bootstrap_source = c.ROOT / prev["bootstrap_source"]
                boot = dict(np.load(bootstrap_source, allow_pickle=False))
                reused = c.ref(bootstrap_source)
            else:
                sub = dict(ar, common=mask)
                features, strata, _ = m.cluster_features(meta, sub, roster)
                print(
                    f"{endpoint} / {name}: {int(mask.sum())} inputs, 50,000 paired draws",
                    flush=True,
                )
                boot = m.bootstrap_from_features(
                    features,
                    strata,
                    len(meta["all_labels"]),
                    50000,
                    20261114,
                    progress=lambda n: print(f"  {endpoint} {name}: {n}/50000", flush=True),
                )
                bootstrap_source = BASE / folder / "bootstrap.npz"
                npzsave(f"{folder}/bootstrap.npz", boot)
                reused = None
            rows = contrasts(endpoint, point, boot, name)
            result = dict(
                endpoint=endpoint,
                mask=name,
                point=point,
                composition=comp,
                removed_from_main=composition(meta, ar, ar["common"] & ~mask, roster),
                contrasts=rows,
                bootstrap_source=c.ref(bootstrap_source),
                bootstrap_sha256=c.sha(bootstrap_source),
                reused=reused,
                labels_below_ten=[label for label, n in comp["gold_support"].items() if n < 10],
                labels_zero=[label for label, n in comp["gold_support"].items() if n == 0],
            )
            save(f"{folder}/results.json", result)
            seen[signature] = folder
            all_contrasts.extend(rows)
            index.append(
                dict(endpoint=endpoint, mask=name, path=c.ref(BASE / folder / "results.json"))
            )
    save("sensitivity_index.json", index)
    save("sensitivity_contrasts.json", all_contrasts)


def score_single(y, p, g, valid, field):
    c.require(valid.any(), "Empty own-valid score")
    yy, pp = (y[valid], p[valid])
    counts = np.stack((pp & yy, pp & ~yy, ~pp & yy), axis=-1).sum(axis=0)
    macro, micro, perlabel = m.f1(counts)
    exact = (pp == yy).all(axis=1)
    if field not in ("hate", "legal"):
        exact &= g[valid]
    out = dict(
        n=int(valid.sum()),
        counts=counts,
        macro_f1=float(macro),
        exact=float(exact.mean()),
        per_label_f1=perlabel,
    )
    if field in ("hate_type", "target_group", "legal"):
        out["micro_f1"] = float(micro)
    return out


def full_bounds(y, p, g, valid, field):
    yy = y[valid]
    pp = p[valid]
    counts = np.stack((pp & yy, pp & ~yy, ~pp & yy), axis=-1).sum(axis=0)
    missing = y[~valid]
    support = missing.sum(axis=0)
    n = len(missing)
    best = counts.copy()
    best[:, 0] += support
    best_macro, best_micro, _ = m.f1(best)
    if field == "severity":
        c.require(np.all(missing.sum(axis=1) == 1), "Severity gold must be one-hot")
        worst_macro = np.inf
        worst_counts = None
        worst_allocation = None
        for split in product(*(range(int(v) + 1) for v in support)):
            trial = counts.copy()
            trial[:, 2] += support
            allocation = np.zeros((3, 3), int)
            for gold in range(3):
                others = [j for j in range(3) if j != gold]
                allocation[gold, others[0]] = split[gold]
                allocation[gold, others[1]] = support[gold] - split[gold]
            trial[:, 1] += allocation.sum(axis=0)
            value = m.f1(trial)[0]
            if value < worst_macro:
                worst_macro = float(value)
                worst_counts = trial
                worst_allocation = allocation
        worst_micro = None
    else:
        trial = counts.copy()
        trial[:, 1] += n - support
        trial[:, 2] += support
        worst_macro, worst_micro, _ = m.f1(trial)
        worst_counts = trial
        worst_allocation = None
    exact = (pp == yy).all(axis=1)
    if field not in ("hate", "legal"):
        exact &= g[valid]
    out = dict(
        missing=n,
        macro_f1=[float(worst_macro), float(best_macro)],
        exact=[float(exact.sum() / len(y)), float((exact.sum() + n) / len(y))],
        worst_counts=worst_counts,
        best_counts=best,
        missing_gold_support=support,
        worst_severity_allocation=worst_allocation,
    )
    if field in ("hate_type", "target_group", "legal"):
        out["micro_f1"] = [float(worst_micro), float(best_micro)]
    return out


def diagnostics():
    roster, arrays, rows = load()
    output = {}
    quality = {}
    repeats = {}
    subgroups = {}
    lengths = c.read(c.SETUP / "supporting_length_groups.json")
    for endpoint, ar in arrays.items():
        meta = load_endpoints()[endpoint]
        field = meta["dimension"]
        records = {}
        for ci, cond in enumerate(NAMES):
            own = []
            bounds = []
            for r in range(3):
                valid = ar["valid"][ci, r]
                own.append(
                    score_single(
                        ar["gold"], ar["predicted"][ci, r], ar["gates"][ci, r], valid, field
                    )
                )
                bounds.append(
                    full_bounds(
                        ar["gold"], ar["predicted"][ci, r], ar["gates"][ci, r], valid, field
                    )
                )
            metrics = ["macro_f1", "exact"] + (
                ["micro_f1"] if field in ("hate_type", "target_group", "legal") else []
            )
            records[cond] = dict(
                own_valid_repeats=own,
                own_valid_mean={k: np.mean([v[k] for v in own]) for k in metrics},
                full_roster_bounds_repeats=bounds,
                full_roster_mean_bounds={
                    k: np.mean([v[k] for v in bounds], axis=0) for k in metrics
                },
            )
        bound_contrasts = []
        for a, b in m.CONTRASTS:
            for metric in metrics:
                aa = records[NAMES[a]]["full_roster_mean_bounds"][metric]
                bb = records[NAMES[b]]["full_roster_mean_bounds"][metric]
                bound_contrasts.append(
                    dict(
                        left=NAMES[a],
                        right=NAMES[b],
                        metric=metric,
                        bounds=[aa[0] - bb[1], aa[1] - bb[0]],
                    )
                )
        output[endpoint] = dict(
            full_n=len(meta["keys"]),
            common_n=int(ar["common"].sum()),
            conditions=records,
            contrasts=bound_contrasts,
            excluded=composition(meta, ar, ~ar["common"], roster),
        )
        quality[endpoint] = {
            cond: dict(
                inner_valid_counts=dict(Counter((str(int(x)) for x in ar["n_valid"][ci].ravel()))),
                repaired_predictions=int((ar["repairs"][ci] > 0).sum()),
                repair_calls=int(ar["repairs"][ci].sum()),
            )
            for ci, cond in enumerate(NAMES)
        }
        p = m.point_metrics(meta, ar)
        repeat_rows = []
        for a, b in m.CONTRASTS:
            for metric in ["macro_f1", "exact"] + (["micro_f1"] if "micro_f1" in p else []):
                v = p[metric + "_repeats"]
                delta = v[a] - v[b]
                repeat_rows.append(
                    dict(
                        left=NAMES[a],
                        right=NAMES[b],
                        metric=metric,
                        differences=delta,
                        leave_one_repeat_out=[np.delete(delta, i).mean() for i in range(3)],
                        cross_repeat_differences=v[a, :, None] - v[b, None, :],
                        mean=delta.mean(),
                    )
                )
        repeats[endpoint] = repeat_rows
        groups = {}
        for src in sorted({roster[k]["source"] for k in meta["keys"]}):
            groups["source/" + src] = np.array([roster[k]["source"] == src for k in meta["keys"]])
        for band, keys in lengths[endpoint]["keys"].items():
            s = set(keys)
            groups["A_length_quartile/" + band] = np.array([k in s for k in meta["keys"]])
        for src in sorted({roster[k]["source"] for k in meta["keys"]}):
            if len({roster[k]["source"] for k in meta["keys"]}) > 1:
                groups["leave_source_out/" + src] = np.array(
                    [roster[k]["source"] != src for k in meta["keys"]]
                )
        scored = {}
        for name, group in groups.items():
            mask = ar["common"] & group
            if mask.any():
                pt = m.point_metrics(meta, ar, mask)
                scored[name] = dict(composition=composition(meta, ar, mask, roster), point=pt)
        source_scores = [
            r["point"]["macro_f1"] for k, r in scored.items() if k.startswith("source/")
        ]
        subgroups[endpoint] = dict(
            groups=scored,
            equal_source_mean_macro=np.mean(source_scores, axis=0),
            limits="Descriptive fixed-label scores; source macro is a different estimand. Tiny/zero-support groups are not new primary tests.",
        )
    ar = arrays["de_legal"]
    mask = ar["common"]
    records = []
    for i, key in enumerate(load_endpoints()["de_legal"]["keys"]):
        if not mask[i]:
            continue
        sub = mask.copy()
        sub[i] = False
        p = m.point_metrics(load_endpoints()["de_legal"], ar, sub)
        records.append(
            dict(
                key=key,
                macro_differences={
                    NAMES[a] + "_minus_" + NAMES[b]: p["macro_f1"][a] - p["macro_f1"][b]
                    for a, b in m.CONTRASTS
                },
            )
        )
    save("missing_output_scores_bounds.json", output)
    save("quality_metadata.json", quality)
    save("repeat_diagnostics.json", repeats)
    save("subgroup_diagnostics.json", subgroups)
    save("legal_leave_one_out.json", records)
    totals = {
        cond: dict(
            Counter(
                predictions=sum((r["condition"] == cond for r in rows)),
                repaired_predictions=sum(
                    (r["condition"] == cond and r["repairs"] > 0 for r in rows)
                ),
                repair_calls=sum((r["repairs"] for r in rows if r["condition"] == cond)),
                missing=sum((r["condition"] == cond and r["result"] is None for r in rows)),
            )
        )
        for cond in NAMES
    }
    save("process_counts.json", totals)
    print("Missing bounds, repeat directions, fixed groups and legal influence saved.", flush=True)


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument("mode", choices=["prepare", "sensitivity", "diagnostics"])
    args = arg.parse_args()
    with c.offline():
        globals()[args.mode]()


if __name__ == "__main__":
    main()

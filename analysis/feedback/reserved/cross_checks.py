import argparse
from collections import Counter
from fractions import Fraction as F
import json
from pathlib import Path
import platform
import re
import time
import numpy as np
from analysis.feedback.reserved import review as io
from analysis.feedback.reserved import cross_checks_metrics as c

ROOT = io.ROOT
BASE = ROOT / "experiments/astra"
STUDY = BASE / "sq3_reserved_slot_20261012_v1"
ASSEMBLY = BASE / "sq3_reserved_audited_assembly_20261019_v1"
OUT = BASE / "sq3_reserved_part9_results_20261020_v1"
PREP = BASE / "sq3_reserved_part9_preparation_20261020_v1"
PARTS = {
    2: BASE / "sq3_reserved_part2_results_20261019_v1",
    3: BASE / "sq3_reserved_part3_results_20261019_v2",
    4: BASE / "sq3_reserved_part4_results_20261020_v1",
    5: BASE / "sq3_reserved_part5_results_20261020_v1",
    6: BASE / "sq3_reserved_part6_results_20261020_v2",
    7: BASE / "sq3_reserved_part7_results_20261020_v1",
    8: BASE / "sq3_reserved_part8_results_20261020_v2",
}
ASSEMBLY_SHA = "f6bbca18ca59dcfcf7a639399acd8367bcdbcc27cdb424143afe0254abaf7d03"
FIELDS = ("artifacts", "bindings", "input_bindings", "report_binding")
need = c.need


def write(path, value):
    need(
        path.resolve().is_relative_to(OUT.resolve())
        or path.resolve().is_relative_to(PREP.resolve()),
        "Write outside Part 9",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(
            value
            if isinstance(value, str)
            else json.dumps(
                value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        )


def close(actual, expected, context):
    if isinstance(actual, dict):
        need(isinstance(expected, dict), context + " expected dictionary")
        for k, v in actual.items():
            need(k in expected, context + " missing " + k)
            close(v, expected[k], context + "/" + k)
    elif isinstance(actual, (list, tuple)):
        need(
            isinstance(expected, (list, tuple)) and len(actual) == len(expected),
            context + " length",
        )
        for i, (a, b) in enumerate(zip(actual, expected)):
            close(a, b, context + f"/{i}")
    elif actual is None:
        need(expected is None, context + " undefined mismatch")
    elif isinstance(actual, (int, float, F, np.integer)):
        need(
            expected is not None and abs(float(actual) - float(expected)) < 1e-12,
            context + f" numeric mismatch: {actual} != {expected}",
        )
    else:
        need(actual == expected, context + " value mismatch")


def exact(actual, expected, context):
    if actual is None:
        need(expected is None, context + " null mismatch")
    else:
        need(actual == F(expected), context + " exact fraction mismatch")


def bindings_phase():
    need(
        io.sha(ASSEMBLY / "complete.json") == ASSEMBLY_SHA, "Original assembly changed"
    )
    manifests = [ASSEMBLY / "complete.json"]
    for part, folder in PARTS.items():
        names = [
            "complete_v2.json" if part == 5 else "complete.json",
            (
                "interpretation_check_v2.json"
                if part == 5
                else "interpretation_check.json"
            ),
            "statistics_complete.json",
            "analysis_complete.json",
            "started.json",
            "context_complete.json",
            "cases_complete.json",
            "tables_figures_complete.json",
        ]
        manifests.extend(folder / n for n in names if (folder / n).exists())
    bound, owners = {}, {}
    manifest_rows = []
    for path in manifests:
        obj = io.read(path)
        manifest_rows.append(
            {
                "path": io.rel(path),
                "sha256": io.sha(path),
                "status": obj.get("status"),
                "historical_checkpoint_only": path.name
                in (
                    "started.json",
                    "statistics_complete.json",
                    "analysis_complete.json",
                ),
            }
        )
        for field in FIELDS:
            for name, sha in obj.get(field, {}).items():
                need(
                    isinstance(sha, str) and re.fullmatch("[a-f0-9]{64}", sha),
                    "Malformed binding " + name,
                )
                need(
                    name not in bound or bound[name] == sha,
                    "Conflicting current binding " + name,
                )
                bound[name] = sha
                owners.setdefault(name, []).append(io.rel(path) + ":" + field)
    for n, (name, sha) in enumerate(sorted(bound.items()), 1):
        need(io.sha(ROOT / name) == sha, "Input hash mismatch: " + name)
        if n % 400 == 0:
            print(f"Bound scientific artifacts checked: {n}/{len(bound)}", flush=True)
    latest = {
        str(p): {
            "root": io.rel(folder),
            "review": io.rel(
                folder
                / (
                    "interpretation_check_v2.json"
                    if p == 5
                    else "interpretation_check.json"
                )
            ),
        }
        for p, folder in PARTS.items()
    }
    write(
        OUT / "evidence_bindings.json",
        dict(
            status="current_scientific_bindings_verified",
            created_utc=io.utc(),
            unique_bindings=len(bound),
            binding_references=sum(map(len, owners.values())),
            bindings=bound,
            owners=owners,
            manifest_inventory=manifest_rows,
            current_parts=latest,
            raw_request_replay=False,
            new_model_calls=0,
            current_version_rules=[
                "Part 2 complete.json is a historical pre-interpretation checkpoint; interpretation_check.json closes the part.",
                "Part 2 primary table/figures v3 preserve an exactly zero interval endpoint.",
                "Part 3 results v2; v1 partial verification remains preserved.",
                "Part 5 complete_v2.json / interpretation_check_v2.json / figures v2 are current.",
                "Part 6 results v2 retain query-specific metadata variants.",
                "Part 8 results v2 with local_prompt_costs_v2.json and independent_resource_check_v2.json are current.",
                "Historical remaining_parts fields are not the current roadmap.",
            ],
        ),
    )
    print(f"Evidence bindings passed: {len(bound)} unique artifacts.", flush=True)


def load_data():
    need(io.sha(ASSEMBLY / "complete.json") == ASSEMBLY_SHA, "Assembly changed")
    plan = io.read(STUDY / "scientific_plan_v1.json")
    roster = list(io.rows(STUDY / "roster.jsonl"))
    files = sorted((ASSEMBLY / "assembly").glob("reserved_slot_*.jsonl"))
    need(len(files) == 16, "Execution assembly batch count")

    def records():
        for p in files:
            yield from io.rows(p)

    data = c.Data(roster, plan["conditions"], records())
    need(len(roster) == 675 and data.valid.size == 149850, "Prediction matrix size")
    need(data.batch == io.read(ASSEMBLY / "batches.json"), "Batch mapping changed")
    need(len({r["cluster"] for r in roster}) == 674, "Cluster count")
    need(
        Counter(map(int, data.valid.flat)) == {0: 84, 1: 372, 2: 1305, 3: 148089},
        "Vote coverage",
    )
    original = io.read(ASSEMBLY / "complete.json")
    for group in ("accounting", "observed_usage"):
        close(
            {k: int(v) for k, v in original[group].items()},
            {
                k.split("/", 1)[1]: v
                for k, v in data.resource.items()
                if k.startswith(group + "/")
            },
            group,
        )
    specs = c.specs(plan)
    profiles = io.read(PARTS[7] / "rate_profiles.json")
    need(set(profiles) == set(specs), "Contrast inventory changed")
    for key, spec in specs.items():
        close(spec, profiles[key]["specification"], key)
    print(
        "Fresh matrix: all 149,850 records, 74 conditions and 130 rate specifications reconciled.",
        flush=True,
    )
    return data, plan, specs, profiles


def exclusions(plan, cohort):
    return [] if cohort == "full" else plan["mandatory_excluded_keys"]


def rate_phase(data, plan, specs, profiles):
    checked, draw_checks, origins = {}, {}, {}
    max_error = 0.0
    for n, (sid, profile) in enumerate(profiles.items(), 1):
        spec = specs[sid]
        for view, saved in profile["views"].items():
            cohort, mask = view.split("/")
            point = data.rate(spec, excluded=exclusions(plan, cohort), mask=mask)
            key = view + "/" + sid
            for field in (
                "eligible_n",
                "common_n",
                "outputs_per_input",
                "real_error_counts",
                "missing_real_outputs",
            ):
                close(point[field], saved[field], key + "/" + field)
            need(
                c.digest_ids(point["common_ids"]) == saved["common_ids_sha256"],
                key + " common input hash",
            )
            exact(point["difference"], saved["difference_exact"], key)
            for a, b in zip(point["rates"], saved["exact_rates"]):
                exact(a, b, key + " arm")
            for a, b in zip(point["bounds"], saved["missing_bounds_exact"]):
                exact(a, b, key + " bound")
            close(point["difference"], saved["difference"], key + " floating point")
            close(
                point["bounds"],
                saved["missing_difference_bounds"],
                key + " float bounds",
            )
            op = saved["origin"]["path"]
            if op not in origins:
                origins[op] = io.read(ROOT / op)
            original = origins[op][saved["origin"]["key"]]
            need(
                original["common_ids"] == point["common_ids"],
                key + " original common ids",
            )
            close(
                point["eligible_n"], original["original_n"], key + " original eligible"
            )
            close(
                point["difference"], original.get("difference"), key + " original point"
            )
            checked[key] = {
                k: point[k]
                for k in (
                    "eligible_n",
                    "common_n",
                    "outputs_per_input",
                    "real_error_counts",
                    "missing_real_outputs",
                )
            }
            checked[key]["difference_exact"] = str(point["difference"])
            if saved["exact_interval"] is not None:
                draw_path = (
                    (ROOT / op).parent
                    / "draws"
                    / (saved["origin"]["key"].replace("/", "__") + ".npy")
                )
                saved_draws = np.load(draw_path, allow_pickle=False)
                recomputed, interval, details = c.rate_draws(data, spec, point)
                need(
                    saved_draws.shape == recomputed.shape == (10000,),
                    key + " draw shape",
                )
                err = float(np.max(np.abs(saved_draws - recomputed)))
                need(err < 1e-12, key + " independently reconstructed draw mismatch")
                max_error = max(max_error, err)
                close(interval, saved["exact_interval"], key + " exact interval")
                close(interval, original["family_interval"], key + " original interval")
                close(
                    spec["confidence"], saved["confidence"], key + " family confidence"
                )
                draw_checks[key] = dict(
                    draws=10000,
                    max_absolute_error=err,
                    exact_interval=[str(x) for x in interval],
                    **details,
                    draw_path=io.rel(draw_path),
                    draw_sha256=io.sha(draw_path),
                )
        if n % 15 == 0:
            print(
                f"Rate specifications / masks / fresh bootstrap checked: {n}/130",
                flush=True,
            )
    omissions = io.read(PARTS[7] / "omission_points_checked.json")
    omission_count = 0
    for key, groups in omissions.items():
        cohort, mask, sid = key.split("/", 2)
        for kind, group in groups.items():
            for obj in group["points"]:
                spec = specs[sid]
                excluded = exclusions(plan, cohort)
                if kind == "pool":
                    spec = c.omit(spec, obj["omitted_pool_seed"])
                else:
                    excluded = list(
                        set(excluded)
                        | {
                            k
                            for k, b in data.batch.items()
                            if b == obj["omitted_batch"]
                        }
                    )
                point = data.rate(spec, excluded=excluded, mask=mask)
                close(point["common_n"], obj["common_n"], key + " omission n")
                exact(
                    point["difference"],
                    obj["difference_exact"],
                    key + " omission exact",
                )
                close(point["difference"], obj["difference"], key + " omission point")
                omission_count += 1
    need(
        len(checked) == 780 and len(draw_checks) == 292 and omission_count == 5536,
        "Rate coverage",
    )
    write(
        OUT / "independent_rates.json",
        dict(
            status="all_rate_masks_points_bounds_omissions_and_overall_draws_reconstructed",
            created_utc=io.utc(),
            views=checked,
            draw_checks=draw_checks,
            rate_views=len(checked),
            omission_packages=len(omissions),
            omission_points=omission_count,
            draw_values_reconstructed=2920000,
            max_absolute_error=max_error,
            bindings={
                io.rel(PARTS[7] / f): io.sha(PARTS[7] / f)
                for f in ("rate_profiles.json", "omission_points_checked.json")
            },
            new_model_calls=0,
        ),
    )
    print(
        "All 780 rate views, 5,536 omission points and 2.92 million fresh bootstrap values passed.",
        flush=True,
    )


def source_phase(data, plan, specs):
    sources = io.read(PARTS[7] / "source_effects.json")
    proofs = io.read(PARTS[7] / "source_integer_checks.json")
    checked = {}
    nvalues = 0
    empty = 0
    for key, obj in sources.items():
        cohort, rest = key.split("/", 1)
        strat = rest.endswith("/stratified")
        sid, source = (
            (rest[: -len("/stratified")], None) if strat else rest.rsplit("/source/", 1)
        )
        p = data.rate(specs[sid], excluded=exclusions(plan, cohort), source=source)
        close(p["common_n"], obj["common_n"], key + " common")
        close(p["eligible_n"], obj["eligible_n"], key + " eligible")
        exact(p["difference"], obj["difference_exact"], key + " point")
        close(p["difference"], obj["result"].get("difference"), key + " original point")
        need(obj["confidence"] == 0.95, key + " source interval not supporting")
        path = ROOT / obj["draw_path"]
        need(io.sha(path) == obj["draw_sha256"], key + " array hash")
        draws = np.load(path, allow_pickle=False)
        proof = proofs[key]
        need(proof["draws_checked"] == len(draws), key + " independent draw coverage")
        if len(draws):
            need(
                len(draws) == 10000 and np.isfinite(draws).all(), key + " invalid draws"
            )
            rng = c.seed(
                ("source_stratified/" + cohort + "/" + sid)
                if strat
                else ("source_specific/" + cohort + "/" + sid + "/" + source)
            )
            saved_seed = (
                obj["result"]["seed"] if strat else obj["result"]["bootstrap"]["seed"]
            )
            need(rng == saved_seed, key + " source seed")
            close(
                np.quantile(draws, [0.025, 0.975]).tolist(),
                obj["exact_interval"],
                key + " source tails",
            )
            close(proof["exact_interval"], obj["exact_interval"], key + " exact proof")
            need(
                proof["max_absolute_error"] < 1e-12,
                key + " independent proof precision",
            )
            nvalues += len(draws)
        else:
            need(
                p["common_n"] == 0 and obj["exact_interval"] is None,
                key + " not-estimable handling",
            )
            empty += 1
        checked[key] = dict(
            eligible_n=p["eligible_n"],
            common_n=p["common_n"],
            difference_exact=(
                str(p["difference"]) if p["difference"] is not None else None
            ),
            draw_values_read_back=len(draws),
            prior_independent_reconstruction_reused=True,
        )
    need(
        len(checked) == 1040 and nvalues == 7800000 and empty == 260, "Source coverage"
    )
    write(
        OUT / "independent_sources.json",
        dict(
            status="all_source_masks_and_points_recomputed_prior_exact_draw_proofs_bound_and_read_back",
            created_utc=io.utc(),
            objects=checked,
            source_objects=1040,
            estimable=780,
            not_estimable=260,
            draw_values_read_back=nvalues,
            draw_values_freshly_regenerated=0,
            bindings={
                io.rel(PARTS[7] / f): io.sha(PARTS[7] / f)
                for f in ("source_effects.json", "source_integer_checks.json")
            },
            new_model_calls=0,
        ),
    )
    print(
        "All 1,040 source objects passed; 7.8 million previously independently verified draws read back.",
        flush=True,
    )


def quality_check(actual, saved, label):
    close(actual["common_n"], saved["common_n"], label + " common n")
    close(
        actual["eligible_n"],
        saved.get("eligible_n", saved.get("original_n")),
        label + " eligible n",
    )
    need(actual["common_ids"] == saved["common_ids"], label + " common ids")
    for metric, vals in actual["metrics"].items():
        close(vals, saved["metrics"][metric], label + "/" + metric)
        if "exact" in saved["metrics"][metric]:
            for k, v in vals.items():
                exact(
                    v,
                    saved["metrics"][metric]["exact"][k],
                    label + "/" + metric + "/" + k,
                )


def quality_phase(data, plan, specs):
    profiles = io.read(PARTS[7] / "quality_profiles.json")
    old = io.read(PARTS[5] / "started.json")["metric_specifications"]
    need({x["name"] for x in old} == set(profiles), "Quality specification coverage")
    expected = set()
    for cohort in c.COHORTS:
        for sid, s in specs.items():
            if s["endpoint"] != "FPR" or s["family"] not in ("supporting", "mechanism"):
                continue
            stem = sid.rsplit("/", 1)[0]
            for fine in (False, True):
                expected.add(
                    cohort
                    + "/"
                    + ("supporting_metrics/" if s["family"] == "mechanism" else "")
                    + stem
                    + ("/types" if fine else "/binary")
                )
    need(
        expected == set(profiles) and len(expected) == 112,
        "Independent quality inventory",
    )
    results = {}
    bound_count = 0
    for name, obj in profiles.items():
        definition = obj["specification"]
        spec = definition["spec"]
        fine = definition["fine"]
        cohort = definition["cohort"]
        original = specs[spec["id"]]
        close(
            {
                k: original[k]
                for k in ("treatment_conditions", "control_conditions", "draws")
            },
            spec,
            name + " arms",
        )
        need(spec["seed"] == c.seed(name), name + " quality seed")
        need(definition["excluded"] == exclusions(plan, cohort), name + " exclusions")
        points = {}
        for mask in c.MASKS:
            point = data.quality(
                spec, fine, excluded=exclusions(plan, cohort), mask=mask
            )
            saved = (
                obj["ordinary"]
                if mask == "ordinary_valid"
                else obj["point_sensitivities"][mask]
            )
            quality_check(point, saved, name + "/" + mask)
            if mask == "ordinary_valid":
                quality_check(point, obj["ordinary_point_check"], name + "/part7_point")
            points[mask] = {
                "common_n": point["common_n"],
                "eligible_n": point["eligible_n"],
                "metrics_exact": {
                    m: {k: str(v) if v is not None else None for k, v in vals.items()}
                    for m, vals in point["metrics"].items()
                },
            }
            if mask != "one_per_cluster":
                bounds = data.quality_bounds(
                    spec, fine, excluded=exclusions(plan, cohort), mask=mask
                )
                close(
                    bounds, obj["missing_bounds"][mask], name + "/" + mask + "/bounds"
                )
                proof = obj["missing_bounds"][mask]["independent_exact"]
                for metric, vals in bounds["metrics"].items():
                    for k, vs in vals.items():
                        for a, b in zip(vs, proof["exact_fractions"][metric][k]):
                            exact(a, b, name + "/bound fraction")
                bound_count += 1
        results[name] = points
    diagnostics = io.read(PARTS[7] / "quality_omissions_and_sources.json")
    diagnostic_count = 0
    for n, (name, obj) in enumerate(diagnostics.items(), 1):
        definition = profiles[name]["specification"]
        spec, fine, excluded = (
            definition["spec"],
            definition["fine"],
            definition["excluded"],
        )
        for source, saved in obj["sources"].items():
            actual = data.quality(spec, fine, excluded=excluded, source=source)
            quality_check(actual, saved, name + "/source/" + source)
            diagnostic_count += 1
        for saved in obj["pool_omissions"]:
            actual = data.quality(
                c.omit(spec, saved["omitted_seed"]), fine, excluded=excluded
            )
            quality_check(actual, saved["result"], name + "/pool")
            diagnostic_count += 1
        for saved in obj["batch_omissions"]:
            removed = set(excluded) | {
                k for k, b in data.batch.items() if b == saved["omitted_batch"]
            }
            actual = data.quality(spec, fine, excluded=removed)
            quality_check(actual, saved["result"], name + "/batch")
            diagnostic_count += 1
        if n % 28 == 0:
            print(
                f"Nonlinear quality diagnostic packages freshly checked: {n}/112",
                flush=True,
            )
    need(bound_count == 224 and diagnostic_count == 2452, "Quality coverage")
    write(
        OUT / "independent_quality_points.json",
        dict(
            status="all_paired_quality_points_bounds_and_source_pool_batch_diagnostics_reconstructed",
            created_utc=io.utc(),
            results=results,
            paired_objects=112,
            ordinary_scalar_points=224,
            sensitivity_scalar_points=448,
            missing_bound_objects=bound_count,
            diagnostic_point_objects=diagnostic_count,
            diagnostic_scalar_points=diagnostic_count * 2,
            bindings={
                io.rel(PARTS[7] / f): io.sha(PARTS[7] / f)
                for f in ("quality_profiles.json", "quality_omissions_and_sources.json")
            },
            new_model_calls=0,
        ),
    )
    print(
        "All paired quality points, 224 bound objects and 2,452 diagnostic objects passed.",
        flush=True,
    )


def run_summary(data, ix, col, strict):
    valid = data.valid[ix, col]
    take = valid >= (3 if strict else 1)
    good = ix[take]
    g = data.gold[good]
    p = data.hate[good, col]
    counts = Counter(
        {
            "eligible": len(ix),
            "valid": len(good),
            "nulls": int((valid == 0).sum()),
            "unusable": int((~take).sum()),
        }
    )
    counts.update(
        {"valid_votes_" + str(k): v for k, v in Counter(map(int, valid)).items()}
    )
    counts.update(
        {
            "tp": int(((g == 1) & (p == 1)).sum()),
            "tn": int(((g == 0) & (p == 0)).sum()),
            "fp": int(((g == 0) & (p == 1)).sum()),
            "fn": int(((g == 1) & (p == 0)).sum()),
        }
    )
    fineix = good[data.gold_types[good] >= 0]
    gt = data.gold_types[fineix]
    pt = data.types[fineix, col]
    goldsets = [set(k for k in range(7) if v & (1 << k)) for v in gt]
    predsets = [set(k for k in range(7) if v & (1 << k)) for v in pt]
    counts.update(
        {
            "type_valid": len(fineix),
            "type_contains_gold": sum(g <= p for g, p in zip(goldsets, predsets)),
            "type_exact": sum(g == p for g, p in zip(goldsets, predsets)),
            "type_extra_labels": sum(len(p - g) for g, p in zip(goldsets, predsets)),
            "type_missing_labels": sum(len(g - p) for g, p in zip(goldsets, predsets)),
            "type_gold_labels": sum(map(len, goldsets)),
            "type_predicted_labels": sum(map(len, predsets)),
        }
    )
    cardinality = Counter("gold_" + str(len(g)) for g in goldsets)
    cardinality.update("prediction_" + str(len(p)) for p in predsets)
    per = {}
    for k, name in enumerate(c.TYPES):
        g = (gt & (1 << k)) != 0
        p = (pt & (1 << k)) != 0
        tp = int((g & p).sum())
        ng = int(g.sum())
        npred = int(p.sum())
        per[name] = dict(
            tp=tp,
            fp=npred - tp,
            fn=ng - tp,
            support=ng,
            predicted=npred,
            precision=c.frac(tp, npred),
            recall=c.frac(tp, ng),
            f1=c.frac(2 * tp, ng + npred),
        )
    metrics = c.score(data.gold[good], data.hate[good, col], False)
    metrics.update(c.score(gt, pt, True))
    metrics.update(
        {
            "FPR": c.frac(counts["fp"], counts["fp"] + counts["tn"], None),
            "FNR": c.frac(counts["fn"], counts["fn"] + counts["tp"], None),
            "type_contains_gold": c.frac(
                counts["type_contains_gold"], len(fineix), None
            ),
            "type_gold_mean_cardinality": c.frac(
                counts["type_gold_labels"], len(fineix), None
            ),
            "type_mean_cardinality": c.frac(
                counts["type_predicted_labels"], len(fineix), None
            ),
        }
    )
    return dict(
        counts=dict(counts),
        cardinality=dict(cardinality),
        per_label=per,
        metrics=metrics,
    )


def run_phase(data, plan):
    count = 0
    mask_inventory = {}
    for filename in (
        "all_own_valid_quality_runs.json",
        "all_common_mask_quality_runs.json",
    ):
        packages = io.read(PARTS[5] / filename)
        for name, package in packages.items():
            cohort, mask = name.split("/")
            strict = mask == "all_three_votes"
            ix = data.selected(exclusions(plan, cohort), mask == "one_per_cluster")
            if filename.startswith("all_common"):
                ix = ix[(data.valid[ix, :] >= (3 if strict else 1)).all(axis=1)]
                need(
                    [data.keys[i] for i in ix] == package["common_ids"],
                    name + " all-condition mask",
                )
                close(len(ix), package["common_n"], name + " all-condition n")
                mask_inventory[name] = dict(
                    common_n=len(ix),
                    benign=int((data.gold[ix] == 0).sum()),
                    hateful=int((data.gold[ix] == 1).sum()),
                )
            need(len(package["runs"]) == 222, name + " all runs")
            seen = set()
            for saved in package["runs"]:
                identity = saved["condition"], saved["repeat"]
                need(identity not in seen, "Repeated quality run")
                seen.add(identity)
                actual = run_summary(data, ix, data.columns[identity], strict)
                close(actual["metrics"], saved["metrics"], name + "/metrics")
                close(actual["per_label"], saved["per_label"], name + "/labels")

                for k in set(actual["counts"]) | set(saved["counts"]):
                    close(
                        actual["counts"].get(k, 0),
                        saved["counts"].get(k, 0),
                        name + "/counts/" + k,
                    )
                need(
                    actual["cardinality"] == saved["cardinality"],
                    name + " cardinalities",
                )
                count += 1
    need(count == 2664, "Run-summary coverage")
    write(
        OUT / "independent_run_scores.json",
        dict(
            status="all_own_valid_and_all_condition_quality_run_scores_reconstructed",
            created_utc=io.utc(),
            runs_checked=count,
            per_label_rows_checked=count * 7,
            all_condition_masks=mask_inventory,
            bindings={
                io.rel(PARTS[5] / f): io.sha(PARTS[5] / f)
                for f in (
                    "all_own_valid_quality_runs.json",
                    "all_common_mask_quality_runs.json",
                )
            },
            new_model_calls=0,
        ),
    )
    print("All 2,664 run summaries and 18,648 per-label rows passed.", flush=True)


def quality_draw_phase():
    profiles = io.read(PARTS[7] / "quality_profiles.json")
    proofs = io.read(PARTS[5] / "independent_checks.json")
    checks = {}
    count = 0
    for name, obj in profiles.items():
        path = PARTS[5] / "draws" / (name.replace("/", "__") + ".npy")
        draws = np.load(path, allow_pickle=False)

        metrics = (
            ("type_macro_f1_fixed7", "type_exact_set")
            if obj["specification"]["fine"]
            else ("binary_macro_f1", "balanced_accuracy")
        )
        proof = proofs[name]
        need(
            draws.shape == (10000, 2)
            and proof["draws_checked"] == 10000
            and proof["values_checked"] == 20000,
            name + " shape/proof",
        )
        need(
            proof["max_absolute_error"] < 1e-12,
            name + " prior independent reconstruction",
        )
        close(obj["ordinary"]["confidence"], 0.95, name + " quality confidence")
        tails = {}
        for i, metric in enumerate(metrics):
            valid = np.isfinite(draws[:, i])
            undefined = int((~valid).sum())
            need(
                undefined == obj["ordinary"]["metrics"][metric]["undefined_draws"],
                name + " undefined bootstrap",
            )
            interval = (
                np.quantile(draws[valid, i], [0.025, 0.975]).tolist()
                if valid.any()
                else None
            )
            close(
                interval,
                obj["ordinary"]["metrics"][metric]["interval95"],
                name + " array interval",
            )
            p = proof["quantiles"][metric]
            close(interval, p["exact_interval"], name + " exact tail witness")
            for tail in p["tails"]:
                (
                    exact(
                        F(tail["exact_quantile"]),
                        F(tail["value"]),
                        name + " exact tail",
                    )
                    if isinstance(tail["value"], str)
                    else close(
                        F(tail["exact_quantile"]), tail["value"], name + " exact tail"
                    )
                )
            tails[metric] = p["exact_interval"]
        checks[name] = dict(
            draw_path=io.rel(path),
            draw_sha256=io.sha(path),
            values_read_back=int(draws.size),
            prior_independent_reconstruction_reused=True,
            exact_intervals=tails,
        )
        count += draws.size
    need(count == 2240000, "Nonlinear draw coverage")
    write(
        OUT / "quality_uncertainty_readback.json",
        dict(
            status="all_quality_arrays_and_independent_fractional_tail_proofs_verified",
            created_utc=io.utc(),
            checks=checks,
            values_read_back=int(count),
            freshly_regenerated_values=0,
            bindings={
                io.rel(PARTS[5] / "independent_checks.json"): io.sha(
                    PARTS[5] / "independent_checks.json"
                )
            },
            new_model_calls=0,
        ),
    )
    print(
        "All 2.24 million quality bootstrap values and exact interval witnesses read back.",
        flush=True,
    )


def main(phase):
    start = time.monotonic()
    if phase == "bindings":
        bindings_phase()
    elif phase == "quality-draws":
        quality_draw_phase()
    else:
        data, plan, specs, profiles = load_data()
        if phase == "rates":
            rate_phase(data, plan, specs, profiles)
        elif phase == "sources":
            source_phase(data, plan, specs)
        elif phase == "quality":
            quality_phase(data, plan, specs)
        elif phase == "runs":
            run_phase(data, plan)
        elif phase == "probe":
            print("Matrix/specification preflight passed.", flush=True)
    print(
        json.dumps(
            dict(
                phase=phase,
                elapsed_seconds=time.monotonic() - start,
                python=platform.python_version(),
                numpy=np.__version__,
                new_model_calls=0,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=[
            "bindings",
            "probe",
            "rates",
            "sources",
            "quality",
            "runs",
            "quality-draws",
        ],
    )
    args = parser.parse_args()
    with io.offline():
        main(args.phase)

"""Sampled banks results."""

import argparse
from collections import Counter, defaultdict
import io
import json
from pathlib import Path
import socket
import time
from unittest.mock import patch
import numpy as np
from analysis.knowledge.sampled_banks import run as b
from analysis.knowledge.sampled_banks import analysis_plan as protocol
from analysis.knowledge.components import results as m

a = b.a
ROOT = b.ROOT
BASE = b.BASE
DEST = b.STUDY / "pool_results_v1"
s = m.s
primary = m.primary
secondary = m.secondary
severity = m.severity
ANALYSIS_SHA = "1fa5569a48957edd73f58d7c1d0179e31f14590d3f37b583c3ddd85353fc3f8c"
SOURCE_NAMES = [
    "analysis/knowledge/sampled_banks/results.py",
    "tests/knowledge/sampled_banks/test_results.py",
    "analysis/knowledge/sampled_banks/analysis_plan.py",
    "analysis/knowledge/components/results.py",
    "analysis/common/scoring.py",
    "analysis/knowledge/banks/primary.py",
    "analysis/knowledge/banks/secondary.py",
    "analysis/knowledge/banks/severity.py",
]


def forbid(*args, **kwargs):
    raise AssertionError("Offline pool scoring: no network or credentials")


def verify(hashes):
    for name, h in hashes.items():
        a.require(a.digest(ROOT / name) == h, "Frozen scoring input changed: " + name)


def keys(config):
    return [(c, r) for c in config["conditions"] for r in config["decoder_repeats"]]


def contract(config):
    a.require(
        config["conditions"] == list(protocol.CONDITIONS)
        and config["decoder_repeats"] == [0, 1, 2]
        and (config["pool_seeds"] == [42, 43, 44])
        and (config["groups"] == protocol.GROUPS),
        "Wrong pool/repeat grouping",
    )
    a.require(
        config["group_weights"]
        == {g: {c: 1 / len(cs) for c in cs} for g, cs in protocol.GROUPS.items()},
        "Unequal pool weights",
    )
    boot = config["contract"]["bootstrap"]
    a.require(
        boot["draws"] == 10000
        and boot["seed"] == 20260926
        and (boot["primary_family_size"] == 6)
        and (boot["family_alpha"] == 0.05)
        and (boot["primary_confidence"] == 1 - 0.05 / 6)
        and (boot["secondary_confidence"] == 0.95)
        and (boot["percentile_method"] == "numpy.quantile(method=linear)"),
        "Wrong fixed uncertainty contract",
    )
    expected_primary = [
        {"endpoint": ep, "contrast": l + "_minus_" + r, "left": l, "right": r}
        for ep in protocol.FOCAL
        for l, r in protocol.PRIMARY
    ]
    expected_secondary = [
        {"endpoint": ep, "contrast": l + "_minus_" + r, "left": l, "right": r}
        for ep in protocol.ENDPOINT_ORDER
        for l, r in protocol.SECONDARY + (() if ep in protocol.FOCAL else protocol.PRIMARY)
    ]
    a.require(
        config["primary_contrasts"] == expected_primary
        and config["secondary_contrasts"] == expected_secondary,
        "Declared contrasts differ",
    )


def complete_stages(schedule, subset, available):
    selected = m.selected_stages(schedule, subset)
    a.require(
        {st["id"] for st in selected} <= set(available),
        "All contributing stages must be audited before scoring",
    )
    return selected


def summary(values):
    if any((v is None for v in values)):
        return {
            "values": values,
            "mean": None,
            "sample_sd": None,
            "min": None,
            "max": None,
            "range": None,
            "qualification": "Undefined where any repeat has no valid eligible score; do not drop it or fill with zero.",
        }
    return m.summarize(values)


def compare(contribs, config, subset, *, _test_draws=None):
    contract(config)
    ordered = keys(config)
    a.require(
        subset in config["endpoints"] and len(ordered) == 33 and (set(contribs) == set(ordered)),
        "Wrong endpoint/settings",
    )
    first = contribs[ordered[0]]
    cc = [contribs[k] for k in ordered]
    idx = first["macro_indices"]
    for c in cc:
        a.require(
            c["ids"] == first["ids"]
            and c["labels"] == first["labels"]
            and (c["macro_indices"] == idx)
            and np.array_equal(c["gold"], first["gold"]),
            "Unpaired scoring inputs",
        )
    common = np.logical_and.reduce([c["valid"] for c in cc])
    independent_mask = protocol.shared_valid(
        [True] * len(first["ids"]), {k: contribs[k]["valid"].tolist() for k in ordered}
    )
    a.require(common.tolist() == independent_mask, "Shared eligible/valid mask differs")
    n = int(common.sum())
    usable = bool(n and idx)
    draws = config["contract"]["bootstrap"]["draws"] if _test_draws is None else _test_draws
    a.require(type(draws) is int and draws > 0, "Invalid bootstrap draw count")
    seed = config["contract"]["bootstrap"]["seed"]
    points = boot = check = None
    if usable:
        values = np.stack([c["counts"][common] for c in cc])
        points = s.macro(values.sum(axis=1), idx)
        a.require(
            np.allclose(
                points, primary.independent_f1(values.sum(axis=1), idx), atol=1e-13, rtol=0
            ),
            "Point F1 differs",
        )
        boot = np.empty((draws, 33))
        check = np.empty_like(boot)
        rng = np.random.default_rng(seed)
        for i in range(draws):
            pick = rng.integers(0, n, n)
            boot[i] = s.macro(values[:, pick].sum(axis=1), idx)
        rng = np.random.default_rng(seed)
        for i in range(draws):
            frequencies = np.bincount(rng.integers(0, n, n), minlength=n)
            check[i] = primary.independent_f1(np.einsum("sijk,i->sjk", values, frequencies), idx)
        a.require(
            np.allclose(boot, check, atol=1e-13, rtol=0), "Frequency-weighted bootstrap differs"
        )
    positions = {
        c: [ordered.index((c, r)) for r in config["decoder_repeats"]] for c in config["conditions"]
    }
    quantile = lambda vals, confidence: np.quantile(
        vals, [(1 - confidence) / 2, 1 - (1 - confidence) / 2], method="linear"
    ).tolist()
    bounds = {}
    per_pool = {}
    group_values = {}
    group_boot = {}
    group_check = {}
    common_aggregation = (
        protocol.aggregate({c: points[positions[c]].tolist() for c in config["conditions"]})
        if usable
        else None
    )
    for c in config["conditions"]:
        setting_bounds = [
            s.score(contribs[c, r])["invalid_inclusive_bounds"] for r in config["decoder_repeats"]
        ]
        bounds[c] = (
            np.mean(setting_bounds, axis=0).tolist()
            if all((v is not None for x in setting_bounds for v in x))
            else [None, None]
        )
        vals = points[positions[c]].tolist() if usable else [None] * 3
        per_pool[c] = {
            "common_valid_summary": summary(vals),
            "mean_full_roster_failure_bounds": bounds[c],
            "decoder_ci95": (
                [quantile(boot[:, i], 0.95) for i in positions[c]] if usable else [None] * 3
            ),
            "mean_ci95": quantile(boot[:, positions[c]].mean(axis=1), 0.95) if usable else None,
        }
        if usable:
            a.require(
                abs(
                    per_pool[c]["common_valid_summary"]["mean"]
                    - common_aggregation["per_pool"][c]["mean"]
                )
                < 1e-14,
                "Pool aggregation differs",
            )
    groups = {}
    for g, members in config["groups"].items():
        member_means = [per_pool[c]["common_valid_summary"]["mean"] for c in members]
        gbounds = (
            np.mean([bounds[c] for c in members], axis=0).tolist()
            if all((v is not None for c in members for v in bounds[c]))
            else [None, None]
        )
        groups[g] = {
            "mean": common_aggregation["groups"][g]["mean"] if usable else None,
            "pool_means": member_means,
            "between_pool_sample_sd": (
                common_aggregation["groups"][g]["between_pool_sample_sd"] if usable else None
            ),
            "between_pool_range": (
                common_aggregation["groups"][g]["between_pool_range"] if usable else None
            ),
            "mean_full_roster_failure_bounds": gbounds,
            "ci95": None,
        }
        if usable:
            group_values[g] = [i for c in members for i in positions[c]]
            group_boot[g] = np.mean([boot[:, positions[c]].mean(axis=1) for c in members], axis=0)
            group_check[g] = np.mean([check[:, positions[c]].mean(axis=1) for c in members], axis=0)
            groups[g]["ci95"] = quantile(group_boot[g], 0.95)
            a.require(
                np.allclose(groups[g]["ci95"], quantile(group_check[g], 0.95), atol=1e-12, rtol=0),
                "Group interval differs",
            )
    contrasts = {}
    for is_primary, items in (
        (True, config["primary_contrasts"]),
        (False, config["secondary_contrasts"]),
    ):
        for row in items:
            if row["endpoint"] != subset:
                continue
            left, right = (row["left"], row["right"])
            lo, hi = (
                groups[left]["mean_full_roster_failure_bounds"],
                groups[right]["mean_full_roster_failure_bounds"],
            )
            dbounds = (
                [lo[0] - hi[1], lo[1] - hi[0]]
                if all((x is not None for x in lo + hi))
                else [None, None]
            )
            confidence = (
                config["contract"]["bootstrap"]["primary_confidence"] if is_primary else None
            )
            out = {
                **row,
                "primary_family_member": is_primary,
                "n_common_valid": n,
                "mean_difference": None,
                "ci95": None,
                "family_ci": None,
                "family_confidence": confidence,
                "invalid_inclusive_difference_bounds": dbounds,
            }
            if usable:
                li, ri = (group_values[left], group_values[right])
                delta = (boot[:, li][:, :, None] - boot[:, ri][:, None, :]).mean(axis=(1, 2))
                checked = (check[:, li][:, :, None] - check[:, ri][:, None, :]).mean(axis=(1, 2))
                point = float((points[li, None] - points[None, ri]).mean())
                out.update(
                    mean_difference=point,
                    ci95=quantile(delta, 0.95),
                    family_ci=quantile(delta, confidence) if confidence else None,
                )
                a.require(
                    abs(point - (groups[left]["mean"] - groups[right]["mean"])) < 1e-14,
                    "Group difference differs",
                )
                a.require(
                    np.allclose(delta, group_boot[left] - group_boot[right], atol=1e-13, rtol=0),
                    "Nested group averaging differs",
                )
                for conf in (0.95, confidence):
                    if conf:
                        a.require(
                            np.allclose(
                                quantile(delta, conf), quantile(checked, conf), atol=1e-12, rtol=0
                            ),
                            "Contrast interval differs",
                        )
            contrasts[row["contrast"]] = out
    return {
        "n_eligible": len(first["ids"]),
        "n_common_valid": n,
        "common_valid_ids": [rid for rid, ok in zip(first["ids"], common) if ok],
        "excluded_from_common_valid": [rid for rid, ok in zip(first["ids"], common) if not ok],
        "defined": usable,
        "conditions": per_pool,
        "groups": groups,
        "contrasts": contrasts,
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "independent_bootstrap_setting_values": draws * 33 if usable else 0,
        "mask": "Original eligible rows and valid in all eleven pools x three repeats; common to every table/contrast.",
        "variation": "Decoder sample SD within each fixed pool; sample SD across three pool means separately. Reference between-pool SD undefined.",
    }


def scorer(endpoint):
    return (
        severity.checked_score
        if endpoint["dimension"] == "severity"
        else primary.checked_score if endpoint["dimension"] == "legal" else secondary.checked_score
    )


def prepare(subset):
    folder = DEST / subset
    a.require(not (folder / "plan.json").exists(), "Preserve endpoint scoring plan")
    a.require(
        a.digest(BASE / "analysis_plan.json") == ANALYSIS_SHA,
        "Pre-response scoring contract changed",
    )
    config = a.read(BASE / "analysis_plan.json")
    contract(config)
    plan = b.validate()
    available = [p.stem for p in (BASE / "stage_reviews").glob("*.json")]
    stages = complete_stages(a.read(BASE / "schedule.json"), subset, available)
    tested = DEST / "development/tests_passed.json"
    tests = a.read(tested)
    a.require(
        tests["passed"]
        and tests["environment"] == b.core.environment()
        and (tests["analysis_sha256"] == ANALYSIS_SHA),
        "Passing current tests required",
    )
    verify(tests["source_hashes"])
    bindings = []
    hashes = {
        **plan["input_hashes"],
        **plan["source_hashes"],
        **plan["generated_hashes"],
        **tests["source_hashes"],
        b.rel(b.PLAN): a.digest(b.PLAN),
        b.rel(tested): a.digest(tested),
    }
    for st in stages:
        rp = BASE / "stage_reviews" / f"{st['id']}.json"
        review = a.read(rp)
        ev_path = ROOT / review["evidence"]
        ev = a.read(ev_path)
        cp = BASE / "live/stages" / st["id"] / "complete.json"
        a.require(
            review["status"] == "audited"
            and review["stage"] == st["id"]
            and (review["plan_sha256"] == a.digest(b.PLAN))
            and (review["evidence_sha256"] == a.digest(ev_path))
            and (review["complete_sha256"] == a.digest(cp)),
            "Stage audit binding differs",
        )
        a.require(
            ev["semantic_replay_items"] == ev["independent_field_vote_checks"] == st["items"]
            and ev["actual_audit_API_calls"] == 0,
            "Incomplete stage audit",
        )
        acceptance_path = b.STUDY / f"stage{int(st['id'][-2:])}_review_v1/audit_acceptance.json"
        accept = a.read(acceptance_path)
        diagnostic = ROOT / accept["diagnostic"]
        a.require(
            accept["stage"] == st["id"]
            and accept["plan_sha256"] == a.digest(b.PLAN)
            and (accept["main_audit_sha256"] == a.digest(ev_path))
            and (accept["diagnostic_sha256"] == a.digest(diagnostic)),
            "Follow-up acceptance differs",
        )
        hashes.update(ev["verified_input_hashes"])
        for path in (rp, ev_path, cp, acceptance_path, diagnostic):
            hashes[b.rel(path)] = a.digest(path)
        bindings.append({"stage": st["id"], "audit": b.rel(ev_path), "complete": b.rel(cp)})
    verify(hashes)
    a.save_new(
        folder / "plan.json",
        {
            "status": "prepared_before_endpoint_scoring",
            "created_utc": a.utc(),
            "subset": subset,
            "analysis_sha256": ANALYSIS_SHA,
            "hashes": hashes,
            "stage_bindings": bindings,
            "scope": "Only a complete DEV endpoint, unchanged thirty-three-setting mask and pre-response aggregation/uncertainty. No TEST, API or adaptation.",
        },
    )
    print("Prepared fixed analysis for " + subset, flush=True)


def validate(subset):
    p = a.read(DEST / subset / "plan.json")
    a.require(
        p["subset"] == subset
        and p["analysis_sha256"] == a.digest(BASE / "analysis_plan.json") == ANALYSIS_SHA,
        "Endpoint plan differs",
    )
    verify(p["hashes"])
    return p


def run(subset):
    import pandas as pd
    import yaml

    p = validate(subset)
    folder = DEST / subset
    config = a.read(BASE / "analysis_plan.json")
    contract(config)
    a.require(
        not any(
            ((folder / name).exists() for name in ("assembled", "results.json", "complete.json"))
        ),
        "Preserve completed/partial score output",
    )
    rows = pd.read_parquet(ROOT / f"data/processed/{subset}.parquet").to_dict("records")
    primary.keyed(rows)
    original = s.roster(
        subset, rows, yaml.safe_load((ROOT / "config/taxonomy.yaml").read_text(encoding="utf8"))
    )
    ep = config["endpoints"][subset]
    a.require(
        {**original, "inherited_core_role": original["role"], "role": ep["role"]} == ep,
        "Original gold roster or label policy differs",
    )
    dev = primary.keyed(b.core.rows(protocol.CORE / "dev_inputs.jsonl"), "key")
    schedule = a.read(BASE / "schedule.json")
    selected = complete_stages(schedule, subset, [x["stage"] for x in p["stage_bindings"]])
    settings = defaultdict(dict)
    origins = {}
    hashes = {}
    plan_sha = a.digest(b.PLAN)
    for st, binding in zip(selected, p["stage_bindings"]):
        a.require(st["id"] == binding["stage"], "Stage order differs")
        audit = a.read(ROOT / binding["audit"])
        complete = a.read(ROOT / binding["complete"])
        for job in st["jobs"]:
            if job["subset"] != subset:
                continue
            directory = b.item_dir(BASE / "live", job, plan_sha)
            rp = directory / "result.json"
            cp = directory / "complete.json"
            for path, h in (
                (rp, audit["verified_output_hashes"][b.rel(rp)]),
                (cp, complete["item_receipt_hashes"][b.rel(cp)]),
            ):
                a.require(a.digest(path) == h, "Audited prediction changed")
                hashes[b.rel(path)] = h
            result = a.read(rp)
            a.require(
                result["identity"] == b.identity(job, plan_sha)
                and result["messages_sha256"] == job["messages_sha256"],
                "Prediction identity differs",
            )
            inp = dev[job["key"]]
            m.check_input_scope(inp, subset, ep["dimension"])
            rid = str(inp["item_id"])
            ident = (job["condition"], job["repeat"])
            a.require(rid not in settings[ident], "Duplicate prediction identity")
            settings[ident][rid] = {"result": result["result"], "n_valid": result["n_valid"]}
            origins[ident, rid] = {
                "key": job["key"],
                "source": b.rel(rp),
                "source_sha256": hashes[b.rel(rp)],
            }
    m.check_settings(
        settings,
        [str(r["id"]) for r in rows],
        {"conditions": config["conditions"], "repeats": config["decoder_repeats"]},
    )
    contributions = {}
    conditions = {}
    score = scorer(ep)
    for c in config["conditions"]:
        runs = []
        for rep in config["decoder_repeats"]:
            contribution, metric = score(rows, settings[c, rep], ep)
            contributions[c, rep] = contribution
            runs.append(
                {
                    "replicate": rep,
                    "score": metric,
                    "valid_vote_distribution": dict(
                        sorted(Counter((x["n_valid"] for x in settings[c, rep].values())).items())
                    ),
                }
            )
        conditions[c] = {
            "runs": runs,
            "own_valid_summary": summary([r["score"]["macro_f1_valid_only"] for r in runs]),
            "null_predictions": sum((r["score"]["n_invalid"] for r in runs)),
        }
    comparison = compare(contributions, config, subset)
    a.require(comparison["bootstrap_draws"] == 10000, "Production draw count differs")
    kept = set(comparison["common_valid_ids"])
    common_rows = [row for row in rows if str(row["id"]) in kept]
    common_ep = {
        **ep,
        "scorable_ids": comparison["common_valid_ids"],
        "scorable_rows": len(kept),
        "rows": len(kept),
    }
    for c in config["conditions"]:
        common_runs = []
        for rep in config["decoder_repeats"]:
            if not comparison["defined"]:
                common_runs.append({"replicate": rep, "score": None})
                continue
            selected_outputs = {
                str(row["id"]): settings[c, rep][str(row["id"])] for row in common_rows
            }
            _, cm = score(common_rows, selected_outputs, common_ep)
            a.require(
                abs(
                    cm["macro_f1_valid_only"]
                    - comparison["conditions"][c]["common_valid_summary"]["values"][rep]
                )
                < 1e-14,
                "Direct common-roster score differs",
            )
            y = np.asarray(
                [
                    s.bits(s.truth(row, ep["dimension"]), ep["dimension"], ep["all_labels"])
                    for row in common_rows
                ]
            )
            x = np.asarray(
                [
                    s.bits(
                        selected_outputs[str(row["id"])]["result"][ep["dimension"]],
                        ep["dimension"],
                        ep["all_labels"],
                    )
                    for row in common_rows
                ]
            )
            cm["exact_match_rate"] = float((y == x).all(axis=1).mean())
            cm["hamming_loss"] = float((y != x).mean())
            cm["mean_gold_cardinality"] = float(y.sum(axis=1).mean())
            cm["mean_predicted_cardinality"] = float(x.sum(axis=1).mean())
            common_runs.append({"replicate": rep, "score": cm})
        conditions[c]["common_runs"] = common_runs
    validate(subset)
    verify(hashes)
    assembled = {}
    for ident in keys(config):
        c, rep = ident
        path = folder / "assembled" / f"{c}_rep{rep}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        exported = []
        with path.open("x", encoding="utf8", newline="\n") as f:
            for row in rows:
                rid = str(row["id"])
                record = {
                    "id": rid,
                    "condition": c,
                    "replicate": rep,
                    **settings[ident][rid],
                    **origins[ident, rid],
                }
                f.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
                exported.append(record)
        a.require(b.core.rows(path) == exported, "Assembled export roundtrip differs")
        assembled[b.rel(path)] = {"sha256": a.digest(path), "rows": len(rows)}
    report = {
        "status": "complete_pool_endpoint_scored_and_independently_checked",
        "created_utc": a.utc(),
        "subset": subset,
        "plan_sha256": a.digest(folder / "plan.json"),
        "analysis_sha256": ANALYSIS_SHA,
        "endpoint": ep,
        "conditions": conditions,
        "paired_comparison": comparison,
        "assembled": assembled,
        "verified_input_hashes": p["hashes"],
        "verified_prediction_hashes": hashes,
        "verification": {
            "settings": 33,
            "predictions": 33 * len(rows),
            "independent_setting_metrics_and_bounds": 33,
            "direct_common_roster_metrics": 33 if comparison["defined"] else 0,
            "independent_bootstrap_setting_values": comparison[
                "independent_bootstrap_setting_values"
            ],
            "contrasts": len(comparison["contrasts"]),
            "export_rows_roundtripped": 33 * len(rows),
        },
        "new_API_calls": 0,
        "gold_rosters_scored": [subset],
        "limitations": [
            config["contract"]["limits"],
            config["contract"]["retrieval_qualification"],
            "Own-valid scores use different per-setting masks; main comparisons use one thirty-three-setting mask.",
            "Full-roster severity bounds relax the categorical constraint and may not be jointly attainable.",
        ],
    }
    a.save_new(folder / "results.json", report)
    date = report["created_utc"][:10].replace("-", "")
    evidence = ROOT / f"Thesis_Log/astra_result/evidence/sq1_pool_{subset}_results_{date}_v1.json"
    a.save_new(evidence, report)
    a.save_new(
        folder / "complete.json",
        {
            "status": report["status"],
            "plan_sha256": report["plan_sha256"],
            "results_sha256": a.digest(folder / "results.json"),
            "evidence": b.rel(evidence),
            "evidence_sha256": a.digest(evidence),
            "assembled": assembled,
        },
    )
    print(
        json.dumps(
            {
                "subset": subset,
                "shared_n": comparison["n_common_valid"],
                "groups": comparison["groups"],
                "contrasts": comparison["contrasts"],
                "evidence": b.rel(evidence),
            },
            indent=2,
        ),
        flush=True,
    )


def tests():
    import unittest

    receipt = DEST / "development/tests_passed.json"
    a.require(not receipt.exists(), "Preserve completed scoring tests")
    attempt = DEST / "development" / ("attempt_" + str(time.time_ns()))
    hashes = {name: a.digest(ROOT / name) for name in SOURCE_NAMES}
    a.save_new(
        attempt / "sources.json",
        {name: (ROOT / name).read_text(encoding="utf8") for name in SOURCE_NAMES[:2]},
    )
    suite = unittest.defaultTestLoader.loadTestsFromName(
        "tests.knowledge.sampled_banks.test_results"
    )
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    record = {
        "created_utc": a.utc(),
        "passed": result.wasSuccessful(),
        "tests": result.testsRun,
        "output": stream.getvalue(),
        "environment": b.core.environment(),
        "source_hashes": hashes,
        "analysis_sha256": a.digest(BASE / "analysis_plan.json"),
    }
    a.save_new(attempt / "tests.json", record)
    print(stream.getvalue(), flush=True)
    a.require(result.wasSuccessful(), "Scoring tests failed; source and output preserved")
    verify(hashes)
    a.save_new(receipt, record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tests", "prepare", "run"])
    parser.add_argument("--subset", choices=b.ENDPOINT_ORDER)
    args = parser.parse_args()
    if args.mode != "tests" and (not args.subset):
        parser.error("--subset required")
    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket, "create_connection", forbid
    ), patch.object(b.core, "live_client", forbid):
        (
            tests()
            if args.mode == "tests"
            else prepare(args.subset) if args.mode == "prepare" else run(args.subset)
        )

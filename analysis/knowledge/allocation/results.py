"""Compare the three definition-allocation policies."""

import argparse
from collections import Counter, defaultdict
import io
import json
import socket
import time
from unittest.mock import patch
import numpy as np
from analysis.knowledge.allocation import run as b
from analysis.knowledge.components import results as metrics

a = b.a
s = metrics.s
ROOT = b.ROOT
BASE = b.BASE
primary = metrics.primary
secondary = metrics.secondary
severity = metrics.severity
check_input_scope = metrics.check_input_scope
selected_stages = metrics.selected_stages
setting_keys = metrics.setting_keys
check_settings = metrics.check_settings
summarize = metrics.summarize
DEST = b.STUDY / "allocation_results_v1"
CONDITIONS = ["flat2", "flat7", "proportional7"]
CONTRASTS = {
    "flat7_minus_flat2": {"flat2": -1, "flat7": 1},
    "proportional_minus_flat2": {"flat2": -1, "proportional7": 1},
    "proportional_minus_flat7": {"flat7": -1, "proportional7": 1},
}
FOCAL = ["en_dev_eval_main", "en_dev_eval_sq3_types"]
SOURCE_NAMES = [
    "analysis/knowledge/allocation/results.py",
    "tests/knowledge/allocation/test_results.py",
    "analysis/knowledge/components/results.py",
    "analysis/common/scoring.py",
    "analysis/knowledge/banks/primary.py",
    "analysis/knowledge/banks/secondary.py",
    "analysis/knowledge/banks/severity.py",
]


def forbidden(*args, **kwargs):
    raise AssertionError("Offline allocation scoring only")


def contract(config):
    a.require(
        config["conditions"] == CONDITIONS and config["repeats"] == [0, 1, 2],
        "Wrong allocation settings",
    )
    a.require(
        config["contrasts"] == CONTRASTS and config["focal_endpoints"] == FOCAL,
        "Wrong allocation contrasts/endpoints",
    )
    a.require(
        config["focal_family_size"] == 6
        and np.isclose(config["focal_family_confidence"], 1 - 0.05 / 6, rtol=0, atol=1e-14),
        "Wrong SIX-contrast focal family",
    )
    a.require(
        config["descriptive_confidence"] == 0.95 and config["no_partial_scoring"],
        "Wrong descriptive/roster contract",
    )


def compare(contribs, config, *, focal):
    contract(config)
    keys = setting_keys(config)
    a.require(
        len(keys) == 9 and set(contribs) == set(keys),
        "Exactly nine condition/repeat settings required",
    )
    first = contribs[keys[0]]
    for c in contribs.values():
        a.require(
            c["ids"] == first["ids"]
            and c["labels"] == first["labels"]
            and (c["macro_indices"] == first["macro_indices"])
            and np.array_equal(c["gold"], first["gold"]),
            "Unpaired scoring inputs",
        )
    report, means, boot, check, bounds = metrics.resample(contribs, CONDITIONS, config)
    for name, weights in config["contrasts"].items():
        delta = sum((w * boot[arm] for arm, w in weights.items()))
        reference = sum((w * check[arm] for arm, w in weights.items()))
        ci = np.percentile(delta, [2.5, 97.5]).tolist()
        a.require(
            np.allclose(ci, np.percentile(reference, [2.5, 97.5]), rtol=0, atol=1e-12),
            "Independent interval differs",
        )
        confidence = config["focal_family_confidence"] if focal else None
        quantiles = [50 * (1 - confidence), 100 - 50 * (1 - confidence)] if focal else None
        family_ci = np.percentile(delta, quantiles).tolist() if focal else None
        if focal:
            a.require(
                np.allclose(family_ci, np.percentile(reference, quantiles), rtol=0, atol=1e-12),
                "Independent family interval differs",
            )
        low = sum((w * bounds[arm][0 if w >= 0 else 1] for arm, w in weights.items()))
        high = sum((w * bounds[arm][1 if w >= 0 else 0] for arm, w in weights.items()))
        report["contrasts"][name] = {
            "weights": weights,
            "n_common_valid": report["n_common_valid"],
            "mean_difference": sum((w * means[arm] for arm, w in weights.items())),
            "ci95": ci,
            "family_ci": family_ci,
            "family_confidence": confidence,
            "family_percentiles": quantiles,
            "invalid_inclusive_difference_bounds": [float(low), float(high)],
        }
    report["mask"] = (
        "Valid in all three conditions and three repeats (nine settings); shared across all contrasts."
    )
    return report


def prepare(subset):
    folder = DEST / subset
    path = folder / "plan.json"
    a.require(not path.exists(), "Endpoint scoring plan already frozen")
    p = b.validate()
    config = a.read(BASE / "analysis_plan.json")
    contract(config)
    a.require(subset in config["endpoints"], "Unknown analysis endpoint")
    schedule = a.read(BASE / "schedule.json")
    tests_path = DEST / "development/tests_passed.json"
    tests = a.read(tests_path)
    a.require(
        tests["passed"] and tests["environment"] == b.core.environment(),
        "Passing scoring tests required",
    )
    a.require(
        tests["analysis_plan_sha256"] == a.digest(BASE / "analysis_plan.json"),
        "Test contract differs",
    )
    for name, h in tests["source_hashes"].items():
        a.require(a.digest(ROOT / name) == h, "Scoring source changed after tests")
    hashes = {
        **p["input_hashes"],
        **p["source_hashes"],
        **p["generated_hashes"],
        b.rel(b.PLAN): a.digest(b.PLAN),
        b.rel(tests_path): a.digest(tests_path),
        **{name: a.digest(ROOT / name) for name in SOURCE_NAMES},
    }
    bindings = []
    for st in selected_stages(schedule, subset):
        rp = BASE / "stage_reviews" / f"{st['id']}.json"
        a.require(
            rp.exists(),
            "All contributing stages must be audited before endpoint scoring: " + st["id"],
        )
        review = a.read(rp)
        ev_path = ROOT / review["evidence"]
        ev = a.read(ev_path)
        cp = BASE / "live/stages" / st["id"] / "complete.json"
        a.require(
            review["status"] == "audited"
            and review["stage"] == st["id"]
            and (review["plan_sha256"] == a.digest(b.PLAN)),
            "Wrong stage review",
        )
        a.require(
            a.digest(ev_path) == review["evidence_sha256"]
            and a.digest(cp) == review["complete_sha256"],
            "Stage binding changed",
        )
        a.require(
            ev["stage"] == st["id"]
            and ev["semantic_replay_items"] == st["items"]
            and (not ev["actual_audit_API_calls"]),
            "Incomplete/wrong stage audit",
        )
        hashes.update(ev["verified_input_hashes"])
        for source in (rp, ev_path, cp):
            hashes[b.rel(source)] = a.digest(source)
        bindings.append({"stage": st["id"], "review": b.rel(rp), "evidence": b.rel(ev_path)})
    for name, h in hashes.items():
        a.require(a.digest(ROOT / name) == h, "Changed scoring input: " + name)
    a.save_new(
        path,
        {
            "created_utc": a.utc(),
            "status": "prepared_before_endpoint_scoring",
            "subset": subset,
            "hashes": hashes,
            "stage_audits": bindings,
            "analysis_plan_sha256": a.digest(BASE / "analysis_plan.json"),
            "tests_sha256": a.digest(tests_path),
            "scope": "Only this complete DEV endpoint; nine-setting common mask, own-valid scores and failure bounds. Six focal contrasts across EN299 and EN467; other endpoints descriptive. No old repeat pooling or automatic promotion.",
        },
    )
    print("Prepared allocation scoring plan for " + subset + ": " + a.digest(path), flush=True)


def validate(subset):
    p = a.read(DEST / subset / "plan.json")
    a.require(
        p["subset"] == subset
        and p["analysis_plan_sha256"] == a.digest(BASE / "analysis_plan.json"),
        "Scoring plan identity changed",
    )
    for name, h in p["hashes"].items():
        a.require(a.digest(ROOT / name) == h, "Frozen scoring input changed: " + name)
    b.validate()
    return p


def run(subset):
    import pandas as pd
    import yaml

    p = validate(subset)
    folder = DEST / subset
    config = a.read(BASE / "analysis_plan.json")
    contract(config)
    a.require(
        not (folder / "assembled").exists() and (not (folder / "results.json").exists()),
        "Preserve completed/partial endpoint publication",
    )
    rows = pd.read_parquet(ROOT / f"data/processed/{subset}.parquet").to_dict("records")
    primary.keyed(rows)
    ep = s.roster(
        subset, rows, yaml.safe_load((ROOT / "config/taxonomy.yaml").read_text(encoding="utf8"))
    )
    frozen = config["endpoints"][subset]
    a.require(
        {**ep, "inherited_core_role": ep["role"], "role": frozen["role"]} == frozen,
        "Frozen endpoint/support/labels differ",
    )
    ep = frozen
    dev = primary.keyed(b.core.rows(b.preparation.CORE / "dev_inputs.jsonl"), "key")
    schedule = a.read(BASE / "schedule.json")
    plan_sha = a.digest(b.PLAN)
    settings = defaultdict(dict)
    origins = {}
    hashes = {}
    stages = selected_stages(schedule, subset)
    a.require(len(stages) == len(p["stage_audits"]), "Incomplete endpoint stage audits")
    for st, binding in zip(stages, p["stage_audits"]):
        a.require(st["id"] == binding["stage"], "Wrong audit stage order")
        ev = a.read(ROOT / binding["evidence"])
        completion = a.read(BASE / "live/stages" / st["id"] / "complete.json")
        for job in st["jobs"]:
            if job["subset"] != subset:
                continue
            d = b.item_dir(BASE / "live", job, plan_sha)
            cp = d / "complete.json"
            rp = d / "result.json"
            for source, h in (
                (cp, completion["item_receipt_hashes"][b.rel(cp)]),
                (rp, ev["verified_output_hashes"][b.rel(rp)]),
            ):
                a.require(a.digest(source) == h, "Audited prediction changed")
                hashes[b.rel(source)] = h
            result = a.read(rp)
            a.require(
                result["identity"] == b.identity(job, plan_sha)
                and result["messages_sha256"] == job["messages_sha256"],
                "Result attribution differs",
            )
            inp = dev[job["key"]]
            check_input_scope(inp, subset, ep["dimension"])
            rid = str(inp["item_id"])
            ident = (job["condition"], job["repeat"])
            a.require(rid not in settings[ident], "Duplicate scheduled prediction")
            settings[ident][rid] = {"result": result["result"], "n_valid": result["n_valid"]}
            origins[ident, rid] = {
                "key": job["key"],
                "source": b.rel(rp),
                "source_sha256": hashes[b.rel(rp)],
            }
    check_settings(settings, [str(r["id"]) for r in rows], config)
    contributions = {}
    conditions = {}
    scorer = (
        severity.checked_score
        if ep["dimension"] == "severity"
        else primary.checked_score if ep["dimension"] == "legal" else secondary.checked_score
    )
    for name in config["conditions"]:
        runs = []
        for rep in config["repeats"]:
            c, metric = scorer(rows, settings[name, rep], ep)
            contributions[name, rep] = c
            runs.append(
                {
                    "replicate": rep,
                    "score": metric,
                    "valid_vote_distribution": dict(
                        sorted(
                            Counter(
                                (out["n_valid"] for out in settings[name, rep].values())
                            ).items()
                        )
                    ),
                }
            )
        conditions[name] = {
            "runs": runs,
            "own_valid_summary": summarize([r["score"]["macro_f1_valid_only"] for r in runs]),
            "null_predictions": sum((r["score"]["n_invalid"] for r in runs)),
            "mean_per_label_f1_own_valid": {
                label: float(np.mean([r["score"]["per_label"][label]["f1"] for r in runs]))
                for label in ep["all_labels"]
            },
        }
    paired = compare(contributions, config, focal=subset in config["focal_endpoints"])
    validate(subset)
    for name, h in hashes.items():
        a.require(a.digest(ROOT / name) == h, "Result changed during scoring")
    assembled = {}
    for ident in setting_keys(config):
        name, rep = ident
        path = folder / "assembled" / f"{name}_rep{rep}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf8", newline="\n") as handle:
            for row in rows:
                rid = str(row["id"])
                handle.write(
                    json.dumps(
                        {
                            "id": rid,
                            "condition": name,
                            "replicate": rep,
                            **settings[ident][rid],
                            **origins[ident, rid],
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
        assembled[b.rel(path)] = {"sha256": a.digest(path), "rows": len(rows)}
    report = {
        "created_utc": a.utc(),
        "status": "complete_endpoint_scored_and_independently_checked",
        "subset": subset,
        "plan_sha256": a.digest(folder / "plan.json"),
        "analysis_plan_sha256": a.digest(BASE / "analysis_plan.json"),
        "endpoint": ep,
        "conditions": conditions,
        "paired_comparison": paired,
        "assembled": assembled,
        "verified_input_hashes": p["hashes"],
        "verified_prediction_hashes": hashes,
        "verification": {
            "settings": 9,
            "predictions": 9 * len(rows),
            "independent_setting_metrics_and_bounds": 9,
            "independent_bootstrap_setting_values": paired["independent_bootstrap_setting_values"],
            "contrasts": 3,
        },
        "new_API_calls": 0,
        "gold_rosters_scored": [subset],
        "limits": [
            config["limits"],
            config["bootstrap"],
            config["validity"],
            "Six focal contrasts jointly across two endpoints use 99.1667% intervals. Other outcomes use descriptive 95% intervals.",
            "Severity failure bounds use conservative bitwise relaxation, not necessarily attainable categorical assignments.",
            "Definition allocation changes coverage, content, order and per-dimension BM25 statistics; equal counts do not imply equal token length.",
        ],
    }
    a.save_new(folder / "results.json", report)
    date = report["created_utc"][:10].replace("-", "")
    evidence = (
        ROOT / f"Thesis_Log/astra_result/evidence/sq1_allocation_{subset}_results_{date}_v1.json"
    )
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
                "common_valid": paired["n_common_valid"],
                "excluded": paired["excluded_from_common_valid"],
                "conditions": {
                    k: {
                        "own": v["own_valid_summary"],
                        "common": paired["conditions"][k]["common_valid_summary"],
                        "nulls": v["null_predictions"],
                    }
                    for k, v in conditions.items()
                },
                "contrasts": paired["contrasts"],
                "evidence": b.rel(evidence),
                "sha256": a.digest(evidence),
            },
            indent=2,
        ),
        flush=True,
    )


def tests():
    import unittest

    path = DEST / "development/tests_passed.json"
    a.require(not path.exists(), "Preserve completed scoring tests")
    attempt = DEST / "development" / ("attempt_" + str(time.time_ns()))
    hashes = {name: a.digest(ROOT / name) for name in SOURCE_NAMES}
    a.save_new(
        attempt / "sources.json",
        {name: (ROOT / name).read_text(encoding="utf8") for name in SOURCE_NAMES[:2]},
    )
    suite = unittest.defaultTestLoader.loadTestsFromName("tests.knowledge.allocation.test_results")
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    record = {
        "created_utc": a.utc(),
        "passed": result.wasSuccessful(),
        "tests": result.testsRun,
        "output": stream.getvalue(),
        "environment": b.core.environment(),
        "source_hashes": hashes,
        "analysis_plan_sha256": a.digest(BASE / "analysis_plan.json"),
    }
    a.save_new(attempt / "tests.json", record)
    print(stream.getvalue(), flush=True)
    a.require(result.wasSuccessful(), "Allocation scoring tests failed; source and output retained")
    for name, h in hashes.items():
        a.require(a.digest(ROOT / name) == h, "Source changed during tests")
    a.save_new(path, record)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tests", "prepare", "run"])
    parser.add_argument("--subset", choices=b.ENDPOINT_ORDER)
    args = parser.parse_args()
    if args.mode != "tests" and (not args.subset):
        parser.error("--subset required")
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(b.core, "live_client", forbidden):
        (
            tests()
            if args.mode == "tests"
            else prepare(args.subset) if args.mode == "prepare" else run(args.subset)
        )

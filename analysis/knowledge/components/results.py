"""Score the component study with separate factorial and reference masks."""

import argparse
from collections import Counter, defaultdict
import json
import socket
from unittest.mock import patch
import numpy as np
from analysis.knowledge.components import run as b
from analysis.knowledge.banks import primary as primary
from analysis.knowledge.banks import secondary as secondary
from analysis.knowledge.banks import severity as severity

a = b.a
s = primary.s
ROOT = b.ROOT
BASE = b.BASE
DEST = b.STUDY / "component_results_v1"
FACTORIAL = ["none", "g", "d", "dg", "e", "eg", "ed", "edg"]
BRIDGE = "common_wording_full_minus_reference"
SOURCE_NAMES = [
    "analysis/knowledge/components/results.py",
    "tests/knowledge/components/test_results.py",
    "analysis/common/scoring.py",
    "analysis/knowledge/banks/primary.py",
    "analysis/knowledge/banks/secondary.py",
    "analysis/knowledge/banks/severity.py",
]


def forbidden(*args, **kwargs):
    raise AssertionError("Offline component scoring only")


def check_input_scope(inp, subset, dimension):
    a.require(
        subset in s.ENDPOINTS and s.ENDPOINTS[subset][0] == dimension,
        "Endpoint scoring dimension differs",
    )
    alias = {
        "en_dev_eval_sq3_types": "hate_type_primary",
        "en_dev_eval_types": "hate_type_secondary",
    }
    a.require(
        inp["scopes"] == [{"dimension": alias.get(subset, dimension), "subset": subset}],
        "Input source scope differs",
    )


def selected_stages(schedule, subset):
    stages = [st for st in schedule["stages"] if any((j["subset"] == subset for j in st["jobs"]))]
    a.require(bool(stages), "Unknown/empty endpoint")
    return stages


def setting_keys(config):
    return [(name, r) for name in config["conditions"] for r in config["repeats"]]


def check_settings(settings, ids, config):
    a.require(len(ids) == len(set(ids)), "Duplicate source identity")
    a.require(set(settings) == set(setting_keys(config)), "Missing/extra condition or repeat")
    for key, rows in settings.items():
        a.require(set(rows) == set(ids), "Incomplete/extra endpoint roster: " + str(key))


def summarize(values):
    out = primary.summarize_three([float(v) for v in values], "decoder_repeat")
    out["range"] = out["max"] - out["min"]
    return out


def resample(contribs, names, config):
    keys = [(name, r) for name in names for r in config["repeats"]]
    cc = [contribs[k] for k in keys]
    first = cc[0]
    common = np.logical_and.reduce([c["valid"] for c in cc])
    n = int(common.sum())
    idx = first["macro_indices"]
    a.require(
        n > 0 and bool(idx), "No common-valid texts or supported labels; special review required"
    )
    values = np.stack([c["counts"][common] for c in cc])
    point = s.macro(values.sum(axis=1), idx)
    draws = config["bootstrap_draws"]
    seed = config["bootstrap_seed"]
    bootstrap = np.empty((draws, len(keys)), dtype=float)
    rng = np.random.default_rng(seed)
    for i in range(draws):
        pick = rng.integers(0, n, n)
        bootstrap[i] = s.macro(values[:, pick].sum(axis=1), idx)
    check = np.empty_like(bootstrap)
    rng = np.random.default_rng(seed)
    for i in range(draws):
        freq = np.bincount(rng.integers(0, n, n), minlength=n)
        totals = np.einsum("sijk,i->sjk", values, freq)
        check[i] = primary.independent_f1(totals, idx)
    a.require(np.allclose(bootstrap, check, rtol=0, atol=1e-13), "Independent bootstrap differs")
    a.require(
        np.allclose(point, primary.independent_f1(values.sum(axis=1), idx), rtol=0, atol=1e-13),
        "Independent point F1 differs",
    )
    conditions = {}
    means = {}
    boot_means = {}
    check_means = {}
    bounds = {}
    for name in names:
        indices = [keys.index((name, r)) for r in config["repeats"]]
        bounds[name] = np.mean(
            [s.score(contribs[name, r])["invalid_inclusive_bounds"] for r in config["repeats"]],
            axis=0,
        )
        conditions[name] = {
            "common_valid_summary": summarize(point[indices]),
            "mean_full_roster_failure_bounds": bounds[name].tolist(),
        }
        means[name] = float(point[indices].mean())
        boot_means[name] = bootstrap[:, indices].mean(axis=1)
        check_means[name] = check[:, indices].mean(axis=1)
    report = {
        "n_common_valid": n,
        "common_valid_ids": [rid for rid, keep in zip(first["ids"], common) if keep],
        "excluded_from_common_valid": [rid for rid, keep in zip(first["ids"], common) if not keep],
        "settings": len(keys),
        "conditions": conditions,
        "contrasts": {},
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "independent_frequency_weighted_bootstrap_verified": True,
        "independent_bootstrap_setting_values": draws * len(keys),
    }
    return (report, means, boot_means, check_means, bounds)


def compare(contribs, config, *, focal):
    keys = setting_keys(config)
    a.require(
        config["conditions"] == FACTORIAL + ["reference_rag"] and config["repeats"] == [0, 1, 2],
        "Wrong component settings",
    )
    a.require(
        len(keys) == 27 and set(contribs) == set(keys),
        "Exactly 27 component/repeat settings required",
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
    cache = {
        "factorial": resample(contribs, FACTORIAL, config),
        "bridge": resample(contribs, ["edg", "reference_rag"], config),
    }
    a.require(
        config["focal_family_size"] == 3
        and len(config["contrasts"]["focal_leave_one_out"]) == 3
        and np.isclose(config["focal_family_confidence"], 1 - 0.05 / 3),
        "Wrong focal family contract",
    )
    for group, contrasts in config["contrasts"].items():
        for name, weights in contrasts.items():
            mask = "bridge" if name == BRIDGE else "factorial"
            report, means, boot, check, bounds = cache[mask]
            a.require(
                set(weights) <= set(means) and abs(sum(weights.values())) < 1e-12,
                "Invalid contrast weights/mask",
            )
            delta = sum((w * boot[arm] for arm, w in weights.items()))
            reference = sum((w * check[arm] for arm, w in weights.items()))
            ci = np.percentile(delta, [2.5, 97.5]).tolist()
            a.require(
                np.allclose(ci, np.percentile(reference, [2.5, 97.5]), rtol=0, atol=1e-12),
                "Independent interval differs",
            )
            adjusted = focal and group == "focal_leave_one_out"
            confidence = config["focal_family_confidence"] if adjusted else None
            family_q = [50 * (1 - confidence), 100 - 50 * (1 - confidence)] if adjusted else None
            family_ci = np.percentile(delta, family_q).tolist() if adjusted else None
            if adjusted:
                a.require(
                    np.allclose(family_ci, np.percentile(reference, family_q), rtol=0, atol=1e-12),
                    "Independent family interval differs",
                )
            low = sum((w * bounds[arm][0 if w >= 0 else 1] for arm, w in weights.items()))
            high = sum((w * bounds[arm][1 if w >= 0 else 0] for arm, w in weights.items()))
            report["contrasts"][name] = {
                "group": group,
                "weights": weights,
                "mask": mask,
                "n_common_valid": report["n_common_valid"],
                "mean_difference": sum((w * means[arm] for arm, w in weights.items())),
                "ci95": ci,
                "family_ci": family_ci,
                "family_confidence": confidence,
                "family_percentiles": family_q,
                "invalid_inclusive_difference_bounds": [float(low), float(high)],
            }
    out = {name: values[0] for name, values in cache.items()}
    out["factorial"][
        "mask"
    ] = "Valid in all eight factorial arms and three repeats (24 settings), excluding the reference."
    out["bridge"][
        "mask"
    ] = "Valid in edg and reference_rag over three repeats (six settings), independently of the other seven arms."
    return out


def prepare(subset):
    folder = DEST / subset
    plan_path = folder / "plan.json"
    a.require(not plan_path.exists(), "Endpoint scoring plan already frozen")
    p = b.validate()
    config = a.read(BASE / "analysis_plan.json")
    schedule = a.read(BASE / "schedule.json")
    a.require(
        subset in config["endpoints"] and config["no_partial_scoring"],
        "Unknown endpoint or partial scoring allowed",
    )
    tests_path = DEST / "development/tests_passed.json"
    tests = a.read(tests_path)
    a.require(
        tests["passed"] and tests["environment"] == b.core.environment(),
        "Passing scoring tests required",
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
            "Stage review binding changed",
        )
        a.require(
            ev["stage"] == st["id"]
            and ev["semantic_replay_items"] == st["items"]
            and (not ev["actual_audit_API_calls"]),
            "Incomplete or wrong stage audit",
        )
        hashes.update(ev["verified_input_hashes"])
        for path in (rp, ev_path, cp):
            hashes[b.rel(path)] = a.digest(path)
        bindings.append({"stage": st["id"], "review": b.rel(rp), "evidence": b.rel(ev_path)})
    for name, h in hashes.items():
        a.require(a.digest(ROOT / name) == h, "Changed scoring input: " + name)
    a.save_new(
        plan_path,
        {
            "created_utc": a.utc(),
            "status": "prepared_before_endpoint_scoring",
            "subset": subset,
            "hashes": hashes,
            "stage_audits": bindings,
            "analysis_plan_sha256": a.digest(BASE / "analysis_plan.json"),
            "tests_sha256": a.digest(tests_path),
            "scope": "Only this complete DEV endpoint. Separate 24-setting factorial and six-setting bridge masks; own-valid scores and full-roster failure bounds. No old repeat pooling or automatic promotion.",
        },
    )
    print("Prepared component scoring plan for " + subset + ": " + a.digest(plan_path), flush=True)


def validate(subset):
    path = DEST / subset / "plan.json"
    p = a.read(path)
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
    a.require(
        not (folder / "assembled").exists() and (not (folder / "results.json").exists()),
        "Preserve completed or partial endpoint publication",
    )
    rows = pd.read_parquet(ROOT / f"data/processed/{subset}.parquet").to_dict("records")
    primary.keyed(rows)
    ep = s.roster(
        subset, rows, yaml.safe_load((ROOT / "config/taxonomy.yaml").read_text(encoding="utf8"))
    )
    frozen = config["endpoints"][subset]
    alias = {
        "en_dev_eval_sq3_types": "hate_type_primary",
        "en_dev_eval_types": "hate_type_secondary",
    }
    expected = {
        **ep,
        "inherited_core_role": ep["role"],
        "role": frozen["role"],
        "source_scope": alias.get(subset, ep["dimension"]),
    }
    a.require(expected == frozen, "Frozen endpoint, support or label mask differs")
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
            for path, h in (
                (cp, completion["item_receipt_hashes"][b.rel(cp)]),
                (rp, ev["verified_output_hashes"][b.rel(rp)]),
            ):
                a.require(a.digest(path) == h, "Audited prediction changed")
                hashes[b.rel(path)] = h
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
    paired = compare(contributions, config, focal=subset == config["focal_endpoint"])
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
            "settings": 27,
            "predictions": 27 * len(rows),
            "independent_setting_metrics_and_bounds": 27,
            "independent_bootstrap_setting_values": sum(
                (x["independent_bootstrap_setting_values"] for x in paired.values())
            ),
            "contrasts": sum((len(x["contrasts"]) for x in paired.values())),
        },
        "new_API_calls": 0,
        "gold_rosters_scored": [subset],
        "limits": [
            config["limits"],
            config["bootstrap"],
            config["multiple_comparisons"],
            config["validity"],
            "Full-roster failure bounds for severity use conservative bitwise relaxation, not necessarily attainable categorical assignments.",
        ],
    }
    a.save_new(folder / "results.json", report)
    date = report["created_utc"][:10].replace("-", "")
    evidence = (
        ROOT / f"Thesis_Log/astra_result/evidence/sq1_components_{subset}_results_{date}_v1.json"
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
                "conditions": {
                    k: {q: v[q] for q in ("own_valid_summary", "null_predictions")}
                    for k, v in conditions.items()
                },
                "paired_comparison": paired,
                "evidence": b.rel(evidence),
                "sha256": a.digest(evidence),
            },
            indent=2,
        ),
        flush=True,
    )


def tests():
    import unittest
    from tests.knowledge.components import test_results as fixtures

    suite = unittest.defaultTestLoader.loadTestsFromModule(fixtures)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    a.require(result.wasSuccessful(), "Component scoring tests failed")
    path = DEST / "development/tests_passed.json"
    a.save_new(
        path,
        {
            "created_utc": a.utc(),
            "passed": True,
            "tests": result.testsRun,
            "environment": b.core.environment(),
            "source_hashes": {name: a.digest(ROOT / name) for name in SOURCE_NAMES},
            "analysis_plan_sha256": a.digest(BASE / "analysis_plan.json"),
        },
    )
    print("Tests receipt: " + b.rel(path), flush=True)


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

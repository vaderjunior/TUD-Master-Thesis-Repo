"""Score the primary full and compact bank comparisons."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import socket
from unittest.mock import patch
import numpy as np
from analysis.common import batch as b

a = b.a
s = b.s
ROOT = b.ROOT
CORE = b.CORE
BASE = b.BASE
DEST = CORE / "primary_results_v1"
PLAN = DEST / "plan.json"
ENDPOINTS = ("en_dev_eval_main", "de_legal_dev_eval")
AUDIT_VERSIONS = ("v1", "v2", "v3", "v3")
SOURCE = ROOT / "analysis/knowledge/banks/primary.py"
TEST = ROOT / "tests/knowledge/banks/test_primary.py"


def forbidden(*args, **kwargs):
    raise AssertionError("Offline primary scoring only")


def keyed(records, field="id"):
    result = {}
    for row in records:
        key = str(row[field])
        a.require(key not in result, "Duplicate identity: " + key)
        result[key] = row
    return result


def independent_counts(rows, outputs, endpoint):
    """Direct label comparisons, independently of scorer truth/bits/counts."""
    labels = endpoint["all_labels"]
    dimension = endpoint["dimension"]
    a.require(set(outputs) == {str(r["id"]) for r in rows}, "Incomplete/extra output identities")
    ids = []
    ys = []
    valid = []
    cs = []
    predictions = []
    for row in rows:
        gold = row["gate"] if dimension == "hate" else row["legal"]
        if isinstance(gold, float) and np.isnan(gold):
            gold = None
        if gold is None:
            continue
        if hasattr(gold, "tolist"):
            gold = gold.tolist()
        if dimension == "hate":
            a.require(type(gold) is bool, "Nonboolean primary gate gold")
            truth = {"hate" if gold else "not_hate"}
        else:
            truth = set(gold)
        out = outputs[str(row["id"])]
        ok = out["result"] is not None
        a.require((out["n_valid"] > 0) == ok, "Invalidity stamp mismatch")
        positive = set()
        if ok:
            value = out["result"][dimension]
            if dimension == "hate":
                a.require(type(value) is bool, "Nonboolean predicted gate")
                positive = {"hate" if value else "not_hate"}
            else:
                positive = set(value)
        a.require(truth <= set(labels) and positive <= set(labels), "Unknown primary label")
        y = [int(label in truth) for label in labels]
        x = [int(label in positive) for label in labels]
        counts = [
            [int(yy and xx), int(not yy and xx), int(yy and (not xx))] if ok else [0, 0, 0]
            for yy, xx in zip(y, x)
        ]
        ids.append(str(row["id"]))
        ys.append(y)
        valid.append(ok)
        cs.append(counts)
        predictions.append(x)
    return {
        "ids": ids,
        "gold": np.array(ys, dtype=np.int64),
        "valid": np.array(valid, dtype=bool),
        "counts": np.array(cs, dtype=np.int64),
        "predictions": np.array(predictions, dtype=np.int64),
        "labels": labels,
        "macro_indices": [labels.index(x) for x in endpoint["macro_labels"]],
    }


def independent_f1(counts, indices):
    cc = np.asarray(counts)
    den = 2 * cc[..., 0] + cc[..., 1] + cc[..., 2]
    values = np.zeros(den.shape, dtype=float)
    np.divide(2 * cc[..., 0], den, out=values, where=den > 0)
    return np.mean(values[..., indices], axis=-1)


def checked_score(rows, outputs, endpoint):
    c = s.contributions(rows, outputs, endpoint)
    d = independent_counts(rows, outputs, endpoint)
    a.require(c["ids"] == d["ids"] == endpoint["scorable_ids"], "Scorable order differs")
    for field in ("gold", "valid", "counts"):
        a.require(np.array_equal(c[field], d[field]), "Independent " + field + " differs")
    result = s.score(c)
    base = d["counts"].sum(axis=0)
    low = base.copy()
    high = base.copy()
    for gold, ok in zip(d["gold"], d["valid"]):
        if not ok:
            for index, y in enumerate(gold):
                low[index, 2 if y else 1] += 1
                if y:
                    high[index, 0] += 1
    expected = float(independent_f1(base, d["macro_indices"])) if d["valid"].any() else None
    a.require(result["macro_f1_valid_only"] == expected, "Independent macro F1 differs")
    a.require(
        np.allclose(
            result["invalid_inclusive_bounds"],
            [independent_f1(low, d["macro_indices"]), independent_f1(high, d["macro_indices"])],
            atol=1e-14,
            rtol=0,
        ),
        "Failure bounds differ",
    )
    result["per_label"] = {
        label: {
            "tp": int(base[i, 0]),
            "fp": int(base[i, 1]),
            "fn": int(base[i, 2]),
            "gold_support_full": int(d["gold"][:, i].sum()),
            "gold_support_valid": int(d["gold"][d["valid"], i].sum()),
            "f1": float(independent_f1(base, [i])),
        }
        for i, label in enumerate(d["labels"])
    }
    if endpoint["dimension"] == "hate":
        sources = defaultdict(Counter)
        for row in rows:
            rid = str(row["id"])
            i = d["ids"].index(rid)
            source = row.get("source", row.get("dataset"))
            if not isinstance(source, str) or not source:
                source = rid.split("-", 1)[0]
            q = sources[source]
            q["rows"] += 1
            q["gold_hate"] += int(d["gold"][i, 1])
            q["valid"] += int(d["valid"][i])
            q["invalid"] += int(not d["valid"][i])
            q["tp"] += int(d["counts"][i, 1, 0])
            q["fp"] += int(d["counts"][i, 1, 1])
            q["fn"] += int(d["counts"][i, 1, 2])
        result["gate_by_source"] = {k: dict(v) for k, v in sorted(sources.items())}
    return (c, result)


def summarize_three(values, kind):
    a.require(len(values) == 3, "Expected exactly three replicates")
    return {
        "values": values,
        "mean": float(np.mean(values)),
        "sample_sd": float(np.std(values, ddof=1)),
        "min": float(min(values)),
        "max": float(max(values)),
        "replicate_kind": kind,
    }


def checked_contrast(left, right, primary):
    """Recompute paired bootstrap with frequency-weighted confusion matrices."""
    result = s.contrast(left, right, draws=10000, seed=20260915, primary=primary)
    both = left + right
    common = np.logical_and.reduce([c["valid"] for c in both])
    n = int(common.sum())
    a.require(n == result["n_common_valid"] and n > 0, "Unexpected common-valid roster")
    expected_ids = [rid for rid, ok in zip(both[0]["ids"], common) if ok]
    a.require(expected_ids == result["common_valid_ids"], "Common-valid IDs differ")
    counts = np.stack([c["counts"][common] for c in both])
    idx = both[0]["macro_indices"]
    values = independent_f1(counts.sum(axis=1), idx)
    point = float(values[: len(left)].mean() - values[len(left) :].mean())
    a.require(
        np.isclose(point, result["mean_difference"], atol=1e-14, rtol=0),
        "Independent contrast point differs",
    )
    rng = np.random.default_rng(20260915)
    boot = []
    for _ in range(10000):
        frequencies = np.bincount(rng.integers(0, n, n), minlength=n)
        totals = np.einsum("sijk,i->sjk", counts, frequencies)
        scores = independent_f1(totals, idx)
        boot.append(float(scores[: len(left)].mean() - scores[len(left) :].mean()))
    a.require(
        np.allclose(np.percentile(boot, [2.5, 97.5]), result["ci95"], atol=1e-12, rtol=0),
        "Independent bootstrap 95% differs",
    )
    if primary:
        a.require(
            np.allclose(
                np.percentile(boot, [0.625, 99.375]), result["family_ci"], atol=1e-12, rtol=0
            ),
            "Independent family interval differs",
        )
    result.update(
        {
            "left_mean_on_common_valid": float(values[: len(left)].mean()),
            "right_mean_on_common_valid": float(values[len(left) :].mean()),
            "excluded_from_common_valid": [
                rid for rid, ok in zip(both[0]["ids"], common) if not ok
            ],
            "independent_frequency_weighted_bootstrap_verified": True,
        }
    )
    return result


def prepare():
    a.require(not PLAN.exists(), "Preserve existing primary assembly plan")
    p = b.validate()
    schedule = a.read(BASE / "schedule.json")
    hashes = {
        **p["input_hashes"],
        **p["source_hashes"],
        **p["generated_hashes"],
        b.rel(b.PLAN): a.digest(b.PLAN),
        b.rel(SOURCE): a.digest(SOURCE),
        b.rel(TEST): a.digest(TEST),
    }
    audits = []
    for stage, version in zip(schedule["stages"][:4], AUDIT_VERSIONS):
        a.require(stage["phase"] == "primary", "Wrong stage phase")
        receipt_path = BASE / "audits" / f"{stage['id']}_{version}" / "complete.json"
        receipt = a.read(receipt_path)
        evidence_path = ROOT / receipt["evidence"]
        ev = a.read(evidence_path)
        a.require(a.digest(evidence_path) == receipt["evidence_sha256"], "Stage evidence changed")
        a.require(
            a.digest(BASE / "stages" / stage["id"] / "complete.json")
            == receipt["stage_complete_sha256"],
            "Stage completion changed",
        )
        a.require(
            ev["stage"] == stage["id"]
            and ev["actual_audit_API_calls"] == 0
            and (ev["semantic_replay_items"] == stage["items"]),
            "Wrong stage audit",
        )
        hashes.update(ev["verified_input_hashes"])
        hashes[b.rel(receipt_path)] = a.digest(receipt_path)
        hashes[b.rel(evidence_path)] = a.digest(evidence_path)
        audits.append(
            {"stage": stage["id"], "receipt": b.rel(receipt_path), "evidence": b.rel(evidence_path)}
        )
    for name, h in hashes.items():
        a.require(a.digest(ROOT / name) == h, "Input/source changed: " + name)
    plan = {
        "created_utc": a.utc(),
        "status": "primary_assembly_prepared_no_new_scoring",
        "hashes": hashes,
        "stage_audits": audits,
        "endpoints": list(ENDPOINTS),
        "scoring_contract": s.contract(),
        "primary_contrasts": s.contract()["primary_contrasts"],
        "descriptive_comparisons": {
            "en_dev_eval_main": ["zero_shot", "encoder_full", "encoder_kb"],
            "de_legal_dev_eval": ["zero_shot", "fewshot_uniform", "encoder_full"],
        },
        "join": "Exact (endpoint, condition, repeat, input identity); require all scheduled rows. Preserve original result/null. Encoder seeds 0/1/2 adapt one available structured prediction as n_valid=1, not decoder votes.",
        "summaries": "Three decoder repeats per condition; three encoder training seeds separately. Few-shot: three repeats within each example set, mean across three set means and sample SD across those set means; no pooled-nine SD.",
        "verification": "Direct independent label/count/F1/bounds reconstruction for every setting; independent frequency-weighted bootstrap using the same 10000 item resamples for every contrast.",
        "scope": "Only two completed primary DEV rosters. No TEST/benchmark, secondary scoring, API/training/retrieval, Priority 3 or LaTeX work.",
    }
    a.save_new(PLAN, plan)
    print("Prepared primary assembly plan: " + a.digest(PLAN), flush=True)


def validate_plan():
    p = a.read(PLAN)
    a.require(p["scoring_contract"] == s.contract(), "Frozen scorer contract changed")
    for name, h in p["hashes"].items():
        a.require(a.digest(ROOT / name) == h, "Primary input/source changed: " + name)
    b.validate()
    return p


def run():
    import pandas as pd
    import yaml

    p = validate_plan()
    a.require(
        not (DEST / "complete.json").exists() and (not (DEST / "results.json").exists()),
        "Preserve completed primary scoring",
    )
    a.require(
        not (DEST / "assembled").exists(), "Partial publication needs review; do not overwrite"
    )
    frozen = a.read(BASE / "scoring_plan.json")
    tax = yaml.safe_load((ROOT / "config/taxonomy.yaml").read_text(encoding="utf8"))
    dev = keyed(b.rows(CORE / "dev_inputs.jsonl"), "key")
    rows = {}
    endpoints = {}
    settings = defaultdict(dict)
    origins = {}
    item_hashes = {}
    stage_totals = []
    for subset in ENDPOINTS:
        rows[subset] = pd.read_parquet(ROOT / f"data/processed/{subset}.parquet").to_dict("records")
        keyed(rows[subset])
        ep = s.roster(subset, rows[subset], tax)
        a.require(ep == frozen["endpoints"][subset], "Frozen gold roster changed")
        endpoints[subset] = ep
    schedule = a.read(BASE / "schedule.json")
    plan_sha = a.digest(b.PLAN)
    all_jobs = set()
    for stage, binding in zip(schedule["stages"][:4], p["stage_audits"]):
        ev = a.read(ROOT / binding["evidence"])
        completion = a.read(BASE / "stages" / stage["id"] / "complete.json")
        stage_totals.append(
            {
                "stage": stage["id"],
                "counts": ev["counts"],
                "accounting": ev["accounting"],
                "http_statuses": ev["http_statuses"],
                "observed_usage_sums": ev["observed_usage_sums"],
            }
        )
        for j in stage["jobs"]:
            identity = {
                "experiment_sha256": plan_sha,
                "condition": j["condition"],
                "repeat": j["repeat"],
                "key": j["key"],
            }
            d = a.storage_path(BASE / "live" / "items" / a.signature(identity))
            cp = d / "complete.json"
            rp = d / "result.json"
            for path, h in (
                (cp, completion["item_receipt_hashes"][b.rel(cp)]),
                (rp, ev["verified_output_hashes"][b.rel(rp)]),
            ):
                a.require(a.digest(path) == h, "Audited item changed")
                item_hashes[b.rel(path)] = h
            r = a.read(rp)
            a.require(
                r["identity"] == identity and r["n_valid"] in (0, 1, 2, 3),
                "Result identity/votes differ",
            )
            row = dev[j["key"]]
            a.require(row["scopes"][0]["subset"] == j["subset"], "Schedule endpoint differs")
            ident = (j["subset"], j["condition"], j["repeat"])
            rid = row["item_id"]
            a.require((ident, rid) not in all_jobs, "Duplicated scheduled prediction")
            all_jobs.add((ident, rid))
            settings[ident][rid] = {"result": r["result"], "n_valid": r["n_valid"]}
            origins[ident, rid] = {
                "key": j["key"],
                "source": b.rel(rp),
                "source_sha256": item_hashes[b.rel(rp)],
                "kind": "qwen_decoder_repeat",
            }
        print("Verified assembly identities/results: " + stage["id"], flush=True)
    a.require(len(all_jobs) == 10107, "Wrong complete primary prediction count")
    for subset in ENDPOINTS:
        language = "en" if subset.startswith("en_") else "de"
        expected = {str(r["id"]) for r in rows[subset]}
        arms = ("full", "kb") if language == "en" else ("full",)
        for arm in arms:
            for seed in range(3):
                path = CORE / (
                    f"encoder_results_v5/en_{arm}_seed{seed}_assembled.jsonl"
                    if language == "en"
                    else f"encoder_audit_v1/de_seed{seed}_de_legal_dev_eval_assembled.jsonl"
                )
                records = keyed(b.rows(path), "key")
                ident = (subset, "encoder_" + arm, seed)
                path_sha = a.digest(path)
                a.require(path_sha == p["hashes"][b.rel(path)], "Encoder artifact changed")
                for rid in [str(r["id"]) for r in rows[subset]]:
                    key = language + "/" + rid
                    r = records[key]
                    a.require(
                        r["item_id"] == rid and r["seed"] == seed and (r["result"] is not None),
                        "Encoder identity/validity differs",
                    )
                    if language == "en":
                        a.require(
                            r["arm"] == arm and r["scopes"] == dev[key]["scopes"],
                            "Encoder arm/scope mismatch",
                        )
                        a.require(
                            r["text_sha256"]
                            == hashlib.sha256(dev[key]["text"].encode()).hexdigest(),
                            "Encoder text mismatch",
                        )
                    else:
                        a.require(r["subset"] == subset, "German encoder scope mismatch")
                    settings[ident][rid] = {"result": r["result"], "n_valid": 1}
                    origins[ident, rid] = {
                        "key": key,
                        "source": b.rel(path),
                        "source_sha256": path_sha,
                        "kind": "encoder_training_seed",
                    }
                a.require(set(settings[ident]) == expected, "Encoder roster mismatch")
    analyses = {}
    contrib = {}
    for subset in ENDPOINTS:
        names = sorted({condition for ep, condition, rep in settings if ep == subset})
        conditions = {}
        expected = {
            "rag_full",
            "rag_small",
            "zero_shot",
            "fewshot_full_s42",
            "fewshot_full_s43",
            "fewshot_full_s44",
            "encoder_full",
        }
        expected.update(
            {"encoder_kb"}
            if subset.startswith("en_")
            else {
                "fewshot_eligible_legal_s42",
                "fewshot_eligible_legal_s43",
                "fewshot_eligible_legal_s44",
            }
        )
        a.require(set(names) == expected, "Unexpected comparison conditions")
        for condition in names:
            runs = []
            for rep in range(3):
                ident = (subset, condition, rep)
                c, score = checked_score(rows[subset], settings[ident], endpoints[subset])
                contrib[ident] = c
                runs.append({"replicate": rep, "score": score})
            kind = "training_seed" if condition.startswith("encoder") else "decoder_repeat"
            summary = summarize_three([r["score"]["macro_f1_valid_only"] for r in runs], kind)
            conditions[condition] = {
                "runs": runs,
                "summary": summary,
                "invalid_predictions": sum((r["score"]["n_invalid"] for r in runs)),
                "mean_invalid_inclusive_bounds": np.mean(
                    [r["score"]["invalid_inclusive_bounds"] for r in runs], axis=0
                ).tolist(),
                "mean_per_label_f1": {
                    label: float(np.mean([r["score"]["per_label"][label]["f1"] for r in runs]))
                    for label in endpoints[subset]["all_labels"]
                },
            }

        def group(prefix):
            names = [prefix + str(seed) for seed in (42, 43, 44)]
            means = [conditions[n]["summary"]["mean"] for n in names]
            return {
                "conditions": names,
                "set_summaries": {n: conditions[n]["summary"] for n in names},
                "across_set_means": summarize_three(means, "example_set_mean"),
                "mean_invalid_inclusive_bounds": np.mean(
                    [conditions[n]["mean_invalid_inclusive_bounds"] for n in names], axis=0
                ).tolist(),
                "invalid_predictions": sum((conditions[n]["invalid_predictions"] for n in names)),
                "mean_per_label_f1": {
                    label: float(
                        np.mean([conditions[n]["mean_per_label_f1"][label] for n in names])
                    )
                    for label in endpoints[subset]["all_labels"]
                },
            }

        groups = {"fewshot_uniform": group("fewshot_full_s")}
        groups["fewshot_eligible"] = (
            groups["fewshot_uniform"]
            if subset.startswith("en_")
            else group("fewshot_eligible_legal_s")
        )
        analyses[subset] = {
            "endpoint": endpoints[subset],
            "conditions": conditions,
            "fewshot_groups": groups,
            "english_eligible_aliases_uniform": subset.startswith("en_"),
            "contrasts": {},
        }
    for subset in ENDPOINTS:
        info = analyses[subset]

        def selected(name):
            names = (
                info["fewshot_groups"][name]["conditions"]
                if name.startswith("fewshot_") and name in info["fewshot_groups"]
                else [name]
            )
            return [contrib[subset, n, r] for n in names for r in range(3)]

        comparisons = [("rag_small", True), ("fewshot_eligible", True)] + [
            (name, False) for name in p["descriptive_comparisons"][subset]
        ]
        for other, primary in comparisons:
            result = checked_contrast(selected("rag_full"), selected(other), primary)
            result.update({"left": "rag_full", "right": other, "primary": primary})
            contrast_name = (
                "rag_full_minus_eligible_fewshot_mean"
                if other == "fewshot_eligible"
                else "rag_full_minus_" + other
            )
            info["contrasts"][contrast_name] = result
            print(subset + " / full minus " + other + ": independently checked", flush=True)
    declared = {(x["endpoint"], x["contrast"]) for x in p["primary_contrasts"]}
    actual = {
        (ep, name)
        for ep, info in analyses.items()
        for name, c in info["contrasts"].items()
        if c["primary"]
    }
    a.require(actual == declared, "Primary contrast family differs from the frozen plan")
    validate_plan()
    for name, h in item_hashes.items():
        a.require(a.digest(ROOT / name) == h, "Selected item changed during scoring")
    assembled = {}
    for ident, outputs in sorted(settings.items()):
        subset, condition, rep = ident
        path = DEST / "assembled" / subset / f"{condition}_rep{rep}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf8", newline="\n") as f:
            for row in rows[subset]:
                rid = str(row["id"])
                f.write(
                    json.dumps(
                        {
                            "id": rid,
                            "condition": condition,
                            "replicate": rep,
                            **outputs[rid],
                            **origins[ident, rid],
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
        assembled[b.rel(path)] = {"sha256": a.digest(path), "rows": len(outputs)}
    report = {
        "created_utc": a.utc(),
        "status": "complete_primary_results_independently_verified",
        "plan_sha256": a.digest(PLAN),
        "source_sha256": a.digest(SOURCE),
        "endpoints": analyses,
        "stage_operational_totals": stage_totals,
        "verified_primary_item_hashes": item_hashes,
        "verified_input_hashes": p["hashes"],
        "assembled": assembled,
        "verification": {
            "independent_setting_counts_f1_bounds": len(settings),
            "independent_bootstrap_contrasts": sum(
                (len(x["contrasts"]) for x in analyses.values())
            ),
            "bootstrap_draws_per_contrast": 10000,
            "bootstrap_seed": 20260915,
            "qwen_predictions": len(all_jobs),
            "encoder_predictions": sum(
                (len(v) for k, v in settings.items() if k[1].startswith("encoder"))
            ),
        },
        "scoring_contract": s.contract(),
        "actual_API_calls": 0,
        "primary_gold_scoring": True,
        "other_rosters_scored": False,
        "limits": [
            "Post-selection DEV comparison, not untouched TEST confirmation.",
            "Intervals condition on saved decoder repeats, three example sets and fitted training seeds; shared-text/source dependence and total selection/training/generation variance are not covered.",
            "RAG versus few-shot is a system-package comparison, including rules/prompt wording; it does not isolate a causal retrieval effect.",
            "Valid-only per-run summaries and common-valid contrasts use different masks when nulls occur; do not subtract table means as the paired estimate.",
            "Failure bounds concern the whole roster and are sensitivity bounds, not uncertainty intervals or imputed predictions.",
            "Stage 02 includes eight administrative unknown outcomes; remote usage/duration are unknown, and aggregate observed cost is incomplete.",
        ],
        "next": "Pause for primary-result review. No stage 05, Priority 3, TEST/benchmark or LaTeX work started.",
    }
    a.save_new(DEST / "results.json", report)
    date = report["created_utc"][:10].replace("-", "")
    out = ROOT / f"Thesis_Log/astra_result/evidence/core_primary_results_{date}_v1.json"
    a.save_new(out, report)
    a.save_new(
        DEST / "complete.json",
        {
            "status": report["status"],
            "plan_sha256": a.digest(PLAN),
            "results_sha256": a.digest(DEST / "results.json"),
            "evidence": b.rel(out),
            "evidence_sha256": a.digest(out),
            "assembled": assembled,
        },
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "evidence": b.rel(out),
                "sha256": a.digest(out),
                "verification": report["verification"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(b, "live_client", forbidden):
        prepare() if args.mode == "prepare" else run()

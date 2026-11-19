"""Score secondary bank outcomes and check their source masks."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import socket
from unittest.mock import patch
import numpy as np
from analysis.knowledge.banks import primary as shared

b = shared.b
a = b.a
s = b.s
ROOT = b.ROOT
CORE = b.CORE
BASE = b.BASE
SOURCE = ROOT / "analysis/knowledge/banks/secondary.py"
TEST = ROOT / "tests/knowledge/banks/test_secondary.py"
AMENDMENT = (
    ROOT / "Thesis_Log/astra_result/evidence/core_secondary_types_scope_failure_20260919_v1.json"
)
ALLOWED = {
    "en_dev_eval_targets": ("target_group", "en"),
    "en_dev_eval_sq3_types": ("hate_type", "en"),
    "en_dev_eval_types": ("hate_type", "en"),
    "de_dev_eval": ("hate", "de"),
}
INPUT_SCOPE_DIMENSIONS = {
    "en_dev_eval_targets": "target_group",
    "en_dev_eval_sq3_types": "hate_type_primary",
    "en_dev_eval_types": "hate_type_secondary",
    "de_dev_eval": "hate",
}


def forbidden(*args, **kwargs):
    raise AssertionError("Offline secondary scoring only")


def destination(subset):
    a.require(subset in ALLOWED, "Endpoint outside remaining secondary/sensitivity scope")
    return CORE / "secondary_results_v4" / subset


def checked_input_scope(inp, subset):
    """Require the exact frozen roster tag; never normalize arbitrary suffixes."""
    destination(subset)
    expected = [{"dimension": INPUT_SCOPE_DIMENSIONS[subset], "subset": subset}]
    a.require(inp["scopes"] == expected, "Wrong prepared input scope")


def conditions(subset):
    dimension, language = ALLOWED[subset]
    names = {"rag_full", "rag_small", "zero_shot", "encoder_full"}
    if language == "en":
        names.add("encoder_kb")
    names.update(("fewshot_full_s" + str(seed) for seed in (42, 43, 44)))
    names.update(("fewshot_eligible_" + dimension + "_s" + str(seed) for seed in (42, 43, 44)))
    return names


def selected_stages(schedule, subset):
    destination(subset)
    found = [st for st in schedule["stages"] if any((j["subset"] == subset for j in st["jobs"]))]
    a.require(
        found and all((st["phase"] in ("secondary", "sensitivity") for st in found)),
        "Wrong/missing stage phase",
    )
    return found


def validate_settings(settings, ids, subset):
    a.require(len(ids) == len(set(ids)), "Duplicate evaluation identity")
    a.require(
        set(settings) == {(name, rep) for name in conditions(subset) for rep in range(3)},
        "Missing/extra condition or repeat",
    )
    for ident, outputs in settings.items():
        a.require(set(outputs) == set(ids), "Incomplete/extra endpoint roster: " + str(ident))


def independent_counts(rows, outputs, endpoint):
    """Direct gold/predicted label membership, independent of scorer masking."""
    shared.keyed(rows)
    labels = endpoint["all_labels"]
    dim = endpoint["dimension"]
    a.require(dim in ("hate", "target_group", "hate_type"), "Unsupported independent dimension")
    a.require(
        set(outputs) == {str(row["id"]) for row in rows}, "Incomplete/extra output identities"
    )
    ids = []
    ys = []
    cs = []
    valid = []
    gate_missed = []
    for row in rows:
        gate = row.get("gate")
        if isinstance(gate, np.bool_):
            gate = bool(gate)
        if isinstance(gate, float) and np.isnan(gate):
            gate = None
        a.require(gate is None or type(gate) is bool, "Nonboolean gold gate")
        if dim == "hate":
            if gate is None:
                continue
            truth = {"hate" if gate else "not_hate"}
        else:
            if gate is not True:
                continue
            gold = row.get({"target_group": "target_groups", "hate_type": "hate_types"}[dim])
            if gold is None or (isinstance(gold, float) and np.isnan(gold)):
                continue
            if isinstance(gold, np.ndarray):
                gold = gold.tolist()
            a.require(isinstance(gold, list), "Multilabel gold is not a list")
            truth = set(gold)
        out = outputs[str(row["id"])]
        ok = out["result"] is not None
        positive = set()
        missed = False
        a.require(
            type(out["n_valid"]) is int
            and out["n_valid"] in (0, 1, 2, 3)
            and ((out["n_valid"] > 0) == ok),
            "Contradictory validity stamp",
        )
        if ok:
            pred = out["result"]
            a.require(type(pred["hate"]) is bool, "Nonboolean predicted gate")
            missed = gate is True and (not pred["hate"])
            if dim == "hate":
                positive = {"hate" if pred["hate"] else "not_hate"}
            else:
                a.require(isinstance(pred[dim], list), "Predicted labels are not a list")
                positive = set(pred[dim])
                a.require(
                    pred["hate"] or not positive, "Fine labels survived a false predicted hate gate"
                )
        a.require(truth <= set(labels) and positive <= set(labels), "Unknown gold/prediction label")
        y = [int(label in truth) for label in labels]
        counts = [
            (
                [
                    int(label in truth and label in positive),
                    int(label not in truth and label in positive),
                    int(label in truth and label not in positive),
                ]
                if ok
                else [0, 0, 0]
            )
            for label in labels
        ]
        ids.append(str(row["id"]))
        ys.append(y)
        cs.append(counts)
        valid.append(ok)
        gate_missed.append(missed)
    n = len(ids)
    k = len(labels)
    return {
        "ids": ids,
        "gold": np.asarray(ys, dtype=np.int64).reshape(n, k),
        "counts": np.asarray(cs, dtype=np.int64).reshape(n, k, 3),
        "valid": np.asarray(valid, dtype=bool),
        "gate_missed": np.asarray(gate_missed, dtype=bool),
        "labels": labels,
        "macro_indices": [labels.index(x) for x in endpoint["macro_labels"]],
    }


def checked_score(rows, outputs, endpoint):
    c = s.contributions(rows, outputs, endpoint)
    d = independent_counts(rows, outputs, endpoint)
    a.require(c["ids"] == d["ids"] == endpoint["scorable_ids"], "Scorable order differs")
    for key in ("gold", "valid", "counts"):
        a.require(np.array_equal(c[key], d[key]), "Independent " + key + " differs")
    result = s.score(c)
    base = d["counts"].sum(axis=0)
    low = base.copy()
    high = base.copy()
    for y, ok in zip(d["gold"], d["valid"]):
        if not ok:
            for i, positive in enumerate(y):
                low[i, 2 if positive else 1] += 1
                if positive:
                    high[i, 0] += 1
    expected = float(shared.independent_f1(base, d["macro_indices"])) if d["valid"].any() else None
    a.require(result["macro_f1_valid_only"] == expected, "Independent F1 differs")
    a.require(
        np.allclose(
            result["invalid_inclusive_bounds"],
            [
                shared.independent_f1(low, d["macro_indices"]),
                shared.independent_f1(high, d["macro_indices"]),
            ],
            atol=1e-14,
            rtol=0,
        ),
        "Independent bounds differ",
    )
    result["per_label"] = {
        label: {
            "tp": int(base[i, 0]),
            "fp": int(base[i, 1]),
            "fn": int(base[i, 2]),
            "gold_support_full": int(d["gold"][:, i].sum()),
            "gold_support_valid": int(d["gold"][d["valid"], i].sum()),
            "gate_suppressed_gold_positives": int(d["gold"][d["gate_missed"], i].sum()),
            "f1": float(shared.independent_f1(base, [i])),
        }
        for i, label in enumerate(d["labels"])
    }
    result["missed_hate_on_gold_positive_scorable_inputs"] = int(d["gate_missed"].sum())
    if endpoint["dimension"] == "hate":
        sources = defaultdict(Counter)
        lookup = shared.keyed(rows)
        i_hate = d["labels"].index("hate")
        for i, rid in enumerate(d["ids"]):
            row = lookup[rid]
            source = row.get("source", row.get("dataset"))
            if not isinstance(source, str) or not source:
                source = rid.split("-", 1)[0]
            q = sources[source]
            q["rows"] += 1
            q["gold_hate"] += int(d["gold"][i, i_hate])
            q["valid"] += int(d["valid"][i])
            q["invalid"] += int(not d["valid"][i])
            for j, key in enumerate(("tp", "fp", "fn")):
                q[key] += int(d["counts"][i, i_hate, j])
        result["gate_by_source"] = {k: dict(v) for k, v in sorted(sources.items())}
    return (c, result)


def checked_audit_binding(stage, receipt, ev):
    a.require(receipt["stage"] == stage["id"] == ev["stage"], "Wrong stage audit binding")
    a.require(
        receipt["status"] == ev["status"] == "completed_stage_verified_next_stage_ready",
        "Audit not complete",
    )
    a.require(
        ev["semantic_replay_items"] == ev["independent_field_vote_checks"] == stage["items"],
        "Audit omits scheduled predictions",
    )
    a.require(
        ev["actual_audit_API_calls"] == 0
        and ev["actual_gold_scoring"] is False
        and (ev["completed_stage_restart_new_POSTs"] == 0),
        "Unexpected audit side effects",
    )
    if stage["id"] == "stage_09_secondary":
        history = ev.get("execution_history", {})
        a.require(
            history.get("status") == "clean_two_attempt_history_verified"
            and history.get("completed_items_preserved") == 8
            and (history.get("resumed_new_items") == 2692)
            and (history.get("administrative_unknown_outcomes") == 0)
            and (
                history.get("interruption_evidence_sha256")
                == "f5e1ffd4bdafaf20fbf2c58bab7196b1fc69fb64d24a6ff8f5317490e2084c03"
            ),
            "Stage-09 history is not the reviewed clean resume",
        )


def prepare(subset):
    dest = destination(subset)
    a.require(not (dest / "plan.json").exists(), "Preserve existing assembly plan")
    p = b.validate()
    schedule = a.read(BASE / "schedule.json")
    stages = selected_stages(schedule, subset)
    endpoint = a.read(BASE / "scoring_plan.json")["endpoints"][subset]
    dev = shared.keyed(b.rows(CORE / "dev_inputs.jsonl"), "key")
    keys = {j["key"] for st in stages for j in st["jobs"] if j["subset"] == subset}
    for key in keys:
        checked_input_scope(dev[key], subset)
    a.require(len(keys) == endpoint["rows"], "Prepared input count differs from endpoint")
    a.require(
        endpoint["dimension"] == ALLOWED[subset][0]
        and endpoint["role"] in ("secondary", "sensitivity"),
        "Wrong endpoint role",
    )
    hashes = {
        **p["input_hashes"],
        **p["source_hashes"],
        **p["generated_hashes"],
        b.rel(b.PLAN): a.digest(b.PLAN),
    }
    for path in (SOURCE, TEST, shared.SOURCE, AMENDMENT):
        hashes[b.rel(path)] = a.digest(path)
    bindings = []
    for stage in stages:
        version = 4 if stage["id"] == "stage_09_secondary" else 3
        rp = BASE / "audits" / f"{stage['id']}_v{version}" / "complete.json"
        a.require(rp.exists(), "Contributing stage has no accepted completed audit: " + stage["id"])
        receipt = a.read(rp)
        ep = ROOT / receipt["evidence"]
        ev = a.read(ep)
        cp = BASE / "stages" / stage["id"] / "complete.json"
        a.require(
            a.digest(ep) == receipt["evidence_sha256"]
            and a.digest(cp) == receipt["stage_complete_sha256"],
            "Stage binding differs",
        )
        checked_audit_binding(stage, receipt, ev)
        hashes.update(ev["verified_input_hashes"])
        hashes[b.rel(rp)] = a.digest(rp)
        hashes[b.rel(ep)] = a.digest(ep)
        bindings.append({"stage": stage["id"], "receipt": b.rel(rp), "evidence": b.rel(ep)})
    for name, h in hashes.items():
        a.require(a.digest(ROOT / name) == h, "Input/source changed: " + name)
    comparisons = ["rag_small", "fewshot_eligible", "zero_shot", "fewshot_uniform", "encoder_full"]
    if ALLOWED[subset][1] == "en":
        comparisons.append("encoder_kb")
    plan = {
        "created_utc": a.utc(),
        "status": "secondary_endpoint_assembly_prepared_no_scoring",
        "subset": subset,
        "endpoint": endpoint,
        "hashes": hashes,
        "stage_audits": bindings,
        "audit_acceptance": "Unchanged v3 for uninterrupted stages; scoped v4 plus fixed history evidence for stage 09 only. No other recovery bypass.",
        "scoring_contract": s.contract(),
        "comparisons": comparisons,
        "input_scope_check": {
            "rows": len(keys),
            "prepared_dimension": INPUT_SCOPE_DIMENSIONS[subset],
            "metric_dimension": endpoint["dimension"],
            "scoring_role": endpoint["role"],
        },
        "scope_amendment_evidence": b.rel(AMENDMENT),
        "scope": "Only the named complete secondary/sensitivity DEV endpoint. No other gold scoring, live calls, training, retrieval, TEST/benchmark, adaptation or LaTeX.",
        "replication": "Three decoder repeats per condition; separate within-set and between-three-set means for few-shot. Encoder seeds are separate, saved predictions unchanged, n_valid=1 denotes one available structured prediction.",
        "verification": "Complete exact joins; independent masks/counts/F1/failure bounds for every setting; independent frequency-weighted paired bootstrap, 10000 draws seed 20260915, pointwise descriptive intervals only.",
        "failures": "Null is not negative; keep all failures. Fine-label gold requires strict gate true and annotation present; [] is annotated empty. Predicted gate suppression is an error, not an invalid output.",
    }
    a.save_new(dest / "plan.json", plan)
    print("Assembly plan SHA256: " + a.digest(dest / "plan.json"), flush=True)


def validate_plan(subset):
    path = destination(subset) / "plan.json"
    p = a.read(path)
    a.require(
        p["subset"] == subset and p["scoring_contract"] == s.contract(), "Frozen contract changed"
    )
    for name, h in p["hashes"].items():
        a.require(a.digest(ROOT / name) == h, "Frozen input/source changed: " + name)
    b.validate()
    return p


def run(subset):
    import pandas as pd
    import yaml

    dest = destination(subset)
    p = validate_plan(subset)
    a.require(
        not any(((dest / x).exists() for x in ("results.json", "complete.json", "assembled"))),
        "Preserve complete/partial publication",
    )
    rows = pd.read_parquet(ROOT / f"data/processed/{subset}.parquet").to_dict("records")
    shared.keyed(rows)
    endpoint = s.roster(
        subset, rows, yaml.safe_load((ROOT / "config/taxonomy.yaml").read_text(encoding="utf8"))
    )
    a.require(
        endpoint == p["endpoint"] == a.read(BASE / "scoring_plan.json")["endpoints"][subset],
        "Frozen endpoint changed",
    )
    dimension, language = ALLOWED[subset]
    dev = shared.keyed(b.rows(CORE / "dev_inputs.jsonl"), "key")
    stages = selected_stages(a.read(BASE / "schedule.json"), subset)
    a.require(len(stages) == len(p["stage_audits"]), "Incomplete stage bindings")
    settings = defaultdict(dict)
    origins = {}
    item_hashes = {}
    qwen_rows = 0
    source_cache = {}
    for stage, binding in zip(stages, p["stage_audits"]):
        a.require(stage["id"] == binding["stage"], "Audit order differs")
        ev = a.read(ROOT / binding["evidence"])
        complete = a.read(BASE / "stages" / stage["id"] / "complete.json")
        for job in stage["jobs"]:
            if job["subset"] != subset:
                continue
            identity = b.identity(job)
            d = b.item_dir(BASE / "live", job)
            cp = d / "complete.json"
            rp = d / "result.json"
            for path, h in (
                (cp, complete["item_receipt_hashes"][b.rel(cp)]),
                (rp, ev["verified_output_hashes"][b.rel(rp)]),
            ):
                a.require(a.digest(path) == h, "Audited prediction changed")
                item_hashes[b.rel(path)] = h
            r = a.read(rp)
            a.require(r["identity"] == identity, "Wrong result identity")
            inp = dev[job["key"]]
            checked_input_scope(inp, subset)
            rid = inp["item_id"]
            ident = (job["condition"], job["repeat"])
            a.require(rid not in settings[ident], "Duplicate scheduled result")
            settings[ident][rid] = {"result": r["result"], "n_valid": r["n_valid"]}
            origins[ident, rid] = {
                "key": job["key"],
                "source": b.rel(rp),
                "source_sha256": item_hashes[b.rel(rp)],
                "kind": "qwen_decoder_repeat",
            }
            qwen_rows += 1
    a.require(qwen_rows == len(rows) * 27, "Incomplete Qwen prediction matrix")
    arms = ("full", "kb") if language == "en" else ("full",)
    for arm in arms:
        for seed in range(3):
            path = CORE / (
                f"encoder_results_v5/en_{arm}_seed{seed}_assembled.jsonl"
                if language == "en"
                else f"encoder_audit_v1/de_seed{seed}_{subset}_assembled.jsonl"
            )
            h = a.digest(path)
            a.require(h == p["hashes"][b.rel(path)], "Encoder source changed")
            records = shared.keyed(b.rows(path), "key")
            source_cache[b.rel(path)] = records
            ident = ("encoder_" + arm, seed)
            for row in rows:
                rid = str(row["id"])
                key = language + "/" + rid
                r = records[key]
                a.require(
                    r["item_id"] == rid and r["seed"] == seed and (r["result"] is not None),
                    "Encoder identity/validity differs",
                )
                if language == "en":
                    a.require(
                        r["arm"] == arm and r["scopes"] == dev[key]["scopes"],
                        "Encoder scope differs",
                    )
                    a.require(
                        r["text_sha256"] == hashlib.sha256(dev[key]["text"].encode()).hexdigest(),
                        "Encoder text differs",
                    )
                else:
                    a.require(r["subset"] == subset, "German encoder scope differs")
                settings[ident][rid] = {"result": r["result"], "n_valid": 1}
                origins[ident, rid] = {
                    "key": key,
                    "source": b.rel(path),
                    "source_sha256": h,
                    "kind": "encoder_training_seed",
                }
    validate_settings(settings, [str(r["id"]) for r in rows], subset)
    analyses = {}
    contrib = {}
    for name in sorted(conditions(subset)):
        runs = []
        for rep in range(3):
            c, score = checked_score(rows, settings[name, rep], endpoint)
            contrib[name, rep] = c
            runs.append({"replicate": rep, "score": score})
        analyses[name] = {
            "runs": runs,
            "summary": shared.summarize_three(
                [r["score"]["macro_f1_valid_only"] for r in runs],
                "training_seed" if name.startswith("encoder_") else "decoder_repeat",
            ),
            "invalid_predictions": sum((r["score"]["n_invalid"] for r in runs)),
            "mean_invalid_inclusive_bounds": np.mean(
                [r["score"]["invalid_inclusive_bounds"] for r in runs], axis=0
            ).tolist(),
            "mean_per_label_f1": {
                label: float(np.mean([r["score"]["per_label"][label]["f1"] for r in runs]))
                for label in endpoint["all_labels"]
            },
            "mean_gate_misses": float(
                np.mean([r["score"]["missed_hate_on_gold_positive_scorable_inputs"] for r in runs])
            ),
        }
    groups = {}
    for group, prefix in [
        ("fewshot_uniform", "fewshot_full_s"),
        ("fewshot_eligible", "fewshot_eligible_" + dimension + "_s"),
    ]:
        names = [prefix + str(seed) for seed in (42, 43, 44)]
        groups[group] = {
            "conditions": names,
            "set_summaries": {n: analyses[n]["summary"] for n in names},
            "across_set_means": shared.summarize_three(
                [analyses[n]["summary"]["mean"] for n in names], "example_set_mean"
            ),
            "invalid_predictions": sum((analyses[n]["invalid_predictions"] for n in names)),
            "mean_invalid_inclusive_bounds": np.mean(
                [analyses[n]["mean_invalid_inclusive_bounds"] for n in names], axis=0
            ).tolist(),
            "mean_per_label_f1": {
                label: float(np.mean([analyses[n]["mean_per_label_f1"][label] for n in names]))
                for label in endpoint["all_labels"]
            },
        }

    def selected(name):
        return [
            contrib[n, rep]
            for n in (groups[name]["conditions"] if name in groups else [name])
            for rep in range(3)
        ]

    contrasts = {}
    for other in p["comparisons"]:
        result = shared.checked_contrast(selected("rag_full"), selected(other), primary=False)
        a.require(result["family_ci"] is None, "Secondary contrast promoted to primary")
        result.update({"left": "rag_full", "right": other, "primary": False})
        contrasts["rag_full_minus_" + other] = result
        print(subset + " full minus " + other + ": independently verified", flush=True)
    encoder_check = False
    old = a.read(ROOT / "Thesis_Log/astra_result/evidence/core_encoder_results_20260915_v5.json")
    for arm in arms:
        previous = (
            old["english_dev"][arm + "/" + subset]
            if language == "en"
            else old["german_dev"][subset]
        )
        current = analyses["encoder_" + arm]["summary"]
        a.require(
            np.isclose(current["mean"], previous["new_mean"], atol=1e-14, rtol=0)
            and np.isclose(current["sample_sd"], previous["new_sample_sd"], atol=1e-14, rtol=0),
            "Prior encoder summary differs",
        )
        a.require(
            np.allclose(
                current["values"],
                [x["new"]["macro_f1"] for x in previous["per_seed"]],
                atol=1e-14,
                rtol=0,
            ),
            "Prior encoder seed scores differ",
        )
        if language == "de":
            for run, prior in zip(analyses["encoder_full"]["runs"], previous["per_seed"]):
                a.require(run["replicate"] == prior["seed"], "German encoder seed order differs")
                for label, counts in run["score"]["per_label"].items():
                    reference = prior["new"]["per_class"][label]
                    for key in ("tp", "fp", "fn"):
                        a.require(
                            counts[key] == reference[key], "German encoder confusion counts differ"
                        )
    encoder_check = True
    validate_plan(subset)
    for name, h in item_hashes.items():
        a.require(a.digest(ROOT / name) == h, "Selected output changed during scoring")
    assembled = {}
    published_rows = 0
    for ident, outputs in sorted(settings.items()):
        name, rep = ident
        path = dest / "assembled" / f"{name}_rep{rep}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf8", newline="\n") as f:
            for row in rows:
                rid = str(row["id"])
                f.write(
                    json.dumps(
                        {
                            "id": rid,
                            "condition": name,
                            "replicate": rep,
                            **outputs[rid],
                            **origins[ident, rid],
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                    )
                    + "\n"
                )
        published = b.rows(path)
        a.require(
            [r["id"] for r in published] == [str(r["id"]) for r in rows], "Published roster differs"
        )
        for row in published:
            a.require(
                row["condition"] == name and row["replicate"] == rep, "Published setting differs"
            )
            if row["kind"] == "qwen_decoder_repeat":
                original = a.read(ROOT / row["source"])
                nv = original["n_valid"]
            else:
                original = source_cache[row["source"]][row["key"]]
                nv = 1
            a.require(
                row["result"] == original["result"] and row["n_valid"] == nv,
                "Published prediction differs from source",
            )
        published_rows += len(published)
        assembled[b.rel(path)] = {"sha256": a.digest(path), "rows": len(published)}
    report = {
        "created_utc": a.utc(),
        "status": "complete_secondary_endpoint_results_independently_verified",
        "plan_sha256": a.digest(dest / "plan.json"),
        "endpoint": endpoint,
        "conditions": analyses,
        "fewshot_groups": groups,
        "contrasts": contrasts,
        "assembled": assembled,
        "verified_input_hashes": p["hashes"],
        "verified_item_hashes": item_hashes,
        "scoring_contract": s.contract(),
        "verification": {
            "independent_setting_counts_f1_bounds": len(settings),
            "independent_bootstrap_contrasts": len(contrasts),
            "bootstrap_draws": 10000,
            "bootstrap_seed": 20260915,
            "qwen_predictions": qwen_rows,
            "encoder_predictions": len(arms) * 3 * len(rows),
            "published_files": len(assembled),
            "published_rows_source_checked": published_rows,
            "encoder_scores_match_prior_audit": encoder_check,
        },
        "actual_API_calls": 0,
        "gold_rosters_scored": [subset],
        "limits": [
            "Descriptive secondary/sensitivity post-selection DEV evidence; pointwise intervals do not join the primary comparison family.",
            "Intervals condition on observed decoder repeats, fixed example sets and encoder seeds, not all selection/generation/training variance or shared-text dependence.",
            "RAG/few-shot differences include prompt/knowledge packages, not solely causal retrieval. Fine-label scores include predicted-gate errors.",
            "All invalid outputs are retained; bounds are sensitivity analysis, not imputed predictions. No F1-driven stopping, selection or tuning.",
        ],
        "next": "Keep completed results and continue the frozen remaining programme after operational audit. No Priority 3, TEST/benchmark or LaTeX work.",
    }
    a.save_new(dest / "results.json", report)
    date = report["created_utc"][:10].replace("-", "")
    out = ROOT / f"Thesis_Log/astra_result/evidence/core_secondary_{subset}_results_{date}_v4.json"
    a.save_new(out, report)
    a.save_new(
        dest / "complete.json",
        {
            "status": report["status"],
            "plan_sha256": a.digest(dest / "plan.json"),
            "results_sha256": a.digest(dest / "results.json"),
            "evidence": b.rel(out),
            "evidence_sha256": a.digest(out),
            "assembled": assembled,
        },
    )
    print("Results evidence SHA256: " + a.digest(out), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run"])
    parser.add_argument("--subset", required=True, choices=sorted(ALLOWED))
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(b, "live_client", forbidden):
        prepare(args.subset) if args.mode == "prepare" else run(args.subset)

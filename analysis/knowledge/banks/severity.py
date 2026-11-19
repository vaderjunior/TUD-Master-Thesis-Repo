"""Banks severity."""

import argparse
from collections import defaultdict
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
SUBSET = "en_dev_eval_severity"
DEST = CORE / "severity_results_v1"
PLAN = DEST / "plan.json"
SOURCE = ROOT / "analysis/knowledge/banks/severity.py"
TEST = ROOT / "tests/knowledge/banks/test_severity.py"
LABELS = ["low", "medium", "high"]
COMPARISONS = [
    "rag_small",
    "fewshot_eligible",
    "zero_shot",
    "fewshot_uniform",
    "encoder_full",
    "encoder_kb",
]


def forbidden(*args, **kwargs):
    raise AssertionError("Offline severity assembly only")


def selected_stages(schedule):
    found = [st for st in schedule["stages"] if any((j["subset"] == SUBSET for j in st["jobs"]))]
    a.require(
        [st["id"] for st in found] == ["stage_05_secondary", "stage_06_secondary"],
        "Severity stage membership changed",
    )
    return found


def validate_settings(settings, ids):
    names = {"rag_full", "rag_small", "zero_shot", "encoder_full", "encoder_kb"}
    names.update(("fewshot_full_s" + str(seed) for seed in (42, 43, 44)))
    names.update(("fewshot_eligible_severity_s" + str(seed) for seed in (42, 43, 44)))
    a.require(
        set(settings) == {(name, rep) for name in names for rep in range(3)},
        "Missing/extra condition or repeat",
    )
    for ident, outputs in settings.items():
        a.require(set(outputs) == set(ids), "Incomplete/extra severity roster: " + str(ident))


def independent_counts(rows, outputs, endpoint):
    """Scalar categorical checks independent of scorer truth/bits/contributions."""
    a.require(
        endpoint["dimension"] == "severity" and endpoint["all_labels"] == LABELS,
        "Unexpected severity taxonomy",
    )
    shared.keyed(rows)
    a.require(set(outputs) == {str(r["id"]) for r in rows}, "Incomplete/extra output identities")
    ids = []
    ys = []
    valid = []
    counts = []
    columns = LABELS + ["none_gate_false", "none_gate_true", "invalid"]
    confusion = {label: {col: 0 for col in columns} for label in LABELS}
    for row in rows:
        gate = row.get("gate")
        gold = row.get("severity")
        if isinstance(gate, np.bool_):
            gate = bool(gate)
        if isinstance(gate, float) and np.isnan(gate):
            gate = None
        a.require(gate is None or type(gate) is bool, "Nonboolean gold gate")
        if gate is not True or gold is None or (isinstance(gold, float) and np.isnan(gold)):
            continue
        a.require(gold in LABELS, "Unknown severity gold")
        out = outputs[str(row["id"])]
        ok = out["result"] is not None
        a.require(
            out["n_valid"] in (0, 1, 2, 3) and (out["n_valid"] > 0) == ok,
            "Contradictory validity stamp",
        )
        value = None
        column = "invalid"
        if ok:
            result = out["result"]
            value = result["severity"]
            pred_gate = result["hate"]
            a.require(
                type(pred_gate) is bool and (value is None or value in LABELS),
                "Unknown prediction/gate",
            )
            a.require(pred_gate or value is None, "Severity survived a false predicted hate gate")
            column = (
                value if value is not None else "none_gate_true" if pred_gate else "none_gate_false"
            )
        confusion[gold][column] += 1
        y = [int(gold == label) for label in LABELS]
        c = []
        for label in LABELS:
            c.append(
                [
                    int(gold == label and value == label),
                    int(gold != label and value == label),
                    int(gold == label and value != label),
                ]
                if ok
                else [0, 0, 0]
            )
        ids.append(str(row["id"]))
        ys.append(y)
        valid.append(ok)
        counts.append(c)
    return {
        "ids": ids,
        "gold": np.asarray(ys, dtype=np.int64).reshape(-1, 3),
        "valid": np.asarray(valid, dtype=bool),
        "counts": np.asarray(counts, dtype=np.int64).reshape(-1, 3, 3),
        "labels": LABELS,
        "macro_indices": [LABELS.index(x) for x in endpoint["macro_labels"]],
        "severity_confusion": confusion,
        "severity_confusion_columns": columns,
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
            for index, positive in enumerate(y):
                low[index, 2 if positive else 1] += 1
                if positive:
                    high[index, 0] += 1
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
        "Independent failure bounds differ",
    )
    result["per_label"] = {
        label: {
            "tp": int(base[i, 0]),
            "fp": int(base[i, 1]),
            "fn": int(base[i, 2]),
            "gold_support_full": int(d["gold"][:, i].sum()),
            "gold_support_valid": int(d["gold"][d["valid"], i].sum()),
            "f1": float(shared.independent_f1(base, [i])),
        }
        for i, label in enumerate(LABELS)
    }
    result["severity_confusion"] = d["severity_confusion"]
    result["severity_confusion_columns"] = d["severity_confusion_columns"]
    return (c, result)


def prepare():
    a.require(not PLAN.exists(), "Preserve existing severity assembly plan")
    p = b.validate()
    schedule = a.read(BASE / "schedule.json")
    stages = selected_stages(schedule)
    hashes = {
        **p["input_hashes"],
        **p["source_hashes"],
        **p["generated_hashes"],
        b.rel(b.PLAN): a.digest(b.PLAN),
    }
    for path in (SOURCE, TEST, shared.SOURCE):
        hashes[b.rel(path)] = a.digest(path)
    bindings = []
    for stage in stages:
        rp = BASE / "audits" / f"{stage['id']}_v3" / "complete.json"
        receipt = a.read(rp)
        ep = ROOT / receipt["evidence"]
        ev = a.read(ep)
        cp = BASE / "stages" / stage["id"] / "complete.json"
        a.require(
            a.digest(ep) == receipt["evidence_sha256"]
            and a.digest(cp) == receipt["stage_complete_sha256"],
            "Stage audit binding differs",
        )
        a.require(
            ev["stage"] == stage["id"]
            and ev["semantic_replay_items"] == stage["items"]
            and (not ev["actual_audit_API_calls"]),
            "Wrong stage audit",
        )
        hashes.update(ev["verified_input_hashes"])
        hashes[b.rel(rp)] = a.digest(rp)
        hashes[b.rel(ep)] = a.digest(ep)
        bindings.append({"stage": stage["id"], "receipt": b.rel(rp), "evidence": b.rel(ep)})
    for name, h in hashes.items():
        a.require(a.digest(ROOT / name) == h, "Source/input changed: " + name)
    plan = {
        "created_utc": a.utc(),
        "status": "severity_assembly_prepared_no_scoring",
        "subset": SUBSET,
        "hashes": hashes,
        "stage_audits": bindings,
        "scoring_contract": s.contract(),
        "comparisons": COMPARISONS,
        "scope": "Only complete 45-text English severity DEV roster; descriptive secondary endpoint, not part of the primary contrast family. No other gold scoring, live API, retrieval, training, TEST/benchmark, adaptation or LaTeX.",
        "replication": "Three decoder repeats per condition; three fixed example sets with within-set repeats and between-set means kept separate. Encoder training seeds 0/1/2 are not decoder repeats.",
        "verification": "Exact complete joins; independent severity confusion/count/F1/bounds checks for all 33 settings; each contrast checked by independent frequency-weighted bootstrap, 10000 draws seed 20260915.",
        "encoder_adapter": "Reuse saved gated predictions unchanged; n_valid=1 denotes one available structured prediction, not one Qwen vote.",
        "failure_policy": "Preserve nulls and gate-suppressed severity; missing severity in a valid output counts as a missed gold class. Never infer severity from gold or resample invalid predictions.",
    }
    a.save_new(PLAN, plan)
    print("Severity plan SHA256: " + a.digest(PLAN), flush=True)


def validate_plan():
    p = a.read(PLAN)
    a.require(
        p["subset"] == SUBSET
        and p["scoring_contract"] == s.contract()
        and (p["comparisons"] == COMPARISONS),
        "Scoring plan changed",
    )
    for name, h in p["hashes"].items():
        a.require(a.digest(ROOT / name) == h, "Frozen assembly input/source changed: " + name)
    b.validate()
    return p


def run():
    import pandas as pd
    import yaml

    p = validate_plan()
    a.require(
        not (DEST / "results.json").exists()
        and (not (DEST / "complete.json").exists())
        and (not (DEST / "assembled").exists()),
        "Preserve complete or partial publication",
    )
    rows = pd.read_parquet(ROOT / f"data/processed/{SUBSET}.parquet").to_dict("records")
    shared.keyed(rows)
    endpoint = s.roster(
        SUBSET, rows, yaml.safe_load((ROOT / "config/taxonomy.yaml").read_text(encoding="utf8"))
    )
    a.require(
        endpoint == a.read(BASE / "scoring_plan.json")["endpoints"][SUBSET],
        "Frozen endpoint changed",
    )
    a.require(
        len(rows) == 45 and endpoint["gold_support"] == dict.fromkeys(LABELS, 15),
        "Severity roster differs",
    )
    a.require(
        all((str(row["id"]).startswith("mhs-") for row in rows)),
        "Severity source composition changed",
    )
    dev = shared.keyed(b.rows(CORE / "dev_inputs.jsonl"), "key")
    schedule = a.read(BASE / "schedule.json")
    stages = selected_stages(schedule)
    settings = defaultdict(dict)
    origins = {}
    item_hashes = {}
    qwen_rows = 0
    a.require(len(p["stage_audits"]) == len(stages), "Incomplete stage audit bindings")
    for stage, binding in zip(stages, p["stage_audits"]):
        a.require(stage["id"] == binding["stage"], "Audit stage order differs")
        ev = a.read(ROOT / binding["evidence"])
        complete = a.read(BASE / "stages" / stage["id"] / "complete.json")
        for job in stage["jobs"]:
            if job["subset"] != SUBSET:
                continue
            identity = b.identity(job)
            d = b.item_dir(BASE / "live", job)
            cp = d / "complete.json"
            rp = d / "result.json"
            for path, h in (
                (cp, complete["item_receipt_hashes"][b.rel(cp)]),
                (rp, ev["verified_output_hashes"][b.rel(rp)]),
            ):
                a.require(a.digest(path) == h, "Audited severity prediction changed")
                item_hashes[b.rel(path)] = h
            r = a.read(rp)
            a.require(r["identity"] == identity, "Wrong result identity")
            inp = dev[job["key"]]
            a.require(
                inp["scopes"] == [{"dimension": "severity", "subset": SUBSET}], "Wrong input scope"
            )
            rid = inp["item_id"]
            ident = (job["condition"], job["repeat"])
            a.require(rid not in settings[ident], "Duplicate scheduled prediction")
            settings[ident][rid] = {"result": r["result"], "n_valid": r["n_valid"]}
            origins[ident, rid] = {
                "key": job["key"],
                "source": b.rel(rp),
                "source_sha256": item_hashes[b.rel(rp)],
                "kind": "qwen_decoder_repeat",
            }
            qwen_rows += 1
    a.require(qwen_rows == 1215, "Incomplete severity prediction matrix")
    for arm in ("full", "kb"):
        for seed in range(3):
            path = CORE / f"encoder_results_v5/en_{arm}_seed{seed}_assembled.jsonl"
            h = a.digest(path)
            a.require(h == p["hashes"][b.rel(path)], "Encoder source changed")
            records = shared.keyed(b.rows(path), "key")
            ident = ("encoder_" + arm, seed)
            for row in rows:
                rid = str(row["id"])
                key = "en/" + rid
                r = records[key]
                a.require(
                    r["item_id"] == rid
                    and r["seed"] == seed
                    and (r["arm"] == arm)
                    and (r["result"] is not None),
                    "Encoder identity/validity differs",
                )
                a.require(
                    r["scopes"] == dev[key]["scopes"]
                    and r["text_sha256"] == hashlib.sha256(dev[key]["text"].encode()).hexdigest(),
                    "Encoder scope/text differs",
                )
                settings[ident][rid] = {"result": r["result"], "n_valid": 1}
                origins[ident, rid] = {
                    "key": key,
                    "source": b.rel(path),
                    "source_sha256": h,
                    "kind": "encoder_training_seed",
                }
    validate_settings(settings, [str(r["id"]) for r in rows])
    conditions = {}
    contrib = {}
    for name in sorted({name for name, rep in settings}):
        runs = []
        for rep in range(3):
            c, score = checked_score(rows, settings[name, rep], endpoint)
            contrib[name, rep] = c
            runs.append({"replicate": rep, "score": score})
        conditions[name] = {
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
                for label in LABELS
            },
            "mean_confusion": {
                label: {
                    col: float(
                        np.mean([r["score"]["severity_confusion"][label][col] for r in runs])
                    )
                    for col in runs[0]["score"]["severity_confusion_columns"]
                }
                for label in LABELS
            },
        }
    groups = {}
    for group, prefix in [
        ("fewshot_uniform", "fewshot_full_s"),
        ("fewshot_eligible", "fewshot_eligible_severity_s"),
    ]:
        names = [prefix + str(seed) for seed in (42, 43, 44)]
        groups[group] = {
            "conditions": names,
            "set_summaries": {n: conditions[n]["summary"] for n in names},
            "across_set_means": shared.summarize_three(
                [conditions[n]["summary"]["mean"] for n in names], "example_set_mean"
            ),
            "invalid_predictions": sum((conditions[n]["invalid_predictions"] for n in names)),
            "mean_invalid_inclusive_bounds": np.mean(
                [conditions[n]["mean_invalid_inclusive_bounds"] for n in names], axis=0
            ).tolist(),
            "mean_per_label_f1": {
                label: float(np.mean([conditions[n]["mean_per_label_f1"][label] for n in names]))
                for label in LABELS
            },
        }

    def selected(name):
        names = groups[name]["conditions"] if name in groups else [name]
        return [contrib[n, rep] for n in names for rep in range(3)]

    contrasts = {}
    for other in COMPARISONS:
        result = shared.checked_contrast(selected("rag_full"), selected(other), primary=False)
        a.require(result["family_ci"] is None, "Secondary contrast treated as primary")
        result.update({"left": "rag_full", "right": other, "primary": False})
        contrasts["rag_full_minus_" + other] = result
        print("Severity full minus " + other + ": independently verified", flush=True)
    validate_plan()
    for name, h in item_hashes.items():
        a.require(a.digest(ROOT / name) == h, "Selected output changed during scoring")
    assembled = {}
    for ident, outputs in sorted(settings.items()):
        name, rep = ident
        path = DEST / "assembled" / f"{name}_rep{rep}.jsonl"
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
        assembled[b.rel(path)] = {"sha256": a.digest(path), "rows": len(outputs)}
    report = {
        "created_utc": a.utc(),
        "status": "complete_severity_results_independently_verified",
        "plan_sha256": a.digest(PLAN),
        "endpoint": endpoint,
        "conditions": conditions,
        "fewshot_groups": groups,
        "contrasts": contrasts,
        "assembled": assembled,
        "verified_input_hashes": p["hashes"],
        "verified_item_hashes": item_hashes,
        "scoring_contract": s.contract(),
        "verification": {
            "setting_count": 33,
            "assembled_rows": 1485,
            "qwen_predictions": 1215,
            "encoder_predictions": 270,
            "independent_counts_f1_bounds": 33,
            "independent_bootstrap_contrasts": 6,
            "draws_per_contrast": 10000,
            "bootstrap_seed": 20260915,
        },
        "actual_API_calls": 0,
        "severity_gold_scoring": True,
        "other_gold_rosters_scored": False,
        "limits": [
            "Secondary descriptive post-selection DEV evidence, only 45 texts from MHS, 15 per severity class.",
            "Pointwise intervals are not multiplicity-adjusted primary claims; observed seeds/repeats/sets are fixed.",
            "Severity is scored after the predicted hate gate; a valid output with no severity is a missed gold class, not an invalid answer.",
            "Unknown-output bounds relax multiclass exclusivity; bounds are not imputed predictions.",
            "Uniform few-shot sets have no severity annotations; eligible sets are separately declared controls, not selected after outcomes.",
            "RAG/few-shot changes prompt packages and available knowledge, not solely causal retrieval.",
        ],
        "next": "Preserve all settings/results; continue frozen secondary stages after operational review. No Priority 3, TEST/benchmark or LaTeX work.",
    }
    a.save_new(DEST / "results.json", report)
    date = report["created_utc"][:10].replace("-", "")
    out = ROOT / f"Thesis_Log/astra_result/evidence/core_severity_results_{date}_v1.json"
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
    print("Severity evidence SHA256: " + a.digest(out), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(b, "live_client", forbidden):
        prepare() if args.mode == "prepare" else run()

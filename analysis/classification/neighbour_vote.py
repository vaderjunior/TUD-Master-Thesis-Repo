"Vote from observed neighbour labels."

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
import yaml
from copy import deepcopy
import os
from pathlib import Path
import re
import socket
import unicodedata
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score
from analysis.common import api as a
from analysis.common import batch as core
from analysis.common import scoring as scoring

ROOT = a.ROOT
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
BASE = ROOT / "experiments/astra/sq2_retrieved_label_vote_20261001_v1"
SOURCE = Path(__file__).resolve()
FIELDS = {
    "hate": "gate",
    "hate_type": "hate_types",
    "target_group": "target_groups",
    "severity": "severity",
    "legal": "legal",
}
FOLDERS = {
    "en_dev_eval_main": "primary_results_v1",
    "de_legal_dev_eval": "primary_results_v1",
    "en_dev_eval_sq3_types": "secondary_results_v4/en_dev_eval_sq3_types",
    "en_dev_eval_targets": "secondary_results_v2/en_dev_eval_targets",
    "en_dev_eval_severity": "severity_results_v1",
    "de_dev_eval": "secondary_results_v3/de_dev_eval",
    "en_dev_eval_types": "secondary_results_v4/en_dev_eval_types",
}
INPUTS = {}


def rel(p):
    return core.rel(p)


def save(p, v):
    a.require(
        Path(p).resolve().is_relative_to(BASE.resolve()), "Write outside new analysis"
    )
    a.save_new(p, v)


def bind(p, expected=None):
    actual = a.digest(p)
    name = rel(p)
    a.require(expected is None or expected == actual, "Input hash differs: " + name)
    a.require(
        name not in INPUTS or INPUTS[name] == actual, "Input changed during reading"
    )
    INPUTS[name] = actual
    return p


def read(p, expected=None):
    return a.read(bind(p, expected))


def rows(p, expected=None):
    return core.rows(bind(p, expected))


def verify(inputs):
    for name, h in inputs.items():
        a.require(a.digest(ROOT / name) == h, "Frozen input changed: " + name)


def no_network(*args, **kwargs):
    raise AssertionError("Offline only")


def normalized(text):
    return re.sub("\\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def test_file():
    return BASE / "development" / ("tests_" + a.digest(SOURCE) + ".json")


def validate_meta(meta, labels):
    gate = meta.get("gate")
    a.require(gate is None or type(gate) is bool, "Unknown gate")
    for head, field in FIELDS.items():
        if head == "hate":
            continue
        value = meta.get(field)
        if value is None:
            continue
        if head == "severity":
            a.require(value in labels[head], "Unknown severity")
        else:
            a.require(
                isinstance(value, list) and set(value) <= set(labels[head]),
                "Unknown annotation/list",
            )


def predict(metas, labels):
    a.require(len(metas) == 5, "Five neighbours required")
    for meta in metas:
        validate_meta(meta, labels)
    gates = [m["gate"] for m in metas if m.get("gate") is not None]
    gate = sum(gates) * 2 > len(gates) if gates else None
    out = {
        "hate": {
            "value": gate,
            "valid": gate is not None,
            "annotated_neighbours": len(gates),
            "reason": "majority_or_false_tie" if gates else "no_observed_gate",
        }
    }
    for head in ["hate_type", "target_group", "severity", "legal"]:
        field = FIELDS[head]
        votes = [
            m[field]
            for m in metas
            if m.get(field) is not None and (head == "legal" or m.get("gate") is True)
        ]
        if head != "legal" and gate is False:
            value = [] if head != "severity" else None
            valid = True
            reason = "gate_forced_empty"
        elif head != "legal" and gate is None:
            value = None
            valid = False
            reason = "unobserved_gate"
        elif not votes:
            value = None
            valid = False
            reason = "no_head_annotation"
        elif head == "severity":
            ordered = sorted((labels[head].index(v) for v in votes))
            value = labels[head][ordered[(len(ordered) - 1) // 2]]
            valid = True
            reason = "lower_median"
        else:
            counts = Counter((l for v in votes for l in set(v)))
            value = [l for l in labels[head] if counts[l] * 2 > len(votes)]
            valid = True
            reason = "strict_majority"
        out[head] = {
            "value": value,
            "valid": valid,
            "annotated_neighbours": len(votes),
            "reason": reason,
        }
    return out


def independent_prediction(metas, labels):
    observed = np.array([m.get("gate") is not None for m in metas])
    bits = np.array([m.get("gate") is True for m in metas])
    gate = bool(np.mean(bits[observed]) > 0.5) if observed.any() else None
    result = {"hate": (gate, gate is not None, int(observed.sum()))}
    for head in ["hate_type", "target_group", "severity", "legal"]:
        votes = [
            m.get(FIELDS[head])
            for m in metas
            if m.get(FIELDS[head]) is not None
            and (head == "legal" or m.get("gate") is True)
        ]
        if head != "legal" and gate is False:
            value = None if head == "severity" else []
            valid = True
        elif head != "legal" and gate is None or not votes:
            value = None
            valid = False
        elif head == "severity":
            values = np.array([labels[head].index(v) for v in votes])
            value = labels[head][int(np.quantile(values, 0.5, method="lower"))]
            valid = True
        else:
            indicators = np.array([[l in v for l in labels[head]] for v in votes])
            value = [
                l for l, c in zip(labels[head], indicators.mean(axis=0)) if c > 0.5
            ]
            valid = True
        result[head] = (value, valid, len(votes))
    return result


def clusters(inputs):
    parent = {k: k for k in inputs}

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def union(x, y):
        parent[find(max(x, y))] = find(min(x, y))

    textgroups = defaultdict(list)
    for k, r in inputs.items():
        textgroups[r["lang"], normalized(r["text"])].append(k)
    path = ROOT / "experiments/astra/sq3_full_kb_taxonomy_20261001_v1/clusters.json"
    groups = list(textgroups.values()) + [
        [f"en/{i}" for i in r["ids"] if f"en/{i}" in inputs] for r in read(path)
    ]
    for group in groups:
        for key in group[1:]:
            union(group[0], key)
    return {k: find(k) for k in inputs}


def prepare():
    a.require(not (BASE / "plan.json").exists(), "Preserve prior plan")
    tests = read(test_file())
    a.require(
        tests["source_sha256"] == a.digest(SOURCE) and tests["passed"],
        "Current passing tests needed",
    )
    bind(SOURCE)
    bind(BASE / "ANALYSIS_PROTOCOL.md")
    bind(ROOT / "analysis/common/scoring.py")
    bind(ROOT / "analysis/common/api.py")
    bind(ROOT / "analysis/common/batch.py")
    original = read(CORE / "plan.json")
    retrieval = read(CORE / "retrieval_v1/plan.json")
    dev = {
        r["key"]: r
        for r in rows(
            CORE / "dev_inputs.jsonl",
            original["generated_hashes"][rel(CORE / "dev_inputs.jsonl")],
        )
    }
    a.require(len(dev) == 1295, "Core roster changed")
    taxonomy = yaml.safe_load(
        bind(
            ROOT / "config/taxonomy.yaml",
            original["input_hashes"]["config/taxonomy.yaml"],
        ).read_text(encoding="utf8")
    )
    labels = {
        h: ["not_hate", "hate"] if h == "hate" else taxonomy["dimensions"][h]["labels"]
        for h in FIELDS
    }
    retrieved = {}
    coverage = {}
    for pool in ["full", "small"]:
        root = CORE / "retrieval_v1" / pool
        done = read(root / "complete.json")
        p = root / "prompts.jsonl"
        prompts = rows(p, done["prompt_sha256"])
        records_path = CORE / "kb" / pool / "records.jsonl"
        records = {
            r["id"]: r
            for r in rows(records_path, retrieval["input_hashes"][rel(records_path)])
        }
        a.require(
            len(prompts) == len({r["key"] for r in prompts}) == 1295,
            "Saved retrieval roster",
        )
        counts = Counter()
        retrieved[pool] = {}
        for row in prompts:
            key = row["key"]
            a.require(
                row["text"] == dev[key]["text"] and row["lang"] == dev[key]["lang"],
                "Query text differs",
            )
            a.require(
                a.signature(row["context"]) == row["context_sha256"]
                and a.signature(row["messages"]) == row["messages_sha256"],
                "Context hash differs",
            )
            groups = row["context"]["example_groups"]
            a.require(
                len(groups) == 1 and groups[0][0] == "Labelled examples",
                "Unexpected example groups",
            )
            examples = groups[0][1]
            ids = [e["id"] for e in examples]
            a.require(
                len(ids) == len(set(ids)) == 5
                and ids == row["retrieved"]["Labelled examples"],
                "Neighbour roster differs",
            )
            for ex in examples:
                rec = records[ex["id"]]
                meta = {k: v for k, v in ex["meta"].items() if k != "_ranks"}
                expected = {
                    **{
                        k: rec.get(k)
                        for k in ["kind", "dimension", "label", "lang", "source"]
                    },
                    **rec["meta"],
                }
                a.require(
                    meta == expected
                    and ex["text"] == rec["text"]
                    and (rec["kind"] == "example"),
                    "Saved neighbour differs from KB record",
                )
                validate_meta(meta, labels)
                for head, field in FIELDS.items():
                    counts[head + "_observed"] += meta.get(field) is not None
            retrieved[pool][key] = {
                "ids": ids,
                "metas": [e["meta"] for e in examples],
                "messages_sha256": row["messages_sha256"],
            }
        coverage[pool] = dict(counts)
    endpoint_data = {}
    exports = {}
    old_results = {}
    for subset, folder in FOLDERS.items():
        base = CORE / folder
        receipt = read(base / "complete.json")
        old = read(base / "results.json", receipt["results_sha256"])
        endpoint = old["endpoints"][subset] if folder == "primary_results_v1" else old
        ep = endpoint["endpoint"]
        old_results[subset] = rel(base / "results.json")
        p = ROOT / f"data/processed/{subset}.parquet"
        data = pd.read_parquet(bind(p, original["input_hashes"][rel(p)])).to_dict(
            "records"
        )
        lookup = {str(r["id"]): r for r in data}
        a.require(len(lookup) == len(data), "Duplicate gold ID")
        lang = "en" if subset.startswith("en_") else "de"
        goldrows = []
        for iid in ep["scorable_ids"]:
            row = lookup[iid]
            key = lang + "/" + iid
            a.require(dev[key]["text"] == row["text"], "Gold/query text mismatch")
            val = scoring.truth(row, ep["dimension"])
            a.require(val is not None, "Unscorable original row")
            goldrows.append(
                {"id": iid, "key": key, "gold": val, "source": iid.split("-", 1)[0]}
            )
        support = Counter(
            (
                l
                for row in goldrows
                for l, b in zip(
                    ep["all_labels"],
                    scoring.bits(row["gold"], ep["dimension"], ep["all_labels"]),
                )
                if b
            )
        )
        a.require(
            {l: support[l] for l in ep["all_labels"]} == ep["gold_support"],
            "Original gold support differs",
        )
        endpoint_data[subset] = {"endpoint": ep, "rows": goldrows}
        for pool in ["full", "small"]:
            for rep in range(3):
                p = (
                    base
                    / "assembled"
                    / (subset if folder == "primary_results_v1" else "")
                    / f"rag_{pool}_rep{rep}.jsonl"
                )
                exported = rows(p, receipt["assembled"][rel(p)]["sha256"])
                a.require(
                    len(exported) == len({r["id"] for r in exported}),
                    "Duplicate export",
                )
                a.require(
                    set(ep["scorable_ids"]) <= {r["id"] for r in exported},
                    "Missing original output",
                )
                exports[f"{subset}/{pool}/{rep}"] = rel(p)
    cluster = clusters(dev)
    save(
        BASE / "frozen_inputs.json",
        {"retrieved": retrieved, "endpoints": endpoint_data, "clusters": cluster},
    )
    bind(BASE / "frozen_inputs.json")
    verify(INPUTS)
    save(
        BASE / "plan.json",
        {
            "status": "frozen_before_neighbour_prediction_scores",
            "created_utc": a.utc(),
            "input_hashes": INPUTS.copy(),
            "environment": core.environment(),
            "source_sha256": a.digest(SOURCE),
            "labels": labels,
            "exports": exports,
            "original_results": old_results,
            "subsets": list(FOLDERS),
            "pools": ["full", "small"],
            "inputs": 1295,
            "deterministic_predictions": 2590,
            "comparison_order": [
                f"{s}/{p}" for s in FOLDERS for p in ["full", "small"]
            ],
            "bootstrap_draws": 10000,
            "bootstrap_seed_base": 202610010,
            "interval": "Exploratory pointwise 95%, 14 comparisons; no multiple-comparison confirmatory claims.",
            "annotation_slot_counts": coverage,
            "clusters": len(set(cluster.values())),
            "actual_API_calls": 0,
        },
    )
    print(
        "Voting protocol, all neighbours and endpoint rosters frozen before new baseline scores.",
        flush=True,
    )


def contribution(y, pred, valid):
    x = np.asarray(pred, dtype=np.int64)
    v = np.asarray(valid, dtype=bool)
    c = np.stack([y * x, (1 - y) * x, y * (1 - x)], axis=-1)
    c[~v] = 0
    return (c, v, x)


def measure(y, c, v, x, indices, labels):
    base = c.sum(axis=0)
    low = base.copy()
    high = base.copy()
    missing = y[~v]
    low[:, 1] += (1 - missing).sum(axis=0)
    low[:, 2] += missing.sum(axis=0)
    high[:, 0] += missing.sum(axis=0)
    f = lambda z: float(scoring.macro(z, indices))
    score = f(base) if v.any() else None
    if v.any():
        independent = float(
            f1_score(
                y[v][:, indices], x[v][:, indices], average="macro", zero_division=0
            )
        )
        a.require(abs(score - independent) < 1e-12, "Independent sklearn score differs")
        expected = np.array(
            [
                [
                    int(((y[:, i] == 1) & (x[:, i] == 1) & v).sum()),
                    int(((y[:, i] == 0) & (x[:, i] == 1) & v).sum()),
                    int(((y[:, i] == 1) & (x[:, i] == 0) & v).sum()),
                ]
                for i in range(len(labels))
            ]
        )
        a.require(np.array_equal(expected, base), "Independent confusion counts differ")
    return {
        "macro_f1_valid_only": score,
        "n_scorable": len(v),
        "n_valid": int(v.sum()),
        "n_invalid": int((~v).sum()),
        "coverage": float(v.mean()),
        "invalid_inclusive_bounds": [f(low), f(high)],
        "per_label_counts": base.tolist(),
        "macro_labels": [labels[i] for i in indices],
    }


def paired(baseline, rag, valid, clusters, indices, seed, draws):
    positions = np.flatnonzero(valid)
    if not len(positions):
        return {"rows": 0, "difference": None, "ci95": None}
    names = sorted({clusters[i] for i in positions})
    lookup = {c: n for n, c in enumerate(names)}
    cube = np.stack([baseline, *rag])
    aggregated = np.zeros((4, len(names), cube.shape[2], 3), dtype=np.int64)
    for i in positions:
        aggregated[:, lookup[clusters[i]]] += cube[:, i]
    orig = scoring.macro(aggregated.sum(axis=1), indices)
    rng = np.random.default_rng(seed)
    samples = []
    check = []
    for start in range(0, draws, 200):
        ix = rng.integers(len(names), size=(min(200, draws - start), len(names)))
        values = scoring.macro(aggregated[:, ix].sum(axis=2), indices)
        samples.extend((values[1:].mean(axis=0) - values[0]).tolist())
        weights = np.array([np.bincount(v, minlength=len(names)) for v in ix])
        totals = np.einsum("bg,sglf->sblf", weights, aggregated)
        tp = totals[:, :, :, 0]
        denom = 2 * tp + totals[:, :, :, 1] + totals[:, :, :, 2]
        f = np.divide(
            2 * tp, denom, out=np.zeros_like(tp, dtype=float), where=denom != 0
        )[:, :, indices].mean(axis=2)
        check.extend((f[1:].mean(axis=0) - f[0]).tolist())
    a.require(
        np.allclose(samples, check, atol=1e-12, rtol=0), "Independent bootstrap differs"
    )
    return {
        "rows": len(positions),
        "clusters": len(names),
        "baseline_f1": float(orig[0]),
        "rag_runs": [float(v) for v in orig[1:]],
        "rag_mean": float(orig[1:].mean()),
        "difference": float(orig[1:].mean() - orig[0]),
        "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
        "seed": seed,
        "draws": draws,
        "bootstrap_sha256": a.signature(samples),
        "draws_independently_checked": True,
    }


def run():
    plan = corrected_plan()
    verify(plan["input_hashes"])
    a.require(plan["environment"] == core.environment(), "Environment changed")
    a.require(not (BASE / "complete.json").exists(), "Preserve completed results")
    frozen = a.read(BASE / "frozen_inputs.json")
    predictions = {}
    checks = 0
    results = {}
    for pool, inputs in frozen["retrieved"].items():
        predictions[pool] = {}
        for key, entry in inputs.items():
            value = predict(entry["metas"], plan["labels"])
            other = independent_prediction(entry["metas"], plan["labels"])
            a.require(
                all(
                    (
                        (v["value"], v["valid"], v["annotated_neighbours"]) == other[h]
                        for h, v in value.items()
                    )
                ),
                "Independent neighbour vote differs",
            )
            predictions[pool][key] = value
            checks += 1
    old_scores = 0
    comparisons = 0
    for subset in plan["subsets"]:
        data = frozen["endpoints"][subset]
        ep = data["endpoint"]
        dimension = ep["dimension"]
        labels = ep["all_labels"]
        idx = [labels.index(l) for l in ep["macro_labels"]]
        y = np.array(
            [scoring.bits(r["gold"], dimension, labels) for r in data["rows"]],
            dtype=np.int64,
        )
        cluster = [frozen["clusters"][r["key"]] for r in data["rows"]]
        old = a.read(ROOT / plan["original_results"][subset])
        old = old["endpoints"][subset] if "endpoints" in old else old
        results[subset] = {"endpoint": ep, "pools": {}}
        for pool in plan["pools"]:
            answers = [predictions[pool][r["key"]][dimension] for r in data["rows"]]
            bc, bv, bx = contribution(
                y,
                [
                    (
                        scoring.bits(v["value"], dimension, labels)
                        if v["valid"]
                        else [0] * len(labels)
                    )
                    for v in answers
                ],
                [v["valid"] for v in answers],
            )
            bm = measure(y, bc, bv, bx, idx, labels)
            rc = []
            rv = []
            rmetrics = []
            for repeat in range(3):
                outputs = {
                    r["id"]: r
                    for r in core.rows(
                        ROOT / plan["exports"][f"{subset}/{pool}/{repeat}"]
                    )
                }
                values = [outputs[r["id"]] for r in data["rows"]]
                a.require(
                    all(((v["result"] is None) == (v["n_valid"] == 0) for v in values)),
                    "Contradictory saved validity",
                )
                c, v, x = contribution(
                    y,
                    [
                        (
                            scoring.bits(r["result"][dimension], dimension, labels)
                            if r["result"] is not None
                            else [0] * len(labels)
                        )
                        for r in values
                    ],
                    [r["result"] is not None for r in values],
                )
                metric = measure(y, c, v, x, idx, labels)
                prior = next(
                    (
                        r["score"]
                        for r in old["conditions"]["rag_" + pool]["runs"]
                        if r["replicate"] == repeat
                    )
                )
                for field in metric:
                    if field == "macro_f1_valid_only":
                        a.require(
                            abs(metric[field] - prior[field]) < 1e-12,
                            "Original RAG F1 differs",
                        )
                    elif field == "invalid_inclusive_bounds":
                        a.require(
                            np.allclose(
                                metric[field], prior[field], atol=1e-12, rtol=0
                            ),
                            "Original bounds differ",
                        )
                    else:
                        a.require(
                            metric[field] == prior[field],
                            "Original metric differs: " + field,
                        )
                old_scores += 1
                rc.append(c)
                rv.append(v)
                rmetrics.append(metric)
            shared = bv & np.logical_and.reduce(rv)
            comparison = paired(
                bc,
                rc,
                shared,
                cluster,
                idx,
                plan["bootstrap_seed_base"] + comparisons,
                plan["bootstrap_draws"],
            )
            comparisons += 1
            strata = {}
            for source in sorted({r["source"] for r in data["rows"]}):
                m = np.array([r["source"] == source for r in data["rows"]])
                strata[source] = measure(y[m], bc[m], bv[m], bx[m], idx, labels)
            results[subset]["pools"][pool] = {
                "baseline": bm,
                "rag_runs": rmetrics,
                "rag_mean_own_valid": float(
                    np.mean([r["macro_f1_valid_only"] for r in rmetrics])
                ),
                "common_valid_rag_minus_baseline": comparison,
                "annotation_neighbour_histogram": dict(
                    Counter((str(v["annotated_neighbours"]) for v in answers))
                ),
                "decision_reasons": dict(Counter((v["reason"] for v in answers))),
                "by_source": strata,
            }
            print(
                f"{subset} {pool}: {bm['n_valid']}/{bm['n_scorable']} answered; baseline {bm['macro_f1_valid_only']}; RAG {results[subset]['pools'][pool]['rag_mean_own_valid']}",
                flush=True,
            )
    verify(plan["input_hashes"])
    save(BASE / "predictions.json", predictions)
    save(
        BASE / "results.json",
        {
            "status": "offline_baseline_scored_and_independently_verified",
            "created_utc": a.utc(),
            "plan_sha256": a.digest(BASE / "plan.json"),
            "endpoints": results,
            "verification": {
                "neighbour_predictions": checks,
                "old_metrics_reproduced": old_scores,
                "paired_comparisons": comparisons,
                "independent_bootstrap_draws": comparisons * plan["bootstrap_draws"],
            },
            "actual_API_calls": 0,
            "new_training": False,
            "limitations": [
                "Exposed DEV; exploratory analysis.",
                "Own-answerable scores require coverage; shared-mask comparisons are conditional.",
                "Baseline still uses pretrained retrieval embeddings.",
                "Pointwise intervals, no 14-way confirmatory claim.",
                "Qwen contrast includes instructions and abstracts as well as generation.",
            ],
        },
    )
    save(
        BASE / "complete.json",
        {
            "status": "offline_label_vote_complete",
            "plan_sha256": a.digest(BASE / "plan.json"),
            "results_sha256": a.digest(BASE / "results.json"),
            "predictions_sha256": a.digest(BASE / "predictions.json"),
            "source_sha256": a.digest(SOURCE),
            "input_bindings": len(plan["input_hashes"]),
            "actual_API_calls": 0,
        },
    )


class Tests(unittest.TestCase):
    labels = {
        "hate": ["not_hate", "hate"],
        "hate_type": ["a", "b"],
        "target_group": ["x", "y"],
        "severity": ["low", "medium", "high"],
        "legal": ["l", "m"],
    }

    def m(self, **kwargs):
        return {
            "gate": None,
            "hate_types": None,
            "target_groups": None,
            "severity": None,
            "legal": None,
            **kwargs,
        }

    def p(self, metas):
        return predict(metas + [self.m()] * (5 - len(metas)), self.labels)

    def test_missing_is_not_empty(self):
        v = self.p(
            [self.m(gate=True, hate_types=["a"]), self.m(gate=True, hate_types=[])]
        )
        self.assertEqual(v["hate_type"]["value"], [])
        self.assertEqual(v["hate_type"]["annotated_neighbours"], 2)
        v = self.p([self.m(gate=True, hate_types=["a"])])
        self.assertEqual(v["hate_type"]["value"], ["a"])

    def test_no_annotation_abstains(self):
        v = self.p([self.m(gate=True)])
        self.assertFalse(v["hate_type"]["valid"])
        self.assertFalse(v["legal"]["valid"])

    def test_gate_unknown_and_legal_independent(self):
        v = self.p([self.m(legal=["l"])])
        self.assertFalse(v["hate"]["valid"])
        self.assertEqual(v["legal"]["value"], ["l"])

    def test_gate_tie_false_forces_empty(self):
        v = self.p(
            [self.m(gate=True, hate_types=["a"]), self.m(gate=False, legal=["l"])]
        )
        self.assertFalse(v["hate"]["value"])
        self.assertEqual(v["hate_type"]["value"], [])
        self.assertEqual(v["legal"]["value"], ["l"])

    def test_duplicate_labels_one_vote(self):
        v = self.p(
            [self.m(gate=True, hate_types=["a", "a"]), self.m(gate=True, hate_types=[])]
        )
        self.assertEqual(v["hate_type"]["value"], [])

    def test_lower_median(self):
        v = self.p(
            [self.m(gate=True, severity="high"), self.m(gate=True, severity="low")]
        )
        self.assertEqual(v["severity"]["value"], "low")

    def test_benign_head_annotation_excluded(self):
        v = self.p(
            [
                self.m(gate=False, hate_types=["b"]),
                self.m(gate=True, hate_types=["a"]),
                self.m(gate=True),
            ]
        )
        self.assertEqual(v["hate_type"]["value"], ["a"])
        self.assertEqual(v["hate_type"]["annotated_neighbours"], 1)

    def test_unknown_label_rejected(self):
        with self.assertRaises(a.IntegrityError):
            self.p([self.m(gate=True, hate_types=["unknown"])])

    def test_all_abstentions_not_negatives(self):
        y = np.array([[1, 0], [0, 1]])
        c, v, x = contribution(y, [[0, 0], [0, 0]], [False, False])
        m = measure(y, c, v, x, [0, 1], ["a", "b"])
        self.assertIsNone(m["macro_f1_valid_only"])
        self.assertEqual(m["invalid_inclusive_bounds"], [0.0, 1.0])


def main():
    os.chdir(ROOT)
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["test", "prepare", "run"])
    mode = p.parse_args().mode
    with patch.object(socket.socket, "connect", no_network), patch.object(
        socket, "create_connection", no_network
    ), patch.object(core, "live_client", no_network):
        if mode == "test":
            result = unittest.TextTestRunner(verbosity=2).run(
                unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
            )
            if not result.wasSuccessful():
                raise SystemExit(1)
            save(
                test_file(),
                {
                    "passed": True,
                    "tests": result.testsRun,
                    "source_sha256": a.digest(SOURCE),
                    "actual_API_calls": 0,
                },
            )
        elif mode == "prepare":
            prepare()
        else:
            run()


def corrected_plan():
    original = a.read(BASE / "plan.json")
    verify(original["input_hashes"])
    tax = yaml.safe_load((ROOT / "config/taxonomy.yaml").read_text(encoding="utf8"))
    corrected = deepcopy(original)
    for head in FIELDS:
        labels = (
            ["not_hate", "hate"]
            if head == "hate"
            else list(tax["dimensions"][head]["labels"])
        )
        a.require(
            set(labels) == set(original["labels"][head]),
            "Changing label membership is forbidden",
        )
        corrected["labels"][head] = labels
    a.require(
        corrected["labels"]["severity"] == ["low", "medium", "high"],
        "Declared ordinal order changed",
    )
    return corrected


if __name__ == "__main__":
    main()

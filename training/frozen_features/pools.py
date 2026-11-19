import argparse
from collections import Counter
import copy
import hashlib
from pathlib import Path
import time
import numpy as np
from training.frozen_features import compact as old
from training.frozen_features import fit as linear
from analysis.common import study as c
from analysis.common.api import save_new

BASE = c.ROOT / "experiments/astra/compact_pools_20261008_v1"
POOLS = c.ROOT / "experiments/astra/sq1_pool_size_composition_20260926_v1"
HEAD_FIELD = {
    "hate": "gate",
    "hate_type": "hate_types",
    "target_group": "target_groups",
    "severity": "severity",
}
DRAW_SEEDS = (101, 102, 103, 104, 105)
SHUFFLE_SEEDS = tuple(range(201, 211))


def rank(salt, seed, key):
    return (hashlib.sha256(f"{salt}|{seed}|{key}".encode("utf8")).hexdigest(), key)


def permute_blocks(rows, seed):
    order = sorted(
        range(len(rows)),
        key=lambda k: rank("astra-type-negative-control-v1", seed, rows[k]["id"]),
    )
    out = [
        r | {"target": rows[j]["target"], "label_donor_id": rows[j]["id"]}
        for r, j in zip(rows, order)
    ]
    c.require(
        Counter((tuple(r["target"]) for r in out))
        == Counter((tuple(r["target"]) for r in rows)),
        "Shuffle changed co-label distribution",
    )
    return out


def pool_data(ids, records, index, template):
    from sklearn.model_selection import StratifiedGroupKFold

    ids = sorted(ids)
    pos = {rid: i for i, rid in enumerate(ids)}
    heads = {}
    for head in old.HEADS:
        label_order = template["heads"][head]["labels"]
        rows = []
        for rid in ids:
            r = records[rid]
            meta = r["meta"]
            value = meta.get(HEAD_FIELD[head])
            if head != "hate" and (meta.get("gate") is not True or value is None):
                continue
            c.require(value is not None, "English gate label missing")
            if head == "hate":
                c.require(type(value) is bool, "Gate annotation must be strict boolean")
                target = int(value)
            elif head == "severity":
                target = label_order.index(value)
            else:
                c.require(
                    isinstance(value, list) and set(value) <= set(label_order),
                    "Unknown/malformed training labels",
                )
                target = [int(lab in value) for lab in label_order]
            rows.append(
                dict(
                    id=rid,
                    feature_row=pos[rid],
                    vector_index=index[rid],
                    target=target,
                    text_hash=c.signature(r["text"]),
                    cluster=c.signature(old.norm(r["text"])),
                    source=r["source"],
                )
            )
        heads[head] = {
            "labels": label_order,
            "kind": template["heads"][head]["kind"],
            "rows": rows,
        }
    gate = heads["hate"]["rows"]
    y = [r["target"] for r in gate]
    groups = [r["cluster"] for r in gate]
    folds = {}
    for seed in old.SEEDS:
        values = np.full(len(ids), -1, dtype=int)
        for k, (train, val) in enumerate(
            StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(
                np.zeros((len(ids), 1)), y, groups
            )
        ):
            c.require(
                not set(np.array(groups)[train]) & set(np.array(groups)[val]),
                "TRAIN fold text overlap",
            )
            values[val] = k
        c.require((values >= 0).all(), "Incomplete folds")
        folds[str(seed)] = values.tolist()
    return {"gate_ids": ids, "heads": heads, "folds": folds}


def prepare():
    original = old.validate()
    small = c.read(old.BASE / "train.json")
    records_all = c.rows(c.CORE / "kb/full/records.jsonl")
    records = {
        r["id"]: r for r in records_all if r["kind"] == "example" and r["lang"] == "en"
    }
    index = {r["id"]: i for i, r in enumerate(records_all)}
    manifest_path = POOLS / "pool_manifests.json"
    mp = c.read(POOLS / "scientific_plan.json")
    c.require(
        mp["input_hashes"][c.rel(manifest_path)] == c.sha(manifest_path),
        "Original random pool manifest changed",
    )
    pools = {"curated_small": small}
    for manifest in c.read(manifest_path):
        name = manifest["condition"]
        pools[name] = pool_data(
            manifest["languages"]["en"]["example_ids"], records, index, small
        )
    prepared = {}
    jobs = []
    counts = {}
    for name, data in pools.items():
        path = BASE / f"train/{name}.json"
        save_new(path, data)
        prepared[c.rel(path)] = c.sha(path)
        counts[name] = {h: len(data["heads"][h]["rows"]) for h in old.HEADS}
        for seed in old.SEEDS:
            jobs.append(
                dict(
                    id=f"{name}_fold{seed}",
                    kind="pool",
                    pool=name,
                    seed=seed,
                    train=c.rel(path),
                )
            )
    eligible = sorted(
        (
            rid
            for rid, r in records.items()
            if r["meta"].get("gate") is True and r["meta"].get("hate_types") is not None
        )
    )
    labels = small["heads"]["hate_type"]["labels"]
    fixed_clusters = {
        r["cluster"]: i for i, r in enumerate(small["heads"]["hate"]["rows"])
    }

    def type_rows(ids):
        out = []
        for rid in sorted(ids):
            r = records[rid]
            value = r["meta"]["hate_types"]
            c.require(set(value) <= set(labels), "Unknown type annotation")
            out.append(
                dict(
                    id=rid,
                    vector_index=index[rid],
                    cluster=c.signature(old.norm(r["text"])),
                    target=[int(lab in value) for lab in labels],
                    source=r["source"],
                )
            )
        return out

    tasks = []
    for draw in DRAW_SEEDS:
        selected = sorted(
            eligible, key=lambda rid: rank("astra-type-equal67-v1", draw, rid)
        )[:67]
        tasks.append((f"equal_type67_s{draw}", "equal_type67", type_rows(selected)))
    original_types = type_rows([r["id"] for r in small["heads"]["hate_type"]["rows"]])
    for draw in SHUFFLE_SEEDS:
        tasks.append(
            (
                f"shuffled_types_s{draw}",
                "shuffled_types",
                permute_blocks(original_types, draw),
            )
        )
    for name, kind, rows in tasks:
        folds = {}
        for seed in old.SEEDS:
            folds[str(seed)] = [
                (
                    small["folds"][str(seed)][fixed_clusters[r["cluster"]]]
                    if r["cluster"] in fixed_clusters
                    else int(
                        rank("astra-type-extra-fold-v1", seed, r["cluster"])[0], 16
                    )
                    % 5
                )
                for r in rows
            ]
        data = {
            "labels": labels,
            "rows": rows,
            "folds": folds,
            "gate_policy": "Fixed original compact BGE gate. Original-pool training texts use original held-fold gate probabilities for selection; novel training texts use the fixed gate trained without those texts.",
        }
        path = BASE / f"train/{name}.json"
        save_new(path, data)
        prepared[c.rel(path)] = c.sha(path)
        counts[name] = {
            "gate_annotations": 147,
            "type_annotations": 67,
            "distinct_training_id_union": len(
                set(small["gate_ids"]) | {r["id"] for r in rows}
            ),
            "changed_label_blocks": sum(
                (r.get("label_donor_id", r["id"]) != r["id"] for r in rows)
            ),
            "actually_changed_label_values": (
                sum(
                    (
                        r["target"]
                        != next(
                            (q["target"] for q in original_types if q["id"] == r["id"])
                        )
                        for r in rows
                    )
                )
                if kind == "shuffled_types"
                else None
            ),
        }
        for seed in old.SEEDS:
            jobs.append(
                dict(
                    id=f"{name}_fold{seed}",
                    kind=kind,
                    pool=name,
                    seed=seed,
                    train=c.rel(path),
                )
            )
    files = [
        Path(__file__),
        Path(linear.__file__),
        Path(old.__file__),
        Path(c.__file__),
        c.ROOT / "tests/training/frozen_features/test_pools.py",
        c.ROOT / "analysis/common/replay.py",
        c.ROOT / "analysis/common/scoring.py",
        old.BASE / "plan.json",
        old.BASE / "train.json",
        old.BASE / "evaluation_inputs.json",
        old.BASE / "evaluation_vectors.npy",
        c.CORE / "kb/full/records.jsonl",
        c.CORE / "retrieval_v1/full_vectors.npy",
        manifest_path,
        POOLS / "scientific_plan.json",
    ]
    files += [old.BASE / f"fit_seed{seed}.json" for seed in old.SEEDS]
    files += [
        c.ROOT / f"data/processed/{name}.parquet" for name in original["endpoints"]
    ]
    hashes = {c.rel(p): c.sha(p) for p in files}
    for key in (
        "experiments/astra/full_kb_core_20260915_v1/kb/full/records.jsonl",
        "experiments/astra/full_kb_core_20260915_v1/retrieval_v1/full_vectors.npy",
    ):
        c.require(
            hashes[key] == original["inputs"][key],
            "Original cached training source changed",
        )
    c.require(len(jobs) == 75, "Unexpected job count")
    save_new(
        BASE / "plan.json",
        {
            "created_utc": c.utc(),
            "status": "frozen_before_pool_fitting",
            "inputs": hashes,
            "prepared": prepared,
            "environment": old.environment(),
            "jobs": jobs,
            "training_counts": counts,
            "endpoints": original["endpoints"],
            "methods": "Original compact BGE TRAIN-only grid, balanced L2 logistic heads, group-exclusive folds, gating and final refit. Existing curated + nine random pool draws. Three fold assignments within each pool are not independent datasets.",
            "equal_annotation": "Five hash-random TRAIN-only draws of 67 type-annotated texts. Fixed original hate gate, same 147 gate annotations. Report union of training texts and annotation budgets; this is not a matched 147-text RAG comparison.",
            "negative_control": "Ten declared permutations of complete 67-row type-label blocks, preserving co-label distribution and feature texts. Refit/reselect type head on each shuffled TRAIN pool, keep gate fixed. Show all draws; descriptive sanity control, not an exact permutation p-value or model selected by these outcomes.",
            "sparse_policy": "Shared helper: empty held folds skipped; empty fitting folds zero probabilities; one-class fits constants. If no selection label has >=2 positives and >=2 negatives, retain C=1/.5 and mark fallback. Untrained fine heads empty/None. Missing annotations never converted to negatives; no redrawing poor pools.",
            "reporting": "All five original English endpoints for pool recipes; type endpoints only for equal-annotation and shuffle tasks. Record every pool/fold, class support, zero-support labels, source mix and timing. Type head and delivered (gated) scores both descriptive. No winner selection, new main-model policy, unadjusted superiority claims or general curation claim.",
            "scope": "Core exposed DEV now. Wider pool inference and RAG comparisons require a separate execution/scoring amendment; no fresh Qwen work in this plan.",
        },
    )
    print(
        "75 CPU jobs frozen; exact training inventories and sparse-label rules recorded.",
        flush=True,
    )


def validate():
    p = c.read(BASE / "plan.json")
    c.verify(p["inputs"])
    c.verify(p["prepared"])
    c.require(p["environment"] == old.environment(), "Changed scientific environment")
    return p


def fit_type(data, x, seed, gate_oof):
    from sklearn.metrics import f1_score

    y = np.asarray([r["target"] for r in data["rows"]], dtype=int)
    fold = np.asarray(data["folds"][str(seed)])
    supported = np.flatnonzero((y.sum(axis=0) >= 2) & ((1 - y).sum(axis=0) >= 2))
    candidates = []
    cached = {}
    checks = 0
    for strength in old.CS:
        p = np.full(y.shape, np.nan)
        for k in range(5):
            held = fold == k
            train = ~held
            if not held.any():
                continue
            for j in range(y.shape[1]):
                model = (
                    old.binary_fit(x[train], y[train, j], strength, seed)
                    if train.any()
                    else {"constant": 0}
                )
                p[held, j] = old.binary_predict(model, x[held])
        c.require(np.isfinite(p).all(), "Missing type OOF probabilities")
        cached[str(strength)] = p
        for cutoff in old.THRESHOLDS:
            prediction = (p >= cutoff).astype(int)
            prediction[~gate_oof] = 0
            value = old.macro(y, prediction, supported) if len(supported) else None
            independent = (
                float(
                    np.mean(
                        [
                            f1_score(y[:, j], prediction[:, j], zero_division=0)
                            for j in supported
                        ]
                    )
                )
                if len(supported)
                else None
            )
            c.require(
                value == independent
                or (value is not None and abs(value - independent) < 1e-12),
                "Type OOF arithmetic mismatch",
            )
            checks += 1
            candidates.append(dict(C=strength, threshold=cutoff, score=value))
    selected = (
        old.choose(candidates)
        if len(supported)
        else dict(C=1.0, threshold=0.5, score=None)
    )
    models = [
        old.binary_fit(x, y[:, j], selected["C"], seed) for j in range(y.shape[1])
    ]
    return (
        dict(
            labels=data["labels"],
            models=models,
            selected=selected,
            candidates=candidates,
            selection_indices=supported.tolist(),
            selection_fallback=not len(supported),
            untrained=False,
            training_ids=[r["id"] for r in data["rows"]],
            oof_probabilities=cached[str(selected["C"])].tolist(),
        ),
        checks,
    )


def fit_one(job, matrix):
    seed = job["seed"]
    data = c.read(c.ROOT / job["train"])
    if job["kind"] == "pool":
        x = np.asarray(
            matrix[[r["vector_index"] for r in data["heads"]["hate"]["rows"]]],
            dtype=np.float64,
        )
        cached, counts = linear.make_oof(data, x, seed)
        heads = linear.fit_recipe(data, x, seed, cached)
        checks = linear.check_selection(data, seed, heads, cached)
        if job["pool"] == "curated_small":
            before = c.read(old.BASE / f"fit_seed{seed}.json")["heads"]
            for head in old.HEADS:
                c.require(
                    heads[head]["models"] == before[head]["models"]
                    and heads[head]["selected"] == before[head]["selected"],
                    "Curated reference differs",
                )
        return (heads, dict(candidate_score_checks=checks, oof_counts=counts))
    rows = data["rows"]
    x = np.asarray(matrix[[r["vector_index"] for r in rows]], dtype=np.float64)
    heads = copy.deepcopy(c.read(old.BASE / f"fit_seed{seed}.json")["heads"])
    gate = c.read(old.BASE / "train.json")
    lookup = {r["cluster"]: i for i, r in enumerate(gate["heads"]["hate"]["rows"])}
    probs = old.binary_predict(heads["hate"]["models"][0], x)
    for j, r in enumerate(rows):
        if r["cluster"] in lookup:
            probs[j] = heads["hate"]["oof_probabilities"][lookup[r["cluster"]]][0]
    gate_oof = probs > heads["hate"]["selected"]["threshold"]
    heads["hate_type"], checks = fit_type(data, x, seed, gate_oof)
    return (
        heads,
        dict(
            candidate_score_checks=checks,
            gate_train_oof_rows=sum((r["cluster"] in lookup for r in rows)),
            gate_frozen_external_rows=sum((r["cluster"] not in lookup for r in rows)),
        ),
    )


def work(job, plan):
    from analysis.common import replay as s

    root = BASE / "jobs" / job["id"]
    done = root / "complete.json"
    if done.exists():
        receipt = c.read(done)
        c.require(
            receipt["plan_sha256"] == c.sha(BASE / "plan.json"),
            "Saved job uses different plan",
        )
        c.verify(receipt["outputs"])
        print("Verified complete", job["id"], flush=True)
        return receipt
    start = time.perf_counter()
    fit_path = root / "fit.json"
    fit_receipt = root / "fit_complete.json"
    if fit_path.exists():
        c.require(
            fit_receipt.exists(), "Interrupted fit publication; preserve for inspection"
        )
        c.verify(c.read(fit_receipt)["outputs"])
        saved = c.read(fit_path)
        c.require(
            saved["plan_sha256"] == c.sha(BASE / "plan.json") and saved["job"] == job,
            "Saved fit identity differs",
        )
    else:
        matrix = np.load(
            c.CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r", allow_pickle=False
        )
        heads, checks = fit_one(job, matrix)
        saved = dict(
            plan_sha256=c.sha(BASE / "plan.json"), job=job, heads=heads, checks=checks
        )
        save_new(fit_path, saved)
        save_new(fit_receipt, {"outputs": {c.rel(fit_path): c.sha(fit_path)}})
    x = np.load(old.BASE / "evaluation_vectors.npy", allow_pickle=False)
    inputs = c.read(old.BASE / "evaluation_inputs.json")
    probs = linear.probabilities(saved["heads"], x)
    pred = linear.decode(saved["heads"], probs)
    ungated = linear.decode(saved["heads"], probs, ungated=True)
    predictions = {
        r["item_id"]: {"result": p, "ungated_diagnostic": u}
        for r, p, u in zip(inputs, pred, ungated)
    }
    pred_path = root / "predictions.jsonl"
    linear.save_rows(pred_path, [dict(item_id=k, **v) for k, v in predictions.items()])
    scores = {}
    for name, ep in plan["endpoints"].items():
        if job["kind"] != "pool" and ep["dimension"] != "hate_type":
            continue
        gold = s.gold_rows(name)
        keys = [str(r["id"]) for r in gold]
        scores[name] = {
            mode: s.summarize(
                s.contributions(gold, {k: predictions[k][mode] for k in keys}, ep)
            )
            for mode in ("result", "ungated_diagnostic")
        }
    output = root / "scores.json"
    save_new(
        output, dict(job=job, endpoints=scores, plan_sha256=c.sha(BASE / "plan.json"))
    )
    receipt = dict(
        plan_sha256=c.sha(BASE / "plan.json"),
        job=job["id"],
        status="fit_predictions_scores_verified",
        seconds_this_invocation=time.perf_counter() - start,
        checks=saved["checks"],
        outputs={
            c.rel(p): c.sha(p) for p in (fit_path, fit_receipt, pred_path, output)
        },
    )
    save_new(done, receipt)
    print(
        "Completed",
        job["id"],
        "seconds",
        round(receipt["seconds_this_invocation"], 1),
        flush=True,
    )
    return receipt


def summarize(plan):
    records = []
    outputs = {}
    for job in plan["jobs"]:
        path = BASE / "jobs" / job["id"] / "complete.json"
        receipt = c.read(path)
        c.require(
            receipt["plan_sha256"] == c.sha(BASE / "plan.json")
            and receipt["job"] == job["id"],
            "Wrong completion identity",
        )
        c.verify(receipt["outputs"])
        outputs[c.rel(path)] = c.sha(path)
        records.append(c.read(BASE / "jobs" / job["id"] / "scores.json"))
    groups = {}
    for record in records:
        for name, value in record["endpoints"].items():
            key = record["job"]["pool"] + "/" + name
            groups.setdefault(key, []).append(
                {
                    "seed": record["job"]["seed"],
                    "macro_f1": value["result"]["macro_f1_valid_only"],
                    "ungated_macro_f1": value["ungated_diagnostic"][
                        "macro_f1_valid_only"
                    ],
                }
            )
    result = {
        key: dict(
            runs=values,
            mean=float(np.mean([r["macro_f1"] for r in values])),
            sample_sd=float(np.std([r["macro_f1"] for r in values], ddof=1)),
        )
        for key, values in groups.items()
    }
    c.verify(plan["inputs"])
    c.verify(plan["prepared"])
    save_new(
        BASE / "results.json",
        {
            "created_utc": c.utc(),
            "status": "all_75_jobs_complete",
            "plan_sha256": c.sha(BASE / "plan.json"),
            "completion_receipts": outputs,
            "groups": result,
            "training_counts": plan["training_counts"],
            "interpretation": "Descriptive exposed-DEV learning curves and controls. Pool draws, validation folds and evaluation texts are distinct sources of variability. No inferential p-value or selected winning system.",
            "Qwen_calls": 0,
            "CUDA_used": False,
            "new_embeddings": 0,
        },
    )
    print("All 75 CPU jobs complete; summary saved.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["prepare", "work", "run", "summarize"])
    p.add_argument("--job")
    args = p.parse_args()
    with c.offline():
        if args.mode == "prepare":
            prepare()
        else:
            plan = validate()
            if args.mode == "summarize":
                summarize(plan)
            else:
                jobs = [
                    j for j in plan["jobs"] if args.mode == "run" or j["id"] == args.job
                ]
                c.require(jobs, "Unknown job")
                for job in jobs:
                    work(job, plan)
                if args.mode == "run":
                    summarize(plan)

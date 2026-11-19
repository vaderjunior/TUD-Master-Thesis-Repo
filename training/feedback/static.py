import os

for _n in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_n] = "1"
import argparse
import ast
from collections import Counter
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import unittest
import uuid
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.common import runtime as c
from training.cpu import common as original
from training.cpu import memory as memory

BASE = c.LINEAR
SCOPE = c.SHARED / "scope"
GROUP = c.CPU / "groups/full_bge_s42/hate"
SOURCES = [
    "training/feedback/static.py",
    "tests/feedback/classifiers/test_static_update.py",
    "analysis/common/runtime.py",
    "training/cpu/common.py",
    "training/cpu/memory.py",
    "scripts/run_classifier_feedback.ps1",
]


def engine(root=BASE):
    c.own(root / "jobs")
    ns = {**vars(original), "BASE": root}
    tree = ast.parse(Path(original.__file__).read_text(encoding="utf8"))
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), original.__file__, "exec"),
        ns,
    )
    return SimpleNamespace(**ns)


def monitor(cpu):
    ns = {**vars(memory), "c": cpu, "BASE": cpu.BASE}
    names = {"stop_reason", "terminate_owned", "monitored_command"}
    tree = ast.parse(Path(memory.__file__).read_text(encoding="utf8"))
    functions = [
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), memory.__file__, "exec"),
        ns,
    )
    return ns["monitored_command"]


def old_head():
    receipt = c.read(GROUP / "result/complete.json")
    c.verify(
        {c.rel(GROUP / "result" / name): h for name, h in receipt["outputs"].items()}
    )
    path = GROUP / "result" / receipt["attempt"] / "head.json"
    head = c.read(path)
    c.verify(head["dependencies"])
    for name in head["dependencies"]:
        dependency = c.ROOT / name
        d = c.read(dependency)
        c.verify({c.rel(dependency.parent / n): h for n, h in d["outputs"].items()})
    c.require(
        head["selection"]["selected"]["C"] == 10
        and head["selection"]["selected"]["threshold"] == 0.6,
        "Do not change the selected recipe",
    )
    return (head, path)


def validate():
    plan = c.read(BASE / "plan.json")
    c.verify(plan["inputs"])
    c.verify(plan["outputs"])
    c.verify(plan["source_hashes"])
    c.require(
        plan["environment"] == original.environment(), "A18 numeric environment differs"
    )
    c.active_guard()
    return plan


def prepare():
    if (BASE / "plan.json").exists():
        validate()
        return
    c.resources(2.0, 3.0)
    c.active_guard()
    cpu = engine()
    parent = c.read(c.CPU / "plan.json")
    c.require(
        parent["environment"] == original.environment(), "Saved CPU environment differs"
    )
    head, hpath = old_head()
    consumed = [
        c.CPU / "data/full_heads.json",
        c.CPU / "data/evaluation.json",
        c.CPU / "data/evaluation_vectors.npy",
        c.CPU / "data/train_texts.jsonl",
    ]
    for p in consumed:
        c.require(c.sha(p) == parent["prepared"][c.rel(p)], "Consumed CPU data changed")
    fullvec = c.CORE / "retrieval_v1/full_vectors.npy"
    c.require(
        c.sha(fullvec) == parent["inputs"][c.rel(fullvec)], "TRAIN vectors changed"
    )
    scope = c.read(SCOPE / "complete.json")
    c.verify(scope["outputs"])
    inherited = c.read(SCOPE / "plan.json")["inputs"]
    for p in (
        c.CANDIDATES / "roster.jsonl",
        c.CANDIDATES / "query_vectors.npy",
        c.SELECTION / "actual_pools.json",
    ):
        c.require(
            c.sha(p) == inherited[c.rel(p)], "Shared-scope source binding differs"
        )
    spec = c.read(c.CPU / "data/full_heads.json")["hate"]
    train = spec["train"]
    val = spec["val"]
    c.require((len(train), len(val)) == (50173, 5580), "Original partitions changed")
    texts = {r["id"]: r for r in c.rows(c.CPU / "data/train_texts.jsonl")}
    sv = np.load(fullvec, mmap_mode="r")
    candidates = list(c.rows(c.CANDIDATES / "roster.jsonl"))
    candidate_vectors = np.load(c.CANDIDATES / "query_vectors.npy", mmap_mode="r")
    c.require(
        not {r["source_id"] for r in train + val} & {r["id"] for r in candidates},
        "Feedback source is already in TRAIN",
    )
    for r in train + val:
        c.require(
            c.text_sha(texts[r["id"]]["text"]) == r["text_sha256"]
            and type(r["target"]) is int
            and (r["target"] in (0, 1)),
            "TRAIN identity/target differs",
        )
    bindings = c.bindings(
        [
            c.CPU / "plan.json",
            hpath,
            GROUP / "result/complete.json",
            *consumed,
            fullvec,
            SCOPE / "plan.json",
            SCOPE / "complete.json",
            SCOPE / "evaluation.jsonl",
            SCOPE / "vectors.npy",
            SCOPE / "sensitivity.json",
            c.A15 / "roster.jsonl",
            c.CANDIDATES / "roster.jsonl",
            c.CANDIDATES / "query_vectors.npy",
            c.SELECTION / "actual_pools.json",
            BASE / "STUDY_PROTOCOL.md",
        ]
    )
    for variant, fit in [("matched_gradient", train), ("all_permitted", train + val)]:
        paths = head["variants"][variant]
        folder = c.ROOT / paths["features"]
        stored = np.load(folder / "fit.npy", mmap_mode="r")
        c.require(stored.shape == (len(fit), 1024), "Stored fit matrix shape differs")
        for start in range(0, len(fit), 256):
            part = np.asarray(
                sv[[r["vector_index"] for r in fit[start : start + 256]]],
                dtype=np.float64,
            )
            c.require(
                np.array_equal(stored[start : start + len(part)], part),
                "Original feature order/values differ",
            )
        receipt = c.read(folder.parent.parent / "complete.json")
        c.verify(
            {c.rel(folder.parent.parent / n): h for n, h in receipt["outputs"].items()}
        )
        bindings.update(
            c.bindings(
                [
                    c.ROOT / paths["model"],
                    c.ROOT / paths["probabilities"],
                    folder / "fit.npy",
                ]
            )
        )
    norm = np.linalg.norm(np.asarray(candidate_vectors, dtype=np.float64), axis=1)
    c.require(
        candidate_vectors.shape == (1024, 1024)
        and np.isfinite(candidate_vectors).all()
        and np.all(abs(norm - 1) < 2e-06),
        "Feedback representations differ",
    )
    evaluation = list(c.rows(SCOPE / "evaluation.jsonl"))
    dev = list(c.rows(c.A15 / "roster.jsonl"))
    roster = [
        {
            "key": r["key"],
            "id": r["id"],
            "text": r["text"],
            "population": r["phase"],
            "gate": r["gate"],
            "inference_cluster": r["inference_cluster"],
        }
        for r in evaluation
    ]
    roster.extend(
        (
            {
                "key": r["key"],
                "id": r["id"],
                "text": r["text"],
                "population": "a15_dev",
                "gate": r["gold"]["hate"],
                "inference_cluster": r["cluster"],
            }
            for r in dev
        )
    )
    c.require(
        len(roster) == len({r["key"] for r in roster}) == 16433,
        "A18 evaluation union differs",
    )
    oldeval = c.read(c.CPU / "data/evaluation.json")
    positions = {r["key"]: i for i, r in enumerate(oldeval)}
    ev = np.load(c.CPU / "data/evaluation_vectors.npy", mmap_mode="r")
    indices = []
    for r in roster:
        i = positions[r["key"]]
        c.require(
            r["text"] == oldeval[i]["text"]
            and c.text_sha(r["text"]) == oldeval[i]["exact_text_sha256"],
            "Evaluation identity/text differs",
        )
        indices.append(i)
    shared = np.load(SCOPE / "vectors.npy", mmap_mode="r")
    for start in range(0, len(evaluation), 256):
        part = evaluation[start : start + 256]
        c.require(
            np.array_equal(
                ev[indices[start : start + len(part)]],
                shared[[r["scope_node_index"] for r in part]],
            ),
            "Transfer/classifier features differ",
        )
    pools = c.read(c.SELECTION / "actual_pools.json")["pools"]
    c.require(
        len(pools) == 18 and all((len(p["ids"]) == 40 for p in pools)),
        "Static correction pools differ",
    )
    c.jsonlines(BASE / "roster.jsonl", roster)
    c.save(
        BASE / "data.json",
        {
            "evaluation_indices": indices,
            "static_pools": pools,
            "baseline_head": head,
            "partitions": {"matched_gradient": train, "all_permitted": train + val},
            "candidate_ids": [r["id"] for r in candidates],
            "candidate_labels": [int(r["gold"]["hate"]) for r in candidates],
        },
    )
    from tests.feedback.classifiers import test_static_update as tests

    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests)
    )
    c.require(result.wasSuccessful(), "A18 tests failed")
    c.save(
        BASE / "plan.json",
        {
            "created_utc": c.utc(),
            "inputs": bindings,
            "outputs": c.bindings((BASE / n for n in ["roster.jsonl", "data.json"])),
            "source_hashes": {n: c.sha(c.ROOT / n) for n in SOURCES},
            "environment": original.environment(),
            "principal_variant": "all_permitted",
            "variants": ["all_permitted", "matched_gradient"],
            "rows": 16433,
            "static_logical_fits": 38,
            "iterative_additional_logical_fits_max": 72,
            "C": 10.0,
            "cutoff": 0.6,
            "cutoff_operator": ">",
            "seed": 42,
            "tests": result.testsRun,
            "full_baseline_reproduction": "required_first_at_execution",
            "bootstrap_draws": 10000,
            "primary_confidence": 0.9875,
            "bootstrap_seeds": list(range(2026101224, 2026101228)),
            "actual_API_calls": 0,
            "new_scientific_fits": 0,
            "resource_policy": {
                "initial_ram_GiB": 3.5,
                "reserve_GiB": 1.25,
                "max_private_GiB": 2.6,
            },
        },
    )
    c.save(
        BASE / "ready.json",
        {
            "status": "offline_preparation_and_synthetic_fits_verified",
            "plan_sha256": c.sha(BASE / "plan.json"),
            "actual_API_calls": 0,
            "scientific_fits": 0,
        },
    )
    validate()
    print(
        "A18 inputs, original matrices and synthetic refit/restart checks pass. No scientific fit yet.",
        flush=True,
    )


def job_identity(variant, ids):
    c.require(
        variant in ("all_permitted", "matched_gradient")
        and len(ids) == len(set(ids)) <= 40,
        "Invalid classifier update",
    )
    return {
        "plan_sha256": c.sha(BASE / "plan.json"),
        "variant": variant,
        "source_ids": sorted(ids),
        "C": 10.0,
        "seed": 42,
    }


def job_folder(variant, ids):
    return BASE / "jobs" / variant / c.sig(job_identity(variant, ids))[:16]


def independent_prob(model, x):
    if model["constants"][0] >= 0:
        return np.full((len(x), 1), model["constants"][0], dtype=float)
    z = (
        np.einsum("ij,j->i", x, model["coef"][0], optimize=False)
        + model["intercept"][0]
    )
    return (1 / (1 + np.exp(-np.clip(z, -700, 700))))[:, None]


def worker(variant, ids):
    validate()
    cpu = engine()
    identity = job_identity(variant, ids)
    directory = job_folder(variant, ids)
    if cpu.inspect_task(directory, identity):
        print("Verified CPU update reused", variant, len(ids))
        return
    if ids:
        c.require(
            cpu.inspect_task(job_folder(variant, []), job_identity(variant, []))
            is not None,
            "Reproduce this original baseline before any updated fit",
        )
    data = c.read(BASE / "data.json")
    fit = data["partitions"][variant]
    mapping = {i: n for n, i in enumerate(data["candidate_ids"])}
    c.require(
        set(ids) <= set(mapping) and (not set(ids) & {r["source_id"] for r in fit}),
        "Wrong or reused update source",
    )

    def perform(attempt):
        started = time.perf_counter()
        raw = np.load(c.CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r")
        x = np.empty((len(fit) + len(ids), 1024), dtype=np.float64)
        for start in range(0, len(fit), 256):
            x[start : start + min(256, len(fit) - start)] = raw[
                [r["vector_index"] for r in fit[start : start + 256]]
            ]
        feedback = np.load(c.CANDIDATES / "query_vectors.npy", mmap_mode="r")
        ordered = sorted(ids)
        if ids:
            x[len(fit) :] = feedback[[mapping[i] for i in ordered]]
        y = np.asarray(
            [r["target"] for r in fit]
            + [data["candidate_labels"][mapping[i]] for i in ordered],
            dtype=int,
        )[:, None]
        loaded = time.perf_counter()
        model = cpu.fit_linear(x, y, 10.0, 42)
        fitted = time.perf_counter()
        del x
        cpu.arrays(attempt / "model.npz", model)
        restored = cpu.load_arrays(attempt / "model.npz")
        c.require(
            all((np.array_equal(model[k], restored[k]) for k in model)),
            "Reloaded model differs",
        )
        if not ids:
            prior = cpu.load_arrays(
                c.ROOT / data["baseline_head"]["variants"][variant]["model"]
            )
            c.require(
                all(
                    (
                        np.allclose(restored[k], prior[k], rtol=0, atol=1e-10)
                        for k in model
                    )
                ),
                "Original classifier refit differs; stop before updates",
            )
        ev = np.load(c.CPU / "data/evaluation_vectors.npy", mmap_mode="r")
        out = []
        maxdiff = 0.0
        for start in range(0, len(data["evaluation_indices"]), 256):
            xx = np.asarray(
                ev[data["evaluation_indices"][start : start + 256]], dtype=np.float64
            )
            prob = cpu.predict_linear(restored, xx)
            check = independent_prob(restored, xx)
            maxdiff = max(maxdiff, float(np.max(abs(prob - check))))
            c.require(
                np.allclose(prob, check, rtol=0, atol=1e-12),
                "Independent updated probabilities differ",
            )
            for cutoff in (0.5, 0.6):
                c.require(
                    np.array_equal(prob > cutoff, check > cutoff),
                    "Numerical drift changes gate decisions",
                )
            out.append(prob)
        prob = np.concatenate(out)
        if not ids:
            old = np.load(
                c.ROOT / data["baseline_head"]["variants"][variant]["probabilities"],
                mmap_mode="r",
            )[data["evaluation_indices"]]
            c.require(
                np.allclose(prob, old, rtol=0, atol=1e-12)
                and all(
                    (
                        np.array_equal(prob > cutoff, old > cutoff)
                        for cutoff in (0.5, 0.6)
                    )
                ),
                "Saved baseline predictions differ",
            )
        cpu.array(attempt / "probabilities.npy", prob)
        cpu.array(attempt / "decisions_06.npy", (prob > 0.6).astype(np.uint8))
        cpu.array(attempt / "decisions_05.npy", (prob > 0.5).astype(np.uint8))
        end = time.perf_counter()
        cpu.save(
            attempt / "attribution.json",
            {
                "identity": identity,
                "source_ids": ordered,
                "fit_rows": len(y),
                "positive": int(y.sum()),
                "negative": int(len(y) - y.sum()),
                "feature_seconds": loaded - started,
                "fit_seconds": fitted - loaded,
                "inference_and_verification_seconds": end - fitted,
                "independent_max_abs_difference": maxdiff,
                "baseline_reproduced": not ids,
                "probability_rows": len(prob),
                "iterations": model["iterations"].tolist(),
                "class_weight_recomputed": True,
                "fine_heads_changed": False,
            },
        )
        return {
            "rows": len(prob),
            "source_ids": ordered,
            "baseline_reproduced": not ids,
            "no_Qwen_or_CUDA": True,
        }

    cpu.task(directory, identity, perform)
    print("CPU update complete", variant, len(ids), flush=True)


def tasks(mode):
    data = c.read(BASE / "data.json")
    entries = [{"condition": "reference", "ids": []}] + [
        {"condition": p["condition"], "ids": p["ids"]} for p in data["static_pools"]
    ]
    deps = {}
    if mode == "iterative":
        donepath = c.ITERATIVE / "complete.json"
        c.require(
            donepath.exists(),
            "Finish and review A17 before iterative classifier updates",
        )
        done = c.read(donepath)
        c.require(
            done["status"]
            == "all_four_simulated_rounds_audited_scientific_review_pending"
            and done["plan_sha256"] == c.sha(c.ITERATIVE / "plan.json"),
            "Wrong A17 completion",
        )
        c.verify(done["phase_receipts"])
        deps[c.rel(donepath)] = c.sha(donepath)
        entries = [{"condition": "reference", "ids": []}]
        for round_no in range(1, 5):
            for seed in c.STREAM_SEEDS:
                p = c.ITERATIVE / f"rounds/r{round_no}/selection_s{seed}.json"
                d = c.read(p)
                c.verify(d["source_bindings"])
                deps[c.rel(p)] = c.sha(p)
                for policy, ids in d["cumulative"].items():
                    entries.append(
                        {"condition": f"r{round_no}_{policy}_s{seed}", "ids": ids}
                    )
    return (entries, deps)


def run(mode):
    validate()
    cpu = engine()
    c.resources(3.5, 5.0)
    entries, deps = tasks(mode)
    plan = c.sha(BASE / "plan.json")
    receipt_path = BASE / f"{mode}_schedule.json"
    c.save(
        receipt_path,
        {
            "mode": mode,
            "entries": entries,
            "source_bindings": deps,
            "plan_sha256": plan,
        },
    )
    with cpu.file_lock(BASE / "cpu_launcher.lock"):
        monitored = monitor(cpu)
        jobs = {}
        total = len(entries) * 2
        ordered = [(v, entries[0]) for v in ("all_permitted", "matched_gradient")] + [
            (v, e) for e in entries[1:] for v in ("all_permitted", "matched_gradient")
        ]
        for number, (variant, entry) in enumerate(ordered, 1):
            ids = entry["ids"]
            identity = job_identity(variant, ids)
            directory = job_folder(variant, ids)
            if cpu.inspect_task(directory, identity) is None:
                command = [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-B",
                    "-u",
                    "-m",
                    __spec__.name,
                    "worker",
                    "--variant",
                    variant,
                    "--ids",
                    ",".join(sorted(ids)),
                ]
                result = monitored(
                    command,
                    BASE
                    / "resource_checks"
                    / f"{time.time_ns()}_{uuid.uuid4().hex[:6]}",
                )
            receipt = cpu.inspect_task(directory, identity)
            c.require(receipt is not None, "Missing verified CPU job")
            jobs[variant + "/" + entry["condition"]] = {
                "receipt": c.rel(directory / "complete.json"),
                "sha256": c.sha(directory / "complete.json"),
                "source_ids": sorted(ids),
            }
            print(
                f"A18 {mode}: {number}/{total} logical jobs verified; {variant}/{entry['condition']}",
                flush=True,
            )
        c.verify(deps)
        validate()
        c.save(
            BASE / f"{mode}_complete.json",
            {
                "status": "classifier_updates_verified_scientific_review_pending",
                "plan_sha256": plan,
                "jobs": jobs,
                "unique_fits": len({v["receipt"] for v in jobs.values()}),
                "logical_jobs": len(jobs),
                "schedule_sha256": c.sha(receipt_path),
                "source_bindings": deps,
                "new_Qwen_calls": 0,
                "CUDA": False,
            },
        )
    print(
        "A18 CPU UPDATE BLOCK COMPLETE. Tell Astra to inspect; no next block or SQ2 training started.",
        flush=True,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "mode", choices=["prepare", "validate", "static", "iterative", "worker"]
    )
    ap.add_argument("--variant")
    ap.add_argument("--ids", default="")
    args = ap.parse_args()
    with c.offline(), threadpool_limits(limits=1):
        if args.mode == "prepare":
            prepare()
        elif args.mode == "validate":
            validate()
            print("A18 frozen inputs verified.")
        elif args.mode == "worker":
            worker(args.variant, [v for v in args.ids.split(",") if v])
        else:
            run(args.mode)


if __name__ == "__main__":
    main()

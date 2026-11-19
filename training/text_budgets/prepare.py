import argparse
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math
import shutil
from training.text_budgets import common as b
from training.frozen_features import roberta as feature
from training.frozen_features import compact as linear_old
import numpy as np

c, BASE = b.c, b.BASE


def prepare():
    if (BASE / "draft.json").exists():
        validate_draft()
        print("Existing Stage B draft verified.")
        return
    prior = c.read(b.PREP / "complete.json")
    c.verify(prior["source_bindings"])
    c.verify(prior["artifacts"])
    previous = c.read(b.PREP / "preparation.json")
    bindings = dict(prior["source_bindings"])

    def bind(path, expected=None):
        path = Path(path)
        actual = c.sha(path)
        c.require(
            expected is None or actual == expected,
            "Prepared input binding differs: " + str(path),
        )
        bindings[c.rel(path)] = actual
        return path

    baseplan = c.read(bind(b.short.BASE / "plan.json"))
    cache = baseplan["base_cache"]
    for name, digest in cache["files"].items():
        c.require(
            c.sha(Path(cache["snapshot"]) / name) == digest,
            "Pinned base checkpoint differs",
        )
    inputs = c.read(
        bind(
            b.short.BASE / "data/inference.json",
            baseplan["prepared"][c.rel(b.short.BASE / "data/inference.json")],
        )
    )
    scopes = c.read(
        bind(
            b.short.BASE / "data/scopes.json",
            baseplan["prepared"][c.rel(b.short.BASE / "data/scopes.json")],
        )
    )
    c.require(
        len(inputs) == len({x["key"] for x in inputs}) == 19082,
        "Evaluation identity count",
    )
    evaluation_clusters = {linear_old.norm(x["text"]) for x in inputs}
    train_inputs = c.read(bind(b.PREP / "TRAIN_feature_inputs.json"))
    records = {x["key"][6:]: x for x in train_inputs}
    c.require(
        len(records) == 13736
        and not {linear_old.norm(x["text"]) for x in train_inputs}
        & evaluation_clusters,
        "TRAIN/evaluation overlap",
    )
    old_plan = c.read(b.POOLS / "plan.json")
    manifests = {}
    summaries = []
    for pool in b.NAMES:
        path = b.POOLS / "train" / f"{pool}.json"
        bind(path, old_plan["prepared"][c.rel(path)])
        data = c.read(path)
        data["texts"] = [records[k]["text"] for k in data["gate_ids"]]
        c.require(len(data["texts"]) == len(set(data["gate_ids"])), "TRAIN identity")
        for head in c.HEADS:
            spec = data["heads"][head]
            y = c.truth(head, spec["rows"], spec["labels"])
            c.require(
                ((y.sum(0) >= 2) & ((1 - y).sum(0) >= 2)).any(),
                "Unexpected unsupported selection head",
            )
            for row in spec["rows"]:
                c.require(
                    data["gate_ids"][row["feature_row"]] == row["id"]
                    and c.signature(data["texts"][row["feature_row"]])
                    == row["text_hash"],
                    "TRAIN text alignment",
                )
        b.save(BASE / "data/train" / f"{pool}.json", data)
        summaries.append(
            {
                "pool": pool,
                "texts": len(data["texts"]),
                "annotations": {h: len(data["heads"][h]["rows"]) for h in c.HEADS},
            }
        )
    for grid in ("full", "one_rate"):
        values = []
        state = Counter()
        by_head = defaultdict(lambda: Counter())
        for pool in b.NAMES:
            data = b.data(pool)
            with b.isolated(grid):
                for job in b.jobs(pool, grid):
                    status, total = b.expected_state(data, job)
                    spec, fit, held = b.split(data, job)
                    epochs = c.config_map()[job["config"]]["epochs"]
                    row = dict(
                        job,
                        state=status,
                        fit_rows=len(fit),
                        held_rows=len(held),
                        steps=total,
                    )
                    values.append(row)
                    state[status] += 1
                    key = pool + "/" + job["head"]
                    by_head[key]["jobs"] += status != "empty_held_fold"
                    by_head[key]["steps"] += total
                    by_head[key]["epoch_logit_bytes"] += (
                        epochs * (128 + len(held) * len(spec["labels"]) * 4)
                        if held
                        else 0
                    )
                    by_head[key]["step_JSON_bytes_upper"] += total * 440
                    by_head[key]["epoch_JSON_bytes_upper"] += (
                        epochs * 500 if held else 0
                    )
        expected = (22680, 22428, 270, 252) if grid == "full" else (7560, 7476, 90, 84)
        c.require(
            (
                len(values),
                state["trained"] + state["initialization_only"],
                state["initialization_only"],
                state["empty_held_fold"],
            )
            == expected,
            "Manifest inventory",
        )
        b.save(BASE / "data" / f"jobs_{grid}.json", values)
        manifests[grid] = {
            "slots": len(values),
            "states": dict(state),
            "steps": sum(x["steps"] for x in values),
            "by_pool_head": dict(by_head),
            "raw_search_artifact_GiB_upper": sum(
                sum(
                    v[k]
                    for k in (
                        "epoch_logit_bytes",
                        "step_JSON_bytes_upper",
                        "epoch_JSON_bytes_upper",
                    )
                )
                for v in by_head.values()
            )
            / 2**30,
            "per_job_other_bytes_allowance": 20000,
        }
    for name, value in [
        ("inference.json", inputs),
        ("scopes.json", scopes),
        ("TRAIN_features.json", train_inputs),
    ]:
        b.save(BASE / "data" / name, value)

    old_features = c.read(bind(feature.BASE / "feature_inputs.json"))
    fp = c.read(bind(feature.BASE / "plan.json"))
    fr = c.read(bind(feature.BASE / "features_complete.json"))
    c.require(
        fp["base_cache"] == cache
        and fr["plan_sha256"] == c.sha(feature.BASE / "plan.json"),
        "Frozen feature model origin",
    )
    bind(
        feature.BASE / "features_mean.npy",
        fr["outputs"][c.rel(feature.BASE / "features_mean.npy")],
    )
    lookup = {v["key"]: (i, v) for i, v in enumerate(old_features)}
    for row in inputs:
        c.require(
            row["key"] in lookup and lookup[row["key"]][1]["text"] == row["text"],
            "Frozen evaluation feature identity",
        )
    rx = np.load(feature.BASE / "features_mean.npy", mmap_mode="r", allow_pickle=False)
    frozen = np.asarray(rx[[lookup[x["key"]][0] for x in inputs]])
    broad = b.ROOT / "experiments/astra/compact_broad_20261008_v1"
    vectors = (
        b.ROOT / "experiments/astra/later_evaluation_20261008_v1/retrieval_v3/vectors"
    )
    bp = c.read(bind(broad / "plan.json"))
    br = c.read(
        bind(
            broad / "prediction_inputs.json",
            bp["inputs"][c.rel(broad / "prediction_inputs.json")],
        )
    )
    vp = c.read(bind(vectors / "complete.json"))
    c.verify(vp["outputs"])
    bindings.update(vp["outputs"])
    cp = c.read(bind(linear_old.BASE / "plan.json"))
    for name in ("evaluation_inputs.json", "evaluation_vectors.npy"):
        path = linear_old.BASE / name
        bind(path, {**cp["inputs"], **cp["prepared"]}[c.rel(path)])
    core_map = {
        x["item_id"]: i
        for i, x in enumerate(c.read(linear_old.BASE / "evaluation_inputs.json"))
    }
    broad_map = {x["key"]: x["vector_index"] for x in br}
    core_ids = {x["key"]: x["source_id"] for x in scopes["core"]}
    origins = [
        json.loads(line)
        for line in (vectors / "query_origins.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    core_origin_path = c.CORE / "dev_inputs.jsonl"
    bind(core_origin_path, cp["inputs"][c.rel(core_origin_path)])
    core_origins = [
        json.loads(line)
        for line in core_origin_path.read_text(encoding="utf-8").splitlines()
    ]
    core_saved = c.read(linear_old.BASE / "evaluation_inputs.json")
    for row in inputs:
        if row["key"] in broad_map:
            origin = origins[broad_map[row["key"]]]
            c.require(
                origin["key"] == row["key"]
                and origin["exact_text_sha256"]
                == hashlib.sha256(row["text"].encode("utf-8")).hexdigest(),
                "Broader BGE feature identity",
            )
        if row["key"] in core_ids:
            original = core_saved[core_map[core_ids[row["key"]]]]
            origin = core_origins[original["source_row"]]
            c.require(
                original["key"] == origin["key"] == row["key"]
                and origin["text"] == row["text"],
                "Core BGE feature identity",
            )
    cx = np.load(
        linear_old.BASE / "evaluation_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    bx = np.load(vectors / "query_vectors.npy", mmap_mode="r", allow_pickle=False)
    bge = np.stack(
        [
            (
                cx[core_map[core_ids[x["key"]]]]
                if x["key"] in core_ids
                else bx[broad_map[x["key"]]]
            )
            for x in inputs
        ]
    )
    c.require(
        frozen.shape == (19082, 768)
        and bge.shape == (19082, 1024)
        and np.isfinite(frozen).all()
        and np.isfinite(bge).all(),
        "Evaluation feature matrices",
    )
    b.arrays(BASE / "data/evaluation_features.npz", {"bge": bge, "roberta": frozen})
    b.save(
        BASE / "data/endpoints.json",
        {"core": cp["endpoints"], "broad": bp["endpoints"]},
    )

    cache_sources = {}
    for row in train_inputs:
        cached = row["cached_feature"]
        if cached is None:
            continue
        name = cached["source"]
        if name not in cache_sources:
            path = ROOT / name
            parent = (
                path.parent
                if path.name == "feature_inputs.json" and path.parent.name != "data"
                else path.parent.parent
            )
            matrix = parent / (
                "features_mean.npy" if parent == feature.BASE else "features/mean.npy"
            )
            source_plan = c.read(parent / "plan.json")
            proof = parent / (
                "features_complete.json"
                if parent == feature.BASE
                else "features/complete.json"
            )
            c.require(
                source_plan["base_cache"] == cache,
                "Cached TRAIN representation uses different pretrained model",
            )
            done = c.read(bind(proof))
            c.require(
                done["plan_sha256"] == c.sha(parent / "plan.json"),
                "Cache completion plan",
            )
            bind(parent / "plan.json")
            bind(path)
            bind(matrix, done["outputs"][c.rel(matrix)])
            cache_sources[name] = {
                "matrix": c.rel(matrix),
                "inputs": name,
                "rows": len(c.read(path)),
            }
        origin = c.read(ROOT / name)[cached["row"]]
        c.require(
            origin["key"] == row["key"]
            and origin["text"] == row["text"]
            and origin["exact_text_sha256"] == row["exact_text_sha256"],
            "Cached TRAIN feature row differs",
        )
    b.save(BASE / "data/cache_sources.json", cache_sources)
    bindings.update(b.source_guard())
    for p in (
        b.PREP / "complete.json",
        b.PREP / "preparation.json",
        b.PROTOCOL,
        feature.tokenizer.REFERENCE,
    ):
        bind(p)
    prepared = b.hashes(p for p in (BASE / "data").rglob("*") if p.is_file())
    b.save(
        BASE / "draft.json",
        dict(
            status="prepared_scope_storage_pilot_seal_pending",
            utc=c.utc(),
            environment=c.environment(),
            inputs=bindings,
            prepared=prepared,
            base_cache=cache,
            policy=b.policy(),
            pools=summaries,
            manifests=manifests,
            maximum_final_models=540,
            maximum_weight_GiB=540 * b.WEIGHT_BYTES_MAX / 2**30,
            feature_new_rows=sum(x["needs_new_RoBERTa_feature"] for x in train_inputs),
            feature_reused_rows=sum(
                not x["needs_new_RoBERTa_feature"] for x in train_inputs
            ),
            candidate_grid_adopted=None,
            stores_adopted=None,
            scientific_fits=0,
            Qwen_calls=0,
        ),
    )
    validate_draft()
    print(
        "Prepared nine aligned pools, both prospective job grids, cached evaluation features and immutable source bindings."
    )
    print(
        {
            grid: {k: v for k, v in m.items() if k != "by_pool_head"}
            for grid, m in manifests.items()
        }
    )


ROOT = b.ROOT


def validate_draft():
    plan = c.read(BASE / "draft.json")
    c.verify(plan["inputs"])
    c.verify(plan["prepared"])
    c.require(
        plan["environment"] == c.environment() and plan["policy"] == b.policy(),
        "Environment/prospective policy changed",
    )
    for name, digest in plan["base_cache"]["files"].items():
        c.require(
            c.sha(Path(plan["base_cache"]["snapshot"]) / name) == digest,
            "Pinned cache changed",
        )
    return plan


def seal():
    draft = validate_draft()
    decision = c.read(BASE / "decision.json")
    sources = b.hashes(b.sources())
    c.require(
        decision["grid"] in ("full", "one_rate") and decision["scope_authorized"],
        "Scope decision required",
    )
    test_path = b.certificate("tests")
    pilot_path = b.certificate("pilot")
    tests = c.read(test_path)
    pilot = c.read(pilot_path)
    c.require(
        tests["passed"]
        and tests["sources"] == sources
        and pilot["passed"]
        and pilot["sources"] == sources,
        "Tests/pilot not current",
    )
    c.verify(pilot["artifacts"])
    c.verify(tests["artifacts"])
    b.validate_stores(decision["stores"])
    c.require(
        pilot["excluded_from_scientific_results"] and decision["capacity_checked"],
        "Capacity check not complete",
    )
    b.store_guard(decision["stores"], initial=True)
    b.save(
        BASE / "plan.json",
        dict(
            draft,
            status="sealed_ready_no_scientific_fits",
            grid=decision["grid"],
            stores=decision["stores"],
            inputs={
                **draft["inputs"],
                **sources,
                **tests["artifacts"],
                **pilot["artifacts"],
                **b.hashes(
                    [BASE / "draft.json", BASE / "decision.json", test_path, pilot_path]
                ),
            },
            technical_amendment=c.rel(b.AMENDMENT),
            pilot_receipt=c.rel(pilot_path),
            tests_receipt=c.rel(test_path),
            pilot_summary=pilot["summary"],
        ),
    )
    validate()
    print("Stage B sealed; scientific execution remains a user action.")


def validate():
    plan = c.read(BASE / "plan.json")
    c.verify(plan["inputs"])
    c.verify(plan["prepared"])
    c.require(
        plan["environment"] == c.environment() and plan["policy"] == b.policy(),
        "Sealed environment/policy differs",
    )
    c.require(
        plan["grid"] in ("full", "one_rate")
        and plan["technical_amendment"] == c.rel(b.AMENDMENT),
        "Sealed grid/amendment differs",
    )
    b.validate_stores(plan["stores"])
    for name, digest in plan["base_cache"]["files"].items():
        c.require(
            c.sha(Path(plan["base_cache"]["snapshot"]) / name) == digest,
            "Pinned cache differs",
        )
    return plan


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "seal", "validate"])
    args = ap.parse_args()
    with b.isolated():
        globals()[args.mode]()

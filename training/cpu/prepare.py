import argparse
import json
from itertools import zip_longest
from pathlib import Path
import time
import numpy as np
from training.cpu import common as c


def prepare():
    import pyarrow.parquet as pq

    c.require(
        not (c.BASE / "plan.json").exists(),
        "Preparation already exists; validate it instead of overwriting",
    )
    start = time.perf_counter()
    before = c.resource_guard(1.25, 2.0)
    inventory_path = (
        c.ROOT / "experiments/astra/sq2_parallel_preparation_20261011_v1/inventory.json"
    )
    inventory = c.read(inventory_path)
    c.verify(inventory["source_hashes"])
    c.verify(inventory["active_source_hashes_unchanged"])
    inputs = {}
    prepared = {}

    def bind(path, expected=None):
        digest = c.sha(path)
        if expected is not None:
            c.require(digest == expected, "Historical input changed: " + c.rel(path))
        inputs[c.rel(path)] = digest

    def output(path, value):
        c.save(path, value)
        prepared[c.rel(path)] = c.sha(path)

    bind(inventory_path)
    core = c.read(c.CORE / "plan.json")
    rp = c.read(c.CORE / "retrieval_v1/plan.json")
    compact_plan = c.read(c.COMPACT / "plan.json")
    broad_plan = c.read(c.BROAD / "plan.json")
    tuning_plan = c.read(c.TUNING / "plan.json")
    neural_plan = c.read(c.LATER / "encoders/plan.json")
    for path in (
        c.CORE / "plan.json",
        c.CORE / "retrieval_v1/plan.json",
        c.COMPACT / "plan.json",
        c.BROAD / "plan.json",
        c.TUNING / "plan.json",
        c.LATER / "encoders/plan.json",
    ):
        bind(path)
    records_path = c.CORE / "kb/full/records.jsonl"
    origins_path = c.CORE / "retrieval_v1/record_vector_sources.jsonl"
    vector_path = c.CORE / "retrieval_v1/full_vectors.npy"
    bind(records_path, compact_plan["inputs"][c.rel(records_path)])
    for path in (origins_path, vector_path):
        bind(path, rp["generated_hashes"][c.rel(path)])
    records = {}
    aliases = {}
    row_count = 0
    for i, pair in enumerate(zip_longest(c.rows(records_path), c.rows(origins_path))):
        row, origin = pair
        c.require(
            row is not None and origin is not None, "Vector-origin length differs"
        )
        c.require(
            origin["id"] == row["id"]
            and origin["text_sha256"] == c.text_sha(row["text"]),
            "Vector text/ID differs",
        )
        row_count += 1
        if row["lang"] == "en" and row["kind"] == "example":
            c.require(row["id"] not in records, "Duplicate English record")
            records[row["id"]] = dict(
                text=row["text"], vector_index=i, meta=row["meta"], source=row["source"]
            )
            for source_id in row["meta"].get(
                "astra_train_ids", [row["id"].removeprefix("ex-")]
            ):
                c.require(source_id not in aliases, "Ambiguous canonical alias")
                aliases[source_id] = row["id"]
    c.require(
        row_count == 71131 and len(records) == 55753, "Full feature membership differs"
    )
    matrix = np.load(vector_path, mmap_mode="r", allow_pickle=False)
    c.require(
        matrix.shape == (71131, 1024) and matrix.dtype == np.float32,
        "Full features malformed",
    )
    for offset in range(0, len(matrix), 512):
        block = matrix[offset : offset + 512]
        c.require(
            np.isfinite(block).all()
            and np.allclose(np.linalg.norm(block, axis=1), 1.0, atol=2e-4),
            "Invalid full features",
        )
    del matrix
    full = {}
    neural = []
    native_rows = 0
    for head in c.HEADS:
        meta_path = c.CORE / f"encoder_data/full/{head}_meta.json"
        bind(meta_path, core["generated_hashes"][c.rel(meta_path)])
        meta = c.read(meta_path)
        spec = dict(labels=meta["labels"], kind=meta["kind"], train=[], val=[])
        for part in ("train", "val"):
            path = c.CORE / f"encoder_data/full/{head}_{part}.parquet"
            bind(path, core["generated_hashes"][c.rel(path)])
            for batch in pq.ParquetFile(path).iter_batches(batch_size=1024):
                for row in batch.to_pylist():
                    source_id = str(row["id"])
                    rid = aliases[source_id.removeprefix("ex-")]
                    ref = records[rid]
                    c.require(
                        row["text"] == ref["text"],
                        "English raw text differs from cached representation",
                    )
                    value = row[c.FIELDS[head]]
                    wanted = ref["meta"][c.FIELDS[head]]
                    c.require(
                        (sorted(value) if isinstance(value, list) else value)
                        == (sorted(wanted) if isinstance(wanted, list) else wanted),
                        "Annotation differs",
                    )
                    if head != "hate":
                        c.require(
                            ref["meta"]["gate"] is True and value is not None,
                            "Missing or nonhateful fine annotation",
                        )
                    target = (
                        int(value)
                        if head == "hate"
                        else (
                            meta["labels"].index(value)
                            if head == "severity"
                            else [int(label in value) for label in meta["labels"]]
                        )
                    )
                    spec[part].append(
                        dict(
                            id=rid,
                            source_id=source_id,
                            target=target,
                            vector_index=ref["vector_index"],
                            cluster=c.cluster(row["text"]),
                            text_sha256=c.text_sha(row["text"]),
                        )
                    )
            c.require(
                len(spec[part])
                == len({r["id"] for r in spec[part]})
                == meta["counts"][part],
                "Partition size differs",
            )
        for field in ("id", "cluster"):
            c.require(
                not {r[field] for r in spec["train"]} & {r[field] for r in spec["val"]},
                "Within-head partition overlap",
            )
        eligible = {
            rid
            for rid, r in records.items()
            if head == "hate"
            or (r["meta"]["gate"] is True and r["meta"].get(c.FIELDS[head]) is not None)
        }
        c.require(
            eligible == {r["id"] for r in spec["train"] + spec["val"]},
            "Full eligible annotations not covered",
        )
        full[head] = spec
        for seed in range(3):
            job = next(
                j
                for j in neural_plan["jobs"]
                if j["arm"] == "full" and j["head"] == head and j["seed"] == seed
            )
            path = c.ROOT / job["reference_attempt"] / "internal_val_logits.jsonl"
            bind(path, neural_plan["inputs"][c.rel(path)])
            saved = list(c.rows(path))
            c.require(
                len(saved) == len(spec["val"])
                and job["meta"]["labels"] == spec["labels"],
                "Saved validation shape differs",
            )
            for row, target in zip(saved, spec["val"]):
                c.require(
                    str(row["id"]) == target["source_id"]
                    and row["target"] == target["target"],
                    "Native-validation identity/target differs",
                )
            logits = np.asarray([r["logits"] for r in saved])
            prob = c.neural_probabilities(head, logits)
            bits = c.decisions(head, prob, None if head == "severity" else 0.5)
            decision = (
                bits.argmax(axis=1).tolist()
                if head in ("hate", "severity")
                else bits.tolist()
            )
            c.require(
                decision == [r["decision"] for r in saved],
                "Default saved neural decisions differ",
            )
            native_rows += len(saved)
            neural.append(
                dict(
                    head=head,
                    seed=seed,
                    logits=c.rel(path),
                    checkpoint=job["checkpoint"],
                    checkpoint_sha256=job["checkpoint_sha256"],
                    job=job["job"],
                )
            )
    print(
        "Verified full English annotations, cache mapping and 12 saved neural-validation files.",
        flush=True,
    )
    compact_path = c.COMPACT / "train.json"
    bind(compact_path, compact_plan["prepared"][c.rel(compact_path)])
    compact = c.read(compact_path)
    for head in c.HEADS:
        spec = compact["heads"][head]
        c.require(
            spec["labels"] == full[head]["labels"], "Compact/full label order differs"
        )
        for row in spec["rows"]:
            ref = records[row["id"]]
            c.require(
                ref["vector_index"] == row["vector_index"],
                "Compact cached feature index differs",
            )
            value = ref["meta"][c.FIELDS[head]]
            expected = (
                int(value)
                if head == "hate"
                else (
                    spec["labels"].index(value)
                    if head == "severity"
                    else [int(label in value) for label in spec["labels"]]
                )
            )
            c.require(row["target"] == expected, "Compact annotation differs")
            c.require(
                compact["gate_ids"][row["feature_row"]] == row["id"],
                "Compact gate/head identity differs",
            )
            for seed in c.SEEDS:
                assignment = compact["folds"][str(seed)]
                c.require(
                    len(assignment) == 147 and set(assignment) == set(range(5)),
                    "Incomplete shared folds",
                )
                by_group = {}
                for rid, fold in zip(compact["gate_ids"], assignment):
                    group = c.cluster(records[rid]["text"])
                    c.require(
                        group not in by_group or by_group[group] == fold,
                        "Compact fold normalized-text leakage",
                    )
                    by_group[group] = fold
    eval_path = c.TUNING / "data/inference.json"
    scope_path = c.TUNING / "data/scopes.json"
    for path in (eval_path, scope_path):
        bind(path, tuning_plan["inputs"][c.rel(path)])
    evaluation = c.read(eval_path)
    scopes = c.read(scope_path)
    c.require(
        len(evaluation) == len({r["key"] for r in evaluation}) == 19082,
        "Evaluation input identity differs",
    )
    c.require(
        all(c.text_sha(r["text"]) == r["exact_text_sha256"] for r in evaluation),
        "Evaluation exact text hash differs",
    )
    train_clusters = {c.cluster(v["text"]) for v in records.values()}
    c.require(
        not train_clusters & {c.cluster(r["text"]) for r in evaluation},
        "TRAIN/evaluation normalized overlap",
    )
    for scope, expected in [("core", 970), ("broad", 18275)]:
        c.require(len(scopes[scope]) == expected, "Scope roster size differs")
        c.require(
            all(evaluation[r["position"]]["key"] == r["key"] for r in scopes[scope]),
            "Scope/input alignment differs",
        )
    core_inputs_path = c.COMPACT / "evaluation_inputs.json"
    bind(core_inputs_path, compact_plan["prepared"][c.rel(core_inputs_path)])
    core_inputs = c.read(core_inputs_path)
    core_origins_path = c.CORE / "retrieval_v1/query_vector_sources.jsonl"
    core_vectors_path = c.CORE / "retrieval_v1/query_vectors.npy"
    for path in (core_origins_path, core_vectors_path):
        bind(path, rp["generated_hashes"][c.rel(path)])
    broad_inputs_path = c.BROAD / "prediction_inputs.json"
    bind(broad_inputs_path, broad_plan["inputs"][c.rel(broad_inputs_path)])
    broad_inputs = c.read(broad_inputs_path)
    broad_origins_path = c.LATER / "retrieval_v3/vectors/query_origins.jsonl"
    broad_vectors_path = c.LATER / "retrieval_v3/vectors/query_vectors.npy"
    for path in (broad_origins_path, broad_vectors_path):
        bind(path, broad_plan["inputs"][c.rel(path)])
    core_origins = list(c.rows(core_origins_path))
    broad_origins = list(c.rows(broad_origins_path))
    destinations = {r["key"]: i for i, r in enumerate(evaluation)}
    evectors = np.full((len(evaluation), 1024), np.nan, dtype=np.float32)
    filled = set()
    overlap = 0
    for inputs_rows, origins, path, index_field, hash_field in [
        (core_inputs, core_origins, core_vectors_path, "source_row", "text_sha256"),
        (
            broad_inputs,
            broad_origins,
            broad_vectors_path,
            "vector_index",
            "exact_text_sha256",
        ),
    ]:
        source = np.load(path, mmap_mode="r", allow_pickle=False)
        for row in inputs_rows:
            index = row[index_field]
            origin = origins[index]
            i = destinations[row["key"]]
            c.require(
                origin["key"] == row["key"]
                and origin[hash_field] == evaluation[i]["exact_text_sha256"],
                "Evaluation vector provenance differs",
            )
            if i in filled:
                c.require(
                    np.array_equal(evectors[i], source[index]),
                    "Same input has different cached features between scopes; review needed",
                )
                overlap += 1
            else:
                evectors[i] = source[index]
                filled.add(i)
        del source
    c.require(
        len(filled) == 19082 and overlap == 163 and np.isfinite(evectors).all(),
        "Evaluation feature union differs",
    )
    c.require(
        np.allclose(np.linalg.norm(evectors, axis=1), 1.0, atol=2e-4),
        "Invalid evaluation feature norms",
    )
    output(c.BASE / "data/full_heads.json", full)
    output(c.BASE / "data/compact.json", compact)
    output(c.BASE / "data/neural.json", neural)
    output(c.BASE / "data/evaluation.json", evaluation)
    output(c.BASE / "data/scopes.json", scopes)
    c.publish(
        c.BASE / "data/train_texts.jsonl",
        "".join(
            json.dumps(dict(id=rid, text=v["text"]), ensure_ascii=False, sort_keys=True)
            + "\n"
            for rid, v in sorted(records.items())
        ).encode("utf8"),
    )
    prepared[c.rel(c.BASE / "data/train_texts.jsonl")] = c.sha(
        c.BASE / "data/train_texts.jsonl"
    )
    c.array(c.BASE / "data/evaluation_vectors.npy", evectors)
    prepared[c.rel(c.BASE / "data/evaluation_vectors.npy")] = c.sha(
        c.BASE / "data/evaluation_vectors.npy"
    )
    for name in (
        "astra_sq2_cpu_common_v1.py",
        "astra_sq2_cpu_prepare_v1.py",
        "astra_sq2_cpu_run_v1.py",
        "test_astra_sq2_cpu_v1.py",
    ):
        bind(c.ROOT / "scratch/analysis" / name)
    bind(c.PROTOCOL)
    bind(c.ROOT / "scripts/run_cpu_baselines.ps1")
    evidence = dict(
        utc=c.utc(),
        seconds=time.perf_counter() - start,
        before=before,
        after=c.resources(),
        full_counts={
            h: {p: len(spec[p]) for p in ("train", "val")} for h, spec in full.items()
        },
        compact_counts={h: len(compact["heads"][h]["rows"]) for h in c.HEADS},
        native_neural_validation_rows=native_rows,
        evaluation_rows=len(evaluation),
        identical_shared_vectors=overlap,
        full_English_examples=len(records),
        cached_rows=row_count,
        scientific_fits=0,
        Qwen_calls=0,
        native_fine_val_also_in_gate_train={
            h: len(
                {r["id"] for r in full[h]["val"]}
                & {r["id"] for r in full["hate"]["train"]}
            )
            for h in c.HEADS[1:]
        },
    )
    output(c.BASE / "preparation.json", evidence)
    c.verify(inventory["active_source_hashes_unchanged"])
    c.save(
        c.BASE / "plan.json",
        dict(
            status="prepared_before_scientific_fitting_requires_readiness",
            utc=c.utc(),
            environment=c.environment(),
            inputs=inputs,
            prepared=prepared,
            active_source_hashes=inventory["active_source_hashes_unchanged"],
            C_grid=list(c.CS),
            thresholds=list(c.CUTOFFS),
            compact_fold_seeds=list(c.SEEDS),
            full_seed=42,
            rows=19082,
            ordered_heads=list(c.HEADS),
            full_vector_path=c.rel(vector_path),
            protocol=c.rel(c.PROTOCOL),
            resource_policy=dict(
                min_available_RAM_GiB=6,
                min_disk_GiB=15,
                threads=1,
                no_parallel_CUDA=True,
            ),
            primary_contrasts=[
                "en_test/hate",
                "en_test/hate_type",
                "larger_dev/target_group",
                "larger_dev/severity",
            ],
            family_size=4,
            bootstrap_draws=10000,
            no_evaluation_gold_in_fitting=True,
        ),
    )
    print(
        "CPU data preparation complete. No scientific fitting or Qwen/CUDA calls.",
        flush=True,
    )


def validate():
    plan = c.read(c.BASE / "plan.json")
    c.verify(plan["inputs"])
    c.verify(plan["prepared"])
    c.require(
        plan["environment"] == c.environment(), "Scientific package environment changed"
    )
    c.require(
        plan["C_grid"] == list(c.CS) and plan["thresholds"] == list(c.CUTOFFS),
        "Scientific grid changed",
    )
    c.require(
        plan["ordered_heads"] == list(c.HEADS)
        and plan["compact_fold_seeds"] == list(c.SEEDS),
        "Head/fold roster changed",
    )
    return plan


def seal():
    plan = validate()
    tests = c.read(c.BASE / "qa/tests.json")
    c.require(
        tests["successful"] is True and tests["tests_run"] >= 10,
        "Required synthetic checks did not pass",
    )
    c.verify(tests["sources"])
    c.verify(plan["active_source_hashes"])
    c.save(
        c.BASE / "readiness.json",
        dict(
            status="code_data_tests_ready_runtime_pilot_required",
            utc=c.utc(),
            plan_sha256=c.sha(c.BASE / "plan.json"),
            tests_sha256=c.sha(c.BASE / "qa/tests.json"),
            technical_real_data_pilot="runs first under resource guards; no scientific full-data pilot has run",
            new_scientific_fits=0,
            Qwen_calls=0,
            CUDA_used=False,
        ),
    )
    print(
        "CPU package sealed. Wait for current CUDA to finish and available RAM before running.",
        flush=True,
    )


if __name__ == "__main__":
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["prepare", "validate", "seal"])
    args = ap.parse_args()
    with c.offline():
        globals()[args.mode]()

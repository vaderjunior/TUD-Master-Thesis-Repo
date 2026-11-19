import os

os.environ.update(
    CUDA_VISIBLE_DEVICES="",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    HF_HUB_OFFLINE="1",
    TRANSFORMERS_OFFLINE="1",
    TOKENIZERS_PARALLELISM="false",
)
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import socket
import time
from unittest.mock import patch
import numpy as np
from scipy.special import expit
from sklearn.metrics import f1_score
from training.matched_annotations import run as m
from training.roberta import audit as audit

c = m.c
ROOT = c.ROOT
BASE = m.BASE
OUT = ROOT / "experiments/astra/matched_types_review_20261012_v1"
FAMILIES = ("fixed_unweighted", "fixed_balanced", "tuned")
GROUPS = dict(equal67=m.POOL_NAMES[:5], balanced67=m.POOL_NAMES[5:])


def save(name, value):
    path = (OUT / name).resolve()
    c.require(
        path.is_relative_to(OUT.resolve()), "Review output outside its own directory"
    )
    data = c.encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        c.require(path.read_bytes() == data, "Preserve differing prior review: " + name)
    else:
        with path.open("xb") as stream:
            stream.write(data)


def array(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        c.require(
            np.array_equal(np.load(path, allow_pickle=False), value),
            "Preserve prior review array",
        )
    else:
        with path.open("xb") as stream:
            np.save(stream, value, allow_pickle=False)


def forbid(*args, **kwargs):
    raise RuntimeError("Offline review; no network access")


def freeze():
    complete = c.read(BASE / "complete.json")
    c.require(
        complete["status"] == "all_ten_matched_type_pools_complete"
        and complete["plan_sha256"] == c.sha(BASE / "plan.json"),
        "Complete sealed study required",
    )
    inputs = c.read(BASE / "data/evaluation_inputs.json")
    pos = sorted(
        sorted(
            range(len(inputs)),
            key=lambda i: hashlib.sha256(
                ("matched-type-review-cpu|" + inputs[i]["key"]).encode()
            ).hexdigest(),
        )[:16]
    )
    jobs = []
    for pool in m.POOL_NAMES:
        selection = c.read(BASE / f"selections/{pool}.json")["families"]
        jobs += [j for j in m.final_jobs(pool, selection) if j["seed"] == 10]
    save(
        "analysis_plan.json",
        dict(
            status="specified_before_review_performance_scores",
            utc=c.utc(),
            scientific_plan_sha256=c.sha(BASE / "plan.json"),
            protocol_sha256=c.sha(BASE / "protocol.json"),
            groups=GROUPS,
            primary_endpoint="broad/en_test/hate_type",
            primary_methods=["tuned_selected_minus_bge", "frozen_roberta_minus_bge"],
            primary_family_size=4,
            aggregation="Compute macro-F1 per fit. Mean over five neural seeds within pool, then mean over five pools. Linear methods have one fit per pool, not five copies.",
            bootstrap=dict(
                draws=10000,
                seed=20261012,
                confidence=0.9875,
                clusters="whole normalized text, common resample for methods, seeds and pools; fitted models and pools fixed",
            ),
            sensitivities=[
                "source-stratified whole-text clusters, exploratory 95%",
                "joint pool/text resampling, preserve paired methods/seeds within pool, exploratory 95% with five pools only",
                "broader rows outside original English core normalized texts",
                "leave one pool out",
                "all three neural families, both cutoff policies, head-only, exact sets, cardinality, labels, sources, length and ranking",
            ],
            cpu_probe=dict(
                positions=pos,
                keys=[inputs[i]["key"] for i in pos],
                jobs=jobs,
                tolerance=1e-4,
                selection="All unique final configurations at seed10, sixteen identity-hash-selected inputs; no outcome-based choice.",
            ),
            feature_probe="Recompute all 631 training mean features on CPU using the pinned backbone, same tokenizer/pooling; tolerance1e-4.",
            limits="Prior research exposure; fixed common gate; 67 type labels plus147 gate labels, union212-214 IDs. Intervals are not untouched confirmation or universal algorithm ranking.",
            Qwen_calls=0,
            new_neural_training=0,
        ),
    )
    print(
        "Matched-type review analysis policy frozen before new performance scoring.",
        flush=True,
    )


class Bindings:
    def __init__(self):
        self.values = {}
        self.files = {}

    def check(self, path, wanted=None):
        path = Path(path).resolve()
        stat = path.stat()
        version = (stat.st_size, stat.st_mtime_ns)
        key = str(path)
        if key in self.files:
            before, digest = self.files[key]
            c.require(before == version, "Audited file changed during review")
        else:
            digest = c.sha(path)
            after = path.stat()
            c.require(
                (after.st_size, after.st_mtime_ns) == version,
                "File changed while hashing",
            )
            self.files[key] = version, digest
        if wanted is not None:
            c.require(digest == wanted, "File hash differs: " + str(path))
        if path.is_relative_to(ROOT):
            self.values[c.rel(path)] = digest
        return digest

    def mapping(self, values):
        for path, digest in values.items():
            self.check(ROOT / path, digest)


def weights_independent(y, balanced):
    positive = y.sum(0)
    counts = np.stack([len(y) - positive, positive], axis=1)
    weights = np.ones(counts.shape)
    for label in range(y.shape[1]):
        if balanced and counts[label].min() > 0:
            for value in range(2):
                weights[label, value] = len(y) / (2 * counts[label, value])
    return dict(counts=counts.tolist(), weights=weights.tolist(), balanced=balanced)


def checked_fit(job, data, plan_sha, bindings):
    directory = BASE / ("search" if job["mode"] == "search" else "refits") / job["id"]
    path = directory / "complete.json"
    bindings.check(path)
    rec = c.read(path)
    c.require(
        rec["status"] == "complete"
        and rec["job"] == job
        and rec["plan_sha256"] == plan_sha,
        "Fit completion identity differs",
    )
    attempt = (BASE / rec["attempt"]).resolve()
    c.require(
        attempt.is_relative_to((directory / "attempts").resolve()),
        "Fit attempt escapes its job",
    )
    for name, digest in rec["products"].items():
        file = (attempt / name).resolve()
        c.require(file.is_relative_to(attempt), "Product path escapes attempt")
        bindings.check(file, digest)
    meta = c.read(attempt / "training.json")
    spec = data["heads"]["hate_type"]
    rows = spec["rows"]
    fold = np.asarray(data["folds"]["42"])
    fi = np.arange(67) if job["fold"] is None else np.flatnonzero(fold != job["fold"])
    he = (
        np.array([], dtype=int)
        if job["fold"] is None
        else np.flatnonzero(fold == job["fold"])
    )
    c.require(
        meta["job"] == job
        and meta["configuration"] == c.config_map()[job["config"]]
        and meta["pilot_only"] is False,
        "Fit configuration differs",
    )
    c.require(
        meta["fit_ids"] == [rows[i]["id"] for i in fi]
        and meta["held_ids"] == [rows[i]["id"] for i in he],
        "Fit/held identity differs",
    )
    c.require(
        not {rows[i]["cluster"] for i in fi} & {rows[i]["cluster"] for i in he},
        "Fold group leakage",
    )
    expected_steps = math.ceil(len(fi) / 16) * meta["configuration"]["epochs"]
    c.require(
        meta["planned_steps"]
        == meta["attempted_steps"]
        == len(meta["steps"])
        == expected_steps,
        "Incomplete training schedule",
    )
    c.require(
        meta["successful_steps"] + meta["skipped_steps"] == expected_steps
        and meta["successful_steps"] > 0,
        "Step accounting differs",
    )
    c.require(
        [v["step"] for v in meta["steps"]] == list(range(1, expected_steps + 1)),
        "Step trace is discontinuous",
    )
    c.require(
        all(
            np.isfinite(v["lr"])
            and np.isfinite(v["scale_before"])
            and np.isfinite(v["scale_after"])
            and (v["skipped"] or np.isfinite(v["grad_norm"]))
            for v in meta["steps"]
        ),
        "Invalid optimizer trace",
    )
    audit.require_learning_rate_trace(meta)
    yy = np.asarray([rows[i]["target"] for i in fi])
    c.require(
        meta["class_weights"]
        == weights_independent(yy, meta["configuration"]["loss"] == "balanced"),
        "Loss weighting uses wrong labels",
    )
    c.require(
        all(np.isfinite(v) and v > 0 for v in meta["changes"].values()),
        "Backbone/head did not change",
    )
    epochs = meta["epochs"]
    c.require(len(epochs) == meta["configuration"]["epochs"], "Incomplete epochs")
    c.require(
        [e["epoch"] for e in epochs] == list(range(1, len(epochs) + 1)),
        "Epoch trace discontinuous",
    )
    c.require(
        all(np.isfinite(e["training_objective"]) for e in epochs), "Nonfinite objective"
    )
    c.require(
        epochs[-1]["attempted_steps"] == expected_steps
        and epochs[-1]["successful_steps"] == meta["successful_steps"],
        "Epoch/step totals differ",
    )
    if len(he):
        logits = np.load(attempt / "held_logits.npy", allow_pickle=False)
        c.require(
            logits.shape == (len(he), 7) and np.isfinite(logits).all(),
            "Malformed OOF logits",
        )
        yheld = np.asarray([rows[i]["target"] for i in he])
        loss = np.mean(np.logaddexp(0.0, logits.astype(float)) - yheld * logits)
        c.require(
            abs(loss - epochs[-1]["held_unweighted_loss"]) < 2e-6
            and all(np.isfinite(e["held_unweighted_loss"]) for e in epochs),
            "Held loss does not reconstruct",
        )
    else:
        c.require(
            {"model/model.safetensors", "model/config.json", "model/study_meta.json"}
            <= rec["products"].keys(),
            "Final weights are not in receipt",
        )
        model_meta = c.read(attempt / "model/study_meta.json")
        c.require(
            model_meta
            == dict(
                job=job,
                labels=spec["labels"],
                kind=spec["kind"],
                config=c.config_map()[job["config"]],
                pilot_only=False,
            ),
            "Export metadata differs",
        )
        config = c.read(attempt / "model/config.json")
        c.require(
            [config["id2label"][str(i)] for i in range(7)] == spec["labels"],
            "Exported head label order differs",
        )
        c.require(
            meta["export_check"]["clean_reload"]
            and meta["export_check"]["max_abs_logit_difference"] <= 1e-4,
            "Recorded CPU export check failed",
        )
        logits = None
    summary = dict(
        job=job,
        attempt=c.rel(attempt),
        fit_rows=len(fi),
        held_rows=len(he),
        **{
            k: meta[k]
            for k in (
                "attempted_steps",
                "successful_steps",
                "skipped_steps",
                "train_seconds",
                "total_seconds",
                "peak_allocated_gib",
                "configuration",
                "token_lengths",
                "export_check",
                "changes",
            )
        },
        first_epoch=epochs[0],
        last_epoch=epochs[-1],
        minimum_held_loss_epoch=(
            None
            if not len(he)
            else min(epochs, key=lambda e: e["held_unweighted_loss"])["epoch"]
        ),
        attempt_directories=[
            dict(
                path=c.rel(p),
                selected=p.resolve() == attempt,
                stopped=(p / "stopped.json").exists(),
                training_saved=(p / "training.json").exists(),
            )
            for p in sorted((directory / "attempts").iterdir())
            if p.is_dir()
        ],
    )
    return summary, logits, he


def independent_select(data, oof, saved):
    y = np.asarray(
        [r["target"] for r in data["heads"]["hate_type"]["rows"]], dtype=bool
    )
    gate = np.asarray(data["gate_oof"], dtype=bool)
    ix = [i for i in range(7) if y[:, i].sum() >= 2 and (~y[:, i]).sum() >= 2]
    count = 0
    for family, record in saved.items():
        allowed = (
            list(c.config_map())
            if family == "tuned"
            else [c.protocol()["fixed_configs"][family]]
        )
        expected = {(cfg, t) for cfg in allowed for t in c.THRESHOLDS}
        c.require(
            record["selection_indices"] == ix
            and len(record["candidates"]) == len(expected)
            and {(x["config"], x["threshold"]) for x in record["candidates"]}
            == expected,
            "Candidate/support grid differs",
        )
        for cand in record["candidates"]:
            values = []
            for seed in c.CV_SEEDS:
                pred = oof[cand["config"]][seed] >= cand["threshold"]
                pred[~gate] = False
                values.append(
                    float(
                        np.mean(
                            [f1_score(y[:, i], pred[:, i], zero_division=0) for i in ix]
                        )
                    )
                )
            c.require(
                np.allclose(values, cand["seed_scores"], rtol=0, atol=1e-12)
                and abs(np.mean(values) - cand["mean_score"]) < 1e-12,
                "Candidate F1 differs",
            )
            count += 1

        def rank(v):
            cfg = c.config_map()[v["config"]]
            t = v["threshold"]
            return (
                -v["mean_score"],
                cfg["epochs"],
                cfg["learning_rate"],
                int(cfg["loss"] == "balanced"),
                abs(t - 0.5),
                -t,
            )

        c.require(
            min(record["candidates"], key=rank) == record["selected"],
            "Selected configuration/cutoff differs",
        )
    return count


def integrity():
    start = time.perf_counter()
    bindings = Bindings()
    plan = c.read(BASE / "plan.json")
    plan_sha = bindings.check(BASE / "plan.json")
    analysis = c.read(OUT / "analysis_plan.json")
    c.require(analysis["scientific_plan_sha256"] == plan_sha, "Analysis plan differs")
    bindings.mapping(plan["inputs"])
    bindings.mapping(plan["prepared"])
    c.require(plan["environment"] == c.environment(), "Runtime environment differs")
    for name, h in plan["base_cache"]["files"].items():
        bindings.check(Path(plan["base_cache"]["snapshot"]) / name, h)
    done = c.read(BASE / "complete.json")
    bindings.check(BASE / "complete.json")
    bindings.mapping(done["outputs"])
    c.require(
        done["status"] == "all_ten_matched_type_pools_complete"
        and done["plan_sha256"] == plan_sha,
        "Not complete",
    )
    evaluations = c.read(BASE / "data/evaluation_inputs.json")
    c.require(
        len(evaluations) == len({r["key"] for r in evaluations}) == 1705,
        "Evaluation identities differ",
    )
    with np.load(BASE / "data/evaluation_features.npz", allow_pickle=False) as z:
        gate = z["gate"].astype(bool)
    feature_inputs = c.read(BASE / "data/feature_inputs.json")
    cached = np.load(BASE / "features/mean.npy", allow_pickle=False)
    c.require(
        cached.shape == (631, 768)
        and np.isfinite(cached).all()
        and np.allclose(np.linalg.norm(cached, axis=1), 1, atol=2e-4),
        "Frozen TRAIN features invalid",
    )
    fr = c.read(BASE / "features/complete.json")
    bindings.check(BASE / "features/complete.json")
    bindings.mapping(fr["outputs"])
    c.require(
        fr["plan_sha256"] == plan_sha
        and fr["reused_old_rows"] == 11
        and fr["new_rows"] == 620,
        "Feature completion differs",
    )
    old_inputs = c.read(m.feature.BASE / "feature_inputs.json")
    old_positions = {r["key"]: (i, r) for i, r in enumerate(old_inputs)}
    old_features = np.load(
        m.feature.BASE / "features_mean.npy", mmap_mode="r", allow_pickle=False
    )
    missing = []
    for i, row in enumerate(feature_inputs):
        if row["key"] in old_positions:
            j, old = old_positions[row["key"]]
            c.require(
                row["text"] == old["text"]
                and np.array_equal(cached[i], old_features[j]),
                "Reused feature differs",
            )
        else:
            missing.append(i)
    for start_row in range(0, len(missing), 64):
        ix = missing[start_row : start_row + 64]
        rp = BASE / f"features/chunks/{start_row:06d}.json"
        rr = c.read(rp)
        c.require(
            rr["identity"]
            == dict(
                plan_sha256=plan_sha,
                positions=ix,
                input_sha256=c.signature([feature_inputs[i] for i in ix]),
            ),
            "Feature chunk input binding differs",
        )
        values = np.load(rp.with_suffix(".npy"), allow_pickle=False)
        c.require(
            c.sha(rp.with_suffix(".npy")) == rr["sha256"]
            and np.array_equal(cached[ix], values),
            "Feature chunks differ",
        )
    del old_features
    small = c.read(m.bge.BASE / "train.json")
    fixed = c.read(m.bge.BASE / "fit_seed42.json")["heads"]["hate"]
    matrix = np.load(
        c.CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    wanted_ids = {v["key"].removeprefix("train/") for v in feature_inputs}
    source_records = {}
    with (c.CORE / "kb/full/records.jsonl").open(encoding="utf8") as stream:
        for line in stream:
            rec = json.loads(line)
            if rec["id"] in wanted_ids:
                source_records[rec["id"]] = rec
    c.require(set(source_records) == wanted_ids, "Original TRAIN identities missing")
    all_runs = []
    pool_records = {}
    candidate_checks = 0
    decision_vectors = 0
    all_neural_logits = 0
    chunks = 0
    gate_groups = {
        r["cluster"]: i for i, r in enumerate(small["heads"]["hate"]["rows"])
    }
    evalnorm = {m.bge.norm(r["text"]) for r in evaluations}
    for pool in m.POOL_NAMES:
        data = m.data(pool)
        rows = data["heads"]["hate_type"]["rows"]
        fold = np.asarray(data["folds"]["42"])
        c.require(
            not {m.bge.norm(t) for t in data["texts"]} & evalnorm,
            "Pool/evaluation text leakage",
        )
        c.require(
            all(
                t == source_records[r["id"]]["text"]
                and m.pools.c.signature(m.bge.norm(t)) == r["cluster"]
                and r["target"]
                == [
                    int(label in source_records[r["id"]]["meta"]["hate_types"])
                    for label in data["heads"]["hate_type"]["labels"]
                ]
                for r, t in zip(rows, data["texts"])
            ),
            "Pool text, group or annotation differs from source",
        )
        parent = c.read(ROOT / data["source_train"])
        c.require(
            parent["rows"]
            == [{k: v for k, v in r.items() if k != "feature_row"} for r in rows]
            and parent["folds"]["42"] == data["folds"]["42"],
            "Pool/fold source differs",
        )
        bx = np.asarray(matrix[[r["vector_index"] for r in rows]], dtype=np.float64)
        gp = m.bge.binary_predict(fixed["models"][0], bx)
        for i, row in enumerate(rows):
            if row["cluster"] in gate_groups:
                j = gate_groups[row["cluster"]]
                c.require(
                    data["folds"]["42"][i] == small["folds"]["42"][j],
                    "Gate overlap fold differs",
                )
                gp[i] = fixed["oof_probabilities"][j][0]
        c.require(
            np.array_equal(gp > fixed["selected"]["threshold"], data["gate_oof"]),
            "Selection gate reconstructed differently",
        )
        selected = c.read(BASE / f"selections/{pool}.json")
        bindings.check(BASE / f"selections/{pool}.json")
        bindings.mapping(selected["search_receipts"])
        c.require(
            selected["pool"] == pool and selected["plan_sha256"] == plan_sha,
            "Selection identity differs",
        )
        oof = {cfg: np.full((3, 67, 7), np.nan) for cfg in c.config_map()}
        search = []
        final = []
        for j in m.search_jobs(pool):
            record, z, held = checked_fit(j, data, plan_sha, bindings)
            search.append(record)
            all_runs.append(record)
            c.require(
                np.isnan(oof[j["config"]][j["seed"], held]).all(),
                "Duplicate held prediction",
            )
            p = expit(z.astype(float))
            c.require(
                np.allclose(p, c.probabilities("hate_type", z), rtol=0, atol=1e-14),
                "Sigmoid reconstruction differs",
            )
            oof[j["config"]][j["seed"], held] = p
        c.require(
            all(np.isfinite(p).all() for p in oof.values()), "Missing held prediction"
        )
        candidate_checks += independent_select(data, oof, selected["families"])
        c.require(
            m.select_neural(data, oof) == selected["families"],
            "Original selection routine disagrees",
        )
        jobs = m.final_jobs(pool, selected["families"])
        saved_probs = {}
        for j in jobs:
            record, _, _ = checked_fit(j, data, plan_sha, bindings)
            final.append(record)
            all_runs.append(record)
            directory = BASE / "inference" / j["id"]
            ir = c.read(directory / "complete.json")
            bindings.check(directory / "complete.json")
            bindings.mapping(ir["artifacts"])
            tr = BASE / "refits" / j["id"] / "complete.json"
            c.require(
                ir["job"] == j
                and ir["plan_sha256"] == plan_sha
                and ir["rows"] == 1705
                and ir["training_receipt_sha256"] == bindings.check(tr),
                "Inference attribution differs",
            )
            logits = np.load(directory / "logits.npy", allow_pickle=False)
            c.require(
                logits.shape == (1705, 7) and np.isfinite(logits).all(),
                "Inference logits malformed",
            )
            for offset in range(0, 1705, 512):
                stop = min(offset + 512, 1705)
                chunk = directory / f"chunks/{offset:06d}_{stop:06d}"
                cr = c.read(chunk.with_suffix(".json"))
                expected = dict(
                    job=j,
                    plan_sha256=plan_sha,
                    training_receipt_sha256=ir["training_receipt_sha256"],
                    start=offset,
                    stop=stop,
                    input_sha256=c.signature(evaluations[offset:stop]),
                )
                c.require(
                    cr["identity"] == expected
                    and np.array_equal(
                        logits[offset:stop],
                        np.load(chunk.with_suffix(".npy"), allow_pickle=False),
                    ),
                    "Raw inference chunk differs",
                )
                chunks += 1
            saved_probs[(j["config"], j["seed"])] = expit(logits.astype(float))
            all_neural_logits += logits.size
        for family, record in selected["families"].items():
            for seed in c.FINAL_SEEDS:
                prob = saved_probs[(record["selected"]["config"], seed)]
                for policy in ("selected", "default"):
                    cutoff = (
                        record["selected"]["threshold"] if policy == "selected" else 0.5
                    )
                    raw = prob >= cutoff
                    delivered = raw.copy()
                    delivered[~gate] = False
                    p = BASE / f"predictions/{pool}/{family}_{policy}_s{seed}.npz"
                    bindings.check(p)
                    with np.load(p, allow_pickle=False) as stored:
                        c.require(
                            set(stored.files) == {"ungated", "delivered"}
                            and np.array_equal(stored["ungated"], raw)
                            and np.array_equal(stored["delivered"], delivered),
                            "Delivered decisions differ from raw logits",
                        )
                    decision_vectors += len(prob)
        pr = c.read(BASE / f"pool_complete/{pool}.json")
        bindings.mapping(pr["outputs"])
        c.require(
            pr["pool"] == pool
            and pr["status"] == "training_and_predictions_complete"
            and pr["plan_sha256"] == plan_sha
            and pr["final_models"] == len(jobs)
            and pr["predictions"] == 30 * 1705,
            "Pool completion differs",
        )
        pool_records[pool] = dict(
            search_fits=len(search),
            final_fits=len(final),
            selected={k: v["selected"] for k, v in selected["families"].items()},
            gate_false_on_type_training=int(
                (~np.asarray(data["gate_oof"], dtype=bool)).sum()
            ),
            train_union=len(set(small["gate_ids"]) | {r["id"] for r in rows}),
        )
        print(
            "Audited all raw training, selection, weights and predictions:",
            pool,
            len(search),
            len(final),
            flush=True,
        )
    c.require(
        sum(v["search_fits"] for v in pool_records.values()) == 1800
        and sum(v["final_fits"] for v in pool_records.values()) == 145,
        "Unexpected fit totals",
    )
    save("training_runs.json", dict(runs=all_runs, pools=pool_records))
    save(
        "integrity.json",
        dict(
            status="pass",
            utc=c.utc(),
            seconds=time.perf_counter() - start,
            search_fits=1800,
            final_fits=145,
            candidate_records_checked=candidate_checks,
            raw_neural_logit_cells=all_neural_logits,
            inference_chunks=chunks,
            reconstructed_prediction_vectors=decision_vectors,
            reconstructed_delivered_and_ungated_bits=decision_vectors * 7 * 2,
            bound_files=len(bindings.values),
            bound_bytes=sum((ROOT / p).stat().st_size for p in bindings.values),
            artifact_bindings=bindings.values,
            review_source_sha256=c.sha(Path(__file__)),
            Qwen_calls=0,
            CUDA_used=False,
        ),
    )
    print(
        "Full integrity audit passed; performance scoring still separate.", flush=True
    )


def linear():
    c.require(
        c.read(OUT / "integrity.json")["status"] == "pass", "Integrity must pass first"
    )
    features = c.read(BASE / "data/feature_inputs.json")
    positions = {r["key"]: i for i, r in enumerate(features)}
    rf = np.load(BASE / "features/mean.npy", allow_pickle=False)
    bf = np.load(
        c.CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    with np.load(BASE / "data/evaluation_features.npz", allow_pickle=False) as f:
        ex = {k: f[k] for k in f.files}
    fixed = c.read(m.bge.BASE / "fit_seed42.json")["heads"]["hate"]
    gate = (
        m.bge.binary_predict(fixed["models"][0], ex["bge"])
        > fixed["selected"]["threshold"]
    )
    c.require(
        np.array_equal(ex["gate"].astype(bool), gate), "Common evaluation gate differs"
    )
    records = []
    for pool in m.POOL_NAMES:
        data = m.data(pool)
        spec = data["heads"]["hate_type"]
        dd = dict(labels=spec["labels"], rows=spec["rows"], folds=data["folds"])
        saved = c.read(BASE / f"linear/{pool}/fits.json")
        receipt = c.read(BASE / f"linear/{pool}/complete.json")
        c.verify(receipt["outputs"])
        c.require(
            saved["plan_sha256"] == receipt["plan_sha256"] == c.sha(BASE / "plan.json")
            and saved["pool"] == pool,
            "Linear identity differs",
        )
        c.require(
            saved["bge_imported"]
            == c.read(ROOT / data["source_bge_fit"])["heads"]["hate_type"],
            "Historical BGE fit changed",
        )
        for name, train, record_name in [
            (
                "bge",
                np.asarray(bf[[r["vector_index"] for r in spec["rows"]]], dtype=float),
                "bge_imported",
            ),
            (
                "roberta",
                np.asarray(
                    rf[[positions["train/" + r["id"]] for r in spec["rows"]]],
                    dtype=float,
                ),
                "roberta",
            ),
        ]:
            fitted, n = m.pools.fit_type(
                dd, train, 42, np.asarray(data["gate_oof"], dtype=bool)
            )
            c.require(
                fitted == saved[record_name],
                "Recomputed linear coefficients/OOF/candidates differ",
            )
            probability = np.column_stack(
                [
                    (
                        np.full(len(ex[name]), v["constant"], dtype=float)
                        if "constant" in v
                        else expit(
                            np.einsum(
                                "ij,j->i", ex[name].astype(float), np.asarray(v["coef"])
                            )
                            + v["intercept"]
                        )
                    )
                    for v in fitted["models"]
                ]
            )
            with np.load(
                BASE / f"linear/{pool}/probabilities.npz", allow_pickle=False
            ) as p:
                drift = float(np.max(np.abs(p[name] - probability)))
                c.require(
                    drift < 1e-12
                    and np.array_equal(
                        p[name] >= fitted["selected"]["threshold"],
                        probability >= fitted["selected"]["threshold"],
                    ),
                    "Independent linear predictions differ",
                )
            records.append(
                dict(
                    pool=pool,
                    method=name,
                    candidates=n,
                    selected=fitted["selected"],
                    max_probability_drift=drift,
                )
            )
        print("Rebuilt both frozen-feature linear procedures:", pool, flush=True)
    save(
        "linear_audit.json",
        dict(
            status="pass",
            checks=records,
            complete_refits=20,
            candidate_checks=sum(r["candidates"] for r in records),
            common_evaluation_gate_reproduced=True,
            source_sha256=c.sha(Path(__file__)),
            Qwen_calls=0,
            CUDA_used=False,
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["freeze", "integrity", "linear"])
    args = ap.parse_args()
    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket, "create_connection", forbid
    ):
        globals()[args.mode]()


if __name__ == "__main__":
    main()

import time
from datetime import datetime
from analysis.feedback.classifiers import common as a
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.feedback.rounds.errors_kernel import arrays as qwen_arrays


def main():
    start = time.perf_counter()
    contract = a.contract()
    h = a.read(a.REVIEW / "part8/classifier_handoff.json")
    done = a.read(a.BASE / "iterative_complete.json")
    schedule = a.read(a.BASE / "iterative_schedule.json")
    data = a.read(a.BASE / "data.json")
    a.c.verify(done["source_bindings"])
    a.c.verify(h["input_bindings"])
    a.need(
        done["plan_sha256"] == a.sha(a.BASE / "plan.json")
        and done["schedule_sha256"] == a.sha(a.BASE / "iterative_schedule.json"),
        "Execution binding differs",
    )
    a.need(
        done["unique_fits"] == 66
        and done["logical_jobs"] == 74
        and not done["CUDA"]
        and done["new_Qwen_calls"] == 0,
        "Unexpected execution",
    )
    entries = {e["condition"]: sorted(e["ids"]) for e in schedule["entries"]}
    a.need(
        entries == {e["condition"]: sorted(e["source_ids"]) for e in h["entries"]},
        "Frozen pool handoff differs",
    )
    paths = {}
    models = []
    probs = []
    jobs = []
    hashes = {}
    candidate = np.load(a.c.CANDIDATES / "query_vectors.npy", mmap_mode="r")
    ev = np.load(a.c.CPU / "data/evaluation_vectors.npy", mmap_mode="r")
    candidate_probs = []
    hashes.update(
        a.bindings(
            [
                a.c.CANDIDATES / "query_vectors.npy",
                a.c.CPU / "data/evaluation_vectors.npy",
            ]
        )
    )
    static = a.read(a.BASE / "static_complete.json")
    static_paths = {v["receipt"] for v in static["jobs"].values()}
    a.need(len(static_paths) == 38, "Static job count changed")
    for value in static["jobs"].values():
        p = a.ROOT / value["receipt"]
        a.need(a.sha(p) == value["sha256"], "Static receipt changed")
        hashes[value["receipt"]] = value["sha256"]
        for name, digest in a.read(p)["outputs"].items():
            f = p.parent / name
            a.need(a.sha(f) == digest, "Static artifact changed")
            hashes[a.rel(f)] = digest
    maxdiff = 0.0
    new_paths = set()
    full_rows = list(a.rows(a.BASE / "roster.jsonl"))
    for j, job in enumerate(h["jobs"]):
        rp = a.ROOT / job["expected_receipt"]
        receipt = a.read(rp)
        paths[job["expected_receipt"]] = j
        a.need(receipt["identity"] == job["identity"], "Fit identity mismatch")
        if job["reused_existing"]:
            a.need(
                a.sha(rp) == job["existing_receipt_sha256"], "Reused baseline changed"
            )
        else:
            new_paths.add(a.rel(rp))
            a.need(
                a.rel(rp) not in static_paths, "New pool overlaps static fit identity"
            )
        hashes[a.rel(rp)] = a.sha(rp)
        for name, digest in receipt["outputs"].items():
            p = rp.parent / name
            a.need(a.sha(p) == digest, "Fit output changed")
            hashes[a.rel(p)] = digest
        at = rp.parent / receipt["attempt"]
        attr = a.read(at / "attribution.json")
        for field in ["fit_rows", "positive", "negative"]:
            a.need(attr[field] == job[field], "Wrong class support")
        a.need(
            attr["class_weight_recomputed"] and not attr["fine_heads_changed"],
            "Wrong adaptation recipe",
        )
        with np.load(at / "model.npz") as z:
            model = {k: z[k] for k in z.files}
        a.need(
            model["coef"].shape == (1, 1024) and model["constants"].tolist() == [-1],
            "Unexpected linear model",
        )
        a.need(
            all(np.isfinite(v).all() for v in model.values())
            and int(model["iterations"].max()) < 2000,
            "Convergence/model concern",
        )
        saved = np.load(at / "probabilities.npy")
        a.need(
            saved.shape == (16433, 1) and np.isfinite(saved).all(),
            "Bad probability array",
        )
        for first in range(0, 16433, 256):
            x = np.asarray(
                ev[data["evaluation_indices"][first : first + 256]], dtype=np.float64
            )
            z = (
                np.einsum("ij,j->i", x, model["coef"][0], optimize=False)
                + model["intercept"][0]
            )
            check = 1 / (1 + np.exp(-np.clip(z, -700, 700)))
            maxdiff = max(
                maxdiff, float(np.max(abs(check - saved[first : first + len(x), 0])))
            )
            a.need(
                np.allclose(
                    check, saved[first : first + len(x), 0], rtol=0, atol=1e-12
                ),
                "Independent probabilities differ",
            )
            for t in [0.6, 0.5]:
                a.need(
                    np.array_equal(check > t, saved[first : first + len(x), 0] > t),
                    "Reconstruction crosses threshold",
                )
        for t, name in [(0.6, "decisions_06.npy"), (0.5, "decisions_05.npy")]:
            a.need(
                np.array_equal(np.load(at / name), saved > t), "Saved decisions differ"
            )
        if not job["source_ids"]:
            old = np.load(
                a.ROOT
                / data["baseline_head"]["variants"][job["variant"]]["probabilities"],
                mmap_mode="r",
            )[data["evaluation_indices"]]
            a.need(
                np.allclose(saved, old, rtol=0, atol=1e-12),
                "Baseline probabilities changed",
            )
        x = np.asarray(candidate, dtype=np.float64)
        z = (
            np.einsum("ij,j->i", x, model["coef"][0], optimize=False)
            + model["intercept"][0]
        )
        candidate_probs.append(1 / (1 + np.exp(-np.clip(z, -700, 700))))
        probs.append(saved[:, 0])
        models.append(model)
        jobs.append(
            dict(
                index=j,
                receipt=a.rel(rp),
                variant=job["variant"],
                aliases=job["aliases"],
                source_ids=job["source_ids"],
                new=not job["reused_existing"],
                attribution=attr,
                utc=receipt["utc"],
                seconds_this_attempt=receipt["seconds_this_attempt"],
                artifact_bytes=sum(
                    (rp.parent / n).stat().st_size for n in receipt["outputs"]
                ),
            )
        )
        if (j + 1) % 11 == 0:
            print(f"Saved-weight replay: {j+1}/66 models", flush=True)
    alias = {}
    for name, v in done["jobs"].items():
        variant, condition = name.split("/", 1)
        a.need(
            v["receipt"] in paths and v["source_ids"] == entries[condition],
            "Wrong logical alias",
        )
        a.need(
            a.sha(a.ROOT / v["receipt"]) == v["sha256"], "Logical alias hash changed"
        )
        idx = paths[v["receipt"]]
        a.need(
            jobs[idx]["variant"] == variant and condition in jobs[idx]["aliases"],
            "Wrong variant alias",
        )
        alias[name] = idx
    a.need(len(new_paths) == 64 and len(paths) == 66, "Unique fits differ")
    meta = list(a.rows(a.REVIEW / "part7/metadata.jsonl"))
    source_meta = list(a.rows(a.REVIEW / "part5/metadata.jsonl"))
    qrows = list(a.rows(a.REVIEW / "part2/reader_v3/matrix.jsonl"))
    qgate, qvotes, qerror = qwen_arrays(qrows)
    n = len(meta)
    prob = np.empty((n, 2, 4, 10))
    mapping = h["evaluation_mapping"]
    full = np.asarray(probs)
    for i, (m, s, join, q) in enumerate(zip(meta, source_meta, mapping, qrows)):
        a.need(
            m["key"] == s["key"] == join["key"] == q["key"]
            and i == join["iterative_index"],
            "Evaluation join differs",
        )
        f = full_rows[join["classifier_index"]]
        a.need(
            f["key"] == m["key"] and f["gate"] == m["gold_hate"],
            "Classifier evaluation label differs",
        )
        a.need(
            a.c.text_sha(f["text"]) == join["exact_text_sha256"]
            and f["text"] == s["text"],
            "Text join differs",
        )
        a.need(
            data["evaluation_indices"][join["classifier_index"]]
            == join["original_feature_index"],
            "Feature join differs",
        )
        for v, variant in enumerate(a.VARIANTS):
            for r in range(4):
                for k, condition in enumerate(a.CONDITIONS):
                    key = (
                        variant
                        + "/"
                        + ("reference" if k == 0 else f"r{r+1}_{condition}")
                    )
                    prob[i, v, r, k] = full[alias[key], join["classifier_index"]]
        meta[i] = {
            **m,
            "text": s["text"],
            "functionality": s.get("functionality"),
            "template_id": s.get("template_id"),
        }
    gold = np.array([m["gold_hate"] for m in meta])
    l_gate = np.stack([prob[:, a.VARIANTS.index(v)] > t for v, t in a.SETTINGS], axis=1)
    l_error = (l_gate != gold[:, None, None, None]).astype(np.int8)
    a.arrays(
        "predictions.npz",
        probabilities=prob,
        linear_gates=l_gate,
        linear_errors=l_error,
        qwen_gates=qgate,
        qwen_votes=qvotes,
        qwen_errors=qerror,
        gold=gold,
    )
    a.arrays(
        "saved_model_replay.npz",
        full_probabilities=full,
        candidate_probabilities=np.asarray(candidate_probs),
        coef=np.stack([m["coef"][0] for m in models]),
        intercept=np.array([m["intercept"][0] for m in models]),
    )
    monitors = []
    by_pool = {(j["variant"], tuple(j["source_ids"])): j for j in jobs if j["new"]}
    for path in sorted((a.BASE / "resource_checks").glob("*/resources.json")):
        m = a.read(path)
        cmd = m["command"]
        if "--variant" not in cmd:
            continue
        key = (
            cmd[cmd.index("--variant") + 1],
            tuple(sorted(x for x in cmd[cmd.index("--ids") + 1].split(",") if x)),
        )
        if key not in by_pool:
            continue
        a.need(
            m["exit_code"] == 0 and m["stop_reason"] is None,
            "Cumulative worker had a failed attempt",
        )
        a.need(
            not m["originals_or_unrelated_processes_terminated"],
            "Unexpected unrelated process action",
        )
        monitors.append(dict(path=a.rel(path), job_index=by_pool[key]["index"], **m))
        hashes[a.rel(path)] = a.sha(path)
    a.need(
        len(monitors) == 64 and len({m["job_index"] for m in monitors}) == 64,
        "Cumulative monitoring incomplete",
    )
    a.jsonlines("metadata.jsonl", meta)
    a.save(
        "execution.json",
        dict(
            jobs=jobs,
            aliases=alias,
            monitors=monitors,
            unique_fits=66,
            new_fits=64,
            reused_baselines=2,
            logical_jobs=74,
            static_fits_preserved=38,
            all_distinct_saved_fits=102,
            independently_replayed_probabilities=66 * 16433,
            max_probability_difference=maxdiff,
            new_candidate_inference="Saved weights on the same cached 1024 candidate vectors; diagnostic in-sample/source only; no fitting or remote calls.",
        ),
    )
    a.c.verify(contract["inputs"])
    a.c.verify(hashes)
    a.save(
        "assembled.json",
        dict(
            status="completed_fits_and_saved_probabilities_verified",
            inputs=hashes,
            outputs=a.bindings(
                [
                    a.OUT / n
                    for n in [
                        "predictions.npz",
                        "saved_model_replay.npz",
                        "metadata.jsonl",
                        "execution.json",
                    ]
                ]
            ),
            seconds=time.perf_counter() - start,
            new_fits=0,
            new_Qwen_calls=0,
            new_embeddings=0,
        ),
    )
    print("Assembly complete; no fits or Qwen calls.", flush=True)


if __name__ == "__main__":
    with a.c.offline(), threadpool_limits(limits=1):
        main()

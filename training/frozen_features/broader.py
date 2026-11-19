import argparse
from collections import defaultdict
import hashlib
from pathlib import Path
import numpy as np
from training.frozen_features import fit as linear
from training.frozen_features import compact as old
from analysis.common import study as c
from analysis.common import result_reader as s
from analysis.common.api import save_new

BASE = c.ROOT / "experiments/astra/compact_broad_20261008_v1"
VECTOR = c.BASE / "retrieval_v3/vectors"
PRIMARY = (
    "en_test/hate",
    "en_test/hate_type",
    "larger_dev/target_group",
    "larger_dev/severity",
)


def text_sha(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def verify_origins(roster, origins):
    c.require(len(roster) == len(origins), "Vector origin count mismatch")
    for row, source in zip(roster, origins):
        c.require(
            row["key"] == source["key"]
            and text_sha(row["text"]) == source["exact_text_sha256"],
            "Vector key/text mismatch",
        )
    c.require(len({r["key"] for r in roster}) == len(roster), "Repeated input identity")


def prepare():
    import yaml

    old.validate()
    previous = c.read(c.PREP / "complete.json")
    c.verify(c.read(old.BASE / "fit_complete.json")["outputs"])
    c.require(
        c.sha(c.PREP / "plan.json") == previous["outputs"][c.rel(c.PREP / "plan.json")],
        "Original broader plan changed",
    )
    rp = c.PREP / "broad_roster.jsonl"
    c.require(
        c.sha(rp) == previous["outputs"][c.rel(rp)], "Original broader roster changed"
    )
    vr = c.read(VECTOR / "complete.json")
    c.verify(vr["outputs"])
    rr = c.rows(rp)
    origins = c.rows(VECTOR / "query_origins.jsonl")
    verify_origins(rr, origins)
    q = np.load(VECTOR / "query_vectors.npy", mmap_mode="r", allow_pickle=False)
    c.require(
        q.shape == (21627, 1024) and np.isfinite(q).all(),
        "Broader query vectors invalid",
    )
    c.require(
        np.allclose(np.linalg.norm(q, axis=1), 1.0, atol=0.0002),
        "Vectors not normalized",
    )
    core = [r for r in c.rows(c.CORE / "dev_inputs.jsonl") if r["lang"] == "en"]
    core_norm = {old.norm(r["text"]) for r in core}
    train = c.read(old.BASE / "train.json")
    train_clusters = {r["cluster"] for r in train["heads"]["hate"]["rows"]}
    chosen = [(i, r) for i, r in enumerate(rr) if r["lang"] == "en"]
    c.require(len(chosen) == 18275, "Changed English scope")
    c.require(
        not train_clusters & {c.signature(old.norm(r["text"])) for _, r in chosen},
        "Normalized TRAIN/evaluation overlap",
    )
    inputs = [
        dict(
            key=r["key"],
            vector_index=i,
            phase=r["phase"],
            exact_text_sha256=text_sha(r["text"]),
            core_text_seen=old.norm(r["text"]) in core_norm,
        )
        for i, r in chosen
    ]
    save_new(BASE / "prediction_inputs.json", inputs)
    taxonomy = yaml.safe_load(
        (c.ROOT / "config/taxonomy.yaml").read_text(encoding="utf8")
    )
    groups = defaultdict(list)
    for _, row in chosen:
        groups[row["phase"]].append(row | {"id": row["key"]})
    endpoints = {}
    for phase, records in groups.items():
        dims = sorted({d for r in records for d in r["endpoints"]})
        for dim in dims:
            ep = s.endpoint(records, dim, taxonomy)
            endpoints[phase + "/" + dim] = ep
    c.require(set(PRIMARY) <= set(endpoints), "Primary endpoint missing")
    files = [
        Path(__file__),
        Path(linear.__file__),
        Path(old.__file__),
        Path(s.__file__),
        Path(c.__file__),
        c.ROOT / "tests/training/frozen_features/test_broader.py",
        c.ROOT / "analysis/common/scoring.py",
        c.ROOT / "config/taxonomy.yaml",
        rp,
        c.PREP / "complete.json",
        c.PREP / "plan.json",
        VECTOR / "complete.json",
        c.CORE / "dev_inputs.jsonl",
        old.BASE / "train.json",
        old.BASE / "plan.json",
        BASE / "prediction_inputs.json",
    ]
    files += [old.BASE / f"fit_seed{seed}.json" for seed in old.SEEDS]
    source_files = {c.ROOT / r["source_file"] for _, r in chosen}
    inherited = c.read(c.PREP / "plan.json")["inputs"]
    for path in source_files | {c.ROOT / "config/taxonomy.yaml"}:
        c.require(
            inherited.get(c.rel(path)) == c.sha(path),
            "Inherited gold/taxonomy source changed",
        )
    files += list(source_files)
    hashes = {c.rel(p): c.sha(p) for p in set(files)}
    hashes.update(vr["outputs"])
    save_new(
        BASE / "plan.json",
        {
            "created_utc": c.utc(),
            "status": "frozen_before_wider_compact_predictions_or_scores",
            "inputs": hashes,
            "environment": old.environment(),
            "seeds": old.SEEDS,
            "input_count": 18275,
            "prediction_count": 54825,
            "endpoints": endpoints,
            "primary_contrasts": list(PRIMARY),
            "family_size": 4,
            "draws": 10000,
            "model": "Reuse exact original compact BGE coefficients, label order and TRAIN-selected cutoffs; no selection from these wider results.",
            "future_contrast": "BGE minus aligned original compact RoBERTa on four named endpoints. Existing neural broader inference and independent assembly must finish before paired contrasts. Never substitute older checkpoints or infer a difference from nonmatching population means.",
            "scoring_contract": s.contract(),
            "exposure": "All rosters have prior project exposure. This is broader prospective-scope evidence, not untouched confirmation.",
            "sensitivity": "Also score exclusion of every exact-normalized original English core text, retaining full-roster macro labels. Source and label metrics are descriptive; HateCheck uses functional groups/template clusters in subsequent paired comparisons.",
            "limits": "BGE-only scores may be published as absolute descriptive results. No superiority interval until paired comparator outputs are complete. Existing broader RAG family and running scripts unchanged.",
        },
    )
    print(
        "Wider frozen compact protocol:",
        {k: len(v["scorable_ids"]) for k, v in endpoints.items()},
        flush=True,
    )


def validate():
    p = c.read(BASE / "plan.json")
    c.verify(p["inputs"])
    c.require(p["environment"] == old.environment(), "Changed scientific environment")
    return p


def independent_predictions(heads, x):

    result = {}
    for head in old.HEADS:
        cols = []
        for model in heads[head]["models"]:
            if "constant" in model:
                cols.append(np.full(len(x), model["constant"], dtype=float))
            else:
                z = (
                    np.einsum("ij,j->i", x, np.asarray(model["coef"]))
                    + model["intercept"]
                )
                cols.append(1 / (1 + np.exp(-np.clip(z, -700, 700))))
        result[head] = np.column_stack(cols)
    return result


def predict():
    p = validate()
    inputs = c.read(BASE / "prediction_inputs.json")
    matrix = np.load(VECTOR / "query_vectors.npy", mmap_mode="r", allow_pickle=False)
    outputs = {}
    bits_checked = 0
    for seed in old.SEEDS:
        weights = c.read(old.BASE / f"fit_seed{seed}.json")["heads"]
        predictions = []
        prob_parts = []
        for start in range(0, len(inputs), 512):
            batch = inputs[start : start + 512]
            x = np.asarray(matrix[[r["vector_index"] for r in batch]], dtype=np.float64)
            probabilities = linear.probabilities(weights, x)
            separate = independent_predictions(weights, x)
            for head in old.HEADS:
                c.require(
                    np.allclose(
                        probabilities[head], separate[head], rtol=0, atol=1e-12
                    ),
                    "Independent score reconstruction differs",
                )
            values = linear.decode(weights, probabilities)
            other = linear.decode(weights, separate)
            c.require(values == other, "Independent numerical route changes decisions")
            predictions += [
                dict(key=r["key"], seed=seed, result=value)
                for r, value in zip(batch, values)
            ]
            prob_parts.append(
                np.concatenate([probabilities[h] for h in old.HEADS], axis=1)
            )
            bits_checked += len(batch) * 18
        path = BASE / f"predictions_seed{seed}.jsonl"
        linear.save_rows(path, predictions)
        outputs[c.rel(path)] = c.sha(path)
        path = BASE / f"probabilities_seed{seed}.npy"
        linear.save_array(path, np.concatenate(prob_parts))
        outputs[c.rel(path)] = c.sha(path)
        print(
            "Predicted and checked",
            seed,
            len(predictions),
            "English inputs",
            flush=True,
        )
    save_new(
        BASE / "predictions_complete.json",
        {
            "created_utc": c.utc(),
            "status": "all_frozen_compact_predictions_complete",
            "plan_sha256": c.sha(BASE / "plan.json"),
            "outputs": outputs,
            "independent_probability_cells_checked": bits_checked,
            "probability_column_heads": {
                h: len(c.read(old.BASE / "fit_seed42.json")["heads"][h]["models"])
                for h in old.HEADS
            },
            "new_training": 0,
            "Qwen_calls": 0,
            "CUDA_used": False,
        },
    )


def summarize_checked(records, predictions, ep):
    from sklearn.metrics import f1_score

    con = s.contributions(records, predictions, ep)
    result = s.summarize(con, records, ep["dimension"])
    if ep["macro_labels"] and len(con["ids"]):
        idx = [con["labels"].index(label) for label in ep["macro_labels"]]
        value = float(
            np.mean(
                [
                    f1_score(con["gold"][:, j], con["pred"][:, j], zero_division=0)
                    for j in idx
                ]
            )
        )
        c.require(
            abs(value - result["macro_f1_valid_only"]) < 1e-12,
            "Independent sklearn F1 mismatch",
        )
    return result


def score():
    p = validate()
    receipt = c.read(BASE / "predictions_complete.json")
    c.verify(receipt["outputs"])
    groups, _ = s.scientific_rosters("broad")
    masks = {
        r["key"]: r["core_text_seen"] for r in c.read(BASE / "prediction_inputs.json")
    }
    models = {
        seed: {
            r["key"]: r["result"]
            for r in c.rows(BASE / f"predictions_seed{seed}.jsonl")
        }
        for seed in old.SEEDS
    }
    endpoints = {}
    cells = 0
    for name, ep in p["endpoints"].items():
        phase, dim = name.split("/")
        records = groups[phase]
        keys = [r["id"] for r in records]
        fresh = [r for r in records if not masks[r["id"]]]
        fresh_ep = ep | {
            "scorable_ids": [key for key in ep["scorable_ids"] if not masks[key]]
        }
        runs = []
        for seed, output in models.items():
            c.require(set(keys) <= set(output), "Missing broad prediction")
            full = summarize_checked(records, {k: output[k] for k in keys}, ep)
            subset = summarize_checked(
                fresh, {r["id"]: output[r["id"]] for r in fresh}, fresh_ep
            )
            runs.append(
                {"seed": seed, "full_roster": full, "outside_core_texts": subset}
            )
            cells += 2
        endpoints[name] = {
            "endpoint": ep,
            "runs": runs,
            "mean": float(
                np.mean([v["full_roster"]["macro_f1_valid_only"] for v in runs])
            ),
            "sample_sd": float(
                np.std([v["full_roster"]["macro_f1_valid_only"] for v in runs], ddof=1)
            ),
            "core_excluded": len(ep["scorable_ids"]) - len(fresh_ep["scorable_ids"]),
        }
        print(
            name, len(ep["scorable_ids"]), round(endpoints[name]["mean"], 6), flush=True
        )
    c.verify(p["inputs"])
    save_new(
        BASE / "results.json",
        {
            "created_utc": c.utc(),
            "status": "BGE_only_broader_scores_complete_paired_comparison_pending",
            "plan_sha256": c.sha(BASE / "plan.json"),
            "endpoints": endpoints,
            "independent_score_cells": cells,
            "primary_paired_contrasts": "pending_aligned_compact_neural_predictions",
            "new_training": 0,
            "Qwen_calls": 0,
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "predict", "score"])
    args = parser.parse_args()
    with c.offline():
        globals()[args.mode]()

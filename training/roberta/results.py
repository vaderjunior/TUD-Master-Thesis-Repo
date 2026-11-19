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
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import socket
from unittest.mock import patch
import numpy as np
from training.roberta import common as c
from analysis.classification.supervised import compact_review as previous

OUT = c.ROOT / "experiments/astra/roberta_tuning_review_20261011_v1"
PRIMARY = [
    "en_test/hate",
    "en_test/hate_type",
    "larger_dev/target_group",
    "larger_dev/severity",
]
VARIANTS = [f"{f}_{p}" for f in c.FAMILIES for p in ("selected", "default")]


def save(name, value):
    path = OUT / name
    payload = c.encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        c.require(path.read_bytes() == payload, "Preserve existing review " + name)
    else:
        path.write_bytes(payload)


def forbid(*args, **kwargs):
    raise AssertionError("Offline review: network disabled")


def freeze():
    p = c.read(c.BASE / "plan.json")
    complete = c.read(c.BASE / "complete.json")
    c.require(
        complete["status"] == "training_and_predictions_complete"
        and complete["plan_sha256"] == c.sha(c.BASE / "plan.json"),
        "Complete sealed run required",
    )
    save(
        "analysis_plan.json",
        dict(
            status="specified_before_this_review_scores",
            primary=PRIMARY,
            left="tuned_selected",
            right="bge",
            left_seeds=list(c.FINAL_SEEDS),
            right_seeds=[42, 43, 44],
            bootstrap=dict(
                draws=10000,
                seed=20261008,
                confidence=0.9875,
                resample="Shared whole text clusters, all five neural and three linear fits kept; mean per-fit F1, not ensembling or artificial seed pairing.",
            ),
            sensitivity="Natural source-stratified whole-cluster resampling, 95%; outside-original-core descriptive subsets.",
            supporting="All six variants; archived/full neural and frozen features; per-label precision/recall, gate errors, exact-set agreement, ungated scores and ranking; training trajectories.",
            selection_sha256=c.sha(c.BASE / "selection.json"),
            run_plan_sha256=c.sha(c.BASE / "plan.json"),
            protocol_sha256=c.sha(c.BASE / "protocol.json"),
            exposure="Already exposed populations; TRAIN-only choices do not establish untouched confirmation.",
            Qwen_calls=0,
        ),
    )


def integrity():
    from training.roberta import prepare as prep
    from training.roberta import audit as audit

    p = prep.validate()
    plan_sha = c.sha(c.BASE / "plan.json")
    complete = c.read(c.BASE / "complete.json")
    c.verify(complete["artifacts"])
    data = c.read(c.BASE / "data/train.json")
    selected = c.read(c.BASE / "selection.json")
    oof, receipts, steps = audit.load_oof(data, plan_sha)
    c.require(
        c.select(data, oof) == selected["families"]
        and receipts == selected["search_receipts"],
        "OOF selection reproduction failed",
    )
    checked = audit.check_selection(data, oof, selected["families"])
    c.require(checked == 392 and steps == 125280, "Candidate/step count differs")
    final = audit.final_check(data, selected["families"], plan_sha)
    c.require(
        final == c.read(c.BASE / "final_fit_audit.json")
        and final["final_checkpoints"] == 60,
        "Final fit audit changed",
    )
    reconstructed = audit.assemble(
        selected["families"], data, plan_sha, c.read(c.BASE / "data/inference.json")
    )
    c.require(
        reconstructed == c.read(c.BASE / "prediction_audit.json"),
        "Saved decoded predictions differ from raw logits",
    )
    entries = []
    bindings = {**p["inputs"], **complete["artifacts"]}
    all_attempts = []
    for job in [*c.search_jobs(), *c.final_jobs(selected["families"])]:
        directory = c.job_dir(job)
        receipt = c.read(directory / "complete.json")
        attempt = c.BASE / receipt["attempt"]
        meta = c.read(attempt / "training.json")
        bindings[c.rel(directory / "complete.json")] = c.sha(
            directory / "complete.json"
        )
        bindings.update({c.rel(attempt / n): h for n, h in receipt["products"].items()})
        fit = set(meta["fit_ids"])
        held = set(meta["held_ids"])
        spec = data["heads"][job["head"]]
        fit_clusters = {r["cluster"] for r in spec["rows"] if r["id"] in fit}
        held_clusters = {r["cluster"] for r in spec["rows"] if r["id"] in held}
        c.require(
            not fit & held and not fit_clusters & held_clusters,
            "Training/held-group leakage",
        )
        c.require(
            len(meta["epochs"]) == meta["configuration"]["epochs"]
            and meta["steps"][-1]["step"] == meta["planned_steps"],
            "Training curve truncated",
        )
        c.require(
            all(
                np.isfinite(v["training_objective"])
                and np.isfinite(v.get("held_unweighted_loss", 0))
                for v in meta["epochs"]
            ),
            "Nonfinite training curve",
        )
        successful = sum(not s["skipped"] for s in meta["steps"])
        c.require(successful == meta["successful_steps"], "Step accounting")
        entries.append(
            dict(
                job=job,
                attempt=c.rel(attempt),
                fit_rows=len(fit),
                held_rows=len(held),
                **{
                    k: meta[k]
                    for k in [
                        "configuration",
                        "attempted_steps",
                        "successful_steps",
                        "skipped_steps",
                        "train_seconds",
                        "total_seconds",
                        "peak_allocated_gib",
                        "changes",
                        "export_check",
                        "token_lengths",
                    ]
                },
                first_epoch=meta["epochs"][0],
                last_epoch=meta["epochs"][-1],
            )
        )
        for path in sorted((directory / "attempts").iterdir()):
            if path.is_dir():
                all_attempts.append(
                    dict(
                        path=c.rel(path),
                        selected=path == attempt,
                        stopped=(path / "stopped.json").exists(),
                        training_saved=(path / "training.json").exists(),
                    )
                )
    inputs = c.read(c.BASE / "data/inference.json")
    scopes = c.read(c.BASE / "data/scopes.json")
    c.require(
        len(inputs) == 19082 and len({r["key"] for r in inputs}) == 19082,
        "Inference identity count",
    )
    c.require(
        not {previous.norm(t) for t in data["texts"]}
        & {previous.norm(r["text"]) for r in inputs},
        "Normalized TRAIN/evaluation overlap",
    )
    for scope, values in scopes.items():
        c.require(
            len({r["source_id"] for r in values}) == len(values),
            "Duplicate scope identity",
        )
        for row in values:
            c.require(
                inputs[row["position"]]["key"] == row["key"],
                "Scope-to-array position differs",
            )
    save("training_runs.json", dict(runs=entries, attempts=all_attempts))
    save(
        "integrity.json",
        dict(
            status="pass",
            search_fits=720,
            final_fits=60,
            candidates_reproduced=checked,
            search_attempted_steps=steps,
            head_decisions_reconstructed=reconstructed[
                "independently_checked_head_decisions"
            ],
            complete_decisions=reconstructed["complete_outputs"],
            scope_counts={k: len(v) for k, v in scopes.items()},
            input_identities=len(inputs),
            normalized_train_evaluation_overlap=0,
            artifact_bindings=bindings,
            bound_files=len(bindings),
            bound_bytes=sum((c.ROOT / n).stat().st_size for n in bindings),
            review_source_sha256=c.sha(Path(__file__)),
            Qwen_calls=0,
            CUDA_used=False,
        ),
    )
    print(
        "Integrity passed: 720 search fits, 60 final models, 392 candidate scores and all 2,289,840 head decisions reproduced.",
        flush=True,
    )


def context(scope):
    eps = c.read(
        previous.BGE / "plan.json" if scope == "core" else previous.BROAD / "plan.json"
    )["endpoints"]
    scopes = c.read(c.BASE / "data/scopes.json")[scope]
    positions = {r["source_id"]: r["position"] for r in scopes}
    return eps, positions


def gold(scope, name, ep):
    rr = previous.gold_records(scope, name)
    ids = []
    yy = []
    kept = []
    for row in rr:
        value = previous.clean(row.get(previous.FIELDS[ep["dimension"]]))
        gate = previous.clean(row.get("gate"))
        if (
            value is None
            or ep["dimension"] not in ("hate", "legal")
            and gate is not True
        ):
            continue
        labs = previous.as_set(value, ep["dimension"])
        c.require(labs <= set(ep["all_labels"]), "Unknown gold label")
        ids.append(str(row["id"]))
        yy.append([label in labs for label in ep["all_labels"]])
        kept.append(row)
    y = np.asarray(yy, dtype=bool)
    c.require(
        ids == ep["scorable_ids"]
        and y.sum(0).tolist() == [ep["gold_support"][k] for k in ep["all_labels"]],
        "Endpoint eligibility/support changed",
    )
    return rr, kept, ids, y


def new_bits(variant, seed, head, ids, positions, ep):
    data = c.read(c.BASE / "data/train.json")
    labels = data["heads"][head]["labels"]
    if head == "hate":
        labels = ["not_hate", "hate"]
    order = [labels.index(k) for k in ep["all_labels"]]
    with np.load(
        c.BASE / "predictions" / f"{variant}_s{seed}.npz", allow_pickle=False
    ) as z:
        a = z[head][[positions[k] for k in ids]][:, order]
        c.require(
            a.shape == (len(ids), len(order)) and set(np.unique(a)) <= {0, 1},
            "Invalid decoded matrix",
        )
    return a.astype(bool)


def scores():
    c.require(
        c.read(OUT / "integrity.json")["status"] == "pass",
        "Integrity must pass before scoring",
    )
    cells = {}
    subgroups = {}
    cache = {}
    verified = 0
    old_scores = c.read(previous.OUT / "scores.json")["cells"]
    coretexts = {
        previous.norm(r["text"])
        for r in previous.rows(previous.CORE / "dev_inputs.jsonl")
        if r["lang"] == "en"
    }
    for scope in ["core", "broad"]:
        eps, positions = context(scope)
        maps = previous.predictions(scope)
        for name, ep in eps.items():
            rr, kept, ids, y = gold(scope, name, ep)
            runs = defaultdict(list)
            all_models = [(variant, list(c.FINAL_SEEDS)) for variant in VARIANTS] + [
                (m, [0, 1, 2] if m.startswith("encoder_") else [42, 43, 44])
                for m in [
                    "encoder_kb",
                    "encoder_full",
                    "bge",
                    "roberta_mean",
                    "roberta_cls",
                ]
            ]
            for model, seeds in all_models:
                for seed in seeds:
                    if model in VARIANTS:
                        x = new_bits(model, seed, ep["dimension"], ids, positions, ep)
                    else:
                        ii, yy, x = previous.arrays(rr, maps[model, seed], ep)
                        c.require(
                            ii == ids and np.array_equal(yy, y),
                            "Comparator gold mismatch",
                        )
                    stats, counts = previous.summarize_bits(y, x, ep)
                    stats["seed"] = seed
                    runs[model].append(stats)
                    if model not in VARIANTS:
                        expected = next(
                            r
                            for r in old_scores[scope + "/" + name]["runs"][model]
                            if r["seed"] == seed
                        )
                        c.require(
                            stats == expected,
                            "Earlier reference result no longer reproduces",
                        )
                        verified += 1
                    if (
                        scope == "broad"
                        and model in ["tuned_selected", "bge"]
                        and name in PRIMARY
                    ):
                        cache[f"{name}|{model}|{seed}"] = counts
                    masks = {
                        source: np.asarray([r["source"] == source for r in kept])
                        for source in sorted({r["source"] for r in kept})
                    }
                    if scope == "broad":
                        masks["outside_original_core"] = np.asarray(
                            [previous.norm(r["text"]) not in coretexts for r in kept]
                        )
                    subgroups[f"{scope}/{name}/{model}/{seed}"] = {
                        g: previous.summarize_bits(y[m], x[m], ep)[0]
                        for g, m in masks.items()
                        if m.any()
                    }
            cells[scope + "/" + name] = dict(
                n=len(ids),
                endpoint={k: v for k, v in ep.items() if k != "scorable_ids"},
                runs=dict(runs),
                means={
                    m: float(np.mean([v["macro_f1"] for v in rs]))
                    for m, rs in runs.items()
                },
                sample_sds={
                    m: float(np.std([v["macro_f1"] for v in rs], ddof=1))
                    for m, rs in runs.items()
                },
            )
            print(
                scope,
                name,
                {
                    m: round(cells[scope + "/" + name]["means"][m], 6)
                    for m in [
                        "fixed_unweighted_default",
                        "fixed_unweighted_selected",
                        "fixed_balanced_selected",
                        "tuned_selected",
                        "bge",
                        "roberta_mean",
                        "encoder_full",
                    ]
                },
                flush=True,
            )
    target = OUT / "paired_counts.npz"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        with np.load(target, allow_pickle=False) as saved:
            c.require(
                set(saved.files) == set(cache)
                and all(np.array_equal(saved[k], v) for k, v in cache.items()),
                "Paired count cache differs",
            )
    else:
        np.savez_compressed(target, **cache)
    save(
        "scores.json",
        dict(
            status="scored",
            cells=cells,
            earlier_reference_cells_reproduced=verified,
            note="All five neural seeds and three comparator fits; these vary different randomization axes. No evaluation-based configuration selection.",
        ),
    )
    save(
        "subgroups.json",
        dict(
            post_result_descriptions=True,
            original_macro_labels_fixed=True,
            cells=subgroups,
        ),
    )


def bootstrap(
    counts, clusters, sources, indices, left=5, draws=10000, stratified=False
):
    unique = sorted(set(clusters))
    mapping = {v: i for i, v in enumerate(unique)}
    agg = np.zeros((len(unique), counts.shape[0], counts.shape[2], 3), np.int64)
    members = defaultdict(set)
    for i, k in enumerate(clusters):
        agg[mapping[k]] += counts[:, i]
        members[k].add(sources[i])
    strata = defaultdict(list)
    for k, i in mapping.items():
        strata[tuple(sorted(members[k])) if stratified else ("all",)].append(i)
    rng = np.random.default_rng(20261008)
    values = []
    flat = agg.reshape(len(unique), -1).astype(float)
    for start in range(0, draws, 64):
        n = min(64, draws - start)
        picks = np.concatenate(
            [
                rng.choice(g, (n, len(g)), replace=True)
                for _, g in sorted(strata.items())
            ],
            axis=1,
        )
        weights = np.asarray(
            [np.bincount(p, minlength=len(unique)) for p in picks], dtype=float
        )
        total = (weights @ flat).reshape(n, counts.shape[0], counts.shape[2], 3)
        c.require(
            np.array_equal(total[0], agg[picks[0]].sum(0)),
            "Cluster bootstrap direct draw differs",
        )
        tp, fp, fn = total[..., 0], total[..., 1], total[..., 2]
        den = 2 * tp + fp + fn
        f = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den != 0)[
            ..., indices
        ].mean(-1)
        values.extend((f[:, :left].mean(1) - f[:, left:].mean(1)).tolist())
    total = counts.sum(1)
    tp, fp, fn = total[..., 0], total[..., 1], total[..., 2]
    den = 2 * tp + fp + fn
    f = np.divide(2 * tp, den, out=np.zeros_like(tp, dtype=float), where=den != 0)[
        :, indices
    ].mean(-1)
    return dict(
        left=float(f[:left].mean()),
        right=float(f[left:].mean()),
        difference=float(f[:left].mean() - f[left:].mean()),
        ci9875=np.quantile(values, [0.00625, 0.99375]).tolist(),
        ci95=np.quantile(values, [0.025, 0.975]).tolist(),
        n=len(clusters),
        clusters=len(unique),
        draws=draws,
        seed=20261008,
        stratified=stratified,
    )


def statistics():
    plans = c.read(previous.BROAD / "plan.json")["endpoints"]
    out = {}
    with np.load(OUT / "paired_counts.npz", allow_pickle=False) as z:
        for name in PRIMARY:
            ep = plans[name]
            _, kept, ids, y = gold("broad", name, ep)
            counts = np.stack(
                [
                    z[f"{name}|{m}|{s}"]
                    for m, seeds in [
                        ("tuned_selected", c.FINAL_SEEDS),
                        ("bge", [42, 43, 44]),
                    ]
                    for s in seeds
                ]
            )
            clusters = [r["text_cluster"] for r in kept]
            sources = [r["source"] for r in kept]
            idx = [ep["all_labels"].index(l) for l in ep["macro_labels"]]
            out[name] = dict(
                primary=bootstrap(counts, clusters, sources, idx),
                source_stratified=bootstrap(
                    counts, clusters, sources, idx, stratified=True
                ),
            )
            print("Paired primary", name, out[name]["primary"], flush=True)
    save(
        "statistics.json",
        dict(
            primary_family=PRIMARY,
            confidence=0.9875,
            draws=10000,
            contrasts=out,
            caution="Conditional on the five neural and three BGE fitted settings. Text bootstrap is not seed uncertainty or untouched TEST confirmation. Source-stratified ci95 is supporting.",
        ),
    )


def self_test():
    counts = np.zeros((8, 3, 2, 3), dtype=np.int64)
    counts[:5, :, :, 0] = 1
    counts[5:, :, :, 2] = 1
    r = bootstrap(counts, ["a", "a", "b"], ["s", "s", "t"], [0, 1], draws=128)
    c.require(
        r["difference"] == 1 and r["ci9875"] == [1.0, 1.0] and r["clusters"] == 2,
        "Unequal five/three-fit bootstrap fails",
    )
    counts[:] = 1
    r = bootstrap(
        counts, ["a", "a", "b"], ["s", "s", "t"], [0, 1], draws=128, stratified=True
    )
    c.require(
        r["difference"] == 0 and r["ci9875"] == [0.0, 0.0],
        "Paired zero/stratification fixture fails",
    )
    previous.self_tests()
    print("Unequal five/three-fit paired-bootstrap checks passed.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["freeze", "integrity", "scores", "statistics", "test"]
    )
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket, "create_connection", forbid
    ):
        {
            "freeze": freeze,
            "integrity": integrity,
            "scores": scores,
            "statistics": statistics,
            "test": self_test,
        }[args.mode]()


if __name__ == "__main__":
    main()

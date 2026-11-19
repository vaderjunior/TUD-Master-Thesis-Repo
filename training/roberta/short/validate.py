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
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import socket
from unittest.mock import patch
import numpy as np
from training.roberta.short import common as s
from training.roberta.short import prepare as prep
from training.roberta.short import audit as audit
from training.roberta import results as prior

c = s.c
ROOT = c.ROOT
BASE = s.BASE
OUT = ROOT / "experiments/astra/roberta_short_review_20261020_v1"
PRIMARY = list(prior.PRIMARY)
VARIANTS = list(prior.VARIANTS)
PLAN_SHA = "a11c0aef95be45c9da73a8e065bb6f39a7c833006b7a831cae60fae27de0203e"
QWEN_PLAN = ROOT / "experiments/astra/sq3_transfer_20261012_v1/execution/plan.json"


def forbid(*args, **kwargs):
    raise RuntimeError("Offline review: network and original evidence writes forbidden")


@contextmanager
def offline():
    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket.socket, "connect_ex", forbid
    ), patch.object(socket, "create_connection", forbid):
        yield


def save(name, value):
    path = (OUT / name).resolve()
    c.require(path.is_relative_to(OUT.resolve()), "Review output boundary")
    content = value if isinstance(value, bytes) else c.encoded(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        c.require(
            path.read_bytes() == content, "Preserve previous review artifact: " + name
        )
    else:
        with path.open("xb") as stream:
            stream.write(content)


def guard():
    p = c.read(QWEN_PLAN)
    c.require(
        c.sha(QWEN_PLAN)
        == "ca08393271cfe2461966376daebe45d5d156dcf0acc38f4ce0e9f480f8a2ed5d",
        "Active Qwen plan changed",
    )
    c.verify(p["source_hashes"])
    return {c.rel(QWEN_PLAN): c.sha(QWEN_PLAN), **p["source_hashes"]}


def freeze():
    c.require(not (OUT / "analysis_plan.json").exists(), "Analysis already specified")
    plan = c.read(BASE / "plan.json")
    complete = c.read(BASE / "complete.json")
    c.require(
        c.sha(BASE / "plan.json") == PLAN_SHA == complete["plan_sha256"],
        "Completion plan differs",
    )
    c.require(
        complete["status"]
        == "short_schedules_training_predictions_complete_review_pending",
        "Completed run required",
    )
    files = [
        BASE / n
        for n in (
            "plan.json",
            "complete.json",
            "selection.json",
            "search_audit.json",
            "final_routes.json",
            "final_audit.json",
            "prediction_audit.json",
        )
    ]
    files += [
        s.PROTOCOL,
        s.OLD / "selection.json",
        prior.OUT / "scores.json",
        prior.OUT / "verification.json",
        Path(__file__),
        Path(prior.__file__),
    ]
    save(
        "analysis_plan.json",
        {
            "status": "specified_before_new_evaluation_scores",
            "utc": c.utc(),
            "original_primary_contract": plan["primary"],
            "primary_endpoints": PRIMARY,
            "left": "expanded_tuned_selected",
            "right": "original_20_60_tuned_selected",
            "final_seeds": list(c.FINAL_SEEDS),
            "bootstrap_draws": 10000,
            "confidence": 0.9875,
            "bootstrap_seed": 20261008,
            "resampling": "Same whole text clusters for both procedures; mean per-seed F1, all five seeds retained; conditional on fitted models.",
            "supporting": [
                "source-stratified cluster bootstrap, 95%",
                "outside-original-core and source descriptive subsets, fixed macro labels",
                "all six expanded variants and all six original variants",
                "BGE and preserved historical/frozen/full-data baselines",
                "per-label precision/recall, exact sets, cardinality, FPR/FNR",
                "ungated fine heads, gate transitions and fixed-cutoff decomposition",
                "all TRAIN candidate durations, held-loss curves, optimizer and resource accounting",
            ],
            "selection_inspected_before_freeze": "TRAIN choices and completion metadata only; no new evaluation score inspected.",
            "exposure": plan["primary"]["exposure"],
            "duration_interpretation": "Fresh schedules with duration-dependent warmup/decay; not fixed-horizon early stopping.",
            "no_scope_change": True,
            "new_training_or_inference": False,
            "CUDA_used": False,
            "Qwen_calls": 0,
            "source_and_evidence_bindings": s.hashes(files),
            "running_Qwen_bindings": guard(),
            "resources": s.resource.resources(),
        },
    )
    print("Review policy recorded before new evaluation scoring.", flush=True)


def assert_existing_arrays(path, values):
    c.require(path.is_file(), "Required completed prediction matrix absent")
    with np.load(path, allow_pickle=False) as saved:
        c.require(
            set(saved.files) == set(values)
            and all(np.array_equal(saved[k], v) for k, v in values.items()),
            "Read-only assembled predictions differ",
        )


def verify_analysis(analysis):
    bindings = dict(analysis["source_and_evidence_bindings"])
    amendment = c.read(OUT / "reader_amendment.json")
    key = c.rel(Path(__file__))
    c.require(
        bindings[key] == amendment["old_source_sha256"]
        and c.sha(OUT / "review_source_before_reader_fix.py") == bindings[key]
        and c.sha(Path(__file__)) == amendment["new_source_sha256"],
        "Reader amendment differs",
    )
    bindings[key] = amendment["new_source_sha256"]
    c.verify(bindings)


def ungated_bits(base, variant, seed, head):

    with np.load(
        base / "predictions" / f"{variant}_s{seed}.npz", allow_pickle=False
    ) as saved:
        key = head + "_ungated"
        if key in saved.files:
            return saved[key]
    c.require(base == s.OLD, "Unexpected missing ungated array in new study")
    family, policy = variant.rsplit("_", 1)
    chosen = c.read(base / "selection.json")["families"][family][head]["selected"]
    logits = np.load(
        base / "inference" / head / chosen["config"] / f"s{seed}" / "logits.npy",
        allow_pickle=False,
    )
    cutoff = chosen["threshold"] if policy == "selected" else 0.5
    return c.decisions(head, c.probabilities(head, logits), cutoff).astype(np.uint8)


def integrity():
    analysis = c.read(OUT / "analysis_plan.json")
    verify_analysis(analysis)
    before = guard()
    plan = prep.validate()
    complete = c.read(BASE / "complete.json")
    c.verify(complete["outputs"])
    data = s.read_data()
    selected_record = c.read(BASE / "selection.json")
    print(
        "Completed output bindings, sealed package and pinned cache verified.",
        flush=True,
    )
    with patch.object(c, "publish_bytes", forbid), patch.object(
        s, "arrays", assert_existing_arrays
    ):
        oof, records = audit.load_search(data, PLAN_SHA)
        with s.isolated():
            selected = c.select(data, oof)
            cells = audit.old.check_selection(data, oof, selected)
        c.require(
            selected == selected_record["families"]
            and records == selected_record["search_receipts"]
            and cells == 1232,
            "Expanded TRAIN selection does not reproduce",
        )
        final = audit.final_check(data, selected, PLAN_SHA)
        c.require(final == c.read(BASE / "final_audit.json"), "Final audit changed")
        decoded = audit.assemble(data, selected, PLAN_SHA)
        c.require(
            decoded == c.read(BASE / "prediction_audit.json"),
            "Decoded prediction audit changed",
        )
    print(
        "2,520 candidate fits, all 1,232 score cells and every delivered prediction reconstruct.",
        flush=True,
    )
    old_selection = c.read(s.OLD / "selection.json")["families"]
    same_fine = {}
    for head in c.HEADS:
        left, right = (
            selected["tuned"][head]["selected"],
            old_selection["tuned"][head]["selected"],
        )
        same_fine[head] = {
            "expanded": left,
            "original": right,
            "same_config_and_cutoff": all(
                left[k] == right[k] for k in ("config", "threshold")
            ),
        }
    array_checks = 0
    for variant in VARIANTS:
        for seed in c.FINAL_SEEDS:
            with np.load(
                BASE / "predictions" / f"{variant}_s{seed}.npz"
            ) as left, np.load(
                s.OLD / "predictions" / f"{variant}_s{seed}.npz"
            ) as right:
                for head in c.HEADS:
                    if not variant.startswith("tuned"):
                        c.require(
                            np.array_equal(left[head], right[head]),
                            "Fixed predictions changed",
                        )
                    elif head != "hate" and same_fine[head]["same_config_and_cutoff"]:
                        c.require(
                            np.array_equal(
                                left[head + "_ungated"],
                                ungated_bits(s.OLD, variant, seed, head),
                            ),
                            "Same fine-head raw decisions changed",
                        )
                    array_checks += 1
    runs, attempts, curve_groups = [], [], defaultdict(list)
    routes = audit.routes(selected)
    jobs = (
        [(s.OLD, j, "historical_search") for j in s.original_search_jobs()]
        + [(BASE, j, "new_search") for j in s.search_jobs()]
        + [
            (
                s.OLD if r["origin"] == "historical_saved_predictions" else BASE,
                r["job"],
                (
                    "historical_final"
                    if r["origin"] == "historical_saved_predictions"
                    else "new_final"
                ),
            )
            for r in routes
        ]
    )
    fresh_bindings = {}
    for base, job, origin in jobs:
        folder = base / ("search" if job["mode"] == "search" else "refits") / job["id"]
        rp = folder / "complete.json"
        receipt = c.read(rp)
        attempt = base / receipt["attempt"]
        meta = c.read(attempt / "training.json")
        spec = data["heads"][job["head"]]
        fit, held = set(meta["fit_ids"]), set(meta["held_ids"])
        fit_clusters = {r["cluster"] for r in spec["rows"] if r["id"] in fit}
        held_clusters = {r["cluster"] for r in spec["rows"] if r["id"] in held}
        c.require(
            not fit & held and not fit_clusters & held_clusters, "TRAIN group leakage"
        )
        c.require(
            all(
                np.isfinite(e["training_objective"])
                and np.isfinite(e.get("held_unweighted_loss", 0.0))
                for e in meta["epochs"]
            ),
            "Nonfinite epoch loss",
        )
        audit.old.require_learning_rate_trace(meta)
        effective = sum(not v["skipped"] and v["lr"] > 0 for v in meta["steps"])
        c.require(effective > 0, "Training never updated at positive learning rate")
        row = {
            "job": job,
            "origin": origin,
            "attempt": c.rel(attempt),
            "fit_rows": len(fit),
            "held_rows": len(held),
            "positive_LR_updates": effective,
            **{
                k: meta[k]
                for k in (
                    "attempted_steps",
                    "successful_steps",
                    "skipped_steps",
                    "train_seconds",
                    "total_seconds",
                    "peak_allocated_gib",
                    "peak_reserved_gib",
                    "configuration",
                    "export_check",
                    "changes",
                    "token_lengths",
                )
            },
            "epochs": meta["epochs"],
        }
        runs.append(row)
        if job["mode"] == "search":
            losses = [e["held_unweighted_loss"] for e in meta["epochs"]]
            curve_groups[job["head"] + "/" + job["config"]].append(
                {
                    "job": job,
                    "first_loss": losses[0],
                    "last_loss": losses[-1],
                    "minimum_loss": min(losses),
                    "earliest_minimum_epoch": losses.index(min(losses)) + 1,
                    "terminal_minus_best_loss": losses[-1] - min(losses),
                }
            )
        if base == BASE:
            fresh_bindings[c.rel(rp)] = c.sha(rp)
            fresh_bindings.update(
                {
                    c.rel(attempt / name): digest
                    for name, digest in receipt["products"].items()
                }
            )
            for candidate in sorted((folder / "attempts").iterdir()):
                if candidate.is_dir():
                    attempts.append(
                        {
                            "path": c.rel(candidate),
                            "selected": candidate == attempt,
                            "stopped": (candidate / "stopped.json").exists(),
                            "training_saved": (candidate / "training.json").exists(),
                        }
                    )
    input_rows = c.read(BASE / "data/inference.json")
    c.require(
        len(input_rows) == len({r["key"] for r in input_rows}) == 19082,
        "Inference roster",
    )
    c.require(
        not {prior.previous.norm(t) for t in data["texts"]}
        & {prior.previous.norm(r["text"]) for r in input_rows},
        "TRAIN/evaluation normalized overlap",
    )
    scope_counts = {}
    for name, rows in c.read(BASE / "data/scopes.json").items():
        c.require(
            len(rows) == len({r["source_id"] for r in rows}),
            "Scope identity duplication",
        )
        c.require(
            all(input_rows[r["position"]]["key"] == r["key"] for r in rows),
            "Scope array alignment",
        )
        scope_counts[name] = len(rows)
    resources = {}
    for origin in sorted({r["origin"] for r in runs}):
        group = [r for r in runs if r["origin"] == origin]
        resources[origin] = {
            "fits": len(group),
            **{
                k: sum(r[k] for r in group)
                for k in (
                    "attempted_steps",
                    "successful_steps",
                    "skipped_steps",
                    "positive_LR_updates",
                    "train_seconds",
                    "total_seconds",
                )
            },
            "peak_allocated_GiB_max": max(r["peak_allocated_gib"] for r in group),
        }
    save(
        "training_runs.json",
        {
            "runs": runs,
            "new_attempt_inventory": attempts,
            "costs_by_origin": resources,
            "curves": dict(curve_groups),
        },
    )
    save(
        "selection_review.json",
        {
            "families": selected,
            "original_families": old_selection,
            "tuned_changes": same_fine,
            "OOF_scores_are_selection_scores_not_generalization_estimates": True,
        },
    )
    save(
        "OOF_probabilities.npz",
        npz_bytes(
            {h + "|" + cfg: v for h, values in oof.items() for cfg, v in values.items()}
        ),
    )
    c.verify(fresh_bindings)
    verify_analysis(analysis)
    c.require(guard() == before, "Qwen sources changed during offline review")
    save(
        "integrity.json",
        {
            "status": "pass",
            "utc": c.utc(),
            "sealed_plan_sha256": PLAN_SHA,
            "sealed_bindings": len(plan["inputs"]),
            "completed_output_bindings": len(complete["outputs"]),
            "fresh_training_artifact_bindings": fresh_bindings,
            "search_fits": 2520,
            "new_search_fits": 1800,
            "reused_search_fits": 720,
            "candidate_scores_independently_verified": cells,
            "new_final_models": len(final["new"]),
            "historical_final_prediction_jobs": len(final["reused"]),
            "scopes": scope_counts,
            "prediction_audit": {k: v for k, v in decoded.items() if k != "outputs"},
            "fixed_and_ungated_array_checks": array_checks,
            "new_epoch_logit_losses_checked": sum(
                r["configuration"]["epochs"]
                for r in runs
                if r["origin"] == "new_search"
            ),
            "positive_learning_rate_updates_in_every_fit": True,
            "normalized_TRAIN_evaluation_overlap": 0,
            "new_attempts": len(attempts),
            "stopped_attempts": sum(r["stopped"] for r in attempts),
            "fresh_review_source_sha256": c.sha(Path(__file__)),
            "running_Qwen_sources_unchanged": before,
            "Qwen_calls": 0,
            "new_fits_or_model_inference": 0,
            "CUDA_used": False,
            "original_output_writes": 0,
            "external_archived_weights_rehashed": False,
            "completion_filesystem_UTC_proxy": datetime.fromtimestamp(
                (BASE / "complete.json").stat().st_mtime, timezone.utc
            ).isoformat(),
            "execution_started": c.read(BASE / "execution_started.json"),
            "resources_after_review": s.resource.resources(),
        },
    )
    print(
        "Independent saved-evidence integrity review passed; zero model calls or fits.",
        flush=True,
    )


def npz_bytes(values):
    import io

    stream = io.BytesIO()
    np.savez_compressed(stream, **values)
    return stream.getvalue()


def context(scope):
    eps = c.read(
        prior.previous.BGE / "plan.json"
        if scope == "core"
        else prior.previous.BROAD / "plan.json"
    )["endpoints"]
    positions = {
        r["source_id"]: r["position"] for r in c.read(BASE / "data/scopes.json")[scope]
    }
    return eps, positions


def bits(base, variant, seed, head, ids, positions, ep, ungated=False):
    labels = (
        s.read_data()["heads"][head]["labels"]
        if head != "hate"
        else ["not_hate", "hate"]
    )
    order = [labels.index(label) for label in ep["all_labels"]]
    with np.load(
        base / "predictions" / f"{variant}_s{seed}.npz", allow_pickle=False
    ) as saved:
        matrix = ungated_bits(base, variant, seed, head) if ungated else saved[head]
        result = matrix[[positions[k] for k in ids]][:, order]
        c.require(
            result.shape == (len(ids), len(order)) and set(np.unique(result)) <= {0, 1},
            "Invalid bits",
        )
    return result.astype(bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["freeze", "integrity"])
    args = ap.parse_args()
    with offline():
        globals()[args.mode]()


if __name__ == "__main__":
    main()

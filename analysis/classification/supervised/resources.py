from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
from analysis.classification.supervised import archive as r
from analysis.classification.supervised import metrics as m
from analysis.classification.supervised import sources as s

BASE = r.ROOT / "experiments/astra/sq2_gpu_resources_review_20261104_v1"
TRAIN = m.TRAIN
save = s.save


def freeze():
    paths = [
        Path(__file__),
        Path(r.__file__),
        r.ROOT
        / "experiments/astra/sq2_gpu_robustness_review_20261104_v1/part7_complete.json",
        r.ROOT
        / "experiments/astra/sq2_gpu_robustness_review_20261104_v1/nesting_correction/review.json",
        r.ROOT
        / "experiments/astra/sq2_gpu_training_review_20261103_v1/part1_complete.json",
        TRAIN / "complete.json",
        r.ROOT
        / "experiments/astra/sq2_gpu_quality_review_20261103_v1/part5_complete.json",
        r.ROOT
        / "experiments/astra/sq2_gpu_controls_review_20261103_v1/part3_complete.json",
        r.PACKAGE / "plan.json",
        r.PACKAGE / "manifest.json",
        r.LINEAR / "linear/complete.json",
        r.LINEAR / "features/complete.json",
        r.ROOT / "Thesis_Log/astra_result/astra_sq2_stage_b_protocol_20261020_v1.md",
        r.ROOT / "Thesis_Log/astra_result/astra_sq2_server_benchmark_20261029_v1.md",
        r.ROOT / "Thesis_Log/astra_result/astra_resource_evidence_20261030_v10.md",
    ]
    save(
        BASE / "analysis_contract.json",
        dict(
            part=8,
            utc=datetime.now(timezone.utc).isoformat(),
            scope="Supporting offline resource and practical-value analysis after Parts 1–7; no new performance selection or run.",
            questions=[
                "Reconcile accepted search/final fits, empty slots, initialization-only states, historical attempts, reused final assignments, inference volume and existing baseline/feature reuse.",
                "Sum recorded canonical fit-loop and full fit-function timers by pool, budget, head, learning rate, duration and loss. Preserve timer boundaries: epoch validation/writes are in loop time; setup before timers, post-return auditing and inference may not be.",
                "Read controller starts/ends, phase worker assignments/completion and telemetry from verified study records. Distinguish elapsed calendar span, controller-active time, phase sampling windows, overlapping worker fit-seconds and off-controller gaps. Do not label unmeasured gaps as pure downtime or infer exact standalone inference latency.",
                "Summarize observed GPU utilization/memory/temperature, container memory and storage samples with coverage/gap qualifications. No energy, carbon or monetary cost invented. Actual allocation is 4 CPU quota/3 affinity, 32 GiB RAM; preserve the 80% allocation and 250 decimal GB policy plus recorded continuation changes.",
                "Reconcile archive manifest bytes by weights, model ancillary data, records, predictions and ZIP payloads without opening weights or rehashing the full backup. Separate archived logical bytes, transferred payloads, observed/forecast server occupancy and deployment footprint.",
                "Inspect available linear baseline and feature extraction records; charge reuse once, report missing timers rather than zero. Separate earlier laptop work, capacity benchmarks and five-fit supplementary work from canonical server study cost.",
                "Compare observed cost and saved outcomes for fixed unweighted, fixed balanced and tuned families, including per-final-pipeline cost/footprint versus whole-research repeat/search cost. Default and selected cutoffs share weights; ten final assignments are exact reuse. No tuned-vs-single-rate causal gain or end-to-end RAG speed superiority is identifiable here.",
                "Keep original primary outcomes, annotation masks, complete-answer and robustness limits, including corrected same-seed nested pool chains. Use resource findings to bound practical claims without picking new models from evaluation results.",
            ],
            boundaries=[
                "No training, inference, Qwen request, server access or deletion",
                "No weights loaded or whole-backup re-audit",
                "No broader-Qwen scientific analysis",
                "No writer-owned thesis/presentation edits",
                "No fabricated energy, labour, price or unrecorded inference timings",
            ],
            inputs={str(p.relative_to(r.ROOT)): r.sha(p) for p in paths},
        ),
    )
    print("Part 8 resource scope and source bindings recorded.")


def discover():
    r.require((BASE / "analysis_contract.json").exists(), "Freeze scope first.")
    study = r.GpuStudy()
    out = {}
    with study.store("study_records") as store:
        counts = Counter(Path(n).name for n in store.entries)
        out["study_names"] = counts
        names = [
            n
            for n in store.entries
            if n.endswith("started.json")
            or n.endswith("finished.json")
            or n.endswith("telemetry.jsonl")
        ]
        out["timeline_names"] = names
        sample = {}
        for name in sorted(store.entries):
            leaf = Path(name).name
            if (
                leaf
                in (
                    "started.json",
                    "finished.json",
                    "selection_lock.json",
                    "fitting_complete_archiving_pending.json",
                )
                or "profile" in leaf
                or "storage" in leaf
            ) and name.endswith(".json"):
                value = store.json(name)
                if leaf == "selection_lock.json":
                    sample[name] = dict(keys=list(value))
                else:
                    sample[name] = value
            elif leaf == "telemetry.jsonl":
                lines = store.raw(name).splitlines()
                sample[name] = dict(
                    lines=len(lines),
                    first=r.parsed(lines[0]) if lines else None,
                    last=r.parsed(lines[-1]) if lines else None,
                )
        out["samples"] = sample
        out["study_bindings"] = store.read_bindings
    with study.pool(r.POOLS[0]) as pool:
        job = pool.jobs[0]
        name = "inference/" + job + "/complete.json"
        out["inference_example"] = pool.store.json(name)
        out["pool_manifest_summary"] = dict(
            files=len(pool.store.entries), parts=pool.store.parts.keys()
        )
    save(BASE / "discovery.json", out)
    print("Study leaf names:", dict(counts))
    print("Timeline:", len(names), "files; discovery saved.")
    print("Inference fields:", list(out["inference_example"]))
    for name, v in sample.items():
        if name.endswith(("started.json", "finished.json", "telemetry.jsonl")):
            print(
                name,
                (
                    dict(lines=v["lines"], first=v["first"], last=v["last"])
                    if "lines" in v
                    else {
                        k: v[k]
                        for k in ("utc", "status", "seconds", "stage", "detail")
                        if k in v
                    }
                ),
            )


if __name__ == "__main__":
    with r.offline():
        if sys.argv[1:] == ["freeze"]:
            freeze()
        elif sys.argv[1:] == ["discover"]:
            discover()
        else:
            raise SystemExit("Use freeze or discover.")

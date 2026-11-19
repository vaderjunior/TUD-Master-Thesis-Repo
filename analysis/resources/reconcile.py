from collections import defaultdict
from pathlib import Path
from analysis.common import replay as c

BASE = c.BASE / "documentation"
REPORTS = c.ROOT / "Thesis_Log/astra_result"
LEDGER = [
    (
        "Core prompting matrix",
        "astra_core_results_summary_20260920_v1.md",
        96822,
        8850,
        496,
        105672,
        90369228,
    ),
    (
        "SQ1 components",
        "astra_sq1_components_final_summary_20260924_v1.md",
        104895,
        9397,
        525,
        114292,
        96460637,
    ),
    (
        "SQ1 definition allocation",
        "astra_sq1_allocation_final_summary_20260925_v1.md",
        34965,
        1708,
        196,
        36673,
        40337139,
    ),
    (
        "SQ1 guidelines",
        "astra_sq1_guideline_final_results_20260926_v1.md",
        31185,
        634,
        71,
        31819,
        26216440,
    ),
    (
        "SQ1 pool composition",
        "astra_sq1_pool_final_summary_20260928_v1.md",
        128205,
        10328,
        0,
        138533,
        139202895,
    ),
    (
        "SQ3 combined explanations/taxonomy",
        "astra_sq3_combined_final_summary_20261004_v1.md",
        220077,
        7576,
        12,
        227653,
        194151413,
    ),
    (
        "SQ3 additions including benign selection",
        "astra_sq3_additions_final_summary_20261006_v1.md",
        67608,
        1347,
        150,
        68955,
        67913175,
    ),
]


def main():
    paths = [REPORTS / x[1] for x in LEDGER]
    for p in paths:
        assert p.is_file(), p
    invpath = (
        c.ROOT / "experiments/astra/qwen_downtime_inventory_20261006_v1/inventory.json"
    )
    inv = c.read(invpath)
    paths += [invpath, Path(__file__).resolve(), c.BASE / "votes/results.json"]
    training = []
    reused = []
    for job in inv["encoder_jobs"]:
        base = c.ROOT / job["attempt"]
        p = base / "training.json"
        if not p.exists():
            reused.append(job["job"])
            continue
        complete = c.read(base / "complete.json")
        assert complete["files"][c.rel(p)] == c.sha(p)
        x = c.read(p)
        paths += [p, base / "complete.json"]
        training.append(
            {
                "job": job["job"],
                "path": c.rel(p),
                "sha256": c.sha(p),
                "train_seconds": x["train_seconds"],
                "peak_allocated_gib": x["peak_allocated_gib"],
                "peak_reserved_gib": x["peak_reserved_gib"],
                "gpu": x["gpu"],
            }
        )
    c.save(
        BASE / "resource_plan.json",
        {
            "created_utc": c.utc(),
            "inputs": {c.rel(p): c.sha(p) for p in paths},
            "scope": "Selected nonoverlapping completed programme ledgers plus recorded current-version training jobs and saved slot costs. Not a whole-thesis total.",
            "missing": "No independent human/curation stopwatch, complete indexing/retrieval wall total, all historical training/reuse cost, all other study totals, electricity or bill is invented.",
        },
    )
    ledger = []
    for name, filename, ordinary, repairs, retries, success, tokens in LEDGER:
        text = (REPORTS / filename).read_text(encoding="utf8").replace(",", "")
        ledger.append(
            {
                "programme": name,
                "source": c.rel(REPORTS / filename),
                "source_sha256": c.sha(REPORTS / filename),
                "ordinary_slots": ordinary,
                "repair_slots": repairs,
                "retry_or_recovered_failure_attempts": retries,
                "attempt_slots": ordinary + repairs + retries,
                "successful_responses": success,
                "reported_tokens": tokens,
                "direct_numeric_presence": {
                    str(v): str(v) in text
                    for v in [ordinary, repairs, retries, success, tokens]
                },
            }
        )
        assert success == ordinary + repairs, (
            name,
            "Successful response accounting mismatch",
        )
    votes = c.read(c.BASE / "votes/results.json")
    costs = {mode: defaultdict(float) for mode in ["slot0", "voted"]}
    for endpoint in votes["endpoints"].values():
        for cond in endpoint["conditions"].values():
            for mode in costs:
                for k, v in cond["costs"][mode].items():
                    costs[mode][k] += v
    grouped = {}
    for arm in ["full", "kb", "de"]:
        chosen = [x for x in training if x["job"].split("/")[0] == arm]
        grouped[arm] = {
            "recorded_jobs": len(chosen),
            "train_seconds_sum": sum((x["train_seconds"] for x in chosen)),
            "max_allocated_gib": max(
                (x["peak_allocated_gib"] for x in chosen), default=None
            ),
        }
    result = {
        "created_utc": c.utc(),
        "plan_sha256": c.sha(BASE / "resource_plan.json"),
        "selected_programmes": ledger,
        "selected_programmes_only_totals": {
            k: sum((x[k] for x in ledger))
            for k in [
                "ordinary_slots",
                "repair_slots",
                "retry_or_recovered_failure_attempts",
                "attempt_slots",
                "successful_responses",
                "reported_tokens",
            ]
        },
        "training_receipts": training,
        "training_summary": grouped,
        "reused_prediction_only_jobs_training_not_counted": reused,
        "core_rag_only_saved_slot_costs": dict(costs),
        "new_model_calls": 0,
        "limits": [
            "Success tokens exclude unknown failed or interrupted remote work",
            "Elapsed request durations overlap under concurrency; their sum is not runtime",
            "The core ledger retains administrative unknown outcomes; attempted slots do not certify all remote execution",
            "Benign selection is already included in additions; do not count it again in balanced-correction execution",
            "No completed balanced-correction result or cost total is asserted",
        ],
    }
    c.save(BASE / "resources.json", result)
    print("Resource ledger saved.", grouped)
    print("Selected programmes only:", result["selected_programmes_only_totals"])


if __name__ == "__main__":
    with c.network_guard():
        main()

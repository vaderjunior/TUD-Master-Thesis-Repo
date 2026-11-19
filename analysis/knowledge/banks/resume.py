"""Banks resume."""

from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
from analysis.knowledge.banks import build as core

HERE = core.BASE / "continuation_v1"
POLICY = HERE / "policy.json"


def quality(info):
    """Attempt completion is separate from producing a valid prediction."""
    core.require(
        info["missing"] == 0,
        "Missing predictions remain; resume the same continuation later.",
    )
    core.require(
        0 <= info["invalid_results"] < info["expected"],
        "Entire run has no valid predictions; preserve it and investigate before continuing.",
    )
    return {
        **info,
        "valid_results": info["expected"] - info["invalid_results"],
        "status": (
            "complete_with_recorded_failures" if info["invalid_results"] else "complete"
        ),
    }


def validate_completed_setup(plan):
    """Use the completed index/retrieval receipts without re-embedding."""
    for pool in core.POOLS:
        expected = {
            **core.index_identity(plan, pool),
            "n_records": plan["pool_counts"][pool]["total_records"],
        }
        core.require(
            core.read_json(core.BASE / "kb" / pool / "index_complete.json") == expected,
            f"Completed index receipt differs: {pool}",
        )
    receipt = core.read_json(core.BASE / "retrieval_complete.json")
    core.require(
        receipt["plan_sha256"] == core.digest(core.PLAN)
        and receipt["abstract_context_equal"],
        "Retrieval check receipt differs",
    )
    counts = {r["scope"]: r["n_items"] for r in plan["runs"]}
    core.require(
        receipt["n_inputs"] == sum(counts.values()), "Retrieval input coverage differs"
    )
    for pool in core.POOLS:
        core.require(
            core.digest(core.BASE / f"retrieval_{pool}.json")
            == receipt["hashes"][pool],
            f"Saved retrieval contexts changed: {pool}",
        )


def prepare():
    if POLICY.exists():
        validate()
        print("Continuation already prepared and unchanged.")
        return
    plan = core.validate_plan()
    validate_completed_setup(plan)
    completed = []
    preserved = {}
    for run in plan["runs"]:
        info = core.check_output(plan, run)
        path = core.ROOT / run["output"]
        if path.exists():
            core.require(
                info["missing"] == 0,
                "Review partial existing runs before preparing this continuation",
            )
            completed.append({"name": run["name"], **quality(info)})
            preserved[run["output"]] = core.digest(path)
            metrics = path.with_suffix(".metrics.json")
            if metrics.exists():
                preserved[core.rel(metrics)] = core.digest(metrics)
    files = [
        Path(__file__),
        core.ROOT / "scripts/resume_bank_comparison.ps1",
        core.ROOT / "tests/knowledge/banks/test_resume.py",
    ]
    logs = {
        core.rel(p): core.digest(p) for p in sorted((core.BASE / "logs").glob("*.log"))
    }
    policy = {
        "created_utc": core.now(),
        "original_plan_sha256": core.digest(core.PLAN),
        "continuation_files": {core.rel(p): core.digest(p) for p in files},
        "preserved_results": preserved,
        "reviewed_logs": logs,
        "reviewed_completed_runs": completed,
        "original_stop": core.read_json(core.BASE / "batch_status.json"),
        "change": "A complete set of attempted items may contain recorded invalid predictions. Keep them as failures, report them, and continue without resampling them. An entirely invalid run still stops the batch. All original provenance checks remain.",
        "analysis_requirement": "Report attempted and valid coverage per condition; compare scores on matched valid item pools with explicit failure accounting. Do not interpret differently filtered per-run scores as a paired KB effect.",
    }
    core.write_once(POLICY, core.json_bytes(policy))
    print(
        json.dumps(
            {
                "policy": str(POLICY),
                "reviewed_runs": completed,
                "remaining_runs": len(plan["runs"]) - len(completed),
            },
            indent=2,
        )
    )
    print("PREPARED ONLY. No model loading, retrieval, API calls or result edits.")


def validate():
    plan = core.validate_plan()
    policy = core.read_json(POLICY)
    core.require(
        policy["original_plan_sha256"] == core.digest(core.PLAN),
        "Original plan changed",
    )
    for key in ("continuation_files", "preserved_results", "reviewed_logs"):
        for name, expected in policy[key].items():
            core.require(
                core.digest(core.ROOT / name) == expected,
                f"Reviewed/frozen file changed: {name}",
            )
    validate_completed_setup(plan)
    return plan


def write_quality(plan, run, info):
    rows = [r for r in core.read_lines(core.ROOT / run["output"]) if "item_id" in r]
    invalid = [
        r["item_id"] for r in rows if r.get("result") is None or r["n_valid"] == 0
    ]
    report = {
        "run": run["name"],
        **quality(info),
        "invalid_item_ids": invalid,
        "valid_vote_counts": dict(Counter((str(r["n_valid"]) for r in rows))),
        "result_sha256": core.digest(core.ROOT / run["output"]),
        "original_plan_sha256": core.digest(core.PLAN),
        "continuation_policy_sha256": core.digest(POLICY),
    }
    core.write_once(HERE / "quality" / f"{run['name']}.json", core.json_bytes(report))


def run_queue(plan):
    completed = []
    for index, run in enumerate(plan["runs"], 1):
        core.validate_plan(check_cache=False)
        info = core.check_output(plan, run)
        print(
            f"BATCH {index}/{len(plan['runs'])}: {run['name']}; missing {info['missing']}",
            flush=True,
        )
        for attempt in range(2):
            if not info["missing"]:
                break
            core.status_write(
                {
                    "status": "classifying",
                    "continuation": "v1",
                    "run": run["name"],
                    "run_number": index,
                    "total_runs": len(plan["runs"]),
                    "attempt": attempt + 1,
                    "completed": completed,
                }
            )
            core.subprocess_logged(
                [
                    "-m",
                    "scripts.run_comparison",
                    "--manifest",
                    run["manifest"],
                    "--out",
                    run["output"],
                    "--yes",
                ],
                run["name"],
            )
            info = core.check_output(plan, run)
        summary = quality(info)
        if summary["invalid_results"]:
            print(
                f"RECORDED FAILURE: {summary['invalid_results']}/{summary['expected']} items have no valid prediction. Preserved without retry; continuing with the same declared protocol.",
                flush=True,
            )
        write_quality(plan, run, info)
        completed.append({"run": run["name"], **summary})
    return completed


def run():
    with core.run_lock():
        plan = validate()
        marker = core.BASE / "batch_complete.json"
        if marker.exists():
            receipt = core.read_json(marker)
            core.require(
                receipt["plan_sha256"] == core.digest(core.PLAN),
                "Completion plan differs",
            )
            for name, expected in receipt["result_hashes"].items():
                core.require(
                    core.digest(core.ROOT / name) == expected,
                    f"Completed result changed: {name}",
                )
            print("ALL 12 RUNS COMPLETE already; no new calls.")
            return
        core.offline_env(plan)
        core.require_cuda()
        core.check_service(plan)
        print(
            "Completed indexes and retrieval checks verified. Reusing them; no rebuilding or retrieval preflight."
        )
        completed = run_queue(plan)
        validate()
        hashes = {
            r["output"]: core.digest(core.ROOT / r["output"]) for r in plan["runs"]
        }
        core.write_once(
            marker,
            core.json_bytes(
                {
                    "plan_sha256": core.digest(core.PLAN),
                    "continuation_policy_sha256": core.digest(POLICY),
                    "completed_utc": core.now(),
                    "runs": completed,
                    "result_hashes": hashes,
                }
            ),
        )
        core.status_write(
            {"status": "complete", "continuation": "v1", "completed": completed}
        )
        failed = sum((r["invalid_results"] for r in completed))
        print(
            f"ALL {len(completed)} RUNS COMPLETE. Recorded invalid predictions: {failed}.",
            flush=True,
        )
        print(f"Results: {core.BASE / 'results'}")
        print(
            "Tell Astra the batch finished. Coverage and paired statistical analysis come next."
        )


def main():
    os.chdir(core.ROOT)
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=("prepare", "validate", "run", "status"))
    stage = ap.parse_args().stage
    if stage == "prepare":
        prepare()
    elif stage == "validate":
        plan = validate()
        for item in plan["runs"]:
            info = core.check_output(plan, item)
            print(item["name"], json.dumps(info))
        print("Continuation validated offline. No model/API execution.")
    elif stage == "status":
        print(json.dumps(core.read_json(core.BASE / "batch_status.json"), indent=2))
    else:
        run()


if __name__ == "__main__":
    main()

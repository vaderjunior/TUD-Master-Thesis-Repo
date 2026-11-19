"""Common execution history."""

from datetime import datetime
from analysis.common import batch as b

a = b.a
ROOT = b.ROOT
BASE = b.BASE
REVIEW = (
    ROOT
    / "Thesis_Log/astra_result/evidence/core_api_stage09_interruption_20260919_v1.json"
)
REVIEW_SHA = "f5e1ffd4bdafaf20fbf2c58bab7196b1fc69fb64d24a6ff8f5317490e2084c03"
ATTEMPTS = ["attempt_1789802405289506000", "attempt_1789802677992314700"]


def validate_partition(jobs, preserved, summary):
    identities = [a.signature(b.identity(j)) for j in jobs]
    a.require(
        len(jobs) == 2700 and len(set(identities)) == 2700,
        "Wrong/duplicate stage roster",
    )
    a.require(preserved == jobs[:8], "Reviewed eight preserved jobs differ")
    a.require(
        summary["submitted"] == 2692
        and summary["saved_items"] == 2700
        and (summary["unsubmitted"] == 0),
        "Resume skipped or resubmitted the wrong jobs",
    )
    a.require(
        not summary["errors"] and (not summary["stopped"]), "Resumed stage stopped"
    )


def validate_epoch(preserved, completed_utc, resume_utc):
    before = datetime.fromisoformat(completed_utc) < datetime.fromisoformat(resume_utc)
    a.require(before == preserved, "Item completion belongs to the wrong attempt")


def inspect(stage, record, plan, target):
    a.require(
        stage["id"] == "stage_09_secondary", "History review only supports stage 09"
    )
    a.require(a.digest(REVIEW) == REVIEW_SHA, "Interruption review changed")
    review = a.read(REVIEW)
    a.require(
        review["status"] == "clean_interruption_reviewed_user_resume_already_running"
        and review["execution_plan_sha256"] == a.digest(b.PLAN),
        "Wrong interruption review",
    )
    a.require(
        review["administrative_outcomes_added"] == 0
        and review["actual_API_calls"] == 0,
        "Unexpected recovery intervention",
    )
    a.require(
        review["pre_resume_inventory_observation"]
        == {
            "completed": 8,
            "not_started": 2692,
            "partial_items": 0,
            "unresolved_intents": 0,
            "qualification": "Initial read-only inventory observed before user restart; completed records subsequently independently audited and bound here. Not a hash snapshot of mutable active work.",
        },
        "Interruption scope differs",
    )
    attempts = sorted(target.glob("attempt_*"))
    a.require([p.name for p in attempts] == ATTEMPTS, "Unexpected attempt history")
    a.require(
        not (attempts[0] / "summary.json").exists(),
        "Interrupted summary must remain absent",
    )
    a.require(
        {p.name for p in attempts[0].iterdir()} == {"start.json"},
        "Unexpected first-attempt files",
    )
    a.require(
        {p.name for p in attempts[1].iterdir()} == {"start.json", "summary.json"},
        "Unexpected resumed-attempt files",
    )
    starts = [a.read(p / "start.json") for p in attempts]
    for start in starts:
        a.require(
            start["stage"] == stage["id"]
            and start["plan_sha256"] == a.digest(b.PLAN)
            and (start["environment"] == plan["environment"]),
            "Attempt attribution differs",
        )
    a.require(
        starts[0]["started_utc"] == review["original_start_utc"]
        and starts[1]["started_utc"] == review["resume_start_utc"],
        "Recorded start times differ",
    )
    summary = a.read(attempts[1] / "summary.json")
    a.require(summary == record["summary"], "Resumed/final summaries differ")
    preserved = [r["job"] for r in review["completed_jobs_before_resume"]]
    validate_partition(stage["jobs"], preserved, summary)
    for mapping in ("verified_input_hashes", "verified_output_hashes"):
        for name, h in review[mapping].items():
            a.require(
                a.digest(ROOT / name) == h, "Pre-resume evidence changed: " + name
            )
    before_ids = {a.signature(b.identity(j)) for j in preserved}
    latest = None
    for job in stage["jobs"]:
        folder = b.item_dir(BASE / "live", job)
        cp = folder / "complete.json"
        receipt = a.read(cp)
        before = folder.name in before_ids
        validate_epoch(before, receipt["created_utc"], starts[1]["started_utc"])
        if before:
            a.require(
                b.rel(cp) in review["verified_output_hashes"],
                "Retained receipt not hash-bound",
            )
            latest = max(latest or receipt["created_utc"], receipt["created_utc"])
    paths = [
        REVIEW,
        ROOT / "analysis/common/execution_history.py",
        ROOT / "tests/common/test_execution_history.py",
        attempts[0] / "start.json",
        attempts[1] / "start.json",
        attempts[1] / "summary.json",
    ]
    inputs = {b.rel(p): a.digest(p) for p in paths}
    inputs.update(review["verified_input_hashes"])
    return {
        "inputs": inputs,
        "start": starts[0],
        "resume_start": starts[1],
        "summary": {
            "status": "clean_two_attempt_history_verified",
            "interruption_evidence_sha256": REVIEW_SHA,
            "attempts": ATTEMPTS,
            "completed_items_preserved": 8,
            "resumed_new_items": 2692,
            "pre_resume_output_hashes_verified": len(review["verified_output_hashes"]),
            "administrative_unknown_outcomes": 0,
            "first_summary_preserved_absent": True,
            "last_pre_resume_item_complete_utc": latest,
            "resume_start_utc": starts[1]["started_utc"],
            "resumed_wall_seconds_including_final_save": (
                datetime.fromisoformat(record["completed_utc"])
                - datetime.fromisoformat(starts[1]["started_utc"])
            ).total_seconds(),
            "gap_last_initial_completion_to_resume_seconds_not_exact_downtime": (
                datetime.fromisoformat(starts[1]["started_utc"])
                - datetime.fromisoformat(latest)
            ).total_seconds(),
            "timing_qualification": "Overall first-start-to-completion wall span includes the interruption gap; not active runtime. No administrative unknown-duration sentinel exists here; observed read timeouts still leave remote work unknown.",
        },
    }

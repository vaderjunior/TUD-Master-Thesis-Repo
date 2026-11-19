from datetime import datetime
from analysis.common import batch as b
from analysis.common import recover_interruption as recovery

a = b.a
ROOT = b.ROOT
BASE = b.BASE


def inspect(stage, stage_record, execution_plan):
    a.require(
        stage["id"] == "stage_02_primary", "This recovery review is scoped to stage 02"
    )
    folder = BASE / "recovery_stage02_20260917_v1"
    plan_path = folder / "plan.json"
    p = a.read(plan_path)
    a.require(
        a.digest(plan_path)
        == "85bb302af25258a2d4e5906dc98a05fc1e78603034b6ca1a3de032d4d4544045",
        "Reviewed recovery plan changed",
    )
    handoff_path = (
        ROOT
        / "Thesis_Log/astra_result/evidence/core_api_stage02_recovery_handoff_20260917_v1.json"
    )
    a.require(
        a.digest(handoff_path)
        == "5fec6f0759d60268112a802460864f64e5caddc04fc8e4e53444779c924bd8a2",
        "Recovery handoff changed",
    )
    handoff = a.read(handoff_path)
    for name, h in handoff["verified_hashes"].items():
        a.require(a.digest(ROOT / name) == h, "Recovery source/launcher/tests changed")
    applied = a.read(folder / "complete.json")
    a.require(
        applied["status"]
        == "interruption_records_added_original_stage_command_can_resume",
        "Recovery was not completed",
    )
    a.require(
        applied["recovery_plan_sha256"] == a.digest(plan_path)
        and applied["execution_plan_sha256"] == a.digest(b.PLAN),
        "Recovery plan binding changed",
    )
    a.require(
        applied["source_sha256"]
        == p["source_sha256"]
        == a.digest(ROOT / "analysis/common/recover_interruption.py"),
        "Recovery implementation changed",
    )
    tests = a.read(folder / "tests.json")
    a.require(
        tests["passed"] and tests["source_sha256"] == p["source_sha256"],
        "Recovery tests do not match",
    )
    a.require(
        applied["actual_API_calls"] == 0
        and applied["completed_items_preserved"] == p["completed_items"] == 2418,
        "Recovery scope/side-effect record differs",
    )
    a.require(
        applied["preexisting_files_verified_unchanged"]
        == len(p["existing_hashes"])
        == 31649,
        "Wrong pre-interruption inventory count",
    )
    for index, (name, h) in enumerate(p["existing_hashes"].items(), 1):
        a.require(
            a.digest(ROOT / name) == h, "Pre-interruption evidence changed: " + name
        )
        if index % 10000 == 0:
            print(
                f"Preserved-file check: {index}/{len(p['existing_hashes'])} hashes verified",
                flush=True,
            )
    target = BASE / "stages" / stage["id"]
    attempts = sorted(target.glob("attempt_*"))
    a.require(
        [x.name for x in attempts]
        == ["attempt_1789631798376405900", "attempt_1789643174600762200"],
        "Unexpected stage-02 attempt history",
    )
    original_start = a.read(attempts[0] / "start.json")
    resume_start = a.read(attempts[1] / "start.json")
    a.require(
        not (attempts[0] / "summary.json").exists(),
        "Interrupted attempt history was overwritten",
    )
    for record in (original_start, resume_start):
        a.require(
            record["plan_sha256"] == a.digest(b.PLAN)
            and record["stage"] == stage["id"]
            and (record["environment"] == execution_plan["environment"]),
            "Attempt plan/environment differs",
        )
    summary = a.read(attempts[1] / "summary.json")
    a.require(
        summary == stage_record["summary"], "Final summary differs from resumed attempt"
    )
    a.require(
        summary["submitted"] == 282
        and (not summary["errors"])
        and (not summary["stopped"]),
        "Unexpected resume submissions/stop",
    )
    a.require(
        datetime.fromisoformat(p["created_utc"])
        < datetime.fromisoformat(applied["created_utc"])
        <= datetime.fromisoformat(resume_start["started_utc"]),
        "Recovery/resume timeline differs",
    )
    partial = {a.signature(b.identity(x)) for x in p["partial_items"]}
    unstarted = {a.signature(b.identity(x)) for x in p["not_started_items"]}
    all_jobs = {a.signature(b.identity(x)) for x in stage["jobs"]}
    complete_before = {
        name.split("/items/")[1].split("/")[0]
        for name in p["existing_hashes"]
        if "/items/" in name and name.endswith("/complete.json")
    }
    a.require(
        len(partial) == 8 and len(unstarted) == 274 and (len(complete_before) == 2418),
        "Wrong interrupted partition sizes",
    )
    a.require(
        not (
            partial & unstarted
            or partial & complete_before
            or unstarted & complete_before
        ),
        "Recovery partitions overlap",
    )
    a.require(
        partial | unstarted | complete_before == all_jobs,
        "Recovery partitions omit or add scheduled jobs",
    )
    expected_outcomes = {x["outcome"] for x in p["missing_outcomes"]}
    a.require(
        len(expected_outcomes) == 8
        and set(applied["administrative_unknown_outcomes_added"]) == expected_outcomes,
        "Wrong recovery-outcome roster",
    )
    unknown_votes = set()
    rows = []
    review_times = set()
    for entry in p["missing_outcomes"]:
        intent_path = ROOT / entry["intent"]
        out_path = ROOT / entry["outcome"]
        old = a.read(intent_path)
        out = a.read(out_path)
        a.require(
            entry["outcome"] not in p["existing_hashes"],
            "Administrative outcome was not originally absent",
        )
        a.require(
            a.digest(intent_path) == entry["intent_sha256"],
            "Interrupted intent changed",
        )
        a.require(
            a.digest(out_path)
            == applied["administrative_unknown_outcomes_added"][entry["outcome"]],
            "Administrative recovery record changed",
        )
        a.require(
            out == recovery.resolution(old, a.digest(plan_path), out["finished_utc"]),
            "Recovery record is not the reviewed explicit unknown-outcome form",
        )
        a.require(
            datetime.fromisoformat(p["created_utc"])
            <= datetime.fromisoformat(out["finished_utc"])
            <= datetime.fromisoformat(applied["created_utc"]),
            "Administrative timestamp is outside recovery interval",
        )
        job_dir = b.item_dir(BASE / "live", entry["job"])
        expected = (
            job_dir
            / "calls"
            / f"vote_{old['vote']}_turn_{old['turn']}/attempt_0/outcome.json"
        )
        a.require(
            b.rel(expected) == entry["outcome"] and old["attempt"] == 0,
            "Wrong recovered vote/turn/attempt",
        )
        next_intent_path = expected.parent.parent / "attempt_1/intent.json"
        next_outcome = next_intent_path.with_name("outcome.json")
        retry = a.read(next_intent_path)
        a.require(
            next_outcome.exists(),
            "Interrupted call did not retry in next remaining slot",
        )
        for key in ("job_stamp", "vote", "turn", "kind", "body_sha256"):
            a.require(
                retry[key] == old[key],
                "Recovered retry changed request identity/payload",
            )
        a.require(
            retry["attempt"] == 1
            and datetime.fromisoformat(retry["started_utc"])
            >= datetime.fromisoformat(resume_start["started_utc"]),
            "Recovery reset attempt budget or resumed before recorded start",
        )
        pair = (job_dir.name, old["vote"])
        a.require(
            pair not in unknown_votes, "Multiple unknown durations in an affected vote"
        )
        unknown_votes.add(pair)
        review_times.add(out["finished_utc"])
        rows.append(
            {
                "job": entry["job"],
                "vote": old["vote"],
                "turn": old["turn"],
                "consumed_attempt": 0,
                "resumed_attempt": 1,
                "intent_preserved": True,
                "identical_retry_payload": True,
                "outcome": entry["outcome"],
            }
        )
    a.require(
        len(review_times) == 1, "Recovery records were not one reviewed application"
    )
    null_before = []
    latest = None
    for name in p["existing_hashes"]:
        if name.endswith("/complete.json") and "/items/" in name:
            stamp = a.read(ROOT / name)["created_utc"]
            latest = max(latest or stamp, stamp)
        if name.endswith("/result.json") and "/items/" in name:
            result = a.read(ROOT / name)
            if result["result"] is None:
                null_before.append(result["identity"])
    saved_partial_votes = sum(
        (
            "/items/" + key + "/votes/" in name
            for key in partial
            for name in p["existing_hashes"]
        )
    )
    a.require(saved_partial_votes == 10, "Saved partial vote count differs")
    inputs = {
        b.rel(path): a.digest(path)
        for path in [
            plan_path,
            folder / "complete.json",
            folder / "tests.json",
            folder / "handoff.json",
            handoff_path,
            ROOT / "analysis/common/recover_interruption.py",
            ROOT / "scripts/resume_core_interruption.ps1",
            ROOT / "analysis/common/validate_interruption.py",
            attempts[0] / "start.json",
            attempts[1] / "start.json",
            attempts[1] / "summary.json",
        ]
    }
    return {
        "inputs": inputs,
        "unknown_vote_pairs": unknown_votes,
        "administrative_outcome_paths": expected_outcomes,
        "original_start": original_start,
        "resume_start": resume_start,
        "summary": {
            "status": "all_pre_interruption_evidence_and_eight_recoveries_verified",
            "pre_interruption_hashes_verified": len(p["existing_hashes"]),
            "completed_predictions_preserved": len(complete_before),
            "partial_items_preserved": len(partial),
            "saved_partial_votes_preserved": saved_partial_votes,
            "previously_unstarted_predictions": len(unstarted),
            "resumed_items": summary["submitted"],
            "administrative_unknown_outcomes": 8,
            "unknown_duration_votes": 8,
            "recovered_requests": rows,
            "null_predictions_already_complete_before_interruption": null_before,
            "last_pre_interruption_completed_utc": latest,
            "original_start_utc": original_start["started_utc"],
            "resume_start_utc": resume_start["started_utc"],
            "recovery_applied_utc": applied["created_utc"],
            "post_resume_wall_seconds_including_final_save": (
                datetime.fromisoformat(stage_record["completed_utc"])
                - datetime.fromisoformat(resume_start["started_utc"])
            ).total_seconds(),
            "gap_last_saved_completion_to_resume_seconds_not_exact_downtime": (
                datetime.fromisoformat(resume_start["started_utc"])
                - datetime.fromisoformat(latest)
            ).total_seconds(),
            "qualifications": "All eight records explicitly mark request delivery/completion/usage/duration unknown. Zero elapsed_s is a compatibility sentinel, not a measured duration. Later retry is identical and consumes remaining original budget. Interrupted gap is not an exact measurement of idle time.",
        },
    }

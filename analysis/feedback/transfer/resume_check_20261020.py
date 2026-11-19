from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import json
import msvcrt
from pathlib import Path
import socket
from unittest.mock import patch
from analysis.feedback.transfer.launch_check_20261020 import ROOT
from analysis.feedback.transfer.launch_check_20261020 import BASE
from analysis.feedback.transfer.launch_check_20261020 import need
from analysis.feedback.transfer.launch_check_20261020 import physical
from analysis.feedback.transfer.launch_check_20261020 import read
from analysis.feedback.transfer.launch_check_20261020 import rel
from analysis.feedback.transfer.launch_check_20261020 import sha

OUT = ROOT / "experiments/astra/sq3_transfer_resume_check_20261020_v1"
PLAN_SHA = "ca08393271cfe2461966376daebe45d5d156dcf0acc38f4ce0e9f480f8a2ed5d"


def forbidden(*args, **kwargs):
    raise RuntimeError(
        "Resume inspection forbids network, execution and original evidence writes"
    )


def save_new(path, value):
    need(path.resolve().is_relative_to(OUT.resolve()), "Check output boundary")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(
            value, stream, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False
        )
        stream.write("\n")


def inventory(path, held_lock_hashes):
    return {
        rel(p): held_lock_hashes[rel(p)] if rel(p) in held_lock_hashes else sha(p)
        for p in physical(path).rglob("*")
        if p.is_file()
    }


@contextmanager
def existing_lock(path):

    path = physical(path)
    need(path.is_file() and path.stat().st_size >= 1, "Expected existing lock absent")
    with path.open("r+b") as stream:
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def main():
    from analysis.common import runtime as c
    from analysis.feedback.transfer import run as api
    from analysis.feedback.common import queue as q
    import requests

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    out = OUT / "checks" / stamp
    started = c.utc()
    with c.offline(), patch.object(
        socket.socket, "connect_ex", forbidden
    ), patch.object(requests.sessions.Session, "request", forbidden), patch.object(
        q.template.core, "live_client", forbidden
    ), patch.object(
        q.template.a.RequestsTransport, "__call__", forbidden
    ), patch.object(
        q.template.a, "save_new", forbidden
    ):
        b = api.queue()
        live = physical(BASE / "execution/live")
        root = live / "transfer"
        locks = [
            c.SHARED / "qwen_queue.lock",
            b.BASE / "launcher.lock",
            root / "launcher.lock",
        ]
        lock_before = {rel(p): sha(p) for p in locks}
        with ExitStack() as held:
            for path in locks:
                held.enter_context(existing_lock(path))
            need(sha(b.PLAN) == PLAN_SHA, "Frozen transfer plan changed")
            q.require_a15_finished()
            plan = b.validate(ready=True)
            print(
                "Frozen transfer bindings, predecessor and three existing queue locks verified.",
                flush=True,
            )
            before = inventory(live, lock_before)
            save_new(out / "live_before.json", before)
            paths = [physical(ROOT / name) for name in before]
            stop_files = [rel(p) for p in paths if p.name.startswith("stop_")]
            partials = [rel(p) for p in paths if ".partial_" in p.name]
            intents = {p.parent for p in paths if p.name == "intent.json"}
            outcomes = {p.parent for p in paths if p.name == "outcome.json"}
            need(
                not stop_files and (not partials),
                "Stop marker or partial atomic write requires recovery review",
            )
            need(
                intents == outcomes,
                "Unknown request or orphan outcome requires recovery review",
            )
            item_roots = {p.name: p for p in (root / "items").iterdir() if p.is_dir()}
            need(bool(item_roots), "No saved items")
            schedule = read(b.BASE / "schedule.json")
            need(
                len(schedule["stages"]) == plan["budget"]["batches"] == 22,
                "Batch count changed",
            )
            present_jobs = {}
            touched_stages = []
            all_jobs = 0
            for stage in schedule["stages"]:
                need(stage["items"] == len(stage["jobs"]), "Scheduled count differs")
                all_jobs += len(stage["jobs"])
                matched = []
                for job in stage["jobs"]:
                    ident = b.a.signature(b.identity(job, PLAN_SHA))
                    if ident in item_roots:
                        need(ident not in present_jobs, "Saved item assigned twice")
                        present_jobs[ident] = job
                        matched.append(job)
                if matched:
                    touched_stages.append((stage, matched))
            need(
                all_jobs == plan["budget"]["predictions"] == 189096,
                "Scientific size changed",
            )
            need(
                set(present_jobs) == set(item_roots),
                "Saved item outside frozen schedule",
            )
            wanted = {(j["key"], j["condition"]) for j in present_jobs.values()}
            schema = ROOT / "config/taxonomy.yaml"
            schema_sha = sha(schema)
            science_sha = sha(BASE / "scientific_plan_v1.json")
            cat = {}
            for row in c.rows(BASE / "prompts.jsonl"):
                key = (row["key"], row["condition"])
                if key not in wanted:
                    continue
                need(key not in cat, "Duplicate saved-item prompt")
                need(
                    row["scientific_plan_sha256"] == science_sha
                    and b.a.signature(row["messages"]) == row["messages_sha256"]
                    and (b.a.signature(row["context"]) == row["context_sha256"]),
                    "Prompt binding differs",
                )
                cat[key] = {
                    "messages": row["messages"],
                    "messages_sha256": row["messages_sha256"],
                    "schema_path": rel(schema),
                    "schema_sha256": schema_sha,
                    "population": row["population"],
                    "entry_sha256": b.a.signature(row),
                }
            need(set(cat) == wanted, "Saved-item prompt missing")
            backends = b.engines(cat)
            counts = Counter()
            stages = []
            for stage, jobs in touched_stages:
                complete_count = pending_count = null_count = 0
                for i in range(0, len(jobs), 100):
                    existing, todo = b.scan_jobs(
                        jobs[i : i + 100], cat, root, PLAN_SHA, "live", backends
                    )
                    complete_count += len(existing)
                    pending_count += len(todo)
                    for _, result in existing:
                        counts.update(result["accounting"])
                        null_count += result["result"] is None
                    print(
                        f"{stage['id']}: checked {min(i + 100, len(jobs))}/{len(jobs)} saved items.",
                        flush=True,
                    )
                target = root / "stages" / stage["id"]
                complete_path = target / "complete.json"
                review_path = target / "review.json"
                need(
                    complete_path.exists() == review_path.exists(),
                    "Interrupted stage audit requires separate inspection",
                )
                if complete_path.exists():
                    b.verify_review(stage, root, PLAN_SHA, "live")
                need(
                    not (complete_count >= 20 and null_count / complete_count >= 0.1),
                    "Saved invalid-output rate requires review",
                )
                stages.append(
                    {
                        "stage": stage["id"],
                        "completed_verified": complete_count,
                        "partial_items_checked": pending_count,
                        "not_started": stage["items"] - len(jobs),
                        "batch_items": stage["items"],
                        "null_results": null_count,
                        "stage_complete": complete_path.exists(),
                        "stage_reviewed": review_path.exists(),
                    }
                )
            need(
                sum(
                    (
                        s["completed_verified"] + s["partial_items_checked"]
                        for s in stages
                    )
                )
                == len(item_roots),
                "Saved-item audit coverage",
            )
            stage_ids = {s["id"] for s in schedule["stages"]}
            need(
                all(
                    (
                        p.name in stage_ids
                        for p in (root / "stages").iterdir()
                        if p.is_dir()
                    )
                ),
                "Unexpected stage directory",
            )
            interrupts = {
                rel(p): read(p) for p in paths if p.name == "interrupted.json"
            }
            starts = {rel(p): read(p) for p in paths if p.name == "start.json"}
            need(
                bool(interrupts),
                "Expected interruption receipt absent; inspect shutdown",
            )
            b.validate(ready=True)
            after = inventory(live, lock_before)
            need(after == before, "Live evidence changed during offline inspection")
            completed = sum((s["completed_verified"] for s in stages))
            receipt = {
                "status": "clean_saved_state_verified_for_existing_user_resume",
                "started_utc": started,
                "finished_utc": c.utc(),
                "plan_sha256": PLAN_SHA,
                "science_sha256": science_sha,
                "frozen_bindings_verified": len(
                    set(plan["input_hashes"]) | set(plan["source_hashes"])
                ),
                "check_sources": {
                    rel(Path(__file__)): sha(Path(__file__)),
                    "analysis/feedback/transfer/launch_check_20261020.py": sha(
                        ROOT / "analysis/feedback/transfer/launch_check_20261020.py"
                    ),
                },
                "evidence_inventory": rel(out / "live_before.json"),
                "evidence_inventory_sha256": sha(out / "live_before.json"),
                "original_live_files_unchanged": len(before),
                "lock_files_unchanged": lock_before,
                "locks_held_through_inspection": True,
                "predecessor_complete": True,
                "total_predictions": all_jobs,
                "completed_predictions_verified": completed,
                "completion_percent": completed * 100 / all_jobs,
                "remaining_predictions": all_jobs - completed,
                "saved_requests": len(intents),
                "saved_outcomes": len(outcomes),
                "unknown_outcomes": 0,
                "orphan_outcomes": 0,
                "partial_atomic_files": 0,
                "execution_stop_markers": 0,
                "saved_accounting": dict(counts),
                "stages": stages,
                "interruptions": interrupts,
                "original_attempt_starts": starts,
                "completed_item_validation": "frozen scan_jobs with full saved-response parse/vote replay",
                "not_a_scientific_result_review": True,
                "network_calls": 0,
                "model_list_GETs": 0,
                "Qwen_POSTs": 0,
                "training_or_embeddings": 0,
                "live_files_written": 0,
                "source_or_protocol_changes": 0,
                "resume_launched": False,
                "resume_requires_user_command": True,
                "command": 'powershell.exe -NoProfile -ExecutionPolicy Bypass -File "'
                + str(ROOT / "scripts/run_feedback_transfer.ps1")
                + '" -Stage run',
            }
        need({rel(p): sha(p) for p in locks} == lock_before, "Lock bytes changed")
        receipt["lock_bytes_verified_after_release"] = True
        receipt["prior_checker_diagnostic"] = {
            "reason": "Initial offline checker attempted to hash a held Windows byte lock through another handle.",
            "exception": "PermissionError",
            "occurred_before_live_inventory_or_audit": True,
            "original_evidence_writes": 0,
            "network_calls": 0,
            "earlier_checker_source": "scratch/analysis/operational_notes_20261020_v1/transfer_resume_checker_before_lock_read_fix.py",
            "earlier_checker_source_sha256": "ef408ff63f0491b00c838a0a6bea6b7bcf5cb0d07a89f1f4426931b4830df5ea",
            "correction": "Hash lock bytes before acquisition and after release; compare all other bytes while held.",
        }
        save_new(out / "complete.json", receipt)
        print(
            json.dumps(
                {
                    k: receipt[k]
                    for k in (
                        "status",
                        "completed_predictions_verified",
                        "remaining_predictions",
                        "completion_percent",
                        "saved_requests",
                        "unknown_outcomes",
                        "original_live_files_unchanged",
                        "network_calls",
                    )
                }
                | {"receipt": rel(out / "complete.json")}
            )
        )


if __name__ == "__main__":
    main()

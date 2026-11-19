from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch

from analysis.feedback.rounds import runtime as runtime

c, q, a = runtime.c, runtime.q, runtime.a
BASE = c.ITERATIVE / "execution/recovery_20261024_v1"
PHASE = c.ITERATIVE / "phases/source_initial"
STORE = PHASE / "execution/live/source"
STAGE = "source_01"


def rel(path):
    return (
        Path(str(path).removeprefix("\\\\?\\")).resolve().relative_to(c.ROOT).as_posix()
    )


@contextmanager
def offline():
    import requests

    with c.offline(), patch.object(
        q.template.core, "live_client", c.forbidden
    ), patch.object(a.RequestsTransport, "__call__", c.forbidden), patch.object(
        requests.sessions.Session, "request", c.forbidden
    ):
        yield


def inventory(root):
    return {
        rel(p): a.digest(p)
        for p in a.storage_path(root).rglob("*")
        if p.is_file() and p.suffix != ".lock"
    }


def partial(job, b, cat, root, plan_sha, backend):
    item = b.item_dir(root, job, plan_sha)
    entry = cat[job["key"], job["condition"]]
    stamp = a.read(item / "job.json")
    a.require(
        stamp
        == dict(
            identity=b.identity(job, plan_sha),
            messages_sha256=entry["messages_sha256"],
            contract=a.CONTRACT,
            implementation=backend.implementation(),
            execution_mode="live",
        ),
        "Partial job changed",
    )
    a.require(
        not (item / "complete.json").exists() and not list(item.rglob("*.partial_*")),
        "Unreviewed partial file",
    )
    parser = backend.bundle.parser
    paths = set()
    terminal = []
    successes = 0
    saved_votes = 0
    failures = Counter()
    response_ids = []
    for vote in range(3):
        convo = list(entry["messages"])
        run = parser.RunResult(result=None)
        for turn in range(3):
            call = item / "calls" / f"vote_{vote}_turn_{turn}"
            if not call.exists():
                a.require(terminal, "Missing request before terminal call")
                break
            dirs = sorted(call.iterdir(), key=lambda p: int(p.name.split("_")[-1]))
            a.require(
                [p.name for p in dirs] == [f"attempt_{i}" for i in range(len(dirs))]
                and 1 <= len(dirs) <= 5,
                "Partial retry sequence differs",
            )
            successful = None
            for attempt, folder in enumerate(dirs):
                a.require(
                    {p.name for p in folder.iterdir()}
                    == {"intent.json", "outcome.json"},
                    "Unresolved request",
                )
                ip, op = folder / "intent.json", folder / "outcome.json"
                paths.update((ip, op))
                intent, outcome = a.read(ip), a.read(op)
                payload = root / "payloads" / f"{intent['body_sha256']}.json"
                expected = dict(
                    job_stamp=a.signature(stamp),
                    vote=vote,
                    turn=turn,
                    attempt=attempt,
                    kind="ordinary" if turn == 0 else "repair",
                    body_sha256=a.signature(a.body(convo)),
                )
                a.require(
                    set(intent) == set(expected) | {"started_utc", "backoff_elapsed_s"}
                    and all(intent[k] == v for k, v in expected.items())
                    and outcome["intent_sha256"] == a.signature(intent),
                    "Request identity differs",
                )
                a.require(
                    a.read(payload) == a.body(convo),
                    "Saved request conversation differs",
                )
                paths.add(payload)
                action, value, _ = a.interpret(outcome)
                a.require(
                    a.finite_nonnegative(outcome["elapsed_s"])
                    and a.finite_nonnegative(intent["backoff_elapsed_s"]),
                    "Timing differs",
                )
                a.require(
                    (
                        intent["backoff_elapsed_s"] == 0
                        if attempt == 0
                        else intent["backoff_elapsed_s"]
                        >= a.CONTRACT["backoff_seconds"][attempt - 1] - 0.1
                    ),
                    "Retry delay differs",
                )
                run.latency_s += outcome["elapsed_s"] + intent["backoff_elapsed_s"]
                if action == "ok":
                    a.require(attempt == len(dirs) - 1, "Request after received answer")
                    successful = value
                    successes += 1
                    response_ids.append(
                        json.loads(
                            __import__("base64").b64decode(outcome["body_base64"])
                        )["id"]
                    )
                else:
                    a.require(
                        action == "retry"
                        and outcome["kind"] == "transport_error"
                        and outcome["remote_completion_uncertain"] is True,
                        "Failure is not a reviewed transport outage",
                    )
                    failures[outcome["category"]] += 1
            if successful is None:
                a.require(len(dirs) == 5, "Partial call did not exhaust its budget")
                terminal.append(
                    dict(
                        store=rel(root),
                        item=rel(item),
                        job=job,
                        vote=vote,
                        turn=turn,
                        body_sha256=intent["body_sha256"],
                    )
                )
                break
            parsed, normalisations, error = parser.parse_response(successful)
            run.raw_outputs.append(successful)
            run.normalisations += normalisations
            run.repairs = turn
            if parsed is not None:
                run.result = parsed
                run.gate_normalised = parsed.gate_normalised
                break
            run.error = error
            convo += [
                {"role": "assistant", "text": successful},
                {"role": "user", "text": parser.REPAIR_TEMPLATE.format(error=error)},
            ]
        if terminal:
            break
        run.parse_failure = run.result is None
        rebuilt = asdict(run)
        rebuilt["result"] = run.result.model_dump() if run.result else None
        vp = item / "votes" / f"{vote}.json"
        a.require(a.read(vp) == rebuilt, "Saved partial vote does not reproduce")
        paths.add(vp)
        saved_votes += 1
    a.require(
        len(terminal) == 1,
        "Exactly one stopped call per incomplete prediction required",
    )
    expected_files = paths | {item / "job.json"}
    actual_files = {p for p in item.rglob("*") if p.is_file() and p.name != "item.lock"}
    a.require(
        actual_files == {p for p in expected_files if p.is_relative_to(item)},
        "Unattributed partial file",
    )
    terminal[0]["prior_hashes"] = {rel(p): a.digest(p) for p in sorted(expected_files)}
    return terminal[0], saved_votes, successes, failures, response_ids


def inspect():
    a.require(not (BASE / "diagnosis.json").exists(), "Preserve the existing diagnosis")
    with offline(), a.item_lock(
        a.storage_path(c.SHARED / "qwen_queue.lock")
    ), a.item_lock(a.storage_path(c.ITERATIVE / "launcher.lock")), a.item_lock(
        a.storage_path(runtime.BASE / "runtime_launcher.lock")
    ):
        runtime.validate()
        runtime.original.validate()
        b = q.queue(PHASE, "source", runtime.original.SOURCES)
        b.validate(ready=True)
        schedule = a.read(b.BASE / "schedule.json")
        (stage,) = schedule["stages"]
        plan_sha = a.digest(b.PLAN)
        a.require(
            stage["id"] == STAGE and stage["items"] == 3072, "Different stopped phase"
        )
        root = a.storage_path(STORE)
        target = root / "stages" / STAGE
        (stop_path,) = target.glob("stop_*.json")
        stop = a.read(stop_path)
        summary_path = c.ROOT / stop["attempt"] / "summary.json"
        summary = a.read(summary_path)
        a.require(
            stop["stage"] == STAGE
            and stop["plan_sha256"] == plan_sha
            and summary["stopped"],
            "Different stop",
        )
        a.require(
            len(summary["errors"]) == 128 and summary["saved_items"] == 505,
            "Different outage inventory",
        )
        cat = b.load_catalog("source")
        engines = b.engines(cat)
        before = inventory(PHASE / "execution/live")
        marker_before = inventory(PHASE / "execution/concurrency_20261024_v1")
        complete = []
        errors = {
            a.signature(b.identity(e["job"], plan_sha)): e for e in summary["errors"]
        }
        calls = []
        votes = 0
        successes = 0
        failures = Counter()
        remote_ids = []
        unstarted = 0
        payloads = set()
        item_ids = set()
        for job in stage["jobs"]:
            item = b.item_dir(root, job, plan_sha)
            ident = a.signature(b.identity(job, plan_sha))
            if (item / "complete.json").exists():
                complete.append(job)
                continue
            if not item.exists():
                unstarted += 1
                continue
            item_ids.add(ident)
            error = errors.pop(ident)
            a.require(
                error["job"] == job
                and error["error_type"] == "CallStopped"
                and error["reason"] == "transport_attempt_budget_exhausted",
                "Different error",
            )
            call, nv, ns, fs, ids = partial(
                job, b, cat, root, plan_sha, engines[job["schema_path"]]
            )
            calls.append(call)
            votes += nv
            successes += ns
            failures.update(fs)
            remote_ids.extend(ids)
            payloads.update(
                Path(n).name for n in call["prior_hashes"] if "/payloads/" in n
            )
        a.require(
            not errors
            and len(complete) == 505
            and len(calls) == 128
            and unstarted == 2439,
            "Stopped roster does not reconcile",
        )

        def replay(job):
            engines[job["schema_path"]].audit_item(
                root,
                b.identity(job, plan_sha),
                cat[job["key"], job["condition"]]["messages"],
                execution_mode="live",
            )
            return b.item_dir(root, job, plan_sha)

        with ThreadPoolExecutor(max_workers=4) as pool:
            for item in pool.map(replay, complete):
                item_ids.add(item.name)
                receipt = a.read(item / "complete.json")
                payloads.update(
                    Path(n).name for n in receipt["hashes"] if n.startswith("payloads/")
                )
                for op in item.rglob("outcome.json"):
                    outcome = a.read(op)
                    if a.interpret(outcome)[0] == "ok":
                        remote_ids.append(
                            json.loads(
                                __import__("base64").b64decode(outcome["body_base64"])
                            )["id"]
                        )
        a.require(
            len(remote_ids) == len(set(remote_ids))
            and all(isinstance(i, str) and i for i in remote_ids),
            "Saved response ID reused",
        )
        a.require(
            {p.name for p in (root / "items").iterdir()} == item_ids,
            "Unattributed item",
        )
        a.require(
            {p.name for p in (root / "payloads").iterdir()} == payloads,
            "Unattributed payload",
        )
        a.require(
            before == inventory(PHASE / "execution/live")
            and marker_before == inventory(PHASE / "execution/concurrency_20261024_v1"),
            "Saved evidence changed",
        )
        a.require(
            not (c.ITERATIVE / "complete.json").exists()
            and {p.name for p in (c.ITERATIVE / "phases").iterdir()}
            == {"source_initial"},
            "Unexpected later phase",
        )
        result = dict(
            status="offline_reviewed_connection_outage",
            created_utc=c.utc(),
            phase=rel(PHASE),
            stage=STAGE,
            original_plan_sha256=plan_sha,
            original_runtime_sha256=a.digest(runtime.MANIFEST),
            inventory=dict(
                completed_predictions=505,
                partial_predictions=128,
                unstarted_predictions=2439,
                completed_saved_votes=1515,
                partial_saved_votes=votes,
                partial_successful_responses=successes,
                outage_failures=dict(failures),
                distinct_saved_response_ids=len(remote_ids),
                unresolved_intents=0,
            ),
            calls=calls,
            reviewed_stop_files={rel(stop_path): a.digest(stop_path)},
            reviewed_failed_summaries={rel(summary_path): a.digest(summary_path)},
            preserved_hashes={**before, **marker_before},
            completed_receipt_hashes={
                rel(b.item_dir(root, j, plan_sha) / "complete.json"): a.digest(
                    b.item_dir(root, j, plan_sha) / "complete.json"
                )
                for j in complete
            },
            completed_raw_replay=True,
            shared_and_study_and_runtime_locks_available=True,
            actual_Qwen_calls=0,
            actual_model_list_GETs=0,
            live_files_modified=False,
        )
        c.save(BASE / "diagnosis.json", result)
        print(
            json.dumps(
                dict(
                    status=result["status"],
                    **result["inventory"],
                    preserved_files=len(result["preserved_hashes"]),
                    actual_Qwen_calls=0,
                ),
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    inspect()

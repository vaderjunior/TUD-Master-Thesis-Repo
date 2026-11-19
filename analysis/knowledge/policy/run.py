"""Policy run."""

import argparse
import base64
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
import gc
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
from types import FunctionType
from analysis.knowledge.policy import common as c
from analysis.knowledge.policy import engine as engine
from analysis.common import batch as core
from analysis.common import runtime as host
from analysis.knowledge.german_guidelines import run as previous

a = c.a
_names = (
    "identity",
    "item_dir",
    "scan_job",
    "independent_decision",
    "verify_review",
    "audit_batch",
)
_namespace = dict(vars(previous))
_namespace.update(c=c, a=a)
for _name in _names:
    _old = getattr(previous, _name)
    _namespace[_name] = FunctionType(
        _old.__code__, _namespace, _name, _old.__defaults__, _old.__closure__
    )
globals().update({name: _namespace[name] for name in _names})


def storage_layout():
    c.require(
        c.OUTPUT.is_dir() and c.normal(c.OUTPUT) == c.OUTPUT,
        "D: output folder unavailable or redirected",
    )
    for path in (c.OUTPUT, *c.OUTPUT.parents):
        c.require(
            not path.is_symlink() and (not path.is_junction()),
            "Output path contains a link or junction",
        )
    return {
        "output_root": c.OUTPUT.as_posix(),
        "volume_device_id": c.OUTPUT.stat().st_dev,
        "workspace": c.ROOT.as_posix(),
        "plan_sha256": c.sha(c.PLAN),
        "old_outputs_moved": False,
    }


def init_storage():
    with c.offline():
        c.validate(ready=False)
        c.require(
            c.OUTPUT.drive == "D:" and c.OUTPUT.parent == Path("D:/Thesis_Experiments"),
            "Unexpected D: destination",
        )
        if not c.OUTPUT.exists():
            c.OUTPUT.mkdir(parents=True)
        else:
            c.require(
                (c.OUTPUT / "storage_layout.json").exists() or not any(c.OUTPUT.iterdir()),
                "Unknown existing destination contents",
            )
        c.save(c.OUTPUT / "storage_layout.json", storage_layout())
        print("Isolated D: storage ready. No API calls.", flush=True)


def verify_storage():
    c.require(
        c.read(c.OUTPUT / "storage_layout.json") == storage_layout(),
        "Output volume or plan identity changed",
    )


def locks():
    stack = ExitStack()
    try:
        for path in (host.SHARED / "qwen_queue.lock", c.OUTPUT / "queue.lock"):
            stack.enter_context(a.item_lock(a.storage_path(path)))
    except BaseException:
        stack.close()
        raise
    return stack


def resources(remaining, initial=False):
    verify_storage()
    snapshot = host.resources(2.0 if initial else 0.75, 8.0)
    free = shutil.disk_usage(c.OUTPUT).free
    required = 8 * 2**30 + remaining * 2**20
    c.require(
        free >= required,
        f"D: needs {required / 2 ** 30:.2f} GiB free for remaining work; currently {free / 2 ** 30:.2f}. Keep saved files.",
    )
    return snapshot | {
        "D_free_GiB": free / 2**30,
        "D_required_GiB": required / 2**30,
        "remaining_predictions": remaining,
    }


def launch_window(started, now=None):
    now = now or datetime.now().astimezone()
    c.require(
        now < c.DEADLINE,
        "Experiment deadline reached. Keep all saved work; no new calls are allowed.",
    )
    c.require(
        started or now <= c.LATEST_START,
        "The 18-hour new-run margin no longer fits. Review the schedule before starting; do not bypass this check.",
    )


class RequestGuard:

    def __init__(self, remaining, clock=None, resource_check=None, token_count=None):
        self.clock = clock or (lambda: datetime.now().astimezone())
        self.resource_check = resource_check or (lambda: resources(remaining))
        self.lock = threading.RLock()
        self.last_resources = None
        self.last_check = -float("inf")
        self.context_limit = None
        self.calls_checked = 0
        self.cache = {}
        if token_count is None:
            from tokenizers import Tokenizer

            self.tokenizer = Tokenizer.from_file(str(c.TOKENIZER / "tokenizer.json"))
            self.tokenizer.no_padding()
            self.tokenizer.no_truncation()
            self.count = self.count_tokens
        else:
            self.count = token_count

    def count_tokens(self, payload):
        messages = payload["messages"]
        c.require(
            messages
            and messages[0]["role"] == "system"
            and all((set(m) == {"role", "content"} for m in messages)),
            "Unexpected text-chat message structure",
        )
        c.require(
            all(
                (
                    m["role"] in ("system", "user", "assistant") and isinstance(m["content"], str)
                    for m in messages
                )
            ),
            "Nontext prompt",
        )
        text = (
            "".join(
                (
                    "<|im_start|>" + m["role"] + "\n" + m["content"] + "<|im_end|>\n"
                    for m in messages
                )
            )
            + "<|im_start|>assistant\n"
        )
        return len(self.tokenizer.encode(text, add_special_tokens=False).ids)

    def __call__(self, payload):
        if self.clock() >= c.DEADLINE:
            raise engine.PausedBeforeRequest(
                "Deadline reached before a new request; all saved replies are kept."
            )
        digest = a.signature(payload)
        with self.lock:
            n = self.cache.get(digest)
            if n is None:
                n = self.count(payload)
                if len(self.cache) >= 2048:
                    self.cache.clear()
                self.cache[digest] = n
            if not 0 < n <= 8192:
                raise engine.PausedBeforeRequest(
                    "Initial or repair prompt exceeds the 8192-token input guard; no truncation or request."
                )
            if self.context_limit is not None and n + payload["max_tokens"] > self.context_limit:
                raise engine.PausedBeforeRequest(
                    "Request plus output allowance exceeds the server-reported context limit; no request."
                )
            now = time.monotonic()
            if now - self.last_check >= 30:
                try:
                    self.last_resources = self.resource_check()
                except (RuntimeError, a.IntegrityError):
                    raise engine.PausedBeforeRequest(
                        "Local RAM/disk guard paused before a new request. Keep saved work and check available space/memory."
                    ) from None
                self.last_check = now
            if self.clock() >= c.DEADLINE:
                raise engine.PausedBeforeRequest(
                    "Deadline reached while checking the request; nothing sent."
                )
            self.calls_checked += 1


def catalogue_for(stage, index):
    keys = set(stage["input_keys"])
    cat = {}
    with (c.BASE / "prompts.jsonl").open("rb") as stream:
        for entry in index:
            if entry["key"] not in keys:
                continue
            stream.seek(entry["offset"])
            raw = stream.read(entry["length"])
            c.require(
                hashlib.sha256(raw).hexdigest() == entry["row_sha256"],
                "Indexed prompt bytes changed",
            )
            row = json.loads(raw)
            c.require(
                row["key"] == entry["key"]
                and row["condition"] == entry["condition"]
                and (
                    a.signature(row["messages"])
                    == entry["messages_sha256"]
                    == row["messages_sha256"]
                ),
                "Prompt identity/hash changed",
            )
            c.require(
                set(row)
                <= {
                    "key",
                    "condition",
                    "population",
                    "lang",
                    "messages",
                    "messages_sha256",
                    "reference_input_tokens",
                    "context_sha256",
                    "original_messages_sha256",
                    "original_context_sha256",
                    "source_prompt_line",
                },
                "Unexpected execution fields",
            )
            cat[row["key"], row["condition"]] = row
    c.require(
        set(cat) == {(k, cond) for k in keys for cond in c.CONDITIONS},
        "Incomplete batch prompt matrix",
    )
    return cat


def availability(folder, transport, guard):
    launch_window(True)
    previous.availability(folder, transport)
    response = c.read(folder / "model_check_response.json")
    data = json.loads(base64.b64decode(response["body_base64"]))
    entries = [v for v in data["data"] if v["id"] == a.MODEL]
    limits = {
        name: entry[name]
        for entry in entries
        for name in ("max_model_len", "context_length", "max_context_length")
        if type(entry.get(name)) is int and entry[name] > 0
    }
    guard.context_limit = min(limits.values()) if limits else None
    c.save(
        folder / "context_metadata.json",
        {
            "reported_positive_limits": limits,
            "effective_reported_limit": guard.context_limit,
            "reference_input_guard": 8192,
            "output_allowance": 4096,
            "limit_verified_by_benchmark": False,
            "note": "An absent context limit is unknown; HTTP errors or truncated answers stop without shortening the prompt.",
        },
    )


def execute_batch(
    root, stage, cat, plan_sha, backend, mode, factory, sleep=time.sleep, availability_check=None
):
    root = a.storage_path(root)
    target = root / "batches" / stage["id"]
    c.require(
        not (target / "review_required.json").exists(),
        "This batch has an acknowledged process-quality stop; preserve it for review",
    )
    if (target / "review.json").exists():
        verify_review(root, stage, plan_sha, mode)
        print(stage["id"] + ": already audited; no new requests.", flush=True)
        return
    existing = []
    pending = []
    for job in stage["jobs"]:
        value = scan_job(root, job, cat[job["key"], job["condition"]], plan_sha, backend, mode)
        (pending if value is None else existing).append(job if value is None else value)
    print(
        f"{stage['id']}: {len(existing)} saved, {len(pending)} pending; {3 * len(pending)} ordinary calls before saved-vote reuse.",
        flush=True,
    )
    if not pending:
        return
    nulls = sum((v["result"] is None for v in existing))
    c.require(
        len(existing) < 20 or nulls / len(existing) < 0.1,
        "Ten percent unavailable outputs require review",
    )
    attempt = target / f"attempt_{time.time_ns()}"
    c.save(
        attempt / "start.json",
        {
            "started_utc": a.utc(),
            "plan_sha256": plan_sha,
            "stage_sha256": a.signature(stage),
            "mode": mode,
            "workers": c.WORKERS,
            "existing_predictions": len(existing),
            "pending_predictions": len(pending),
        },
    )
    try:
        transport = factory()
        if mode == "live":
            c.require(availability_check is not None, "Live model check missing")
            availability_check(attempt, transport)

        def work(job):
            return backend.execute_item(
                root,
                identity(job, plan_sha),
                cat[job["key"], job["condition"]]["messages"],
                transport,
                sleep=sleep,
                execution_mode=mode,
            )

        def progress(job, result, stats):
            if stats["complete"] % 100 == 0 or stats["complete"] == stage["items"]:
                print(
                    f"{stage['id']}: saved {stats['complete']}/{stage['items']}; unavailable {stats['null']}.",
                    flush=True,
                )

        outcome = core.bounded_jobs(
            pending,
            work,
            workers=c.WORKERS,
            initial={"complete": len(existing), "null": nulls},
            on_result=progress,
        )
        summary = {k: v for k, v in outcome.items() if k != "results"}
        c.save(attempt / "summary.json", summary | {"finished_utc": a.utc()})
        if outcome["stopped"] and (not outcome["errors"]):
            c.save(
                target / "review_required.json",
                {
                    "reason": "Unavailable-output stop threshold reached; review required.",
                    "summary_sha256": c.sha(attempt / "summary.json"),
                    "attempt": c.ref(attempt),
                    "plan_sha256": plan_sha,
                },
            )
        c.require(
            not outcome["stopped"]
            and (not outcome["errors"])
            and (outcome["stats"]["complete"] == stage["items"]),
            "Queue stopped safely. Keep all saved files and send the final message for review. Retry budgets remain intact.",
        )
    except BaseException as exc:
        c.save(
            attempt / "stop.json",
            {
                "finished_utc": a.utc(),
                "error_type": type(exc).__name__,
                "reason": (
                    str(exc)[:400]
                    if type(exc)
                    in (
                        a.CallStopped,
                        a.IncompleteAttempt,
                        a.IntegrityError,
                        engine.PausedBeforeRequest,
                    )
                    else "Inspect preserved evidence"
                ),
                "policy": "Ctrl+C once drains submitted jobs. Completed replies are reused. Unknown or terminal requests require review.",
            },
        )
        raise


def finalize(root, schedule, plan_sha, mode, verify_all=True):
    root = a.storage_path(root)
    counts = Counter()
    reviews = []
    checks = []
    expected_items = {
        a.signature(identity(job, plan_sha))
        for stage in schedule["stages"]
        for job in stage["jobs"]
    }
    c.require(
        {p.name for p in (root / "items").iterdir()} == expected_items,
        "Unexpected or missing item directory",
    )
    c.require(
        not list((root / "payloads").glob("*.partial_*")),
        "Unfinished atomic payload publication requires review",
    )
    for stage in schedule["stages"]:
        review = verify_review(root, stage, plan_sha, mode)
        counts.update(review["counts"])
        reviews.append(root / "batches" / stage["id"] / "review.json")
        item_hashes = {}
        if verify_all:
            for index, job in enumerate(stage["jobs"], 1):
                folder = item_dir(root, job, plan_sha)
                receipt = c.read(folder / "complete.json")
                a.verify_inventory(root, folder, receipt["hashes"])
                item_hashes[a.signature(identity(job, plan_sha))] = c.sha(folder / "complete.json")
            path = root / "final_checks" / (stage["id"] + ".json")
            c.save(
                path,
                {
                    "plan_sha256": plan_sha,
                    "stage_sha256": a.signature(stage),
                    "item_receipts": item_hashes,
                    "status": "all_item_files_rehashed",
                },
            )
            checks.append(path)
            print(
                f"Final saved-file check: {stage['id']}, {stage['items']} predictions verified.",
                flush=True,
            )
    expected = sum((stage["items"] for stage in schedule["stages"]))
    c.require(
        counts["predictions"] == expected and counts["ordinary_logical_calls"] == 3 * expected,
        "Final count differs",
    )
    record = {
        "status": "all_three_conditions_executed_and_audited",
        "mode": mode,
        "plan_sha256": plan_sha,
        "counts": dict(counts),
        "review_hashes": c.bindings(reviews),
        "final_integrity_hashes": c.bindings(checks),
    }
    c.save(root / "complete.json", record)
    return record


def inspect_progress(schedule, plan_sha):
    done = 0
    for stage in schedule["stages"]:
        path = c.OUTPUT / "live/batches" / stage["id"] / "review.json"
        if path.exists():
            review = c.read(path)
            c.require(
                review["plan_sha256"] == plan_sha
                and review["stage_sha256"] == a.signature(stage)
                and (review["mode"] == "live")
                and (review["status"] == "raw_responses_and_votes_verified"),
                "Progress receipt changed",
            )
            done += stage["items"]
    return done


def check(ready=True):
    with c.offline():
        plan = c.validate(ready=ready)
        verify_storage()
        schedule = c.read(c.BASE / "schedule.json")
        plan_sha = c.sha(c.PLAN)
        with locks():
            c.require(
                not list((c.OUTPUT / "live/payloads").glob("*.partial_*")),
                "Incomplete payload publication requires review",
            )
            done = inspect_progress(schedule, plan_sha)
            remaining = c.COUNTS["predictions"] - done
            if remaining:
                launch_window((c.OUTPUT / "live/started.json").exists())
            status = resources(remaining, initial=True)
        record = {
            "status": "offline_preflight_pass",
            "checked_utc": a.utc(),
            "plan_sha256": plan_sha,
            "audited_predictions": done,
            "remaining_predictions": remaining,
            "resources": status,
            "workers": c.WORKERS,
            "actual_Qwen_calls": 0,
            "actual_model_list_GETs": 0,
            "credentials_read": False,
        }
        path = c.BASE / "offline_checks" / f"preflight_{time.time_ns()}.json"
        c.save(path, record)
        print(json.dumps(record | {"report": c.ref(path)}), flush=True)
        return path


def run():
    with locks():
        with c.offline():
            plan = c.validate()
            verify_storage()
            plan_sha = c.sha(c.PLAN)
            schedule = c.read(c.BASE / "schedule.json")
            index = c.read(c.BASE / "prompt_index.json")
            c.require(
                not list((c.OUTPUT / "live/payloads").glob("*.partial_*")),
                "Incomplete payload publication requires review",
            )
            done = inspect_progress(schedule, plan_sha)
            if done < c.COUNTS["predictions"]:
                launch_window((c.OUTPUT / "live/started.json").exists())
            resources(c.COUNTS["predictions"] - done, initial=True)
        print(
            "Complete policy: one queue, 128 workers, three conditions and three outer repeats.",
            flush=True,
        )
        print(
            "Press Ctrl+C once and wait for active requests to finish before closing PowerShell.",
            flush=True,
        )
        print("All raw results are saved under " + str(c.OUTPUT), flush=True)
        if not (c.OUTPUT / "live/started.json").exists() and done < c.COUNTS["predictions"]:
            c.save(
                c.OUTPUT / "live/started.json",
                {
                    "started_utc": a.utc(),
                    "plan_sha256": plan_sha,
                    "deadline": c.DEADLINE.isoformat(),
                    "latest_new_start": c.LATEST_START.isoformat(),
                },
            )
        inv = c.OUTPUT / "invocations" / str(time.time_ns())
        c.save(
            inv / "start.json",
            {"started_utc": a.utc(), "plan_sha256": plan_sha, "already_audited": done},
        )
        began = time.perf_counter()
        new_done = 0
        try:
            for stage in schedule["stages"]:
                with c.offline():
                    c.verify(plan["source_hashes"])
                    if (c.OUTPUT / "live/batches" / stage["id"] / "review.json").exists():
                        verify_review(c.OUTPUT / "live", stage, plan_sha, "live")
                        continue
                    resources(c.COUNTS["predictions"] - done)
                    cat = catalogue_for(stage, index)
                    guard = RequestGuard(c.COUNTS["predictions"] - done)
                    backend = engine.adapter(guard)
                execute_batch(
                    c.OUTPUT / "live",
                    stage,
                    cat,
                    plan_sha,
                    backend,
                    "live",
                    core.live_client,
                    availability_check=lambda p, t: availability(p, t, guard),
                )
                with c.offline():
                    audit_batch(c.OUTPUT / "live", stage, cat, plan_sha, backend, "live")
                done += stage["items"]
                new_done += stage["items"]
                print(
                    f"Overall audited: {done}/{c.COUNTS['predictions']} ({done / c.COUNTS['predictions']:.1%}).",
                    flush=True,
                )
                elapsed = time.perf_counter() - began
                estimate = elapsed / new_done * (c.COUNTS["predictions"] - done) / 3600
                c.save(
                    inv / (stage["id"] + "_progress.json"),
                    {
                        "audited_predictions": done,
                        "elapsed_this_invocation_s": elapsed,
                        "rough_remaining_hours_at_observed_pace": estimate,
                        "note": "Throughput only; excludes future downtime and final file check. No accuracy inspected.",
                    },
                )
                print(
                    f"Rough remaining run time at this pace: {estimate:.1f} hours, plus final saved-file checks.",
                    flush=True,
                )
                del backend, guard, cat
                gc.collect()
            with c.offline():
                result = finalize(c.OUTPUT / "live", schedule, plan_sha, "live")
                c.require(
                    result["counts"]["predictions"] == c.COUNTS["predictions"],
                    "Final matrix differs",
                )
            c.save(
                inv / "finished.json",
                {
                    "finished_utc": a.utc(),
                    "status": "complete",
                    "completion_sha256": c.sha(c.OUTPUT / "live/complete.json"),
                },
            )
        except BaseException as exc:
            c.save(
                inv / "finished.json",
                {
                    "finished_utc": a.utc(),
                    "status": "stopped",
                    "error_type": type(exc).__name__,
                    "message": "Saved results are preserved. Send the terminal message for review.",
                },
            )
            raise
    print(
        "COMPLETE-POLICY EXPERIMENT AND ALL AUDITS COMPLETE. Send this message for analysis. No other experiment started.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("init-storage", "check", "pre-ready-check", "run"))
    args = parser.parse_args()
    if args.mode == "pre-ready-check":
        check(ready=False)
    else:
        {"init-storage": init_storage, "check": check, "run": run}[args.mode]()

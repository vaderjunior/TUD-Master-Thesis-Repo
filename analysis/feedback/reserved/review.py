from src.hsrag.paths import PROJECT_ROOT
import argparse
import base64
from collections import Counter
from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
from unittest.mock import patch

ROOT = PROJECT_ROOT
STUDY = ROOT / "experiments/astra/sq3_reserved_slot_20261012_v1"
EXECUTION = STUDY / "execution"
PREP = EXECUTION / "recovery_20261016_v2"
from analysis.feedback.reserved import transport as transport

OUTPUT = ROOT / "experiments/astra/sq3_reserved_review_results_20261016_v2"
PLAN_SHA = "83483fc8811e36418a2d8a82a45b5d39c856168511d4040c4e26a75bf67446e8"
ADMIN = "LocalProcessInterruptedOutcomeUnknown"
FIELDS = ("hate", "hate_type", "target_group", "severity", "legal")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def logical(path):
    return Path(str(path).removeprefix("\\\\?\\")).resolve()


def physical(path):
    value = str(logical(path))
    return Path("\\\\?\\" + value) if os.name == "nt" else Path(value)


def rel(path):
    return logical(path).relative_to(ROOT).as_posix()


def inside(root, name):
    p = logical(root / name)
    require(p.is_relative_to(logical(root)), "Path escapes its evidence root")
    return physical(p)


def read(path):
    return json.loads(physical(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with physical(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with physical(path).open(encoding="utf8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def verify(root, bindings):
    for name, expected in bindings.items():
        require(sha(inside(root, name)) == expected, "Changed evidence: " + name)


def write(path, value):
    path = logical(path)
    require(
        any((path.is_relative_to(p) for p in (PREP, OUTPUT))),
        "Write outside new review folders",
    )
    data = (
        value
        if isinstance(value, str)
        else json.dumps(
            value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with physical(path).open("x", encoding="utf8", newline="\n") as f:
        f.write(data)


def utc():
    return datetime.now(timezone.utc).isoformat()


def forbidden(*args, **kwargs):
    raise AssertionError(
        "Offline read-only A15 review: network, execution, and original writes forbidden"
    )


@contextmanager
def offline():
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket.socket, "connect_ex", forbidden
    ), patch.object(socket, "create_connection", forbidden):
        yield


def completion_gate(
    execution=EXECUTION,
    expected_plan_sha=PLAN_SHA,
    *,
    expected_predictions=149850,
    expected_stages=16,
    mode="live",
):

    final = physical(execution / mode / "complete.json")
    require(
        final.is_file(),
        "WAIT: original queue has no final completion/audit receipt; no predictions opened",
    )
    require(sha(execution / "plan.json") == expected_plan_sha, "Wrong execution plan")
    done = read(final)
    require(
        done["status"] == "all_studies_executed_and_audited"
        and done["mode"] == mode
        and (done["plan_sha256"] == expected_plan_sha),
        "Queue is not completely audited",
    )
    require(
        done["counts"]["predictions"] == expected_predictions
        and len(done["stages"]) == expected_stages,
        "Incomplete final matrix or stage roster",
    )
    require(
        len({x["stage"] for x in done["stages"]}) == expected_stages
        and len(done["review_hashes"]) == expected_stages,
        "Duplicate/missing final stage review",
    )
    require(
        done["actual_finalization_API_calls"] == 0 and done["gold_scoring"] is False,
        "Wrong finalization contract",
    )
    verify(ROOT, done["review_hashes"])
    return done


def independent_vote(predictions):
    valid = [p for p in predictions if p is not None]
    if not valid:
        return (None, False)
    n = len(valid)
    yes = sum((p["hate"] for p in valid))
    final = {"hate": yes > n / 2}
    for field in ("hate_type", "target_group", "legal"):
        labels = sorted({label for p in valid for label in p[field]})
        final[field] = [
            label for label in labels if sum((label in p[field] for p in valid)) > n / 2
        ]
    severity = sorted(
        (
            ("low", "medium", "high").index(p["severity"])
            for p in valid
            if p["severity"] is not None
        )
    )
    final["severity"] = (
        ("low", "medium", "high")[severity[(len(severity) - 1) // 2]]
        if severity
        else None
    )
    if not final["hate"]:
        final.update(hate_type=[], target_group=[], severity=None)
    return (final, yes * 2 == n)


def identity(job, plan_sha):
    return {
        "experiment_sha256": plan_sha,
        "condition": job["study"] + "::" + job["condition"],
        "repeat": job["repeat"],
        "key": job["key"],
    }


class PromptIndex:
    """Seek into the 382 MB frozen JSONL; never load the full prompt matrix."""

    def __init__(self, path, roster, science, science_sha):
        from analysis.common import api as a

        self.path = path
        self.offsets = {}
        self.meta = {}
        bykey = {r["key"]: r for r in roster}
        with physical(path).open("rb") as f:
            while True:
                offset = f.tell()
                line = f.readline()
                if not line:
                    break
                row = json.loads(line)
                key = (row["key"], row["condition"])
                require(
                    key not in self.offsets
                    and key[0] in bykey
                    and (key[1] in science["conditions"]),
                    "Bad prompt identity",
                )
                require(
                    row["scientific_plan_sha256"] == science_sha
                    and row["text"] == bykey[key[0]]["text"],
                    "Prompt source differs",
                )
                require(
                    a.signature(row["messages"]) == row["messages_sha256"]
                    and a.signature(row["context"]) == row["context_sha256"],
                    "Prompt/context digest differs",
                )
                require(
                    [m["role"] for m in row["messages"]] == ["system", "user"]
                    and 0 < row["reference_input_tokens"] <= 8192,
                    "Unreviewed prompt budget or roles",
                )
                a.body(row["messages"])
                self.offsets[key] = offset
                self.meta[key] = {
                    k: row[k]
                    for k in (
                        "key",
                        "condition",
                        "population",
                        "messages_sha256",
                        "reference_input_tokens",
                        "source_id",
                        "selected_correction_cosine",
                    )
                }
        require(
            set(self.offsets)
            == {(r["key"], c) for r in roster for c in science["conditions"]},
            "Incomplete prompt catalog",
        )

    def get(self, key):
        with physical(self.path).open("rb") as f:
            f.seek(self.offsets[key])
            row = json.loads(f.readline())
        require(
            (row["key"], row["condition"]) == key
            and row["messages_sha256"] == self.meta[key]["messages_sha256"],
            "Indexed prompt moved/changed",
        )
        return row


def audit_item(
    root, job, plan_sha, prompt, backend, *, mode, start, end, administrative_paths
):

    from analysis.common import api as a

    ident = identity(job, plan_sha)
    folder = physical(root / "items" / a.signature(ident))
    require(
        prompt["messages_sha256"] == job["messages_sha256"], "Schedule/prompt mismatch"
    )
    require(backend.taxonomy_sha256 == job["schema_sha256"], "Schedule/schema mismatch")
    with patch.object(a, "save_new", forbidden), patch.object(
        a, "item_lock", forbidden
    ), patch.object(a.RequestsTransport, "__call__", forbidden):
        backend.audit_item(root, ident, prompt["messages"], execution_mode=mode)
    saved = read(folder / "result.json")
    raw_votes = []
    for raw_run in saved["raw_runs"]:
        last = None
        for number, raw in enumerate(raw_run):
            parsed, _, _ = backend.bundle.parser.parse_response(raw)
            if parsed is not None:
                require(
                    number == len(raw_run) - 1, "Replies continue after a valid vote"
                )
                last = parsed.model_dump()
        raw_votes.append(last)
    require(
        raw_votes == saved["run_predictions"] and len(raw_votes) == 3,
        "Reparsed vote roster differs",
    )
    final, tie = independent_vote(raw_votes)
    require(
        final == ({k: saved["result"][k] for k in FIELDS} if saved["result"] else None)
        and tie == saved["uncertain"]
        and (sum((p is not None for p in raw_votes)) == saved["n_valid"]),
        "Independent final vote differs",
    )
    usage = Counter()
    statuses = Counter()
    ids = []
    request_times = []
    backoffs = []
    unknown = []
    previous = None
    ordinary_usage = Counter()
    repair_usage = Counter()
    for event in saved["events"]:
        call = (
            folder
            / "calls"
            / f"vote_{event['vote']}_turn_{event['turn']}"
            / f"attempt_{event['attempt']}"
        )
        intent, outcome = (read(call / "intent.json"), read(call / "outcome.json"))
        began, finished = (
            datetime.fromisoformat(intent["started_utc"]),
            datetime.fromisoformat(outcome["finished_utc"]),
        )
        require(
            began.tzinfo is not None
            and finished.tzinfo is not None
            and (start <= began <= finished <= end)
            and (previous is None or previous <= began),
            "Request chronology differs",
        )
        previous = finished
        require(
            type(intent["attempt"]) is int
            and intent["attempt"]
            in transport.reviewed_attempts(
                root, folder, intent["vote"], intent["turn"]
            ),
            "Retry cap differs",
        )
        require(
            a.finite_nonnegative(outcome["elapsed_s"])
            and a.finite_nonnegative(intent["backoff_elapsed_s"]),
            "Invalid request duration",
        )
        if intent["attempt"] == 0:
            require(intent["backoff_elapsed_s"] == 0, "Unexpected initial backoff")
        elif mode == "live":
            require(
                intent["backoff_elapsed_s"]
                >= transport.reviewed_backoff(intent["attempt"]) - 0.1,
                "Insufficient retry backoff",
            )
        name = rel(call / "outcome.json")
        administrative = outcome.get("category") == ADMIN
        require(
            administrative == (name in administrative_paths),
            "Unregistered or missing administrative unknown outcome",
        )
        if administrative:
            require(
                outcome["kind"] == "transport_error"
                and outcome["elapsed_s"] == 0
                and (
                    outcome["administrative_recovery"][
                        "elapsed_s_is_unknown_zero_sentinel"
                    ]
                    is True
                )
                and (
                    outcome["administrative_recovery"][
                        "finished_utc_is_review_time_not_request_end"
                    ]
                    is True
                ),
                "Unknown timing no longer explicit",
            )
            unknown.append(name)
        else:
            request_times.append(outcome["elapsed_s"])
        backoffs.append(intent["backoff_elapsed_s"])
        if outcome["kind"] == "transport_error":
            statuses[
                (
                    "administrative_unknown"
                    if administrative
                    else "transport_error:" + outcome["category"]
                )
            ] += 1
            continue
        statuses["http_" + str(outcome["status"])] += 1
        if event["action"] != "ok":
            continue
        response = json.loads(base64.b64decode(outcome["body_base64"], validate=True))
        u = response["usage"]
        require(
            isinstance(response.get("id"), str) and response["id"],
            "Missing response identity",
        )
        require(
            all(
                (
                    type(u.get(k)) is int and u[k] >= 0
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                )
            )
            and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"],
            "Invalid saved usage",
        )
        ids.append(response["id"])
        observed = {
            k: u[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        usage.update(observed)
        (ordinary_usage if event["turn"] == 0 else repair_usage).update(observed)
    require(
        len(ids) == saved["accounting"]["http_success_with_text"]
        and len(ids) == len(set(ids)),
        "Duplicate/missing response ID",
    )
    compact = {
        "key": job["key"],
        "condition": job["condition"],
        "repeat": job["repeat"],
        "result": final,
        "n_valid": saved["n_valid"],
        "uncertain": tie,
        "messages_sha256": prompt["messages_sha256"],
        "item_receipt": rel(folder / "complete.json"),
        "item_receipt_sha256": sha(folder / "complete.json"),
        "accounting": saved["accounting"],
        "observed_usage": dict(usage),
        "ordinary_usage": dict(ordinary_usage),
        "repair_usage": dict(repair_usage),
        "known_request_seconds_sum": math.fsum(request_times),
        "known_request_duration_count": len(request_times),
        "retry_backoff_seconds_sum": math.fsum(backoffs),
        "known_request_durations_s": request_times,
        "administrative_unknown_requests": len(unknown),
        **{
            k: saved[k]
            for k in ("repairs", "parse_failures", "normalisations", "gate_normalised")
        },
    }
    return (compact, ids, statuses, unknown)


def audit_stage(
    stage,
    root,
    plan_sha,
    prompt_index,
    backend,
    *,
    mode,
    administrative_paths,
    stage_review_sha,
):
    from analysis.common import api as a

    target = physical(root / "stages" / stage["id"])
    transport.verify_stops(root, target)
    review = read(target / "review.json")
    require(
        sha(target / "review.json") == stage_review_sha,
        "Final queue/review binding differs",
    )
    require(
        review["status"] == "audited"
        and all(
            (
                review[k] == v
                for k, v in {
                    "stage": stage["id"],
                    "study": stage["study"],
                    "mode": mode,
                    "plan_sha256": plan_sha,
                }.items()
            )
        ),
        "Stage audit identity differs",
    )
    verify(target, review["artifacts"])
    complete = read(target / "complete.json")
    report = read(target / review["report"])
    require(
        sha(target / "complete.json")
        == review["complete_sha256"]
        == report["stage_complete_sha256"],
        "Stage complete binding differs",
    )
    require(
        all(
            (
                complete[k] == v
                for k, v in {
                    "stage": stage["id"],
                    "study": stage["study"],
                    "mode": mode,
                    "plan_sha256": plan_sha,
                }.items()
            )
        ),
        "Completion identity differs",
    )
    summary = complete["summary"]
    require(
        not summary["stopped"]
        and (not summary["errors"])
        and (summary["saved_items"] == stage["items"]),
        "Stage incomplete",
    )
    final_summary = inside(ROOT, complete["final_attempt"]) / "summary.json"
    require(
        sha(final_summary) == complete["summary_sha256"]
        and read(final_summary) == summary,
        "Attempt summary differs",
    )
    expected_items = {
        f"items/{a.signature(identity(j, plan_sha))}/complete.json"
        for j in stage["jobs"]
    }
    require(
        set(complete["item_receipt_hashes"]) == expected_items
        and len(expected_items) == stage["items"],
        "Wrong stage item set",
    )
    verify(root, complete["item_receipt_hashes"])
    verify(root, report["attempt_hashes"])
    verify(ROOT, report["source_hashes"])
    raw_hashes = read(target / review["raw_hashes"])
    verify(root, raw_hashes)
    end = datetime.fromisoformat(complete["completed_utc"])
    starts = []
    model_checks = 0
    history = {rel(p): sha(p) for p in target.glob("stop_*.json")}
    for attempt in sorted(target.glob("attempt_*")):
        names = {p.name for p in attempt.iterdir()}
        require(
            names
            <= {
                "start.json",
                "inventory.json",
                "summary.json",
                "interrupted.json",
                "model_check_intent.json",
                "model_check_response.json",
            }
            and "start.json" in names,
            "Unreviewed stage attempt artifact",
        )
        start = read(attempt / "start.json")
        began = datetime.fromisoformat(start["started_utc"])
        require(
            all(
                (
                    start[k] == v
                    for k, v in {
                        "stage": stage["id"],
                        "study": stage["study"],
                        "mode": mode,
                        "plan_sha256": plan_sha,
                    }.items()
                )
            )
            and began <= end,
            "Attempt identity/chronology differs",
        )
        starts.append(began)
        if "inventory.json" in names:
            inv = read(attempt / "inventory.json")
            pending = inv["pending_identity_hashes"]
            require(
                inv["completed"] + inv["pending"] == stage["items"]
                and len(set(pending)) == len(pending) == inv["pending"]
                and (
                    set(pending)
                    <= {a.signature(identity(j, plan_sha)) for j in stage["jobs"]}
                ),
                "Bad attempt inventory",
            )
        if "summary.json" in names:
            old = read(attempt / "summary.json")
            transport.verify_attempt_summary(attempt, old)
        if "model_check_intent.json" in names:
            intent = read(attempt / "model_check_intent.json")
            response = read(attempt / "model_check_response.json")
            raw = base64.b64decode(response["body_base64"], validate=True)
            require(
                intent["action"] == "GET PEASEC /models"
                and intent["model"] == a.MODEL
                and (began <= datetime.fromisoformat(intent["started_utc"]) <= end)
                and (response["status"] == 200)
                and (hashlib.sha256(raw).hexdigest() == response["body_sha256"])
                and (a.MODEL in {v["id"] for v in json.loads(raw)["data"]}),
                "Saved availability check differs",
            )
            model_checks += 1
        else:
            require(
                "model_check_response.json" not in names,
                "Availability outcome without intent",
            )
        history.update({rel(p): sha(p) for p in attempt.iterdir()})
    require(
        starts and (mode == "mock" or model_checks > 0),
        "Missing attempt/model check history",
    )
    require(
        history
        == {rel(inside(root, k)): v for k, v in report["attempt_hashes"].items()},
        "Omitted/unattributed attempt history",
    )
    counts = Counter()
    accounting = Counter()
    usage = Counter()
    statuses = Counter()
    all_ids = []
    records = []
    unknown = []
    rebuilt_hashes = {}
    for number, job in enumerate(stage["jobs"], 1):
        prompt = prompt_index.get((job["key"], job["condition"]))
        value, ids, status, admin = audit_item(
            root,
            job,
            plan_sha,
            prompt,
            backend,
            mode=mode,
            start=min(starts),
            end=end,
            administrative_paths=administrative_paths,
        )
        value["stage"] = stage["id"]
        records.append(value)
        all_ids.extend(ids)
        statuses.update(status)
        unknown.extend(admin)
        counts["predictions"] += 1
        counts["nulls"] += value["result"] is None
        counts["gate_ties"] += value["uncertain"]
        counts["valid_votes_" + str(value["n_valid"])] += 1
        for k in ("repairs", "parse_failures", "normalisations", "gate_normalised"):
            counts[k] += value[k]
        accounting.update(value["accounting"])
        usage.update(value["observed_usage"])
        receipt_path = inside(ROOT, value["item_receipt"])
        receipt = read(receipt_path)
        for name, digest in receipt["hashes"].items():
            require(
                name not in rebuilt_hashes or rebuilt_hashes[name] == digest,
                "Conflicting payload binding",
            )
            rebuilt_hashes[name] = digest
        rebuilt_hashes[logical(receipt_path).relative_to(logical(root)).as_posix()] = (
            value["item_receipt_sha256"]
        )
        if number % 250 == 0:
            print(
                f"Read-only review {stage['id']}: {number}/{stage['items']}", flush=True
            )
    require(rebuilt_hashes == raw_hashes, "Raw audit manifest omits/adds an artifact")
    require(
        counts == Counter(report["counts"])
        and accounting == Counter(report["accounting"])
        and (usage == Counter(report["observed_usage"])),
        "Reconstructed counts/usage differ",
    )
    require(
        len(all_ids) == len(set(all_ids)) == report["distinct_response_ids"]
        and all_ids == read(target / review["response_ids"]),
        "Stage response IDs differ",
    )
    require(
        len(unknown) == report.get("administrative_unknown_requests", 0),
        "Administrative count differs",
    )
    expected_statuses = {
        (
            "administrative_unknown"
            if k == "transport_error:" + ADMIN
            else "http_" + k if k.isdigit() else k
        ): v
        for k, v in report["http_statuses"].items()
    }
    require(
        statuses == Counter(expected_statuses)
        and sum(statuses.values()) == accounting["post_attempts"],
        "Independent HTTP/transport accounting differs",
    )
    if stage["id"] == transport.context()["stage"]:
        require(
            report["reviewed_transport_amendment"] == transport.provenance(),
            "Transport audit provenance differs",
        )
    return (
        records,
        all_ids,
        statuses,
        unknown,
        {k.split("/", 1)[1] for k in raw_hashes if k.startswith("payloads/")},
    )


def environment():
    import importlib.metadata

    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("numpy", "pydantic", "PyYAML")
        },
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def reviewer_sources():
    names = [
        "analysis/feedback/reserved/review.py",
        "analysis/feedback/reserved/rates.py",
        "analysis/feedback/reserved/support.py",
        "analysis/feedback/reserved/bounds.py",
        "analysis/feedback/reserved/scoring.py",
        "analysis/feedback/reserved/transport.py",
        "analysis/feedback/reserved/resume.py",
        "analysis/feedback/reserved/review_earlier.py",
        "analysis/feedback/reserved/scoring.py",
    ]
    return {
        rel(Path(__file__).with_name(n)): sha(Path(__file__).with_name(n))
        for n in names
    }


def review():
    done = completion_gate()
    require(
        not OUTPUT.exists(),
        "Preserve previous review output; inspect it before a new version",
    )
    readiness = read(PREP / "ready.json")
    verify(ROOT, readiness["analysis_source_hashes"])
    require(
        readiness["status"] == "offline_verified_ready_for_user",
        "Analysis preparation not sealed",
    )
    from analysis.feedback.reserved import resume as recovery
    from analysis.common import schema as schema
    from analysis.common import api as a

    scope = recovery.scope_from_disk()
    scope.verify_records()
    require(
        done["reviewed_transport_amendment"] == transport.provenance(),
        "Final transport amendment differs",
    )
    plan = read(EXECUTION / "plan.json")
    verify(ROOT, plan["source_hashes"])
    verify(ROOT, plan["input_hashes"])
    schedule = read(EXECUTION / "schedule.json")
    science = read(STUDY / "scientific_plan_v1.json")
    roster = list(rows(STUDY / "roster.jsonl"))
    require(
        len(roster) == 675
        and len({r["key"] for r in roster}) == 675
        and (len(science["conditions"]) == 74),
        "Frozen study dimensions differ",
    )
    require(
        done["stages"]
        == [
            {
                "study": x["study"],
                "stage": x["id"],
                "review": rel(
                    EXECUTION / "live" / x["study"] / "stages" / x["id"] / "review.json"
                ),
            }
            for x in schedule["stages"]
        ],
        "Final stage order differs",
    )
    root = physical(EXECUTION / "live/reserved_slot")
    admin = set(scope.applied["administrative_outcomes"])
    require(
        len(admin) == 8 and done["administrative_unknown_requests"] == 8,
        "Recovery scope differs",
    )
    index = PromptIndex(
        STUDY / "prompts.jsonl", roster, science, sha(STUDY / "scientific_plan_v1.json")
    )
    backend = transport.adapter(ROOT / "config/taxonomy.yaml")
    write(
        OUTPUT / "started.json",
        {
            "started_utc": utc(),
            "queue_complete_sha256": sha(EXECUTION / "live/complete.json"),
            "analysis_source_hashes": reviewer_sources(),
            "original_plan_sha256": PLAN_SHA,
            "environment": environment(),
            "new_model_calls": 0,
        },
    )
    all_ids = set()
    all_jobs = set()
    payloads = set()
    unknown = []
    counts = Counter()
    usage = Counter()
    accounting = Counter()
    statuses = Counter()
    assembled = []
    batch_by_key = {}
    stage_bindings = {}
    for stage in schedule["stages"]:
        scope.verify_stops(root / "stages" / stage["id"])
        require(
            set(stage["input_keys"]) == {j["key"] for j in stage["jobs"]},
            "Batch input list differs",
        )
        for key in stage["input_keys"]:
            require(key not in batch_by_key, "Text split across batches")
            batch_by_key[key] = stage["id"]
        review_name = rel(root / "stages" / stage["id"] / "review.json")
        records, ids, status, interrupted, blobs = audit_stage(
            stage,
            root,
            PLAN_SHA,
            index,
            backend,
            mode="live",
            administrative_paths=admin,
            stage_review_sha=done["review_hashes"][review_name],
        )
        require(not all_ids.intersection(ids), "Remote response reused across batches")
        all_ids.update(ids)
        payloads.update(blobs)
        statuses.update(status)
        unknown.extend(interrupted)
        for row in records:
            ident = (row["key"], row["condition"], row["repeat"])
            require(ident not in all_jobs, "Duplicate assembled prediction")
            all_jobs.add(ident)
            counts["predictions"] += 1
            counts["nulls"] += row["result"] is None
            counts["valid_votes_" + str(row["n_valid"])] += 1
            counts["gate_ties"] += row["uncertain"]
            for k in ("repairs", "parse_failures", "normalisations", "gate_normalised"):
                counts[k] += row[k]
            usage.update(row["observed_usage"])
            accounting.update(row["accounting"])
        out = OUTPUT / "assembly" / (stage["id"] + ".jsonl")
        write(
            out,
            "".join(
                (
                    json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    + "\n"
                    for r in records
                )
            ),
        )
        stage_bindings[rel(out)] = sha(out)
    expected = {
        (r["key"], c, rep)
        for r in roster
        for c in science["conditions"]
        for rep in range(3)
    }
    require(
        all_jobs == expected and len(all_jobs) == 149850, "Assembly matrix incomplete"
    )
    expected_items = {
        a.signature(
            {
                "experiment_sha256": PLAN_SHA,
                "condition": "reserved_slot::" + c,
                "repeat": rep,
                "key": k,
            }
        )
        for k, c, rep in expected
    }
    require(
        {p.name for p in (root / "items").iterdir() if p.is_dir()} == expected_items,
        "Unattributed/missing item directory",
    )
    require(
        {p.name for p in (root / "payloads").iterdir()} == payloads
        and {p.name for p in (root / "stages").iterdir()}
        == {s["id"] for s in schedule["stages"]},
        "Unattributed payload/stage",
    )
    require(
        counts == Counter(done["counts"])
        and usage == Counter(done["observed_usage"])
        and (accounting == Counter(done["accounting"]))
        and (len(all_ids) == done["distinct_remote_ids"]),
        "Final queue accounting differs",
    )
    require(
        len(unknown) == 8 and set(unknown) == admin, "Unknown request roster differs"
    )
    write(OUTPUT / "assembly/batches.json", batch_by_key)
    stage_bindings[rel(OUTPUT / "assembly/batches.json")] = sha(
        OUTPUT / "assembly/batches.json"
    )
    verify(ROOT, plan["source_hashes"])
    verify(ROOT, plan["input_hashes"])
    scope.verify_records()
    completion_gate()
    verify(ROOT, read(OUTPUT / "started.json")["analysis_source_hashes"])
    write(
        OUTPUT / "assembly/complete.json",
        {
            "status": "complete_independently_replayed_unscored",
            "completed_utc": utc(),
            "queue_complete_sha256": sha(EXECUTION / "live/complete.json"),
            "plan_sha256": PLAN_SHA,
            "scientific_plan_sha256": sha(STUDY / "scientific_plan_v1.json"),
            "artifacts": stage_bindings,
            "analysis_source_hashes": reviewer_sources(),
            "environment": environment(),
            "counts": dict(counts),
            "accounting": dict(accounting),
            "observed_usage": dict(usage),
            "statuses": dict(statuses),
            "unique_successful_response_ids": len(all_ids),
            "administrative_unknown_requests": 8,
            "reviewed_transport_amendment": transport.provenance(),
            "unknown_delivery_usage_duration": True,
            "new_model_calls": 0,
            "gold_scoring": False,
        },
    )
    print(
        "Complete saved-response assembly verified. Scientific scoring remains separate.",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", action="store_true")
    args = parser.parse_args()
    with offline():
        if args.review:
            review()
        else:
            try:
                value = completion_gate()
                print(
                    json.dumps(
                        {
                            "status": "queue_complete_review_not_started",
                            "predictions": value["counts"]["predictions"],
                        }
                    )
                )
            except ValueError as exc:
                if str(exc).startswith("WAIT:"):
                    print(
                        json.dumps(
                            {
                                "status": "waiting_for_original_queue",
                                "reason": str(exc),
                                "predictions_opened": 0,
                                "new_model_calls": 0,
                            }
                        )
                    )
                else:
                    raise


if __name__ == "__main__":
    main()

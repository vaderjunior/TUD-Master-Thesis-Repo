"""Save and audit model requests for study runs."""

from src.hsrag.paths import evaluation_path
from src.hsrag.paths import PROJECT_ROOT
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import base64
import hashlib
import json
import math
import os
from pathlib import Path
import time
from urllib.parse import urlsplit
import uuid

ROOT = PROJECT_ROOT
MODEL = "Qwen/Qwen3-VL-32B-Instruct"
CONTRACT = {
    "model": MODEL,
    "provider": "peasec",
    "temperature": 1.0,
    "max_tokens": 4096,
    "timeout_seconds": 180,
    "votes": 3,
    "max_repairs": 2,
    "max_transport_attempts": 5,
    "backoff_seconds": [2, 2, 4, 8],
    "allow_fallback": False,
    "interrupted_intent": "stop_for_review_without_resending",
    "schema_parse_failure": "completed_vote_after_at_most_two_repairs",
    "transport_failure": "stop_item_preserve_votes_do_not_reset_budget",
    "latency": "sum_saved_POST_elapsed_and_actual_retry_wait; excludes offline preparation and restart downtime",
}
DEPENDENCIES = [
    "analysis/common/api.py",
    "src/hsrag/parse.py",
    "src/hsrag/vote.py",
    "src/hsrag/schema.py",
    "config/taxonomy.yaml",
]


def require(ok, message):
    if not ok:
        raise IntegrityError(message)


class IntegrityError(RuntimeError):
    pass


class IncompleteAttempt(RuntimeError):
    pass


class CallStopped(RuntimeError):
    pass


class TransportFault(Exception):
    """Safe network error category only; never embed URLs/headers/tokens."""

    def __init__(self, category):
        self.category = category
        super().__init__(category)


def encoded(x):
    return (
        json.dumps(x, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf8")


def signature(x):
    return hashlib.sha256(encoded(x)).hexdigest()


def storage_path(p):
    """Use the Windows extended-length namespace for deep experiment paths."""
    p = Path(p).absolute()
    s = str(p)
    if os.name == "nt" and (not s.startswith("\\\\?\\")):
        s = "\\\\?\\UNC\\" + s[2:] if s.startswith("\\\\") else "\\\\?\\" + s
        return Path(s)
    return p


def read(p):
    return json.loads(storage_path(evaluation_path(p)).read_text(encoding="utf8"))


def digest(p):
    return hashlib.sha256(storage_path(evaluation_path(p)).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def implementation():
    return {name: digest(ROOT / name) for name in DEPENDENCIES}


def save_new(path, obj):
    """Atomic no-overwrite publication, including concurrent identical blobs."""
    path = storage_path(path)
    data = encoded(obj)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, f"Preserve conflicting artifact: {path.name}")
        return
    temp = path.with_name(path.name + ".partial_" + uuid.uuid4().hex)
    with temp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        try:
            os.link(temp, path)
        except FileExistsError:
            require(path.read_bytes() == data, f"Concurrent artifact conflict: {path.name}")
    finally:
        temp.unlink()


@contextmanager
def item_lock(path):
    """OS lock releases after process death; stale file alone is harmless."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as f:
        if path.stat().st_size == 0:
            f.write(b"0")
            f.flush()
        f.seek(0)
        if os.name == "nt":
            import msvcrt

            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                raise IntegrityError("This item is already running") from None
        else:
            import fcntl

            try:
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                raise IntegrityError("This item is already running") from None
        try:
            yield
        finally:
            f.seek(0)
            if os.name == "nt":
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(f, fcntl.LOCK_UN)


def body(messages):
    require(type(messages) is list and bool(messages), "Messages required")
    for m in messages:
        require(
            set(m) == {"role", "text"}
            and m["role"] in {"system", "user", "assistant"}
            and (type(m["text"]) is str),
            "Unexpected internal message shape",
        )
    return {
        "model": MODEL,
        "temperature": CONTRACT["temperature"],
        "max_tokens": CONTRACT["max_tokens"],
        "messages": [{"role": m["role"], "content": m["text"]} for m in messages],
    }


@dataclass(frozen=True)
class WireReply:
    status: int
    content: bytes
    headers: dict


class RequestsTransport:
    """Explicit live adapter. No .env loading, fallback, redirects or hidden retry."""

    def __init__(self, base_url, token):
        url = urlsplit(base_url)
        require(
            url.scheme == "https"
            and url.hostname == "openwebui.srv.peasec.de"
            and (url.port in (None, 443))
            and (not url.username)
            and (not url.password)
            and (not url.query)
            and (not url.fragment)
            and (url.path.rstrip("/") == "/api/v1"),
            "Unexpected PEASEC endpoint; review before sending data",
        )
        require(isinstance(token, str) and bool(token.strip()), "Missing PEASEC token")
        self._endpoint = base_url.rstrip("/") + "/chat/completions"
        self._token = token

    def __call__(self, payload):
        import requests

        try:
            response = requests.post(
                self._endpoint,
                json=payload,
                headers={
                    "Authorization": "Bearer " + self._token,
                    "Content-Type": "application/json",
                },
                timeout=CONTRACT["timeout_seconds"],
                allow_redirects=False,
            )
            return WireReply(response.status_code, response.content, dict(response.headers))
        except requests.exceptions.RequestException as exc:
            raise TransportFault(type(exc).__name__) from None


SAFE_HEADERS = {"content-type", "date", "retry-after", "x-request-id", "request-id"}


def interpret(outcome):
    """Gateway/transport failures are distinct from classifier JSON failures."""
    if outcome["kind"] == "transport_error":
        return ("retry", outcome["category"], None)
    require(outcome["kind"] == "http_response", "Unknown response outcome kind")
    raw = base64.b64decode(outcome["body_base64"], validate=True)
    require(hashlib.sha256(raw).hexdigest() == outcome["body_sha256"], "Response bytes changed")
    status = outcome["status"]
    text = raw.decode("utf8", errors="replace")
    if status in (401, 403):
        return ("stop", "authentication", None)
    if status >= 400 and "not available" in text.lower():
        return ("stop", "model_unavailable", None)
    if status == 429 or 500 <= status <= 599:
        return ("retry", "http_" + str(status), None)
    if not 200 <= status <= 299:
        return ("stop", "http_" + str(status), None)
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return ("retry", "invalid_gateway_json", None)
    if not isinstance(data, dict):
        return ("stop", "malformed_gateway_response", None)
    reported = data.get("model")
    if reported is not None and reported != MODEL:
        return ("stop", "reported_model_mismatch", None)
    try:
        choice = data["choices"][0]
        content = choice["message"]["content"]
        finish = choice.get("finish_reason")
    except (KeyError, IndexError, TypeError):
        return ("stop", "malformed_gateway_response", None)
    if finish == "length":
        return ("stop", "truncated_response", None)
    if finish not in ("stop", None):
        return ("stop", "unexpected_finish_reason", None)
    if not isinstance(content, str):
        return ("stop", "nontext_response", None)
    return (
        "ok",
        content,
        {"reported_model": reported, "finish_reason": finish, "usage": data.get("usage")},
    )


def finite_nonnegative(x):
    return type(x) in (int, float) and math.isfinite(x) and (x >= 0)


class DurableVoteClient:

    def __init__(self, root, job, job_stamp, vote, transport, sleep=time.sleep):
        self.root, self.job = (Path(root), Path(job))
        self.stamp = job_stamp
        self.vote = vote
        self.transport, self.sleep = (transport, sleep)
        self.turn = 0
        self.active_model = MODEL
        self.used = set()
        self.events = []

    def complete(self, messages):
        require(0 <= self.vote < 3 and 0 <= self.turn <= 2, "Vote/repair budget exceeded")
        payload = body(messages)
        body_sha = signature(payload)
        blob = self.root / "payloads" / f"{body_sha}.json"
        save_new(blob, payload)
        self.used.add(blob)
        call = self.job / "calls" / f"vote_{self.vote}_turn_{self.turn}"
        for attempt in range(5):
            folder = call / f"attempt_{attempt}"
            intent_path = folder / "intent.json"
            out_path = folder / "outcome.json"
            intent_identity = {
                "job_stamp": self.stamp,
                "vote": self.vote,
                "turn": self.turn,
                "attempt": attempt,
                "kind": "ordinary" if self.turn == 0 else "repair",
                "body_sha256": body_sha,
            }
            if intent_path.exists():
                intent = read(intent_path)
                require(
                    set(intent) == set(intent_identity) | {"started_utc", "backoff_elapsed_s"},
                    "Intent fields differ",
                )
                require(
                    all((intent[k] == v for k, v in intent_identity.items())),
                    "Resumed request identity/body differs",
                )
                if not out_path.exists():
                    raise IncompleteAttempt(
                        "Saved request intent has no outcome; inspect before any resend"
                    )
                outcome = read(out_path)
            else:
                require(not out_path.exists(), "Outcome without request intent")
                waited = 0.0
                if attempt:
                    start = time.perf_counter()
                    self.sleep(CONTRACT["backoff_seconds"][attempt - 1])
                    waited = time.perf_counter() - start
                intent = {**intent_identity, "started_utc": utc(), "backoff_elapsed_s": waited}
                save_new(intent_path, intent)
                start = time.perf_counter()
                try:
                    reply = self.transport(payload)
                    require(
                        isinstance(reply, WireReply)
                        and type(reply.status) is int
                        and isinstance(reply.content, bytes),
                        "Invalid transport adapter result",
                    )
                    outcome = {
                        "kind": "http_response",
                        "status": reply.status,
                        "body_base64": base64.b64encode(reply.content).decode("ascii"),
                        "body_sha256": hashlib.sha256(reply.content).hexdigest(),
                        "headers": {
                            k.lower(): v
                            for k, v in reply.headers.items()
                            if k.lower() in SAFE_HEADERS
                        },
                    }
                except TransportFault as exc:
                    outcome = {
                        "kind": "transport_error",
                        "category": exc.category,
                        "remote_completion_uncertain": True,
                    }
                outcome.update(
                    intent_sha256=signature(intent),
                    finished_utc=utc(),
                    elapsed_s=time.perf_counter() - start,
                )
                save_new(out_path, outcome)
            require(
                outcome["intent_sha256"] == signature(intent),
                "Response belongs to different intent",
            )
            require(
                finite_nonnegative(outcome["elapsed_s"])
                and finite_nonnegative(intent["backoff_elapsed_s"]),
                "Invalid saved timing",
            )
            self.used.update([intent_path, out_path])
            action, value, extra = interpret(outcome)
            self.events.append(
                {
                    "vote": self.vote,
                    "turn": self.turn,
                    "attempt": attempt,
                    "kind": intent["kind"],
                    "action": action,
                    "response_class": "success" if action == "ok" else value,
                    "elapsed_s": outcome["elapsed_s"],
                    "backoff_elapsed_s": intent["backoff_elapsed_s"],
                    "model_response_metadata": extra,
                }
            )
            if action == "ok":
                self.turn += 1
                return value
            if action == "stop":
                raise CallStopped(value)
        raise CallStopped("transport_attempt_budget_exhausted")


def execute_item(root, identity, messages, transport, sleep=time.sleep, execution_mode="mock"):
    """Resume one three-vote item. No gold, retrieval, .env or batch scheduling."""
    from src.hsrag.parse import run_once
    from src.hsrag.vote import aggregate

    root = storage_path(root).resolve()
    body(messages)
    require(execution_mode in ("mock", "live"), "Explicit execution mode required")
    require(
        (execution_mode == "live") == isinstance(transport, RequestsTransport),
        "Live adapter must be explicitly marked live; simulated outputs cannot be real results",
    )
    require(
        type(identity) is dict
        and set(identity) == {"experiment_sha256", "condition", "repeat", "key"},
        "Explicit experiment/condition/repeat/input identity required",
    )
    require(
        type(identity["repeat"]) is int and identity["repeat"] in (0, 1, 2),
        "Unexpected decoder repeat",
    )
    require(
        all(
            (
                isinstance(identity[k], str) and identity[k]
                for k in ("experiment_sha256", "condition", "key")
            )
        ),
        "Empty job identity",
    )
    stamp = {
        "identity": identity,
        "messages_sha256": signature(messages),
        "contract": CONTRACT,
        "implementation": implementation(),
        "execution_mode": execution_mode,
    }
    job = root / "items" / signature(identity)
    done = job / "complete.json"
    with item_lock(job / "item.lock"):
        save_new(job / "job.json", stamp)
        if done.exists():
            receipt = read(done)
            require(receipt["job_stamp"] == signature(stamp), "Completed job identity differs")
            verify_inventory(root, job, receipt["hashes"])
            require(receipt["status"] == "complete", "Unexpected item status")
            return read(job / "result.json")
        used = {job / "job.json"}
        runs = []
        events = []
        for vote in range(3):
            client = DurableVoteClient(root, job, signature(stamp), vote, transport, sleep)
            run = run_once(client, messages, max_repairs=2)
            run.latency_s = sum((e["elapsed_s"] + e["backoff_elapsed_s"] for e in client.events))
            saved = asdict(run)
            saved["result"] = run.result.model_dump() if run.result is not None else None
            saved["gate_normalised"] = run.gate_normalised
            vote_path = job / "votes" / f"{vote}.json"
            save_new(vote_path, saved)
            used.add(vote_path)
            used.update(client.used)
            runs.append(run)
            events.extend(client.events)
        vote = aggregate(runs)
        result = {
            "identity": identity,
            "messages_sha256": signature(messages),
            "active_model": MODEL if execution_mode == "live" else None,
            "requested_model": MODEL,
            "execution_mode": execution_mode,
            "result": vote.result.model_dump() if vote.result is not None else None,
            "n_valid": vote.n_valid,
            "n_runs": vote.n_runs,
            "uncertain": vote.uncertain,
            "agreement": vote.agreement,
            "raw_runs": [r.raw_outputs for r in runs],
            "run_predictions": [
                r.result.model_dump() if r.result is not None else None for r in runs
            ],
            "parse_failures": sum((r.parse_failure for r in runs)),
            "repairs": sum((r.repairs for r in runs)),
            "normalisations": sum((r.normalisations for r in runs)),
            "gate_normalised": sum((r.gate_normalised for r in runs)),
            "latencies": [r.latency_s for r in runs],
            "accounting": {
                "ordinary_logical_calls": 3,
                "repair_logical_calls": sum((r.repairs for r in runs)),
                "post_attempts": len(events),
                "retry_attempts": sum((e["attempt"] > 0 for e in events)),
                "retryable_failures": sum((e["action"] == "retry" for e in events)),
                "reported_model_present": sum(
                    (
                        e["action"] == "ok"
                        and e["model_response_metadata"]["reported_model"] is not None
                        for e in events
                    )
                ),
                "http_success_with_text": sum((e["action"] == "ok" for e in events)),
            },
            "events": events,
        }
        result_path = job / "result.json"
        save_new(result_path, result)
        used.add(result_path)
        hashes = {p.relative_to(root).as_posix(): digest(p) for p in sorted(used)}
        verify_inventory(root, job, hashes)
        save_new(
            done,
            {
                "status": "complete",
                "job_stamp": signature(stamp),
                "created_utc": utc(),
                "hashes": hashes,
            },
        )
        return result


def verify_inventory(root, job, hashes):
    require(isinstance(hashes, dict) and bool(hashes), "Missing item hash inventory")
    for name, h in hashes.items():
        path = (root / name).resolve()
        require(
            path.is_relative_to(root) and path.is_file() and (digest(path) == h),
            "Changed/missing completed artifact",
        )
        require(
            path.is_relative_to(job) or path.parent == root / "payloads",
            "Artifact outside this item/payload store",
        )
    actual = {
        p.relative_to(root).as_posix()
        for p in job.rglob("*")
        if p.is_file() and p not in (job / "item.lock", job / "complete.json")
    }
    expected = {name for name in hashes if (root / name).is_relative_to(job)}
    require(actual == expected, "Unreviewed item attempt, partial file or missing artifact")


def audit_item(root, identity, messages, execution_mode="mock"):
    """Read-only semantic replay from saved bodies through original parse/vote."""
    from src.hsrag.parse import parse_response, REPAIR_TEMPLATE, RunResult
    from src.hsrag.vote import aggregate

    root = storage_path(root).resolve()
    job = root / "items" / signature(identity)
    receipt = read(job / "complete.json")
    verify_inventory(root, job, receipt["hashes"])
    stamp = read(job / "job.json")
    require(
        stamp
        == {
            "identity": identity,
            "messages_sha256": signature(messages),
            "contract": CONTRACT,
            "implementation": implementation(),
            "execution_mode": execution_mode,
        },
        "Audit input/source changed",
    )
    require(receipt["job_stamp"] == signature(stamp), "Audit completion stamp differs")
    saved = read(job / "result.json")
    runs = []
    events = []
    for vote in range(3):
        convo = list(messages)
        run = RunResult(result=None)
        for turn in range(3):
            success = False
            for attempt in range(5):
                folder = job / "calls" / f"vote_{vote}_turn_{turn}" / f"attempt_{attempt}"
                intent = read(folder / "intent.json")
                outcome = read(folder / "outcome.json")
                expected = {
                    "job_stamp": signature(stamp),
                    "vote": vote,
                    "turn": turn,
                    "attempt": attempt,
                    "kind": "ordinary" if turn == 0 else "repair",
                    "body_sha256": signature(body(convo)),
                }
                require(
                    all((intent[k] == v for k, v in expected.items())),
                    "Recorded repair conversation/identity differs",
                )
                require(
                    read(root / "payloads" / f"{intent['body_sha256']}.json") == body(convo),
                    "Recorded wire payload differs",
                )
                require(
                    outcome["intent_sha256"] == signature(intent),
                    "Recorded outcome/intent mismatch",
                )
                action, value, extra = interpret(outcome)
                event = {
                    "vote": vote,
                    "turn": turn,
                    "attempt": attempt,
                    "kind": intent["kind"],
                    "action": action,
                    "response_class": "success" if action == "ok" else value,
                    "elapsed_s": outcome["elapsed_s"],
                    "backoff_elapsed_s": intent["backoff_elapsed_s"],
                    "model_response_metadata": extra,
                }
                events.append(event)
                run.latency_s += outcome["elapsed_s"] + intent["backoff_elapsed_s"]
                if action == "ok":
                    success = True
                    break
                require(action == "retry", "Terminal request cannot be part of completed item")
            require(success, "Exhausted call cannot be part of completed item")
            result, n, error = parse_response(value)
            run.raw_outputs.append(value)
            run.normalisations += n
            run.repairs = turn
            if result is not None:
                run.result = result
                run.gate_normalised = result.gate_normalised
                break
            run.error = error
            convo = convo + [
                {"role": "assistant", "text": value},
                {"role": "user", "text": REPAIR_TEMPLATE.format(error=error)},
            ]
        run.parse_failure = run.result is None
        runs.append(run)
        record = asdict(run)
        record["result"] = run.result.model_dump() if run.result else None
        require(
            record == read(job / "votes" / f"{vote}.json"),
            "Saved vote does not reproduce from raw responses",
        )
    final = aggregate(runs)
    require(
        saved["identity"] == identity
        and saved["messages_sha256"] == signature(messages)
        and (saved["active_model"] == (MODEL if execution_mode == "live" else None))
        and (saved["requested_model"] == MODEL)
        and (saved["execution_mode"] == execution_mode),
        "Saved item attribution differs",
    )
    expected = {
        "result": final.result.model_dump() if final.result else None,
        "n_valid": final.n_valid,
        "n_runs": 3,
        "uncertain": final.uncertain,
        "agreement": final.agreement,
        "events": events,
        "raw_runs": [r.raw_outputs for r in runs],
        "run_predictions": [r.result.model_dump() if r.result else None for r in runs],
        "parse_failures": sum((r.parse_failure for r in runs)),
        "repairs": sum((r.repairs for r in runs)),
        "normalisations": sum((r.normalisations for r in runs)),
        "gate_normalised": sum((r.gate_normalised for r in runs)),
        "latencies": [r.latency_s for r in runs],
        "accounting": {
            "ordinary_logical_calls": 3,
            "repair_logical_calls": sum((r.repairs for r in runs)),
            "post_attempts": len(events),
            "retry_attempts": sum((e["attempt"] > 0 for e in events)),
            "retryable_failures": sum((e["action"] == "retry" for e in events)),
            "reported_model_present": sum(
                (
                    e["action"] == "ok"
                    and e["model_response_metadata"]["reported_model"] is not None
                    for e in events
                )
            ),
            "http_success_with_text": sum((e["action"] == "ok" for e in events)),
        },
    }
    require(
        set(saved)
        == set(expected)
        | {"identity", "messages_sha256", "active_model", "requested_model", "execution_mode"},
        "Unexpected item result fields",
    )
    require(
        all((saved[k] == v for k, v in expected.items())), "Saved classification does not reproduce"
    )
    return {
        "status": "verified_from_saved_responses",
        "post_attempts": len(events),
        "valid_votes": final.n_valid,
        "null_result": final.result is None,
        "messages_sha256": signature(messages),
    }

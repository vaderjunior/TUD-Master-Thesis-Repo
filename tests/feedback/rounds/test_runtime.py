from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import json
import threading
import time
import uuid
import pytest
from analysis.feedback.rounds import runtime as m
from tests.feedback.common import test_queue as queue_fixtures
from tests.feedback.rounds import test_sequence as sequence_fixtures


@pytest.fixture(autouse=True)
def no_network():
    with m.c.offline(), patch.object(m.q.template.core, "live_client", m.c.forbidden):
        yield


def fixture(n=4):
    return queue_fixtures.QueueTests().fixture(n=n, conditions=("reference",))


def subset(stage, n):
    jobs = stage["jobs"][:n]
    return dict(stage, jobs=jobs, items=len(jobs), ordinary_calls=len(jobs) * 3)


def record():
    return dict(
        status="offline_verified_ready_for_user",
        effective_workers=128,
        original_workers=8,
        api_contract=deepcopy(m.a.CONTRACT),
        parent_plan_sha256=m.c.sha(m.c.ITERATIVE / "plan.json"),
        predictions=259224,
        ordinary_calls=777672,
        rounds=4,
        streams=3,
    )


def test_approved_contract():
    m.authorize(record())


@pytest.mark.parametrize(
    "key,value",
    [
        ("effective_workers", True),
        ("effective_workers", 64),
        ("effective_workers", 129),
        ("effective_workers", 128.0),
        ("original_workers", 128),
        ("predictions", 259223),
        ("ordinary_calls", 777673),
        ("rounds", 3),
        ("streams", 2),
        ("status", "untested"),
        ("api_contract", {}),
    ],
)
def test_scope_and_retry_changes_rejected(key, value):
    r = record()
    r[key] = value
    with pytest.raises(RuntimeError):
        m.authorize(r)


def test_only_worker_and_provenance_globals_change():
    b, *_ = fixture()
    fn = b.execute_stage
    validator = b.validate
    m.bind_stage(b, {"fixture": True})
    assert b.execute_stage.__code__ is fn.__code__
    assert (
        b.execute_stage.__defaults__ == fn.__defaults__
        and b.execute_stage.__kwdefaults__ == fn.__kwdefaults__
    )
    assert {
        k for k, v in b.execute_stage.__globals__.items() if v is not fn.__globals__[k]
    } == {"WORKERS", "save"}
    assert (
        b.validate is validator
        and validator.__globals__["WORKERS"] == 8
        and fn.__globals__["WORKERS"] == 8
    )
    with pytest.raises(RuntimeError):
        m.bind_stage(b, {"fixture": True})


def test_original_controller_changes_only_phase_execution():
    fn = m.original.run
    bound = m.bound_controller()
    assert bound.__code__ is fn.__code__ and bound.__defaults__ == fn.__defaults__
    assert {k for k, v in bound.__globals__.items() if v is not fn.__globals__[k]} == {
        "q"
    }
    proxy = bound.__globals__["q"]
    assert {k for k, v in vars(proxy).items() if v is not vars(m.q)[k]} == {"execute"}


def test_128_concurrent_original_votes_and_restart():
    b, cat, s, root = fixture(n=180)
    stage = subset(s["stages"][0], 256)
    transport = m.q.SyntheticTransport()
    lock = threading.Lock()
    active = 0
    peak = 0

    def call(payload):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.015)
            return transport(payload)
        finally:
            with lock:
                active -= 1

    m.bind_stage(b, {"fixture": True})
    b.execute_stage(
        stage, cat, root, "fixture-plan", mode="mock", client_factory=lambda: call
    )
    b.audit.audit_stage(stage, cat, root, "fixture-plan", mode="mock")
    before = transport.count
    b.execute_stage(
        stage, cat, root, "fixture-plan", mode="mock", client_factory=m.c.forbidden
    )
    b.verify_review(stage, root, "fixture-plan", "mock")
    assert transport.count == before == 256 * 3 and 1 < peak <= 128
    markers = list((b.BASE / "concurrency_20261024_v1/provenance").rglob("*.json"))
    assert len(markers) == 1 and m.c.read(markers[0])["effective_workers"] == 128


def test_saved_completed_item_and_partial_vote_reused():
    b, cat, s, root = fixture(n=4)
    stage = s["stages"][0]
    m.bind_stage(b, {"fixture": True})
    transport = m.q.SyntheticTransport()
    backend = b.engines(cat)["config/taxonomy.yaml"]
    jobs = stage["jobs"]

    attempt = root / "stages" / stage["id"] / "attempt_0"
    b.execute_stage.__globals__["save"](
        attempt / "start.json",
        dict(
            plan_sha256="fixture-plan",
            stage=stage["id"],
            study=stage["study"],
            mode="mock",
            started_utc=m.a.utc(),
        ),
    )
    backend.execute_item(
        root,
        b.identity(jobs[0], "fixture-plan"),
        cat[jobs[0]["key"], "reference"]["messages"],
        transport,
        execution_mode="mock",
    )
    calls = 0

    def stop(payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt()
        return transport(payload)

    with pytest.raises(KeyboardInterrupt):
        backend.execute_item(
            root,
            b.identity(jobs[1], "fixture-plan"),
            cat[jobs[1]["key"], "reference"]["messages"],
            stop,
            execution_mode="mock",
        )

    before = {
        p: m.a.digest(p)
        for job in jobs[:2]
        for p in b.item_dir(root, job, "fixture-plan").rglob("*")
        if p.is_file()
    }
    with pytest.raises(m.a.IncompleteAttempt):
        b.execute_stage(
            stage, cat, root, "fixture-plan", mode="mock", client_factory=m.c.forbidden
        )
    assert transport.count == 4 and all(m.a.digest(p) == h for p, h in before.items())


def test_clean_completed_item_reused_before_rest_of_batch():
    b, cat, s, root = fixture()
    stage = s["stages"][0]
    m.bind_stage(b, {"fixture": True})
    transport = m.q.SyntheticTransport()
    attempt = root / "stages" / stage["id"] / "attempt_0"
    b.execute_stage.__globals__["save"](
        attempt / "start.json",
        dict(
            plan_sha256="fixture-plan",
            stage=stage["id"],
            study=stage["study"],
            mode="mock",
            started_utc=m.a.utc(),
        ),
    )
    job = stage["jobs"][0]
    backend = b.engines(cat)[job["schema_path"]]
    backend.execute_item(
        root,
        b.identity(job, "fixture-plan"),
        cat[job["key"], "reference"]["messages"],
        transport,
        execution_mode="mock",
    )
    before = {
        p: m.a.digest(p)
        for p in b.item_dir(root, job, "fixture-plan").rglob("*")
        if p.is_file()
    }
    b.save(
        attempt / "interrupted.json",
        dict(finished_utc=m.a.utc(), reason="Synthetic clean interruption"),
    )
    b.execute_stage(
        stage, cat, root, "fixture-plan", mode="mock", client_factory=lambda: transport
    )
    b.audit.audit_stage(stage, cat, root, "fixture-plan", mode="mock")
    assert transport.count == stage["items"] * 3 and all(
        m.a.digest(p) == h for p, h in before.items()
    )


def test_partial_vote_resume_after_known_transport_failure():
    b, cat, s, root = fixture(n=2)
    stage = s["stages"][0]
    m.bind_stage(b, {"fixture": True})
    attempt = root / "stages" / stage["id"] / "attempt_0"
    transport = m.q.SyntheticTransport()
    calls = 0
    b.execute_stage.__globals__["save"](
        attempt / "start.json",
        dict(
            plan_sha256="fixture-plan",
            stage=stage["id"],
            study=stage["study"],
            mode="mock",
            started_utc=m.a.utc(),
        ),
    )

    def call(payload):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise m.a.TransportFault("ReadTimeout")
        return transport(payload)

    def pause(_):
        raise KeyboardInterrupt()

    job = stage["jobs"][0]
    backend = b.engines(cat)[job["schema_path"]]
    with pytest.raises(KeyboardInterrupt):
        backend.execute_item(
            root,
            b.identity(job, "fixture-plan"),
            cat[job["key"], "reference"]["messages"],
            call,
            sleep=pause,
            execution_mode="mock",
        )
    before = {
        p: m.a.digest(p)
        for p in b.item_dir(root, job, "fixture-plan").rglob("*")
        if p.is_file()
    }
    b.save(
        attempt / "interrupted.json",
        dict(finished_utc=m.a.utc(), reason="Synthetic interruption during backoff"),
    )
    b.execute_stage(
        stage,
        cat,
        root,
        "fixture-plan",
        mode="mock",
        client_factory=lambda: call,
        sleep=lambda _: None,
    )
    b.audit.audit_stage(stage, cat, root, "fixture-plan", mode="mock")
    assert calls == stage["items"] * 3 + 1 and transport.count == stage["items"] * 3
    assert all(m.a.digest(p) == h for p, h in before.items())


def test_transport_backoff_stays_bounded_and_audited():
    b, cat, s, root = fixture(n=2)
    stage = subset(s["stages"][0], 1)
    m.bind_stage(b, {"fixture": True})
    transport = m.q.SyntheticTransport()
    attempts = 0
    waits = []

    def call(payload):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise m.a.TransportFault("ReadTimeout")
        return transport(payload)

    b.execute_stage(
        stage,
        cat,
        root,
        "fixture-plan",
        mode="mock",
        client_factory=lambda: call,
        sleep=waits.append,
    )
    b.audit.audit_stage(stage, cat, root, "fixture-plan", mode="mock")
    out = m.a.read(b.item_dir(root, stage["jobs"][0], "fixture-plan") / "result.json")
    assert attempts == 4 and waits == [2] and out["accounting"]["retry_attempts"] == 1


def test_exhausted_retry_stops_and_cannot_reset_budget():
    b, cat, s, root = fixture(n=2)
    stage = subset(s["stages"][0], 1)
    m.bind_stage(b, {"fixture": True})
    calls = 0

    def fail(payload):
        nonlocal calls
        calls += 1
        raise m.a.TransportFault("ReadTimeout")

    with pytest.raises(m.a.IntegrityError):
        b.execute_stage(
            stage,
            cat,
            root,
            "fixture-plan",
            mode="mock",
            client_factory=lambda: fail,
            sleep=lambda _: None,
        )
    with pytest.raises(m.a.IntegrityError):
        b.execute_stage(
            stage, cat, root, "fixture-plan", mode="mock", client_factory=m.c.forbidden
        )
    assert calls == 5 and list((root / "stages" / stage["id"]).glob("stop_*.json"))


def test_schema_invalid_output_remains_completed_not_resampled():
    b, cat, s, root = fixture(n=2)
    stage = subset(s["stages"][0], 1)
    m.bind_stage(b, {"fixture": True})
    calls = 0

    def invalid(payload):
        nonlocal calls
        calls += 1
        return m.a.WireReply(
            200,
            json.dumps(
                {
                    "id": "invalid-" + uuid.uuid4().hex,
                    "model": m.a.MODEL,
                    "choices": [
                        {"message": {"content": "not JSON"}, "finish_reason": "stop"}
                    ],
                    "usage": {
                        "prompt_tokens": 10,
                        "completion_tokens": 5,
                        "total_tokens": 15,
                    },
                }
            ).encode(),
            {},
        )

    b.execute_stage(
        stage, cat, root, "fixture-plan", mode="mock", client_factory=lambda: invalid
    )
    b.audit.audit_stage(stage, cat, root, "fixture-plan", mode="mock")
    b.execute_stage(
        stage, cat, root, "fixture-plan", mode="mock", client_factory=m.c.forbidden
    )
    out = m.a.read(b.item_dir(root, stage["jobs"][0], "fixture-plan") / "result.json")
    assert calls == 9 and out["result"] is None and out["n_valid"] == 0


def test_marker_mismatch_rejected():
    b, cat, s, root = fixture(n=2)
    stage = subset(s["stages"][0], 1)
    m.bind_stage(b, {"fixture": True})
    transport = m.q.SyntheticTransport()
    b.execute_stage(
        stage, cat, root, "fixture-plan", mode="mock", client_factory=lambda: transport
    )
    b.audit.audit_stage(stage, cat, root, "fixture-plan", mode="mock")
    path = next((b.BASE / "concurrency_20261024_v1/provenance").rglob("*.json"))
    old = path.read_bytes()
    path.with_name(path.name + ".original").write_bytes(old)
    record = m.c.read(path)
    record["effective_workers"] = 64
    path.write_bytes(m.c.encoded(record))
    with pytest.raises(RuntimeError):
        b.verify_review(stage, root, "fixture-plan", "mock")


def test_marker_write_failure_prevents_any_client():
    b, cat, s, root = fixture()
    fn = b.execute_stage
    old = fn.__globals__["save"]

    def fail(path, value):
        if "provenance" in Path(str(path)).parts:
            raise OSError("Synthetic marker failure")
        return old(path, value)

    with patch.dict(fn.__globals__, save=fail):
        m.bind_stage(b, {"fixture": True})
        with pytest.raises(OSError):
            b.execute_stage(
                s["stages"][0],
                cat,
                root,
                "fixture-plan",
                mode="mock",
                client_factory=m.c.forbidden,
            )


def test_all_four_rounds_and_zero_call_restart_with_amended_controller():
    root, visits, transports, stack = sequence_fixtures.Sequence().harness()
    amended = []

    def execute(b):
        amended.append(b.BASE)
        return m.q.execute(b)

    with stack, patch.object(m, "execute", execute):
        m.bound_controller()()
        m.bound_controller()()
    assert len(visits) == len(amended) == 28 and len(transports) == 14
    assert m.c.read(root / "complete.json")["totals"] == {
        "predictions": 259224,
        "ordinary_calls": 777672,
    }


def test_failed_screen_does_not_reach_next_evaluation():
    root, visits, transports, stack = sequence_fixtures.Sequence().harness("r2_source")
    with stack, patch.object(m, "execute", lambda b: m.q.execute(b)), pytest.raises(
        RuntimeError
    ):
        m.bound_controller()()
    assert (
        not any("r2_evaluation" in str(path) for path in visits)
        and not (root / "complete.json").exists()
    )


def test_lock_prevents_a_second_launcher():
    path = (
        m.c.ITERATIVE / "offline_check" / ("runtime_lock_" + uuid.uuid4().hex) / "lock"
    )
    with m.a.item_lock(path), pytest.raises(m.a.IntegrityError):
        with m.a.item_lock(path):
            raise AssertionError("Second lock acquired")

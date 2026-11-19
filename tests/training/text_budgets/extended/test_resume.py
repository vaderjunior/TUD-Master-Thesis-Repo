import json
from pathlib import Path
import subprocess
import sys
import uuid
from unittest.mock import Mock
import pytest
from training.text_budgets.extended import resume as r
from training.text_budgets.extended import common as b
from training.text_budgets.extended import train as train
from training.text_budgets.extended import audit as audit


@pytest.fixture
def isolated(monkeypatch):

    tmp_path = r.RECOVERY / "t" / uuid.uuid4().hex[:8]
    tmp_path.mkdir(parents=True)
    monkeypatch.setattr(r, "RECOVERY", tmp_path)
    with b.isolated(base=tmp_path):
        yield tmp_path


def test_controller_import_is_lightweight():
    script = (
        "import sys; from training.text_budgets.extended import resume; "
        'assert not any(x in sys.modules for x in ("numpy","torch","scipy","transformers"))'
    )
    subprocess.run(
        [sys.executable, "-X", "utf8", "-B", "-c", script], check=True, cwd=r.ROOT
    )


def test_immutable_receipts_and_path_boundary(isolated):
    path = isolated / "test.json"
    r.write_new(path, {"original": True})
    with pytest.raises(RuntimeError, match="Never replace"):
        r.write_new(path, {"original": False})
    assert r.read(path) == {"original": True}
    with pytest.raises(RuntimeError, match="escapes"):
        r.write_new(isolated.parent / "outside.json", {})


def test_controller_lock_exclusion(isolated):
    with r.controller_lock():
        with pytest.raises(RuntimeError, match="already running"):
            with r.controller_lock():
                pytest.fail("Second controller acquired lock")
    with r.controller_lock():
        pass


def test_existing_gpu_locks_exclude_second_process():
    script = (
        "import sys; from pathlib import Path; from training.text_budgets.extended import common as b"
        + chr(10)
        + "b.BASE = Path(sys.argv[1]); b.short.BASE = Path(sys.argv[2]); b.short.OLD = Path(sys.argv[3])"
        + chr(10)
        + "b.short.ROOT = Path(sys.argv[4])"
        + chr(10)
        + "try:"
        + chr(10)
        + " with b.locks(): pass"
        + chr(10)
        + "except RuntimeError: raise SystemExit(9)"
        + chr(10)
    )
    command = [
        sys.executable,
        "-X",
        "utf8",
        "-B",
        "-c",
        script,
        str(b.BASE),
        str(b.short.BASE),
        str(b.short.OLD),
        str(b.short.ROOT),
    ]
    with b.locks():
        result = subprocess.run(command, cwd=r.ROOT, capture_output=True, text=True)
        assert result.returncode == 9, result.stderr
    result = subprocess.run(command, cwd=r.ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_drive_preserves_pool_order_and_selection_barrier():
    seen = []
    jobs = [{"id": "one"}, {"id": "two"}]

    def call(task):
        seen.append(task)
        if task["op"] == "preflight":
            return dict(
                complete=False,
                pools=["z", "a"],
                pending={"a": [], "z": [{"id": "search"}]},
            )
        if task["op"] == "seal_selections":
            return {"jobs": jobs}
        return {}

    r.drive(call)
    assert [x["op"] for x in seen] == [
        "preflight",
        "search",
        "select",
        "select",
        "seal_selections",
        "final",
        "final",
        "audit",
        "audit",
        "complete",
    ]
    assert [x["pool"] for x in seen if x["op"] == "select"] == ["z", "a"]


def test_complete_restart_does_no_work():
    calls = []

    def call(task):
        calls.append(task)
        return {"complete": True}

    r.drive(call)
    assert calls == [{"op": "preflight"}]


@pytest.mark.parametrize(
    "failure_at", ["search", "select", "seal_selections", "final", "audit"]
)
def test_controller_stops_without_retry(failure_at):
    seen = []

    def call(task):
        seen.append(task["op"])
        if task["op"] == failure_at:
            raise RuntimeError("deliberate stop")
        if task["op"] == "preflight":
            return dict(complete=False, pools=["p"], pending={"p": [{"id": "s"}]})
        if task["op"] == "seal_selections":
            return {"jobs": [{"id": "f"}]}
        return {}

    with pytest.raises(RuntimeError, match="deliberate"):
        r.drive(call)
    assert (
        seen[-1] == failure_at
        and seen.count(failure_at) == 1
        and "complete" not in seen
    )


def test_search_uses_exact_original_trainer_and_guard(isolated, monkeypatch):
    job = b.job(b.NAMES[0], "hate", "l1_e20_u", 0, 0)
    data = {"sentinel": "data"}
    guard = Mock()
    execute = Mock(return_value="trained")
    monkeypatch.setattr(b, "store_guard", guard)
    monkeypatch.setattr(b, "data", lambda p: data)
    monkeypatch.setattr(train, "execute", execute)
    plan = {"grid": b.GRID, "stores": {"unchanged": True}}
    result = r.perform({"op": "search", "job": job}, plan, "original_plan")
    execute.assert_called_once_with(job, data, plan, "original_plan")
    guard.assert_called_once_with(plan["stores"], remaining=None)
    assert result["fit"] == "trained"


@pytest.mark.parametrize(
    "field,value",
    [("seed", 10), ("config", "l1_e10_u"), ("fold", None), ("id", "wrong")],
)
def test_rejects_undeclared_search_jobs(isolated, monkeypatch, field, value):
    job = b.job(b.NAMES[0], "hate", "l1_e20_u", 0, 0)
    job[field] = value
    execute = Mock()
    monkeypatch.setattr(train, "execute", execute)
    with pytest.raises((RuntimeError, KeyError)):
        r.perform({"op": "search", "job": job}, {"grid": b.GRID, "stores": None}, "p")
    execute.assert_not_called()


def test_guard_failure_cannot_start_fit(isolated, monkeypatch):
    job = b.job(b.NAMES[0], "hate", "l1_e20_u", 0, 0)
    monkeypatch.setattr(
        b, "store_guard", Mock(side_effect=RuntimeError("Need 3.5 GiB"))
    )
    execute = Mock()
    monkeypatch.setattr(train, "execute", execute)
    with pytest.raises(RuntimeError, match="3.5"):
        r.perform({"op": "search", "job": job}, {"grid": b.GRID, "stores": None}, "p")
    execute.assert_not_called()


def test_no_search_after_final_fit(isolated, monkeypatch):
    path = isolated / "refits/old/complete.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")
    execute = Mock()
    monkeypatch.setattr(train, "execute", execute)
    with pytest.raises(RuntimeError, match="after final"):
        r.perform(
            {"op": "search", "job": b.job(b.NAMES[0], "hate", "l1_e20_u", 0, 0)},
            {"grid": b.GRID},
            "p",
        )
    execute.assert_not_called()


def test_final_requires_all_selections(isolated, monkeypatch):
    job = b.job(b.NAMES[0], "hate", "l1_e20_u", 10, None)
    monkeypatch.setattr(r, "final_jobs", lambda *args: ({}, [job]))
    execute = Mock()
    monkeypatch.setattr(train, "execute", execute)
    with pytest.raises(RuntimeError, match="before complete"):
        r.perform({"op": "final", "job": job}, {}, "p")
    execute.assert_not_called()


def test_final_fit_then_original_chunk_resume(isolated, monkeypatch):
    job = b.job(b.NAMES[0], "hate", "l1_e20_u", 10, None)
    data = {"original": True}
    b.save(isolated / "all_selections.json", {})
    b.save(isolated / "data/inference.json", [{"text": "fixture"}])
    monkeypatch.setattr(r, "final_jobs", lambda *args: ({}, [job]))
    monkeypatch.setattr(b, "store_guard", Mock())
    monkeypatch.setattr(b, "data", lambda p: data)
    order = []

    def fit(*args):
        order.append(("fit", args))
        return "verified"

    def infer(*args):
        order.append(("inference", args))
        return "new"

    monkeypatch.setattr(train, "execute", fit)
    monkeypatch.setattr(train, "inference_job", infer)
    plan = {"stores": None}
    result = r.perform({"op": "final", "job": job}, plan, "p")
    assert order == [
        ("fit", (job, data, plan, "p")),
        ("inference", (job, data, "p", [{"text": "fixture"}])),
    ]
    assert result["fit"] == "verified" and result["inference"] == "new"


def test_fit_error_never_runs_inference(isolated, monkeypatch):
    job = b.job(b.NAMES[0], "hate", "l1_e20_u", 10, None)
    b.save(isolated / "all_selections.json", {})
    monkeypatch.setattr(r, "final_jobs", lambda *args: ({}, [job]))
    monkeypatch.setattr(b, "store_guard", Mock())
    monkeypatch.setattr(b, "data", lambda p: {})
    monkeypatch.setattr(
        train, "execute", Mock(side_effect=RuntimeError("training failed"))
    )
    infer = Mock()
    monkeypatch.setattr(train, "inference_job", infer)
    with pytest.raises(RuntimeError, match="training failed"):
        r.perform({"op": "final", "job": job}, {"stores": None}, "p")
    infer.assert_not_called()


def test_worker_error_stops_controller_without_retry(isolated, monkeypatch):
    manifest = isolated / "manifest.json"
    manifest.write_text("{}")
    monkeypatch.setattr(r, "MANIFEST", manifest)
    process = Mock()
    process.wait.return_value = 1
    popen = Mock(return_value=process)
    monkeypatch.setattr(r.subprocess, "Popen", popen)
    with pytest.raises(RuntimeError, match="no automatic retry"):
        r.launch_worker(isolated / "sessions/test", 1, {"op": "preflight"})
    popen.assert_called_once()
    process.wait.assert_called_once()


def test_ctrl_c_waits_for_child_and_propagates(monkeypatch):
    process = Mock()
    process.wait.side_effect = [KeyboardInterrupt(), 1]
    handler = Mock(return_value="old")
    monkeypatch.setattr(r.signal, "signal", handler)
    with pytest.raises(KeyboardInterrupt):
        r.wait_child(process)
    assert process.wait.call_count == 2 and handler.call_count == 2


def test_no_unbound_worker_request(isolated):
    with pytest.raises(RuntimeError, match="Invalid worker"):
        r.worker(isolated / "request.json")


def test_manifest_tampering_rejected(isolated, monkeypatch):
    monkeypatch.setattr(r, "BASE", isolated)
    monkeypatch.setattr(r, "MANIFEST", isolated / "manifest.json")
    (isolated / "plan.json").write_text("changed")
    (isolated / "manifest.json").write_text(
        json.dumps(dict(status="tested_ready", plan_sha256=r.PLAN_SHA, bindings={}))
    )
    with pytest.raises(RuntimeError, match="sealed plan changed"):
        r.verify_manifest()


def test_finish_refuses_missing_audits(isolated, monkeypatch):
    monkeypatch.setattr(r, "final_jobs", lambda *args: ({}, [{}] * 360))
    with pytest.raises(FileNotFoundError):
        r.perform({"op": "complete"}, {}, "p")
    assert not (isolated / "complete.json").exists()

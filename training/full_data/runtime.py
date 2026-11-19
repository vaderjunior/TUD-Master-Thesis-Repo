from contextlib import contextmanager
from datetime import datetime, timezone, timedelta
import importlib.metadata
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import signal
import socket
import subprocess
import time
from functools import lru_cache
from training.full_data.common import require
from training.full_data.common import read
from training.full_data.common import save
from training.full_data.common import sha
from training.full_data.plan import PACKAGE

OUTPUT = Path(os.environ.get("SQ2_RESULTS", "/results/sq2_full_20261029_v1"))
STOP = False


class Pause(RuntimeError):
    pass


def inside(root, relative):
    raw = str(relative)
    name = PurePosixPath(raw)
    require(
        raw
        and not name.is_absolute()
        and "\\" not in raw
        and ":" not in raw
        and all(p not in ("", "..", ".") for p in raw.split("/")),
        "Unsafe relative path.",
    )
    require(not Path(root).is_symlink(), "Redirected root.")
    root = Path(root).resolve()
    dest = root.joinpath(*name.parts)
    current = root
    for part in name.parts:
        current = current / part
        require(not current.is_symlink(), "Symlink is not allowed.")
    require(
        dest.resolve().is_relative_to(root) and dest.resolve() != root,
        "Path escapes its root.",
    )
    return dest


def files(root):
    root = Path(root)
    require(root.is_dir() and not root.is_symlink(), "Missing or redirected directory.")
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            require(not (Path(directory) / name).is_symlink(), "Unexpected symlink.")
        for name in sorted(names):
            yield Path(directory) / name


def tree_bytes(root):
    root = Path(root)
    require(
        root.is_dir() and not root.is_symlink(), "Missing or redirected storage root."
    )
    total = 4096
    for directory, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            require(
                not (Path(directory) / name).is_symlink(), "Unexpected storage symlink."
            )
        total += 4096 * len(dirs)
        for name in names:
            info = (Path(directory) / name).stat()
            total += max(info.st_size, getattr(info, "st_blocks", 0) * 512) + 4096
    return total


def verify_tree(root, bindings):
    for name, digest in bindings.items():
        require(sha(inside(root, name)) == digest, "Changed saved file: " + name)


def manifest(root):
    return {p.relative_to(root).as_posix(): sha(p) for p in files(root)}


@contextmanager
def lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(
                "Another process holds this lease: " + str(path)
            ) from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


@contextmanager
def offline():
    from unittest.mock import patch

    def blocked(*args, **kwargs):
        raise RuntimeError("Network calls are disabled.")

    with patch.object(socket.socket, "connect", blocked), patch.object(
        socket.socket, "connect_ex", blocked
    ), patch.object(socket, "create_connection", blocked):
        yield


@lru_cache(maxsize=1)
def settings():
    return read(PACKAGE / "plan.json")


def validate_package():
    expected = read(PACKAGE / "manifest.json")
    actual = {p.relative_to(PACKAGE).as_posix() for p in files(PACKAGE)}
    require(
        actual == set(expected["files"]) | {"manifest.json"},
        "Bundle has missing or unexpected files.",
    )
    verify_tree(PACKAGE, expected["files"])
    cfg = settings()
    require(
        cfg["project_bytes"] == 250_000_000_000 and cfg["Qwen_calls"] == 0,
        "Wrong storage/network contract.",
    )
    require(
        cfg["input_bytes"]
        + cfg["output_bytes"]
        + cfg["image_log_bytes"]
        + cfg["other_bytes"]
        == cfg["project_bytes"],
        "Storage allowances do not add up.",
    )
    return sha(PACKAGE / "manifest.json")


def configure():
    require(
        platform.system() == "Linux",
        "The scientific runner is for the assigned Linux container.",
    )
    require(
        PACKAGE.resolve() == Path("/inputs/sq2_full")
        and OUTPUT.resolve() == Path("/results/sq2_full_20261029_v1"),
        "Keep the prepared input and result locations.",
    )
    require(
        Path("/sys/fs/cgroup/cpu.max").read_text().split() == ["400000", "100000"],
        "Changed CPU service limit.",
    )
    require(
        int(Path("/sys/fs/cgroup/memory.max").read_text()) == 32 * 2**30,
        "Changed RAM service limit.",
    )
    cpus = sorted(os.sched_getaffinity(0))
    require(len(cpus) >= 3, "Need the three assigned CPU slots.")
    os.sched_setaffinity(0, cpus[:3])
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[key] = "1"
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        TOKENIZERS_PARALLELISM="false",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONDONTWRITEBYTECODE="1",
        THESIS_INPUT_DIR=str(PACKAGE / "data"),
        HF_HOME="/results/sq2_full_20261029_v1/cache",
        XDG_CACHE_HOME="/results/sq2_full_20261029_v1/cache",
    )
    require(
        platform.python_version() == settings()["python"], "Python version changed."
    )
    for name, version in settings()["packages"].items():
        require(
            importlib.metadata.version(name) == version,
            "Package version changed: " + name,
        )


def memory():
    import psutil

    limit = int(Path("/sys/fs/cgroup/memory.max").read_text())
    current = int(Path("/sys/fs/cgroup/memory.current").read_text())
    available = min(psutil.virtual_memory().available, limit - current)
    if current > int(0.8 * limit) or available < 3.5 * 2**30:
        raise Pause(
            "RAM limit reached. Keep all outputs and resume after freeing memory."
        )
    return dict(current=current, limit=limit, available=available)


def gpu():
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total,memory.used,utilization.gpu,temperature.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    lines = result.stdout.strip().splitlines()
    require(len(lines) == 1, "Exactly one GPU must be visible.")
    uid, name, total, used, busy, temp, driver = [
        x.strip() for x in lines[0].split(",")
    ]
    require("6000" in name and "Blackwell" in name, "Wrong assigned GPU.")
    require(
        driver == settings()["driver_version"],
        "GPU driver changed; review before reusing this profile.",
    )
    value = dict(
        uuid=uid,
        name=name,
        total_mib=int(total),
        used_mib=int(used),
        busy=int(busy),
        temperature=int(temp),
        driver=driver,
    )
    if value["used_mib"] > 0.8 * value["total_mib"] or value["temperature"] >= 85:
        raise Pause("GPU memory or temperature guard reached.")
    return value


def storage(reserve=0, inputs=Path("/inputs"), outputs=Path("/results")):
    cfg = settings()
    a, b = tree_bytes(inputs), tree_bytes(outputs)
    free = shutil.disk_usage(outputs).free
    if (
        a > cfg["input_bytes"]
        or b + reserve > cfg["output_bytes"]
        or free < cfg["free_disk_bytes"] + reserve
    ):
        raise Pause(
            "Storage guard reached. Preserve all files; archive and review before resuming."
        )
    return dict(
        inputs=a,
        outputs=b,
        reserve=reserve,
        free=free,
        project_limit=cfg["project_bytes"],
    )


def check_external_storage(path):
    v = read(path)
    cfg = settings()
    require(
        type(v["image_build_log_upper_bytes"]) is int
        and type(v["other_project_upper_bytes"]) is int,
        "Fill in integer byte bounds before launching.",
    )
    require(
        v["image_build_log_upper_bytes"] <= cfg["image_log_bytes"]
        and v["other_project_upper_bytes"] <= cfg["other_bytes"],
        "External storage exceeds its allowances.",
    )
    require(
        v["image_build_log_upper_bytes"] > 0
        and v["other_project_upper_bytes"] >= 0
        and len(v["evidence"].strip()) >= 10,
        "Record how the external storage bounds were checked.",
    )
    age = datetime.now(timezone.utc) - datetime.fromisoformat(v["checked_utc"])
    require(
        timedelta(0) <= age <= timedelta(days=7),
        "Refresh external image/log storage accounting.",
    )
    return v


def request_stop(*args):
    global STOP
    STOP = True


def deadline_guard(session=None):
    if STOP:
        raise Pause("Stop requested. Saved work is preserved.")
    now = datetime.now(timezone.utc)
    if now >= datetime.fromisoformat(settings()["stop_utc"]):
        raise Pause("The experiment stop date has been reached.")
    if session:
        if now >= datetime.fromisoformat(session["end_utc"]):
            raise Pause("The active compute budget has been reached.")
        last = getattr(deadline_guard, "last_check", 0.0)
        if time.monotonic() - last >= 1:
            if Path(session["stop_file"]).exists():
                raise Pause("Controller requested stop.")
            import psutil

            try:
                parent = psutil.Process(session["controller_pid"])
                require(
                    abs(parent.create_time() - session["controller_created"]) < 0.001,
                    "Controller process identity changed.",
                )
            except psutil.NoSuchProcess as exc:
                raise Pause("Controller exited; keep this attempt.") from exc
            deadline_guard.last_check = time.monotonic()


def stop_children(children, stop_file):
    Path(stop_file).touch(exist_ok=True)
    end = time.monotonic() + 45
    while any(p.poll() is None for p in children) and time.monotonic() < end:
        time.sleep(0.25)
    for p in children:
        if p.poll() is None:
            p.terminate()
    end = time.monotonic() + 30
    while any(p.poll() is None for p in children) and time.monotonic() < end:
        time.sleep(0.25)
    for p in children:
        if p.poll() is None:
            p.kill()
    for p in children:
        p.wait(timeout=30)

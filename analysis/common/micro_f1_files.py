"""Common micro f1 files."""

from src.hsrag.paths import PROJECT_ROOT
import os
import socket
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[name] = "1"


def blocked(*args, **kwargs):
    raise RuntimeError("This setup is offline; network calls are forbidden.")


socket.socket.connect = blocked
socket.socket.connect_ex = blocked
socket.create_connection = blocked
ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/micro_f1_setup_20261113_v1"
BINDINGS = {}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def bind(path):
    path = Path(path).resolve()
    key = rel(path)
    value = sha(path)
    require(key not in BINDINGS or BINDINGS[key] == value, "Source changed: " + key)
    BINDINGS[key] = value
    return path


def read(path):
    return json.loads(bind(path).read_text(encoding="utf-8"))


def lines(path):
    with bind(path).open(encoding="utf-8") as f:
        yield from (json.loads(line) for line in f if line.strip())


def save(name, value):
    path = (BASE / name).resolve()
    require(path.is_relative_to(BASE.resolve()), "Write outside new setup folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def freeze():
    require(not BASE.exists(), "Setup folder already exists; preserve it.")
    BASE.mkdir(parents=True)
    save(
        "preservation_before.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "authorised_part": 1,
            "scope": "Freeze comparison inventory, prepare readers, reproduce old macro; no wider real micro estimates.",
        },
    )
    print("Part 1 preservation snapshot saved.", flush=True)


if __name__ == "__main__":
    freeze()

"""Common runtime."""

from src.hsrag.paths import PROJECT_ROOT
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
import unicodedata
import uuid
from unittest.mock import patch

ROOT = PROJECT_ROOT
SHARED = ROOT / "experiments/astra/sq3_followups_20261012_v1"
TRANSFER = ROOT / "experiments/astra/sq3_transfer_20261012_v1"
ITERATIVE = ROOT / "experiments/astra/sq3_iterative_20261012_v1"
LINEAR = ROOT / "experiments/astra/sq3_linear_update_20261012_v1"
A15 = ROOT / "experiments/astra/sq3_reserved_slot_20261012_v1"
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
LATER = ROOT / "experiments/astra/later_evaluation_20261008_v1"
PREP = ROOT / "experiments/astra/offline_closure_20261006_v1/evaluation_preparation"
CANDIDATES = ROOT / "experiments/astra/sq3_additions_20261004_v1/benign_selection"
SELECTION = ROOT / "experiments/astra/sq3_additions_results_20261006_v1/benign_selection"
CPU = ROOT / "experiments/astra/sq2_cpu_baselines_20261012_v2"
NEAR = 0.9495
STREAM_SEEDS = (2026101201, 2026101202, 2026101203)


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf8")


def sig(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def norm(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def prefix(text):
    return norm(text).lstrip(": ")


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with Path(path).open(encoding="utf8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def own(path):
    p = Path(path).resolve()
    require(
        any(
            (
                p.is_relative_to(b.resolve()) and p != b.resolve()
                for b in (SHARED, TRANSFER, ITERATIVE, LINEAR)
            )
        ),
        "Outside new SQ3 roots",
    )
    return p


def save(path, value):
    from analysis.common.api import save_new

    save_new(own(path), value)


def write_bytes(path, data):
    p = own(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        require(p.read_bytes() == data, "Preserve existing file: " + rel(p))
        return
    temp = p.with_name(p.name + ".partial_" + uuid.uuid4().hex)
    with temp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.link(temp, p)
    temp.unlink()


def jsonlines(path, values):
    p = own(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_name(p.name + ".partial_" + uuid.uuid4().hex)
    with temp.open("xb") as f:
        for v in values:
            f.write(
                (json.dumps(v, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n").encode(
                    "utf8"
                )
            )
        f.flush()
        os.fsync(f.fileno())
    if p.exists():
        require(sha(temp) == sha(p), "Existing rows differ: " + rel(p))
    else:
        os.link(temp, p)
    temp.unlink()


def array(path, value):
    import io, numpy as np

    buf = io.BytesIO()
    np.save(buf, value, allow_pickle=False)
    write_bytes(path, buf.getvalue())


def bindings(paths):
    return {rel(p): sha(p) for p in paths}


def verify(mapping):
    for name, h in mapping.items():
        require(sha(ROOT / name) == h, "Frozen dependency changed: " + name)


def forbidden(*args, **kwargs):
    raise RuntimeError("Offline SQ3 preparation forbids network/model calls")


@contextmanager
def offline():
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        ANONYMIZED_TELEMETRY="False",
        TOKENIZERS_PARALLELISM="false",
        CUDA_VISIBLE_DEVICES="",
    )
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ):
        yield


def resources(min_ram=2.0, min_disk=3.0):
    import ctypes, shutil

    class Memory(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (k, ctypes.c_ulonglong)
            for k in (
                "total",
                "avail",
                "totalPage",
                "availPage",
                "totalVirtual",
                "availVirtual",
                "availExtended",
            )
        ]

    s = Memory()
    s.length = ctypes.sizeof(s)
    available = None
    if hasattr(ctypes, "windll"):
        require(
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(s)) != 0, "Memory query failed"
        )
        available = s.avail / 2**30
        require(
            available >= min_ram,
            f"Need {min_ram:.2f} GiB free RAM; currently {available:.2f}. Keep active Qwen intact.",
        )
    free = shutil.disk_usage(ROOT).free / 2**30
    require(free >= min_disk, f"Need {min_disk:.1f} GiB free disk; currently {free:.2f}.")
    return {"available_RAM_GiB": available, "free_disk_GiB": free}


def active_guard():
    p = read(A15 / "execution/plan.json")
    verify(p["source_hashes"])
    require(
        sha(A15 / "execution/plan.json") == read(A15 / "execution/ready.json")["plan_sha256"],
        "A15 plan changed",
    )
    return {
        "plan_sha256": sha(A15 / "execution/plan.json"),
        "source_hashes": p["source_hashes"],
        "Qwen_outcomes_read": False,
    }


class Union:

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def join(self, a, b):
        a, b = (self.find(a), self.find(b))
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def groups(self):
        from collections import defaultdict

        out = defaultdict(list)
        for i in range(len(self.parent)):
            out[self.find(i)].append(i)
        return list(out.values())


def hash_sample(rows, goal, seed, cluster_field):
    """Outcome-blind complete-cluster sample; stop after first whole cluster crosses goal."""
    from collections import defaultdict

    groups = defaultdict(list)
    for r in rows:
        groups[r[cluster_field]].append(r["key"])
    selected = []
    for g in sorted(groups, key=lambda g: text_sha(f"{seed}|{g}")):
        if len(selected) >= goal:
            break
        selected.extend(sorted(groups[g]))
    return sorted(selected)

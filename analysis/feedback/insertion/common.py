from contextlib import contextmanager
from copy import deepcopy
import os
from pathlib import Path
import socket
from unittest.mock import patch
from analysis.feedback.balanced import common as old
from analysis.common import api as a

ROOT = old.ROOT
BASE = ROOT / "experiments/astra/sq3_five_slot_20261010_v2"
CORE, SELECTION, CANDIDATES, TAXONOMY, SEEDS = (
    old.CORE,
    old.SELECTION,
    old.CANDIDATES,
    old.TAXONOMY,
    old.SEEDS,
)
require, read, rows, sha, sig, rel, utc, verify, bindings = (
    old.require,
    old.read,
    old.rows,
    old.sha,
    old.sig,
    old.rel,
    old.utc,
    old.verify,
    old.bindings,
)
ARMS = [
    "error_positive",
    "error_balanced",
    "correct_positive",
    "correct_balanced",
    "random_positive",
    "random_balanced",
]


def own(path):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE.resolve()), "Write outside owned five-slot study")
    return path


def save(path, value):
    a.save_new(own(path), value)


def write(path, text):
    path = own(path)
    data = text.encode("utf8")
    if path.exists():
        require(path.read_bytes() == data, "Preserve existing file: " + rel(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".partial_" + __import__("uuid").uuid4().hex)
    with temp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    try:
        try:
            os.link(temp, path)
        except FileExistsError:
            require(path.read_bytes() == data, "Concurrent publication differs")
    finally:
        temp.unlink()


def jsonlines(path, values):
    import json

    write(
        path,
        "".join(
            (
                json.dumps(v, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
                for v in values
            )
        ),
    )


def ordinary(record):

    r = deepcopy(record)
    require(r["kind"] == "feedback" and r["lang"] == "en", "Unexpected source record")
    require(
        set(r["meta"]) == {"gate", "hate_types", "target_groups", "severity", "legal"},
        "Unknown annotation",
    )
    require(
        type(r["meta"]["gate"]) is bool
        and all((v is None for k, v in r["meta"].items() if k != "gate")),
        "Gate-only annotations required",
    )
    r.update(kind="example", dimension=None, label=None)
    return r


def forbidden(*args, **kwargs):
    raise AssertionError(
        "Offline five-slot operation: network and live clients disabled"
    )


@contextmanager
def offline():
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ):
        yield


def gather(folder, items, make, stamp, chunk_size=25, progress=None):
    folder = own(folder)
    folder.mkdir(parents=True, exist_ok=True)
    expected = {f"chunk_{i:06d}.json" for i in range(0, len(items), chunk_size)}
    require(
        {p.name for p in folder.iterdir()} <= expected,
        "Unknown/partial chunk: preserve for review",
    )
    result = []
    reused = 0
    for start in range(0, len(items), chunk_size):
        group = items[start : start + chunk_size]
        keys = [r["key"] for r in group]
        path = folder / f"chunk_{start:06d}.json"
        if path.exists():
            record = read(path)
            require(
                record["stamp"] == stamp
                and record["keys"] == keys
                and (record["rows_sha256"] == sig(record["rows"])),
                "Chunk attribution changed",
            )
            values = record["rows"]
            reused += len(values)
        else:
            values = [make(i, row) for i, row in enumerate(group, start)]
            require([r["key"] for r in values] == keys, "Producer changed roster")
            save(
                path,
                {
                    "stamp": stamp,
                    "keys": keys,
                    "rows": values,
                    "rows_sha256": sig(values),
                },
            )
        require([r["key"] for r in values] == keys, "Saved chunk roster changed")
        result.extend(values)
        if progress:
            progress(len(result), len(items), reused)
    return (
        result,
        {
            "rows": len(result),
            "reused_rows": reused,
            "chunks": len(expected),
            "hashes": bindings(sorted(folder.iterdir())),
        },
    )

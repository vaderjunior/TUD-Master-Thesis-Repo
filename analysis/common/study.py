"""Common study."""

from src.hsrag.paths import PROJECT_ROOT
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
from unittest.mock import patch

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/later_evaluation_20261008_v1"
PREP = ROOT / "experiments/astra/offline_closure_20261006_v1/evaluation_preparation"
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
CACHE = ROOT / "Thesis_Log/astra_result/evidence/astra_cached_vector_feasibility_20261008_v1.json"
EMPTY_KEY = "de/gahd-f4c7eaf2"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def rel(path):
    return Path(str(path).removeprefix("\\\\?\\")).resolve().relative_to(ROOT).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def rows(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(s) for s in f if s.strip()]


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    ).encode("utf-8")


def signature(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(mapping):
    for name, digest in mapping.items():
        require(sha(ROOT / name) == digest, "Changed frozen file: " + name)


def own(path):
    path = Path(path).resolve()
    require(
        path.is_relative_to(BASE.resolve()) and path != BASE.resolve(),
        "Outside later-evaluation root",
    )
    return path


def save(path, value):
    from analysis.common.api import save_new

    save_new(own(path), value)


def save_rows(path, values):
    path = own(path)
    data = "".join(
        (json.dumps(x, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n" for x in values)
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(path.read_bytes() == data, "Refusing to replace existing rows")
        return
    with path.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())


def forbidden(*args, **kwargs):
    raise RuntimeError("Offline preparation: network access forbidden")


@contextmanager
def offline():
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        ANONYMIZED_TELEMETRY="False",
        TOKENIZERS_PARALLELISM="false",
    )
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ):
        yield


def scientific_inputs():
    plan = read(PREP / "plan.json")
    complete = read(PREP / "complete.json")
    amendment = read(PREP / "coverage_amendment.json")
    coverage = read(PREP / "coverage_preparation.json")
    bindings = {
        **plan["inputs"],
        **complete["outputs"],
        **amendment["inputs"],
        **coverage["outputs"],
    }
    for name in [
        "complete.json",
        "coverage_preparation.json",
        "verification.json",
        "coverage_verification.json",
    ]:
        bindings[rel(PREP / name)] = sha(PREP / name)
    require(
        read(PREP / "verification.json")["status"] == "independent_offline_preparation_pass",
        "Original manipulation verification missing",
    )
    require(
        coverage["total_conditions"] == 13
        and coverage["fallback_count"] == 0
        and (coverage["broad_plus_matched_ordinary_calls"] == 694026),
        "Coverage amendment differs",
    )
    verify(bindings)
    return bindings


def matched_rows():
    result = rows(PREP / "matched_prompts.jsonl") + rows(PREP / "coverage_matched_prompts.jsonl")
    require(
        len(result) == len({(r["key"], r["condition"]) for r in result}) == 12233,
        "Incomplete matched matrix",
    )
    return result

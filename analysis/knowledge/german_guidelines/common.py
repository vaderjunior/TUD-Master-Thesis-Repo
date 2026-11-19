"""German guidelines common."""

from contextlib import contextmanager
from datetime import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket
import sys
from unittest.mock import patch
from analysis.common import api as a

ROOT = a.ROOT
BASE = ROOT / "experiments/astra/german_guidelines_20261112_v1"
OUTPUT = Path("D:/Thesis_Experiments/german_guidelines_20261112_v1")
PLAN = BASE / "plan.json"
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
BROAD = ROOT / "experiments/astra/later_evaluation_20261008_v1"
PREP = ROOT / "experiments/astra/offline_closure_20261006_v1/evaluation_preparation"
TOKENIZER = ROOT / "experiments/astra/qwen_tokenizer_assets_20261001_v1"
CONDITIONS = ("mixed_2", "general_2", "general_2_legal_1")
COUNTS = {
    "de_dev_eval": 150,
    "de_legal_dev_eval": 175,
    "de_test": 3175,
    "de_legal_test": 177,
}
WORKERS = 128
DEADLINE = datetime.fromisoformat("2026-11-15T23:59:59+01:00")
PROTOCOL = ROOT / "Thesis_Log/astra_result/german_guidelines_protocol_20261112_v1.md"
require = a.require
read = a.read
signature = a.signature
save = a.save_new


def normal(path):
    return Path(str(a.storage_path(path).resolve()).removeprefix("\\\\?\\"))


def ref(path):
    path = normal(path)
    return (
        path.relative_to(ROOT).as_posix()
        if path.is_relative_to(ROOT)
        else path.as_posix()
    )


def sha(path):
    h = hashlib.sha256()
    with a.storage_path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rows(path):
    with a.storage_path(path).open(encoding="utf8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_bytes(path, data):
    path = Path(path)
    require(normal(path).is_relative_to(BASE), "Preparation output outside new study")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        require(
            path.read_bytes() == data,
            "Preserve conflicting preparation file: " + path.name,
        )
    else:
        with path.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())


def jsonlines(path, values):
    write_bytes(
        path,
        "".join(
            (
                json.dumps(v, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
                for v in values
            )
        ).encode("utf8"),
    )


def bindings(paths):
    return {ref(p): sha(p) for p in paths}


def verify(hashes):
    for name, expected in hashes.items():
        require(sha(ROOT / name) == expected, "Changed frozen dependency: " + name)


def environment():
    names = (
        "numpy",
        "pandas",
        "pyarrow",
        "pydantic",
        "PyYAML",
        "requests",
        "urllib3",
        "python-dotenv",
        "rank-bm25",
        "tokenizers",
        "jinja2",
    )
    return {
        "python": sys.version.split()[0],
        "packages": {n: importlib.metadata.version(n) for n in names},
    }


def forbidden(*args, **kwargs):
    raise RuntimeError(
        "Offline preparation: network and live credentials are forbidden"
    )


@contextmanager
def offline():
    import requests
    import dotenv
    from analysis.common import batch as core

    os.environ.update(
        CUDA_VISIBLE_DEVICES="",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        HF_HUB_DISABLE_TELEMETRY="1",
        TOKENIZERS_PARALLELISM="false",
    )
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket.socket, "connect_ex", forbidden
    ), patch.object(socket, "create_connection", forbidden), patch.object(
        requests.sessions.Session, "request", forbidden
    ), patch.object(
        core, "live_client", forbidden
    ), patch.object(
        dotenv, "dotenv_values", forbidden
    ):
        yield


def source_files():
    names = [
        "analysis/knowledge/german_guidelines/common.py",
        "analysis/knowledge/german_guidelines/prepare.py",
        "analysis/knowledge/german_guidelines/run.py",
        "analysis/knowledge/german_guidelines/verify.py",
        "tests/knowledge/german_guidelines/test_run.py",
        "analysis/common/api.py",
        "analysis/common/batch.py",
        "analysis/common/scoring.py",
        "analysis/common/schema.py",
        "analysis/common/prompt_archive.py",
        "analysis/common/prompt.py",
        "analysis/common/result_reader.py",
        "analysis/common/runtime.py",
        "analysis/common/replay.py",
    ]
    return [
        *(ROOT / n for n in names),
        *(
            ROOT / n
            for n in (
                "src/hsrag/parse.py",
                "src/hsrag/vote.py",
                "src/hsrag/schema.py",
                "src/hsrag/prompt.py",
                "src/hsrag/retrieve.py",
                "config/taxonomy.yaml",
                "prompts/classify_v1.txt",
                "scripts/run_german_guidelines.ps1",
            )
        ),
        PROTOCOL,
    ]


def validate(ready=True):
    plan = read(PLAN)
    require(plan["environment"] == environment(), "Python environment changed")
    require(
        plan["workers"] == WORKERS and plan["api_contract"] == a.CONTRACT,
        "Frozen request contract changed",
    )
    require(
        plan["conditions"] == list(CONDITIONS) and plan["populations"] == COUNTS,
        "Frozen study matrix changed",
    )
    require(plan["output_root"] == OUTPUT.as_posix(), "Output drive changed")
    verify(plan["source_hashes"])
    verify(plan["input_hashes"])
    verify(plan["prepared_hashes"])
    if ready:
        receipt = read(BASE / "ready.json")
        require(
            receipt["status"] == "offline_verified_ready_for_user"
            and receipt["plan_sha256"] == sha(PLAN),
            "Offline readiness missing",
        )
        verify(receipt["checks"])
    return plan

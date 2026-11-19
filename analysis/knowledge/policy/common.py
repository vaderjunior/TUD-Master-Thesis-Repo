"""Policy common."""

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
SETUP = ROOT / "experiments/astra/complete_policy_setup_20261114_v1"
BASE = ROOT / "experiments/astra/complete_policy_20261114_v1"
OUTPUT = Path("D:/Thesis_Experiments/complete_policy_20261114_v1")
PLAN = BASE / "plan.json"
TOKENIZER = ROOT / "experiments/astra/qwen_tokenizer_assets_20261001_v1"
ORIGINAL_PROMPTS = (
    ROOT
    / "experiments/astra/later_evaluation_20261008_v1/retrieval_v3/full/prompts.jsonl"
)
CONDITIONS = ("retrieved_policy", "complete_definitions", "complete_policy")
WORKERS = 128
DEADLINE = datetime.fromisoformat("2026-11-15T23:59:59+01:00")
LATEST_START = datetime.fromisoformat("2026-11-15T05:59:59+01:00")
COUNTS = {
    "inputs": 6906,
    "prompts": 20718,
    "predictions": 62154,
    "ordinary_calls": 186462,
    "max_logical_calls": 559386,
    "max_transport_POST_attempts": 2796930,
}
SEED = "complete-policy-20261114-v1"
require, read, signature, save = (a.require, a.read, a.signature, a.save_new)


def normal(path):
    value = str(a.storage_path(path).resolve())
    if value.startswith("\\\\?\\UNC\\"):
        return Path("\\\\" + value[8:])
    return Path(value.removeprefix("\\\\?\\"))


def ref(path):
    path = normal(path)
    return (
        path.relative_to(ROOT).as_posix()
        if path.is_relative_to(ROOT)
        else path.as_posix()
    )


def sha(path):
    h = hashlib.sha256()
    with a.storage_path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path):
    with a.storage_path(path).open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def bindings(paths):
    return {ref(p): sha(p) for p in paths}


def verify(values):
    for name, expected in values.items():
        require(sha(ROOT / name) == expected, "Changed frozen file: " + name)


def environment():
    packages = (
        "numpy",
        "pydantic",
        "PyYAML",
        "requests",
        "urllib3",
        "python-dotenv",
        "tokenizers",
        "jinja2",
    )
    return {
        "python": sys.version.split()[0],
        "packages": {n: importlib.metadata.version(n) for n in packages},
    }


def forbidden(*args, **kwargs):
    raise RuntimeError(
        "Offline check: network access and live credentials are forbidden"
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
        socket, "getaddrinfo", forbidden
    ), patch.object(
        requests.sessions.Session, "request", forbidden
    ), patch.object(
        core, "live_client", forbidden
    ), patch.object(
        dotenv, "dotenv_values", forbidden
    ):
        yield


def source_files():
    names = [
        "analysis/knowledge/policy/common.py",
        "analysis/knowledge/policy/prepare.py",
        "analysis/knowledge/policy/engine.py",
        "analysis/knowledge/policy/run.py",
        "analysis/knowledge/policy/verify.py",
        "tests/knowledge/policy/test_run.py",
        "analysis/common/api.py",
        "analysis/common/batch.py",
        "analysis/common/schema.py",
        "analysis/common/prompt_archive.py",
        "analysis/common/prompt.py",
        "analysis/common/runtime.py",
        "analysis/knowledge/german_guidelines/run.py",
        "analysis/knowledge/german_guidelines/common.py",
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
                "config/taxonomy.yaml",
                "prompts/classify_v1.txt",
                "scripts/run_complete_policy.ps1",
            )
        ),
    ]


def part1_check():
    receipt = read(SETUP / "part1_complete.json")
    require(receipt["status"] == "OFFLINE_PART_1_COMPLETE", "Part 1 is incomplete")
    verify(receipt["reports"])
    verify(receipt["scripts"])
    verify({ref(SETUP / name): h for name, h in receipt["artifacts"].items()})
    verify(read(SETUP / "frozen_source_bindings.json"))
    protocol = read(SETUP / "protocol.json")
    require(
        [v["id"] for v in protocol["conditions"]] == list(CONDITIONS)
        and protocol["workers"] == WORKERS,
        "Part 1 design differs",
    )
    require(
        all((a.CONTRACT[k] == v for k, v in protocol["model_contract"].items())),
        "Request contract differs",
    )
    require(
        protocol["execution_design"]["deadline_local"] == DEADLINE.isoformat()
        and protocol["execution_design"]["latest_new_launch_local"]
        == LATEST_START.isoformat(),
        "Deadline changed",
    )
    return protocol


def validate(ready=True):
    plan = read(PLAN)
    require(plan["environment"] == environment(), "Python environment changed")
    require(
        plan["api_contract"] == a.CONTRACT and plan["workers"] == WORKERS,
        "Request contract changed",
    )
    require(
        plan["conditions"] == list(CONDITIONS) and plan["budget"] == COUNTS,
        "Study matrix changed",
    )
    require(plan["output_root"] == OUTPUT.as_posix(), "Output root changed")
    require(
        plan["deadline"] == DEADLINE.isoformat()
        and plan["latest_new_start"] == LATEST_START.isoformat(),
        "Run window changed",
    )
    for field in ("source_hashes", "input_hashes", "prepared_hashes"):
        verify(plan[field])
    if ready:
        record = read(BASE / "ready.json")
        require(
            record["status"] == "offline_verified_ready_for_user"
            and record["plan_sha256"] == sha(PLAN),
            "Offline readiness missing",
        )
        verify(record["checks"])
    return plan

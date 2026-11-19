from contextlib import contextmanager, ExitStack
from pathlib import Path
import socket
from unittest.mock import patch
import numpy as np
from training.roberta import common as c
from training.cpu import common as resource

ROOT = c.ROOT
OLD = c.BASE
BASE = ROOT / "experiments/astra/compact_roberta_short_20261012_v1"
SHORT = (1, 2, 5, 10, 15)
DURATIONS = SHORT + (20, 60)
PROTOCOL = (
    ROOT / "Thesis_Log/astra_result/astra_sq2_short_training_protocol_20261011_v1.md"
)
ORIGINAL_CONFIGS = c.configs
ORIGINAL_SEARCH = c.search_jobs
OLD_PLAN_SHA = "50e77159f14b8790b557d24f2ad9da913b4ea99eea5f809577cb7558fa7268db"
ARCHIVE = ROOT / "experiments/astra/storage_review_20261011_v1"


def configs(durations=DURATIONS):
    return [
        dict(id=f"l{i}_e{e}_{loss[0]}", learning_rate=lr, epochs=e, loss=loss)
        for i, lr in enumerate((5e-6, 2e-5, 5e-5))
        for e in durations
        for loss in ("unweighted", "balanced")
    ]


def search_jobs():
    return [
        dict(
            id=f'{h}/{cfg["id"]}/s{s}_f{f}',
            mode="search",
            head=h,
            config=cfg["id"],
            seed=s,
            fold=f,
        )
        for h in c.HEADS
        for cfg in configs(SHORT)
        for s in c.CV_SEEDS
        for f in range(5)
    ]


@contextmanager
def isolated(base=None, expanded=True):
    def no_network(*args, **kwargs):
        raise RuntimeError("Short-schedule study is offline; network disabled")

    with patch.object(c, "BASE", BASE if base is None else Path(base)), patch.object(
        c, "configs", configs if expanded else ORIGINAL_CONFIGS
    ), patch.object(socket.socket, "connect", no_network), patch.object(
        socket, "create_connection", no_network
    ):
        yield


def save(path, value):
    with isolated():
        c.save(path, value)


def source_paths():
    return [
        ROOT / "training/roberta/short/common.py",
        ROOT / "training/roberta/short/prepare.py",
        ROOT / "training/roberta/short/train.py",
        ROOT / "training/roberta/short/audit.py",
        ROOT / "training/roberta/short/run.py",
    ] + [
        ROOT / "tests/training/roberta/short/test_training.py",
        ROOT / "scripts/run_short_schedule.ps1",
        Path(c.__file__),
        Path(resource.__file__),
        ROOT / "training/roberta/train.py",
        ROOT / "training/roberta/audit.py",
        ROOT / "training/roberta/tokenizer.py",
    ]


def hashes(paths):
    return {c.rel(p): c.sha(p) for p in paths}


def development_receipt(kind):
    return (
        BASE / "development" / f"{kind}_{c.signature(hashes(source_paths()))[:16]}.json"
    )


@contextmanager
def locks():

    with ExitStack() as stack:
        for folder in [OLD, ROOT / "experiments/astra/matched_type_pools_20261011_v1"]:
            stack.enter_context(
                resource.file_lock(folder / "launcher.lock", existing=True)
            )
        for folder in [
            "experiments/astra/compact_roberta_probe_20261008_v1",
            "experiments/astra/later_evaluation_20261008_v1/encoders",
        ]:
            stack.enter_context(
                resource.file_lock(ROOT / folder / "inference.lock", existing=True)
            )
        stack.enter_context(resource.file_lock(BASE / "launcher.lock"))
        yield


def arrays(path, values):
    if path.exists():
        with np.load(path, allow_pickle=False) as z:
            c.require(
                set(z.files) == set(values)
                and all(np.array_equal(z[k], v) for k, v in values.items()),
                "Existing arrays differ",
            )
    else:
        with isolated():
            c.arrays(path, values)


def read_data():
    return c.read(BASE / "data/train.json")


def all_jobs(selection):
    with isolated():
        return c.final_jobs(selection)


def original_final_jobs():
    with isolated(OLD, expanded=False):
        return c.final_jobs(c.read(OLD / "selection.json")["families"])


def original_search_jobs():
    with isolated(OLD, expanded=False):
        return ORIGINAL_SEARCH()


def old_paths(job):
    receipt = (
        OLD
        / ("search" if job["mode"] == "search" else "refits")
        / job["id"]
        / "complete.json"
    )
    value = c.read(receipt)
    c.require(
        value["job"] == job
        and value["plan_sha256"] == OLD_PLAN_SHA
        and value["status"] == "complete",
        "Historical job identity mismatch",
    )
    attempt = (OLD / value["attempt"]).resolve()
    c.require(
        attempt.is_relative_to(receipt.parent.resolve()),
        "Historical attempt escaped job",
    )
    return receipt, value, attempt


def verify_old_training(job, archive=None):
    receipt, value, attempt = old_paths(job)
    missing = []
    for name, digest in value["products"].items():
        target = (attempt / name).resolve()
        c.require(target.is_relative_to(attempt), "Historical product escaped attempt")
        if target.is_file():
            c.require(c.sha(target) == digest, "Historical product changed")
        else:
            c.require(
                job["mode"] == "final"
                and name == "model/model.safetensors"
                and archive is not None,
                "Unexpected missing historical product",
            )
            entry = archive.get(c.rel(target))
            c.require(
                entry is not None
                and entry["sha256"] == digest
                and entry["receipt"] == c.rel(receipt)
                and entry["receipt_sha256"] == c.sha(receipt),
                "Missing weight is not the recorded archive",
            )
            missing.append(c.rel(target))
    return value, attempt, missing


def archive_map():
    path = ARCHIVE / "roberta_weights_plan.json"
    plan = c.read(path)
    c.require(
        c.sha(path)
        == "87b54cb681cc0a0892e8c764b0b1058f080e8e19a98ef303bc498ddedee94a34",
        "Approved archive manifest changed",
    )
    copy = c.read(ARCHIVE / "external_copy_verified.json")
    c.require(
        copy["plan_sha256"] == c.sha(path)
        and copy["status"] == "external_copy_verified_originals_retained",
        "Recorded copy identity differs",
    )
    entries = {v["path"]: v for v in plan["files"]}
    c.require(len(entries) == 60, "Archive roster differs")
    c.require(
        len(copy["files"]) == 60
        and {v["path"] for v in copy["files"]} == set(entries)
        and all(
            v["verified"]
            and entries[v["path"]]["sha256"] == v["sha256"]
            and entries[v["path"]]["bytes"] == v["bytes"]
            for v in copy["files"]
        ),
        "Recorded copy verification differs",
    )
    return entries


def compare_logits(receipt_path, job, plan_sha, train_sha, inputs, labels):
    rec = c.read(receipt_path)
    c.require(
        rec["job"] == job
        and rec["plan_sha256"] == plan_sha
        and rec["training_receipt_sha256"] == train_sha
        and rec["rows"] == len(inputs),
        "Inference identity mismatch",
    )
    c.verify(rec["artifacts"])
    folder = receipt_path.parent
    expected = {c.rel(folder / "logits.npy")}
    for start in range(0, len(inputs), 512):
        stop = min(start + 512, len(inputs))
        stem = folder / "chunks" / f"{start:06d}_{stop:06d}"
        expected.update(c.rel(stem.with_suffix(suffix)) for suffix in (".json", ".npy"))
    c.require(set(rec["artifacts"]) == expected, "Inference artifact roster incomplete")
    logits = np.load(folder / "logits.npy", allow_pickle=False)
    c.require(
        logits.shape == (len(inputs), len(labels)) and np.isfinite(logits).all(),
        "Inference shape/finiteness",
    )
    for start in range(0, len(inputs), 512):
        stop = min(start + 512, len(inputs))
        chunk = folder / "chunks" / f"{start:06d}_{stop:06d}"
        saved = c.read(chunk.with_suffix(".json"))
        c.require(
            saved["identity"]
            == dict(
                job=job,
                plan_sha256=plan_sha,
                training_receipt_sha256=train_sha,
                start=start,
                stop=stop,
                input_sha256=c.signature(inputs[start:stop]),
            ),
            "Inference chunk input binding changed",
        )
        c.require(
            saved["sha256"] == c.sha(chunk.with_suffix(".npy")),
            "Inference chunk hash differs",
        )
        c.require(
            np.array_equal(
                logits[start:stop],
                np.load(chunk.with_suffix(".npy"), allow_pickle=False),
            ),
            "Raw chunk assembly differs",
        )
    return logits


def reserve_for_jobs(jobs):

    remaining = sum(
        not (BASE / "refits" / j["id"] / "complete.json").exists() for j in jobs
    )
    return max(15.0, 12.0 + 0.55 * remaining)

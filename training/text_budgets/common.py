from contextlib import contextmanager, ExitStack
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import io
import math
import os
import shutil
import socket
import uuid
from unittest.mock import patch
from training.roberta import common as c
from training.roberta.short import common as short
from training.cpu import common as resource
import numpy as np

ROOT = c.ROOT
BASE = ROOT / "experiments/astra/sq2_stage_b_20261020_v1"
POOLS = ROOT / "experiments/astra/compact_pools_20261008_v1"
PREP = ROOT / "experiments/astra/sq2_stage_b_preparation_20261020_v1"
PROTOCOL = ROOT / "Thesis_Log/astra_result/astra_sq2_stage_b_protocol_20261020_v1.md"
NAMES = [
    f"uniform_{size}_s{seed}"
    for size in ("compact", "1000", "5000")
    for seed in (42, 43, 44)
]
DURATIONS = (1, 2, 5, 10, 15, 20, 60)
PRIMARY = [
    "en_test/hate",
    "en_test/hate_type",
    "larger_dev/target_group",
    "larger_dev/severity",
]
WEIGHT_BYTES_MAX = 500_000_000
EXPORT_BYTES_MAX = 505_000_000
AMENDMENT = (
    ROOT
    / "Thesis_Log/astra_result/astra_sq2_stage_b_numerical_amendment_20261020_v1.md"
)
MODEL_STORES = None


def configs(grid="full"):
    c.require(grid in ("full", "one_rate"), "Unknown prospective search grid")
    return [
        x
        for x in short.configs(DURATIONS)
        if grid == "full" or x["learning_rate"] == 2e-5
    ]


def sources():
    own = [
        ROOT / "training/text_budgets/common.py",
        ROOT / "training/text_budgets/prepare.py",
        ROOT / "training/text_budgets/train.py",
        ROOT / "training/text_budgets/audit.py",
        ROOT / "training/text_budgets/features.py",
        ROOT / "training/text_budgets/linear.py",
        ROOT / "training/text_budgets/run.py",
        ROOT / "training/text_budgets/validate_pilot.py",
    ]
    own += [
        ROOT / "tests/training/text_budgets/test_training.py",
        ROOT / "scripts/run_text_budgets.ps1",
        PROTOCOL,
        AMENDMENT,
    ]
    inherited = [c, short, resource]
    files = own + [Path(x.__file__) for x in inherited]
    files += [
        ROOT / "training/roberta/short/train.py",
        ROOT / "training/roberta/audit.py",
        ROOT / "training/roberta/tokenizer.py",
        ROOT / "training/frozen_features/fit.py",
        ROOT / "training/frozen_features/compact.py",
        ROOT / "training/frozen_features/roberta.py",
        ROOT / "scratch/analysis/astra_later_common_v1.py",
        ROOT / "scratch/analysis/astra_core_api_v2.py",
    ]
    return list(dict.fromkeys(files))


def hashes(paths):
    return {c.rel(p): c.sha(p) for p in paths}


def certificate(kind):
    c.require(kind in ("tests", "pilot"), "Unknown certificate type")
    return (
        BASE
        / "development/certificates"
        / f"{kind}_{c.signature(hashes(sources()))}.json"
    )


@contextmanager
def isolated(grid="full", stores=None, base=None):
    def forbidden(*args, **kwargs):
        raise RuntimeError("Stage B is offline; all network connections disabled")

    global MODEL_STORES
    old_stores = MODEL_STORES
    MODEL_STORES = stores
    try:
        with patch.object(
            c, "BASE", BASE if base is None else Path(base)
        ), patch.object(c, "configs", lambda: configs(grid)), patch.object(
            socket.socket, "connect", forbidden
        ), patch.object(
            socket.socket, "connect_ex", forbidden
        ), patch.object(
            socket, "create_connection", forbidden
        ):
            yield
    finally:
        MODEL_STORES = old_stores


def save(path, value):
    c.save(path, value)


def array(path, value):
    c.array(path, value)


def arrays(path, values):
    if Path(path).exists():
        with np.load(path, allow_pickle=False) as z:
            c.require(
                set(z.files) == set(values)
                and all(np.array_equal(z[k], v) for k, v in values.items()),
                "Saved arrays differ",
            )
    else:
        c.arrays(path, values)


def job(pool, head, config, seed, fold):
    c.require(pool in NAMES and head in c.HEADS, "Unknown pool/head")
    return dict(
        id=f"{pool}/{head}/{config}/s{seed}" + ("" if fold is None else f"_f{fold}"),
        pool=pool,
        head=head,
        config=config,
        seed=seed,
        fold=fold,
        mode="final" if fold is None else "search",
    )


def jobs(pool, grid="full"):
    return [
        job(pool, head, cfg["id"], seed, fold)
        for head in c.HEADS
        for cfg in configs(grid)
        for seed in c.CV_SEEDS
        for fold in range(5)
    ]


def final_jobs(pool, selected):
    values = {}
    for heads in selected.values():
        for head in c.HEADS:
            for seed in c.FINAL_SEEDS:
                j = job(pool, head, heads[head]["selected"]["config"], seed, None)
                values[j["id"]] = j
    return list(values.values())


def data(pool):
    c.require(pool in NAMES, "Unknown pool")
    return c.read(BASE / "data/train" / (pool + ".json"))


def split(data, job):
    spec = data["heads"][job["head"]]
    folds = data["folds"]["42"]
    fitting = [
        x
        for x in spec["rows"]
        if job["fold"] is None or folds[x["feature_row"]] != job["fold"]
    ]
    held = [
        x
        for x in spec["rows"]
        if job["fold"] is not None and folds[x["feature_row"]] == job["fold"]
    ]
    c.require(
        fitting and not {x["cluster"] for x in fitting} & {x["cluster"] for x in held},
        "Empty fit or TRAIN group leakage",
    )
    return spec, fitting, held


def expected_state(data, job):
    _, fit, held = split(data, job)
    total = math.ceil(len(fit) / 16) * c.config_map()[job["config"]]["epochs"]
    if job["mode"] == "search" and not held:
        return "empty_held_fold", 0
    return ("initialization_only" if total == 1 else "trained"), total


def model_path(job, attempt_name, stores=None):
    stores = MODEL_STORES if stores is None else stores
    if stores is None:

        return c.own(c.BASE / "pilot_weights" / job["id"] / attempt_name)
    root = Path(stores["pools"][job["pool"]]).resolve()
    path = (root / job["id"] / attempt_name).resolve()
    c.require(
        path.is_relative_to(root) and path != root, "Export path escapes allocated root"
    )
    return path


def check_models(receipt, job, attempt):
    if job["mode"] != "final":
        c.require(
            not receipt.get("model_products"), "Search must not retain full weights"
        )
        return None
    expected = model_path(job, attempt.name)
    c.require(
        receipt.get("model_directory") == str(expected),
        "Model export allocation changed",
    )
    products = receipt["model_products"]
    c.require(
        {"model.safetensors", "config.json", "study_meta.json"} <= set(products),
        "Incomplete model export",
    )
    for name, digest in products.items():
        path = (expected / name).resolve()
        c.require(
            path.is_relative_to(expected) and c.sha(path) == digest,
            "Model export changed",
        )
    return expected


def receipt(job, plan_sha):
    path = c.job_dir(job) / "complete.json"
    value = c.read(path)
    c.require(
        value["job"] == job
        and value["plan_sha256"] == plan_sha
        and value["status"] == "complete",
        "Job receipt identity",
    )
    attempt = c.own(c.BASE / value["attempt"])
    c.require(
        attempt.is_relative_to(c.job_dir(job).resolve()), "Attempt path escapes job"
    )
    c.require(value["products"], "Empty completion products")
    for name, digest in value["products"].items():
        target = c.own(attempt / name)
        c.require(
            target.is_relative_to(attempt) and c.sha(target) == digest,
            "Saved job product changed",
        )
    check_models(value, job, attempt)
    return value, attempt


@contextmanager
def locks():

    with short.locks(), resource.file_lock(BASE / "launcher.lock"):
        yield


def source_guard():
    path = ROOT / "experiments/astra/sq3_transfer_20261012_v1/execution/plan.json"
    c.require(
        c.sha(path)
        == "ca08393271cfe2461966376daebe45d5d156dcf0acc38f4ce0e9f480f8a2ed5d",
        "Active Qwen plan changed",
    )
    sources = c.read(path)["source_hashes"]
    c.verify(sources)
    return {c.rel(path): c.sha(path), **sources}


def store_guard(stores, remaining=None, initial=False):
    resource.resource_guard(3.5, 12)
    if stores is None:
        return

    c.require(
        set(stores["pools"]) == set(NAMES),
        "Storage allocation does not cover exactly nine pools",
    )
    expected = {root: 0 for root in set(stores["pools"].values())}
    if remaining is None:
        for root in stores["pools"].values():
            expected[root] += 60
    else:
        for j in remaining:
            if not (c.job_dir(j) / "complete.json").exists():
                expected[stores["pools"][j["pool"]]] += 1
    by_drive = {}
    for root, count in expected.items():
        drive = Path(root).anchor
        by_drive[drive] = by_drive.get(drive, 0) + count * EXPORT_BYTES_MAX
    for drive, needed in by_drive.items():
        reserve = stores[
            "initial_drive_reserve_GiB" if initial else "drive_reserve_GiB"
        ][drive]
        c.require(
            shutil.disk_usage(drive).free >= needed + reserve * 2**30,
            f"Insufficient storage reserve on {drive}; preserve all evidence and inspect before resuming",
        )


def validate_stores(stores):
    c.require(set(stores["pools"]) == set(NAMES), "Storage allocation pool coverage")
    allowed_name = "astra_sq2_stage_b_20261020_v1"
    for root in set(stores["pools"].values()):
        path = Path(root)
        c.require(
            path.is_absolute() and path.resolve() == path and path.name == allowed_name,
            "Use a new absolute isolated export directory with the declared study name",
        )
        c.require(
            not path.is_symlink() and not (path.exists() and path.is_junction()),
            "Export root must not redirect to another directory",
        )
        c.require(
            path.anchor in stores["drive_reserve_GiB"]
            and path.anchor in stores["initial_drive_reserve_GiB"],
            "Missing drive reserve",
        )
    c.require(
        all(v >= 12 for v in stores["drive_reserve_GiB"].values()),
        "Unsafe drive reserve",
    )
    c.require(
        all(
            v >= stores["drive_reserve_GiB"][k]
            for k, v in stores["initial_drive_reserve_GiB"].items()
        ),
        "Initial reserve smaller than running reserve",
    )


def deadline_guard():
    c.require(
        datetime.now(timezone.utc) < datetime(2026, 11, 9, 22, 59, tzinfo=timezone.utc),
        "The 9 November experiment stop has been reached. Preserve every output; do not bypass this guard.",
    )


def ac_guard():
    import ctypes

    class Power(ctypes.Structure):
        _fields_ = [
            ("ac", ctypes.c_ubyte),
            ("flags", ctypes.c_ubyte),
            ("percent", ctypes.c_ubyte),
            ("reserved", ctypes.c_ubyte),
            ("life", ctypes.c_ulong),
            ("full_life", ctypes.c_ulong),
        ]

    state = Power()
    c.require(
        bool(ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(state)))
        and state.ac == 1,
        "Connect the full-power charger before local GPU work",
    )


def policy():
    return dict(
        populations=NAMES,
        text_budgets=[147, 1000, 5000],
        draws=[42, 43, 44],
        heads=list(c.HEADS),
        folds="Existing shared TRAIN group assignment42; all five folds.",
        neural_initializations=dict(search=list(c.CV_SEEDS), final=list(c.FINAL_SEEDS)),
        durations=list(DURATIONS),
        methods=[
            "BGE balanced logistic",
            "Frozen RoBERTa mean + same logistic",
            "Fine-tuned RoBERTa",
        ],
        linear="C .1/1/10/100; global cutoff .1..9; method-specific same-pool held-out gate; fold42 primary. BGE fold43/44 preserved as supporting sensitivity.",
        neural="Frozen AdamW/FP16/batch16/schedule; fixed20-epoch balanced/unweighted and selected families; default and selected cutoffs share weights.",
        sparse="Skip empty held folds with explicit receipts; every annotated OOF row exactly once. One planned optimizer step retains explicit initialization-only status and whole-parameter hash equality. Unexpected no-update fits stop.",
        support="TRAIN labels with at least2 positives and2 negatives; no label imputation or pool redraw. No unsupported head occurs in these nine pools; fail if bindings change.",
        primary=dict(
            endpoints=PRIMARY,
            sizes=[147, 1000, 5000],
            contrasts=[
                "tuned_selected_minus_BGE_fold42",
                "frozen_RoBERTa_fold42_minus_BGE_fold42",
            ],
            family_size=24,
            confidence=1 - 0.05 / 24,
            bootstrap_draws=10000,
            random_seed=20261020,
            statistic="Mean per-initialization F1 within pool then equal mean over three pools. Frozen/logistic one fold42 fit per pool; no fabricated neural repetitions.",
            resampling="Shared paired whole-text clusters across every method/pool; conditional on fitted procedures and fixed pools. Pool and neural seed spread separate; exploratory joint pool/text sensitivity, only three draws per budget.",
        ),
        supporting="All controls, every pool/seed/label, exact sets/cardinality, gate and ungated outputs, source and old-core-exclusion; exposure retained; no evaluation winner/threshold selected.",
        evaluation="Existing 19082 unique English identities, core970 and broad18275; text-only inference; separate gold scoring after all completed selections.",
        exposure="Earlier DEV/TEST/HateCheck exposure retained; TRAIN-only selection is not untouched confirmation.",
        no_Qwen=True,
        no_German=True,
        no_extra_annotations=True,
        no_checkpoint_early_stopping=True,
        preservation="Fresh isolated outputs; immutable attempts/chunks; no old model deletion or data edits; all final exports retained in predeclared stores.",
    )

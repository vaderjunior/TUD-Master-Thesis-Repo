"Define the fixed training grid and partitions."

from pathlib import Path
import os
from src.hsrag.paths import PROJECT_ROOT
import json
import math
from functools import lru_cache

PACKAGE = Path(
    os.environ.get(
        "HSRAG_TRAINING_PACKAGE",
        str(PROJECT_ROOT / "deploy/full_training"),
    )
)
HEADS = ("hate", "hate_type", "target_group", "severity")
CV_SEEDS = (0, 1, 2)
FINAL_SEEDS = (10, 11, 12, 13, 14)
THRESHOLDS = tuple(i / 10 for i in range(1, 10))
FAMILIES = ("fixed_unweighted", "fixed_balanced", "tuned")
NAMES = tuple(
    f"uniform_{size}_s{seed}"
    for size in ("compact", "1000", "5000")
    for seed in (42, 43, 44)
)
EXPORT_BYTES_MAX = 505_000_000


@lru_cache(maxsize=1)
def configs():
    return json.loads((PACKAGE / "data/configs.json").read_text(encoding="utf8"))


@lru_cache(maxsize=1)
def config_map():
    return {v["id"]: v for v in configs()}


def protocol():
    return dict(
        fixed_configs=dict(fixed_unweighted="l1_e20_u", fixed_balanced="l1_e20_b")
    )


def data(pool):
    assert pool in NAMES
    return json.loads(
        (PACKAGE / "data/train" / f"{pool}.json").read_text(encoding="utf8")
    )


def job(pool, head, config, seed, fold):
    assert pool in NAMES and head in HEADS and config in config_map()
    assert seed in (FINAL_SEEDS if fold is None else CV_SEEDS)
    assert fold is None or fold in range(5)
    return dict(
        id=f"{pool}/{head}/{config}/s{seed}" + ("" if fold is None else f"_f{fold}"),
        pool=pool,
        head=head,
        config=config,
        seed=seed,
        fold=fold,
        mode="final" if fold is None else "search",
    )


def jobs(pool):
    return [
        job(pool, h, cfg["id"], s, f)
        for h in HEADS
        for cfg in configs()
        for s in CV_SEEDS
        for f in range(5)
    ]


def final_jobs(pool, selected):
    result = {}
    for heads in selected.values():
        for head in HEADS:
            for seed in FINAL_SEEDS:
                j = job(pool, head, heads[head]["selected"]["config"], seed, None)
                result[j["id"]] = j
    return list(result.values())


def split(data, job):
    from training.full_data.common import require

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
    require(
        fitting and not {x["cluster"] for x in fitting} & {x["cluster"] for x in held},
        "Empty fit or TRAIN group leakage",
    )
    return spec, fitting, held


def expected_state(data, job):
    _, fit, held = split(data, job)
    total = math.ceil(len(fit) / 16) * config_map()[job["config"]]["epochs"]
    if job["mode"] == "search" and not held:
        return "empty_held_fold", 0
    return ("initialization_only" if total == 1 else "trained"), total


def allocation(pool, count=8):
    from training.full_data.common import require

    require(count == 8, "The adopted search has exactly eight 10-percent clients.")
    d = data(pool)
    measured = [(j, expected_state(d, j)[1]) for j in jobs(pool)]
    groups = [[] for _ in range(count)]
    costs = [0] * count
    for j, steps in sorted(measured, key=lambda v: (-v[1], v[0]["id"])):
        i = min(range(count), key=lambda i: (costs[i], i))
        groups[i].append(j)
        costs[i] += steps + 20
    require(
        len({j["id"] for group in groups for j in group}) == len(measured),
        "Scheduling duplicate or omission.",
    )
    return groups

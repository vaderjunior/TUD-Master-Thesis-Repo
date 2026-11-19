from datetime import datetime, timezone, timedelta
from pathlib import Path
import math
import numpy as np

POOL = "uniform_compact_s42"
CONFIG = "l0_e2_u"
SEEDS = (10, 11, 12, 13, 14)
FIELDS = ("hate", "hate_type", "target_group", "severity")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def jobs():
    return [
        dict(
            id=f"{POOL}/hate_type/{CONFIG}/s{s}",
            pool=POOL,
            head="hate_type",
            config=CONFIG,
            seed=s,
            fold=None,
            mode="final",
        )
        for s in SEEDS
    ]


def validate_plan(v):
    require(v["version"] == 1 and v["jobs"] == jobs(), "The five-job scope changed.")
    require(
        v["configuration"]
        == dict(id=CONFIG, epochs=2, learning_rate=5e-6, loss="unweighted"),
        "Training settings changed.",
    )
    require(
        v["cutoff"] == 0.5 and len(v["fit_ids"]) == len(set(v["fit_ids"])) == 5,
        "Cutoff or fitting rows changed.",
    )
    require(
        len(v["labels"]) == len(set(v["labels"])) == 7 and v["inference_rows"] == 19082,
        "Label order or prediction scope changed.",
    )
    require(
        v["active_seconds"] == 3600
        and v["project_bytes"] == 250_000_000_000
        and v["additional_reserve_bytes"] == 4_000_000_000
        and v["gpu_percent"] == 80
        and v["Qwen_calls"] == v["new_CV_fits"] == 0,
        "Resource or network scope changed.",
    )
    require(
        v["original_results"] == "/results/sq2_full_20261029_v1"
        and v["results"] == "/results/sq2_type_update_20261103_v1",
        "Result location changed.",
    )
    require(
        v["proposal_sha256"]
        == "ee2e0b0724bc888c0f366d82305a24f40163ddc5874fbba0b95caabc87aa0e2a",
        "The reviewed TRAIN decision changed.",
    )


def separate_root(new, old):
    for path in (Path(new), Path(old)):
        for part in (path, *path.parents):
            require(not part.is_symlink(), "A result path is redirected.")
    new, old = Path(new).resolve(), Path(old).resolve()
    require(
        not new.is_relative_to(old) and not old.is_relative_to(new),
        "Supplementary and original output roots overlap.",
    )


def time_budget(records, stop_utc, now=None):
    now = now or datetime.now(timezone.utc)
    used = 0.0
    for item in records:
        require(
            item is not None and item["status"] in ("complete", "paused"),
            "An unclosed or failed invocation needs review.",
        )
        require(
            math.isfinite(item["seconds"]) and item["seconds"] >= 0,
            "Invalid elapsed time.",
        )
        used += item["seconds"]
    left = 3600 - used
    end = min(now + timedelta(seconds=left), datetime.fromisoformat(stop_utc))
    require(left > 0 and end > now, "The supplementary time budget has ended.")
    return left, end


def capacity(inputs, outputs, free, image_bound, other_bound, reserve=4_000_000_000):
    require(
        all(
            type(x) is int and x >= 0
            for x in (inputs, outputs, free, image_bound, other_bound, reserve)
        ),
        "Invalid storage measurement.",
    )
    require(
        inputs <= 5_000_000_000 and outputs + reserve <= 200_000_000_000,
        "Input or result storage would exceed the existing allowance.",
    )
    require(
        0 < image_bound <= 30_000_000_000 and other_bound <= 15_000_000_000,
        "Image or other-project allowance exceeded.",
    )
    require(free >= 12_000_000_000 + reserve, "Insufficient host free-space reserve.")
    require(
        inputs + outputs + reserve + image_bound + other_bound <= 250_000_000_000,
        "The 250 GB total project limit would be exceeded.",
    )
    return dict(
        inputs=inputs,
        outputs=outputs,
        reserve=reserve,
        free=free,
        measured_project_forecast=inputs
        + outputs
        + reserve
        + image_bound
        + other_bound,
        conservative_project_forecast=outputs + reserve + 50_000_000_000,
        project_limit=250_000_000_000,
    )


def updated_model(meta, job, fit_ids):
    require(job in jobs() and meta["job"] == job, "Unexpected final model.")
    require(
        meta["fit_ids"] == fit_ids and meta["held_ids"] == [], "Fitting rows changed."
    )
    require(
        meta["configuration"]
        == dict(id=CONFIG, epochs=2, learning_rate=5e-6, loss="unweighted"),
        "Final model settings changed.",
    )
    require(
        not meta["initialization_only"]
        and not meta["pilot_only"]
        and meta["planned_steps"] == meta["attempted_steps"] == len(meta["steps"]) == 2,
        "Wrong training horizon or initialization-only model.",
    )
    require(
        any(not x["skipped"] and x["lr"] > 0 for x in meta["steps"])
        and len(meta["changes"]) == 3
        and all(math.isfinite(x) and x > 0 for x in meta["changes"].values()),
        "No actual encoder/head update. Keep this attempt and review it.",
    )
    require(
        meta["export_check"]["clean_reload"]
        and math.isfinite(meta["export_check"]["max_abs_logit_difference"])
        and meta["export_check"]["max_abs_logit_difference"] <= 1e-4,
        "Model reload check failed.",
    )


def replace_type(original, probability, cutoff=0.5):
    expected = set(FIELDS) | {x + "_ungated" for x in FIELDS}
    require(set(original) == expected, "Original prediction fields differ.")
    n = len(probability)
    require(
        n > 0
        and probability.shape == (n, 7)
        and np.isfinite(probability).all()
        and ((probability >= 0) & (probability <= 1)).all(),
        "Invalid type probabilities.",
    )
    for field in FIELDS:
        width = dict(hate=2, hate_type=7, target_group=7, severity=3)[field]
        for key in (field, field + "_ungated"):
            v = original[key]
            require(
                v.shape == (n, width) and ((v == 0) | (v == 1)).all(),
                "Original prediction shape/values differ.",
            )
    require(
        (original["hate"].sum(axis=1) == 1).all()
        and np.array_equal(original["hate"], original["hate_ungated"]),
        "Invalid unchanged hate gate.",
    )
    require(cutoff == 0.5, "The supplementary cutoff changed.")
    output = {name: value.copy() for name, value in original.items()}
    raw = (probability >= cutoff).astype(np.uint8)
    gated = raw.copy()
    gated[original["hate"][:, 1] == 0] = 0
    output["hate_type"] = gated
    output["hate_type_ungated"] = raw
    return output

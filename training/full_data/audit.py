from training.recovery.update_rules import verify as _verify_update
import math
from pathlib import Path
import numpy as np
from training.full_data import plan as b
from training.full_data import common as c
from training.full_data import reference as reference


def metadata(job, data, attempt, *, model_directory=None, pilot=False):
    m = c.read(attempt / "training.json")
    spec, fit, held = b.split(data, job)
    cfg = c.config_map()[job["config"]]
    total = math.ceil(len(fit) / 16) * cfg["epochs"]
    steps = m["steps"]
    c.require(
        m["job"] == job and m["configuration"] == cfg and m["pilot_only"] == pilot,
        "Training identity differs",
    )
    c.require(
        m["fit_ids"] == [r["id"] for r in fit]
        and m["held_ids"] == [r["id"] for r in held],
        "TRAIN row order differs",
    )
    c.require(
        m["planned_steps"] == total
        and m["attempted_steps"] == len(steps) == (min(8, total) if pilot else total),
        "Training horizon differs",
    )
    c.require(
        [s["step"] for s in steps] == list(range(1, len(steps) + 1)),
        "Step ordering differs",
    )
    c.require(
        m["successful_steps"] + m["skipped_steps"] == len(steps)
        and sum(s["skipped"] for s in steps) == m["skipped_steps"],
        "AMP step accounting differs",
    )
    c.require(
        m["class_weights"]
        == c.class_weights(
            job["head"],
            [v["target"] for v in fit],
            len(spec["labels"]),
            cfg["loss"] == "balanced",
        ),
        "Weights are not fitting-fold-only",
    )
    reference.require_learning_rate_trace(m)
    expected_epochs = math.ceil(len(steps) / math.ceil(len(fit) / 16))
    c.require(
        len(m["epochs"]) == expected_epochs
        and [v["epoch"] for v in m["epochs"]] == list(range(1, expected_epochs + 1)),
        "Epoch trace differs",
    )
    for e in m["epochs"]:
        trace = [v for v in steps if v["epoch"] <= e["epoch"]]
        c.require(
            e["attempted_steps"] == len(trace)
            and e["successful_steps"] == sum(not v["skipped"] for v in trace),
            "Epoch step accounting differs",
        )
        c.require(math.isfinite(e["training_objective"]), "Nonfinite objective")
    _verify_update(
        m,
        (
            c.read(attempt / "parameter_equivalence.json")
            if (attempt / "parameter_equivalence.json").exists()
            else None
        ),
    )
    if held:
        z = np.load(attempt / "held_logits.npy", allow_pickle=False)
        c.require(
            z.dtype == np.float32
            and z.shape == (len(held), len(spec["labels"]))
            and np.isfinite(z).all(),
            "Held logits invalid",
        )
        paths = list((attempt / "epoch_logits").glob("*.npy"))
        c.require(len(paths) == expected_epochs, "Missing/extra per-epoch logits")
        y = np.asarray([r["target"] for r in held])
        errors = []
        for e in m["epochs"]:
            q = np.load(
                attempt / "epoch_logits" / f"{e['epoch']:03d}.npy", allow_pickle=False
            )
            c.require(
                q.dtype == np.float32 and q.shape == z.shape and np.isfinite(q).all(),
                "Epoch logits invalid",
            )
            q64 = q.astype(float)
            if job["head"] in ("hate_type", "target_group"):
                loss = float(np.mean(np.logaddexp(0, q64) - y * q64))
            else:
                mx = q64.max(1)
                lse = mx + np.log(np.exp(q64 - mx[:, None]).sum(1))
                loss = float(np.mean(lse - q64[np.arange(len(y)), y]))
            error = abs(loss - e["held_unweighted_loss"])
            c.require(error < 2e-5, "Saved epoch loss disagrees with raw logits")
            errors.append(error)
        c.require(
            np.array_equal(
                z,
                np.load(
                    attempt / "epoch_logits" / f"{expected_epochs:03d}.npy",
                    allow_pickle=False,
                ),
            ),
            "Terminal logits differ",
        )
    if model_directory is not None:
        path = Path(model_directory)
        c.require(
            c.read(path / "study_meta.json")
            == dict(
                job=job,
                labels=spec["labels"],
                kind=spec["kind"],
                config=cfg,
                pilot_only=pilot,
            ),
            "Export identity differs",
        )
        c.require(
            m["model_directory"] == str(path)
            and m["export_check"]["clean_reload"]
            and m["export_check"]["max_abs_logit_difference"] <= 1e-4,
            "CPU export check failed",
        )
    return m


def place_oof(destination, data, job, logits):
    spec, fit, held = b.split(data, job)
    c.require(held and job["seed"] in c.CV_SEEDS, "Invalid OOF job")
    indices = [
        i
        for i, r in enumerate(spec["rows"])
        if data["folds"]["42"][r["feature_row"]] == job["fold"]
    ]
    c.require(
        logits.shape == (len(indices), len(spec["labels"]))
        and np.isfinite(logits).all(),
        "OOF dimensions/values differ",
    )
    seed_index = list(c.CV_SEEDS).index(job["seed"])
    c.require(np.isnan(destination[seed_index, indices]).all(), "OOF row filled twice")
    destination[seed_index, indices] = c.probabilities(job["head"], logits)


def decoded(probs, heads, policy):
    c.require(policy in ("default", "selected"), "Unknown cutoff policy")
    output = {}
    gate = None
    for head in c.HEADS:
        cutoff = (
            None
            if head == "severity"
            else 0.5 if policy == "default" else heads[head]["selected"]["threshold"]
        )
        raw = c.decisions(head, probs[head], cutoff)
        given = raw.copy()
        if head == "hate":
            gate = raw[:, 1].astype(bool)
        else:
            given[~gate] = 0
        if head == "hate":
            v = probs[head][:, 1] > cutoff
            expected = np.column_stack((~v, v))
        elif head == "severity":
            expected = np.eye(probs[head].shape[1], dtype=bool)[probs[head].argmax(1)]
        else:
            expected = probs[head] >= cutoff
        if head != "hate":
            expected[~gate] = False
        c.require(np.array_equal(given, expected), "Independent decoding differs")
        output[head] = given.astype(np.uint8)
        output[head + "_ungated"] = raw.astype(np.uint8)
    return output

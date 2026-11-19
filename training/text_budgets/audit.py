from collections import Counter
import math
from pathlib import Path
import numpy as np
from scipy.special import expit, softmax
from training.text_budgets import common as b
from training.roberta import audit as reference

c = b.c


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
    c.require(
        m["initialization_only"] == (total == 1),
        "Unexpected initialization-only status",
    )
    if m["initialization_only"]:
        before = m["initial_parameter_fingerprint"]
        after = m["final_parameter_fingerprint"]
        c.require(
            len(before["zero_canonical_sha256"]) == 64
            and before["zero_canonical_sha256"] == after["zero_canonical_sha256"]
            and all(v == 0 for v in m["changes"].values()),
            "Initialization numerical parameters changed",
        )
        c.require(
            c.read(attempt / "parameter_equivalence.json")
            == dict(before=before, after=after, changes=m["changes"]),
            "Parameter fingerprint receipt differs",
        )
        c.require(
            all(v["lr"] == 0 for v in steps),
            "Initialization-only recipe has positive rate",
        )
    else:
        c.require(
            any(not v["skipped"] and v["lr"] > 0 for v in steps)
            and all(v > 0 for v in m["changes"].values()),
            "Unreported no-update fit",
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


def load_search(pool, grid, plan_sha):
    data = b.data(pool)
    oof = {
        h: {
            cfg["id"]: np.full(
                (
                    len(c.CV_SEEDS),
                    len(data["heads"][h]["rows"]),
                    len(data["heads"][h]["labels"]),
                ),
                np.nan,
            )
            for cfg in b.configs(grid)
        }
        for h in c.HEADS
    }
    receipts = {}
    counts = Counter()
    steps = 0
    for job in b.jobs(pool, grid):
        r, attempt = b.receipt(job, plan_sha)
        state, total = b.expected_state(data, job)
        c.require(r["state"] == state, "Job classification differs")
        if state == "empty_held_fold":
            spec, fit, held = b.split(data, job)
            m = c.read(attempt / "empty_held_fold.json")
            c.require(
                not held
                and m["job"] == job
                and m["plan_sha256"] == plan_sha
                and m["fit_ids"] == [x["id"] for x in fit]
                and m["held_ids"] == []
                and m["fits"] == m["steps"] == 0,
                "Empty-fold receipt differs",
            )
            c.require(
                not (attempt / "held_logits.npy").exists()
                and not (attempt / "training.json").exists(),
                "Empty fold invented predictions",
            )
        else:
            metadata(job, data, attempt)
            place_oof(
                oof[job["head"]][job["config"]],
                data,
                job,
                np.load(attempt / "held_logits.npy", allow_pickle=False),
            )
        receipts[c.rel(c.job_dir(job) / "complete.json")] = c.sha(
            c.job_dir(job) / "complete.json"
        )
        counts[state] += 1
        steps += total
    c.require(
        all(np.isfinite(v).all() for head in oof.values() for v in head.values()),
        "Missing OOF rows",
    )
    return (
        data,
        oof,
        dict(receipts=receipts, states=dict(counts), attempted_steps=steps),
    )


def select(pool, grid, plan_sha):
    data, oof, proof = load_search(pool, grid, plan_sha)
    selected = c.select(data, oof)
    checks = reference.check_selection(data, oof, selected)
    path = c.BASE / "selections" / f"{pool}.json"
    b.save(
        path,
        dict(
            plan_sha256=plan_sha,
            pool=pool,
            grid=grid,
            families=selected,
            search_proof=proof,
            independent_candidate_checks=checks,
            evaluation_gold_used=False,
        ),
    )
    return selected


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


def final_check(pool, selected, plan_sha):
    data = b.data(pool)
    inputs = c.read(c.BASE / "data/inference.json")
    outputs = {}
    count = 0
    states = Counter()
    for job in b.final_jobs(pool, selected):
        receipt, attempt = b.receipt(job, plan_sha)
        metadata(job, data, attempt, model_directory=receipt["model_directory"])
        expected, _ = b.expected_state(data, job)
        c.require(receipt["state"] == expected, "Final state differs")
        states[expected] += 1
        rp = c.job_dir(job) / "complete.json"
        ip = c.BASE / "inference" / job["id"] / "complete.json"
        b.short.compare_logits(
            ip, job, plan_sha, c.sha(rp), inputs, data["heads"][job["head"]]["labels"]
        )
        outputs.update(b.hashes([rp, ip]))
        outputs.update(c.read(ip)["artifacts"])
        count += 1
    result = dict(
        status="pass",
        pool=pool,
        plan_sha256=plan_sha,
        final_models=count,
        states=dict(states),
        outputs=outputs,
    )
    b.save(c.BASE / "audits" / f"{pool}_final.json", result)
    return result


def assemble(pool, selected, plan_sha):
    data = b.data(pool)
    inputs = c.read(c.BASE / "data/inference.json")
    outputs = {}
    bits = 0
    raw_values = 0
    for family, heads in selected.items():
        for seed in c.FINAL_SEEDS:
            probs = {}
            for head in c.HEADS:
                job = b.job(pool, head, heads[head]["selected"]["config"], seed, None)
                directory = c.BASE / "inference" / job["id"]
                rp = c.job_dir(job) / "complete.json"
                logits = b.short.compare_logits(
                    directory / "complete.json",
                    job,
                    plan_sha,
                    c.sha(rp),
                    inputs,
                    data["heads"][head]["labels"],
                )
                probs[head] = c.probabilities(head, logits)
                independent = (
                    expit(logits.astype(float))
                    if head in ("hate_type", "target_group")
                    else softmax(logits.astype(float), axis=1)
                )
                c.require(
                    np.allclose(probs[head], independent, rtol=0, atol=1e-14),
                    "Independent probability conversion differs",
                )
                raw_values += logits.size
            for cutoff_policy in ("selected", "default"):
                values = decoded(probs, heads, cutoff_policy)
                path = (
                    c.BASE
                    / "predictions"
                    / pool
                    / f"{family}_{cutoff_policy}_s{seed}.npz"
                )
                b.arrays(path, values)
                outputs.update(b.hashes([path]))
                bits += sum(v.size for v in values.values())
    result = dict(
        status="pass",
        pool=pool,
        plan_sha256=plan_sha,
        variants=6,
        final_seeds=list(c.FINAL_SEEDS),
        text_predictions=len(inputs) * 30,
        decision_bits_checked=bits,
        logit_values_checked=raw_values,
        outputs=outputs,
    )
    b.save(c.BASE / "audits" / f"{pool}_assembly.json", result)
    return result

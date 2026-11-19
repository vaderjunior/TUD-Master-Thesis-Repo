import math
import numpy as np
from training.roberta import common as c


def load_oof(data, plan_sha):
    out = {
        h: {
            cfg: np.full(
                (3, len(data["heads"][h]["rows"]), len(data["heads"][h]["labels"])),
                np.nan,
            )
            for cfg in c.config_map()
        }
        for h in c.HEADS
    }
    receipts = {}
    checked_steps = 0
    for job in c.search_jobs():
        path = c.job_dir(job) / "complete.json"
        receipt, attempt = c.verify_receipt(path, job, plan_sha)
        meta = c.read(attempt / "training.json")
        spec = data["heads"][job["head"]]
        rows = spec["rows"]
        folds = data["folds"]["42"]
        held = [r for r in rows if folds[r["feature_row"]] == job["fold"]]
        fit = [r for r in rows if folds[r["feature_row"]] != job["fold"]]
        expected_steps = (
            math.ceil(len(fit) / 16) * c.config_map()[job["config"]]["epochs"]
        )
        c.require(
            meta["job"] == job
            and meta["pilot_only"] is False
            and meta["attempted_steps"] == expected_steps,
            "Wrong search training identity/duration",
        )
        c.require(
            meta["configuration"] == c.config_map()[job["config"]],
            "Search hyperparameters differ from job",
        )
        c.require(
            meta["fit_ids"] == [r["id"] for r in fit]
            and meta["held_ids"] == [r["id"] for r in held],
            "OOF membership mismatch",
        )
        c.require(
            meta["successful_steps"] > 0
            and meta["successful_steps"] + meta["skipped_steps"] == expected_steps,
            "Optimizer-step accounting differs",
        )
        c.require(
            sum(x["skipped"] for x in meta["steps"]) == meta["skipped_steps"],
            "Skipped-step audit mismatch",
        )
        weights = c.class_weights(
            job["head"],
            [r["target"] for r in fit],
            len(spec["labels"]),
            c.config_map()[job["config"]]["loss"] == "balanced",
        )
        c.require(meta["class_weights"] == weights, "Loss used different class support")
        require_learning_rate_trace(meta)
        c.require(
            all(float(v) > 0 for v in meta["changes"].values()),
            "No encoder/classifier change",
        )
        logits = np.load(attempt / "held_logits.npy", allow_pickle=False)
        c.require(
            logits.shape == (len(held), len(spec["labels"]))
            and np.isfinite(logits).all(),
            "Held logits shape/finite check failed",
        )
        indices = [
            i for i, r in enumerate(rows) if folds[r["feature_row"]] == job["fold"]
        ]
        c.require(
            np.isnan(out[job["head"]][job["config"]][job["seed"], indices]).all(),
            "Held row filled twice",
        )
        out[job["head"]][job["config"]][job["seed"], indices] = c.probabilities(
            job["head"], logits
        )
        receipts[c.rel(path)] = c.sha(path)
        checked_steps += expected_steps
    c.require(
        all(np.isfinite(x).all() for v in out.values() for x in v.values()),
        "Missing OOF predictions",
    )
    return out, receipts, checked_steps


def require_learning_rate_trace(meta):
    total = meta["planned_steps"]
    warm = math.ceil(0.1 * total)
    successful = 0
    for step in meta["steps"]:
        factor = (
            successful / max(1, warm)
            if successful < warm
            else max(0.0, (total - successful) / max(1, total - warm))
        )
        expected = meta["configuration"]["learning_rate"] * factor
        c.require(
            abs(step["lr"] - expected) < 1e-14,
            "Learning-rate schedule differs from frozen horizon",
        )
        c.require(
            step["skipped"] == (step["scale_after"] < step["scale_before"]),
            "AMP skip trace mismatch",
        )
        if not step["skipped"]:
            successful += 1
    c.require(successful == meta["successful_steps"], "Successful-step total differs")


def check_selection(data, oof, selection):

    from sklearn.metrics import f1_score

    checked = 0
    for family, heads in selection.items():
        gates = None
        for head in c.HEADS:
            spec = data["heads"][head]
            y = np.asarray([r["target"] for r in spec["rows"]])
            labels = spec["labels"]
            target = (
                y
                if head in ("hate_type", "target_group")
                else np.asarray([[int(i == v) for i in range(len(labels))] for v in y])
            )
            supported = [
                j
                for j in range(target.shape[1])
                if target[:, j].sum() >= 2 and (1 - target[:, j]).sum() >= 2
            ]
            c.require(
                supported == heads[head]["selection_indices"],
                "Selection-support mismatch",
            )
            allowed = (
                [c.protocol()["fixed_configs"][family]]
                if family != "tuned"
                else list(c.config_map())
            )
            expected = {
                (cfg, t)
                for cfg in allowed
                for t in ((None,) if head == "severity" else c.THRESHOLDS)
            }
            c.require(
                {(v["config"], v["threshold"]) for v in heads[head]["candidates"]}
                == expected
                and len(heads[head]["candidates"]) == len(expected),
                "Candidate grid incomplete",
            )
            for candidate in heads[head]["candidates"]:
                values = []
                for seed in c.CV_SEEDS:
                    prob = oof[head][candidate["config"]][seed]
                    if head == "hate":
                        z = (prob[:, 1] > candidate["threshold"]).astype(int)
                        pred = np.column_stack((1 - z, z))
                    elif head == "severity":
                        z = prob.argmax(1)
                        pred = np.asarray(
                            [[int(j == v) for j in range(len(labels))] for v in z]
                        )
                    else:
                        pred = (prob >= candidate["threshold"]).astype(int)
                    if head != "hate":
                        pred[~gates[seed, [r["feature_row"] for r in spec["rows"]]]] = 0
                    values.append(
                        float(
                            np.mean(
                                [
                                    f1_score(target[:, j], pred[:, j], zero_division=0)
                                    for j in supported
                                ]
                            )
                        )
                    )
                c.require(
                    np.allclose(candidate["seed_scores"], values, rtol=0, atol=1e-12)
                    and abs(candidate["mean_score"] - np.mean(values)) < 1e-12,
                    "Independent TRAIN candidate score failed",
                )
                checked += 1

            def ranking(v):
                cfg = c.config_map()[v["config"]]
                t = 0.5 if v["threshold"] is None else v["threshold"]
                return (
                    -v["mean_score"],
                    cfg["epochs"],
                    cfg["learning_rate"],
                    int(cfg["loss"] == "balanced"),
                    abs(t - 0.5),
                    -t,
                )

            c.require(
                sorted(heads[head]["candidates"], key=ranking)[0]
                == heads[head]["selected"],
                "Independent selection/tie check failed",
            )
            if head == "hate":
                selected = heads[head]["selected"]
                gates = oof[head][selected["config"]][:, :, 1] > selected["threshold"]
    return checked


def final_check(data, selection, plan_sha):
    checked = 0
    receipts = {}
    for job in c.final_jobs(selection):
        path = c.job_dir(job) / "complete.json"
        receipt, attempt = c.verify_receipt(path, job, plan_sha)
        info = c.read(attempt / "training.json")
        spec = data["heads"][job["head"]]
        c.require(
            info["job"] == job
            and info["configuration"] == c.config_map()[job["config"]],
            "Final hyperparameters/job differ",
        )
        c.require(
            {"model/model.safetensors", "model/config.json", "model/study_meta.json"}
            <= set(receipt["products"]),
            "Final checkpoint files missing from receipt",
        )
        model_meta = c.read(attempt / "model/study_meta.json")
        c.require(
            model_meta
            == dict(
                job=job,
                labels=spec["labels"],
                kind=spec["kind"],
                config=c.config_map()[job["config"]],
                pilot_only=False,
            ),
            "Exported model metadata differs",
        )
        c.require(
            info["fit_ids"] == [r["id"] for r in spec["rows"]]
            and not info["held_ids"]
            and not info["pilot_only"],
            "Final refit membership mismatch",
        )
        expected = (
            math.ceil(len(spec["rows"]) / 16) * c.config_map()[job["config"]]["epochs"]
        )
        c.require(
            info["attempted_steps"] == expected and info["successful_steps"] > 0,
            "Final duration mismatch",
        )
        expected_weights = c.class_weights(
            job["head"],
            [r["target"] for r in spec["rows"]],
            len(spec["labels"]),
            c.config_map()[job["config"]]["loss"] == "balanced",
        )
        c.require(info["class_weights"] == expected_weights, "Final weighting mismatch")
        require_learning_rate_trace(info)
        c.require(
            info["export_check"]["clean_reload"]
            and info["export_check"]["max_abs_logit_difference"] <= 1e-4,
            "Final export not independently reloaded",
        )
        receipts[c.rel(path)] = c.sha(path)
        checked += 1
    return dict(final_checkpoints=checked, receipts=receipts)


def assemble(selection, data, plan_sha, inputs):
    from scipy.special import expit, softmax

    artifacts = {}
    checked = 0
    for family, heads in selection.items():
        for seed in c.FINAL_SEEDS:
            probs = {}
            independent = {}
            for head in c.HEADS:
                cfg = heads[head]["selected"]["config"]
                job = dict(
                    id=f"{head}/{cfg}/s{seed}",
                    mode="final",
                    head=head,
                    config=cfg,
                    seed=seed,
                    fold=None,
                )
                directory = c.BASE / "inference" / job["id"]
                receipt = c.read(directory / "complete.json")
                c.require(
                    receipt["job"] == job
                    and receipt["plan_sha256"] == plan_sha
                    and receipt["rows"] == len(inputs),
                    "Wrong inference identity",
                )
                c.require(
                    receipt["training_receipt_sha256"]
                    == c.sha(c.job_dir(job) / "complete.json"),
                    "Inference checkpoint attribution differs",
                )
                c.verify(receipt["artifacts"])
                logits = np.load(directory / "logits.npy", allow_pickle=False)
                c.require(
                    logits.shape == (len(inputs), len(data["heads"][head]["labels"]))
                    and np.isfinite(logits).all(),
                    "Final logit shape differs",
                )

                for start in range(0, len(inputs), 512):
                    stop = min(start + 512, len(inputs))
                    chunk = directory / "chunks" / f"{start:06d}_{stop:06d}"
                    identity = c.read(chunk.with_suffix(".json"))["identity"]
                    c.require(
                        identity
                        == dict(
                            job=job,
                            plan_sha256=plan_sha,
                            training_receipt_sha256=receipt["training_receipt_sha256"],
                            start=start,
                            stop=stop,
                            input_sha256=c.signature(inputs[start:stop]),
                        ),
                        "Chunk input binding differs",
                    )
                    c.require(
                        np.array_equal(
                            logits[start:stop],
                            np.load(chunk.with_suffix(".npy"), allow_pickle=False),
                        ),
                        "Concatenated logits differ from raw chunk",
                    )
                probs[head] = c.probabilities(head, logits)
                independent[head] = (
                    expit(logits.astype(float))
                    if head in ("hate_type", "target_group")
                    else softmax(logits.astype(float), axis=1)
                )
                c.require(
                    np.allclose(probs[head], independent[head], rtol=0, atol=1e-14),
                    "Independent probability calculation failed",
                )
                artifacts[c.rel(directory / "complete.json")] = c.sha(
                    directory / "complete.json"
                )
            for policy in ("selected", "default"):
                output = {}
                gate = None
                for head in c.HEADS:
                    cutoff = (
                        None
                        if head == "severity"
                        else (
                            0.5
                            if policy == "default"
                            else heads[head]["selected"]["threshold"]
                        )
                    )
                    output[head] = c.decisions(head, probs[head], cutoff, gate).astype(
                        np.uint8
                    )
                    if head == "hate":
                        gate = output[head][:, 1].astype(bool)
                    p = independent[head]
                    if head == "hate":
                        z = p[:, 1] > cutoff
                        expected = np.column_stack((~z, z)).astype(np.uint8)
                    elif head == "severity":
                        expected = np.eye(p.shape[1], dtype=np.uint8)[
                            np.argmax(p, axis=1)
                        ]
                    else:
                        expected = (p >= cutoff).astype(np.uint8)
                    if head != "hate":
                        expected[~gate] = 0
                    c.require(
                        np.array_equal(output[head], expected),
                        "Independent final decision mismatch",
                    )
                    checked += len(inputs)
                path = c.BASE / "predictions" / f"{family}_{policy}_s{seed}.npz"

                if path.exists():
                    with np.load(path, allow_pickle=False) as saved:
                        c.require(
                            set(saved.files) == set(output)
                            and all(
                                np.array_equal(saved[k], v) for k, v in output.items()
                            ),
                            "Existing decoded outputs differ",
                        )
                else:
                    c.arrays(path, output)
                artifacts[c.rel(path)] = c.sha(path)
    return dict(
        independently_checked_head_decisions=checked,
        complete_outputs=checked // 4,
        artifacts=artifacts,
        schema="npz head matrices: hate [false,true]; fine matrices in TRAIN taxonomy order; non-hate clears fine labels; legal not studied",
    )

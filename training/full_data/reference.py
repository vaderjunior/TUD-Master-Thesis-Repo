import math
import numpy as np
from training.full_data import common as c


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

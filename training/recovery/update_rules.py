"Verify initialization-only training outcomes."

import math
import re

WATCHED = {
    "roberta.encoder.layer.0.attention.self.query.weight",
    "roberta.encoder.layer.11.attention.self.query.weight",
    "classifier.out_proj.weight",
}
AMP_STATE = "amp_initialization_only"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def two_step_case(steps, total):
    if total != 2 or len(steps) != 2:
        return False
    if any(type(s["skipped"]) is not bool for s in steps):
        return False
    return sum(s["skipped"] for s in steps) == 1 and not any(
        not s["skipped"] and s["lr"] > 0 for s in steps
    )


def verify(m, equivalence=None):
    total = m["planned_steps"]
    steps = m["steps"]
    require(
        m["initialization_only"] == (total == 1),
        "Planned initialization status changed.",
    )
    require(set(m["changes"]) == WATCHED, "Watched parameter roster changed.")
    positive = sum(not v["skipped"] and v["lr"] > 0 for v in steps)
    if total == 1:
        state = "initialization_only"
    elif two_step_case(steps, total):
        state = AMP_STATE
    else:
        state = "trained"
    if "update_state" in m:
        require(
            m["update_state"] == state and m["accepted_positive_updates"] == positive,
            "Actual update classification differs.",
        )
    else:
        require(state != AMP_STATE, "Missing amended update classification.")
    if state == "trained":
        require(
            positive > 0 and all(v > 0 for v in m["changes"].values()),
            "Unreported no-update fit.",
        )
        return state
    before, after = m["initial_parameter_fingerprint"], m["final_parameter_fingerprint"]
    require(
        isinstance(before, dict) and isinstance(after, dict),
        "Missing whole-model fingerprints.",
    )
    for value in (before, after):
        for name in ("raw_sha256", "zero_canonical_sha256"):
            require(
                isinstance(value.get(name), str)
                and re.fullmatch("[0-9a-f]{64}", value[name]),
                "Invalid model fingerprint.",
            )
        require(
            type(value.get("negative_zero_count")) is int
            and value["negative_zero_count"] >= 0,
            "Invalid signed-zero count.",
        )
    require(
        before["zero_canonical_sha256"] == after["zero_canonical_sha256"]
        and all(v == 0 for v in m["changes"].values()),
        "Initialization numerical parameters changed.",
    )
    require(
        equivalence == dict(before=before, after=after, changes=m["changes"]),
        "Parameter proof differs.",
    )
    if state == "initialization_only":
        require(
            all(v["lr"] == 0 for v in steps),
            "One-step initialization used a positive rate.",
        )
    else:
        require(
            m["successful_steps"] == m["skipped_steps"] == 1
            and m["attempted_steps"] == 2,
            "Different two-step update counts.",
        )
        rate = m["configuration"]["learning_rate"]
        successes = 0
        scale = 65536.0
        require(math.isfinite(rate) and rate > 0, "Invalid configured learning rate.")
        for index, step in enumerate(steps, 1):
            require(
                step["step"] == index and step["scale_before"] == scale,
                "Two-step scaler or step order differs.",
            )
            require(
                step["lr"] == (0.0 if successes == 0 else rate),
                "Two-step scheduler differs.",
            )
            expected = scale / 2 if step["skipped"] else scale
            require(step["scale_after"] == expected, "Unexpected scaler transition.")
            if not step["skipped"]:
                require(
                    math.isfinite(step["grad_norm"]) and step["lr"] == 0,
                    "Accepted update was not finite zero-rate warm-up.",
                )
                successes += 1
            scale = expected
    return state

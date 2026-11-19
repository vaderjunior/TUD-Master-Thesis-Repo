import hashlib
import numpy as np

METRICS = (
    "reported_tokens",
    "ordinary_tokens",
    "repair_tokens",
    "repair_calls",
    "retry_attempts",
    "POST_attempts",
    "request_seconds",
    "retry_wait_seconds",
)
COLUMNS = (
    "reference",
    "feedback_mean",
    "difference",
    "pool_1_difference",
    "pool_2_difference",
    "pool_3_difference",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def paired(x):
    x = np.asarray(x, dtype=np.float64)
    require(
        x.ndim == 4 and x.shape[1:] == (4, 3, 8) and len(x) > 0,
        "Expected input x condition x actual repeat x metric",
    )
    require(
        np.isfinite(x).all() and (x >= 0).all(), "Finite nonnegative recorded resources"
    )
    means = x.mean(axis=2)
    ref = means[:, 0]
    feedback = means[:, 1:].mean(axis=1)
    return np.stack(
        (ref, feedback, feedback - ref, *[means[:, j] - ref for j in (1, 2, 3)]), axis=2
    )


def seed(name):
    return int.from_bytes(
        hashlib.sha256(("A16-part9-resource-v1|" + name).encode()).digest()[:8],
        "little",
    )


def bootstrap(features, clusters, seed_value, draws=10000):
    features = np.asarray(features, dtype=np.float64)
    require(
        len(features) == len(clusters) > 0 and features.shape[1:] == (8, 6),
        "Paired feature shape",
    )
    labels = sorted(set(clusters))
    mapping = {value: i for i, value in enumerate(labels)}
    codes = np.array([mapping[value] for value in clusters])
    sizes = np.bincount(codes)
    sums = np.zeros((len(labels), 8, 6))
    np.add.at(sums, codes, features)
    result = np.empty((draws, 8, 6))
    rng = np.random.Generator(np.random.PCG64(seed_value))
    for start in range(0, draws, 8):
        chosen = rng.integers(0, len(labels), size=(min(8, draws - start), len(labels)))
        result[start : start + len(chosen)] = (
            sums[chosen].sum(axis=1) / sizes[chosen].sum(axis=1)[:, None, None]
        )
    return result


def error_cost(delta_fpr, delta_fnr, prevalence, miss_to_false_alarm_cost=1.0):
    require(
        0 <= prevalence <= 1 and miss_to_false_alarm_cost >= 0,
        "Hypothetical prevalence and relative cost",
    )
    require(
        np.isfinite([delta_fpr, delta_fnr, prevalence, miss_to_false_alarm_cost]).all(),
        "Finite scenario inputs",
    )
    return 1000 * (
        (1 - prevalence) * delta_fpr + prevalence * miss_to_false_alarm_cost * delta_fnr
    )


def break_even_ratio(delta_fpr, delta_fnr, prevalence):
    require(
        delta_fpr > 0 and delta_fnr < 0 and 0 < prevalence < 1,
        "Trade-off with interior prevalence",
    )
    return (1 - prevalence) * delta_fpr / (-prevalence * delta_fnr)

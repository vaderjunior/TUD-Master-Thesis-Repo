import hashlib
import numpy as np

METRICS = [
    "reported_tokens",
    "ordinary_tokens",
    "repair_tokens",
    "repair_calls",
    "retry_attempts",
    "POST_attempts",
    "request_seconds",
    "retry_wait_seconds",
]
COLUMNS = [
    "reference",
    "adaptive",
    "static",
    "random",
    "adaptive-static",
    "adaptive-random",
    "adaptive-reference",
]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def resource_row(r):
    return [
        r["observed_usage"]["total_tokens"],
        r["ordinary_usage"]["total_tokens"],
        r["repair_usage"].get("total_tokens", 0),
        r["accounting"]["repair_logical_calls"],
        r["accounting"]["retry_attempts"],
        r["accounting"]["post_attempts"],
        r["known_request_seconds_sum"],
        r["retry_backoff_seconds_sum"],
    ]


def paired(x):
    x = np.asarray(x, dtype=float)
    need(
        x.ndim == 4 and x.shape[1:] == (10, 3, 8) and len(x) > 0,
        "Expected input/condition/repeat/metric",
    )
    need(np.isfinite(x).all() and (x >= 0).all(), "Invalid resource values")
    m = x.mean(axis=2)
    p = np.stack(
        [
            m[:, 0],
            m[:, [1, 4, 7]].mean(axis=1),
            m[:, [2, 5, 8]].mean(axis=1),
            m[:, [3, 6, 9]].mean(axis=1),
        ],
        axis=2,
    )
    return np.concatenate(
        [
            p,
            p[:, :, 1:2] - p[:, :, 2:3],
            p[:, :, 1:2] - p[:, :, 3:4],
            p[:, :, 1:2] - p[:, :, :1],
        ],
        axis=2,
    )


def seed(name):
    return int.from_bytes(
        hashlib.sha256(("A17-part8-resources-v1|" + name).encode()).digest()[:8],
        "little",
    )


def bootstrap(features, clusters, rng_seed, draws=10000):
    features = np.asarray(features)
    need(
        features.shape == (len(clusters), 8, 7) and len(clusters) > 0,
        "Paired resource feature shape",
    )
    labels = sorted(set(clusters))
    codes = np.array([labels.index(c) for c in clusters])
    sizes = np.bincount(codes)
    sums = np.zeros((len(labels), 8, 7))
    np.add.at(sums, codes, features)
    out = np.empty((draws, 8, 7))
    rng = np.random.Generator(np.random.PCG64(rng_seed))
    for start in range(0, draws, 8):
        selected = rng.integers(
            0, len(labels), size=(min(8, draws - start), len(labels))
        )
        out[start : start + len(selected)] = (
            sums[selected].sum(axis=1) / sizes[selected].sum(axis=1)[:, None, None]
        )
    return out


def unique_jobs(entries, variants):
    groups = {}
    for variant in variants:
        for entry in entries:
            ids = entry["source_ids"]
            need(len(ids) == len(set(ids)), "Duplicate feedback ID")
            key = (variant, tuple(sorted(ids)))
            groups.setdefault(key, []).append(entry["condition"])
    return groups


def error_counts_per_1000(fpr, fnr):
    need(0 <= fpr <= 1 and 0 <= fnr <= 1, "Error rate outside [0,1]")
    return dict(
        false_alarms_per_1000_benign=1000 * fpr, misses_per_1000_hateful=1000 * fnr
    )

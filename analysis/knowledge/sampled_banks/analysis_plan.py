"""Sampled banks analysis plan."""

from copy import deepcopy
import math
from statistics import mean, stdev
from analysis.common import api as a

ROOT = a.ROOT
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
STUDY = ROOT / "experiments/astra/sq1_pool_size_composition_20260926_v1"
CONDITIONS = ("curated_small", "full_reference") + tuple(
    (f"uniform_{size}_s{seed}" for size in ("compact", "1000", "5000") for seed in (42, 43, 44))
)
ENDPOINT_ORDER = (
    "en_dev_eval_sq3_types",
    "en_dev_eval_main",
    "en_dev_eval_targets",
    "en_dev_eval_severity",
    "de_dev_eval",
    "de_legal_dev_eval",
    "en_dev_eval_types",
)
SCOPE_COUNTS = dict(zip(ENDPOINT_ORDER, (467, 299, 56, 45, 150, 175, 103)))
FOCAL = ("en_dev_eval_main", "en_dev_eval_sq3_types", "de_legal_dev_eval")
GROUPS = {name: [name] for name in CONDITIONS[:2]} | {
    "uniform_" + size: [f"uniform_{size}_s{seed}" for seed in (42, 43, 44)]
    for size in ("compact", "1000", "5000")
}
PRIMARY = (("full_reference", "uniform_compact"), ("curated_small", "uniform_compact"))
SECONDARY = (
    ("full_reference", "curated_small"),
    ("uniform_1000", "uniform_compact"),
    ("uniform_5000", "uniform_compact"),
    ("uniform_5000", "uniform_1000"),
    ("full_reference", "uniform_1000"),
    ("full_reference", "uniform_5000"),
)


def forbid(*args, **kwargs):
    raise AssertionError("Offline pool API preparation: no network or credentials")


def aggregate(scores):
    """Score each decoder repeat first; never pool predictions to compute F1."""
    a.require(set(scores) == set(CONDITIONS), "All eleven conditions required")
    for values in scores.values():
        a.require(
            len(values) == 3
            and all(
                (type(x) in (int, float) and math.isfinite(x) and (0 <= x <= 1) for x in values)
            ),
            "Three finite F1 values per pool required",
        )
    per_pool = {
        name: {
            "decoder_values": list(values),
            "mean": mean(values),
            "decoder_sample_sd": stdev(values),
            "decoder_range": [min(values), max(values)],
        }
        for name, values in scores.items()
    }
    groups = {}
    for name, members in GROUPS.items():
        means = [per_pool[member]["mean"] for member in members]
        groups[name] = {
            "mean": mean(means),
            "pool_means": means,
            "between_pool_sample_sd": stdev(means) if len(means) > 1 else None,
            "between_pool_range": [min(means), max(means)] if len(means) > 1 else None,
        }
    return {"per_pool": per_pool, "groups": groups}


def shared_valid(eligible, validity):
    """Single eligible-and-valid mask across all 33 settings in an endpoint."""
    a.require(
        set(validity) == {(c, r) for c in CONDITIONS for r in range(3)}, "All 33 settings required"
    )
    a.require(all((type(x) is bool for x in eligible)), "Eligibility must be boolean")
    a.require(
        all(
            (
                len(v) == len(eligible) and all((type(x) is bool for x in v))
                for v in validity.values()
            )
        ),
        "Wrong validity vector",
    )
    return [ok and all((v[i] for v in validity.values())) for i, ok in enumerate(eligible)]


def analysis_plan(scientific, inherited, scoring_sha256, gold_hashes):
    a.require(scientific["conditions"] == list(CONDITIONS), "Scientific conditions differ")
    e = scientific["evaluation"]
    a.require(
        e["repeats"] == e["votes_per_prediction"] == 3
        and e["focal_rosters"] == list(FOCAL)
        and (e["primary_family"]["contrasts"] == 6)
        and (e["primary_family"]["family_alpha"] == 0.05),
        "Scientific family/repeats differ",
    )
    a.require(set(inherited["endpoints"]) == set(SCOPE_COUNTS), "Original endpoint roster differs")
    endpoints = deepcopy(inherited["endpoints"])
    for name, endpoint in endpoints.items():
        a.require(
            endpoint["rows"] == SCOPE_COUNTS[name] == e["all_rosters"][name]["items"]
            and endpoint["scorable_rows"] == len(endpoint["scorable_ids"])
            and (len(set(endpoint["scorable_ids"])) == endpoint["scorable_rows"]),
            "Gold roster differs",
        )
        expected = (
            endpoint["all_labels"]
            if endpoint["dimension"] == "hate"
            else [
                label for label in endpoint["all_labels"] if endpoint["gold_support"][label] >= 10
            ]
        )
        a.require(
            endpoint["macro_labels"] == expected,
            "Frozen labels do not follow original support policy",
        )
        endpoint["inherited_core_role"] = endpoint["role"]
        endpoint["role"] = (
            "focal"
            if name in FOCAL
            else "sensitivity" if name == "en_dev_eval_types" else "secondary"
        )
    inherited_fields = (
        "bounds",
        "diagnostics",
        "fixed_labels",
        "gate_mapping",
        "invalid_outputs",
        "minimum_full_roster_support",
        "missing_annotations",
    )
    contract = {k: deepcopy(inherited["contract"][k]) for k in inherited_fields}
    contract.update(
        {
            "aggregation": e["aggregation"],
            "mask": "Original gold eligibility AND non-null final prediction in all 33 settings, one shared mask per endpoint for every main table and contrast. Also report each setting on its own valid eligible rows as explicitly separate coverage diagnostics. Never fill missing predictions with negatives.",
            "empty_mask": "If the all-setting eligible mask is empty, scores/intervals on it are undefined (null), not zero. Still report original denominators, coverage and invalid-inclusive full-roster bounds.",
            "bounds_aggregation": "Compute original full-roster low/high F1 separately per setting, then use the same decoder-then-pool averaging as point scores. Difference bounds are mean(left low)-mean(right high), mean(left high)-mean(right low). Never label a bound as a model prediction.",
            "bootstrap": {
                "draws": 10000,
                "seed": 20260926,
                "rng": "numpy.random.default_rng(20260926), reset per endpoint",
                "unit": "eligible common-valid input ID, sampled with replacement to the common-valid roster length",
                "paired": "Identical index draws for all 33 settings in an endpoint; hold all macro labels fixed; rescore each setting then aggregate.",
                "percentile_method": "numpy.quantile(method=linear)",
                "secondary_confidence": 0.95,
                "primary_family_size": 6,
                "family_alpha": 0.05,
                "primary_confidence": 1 - 0.05 / 6,
            },
            "variation": "Report three decoder values, their mean, sample SD (ddof=1) and min/max for every pool. Separately report the three pool means and their sample SD/min/max per uniform size. References have decoder variation only; between-pool SD is undefined. Nine setting/repeats are not nine independent pool draws.",
            "completion": "Score an endpoint only when all its original rows in all 33 settings are transport-complete and every contributing stage is audited. No accuracy-based stopping, changed masks, redrawing pools, dropping difficult items or promoting secondary contrasts.",
            "limits": "Post-selection DEV size/selection-policy study. Bootstrap conditions on these fixed three sampled pools, decoder repeats and saved first-pass retrieval. It does not estimate curation replication, retrieval variation or all sampling/training/selection variance; it does not isolate a causal annotation-coverage effect. No new TEST/benchmark/adaptation claims.",
            "retrieval_qualification": "Retain original accepted first-pass messages for every item; 13 sampled prompts changed on replay, plus 49 trace-only changes. No replay substitution or selective rerun. Both references receive fresh decoder repeats; old classifier responses are historical, not additional repeats.",
        }
    )
    primary = [
        {"endpoint": endpoint, "contrast": left + "_minus_" + right, "left": left, "right": right}
        for endpoint in FOCAL
        for left, right in PRIMARY
    ]
    secondary = [
        {"endpoint": endpoint, "contrast": left + "_minus_" + right, "left": left, "right": right}
        for endpoint in ENDPOINT_ORDER
        for left, right in SECONDARY + (() if endpoint in FOCAL else PRIMARY)
    ]
    return {
        "status": "frozen_before_fresh_classifier_responses",
        "scientific_plan_sha256": a.digest(STUDY / "scientific_plan.json"),
        "inherited_scoring_sha256": scoring_sha256,
        "gold_input_hashes": gold_hashes,
        "conditions": list(CONDITIONS),
        "decoder_repeats": [0, 1, 2],
        "pool_seeds": [42, 43, 44],
        "groups": deepcopy(GROUPS),
        "group_weights": {k: {c: 1 / len(v) for c in v} for k, v in GROUPS.items()},
        "endpoints": endpoints,
        "primary_contrasts": primary,
        "secondary_contrasts": secondary,
        "curve_reporting": "All five group means, all eleven pool means and all 33 setting scores on every complete endpoint, with descriptive 95% paired-item intervals. Fixed 1000/5000 contrasts remain secondary.",
        "contract": contract,
    }

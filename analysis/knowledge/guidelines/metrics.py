"""Guidelines metrics."""

import numpy as np


def validate_gates(gates):
    gates = np.asarray(gates)
    if gates.ndim != 3 or gates.shape[2] != 3 or (not np.isin(gates, [-1, 0, 1]).all()):
        raise ValueError("Expected items x conditions x three repeats, values -1/0/1")
    return gates


def common_mask(gates):
    return (validate_gates(gates) != -1).all(axis=(1, 2))


def scalar_summary(values):
    values = np.asarray(values, dtype=float)
    return {
        "values": values.tolist(),
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
        "range": float(np.ptp(values)),
    }


def counts_and_rates(gates, conditions):
    gates = validate_gates(gates)
    if len(conditions) != gates.shape[1]:
        raise ValueError("Condition count differs")
    n = len(gates)
    if not n:
        raise ValueError("No roster items")
    common = common_mask(gates)
    settings = []
    summaries = {}
    for ci, condition in enumerate(conditions):
        common_rates = []
        own_rates = []
        low = []
        high = []
        for repeat in range(3):
            column = gates[:, ci, repeat]
            valid = column != -1
            fp = int((column == 1).sum())
            null = int((~valid).sum())
            cfp = int((column[common] == 1).sum())
            cn = int(common.sum())
            rate = fp / int(valid.sum()) if valid.any() else None
            cr = cfp / cn if cn else None
            lower = fp / n
            upper = (fp + null) / n
            settings.append(
                {
                    "condition": condition,
                    "repeat": repeat,
                    "roster_items": n,
                    "valid": int(valid.sum()),
                    "null": null,
                    "false_positives": fp,
                    "own_valid_rate": rate,
                    "common_items": cn,
                    "common_false_positives": cfp,
                    "common_rate": cr,
                    "full_roster_bounds": [lower, upper],
                }
            )
            common_rates.append(cr)
            own_rates.append(rate)
            low.append(lower)
            high.append(upper)
        summaries[condition] = {
            "common": scalar_summary(common_rates) if common.any() else None,
            "own_valid": (
                scalar_summary(own_rates) if all((x is not None for x in own_rates)) else None
            ),
            "full_roster_mean_bounds": [float(np.mean(low)), float(np.mean(high))],
        }
    return {
        "roster_items": n,
        "common_items": int(common.sum()),
        "settings": settings,
        "conditions": summaries,
    }


def cluster_structure(template_ids):
    unique = sorted(set(template_ids))
    groups = [np.flatnonzero(np.asarray(template_ids) == template) for template in unique]
    return (unique, groups)


def cluster_rates(common_gates, template_ids, draws):
    gates = validate_gates(common_gates)
    if not len(gates) or (gates < 0).any():
        raise ValueError("Bootstrap needs nonempty common-valid rows")
    if len(template_ids) != len(gates):
        raise ValueError("Template count differs")
    templates, groups = cluster_structure(template_ids)
    draws = np.asarray(draws)
    if (
        draws.ndim != 2
        or draws.shape[1] != len(templates)
        or (not np.issubdtype(draws.dtype, np.integer))
    ):
        raise ValueError("Invalid cluster draw shape/type")
    if (draws < 0).any() or (draws >= len(templates)).any():
        raise ValueError("Invalid cluster index")
    sizes = np.array([len(group) for group in groups], dtype=np.int64)
    sums = np.array([gates[group].sum(axis=(0, 2), dtype=np.int64) for group in groups])
    out = np.empty((len(draws), gates.shape[1]), dtype=np.float64)
    for start in range(0, len(draws), 256):
        selection = draws[start : start + 256]
        numerator = sums[selection].sum(axis=1, dtype=np.int64)
        denominator = sizes[selection].sum(axis=1, dtype=np.int64) * gates.shape[2]
        out[start : start + len(selection)] = numerator / denominator[:, None]
    return out


def interval(values, confidence):
    if not 0 < confidence < 1:
        raise ValueError("Invalid confidence")
    tail = (1 - confidence) / 2
    return np.quantile(values, [tail, 1 - tail], method="linear").tolist()


def contrasts(
    summary, boot_rates, conditions, contrasts, primary_names, family_confidence, global_scope
):
    result = {}
    for name, weights in contrasts.items():
        if len(weights) != 2 or sorted(weights.values()) != [-1, 1]:
            raise ValueError("Only signed paired contrasts")
        left = next((k for k, v in weights.items() if v == 1))
        right = next((k for k, v in weights.items() if v == -1))
        ls = summary["conditions"][left]
        rs = summary["conditions"][right]
        if ls["common"] is None or rs["common"] is None:
            raise ValueError("Unscorable common pool")
        difference = ls["common"]["mean"] - rs["common"]["mean"]
        samples = boot_rates[:, conditions.index(left)] - boot_rates[:, conditions.index(right)]
        ci95 = interval(samples, 0.95)
        family = (
            interval(samples, family_confidence) if global_scope and name in primary_names else None
        )
        lb, ub = ls["full_roster_mean_bounds"]
        rb, ru = rs["full_roster_mean_bounds"]
        result[name] = {
            "left": left,
            "right": right,
            "mean_difference": difference,
            "per_repeat_difference": [
                x - y for x, y in zip(ls["common"]["values"], rs["common"]["values"])
            ],
            "descriptive_95_interval": ci95,
            "primary_family_interval": family,
            "primary_family_confidence": family_confidence if family is not None else None,
            "family_interval_excludes_zero": (
                family[1] < 0 or family[0] > 0 if family is not None else None
            ),
            "full_roster_missing_output_difference_bounds": [lb - ru, ub - rb],
        }
    return result

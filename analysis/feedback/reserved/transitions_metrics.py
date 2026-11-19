from collections import Counter
from fractions import Fraction
import hashlib

TRANSITIONS = ("fix", "regression", "unchanged_correct", "unchanged_error")
CATEGORIES = ("exact", "only_extra", "only_missing", "both_extra_and_missing")


def category(gold, pred):
    extra = set(pred) - set(gold)
    missing = set(gold) - set(pred)
    return (
        "both_extra_and_missing"
        if extra and missing
        else "only_extra" if extra else "only_missing" if missing else "exact"
    )


def transition(t_correct, c_correct):
    return (
        "unchanged_correct"
        if t_correct and c_correct
        else "fix" if t_correct else "regression" if c_correct else "unchanged_error"
    )


def compare(gold, t, c):
    tr, cr = t["result"], c["result"]
    known = gold["hate_type"] is not None
    status = (
        "both_null"
        if tr is None and cr is None
        else (
            "treatment_null"
            if tr is None
            else "control_null" if cr is None else "valid"
        )
    )
    out = {
        "status": status,
        "known_type_gold": known,
        "strict_pair_valid": status == "valid" and t["n_valid"] == c["n_valid"] == 3,
    }
    if status != "valid":
        return out
    out.update(
        binary=transition(tr["hate"] == gold["hate"], cr["hate"] == gold["hate"]),
        gates=[tr["hate"], cr["hate"]],
        types_changed=set(tr["hate_type"]) != set(cr["hate_type"]),
        both_gates_positive=tr["hate"] and cr["hate"],
    )
    if not known:
        return out
    gs = set(gold["hate_type"])
    ts = set(tr["hate_type"])
    cs = set(cr["hate_type"])
    out.update(
        type_categories=[category(gs, ts), category(gs, cs)],
        type_exact=transition(ts == gs, cs == gs),
        correct_labels_added=len((ts - cs) & gs),
        correct_labels_lost=len((cs - ts) & gs),
        extra_labels_added=len((ts - cs) - gs),
        extra_labels_removed=len((cs - ts) - gs),
    )
    return out


def update(ct, v):
    ct["eligible_pairs"] += 1
    ct[v["status"]] += 1
    if v["status"] != "valid":
        return
    ct[v["binary"]] += 1
    ct["strict_valid_pairs"] += v["strict_pair_valid"]
    ct["types_changed"] += v["types_changed"]
    ct["both_gates_positive"] += v["both_gates_positive"]
    ct["unknown_type_pairs"] += not v["known_type_gold"]
    if not v["known_type_gold"]:
        return
    ct["known_type_pairs"] += 1
    ct["type_" + v["type_exact"]] += 1
    ct["category_" + v["type_categories"][1] + "->" + v["type_categories"][0]] += 1
    for name in (
        "correct_labels_added",
        "correct_labels_lost",
        "extra_labels_added",
        "extra_labels_removed",
    ):
        ct[name] += v[name]
    if v["both_gates_positive"]:
        ct["both_positive_type_" + v["type_exact"]] += 1


def finish(ct):
    out = dict(ct)
    n = ct.get("valid", 0)
    nt = ct.get("known_type_pairs", 0)
    out["binary_error_difference"] = (
        float(Fraction(ct.get("regression", 0) - ct.get("fix", 0), n)) if n else None
    )
    out["type_exact_difference"] = (
        float(Fraction(ct.get("type_fix", 0) - ct.get("type_regression", 0), nt))
        if nt
        else None
    )
    return out


def condition_pairs(t, c):
    if len(t) == len(c):
        return list(zip(t, c))
    if len(c) == 1:
        return [(x, c[0]) for x in t]
    raise ValueError("Unsupported unequal pool pairing")


def direction(ct):
    net = ct.get("regression", 0) - ct.get("fix", 0)
    return (
        "fewer_errors"
        if net < 0
        else "more_errors" if net > 0 else "same_average_errors"
    )


def rank(slot, value):
    return hashlib.sha256(
        ("A15-part6-example-v1|" + slot + "|" + value).encode()
    ).hexdigest()


def select_key(slot, keys, used):
    available = set(keys) - set(used)
    return min(available, key=lambda k: rank(slot, k)) if available else None

from itertools import permutations
import hashlib

FIELDS = ("hate", "hate_type", "target_group", "severity", "legal")
HEADS = (
    "binary",
    "known_answer",
    "hate_type",
    "gated_hate_type",
    "target_group",
    "gated_target_group",
    "severity",
    "gated_severity",
)
LABELS = {
    "hate_type": [
        "explicit",
        "incitement",
        "inferiority",
        "irony",
        "stereotyping",
        "threatening",
        "grievance",
    ],
    "target_group": [
        "race",
        "religion",
        "gender",
        "sexual_orientation",
        "disability",
        "national_origin",
        "other",
    ],
    "legal": ["propaganda", "incitement_public_peace", "insult_defamation"],
}
GOLD = {
    "hate_type": "gold_hate_types",
    "target_group": "gold_target_groups",
    "severity": "gold_severity",
    "legal": "gold_legal",
}
TRANSITIONS = ("unchanged_error", "fix", "regression", "unchanged_correct")
PERMUTATIONS = list(permutations(range(3)))
MISSING = -99


def code(value):
    if value is None:
        return (MISSING,) * 5
    out = []
    for field in FIELDS:
        v = value[field]
        if field == "hate":
            out.append(int(v))
        elif field == "severity":
            out.append(-1 if v is None else ("low", "medium", "high").index(v))
        else:
            out.append(sum(1 << LABELS[field].index(label) for label in set(v)))
    return tuple(out)


def eligible(meta, head):
    if head in ("binary", "known_answer"):
        return True
    head = head.removeprefix("gated_")
    return meta[GOLD[head]] is not None and (head == "legal" or meta["gold_hate"])


def correctness(meta, value):
    out = []
    for head in HEADS:
        if not eligible(meta, head):
            out.append(-2)
            continue
        if value is None:
            out.append(-1)
            continue
        if head == "binary":
            v = value["hate"] == meta["gold_hate"]
        elif head == "known_answer":
            v = value["hate"] == meta["gold_hate"]
            for f, g in GOLD.items():
                if eligible(meta, f):
                    v &= (
                        value[f] == meta[g]
                        if f == "severity"
                        else set(value[f]) == set(meta[g])
                    )
        else:
            f = head.removeprefix("gated_")
            v = (
                value[f] == meta[GOLD[f]]
                if f == "severity"
                else set(value[f]) == set(meta[GOLD[f]])
            )
            if head.startswith("gated_"):
                v &= value["hate"]
        out.append(int(v))
    return out


def transition(reference, treatment):
    if reference == -2 or treatment == -2:
        return -2
    if reference == -1 or treatment == -1:
        return -1
    return 2 * reference + treatment


def profile(scores):

    if any(v < 0 for row in scores for v in row):
        return {"status": "unavailable"}
    ref = sum(scores[0])
    trt = [sum(row) for row in scores[1:]]
    n = sum(trt) - 3 * ref
    paired = [
        transition(scores[0][r], scores[c][r]) for c in range(1, 4) for r in range(3)
    ]
    state = (
        "consistent_fix"
        if ref == 0 and sum(trt) == 9
        else (
            "consistent_regression"
            if ref == 3 and sum(trt) == 0
            else (
                "persistently_correct"
                if ref == 3 and sum(trt) == 9
                else "persistently_wrong" if ref == 0 and sum(trt) == 0 else "variable"
            )
        )
    )
    directions = [(x > ref) - (x < ref) for x in trt]
    return dict(
        status="complete",
        reference_correct=ref,
        pool_correct=trt,
        feedback_correct=sum(trt),
        improvement_numerator_over_9=n,
        direction="improved" if n > 0 else "regressed" if n < 0 else "equal_average",
        stability=state,
        fixes=paired.count(1),
        regressions=paired.count(2),
        offsetting=1 in paired and 2 in paired,
        pool_directions=directions,
        opposing_pool_directions=-1 in directions and 1 in directions,
    )


def vote_stats(votes, aggregate):
    cc = [code(v) for v in votes if v is not None]
    n = len(cc)
    distinct = [len({v[i] for v in cc}) for i in range(5)] + [len(set(cc))]
    disagree = [
        sum(
            (x[i] != y[i]) if i < 5 else x != y
            for j, x in enumerate(cc)
            for y in cc[j + 1 :]
        )
        for i in range(6)
    ]
    return dict(
        n_valid=n,
        distinct=distinct,
        disagree_pairs=disagree,
        available_pairs=n * (n - 1) // 2,
        aggregate_not_observed=aggregate is not None and code(aggregate) not in cc,
    )


def rank(slot, key):
    return hashlib.sha256(
        ("A16-part6-cases-v1|" + slot + "|" + key).encode()
    ).hexdigest()

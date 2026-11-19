"""Tests for policy classification."""

from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.metrics import f1_score, jaccard_score
from analysis.knowledge.policy import metrics as m
from analysis.knowledge.policy import reader as r
from analysis.knowledge.policy import common as c
from analysis.knowledge.german_guidelines import reader as old


def test_classification_boundaries():
    tests = []

    def check(name, condition):
        c.require(bool(condition), name)
        tests.append(name)

    def close(a, b):
        return np.allclose(a, b, atol=1e-12, rtol=0)

    rng = np.random.default_rng(4)
    y = rng.integers(0, 2, size=(21, 7)).astype(bool)
    y[:, -1] = False
    p = rng.integers(0, 2, size=(3, 3, 21, 7)).astype(bool)
    counts = m.contributions(y, p).sum(axis=2)
    macro, micro, _ = m.f1(counts)
    for ci in range(3):
        for repeat in range(3):
            check(
                f"sklearn_macro_micro_{ci}_{repeat}",
                close(
                    [macro[ci, repeat], micro[ci, repeat]],
                    [
                        f1_score(y, p[ci, repeat], average="macro", zero_division=0),
                        f1_score(y, p[ci, repeat], average="micro", zero_division=0),
                    ],
                ),
            )
    check("zero_denominator", close(m.f1(np.zeros((3, 3)))[0], 0))
    repeat_counts = np.array([[[1, 0, 0], [0, 0, 1]], [[100, 0, 0], [0, 0, 0]]])
    check(
        "mean_of_repeats_not_pooled",
        abs(m.f1(repeat_counts)[1].mean() - m.f1(repeat_counts.sum(axis=0))[1]) > 0.1,
    )
    keys = ["a", "b", "c", "d"]
    labels = ["x", "y"]
    roster = {
        key: dict(gate=True, hate_types=gold, source=source, text_cluster=cluster)
        for key, gold, source, cluster in [
            ("a", ["x"], "s1", "q"),
            ("b", [], "s2", "q"),
            ("c", ["y"], "s1", "r"),
            ("d", ["x"], "s2", "s"),
        ]
    }
    meta = dict(
        keys=keys,
        dimension="hate_type",
        all_labels=labels,
        macro_labels=labels,
        gold_support={"x": 2, "y": 1},
    )

    def pred(hate=True, types=()):
        return dict(hate=hate, hate_type=list(types), target_group=[], severity=None, legal=[])

    index = {
        (key, name, j): dict(result=pred(types=roster[key]["hate_types"]), n_valid=3, repairs=0)
        for key in keys
        for name in c.CONDITIONS
        for j in range(3)
    }
    index["b", c.CONDITIONS[0], 0]["result"] = pred(False)
    index["d", c.CONDITIONS[2], 2].update(result=None, n_valid=0)
    arrays = m.endpoint_arrays(meta, roster, index, c.CONDITIONS)
    check("one_invalid_removes_all_nine", arrays["common"].tolist() == [True, True, True, False])
    check(
        "closed_gate_empty_gold_not_exact",
        not arrays["correct"][0, 0, 1] and arrays["correct"][0, 1, 1],
    )
    check(
        "open_and_closed_empty_states_differ",
        arrays["states"][0, 0, 1] != arrays["states"][0, 1, 1],
    )
    point = m.point_metrics(meta, arrays)
    check("empty_empty_jaccard_one", close(point["jaccard"], 1))
    check("exact_retains_gate_penalty", close(point["exact"][0], 8 / 9))
    check("partial_jaccard_half", close(jaccard_score([[1, 0]], [[1, 1]], average="samples"), 0.5))
    check(
        "correctness_partition",
        close(
            sum(
                (
                    point["repeatability"][name]
                    for name in [
                        "stable_correct",
                        "stable_wrong",
                        "changing_but_always_wrong",
                        "variable_correctness",
                    ]
                )
            ),
            1,
        ),
    )
    index["b", c.CONDITIONS[1], 1].update(result=None, n_valid=0)
    changed = m.endpoint_arrays(meta, roster, index, c.CONDITIONS)
    features, strata, members = m.cluster_features(meta, changed, roster)
    check(
        "cluster_stratum_before_removal", ("s1", "s2") in members and len(members["s1", "s2"]) == 1
    )
    features, strata, _ = m.cluster_features(meta, arrays, roster)
    check("cross_source_duplicate_cluster_kept", len(features) == 2 and features[:, -1].sum() == 3)
    actual = m.bootstrap_from_features(features, strata, 2, 7, 91, chunk=7)
    manual_rng = np.random.default_rng(91)
    weights = np.zeros((7, len(features)))
    for ids in strata:
        weights[:, ids] = manual_rng.multinomial(len(ids), np.ones(len(ids)) / len(ids), size=7)
    for draw, w in enumerate(weights):
        sampled = np.concatenate(
            [np.repeat(features[i : i + 1], int(n), axis=0) for i, n in enumerate(w)]
        )
        total = sampled.sum(axis=0)
        cs = total[:54].reshape(3, 3, 2, 3)
        ma, mi, _ = m.f1(cs)
        check(
            f"paired_cluster_draw_{draw}",
            close(actual["macro_f1"][draw], ma.mean(axis=1))
            and close(actual["micro_f1_all_labels"][draw], mi.mean(axis=1))
            and close(actual["mean_pairwise_repeat_agreement"][draw], total[54:57] / total[-1]),
        )
    lr = {
        key: dict(row, gate=None, legal=["x"] if key == "a" else []) for key, row in roster.items()
    }
    li = {
        (key, name, j): dict(
            result=dict(pred(False), legal=["x"] if key == "a" else []), n_valid=3, repairs=0
        )
        for key in keys
        for name in c.CONDITIONS
        for j in range(3)
    }
    lm = dict(meta, dimension="legal", gold_support={"x": 1, "y": 0})
    la = m.endpoint_arrays(lm, lr, li, c.CONDITIONS)
    check("legal_independent_of_gate", la["correct"].all())
    lp = m.point_metrics(lm, la)
    check("legal_nonempty_companion", lp["nonempty_gold_n"] == 1 and lp["empty_gold_n"] == 3)
    sr = {
        key: dict(row, severity=["low", "medium", "high", "high"][i])
        for i, (key, row) in enumerate(roster.items())
    }
    si = {
        (key, name, j): dict(
            result=dict(pred(), severity=["high", "low", "high", None][i]), n_valid=3, repairs=0
        )
        for i, key in enumerate(keys)
        for name in c.CONDITIONS
        for j in range(3)
    }
    for name in c.CONDITIONS:
        for j in range(3):
            si["d", name, j]["result"]["hate"] = False
    sm = dict(
        meta,
        dimension="severity",
        all_labels=["low", "medium", "high"],
        macro_labels=["low", "medium", "high"],
        gold_support={"low": 1, "medium": 1, "high": 2},
    )
    sa = m.endpoint_arrays(sm, sr, si, c.CONDITIONS)
    sp = m.point_metrics(sm, sa)
    check(
        "severity_missing_not_fourth_band",
        close(sp["severity"]["mae_emitted"], 1)
        and close(sp["gate_closed_rate"], 0.25)
        and (sp["severity"]["common_emitted_n"] == 3),
    )
    check(
        "severity_signed_errors",
        np.all(sp["severity"]["signed_error_counts_repeats"]["2"] == 1)
        and np.all(sp["severity"]["signed_error_counts_repeats"]["-1"] == 1),
    )
    vote = dict(
        hate=True,
        hate_type=["x", "x"],
        target_group=[],
        legal=[],
        severity="high",
        reasoning="test",
    )
    blank = dict(vote, hate_type=[], severity="low")
    result = old.independent_vote([vote, blank, None], ["low", "medium", "high"])
    check("historical_duplicate_vote_retained", result["result"]["hate_type"] == ["x"])
    check("lower_median_retained", result["result"]["severity"] == "low")
    tie = old.independent_vote([vote, dict(blank, hate=False), None], ["low", "medium", "high"])
    check(
        "tie_closes_gate",
        not tie["result"]["hate"] and tie["uncertain"] and (tie["result"]["severity"] is None),
    )
    check(
        "no_valid_inner_is_missing",
        old.independent_vote([None] * 3, ["low", "medium", "high"])["result"] is None,
    )
    try:
        c.forbidden()
    except RuntimeError:
        check("offline_guard_raises", True)
    else:
        check("offline_guard_raises", False)
    output = dict(
        status="passed",
        tests=tests,
        count=len(tests),
        metric_source_sha256=c.sha(Path(m.__file__)),
        test_source_sha256=c.sha(Path(__file__)),
    )

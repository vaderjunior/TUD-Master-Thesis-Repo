"""Tests for policy robustness."""

from itertools import product
import numpy as np
from analysis.knowledge.policy import robustness as p


def independent(y, pred):
    scores = []
    tp = fp = fn = 0
    for j in range(y.shape[1]):
        a = sum((bool(v[j]) and bool(w[j]) for v, w in zip(y, pred)))
        b = sum((not bool(v[j]) and bool(w[j]) for v, w in zip(y, pred)))
        d = sum((bool(v[j]) and (not bool(w[j])) for v, w in zip(y, pred)))
        scores.append(2 * a / (2 * a + b + d) if 2 * a + b + d else 0)
        tp += a
        fp += b
        fn += d
    return (float(np.mean(scores)), 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0)


def test_robustness_boundaries():
    checks = []
    rng = np.random.default_rng(513)
    for field, L in [("hate", 2), ("severity", 3), ("hate_type", 3), ("legal", 3)]:
        for missing_n in range(4):
            n = 6
            if field in ("hate", "severity"):
                y = np.eye(L, dtype=bool)[rng.integers(L, size=n)]
                pred = np.eye(L, dtype=bool)[rng.integers(L, size=n)]
                options = list(np.eye(L, dtype=bool))
            else:
                y = rng.integers(2, size=(n, L)).astype(bool)
                pred = rng.integers(2, size=(n, L)).astype(bool)
                options = [np.array(v, bool) for v in product([False, True], repeat=L)]
            valid = np.ones(n, bool)
            valid[:missing_n] = False
            g = np.ones(n, bool)
            expected = []
            for fill in product(options, repeat=missing_n):
                trial = pred.copy()
                if missing_n:
                    trial[:missing_n] = fill
                expected.append(independent(y, trial))
            got = p.full_bounds(y, pred, g, valid, field)
            a = np.array(expected)
            np.testing.assert_allclose(
                got["macro_f1"], [a[:, 0].min(), a[:, 0].max()], atol=1e-12, rtol=0
            )
            if "micro_f1" in got:
                np.testing.assert_allclose(
                    got["micro_f1"], [a[:, 1].min(), a[:, 1].max()], atol=1e-12, rtol=0
                )
            checks.append(
                dict(
                    field=field, missing_n=missing_n, exhaustive_fillings=len(expected), passed=True
                )
            )
    y = np.array([[1, 0], [0, 1]], bool)
    pred = np.array([[0, 0], [0, 1]], bool)
    valid = np.array([False, True])
    own = p.score_single(y, pred, np.ones(2, bool), valid, "legal")
    assert own["n"] == 1 and own["exact"] == 1 and (own["macro_f1"] == 0.5)
    checks.append(dict(case="null_not_empty_gold_or_empty_prediction", passed=True))
    y = pred = np.zeros((1, 3), bool)
    g = np.zeros(1, bool)
    valid = np.ones(1, bool)
    assert p.score_single(y, pred, g, valid, "hate_type")["exact"] == 0
    assert p.score_single(y, pred, g, valid, "legal")["exact"] == 1
    checks.append(dict(case="fine_gate_and_independent_legal", passed=True))
    result = dict(passed=True, checks=checks, sources=p.c.bindings([__file__, p.__file__]))

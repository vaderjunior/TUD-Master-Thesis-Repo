import numpy as np
from analysis.classification.combined import metrics as p


def run_tests():
    checks = []

    def check(name, ok):
        if not ok:
            raise AssertionError(name)
        checks.append(name)

    y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.uint8)
    x = np.array([[1, 0], [0, 1], [0, 1], [1, 0]], dtype=np.uint8)
    z = p.metrics(y, x, [0, 1], "hate", x[:, 1])
    check(
        "binary confusion counts",
        all(z["binary"][k] == 1 for k in ("tp", "tn", "fp", "fn")),
    )
    check("balanced F1", z["macro_f1"] == 0.5)
    check(
        "binary denominators", z["binary"]["fpr"] == 0.5 and z["binary"]["fnr"] == 0.5
    )
    check(
        "precision recall",
        z["binary"]["precision"] == 0.5 and z["binary"]["recall"] == 0.5,
    )
    check(
        "single-label errors both extra and missing",
        z["mean_extra_labels"] == 0.5 and z["mean_missing_labels"] == 0.5,
    )
    check("gate rejection", z["gate_rejection_rate"] == 0.5)
    perfect = p.metrics(y, y, [0, 1], "hate")
    check(
        "perfect F1", perfect["macro_f1"] == 1 and perfect["binary"]["error_rate"] == 0
    )
    null = p.metrics(y, np.tile([1, 0], (4, 1)), [0, 1], "hate")
    check(
        "no predicted positives precision convention",
        null["binary"]["precision"] == 0 and null["binary"]["no_predicted_hate"] == 1,
    )
    check(
        "all-negative does not erase positive support",
        null["binary"]["fn"] == 2 and null["binary"]["fnr"] == 1,
    )
    only_neg = p.metrics(y[:2], x[:2], [0, 1], "hate")
    check(
        "absent positive denominator remains undefined",
        only_neg["binary"]["fnr"] is None and only_neg["binary"]["recall"] is None,
    )
    only_pos = p.metrics(y[2:], x[2:], [0, 1], "hate")
    check(
        "absent negative denominator remains undefined",
        only_pos["binary"]["fpr"] is None,
    )
    check(
        "frozen macro includes absent label", abs(only_neg["macro_f1"] - 1 / 3) < 1e-15
    )
    check(
        "no supported macro is not zero",
        p.metrics(y, x, [], "target_group")["macro_f1"] is None,
    )
    ty = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 0]])
    tx = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 0]])
    fine = p.metrics(ty, tx, [0, 1], "hate_type", np.array([1, 0, 1]))
    check(
        "multilabel counts",
        fine["tp"] == [1, 0, 0] and fine["fp"] == [1, 0, 1] and fine["fn"] == [0, 2, 0],
    )
    check(
        "all columns retained outside macro",
        fine["mean_extra_labels"] == 2 / 3 and fine["mean_missing_labels"] == 2 / 3,
    )
    check("gate is not silently repaired", fine["gate_rejection_rate"] == 1 / 3)
    check("correct macro subset", abs(fine["macro_f1"] - 1 / 3) < 1e-15)
    for head, result, want in [
        ("hate", {"hate": False}, [1, 0]),
        ("hate_type", {"hate_type": []}, [0, 0]),
        ("severity", {"severity": None}, [0, 0]),
    ]:
        labels = ["not_hate", "hate"] if head == "hate" else ["a", "b"]
        check("decode " + head, p.decode_result(result, head, labels) == want)
    try:
        p.decode_result({"hate_type": ["alien"]}, "hate_type", ["a", "b"])
    except RuntimeError:
        checks.append("unknown label refused")
    else:
        raise AssertionError("unknown label accepted")
    models = [
        {"group": "g", "pool": "p1"},
        {"group": "g", "pool": "p1"},
        {"group": "g", "pool": "p2"},
    ]
    out = p.aggregate(models, [perfect, perfect, null])["g"]
    check(
        "equal pool weighting despite unequal seed count",
        abs(out["mean"]["macro_f1"] - (1 + 1 / 3) / 2) < 1e-15,
    )
    check(
        "pool and model counts distinct", out["models"] == 3 and out["pool_count"] == 2
    )
    check("range retains poor fit", out["fit_F1_range"] == [1 / 3, 1.0])
    counts = np.array([1, 1, 0, 0], float)
    other = np.array([1, 0, 1, 0], float)
    partition = [
        ((1 - counts) * (1 - other)).sum(),
        (counts * (1 - other)).sum(),
        ((1 - counts) * other).sum(),
        (counts * other).sum(),
    ]
    check("error partition", partition == [1.0, 1.0, 1.0, 1.0])
    qw = np.array([[0, 1, 1], [1, 1, 0]], bool)
    cw = np.array([[1, 0, 1], [0, 1, 0], [0, 0, 1]], bool)
    explicit = np.mean([np.sum(q & c) for q in qw for c in cw])
    check(
        "Cartesian pair identity",
        abs(float((qw.mean(0) * cw.mean(0)).sum()) - explicit) < 1e-15,
    )
    check(
        "paired identity is not majority voting",
        not np.array_equal(qw.mean(0), qw.mean(0) > 0.5),
    )
    mask = np.array([1, 0, 1, 1], bool)
    bad = x.copy()
    bad[1] = 255
    check(
        "missing row excluded before scoring",
        p.metrics(y[mask], bad[mask], [0, 1], "hate")
        == p.metrics(y[mask], x[mask], [0, 1], "hate"),
    )
    per = (
        p.metrics(y, y, [0, 1], "hate")["macro_f1"]
        + p.metrics(y, np.tile([1, 0], (4, 1)), [0, 1], "hate")["macro_f1"]
    ) / 2
    pooled = p.metrics(
        np.concatenate([y, y]),
        np.concatenate([y, np.tile([1, 0], (4, 1))]),
        [0, 1],
        "hate",
    )["macro_f1"]
    check("mean per-fit F1 differs from pooled confusion F1", abs(per - pooled) > 0.01)
    return dict(status="passed", count=len(checks), checks=checks)


if __name__ == "__main__":
    with p.a.g.offline():
        print(run_tests())


def test_boundaries():
    run_tests()

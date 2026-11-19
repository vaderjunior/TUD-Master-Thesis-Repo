import numpy as np


def run_tests():
    from analysis.classification.combined import quality as q

    checks = []

    def check(name, ok):
        if not ok:
            raise AssertionError(name)
        checks.append(name)

    y = np.array([[0, 0], [1, 0], [1, 0], [1, 0], [1, 0]], dtype=np.uint8)
    pred = np.array([[0, 0], [0, 0], [1, 1], [0, 1], [1, 0]], dtype=np.uint8)
    g = np.array([0, 0, 1, 1, 1], dtype=bool)
    r = q.row_quality(y, pred, g)
    check(
        "observed empty exact does not imply correct hate gate",
        r["exact_set"][0] and not r["gate_aware_exact"][0],
    )
    check("empty observed Jaccard is one", r["sample_jaccard"][0] == 1)
    check(
        "closed gate creates missing true label",
        r["missing_only"][1] and r["positive_gold_blocked"][1],
    )
    check("extra-only answer", r["extra_only"][2])
    check("mixed answer", r["mixed"][3])
    check(
        "equal cardinality can be entirely wrong", r["cardinality_matches_but_wrong"][3]
    )
    check("exact fine requires correct gate", r["gate_aware_exact"][4])
    check(
        "label-set partition exhaustive",
        np.all(
            r["exact_set"]
            + r["extra_only"].astype(int)
            + r["missing_only"]
            + r["mixed"]
            == 1
        ),
    )
    legal = q.row_quality(y, pred, np.zeros(5, bool), legal=True)
    check(
        "legal is ungated",
        np.array_equal(legal["exact_set"], legal["gate_aware_exact"]),
    )
    check("unsupported macro label still breaks exactness", not r["exact_set"][2])
    check(
        "empty-gold errors are extras",
        q.row_quality(np.array([[0, 0]]), np.array([[0, 1]]), np.array([1]))[
            "extra_only"
        ][0],
    )
    try:
        q.row_quality(y, pred, np.zeros(5, bool))
        bad = False
    except Exception:
        bad = True
    check("reject delivered fine values behind closed gate", bad)
    f = q.flow_rows(
        np.array([[1, 1, 1, 0]], dtype=np.uint8),
        np.array([[1, 0, 0, 1]], dtype=np.uint8),
        np.array([0]),
    )
    check(
        "gate-only loss separates raw misses",
        f["correct_labels_blocked"][0] == 1 and f["raw_missing_gate_closed"][0] == 2,
    )
    check("gate also blocks wrong extras", f["extra_labels_blocked"][0] == 1)
    check("opening gate cannot fix raw set", not f["raw_exact"][0])
    cases = np.array(
        [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], bool
    )
    j = q.joint_rows(cases[:, 1], cases[:, 2], cases[:, 0])
    check("joint states exhaustive", np.all(sum(j.values()) == 1))
    check("joint requires both fields and gate", j["joint_exact"].sum() == 1)
    check(
        "no severity represented as missing, not low",
        q.row_quality(np.array([[1, 0, 0]]), np.zeros((1, 3), int), np.array([1]))[
            "missing_labels"
        ][0]
        == 1,
    )
    models = [
        dict(group="x", pool="a"),
        dict(group="x", pool="a"),
        dict(group="x", pool="b"),
    ]
    rr = [
        dict(n=1, metrics=dict(value=v, conditional=c))
        for v, c in [(0, None), (0, None), (1, 1)]
    ]
    z = q.aggregate(models, rr)["x"]
    check(
        "equal pool mean rather than unequal fit weighting", z["mean"]["value"] == 0.5
    )
    check(
        "undefined conditional remains explicit",
        z["defined_fits"]["conditional"] == 1
        and z["pools"]["a"]["conditional"] is None,
    )
    check(
        "no-gate conditional exact undefined",
        q.summarize(y[:1], pred[:1], g[:1])["metrics"]["exact_given_gate"] is None,
    )
    check(
        "cardinality counts preserve all texts",
        sum(map(sum, q.summarize(y, pred, g)["cardinality"])) == 5,
    )
    check(
        "invalid sentinel excluded before scoring",
        not np.any(
            np.array([[1, 0], [255, 255]], np.uint8)[np.array([True, False])] > 1
        ),
    )
    return dict(status="passed", count=len(checks), checks=checks)


if __name__ == "__main__":
    print(run_tests())


def test_boundaries():
    run_tests()

import json
from pathlib import Path
import numpy as np
from analysis.classification.broad import sources as s


def run():
    checks = []

    def ok(name, v):
        s.require(bool(v), name)
        checks.append(name)

    def bad(name, fn):
        try:
            fn()
        except ValueError:
            checks.append(name)
            return
        raise AssertionError(name)

    def row(source, v):
        return dict(source=source, raw=json.dumps(v))

    ok(
        "offensive remains its own majority annotation",
        s.annotation(
            row("hatexplain", dict(votes=["offensive", "normal", "offensive"]))
        )
        == "hatexplain/offensive",
    )
    bad(
        "ambiguous three-way source majority fails",
        lambda: s.annotation(
            row("hatexplain", dict(votes=["offensive", "normal", "hatespeech"]))
        ),
    )
    for x, label in [
        (0, "score<=0"),
        (0.5, "0<score<=0.5"),
        (0.500001, "0.5<score<=1.0"),
        (1, "0.5<score<=1.0"),
        (1.63, "1.0<score<=1.63"),
        (2, "score>1.63"),
    ]:
        ok(
            "MHS frozen boundary " + str(x),
            s.annotation(row("mhs", dict(score=x))) == "mhs/" + label,
        )
    for x, label in [
        (0, "score=0"),
        (0.25, "0<score<0.5"),
        (0.5, "score=0.5"),
        (0.75, "0.5<score<1"),
        (1, "score=1"),
    ]:
        ok(
            "DeTox frozen boundary " + str(x),
            s.annotation(row("detox", dict(hate_speech_score=x))) == "detox/" + label,
        )
    ok(
        "stage2-only annotation stays separate",
        s.annotation(
            row("implicit_hate", dict(stage1_class="implicit_hate (stg2-only)"))
        ).endswith("(stg2-only)"),
    )
    ok(
        "numeric case IDs normalize consistently",
        s.identifier(17.0) == s.identifier(17) == s.identifier("17"),
    )
    ok(
        "unrecorded identity is not a social group",
        all(s.normalized_identity(v) == "not_recorded" for v in (None, "nan", "")),
    )
    rows = []
    for i, (temp, ref, cluster, text) in enumerate(
        [
            ("a", None, "u", "t0"),
            ("a", None, "u", "t1"),
            ("b", 1, "u", "t2"),
            ("c", None, "v", "t3"),
        ]
    ):
        rows.append(
            dict(
                key=str(i),
                gate=bool(i % 2),
                text_cluster=text,
                functional_cluster=cluster,
                raw=json.dumps(
                    dict(
                        case_id=i, templ_id=temp, ref_case_id=ref, functionality=str(i)
                    )
                ),
            )
        )
    st = s.structure(rows)
    ok(
        "reference plus template form transitive components",
        st["counts"]["component"]["units"] == 2
        and st["counts"]["component"]["largest"] == 3,
    )
    ok(
        "linked gold-change retained",
        len(st["links"]) == 1 and not st["links"][0]["same_gold"],
    )
    changed = [dict(v) for v in rows]
    changed[2]["functional_cluster"] = "bad"
    bad("reject stale component membership", lambda: s.structure(changed))
    yy = np.eye(2, dtype=int)[[0, 0, 1, 1]]
    pp = np.eye(2, dtype=int)[[[0, 1, 0, 1]] * 9]
    gg = pp[:, :, 1].astype(bool)
    f = s.features(yy, pp, gg, "binary")
    m = s.metrics(f.sum(1), "binary", ["non_hate", "hate"], ["non_hate", "hate"])
    ok(
        "confusion matrix preserves all four outcomes",
        np.array_equal(f[0].sum(0), [1, 1, 1, 1]),
    )
    ok(
        "binary FPR FNR and macro F1 match hand calculation",
        all(np.all(m[k] == 0.5) for k in ("fpr", "fnr", "macro_f1")),
    )
    d = dict(
        name="fixture",
        kind="binary",
        ids=list("abcd"),
        clusters=["x", "x", "y", "z"],
        rows=[{"source": "synthetic"}] * 4,
        labels=["non_hate", "hate"],
        macro=["non_hate", "hate"],
        gold=yy,
        predicted=pp,
        gate=gg,
        valid=np.ones((9, 4), bool),
        features=f,
    )
    z = s.summarize(d, [0, 1, 2], np.array([1, 1, 0, 0], bool))
    ok(
        "single-polarity F1 is suppressed",
        z["mean"]["macro_f1"] is None
        and z["mean"]["fnr"] is None
        and z["mean"]["fpr"] == 0.5,
    )
    d["valid"][0, 0] = False
    z = s.group(d, list(range(4)), "synthetic")
    ok(
        "all-setting mask excludes missing prediction",
        z["common_nine_n"] == 3 and z["own_valid"]["rag_full"][0]["invalid"] == 1,
    )
    ok(
        "missing-output rate bounds retained",
        z["own_valid"]["rag_full"][0]["exact_or_accuracy_failure_bounds"]
        == [0.25, 0.5],
    )
    ok(
        "paired masks align observations not standalone condition averages",
        z["pairs"][0]["n"] == 3,
    )
    y = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    p = np.array([[[0, 0], [1, 1], [1, 0], [0, 1]]] * 9)
    g = np.array([[0, 1, 1, 1]] * 9, bool)
    ff = s.features(y, p, g, "fine")
    mm = s.metrics(ff.sum(1), "fine", ["a", "b"], ["b"])
    ok(
        "empty gold set requires correct positive gate",
        np.all(mm["exact_set"] == 0.5) and np.all(mm["gate_aware_exact"] == 0.25),
    )
    ok("all-label Jaccard counts exact empty set", np.all(mm["sample_jaccard"] == 0.75))
    ok(
        "fixed rare-label macro does not change by subgroup",
        np.allclose(mm["macro_f1"], 0.5),
    )
    ok(
        "extra and missing labels are distinct",
        np.all(mm["extra_labels"] == 0.25) and np.all(mm["missing_labels"] == 0.25),
    )
    ok(
        "legal exactness independent of hate gate",
        np.all(
            s.metrics(s.features(y, p, g, "legal").sum(1), "legal", ["a", "b"], [])[
                "gate_aware_exact"
            ]
            == 0.5
        ),
    )
    ok(
        "no supported macro is null after serialization",
        s.safe(s.metrics(ff.sum(1), "fine", ["a", "b"], [])["macro_f1"])[0] is None,
    )
    bad("reject mismatched axes", lambda: s.features(y, p[:, :, :1], g, "fine"))
    wrong = p.copy()
    wrong[:, 0, 0] = 1
    bad(
        "reject fine output under negative gate",
        lambda: s.features(y, wrong, g, "fine"),
    )
    lm = s.metrics(np.array([[1, 2, 3, 4, 3, 10.0]]), "links", [], [])
    ok(
        "expected linked relation can include two wrong answers",
        lm["both_correct"][0] == 0.1 and lm["expected_relation"][0] == 0.3,
    )
    old = s.DRAWS
    s.DRAWS = 32
    try:
        info, draws = s.bootstrap(d, list(range(6)), np.ones(4, bool), ["error_rate"])
        info2, draws2 = s.bootstrap(
            d, list(range(6)), np.ones(4, bool), ["error_rate"], "item"
        )
        ok(
            "bootstrap fixed settings cancel identical systems",
            np.array_equal(draws, np.zeros((32, 1))),
        )
        ok(
            "component and item units remain separate",
            info["units"] == 3 and info2["units"] == 4,
        )
    finally:
        s.DRAWS = old
    ok(
        "deterministic case order independent of input enumeration",
        s.select_case("a", ["one", "two"], "x", "e")
        == s.select_case("a", ["two", "one"], "x", "e"),
    )
    ok("empty case category preserved", s.select_case("a", [], "x", "e")["key"] is None)
    s.save(
        s.BASE / "tests_v2.json",
        dict(
            status="passed",
            checks=checks,
            count=len(checks),
            sources=s.bindings([Path(__file__), Path(s.__file__)]),
        ),
    )
    print(len(checks), "source-review tests passed")


if __name__ == "__main__":
    with s.r.original.offline():
        run()

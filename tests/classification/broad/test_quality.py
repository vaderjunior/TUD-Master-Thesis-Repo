from pathlib import Path
import numpy as np
from analysis.classification.broad import quality as q


def run():
    checks = []

    def ok(name, condition):
        q.require(bool(condition), name)
        checks.append(name)

    def raises(name, fn):
        try:
            fn()
        except ValueError:
            checks.append(name)
            return
        raise AssertionError(name)

    y = np.array([[0, 0], [1, 0], [1, 0], [1, 0], [0, 0], [0, 1]])
    p = np.array([[[0, 0], [1, 1], [0, 0], [0, 1], [0, 1], [0, 1]]] * 3)
    g = np.array([[False, True, False, True, True, True]] * 3)
    v = q.row_quality(y, p, g)
    ok(
        "known-empty set is exact but wrong negative gate is not complete",
        v["exact_set"][0, 0] and not v["gate_aware_exact"][0, 0],
    )
    ok(
        "extra-only differs from mixed error",
        v["extra_only"][0, 1] and v["mixed"][0, 3],
    )
    ok(
        "blocked positive gold is missing-only",
        v["missing_only"][0, 2] and v["positive_gold_blocked"][0, 2],
    )
    ok("empty gold hallucination remains extra-only", v["extra_only"][0, 4])
    ok(
        "empty-empty Jaccard and sample F1 are one",
        v["sample_jaccard"][0, 0] == v["sample_f1"][0, 0] == 1,
    )
    ok("partial overlap Jaccard is half", v["sample_jaccard"][0, 1] == 0.5)
    ok("partial overlap sample F1 is two thirds", v["sample_f1"][0, 1] == 2 / 3)
    ok("correct positive answer is gate-aware exact", v["gate_aware_exact"][0, 5])
    ok(
        "error partition exhaustive",
        np.all(
            sum(v[k] for k in ("exact_set", "extra_only", "missing_only", "mixed")) == 1
        ),
    )
    legal = q.row_quality(y, p, g, True)
    ok(
        "legal exactness does not require hate gate",
        legal["gate_aware_exact"][0, 0] and legal["positive_gold_blocked"].sum() == 0,
    )
    wrong = p.copy()
    wrong[0, 0, 0] = 1
    raises(
        "reject inconsistent delivered fine label under closed gate",
        lambda: q.row_quality(y, wrong, g),
    )
    ok(
        "legal emission under closed hate gate is permitted",
        q.row_quality(y, wrong, g, True)["extra_only"][0, 0],
    )
    raises("reject wrong axes", lambda: q.row_quality(y, p[:, :, :1], g))
    s = q.summarize(y, p, g, ["a", "b"], ["a"])
    ok("fixed macro columns retained", s["repeats"][0]["macro_f1"] == 0.5)
    ok(
        "all-label detail retained outside macro",
        len(s["per_label_mean"]) == 2
        and not s["per_label_mean"][1]["included_in_original_macro"],
    )
    ok(
        "cardinality inventory is complete",
        np.sum(s["repeats"][0]["cardinality"]) == len(y),
    )
    ok("sample SD is zero for identical repeats", all(x == 0 for x in s["sd"].values()))
    zero = q.summarize(
        np.zeros((2, 2), int),
        np.zeros((3, 2, 2), int),
        np.ones((3, 2), bool),
        ["a", "b"],
        [],
    )
    ok(
        "unsupported macro is null rather than zero",
        zero["repeats"][0]["macro_f1"] is None,
    )
    ok(
        "undefined precision and recall remain null",
        zero["per_label_mean"][0]["precision"] is None
        and zero["per_label_mean"][0]["recall"] is None,
    )
    ok(
        "undefined repetition counts are explicit",
        zero["per_label_mean"][0]["precision_undefined_repeats"] == 3,
    )
    ok(
        "conventional zero-denominator label F1 is flagged",
        zero["repeats"][0]["per_label"][0]["f1"] == 0
        and zero["repeats"][0]["per_label"][0]["f1_denominator_zero"],
    )
    vv = np.array([[1, 1, 0, 1, 1, 1]] * 3, bool)
    bb = q.bounds(v, vv, "gate_aware_exact")
    ok(
        "invalid exactness yields failure bounds not empty guess",
        np.allclose(bb, [[1 / 6, 2 / 6]] * 3),
    )
    yg = np.eye(3, dtype=int)[[0, 1, 2, 0, 2, 1, 0]]
    pg = np.zeros((3, 7, 3), int)
    for j, k in enumerate([None, None, 0, 2, 1, 2, 0]):
        if k is not None:
            pg[:, j, k] = 1
    gg = np.array([[0, 1, 1, 1, 1, 1, 1]] * 3, bool)
    sev = q.detail(yg, pg, gg, ["low", "medium", "high"], "severity")[0]
    ok(
        "severity separates blocked and open-but-missing",
        sev["counts"]["gate_closed"] == sev["counts"]["gate_open_missing"] == 1,
    )
    ok(
        "severity two-level under and over retained",
        sev["counts"]["under_two"] == sev["counts"]["over_two"] == 1,
    )
    ok("severity confusion conserves rows", np.sum(sev["confusion"]) == 7)
    ok(
        "severity conditional distance excludes missing classes",
        sev["conditional_ordinal_mae"] == 1.2,
    )
    raises(
        "severity rejects multiple emitted classes",
        lambda: q.detail(
            yg, np.ones_like(pg), gg, ["low", "medium", "high"], "severity"
        ),
    )
    ld = q.detail(y, p, g, ["a", "b"], "legal")[0]
    ok(
        "legal known-empty denominator is explicit",
        ld["known_empty"] == 2 and ld["empty_gold_flagged_rate"] == 0.5,
    )
    ok(
        "legal missed positives use different denominator",
        ld["positive_gold_unflagged_rate"] == 0.25,
    )
    ok(
        "legal independent detection partitions all observed rows",
        sum(
            ld[k]
            for k in ("any_legal_tp", "any_legal_fp", "any_legal_fn", "any_legal_tn")
        )
        == 6,
    )
    a = np.array([[1, 0, 1], [1, 0, 0], [1, 0, 1]], bool)
    b = np.array([[0, 1, 1], [0, 1, 0], [0, 1, 0]], bool)
    ev = q.event_counts(a, b)
    ok(
        "cross-repeat fixed-minus-lost equals mean score change",
        np.isclose(ev["net"], a.sum(1).mean() - b.sum(1).mean()),
    )
    ok(
        "stable fixes and losses require all repeats",
        ev["stable_fixed"] == 1 and ev["stable_lost"] == 1,
    )
    ids = ["z", "y", "x"]
    chosen = q.choose("test", ids, a, b)
    ok(
        "systematic case selection includes empty shared-error slot",
        chosen[2]["key"] is None,
    )
    ok(
        "systematic case selection invariant to row ordering",
        q.choose("test", ids[::-1], a[:, ::-1], b[:, ::-1]) == chosen,
    )
    old = q.DRAWS
    q.DRAWS = 200
    try:
        profile = np.array([[1.0, 0.0], [0.0, -1.0], [0.0, 0.0]])
        info, draws = q.bootstrap(profile, ["a", "a", "b"], ["x", "y"])
        rng = np.random.default_rng(q.SEED)
        direct = []
        for start in range(0, 200, 64):
            picks = rng.choice([0, 1], (min(64, 200 - start), 2), replace=True)
            for pick in picks:
                selected = [
                    i
                    for cluster in pick
                    for i, k in enumerate([0, 0, 1])
                    if k == cluster
                ]
                direct.append(profile[selected].mean(0))
        ok(
            "cluster bootstrap weights texts within whole sampled components",
            np.allclose(draws, direct, atol=0, rtol=0),
        )
        ok(
            "point estimate is text-weighted not mean cluster mean",
            info["metrics"]["x"]["difference"] == 1 / 3,
        )
        ok(
            "all supporting draws finite and counted",
            draws.shape == (200, 2) and np.isfinite(draws).all(),
        )
        zz, _ = q.bootstrap(np.zeros((3, 1)), ["a", "a", "b"], ["z"])
        ok(
            "identical methods yield zero interval",
            zz["metrics"]["z"]["ci95"] == [0.0, 0.0],
        )
    finally:
        q.DRAWS = old
    ok("production resample count restored", q.DRAWS == 10000)
    q.save(
        q.BASE / "tests.json",
        dict(
            status="passed",
            tests=len(checks),
            checks=checks,
            sources=q.bindings([Path(__file__), Path(q.__file__)]),
        ),
    )
    print(f"{len(checks)} scientific tests passed.", flush=True)


if __name__ == "__main__":
    with q.r.original.offline():
        run()

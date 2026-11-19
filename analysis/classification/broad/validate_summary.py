from pathlib import Path
import numpy as np
from analysis.classification.broad import summary as s

x = s.x
checked = 0
maximum = 0.0


def same(a, b, label):
    global checked, maximum
    if a is None or b is None:
        x.require(a is b, "Missing value differs: " + label)
        checked += 1
        return
    aa = np.asarray(a)
    bb = np.asarray(b)
    x.require(aa.shape == bb.shape, "Shape differs: " + label)
    if aa.dtype.kind in "biuf" and bb.dtype.kind in "biuf":
        difference = (
            float(np.max(np.abs(aa.astype(float) - bb.astype(float))))
            if aa.size
            else 0.0
        )
        maximum = max(maximum, difference)
        x.require(
            np.allclose(aa, bb, atol=2e-12, rtol=2e-12), "Number differs: " + label
        )
    else:
        x.require(np.array_equal(aa, bb), "Value differs: " + label)
    checked += aa.size


def score(y, p, columns):
    if not columns or not len(y):
        return None
    tp = np.logical_and(y, p).sum(0)
    den = y.sum(0) + p.sum(0)
    return float(
        np.divide(2 * tp, den, out=np.zeros(len(den), float), where=den > 0)[
            columns
        ].mean()
    )


def percentile(values, probabilities):
    a = sorted(float(z) for z in values)
    n = len(a)
    x.require(n > 0, "Empty interval input")
    out = []
    for q in probabilities:
        x.require(0 <= q <= 1, "Invalid quantile")
        position = q * (n - 1)
        i = int(position)
        out.append(a[i] + (a[min(i + 1, n - 1)] - a[i]) * (position - i))
    return out


def relation(interval):
    if interval is None or interval[0] is None:
        return "not_scorable"
    x.require(interval[0] <= interval[1], "Reversed interval")
    return (
        "above_zero"
        if interval[0] > 0
        else "below_zero" if interval[1] < 0 else "crosses_or_touches_zero"
    )


def selected_interval(comparison):
    x.require(
        comparison["role"] in ("primary", "supporting"), "Unknown comparison role"
    )
    if comparison["role"] == "primary":
        x.require(
            comparison["family_size"] == 4 and comparison["right"] == "rag_small",
            "Primary family changed",
        )
        return comparison["family_interval"]
    return comparison["ci95"]


def fixtures():
    n = 0

    def check(value, message):
        nonlocal n
        x.require(value, "Integration fixture: " + message)
        n += 1

    y = np.array([[1, 0], [0, 1], [1, 0]], bool)
    p = np.array([[1, 0], [1, 0], [1, 0]], bool)
    check(abs(score(y, p, [0, 1]) - 0.4) < 1e-14, "TP denominator")
    check(score(y, y, [0, 1]) == 1, "perfect")
    check(score(y, p, []) is None, "unsupported labels")
    check(score(y[:0], p[:0], [0, 1]) is None, "empty support")
    check(percentile([0, 1, 2, 3], [0.25, 0.75]) == [0.75, 2.25], "linear percentiles")
    for interval, result in [
        ([0.01, 0.1], "above_zero"),
        ([-0.2, -0.1], "below_zero"),
        ([0, 0.1], "crosses_or_touches_zero"),
        ([-0.1, 0], "crosses_or_touches_zero"),
        ([-0.1, 0.1], "crosses_or_touches_zero"),
        (None, "not_scorable"),
    ]:
        check(relation(interval) == result, "interval classification")
    v = {
        "role": "primary",
        "family_size": 4,
        "right": "rag_small",
        "family_interval": [-0.01, 0.1],
        "ci95": [0.01, 0.09],
    }
    check(
        selected_interval(v) == [-0.01, 0.1], "do not select narrower positive interval"
    )
    vv = np.array([[True, True, False], [True, False, True]])
    check(vv.all(0).tolist() == [True, False, False], "common intersection")
    known_empty = np.array([True, True])
    gate = np.array([False, True])
    check((known_empty & gate).tolist() == [False, True], "exact set needs hate gate")
    check(known_empty.tolist() == [True, True], "legal exactness independent")
    check(np.mean([0.2, 0.8]) == 0.5, "mean individual scores")
    return n


def run():
    tests = fixtures()
    receipt = s.read(s.BASE / "consolidation_complete.json")
    s.verify_map(receipt["sources"])
    s.verify_map(receipt["outputs"])
    summary = s.read(s.BASE / "main_evidence.json")
    db = {}
    arrays = {}
    own_scores = 0
    pair_count = 0
    draw_count = 0
    primary = []
    sources = []
    for name in s.ORDER:
        path = s.MAIN / "endpoints" / name.replace("/", "__") / "results.json"
        v = s.read(path)
        db[name] = v
        ep = v["endpoint"]
        settings = [tuple(a) for a in v["settings"]]
        sources += [path, s.ROOT / v["encoded_path"]]
        with np.load(s.ROOT / v["encoded_path"], allow_pickle=False) as a:
            y = a["gold"].astype(bool)
            p = a["predicted"].astype(bool)
            valid = a["valid"]
            counts = a["counts"]
        cols = np.flatnonzero(y.sum(0) >= 10).tolist()
        same(
            [ep["all_labels"][i] for i in cols],
            ep["macro_labels"],
            "fixed supported labels",
        )
        independent = np.stack((y & p, ~y & p, y & ~p), -1).astype(np.int8)
        independent[~valid] = 0
        same(independent, counts, "saved contributions")
        own_bounds = {}
        own = {}
        for co in ep["available_conditions"]:
            rr = []
            bounds = []
            for rep in range(3):
                j = settings.index((co, rep))
                mask = valid[j]
                z = score(y[mask], p[j, mask], cols)
                old = v["own_valid"][co]["repeats"][rep]
                same(z, old["macro_f1_valid_only"], "own F1")
                same(mask.sum(), old["n_valid"], "own valid")
                same((~mask).sum(), old["n_invalid"], "own invalid")
                rr.append(z)
                own_scores += 1
                if cols:
                    tp = (y[mask] & p[j, mask]).sum(0)
                    pred = p[j, mask].sum(0)
                    gold = y.sum(0)
                    miss = y[~mask]
                    lo_pred = pred + (~miss).sum(0)
                    hi_tp = tp + miss.sum(0)
                    hi_pred = pred + miss.sum(0)
                    f = lambda t, q: np.divide(
                        2 * t,
                        q + gold,
                        out=np.zeros(len(gold), float),
                        where=q + gold > 0,
                    )[cols].mean()
                    b = [float(f(tp, lo_pred)), float(f(hi_tp, hi_pred))]
                else:
                    b = [None, None]
                same(b, old["invalid_inclusive_bounds"], "failure bounds")
                bounds.append(b)
            own[co] = rr
            own_bounds[co] = np.mean(bounds, axis=0).tolist() if cols else [None, None]
            same(
                np.mean(rr) if cols else None,
                v["own_valid"][co]["mean_macro_f1"],
                "mean repeat F1",
            )
        ix9 = [settings.index((co, rep)) for co in x.CONDITIONS for rep in range(3)]
        mask9 = valid[ix9].all(0)
        same(mask9.sum(), v["three_condition_common"]["n_common_valid"], "common nine")
        same(
            [key for key, on in zip(ep["scorable_ids"], mask9) if not on],
            v["three_condition_common"]["excluded_ids"],
            "common excluded ids",
        )
        for co in x.CONDITIONS:
            values = [
                score(y[mask9], p[settings.index((co, j)), mask9], cols)
                for j in range(3)
            ]
            same(
                values,
                v["three_condition_common"]["per_condition"][co]["repeat_macro_f1"],
                "common repeat values",
            )
            same(
                np.mean(values) if cols else None,
                v["three_condition_common"]["per_condition"][co]["mean_macro_f1"],
                "common mean",
            )
        for c in v["contrasts"]:
            indices = [
                settings.index((co, j))
                for co in (c["left"], c["right"])
                for j in range(3)
            ]
            mask = valid[indices].all(0)
            same(mask.sum(), c["n_common_valid"], "pair common")
            same(
                [key for key, on in zip(ep["scorable_ids"], mask) if on],
                c["common_valid_ids"],
                "pair identities",
            )
            pair_count += 1
            if c["role"] == "primary":
                primary.append(name)
            if cols:
                pts = [score(y[mask], p[j, mask], cols) for j in indices]
                same(pts, c["common_f1_by_setting"], "pair points")
                same(
                    np.mean(pts[:3]) - np.mean(pts[3:]),
                    c["mean_difference"],
                    "pair difference",
                )
                same(np.mean(pts[:3]), c["left_common_mean"], "left mean")
                same(np.mean(pts[3:]), c["right_common_mean"], "right mean")
                same(
                    [
                        own_bounds[c["left"]][0] - own_bounds[c["right"]][1],
                        own_bounds[c["left"]][1] - own_bounds[c["right"]][0],
                    ],
                    c["full_roster_failure_difference_bounds"],
                    "pair failure bounds",
                )
                path = s.ROOT / c["draws_path"]
                same(x.c.sha(path), c["draws_sha256"], "retained draws hash")
                draws = np.load(path, allow_pickle=False)
                sources.append(path)
                same(len(draws), 10000, "draw count")
                draw_count += len(draws)
                same(
                    percentile(draws, [0.025, 0.975]), c["ci95"], "supporting interval"
                )
                if c["family_size"]:
                    same(
                        percentile(
                            draws,
                            [
                                0.05 / (2 * c["family_size"]),
                                1 - 0.05 / (2 * c["family_size"]),
                            ],
                        ),
                        c["family_interval"],
                        "adjusted interval",
                    )
            else:
                same(c["mean_difference"], None, "unsupported difference")
                same(c["actual_draws"], 0, "no unsupported draws")
            same(
                relation(selected_interval(c)),
                c["interval_relation"],
                "original interval classification",
            )
        arrays[name] = (y, p, valid, settings, mask9)
        print("Integrated points, masks and retained intervals:", name, flush=True)
    same(sorted(primary), sorted(s.PRIMARY), "four primary questions")
    same([own_scores, pair_count, draw_count], [153, 40, 380000], "main scope")

    errors = s.read(s.part("errors") / "binary_results.json")
    sources.append(s.part("errors") / "binary_results.json")
    for phase, old in errors.items():
        y, p, valid, settings, mask9 = arrays[phase + "/hate"]
        gold = y[:, 1]

        def metrics(indices, mask):
            rows = []
            for j in indices:
                pred = p[j, :, 1]
                tn = int((~pred & ~gold & mask).sum())
                fp = int((pred & ~gold & mask).sum())
                fn = int((~pred & gold & mask).sum())
                tp = int((pred & gold & mask).sum())
                rows.append(
                    dict(
                        tn=tn,
                        fp=fp,
                        fn=fn,
                        tp=tp,
                        fpr=fp / (tn + fp),
                        fnr=fn / (fn + tp),
                        error_rate=(fp + fn) / int(mask.sum()),
                    )
                )
            return {k: float(np.mean([z[k] for z in rows])) for k in rows[0]}

        for co in x.CONDITIONS:
            calc = metrics([settings.index((co, j)) for j in range(3)], mask9)
            for k, z in calc.items():
                same(z, old["common_nine"][co]["mean"][k], "binary common " + k)
        for pair in old["paired"]:
            ii = [
                settings.index((co, j))
                for co in ("rag_full", pair["reference"])
                for j in range(3)
            ]
            mask = valid[ii].all(0)
            a = metrics(ii[:3], mask)
            b = metrics(ii[3:], mask)
            for k in a:
                same(a[k], pair["full"]["mean"][k], "binary paired full")
                same(
                    b[k],
                    pair["reference_summary"]["mean"][k],
                    "binary paired reference",
                )
            for k in ("fpr", "fnr", "error_rate"):
                same(
                    a[k] - b[k],
                    pair["diagnostic_intervals"]["metrics"][k]["difference"],
                    "binary diagnostic point",
                )

    quality = {}
    qarrays = {}
    for name in s.ORDER:
        if name.endswith("/hate"):
            continue
        path = (
            s.part("quality") / "endpoints" / name.replace("/", "__") / "results.json"
        )
        v = s.read(path)
        quality[name] = v
        sources.extend([path, s.ROOT / v["encoded_path"]])
        with np.load(s.ROOT / v["encoded_path"], allow_pickle=False) as z:
            q = {k: z[k] for k in z.files}
        y, p, valid, settings, mask9 = arrays[name]
        same(q["gold"], y, "fine gold")
        same(q["predicted"], p, "fine prediction")
        same(q["valid"], valid, "fine validity")
        same(q["common_nine"], mask9, "fine mask")
        exact = (p == y).all(2)
        gate = q["gate"]
        correct = exact if name.endswith("/legal") else exact & gate
        for co in x.CONDITIONS:
            ii = [settings.index((co, j)) for j in range(3)]
            same(
                float(correct[ii][:, mask9].mean()),
                v["common_nine"][co]["mean"]["gate_aware_exact"],
                "gate exact",
            )
            same(
                float(exact[ii][:, mask9].mean()),
                v["common_nine"][co]["mean"]["exact_set"],
                "set exact",
            )
        qarrays[name] = q
    joint = s.read(s.part("quality") / "joint_results.json")
    sources.append(s.part("quality") / "joint_results.json")
    for phase, old in joint.items():
        t = quality[phase + "/target_group"]
        v = quality[phase + "/severity"]
        tq = qarrays[phase + "/target_group"]
        vq = qarrays[phase + "/severity"]
        ix = [t["ids"].index(k) for k in v["ids"]]
        te = (tq["predicted"][:, ix] == tq["gold"][ix]).all(2)
        se = (vq["predicted"] == vq["gold"]).all(2)
        g = vq["gate"]
        same(tq["gate"][:, ix], g, "joint gate")
        parts = dict(
            joint_exact=g & te & se,
            gate_closed=~g,
            target_only_wrong=g & ~te & se,
            severity_only_wrong=g & te & ~se,
            both_wrong=g & ~te & ~se,
        )
        same(
            sum(z.astype(int) for z in parts.values()),
            np.ones_like(g, int),
            "joint partition",
        )
        settings = [tuple(z) for z in old["settings"]]
        m = vq["common_nine"]
        with np.load(s.ROOT / old["encoded_path"], allow_pickle=False) as z:
            for k, value in parts.items():
                same(value, z[k], "joint saved components")
        for co in x.CONDITIONS:
            ii = [settings.index((co, j)) for j in range(3)]
            for k, value in parts.items():
                same(
                    value[ii][:, m].mean(),
                    old["common_nine"][co]["mean"][k],
                    "joint mean",
                )
    robust = s.read(s.part("robustness") / "derived_findings.json")
    sources.append(s.part("robustness") / "derived_findings.json")
    for name, v in db.items():
        for c in v["contrasts"]:
            o = robust["comparisons"][name + "/" + c["right"]]["original"]
            same(o["difference"], c["mean_difference"], "robustness original point")
            same(o["interval"], selected_interval(c), "robustness original interval")
            same(o["n"], c["n_common_valid"], "robustness original mask")
    for scenario in (
        "required_sampling",
        "replay_exclusion",
        "whitespace_exclusion",
        "core_exclusion",
    ):
        same(
            robust["scenario_overview"][scenario]["classification_changes"],
            0,
            "required sensitivity classification",
        )
    resource = s.read(s.part("resources") / "derived_findings.json")
    sources.append(s.part("resources") / "derived_findings.json")
    same(
        sum(z["total_tokens"] for z in resource["conditions"]),
        resource["ledger"]["total_tokens"],
        "token total",
    )
    same(
        resource["ledger"]["ordinary_logical_calls"]
        + resource["ledger"]["repair_logical_calls"],
        resource["ledger"]["http_success_with_text"],
        "logical calls",
    )
    same(
        resource["ledger"]["http_success_with_text"]
        + resource["ledger"]["retry_attempts"],
        resource["ledger"]["post_attempts"],
        "POSTs",
    )
    tables = s.read(s.BASE / "consolidated_tables.json")
    rendered = (s.BASE / "tables.md").read_text(encoding="utf8")
    for t in tables:
        source = s.ROOT / t["source"]
        same(x.c.sha(source), t["source_sha256"], "table source")
        block = s.blocks(source)[t["source_block"]]
        same(block["markdown"], t["markdown"], "complete table")
        same(block["rows"], t["rows"], "table rows")
        x.require(t["markdown"] in rendered, "Consolidated table missing")
    s.save(
        s.BASE / "integration_review.json",
        dict(
            status="main_points_intervals_and_cross_part_integration_passed",
            tests=tests,
            checked_values=checked,
            maximum_absolute_difference=maximum,
            own_setting_scores=own_scores,
            original_contrasts=pair_count,
            retained_draw_values_for_quantiles=draw_count,
            new_bootstrap_draws=0,
            tables=len(tables),
            table_rows=sum(t["rows"] for t in tables),
            sources=s.bind(
                [Path(__file__), s.BASE / "consolidation_complete.json", *sources]
            ),
            new_Qwen_calls=0,
            distinction="Main F1 points/masks and interval quantiles independently reconstructed; old bootstrap draws were reused, not regenerated. Binary, exact, joint, original robustness and resource identities reconciled. Existing source/function/context/raw checks remain their prior verified evidence.",
        ),
    )
    print(
        "Integration passed:",
        checked,
        "values;",
        tests,
        "fixtures;",
        pair_count,
        "original contrasts.",
        flush=True,
    )


if __name__ == "__main__":
    with x.r.original.offline(), x.prior.threadpool_limits(limits=2):
        run()

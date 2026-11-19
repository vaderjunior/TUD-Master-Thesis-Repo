from collections import Counter
import hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.classification.broad import quality as q

checked = 0
maximum = 0.0
draw_values = 0


def equal(want, got, path="value"):
    global checked, maximum
    if isinstance(want, dict):
        for k, v in want.items():
            q.require(k in got, path + "/" + k + " absent")
            equal(v, got[k], path + "/" + str(k))
    elif isinstance(want, (list, tuple)):
        q.require(len(want) == len(got), path + " length")
        for i, (a, b) in enumerate(zip(want, got)):
            equal(a, b, path + "/" + str(i))
    elif isinstance(want, np.ndarray):
        a = np.asarray(got)
        q.require(a.shape == want.shape, path + " array shape")
        if np.issubdtype(want.dtype, np.number):
            delta = np.abs(want.astype(float) - a.astype(float))
            m = float(delta.max(initial=0))
            q.require(m < 2e-11, path + " array value")
            maximum = max(maximum, m)
            checked += want.size
        else:
            q.require(np.array_equal(want, a), path + " array values")
            checked += want.size
    elif isinstance(want, (int, float, np.number)) and not isinstance(
        want, (bool, np.bool_)
    ):
        q.require(
            got is not None and abs(float(want) - float(got)) < 2e-11, path + " numeric"
        )
        maximum = max(maximum, abs(float(want) - float(got)))
        checked += 1
    else:
        q.require(want == got, path + " differs")


def ratio(a, b):
    return float(a / b) if b else None


def direct_rows(y, p, g, legal):
    values = {}
    for rep in range(len(p)):
        rows = []
        for j in range(len(y)):
            truth = set(np.flatnonzero(y[j]))
            pred = set(np.flatnonzero(p[rep, j]))
            extra = pred - truth
            missing = truth - pred
            tp = pred & truth
            union = pred | truth
            exact = pred == truth
            gate = bool(g[rep, j])
            rows.append(
                dict(
                    exact_set=exact,
                    gate_aware_exact=exact and (legal or gate),
                    extra_labels=len(extra),
                    missing_labels=len(missing),
                    predicted_labels=len(pred),
                    gold_labels=len(truth),
                    sample_jaccard=len(tp) / len(union) if union else 1.0,
                    sample_f1=(
                        2 * len(tp) / (len(truth) + len(pred)) if truth or pred else 1.0
                    ),
                    extra_only=bool(extra) and not missing,
                    missing_only=bool(missing) and not extra,
                    mixed=bool(extra) and bool(missing),
                    has_extra=bool(extra),
                    has_missing=bool(missing),
                    true_positive_labels=len(tp),
                    empty_gold=not truth,
                    predicted_empty=not pred,
                    gate_pass=gate,
                    empty_exact_wrong_gate=not legal
                    and not truth
                    and exact
                    and not gate,
                    positive_gold_blocked=not legal and bool(truth) and not gate,
                )
            )
        for k in rows[0]:
            values.setdefault(k, []).append([v[k] for v in rows])
    return {k: np.asarray(v, dtype=float) for k, v in values.items()}


def summary(y, p, g, labels, macro, legal):
    rows = direct_rows(y, p, g, legal)
    out = []
    for i in range(len(p)):
        per = []
        for j, label in enumerate(labels):
            pairs = list(zip(y[:, j].tolist(), p[i, :, j].tolist()))
            tp = sum(a == b == 1 for a, b in pairs)
            fp = sum(a == 0 and b == 1 for a, b in pairs)
            fn = sum(a == 1 and b == 0 for a, b in pairs)
            per.append(
                dict(
                    label=label,
                    gold_support=tp + fn,
                    predicted=tp + fp,
                    tp=tp,
                    fp=fp,
                    fn=fn,
                    precision=ratio(tp, tp + fp),
                    recall=ratio(tp, tp + fn),
                    f1=ratio(2 * tp, 2 * tp + fp + fn) or 0.0,
                    f1_denominator_zero=not (2 * tp + fp + fn),
                    included_in_original_macro=label in macro,
                )
            )
        card = np.zeros((len(labels) + 1, len(labels) + 1), int)
        for truth, pred in zip(y, p[i]):
            card[int(sum(truth)), int(sum(pred))] += 1
        out.append(
            dict(
                n=len(y),
                metrics={k: float(sum(v[i]) / len(y)) for k, v in rows.items()},
                counts={k: float(sum(v[i])) for k, v in rows.items()},
                per_label=per,
                macro_f1=(
                    float(
                        np.mean(
                            [v["f1"] for v in per if v["included_in_original_macro"]]
                        )
                    )
                    if macro
                    else None
                ),
                exact_given_gate=(
                    ratio(sum(rows["exact_set"][i] * g[i]), sum(g[i]))
                    if not legal
                    else None
                ),
                cardinality=card.tolist(),
            )
        )
    mean = {k: float(np.mean([v["metrics"][k] for v in out])) for k in rows}
    pl = []
    for j, label in enumerate(labels):
        d = dict(
            label=label,
            included_in_original_macro=label in macro,
            gold_support=int(sum(y[:, j])),
        )
        for k in ("predicted", "tp", "fp", "fn", "precision", "recall", "f1"):
            vv = [v["per_label"][j][k] for v in out]
            d[k] = float(np.mean(vv)) if all(v is not None for v in vv) else None
            if k in ("precision", "recall"):
                d[k + "_undefined_repeats"] = sum(v is None for v in vv)
        pl.append(d)
    return dict(
        repeats=out,
        mean=mean,
        sd={k: float(np.std([v["metrics"][k] for v in out], ddof=1)) for k in rows},
        per_label_mean=pl,
    )


def check_detail(y, p, g, dim, got):
    for i in range(len(p)):
        if dim == "severity":
            matrix = np.zeros((3, 5), int)
            counts = Counter()
            dist = []
            for truth, pred, gate in zip(y, p[i], g[i]):
                a = list(truth).index(1)
                b = list(pred).index(1) if any(pred) else None
                matrix[a, 0 if not gate else 1 if b is None else b + 2] += 1
                if not gate:
                    counts["gate_closed"] += 1
                elif b is None:
                    counts["gate_open_missing"] += 1
                else:
                    d = b - a
                    counts[
                        {
                            0: "correct",
                            -1: "under_one",
                            -2: "under_two",
                            1: "over_one",
                            2: "over_two",
                        }[d]
                    ] += 1
                    dist.append(abs(d))
            equal(matrix, np.asarray(got[i]["confusion"]), "severity confusion")
            equal(
                {k: counts[k] for k in got[i]["counts"]},
                got[i]["counts"],
                "severity categories",
            )
            equal(
                float(np.mean(dist)) if dist else None,
                got[i]["conditional_ordinal_mae"],
                "ordinal distance",
            )
            equal(len(dist), got[i]["emitted"], "emitted severity")
        elif dim == "legal":
            gold = [bool(any(v)) for v in y]
            pred = [bool(any(v)) for v in p[i]]
            exact = [list(a) == list(b) for a, b in zip(y, p[i])]
            tp = sum(a and b for a, b in zip(gold, pred))
            fp = sum(not a and b for a, b in zip(gold, pred))
            fn = sum(a and not b for a, b in zip(gold, pred))
            tn = len(y) - tp - fp - fn
            equal(
                dict(
                    known_empty=fp + tn,
                    annotated_positive=tp + fn,
                    empty_gold_flagged=fp,
                    empty_gold_flagged_rate=ratio(fp, fp + tn),
                    positive_gold_unflagged=fn,
                    positive_gold_unflagged_rate=ratio(fn, tp + fn),
                    any_legal_tp=tp,
                    any_legal_fp=fp,
                    any_legal_fn=fn,
                    any_legal_tn=tn,
                    by_predicted_hate={
                        str(flag): dict(
                            n=sum(v == flag for v in g[i]),
                            exact=sum(e and v == flag for e, v in zip(exact, g[i])),
                            legal_present=sum(
                                e and v == flag for e, v in zip(pred, g[i])
                            ),
                            gold_positive=sum(
                                e and v == flag for e, v in zip(gold, g[i])
                            ),
                        )
                        for flag in (False, True)
                    },
                ),
                got[i],
                "legal detail",
            )


def check_boot(profile, clusters, keys, pair):
    global draw_values
    with np.load(q.ROOT / pair["bootstrap_path"], allow_pickle=False) as z:
        equal(profile, z["profile"], "paired profile")
        saved = z["draws"].copy()
    keys_sorted = sorted(set(clusters))
    groups = [[i for i, k in enumerate(clusters) if k == key] for key in keys_sorted]
    totals = np.array([profile[ii].sum(0) for ii in groups])
    sizes = np.array([len(ii) for ii in groups])
    rng = np.random.default_rng(q.SEED)
    direct = []
    for start in range(0, q.DRAWS, 64):
        picks = rng.choice(
            list(range(len(groups))),
            (min(64, q.DRAWS - start), len(groups)),
            replace=True,
        )
        direct.append(totals[picks].sum(1) / sizes[picks].sum(1)[:, None])
    direct = np.concatenate(direct)
    equal(direct, saved, "all bootstrap draws")
    draw_values += saved.size
    for j, key in enumerate(keys):
        equal(
            dict(
                difference=profile[:, j].mean(),
                ci95=np.quantile(direct[:, j], [0.025, 0.975]).tolist(),
            ),
            pair["intervals"]["metrics"][key],
            "interval " + key,
        )
    equal(
        dict(n=len(profile), n_clusters=len(groups), draws=q.DRAWS, seed=q.SEED),
        pair["intervals"],
        "bootstrap population",
    )


def events(a, b):
    fixes = []
    losses = []
    for ai in a:
        for bi in b:
            fixes.append(sum(bool(x) and not bool(y) for x, y in zip(ai, bi)))
            losses.append(sum(bool(y) and not bool(x) for x, y in zip(ai, bi)))
    return dict(
        fixed=float(np.mean(fixes)),
        lost=float(np.mean(losses)),
        net=float(np.mean(fixes) - np.mean(losses)),
        stable_fixed=sum(all(a[:, j]) and not any(b[:, j]) for j in range(a.shape[1])),
        stable_lost=sum(not any(a[:, j]) and all(b[:, j]) for j in range(a.shape[1])),
        stable_both_wrong=sum(
            not any(a[:, j]) and not any(b[:, j]) for j in range(a.shape[1])
        ),
        stable_both_right=sum(all(a[:, j]) and all(b[:, j]) for j in range(a.shape[1])),
    )


def run():
    receipt = q.c.read(q.BASE / "analysis_complete.json")
    q.r.verify(receipt["sources"])
    q.r.verify(receipt["outputs"])
    groups, _, runs, endpoints, _ = q.reader.load()
    cached = {}
    common_views = 0
    own_views = 0
    pair_views = 0
    gate_views = 0
    previous = q.c.read(q.errors.BASE / "gate_delivery.json")["rows"]
    parts = []
    for path in sorted((q.BASE / "endpoints").glob("*/results.json")):
        out = q.c.read(path)
        name = out["endpoint"]
        phase, dim = name.split("/")
        ep = endpoints[name]
        lookup = {row["key"]: row for row in groups[phase]}
        ids = out["ids"]
        labels = out["labels"]
        settings = [tuple(v) for v in out["settings"]]
        y = []
        for key in ids:
            row = lookup[key]
            value = row[q.FIELDS[dim]]
            truth = {value} if dim == "severity" else set(value)
            q.require(dim == "legal" or row["gate"] is True, "Fine annotation gate")
            y.append([int(l in truth) for l in labels])
        y = np.array(y)
        p = []
        v = []
        g = []
        for cond, rep in settings:
            pp = []
            vv = []
            gg = []
            for key in ids:
                pred = runs[cond, rep][key]
                ok = pred is not None
                value = pred[dim] if ok else None
                ss = (
                    {value}
                    if dim == "severity" and value is not None
                    else set(value or []) if dim != "severity" else set()
                )
                pp.append([int(l in ss) for l in labels])
                vv.append(ok)
                gg.append(bool(pred["hate"]) if ok else False)
            p.append(pp)
            v.append(vv)
            g.append(gg)
        p = np.array(p)
        v = np.array(v, bool)
        g = np.array(g, bool)
        ix9 = [settings.index((cond, j)) for cond in q.CONDITIONS for j in range(3)]
        m9 = v[ix9].all(0)
        with np.load(q.ROOT / out["encoded_path"], allow_pickle=False) as z:
            for k, want in dict(
                gold=y, predicted=p, valid=v, gate=g, common_nine=m9
            ).items():
                equal(want, z[k], name + "/" + k)
        allrows = direct_rows(y, p, g, dim == "legal")
        cached[name] = (y, p, g, v, settings, ids, allrows)
        equal(int(m9.sum()), out["common_nine_n"], name + " shared n")
        for cond, own in out["own_valid"].items():
            ix = [settings.index((cond, j)) for j in range(3)]
            for j, s in enumerate(ix):
                m = v[s]
                calc = summary(
                    y[m],
                    np.repeat(p[s : s + 1, m], 3, 0),
                    np.repeat(g[s : s + 1, m], 3, 0),
                    labels,
                    out["macro_labels"],
                    dim == "legal",
                )["repeats"][0]
                equal(calc, own[j], name + " own")
                bounds = lambda key: [
                    float(allrows[key][s, m].sum() / len(y)),
                    float((allrows[key][s, m].sum() + (~m).sum()) / len(y)),
                ]
                equal(
                    dict(
                        eligible=len(y),
                        invalid=int((~m).sum()),
                        exact_bounds=bounds("exact_set"),
                        gate_exact_bounds=bounds("gate_aware_exact"),
                    ),
                    own[j],
                    "own failures",
                )
                own_views += 1
                if dim in ("severity", "legal"):
                    check_detail(
                        y[m], p[s : s + 1, m], g[s : s + 1, m], dim, [own[j]["detail"]]
                    )
            if cond in q.CONDITIONS:
                equal(
                    summary(
                        y[m9],
                        p[ix][:, m9],
                        g[ix][:, m9],
                        labels,
                        out["macro_labels"],
                        dim == "legal",
                    ),
                    out["common_nine"][cond],
                    name + " common summary",
                )
                common_views += 1
                if dim in ("severity", "legal"):
                    check_detail(
                        y[m9],
                        p[ix][:, m9],
                        g[ix][:, m9],
                        dim,
                        out["common_nine"][cond]["detail"],
                    )
                if dim != "legal":
                    for j, s in enumerate(ix):
                        for maskname, m in (("common_nine", m9), ("own_valid", v[s])):
                            old = next(
                                x
                                for x in previous
                                if x["population"] == phase
                                and x["dimension"] == dim
                                and x["condition"] == cond
                                and x["repeat"] == j
                                and x["mask"] == maskname
                            )
                            equal(
                                dict(
                                    valid=int(m.sum()),
                                    gate_false=int((~g[s, m]).sum()),
                                    positive_gold_gated_off=int(
                                        ((y[m].sum(1) > 0) & ~g[s, m]).sum()
                                    ),
                                    empty_gold_gate_false=int(
                                        ((y[m].sum(1) == 0) & ~g[s, m]).sum()
                                    ),
                                ),
                                old,
                                "Part 3 gate reconciliation",
                            )
                            gate_views += 1
        for pair in out["pairs"]:
            right = pair["reference"]
            ix = [
                settings.index((cond, j))
                for cond in ("rag_full", right)
                for j in range(3)
            ]
            m = v[ix].all(0)
            rows = direct_rows(y[m], p[ix][:, m], g[ix][:, m], dim == "legal")
            equal(
                summary(
                    y[m],
                    p[ix[:3]][:, m],
                    g[ix[:3]][:, m],
                    labels,
                    out["macro_labels"],
                    dim == "legal",
                ),
                pair["full"],
                "pair full",
            )
            equal(
                summary(
                    y[m],
                    p[ix[3:]][:, m],
                    g[ix[3:]][:, m],
                    labels,
                    out["macro_labels"],
                    dim == "legal",
                ),
                pair["reference_summary"],
                "pair reference",
            )
            equal(int(m.sum()), pair["n"], "pair n")
            equal(
                events(rows["gate_aware_exact"][:3], rows["gate_aware_exact"][3:]),
                pair["exact_events"],
                "exact transitions",
            )
            a = out["own_valid"]["rag_full"]
            b = out["own_valid"][right]
            equal(
                [
                    np.mean([x["gate_exact_bounds"][0] for x in a])
                    - np.mean([x["gate_exact_bounds"][1] for x in b]),
                    np.mean([x["gate_exact_bounds"][1] for x in a])
                    - np.mean([x["gate_exact_bounds"][0] for x in b]),
                ],
                pair["full_roster_gate_exact_difference_bounds"],
                "pair failure bounds",
            )
            if right in ("rag_small", "zero_shot"):
                profile = np.array(
                    [
                        [rows[k][:3, j].mean() - rows[k][3:, j].mean() for k in q.BOOT]
                        for j in range(m.sum())
                    ]
                )
                check_boot(
                    profile,
                    [cl for cl, on in zip(ep["row_clusters"], m) if on],
                    q.BOOT,
                    pair,
                )
            pair_views += 1
        parts.append(path)
        print("Independent fine review:", name, flush=True)
    joints = q.c.read(q.BASE / "joint_results.json")
    joint_rows = {}
    for phase, out in joints.items():
        ty, tp, tg, tv, ts, tids, _ = cached[phase + "/target_group"]
        sy, sp, sg, sv, ss, sids, _ = cached[phase + "/severity"]
        ix = [tids.index(key) for key in sids]
        m9 = sv[[ss.index((cond, j)) for cond in q.CONDITIONS for j in range(3)]].all(0)
        vectors = {
            k: np.zeros(sg.shape, bool)
            for k in (
                "joint_exact",
                "gate_closed",
                "target_only_wrong",
                "severity_only_wrong",
                "both_wrong",
            )
        }
        for s in range(len(ss)):
            for j in range(len(sids)):
                te = np.array_equal(tp[s, ix[j]], ty[ix[j]])
                se = np.array_equal(sp[s, j], sy[j])
                key = (
                    "gate_closed"
                    if not sg[s, j]
                    else (
                        "joint_exact"
                        if te and se
                        else (
                            "target_only_wrong"
                            if se
                            else "severity_only_wrong" if te else "both_wrong"
                        )
                    )
                )
                vectors[key][s, j] = True
        with np.load(q.ROOT / out["encoded_path"], allow_pickle=False) as z:
            for k, want in dict(vectors, valid=sv, common_nine=m9).items():
                equal(want, z[k], "joint " + k)
        joint_rows[phase] = (vectors, sv, ss, sids)
        summary_joint = lambda ii, m: dict(
            repeats=[
                {k: float(v[s, m].mean()) for k, v in vectors.items()} for s in ii
            ],
            mean={k: float(v[ii][:, m].mean()) for k, v in vectors.items()},
        )
        for cond, own in out["own_valid"].items():
            ii = [ss.index((cond, j)) for j in range(3)]
            for j, s in enumerate(ii):
                equal(
                    dict(
                        n=int(sv[s].sum()),
                        invalid=int((~sv[s]).sum()),
                        metrics=summary_joint([s], sv[s])["mean"],
                        bounds=[
                            float(vectors["joint_exact"][s, sv[s]].sum() / len(sids)),
                            float(
                                (
                                    vectors["joint_exact"][s, sv[s]].sum()
                                    + (~sv[s]).sum()
                                )
                                / len(sids)
                            ),
                        ],
                    ),
                    own[j],
                    "joint own",
                )
            if cond in q.CONDITIONS:
                equal(summary_joint(ii, m9), out["common_nine"][cond], "joint shared")
        for pair in out["pairs"]:
            right = pair["reference"]
            ii = [ss.index((cond, j)) for cond in ("rag_full", right) for j in range(3)]
            m = sv[ii].all(0)
            a = vectors["joint_exact"][ii[:3]][:, m]
            b = vectors["joint_exact"][ii[3:]][:, m]
            equal(summary_joint(ii[:3], m), pair["full"], "joint pair full")
            equal(summary_joint(ii[3:], m), pair["reference_summary"], "joint pair ref")
            equal(events(a, b), pair["exact_events"], "joint transitions")
            equal(
                [
                    np.mean([x["bounds"][0] for x in out["own_valid"]["rag_full"]])
                    - np.mean([x["bounds"][1] for x in out["own_valid"][right]]),
                    np.mean([x["bounds"][1] for x in out["own_valid"]["rag_full"]])
                    - np.mean([x["bounds"][0] for x in out["own_valid"][right]]),
                ],
                pair["full_roster_difference_bounds"],
                "joint failure bounds",
            )
            if right in ("rag_small", "zero_shot"):
                check_boot(
                    (a.mean(0) - b.mean(0))[:, None],
                    [
                        cl
                        for cl, on in zip(
                            endpoints[phase + "/severity"]["row_clusters"], m
                        )
                        if on
                    ],
                    ["joint_exact"],
                    pair,
                )
        print("Independent joint review:", phase, flush=True)
    selected = q.c.read(q.BASE / "case_selection.json")["slots"]
    cases = q.c.read(q.BASE / "cases.json")
    cache_by_case = {}
    for row in q.reader.rows(q.reader.BASE / "predictions.jsonl"):
        if row["key"] in {slot["key"] for slot in selected}:
            cache_by_case.setdefault(row["key"], []).append(row)
    for slot, case in zip(selected, cases):
        name = slot["endpoint"]
        phase, dim = name.split("/")
        if dim == "joint":
            vectors, v, ss, ids = joint_rows[phase]
            data = vectors["joint_exact"]
        else:
            _, _, _, v, ss, ids, rows = cached[name]
            data = rows["gate_aware_exact"].astype(bool)
        ii = [
            ss.index((cond, j)) for cond in ("rag_full", "rag_small") for j in range(3)
        ]
        m = v[ii].all(0)
        category = slot["category"]
        if category == "empty_exact_wrong_gate_all_full_repeats":
            yy, pp, gg, v, ss, ids, rows = cached[name]
            ii = ii[:3]
            candidates = [
                key
                for j, key in enumerate(ids)
                if all(v[ii, j]) and not any(yy[j]) and not any(gg[ii, j])
            ]
            digest = lambda key: hashlib.sha256(
                ("sq2-broad-part4-empty|" + key).encode()
            ).hexdigest()
        else:
            candidates = []
            for j, key in enumerate(ids):
                if not m[j]:
                    continue
                a = data[ii[:3], j]
                b = data[ii[3:], j]
                if {
                    "stable_fix": all(a) and not any(b),
                    "stable_regression": not any(a) and all(b),
                    "stable_shared_error": not any(a) and not any(b),
                }[category]:
                    candidates.append(key)
            digest = lambda key: hashlib.sha256(
                f"sq2-broad-part4-v1|{name}|{category}|{key}".encode()
            ).hexdigest()
        key = (
            min(candidates, key=lambda key: (digest(key), key)) if candidates else None
        )
        equal(
            dict(
                key=key,
                candidates=len(candidates),
                selection_sha256=digest(key) if key else None,
            ),
            slot,
            "case rule",
        )
        if key:
            equal(cache_by_case[key], case["predictions"], "all selected decisions")
            equal(
                next(v for v in groups[phase] if v["key"] == key),
                case["gold"],
                "selected gold",
            )
    inventory = {
        phase: dict(
            Counter(
                "+".join(
                    (["hate"] if row.get("gate") is not None else [])
                    + [
                        dim
                        for dim, field in q.FIELDS.items()
                        if row.get(field) is not None
                    ]
                )
                for row in rr
            )
        )
        for phase, rr in groups.items()
    }
    equal(
        inventory,
        q.c.read(q.BASE / "annotation_inventory.json")["raw_observed_patterns"],
        "annotation inventory",
    )
    q.require(
        draw_values == 840000
        and common_views == 24
        and pair_views == 29
        and gate_views == 126,
        "Review inventory incomplete",
    )
    q.r.verify(receipt["sources"])
    q.r.verify(receipt["outputs"])
    q.save(
        q.BASE / "independent_review.json",
        dict(
            status="all_delivered_arrays_quality_metrics_joint_answers_draws_and_case_selection_passed",
            sources=q.bindings([Path(__file__), q.BASE / "analysis_complete.json"]),
            numeric_values_checked=checked,
            max_absolute_difference=maximum,
            diagnostic_draw_values=draw_values,
            own_setting_views=own_views,
            shared_condition_views=common_views,
            fine_pair_views=pair_views,
            joint_populations=2,
            part3_gate_views_reconciled=gate_views,
            case_slots=len(selected),
            case_predictions=sum(len(x["predictions"]) for x in cases),
            manual_case_review_pending=True,
            independent_method="Direct Python label sets and case loops; all arrays rebuilt from delivered result cache; cluster draws by direct sampled cluster-sum indexing, independent of producer weighted matrix product.",
            new_model_calls=0,
        ),
    )
    print(
        "Independent Part 4 checks passed:",
        checked,
        "numeric values; max difference",
        maximum,
        flush=True,
    )


if __name__ == "__main__":
    with q.r.original.offline(), threadpool_limits(limits=2):
        run()

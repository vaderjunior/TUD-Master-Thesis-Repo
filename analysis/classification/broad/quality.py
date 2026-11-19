import argparse
from collections import Counter
import hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.classification.broad import errors as errors

main, reader, c, r = errors.main, errors.reader, errors.c, errors.r
ROOT = reader.ROOT
BASE = ROOT / "experiments/astra/sq2_broad_quality_review_20261104_v1"
CONDITIONS = reader.CONDITIONS
require, bindings, safe = reader.require, reader.bindings, errors.safe
FIELDS = {
    "hate_type": "hate_types",
    "target_group": "target_groups",
    "severity": "severity",
    "legal": "legal",
}
BOOT = (
    "exact_set",
    "gate_aware_exact",
    "extra_labels",
    "missing_labels",
    "sample_jaccard",
)
DRAWS, SEED = 10000, 20261008


def save(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE.resolve()), "Output outside Part 4")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(c.encoded(safe(value)))


def array(path, **values):
    require(path.resolve().is_relative_to(BASE.resolve()), "Array outside Part 4")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, **values)


def div(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.full(a.shape, np.nan, dtype=float), where=b != 0)


def row_quality(y, p, g, legal=False):
    y, p, g = np.asarray(y, bool), np.asarray(p, bool), np.asarray(g, bool)
    require(
        y.ndim == 2
        and p.ndim == 3
        and p.shape[1:] == y.shape
        and g.shape == p.shape[:2],
        "Wrong quality axes",
    )
    if not legal:
        require(not p[~g].any(), "Delivered negative gate has fine labels")
    tp = (p & y).sum(2)
    extra = (p & ~y).sum(2)
    missing = (~p & y).sum(2)
    exact = (extra + missing) == 0
    observed = y.sum(1)
    predicted = p.sum(2)
    union = tp + extra + missing
    jaccard = np.divide(tp, union, out=np.ones(tp.shape, float), where=union > 0)
    f1 = np.divide(
        2 * tp,
        2 * tp + extra + missing,
        out=np.ones(tp.shape, float),
        where=(2 * tp + extra + missing) > 0,
    )
    out = dict(
        exact_set=exact,
        gate_aware_exact=exact if legal else exact & g,
        extra_labels=extra,
        missing_labels=missing,
        predicted_labels=predicted,
        gold_labels=np.broadcast_to(observed, p.shape[:2]),
        sample_jaccard=jaccard,
        sample_f1=f1,
        extra_only=(extra > 0) & (missing == 0),
        missing_only=(extra == 0) & (missing > 0),
        mixed=(extra > 0) & (missing > 0),
        has_extra=extra > 0,
        has_missing=missing > 0,
        true_positive_labels=tp,
        empty_gold=np.broadcast_to(observed == 0, g.shape),
        predicted_empty=predicted == 0,
        gate_pass=g,
        empty_exact_wrong_gate=(
            (observed == 0) & exact & ~g if not legal else np.zeros_like(g)
        ),
        positive_gold_blocked=(observed > 0) & ~g if not legal else np.zeros_like(g),
    )
    require(
        np.all(
            sum(
                out[k].astype(int)
                for k in ("exact_set", "extra_only", "missing_only", "mixed")
            )
            == 1
        ),
        "Error partition incomplete",
    )
    return out


def summarize(y, p, g, labels, macro_labels, legal=False):
    require(len(p) == 3 and p.shape[1] > 0, "Three nonempty repeats required")
    q = row_quality(y, p, g, legal)
    reps = []
    for i in range(3):
        tp = (p[i].astype(bool) & y.astype(bool)).sum(0)
        fp = (p[i].astype(bool) & ~y.astype(bool)).sum(0)
        fn = (~p[i].astype(bool) & y.astype(bool)).sum(0)
        f = div(2 * tp, 2 * tp + fp + fn)
        conventional = np.nan_to_num(f, nan=0.0)
        per_label = []
        for j, label in enumerate(labels):
            per_label.append(
                dict(
                    label=label,
                    gold_support=int(y[:, j].sum()),
                    predicted=int(p[i, :, j].sum()),
                    tp=int(tp[j]),
                    fp=int(fp[j]),
                    fn=int(fn[j]),
                    precision=safe(div(tp[j], tp[j] + fp[j])),
                    recall=safe(div(tp[j], tp[j] + fn[j])),
                    f1=float(conventional[j]),
                    f1_denominator_zero=bool(2 * tp[j] + fp[j] + fn[j] == 0),
                    included_in_original_macro=label in macro_labels,
                )
            )
        d = dict(
            n=len(y),
            metrics={k: float(v[i].mean()) for k, v in q.items()},
            counts={k: safe(v[i].sum()) for k, v in q.items()},
            per_label=per_label,
            macro_f1=(
                float(conventional[[labels.index(l) for l in macro_labels]].mean())
                if macro_labels
                else None
            ),
            exact_given_gate=(
                safe(div((q["exact_set"][i] & g[i]).sum(), g[i].sum()))
                if not legal
                else None
            ),
            cardinality=np.asarray(
                [
                    [
                        int(((y.sum(1) == a) & (p[i].sum(1) == b)).sum())
                        for b in range(len(labels) + 1)
                    ]
                    for a in range(len(labels) + 1)
                ]
            ).tolist(),
        )
        reps.append(d)
    means = {k: float(np.mean([v["metrics"][k] for v in reps])) for k in q}
    pl = []
    for j, label in enumerate(labels):
        one = {
            "label": label,
            "included_in_original_macro": label in macro_labels,
            "gold_support": int(y[:, j].sum()),
        }
        for k in ("predicted", "tp", "fp", "fn", "precision", "recall", "f1"):
            vv = [v["per_label"][j][k] for v in reps]
            one[k] = float(np.mean(vv)) if all(v is not None for v in vv) else None
            if k in ("precision", "recall"):
                one[k + "_undefined_repeats"] = sum(v is None for v in vv)
        pl.append(one)
    return dict(
        repeats=reps,
        mean=means,
        sd={k: float(np.std([v["metrics"][k] for v in reps], ddof=1)) for k in q},
        per_label_mean=pl,
    )


def detail(y, p, g, labels, dim):
    out = []
    for i in range(3):
        if dim == "severity":
            require(
                labels == ["low", "medium", "high"]
                and np.all(y.sum(1) == 1)
                and np.all(p[i].sum(1) <= 1),
                "Severity not ordinal single class",
            )
            emitted = p[i].sum(1) > 0
            gold = y.argmax(1)
            pred = p[i].argmax(1)
            delta = pred - gold
            masks = {
                "gate_closed": ~g[i],
                "gate_open_missing": g[i] & ~emitted,
                "correct": g[i] & emitted & (delta == 0),
                "under_one": g[i] & emitted & (delta == -1),
                "under_two": g[i] & emitted & (delta == -2),
                "over_one": g[i] & emitted & (delta == 1),
                "over_two": g[i] & emitted & (delta == 2),
            }
            require(
                np.all(sum(v.astype(int) for v in masks.values()) == 1),
                "Severity partition incomplete",
            )
            guesses = np.where(~g[i], 0, np.where(~emitted, 1, pred + 2))
            confusion = [
                [int(((gold == a) & (guesses == b)).sum()) for b in range(5)]
                for a in range(3)
            ]
            out.append(
                dict(
                    columns=["gate_closed", "gate_open_missing"] + labels,
                    confusion=confusion,
                    counts={k: int(v.sum()) for k, v in masks.items()},
                    conditional_ordinal_mae=safe(
                        div(np.abs(delta[emitted]).sum(), emitted.sum())
                    ),
                    emitted=int(emitted.sum()),
                    note="Ordinal distance is conditional on an emitted class; missing or blocked is not low severity.",
                )
            )
        elif dim == "legal":
            positive = y.any(1)
            flag = p[i].any(1)
            exact = (y == p[i]).all(1)
            out.append(
                dict(
                    known_empty=int((~positive).sum()),
                    annotated_positive=int(positive.sum()),
                    empty_gold_flagged=int((~positive & flag).sum()),
                    empty_gold_flagged_rate=safe(
                        div((~positive & flag).sum(), (~positive).sum())
                    ),
                    positive_gold_unflagged=int((positive & ~flag).sum()),
                    positive_gold_unflagged_rate=safe(
                        div((positive & ~flag).sum(), positive.sum())
                    ),
                    any_legal_tp=int((positive & flag).sum()),
                    any_legal_fp=int((~positive & flag).sum()),
                    any_legal_fn=int((positive & ~flag).sum()),
                    any_legal_tn=int((~positive & ~flag).sum()),
                    by_predicted_hate={
                        str(b): dict(
                            n=int((g[i] == b).sum()),
                            exact=int((exact & (g[i] == b)).sum()),
                            legal_present=int((flag & (g[i] == b)).sum()),
                            gold_positive=int((positive & (g[i] == b)).sum()),
                        )
                        for b in (False, True)
                    },
                )
            )
    return out


def bootstrap(profile, clusters, keys):

    require(
        profile.ndim == 2
        and len(profile) == len(clusters)
        and profile.shape[1] == len(keys),
        "Bootstrap shape mismatch",
    )
    unique = sorted(set(clusters))
    lookup = {k: i for i, k in enumerate(unique)}
    ix = [lookup[k] for k in clusters]
    totals = np.zeros((len(unique), len(keys)), float)
    np.add.at(totals, ix, profile)
    sizes = np.bincount(ix, minlength=len(unique))
    rng = np.random.default_rng(SEED)
    draws = []
    for start in range(0, DRAWS, 64):
        n = min(64, DRAWS - start)
        picks = rng.choice(list(range(len(unique))), (n, len(unique)), replace=True)
        w = np.stack([np.bincount(v, minlength=len(unique)) for v in picks]).astype(
            float
        )
        draws.append((w @ totals) / (w @ sizes)[:, None])
    draws = np.concatenate(draws)
    point = profile.mean(0)
    return (
        dict(
            metrics={
                key: dict(
                    difference=float(point[j]),
                    ci95=np.quantile(draws[:, j], [0.025, 0.975]).tolist(),
                    role="supporting pointwise; fixed repeats and original primary F1 family unchanged",
                )
                for j, key in enumerate(keys)
            },
            n=len(profile),
            n_clusters=len(unique),
            draws=DRAWS,
            seed=SEED,
        ),
        draws,
    )


def bounds(q, valid, key):
    return [
        [
            float(q[key][i, valid[i]].sum() / valid.shape[1]),
            float((q[key][i, valid[i]].sum() + (~valid[i]).sum()) / valid.shape[1]),
        ]
        for i in range(3)
    ]


def event_counts(a, b):
    require(a.shape == b.shape and a.shape[0] == 3, "Wrong exact transition shape")
    fixed = (a[None, :, :] & ~b[:, None, :]).sum(2)
    lost = (~a[None, :, :] & b[:, None, :]).sum(2)
    return dict(
        fixed=float(fixed.mean()),
        lost=float(lost.mean()),
        net=float(fixed.mean() - lost.mean()),
        stable_fixed=int((a.all(0) & ~b.any(0)).sum()),
        stable_lost=int((~a.any(0) & b.all(0)).sum()),
        stable_both_wrong=int((~a.any(0) & ~b.any(0)).sum()),
        stable_both_right=int((a.all(0) & b.all(0)).sum()),
        note="All nine cross-repeat pairs with equal weight, not temporal fixes or independent texts.",
    )


def choose(name, ids, a, b):
    chosen = []
    masks = {
        "stable_fix": a.all(0) & ~b.any(0),
        "stable_regression": ~a.any(0) & b.all(0),
        "stable_shared_error": ~a.any(0) & ~b.any(0),
    }
    for category, mask in masks.items():
        candidates = [key for key, on in zip(ids, mask) if on]
        digest = lambda key: hashlib.sha256(
            f"sq2-broad-part4-v1|{name}|{category}|{key}".encode()
        ).hexdigest()
        key = min(candidates, key=lambda x: (digest(x), x)) if candidates else None
        chosen.append(
            dict(
                endpoint=name,
                reference="rag_small",
                category=category,
                candidates=len(candidates),
                key=key,
                selection_sha256=digest(key) if key else None,
            )
        )
    return chosen


def prepare():
    save(
        BASE / "analysis_contract.json",
        dict(
            status="part4_scope_recorded_before_quality_calculation",
            created_utc=c.utc(),
            sources=bindings(
                [
                    reader.PLAN,
                    reader.BASE / "part1_complete.json",
                    main.BASE / "part2_complete.json",
                    errors.BASE / "part3_complete.json",
                    Path(__file__),
                ]
            ),
            scope="All eight fine/legal endpoints. Three Qwen conditions, all repeats, own-valid and common-nine views; original historical encoder fine pairs are descriptive on their original six-setting masks.",
            labels="All taxonomy labels, frozen macro support >=10 from full gold roster. Keep rare/absent labels, undefined precision/recall and conventional zero-denominator F1 separately.",
            metrics=list(BOOT)
            + [
                "sample_f1",
                "exact/extra-only/missing-only/mixed partitions",
                "extra/missing incidence and cardinality matrix",
                "conditional exactness given gate",
                "severity confusion and ordinal distance conditional on emission",
                "independent legal false flags and missed positives",
            ],
            gates="Fine gold requires hateful and annotated; gate-aware exact requires delivered hate=true and exact fine set. Empty gold is observed, never missing. Independent legal exactness does not require a hate gate.",
            populations="No fabricated all-field gold. Joint hate+target+severity only on the two actual MHS intersections (DEV 1606, TEST 1523). Full roster and each actual annotation-pattern inventory retained; do not average different endpoints into one full-answer score.",
            comparisons="Full versus small/zero: supporting 95% cluster bootstrap for five listed per-row metrics, and joint exactness. Original common-six masks, 10000 draws, seed 20261008, natural text/HateCheck connected components. No new primary family. No per-label significance tests. Historical compatible encoder pairs are descriptive, not equal-budget GPU substitutes.",
            uncertainty="Conditional on fixed three repeats and observed data; full-roster binary exactness failure bounds separate from sampling intervals. No invalid-as-empty or dropped undefined values.",
            joint="Disjoint gate-closed/target-only-wrong/severity-only-wrong/both-wrong/joint-correct outcomes; all-nine exact fix/loss events.",
            normalization="Score delivered outputs only. Reconstruct pre-final-gate aggregation from parsed valid inner votes as a supporting structural diagnostic, without scoring it or claiming recovery of parser-normalized raw assertions. Preserve saved parser normalization counts; no raw-logit or counterfactual gate repair.",
            cases="Freeze one stable exact fix/regression/shared error versus small for EN TEST types/targets/severity, German legal, and EN TEST joint (15 slots). Min SHA256 sq2-broad-part4-v1|endpoint|category|key. Add one German target false-gate empty-set inflation case by min SHA256 sq2-broad-part4-empty|key among full-all-three cases. Keep empty slots and duplicates. Inspect all nine outputs; show first-repeat full/small rationales, not chosen for persuasiveness.",
            interpretation="Structured annotation agreement is not faithful reasoning or real-world legal advice. Severity is mapped perceived-intensity terciles. Different annotation sources/definitions and missing fields limit complete-answer claims.",
            deferred=[
                "source/function breakdown Part 5",
                "retrieval contexts Part 6",
                "replay/whitespace/partial-vote/weighting sensitivity Part 7",
                "resources Part 8",
                "synthesis Part 9",
            ],
            prohibited=[
                "new Qwen calls",
                "training/inference/model loading",
                "retrieval reruns",
                "server access",
                "source/evidence or writer edits",
                "relabelling/retuning/deletion",
            ],
        ),
    )


def run():
    contract = c.read(BASE / "analysis_contract.json")
    r.verify(contract["sources"])
    tests = c.read(BASE / "tests.json")
    require(tests["status"] == "passed", "Tests required")
    r.verify(tests["sources"])
    for p in (main.BASE / "part2_complete.json", errors.BASE / "part3_complete.json"):
        seal = c.read(p)
        r.verify(seal["outputs"])
        r.verify(seal["original_sources"])
    groups, _, runs, endpoints, _ = reader.load()
    before = bindings(
        [
            Path(__file__),
            BASE / "analysis_contract.json",
            BASE / "tests.json",
            errors.BASE / "part3_complete.json",
            main.BASE / "part2_complete.json",
            reader.BASE / "part1_complete.json",
        ]
    )
    save(BASE / "preservation_before.json", before)
    all_results = {}
    arrays = {}
    case_slots = []
    draw_values = 0
    for name, ep in endpoints.items():
        phase, dim = name.split("/")
        if dim == "hate":
            continue
        old = c.read(main.BASE / "endpoints" / main.slug(name) / "results.json")
        settings = [tuple(v) for v in old["settings"]]
        with np.load(ROOT / old["encoded_path"], allow_pickle=False) as z:
            y = z["gold"].copy()
            p = z["predicted"].copy()
            v = z["valid"].copy()
        ids = ep["scorable_ids"]
        g = np.array(
            [
                [
                    (
                        bool(runs[cond, rep][key]["hate"])
                        if runs[cond, rep][key] is not None
                        else False
                    )
                    for key in ids
                ]
                for cond, rep in settings
            ],
            bool,
        )
        labels = ep["all_labels"]
        macro = ep["macro_labels"]
        legal = dim == "legal"
        ix9 = [settings.index((cond, i)) for cond in CONDITIONS for i in range(3)]
        mask9 = v[ix9].all(0)
        out = dict(
            endpoint=name,
            labels=labels,
            macro_labels=macro,
            full_gold_support=ep["gold_support"],
            eligible=len(ids),
            common_nine_n=int(mask9.sum()),
            own_valid={},
            common_nine={},
            pairs=[],
        )
        for cond in ep["available_conditions"]:
            ix = [settings.index((cond, i)) for i in range(3)]
            q = row_quality(y, p[ix], g[ix], legal)
            own = []
            for j, s in enumerate(ix):
                m = v[s]
                one = summarize(
                    y[m],
                    np.repeat(p[s : s + 1, m], 3, 0),
                    np.repeat(g[s : s + 1, m], 3, 0),
                    labels,
                    macro,
                    legal,
                )["repeats"][0]
                ref = old["own_valid"][cond]["repeats"][j]["macro_f1_valid_only"]
                require(
                    (ref is None and one["macro_f1"] is None)
                    or (ref is not None and abs(one["macro_f1"] - ref) < 2e-15),
                    "Part 2 own F1 differs",
                )
                one.update(
                    eligible=len(ids),
                    invalid=int((~m).sum()),
                    exact_bounds=bounds(q, v[ix], "exact_set")[j],
                    gate_exact_bounds=bounds(q, v[ix], "gate_aware_exact")[j],
                    detail=(
                        detail(
                            y[m],
                            np.repeat(p[s : s + 1, m], 3, 0),
                            np.repeat(g[s : s + 1, m], 3, 0),
                            labels,
                            dim,
                        )[0]
                        if dim in ("severity", "legal")
                        else None
                    ),
                )
                own.append(one)
            out["own_valid"][cond] = own
            if cond in CONDITIONS:
                out["common_nine"][cond] = summarize(
                    y[mask9], p[ix][:, mask9], g[ix][:, mask9], labels, macro, legal
                )
                out["common_nine"][cond]["detail"] = detail(
                    y[mask9], p[ix][:, mask9], g[ix][:, mask9], labels, dim
                )
        for prev in old["contrasts"]:
            right = prev["right"]
            ix = [
                settings.index((cond, i))
                for cond in ("rag_full", right)
                for i in range(3)
            ]
            m = v[ix].all(0)
            require(
                [key for key, on in zip(ids, m) if on] == prev["common_valid_ids"],
                "Pair population differs",
            )
            q = row_quality(y[m], p[ix][:, m], g[ix][:, m], legal)
            left = summarize(
                y[m], p[ix[:3]][:, m], g[ix[:3]][:, m], labels, macro, legal
            )
            ref = summarize(
                y[m], p[ix[3:]][:, m], g[ix[3:]][:, m], labels, macro, legal
            )
            pair = dict(
                reference=right,
                n=int(m.sum()),
                full=left,
                reference_summary=ref,
                exact_events=event_counts(
                    q["gate_aware_exact"][:3], q["gate_aware_exact"][3:]
                ),
                full_roster_gate_exact_difference_bounds=[
                    float(
                        np.mean(
                            [
                                a["gate_exact_bounds"][0]
                                for a in out["own_valid"]["rag_full"]
                            ]
                        )
                        - np.mean(
                            [a["gate_exact_bounds"][1] for a in out["own_valid"][right]]
                        )
                    ),
                    float(
                        np.mean(
                            [
                                a["gate_exact_bounds"][1]
                                for a in out["own_valid"]["rag_full"]
                            ]
                        )
                        - np.mean(
                            [a["gate_exact_bounds"][0] for a in out["own_valid"][right]]
                        )
                    ),
                ],
            )
            if right in ("rag_small", "zero_shot"):
                profile = np.stack(
                    [q[k][:3].mean(0) - q[k][3:].mean(0) for k in BOOT], axis=1
                )
                info, draws = bootstrap(
                    profile, [cl for cl, on in zip(ep["row_clusters"], m) if on], BOOT
                )
                pair["intervals"] = info
                path = (
                    BASE
                    / "endpoints"
                    / main.slug(name)
                    / ("full_minus_" + right + ".npz")
                )
                array(
                    path, profile=profile, draws=draws, common_indices=np.flatnonzero(m)
                )
                pair["bootstrap_path"] = c.rel(path)
                draw_values += draws.size
                if right == "rag_small" and name in (
                    "en_test/hate_type",
                    "en_test/target_group",
                    "en_test/severity",
                    "de_legal_test/legal",
                ):
                    case_slots += choose(
                        name,
                        [key for key, on in zip(ids, m) if on],
                        q["gate_aware_exact"][:3],
                        q["gate_aware_exact"][3:],
                    )
            else:
                pair["intervals"] = None
            out["pairs"].append(pair)
        path = BASE / "endpoints" / main.slug(name) / "encoded.npz"
        array(path, gold=y, predicted=p, valid=v, gate=g, common_nine=mask9)
        out["encoded_path"] = c.rel(path)
        out["settings"] = [list(s) for s in settings]
        out["ids"] = ids
        save(BASE / "endpoints" / main.slug(name) / "results.json", out)
        all_results[name] = out
        arrays[name] = (y, p, g, v, settings, ids)
        print("Fine quality:", name, flush=True)
    joint = {}
    for phase in ("larger_dev", "en_test"):
        ty, tp, tg, tv, settings, tids = arrays[phase + "/target_group"]
        sy, sp, sg, sv, ss, sids = arrays[phase + "/severity"]
        require(ss == settings and set(sids) <= set(tids), "Joint identity mismatch")
        ix = [tids.index(key) for key in sids]
        tp, tg, tv = tp[:, ix], tg[:, ix], tv[:, ix]
        ty = ty[ix]
        require(
            np.array_equal(tg, sg) and np.array_equal(tv, sv),
            "Joint gates/validity differ",
        )
        te = (tp == ty).all(2)
        se = (sp == sy).all(2)
        q = dict(
            joint_exact=sg & te & se,
            gate_closed=~sg,
            target_only_wrong=sg & ~te & se,
            severity_only_wrong=sg & te & ~se,
            both_wrong=sg & ~te & ~se,
        )
        require(
            np.all(sum(x.astype(int) for x in q.values()) == 1),
            "Joint partition incomplete",
        )
        ix9 = [settings.index((cond, i)) for cond in CONDITIONS for i in range(3)]
        mask9 = sv[ix9].all(0)
        summary = lambda ix, m: dict(
            repeats=[{k: float(v[j, m].mean()) for k, v in q.items()} for j in ix],
            mean={k: float(v[ix][:, m].mean()) for k, v in q.items()},
        )
        out = dict(
            eligible=len(sids),
            ids=sids,
            fields=["hate", "target_group", "severity"],
            common_nine_n=int(mask9.sum()),
            settings=[list(x) for x in settings],
            own_valid={},
            common_nine={},
            pairs=[],
        )
        for cond in endpoints[phase + "/severity"]["available_conditions"]:
            ii = [settings.index((cond, j)) for j in range(3)]
            own = []
            for j, s in enumerate(ii):
                own.append(
                    dict(
                        n=int(sv[s].sum()),
                        invalid=int((~sv[s]).sum()),
                        metrics=summary([s], sv[s])["mean"],
                        bounds=bounds({"exact": q["joint_exact"][ii]}, sv[ii], "exact")[
                            j
                        ],
                    )
                )
            out["own_valid"][cond] = own
            if cond in CONDITIONS:
                out["common_nine"][cond] = summary(ii, mask9)
        for right in [
            p["reference"] for p in all_results[phase + "/severity"]["pairs"]
        ]:
            ii = [
                settings.index((cond, j))
                for cond in ("rag_full", right)
                for j in range(3)
            ]
            m = sv[ii].all(0)
            pair = dict(
                reference=right,
                n=int(m.sum()),
                full=summary(ii[:3], m),
                reference_summary=summary(ii[3:], m),
                exact_events=event_counts(
                    q["joint_exact"][ii[:3]][:, m], q["joint_exact"][ii[3:]][:, m]
                ),
            )
            pair["full_roster_difference_bounds"] = [
                float(
                    np.mean([v["bounds"][0] for v in out["own_valid"]["rag_full"]])
                    - np.mean([v["bounds"][1] for v in out["own_valid"][right]])
                ),
                float(
                    np.mean([v["bounds"][1] for v in out["own_valid"]["rag_full"]])
                    - np.mean([v["bounds"][0] for v in out["own_valid"][right]])
                ),
            ]
            if right in ("rag_small", "zero_shot"):
                profile = (
                    q["joint_exact"][ii[:3]][:, m].mean(0)
                    - q["joint_exact"][ii[3:]][:, m].mean(0)
                )[:, None]
                info, draws = bootstrap(
                    profile,
                    [
                        cl
                        for cl, on in zip(
                            endpoints[phase + "/severity"]["row_clusters"], m
                        )
                        if on
                    ],
                    ["joint_exact"],
                )
                pair["intervals"] = info
                path = BASE / "joint" / phase / ("full_minus_" + right + ".npz")
                array(
                    path, profile=profile, draws=draws, common_indices=np.flatnonzero(m)
                )
                pair["bootstrap_path"] = c.rel(path)
                draw_values += draws.size
                if phase == "en_test" and right == "rag_small":
                    case_slots += choose(
                        phase + "/joint",
                        [key for key, on in zip(sids, m) if on],
                        q["joint_exact"][ii[:3]][:, m],
                        q["joint_exact"][ii[3:]][:, m],
                    )
            else:
                pair["intervals"] = None
            out["pairs"].append(pair)
        path = BASE / "joint" / phase / "encoded.npz"
        array(path, **q, valid=sv, common_nine=mask9)
        out["encoded_path"] = c.rel(path)
        joint[phase] = out
        print("Joint quality:", phase, flush=True)
    save(BASE / "joint_results.json", joint)
    dy, dp, dg, dv, ds, dids = arrays["de_test/target_group"]
    ii = [ds.index(("rag_full", j)) for j in range(3)]
    candidates = [
        key
        for j, key in enumerate(dids)
        if dv[ii, j].all() and not dy[j].any() and not dg[ii, j].any()
    ]
    digest = lambda key: hashlib.sha256(
        ("sq2-broad-part4-empty|" + key).encode()
    ).hexdigest()
    key = min(candidates, key=lambda k: (digest(k), k)) if candidates else None
    case_slots.append(
        dict(
            endpoint="de_test/target_group",
            reference="rag_small",
            category="empty_exact_wrong_gate_all_full_repeats",
            candidates=len(candidates),
            key=key,
            selection_sha256=digest(key) if key else None,
        )
    )
    for i, slot in enumerate(case_slots, 1):
        slot["case_id"] = f"Q{i:02d}"
    save(BASE / "case_selection.json", dict(rule=contract["cases"], slots=case_slots))
    keys = {v["key"] for v in case_slots if v["key"]}
    cache = {key: [] for key in keys}
    normal = {}
    focused = {}
    sources = c.read(reader.BASE / "item_bindings.json")
    for row in reader.rows(reader.BASE / "predictions.jsonl"):
        cell = normal.setdefault(
            (row["phase"], row["condition"], row["repeat"]), Counter()
        )
        cell["predictions"] += 1
        cell["parser_normalisations"] += row["normalisations"]
        cell["parser_gate_normalised"] += row["gate_normalised"]
        result = row["result"]
        votes = [v for v in row["votes"] if v is not None]
        if result is None:
            cell["invalid"] += 1
        else:
            cell["valid"] += 1
            pre = {
                field: sorted(
                    k
                    for k, n in Counter(k for v in votes for k in v[field]).items()
                    if 2 * n > len(votes)
                )
                for field in ("hate_type", "target_group", "legal")
            }
            sevs = sorted(
                ["low", "medium", "high"].index(v["severity"])
                for v in votes
                if v["severity"] is not None
            )
            pre["severity"] = (
                ["low", "medium", "high"][sevs[(len(sevs) - 1) // 2]] if sevs else None
            )
            if not result["hate"]:
                cell["final_negative"] += 1
                for field in ("hate_type", "target_group", "severity"):
                    cell["final_gate_clears_" + field] += int(bool(pre[field]))
                    pre[field] = None if field == "severity" else []
                cell["negative_with_positive_inner_vote"] += int(
                    any(v["hate"] for v in votes)
                )
            else:
                for field in ("hate_type", "target_group", "severity"):
                    cell["positive_empty_" + field] += int(not pre[field])
            require(
                all(pre[field] == result[field] for field in pre),
                "Pre/final gate diagnostic reconstruction differs",
            )
        if row["key"] in keys:
            path = Path(row["result_path"])
            require(
                c.sha(path) == sources[row["result_path"]],
                "Case original result changed",
            )
            focused[row["result_path"]] = sources[row["result_path"]]
            cache[row["key"]].append(row)
    save(
        BASE / "normalization_diagnostic.json",
        dict(
            rows=[
                dict(population=k[0], condition=k[1], repeat=k[2], **v)
                for k, v in sorted(normal.items())
            ],
            note="Parsed inner votes and delivered aggregation only. Parser gate-normalization flags do not preserve removed raw assertions in this cache. No pre-normalization accuracy or gate-repair counterfactual is scored.",
        ),
    )
    gold = {v["key"]: v for rr in groups.values() for v in rr}
    cases = [
        slot
        | dict(
            status="selected" if slot["key"] else "empty category",
            gold=gold[slot["key"]] if slot["key"] else None,
            predictions=cache[slot["key"]] if slot["key"] else [],
        )
        for slot in case_slots
    ]
    require(
        all(len(v["predictions"]) == 9 for v in cases if v["key"]),
        "Incomplete selected outputs",
    )
    save(BASE / "cases.json", cases)
    save(BASE / "focused_source_bindings.json", focused)
    inventory = {
        phase: dict(
            Counter(
                "+".join(
                    (["hate"] if row["gate"] is not None else [])
                    + [dim for dim, field in FIELDS.items() if row[field] is not None]
                )
                for row in rr
            )
        )
        for phase, rr in groups.items()
    }
    save(
        BASE / "annotation_inventory.json",
        dict(
            raw_observed_patterns=inventory,
            eligible_endpoints={
                name: dict(
                    n=(
                        ep["scorable_rows"]
                        if "scorable_rows" in ep
                        else len(ep["scorable_ids"])
                    ),
                    missing=ep["annotation_missing"],
                    observed_empty=ep["observed_empty_eligible"],
                )
                for name, ep in endpoints.items()
                if ep["dimension"] != "hate"
            },
            jointly_annotated_target_severity={
                phase: len(joint[phase]["ids"]) for phase in joint
            },
            note="No all-field exact score; raw observed patterns retain empty lists and exclude None.",
        ),
    )
    r.verify(before)
    outputs = bindings([p for p in BASE.rglob("*") if p.is_file()])
    save(
        BASE / "analysis_complete.json",
        dict(
            status="part4_calculation_complete_independent_and_manual_review_pending",
            sources=before,
            outputs=outputs,
            fine_endpoints=len(all_results),
            fine_pairs=sum(len(v["pairs"]) for v in all_results.values()),
            joint_populations=len(joint),
            diagnostic_draw_values=draw_values,
            case_slots=len(cases),
            selected_cases=sum(bool(v["key"]) for v in cases),
            unique_case_inputs=len(keys),
            direct_source_hashes=len(focused),
            new_Qwen_calls=0,
            new_training=0,
        ),
    )
    print("Part 4 calculations saved; independent review is next.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["prepare", "run"])
    args = p.parse_args()
    with r.original.offline(), threadpool_limits(limits=2):
        {"prepare": prepare, "run": run}[args.mode]()

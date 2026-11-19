from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.classification.supervised import archive as r
from analysis.classification.supervised import metrics as m
from analysis.classification.supervised import controls as c
from analysis.classification.supervised import type_update as s

BASE = r.ROOT / "experiments/astra/sq2_gpu_errors_review_20261103_v1"
CONTROLS = c.BASE / "runs/b32f91704667"
ROSTER = (
    r.ROOT
    / "experiments/astra/offline_closure_20261006_v1/evaluation_preparation/broad_roster.jsonl"
)
PAIRS = (
    ("tuned/selected", "bge/selected"),
    ("frozen_roberta/selected", "bge/selected"),
    ("tuned/selected", "fixed_balanced/selected"),
    ("fixed_balanced/default", "fixed_unweighted/default"),
    ("fixed_balanced/selected", "fixed_unweighted/selected"),
) + tuple((f + "/selected", f + "/default") for f in c.METHODS)
CROSS = (("tuned", "bge"), ("frozen_roberta", "bge"), ("tuned", "fixed_balanced"))
save = c.save


def source(path, bindings):
    rel = str(path.relative_to(r.ROOT))
    expected = r.read(c.BASE / "part3_complete.json")["artifacts"][rel]
    r.checked_file(path, expected, bindings)
    return r.read(path)


def old_arrays(endpoint, bindings):
    path = CONTROLS / (m.slug(endpoint) + "_arrays.npz")
    rel = str(path.relative_to(r.ROOT))
    expected = r.read(c.BASE / "part3_complete.json")["artifacts"][rel]
    r.checked_file(path, expected, bindings)
    return r.arrays_from_bytes(path.read_bytes())


def divide(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.zeros(a.shape, dtype=float), where=b != 0)


def binary_counts(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    r.require(
        y.ndim == 1
        and p.ndim == 2
        and p.shape[1] == len(y)
        and len(y) > 0
        and np.isin(y, [0, 1]).all()
        and np.isin(p, [0, 1]).all(),
        "Invalid binary error inputs.",
    )
    y = y.astype(bool)
    p = p.astype(bool)
    return np.stack(
        [(p & y).sum(1), (p & ~y).sum(1), (~p & y).sum(1), (~p & ~y).sum(1)], axis=-1
    )


def binary_metrics(counts):
    r.require(
        counts.shape[-1] == 4 and (counts >= 0).all(), "Invalid confusion counts."
    )
    tp, fp, fn, tn = np.moveaxis(counts, -1, 0)
    n = tp + fp + fn + tn
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        errors=fp + fn,
        fpr=divide(fp, fp + tn),
        fnr=divide(fn, fn + tp),
        precision=divide(tp, tp + fp),
        recall=divide(tp, tp + fn),
        specificity=divide(tn, tn + fp),
        accuracy=divide(tp + tn, n),
        predicted_hate=divide(tp + fp, n),
        positive_f1=divide(2 * tp, 2 * tp + fp + fn),
        macro_f1=(divide(2 * tp, 2 * tp + fp + fn) + divide(2 * tn, 2 * tn + fp + fn))
        / 2,
    )


def paired_events(y, before, after):
    r.require(
        before.shape == after.shape and before.ndim == 2 and before.shape[1] == len(y),
        "Invalid paired event axes.",
    )
    a, b = before.astype(bool), after.astype(bool)
    y = np.asarray(y, dtype=bool)
    return dict(
        fp_fixed=a & ~b & ~y,
        fp_introduced=~a & b & ~y,
        fn_fixed=~a & b & y,
        fn_introduced=a & ~b & y,
    )


def pair_indices(models, after, before, pool):
    am, av = after.split("/")
    bm, bv = before.split("/")
    seeds = r.SEEDS if am in c.FAMILIES or bm in c.FAMILIES else (None,)
    out = []
    for seed in seeds:
        match = []
        for method, variant in ((am, av), (bm, bv)):
            ss = seed if method in c.FAMILIES else None
            ix = [
                j
                for j, x in enumerate(models)
                if x["pool"] == pool
                and x["method"] == method
                and x["variant"] == variant
                and x["seed"] == ss
            ]
            r.require(len(ix) == 1, "Pair is missing or duplicates a fit.")
            match.append(ix[0])
        out.append((match[0], match[1]))
    return out


def fine_flow(y, raw, gate):
    y = np.asarray(y)
    raw = np.asarray(raw)
    gate = np.asarray(gate)
    r.require(
        y.ndim == 2
        and raw.ndim == 3
        and raw.shape[1:] == y.shape
        and gate.shape == raw.shape[:2]
        and np.isin(y, [0, 1]).all()
        and np.isin(raw, [0, 1]).all()
        and np.isin(gate, [0, 1]).all(),
        "Invalid gate-flow axes.",
    )
    y = y.astype(bool)
    raw = raw.astype(bool)
    gate = gate.astype(bool)
    gated = raw & gate[:, :, None]
    fn_head_pass = (y & ~raw & gate[:, :, None]).sum((1, 2))
    fn_head_blocked = (y & ~raw & ~gate[:, :, None]).sum((1, 2))
    fn_gate_only = (y & raw & ~gate[:, :, None]).sum((1, 2))
    fn_total = (y & ~gated).sum((1, 2))
    r.require(
        np.array_equal(fn_total, fn_head_pass + fn_head_blocked + fn_gate_only),
        "Missed-label partition fails.",
    )
    exact_raw = (raw == y).all(2)
    exact_gated = (gated == y).all(2)
    return dict(
        gate_pass=gate.sum(1),
        gate_block=(~gate).sum(1),
        fn_head_pass=fn_head_pass,
        fn_head_blocked=fn_head_blocked,
        fn_gate_only=fn_gate_only,
        fn_total=fn_total,
        tp=(y & gated).sum((1, 2)),
        fp=(~y & gated).sum((1, 2)),
        fp_removed_by_gate=(~y & raw & ~gate[:, :, None]).sum((1, 2)),
        raw_exact_pass=(exact_raw & gate).sum(1),
        raw_exact_blocked=(exact_raw & ~gate).sum(1),
        raw_wrong_pass=(~exact_raw & gate).sum(1),
        raw_wrong_blocked=(~exact_raw & ~gate).sum(1),
        gated_exact=exact_gated.sum(1),
        gate_aware_exact=(exact_gated & gate).sum(1),
        empty_gold_correct_only_after_block=(exact_gated & ~exact_raw & ~gate).sum(1),
        rows=len(y),
        gold_labels=int(y.sum()),
        gold_empty=int((y.sum(1) == 0).sum()),
    )


def count_resamples(y, p, clusters, draws=10000, batch=64):
    unique, inv = np.unique(np.asarray(clusters), return_inverse=True)
    y = np.asarray(y, dtype=bool)
    p = np.asarray(p, dtype=bool)
    terms = np.empty((len(unique), 2 * len(p) + 2), dtype=float)
    for j in range(len(p)):
        terms[:, j] = np.bincount(inv, weights=p[j] & y, minlength=len(unique))
        terms[:, len(p) + j] = np.bincount(
            inv, weights=p[j] & ~y, minlength=len(unique)
        )
    terms[:, -2] = np.bincount(inv, weights=y, minlength=len(unique))
    terms[:, -1] = np.bincount(inv, weights=~y, minlength=len(unique))
    rng = np.random.default_rng(20261020)
    hasher = hashlib.sha256()
    out = np.empty((draws, len(p), 4), dtype=np.int32)
    for start in range(0, draws, batch):
        n = min(batch, draws - start)
        picks = rng.integers(len(unique), size=(n, len(unique)), dtype=np.int64)
        hasher.update(picks.astype("<i8", copy=False).tobytes())
        w = np.asarray(
            [np.bincount(x, minlength=len(unique)) for x in picks], dtype=float
        )
        total = w @ terms
        tp = total[:, : len(p)]
        fp = total[:, len(p) : 2 * len(p)]
        cc = np.stack(
            [tp, fp, total[:, -2, None] - tp, total[:, -1, None] - fp], axis=-1
        )
        r.require(
            np.array_equal(cc, cc.astype(np.int32)),
            "Noninteger resampled confusion count.",
        )
        out[start : start + n] = cc
    return out, dict(
        draws=draws,
        seed=20261020,
        clusters=len(unique),
        picks_sha256=hasher.hexdigest(),
    )


def annotation_stratum(row):
    raw = json.loads(row["raw"]) if isinstance(row["raw"], str) else row["raw"]
    if row["source"] == "mhs":
        score = raw["score"]
        return "mhs/" + (
            "score<=0"
            if score <= 0
            else (
                "0<score<=0.5"
                if score <= 0.5
                else (
                    "0.5<score<=1.0"
                    if score <= 1
                    else "1.0<score<=1.63" if score <= 1.63 else "score>1.63"
                )
            )
        )
    if row["source"] == "hatexplain":
        votes = Counter(raw["votes"])
        label, count = votes.most_common(1)[0]
        r.require(count >= 2, "Unresolved majority in frozen HateXplain row.")
        return "hatexplain/" + label
    r.require(row["source"] == "implicit_hate", "Unplanned binary source.")
    label = raw["stage1_class"]
    r.require(
        label
        in ("not_hate", "explicit_hate", "implicit_hate", "implicit_hate (stg2-only)"),
        "Unknown stage-one class.",
    )
    return "implicit_hate/" + label


def freeze():
    r.offline()
    BASE.mkdir(parents=True, exist_ok=True)
    paths = [
        c.BASE / "part3_complete.json",
        m.BASE / "part2_complete.json",
        r.ROOT
        / "experiments/astra/sq2_type_sensitivity_20261103_v1/supplement_analysis_complete.json",
        ROSTER,
        Path(__file__),
        Path(r.__file__),
        Path(m.__file__),
        Path(c.__file__),
        r.ROOT / "src/hsrag/data/hatexplain.py",
        r.ROOT / "src/hsrag/data/mhs.py",
        r.ROOT / "src/hsrag/data/implicit_hate.py",
    ]
    save(
        BASE / "analysis_contract.json",
        dict(
            part=4,
            utc=datetime.now(timezone.utc).isoformat(),
            question="Which false alarms and missed hate explain the saved method differences, and how do their gates propagate or suppress fine-label answers?",
            prior_exposure="All Parts 1–3 and the five-model supplement have been read. These explanatory checks are supporting and do not alter the original primary family.",
            methods=list(c.METHODS),
            policies=["default", "selected"],
            pools=list(r.POOLS),
            seeds=list(r.SEEDS),
            binary_endpoint="en_test/hate",
            fine_endpoints=list(r.PRIMARY[1:]),
            paired_comparisons=[list(x) for x in PAIRS],
            head_gate_crosses=[list(x) for x in CROSS],
            analysis=[
                "Binary TP/FP/FN/TN, positive precision/recall, FPR/FNR, retained seed and pool variation",
                "Paired fixes versus newly introduced FP/FN on the same texts; averaged model counts, no ensemble and no fake baseline repeats",
                "Fine missed-label partition: head miss with gate open, head miss with gate closed, and correct raw label removed only by gate",
                "Correct and incorrect raw sets passed/blocked; gold-empty exception handled; gate-aware exactness",
                "Cross saved fine heads and saved hate gates within the same pool, and seed where both are neural; average both arithmetic decomposition orders",
                "Fine outputs accompanying false hate alarms on TEST; no per-label truth or F1 invented for unannotated negative rows",
                "Limited source/annotation strata for explanation, with frozen mappings: source, MHS existing severity bands plus <=0/0..0.5, HateXplain majority, implicit/explicit stage-one classes",
                "Error persistence across existing repetitions; do not relabel, tune or discard a seed",
                "Deterministic illustrative cases: 5000 tuned versus BGE four transition classes, two per class across sources when possible; rank by paired frequency then text hash",
                "One example per fine endpoint of correct raw answer blocked and wrong raw answer passed, plus two systematic 1000-type cutoff exactness losses; rank by mean repeat frequency then text hash",
                "Separate saved five-type-model supplement for gate-flow sensitivity; no change to original tables",
            ],
            uncertainty=dict(
                draws=10000,
                seed=20261020,
                interval=0.95,
                paired_normalized_text_clusters=True,
                conditional_on_fixed_fits_and_pools=True,
                supporting_unadjusted=True,
                original_24_primary_family_unchanged=True,
            ),
            exclusions=[
                "No new training/inference/Qwen/server operation or weights",
                "No test threshold search or primary-procedure replacement",
                "Full label/source/HateCheck review remains Part 6",
                "Cases illustrate observed errors, not a prevalence-coded human annotation study or access to model reasoning",
            ],
            inputs={str(p.relative_to(r.ROOT)): r.sha(p) for p in paths},
        ),
    )
    print(
        "Part 4 contract saved before calculating new explanatory outcomes.", flush=True
    )


def extract_full_test(folder, meta, old, bindings):
    study = r.GpuStudy()
    linear = r.LinearStudy(study)
    expected = {x["id"]: j for j, x in enumerate(meta["models"])}
    filled = set()
    matrices = {
        h: {
            k: np.zeros(
                (
                    len(expected),
                    len(meta["ids"]),
                    len(
                        study.package_json("data/train/uniform_compact_s42.json")[
                            "heads"
                        ][h]["labels"]
                    ),
                ),
                dtype=np.uint8,
            )
            for k in ("gated", "raw")
        }
        for h in r.HEADS[1:]
    }
    ix = meta["positions"]
    for pool in r.POOLS:
        with study.pool(pool) as saved:
            for method in c.METHODS:
                for seed in (r.SEEDS if method in c.FAMILIES else (None,)):
                    for policy in ("default", "selected"):
                        pred = (
                            saved.predictions(method, policy, seed)
                            if method in c.FAMILIES
                            else linear.predictions(pool, method, 42, policy)
                        )
                        ident = (
                            f"{pool}/{method}/"
                            + ("fold42" if seed is None else f"s{seed}")
                            + "/"
                            + policy
                        )
                        j = expected[ident]
                        r.require(j not in filled, "Repeated full TEST model.")
                        filled.add(j)
                        r.require(
                            np.array_equal(pred["hate"][ix], old["predictions"][j]),
                            "Saved hate predictions differ from Part 3.",
                        )
                        for head in r.HEADS[1:]:
                            matrices[head]["gated"][j] = pred[head][ix]
                            matrices[head]["raw"][j] = pred[head + "_ungated"][ix]
            save(
                folder / "archive_bindings" / f"{pool}.json", saved.store.read_bindings
            )
        linear.release_pool()
        print(
            pool + ": full TEST outputs verified for false-alarm propagation.",
            flush=True,
        )
    r.require(filled == set(range(len(expected))), "Incomplete full TEST roster.")
    for h, values in matrices.items():
        m.array(folder / f"test_full_{h}.npz", **values)
    save(
        folder / "full_test_source_bindings.json",
        dict(study=study.bindings, linear=linear.bindings),
    )
    return matrices


def analyse_binary(folder, meta, z, records):
    models = meta["models"]
    y = z["truth"][:, 1]
    p = z["predictions"][:, :, 1]
    counts = binary_counts(y, p)
    metrics = binary_metrics(counts)
    draws, info = count_resamples(y, p, meta["clusters"])
    drawmetrics = binary_metrics(draws)
    m.array(folder / "binary_counts.npz", point=counts, draws=draws)
    save(folder / "binary_model_metrics.json", metrics)
    result = dict(
        bootstrap=info,
        absolute={},
        pool={},
        contrasts=[],
        paired_transitions=[],
        sources=[],
        strata=[],
    )
    point_budget = {}
    draw_budget = {}
    for key, value in metrics.items():
        pp, bp = c.aggregate(models, np.asarray(value))
        _, bd = c.aggregate(models, drawmetrics[key])
        point_budget[key] = bp
        draw_budget[key] = bd
        result["absolute"][key] = {k: c.interval(v, bd[k]) for k, v in bp.items()}
        result["pool"][key] = pp

    oldq = r.read(CONTROLS / "en_test__hate_quality.json")
    r.require(
        np.allclose(metrics["macro_f1"], oldq["macro_f1"], rtol=0, atol=1e-15),
        "Binary F1 no longer matches Part 3.",
    )
    for budget in (147, 1000, 5000):
        for after, before in PAIRS:
            a, b = f"{budget}/{after}", f"{budget}/{before}"
            result["contrasts"].append(
                dict(
                    budget=budget,
                    after=after,
                    before=before,
                    differences={
                        k: c.interval(
                            point_budget[k][a] - point_budget[k][b],
                            draw_budget[k][a] - draw_budget[k][b],
                        )
                        for k in metrics
                    },
                )
            )
            event_mean = {
                k: np.zeros(len(y))
                for k in ("fp_fixed", "fp_introduced", "fn_fixed", "fn_introduced")
            }
            pool_counts = {}
            for pool in r.POOLS:
                if m.budget(pool) != budget:
                    continue
                pairs = pair_indices(models, after, before, pool)
                events = paired_events(
                    y, p[[b for a, b in pairs]], p[[a for a, b in pairs]]
                )
                pool_counts[pool] = {k: v.sum(1).mean() for k, v in events.items()}
                for k, v in events.items():
                    event_mean[k] += v.mean(0) / 3
            result["paired_transitions"].append(
                dict(
                    budget=budget,
                    after=after,
                    before=before,
                    mean_counts={k: v.sum() for k, v in event_mean.items()},
                    pool_counts=pool_counts,
                    distinct_texts_with_any_pair_event={
                        k: int((v > 0).sum()) for k, v in event_mean.items()
                    },
                    persistent_texts_all_pairs={
                        k: int(np.isclose(v, 1, atol=1e-12).sum())
                        for k, v in event_mean.items()
                    },
                )
            )
    for field, group in [
        ("sources", [row["source"] for row in records]),
        ("strata", [annotation_stratum(row) for row in records]),
    ]:
        for label in sorted(set(group)):
            mask = np.asarray(group) == label
            cc = binary_counts(y[mask], p[:, mask])
            mm = binary_metrics(cc)
            result[field].append(
                dict(
                    group=label,
                    rows=int(mask.sum()),
                    gold_hate=int(y[mask].sum()),
                    gold_nonhate=int((1 - y[mask]).sum()),
                    budget_metrics={
                        k: c.aggregate(models, val)[1] for k, val in mm.items()
                    },
                    undefined_rates={
                        "fpr": int((1 - y[mask]).sum()) == 0,
                        "fnr": int(y[mask].sum()) == 0,
                    },
                )
            )
    persist = []
    for budget in (147, 1000, 5000):
        for method in c.METHODS:
            for policy in ("default", "selected"):
                ii = [
                    j
                    for j, x in enumerate(models)
                    if x["budget"] == budget
                    and x["method"] == method
                    and x["variant"] == policy
                ]
                err = (p[ii] != y).sum(0)
                persist.append(
                    dict(
                        budget=budget,
                        method=method,
                        policy=policy,
                        fits=len(ii),
                        fp_histogram={
                            str(k): int(((err == k) & (y == 0)).sum())
                            for k in range(len(ii) + 1)
                        },
                        fn_histogram={
                            str(k): int(((err == k) & (y == 1)).sum())
                            for k in range(len(ii) + 1)
                        },
                    )
                )
    result["persistence"] = persist
    save(folder / "binary_results.json", result)
    return metrics


def prepare_fine(endpoint, models, bindings):
    meta = source(CONTROLS / (m.slug(endpoint) + "_metadata.json"), bindings)
    z = old_arrays(endpoint, bindings)
    byid = {x["id"]: j for j, x in enumerate(meta["models"])}
    raw = []
    gated = []
    gates = []
    for model in models:
        j = byid[model["id"]]
        ri = meta["models"][j]["raw_index"]
        rm = meta["raw_models"][ri]
        prob = z["probabilities"][ri]
        if meta["head"] == "severity":
            u = np.eye(prob.shape[1], dtype=np.uint8)[prob.argmax(1)]
        else:
            u = (
                prob >= (0.5 if model["variant"] == "default" else rm["head_cutoff"])
            ).astype(np.uint8)
        g = z["gates"][j]
        r.require(
            np.array_equal(u * g[:, None], z["predictions"][j]),
            "Fine raw/gate reconstruction differs.",
        )
        raw.append(u)
        gated.append(z["predictions"][j])
        gates.append(g)
    return meta, dict(
        truth=z["truth"],
        raw=np.asarray(raw),
        gated=np.asarray(gated),
        gate=np.asarray(gates),
    )


def analyse_cross(folder, endpoint, meta, z, models):
    y = z["truth"]
    matrices = []
    roster = []
    for after, before in CROSS:
        for pool in r.POOLS:
            pairs = pair_indices(
                models, after + "/selected", before + "/selected", pool
            )
            for ai, bi in pairs:
                for variant, hi, gi in [
                    ("before", bi, bi),
                    ("gate_only", bi, ai),
                    ("head_only", ai, bi),
                    ("after", ai, ai),
                ]:
                    roster.append(
                        dict(
                            pair=after + "__" + before,
                            pool=pool,
                            budget=m.budget(pool),
                            seed=models[ai]["seed"],
                            variant=variant,
                            head_model=models[hi]["id"],
                            gate_model=models[gi]["id"],
                        )
                    )
                    matrices.append(z["raw"][hi] * z["gate"][gi, :, None])
    matrices = np.asarray(matrices)
    scores, counts = m.point_scores(y, matrices, meta["supported_indices"])
    draws, info = m.bootstrap(
        y,
        matrices,
        meta["clusters"],
        meta["supported_indices"],
        progress=lambda n: print(
            endpoint + f": crossed-head/gate paired resamples {n}/10000", flush=True
        ),
    )
    output = []
    for after, before in CROSS:
        for budget in (147, 1000, 5000):
            points = {}
            boot = {}
            for variant in ("before", "gate_only", "head_only", "after"):
                ix = [
                    j
                    for j, x in enumerate(roster)
                    if x["pair"] == after + "__" + before
                    and x["budget"] == budget
                    and x["variant"] == variant
                ]
                r.require(
                    len(ix) == (15 if after in c.FAMILIES else 3),
                    "Wrong crossed model weights.",
                )
                points[variant] = scores[ix].mean()
                boot[variant] = draws[:, ix].mean(1)
            d = c.decomposition(
                *(points[v] for v in ("before", "gate_only", "head_only", "after"))
            )
            dd = c.decomposition(
                *(boot[v] for v in ("before", "gate_only", "head_only", "after"))
            )
            output.append(
                dict(
                    after=after,
                    before=before,
                    budget=budget,
                    scores={
                        k: c.interval(value, boot[k]) for k, value in points.items()
                    },
                    contributions={
                        k: c.interval(value, dd[k]) for k, value in d.items()
                    },
                )
            )
    slug = m.slug(endpoint)
    m.array(
        folder / (slug + "_cross.npz"), predictions=matrices, scores=scores, draws=draws
    )
    save(
        folder / (slug + "_cross.json"),
        dict(roster=roster, bootstrap=info, comparisons=output),
    )


def run():
    r.offline()
    plan = r.read(BASE / "analysis_contract.json")
    for name, digest in plan["inputs"].items():
        r.checked_file(r.ROOT / name, digest)
    folder = BASE / "runs" / uuid.uuid4().hex[:12]
    started = time.monotonic()
    save(
        folder / "started.json",
        dict(
            utc=datetime.now(timezone.utc).isoformat(),
            contract_sha256=r.sha(BASE / "analysis_contract.json"),
        ),
    )
    print("RUN " + str(folder), flush=True)
    bindings = {}
    try:
        with threadpool_limits(limits=2):
            meta = source(CONTROLS / "en_test__hate_metadata.json", bindings)
            old = old_arrays("en_test/hate", bindings)
            roster = [
                r.parsed(line)
                for line in ROSTER.read_bytes().splitlines()
                if line.strip()
            ]
            byid = {row["key"]: row for row in roster}
            r.require(len(byid) == len(roster), "Duplicate roster identity.")
            records = [byid[k] for k in meta["ids"]]
            r.require(
                [r.digest(r.normalized(x["text"]).encode("utf8")) for x in records]
                == meta["clusters"]
                and [x["gate"] for x in records]
                == old["truth"][:, 1].astype(bool).tolist(),
                "Hate text/gold identity differs.",
            )
            save(folder / "binary_metadata.json", meta)
            save(
                folder / "binary_records.json",
                [
                    {
                        k: x[k]
                        for k in (
                            "key",
                            "text",
                            "source",
                            "raw",
                            "gate",
                            "hate_types",
                            "target_groups",
                            "severity",
                            "text_cluster",
                        )
                    }
                    for x in records
                ],
            )
            full = extract_full_test(folder, meta, old, bindings)
            metrics = analyse_binary(folder, meta, old, records)
            print(
                "Binary errors, paired transitions and annotation strata complete.",
                flush=True,
            )

            cascade = {}
            neg = ~old["truth"][:, 1].astype(bool)
            p = old["predictions"][:, :, 1].astype(bool)
            for head, values in full.items():
                emitted = values["gated"].sum(2)
                false = p & neg
                cc = dict(
                    false_hate=(false).sum(1),
                    false_hate_with_nonempty_fine=(false & (emitted > 0)).sum(1),
                    fine_labels_on_false_hate=(emitted * false).sum(1),
                    false_hate_with_multiple_labels=(false & (emitted > 1)).sum(1),
                )
                cascade[head] = dict(
                    model_counts=cc,
                    budget_counts={
                        k: c.aggregate(meta["models"], v)[1] for k, v in cc.items()
                    },
                )
            save(folder / "false_alarm_cascade.json", cascade)
            del full
            for endpoint in r.PRIMARY[1:]:
                fine_meta, z = prepare_fine(endpoint, meta["models"], bindings)
                fine_meta["base_models"] = meta["models"]
                save(folder / (m.slug(endpoint) + "_metadata.json"), fine_meta)
                rr = [byid[k] for k in fine_meta["ids"]]
                r.require(
                    all(x["gate"] is True for x in rr),
                    "Fine endpoint includes non-hate gold.",
                )
                save(
                    folder / (m.slug(endpoint) + "_records.json"),
                    [
                        {
                            k: x[k]
                            for k in (
                                "key",
                                "text",
                                "source",
                                "raw",
                                "gate",
                                "hate_types",
                                "target_groups",
                                "severity",
                                "text_cluster",
                            )
                        }
                        for x in rr
                    ],
                )
                m.array(folder / (m.slug(endpoint) + "_flow.npz"), **z)
                flow = fine_flow(z["truth"], z["raw"], z["gate"])
                out = dict(
                    model_counts=flow,
                    budget={},
                    pool={},
                    source_population=dict(Counter(x["source"] for x in rr)),
                )
                for key, value in flow.items():
                    if isinstance(value, np.ndarray):
                        pp, bp = c.aggregate(meta["models"], value)
                        out["pool"][key] = pp
                        out["budget"][key] = bp
                save(folder / (m.slug(endpoint) + "_flow.json"), out)
                analyse_cross(folder, endpoint, fine_meta, z, meta["models"])
            save(folder / "source_bindings.json", bindings)
            r.require(
                "torch" not in sys.modules, "Unexpected model-weight library loaded."
            )
            save(
                folder / "complete.json",
                dict(
                    status="complete",
                    utc=datetime.now(timezone.utc).isoformat(),
                    seconds=time.monotonic() - started,
                    outputs={
                        str(p.relative_to(folder)): r.sha(p)
                        for p in folder.rglob("*")
                        if p.is_file()
                    },
                    models=306,
                    fine_cross_views=1188,
                    training_calls=0,
                    new_inference=0,
                    Qwen_calls=0,
                    server_operations=0,
                    weights_loaded=0,
                ),
            )
            print("COMPLETE " + str(folder), flush=True)
    except BaseException as exc:
        save(
            folder / "stopped.json",
            dict(
                type=type(exc).__name__,
                message=str(exc),
                utc=datetime.now(timezone.utc).isoformat(),
            ),
        )
        raise


if __name__ == "__main__":
    if sys.argv[1:] == ["freeze"]:
        freeze()
    elif sys.argv[1:] == ["run"]:
        run()
    else:
        raise SystemExit("Use freeze or run.")

import argparse
from collections import Counter
import hashlib
from pathlib import Path
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.classification.broad import metrics as main

reader, c, r = main.reader, main.c, main.r
ROOT = main.ROOT
BASE = ROOT / "experiments/astra/sq2_broad_errors_review_20261104_v1"
PHASES = ("en_test", "de_test", "en_hatecheck_eval")
CONDITIONS = main.CONDITIONS
METRICS = ("fpr", "fnr", "error_rate")
EVENTS = (
    "fp_fixed",
    "fp_introduced",
    "fn_fixed",
    "fn_introduced",
    "fp_persistent",
    "fn_persistent",
    "both_correct_negative",
    "both_correct_positive",
)
STABLE = ("correct", "fixed", "introduced", "persistent_error", "variable")
DRAWS = 10000
SEED = 20261008
require = reader.require
bindings = reader.bindings


def save(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE.resolve()), "Part 3 output outside directory")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(c.encoded(value))


def array(path, **values):
    require(path.resolve().is_relative_to(BASE.resolve()), "Array outside Part 3")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        np.savez_compressed(stream, **values)


def safe(value):
    if isinstance(value, dict):
        return {key: safe(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def divide(a, b):
    a, b = np.broadcast_arrays(a, b)
    return np.divide(a, b, out=np.full(a.shape, np.nan, dtype=float), where=b != 0)


def metrics(counts):
    z = np.asarray(counts)
    require(z.shape[-1] == 4 and (z >= 0).all(), "Invalid confusion matrix")
    tp, fp, fn, tn = np.moveaxis(z, -1, 0)
    n = z.sum(axis=-1)
    positive = divide(2 * tp, 2 * tp + fp + fn)
    negative = divide(2 * tn, 2 * tn + fp + fn)
    return dict(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        n=n,
        positive_gold=tp + fn,
        negative_gold=tn + fp,
        fpr=divide(fp, fp + tn),
        fnr=divide(fn, fn + tp),
        precision=divide(tp, tp + fp),
        recall=divide(tp, tp + fn),
        error_rate=divide(fp + fn, n),
        accuracy=divide(tp + tn, n),
        predicted_hate=divide(tp + fp, n),
        balanced_accuracy=(divide(tp, tp + fn) + divide(tn, tn + fp)) / 2,
        macro_f1=np.where(
            n > 0,
            (np.nan_to_num(positive, nan=0) + np.nan_to_num(negative, nan=0)) / 2,
            np.nan,
        ),
    )


def contributions(y, p):
    y = np.asarray(y)
    p = np.asarray(p)
    require(y.ndim == 1 and p.ndim == 2 and p.shape[1] == len(y), "Wrong binary axes")
    require(np.isin(y, [0, 1]).all() and np.isin(p, [0, 1]).all(), "Nonbinary decision")
    y = y.astype(bool)
    p = p.astype(bool)
    return np.stack((p & y, p & ~y, ~p & y, ~p & ~y), axis=-1).astype(np.int64)


def summarize(y, p):
    values = metrics(contributions(y, p).sum(axis=1))
    return dict(
        repeats=[
            safe({key: value[i] for key, value in values.items()})
            for i in range(len(p))
        ],
        mean=safe({key: np.mean(value) for key, value in values.items()}),
    )


def transitions(y, full, reference):
    require(
        full.shape == reference.shape and full.shape[0] == 3,
        "Three fixed repeats per side required",
    )
    y = np.asarray(y, dtype=bool)
    a = full.astype(bool)[None, :, :]
    b = reference.astype(bool)[:, None, :]
    event = np.stack(
        (
            ~a & b & ~y,
            a & ~b & ~y,
            a & ~b & y,
            ~a & b & y,
            a & b & ~y,
            ~a & ~b & y,
            ~a & ~b & ~y,
            a & b & y,
        ),
        axis=-1,
    )
    require(
        event.shape == (3, 3, len(y), 8) and np.all(event.sum(axis=-1) == 1),
        "Incomplete transition partition",
    )
    fc = (full == y).sum(axis=0)
    rc = (reference == y).sum(axis=0)
    stable = np.full(len(y), 4, dtype=np.int8)
    for index, on in enumerate(
        (
            (fc == 3) & (rc == 3),
            (fc == 3) & (rc == 0),
            (fc == 0) & (rc == 3),
            (fc == 0) & (rc == 0),
        )
    ):
        stable[on] = index
    pair_counts = event.sum(axis=2)
    numerators = event.sum(axis=(0, 1))
    require(
        np.all(numerators.sum(axis=1) == 9), "Transition numerator denominator mismatch"
    )
    return dict(
        event_numerators=numerators,
        stable=stable,
        pair_counts=pair_counts,
        summary=dict(
            mean_events={
                name: float(pair_counts[:, :, i].mean())
                for i, name in enumerate(EVENTS)
            },
            stable_by_gold={
                str(gold): {
                    name: int(((stable == i) & (y == gold)).sum())
                    for i, name in enumerate(STABLE)
                }
                for gold in (0, 1)
            },
            pairing="All nine cross-repeat pairs, equally weighted; not nine independent texts or a temporal intervention.",
        ),
    )


def bootstrap(y, p, clusters):
    item = contributions(y, p)
    unique = sorted(set(clusters))
    lookup = {key: i for i, key in enumerate(unique)}
    grouped = np.zeros((len(unique), len(p), 4), dtype=np.int64)
    np.add.at(grouped, [lookup[key] for key in clusters], item.transpose(1, 0, 2))
    flat = grouped.reshape(len(unique), -1).astype(float)
    rng = np.random.default_rng(SEED)
    draws = []
    f1 = []
    for start in range(0, DRAWS, 64):
        size = min(64, DRAWS - start)
        picks = rng.choice(list(range(len(unique))), (size, len(unique)), replace=True)
        weights = np.stack(
            [np.bincount(row, minlength=len(unique)) for row in picks]
        ).astype(float)
        totals = (weights @ flat).reshape(size, len(p), 4)
        require(np.array_equal(totals, np.rint(totals)), "Bootstrap count not integer")
        require(
            np.array_equal(totals[0], grouped[picks[0]].sum(axis=0)),
            "Direct cluster check failed",
        )
        values = metrics(totals)
        delta = lambda a: (a[:, :3, None] - a[:, None, 3:]).mean(axis=(1, 2))
        draws.extend(
            np.stack([delta(values[name]) for name in METRICS], axis=1).tolist()
        )
        f1.extend(delta(values["macro_f1"]).tolist())
    draws = np.asarray(draws)
    point = metrics(item.sum(axis=1))
    out = {}
    for j, name in enumerate(METRICS):
        invalid = int((~np.isfinite(draws[:, j])).sum())
        out[name] = dict(
            difference=safe(float(point[name][:3].mean() - point[name][3:].mean())),
            ci95=(
                np.quantile(draws[:, j], [0.025, 0.975]).tolist()
                if not invalid
                else None
            ),
            undefined_draws=invalid,
            direction="Full minus reference; positive means more errors.",
            role="supporting diagnostic, no change to primary macro-F1 family",
        )
    return (
        dict(metrics=out, n_clusters=len(unique), draws=DRAWS, seed=SEED),
        draws,
        np.asarray(f1),
    )


def choose(ids, y, stable, phase, right):
    cells = []
    for gold in (0, 1):
        for label, index in (("fixed", 1), ("introduced", 2), ("persistent_error", 3)):
            category = ("fp_" if gold == 0 else "fn_") + label
            candidates = [
                key for key, g, s in zip(ids, y, stable) if g == gold and s == index
            ]
            digest = lambda key: hashlib.sha256(
                f"sq2-broad-part3-v1|{phase}|{right}|{category}|{key}".encode("utf8")
            ).hexdigest()
            chosen = (
                min(candidates, key=lambda key: (digest(key), key))
                if candidates
                else None
            )
            cells.append(
                dict(
                    population=phase,
                    reference=right,
                    category=category,
                    candidates=len(candidates),
                    key=chosen,
                    selection_sha256=digest(chosen) if chosen else None,
                )
            )
    return cells


def propagation(rr, outputs, dim=None):
    valid = [row for row in rr if outputs[row["key"]] is not None]
    if dim:
        field = {
            "target_group": "target_groups",
            "hate_type": "hate_types",
            "severity": "severity",
        }[dim]
        has = lambda row: (
            row[field] is not None if dim == "severity" else bool(row[field])
        )
        blocked = [row for row in valid if not outputs[row["key"]]["hate"]]
        return dict(
            eligible=len(rr),
            valid=len(valid),
            invalid=len(rr) - len(valid),
            gold_positive_information=sum(has(row) for row in valid),
            gate_false=len(blocked),
            positive_gold_gated_off=sum(has(row) for row in blocked),
            positive_gold_bits_gated_off=sum(
                1 if dim == "severity" else len(set(row[field]))
                for row in blocked
                if has(row)
            ),
            empty_gold_gate_false=sum(not has(row) for row in blocked),
        )
    benign = [row for row in valid if row["gate"] is False]
    fp = [row for row in benign if outputs[row["key"]]["hate"]]

    def present(row, field):
        value = outputs[row["key"]][field]
        return value is not None if field == "severity" else bool(value)

    return dict(
        eligible=len(rr),
        valid=len(valid),
        invalid=len(rr) - len(valid),
        valid_negative=len(benign),
        false_positive=len(fp),
        fp_type=sum(present(row, "hate_type") for row in fp),
        fp_target=sum(present(row, "target_group") for row in fp),
        fp_severity=sum(present(row, "severity") for row in fp),
        fp_any_fine=sum(
            any(
                present(row, field)
                for field in ("hate_type", "target_group", "severity")
            )
            for row in fp
        ),
        fp_all_fine=sum(
            all(
                present(row, field)
                for field in ("hate_type", "target_group", "severity")
            )
            for row in fp
        ),
        fp_legal_present=sum(present(row, "legal") for row in fp),
        note="Fine assertions attached to gold non-hate; legal presence is not a legal error without matching legal gold.",
    )


def prepare():
    save(
        BASE / "analysis_contract.json",
        dict(
            status="part3_scope_before_error_calculation_and_case_selection",
            created_utc=c.utc(),
            sources=bindings(
                [
                    main.BASE / "part2_complete.json",
                    main.BASE / "final_review.json",
                    reader.PLAN,
                    reader.BASE / "part1_complete.json",
                ]
            ),
            binary_populations=list(PHASES),
            comparators="All original eleven binary full-versus-small/zero/historical-encoder pairs; Qwen-focused six pairs for illustrative cases.",
            metrics="TP/FP/FN/TN, FPR/FNR, positive precision/recall, error rate/accuracy, balanced accuracy, predicted-hate share. Mean individual-repeat metrics and own-valid/shared masks; no pooled predictions.",
            diagnostic_intervals=dict(
                metrics=list(METRICS),
                draws=DRAWS,
                seed=SEED,
                confidence=0.95,
                clusters="Original natural text groups / HateCheck components, paired common-valid six-setting mask.",
                role="Analysis-time supporting diagnostics; original four-primary macro-F1 family unchanged.",
                undefined="Zero denominator is undefined; retain undefined bootstrap count and no interval if any draw undefined, never silently drop draws.",
            ),
            transitions="All nine cross-repeat pairings equally weighted. Fixed/new/persistent errors and both-correct partitions. Stable classifications require all three runs agree within each system. Counts do not imply a temporal intervention or independent repeats.",
            gate_delivery="Gold-negative binary populations: false-positive fine-field attachments. Seven fine-gold endpoints: gate blocks, positive information lost and known-empty sets separately. Qwen conditions, own-valid and common-nine views. German legal output independent of the hate gate; no unannotated legal-error claim.",
            case_rule="For each of three binary populations and full-versus-small/zero, choose one all-repeat-stable fixed/new/persistent error for each gold class. Select smallest SHA256 of sq2-broad-part3-v1|population|reference|category|key, key tie-break. Keep empty cells. Up to 36 slots, with duplicates retained if selected twice. No choosing by persuasive text.",
            manual_review="After the roster is frozen, read every chosen input and all saved final decisions. First-repeat rationale may illustrate the delivered explanation, not faithful reasoning. Preserve gold; mark possible annotation/boundary issues as interpretations. Save direct result hashes.",
            deferred=[
                "full fine-label and joint answer scoring Part 4",
                "dataset/function outcome breakdown Part 5",
                "retrieved contexts Part 6",
                "source/item/replay/whitespace/partial-vote sensitivity Part 7",
                "resources/synthesis Parts 8–9",
            ],
            prohibited=[
                "new Qwen calls",
                "training/inference/model loading",
                "retrieval rerun",
                "source/evidence modification",
                "new threshold or relabelling",
                "server actions",
                "writer edits",
            ],
        ),
    )


def run():
    contract = c.read(BASE / "analysis_contract.json")
    r.verify(contract["sources"])
    tests = c.read(BASE / "tests.json")
    r.verify(tests["sources"])
    require(tests["status"] == "passed", "Tests required")
    seal = c.read(main.BASE / "part2_complete.json")
    r.verify(seal["outputs"])
    r.verify(seal["original_sources"])
    groups, _, runs, endpoints, comparisons = reader.load()
    before = bindings(
        [
            Path(__file__),
            BASE / "analysis_contract.json",
            BASE / "tests.json",
            main.BASE / "part2_complete.json",
            reader.BASE / "part1_complete.json",
        ]
    )
    save(BASE / "preservation_before.json", before)
    results = {}
    cases = []
    draw_count = 0
    f1_count = 0
    prop = []
    for phase in PHASES:
        name = phase + "/hate"
        ep = endpoints[name]
        path = main.BASE / "endpoints" / main.slug(name) / "results.json"
        old = c.read(path)
        settings = [tuple(x) for x in old["settings"]]
        with np.load(ROOT / old["encoded_path"], allow_pickle=False) as enc:
            y = enc["gold"][:, ep["all_labels"].index("hate")].astype(bool)
            p = enc["predicted"][:, :, ep["all_labels"].index("hate")].astype(bool)
            v = enc["valid"].copy()
        ids = ep["scorable_ids"]
        lookup = {row["key"]: row for row in groups[phase]}
        own = {}
        shared = {}
        paired = []
        for cond in ep["available_conditions"]:
            own[cond] = []
            for rep in range(3):
                s = settings.index((cond, rep))
                mask = v[s]
                value = summarize(y[mask], p[s : s + 1, mask])["repeats"][0]
                require(
                    abs(
                        value["macro_f1"]
                        - old["own_valid"][cond]["repeats"][rep]["macro_f1_valid_only"]
                    )
                    < 2e-15,
                    "Own macro F1 differs from Part 2",
                )
                own[cond].append(
                    value
                    | dict(
                        n_invalid=int((~mask).sum()),
                        invalid_negative=int((~mask & ~y).sum()),
                        invalid_positive=int((~mask & y).sum()),
                    )
                )
        nine = [settings.index((cond, rep)) for cond in CONDITIONS for rep in range(3)]
        common = v[nine].all(axis=0)
        for cond in CONDITIONS:
            ix = [settings.index((cond, rep)) for rep in range(3)]
            shared[cond] = summarize(y[common], p[ix][:, common])
            for rep in range(3):
                rr = [lookup[key] for key, on in zip(ids, common) if on]
                prop.append(
                    dict(
                        population=phase,
                        dimension="hate",
                        condition=cond,
                        repeat=rep,
                        mask="common_nine",
                        **propagation(rr, runs[cond, rep]),
                    )
                )
                prop.append(
                    dict(
                        population=phase,
                        dimension="hate",
                        condition=cond,
                        repeat=rep,
                        mask="own_valid",
                        **propagation(groups[phase], runs[cond, rep]),
                    )
                )
        for previous in old["contrasts"]:
            right = previous["right"]
            ix = [
                settings.index((cond, rep))
                for cond in ("rag_full", right)
                for rep in range(3)
            ]
            mask = v[ix].all(axis=0)
            selected = [key for key, on in zip(ids, mask) if on]
            require(
                selected == previous["common_valid_ids"],
                "Pair mask differs from Part 2",
            )
            yy = y[mask]
            pp = p[ix][:, mask]
            cl = [cluster for cluster, on in zip(ep["row_clusters"], mask) if on]
            estimates, draws, f1 = bootstrap(yy, pp, cl)
            reference_f1 = np.load(ROOT / previous["draws_path"], allow_pickle=False)
            require(
                np.allclose(f1, reference_f1, rtol=0, atol=2e-15),
                "Binary bootstrap F1 differs from Part 2",
            )
            f1_count += len(f1)
            draw_count += draws.size
            delta = transitions(yy, pp[:3], pp[3:])
            left_summary = summarize(yy, pp[:3])
            right_summary = summarize(yy, pp[3:])
            events = delta["summary"]["mean_events"]
            require(
                abs(
                    (events["fp_introduced"] - events["fp_fixed"])
                    - (left_summary["mean"]["fp"] - right_summary["mean"]["fp"])
                )
                < 1e-10,
                "FP flow mismatch",
            )
            require(
                abs(
                    (events["fn_introduced"] - events["fn_fixed"])
                    - (left_summary["mean"]["fn"] - right_summary["mean"]["fn"])
                )
                < 1e-10,
                "FN flow mismatch",
            )
            folder = BASE / phase / ("full_minus_" + right)
            array(
                folder / "diagnostics.npz",
                metric_draws=draws,
                common_indices=np.flatnonzero(mask),
                event_numerators=delta["event_numerators"],
                stable=delta["stable"],
                pair_counts=delta["pair_counts"],
            )
            paired.append(
                dict(
                    reference=right,
                    n_common=len(yy),
                    n_positive=int(yy.sum()),
                    n_negative=int((~yy).sum()),
                    full=left_summary,
                    reference_summary=right_summary,
                    diagnostic_intervals=estimates,
                    transitions=delta["summary"],
                    arrays=c.rel(folder / "diagnostics.npz"),
                    arrays_sha256=c.sha(folder / "diagnostics.npz"),
                    original_primary_F1_role=previous["role"],
                )
            )
            if right in ("rag_small", "zero_shot"):
                cases.extend(choose(selected, yy, delta["stable"], phase, right))
            print("Error comparison:", phase, right, flush=True)
        results[phase] = dict(
            endpoint=name,
            settings=[list(x) for x in settings],
            own_valid=own,
            common_nine=shared,
            common_nine_texts=int(common.sum()),
            eligible=len(ids),
            paired=paired,
        )
    for name, ep in endpoints.items():
        phase, dim = name.split("/")
        if dim in ("hate", "legal"):
            continue
        lookup = {row["key"]: row for row in groups[phase]}
        rr = [lookup[key] for key in ep["scorable_ids"]]
        common = [
            row
            for row in rr
            if all(
                runs[cond, rep][row["key"]] is not None
                for cond in CONDITIONS
                for rep in range(3)
            )
        ]
        for cond in CONDITIONS:
            for rep in range(3):
                for mask, roster in (("own_valid", rr), ("common_nine", common)):
                    prop.append(
                        dict(
                            population=phase,
                            dimension=dim,
                            condition=cond,
                            repeat=rep,
                            mask=mask,
                            **propagation(roster, runs[cond, rep], dim),
                        )
                    )
    legal = []
    for cond in CONDITIONS:
        for rep in range(3):
            values = [runs[cond, rep][row["key"]] for row in groups["de_legal_test"]]
            valid = [v for v in values if v is not None]
            legal.append(
                dict(
                    condition=cond,
                    repeat=rep,
                    eligible=len(values),
                    valid=len(valid),
                    predicted_nonhate=sum(not v["hate"] for v in valid),
                    nonhate_with_legal=sum(
                        not v["hate"] and bool(v["legal"]) for v in valid
                    ),
                    note="Independent legal branch presence, not legal correctness; legal quality is Part 4.",
                )
            )
    for index, case in enumerate(cases, 1):
        case["case_id"] = f"C{index:02d}"
    save(BASE / "case_selection.json", dict(rule=contract["case_rule"], slots=cases))
    save(BASE / "binary_results.json", results)
    save(BASE / "gate_delivery.json", dict(rows=prop, legal_independence=legal))
    require(
        len(cases) == 36
        and draw_count == 330000
        and f1_count == 110000
        and len(prop) == 180,
        "Incomplete diagnostic inventory",
    )
    selected = {case["key"] for case in cases if case["key"] is not None}
    cache = {key: [] for key in selected}
    source_bindings = c.read(reader.BASE / "item_bindings.json")
    focused = {}
    for row in reader.rows(reader.BASE / "predictions.jsonl"):
        if row["key"] not in selected:
            continue
        path = Path(row["result_path"])
        digest = c.sha(path)
        require(
            digest == source_bindings[row["result_path"]],
            "Selected original result changed",
        )
        focused[row["result_path"]] = digest
        cache[row["key"]].append(row)
    gold = {row["key"]: row for rr in groups.values() for row in rr}
    records = []
    for case in cases:
        if case["key"] is None:
            records.append(case | dict(status="empty stable-error category"))
            continue
        require(len(cache[case["key"]]) == 9, "Selected case missing Qwen settings")
        records.append(
            case
            | dict(
                status="selected",
                gold=gold[case["key"]],
                predictions=cache[case["key"]],
            )
        )
    save(BASE / "cases.json", records)
    save(BASE / "focused_source_bindings.json", focused)
    r.verify(before)
    outputs = bindings([path for path in BASE.rglob("*") if path.is_file()])
    save(
        BASE / "analysis_complete.json",
        dict(
            status="part3_computation_complete_independent_and_manual_reviews_pending",
            sources=before,
            outputs=outputs,
            binary_pair_comparisons=11,
            diagnostic_bootstrap_values=draw_count,
            original_F1_draws_reconciled=f1_count,
            case_slots=36,
            selected_slots=sum(case["key"] is not None for case in cases),
            unique_case_texts=len(selected),
            direct_source_results_verified=len(focused),
            gate_delivery_setting_views=len(prop),
            new_Qwen_calls=0,
            new_training=0,
            new_inference=0,
        ),
    )
    print(
        "Part 3 calculation complete. Independent and selected-case reviews are next.",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", required=True, choices=["prepare", "run"])
    args = p.parse_args()
    with r.original.offline(), threadpool_limits(limits=2):
        {"prepare": prepare, "run": run}[args.mode]()

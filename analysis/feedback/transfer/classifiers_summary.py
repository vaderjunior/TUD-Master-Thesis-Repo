from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import numpy as np
from analysis.feedback.transfer import classifiers as p

a = p.a
k = p.k
OUT = p.OUT
masks = p.masks


def cohort(meta, q, v, endpoint, view, excluded, subset=None, arms=(1, 2, 3)):
    pop, kind = endpoint.rsplit("_", 1)
    strict, rep, ex = masks.VIEWS[view]
    return masks.cohort(
        meta,
        q,
        v,
        pop,
        kind == "FNR",
        strict=strict,
        representatives=rep,
        excluded=excluded if ex else (),
        eligible_override=subset,
        arms=arms,
    )


def omitted(raw, ids, arms=(1, 2, 3)):
    if not len(ids):
        return None
    totals = raw[ids].sum(axis=0, dtype=np.int64)
    n = len(ids)
    q = totals[16:]
    out = []
    for s in range(4):
        l = totals[s * 4 : s * 4 + 4]
        before = l[0] / n
        after = l[list(arms)].sum() / (len(arms) * n)
        ref = q[0] / (3 * n)
        updated = q[list(arms)].sum() / (3 * len(arms) * n)
        ld = (l[list(arms)].sum() - len(arms) * l[0]) / (len(arms) * n)
        qd = (q[list(arms)].sum() - len(arms) * q[0]) / (3 * len(arms) * n)
        out.append(
            dict(
                linear_baseline=before,
                linear_updated=after,
                qwen_reference=ref,
                qwen_updated=updated,
                linear_change=ld,
                qwen_change=qd,
                difference_in_change=ld - qd,
                updated_absolute_difference=after - updated,
            )
        )
    return out


def transitions(linear, qwen, ids):
    l = linear[ids]
    q = qwen[ids]
    out = []
    for s in range(4):
        le = l.reshape(len(ids), 4, 4)[:, s]
        qcounts = q.sum(axis=2)
        lp = le[:, 1:] - le[:, :1]
        qp = qcounts[:, 1:] - qcounts[:, :1]
        joint = np.zeros((3, 3), dtype=np.int64)
        for x, y in zip(
            np.sign(lp.sum(axis=1)).astype(int),
            np.sign(qcounts[:, 1:].sum(axis=1) - 3 * qcounts[:, 0]).astype(int),
        ):
            joint[x + 1, y + 1] += 1
        pools = []
        for j in range(3):
            mat = np.zeros((3, 3), dtype=np.int64)
            for x, y in zip(
                np.sign(lp[:, j]).astype(int), np.sign(qp[:, j]).astype(int)
            ):
                mat[x + 1, y + 1] += 1
            disagree = (
                np.asarray(
                    [
                        ((le[:, j + 1] == x) & (q[:, j + 1, t] == y)).sum()
                        for x, y in ((0, 0), (0, 1), (1, 0), (1, 1))
                        for t in range(3)
                    ]
                )
                .reshape(4, 3)
                .sum(axis=1)
            )
            pools.append(
                dict(
                    condition=p.CONDITIONS[j + 1],
                    linear_fixes=int((lp[:, j] < 0).sum()),
                    linear_regressions=int((lp[:, j] > 0).sum()),
                    linear_unchanged_correct=int(
                        ((le[:, 0] == 0) & (le[:, j + 1] == 0)).sum()
                    ),
                    linear_unchanged_error=int(
                        ((le[:, 0] == 1) & (le[:, j + 1] == 1)).sum()
                    ),
                    directional_matrix=mat,
                    updated_error_pairs=disagree,
                    pair_count=3 * len(ids),
                )
            )
        out.append(
            dict(
                inputs=len(ids),
                directional_order=["improved", "unchanged_average", "worsened"],
                mean_direction_matrix=joint,
                pools=pools,
                linear_consistent_fixes=int(
                    ((le[:, 0] == 1) & np.all(le[:, 1:] == 0, axis=1)).sum()
                ),
                linear_consistent_regressions=int(
                    ((le[:, 0] == 0) & np.all(le[:, 1:] == 1, axis=1)).sum()
                ),
                linear_any_fix=int((lp < 0).any(axis=1).sum()),
                linear_any_regression=int((lp > 0).any(axis=1).sum()),
                linear_changed_any_pool=int((lp != 0).any(axis=1).sum()),
                linear_opposing_pool_directions=int(
                    ((lp < 0).any(axis=1) & (lp > 0).any(axis=1)).sum()
                ),
                error_pair_order=[
                    "both_correct",
                    "linear_correct_qwen_wrong",
                    "linear_wrong_qwen_correct",
                    "both_wrong",
                ],
                dependent_reference_and_pool_comparisons=True,
            )
        )
    return out


def main():
    a.need(
        not (OUT / "calculated.json").exists() and not (OUT / "families.json").exists(),
        "Preserve earlier Part 8 calculations",
    )
    assembled = a.read(OUT / "assembled.json")
    contract = a.read(OUT / "analysis_contract.json")
    for name, digest in assembled["artifacts"].items():
        a.need(a.sha(a.ROOT / name) == digest, "Changed assembled input")
    meta = list(a.rows(p.P3 / "metadata.jsonl"))
    roster = list(a.rows(a.OUT / "roster.jsonl"))
    N = len(meta)
    excluded = contract["replay_exclusions"]
    eras = [
        r["actual_execution_era"] for r in p.gzrows(p.P7 / "input_execution.jsonl.gz")
    ]
    with np.load(OUT / "arrays.npz", allow_pickle=False) as f:
        arrays = {name: f[name] for name in f.files}
    l = arrays["linear_errors"]
    q = arrays["qwen_errors"]
    v = arrays["valid_votes"]
    raw = arrays["raw_error_counts"]
    gold = np.asarray([m["gold_hate"] for m in meta], dtype=np.int8)
    with np.load(p.P7 / "vote_completion_arrays.npz", allow_pickle=False) as f:
        logical_lo = f["lower_error"]
        logical_hi = f["upper_error"]
    families = {}
    rates = {}
    draws_count = 0

    def evaluate(
        name, endpoint, view, subset=None, draw=False, mode="cluster", primary=False
    ):
        nonlocal draws_count
        c = cohort(meta, q, v, endpoint, view, excluded, subset)
        eligible, common = c["eligible"], c["common"]
        n = len(common)
        record = dict(
            endpoint=endpoint,
            view=view,
            resampling_mode=mode,
            eligible_indices=eligible,
            common_indices=common,
            eligible_n=len(eligible),
            common_n=n,
            clusters=len({meta[i]["inference_cluster"] for i in common}),
            source_counts=Counter(meta[i]["source"] for i in common),
            strict=c["strict"],
            primary_family=primary,
        )
        values = k.expand(raw[common].sum(axis=0, dtype=np.int64), n) if n else None
        draws = None
        if draw and n:
            rng_seed = (
                contract["primary"]["seeds"][p.ENDPOINTS.index(endpoint)]
                if primary
                else k.seed(name)
            )
            draws, bootstrap = k.bootstrap(meta, raw, common, eligible, rng_seed, mode)
            path = OUT / "draws" / (name.replace("/", "__") + ".npy")
            path.parent.mkdir(exist_ok=True)
            with path.open("xb") as stream:
                np.save(stream, draws, allow_pickle=False)
            record.update(bootstrap=bootstrap, draw_file=a.rel(path))
            draws_count += 1
        bounds = k.bounds(l, q, v, eligible, c["strict"])
        logical = k.bounds(l, q, v, eligible, lower=logical_lo, upper=logical_hi)
        record["raw_totals"] = raw[common].sum(axis=0, dtype=np.int64)
        record["qwen_actual_errors"] = q[common].sum(axis=0, dtype=np.int64)
        families[name] = record
        for s in range(4):
            name_s = k.setting_name(s) + "/" + name
            point = (
                dict(zip(k.COLUMNS, values[s]))
                if values is not None
                else {metric: None for metric in k.COLUMNS}
            )
            row = dict(
                family=name,
                setting_index=s,
                variant=k.SETTINGS[s][0],
                cutoff=k.SETTINGS[s][1],
                endpoint=endpoint,
                view=view,
                common_n=n,
                eligible_n=len(eligible),
                clusters=record["clusters"],
                estimates=point,
                intervals95={
                    metric: (
                        np.quantile(draws[:, s, j], [0.025, 0.975]).tolist()
                        if draws is not None
                        else None
                    )
                    for j, metric in enumerate(k.COLUMNS)
                },
                primary_interval98_75=(
                    np.quantile(draws[:, s, 6], [0.00625, 0.99375]).tolist()
                    if primary and s == 0
                    else None
                ),
                inference_role=(
                    "prespecified four-contrast primary"
                    if primary and s == 0
                    else (
                        "supporting 95% paired comparison"
                        if draws is not None
                        else "descriptive point only"
                    )
                ),
                full_eligible_missing_bounds=bounds[s] if bounds else None,
                hypothetical_invalid_vote_completion_bounds=(
                    logical[s] if logical else None
                ),
            )
            rates[name_s] = row
        if draw:
            print(
                f"Checked cohort and calculated {name}: {n} common inputs.", flush=True
            )

    for view in masks.VIEWS:
        for endpoint in p.ENDPOINTS:
            evaluate(
                "cohort/" + view + "/" + endpoint,
                endpoint,
                view,
                draw=True,
                primary=view == "main",
            )
    for view in ("main", "strict", "replay_excluded"):
        for endpoint in p.ENDPOINTS:
            mode = "source_stratified" if endpoint.startswith("en_test_") else "item"
            evaluate(
                "resampling/" + mode + "/" + view + "/" + endpoint,
                endpoint,
                view,
                draw=True,
                mode=mode,
            )
    for source in ("hatexplain", "implicit_hate", "mhs"):
        subset = [i for i, m in enumerate(meta) if m["source"] == source]
        for view in ("main", "strict"):
            for kind in ("FPR", "FNR"):
                evaluate(
                    "source/" + source + "/" + view + "/en_test_" + kind,
                    "en_test_" + kind,
                    view,
                    subset,
                    draw=True,
                )
    for label in ("all_8", "all_128"):
        subset = [i for i, e in enumerate(eras) if e == label]
        for view in ("main", "strict"):
            for endpoint in p.ENDPOINTS:
                evaluate(
                    "era/" + label + "/" + view + "/" + endpoint, endpoint, view, subset
                )
    for fn in sorted({m["functionality"] for m in meta if m["source"] == "hatecheck"}):
        subset = [i for i, m in enumerate(meta) if m.get("functionality") == fn]
        endpoint = "en_hatecheck_eval_" + ("FNR" if fn.endswith("_h") else "FPR")
        for view in ("main", "strict"):
            evaluate("function/" + fn + "/" + view, endpoint, view, subset)
    templates = defaultdict(list)
    for i, m in enumerate(meta):
        if m["source"] == "hatecheck":
            templates[str(m["template_id"])].append(i)
    for label, subset in sorted(templates.items()):
        for truth in sorted({meta[i]["gold_hate"] for i in subset}):
            endpoint = "en_hatecheck_eval_" + ("FNR" if truth else "FPR")
            evaluate("template/" + label + "/" + endpoint, endpoint, "main", subset)
    p.save("families.json", families)
    p.save("rates.json", rates)

    omissions = {}
    for view in masks.VIEWS:
        for endpoint in p.ENDPOINTS:
            c = cohort(meta, q, v, endpoint, view, excluded)
            common = c["common"]
            rows = []
            for j in (1, 2, 3):
                arms = tuple(x for x in (1, 2, 3) if x != j)
                for kind in ("fixed_common", "recomputed_common"):
                    selected = (
                        common
                        if kind == "fixed_common"
                        else cohort(meta, q, v, endpoint, view, excluded, arms=arms)[
                            "common"
                        ]
                    )
                    rows.append(
                        dict(
                            kind="pool_" + kind,
                            omitted=p.CONDITIONS[j],
                            common_n=len(selected),
                            indices=selected,
                            estimates=omitted(raw, selected, arms),
                        )
                    )
            for batch in sorted({m["stage"] for m in meta}):
                selected = np.asarray(
                    [i for i in common if meta[i]["stage"] != batch], dtype=np.int64
                )
                rows.append(
                    dict(
                        kind="execution_batch",
                        omitted=batch,
                        common_n=len(selected),
                        indices=selected,
                        estimates=omitted(raw, selected),
                    )
                )
            omissions[view + "/" + endpoint] = rows
    p.save("omissions.json", omissions)
    influence = {}
    influence_rows = 0
    with p.gzopen("group_omissions.jsonl.gz") as stream:
        for endpoint in p.ENDPOINTS:
            c = cohort(meta, q, v, endpoint, "main", excluded)
            ids = c["common"]
            base = omitted(raw, ids)[0]["difference_in_change"]
            for field in (
                ("inference_cluster", "functionality")
                if endpoint.startswith("en_hatecheck_eval_")
                else ("inference_cluster",)
            ):
                groups = defaultdict(list)
                for i in ids:
                    groups[meta[i][field]].append(int(i))
                total = raw[ids].sum(axis=0, dtype=np.int64)
                items = []
                for label, members in sorted(groups.items()):
                    remaining = len(ids) - len(members)
                    delta = (
                        float(
                            k.expand(
                                total - raw[members].sum(axis=0, dtype=np.int64),
                                remaining,
                            )[0, 6]
                        )
                        if remaining
                        else None
                    )
                    row = dict(
                        endpoint=endpoint,
                        unit=field,
                        omitted=label,
                        indices=members,
                        remaining_inputs=remaining,
                        difference_in_change=delta,
                    )
                    p.emit(stream, row)
                    items.append(row)
                    influence_rows += 1
                finite = [
                    row for row in items if row["difference_in_change"] is not None
                ]
                influence[endpoint + "/" + field] = dict(
                    inputs=len(ids),
                    groups=len(items),
                    base_difference_in_change=base,
                    minimum=min(r["difference_in_change"] for r in finite),
                    maximum=max(r["difference_in_change"] for r in finite),
                    opposite_sign_omissions=sum(
                        r["difference_in_change"] * base < 0 for r in finite
                    ),
                    exact_zero_omissions=sum(
                        r["difference_in_change"] == 0 for r in finite
                    ),
                    largest_change=sorted(
                        finite,
                        key=lambda r: (
                            -abs(r["difference_in_change"] - base),
                            str(r["omitted"]),
                        ),
                    )[:10],
                )
    p.save("group_influence.json", influence)

    quality = {}
    trans = {}
    for pop in p.POPS:
        for view in masks.VIEWS:
            selected = np.asarray(
                sorted(
                    np.concatenate(
                        [
                            cohort(meta, q, v, pop + "_" + kind, view, excluded)[
                                "common"
                            ]
                            for kind in ("FPR", "FNR")
                        ]
                    )
                ),
                dtype=np.int64,
            )
            selected_gold = gold[selected]
            n = len(selected)
            qp = arrays["qwen_decisions"][selected].reshape(n, 12)
            for s in range(4):
                vi = s // 2
                ti = s % 2
                pred = np.column_stack(
                    (arrays["linear_decisions"][selected, vi, ti], qp)
                )
                conf, metrics = k.binary_scores(selected_gold, pred)
                results = {}
                for metric, values in metrics.items():
                    arms = [
                        values[0],
                        values[1:4].mean(),
                        values[4:7].mean(),
                        values[7:].mean(),
                    ]
                    results[metric] = dict(
                        zip(
                            k.COLUMNS[:8],
                            [
                                *arms,
                                arms[1] - arms[0],
                                arms[3] - arms[2],
                                (arms[1] - arms[0]) - (arms[3] - arms[2]),
                                arms[1] - arms[3],
                            ],
                        )
                    )
                quality[k.setting_name(s) + "/" + pop + "/" + view] = dict(
                    common_n=n,
                    indices=selected,
                    gold_counts=Counter(str(int(x)) for x in selected_gold),
                    confusion_order=["TN", "FP", "FN", "TP"],
                    actual_model_counts=conf,
                    model_order=[
                        "linear_reference",
                        *["linear_" + c for c in p.CONDITIONS[1:]],
                        *[f"qwen_{c}_r{t+1}" for c in p.CONDITIONS for t in range(3)],
                    ],
                    metrics=results,
                    role="Supporting descriptive points, not new quality confidence intervals; binary only.",
                )
            summary = transitions(l, q, selected)
            for s in range(4):
                trans[k.setting_name(s) + "/" + pop + "/" + view] = summary[s]
    p.save("binary_quality.json", quality)
    p.save("transitions.json", trans)

    prob = arrays["probabilities"]
    descriptive = {}
    groups = {"all": list(range(N))}
    for pop in p.POPS:
        groups["population/" + pop] = [
            i for i, m in enumerate(meta) if m["population"] == pop
        ]
    for source in sorted({m["source"] for m in meta}):
        groups["source/" + source] = [
            i for i, m in enumerate(meta) if m["source"] == source
        ]
    for fn in sorted({m["functionality"] for m in meta if m["source"] == "hatecheck"}):
        groups["function/" + fn] = [
            i for i, m in enumerate(meta) if m.get("functionality") == fn
        ]
    for label, ids in groups.items():
        for vi, variant in enumerate(k.VARIANTS):
            for j in (1, 2, 3):
                before = prob[ids, vi, 0]
                after = prob[ids, vi, j]
                delta = after - before
                absolute = np.abs(delta)
                cuts = {}
                for ti, cutoff in enumerate(k.CUTOFFS):
                    base = arrays["linear_decisions"][ids, vi, ti, 0]
                    upd = arrays["linear_decisions"][ids, vi, ti, j]
                    truth = gold[ids]
                    changed = base != upd
                    cuts[str(cutoff)] = dict(
                        changed=int(changed.sum()),
                        fixes=int((changed & (upd == truth)).sum()),
                        regressions=int((changed & (upd != truth)).sum()),
                        benign_to_hateful=int(((base == 0) & (upd == 1)).sum()),
                        hateful_to_benign=int(((base == 1) & (upd == 0)).sum()),
                        changed_input_indices=[
                            int(ids[z]) for z in np.flatnonzero(changed)
                        ],
                        mean_baseline_distance_to_cutoff_among_changed=(
                            float(np.abs(before[changed] - cutoff).mean())
                            if changed.any()
                            else None
                        ),
                    )
                descriptive[variant + "/" + p.CONDITIONS[j] + "/" + label] = dict(
                    inputs=len(ids),
                    mean_signed_probability_change=float(delta.mean()),
                    median_absolute_probability_change=float(np.median(absolute)),
                    p95_absolute_probability_change=float(np.quantile(absolute, 0.95)),
                    maximum_absolute_probability_change=float(absolute.max()),
                    thresholds=cuts,
                )
    p.save("probability_changes.json", descriptive)

    with p.gzopen("input_comparisons.jsonl.gz") as stream:
        for i, m in enumerate(meta):
            p.emit(
                stream,
                dict(
                    input_index=i,
                    key=m["key"],
                    population=m["population"],
                    source=m["source"],
                    gold_hate=m["gold_hate"],
                    inference_cluster=m["inference_cluster"],
                    linear_probabilities=prob[i],
                    linear_decisions=arrays["linear_decisions"][i],
                    qwen_decisions=arrays["qwen_decisions"][i],
                    valid_votes=v[i],
                    main_common=bool(np.all(q[i] >= 0)),
                    strict_common=bool(np.all(v[i] == 3)),
                ),
            )
    selected_cases = []
    for pop in p.POPS:
        common = np.asarray(
            [
                i
                for i, m in enumerate(meta)
                if m["population"] == pop and np.all(q[i] >= 0)
            ],
            dtype=np.int64,
        )
        le = l[common, 0, 0]
        qe = q[common]
        ld = le[:, 1:].sum(axis=1) - 3 * le[:, 0]
        qd = qe[:, 1:].sum(axis=(1, 2)) - 3 * qe[:, 0].sum(axis=1)
        categories = {
            "linear_consistent_fix": (le[:, 0] == 1) & np.all(le[:, 1:] == 0, axis=1),
            "linear_consistent_regression": (le[:, 0] == 0)
            & np.all(le[:, 1:] == 1, axis=1),
            "linear_improves_qwen_worsens": (ld < 0) & (qd > 0),
            "linear_worsens_qwen_improves": (ld > 0) & (qd < 0),
            "linear_unchanged_qwen_improves": (ld == 0) & (qd < 0),
            "linear_unchanged_qwen_worsens": (ld == 0) & (qd > 0),
            "updated_linear_correct_qwen_wrong": np.all(le[:, 1:] == 0, axis=1)
            & np.all(qe[:, 1:] == 1, axis=(1, 2)),
            "updated_linear_wrong_qwen_correct": np.all(le[:, 1:] == 1, axis=1)
            & np.all(qe[:, 1:] == 0, axis=(1, 2)),
        }
        for category, keep in categories.items():
            candidates = common[keep].tolist()
            chosen = (
                min(
                    candidates,
                    key=lambda i: (
                        hashlib.sha256(
                            (
                                "A18-part8-case-v1|"
                                + pop
                                + "|"
                                + category
                                + "|"
                                + meta[i]["key"]
                            ).encode()
                        ).hexdigest(),
                        meta[i]["key"],
                    ),
                )
                if candidates
                else None
            )
            selected_cases.append(
                dict(
                    population=pop,
                    category=category,
                    eligible_cases=len(candidates),
                    input_index=chosen,
                    key=meta[chosen]["key"] if chosen is not None else None,
                )
            )
    p.save("case_selection.json", selected_cases)
    files = [path for path in OUT.rglob("*") if path.is_file()]
    p.save(
        "calculated.json",
        dict(
            status="part8_calculated_independent_review_pending",
            created_utc=a.utc(),
            assembled_sha256=a.sha(OUT / "assembled.json"),
            script_sha256=a.sha(Path(__file__)),
            artifacts={a.rel(path): a.sha(path) for path in files},
            rate_families=len(families),
            rate_views=len(rates),
            bootstrap_sets=draws_count,
            bootstrap_scalars=draws_count * 10000 * 80,
            binary_quality_views=len(quality),
            transition_views=len(trans),
            probability_description_views=len(descriptive),
            omission_families=len(omissions),
            omission_rows=sum(len(items) for items in omissions.values()) * 4,
            influence_rows=influence_rows,
            case_slots=len(selected_cases),
            empty_case_slots=sum(r["input_index"] is None for r in selected_cases),
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
            new_primary_families=0,
        ),
    )
    print(
        "All Part 8 calculations saved; independent verification follows.", flush=True
    )


if __name__ == "__main__":
    with a.offline():
        main()

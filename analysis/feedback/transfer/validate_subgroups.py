import os

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "1"
from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from analysis.feedback.transfer import archive as a

OUT = a.ROOT / "experiments/astra/sq3_transfer_part3_20261023_v1"
P2 = a.ROOT / "experiments/astra/sq3_transfer_part2_20261023_v1"
VIEWS = dict(
    main=(False, False, False),
    strict=(True, False, False),
    replay_excluded=(False, False, True),
    strict_replay_excluded=(True, False, True),
    representative=(False, True, False),
    representative_replay_excluded=(False, True, True),
)
HEADS = ("hate_types", "target_groups", "severity", "legal")


def check(ok, message):
    if not ok:
        raise AssertionError(message)


def near(x, y, message):
    if isinstance(y, dict):
        check(set(x) == set(y), message + ": keys")
        for k, v in y.items():
            near(x[k], v, message + "/" + k)
    elif isinstance(y, (list, tuple)):
        check(len(x) == len(y), message + ": length")
        for i, v in enumerate(y):
            near(x[i], v, message + "/" + str(i))
    elif y is None or isinstance(y, (str, bool)):
        check(x == y, message)
    else:
        check(x is not None and abs(float(x) - float(y)) < 1e-13, message)


def write(name, value):
    with (OUT / name).open("xb") as f:
        f.write(a.encoded(value, pretty=True))


def rebuild():
    rows = []
    raw = []
    error = np.full((15758, 4, 3), -1, dtype=np.int8)
    votes = np.zeros_like(error)
    for i, (r, m) in enumerate(
        zip(a.rows(a.OUT / "roster.jsonl"), a.rows(a.OUT / "matrix.jsonl"), strict=True)
    ):
        check(r["key"] == m["key"] and m["input_index"] == i, "Original identity")
        rows.append(r)
        raw.append(json.loads(r["raw"]))
        for c in range(4):
            for rep in range(3):
                pred = m["predictions"][c][rep]
                votes[i, c, rep] = pred["n_valid"]
                if pred["result"] is not None:
                    error[i, c, rep] = int(pred["result"]["hate"] != r["gate"])
    return rows, raw, error, votes


def member(name, rows, raw):
    out = []
    for i, (r, d) in enumerate(zip(rows, raw)):
        source = r["source"]
        match = False
        if name.startswith("source/"):
            match = source == name.split("/")[1]
        elif name.startswith("function/"):
            match = source == "hatecheck" and d["functionality"] == name.split("/")[1]
        elif name.startswith("subtype/"):
            _, s, v = name.split("/", 2)
            if source == s:
                if s == "hatexplain":
                    majority, count = Counter(d["votes"]).most_common(1)[0]
                    match = v == majority + ("_unanimous" if count == 3 else "_split")
                elif s == "implicit_hate":
                    match = v == d["stage1_class"]
                else:
                    match = v == r["severity"]
        else:
            field, v = name.removeprefix("hatecheck_").split("/", 1)
            if source == "hatecheck":
                value = d["target_ident"] if field == "target_identity" else d[field]
                if value is None or isinstance(value, float) and math.isnan(value):
                    value = "not_specified"
                match = value == v
        if match:
            out.append(i)
    return out


def select(members, rows, error, votes, truth, view, exclusions):
    strict, rep, ex = VIEWS[view]
    eligible = [
        i
        for i in members
        if rows[i]["gate"] == truth and (not ex or rows[i]["key"] not in exclusions)
    ]
    if rep:
        selected = {}
        for i in eligible:
            label = rows[i]["inference_cluster"]
            rank = hashlib.sha256(
                ("A16-representative|" + rows[i]["key"]).encode()
            ).digest()
            if label not in selected or rank < selected[label][0]:
                selected[label] = (rank, i)
        keep = {pair[1] for pair in selected.values()}
        eligible = [i for i in eligible if i in keep]
    common = [
        i
        for i in eligible
        if all(
            error[i, c, t] >= 0 and (not strict or votes[i, c, t] == 3)
            for c in range(4)
            for t in range(3)
        )
    ]
    return eligible, common


def points(eligible, common, error, votes, strict):
    n = len(eligible)
    cn = len(common)
    actual = [int(error[common, c, :].sum()) for c in range(4)]
    observed = [0] * 4
    missing = [0] * 4
    for i in eligible:
        for c in range(4):
            for t in range(3):
                if error[i, c, t] < 0 or strict and votes[i, c, t] != 3:
                    missing[c] += 1
                else:
                    observed[c] += int(error[i, c, t])
    rate = lambda num, den: float(Fraction(int(num), den)) if den else None
    ref = rate(actual[0], 3 * cn)
    treat = rate(sum(actual[1:]), 9 * cn)
    rb = (
        [rate(observed[0], 3 * n), rate(observed[0] + missing[0], 3 * n)] if n else None
    )
    tb = (
        [
            rate(sum(observed[1:]), 9 * n),
            rate(sum(observed[1:]) + sum(missing[1:]), 9 * n),
        ]
        if n
        else None
    )
    return dict(
        eligible_n=n,
        common_n=cn,
        excluded_by_availability=n - cn,
        reference_rate=ref,
        treatment_rate=treat,
        difference=rate(sum(actual[1:]) - 3 * actual[0], 9 * cn),
        reference_error_count=actual[0],
        treatment_error_count=sum(actual[1:]),
        reference_output_count=3 * cn,
        treatment_output_count=9 * cn,
        pool_rates=[rate(x, 3 * cn) for x in actual[1:]] if cn else None,
        observed_errors_by_condition_full_eligible=observed,
        missing_outputs_by_condition_full_eligible=missing,
        reference_missing_bounds=rb,
        treatment_missing_bounds=tb,
        difference_missing_bounds=[tb[0] - rb[1], tb[1] - rb[0]] if n else None,
    )


def quantile(values, q):
    ordered = np.sort(values, axis=0)
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def replay(record, rows, error):
    ids = record["common_indices"]
    labels = sorted({rows[i]["inference_cluster"] for i in ids})
    index = {x: j for j, x in enumerate(labels)}
    text_to_unit = np.asarray(
        [index[rows[i]["inference_cluster"]] for i in ids], dtype=np.int64
    )
    totals = error[ids].sum(axis=2, dtype=np.int64)
    expected_seed = int.from_bytes(
        hashlib.sha256(("A16-part3-support-v1|" + record["id"]).encode()).digest()[:8],
        "little",
    )
    check(
        record["bootstrap"]["seed"] == expected_seed
        and record["bootstrap"]["units"] == len(labels),
        "Bootstrap convention",
    )
    saved = np.load(a.ROOT / record["draw_file"], allow_pickle=False)
    check(saved.shape == (10000, 6) and np.all(np.isfinite(saved)), "Draw shape/finite")
    rng = np.random.Generator(np.random.PCG64(expected_seed))
    for start in range(0, 10000, 32):
        count = min(32, 10000 - start)
        chosen = rng.integers(0, len(labels), (count, len(labels)), dtype=np.int64)
        weights = np.stack([np.bincount(x, minlength=len(labels)) for x in chosen])[
            :, text_to_unit
        ]
        n = weights.sum(axis=1)
        sums = weights @ totals
        expected = np.empty((count, 6), dtype=np.float64)
        expected[:, :4] = sums / (3 * n[:, None])
        expected[:, 4] = sums[:, 1:].sum(axis=1) / (9 * n)
        expected[:, 5] = (sums[:, 1:].sum(axis=1) - 3 * sums[:, 0]) / (9 * n)
        check(
            np.allclose(expected, saved[start : start + count], rtol=0, atol=1e-15),
            "Original-text bootstrap replay: " + record["id"],
        )
    near(
        record["interval95"],
        [quantile(saved[:, 5], 0.025), quantile(saved[:, 5], 0.975)],
        "Difference percentiles",
    )
    deltas = saved[:, 1:4] - saved[:, :1]
    near(
        record["pool_difference_intervals95"],
        np.stack([quantile(deltas, 0.025), quantile(deltas, 0.975)], axis=1).tolist(),
        "Pool percentiles",
    )
    check(
        record["degenerate_difference_draws"]
        == bool(record["interval95"][0] == record["interval95"][1]),
        "Degenerate flag",
    )
    return saved.size


def verify_description(record, desc, error, conditions):
    ids = record["common_indices"]
    n = len(ids)
    rr = error[ids]
    ref = rr[:, 0, :]
    for r in desc["runs"]:
        c = conditions.index(r["condition"])
        count = int(rr[:, c, r["repeat"]].sum())
        near(
            {k: r[k] for k in ("errors", "n", "rate")},
            dict(errors=count, n=n, rate=count / n),
            "Run rates",
        )
    check(len(desc["runs"]) == 12 and len(desc["pools"]) == 3, "Actual runs/pools")
    for c, p in enumerate(desc["pools"], 1):
        target = rr[:, c, :]
        fix = int(((ref == 1) & (target == 0)).sum())
        new = int(((ref == 0) & (target == 1)).sum())
        expected = dict(
            condition=conditions[c],
            errors=int(target.sum()),
            outputs=3 * n,
            rate=float(Fraction(int(target.sum()), 3 * n)),
            difference=float(Fraction(new - fix, 3 * n)),
            matched_repeat_fixes=fix,
            matched_repeat_new_errors=new,
            matched_repeat_persistent_errors=int(((ref == 1) & (target == 1)).sum()),
            matched_repeat_stable_correct=int(((ref == 0) & (target == 0)).sum()),
        )
        near(p, expected, "Pool/transition")
    directions = Counter()
    variation = Counter()
    for r in rr:
        counts = r.sum(axis=1)
        d = int(counts[1:].sum() - 3 * counts[0])
        directions[
            (
                "fewer_errors"
                if d < 0
                else "more_errors" if d > 0 else "same_average_errors"
            )
        ] += 1
        variation["reference_disagreement_inputs"] += int(len(set(r[0].tolist())) > 1)
        variation["any_treatment_repeat_disagreement_inputs"] += int(
            any(len(set(x.tolist())) > 1 for x in r[1:])
        )
        pd = [int(x - counts[0]) for x in counts[1:]]
        variation["pool_directions_disagree_inputs"] += int(min(pd) < 0 < max(pd))
        variation["all_twelve_correct"] += int(not r.any())
        variation["all_twelve_wrong"] += int(bool(r.all()))
        variation["equal_mean_with_offsetting_repeat_changes"] += int(
            d == 0 and np.any(r[1:] != r[:1])
        )
    near(desc["per_text_average_directions"], dict(directions), "Text directions")
    near(desc["variation"], dict(variation), "Variation")


def main():
    calc = a.read(OUT / "calculated.json")
    contract = a.read(OUT / "analysis_contract.json")
    for name, digest in {**calc["bindings"], **calc["artifacts"]}.items():
        check(a.sha(a.ROOT / name) == digest, "Changed file " + name)
    rows, raw, error, votes = rebuild()
    metadata = list(a.rows(OUT / "metadata.jsonl"))
    for i, (m, r, d) in enumerate(zip(metadata, rows, raw, strict=True)):
        for key in (
            "key",
            "id",
            "population",
            "source",
            "inference_cluster",
            "functional_cluster",
            "text_cluster",
            "overlap_cluster",
        ):
            check(m[key] == r[key], "Metadata " + key)
        check(
            m["gold_hate"] == r["gate"] and m["input_index"] == i, "Metadata gate/index"
        )
        check(m["fine_gold"] == {h: r.get(h) for h in HEADS}, "Fine gold metadata")
        if r["source"] == "hatecheck":
            for key, original in [
                ("functionality", "functionality"),
                ("template_id", "templ_id"),
                ("case_id", "case_id"),
                ("ref_case_id", "ref_case_id"),
                ("block", "block"),
                ("direction", "direction"),
            ]:
                check(m[key] == d[original], "HateCheck " + key)
            ident = d["target_ident"]
            ident = (
                "not_specified"
                if ident is None or isinstance(ident, float) and math.isnan(ident)
                else ident
            )
            check(m["target_identity"] == ident, "HateCheck identity")
        if r["source"] == "hatexplain":
            lab, count = Counter(d["votes"]).most_common(1)[0]
            check(
                m["annotator_votes"] == d["votes"]
                and m["majority_label"] == lab
                and m["original_split"] == d["official_split"],
                "HateXplain labels",
            )
            check(
                m["annotation_agreement"] == ("unanimous" if count == 3 else "split"),
                "Annotation agreement",
            )
        if r["source"] == "implicit_hate":
            check(m["stage1_class"] == d["stage1_class"], "Stage class")
        if r["source"] == "mhs":
            check(
                m["score"] == d["score"]
                and m["annotation_row_count"] == d["n_annotators"],
                "MHS raw metadata",
            )
    defs = a.read(OUT / "group_definitions.json")
    science = a.read(a.STUDY / "scientific_plan_v1.json")
    exclusions = set(science["mandatory_replay_keys"])
    expected_functions = {
        d["functionality"] for r, d in zip(rows, raw) if r["source"] == "hatecheck"
    }
    check(
        {n.split("/")[1] for n in defs if n.startswith("function/")}
        == expected_functions
        and len(expected_functions) == 29,
        "All functions published",
    )
    members = {}
    for name, spec in defs.items():
        members[name] = member(name, rows, raw)
        check(members[name] == spec["indices"], "Group membership " + name)
        if spec["family"] in ("planned_source", "planned_function"):
            check(
                spec["views"] == list(VIEWS) and spec["truths"] == [False, True],
                "All planned views retained",
            )
    rates = a.read(OUT / "rates.json")
    descs = a.read(OUT / "descriptions.json")
    contrib = a.read(OUT / "contributions.json")
    omits = a.read(OUT / "leave_one_cluster_out.json")
    prior = a.read(P2 / "rates.json")
    count_sets = scalar_count = reused = omission_count = 0
    expected_ids = {
        name + "/" + v + "/" + ("FNR" if t else "FPR")
        for name, spec in defs.items()
        for v in spec["views"]
        for t in spec["truths"]
    }
    check(set(rates) == expected_ids, "Complete rate inventory")
    for name, r in rates.items():
        eligible, common = select(
            members[r["group"]],
            rows,
            error,
            votes,
            r["endpoint"] == "FNR",
            r["view"],
            exclusions,
        )
        check(
            r["eligible_indices"] == eligible and r["common_indices"] == common,
            "Mask " + name,
        )
        p = points(eligible, common, error, votes, VIEWS[r["view"]][0])
        near({k: r[k] for k in p}, p, "Rates " + name)
        check(
            r["cluster_count"] == len({rows[i]["inference_cluster"] for i in common}),
            "Cluster support",
        )
        if r["population"] == "en_hatecheck_eval":
            check(
                r["template_count"] == len({raw[i]["templ_id"] for i in common}),
                "Template support",
            )
        status = (
            "defined"
            if common
            else (
                "undefined_no_eligible_gold"
                if not eligible
                else "undefined_no_common_outputs"
            )
        )
        check(r["status"] == status, "Undefined status")
        if "draw_file" in r:
            if "reused_from" in r:
                old = prior[r["reused_from"]["id"]]
                for field in (
                    "interval95",
                    "pool_difference_intervals95",
                    "bootstrap",
                    "draw_file",
                ):
                    near(r[field], old[field], "Reused source proof " + field)
                reused += 1
            else:
                scalar_count += replay(r, rows, error)
                count_sets += 1
                if count_sets % 20 == 0:
                    print(
                        f"Independently replayed {count_sets} new draw sets", flush=True
                    )
        if name in descs:
            verify_description(r, descs[name], error, science["conditions"])
            denominator = prior["main/" + r["population"] + "_" + r["endpoint"]][
                "common_n"
            ]
            numerator = p["treatment_error_count"] - 3 * p["reference_error_count"]
            near(
                contrib[name],
                dict(
                    difference_ninths=numerator,
                    endpoint_denominator=denominator,
                    contribution_to_population_rate=float(
                        Fraction(numerator, 9 * denominator)
                    ),
                ),
                "Contribution",
            )
            check(
                {x["cluster"] for x in omits[name]}
                == {rows[i]["inference_cluster"] for i in common},
                "Omission inventory",
            )
            for omitted in omits[name]:
                keep = [
                    i
                    for i in common
                    if rows[i]["inference_cluster"] != omitted["cluster"]
                ]
                n = len(keep)
                total = error[keep].sum(axis=(0, 2), dtype=np.int64)
                expected = (
                    float(Fraction(int(total[1:].sum() - 3 * total[0]), 9 * n))
                    if n
                    else None
                )
                near(
                    omitted,
                    dict(
                        cluster=omitted["cluster"],
                        removed_n=len(common) - n,
                        remaining_n=n,
                        difference=expected,
                    ),
                    "Cluster omission",
                )
                omission_count += 1
    for pop, prefix in [("en_test", "source/"), ("en_hatecheck_eval", "function/")]:
        for ep in ("FPR", "FNR"):
            subset = [
                r
                for r in rates.values()
                if r["group"].startswith(prefix)
                and r["view"] == "main"
                and r["endpoint"] == ep
            ]
            old = prior["main/" + pop + "_" + ep]
            check(
                sum(r["common_n"] for r in subset) == old["common_n"],
                "Partition common denominator",
            )
            for field in ("reference_error_count", "treatment_error_count"):
                check(
                    sum(r[field] for r in subset) == old[field],
                    "Partition error counts",
                )
            near(
                sum(
                    contrib[r["id"]]["contribution_to_population_rate"]
                    for r in subset
                    if r["common_n"]
                ),
                old["difference"],
                "Partition effects sum to Part 2",
            )
    table = a.read(OUT / "template_cluster_points.json")
    table_count = 0
    for field, records in table.items():
        expected = {
            (
                (
                    str(d["templ_id"])
                    if field == "template_id"
                    else r["inference_cluster"]
                ),
                r["gate"],
                view,
            )
            for r, d in zip(rows, raw)
            if r["source"] == "hatecheck"
            for view in ("main", "strict")
        }
        check(
            {(r["unit"], r["gold_hate"], r["view"]) for r in records} == expected,
            "Complete unit table",
        )
        for record in records:
            ids = [
                i
                for i, (r, d) in enumerate(zip(rows, raw))
                if r["source"] == "hatecheck"
                and str(
                    d["templ_id"] if field == "template_id" else r["inference_cluster"]
                )
                == record["unit"]
            ]
            e, c = select(
                ids, rows, error, votes, record["gold_hate"], record["view"], exclusions
            )
            check(
                record["eligible_indices"] == e and record["common_indices"] == c,
                "Unit masks",
            )
            check(
                record["functions"] == sorted({raw[i]["functionality"] for i in e}),
                "Unit functions",
            )
            expected_point = points(e, c, error, votes, record["view"] == "strict")
            near(
                {k: record[k] for k in expected_point},
                expected_point,
                "Template/cluster points",
            )
            table_count += 1
    support = a.read(OUT / "known_label_support.json")
    for name, views in support.items():
        ids = (
            members[name]
            if name.startswith("function/")
            else [i for i, r in enumerate(rows) if r["source"] == name]
        )
        for view, result in views.items():
            selected = [
                i
                for i in ids
                if view == "full"
                or (
                    np.all(error[i] >= 0)
                    and (view != "strict" or np.all(votes[i] == 3))
                )
            ]
            check(
                result["n"] == len(selected)
                and result["gold_hateful"] == sum(rows[i]["gate"] for i in selected),
                "Support cohorts",
            )
            for head, data in result["heads"].items():
                known = [
                    i
                    for i in selected
                    if rows[i].get(head) is not None
                    and (head == "legal" or rows[i]["gate"])
                ]
                positives = Counter()
                for i in known:
                    value = rows[i][head]
                    positives.update([value] if isinstance(value, str) else value)
                near(
                    data,
                    dict(
                        eligible_known=len(known),
                        unknown_or_ineligible=len(selected) - len(known),
                        known_empty=sum(rows[i][head] == [] for i in known),
                        positive_support=dict(positives),
                    ),
                    "Known label support",
                )
    for population, sources in [
        ("en_test", ["hatexplain", "implicit_hate", "mhs"]),
        ("en_hatecheck_eval", ["hatecheck"]),
    ]:
        for head in HEADS:
            spec = science["fixed_support"][population][head]
            check(
                sum(
                    support[s]["full"]["heads"][head]["eligible_known"] for s in sources
                )
                == spec["rows"],
                "Frozen fine denominator",
            )
            combined = Counter()
            for s in sources:
                combined.update(support[s]["full"]["heads"][head]["positive_support"])
            check(
                dict(combined) == spec["positive_support"],
                "Frozen fine positive support",
            )
    structure = a.read(OUT / "structure.json")
    for field, record in structure["hatecheck"].items():
        units = defaultdict(list)
        for i, (r, d) in enumerate(zip(rows, raw)):
            if r["source"] == "hatecheck":
                units[
                    str(d["templ_id"] if field == "template_id" else r[field])
                ].append(i)
        check(record["count"] == len(units), "Structure count")
        check(
            record["largest_size"] == max(map(len, units.values())), "Largest component"
        )
        check(
            record["size_histogram"]
            == {str(k): v for k, v in Counter(map(len, units.values())).items()},
            "Size histogram",
        )
        check(
            record["crosses_function"]
            == sum(
                len({raw[i]["functionality"] for i in ids}) > 1
                for ids in units.values()
            ),
            "Cross-function units",
        )
        check(
            record["crosses_gold"]
            == sum(len({rows[i]["gate"] for i in ids}) > 1 for ids in units.values()),
            "Cross-polarity units",
        )
        for unit, ids in units.items():
            near(
                record["members"][unit],
                dict(
                    keys=[rows[i]["key"] for i in ids],
                    functions=sorted({raw[i]["functionality"] for i in ids}),
                    gold_counts=dict(Counter(str(rows[i]["gate"]) for i in ids)),
                ),
                "Structure membership",
            )
    pairs = a.read(OUT / "linked_reference_pairs.json")
    by_case = {
        d["case_id"]: i
        for i, (r, d) in enumerate(zip(rows, raw))
        if r["source"] == "hatecheck"
    }
    expected_pairs = []
    for i, (r, d) in enumerate(zip(rows, raw)):
        if r["source"] == "hatecheck" and d["ref_case_id"] is not None:
            j = by_case[d["ref_case_id"]]
            check(
                r["inference_cluster"] == rows[j]["inference_cluster"],
                "Reference edge stays in cluster",
            )
            expected_pairs.append(
                dict(
                    child=i,
                    parent=j,
                    child_key=r["key"],
                    parent_key=rows[j]["key"],
                    function=d["functionality"],
                    parent_function=raw[j]["functionality"],
                    gold_relation=(
                        "opposite" if r["gate"] != rows[j]["gate"] else "same"
                    ),
                    cluster=r["inference_cluster"],
                )
            )
    near(pairs, expected_pairs, "All original reference pairs")
    linked = a.read(OUT / "linked_reference_summary.json")
    check(
        {(r["function"], r["gold_relation"], r["view"]) for r in linked}
        == {
            (p["function"], p["gold_relation"], v)
            for p in pairs
            for v in ("main", "strict")
        },
        "Pair summary inventory",
    )
    for record in linked:
        candidates = [
            p
            for p in pairs
            if p["function"] == record["function"]
            and p["gold_relation"] == record["gold_relation"]
        ]
        common = [
            p
            for p in candidates
            if all(
                error[i, c, t] >= 0
                and (record["view"] != "strict" or votes[i, c, t] == 3)
                for i in [p["child"], p["parent"]]
                for c in range(4)
                for t in range(3)
            )
        ]
        counts = [
            [
                sum(
                    error[p["child"], c, t] == 0 and error[p["parent"], c, t] == 0
                    for p in common
                )
                for t in range(3)
            ]
            for c in range(4)
        ]
        n = len(common)
        ref = sum(counts[0]) / (3 * n) if n else None
        treat = sum(sum(c) for c in counts[1:]) / (9 * n) if n else None
        near(
            record,
            dict(
                function=record["function"],
                gold_relation=record["gold_relation"],
                view=record["view"],
                eligible_pairs=len(candidates),
                common_pairs=n,
                clusters=len({p["cluster"] for p in common}),
                condition_repeat_both_correct=counts,
                reference_both_correct=ref,
                treatment_both_correct=treat,
                difference=treat - ref if n else None,
            ),
            "Linked pair performance",
        )
    check(
        count_sets == calc["new_draw_sets"] == 130
        and reused == calc["reused_draw_sets"] == 6,
        "Draw inventory",
    )
    write(
        "independent_check.json",
        dict(
            status="all_part3_numerical_checks_pass",
            checked_utc=a.utc(),
            rate_views=len(rates),
            new_bootstrap_sets=count_sets,
            new_bootstrap_draws=count_sets * 10000,
            new_scalar_values=scalar_count,
            reused_source_sets=reused,
            template_cluster_points=table_count,
            leave_one_cluster_out_points=omission_count,
            linked_pairs=len(pairs),
            metadata_rows=len(metadata),
            known_support_groups=len(support),
            group_descriptions=len(descs),
            original_metadata_join=True,
            all_functions_and_sources_retained=True,
            undefined_endpoints_retained=True,
            all_four_primary_count_and_difference_partitions_match=True,
            original_work_unchanged=True,
            source_sha256=a.sha(Path(__file__)),
            calculated_sha256=a.sha(OUT / "calculated.json"),
            new_Qwen_calls=0,
            new_fits=0,
        ),
    )
    write(
        "verified.json",
        dict(
            status="part3_numerically_verified_interpretation_pending",
            calculated_sha256=a.sha(OUT / "calculated.json"),
            independent_check_sha256=a.sha(OUT / "independent_check.json"),
            checked_utc=a.utc(),
        ),
    )
    print("All Part 3 numerical checks passed.", flush=True)


if __name__ == "__main__":
    with a.offline():
        main()

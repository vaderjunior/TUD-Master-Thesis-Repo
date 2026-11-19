from collections import Counter, defaultdict
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from analysis.feedback.rounds import subgroups_inputs as a


def near(x, y):
    a.need(
        x is not None and math.isfinite(x) and abs(x - y) < 1e-13,
        "Independent arithmetic differs",
    )


def terms(column):
    rounds, policies = column.split("_", 1)
    return [
        (
            int(r[1:]) - 1,
            a.POLICIES.index(p),
            (1 if ri == 0 else -1) * (1 if pi == 0 else -1),
        )
        for ri, r in enumerate(rounds.split("-"))
        for pi, p in enumerate(policies.split("-"))
    ]


def main():
    assembly = a.read(a.OUT / "assembled.json")
    contract = a.read(a.OUT / "analysis_contract.json")
    for path, digest in {
        **assembly["input_bindings"],
        **assembly["artifacts"],
        **contract["source_hashes"],
    }.items():
        a.need(a.sha(a.ROOT / path) == digest, "Verified input changed: " + path)
    result = a.read(a.OUT / "analysis.json")
    meta = list(a.rows(a.OUT / "metadata.jsonl"))
    matrix = list(a.rows(a.READER / "matrix.jsonl"))
    original = list(a.rows(a.STUDY / "evaluation.jsonl"))
    groups = a.read(a.OUT / "groups.json")
    n = len(original)
    counts = np.zeros((n, 4, 4), dtype=np.int64)
    miss = np.zeros_like(counts)
    code = np.full((n, 4, 10, 3), -1, dtype=np.int8)
    valid = np.zeros((n, 4), bool)
    strict = np.zeros_like(valid)
    independent_meta = []
    for i, (raw, mm, derived) in enumerate(zip(original, matrix, meta, strict=True)):
        a.need(
            raw["key"] == mm["key"] == derived["key"] and i == derived["input_index"],
            "Metadata key order differs",
        )
        v = json.loads(raw["raw"])
        source = raw["source"]
        m = dict(source=source, gold_hate=raw["gate"])
        if source == "hatecheck":
            fn = v["functionality"]
            a.need(
                fn in contract["functions"] and raw["gate"] == fn.endswith("_h"),
                "Function gold differs",
            )
            target = v["target_ident"]
            target = (
                "not_specified"
                if target is None or isinstance(target, float) and math.isnan(target)
                else target
            )
            fields = dict(
                functionality=fn,
                template_id=v["templ_id"],
                case_id=v["case_id"],
                ref_case_id=v["ref_case_id"],
                target_identity=target,
                block=v["block"],
                direction=v["direction"],
            )
        elif source == "hatexplain":
            votes = Counter(v["votes"])
            label, num = votes.most_common(1)[0]
            a.need(
                num >= 2
                and len(v["votes"]) == 3
                and raw["gate"] == (label == "hatespeech"),
                "HateXplain gold differs",
            )
            fields = dict(
                majority_label=label,
                annotation_agreement="unanimous" if num == 3 else "split",
                annotator_votes=v["votes"],
                original_split=v["official_split"],
            )
        elif source == "implicit_hate":
            a.need(
                raw["gate"] == (v["stage1_class"] != "not_hate"),
                "Implicit Hate gold differs",
            )
            fields = dict(stage1_class=v["stage1_class"])
        else:
            score = v["score"]
            a.need(raw["gate"] == (score > 0.5), "MHS gold differs")
            expected = (
                None
                if score <= 0.5
                else "low" if score <= 1 else "medium" if score <= 1.63 else "high"
            )
            a.need(raw["severity"] == expected, "Frozen MHS bands differ")
            fields = dict(score=score, annotation_row_count=v["n_annotators"])
        for key, value in fields.items():
            a.need(derived[key] == value, "Raw annotation mapping differs")
        for f in ["hate_types", "target_groups", "severity", "legal"]:
            a.need(
                derived["fine_gold"][f] == raw.get(f), "Unknown fine support changed"
            )
        for f in [
            "key",
            "source",
            "text",
            "population",
            "inference_cluster",
            "functional_cluster",
            "text_cluster",
            "overlap_cluster",
        ]:
            a.need(derived[f] == raw[f], "Metadata field differs")
        m.update(fields)
        independent_meta.append(m)
        for rnd, arms in enumerate(mm["predictions"]):
            valid[i, rnd] = all(p["result"] is not None for arm in arms for p in arm)
            strict[i, rnd] = all(p["n_valid"] == 3 for arm in arms for p in arm)
            for ci, arm in enumerate(arms):
                for rep, p in enumerate(arm):
                    if p["result"] is not None:
                        code[i, rnd, ci, rep] = int(p["result"]["hate"] != raw["gate"])
            for pi, policy in enumerate(a.POLICIES):
                cs = (
                    [0]
                    if pi == 0
                    else [
                        j
                        for j, c in enumerate(a.CONDITIONS)
                        if c.startswith(policy + "_s")
                    ]
                )
                vals = [code[i, rnd, j, rep] for j in cs for rep in range(3)]
                factor = 3 if pi == 0 else 1
                counts[i, rnd, pi] = sum(v == 1 for v in vals) * factor
                miss[i, rnd, pi] = sum(v < 0 for v in vals) * factor
        a.need(
            derived["masks"]["round_common"] == valid[i].tolist()
            and derived["masks"]["round_strict"] == strict[i].tolist()
            and derived["masks"]["all_round_common"] == bool(valid[i].all()),
            "Inherited validity mask differs",
        )

    def members(group):
        parts = group.split("/")
        family = parts[0]
        if family == "source":
            return [
                i for i, m in enumerate(independent_meta) if m["source"] == parts[1]
            ]
        if family == "function":
            return [
                i
                for i, m in enumerate(independent_meta)
                if m.get("functionality") == parts[1]
            ]
        if family == "metadata":
            return [
                i
                for i, m in enumerate(independent_meta)
                if m["source"] == "hatecheck"
                and str(m[parts[1]]) == "/".join(parts[2:])
            ]
        source = parts[1]
        value = "/".join(parts[2:])
        ids = []
        for i, m in enumerate(independent_meta):
            if m["source"] != source:
                continue
            label = (
                m["majority_label"] + "/" + m["annotation_agreement"]
                if source == "hatexplain"
                else (
                    m["stage1_class"]
                    if source == "implicit_hate"
                    else original[i]["severity"]
                )
            )
            if label == value:
                ids.append(i)
        return ids

    for group, g in groups.items():
        a.need(
            members(group) == g["indices"], "Independent subgroup membership differs"
        )

    def eligible(group, ep, view):
        return [
            i
            for i in members(group)
            if original[i]["gate"] == (ep == "FNR")
            and (
                valid[i].all()
                if view == "fixed"
                else (
                    strict[i, 3]
                    if view == "strict"
                    else valid[i, 3 if view == "final" else int(view[-1]) - 1]
                )
            )
        ]

    points = scalars = drawsets = undefined = 0
    for name, rec in result["records"].items():
        ids = eligible(rec["group"], rec["endpoint"], rec["view"])
        a.need(
            ids == rec["input_indices"] and len(ids) == rec["inputs"],
            "Independent scoring cohort differs",
        )
        if not ids:
            a.need(
                rec["status"] == "undefined_no_valid_class_support"
                and rec["points"] is None
                and rec["intervals"] is None,
                "Undefined is not zero",
            )
            undefined += 1
            continue
        columns = rec["column_order"]
        values = np.array(
            [
                [sum(w * int(counts[i, r, p]) for r, p, w in terms(c)) for c in columns]
                for i in ids
            ],
            dtype=np.int64,
        )
        for j, col in enumerate(columns):
            num = sum(int(v) for v in values[:, j])
            a.need(
                num == rec["integer_numerators"][col]
                and rec["denominator"] == 9 * len(ids),
                "Exact subgroup fraction differs",
            )
            near(rec["points"][col], float(Fraction(num, 9 * len(ids))))
            points += 1
        clusters = [original[i]["inference_cluster"] for i in ids]
        labels = sorted(set(clusters))
        membership = np.array([labels.index(v) for v in clusters])
        a.need(
            labels == rec["cluster_labels"]
            and [clusters.count(c) for c in labels] == rec["cluster_sizes"],
            "Cluster membership differs",
        )
        if "draw_file" not in rec:
            a.need(rec["intervals"] is None, "Unexpected unverified interval")
            continue
        expected_seed = int.from_bytes(
            hashlib.sha256(("A17-part5-support-v1|" + name).encode()).digest()[:8],
            "little",
        )
        a.need(
            rec["seed"] == expected_seed
            and rec["confidence"] == 0.95
            and len(labels) > 1,
            "Bootstrap plan differs",
        )
        path = a.ROOT / rec["draw_file"]
        a.need(a.sha(path) == rec["draw_sha256"], "Saved bootstrap changed")
        with np.load(path, allow_pickle=False) as archive:
            draws = archive["draws"]
        a.need(draws.shape == (10000, len(columns)), "Draw shape differs")
        rng = np.random.Generator(np.random.PCG64(expected_seed))
        for start in range(0, 10000, 32):
            end = min(start + 32, 10000)
            chosen = rng.integers(0, len(labels), size=(end - start, len(labels)))
            weights = np.array(
                [np.bincount(v, minlength=len(labels))[membership] for v in chosen]
            )
            calc = (weights @ values) / (9 * weights.sum(axis=1))[:, None]
            a.need(
                np.array_equal(calc, draws[start:end]),
                "Independent row-multiplicity bootstrap differs",
            )
            scalars += calc.size
        ordered = np.sort(draws, axis=0)
        for j, col in enumerate(columns):
            for side, q in enumerate([0.025, 0.975]):
                index = 9999 * q
                lo = math.floor(index)
                frac = index - lo
                near(
                    rec["intervals"][col][side],
                    float(
                        ordered[lo, j] * (1 - frac)
                        + ordered[min(lo + 1, 9999), j] * frac
                    ),
                )
        a.need(
            rec["degenerate_columns"]
            == [c for j, c in enumerate(columns) if len(set(draws[:, j])) == 1],
            "Degenerate flags differ",
        )
        drawsets += 1
    print(
        f"Independent rates and all bootstraps pass: {points} points, {scalars} scalars.",
        flush=True,
    )
    for r in result["supports"]:
        ids = [
            i
            for i in members(r["group"])
            if original[i]["gate"] == (r["endpoint"] == "FNR")
        ]
        a.need(
            len(ids) == r["roster"]
            and len({original[i]["inference_cluster"] for i in ids})
            == r["inference_clusters"],
            "Full support differs",
        )
        for view in ["fixed", "final", "strict"]:
            a.need(
                len(eligible(r["group"], r["endpoint"], view)) == r[view],
                "View support differs",
            )
        a.need(
            r["round_counts"]
            == [
                len(eligible(r["group"], r["endpoint"], "round" + str(j)))
                for j in range(1, 5)
            ],
            "Round supports differ",
        )
        a.need(
            r["fixed_clusters"]
            == len(
                {
                    original[i]["inference_cluster"]
                    for i in eligible(r["group"], r["endpoint"], "fixed")
                }
            ),
            "Fixed clusters differ",
        )
    for r in result["final_missing_bounds"]:
        ids = [
            i
            for i in members(r["group"])
            if original[i]["gate"] == (r["endpoint"] == "FNR")
        ]
        if not ids:
            a.need(r["values"] is None, "Empty bound should be undefined")
            continue
        b = r["values"]
        n9 = 9 * len(ids)
        ranges = {}
        for p, policy in enumerate(a.POLICIES):
            err = int(counts[ids, 3, p].sum())
            missing = int(miss[ids, 3, p].sum())
            v = b["policies"][policy]
            a.need(
                v["error_numerator_over_nine"] == err
                and v["missing_numerator_over_nine"] == missing
                and v["denominator_over_nine"] == n9,
                "Missing-bound components",
            )
            ranges[policy] = (Fraction(err, n9), Fraction(err + missing, n9))
            for actual, expected in zip(v["bound"], ranges[policy]):
                near(actual, float(expected))
            if n9 > missing:
                near(v["available_case_rate"], err / (n9 - missing))
        for contrast, interval in b["contrasts"].items():
            left, right = contrast.split("-")
            near(interval[0], float(ranges[left][0] - ranges[right][1]))
            near(interval[1], float(ranges[left][1] - ranges[right][0]))
    for r in result["fine_support"]:
        ids = (
            [
                i
                for i in members(r["group"])
                if original[i]["gate"] == (r["endpoint"] == "FNR")
            ]
            if r["view"] == "roster"
            else eligible(r["group"], r["endpoint"], r["view"])
        )
        a.need(len(ids) == r["inputs"], "Fine support denominator")
        for field, record in r["fields"].items():
            values = [original[i].get(field) for i in ids]
            a.need(
                record["known"] == sum(v is not None for v in values)
                and record["unknown"] == sum(v is None for v in values)
                and record["known_empty"] == sum(v == [] for v in values),
                "Known/unknown fine distinction",
            )
            a.need(
                record["labels"]
                == dict(
                    Counter(
                        v
                        for labels in values
                        if isinstance(labels, list)
                        for v in labels
                    )
                ),
                "Fine positive label supports",
            )
    for r in result["durability"]:
        ids = eligible(r["group"], r["endpoint"], "fixed")
        p = a.POLICIES.index(r["policy"])
        vals = counts[ids, :, p]
        ref = counts[ids, :, 0]
        flags = defaultdict(int)
        for e, re in zip(vals, ref):
            flags["all_saved_correct"] += bool(np.all(e == 0))
            flags["all_saved_wrong"] += bool(np.all(e == 9))
            flags["persistent_error"] += bool(re[0] == 9 and np.all(e == 9))
            flags["sustained_new_error"] += bool(
                re[0] == 0 and any(np.all(e[start:] == 9) for start in range(3))
            )
            flags["recovered_new_error"] += bool(
                re[0] == 0 and np.any(e[:3] > 0) and e[3] == 0
            )
            flags["delayed_clean_suffix"] += bool(
                re[0] == 9 and e[3] == 0 and np.any(e > 0)
            )
            flags["retained_mean_advantage"] += bool(np.all(e < re))
            flags["mixed_mean_direction"] += bool(np.any(e < re) and np.any(e > re))
            flags["r4_vs_r1_improved"] += bool(e[3] < e[0])
            flags["r4_vs_r1_worsened"] += bool(e[3] > e[0])
        a.need(
            len(ids) == r["inputs"]
            and all(r[key] == value for key, value in flags.items()),
            "Subgroup durability differs",
        )
    for r in result["function_influence"]:
        ids = eligible(
            "function/" + r["function"],
            "FNR" if r["function"].endswith("_h") else "FPR",
            "fixed",
        )
        labels = sorted({original[i]["inference_cluster"] for i in ids})
        a.need(
            len(labels) == r["clusters"] and len(ids) == r["inputs"],
            "Influence supports",
        )

        def calculate(keep, col):
            return float(
                Fraction(
                    sum(
                        sum(w * int(counts[i, rr, p]) for rr, p, w in terms(col))
                        for i in keep
                    ),
                    9 * len(keep),
                )
            )

        for j, col in enumerate(r["columns"]):
            near(r["full"][j], calculate(ids, col))
            near(
                r["equal_cluster_mean"][j],
                sum(
                    calculate(
                        [i for i in ids if original[i]["inference_cluster"] == c], col
                    )
                    for c in labels
                )
                / len(labels),
            )
        for row in r["leave_one_out"]:
            keep = [
                i for i in ids if original[i]["inference_cluster"] != row["cluster"]
            ]
            a.need(
                row["remaining_inputs"] == len(keep), "Leave-one-cluster denominator"
            )
            for actual, col in zip(row["values"], r["columns"]):
                near(actual, calculate(keep, col))
    for r in result["template_points"]:
        ids = [
            i
            for i in members("function/" + r["function"])
            if independent_meta[i]["template_id"] == r["template"]
        ]
        keep = [i for i in ids if valid[i].all()]
        a.need(
            r["roster"] == len(ids)
            and r["input_indices"] == keep
            and r["denominator"] == 9 * len(keep),
            "Template support differs",
        )
        a.need(
            r["error_numerators"] == counts[keep].sum(axis=0).tolist(),
            "Template error counts differ",
        )
    edges = list(a.rows(a.OUT / "linked_edges.jsonl"))
    parent_by_case = {
        m["case_id"]: i
        for i, m in enumerate(independent_meta)
        if m["source"] == "hatecheck"
    }
    expected_children = [
        i
        for i, m in enumerate(independent_meta)
        if m["source"] == "hatecheck" and m["ref_case_id"] is not None
    ]
    a.need(
        [r["child"] for r in edges] == expected_children,
        "Reference-edge coverage differs",
    )
    pair_values = defaultdict(list)
    for r in edges:
        child = r["child"]
        parent = parent_by_case.get(int(independent_meta[child]["ref_case_id"]))
        state = (
            "parent_outside_fixed_roster"
            if parent is None
            else (
                "missing_output"
                if not (valid[child].all() and valid[parent].all())
                else "usable"
            )
        )
        a.need(r["parent"] == parent and r["status"] == state, "Reference edge status")
        if state != "usable":
            continue
        vals = np.zeros((4, 4), int)
        for rnd in range(4):
            for p, policy in enumerate(a.POLICIES):
                cs = (
                    [0]
                    if p == 0
                    else [
                        j
                        for j, c in enumerate(a.CONDITIONS)
                        if c.startswith(policy + "_s")
                    ]
                )
                vals[rnd, p] = sum(
                    code[child, rnd, j, rep] == 0 and code[parent, rnd, j, rep] == 0
                    for j in cs
                    for rep in range(3)
                ) * (3 if p == 0 else 1)
        a.need(
            r["joint_correct_over_nine"] == vals.tolist(),
            "Linked both-correct arithmetic",
        )
        pair_values[(r["function"], r["same_gold"])].append(vals)
    for r in result["linked_pairs"]:
        a.need(
            r["joint_correct_numerators"]
            == np.array(pair_values[r["function"], r["same_gold"]])
            .sum(axis=0)
            .tolist(),
            "Linked pair summary differs",
        )

    selection = a.read(a.OUT / "case_selection.json")
    a.need(len(selection) == 53, "Expected 24 source and 29 function case slots")
    for r in selection:
        ids = eligible(r["group"], r["endpoint"], "fixed")
        cat = r["category"]

        def rank(i):
            return hashlib.sha256(
                f'A17-part5-case-v1|{r["group"]+("/"+r["endpoint"] if cat!="function_extreme" else "")}|{cat}|{original[i]["key"]}'.encode()
            ).hexdigest()

        if cat == "function_extreme":
            keep = ids
            chosen = (
                min(
                    keep,
                    key=lambda i: (
                        -abs(int(counts[i, 3, 1] - counts[i, 3, 0])),
                        -int(counts[i, 3, 1]),
                        rank(i),
                    ),
                )
                if keep
                else None
            )
        else:
            keep = []
            for i in ids:
                e = counts[i, :, 1]
                delta = e - counts[i, :, 0]
                yes = (
                    np.all(delta < 0)
                    if cat == "retained_average_benefit"
                    else (
                        delta[3] > 0
                        if cat == "final_regression"
                        else (
                            np.any(delta < 0) and np.any(delta > 0)
                            if cat == "mixed_relative_direction"
                            else np.all(e == 9)
                        )
                    )
                )
                if yes:
                    keep.append(i)
            chosen = min(keep, key=rank) if keep else None
        a.need(
            r["candidates"] == len(keep)
            and r["input_index"] == chosen
            and r["status"] == ("empty" if chosen is None else "selected"),
            "Independent case rule differs",
        )
    a.write(
        a.OUT / "independent_verification.json",
        dict(
            status="part5_independent_verification_pass",
            created_utc=a.pure.utc(),
            inputs={a.rel(a.OUT / "assembled.json"): a.sha(a.OUT / "assembled.json")},
            source_sha256=a.sha(Path(__file__)),
            metadata_rows=n,
            evaluation_gate_slots=int(code.size),
            evaluation_nulls=int((code < 0).sum()),
            records=len(result["records"]),
            exact_points=points,
            undefined_views=undefined,
            bootstrap_sets=drawsets,
            bootstrap_scalars=scalars,
            max_bootstrap_difference=0,
            supports=len(result["supports"]),
            fine_support_cells=len(result["fine_support"]),
            durability_cells=len(result["durability"]),
            functions=len(result["function_influence"]),
            templates=len(result["template_points"]),
            linked_edges=len(edges),
            case_slots=len(selection),
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        "Independent Part 5 metadata, subgroup estimates, intervals, supports, influence, pairs and cases PASS.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()

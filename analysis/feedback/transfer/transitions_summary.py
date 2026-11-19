from collections import Counter
from pathlib import Path
import numpy as np
from analysis.feedback.transfer import transitions as p
from analysis.feedback.transfer import transitions_metrics as k

a = p.a
OUT = p.OUT


def counts4(values):
    return np.bincount(np.asarray(values).reshape(-1), minlength=4)


def text_counts(s):
    ref = s[:, 0].sum(axis=1)
    pool = s[:, 1:].sum(axis=2)
    trt = pool.sum(axis=1)
    delta = trt - 3 * ref
    paired = 2 * s[:, 0, None] + s[:, 1:]
    fix = (paired == 1).sum(axis=(1, 2))
    reg = (paired == 2).sum(axis=(1, 2))
    pd = pool - ref[:, None]
    return dict(
        direction=dict(
            improved=int((delta > 0).sum()),
            regressed=int((delta < 0).sum()),
            equal_average=int((delta == 0).sum()),
        ),
        stability=dict(
            consistent_fix=int(((ref == 0) & (trt == 9)).sum()),
            consistent_regression=int(((ref == 3) & (trt == 0)).sum()),
            persistently_correct=int(((ref == 3) & (trt == 9)).sum()),
            persistently_wrong=int(((ref == 0) & (trt == 0)).sum()),
            variable=int(
                (~(((ref == 0) | (ref == 3)) & ((trt == 0) | (trt == 9)))).sum()
            ),
        ),
        any_aligned_fix=int((fix > 0).sum()),
        any_aligned_regression=int((reg > 0).sum()),
        offsetting_aligned_changes=int(((fix > 0) & (reg > 0)).sum()),
        equal_average_with_aligned_changes=int(
            ((delta == 0) & (fix > 0) & (reg > 0)).sum()
        ),
        opposing_pool_directions=int(
            ((pd > 0).any(axis=1) & (pd < 0).any(axis=1)).sum()
        ),
        ref_vs_feedback_correct_histogram=np.bincount(
            ref * 10 + trt, minlength=40
        ).reshape(4, 10),
        ref_vs_each_pool_correct_histogram=np.stack(
            [
                np.bincount(ref * 4 + pool[:, c], minlength=16).reshape(4, 4)
                for c in range(3)
            ]
        ),
        pool_direction_histogram=dict(
            Counter(",".join(str(int(np.sign(v))) for v in row) for row in pd)
        ),
    )


def transition_summary(ids, head, arrays, meta):
    ids = np.asarray(ids, dtype=int)
    h = k.HEADS.index(head)
    s = arrays["scores"][ids, ..., h]
    N = len(ids)
    a.need(np.all(s >= 0), "Invalid score in existing common mask")
    x = arrays["transitions"][ids, ..., h]
    ct = counts4(x)
    perms = np.stack(
        [counts4(2 * s[:, 0, None] + s[:, 1:][:, :, perm]) for perm in k.PERMUTATIONS]
    )
    cross = perms.sum(axis=0) // 2
    a.need(np.all(perms.sum(axis=0) % 2 == 0), "All repeat alignments reconcile")
    a.need(
        all(q[1] - q[2] == ct[1] - ct[2] for q in perms),
        "Repeat alignment changed net effect",
    )
    by = np.stack([counts4(x[:, c, r]) for c in range(3) for r in range(3)]).reshape(
        3, 3, 4
    )
    binary = arrays["transitions"][ids, ..., 0]
    joint = arrays["transitions"][ids, ..., 1]
    bjoint = np.bincount((4 * binary + joint).reshape(-1), minlength=16).reshape(4, 4)
    out = dict(
        head=head,
        texts=N,
        clusters=len({meta[i]["inference_cluster"] for i in ids}),
        dependent_pairs=9 * N,
        counts=dict(zip(k.TRANSITIONS, ct)),
        per_pool_repeat_counts=by,
        reference_actual_correct=s[:, 0].sum(),
        feedback_actual_correct=s[:, 1:].sum(),
        reference_rate=float(s[:, 0].mean()) if N else None,
        feedback_rate=float(s[:, 1:].mean()) if N else None,
        exact_difference=float((ct[1] - ct[2]) / (9 * N)) if N else None,
        error_difference=float((ct[2] - ct[1]) / (9 * N)) if N else None,
        repeat_alignments=dict(
            permutations=k.PERMUTATIONS,
            counts=perms,
            all_cross_repeat_counts=cross,
            all_cross_repeat_pairs=27 * N,
        ),
        text_patterns=text_counts(s),
        actual_correct_by_condition_repeat=s.sum(axis=0),
        binary_vs_known_exact_transition_matrix=bjoint,
        head_fix_with_binary_fix=int(((x == 1) & (binary == 1)).sum()),
        binary_fix_without_head_exact=int(((binary == 1) & (s[:, 1:] == 0)).sum()),
    )
    return p.native(out)


def disagreement(codes):
    out = []
    for r, s in ((0, 1), (0, 2), (1, 2)):
        d = codes[:, r] != codes[:, s]
        out.append(np.concatenate([d, d.any(axis=-1, keepdims=True)], axis=-1))
    return np.stack(out, axis=1)


def main():
    assembled = a.read(OUT / "assembled.json")
    for name, digest in {
        **assembled["analysis_sources"],
        **assembled["artifacts"],
    }.items():
        a.need(a.sha(a.ROOT / name) == digest, "Changed Part 6 assembly")
    a.need(not (OUT / "calculated.json").exists(), "Preserve previous analysis")
    meta = list(a.rows(p.P4 / "metadata.jsonl"))
    matrix = list(a.rows(a.OUT / "matrix.jsonl"))
    N = len(meta)
    with np.load(OUT / "arrays.npz", allow_pickle=False) as f:
        x = {key: f[key] for key in f}
    quality = a.read(p.P4 / "quality_results.json")
    rates = a.read(p.P2 / "rates.json")
    summary = {}
    checks = []
    for name, q in quality.items():
        prefix = name.split("/")[0]
        if prefix not in ("population", "source", "function") or not q["common_n"]:
            continue
        if prefix != "population" and q["view"] not in ("main", "strict"):
            continue
        if prefix == "function" and q["view"] != "main":
            continue
        base = q["head"]
        if base not in (
            "binary",
            "known_answer",
            "hate_type",
            "target_group",
            "severity",
        ):
            continue
        for head in (
            [base, "gated_" + base]
            if base in ("hate_type", "target_group", "severity")
            else [base]
        ):
            out = transition_summary(q["common_indices"], head, x, meta)
            newname = name.replace("/" + base + "/", "/" + head + "/")
            metric = (
                "accuracy"
                if head == "binary"
                else (
                    "known_answer_exact"
                    if head == "known_answer"
                    else "gated_exact_set" if head.startswith("gated_") else "exact_set"
                )
            )
            prior = q["metrics"][metric]
            a.need(
                abs(out["exact_difference"] - prior["difference"]) < 1e-13,
                "Prior exact effect " + newname,
            )
            out.update(
                id=newname,
                part4_id=name,
                part4_metric=metric,
                reused_interval95=prior.get("interval95"),
                common_indices=q["common_indices"],
            )
            summary[newname] = out
            checks.append(newname)
    for pop in ("en_test", "en_hatecheck_eval"):
        for ep in ("FPR", "FNR"):
            for view in p.VIEWS:
                q = rates[view + "/" + pop + "_" + ep]
                name = f"endpoint/{pop}_{ep}/{view}"
                out = transition_summary(q["common_indices"], "binary", x, meta)
                a.need(
                    abs(out["error_difference"] - q["difference"]) < 1e-13,
                    "Primary error transition reconciliation",
                )
                out.update(
                    id=name,
                    part2_id=q["id"],
                    reused_interval95=q["interval95"],
                    reused_interval98_75=q["interval98_75"],
                    common_indices=q["common_indices"],
                )
                summary[name] = out
                checks.append(name)
    p.save("transition_summary.json", summary)
    repeat = {}
    missing = {}
    identical = {}
    for pop in ("en_test", "en_hatecheck_eval"):
        allids = [i for i, m in enumerate(meta) if m["population"] == pop]
        nvalid = x["n_valid"][allids]
        missing[pop] = dict(
            texts=len(allids),
            vote_count_histogram_by_condition=[
                [int((nvalid[:, c] == n).sum()) for n in range(4)] for c in range(4)
            ],
            common_texts=sum(matrix[i]["masks"]["common_valid"] for i in allids),
            strict_texts=sum(matrix[i]["masks"]["strict_all_votes"] for i in allids),
            pair_status_counts=dict(
                Counter(
                    (
                        "valid"
                        if x["n_valid"][i, 0, r] and x["n_valid"][i, c, r]
                        else (
                            "both_missing"
                            if x["n_valid"][i, 0, r] == x["n_valid"][i, c, r] == 0
                            else (
                                "reference_missing"
                                if x["n_valid"][i, 0, r] == 0
                                else "feedback_missing"
                            )
                        )
                    )
                    for i in allids
                    for c in range(1, 4)
                    for r in range(3)
                )
            ),
        )
        for view in p.VIEWS:
            ids = quality[f"population/{pop}/binary/{view}"]["common_indices"]
            for gold in ("all", 0, 1):
                ii = np.asarray(
                    [
                        i
                        for i in ids
                        if gold == "all" or int(meta[i]["gold_hate"]) == gold
                    ],
                    dtype=int,
                )
                n = len(ii)
                for c in range(4):
                    code = x["codes"][ii, c]
                    dis = disagreement(code)
                    div = dis.sum(axis=1)
                    unique = np.where(div == 0, 1, np.where(div == 2, 2, 3))
                    nv = x["n_valid"][ii, c]
                    vd = x["vote_distinct"][ii, c]
                    strict = nv == 3
                    name = f"{pop}/{view}/gold_{gold}/{p.CONDITIONS[c]}"
                    repeat[name] = p.native(
                        dict(
                            id=name,
                            texts=n,
                            actual_outputs=3 * n,
                            fields=(*k.FIELDS, "full_output"),
                            aggregate_repeat_distinct_histogram=np.stack(
                                [(unique == z).sum(axis=0) for z in (1, 2, 3)]
                            ),
                            aggregate_repeat_disagree_pairs=dis.sum(axis=(0, 1)),
                            aggregate_repeat_available_pairs=3 * n,
                            vote_count_histogram=[
                                int((nv == z).sum()) for z in range(4)
                            ],
                            three_vote_outputs=int(strict.sum()),
                            three_vote_distinct_histogram=np.stack(
                                [
                                    ((vd == z) & strict[..., None]).sum(axis=(0, 1))
                                    for z in (1, 2, 3)
                                ]
                            ),
                            valid_vote_disagree_pairs=x["vote_disagree_pairs"][
                                ii, c
                            ].sum(axis=(0, 1)),
                            valid_vote_available_pairs=int((nv * (nv - 1) // 2).sum()),
                            aggregate_not_observed=int(
                                x["aggregate_not_observed"][ii, c].sum()
                            ),
                            aggregate_not_observed_three_votes=int(
                                (x["aggregate_not_observed"][ii, c] & strict).sum()
                            ),
                            gate_ties=int(x["gate_ties"][ii, c].sum()),
                        )
                    )
            if view not in ("main", "strict"):
                continue
            for kind in ("same_messages", "different_messages"):
                selected = []
                count = np.zeros(6, dtype=np.int64)
                allcross = np.zeros(6, dtype=np.int64)
                perpair = {}
                for c in range(1, 4):
                    for d in range(c + 1, 4):
                        take = [
                            i
                            for i in ids
                            if (x["messages"][i, c] == x["messages"][i, d])
                            == (kind == "same_messages")
                        ]
                        ii = np.asarray(take, dtype=int)
                        cc = x["codes"][ii, c]
                        dd = x["codes"][ii, d]
                        z = cc != dd
                        diff = np.concatenate(
                            [z, z.any(axis=-1, keepdims=True)], axis=-1
                        ).sum(axis=(0, 1))
                        count += diff
                        v = cc[:, :, None] != dd[:, None, :]
                        v = np.concatenate(
                            [v, v.any(axis=-1, keepdims=True)], axis=-1
                        ).sum(axis=(0, 1, 2))
                        allcross += v
                        perpair[f"{c}-{d}"] = dict(
                            texts=len(take),
                            aligned_disagreements=diff,
                            all_cross_disagreements=v,
                        )
                        selected.extend(take)
                name = f"{pop}/{view}/{kind}"
                identical[name] = p.native(
                    dict(
                        id=name,
                        query_pool_pairs=len(selected),
                        unique_texts=len(set(selected)),
                        aligned_output_pairs=3 * len(selected),
                        aligned_disagreements=count,
                        all_cross_output_pairs=9 * len(selected),
                        all_cross_disagreements=allcross,
                        per_pool_pair=perpair,
                        fields=(*k.FIELDS, "full_output"),
                        note="Selected subsets differ. Same messages establish observed generation variability, not a randomized noise subtraction or backend attribution.",
                    )
                )
    p.save("repeat_variation.json", repeat)
    p.save("missing_outputs.json", missing)
    p.save("same_message_variation.json", identical)

    edits = {}
    priorpairs = a.read(p.P4 / "conditional_gate_pairs.json")
    for pop, head in [
        ("en_test", "hate_type"),
        ("en_test", "target_group"),
        ("en_test", "severity"),
        ("en_hatecheck_eval", "target_group"),
    ]:
        j = k.FIELDS.index(head)
        labels = ["low", "medium", "high"] if head == "severity" else k.LABELS[head]
        for view in ("main", "strict"):
            name = f"population/{pop}/{head}/{view}"
            ids = quality[name]["common_indices"]
            ii = np.asarray(ids, dtype=int)
            old = x["codes"][ii, 0, :, j][:, None, :]
            new = x["codes"][ii, 1:, :, j]
            if head == "severity":
                old = np.where(old >= 0, 1 << np.maximum(old, 0), 0)
                new = np.where(new >= 0, 1 << np.maximum(new, 0), 0)
            gold = np.array(
                [
                    (
                        1 << labels.index(meta[i][k.GOLD[head]])
                        if head == "severity"
                        else sum(1 << labels.index(v) for v in meta[i][k.GOLD[head]])
                    )
                    for i in ids
                ],
                dtype=np.int16,
            )[:, None, None]
            added = new & ~old
            removed = old & ~new
            bothpos = (x["codes"][ii, 0, :, 0][:, None, :] == 1) & (
                x["codes"][ii, 1:, :, 0] == 1
            )
            rows = {}
            for part, mask in [
                ("all", np.ones(new.shape, dtype=bool)),
                ("both_positive", bothpos),
            ]:
                each = {}
                for l, label in enumerate(labels):
                    bit = 1 << l
                    g = (gold & bit) > 0
                    ad = (added & bit) > 0
                    rm = (removed & bit) > 0
                    each[label] = dict(
                        correct_added=int((ad & g & mask).sum()),
                        correct_removed=int((rm & g & mask).sum()),
                        extra_added=int((ad & ~g & mask).sum()),
                        extra_removed=int((rm & ~g & mask).sum()),
                    )
                rows[part] = dict(pairs=int(mask.sum()), labels=each)
                if view == "main":
                    oldsum = priorpairs[name]["partitions"][part]
                    for key, oldkey in [
                        ("correct_added", "correct_labels_added"),
                        ("correct_removed", "correct_labels_removed"),
                        ("extra_added", "extra_labels_added"),
                        ("extra_removed", "extra_labels_removed"),
                    ]:
                        a.need(
                            sum(q[key] for q in each.values()) == oldsum[oldkey],
                            "Existing label edit totals",
                        )
            edits[name] = rows
    p.save("fine_label_edits.json", edits)
    files = [
        OUT / n
        for n in (
            "transition_summary.json",
            "repeat_variation.json",
            "missing_outputs.json",
            "same_message_variation.json",
            "fine_label_edits.json",
        )
    ]
    p.save(
        "calculated.json",
        dict(
            status="part6_descriptive_transitions_calculated_verification_pending",
            created_utc=a.utc(),
            assembled_sha256=a.sha(OUT / "assembled.json"),
            script_sha256=a.sha(Path(__file__)),
            artifacts={a.rel(f): a.sha(f) for f in files},
            transition_views=len(summary),
            repeat_views=len(repeat),
            same_message_views=len(identical),
            fine_edit_views=len(edits),
            reconciled_prior_net_effects=len(checks),
            new_bootstrap_sets=0,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        f"Calculated {len(summary)} transition, {len(repeat)} repeat and {len(identical)} message-variation views; all prior net effects reconcile."
    )


if __name__ == "__main__":
    with a.offline():
        main()

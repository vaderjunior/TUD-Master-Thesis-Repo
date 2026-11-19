from collections import Counter
from pathlib import Path
import numpy as np
from analysis.feedback.rounds import quality_inputs as a
from analysis.feedback.rounds import quality_metrics as q

k = a.quality


def pids(policy):
    return (
        [0]
        if policy == "reference"
        else [a.CONDITIONS.index(f"{policy}_s{s}") for s in a.STREAMS]
    )


def slots(values, policy):
    return values[:, :, pids(policy), :].reshape(len(values), 4, -1)


def partition(left, right, lg, rg, keys, binary_left=None, binary_right=None):
    masks = {
        "all": np.ones(left.shape, dtype=bool),
        "both_positive": lg & rg,
        "gained_positive_gate": ~lg & rg,
        "lost_positive_gate": lg & ~rg,
        "both_negative": ~lg & ~rg,
    }
    out = {}
    for label, mask in masks.items():
        group = dict(
            pairs=int(mask.sum()),
            unique_texts=int(np.any(mask, axis=1).sum()),
            exact_fix=int((mask & ~left & right).sum()),
            exact_regression=int((mask & left & ~right).sum()),
            both_exact=int((mask & left & right).sum()),
            neither_exact=int((mask & ~left & ~right).sum()),
        )
        if binary_left is not None:
            fix = ~binary_left & binary_right
            group.update(
                binary_fix=int((mask & fix).sum()),
                binary_fix_with_exact=int((mask & fix & right).sum()),
                binary_fix_without_exact=int((mask & fix & ~right).sum()),
                binary_regression=int((mask & binary_left & ~binary_right).sum()),
            )
        a.need(
            group["pairs"]
            == sum(
                group[v]
                for v in [
                    "exact_fix",
                    "exact_regression",
                    "both_exact",
                    "neither_exact",
                ]
            ),
            "Pair partition",
        )
        out[label] = group
    return out


def main():
    a.need(
        a.read(a.OUT / "independent_verification.json")["status"]
        == "part6_independent_verification_pass",
        "Independent scoring check required",
    )
    result = a.read(a.OUT / "analysis.json")
    meta = list(a.rows(a.OUT / "metadata.jsonl"))
    records = result["records"]
    with np.load(a.OUT / "features.npz", allow_pickle=False) as f:
        data = {h: f[h] for h in f.files}
    with np.load(a.OUT / "prediction_codes.npz", allow_pickle=False) as f:
        codes = {h: f[h] for h in f.files}
    exact = {
        h: data[h][..., k.names(h).index("exact")].astype(bool)
        for h in a.HEADS
        if h != "binary"
    }
    binary = (
        data["binary"][..., k.names("binary").index("tp")]
        + data["binary"][..., k.names("binary").index("tn")]
    ).astype(bool)
    perlabel = {}
    shapes = {}
    for name, r in records.items():
        h = r["head"]
        if (
            h not in ["hate_type", "target_group", "severity"]
            or not r["n"]
            or r["view"] not in ["fixed", "final", "strict"]
        ):
            continue
        count = np.asarray(r["counts"])
        cols = k.names(h)
        ix = {v: j for j, v in enumerate(cols)}
        rounds = [1, 2, 3, 4] if r["view"] == "fixed" else [4]
        labels = k.LABELS[h]
        rows = {}
        for label in labels:
            stats = np.stack(
                [count[..., ix[label + ":" + v]] for v in ["tp", "fp", "fn"]], axis=-1
            )
            tp, fp, fn = [stats[..., j] for j in range(3)]
            score = np.stack(
                [
                    k.divide(tp, tp + fp, 0),
                    k.divide(tp, tp + fn, 0),
                    k.divide(2 * tp, 2 * tp + fp + fn, 0),
                ],
                axis=-1,
            )
            rows[label] = dict(
                in_macro=label in r["macro_labels"],
                eligible_positive_support=sum(
                    (
                        meta[i][k.GOLD[h]] == label
                        if h == "severity"
                        else label in meta[i][k.GOLD[h]]
                    )
                    for i in r["eligible_indices"]
                ),
                common_positive_support=sum(
                    (
                        meta[i][k.GOLD[h]] == label
                        if h == "severity"
                        else label in meta[i][k.GOLD[h]]
                    )
                    for i in r["indices"]
                ),
                counts=stats[np.array(rounds) - 1],
                run_precision_recall_f1=score[np.array(rounds) - 1],
                policy_precision_recall_f1=q.policy_means(score)[np.array(rounds) - 1],
            )
        perlabel[name] = dict(rounds=rounds, labels=rows)
        if h == "severity":
            confusion = np.stack(
                [
                    count[..., ix[f"confusion:{g}:{p}"]]
                    for g in labels
                    for p in ["none", *labels]
                ],
                axis=-1,
            ).reshape(4, 10, 3, 3, 4)
            shapes[name] = dict(
                rounds=rounds,
                confusion=confusion[np.array(rounds) - 1],
                gold_order=labels,
                prediction_order=["none", *labels],
                none_with_positive_gate=[
                    (
                        (codes[h][r["indices"], rnd - 1] < 0)
                        & (codes["gates"][r["indices"], rnd - 1] == 1)
                    ).sum(axis=0)
                    for rnd in rounds
                ],
            )
        else:
            gold = [len(set(meta[i][k.GOLD[h]])) for i in r["indices"]]
            hist = []
            for rnd in rounds:
                hist.append(
                    [
                        [
                            np.bincount(
                                [
                                    int(v).bit_count()
                                    for v in codes[h][r["indices"], rnd - 1, ci, rep]
                                ],
                                minlength=len(labels) + 1,
                            )
                            for rep in range(3)
                        ]
                        for ci in range(10)
                    ]
                )
            shapes[name] = dict(
                rounds=rounds,
                gold_cardinality=np.bincount(gold, minlength=len(labels) + 1),
                predicted_cardinality=hist,
                known_empty=sum(v == 0 for v in gold),
                mean_gold_labels=np.mean(gold),
                categories={
                    v: count[np.array(rounds) - 1, ..., ix[v]]
                    for v in [
                        "exact",
                        "contains_gold",
                        "only_extra",
                        "only_missing",
                        "both_extra_missing",
                        "extra_labels",
                        "missing_labels",
                    ]
                },
            )
    pair_records = []
    paths = []
    path_summaries = []
    for pop in a.POPS:
        for head in ["hate_type", "target_group", "severity", "known_answer"]:
            main = records[f"population/{pop}|{head}|final"]
            fixed = records[f"population/{pop}|{head}|fixed"]
            if not main["n"]:
                continue
            ids = main["indices"]
            keys = [meta[i]["key"] for i in ids]
            ex = exact[head][ids]
            g = codes["gates"][ids] == 1
            b = binary[ids]
            for left, right in [("reference", p) for p in a.POLICIES[1:]] + [
                ("static", "adaptive"),
                ("random", "adaptive"),
            ]:
                l = slots(ex, left)[:, 3]
                r = slots(ex, right)[:, 3]
                lg = slots(g, left)[:, 3]
                rg = slots(g, right)[:, 3]
                bl = slots(b, left)[:, 3]
                br = slots(b, right)[:, 3]
                if left == "reference":
                    l = np.tile(l, (1, 3))
                    lg = np.tile(lg, (1, 3))
                    bl = np.tile(bl, (1, 3))
                pair_records.append(
                    dict(
                        population=pop,
                        head=head,
                        kind="final_policy",
                        left=left,
                        right=right,
                        left_round=4,
                        right_round=4,
                        inputs=ids,
                        partitions=partition(l, r, lg, rg, keys, bl, br),
                    )
                )
            ids = fixed["indices"]
            keys = [meta[i]["key"] for i in ids]
            ex = exact[head][ids]
            g = codes["gates"][ids] == 1
            b = binary[ids]
            for policy in a.POLICIES[1:]:
                vals = slots(ex, policy)
                gv = slots(g, policy)
                bv = slots(b, policy)
                for rnd in range(1, 4):
                    pair_records.append(
                        dict(
                            population=pop,
                            head=head,
                            kind="adjacent_round",
                            left=policy,
                            right=policy,
                            left_round=rnd,
                            right_round=rnd + 1,
                            inputs=ids,
                            partitions=partition(
                                vals[:, rnd - 1],
                                vals[:, rnd],
                                gv[:, rnd - 1],
                                gv[:, rnd],
                                keys,
                                bv[:, rnd - 1],
                                bv[:, rnd],
                            ),
                        )
                    )
            ref = slots(ex, "reference")
            refg = slots(g, "reference")
            for n, i in enumerate(ids):
                policies = {}
                for policy in a.POLICIES[1:]:
                    vals = slots(ex, policy)[n]
                    gv = slots(g, policy)[n]
                    num = vals.sum(axis=1)
                    rn = ref[n].sum(axis=1)
                    policies[policy] = dict(
                        exact_by_repeat=vals.astype(int),
                        gate_by_repeat=gv.astype(int),
                        exact_counts=num,
                        relative_path="".join(
                            "B" if x > 3 * y else "W" if x < 3 * y else "T"
                            for x, y in zip(num, rn)
                        ),
                        always_exact=bool(vals.all()),
                        never_exact=bool(not vals.any()),
                        improved=int(num[3]) > int(num[0]),
                        worsened=int(num[3]) < int(num[0]),
                    )
                paths.append(
                    dict(
                        input_index=i,
                        key=meta[i]["key"],
                        population=pop,
                        head=head,
                        gold_profile=meta[i]["gold_profile"],
                        reference_exact_by_repeat=ref[n].astype(int),
                        reference_gate_by_repeat=refg[n].astype(int),
                        policies=policies,
                    )
                )
            selected = [
                p for p in paths if p["population"] == pop and p["head"] == head
            ]
            for policy in a.POLICIES[1:]:
                pp = [p["policies"][policy] for p in selected]
                path_summaries.append(
                    dict(
                        population=pop,
                        head=head,
                        policy=policy,
                        inputs=len(ids),
                        all36_exact=sum(p["always_exact"] for p in pp),
                        never_exact=sum(p["never_exact"] for p in pp),
                        retained_mean_benefit=sum(
                            p["relative_path"] == "BBBB" for p in pp
                        ),
                        relative_paths=dict(Counter(p["relative_path"] for p in pp)),
                        improved=sum(p["improved"] for p in pp),
                        worsened=sum(p["worsened"] for p in pp),
                        equal=sum(not p["improved"] and not p["worsened"] for p in pp),
                    )
                )

    cases = []

    def choose(pop, head, category, candidates):
        group = pop + "/" + head
        ids = sorted(set(candidates))
        i = (
            min(ids, key=lambda i: a.rank(group, category, meta[i]["key"]))
            if ids
            else None
        )
        cases.append(
            dict(
                group=group,
                population=pop,
                head=head,
                category=category,
                eligible_candidates=len(ids),
                status="selected" if i is not None else "empty",
                input_index=i,
                key=meta[i]["key"] if i is not None else None,
                selection_hash=(
                    a.rank(group, category, meta[i]["key"]) if i is not None else None
                ),
            )
        )

    for pop, head in [
        ("en_test", "hate_type"),
        ("en_test", "target_group"),
        ("en_test", "severity"),
        ("en_hatecheck_eval", "target_group"),
    ]:
        pp = [p for p in paths if p["population"] == pop and p["head"] == head]
        choose(
            pop,
            head,
            "retained_mean_exact_benefit",
            [
                p["input_index"]
                for p in pp
                if p["policies"]["adaptive"]["relative_path"] == "BBBB"
            ],
        )
        regression = []
        gate_only = []
        for p in pp:
            ad = p["policies"]["adaptive"]
            ref = np.tile(p["reference_exact_by_repeat"][3], 3).astype(bool)
            rg = np.tile(p["reference_gate_by_repeat"][3], 3).astype(bool)
            ae = np.asarray(ad["exact_by_repeat"][3], dtype=bool)
            ag = np.asarray(ad["gate_by_repeat"][3], dtype=bool)
            if np.any(ref & ~ae & rg & ag):
                regression.append(p["input_index"])
            if np.any(~rg & ag & ~ae):
                gate_only.append(p["input_index"])
        choose(pop, head, "fine_regression_both_gates_positive", regression)
        choose(pop, head, "gate_fix_without_fine_exact", gate_only)
        if head != "severity":
            choose(
                pop,
                head,
                "known_empty_gold",
                [
                    p["input_index"]
                    for p in pp
                    if meta[p["input_index"]][k.GOLD[head]] == []
                ],
            )
    for pop in a.POPS:
        pp = [
            p
            for p in paths
            if p["population"] == pop
            and p["head"] == "known_answer"
            and meta[p["input_index"]]["any_fine_gold"]
        ]
        for category, last in [
            ("final_known_exact_benefit", "B"),
            ("final_known_exact_regression", "W"),
        ]:
            choose(
                pop,
                "known_answer",
                category,
                [
                    p["input_index"]
                    for p in pp
                    if p["policies"]["adaptive"]["relative_path"][-1] == last
                ],
            )
    a.lines(a.OUT / "input_quality_paths.jsonl", paths)
    a.write(a.OUT / "case_selection.json", cases)
    a.write(
        a.OUT / "diagnostics.json",
        dict(
            status="part6_diagnostics_pending_independent_readback",
            created_utc=a.pure.utc(),
            per_label=perlabel,
            set_and_severity=shapes,
            pair_transitions=pair_records,
            path_summaries=path_summaries,
            note="All gate partitions are outcome-selected descriptive paired counts. Reused reference answers do not add independent samples. Fine exact and gate-plus-fine exact differ on known-empty gold.",
            new_Qwen_calls=0,
        ),
    )
    files = ["input_quality_paths.jsonl", "case_selection.json", "diagnostics.json"]
    a.write(
        a.OUT / "diagnostics_complete.json",
        dict(
            status="part6_diagnostics_calculated_pending_independent_readback",
            created_utc=a.pure.utc(),
            inputs={
                a.rel(a.OUT / f): a.sha(a.OUT / f)
                for f in [
                    "analysis.json",
                    "metadata.jsonl",
                    "features.npz",
                    "prediction_codes.npz",
                    "independent_verification.json",
                ]
            },
            artifacts={a.rel(a.OUT / f): a.sha(a.OUT / f) for f in files},
            source_sha256=a.sha(Path(__file__)),
            label_views=len(perlabel),
            pair_views=len(pair_records),
            input_head_paths=len(paths),
            case_slots=len(cases),
            unique_case_inputs=len(
                {c["input_index"] for c in cases if c["status"] == "selected"}
            ),
            empty_slots=sum(c["status"] == "empty" for c in cases),
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        f"Part 6 detail: {len(perlabel)} label/shape views, {len(pair_records)} pair views, {len(paths)} paths; {len(cases)} case slots.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()

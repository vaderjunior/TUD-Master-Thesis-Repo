from collections import Counter
from fractions import Fraction
from itertools import combinations, permutations
from pathlib import Path
import numpy as np
from analysis.feedback.transfer import transitions as p

a = p.a
OUT = p.OUT
FIELDS = ("hate", "hate_type", "target_group", "severity", "legal")
HEADS = (
    "binary",
    "known_answer",
    "hate_type",
    "gated_hate_type",
    "target_group",
    "gated_target_group",
    "severity",
    "gated_severity",
)
GOLD = {
    "hate_type": "gold_hate_types",
    "target_group": "gold_target_groups",
    "severity": "gold_severity",
    "legal": "gold_legal",
}
ELIG = {
    "hate_type": "hate_types",
    "target_group": "target_groups",
    "severity": "severity",
    "legal": "legal",
}
STATES = ("unchanged_error", "fix", "regression", "unchanged_correct")


def sig(value):
    return (
        None
        if value is None
        else (
            value["hate"],
            tuple(sorted(value["hate_type"])),
            tuple(sorted(value["target_group"])),
            value["severity"],
            tuple(sorted(value["legal"])),
        )
    )


def equal(left, right, label):
    if isinstance(left, np.ndarray):
        left = left.tolist()
    if isinstance(right, np.ndarray):
        right = right.tolist()
    if isinstance(left, dict):
        a.need(set(left) == set(right), "Keys differ " + label)
        for key in left:
            equal(left[key], right[key], label + "/" + str(key))
    elif isinstance(left, (tuple, list)):
        a.need(len(left) == len(right), "Length differs " + label)
        for j, (x, y) in enumerate(zip(left, right)):
            equal(x, y, label + "/" + str(j))
    elif isinstance(left, (float, np.floating)) or isinstance(
        right, (float, np.floating)
    ):
        a.need(
            left is not None
            and right is not None
            and abs(float(left) - float(right)) < 1e-12,
            "Float differs " + label,
        )
    else:
        a.need(left == right, "Value differs " + label)


def profile(s):
    if min(v for row in s for v in row) < 0:
        return {"status": "unavailable"}
    rc = sum(s[0])
    tc = [sum(row) for row in s[1:]]
    total = sum(tc)
    net = total - 3 * rc
    fs = sum(not s[0][r] and s[c][r] for c in (1, 2, 3) for r in range(3))
    rs = sum(s[0][r] and not s[c][r] for c in (1, 2, 3) for r in range(3))
    if rc == 0 and total == 9:
        stable = "consistent_fix"
    elif rc == 3 and total == 0:
        stable = "consistent_regression"
    elif rc == 3 and total == 9:
        stable = "persistently_correct"
    elif rc == 0 and total == 0:
        stable = "persistently_wrong"
    else:
        stable = "variable"
    dd = [1 if n > rc else -1 if n < rc else 0 for n in tc]
    return dict(
        status="complete",
        reference_correct=rc,
        pool_correct=tc,
        feedback_correct=total,
        improvement_numerator_over_9=net,
        direction=(
            "improved" if net > 0 else "regressed" if net < 0 else "equal_average"
        ),
        stability=stable,
        fixes=fs,
        regressions=rs,
        offsetting=fs > 0 and rs > 0,
        pool_directions=dd,
        opposing_pool_directions=min(dd) < 0 < max(dd),
    )


def main():
    a.need(not (OUT / "verified.json").exists(), "Preserve previous verification")
    assembled = a.read(OUT / "assembled.json")
    calc = a.read(OUT / "calculated.json")
    for name, digest in {
        **assembled["analysis_sources"],
        **assembled["artifacts"],
        **calc["artifacts"],
    }.items():
        a.need(a.sha(a.ROOT / name) == digest, "Changed evidence " + name)
    matrix = list(a.rows(a.OUT / "matrix.jsonl"))
    meta = list(a.rows(p.P4 / "metadata.jsonl"))
    N = len(matrix)
    with np.load(OUT / "arrays.npz", allow_pickle=False) as f:
        x = {name: f[name] for name in f}
    schema = a.read(OUT / "array_schema.json")
    labels = schema["prediction_codes"]["labels"]

    def decode(v):
        if v[0] == -99:
            return None
        return (
            bool(v[0]),
            tuple(
                sorted(l for j, l in enumerate(labels["hate_type"]) if v[1] & (1 << j))
            ),
            tuple(
                sorted(
                    l for j, l in enumerate(labels["target_group"]) if v[2] & (1 << j)
                )
            ),
            None if v[3] < 0 else ("low", "medium", "high")[v[3]],
            tuple(sorted(l for j, l in enumerate(labels["legal"]) if v[4] & (1 << j))),
        )

    S = np.full(x["scores"].shape, -2, dtype=np.int8)
    signatures = []
    for i, row in enumerate(matrix):
        mm = []
        for c, preds in enumerate(row["predictions"]):
            ss = []
            for r, item in enumerate(preds):
                val = item["result"]
                v = sig(val)
                ss.append(v)
                a.need(v == decode(x["codes"][i, c, r]), "Original canonical output")
                a.need(
                    item["n_valid"] == x["n_valid"][i, c, r]
                    and item["uncertain"] == x["gate_ties"][i, c, r],
                    "Original availability flags",
                )
                for h, head in enumerate(HEADS):
                    base = head.removeprefix("gated_")
                    eligible = (
                        head in ("binary", "known_answer")
                        or row["eligible_fine_gold"][ELIG[base]]
                    )
                    if not eligible:
                        continue
                    if val is None:
                        S[i, c, r, h] = -1
                        continue
                    if head == "binary":
                        correct = val["hate"] == row["gold_hate"]
                    elif head == "known_answer":
                        correct = val["hate"] == row["gold_hate"]
                        for f, g in GOLD.items():
                            if row["eligible_fine_gold"][ELIG[f]]:
                                correct = correct and (
                                    val[f] == row[g]
                                    if f == "severity"
                                    else set(val[f]) == set(row[g])
                                )
                    else:
                        correct = (
                            val[base] == row[GOLD[base]]
                            if base == "severity"
                            else set(val[base]) == set(row[GOLD[base]])
                        )
                        if head.startswith("gated_"):
                            correct = correct and val["hate"]
                    S[i, c, r, h] = int(correct)
            mm.append(ss)
        signatures.append(mm)
    a.need(np.array_equal(S, x["scores"]), "All reconstructed correctness cells")
    expected = np.full(x["transitions"].shape, -2, dtype=np.int8)
    ref = S[:, 0, None]
    trt = S[:, 1:]
    expected[((ref == -1) | (trt == -1)) & (ref != -2) & (trt != -2)] = -1
    for value, (r, t) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        expected[(ref == r) & (trt == t)] = value
    a.need(np.array_equal(expected, x["transitions"]), "Independent transition enum")
    for i, c, r in np.ndindex(N, 3, 3):
        left = signatures[i][0][r]
        right = signatures[i][c + 1][r]
        bits = (
            [-1] * 6
            if left is None or right is None
            else [int(v != w) for v, w in zip(left, right)] + [int(left != right)]
        )
        equal(bits, x["changed"][i, c, r], "Canonical pair changes")

    provenance = iter(p.rows(OUT / "prediction_provenance.jsonl.gz"))
    seen = set()
    checkedvotes = 0
    for path in sorted((a.OUT / "assembly").glob("transfer_*.jsonl")):
        for r in a.rows(path):
            i, c, t = r["input_index"], p.CONDITIONS.index(r["condition"]), r["repeat"]
            slot = (i, c, t)
            a.need(slot not in seen, "Duplicate source output")
            seen.add(slot)
            pr = next(provenance)
            equal(pr, {key: r[key] for key in pr}, "Prediction provenance")
            a.need(
                sig(r["result"]) == signatures[i][c][t]
                and r["messages_sha256"] == x["messages"][i, c],
                "Original output/message",
            )
            vv = [sig(v) for v in r["votes"]]
            a.need(vv == [decode(v) for v in x["votes"][i, c, t]], "Original votes")
            valid = [v for v in vv if v is not None]
            checkedvotes += len(valid)
            ds = []
            dp = []
            for j in range(6):
                v = [q[j] if j < 5 else q for q in valid]
                ds.append(len(set(v)))
                dp.append(sum(z != w for z, w in combinations(v, 2)))
            equal(ds, x["vote_distinct"][i, c, t], "Vote distinct")
            equal(dp, x["vote_disagree_pairs"][i, c, t], "Vote pairs")
            a.need(
                bool(x["aggregate_not_observed"][i, c, t])
                == (r["result"] is not None and sig(r["result"]) not in valid),
                "Composite aggregate membership",
            )
        print("Independent saved-vote check: " + path.stem, flush=True)
    a.need(
        len(seen) == 189096
        and next(provenance, None) is None
        and checkedvotes == 560485,
        "Complete original prediction coverage",
    )
    profiles = []
    for row in p.rows(OUT / "per_text.jsonl.gz"):
        i = row["input_index"]
        a.need(
            row["key"] == matrix[i]["key"] and row["masks"] == matrix[i]["masks"],
            "Text ledger identity",
        )
        pp = {head: profile(S[i, ..., h].tolist()) for h, head in enumerate(HEADS)}
        equal(row["profiles"], pp, "Per-text profiles")
        profiles.append(pp)
    a.need(len(profiles) == N, "Text ledger size")
    n = 0
    for row in p.rows(OUT / "transition_pairs.jsonl.gz"):
        i, c, t = row["input_index"], row["pool_index"], row["repeat"]
        a.need(
            n == i * 9 + (c - 1) * 3 + t and row["key"] == meta[i]["key"],
            "Pair order/identity",
        )
        equal(
            row["transitions"],
            dict(zip(HEADS, expected[i, c - 1, t].tolist())),
            "Pair transition values",
        )
        equal(
            row["changed"],
            dict(zip((*FIELDS, "full_output"), x["changed"][i, c - 1, t].tolist())),
            "Pair change values",
        )
        old, new = signatures[i][0][t], signatures[i][c][t]
        status = (
            "both_missing"
            if old is None and new is None
            else (
                "reference_missing"
                if old is None
                else "feedback_missing" if new is None else "valid"
            )
        )
        a.need(
            status == row["status"]
            and row["same_messages"] == (x["messages"][i, 0] == x["messages"][i, c]),
            "Pair availability/messages",
        )
        a.need(
            row["reference_messages_sha256"] == x["messages"][i, 0]
            and row["feedback_messages_sha256"] == x["messages"][i, c],
            "Pair message hashes",
        )
        n += 1
    a.need(n == 141822, "All dependent pairs")
    summary = a.read(OUT / "transition_summary.json")
    quality = a.read(p.P4 / "quality_results.json")
    rates = a.read(p.P2 / "rates.json")
    for name, q in summary.items():
        ids = q["common_indices"]
        h = HEADS.index(q["head"])
        s = S[ids, ..., h]
        ct = Counter()
        by = np.zeros((3, 3, 4), dtype=int)
        for c, t in np.ndindex(3, 3):
            v = s[:, 0, t]
            w = s[:, c + 1, t]
            counts = [
                int(((v == r) & (w == z)).sum())
                for r, z in [(0, 0), (0, 1), (1, 0), (1, 1)]
            ]
            by[c, t] = counts
            ct.update(dict(zip(STATES, counts)))
        equal(q["counts"], dict(ct), "Summary transition counts")
        equal(q["per_pool_repeat_counts"], by, "Run-specific counts")
        points = [profiles[i][q["head"]] for i in ids]
        pat = q["text_patterns"]
        for field, possible in [
            ("direction", ["improved", "regressed", "equal_average"]),
            (
                "stability",
                [
                    "consistent_fix",
                    "consistent_regression",
                    "persistently_correct",
                    "persistently_wrong",
                    "variable",
                ],
            ),
        ]:
            counts = Counter(z[field] for z in points)
            equal(pat[field], {v: counts[v] for v in possible}, "Text categories")
        for field, fn in [
            ("any_aligned_fix", lambda z: z["fixes"] > 0),
            ("any_aligned_regression", lambda z: z["regressions"] > 0),
            ("offsetting_aligned_changes", lambda z: z["offsetting"]),
            (
                "equal_average_with_aligned_changes",
                lambda z: z["direction"] == "equal_average" and z["offsetting"],
            ),
            ("opposing_pool_directions", lambda z: z["opposing_pool_directions"]),
        ]:
            equal(pat[field], sum(fn(z) for z in points), field)
        hist = Counter((z["reference_correct"], z["feedback_correct"]) for z in points)
        equal(
            pat["ref_vs_feedback_correct_histogram"],
            [[hist[r, t] for t in range(10)] for r in range(4)],
            "Text correctness histogram",
        )
        for c in range(3):
            hist = Counter(
                (z["reference_correct"], z["pool_correct"][c]) for z in points
            )
            equal(
                pat["ref_vs_each_pool_correct_histogram"][c],
                [[hist[r, t] for t in range(4)] for r in range(4)],
                "Pool correctness histogram",
            )
        equal(
            pat["pool_direction_histogram"],
            dict(Counter(",".join(map(str, z["pool_directions"])) for z in points)),
            "Pool direction signatures",
        )
        numerator = ct["fix"] - ct["regression"]
        denom = 9 * len(ids)
        equal(
            q["exact_difference"],
            float(Fraction(numerator, denom)),
            "Net exact Fraction",
        )
        equal(
            q["error_difference"],
            float(Fraction(-numerator, denom)),
            "Net error Fraction",
        )
        equal(
            q["reference_actual_correct"],
            int(s[:, 0].sum()),
            "Actual reference correctness",
        )
        equal(
            q["feedback_actual_correct"],
            int(s[:, 1:].sum()),
            "Actual feedback correctness",
        )
        equal(
            q["reference_rate"],
            float(Fraction(int(s[:, 0].sum()), 3 * len(ids))),
            "Reference mean",
        )
        equal(
            q["feedback_rate"],
            float(Fraction(int(s[:, 1:].sum()), 9 * len(ids))),
            "Feedback mean",
        )
        equal(
            q["actual_correct_by_condition_repeat"], s.sum(axis=0), "Actual run counts"
        )
        for perm, saved in zip(
            permutations(range(3)), q["repeat_alignments"]["counts"]
        ):
            cv = Counter()
            for c in range(1, 4):
                for t in range(3):
                    old = s[:, 0, t]
                    new = s[:, c, perm[t]]
                    for j, (u, v) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
                        cv[j] += int(((old == u) & (new == v)).sum())
            equal(saved, [cv[j] for j in range(4)], "Global repeat alignment")
        cross = Counter()
        for z in points:
            rc = z["reference_correct"]
            for tc in z["pool_correct"]:
                cross.update(
                    {
                        0: (3 - rc) * (3 - tc),
                        1: (3 - rc) * tc,
                        2: rc * (3 - tc),
                        3: rc * tc,
                    }
                )
        equal(
            q["repeat_alignments"]["all_cross_repeat_counts"],
            [cross[j] for j in range(4)],
            "All-cross repeat products",
        )
        bj = Counter(
            (int(expected[i, c, t, 0]), int(expected[i, c, t, 1]))
            for i in ids
            for c in range(3)
            for t in range(3)
        )
        equal(
            q["binary_vs_known_exact_transition_matrix"],
            [[bj[b, j] for j in range(4)] for b in range(4)],
            "Binary/joint transition matrix",
        )
        equal(
            q["head_fix_with_binary_fix"],
            int(((expected[ids, ..., h] == 1) & (expected[ids, ..., 0] == 1)).sum()),
            "Joint fix relation",
        )
        equal(
            q["binary_fix_without_head_exact"],
            int(((expected[ids, ..., 0] == 1) & (S[ids, 1:, :, h] == 0)).sum()),
            "Binary fix/fine failure",
        )
        prior = rates[q["part2_id"]] if "part2_id" in q else quality[q["part4_id"]]
        a.need(
            ids == prior["common_indices"]
            and len(ids) == q["texts"]
            and q["clusters"] == len({meta[i]["inference_cluster"] for i in ids}),
            "Inherited mask identity",
        )
        if "part2_id" in q:
            equal(q["error_difference"], prior["difference"], "Part 2 effect")
            equal(
                q["reused_interval98_75"],
                prior["interval98_75"],
                "Frozen primary interval",
            )
            equal(q["reused_interval95"], prior["interval95"], "Supporting interval")
        else:
            z = prior["metrics"][q["part4_metric"]]
            equal(q["exact_difference"], z["difference"], "Part 4 exact effect")
            equal(
                q["reused_interval95"], z.get("interval95"), "Existing interval or null"
            )

    repeat = a.read(OUT / "repeat_variation.json")
    for name, q in repeat.items():
        pop, view, gold, condition = name.split("/")
        ids = quality[f"population/{pop}/binary/{view}"]["common_indices"]
        c = p.CONDITIONS.index(condition)
        ids = [
            i
            for i in ids
            if gold == "gold_all" or int(meta[i]["gold_hate"]) == int(gold[-1])
        ]
        hist = np.zeros((3, 6), dtype=int)
        diff = np.zeros(6, dtype=int)
        for i in ids:
            vv = signatures[i][c]
            for j in range(6):
                values = [v[j] if j < 5 else v for v in vv]
                hist[len(set(values)) - 1, j] += 1
                diff[j] += sum(u != v for u, v in combinations(values, 2))
        equal(
            q["aggregate_repeat_distinct_histogram"], hist, "Aggregate repeat histogram"
        )
        equal(q["aggregate_repeat_disagree_pairs"], diff, "Aggregate repeat pairs")
        nv = x["n_valid"][ids, c]
        three = nv == 3
        vd = x["vote_distinct"][ids, c]
        equal(
            q["vote_count_histogram"],
            [int((nv == z).sum()) for z in range(4)],
            "Vote availability histogram",
        )
        equal(
            q["three_vote_distinct_histogram"],
            [
                [int(((vd[..., j] == z) & three).sum()) for j in range(6)]
                for z in (1, 2, 3)
            ],
            "Three-vote stability",
        )
        equal(
            q["valid_vote_disagree_pairs"],
            x["vote_disagree_pairs"][ids, c].sum(axis=(0, 1)),
            "Vote pair totals",
        )
        equal(
            q["valid_vote_available_pairs"],
            int(sum(int(v) * (int(v) - 1) // 2 for v in nv.ravel())),
            "Valid vote pair denominator",
        )
        equal(
            q["aggregate_not_observed"],
            int(x["aggregate_not_observed"][ids, c].sum()),
            "Unseen composite outputs",
        )
        equal(
            q["aggregate_not_observed_three_votes"],
            int((x["aggregate_not_observed"][ids, c] & three).sum()),
            "Three-vote composite outputs",
        )
        equal(q["gate_ties"], int(x["gate_ties"][ids, c].sum()), "Actual gate ties")
        equal(q["texts"], len(ids), "Repeat denominator")
        equal(q["three_vote_outputs"], int(three.sum()), "Three-vote denominator")
    shared = a.read(OUT / "same_message_variation.json")
    for name, q in shared.items():
        pop, view, kind = name.split("/")
        ids = quality[f"population/{pop}/binary/{view}"]["common_indices"]
        ct = np.zeros(6, dtype=int)
        cross = np.zeros(6, dtype=int)
        texts = []
        for c, d in combinations(range(1, 4), 2):
            take = [
                i
                for i in ids
                if (x["messages"][i, c] == x["messages"][i, d])
                == (kind == "same_messages")
            ]
            z = np.zeros(6, dtype=int)
            zz = np.zeros(6, dtype=int)
            for i in take:
                for rr in range(3):
                    for tt in range(3):
                        left, right = signatures[i][c][rr], signatures[i][d][tt]
                        v = np.array(
                            [int(u != v) for u, v in zip(left, right)]
                            + [int(left != right)]
                        )
                        zz += v
                        if rr == tt:
                            z += v
            equal(
                q["per_pool_pair"][f"{c}-{d}"],
                dict(
                    texts=len(take),
                    aligned_disagreements=z.tolist(),
                    all_cross_disagreements=zz.tolist(),
                ),
                "Identical-message pool pair",
            )
            texts.extend(take)
            ct += z
            cross += zz
        equal(q["aligned_disagreements"], ct, "Same-message aligned")
        equal(q["all_cross_disagreements"], cross, "Same-message all-cross")
        equal(q["query_pool_pairs"], len(texts), "Shared-prompt pairs")
        equal(q["unique_texts"], len(set(texts)), "Shared-prompt distinct inputs")
    edits = a.read(OUT / "fine_label_edits.json")
    for name, parts in edits.items():
        _, pop, head, view = name.split("/")
        ids = quality[name]["common_indices"]
        field = FIELDS.index(head)
        for part, record in parts.items():
            counters = {label: Counter() for label in record["labels"]}
            pairs = 0
            for i in ids:
                gold = meta[i][GOLD[head]]
                g = {gold} if head == "severity" else set(gold)
                for c, t in np.ndindex(3, 3):
                    old, new = signatures[i][0][t], signatures[i][c + 1][t]
                    if part == "both_positive" and not (old[0] and new[0]):
                        continue
                    pairs += 1
                    left = (
                        ({old[field]} if old[field] is not None else set())
                        if head == "severity"
                        else set(old[field])
                    )
                    right = (
                        ({new[field]} if new[field] is not None else set())
                        if head == "severity"
                        else set(new[field])
                    )
                    for label in right - left:
                        counters[label][
                            "correct_added" if label in g else "extra_added"
                        ] += 1
                    for label in left - right:
                        counters[label][
                            "correct_removed" if label in g else "extra_removed"
                        ] += 1
            equal(record["pairs"], pairs, "Fine edit denominator")
            equal(
                record["labels"],
                {
                    label: {
                        key: ct[key]
                        for key in (
                            "correct_added",
                            "correct_removed",
                            "extra_added",
                            "extra_removed",
                        )
                    }
                    for label, ct in counters.items()
                },
                "Set-difference label changes",
            )
    missing = a.read(OUT / "missing_outputs.json")
    for pop, q in missing.items():
        ids = [i for i, m in enumerate(meta) if m["population"] == pop]
        ct = Counter()
        for i in ids:
            for c, t in np.ndindex(3, 3):
                old, new = signatures[i][0][t], signatures[i][c + 1][t]
                ct[
                    (
                        "both_missing"
                        if old is None and new is None
                        else (
                            "reference_missing"
                            if old is None
                            else "feedback_missing" if new is None else "valid"
                        )
                    )
                ] += 1
        equal(q["pair_status_counts"], dict(ct), "Full-roster pair availability")
        equal(
            q["vote_count_histogram_by_condition"],
            [
                [int((x["n_valid"][ids, c] == n).sum()) for n in range(4)]
                for c in range(4)
            ],
            "Full-roster vote counts",
        )
    p.save(
        "verified.json",
        dict(
            status="part6_all_transitions_and_repeat_variation_independently_verified_cases_pending",
            checked_utc=a.utc(),
            assembled_sha256=a.sha(OUT / "assembled.json"),
            calculated_sha256=a.sha(OUT / "calculated.json"),
            verifier_sha256=a.sha(Path(__file__)),
            correctness_cells=int(S.size),
            actual_outputs=len(seen),
            valid_votes=checkedvotes,
            dependent_pair_rows=n,
            text_profiles=N * len(HEADS),
            transition_views=len(summary),
            repeat_views=len(repeat),
            shared_message_views=len(shared),
            fine_edit_views=len(edits),
            all_prior_net_effects_reconciled=True,
            all_canonical_votes_and_outputs_checked=True,
            new_bootstrap_sets=0,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        "All transition counts, text profiles, repeat/vote variation, shared messages and fine-label changes independently reconcile."
    )


if __name__ == "__main__":
    with a.offline():
        main()

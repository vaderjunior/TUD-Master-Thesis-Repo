from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
import numpy as np
from analysis.common import replay as c

BASE = c.BASE / "votes"


def positive(value, dim):
    if dim == "hate":
        return {"hate" if value else "not_hate"}
    if dim == "severity":
        return {value} if value is not None else set()
    return set(value or [])


def main():
    plan = c.read(BASE / "plan.json")
    c.verify(plan["input_hashes"])
    report = c.read(BASE / "results.json")
    assert c.read(BASE / "complete.json")["results_sha256"] == c.sha(
        BASE / "results.json"
    )
    assert c.read(BASE / "extraction.json")["records_sha256"] == c.sha(
        BASE / "records.jsonl"
    )
    groups = defaultdict(dict)
    for row in c.rows(BASE / "records.jsonl"):
        groups[row["subset"], row["condition"], row["repeat"]][row["item_id"]] = row
    checks = 0
    draw_checks = 0
    paired_reliability = {}
    run_lines = []
    for subset, ep in plan["endpoints"].items():
        gold = {str(r["id"]): r for r in c.gold_rows(subset)}
        ids = ep["scorable_ids"]
        labs = ep["all_labels"]
        idx = [labs.index(x) for x in ep["macro_labels"]]
        dim = ep["dimension"]
        for cond in plan["conditions"]:
            original = report["endpoints"][subset]["conditions"][cond]
            counts = []
            valids = []
            for mode in ["voted", "slot0"]:
                for rep in range(3):
                    totals = np.zeros((len(ids), len(labs), 3), dtype=np.int64)
                    valid = []
                    for n, rid in enumerate(ids):
                        row = groups[subset, cond, rep][rid]
                        p = row["voted"] if mode == "voted" else row["slots"][0]
                        y = positive(gold[rid][c.original.GOLD_FIELDS[dim]], dim)
                        valid.append(p is not None)
                        if p is None:
                            continue
                        x = positive(p[dim], dim)
                        for k, lab in enumerate(labs):
                            totals[n, k] = [
                                int(lab in y and lab in x),
                                int(lab in x and lab not in y),
                                int(lab in y and lab not in x),
                            ]
                    summed = totals.sum(axis=0)
                    den = 2 * summed[:, 0] + summed[:, 1] + summed[:, 2]
                    f = np.array(
                        [
                            2 * t[0] / (2 * t[0] + t[1] + t[2]) if sum(t) else 0.0
                            for t in summed
                        ]
                    )
                    old = original["modes"][mode]["runs"][rep]
                    assert (
                        old["per_label_counts"] == summed.tolist()
                        and abs(old["macro_f1_valid_only"] - float(f[idx].mean()))
                        < 1e-12
                    )
                    counts.append(totals)
                    valids.append(valid)
                    checks += 1
            common = np.logical_and.reduce(valids)
            raw = np.stack(counts)[:, common]
            kept = [rid for rid, ok in zip(ids, common) if ok]
            for cluster_mode in ("text", "item"):
                names = (
                    [
                        groups[subset, cond, 0][rid]["normalized_text_cluster"]
                        for rid in kept
                    ]
                    if cluster_mode == "text"
                    else [ids.index(rid) for rid in kept]
                )
                unique = sorted(set(names))
                order = {x: i for i, x in enumerate(unique)}
                totals = np.zeros((6, len(unique), len(labs), 3), dtype=np.int64)
                for i, name in enumerate(names):
                    totals[:, order[name]] += raw[:, i]
                rng = np.random.default_rng(20261006)
                draws = []
                for _ in range(10000):
                    chosen = rng.integers(0, len(unique), len(unique))
                    a = totals[:, chosen].sum(axis=1)
                    denominator = 2 * a[..., 0] + a[..., 1] + a[..., 2]
                    scores = np.divide(
                        2 * a[..., 0],
                        denominator,
                        out=np.zeros_like(denominator, dtype=float),
                        where=denominator != 0,
                    )[:, idx].mean(axis=-1)
                    draws.append(float(scores[:3].mean() - scores[3:].mean()))
                wanted = original[
                    (
                        "voted_minus_slot0"
                        if cluster_mode == "text"
                        else "item_bootstrap_sensitivity"
                    )
                ]
                assert np.allclose(
                    np.quantile(draws, [0.025, 0.975]),
                    wanted["ci95"],
                    rtol=0,
                    atol=1e-12,
                )
                if wanted["family_interval"] is not None:
                    assert np.allclose(
                        np.quantile(draws, [0.0125, 0.9875]),
                        wanted["family_interval"],
                        rtol=0,
                        atol=1e-12,
                    )
                draw_checks += 10000
            acc = Counter()
            changes = Counter()
            for rid in ids:
                for a, b in combinations(range(3), 2):
                    ra = groups[subset, cond, a][rid]
                    rb = groups[subset, cond, b][rid]
                    one, two, va, vb = (
                        ra["slots"][0],
                        rb["slots"][0],
                        ra["voted"],
                        rb["voted"],
                    )
                    acc["all_pairs"] += 1
                    if any((x is None for x in (one, two, va, vb))):
                        continue
                    acc["paired_valid_pairs"] += 1
                    single = positive(one[dim], dim) == positive(two[dim], dim)
                    voted = positive(va[dim], dim) == positive(vb[dim], dim)
                    acc["single_equal"] += single
                    acc["voted_equal"] += voted
                    changes["changed_disagreement_to_agreement"] += voted and (
                        not single
                    )
                    changes["changed_agreement_to_disagreement"] += single and (
                        not voted
                    )
            n = acc["paired_valid_pairs"]
            paired_reliability[f"{subset}/{cond}"] = {
                **dict(acc),
                **dict(changes),
                "single_agreement": acc["single_equal"] / n,
                "voted_agreement": acc["voted_equal"] / n,
                "status": "Post-score descriptive matched-validity sensitivity; no new primary or policy selection.",
            }
            for rep in range(3):
                a = original["modes"]["slot0"]["runs"][rep]
                b = original["modes"]["voted"]["runs"][rep]
                run_lines.append(
                    f"| {subset} | {cond} | {rep} | {a['macro_f1_valid_only']:.6f} | {b['macro_f1_valid_only']:.6f} | {a['n_invalid']} / {b['n_invalid']} |"
                )
            print("Independently verified " + subset + "/" + cond, flush=True)
    assert checks == 84 and draw_checks == 280000
    c.save(
        BASE / "verification.json",
        {
            "created_utc": c.utc(),
            "status": "passed",
            "results_sha256": c.sha(BASE / "results.json"),
            "source_sha256": c.sha(Path(__file__)),
            "independent_run_count_checks": checks,
            "independent_direct_bootstrap_draws": draw_checks,
            "matched_validity_repeat_agreement": paired_reliability,
            "actual_API_calls": 0,
        },
    )
    lines = [
        "# Every run: single answer versus three-answer vote",
        "",
        "These are the three outer decoder repeats; each voted result has three inner slots. Slot 0 preserves its original repairs and any null.",
        "",
        "| Endpoint | KB | Repeat | Single macro-F1 | Voted macro-F1 | Single / voted nulls |",
        "| --- | --- | ---: | ---: | ---: | --- |",
        *run_lines,
        "",
        "## Repeat agreement on the same available pairs",
        "",
        "Descriptive matched-validity sensitivity: each comparison includes exactly the same repeat pairs for both modes. This supplements the predeclared own-valid pair rates; no new primary or selected system follows.",
        "",
        "| Endpoint / KB | Available pairs | Single agreement | Voted agreement |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, x in paired_reliability.items():
        lines.append(
            f"| {name} | {x['paired_valid_pairs']} | {x['single_agreement']:.3%} | {x['voted_agreement']:.3%} |"
        )
    lines += [
        "",
        "Verification: 84 independent set-based run scores/confusion matrices and all 280,000 direct cluster/item bootstrap draws reproduce. [Verification receipt](../../experiments/astra/offline_closure_20261006_v1/votes/verification.json).",
    ]
    path = (
        c.ROOT
        / "Thesis_Log/astra_result/astra_single_three_vote_run_tables_20261006_v1.md"
    )
    with path.open("x", encoding="utf8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("All independent checks passed.", flush=True)


if __name__ == "__main__":
    with c.network_guard():
        main()

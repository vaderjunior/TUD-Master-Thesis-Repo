from collections import Counter, defaultdict
import socket
import statistics
from unittest.mock import patch
import numpy as np
from analysis.feedback.supplements.archive import ROOT
from analysis.feedback.supplements.archive import DEST
from analysis.feedback.supplements.archive import REVIEW
from analysis.feedback.supplements.archive import STUDIES
from analysis.feedback.supplements.archive import LABELS
from analysis.feedback.supplements.archive import SEEDS
from analysis.feedback.supplements.archive import read
from analysis.feedback.supplements.archive import rows
from analysis.feedback.supplements.archive import sha
from analysis.feedback.supplements.archive import rel
from analysis.feedback.supplements.archive import save
from analysis.feedback.supplements.archive import verify
from analysis.feedback.supplements.archive import require
from analysis.feedback.supplements.archive import utc
from analysis.feedback.supplements.archive import forbid


def eq(a, b):
    require(
        np.allclose(a, b, rtol=0, atol=2e-12), f"Diagnostic discrepancy: {a} vs {b}"
    )


def divide(a, b):
    return np.divide(a, b, out=np.zeros(np.shape(b), dtype=float), where=b != 0)


def sampling(ids, roster, seed):
    clusters = sorted({roster[i]["cluster"] for i in ids})
    index = {c: n for n, c in enumerate(clusters)}
    picks = np.random.Generator(np.random.PCG64(seed)).integers(
        0, len(clusters), (10000, len(clusters))
    )
    return index, np.array(
        [np.bincount(p, minlength=len(clusters)) for p in picks], dtype=float
    )


def main():
    dest = DEST / "diagnostics"
    receipt = read(dest / "complete.json")
    verify(receipt["inputs"])
    verify(receipt["outputs"])
    records = {
        (r["study"], r["condition"], r["repeat"], r["key"]): r
        for r in rows(REVIEW / "predictions.jsonl")
    }
    rosters = {
        s: {r["key"]: r for r in rows(p / "roster.jsonl")} for s, p in STUDIES.items()
    }
    availability = {
        (r["key"], r["pool_seed"]): r
        for r in rows(STUDIES["human_meaning"] / "availability.jsonl")
    }
    checks = Counter()

    def curve(
        study, ids, condition, repeat, index, multiplicities, gate=False, target=None
    ):
        count = np.zeros((len(index), 1 if gate else 7, 2), dtype=int)
        for i in ids:
            r = rosters[study][i]
            v = records[study, condition, repeat, i]["result"]
            require(v is not None, "Missing record in common mask")
            loc = index[r["cluster"]]
            if gate:
                count[loc, 0] += [v["hate"] != r["gold"]["hate"], 1]
            else:
                gold = set(r["gold"]["hate_type"] or [])
                pred = set(v["hate_type"])
                for n, label in enumerate(LABELS):
                    if target is not None and label != target:
                        continue
                    count[loc, n] += [
                        label in gold and label in pred,
                        (label in gold) + (label in pred),
                    ]
        denominator = count.sum(axis=0)[:, 1]
        point = divide((1 if gate else 2) * count.sum(axis=0)[:, 0], denominator)
        summed = (multiplicities @ count.reshape(len(index), -1)).reshape(
            10000, count.shape[1], 2
        )
        bs = divide((1 if gate else 2) * summed[:, :, 0], summed[:, :, 1])
        checks["run_count_arrays"] += 1
        return point, bs

    rules = {
        "available": lambda v: v["human_eligible"],
        "old_assistant_and_human": lambda v: v["human_eligible"]
        and v["old_assistant_available"],
        "flagged": lambda v: v["human_eligible"] and v["source_flagged"],
        "fallback": lambda v: not v["human_eligible"],
    }
    for name, rule in rules.items():
        for hateful, pop in [(True, "hateful"), (False, "benign")]:
            value = read(dest / ("coverage_" + name + "_" + pop + ".json"))
            roster = rosters["human_meaning"]
            chosen = {}
            for seed in SEEDS:
                eligible = sorted(
                    i
                    for i, r in roster.items()
                    if r["gold"]["hate"] == hateful and rule(availability[i, seed])
                )
                common = [
                    i
                    for i in eligible
                    if all(
                        records["human_meaning", f"{c}_s{seed}", rep, i]["result"]
                        is not None
                        for c in ["plain", "human", "generic"]
                        for rep in range(3)
                    )
                ]
                v = value["by_pool"][str(seed)]
                require(
                    v["original_ids"] == eligible and v["common_ids"] == common,
                    "Coverage mask differs",
                )
                require(
                    v["excluded_ids"] == sorted(set(eligible) - set(common)),
                    "Coverage exclusions differ",
                )
                chosen[seed] = common
                checks["masks"] += 1
            union = sorted({i for v in chosen.values() for i in v})
            index, w = sampling(union, roster, value["bootstrap"]["seed"])
            points = defaultdict(list)
            curves = defaultdict(list)
            for seed, ids in chosen.items():
                for policy in ["plain", "human", "generic"]:
                    for rep in range(3):
                        p, b = curve(
                            "human_meaning",
                            ids,
                            f"{policy}_s{seed}",
                            rep,
                            index,
                            w,
                            gate=not hateful,
                        )
                        points[policy].append(float(p.mean()))
                        curves[policy].append(b.mean(axis=1))
                        observed = value["settings"][f"{policy}_s{seed}_rep{rep}"]
                        eq(
                            observed["macro_f1" if hateful else "false_positive_rate"],
                            p.mean(),
                        )
                        if hateful:
                            eq(observed["per_label_f1"], p)
            means = {p: statistics.mean(v) for p, v in points.items()}
            boot = {p: np.mean(v, axis=0) for p, v in curves.items()}
            for p, values in points.items():
                eq(value["groups"][p]["mean"], means[p])
                eq(
                    value["groups"][p]["pool_means"],
                    np.array(values).reshape(3, 3).mean(axis=1),
                )
            for contrast, observed in value["contrasts"].items():
                left, right = contrast.split("_minus_")
                eq(observed["difference"], means[left] - means[right])
                eq(
                    observed["ci95"],
                    np.quantile(boot[left] - boot[right], [0.025, 0.975]),
                )
                checks["intervals"] += 1
            print("Verified coverage: " + name + "/" + pop, flush=True)

    for hate, pop in [(True, "hateful"), (False, "benign")]:
        value = read(dest / ("human_gate_all_three_valid_" + pop + ".json"))
        roster = rosters["human_meaning"]
        groups = {
            "reference": ["reference"],
            **{p: [f"{p}_s{s}" for s in SEEDS] for p in ["plain", "human", "generic"]},
        }
        ids = sorted(i for i, r in roster.items() if r["gold"]["hate"] == hate)
        common = [
            i
            for i in ids
            if all(
                records["human_meaning", c, r, i]["result"] is not None
                and records["human_meaning", c, r, i]["n_valid"] == 3
                for cs in groups.values()
                for c in cs
                for r in range(3)
            )
        ]
        require(common == value["common_ids"], "Strict gate mask differs")
        checks["masks"] += 1
        index, w = sampling(common, roster, value["bootstrap"]["seed"])
        means = {}
        boot = {}
        for group, cs in groups.items():
            values = [
                curve("human_meaning", common, c, r, index, w, gate=True)
                for c in cs
                for r in range(3)
            ]
            means[group] = np.mean([p[0] for p, b in values])
            boot[group] = np.mean([b[:, 0] for p, b in values], axis=0)
            eq(value["groups"][group]["error_rate"], means[group])
            eq(value["groups"][group]["ci95"], np.quantile(boot[group], [0.025, 0.975]))
            checks["intervals"] += 1
        for contrast, observed in value["contrasts"].items():
            left, right = contrast.split("_minus_")
            eq(observed["difference"], means[left] - means[right])
            eq(observed["ci95"], np.quantile(boot[left] - boot[right], [0.025, 0.975]))
            checks["intervals"] += 1

    value = read(dest / "factorial_categories.json")
    primary = read(DEST / "factorial/primary.json")
    ids = primary["common_ids"]
    roster = rosters["factorial"]
    index, w = sampling(ids, roster, 202610041)
    terms = {
        "marginal_example_block_effect": {
            "E1D0": 0.5,
            "E0D0": -0.5,
            "E1D1": 0.5,
            "E0D1": -0.5,
        },
        "marginal_definition_block_effect": {
            "E0D1": 0.5,
            "E0D0": -0.5,
            "E1D1": 0.5,
            "E1D0": -0.5,
        },
        "joint_restoration": {"E1D1": 1, "E0D0": -1},
        "interaction": {"E1D1": 1, "E1D0": -1, "E0D1": -1, "E0D0": 1},
        "examples_when_definitions_withdrawn": {"E1D0": 1, "E0D0": -1},
        "examples_when_definitions_full": {"E1D1": 1, "E0D1": -1},
        "definitions_when_examples_withdrawn": {"E0D1": 1, "E0D0": -1},
        "definitions_when_examples_full": {"E1D1": 1, "E1D0": -1},
    }
    other_point = defaultdict(list)
    other_boot = defaultdict(list)
    for n, label in enumerate(LABELS):
        chosen = [i for i in ids if roster[i]["target_label"] == label]
        v = value["per_category"][label]
        eq(v["pairs"], len(chosen))
        eq(
            v["positive_support"],
            sum(label in roster[i]["gold"]["hate_type"] for i in chosen),
        )
        means = {}
        boot = {}
        for c in ["E0D0", "E1D0", "E0D1", "E1D1"]:
            scored = [curve("factorial", chosen, c, r, index, w) for r in range(3)]
            means[c] = np.mean([p[n] for p, b in scored])
            boot[c] = np.mean([b[:, n] for p, b in scored], axis=0)
            eq(v["cell_f1"][c], means[c])
            other_point[c].extend(np.delete(p, n).mean() for p, b in scored)
            other_boot[c].extend(
                np.delete(b, n, axis=1).mean(axis=1) for p, b in scored
            )
        for effect, t in terms.items():
            eq(
                v["effects"][effect]["difference"],
                sum(weight * means[c] for c, weight in t.items()),
            )
            eq(
                v["effects"][effect]["ci95"],
                np.quantile(
                    sum(weight * boot[c] for c, weight in t.items()), [0.025, 0.975]
                ),
            )
            checks["intervals"] += 1
    for effect, t in terms.items():
        observed = value["other_six_label_effects"][effect]
        eq(
            observed["difference"],
            sum(weight * np.mean(other_point[c]) for c, weight in t.items()),
        )
        eq(
            observed["ci95"],
            np.quantile(
                sum(weight * np.mean(other_boot[c], axis=0) for c, weight in t.items()),
                [0.025, 0.975],
            ),
        )
        checks["intervals"] += 1
    for contrast, observed in value["paired_flips"].items():
        left, right = contrast.split("_minus_")
        total = Counter()
        for label in LABELS:
            count = Counter()
            for i in ids:
                if roster[i]["target_label"] != label:
                    continue
                for rep in range(3):
                    a = records["factorial", left, rep, i]["result"]
                    b = records["factorial", right, rep, i]["result"]
                    new = label in a["hate_type"]
                    old = label in b["hate_type"]
                    if new == old:
                        continue
                    event = (
                        "tp" if label in roster[i]["gold"]["hate_type"] else "fp"
                    ) + ("_gained" if new else "_lost")
                    count[event] += 1
                    count[
                        event
                        + (
                            "_both_gates_hate"
                            if a["hate"] and b["hate"]
                            else "_gate_changed"
                        )
                    ] += 1
            require(
                dict(count) == observed["by_label"][label], "Per-label flips differ"
            )
            total.update(count)
        require(
            dict(total) == observed["summed_over_three_repeats"], "Total flips differ"
        )
        checks["flip_tables"] += 1
    prompts = Counter(
        p["messages_sha256"]
        for p in rows(STUDIES["factorial"] / "prompts.jsonl")
        if p["condition"] == "E1D1"
    )
    eq(value["full_cell_distinct_prompts"], len(prompts))
    eq(value["full_cell_prompt_rows"], sum(prompts.values()))
    require(
        {str(k): v for k, v in Counter(prompts.values()).items()}
        == value["full_cell_reused_message_multiplicities"],
        "Identical-message multiplicity differs",
    )
    save(
        dest / "independent_verification.json",
        {
            "status": "passed",
            "created_utc": utc(),
            "checks": dict(checks),
            "source_sha256": sha(__file__),
            "diagnostics_receipt_sha256": sha(dest / "complete.json"),
            "method": "Separate set/count implementation; all coverage masks and 10000-draw intervals, strict gate sensitivities, category/other-label effects and paired flips independently rebuilt.",
            "new_classifier_calls": 0,
        },
    )
    print("Supporting diagnostics independently verified.", flush=True)


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits

    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket, "create_connection", forbid
    ), threadpool_limits(limits=2):
        main()

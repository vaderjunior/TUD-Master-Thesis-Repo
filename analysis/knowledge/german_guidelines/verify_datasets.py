"""German guidelines verify datasets."""

from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from analysis.knowledge.german_guidelines import datasets as d
from analysis.knowledge.german_guidelines.verify_context import same
from analysis.knowledge.german_guidelines.verify_hate import rates


def ref_event(a, b):
    if a == -1 and b == -1:
        return "both_missing"
    if a == -1:
        return "left_missing"
    if b == -1:
        return "right_missing"
    return {(0, 0): "both_wrong", (0, 1): "regression", (1, 0): "fix", (1, 1): "both_correct"}[
        int(a), int(b)
    ]


def ref_pair(left, right):
    n = left.shape[1]
    counts = np.zeros((3, 3, 7), dtype=int)
    matrix = np.zeros((4, 4), dtype=int)
    stable = Counter()
    relative = Counter()
    reversal = 0
    for i in range(n):
        a, b = (left[:, i].tolist(), right[:, i].tolist())
        k, j = (a.count(1), b.count(1))
        for r, s in product(range(3), repeat=2):
            counts[r, s, d.EVENTS.index(ref_event(a[r], b[s]))] += 1
        events = [ref_event(a[r], b[r]) for r in range(3)]
        reversal += int("fix" in events and "regression" in events)
        if -1 in a + b:
            stable["unavailable"] += 1
            relative["unavailable"] += 1
        else:
            matrix[j, k] += 1
            relative["better" if k > j else "worse" if k < j else "same"] += 1
            stable[
                {
                    (3, 0): "stable_fix",
                    (0, 3): "stable_regression",
                    (3, 3): "both_always_correct",
                    (0, 0): "both_always_wrong",
                }.get((k, j), "variable")
            ] += 1
    return {
        "n": n,
        "stability_counts": {name: stable[name] for name in d.STABLE},
        "correct_repeat_matrix_rows_reference_columns_alternative": matrix,
        "ordinal_direction_reversal": reversal,
        "event_order": d.EVENTS,
        "all_nine_pair_counts": counts,
        "mean_all_nine_events": {name: counts[:, :, i].mean() for i, name in enumerate(d.EVENTS)},
        "same_repeat_events": [
            {name: int(counts[r, r, i]) for i, name in enumerate(d.EVENTS)} for r in range(3)
        ],
        "relative_repeat_counts": {
            name: relative[name] for name in ["better", "worse", "same", "unavailable"]
        },
    }


def ref_path(states):
    rows = states.tolist()
    if any((-1 in row for row in rows)):
        return "unavailable"
    if any((len(set(row)) > 1 for row in rows)):
        return "variable"
    return d.PATHS["".join((str(row[0]) for row in rows))]


def ref_stability(y, p, states):
    n = len(y)
    conditions = {}
    legal = y.ndim == 2
    for ci, cond in enumerate(d.CONDITIONS):
        hist = Counter()
        sameanswer = 0
        variable = 0
        wrongvariable = 0
        missing = 0
        for i in range(n):
            s = states[ci, :, i].tolist()
            if -1 in s:
                missing += 1
                continue
            hist[sum(s)] += 1
            answers = [tuple(p[ci, r, i]) if legal else int(p[ci, r, i]) for r in range(3)]
            if len(set(answers)) == 1:
                sameanswer += 1
            else:
                variable += 1
                wrongvariable += int(sum(s) == 0)
        conditions[cond] = {
            "full_inputs": n,
            "all_three_available": n - missing,
            "incomplete_inputs": missing,
            "missing_by_repeat": [(states[ci, r] < 0).sum() for r in range(3)],
            "correct_repeat_histogram": {str(i): hist[i] for i in range(4)},
            "consistent_answer": sameanswer,
            "variable_answer": variable,
            "always_wrong_but_variable_answer": wrongvariable,
        }
    return {
        "n": n,
        "conditions": conditions,
        "pairs": {
            a
            + "_minus_"
            + b: ref_pair(states[d.CONDITIONS.index(a)], states[d.CONDITIONS.index(b)])
            for a, b in d.PAIRS
        },
        "joint_paths": dict(Counter((ref_path(states[:, :, i]) for i in range(n)))),
        "joint_correct_repeat_counts": dict(
            Counter(
                (
                    ",".join((str(int(v)) for v in states[:, :, i].sum(axis=1)))
                    for i in range(n)
                    if (states[:, :, i] >= 0).all()
                )
            )
        ),
        "order": d.CONDITIONS,
        "meaning": "Repeat-count and stable-path descriptions, not an outer-repeat ensemble or chronological learning. Ordinal and nine-pair comparisons are not independent observations.",
    }


def boot_metrics(z):
    tn, fp, fn, tp = [z[..., i].astype(float) for i in range(4)]

    def ratio(a, b, zero=np.nan):
        return np.divide(a, b, out=np.full_like(a, zero), where=b != 0)

    return {
        "macro_f1": (ratio(2 * tp, 2 * tp + fp + fn, 0) + ratio(2 * tn, 2 * tn + fp + fn, 0)) / 2,
        "fp": fp,
        "fn": fn,
        "fpr": ratio(fp, fp + tn),
        "fnr": ratio(fn, fn + tp),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "accuracy": ratio(tp + tn, z.sum(axis=-1)),
        "predicted_hate": ratio(tp + fp, z.sum(axis=-1)),
    }


def check_interval(actual, values):
    bad = int((~np.isfinite(values)).sum())
    same(actual["undefined_draws"], bad)
    same(actual["ci95"], None if bad else np.quantile(values, [0.025, 0.975]))
    same(actual["family_interval"], None)
    same(actual["family_size"], None)


def verify(path):
    with d.offline():
        c = d.c
        manifest = c.read(path)
        c.verify(manifest["outputs"])
        c.verify(manifest["sources"])
        contract = c.read(d.BASE / "analysis_contract.json")
        c.verify(contract["inputs"])
        out = c.ROOT / manifest["run"]
        _, hp = d.h.verified()
        _, lp = d.l.verified()
        _, rp = d.reader.read_verified()
        items = {r["key"]: r for r in c.rows(out / "individuals.jsonl")}
        checks = Counter()
        for phase in d.PHASES:
            legal = "legal" in phase
            prior = lp if legal else hp
            idx = c.read(prior / phase / "input_roster.json")
            keys = idx["keys"]
            with np.load(prior / phase / "arrays.npz", allow_pickle=False) as f:
                y, p, mask = (f["gold"], f["predictions"], f["common_mask"])
            states = np.full((3, 3, len(y)), -1, dtype=np.int8)
            for ci, r, i in product(range(3), range(3), range(len(y))):
                value = p[ci, r, i]
                if (value >= 0).all():
                    states[ci, r, i] = int(np.array_equal(value, y[i]))
            result = c.read(out / phase / "results.json")
            same(result["full_stability"], ref_stability(y, p, states))
            same(
                result["common_stability"],
                ref_stability(y[mask], p[:, :, mask], states[:, :, mask]),
            )
            for i, key in enumerate(keys):
                saved = items[key]
                same(saved["correct_states"], states[:, :, i])
                same(saved["predictions"], p[:, :, i])
                same(saved["gold"], y[i])
                same(saved["common_valid"], bool(mask[i]))
                same(saved["joint_path"], ref_path(states[:, :, i]))
                for left, right in d.PAIRS:
                    li, ri = (d.CONDITIONS.index(left), d.CONDITIONS.index(right))
                    v = saved["pairs"][left + "_minus_" + right]
                    actual = ref_pair(states[li, :, i : i + 1], states[ri, :, i : i + 1])
                    same(
                        v["stability"],
                        next((k for k, n in actual["stability_counts"].items() if n)),
                    )
                    same(
                        v["same_repeat_events"],
                        [ref_event(states[li, r, i], states[ri, r, i]) for r in range(3)],
                    )
                    same(
                        v["all_nine_events"],
                        [
                            ref_event(states[li, r, i], states[ri, s, i])
                            for r, s in product(range(3), repeat=2)
                        ],
                    )
                    same(
                        v["ordinal_direction_reversal"], bool(actual["ordinal_direction_reversal"])
                    )
                    same(v["left_correct_repeats"], int((states[li, :, i] == 1).sum()))
                    same(v["right_correct_repeats"], int((states[ri, :, i] == 1).sum()))
                    same(
                        v["correct_repeat_difference"],
                        (
                            None
                            if v["stability"] == "unavailable"
                            else int(states[li, :, i].sum() - states[ri, :, i].sum())
                        ),
                    )
                    if legal and mask[i]:
                        same(
                            saved["legal_error_differences"][left + "_minus_" + right],
                            int(
                                np.not_equal(p[li, :, i], y[i]).sum()
                                - np.not_equal(p[ri, :, i], y[i]).sum()
                            ),
                        )
                checks["individuals"] += 1
            for source, s in result["source_results"].items():
                full = np.array(idx["sources"]) == source
                select = full & mask
                yy, pp = (y[select], p[:, :, select])
                same(s["full_inputs"], full.sum())
                same(s["common_inputs"], select.sum())
                same(s["common_stability"], ref_stability(yy, pp, states[:, :, select]))
                same(s["full_stability"], ref_stability(y[full], p[:, :, full], states[:, :, full]))
                if legal:
                    same(s["common"], c.read(prior / phase / "results.json")["common"])
                    same(s["contrasts"], c.read(prior / phase / "results.json")["contrasts"])
                else:
                    for ci, cond in enumerate(d.CONDITIONS):
                        ref = [rates(yy, v) for v in pp[ci]]
                        same(s["common"][cond]["repeats"], ref)
                        for name in ref[0]:
                            vals = np.array([np.nan if r[name] is None else r[name] for r in ref])
                            same(s["common"][cond]["mean"][name], vals.mean())
                            same(s["common"][cond]["sample_sd"][name], vals.std(ddof=1))
                    design = c.read(out / phase / source / "bootstrap_design.json")
                    with np.load(out / phase / source / "bootstrap.npz", allow_pickle=False) as f:
                        boot = f["counts"]
                    clusters = [v for v, on in zip(idx["clusters"], select) if on]
                    unique = sorted(set(clusters))
                    require = d.require
                    require(unique == design["clusters"], "Bootstrap cluster order differs")
                    n = len(yy)
                    cells = pp.reshape(9, n)
                    item = np.stack(
                        [
                            (cells == 0) & (yy == 0),
                            (cells == 1) & (yy == 0),
                            (cells == 0) & (yy == 1),
                            (cells == 1) & (yy == 1),
                        ],
                        axis=-1,
                    ).astype(np.int64)
                    grouped = np.stack(
                        [item[:, np.array(clusters) == cl].sum(axis=1) for cl in unique], axis=1
                    )
                    rng = np.random.default_rng(d.SEEDS[phase + "/" + source])
                    digest = hashlib.sha256()
                    for start in range(0, d.DRAWS, 64):
                        b = min(64, d.DRAWS - start)
                        picks = np.concatenate(
                            [
                                rng.choice(
                                    s0["cluster_indices"],
                                    (b, len(s0["cluster_indices"])),
                                    replace=True,
                                )
                                for s0 in design["strata"]
                            ],
                            axis=1,
                        )
                        digest.update(picks.astype("<i4").tobytes())
                        direct = (
                            grouped[:, picks].sum(axis=2).transpose(1, 0, 2).reshape(b, 3, 3, 4)
                        )
                        require(
                            np.array_equal(direct, boot[start : start + b]),
                            "Independent source resampling totals differ",
                        )
                    require(
                        digest.hexdigest() == design["pick_stream_sha256"],
                        "Bootstrap index stream differs",
                    )
                    vals = {k: v.mean(axis=2) for k, v in boot_metrics(boot).items()}
                    for ci, cond in enumerate(d.CONDITIONS):
                        for name in d.METRICS:
                            check_interval(s["condition_intervals"][cond][name], vals[name][:, ci])
                            checks["intervals"] += 1
                    for j, (a, b) in enumerate(d.PAIRS):
                        li, ri = (d.CONDITIONS.index(a), d.CONDITIONS.index(b))
                        for name in d.METRICS:
                            q = s["contrasts"][j]["metrics"][name]
                            check_interval(q, vals[name][:, li] - vals[name][:, ri])
                            same(
                                q["difference"],
                                s["common"][a]["mean"][name] - s["common"][b]["mean"][name],
                            )
                            checks["intervals"] += 1
                    checks["independent_draws"] += d.DRAWS
                checks["source_summaries"] += 1
            if legal:
                yy, pp, ss = (y[mask], p[:, :, mask], states[:, :, mask])
                detail = result["legal_detail"]
                errors = np.not_equal(pp, yy).sum(axis=-1)
                for j, label in enumerate(d.l.LABELS):
                    same(
                        detail["by_label"][label],
                        ref_stability(
                            yy[:, j], pp[:, :, :, j], (pp[:, :, :, j] == yy[:, j]).astype(np.int8)
                        ),
                    )
                for a, b in d.PAIRS:
                    li, ri = (d.CONDITIONS.index(a), d.CONDITIONS.index(b))
                    v = detail["pairs"][a + "_minus_" + b]
                    diffs = np.stack(
                        [errors[li, r] - errors[ri, s] for r, s in product(range(3), repeat=2)]
                    )
                    same(
                        v["label_error_change_all_nine"],
                        {str(i): float((diffs == i).sum(axis=1).mean()) for i in range(-3, 4)},
                    )
                    same(v["mean_label_error_difference"], diffs.sum(axis=1).mean())
                    bw = (ss[li] == 0).all(axis=0) & (ss[ri] == 0).all(axis=0)
                    delta = errors[li].sum(axis=0) - errors[ri].sum(axis=0)
                    same(
                        v["both_always_exact_wrong"],
                        {
                            "n": bw.sum(),
                            "fewer_label_errors": (bw & (delta < 0)).sum(),
                            "more_label_errors": (bw & (delta > 0)).sum(),
                            "same_label_error_total": (bw & (delta == 0)).sum(),
                        },
                    )
                checks["legal_detail_populations"] += 1
        groups = defaultdict(list)
        expected = defaultdict(list)
        for key, row in items.items():
            if not row["common_valid"]:
                expected[key].append("full_roster_missing_output")
                continue
            test = row["phase"] in ["de_test", "de_legal_test"]
            legal = "legal" in row["phase"]
            for pair, v in row["pairs"].items():
                cat = []
                st = v["stability"]
                change = v["correct_repeat_difference"]
                if st in ["stable_fix", "stable_regression"]:
                    cat.append(st if legal else st + ("_FN" if row["gold"] else "_FP"))
                if legal and st == "both_always_wrong":
                    val = row["legal_error_differences"][pair]
                    if val:
                        cat.append(
                            "both_wrong_partial_improvement"
                            if val < 0
                            else "both_wrong_partial_worsening"
                        )
                if v["ordinal_direction_reversal"]:
                    cat.append("ordinal_direction_reversal")
                if st == "variable" and change:
                    cat.append(
                        "unstable_correct_count_improvement"
                        if change > 0
                        else "unstable_correct_count_worsening"
                    )
                if st in ["both_always_wrong", "both_always_correct"]:
                    cat.append(st)
                if not test:
                    cat = [
                        name
                        for name in cat
                        if pair == "general_2_minus_mixed_2"
                        and (
                            name.startswith("stable_fix")
                            or name.startswith("stable_regression")
                            or name == "ordinal_direction_reversal"
                        )
                    ]
                for name in cat:
                    groups["/".join([row["phase"], row["source"], pair, name])].append(key)
            if test and row["joint_path"] not in ["variable", "unavailable"]:
                groups[
                    "/".join([row["phase"], row["source"], "three_conditions", row["joint_path"]])
                ].append(key)
        expected_groups = {}
        for group, keys in sorted(groups.items()):
            key = sorted(
                keys, key=lambda k: hashlib.sha256(("part5-cases|" + k).encode()).hexdigest()
            )[0]
            expected[key].append(group)
            expected_groups[group] = {"eligible": len(keys), "selected": key}
        selection = c.read(out / "case_selection.json")
        same(selection["groups"], expected_groups)
        same(selection["selected"], dict(expected))
        cases = {r["key"]: r for r in c.rows(out / "cases.jsonl")}
        same(sorted(cases), sorted(expected))
        roster = {r["key"]: r for r in c.rows(c.BASE / "roster.jsonl")}
        for key, row in cases.items():
            same(row["individual"], items[key])
            same(row["gold_hate"], roster[key].get("gate"))
            same(row["gold_legal"], roster[key].get("legal"))
            same(row["text"], roster[key]["text"])
        for row in c.rows(c.BASE / "prompts.jsonl"):
            if row["key"] in cases:
                d.require(
                    cases[row["key"]]["prompts"][row["condition"]] == row, "Case prompt changed"
                )
                checks["case_prompts"] += 1
        for row in c.rows(rp / "predictions.jsonl"):
            if row["key"] in cases:
                value = cases[row["key"]]["predictions"][row["condition"]][str(row["repeat"])]
                d.require(
                    value == {k: row[k] for k in ["result", "evidence", "events", "n_valid"]},
                    "Case output differs",
                )
                checks["case_answers"] += 1
        c.verify(contract["inputs"])
        c.verify(manifest["outputs"])
        d.save(
            out / "independent_check.json",
            {
                "status": "part5_independent_checks_pass",
                "checks": dict(checks),
                "actual_Qwen_calls": 0,
                "D_writes": 0,
            },
        )
        d.save(
            d.BASE / "complete.json",
            {
                "status": "part5_numerically_verified",
                "manifest": c.ref(path),
                "created_utc": c.a.utc(),
                "bindings": c.bindings(
                    [
                        path,
                        out / "independent_check.json",
                        d.BASE / "analysis_contract.json",
                        *d.sources(),
                    ]
                ),
                "actual_Qwen_calls": 0,
            },
        )
        print(
            json.dumps({"status": "part5_independent_checks_pass", "checks": dict(checks)}),
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    verify(Path(parser.parse_args().manifest))

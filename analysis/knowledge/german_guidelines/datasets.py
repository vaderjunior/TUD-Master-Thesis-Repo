"""German guidelines datasets."""

from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np
from analysis.knowledge.german_guidelines import context as context

c, h, l, reader = (context.c, context.h, context.l, context.reader)
require, offline = (context.require, context.offline)
BASE = c.ROOT / "experiments/astra/german_guidelines_dataset_analysis_20261113_v1"
PHASES = context.PHASES
CONDITIONS = h.CONDITIONS
PAIRS = h.PAIRS
DRAWS = 10000
SEEDS = {
    "de_test/detox": 2026111351,
    "de_test/gahd": 2026111352,
    "de_dev_eval/detox": 2026111353,
    "de_dev_eval/gahd": 2026111354,
}
METRICS = (
    "macro_f1",
    "fp",
    "fn",
    "fpr",
    "fnr",
    "precision",
    "recall",
    "accuracy",
    "predicted_hate",
)
EVENTS = (
    "fix",
    "regression",
    "both_correct",
    "both_wrong",
    "left_missing",
    "right_missing",
    "both_missing",
)
STABLE = (
    "stable_fix",
    "stable_regression",
    "both_always_correct",
    "both_always_wrong",
    "variable",
    "unavailable",
)
PATHS = {
    "000": "always_wrong_all_settings",
    "001": "stable_legal_only_fix",
    "010": "stable_general_fix_reversed",
    "011": "stable_general_fix_retained",
    "100": "stable_general_regression_persists",
    "101": "stable_general_regression_restored",
    "110": "stable_added_legal_regression",
    "111": "always_correct_all_settings",
}


def save(path, value):
    require(c.normal(path).is_relative_to(BASE), "Part 5 write outside new folder")
    c.save(path, h.safe(value))


def jsonlines(path, values):
    require(c.normal(path).is_relative_to(BASE), "Part 5 rows outside new folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        for value in values:
            f.write(
                json.dumps(h.safe(value), sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            )


def arrays(path, **values):
    require(c.normal(path).is_relative_to(BASE), "Part 5 arrays outside new folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, **values)


def correct(y, p):
    y, p = (np.asarray(y), np.asarray(p))
    legal = y.ndim == 2
    require(
        p.shape == (3, 3) + y.shape and np.isin(p, [-1, 0, 1]).all() and np.isin(y, [0, 1]).all(),
        "Prediction axes or gold values differ",
    )
    if legal:
        valid = (p >= 0).all(axis=-1)
        require(
            np.array_equal((p < 0).any(axis=-1), (p < 0).all(axis=-1)),
            "Partially missing legal set",
        )
        good = (p == y).all(axis=-1)
    else:
        valid = p >= 0
        good = p == y
    return np.where(valid, good.astype(np.int8), -1)


def event(left, right):
    if left < 0:
        return "both_missing" if right < 0 else "left_missing"
    if right < 0:
        return "right_missing"
    return (
        "both_correct"
        if left and right
        else "fix" if left else "regression" if right else "both_wrong"
    )


def individual_pair(left, right):
    left, right = (np.asarray(left), np.asarray(right))
    require(
        left.shape == right.shape == (3,)
        and np.isin(left, [-1, 0, 1]).all()
        and np.isin(right, [-1, 0, 1]).all(),
        "Wrong repeat states",
    )
    names = [event(int(a), int(b)) for a, b in zip(left, right)]
    known = bool((left >= 0).all() and (right >= 0).all())
    a, b = (int((left == 1).sum()), int((right == 1).sum()))
    if not known:
        stable = "unavailable"
    elif a == 3 and b == 0:
        stable = "stable_fix"
    elif a == 0 and b == 3:
        stable = "stable_regression"
    elif a == b == 3:
        stable = "both_always_correct"
    elif a == b == 0:
        stable = "both_always_wrong"
    else:
        stable = "variable"
    return {
        "same_repeat_events": names,
        "stability": stable,
        "left_correct_repeats": a,
        "right_correct_repeats": b,
        "correct_repeat_difference": a - b if known else None,
        "ordinal_direction_reversal": "fix" in names and "regression" in names,
        "all_nine_events": [event(int(a), int(b)) for a in left for b in right],
    }


def joint_path(states):
    states = np.asarray(states)
    require(states.shape == (3, 3), "Wrong joint-path axes")
    if (states < 0).any():
        return "unavailable"
    counts = states.sum(axis=1)
    if not np.isin(counts, [0, 3]).all():
        return "variable"
    return PATHS["".join((str(int(v == 3)) for v in counts))]


def pair_summary(left, right):
    require(left.shape == right.shape and left.shape[0] == 3, "Wrong pair summary axes")
    items = [individual_pair(left[:, i], right[:, i]) for i in range(left.shape[1])]
    allnine = np.zeros((3, 3, len(EVENTS)), dtype=np.int64)
    for i, item in enumerate(items):
        for a, b in product(range(3), repeat=2):
            allnine[a, b, EVENTS.index(item["all_nine_events"][3 * a + b])] += 1
    hist = np.zeros((4, 4), dtype=np.int64)
    for item in items:
        if item["stability"] != "unavailable":
            hist[item["right_correct_repeats"], item["left_correct_repeats"]] += 1
    return h.safe(
        {
            "n": left.shape[1],
            "stability_counts": {
                name: sum((v["stability"] == name for v in items)) for name in STABLE
            },
            "correct_repeat_matrix_rows_reference_columns_alternative": hist,
            "ordinal_direction_reversal": sum((v["ordinal_direction_reversal"] for v in items)),
            "event_order": EVENTS,
            "all_nine_pair_counts": allnine,
            "mean_all_nine_events": {
                name: allnine[:, :, j].mean() for j, name in enumerate(EVENTS)
            },
            "same_repeat_events": [
                {name: int(allnine[r, r, j]) for j, name in enumerate(EVENTS)} for r in range(3)
            ],
            "relative_repeat_counts": {
                "better": sum(
                    (
                        v["correct_repeat_difference"] is not None
                        and v["correct_repeat_difference"] > 0
                        for v in items
                    )
                ),
                "worse": sum(
                    (
                        v["correct_repeat_difference"] is not None
                        and v["correct_repeat_difference"] < 0
                        for v in items
                    )
                ),
                "same": sum((v["correct_repeat_difference"] == 0 for v in items)),
                "unavailable": sum((v["correct_repeat_difference"] is None for v in items)),
            },
        }
    )


def stability_summary(y, p, states):
    legal = y.ndim == 2
    n = len(y)
    condition = {}
    for ci, cond in enumerate(CONDITIONS):
        known = (states[ci] >= 0).all(axis=0)
        v = states[ci][:, known]
        same_answer = np.all(p[ci] == p[ci, 0], axis=(0, 2) if legal else 0)
        condition[cond] = {
            "full_inputs": n,
            "all_three_available": int(known.sum()),
            "incomplete_inputs": int((~known).sum()),
            "missing_by_repeat": (states[ci] < 0).sum(axis=1),
            "correct_repeat_histogram": {str(i): int((v.sum(axis=0) == i).sum()) for i in range(4)},
            "consistent_answer": int((known & same_answer).sum()),
            "variable_answer": int((known & ~same_answer).sum()),
            "always_wrong_but_variable_answer": int(
                (known & (states[ci] == 0).all(axis=0) & ~same_answer).sum()
            ),
        }
    pairs = {
        left
        + "_minus_"
        + right: pair_summary(states[CONDITIONS.index(left)], states[CONDITIONS.index(right)])
        for left, right in PAIRS
    }
    paths = Counter((joint_path(states[:, :, i]) for i in range(n)))
    counts = Counter(
        (
            ",".join((str(int(v)) for v in states[:, :, i].sum(axis=1)))
            for i in range(n)
            if (states[:, :, i] >= 0).all()
        )
    )
    return {
        "n": n,
        "conditions": h.safe(condition),
        "pairs": pairs,
        "joint_paths": dict(paths),
        "joint_correct_repeat_counts": dict(counts),
        "order": CONDITIONS,
        "meaning": "Repeat-count and stable-path descriptions, not an outer-repeat ensemble or chronological learning. Ordinal and nine-pair comparisons are not independent observations.",
    }


def legal_detail(y, p, states):
    require((p >= 0).all(), "Legal partial-error summaries require common mask")
    label_states = np.moveaxis((p == y).astype(np.int8), -1, 0)
    errors = np.abs(p - y).sum(axis=-1)
    pairs = {}
    for left, right in PAIRS:
        li, ri = (CONDITIONS.index(left), CONDITIONS.index(right))
        diff = errors[li][:, None, :] - errors[ri][None, :, :]
        bothwrong = (states[li] == 0).all(axis=0) & (states[ri] == 0).all(axis=0)
        average = errors[li].sum(axis=0) - errors[ri].sum(axis=0)
        pairs[left + "_minus_" + right] = {
            "label_error_change_all_nine": {
                str(i): float((diff == i).sum(axis=2).mean()) for i in range(-3, 4)
            },
            "mean_label_error_difference": float(diff.sum(axis=2).mean()),
            "both_always_exact_wrong": {
                "n": int(bothwrong.sum()),
                "fewer_label_errors": int((bothwrong & (average < 0)).sum()),
                "more_label_errors": int((bothwrong & (average > 0)).sum()),
                "same_label_error_total": int((bothwrong & (average == 0)).sum()),
            },
        }
    return {
        "by_label": {
            label: stability_summary(y[:, j], p[:, :, :, j], label_states[j])
            for j, label in enumerate(l.LABELS)
        },
        "pairs": pairs,
    }


def sources():
    return [
        Path(__file__),
        c.ROOT / "analysis/knowledge/german_guidelines/verify_datasets.py",
        c.ROOT / "tests/knowledge/german_guidelines/test_datasets.py",
        Path(h.__file__),
        Path(l.__file__),
        Path(context.__file__),
    ]


def prepare():
    c.validate()
    _, hp = h.verified()
    _, lp = l.verified()
    _, xp = context.verified()
    _, rp = reader.read_verified()
    tests = sorted((BASE / "tests").glob("tests_*/report.json"))
    require(
        tests
        and c.read(tests[-1])["passed"]
        and (c.read(tests[-1])["sources"] == c.bindings(sources())),
        "Current tests must pass",
    )
    save(
        BASE / "analysis_contract.json",
        {
            "status": "part5_plan_before_new_source_effect_and_case_inspection",
            "created_utc": c.a.utc(),
            "inputs": c.bindings(
                [
                    c.PLAN,
                    c.PROTOCOL,
                    reader.ANALYSIS_PLAN,
                    *[b / "complete.json" for b in [reader.BASE, h.BASE, l.BASE, context.BASE]],
                    *[p / "manifest.json" for p in [hp, lp, xp, rp]],
                    tests[-1],
                    *sources(),
                ]
            ),
            "known": "Parts 2–4 aggregate/context outcomes and their context-selected cases have already been inspected. This remains an exploratory, exposed-data supplement.",
            "scope": "Only de_test, de_dev_eval, de_legal_test and de_legal_dev_eval. Hate sources DeTox and GAHD; legal BoTox. No English or HateCheck extension.",
            "scores": "Use original common-valid mask across all nine outputs, fixed endpoint labels and three separately scored repeats. Retain full-roster transitions/missing separately; no imputation.",
            "source_uncertainty": {
                "draws": DRAWS,
                "seeds": SEEDS,
                "method": "Reuse verified hate cluster bootstrap within each source with full-roster cluster memberships; common draws for all nine settings; fixed label universe. All source CIs are descriptive 95%, no new confirmatory family. Reuse prior BoTox intervals because it is the whole legal endpoint.",
            },
            "stability": "Per condition count correct repeats 0..3 and exact answer consistency. Per contrast distinguish all-three fix/regression, both always correct/wrong, variable, unavailable. Save same-ordinal and all-nine cross-repeat transitions; neither implies shared decoding seeds or chronological correction. Never vote across outer repeats.",
            "legal": "Keep legal independent of hate. Exact-set transitions, three label-specific transitions and partial label-error changes when both conditions remain wrong. Unannotated fields are not scored.",
            "paths": "All eight paths with each setting correct in all three repeats or wrong in all three; all other paths variable or unavailable. Also retain the full 4x4x4 correct-count pattern.",
            "case_selection": "Minimum SHA256(part5-cases|key) for each nonempty TEST source/contrast category: stable FP/FN fixes and regressions (hate), stable exact fixes/regressions and both-wrong partial improvement/worsening (legal), ordinal direction reversal, unstable correct-count improvement/worsening, persistent wrong, persistent correct. Also one per nonempty TEST source stable three-setting path. DEV: one per source stable fix/regression/direction reversal for general versus shared. Include every input outside the original common mask. Deduplicate keys, keep all reasons. No extra hand-picked cases or gold editing.",
            "limitations": "Case predicates describe outputs and fixed gold, not true causal mechanisms. Review selected examples for possible annotation/context ambiguity without changing gold. Do not treat model rationales as proof. Leave formal exclusions and operational robustness for Part 6.",
            "actual_Qwen_calls": 0,
        },
    )
    print("Part 5 analysis and case-selection contract recorded.", flush=True)


def case_categories(item, pair, legal):
    v = item["pairs"][pair]
    s = v["stability"]
    d = v["correct_repeat_difference"]
    categories = []
    if legal:
        if s in ["stable_fix", "stable_regression"]:
            categories.append(s)
        if s == "both_always_wrong":
            delta = item["legal_error_differences"][pair]
            if delta < 0:
                categories.append("both_wrong_partial_improvement")
            if delta > 0:
                categories.append("both_wrong_partial_worsening")
    elif s in ["stable_fix", "stable_regression"]:
        categories.append(s + ("_FN" if item["gold"] else "_FP"))
    if v["ordinal_direction_reversal"]:
        categories.append("ordinal_direction_reversal")
    if s == "variable" and d is not None and (d > 0):
        categories.append("unstable_correct_count_improvement")
    if s == "variable" and d is not None and (d < 0):
        categories.append("unstable_correct_count_worsening")
    if s in ["both_always_wrong", "both_always_correct"]:
        categories.append(s)
    return categories


def choose_cases(items):
    groups = defaultdict(list)
    selected = defaultdict(list)
    for row in items:
        if not row["common_valid"]:
            selected[row["key"]].append("full_roster_missing_output")
            continue
        test = row["phase"] in ["de_test", "de_legal_test"]
        for pair, v in row["pairs"].items():
            categories = case_categories(row, pair, "legal" in row["phase"])
            if not test:
                if pair != "general_2_minus_mixed_2":
                    continue
                categories = [
                    name
                    for name in categories
                    if name.startswith("stable_fix")
                    or name.startswith("stable_regression")
                    or name == "ordinal_direction_reversal"
                ]
            for name in categories:
                groups[row["phase"], row["source"], pair, name].append(row["key"])
        if test and row["joint_path"] not in ["variable", "unavailable"]:
            groups[row["phase"], row["source"], "three_conditions", row["joint_path"]].append(
                row["key"]
            )
    for group, keys in sorted(groups.items()):
        key = min(keys, key=lambda k: hashlib.sha256(("part5-cases|" + k).encode()).hexdigest())
        selected[key].append("/".join(group))
    return (
        dict(selected),
        {
            "/".join(k): {
                "eligible": len(v),
                "selected": min(
                    v, key=lambda t: hashlib.sha256(("part5-cases|" + t).encode()).hexdigest()
                ),
            }
            for k, v in sorted(groups.items())
        },
    )


def run():
    contract = c.read(BASE / "analysis_contract.json")
    c.verify(contract["inputs"])
    c.validate()
    _, hp = h.verified()
    _, lp = l.verified()
    _, xp = context.verified()
    _, rp = reader.read_verified()
    context_rows = {r["key"]: r for r in c.rows(xp / "contexts.jsonl")}
    roster = {r["key"]: r for r in c.rows(c.BASE / "roster.jsonl")}
    endpoints = c.read(rp / "endpoint_setup.json")
    out = BASE / "runs" / f"run_{time.time_ns()}"
    out.mkdir(parents=True)
    all_items = []
    for phase in PHASES:
        legal = "legal" in phase
        prior = lp if legal else hp
        index = c.read(prior / phase / "input_roster.json")
        old = c.read(prior / phase / "results.json")
        with np.load(prior / phase / "arrays.npz", allow_pickle=False) as f:
            y, p, mask = (f["gold"], f["predictions"], f["common_mask"])
        states = correct(y, p)
        require(
            np.array_equal((states >= 0).all(axis=(0, 1)), mask), "Original common mask changed"
        )
        require(
            [k for k, on in zip(index["keys"], mask) if on] == old["common_keys"],
            "Original common roster changed",
        )
        full_stability = stability_summary(y, p, states)
        common_stability = stability_summary(y[mask], p[:, :, mask], states[:, :, mask])
        result = {
            "phase": phase,
            "full_inputs": len(y),
            "common_inputs": int(mask.sum()),
            "full_stability": full_stability,
            "common_stability": common_stability,
            "source_results": {},
        }
        if legal:
            result["legal_detail"] = legal_detail(y[mask], p[:, :, mask], states[:, :, mask])
        for source in sorted(set(index["sources"])):
            full = np.array([s == source for s in index["sources"]])
            selected = full & mask
            sy, sp = (y[selected], p[:, :, selected])
            scores = {
                cond: (
                    l.summarize(*l.totals(sy, sp[ci]), [0, 1, 2])
                    if legal
                    else h.summarize(h.contributions(sy, sp[ci]).sum(axis=-2))
                )
                for ci, cond in enumerate(CONDITIONS)
            }
            source_result = {
                "full_inputs": int(full.sum()),
                "common_inputs": int(selected.sum()),
                "full_gold_support": (
                    dict(zip(l.LABELS, y[full].sum(axis=0).tolist()))
                    if legal
                    else {"hate": int(y[full].sum()), "not_hate": int((y[full] == 0).sum())}
                ),
                "common_gold_support": (
                    dict(zip(l.LABELS, sy.sum(axis=0).tolist()))
                    if legal
                    else {"hate": int(sy.sum()), "not_hate": int((sy == 0).sum())}
                ),
                "common": scores,
                "full_stability": stability_summary(y[full], p[:, :, full], states[:, :, full]),
                "common_stability": stability_summary(sy, sp, states[:, :, selected]),
            }
            if legal:
                source_result["uncertainty_reused_from"] = c.ref(prior / phase / "results.json")
                source_result["condition_intervals"] = old["condition_descriptive_intervals"]
                source_result["contrasts"] = old["contrasts"]
            else:
                clusters = [v for v, on in zip(index["clusters"], selected) if on]
                boot, design = h.bootstrap(
                    sy,
                    sp,
                    clusters,
                    endpoints[phase]["source_memberships_by_cluster"],
                    SEEDS[phase + "/" + source],
                )
                values = {k: v.mean(axis=2) for k, v in h.metrics(boot).items()}
                source_result["condition_intervals"] = {
                    cond: {name: h.interval(values[name][:, ci]) for name in METRICS}
                    for ci, cond in enumerate(CONDITIONS)
                }
                source_result["contrasts"] = []
                for left, right in PAIRS:
                    li, ri = (CONDITIONS.index(left), CONDITIONS.index(right))
                    source_result["contrasts"].append(
                        {
                            "left": left,
                            "right": right,
                            "role": "descriptive source diagnostic, not a new main family",
                            "metrics": {
                                name: dict(
                                    h.interval(values[name][:, li] - values[name][:, ri]),
                                    difference=scores[left]["mean"][name]
                                    - scores[right]["mean"][name],
                                    by_repeat=[
                                        scores[left]["repeats"][i][name]
                                        - scores[right]["repeats"][i][name]
                                        for i in range(3)
                                    ],
                                )
                                for name in METRICS
                            },
                        }
                    )
                save(out / phase / source / "bootstrap_design.json", design)
                arrays(out / phase / source / "bootstrap.npz", counts=boot)
            result["source_results"][source] = source_result
        for i, key in enumerate(index["keys"]):
            legal_errors = np.abs(p[:, :, i] - y[i]).sum(axis=-1) if legal and mask[i] else None
            row = {
                "key": key,
                "phase": phase,
                "source": index["sources"][i],
                "text_cluster": index["clusters"][i],
                "common_valid": bool(mask[i]),
                "gold": y[i].tolist(),
                "correct_states": states[:, :, i],
                "predictions": p[:, :, i],
                "joint_path": joint_path(states[:, :, i]),
                "pairs": {
                    left
                    + "_minus_"
                    + right: individual_pair(
                        states[CONDITIONS.index(left), :, i], states[CONDITIONS.index(right), :, i]
                    )
                    for left, right in PAIRS
                },
                "legal_error_differences": (
                    {
                        left
                        + "_minus_"
                        + right: int(
                            legal_errors[CONDITIONS.index(left)].sum()
                            - legal_errors[CONDITIONS.index(right)].sum()
                        )
                        for left, right in PAIRS
                    }
                    if legal_errors is not None
                    else {}
                ),
                "context": {
                    k: context_rows[key][k]
                    for k in [
                        "guideline_ids",
                        "original_legal_count",
                        "same_messages_mixed_general",
                        "added_legal_status",
                    ]
                },
            }
            all_items.append(h.safe(row))
        for cond in CONDITIONS:
            for metric in (
                ("exact_correct", "extra_labels", "missed_labels")
                if legal
                else ("tp", "tn", "fp", "fn")
            ):
                require(
                    abs(
                        sum(
                            (
                                v["common"][cond]["mean"][metric]
                                for v in result["source_results"].values()
                            )
                        )
                        - old["common"][cond]["mean"][metric]
                    )
                    < 1e-09,
                    "Source totals differ from earlier results",
                )
        save(out / phase / "results.json", result)
        print(phase + ": source scores, transitions and repeat consistency saved.", flush=True)
    jsonlines(out / "individuals.jsonl", all_items)
    selected, groups = choose_cases(all_items)
    save(
        out / "case_selection.json",
        {"rule": contract["case_selection"], "groups": groups, "selected": selected},
    )
    prompts = defaultdict(dict)
    predictions = defaultdict(lambda: defaultdict(dict))
    items = {r["key"]: r for r in all_items}
    for row in c.rows(c.BASE / "prompts.jsonl"):
        if row["key"] in selected:
            prompts[row["key"]][row["condition"]] = row
    bindings_checked = 0
    for row in c.rows(rp / "predictions.jsonl"):
        require(
            row["messages_sha256"] == context_rows[row["key"]]["messages_sha256"][row["condition"]],
            "Result prompt identity changed",
        )
        bindings_checked += 1
        if row["key"] in selected:
            predictions[row["key"]][row["condition"]][str(row["repeat"])] = {
                k: row[k] for k in ["result", "evidence", "events", "n_valid"]
            }
    require(
        bindings_checked == 33093 and len(all_items) == 3677, "Incomplete Part 5 evidence matrix"
    )
    cases = []
    for key in sorted(selected):
        cases.append(
            {
                "key": key,
                "selection_reasons": selected[key],
                "individual": items[key],
                "text": roster[key]["text"],
                "gold_hate": roster[key].get("gate"),
                "gold_legal": roster[key].get("legal"),
                "prompts": prompts[key],
                "predictions": predictions[key],
            }
        )
    jsonlines(out / "cases.jsonl", cases)
    save(
        out / "summary.json",
        {
            "inputs": len(all_items),
            "cases": len(cases),
            "selection_groups": len(groups),
            "prediction_bindings_checked": bindings_checked,
            "missing_inputs": [r["key"] for r in all_items if not r["common_valid"]],
            "actual_Qwen_calls": 0,
            "D_writes": 0,
        },
    )
    c.verify(contract["inputs"])
    c.validate()
    save(
        out / "manifest.json",
        {
            "status": "part5_pending_independent_check",
            "created_utc": c.a.utc(),
            "run": c.ref(out),
            "outputs": c.bindings((p for p in out.rglob("*") if p.is_file())),
            "sources": c.bindings(sources()),
            "actual_Qwen_calls": 0,
        },
    )
    print(json.dumps({"manifest": c.ref(out / "manifest.json"), "cases": len(cases)}), flush=True)


def verified():
    done = c.read(BASE / "complete.json")
    require(done["status"] == "part5_numerically_verified", "Part 5 verification missing")
    c.verify(done["bindings"])
    manifest = c.read(c.ROOT / done["manifest"])
    c.verify(manifest["outputs"])
    return (manifest, c.ROOT / manifest["run"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run"])
    args = parser.parse_args()
    with offline():
        prepare() if args.mode == "prepare" else run()

from src.hsrag.paths import PROJECT_ROOT
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import socket
from statistics import mean, stdev
import time
import traceback
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

ROOT = PROJECT_ROOT
STUDY = ROOT / "experiments/astra/sq3_p3_audit_20260928_v1"
EVIDENCE = (
    ROOT / "Thesis_Log/astra_result/evidence/sq3_p3_historical_audit_20260928_v1.json"
)
LABELS = [
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
]
FIELDS = ["hate", "target_group", "hate_type", "severity", "legal"]
DRAWS = 10000
SEED = 20260928
SOURCES = [
    "src/hsrag/metrics.py",
    "src/hsrag/schema.py",
    "src/hsrag/vote.py",
    "src/hsrag/classify.py",
    "src/hsrag/kb.py",
    "src/hsrag/prompt.py",
    "analysis/feedback/relevance/prototype_scores.py",
    "analysis/prototype/validate_restoration.py",
    "analysis/prototype/unreachable_sources.py",
    "scripts/apply_feedback.py",
    "scripts/make_sq3_subset.py",
    "analysis/prototype/prepare_restoration.py",
    "analysis/prototype/relevance_run.py",
    "analysis/prototype/score_roles.py",
    "Thesis_Log/sq3_log.md",
    "Thesis_Log/experiment_log.md",
    "config/taxonomy.yaml",
    "config/taxonomy_backup.yaml",
    "config/taxonomy_no_grievance.yaml",
    "data/processed/en_dev_eval_sq3_types.parquet",
    "data/processed/en_train.parquet",
    "experiments/sq3_sentinel_r0.json",
    "kb/records.jsonl",
    "kb/records_no_grievance.jsonl",
    "kb/feedback_log.jsonl",
]
GROUPS = {
    "baseline_rag": [(f"sq3_round0_r{i}", "rag") for i in range(1, 5)],
    "feedback": [(s, "rag") for s in ["sq3_r4_fb", "sq3_r4_fb_rep2", "sq3_r4_fb_rep3"]],
    "control": [
        (s, "rag") for s in ["sq3_r4_ctl", "sq3_r4_ctl_rep2", "sq3_r4_ctl_rep3"]
    ],
    "oracle_A": [(s, "rag") for s in ["oracle_A", "oracle_A_rep2", "oracle_A_rep3"]],
    "oracle_B": [(s, "rag") for s in ["oracle_B", "oracle_B_rep2", "oracle_B_rep3"]],
    "restored_zero_shot": [(f"sq3_types_sq2_r{i}", "zero_shot") for i in range(1, 4)],
    "removed_zero_shot": [(f"nolabel_zs_r{i}", "zero_shot") for i in range(1, 3)],
    "removed_rag": [(f"nolabel_rag_r{i}", "rag") for i in range(1, 3)],
}
EXTRA = [(f"sq3_r{i}_{arm}", "rag") for i in range(1, 4) for arm in ("fb", "ctl")]
COMPARISONS = [
    ("feedback_minus_control_fixed", "feedback", "control", "fixed"),
    ("feedback_minus_control_held_out", "feedback", "control", "held_out"),
    ("oracle_A_minus_B", "oracle_A", "oracle_B", "held_out"),
    ("oracle_A_minus_feedback", "oracle_A", "feedback", "held_out"),
    ("taxonomy_zero_shot", "restored_zero_shot", "removed_zero_shot", "all"),
    ("taxonomy_rag", "baseline_rag", "removed_rag", "all"),
]


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf8"))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")


def jsonl(path):
    with Path(path).open(encoding="utf8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def verify(hashes):
    for name, value in hashes.items():
        require(digest(ROOT / name) == value, "Input changed: " + name)


def forbid(*args, **kwargs):
    raise AssertionError("Network/model access forbidden in offline audit")


def canon(text):
    return " ".join(str(text).lower().split())


def vote(predictions):
    valid = [p for p in predictions if p is not None]
    if not valid:
        return None, 0, False
    n = len(valid)
    yes = sum(p["hate"] for p in valid)
    out = {"hate": yes * 2 > n}
    for field in ("target_group", "hate_type", "legal"):
        counts = Counter(label for p in valid for label in p[field])
        out[field] = sorted(l for l, c in counts.items() if c * 2 > n)
    sev = ["low", "medium", "high"]
    ranks = sorted(sev.index(p["severity"]) for p in valid if p["severity"] is not None)
    out["severity"] = sev[ranks[(len(ranks) - 1) // 2]] if ranks else None
    if not out["hate"]:
        out.update(target_group=[], hate_type=[], severity=None)
    return out, n, yes * 2 == n


def check_prediction(value, allowed):
    if value is None:
        return
    require(type(value["hate"]) is bool, "Nonboolean gate")
    for field in ("target_group", "hate_type", "legal"):
        require(
            isinstance(value[field], list)
            and len(value[field]) == len(set(value[field])),
            "Duplicate/nonlist labels",
        )
    require(set(value["hate_type"]) <= set(allowed), "Out-of-taxonomy saved type")
    require(value["severity"] in (None, "low", "medium", "high"), "Invalid severity")
    require(
        value["hate"]
        or (
            value["target_group"] == []
            and value["hate_type"] == []
            and value["severity"] is None
        ),
        "Uncleared fine labels",
    )


def f1counts(tp, fp, fn):
    den = 2 * tp + fp + fn
    return np.divide(2 * tp, den, out=np.zeros_like(den, dtype=float), where=den != 0)


def bits(ids, gold, outputs):
    y = np.asarray([[l in gold[i] for l in LABELS] for i in ids], dtype=bool)
    x = np.asarray(
        [[l in outputs[i]["result"]["hate_type"] for l in LABELS] for i in ids],
        dtype=bool,
    )
    return y, x


def metric(ids, gold, outputs):
    valid = [i for i in ids if outputs[i]["result"] is not None]
    missing = sorted(set(ids) - set(valid))
    require(valid, "Empty scored set")
    y, x = bits(valid, gold, outputs)
    tp = (y & x).sum(0)
    fp = (~y & x).sum(0)
    fn = (y & ~x).sum(0)
    values = f1counts(tp, fp, fn)
    per = {}
    for j, label in enumerate(LABELS):
        t = sum(
            label in gold[i] and label in outputs[i]["result"]["hate_type"]
            for i in valid
        )
        f = sum(
            label not in gold[i] and label in outputs[i]["result"]["hate_type"]
            for i in valid
        )
        n = sum(
            label in gold[i] and label not in outputs[i]["result"]["hate_type"]
            for i in valid
        )
        require((t, f, n) == (tp[j], fp[j], fn[j]), "Scalar/count disagreement")
        per[label] = {
            "tp": t,
            "fp": f,
            "fn": n,
            "f1": float(values[j]),
            "support_valid": int(y[:, j].sum()),
            "support_original": sum(label in gold[i] for i in ids),
        }
    pos = np.asarray([sum(l in gold[i] for i in missing) for l in LABELS])
    neg = len(missing) - pos
    return {
        "n": len(ids),
        "valid": len(valid),
        "null_ids": missing,
        "macro_f1": float(values.mean()),
        "full_roster_failure_bounds": [
            float(f1counts(tp, fp + neg, fn + pos).mean()),
            float(f1counts(tp + pos, fp, fn).mean()),
        ],
        "contains": float((~y | x).all(1).mean()),
        "exact": float((y == x).all(1).mean()),
        "predicted_labels": float(x.sum(1).mean()),
        "gate_false": sum(not outputs[i]["result"]["hate"] for i in valid),
        "per_label": per,
    }


def describe(values):
    return {
        "values": values,
        "mean": mean(values),
        "sample_sd": stdev(values) if len(values) > 1 else None,
        "range": max(values) - min(values),
    }


def sign_counts(left, right):
    require(len(left) == len(right), "Unequal item counts")
    better = sum(x > y for x, y in zip(left, right))
    worse = sum(x < y for x, y in zip(left, right))
    n = better + worse
    p = (
        min(1.0, 2 * sum(math.comb(n, k) for k in range(min(better, worse) + 1)) / 2**n)
        if n
        else 1.0
    )
    return {
        "better": better,
        "worse": worse,
        "tied": len(left) - n,
        "discordant_items": n,
        "two_sided_exact_p": p,
    }


def contains_vector(ids, gold, runs):
    return [
        Fraction(
            sum(gold[i] <= set(r[i]["result"]["hate_type"]) for r in runs), len(runs)
        )
        for i in ids
    ]


def paired_bootstrap(ids, gold, left, right, draws=DRAWS):
    n = len(ids)
    rng = np.random.default_rng(SEED)
    indices = rng.integers(0, n, size=(draws, n))
    weights = np.zeros((draws, n), dtype=np.int32)
    for j, idx in enumerate(indices):
        weights[j] = np.bincount(idx, minlength=n)
    results = []
    verified = 0
    for outputs in left + right:
        y, x = bits(ids, gold, outputs)
        contributions = [
            (y & x).astype(np.int32),
            (~y & x).astype(np.int32),
            (y & ~x).astype(np.int32),
        ]
        scores = f1counts(*(weights @ c for c in contributions)).mean(1)
        for begin in range(0, draws, 256):
            picked = indices[begin : begin + 256]
            alternate = f1counts(*(c[picked].sum(1) for c in contributions)).mean(1)
            require(
                np.allclose(scores[begin : begin + 256], alternate, rtol=0, atol=1e-13),
                "Expanded bootstrap differs",
            )
        results.append(scores)
        verified += draws
    gap = np.mean(results[: len(left)], axis=0) - np.mean(results[len(left) :], axis=0)
    return {
        "draws": draws,
        "seed": SEED,
        "ci95": np.quantile(gap, [0.025, 0.975], method="linear").tolist(),
        "independently_verified_setting_draws": verified,
        "scope": "Exploratory paired items; fixed original seven labels and fixed observed decoder repeats, unequal repeat counts retained; unadjusted 95%, not new confirmatory evidence.",
    }


def tests():
    counts = 0

    def p(h, t=(), legal=(), s=None):
        return {
            "hate": h,
            "hate_type": list(t),
            "target_group": [],
            "legal": list(legal),
            "severity": s,
        }

    v, n, tie = vote(
        [
            p(True, ["explicit"], ["insult_defamation"], "high"),
            p(False, legal=["insult_defamation"]),
        ]
    )
    require(
        v == p(False, legal=["insult_defamation"]) and n == 2 and tie,
        "Tie/legal fixture",
    )
    counts += 1
    require(vote([None, None, None]) == (None, 0, False), "Null vote fixture")
    counts += 1
    v, _, _ = vote([p(True, ["explicit"], s="high"), p(True, ["explicit"], s="low")])
    require(
        v["severity"] == "low" and v["hate_type"] == ["explicit"], "Even median fixture"
    )
    counts += 1
    try:
        check_prediction(p(False, ["explicit"]), LABELS)
    except AssertionError:
        pass
    else:
        raise AssertionError("False gate fixture accepted")
    counts += 1
    try:
        check_prediction(p(True, ["grievance"]), LABELS[:-1])
    except AssertionError:
        pass
    else:
        raise AssertionError("Removed label fixture accepted")
    counts += 1
    gold = {"a": {"grievance"}, "b": set()}
    out = {"a": {"result": p(False)}, "b": {"result": p(True)}}
    m = metric(["a", "b"], gold, out)
    require(
        m["macro_f1"] == 0 and m["contains"] == 0.5 and m["n"] == 2,
        "Known empty/fixed-label fixture",
    )
    counts += 1
    missing = {"a": {"result": None}, "b": {"result": p(True)}}
    m = metric(["a", "b"], gold, missing)
    require(
        m["valid"] == 1
        and m["null_ids"] == ["a"]
        and m["full_roster_failure_bounds"] == [0, 1 / 7],
        "Failure bounds fixture",
    )
    counts += 1
    sc = sign_counts(
        [Fraction(2, 3), Fraction(1, 3), Fraction(0)],
        [Fraction(1, 2), Fraction(1, 2), Fraction(0)],
    )
    require(
        (sc["better"], sc["worse"], sc["tied"], sc["two_sided_exact_p"])
        == (1, 1, 1, 1),
        "Unequal-repeat sign fixture",
    )
    counts += 1
    require(
        abs(
            sign_counts([1] * 15 + [0] * 2, [0] * 15 + [1] * 2)["two_sided_exact_p"]
            - 0.002349853515625
        )
        < 1e-15,
        "Exact sign fixture",
    )
    counts += 1
    left = {"a": {"result": p(True, ["grievance"])}, "b": {"result": p(True)}}
    bs = paired_bootstrap(["a", "b"], gold, [left], [out], draws=31)
    require(
        bs["independently_verified_setting_draws"] == 62 and bs["ci95"][0] >= 0,
        "Bootstrap fixed labels fixture",
    )
    counts += 1
    record = {
        "passed": True,
        "fixtures": counts,
        "script_sha256": digest(__file__),
        "created_utc": utc(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    path = STUDY / "development" / ("tests_" + record["script_sha256"][:16] + ".json")
    save(path, record)
    print(json.dumps(record, indent=2), flush=True)


def run():
    require(
        not EVIDENCE.exists() and not (STUDY / "complete.json").exists(),
        "Preserve finished audit",
    )
    tested = STUDY / "development" / ("tests_" + digest(__file__)[:16] + ".json")
    test = read(tested)
    require(
        test["passed"] and test["script_sha256"] == digest(__file__),
        "Current passing fixtures required",
    )
    attempt = STUDY / "attempts" / str(time.time_ns())
    attempt.mkdir(parents=True)
    (attempt / "source.py").write_bytes(Path(__file__).read_bytes())
    try:
        execute(attempt, tested)
    except Exception:
        save(
            attempt / "failure.json",
            {
                "created_utc": utc(),
                "traceback": traceback.format_exc(),
                "script_sha256": digest(__file__),
                "new_API_calls": 0,
            },
        )
        raise


def execute(attempt, tested):
    from scipy.stats import binomtest
    from src.hsrag.metrics import score_multilabel

    selections = list(
        dict.fromkeys([k for rows in GROUPS.values() for k in rows] + EXTRA)
    )
    paths = [ROOT / name for name in SOURCES] + [Path(__file__), tested]
    paths += [
        ROOT / f"experiments/results/{stem}_live.jsonl"
        for stem in dict.fromkeys(stem for stem, _ in selections)
    ]
    paths += [
        ROOT / f"kb/records_sq3_{arm}_r{i}.jsonl"
        for i in range(1, 5)
        for arm in ("fb", "ctl")
    ]
    paths += [
        ROOT / f"data/encoder/{arm}/hate_type_meta.json"
        for arm in ("full", "full_nogrievance")
    ]
    paths += [
        ROOT / f"models/encoder/{arm}/hate_type/seed{s}/meta.json"
        for arm in ("full", "full_nogrievance")
        for s in range(3)
    ]
    hashes = {rel(p): digest(p) for p in paths}
    save(
        attempt / "plan.json",
        {
            "created_utc": utc(),
            "input_hashes": hashes,
            "groups": GROUPS,
            "comparisons": COMPARISONS,
            "fixed_labels": LABELS,
            "draws": DRAWS,
            "seed": SEED,
            "new_API_calls": 0,
            "analysis_status": "post-hoc audit of already inspected historical DEV",
        },
    )
    df = pd.read_parquet(ROOT / "data/processed/en_dev_eval_sq3_types.parquet")
    rows = {str(r.id): r for r in df.itertuples(index=False)}
    require(len(rows) == len(df) == 467, "Gold IDs changed")
    gold = {
        i: set(
            r.hate_types.tolist() if hasattr(r.hate_types, "tolist") else r.hate_types
        )
        for i, r in rows.items()
    }
    require(
        all(
            bool(r.gate) and gold[i] and gold[i] <= set(LABELS) for i, r in rows.items()
        ),
        "Gold eligibility differs",
    )
    strata = {
        "all": list(rows),
        "fixed": [i for i, r in rows.items() if r.sq3_role in ("pool", "held_out")],
        "held_out": [i for i, r in rows.items() if r.sq3_role == "held_out"],
        "pool": [i for i, r in rows.items() if r.sq3_role == "pool"],
    }
    require(
        {k: len(v) for k, v in strata.items()}
        == {"all": 467, "fixed": 350, "held_out": 121, "pool": 229},
        "Partition differs",
    )
    all_runs = {}
    inventory = []
    verified_votes = 0
    for stem, arm in selections:
        selected = {}
        all_count = 0
        for row in jsonl(ROOT / f"experiments/results/{stem}_live.jsonl"):
            if "_manifest" in row:
                continue
            all_count += 1
            if row.get("arm") != arm:
                continue
            rid = row["item_id"]
            require(rid not in selected, "Duplicate saved identity " + stem + "/" + rid)
            selected[rid] = row
        wanted = set(
            strata["held_out"] if stem.startswith("oracle_") else strata["all"]
        )
        require(set(selected) == wanted, "Run IDs differ " + stem)
        stamps = {
            k: Counter()
            for k in (
                "active_model",
                "prompt_version",
                "taxonomy_version",
                "code_version",
                "kb_version",
                "temperature",
                "workers",
                "n_runs",
            )
        }
        counts = Counter()
        retrieval_configs = Counter()
        field_differences = []
        allowed = LABELS[:-1] if stem.startswith("nolabel_") else LABELS
        for rid, row in selected.items():
            require(
                row["text_hash"]
                == hashlib.sha256(
                    " ".join(str(rows[rid].text).split()).encode()
                ).hexdigest()[:16],
                "Input text differs " + stem + "/" + rid,
            )
            require(
                row["n_runs"]
                == len(row["run_predictions"])
                == len(row["raw_runs"])
                == 3,
                "Vote list differs",
            )
            check_prediction(row["result"], allowed)
            for pred in row["run_predictions"]:
                check_prediction(pred, allowed)
            v, n, tie = vote(row["run_predictions"])
            expected = (
                None if row["result"] is None else {f: row["result"][f] for f in FIELDS}
            )
            require(
                v == expected and n == row["n_valid"] and tie == row["uncertain"],
                "Independent vote differs " + stem + "/" + rid,
            )
            require(row["parse_failures"] == 3 - n, "Parse accounting differs")
            verified_votes += 1
            counts["predictions"] += 1
            counts["null_predictions"] += row["result"] is None
            counts[f"valid_votes_{n}"] += 1
            for key in (
                "parse_failures",
                "repairs",
                "normalisations",
                "gate_normalised",
            ):
                counts[key] += row[key]
            counts["saved_raw_response_turns"] += sum(
                len(turns) for turns in row["raw_runs"]
            )
            for key in stamps:
                stamps[key][json.dumps(row.get(key), sort_keys=True)] += 1
            retrieval_configs[
                json.dumps(row.get("retrieval_config"), sort_keys=True)
            ] += 1
        require(
            set(stamps["active_model"]) == {json.dumps("Qwen/Qwen3-VL-32B-Instruct")},
            "Unexpected model",
        )
        require(
            set(stamps["prompt_version"]) == {json.dumps("classify_v1-8e588db7")},
            "Unexpected template",
        )
        require(set(stamps["temperature"]) <= {"1.0", "1"}, "Unexpected temperature")
        all_runs[stem, arm] = selected
        inventory.append(
            {
                "stem": stem,
                "arm": arm,
                "source": f"experiments/results/{stem}_live.jsonl",
                "source_rows_all_arms": all_count,
                "counts": dict(counts),
                "stamps": {k: dict(v) for k, v in stamps.items()},
                "retrieval_configs": dict(retrieval_configs),
                "timestamp_min": min(r["timestamp"] for r in selected.values()),
                "timestamp_max": max(r["timestamp"] for r in selected.values()),
            }
        )
        print("Checked saved votes and identity: " + stem + "/" + arm, flush=True)
    runs = {g: [all_runs[k] for k in keys] for g, keys in GROUPS.items()}
    scored = {}
    gold_frame = rows
    for group, group_runs in runs.items():
        applicable = ["held_out"] if group.startswith("oracle_") else list(strata)
        scored[group] = {}
        for stratum in applicable:
            ids = strata[stratum]
            results = []
            for run in group_runs:
                m = metric(ids, gold, run)
                old = score_multilabel(
                    [run[i] for i in ids], gold_frame, "hate_types", "hate_type"
                )
                require(
                    set(old.labels_averaged) == set(LABELS)
                    and abs(old.macro_f1 - m["macro_f1"]) < 1e-13,
                    "Independent/project score differs",
                )
                require(
                    old.n_items == m["valid"] and old.n_unscored == len(m["null_ids"]),
                    "Original coverage differs",
                )
                require(
                    abs(old.extra["gold_subset_of_pred"] - m["contains"]) < 1e-13,
                    "Contains score differs",
                )
                results.append(m)
            scored[group][stratum] = {
                "runs": results,
                "macro": describe([r["macro_f1"] for r in results]),
                "contains": describe([r["contains"] for r in results]),
            }
    comparisons = {}
    for name, left_name, right_name, stratum in COMPARISONS:
        left = runs[left_name]
        right = runs[right_name]
        original = strata[stratum]
        ids = [
            i for i in original if all(r[i]["result"] is not None for r in left + right)
        ]
        lm = [metric(ids, gold, r) for r in left]
        rm = [metric(ids, gold, r) for r in right]
        lv = [m["macro_f1"] for m in lm]
        rv = [m["macro_f1"] for m in rm]
        gap = mean(lv) - mean(rv)
        floor = max(max(lv) - min(lv), max(rv) - min(rv))
        paired = sign_counts(
            contains_vector(ids, gold, left), contains_vector(ids, gold, right)
        )
        n = paired["discordant_items"]
        require(
            n == 0
            or abs(
                paired["two_sided_exact_p"] - binomtest(paired["better"], n, 0.5).pvalue
            )
            < 1e-12,
            "Exact p disagrees with scipy",
        )
        individual = []
        for j, l in enumerate(left):
            for k, r in enumerate(right):
                c = sign_counts(
                    contains_vector(ids, gold, [l]), contains_vector(ids, gold, [r])
                )
                individual.append(
                    {"left_repeat_index": j + 1, "right_repeat_index": k + 1, **c}
                )
        target = [i for i in ids if "grievance" in gold[i]]
        bounds = [
            mean(
                scored[left_name][stratum]["runs"][j]["full_roster_failure_bounds"][0]
                for j in range(len(left))
            )
            - mean(
                scored[right_name][stratum]["runs"][j]["full_roster_failure_bounds"][1]
                for j in range(len(right))
            ),
            mean(
                scored[left_name][stratum]["runs"][j]["full_roster_failure_bounds"][1]
                for j in range(len(left))
            )
            - mean(
                scored[right_name][stratum]["runs"][j]["full_roster_failure_bounds"][0]
                for j in range(len(right))
            ),
        ]
        comparisons[name] = {
            "left": left_name,
            "right": right_name,
            "stratum": stratum,
            "original_n": len(original),
            "shared_n": len(ids),
            "shared_ids": ids,
            "excluded_ids": sorted(set(original) - set(ids)),
            "left_repeats": len(left),
            "right_repeats": len(right),
            "left_macro": describe(lv),
            "right_macro": describe(rv),
            "macro_difference": gap,
            "maximum_decoder_range": floor,
            "difference_over_range": gap / floor if floor else None,
            "full_roster_failure_difference_bounds": bounds,
            "contains_mean_difference": float(
                mean(contains_vector(ids, gold, left))
                - mean(contains_vector(ids, gold, right))
            ),
            "paired_mean_contains_sign": paired,
            "per_repeat_contains_pairs": individual,
            "grievance_target_n": len(target),
            "grievance_target_mean_contains_sign": sign_counts(
                contains_vector(target, gold, left),
                contains_vector(target, gold, right),
            ),
            "macro_bootstrap": paired_bootstrap(ids, gold, left, right),
        }
        print("Recomputed paired comparison: " + name, flush=True)
    kb = {}
    kb_audit = {}
    train_df = pd.read_parquet(ROOT / "data/processed/en_train.parquet")
    train = {str(r.id): r for r in train_df.itertuples(index=False)}
    base_records = None
    origin_overlaps = []
    for arm in ("fb", "ctl"):
        previous = set()
        for round_n in range(1, 5):
            path = ROOT / f"kb/records_sq3_{arm}_r{round_n}.jsonl"
            records = list(jsonl(path))
            lookup = {r["id"]: r for r in records}
            require(len(lookup) == len(records), "Duplicate KB IDs")
            fb = {i: r for i, r in lookup.items() if r["kind"] == "feedback"}
            base = {i: r for i, r in lookup.items() if r["kind"] != "feedback"}
            if base_records is None:
                base_records = base
            require(
                base == base_records, "Historical base records changed across rounds"
            )
            require(
                previous <= set(fb) and len(fb) == [20, 33, 48, 60][round_n - 1],
                "Correction pool size/lineage differs",
            )
            info = {
                "records": len(records),
                "feedback": len(fb),
                "added": len(set(fb) - previous),
                "train_topups": 0,
                "eval_origins": [],
            }
            for rid, rec in fb.items():
                origin = rec["meta"]["origin_item_id"]
                is_eval = origin in rows
                source = rows[origin] if is_eval else train[origin]
                if is_eval:
                    require(
                        source.sq3_role == "batches",
                        "Correction entered held-out/pool role",
                    )
                    info["eval_origins"].append(origin)
                else:
                    require(arm == "ctl", "Feedback correction from TRAIN")
                    info["train_topups"] += 1
                require(
                    rec["text"] == str(source.text)
                    and set(rec["meta"]["hate_types"]) == set(source.hate_types)
                    and rec["meta"]["gate"] == bool(source.gate),
                    "Correction source label/text differs",
                )
                require(
                    rid
                    == "fb-"
                    + hashlib.sha1(" ".join(rec["text"].split()).encode()).hexdigest()[
                        :12
                    ],
                    "Correction ID differs",
                )
                for iid in strata["fixed"]:
                    if canon(rec["text"]).lstrip(": ") == canon(rows[iid].text).lstrip(
                        ": "
                    ):
                        origin_overlaps.append(
                            {
                                "arm": arm,
                                "round": round_n,
                                "record": rid,
                                "origin": origin,
                                "fixed_item": iid,
                                "exact_normalized": canon(rec["text"])
                                == canon(rows[iid].text),
                            }
                        )
            stem = f"sq3_r{round_n}_{arm}"
            run = all_runs[stem, "rag"]
            require(
                {r["kb_version"] for r in run.values()} == {digest(path)[:16]},
                "KB stamp differs " + stem,
            )
            hits = []
            for iid in strata["held_out"]:
                ids = run[iid]["retrieved"]["Additional labelled examples"]
                require(
                    len(ids) == 1 and ids[0] in fb, "Missing/unknown feedback retrieval"
                )
                require(
                    len(run[iid]["retrieved"]["Labelled examples"]) == 5,
                    "Regular example count changed",
                )
                hits.append(bool(gold[iid] & set(fb[ids[0]]["meta"]["hate_types"])))
            info.update(
                label_relevant_heldout=sum(hits),
                heldout_n=len(hits),
                label_relevance=sum(hits) / len(hits),
            )
            kb[arm, round_n] = lookup
            kb_audit[stem] = info
            previous = set(fb)
    held = strata["held_out"]
    oracles = {}
    injection_changes = []
    context_changes = []

    def stripped(row):
        return {
            k: v
            for k, v in row["retrieved"].items()
            if k not in ("injected", "Additional labelled examples")
        }

    reference = runs["oracle_A"][0]
    for condition in ("A", "B"):
        condition_runs = runs["oracle_" + condition]
        changed = []
        for j, run in enumerate(condition_runs):
            for iid in held:
                row = run[iid]
                ids = row["retrieved"].get("injected", [])
                require(
                    len(ids) == 1
                    and ids == row["retrieved"]["Additional labelled examples"],
                    "Injection slot differs",
                )
                rec = kb["fb", 4][ids[0]]
                require(
                    rec["kind"] == "feedback"
                    and bool(set(rec["meta"]["hate_types"]) & gold[iid])
                    == (condition == "A"),
                    "Wrong oracle label manipulation",
                )
                require(
                    row["kb_version"]
                    == digest(ROOT / "kb/records_sq3_fb_r4.jsonl")[:16],
                    "Oracle KB stamp differs",
                )
                if ids != condition_runs[0][iid]["retrieved"]["injected"]:
                    injection_changes.append(
                        {"condition": condition, "repeat": j + 1, "item": iid}
                    )
                if stripped(row) != stripped(reference[iid]):
                    context_changes.append(
                        {"condition": condition, "repeat": j + 1, "item": iid}
                    )
        oracles[condition] = {
            "verified_injections": len(held) * len(condition_runs),
            "unique_injected_records": len(
                {r[i]["retrieved"]["injected"][0] for r in condition_runs for i in held}
            ),
        }
    table = Counter()
    never = []
    always = []
    partial_equal = []
    unequal = []
    subgroups = defaultdict(list)
    for iid in held:
        ca = sum(
            gold[iid] <= set(r[iid]["result"]["hate_type"]) for r in runs["oracle_A"]
        )
        cb = sum(
            gold[iid] <= set(r[iid]["result"]["hate_type"]) for r in runs["oracle_B"]
        )
        table[f"{ca},{cb}"] += 1
        if ca == cb == 0:
            never.append(iid)
            preds = [r[iid]["result"] for r in runs["oracle_A"] + runs["oracle_B"]]
            group = (
                "gate_false_all_six"
                if all(not p["hate"] for p in preds)
                else (
                    "identical_nonempty_wrong_set"
                    if len({tuple(sorted(p["hate_type"])) for p in preds}) == 1
                    else "varying_wrong_sets"
                )
            )
            subgroups[group].append(iid)
        elif ca == cb == 3:
            always.append(iid)
        elif ca == cb:
            partial_equal.append(iid)
        else:
            unequal.append(iid)
    natural_context_differences = [
        i for i in held if stripped(runs["feedback"][0][i]) != stripped(reference[i])
    ]
    pooled = Counter()
    for k in range(3):
        pc = sign_counts(
            contains_vector(held, gold, [runs["oracle_A"][k]]),
            contains_vector(held, gold, [runs["oracle_B"][k]]),
        )
        pooled.update({f: pc[f] for f in ("better", "worse", "tied")})
    sentinel = read(ROOT / "experiments/sq3_sentinel_r0.json")
    loo = []
    for excluded in range(4):
        selected = [
            i
            for i in strata["pool"]
            if all(
                gold[i] <= set(r[i]["result"]["hate_type"])
                for j, r in enumerate(runs["baseline_rag"])
                if j != excluded
            )
        ]
        flips = sum(
            not gold[i] <= set(runs["baseline_rag"][excluded][i]["result"]["hate_type"])
            for i in selected
        )
        loo.append(
            {
                "test_repeat": excluded + 1,
                "n": len(selected),
                "flips": flips,
                "rate": flips / len(selected),
            }
        )
        if excluded == 3:
            require(
                set(selected) == set(sentinel["sentinel_ids"]),
                "Frozen sentinel membership differs",
            )
    removed = list(jsonl(ROOT / "kb/records_no_grievance.jsonl"))
    removed_lookup = {r["id"]: r for r in removed}
    dropped = [rec for rid, rec in base_records.items() if rid not in removed_lookup]
    preserved_differences = [
        i for i in removed_lookup if removed_lookup[i] != base_records.get(i)
    ]
    tax_docs = {
        n: yaml.safe_load((ROOT / n).read_text(encoding="utf8"))
        for n in (
            "config/taxonomy.yaml",
            "config/taxonomy_backup.yaml",
            "config/taxonomy_no_grievance.yaml",
        )
    }
    tax_labels = {
        n: list(v["dimensions"]["hate_type"]["labels"]) for n, v in tax_docs.items()
    }
    ds = {
        arm: read(ROOT / f"data/encoder/{arm}/hate_type_meta.json")
        for arm in ("full", "full_nogrievance")
    }
    times = {
        arm: [
            read(ROOT / f"models/encoder/{arm}/hate_type/seed{s}/meta.json")[
                "train_seconds"
            ]
            for s in range(3)
        ]
        for arm in ds
    }
    result = {
        "status": "historical_sq3_central_claims_audited",
        "created_utc": utc(),
        "input_hashes": hashes,
        "plan": rel(attempt / "plan.json"),
        "plan_sha256": digest(attempt / "plan.json"),
        "tests": read(tested),
        "fixed_labels": LABELS,
        "strata": {
            k: {
                "n": len(v),
                "gold_support": dict(Counter(l for i in v for l in gold[i])),
                "ids": v,
            }
            for k, v in strata.items()
        },
        "run_inventory": inventory,
        "saved_aggregates_independently_revoted": verified_votes,
        "group_scores": scored,
        "comparisons": comparisons,
        "feedback_kbs": kb_audit,
        "known_prefix_normalized_overlaps": origin_overlaps,
        "oracle": {
            "injection_checks": oracles,
            "injection_changes_across_repeats": injection_changes,
            "nonfeedback_context_changes_against_A1": context_changes,
            "natural_feedback_rep1_nonfeedback_context_changes": natural_context_differences,
            "correct_count_joint_table": dict(table),
            "correct_all_six_ids": always,
            "incorrect_all_six_ids": never,
            "equal_partial_ids": partial_equal,
            "unequal_correct_count_ids": unequal,
            "incorrect_subgroups": dict(subgroups),
            "naively_pooled_matched_repeat_discordances": dict(pooled),
        },
        "sentinel_leave_one_out": loo,
        "sentinel_note": "Overlapping leave-one-out selections, not four statistically independent validation sets.",
        "taxonomy_kb": {
            "base_records": len(base_records),
            "removed_records": len(removed),
            "dropped_ids": [r["id"] for r in dropped],
            "dropped_kinds": dict(Counter(r["kind"] for r in dropped)),
            "changed_preserved_records": preserved_differences,
            "taxonomy_label_lists": tax_labels,
            "file_taxonomy_stamps": {
                n: "taxonomy-" + digest(ROOT / n)[:8] for n in tax_docs
            },
        },
        "historical_encoder_cost_metadata": {
            "rows": {
                k: {f: v[f] for f in ("n_train", "n_val", "leakage_excluded")}
                for k, v in ds.items()
            },
            "difference_train_plus_val": sum(
                ds["full"][f] - ds["full_nogrievance"][f] for f in ("n_train", "n_val")
            ),
            "train_seconds_by_seed": times,
            "training_seconds_sum": {k: sum(v) for k, v in times.items()},
            "qualification": "Legacy workflow metadata only: does not establish minimal annotation need, one-checkpoint cost, end-to-end update cost or aligned current full-KB encoder results.",
        },
        "new_API_calls": 0,
        "new_retrieval": False,
        "new_training": False,
        "new_TEST_scoring": False,
        "limits": [
            "Post-hoc DEV audit, not a new model experiment.",
            "Stored prediction votes and classification fields replayed; historical raw-response parser/repair chronology not rerun.",
            "Prompt template/record IDs and available stamps do not prove byte-identical historical payloads or immutable remote model weights.",
            "Paired macro intervals are new, exploratory, unadjusted and conditional on the fixed observed repeats. No replicate pooling into independent inputs.",
            "No global impossibility bound, no proof of a novel concept learned, no blanket audit of every original SQ3 side experiment.",
        ],
    }
    verify(hashes)
    save(EVIDENCE, result)
    save(
        STUDY / "complete.json",
        {
            "status": result["status"],
            "evidence": rel(EVIDENCE),
            "evidence_sha256": digest(EVIDENCE),
            "new_API_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "revoted": verified_votes,
                "comparisons": {
                    k: {
                        f: r[f]
                        for f in (
                            "shared_n",
                            "macro_difference",
                            "paired_mean_contains_sign",
                            "grievance_target_mean_contains_sign",
                            "macro_bootstrap",
                        )
                    }
                    for k, r in comparisons.items()
                },
                "oracle_joint": dict(table),
                "oracle_pooled": dict(pooled),
                "subgroups": {k: len(v) for k, v in subgroups.items()},
                "context_changes": len(context_changes),
                "injection_changes": len(injection_changes),
                "known_overlaps": origin_overlaps,
                "cost": result["historical_encoder_cost_metadata"],
                "evidence_sha256": digest(EVIDENCE),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tests", "run"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket, "create_connection", forbid
    ):
        tests() if args.mode == "tests" else run()

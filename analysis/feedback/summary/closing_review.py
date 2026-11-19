from src.hsrag.paths import PROJECT_ROOT
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
from statistics import mean
from unittest.mock import patch

ROOT = PROJECT_ROOT
OLD = ROOT / "experiments/astra/sq3_combined_results_20261004_v1"
DEST = ROOT / "experiments/astra/sq3_closing_review_20261004_v1"
STUDIES = {
    s: ROOT / f"experiments/astra/sq3_full_kb_{s}_20261001_v1"
    for s in ["taxonomy", "explanations"]
}
LABELS = [
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
]
SEEDS = [202609281, 202609282, 202609283]
PARTS = [
    "example_and_definition",
    "example_only",
    "definition_only",
    "neither_changed",
    "identical_messages",
]


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def rows(p):
    with p.open(encoding="utf8") as f:
        yield from (json.loads(line) for line in f if line.strip())


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def rel(p):
    return p.resolve().relative_to(ROOT).as_posix()


def require(v, msg):
    if not v:
        raise AssertionError(msg)


def save(name, value):
    p = DEST / name
    require(p.resolve().is_relative_to(DEST.resolve()), "Destination escape")
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("x", encoding="utf8", newline="\n") as f:
        json.dump(
            value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        f.write("\n")


def verify(bindings):
    for p, h in bindings.items():
        require(sha(ROOT / p) == h, "Changed input: " + p)


def ratio(a, b):
    return a / b if b else None


def f1(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0


def forbidden(*a, **k):
    raise RuntimeError("This review is offline")


def plan():
    files = [
        OLD / "complete.json",
        OLD / "review/predictions.jsonl",
        OLD / "taxonomy/primary.json",
        OLD / "explanations/primary_explanation.json",
        OLD / "explanations/primary_application.json",
        OLD / "explanations/all_transfer.json",
        OLD / "explanations/benign.json",
        STUDIES["taxonomy"] / "prompts.jsonl",
        STUDIES["explanations"] / "choices.jsonl",
        ROOT
        / "experiments/astra/sq3_full_kb_feedback_20260928_v1/feasibility_v1/roster.jsonl",
        ROOT / "src/hsrag/data/implicit_hate.py",
        ROOT / "data/mappings/implicit_hate_types.csv",
        ROOT / "data/raw/implicit_hate/implicit_hate_v1_stg1_posts.tsv",
        ROOT / "data/raw/implicit_hate/implicit_hate_v1_stg2_posts.tsv",
        Path(__file__),
    ]
    files += [p / "roster.jsonl" for p in STUDIES.values()]
    require(all(p.exists() for p in files), "Missing proposed input")

    execution = read(
        ROOT / "experiments/astra/sq3_combined_execution_20261001_v1/plan.json"
    )
    frozen = {**execution["input_hashes"], **execution["source_hashes"]}
    verify(frozen)
    save(
        "plan.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "status": "post_result_descriptive_plan_fixed_before_additional_calculations",
            "inputs": {rel(p): sha(p) for p in files},
            "frozen_execution_bindings": frozen,
            "questions": [
                "Partition target TP/FN changes into hate-gate changes and type changes while both gates are positive.",
                "Partition counts by delivered target example/definition, changed non-target context, or identical messages; preserve all categories.",
                "Measure added/dropped labels matching the injected correction versus other labels, on the fixed common hateful mask; benign gates stay separate.",
                "Independently rebuild local ImplicitHate gold from raw stage files, inspect annotation cardinality and explicit/implicit mixing without relabelling.",
            ],
            "taxonomy_mask": "Original 24-output primary mask, n466, for every target and repeat; use all rows including gate misses.",
            "correction_mask": "Original all-policy common hateful n349 and benign n203; three fixed pools, three repeats; shared reference counted once when reporting its standalone score.",
            "estimation": "Descriptive whole-roster counts and equal-repeat/pool means. No new primary, p value, selected threshold or significance claim. Delivery strata are observational; transitions do not identify a causal mediation effect.",
            "annotation": "Reproduce source loader ordering and primary/extra implicit label union from raw CSV without importing it. Check every hateful study roster ID/text/gold. Explicit is a parent-class-derived singleton; do not treat unannotated implicit subtypes as newly established correct labels.",
            "preservation": "No model requests, training, retrieval regeneration, prompt/gold edits, raw deletions or modification of completed scores.",
            "new_model_calls": 0,
        },
    )
    print("Additional descriptive review plan saved.", flush=True)


def bin_counts(gold, pred, label):
    y, x = label in gold, label in pred["hate_type"]
    return {
        "tp": int(y and x),
        "fp": int(not y and x),
        "fn": int(y and not x),
        "tn": int(not y and not x),
    }


def taxonomy(rr, roster, prompts):
    ids = read(OLD / "taxonomy/primary.json")["common_valid_ids"]
    out = {"ids": ids, "labels": {}, "runs": []}
    for label in LABELS:
        partitions = {}
        for i in ids:
            full = prompts[i, "rag_full"]
            base = prompts[i, "rag_schema_" + label]
            ex = any(
                label in (e["meta"].get("hate_types") or [])
                for _, es in full["context"]["example_groups"]
                for e in es
            )
            definition = any(
                d["id"] == "def-hate_type-" + label
                for d in full["context"]["definitions"]
            )
            same = full["messages_sha256"] == base["messages_sha256"]
            require(
                not (same and (ex or definition)),
                "Target knowledge in identical withdrawal context",
            )
            partitions[i] = (
                "identical_messages"
                if same
                else (
                    "example_and_definition"
                    if ex and definition
                    else (
                        "example_only"
                        if ex
                        else "definition_only" if definition else "neither_changed"
                    )
                )
            )
        label_runs = []
        for rep in range(3):
            bins = {p: Counter() for p in PARTS}
            total = Counter()
            for i in ids:
                a = rr["taxonomy", "rag_schema_" + label, rep][i]["result"]
                b = rr["taxonomy", "rag_full", rep][i]["result"]
                require(a is not None and b is not None, "Null on shared mask")
                gold = roster[i]["gold"]["hate_type"]
                y = label in gold
                pa, pb = label in a["hate_type"], label in b["hate_type"]
                ga, gb = a["hate"], b["hate"]
                c = Counter(
                    {
                        "n": 1,
                        "positive": int(y),
                        "negative": int(not y),
                        "base_gate_miss": int(not ga),
                        "full_gate_miss": int(not gb),
                    }
                )
                for arm, pred in [("base", a), ("full", b)]:
                    c.update(
                        {
                            arm + "_" + k: v
                            for k, v in bin_counts(gold, pred, label).items()
                        }
                    )
                    c[arm + "_positive_gate_miss"] += int(y and not pred["hate"])
                    c[arm + "_positive_type_miss_with_hate"] += int(
                        y and pred["hate"] and label not in pred["hate_type"]
                    )
                if y:
                    c["tp_gain_gate"] += int(pb and not pa and not ga)
                    c["tp_gain_type"] += int(pb and not pa and ga)
                    c["tp_loss_gate"] += int(pa and not pb and not gb)
                    c["tp_loss_type"] += int(pa and not pb and gb)
                else:
                    c["fp_gain_gate"] += int(pb and not pa and not ga)
                    c["fp_gain_type"] += int(pb and not pa and ga)
                    c["fp_loss_gate"] += int(pa and not pb and not gb)
                    c["fp_loss_type"] += int(pa and not pb and gb)
                bins[partitions[i]].update(c)
                total.update(c)
            row = {
                "label": label,
                "repeat": rep,
                "counts": dict(total),
                "delivery_partitions": {p: dict(c) for p, c in bins.items()},
                "base_f1": f1(total["base_tp"], total["base_fp"], total["base_fn"]),
                "full_f1": f1(total["full_tp"], total["full_fp"], total["full_fn"]),
            }
            out["runs"].append(row)
            label_runs.append(row)
        out["labels"][label] = {
            "n": len(ids),
            "positive_support": label_runs[0]["counts"]["positive"],
            "mean_counts": {
                k: mean(r["counts"].get(k, 0) for r in label_runs)
                for k in label_runs[0]["counts"]
            },
            "delivery_partitions": {
                p: {
                    k: mean(r["delivery_partitions"][p].get(k, 0) for r in label_runs)
                    for k in label_runs[0]["counts"]
                }
                for p in PARTS
            },
            "base_f1": mean(r["base_f1"] for r in label_runs),
            "full_f1": mean(r["full_f1"] for r in label_runs),
        }
    out["primary_reproduction"] = {
        arm: mean(r[arm + "_f1"] for r in out["runs"]) for arm in ["base", "full"]
    }
    out["sum_of_label_mean_counts"] = {
        k: sum(v["mean_counts"][k] for v in out["labels"].values())
        for k in out["runs"][0]["counts"]
    }
    out["partition_sum_of_label_mean_counts"] = {
        p: {
            k: sum(v["delivery_partitions"][p][k] for v in out["labels"].values())
            for k in out["runs"][0]["counts"]
        }
        for p in PARTS
    }
    return out


def correction(rr, roster, choices, sources):
    ids = read(OLD / "explanations/all_transfer.json")["common_valid_ids"]
    bids = read(OLD / "explanations/benign.json")["common_valid_ids"]
    out = {"hateful_ids": ids, "benign_ids": bids, "runs": []}
    for seed in SEEDS:
        for rep in range(3):
            total = Counter()
            parts = {"source_labels": Counter(), "other_labels": Counter()}
            for i in ids:
                a = rr["explanations", "reference", rep][i]["result"]
                b = rr["explanations", f"plain_always_s{seed}", rep][i]["result"]
                gold = set(roster[i]["gold"]["hate_type"])
                aa = set(a["hate_type"])
                bb = set(b["hate_type"])
                choice = choices[seed, i]
                sg = set(sources[choice["source_id"]]["gold"]["hate_type"])
                total.update(
                    {
                        "n": 1,
                        "base_gate_miss": int(not a["hate"]),
                        "plain_gate_miss": int(not b["hate"]),
                    }
                )
                for label in LABELS:
                    y, x, z = label in gold, label in aa, label in bb
                    c = Counter(
                        {
                            "opportunities": 1,
                            "gold_positive": int(y),
                            "gold_negative": int(not y),
                            "added_true": int(y and z and not x),
                            "added_false": int(not y and z and not x),
                            "dropped_true": int(y and x and not z),
                            "dropped_false": int(not y and x and not z),
                            "base_positive": int(x),
                            "plain_positive": int(z),
                        }
                    )
                    parts["source_labels" if label in sg else "other_labels"].update(c)
                total["base_contains"] += gold <= aa
                total["plain_contains"] += gold <= bb
                total["base_exact"] += gold == aa
                total["plain_exact"] += gold == bb
            benign = Counter()
            for i in bids:
                a = rr["explanations", "reference", rep][i]["result"]["hate"]
                b = rr["explanations", f"plain_always_s{seed}", rep][i]["result"][
                    "hate"
                ]
                benign.update(
                    {
                        "n": 1,
                        "base_fp": int(a),
                        "plain_fp": int(b),
                        "new_fp": int(b and not a),
                        "removed_fp": int(a and not b),
                    }
                )
            out["runs"].append(
                {
                    "seed": seed,
                    "repeat": rep,
                    "hateful": dict(total),
                    "label_transitions": {k: dict(v) for k, v in parts.items()},
                    "benign": dict(benign),
                }
            )
    out["mean_hateful_counts"] = {
        k: mean(r["hateful"][k] for r in out["runs"]) for k in out["runs"][0]["hateful"]
    }
    out["mean_benign_counts"] = {
        k: mean(r["benign"][k] for r in out["runs"]) for k in out["runs"][0]["benign"]
    }
    out["mean_label_transitions"] = {
        p: {
            k: mean(r["label_transitions"][p][k] for r in out["runs"])
            for k in out["runs"][0]["label_transitions"][p]
        }
        for p in ["source_labels", "other_labels"]
    }
    return out


def annotation(rr, rosters):
    with (ROOT / "data/mappings/implicit_hate_types.csv").open(
        encoding="utf8", newline=""
    ) as f:
        mapping = {r["source_label"]: r["unified_label"] for r in csv.DictReader(f)}
    raw = ROOT / "data/raw/implicit_hate"
    with (raw / "implicit_hate_v1_stg2_posts.tsv").open(
        encoding="utf8", newline=""
    ) as f:
        stage2 = {
            r["post"]: [
                v for v in [r["implicit_class"], r["extra_implicit_class"]] if v.strip()
            ]
            for r in csv.DictReader(f, delimiter="\t")
        }

    def mapped(ts):
        return sorted(set(mapping[t] for t in ts if mapping[t] != "EXCLUDE"))

    records = []
    matched = set()
    with (raw / "implicit_hate_v1_stg1_posts.tsv").open(
        encoding="utf8", newline=""
    ) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            text = row["post"]
            cl = row["class"]
            labels = (
                ["explicit"]
                if cl == "explicit_hate"
                else mapped(stage2.get(text, [])) if cl == "implicit_hate" else None
            )
            if cl == "implicit_hate" and text in stage2:
                matched.add(text)
            records.append(
                {
                    "text": text,
                    "gold": {"hate": cl != "not_hate", "hate_type": labels or None},
                    "raw_class": cl,
                }
            )
    for text, ts in stage2.items():
        if text not in matched:
            records.append(
                {
                    "text": text,
                    "gold": {"hate": True, "hate_type": mapped(ts) or None},
                    "raw_class": "implicit_hate (stg2-only)",
                }
            )
    source_rows = {f"implicit_hate-{n}": r for n, r in enumerate(records)}
    out = {
        "reconstructed_source_records": len(records),
        "studies": {},
        "prediction_patterns": {},
    }
    for st, roster in rosters.items():
        ids = sorted(i for i, v in roster.items() if v["gold"]["hate"])
        for i in ids:
            require(
                roster[i]["text"] == source_rows[i]["text"],
                "Raw/source text mismatch " + i,
            )
            require(
                roster[i]["gold"]["hate"] == source_rows[i]["gold"]["hate"]
                and sorted(roster[i]["gold"]["hate_type"])
                == source_rows[i]["gold"]["hate_type"],
                "Raw/source gold mismatch " + i,
            )
        out["studies"][st] = {
            "verified_hateful_roster": len(ids),
            "gold_cardinality": dict(
                Counter(len(roster[i]["gold"]["hate_type"]) for i in ids)
            ),
            "raw_classes": dict(Counter(source_rows[i]["raw_class"] for i in ids)),
            "explicit_and_implicit_gold": sum(
                "explicit" in roster[i]["gold"]["hate_type"]
                and len(roster[i]["gold"]["hate_type"]) > 1
                for i in ids
            ),
        }
    for st, c in [
        ("taxonomy", "rag_full"),
        ("taxonomy", "zs_full"),
        ("explanations", "reference"),
        *[("explanations", f"plain_always_s{s}") for s in SEEDS],
    ]:
        ids = read(
            OLD
            / (
                "taxonomy/primary.json"
                if st == "taxonomy"
                else "explanations/all_transfer.json"
            )
        )["common_valid_ids"]
        runs = []
        for rep in range(3):
            co = Counter()
            matrix = Counter()
            for i in ids:
                prediction = rr[st, c, rep][i]["result"]
                if prediction is None:
                    co["null"] += 1
                    continue
                gold = set(rosters[st][i]["gold"]["hate_type"])
                pred = set(prediction["hate_type"])
                co["n"] += 1
                co["exact"] += gold == pred
                co["contains_with_extras"] += gold < pred
                co["explicit_and_implicit_predictions"] += (
                    "explicit" in pred and len(pred) > 1
                )
                co["false_type_labels"] += len(pred - gold)
                co["missed_type_labels"] += len(gold - pred)
                co["extra_implicit_on_explicit_gold"] += len(
                    (pred - {"explicit"}) if gold == {"explicit"} else set()
                )
                co["extra_explicit_on_implicit_gold"] += int(
                    "explicit" in pred and "explicit" not in gold
                )
                matrix[f"{len(gold)}gold_{len(pred)}pred"] += 1
            runs.append(
                {"repeat": rep, "counts": dict(co), "cardinality_matrix": dict(matrix)}
            )
        out["prediction_patterns"][st + "/" + c] = {
            "runs": runs,
            "mean_counts": {
                k: mean(r["counts"].get(k, 0) for r in runs) for k in runs[0]["counts"]
            },
        }
    return out


def analyse():
    p = read(DEST / "plan.json")
    verify(p["inputs"])
    verify(p["frozen_execution_bindings"])
    rr = defaultdict(dict)
    for r in rows(OLD / "review/predictions.jsonl"):
        rr[r["study"], r["condition"], r["repeat"]][r["id"]] = r
    rosters = {
        s: {r["id"]: r for r in rows(base / "roster.jsonl")}
        for s, base in STUDIES.items()
    }
    prompts = {
        (r["item_id"], r["condition"]): r
        for r in rows(STUDIES["taxonomy"] / "prompts.jsonl")
    }
    choices = {
        (r["pool_seed"], r["query_id"]): r
        for r in rows(STUDIES["explanations"] / "choices.jsonl")
    }
    sources = {
        r["id"]: r
        for r in rows(
            ROOT
            / "experiments/astra/sq3_full_kb_feedback_20260928_v1/feasibility_v1/roster.jsonl"
        )
    }
    outputs = {
        "taxonomy_decomposition.json": taxonomy(rr, rosters["taxonomy"], prompts),
        "correction_transitions.json": correction(
            rr, rosters["explanations"], choices, sources
        ),
        "annotation_mapping_review.json": annotation(rr, rosters),
    }
    for name, v in outputs.items():
        save(name, v)
    verify(p["inputs"])
    verify(p["frozen_execution_bindings"])
    save(
        "analysis_complete.json",
        {
            "status": "descriptive_analysis_complete",
            "plan_sha256": sha(DEST / "plan.json"),
            "outputs": {rel(DEST / n): sha(DEST / n) for n in outputs},
            "source_sha256": sha(Path(__file__)),
            "new_model_calls": 0,
        },
    )
    print(
        "Taxonomy counts:",
        outputs["taxonomy_decomposition.json"]["sum_of_label_mean_counts"],
        flush=True,
    )
    print(
        "Correction label transitions:",
        outputs["correction_transitions.json"]["mean_label_transitions"],
        flush=True,
    )
    print(
        "Annotation source verification:",
        outputs["annotation_mapping_review.json"]["studies"],
        flush=True,
    )


def check():
    p = read(DEST / "plan.json")
    verify(p["inputs"])
    verify(p["frozen_execution_bindings"])
    a = read(DEST / "analysis_complete.json")
    verify(a["outputs"])
    tax = read(DEST / "taxonomy_decomposition.json")
    anchors = read(OLD / "taxonomy/primary.json")["primary"]
    require(
        abs(tax["primary_reproduction"]["full"] - anchors["full_target_f1"]) < 1e-13,
        "Full F1 reproduction",
    )
    require(
        abs(tax["primary_reproduction"]["base"] - anchors["schema_only_target_f1"])
        < 1e-13,
        "Schema F1 reproduction",
    )
    checks = 0
    for r in tax["runs"]:
        for k, v in r["counts"].items():
            require(
                sum(c.get(k, 0) for c in r["delivery_partitions"].values()) == v,
                "Nonpartition " + k,
            )
            checks += 1
        for c in [r["counts"], *r["delivery_partitions"].values()]:
            for field in ["tp", "fp"]:
                require(
                    c.get("full_" + field, 0) - c.get("base_" + field, 0)
                    == c.get(field + "_gain_gate", 0)
                    + c.get(field + "_gain_type", 0)
                    - c.get(field + "_loss_gate", 0)
                    - c.get(field + "_loss_type", 0),
                    "Transition identity",
                )
                checks += 1
            for arm in ["base", "full"]:
                require(
                    c.get(arm + "_fn", 0)
                    == c.get(arm + "_positive_gate_miss", 0)
                    + c.get(arm + "_positive_type_miss_with_hate", 0),
                    "FN partition",
                )
                checks += 1
    corr = read(DEST / "correction_transitions.json")
    for r in corr["runs"]:
        for c in r["label_transitions"].values():
            require(
                c["plain_positive"] - c["base_positive"]
                == c["added_true"]
                + c["added_false"]
                - c["dropped_true"]
                - c["dropped_false"],
                "Correction transition identity",
            )
            checks += 1
        b = r["benign"]
        require(
            b["plain_fp"] - b["base_fp"] == b["new_fp"] - b["removed_fp"],
            "Benign gate identity",
        )
        checks += 1

    runs = defaultdict(dict)
    for r in rows(OLD / "review/predictions.jsonl"):
        if r["study"] == "taxonomy":
            runs[r["condition"], r["repeat"]][r["id"]] = r["result"]
    roster = {r["id"]: r for r in rows(STUDIES["taxonomy"] / "roster.jsonl")}
    universe = set(tax["ids"])
    for r in tax["runs"]:
        label = r["label"]
        positive = {i for i in universe if label in roster[i]["gold"]["hate_type"]}
        for arm, condition in [("base", "rag_schema_" + label), ("full", "rag_full")]:
            predicted = {
                i
                for i in universe
                if label in runs[condition, r["repeat"]][i]["hate_type"]
            }
            independent = {
                "tp": len(positive & predicted),
                "fp": len(predicted - positive),
                "fn": len(positive - predicted),
                "tn": len(universe - (positive | predicted)),
            }
            for k, v in independent.items():
                require(
                    v == r["counts"][arm + "_" + k], "Independent set metric mismatch"
                )
                checks += 1
    save(
        "verification.json",
        {
            "status": "passed",
            "arithmetic_and_set_checks": checks,
            "verified_original_primary": anchors,
            "inputs_rehashed": len(p["inputs"]),
            "frozen_execution_bindings_unchanged": len(p["frozen_execution_bindings"]),
            "analysis_complete_sha256": sha(DEST / "analysis_complete.json"),
            "new_model_calls": 0,
        },
    )
    print(
        {
            "status": "passed",
            "checks": checks,
            "original_taxonomy_difference": anchors["difference"],
        },
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["plan", "analyse", "check"])
    args = parser.parse_args()
    with patch.object(socket, "create_connection", forbidden), patch.object(
        socket.socket, "connect", forbidden
    ):
        globals()[args.mode]()

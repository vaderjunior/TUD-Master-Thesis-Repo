from collections import Counter
import hashlib
import json
import socket
from unittest.mock import patch
from analysis.feedback.supplements.archive import ROOT
from analysis.feedback.supplements.archive import DEST
from analysis.feedback.supplements.archive import REVIEW
from analysis.feedback.supplements.archive import STUDIES
from analysis.feedback.supplements.archive import read
from analysis.feedback.supplements.archive import rows
from analysis.feedback.supplements.archive import sha
from analysis.feedback.supplements.archive import rel
from analysis.feedback.supplements.archive import save
from analysis.feedback.supplements.archive import verify
from analysis.feedback.supplements.archive import require
from analysis.feedback.supplements.archive import utc
from analysis.feedback.supplements.archive import forbid
from analysis.feedback.balanced import selection_pools as frozen


def ratio(n, d):
    return n / d if d else 0.0


def main():
    review = read(REVIEW / "complete.json")
    verify(review["inputs"])
    verify(review["outputs"])
    base = STUDIES["benign_selection"]
    roster = list(rows(base / "roster.jsonl"))
    byid = {r["id"]: r for r in roster}
    exported = [
        r
        for r in rows(REVIEW / "predictions.jsonl")
        if r["study"] == "benign_selection"
    ]
    require(
        len(exported) == 3072 and len({r["condition"] for r in exported}) == 1,
        "Source baseline matrix changed",
    )
    predictions = {(r["id"], r["repeat"]): r["result"] for r in exported}
    statuses = frozen.classify_candidates(roster, predictions)
    selection = frozen.select_pools(statuses)
    independently = {}
    for r in roster:
        answers = [predictions[r["id"], repeat] for repeat in range(3)]
        gate = [x["hate"] if x is not None else None for x in answers]
        if None in gate:
            out = {"status": "invalid", "hate": r["gold"]["hate"], "decisions": gate}
        else:
            vote = sorted(gate)[1]
            out = {
                "status": "correct" if vote == r["gold"]["hate"] else "error",
                "hate": r["gold"]["hate"],
                "decisions": gate,
                "majority_hate": vote,
            }
        independently[r["id"]] = out
    require(statuses == independently, "Independent source status mismatch")
    counts = {
        f"{g}_{st}": sum(
            r["hate"] == g and r["status"] == st for r in statuses.values()
        )
        for g in [False, True]
        for st in ["correct", "error"]
    }
    capacity = min(counts.values())
    budget = 40 if capacity >= 40 else 20 if capacity >= 20 else None
    require(
        selection["capacity"] == capacity and selection["counts"] == counts,
        "Independent capacity mismatch",
    )
    require(selection.get("budget") == budget, "Predeclared budget differs")
    independent_pools = []
    if budget is not None:
        for seed in [2026100441, 2026100442, 2026100443]:
            for arm in ["error", "correct", "random"]:
                order = {}
                for g in [True, False]:
                    ids = [
                        i
                        for i, r in independently.items()
                        if r["hate"] == g
                        and r["status"] != "invalid"
                        and (arm == "random" or r["status"] == arm)
                    ]
                    order[g] = sorted(
                        ids,
                        key=lambda i: hashlib.sha256(
                            (str(seed) + "/" + arm + "/" + str(g) + "|" + i).encode()
                        ).hexdigest(),
                    )
                positive = order[True][:budget]
                balanced = positive[: budget // 2] + order[False][: budget // 2]
                for polarity, ids in [("positive", positive), ("balanced", balanced)]:
                    independent_pools.append(
                        {
                            "condition": f"{arm}_{polarity}_s{seed}",
                            "selection": arm,
                            "polarity": polarity,
                            "seed": seed,
                            "budget": budget,
                            "ids": ids,
                            "actual_status_counts": dict(
                                Counter(independently[i]["status"] for i in ids)
                            ),
                        }
                    )
        require(
            selection["pools"] == independent_pools,
            "Actual source pools differ from independent stable-hash selection",
        )
    evaluated = {
        r["id"]
        for r in rows(
            ROOT / "experiments/astra/sq3_full_kb_taxonomy_20261001_v1/roster.jsonl"
        )
    }
    require(not set(byid) & evaluated, "Correction/evaluation IDs overlap")
    records = {
        i: frozen.gate_only_record(byid[i])
        for p in selection["pools"]
        for i in p["ids"]
    }
    for i, r in records.items():
        require(
            r["text"] == byid[i]["text"]
            and r["meta"]
            == {
                "gate": byid[i]["gold"]["hate"],
                "hate_types": None,
                "target_groups": None,
                "severity": None,
                "legal": None,
            },
            "Feedback invents a fine-head label",
        )
    metrics = []
    for repeat in range(3):
        valid = [
            (r, predictions[r["id"], repeat])
            for r in roster
            if predictions[r["id"], repeat] is not None
        ]
        tp = sum(r["gold"]["hate"] and p["hate"] for r, p in valid)
        fp = sum(not r["gold"]["hate"] and p["hate"] for r, p in valid)
        tn = sum(not r["gold"]["hate"] and not p["hate"] for r, p in valid)
        fn = sum(r["gold"]["hate"] and not p["hate"] for r, p in valid)
        metrics.append(
            {
                "repeat": repeat,
                "n_valid": len(valid),
                "nulls": len(roster) - len(valid),
                "tp": tp,
                "fp": fp,
                "tn": tn,
                "fn": fn,
                "false_positive_rate": ratio(fp, fp + tn),
                "false_negative_rate": ratio(fn, fn + tp),
                "binary_macro_f1": 0.5
                * (ratio(2 * tp, 2 * tp + fp + fn) + ratio(2 * tn, 2 * tn + fp + fn)),
            }
        )
    overlaps = []
    for n, p in enumerate(selection["pools"]):
        for q in selection["pools"][n + 1 :]:
            union = set(p["ids"]) | set(q["ids"])
            shared = set(p["ids"]) & set(q["ids"])
            overlaps.append(
                {
                    "left": p["condition"],
                    "right": q["condition"],
                    "shared": len(shared),
                    "jaccard": len(shared) / len(union),
                }
            )
    dest = DEST / "benign_selection"
    outputs = {}

    def put(name, value):
        path = dest / name
        save(path, value)
        outputs[rel(path)] = sha(path)

    put("candidate_statuses.json", statuses)
    put("actual_pools.json", selection)
    put("gate_only_records.json", records)
    summary = {
        "status": selection["status"],
        "roster_size": len(roster),
        "by_gold_gate": dict(Counter(str(r["gold"]["hate"]) for r in roster)),
        "eligible_status_counts": counts,
        "excluded_invalid": sum(r["status"] == "invalid" for r in statuses.values()),
        "majority_unanimous": sum(
            r["status"] != "invalid" and len(set(r["decisions"])) == 1
            for r in statuses.values()
        ),
        "majority_two_one": sum(
            r["status"] != "invalid" and len(set(r["decisions"])) == 2
            for r in statuses.values()
        ),
        "capacity": capacity,
        "budget": budget,
        "pool_count": len(selection["pools"]),
        "distinct_insertable_sources": len(records),
        "baseline_runs": metrics,
        "pool_overlaps": overlaps,
        "labels_inspected": len(roster),
        "annotation_scope": "Binary hate gate only; all other heads unannotated in added records.",
        "independently_verified_statuses": len(independently),
        "independently_verified_pools": len(independent_pools),
        "update_stage_not_prepared_or_executed": True,
        "planned_update_predictions": 38475 if budget else 0,
        "planned_update_ordinary_calls": 115425 if budget else 0,
        "new_classifier_calls": 0,
        "interpretation": "Candidate-selection baseline and feasibility only; not evidence that balanced corrections improve evaluation performance.",
    }
    put("summary.json", summary)
    inputs = {
        rel(REVIEW / "complete.json"): sha(REVIEW / "complete.json"),
        **review["outputs"],
        rel(base / "roster.jsonl"): sha(base / "roster.jsonl"),
        rel(base / "STUDY_PROTOCOL.md"): sha(base / "STUDY_PROTOCOL.md"),
        rel(frozen.__file__): sha(frozen.__file__),
        rel(__file__): sha(__file__),
    }
    save(
        dest / "complete.json",
        {
            "status": "actual_pools_frozen_and_independently_verified",
            "created_utc": utc(),
            "inputs": inputs,
            "outputs": outputs,
            "new_classifier_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in [
                    "status",
                    "eligible_status_counts",
                    "excluded_invalid",
                    "capacity",
                    "budget",
                    "pool_count",
                    "distinct_insertable_sources",
                ]
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket, "create_connection", forbid
    ):
        main()

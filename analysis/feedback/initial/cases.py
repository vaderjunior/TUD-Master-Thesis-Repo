from collections import defaultdict
import hashlib
from analysis.feedback.initial.archive import ROOT
from analysis.feedback.initial.archive import DEST
from analysis.feedback.initial.archive import REVIEW
from analysis.feedback.initial.archive import STUDIES
from analysis.feedback.initial.archive import LABELS
from analysis.feedback.initial.archive import SEEDS
from analysis.feedback.initial.archive import read
from analysis.feedback.initial.archive import rows
from analysis.feedback.initial.archive import sha
from analysis.feedback.initial.archive import save
from analysis.feedback.initial.archive import rel
from analysis.feedback.initial.archive import require
from analysis.feedback.initial.archive import utc


def pick(candidates):
    return sorted(
        candidates,
        key=lambda v: hashlib.sha256(
            (
                "20261004-sq3-illustrations|"
                + v["id"]
                + "|"
                + str(v.get("seed"))
                + "|"
                + str(v.get("label"))
            ).encode()
        ).hexdigest(),
    )[:2]


def main():
    require(
        read(DEST / "independent_verification.json")["status"] == "passed",
        "Verified results required",
    )
    runs = defaultdict(dict)
    for r in rows(REVIEW / "predictions.jsonl"):
        runs[r["study"], r["condition"], r["repeat"]][r["id"]] = r
    rosters = {
        s: {r["id"]: r for r in rows(p / "roster.jsonl")} for s, p in STUDIES.items()
    }
    choices = {
        (v["pool_seed"], v["query_id"]): v
        for v in rows(STUDIES["explanations"] / "choices.jsonl")
    }
    cards = {
        v["source_id"]: v
        for v in rows(
            ROOT
            / "experiments/astra/sq3_explanation_authoring_20261001_v1/explanation_cards.jsonl"
        )
    }
    source = {
        v["id"]: v
        for v in rows(
            ROOT
            / "experiments/astra/sq3_full_kb_feedback_20260928_v1/feasibility_v1/roster.jsonl"
        )
    }
    candidates = defaultdict(list)
    for i, roster in rosters["explanations"].items():
        if not roster["gold"]["hate"]:
            continue
        for seed in SEEDS:
            c = choices[seed, i]
            if not c["explanation_available"]:
                continue
            a = [
                runs["explanations", f"specific_always_s{seed}", r][i] for r in range(3)
            ]
            b = [runs["explanations", f"plain_always_s{seed}", r][i] for r in range(3)]
            if any(v["result"] is None for v in a + b):
                continue
            y = set(roster["gold"]["hate_type"])
            delta = sum(set(v["result"]["hate_type"]) == y for v in a) - sum(
                set(v["result"]["hate_type"]) == y for v in b
            )
            if abs(delta) < 2:
                continue
            name = (
                "explanation_improved_exact"
                if delta > 0
                else "explanation_worsened_exact"
            )
            candidates[name].append(
                {
                    "id": i,
                    "seed": seed,
                    "text": roster["text"],
                    "gold": roster["gold"],
                    "net_exact_decisions_changed_of_three": delta,
                    "source_id": c["source_id"],
                    "source_text": source[c["source_id"]]["text"],
                    "source_gold": source[c["source_id"]]["gold"],
                    "specific_explanation": cards[c["source_id"]][
                        "specific_explanation"
                    ],
                    "with_explanation": [v["result"] for v in a],
                    "plain": [v["result"] for v in b],
                    "result_files": [v["source"] for v in a + b],
                }
            )
    for i, roster in rosters["taxonomy"].items():
        if not roster["gold"]["hate"]:
            continue
        for label in LABELS:
            a = [runs["taxonomy", "rag_full", r][i] for r in range(3)]
            b = [runs["taxonomy", f"rag_schema_{label}", r][i] for r in range(3)]
            if (
                any(v["result"] is None for v in a + b)
                or a[0]["messages_sha256"] == b[0]["messages_sha256"]
            ):
                continue
            y = label in roster["gold"]["hate_type"]
            delta = sum((label in v["result"]["hate_type"]) == y for v in a) - sum(
                (label in v["result"]["hate_type"]) == y for v in b
            )
            if abs(delta) < 2:
                continue
            name = (
                "taxonomy_"
                + label
                + ("_improved_target" if delta > 0 else "_worsened_target")
            )
            candidates[name].append(
                {
                    "id": i,
                    "label": label,
                    "text": roster["text"],
                    "gold": roster["gold"],
                    "net_correct_target_decisions_changed_of_three": delta,
                    "full_package": [v["result"] for v in a],
                    "schema_only": [v["result"] for v in b],
                    "result_files": [v["source"] for v in a + b],
                }
            )
    selected = {k: pick(v) for k, v in candidates.items()}
    save(
        DEST / "illustrative_cases.json",
        {
            "created_utc": utc(),
            "role": "outcome_selected_descriptive_illustrations_not_independent_human_gold_review",
            "selection": "Among complete three-repeat pairs, select net correctness gain/loss of at least two decisions; taxonomy messages must differ, explanations must be available. At most two per direction, ranked by fixed SHA256 salt. No minimum is relaxed and no gold label changes.",
            "eligible_counts": {k: len(v) for k, v in candidates.items()},
            "cases": selected,
            "source_sha256": sha(__file__),
            "verified_scoring_sha256": sha(DEST / "independent_verification.json"),
            "new_model_calls": 0,
        },
    )
    print(
        "Saved",
        sum(len(v) for v in selected.values()),
        "deterministically selected illustrative cases.",
        flush=True,
    )


if __name__ == "__main__":
    main()

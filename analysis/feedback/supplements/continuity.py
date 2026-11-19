import statistics
import socket
from unittest.mock import patch
from analysis.feedback.supplements.archive import ROOT
from analysis.feedback.supplements.archive import DEST
from analysis.feedback.supplements.archive import REVIEW
from analysis.feedback.supplements.archive import STUDIES
from analysis.feedback.supplements.archive import LABELS
from analysis.feedback.supplements.archive import read
from analysis.feedback.supplements.archive import rows
from analysis.feedback.supplements.archive import sha
from analysis.feedback.supplements.archive import rel
from analysis.feedback.supplements.archive import save
from analysis.feedback.supplements.archive import require
from analysis.feedback.supplements.archive import utc
from analysis.feedback.supplements.archive import forbid


def main():
    previous = ROOT / "experiments/astra/sq3_combined_results_20261004_v1/review"
    old_path = previous / "predictions.jsonl"
    old_receipt = read(previous / "complete.json")
    require(
        old_receipt["outputs"][rel(old_path)] == sha(old_path),
        "Historical audited export changed",
    )
    roster = {r["key"]: r for r in rows(STUDIES["factorial"] / "roster.jsonl")}
    prompts = {
        (r["key"], r["condition"]): r
        for r in rows(STUDIES["factorial"] / "prompts.jsonl")
    }
    old = {
        (r["key"], r["condition"], r["repeat"]): r
        for r in rows(old_path)
        if r["study"] == "taxonomy"
    }
    new = {
        (r["key"], r["condition"], r["repeat"]): r
        for r in rows(REVIEW / "predictions.jsonl")
        if r["study"] == "factorial"
    }
    ids = read(DEST / "factorial/primary.json")["common_ids"]
    checks = 0
    matched = []
    excluded = []
    for i in ids:
        r = roster[i]
        okay = True
        for condition in ["E0D0", "E1D1"]:
            previous_condition = (
                "rag_full" if condition == "E1D1" else "rag_schema_" + r["target_label"]
            )
            for repeat in range(3):
                before = old[r["original_key"], previous_condition, repeat]
                require(
                    before["messages_sha256"]
                    == prompts[i, condition]["messages_sha256"],
                    "Historical endpoint message differs",
                )
                require(before["gold"] == r["gold"], "Historical gold differs")
                checks += 1
                if (
                    before["result"] is None
                    or new[i, condition, repeat]["result"] is None
                ):
                    okay = False
        (matched if okay else excluded).append(i)

    def score(phase, condition, repeat):
        by_label = {}
        for label in LABELS:
            eligible = [i for i in matched if roster[i]["target_label"] == label]
            tp = gold_count = pred_count = 0
            for i in eligible:
                r = roster[i]
                if phase == "old":
                    c = "rag_full" if condition == "E1D1" else "rag_schema_" + label
                    result = old[r["original_key"], c, repeat]["result"]
                else:
                    result = new[i, condition, repeat]["result"]
                gold = label in r["gold"]["hate_type"]
                pred = label in result["hate_type"]
                tp += gold and pred
                gold_count += gold
                pred_count += pred
            by_label[label] = {
                "tp": tp,
                "gold_support": gold_count,
                "predicted_support": pred_count,
                "f1": (
                    2 * tp / (gold_count + pred_count)
                    if gold_count + pred_count
                    else 0.0
                ),
            }
        return {
            "macro_target_f1": statistics.mean(v["f1"] for v in by_label.values()),
            "per_category": by_label,
        }

    runs = {
        f"{phase}_{condition}_rep{r}": score(phase, condition, r)
        for phase in ["old", "new"]
        for condition in ["E0D0", "E1D1"]
        for r in range(3)
    }
    means = {
        f"{phase}_{condition}": statistics.mean(
            runs[f"{phase}_{condition}_rep{r}"]["macro_target_f1"] for r in range(3)
        )
        for phase in ["old", "new"]
        for condition in ["E0D0", "E1D1"]
    }
    output = {
        "status": "descriptive_same_subset_continuity",
        "created_utc": utc(),
        "scope": "Post-hoc check after new primary results; selected both-blocks-changed population. Identical messages, separate fresh responses. No significance or equivalence claim.",
        "original_n": len(ids),
        "common_n": len(matched),
        "common_ids": matched,
        "excluded_ids": excluded,
        "endpoint_message_gold_checks": checks,
        "runs": runs,
        "means": means,
        "interpretation": "The old full-KB score on its broader population is not directly comparable to the selected-subset new full-KB score. These matches check continuity, not an additional improvement over the full KB.",
        "inputs": {
            rel(old_path): sha(old_path),
            rel(previous / "complete.json"): sha(previous / "complete.json"),
            rel(REVIEW / "predictions.jsonl"): sha(REVIEW / "predictions.jsonl"),
            rel(STUDIES["factorial"] / "roster.jsonl"): sha(
                STUDIES["factorial"] / "roster.jsonl"
            ),
            rel(STUDIES["factorial"] / "prompts.jsonl"): sha(
                STUDIES["factorial"] / "prompts.jsonl"
            ),
            rel(__file__): sha(__file__),
        },
        "new_classifier_calls": 0,
    }
    save(DEST / "diagnostics/historical_continuity.json", output)
    print({"common_n": len(matched), "excluded_ids": excluded, "means": means})


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", forbid), patch.object(
        socket, "create_connection", forbid
    ):
        main()

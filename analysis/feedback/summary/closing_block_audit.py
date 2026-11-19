from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from analysis.feedback.summary.closing_review import ROOT
from analysis.feedback.summary.closing_review import OLD
from analysis.feedback.summary.closing_review import DEST
from analysis.feedback.summary.closing_review import STUDIES
from analysis.feedback.summary.closing_review import LABELS
from analysis.feedback.summary.closing_review import read
from analysis.feedback.summary.closing_review import rows
from analysis.feedback.summary.closing_review import sha
from analysis.feedback.summary.closing_review import rel
from analysis.feedback.summary.closing_review import save
from analysis.feedback.summary.closing_review import verify
from analysis.feedback.summary.closing_review import require
from analysis.feedback.summary.validate_closing_feasibility import blocks


def main():
    require(
        read(DEST / "feasibility_verification.json")["status"] == "passed",
        "Visible block check required",
    )
    files = [
        Path(__file__),
        ROOT / "analysis/feedback/summary/validate_closing_feasibility.py",
        DEST / "feasibility_verification.json",
        OLD / "review/predictions.jsonl",
        STUDIES["taxonomy"] / "prompts.jsonl",
        STUDIES["taxonomy"] / "roster.jsonl",
        OLD / "taxonomy/primary.json",
    ]
    inputs = {rel(p): sha(p) for p in files}
    save(
        "block_audit_plan.json",
        {
            "inputs": inputs,
            "scope": "Post-result description by E0D0/E1D0/E0D1/E1D1 visible prompt changes; original primary mask and all labels/repeats. Reconcile additive target counts. Count future missing-factorial prompts using no outcomes. No new inferential primary or model call.",
        },
    )
    prompts = {
        (r["item_id"], r["condition"]): r
        for r in rows(STUDIES["taxonomy"] / "prompts.jsonl")
    }
    roster = {r["id"]: r for r in rows(STUDIES["taxonomy"] / "roster.jsonl")}
    visible = {
        k: blocks(p)
        for k, p in prompts.items()
        if k[1] == "rag_full" or k[1].startswith("rag_schema_")
    }
    answers = defaultdict(dict)
    for r in rows(OLD / "review/predictions.jsonl"):
        if r["study"] == "taxonomy":
            answers[r["condition"], r["repeat"]][r["id"]] = r["result"]
    keep = read(OLD / "taxonomy/primary.json")["common_valid_ids"]
    runs = []
    double = []
    categories = {}
    double_counts = defaultdict(Counter)
    for label in LABELS:
        for i, r in roster.items():
            f = visible[i, "rag_full"]
            s = visible[i, "rag_schema_" + label]
            e = f["examples"] != s["examples"]
            d = f["definitions"] != s["definitions"]
            require(
                f["guidelines"] == s["guidelines"],
                "Guideline changes prevent two-factor description",
            )
            categories[i, label] = f"E{int(e)}D{int(d)}"
            if e and d:
                double.append(
                    {
                        "id": i,
                        "label": label,
                        "population": r["population"],
                        "target_positive": label in (r["gold"]["hate_type"] or []),
                    }
                )
                double_counts[label][r["population"]] += 1
                double_counts[label]["target_positive"] += label in (
                    r["gold"]["hate_type"] or []
                )
        for rep in range(3):
            parts = {p: Counter() for p in ["E0D0", "E1D0", "E0D1", "E1D1"]}
            for i in keep:
                a = answers["rag_schema_" + label, rep][i]
                b = answers["rag_full", rep][i]
                y = label in roster[i]["gold"]["hate_type"]
                x = label in a["hate_type"]
                z = label in b["hate_type"]
                c = Counter(
                    {
                        "n": 1,
                        "positive": int(y),
                        "base_tp": int(y and x),
                        "full_tp": int(y and z),
                        "base_fp": int(not y and x),
                        "full_fp": int(not y and z),
                    }
                )
                parts[categories[i, label]].update(c)
            runs.append(
                {
                    "label": label,
                    "repeat": rep,
                    "parts": {p: dict(c) for p, c in parts.items()},
                }
            )
    summed = {
        p: {
            k: sum(r["parts"][p].get(k, 0) for r in runs) / 3
            for k in ["n", "positive", "base_tp", "full_tp", "base_fp", "full_fp"]
        }
        for p in ["E0D0", "E1D0", "E0D1", "E1D1"]
    }
    old = read(DEST / "taxonomy_decomposition.json")["sum_of_label_mean_counts"]
    for k in ["n", "positive", "base_tp", "full_tp", "base_fp", "full_fp"]:
        require(
            abs(sum(c[k] for c in summed.values()) - old[k]) < 1e-10,
            "Block decomposition does not sum " + k,
        )
    output = {
        "runs": runs,
        "sum_of_label_repeat_means": summed,
        "both_changed_pair_inventory": double,
        "both_changed_by_label": double_counts,
        "both_changed_pairs": len(double),
        "both_changed_distinct_inputs": len({r["id"] for r in double}),
        "ordinary_calls_for_two_missing_cells_only": len(double) * 2 * 3 * 3,
        "ordinary_calls_for_fresh_four_cells_on_both_changed_pairs": len(double)
        * 4
        * 3
        * 3,
        "limits": [
            "Input/category pairs are not independent texts. Existing one-block contrasts have input-specific scope; comparing effects between strata is not randomized examples-versus-definitions comparison.",
            "New missing-cell calls alone mix old and new generation phases; prefer fresh matched four-cell contrasts on the declared both-changed subset for an interaction study, with separate reporting of historical one-factor strata.",
            "Even swapping a whole example block changes non-target examples, lengths and co-label exposure. The interpretation is block-level delivery, not a pure label-token causal effect.",
            "This inventory is not a frozen runnable extension or approval to choose favorable categories. Keep all seven categories and every prespecified subset regardless of results.",
        ],
        "inputs": inputs,
        "new_model_calls": 0,
    }
    verify(inputs)
    save("rendered_block_audit.json", output)
    print(
        {
            "parts": summed,
            "both_changed_pairs": len(double),
            "distinct_inputs": output["both_changed_distinct_inputs"],
            "by_label": dict(double_counts),
            "fresh_four_cell_calls": output[
                "ordinary_calls_for_fresh_four_cells_on_both_changed_pairs"
            ],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()

from collections import defaultdict
import numpy as np
from sklearn.metrics import f1_score
from analysis.feedback.initial.archive import ROOT
from analysis.feedback.initial.archive import DEST
from analysis.feedback.initial.archive import REVIEW
from analysis.feedback.initial.archive import STUDIES
from analysis.feedback.initial.archive import LABELS
from analysis.feedback.initial.archive import SEEDS
from analysis.feedback.initial.archive import read
from analysis.feedback.initial.archive import rows
from analysis.feedback.initial.archive import save
from analysis.feedback.initial.archive import sha
from analysis.feedback.initial.archive import require
from analysis.feedback.initial.archive import utc


def main():
    require(
        read(DEST / "independent_verification.json")["status"] == "passed",
        "Checked results required",
    )
    rr = defaultdict(dict)
    for v in rows(REVIEW / "predictions.jsonl"):
        rr[v["study"], v["condition"], v["repeat"]][v["id"]] = v
    roster = {
        s: {r["id"]: r for r in rows(p / "roster.jsonl")} for s, p in STUDIES.items()
    }

    def per(study, conditions, ids):
        scores = []
        for c in conditions:
            for r in range(3):
                y = np.array(
                    [
                        [l in roster[study][i]["gold"]["hate_type"] for l in LABELS]
                        for i in ids
                    ]
                )
                x = np.array(
                    [
                        [l in rr[study, c, r][i]["result"]["hate_type"] for l in LABELS]
                        for i in ids
                    ]
                )
                scores.append(f1_score(y, x, average=None, zero_division=0))
        return np.mean(scores, axis=0)

    sensitivity = {}
    ids = read(STUDIES["explanations"] / "scientific_plan_v1.json")["rosters"][
        "transfer"
    ]
    for name, left, right in [
        ("explanation", "specific_always", "plain_always"),
        ("application", "plain_selective", "plain_always"),
    ]:
        lc = [f"{left}_s{s}" for s in SEEDS]
        rc = [f"{right}_s{s}" for s in SEEDS]
        keep = sorted(
            i
            for i in ids
            if all(
                rr["explanations", c, r][i]["n_valid"] == 3
                for c in lc + rc
                for r in range(3)
            )
        )
        a = per("explanations", lc, keep)
        b = per("explanations", rc, keep)
        sensitivity[name] = {
            "original_n": len(ids),
            "all_three_votes_valid_common_n": len(keep),
            "excluded_ids": sorted(set(ids) - set(keep)),
            "left": float(a.mean()),
            "right": float(b.mean()),
            "difference": float((a - b).mean()),
        }
    ids = [i for i, v in roster["taxonomy"].items() if v["gold"]["hate"]]
    conditions = ["rag_full", *["rag_schema_" + l for l in LABELS]]
    keep = sorted(
        i
        for i in ids
        if all(
            rr["taxonomy", c, r][i]["n_valid"] == 3
            for c in conditions
            for r in range(3)
        )
    )
    full = per("taxonomy", ["rag_full"], keep)
    schema = np.array(
        [per("taxonomy", ["rag_schema_" + l], keep)[n] for n, l in enumerate(LABELS)]
    )
    sensitivity["taxonomy"] = {
        "original_n": len(ids),
        "all_three_votes_valid_common_n": len(keep),
        "excluded_ids": sorted(set(ids) - set(keep)),
        "full": float(full.mean()),
        "schema_only": float(schema.mean()),
        "difference": float((full - schema).mean()),
    }
    prompts = {
        (v["item_id"], v["condition"]): v
        for v in rows(STUDIES["taxonomy"] / "prompts.jsonl")
    }
    delivery = {}
    for l in LABELS:
        d = {}
        for positive in [True, False]:
            ii = [
                i
                for i in ids
                if (l in roster["taxonomy"][i]["gold"]["hate_type"]) == positive
            ]
            component = changed = 0
            for i in ii:
                p = prompts[i, "rag_full"]
                examples = [
                    e for _, es in p["context"].get("example_groups", []) for e in es
                ]
                ex = any(l in (e["meta"].get("hate_types") or []) for e in examples)
                definition = any(
                    e["id"] == "def-hate_type-" + l
                    for e in p["context"].get("definitions", [])
                )
                component += ex or definition
                changed += (
                    p["messages_sha256"]
                    != prompts[i, "rag_schema_" + l]["messages_sha256"]
                )
            d["target_positive" if positive else "other_type_negative"] = {
                "inputs": len(ii),
                "target_example_or_definition_delivered": component,
                "messages_changed": changed,
            }
        delivery[l] = d
    save(
        DEST / "deeper_sensitivity.json",
        {
            "created_utc": utc(),
            "status": "post_run_descriptive_checks_complete",
            "three_valid_vote_sensitivity": sensitivity,
            "taxonomy_delivery_by_gold_target": delivery,
            "limits": [
                "Post-run diagnostics, not newly declared primary tests or parameter selection.",
                "Requiring all votes valid selects a different population; no new superiority claim follows. Primary null bounds and original intervals remain authoritative.",
                "Gold-stratified delivery is analysis only and never changes retrieval. Component availability is associated with input content and does not separately identify causal example/definition effects.",
            ],
            "source_sha256": sha(__file__),
            "independent_verification_sha256": sha(
                DEST / "independent_verification.json"
            ),
            "new_model_calls": 0,
        },
    )
    print("Three-valid-vote sensitivity:", sensitivity, flush=True)
    print("Gold-stratified taxonomy delivery:", delivery, flush=True)


if __name__ == "__main__":
    main()

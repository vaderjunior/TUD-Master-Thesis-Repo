from collections import Counter
from pathlib import Path
import hashlib
import json
import socket
from unittest.mock import patch
from analysis.feedback.common import offline as h

ROOT = h.ROOT
OLD = h.OLD
OUT = ROOT / "experiments/astra/sq3_full_kb_relevance_20260930_v1"
POLICIES = [
    "natural",
    "relevant_full",
    "relevant_labels",
    "disjoint_full",
    "disjoint_labels",
]
SELECTION_SEED = "20260930-sq3-relevance-v1"


def pick(ids, pool_seed, query_id, purpose):
    h.require(ids and len(ids) == len(set(ids)), "Nonempty unique choice set required")
    return min(
        ids,
        key=lambda iid: hashlib.sha256(
            f"{SELECTION_SEED}|{pool_seed}|{query_id}|{purpose}|{iid}".encode()
        ).hexdigest(),
    )


def prepare():
    h.require(not OUT.exists(), "Preserve existing relevance plan")
    phase = h.OUT
    done = h.read(phase / "complete.json")
    checked = h.read(phase / "independent_verification.json")
    h.require(
        checked["status"] == "passed"
        and checked["completion_sha256"] == h.digest(phase / "complete.json"),
        "Offline review not verified",
    )
    for name, sha in done["outputs"].items():
        h.require(h.digest(ROOT / name) == sha, "Offline output changed")
    old_done = h.read(OLD / "update_payloads_v1/complete.json")
    for name, sha in old_done["outputs"].items():
        h.require(h.digest(ROOT / name) == sha, "Original payload output changed")
    science = h.read(OLD / "scientific_plan_v1.json")
    roster = {row["id"]: row for row in h.rows(OLD / "feasibility_v1/roster.jsonl")}
    transfer = science["rosters"]["transfer"]
    pools = [
        p
        for p in h.read(OLD / "update_payloads_v1/pools.json")["pools"]
        if p["size"] == 40 and p["arm"] == "error"
    ]
    retrieval = h.read(OLD / "results_v1/retrieval_diagnostics.json")["per_item"]
    h.require(len(pools) == 3 and len(transfer) == 350, "Source design differs")
    h.require(
        pick(["a", "b", "c"], 1, "q", "r") == pick(["c", "b", "a"], 1, "q", "r"),
        "Choice depends on source order",
    )
    h.require(pick(["only"], 1, "q", "r") == "only", "Singleton choice differs")
    choices = []
    covered = {iid: True for iid in transfer}
    for pool in pools:
        for iid in transfer:
            gold = set(roster[iid]["gold"]["hate_type"])
            cap = h.coverage(gold, pool["ids"], roster)
            relevant_candidates = cap["full_cover_ids"]
            fallback = not bool(relevant_candidates)
            if fallback:
                best = max(
                    len(gold & set(roster[j]["gold"]["hate_type"])) for j in pool["ids"]
                )
                h.require(best > 0, "No partial-overlap fallback available")
                relevant_candidates = [
                    j
                    for j in pool["ids"]
                    if len(gold & set(roster[j]["gold"]["hate_type"])) == best
                ]
            relevant = pick(relevant_candidates, pool["seed"], iid, "relevant")
            disjoint = pick(cap["disjoint_ids"], pool["seed"], iid, "disjoint")
            natural = retrieval[pool["condition"]][iid]["source_id"]
            h.require(
                natural in pool["ids"] and iid not in pool["ids"],
                "Natural/source eligibility differs",
            )
            covered[iid] &= not fallback
            choices.append(
                {
                    "query_id": iid,
                    "pool_seed": pool["seed"],
                    "old_pool_condition": pool["condition"],
                    "natural_source_id": natural,
                    "relevant_source_id": relevant,
                    "disjoint_source_id": disjoint,
                    "full_cover_available": not fallback,
                    "fallback": "maximum_gold_label_overlap" if fallback else None,
                    "full_cover_candidate_count": len(cap["full_cover_ids"]),
                    "disjoint_candidate_count": len(cap["disjoint_ids"]),
                }
            )
    primary = [iid for iid in transfer if covered[iid]]
    secondary_only = [iid for iid in transfer if not covered[iid]]
    expected = h.read(phase / "summary.json")[
        "oracle_common_full_and_disjoint_error_pool_ids"
    ]
    h.require(
        primary == expected and len(primary) == 343 and len(secondary_only) == 7,
        "Coverage mask differs",
    )
    sentinels = h.read(OLD / "baseline_review_v1/selection.json")["sentinel_ids"]
    cohort = sorted(
        transfer,
        key=lambda iid: hashlib.sha256(
            ("20260930-relevance-batches|" + iid).encode()
        ).hexdigest(),
    )
    batches = [cohort[:117], cohort[117:234], cohort[234:]]
    paths = [
        Path(__file__),
        Path(h.__file__),
        phase / "plan.json",
        phase / "complete.json",
        phase / "summary.json",
        phase / "independent_verification.json",
        phase / "case_review_assistant_v1.json",
        OLD / "scientific_plan_v1.json",
        OLD / "feasibility_v1/roster.jsonl",
        OLD / "feasibility_v1/baseline_prompts.jsonl",
        OLD / "update_payloads_v1/complete.json",
        OLD / "update_payloads_v1/pools.json",
        OLD / "update_payloads_v1/added_records.jsonl",
        OLD / "update_payloads_v1/prompt_inventory.jsonl",
        OLD / "update_payloads_v1/prompts.jsonl",
        OLD / "results_v1/retrieval_diagnostics.json",
        OLD / "baseline_review_v1/selection.json",
        OLD / "update_api_v1/plan.json",
        ROOT / "config/taxonomy.yaml",
        ROOT / "prompts/classify_v1.txt",
        ROOT / "src/hsrag/data/implicit_hate.py",
        ROOT / "data/mappings/implicit_hate_types.csv",
        ROOT
        / "Thesis_Log/astra_result/astra_sq3_followups_accepted_plan_20260930_v1.md",
    ]
    inputs = {h.rel(p): h.digest(p) for p in paths}
    plan = {
        "status": "science_and_choices_frozen_payloads_and_execution_not_ready",
        "created_utc": h.now(),
        "question": "Does deliberately supplying a label-relevant correction outperform natural correction retrieval under the frozen full-KB context?",
        "authorization": "Ajay approved all four SQ3 follow-ups on 30 September2026; long runs remain user-run.",
        "history": "Exploratory extension informed by completed small/full feedback results and an offline case review. Existing DEV is previously inspected.",
        "inputs": inputs,
        "labels": h.LABELS,
        "pool_seeds": h.SEEDS,
        "pool_size": 40,
        "rosters": {
            "evaluation": transfer,
            "primary_full_cover": primary,
            "secondary_only_partial_cover": secondary_only,
            "historical_sentinels": sentinels,
            "holdout_from_original_selection": science["rosters"]["holdout"],
        },
        "conditions": {
            "per_pool": POLICIES,
            "shared_reference": "reference",
            "decoder_repeats": 3,
            "votes_per_prediction": 3,
            "all_outputs_fresh": True,
            "old_outputs_role": "Historical/descriptive only, not extra repeats or live-reference substitution.",
        },
        "selection": {
            "seed": SELECTION_SEED,
            "rule": "Minimum SHA256 of seed|pool_seed|query_id|purpose|source_id within the eligible set; fixed across decoder repeats.",
            "relevant": "All query gold types contained in source labels. If none, maximize count of overlapping gold labels, then the same hash rule; fallback only secondary.",
            "disjoint": "No shared query/source gold types, then hash rule.",
            "natural": "Previously frozen hybrid selection from the same40 error-example pool; original selected source reused, response is fresh.",
            "labels_variants": "Reuse the exact relevant_full/disjoint_full source and metadata; replace only its text with the fixed placeholder.",
            "choices_file": "choices.jsonl",
            "gold_access": "Only the declared diagnostic selections use query gold. No production-retrieval claim.",
        },
        "prompt": {
            "base": "Exact saved full-KB context: five ordinary examples, original two definitions and two guidelines; unchanged schema/instructions.",
            "full": "One additional labelled example in the existing neutral Additional labelled examples group.",
            "labels_only_placeholder": "[Example text withheld for this diagnostic.]",
            "labels_only_meta": "Same source hate=True/types; targets/severity/legal remain unannotated.",
            "reference": "Unchanged five-example base with no added group.",
            "interpretation": "Full versus labels-only changes example text and token length. It does not alone isolate semantic understanding.",
        },
        "primary": {
            "contrast": "relevant_full minus natural",
            "endpoint": "Seven-type macro-F1 on the343 full-cover texts.",
            "average": "Mean of three separate decoder-repeat F1s within pool, then equal mean of three pool means.",
            "mask": "Shared-valid across all relevant_full/natural settings on the declared primary roster; retain failures and bounds.",
            "new_study_expected_sign": "No assumed positive effect; finish and report either outcome.",
        },
        "inference": {
            "draws": 10000,
            "seed": 20260930,
            "cluster": "Existing singleton text clusters; same sampled texts across compared settings.",
            "conditional_on": "Observed pool choices and decoder repeats; no artificial repetition of shared reference.",
            "family": "Four accepted SQ3 follow-up workstreams, one primary contrast per workstream; prior completed feedback study is separate.",
            "primary_family_interval_level": 0.9875,
            "primary_quantiles": [0.00625, 0.99375],
            "family_method": "Bonferroni allocation of0.05 across four primary questions; applied to paired percentile-bootstrap intervals.",
            "descriptive_interval_level": 0.95,
            "secondary_intervals": "Pointwise descriptive only.",
            "future_protocol_rule": "Each later primary contrast must be specified before that workstream runs. Extra primary tests require a prospective family amendment; no narrowing after outcomes.",
        },
        "secondary": [
            "All350 outcomes with seven disclosed partial-cover fallbacks",
            "Separate121 original selection holdout and99 historical sentinels; overlapping subsets",
            "relevant_full minus disjoint_full",
            "relevant_full minus relevant_labels",
            "disjoint_full minus disjoint_labels",
            "relevant_labels minus disjoint_labels",
            "(relevant_full-relevant_labels)-(disjoint_full-disjoint_labels)",
            "Natural and relevant_full versus fresh shared reference",
            "Per-label F1, contains, exact, gate misses, label count, failures, costs",
        ],
        "batching": {
            "input_batches": batches,
            "all_conditions_for_input_together": True,
            "predictions_per_input": 48,
            "predictions_by_batch": [len(b) * 48 for b in batches],
            "ordinary_responses_by_batch": [len(b) * 144 for b in batches],
            "combined_command": "Prepare one resumable user launcher, with automatic integrity audit before each batch advance; no accuracy-based stopping.",
        },
        "budget": {
            "inputs": 350,
            "logical_settings": 16,
            "distinct_prompts_before_identity_dedup": 5600,
            "predictions": 16800,
            "ordinary_responses": 50400,
            "max_logical_responses_with_repairs": 151200,
            "max_transport_POST_attempts": 756000,
            "limit_note": "Arithmetic inherited repair/retry ceilings, not expected consumption.",
        },
        "scope": [
            "No model retraining, CUDA or active KB/taxonomy edit.",
            "No TEST execution or LaTeX.",
            "Label-only scores are artificial diagnostics. Oracle scores are not deployable accuracy or universal ceilings.",
            "No item is removed for an unfavorable prediction. The343 eligibility mask is determined by source label coverage.",
            "The other accepted explanation, applicability and taxonomy studies remain required regardless of this result.",
        ],
        "actual_API_calls": 0,
    }
    h.save(OUT / "scientific_plan_v1.json", plan)
    h.save_rows(OUT / "choices.jsonl", choices)
    h.require(
        all(h.digest(ROOT / name) == sha for name, sha in inputs.items()),
        "Source changed during freeze",
    )
    h.save(
        OUT / "design_complete.json",
        {
            "status": plan["status"],
            "created_utc": h.now(),
            "scientific_plan_sha256": h.digest(OUT / "scientific_plan_v1.json"),
            "choices_sha256": h.digest(OUT / "choices.jsonl"),
            "selection_checks": 2,
            "natural_choices": 1050,
            "primary_texts": len(primary),
            "secondary_fallbacks": len(secondary_only),
            "actual_API_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": plan["status"],
                "primary": len(primary),
                "fallback": secondary_only,
                "budget": plan["budget"],
                "predictions_by_batch": plan["batching"]["predictions_by_batch"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    with patch.object(
        socket,
        "create_connection",
        side_effect=RuntimeError("No network during scientific preparation"),
    ), patch.object(
        socket.socket,
        "connect",
        side_effect=RuntimeError("No network during scientific preparation"),
    ):
        prepare()

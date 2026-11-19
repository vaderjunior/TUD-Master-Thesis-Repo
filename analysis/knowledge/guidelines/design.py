"""Guidelines design."""

from datetime import datetime, timezone
from pathlib import Path
import json
import socket
from unittest.mock import patch
from analysis.knowledge.guidelines import inventory as inv

r = inv.r
u = inv.u
ROOT = inv.ROOT
BASE = inv.BASE
PLAN = BASE / "plan.json"
ANALYSIS = BASE / "analysis_plan.json"
CONDITIONS = ["no_guidelines", "natural", "matched_rule", "alternative_rule", "filler_only"]
PRIMARY = {
    "matched_minus_none": {"matched_rule": 1, "no_guidelines": -1},
    "matched_minus_alternative": {"matched_rule": 1, "alternative_rule": -1},
    "natural_minus_none": {"natural": 1, "no_guidelines": -1},
    "matched_minus_filler": {"matched_rule": 1, "filler_only": -1},
}
SECONDARY = {
    "alternative_minus_none": {"alternative_rule": 1, "no_guidelines": -1},
    "filler_minus_none": {"filler_only": 1, "no_guidelines": -1},
    "matched_minus_natural": {"matched_rule": 1, "natural": -1},
    "alternative_minus_natural": {"alternative_rule": 1, "natural": -1},
    "filler_minus_natural": {"filler_only": 1, "natural": -1},
    "alternative_minus_filler": {"alternative_rule": 1, "filler_only": -1},
}


def main():
    r.require(not PLAN.exists() and (not ANALYSIS.exists()), "Preserve scientific design")
    evidence = r.read(inv.EVIDENCE)
    done = r.read(inv.DEST / "complete.json")
    r.require(
        done["evidence_sha256"] == r.sha(inv.EVIDENCE)
        and evidence["status"] == "offline_roster_rules_and_query_cache_verified",
        "Inventory binding differs",
    )
    r.verify_hashes(evidence["input_hashes"])
    r.verify_hashes(evidence["generated_hashes"])
    r.require(
        evidence["selected_rows"] == 693
        and evidence["selected_templates"] == 302
        and evidence["all_selected_gold_non_hate"],
        "Wrong planned roster",
    )
    r.require(
        not any(
            (
                evidence[k]
                for k in (
                    "new_embedding_calls",
                    "CUDA_used",
                    "new_retrieval_calls",
                    "new_API_calls",
                    "new_gold_scoring",
                )
            )
        ),
        "Inventory scope changed",
    )
    core = r.read(r.CORE / "plan.json")
    environment = u.environment()
    r.require(environment == core["environment"], "Environment changed since core")
    old_path = ROOT / "experiments/astra/sq1_full_kb_strategy_20260920_v1/plan.json"
    old = r.read(old_path)
    source = r.BASE / "full/chroma"
    physical = {r.rel(p): r.sha(p) for p in sorted(source.rglob("*")) if p.is_file()}
    r.require(
        physical == old["source_index_hashes"] and len(physical) == 6, "Full source index differs"
    )
    r.require(
        core["model"] == "Qwen/Qwen3-VL-32B-Instruct"
        and core["temperature"] == 1.0
        and (core["votes_per_item"] == 3),
        "Model recipe differs",
    )
    rows = r.rows(inv.DEST / "inputs.jsonl")
    all_ids = [x["item_id"] for x in rows]
    r.require(len(set(all_ids)) == len(all_ids) == 693, "Duplicate input")
    pairs = {frozenset(weights) for weights in {**PRIMARY, **SECONDARY}.values()}
    r.require(
        len(PRIMARY) == 4
        and len(pairs) == 10
        and all(
            (
                set(w.values()) == {-1, 1} and set(w) <= set(CONDITIONS)
                for w in {**PRIMARY, **SECONDARY}.values()
            )
        ),
        "Pairwise design differs",
    )
    analysis = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "frozen_before_new_predictions",
        "subset": "en_hatecheck_eval",
        "stratum": "guideline_covered",
        "dimension": "hate",
        "outcome": "false_positive_rate",
        "all_gold_hate": False,
        "ordered_item_ids": all_ids,
        "conditions": CONDITIONS,
        "repeats": [0, 1, 2],
        "primary_contrasts": PRIMARY,
        "secondary_contrasts": SECONDARY,
        "primary_family_size": 4,
        "primary_family_confidence": 1 - 0.05 / 4,
        "descriptive_confidence": 0.95,
        "bootstrap_draws": 10000,
        "bootstrap_seed": 20260925,
        "bootstrap_unit": "templ_id",
        "bootstrap_estimator": "Item-weighted mean FP rate across three fixed repeats; whole common-item template contents drawn jointly across all settings.",
        "verification": "Independent item-expanded bootstrap using identical cluster draws; independently reconstruct FP/null counts and full-roster bounds.",
        "validity": "One shared mask valid in all 15 settings. Retain own-valid rates, null counts, affected IDs/templates and full-roster null bounds. No null-to-negative conversion.",
        "per_functionality": "All six groups, descriptive 95% only; preserve support and small template counts.",
        "repeat_summary": "All three rates; mean/sample SD/range. Historical range floor descriptive only.",
        "no_partial_scoring": True,
        "no_equivalence_or_automatic_promotion": True,
        "limits": evidence["limits"]
        + [
            "Pointwise intervals never replace the prespecified four-contrast family rule.",
            "No positive examples in this roster: no hate-recall or overall-system-quality conclusion.",
        ],
    }
    sizes = [174, 174, 174, 171]
    offset = 0
    stages = []
    for index, n in enumerate(sizes, 1):
        selected = rows[offset : offset + n]
        offset += n
        stages.append(
            {
                "id": f"stage_{index:02d}",
                "inputs": n,
                "predictions": n * 15,
                "ordinary_responses": n * 45,
                "first_input": selected[0]["key"],
                "last_input": selected[-1]["key"],
                "input_keys_sha256": r.signature([x["key"] for x in selected]),
            }
        )
    r.require(
        offset == 693 and sum((x["predictions"] for x in stages)) == 10395,
        "Stage arithmetic differs",
    )
    bindings = {**evidence["input_hashes"], **evidence["generated_hashes"]}
    for path in (
        inv.EVIDENCE,
        inv.DEST / "complete.json",
        Path(__file__),
        BASE / "PROTOCOL.md",
        old_path,
        r.BASE / "full/origin.json",
        r.PLAN,
        ROOT / "src/hsrag/retrieve.py",
        ROOT / "analysis/common/prompt.py",
        ROOT / "analysis/knowledge/strategies/prepare.py",
        ROOT / "analysis/knowledge/allocation/verify_preparation.py",
    ):
        bindings[r.rel(path)] = r.sha(path)
    inv.write_new(ANALYSIS, analysis)
    plan = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "scientific_design_frozen_inputs_checked_prompt_preparation_pending",
        "run_id": BASE.name,
        "input_hashes": bindings,
        "analysis_plan_sha256": r.sha(ANALYSIS),
        "environment": environment,
        "source_index": r.rel(source),
        "source_index_hashes": physical,
        "inventory_sha256": r.sha(inv.EVIDENCE),
        "model": core["model"],
        "provider": core["provider"],
        "temperature": 1.0,
        "votes_per_prediction": 3,
        "decoder_repeats": [0, 1, 2],
        "retrieval": core["retrieval"],
        "conditions": CONDITIONS,
        "primary_family_size": 4,
        "primary_family_confidence": 1 - 0.05 / 4,
        "counts": {
            "inputs": 693,
            "templates": 302,
            "conditions": 5,
            "prompt_payloads": 3465,
            "predictions": 10395,
            "ordinary_responses": 31185,
        },
        "planned_stages": stages,
        "planned_stage_count": 4,
        "scope_gate": {
            "currently_authorized": "Offline preparation of the targeted existing 693-case oracle stratum only.",
            "live_execution": "Not prepared or started. Requires independent prompt audit and separate tested API readiness checkpoint before giving Ajay a named-stage command.",
            "parked": [
                "remaining 3035 HateCheck cases",
                "other TEST/benchmarks",
                "Priority 3/SQ3 adaptation",
                "LaTeX",
            ],
        },
        "prompt_contract": "Original renderer/caution wording. Same two definitions and five examples across all arms. Metadata never supplied to model. Natural guideline bucket follows existing cross-language abstract retrieval; forced rules unchanged.",
        "preparation_contract": "Verified cached FP32 queries/full vectors only; hashed independent source-index copies, traced/untraced smoke, full first pass and one cold replay, independent audit. No source-index client, model loading, embedding fallback or API.",
        "smoke_keys": [
            next((x["key"] for x in rows if x["functionality"] == fn))
            for fn in sorted(evidence["by_functionality"])
        ],
        "immutability": "Scientific design, inventory and historical outputs are frozen. Separate implementation manifest/tests may be added, bound to this plan; defects require explicit versioned amendments.",
        "new_API_calls": 0,
        "new_retrieval_calls": 0,
        "new_embedding_calls": 0,
        "next": "Build and test the isolated CPU prompt preparation, then audit before any API handoff.",
    }
    inv.write_new(PLAN, plan)
    inv.write_new(
        BASE / "design_complete.json",
        {
            "status": plan["status"],
            "plan_sha256": r.sha(PLAN),
            "analysis_sha256": r.sha(ANALYSIS),
            "inventory_sha256": r.sha(inv.EVIDENCE),
            "source_sha256": r.sha(Path(__file__)),
            "new_API_calls": 0,
            "new_retrieval_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": plan["status"],
                "counts": plan["counts"],
                "stages": stages,
                "source_index_files": len(physical),
                "plan_sha256": r.sha(PLAN),
                "analysis_sha256": r.sha(ANALYSIS),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", inv.forbidden), patch.object(
        socket, "create_connection", inv.forbidden
    ):
        main()

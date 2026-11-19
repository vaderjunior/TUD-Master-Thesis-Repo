"""Sampled banks design."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
from analysis.knowledge.sampled_banks import annotations as c

ROOT = c.ROOT
CORE = c.CORE
BASE = ROOT / "experiments/astra/sq1_pool_size_composition_20260926_v1"
SEEDS = (42, 43, 44)
SIZES = {"en": (147, 1000, 5000), "de": (153, 1000, 5000)}
SALT = "astra-sq1-pools-v1"


def rank(record_id, lang, seed):
    return (hashlib.sha256(f"{SALT}|{lang}|{seed}|{record_id}".encode("utf8")).digest(), record_id)


def main():
    socket.socket.connect = c.block_network
    socket.create_connection = c.block_network
    c.require(not (BASE / "scientific_plan.json").exists(), "Preserve frozen pool design")
    audit = c.read(c.BASE / "results.json")
    verified = c.read(c.BASE / "independent_check.json")
    c.require(
        verified["status"] == "annotation_census_independently_verified"
        and verified["results_sha256"] == c.digest(c.BASE / "results.json"),
        "Coverage audit not verified",
    )
    for name, sha in audit["input_hashes"].items():
        c.require(c.digest(ROOT / name) == sha, "Audit input changed")
    full = c.rows(CORE / "kb/full/records.jsonl")
    small = c.rows(CORE / "kb/small/records.jsonl")
    by_id = {r["id"]: r for r in full}
    language_pools = {
        lang: [r for r in full if r["kind"] == "example" and r["lang"] == lang] for lang in SIZES
    }
    abstracts = [r["id"] for r in full if r["kind"] != "example"]
    c.require(
        len(abstracts) == 40 and all((r["id"] in by_id and by_id[r["id"]] == r for r in small)),
        "Pool identity error",
    )
    manifests = []
    for size_index, size_name in enumerate(("compact", "1000", "5000")):
        for seed in SEEDS:
            entry = {
                "condition": f"uniform_{size_name}_s{seed}",
                "seed": seed,
                "size_name": size_name,
                "abstract_ids": abstracts,
                "languages": {},
            }
            for lang, pool in language_pools.items():
                ordered = sorted(pool, key=lambda r: rank(r["id"], lang, seed))
                n = SIZES[lang][size_index]
                chosen = {r["id"] for r in ordered[:n]}
                selected = [r for r in pool if r["id"] in chosen]
                c.require(len(selected) == n, "Wrong subset cardinality")
                entry["languages"][lang] = {
                    "size": n,
                    "example_ids": [r["id"] for r in selected],
                    "selection_prefix_ids": [r["id"] for r in ordered[:n]],
                    "inventory": c.describe(selected),
                }
            manifests.append(entry)
    c.save_new(BASE / "pool_manifests.json", manifests)
    paths = [
        Path(__file__),
        ROOT / "analysis/knowledge/sampled_banks/verify_design.py",
        ROOT / "analysis/knowledge/sampled_banks/annotations.py",
        ROOT / "analysis/knowledge/sampled_banks/verify_annotations.py",
        c.BASE / "results.json",
        c.BASE / "independent_check.json",
        CORE / "plan.json",
        CORE / "dev_inputs.jsonl",
        CORE / "kb/full/records.jsonl",
        CORE / "kb/small/records.jsonl",
        CORE / "retrieval_v1/full/prompts.jsonl",
        CORE / "retrieval_v1/small/prompts.jsonl",
        CORE / "retrieval_v1/plan.json",
        BASE / "pool_manifests.json",
    ]
    plan = {
        "status": "scientific_design_and_TRAIN_pool_manifests_frozen_execution_not_ready",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "questions": [
            "How does performance change as a uniformly sampled candidate pool grows, while only five examples reach each prompt?",
            "At the same record budget, how does the existing curated small KB compare with three random subsets of the full eligible TRAIN pool?",
        ],
        "rationale": "The verified census found simultaneous changes in pool size, class/source mixture and annotation density. Existing full/small scores cannot isolate size.",
        "selection_history": "Designed after inspecting existing DEV accuracy and the annotation census. This is a post-selection mechanism study, not untouched confirmation.",
        "sampling": {
            "seeds": SEEDS,
            "sizes_by_language": SIZES,
            "salt": SALT,
            "algorithm": "Sort every eligible language-specific example by (raw SHA256 bytes of UTF8 salt|language|seed|record_id, record_id). Take prefixes. Retain original full-KB order for index construction.",
            "nested": "For each language and seed, compact is a subset of 1000, which is a subset of 5000, which is a subset of full.",
            "policy": "Uniform hash ordering without replacement; no quotas, rebalancing, coverage-based rejection or resampling. Retain poor label coverage if drawn.",
            "interpretation": "Same sampling policy across sizes; realised class/source/annotation composition can vary. Three fixed draws quantify limited pool variation, not a precise population distribution.",
            "labels_used_to_sample": False,
            "query_text_or_gold_used_to_sample": False,
        },
        "conditions": ["curated_small", "full_reference"] + [m["condition"] for m in manifests],
        "reference_prompts": {
            "curated_small": c.rel(CORE / "retrieval_v1/small/prompts.jsonl"),
            "full_reference": c.rel(CORE / "retrieval_v1/full/prompts.jsonl"),
        },
        "context_control": {
            "examples": 5,
            "definitions": 2,
            "guidelines": 2,
            "strategy": "existing hybrid",
            "abstracts": "Use the original frozen full-reference definitions/guidelines for each input in every arm. The census verified identical ordered full/small abstract lists for all 1295 inputs.",
            "renderer": "Original corrected renderer and RAG caution, original system taxonomy, target text, language and message order.",
            "reference": "Use byte-identical accepted first-pass full/small message payloads; run fresh classification repeats for both. No old predictions added as new repeats.",
            "retrieval": "Only candidate example pool changes. Rebuild BM25/index statistics for that pool using the unchanged hybrid recipe and frozen embedding/query vectors. Persist first-pass prompts and cold replay before API readiness.",
            "language": "Preserve the existing language filter and existing retrieval settings; no task-head, gate or query-gold filtering.",
            "index_records": "Each sampled condition combines its EN and DE sample plus the same 40 abstract records, in inherited full order. Abstract prompt selections stay fixed.",
            "record_content": "IDs refer to byte-identical frozen full TRAIN records and labels; no source cleanup or annotation imputation.",
        },
        "evaluation": {
            "all_rosters": {
                s: {"lang": lang, "field": field, "items": n}
                for s, (lang, field, n) in c.SCOPES.items()
            },
            "focal_rosters": ["en_dev_eval_main", "en_dev_eval_sq3_types", "de_legal_dev_eval"],
            "primary_contrasts_per_focal_roster": [
                "full_reference minus mean(uniform_compact_s42,s43,s44)",
                "curated_small minus mean(uniform_compact_s42,s43,s44)",
            ],
            "primary_family": {
                "contrasts": 6,
                "family_alpha": 0.05,
                "per_contrast_two_sided_CI_percent": 100 * (1 - 0.05 / 6),
            },
            "secondary": "All seven rosters retain full tables, per-label/errors/coverage; intermediate 1000/5000 size curves and full-minus-curated replication use descriptive 95% intervals. Do not select or promote a winning size from these intervals.",
            "repeats": 3,
            "votes_per_prediction": 3,
            "aggregation": "Calculate fixed-label macro-F1 for each pool and decoder repeat separately, average three repeats per pool, then equally average three pools for each uniform size. Do not pool examples/predictions or count nine settings as nine independent seeds.",
            "uncertainty": "10000 paired item bootstrap draws, seed 20260926, same item draw for all settings in an endpoint. Average per-setting F1 exactly as above. Conditional on the three fixed pool draws and decoder repeats; separately report between-pool and within-pool variation. Shared all-33-setting complete mask per endpoint plus full-roster null bounds and original denominators.",
            "failure": "Retain final nulls and invalid votes under the existing bounded repair policy; no favourable reruns. Renderer missing TRAIN annotations differ from missing/empty model outputs.",
            "limits": "The single historical curated pool is not replicated curation; differences concern this curation instance. More label records are not more independent evaluation texts. No new TEST, benchmark or adaptation scope.",
        },
        "budget": {
            "distinct_inputs": 1295,
            "conditions_per_input": 11,
            "prompts": 1295 * 11,
            "predictions": 1295 * 11 * 3,
            "ordinary_responses": 1295 * 11 * 3 * 3,
            "logical_response_ceiling_with_two_repairs_per_vote": 1295 * 11 * 3 * 9,
            "additional_transport_retries": "Separately bounded/accounted by the runner; no unattended recovery of unknown outcomes.",
            "staging": "Prepare four reviewable user-run stages where feasible; actual schedule/interleaving, timing and API readiness require a separate tested runner. No automatic stage advance.",
        },
        "preservation": "New isolated manifests only. All old results, source pools and active indexes remain unchanged; full KB/hybrid/original two definitions stays main.",
        "remaining_before_execution": [
            "Independent manifest verification",
            "Frozen retrieval protocol, source/environment binding, isolated indexes and vector identity checks",
            "Saved prompts plus full cold replay and independent retrieval/prompt audit",
            "Freeze complete scoring/failure contract, schedule, bounded resumable runner, mock execution and restart checks",
        ],
        "deferred": "Any intervention on label availability itself needs a separate matched-text design; this study tests size/sampling policy. Priority 3, broader TEST/benchmarks and LaTeX remain parked.",
        "input_hashes": {c.rel(p): c.digest(p) for p in paths},
    }
    c.save_new(BASE / "scientific_plan.json", plan)
    c.save_new(
        BASE / "preparation.json",
        {
            "status": plan["status"],
            "scientific_plan_sha256": c.digest(BASE / "scientific_plan.json"),
            "pool_manifests_sha256": c.digest(BASE / "pool_manifests.json"),
            "actual_API_calls": 0,
            "retrieval_started": False,
            "CUDA_used": False,
        },
    )
    print(
        json.dumps(
            {
                "status": plan["status"],
                "candidate_manifests": len(manifests),
                "language_subsets": len(manifests) * 2,
                "scientific_plan_sha256": c.digest(BASE / "scientific_plan.json"),
                "budget": plan["budget"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

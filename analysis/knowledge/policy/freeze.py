"""Policy freeze."""

from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import hashlib
import math
from pathlib import Path
import shutil
import yaml
from analysis.knowledge.policy import setup as s

CHOICE = "en2400_de1500_all_fine"
RESOURCE = (
    s.ROOT
    / "experiments/astra/german_guidelines_synthesis_20261113_v1/runs/run_1794534112318762800"
)
ENDPOINTS = {
    "en_hate": ("en_test", "hate", "main"),
    "en_hate_type": ("en_test", "hate_type", "main"),
    "de_hate": ("de_test", "hate", "main"),
    "de_legal": ("de_legal_test", "legal", "main"),
    "en_target_group": ("en_test", "target_group", "supporting"),
    "en_severity": ("en_test", "severity", "supporting"),
}


def freeze():
    before = s.read(s.BASE / "source_snapshot.json")
    for name, value in before["sources"].items():
        s.require(s.sha(s.ROOT / name) == value["sha256"], "Changed original source: " + name)
    candidate = s.read(s.BASE / "candidates" / (CHOICE + ".json"))
    profile = next((v for v in s.read(s.BASE / "candidate_profiles.json") if v["name"] == CHOICE))
    original = {r["key"]: r for r in s.rows(s.GOLD)}
    selected = set(candidate["selected_keys"])
    selected_rows = [original[k] for k in sorted(selected)]
    taxonomy = yaml.safe_load((s.ROOT / "config/taxonomy.yaml").read_text(encoding="utf-8"))
    inventory = s.read(s.BASE / "policy_inventory.json")
    flatten = lambda text: " ".join(text.split())
    for item in inventory:
        r = item["record"]
        if r["kind"] == "definition":
            t = taxonomy["dimensions"][r["dimension"]]
            expected = (
                t["definition"]
                if r["dimension"] == "hate"
                else t["labels"][r["label"]]["definition"]
            )
            s.require(
                flatten(r["text"]) == flatten(expected),
                "Definition differs from taxonomy: " + r["id"],
            )
    s.require(len({r["record"]["id"] for r in inventory}) == 40, "Repeated policy identity")
    s.require(len({r["text_sha256"] for r in inventory}) == 40, "Repeated policy text")
    endpoints = {}
    for name, (phase, dimension, role) in ENDPOINTS.items():
        keys = candidate["endpoints"][name]
        labels = (
            ["not_hate", "hate"]
            if dimension == "hate"
            else list(taxonomy["dimensions"][dimension]["labels"])
        )
        support = Counter()
        for key in keys:
            r = original[key]
            v = s.truth(r, dimension)
            s.require(r["phase"] == phase and v is not None, "Invalid endpoint membership")
            positive = (
                ["hate" if v else "not_hate"]
                if dimension == "hate"
                else [v] if dimension == "severity" else v
            )
            s.require(set(positive) <= set(labels), "Unknown gold label")
            support.update(positive)
        macro = (
            labels if dimension == "hate" else [label for label in labels if support[label] >= 10]
        )
        clusters = Counter((original[k]["text_cluster"] for k in keys))
        endpoints[name] = {
            "phase": phase,
            "dimension": dimension,
            "role": role,
            "keys": keys,
            "n": len(keys),
            "all_labels": labels,
            "macro_labels": macro,
            "gold_support": {label: support[label] for label in labels},
            "minimum_macro_support": 10,
            "text_clusters": len(clusters),
            "sources": dict(Counter((original[k]["source"] for k in keys))),
            "valid_empty_gold": sum((s.truth(original[k], dimension) == [] for k in keys)),
        }
    s.save("endpoints.json", endpoints)
    s.write_rows(
        "roster.jsonl",
        [
            r
            | {
                "analysis_endpoints": [
                    e for e in endpoints if r["key"] in set(endpoints[e]["keys"])
                ]
            }
            for r in selected_rows
        ],
    )
    token_rows = [
        r for r in s.rows(s.BASE / "candidate_prompt_counts.jsonl") if r["key"] in selected
    ]
    s.write_rows("prompt_bindings.jsonl", token_rows)
    cluster_sources = defaultdict(set)
    for r in selected_rows:
        cluster_sources[r["text_cluster"]].add(r["source"])
    overlap = {
        a: {b: len(set(endpoints[a]["keys"]) & set(endpoints[b]["keys"])) for b in endpoints}
        for a in endpoints
    }
    ann_path = s.BROAD / "replay_review_20261009_v1/sensitivity_plan.json"
    flags = s.read(ann_path)["exclusion_roster"]
    ann_keys = sorted(set((r["key"] for r in flags)) & selected)
    sensitivity = {
        "ann_replay_exclusion_keys": ann_keys,
        "whitespace_keys": sorted((r["key"] for r in selected_rows if not r["text"].strip())),
        "endpoint_ann_keys": {
            e: sorted(set(v["keys"]) & set(ann_keys)) for e, v in endpoints.items()
        },
        "rule": "Same removals in A/B/C, original fixed label scopes; no outcome-driven exclusions.",
    }
    s.save("sensitivity_rosters.json", sensitivity)
    summary = {
        "selected_candidate": CHOICE,
        "inputs": len(selected),
        "populations": dict(Counter((r["phase"] for r in selected_rows))),
        "sources": dict(Counter((r["source"] for r in selected_rows))),
        "normalized_text_clusters": len(cluster_sources),
        "clusters_crossing_sources": sum((len(v) > 1 for v in cluster_sources.values())),
        "endpoint_overlap": overlap,
        "sampling_strata": candidate["strata"],
        "fine_union": len(
            set().union(
                *(
                    set(endpoints[e]["keys"])
                    for e in ("en_hate_type", "en_target_group", "en_severity")
                )
            )
        ),
        "fine_only_outside_en_hate": len(
            set().union(
                *(
                    set(endpoints[e]["keys"])
                    for e in ("en_hate_type", "en_target_group", "en_severity")
                )
            )
            - set(endpoints["en_hate"]["keys"])
        ),
        "unknown_hate_excluded_from_sampling": sum(
            (r["phase"] == "en_test" and r["gate"] is None for r in original.values())
        ),
        "candidate_profiles": s.read(s.BASE / "candidate_profiles.json"),
    }
    s.save("roster_summary.json", summary)
    contrasts = [
        ("complete_policy", "retrieved_policy"),
        ("complete_definitions", "retrieved_policy"),
        ("complete_policy", "complete_definitions"),
    ]
    registry = []
    for ep, (_, dimension, role) in ENDPOINTS.items():
        for index, (left, right) in enumerate(contrasts):
            family = (
                ("main_macro_4" if index == 0 else "explanatory_macro_8")
                if role == "main"
                else None
            )
            registry.append(
                {
                    "endpoint": ep,
                    "metric": "macro_f1",
                    "left": left,
                    "right": right,
                    "family": family,
                    "role": "primary" if family == "main_macro_4" else "supporting",
                }
            )
            if dimension in ("hate_type", "target_group", "legal"):
                registry.append(
                    {
                        "endpoint": ep,
                        "metric": "micro_f1_all_labels",
                        "left": left,
                        "right": right,
                        "family": "complementary_micro_9",
                        "role": "complementary",
                    }
                )
            registry.append(
                {
                    "endpoint": ep,
                    "metric": "mean_pairwise_repeat_agreement",
                    "left": left,
                    "right": right,
                    "family": "complete_policy_agreement_6" if index == 0 else None,
                    "role": "supporting",
                }
            )
    families = {
        name: {"size": size, "confidence": 1 - 0.05 / size, "tail_quantile": 0.05 / (2 * size)}
        for name, size in [
            ("main_macro_4", 4),
            ("explanatory_macro_8", 8),
            ("complementary_micro_9", 9),
            ("complete_policy_agreement_6", 6),
        ]
    }
    for name, family in families.items():
        s.require(
            sum((row["family"] == name for row in registry)) == family["size"],
            "Family count differs",
        )
    s.save(
        "comparison_registry.json",
        {
            "rows": registry,
            "families": families,
            "draws": 50000,
            "seed": 20261114,
            "method": "paired normalized-text cluster bootstrap stratified by sorted source-membership tuple",
            "mask": "All three conditions and all three outer repeats valid within each endpoint; use this same nine-setting mask for every main contrast.",
            "label_scopes": "Frozen before outcomes in endpoints.json; no support filtering within bootstrap or sensitivities.",
            "averaging": "Compute each outer-repeat score then average three scores. Never pool nine votes as independent texts.",
            "family_caution": "Separate prespecified question families, not one overall 5% error guarantee across every result. Supporting intervals outside these families are descriptive.",
            "bootstrap_caution": "Conditional on the selected, previously exposed texts and the three fixed repeats. No confirmatory or untouched-test claim.",
        },
    )
    resources = s.read(RESOURCE / "resources.json")["groups"]["all"]
    timing = s.read(RESOURCE / "timing.json")["wall"]
    storage = s.read(RESOURCE / "storage.json")["D_study"]
    ordinary = profile["ordinary_calls"]
    initial_tokens = sum((v["ordinary_input_tokens"] for v in profile["tokens"].values()))
    prior_calls = resources["ordinary_logical_calls"]
    repair_ratio = resources["repair_logical_calls"] / prior_calls
    previous = resources["usage"]["ordinary"]
    output_mean = previous["completion_tokens"]["mean"]
    predicted_total = initial_tokens + ordinary * output_mean
    prior_total = previous["total_tokens"]["sum"]
    request_hours = timing["filesystem_indicated_end_to_end_s"] / 3600 * ordinary / prior_calls
    token_hours = timing["filesystem_indicated_end_to_end_s"] / 3600 * predicted_total / prior_total
    start_reserve_hours = 18
    deadline = datetime(2026, 11, 15, 23, 59, 59, tzinfo=timezone(timedelta(hours=1)))
    storage_guard = 8 * 2**30 + profile["predictions"] * 2**20
    budget = {
        "candidate": CHOICE,
        "counts": {
            key: profile[key] for key in ("inputs", "prompts", "predictions", "ordinary_calls")
        },
        "logical_call_ceiling": ordinary * 3,
        "physical_POST_ceiling": ordinary * 15,
        "expected_repairs_if_prior_rate_repeats": ordinary * repair_ratio,
        "prior_repair_per_ordinary": repair_ratio,
        "initial_input_tokens": initial_tokens,
        "condition_input_tokens": profile["tokens"],
        "maximum_initial_input_tokens": max((v["max"] for v in profile["tokens"].values())),
        "output_token_allowance_per_request": 4096,
        "maximum_initial_input_plus_output_allowance": max(
            (v["max"] for v in profile["tokens"].values())
        )
        + 4096,
        "initial_input_limit": 8192,
        "prior_reference": s.ref(RESOURCE),
        "prior_seconds": timing["filesystem_indicated_end_to_end_s"],
        "prior_ordinary_calls": prior_calls,
        "prior_initial_input_tokens": previous["prompt_tokens"]["sum"],
        "prior_initial_output_tokens": previous["completion_tokens"]["sum"],
        "prior_output_tokens_per_ordinary": output_mean,
        "request_rate_scenario_hours": request_hours,
        "proportional_total_token_scenario_hours": token_hours,
        "planning_window_hours": [8, 16],
        "launch_reserve_hours": start_reserve_hours,
        "deadline_local": deadline.isoformat(),
        "latest_new_launch_local": (deadline - timedelta(hours=start_reserve_hours)).isoformat(),
        "estimate_caution": "Scenarios, not a live benchmark or guarantee. Token-linear scaling uses previous end-to-end wall time and ordinary-token ratio; repairs, audits and failures can differ. No cache savings assumed.",
        "raw_root_planned": "D:/Thesis_Experiments/complete_policy_20261114_v1",
        "free_bytes_at_freeze": {d: shutil.disk_usage(d).free for d in ("C:/", "D:/")},
        "D_practical_start_headroom_bytes": storage_guard,
        "D_minimum_free_floor_bytes": 8 * 2**30,
        "D_remaining_allowance_bytes_per_prediction": 2**20,
        "C_minimum_free_floor_bytes": 8 * 2**30,
        "storage_caution": "Practical conservative guard, not a mathematical worst-case bound for all retries. Recalculate before launch and each batch. Preserve evidence and stop before violating the floor.",
        "prior_study_storage": {
            "files": storage["files"],
            "logical_bytes": storage["logical_bytes"],
        },
        "prior_size_per_prediction": storage["logical_bytes"] / resources["predictions"],
        "hypothetical_all_token_ceiling": ordinary * 3 * 4096,
        "ceiling_caution": "Repair/retry ceilings are hard multiplicative limits, not expected traffic or runtime. Retries can consume unreported remote work.",
    }
    s.require(
        budget["maximum_initial_input_tokens"] < budget["initial_input_limit"],
        "Prompt exceeds initial cap",
    )
    s.require(budget["free_bytes_at_freeze"]["D:/"] > storage_guard, "D headroom inadequate")
    s.save("budget.json", budget)
    extra = [
        ann_path,
        RESOURCE / "resources.json",
        RESOURCE / "timing.json",
        RESOURCE / "storage.json",
        s.ROOT / "analysis/knowledge/policy/setup.py",
        Path(__file__),
        s.ROOT / "analysis/common/scoring.py",
        s.ROOT / "analysis/common/result_reader.py",
        s.ROOT / "src/hsrag/parse.py",
        s.ROOT / "src/hsrag/vote.py",
        s.ROOT / "src/hsrag/schema.py",
        s.ROOT / "Thesis_Log/astra_result/micro_f1_comparison_setup_20261113_v1.md",
    ]
    bindings = {name: value["sha256"] for name, value in before["sources"].items()}
    bindings.update({s.ref(p): s.sha(p) for p in extra})
    s.save("frozen_source_bindings.json", bindings)
    s.save(
        "protocol.json",
        {
            "version": 1,
            "frozen_utc": datetime.now(timezone.utc).isoformat(),
            "status": "PART_1_DESIGN_ONLY_NO_EXECUTION",
            "selection_seed": s.SELECTION_SEED,
            "candidate": CHOICE,
            "conditions": [
                {
                    "id": name,
                    "definitions": 2 if i == 0 else 21,
                    "guidelines": 19 if i == 2 else 2,
                    "examples": 5,
                }
                for i, name in enumerate(s.CONDITIONS)
            ],
            "policy_order": "Original bank index order for complete blocks; original retrieved order in unchanged blocks.",
            "query_languages": ["en", "de"],
            "policy_languages": {"definitions": {"en": 21}, "guidelines": {"en": 11, "de": 8}},
            "outer_repeats": [0, 1, 2],
            "inner_votes": 3,
            "workers": 128,
            "single_queue": True,
            "new_reference": True,
            "prompt_sources": "Original full-bank saved prompts. No new retrieval, gold-conditioned policy selection, translation or content edit.",
            "sampling": "Largest-remainder source x gold-hate proportional quota, SHA256(seed|key) order. All eligible EN fine fields and DE legal retained. Separate endpoint denominators.",
            "model_contract": {
                "model": "Qwen/Qwen3-VL-32B-Instruct",
                "temperature": 1.0,
                "max_tokens": 4096,
                "timeout_seconds": 180,
                "votes": 3,
                "max_repairs": 2,
                "max_transport_attempts": 5,
                "backoff_seconds": [2, 2, 4, 8],
                "allow_fallback": False,
            },
            "execution_design": {
                "group_size_inputs": 150,
                "max_predictions_per_batch": 1350,
                "interleaving": "SHA256(seed|schedule|key) query order. All nine condition/repeat jobs together per key with cyclic rotation by key position. Fresh output for every job.",
                "deadline_local": deadline.isoformat(),
                "latest_new_launch_local": budget["latest_new_launch_local"],
                "completion": "All 9N prediction receipts plus item and batch replay audits; no automatic subsequent study.",
                "resume": "Use saved intent/outcome state, locks and bounded retry history. Unknown outcomes stop for review; no blind resend.",
                "repair_context": "Check complete repair-message tokens before every transport request; stop without truncation on overflow.",
                "no_quality_stopping": True,
            },
            "analysis_scope": "One late SQ1 architectural supplement, not a new tuned main configuration or fresh independent test. Existing three-condition design is not a factorial or equal-token experiment.",
            "deferred_to_part2": [
                "Executable prompt matrix and ground-truth-free job table",
                "128-worker interleaving and single-queue enforcement tests",
                "Parser/vote/repair replay tests",
                "Unknown-response recovery and bounded transport tests",
                "Deadline/long-path/resource/PowerShell checks",
                "Actual deployment model and context compatibility on user launch",
            ],
            "source_bindings_sha256": s.sha(s.BASE / "frozen_source_bindings.json"),
            "artifacts": {
                name: s.sha(s.BASE / name)
                for name in [
                    "policy_inventory.json",
                    "endpoints.json",
                    "roster.jsonl",
                    "prompt_bindings.jsonl",
                    "sensitivity_rosters.json",
                    "roster_summary.json",
                    "comparison_registry.json",
                    "budget.json",
                ]
            },
        },
    )
    print(
        "Frozen offline design: "
        + str(profile["inputs"])
        + " inputs; "
        + str(ordinary)
        + " ordinary requests if later launched."
    )
    print(
        "Timing scenarios: %.2f / %.2f hours; storage headroom %.2f GiB."
        % (request_hours, token_hours, storage_guard / 2**30)
    )
    print("Population counts: " + str(summary["populations"]))
    print(
        "Fine union: %s; clusters: %s; ANN keys: %s; whitespace: %s"
        % (
            summary["fine_union"],
            summary["normalized_text_clusters"],
            len(ann_keys),
            len(sensitivity["whitespace_keys"]),
        )
    )


if __name__ == "__main__":
    with s.offline():
        freeze()

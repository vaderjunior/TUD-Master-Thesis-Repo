import argparse
import json
from pathlib import Path
from analysis.common import study as c

OUT = (
    c.ROOT
    / "Thesis_Log/astra_result/evidence/astra_later_preparation_review_20261008_v1.json"
)


def main(write=False):
    if write:
        c.require(not OUT.exists(), "Preserve existing review receipt")
    hashes = {}

    def bind(path):
        hashes[c.rel(path)] = c.sha(path)
        return c.read(path)

    matched = c.BASE / "execution/matched"
    plan = bind(matched / "plan.json")
    c.verify(plan["input_hashes"])
    c.verify(plan["source_hashes"])
    ready = bind(matched / "ready.json")
    c.require(
        ready["plan_sha256"] == c.sha(matched / "plan.json"), "Ready plan differs"
    )
    evidence = bind(c.ROOT / ready["evidence"])
    c.require(
        c.sha(c.ROOT / ready["evidence"]) == ready["evidence_sha256"],
        "Ready proof changed",
    )
    c.require(
        evidence["synthetic_predictions"] == 117
        and evidence["synthetic_replies"] == 351
        and (evidence["new_replies_on_restart"] == evidence["actual_Qwen_calls"] == 0),
        "Wrong rehearsal",
    )
    test_path = next(
        (c.ROOT / n for n in plan["input_hashes"] if "/development/tests_" in n)
    )
    test = bind(test_path)
    c.require(
        test["passed"] and test["tests_run"] == 30 and (test["actual_Qwen_calls"] == 0),
        "Current tests missing",
    )
    ep = bind(c.BASE / "encoders/plan.json")
    c.verify(ep["inputs"])
    preflight = bind(c.BASE / "encoders/preflight.json")
    runtime = bind(c.BASE / "encoders/runtime_preflight.json")
    c.require(
        preflight["status"] == "preflight_pass"
        and preflight["plan_sha256"] == c.sha(c.BASE / "encoders/plan.json"),
        "Encoder preflight missing",
    )
    c.require(
        runtime["preflight_sha256"] == c.sha(c.BASE / "encoders/preflight.json")
        and runtime["entry_sha256"] == c.sha(c.ROOT / runtime["entry"]),
        "Native entry binding changed",
    )
    c.require(
        len(preflight["tokenizer_checks"]) == 30
        and len(preflight["reference_probes"]) == 4
        and all((x["decisions_exact"] for x in preflight["reference_probes"])),
        "Incomplete encoder verification",
    )
    rp = bind(c.BASE / "retrieval_v3/plan.json")
    c.verify(rp["inputs"])
    c.verify(rp["original_index_hashes"])
    vectors = bind(c.BASE / "retrieval_v3/vectors/complete.json")
    c.verify(vectors["outputs"])
    comparison = bind(c.BASE / "retrieval_v3/probe_comparison.json")
    c.require(
        all(
            (
                comparison[k]
                for k in [
                    "same_keys",
                    "same_messages",
                    "same_retrieved_ids",
                    "same_contexts",
                ]
            )
        ),
        "Checkpointed retrieval differs from initial probe",
    )
    probes = {}
    for pool in ("small", "full"):
        path = sorted((c.BASE / "retrieval_v3" / pool).glob("probe_*/complete.json"))[
            -1
        ]
        probe = bind(path)
        c.verify(probe["outputs"])
        c.verify(probe["original_index_hashes_after"])
        c.require(
            probe["rows"] == 26
            and probe["Qwen_calls"] == 0
            and (probe["index_vector_max_abs_difference"] < 2e-07),
            "Retrieval probe failed",
        )
        probes[pool] = {
            "rows": probe["rows"],
            "seconds": probe["seconds"],
            "index_max_abs_error": probe["index_vector_max_abs_difference"],
        }
    for name in [
        "scripts/prepare_comparison_resume.ps1",
        "scripts/run_comparison.ps1",
        "training/original/inference_entry.py",
        "analysis/common/review_preparation.py",
    ]:
        hashes[name] = c.sha(c.ROOT / name)
    result = {
        "status": "preparation_checks_pass_long_offline_batches_not_run",
        "created_utc": c.utc(),
        "matched_budget": plan["budget"],
        "matched_offline_ready": True,
        "synthetic_rehearsal": {
            "predictions": 117,
            "replies": 351,
            "restart_replies": 0,
        },
        "tests": 30,
        "broad_inputs": 21627,
        "broad_Qwen_calls_planned": 583929,
        "combined_Qwen_calls_planned": 694026,
        "vectors": {
            "total": 21627,
            "existing_cache": 21626,
            "exact_whitespace_input_encoded_on_CPU": 1,
        },
        "encoder_budget": ep["budget"],
        "tokenizers_checked": 30,
        "CPU_probe_rows": sum((x["rows"] for x in preflight["reference_probes"])),
        "CPU_reference_max_abs_logit_error": max(
            (x["max_abs_logit_difference"] for x in preflight["reference_probes"])
        ),
        "retrieval_probes": probes,
        "full_probe_matches_before_after_checkpointing": True,
        "new_Qwen_calls": 0,
        "new_training_steps": 0,
        "new_broad_predictions": 0,
        "CUDA_used": False,
        "remaining_user_runs": [
            "CPU broad retrieval, two cold replays, independent prompt check and broad queue rehearsal",
            "CUDA fixed-checkpoint inference and independent assembly",
        ],
        "remaining_service_runs": [
            "A12 balanced correction continuation remains separate",
            "Matched and broad fresh Qwen matrices only when service returns and user explicitly launches",
        ],
        "development_history": [
            "v1 stopped before vector assembly because whitespace-only text was one space, not an empty string; exact text preserved in v2/v3.",
            "v2 bounded probe established timing; v3 adds durable 100-input chunks and single-launcher lock, retaining identical 26-query prompts/contexts.",
            "Direct encoder CPU preflight hit a native Arrow-after-Torch import access violation. Dedicated entry loads Arrow first; all 30 tokenizers and four CPU reference probes then passed. No packages or frozen model code changed.",
        ],
        "verified_evidence_hashes": hashes,
        "limits": "Tests and bounded probes certify preparation behavior, not completion/accuracy of the long pending evaluations. Runtime estimate is a small-sample extrapolation; no thesis outcome has changed.",
    }
    if write:
        with OUT.open("xb") as f:
            f.write(c.encoded(result))
    print(
        json.dumps(
            {
                k: result[k]
                for k in [
                    "status",
                    "tests",
                    "tokenizers_checked",
                    "CPU_probe_rows",
                    "CPU_reference_max_abs_logit_error",
                    "new_Qwen_calls",
                    "new_training_steps",
                    "new_broad_predictions",
                    "retrieval_probes",
                ]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--write", action="store_true")
    main(p.parse_args().write)

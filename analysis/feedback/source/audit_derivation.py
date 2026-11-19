from src.hsrag.paths import PROJECT_ROOT

import ast

import hashlib

import json

from pathlib import Path

ROOT = PROJECT_ROOT

BASE = ROOT / "experiments/astra/sq3_full_kb_feedback_20260928_v1/baseline_api_v1"


def auditor_source(original):
    result = original.replace("astra_sq1_pool_api_v1", "astra_sq3_feedback_api_v1")
    result = result.replace(
        "astra_sq1_pool_stage_check_v1", "astra_sq3_feedback_stage_check_v1"
    )
    result = result.replace(
        "SQ1 sampled-pool Qwen stage", "SQ3 full-KB baseline Qwen stage"
    )
    result = result.replace(
        "sq1_pool_api_{stage_id}_audit_", "sq3_feedback_baseline_{stage_id}_audit_"
    )
    result = result.replace(
        "completed_stage_verified_next_stage_ready",
        "baseline_verified_selection_review_pending",
    )
    result = result.replace(
        "    # Exercise the actual completed-stage path",
        "    print('Raw response audit finished. Verifying complete-stage restart and source hashes; no new calls.', flush=True)\n    # Exercise the actual completed-stage path",
    )
    return result

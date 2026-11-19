from pathlib import Path

import os

from analysis.feedback.rounds import resources_inputs as a

LOG = a.ROOT / "Thesis_Log/astra_result"

REPORT = "astra_sq3_iterative_part8_resources_20261027_v1.md"

SYNTHESIS = "astra_sq3_iterative_final_results_20261027_v1.md"

HANDOFF = "astra_sq3_iterative_writing_handoff_20261027_v1.md"

RESOURCE = "astra_resource_evidence_20261027_v8.md"

ROADMAP = "astra_remaining_work_before_writing_20261027_v98.md"

CHECKLIST = "astra_sq3_completion_checklist_20261027_v14.md"

PROGRESS = "astra_sq3_iterative_review_progress_20261027_v8.md"

REPORTS = {
    2: "astra_sq3_iterative_part2_errors_20261026_v1.md",
    3: "astra_sq3_iterative_part3_context_20261026_v1.md",
    4: "astra_sq3_iterative_part4_durability_20261026_v1.md",
    5: "astra_sq3_iterative_part5_sources_20261027_v1.md",
    6: "astra_sq3_iterative_part6_quality_20261027_v1.md",
    7: "astra_sq3_iterative_part7_robustness_20261027_v1.md",
    8: REPORT,
}

POP = {"en_test": "English TEST", "en_hatecheck_eval": "HateCheck"}

POL = {
    "reference": "Unchanged reference",
    "adaptive": "Current errors",
    "static": "Initial errors",
    "random": "Balanced random",
}


def extract_table(text, heading):
    start = text.index(heading)
    body = text[start:]
    lines = body.splitlines()
    out = []
    started = False
    for line in lines:
        if line.startswith("| "):
            out.append(line)
            started = True
        elif started:
            break
    a.need(out, "Missing source table " + heading)
    return "\n".join(out)

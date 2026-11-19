import hashlib

import json

from pathlib import Path

from analysis.feedback.summary import final_synthesis as s

REPORT = "astra_sq3_final_combined_synthesis_20261027_v1.md"

HANDOVER = "astra_sq3_final_writing_handover_20261027_v1.md"

ROAD = "astra_remaining_work_before_writing_20261027_v102.md"

CHECK = "astra_sq3_completion_checklist_20261027_v18.md"

RESOURCE = "astra_resource_evidence_20261027_v9.md"

UPDATE = "WR-20261027-SQ3-FINAL-COMBINED-v1"


def section(text, title):
    return text.split("## " + title, 1)[1].split("\n## ", 1)[0]


def inventory(text):
    return [
        r.split("|")[1].strip()
        for r in section(
            text, "Full accepted inventory retained from the previous roadmap"
        ).splitlines()
        if r.startswith("| ")
    ][1:]

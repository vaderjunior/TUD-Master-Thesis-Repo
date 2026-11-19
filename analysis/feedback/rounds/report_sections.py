import re

from analysis.feedback.rounds import subgroups_inputs as a

LOG = a.ROOT / "Thesis_Log/astra_result"

PUB = a.OUT / "publication_v1"

REPORT = "astra_sq3_iterative_part5_sources_20261027_v1.md"

PROGRESS = "astra_sq3_iterative_review_progress_20261027_v5.md"

ROADMAP = "astra_remaining_work_before_writing_20261027_v95.md"

CHECKLIST = "astra_sq3_completion_checklist_20261027_v11.md"


def line(text, prefix, value):
    found = [r for r in text.splitlines() if r.startswith(prefix)]
    a.need(len(found) == 1, "Nonunique line: " + prefix)
    return replace(text, found[0], value)


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

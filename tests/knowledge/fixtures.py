"""Synthetic scoring setups; these are not historical gold rosters."""

from copy import deepcopy
import yaml
from src.hsrag.paths import project_path
from analysis.common import scoring
from analysis.knowledge.components import prepare as components
from analysis.knowledge.allocation import prepare as allocation
from analysis.knowledge.sampled_banks import analysis_plan as sampled


def inherited_roster():
    taxonomy = yaml.safe_load(project_path("config/taxonomy.yaml").read_text(encoding="utf-8"))
    endpoints = {}
    for subset, count in sampled.SCOPE_COUNTS.items():
        dimension, role = scoring.ENDPOINTS[subset]
        labels = (
            ["not_hate", "hate"]
            if dimension == "hate"
            else list(taxonomy["dimensions"][dimension]["labels"])
        )
        endpoints[subset] = {
            "subset": subset,
            "dimension": dimension,
            "role": role,
            "rows": count,
            "scorable_rows": count,
            "scorable_ids": [f"fixture-{i}" for i in range(count)],
            "all_labels": labels,
            "macro_labels": labels,
            "gold_support": {label: 10 for label in labels},
        }
    return {"endpoints": endpoints, "contract": scoring.contract()}


def configuration(study):
    if study == "components":
        return components.analysis_plan({"endpoints_inherited_masks_and_labels": {}})
    if study == "allocation":
        return allocation.analysis_plan({"endpoints_inherited_masks_and_labels": {}})
    if study == "sampled_banks":
        science = sampled.a.read(sampled.STUDY / "scientific_plan.json")
        return sampled.analysis_plan(science, inherited_roster(), "synthetic-scoring-input", {})
    raise ValueError(study)


def schedule_fixture(counts):
    """Four synthetic stages with the listed endpoint memberships."""
    return {
        "stages": [
            {
                "id": f"stage_{i:02d}",
                "jobs": [{"subset": name} for name, n in counts.items() if i <= n],
            }
            for i in range(1, 5)
        ]
    }

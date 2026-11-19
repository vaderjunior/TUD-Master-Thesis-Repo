"""Output labels and validation rules from the configured taxonomy."""

from enum import Enum
from pathlib import Path
from src.hsrag.paths import project_path

import yaml
from pydantic import BaseModel, ConfigDict, PrivateAttr, model_validator

TAXONOMY = project_path("config/taxonomy.yaml")


def _labels(tax: dict, dim: str) -> list[str]:
    labels = tax["dimensions"][dim]["labels"]
    return list(labels) if isinstance(labels, dict) else list(labels)

_tax = yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))

TargetGroup = Enum("TargetGroup", {l: l for l in _labels(_tax, "target_group")},
                   type=str)
HateType = Enum("HateType", {l: l for l in _labels(_tax, "hate_type")}, type=str)
Severity = Enum("Severity", {l: l for l in _labels(_tax, "severity")}, type=str)
Legal = Enum("Legal", {l: l for l in _labels(_tax, "legal")}, type=str)

SEVERITY_ORDER = {s: i for i, s in enumerate(_labels(_tax, "severity"))}

FIELD_ALIASES = {
    "target_groups": "target_group",
    "hate_types": "hate_type",
    "is_hate": "hate",
    "hateful": "hate",
    "legal_class": "legal",
    "legal_classes": "legal",
    "legals": "legal",
}

LIST_FIELDS = ("target_group", "hate_type", "legal")


class Result(BaseModel):
    """A validated classification with gate-consistent hate-specific fields."""

    model_config = ConfigDict(use_enum_values=True, extra="forbid")

    reasoning: str
    hate: bool
    target_group: list[TargetGroup] = []
    hate_type: list[HateType] = []
    severity: Severity | None = None
    legal: list[Legal] = []

    _gate_normalised: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def gate_consistency(self):
        """Clear targets, types and severity when hate is false; retain legal labels."""
        if not self.hate and (self.target_group or self.hate_type
                              or self.severity is not None):
            self._gate_normalised = True
            self.target_group = []
            self.hate_type = []
            self.severity = None
        return self

    @property
    def gate_normalised(self) -> bool:
        return self._gate_normalised


def normalise_raw(obj: dict) -> tuple[dict, int]:
    """Normalise field aliases and label spelling; unknown labels still fail validation."""
    n = 0
    out = {}

    for key, val in obj.items():
        k = str(key).strip().lower()
        canonical = FIELD_ALIASES.get(k, k)
        if canonical != key:
            n += 1
        out[canonical] = val

    def norm_label(v):
        nonlocal n
        if not isinstance(v, str):
            return v
        s = v.strip().lower().replace(" ", "_").replace("-", "_")
        if s != v:
            n += 1
        return s

    for field in LIST_FIELDS:
        val = out.get(field)
        if val is None:
            if field in out:
                n += 1
            out[field] = []
        elif isinstance(val, str):
            out[field] = [norm_label(val)]
            n += 1
        elif isinstance(val, list):
            out[field] = [norm_label(v) for v in val]

    if isinstance(out.get("severity"), str):
        s = norm_label(out["severity"])

        out["severity"] = None if s in {"none", "null", "n/a", ""} else s

    if isinstance(out.get("hate"), str):
        s = out["hate"].strip().lower()
        if s in {"true", "yes", "1"}:
            out["hate"], n = True, n + 1
        elif s in {"false", "no", "0"}:
            out["hate"], n = False, n + 1

    return out, n

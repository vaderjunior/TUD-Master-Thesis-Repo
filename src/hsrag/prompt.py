"""Render label rules and supplied knowledge into model messages."""

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from src.hsrag.paths import project_path

import yaml

from src.hsrag.retrieve import Hit

TEMPLATE = project_path("prompts/classify_v1.txt")
TAXONOMY = project_path("config/taxonomy.yaml")

_SECTION = re.compile(r"^=== (\w+) ===$", re.MULTILINE)


@dataclass
class PromptContext:
    """Definitions, guidelines and example groups supplied to the model."""
    definitions: list[Hit] = field(default_factory=list)
    guidelines: list[Hit] = field(default_factory=list)
    example_groups: list[tuple[str, list[Hit]]] = field(default_factory=list)

    @classmethod
    def zero_shot(cls) -> "PromptContext":
        return cls()

    @classmethod
    def from_retrieval(cls, res) -> "PromptContext":
        """Separate ordinary examples, feedback and legal illustrations."""
        regular = [h for h in res.examples if not h.meta.get("illustrative_only")]
        legal = [h for h in res.examples if h.meta.get("illustrative_only")]
        groups = []
        if regular:
            groups.append(("Labelled examples", regular))

        if getattr(res, "feedback", None):
            groups.append(("Additional labelled examples", res.feedback))
        if legal:
            groups.append(("Legal context illustrations", legal))
        return cls(definitions=res.definitions, guidelines=res.guidelines,
                   example_groups=groups)


def _load_template() -> dict[str, str]:
    """Split the template file on '=== NAME ===' markers."""
    raw = TEMPLATE.read_text(encoding="utf-8")
    parts = _SECTION.split(raw)

    return {parts[i]: parts[i + 1].strip() for i in range(1, len(parts), 2)}

def _flatten(text: str) -> str:
    """Collapse whitespace when rendering example texts."""
    return " ".join(str(text).split())

def prompt_version() -> str:
    """Return the template name and its byte hash."""
    h = hashlib.sha256(TEMPLATE.read_bytes()).hexdigest()[:8]
    return f"{TEMPLATE.stem}-{h}"


def taxonomy_version() -> str:
    """Return the taxonomy name and its byte hash."""
    h = hashlib.sha256(TAXONOMY.read_bytes()).hexdigest()[:8]
    return f"{TAXONOMY.stem}-{h}"


def render_label_space(tax: dict | None = None) -> str:
    """List the permitted labels for every output dimension."""
    tax = tax or yaml.safe_load(TAXONOMY.read_text(encoding="utf-8"))
    lines = []
    for dim_name, dim in tax["dimensions"].items():
        dtype = dim["type"]
        if dtype == "binary":
            lines.append(f"- {dim_name} ({dtype}): true or false")
            continue
        labels = dim["labels"]
        names = list(labels) if isinstance(labels, dict) else list(labels)
        lines.append(f"- {dim_name} ({dtype}): {', '.join(names)}")
    return "\n".join(lines)


def _render_gold(meta):
    """Render only observed annotations, including empty label sets."""
    bits = []
    gate = meta.get('gate')
    if gate is not None:
        if not isinstance(gate, bool):
            raise ValueError('Gate must be a boolean or unannotated')
        bits.append(f'hate={str(gate).lower()}')
    for key, label in (('target_groups', 'target_group'), ('hate_types', 'hate_type')):
        if meta.get(key) is not None:
            bits.append(f"{label}=[{', '.join(meta[key])}]")
    if meta.get('severity') is not None:
        bits.append(f"severity={meta['severity']}")
    if meta.get('legal') is not None:
        bits.append(f"legal=[{', '.join(meta['legal'])}]")
    text = ', '.join(bits)
    if meta.get('astra_aux_stgb'):
        text += '\n  Auxiliary DeTox paragraph flags (illustrative, not legal gold): §' + meta['astra_aux_stgb']
    return text


def build_prompt(text: str, lang: str, ctx: PromptContext) -> list[dict]:
    """Render the message list. No retrieval, no API, no I/O beyond templates."""
    tpl = _load_template()
    system = tpl["SYSTEM"].format(label_space=render_label_space())

    parts = []

    if ctx.definitions:
        parts.append("## Label definitions\n" + "\n".join(

            f"- {h.meta.get('dimension')}"
            + (f" / {h.meta['label']}" if h.meta.get("label") else "")
            + f": {_flatten(h.text)}"
            for h in ctx.definitions))

    if ctx.guidelines:
        parts.append("## Annotation guidelines\n" + "\n".join(
            f"- {h.text}" for h in ctx.guidelines))

    for name, hits in ctx.example_groups:
        if not hits:
            continue
        legal = any(h.meta.get("illustrative_only") for h in hits)

        static = any(h.via == "static" for h in hits)
        key = ("ILLUSTRATIONS_CAUTION" if legal
               else "STATIC_EXAMPLES_CAUTION" if static
               else "EXAMPLES_CAUTION")
        caution = tpl[key]
        rows = []
        for h in hits:
            if h.meta.get("illustrative_only"):

                tag = h.meta.get("stgb", "unspecified")
                rows.append(f'- "{_flatten(h.text)}"\n  flagged under StGB §{tag}')
            else:
                rows.append(f'- "{_flatten(h.text)}"\n  labels: {_render_gold(h.meta)}')
        parts.append(f"## {name}\n{caution}\n\n" + "\n".join(rows))

    parts.append(f"## Text to classify (language: {lang})\n{text}")

    return [
        {"role": "system", "text": system},
        {"role": "user", "text": "\n\n".join(parts)},
    ]

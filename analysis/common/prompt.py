"""Render the original and corrected saved-study prompt conditions."""

from src.hsrag.paths import PROJECT_ROOT
import ast
from functools import lru_cache
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

ROOT = PROJECT_ROOT


def corrected_gold(meta):
    bits = []
    gate = meta.get("gate")
    if gate is not None:
        if not isinstance(gate, bool):
            raise ValueError("Gate must be a boolean or unannotated")
        bits.append(f"hate={str(gate).lower()}")
    for key, label in (("target_groups", "target_group"), ("hate_types", "hate_type")):
        if meta.get(key) is not None:
            bits.append(f"{label}=[{', '.join(meta[key])}]")
    if meta.get("severity") is not None:
        bits.append(f"severity={meta['severity']}")
    if meta.get("legal") is not None:
        bits.append(f"legal=[{', '.join(meta['legal'])}]")
    text = ", ".join(bits)
    if meta.get("astra_aux_stgb"):
        text += (
            "\n  Auxiliary DeTox paragraph flags (illustrative, not legal gold): §"
            + meta["astra_aux_stgb"]
        )
    return text


def _legacy_gold(meta: dict) -> str:
    bits = [f"hate={str(bool(meta.get('gate'))).lower()}"]
    for key, out in [("target_groups", "target_group"), ("hate_types", "hate_type")]:
        val = meta.get(key)
        if val is not None:
            bits.append(f"{out}=[{', '.join(val)}]")
    if meta.get("severity") is not None:
        bits.append(f"severity={meta['severity']}")
    return ", ".join(bits)


@lru_cache(maxsize=2)
def module(variant):
    assert variant in ("legacy", "corrected")
    path = ROOT / "src/hsrag/prompt.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        n
        for n in tree.body
        if not (isinstance(n, ast.ImportFrom) and n.module == "src.hsrag.retrieve")
    ]
    name = "study_prompt_" + variant
    m = ModuleType(name)
    m.Hit = SimpleNamespace
    sys.modules[name] = m
    exec(compile(tree, str(path), "exec"), m.__dict__)
    m.TEMPLATE = ROOT / "prompts/classify_v1.txt"
    m.TAXONOMY = ROOT / "config/taxonomy.yaml"
    if variant == "corrected":
        m._render_gold = corrected_gold
    else:
        m._render_gold = _legacy_gold
    return m


def render_messages(text, lang, context, variant):
    m = module(variant)
    hit = lambda r: SimpleNamespace(**r)
    ctx = m.PromptContext(
        definitions=[hit(r) for r in context["definitions"]],
        guidelines=[hit(r) for r in context["guidelines"]],
        example_groups=[(name, [hit(r) for r in rows]) for name, rows in context["example_groups"]],
    )
    return m.build_prompt(text, lang, ctx)


def from_retrieval(res):

    def hit(h):
        return {"id": h.id, "text": h.text, "meta": h.meta, "via": h.via}

    regular = [hit(h) for h in res.examples if not h.meta.get("illustrative_only")]
    illustrations = [hit(h) for h in res.examples if h.meta.get("illustrative_only")]
    groups = []
    if regular:
        groups.append(["Labelled examples", regular])
    if illustrations:
        groups.append(["Legal context illustrations", illustrations])
    return {
        "definitions": [hit(h) for h in res.definitions],
        "guidelines": [hit(h) for h in res.guidelines],
        "example_groups": groups,
    }


def hit_ids(context):
    return {
        "definitions": [r["id"] for r in context["definitions"]],
        "guidelines": [r["id"] for r in context["guidelines"]],
        **{name: [r["id"] for r in rows] for name, rows in context["example_groups"]},
    }

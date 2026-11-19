"""Classify a text and keep the sampled responses and source versions."""

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from src.hsrag.paths import PROJECT_ROOT
import threading
import time

from src.hsrag.parse import RunResult, run_once
from src.hsrag.prompt import (PromptContext, build_prompt, prompt_version,
                              taxonomy_version)
from src.hsrag.retrieve import Hit
from src.hsrag.schema import Result
from src.hsrag.vote import aggregate

ARMS = ("zero_shot", "few_shot", "rag")


def text_hash(text: str) -> str:
    """Hash text after collapsing whitespace."""
    return hashlib.sha256(" ".join(str(text).split()).encode()).hexdigest()[:16]


@lru_cache(maxsize=1)
def code_version() -> str:
    """Hash library paths and bytes once per process."""
    h = hashlib.sha256()
    for p in sorted((PROJECT_ROOT / "src/hsrag").rglob("*.py")):
        h.update(p.relative_to(PROJECT_ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:8]


@dataclass
class ItemResult:
    """A prediction, sampled responses and provenance for one text."""

    item_id: str
    text_hash: str
    lang: str
    arm: str

    timestamp: float = 0.0
    workers: int = 1
    temperature: float | None = None

    active_model: str | None = None
    prompt_version: str | None = None

    taxonomy_version: str | None = None

    code_version: str | None = None
    kb_version: str | None = None
    retrieval_config: dict | None = None
    retrieved: dict = field(default_factory=dict)

    result: dict | None = None
    uncertain: bool = False
    n_valid: int = 0
    n_runs: int = 0
    agreement: dict = field(default_factory=dict)

    parse_failures: int = 0
    repairs: int = 0
    normalisations: int = 0
    gate_normalised: int = 0

    encoder_meta: dict | None = None

    raw_runs: list = field(default_factory=list)
    run_predictions: list = field(default_factory=list)
    latencies: list = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


class Arms:
    """Build zero-shot, fixed-example or retrieved prompt contexts."""

    def __init__(self, retriever=None, records_path: Path | None = None,
                 k_examples: int = 5, seed: int = 42):
        self.retriever = retriever
        self.k_examples = k_examples
        self._fewshot: dict[str, list[Hit]] = {}
        if records_path is not None:
            self._build_fewshot(Path(records_path), seed)

        self._lock = threading.Lock()

    def _build_fewshot(self, records_path: Path, seed: int) -> None:
        """Sample a fixed eligible example set for each language."""
        records = [json.loads(line) for line in
                   records_path.read_text(encoding="utf-8").splitlines()
                   if line.strip()]
        for lang in ("en", "de"):
            pool = [r for r in records
                    if r["kind"] == "example" and r["lang"] == lang
                    and not r.get("meta", {}).get("illustrative_only")]
            rng = random.Random(seed)
            picked = rng.sample(pool, min(self.k_examples, len(pool)))
            self._fewshot[lang] = [
                Hit(id=r["id"], kind=r["kind"], text=r["text"],
                    meta={**r.get("meta", {}), "lang": r["lang"],
                          "source": r["source"], "dimension": r["dimension"],
                          "label": r["label"], "kind": r["kind"]},
                    score=0.0, via="static")
                for r in picked
            ]

    def context(self, arm: str, text: str, lang: str) -> PromptContext:
        if arm == "zero_shot":
            return PromptContext.zero_shot()
        if arm == "few_shot":
            hits = self._fewshot.get(lang, [])
            return PromptContext(
                example_groups=[("Labelled examples", hits)] if hits else [])
        if arm == "rag":
            if self.retriever is None:
                raise ValueError("rag arm needs a retriever")
            with self._lock:
                res = self.retriever.retrieve(text, lang)
            return PromptContext.from_retrieval(res)
        raise ValueError(f"unknown arm: {arm}")


def _hit_ids(ctx: PromptContext) -> dict:
    """Record the identifiers included in each prompt section."""
    out = {"definitions": [h.id for h in ctx.definitions],
           "guidelines": [h.id for h in ctx.guidelines]}
    for name, hits in ctx.example_groups:
        out[name] = [h.id for h in hits]
    return out


def classify(item_id: str, text: str, lang: str, arm: str, client, arms: Arms,
             n_votes: int = 3, max_repairs: int = 2,
             pinned_model: str | None = None,
             kb_version: str | None = None, workers: int = 1, temperature: float | None = None) -> ItemResult:
    """Sample responses, repair invalid outputs and aggregate accepted predictions."""
    if n_votes < 1 or max_repairs < 0:
        raise ValueError("votes must be positive and repairs must be non-negative")
    ctx = arms.context(arm, text, lang)
    messages = build_prompt(text, lang, ctx)

    out = ItemResult(
        item_id=item_id, text_hash=text_hash(text), lang=lang, arm=arm,
        prompt_version=prompt_version(),
        taxonomy_version=taxonomy_version(),
        code_version=code_version(),

        kb_version=kb_version if arm == "rag" else None,
        retrieval_config=dict(arms.retriever.cfg) if arm == "rag" else None,
        retrieved=_hit_ids(ctx),
        n_runs=n_votes,
        timestamp=time.time(),
        workers=workers,
        temperature=temperature,
    )

    runs: list[RunResult] = []
    for _ in range(n_votes):
        r = run_once(client, messages, max_repairs=max_repairs)
        runs.append(r)
        out.raw_runs.append(r.raw_outputs)
        out.latencies.append(round(r.latency_s, 2))
        out.repairs += r.repairs
        out.normalisations += r.normalisations
        out.gate_normalised += int(r.gate_normalised)
        out.parse_failures += int(r.parse_failure)
        out.run_predictions.append(
            r.result.model_dump() if r.result is not None else None)

    if pinned_model is not None and client.active_model != pinned_model:

        raise RuntimeError(
            f"expected pinned model {pinned_model}, got {client.active_model}")
    out.active_model = client.active_model

    vote = aggregate(runs)
    out.result = vote.result.model_dump() if vote.result is not None else None
    out.uncertain = vote.uncertain
    out.n_valid = vote.n_valid
    out.agreement = vote.agreement
    return out

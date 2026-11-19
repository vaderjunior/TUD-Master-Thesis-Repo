"""Resolve experiment settings and save the configuration with results."""

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from src.hsrag.paths import project_path

import pandas as pd
import yaml

MANIFESTS = project_path("experiments/manifests")
PROCESSED = project_path("data/processed")
CONFIG = project_path("config/config.yaml")

VALID_ARMS = {"zero_shot", "few_shot", "rag"}
VALID_GATE_MAPPING = {"strict", "lenient", "both"}

GERMAN_SAFE_PROVIDERS = {"tudagpt", "peasec", "mock"}

SECONDS_PER_CALL = {"tudagpt": 17.0, "peasec": 6.5, "mock": 0.01}
DEFAULT_SECONDS_PER_CALL = 17.0


@dataclass
class Manifest:
    name: str
    subset: str
    arms: list[str]
    provider: str = "tudagpt"
    pinned_model: str | None = None
    allow_fallback: bool = False
    temperature: float | None = None
    n_votes: int = 3
    max_repair_retries: int = 2
    workers: int = 4
    limit: int | None = None
    seed: int = 42
    description: str = ""

    retrieval: dict | None = None

    records_path: str | None = None
    chroma_path: str | None = None

    gate_mapping: str = "both"
    hate_type_scoring: str = "multilabel"

    kb_version: str | None = None
    prompt_version: str | None = None
    n_items: int = 0
    lang: str = ""

    def validate(self) -> None:
        bad = set(self.arms) - VALID_ARMS
        assert not bad, f"unknown arms: {bad}"
        assert self.arms, "manifest has no arms"
        assert self.gate_mapping in VALID_GATE_MAPPING, self.gate_mapping
        assert self.n_votes >= 1, "n_votes must be at least 1"

        path = PROCESSED / f"{self.subset}.parquet"
        assert path.exists(), f"subset not found: {path}"

        assert (self.records_path is None) == (self.chroma_path is None), (
            "records_path and chroma_path must be set together: the jsonl "
            "feeds BM25 and Chroma feeds the dense channel, so setting one "
            "alone gives a retriever whose two channels disagree about which "
            "knowledge base they are querying")

        if self.lang == "de":
            assert self.provider in GERMAN_SAFE_PROVIDERS, (
                f"provider '{self.provider}' may not receive German data "
                f"(DeTox Zenodo terms). Allowed: {sorted(GERMAN_SAFE_PROVIDERS)}")

        if "_test_" in self.subset:
            print(f"  !! WARNING: {self.subset} is a TEST subset. "
                  f"Tuning must happen on dev.")

    @property
    def calls(self) -> int:
        return self.n_items * len(self.arms) * self.n_votes

    def estimate(self, seconds_per_call: float | None = None) -> dict:
        s = seconds_per_call or SECONDS_PER_CALL.get(
            self.provider, DEFAULT_SECONDS_PER_CALL)
        wall = self.calls * s
        return {"calls": self.calls, "seconds": wall, "minutes": wall / 60,
                "hours": wall / 3600, "seconds_per_call": s}

    def hash(self) -> str:
        """Hash the resolved settings, excluding the name and description."""
        payload = {k: v for k, v in asdict(self).items()
                   if k not in ("description", "name")}
        blob = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    def header(self) -> str:
        """First line of the results file: a self-describing result set."""
        return json.dumps({"_manifest": asdict(self), "_hash": self.hash()},
                          ensure_ascii=False)


def load(name_or_path: str) -> Manifest:
    path = Path(name_or_path)
    if not path.exists():
        path = MANIFESTS / f"{name_or_path}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return Manifest(**raw)


def resolve(m: Manifest, retriever=None) -> Manifest:
    """Fill gaps from config.yaml and stamp run-time facts."""
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))

    if m.temperature is None:
        m.temperature = cfg["api"]["temperature"]
    if m.pinned_model is None:

        m.pinned_model = (cfg["api"][m.provider]["default_model"]
                          if m.provider in cfg["api"]
                          else cfg["classify"]["pinned_model"])
    if m.retrieval is None and "rag" in m.arms:
        m.retrieval = dict(cfg["retrieval"])

    df = pd.read_parquet(PROCESSED / f"{m.subset}.parquet")
    if m.limit:
        df = df.head(m.limit)
    m.n_items = len(df)
    langs = set(df["lang"])
    assert len(langs) == 1, f"subset mixes languages: {langs}"
    m.lang = langs.pop()

    from src.hsrag.prompt import prompt_version
    m.prompt_version = prompt_version()

    if "rag" in m.arms and retriever is not None:
        m.kb_version = retriever.col.metadata.get("kb_version")

    m.validate()
    return m


def describe(m: Manifest, seconds_per_call: float = DEFAULT_SECONDS_PER_CALL) -> str:
    e = m.estimate(seconds_per_call)
    lines = [
        f"manifest      {m.name}  [{m.hash()}]",
        f"  {m.description}" if m.description else "",
        f"subset        {m.subset}  ({m.n_items} items, lang={m.lang})",
        f"arms          {', '.join(m.arms)}",
        f"model         {m.provider} / {m.pinned_model}  "
        f"fallback={'ON' if m.allow_fallback else 'off'}  T={m.temperature}",
        f"sampling      n_votes={m.n_votes}  repairs<={m.max_repair_retries}  "
        f"workers={m.workers}",
        f"prompt        {m.prompt_version}",
        f"kb            {m.kb_version or '(not used)'}",
        f"retrieval     {m.retrieval if m.retrieval else '(not used)'}",
        f"scoring       gate_mapping={m.gate_mapping}  "
        f"hate_type={m.hate_type_scoring}",
        "",
        f"COST          {e['calls']} calls "
        f"({m.n_items} items x {len(m.arms)} arms x {m.n_votes} votes)",
        f"              ~{e['minutes']:.0f} min ({e['hours']:.1f} h) "
        f"at {e['seconds_per_call']:g} s/call, wall-clock at "
        f"workers={m.workers}",
        f"              measured: tudagpt 13-17 s/call at 4 workers, "
        f"peasec 6.5 s/call at 8 workers (2.0 for the 8B)",
    ]
    return "\n".join(l for l in lines if l)

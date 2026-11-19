"""Check retrieval budgets, metadata and selected probe cases."""

from pathlib import Path

import pytest
import yaml

from src.hsrag.retrieve import Retriever

pytestmark = pytest.mark.skipif(
    not Path("kb/records.jsonl").is_file() or not Path("kb/chroma").is_dir(),
    reason="Prepare knowledge records and the retrieval index for these checks",
)

PROBES = {
    "misogyny": ("women are too stupid to vote", "en"),
    "religion": ("muslims are terrorists and should be banned", "en"),
    "german": ("Ausländer raus, wir wollen euch hier nicht", "de"),
    "profanity": ("you're such an idiot, get lost", "en"),
    "quoted_slur": ("he called me the n-word and I was shocked", "en"),
    "neutral": ("the weather is lovely for a walk today", "en"),
}


@pytest.fixture(scope="module")
def cfg_all():
    return yaml.safe_load(Path("config/config.yaml").read_text(encoding="utf-8"))

_CACHE: dict[str, Retriever] = {}


def _retriever(cfg_all, strategy):
    """Memoised: parametrised tests would otherwise load BGE-M3 16 times."""
    if strategy not in _CACHE:
        cfg = dict(cfg_all["retrieval"])
        cfg["strategy"] = strategy
        kb = cfg_all["kb"]
        _CACHE[strategy] = Retriever(
            chroma_path=Path(kb["chroma_path"]),
            records_path=Path(kb["records_path"]),
            model_name=kb["embedding_model"],
            cfg=cfg,
        )
    return _CACHE[strategy]


@pytest.fixture(scope="module")
def retriever_hybrid(cfg_all):
    return _retriever(cfg_all, "hybrid")


@pytest.fixture(scope="module")
def retriever_dense(cfg_all):
    return _retriever(cfg_all, "dense")


@pytest.mark.parametrize("probe", list(PROBES))
@pytest.mark.parametrize("strategy", ["dense", "hybrid"])
def test_every_bucket_is_filled(cfg_all, probe, strategy):
    """Dense and hybrid retrieval should fill each configured section."""
    text, lang = PROBES[probe]
    res = _retriever(cfg_all, strategy).retrieve(text, lang)
    assert len(res.definitions) == cfg_all["retrieval"]["k_definitions"]
    assert len(res.guidelines) == cfg_all["retrieval"]["k_guidelines"]
    assert len(res.examples) == cfg_all["retrieval"]["k_examples"]


def test_example_language_filter(retriever_hybrid):
    """Examples should match the query language."""
    for probe, (text, lang) in PROBES.items():
        res = retriever_hybrid.retrieve(text, lang)
        assert all(h.meta["lang"] == lang for h in res.examples), probe
        assert all(h.meta["lang"] == "en" for h in res.definitions), probe


def test_no_duplicate_text_within_a_bucket(retriever_hybrid):
    """Repeated texts should not occupy multiple example slots."""
    for probe, (text, lang) in PROBES.items():
        res = retriever_hybrid.retrieve(text, lang)
        for bucket in (res.definitions, res.guidelines, res.examples):
            texts = [" ".join(h.text.lower().split()) for h in bucket]
            assert len(texts) == len(set(texts)), probe


def test_sentinels_are_reversed(retriever_hybrid):
    """Missing values and observed label lists should survive indexing."""
    text, lang = PROBES["misogyny"]
    for hit in retriever_hybrid.retrieve(text, lang).all_hits():
        for key, val in hit.meta.items():
            assert val != "__none__", f"{hit.id}.{key}"
        tg = hit.meta.get("target_groups")
        assert tg is None or isinstance(tg, list), hit.id


def test_mmr_vectors_do_not_leak(retriever_hybrid):
    """Return prompt records without temporary MMR vectors."""
    text, lang = PROBES["misogyny"]
    assert all(h.vec is None
               for h in retriever_hybrid.retrieve(text, lang).all_hits())


@pytest.mark.parametrize("strategy", ["dense", "hybrid"])
def test_religion_definition_retrieved(cfg_all, strategy):
    """Hit at rank 1 under dense, bm25 and hybrid."""
    text, lang = PROBES["religion"]
    res = _retriever(cfg_all, strategy).retrieve(text, lang)
    assert any(h.id == "def-target_group-religion" for h in res.definitions)


@pytest.mark.parametrize("strategy", ["dense", "hybrid"])
def test_profanity_guideline_retrieved(cfg_all, strategy):
    """Check the recorded profanity guideline case."""
    text, lang = PROBES["profanity"]
    res = _retriever(cfg_all, strategy).retrieve(text, lang)
    assert any(h.id == "guide-profanity-without-target" for h in res.guidelines)


def test_misogyny_retrieves_gender_example(retriever_dense):
    """Check that the query reaches a gender example."""
    text, lang = PROBES["misogyny"]
    res = retriever_dense.retrieve(text, lang)
    tgs = [t for h in res.examples for t in (h.meta.get("target_groups") or [])]
    assert "gender" in tgs


def test_german_gets_english_knowledge_cross_lingually(retriever_hybrid):
    """Check cross-language guidance and German examples."""
    text, lang = PROBES["german"]
    res = retriever_hybrid.retrieve(text, lang)
    assert res.definitions and res.guidelines
    assert all(h.meta["lang"] == "de" for h in res.examples)
    assert all(h.meta["lang"] == "en" for h in res.definitions)


def test_neutral_probe_less_similar_than_hate(retriever_dense):
    """Compare dense example similarities for neutral and hateful probes."""
    hate, _ = PROBES["religion"]
    neutral, _ = PROBES["neutral"]
    best_hate = retriever_dense.retrieve(hate, "en").examples[0].score
    best_neutral = retriever_dense.retrieve(neutral, "en").examples[0].score
    assert best_neutral < best_hate

def test_empty_list_metadata_survives_chroma(retriever_hybrid):
    """Preserve an observed empty list through indexing."""
    res = retriever_hybrid.retrieve("Deutschland erwache", "de")
    legals = [h.meta.get("legal") for h in res.examples
              if "legal" in h.meta]
    assert legals, "no BoTox example retrieved; is the KB rebuilt?"
    for v in legals:
        assert v is None or isinstance(v, list), f"bad legal value: {v!r}"
        assert v != "[]", "empty list came back as a string"

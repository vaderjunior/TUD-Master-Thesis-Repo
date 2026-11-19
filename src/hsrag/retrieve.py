"""Retrieve definitions, guidelines and examples with separate budgets."""

from collections import Counter
from dataclasses import dataclass, field

from pathlib import Path

from src.hsrag.kb import load_collection, unsentinel_meta

import json
import re


@dataclass
class Hit:
    """One knowledge record and its retrieval provenance."""

    id: str
    kind: str
    text: str
    meta: dict
    score: float
    via: str
    vec: list[float] | None = None


@dataclass
class RetrievalResult:
    """Retrieved records grouped by their prompt role."""

    definitions: list[Hit] = field(default_factory=list)
    guidelines: list[Hit] = field(default_factory=list)
    examples: list[Hit] = field(default_factory=list)

    feedback: list[Hit] = field(default_factory=list)

    def all_hits(self) -> list[Hit]:
        """Flat view, for logging and the debug CLI."""
        return self.definitions + self.guidelines + self.examples + self.feedback

    def __len__(self) -> int:
        return len(self.all_hits())


def tokenize(text: str) -> list[str]:
    """Return lowercase word tokens for lexical retrieval."""
    return re.findall(r"\w+", text.lower())


def rrf_fuse(rank_lists: dict[str, list[Hit]], k: int = 60, top_n: int = 5) -> list[Hit]:
    """Combine channel positions with reciprocal-rank fusion; ties use record IDs."""
    scores: dict[str, float] = {}
    hits_by_id: dict[str, Hit] = {}
    ranks_by_id: dict[str, dict[str, int]] = {}

    for channel, hits in rank_lists.items():
        for rank, hit in enumerate(hits, start=1):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k + rank)
            hits_by_id.setdefault(hit.id, hit)
            ranks_by_id.setdefault(hit.id, {})[channel] = rank

    fused = []

    for hit_id, score in sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_n]:
        src = hits_by_id[hit_id]
        fused.append(
            Hit(
                id=src.id,
                kind=src.kind,
                text=src.text,
                meta={**src.meta, "_ranks": ranks_by_id[hit_id]},
                score=score,
                via="rrf",
            )
        )
    return fused


def mmr_select(qvec: list[float], hits: list[Hit], lam: float = 0.7, top_n: int = 5) -> list[Hit]:
    """Choose dense candidates by relevance and redundancy, then discard vectors."""
    import numpy as np

    if not hits:
        return []
    if len(hits) <= top_n or any(h.vec is None for h in hits):
        for h in hits:
            h.vec = None
        return hits[:top_n]

    q = np.asarray(qvec, dtype=float).ravel()
    q = q / (np.linalg.norm(q) or 1.0)

    V = np.asarray([h.vec for h in hits], dtype=float)
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)

    rel = V @ q
    sim = V @ V.T

    selected = [int(np.argmax(rel))]
    while len(selected) < top_n:
        best_i, best_score = None, -np.inf
        for i in range(len(hits)):
            if i in selected:
                continue
            redundancy = max(sim[i][j] for j in selected)
            score = lam * rel[i] - (1.0 - lam) * redundancy
            if score > best_score:
                best_score, best_i = score, i
        selected.append(best_i)

    out = [hits[i] for i in selected]
    for h in out:
        h.vec = None
    return out


class Retriever:
    """Reuse the model/index and apply separate budgets to each record kind."""

    def __init__(
        self,
        chroma_path: Path,
        records_path: Path,
        model_name: str,
        cfg: dict,
        *,
        encoder=None,
        collection=None,
    ):
        from rank_bm25 import BM25Okapi

        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(model_name)
        self.model = encoder
        self.col = (
            collection if collection is not None else load_collection(chroma_path, model_name)
        )
        self.cfg = cfg

        self._bm25 = {}
        self._bm25_records = {}

        records = [
            json.loads(line)
            for line in Path(records_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        buckets = {}

        self._def_counts = Counter()
        for r in records:

            if r["kind"] in ("example", "feedback"):
                key = f"{r['kind']}:{r['lang']}"
            else:
                key = r["kind"]
            buckets.setdefault(key, []).append(r)
            if r["kind"] == "definition":
                dim = r.get("dimension") or "(none)"
                self._def_counts[dim] += 1

                buckets.setdefault(f"definition:{dim}", []).append(r)

        for key, recs in buckets.items():
            self._bm25_records[key] = recs
            self._bm25[key] = BM25Okapi([tokenize(r["text"]) for r in recs])

    def _definition_specs(self, cfg: dict) -> list[tuple[dict, str, int]]:
        """Return definition filters and quotas in stored dimension order."""
        mode = cfg.get("k_definitions_mode", "flat")
        if mode == "flat":
            return [({"kind": "definition"}, "definition", cfg["k_definitions"])]

        rate = cfg.get("k_definitions_per_label", 0.3)
        flat_k = cfg.get("k_definitions_per_dimension", 1)
        out = []
        for dim, n in self._def_counts.items():

            k = flat_k if mode == "per_dimension" else max(1, round(n * rate))
            out.append(
                (
                    {"$and": [{"kind": "definition"}, {"dimension": dim}]},
                    f"definition:{dim}",
                    int(k),
                )
            )
        if not out:
            raise ValueError(
                f"k_definitions_mode={mode!r} but the KB holds no definitions "
                f"with a `dimension` field; nothing to budget"
            )
        return out

    def _embed(self, text: str) -> list[list[float]]:
        return self.model.encode([text], normalize_embeddings=True).tolist()

    def _query_kind(
        self, qvec, where: dict, k: int, via: str, want_vecs: bool = False
    ) -> list[Hit]:
        """Read filtered dense hits, optionally including vectors for MMR."""
        if k <= 0:
            return []

        include = ["metadatas", "documents", "distances"]
        if want_vecs:
            include = include + ["embeddings"]

        res = self.col.query(query_embeddings=qvec, n_results=k, where=where, include=include)

        embs = res.get("embeddings") if want_vecs else None

        hits = []
        for i in range(len(res["ids"][0])):
            meta = unsentinel_meta(res["metadatas"][0][i])
            hits.append(
                Hit(
                    id=res["ids"][0][i],
                    kind=meta["kind"],
                    text=res["documents"][0][i],
                    meta=meta,
                    score=1.0 - res["distances"][0][i],
                    via=via,
                    vec=(list(embs[0][i]) if embs is not None else None),
                )
            )
        return hits

    def _bm25_kind(self, query: str, bucket_key: str, k: int) -> list[Hit]:
        """BM25 within one bucket."""
        if k <= 0 or bucket_key not in self._bm25:
            return []

        scores = self._bm25[bucket_key].get_scores(tokenize(query))
        recs = self._bm25_records[bucket_key]

        ranked = sorted(zip(scores, recs), key=lambda x: -x[0])[:k]

        hits = []
        for score, r in ranked:
            if score <= 0:
                continue
            meta = {
                "kind": r["kind"],
                "lang": r["lang"],
                "source": r["source"],
                "dimension": r["dimension"],
                "label": r["label"],
                **r.get("meta", {}),
            }
            hits.append(
                Hit(
                    id=r["id"],
                    kind=r["kind"],
                    text=r["text"],
                    meta=meta,
                    score=float(score),
                    via="bm25",
                )
            )
        return hits

    def retrieve(self, text: str, lang: str = "en") -> RetrievalResult:
        cfg = self.cfg
        strategy = cfg.get("strategy", "dense")

        if strategy == "dense":
            qvec = self._embed(text)
            ex_where = {"$and": [{"kind": "example"}, {"lang": lang}]}
            k_ex = cfg["k_examples"]

            if cfg.get("use_mmr", False):
                pool = cfg.get("mmr_pool_factor", 3) * k_ex
                cands = self._query_kind(qvec, ex_where, pool, "dense", want_vecs=True)
                examples = mmr_select(qvec, cands, cfg.get("mmr_lambda", 0.7), k_ex)
            else:
                examples = self._query_kind(qvec, ex_where, k_ex, "dense")

            definitions = []
            for _where, _bm, _k in self._definition_specs(cfg):
                definitions += self._query_kind(qvec, _where, _k, "dense")

            return RetrievalResult(
                definitions=definitions,
                guidelines=self._query_kind(
                    qvec, {"kind": "guideline"}, cfg["k_guidelines"], "dense"
                ),
                examples=examples,
                feedback=self._query_kind(
                    qvec,
                    {"$and": [{"kind": "feedback"}, {"lang": lang}]},
                    cfg.get("k_examples_feedback", 0),
                    "dense",
                ),
            )

        if strategy == "bm25":
            definitions = []
            for _where, _bm, _k in self._definition_specs(cfg):
                definitions += self._bm25_kind(text, _bm, _k)
            return RetrievalResult(
                definitions=definitions,
                guidelines=self._bm25_kind(text, "guideline", cfg["k_guidelines"]),
                examples=self._bm25_kind(text, f"example:{lang}", cfg["k_examples"]),
                feedback=self._bm25_kind(
                    text, f"feedback:{lang}", cfg.get("k_examples_feedback", 0)
                ),
            )
        if strategy == "hybrid":
            qvec = self._embed(text)
            rrf_k = cfg.get("rrf_k", 60)

            depth = 10

            def_specs = self._definition_specs(cfg)
            buckets = {}
            for bucket, where, bm25_key, k_out in [
                (f"definitions::{i}", w, b, k) for i, (w, b, k) in enumerate(def_specs)
            ] + [
                ("guidelines", {"kind": "guideline"}, "guideline", cfg["k_guidelines"]),
                (
                    "examples",
                    {"$and": [{"kind": "example"}, {"lang": lang}]},
                    f"example:{lang}",
                    cfg["k_examples"],
                ),
                (
                    "feedback",
                    {"$and": [{"kind": "feedback"}, {"lang": lang}]},
                    f"feedback:{lang}",
                    cfg.get("k_examples_feedback", 0),
                ),
            ]:

                if k_out <= 0:
                    buckets[bucket] = []
                    continue

                if bucket == "examples" and cfg.get("use_mmr", False):
                    pool = cfg.get("mmr_pool_factor", 3) * depth
                    cands = self._query_kind(qvec, where, pool, "dense", want_vecs=True)
                    dense_hits = mmr_select(qvec, cands, cfg.get("mmr_lambda", 0.7), depth)
                else:
                    dense_hits = self._query_kind(qvec, where, depth, "dense")
                bm25_hits = self._bm25_kind(text, bm25_key, depth)
                buckets[bucket] = rrf_fuse(
                    {"dense": dense_hits, "bm25": bm25_hits},
                    k=rrf_k,
                    top_n=k_out,
                )

            definitions = []
            for i in range(len(def_specs)):
                definitions.extend(buckets.pop(f"definitions::{i}", []))
            buckets["definitions"] = definitions

            return RetrievalResult(**buckets)

        raise ValueError(f"Unknown strategy: {strategy}")

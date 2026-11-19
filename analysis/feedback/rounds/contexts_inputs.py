import os

for name in [
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ[name] = "1"
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from rank_bm25 import BM25Okapi
from analysis.feedback.rounds import reader as previous
from analysis.feedback.transfer import contexts_metrics as context

ROOT, STUDY, REVIEW = previous.ROOT, previous.STUDY, previous.REVIEW
PART2, READER = previous.OUT, previous.READER
OUT = REVIEW / "part3"
CAND = ROOT / "experiments/astra/sq3_additions_20261004_v1/benign_selection"
SCOPE = ROOT / "experiments/astra/sq3_followups_20261012_v1/scope"
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
LATER = ROOT / "experiments/astra/later_evaluation_20261008_v1"
STREAMS, POLICIES, CONDITIONS = previous.STREAMS, previous.POLICIES, previous.CONDITIONS
need, sha, read, rows, rel, physical, pure = (
    previous.need,
    previous.sha,
    previous.read,
    previous.rows,
    previous.rel,
    previous.physical,
    previous.pure,
)
FIELDS = context.FIELDS


def native(value):
    if isinstance(value, np.ndarray):
        return native(value.tolist())
    if isinstance(value, np.generic):
        return native(value.item())
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def write(path, value):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside new Part 3")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(
            value
            if isinstance(value, bytes)
            else pure.encoded(native(value), pretty=True)
        )


def lines(path, values):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside new Part 3")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        for value in values:
            stream.write(pure.encoded(native(value)))


def rank(seed, rnd, role, gold, source_id):
    return hashlib.sha256(
        f"A17|{seed}|{rnd}|{role}|{int(gold)}|{source_id}".encode()
    ).hexdigest()


def majority(values):
    need(
        len(values) == 3 and all(v is None or type(v) is bool for v in values),
        "Wrong screening repeats",
    )
    return None if None in values else sum(values) >= 2


def selection(candidates, initial, current, prior, seed, rnd):
    valid = [
        r
        for r in candidates
        if majority(initial[r["id"]]) is not None
        and majority(current[r["id"]]) is not None
    ]
    groups = {p: {g: [] for g in [False, True]} for p in POLICIES}
    for row in valid:
        sid, gold = row["id"], row["gold"]["hate"]
        for p, prediction in [
            ("adaptive", majority(current[sid])),
            ("static", majority(initial[sid])),
            ("random", None),
        ]:
            if p == "random" or prediction != gold:
                groups[p][gold].append(sid)
    budget = min(
        5, *(len(groups[p][g]) for p in ["adaptive", "static"] for g in [False, True])
    )
    additions = {}
    for p in POLICIES:
        additions[p] = sorted(
            sid
            for g in [False, True]
            for sid in sorted(
                groups[p][g],
                key=lambda sid: rank(
                    seed, rnd, "random" if p == "random" else "mistake", g, sid
                ),
            )[:budget]
        )
    return {
        "per_class_budget": budget,
        "eligible_ids": sorted(r["id"] for r in valid),
        "invalid_ids": sorted(
            set(r["id"] for r in candidates) - set(r["id"] for r in valid)
        ),
        "eligible_counts": {
            p: {str(g): len(groups[p][g]) for g in [False, True]} for p in POLICIES
        },
        "additions": additions,
        "cumulative": {p: sorted(prior[p] + additions[p]) for p in POLICIES},
    }


class ReplayBucket:
    """Replay the frozen small retriever using saved vectors and pinned BM25."""

    def __init__(self, records, vectors):
        order = sorted(range(len(records)), key=lambda j: records[j]["id"])
        self.records = [records[j] for j in order]
        self.ids = [r["id"] for r in self.records]
        need(
            1 <= len(self.ids) <= 40 and len(set(self.ids)) == len(self.ids),
            "Wrong pool",
        )
        matrix = np.asarray(vectors, dtype=np.float64)[order]
        self.vectors = matrix / np.linalg.norm(matrix, axis=1)[:, None]
        self.bm25 = BM25Okapi(
            [re.findall(r"\w+", r["text"].lower()) for r in self.records],
            k1=1.5,
            b=0.75,
            epsilon=0.25,
        )

    def retrieve(self, text, vector):
        vector = np.asarray(vector, dtype=np.float64)
        dense = self.vectors @ (vector / np.linalg.norm(vector))
        lexical = self.bm25.get_scores(re.findall(r"\w+", text.lower()))
        d = sorted(range(len(self.ids)), key=lambda j: (-float(dense[j]), self.ids[j]))[
            :10
        ]
        b = sorted(
            range(len(self.ids)), key=lambda j: (-float(lexical[j]), self.ids[j])
        )[:10]
        b = [j for j in b if lexical[j] > 0]
        fused = {}
        for ranking in [d, b]:
            for place, j in enumerate(ranking, 1):
                fused[j] = fused.get(j, 0.0) + 1 / (60 + place)
        order = sorted(fused, key=lambda j: (-fused[j], self.ids[j]))
        return {
            "dense": [{"id": self.ids[j], "score": float(dense[j])} for j in d],
            "bm25": [{"id": self.ids[j], "score": float(lexical[j])} for j in b],
            "rrf": [{"id": self.ids[j], "score": fused[j]} for j in order],
            "selected_id": self.ids[order[0]],
        }


def check_trace(saved, replay):
    need(
        saved["selected_id"] == replay["selected_id"],
        "Selected correction differs in cached replay",
    )
    for channel in ["dense", "bm25", "rrf"]:
        need(
            [v["id"] for v in saved[channel]] == [v["id"] for v in replay[channel]],
            "Saved ranking differs: " + channel,
        )
        need(
            all(
                abs(x["score"] - y["score"]) < 1e-10
                for x, y in zip(saved[channel], replay[channel])
            ),
            "Saved scores differ",
        )


def input_context(base, source, query_vector, kb, kb_vectors):
    ctx = base["context"]
    hits = ctx["example_groups"][0][1]
    need(len(hits) == 5, "Wrong reference example budget")
    for hit in [*ctx["definitions"], *ctx["guidelines"], *hits]:
        index, record = kb[hit["id"]]
        metadata = {
            **record["meta"],
            **{f: record[f] for f in ["dimension", "kind", "label", "lang", "source"]},
        }
        need(
            hit["text"] == record["text"]
            and {k: v for k, v in hit["meta"].items() if k != "_ranks"} == metadata,
            "Original KB example/definition metadata differs",
        )
    cosines = [context.cosine(query_vector, kb_vectors[kb[h["id"]][0]]) for h in hits]
    fifth = hits[4]
    known = {f: sum(h["meta"].get(f) is not None for h in hits) for f in FIELDS}
    return {
        "key": source["key"],
        "id": source["id"],
        "text": source["text"],
        "source": source["source"],
        "population": source["population"],
        "context": ctx,
        "messages": base["messages"],
        "messages_sha256": base["messages_sha256"],
        "context_sha256": base["context_sha256"],
        "ordinary_cosines": cosines,
        "removed_id": fifth["id"],
        "removed_gate": fifth["meta"]["gate"],
        "removed_source": fifth["meta"]["source"],
        "removed_known_fine": {f: fifth["meta"].get(f) is not None for f in FIELDS},
        "known_fine_counts": known,
        "remaining_known_fine_counts": {
            f: known[f] - (fifth["meta"].get(f) is not None) for f in FIELDS
        },
        "reference_hateful_examples": sum(h["meta"]["gate"] for h in hits),
        "fifth_is_lowest_cosine": cosines[4] <= min(cosines) + 1e-12,
        "fifth_cosine_rank": 1 + sum(v > cosines[4] + 1e-12 for v in cosines),
    }

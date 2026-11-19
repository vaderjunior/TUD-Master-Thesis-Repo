import re
import numpy as np
from rank_bm25 import BM25Okapi
from analysis.feedback.source import design as design
from analysis.common import prompt as renderer


class FeedbackBucket:

    def __init__(self, records, vectors):
        ids = [r["id"] for r in records]
        design.u.require(
            1 <= len(ids) <= 40 and len(ids) == len(set(ids)),
            "Unique bounded feedback pool required",
        )
        design.u.require(
            all((isinstance(i, str) and i for i in ids)), "Nonempty record IDs required"
        )
        order = sorted(range(len(ids)), key=lambda i: ids[i])
        self.records = [records[i] for i in order]
        self.ids = [ids[i] for i in order]
        x = np.asarray(vectors, dtype=np.float64)
        design.u.require(
            x.ndim == 2
            and len(x) == len(ids)
            and (x.shape[1] > 0)
            and np.isfinite(x).all(),
            "Invalid vector matrix",
        )
        x = x[order]
        norms = np.linalg.norm(x, axis=1)
        design.u.require(np.all(norms > 0), "Zero feedback vector")
        self.vectors = x / norms[:, None]
        corpus = [re.findall("\\w+", r["text"].lower()) for r in self.records]
        design.u.require(sum(map(len, corpus)) > 0, "Empty BM25 vocabulary")
        self.bm25 = BM25Okapi(corpus, k1=1.5, b=0.75, epsilon=0.25)

    def retrieve(self, text, vector):
        q = np.asarray(vector, dtype=np.float64)
        design.u.require(
            q.shape == (self.vectors.shape[1],) and np.isfinite(q).all(),
            "Invalid query vector",
        )
        norm = np.linalg.norm(q)
        design.u.require(norm > 0, "Zero query vector")
        dense = self.vectors @ (q / norm)
        lexical = self.bm25.get_scores(re.findall("\\w+", text.lower()))
        design.u.require(
            np.isfinite(dense).all() and np.isfinite(lexical).all(),
            "Nonfinite retrieval score",
        )
        d = sorted(range(len(self.ids)), key=lambda i: (-float(dense[i]), self.ids[i]))[
            :10
        ]
        b = sorted(
            range(len(self.ids)), key=lambda i: (-float(lexical[i]), self.ids[i])
        )[:10]
        b = [i for i in b if lexical[i] > 0]
        fused = {}
        for ranking in (d, b):
            for rank, i in enumerate(ranking, 1):
                fused[i] = fused.get(i, 0.0) + 1.0 / (60 + rank)
        final = sorted(fused, key=lambda i: (-fused[i], self.ids[i]))
        trace = {
            "dense": [{"id": self.ids[i], "score": float(dense[i])} for i in d],
            "bm25": [{"id": self.ids[i], "score": float(lexical[i])} for i in b],
            "rrf": [{"id": self.ids[i], "score": fused[i]} for i in final],
            "selected_id": self.ids[final[0]],
        }
        return (self.records[final[0]], trace)


def updated_prompt(base_entry, bucket, query_vector):
    record, trace = bucket.retrieve(base_entry["text"], query_vector)
    context = design.append_feedback(base_entry["context"], record)
    messages = renderer.render_messages(
        base_entry["text"], base_entry["lang"], context, "corrected"
    )
    return {"context": context, "messages": messages, "retrieval": trace}

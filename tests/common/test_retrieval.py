from unittest.mock import Mock
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from analysis.common import retrieval as r


class RetrievalPreparationTests(unittest.TestCase):

    def test_cache_identity_and_payload(self):
        v = np.zeros((2, 1024), np.float32)
        v[:, 0] = 1
        identity = {"rows": 2, "revision": "fixed", "texts_sha256": "fixed"}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "cache.npz"
            meta = {**identity, "vectors_sha256": hashlib.sha256(v.tobytes()).hexdigest()}
            np.savez(path, vectors=v, metadata=np.asarray(json.dumps(meta)))
            np.testing.assert_array_equal(r.load_chunk(path, identity), v)
            with self.assertRaisesRegex(RuntimeError, "identity"):
                r.load_chunk(path, {**identity, "revision": "wrong"})
            v[:, 0] = 0
            v[:, 1] = 1
            np.savez(path, vectors=v, metadata=np.asarray(json.dumps(meta)))
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                r.load_chunk(path, identity)

    def test_vector_guards(self):
        v = np.zeros((1, 1024), np.float32)
        v[0, 0] = 1
        r.validate_vectors(v, 1)
        for bad in (v.astype(np.float64), v[:, :100], v * 0, v * np.nan):
            with self.assertRaises(RuntimeError):
                r.validate_vectors(bad, 1)

    def test_no_query_fallback(self):
        v = np.zeros((1, 1024), np.float32)
        v[0, 0] = 1
        encoder = r.CachedEncoder(["exact text"], v)
        np.testing.assert_array_equal(encoder.encode(["exact text"]), v)
        with self.assertRaisesRegex(RuntimeError, "Uncached"):
            encoder.encode([" exact text"])
        with self.assertRaisesRegex(RuntimeError, "normalization"):
            encoder.encode(["exact text"], False)

    def test_output_boundary_and_preservation(self):
        with tempfile.TemporaryDirectory() as d, patch.object(r, "BASE", Path(d) / "new"):
            with self.assertRaisesRegex(RuntimeError, "outside"):
                r.put(Path(d) / "old.json", {})
            p = r.BASE / "x.json"
            r.put(p, {"a": 1})
            r.put(p, {"a": 1})
            with self.assertRaisesRegex(RuntimeError, "Preserve"):
                r.put(p, {"a": 2})
            self.assertEqual(r.read(p), {"a": 1})

    def context(self):
        records = {}
        ctx = {"definitions": [], "guidelines": [], "example_groups": [["Labelled examples", []]]}
        for kind, n in [("definition", 2), ("guideline", 2), ("example", 5)]:
            for i in range(n):
                rec = {
                    "id": f"{kind}-{i}",
                    "kind": kind,
                    "lang": "de",
                    "source": "unit",
                    "dimension": None,
                    "label": None,
                    "text": f"text {kind} {i}",
                    "meta": {"gate": None, "legal": []},
                }
                records[rec["id"]] = rec
                hit = {
                    "id": rec["id"],
                    "text": rec["text"],
                    "meta": r.record_meta(rec),
                    "via": "dense",
                }
                (ctx["example_groups"][0][1] if kind == "example" else ctx[kind + "s"]).append(hit)
        return (records, ctx)

    def test_context_annotations_and_budgets(self):
        records, ctx = self.context()
        self.assertEqual(len(r.validate_context(ctx, records, "de")), 5)
        bad = copy.deepcopy(ctx)
        bad["example_groups"][0][1][0]["meta"]["gate"] = False
        with self.assertRaisesRegex(RuntimeError, "annotation"):
            r.validate_context(bad, records, "de")
        bad = copy.deepcopy(ctx)
        bad["example_groups"][0][1][0]["meta"]["legal"] = None
        with self.assertRaisesRegex(RuntimeError, "annotation"):
            r.validate_context(bad, records, "de")
        bad = copy.deepcopy(ctx)
        bad["example_groups"][0][1].pop()
        with self.assertRaisesRegex(RuntimeError, "Five"):
            r.validate_context(bad, records, "de")
        with self.assertRaisesRegex(RuntimeError, "Cross-language"):
            r.validate_context(ctx, records, "en")

    def test_renderer_keeps_missing_vs_empty(self):
        from analysis.common.prompt import corrected_gold

        rendered = corrected_gold({"gate": None, "legal": []})
        self.assertIn("legal=[]", rendered)
        self.assertNotIn("hate=", rendered)
        self.assertIn("hate=false", corrected_gold({"gate": False, "legal": None}))

    def test_cached_constructor_never_loads_model(self):
        from types import SimpleNamespace
        from src.hsrag.retrieve import Retriever

        records, ctx = self.context()
        vector = np.zeros((1, 1024), np.float32)
        vector[0, 0] = 1
        encoder = r.CachedEncoder(["query"], vector)
        col = SimpleNamespace(metadata={"embedding_model": r.MODEL})
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "records.jsonl"
            path.write_text("\n".join((json.dumps(x) for x in records.values())), encoding="utf8")
            with patch.dict(
                "sys.modules",
                {
                    "sentence_transformers": SimpleNamespace(
                        SentenceTransformer=Mock(side_effect=AssertionError("Neural model loaded"))
                    )
                },
            ):
                current = r.make_retriever(col, path, {"strategy": "bm25"}, encoder)
            self.assertIs(current.model, encoder)
            self.assertIs(current.col, col)
            with patch.dict(
                "sys.modules",
                {
                    "sentence_transformers": SimpleNamespace(
                        SentenceTransformer=Mock(return_value=encoder)
                    )
                },
            ), patch("src.hsrag.retrieve.load_collection", return_value=col):
                reference = Retriever(Path(d), path, r.MODEL, {"strategy": "bm25"})
            self.assertEqual(current._bm25_records, reference._bm25_records)
            np.testing.assert_array_equal(current._embed("query"), reference._embed("query"))
            self.assertEqual(
                current._bm25_kind("text example 1", "example:de", 5),
                reference._bm25_kind("text example 1", "example:de", 5),
            )


if __name__ == "__main__":
    unittest.main()

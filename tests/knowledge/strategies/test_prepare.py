from types import SimpleNamespace
from unittest.mock import Mock
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
from analysis.knowledge.strategies import prepare as s


class StrategyPreparationTests(unittest.TestCase):

    def context(self):
        records = {}
        context = {
            "definitions": [],
            "guidelines": [],
            "example_groups": [["Labelled examples", []]],
        }
        for kind, count in [("definition", 2), ("guideline", 2), ("example", 5)]:
            for i in range(count):
                record = {
                    "id": f"{kind}-{i}",
                    "kind": kind,
                    "lang": "de",
                    "source": "fixture",
                    "dimension": None,
                    "label": None,
                    "text": f"one text {kind} {i}",
                    "meta": {"gate": None, "legal": []},
                }
                records[record["id"]] = record
                hit = {
                    "id": record["id"],
                    "text": record["text"],
                    "meta": s.r.record_meta(record),
                    "via": "bm25",
                }
                (
                    context["example_groups"][0][1] if kind == "example" else context[kind + "s"]
                ).append(hit)
        return (records, context)

    def test_bm25_underfill_and_empty_are_kept(self):
        records, context = self.context()
        context["definitions"] = []
        context["guidelines"] = []
        context["example_groups"][0][1] = context["example_groups"][0][1][:2]
        self.assertEqual(len(s.validate_context(context, records, "de", "bm25")), 2)
        for strategy in ("dense", "hybrid"):
            with self.assertRaisesRegex(RuntimeError, "budget"):
                s.validate_context(context, records, "de", strategy)
        context["example_groups"] = []
        self.assertEqual(s.validate_context(context, records, "de", "bm25"), [])
        messages = s.prompt.render_messages("Testtext", "de", context, "corrected")
        self.assertTrue(messages)

    def test_duplicate_or_over_budget_rejected(self):
        records, context = self.context()
        context["example_groups"][0][1][-1] = copy.deepcopy(context["example_groups"][0][1][0])
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            s.validate_context(context, records, "de", "dense")
        context["definitions"].append(context["definitions"][0])
        with self.assertRaisesRegex(RuntimeError, "budget"):
            s.validate_context(context, records, "de", "bm25")

    def test_cross_language_and_wrong_source_rejected(self):
        records, context = self.context()
        with self.assertRaisesRegex(RuntimeError, "Cross-language"):
            s.validate_context(context, records, "en", "dense")
        context["definitions"][0]["text"] += " changed"
        with self.assertRaisesRegex(RuntimeError, "text differs"):
            s.validate_context(context, records, "de", "dense")

    def test_missing_empty_and_negative_cannot_be_changed(self):
        records, context = self.context()
        self.assertEqual(len(s.validate_context(context, records, "de", "hybrid")), 5)
        for field, value in [("gate", False), ("legal", None)]:
            bad = copy.deepcopy(context)
            bad["example_groups"][0][1][0]["meta"][field] = value
            with self.assertRaisesRegex(RuntimeError, "annotation"):
                s.validate_context(bad, records, "de", "bm25")
        bad = copy.deepcopy(context)
        del bad["example_groups"][0][1][0]["meta"]["gate"]
        with self.assertRaisesRegex(RuntimeError, "annotation"):
            s.validate_context(bad, records, "de", "dense")

    def test_zero_only_timing_is_not_in_entry_comparison(self):
        original = {k: [k] for k in ("context", "messages", "retrieved", "trace", "final_hits")}
        original["key"] = "test"
        self.assertTrue(all(s.comparison(original, copy.deepcopy(original)).values()))
        changed = copy.deepcopy(original)
        changed["trace"].append("rank change")
        result = s.comparison(original, changed)
        self.assertFalse(result["trace"])
        self.assertFalse(result["entry"])
        self.assertTrue(result["messages"])
        changed = copy.deepcopy(original)
        changed["messages"].append("changed prompt")
        self.assertFalse(s.comparison(original, changed)["messages"])

    def test_nonfinite_trace_rejected(self):
        hit = SimpleNamespace(id="x", kind="example", score=float("nan"), via="dense", meta={})
        with self.assertRaisesRegex(RuntimeError, "Nonfinite"):
            s.hit_trace(hit)

    def test_output_ownership_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            s, "BASE", Path(directory) / "new"
        ):
            with self.assertRaisesRegex(RuntimeError, "outside"):
                s.save(Path(directory) / "old.json", {})
            target = s.BASE / "record.json"
            s.save(target, {"a": 1})
            s.save(target, {"a": 1})
            with self.assertRaisesRegex(RuntimeError, "Preserve"):
                s.save(target, {"a": 2})
            self.assertEqual(s.r.read(target), {"a": 1})

    def test_copy_rejects_changed_source_and_preserves_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "old"
            source.mkdir()
            file = source / "data.bin"
            file.write_bytes(b"original")
            with patch.object(s, "ROOT", root), patch.object(s.r, "ROOT", root), patch.object(
                s, "SOURCE_INDEX", source
            ), patch.object(s, "BASE", root / "new"):
                s.BASE.mkdir()
                plan = {"source_index_hashes": {"old/data.bin": s.r.sha(file)}}
                s.copy_index(s.BASE / "copy", plan)
                self.assertEqual((s.BASE / "copy/data.bin").read_bytes(), b"original")
                with self.assertRaisesRegex(RuntimeError, "Preserve"):
                    s.copy_index(s.BASE / "copy", plan)
                file.write_bytes(b"changed")
                with self.assertRaisesRegex(RuntimeError, "Source index"):
                    s.copy_index(s.BASE / "second", plan)
                self.assertFalse((s.BASE / "second").exists())

    def test_trace_preserves_original_selection_all_strategies(self):
        from src.hsrag.kb import _flatten_meta

        records = []
        for kind, number in [("definition", 12), ("guideline", 12), ("example", 36)]:
            for i in range(number):
                records.append(
                    {
                        "id": f"{kind}-{i:02}",
                        "kind": kind,
                        "lang": "en",
                        "source": "fixture",
                        "dimension": "hate_type" if kind == "definition" else None,
                        "label": "irony" if kind == "definition" else None,
                        "text": f"{kind} query rare{i} repeated theme{i % 5}",
                        "meta": {"gate": True, "hate_types": ["irony"]},
                    }
                )
        rng = np.random.default_rng(73)
        vectors = rng.normal(size=(len(records), 1024)).astype(np.float32)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        q = vectors[-1:]

        def matches(record, where):
            return (
                all((matches(record, part) for part in where["$and"]))
                if "$and" in where
                else all((record[k] == v for k, v in where.items()))
            )

        class Collection:
            metadata = {"embedding_model": s.r.MODEL}

            def query(self, query_embeddings, n_results, where, include):
                eligible = [i for i, record in enumerate(records) if matches(record, where)]
                query = np.asarray(query_embeddings)[0]
                chosen = sorted(
                    eligible, key=lambda i: (-float(vectors[i] @ query), records[i]["id"])
                )[:n_results]
                result = {
                    "ids": [[records[i]["id"] for i in chosen]],
                    "documents": [[records[i]["text"] for i in chosen]],
                    "metadatas": [[_flatten_meta(records[i]) for i in chosen]],
                    "distances": [[1 - float(vectors[i] @ query) for i in chosen]],
                }
                if "embeddings" in include:
                    result["embeddings"] = [[vectors[i].tolist() for i in chosen]]
                return result

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_text("\n".join((json.dumps(record) for record in records)), encoding="utf8")
            cfg = {
                "k_examples": 5,
                "k_definitions": 2,
                "k_guidelines": 2,
                "k_examples_feedback": 0,
                "k_definitions_mode": "flat",
                "use_mmr": True,
                "mmr_pool_factor": 3,
                "mmr_lambda": 0.7,
                "rrf_k": 60,
            }
            encoder = s.r.CachedEncoder(["query"], q)
            for strategy in s.STRATEGIES:
                with self.subTest(strategy=strategy), patch.dict(
                    "sys.modules",
                    {
                        "sentence_transformers": SimpleNamespace(
                            SentenceTransformer=Mock(
                                side_effect=AssertionError("No neural fallback")
                            )
                        )
                    },
                ):
                    plain = s.r.make_retriever(
                        Collection(), path, {**cfg, "strategy": strategy}, encoder
                    )
                    traced = s.r.make_retriever(
                        Collection(), path, {**cfg, "strategy": strategy}, encoder
                    )
                    events = s.instrument(traced)
                    left, right = (plain.retrieve("query", "en"), traced.retrieve("query", "en"))
                self.assertEqual(s.prompt.from_retrieval(left), s.prompt.from_retrieval(right))
                self.assertEqual(
                    [s.hit_trace(h) for h in left.all_hits()],
                    [s.hit_trace(h) for h in right.all_hits()],
                )
                self.assertEqual(
                    s.prompt.render_messages(
                        "query", "en", s.prompt.from_retrieval(left), "corrected"
                    ),
                    s.prompt.render_messages(
                        "query", "en", s.prompt.from_retrieval(right), "corrected"
                    ),
                )
                self.assertTrue(events)
                mmr = [e for e in events if e["operation"] == "mmr"]
                if strategy == "bm25":
                    self.assertFalse(mmr)
                else:
                    self.assertEqual(len(mmr[0]["input_ids"]), 15 if strategy == "dense" else 30)
                    self.assertEqual(len(mmr[0]["hits"]), 5 if strategy == "dense" else 10)
                if strategy == "hybrid":
                    self.assertEqual(len([e for e in events if e["operation"] == "rrf"]), 3)

    def test_network_guard(self):
        with self.assertRaisesRegex(RuntimeError, "forbidden"):
            s.forbidden()


if __name__ == "__main__":
    unittest.main()

"""Tests for strategies verify."""

from copy import deepcopy
import json
from pathlib import Path
import time
import unittest
import numpy as np
from analysis.knowledge.strategies import verify as a


class AuditTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not (a.s.BASE / "runs/prepare/complete.json").exists():
            raise unittest.SkipTest("Original archived input is not available")
        receipt = a.r.read(a.s.BASE / "runs/prepare/complete.json")
        with (a.s.ROOT / receipt["prompts"]).open(encoding="utf8") as handle:
            cls.entry = json.loads(handle.readline())
        ids = {h["id"] for h in cls.entry["final_hits"]}
        cls.records = {
            r["id"]: r for r in a.r.rows(a.s.CORE / "kb/full/records.jsonl") if r["id"] in ids
        }
        cls.row = {k: cls.entry[k] for k in ("item_id", "key", "lang", "scopes", "text")}

    def test_source_valid_then_annotation_tampered_with_matching_hash(self):
        a.audit_context(self.entry, self.row, self.records)
        changed = deepcopy(self.entry)
        hit = changed["context"]["example_groups"][0][1][0]
        self.assertIsNone(hit["meta"]["hate_types"])
        hit["meta"]["hate_types"] = []
        changed["context_sha256"] = a.r.signature(changed["context"])
        with self.assertRaisesRegex(RuntimeError, "source annotation"):
            a.audit_context(changed, self.row, self.records)

    def test_false_is_not_numeric_zero(self):
        changed = deepcopy(self.entry)
        hit = changed["context"]["example_groups"][0][1][0]
        self.assertIs(hit["meta"]["gate"], False)
        hit["meta"]["gate"] = 0
        changed["context_sha256"] = a.r.signature(changed["context"])
        with self.assertRaisesRegex(RuntimeError, "source annotation"):
            a.audit_context(changed, self.row, self.records)

    def test_text_and_message_corruption(self):
        for field in ("text", "messages_sha256"):
            changed = deepcopy(self.entry)
            changed[field] = "tampered"
            with self.assertRaises(RuntimeError):
                a.audit_context(changed, self.row, self.records)

    def test_matrix_rejects_duplicates_missing_or_reordered(self):
        entries = [{"key": "one", "strategy": "dense"}, {"key": "one", "strategy": "hybrid"}]
        expected = [("one", "dense"), ("one", "hybrid")]
        a.matrix_check(entries, expected)
        for bad in (entries[:1], entries[::-1], [entries[0], entries[0]]):
            with self.assertRaises(RuntimeError):
                a.matrix_check(bad, expected)

    def test_replay_preserves_metadata_and_real_prompt_differences(self):
        old = {"key": "one", "strategy": "dense", **{k: [] for k in a.FIELDS if k != "entry"}}
        actual = deepcopy(old)
        actual["trace"] = ["changed"]
        saved = {
            "key": "one",
            "strategy": "dense",
            "actual_entry": actual,
            "matches": {k: k not in ("trace", "entry") for k in a.FIELDS},
        }
        self.assertEqual(a.compare_replay(old, saved), actual)
        bad = deepcopy(saved)
        bad["matches"]["messages"] = False
        with self.assertRaises(RuntimeError):
            a.compare_replay(old, bad)
        bad = deepcopy(saved)
        del bad["actual_entry"]
        with self.assertRaises(RuntimeError):
            a.compare_replay(old, bad)

    def test_lexical_positive_only_tie_and_no_fallback(self):
        records = [
            {"id": str(i), "kind": "example", "text": text}
            for i, text in enumerate(
                ["rare one", "rare two", "other three", "other four", "other five"]
            )
        ]
        lexicon = a.Lexicon(records)
        self.assertEqual(a.plain_hits(lexicon.top("rare", 5)), ["0", "1"])
        self.assertEqual(lexicon.top("unseen", 5), [])

    def test_mmr_selects_diverse_candidate_then_rejects_wrong_trace_scores(self):
        chosen, _ = a.independent_mmr(
            np.array([1.0, 0.0]), np.array([[1.0, 0.0], [0.999, 0.001], [0.8, 0.6]]), 2, 0.3
        )
        self.assertEqual(chosen, [0, 2])
        good = {"id": "a", "kind": "example", "score": 0.4, "via": "dense", "ranks": {}}
        for bad in ({**good, "score": float("nan")}, {**good, "id": "b"}, {**good, "score": 0.5}):
            with self.assertRaises(RuntimeError):
                a.compare_hits([bad], [good])

    def test_rrf_tie_by_id_and_duplicate_rejection(self):
        hit = lambda x: {"id": x, "kind": "example", "score": 1.0, "via": "dense", "ranks": {}}
        got = a.independent_rrf({"dense": [hit("z")], "bm25": [hit("a")]}, 2)
        self.assertEqual(a.plain_hits(got), ["a", "z"])
        self.assertEqual(got[0]["ranks"], {"bm25": 1})
        with self.assertRaises(RuntimeError):
            a.independent_rrf({"dense": [hit("a"), hit("a")]}, 2)

    def test_only_known_database_qualification_allowed(self):
        expected = [("segments", 1, "collection", 0), ("segments", 2, "collection", 0)]
        a.database_findings([("ok",)], expected, 0)
        for integrity, foreign, orphans in [
            ([("corrupt",)], expected, 0),
            ([("ok",)], expected, 1),
            ([("ok",)], expected + [("new", 1, "x", 0)], 0),
        ]:
            with self.assertRaises(RuntimeError):
                a.database_findings(integrity, foreign, orphans)


if __name__ == "__main__":
    destination = a.s.BASE / "development/full_audit_tests_v1.json"
    a.need(not destination.exists(), "Preserve previous test evidence")
    tick = time.perf_counter()
    with a.s.patch.object(a.s.socket.socket, "connect", a.s.forbidden), a.s.patch.object(
        a.s.socket, "create_connection", a.s.forbidden
    ):
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests)
        )
    a.s.save(
        destination,
        {
            "created_utc": a.s.utc(),
            "status": "passed" if result.wasSuccessful() else "failed",
            "tests_run": result.testsRun,
            "seconds": time.perf_counter() - tick,
            "failures": [str(case) for case, _ in result.failures],
            "errors": [str(case) for case, _ in result.errors],
            "source_hashes": {a.r.rel(p): a.r.sha(p) for p in (Path(__file__), Path(a.__file__))},
        },
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)

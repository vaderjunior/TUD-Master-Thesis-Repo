"""Tests for sampled banks retrieval."""

from copy import deepcopy
import hashlib
import math
import unittest
import numpy as np
from analysis.knowledge.sampled_banks import retrieval as p
from analysis.knowledge.sampled_banks import verify_retrieval as check


class PoolPreparationTests(unittest.TestCase):

    def fixture(self):
        full = [
            {"id": "a", "kind": "definition", "lang": "en"},
            {"id": "en1", "kind": "example", "lang": "en"},
            {"id": "de1", "kind": "example", "lang": "de"},
            {"id": "en2", "kind": "example", "lang": "en"},
        ]
        manifest = {
            "abstract_ids": ["a"],
            "languages": {
                "en": {"size": 2, "example_ids": ["en1", "en2"]},
                "de": {"size": 1, "example_ids": ["de1"]},
            },
        }
        return (full, manifest)

    def test_inherited_order_across_languages(self):
        full, m = self.fixture()
        self.assertEqual([r["id"] for r in p.selection(full, m)], ["a", "en1", "de1", "en2"])

    def test_duplicate_sample_rejected(self):
        full, m = self.fixture()
        m["languages"]["en"]["example_ids"] = ["en1", "en1"]
        with self.assertRaises(RuntimeError):
            p.selection(full, m)

    def test_missing_train_record_rejected(self):
        full, m = self.fixture()
        m["languages"]["en"]["example_ids"] = ["en1", "unseen"]
        with self.assertRaises(RuntimeError):
            p.selection(full, m)

    def test_wrong_language_rejected(self):
        full, m = self.fixture()
        full[1]["lang"] = "de"
        with self.assertRaises(RuntimeError):
            p.selection(full, m)

    def test_changed_record_order_rejected(self):
        full, m = self.fixture()
        m["languages"]["en"]["example_ids"].reverse()
        with self.assertRaises(RuntimeError):
            p.selection(full, m)

    def test_fix_abstracts_preserve_examples_without_mutating_inputs(self):
        raw = {
            "definitions": ["new"],
            "guidelines": ["new"],
            "example_groups": [["Labelled examples", [{"id": "x"}]]],
        }
        ref = {"context": {"definitions": ["fixed"], "guidelines": ["fixed"]}}
        original = deepcopy(raw)
        out = p.delivered_context(raw, ref)
        self.assertEqual(out["definitions"], ["fixed"])
        self.assertEqual(out["example_groups"], raw["example_groups"])
        out["example_groups"][0][1][0]["id"] = "changed"
        self.assertEqual(raw, original)
        self.assertEqual(ref["context"]["definitions"], ["fixed"])

    def test_extra_feedback_group_rejected(self):
        raw = {
            "definitions": [],
            "guidelines": [],
            "example_groups": [["Additional labelled examples", []]],
        }
        with self.assertRaises(RuntimeError):
            p.delivered_context(raw, {"context": raw})

    def test_semantic_false_and_zero_are_not_equal(self):
        a = {key: {} for key in p.COMPARE_FIELDS}
        b = deepcopy(a)
        a["context"] = {"gate": False}
        b["context"] = {"gate": 0}
        self.assertFalse(p.comparison(a, b)["context"])
        self.assertFalse(p.comparison(a, b)["entry"])

    def test_replay_changed_payload_must_be_retained(self):
        a = {key: [] for key in p.COMPARE_FIELDS}
        a.update(key="en/1", condition="c")
        b = deepcopy(a)
        b["messages"] = ["changed"]
        saved = {"key": "en/1", "condition": "c", "matches": p.comparison(a, b)}
        with self.assertRaises(RuntimeError):
            check.replay_entry(a, saved)
        saved["actual_entry"] = b
        self.assertEqual(check.replay_entry(a, saved), b)

    def test_unchanged_replay_cannot_hide_an_alternative(self):
        a = {key: [] for key in p.COMPARE_FIELDS}
        a.update(key="en/1", condition="c")
        saved = {"key": "en/1", "condition": "c", "matches": p.comparison(a, a), "actual_entry": a}
        with self.assertRaises(RuntimeError):
            check.replay_entry(a, saved)

    def test_reference_target_mismatch_rejected(self):
        with self.assertRaises(RuntimeError):
            p.reference_entry({"text": "original"}, "c", {"text": "other"}, "hash")

    def test_write_boundary(self):
        self.assertTrue(p.owned(p.BASE / "test.json").is_relative_to(p.BASE))
        with self.assertRaises(RuntimeError):
            p.owned(p.CORE / "plan.json")
        with self.assertRaises(RuntimeError):
            p.owned(p.BASE / ".." / "scientific_plan.json")

    def test_cached_encoder_has_no_fallback(self):
        enc = p.r.CachedEncoder(["known"], np.ones((1, 1024), dtype=np.float32))
        with self.assertRaises(RuntimeError):
            enc.encode(["unknown"])

    def test_vector_provenance_is_text_sensitive(self):
        vectors = np.zeros((1, 1024), dtype=np.float32)
        vectors[0, 0] = 1.0
        rows = [{"id": "a", "text": "text"}]
        prov = [{"id": "a", "text_sha256": hashlib.sha256(b"text").hexdigest()}]
        p.verify_provenance(rows, vectors, prov, "id", "text")
        rows[0]["text"] = "changed"
        with self.assertRaises(RuntimeError):
            p.verify_provenance(rows, vectors, prov, "id", "text")

    def test_lexical_hand_value_and_stable_tie(self):
        rows = [
            {"id": rid, "kind": "example", "text": text}
            for rid, text in [
                ("z", "rare shared"),
                ("a", "rare shared"),
                ("c", "third unique"),
                ("d", "fourth token"),
                ("e", "fifth extra"),
            ]
        ]
        hits = check.OrderedLexicon(rows).top("rare", 10)
        self.assertEqual([h["id"] for h in hits], ["z", "a"])
        self.assertAlmostEqual(hits[0]["score"], math.log(3.5) - math.log(2.5), places=14)
        self.assertEqual(check.OrderedLexicon(rows).top("unknown", 10), [])

    def test_annotation_unknown_and_observed_empty_differ(self):
        self.assertEqual(check.gold_line({"gate": None, "legal": []}), "legal=[]")
        self.assertEqual(
            check.gold_line({"gate": False, "legal": ["propaganda"]}),
            "hate=false, legal=[propaganda]",
        )
        self.assertEqual(check.gold_line({"gate": False}), "hate=false")

    def test_index_foreign_key_allowance_does_not_allow_real_orphans(self):
        known = [("segments", 1, "collection", 0), ("segments", 2, "collection", 0)]
        check.database_policy([("ok",)], [], 0)
        check.database_policy([("ok",)], known, 0)
        with self.assertRaises(RuntimeError):
            check.database_policy([("ok",)], known, 1)
        with self.assertRaises(RuntimeError):
            check.database_policy([("ok",)], [("other", 1, "x", 0)], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

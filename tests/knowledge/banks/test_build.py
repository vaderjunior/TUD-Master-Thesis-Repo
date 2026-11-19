"""Tests for banks build."""

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from analysis.knowledge.banks import build as kb


def row(rid, text, **gold):
    return {
        "id": rid,
        "text": text,
        "source": "fixture",
        "lang": "en",
        "gate": False,
        "target_groups": None,
        "hate_types": None,
        "severity": None,
        **gold,
    }


class PreparationTests(unittest.TestCase):

    def test_nested_pools_preserve_null_empty_and_old_records(self):
        a = row("a", "An annotated negative", target_groups=[])
        b = row("b", "An unannotated dimension")
        original = [kb.example(a)]
        pools, removed = kb.assemble(original, [a, b], [], set())
        self.assertEqual(removed, [])
        self.assertEqual(pools["small"], original)
        self.assertEqual(pools["all"][0], original[0])
        self.assertEqual(pools["all"][0]["meta"]["target_groups"], [])
        self.assertIsNone(pools["all"][1]["meta"]["target_groups"])

    def test_same_exclusions_apply_to_both_pools(self):
        a, b, c = (row("a", "Alpha"), row("b", "Beta"), row("c", ' "HELD OUT" '))
        pools, removed = kb.assemble(
            [kb.example(a), kb.example(b)], [a, b, c], ["ex-a"], {kb.norm("held out")}
        )
        self.assertEqual([r["id"] for r in pools["small"]], ["ex-b"])
        self.assertEqual([r["id"] for r in pools["all"]], ["ex-b"])
        self.assertEqual(
            {r["reason"] for r in removed}, {"known_near_duplicate", "normalized_heldout_overlap"}
        )

    def test_existing_gold_disagreement_is_not_silently_repaired(self):
        a = row("a", "Alpha")
        original = kb.example(a)
        original["meta"]["gate"] = True
        with self.assertRaisesRegex(RuntimeError, "differ"):
            kb.assemble([original], [a], [], set())

    def test_duplicate_id_and_text_require_review(self):
        a = row("a", "Alpha")
        for rows in ([a, a], [a, row("b", "ALPHA")]):
            with self.assertRaisesRegex(RuntimeError, "Duplicate|collision"):
                kb.assemble([], rows, [], set())

    def test_unannotated_row_is_not_a_negative(self):
        pools, removed = kb.assemble([], [row("a", "Alpha", gate=None)], [], set())
        self.assertEqual(pools["all"], [])
        self.assertEqual(removed[0]["reason"], "no_gold")

    def test_within_train_punctuation_variants_preserved(self):
        a, b = (row("a", "Alpha"), row("b", '"Alpha"'))
        pools, removed = kb.assemble([], [a, b], [], set())
        self.assertEqual(len(pools["all"]), 2)
        self.assertEqual(removed, [])

    def test_writer_refuses_escape_and_different_existing_content(self):
        with tempfile.TemporaryDirectory() as td:
            base = Path(td) / "experiment"
            with patch.object(kb, "BASE", base):
                target = base / "records.jsonl"
                kb.write_once(target, b"original")
                kb.write_once(target, b"original")
                with self.assertRaisesRegex(RuntimeError, "preserving"):
                    kb.write_once(target, b"replacement")
                self.assertEqual(target.read_bytes(), b"original")
                with self.assertRaisesRegex(RuntimeError, "outside"):
                    kb.write_once(base / ".." / "escaped.json", b"bad")


class ResumeTests(unittest.TestCase):

    def setUp(self):
        self.mf = {
            "name": "fixture",
            "subset": "dev",
            "arms": ["rag"],
            "provider": "peasec",
            "pinned_model": "expected-model",
            "temperature": 1.0,
            "n_votes": 3,
            "workers": 8,
            "max_repair_retries": 2,
            "retrieval": {"k_examples": 5},
            "records_path": "fixture/records.jsonl",
            "chroma_path": "fixture/chroma",
        }
        self.stamps = {
            "code_version": "c",
            "prompt_version": "p",
            "taxonomy_version": "t",
            "kb_version": "k",
        }
        self.expected = {"a": "hash-a", "b": "hash-b"}
        ids = {
            "definitions": ["d"],
            "guidelines": ["g"],
            "Labelled examples": ["x", "y", "z", "w", "v"],
        }
        self.context = {"dev/a": ids, "dev/b": ids}
        self.header = {"_manifest": {**self.mf, "kb_version": "k"}}
        self.record = {
            "item_id": "a",
            "text_hash": "hash-a",
            "lang": "en",
            "arm": "rag",
            "active_model": "expected-model",
            "temperature": 1.0,
            "workers": 8,
            "n_runs": 3,
            "retrieval_config": self.mf["retrieval"],
            "retrieved": ids,
            "raw_runs": [[], [], []],
            "run_predictions": [{}, {}, {}],
            "n_valid": 3,
            "result": {"hate": False},
            **self.stamps,
        }

    def check_rows(self, rows):
        return kb.verify_result_rows(rows, self.mf, self.expected, self.context, self.stamps)

    def test_partial_run_counts_missing_without_repeating_saved_id(self):
        result = self.check_rows([self.header, self.record])
        self.assertEqual(result["written"], 1)
        self.assertEqual(result["missing"], 1)

    def test_resume_rejects_duplicate_model_kb_and_context_drift(self):
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            self.check_rows([self.header, self.record, self.record])
        for field, bad in [
            ("active_model", "wrong-model"),
            ("kb_version", "old-kb"),
            ("text_hash", "wrong-input"),
            ("retrieved", {}),
        ]:
            record = deepcopy(self.record)
            record[field] = bad
            with self.assertRaises(RuntimeError):
                self.check_rows([self.header, record])

    def test_zero_valid_votes_are_exposed_not_counted_as_valid(self):
        record = {
            **self.record,
            "n_valid": 0,
            "result": None,
            "run_predictions": [None, None, None],
        }
        result = self.check_rows([self.header, record])
        self.assertEqual(result["invalid_results"], 1)

    def test_wrong_header_and_partial_vote_material_are_rejected(self):
        header = deepcopy(self.header)
        header["_manifest"]["records_path"] = "old/records.jsonl"
        with self.assertRaises(RuntimeError):
            self.check_rows([header, self.record])
        with self.assertRaises(RuntimeError):
            self.check_rows([self.header, {**self.record, "raw_runs": [[]]}])


if __name__ == "__main__":
    unittest.main()

"""Tests for banks design."""

import unittest
import pandas as pd
from analysis.knowledge.banks import design as p


def rec(rid, lang="en", text=None):
    return {
        "id": rid,
        "kind": "example",
        "lang": lang,
        "text": text or rid,
        "source": "test-train",
        "dimension": None,
        "label": None,
        "meta": {
            "gate": True,
            "target_groups": None,
            "hate_types": None,
            "severity": None,
            "legal": None,
            "illustrative_only": False,
        },
    }


class PrepareTests(unittest.TestCase):

    def test_cross_language_collision(self):
        with self.assertRaises(AssertionError):
            p.combine([rec("a", text=" SAME ")], [rec("b", "de", "same")], [])
        with self.assertRaises(AssertionError):
            p.combine([rec("a")], [rec("a", "de", "other")], [])

    def test_filter_ids_and_order(self):
        df = pd.DataFrame({"id": ["ex-c", "ex-a", "ex-b"], "text": ["c", "a", "b"]})
        new, removed = p.filtered(df, {"a"})
        self.assertEqual(new.id.tolist(), ["ex-c", "ex-b"])
        self.assertEqual(removed, ["a"])
        with self.assertRaises(AssertionError):
            p.filtered(pd.DataFrame({"id": ["a", "ex-a"]}), set())

    def test_head_masks_and_legal(self):
        a = rec("a", "de")
        a["meta"].update(gate=None, legal=[])
        self.assertEqual(p.record_sources([a], "hate"), {})
        self.assertIn("a", p.record_sources([a], "legal"))
        b = rec("b")
        b["meta"].update(gate=False, hate_types=["irony"])
        self.assertEqual(p.record_sources([b], "hate_type"), {})

    def test_static_sets_repeat_and_language(self):
        rows = [rec(f"{lang}-{i}", lang) for lang in ("en", "de") for i in range(20)]
        a = p.fixed_sets(rows)
        self.assertEqual(a, p.fixed_sets(rows))
        self.assertEqual(len(a), 6)
        for x in a:
            self.assertEqual(len(x["records"]), 5)
            self.assertTrue(all((r["lang"] == x["lang"] for r in x["records"])))

    def test_output_guard(self):
        with self.assertRaises(AssertionError):
            p.own(p.ROOT / "kb/records.jsonl")

    def test_eligible_controls_keep_annotated_empty(self):
        rows = [rec(str(i), "de") for i in range(10)]
        for r in rows[:5]:
            r["meta"].update(gate=None, legal=[])
        selected = p.eligible_sets(rows, "de", "legal")
        self.assertTrue(all((x["pool_size"] == 5 for x in selected)))
        self.assertTrue(all((r["meta"]["legal"] == [] for x in selected for r in x["records"])))

    def test_real_message_schema(self):
        text = "A diagnostic sentence."
        messages = p.prompt.render_messages(
            text, "en", {"definitions": [], "guidelines": [], "example_groups": []}, "corrected"
        )
        p.check_messages(messages, text)
        with self.assertRaises(AssertionError):
            p.check_messages([{"role": x["role"], "content": x["text"]} for x in messages], text)


if __name__ == "__main__":
    unittest.main()

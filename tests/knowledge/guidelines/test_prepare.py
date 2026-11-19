"""Tests for guidelines prepare."""

from contextlib import nullcontext
from copy import deepcopy
import math
from pathlib import Path
import socket
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from analysis.knowledge.guidelines import prepare as p
from analysis.knowledge.guidelines import verify_preparation as c

r = p.r


class GuidelineContracts(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if not (p.INVENTORY / "rule_mapping.json").exists():
            raise unittest.SkipTest("Original archived input is not available")
        cls.mapping = r.read(p.INVENTORY / "rule_mapping.json")
        cls.rows = r.rows(p.INVENTORY / "inputs.jsonl")
        records = r.rows(r.CORE / "kb/full/records.jsonl")
        cls.records = {x["id"]: x for x in records}

        def hit(record):
            return {
                "id": record["id"],
                "text": record["text"],
                "meta": r.record_meta(record),
                "via": "rrf",
            }

        definitions = [hit(x) for x in records if x["kind"] == "definition"][:2]
        examples = [hit(x) for x in records if x["kind"] == "example" and x["lang"] == "en"][:5]
        guidelines = [hit(x) for x in records if x["kind"] == "guideline" and x["lang"] == "en"][:2]
        cls.context = {
            "definitions": definitions,
            "guidelines": guidelines,
            "example_groups": [["Labelled examples", examples]],
        }

    def fixture(self, row=None):
        row = deepcopy(row or self.rows[0])
        context = deepcopy(self.context)
        messages = p.prompt.render_messages(row["text"], row["lang"], context, "corrected")
        base = {
            **{k: row[k] for k in p.IDENTITY},
            "context": context,
            "messages": messages,
            "retrieved": p.prompt.hit_ids(context),
            "messages_sha256": r.signature(messages),
        }
        return (row, base)

    def entry(self, condition="matched_rule"):
        row, base = self.fixture()
        return (row, base, p.make_entry(row, base, condition, self.mapping, self.records))

    def test_all_six_mappings_five_conditions_and_fixed_backgrounds(self):
        for functionality in self.mapping:
            row, base = self.fixture(
                next((x for x in self.rows if x["functionality"] == functionality))
            )
            before = r.signature(base)
            for condition, expected in zip(p.CONDITIONS, (0, 2, 2, 2, 1)):
                with self.subTest(functionality=functionality, condition=condition):
                    entry = p.make_entry(row, base, condition, self.mapping, self.records)
                    c.audit_arm(row, base, entry, self.mapping, self.records)
                    self.assertEqual(len(entry["context"]["guidelines"]), expected)
                    self.assertEqual(
                        entry["context"]["definitions"], base["context"]["definitions"]
                    )
                    self.assertEqual(
                        entry["context"]["example_groups"], base["context"]["example_groups"]
                    )
            self.assertEqual(r.signature(base), before)

    def test_natural_exact_prompt_and_forced_filler_last(self):
        row, base = self.fixture()
        entries = {
            name: p.make_entry(row, base, name, self.mapping, self.records) for name in p.CONDITIONS
        }
        self.assertEqual(entries["natural"]["messages"], base["messages"])
        filler = entries["filler_only"]["context"]["guidelines"][0]
        self.assertEqual(entries["matched_rule"]["context"]["guidelines"][1], filler)
        self.assertEqual(entries["alternative_rule"]["context"]["guidelines"][1], filler)
        entries["matched_rule"]["context"]["definitions"][0]["text"] = "mutation"
        self.assertEqual(
            entries["natural"]["context"]["definitions"], base["context"]["definitions"]
        )

    def test_natural_german_rule_not_silently_filtered(self):
        row, base = self.fixture()
        record = next(
            (x for x in self.records.values() if x["kind"] == "guideline" and x["lang"] == "de")
        )
        base["context"]["guidelines"][0] = {
            "id": record["id"],
            "text": record["text"],
            "meta": r.record_meta(record),
            "via": "rrf",
        }
        base["messages"] = p.prompt.render_messages(
            row["text"], row["lang"], base["context"], "corrected"
        )
        entry = p.make_entry(row, base, "natural", self.mapping, self.records)
        c.audit_arm(row, base, entry, self.mapping, self.records)
        self.assertEqual(entry["context"]["guidelines"][0]["meta"]["lang"], "de")

    def test_metadata_never_changes_model_messages(self):
        row, base, original = self.entry()
        for key in ("gate", "templ_id", "ref_case_id", "text_sha256"):
            row[key] = "PRIVATE_METADATA_CANARY_" + key
        changed = p.make_entry(row, base, "matched_rule", self.mapping, self.records)
        self.assertEqual(changed["messages"], original["messages"])
        self.assertNotEqual(changed["input_sha256"], original["input_sha256"])
        self.assertNotIn("PRIVATE_METADATA_CANARY", r.js(changed["messages"]).decode("utf8"))

    def test_reject_missing_duplicate_unknown_rules(self):
        row, base = self.fixture()
        mapping = deepcopy(self.mapping)
        mapping[row["functionality"]]["filler"] = mapping[row["functionality"]]["matched"]
        with self.assertRaisesRegex(RuntimeError, "Invalid rule mapping"):
            p.make_entry(row, base, "matched_rule", mapping, self.records)
        mapping = deepcopy(self.mapping)
        mapping[row["functionality"]]["matched"] = "not-in-kb"
        with self.assertRaisesRegex(RuntimeError, "Missing forced rule"):
            p.make_entry(row, base, "matched_rule", mapping, self.records)
        with self.assertRaisesRegex(RuntimeError, "Unknown condition"):
            p.make_entry(row, base, "new_condition", self.mapping, self.records)

    def test_reject_definition_or_example_change(self):
        for field in ("definitions", "example_groups"):
            row, base, entry = self.entry()
            target = (
                entry["context"][field][0]
                if field == "definitions"
                else entry["context"][field][0][1][0]
            )
            target["text"] += " changed"
            with self.assertRaises(RuntimeError):
                p.validate_arm(row, base, entry, self.mapping, self.records)
            with self.assertRaises(RuntimeError):
                c.audit_arm(row, base, entry, self.mapping, self.records)

    def test_reject_rule_change_and_reordered_lead(self):
        for mode in ("text", "order", "metadata"):
            row, base, entry = self.entry()
            if mode == "text":
                entry["context"]["guidelines"][0]["text"] += " changed"
            elif mode == "metadata":
                entry["context"]["guidelines"][0]["meta"]["source"] = "fabricated"
            else:
                entry["context"]["guidelines"].reverse()
            entry["context_sha256"] = r.signature(entry["context"])
            with self.assertRaises(RuntimeError):
                c.audit_arm(row, base, entry, self.mapping, self.records)

    def test_reject_rehashed_message_tampering(self):
        for message in (0, 1):
            row, base, entry = self.entry()
            entry["messages"][message]["text"] += "\nGold is non-hate."
            entry["messages_sha256"] = r.signature(entry["messages"])
            with self.assertRaisesRegex(RuntimeError, "Independent arm rendering"):
                c.audit_arm(row, base, entry, self.mapping, self.records)

    def test_renderer_distinguishes_unknown_false_and_empty(self):
        row, base = self.fixture()
        for gate in (None, False, True):
            for lists in (None, []):
                ctx = deepcopy(base["context"])
                meta = ctx["example_groups"][0][1][0]["meta"]
                meta.update(
                    gate=gate, target_groups=lists, hate_types=lists, legal=lists, severity=None
                )
                expected = p.prompt.render_messages(row["text"], row["lang"], ctx, "corrected")
                self.assertEqual(c.render_independently(row["text"], row["lang"], ctx), expected)

    def test_reject_incomplete_duplicate_reordered_matrix(self):
        row, base = self.fixture()
        entries = [
            p.make_entry(row, base, name, self.mapping, self.records) for name in p.CONDITIONS
        ]
        c.assert_matrix([row], [base], entries)
        for altered in (entries[:-1], list(reversed(entries)), entries + [entries[0]]):
            with self.assertRaises(RuntimeError):
                c.assert_matrix([row], [base], altered)

    def test_replay_preserves_real_difference_and_boolean_type(self):
        row, base, entry = self.entry()
        new = deepcopy(entry)
        self.assertTrue(
            all(c.replay_flags(entry, new, ("context", "messages", "retrieved")).values())
        )
        new["messages"][1]["text"] += " changed"
        self.assertFalse(c.replay_flags(entry, new, ("context", "messages", "retrieved"))["entry"])
        self.assertFalse(p.compare({"x": False}, {"x": 0}, ("x",))["x"])

    def test_cache_missing_query_fails_without_fallback(self):
        vector = np.zeros((1, 1024), dtype=np.float32)
        vector[0, 0] = 1
        encoder = r.CachedEncoder(["known"], vector)
        np.testing.assert_array_equal(encoder.encode(["known"]), vector)
        with self.assertRaisesRegex(RuntimeError, "Uncached query"):
            encoder.encode(["unknown"])

    def test_offline_guard_blocks_network(self):
        with p.offline(), self.assertRaisesRegex(RuntimeError, "forbidden"):
            socket.create_connection(("localhost", 1))

    def test_save_preserves_existing_and_stays_inside_directory(self):
        with tempfile.TemporaryDirectory(dir=p.IMPL / "development") as directory, patch.object(
            p, "IMPL", Path(directory)
        ):
            target = Path(directory) / "test.json"
            p.save(target, {"a": 1})
            before = target.read_bytes()
            with self.assertRaisesRegex(RuntimeError, "Preserve existing"):
                p.save(target, {"a": 2})
            self.assertEqual(target.read_bytes(), before)
            with self.assertRaisesRegex(RuntimeError, "outside"):
                p.save(Path(directory).parent / "escaped.json", {})

    def test_completed_launcher_sends_no_work(self):
        with patch.object(p, "lock", return_value=nullcontext()), patch.object(
            p, "validate"
        ), patch.object(p, "receipt", return_value={"complete": True}), patch.object(
            p.subprocess, "Popen"
        ) as spawn:
            p.run("run")
            spawn.assert_not_called()

    def test_failed_attempt_not_silently_retried(self):
        with tempfile.TemporaryDirectory(dir=p.IMPL / "development") as directory, patch.object(
            p, "IMPL", Path(directory)
        ), patch.object(p, "validate", return_value={}), patch.object(
            p, "receipt", return_value=None
        ):
            (Path(directory) / "runs/smoke").mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, "Prior incomplete run"):
                p.worker("smoke")


class RankingArithmetic(unittest.TestCase):

    def fixture(self):
        return [
            {"id": rid, "kind": "definition", "text": text}
            for rid, text in [
                ("b", "rare same"),
                ("a", "rare same"),
                ("c", "third unique"),
                ("d", "fourth token"),
                ("e", "fifth extra"),
            ]
        ]

    def test_hand_computed_rare_token_and_stable_tie(self):
        hits = c.OrderedLexicon(self.fixture()).top("rare", 10)
        self.assertEqual([x["id"] for x in hits], ["b", "a"])
        for hit in hits:
            self.assertAlmostEqual(hit["score"], math.log(3.5) - math.log(2.5), places=14)

    def test_no_overlap_no_padding(self):
        self.assertEqual(c.OrderedLexicon(self.fixture()).top("absent", 10), [])
        self.assertEqual(
            c.OrderedLexicon([{"id": "x", "kind": "definition", "text": "word"}]).top("word", 10),
            [],
        )

    def test_corpus_and_duplicate_query_token_arithmetic(self):
        import re
        from rank_bm25 import BM25Okapi

        records = self.fixture() + [
            {"id": "z", "kind": "definition", "text": "RARE rare same extra long word"}
        ]
        source = BM25Okapi([re.findall("\\w+", x["text"].lower()) for x in records])
        independent = c.OrderedLexicon(records)
        for query in ("rare rare!", "same unique third", "unseen", "rare extra"):
            scores = source.get_scores(re.findall("\\w+", query.lower()))
            order = [i for i in np.argsort(-scores, kind="stable") if scores[i] > 0]
            expected = [
                {
                    "id": records[i]["id"],
                    "kind": "definition",
                    "score": float(scores[i]),
                    "via": "bm25",
                    "ranks": {},
                }
                for i in order
            ]
            self.assertEqual(independent.top(query, 10), expected)

    def test_reject_rank_and_score_mutation(self):
        hits = c.OrderedLexicon(self.fixture()).top("rare", 10)
        with self.assertRaises(RuntimeError):
            c.a.compare_hits(list(reversed(hits)), hits, 1e-10)
        other = deepcopy(hits)
        other[0]["score"] += 0.001
        with self.assertRaises(RuntimeError):
            c.a.compare_hits(other, hits, 1e-10)


if __name__ == "__main__":
    unittest.main()

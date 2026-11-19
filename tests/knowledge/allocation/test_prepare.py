from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from analysis.knowledge.allocation import prepare as p
from analysis.knowledge.allocation import verify_legacy_preparation as check


class AllocationPreparationTests(unittest.TestCase):

    def fixture(self):
        records = {}
        hits = []
        for dim, n in p.QUOTAS.items():
            for i in range(n):
                record = {
                    "id": f"{dim}-{i}",
                    "kind": "definition",
                    "text": "definition " + dim + str(i),
                    "lang": "en",
                    "source": "fixture",
                    "dimension": dim,
                    "label": str(i),
                }
                records[record["id"]] = record
                hits.append(
                    {
                        "id": record["id"],
                        "text": record["text"],
                        "meta": p.r.record_meta(record),
                        "via": "rrf",
                    }
                )
        source = {
            "key": "en/x",
            "item_id": "x",
            "text": "input",
            "lang": "en",
            "scopes": [{"subset": "en_dev_eval_main", "dimension": "hate"}],
            "context": {
                "definitions": deepcopy(hits[:2]),
                "guidelines": [{"text": "fixed guidance"}],
                "example_groups": [
                    [
                        "Labelled examples",
                        [{"id": "ex1", "meta": {"gate": None, "legal": []}, "text": "unchanged"}],
                    ]
                ],
            },
            "messages": [{"role": "system", "text": "system"}, {"role": "user", "text": "input"}],
        }
        return (records, hits, source)

    def test_frozen_background_and_null_annotation(self):
        records, hits, source = self.fixture()
        ctx = deepcopy(source["context"])
        ctx["definitions"] = hits
        p.validate_context(source, ctx, "proportional7", records)
        for field, value in [("gate", False), ("legal", None)]:
            bad = deepcopy(ctx)
            bad["example_groups"][0][1][0]["meta"][field] = value
            with self.assertRaisesRegex(p.a.IntegrityError, "Background"):
                p.validate_context(source, bad, "proportional7", records)

    def test_guidelines_cannot_change(self):
        records, hits, source = self.fixture()
        ctx = deepcopy(source["context"])
        ctx["definitions"] = hits
        ctx["guidelines"][0]["text"] = "changed"
        with self.assertRaisesRegex(p.a.IntegrityError, "Background"):
            p.validate_context(source, ctx, "proportional7", records)

    def test_flat7_must_extend_same_first_two(self):
        records, hits, source = self.fixture()
        ctx = deepcopy(source["context"])
        ctx["definitions"] = hits
        p.validate_context(source, ctx, "flat7", records)
        ctx["definitions"] = hits[1:] + hits[:1]
        with self.assertRaisesRegex(p.a.IntegrityError, "extend"):
            p.validate_context(source, ctx, "flat7", records)

    def test_proportional_order_and_duplicates(self):
        records, hits, source = self.fixture()
        ctx = deepcopy(source["context"])
        ctx["definitions"] = list(reversed(hits))
        with self.assertRaisesRegex(p.a.IntegrityError, "quotas/order"):
            p.validate_context(source, ctx, "proportional7", records)
        ctx["definitions"] = hits[:-1] + hits[:1]
        with self.assertRaisesRegex(p.a.IntegrityError, "count/duplicate"):
            p.validate_context(source, ctx, "flat7", records)

    def test_changed_definition_text_rejected(self):
        records, hits, source = self.fixture()
        ctx = deepcopy(source["context"])
        ctx["definitions"][0]["text"] = "rewritten"
        with self.assertRaisesRegex(p.a.IntegrityError, "source"):
            p.validate_context(source, ctx, "flat2", records)

    def test_flat_expansion_uses_existing_rrf(self):
        records, hits, source = self.fixture()
        ids = list(records)
        ranked = lambda rr: [
            {"id": rid, "kind": "definition", "score": 1 - i * 0.01, "via": "dense", "ranks": {}}
            for i, rid in enumerate(rr)
        ]
        dense = {
            "operation": "dense_query",
            "where": {"kind": "definition"},
            "k": 10,
            "hits": ranked(ids),
        }
        bm = {
            "operation": "bm25_query",
            "bucket": "definition",
            "k": 10,
            "hits": ranked(list(reversed(ids))),
        }
        trace = [dense, bm, {"operation": "rrf"}]
        two = p.definitions_from_trace(trace, records, 2)
        seven = p.definitions_from_trace(trace, records, 7)
        self.assertEqual(two, seven[:2])
        self.assertEqual(len(seven), 7)
        independent = check.independent.independent_rrf(
            {"dense": dense["hits"], "bm25": bm["hits"]}, 7
        )
        self.assertEqual([x["id"] for x in seven], [x["id"] for x in independent])

    def test_unknown_condition_rejected(self):
        records, hits, source = self.fixture()
        with self.assertRaisesRegex(p.a.IntegrityError, "Unknown"):
            p.validate_context(source, source["context"], "oracle", records)

    def test_output_preservation(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(p, "BASE", Path(folder) / "new"):
            with self.assertRaisesRegex(p.a.IntegrityError, "outside"):
                p.save(Path(folder) / "old.json", {})
            p.save(p.BASE / "new.json", {"kept": True})
            with self.assertRaises((p.a.IntegrityError, FileExistsError)):
                p.save(p.BASE / "new.json", {"kept": False})
            self.assertEqual(p.a.read(p.BASE / "new.json"), {"kept": True})

    def test_replay_mismatch_must_be_saved(self):
        row = {
            "key": "en/x",
            "condition": "flat2",
            **{k: [] for k in ("context", "messages", "retrieved", "definition_trace")},
        }
        flags = {k: True for k in ("context", "messages", "retrieved", "definition_trace", "entry")}
        self.assertEqual(
            check.replay_entry(row, {"key": "en/x", "condition": "flat2", "matches": flags}), row
        )
        flags["messages"] = False
        with self.assertRaisesRegex(p.a.IntegrityError, "Hidden"):
            check.replay_entry(row, {"key": "en/x", "condition": "flat2", "matches": flags})

    def test_six_predeclared_focal_comparisons(self):
        old = {
            "endpoints_inherited_masks_and_labels": {
                s: {"role": "old"} for s in p.previous.SCOPE_COUNTS
            }
        }
        plan = p.analysis_plan(old)
        self.assertEqual(len(plan["focal_endpoints"]) * len(plan["contrasts"]), 6)
        self.assertAlmostEqual(plan["focal_family_confidence"], 1 - 0.05 / 6)
        self.assertEqual(plan["endpoints"]["de_legal_dev_eval"]["role"], "secondary")
        self.assertEqual(plan["endpoints"]["en_dev_eval_types"]["role"], "sensitivity")


if __name__ == "__main__":
    unittest.main()

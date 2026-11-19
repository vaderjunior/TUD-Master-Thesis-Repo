import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from analysis.knowledge.banks import german_prepare as prep
from analysis.knowledge.banks import german_resume as run
from analysis.common import prompt as prompt


class GermanComparisonTests(unittest.TestCase):

    def test_cosine_storage_rounding_tolerance_is_bounded(self):
        import numpy as np

        ref = np.full(1024, 1 / 32, dtype=np.float32)
        stored = ref.copy()
        stored[0] += 1e-08
        self.assertTrue(run.verify_index_vector(stored, ref))
        stored[0] += 0.001
        self.assertFalse(run.verify_index_vector(stored, ref))

    def test_missing_gate_and_empty_legal_preserved(self):
        self.assertEqual(
            prompt.corrected_gold({"gate": None, "legal": ["propaganda"]}),
            "legal=[propaganda]",
        )
        self.assertEqual(prompt.corrected_gold({"gate": None, "legal": []}), "legal=[]")
        self.assertEqual(
            prompt.corrected_gold({"gate": False, "legal": None}), "hate=false"
        )
        with self.assertRaises(ValueError):
            prompt.corrected_gold({"gate": "__none__"})

    def test_existing_english_labels_unchanged(self):
        if not (prep.ROOT / "kb/records.jsonl").is_file():
            self.skipTest("Original English knowledge records are absent")
        records = prep.u.read_lines(prep.ROOT / "kb/records.jsonl")
        en = [r for r in records if r["kind"] == "example" and r["lang"] == "en"]
        self.assertEqual(len(en), 149)
        for r in en:
            self.assertEqual(
                prompt.module("legacy")._render_gold(r["meta"]),
                prompt.corrected_gold(r["meta"]),
            )

    def test_real_template_separates_illustrations_and_gold(self):
        hit = {
            "id": "x",
            "text": "Example text",
            "via": "hybrid",
            "meta": {"gate": None, "legal": ["propaganda"], "illustrative_only": False},
        }
        context = {
            "definitions": [],
            "guidelines": [],
            "example_groups": [["Labelled examples", [hit]]],
        }
        legacy = prompt.render_messages("Probe", "de", context, "legacy")
        fixed = prompt.render_messages("Probe", "de", context, "corrected")
        self.assertEqual(legacy[0], fixed[0])
        self.assertIn("labels: hate=false", legacy[1]["text"])
        self.assertIn("labels: legal=[propaganda]", fixed[1]["text"])
        hit["meta"] = {"illustrative_only": True, "stgb": "185", "gate": True}
        context["example_groups"][0][0] = "Legal context illustrations"
        rendered = prompt.render_messages("Probe", "de", context, "corrected")[1][
            "text"
        ]
        self.assertIn("flagged under StGB §185", rendered)
        self.assertNotIn("labels:", rendered)

    def test_complementary_annotations_do_not_promote_paragraph_flags(self):
        rows = [
            {
                "id": "detox-1",
                "source": "detox",
                "gate": True,
                "raw": json.dumps({"legal_paragraphs": {"p_185": 1.0}}),
            },
            {
                "id": "botox-1",
                "source": "botox",
                "gate": None,
                "legal": [],
                "raw": "{}",
            },
        ]
        meta = prep.merged_meta(rows)
        self.assertIs(meta["gate"], True)
        self.assertEqual(meta["legal"], [])
        self.assertEqual(meta["astra_aux_stgb"], "185")
        self.assertIn("legal=[]", prompt.corrected_gold(meta))
        rows[1]["gate"] = False
        with self.assertRaises(RuntimeError):
            prep.merged_meta(rows)

    def test_pool_nesting_preserves_original_and_combines_selected_alias(self):
        raw = [
            {
                "id": "detox-1",
                "source": "detox",
                "lang": "de",
                "text": "Some text",
                "gate": True,
                "raw": "{}",
            },
            {
                "id": "botox-2",
                "source": "botox",
                "lang": "de",
                "text": "Some  text",
                "gate": None,
                "legal": ["propaganda"],
                "raw": "{}",
            },
            {
                "id": "gahd-3",
                "source": "gahd",
                "lang": "de",
                "text": "Different",
                "gate": False,
                "raw": "{}",
            },
        ]
        old = {
            "id": "ex-detox-legal-1",
            "kind": "example",
            "lang": "de",
            "dimension": None,
            "label": None,
            "text": "Some text",
            "source": "detox-train-legal",
            "meta": {"gate": True, "stgb": "185", "illustrative_only": True},
        }
        before = copy.deepcopy(old)
        pools, provenance = prep.build_pools(raw, [old], set())
        self.assertEqual(pools["original_small"], [before])
        self.assertEqual(old, before)
        self.assertEqual(len(pools["canonical_full"]), 2)
        self.assertEqual(pools["canonical_small"][0], pools["canonical_full"][0])
        self.assertEqual(pools["canonical_small"][0]["meta"]["legal"], ["propaganda"])
        self.assertFalse(pools["canonical_small"][0]["meta"]["illustrative_only"])

    def test_output_paths_and_immutable_writes(self):
        with tempfile.TemporaryDirectory() as t, patch.object(prep, "BASE", Path(t)):
            path = Path(t) / "test.json"
            prep.write(path, b"first")
            prep.write(path, b"first")
            with self.assertRaises(RuntimeError):
                prep.write(path, b"changed")
            with self.assertRaises(RuntimeError):
                prep.write(Path(t) / "../bad.json", b"bad")

    def test_classifier_injection_and_item_resume_with_mock_only(self):
        with patch("dotenv.load_dotenv", return_value=False):
            from src.hsrag.classify import classify as original
            import src.hsrag.classify as original_module
        old_builder = original_module.build_prompt
        with tempfile.TemporaryDirectory() as t, patch.object(
            prep, "BASE", Path(t)
        ), patch.object(run, "BASE", Path(t)), patch.object(
            run, "PLAN", Path(t) / "plan.json"
        ), patch.object(
            run, "AMENDMENT", Path(t) / "amendment.json"
        ):
            root = Path(t)
            (root / "plan.json").write_text("{}")
            (root / "amendment.json").write_text("{}")
            (root / "kb/small").mkdir(parents=True)
            (root / "kb/small/records.jsonl").write_text("{}")
            plan = {
                "code_version": "old",
                "prompt_version": "template",
                "taxonomy_version": original_module.taxonomy_version(),
                "renderer_sha256": prep.u.digest(
                    prep.ROOT / "analysis/common/prompt.py"
                ),
                "model": "MOCK-ONLY",
                "retrieval": {"k_examples": 5},
            }
            spec = {
                "name": "test",
                "pool": "small",
                "renderer": "corrected",
                "condition": "b_fixed",
                "subset": "synthetic",
                "n_items": 1,
            }
            ctx = {"definitions": [], "guidelines": [], "example_groups": []}
            msg = prompt.render_messages("Engineering probe", "de", ctx, "corrected")
            entry = {
                "item_id": "probe",
                "text": "Engineering probe",
                "lang": "de",
                "subset": "synthetic",
                "context": ctx,
                "messages": msg,
                "messages_sha256": run.signature(msg),
                "context_sha256": run.signature(ctx),
                "retrieved": prompt.hit_ids(ctx),
            }

            class Mock:
                active_model = "MOCK-ONLY"

                def __init__(self):
                    self.calls = 0

                def complete(self, messages):
                    self.calls += 1
                    self.assertion = messages == msg
                    return json.dumps(
                        {
                            "reasoning": "Engineering probe",
                            "hate": False,
                            "target_group": [],
                            "hate_type": [],
                            "severity": None,
                            "legal": ["insult_defamation"],
                        }
                    )

            mock = Mock()
            row = run.infer(plan, spec, entry, mock)
            self.assertEqual(mock.calls, 3)
            self.assertTrue(mock.assertion)
            self.assertEqual(row["result"]["legal"], ["insult_defamation"])
            self.assertFalse(row["result"]["hate"])
            self.assertEqual(row["n_valid"], 3)
            self.assertIs(original_module.build_prompt, old_builder)
            prep.write(root / "prompts_b_fixed.jsonl", prep.lines([entry]))
            prep.write(run.item_path(spec, entry), prep.js(row))
            complete, missing = run.check_run(plan, spec)
            self.assertEqual(len(complete), 1)
            self.assertEqual(missing, [])
            altered = copy.deepcopy(row)
            altered["astra"]["messages_sha256"] = "bad"
            with self.assertRaises(RuntimeError):
                run.check_item(altered, plan, spec, entry)

    def test_client_configuration_without_credentials_or_requests(self):
        with patch("dotenv.load_dotenv", return_value=False):
            import src.hsrag.client as clients
        with patch.object(clients, "make_client", return_value="fake") as factory:
            self.assertEqual(
                run.client_factory({"model": "pinned", "temperature": 1.0}), "fake"
            )
            self.assertEqual(factory.call_args.kwargs["timeout"], 180)
            self.assertEqual(factory.call_args.kwargs["max_tokens"], 4096)
            self.assertFalse(factory.call_args.kwargs["allow_fallback"])


if __name__ == "__main__":
    unittest.main()

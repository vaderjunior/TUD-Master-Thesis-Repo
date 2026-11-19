import json
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from analysis.knowledge.banks import german_audit as audit


class GermanAuditTests(unittest.TestCase):

    def test_illustration_alias_cannot_escape_exclusions(self):
        self.assertEqual(audit.canonical_id("ex-detox-legal-123"), "detox-123")
        self.assertEqual(audit.canonical_id("ex-detox-123"), "detox-123")
        self.assertEqual(audit.canonical_id("ex-botox-42"), "botox-42")
        self.assertEqual(audit.canonical_id("gahd-a"), "gahd-a")

    def test_current_renderer_keeps_missing_gate_and_known_legal(self):
        render, _ = audit.extract_renderer()
        self.assertEqual(
            render({"gate": None, "legal": ["propaganda"]}), "legal=[propaganda]"
        )
        self.assertEqual(render({"gate": None, "legal": []}), "legal=[]")
        self.assertEqual(
            render({"gate": True, "hate_types": ["derogation"]}),
            "hate=true, hate_type=[derogation]",
        )

    def test_all_training_neighbors_are_flagged_not_only_eval_winner(self):
        tr = np.asarray([[1, 0], [0.99, 0.01], [0, 1]], dtype=np.float32)
        ev = np.asarray([[1, 0], [0.7, 0.7]], dtype=np.float32)
        mx, ix = audit.nearest(tr, ev, 1, 1)
        np.testing.assert_allclose(mx, [1, 0.99, 0.7])
        np.testing.assert_array_equal(ix, [0, 0, 1])
        self.assertEqual(int((mx >= 0.95).sum()), 2)

    def test_chunked_similarity_matches_full_matrix_and_ties(self):
        rng = np.random.default_rng(5)
        tr = rng.normal(size=(13, 6)).astype(np.float32)
        ev = rng.normal(size=(9, 6)).astype(np.float32)
        ev[8] = ev[0]
        mx, ix = audit.nearest(tr, ev, 4, 3)
        dense = tr @ ev.T
        np.testing.assert_allclose(mx, dense.max(axis=1), atol=1e-06)
        np.testing.assert_array_equal(ix, dense.argmax(axis=1))

    def test_resume_validates_content_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            audit, "BASE", Path(tmp)
        ):
            p = Path(tmp) / "chunks/a.npz"
            ident = {"rows": 2, "texts_sha256": "first"}
            arr = np.zeros((2, 1024), dtype=np.float32)
            arr[:, 0] = 1
            audit.save_chunk(p, ident, arr)
            np.testing.assert_array_equal(audit.load_chunk(p, ident), arr)
            with self.assertRaises(RuntimeError):
                audit.load_chunk(p, {**ident, "texts_sha256": "changed"})
            with self.assertRaises(RuntimeError):
                audit.save_chunk(p, ident, arr)
            with np.load(p, allow_pickle=False) as data:
                meta = data["metadata"].copy()
            arr[:, 0] = -1
            with p.open("wb") as f:
                np.savez(f, vectors=arr, metadata=meta)
            with self.assertRaisesRegex(RuntimeError, "payload changed"):
                audit.load_chunk(p, ident)

    def test_output_guard_and_immutable_json(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            audit, "BASE", Path(tmp)
        ):
            p = Path(tmp) / "result.json"
            audit.write_once(p, b"one")
            audit.write_once(p, b"one")
            with self.assertRaises(RuntimeError):
                audit.write_once(p, b"two")
            with self.assertRaises(RuntimeError):
                audit.write_once(Path(tmp) / "../outside.json", b"x")
            self.assertEqual(p.read_bytes(), b"one")

    def test_encoding_resume_never_calls_model_for_finished_chunks(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            audit, "BASE", Path(tmp)
        ):
            plan_path = Path(tmp) / "plan.json"
            plan_path.write_text("{}")
            plan = {
                "cache_chunk": 2,
                "batch_size": 1,
                "model": "synthetic",
                "precision": "float32",
                "embedding_cache": {"revision": "test"},
            }
            calls = []

            class Model:

                def encode(self, texts, **kwargs):
                    calls.append(texts)
                    x = np.zeros((len(texts), 1024), dtype=np.float32)
                    x[:, 0] = 1
                    return x

            def forbidden_model():
                raise AssertionError("A completed chunk loaded the model")

            with patch.object(audit, "PLAN", plan_path), redirect_stdout(io.StringIO()):
                first, paths = audit.encode_rows(plan, ["a", "b", "c"], "test", Model)
                second, again = audit.encode_rows(
                    plan, ["a", "b", "c"], "test", forbidden_model
                )
            self.assertEqual(len(calls), 2)
            self.assertEqual(paths, again)
            np.testing.assert_array_equal(first, second)

    def test_finalize_propagates_duplicate_alias_exclusions(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            audit, "BASE", Path(tmp)
        ):
            root = Path(tmp)
            (root / "plan.json").write_text("{}")
            plan = {"threshold": 0.95, "very_near": 0.98}
            train = [
                {
                    "id": "detox-1",
                    "split": "de_train",
                    "source": "detox",
                    "text": "Same text",
                },
                {
                    "id": "botox-2",
                    "split": "de_legal_train",
                    "source": "botox",
                    "text": "same  TEXT",
                },
                {
                    "id": "gahd-3",
                    "split": "de_train",
                    "source": "gahd",
                    "text": "different",
                },
            ]
            held = [{"text": "held", "references": [{"id": "h", "file": "heldout"}]}]
            survey = {"cheap_excluded_train_ids": ["detox-1"]}
            original_read = audit.util.read_json

            def reader(p):
                return (
                    survey if Path(p).name == "preparation.json" else original_read(p)
                )

            with patch.object(audit, "PLAN", root / "plan.json"), patch.object(
                audit.util, "read_json", side_effect=reader
            ), patch.object(audit.util, "read_lines", return_value=[]), patch.object(
                audit.util, "rel", side_effect=lambda p: Path(p).name
            ):
                with redirect_stdout(io.StringIO()):
                    audit.finalize(
                        plan,
                        train,
                        held,
                        np.array([0.5, 0.4, 0.3]),
                        np.zeros(3, dtype=int),
                        [],
                        {},
                    )
            report = json.loads((root / "report.json").read_text())
            self.assertEqual(report["proposed_excluded_ids"], ["botox-2", "detox-1"])
            self.assertEqual(report["remaining_train_rows"], 1)


if __name__ == "__main__":
    unittest.main()

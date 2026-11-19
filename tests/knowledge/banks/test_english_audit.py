from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from analysis.knowledge.banks import english_audit as audit


class EnglishAuditTests(unittest.TestCase):

    def test_all_train_neighbors_and_ties_match_dense_arithmetic(self):
        a = np.array([[1, 0], [0.99, 0.01], [0, 1]], dtype=np.float32)
        b = np.array([[1, 0], [0.7, 0.7], [1, 0]], dtype=np.float32)
        with redirect_stdout(io.StringIO()):
            mx, ix = audit.numerical.nearest(a, b, 1, 1)
        np.testing.assert_allclose(mx, (a @ b.T).max(1))
        np.testing.assert_array_equal(ix, (a @ b.T).argmax(1))
        self.assertEqual(int((mx >= 0.95).sum()), 2)

    def test_blocked_random_similarity_matches_dense(self):
        rng = np.random.default_rng(17)
        a = rng.normal(size=(23, 13)).astype("float32")
        b = rng.normal(size=(19, 13)).astype("float32")
        with redirect_stdout(io.StringIO()):
            mx, ix = audit.numerical.nearest(a, b, 7, 4)
        np.testing.assert_allclose(mx, (a @ b.T).max(1), atol=2e-06)
        np.testing.assert_array_equal(ix, (a @ b.T).argmax(1))

    def test_immutable_output_and_path_guard(self):
        with tempfile.TemporaryDirectory() as t, patch.object(audit, "BASE", Path(t)):
            path = Path(t) / "file.json"
            audit.write_once(path, b"first")
            audit.write_once(path, b"first")
            with self.assertRaises(RuntimeError):
                audit.write_once(path, b"changed")
            with self.assertRaises(RuntimeError):
                audit.write_once(Path(t).parent / "outside.json", b"bad")

    def test_cache_detects_content_identity_and_payload_changes(self):
        with tempfile.TemporaryDirectory() as t, patch.object(audit, "BASE", Path(t)):
            path = Path(t) / "a.npz"
            ident = {"rows": 2, "texts_sha256": "original"}
            vv = np.zeros((2, 1024), dtype="float32")
            vv[:, 0] = 1
            audit.save_chunk(path, ident, vv)
            np.testing.assert_array_equal(audit.numerical.load_chunk(path, ident), vv)
            with self.assertRaises(RuntimeError):
                audit.numerical.load_chunk(path, {**ident, "texts_sha256": "changed"})
            with self.assertRaises(RuntimeError):
                audit.save_chunk(path, ident, vv)
            with np.load(path, allow_pickle=False) as d:
                meta = d["metadata"].copy()
            vv[:, 0] = -1
            with path.open("wb") as f:
                np.savez(f, vectors=vv, metadata=meta)
            with self.assertRaisesRegex(RuntimeError, "payload changed"):
                audit.numerical.load_chunk(path, ident)

    def test_resume_does_not_load_model_and_plan_changes_do_not_reuse(self):
        with tempfile.TemporaryDirectory() as t, patch.object(
            audit, "BASE", Path(t)
        ), patch.object(audit, "PLAN", Path(t) / "plan.json"):
            audit.PLAN.write_text("{}")
            calls = []
            plan = {
                "cache_chunk": 2,
                "batch_size": 1,
                "model": "synthetic",
                "precision": "float32",
                "embedding_cache": {"revision": "mock"},
            }

            class Model:

                def encode(self, texts, **kwargs):
                    calls.append(texts)
                    vv = np.zeros((len(texts), 1024), dtype="float32")
                    vv[:, 0] = 1
                    return vv

            def forbidden():
                raise AssertionError("Loaded model for completed chunks")

            with redirect_stdout(io.StringIO()):
                aa, paths = audit.encode_rows(plan, ["a", "b", "c"], "probe", Model)
                bb, again = audit.encode_rows(plan, ["a", "b", "c"], "probe", forbidden)
                self.assertEqual(len(calls), 2)
                self.assertEqual(paths, again)
                np.testing.assert_array_equal(aa, bb)
                audit.PLAN.write_text('{"changed":true}')
                with self.assertRaises(AssertionError):
                    audit.encode_rows(plan, ["a", "b", "c"], "probe", forbidden)

    def test_direct_historical_pair_is_checked_even_when_not_nearest(self):
        train = [{"id": "t", "text": "train"}]
        held = [
            {"text": "near", "references": [{"id": "near"}]},
            {"text": "old", "references": [{"id": "old"}]},
        ]
        a = np.array([[1, 0]], dtype="float32")
        b = np.array([[1, 0], [0.94, 0.1]], dtype="float32")
        warnings = [
            {
                "train_id": "t",
                "eval_id": "old",
                "subset": "HateCheck",
                "cosine": 0.953,
                "in_previous_full": True,
                "in_previous_small": False,
            }
        ]
        out = audit.known_pair_scores(train, held, a, b, warnings)
        self.assertAlmostEqual(out[0]["fresh_cosine"], 0.94, places=6)
        self.assertEqual(out[0]["reported_historical_cosine"], 0.953)

    def test_final_report_exclusions_review_band_and_completion_integrity(self):
        with tempfile.TemporaryDirectory() as t, patch.object(
            audit, "ROOT", Path(t)
        ), patch.object(audit, "BASE", Path(t) / "run"), patch.object(
            audit, "PLAN", Path(t) / "run/plan.json"
        ):
            audit.BASE.mkdir()
            audit.PLAN.write_text("{}")
            survey = {
                "cheap_excluded_ids": ["old"],
                "previous_small_ids": ["old", "near", "boundary", "kept"],
            }
            audit.write_once(audit.BASE / "preparation.json", audit.js(survey))
            plan = {"threshold": 0.95, "very_near": 0.98, "review_lower": 0.94}
            train = [
                {"id": i, "source": "synthetic", "text": i}
                for i in ("old", "near", "boundary", "kept")
            ]
            held = [{"text": "held", "references": [{"id": "h"}]}]
            with redirect_stdout(io.StringIO()):
                audit.finalize(
                    plan,
                    train,
                    held,
                    np.array([0.1, 0.96, 0.945, 0.2]),
                    np.zeros(4, dtype=int),
                    [],
                    {"synthetic": True},
                    [],
                )
            report = json.loads((audit.BASE / "report.json").read_text())
            self.assertEqual(report["proposed_excluded_ids"], ["near", "old"])
            self.assertEqual(report["new_excluded_ids"], ["near"])
            self.assertEqual(report["provisional_remaining_previous_small"], 2)
            review = [
                json.loads(l)
                for l in (audit.BASE / "review_pairs.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                {r["train_id"] for r in review}, {"old", "near", "boundary"}
            )
            self.assertTrue(audit.check_complete())
            (audit.BASE / "known_pair_checks.json").write_text('["changed"]')
            with self.assertRaises(RuntimeError):
                audit.check_complete()


if __name__ == "__main__":
    unittest.main()

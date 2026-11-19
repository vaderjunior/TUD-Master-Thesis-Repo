from copy import deepcopy
from pathlib import Path
import unittest
import uuid
from unittest.mock import patch
from analysis.feedback.insertion import common as c
from analysis.feedback.insertion import validate as check
from analysis.feedback.insertion import scores_earlier as score
from analysis.common import api as a


def synthetic_catalog():
    from analysis.feedback.insertion import run as api

    conditions = ["reference"] + [
        f"{arm}_s{seed}" for seed in c.SEEDS for arm in c.ARMS
    ]
    messages = [
        {"role": "system", "text": "OFFLINE SYNTHETIC TEST ONLY"},
        {"role": "user", "text": "Synthetic input; no research outcome."},
    ]
    entry = {
        "messages": messages,
        "messages_sha256": a.signature(messages),
        "schema_path": "config/taxonomy.yaml",
        "schema_sha256": c.sha(c.ROOT / "config/taxonomy.yaml"),
        "entry_sha256": "offline-test",
    }
    return {
        (f"synthetic-{i:04d}", condition): {
            **entry,
            "population": "benign" if i < 208 else "hateful_types",
        }
        for i in range(675)
        for condition in conditions
    }


class FiveSlotTests(unittest.TestCase):

    def record(self):
        return {
            "id": "added-source",
            "kind": "feedback",
            "lang": "en",
            "text": "Synthetic source.",
            "source": "test",
            "meta": {
                "gate": False,
                "hate_types": None,
                "target_groups": None,
                "severity": None,
                "legal": None,
            },
            "provenance": {"source_id": "source"},
        }

    def test_conversion_preserves_unknown_annotations_and_source(self):
        source = self.record()
        before = deepcopy(source)
        out = c.ordinary(source)
        self.assertEqual(source, before)
        self.assertEqual(out["kind"], "example")
        self.assertIsNone(out["dimension"])
        self.assertEqual(out["meta"], source["meta"])
        self.assertEqual(out["provenance"], source["provenance"])
        self.assertEqual(check.renderer.corrected_gold(out["meta"]), "hate=false")

    def test_copy_identity_does_not_request_distance_change(self):
        from analysis.feedback.insertion.prepare import metadata_identity

        origin = {
            "hnsw:space": "cosine",
            "embedding_model": "BAAI/bge-m3",
            "kb_version": "old",
            "plan_sha256": "old-plan",
        }
        out = metadata_identity(origin, "new-records-hash", "new-plan", "reference")
        self.assertNotIn("hnsw:space", out)
        self.assertEqual(origin["hnsw:space"], "cosine")
        self.assertEqual(out["astra_original_kb_version"], "old")
        self.assertEqual(out["kb_version"], "new-records-hash")

    def test_ranking_runtime_never_imports_neural_libraries(self):
        import sys
        from analysis.feedback.insertion.prepare import runtime

        before = set(sys.modules)
        rt = runtime()
        added = set(sys.modules) - before
        self.assertFalse(
            any(
                (
                    n == "torch"
                    or n.startswith("torch.")
                    or n == "sentence_transformers"
                    or n.startswith("sentence_transformers.")
                    for n in added
                )
            )
        )
        with self.assertRaises(AssertionError):
            rt.Retriever.__init__.__globals__["SentenceTransformer"]("anything")
        self.assertEqual(
            rt.kb.unsentinel_meta({"dimension": "", "hate_types": "__none__"}),
            {"dimension": None, "hate_types": None},
        )

    def test_reject_fabricated_fine_labels(self):
        source = self.record()
        source["meta"]["hate_types"] = []
        with self.assertRaises(AssertionError):
            c.ordinary(source)

    def test_reject_missing_binary_gold_and_language_mixing(self):
        for key, value in [("gate", None), ("gate", 1)]:
            source = self.record()
            source["meta"][key] = value
            with self.assertRaises(AssertionError):
                c.ordinary(source)
        source = self.record()
        source["lang"] = "de"
        with self.assertRaises(AssertionError):
            c.ordinary(source)

    def test_independent_rrf_ties_both_channels_and_depth(self):
        channels = {
            "dense": [{"id": "z", "score": 0.9}, {"id": "a", "score": 0.8}],
            "bm25": [{"id": "a", "score": 3.0}, {"id": "z", "score": 2.0}],
        }
        ids, ranks = check.fused_ids(channels)
        self.assertEqual(ids, ["a", "z"])
        self.assertEqual(ranks["a"], {"dense": 2, "bm25": 1})
        channels["bm25"][1]["score"] = 0
        with self.assertRaises(AssertionError):
            check.fused_ids(channels)

    def test_independent_mmr_matches_existing_diversity_and_ties(self):
        import ast
        from types import SimpleNamespace
        import numpy as np

        path = c.ROOT / "src/hsrag/retrieve.py"
        tree = ast.parse(path.read_text(encoding="utf8"))
        node = next(
            (
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "mmr_select"
            )
        )
        ns = {"Hit": SimpleNamespace}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
        for seed in range(8):
            rng = np.random.default_rng(seed)
            vectors = rng.normal(size=(30, 16))
            vectors[1] = vectors[0]
            query = rng.normal(size=16)
            ids = [str(i) for i in range(30)]
            hits = [
                SimpleNamespace(id=rid, vec=v.tolist()) for rid, v in zip(ids, vectors)
            ]
            expected = [h.id for h in ns["mmr_select"]([query.tolist()], hits, 0.7, 10)]
            actual = check.mmr_ids(
                query, [{"id": rid} for rid in ids], dict(zip(ids, vectors))
            )
            self.assertEqual(actual, expected)

    def test_chunk_resume_reuses_completed_inputs(self):
        folder = c.BASE / "execution/offline_check" / ("chunks_" + uuid.uuid4().hex)
        items = [{"key": str(i)} for i in range(5)]
        calls = []

        def make(n, row):
            calls.append(n)
            return {"key": row["key"], "value": n}

        first, receipt = c.gather(folder, items, make, {"test": True}, chunk_size=2)
        second, again = c.gather(
            folder, items, c.forbidden, {"test": True}, chunk_size=2
        )
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 5)
        self.assertEqual(again["reused_rows"], 5)
        with self.assertRaises(AssertionError):
            c.gather(
                folder, list(reversed(items)), c.forbidden, {"test": True}, chunk_size=2
            )

    def test_writes_cannot_escape_new_study(self):
        with self.assertRaises(AssertionError):
            c.own(c.CORE / "forbidden.json")

    def test_full_schedule_every_comparison_once(self):
        from analysis.feedback.insertion import run as api

        b = api.queue()
        cat = synthetic_catalog()
        schedule = b.make_schedule({api.STUDY: cat})
        checked = b.check.check_schedule(schedule, {api.STUDY: cat})
        self.assertEqual(checked["identities"], 38475)
        self.assertEqual(checked["ordinary_calls"], 115425)
        schedule["stages"][0]["jobs"][0]["repeat"] = 7
        with self.assertRaises(a.IntegrityError):
            b.check.check_schedule(schedule, {api.STUDY: cat})

    def test_old_queue_namespace_unchanged(self):
        from analysis.feedback.insertion import run as api
        from analysis.common import vote_api as original

        before = original.BASE
        b = api.queue()
        self.assertEqual(original.BASE, before)
        self.assertNotEqual(b.BASE, before)
        self.assertEqual(b.execute_stage.__globals__["BASE"], api.BASE)
        self.assertEqual(b.audit.b.BASE, api.BASE)

    def test_completed_mock_restarts_without_new_calls(self):
        from analysis.feedback.insertion import run as api

        b = api.queue()
        cat = synthetic_catalog()
        schedule = b.make_schedule({api.STUDY: cat})
        job = schedule["stages"][0]["jobs"][0]
        stage = {
            "study": api.STUDY,
            "id": "test_01",
            "input_keys": [job["key"]],
            "items": 1,
            "ordinary_calls": 3,
            "jobs": [job],
        }
        root = api.BASE / "offline_check" / ("single_" + uuid.uuid4().hex) / api.STUDY
        transport = b.check.SyntheticTransport()
        with c.offline():
            b.execute_stage(
                stage,
                cat,
                root,
                "synthetic-plan",
                mode="mock",
                client_factory=lambda: transport,
                sleep=lambda _: None,
            )
            b.audit.audit_stage(stage, cat, root, "synthetic-plan", mode="mock")
            b.execute_stage(
                stage,
                cat,
                root,
                "synthetic-plan",
                mode="mock",
                client_factory=c.forbidden,
            )
        self.assertEqual(transport.count, 3)
        with self.assertRaises(a.IntegrityError):
            b.checked_root(c.CORE, api.STUDY, "mock")

    def test_unknown_saved_request_refuses_resend(self):
        from analysis.feedback.insertion import run as api

        b = api.queue()
        cat = synthetic_catalog()
        job = b.make_schedule({api.STUDY: cat})["stages"][0]["jobs"][0]
        root = a.storage_path(
            api.BASE / "offline_check" / ("uncertain_" + uuid.uuid4().hex) / api.STUDY
        )
        backend = b.engines(cat)[job["schema_path"]]
        folder = b.item_dir(root, job, "synthetic-plan")
        expected = {
            "identity": b.identity(job, "synthetic-plan"),
            "messages_sha256": job["messages_sha256"],
            "contract": a.CONTRACT,
            "implementation": backend.implementation(),
            "execution_mode": "mock",
        }
        a.save_new(folder / "job.json", expected)
        a.save_new(
            folder / "calls/vote_0_turn_0/attempt_0/intent.json", {"unresolved": True}
        )
        with self.assertRaises(a.IncompleteAttempt):
            b.scan_jobs([job], cat, root, "synthetic-plan", "mock", b.engines(cat))

    def data(self):
        roster = [
            {
                "key": str(i),
                "cluster": "duplicate" if i < 2 else "singleton",
                "gold": {"hate": False},
            }
            for i in range(3)
        ]
        results = {}
        for row in roster:
            for condition in ["reference"] + [f"error_balanced_s{s}" for s in c.SEEDS]:
                for rep in range(3):
                    results[row["key"], condition, rep] = {
                        "result": {
                            "hate": condition == "reference"
                            and row["key"] in ["0", "1"]
                        },
                        "n_valid": 3,
                    }
        return (roster, results)

    def test_primary_keeps_cluster_and_shared_reference_structure(self):
        roster, results = self.data()
        out = score.primary(roster, results, False, 99, draws=100)
        self.assertEqual(out["clusters"], 2)
        self.assertEqual(out["common_n"], 3)
        self.assertAlmostEqual(out["difference"], -2 / 3)
        self.assertEqual(out["shared_reference_repeats"], 3)
        one = score.primary(roster, results, False, 99, representatives=True, draws=100)
        self.assertEqual(one["common_n"], 2)
        self.assertAlmostEqual(one["difference"], -0.5)

    def test_null_masks_bounds_and_strict_votes(self):
        roster, results = self.data()
        results["0", "reference", 0]["result"] = None
        out = score.primary(roster, results, False, 99, draws=100)
        self.assertEqual(out["excluded_ids"], ["0"])
        self.assertEqual(out["common_n"], 2)
        self.assertLess(
            out["full_roster_difference_missing_bounds"][0],
            out["full_roster_difference_missing_bounds"][1],
        )
        results["1", "reference", 0]["n_valid"] = 2
        self.assertEqual(
            score.primary(roster, results, False, 99, strict=True, draws=100)[
                "common_n"
            ],
            1,
        )
        for v in results.values():
            v["result"] = None
        self.assertEqual(
            score.primary(roster, results, False, 99, draws=100)["status"],
            "not_estimable_no_common_valid_inputs",
        )


if __name__ == "__main__":
    unittest.main()

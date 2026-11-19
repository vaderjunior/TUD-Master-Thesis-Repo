import contextlib
import io
from pathlib import Path
import socket
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from threadpoolctl import threadpool_limits
from training.cpu import common as c
from training.cpu import run as run


class Algebra(unittest.TestCase):
    def test_dense_sparse_and_constants(self):
        x = np.random.default_rng(6).normal(size=(30, 8))
        y = np.column_stack((np.arange(30) % 2, np.zeros(30), np.ones(30))).astype(int)
        a = c.fit_linear(x, y, 1.0, 42)
        b = c.fit_linear(sparse.csr_matrix(x), y, 1.0, 42)
        np.testing.assert_allclose(
            c.predict_linear(a, x),
            c.predict_linear(b, sparse.csr_matrix(x)),
            atol=1e-9,
            rtol=0,
        )
        self.assertEqual(a["constants"].tolist(), [-1, 0, 1])
        self.assertTrue((c.predict_linear(a, x)[:, 1] == 0).all())
        self.assertTrue((c.predict_linear(a, x)[:, 2] == 1).all())

    def test_gate_and_boundary_rules(self):
        p = np.array([[0.5], [0.5001]])
        self.assertEqual(c.decisions("hate", p, 0.5).tolist(), [[1, 0], [0, 1]])
        self.assertEqual(
            c.decisions("hate_type", p, 0.5, [False, True]).tolist(), [[0], [1]]
        )
        self.assertEqual(
            c.decisions("severity", np.array([[0.7, 0.7, 0.2]]), None).tolist(),
            [[1, 0, 0]],
        )

    def test_neural_softmax_not_binary_sigmoid(self):
        p = c.neural_probabilities("hate", [[2.0, 3.0], [4.0, 4.0]])
        np.testing.assert_allclose(p[:, 0], [1 / (1 + np.exp(-1)), 0.5])
        np.testing.assert_allclose(
            c.neural_probabilities("hate_type", [[0.0, 0.0]]), [[0.5, 0.5]]
        )
        self.assertEqual(
            c.decisions(
                "severity", c.neural_probabilities("severity", [[-1.0, 3.0, 2.0]]), None
            ).tolist(),
            [[0, 1, 0]],
        )

    def test_f1_selection_and_ties(self):
        y = np.array([[0], [0], [1], [1]])
        p = np.array([[0.1], [0.3], [0.7], [0.9]])
        selected = c.selection("hate", y, {10.0: p, 0.1: p, 1.0: p})
        self.assertEqual(selected["selected"], dict(C=0.1, threshold=0.5, score=1.0))
        equal = [dict(C=1.0, threshold=t, score=0.2) for t in (0.4, 0.6)]
        self.assertEqual(c.choose(equal)["threshold"], 0.6)

    def test_unsupported_labels_retained_but_excluded_from_compact_selection(self):
        y = np.array([[0, 1], [0, 0], [1, 0], [1, 0]])
        p = np.array([[0.2, 0.7], [0.2, 0.2], [0.7, 0.2], [0.7, 0.2]])
        selected = c.selection("hate_type", y, {1.0: p}, compact=True)
        self.assertEqual(selected["selection_indices"], [0])
        self.assertEqual(
            c.decisions("hate_type", p, selected["selected"]["threshold"]).shape, (4, 2)
        )
        native = c.selection("hate_type", y, {1.0: p})
        self.assertEqual(native["selection_indices"], [0, 1])

    def test_gated_selection_changes_score(self):
        y = np.array([[1, 0], [1, 0], [0, 1], [0, 1]])
        p = y * 0.8 + 0.1
        self.assertEqual(
            c.selection("hate_type", y, {1.0: p})["selected"]["score"], 1.0
        )
        self.assertEqual(
            c.selection("hate_type", y, {1.0: p}, np.zeros(4, dtype=bool))["selected"][
                "score"
            ],
            0.0,
        )

    def test_no_held_vocabulary_or_idf_fit(self):
        fit = ["ordinary alpha", "ordinary beta", "alpha gamma"]
        held = ["neverobserved omega", "ordinary alpha"]
        x, state, idf = c.fit_text_features(fit)
        self.assertNotIn("neverobserved", state["word"]["vocabulary"])
        old = c.signature(state)
        before = idf["word"].copy()
        out = c.transform_texts(held, state, idf)
        self.assertEqual(c.signature(state), old)
        np.testing.assert_array_equal(idf["word"], before)
        word = c.vectorizer("word").fit(fit)
        self.assertEqual(state["word"]["vocabulary"], word.vocabulary_)
        np.testing.assert_array_equal(idf["word"], word.idf_)
        np.testing.assert_allclose(
            c.transform_texts(fit, state, idf).toarray(), x.toarray(), atol=1e-12
        )
        self.assertEqual(out.shape[1], x.shape[1])

    def test_empty_word_channel_keeps_character_evidence(self):
        x, state, idf = c.fit_text_features(["a", "b", "a"])
        self.assertEqual(state["word"]["vocabulary"], {})
        self.assertGreater(x.shape[1], 0)
        self.assertEqual(c.transform_texts(["c"], state, idf).shape, (1, x.shape[1]))
        with self.assertRaisesRegex(RuntimeError, "Completely empty"):
            c.fit_text_features(["", " "])

    def test_missing_labels_are_not_imputed(self):
        spec = dict(labels=["x", "y"], rows=[dict(id="annotated", target=[1, 0])])
        self.assertEqual(c.target_bits("hate_type", spec).tolist(), [[1, 0]])
        self.assertEqual(c.target_bits("hate_type", spec, []).shape, (0, 2))
        with self.assertRaisesRegex(RuntimeError, "Empty"):
            c.fit_linear(np.empty((0, 3)), np.empty((0, 2), dtype=int), 1.0, 42)

    def test_offline_guard(self):
        with c.offline(), self.assertRaisesRegex(RuntimeError, "Network access"):
            socket.create_connection(("example.com", 443))

    def test_resource_guard_and_lock(self):
        with patch.object(
            c, "resources", return_value=dict(available_GiB=2.0, free_disk_GiB=100.0)
        ):
            with self.assertRaisesRegex(RuntimeError, "available RAM"):
                c.resource_guard()
        with patch.object(
            c, "resources", return_value=dict(available_GiB=8.0, free_disk_GiB=8.0)
        ):
            with self.assertRaisesRegex(RuntimeError, "free disk"):
                c.resource_guard()
        with tempfile.TemporaryDirectory(
            dir=c.BASE / "qa", prefix="synthetic_lock_"
        ) as name:
            path = Path(name) / "launcher.lock"
            with c.file_lock(path):
                with self.assertRaisesRegex(RuntimeError, "already active"):
                    with c.file_lock(path, existing=True):
                        pass
            self.assertEqual(path.read_bytes(), b"0")


class Durable(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(
            dir=c.BASE / "qa", prefix="synthetic_resume_"
        )
        self.base = Path(self.temp.name)
        self.patch = patch.object(c, "BASE", self.base)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def test_resume_keeps_attempt_and_skips_completed(self):
        directory = self.base / "job"
        ident = dict(plan="synthetic", job=1)

        def fail(path):
            c.save(path / "partial.json", dict(partial=True))
            raise InterruptedError("synthetic interruption")

        with self.assertRaises(InterruptedError):
            c.task(directory, ident, fail)
        self.assertFalse((directory / "complete.json").exists())

        def succeed(path):
            c.save(path / "value.json", dict(ok=True))
            return dict(synthetic=True)

        receipt = c.task(directory, ident, succeed)
        self.assertEqual(len(list((directory / "attempts").iterdir())), 2)
        self.assertEqual(
            c.task(directory, ident, lambda _: self.fail("Repeated completed work")),
            receipt,
        )
        with self.assertRaisesRegex(RuntimeError, "different identity"):
            c.task(directory, dict(job=2), succeed)
        value = c.task_path(directory, receipt) / "value.json"
        value.write_text("changed", encoding="utf8")
        with self.assertRaisesRegex(RuntimeError, "artifact changed"):
            c.inspect_task(directory, ident)

    def test_immutable_and_path_boundary(self):
        path = self.base / "a.json"
        c.save(path, dict(a=1))
        c.save(path, dict(a=1))
        with self.assertRaisesRegex(RuntimeError, "Refusing to replace"):
            c.save(path, dict(a=2))
        with self.assertRaisesRegex(RuntimeError, "outside isolated"):
            c.save(self.base.parent / "outside.json", dict(a=1))


class Pipeline(unittest.TestCase):
    def test_full_small_pipeline_resume_and_nested_corruption(self):

        with tempfile.TemporaryDirectory(
            dir=c.BASE / "qa", prefix="synthetic_pipeline_"
        ) as name:
            base = Path(name)
            with patch.object(c, "BASE", base):
                self._exercise(base)

    def _exercise(self, base):
        rng = np.random.default_rng(8)
        matrix = rng.normal(size=(50, 1024)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
        texts = {
            f"synthetic{i}": f'fictional {"sunny" if i % 2 else "cloudy"} record code{i} group{i % 3}'
            for i in range(50)
        }
        labels = dict(
            hate=["false", "true"],
            hate_type=["a", "b"],
            target_group=["x", "y", "z"],
            severity=["low", "medium", "high"],
        )
        full = {}
        compact = dict(
            heads={},
            gate_ids=list(texts),
            folds={str(s): [i % 5 for i in range(50)] for s in c.SEEDS},
        )
        for h in c.HEADS:
            values = []
            for i, rid in enumerate(texts):
                if h != "hate" and not i % 2:
                    continue
                y = (
                    i % 2
                    if h == "hate"
                    else (
                        i % 3
                        if h == "severity"
                        else [
                            int((i + j) % (j + 2) == 0) for j in range(len(labels[h]))
                        ]
                    )
                )
                values.append(
                    dict(
                        id=rid,
                        source_id=rid,
                        target=y,
                        vector_index=i,
                        feature_row=i,
                        cluster=c.cluster(texts[rid]),
                    )
                )
            full[h] = dict(
                labels=labels[h], kind="synthetic", train=values[:-10], val=values[-10:]
            )
            compact["heads"][h] = dict(labels=labels[h], kind="synthetic", rows=values)
        c.save(base / "data/full_heads.json", full)
        c.save(base / "data/compact.json", compact)
        evaluation = [
            dict(key=f"held{i}", text=f"fictional unseen other code{i}")
            for i in range(7)
        ]
        c.save(base / "data/evaluation.json", evaluation)
        c.array(base / "data/evaluation_vectors.npy", matrix[:7])

        def fake_dense(rows):
            return np.asarray(
                matrix[[r["vector_index"] for r in rows]], dtype=np.float64
            )

        fake_resource = dict(
            available_GiB=8.0,
            free_disk_GiB=20.0,
            process_peak_working_set_GiB=0.0,
            synthetic=True,
        )
        with patch.object(run, "load_texts", return_value=texts), patch.object(
            run, "dense_rows", side_effect=fake_dense
        ), patch.object(c, "resources", return_value=fake_resource), patch.object(
            c, "current_cuda_guard"
        ), contextlib.redirect_stdout(
            io.StringIO()
        ):
            for stage in ("full_bge", "full_tfidf", "compact_tfidf"):
                for head in c.HEADS:
                    run.worker(stage, head, 42, "fixture")
                    result = run.verified_group(stage, head, 42, "fixture")
                    self.assertIsNotNone(result)
                    with patch.object(
                        c, "fit_linear", side_effect=AssertionError("Refit on resume")
                    ):
                        run.worker(stage, head, 42, "fixture")
                    if stage != "compact_tfidf":
                        matched = (
                            c.ROOT / result["variants"]["matched_gradient"]["model"]
                        )
                        allfit = c.ROOT / result["variants"]["all_permitted"]["model"]
                        self.assertEqual(
                            c.read(matched.parent.parent.parent / "complete.json")[
                                "detail"
                            ]["fit_rows"],
                            len(full[head]["train"]),
                        )
                        self.assertEqual(
                            c.read(allfit.parent.parent.parent / "complete.json")[
                                "detail"
                            ]["fit_rows"],
                            len(full[head]["train"] + full[head]["val"]),
                        )
            with patch.object(c, "SEEDS", (42,)):
                run.assemble("fixture")
            for receipt_path in (base / "pipelines").glob("*/complete.json"):
                rec = c.read(receipt_path)
                pp = c.task_path(receipt_path.parent, rec)
                contract = c.read(pp / "contract.json")
                output = c.load_arrays(pp / "decisions.npz")
                self.assertEqual(rec["detail"]["rows"], 7)
                for policy in ("default", "selected"):
                    gate = output[policy + "_hate"][:, 1].astype(bool)
                    for head in c.HEADS[1:]:
                        self.assertTrue((output[policy + "_" + head][~gate] == 0).all())
                        np.testing.assert_array_equal(
                            output[policy + "_" + head][gate],
                            output[policy + "_" + head + "_ungated"][gate],
                        )
            self.assertEqual(len(list((base / "pipelines").glob("*/complete.json"))), 5)
            d = run.group_path("compact_tfidf", "hate", 42)
            (d / "selected_oof.npy").write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "OOF probabilities changed"):
                run.verified_group("compact_tfidf", "hate", 42, "fixture")


def main():
    c.BASE.joinpath("qa").mkdir(parents=True, exist_ok=True)
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    with c.offline(), threadpool_limits(limits=1):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    files = [
        Path(__file__),
        Path(c.__file__),
        Path(run.__file__),
        Path(run.prep.__file__),
        c.PROTOCOL,
        c.ROOT / "scripts/run_cpu_baselines.ps1",
    ]
    c.save(
        c.BASE / "qa/tests.json",
        dict(
            successful=True,
            tests_run=result.testsRun,
            failures=0,
            errors=0,
            sources={c.rel(p): c.sha(p) for p in files},
            scope="Synthetic-only tests, real fitting/serialization/queue functions; no scientific results",
            CUDA_used=False,
            Qwen_calls=0,
            utc=c.utc(),
        ),
    )


if __name__ == "__main__":
    main()

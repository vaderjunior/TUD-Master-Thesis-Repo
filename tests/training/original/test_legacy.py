import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from training.original import legacy as m
from training.encoder import encode_targets
from training.encoder import macro_f1

ENCODER_ARCHIVE_AVAILABLE = (m.CORE / "encoder_plan.json").is_file() and (
    m.CORE / "encoder_data"
).is_dir()
TRANSFORMERS_AVAILABLE = importlib.util.find_spec("transformers") is not None
TOKENIZER_REFERENCE_AVAILABLE = (
    hasattr(m, "token_guard") and m.token_guard.REFERENCE.is_file()
)


class CoreEncoders(unittest.TestCase):
    def test_output_boundary(self):
        for path in (
            m.ROOT / "models/encoder/full/hate/seed0",
            m.BASE,
            m.BASE / "../plan.json",
        ):
            with self.assertRaises(RuntimeError):
                m.own(path)
        self.assertEqual(
            m.own(m.BASE / "attempts/x/result.json"), m.BASE / "attempts/x/result.json"
        )

    def test_targets_preserve_empty_and_label_order(self):
        values = [["b"], [], ["a", "b"]]
        y = encode_targets(values, "multilabel", ["b", "a"])
        np.testing.assert_array_equal(y, [[1, 0], [0, 0], [1, 1]])
        with self.assertRaises(ValueError):
            encode_targets([["unknown"]], "multilabel", ["a"])
        np.testing.assert_array_equal(
            encode_targets(["high", "low"], "ordinal", ["low", "medium", "high"]),
            [2, 0],
        )

    def test_fixed_threshold_and_ties(self):
        np.testing.assert_array_equal(
            m.decisions(np.array([[-1000.0, 0.0, 1000.0]]), "multilabel"), [[0, 1, 1]]
        )
        np.testing.assert_array_equal(
            m.decisions(np.array([[0.0, 0.0], [0.0, 1.0]]), "binary"), [0, 1]
        )

    def test_selection_metric_absent_class(self):

        self.assertAlmostEqual(
            macro_f1(np.array([0, 0]), np.array([0, 1]), False, 2), 2 / 3
        )
        self.assertAlmostEqual(
            macro_f1(np.array([[1, 0], [1, 0]]), np.array([[1, 1], [0, 0]]), True, 2),
            2 / 3,
        )

    @unittest.skipUnless(
        ENCODER_ARCHIVE_AVAILABLE, "Original head data and encoder plan are absent"
    )
    def test_all_frozen_head_targets(self):
        ep = m.read(m.CORE / "encoder_plan.json")
        seen = set()
        for job in ep["train_jobs"] + ep["weight_reuse"]:
            key = (job["arm"], job["head"])
            if key in seen:
                continue
            seen.add(key)
            meta, frames, targets = m.data_for(job)
            for split in ("train", "val"):
                self.assertEqual(len(targets[split]), len(frames[split]))
                self.assertTrue(np.isfinite(targets[split]).all())
                if meta["kind"] == "multilabel":
                    self.assertEqual(targets[split].shape[1], len(meta["labels"]))
        self.assertEqual(len(seen), 10)

    @unittest.skipUnless(
        ENCODER_ARCHIVE_AVAILABLE and TRANSFORMERS_AVAILABLE,
        "Original encoder plan or Transformers is absent",
    )
    def test_actual_training_arguments(self):
        ep = m.read(m.CORE / "encoder_plan.json")
        with tempfile.TemporaryDirectory() as tmp:
            for arm in ("full", "kb", "de"):
                job = next(j for j in ep["train_jobs"] if j["arm"] == arm)
                for smoke in (False, True):
                    a = m.arguments(job, Path(tmp) / arm, smoke)
                    self.assertEqual(a.load_best_model_at_end, arm != "kb")
                    self.assertEqual(a.num_train_epochs, 20 if arm == "kb" else 5)
                    self.assertEqual(
                        a.max_steps, (8 if arm == "de" else 2) if smoke else -1
                    )
                    self.assertTrue(a.fp16)
                    self.assertFalse(a.bf16)
                    self.assertTrue(a.save_only_model)
                    self.assertIsNone(a.save_total_limit)
                    self.assertEqual(
                        a.save_strategy.value,
                        "no" if arm == "kb" else ("steps" if smoke else "epoch"),
                    )
                    if smoke:
                        self.assertEqual(a.eval_steps, 8 if arm == "de" else 2)

    def test_no_original_trainer_or_client_execution(self):
        import ast

        tree = ast.parse(m.SELF.read_text(encoding="utf-8"))
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
        self.assertFalse(
            any(n in ("src.hsrag.classify", "src.hsrag.client") for n in names)
        )
        self.assertNotIn("training.predict", names)

    def test_completion_rejects_tamper_and_conflicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "experiment"
            attempt = base / "attempts/one"
            attempt.mkdir(parents=True)
            source = root / "runner.py"
            source.write_text("source", encoding="utf-8")
            plan = base / "execution.json"
            plan.write_text("{}", encoding="utf-8")
            with patch.multiple(m, ROOT=root, BASE=base, PLAN=plan, SELF=source):
                (attempt / "source_snapshot.py").write_bytes(source.read_bytes())
                m.write(
                    attempt / "start.json",
                    {
                        "job": "kb/hate_type/seed0",
                        "mode": "reuse",
                        "execution_sha256": m.sha(plan),
                    },
                )
                for name in (
                    "verification.json",
                    "internal_val_logits.jsonl",
                    "dev_head_logits.jsonl",
                ):
                    (attempt / name).write_text("{}\n", encoding="utf-8")
                files = {m.rel(p): m.sha(p) for p in attempt.iterdir()}
                m.write(
                    attempt / "complete.json",
                    {
                        "status": "complete",
                        "job": "kb/hate_type/seed0",
                        "mode": "reuse",
                        "execution_sha256": m.sha(plan),
                        "files": files,
                    },
                )
                self.assertEqual(m.completion(attempt)["mode"], "reuse")
                (attempt / "dev_head_logits.jsonl").write_text(
                    "changed", encoding="utf-8"
                )
                with self.assertRaisesRegex(RuntimeError, "File changed"):
                    m.completion(attempt)
                m.write(attempt / "failure.json", {"failure": True})
                with self.assertRaisesRegex(RuntimeError, "Conflicting"):
                    m.completion(attempt)

    def test_write_never_overwrites(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(m, "BASE", Path(tmp)):
                p = Path(tmp) / "record.json"
                m.write(p, {"old": True})
                with self.assertRaises(FileExistsError):
                    m.write(p, {"new": True})
                self.assertEqual(json.loads(p.read_text()), {"old": True})

    def test_loading_info_sets_are_json_serializable(self):
        report = {
            "missing_keys": {"classifier.weight", "classifier.bias"},
            "unexpected_keys": set(),
            "nested": [({"b", "a"},)],
        }
        value = m.json_ready(report)
        self.assertEqual(
            json.loads(json.dumps(value, allow_nan=False)),
            {
                "missing_keys": ["classifier.bias", "classifier.weight"],
                "unexpected_keys": [],
                "nested": [[["a", "b"]]],
            },
        )

    def test_fused_amp_overflow_accounting(self):
        self.assertTrue(m.overflow_skipped(65536, 32768, False))
        self.assertTrue(m.overflow_skipped(32768, 32768, True))
        self.assertFalse(m.overflow_skipped(32768, 32768, False))
        self.assertFalse(m.overflow_skipped(32768, 65536, False))
        with self.assertRaises(RuntimeError):
            m.overflow_skipped(32768, 0, False)


class InheritedReceipts(unittest.TestCase):
    def test_inherited_receipts_use_the_previous_encoder(self):
        from training.original import loading, validation

        receipt = dict(mode="smoke", job="kb/hate_type/seed0")
        entry = dict(
            attempt="earlier/attempt", job=receipt["job"], receipt_sha256="receipt"
        )
        plan = dict(
            core_plan_sha256=validation.CORE_SHA,
            input_hashes={},
            inherited_smokes=[entry],
        )
        with patch.multiple(
            validation,
            validate_core=lambda: {},
            read=lambda path: plan,
            verify_hashes=lambda hashes: None,
            sha=lambda path: "receipt",
        ), patch.object(
            loading, "validate", return_value={}
        ) as validate_previous, patch.object(
            loading, "completion", return_value=receipt
        ) as completion:
            self.assertEqual(validation.validate(), plan)
            validate_previous.assert_called_once_with()
            completion.assert_called_once_with(validation.ROOT / entry["attempt"])
            receipt["mode"] = "reuse"
            with self.assertRaisesRegex(RuntimeError, "Inherited smoke changed"):
                validation.validate()


if __name__ == "__main__":
    unittest.main()

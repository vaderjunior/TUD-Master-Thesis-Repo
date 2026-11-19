import json
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from training.original import run as m
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
                    self.assertEqual(a.max_steps, 8 if smoke else -1)
                    self.assertTrue(a.fp16)
                    self.assertFalse(a.bf16)
                    self.assertTrue(a.save_only_model)
                    self.assertIsNone(a.save_total_limit)
                    self.assertEqual(
                        a.save_strategy.value,
                        "no" if arm == "kb" else ("steps" if smoke else "epoch"),
                    )
                    if smoke:
                        self.assertEqual(a.eval_steps, 8)

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

    @unittest.skipUnless(
        ENCODER_ARCHIVE_AVAILABLE and TRANSFORMERS_AVAILABLE,
        "Original encoder plan or Transformers is absent",
    )
    def test_full_training_recipe_unchanged(self):
        from training.original import legacy as old

        ep = m.read(m.CORE / "encoder_plan.json")
        with tempfile.TemporaryDirectory() as tmp:
            for job in ep["train_jobs"]:
                if job["arm"] == "de":
                    continue
                path = Path(tmp) / "same"
                self.assertEqual(
                    m.arguments(job, path).to_dict(), old.arguments(job, path).to_dict()
                )

    @unittest.skipUnless(
        TOKENIZER_REFERENCE_AVAILABLE and TRANSFORMERS_AVAILABLE,
        "Original tokenizer assets or Transformers are absent",
    )
    def test_real_v4_tokenizer_defect_rejected(self):
        from transformers import RobertaTokenizerFast

        cache = m.read(m.CORE / "plan.json")["base_caches"]["roberta-base"]["snapshot"]
        bad = RobertaTokenizerFast.from_pretrained(cache, local_files_only=True)
        with self.assertRaisesRegex(RuntimeError, "50,000 BPE merges"):
            m.token_guard.check(bad)

    @unittest.skipUnless(
        TOKENIZER_REFERENCE_AVAILABLE and TRANSFORMERS_AVAILABLE,
        "Original tokenizer assets or Transformers are absent",
    )
    def test_correct_tokenizer_save_reload_and_padding(self):
        cache = m.read(m.CORE / "plan.json")["base_caches"]["roberta-base"]["snapshot"]
        tok = m.token_guard.load(cache)
        with tempfile.TemporaryDirectory() as tmp:
            tok.save_pretrained(tmp)
            loaded = m.token_guard.load(tmp)
            for padding in (False, True, "max_length"):
                a = m.token_guard.encode(tok, m.token_guard.PROBES, padding=padding)
                b = m.token_guard.encode(loaded, m.token_guard.PROBES, padding=padding)
                self.assertEqual(a["input_ids"], b["input_ids"])
                self.assertEqual(a["attention_mask"], b["attention_mask"])

    @unittest.skipUnless(
        TOKENIZER_REFERENCE_AVAILABLE, "Original tokenizer reference is absent"
    )
    def test_reference_rejects_wrong_actual_ids_and_masks(self):
        from copy import deepcopy

        texts = ["This is a simple sentence.", "short"]
        valid = m.token_guard.reference_encoding(texts, padding=True)
        for field in ("input_ids", "attention_mask"):
            bad = deepcopy(valid)
            bad[field][0][0] += 1
            with self.assertRaisesRegex(RuntimeError, field):
                m.token_guard.check_encoding(bad, texts, padding=True)

    @unittest.skipUnless(
        TOKENIZER_REFERENCE_AVAILABLE and TRANSFORMERS_AVAILABLE,
        "Original tokenizer assets or Transformers are absent",
    )
    def test_backend_rule_and_wrapper_changes_rejected(self):
        from transformers import PreTrainedTokenizerFast
        from tokenizers import Tokenizer

        cache = m.read(m.CORE / "plan.json")["base_caches"]["roberta-base"]["snapshot"]
        tok = m.token_guard.load(cache)
        tok.padding_side = "left"
        with self.assertRaisesRegex(RuntimeError, "side changed"):
            m.token_guard.check(tok)
        correct = m.token_guard.semantics(m.token_guard.reference())
        correct["pre_tokenizer"] = None
        bad = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer.from_str(json.dumps(correct))
        )
        with self.assertRaisesRegex(RuntimeError, "ByteLevel"):
            m.token_guard.check(bad)

    def test_no_german_refitting_in_repair(self):
        with self.assertRaisesRegex(RuntimeError, "English only"):
            m.load_model("unused", {"lang": "de"})

    def test_full_run_requires_completed_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(m, "BASE", Path(tmp)):
                with self.assertRaisesRegex(RuntimeError, "not complete"):
                    m.require_preflight()


if __name__ == "__main__":
    unittest.main()

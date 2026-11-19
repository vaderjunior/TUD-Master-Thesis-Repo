import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from training.frozen_features import roberta as r


class FeatureResume(unittest.TestCase):
    def setUp(self):
        base = r.BASE / "storage_fixtures"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=base)
        self.folder = Path(self.temp.name).resolve()
        self.assertTrue(self.folder.is_relative_to(base.resolve()))

    def tearDown(self):
        self.temp.cleanup()

    def payload(self, rows, plan="plan1", dim=768):
        data = np.ones((len(rows), dim), dtype=np.float32) / np.sqrt(dim)
        b = io.BytesIO()
        np.savez(
            b,
            mean=data,
            cls=data,
            plan_sha=np.array(plan),
            rows_sha=np.array(r.c.signature(rows)),
        )
        return b.getvalue()

    def test_completed_blob_is_reused_and_conflict_preserved(self):
        dest = self.folder / "data.npz"
        rows = [{"key": "a", "text": "neutral"}]
        data = self.payload(rows)
        r.blob(dest, data)
        r.blob(dest, data)
        self.assertEqual(dest.read_bytes(), data)
        with self.assertRaises(RuntimeError):
            r.blob(dest, self.payload(rows, "different"))
        self.assertEqual(dest.read_bytes(), data)
        self.assertEqual(r.check_chunk(dest, rows, "plan1")["mean"].shape, (1, 768))

    def test_wrong_plan_roster_shape_rejected(self):
        dest = self.folder / "data.npz"
        rows = [{"key": "a", "text": "neutral"}]
        r.blob(dest, self.payload(rows))
        with self.assertRaises(RuntimeError):
            r.check_chunk(dest, rows, "wrong")
        with self.assertRaises(RuntimeError):
            r.check_chunk(dest, [{"key": "a", "text": "changed"}], "plan1")
        other = self.folder / "wrong_shape.npz"
        r.blob(other, self.payload(rows, dim=3))
        with self.assertRaises(RuntimeError):
            r.check_chunk(other, rows, "plan1")

    def test_lock_releases_after_exception(self):
        lock = self.folder / "inference.lock"
        with self.assertRaises(ValueError):
            with r.item_lock(lock):
                raise ValueError("interrupted operation")
        with r.item_lock(lock):
            pass


if __name__ == "__main__":
    unittest.main()

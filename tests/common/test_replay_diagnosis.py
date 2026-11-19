import copy
import unittest
import tempfile
from pathlib import Path
from analysis.common import diagnose_replay as d


class ReplayDiagnosisTests(unittest.TestCase):

    def test_persistent_unlocked_launcher_file_is_not_an_active_job(self):
        from analysis.common.api import item_lock

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "launcher.lock"
            path.write_bytes(b"0")
            with item_lock(path):
                pass
            self.assertEqual(path.read_bytes(), b"0")

    def row(self):
        return {
            "key": "a",
            "context": {
                "examples": [
                    {"id": "x", "meta": {"_ranks": {"dense": 1}, "gate": True}}
                ]
            },
            "messages": [{"content": "x"}],
            "retrieved": {"examples": ["x", "y"]},
        }

    def test_exact(self):
        r = self.row()
        self.assertEqual(d.compare(r, copy.deepcopy(r))["kind"], "exact")

    def test_rank_only(self):
        r = self.row()
        v = copy.deepcopy(r)
        v["context"]["examples"][0]["meta"]["_ranks"]["dense"] = 2
        x = d.compare(r, v)
        self.assertEqual(x["kind"], "metadata_only")
        self.assertTrue(x["context_equal_without_ranks"])

    def test_label_change_not_dismissed_as_rank_only(self):
        r = self.row()
        v = copy.deepcopy(r)
        v["context"]["examples"][0]["meta"]["gate"] = False
        self.assertFalse(d.compare(r, v)["context_equal_without_ranks"])

    def test_order_not_membership(self):
        r = self.row()
        v = copy.deepcopy(r)
        v["retrieved"]["examples"].reverse()
        v["messages"][0]["content"] = "y x"
        self.assertEqual(d.compare(r, v)["kind"], "order")

    def test_membership(self):
        r = self.row()
        v = copy.deepcopy(r)
        v["retrieved"]["examples"] = ["x", "z"]
        v["messages"][0]["content"] = "x z"
        self.assertEqual(d.compare(r, v)["kind"], "membership")


if __name__ == "__main__":
    unittest.main()

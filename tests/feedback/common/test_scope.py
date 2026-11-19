import socket
import unittest
import uuid
from analysis.common import runtime as c
from analysis.feedback.common import scope as scope


class ScopeTests(unittest.TestCase):

    def test_transitive_source_component(self):
        u = c.Union(5)
        for a, b in [(0, 1), (2, 3), (1, 2)]:
            u.join(a, b)
        self.assertEqual(sorted(u.groups()), [[0, 1, 2, 3], [4]])

    def test_prefix_normalization_preserves_inner_punctuation(self):
        self.assertEqual(c.prefix(" :  Ａ  B "), "a b")
        self.assertNotEqual(c.prefix("a:b"), c.prefix("ab"))

    def test_equality_connects_different_roles(self):
        data = [
            {"id": "a", "text": "Hello"},
            {"id": "b", "text": " : hello"},
            {"id": "c", "text": "different"},
        ]
        self.assertEqual(scope.exact_edges(data), [(0, 1)])

    def test_sampling_keeps_whole_clusters(self):
        data = [{"key": str(i), "cluster": "a" if i < 3 else "b"} for i in range(5)]
        keep = c.hash_sample(data, 1, "fixed", "cluster")
        self.assertIn(keep, [["0", "1", "2"], ["3", "4"]])

    def test_sampling_does_not_read_labels_or_outcomes(self):
        data = [
            {"key": str(i), "cluster": str(i // 2), "gold": False, "prediction": True}
            for i in range(20)
        ]
        first = c.hash_sample(data, 5, "fixed", "cluster")
        for r in data:
            r["gold"] = True
            r["prediction"] = False
        self.assertEqual(
            first, c.hash_sample(list(reversed(data)), 5, "fixed", "cluster")
        )

    def test_reject_active_study_write(self):
        with self.assertRaises(RuntimeError):
            c.own(c.A15 / "this_must_not_exist.json")

    def test_atomic_rows_restart_and_change_detection(self):
        p = c.SHARED / "offline_check" / uuid.uuid4().hex / "records.jsonl"
        data = [{"x": 1}, {"x": 2}]
        c.jsonlines(p, data)
        before = c.sha(p)
        c.jsonlines(p, iter(data))
        self.assertEqual(c.sha(p), before)
        with self.assertRaises(RuntimeError):
            c.jsonlines(p, [{"x": 3}])
        self.assertEqual(c.sha(p), before)

    def test_offline_forbids_connections(self):
        with c.offline():
            with self.assertRaises(RuntimeError):
                socket.create_connection(("example.com", 443))


if __name__ == "__main__":
    unittest.main()

from types import SimpleNamespace
import unittest
import numpy as np
from analysis.feedback.balanced import scores as s


class BalancedScoreTests(unittest.TestCase):
    def fixture(self):
        ids = ["a", "b", "c", "d"]
        truth = np.array([False, False, True, True])
        valid = np.array([[True, True, True, True], [True, False, True, True]])
        gate = np.array([[False, True, False, True], [False, False, True, False]])
        return SimpleNamespace(
            ids=ids,
            truth=truth,
            valid=valid,
            gate=gate,
            indices=lambda keys: np.array([ids.index(i) for i in keys]),
        )

    def test_binary_confusion_and_balanced_accuracy(self):
        d = self.fixture()
        m = s.binary_run(d, d.ids, 0)
        self.assertEqual([m[k] for k in ["tp", "fp", "fn", "tn"]], [1, 1, 1, 1])
        self.assertEqual(m["macro_f1"], 0.5)
        self.assertEqual(m["balanced_accuracy"], 0.5)

    def test_null_is_excluded_not_silently_benign_and_bounds_ordered(self):
        d = self.fixture()
        m = s.binary_run(d, d.ids, 1)
        self.assertEqual(m["n"], 3)
        self.assertEqual(m["null_ids"], ["b"])
        lo = s.binary_run(d, d.ids, 1, "worst")
        hi = s.binary_run(d, d.ids, 1, "best")
        self.assertEqual(lo["fp"], 1)
        self.assertEqual(hi["fp"], 0)
        self.assertLess(lo["macro_f1"], hi["macro_f1"])

    def test_binary_one_class_balanced_accuracy_is_unavailable(self):
        d = self.fixture()
        self.assertIsNone(s.binary_run(d, ["a", "b"], 0)["balanced_accuracy"])

    def test_primary_mask_does_not_follow_supporting_arm_failure(self):
        data = SimpleNamespace(
            ids=["a", "b"],
            pos={"a": 0, "b": 1},
            valid=np.array([[True, True], [True, True], [True, False]]),
            votes=np.array([[3, 3], [3, 2], [3, 0]]),
        )
        data.indices = lambda ids: np.array([data.pos[k] for k in ids])
        self.assertEqual(s.base.Data.common(data, data.ids, [0, 1]), ["a", "b"])
        self.assertEqual(s.base.Data.common(data, data.ids, [0, 1], True), ["a"])
        self.assertEqual(s.base.Data.common(data, data.ids, [0, 1, 2]), ["a"])

    def test_stratified_cluster_draw_preserves_sources_and_whole_cluster(self):
        ids = ["a", "b", "c", "d"]
        roster = {
            "a": dict(cluster="x", source="one"),
            "b": dict(cluster="x", source="one"),
            "c": dict(cluster="y", source="two"),
            "d": dict(cluster="z", source="two"),
        }
        w, m, p = s.source_weights(ids, roster, 123, draws=20)
        self.assertTrue(np.all(w[:, 0] == 1))
        self.assertTrue(np.all(w[:, 1:].sum(axis=1) == 2))
        self.assertEqual(m[0], m[1])

    def test_fixed_representative_ignores_outcomes_and_input_order(self):
        roster = {
            "a": dict(cluster="x"),
            "b": dict(cluster="x"),
            "c": dict(cluster="y"),
        }
        left = s.representatives(SimpleNamespace(ids=["a", "b", "c"], roster=roster))
        right = s.representatives(SimpleNamespace(ids=["c", "b", "a"], roster=roster))
        self.assertEqual(set(left), set(right))
        self.assertEqual(len(left), 2)

    def test_family_interval_and_opposing_bounds(self):
        x = np.linspace(-0.2, 0.1, 10000)
        effect = s.base.effect(
            {"a": 0.2, "b": 0.3},
            {"a": x, "b": np.zeros(10000)},
            {"a": [0.1, 0.4], "b": [0.2, 0.5]},
            {"a": 1, "b": -1},
            0.975,
        )
        self.assertAlmostEqual(effect["difference"], -0.1)
        np.testing.assert_allclose(effect["full_roster_missing_bounds"], [-0.4, 0.2])
        self.assertLess(effect["family_interval"][0], effect["ci95"][0])
        self.assertGreater(effect["family_interval"][1], effect["ci95"][1])


if __name__ == "__main__":
    unittest.main()

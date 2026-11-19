import unittest
from analysis.feedback.classifiers import common as a
import numpy as np
from analysis.feedback.classifiers import kernel as k


class ReviewTests(unittest.TestCase):
    def test_raw_reference_and_stream_weights(self):
        l = np.zeros((2, 4, 4, 10), dtype=int)
        q = np.zeros((2, 4, 10, 3), dtype=int)
        l[0, 0, :, 0] = 1
        l[0, 0, :, 1] = 1
        q[0, :, 0, 0] = 1
        q[0, :, 1, :] = 1
        out = k.raw_counts(l, q)
        np.testing.assert_array_equal(out[0, :, 0], 9)
        np.testing.assert_array_equal(out[0, :, 1], 3)
        np.testing.assert_array_equal(out[0, :, 16], 3)
        np.testing.assert_array_equal(out[0, :, 17], 3)

    def test_difference_in_change_is_not_absolute_winner(self):
        raw = np.zeros(20)
        raw[:4] = [0.5, 0.48, 0.5, 0.5]
        raw[16:] = [0.2, 0.3, 0.2, 0.2]
        point = k.expand(raw)[0, 0]
        self.assertAlmostEqual(point[6], -0.12)
        self.assertAlmostEqual(point[7], 0.18)

    def test_strict_cutoff(self):
        x = np.array([0.5, 0.6, np.nextafter(0.6, 1)])
        self.assertEqual((x > 0.6).tolist(), [False, False, True])

    def test_unequal_clusters_text_weighted(self):
        values = np.array([[0], [0], [9]])
        p, d, n, s = k.bootstrap(values, ["a", "a", "b"], 9, draws=100)
        self.assertAlmostEqual(p[0], 1 / 3)
        self.assertTrue(set(d[:, 0]) <= {0.0, 1 / 3, 1.0})

    def test_equal_cluster_sensitivity(self):
        p, *_ = k.bootstrap(
            np.array([[0], [0], [9]]),
            ["a", "a", "b"],
            9,
            mode="cluster_equal",
            draws=100,
        )
        self.assertAlmostEqual(p[0], 0.5)

    def test_empty_is_undefined(self):
        with self.assertRaises(ValueError):
            k.bootstrap(np.empty((0, 1)), [], 9)

    def test_bootstrap_keeps_joint_columns(self):
        x = np.arange(6)[:, None] * np.array([[1, 2]])
        _, d, *_ = k.bootstrap(x, ["a", "a", "b", "c", "d", "d"], 13, draws=100)
        np.testing.assert_array_equal(d[:, 1], 2 * d[:, 0])

    def test_repeat_mean_not_majority(self):
        l = np.zeros((1, 4, 4, 10), int)
        q = np.zeros((1, 4, 10, 3), int)
        q[:, :, :, 0] = 1
        self.assertAlmostEqual(k.expand(k.raw_counts(l, q)[0, 0] / 9)[0, 0, 3], 1 / 3)

    def test_bound_signs(self):
        l = np.zeros(16)
        l[1] = 0.1
        b = k.bound_metrics(l, np.array([0.2, 0.3, 0, 0]), np.array([0.4, 0.8, 0, 0]))[
            0, 0
        ]
        np.testing.assert_allclose(b[5], [-0.1, 0.6])
        np.testing.assert_allclose(b[6], [-0.5, 0.2])
        np.testing.assert_allclose(b[7], [-0.7, -0.2])

    def test_vote_completion(self):
        v = np.ones((1, 4, 10, 3, 3), dtype=int)
        v[..., 1:] = -1
        lo, hi = k.majority_completion(v, np.array([False]))
        self.assertFalse(lo.any())
        self.assertTrue(hi.all())

    def test_paths_partition_first_fixes(self):
        e = np.array(
            [[1, 0, 0, 0, 0], [1, 0, 1, 0, 0], [1, 0, 0, 0, 1], [0, 1, 1, 1, 1]]
        )
        v = k.path_counts(e)
        self.assertEqual(v["first_fixes"], 3)
        self.assertEqual(
            sum(
                v[x]
                for x in [
                    "first_fixes_stay_correct",
                    "first_fixes_reverse_then_recover",
                    "first_fixes_end_wrong",
                ]
            ),
            3,
        )
        self.assertEqual(v["first_new_errors_stay_wrong"], 1)

    def test_binary_model_metrics_before_mean(self):
        _, v = k.binary_scores([0, 0, 1, 1], np.array([[0, 1], [0, 1], [0, 1], [0, 1]]))
        np.testing.assert_allclose(v["macro_f1"], [1 / 3, 1 / 3])
        np.testing.assert_allclose(v["balanced_accuracy"], [0.5, 0.5])

    def test_trajectory_uses_fresh_reference(self):
        x = np.zeros((4, 20))
        x[:, 0] = 0.2
        x[:, 1] = 0.2
        x[:, 16] = [0.1, 0.2, 0.3, 0.4]
        x[:, 17] = 0.3
        e = k.expand(x)[:, 0, 0]
        self.assertAlmostEqual(e[-1, 6] - e[0, 6], 0.3)

    def test_no_network(self):
        import socket

        with a.c.offline():
            with self.assertRaises(RuntimeError):
                socket.create_connection(("127.0.0.1", 1))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReviewTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    a.save(
        "tests.json",
        dict(
            status="passed",
            tests=result.testsRun,
            failures=0,
            errors=0,
            network_calls=0,
            fits=0,
        ),
    )

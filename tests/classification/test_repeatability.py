import socket
import unittest
import numpy as np
from analysis.classification import repeatability as a


class AgreementTests(unittest.TestCase):
    def one(self, states, correct):
        return a.pair_arrays(
            np.array(states)[None, :, None], np.array(correct)[None, :, None]
        )

    def test_three_identical_correct(self):
        agree, right, wrong = self.one([7, 7, 7], [True] * 3)
        self.assertEqual(agree.sum(), 3)
        self.assertEqual(right.sum(), 3)
        self.assertEqual(wrong.sum(), 0)

    def test_three_identical_wrong(self):
        agree, right, wrong = self.one([7, 7, 7], [False] * 3)
        self.assertEqual((agree.sum(), right.sum(), wrong.sum()), (3, 0, 3))

    def test_two_identical(self):
        agree, right, wrong = self.one([7, 7, 4], [False, False, True])
        self.assertAlmostEqual(agree.mean(), 1 / 3)
        self.assertEqual(wrong.sum(), 1)

    def test_all_different(self):
        agree, right, wrong = self.one([1, 2, 4], [False, True, False])
        self.assertEqual((agree.sum(), right.sum(), wrong.sum()), (0, 0, 0))

    def test_label_set_order(self):
        labels = ["race", "religion", "gender"]
        bits = lambda values: sum(
            2**i for i, label in enumerate(labels) if label in set(values)
        )
        self.assertEqual(bits(["gender", "race"]), bits(["race", "gender"]))
        self.assertEqual(bits(["race", "race"]), bits(["race"]))

    def test_hate_gate_distinguishes_empty_fine_answer(self):

        agree, right, wrong = self.one([128, 0, 128], [True, False, True])
        self.assertEqual((agree.sum(), right.sum(), wrong.sum()), (1, 1, 0))

    def test_severity_is_exact_class(self):
        agree, _, _ = self.one([129, 130, 132], [False, True, False])
        self.assertEqual(agree.sum(), 0)

    def test_legal_ignores_hate_gate(self):
        agree, right, _ = self.one([5, 5, 5], [True] * 3)
        self.assertEqual(agree.sum(), 3)
        self.assertEqual(right.sum(), 3)

    def test_binary_unanimity_identity(self):
        states = np.array([[[0, 0, 1, 1], [0, 1, 0, 1], [0, 1, 1, 1]]])
        agree, _, _ = a.pair_arrays(states, states == 1)
        unanimous = agree.all(axis=1).mean()
        self.assertAlmostEqual(float(agree.mean()), (1 + 2 * unanimous) / 3)

    def test_same_answer_inconsistent_gold_flag_rejected(self):
        with self.assertRaises(ValueError):
            self.one([5, 5, 5], [True, False, True])

    def test_different_answers_both_correct_rejected(self):
        with self.assertRaises(ValueError):
            self.one([1, 2, 4], [True, True, False])

    def test_common_valid_mask_does_not_make_missing_negative(self):
        valid = np.ones((3, 3, 3), dtype=bool)
        valid[1, 2, 0] = False
        valid[2, 1, 1] = False
        self.assertEqual(valid.all(axis=(0, 1)).tolist(), [False, False, True])
        self.assertEqual(valid[[0, 2]].all(axis=(0, 1)).tolist(), [True, False, True])

    def test_shared_draws_preserve_paired_zero_difference(self):
        counts = np.array([[3, 3], [1, 1], [0, 0], [3, 3]])
        draws = a.bootstrap(counts, ["a", "b", "b", "c"], 257, 55)
        self.assertTrue(np.array_equal(draws[:, 0], draws[:, 1]))

    def test_clusters_are_whole_and_text_weighted(self):
        counts = np.array([[3]] * 10 + [[0]])
        draws = a.bootstrap(counts, ["a"] * 10 + ["b"], 257, 91)
        self.assertEqual(set(np.round(draws[:, 0], 12)), {0.0, round(10 / 11, 12), 1.0})
        self.assertNotIn(0.5, draws[:, 0])

    def test_bootstrap_matches_explicit_text_expansion(self):
        counts = np.array([[3, 1], [1, 0], [0, 3], [3, 3]])
        clusters = np.array(["a", "a", "b", "c"])
        draws = a.bootstrap(counts, clusters, 257, 71)
        rng = np.random.default_rng(71)
        expected = []
        members = [np.flatnonzero(clusters == label) for label in sorted(set(clusters))]
        for selected in rng.integers(0, 3, size=(257, 3)):
            ii = np.concatenate([members[k] for k in selected])
            expected.append(counts[ii].sum(axis=0) / (3 * len(ii)))
        np.testing.assert_array_equal(draws, expected)

    def test_interval_constants_and_pair_count_decomposition(self):
        self.assertEqual(a.interval(np.zeros(50), 1 - 0.05 / 11), [0.0, 0.0])
        self.assertEqual(a.interval(np.ones(50), 0.95), [1.0, 1.0])
        agree, right, wrong = self.one([7, 7, 4], [False, False, True])
        np.testing.assert_array_equal(agree, right | wrong)

    def test_empty_subset_is_not_zero_performance(self):
        value = a.subset_result(np.array([False, False]), np.ones((3, 3, 2), bool))
        self.assertEqual(value, {"n": 0, "agreement": None, "differences": None})

    def test_network_is_blocked(self):
        with a.offline(), self.assertRaises(RuntimeError):
            socket.create_connection(("unused.invalid", 1))


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(AgreementTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    a.save(
        a.BASE / "tests.json",
        dict(
            status="passed" if result.wasSuccessful() else "failed",
            tests=result.testsRun,
            failures=len(result.failures),
            errors=len(result.errors),
            test_code=a.sha(__file__),
            analysis_code=a.sha(a.__file__),
        ),
    )
    if not result.wasSuccessful():
        raise SystemExit(1)

import io
from pathlib import Path
import unittest
import uuid
from datetime import datetime, timezone

import numpy as np
from sklearn.metrics import f1_score

from analysis.classification.supervised import type_update as s
from tests.classification.supervised import test_metrics as original_tests


class SupplementTests(unittest.TestCase):
    def test_zero_predictions_keep_all_macro_labels(self):
        y = np.array([[1, 0], [0, 0]])
        p = np.zeros((1, 2, 2), bool)
        q = s.quality(y, p, np.ones((1, 2)))
        self.assertEqual(q["macro_f1"][0], 0)
        self.assertEqual(q["exact_set"][0], 0.5)
        self.assertEqual(q["micro_precision"][0], 0)

    def test_empty_label_set_with_missed_hate_is_not_joint_correct(self):
        q = s.quality(np.zeros((2, 2)), np.zeros((1, 2, 2)), np.array([[0, 1]]))
        self.assertEqual(q["exact_set"][0], 1)
        self.assertEqual(q["hate_and_type_exact"][0], 0.5)

    def test_point_metrics_against_sklearn(self):
        y = np.array([[1, 0], [1, 1], [0, 1], [0, 0]])
        p = np.array(
            [[[1, 0], [0, 1], [1, 0], [0, 0]], [[1, 1], [1, 1], [0, 1], [1, 0]]]
        )
        q = s.quality(y, p, np.ones((2, 4)))
        for i in range(2):
            self.assertAlmostEqual(
                q["macro_f1"][i], f1_score(y, p[i], average="macro", zero_division=0)
            )
            self.assertAlmostEqual(
                q["micro_f1"][i], f1_score(y, p[i], average="micro", zero_division=0)
            )
            self.assertAlmostEqual(q["hamming_error"][i], (y != p[i]).mean())

    def test_transition_directions_and_mixed_changes(self):
        y = np.array([[1, 0], [0, 1], [1, 0]])
        old = np.array([[[0, 0], [0, 0], [1, 0]]])
        new = np.array([[[1, 1], [0, 1], [0, 0]]])
        t = s.transitions(y, old, new)
        self.assertEqual(t["label_fixes"].tolist(), [2])
        self.assertEqual(t["label_new_errors"].tolist(), [2])
        self.assertEqual(t["exact_fixes"].tolist(), [1])
        self.assertEqual(t["exact_harms"].tolist(), [1])
        self.assertEqual(t["equal_label_errors_changed"].tolist(), [1])

    def test_identical_predictions_have_no_changes(self):
        y = np.eye(2, dtype=bool)
        old = np.stack([y, y])
        t = s.transitions(y, old, old.copy())
        self.assertEqual(t["unique_changed_rows"], 0)
        self.assertEqual(t["changed_seed_histogram"], {0: 2})
        self.assertEqual(t["exact_harms"].tolist(), [0, 0])

    def test_seeds_are_not_independent_texts(self):
        y = np.array([[1, 0], [0, 1]])
        old = np.zeros((5, 2, 2))
        new = np.repeat(y[None], 5, axis=0)
        t = s.transitions(y, old, new)
        self.assertEqual(t["unique_changed_rows"], 2)
        self.assertEqual(t["changed_seed_histogram"], {5: 2})
        self.assertEqual(int(t["changed_rows"].sum()), 10)

    def test_unequal_cluster_row_mean_against_direct_sampling(self):
        values = np.array([[1, -1, 2, 3], [4, 0, -2, 1]], float)
        clusters = ["a", "a", "b", "c"]
        result = s.row_change_draws(values, clusters, draws=129, seed=9)
        unique = sorted(set(clusters))
        rng = np.random.default_rng(9)
        direct = []
        for _ in range(129):
            picks = rng.choice(len(unique), size=len(unique), replace=True)
            ix = [i for k in picks for i, c in enumerate(clusters) if c == unique[k]]
            direct.append(values[:, ix].mean(1))
        np.testing.assert_allclose(result, direct, rtol=0, atol=1e-15)

    def test_row_pairing_zero_changes(self):
        result = s.row_change_draws(np.zeros((2, 4)), ["a", "b", "c", "d"], draws=20)
        self.assertEqual(np.count_nonzero(result), 0)

    def test_invalid_shapes_and_empty_population_fail(self):
        with self.assertRaises(s.r.IntegrityError):
            s.quality(np.zeros((0, 2)), np.zeros((1, 0, 2)), np.zeros((1, 0)))
        with self.assertRaises(s.r.IntegrityError):
            s.quality(np.zeros((2, 2)), np.zeros((1, 2, 2)), np.zeros((2, 2)))
        with self.assertRaises(s.r.IntegrityError):
            s.transitions(np.zeros((2, 2)), np.zeros((1, 2, 2)), np.zeros((1, 3, 2)))
        with self.assertRaises(s.r.IntegrityError):
            s.row_change_draws(np.zeros((1, 2)), ["a"])

    def test_quality_averages_model_scores_not_ensemble(self):
        y = np.array([[1, 0], [0, 1]])
        p = np.array([[[1, 1], [1, 1]], [[0, 0], [0, 0]]])
        q = s.quality(y, p, np.ones((2, 2)))
        self.assertAlmostEqual(q["macro_f1"].mean(), 1 / 3)
        self.assertNotEqual(
            q["macro_f1"].mean(), f1_score(y, p.max(0), average="macro")
        )


def main():
    target = s.BASE / "development" / ("tests_" + uuid.uuid4().hex[:12])
    target.mkdir(parents=True)
    buffer = io.StringIO()
    suite = unittest.TestSuite(
        [
            unittest.defaultTestLoader.loadTestsFromModule(original_tests),
            unittest.defaultTestLoader.loadTestsFromTestCase(SupplementTests),
        ]
    )
    with s.r.offline():
        result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(suite)
    (target / "output.txt").write_bytes(buffer.getvalue().encode("utf8"))
    s.save(
        target / "result.json",
        dict(
            status="pass" if result.wasSuccessful() else "fail",
            tests=result.testsRun,
            errors=len(result.errors),
            failures=len(result.failures),
            utc=datetime.now(timezone.utc).isoformat(),
            sources={
                str(Path(__file__)): s.r.sha(__file__),
                str(Path(s.__file__)): s.r.sha(s.__file__),
            },
            output_sha256=s.r.sha(target / "output.txt"),
        ),
    )
    print(buffer.getvalue())
    print("TEST RECORD", target)
    if not result.wasSuccessful():
        raise SystemExit(1)


if __name__ == "__main__":
    main()

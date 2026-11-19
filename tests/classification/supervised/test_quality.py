import socket
import unittest
import numpy as np
from sklearn.metrics import f1_score, jaccard_score
from analysis.classification.supervised import quality as q
from analysis.classification.supervised import archive as r


class QualityTests(unittest.TestCase):
    def test_exact_missing_extra_mixed_are_disjoint(self):
        y = np.array([[1, 0], [1, 1], [1, 0], [1, 0]])
        p = np.array([[[1, 0], [1, 0], [1, 1], [0, 1]]])
        g = np.ones((1, 4), int)
        point, row = q.quality(y, p, g, p)
        for k, j in [
            ("exact_set", 0),
            ("only_missing", 1),
            ("only_extra", 2),
            ("mixed", 3),
        ]:
            np.testing.assert_array_equal(row[k], np.eye(4, dtype=bool)[j][None, :])
        self.assertEqual(point["extra_labels"][0], 2)
        self.assertEqual(point["missing_labels"][0], 2)

    def test_jaccard_and_sample_f1_against_sklearn(self):
        y = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
        p = np.array(
            [[[1, 1, 0], [1, 0, 0], [0, 0, 0]], [[1, 0, 0], [1, 1, 1], [1, 0, 0]]]
        )
        g = np.ones((2, 3), int)
        point, _ = q.quality(y, p, g, p)
        for j in range(2):
            self.assertAlmostEqual(
                point["sample_jaccard"][j],
                jaccard_score(y, p[j], average="samples", zero_division=1),
            )
            self.assertAlmostEqual(
                point["sample_f1"][j],
                f1_score(y, p[j], average="samples", zero_division=1),
            )
            self.assertAlmostEqual(
                point["macro_f1"][j],
                f1_score(y, p[j], average="macro", zero_division=0),
            )

    def test_empty_gold_has_exact_fine_set_but_wrong_gate(self):
        y = np.array([[0, 0], [1, 0]])
        raw = np.array([[[1, 0], [1, 0]]])
        g = np.array([[0, 1]])
        p = raw * g[:, :, None]
        point, row = q.quality(y, p, g, raw)
        self.assertEqual(point["exact_set"][0], 1)
        self.assertEqual(point["gate_aware_exact"][0], 0.5)
        self.assertEqual(row["sample_jaccard"][0, 0], 1)

    def test_missing_prediction_is_not_zero_error_or_empty_gold(self):
        y = np.array([[1, 0]])
        raw = np.array([[[1, 0]]])
        g = np.array([[0]])
        p = raw * g[:, :, None]
        point, row = q.quality(y, p, g, raw)
        self.assertEqual(point["sample_f1"][0], 0)
        self.assertEqual(point["only_missing"][0], 1)
        self.assertEqual(point["raw_exact"][0], 1)
        self.assertEqual(point["gate_aware_exact"][0], 0)

    def test_inconsistent_gate_and_prediction_rejected(self):
        with self.assertRaises(r.IntegrityError):
            q.quality(
                np.array([[1, 0]]),
                np.array([[[1, 0]]]),
                np.array([[0]]),
                np.array([[[1, 0]]]),
            )

    def test_cardinality_and_macro_label_contributions(self):
        y = np.array([[1, 0], [1, 1]])
        p = np.array([[[1, 0], [1, 0]]])
        point, _ = q.quality(y, p, np.ones((1, 2)), p)
        np.testing.assert_array_equal(point["gold_cardinality"], [0, 1, 1])
        np.testing.assert_array_equal(point["predicted_cardinality"], [[0, 2, 0]])
        self.assertAlmostEqual(point["label_f1"][0].mean(), point["macro_f1"][0])
        self.assertEqual(point["label_recall"][0, 1], 0)

    def test_severity_blocked_is_not_low_prediction(self):
        y = np.eye(3, dtype=int)
        p = np.array([[[0, 0, 0], [0, 0, 1], [1, 0, 0]]])
        g = np.array([[0, 1, 1]])
        d = q.severity_detail(y, p, g)
        np.testing.assert_array_equal(
            d["confusion"][0], [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]]
        )
        self.assertEqual(d["model_counts"]["under_two"][0], 1)
        self.assertEqual(d["model_counts"]["over_one"][0], 1)
        self.assertEqual(d["conditional_absolute_ordinal_error"][0], 1.5)

    def test_severity_empty_coverage_is_marked_undefined(self):
        d = q.severity_detail(
            np.array([[0, 1, 0]]), np.zeros((1, 1, 3), int), np.zeros((1, 1), int)
        )
        self.assertFalse(d["conditional_error_defined"][0])
        self.assertEqual(d["model_counts"]["blocked"][0], 1)

    def test_severity_non_onehot_rejected(self):
        with self.assertRaises(r.IntegrityError):
            q.severity_detail(
                np.array([[1, 1, 0]]), np.array([[[1, 0, 0]]]), np.array([[1]])
            )

    def test_joint_uses_identity_alignment_not_position(self):
        models = [dict(id="a")]
        tm = dict(ids=["t2", "t1"], base_models=models)
        sm = dict(ids=["t1", "t2"], base_models=models)
        tz = dict(
            truth=np.array([[1, 0], [0, 1]]),
            raw=np.array([[[1, 0], [0, 1]]]),
            gated=np.array([[[1, 0], [0, 1]]]),
            gate=np.ones((1, 2), int),
        )
        sz = dict(
            truth=np.array([[1, 0, 0], [0, 0, 1]]),
            raw=np.array([[[1, 0, 0], [1, 0, 0]]]),
            gated=np.array([[[1, 0, 0], [1, 0, 0]]]),
            gate=np.ones((1, 2), int),
        )
        ix, rows = q.joint_data(tm, tz, sm, sz)
        np.testing.assert_array_equal(ix, [1, 0])
        np.testing.assert_array_equal(rows["joint_exact"], [[1, 0]])
        np.testing.assert_array_equal(rows["severity_wrong_only"], [[0, 1]])

    def test_joint_does_not_accept_different_gates_or_missing_ids(self):
        tm = dict(ids=["a"], base_models=[])
        sm = dict(ids=["b"], base_models=[])
        with self.assertRaises(r.IntegrityError):
            q.joint_data(tm, {}, sm, {})
        sm["ids"] = ["a"]
        with self.assertRaises(r.IntegrityError):
            q.joint_data(tm, dict(gate=np.array([[0]])), sm, dict(gate=np.array([[1]])))

    def test_bootstrap_resamples_whole_unequal_clusters(self):
        values = np.array(
            [
                [[0.0, 1.0], [1.0, 0.0], [1.0, 2.0], [0.0, 0.0]],
                [[0.5, 2.0], [0.3, 1.0], [0.7, 1.0], [1.0, 3.0]],
            ]
        )
        draws, _ = q.row_bootstrap(values, ["a", "a", "b", "c"], draws=21, batch=4)
        rng = np.random.default_rng(20261020)
        groups = [[0, 1], [2], [3]]
        for i in range(21):
            ix = np.concatenate([groups[j] for j in rng.integers(3, size=3)])
            np.testing.assert_allclose(
                draws[i], values[:, ix].mean(1), rtol=0, atol=1e-15
            )

    def test_bootstrap_batch_size_does_not_change_pairs(self):
        values = np.arange(24).reshape(2, 4, 3) / 24
        a, ai = q.row_bootstrap(values, ["a", "a", "b", "c"], draws=21, batch=4)
        b, bi = q.row_bootstrap(values, ["a", "a", "b", "c"], draws=21, batch=9)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(ai, bi)

    def test_actual_offline_context_blocks_network(self):
        with r.offline(), self.assertRaises(r.IntegrityError):
            socket.create_connection(("example.invalid", 443))


if __name__ == "__main__":
    unittest.main()

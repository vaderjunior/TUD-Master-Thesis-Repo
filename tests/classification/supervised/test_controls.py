import unittest
import tempfile
from pathlib import Path
import numpy as np
from analysis.classification.supervised import controls as c
from analysis.classification.supervised import archive as r
from analysis.classification.supervised import validate_controls as v


class ControlTests(unittest.TestCase):
    def fixture(self):
        probs = dict(
            hate=np.array([[0.5, 0.5], [0.7, 0.3], [0.1, 0.9]]),
            hate_type=np.array([[0.5, 0.2], [0.1, 0.7], [0.8, 0.3]]),
            target_group=np.array([[0.5, 0.2], [0.1, 0.7], [0.8, 0.3]]),
            severity=np.array([[0.6, 0.4], [0.3, 0.7], [0.8, 0.2]]),
        )
        settings = {
            h: dict(selected=dict(threshold=0.2 if h == "hate" else 0.3))
            for h in r.HEADS
        }
        return r.decode(probs, settings, "default"), r.decode(
            probs, settings, "selected"
        )

    def test_crossed_gate_and_head_have_distinct_results(self):
        a, b = self.fixture()
        out = c.factorial(a, b, "hate_type")
        np.testing.assert_array_equal(out["gate_only"][0], [[1, 0], [0, 1], [1, 0]])
        np.testing.assert_array_equal(out["head_only"][0], [[0, 0], [0, 0], [1, 1]])
        np.testing.assert_array_equal(out["selected"][0], [[1, 0], [0, 1], [1, 1]])
        np.testing.assert_array_equal(a["hate_type"], [[0, 0], [0, 0], [1, 0]])

    def test_hate_strict_and_multilabel_inclusive_cutoffs(self):
        a, b = self.fixture()
        self.assertEqual(a["hate"][0, 1], 0)
        self.assertEqual(a["hate_type_ungated"][0, 0], 1)
        self.assertEqual(b["hate_type_ungated"][2, 1], 1)

    def test_severity_changes_only_gate(self):
        a, b = self.fixture()
        out = c.factorial(a, b, "severity")
        self.assertEqual(set(out), {"default", "selected", "ungated"})
        np.testing.assert_array_equal(out["ungated"][0], a["severity_ungated"])
        self.assertTrue((out["ungated"][0].sum(1) == 1).all())

    def test_severity_rejects_different_argmax(self):
        a, b = self.fixture()
        b["severity_ungated"] = b["severity_ungated"][:, ::-1]
        with self.assertRaises(r.IntegrityError):
            c.factorial(a, b, "severity")

    def test_decomposition_preserves_interaction_and_total(self):
        d = c.decomposition(0.2, 0.3, 0.25, 0.5)
        self.assertAlmostEqual(d["head"], 0.125)
        self.assertAlmostEqual(d["gate"], 0.175)
        self.assertAlmostEqual(d["interaction"], 0.15)
        self.assertAlmostEqual(d["head"] + d["gate"], d["total"])

    def test_decomposition_accepts_paired_draws(self):
        x = np.array([0.0, 0.1, 0.2])
        d = c.decomposition(x, x + 0.2, x + 0.1, x + 0.4)
        np.testing.assert_allclose(d["head"] + d["gate"], 0.4)

    def roster(self):
        return [
            dict(
                id=f"{p}/tuned/{s}", pool=p, method="tuned", variant="selected", seed=s
            )
            for p in r.POOLS
            for s in r.SEEDS
        ]

    def test_equal_pool_means_and_complete_seed_roster(self):
        rows = self.roster()
        values = np.arange(45) / 100.0
        p, b = c.aggregate(rows, values)
        self.assertAlmostEqual(b["147/tuned/selected"], 0.07)
        self.assertAlmostEqual(b["1000/tuned/selected"], 0.22)
        self.assertAlmostEqual(b["5000/tuned/selected"], 0.37)
        _, draws = c.aggregate(rows, np.stack([values, values + 0.1]))
        np.testing.assert_allclose(draws["147/tuned/selected"], [0.07, 0.17])

    def test_missing_and_duplicate_seed_rejected(self):
        rows = self.roster()
        with self.assertRaises(r.IntegrityError):
            c.aggregate(rows[:-1], np.zeros(44))
        with self.assertRaises(r.IntegrityError):
            c.aggregate(rows + [rows[0]], np.zeros(46))

    def test_linear_has_one_fit_per_pool(self):
        rows = [
            dict(id=p, pool=p, method="bge", variant="selected", seed=None)
            for p in r.POOLS
        ]
        _, b = c.aggregate(rows, np.arange(9) / 10.0)
        self.assertAlmostEqual(b["147/bge/selected"], 0.1)
        rows[0]["seed"] = 10
        with self.assertRaises(r.IntegrityError):
            c.aggregate(rows, np.zeros(9))

    def test_no_borrowed_family_or_missing_control(self):
        rows = self.roster()
        rows[0]["method"] = "fixed_unweighted"
        with self.assertRaises(r.IntegrityError):
            c.aggregate(rows, np.zeros(45))

    def test_bounded_review_loader_checks_shape_and_member_names(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "draws.npz"
            np.savez_compressed(path, scores=np.zeros((10000, 1)))
            self.assertEqual(v.load_bootstrap(path, 1).shape, (10000, 1))
            with self.assertRaises(r.IntegrityError):
                v.load_bootstrap(path, 2)
            np.savez_compressed(path, scores=np.zeros((10000, 1)), extra=np.zeros(1))
            with self.assertRaises(r.IntegrityError):
                v.load_bootstrap(path, 1)

    def test_bounded_review_loader_rejects_nonfinite_and_out_of_range(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "draws.npz"
            for value in (np.nan, 1.1, -0.1):
                np.savez_compressed(path, scores=np.full((10000, 1), value))
                with self.assertRaises(r.IntegrityError):
                    v.load_bootstrap(path, 1)


if __name__ == "__main__":
    unittest.main()

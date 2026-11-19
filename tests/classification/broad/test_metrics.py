from itertools import product
from pathlib import Path
import unittest
import numpy as np
from threadpoolctl import threadpool_limits
from analysis.classification.broad import metrics as m


def fixture(gold, predicted, valid=None, indices=None):
    y = np.asarray(gold, dtype=np.int64)
    p = np.asarray(predicted, dtype=np.int64)
    v = np.ones(len(y), dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    z = np.stack((y * p, (1 - y) * p, y * (1 - p)), axis=-1)
    z[~v] = 0
    return dict(
        ids=[str(i) for i in range(len(y))],
        gold=y,
        pred=p,
        valid=v,
        counts=z,
        labels=[str(i) for i in range(y.shape[1])],
        macro_indices=list(range(y.shape[1])) if indices is None else indices,
    )


def decision(hate=True, targets=None, legal=None):
    return dict(
        hate=hate,
        hate_type=[],
        target_group=targets or [],
        severity=None,
        legal=legal or [],
    )


def direct_f1(y, p, indices):
    if not indices:
        return None
    values = []
    for j in indices:
        tp = sum(bool(a[j]) and bool(b[j]) for a, b in zip(y, p))
        predicted = sum(int(b[j]) for b in p)
        actual = sum(int(a[j]) for a in y)
        values.append(2 * tp / (predicted + actual) if predicted + actual else 0.0)
    return sum(values) / len(values)


class ScientificChecks(unittest.TestCase):
    def test_macro_manual(self):
        a = fixture([[1, 0], [1, 0], [0, 1]], [[1, 0], [0, 1], [0, 1]])
        self.assertAlmostEqual(m.f1_on_mask(a, a["valid"]), 2 / 3)

    def test_absent_fixed_class_is_zero(self):
        a = fixture([[1, 0], [1, 0]], [[1, 0], [1, 0]])
        self.assertEqual(m.f1_on_mask(a, a["valid"]), 0.5)

    def test_empty_macro_is_not_zero(self):
        a = fixture([[0, 0]], [[0, 0]], indices=[])
        self.assertIsNone(m.f1_on_mask(a, a["valid"]))
        self.assertIsNone(m.scoring.original.score(a)["macro_f1_valid_only"])

    def test_missing_is_not_negative(self):
        a = fixture([[1, 0], [0, 1]], [[0, 0], [0, 1]], [False, True])
        s = m.scoring.original.score(a)
        self.assertEqual(s["n_invalid"], 1)
        self.assertEqual(s["macro_f1_valid_only"], 0.5)
        self.assertEqual(a["counts"][0].sum(), 0)

    def test_bounds_bracket_every_multilabel_completion(self):
        a = fixture(
            [[1, 0], [0, 1], [1, 1]], [[1, 1], [0, 0], [0, 0]], [True, False, False]
        )
        low, high = m.scoring.original.score(a)["invalid_inclusive_bounds"]
        for bits in product((0, 1), repeat=4):
            p = a["pred"].copy()
            p[1:] = np.array(bits).reshape(2, 2)
            score = direct_f1(a["gold"], p, [0, 1])
            self.assertLessEqual(low, score + 1e-15)
            self.assertLessEqual(score, high + 1e-15)

    def test_severity_relaxation_brackets_single_class(self):
        a = fixture(
            np.eye(3, dtype=int),
            [[1, 0, 0], [0, 0, 0], [0, 0, 0]],
            [True, False, False],
        )
        low, high = m.scoring.original.score(a)["invalid_inclusive_bounds"]
        for j, k in product(range(3), repeat=2):
            p = a["pred"].copy()
            p[1, j] = 1
            p[2, k] = 1
            score = direct_f1(a["gold"], p, [0, 1, 2])
            self.assertLessEqual(low, score + 1e-15)
            self.assertLessEqual(score, high + 1e-15)

    def test_all_invalid_no_valid_point_but_bounds(self):
        a = fixture([[1, 0], [0, 1]], [[0, 0], [0, 0]], [False, False])
        s = m.scoring.original.score(a)
        self.assertIsNone(s["macro_f1_valid_only"])
        self.assertEqual(s["invalid_inclusive_bounds"], [0.0, 1.0])

    def test_annotation_empty_differs_from_absent(self):
        rows = [
            dict(id="a", gate=True, target_groups=[]),
            dict(id="b", gate=True, target_groups=None),
        ]
        ep = m.scoring.endpoint(
            rows, "target_group", {"dimensions": {"target_group": {"labels": ["x"]}}}
        )
        self.assertEqual(ep["scorable_ids"], ["a"])

    def test_fine_gold_gate_required(self):
        for gate in (False, None):
            self.assertIsNone(
                m.scoring.original.truth(
                    dict(gate=gate, target_groups=["x"]), "target_group"
                )
            )

    def test_legal_independent_of_gate(self):
        self.assertEqual(
            m.scoring.original.truth(dict(gate=False, legal=["x"]), "legal"), ["x"]
        )

    def test_supported_labels_full_roster_only(self):
        rows = [
            dict(id=str(i), gate=True, target_groups=["x"] if i < 10 else ["y"])
            for i in range(19)
        ]
        ep = m.scoring.endpoint(
            rows,
            "target_group",
            {"dimensions": {"target_group": {"labels": ["x", "y", "z"]}}},
        )
        self.assertEqual(ep["macro_labels"], ["x"])
        self.assertEqual(ep["rare_labels"], ["y"])
        self.assertEqual(ep["absent_labels"], ["z"])

    def test_missing_completed_item_rejected(self):
        rows = [dict(id="a", gate=True)]
        ep = m.scoring.endpoint(rows, "hate", {})
        with self.assertRaises(ValueError):
            m.scoring.contributions(rows, {}, ep)

    def test_negative_gate_with_fine_prediction_rejected(self):
        rows = [dict(id="a", gate=True)]
        ep = m.scoring.endpoint(rows, "hate", {})
        with self.assertRaises(Exception):
            m.scoring.contributions(rows, {"a": decision(False, ["x"])}, ep)

    def test_predictions_unchanged(self):
        a = fixture([[1, 0], [0, 1]], [[1, 0], [1, 0]])
        copies = {k: v.copy() for k, v in a.items() if isinstance(v, np.ndarray)}
        m.capture_contrast([a] * 3, [a] * 3, ["a", "b"], draws=100)
        for key, value in copies.items():
            np.testing.assert_array_equal(value, a[key])

    def test_individual_mean_is_not_pooled_f1(self):
        a = fixture([[1], [1], [0]], [[1], [1], [0]])
        b = fixture([[1], [1], [0]], [[0], [0], [0]])
        individual = m.summarize_runs([a, b, b])["mean_macro_f1"]
        pooled = float(
            m.scoring.original.macro(
                sum(v["counts"].sum(axis=0) for v in [a, b, b]), [0]
            )
        )
        self.assertAlmostEqual(individual, 1 / 3)
        self.assertNotEqual(individual, pooled)

    def test_invalid_mask_rejected(self):
        a = fixture([[1]], [[0]], [False])
        with self.assertRaises(ValueError):
            m.f1_on_mask(a, np.array([True]))

    def test_common_valid_checks_every_repeat(self):
        y = [[1, 0], [0, 1], [1, 0]]
        a = fixture(y, y)
        b = fixture(y, y, [True, False, True])
        s, _ = m.capture_contrast([a, a, a], [a, b, a], ["a", "b", "c"], draws=100)
        self.assertEqual(s["common_valid_ids"], ["0", "2"])
        self.assertEqual(s["excluded_ids"], ["1"])

    def test_identical_conditions_are_exact_zero(self):
        a = fixture([[1, 0], [0, 1], [1, 0]], [[0, 1], [0, 1], [1, 0]])
        s, d = m.capture_contrast(
            [a] * 3, [a] * 3, ["a", "b", "c"], draws=129, family=4
        )
        self.assertEqual(s["mean_difference"], 0)
        np.testing.assert_array_equal(d, np.zeros(129))
        self.assertEqual(s["family_interval"], [0.0, 0.0])

    def test_acceleration_matches_original_every_draw(self):
        rng = np.random.default_rng(44)
        y = rng.integers(0, 2, (43, 4))
        allc = [
            fixture(y, rng.integers(0, 2, y.shape), rng.random(43) > 0.06)
            for _ in range(6)
        ]
        clusters = [str(i // 3) for i in range(43)]
        sa, da = m.capture_contrast(allc[:3], allc[3:], clusters, draws=257, family=4)
        sb, db = m.capture_contrast(
            allc[:3], allc[3:], clusters, draws=257, family=4, accelerate=False
        )
        np.testing.assert_array_equal(da, db)
        for key in sb:
            if key not in ("integer_sum_acceleration", "exact_integer_sum_blocks"):
                self.assertEqual(sa[key], sb[key])

    def test_clusters_are_resampled_whole(self):
        y = np.array([[1, 0]] * 8 + [[0, 1]] * 4)
        a = fixture(y, y)
        b = fixture(y, np.flip(y, axis=1))
        clusters = ["a"] * 8 + ["b"] * 4
        s, d = m.capture_contrast([a] * 3, [b] * 3, clusters, draws=129)
        rng = np.random.default_rng(m.SEED)
        expected = []
        for _ in range(129):
            picks = rng.integers(0, 2, 2)
            ids = np.concatenate(
                [np.flatnonzero(np.array(clusters) == ["a", "b"][i]) for i in picks]
            )
            expected.append(
                direct_f1(y[ids], a["pred"][ids], [0, 1])
                - direct_f1(y[ids], b["pred"][ids], [0, 1])
            )
        np.testing.assert_allclose(d, expected, rtol=0, atol=1e-15)
        self.assertEqual(s["n_clusters"], 2)

    def test_rng_equivalence_for_independent_replay(self):
        for count in (1, 2, 11, 570, 12041):
            a = np.random.default_rng(m.SEED)
            b = np.random.default_rng(m.SEED)
            for size in (64, 64, 7):
                np.testing.assert_array_equal(
                    a.choice(list(range(count)), (size, count), replace=True),
                    b.integers(0, count, (size, count)),
                )

    def test_family_quantiles(self):
        y = [[1, 0], [0, 1], [1, 0], [0, 1]]
        a = fixture(y, y)
        b = fixture(y, [[1, 0]] * 4)
        s, d = m.capture_contrast([a] * 3, [b] * 3, list("abcd"), draws=259, family=4)
        self.assertEqual(s["family_confidence"], 0.9875)
        np.testing.assert_array_equal(
            s["family_interval"], np.quantile(d, [0.00625, 0.99375])
        )
        self.assertLessEqual(s["family_interval"][0], s["ci95"][0])
        self.assertGreaterEqual(s["family_interval"][1], s["ci95"][1])

    def test_no_labels_no_bootstrap(self):
        a = fixture([[0, 0]], [[0, 0]], indices=[])
        s, d = m.capture_contrast([a] * 3, [a] * 3, ["a"], draws=100)
        self.assertEqual(len(d), 0)
        self.assertEqual(s["actual_draws"], 0)
        self.assertIsNone(s["mean_difference"])

    def test_no_common_no_bootstrap(self):
        a = fixture([[1, 0], [0, 1]], [[1, 0], [0, 1]], [True, False])
        b = fixture(a["gold"], a["pred"], [False, True])
        s, d = m.capture_contrast([a] * 3, [b] * 3, ["a", "b"], draws=100)
        self.assertEqual(len(d), 0)
        self.assertIsNone(s["mean_difference"])

    def test_unpaired_ids_rejected(self):
        a = fixture([[1]], [[1]])
        b = a | {"ids": ["different"]}
        with self.assertRaises(Exception):
            m.capture_contrast([a] * 3, [b] * 3, ["a"], draws=100)

    def test_failure_difference_combines_opposite_bounds(self):
        a = fixture([[1, 0], [0, 1]], [[1, 0], [0, 0]], [True, False])
        b = fixture(a["gold"], [[0, 1], [0, 1]])
        s, _ = m.capture_contrast([a] * 3, [b] * 3, ["a", "b"], draws=100)
        sa = m.scoring.original.score(a)["invalid_inclusive_bounds"]
        sb = m.scoring.original.score(b)["invalid_inclusive_bounds"]
        np.testing.assert_allclose(
            s["full_roster_failure_difference_bounds"],
            [sa[0] - sb[1], sa[1] - sb[0]],
            rtol=0,
            atol=1e-15,
        )

    def test_no_network_guard(self):
        import socket

        with self.assertRaises(Exception):
            socket.create_connection(("invalid.example", 443), timeout=0.01)


if __name__ == "__main__":
    with m.r.original.offline(), threadpool_limits(limits=2):
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ScientificChecks)
        names = [test.id() for test in suite]
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        m.require(
            result.wasSuccessful(), "Scientific test failure; no empirical scoring"
        )
        m.save(
            m.BASE / "tests.json",
            dict(
                status="passed",
                tests_run=result.testsRun,
                tests=names,
                sources=m.bindings(
                    [
                        Path(__file__),
                        Path(m.__file__),
                        Path(m.scoring.__file__),
                        Path(m.scoring.original.__file__),
                    ]
                ),
                scope="Synthetic scientific invariants, including complete frozen/accelerated draw equivalence. No empirical outcomes selected.",
            ),
        )

from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from analysis.classification.broad import errors as e


def answer(hate, types=(), targets=(), severity=None, legal=()):
    return dict(
        hate=hate,
        hate_type=list(types),
        target_group=list(targets),
        severity=severity,
        legal=list(legal),
    )


class Checks(unittest.TestCase):
    def test_confusion_order(self):
        y = [1, 0, 1, 0]
        p = [[1, 1, 0, 0]]
        np.testing.assert_array_equal(e.contributions(y, p).sum(axis=1), [[1, 1, 1, 1]])

    def test_confusion_partition(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([[0, 1, 0, 1], [1, 0, 1, 0]])
        np.testing.assert_array_equal(
            e.contributions(y, p).sum(axis=-1), np.ones((2, 4))
        )

    def test_nonbinary_rejected(self):
        with self.assertRaises(ValueError):
            e.contributions([0, 2], [[0, 1]])

    def test_wrong_axes_rejected(self):
        with self.assertRaises(ValueError):
            e.contributions([0, 1], [0, 1])

    def test_manual_rates(self):
        d = e.metrics(np.array([8, 2, 4, 6]))
        self.assertAlmostEqual(d["fpr"], 0.25)
        self.assertAlmostEqual(d["fnr"], 1 / 3)
        self.assertAlmostEqual(d["precision"], 0.8)
        self.assertAlmostEqual(d["recall"], 2 / 3)
        self.assertAlmostEqual(d["error_rate"], 0.3)
        self.assertAlmostEqual(d["macro_f1"], (0.7272727272727273 + 2 / 3) / 2)

    def test_no_positive_prediction_precision_undefined(self):
        d = e.safe(e.metrics(np.array([0, 0, 2, 4])))
        self.assertIsNone(d["precision"])
        self.assertEqual(d["recall"], 0)

    def test_no_gold_negative_fpr_undefined(self):
        self.assertIsNone(e.safe(e.metrics(np.array([3, 0, 1, 0])))["fpr"])

    def test_no_gold_positive_fnr_undefined(self):
        self.assertIsNone(e.safe(e.metrics(np.array([0, 1, 0, 4])))["fnr"])

    def test_no_valid_rows_not_zero_f1(self):
        self.assertIsNone(e.safe(e.metrics(np.zeros(4, dtype=int)))["macro_f1"])

    def test_individual_rate_mean_differs_from_pool(self):
        y = np.array([1, 0, 0, 0])
        p = np.array([[1, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]])
        value = e.summarize(y, p)["mean"]["precision"]
        pooled = e.metrics(e.contributions(y, p).sum(axis=(0, 1)))["precision"]
        self.assertAlmostEqual(value, 0.5)
        self.assertNotEqual(value, pooled)

    def test_fp_fixed_and_fn_fixed(self):
        y = np.array([0, 1])
        full = np.array([[0, 1]] * 3)
        ref = 1 - full
        t = e.transitions(y, full, ref)
        np.testing.assert_array_equal(t["stable"], [1, 1])
        self.assertEqual(t["summary"]["mean_events"]["fp_fixed"], 1)
        self.assertEqual(t["summary"]["mean_events"]["fn_fixed"], 1)

    def test_new_error_direction(self):
        y = np.array([0, 1])
        ref = np.array([[0, 1]] * 3)
        full = 1 - ref
        t = e.transitions(y, full, ref)
        np.testing.assert_array_equal(t["stable"], [2, 2])
        self.assertEqual(t["summary"]["mean_events"]["fp_introduced"], 1)
        self.assertEqual(t["summary"]["mean_events"]["fn_introduced"], 1)

    def test_persistent_and_both_correct(self):
        y = np.array([0, 1, 0, 1])
        full = np.array([[1, 0, 0, 1]] * 3)
        t = e.transitions(y, full, full)
        np.testing.assert_array_equal(t["stable"], [3, 3, 0, 0])
        self.assertEqual(t["event_numerators"].sum(), 36)

    def test_variable_is_not_stable(self):
        t = e.transitions(
            np.array([1]), np.array([[1], [1], [0]]), np.array([[0], [0], [0]])
        )
        self.assertEqual(t["stable"][0], 4)
        self.assertAlmostEqual(t["summary"]["mean_events"]["fn_fixed"], 2 / 3)

    def test_all_nine_pairs_permutation_invariant(self):
        y = np.array([0, 1, 1])
        a = np.array([[0, 1, 0], [1, 1, 0], [0, 0, 1]])
        b = 1 - a
        x = e.transitions(y, a, b)
        z = e.transitions(y, a[[2, 0, 1]], b[[1, 2, 0]])
        np.testing.assert_array_equal(x["event_numerators"], z["event_numerators"])
        self.assertEqual(x["summary"], z["summary"])

    def test_identical_mixed_repeats_have_symmetric_churn(self):
        y = np.array([0])
        p = np.array([[0], [1], [0]])
        s = e.transitions(y, p, p)["summary"]["mean_events"]
        self.assertAlmostEqual(s["fp_fixed"], 2 / 9)
        self.assertAlmostEqual(s["fp_introduced"], 2 / 9)

    def test_flow_identity_for_random_predictions(self):
        rng = np.random.default_rng(22)
        y = rng.integers(0, 2, 100)
        a = rng.integers(0, 2, (3, 100))
        b = rng.integers(0, 2, (3, 100))
        s = e.transitions(y, a, b)["summary"]["mean_events"]
        aa = e.summarize(y, a)["mean"]
        bb = e.summarize(y, b)["mean"]
        self.assertAlmostEqual(s["fp_introduced"] - s["fp_fixed"], aa["fp"] - bb["fp"])
        self.assertAlmostEqual(s["fn_introduced"] - s["fn_fixed"], aa["fn"] - bb["fn"])

    def test_case_selection_reproducible_under_order(self):
        ids = ["a", "b", "c"]
        y = [0, 0, 1]
        s = [1, 1, 3]
        a = e.choose(ids, y, s, "en_test", "rag_small")
        b = e.choose(ids[::-1], y[::-1], s[::-1], "en_test", "rag_small")
        self.assertEqual(a, b)

    def test_empty_case_slot_not_filled(self):
        cells = e.choose(["a"], [0], [1], "en_test", "rag_small")
        self.assertEqual(len(cells), 6)
        self.assertEqual(sum(c["key"] is not None for c in cells), 1)

    def test_variable_cases_not_substituted(self):
        self.assertTrue(
            all(
                c["key"] is None
                for c in e.choose(["a"], [0], [4], "de_test", "zero_shot")
            )
        )

    def test_false_alarm_field_attachment(self):
        rows = [
            dict(key="a", gate=False),
            dict(key="b", gate=False),
            dict(key="c", gate=True),
        ]
        out = {
            "a": answer(True, ["x"], ["y"], "high"),
            "b": answer(False),
            "c": answer(True, ["x"]),
        }
        s = e.propagation(rows, out)
        self.assertEqual(s["valid_negative"], 2)
        self.assertEqual(s["false_positive"], 1)
        self.assertEqual(s["fp_all_fine"], 1)

    def test_null_not_false_positive(self):
        s = e.propagation([dict(key="a", gate=False)], {"a": None})
        self.assertEqual(s["invalid"], 1)
        self.assertEqual(s["false_positive"], 0)
        self.assertEqual(s["valid_negative"], 0)

    def test_known_empty_block_separate(self):
        rows = [
            dict(key="a", gate=True, target_groups=[]),
            dict(key="b", gate=True, target_groups=["x", "y"]),
        ]
        s = e.propagation(
            rows, {"a": answer(False), "b": answer(False)}, "target_group"
        )
        self.assertEqual(s["gate_false"], 2)
        self.assertEqual(s["positive_gold_gated_off"], 1)
        self.assertEqual(s["positive_gold_bits_gated_off"], 2)
        self.assertEqual(s["empty_gold_gate_false"], 1)

    def test_severity_gate_block(self):
        s = e.propagation(
            [dict(key="a", gate=True, severity="low")], {"a": answer(False)}, "severity"
        )
        self.assertEqual(s["positive_gold_bits_gated_off"], 1)

    def test_legal_presence_is_not_fine_attachment(self):
        s = e.propagation([dict(key="a", gate=False)], {"a": answer(True, legal=["x"])})
        self.assertEqual(s["fp_any_fine"], 0)
        self.assertEqual(s["fp_legal_present"], 1)

    def test_cluster_bootstrap_matches_direct_draws(self):
        y = np.array([0, 1, 0, 1])
        p = np.array([[0, 1, 0, 1]] * 3 + [[1, 1, 0, 0]] * 3)
        cl = ["a", "a", "b", "b"]
        with patch.object(e, "DRAWS", 71):
            summary, draws, f = e.bootstrap(y, p, cl)
        rng = np.random.default_rng(e.SEED)
        expected = []
        for _ in range(71):
            choose = rng.integers(0, 2, 2)
            ix = np.concatenate(
                [np.flatnonzero(np.array(cl) == ["a", "b"][x]) for x in choose]
            )
            a = e.summarize(y[ix], p[:3, ix])["mean"]
            b = e.summarize(y[ix], p[3:, ix])["mean"]
            expected.append([a[key] - b[key] for key in e.METRICS])
        np.testing.assert_allclose(draws, expected, rtol=0, atol=1e-15)
        self.assertEqual(summary["n_clusters"], 2)

    def test_undefined_draws_not_dropped(self):
        y = np.array([0, 1])
        p = np.array([[0, 1]] * 6)
        with patch.object(e, "DRAWS", 71):
            summary, draws, _ = e.bootstrap(y, p, ["a", "b"])
        self.assertGreater(summary["metrics"]["fpr"]["undefined_draws"], 0)
        self.assertIsNone(summary["metrics"]["fpr"]["ci95"])
        self.assertEqual(draws.shape, (71, 3))

    def test_original_inputs_preserved(self):
        y = np.array([0, 1])
        p = np.array([[0, 1]] * 6)
        copy = p.copy()
        with patch.object(e, "DRAWS", 17):
            e.bootstrap(y, p, ["a", "a"])
        np.testing.assert_array_equal(p, copy)


if __name__ == "__main__":
    with e.r.original.offline():
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
        names = [test.id() for test in suite]
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        e.require(result.wasSuccessful(), "Scientific tests failed")
        e.save(
            e.BASE / "tests.json",
            dict(
                status="passed",
                tests_run=result.testsRun,
                tests=names,
                sources=e.bindings([Path(__file__), Path(e.__file__)]),
            ),
        )

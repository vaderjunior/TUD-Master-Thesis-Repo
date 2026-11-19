"""Tests for german guidelines legal."""

from itertools import product
import json
import time
import unittest
import numpy as np
from analysis.knowledge.german_guidelines import legal as l
from analysis.knowledge.german_guidelines import verify_legal as independent
from analysis.common import scoring as original


class LegalChecks(unittest.TestCase):

    def test_empty_gold_is_annotated_and_none_is_not(self):
        self.assertEqual(l.bits([]), [0, 0, 0])
        with self.assertRaises(ValueError):
            l.bits(None)
        with self.assertRaises(ValueError):
            l.bits(["not_a_legal_label"])
        self.assertEqual(original.truth({"gate": False, "legal": []}, "legal"), [])
        self.assertIsNone(original.truth({"gate": False, "legal": None}, "legal"))

    def test_gold_and_prediction_legal_not_gated_by_hate(self):
        labels = list(l.LABELS)
        rows = [
            {"id": "a", "gate": False, "legal": [labels[0]]},
            {"id": "b", "gate": None, "legal": []},
        ]
        outs = {
            "a": {"result": {"hate": False, "legal": [labels[0]]}, "n_valid": 3},
            "b": {"result": {"hate": True, "legal": []}, "n_valid": 3},
        }
        ep = dict(
            all_labels=labels, macro_labels=labels, dimension="legal", scorable_ids=["a", "b"]
        )
        value = original.score(original.contributions(rows, outs, ep))
        self.assertEqual(value["n_valid"], 2)
        self.assertAlmostEqual(value["macro_f1_valid_only"], 1 / 3)

    def test_missing_predictions_cannot_be_scored_as_empty(self):
        with self.assertRaises(ValueError):
            l.item_values(np.zeros((2, 3), int), np.array([[0, 0, 0], [-1, -1, -1]]))
        with self.assertRaises(ValueError):
            l.failure_bounds(
                np.zeros((2, 3), int), np.tile([[0, 0, 0], [-1, 0, -1]], (3, 1, 1)), [0, 1, 2]
            )

    def test_all_64_gold_prediction_sets_match_independent(self):
        patterns = np.asarray(list(product([0, 1], repeat=3)))
        for gold in patterns:
            y = np.tile(gold, (8, 1))
            z, f = l.totals(y, patterns)
            zz, ff = independent.reference_counts(y, patterns)
            np.testing.assert_array_equal(z, zz)
            np.testing.assert_array_equal(f, ff)
            a, p = l.metrics(z, f)
            aa, pp = independent.reference_metrics(zz, ff, [0, 1, 2])
            independent.same(l.safe(a), aa)
            independent.same(l.safe(p), pp)

    def test_wrong_label_can_have_one_extra_and_one_missed(self):
        y, p = (np.array([[1, 0, 0]]), np.array([[0, 1, 0]]))
        a, _ = l.metrics(*l.totals(y, p))
        self.assertEqual(a["extra_labels"], 1)
        self.assertEqual(a["missed_labels"], 1)
        self.assertEqual(a["mixed_error"], 1)
        self.assertEqual(a["exact_wrong"], 1)
        self.assertEqual(a["label_errors"], 2)

    def test_exact_set_and_partial_correctness_differ(self):
        y = np.array([[1, 1, 0], [0, 0, 0]])
        p = np.array([[1, 0, 0], [0, 0, 0]])
        a, _ = l.metrics(*l.totals(y, p))
        self.assertEqual(a["exact_correct"], 1)
        self.assertEqual(a["correct_labels"], 1)
        self.assertEqual(a["gold_empty"], 1)
        self.assertEqual(a["empty_gold_correct"], 1)

    def test_fixed_macro_labels_not_dropped_when_absent(self):
        y = np.array([[1, 0, 0], [0, 0, 0]])
        a, _ = l.metrics(*l.totals(y, y))
        self.assertAlmostEqual(a["macro_f1"], 1 / 3)
        a_one, _ = l.metrics(*l.totals(y, y), [0])
        self.assertEqual(a_one["macro_f1"], 1)

    def test_undefined_ratios_remain_explicit(self):
        a, p = l.metrics(*l.totals(np.zeros((3, 3), int), np.zeros((3, 3), int)))
        self.assertIsNone(l.safe(a)["micro_precision"])
        self.assertIsNone(l.safe(a)["nonempty_gold_exact_accuracy"])
        self.assertEqual(l.safe(p)["f1"], [0, 0, 0])
        self.assertIsNone(l.interval([0.0, np.nan, 0.1])["ci95"])

    def test_repeat_average_not_pooled_score(self):
        y = np.array([[1, 0, 0], [0, 0, 0]])
        pred = np.asarray([y, [[0, 0, 0], [0, 0, 0]], [[1, 0, 0], [1, 0, 0]]])
        z, f = l.totals(y, pred)
        value = l.summarize(z, f, [0, 1, 2])
        pooled, _ = l.metrics(z.sum(axis=0), f.sum(axis=0))
        self.assertNotAlmostEqual(value["mean"]["macro_f1"], pooled["macro_f1"])

    def test_exact_bounds_include_all_sets_and_preserve_nulls(self):
        y = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 1]])
        p = np.asarray(
            [
                [[-1, -1, -1], [0, 1, 0], [-1, -1, -1]],
                [[1, 1, 0], [-1, -1, -1], [0, 0, 1]],
                [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
            ]
        )
        before = p.copy()
        bounds = l.failure_bounds(y, p, [0, 1, 2])
        self.assertEqual(bounds["enumerated_sets_by_repeat"], [64, 8, 1])
        for r, row in enumerate(p):
            missing = np.flatnonzero((row < 0).all(axis=1))
            scenarios = []
            for codes in product(range(8), repeat=len(missing)):
                v = row.copy()
                for i, code in zip(missing, codes):
                    v[i] = [code >> j & 1 for j in range(3)]
                scenarios.append(independent.reference(y, v, [0, 1, 2])[0])
            for name in l.BOUND_METRICS:
                independent.same(
                    bounds["bounds"][name]["by_repeat"][r],
                    [min((v[name] for v in scenarios)), max((v[name] for v in scenarios))],
                )
        np.testing.assert_array_equal(p, before)

    def test_bootstrap_whole_clusters_and_shared_settings(self):
        y = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]])
        p = np.tile(np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0], [0, 0, 1], [1, 0, 0]]), (3, 3, 1, 1))
        clusters = ["a", "a", "b", "c", "d"]
        membership = {"a": ["x", "y"], "b": ["x"], "c": ["y"], "d": ["x"]}
        z, f, design = l.bootstrap(y, p, clusters, membership, 17, draws=129)
        rng = np.random.default_rng(17)
        for start in range(0, 129, 64):
            size = min(64, 129 - start)
            picks = np.concatenate(
                [
                    rng.choice(
                        s["cluster_indices"], (size, len(s["cluster_indices"])), replace=True
                    )
                    for s in design["strata"]
                ],
                axis=1,
            )
            for j, selected in enumerate(picks):
                rows = [k for i in selected for k in design["cluster_rows"][i]]
                zz, ff = l.totals(y[rows], p[:, :, rows])
                np.testing.assert_array_equal(z[start + j], zz)
                np.testing.assert_array_equal(f[start + j], ff)
                self.assertEqual(rows.count(0), rows.count(1))
        m, _ = l.metrics(z, f)
        np.testing.assert_array_equal(m["macro_f1"][:, 0], m["macro_f1"][:, 1])

    def test_exact_and_label_flows_and_repeat_permutation(self):
        y = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 1]])
        a = np.array([[[1, 0, 0], [1, 0, 0], [0, 1, 0]], y, [[0, 0, 0], [0, 1, 0], [0, 1, 1]]])
        b = np.array([y, [[0, 0, 0], [0, 0, 0], [0, 1, 1]], y])
        flow = l.transitions(y, a, b)
        swapped = l.transitions(y, a[[2, 0, 1]], b[[1, 2, 0]])
        self.assertEqual(flow["mean_label_events"], swapped["mean_label_events"])
        self.assertEqual(flow["mean_exact_events"], swapped["mean_exact_events"])
        self.assertAlmostEqual(sum(flow["mean_exact_events"].values()), 3)
        self.assertAlmostEqual(sum(flow["mean_label_events"].values()), 9)

    def test_full_set_confusion_matrix(self):
        y = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 1]])
        p = np.tile(np.array([[1, 0, 0], [1, 0, 0], [0, 0, 1]]), (3, 1, 1))
        result = l.set_matrix(y, p)
        self.assertEqual(result["mean"][0][1], 1)
        self.assertEqual(result["mean"][3][1], 1)
        self.assertEqual(result["mean"][4][4], 1)

    def test_family_remains_four(self):
        value = l.interval(np.arange(10001) / 10000, 4)
        self.assertEqual(value["family_confidence"], 0.9875)
        np.testing.assert_allclose(value["family_interval"], [0.00625, 0.99375])

    def test_live_transport_and_old_write_are_blocked(self):
        with self.assertRaises(RuntimeError):
            l.c.a.RequestsTransport.__call__(None, {})
        with self.assertRaises(ValueError):
            l.save(l.reader.LIVE / "forbidden.json", {})


def main():
    with l.offline():
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(LegalChecks)
        )
        dest = l.BASE / "tests" / f"tests_{time.time_ns()}" / "report.json"
        l.save(
            dest,
            {
                "passed": result.wasSuccessful(),
                "tests": result.testsRun,
                "failures": [str(x) for x, _ in result.failures],
                "errors": [str(x) for x, _ in result.errors],
                "sources": l.c.bindings(l.own_sources()),
                "actual_Qwen_calls": 0,
                "empirical_effectiveness_computed": False,
            },
        )
        print(json.dumps({"test_report": l.c.ref(dest), "passed": result.wasSuccessful()}))
        if not result.wasSuccessful():
            raise SystemExit(1)


if __name__ == "__main__":
    main()

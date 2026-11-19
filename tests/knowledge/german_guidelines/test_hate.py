"""Tests for german guidelines hate."""

from itertools import product
import json
import time
import unittest
import numpy as np
from analysis.knowledge.german_guidelines import hate as h
from analysis.common import scoring as original


class HateChecks(unittest.TestCase):

    def test_known_confusion_and_rates(self):
        y = np.array([0, 0, 0, 1, 1])
        p = np.array([0, 0, 1, 0, 1])
        z = h.contributions(y, p).sum(axis=0)
        np.testing.assert_array_equal(z, [2, 1, 1, 1])
        m = h.metrics(z)
        self.assertAlmostEqual(m["macro_f1"], (2 / 3 + 0.5) / 2)
        self.assertAlmostEqual(m["fpr"], 1 / 3)
        self.assertAlmostEqual(m["fnr"], 0.5)
        self.assertAlmostEqual(m["precision"], 0.5)

    def test_binary_macro_matches_original_contract(self):
        for tn, fp, fn, tp in product(range(3), repeat=4):
            if tn + fp + fn + tp == 0:
                continue
            expected = original.macro(np.array([[tn, fn, fp], [tp, fp, fn]]), [0, 1])
            self.assertAlmostEqual(h.metrics([tn, fp, fn, tp])["macro_f1"], expected)

    def test_missing_is_not_a_negative_prediction(self):
        with self.assertRaises(ValueError):
            h.contributions([0, 1], [0, -1])
        with self.assertRaises(ValueError):
            h.contributions([0, 1], [0, 2])

    def test_mean_repeat_f1_is_not_pooled_f1(self):
        y = np.array([0, 1])
        p = np.array([[0, 1], [0, 0], [1, 1]])
        counts = h.contributions(y, p).sum(axis=1)
        mean = h.summarize(counts)["mean"]["macro_f1"]
        self.assertAlmostEqual(mean, 5 / 9)
        self.assertAlmostEqual(h.metrics(counts.sum(axis=0))["macro_f1"], 2 / 3)
        self.assertNotAlmostEqual(mean, h.metrics(counts.sum(axis=0))["macro_f1"])

    def test_zero_denominators_remain_explicit(self):
        value = h.safe(h.metrics([3, 0, 0, 0]))
        self.assertEqual(value["macro_f1"], 0.5)
        self.assertIsNone(value["fnr"])
        self.assertIsNone(value["precision"])
        self.assertIsNone(h.safe(h.metrics([0, 0, 0, 0]))["macro_f1"])
        self.assertEqual(h.interval(np.array([0.1, np.nan, 0.2]))["undefined_draws"], 1)
        self.assertIsNone(h.interval(np.array([0.1, np.nan, 0.2]))["ci95"])

    def test_bounds_match_exhaustive_possible_answers(self):
        y = np.array([0, 1, 0, 1, 0])
        pred = np.array([[1, -1, 0, -1, -1], [-1, 0, 1, 1, -1], [0, 1, 0, 1, 0]])
        bounds = h.failure_bounds(y, pred)
        for i, row in enumerate(pred):
            missing = np.flatnonzero(row < 0)
            scores = []
            for bits in product([0, 1], repeat=len(missing)):
                candidate = row.copy()
                candidate[missing] = bits
                scores.append(h.metrics(h.contributions(y, candidate).sum(axis=0)))
            for name in h.BOUND_METRICS:
                expected = [
                    float(min((v[name] for v in scores))),
                    float(max((v[name] for v in scores))),
                ]
                np.testing.assert_allclose(
                    bounds["bounds"][name]["by_repeat"][i], expected, rtol=0, atol=1e-15
                )
        np.testing.assert_array_equal(pred[0], [1, -1, 0, -1, -1])

    def test_transitions_partition_and_net_direction(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        left = np.array([[0, 1, 0, 1, 0, 1]] * 3)
        right = np.array([[1, 0, 0, 0, 1, 1]] * 3)
        flow = h.transition_summary(y, left, right)
        self.assertEqual(sum(flow["mean_events"].values()), len(y))
        for field in ("fp_fixed", "fp_introduced", "fn_fixed", "fn_introduced"):
            self.assertEqual(flow["mean_events"][field], 1)
        for field in ("both_correct_negative", "both_correct_positive"):
            self.assertEqual(flow["mean_events"][field], 1)

    def test_transitions_do_not_depend_on_repeat_numbering(self):
        y = np.array([0, 0, 1, 1])
        a = np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 0, 1]])
        b = np.array([[0, 0, 1, 1], [0, 1, 0, 1], [1, 1, 0, 0]])
        x = h.transition_summary(y, a, b)
        z = h.transition_summary(y, a[[2, 0, 1]], b[[1, 2, 0]])
        self.assertEqual(x["mean_events"], z["mean_events"])

    def test_source_cluster_bootstrap_never_splits_cluster(self):
        y = np.array([0, 1, 1, 0, 1, 0])
        p = np.tile(np.array([0, 0, 1, 1, 1, 0]), (3, 3, 1))
        clusters = ["a", "a", "b", "c", "d", "e"]
        membership = {"a": ["s1", "s2"], "b": ["s1"], "c": ["s2"], "d": ["s1"], "e": ["s2"]}
        counts, design = h.bootstrap(y, p, clusters, membership, seed=17, draws=129)
        self.assertEqual(design["cluster_rows"][0], [0, 1])
        strata = [x["cluster_indices"] for x in design["strata"]]
        rng = np.random.default_rng(17)
        seen = 0
        for start in range(0, 129, 64):
            size = min(64, 129 - start)
            picks = np.concatenate(
                [rng.choice(group, (size, len(group)), replace=True) for group in strata], axis=1
            )
            for j, sampled in enumerate(picks):
                rows = [i for group in sampled for i in design["cluster_rows"][group]]
                direct = h.contributions(y[rows], p[:, :, rows]).sum(axis=-2)
                np.testing.assert_array_equal(counts[start + j], direct)
                self.assertEqual(rows.count(0), rows.count(1))
                seen += 1
        self.assertEqual(seen, 129)

    def test_all_conditions_share_the_same_bootstrap_draw(self):
        y = np.array([0, 1, 0, 1])
        p = np.tile(np.array([0, 0, 1, 1]), (3, 3, 1))
        counts, _ = h.bootstrap(
            y, p, list("abcd"), {x: ["source"] for x in "abcd"}, seed=19, draws=67
        )
        f1 = h.metrics(counts)["macro_f1"].mean(axis=2)
        np.testing.assert_array_equal(f1[:, 0], f1[:, 1])
        self.assertEqual(h.interval(f1[:, 1] - f1[:, 0], 4)["family_interval"], [0.0, 0.0])

    def test_four_comparison_family_is_not_reduced_to_two(self):
        values = np.arange(10001) / 10000
        out = h.interval(values, family=4)
        self.assertEqual(out["family_confidence"], 0.9875)
        np.testing.assert_allclose(out["family_interval"], [0.00625, 0.99375])
        self.assertIsNone(h.interval(values)["family_interval"])

    def test_offline_guard_blocks_live_transport(self):
        with self.assertRaises(RuntimeError):
            h.c.a.RequestsTransport.__call__(None, {})

    def test_write_guard_protects_original_evidence(self):
        with self.assertRaises(ValueError):
            h.save(h.reader.LIVE / "not_allowed.json", {})


def main():
    with h.offline():
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(HateChecks)
        )
        dest = h.BASE / "tests" / f"tests_{time.time_ns()}" / "report.json"
        h.save(
            dest,
            {
                "passed": result.wasSuccessful(),
                "tests": result.testsRun,
                "failures": [str(x) for x, _ in result.failures],
                "errors": [str(x) for x, _ in result.errors],
                "sources": h.c.bindings(h.own_sources()),
                "actual_Qwen_calls": 0,
                "empirical_effectiveness_computed": False,
            },
        )
        print(json.dumps({"test_report": h.c.ref(dest), "passed": result.wasSuccessful()}))
        if not result.wasSuccessful():
            raise SystemExit(1)


if __name__ == "__main__":
    main()

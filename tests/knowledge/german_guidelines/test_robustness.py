"""Tests for german guidelines robustness."""

import json
import socket
import time
import unittest
import numpy as np
from analysis.knowledge.german_guidelines import robustness as r
from analysis.knowledge.german_guidelines import verify_robustness as check


class Checks(unittest.TestCase):

    def test_exclusion_retains_missing_mask(self):
        self.assertEqual(
            r.exclude_mask(["a", "b", "c"], [True, False, True], ["a"]).tolist(),
            [False, False, True],
        )

    def test_unknown_exclusion_rejected(self):
        with self.assertRaises(ValueError):
            r.exclude_mask(["a"], [True], ["other"])

    def test_exclusions_do_not_touch_inputs(self):
        original = np.array([True, True])
        out = r.exclude_mask(["a", "b"], original, ["b"])
        self.assertEqual(original.tolist(), [True, True])
        self.assertEqual(out.tolist(), [True, False])

    def test_quality_checks_all_conditions_and_repeats(self):
        nv = np.full((3, 3, 4), 3)
        rep = np.zeros_like(nv)
        ties = np.zeros_like(nv, bool)
        nv[2, 2, 0] = 2
        rep[1, 1, 1] = 1
        ties[0, 1, 2] = True
        self.assertEqual(
            r.quality_mask([True] * 4, nv, rep, ties, "three_valid_inner_votes").tolist(),
            [False, True, True, True],
        )
        self.assertEqual(
            r.quality_mask([True] * 4, nv, rep, ties, "no_repairs").tolist(),
            [False, False, True, True],
        )
        self.assertEqual(
            r.quality_mask([True] * 4, nv, rep, ties, "no_hate_ties").tolist(),
            [True, True, False, True],
        )

    def test_quality_never_restores_missing(self):
        nv = np.full((3, 3, 2), 3)
        z = np.zeros_like(nv)
        for kind in r.QUALITY:
            self.assertEqual(r.quality_mask([False, True], nv, z, z, kind).tolist(), [False, True])

    def test_missing_is_not_negative(self):
        y = np.array([0, 1])
        p = np.zeros((3, 3, 2), int)
        p[0, 0, 1] = -1
        with self.assertRaises(ValueError):
            r.score(y, p, [True, True])

    def test_legal_keeps_absent_labels_in_macro(self):
        y = np.array([[1, 0, 0]])
        p = np.broadcast_to(y, (3, 3, 1, 3)).copy()
        s = r.score(y, p, [True])
        self.assertAlmostEqual(s["mixed_2"]["mean"]["macro_f1"], 1 / 3)
        self.assertEqual(s["mixed_2"]["mean"]["exact_set_accuracy"], 1)

    def test_legal_empty_is_valid(self):
        y = np.zeros((1, 3), int)
        p = np.zeros((3, 3, 1, 3), int)
        s = r.score(y, p, [True])
        self.assertEqual(s["mixed_2"]["mean"]["exact_set_accuracy"], 1)
        self.assertEqual(s["mixed_2"]["mean"]["macro_f1"], 0)

    def test_repeat_scores_are_not_pooled(self):
        y = np.array([0, 0, 0, 1])
        p = np.broadcast_to(y, (3, 3, 4)).copy()
        p[1, 0] = 1
        p[1, 1] = 0
        s = r.score(y, p, [True] * 4)
        expected = np.mean([check.hc.rates(y, v)["macro_f1"] for v in p[1]])
        self.assertAlmostEqual(s["general_2"]["mean"]["macro_f1"], expected)
        self.assertNotAlmostEqual(
            expected, check.hc.rates(np.tile(y, 3), p[1].reshape(-1))["macro_f1"]
        )

    def test_leave_one_out_is_two_scores(self):
        y = np.array([0, 0, 0, 1])
        p = np.broadcast_to(y, (3, 3, 4)).copy()
        p[1, 0] = 1
        p[1, 1] = 0
        s = r.score(y, p, [True] * 4)
        v = r.repeat_summary(s, "hate")["general_2_minus_mixed_2"]["macro_f1"]
        self.assertAlmostEqual(v["leave_one_repeat_out"][0], sum(v["by_repeat"][1:]) / 2)
        self.assertEqual(np.asarray(v["all_nine_cross_repeat_differences"]).shape, (3, 3))

    def test_cluster_bootstrap_preserves_multirow_cross_source_unit(self):
        y = np.array([0, 1, 1, 0])
        p = np.broadcast_to(y, (3, 3, 4)).copy()
        p[1, 0, 1] = 0
        clusters = ["a", "a", "b", "c"]
        members = {"a": ["detox", "gahd"], "b": ["detox"], "c": ["detox"]}
        z, design = r.h.bootstrap(y, p, clusters, members, 71, draws=127)
        check.direct_draws(y, p, clusters, members, 71, z, design)
        self.assertEqual(len(design["strata"]), 2)
        self.assertEqual(design["cluster_rows"][0], [0, 1])

    def test_guards(self):
        with r.offline():
            with self.assertRaises(RuntimeError):
                socket.create_connection(("example.invalid", 443))
            with self.assertRaises(ValueError):
                r.save(r.c.BASE / "forbidden_part6.json", {})


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    )
    path = r.BASE / "tests" / f"tests_{time.time_ns()}" / "report.json"
    r.save(
        path,
        {
            "passed": result.wasSuccessful(),
            "tests": result.testsRun,
            "failures": [str(v) for v in result.failures],
            "errors": [str(v) for v in result.errors],
            "sources": r.c.bindings(r.sources()),
            "actual_Qwen_calls": 0,
        },
    )
    print(json.dumps({"passed": result.wasSuccessful(), "report": r.c.ref(path)}))
    raise SystemExit(0 if result.wasSuccessful() else 1)

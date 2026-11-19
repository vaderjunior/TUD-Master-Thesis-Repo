"""Tests for german guidelines datasets."""

import json
import time
import unittest
import numpy as np
from analysis.knowledge.german_guidelines import datasets as d
from analysis.knowledge.german_guidelines import verify_datasets as check


class Checks(unittest.TestCase):

    def test_missing_is_not_wrong(self):
        self.assertEqual(d.event(-1, 1), "left_missing")
        self.assertEqual(d.event(0, -1), "right_missing")
        self.assertEqual(d.event(-1, -1), "both_missing")
        self.assertEqual(d.individual_pair([1, -1, 1], [0, 0, 0])["stability"], "unavailable")

    def test_stable_fix_and_regression(self):
        self.assertEqual(d.individual_pair([1, 1, 1], [0, 0, 0])["stability"], "stable_fix")
        self.assertEqual(d.individual_pair([0, 0, 0], [1, 1, 1])["stability"], "stable_regression")

    def test_same_correct_count_does_not_mean_same_repeats(self):
        p = d.individual_pair([1, 0, 1], [0, 1, 1])
        self.assertEqual(p["correct_repeat_difference"], 0)
        self.assertTrue(p["ordinal_direction_reversal"])
        self.assertEqual(p["stability"], "variable")

    def test_all_nine_pairs_not_just_three(self):
        p = d.pair_summary(np.array([[1], [0], [1]]), np.array([[0], [1], [1]]))
        self.assertAlmostEqual(p["mean_all_nine_events"]["fix"], 2 / 9)
        self.assertEqual(sum(p["mean_all_nine_events"].values()), 1)
        check.same(p, check.ref_pair(np.array([[1], [0], [1]]), np.array([[0], [1], [1]])))

    def test_all_joint_paths_and_variable_missing(self):
        for bits, name in d.PATHS.items():
            self.assertEqual(
                d.joint_path(np.repeat(np.array(list(map(int, bits)))[:, None], 3, axis=1)), name
            )
        self.assertEqual(d.joint_path(np.array([[1, 0, 1], [1, 1, 1], [0, 0, 0]])), "variable")
        self.assertEqual(d.joint_path(np.array([[-1, 0, 1], [1, 1, 1], [0, 0, 0]])), "unavailable")

    def test_exact_legal_wrong_can_still_change_answer(self):
        y = np.array([[1, 0, 0]])
        p = np.zeros((3, 3, 1, 3), dtype=np.int8)
        p[:, 0, 0] = [0, 1, 0]
        p[:, 1, 0] = [0, 0, 1]
        states = d.correct(y, p)
        self.assertFalse(states.any())
        s = d.stability_summary(y, p, states)
        self.assertEqual(s["conditions"]["mixed_2"]["always_wrong_but_variable_answer"], 1)
        check.same(s, check.ref_stability(y, p, states))

    def test_partial_legal_improvement_not_exact_fix(self):
        y = np.array([[1, 0, 0]])
        p = np.zeros((3, 3, 1, 3), dtype=np.int8)
        p[0, :, 0] = [0, 1, 1]
        p[1, :, 0] = [0, 0, 1]
        p[2, :, 0] = [1, 0, 0]
        states = d.correct(y, p)
        v = d.legal_detail(y, p, states)
        self.assertEqual(
            v["pairs"]["general_2_minus_mixed_2"]["both_always_exact_wrong"]["fewer_label_errors"],
            1,
        )
        self.assertEqual(d.pair_summary(states[1], states[0])["stability_counts"]["stable_fix"], 0)

    def test_legal_empty_and_missing_differ(self):
        y = np.array([[0, 0, 0]])
        p = np.zeros((3, 3, 1, 3), dtype=np.int8)
        p[0, 0, 0] = -1
        states = d.correct(y, p)
        self.assertEqual(states[0, 0, 0], -1)
        self.assertEqual(states[0, 1, 0], 1)
        p[0, 0, 0] = [-1, 0, 0]
        with self.assertRaises(ValueError):
            d.correct(y, p)

    def test_random_summaries_match_independent_counter(self):
        rng = np.random.default_rng(12)
        for legal in [False, True]:
            y = rng.integers(0, 2, (20, 3) if legal else 20)
            p = rng.integers(0, 2, (3, 3) + y.shape)
            p[1, 0, 2] = -1
            s = d.correct(y, p)
            check.same(d.stability_summary(y, p, s), check.ref_stability(y, p, s))

    def test_fixed_binary_macro_bootstrap_formula(self):
        z = np.array([[[[2, 1, 3, 4]] * 3] * 3])
        v = check.boot_metrics(z)
        self.assertAlmostEqual(float(v["macro_f1"][0, 0, 0]), (8 / 12 + 4 / 8) / 2)
        self.assertEqual(v["fp"][0, 0, 0], 1)

    def test_case_selection_order_independent(self):
        items = []
        for key in ["b", "a", "c"]:
            items.append(
                {
                    "key": key,
                    "phase": "de_test",
                    "source": "unit",
                    "common_valid": True,
                    "gold": 0,
                    "joint_path": "stable_general_fix_retained",
                    "pairs": {"general_2_minus_mixed_2": d.individual_pair([1, 1, 1], [0, 0, 0])},
                    "legal_error_differences": {},
                }
            )
        self.assertEqual(d.choose_cases(items), d.choose_cases(items[::-1]))
        items.append({"key": "missing", "common_valid": False})
        self.assertIn("missing", d.choose_cases(items)[0])

    def test_offline_and_write_guards(self):
        with d.offline():
            import requests

            with self.assertRaises(RuntimeError):
                requests.get("https://example.invalid")
            with self.assertRaises(ValueError):
                d.save(d.c.BASE / "forbidden_part5.json", {})


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    path = d.BASE / "tests" / f"tests_{time.time_ns()}" / "report.json"
    d.save(
        path,
        {
            "passed": result.wasSuccessful(),
            "tests": result.testsRun,
            "failures": [str(v) for v in result.failures],
            "errors": [str(v) for v in result.errors],
            "sources": d.c.bindings(d.sources()),
            "actual_Qwen_calls": 0,
        },
    )
    print(json.dumps({"passed": result.wasSuccessful(), "report": d.c.ref(path)}))
    raise SystemExit(0 if result.wasSuccessful() else 1)

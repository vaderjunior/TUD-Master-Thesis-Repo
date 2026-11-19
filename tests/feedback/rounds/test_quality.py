import unittest
import numpy as np
from analysis.feedback.rounds import quality_metrics as q

k = q.k


def row(**kwargs):
    return dict(
        gold_hate=True,
        gold_hate_types=["explicit"],
        gold_target_groups=None,
        gold_severity=None,
        gold_legal=None,
        **kwargs
    )


def answer(hate=True, types=None, targets=None, severity=None):
    return dict(
        hate=hate,
        hate_type=types or [],
        target_group=targets or [],
        severity=severity,
        legal=[],
    )


class QualityBoundary(unittest.TestCase):
    def test_unknown_is_not_empty(self):
        self.assertFalse(k.eligible(dict(row(), gold_hate_types=None), "hate_type"))

    def test_known_empty_retained(self):
        self.assertTrue(k.eligible(dict(row(), gold_target_groups=[]), "target_group"))

    def test_null_has_no_denominator(self):
        self.assertEqual(k.features(row(), None, "hate_type").sum(), 0)

    def test_false_gate_is_missing_fine(self):
        f = dict(
            zip(k.names("hate_type"), k.features(row(), answer(False), "hate_type"))
        )
        self.assertEqual((f["n"], f["exact"], f["missing_labels"]), (1, 0, 1))

    def test_empty_match_needs_gate(self):
        f = dict(
            zip(
                k.names("target_group"),
                k.features(
                    dict(row(), gold_target_groups=[]), answer(False), "target_group"
                ),
            )
        )
        self.assertEqual((f["exact"], f["gated_exact"]), (1, 0))

    def test_containment_is_not_exact(self):
        f = dict(
            zip(
                k.names("hate_type"),
                k.features(row(), answer(types=["explicit", "irony"]), "hate_type"),
            )
        )
        self.assertEqual((f["contains_gold"], f["exact"], f["only_extra"]), (1, 0, 1))

    def test_duplicate_final_labels_set_semantics(self):
        self.assertTrue(
            np.array_equal(
                k.features(row(), answer(types=["explicit"]), "hate_type"),
                k.features(row(), answer(types=["explicit", "explicit"]), "hate_type"),
            )
        )

    def test_missing_severity_not_low(self):
        f = dict(
            zip(
                k.names("severity"),
                k.features(dict(row(), gold_severity="low"), answer(), "severity"),
            )
        )
        self.assertEqual(
            (f["provided"], f["exact"], f["confusion:low:none"]), (0, 0, 1)
        )

    def test_severity_direction(self):
        f = dict(
            zip(
                k.names("severity"),
                k.features(
                    dict(row(), gold_severity="high"),
                    answer(severity="low"),
                    "severity",
                ),
            )
        )
        self.assertEqual((f["rank_absolute"], f["rank_signed"], f["under"]), (2, -2, 1))

    def test_joint_checks_all_known(self):
        r = dict(row(), gold_target_groups=["gender"])
        f = k.features(r, answer(types=["explicit"], targets=["race"]), "known_answer")
        self.assertEqual(f.tolist(), [1, 0, 1])

    def test_no_gold_joint_is_gate_only(self):
        f = k.features(dict(row(), gold_hate_types=None), answer(), "known_answer")
        self.assertEqual(f.tolist(), [1, 1, 1])

    def test_reference_not_treated_as_nine_generations(self):
        values = np.zeros((4, 10, 3, 1))
        values[:, 0] = np.arange(3)[None, :, None]
        values[:, 1:] = 7
        means = q.policy_means(values)
        self.assertTrue(np.all(means[:, 0] == 1))
        self.assertTrue(np.all(means[:, 1:] == 7))

    def test_relative_round_change(self):
        values = np.zeros((4, 10, 3, 1))
        values[3] = 5
        values[3, 0] = 2
        cols = q.columns([1, 2, 3, 4])
        s = q.summarize(values, cols)
        self.assertEqual(s[list(cols).index("r4-r1_adaptive-reference"), 0], 3)

    def test_f1_average_not_pooled(self):
        cs = np.array([[1, 1, 0, 0, 0], [9, 0, 0, 9, 0]])
        scores = k.score(cs, "binary", [0, 1])[:, 0]
        pooled = k.score(cs.sum(axis=0), "binary", [0, 1])[0]
        self.assertNotEqual(scores.mean(), pooled)

    def test_fixed_macro_absent_label_remains(self):
        f = k.features(row(), answer(types=["explicit"]), "hate_type")
        s = k.score(f, "hate_type", [0, 1])
        self.assertEqual(s[0], 0.5)

    def test_undefined_empty_cohort(self):
        self.assertTrue(
            np.isnan(k.score(np.zeros(len(k.names("binary"))), "binary", [0, 1])).all()
        )

    def test_fixed_masks(self):
        m = dict(
            row(),
            masks=dict(
                all_round_common=False,
                round_common=[True] * 4,
                round_strict=[False] * 4,
            ),
        )
        self.assertEqual(q.selected([m], [0], "binary", "fixed"), ([0], []))
        self.assertEqual(q.selected([m], [0], "binary", "final"), ([0], [0]))

    def test_all_missing_exact_bounds(self):
        m = row()
        data = np.zeros((1, 4, 10, 3, len(k.names("hate_type"))), dtype=int)
        v = np.zeros((1, 4, 10, 3), dtype=bool)
        bounds = q.missing_bounds([m], data, [0], v, "hate_type", [0])
        self.assertEqual(bounds["metrics"]["exact_set"]["adaptive-reference"], [-1, 1])


if __name__ == "__main__":
    unittest.main()

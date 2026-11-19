import json
import unittest
import numpy as np
from sklearn.metrics import f1_score
from analysis.classification.supervised import sources as a
from analysis.classification.supervised import source_summary as s
from analysis.classification.supervised import archive as r


def row(i, template, cluster, ref=None, gold=True):
    return dict(
        key="en/hatecheck-" + str(i),
        gate=gold,
        functional_cluster=cluster,
        raw=json.dumps(
            dict(
                case_id=i,
                templ_id=template,
                ref_case_id=ref,
                functionality="threat_dir_h" if gold else "counter_quote_nh",
            )
        ),
    )


class SourceTests(unittest.TestCase):
    def test_missing_negative_rate_is_marked(self):
        metric, undefined = a.binary(np.array([1, 1]), np.array([[1, 0], [0, 0]]))
        self.assertTrue(undefined["fpr"])
        self.assertFalse(undefined["fnr"])
        np.testing.assert_allclose(metric["fnr"], [0.5, 1])

    def test_missing_positive_rate_is_marked(self):
        metric, undefined = a.binary(np.array([0, 0]), np.array([[1, 0]]))
        self.assertTrue(undefined["fnr"])
        self.assertFalse(undefined["fpr"])
        self.assertEqual(metric["fpr"][0], 0.5)

    def test_binary_macro_keeps_two_original_columns(self):
        y = np.array([1, 1])
        p = np.array([[1, 0]])
        metric, _ = a.binary(y, p)
        self.assertAlmostEqual(
            metric["macro_f1"][0],
            f1_score(y, p[0], labels=[0, 1], average="macro", zero_division=0),
        )

    def test_reference_preserves_gold_change(self):
        d = s.structure([row(1, 1, "a"), row(2, 2, "a", 1, False)])
        self.assertEqual(d["reference_links"][0]["parent"], 0)
        self.assertFalse(d["reference_links"][0]["same_gold"])
        self.assertEqual(d["counts"]["functional_cluster"]["crosses_gold"], 1)

    def test_reference_same_gold_is_not_a_flip(self):
        d = s.structure([row(1, 1, "a"), row(2, 2, "a", 1)])
        self.assertTrue(d["reference_links"][0]["same_gold"])

    def test_missing_reference_fails(self):
        with self.assertRaises(r.IntegrityError):
            s.structure([row(1, 1, "a", 99)])

    def test_cross_component_reference_fails(self):
        with self.assertRaises(r.IntegrityError):
            s.structure([row(1, 1, "a"), row(2, 2, "b", 1)])

    def test_duplicate_case_fails(self):
        with self.assertRaises(r.IntegrityError):
            s.structure([row(1, 1, "a"), row(1, 2, "a")])

    def test_component_and_template_counts_are_distinct(self):
        d = s.structure([row(1, 1, "a"), row(2, 1, "a"), row(3, 2, "a", 1, False)])
        self.assertEqual(d["counts"]["template"]["groups"], 2)
        self.assertEqual(d["counts"]["functional_cluster"]["groups"], 1)
        self.assertEqual(d["counts"]["functional_cluster"]["crosses_function"], 1)

    def test_expected_relation_can_hold_when_both_wrong(self):
        gold = np.array([1, 0])
        guess = np.array([0, 1])
        self.assertEqual(guess[0] == guess[1], gold[0] == gold[1])
        self.assertFalse((guess == gold).all())

    def test_all_hatecheck_functions_have_unambiguous_polarity(self):
        self.assertEqual(len(a.FUNCTIONS), 29)
        self.assertEqual(sum(x.endswith("_h") for x in a.FUNCTIONS), 18)

    def test_fixed_fine_support_stays_fixed_on_source(self):
        y = np.array([[1, 0, 0], [1, 0, 0]])
        p = np.array([[[1, 0, 1], [1, 0, 0]]])
        score, _ = a.m.point_scores(y, p, [0, 1])
        self.assertEqual(score[0], 0.5)
        exact = (p == y).all(2).mean()
        self.assertEqual(exact, 0.5)

        self.assertAlmostEqual(
            score[0], f1_score(y[:, :2], p[0, :, :2], average="macro", zero_division=0)
        )


if __name__ == "__main__":
    unittest.main()

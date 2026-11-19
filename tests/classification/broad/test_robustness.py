import unittest
from pathlib import Path
import numpy as np
from analysis.classification.broad import robustness as x


def term(y, p, valid=None, labels=None, macro=None):
    y = np.asarray(y, np.int64)
    p = np.asarray(p, np.int64)
    v = np.ones(len(y), bool) if valid is None else np.array(valid, bool)
    lab = labels or ["a", "b"]
    counts = np.stack((y * p, (1 - y) * p, y * (1 - p)), axis=-1)
    counts[~v] = 0
    return dict(
        ids=[str(i) for i in range(len(y))],
        gold=y,
        pred=p,
        counts=counts,
        valid=v,
        labels=lab,
        macro_indices=list(range(len(lab))) if macro is None else macro,
    )


def vote(h=True, types=None, target=None, severity="medium", legal=None):
    return dict(
        hate=h,
        hate_type=types or [],
        target_group=target or [],
        severity=severity if h else None,
        legal=legal or [],
        reasoning="fixture",
    )


class ScientificChecks(unittest.TestCase):
    def setUp(self):
        self.a = term(
            [[1, 0], [0, 1], [1, 0], [0, 1]], [[1, 0], [0, 1], [0, 1], [0, 1]]
        )
        self.b = term(
            [[1, 0], [0, 1], [1, 0], [0, 1]], [[0, 1], [0, 1], [0, 1], [1, 0]]
        )

    def test_01_capture_equals_frozen(self):
        actual, draws = x.capture(
            [self.a] * 3,
            [self.b] * 3,
            ["c", "d", "e", "f"],
            ["s"] * 4,
            draws=97,
            family=4,
        )
        expected = x.scoring.contrast(
            [self.a] * 3, [self.b] * 3, ["c", "d", "e", "f"], draws=97, family=4
        )
        for key in (
            "mean_difference",
            "ci95",
            "family_interval",
            "common_f1_by_setting",
        ):
            np.testing.assert_allclose(actual[key], expected[key], atol=1e-15)
        self.assertEqual(len(draws), 97)

    def test_02_strata_cross_source_cluster_stays_whole(self):
        v, _ = x.capture(
            [self.a],
            [self.b],
            ["same", "same", "c", "d"],
            ["x", "y", "x", "y"],
            stratify=True,
            draws=67,
        )
        self.assertEqual(v["cluster_strata"], {"x|y": 1, "x": 1, "y": 1})
        self.assertEqual(v["n_clusters"], 3)

    def test_03_full_membership_before_invalid_mask(self):
        a = self.a | dict(valid=np.array([False, True, True, True]))
        a["counts"] = a["counts"].copy()
        a["counts"][0] = 0
        v, _ = x.capture(
            [a],
            [self.b],
            ["same", "same", "c", "d"],
            ["x", "y", "x", "y"],
            stratify=True,
            draws=67,
        )
        self.assertIn("x|y", v["cluster_strata"])
        self.assertEqual(v["n_common_valid"], 3)

    def test_04_single_stratum_identical_draws(self):
        _, a = x.capture([self.a], [self.b], ["a", "b", "c", "d"], ["s"] * 4, draws=67)
        _, b = x.capture(
            [self.a], [self.b], ["a", "b", "c", "d"], ["s"] * 4, stratify=True, draws=67
        )
        np.testing.assert_array_equal(a, b)

    def test_05_whole_cluster_not_item(self):
        _, a = x.capture([self.a], [self.b], ["a", "a", "b", "b"], ["s"] * 4, draws=67)
        _, b = x.capture([self.a], [self.b], ["a", "b", "c", "d"], ["s"] * 4, draws=67)
        self.assertFalse(np.array_equal(a, b))

    def test_06_fixed_macro_after_removal(self):
        v = x.restrict(self.a, np.array([True, False, True, False]))
        self.assertEqual(v["macro_indices"], [0, 1])
        self.assertEqual(v["gold"][:, 1].sum(), 0)
        self.assertLess(x.scoring.original.score(v)["macro_f1_valid_only"], 1)

    def test_07_remove_same_ids_both_sides(self):
        mask = np.array([True, False, True, True])
        a = x.restrict(self.a, mask)
        b = x.restrict(self.b, mask)
        self.assertEqual(a["ids"], b["ids"])
        self.assertNotIn("1", a["ids"])
        self.assertEqual(a["labels"], self.a["labels"])

    def test_08_common_mask_all_repeats(self):
        a = self.a | dict(valid=np.array([False, True, True, True]))
        b = self.b | dict(valid=np.array([True, False, True, True]))
        v, _ = x.capture(
            [self.a, a, self.a],
            [self.b, b, self.b],
            ["a", "b", "c", "d"],
            ["s"] * 4,
            draws=32,
        )
        self.assertEqual(v["common_valid_ids"], ["2", "3"])

    def test_09_null_not_negative(self):
        v = term([[1, 0], [0, 1]], [[0, 0], [0, 1]], [False, True])
        s = x.scoring.original.score(v)
        self.assertEqual(s["n_invalid"], 1)
        self.assertEqual(sum(sum(z) for z in v["counts"][0]), 0)

    def test_10_no_macro_returns_none(self):
        a = self.a | dict(macro_indices=[])
        v, d = x.capture([a], [a], ["a", "b", "c", "d"], ["s"] * 4, draws=32)
        self.assertIsNone(v["mean_difference"])
        self.assertEqual(len(d), 0)

    def test_11_no_valid_returns_none(self):
        a = term(self.a["gold"], self.a["pred"], [False] * 4)
        v, d = x.capture([a], [self.b], ["a", "b", "c", "d"], ["s"] * 4, draws=32)
        self.assertIsNone(v["ci95"])
        self.assertEqual(len(d), 0)

    def test_12_family_is_preserved(self):
        v, _ = x.capture(
            [self.a], [self.b], ["a", "b", "c", "d"], ["s"] * 4, family=4, draws=97
        )
        self.assertEqual(v["family_confidence"], 0.9875)
        self.assertLessEqual(v["family_interval"][0], v["ci95"][0])
        self.assertGreaterEqual(v["family_interval"][1], v["ci95"][1])

    def test_13_repeat_average_not_pooled(self):
        weak = term(self.a["gold"], np.tile([1, 0], (4, 1)))
        v, _ = x.capture(
            [self.a, weak], [self.b], ["a", "b", "c", "d"], ["s"] * 4, draws=32
        )
        expected = (
            np.mean(
                [
                    x.scoring.original.score(a)["macro_f1_valid_only"]
                    for a in (self.a, weak)
                ]
            )
            - x.scoring.original.score(self.b)["macro_f1_valid_only"]
        )
        self.assertAlmostEqual(v["mean_difference"], expected)

    def test_14_severity_bounds_conservative(self):
        v = term(
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[1, 0, 0], [0, 0, 0], [0, 0, 1]],
            [True, False, True],
            labels=["low", "medium", "high"],
        )
        s = x.scoring.original.score(v)
        for k in range(3):
            p = v["pred"].copy()
            p[1, k] = 1
            t = term(v["gold"], p, labels=v["labels"])
            point = x.scoring.original.score(t)["macro_f1_valid_only"]
            self.assertLessEqual(s["invalid_inclusive_bounds"][0], point)
            self.assertGreaterEqual(s["invalid_inclusive_bounds"][1], point)

    def test_15_agreement_requires_two(self):
        self.assertIsNone(x.agree([vote(), None, None], "hate")["unanimous"])

    def test_16_null_agreement(self):
        self.assertEqual(x.agree([None] * 3, "hate")["pairs"], 0)

    def test_17_vote_pair_denominator(self):
        self.assertEqual(
            x.agree([vote(), vote(), vote(False)], "hate")["agreeing_pairs"], 1
        )

    def test_18_list_order_irrelevant(self):
        self.assertTrue(
            x.agree(
                [vote(target=["race", "religion"]), vote(target=["religion", "race"])],
                "target_group",
            )["unanimous"]
        )

    def test_19_duplicates_not_extra_agreement(self):
        self.assertTrue(
            x.agree(
                [vote(target=["race", "race"]), vote(target=["race"])], "target_group"
            )["unanimous"]
        )

    def test_20_tie_defaults_negative(self):
        self.assertFalse(
            x.reader.delivered_aggregate([vote(), vote(False), None])["hate"]
        )

    def test_21_gate_clears_fine(self):
        self.assertEqual(
            x.reader.delivered_aggregate([vote(types=["explicit"]), vote(False), None])[
                "hate_type"
            ],
            [],
        )

    def test_22_legal_independent(self):
        self.assertEqual(
            x.reader.delivered_aggregate(
                [
                    vote(False, legal=["propaganda"]),
                    vote(False, legal=["propaganda"]),
                    vote(),
                ]
            )["legal"],
            ["propaganda"],
        )

    def test_23_one_valid_retained(self):
        self.assertTrue(x.reader.delivered_aggregate([None, vote(), None])["hate"])

    def test_24_zero_valid_is_null(self):
        self.assertIsNone(x.reader.delivered_aggregate([None] * 3))

    def test_25_duplicate_occurrence_voter_retained(self):
        self.assertEqual(
            x.reader.delivered_aggregate(
                [vote(target=["race", "race"]), vote(), vote()]
            )["target_group"],
            ["race"],
        )

    def test_26_two_labels_tie_excluded(self):
        self.assertEqual(
            x.reader.delivered_aggregate([vote(target=["race"]), vote(), None])[
                "target_group"
            ],
            [],
        )

    def test_27_empty_annotation_is_scorable(self):
        self.assertEqual(
            x.scoring.original.truth(dict(gate=True, target_groups=[]), "target_group"),
            [],
        )

    def test_28_missing_annotation_excluded(self):
        self.assertIsNone(
            x.scoring.original.truth(
                dict(gate=True, target_groups=None), "target_group"
            )
        )

    def test_29_wrong_gold_gate_ineligible(self):
        self.assertIsNone(
            x.scoring.original.truth(
                dict(gate=False, target_groups=["race"]), "target_group"
            )
        )

    def test_30_legal_gold_independent(self):
        self.assertEqual(
            x.scoring.original.truth(dict(gate=False, legal=["propaganda"]), "legal"),
            ["propaganda"],
        )

    def test_31_interval_touch_zero_unresolved(self):
        self.assertEqual(x.interval_relation([0, 0.1]), "crosses_or_touches_zero")

    def test_32_unscorable_interval(self):
        self.assertEqual(x.interval_relation(None), "not_scorable")

    def test_33_mask_rejects_integer(self):
        with self.assertRaises(ValueError):
            x.restrict(self.a, np.array([1, 0, 1, 1]))

    def test_34_output_scope(self):
        with self.assertRaises(ValueError):
            x.save(x.ROOT / "should_not_exist_part7.json", {})

    def test_35_identical_predictions_zero(self):
        v, a = x.capture(
            [self.a] * 3, [self.a] * 3, ["a", "b", "c", "d"], ["s"] * 4, draws=67
        )
        self.assertEqual(v["mean_difference"], 0)
        np.testing.assert_array_equal(a, np.zeros(67))

    def test_36_source_stratification_changes_draws_not_point(self):
        a, aa = x.capture(
            [self.a], [self.b], ["a", "b", "c", "d"], ["x", "x", "y", "y"], draws=67
        )
        b, bb = x.capture(
            [self.a],
            [self.b],
            ["a", "b", "c", "d"],
            ["x", "x", "y", "y"],
            stratify=True,
            draws=67,
        )
        self.assertEqual(a["mean_difference"], b["mean_difference"])
        self.assertFalse(np.array_equal(aa, bb))


def run():
    result = unittest.TextTestRunner(verbosity=1).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(ScientificChecks)
    )
    x.require(result.wasSuccessful(), "Scientific tests failed")
    x.save(
        x.BASE / "tests.json",
        dict(
            status="passed",
            tests=result.testsRun,
            sources=x.bindings(
                [
                    Path(__file__),
                    Path(x.__file__),
                    Path(x.scoring.__file__),
                    Path(x.scoring.original.__file__),
                ]
            ),
            new_Qwen_calls=0,
        ),
    )


if __name__ == "__main__":
    with x.r.original.offline():
        run()

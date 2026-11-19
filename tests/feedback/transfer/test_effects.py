from fractions import Fraction
import unittest
import numpy as np
from analysis.feedback.transfer import effects_metrics as k


class RateTests(unittest.TestCase):
    def fixture(self):
        meta = [
            dict(
                key=str(i),
                population="en_test",
                gold_hate=False,
                source="a",
                inference_cluster="pair" if i < 2 else "single",
                stage="s",
            )
            for i in range(3)
        ]
        err = np.zeros((3, 4, 3), dtype=np.int8)
        err[0, 0] = 1
        err[2, 1:] = 1
        votes = np.full_like(err, 3)
        return meta, err, votes

    def test_reference_once_and_three_equal_pools(self):
        meta, err, votes = self.fixture()
        c = k.cohort(meta, err, votes, "en_test", False)
        self.assertEqual(c["point"]["reference_output_count"], 9)
        self.assertEqual(c["point"]["treatment_output_count"], 27)
        self.assertEqual(c["point"]["reference_rate"], float(Fraction(1, 3)))
        self.assertEqual(c["point"]["difference"], 0)

    def test_common_mask_uses_every_condition_repeat(self):
        meta, err, votes = self.fixture()
        err[1, 3, 2] = -1
        votes[1, 3, 2] = 0
        c = k.cohort(meta, err, votes, "en_test", False)
        self.assertEqual(c["common"].tolist(), [0, 2])
        omitted = k.cohort(meta, err, votes, "en_test", False, arms=(1, 2))
        self.assertEqual(omitted["common"].tolist(), [0, 1, 2])

    def test_strict_and_null_bounds(self):
        meta, err, votes = self.fixture()
        err[1, 0, 0] = -1
        votes[1, 0, 0] = 0
        votes[0, 1, 0] = 2
        c = k.cohort(meta, err, votes, "en_test", False)
        self.assertEqual(c["common"].tolist(), [0, 2])
        self.assertAlmostEqual(c["point"]["difference_missing_bounds"][0], -1 / 9)
        self.assertAlmostEqual(c["point"]["difference_missing_bounds"][1], 0)
        self.assertEqual(
            k.cohort(meta, err, votes, "en_test", False, strict=True)[
                "common"
            ].tolist(),
            [2],
        )

    def test_representative_chosen_before_validity(self):
        meta, err, votes = self.fixture()
        selected = k.eligible_indices(meta, "en_test", False, representatives=True)
        pair = next(i for i in selected if i < 2)
        err[pair, 0, 0] = -1
        votes[pair, 0, 0] = 0
        c = k.cohort(meta, err, votes, "en_test", False, representatives=True)
        self.assertEqual(c["common"].tolist(), [2])

    def test_bootstrap_weights_original_items_not_cluster_means(self):
        meta, err, votes = self.fixture()
        c = k.cohort(meta, err, votes, "en_test", False)
        out, _ = k.bootstrap(meta, err, c, 123, draws=20)
        rng = np.random.Generator(np.random.PCG64(123))
        values = []
        for _ in range(20):
            selected = rng.integers(0, 2, 2)
            expanded = [i for j in selected for i in ([0, 1] if j == 0 else [2])]
            r = err[expanded]
            values.append(
                float(
                    Fraction(
                        int(r[:, 1:].sum()) - 3 * int(r[:, 0].sum()), 9 * len(expanded)
                    )
                )
            )
        np.testing.assert_allclose(out[:, 5], values, rtol=0, atol=1e-15)
        self.assertTrue(set(out[:, 5]) <= {-0.5, 0.0, 1.0})

    def test_stratification_never_splits_mixed_cluster(self):
        meta, err, votes = self.fixture()
        meta[1]["source"] = "b"
        c = k.cohort(meta, err, votes, "en_test", False)
        _, _, _, strata = k.grouped(
            meta, err, c["common"], mode="source_stratified", eligible=c["eligible"]
        )
        self.assertEqual(strata, {("a", "b"): [0], ("a",): [1]})

    def test_rng_chunk_invariance_for_all_modes(self):
        meta, err, votes = self.fixture()
        c = k.cohort(meta, err, votes, "en_test", False)
        for mode in ("cluster", "item", "source_stratified"):
            one, _ = k.bootstrap(meta, err, c, 123, draws=29, mode=mode, block_size=1)
            many, _ = k.bootstrap(meta, err, c, 123, draws=29, mode=mode, block_size=8)
            np.testing.assert_array_equal(one, many)

    def test_empty_cohort_is_undefined(self):
        meta, err, votes = self.fixture()
        c = k.cohort(meta, err, votes, "en_test", True)
        self.assertIsNone(c["point"]["difference"])
        self.assertIsNone(c["point"]["difference_missing_bounds"])
        draws, _ = k.bootstrap(meta, err, c, 1, draws=3)
        self.assertEqual(draws.shape, (0, 6))
        self.assertIsNone(k.intervals(draws)["interval95"])

    def test_transition_counts_reconcile(self):
        meta, err, votes = self.fixture()
        c = k.cohort(meta, err, votes, "en_test", False)
        d, inputs = k.description(meta, err, votes, c, ["ref", "p1", "p2", "p3"])
        self.assertEqual(
            d["per_text_average_directions"],
            {"fewer_errors": 1, "same_average_errors": 1, "more_errors": 1},
        )
        for p in d["pools"]:
            self.assertEqual(p["matched_repeat_fixes"], 3)
            self.assertEqual(p["matched_repeat_new_errors"], 3)

    def test_exclusions_do_not_depend_on_outcomes(self):
        meta, err, votes = self.fixture()
        c = k.cohort(meta, err, votes, "en_test", False, excluded=["0"])
        self.assertEqual(c["eligible"].tolist(), [1, 2])
        self.assertEqual(c["point"]["difference"], 0.5)


if __name__ == "__main__":
    unittest.main()

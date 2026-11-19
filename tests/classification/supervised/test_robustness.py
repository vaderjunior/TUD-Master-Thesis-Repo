import unittest
import numpy as np
from analysis.classification.supervised import robustness as b
from analysis.classification.supervised import robustness_summary as a
from analysis.classification.supervised import archive as r


def fixture():
    models = []
    values = []
    for pi, pool in enumerate(r.POOLS):
        for mi, method in enumerate(b.METHODS):
            for policy in r.POLICIES:
                for seed in (r.SEEDS if method in r.FAMILIES else (None,)):
                    models.append(
                        dict(
                            id=f"{pool}/{method}/{seed}/{policy}",
                            pool=pool,
                            method=method,
                            variant=policy,
                            seed=seed,
                            budget=b.m.budget(pool),
                        )
                    )
                    values.append(
                        0.1
                        + 0.01 * pi
                        + 0.02 * mi
                        + (0 if seed is None else 0.005 * (seed - 10))
                        + (0 if policy == "default" else 0.01)
                    )
    return models, np.array(values)


class RobustnessTests(unittest.TestCase):
    def test_342_views_keep_neural_seeds_and_fold_assignments_distinct(self):
        models, values = fixture()
        self.assertEqual(len(models), 342)
        pools, budgets = b.aggregate(models, values)
        self.assertAlmostEqual(pools["uniform_compact_s42/tuned/selected"], 0.16)
        self.assertAlmostEqual(budgets["147/tuned/selected"], 0.17)

    def test_fold_mean_is_mean_of_scores_within_each_pool(self):
        models, values = fixture()
        pools, budgets = b.aggregate(models, values)
        expected = np.mean(
            [
                pools["uniform_compact_s42/" + method + "/selected"]
                for method in ("bge", "bge_f43", "bge_f44")
            ]
        )
        self.assertAlmostEqual(
            pools["uniform_compact_s42/bge_fold_mean/selected"], expected
        )
        self.assertAlmostEqual(budgets["147/bge_fold_mean/selected"], expected + 0.01)

    def test_resample_axis_preserved(self):
        models, values = fixture()
        _, bb = b.aggregate(models, np.stack([values, values * 0.5]))
        np.testing.assert_allclose(bb["147/tuned/selected"], [0.17, 0.085])

    def test_missing_neural_seed_rejected(self):
        models, values = fixture()
        with self.assertRaises(r.IntegrityError):
            b.aggregate(models[1:], values[1:])

    def test_wrong_final_seed_rejected(self):
        models, values = fixture()
        models[0]["seed"] = 99
        with self.assertRaises(r.IntegrityError):
            b.aggregate(models, values)

    def test_duplicate_baseline_is_not_extra_replication(self):
        models, values = fixture()
        j = next(i for i, x in enumerate(models) if x["method"] == "bge")
        with self.assertRaises(r.IntegrityError):
            b.aggregate(models + [models[j]], np.append(values, values[j]))

    def test_each_alternative_fold_needs_every_pool(self):
        models, values = fixture()
        j = next(i for i, x in enumerate(models) if x["method"] == "bge_f44")
        with self.assertRaises(r.IntegrityError):
            b.aggregate(models[:j] + models[j + 1 :], np.delete(values, j))

    def test_seed_combinations_do_not_change_the_overall_mean(self):
        models, values = fixture()
        result = a.stability(models, values)
        for row in result["contrasts"]:
            if row["left"] == "frozen_roberta":
                self.assertNotIn("one_seed_per_pool", row)
                continue
            self.assertEqual(len(row["one_seed_per_pool"]), 125)
            self.assertAlmostEqual(
                row["combination_statistics"]["mean"], row["difference"]
            )

    def test_seed_omission_preserves_equal_pool_weights(self):
        models, values = fixture()
        row = next(
            x
            for x in a.stability(models, values)["contrasts"]
            if x["budget"] == 147 and x["left"] == "tuned" and x["right"] == "bge"
        )
        entry = next(
            x
            for x in row["single_seed_omissions"]
            if x["pool"] == "uniform_compact_s42" and x["seed"] == 10
        )
        self.assertAlmostEqual(entry["change"], 0.0025 / 3)
        self.assertEqual(len(row["single_seed_omissions"]), 15)

    def test_matched_neural_control_seeds_cancel_shared_variation(self):
        models, values = fixture()
        row = next(
            x
            for x in a.stability(models, values)["contrasts"]
            if x["budget"] == 147 and x["right"] == "fixed_balanced"
        )
        self.assertAlmostEqual(row["matched_fit_statistics"]["sd"], 0)
        self.assertAlmostEqual(row["maximum_single_seed_influence"], 0)

    def test_descriptive_sign_count_handles_ties(self):
        value = a.stats([-0.1, 0, 0.1])
        self.assertEqual(
            [value[x] for x in ("positive", "negative", "zero")], [1, 1, 1]
        )
        self.assertIsNone(a.stats([1])["sd"])

    def test_all_pools_omitted_symmetrically(self):
        models, values = fixture()
        result = a.stability(models, values)
        for row in result["contrasts"]:
            self.assertEqual(len(row["leave_one_pool_out"]), 3)
            self.assertAlmostEqual(
                np.mean([x["difference"] for x in row["leave_one_pool_out"]]),
                row["difference"],
            )


if __name__ == "__main__":
    unittest.main()

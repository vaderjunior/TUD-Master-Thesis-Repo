import unittest
import numpy as np
from training.frozen_features import fit as m


class LinearContracts(unittest.TestCase):
    def data(self):
        rows = [
            {
                "id": str(i),
                "feature_row": i,
                "target": i % 2,
                "original_partition": "train" if i < 8 else "val",
            }
            for i in range(10)
        ]
        heads = {"hate": {"labels": ["not_hate", "hate"], "rows": rows}}
        for h in ("hate_type", "target_group", "severity"):
            heads[h] = {
                "labels": ["a", "b"],
                "rows": [
                    r
                    | {
                        "target": (
                            i % 2
                            if h == "severity"
                            else [int(i % 2 == 0), int(i % 2 != 0)]
                        )
                    }
                    for i, r in enumerate(rows)
                    if r["target"] == 1
                ],
            }
        return {"heads": heads, "folds": {"42": [i % 5 for i in range(10)]}}

    def test_complete_oof_and_train_only_partition_control(self):
        data = self.data()
        x = np.arange(30).reshape(10, 3) / 30
        cached, counts = m.make_oof(data, x, 42)
        h = m.fit_recipe(data, x, 42, cached, tune_c=False, tune_threshold=False)
        r = m.fit_recipe(data, x, 42, cached, original_fit=True, locked_selection=h)
        self.assertEqual(len(h["hate"]["training_ids"]), 10)
        self.assertEqual(len(r["hate"]["training_ids"]), 8)
        self.assertEqual(h["hate"]["selected"], r["hate"]["selected"])
        self.assertTrue(
            all(np.isfinite(v).all() for d in cached.values() for v in d.values())
        )
        self.assertGreater(m.check_selection(data, 42, h, cached), 0)

    def test_missing_head_is_untrained_without_fake_negatives(self):
        data = self.data()
        data["heads"]["hate_type"]["rows"] = []
        x = np.eye(10)
        cached, counts = m.make_oof(data, x, 42)
        heads = m.fit_recipe(data, x, 42, cached)
        self.assertTrue(heads["hate_type"]["untrained"])
        self.assertEqual(heads["hate_type"]["permitted_annotation_rows"], 0)
        self.assertEqual(heads["hate_type"]["selected"]["C"], 1.0)
        pred = m.decode(heads, m.probabilities(heads, x), ungated=True)
        self.assertTrue(all(v["hate_type"] == [] for v in pred))

    def test_constant_classes_and_sparse_folds_are_retained(self):
        data = self.data()
        data["heads"]["target_group"]["rows"] = data["heads"]["target_group"]["rows"][
            :1
        ]
        data["heads"]["target_group"]["rows"][0]["target"] = [1, 0]
        cached, counts = m.make_oof(data, np.eye(10), 42)
        self.assertGreater(counts["empty_fit_tasks"], 0)
        heads = m.fit_recipe(data, np.eye(10), 42, cached)
        self.assertTrue(heads["target_group"]["selection_fallback"])
        self.assertEqual(
            heads["target_group"]["models"], [{"constant": 1}, {"constant": 0}]
        )

    def test_gate_masks_all_fine_outputs(self):
        data = self.data()
        x = np.eye(10)
        cached, _ = m.make_oof(data, x, 42)
        heads = m.fit_recipe(data, x, 42, cached)
        p = m.probabilities(heads, x)
        p["hate"][:] = 0
        pred = m.decode(heads, p)
        self.assertTrue(
            all(
                not r["hate"]
                and r["hate_type"] == []
                and r["target_group"] == []
                and r["severity"] is None
                for r in pred
            )
        )


if __name__ == "__main__":
    unittest.main()

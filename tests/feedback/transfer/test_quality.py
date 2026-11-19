import hashlib
import unittest
import numpy as np
from analysis.feedback.transfer import quality_metrics as k


def row(**kw):
    r = dict(
        key="x",
        population="p",
        source="s",
        inference_cluster="c",
        gold_hate=True,
        gold_hate_types=["explicit"],
        gold_target_groups=None,
        gold_severity=None,
        gold_legal=None,
    )
    r.update(kw)
    return r


def output(**kw):
    p = dict(
        hate=True, hate_type=["explicit"], target_group=[], severity=None, legal=[]
    )
    p.update(kw)
    return p


class QualityBoundaries(unittest.TestCase):
    def test_f1_before_averaging(self):
        c = np.asarray([[10, 1, 8, 1, 0], [10, 0, 1, 8, 1]], dtype=np.int64)
        individual = k.score(c, "binary", [0, 1])[:, 0].mean()
        pooled = k.score(c.sum(axis=0), "binary", [0, 1])[0]
        self.assertGreater(abs(individual - pooled), 0.02)

    def test_extra_labels_containment_is_not_exact(self):
        f = k.features(row(), output(hate_type=["explicit", "irony"]), "hate_type")
        ix = {v: i for i, v in enumerate(k.names("hate_type"))}
        self.assertEqual(f[ix["contains_gold"]], 1)
        self.assertEqual(f[ix["exact"]], 0)
        self.assertEqual(f[ix["only_extra"]], 1)

    def test_unknown_annotation_not_negative(self):
        r = row(gold_hate_types=None)
        self.assertFalse(k.eligible(r, "hate_type"))
        self.assertEqual(k.features(r, output(), "hate_type").sum(), 0)
        self.assertEqual(k.features(r, output(), "known_answer")[1], 1)

    def test_known_empty_needs_gate_for_complete_answer(self):
        r = row(gold_hate_types=None, gold_target_groups=[])
        p = output(hate=False, hate_type=[])
        f = k.features(r, p, "target_group")
        ix = {v: i for i, v in enumerate(k.names("target_group"))}
        self.assertEqual(f[ix["exact"]], 1)
        self.assertEqual(f[ix["gated_exact"]], 0)
        self.assertEqual(k.features(r, p, "known_answer")[1], 0)

    def test_null_is_not_benign_prediction(self):
        self.assertEqual(k.features(row(), None, "binary").sum(), 0)
        self.assertEqual(
            k.features(row(), output(hate=False, hate_type=[]), "binary").tolist(),
            [1, 0, 0, 0, 1],
        )

    def test_severity_none_is_failure_not_low(self):
        f = k.features(row(gold_severity="low"), output(), "severity")
        ix = {v: i for i, v in enumerate(k.names("severity"))}
        self.assertEqual(f[ix["low:fn"]], 1)
        self.assertEqual(f[ix["confusion:low:none"]], 1)
        self.assertEqual(f[ix["provided"]], 0)
        metrics = k.score(f, "severity", [0, 1, 2])
        self.assertEqual(metrics[2], 0)
        self.assertTrue(np.isnan(metrics[5]))

    def test_ordinal_direction(self):
        for gold, pred, delta in [
            ("low", "high", 2),
            ("high", "medium", -1),
            ("medium", "medium", 0),
        ]:
            f = k.features(row(gold_severity=gold), output(severity=pred), "severity")
            ix = {v: i for i, v in enumerate(k.names("severity"))}
            self.assertEqual(f[ix["rank_signed"]], delta)
            self.assertEqual(f[ix["rank_absolute"]], abs(delta))

    def test_absent_label_excluded_only_from_macro(self):
        f = k.features(
            row(gold_target_groups=["race"]),
            output(target_group=["race", "other"]),
            "target_group",
        )
        s = k.score(f, "target_group", [0])
        self.assertEqual(s[0], 1)
        self.assertAlmostEqual(s[1], 2 / 3)
        self.assertEqual(s[2], 0)

    def test_fixed_label_zero_denominator(self):
        f = k.features(row(), output(), "hate_type")
        self.assertAlmostEqual(k.score(f, "hate_type", list(range(7)))[0], 1 / 7)
        self.assertTrue(
            np.isnan(k.score(np.zeros(5, dtype=int), "binary", [0, 1])).all()
        )

    def test_representatives_by_gold_before_validity(self):
        rows = [
            row(key=x, gold_hate=g) for x, g in [("a", False), ("b", True), ("c", True)]
        ]
        chosen = min(
            [1, 2],
            key=lambda i: hashlib.sha256(
                ("A16-representative|" + rows[i]["key"]).encode()
            ).hexdigest(),
        )
        valid = np.ones((3, 4, 3), dtype=bool)
        valid[chosen, 0, 0] = False
        v = np.full((3, 4, 3), 3, dtype=np.int8)
        v[chosen, 0, 0] = 0
        eligible, common = k.selected(
            rows, [0, 1, 2], "binary", "representative", set(), valid, v
        )
        self.assertEqual(len(eligible), 2)
        self.assertEqual(common.tolist(), [0])

    def test_missing_bounds_enclose_manual_binary(self):
        rows = [row(gold_hate=False), row(gold_hate=True)]
        data = np.zeros((2, 4, 3, 5), dtype=np.int16)
        data[0, :, :, :] = [1, 0, 1, 0, 0]
        valid = np.ones((2, 4, 3), dtype=bool)
        valid[1] = False
        votes = valid.astype(np.int8) * 3
        bounds = k.missing_bounds(
            rows, data, np.array([0, 1]), valid, votes, "binary", [0, 1]
        )
        self.assertEqual(
            bounds["metrics"]["balanced_accuracy"]["reference"], [0.5, 1.0]
        )
        self.assertAlmostEqual(bounds["metrics"]["macro_f1"]["reference"][0], 1 / 3)

    def test_identical_runs_exact_zero_difference(self):
        scores = np.tile(
            np.array([[0.123, 0.6], [0.647, 0.5], [0.912, 0.2]]), (4, 1, 1)
        )
        self.assertTrue(np.array_equal(k.summaries(scores)[5], [0.0, 0.0]))

    def test_cluster_bootstrap_preserves_connected_items(self):
        meta = [row(key="a"), row(key="b")]
        data = np.zeros((2, 4, 3, 3), dtype=np.int16)
        data[..., 0] = 1
        data[0, ..., 1:] = 1
        data[1, 1:, :, 1:] = 1
        draws, info = k.bootstrap(
            meta,
            data,
            np.array([0, 1]),
            np.array([0, 1]),
            "known_answer",
            [],
            9,
            draws=40,
        )
        self.assertEqual(info["units"], 1)
        self.assertTrue(np.all(draws[:, 5, 0] == 0.5))


if __name__ == "__main__":
    unittest.main()

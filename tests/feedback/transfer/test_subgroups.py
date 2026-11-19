import hashlib
import json
import unittest
import numpy as np
from analysis.feedback.transfer import subgroups as p


def base(key="a", source="hatecheck", truth=False):
    return dict(
        key=key,
        id=key,
        population="en_hatecheck_eval",
        source=source,
        inference_cluster="c",
        functional_cluster="f",
        text_cluster=key,
        overlap_cluster=key,
        gate=truth,
        hate_types=None,
        target_groups=None,
        severity=None,
    )


class Boundaries(unittest.TestCase):
    def test_hatecheck_nan_identity_and_unknown_gold(self):
        r = base()
        r["raw"] = json.dumps(
            dict(
                functionality="target_indiv_nh",
                templ_id=1,
                case_id=1,
                ref_case_id=None,
                target_ident=float("nan"),
                block="guideline_covered",
                direction="-",
            )
        )
        m = p.annotation(r)
        self.assertEqual(m["target_identity"], "not_specified")
        self.assertIsNone(m["fine_gold"]["target_groups"])

    def test_function_polarity_guard(self):
        r = base(truth=True)
        r["raw"] = json.dumps(dict(functionality="target_indiv_nh"))
        with self.assertRaises(Exception):
            p.annotation(r)

    def test_hatexplain_offensive_stays_benign(self):
        r = base(source="hatexplain")
        r["raw"] = json.dumps(
            dict(votes=["offensive", "offensive", "hatespeech"], official_split="train")
        )
        m = p.annotation(r)
        self.assertFalse(m["gold_hate"])
        self.assertEqual(m["annotation_agreement"], "split")

    def test_mhs_frozen_boundaries(self):
        for score, truth, severity in [
            (0.5, False, None),
            (0.51, True, "low"),
            (1, True, "low"),
            (1.63, True, "medium"),
            (1.64, True, "high"),
        ]:
            r = base(source="mhs", truth=truth)
            r.update(
                severity=severity, raw=json.dumps(dict(score=score, n_annotators=3))
            )
            self.assertEqual(p.annotation(r)["fine_gold"]["severity"], severity)

    def test_representative_inside_group_before_availability(self):
        keys = sorted(
            ["a", "b"],
            key=lambda s: hashlib.sha256((p.k.REP_PREFIX + s).encode()).hexdigest(),
        )
        meta = [
            dict(key=k, gold_hate=False, population="x", inference_cluster="same")
            for k in keys
        ]
        error = np.zeros((2, 4, 3), dtype=np.int8)
        votes = np.full_like(error, 3)
        error[0, 0, 0] = -1
        votes[0, 0, 0] = 0
        c = p.subset(meta, error, votes, [0, 1], False, "representative", set())
        self.assertEqual(c["eligible"].tolist(), [0])
        self.assertEqual(len(c["common"]), 0)
        c = p.subset(meta, error, votes, [1], False, "representative", set())
        self.assertEqual(c["common"].tolist(), [1])

    def test_undefined_opposite_polarity(self):
        meta = [dict(key="a", gold_hate=True, population="x", inference_cluster="a")]
        e = np.zeros((1, 4, 3), dtype=np.int8)
        v = np.full_like(e, 3)
        c = p.subset(meta, e, v, [0], False, "main", set())
        self.assertIsNone(c["point"]["difference"])
        self.assertEqual(c["point"]["eligible_n"], 0)

    def test_replay_before_representative(self):
        meta = [
            dict(key=k, gold_hate=True, population="x", inference_cluster="a")
            for k in ["a", "b"]
        ]
        e = np.zeros((2, 4, 3), dtype=np.int8)
        v = np.full_like(e, 3)
        c = p.subset(meta, e, v, [0, 1], True, "representative_replay_excluded", {"a"})
        self.assertEqual(c["eligible"].tolist(), [1])

    def test_one_connected_cluster_remains_one(self):
        meta = [
            dict(key=k, gold_hate=True, population="x", inference_cluster="a")
            for k in ["a", "b"]
        ]
        e = np.zeros((2, 4, 3), dtype=np.int8)
        e[0, 1:, :] = 1
        v = np.full_like(e, 3)
        c = p.subset(meta, e, v, [0, 1], True, "main", set())
        d, info = p.k.bootstrap(meta, e, c, 12, draws=40)
        self.assertEqual(info["units"], 1)
        self.assertTrue(np.all(d[:, 5] == 0.5))


if __name__ == "__main__":
    unittest.main()

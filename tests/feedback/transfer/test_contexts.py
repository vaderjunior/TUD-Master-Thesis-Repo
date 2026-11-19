import unittest
from copy import deepcopy
import numpy as np
from analysis.feedback.transfer import contexts_metrics as k


def hit(i, gate=True):
    return dict(
        id=str(i),
        text="Text " + str(i),
        meta=dict(
            gate=gate, hate_types=None, target_groups=None, severity=None, legal=None
        ),
        via="rrf",
    )


def ctx():
    return dict(
        definitions=[{}, {}],
        guidelines=[{}, {}],
        example_groups=[["Labelled examples", [hit(i) for i in range(5)]]],
    )


class Boundaries(unittest.TestCase):
    def test_replacement_preserves_first_four(self):
        r = ctx()
        t = deepcopy(r)
        h = hit(9)
        t["example_groups"][0][1][-1] = h
        self.assertEqual(k.replacement(r, t, h)["id"], "4")
        t["example_groups"][0][1][0], t["example_groups"][0][1][1] = (
            t["example_groups"][0][1][1],
            t["example_groups"][0][1][0],
        )
        with self.assertRaises(AssertionError):
            k.replacement(r, t, h)

    def test_metadata_ranks_are_query_specific(self):
        r = ctx()
        t = deepcopy(r)
        h = hit(9)
        t["example_groups"][0][1][-1] = h
        t["example_groups"][0][1][0]["meta"]["_ranks"] = {"dense": 1}
        with self.assertRaises(AssertionError):
            k.replacement(r, t, h)

    def test_correction_fine_gold_cannot_be_invented(self):
        r = ctx()
        t = deepcopy(r)
        h = hit(9)
        h["meta"]["target_groups"] = []
        t["example_groups"][0][1][-1] = h
        with self.assertRaises(AssertionError):
            k.replacement(r, t, h)

    def test_unknown_versus_empty_rendering(self):
        self.assertEqual(k.gold(hit(0, False)["meta"]), "hate=false")
        h = hit(0)
        h["meta"]["hate_types"] = []
        self.assertEqual(k.gold(h["meta"]), "hate=true, hate_type=[]")

    def test_whitespace_and_literal_text(self):
        h = hit(0)
        h["text"] = 'a\n\n b "c"'
        self.assertEqual(k.example(h), '- "a b "c""\n  labels: hate=true')

    def test_dense_and_lexical_rank_fusion(self):
        ids = [str(i) for i in range(10)]
        trace = {
            "dense": [dict(id=s, score=1 - i * 0.03) for i, s in enumerate(ids)],
            "bm25": [dict(id="2", score=2.0), dict(id="1", score=1.0)],
        }
        scores = {x: 1 / (61 + i) for i, x in enumerate(ids)}
        scores["2"] += 1 / 61
        scores["1"] += 1 / 62
        trace["rrf"] = [
            dict(id=x, score=scores[x])
            for x in sorted(scores, key=lambda x: (-scores[x], x))
        ]
        trace["selected_id"] = trace["rrf"][0]["id"]
        out = k.fuse(trace, set(ids))
        self.assertFalse(out["dense_winner_same"])
        self.assertEqual(out["retrieval_channels"], "both")
        trace["bm25"][0]["score"] = 0
        with self.assertRaises(AssertionError):
            k.fuse(trace, set(ids))

    def test_cosine_is_normalized(self):
        self.assertEqual(k.cosine([2, 0], [4, 0]), 1.0)
        with self.assertRaises(AssertionError):
            k.cosine([0, 0], [1, 0])

    def test_bins_have_fixed_boundaries(self):
        self.assertEqual(
            [k.similarity_bin(x) for x in [0.399, 0.4, 0.5, 0.6, 0.7, 0.9494]],
            [0, 1, 2, 3, 4, 4],
        )

    def test_counts_are_assignments_not_independent_references(self):
        x = k.point([2, 3, 6])
        self.assertEqual(x["reference"], 0.5)
        self.assertEqual(x["difference"], 0.5)
        self.assertIsNone(k.point([0, 0, 0])["difference"])

    def test_concentration_keeps_unused_sources(self):
        x = k.concentration([8, 2, 0])
        self.assertEqual(x["used"], 2)
        self.assertEqual(x["max_share"], 0.8)
        self.assertAlmostEqual(x["inverse_simpson"], 100 / 68)

    def test_cluster_bootstrap_keeps_connected_rows(self):
        x = k.boot(np.array([[1, 3, 0], [1, 0, 3]]), ["shared", "shared"], 1, draws=40)
        self.assertTrue(np.array_equal(x, np.tile([0.5, 0.5, 0.0], (40, 1))))


if __name__ == "__main__":
    unittest.main()

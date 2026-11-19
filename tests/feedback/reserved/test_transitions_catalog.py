import unittest
from analysis.feedback.reserved import transitions_catalog as c


class Catalog(unittest.TestCase):
    def test_query_rank_variants_are_preserved(self):
        d = {}
        base = {
            "id": "ex1",
            "text": "same",
            "meta": {"gate": True, "_ranks": {"bm25": 1}},
        }
        first = c.retain(d, base)
        second = c.retain(d, {**base, "meta": {"gate": True, "_ranks": {"bm25": 3}}})
        self.assertNotEqual(first, second)
        self.assertEqual(len(d["ex1"]["metadata_variants"]), 2)
        self.assertEqual(c.retain(d, base), first)

    def test_hidden_label_stays_unknown(self):
        d = {}
        e = {"id": "feedback", "text": "same", "meta": {"gate": True}}
        visible = c.retain(d, e)
        hidden = c.retain(d, {**e, "meta": {"gate": None}})
        self.assertIsNone(d["feedback"]["metadata_variants"][hidden]["gate"])
        self.assertTrue(d["feedback"]["metadata_variants"][visible]["gate"])

    def test_changed_text_is_rejected(self):
        d = {}
        e = {"id": "ex1", "text": "same", "meta": {}}
        c.retain(d, e)
        with self.assertRaises(ValueError):
            c.retain(d, {**e, "text": "different"})

    def test_metadata_key_order_does_not_create_variant(self):
        d = {}
        e = {"id": "ex1", "text": "same", "meta": {"a": 1, "b": 2}}
        self.assertEqual(c.retain(d, e), c.retain(d, {**e, "meta": {"b": 2, "a": 1}}))
        self.assertEqual(len(d["ex1"]["metadata_variants"]), 1)


if __name__ == "__main__":
    unittest.main()

import unittest
from analysis.common import review_replay as r


class ReviewTests(unittest.TestCase):
    def hit(self, key, score=1):
        return dict(id=key, score=score, vec_sha256="fixed")

    def trace(self, rows):
        return [dict(operation="dense", hits=rows)]

    def test_membership_is_not_tie_order(self):
        change = r.trace_changes(
            self.trace([self.hit("a"), self.hit("b")]),
            self.trace([self.hit("a"), self.hit("c")]),
        )[0]
        self.assertFalse(change["same_candidate_set"])
        self.assertTrue(change["common_hit_values_equal"])
        self.assertEqual(change["only_left"], ["b"])
        self.assertEqual(change["only_right"], ["c"])

    def test_scores_cannot_be_misreported_as_unchanged(self):
        change = r.trace_changes(
            self.trace([self.hit("a")]), self.trace([self.hit("a", 2)])
        )[0]
        self.assertTrue(change["same_candidate_set"])
        self.assertFalse(change["common_hit_values_equal"])

    def test_candidate_order_only(self):
        change = r.trace_changes(
            self.trace([self.hit("a"), self.hit("b")]),
            self.trace([self.hit("b"), self.hit("a")]),
        )[0]
        self.assertTrue(change["same_candidate_set"])
        self.assertFalse(change["same_order"])

    def test_unexpected_operation_sequence_refused(self):
        with self.assertRaises(RuntimeError):
            r.trace_changes([dict(operation="dense")], [dict(operation="bm25")])

    def test_sensitivity_keeps_original_identity_order(self):
        rows = [
            dict(key="b", phase="en"),
            dict(key="a", phase="de"),
            dict(key="c", phase="en"),
        ]
        self.assertEqual(
            r.exclusion_roster(rows, ["a", "b"]),
            [dict(key="b", population="en"), dict(key="a", population="de")],
        )
        with self.assertRaises(RuntimeError):
            r.exclusion_roster(rows, ["absent"])
        with self.assertRaises(RuntimeError):
            r.exclusion_roster(rows + rows, ["a"])


if __name__ == "__main__":
    unittest.main()

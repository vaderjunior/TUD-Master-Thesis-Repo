import unittest
from analysis.feedback.rounds import policy as p


class Policies(unittest.TestCase):
    def rows(self, n=64):
        return [{"id": str(i), "gold": {"hate": bool(i % 2)}} for i in range(n)]

    def example(self, n=20):
        rows = self.rows(n)
        wrong = {r["id"]: [not r["gold"]["hate"]] * 3 for r in rows}
        return (
            rows,
            wrong,
            {k: list(v) for k, v in wrong.items()},
            {s: [] for s in p.POLICIES},
        )

    def test_partition_all_four_blocks_balanced(self):
        rows = self.rows()
        blocks = p.partition(rows, 9)
        self.assertEqual(
            sorted(i for b in blocks for i in b), sorted(r["id"] for r in rows)
        )
        self.assertTrue(
            all(len(b) == 16 and sum(int(i) % 2 for i in b) == 8 for b in blocks)
        )
        self.assertEqual(blocks, p.partition(list(reversed(rows)), 9))

    def test_duplicate_source_rejected(self):
        with self.assertRaises(RuntimeError):
            p.partition(self.rows() + self.rows()[:1], 9)

    def test_majority_and_any_null(self):
        self.assertIs(p.gate([True, False, True]), True)
        self.assertIsNone(p.gate([True, None, True]))

    def test_equal_balanced_budgets(self):
        a = p.verified_selection(*self.example(), 9, 1)
        self.assertEqual(a["per_class_budget"], 5)
        self.assertTrue(
            all(
                len(v) == 10 and sum(int(i) % 2 for i in v) == 5
                for v in a["additions"].values()
            )
        )
        self.assertEqual(a["additions"]["adaptive"], a["additions"]["static"])

    def test_scarcity_limits_every_arm(self):
        rows, old, new, prev = self.example()
        for r in rows:
            if r["gold"]["hate"] and r["id"] != "1":
                new[r["id"]] = [True] * 3
        a = p.verified_selection(rows, old, new, prev, 9, 1)
        self.assertEqual(a["per_class_budget"], 1)
        self.assertTrue(all(len(v) == 2 for v in a["additions"].values()))

    def test_no_error_no_invented_fill(self):
        rows, old, new, prev = self.example()
        for r in rows:
            new[r["id"]] = [r["gold"]["hate"]] * 3
        a = p.verified_selection(rows, old, new, prev, 9, 1)
        self.assertEqual(a["per_class_budget"], 0)
        self.assertEqual(a["cumulative"], prev)

    def test_null_common_mask(self):
        rows, old, new, prev = self.example()
        new["1"][0] = None
        a = p.verified_selection(rows, old, new, prev, 9, 1)
        self.assertEqual(a["invalid_ids"], ["1"])
        self.assertTrue(all("1" not in v for v in a["additions"].values()))

    def test_current_only_changes_adaptive_error_set(self):
        rows, old, new, prev = self.example(40)
        a = p.verified_selection(rows, old, new, prev, 9, 1)
        new["3"] = [True] * 3
        new["4"] = [False] * 3
        b = p.verified_selection(rows, old, new, prev, 9, 1)
        self.assertEqual(a["additions"]["static"], b["additions"]["static"])
        self.assertEqual(a["additions"]["random"], b["additions"]["random"])

    def test_prior_identity_reuse_rejected(self):
        rows, old, new, prev = self.example()
        prev["adaptive"] = ["1"]
        with self.assertRaises(RuntimeError):
            p.select(rows, old, new, prev, 9, 1)

    def test_four_round_cumulative_and_order_invariance(self):
        rows = self.rows(160)
        by = {r["id"]: r for r in rows}
        prev = {s: [] for s in p.POLICIES}
        for n, ids in enumerate(p.partition(rows, 18), 1):
            block = [by[i] for i in ids]
            wrong = {i: [not by[i]["gold"]["hate"]] * 3 for i in ids}
            a = p.verified_selection(block, wrong, wrong, prev, 18, n)
            b = p.verified_selection(list(reversed(block)), wrong, wrong, prev, 18, n)
            self.assertEqual(a, b)
            prev = a["cumulative"]
        self.assertTrue(all(len(v) == 40 and len(set(v)) == 40 for v in prev.values()))


if __name__ == "__main__":
    unittest.main()

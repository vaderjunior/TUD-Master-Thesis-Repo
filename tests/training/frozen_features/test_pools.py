import unittest
from collections import Counter
from training.frozen_features import pools as m


class PoolContracts(unittest.TestCase):
    def test_shuffle_retains_complete_blocks_and_ordered_text_ids(self):
        rows = [{"id": str(i), "target": [i % 2, int(i % 3 == 0)]} for i in range(15)]
        output = m.permute_blocks(rows, 201)
        self.assertEqual([r["id"] for r in rows], [r["id"] for r in output])
        self.assertEqual(
            Counter(tuple(r["target"]) for r in rows),
            Counter(tuple(r["target"]) for r in output),
        )
        self.assertNotEqual([r["target"] for r in rows], [r["target"] for r in output])
        self.assertEqual(output, m.permute_blocks(rows, 201))

    def test_hash_draw_is_independent_of_input_order(self):
        ids = ["c", "a", "b", "d"]
        self.assertEqual(
            sorted(ids, key=lambda k: m.rank("test", 101, k)),
            sorted(reversed(ids), key=lambda k: m.rank("test", 101, k)),
        )


if __name__ == "__main__":
    unittest.main()

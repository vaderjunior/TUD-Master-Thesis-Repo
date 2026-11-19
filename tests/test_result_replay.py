import tempfile
from pathlib import Path
import unittest
from scripts import replay_results as r


class ReplayTests(unittest.TestCase):
    def test_missing_remains_missing(self):
        y = [{"yes"}, {"no"}]
        p = [{"yes"}, None]
        self.assertEqual(r.f1(y, p, ["yes"]), 1.0)
        self.assertEqual(r.failure_bounds(y, p, ["yes"]), [2 / 3, 1.0])
        self.assertIsNone(r.f1(y, [None, None], ["yes"]))

    def test_macro_not_micro(self):
        self.assertAlmostEqual(
            r.f1([{"a", "b"}, {"a"}, set()], [{"a"}, set(), set()], ["a", "b"]), 1 / 3
        )

    def test_empty_label_denominator_zero(self):
        self.assertEqual(r.f1([set()], [set()], ["a"]), 0.0)

    def test_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                r.checked(Path(d), "../outside.json")


if __name__ == "__main__":
    unittest.main()

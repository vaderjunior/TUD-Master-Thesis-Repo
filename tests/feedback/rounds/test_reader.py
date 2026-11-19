import unittest
from tests.feedback.rounds.test_reader_earlier import InheritedPureTests
from tests.feedback.rounds.test_reader_earlier import IterativeReaderTests
from analysis.feedback.rounds import reader as a


class ExplicitAxesTests(unittest.TestCase):
    def test_evaluation_policy_and_stream(self):
        original = {
            "role": "evaluation",
            "condition": "static_s2026101202",
            "stream": None,
            "result": None,
        }
        row = a.explicit_axes(original)
        self.assertEqual(
            (row["policy"], row["stream"], row["source_phase_stream"]),
            ("static", 2026101202, None),
        )
        self.assertEqual(original["stream"], None)
        self.assertIsNone(row["result"])

    def test_shared_reference_and_source(self):
        reference = a.explicit_axes(
            {"role": "evaluation", "condition": "reference", "stream": None}
        )
        self.assertEqual(
            (reference["policy"], reference["stream"]), ("reference", None)
        )
        source = a.explicit_axes(
            {"role": "source", "condition": "adaptive", "stream": 2026101203}
        )
        self.assertEqual(
            (source["policy"], source["stream"], source["source_phase_stream"]),
            ("adaptive", 2026101203, 2026101203),
        )


if __name__ == "__main__":
    unittest.main()

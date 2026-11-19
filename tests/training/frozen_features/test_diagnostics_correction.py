import unittest
from training.frozen_features.diagnostics import roster_outputs


class EndpointAlignment(unittest.TestCase):
    def test_exact_requested_roster_order(self):
        values = {"a": 1, "b": 2, "c": 3}
        self.assertEqual(
            list(roster_outputs([{"id": "b"}, {"id": "a"}], values).items()),
            [("b", 2), ("a", 1)],
        )

    def test_missing_requested_prediction_stops(self):
        with self.assertRaises(RuntimeError):
            roster_outputs([{"id": "z"}], {"a": 1})

    def test_duplicate_requested_identity_stops(self):
        with self.assertRaises(RuntimeError):
            roster_outputs([{"id": "a"}, {"id": "a"}], {"a": 1})


if __name__ == "__main__":
    unittest.main()

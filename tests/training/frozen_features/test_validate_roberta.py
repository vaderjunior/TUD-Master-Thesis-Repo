import json
import unittest
from training.frozen_features.validate_roberta import json_info


class LoadingMetadata(unittest.TestCase):
    def test_sets_become_stable_lists_without_losing_keys(self):
        info = {
            "missing_keys": set(),
            "unexpected_keys": {"lm_head.b", "lm_head.a"},
            "nested": ({"a": frozenset([2, 1])},),
        }
        result = json.loads(json.dumps(json_info(info)))
        self.assertEqual(
            result,
            {
                "missing_keys": [],
                "unexpected_keys": ["lm_head.a", "lm_head.b"],
                "nested": [{"a": [1, 2]}],
            },
        )


if __name__ == "__main__":
    unittest.main()

import copy
import unittest
from analysis.common.replay_retrieval import comparison


class ReplayTests(unittest.TestCase):
    def base(self):
        return {
            "key": "en/test",
            "messages": [{"role": "user", "text": "saved prompt"}],
            "context": {"hit": {"id": "a", "rank": 1}},
            "retrieved": {"examples": ["a"]},
        }

    def test_identical(self):
        x = self.base()
        self.assertTrue(comparison(x, copy.deepcopy(x))["same_entry"])

    def test_metadata_only_is_retained(self):
        x = self.base()
        y = copy.deepcopy(x)
        y["context"]["hit"]["rank"] = 2
        c = comparison(x, y)
        self.assertFalse(c["same_entry"])
        self.assertTrue(c["same_messages"])
        self.assertEqual(c["changed_fields"], ["context"])
        self.assertEqual(x["context"]["hit"]["rank"], 1)

    def test_changed_prompt_and_ids(self):
        x = self.base()
        y = copy.deepcopy(x)
        y["messages"][0]["text"] = "different"
        y["retrieved"]["examples"] = ["b"]
        c = comparison(x, y)
        self.assertFalse(c["same_messages"])
        self.assertFalse(c["same_retrieved_ids"])
        self.assertEqual(c["changed_fields"], ["messages", "retrieved"])

    def test_provenance_difference_not_hidden(self):
        x = self.base()
        y = copy.deepcopy(x)
        y["unexpected_field"] = None
        self.assertEqual(comparison(x, y)["changed_fields"], ["unexpected_field"])


if __name__ == "__main__":
    unittest.main()

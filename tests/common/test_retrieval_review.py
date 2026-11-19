import copy
import unittest
from analysis.common import validate_retrieval as audit


class ReviewCorruptionTests(unittest.TestCase):
    def setUp(self):
        self.old = {
            "key": "en/test",
            "context": {"rank": 1},
            "messages": [{"role": "user", "text": "example"}],
            "retrieved": {"examples": ["one"]},
        }
        for part in ("context", "messages"):
            self.old[part + "_sha256"] = audit.signature(self.old[part])
        self.row = {
            "key": "en/test",
            "same_entry": True,
            "same_context": True,
            "same_messages": True,
            "same_retrieved_ids": True,
            "changed_fields": [],
            "actual_context_sha256": self.old["context_sha256"],
            "actual_messages_sha256": self.old["messages_sha256"],
        }

    def test_complete_exact_entry(self):
        self.assertEqual(
            audit.check_log([self.row], [self.old])["entry_differences"], 0
        )

    def test_omitted_input_rejected(self):
        with self.assertRaises(RuntimeError):
            audit.check_log([], [self.old])

    def test_false_claim_of_matching_hash_rejected(self):
        self.row["actual_messages_sha256"] = "0" * 64
        with self.assertRaises(RuntimeError):
            audit.check_log([self.row], [self.old])

    def test_missing_actual_difference_rejected(self):
        self.row["same_entry"] = False
        self.row["changed_fields"] = ["context"]
        with self.assertRaises(RuntimeError):
            audit.check_log([self.row], [self.old])

    def test_captured_metadata_difference_is_not_hidden(self):
        actual = copy.deepcopy(self.old)
        actual["context"]["rank"] = 2
        actual["context_sha256"] = audit.signature(actual["context"])
        self.row.update(
            actual_entry=actual,
            same_entry=False,
            same_context=False,
            changed_fields=["context", "context_sha256"],
            actual_context_sha256=actual["context_sha256"],
        )
        totals = audit.check_log([self.row], [self.old])
        self.assertEqual(
            totals,
            {
                "items": 1,
                "entry_differences": 1,
                "context_differences": 1,
                "message_differences": 0,
                "retrieved_id_differences": 0,
            },
        )


if __name__ == "__main__":
    unittest.main()

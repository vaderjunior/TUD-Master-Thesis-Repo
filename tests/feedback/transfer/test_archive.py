from copy import deepcopy
import hashlib
import socket
import unittest
from unittest.mock import Mock, patch

from analysis.feedback.transfer import archive as a


def vote(hate=True, types=None, severity="medium", legal=None):
    return {
        "hate": hate,
        "hate_type": (types or []) if hate else [],
        "target_group": [],
        "severity": severity if hate else None,
        "legal": legal or [],
    }


class ReaderTests(unittest.TestCase):
    def test_vote_majority_and_lower_median(self):
        votes = [
            vote(types=["irony"], severity="high"),
            vote(types=["irony"], severity="low"),
            vote(False),
        ]
        result, tie = a.aggregate_votes(votes)
        self.assertEqual(result, vote(types=["irony"], severity="low"))
        self.assertFalse(tie)

    def test_null_and_partial_vote(self):
        self.assertEqual(a.aggregate_votes([None] * 3), (None, False))
        self.assertEqual(a.aggregate_votes([None, vote(), None]), (vote(), False))

    def test_tie_false_gate_but_legal_retained(self):
        yes = vote(types=["irony"], legal=["insult_defamation"])
        no = vote(False, legal=["insult_defamation"])
        self.assertEqual(a.aggregate_votes([yes, no, None]), (no, True))

    def test_bad_labels_and_false_gate_rejected(self):
        for value in (
            vote(types=["unknown"]),
            vote(types=["irony", "irony"]),
            dict(vote(), hate=1),
            dict(vote(False), hate_type=["irony"]),
        ):
            with self.assertRaises(ValueError):
                a.validate_prediction(value)

    def test_hash_change_rejected(self):
        raw = b'{"x": 1}'
        fake = Mock()
        fake.read_bytes.return_value = raw
        with patch.object(a, "physical", return_value=fake):
            self.assertEqual(
                a.read_bound("unused", hashlib.sha256(raw).hexdigest()), {"x": 1}
            )
            with self.assertRaises(ValueError):
                a.read_bound("unused", "0" * 64)

    def test_identity_axes_change_item_signature(self):
        job = {"study": "transfer", "key": "a", "condition": "reference", "repeat": 0}
        values = {a.item_id(job)}
        for key, value in [
            ("study", "other"),
            ("key", "b"),
            ("condition", "other"),
            ("repeat", 1),
        ]:
            values.add(a.item_id(dict(job, **{key: value})))
        self.assertEqual(len(values), 5)

    def test_saved_identity_prompt_count_and_aggregate_rejected(self):
        job = {
            "study": "transfer",
            "key": "a",
            "population": "en_test",
            "condition": "reference",
            "repeat": 0,
            "messages_sha256": "m",
        }
        saved = {
            "identity": a.identity(job),
            "execution_mode": "live",
            "messages_sha256": "m",
            "run_predictions": [vote()] * 3,
            "n_runs": 3,
            "n_valid": 3,
            "result": vote(),
            "uncertain": False,
        }
        bad_identity = dict(saved, identity=dict(saved["identity"], key="other"))
        cases = [
            bad_identity,
            dict(saved, messages_sha256="other"),
            dict(saved, n_valid=2),
            dict(saved, result=vote(False)),
            dict(saved, uncertain=True),
        ]
        for bad in cases:
            with self.assertRaises(ValueError):
                a.compact(bad, job, {}, "receipt", "result")

    def test_output_and_path_boundaries(self):
        with self.assertRaises(ValueError):
            a.write(a.STUDY / "forbidden.json", {})
        with self.assertRaises(ValueError):
            a.inside(a.OUT, "../elsewhere")

    def test_network_block(self):
        with a.offline(), self.assertRaises(AssertionError):
            socket.create_connection(("127.0.0.1", 1))

    def fixture(self):
        source = {
            "key": "a",
            "population": "en_test",
            "source": "example",
            "source_file": "example",
            "gate": True,
            "hate_types": None,
            "target_groups": [],
            "severity": None,
            **{
                k: "cluster"
                for k in (
                    "inference_cluster",
                    "functional_cluster",
                    "text_cluster",
                    "overlap_cluster",
                )
            },
        }
        conditions = ["reference", "one", "two", "three"]
        records = [
            {
                "key": "a",
                "population": "en_test",
                "condition": c,
                "repeat": repeat,
                "stage": "transfer_01",
                "execution_period": "original_8",
                "n_valid": 3,
                "result": vote(),
                "uncertain": False,
            }
            for c in conditions
            for repeat in range(3)
        ]
        return source, conditions, records

    def test_missing_gold_not_negative(self):
        source, conditions, records = self.fixture()
        row = a.matrix_row(source, 0, records, conditions, set())
        self.assertIsNone(row["gold_hate_types"])
        self.assertFalse(row["known_fine_gold"]["hate_types"])
        self.assertEqual(row["gold_target_groups"], [])
        self.assertTrue(row["known_fine_gold"]["target_groups"])
        self.assertEqual([len(x) for x in row["predictions"]], [3] * 4)

    def test_null_and_strict_and_replay_masks(self):
        source, conditions, records = self.fixture()
        records[0]["result"] = None
        records[0]["n_valid"] = 0
        row = a.matrix_row(source, 0, records, conditions, {"a"})
        self.assertFalse(row["masks"]["common_valid"])
        self.assertFalse(row["masks"]["strict_all_votes"])
        self.assertTrue(row["masks"]["replay_exclusion_member"])
        records[0]["result"] = vote()
        records[0]["n_valid"] = 1
        row = a.matrix_row(source, 0, records, conditions, {"a"})
        self.assertTrue(row["masks"]["common_valid"])
        self.assertFalse(row["masks"]["strict_all_votes"])
        self.assertFalse(row["masks"]["common_valid_exclude_replay"])

    def test_missing_duplicate_wrong_join_rejected(self):
        source, conditions, records = self.fixture()
        variants = [records[:-1], records[:-1] + [records[0]], deepcopy(records)]
        variants[-1][0]["key"] = "other"
        for bad in variants:
            with self.assertRaises(ValueError):
                a.matrix_row(source, 0, bad, conditions, set())

    def test_fixed_support_requires_known_hateful_gold(self):
        source, _, _ = self.fixture()
        roster = [
            source,
            dict(source, key="b", gate=False, hate_types=["irony"]),
            dict(source, key="c", hate_types=["explicit"]),
        ]
        support = a.fixed_support(roster)
        self.assertEqual(support["en_test"]["hate_types"]["rows"], 1)
        self.assertEqual(
            support["en_test"]["hate_types"]["positive_support"], {"explicit": 1}
        )
        self.assertEqual(support["en_test"]["legal"]["rows"], 0)


if __name__ == "__main__":
    unittest.main()

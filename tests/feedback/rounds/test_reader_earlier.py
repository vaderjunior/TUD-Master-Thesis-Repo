from copy import deepcopy
from pathlib import Path
import unittest

from analysis.feedback.rounds import reader_earlier as a
from tests.feedback.transfer.test_archive import ReaderTests as InheritedPureTests
from tests.feedback.transfer.test_archive import vote


class IterativeReaderTests(unittest.TestCase):
    def test_duplicate_vote_labels_match_frozen_voter(self):
        from types import SimpleNamespace
        from src.hsrag.schema import Result
        from src.hsrag.vote import aggregate

        votes = [vote(types=["irony", "irony"]), vote(), vote()]
        for v in votes:
            a.validate_prediction(v)
        actual = aggregate(
            [
                SimpleNamespace(ok=True, result=Result(reasoning="fixture", **v))
                for v in votes
            ]
        )
        expected, tie = a.pure.aggregate_votes(votes)
        self.assertEqual(
            expected, {k: actual.result.model_dump()[k] for k in a.pure.FIELDS}
        )
        self.assertEqual(expected["hate_type"], ["irony"])
        self.assertEqual(tie, actual.uncertain)
        with self.assertRaises(ValueError):
            a.validate_prediction(vote(types=["unknown"]))

    def fixture(self):
        source = {
            "key": "x",
            "population": "en_test",
            "source": "example",
            "source_file": "example",
            "gate": True,
            "hate_types": None,
            "target_groups": [],
            "severity": None,
            **{
                k: "cluster"
                for k in [
                    "inference_cluster",
                    "functional_cluster",
                    "text_cluster",
                    "overlap_cluster",
                ]
            },
        }
        records = [
            {
                "key": "x",
                "population": "en_test",
                "condition": c,
                "repeat": d,
                "round": r,
                "role": "evaluation",
                "stage": "evaluation_01",
                "result": vote(),
                "n_valid": 3,
                "uncertain": False,
            }
            for r in range(1, 5)
            for c in a.CONDITIONS
            for d in range(3)
        ]
        return source, records

    def test_phase_namespace_isolation(self):
        job = {"study": "evaluation", "key": "x", "condition": "reference", "repeat": 0}
        one = a.bound_helpers("1" * 64, a.STUDY / "first")
        two = a.bound_helpers("2" * 64, a.STUDY / "second")
        original = a.pure.PLAN_SHA
        self.assertNotEqual(one.item_id(job), two.item_id(job))
        self.assertEqual(one.identity(job)["experiment_sha256"], "1" * 64)
        self.assertEqual(two.identity(job)["experiment_sha256"], "2" * 64)
        self.assertEqual(a.pure.PLAN_SHA, original)
        self.assertIsNot(one.compact.__globals__, two.compact.__globals__)

    def test_wrong_phase_saved_identity_rejected(self):
        job = {"study": "evaluation", "key": "x", "condition": "reference", "repeat": 0}
        one = a.bound_helpers("1" * 64, a.STUDY)
        two = a.bound_helpers("2" * 64, a.STUDY)
        with self.assertRaises(ValueError):
            one.compact({"identity": two.identity(job)}, job, {}, "", "")

    def test_matrix_axes_and_one_reference(self):
        source, records = self.fixture()
        row = a.matrix_row(source, 0, records, set())
        self.assertEqual(len(row["predictions"]), 4)
        self.assertTrue(
            all(
                len(r) == 10 and all(len(c) == 3 for c in r) for r in row["predictions"]
            )
        )
        self.assertEqual(sum(len(r[0]) for r in row["predictions"]), 12)
        self.assertIsNone(row["gold_hate_types"])
        self.assertEqual(row["gold_target_groups"], [])
        self.assertNotIn("execution_period", row)

    def test_round_and_longitudinal_masks(self):
        source, records = self.fixture()
        records[0].update(result=None, n_valid=0)
        records[30].update(n_valid=1)
        row = a.matrix_row(source, 0, records, {"x"})
        self.assertEqual(row["masks"]["round_common"], [False, True, True, True])
        self.assertEqual(row["masks"]["round_strict"], [False, False, True, True])
        self.assertFalse(row["masks"]["all_round_common"])
        self.assertTrue(row["masks"]["replay_exclusion_member"])

    def test_missing_duplicate_wrong_role_and_batch_rejected(self):
        source, records = self.fixture()
        variants = [records[:-1], records[:-1] + [records[0]]]
        for field, value in [("role", "source"), ("key", "other"), ("stage", "other")]:
            bad = deepcopy(records)
            bad[0][field] = value
            variants.append(bad)
        for bad in variants:
            with self.assertRaises(ValueError):
                a.matrix_row(source, 0, bad, set())

    def test_phase_roles_and_output_boundary(self):
        self.assertEqual(a.phase_axes("source_initial")["round"], 0)
        self.assertEqual(a.phase_axes("r3_source_s2026101202")["stream"], 2026101202)
        self.assertEqual(a.phase_axes("r4_evaluation")["role"], "evaluation")
        for bad in ["r5_evaluation", "r1_source_s2026101201", "transfer"]:
            with self.assertRaises(ValueError):
                a.phase_axes(bad)
        with self.assertRaises(ValueError):
            a.write(a.STUDY / "forbidden.json", {})


if __name__ == "__main__":
    unittest.main()

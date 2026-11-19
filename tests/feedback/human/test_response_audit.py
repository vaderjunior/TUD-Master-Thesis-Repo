import copy
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch
from tests.feedback.human.fixtures import response_fixture
from analysis.feedback.human import response_audit as a

ORIGINAL_QA = a.QA
ORIGINAL_QA_AVAILABLE = (ORIGINAL_QA / "source_mapping.json").is_file() and (
    ORIGINAL_QA / "project_mapping.json"
).is_file()


class ResponseAudit(unittest.TestCase):
    def setUp(self):
        self.data, self.mapping, self.fid = response_fixture("source")
        self.outputs = tempfile.TemporaryDirectory()
        self.addCleanup(self.outputs.cleanup)

    def decode(self, data=None):
        return a.decode(
            data or self.data, self.mapping, self.fid, allow_synthetic=True
        )[0]

    @unittest.skipUnless(
        ORIGINAL_QA_AVAILABLE,
        "Original browser QA exports and UI expectations are absent",
    )
    def test_both_actual_roundtrips_match_scripted_UI(self):
        def save(path, value):
            target = Path(self.outputs.name) / Path(path).name
            target.write_text(a.json.dumps(value), encoding="utf-8")

        with patch.object(a, "QA", ORIGINAL_QA), patch.object(a.b, "save", save):
            for role in ["source", "project"]:
                self.assertEqual(a.audit_fixture(role)["cards_matched"], 24)

    def test_QA_cannot_be_imported_as_human_by_default(self):
        with self.assertRaises(AssertionError):
            a.decode(self.data, self.mapping, self.fid)

    def test_uncertainty_and_missing_type_are_not_nonhate(self):
        cards = {x["id"]: x for x in self.decode()["cards"]}
        self.assertIsNone(cards["R031"]["binary_hate"])
        self.assertIsNone(cards["R025"]["binary_hate"])
        self.assertEqual(cards["R023"]["type_state"], "unassigned")
        self.assertTrue(cards["R023"]["binary_hate"])
        self.assertFalse(cards["R051"]["binary_hate"])

    def test_no_secondary_and_duplicate_have_distinct_handling(self):
        cards = {x["id"]: x for x in self.decode()["cards"]}
        self.assertEqual(cards["R008"]["types"], ["incitement"])
        self.assertEqual(cards["R039"]["types"], ["grievance"])
        self.assertIn("duplicate_primary_secondary", cards["R039"]["review_flags"])

    def test_unknown_choice_and_missing_gate_stop(self):
        gate = next(
            q["id"]
            for q in self.data["questions"]
            if q["title"] == "R051 · Classification"
        )
        for bad in [["unrecognised-option"], []]:
            data = copy.deepcopy(self.data)
            next(
                r
                for r in data["submissions"][0]["responses"]
                if r["questionId"] == gate
            )["answer"] = bad
            with self.assertRaises(AssertionError):
                self.decode(data)

    def test_wrong_form_partial_pagination_and_duplicate_response_stop(self):
        with self.assertRaises(AssertionError):
            a.decode(self.data, self.mapping, "wrong", True)
        for change in ["pagination", "incomplete", "duplicate"]:
            data = copy.deepcopy(self.data)
            if change == "pagination":
                data["hasMore"] = True
            elif change == "incomplete":
                data["submissions"][0]["isCompleted"] = False
            else:
                data["submissions"][0]["responses"].append(
                    data["submissions"][0]["responses"][0]
                )
            with self.assertRaises(AssertionError):
                self.decode(data)

    def test_nonapplicable_stale_label_retained_raw_but_ignored(self):
        data = copy.deepcopy(self.data)
        sub = data["submissions"][0]
        qid = next(
            q["id"]
            for q in data["questions"]
            if q["title"] == "R051 · Primary implicit type (if applicable)"
        )
        sub["responses"] = [r for r in sub["responses"] if r["questionId"] != qid]
        sub["responses"].append(
            {
                "questionId": qid,
                "formId": self.fid,
                "submissionId": sub["id"],
                "answer": [
                    self.mapping["fields"]["R051/primary"]["option_uuids"]["irony"]
                ],
            }
        )
        decoded = self.decode(data)
        card = next(x for x in decoded["cards"] if x["id"] == "R051")
        self.assertEqual(decoded["raw_fields"]["R051/primary"], ["irony"])
        self.assertEqual(card["types"], [])
        self.assertIn("ignored_nonapplicable_types", card["review_flags"])


if __name__ == "__main__":
    unittest.main()

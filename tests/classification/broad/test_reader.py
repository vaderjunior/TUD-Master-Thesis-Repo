import copy
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest

import yaml
from analysis.classification.broad import reader as m


def decision(hate=False, **changes):
    return (
        dict(
            reasoning="Synthetic test only",
            hate=hate,
            target_group=[],
            hate_type=[],
            severity=None,
            legal=[],
            **{}
        )
        | changes
    )


def vote_record(votes):
    result = m.gold.independent_aggregate(votes)
    valid = [v for v in votes if v is not None]
    if result is not None:
        chosen = next((v for v in valid if v["hate"] == result["hate"]), valid[0])
        result["reasoning"] = chosen["reasoning"]
    return dict(
        run_predictions=votes,
        raw_runs=[["{}"] for _ in votes],
        n_runs=len(votes),
        n_valid=len(valid),
        result=result,
        uncertain=bool(valid) and sum(v["hate"] for v in valid) * 2 == len(valid),
    )


class ScientificRules(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.taxonomy = yaml.safe_load(
            (m.ROOT / "config/taxonomy.yaml").read_text(encoding="utf8")
        )

    def test_all_null_stays_null(self):
        value = vote_record([None, None, None])
        m.check_votes(value, self.taxonomy)
        self.assertIsNone(value["result"])

    def test_null_does_not_vote_against_one_valid(self):
        value = vote_record(
            [None, decision(True, target_group=["race"], severity="high"), None]
        )
        m.check_votes(value, self.taxonomy)
        self.assertTrue(value["result"]["hate"])

    def test_tie_resolves_false_and_legal_remains(self):
        value = vote_record(
            [
                decision(True, legal=["insult_defamation"]),
                decision(legal=["insult_defamation"]),
                None,
            ]
        )
        m.check_votes(value, self.taxonomy)
        self.assertTrue(value["uncertain"])
        self.assertFalse(value["result"]["hate"])
        self.assertEqual(value["result"]["legal"], ["insult_defamation"])

    def test_label_half_is_not_majority(self):
        value = vote_record(
            [decision(True, target_group=["race"]), decision(True), None]
        )
        m.check_votes(value, self.taxonomy)
        self.assertEqual(value["result"]["target_group"], [])

    def test_severity_uses_lower_median(self):
        value = vote_record(
            [decision(True, severity="low"), decision(True, severity="high"), None]
        )
        m.check_votes(value, self.taxonomy)
        self.assertEqual(value["result"]["severity"], "low")

    def test_gate_disagreement_clears_fine_aggregate(self):
        value = vote_record(
            [decision(True, target_group=["race"], severity="high"), decision(), None]
        )
        m.check_votes(value, self.taxonomy)
        self.assertEqual(value["result"]["target_group"], [])
        self.assertIsNone(value["result"]["severity"])

    def test_missing_gold_is_not_empty_gold(self):
        self.assertIsNone(
            m.eligible_value(dict(gate=True, target_groups=None), "target_group")
        )
        self.assertEqual(
            m.eligible_value(dict(gate=True, target_groups=[]), "target_group"), []
        )

    def test_gold_gate_restricts_fine_not_legal(self):
        row = dict(gate=False, target_groups=["race"], legal=["insult_defamation"])
        self.assertIsNone(m.eligible_value(row, "target_group"))
        self.assertEqual(m.eligible_value(row, "legal"), ["insult_defamation"])

    def test_known_empty_needs_gate_for_complete_answer(self):
        row = dict(gate=True, target_groups=[])
        pred = decision()
        self.assertEqual(m.eligible_value(row, "target_group"), pred["target_group"])
        self.assertNotEqual(row["gate"], pred["hate"])

    def test_fixed_support_uses_full_gold_roster(self):
        rr = [
            dict(
                id=str(i), gate=True, target_groups=["race"] if i < 10 else ["religion"]
            )
            for i in range(15)
        ]
        ep = m.scoring.endpoint(rr, "target_group", self.taxonomy)
        self.assertEqual(ep["macro_labels"], ["race"])
        self.assertEqual(ep["rare_labels"], ["religion"])
        run = {str(i): None if i == 0 else decision() for i in range(15)}
        self.assertEqual(m.invalid_ids(ep["scorable_ids"], [run]), ["0"])
        self.assertEqual(ep["macro_labels"], ["race"])

    def test_all_compared_repeats_form_common_mask(self):
        one = {"a": decision(), "b": None, "c": decision()}
        two = {"a": None, "b": decision(), "c": decision()}
        self.assertEqual(m.invalid_ids(["a", "b", "c"], [one, two]), ["a", "b"])

    def test_missing_row_is_not_invalid_answer(self):
        with self.assertRaisesRegex(ValueError, "Missing compared input"):
            m.invalid_ids(["a", "b"], [{"a": decision()}])

    def test_reject_malformed_vote_count(self):
        value = vote_record([decision(), None, None])
        value["n_valid"] = 3
        with self.assertRaises(ValueError):
            m.check_votes(value, self.taxonomy)

    def test_reject_null_relabelled_negative(self):
        value = vote_record([None, None, None])
        value["result"] = decision()
        with self.assertRaises(ValueError):
            m.check_votes(value, self.taxonomy)

    def test_reject_contradictory_gate(self):
        with self.assertRaises(ValueError):
            m.check_decision(decision(target_group=["race"]), self.taxonomy)

    def test_reject_unknown_label(self):
        with self.assertRaises(ValueError):
            m.check_decision(decision(True, target_group=["invented"]), self.taxonomy)

    def test_reject_duplicate_labels(self):
        with self.assertRaises(ValueError):
            m.check_decision(
                decision(True, target_group=["race", "race"]), self.taxonomy
            )

    def test_parser_preserves_duplicate_inner_labels(self):
        parser = m.r.e.isolated.adapter(m.ROOT / "config/taxonomy.yaml").bundle.parser
        parsed, _, error = parser.parse_response(
            json.dumps(decision(True, target_group=["race", "race"]))
        )
        self.assertIsNone(error)
        self.assertEqual(parsed.target_group, ["race", "race"])
        m.check_decision(parsed.model_dump(), self.taxonomy, allow_duplicates=True)

    def test_frozen_voter_counts_occurrences(self):
        bundle = m.r.e.isolated.adapter(m.ROOT / "config/taxonomy.yaml").bundle
        votes = [
            decision(True, target_group=["race", "race"]),
            decision(True),
            decision(),
        ]
        runs = [
            bundle.parser.RunResult(
                result=bundle.parser.parse_response(json.dumps(v))[0]
            )
            for v in votes
        ]
        actual = bundle.voter.aggregate(runs).result.model_dump()
        self.assertEqual(actual["target_group"], ["race"])
        self.assertEqual(m.delivered_aggregate(votes), m.gold.decision(actual))
        self.assertEqual(m.gold.independent_aggregate(votes)["target_group"], [])

    def test_duplicate_diagnostic_does_not_replace_delivered_answer(self):
        votes = [
            decision(True, target_group=["race", "race"]),
            decision(True),
            decision(),
        ]
        value = vote_record(votes)
        value["result"]["target_group"] = ["race"]
        m.check_votes(value, self.taxonomy)
        self.assertEqual(
            m.duplicate_fields(votes),
            [{"vote": 0, "field": "target_group", "repeated_labels": {"race": 2}}],
        )
        self.assertEqual(value["result"]["target_group"], ["race"])

    def test_reject_changed_aggregate(self):
        value = vote_record([decision(True), decision(True), decision()])
        value["result"] = decision()
        with self.assertRaises(ValueError):
            m.check_votes(value, self.taxonomy)

    def test_rationale_comes_from_first_gate_agreeing_vote(self):
        value = vote_record(
            [
                decision(reasoning="no"),
                decision(True, reasoning="yes1"),
                decision(True, reasoning="yes2"),
            ]
        )
        self.assertEqual(value["result"]["reasoning"], "yes1")
        m.check_votes(value, self.taxonomy)
        value["result"]["reasoning"] = "no"
        with self.assertRaises(ValueError):
            m.check_votes(value, self.taxonomy)

    def test_network_denied(self):
        with m.r.original.offline(), self.assertRaises(Exception):
            socket.create_connection(("127.0.0.1", 1))

    def test_external_reference_rejected(self):
        with self.assertRaises(Exception):
            m.r.resolve_reference("C:/outside-study/result.json")

    def test_parent_reference_rejected(self):
        with self.assertRaises(Exception):
            m.r.resolve_reference("../outside.json")

    def test_no_empty_comparison(self):
        with self.assertRaises(ValueError):
            m.invalid_ids(["a"], [])


class ReaderChain(unittest.TestCase):
    def setUp(self):
        m.BASE.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="synthetic_", dir=m.BASE)
        self.root = Path(self.temp.name)
        self.store = self.root / "broad"
        self.digest = "frozen-plan"
        self.job = dict(
            key="en/a",
            condition="rag_full",
            repeat=0,
            study="broad",
            messages_sha256="prompt-sha",
        )
        self.stage = dict(id="synthetic", study="broad", jobs=[self.job])
        self.schedule = {"stages": [self.stage]}
        self.identity = lambda j, h: dict(
            key=j["key"],
            condition="broad::" + j["condition"],
            repeat=j["repeat"],
            experiment_sha256=h,
        )
        self.folder = self.store / "items/one"
        self.target = self.store / "stages/synthetic"
        self.folder.mkdir(parents=True)
        self.target.mkdir(parents=True)
        self.value = dict(
            identity=self.identity(self.job, self.digest),
            execution_mode="live",
            messages_sha256="prompt-sha",
            result=None,
        )
        self.b = SimpleNamespace(
            identity=self.identity,
            item_dir=lambda root, job, sha: m.a.storage_path(root) / "items/one",
            verify_review=lambda *args: None,
        )
        self.write_fixture()

    def tearDown(self):
        self.temp.cleanup()

    def put(self, path, value):
        path.write_text(json.dumps(value), encoding="utf8")

    def write_fixture(self):
        self.put(self.folder / "result.json", self.value)
        self.put(
            self.folder / "complete.json",
            {
                "status": "complete",
                "hashes": {
                    "items/one/result.json": m.c.sha(self.folder / "result.json")
                },
            },
        )
        self.put(
            self.target / "complete.json",
            {
                "item_receipt_hashes": {
                    "items/one/complete.json": m.c.sha(self.folder / "complete.json")
                }
            },
        )
        self.put(self.target / "review.json", {"status": "synthetic"})
        self.put(
            self.root / "complete.json",
            {
                "status": "all_studies_executed_and_audited",
                "mode": "live",
                "plan_sha256": self.digest,
                "counts": {"predictions": 1, "nulls": 1},
                "review_hashes": {
                    m.r.reference(self.target / "review.json"): m.c.sha(
                        self.target / "review.json"
                    )
                },
            },
        )

    def read(self):
        return m.original_reader.read_matrix(
            self.b, self.schedule, self.root, self.digest, mode="live", expected_count=1
        )

    def test_original_reader_retains_null(self):
        values, meta = self.read()
        self.assertEqual(values, {("rag_full", 0): {"en/a": None}})
        self.assertEqual(meta["nulls"], 1)
        self.assertEqual(len(meta["result_bindings"]), 2)

    def test_changed_result_rejected(self):
        with (self.folder / "result.json").open("a") as f:
            f.write(" ")
        with self.assertRaises(Exception):
            self.read()

    def test_changed_item_receipt_rejected(self):
        with (self.folder / "complete.json").open("a") as f:
            f.write(" ")
        with self.assertRaises(Exception):
            self.read()

    def test_wrong_prompt_rejected_even_if_resealed(self):
        self.value["messages_sha256"] = "different"
        self.write_fixture()
        with self.assertRaises(Exception):
            self.read()

    def test_wrong_repeat_rejected_even_if_resealed(self):
        self.value["identity"]["repeat"] = 1
        self.write_fixture()
        with self.assertRaises(Exception):
            self.read()

    def test_mock_cannot_be_live(self):
        self.value["execution_mode"] = "mock"
        self.write_fixture()
        with self.assertRaises(Exception):
            self.read()

    def test_extra_review_rejected(self):
        done = m.c.read(self.root / "complete.json")
        done["review_hashes"]["extra"] = "bad"
        self.put(self.root / "complete.json", done)
        with self.assertRaises(Exception):
            self.read()

    def test_unreviewed_stop_rejected(self):
        self.put(self.target / "stop_test.json", {})
        with self.assertRaises(Exception):
            self.read()

    def test_duplicate_scheduled_job_rejected(self):
        self.stage["jobs"].append(self.job.copy())
        with self.assertRaises(Exception):
            self.read()

    def test_missing_receipt_rejected(self):
        (self.folder / "complete.json").unlink()
        with self.assertRaises(Exception):
            self.read()


if __name__ == "__main__":
    unittest.main(verbosity=2)

from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from analysis.feedback.insertion import replay_review as r
from analysis.feedback.insertion import scores as score


def pair():
    a = {
        "key": "synthetic",
        "condition": "reference",
        "text": "Synthetic input",
        "messages": ["old"],
        "retrieved": ["old"],
        "channels": {"bm25": [{"id": "same", "score": 3}]},
        "dense_candidates": [{"id": str(i), "score": 1 - i * 0.01} for i in range(30)],
        "inserted_ids_delivered": [],
        "delivered_gates": [],
    }
    b = deepcopy(a)
    b["dense_candidates"][-1] = {"id": "replacement", "score": 0.70}
    b["messages"] = ["new"]
    b["retrieved"] = ["new"]
    return a, b


def delta(a, b):
    return {
        "key": a["key"],
        "changed_fields": sorted(
            k for k in a.keys() | b.keys() if a.get(k) != b.get(k)
        ),
        "same_messages": a["messages"] == b["messages"],
        "same_retrieved_ids": a["retrieved"] == b["retrieved"],
        "actual_entry": b,
    }


class ReplayReviewTests(unittest.TestCase):
    def test_observed_candidate_swap_accepted_and_original_preserved(self):
        a, b = pair()
        frozen = deepcopy(a)
        change = r.assess_change(a, delta(a, b))
        self.assertEqual(a, frozen)
        self.assertEqual(change["removed_candidates"][0]["id"], "29")
        self.assertFalse(change["same_messages"])

    def test_no_change_returns_none(self):
        a, _ = pair()
        self.assertIsNone(r.assess_change(a, delta(a, a)))

    def test_false_flags_and_hidden_fields_rejected(self):
        a, b = pair()
        d = delta(a, b)
        d["same_messages"] = True
        with self.assertRaises(AssertionError):
            r.assess_change(a, d)
        d = delta(a, b)
        d["changed_fields"] = []
        with self.assertRaises(AssertionError):
            r.assess_change(a, d)

    def test_query_condition_and_label_changes_rejected(self):
        for field, value in [
            ("text", "different query"),
            ("condition", "chosen replay"),
            ("delivered_gates", [True]),
        ]:
            a, b = pair()
            b[field] = value
            with self.assertRaises(AssertionError):
                r.assess_change(a, delta(a, b))

    def test_bm25_change_rejected(self):
        a, b = pair()
        b["channels"]["bm25"][0]["score"] = 4
        with self.assertRaises(AssertionError):
            r.assess_change(a, delta(a, b))

    def test_shared_candidate_score_change_rejected(self):
        a, b = pair()
        b["dense_candidates"][0]["score"] = 0.999
        with self.assertRaises(AssertionError):
            r.assess_change(a, delta(a, b))

    def test_unreviewed_correction_delivery_rejected(self):
        a, b = pair()
        b["inserted_ids_delivered"] = ["correction"]
        with self.assertRaises(AssertionError):
            r.assess_change(a, delta(a, b))

    def test_duplicate_candidate_and_order_only_change_rejected(self):
        a, b = pair()
        b["dense_candidates"][-1] = deepcopy(b["dense_candidates"][0])
        with self.assertRaises(AssertionError):
            r.assess_change(a, delta(a, b))
        a, b = pair()
        b["dense_candidates"] = list(reversed(a["dense_candidates"]))
        with self.assertRaises(AssertionError):
            r.assess_change(a, delta(a, b))

    def test_durable_chunk_content_and_combined_file_must_agree(self):
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            path = folder / "chunk_000000.json"
            roster = [{"key": "a"}]
            values = [{"key": "a", "value": 1}]
            stamp = {"test": True}
            r.c.a.save_new(
                path,
                {
                    "stamp": stamp,
                    "keys": ["a"],
                    "rows": values,
                    "rows_sha256": r.c.sig(values),
                },
            )

            with patch.object(
                r.c, "bindings", lambda paths: {p.name: r.c.sha(p) for p in paths}
            ):
                receipt = {
                    "checkpoint": {"chunks": 1, "hashes": {path.name: r.c.sha(path)}}
                }
                r.check_chunks(folder, roster, values, stamp, receipt)
                with self.assertRaises(AssertionError):
                    r.check_chunks(
                        folder, roster, [{"key": "a", "value": 2}], stamp, receipt
                    )

    def fixture(self):
        conditions = ["reference"] + [f"error_balanced_s{s}" for s in r.c.SEEDS]
        roster = [
            {
                "key": str(i),
                "gold": {"hate": i >= 2},
                "population": "hateful_types" if i >= 2 else "benign",
                "cluster": str(i),
            }
            for i in range(4)
        ]
        results = {}
        for row in roster:
            for cond in conditions:
                for rep in range(3):
                    truth = row["gold"]["hate"]
                    pred = (
                        not truth
                        if cond != "reference" and row["key"] == "0"
                        else truth
                    )
                    results[row["key"], cond, rep] = {
                        "result": {"hate": pred},
                        "n_valid": 3,
                    }
        science = {
            "conditions": conditions,
            "primary": [
                {"name": "false_positive", "gold_hate": False, "bootstrap_seed": 1},
                {"name": "false_negative", "gold_hate": True, "bootstrap_seed": 2},
            ],
        }
        return roster, results, science

    def test_common_exclusion_changes_every_arm_and_primary_equally(self):
        roster, results, science = self.fixture()
        all_scores = score.sensitivity_scores(roster, results, science, [], 100)
        reduced = score.sensitivity_scores(roster, results, science, ["0"], 100)
        self.assertEqual(
            all_scores["primary"]["false_positive/primary"]["difference"], 0.5
        )
        self.assertEqual(reduced["primary"]["false_positive/primary"]["difference"], 0)
        self.assertTrue(all(v["inputs"] == 3 for v in reduced["all_runs"]))
        self.assertEqual(
            reduced["primary"]["false_positive/primary"]["common_ids"], ["1"]
        )

    def test_exclusion_retains_null_and_strict_vote_rules(self):
        roster, results, science = self.fixture()
        cond = science["conditions"][1]
        results["1", cond, 0]["n_valid"] = 2
        out = score.sensitivity_scores(roster, results, science, ["0"], 100)
        self.assertEqual(out["primary"]["false_positive/primary"]["common_n"], 1)
        self.assertEqual(
            out["primary"]["false_positive/all_three_votes"]["common_n"], 0
        )
        results["1", cond, 0]["result"] = None
        out = score.sensitivity_scores(roster, results, science, ["0"], 100)
        self.assertEqual(out["primary"]["false_positive/primary"]["common_n"], 0)
        self.assertEqual(sum(v["nulls"] for v in out["all_runs"]), 1)

    def test_unknown_duplicate_exclusions_rejected(self):
        roster, results, science = self.fixture()
        for keys in [["unknown"], ["0", "0"]]:
            with self.assertRaises(AssertionError):
                score.sensitivity_scores(roster, results, science, keys, 10)

    def test_changed_bound_review_file_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bound.txt"
            path.write_text("before")
            digest = r.c.sha(path)
            path.write_text("after")
            with self.assertRaises(AssertionError):
                r.c.verify({str(path): digest})


if __name__ == "__main__":
    unittest.main()

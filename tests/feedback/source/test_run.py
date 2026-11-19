from tests.feedback.source_checks import syntax

from copy import deepcopy
import math
import re
import time
import unittest
from unittest.mock import patch
import numpy as np
from analysis.feedback.source import run as b

ORIGINAL_PROMPTS = b.preparation.OUT / "baseline_prompts.jsonl"
from analysis.feedback.source import retrieval as r
from tests.common.test_api import reply, QueueTransport, never, MESSAGES


def fixture():
    cat = {}
    for condition in b.CONDITIONS:
        for key, subset in [
            ("en/c", "candidate"),
            ("en/p", "pool"),
            ("en/h", "held_out"),
        ]:
            if condition == "selection_baseline" and subset == "held_out":
                continue
            messages = deepcopy(MESSAGES)
            messages[-1]["text"] += key
            cat[key, condition] = {
                "messages": messages,
                "messages_sha256": b.a.signature(messages),
                "subset": subset,
                "strategy": "hybrid",
            }
    return cat


class BaselineTests(unittest.TestCase):

    def test_stage_engine_exact_inheritance(self):
        old = (b.ROOT / "analysis/knowledge/sampled_banks/run.py").read_text(
            encoding="utf8"
        )
        new = (b.ROOT / "analysis/feedback/source/run.py").read_text(encoding="utf8")
        segment = lambda s: s[
            s.index("def identity(") : s.index("def run_stage(")
        ].strip()
        self.assertEqual(syntax(segment(old)), syntax(segment(new)))

    def test_phase_coverage_order_identity_and_no_holdout_selection(self):
        cat = fixture()
        schedule = b.make_schedule(cat)
        jobs = schedule["stages"][0]["jobs"]
        self.assertEqual(len(jobs), 15)
        self.assertEqual(
            {(j["key"], j["condition"], j["repeat"]) for j in jobs},
            {(k, c, v) for k, c in cat for v in range(3)},
        )
        self.assertEqual(
            [j["phase"] for j in jobs], ["selection"] * 6 + ["reference"] * 9
        )
        self.assertEqual(
            len({b.a.signature(b.identity(j, "fixture")) for j in jobs}), 15
        )
        self.assertEqual(
            cat["en/c", b.CONDITIONS[0]]["messages"],
            cat["en/c", b.CONDITIONS[1]]["messages"],
        )
        self.assertFalse(
            any((j["subset"] == "held_out" and j["phase"] == "selection" for j in jobs))
        )

    def test_holdout_in_selection_or_unknown_phase_refused(self):
        cat = fixture()
        cat["en/h", "selection_baseline"] = cat["en/h", "reference_baseline"]
        with self.assertRaises(b.a.IntegrityError):
            b.make_schedule(cat)
        cat = fixture()
        cat["en/extra", "wrong"] = next(iter(cat.values()))
        with self.assertRaises(b.a.IntegrityError):
            b.make_schedule(cat)
        with self.assertRaises(b.a.IntegrityError):
            b.run_stage("stage_02")

    def test_both_phases_send_fresh_responses_then_restart_sends_zero(self):
        cat = fixture()
        stage = b.make_schedule(cat)["stages"][0]
        root = b.BASE / "offline_check" / ("phase_restart_" + str(time.time_ns()))
        calls = []

        def factory():

            def send(payload):
                calls.append(payload)
                return reply()

            return send

        b.execute_stage(
            stage, cat, root, "fixture", client_factory=factory, mode="mock"
        )
        self.assertEqual(len(calls), 45)
        b.execute_stage(stage, cat, root, "fixture", client_factory=never, mode="mock")
        self.assertEqual(len(calls), 45)
        changed = deepcopy(cat)
        changed["en/c", "selection_baseline"]["messages"][0]["text"] += "changed"
        with self.assertRaises(b.a.IntegrityError):
            b.execute_stage(
                stage, changed, root, "fixture", client_factory=never, mode="mock"
            )

    def test_partial_response_resume_does_not_repeat_remote_call(self):
        cat = fixture()
        stage = b.make_schedule(cat)["stages"][0]
        job = stage["jobs"][0]
        root = b.BASE / "offline_check" / ("partial_" + str(time.time_ns()))
        with patch("src.hsrag.parse.parse_response", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                b.a.execute_item(
                    root,
                    b.identity(job, "fixture"),
                    cat[job["key"], job["condition"]]["messages"],
                    QueueTransport([reply()]),
                )
        calls = []

        def factory():

            def send(payload):
                calls.append(payload)
                return reply()

            return send

        b.execute_stage(
            stage, cat, root, "fixture", client_factory=factory, mode="mock"
        )
        self.assertEqual(len(calls), 44)

    def test_unresolved_intent_stops_without_new_factory_and_remains_stopped(self):
        cat = fixture()
        stage = b.make_schedule(cat)["stages"][0]
        job = stage["jobs"][-1]
        root = b.BASE / "offline_check" / ("unknown_" + str(time.time_ns()))
        with self.assertRaises(KeyboardInterrupt):
            b.a.execute_item(
                root,
                b.identity(job, "fixture"),
                cat[job["key"], job["condition"]]["messages"],
                QueueTransport([KeyboardInterrupt()]),
            )
        factory = unittest.mock.Mock(side_effect=AssertionError("No client allowed"))
        with self.assertRaises(b.a.IncompleteAttempt):
            b.execute_stage(
                stage, cat, root, "fixture", client_factory=factory, mode="mock"
            )
        factory.assert_not_called()
        with self.assertRaises(b.a.IntegrityError):
            b.execute_stage(
                stage, cat, root, "fixture", client_factory=factory, mode="mock"
            )

    def test_mock_cannot_write_live_folder(self):
        cat = fixture()
        stage = b.make_schedule(cat)["stages"][0]
        with self.assertRaises(b.a.IntegrityError):
            b.execute_stage(
                stage,
                cat,
                b.BASE / "live",
                "fixture",
                client_factory=never,
                mode="mock",
            )


class BucketTests(unittest.TestCase):

    def records(self, n=12):
        return [
            {
                "id": f"f{i:02d}",
                "text": (
                    "common alpha alpha"
                    if i == 0
                    else "common beta" if i == 1 else f"common word{i}"
                ),
                "kind": "feedback",
                "lang": "en",
                "meta": {
                    "gate": True,
                    "hate_types": ["explicit"],
                    "target_groups": None,
                    "severity": None,
                    "legal": None,
                },
            }
            for i in range(n)
        ]

    def test_dense_bm25_rrf_against_scalar_independent_formula(self):
        records = self.records()
        x = np.arange(36, dtype=float).reshape(12, 3) + 1
        q = np.array([1.0, 4.0, 2.0])
        bucket = r.FeedbackBucket(records, x)
        chosen, trace = bucket.retrieve("alpha alpha common", q)
        scalar = [
            sum((float(v) * float(z) for v, z in zip(row, q)))
            / (
                math.sqrt(sum((float(v) ** 2 for v in row)))
                * math.sqrt(sum((float(z) ** 2 for z in q)))
            )
            for row in x
        ]
        dense = sorted(range(12), key=lambda i: (-scalar[i], records[i]["id"]))[:10]
        self.assertEqual(
            [z["id"] for z in trace["dense"]], [records[i]["id"] for i in dense]
        )
        for value, i in zip(trace["dense"], dense):
            self.assertAlmostEqual(value["score"], scalar[i], places=14)
        corpus = [re.findall("\\w+", row["text"].lower()) for row in records]
        terms = sorted({w for row in corpus for w in row})
        n = len(corpus)
        avg = sum(map(len, corpus)) / n
        idf = {
            t: math.log(n - sum((t in row for row in corpus)) + 0.5)
            - math.log(sum((t in row for row in corpus)) + 0.5)
            for t in terms
        }
        epsilon = 0.25 * sum(idf.values()) / len(idf)
        idf = {t: max(v, epsilon) if v < 0 else v for t, v in idf.items()}
        bm = [
            sum(
                (
                    idf.get(t, 0)
                    * row.count(t)
                    * 2.5
                    / (row.count(t) + 1.5 * (0.25 + 0.75 * len(row) / avg))
                    for t in ["alpha", "alpha", "common"]
                )
            )
            for row in corpus
        ]
        order = [
            i
            for i in sorted(range(n), key=lambda i: (-bm[i], records[i]["id"]))[:10]
            if bm[i] > 0
        ]
        self.assertEqual(
            [z["id"] for z in trace["bm25"]], [records[i]["id"] for i in order]
        )
        for value, i in zip(trace["bm25"], order):
            self.assertAlmostEqual(value["score"], bm[i], places=14)
        fused = {
            i: (1 / (61 + dense.index(i)) if i in dense else 0)
            + (1 / (61 + order.index(i)) if i in order else 0)
            for i in set(dense) | set(order)
        }
        expected = sorted(fused, key=lambda i: (-fused[i], records[i]["id"]))
        self.assertEqual(
            [z["id"] for z in trace["rrf"]], [records[i]["id"] for i in expected]
        )
        self.assertEqual(chosen["id"], records[expected[0]]["id"])

    def test_ties_permutations_no_lexical_match_and_top10(self):
        records = self.records()
        x = np.ones((12, 3))
        q = np.ones(3)
        chosen, trace = r.FeedbackBucket(records, x).retrieve("absent", q)
        self.assertEqual(chosen["id"], "f00")
        self.assertEqual(trace["bm25"], [])
        self.assertEqual(
            [v["id"] for v in trace["dense"]], [f"f{i:02d}" for i in range(10)]
        )
        perm = list(reversed(range(12)))
        other = r.FeedbackBucket([records[i] for i in perm], x[perm]).retrieve(
            "absent", q
        )
        self.assertEqual((chosen, trace), other)

    def test_nonpositive_bm25_dropped(self):
        records = self.records(2)
        for row in records:
            row["text"] = "same same"
        _, trace = r.FeedbackBucket(records, np.eye(2)).retrieve("same", [1.0, 0.0])
        self.assertEqual(trace["bm25"], [])

    def test_ranking_does_not_use_labels(self):
        records = self.records()
        x = np.arange(36).reshape(12, 3) + 1
        one = r.FeedbackBucket(records, x).retrieve("alpha", [3, 1, 2])[1]
        changed = deepcopy(records)
        for row in changed:
            row["meta"]["hate_types"] = ["irony"]
        two = r.FeedbackBucket(changed, x).retrieve("alpha", [3, 1, 2])[1]
        self.assertEqual(one, two)

    def test_bad_vectors_duplicate_or_oversized_pool_refused(self):
        rows = self.records(2)
        for records, x in [
            (rows, [[0, 0], [1, 1]]),
            (rows, [[1, float("nan")], [1, 1]]),
            (rows + rows, np.ones((4, 2))),
            (self.records(41), np.ones((41, 2))),
        ]:
            with self.assertRaises(AssertionError):
                r.FeedbackBucket(records, x)
        bucket = r.FeedbackBucket(rows, np.eye(2))
        for q in ([0, 0], [1], [float("inf"), 1]):
            with self.assertRaises(AssertionError):
                bucket.retrieve("a", q)

    def test_append_renders_one_missing_annotations_stay_missing(self):
        if not (ORIGINAL_PROMPTS).is_file():
            self.skipTest("Original baseline prompt archive is absent")
        base = b.core.rows(ORIGINAL_PROMPTS)[0]
        old = deepcopy(base)
        rows = self.records(2)
        bucket = r.FeedbackBucket(rows, np.eye(2))
        result = r.updated_prompt(base, bucket, [1, 0])
        self.assertEqual(base, old)
        self.assertEqual(
            result["context"]["example_groups"][:-1], base["context"]["example_groups"]
        )
        self.assertEqual(result["messages"][0], base["messages"][0])
        tail = result["messages"][-1]["text"].split("Additional labelled examples", 1)[
            1
        ]
        self.assertIn("hate_type=[explicit]", tail)
        hit = result["context"]["example_groups"][-1][1][0]
        self.assertTrue(
            all(
                (hit["meta"][k] is None for k in ("target_groups", "severity", "legal"))
            )
        )
        self.assertEqual(len(result["context"]["example_groups"][-1][1]), 1)
        self.assertIn(rows[0]["text"], tail)
        with self.assertRaises(AssertionError):
            r.design.append_feedback(result["context"], rows[0])


if __name__ == "__main__":
    unittest.main()

"""Tests for strategies run."""

from copy import deepcopy
import json
import time
import unittest
from unittest.mock import patch
from analysis.knowledge.strategies import run as b
from tests.common.test_api import reply, QueueTransport, never, MESSAGES


def fixture(n=8):
    dev, cat = ([], {})
    for i in range(n):
        subset = b.ENDPOINT_ORDER[i % len(b.ENDPOINT_ORDER)]
        key = ("de/" if subset.startswith("de_") else "en/") + str(i)
        dev.append({"key": key, "scopes": [{"subset": subset}]})
        for strategy in b.STRATEGIES:
            messages = deepcopy(MESSAGES)
            messages[-1]["text"] += " " + key + " " + strategy
            cat[key, "rag_" + strategy] = {
                "messages": messages,
                "messages_sha256": b.a.signature(messages),
                "strategy": strategy,
                "subset": subset,
            }
    return (dev, cat)


class StrategyTests(unittest.TestCase):

    def test_schedule_balanced_units_cap_coverage_and_rotation(self):
        dev, cat = fixture(17)
        schedule = b.make_schedule(dev, cat, cap=27)
        self.assertEqual(schedule, b.make_schedule(dev, cat, cap=27))
        seen, position = (set(), 0)
        for stage in schedule["stages"]:
            self.assertLessEqual(stage["items"], 27)
            self.assertEqual(stage["ordinary_responses"], stage["items"] * 3)
            for i in range(0, len(stage["jobs"]), 9):
                unit = stage["jobs"][i : i + 9]
                self.assertEqual(len({j["key"] for j in unit}), 1)
                self.assertEqual(
                    {(j["strategy"], j["repeat"]) for j in unit},
                    {(s, r) for s in b.STRATEGIES for r in range(3)},
                )
                self.assertNotIn(unit[0]["key"], seen)
                seen.add(unit[0]["key"])
                for block in range(3):
                    jobs = unit[block * 3 : block * 3 + 3]
                    repeat = (position + block) % 3
                    self.assertEqual([j["repeat"] for j in jobs], [repeat] * 3)
                    self.assertEqual(
                        [j["strategy"] for j in jobs],
                        [b.STRATEGIES[(position + repeat + k) % 3] for k in range(3)],
                    )
                position += 1
        self.assertEqual(seen, {row["key"] for row in dev})

    def test_missing_extra_wrong_scope_and_small_cap_refused(self):
        dev, cat = fixture()
        missing = dict(cat)
        missing.pop(next(iter(missing)))
        extra = dict(cat)
        extra["en/unplanned", "rag_dense"] = next(iter(cat.values()))
        badscope = deepcopy(cat)
        next(iter(badscope.values()))["subset"] = "wrong"
        for bad in (missing, extra, badscope):
            with self.assertRaises(b.a.IntegrityError):
                b.make_schedule(dev, bad)
        with self.assertRaises(b.a.IntegrityError):
            b.make_schedule(dev, cat, cap=8)

    def test_analysis_roles_and_new_three_contrast_family(self):
        plan = b.a.read(b.STUDY / "plan.json")
        out = b.analysis_plan(plan)
        self.assertEqual(out["focal_family_size"], 3)
        self.assertEqual(out["analysis"]["focal_interval_confidence"], 1 - 0.05 / 3)
        for name, row in out["endpoints"].items():
            self.assertEqual(row["role"], plan["analysis_roles"][name])
            self.assertEqual(
                row["scorable_ids"],
                plan["endpoints_inherited_masks_and_labels"][name]["scorable_ids"],
            )
        self.assertEqual(out["endpoints"]["en_dev_eval_sq3_types"]["role"], "focal")
        self.assertNotIn("primary_contrasts", out)

    def test_complete_stage_restart_sends_nothing_and_does_not_advance(self):
        dev, cat = fixture(2)
        stage = b.make_schedule(dev, cat, cap=9)["stages"][0]
        root = b.BASE / "offline_check" / ("unit_restart_" + str(time.time_ns()))
        calls = []

        def factory():

            def send(payload):
                calls.append(payload)
                return reply()

            return send

        b.execute_stage(stage, cat, root, "mock-plan", client_factory=factory, mode="mock")
        self.assertEqual(len(calls), 27)
        b.execute_stage(stage, cat, root, "mock-plan", client_factory=never, mode="mock")
        self.assertEqual(len(calls), 27)
        self.assertFalse((root / "stages/stage_02").exists())
        changed = deepcopy(cat)
        changed[stage["jobs"][0]["key"], stage["jobs"][0]["condition"]]["messages"][0][
            "text"
        ] += " changed"
        with self.assertRaises(b.a.IntegrityError):
            b.execute_stage(stage, changed, root, "mock-plan", client_factory=never, mode="mock")

    def test_partial_saved_replies_resume_without_repeating(self):
        dev, cat = fixture(1)
        stage = b.make_schedule(dev, cat)["stages"][0]
        root = b.BASE / "offline_check" / ("unit_partial_" + str(time.time_ns()))
        job = stage["jobs"][0]
        entry = cat[job["key"], job["condition"]]
        first = QueueTransport([reply()])
        with patch("src.hsrag.parse.parse_response", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                b.a.execute_item(root, b.identity(job, "mock-plan"), entry["messages"], first)
        existing, todo = b.scan_jobs(stage["jobs"], cat, root, "mock-plan", "mock")
        self.assertEqual(len(todo), 9)
        self.assertEqual(existing, [])
        calls = []

        def factory():

            def send(payload):
                calls.append(payload)
                return reply()

            return send

        b.execute_stage(stage, cat, root, "mock-plan", client_factory=factory, mode="mock")
        self.assertEqual(len(calls), 26)
        b.check_stage_complete(stage, cat, root, "mock-plan", "mock")

    def test_unresolved_intent_blocks_entire_stage_before_new_requests(self):
        dev, cat = fixture(1)
        stage = b.make_schedule(dev, cat)["stages"][0]
        root = b.BASE / "offline_check" / ("unit_unknown_" + str(time.time_ns()))
        job = stage["jobs"][-1]
        entry = cat[job["key"], job["condition"]]
        with self.assertRaises(KeyboardInterrupt):
            b.a.execute_item(
                root,
                b.identity(job, "mock-plan"),
                entry["messages"],
                QueueTransport([KeyboardInterrupt()]),
            )
        factory = unittest.mock.Mock(side_effect=AssertionError("Must not create another client"))
        with self.assertRaises(b.a.IncompleteAttempt):
            b.execute_stage(stage, cat, root, "mock-plan", client_factory=factory, mode="mock")
        factory.assert_not_called()
        self.assertEqual(len(list((root / "stages/stage_01").glob("stop_*.json"))), 1)
        with self.assertRaises(b.a.IntegrityError):
            b.execute_stage(stage, cat, root, "mock-plan", client_factory=factory, mode="mock")

    def test_model_unavailable_records_response_without_post(self):
        import requests

        attempt = b.BASE / "offline_check" / ("unit_model_" + str(time.time_ns()))
        client = b.a.RequestsTransport(
            "https://openwebui.srv.peasec.de/api/v1", "fixture-never-send"
        )
        response = unittest.mock.Mock(status_code=200, content=b'{"data":[]}')
        response.json.return_value = {"data": []}
        with patch.object(requests, "get", return_value=response) as get, patch.object(
            requests, "post", side_effect=AssertionError
        ) as post:
            with self.assertRaises(b.a.IntegrityError):
                b.live_availability(attempt, client)
        self.assertEqual(get.call_count, 1)
        post.assert_not_called()
        self.assertFalse(get.call_args.kwargs["allow_redirects"])
        self.assertEqual(b.a.read(attempt / "model_check_response.json")["status"], 200)
        self.assertNotIn("fixture-never-send", (attempt / "model_check_response.json").read_text())

    def test_mode_boundary_and_previous_review_required(self):
        dev, cat = fixture(2)
        schedule = b.make_schedule(dev, cat, cap=9)
        with self.assertRaises(b.a.IntegrityError):
            b.execute_stage(
                schedule["stages"][0],
                cat,
                b.BASE / "live",
                "mock-plan",
                client_factory=never,
                mode="mock",
            )
        with self.assertRaises(b.a.IntegrityError):
            b.previous_review(1, schedule, "mock-plan")
        b.previous_review(0, schedule, "mock-plan")


if __name__ == "__main__":
    unittest.main()

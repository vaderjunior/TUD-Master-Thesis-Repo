from tests.feedback.source_checks import syntax

from copy import deepcopy
import time
import unittest
from unittest.mock import patch
from analysis.feedback.source import updates as b
from tests.common.test_api import MESSAGES, reply, QueueTransport, never


def fixture(n=3):
    cat = {}
    for i in range(n):
        for condition in b.CONDITIONS:
            arm, seed, level = condition.split("_")
            size = int(level[1:])
            messages = deepcopy(MESSAGES)
            messages[-1]["text"] += str(i)
            cat[f"en/{i}", condition] = {
                "messages": messages,
                "messages_sha256": b.a.signature(messages),
                "strategy": "hybrid",
                "subset": b.ENDPOINT_ORDER[i % 3],
                "arm": arm,
                "seed": int(seed[1:]),
                "budget_level": size,
                "pool_size": 10 * size,
            }
    return cat


class UpdateTests(unittest.TestCase):

    def test_exact_inherited_engine(self):
        old = (b.ROOT / "analysis/feedback/source/run.py").read_text(encoding="utf8")
        new = (b.ROOT / "analysis/feedback/source/updates.py").read_text(
            encoding="utf8"
        )
        segment = lambda text: text[
            text.index("def identity(") : text.index("def run_stage(")
        ].strip()
        self.assertEqual(syntax(segment(old)), syntax(segment(new)))

    def test_schedule_complete_unique_balanced_and_rotating(self):
        cat = fixture(12)
        schedule = b.make_schedule(cat)
        jobs = [j for s in schedule["stages"] for j in s["jobs"]]
        self.assertEqual(len(jobs), 12 * 24 * 3)
        self.assertEqual(
            {(j["key"], j["condition"], j["repeat"]) for j in jobs},
            {(k, c, r) for k, c in cat for r in range(3)},
        )
        self.assertEqual(
            len({b.a.signature(b.identity(j, "fixture")) for j in jobs}), len(jobs)
        )
        self.assertEqual([s["items"] for s in schedule["stages"]], [432, 216, 216])
        for stage in schedule["stages"]:
            conditions = [c for c in b.CONDITIONS if int(c[-2:]) in stage["levels"]]
            width = len(conditions) * 3
            for ordinal in range(12):
                unit = stage["jobs"][ordinal * width : (ordinal + 1) * width]
                self.assertEqual(len({j["key"] for j in unit}), 1)
                for block in range(3):
                    group = unit[
                        block * len(conditions) : (block + 1) * len(conditions)
                    ]
                    rep = (ordinal + block) % 3
                    self.assertEqual(
                        [j["repeat"] for j in group], [rep] * len(conditions)
                    )
                    self.assertEqual(
                        [j["condition"] for j in group],
                        [
                            conditions[(ordinal + rep + i) % len(conditions)]
                            for i in range(len(conditions))
                        ],
                    )

    def test_missing_wrong_or_extra_setting_refused(self):
        cat = fixture()
        cat.pop(next(iter(cat)))
        with self.assertRaises(b.a.IntegrityError):
            b.make_schedule(cat)
        cat = fixture()
        next(iter(cat.values()))["pool_size"] = 99
        with self.assertRaises(b.a.IntegrityError):
            b.make_schedule(cat)
        cat = fixture()
        cat["en/0", "unknown"] = next(iter(cat.values()))
        with self.assertRaises(b.a.IntegrityError):
            b.make_schedule(cat)

    def test_progress_does_not_call_all_budgets_complete_after_first(self):
        schedule = b.make_schedule(fixture())
        for row in b.endpoint_progress(schedule, 0).values():
            self.assertEqual(row["completed_inputs"], row["total_inputs"])
            self.assertFalse(row["complete"])
            self.assertEqual(
                (row["completed_predictions"], row["total_predictions"]), (36, 72)
            )
        self.assertTrue(
            all((r["complete"] for r in b.endpoint_progress(schedule, 2).values()))
        )

    def test_sequence_runs_audits_verifies_before_advancing(self):
        events = []
        stages = ["stage_02", "stage_03", "stage_04"]
        b.run_sequence(
            stages,
            lambda s: events.append(("run", s)),
            lambda s: events.append(("audit", s)),
            lambda s: False,
            lambda s: events.append(("verify", s)),
        )
        self.assertEqual(
            events,
            [(action, s) for s in stages for action in ("run", "audit", "verify")],
        )

    def test_failed_execution_or_audit_or_receipt_prevents_next_stage(self):
        for failing in ("run", "audit", "verify"):
            events = []

            def action(kind, s):
                events.append((kind, s))
                if kind == failing:
                    raise RuntimeError("fixture stop")

            with self.subTest(failing=failing), self.assertRaises(RuntimeError):
                b.run_sequence(
                    ["stage_02", "stage_03"],
                    lambda s: action("run", s),
                    lambda s: action("audit", s),
                    lambda s: False,
                    lambda s: action("verify", s),
                )
            self.assertFalse(any((s == "stage_03" for _, s in events)))

    def test_existing_audit_is_verified_without_republication(self):
        events = []
        b.run_sequence(
            ["stage_02"],
            lambda s: events.append("run"),
            lambda s: self.fail("Do not republish"),
            lambda s: True,
            lambda s: events.append("verify"),
        )
        self.assertEqual(events, ["run", "verify"])

    def test_completed_stage_restart_reuses_identical_prompts_across_conditions(self):
        cat = fixture(1)
        stage = b.make_schedule(cat)["stages"][0]
        root = b.BASE / "offline_check" / ("restart_" + str(time.time_ns()))
        calls = []

        def factory():

            def send(payload):
                calls.append(payload)
                return reply()

            return send

        b.execute_stage(
            stage, cat, root, "fixture", client_factory=factory, mode="mock"
        )
        self.assertEqual(len(calls), 108)
        b.execute_stage(stage, cat, root, "fixture", client_factory=never, mode="mock")
        self.assertEqual(len(calls), 108)

    def test_partial_reply_resume_and_unknown_intent(self):
        cat = fixture(1)
        stage = b.make_schedule(cat)["stages"][1]
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
        self.assertEqual(len(calls), 53)
        blocked = b.BASE / "offline_check" / ("unknown_" + str(time.time_ns()))
        with self.assertRaises(KeyboardInterrupt):
            b.a.execute_item(
                blocked,
                b.identity(job, "fixture"),
                cat[job["key"], job["condition"]]["messages"],
                QueueTransport([KeyboardInterrupt()]),
            )
        with self.assertRaises(b.a.IncompleteAttempt):
            b.execute_stage(
                stage, cat, blocked, "fixture", client_factory=never, mode="mock"
            )

    def test_prior_review_missing_or_changed_refused(self):
        schedule = b.make_schedule(fixture())
        temporary = b.BASE / "offline_check" / ("review_" + str(time.time_ns()))
        with patch.object(b, "BASE", temporary):
            with self.assertRaises(b.a.IntegrityError):
                b.previous_review(1, schedule, "fixture")
            b.save(
                temporary / "stage_reviews/stage_02.json",
                {"status": "audited", "plan_sha256": "wrong", "stage": "stage_02"},
            )
            with self.assertRaises(b.a.IntegrityError):
                b.previous_review(1, schedule, "fixture")


if __name__ == "__main__":
    unittest.main()

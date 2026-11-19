from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import uuid
from analysis.common import runtime as c
from analysis.feedback.rounds import policy as policy
from analysis.feedback.rounds import run as m


class Sequence(unittest.TestCase):

    def harness(self, fail=None):
        root = c.ITERATIVE / "offline_check" / ("sequence_" + uuid.uuid4().hex)
        rows = [
            {"id": str(i), "key": "fixture/" + str(i), "gold": {"hate": bool(i % 2)}}
            for i in range(1024)
        ]
        evaluation = [
            {
                "id": "eval" + str(i),
                "key": "evaluation/" + str(i),
                "population": "fixture",
            }
            for i in range(2077)
        ]
        c.save(root / "plan.json", {"synthetic_only": True})
        c.save(
            root / "source_blocks.json",
            {"blocks": {str(s): policy.partition(rows, s) for s in c.STREAM_SEEDS}},
        )
        visits = []
        transports = []

        def make(path, study, rr, pools, deps, data):
            return SimpleNamespace(
                BASE=path / "execution", rows=rr, pools=pools, study=study
            )

        def execute(b):
            visits.append(b.BASE)
            if fail and fail in str(b.BASE):
                raise RuntimeError("synthetic forced stop")
            if (b.BASE / "live/complete.json").exists():
                return
            transports.append(b.BASE)
            n = len(b.rows) * len(b.pools) * 3
            c.save(
                b.BASE / "live/complete.json",
                {
                    "synthetic_only": True,
                    "counts": {"predictions": n},
                    "accounting": {"ordinary_logical_calls": n * 3},
                },
            )

        def extract(b):
            self.assertEqual(b.study, "source")
            return ({r["id"]: [not r["gold"]["hate"]] * 3 for r in b.rows}, {})

        stack = ExitStack()
        for name, value in [
            ("BASE", root),
            ("validate", lambda: None),
            ("require_a16_finished", lambda: None),
            ("load_data", lambda: (rows, None, evaluation, None, None, None, None)),
            ("make_phase", make),
            ("ready_phase", lambda _: None),
            ("extract", extract),
        ]:
            stack.enter_context(patch.object(m, name, value))
        stack.enter_context(patch.object(m.q, "require_a15_finished", lambda: None))
        stack.enter_context(patch.object(m.q, "execute", execute))
        return (root, visits, transports, stack)

    def test_all_rounds_balanced_pools_and_exact_budget(self):
        root, visits, transports, stack = self.harness()
        with stack:
            m.run()
        self.assertEqual(len(visits), 14)
        self.assertEqual(len(transports), 14)
        d = c.read(root / "complete.json")
        self.assertEqual(d["totals"], {"ordinary_calls": 777672, "predictions": 259224})
        for n in range(1, 5):
            for seed in c.STREAM_SEEDS:
                d = c.read(root / f"rounds/r{n}/selection_s{seed}.json")
                self.assertTrue(
                    all((len(v) == n * 10 for v in d["cumulative"].values()))
                )
                self.assertFalse(d["evaluation_outcomes_read"])

    def test_restart_reuses_all_phases(self):
        root, visits, transports, stack = self.harness()
        with stack:
            m.run()
            m.run()
        self.assertEqual(len(visits), 28)
        self.assertEqual(len(transports), 14)

    def test_failed_source_stops_before_future_evaluation(self):
        root, visits, transports, stack = self.harness("r2_source")
        with stack, self.assertRaisesRegex(RuntimeError, "synthetic forced stop"):
            m.run()
        self.assertFalse((root / "complete.json").exists())
        self.assertFalse(any(("r2_evaluation" in str(p) for p in visits)))


if __name__ == "__main__":
    with c.offline():
        unittest.main()

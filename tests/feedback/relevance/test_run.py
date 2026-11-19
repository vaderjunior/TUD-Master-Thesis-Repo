import unittest
from analysis.feedback.relevance import run as b


class RelevanceAPITests(unittest.TestCase):
    def fixture_catalog(self):
        if not b.SCIENCE.is_file():
            self.skipTest("Original relevance roster and scientific plan are absent")
        science = b.a.read(b.SCIENCE)
        return {
            ("en/" + i, c): {
                "messages_sha256": "a" * 64,
                "strategy": "hybrid",
                "subset": "pool",
                "policy": c.rsplit("_s", 1)[0] if c != "reference" else c,
                "pool_seed": int(c.rsplit("_s", 1)[1]) if c != "reference" else None,
            }
            for i in science["rosters"]["evaluation"]
            for c in b.CONDITIONS
        }

    def test_schedule_complete_and_reference_not_triplicated(self):
        schedule = b.make_schedule(self.fixture_catalog())
        self.assertEqual([s["items"] for s in schedule["stages"]], [5616, 5616, 5568])
        jobs = [j for s in schedule["stages"] for j in s["jobs"]]
        self.assertEqual(
            len({(j["key"], j["condition"], j["repeat"]) for j in jobs}), 16800
        )
        self.assertEqual(sum(j["condition"] == "reference" for j in jobs), 1050)
        sets = [{j["key"] for j in s["jobs"]} for s in schedule["stages"]]
        self.assertTrue(all(not sets[i] & sets[j] for i in range(3) for j in range(i)))

    def test_missing_prompt_rejected(self):
        cat = self.fixture_catalog()
        cat.pop(next(iter(cat)))
        with self.assertRaises(b.a.IntegrityError):
            b.make_schedule(cat)

    def test_plan_condition_and_repeat_change_identity(self):
        job = {"key": "en/example", "condition": "reference", "repeat": 0}
        identities = [
            b.a.signature(b.identity(job, "plan-a")),
            b.a.signature(b.identity(job, "plan-b")),
            b.a.signature(
                b.identity({**job, "condition": "natural_s202609281"}, "plan-a")
            ),
            b.a.signature(b.identity({**job, "repeat": 1}, "plan-a")),
        ]
        self.assertEqual(len(set(identities)), 4)

    def test_failed_audit_blocks_next_batch(self):
        log = []

        def audit(stage):
            log.append(("audit", stage))
            raise RuntimeError("fixture audit failure")

        with self.assertRaises(RuntimeError):
            b.run_sequence(
                ["a", "b"],
                lambda s: log.append(("run", s)),
                audit,
                lambda s: False,
                lambda s: log.append(("verify", s)),
            )
        self.assertEqual(log, [("run", "a"), ("audit", "a")])

    def test_failed_receipt_check_blocks_next_batch(self):
        log = []

        def verify(stage):
            log.append(("verify", stage))
            raise RuntimeError("fixture altered review")

        with self.assertRaises(RuntimeError):
            b.run_sequence(
                ["a", "b"],
                lambda s: log.append(("run", s)),
                lambda s: None,
                lambda s: True,
                verify,
            )
        self.assertEqual(log, [("run", "a"), ("verify", "a")])


if __name__ == "__main__":
    unittest.main()

from copy import deepcopy
import unittest
import uuid
from analysis.common import runtime as c
from analysis.feedback.common import queue as q


class QueueTests(unittest.TestCase):

    def fixture(self, n=4, conditions=("reference", "correction")):
        root = c.SHARED / "offline_check" / ("queue_" + uuid.uuid4().hex)
        b = q.queue(
            root,
            "fixture",
            [],
            {
                "inputs": n,
                "conditions": len(conditions),
                "batches": 2,
                "predictions": n * len(conditions) * 3,
            },
        )
        cat = {}
        schema = "config/taxonomy.yaml"
        for i in range(n):
            for cond in conditions:
                messages = [
                    {"role": "system", "text": "OFFLINE fixture only"},
                    {"role": "user", "text": f"Synthetic item {i}"},
                ]
                cat[str(i), cond] = {
                    "messages": messages,
                    "messages_sha256": b.a.signature(messages),
                    "schema_path": schema,
                    "schema_sha256": c.sha(c.ROOT / schema),
                    "population": "benign" if i % 2 else "hateful",
                    "entry_sha256": "fixture",
                }
        schedule = b.make_schedule({"fixture": cat})
        return (b, cat, schedule, b.BASE / "offline_check" / "run" / "fixture")

    def test_full_matrix_and_single_batch_pairing(self):
        b, cat, s, root = self.fixture()
        value = q.validate_schedule(b, s, {"fixture": cat})
        self.assertEqual(value["predictions"], 24)

    def test_missing_job_rejected(self):
        b, cat, s, root = self.fixture()
        s["stages"][0]["jobs"].pop()
        with self.assertRaises(RuntimeError):
            q.validate_schedule(b, s, {"fixture": cat})

    def test_duplicate_repeat_rejected(self):
        b, cat, s, root = self.fixture()
        s["stages"][0]["jobs"][2] = deepcopy(s["stages"][0]["jobs"][0])
        with self.assertRaises(RuntimeError):
            q.validate_schedule(b, s, {"fixture": cat})

    def test_wrong_payload_rejected(self):
        b, cat, s, root = self.fixture()
        s["stages"][0]["jobs"][0]["messages_sha256"] = "bad"
        with self.assertRaises(RuntimeError):
            q.validate_schedule(b, s, {"fixture": cat})

    def test_namespace_cannot_write_active_study(self):
        b, cat, s, root = self.fixture()
        with self.assertRaises(b.a.IntegrityError):
            b.save(c.A15 / "forbidden.json", {})
        with self.assertRaises(RuntimeError):
            q.queue(
                c.A15,
                "forbidden",
                [],
                {"inputs": 1, "conditions": 1, "batches": 1, "predictions": 3},
            )

    def test_mock_full_audit_and_call_free_resume(self):
        b, cat, s, root = self.fixture()
        transport = q.SyntheticTransport()
        for stage in s["stages"]:
            b.execute_stage(
                stage,
                cat,
                root,
                "fixture-plan",
                mode="mock",
                client_factory=lambda: transport,
                sleep=lambda _: None,
            )
            b.audit.audit_stage(stage, cat, root, "fixture-plan", mode="mock")
        for stage in s["stages"]:
            b.execute_stage(
                stage,
                cat,
                root,
                "fixture-plan",
                mode="mock",
                client_factory=c.forbidden,
            )
            b.verify_review(stage, root, "fixture-plan", "mock")
        self.assertEqual(transport.count, 72)

    def test_unknown_intent_stops_before_transport(self):
        b, cat, s, root = self.fixture()
        job = s["stages"][0]["jobs"][0]
        engine = b.engines(cat)[job["schema_path"]]
        folder = b.item_dir(root, job, "fixture-plan")
        stamp = {
            "identity": b.identity(job, "fixture-plan"),
            "messages_sha256": job["messages_sha256"],
            "contract": b.a.CONTRACT,
            "implementation": engine.implementation(),
            "execution_mode": "mock",
        }
        b.a.save_new(folder / "job.json", stamp)
        b.a.save_new(
            folder / "calls/vote_0_turn_0/attempt_0/intent.json", {"unknown": True}
        )
        with self.assertRaises(b.a.IncompleteAttempt):
            b.scan_jobs([job], cat, root, "fixture-plan", "mock", b.engines(cat))

    def test_identical_messages_do_not_share_outcomes(self):
        b, cat, s, root = self.fixture()
        job = s["stages"][0]["jobs"][0]
        other = dict(job, condition="other")
        self.assertNotEqual(
            b.item_dir(root, job, "fixture-plan"),
            b.item_dir(root, other, "fixture-plan"),
        )

    def test_failure_prevents_following_batch(self):
        b, cat, s, root = self.fixture()
        visited = []

        def fail(stage):
            visited.append(stage["id"])
            raise RuntimeError("fixture failure")

        with self.assertRaises(RuntimeError):
            b.run_sequence(
                s["stages"], fail, lambda _: None, lambda _: False, lambda _: None
            )
        self.assertEqual(len(visited), 1)


if __name__ == "__main__":
    with c.offline():
        unittest.main()

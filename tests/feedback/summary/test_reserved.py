from copy import deepcopy
import unittest
import uuid
from analysis.feedback.reserved import common as c
from analysis.feedback.reserved import validate as check
from analysis.common import api as a


def context():

    def hit(i):
        return {
            "id": f"base-{i}",
            "text": f"Synthetic example {i}",
            "meta": {
                "gate": False,
                "hate_types": None,
                "target_groups": None,
                "severity": None,
                "legal": None,
            },
            "via": "rrf",
        }

    return {
        "definitions": [{"id": "d1"}, {"id": "d2"}],
        "guidelines": [{"id": "g1"}, {"id": "g2"}],
        "example_groups": [["Labelled examples", [hit(i) for i in range(5)]]],
    }


def correction():
    return {
        "id": "correction",
        "text": "Synthetic correction",
        "meta": {
            "gate": True,
            "hate_types": None,
            "target_groups": None,
            "severity": None,
            "legal": None,
        },
        "via": "rrf",
    }


def synthetic_catalog():
    pools = [
        {"condition": f"{selection}_{polarity}_s{seed}"}
        for seed in c.SEEDS
        for selection in ("error", "correct", "random")
        for polarity in ("positive", "balanced")
    ]
    messages = [
        {"role": "system", "text": "OFFLINE SYNTHETIC TEST ONLY"},
        {"role": "user", "text": "Synthetic input; no scientific outcome."},
    ]
    entry = {
        "messages": messages,
        "messages_sha256": a.signature(messages),
        "schema_path": "config/taxonomy.yaml",
        "schema_sha256": c.sha(c.ROOT / "config/taxonomy.yaml"),
        "entry_sha256": "offline-test",
    }
    return {
        (f"synthetic-{i:04d}", cond): {
            **entry,
            "population": "benign" if i < 208 else "hateful_types",
        }
        for i in range(c.N_INPUTS)
        for cond in c.condition_list(pools)
    }


class ReservedTests(unittest.TestCase):

    def test_exact_five_four_six_budgets_and_originals_preserved(self):
        base = context()
        source = correction()
        before = deepcopy((base, source))
        for mode, n in [
            ("reference5", 5),
            ("reference4", 4),
            ("reserved5", 5),
            ("extended6", 6),
            ("masked5", 5),
        ]:
            out = c.intervene(
                base, mode, None if mode.startswith("reference") else source
            )
            self.assertEqual(len(c.examples(out)), n)
            self.assertEqual(out["definitions"], base["definitions"])
            self.assertEqual(out["guidelines"], base["guidelines"])
            self.assertEqual(c.examples(out)[:4], c.examples(base)[:4])
        self.assertEqual((base, source), before)

    def test_binary_label_is_only_masked_difference(self):
        base = context()
        source = correction()
        labelled = c.intervene(base, "reserved5", source)
        masked = c.intervene(base, "masked5", source)
        altered = deepcopy(labelled)
        altered["example_groups"][0][1][-1]["meta"]["gate"] = None
        self.assertEqual(altered, masked)
        self.assertEqual(
            check.renderer.corrected_gold(c.examples(masked)[-1]["meta"]), ""
        )
        self.assertEqual(
            check.renderer.corrected_gold(c.examples(labelled)[-1]["meta"]), "hate=true"
        )

    def test_reject_fabricated_annotation_even_empty_list(self):
        for field in ("hate_types", "target_groups", "severity", "legal"):
            record = correction()
            record["meta"][field] = []
            with self.assertRaises(AssertionError):
                c.intervene(context(), "reserved5", record)

    def test_reject_missing_or_integer_gate(self):
        for value in (None, 0, 1, "false"):
            record = correction()
            record["meta"]["gate"] = value
            with self.assertRaises(AssertionError):
                c.intervene(context(), "reserved5", record)

    def test_reject_duplicate_source_or_extra_correction(self):
        record = correction()
        record["id"] = "base-0"
        with self.assertRaises(AssertionError):
            c.intervene(context(), "reserved5", record)
        six = c.intervene(context(), "extended6", correction())
        with self.assertRaises(AssertionError):
            c.intervene(six, "reserved5", correction())

    def test_independent_transform_agrees_for_all_policies(self):
        for gate in (False, True):
            record = correction()
            record["meta"]["gate"] = gate
            for mode in (
                "reference5",
                "reference4",
                "reserved5",
                "extended6",
                "masked5",
            ):
                self.assertEqual(
                    c.intervene(
                        context(),
                        mode,
                        None if mode.startswith("reference") else record,
                    ),
                    check.expected_context(context(), mode, record),
                )

    def test_extended_keeps_removed_neighbour_and_same_source(self):
        five = c.intervene(context(), "reserved5", correction())
        six = c.intervene(context(), "extended6", correction())
        self.assertEqual(c.examples(five)[-1], c.examples(six)[-1])
        self.assertEqual(c.examples(six)[4], c.examples(context())[4])

    def test_write_scope_rejects_parent_artifacts(self):
        with self.assertRaises(AssertionError):
            c.save(c.A14 / "should_never_exist.json", {})

    def test_atomic_jsonl_resume_preserves_identical_and_refuses_changed(self):
        path = (
            c.BASE
            / "execution/offline_check"
            / ("io_" + uuid.uuid4().hex)
            / "rows.jsonl"
        )
        c.jsonlines(path, [{"v": 1}])
        c.jsonlines(path, [{"v": 1}])
        with self.assertRaises(AssertionError):
            c.jsonlines(path, [{"v": 2}])
        self.assertEqual(c.rows(path), [{"v": 1}])

    def test_queue_namespace_is_isolated(self):
        from analysis.feedback.reserved import run as api
        from analysis.common import vote_api as old

        before = old.BASE
        b = api.queue()
        self.assertEqual(old.BASE, before)
        self.assertEqual(b.execute_stage.__globals__["BASE"], api.BASE)
        self.assertEqual(b.audit.b.BASE, api.BASE)
        self.assertNotEqual(api.BASE, before)

    def test_full_matrix_and_mutated_repeat_rejected(self):
        from analysis.feedback.reserved import run as api

        b = api.queue()
        cat = synthetic_catalog()
        schedule = b.make_schedule({api.STUDY: cat})
        receipt = check.check_schedule(schedule, {api.STUDY: cat}, b)
        self.assertEqual(receipt["ordinary_calls"], 449550)
        self.assertEqual(receipt["predictions"], 149850)
        schedule["stages"][0]["jobs"][0]["repeat"] = 9
        with self.assertRaises(AssertionError):
            check.check_schedule(schedule, {api.STUDY: cat}, b)

    def one_job(self):
        from analysis.feedback.reserved import run as api

        b = api.queue()
        messages = [
            {"role": "system", "text": "OFFLINE ONLY"},
            {"role": "user", "text": "Synthetic fixture"},
        ]
        entry = {
            "messages": messages,
            "messages_sha256": a.signature(messages),
            "schema_path": "config/taxonomy.yaml",
            "schema_sha256": c.sha(c.ROOT / "config/taxonomy.yaml"),
            "population": "benign",
            "entry_sha256": "fixture",
        }
        job = {
            "study": api.STUDY,
            "key": "fixture",
            "condition": "reference5",
            "repeat": 0,
            **{
                k: entry[k]
                for k in (
                    "messages_sha256",
                    "schema_path",
                    "schema_sha256",
                    "population",
                )
            },
        }
        stage = {
            "id": "fixture_stage",
            "study": api.STUDY,
            "input_keys": ["fixture"],
            "jobs": [job],
            "items": 1,
            "ordinary_calls": 3,
        }
        root = api.BASE / "offline_check" / ("single_" + uuid.uuid4().hex) / api.STUDY
        return (b, {("fixture", "reference5"): entry}, stage, root)

    def test_completed_mock_audits_and_restarts_without_calls(self):
        b, cat, stage, root = self.one_job()
        transport = check.SyntheticTransport()
        b.execute_stage(
            stage,
            cat,
            root,
            "test-plan",
            mode="mock",
            client_factory=lambda: transport,
            sleep=lambda _: None,
        )
        b.audit.audit_stage(stage, cat, root, "test-plan", mode="mock")
        b.execute_stage(
            stage, cat, root, "test-plan", mode="mock", client_factory=c.forbidden
        )
        self.assertEqual(transport.count, 3)
        with self.assertRaises(a.IntegrityError):
            b.checked_root(c.A14, stage["study"], "mock")

    def test_unknown_request_refuses_blind_resend(self):
        b, cat, stage, root = self.one_job()
        job = stage["jobs"][0]
        engine = b.engines(cat)[job["schema_path"]]
        folder = b.item_dir(root, job, "test-plan")
        stamp = {
            "identity": b.identity(job, "test-plan"),
            "messages_sha256": job["messages_sha256"],
            "contract": a.CONTRACT,
            "implementation": engine.implementation(),
            "execution_mode": "mock",
        }
        a.save_new(folder / "job.json", stamp)
        a.save_new(
            folder / "calls/vote_0_turn_0/attempt_0/intent.json", {"unknown": True}
        )
        with self.assertRaises(a.IncompleteAttempt):
            b.scan_jobs(stage["jobs"], cat, root, "test-plan", "mock", b.engines(cat))

    def test_duplicate_content_retains_distinct_condition_identity(self):
        b, cat, stage, root = self.one_job()
        job = stage["jobs"][0]
        other = {**job, "condition": "reference4"}
        self.assertNotEqual(
            b.item_dir(root, job, "test-plan"), b.item_dir(root, other, "test-plan")
        )

    def test_failure_stops_following_stages(self):
        from analysis.feedback.reserved import run as api

        calls = []

        def fail(stage):
            calls.append(stage)
            raise RuntimeError("fixture failure")

        with self.assertRaises(RuntimeError):
            api.queue().run_sequence(
                [1, 2], fail, lambda _: None, lambda _: False, lambda _: None
            )
        self.assertEqual(calls, [1])


if __name__ == "__main__":
    with c.offline():
        unittest.main()

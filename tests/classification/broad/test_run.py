from copy import deepcopy
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import socket
from types import SimpleNamespace
import time
import unittest
from unittest.mock import patch
from analysis.classification.broad import run as r
from tests.common import test_execution as inherited
from tests.feedback.summary import test_balanced as fixtures

a, e, c = (r.a, r.e, r.c)
TEST_ROOT = r.BASE / "offline_check" / ("storage_tests_" + str(time.time_ns()))
AMENDMENT = dict(
    path=r.reference(r.PROTOCOL),
    sha256=c.sha(r.PROTOCOL) if r.PROTOCOL.exists() else "0" * 64,
    synthetic=True,
)


class StorageTests(unittest.TestCase):

    def setUp(self):
        self.base = TEST_ROOT / self._testMethodName
        r.BASE.mkdir(parents=True, exist_ok=True)
        if not r.LAYOUT.exists():
            r.save(r.LAYOUT, r.layout_record())
        deadline = patch.object(
            r.original, "DEADLINE", datetime.now().astimezone() + timedelta(hours=1)
        )
        deadline.start()
        self.addCleanup(deadline.stop)

    @unittest.skipUnless(
        Path(
            "D:/Thesis_Experiments/sq2_broad_20261030_v1/storage_layout.json"
        ).is_file(),
        "Original D: volume and storage-layout archive are absent",
    )
    def test_volume_marker_and_no_C_live(self):
        r.check_layout()
        self.assertEqual(a.read(r.LAYOUT), r.layout_record())
        self.assertNotEqual(r.BASE.stat().st_dev, c.ROOT.stat().st_dev)
        self.assertFalse((r.SOURCE_BASE / "live").exists())

    def test_actual_hard_link_and_atomic_reopen_on_D(self):
        p = self.base / "original.json"
        r.save(p, {"fixture": "Grüße"})
        other = self.base / "hardlink.json"
        os.link(p, other)
        self.assertTrue(os.path.samefile(p, other))
        self.assertEqual(a.read(other), {"fixture": "Grüße"})
        r.save(p, {"fixture": "Grüße"})
        with self.assertRaises(a.IntegrityError):
            r.save(p, {"fixture": "changed"})

    def test_D_long_paths_and_C_sources_roundtrip(self):
        p = self.base / ("a" * 120) / ("b" * 120) / "receipt.json"
        r.save(p, {"long": True})
        self.assertGreater(len(str(p)), 260)
        self.assertEqual(r.reference(p), r.reference(a.storage_path(p)))
        self.assertEqual(a.read(r.resolve_reference(r.reference(p))), {"long": True})
        name = r.reference(c.ROOT / "config/taxonomy.yaml")
        self.assertEqual(name, "config/taxonomy.yaml")
        self.assertEqual(c.sha(r.resolve_reference(name)), c.sha(c.ROOT / name))
        self.assertEqual(r.normal(c.ROOT / r.reference(p)), r.normal(p))

    def test_reference_escapes_and_other_drives_rejected(self):
        for name in [
            "../outside",
            "D:relative.json",
            "/rooted.json",
            "D:/outside.json",
            "C:/Windows/win.ini",
            "config/../config/taxonomy.yaml",
        ]:
            with self.subTest(name=name), self.assertRaises(a.IntegrityError):
                r.resolve_reference(name)
        with self.assertRaises(a.IntegrityError):
            r.reference(Path("D:/outside.json"))

    def test_output_saves_cannot_touch_old_files_or_other_D_folders(self):
        for path in [
            c.ROOT / "must-not-exist.json",
            r.SOURCE_BASE / "plan.json",
            r.BASE.parent / "must-not-exist.json",
            r.BASE,
            r.LOCAL,
        ]:
            with self.subTest(path=str(path)), self.assertRaises(a.IntegrityError):
                r.save(path, {})
        with self.assertRaises(a.IntegrityError):
            r.queue(AMENDMENT, base=c.ROOT)

    def test_original_stage_and_auditor_code_and_science_unchanged(self):
        before = e.queue("broad")
        b = r.queue(AMENDMENT)
        self.assertEqual(b.execute_stage.__code__, before.execute_stage.__code__)
        self.assertEqual(
            b.audit.audit_stage.__code__, before.audit.audit_stage.__code__
        )
        self.assertEqual(
            b.audit.finalize_queue.__code__, before.audit.finalize_queue.__code__
        )
        self.assertEqual(
            (b.BASE, b.PLAN, b.ROOT, b.WORKERS), (r.BASE, r.PLAN, c.ROOT, 128)
        )
        self.assertEqual(e.queue("broad").BASE, r.SOURCE_BASE)
        self.assertEqual(e.template.WORKERS, 8)
        self.assertEqual(e.queue("broad").WORKERS, 8)
        for name in [
            "provenance_path",
            "provenance",
            "verify_provenance",
            "bind_stage",
            "execute_schedule",
            "runtime_completion",
        ]:
            self.assertEqual(
                getattr(r.runtime(), name).__code__, getattr(r.original, name).__code__
            )

    def test_availability_default_writer_is_rebound_to_D(self):
        b = r.queue(AMENDMENT)
        availability = b.execute_stage.__kwdefaults__["availability_check"]
        self.assertIs(availability, b.availability)
        self.assertEqual(availability.__globals__["BASE"], r.BASE)
        raw = json.dumps({"data": [{"id": a.MODEL}]}).encode()
        response = SimpleNamespace(
            status_code=200, content=raw, json=lambda: json.loads(raw)
        )
        client = SimpleNamespace(
            _endpoint="https://example.invalid/chat/completions", _token="synthetic"
        )
        with patch("requests.get", return_value=response) as get:
            availability(self.base, client)
        self.assertEqual(get.call_count, 1)
        self.assertEqual(a.read(self.base / "model_check_response.json")["status"], 200)

    def test_per_item_identity_and_payload_ignore_storage(self):
        old = e.queue("broad")
        new = r.queue(AMENDMENT)
        job = fixtures.fixture_job(study="broad")
        self.assertEqual(
            old.identity(job, r.original.PLANS["broad"]),
            new.identity(job, r.original.PLANS["broad"]),
        )
        self.assertEqual(
            old.item_dir(old.BASE, job, "same").name,
            new.item_dir(new.BASE, job, "same").name,
        )
        self.assertEqual(
            old.engines(fixtures.fixture_catalog([job]))[
                "config/taxonomy.yaml"
            ].implementation(),
            new.engines(fixtures.fixture_catalog([job]))[
                "config/taxonomy.yaml"
            ].implementation(),
        )

    def test_live_mode_boundaries_and_mock_separation(self):
        b = r.queue(AMENDMENT)
        self.assertEqual(
            r.normal(b.checked_root(r.BASE / "live/broad", "broad", "live")),
            r.BASE / "live/broad",
        )
        for root, mode in [
            (r.SOURCE_BASE / "live/broad", "live"),
            (self.base, "live"),
            (r.BASE / "live/broad", "mock"),
            (r.BASE.parent, "mock"),
        ]:
            with self.subTest(root=str(root), mode=mode), self.assertRaises(
                a.IntegrityError
            ):
                b.checked_root(root, "broad", mode)

    def test_storage_budget_arithmetic_and_invalid_counts(self):
        self.assertEqual(r.required_d_bytes(194643), 354102181888)
        self.assertEqual(r.required_d_bytes(0), 328589934592)
        for value in [-1, True, 194644, 0.5]:
            with self.assertRaises(a.IntegrityError):
                r.required_d_bytes(value)

    def test_C_and_D_guards_check_the_correct_volumes(self):
        with patch.object(
            r.original.resources, "resources", return_value={}
        ) as host, patch.object(
            r.shutil, "disk_usage", return_value=SimpleNamespace(free=400000000000)
        ) as disk:
            r.host_check(194643, initial=True)
            host.assert_called_with(2.5, 8)
            disk.assert_called_once_with(r.BASE)
            r.host_check(1)
            host.assert_called_with(0.75, 8)
        with patch.object(
            r.original.resources, "resources", return_value={}
        ), patch.object(
            r.shutil,
            "disk_usage",
            return_value=SimpleNamespace(free=r.required_d_bytes(1) - 1),
        ):
            with self.assertRaises(a.IntegrityError):
                r.host_check(1)

    def test_changed_storage_identity_stops(self):
        changed = dict(r.layout_record(), volume_device_id=-1)
        with patch.object(r, "layout_record", return_value=changed), self.assertRaises(
            a.IntegrityError
        ):
            r.check_layout()

    def test_shared_legacy_and_D_locks_all_required(self):
        with r.queue_locks():
            with self.assertRaises(a.IntegrityError), r.queue_locks():
                pass
            for path in r.original.lock_paths():
                with self.assertRaises(a.IntegrityError), a.item_lock(
                    a.storage_path(path)
                ):
                    pass
            with self.assertRaises(a.IntegrityError), a.item_lock(
                a.storage_path(r.BASE / "combined_launcher.lock")
            ):
                pass
        with r.queue_locks():
            pass

    def test_failed_readiness_precedes_credentials(self):
        with patch.object(
            r, "validate", side_effect=a.IntegrityError("synthetic readiness failure")
        ), patch.object(e.core, "live_client") as live, patch.object(
            r.original, "load_catalog"
        ) as cat:
            with self.assertRaises(a.IntegrityError):
                r.run()
            live.assert_not_called()
            cat.assert_not_called()

    def test_network_and_credentials_blocked_offline(self):
        import requests

        with r.original.offline():
            for invoke in [
                lambda: socket.create_connection(("example.invalid", 1)),
                lambda: socket.socket().connect_ex(("127.0.0.1", 1)),
                lambda: requests.get("https://example.invalid"),
                lambda: e.core.live_client(),
            ]:
                with self.assertRaises(RuntimeError):
                    invoke()

    def test_missing_or_changed_provenance_stops_before_factory(self):
        b = r.queue(AMENDMENT)
        j = fixtures.fixture_job(study="broad")
        cat = fixtures.fixture_catalog([j])
        stage = dict(
            id="synthetic_01", study="broad", items=1, ordinary_calls=3, jobs=[j]
        )
        rt = r.runtime()
        for variant in ["missing", "changed"]:
            root = self.base / variant
            rt.execute_schedule(
                b,
                {"stages": [stage]},
                cat,
                root,
                "fixture",
                mode="mock",
                factory=lambda: fixtures.Scripted([None] * 3),
            )
            store = root / "broad"
            attempt = next(
                (a.storage_path(store) / "stages/synthetic_01").glob("attempt_*")
            )
            marker = rt.provenance_path(b, store, stage["id"], attempt.name)
            original_bytes = a.storage_path(marker).read_bytes()
            r.save(self.base / (variant + "_original_marker.json"), a.read(marker))
            if variant == "missing":
                marker.rename(marker.with_suffix(".preserved"))
            else:
                marker.write_bytes(a.encoded({"altered_synthetic_marker": True}))
            with patch.object(b, "execute_stage") as execute:
                with self.assertRaises((a.IntegrityError, FileNotFoundError)):
                    rt.execute_schedule(
                        b,
                        {"stages": [stage]},
                        cat,
                        root,
                        "fixture",
                        mode="mock",
                        factory=fixtures.never,
                    )
                execute.assert_not_called()
            self.assertTrue(original_bytes)


def run_tests():
    b = r.queue(AMENDMENT)
    legacy, old_root = inherited.inherited_suite(b)
    suite = unittest.TestSuite(
        [legacy, unittest.defaultTestLoader.loadTestsFromTestCase(StorageTests)]
    )
    with r.original.offline():
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    record = dict(
        passed=result.wasSuccessful(),
        tests_run=result.testsRun,
        errors=len(result.errors),
        failures=len(result.failures),
        preserved_fixtures=[r.reference(TEST_ROOT), r.reference(old_root)],
        source_hashes={r.reference(p): c.sha(p) for p in r.sources()},
        environment=e.core.environment(),
        actual_Qwen_calls=0,
        actual_model_list_GETs=0,
        training_fits=0,
    )
    r.save(TEST_ROOT / "result.json", record)
    if not result.wasSuccessful():
        raise SystemExit(1)
    r.save(r.LOCAL / "tests.json", record)


if __name__ == "__main__":
    run_tests()

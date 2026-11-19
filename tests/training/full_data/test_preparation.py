from contextlib import ExitStack
import importlib.util
import io
import json
import os
from pathlib import Path
from src.hsrag.paths import PROJECT_ROOT
import socket
import subprocess
import sys
import time
import unittest
from unittest.mock import patch, MagicMock
import uuid
import zipfile
import numpy as np

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))
from training.full_data import plan as p
from training.full_data import common as c
from training.full_data import runtime as r
from training.full_data import study as study
from training.full_data import archive as archive
from training.full_data import selection as selection
from training.full_data import reference as reference
from training.full_data import audit as audit
from training.full_data import controller as controller
from training.full_data import backup
from training.full_data import unpack
from training.recovery import update_rules
from training.full_data import backup_capacity

TRAIN_ARCHIVE_AVAILABLE = all(
    (p.PACKAGE / "data/train" / (pool + ".json")).is_file() for pool in p.NAMES
)


class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(os.environ["SQ2_TEST_OUTPUT"]) / uuid.uuid4().hex[:8]
        self.root.mkdir(parents=True)
        c.save(
            self.root / "test.json",
            dict(test=self._testMethodName, synthetic=True, training=False),
        )
        self.patches = ExitStack()
        self.addCleanup(self.patches.close)
        self.patches.enter_context(patch.object(r, "OUTPUT", self.root))

    def job(self, final=False):
        return p.job(
            p.NAMES[0], "hate", "l1_e1_u", 10 if final else 0, None if final else 0
        )

    def fake_identity(self, job):
        return dict(
            job=job,
            plan="fixture",
            profile="final" if job["mode"] == "final" else "search",
        )

    def fake_fit(self, attempt, job, data, cache, **kwargs):
        c.save(attempt / "training.json", dict(fixture=True, job=job))
        c.array(attempt / "held_logits.npy", np.ones((2, 2), dtype=np.float32))
        return dict(model_directory=None)

    def test_publish_idempotent_conflict_and_partial_preserved(self):
        path = self.root / "result.json"
        c.save(path, {"x": 1})
        c.save(path, {"x": 1})
        with self.assertRaises(RuntimeError):
            c.save(path, {"x": 2})
        partial = self.root / "result.json.old.writing"
        partial.write_text("old")
        c.save(self.root / "new.json", {"ok": True})
        self.assertEqual(partial.read_text(), "old")

    def test_arrays_idempotent_and_changed_array_rejected(self):
        path = self.root / "a.npz"
        c.arrays(path, {"a": np.arange(3)})
        c.arrays(path, {"a": np.arange(3)})
        with self.assertRaises(RuntimeError):
            c.arrays(path, {"a": np.arange(4)})

    def test_relative_paths_and_symlinks(self):
        for name in ("../a", "/tmp/a", "C:/a", "a\\b", "a//b", "a/./b", "."):
            with self.assertRaises(RuntimeError):
                r.inside(self.root, name)
        self.assertEqual(r.inside(self.root, "a/b"), self.root / "a/b")
        if os.name != "nt":
            link = self.root / "link"
            link.symlink_to(self.root, target_is_directory=True)
            try:
                with self.assertRaises(RuntimeError):
                    r.inside(self.root, "link/x")
                c.save(
                    self.root / "symlink_fixture.json",
                    dict(target=str(self.root), rejected=True),
                )
            finally:
                link.unlink()
            self.assertFalse(link.is_symlink())
            r.tree_bytes(self.root)

    def test_real_process_lock_excludes_and_releases(self):
        lease = self.root / "lease.lock"
        script = "import sys; sys.path.insert(0,sys.argv[1]); from training.full_data.runtime import lock; from pathlib import Path\nwith lock(Path(sys.argv[2])): print('locked')"
        with r.lock(lease):
            result = subprocess.run(
                [sys.executable, "-B", "-c", script, str(PROJECT_ROOT), str(lease)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertNotEqual(result.returncode, 0)
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, str(PROJECT_ROOT), str(lease)],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_no_network_guard(self):
        with r.offline():
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                socket.create_connection(("localhost", 9))

    @unittest.skipUnless(
        TRAIN_ARCHIVE_AVAILABLE, "Original full-data TRAIN memberships are absent"
    )
    def test_full_grid_counts_and_unique_scheduling(self):
        self.assertEqual(len(p.configs()), 42)
        slots = []
        states = {}
        total = 0
        for pool in p.NAMES:
            jobs = p.jobs(pool)
            scheduled = [j for group in p.allocation(pool) for j in group]
            self.assertEqual(
                sorted(j["id"] for j in jobs), sorted(j["id"] for j in scheduled)
            )
            data = p.data(pool)
            for j in jobs:
                state, steps = p.expected_state(data, j)
                states[state] = states.get(state, 0) + 1
                total += steps
            slots += jobs
        self.assertEqual(len({j["id"] for j in slots}), 22680)
        self.assertEqual(total, 13436604)
        self.assertEqual(
            states, dict(empty_held_fold=252, initialization_only=270, trained=22158)
        )

    def test_no_unreviewed_profile(self):
        self.assertEqual(
            controller.gpu_environment(8, 97887)["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"],
            "10",
        )
        self.assertEqual(
            controller.gpu_environment(1, 97887)["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"],
            "80",
        )
        for count in (2, 3, 4, 16):
            with self.assertRaises(RuntimeError):
                controller.gpu_environment(count, 97887)

    def test_fold_leakage_rejected(self):
        d = {
            "heads": {
                "hate": {
                    "rows": [
                        dict(feature_row=0, cluster="x"),
                        dict(feature_row=1, cluster="x"),
                    ]
                }
            },
            "folds": {"42": [0, 1]},
        }
        with self.assertRaises(RuntimeError):
            p.split(d, {"head": "hate", "fold": 0})

    @unittest.skipUnless(
        TRAIN_ARCHIVE_AVAILABLE, "Original full-data TRAIN memberships are absent"
    )
    def test_selection_independent_complete_candidate_check(self):
        data = p.data(p.NAMES[0])
        rng = np.random.default_rng(921)
        oof = {
            h: {
                cfg["id"]: rng.uniform(
                    0.01,
                    0.99,
                    (3, len(data["heads"][h]["rows"]), len(data["heads"][h]["labels"])),
                )
                for cfg in p.configs()
            }
            for h in p.HEADS
        }
        selected = selection.select(data, oof)
        self.assertEqual(reference.check_selection(data, oof, selected), 1232)
        selected["tuned"]["hate"]["selected"]["mean_score"] += 0.1
        with self.assertRaises(RuntimeError):
            reference.check_selection(data, oof, selected)

    def test_declared_tie_order(self):
        rows = [
            dict(config=cfg, threshold=t, mean_score=0.7)
            for cfg in ("l0_e1_u", "l1_e1_u", "l0_e2_u", "l0_e1_b")
            for t in (0.4, 0.5, 0.6)
        ]
        self.assertEqual(
            min(rows, key=selection.candidate_key),
            dict(config="l0_e1_u", threshold=0.5, mean_score=0.7),
        )

    def test_final_dedup_and_seeds(self):
        selected = {
            family: {h: dict(selected=dict(config="l1_e20_u")) for h in p.HEADS}
            for family in p.FAMILIES
        }
        jobs = p.final_jobs(p.NAMES[0], selected)
        self.assertEqual(len(jobs), 20)
        self.assertEqual({j["seed"] for j in jobs}, set(p.FINAL_SEEDS))
        selected["fixed_balanced"] = {
            h: dict(selected=dict(config="l1_e20_b")) for h in p.HEADS
        }
        selected["tuned"] = {h: dict(selected=dict(config="l0_e60_u")) for h in p.HEADS}
        self.assertEqual(len(p.final_jobs(p.NAMES[0], selected)), 60)

    def test_threshold_boundaries_and_matching_gate(self):
        prob = {
            "hate": np.array([[0.5, 0.5], [0.4, 0.6]]),
            "hate_type": np.ones((2, 2)) * 0.5,
            "target_group": np.ones((2, 2)) * 0.5,
            "severity": np.array([[0.1, 0.9], [0.9, 0.1]]),
        }
        heads = {h: dict(selected=dict(threshold=0.5)) for h in p.HEADS}
        value = audit.decoded(prob, heads, "selected")
        self.assertEqual(value["hate"].tolist(), [[1, 0], [0, 1]])
        self.assertEqual(value["hate_type"].tolist(), [[0, 0], [1, 1]])
        self.assertEqual(value["hate_type_ungated"].tolist(), [[1, 1], [1, 1]])
        self.assertEqual(value["severity"].tolist(), [[0, 0], [1, 0]])

    @unittest.skipUnless(
        TRAIN_ARCHIVE_AVAILABLE, "Original full-data TRAIN memberships are absent"
    )
    def test_oof_duplicate_rejected(self):
        job = self.job()
        data = p.data(job["pool"])
        spec, fit, held = p.split(data, job)
        dest = np.full((3, len(spec["rows"]), 2), np.nan)
        logits = np.zeros((len(held), 2), np.float32)
        audit.place_oof(dest, data, job, logits)
        with self.assertRaises(RuntimeError):
            audit.place_oof(dest, data, job, logits)

    def test_learning_rate_trace_and_amp_skip(self):
        m = dict(
            planned_steps=3,
            configuration=dict(learning_rate=2e-5),
            successful_steps=2,
            steps=[
                dict(lr=0, skipped=False, scale_before=8, scale_after=8),
                dict(lr=2e-5, skipped=True, scale_before=8, scale_after=4),
                dict(lr=2e-5, skipped=False, scale_before=4, scale_after=4),
            ],
        )
        reference.require_learning_rate_trace(m)
        m["steps"][2]["lr"] = 1e-5
        with self.assertRaises(RuntimeError):
            reference.require_learning_rate_trace(m)

    @unittest.skipUnless(
        TRAIN_ARCHIVE_AVAILABLE, "Original full-data TRAIN memberships are absent"
    )
    def test_completed_fit_is_reused_and_corruption_stops(self):
        from training.full_data import train as train

        job = self.job()
        data = p.data(job["pool"])
        with patch.object(study, "identity", self.fake_identity), patch.object(
            r, "deadline_guard"
        ), patch.object(
            train, "train_attempt", side_effect=self.fake_fit
        ) as fit, patch.object(
            audit, "metadata"
        ):
            study.execute(job, data, {})
            study.execute(job, data, {})
            self.assertEqual(fit.call_count, 1)
            _, attempt = study.receipt(job)
            (attempt / "held_logits.npy").write_bytes(b"bad")
            with self.assertRaises(RuntimeError):
                study.execute(job, data, {})
            self.assertEqual(fit.call_count, 1)

    @unittest.skipUnless(
        TRAIN_ARCHIVE_AVAILABLE, "Original full-data TRAIN memberships are absent"
    )
    def test_interrupted_fit_uses_new_attempt(self):
        from training.full_data import train as train

        job = self.job()
        data = p.data(job["pool"])
        with patch.object(study, "identity", self.fake_identity), patch.object(
            r, "deadline_guard"
        ), patch.object(
            train, "train_attempt", side_effect=r.Pause("fixture stop")
        ), patch.object(
            audit, "metadata"
        ):
            with self.assertRaises(r.Pause):
                study.execute(job, data, {})
        with patch.object(study, "identity", self.fake_identity), patch.object(
            r, "deadline_guard"
        ), patch.object(
            train, "train_attempt", side_effect=self.fake_fit
        ), patch.object(
            audit, "metadata"
        ):
            study.execute(job, data, {})
        attempts = list((study.job_dir(job) / "attempts").iterdir())
        self.assertEqual(len(attempts), 2)
        self.assertEqual(
            len(list((study.job_dir(job) / "attempts").glob("*/stopped.json"))), 1
        )

    @unittest.skipUnless(
        TRAIN_ARCHIVE_AVAILABLE, "Original full-data TRAIN memberships are absent"
    )
    def test_numerical_failure_is_not_silently_retried(self):
        from training.full_data import train as train

        job = self.job()
        data = p.data(job["pool"])
        with patch.object(study, "identity", self.fake_identity), patch.object(
            r, "deadline_guard"
        ), patch.object(
            train, "train_attempt", side_effect=RuntimeError("bad numerics")
        ) as fit:
            with self.assertRaises(RuntimeError):
                study.execute(job, data, {})
            with self.assertRaisesRegex(RuntimeError, "needs review"):
                study.execute(job, data, {})
            self.assertEqual(fit.call_count, 1)

    @unittest.skipUnless(
        TRAIN_ARCHIVE_AVAILABLE, "Original full-data TRAIN memberships are absent"
    )
    def test_empty_fold_never_calls_trainer(self):
        from training.full_data import train as train

        data = p.data(p.NAMES[0])
        job = next(
            j
            for j in p.jobs(p.NAMES[0])
            if p.expected_state(data, j)[0] == "empty_held_fold"
        )
        with patch.object(study, "identity", self.fake_identity), patch.object(
            r, "deadline_guard"
        ), patch.object(train, "train_attempt") as fit:
            study.execute(job, data, {})
            saved, attempt = study.receipt(job)
            self.assertEqual(saved["state"], "empty_held_fold")
            fit.assert_not_called()
            self.assertFalse((attempt / "training.json").exists())

    def test_budget_does_not_reset_and_unknown_crash_stops(self):
        path = self.root / "invocations/a/started.json"
        c.save(path, {"start": "fixture"})
        with self.assertRaises(RuntimeError):
            controller.used_seconds()
        c.save(
            path.parent / "finished.json", dict(started_sha256=c.sha(path), seconds=100)
        )
        self.assertEqual(controller.used_seconds(), 100)
        path = self.root / "invocations/b/started.json"
        c.save(path, {"start": "fixture"})
        c.save(
            path.parent / "finished.json", dict(started_sha256=c.sha(path), seconds=200)
        )
        self.assertEqual(controller.used_seconds(), 300)

    def test_storage_guards_include_other_results_and_reserve(self):
        inputs = self.root / "input_volume"
        outputs = self.root / "output_volume"
        inputs.mkdir()
        outputs.mkdir()
        empty = r.tree_bytes(outputs)
        (outputs / "prior-work").write_bytes(b"x" * 100)
        occupied = r.tree_bytes(outputs)
        self.assertGreater(occupied, empty)

        limits = dict(
            input_bytes=r.tree_bytes(inputs) + 10000,
            output_bytes=occupied + 10000,
            free_disk_bytes=0,
            project_bytes=250000000000,
        )
        with patch.object(r, "settings", return_value=limits):
            result = r.storage(0, inputs, outputs)
            self.assertEqual(result["outputs"], occupied)
            r.storage(10000, inputs, outputs)
            with self.assertRaises(r.Pause):
                r.storage(10001, inputs, outputs)
        self.assertEqual((outputs / "prior-work").read_bytes(), b"x" * 100)

    def archive_fixture(self):
        pool = p.NAMES[0]
        for stem in ("jobs", "models", "inference", "predictions"):
            c.save(self.root / stem / pool / "fixture.json", dict(stem=stem, pool=pool))
        c.save(
            self.root / "audits" / f"{pool}_complete.json",
            dict(status="pass", pool=pool, outputs={}),
        )
        c.save(self.root / "selections" / f"{pool}.json", dict(search_receipts={}))
        with patch.object(r, "storage"), patch.object(r, "deadline_guard"):
            archive.build(pool)
        root = archive.archive_root(pool)
        value = c.read(root / "complete.json")
        parts = {
            k: dict(bytes=v["bytes"], sha256=v["sha256"])
            for k, v in value["parts"].items()
        }
        receipt = dict(
            pool=pool,
            archive_manifest_sha256=c.sha(root / "complete.json"),
            copies=[
                dict(volume_id=v, directory=v, parts=parts, all_members_verified=True)
                for v in ("disk-A", "disk-B")
            ],
        )
        path = self.root / "backup.json"
        c.save(path, receipt)
        return pool, path

    def test_archive_roundtrip_and_two_volume_requirement(self):
        pool, path = self.archive_fixture()
        archive.verify(pool)
        value = c.read(path)
        value["copies"][1]["volume_id"] = "disk-A"
        with self.assertRaises(RuntimeError):
            archive.validate_copies(
                value,
                c.read(archive.archive_root(pool) / "complete.json"),
                c.sha(archive.archive_root(pool) / "complete.json"),
            )

    def test_archive_corruption_rejected(self):
        pool, path = self.archive_fixture()
        part = next(archive.archive_root(pool).glob("*.zip"))
        with part.open("ab") as f:
            f.write(b"bad")
        with self.assertRaises(RuntimeError):
            archive.verify(pool)

    def test_release_only_verified_pool_files(self):
        pool, path = self.archive_fixture()
        outside = self.root / "models/other_pool/keep.txt"
        outside.parent.mkdir(parents=True)
        outside.write_text("keep")
        archive.release(pool, path)
        archive.verify(pool)
        archive.release(pool, path)
        self.assertTrue(outside.exists())
        self.assertTrue((self.root / "jobs" / pool / "fixture.json").exists())
        self.assertFalse((self.root / "models" / pool / "fixture.json").exists())

    def test_release_interruption_is_resumable(self):
        pool, path = self.archive_fixture()
        original = Path.unlink
        calls = [0]

        def interrupt(target, *args, **kwargs):
            calls[0] += 1
            if calls[0] == 2:
                raise OSError("fixture interruption")
            return original(target, *args, **kwargs)

        with patch.object(Path, "unlink", interrupt):
            with self.assertRaises(OSError):
                archive.release(pool, path)
        self.assertTrue((archive.archive_root(pool) / "release_intent.json").exists())
        archive.release(pool, path)
        archive.verify_release(pool)

    def test_archive_partition_bound(self):
        entries = {f"file{i}": dict(bytes=100, sha256="0" * 64) for i in range(4)}
        parts = archive.groups(entries, limit=9000)
        self.assertEqual(len(parts), 2)
        self.assertEqual(sum(len(x) for x in parts), 4)
        with self.assertRaises(RuntimeError):
            archive.groups(entries, limit=100)

    def test_archive_path_traversal_rejected(self):
        path = self.root / "bad.zip"
        with zipfile.ZipFile(path, "x", compression=zipfile.ZIP_STORED) as z:
            z.writestr("../escape", b"hi")
        record = dict(
            bytes=path.stat().st_size,
            sha256=c.sha(path),
            files={"../escape": dict(bytes=2, sha256="x")},
        )
        with self.assertRaises(RuntimeError):
            archive.verify_part(path, record)

    def test_download_resume_and_checksum(self):
        payload = b"0123456789"
        dest = self.root / "part.zip"
        partial = dest.with_suffix(".zip.partial")
        partial.write_bytes(payload[:3])
        digest = __import__("hashlib").sha256(payload).hexdigest()
        expected = dict(bytes=10, sha256=digest)
        response = io.BytesIO(payload[3:])
        response.status = 206
        response.headers = {"Content-Range": "bytes 3-9/10", "Content-Length": "7"}
        with patch.object(backup, "fetch", return_value=response):
            backup.download(
                None,
                "hidden",
                backup.HOST
                + "/dav/results/sq2_full_20261029_v1/archives/pool/part.zip",
                dest,
                expected,
            )
        self.assertEqual(dest.read_bytes(), payload)

    def test_range_refusal_preserves_partial(self):
        dest = self.root / "part.zip"
        partial = dest.with_suffix(".zip.partial")
        partial.write_bytes(b"012")
        response = io.BytesIO(b"0123456789")
        response.status = 200
        response.headers = {"Content-Length": "10"}
        with patch.object(backup, "fetch", return_value=response):
            with self.assertRaises(RuntimeError):
                backup.download(
                    None,
                    "hidden",
                    backup.HOST
                    + "/dav/results/sq2_full_20261029_v1/archives/pool/part.zip",
                    dest,
                    dict(bytes=10, sha256="0" * 64),
                )
        self.assertEqual(partial.read_bytes(), b"012")

    def test_download_credentials_cannot_redirect(self):
        with self.assertRaises(RuntimeError):
            backup.NoRedirect().redirect_request(None, None, None, None, None, None)
        with self.assertRaises(RuntimeError):
            backup.fetch(None, "hidden", "https://example.org/file")

    def test_real_volume_identity(self):
        first = backup.volume_id(self.root)
        second = backup.volume_id(self.root.parent)
        self.assertTrue(first)
        self.assertEqual(first, second)

    def test_backup_capacity_rejects_same_volume(self):
        with self.assertRaises(RuntimeError):
            backup_capacity.measure(self.root, self.root.parent)

    def test_missing_backup_capacity_pauses_after_selection(self):
        selected = {
            family: {h: dict(selected=dict(config="l1_e20_u")) for h in p.HEADS}
            for family in p.FAMILIES
        }
        selected["fixed_balanced"] = {
            h: dict(selected=dict(config="l1_e20_b")) for h in p.HEADS
        }
        for pool in p.NAMES:
            c.save(self.root / "selections" / f"{pool}.json", dict(families=selected))
        actual = Path.exists

        def exists(path):
            return (
                False
                if str(path).replace("\\", "/") == "/inputs/backup_capacity.json"
                else actual(path)
            )

        with patch.object(Path, "exists", exists):
            with self.assertRaises(r.Pause):
                controller.backup_capacity()
        value = c.read(self.root / "final_export_plan.json")
        self.assertEqual(value["models_total"], 360)
        self.assertEqual(
            value["required_initial_free_per_backup_volume_bytes"],
            360 * 505000000 + 37000000000,
        )

    def test_unpack_checks_hash_members_and_existing_destination(self):
        source = self.root / "bundle.zip"
        payload = b"fixture"
        digest = __import__("hashlib").sha256(payload).hexdigest()
        with zipfile.ZipFile(source, "x") as z:
            z.writestr("data/value.txt", payload)
            z.writestr(
                "manifest.json", json.dumps(dict(files={"data/value.txt": digest}))
            )
        destination = self.root / "sq2_full"
        with self.assertRaises(RuntimeError):
            unpack.unpack(source, destination, "0" * 64)
        self.assertFalse(destination.exists())
        unpack.unpack(source, destination, c.sha(source))
        self.assertEqual((destination / "data/value.txt").read_bytes(), payload)
        with self.assertRaises(RuntimeError):
            unpack.unpack(source, destination, c.sha(source))

    def test_unpack_refuses_traversal_before_writing(self):
        source = self.root / "bundle.zip"
        with zipfile.ZipFile(source, "x") as z:
            z.writestr("../escape", b"hi")
            z.writestr("manifest.json", json.dumps(dict(files={"../escape": "x"})))
        with self.assertRaises(RuntimeError):
            unpack.unpack(source, self.root / "sq2_full", c.sha(source))
        self.assertFalse((self.root / "sq2_full").exists())

    def test_gpu_driver_or_vram_change_stops(self):
        value = MagicMock(
            stdout="GPU-id, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 100, 0, 50, 590.44.01\n"
        )
        with patch.object(r.subprocess, "run", return_value=value):
            self.assertEqual(r.gpu()["driver"], "590.44.01")
        value.stdout = value.stdout.replace("590.44.01", "999.1")
        with patch.object(r.subprocess, "run", return_value=value):
            with self.assertRaises(RuntimeError):
                r.gpu()
        value.stdout = "GPU-id, NVIDIA RTX PRO 6000 Blackwell Server Edition, 97887, 90000, 0, 50, 590.44.01\n"
        with patch.object(r.subprocess, "run", return_value=value):
            with self.assertRaises(r.Pause):
                r.gpu()

    def test_owned_child_cleanup(self):
        child = subprocess.Popen(
            [sys.executable, "-B", "-c", "import time; time.sleep(30)"]
        )

        child.terminate()
        child.wait(timeout=10)
        r.stop_children([child], self.root / "STOP")
        self.assertIsNotNone(child.poll())
        self.assertTrue((self.root / "STOP").exists())

    def controller_fixture(self, fail=False):
        worker = self.root / "fake_worker.py"
        worker.write_text(
            "import sys,json,hashlib,time\nfrom pathlib import Path\np=Path(sys.argv[1]); v=json.loads(p.read_text())\nif sys.argv[2]=='fail':\n if v['worker']==0: sys.exit(4)\n end=time.monotonic()+20\n while not Path(v['session']['stop_file']).exists() and time.monotonic()<end: time.sleep(.05)\n sys.exit(5)\nd=dict(assignment_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),jobs=len(v['jobs']))\np.with_suffix('.complete.json').write_text(json.dumps(d))\n",
            encoding="utf8",
        )
        native = subprocess.Popen
        children = []

        def spawn(command, **kwargs):
            child = native(
                [
                    sys.executable,
                    "-B",
                    str(worker),
                    command[-1],
                    "fail" if fail else "pass",
                ],
                **kwargs,
            )
            children.append(child)
            return child

        invocation = self.root / "invocation"
        invocation.mkdir()
        real_sha = c.sha

        def digest(path):
            return (
                "0" * 64
                if Path(path) == p.PACKAGE / "manifest.json"
                else real_sha(path)
            )

        with patch.object(
            controller.subprocess, "Popen", side_effect=spawn
        ), patch.object(c, "sha", side_effect=digest), patch.object(
            r, "deadline_guard"
        ), patch.object(
            r, "memory", return_value={}
        ), patch.object(
            r, "storage", return_value={}
        ), patch.object(
            r,
            "gpu",
            return_value=dict(
                uuid=r.settings()["gpu_uuid"], used_mib=0, busy=0, total_mib=97887
            ),
        ):
            if fail:
                with self.assertRaises(RuntimeError):
                    controller.run_workers(
                        p.NAMES[0],
                        "search",
                        dict(stop_file=str(self.root / "STOP")),
                        invocation,
                    )
            else:
                controller.run_workers(
                    p.NAMES[0],
                    "search",
                    dict(stop_file=str(self.root / "STOP")),
                    invocation,
                )
        self.assertEqual(len(children), 8)
        self.assertTrue(all(x.poll() is not None for x in children))

    @unittest.skipUnless(
        TRAIN_ARCHIVE_AVAILABLE, "Original full-data TRAIN memberships are absent"
    )
    def test_controller_runs_all_synthetic_workers_once(self):
        self.controller_fixture()

    @unittest.skipUnless(
        TRAIN_ARCHIVE_AVAILABLE, "Original full-data TRAIN memberships are absent"
    )
    def test_failed_worker_stops_its_siblings(self):
        self.controller_fixture(fail=True)

    @unittest.skipUnless(
        TRAIN_ARCHIVE_AVAILABLE, "Original full-data TRAIN memberships are absent"
    )
    def test_inference_chunks_resume_and_orphan_validation(self):
        from training.full_data import train as train

        job = self.job(final=True)
        data = p.data(job["pool"])
        root = self.root / "inference" / job["id"]
        inputs = [dict(key=str(i), text="fixture " + str(i)) for i in range(1025)]
        c.save(study.job_dir(job) / "complete.json", {"fixture": True})
        read = c.read

        def reader(path):
            return (
                inputs
                if Path(path) == p.PACKAGE / "data/inference.json"
                else read(path)
            )

        def infer(model, tokens):
            return np.tile(np.array([[0.25, 0.75]], np.float32), (len(tokens), 1))

        source = dict(model_directory=str(self.root / "dummy_model"))
        torch = MagicMock()
        with patch.object(c, "read", side_effect=reader), patch.object(
            study, "identity", self.fake_identity
        ), patch.object(
            study, "receipt", return_value=(source, self.root)
        ), patch.object(
            r, "deadline_guard"
        ), patch.object(
            train, "setup", return_value=torch
        ), patch.object(
            train, "load", return_value=(None, None, None)
        ) as load, patch.object(
            train, "tokens", side_effect=lambda tok, texts: texts
        ), patch.object(
            train, "infer", side_effect=infer
        ):

            c.array(root / "chunks/000000_000512.npy", infer(None, list(range(512))))
            study.inference(job, data, {})
            study.inference(job, data, {})
            self.assertEqual(load.call_count, 1)
            self.assertEqual(
                study.check_inference(
                    job, inputs, data["heads"]["hate"]["labels"]
                ).shape,
                (1025, 2),
            )

            (root / "chunks/000512_001024.npy").write_bytes(b"bad")
            with self.assertRaises(RuntimeError):
                study.inference(job, data, {})
            self.assertEqual(load.call_count, 1)

    @unittest.skipUnless(
        TRAIN_ARCHIVE_AVAILABLE, "Original full-data TRAIN memberships are absent"
    )
    def test_changed_fit_identity_stops(self):
        from training.full_data import train as train

        job = self.job()
        data = p.data(job["pool"])
        with patch.object(study, "identity", self.fake_identity), patch.object(
            r, "deadline_guard"
        ), patch.object(
            train, "train_attempt", side_effect=self.fake_fit
        ), patch.object(
            audit, "metadata"
        ):
            study.execute(job, data, {})
            path = study.job_dir(job) / "complete.json"
            value = c.read(path)
            value["identity"]["job"]["seed"] = 2
            path.write_bytes(c.encoded(value))
            with self.assertRaises(RuntimeError):
                study.receipt(job)

    def test_zero_step_parameter_change_rejected(self):
        job = p.job(p.NAMES[0], "hate", "l0_e1_u", 0, 0)
        rows = [
            dict(id="fit", feature_row=0, cluster="a", target=0),
            dict(id="held", feature_row=1, cluster="b", target=1),
        ]
        data = dict(
            folds={"42": [1, 0]},
            heads={
                "hate": dict(rows=rows, labels=["not_hate", "hate"], kind="singlelabel")
            },
        )
        fingerprint = dict(
            raw_sha256="0" * 64, zero_canonical_sha256="0" * 64, negative_zero_count=0
        )
        m = dict(
            job=job,
            configuration=p.config_map()[job["config"]],
            pilot_only=False,
            fit_ids=["fit"],
            held_ids=["held"],
            planned_steps=1,
            attempted_steps=1,
            successful_steps=0,
            skipped_steps=1,
            steps=[
                dict(step=1, epoch=1, lr=0, skipped=True, scale_before=8, scale_after=4)
            ],
            class_weights=c.class_weights("hate", [0], 2, False),
            epochs=[
                dict(
                    epoch=1,
                    attempted_steps=1,
                    successful_steps=0,
                    training_objective=0.2,
                    held_unweighted_loss=float(np.log(2)),
                )
            ],
            initialization_only=True,
            initial_parameter_fingerprint=fingerprint,
            final_parameter_fingerprint=fingerprint,
            changes={name: 0 for name in update_rules.WATCHED},
        )
        c.save(self.root / "training.json", m)
        c.save(
            self.root / "parameter_equivalence.json",
            dict(before=fingerprint, after=fingerprint, changes=m["changes"]),
        )
        c.array(self.root / "held_logits.npy", np.zeros((1, 2), np.float32))
        c.array(self.root / "epoch_logits/001.npy", np.zeros((1, 2), np.float32))
        audit.metadata(job, data, self.root)
        m["changes"]["classifier.out_proj.weight"] = 0.1
        (self.root / "training.json").write_bytes(c.encoded(m))
        with self.assertRaises(RuntimeError):
            audit.metadata(job, data, self.root)

    def test_wrong_held_loss_is_rejected(self):
        self.test_zero_step_parameter_change_rejected()
        m = c.read(self.root / "training.json")
        m["changes"]["classifier.out_proj.weight"] = 0
        m["epochs"][0]["held_unweighted_loss"] += 0.01
        (self.root / "training.json").write_bytes(c.encoded(m))
        rows = [
            dict(id="fit", feature_row=0, cluster="a", target=0),
            dict(id="held", feature_row=1, cluster="b", target=1),
        ]
        data = dict(
            folds={"42": [1, 0]},
            heads={
                "hate": dict(rows=rows, labels=["not_hate", "hate"], kind="singlelabel")
            },
        )
        with self.assertRaisesRegex(RuntimeError, "epoch loss"):
            audit.metadata(m["job"], data, self.root)


if __name__ == "__main__":
    output = Path(os.environ["SQ2_TEST_OUTPUT"])
    output.mkdir(parents=True, exist_ok=True)
    with (output / "results.txt").open("x", encoding="utf8") as stream:
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(PreparationTests)
        )
    c.save(
        output / "results.json",
        dict(
            tests=result.testsRun,
            failures=len(result.failures),
            errors=len(result.errors),
            skipped=len(result.skipped),
            success=result.wasSuccessful(),
            new_model_fits=0,
            Qwen_calls=0,
            report_sha256=c.sha(output / "results.txt"),
        ),
    )
    print((output / "results.txt").read_text(encoding="utf8"))
    sys.exit(0 if result.wasSuccessful() else 1)

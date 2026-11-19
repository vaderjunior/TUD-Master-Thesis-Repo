import argparse
import copy
from datetime import datetime, timezone, timedelta
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from src.hsrag.paths import PROJECT_ROOT
import socket
import sys
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from training.type_update import checks

HERE = PROJECT_ROOT / "deploy/type_update"
ORIGINAL = Path(
    os.environ.get(
        "HSRAG_ORIGINAL_TRAINING_PACKAGE",
        str(HERE.parent / "full_training"),
    )
)


TRAIN_ARCHIVE_AVAILABLE = (
    HERE.parent / "full_training/data/train/uniform_5000_s42.json"
).is_file()


class SupplementTests(unittest.TestCase):
    def setUp(self):
        self.protocol = json.loads((HERE / "protocol.json").read_text(encoding="utf8"))
        self.meta = dict(
            job=checks.jobs()[0],
            configuration=self.protocol["configuration"],
            fit_ids=self.protocol["fit_ids"],
            held_ids=[],
            initialization_only=False,
            pilot_only=False,
            planned_steps=2,
            attempted_steps=2,
            steps=[dict(lr=0.0, skipped=False), dict(lr=5e-6, skipped=False)],
            changes={"first_encoder": 0.001, "last_encoder": 0.001, "head": 0.001},
            export_check=dict(clean_reload=True, max_abs_logit_difference=1e-7),
        )

    def original_arrays(self):
        out = {}
        for name, width in dict(
            hate=2, hate_type=7, target_group=7, severity=3
        ).items():
            out[name] = np.zeros((3, width), dtype=np.uint8)
            out[name + "_ungated"] = np.zeros((3, width), dtype=np.uint8)
        out["hate"][:] = [[0, 1], [1, 0], [0, 1]]
        out["hate_ungated"][:] = out["hate"]
        out["target_group"][0, 2] = 1
        out["severity"][2, 1] = 1
        return out

    def test_exact_five_seeds_and_full_final_identity(self):
        self.assertEqual([x["seed"] for x in checks.jobs()], [10, 11, 12, 13, 14])
        self.assertEqual(len({x["id"] for x in checks.jobs()}), 5)
        self.assertTrue(
            all(
                x["head"] == "hate_type" and x["fold"] is None and x["mode"] == "final"
                for x in checks.jobs()
            )
        )
        checks.validate_plan(self.protocol)

    def test_other_pool_head_or_seed_rejected(self):
        for key, value in [
            ("pool", "uniform_1000_s42"),
            ("head", "hate"),
            ("seed", 15),
            ("fold", 0),
            ("mode", "search"),
        ]:
            v = copy.deepcopy(self.protocol)
            v["jobs"][0][key] = value
            with self.assertRaises(RuntimeError):
                checks.validate_plan(v)

    def test_changed_or_extra_jobs_rejected(self):
        for jobs in (
            checks.jobs()[:-1],
            checks.jobs() + checks.jobs()[:1],
            list(reversed(checks.jobs())),
        ):
            v = copy.deepcopy(self.protocol)
            v["jobs"] = jobs
            with self.assertRaises(RuntimeError):
                checks.validate_plan(v)

    def test_changed_training_recipe_rejected(self):
        for key, value in [
            ("epochs", 20),
            ("learning_rate", 2e-5),
            ("loss", "balanced"),
        ]:
            v = copy.deepcopy(self.protocol)
            v["configuration"][key] = value
            with self.assertRaises(RuntimeError):
                checks.validate_plan(v)

    def test_resource_and_network_limits_cannot_expand(self):
        for key, value in [
            ("active_seconds", 7200),
            ("gpu_percent", 100),
            ("project_bytes", 251_000_000_000),
            ("Qwen_calls", 1),
            ("new_CV_fits", 1),
            ("additional_reserve_bytes", 0),
        ]:
            v = copy.deepcopy(self.protocol)
            v[key] = value
            with self.assertRaises(RuntimeError):
                checks.validate_plan(v)

    def test_changed_cutoff_labels_and_row_roster_rejected(self):
        for key, value in [
            ("cutoff", 0.4),
            ("labels", ["x"] * 7),
            ("fit_ids", ["x"] * 5),
            ("inference_rows", 19081),
        ]:
            v = copy.deepcopy(self.protocol)
            v[key] = value
            with self.assertRaises(RuntimeError):
                checks.validate_plan(v)

    def test_output_locations_and_decision_are_bound(self):
        for key, value in [
            ("results", self.protocol["original_results"]),
            ("proposal_sha256", "0" * 64),
        ]:
            v = copy.deepcopy(self.protocol)
            v[key] = value
            with self.assertRaises(RuntimeError):
                checks.validate_plan(v)

    def test_no_output_root_overlap(self):
        with tempfile.TemporaryDirectory() as d:
            old = Path(d) / "original"
            old.mkdir()
            for new in (old, old / "child", old.parent):
                with self.assertRaises(RuntimeError):
                    checks.separate_root(new, old)
            checks.separate_root(Path(d) / "extra", old)

    def test_cumulative_hour_does_not_reset_on_pause(self):
        now = datetime(2026, 11, 3, tzinfo=timezone.utc)
        records = [
            dict(status="paused", seconds=900.0),
            dict(status="paused", seconds=1700.0),
        ]
        left, end = checks.time_budget(records, self.protocol["stop_utc"], now)
        self.assertEqual(left, 1000.0)
        self.assertEqual(end, now + timedelta(seconds=1000))

    def test_stop_deadline_bounds_execution(self):
        now = datetime(2026, 11, 9, 22, 58, tzinfo=timezone.utc)
        left, end = checks.time_budget([], self.protocol["stop_utc"], now)
        self.assertEqual((end - now).total_seconds(), 60)
        with self.assertRaises(RuntimeError):
            checks.time_budget([], self.protocol["stop_utc"], end)

    def test_unclosed_failed_and_exhausted_invocations_stop(self):
        now = datetime(2026, 11, 3, tzinfo=timezone.utc)
        for records in (
            [None],
            [dict(status="failed", seconds=1)],
            [dict(status="paused", seconds=3600)],
            [dict(status="paused", seconds=-1)],
            [dict(status="paused", seconds=float("nan"))],
        ):
            with self.assertRaises(RuntimeError):
                checks.time_budget(records, self.protocol["stop_utc"], now)

    def test_capacity_retains_full_allowances(self):
        v = checks.capacity(
            3_400_000_000,
            180_000_000_000,
            100_000_000_000,
            26_000_000_000,
            2_000_000_000,
        )
        self.assertEqual(v["conservative_project_forecast"], 234_000_000_000)
        self.assertEqual(v["project_limit"], 250_000_000_000)

    def test_capacity_exact_output_boundary(self):
        checks.capacity(
            5_000_000_000,
            196_000_000_000,
            16_000_000_000,
            30_000_000_000,
            15_000_000_000,
        )
        with self.assertRaises(RuntimeError):
            checks.capacity(
                5_000_000_000,
                196_000_000_001,
                16_000_000_000,
                30_000_000_000,
                15_000_000_000,
            )

    def test_capacity_input_free_image_and_other_limits(self):
        good = [
            3_000_000_000,
            180_000_000_000,
            100_000_000_000,
            26_000_000_000,
            2_000_000_000,
        ]
        for index, value in [
            (0, 5_000_000_001),
            (2, 15_999_999_999),
            (3, 30_000_000_001),
            (4, 15_000_000_001),
            (1, -1),
            (0, True),
        ]:
            values = good.copy()
            values[index] = value
            with self.assertRaises(RuntimeError):
                checks.capacity(*values)

    def test_positive_update_and_clean_reload_accepted(self):
        checks.updated_model(self.meta, checks.jobs()[0], self.protocol["fit_ids"])

    def test_zero_rate_and_skipped_update_rejected(self):
        for values in (
            [dict(lr=0.0, skipped=False)] * 2,
            [dict(lr=0.0, skipped=False), dict(lr=5e-6, skipped=True)],
        ):
            v = copy.deepcopy(self.meta)
            v["steps"] = values
            with self.assertRaises(RuntimeError):
                checks.updated_model(v, checks.jobs()[0], self.protocol["fit_ids"])

    def test_no_encoder_or_head_change_rejected(self):
        for key in self.meta["changes"]:
            for value in (0.0, float("nan"), float("inf")):
                v = copy.deepcopy(self.meta)
                v["changes"][key] = value
                with self.assertRaises(RuntimeError):
                    checks.updated_model(v, checks.jobs()[0], self.protocol["fit_ids"])

    def test_no_silent_horizon_extension_or_row_change(self):
        for key, value in [
            ("attempted_steps", 3),
            ("planned_steps", 20),
            ("initialization_only", True),
            ("pilot_only", True),
            ("fit_ids", list(reversed(self.protocol["fit_ids"]))),
            ("held_ids", ["extra"]),
        ]:
            v = copy.deepcopy(self.meta)
            v[key] = value
            with self.assertRaises(RuntimeError):
                checks.updated_model(v, checks.jobs()[0], self.protocol["fit_ids"])

    def test_reload_failure_rejected(self):
        for record in (
            dict(clean_reload=False, max_abs_logit_difference=0.0),
            dict(clean_reload=True, max_abs_logit_difference=0.0001001),
            dict(clean_reload=True, max_abs_logit_difference=float("nan")),
        ):
            v = copy.deepcopy(self.meta)
            v["export_check"] = record
            with self.assertRaises(RuntimeError):
                checks.updated_model(v, checks.jobs()[0], self.protocol["fit_ids"])

    def test_type_gate_cutoff_equality_and_other_heads_unchanged(self):
        old = self.original_arrays()
        snapshot = {k: v.copy() for k, v in old.items()}
        probabilities = np.asarray([[0.49, 0.5, 0.51, 0.0, 1.0, 0.2, 0.8]] * 3)
        output = checks.replace_type(old, probabilities)
        self.assertTrue(np.array_equal(output["hate_type"][0], [0, 1, 1, 0, 1, 0, 1]))
        self.assertEqual(output["hate_type"][1].sum(), 0)
        self.assertEqual(output["hate_type_ungated"][1].sum(), 4)
        for key in old:
            self.assertTrue(np.array_equal(old[key], snapshot[key]))
            if not key.startswith("hate_type"):
                self.assertTrue(np.array_equal(old[key], output[key]))
                self.assertFalse(np.shares_memory(old[key], output[key]))

    def test_invalid_probability_shape_values_and_cutoff_rejected(self):
        old = self.original_arrays()
        for probability in (
            np.zeros((3, 6)),
            np.full((3, 7), np.nan),
            np.full((3, 7), 1.01),
        ):
            with self.assertRaises(RuntimeError):
                checks.replace_type(old, probability)
        with self.assertRaises(RuntimeError):
            checks.replace_type(old, np.zeros((3, 7)), 0.4)

    def test_invalid_or_missing_original_gate_rejected(self):
        for case in ("missing", "not_one_hot", "different_ungated", "wrong_rows"):
            old = self.original_arrays()
            if case == "missing":
                del old["hate"]
            elif case == "not_one_hot":
                old["hate"][0] = [1, 1]
            elif case == "different_ungated":
                old["hate_ungated"][0] = [1, 0]
            else:
                old["severity"] = old["severity"][:2]
            with self.assertRaises(RuntimeError):
                checks.replace_type(old, np.zeros((3, 7)))

    @unittest.skipUnless(
        all(
            (ORIGINAL / name).is_file()
            for name in (
                "manifest.json",
                "app/training_kernels.py",
                "app/study.py",
                "app/audit_kernels.py",
                "app/reference.py",
                "app/tokenizer_rules.py",
                "app/selection.py",
                "app/plan.py",
                "app/runtime.py",
            )
        ),
        "Original frozen training source package is absent",
    )
    def test_frozen_training_and_inference_sources_match_manifest(self):
        manifest = json.loads((ORIGINAL / "manifest.json").read_text(encoding="utf8"))
        for name in (
            "app/training_kernels.py",
            "app/study.py",
            "app/audit_kernels.py",
            "app/reference.py",
            "app/tokenizer_rules.py",
            "app/selection.py",
            "app/plan.py",
            "app/runtime.py",
        ):
            self.assertEqual(
                hashlib.sha256((ORIGINAL / name).read_bytes()).hexdigest(),
                manifest["files"][name],
            )

    @unittest.skipUnless(
        TRAIN_ARCHIVE_AVAILABLE, "Original full-data TRAIN memberships are absent"
    )
    def test_real_train_rows_and_job_split_match(self):
        from training.full_data import plan as p

        data = p.data(checks.POOL)
        for job in checks.jobs():
            self.assertEqual(
                job, p.job(checks.POOL, "hate_type", checks.CONFIG, job["seed"], None)
            )
            spec, fit, held = p.split(data, job)
            self.assertEqual([x["id"] for x in fit], self.protocol["fit_ids"])
            self.assertEqual(spec["labels"], self.protocol["labels"])
            self.assertEqual(held, [])
            self.assertEqual(p.expected_state(data, job), ("trained", 2))

    def test_original_network_block_remains_effective(self):
        from training.full_data import runtime as r

        with r.offline():
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                socket.create_connection(("localhost", 9))

    def test_original_single_controller_lock_excludes_process(self):
        from training.full_data import runtime as r
        import subprocess

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "worker.lock"
            code = "import sys;from training.full_data.runtime import lock;from pathlib import Path\nwith lock(Path(sys.argv[1])): print('locked')"
            command = [sys.executable, "-B", "-c", code, str(path)]
            with r.lock(path):
                self.assertNotEqual(
                    subprocess.run(command, capture_output=True, timeout=30).returncode,
                    0,
                )
            self.assertEqual(
                subprocess.run(command, capture_output=True, timeout=30).returncode, 0
            )

    def worker_fixture(self, fail_seed=None, bad_jobs=False):
        from training.full_data import common as c
        from training.full_data import runtime as r
        from training.full_data import plan as p
        from training.full_data import study as study
        from training.full_data import controller as controller
        from training.full_data import train as train

        spec = importlib.util.spec_from_file_location(
            "supplement_runner_fixture", PROJECT_ROOT / "training/type_update/run.py"
        )
        runner = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runner)
        with tempfile.TemporaryDirectory() as d, ExitStack() as stack:
            base = Path(d)
            root = base / "extra"
            package = base / "package"
            package.mkdir()
            (package / "manifest.json").write_text('{"fixture": true}', encoding="utf8")
            c.save(package / "protocol.json", self.protocol)
            assignment = root / "invocations/fixture/worker.json"
            roster = checks.jobs()
            if bad_jobs:
                roster[0] = {**roster[0], "head": "hate"}
            value = dict(
                jobs=roster,
                session=dict(stop_file=str(assignment.parent / "STOP")),
                bundle_sha256=c.sha(package / "manifest.json"),
                protocol_sha256=c.sha(package / "protocol.json"),
            )
            c.save(assignment, value)
            executed, inferred, reviewed = [], [], []

            def execute(job, data, session):
                self.assertEqual(r.OUTPUT, root)
                executed.append(job["seed"])
                if job["seed"] == fail_seed:
                    raise RuntimeError("fixture numerical stop")

            fake_torch = SimpleNamespace(
                cuda=SimpleNamespace(
                    device_count=lambda: 1,
                    get_device_properties=lambda _: SimpleNamespace(
                        major=12,
                        multi_processor_count=150,
                        name="fixture",
                        total_memory=1,
                    ),
                ),
                version=SimpleNamespace(cuda="13.0"),
                __version__="fixture",
            )
            data_reader = stack.enter_context(
                patch.object(p, "data", return_value={"fixture": True})
            )
            stack.enter_context(patch.object(runner, "HERE", package))
            stack.enter_context(patch.object(runner, "OUTPUT", root))
            stack.enter_context(
                patch.object(
                    runner, "original", return_value=(c, r, p, study, controller)
                )
            )
            stack.enter_context(
                patch.object(
                    runner,
                    "configure",
                    side_effect=lambda *args: setattr(r, "OUTPUT", root),
                )
            )
            stack.enter_context(patch.object(r, "OUTPUT", root))
            stack.enter_context(
                patch.object(runner, "validate", return_value=self.protocol)
            )
            stack.enter_context(patch.object(r, "deadline_guard"))
            stack.enter_context(patch.object(r, "memory", return_value={}))
            stack.enter_context(
                patch.object(r, "gpu", return_value=dict(total_mib=97887))
            )
            stack.enter_context(patch.object(runner.signal, "signal"))
            setup = stack.enter_context(
                patch.object(train, "setup", return_value=fake_torch)
            )
            stack.enter_context(patch.object(study, "execute", side_effect=execute))
            stack.enter_context(
                patch.object(
                    study,
                    "inference",
                    side_effect=lambda job, *args: inferred.append(job["seed"]),
                )
            )
            stack.enter_context(
                patch.object(
                    runner,
                    "model_review",
                    side_effect=lambda *args: reviewed.append(args[-1]["seed"]),
                )
            )
            stack.enter_context(
                patch.dict(
                    os.environ,
                    {
                        **controller.gpu_environment(1, 97887),
                        "CUDA_MPS_PIPE_DIRECTORY": "/run/nvidia-mps",
                    },
                )
            )
            if fail_seed is not None or bad_jobs:
                with self.assertRaises(RuntimeError):
                    runner.worker(assignment)
                self.assertFalse(assignment.with_suffix(".complete.json").exists())
            else:
                runner.worker(assignment)
                self.assertEqual(
                    c.read(assignment.with_suffix(".complete.json"))["models"], 5
                )
            if bad_jobs:
                setup.assert_not_called()
            else:
                data_reader.assert_called_once_with(checks.POOL)
            return executed, inferred, reviewed

    def test_worker_runs_all_five_and_no_other_head(self):
        values = self.worker_fixture()
        self.assertEqual(values, ([10, 11, 12, 13, 14],) * 3)

    def test_worker_failure_stops_without_silent_retry(self):
        self.assertEqual(self.worker_fixture(fail_seed=11), ([10, 11], [10], [10]))

    def test_changed_worker_scope_rejected_before_gpu_setup(self):
        self.assertEqual(self.worker_fixture(bad_jobs=True), ([], [], []))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    ORIGINAL = args.package.resolve()
    sys.path.insert(0, str(ORIGINAL))
    args.output.mkdir(parents=True, exist_ok=False)
    buffer = io.StringIO()
    result = unittest.TextTestRunner(stream=buffer, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(SupplementTests)
    )
    report = buffer.getvalue()
    (args.output / "results.txt").write_bytes(report.encode("utf8"))
    (args.output / "results.json").write_text(
        json.dumps(
            dict(
                success=result.wasSuccessful(),
                tests=result.testsRun,
                failures=len(result.failures),
                errors=len(result.errors),
                skipped=len(result.skipped),
                report_sha256=hashlib.sha256(report.encode("utf8")).hexdigest(),
                training_calls=0,
                Qwen_calls=0,
            ),
            indent=2,
        )
        + "\n",
        encoding="utf8",
    )
    print(report)
    raise SystemExit(0 if result.wasSuccessful() else 1)

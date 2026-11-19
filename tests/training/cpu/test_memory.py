from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from threadpoolctl import threadpool_limits
from training.cpu import common as c
from training.cpu import memory as memory
from tests.training.cpu import test_training as original_tests


class MemoryExecution(unittest.TestCase):
    def test_default_changed_explicit_inner_guard_preserved(self):
        with patch.object(
            c, "resources", return_value=dict(available_GiB=4.0, free_disk_GiB=30.0)
        ):
            memory.measured_guard()
            with self.assertRaisesRegex(RuntimeError, "available RAM"):
                memory.LEGACY_GUARD()
        with patch.object(
            c, "resources", return_value=dict(available_GiB=1.1, free_disk_GiB=3.0)
        ):
            memory.measured_guard(1.0, 2.0)
            with self.assertRaisesRegex(RuntimeError, "available RAM"):
                memory.measured_guard()

    def test_root_and_guard_scoped_without_scientific_function_change(self):
        guard, base = c.resource_guard, c.BASE
        functions = (
            c.fit_text_features,
            c.fit_linear,
            c.selection,
            memory.run.worker,
            memory.run.assemble,
        )
        with memory.execution_context():
            self.assertEqual(c.BASE, memory.BASE)
            self.assertIs(c.resource_guard, memory.measured_guard)
            self.assertEqual(
                functions,
                (
                    c.fit_text_features,
                    c.fit_linear,
                    c.selection,
                    memory.run.worker,
                    memory.run.assemble,
                ),
            )
        self.assertIs(c.resource_guard, guard)
        self.assertEqual(c.BASE, base)

    def test_low_memory_never_spawns_worker(self):
        with patch.object(
            subprocess, "Popen", side_effect=AssertionError("Must not start a worker")
        ):
            with self.assertRaisesRegex(RuntimeError, "3.5 GiB"):
                memory.monitored_command(
                    ["unused"],
                    memory.BASE / "qa/never_created",
                    available_reader=lambda: 3.4,
                )

    def test_live_owned_worker_stop_does_not_stop_unrelated_process(self):
        with tempfile.TemporaryDirectory(
            dir=c.BASE / "qa", prefix="watchdog_fixture_"
        ) as temp:
            with patch.object(c, "BASE", Path(temp)):
                unrelated = subprocess.Popen(
                    [sys.executable, "-B", "-c", "import time; time.sleep(30)"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                )
                reads = iter([8.0, 0.5])
                try:
                    with self.assertRaisesRegex(RuntimeError, "CPU worker stopped"):
                        memory.monitored_command(
                            [sys.executable, "-B", "-c", "import time; time.sleep(30)"],
                            Path(temp) / "owned",
                            available_reader=lambda: next(reads, 0.5),
                        )
                    self.assertIsNone(unrelated.poll())
                    rec = c.read(Path(temp) / "owned/resources.json")
                    self.assertEqual(rec["stop_reason"], "available_RAM_below_1.25_GiB")
                    self.assertIsNotNone(rec["exit_code"])
                finally:
                    unrelated.terminate()
                    unrelated.wait(timeout=10)

    def test_live_success_and_nonzero_worker_fail_preserve_log(self):
        with tempfile.TemporaryDirectory(
            dir=c.BASE / "qa", prefix="watchdog_exit_"
        ) as temp:
            with patch.object(c, "BASE", Path(temp)):
                good = memory.monitored_command(
                    [sys.executable, "-B", "-c", 'print("synthetic success")'],
                    Path(temp) / "good",
                    available_reader=lambda: 8.0,
                )
                self.assertEqual(good["exit_code"], 0)
                with self.assertRaisesRegex(RuntimeError, "worker error"):
                    memory.monitored_command(
                        [
                            sys.executable,
                            "-B",
                            "-c",
                            'raise RuntimeError("synthetic failure")',
                        ],
                        Path(temp) / "bad",
                        available_reader=lambda: 8.0,
                    )
                self.assertIn(
                    "synthetic failure",
                    (Path(temp) / "bad/worker.log").read_text(encoding="utf8"),
                )

    def test_policy_stops_memory_growth_and_technical_timeout(self):
        self.assertIsNone(memory.stop_reason(1.25, 2.6, 600, 600))
        self.assertEqual(
            memory.stop_reason(8.0, 2.7, 1), "CPU_worker_private_memory_above_2.6_GiB"
        )
        self.assertEqual(
            memory.stop_reason(8.0, 1.0, 601, 600), "technical_test_time_limit"
        )


def main():
    memory.BASE.joinpath("qa").mkdir(parents=True, exist_ok=True)
    suite = unittest.TestSuite(
        [
            unittest.defaultTestLoader.loadTestsFromModule(original_tests),
            unittest.defaultTestLoader.loadTestsFromTestCase(MemoryExecution),
        ]
    )
    with memory.execution_context(), c.offline(), threadpool_limits(limits=1):
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
        sources = [
            Path(__file__),
            Path(memory.__file__),
            Path(c.__file__),
            Path(memory.run.__file__),
            Path(original_tests.__file__),
            memory.ADDENDUM,
            c.ROOT / "scripts/run_cpu_baselines_memory.ps1",
        ]
        c.save(
            memory.BASE / "qa/tests.json",
            dict(
                successful=True,
                tests_run=result.testsRun,
                failures=0,
                errors=0,
                sources={c.rel(p): c.sha(p) for p in sources},
                utc=c.utc(),
                scope="Inherited scientific/restart/corruption fixtures and real isolated watchdog process tests.",
                scientific_fits=0,
                Qwen_calls=0,
                CUDA_used=False,
            ),
        )


if __name__ == "__main__":
    main()

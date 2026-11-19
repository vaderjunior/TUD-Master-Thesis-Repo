"""Tests for banks resume."""

import unittest
from unittest.mock import patch
from analysis.knowledge.banks import resume as resume


def info(missing=0, invalid=1, n=299):
    return {
        "written": n - missing,
        "expected": n,
        "missing": missing,
        "invalid_results": invalid,
        "items_with_fewer_than_three_valid_votes": invalid,
    }


class CompletionPolicyTests(unittest.TestCase):

    def test_one_recorded_failure_does_not_block_attempted_run(self):
        result = resume.quality(info())
        self.assertEqual(result["valid_results"], 298)
        self.assertEqual(result["invalid_results"], 1)
        self.assertEqual(result["status"], "complete_with_recorded_failures")

    def test_missing_and_entirely_invalid_runs_still_stop(self):
        for value in (info(missing=1), info(invalid=299)):
            with self.assertRaises(RuntimeError):
                resume.quality(value)

    def test_zero_failures_is_a_clean_completion(self):
        self.assertEqual(resume.quality(info(invalid=0))["status"], "complete")

    def test_skip_first_completed_run_and_execute_remaining_eleven(self):
        runs = [
            {"name": f"run{i}", "manifest": f"run{i}.yaml", "output": f"run{i}.jsonl"}
            for i in range(12)
        ]
        done = {"run0"}

        def check(plan, run):
            return info(
                missing=0 if run["name"] in done else 299,
                invalid=1 if run["name"] == "run0" else 0,
            )

        def execute(command, label):
            self.assertEqual(command[:2], ["-m", "scripts.run_comparison"])
            done.add(label)

        with patch.object(resume.core, "validate_plan"), patch.object(
            resume.core, "check_output", side_effect=check
        ), patch.object(
            resume.core, "subprocess_logged", side_effect=execute
        ) as launch, patch.object(
            resume.core, "status_write"
        ), patch.object(
            resume, "write_quality"
        ) as report:
            completed = resume.run_queue({"runs": runs})
        self.assertEqual(launch.call_count, 11)
        self.assertNotIn("run0", [c.args[1] for c in launch.call_args_list])
        self.assertEqual(report.call_count, 12)
        self.assertEqual(completed[0]["invalid_results"], 1)
        self.assertEqual(len(completed), 12)

    def test_provenance_failure_stops_before_launching(self):
        with patch.object(resume.core, "validate_plan"), patch.object(
            resume.core, "check_output", side_effect=RuntimeError("Wrong KB stamp")
        ), patch.object(resume.core, "subprocess_logged") as launch:
            with self.assertRaisesRegex(RuntimeError, "Wrong KB"):
                resume.run_queue({"runs": [{"name": "bad"}]})
            launch.assert_not_called()

    def test_missing_predictions_get_only_two_bounded_passes(self):
        run = {"name": "partial", "manifest": "m.yaml", "output": "r.jsonl"}
        with patch.object(resume.core, "validate_plan"), patch.object(
            resume.core, "check_output", return_value=info(missing=1)
        ), patch.object(resume.core, "subprocess_logged") as launch, patch.object(
            resume.core, "status_write"
        ):
            with self.assertRaisesRegex(RuntimeError, "Missing"):
                resume.run_queue({"runs": [run]})
            self.assertEqual(launch.call_count, 2)


if __name__ == "__main__":
    unittest.main()

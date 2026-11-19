import copy
import unittest
import numpy as np
from analysis.classification.supervised import resource_summary as a
from analysis.classification.supervised import archive as r


class ResourceBoundaries(unittest.TestCase):
    def test_no_parallel_sum_as_wall(self):
        self.assertEqual(a.sequence([(0, 5), (8, 12)])["active_seconds"], 9)
        self.assertEqual(a.sequence([(0, 5), (8, 12)])["calendar_seconds"], 12)

    def test_overlap_rejected(self):
        with self.assertRaises(Exception):
            a.sequence([(0, 5), (4, 10)])

    def test_negative_time_rejected(self):
        with self.assertRaises(Exception):
            a.sequence([(4, 0)])

    def test_timezone_required(self):
        with self.assertRaises(Exception):
            a.stamp("2026-11-01T02:30:00")

    def test_timezone_conversion(self):
        self.assertEqual(
            a.stamp("2026-11-01T02:00:00+01:00"), a.stamp("2026-11-01T01:00:00+00:00")
        )

    def test_missing_timer_is_not_timed_fit(self):
        empty = dict(
            state="empty_held_fold",
            **{
                k: 0
                for k in (
                    "planned_steps",
                    "successful_steps",
                    "skipped_steps",
                    "positive_updates",
                    "accepted_zero_rate_steps",
                )
            }
        )
        out = a.summarize([empty])
        self.assertEqual(out["untimed_slots"], 1)
        self.assertEqual(out["timed_fits"], 0)

    def test_wrong_timer_boundary_rejected(self):
        with self.assertRaises(Exception):
            a.summarize([dict(total_seconds=1, train_seconds=2)])

    @staticmethod
    def samples():
        x = np.zeros((4, 14))
        x[:, 0] = [0, 2, 20, 22]
        x[:, 1] = [10, 99, 30, 0]
        x[:, 3] = 100
        x[:, 6] = 100
        x[:, 9:] = -1
        return x

    def test_long_gap_not_interpolated(self):
        out = a.telemetry_stats([self.samples()])
        self.assertEqual(out["weighted_covered_seconds"], 4)
        self.assertEqual(out["ignored_long_gap_seconds"], 18)

    def test_time_weight_boundary(self):
        out = a.telemetry_stats([self.samples()])
        self.assertEqual(out["weighted_gpu_busy_pct"], 20)

    def test_phase_boundaries_not_joined(self):
        x = self.samples()[:2]
        y = x.copy()
        y[:, 0] += 100
        out = a.telemetry_stats([x, y])
        self.assertEqual(out["window_seconds"], 4)

    def test_memory_limit_not_busy_percent(self):
        x = self.samples()
        x[:, 1] = 100
        x[:, 2] = 20
        x[:, 5] = 79
        out = a.telemetry_stats([x])
        self.assertEqual(out["gpu_memory_over_80pct_samples"], 0)
        self.assertEqual(out["ram_over_80pct_samples"], 0)

    def test_missing_storage_not_zero_observation(self):
        out = a.telemetry_stats([self.samples()])
        self.assertEqual(out["storage_samples"], 0)
        self.assertNotIn("max_outputs_bytes", out)


ARCHIVE_OUTPUT = a.OUT


class SavedEvidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not all(
            (ARCHIVE_OUTPUT / name).is_file()
            for name in (
                "fit_accounting.json",
                "storage_summary.json",
                "timeline_summary.json",
            )
        ):
            raise unittest.SkipTest(
                "Original GPU fit/resource analysis archive is absent"
            )
        cls.f = r.read(ARCHIVE_OUTPUT / "fit_accounting.json")
        cls.s = r.read(ARCHIVE_OUTPUT / "storage_summary.json")
        cls.t = r.read(ARCHIVE_OUTPUT / "timeline_summary.json")

    def test_cv_roster(self):
        self.assertEqual(
            self.f["phases"]["search"]["total"]["slots"], 9 * 4 * 3 * 7 * 2 * 3 * 5
        )

    def test_timed_plus_empty(self):
        z = self.f["phases"]["search"]["total"]
        self.assertEqual(z["timed_fits"] + z["untimed_slots"], z["slots"])

    def test_steps_partition(self):
        for z in self.f["phases"].values():
            x = z["total"]
            self.assertEqual(
                x["planned_steps"], x["successful_steps"] + x["skipped_steps"]
            )
            self.assertEqual(
                x["successful_steps"],
                x["positive_updates"] + x["accepted_zero_rate_steps"],
            )

    def test_final_reuse(self):
        self.assertEqual(
            540 - sum(x["assignments"] - 1 for x in self.f["reused_assignments"]), 530
        )

    def test_inference_count_and_missing_timer(self):
        self.assertEqual(self.f["inference_head_rows"], 530 * 19082)
        self.assertIsNone(self.f["standalone_inference_seconds"])

    def test_subsets_not_new_runs(self):
        z = self.f["executed_search_subsets"]
        self.assertEqual(
            z["one_rate_all_durations_losses"]["slots"] + z["extra_two_rates"]["slots"],
            22680,
        )

    def test_historical_attempts_not_canonical(self):
        self.assertEqual(len(self.f["historical_attempts"]), 16)
        self.assertTrue(
            all(
                not x["used_in_scientific_OOF_or_final_prediction"]
                for x in self.f["historical_attempts"]
            )
        )

    def test_calendar_reconciles(self):
        z = self.t["summary"]
        self.assertAlmostEqual(
            z["active_seconds"] + z["gap_seconds"], z["calendar_seconds"], places=5
        )

    def test_clock_agreement(self):
        self.assertLess(
            abs(self.t["summary"]["timestamp_vs_monotonic_difference_seconds"]), 0.1
        )

    def test_storage_duplicate_accounting(self):
        self.assertEqual(
            self.s["unique_logical_bytes"] + self.s["duplicate_metadata_bytes"],
            self.s["manifest_entry_bytes"],
        )

    def test_weight_plus_record_total(self):
        self.assertEqual(
            self.s["weight_bytes"] + self.s["nonweight_unique_logical_bytes"],
            self.s["unique_logical_bytes"],
        )

    def test_archived_payload(self):
        self.assertEqual(self.s["total_payload_bytes"], 271266277045)

    def test_zip_direct_inventory(self):
        self.assertEqual(self.s["files"] - self.s["direct_weight_files"], 522087)

    def test_decimal_cap_distinct_from_gib(self):
        self.assertLess(250000000000 / 2**30, 250)

    def test_pipeline_deployment_not_all_repeats(self):
        self.assertEqual(len(self.f["pipeline_summary"]), 9)
        self.assertTrue(
            all(2e9 < x["mean_model_bytes"] < 2.1e9 for x in self.f["pipeline_summary"])
        )


if __name__ == "__main__":
    with r.offline():
        unittest.main()

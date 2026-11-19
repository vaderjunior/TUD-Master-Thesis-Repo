from copy import deepcopy
import unittest
from analysis.feedback.reserved.resources_metrics import Group
from analysis.feedback.reserved.resources_metrics import accounting_check
from analysis.feedback.reserved.resources_metrics import attempt_partition
from analysis.feedback.reserved.resources_metrics import seconds
from analysis.feedback.reserved.resources_metrics import quantile


def row():
    return {
        "accounting": {
            "ordinary_logical_calls": 3,
            "repair_logical_calls": 1,
            "http_success_with_text": 4,
            "post_attempts": 6,
            "retry_attempts": 2,
            "retryable_failures": 2,
        },
        "observed_usage": {
            "prompt_tokens": 40,
            "completion_tokens": 8,
            "total_tokens": 48,
        },
        "ordinary_usage": {
            "prompt_tokens": 30,
            "completion_tokens": 6,
            "total_tokens": 36,
        },
        "repair_usage": {
            "prompt_tokens": 10,
            "completion_tokens": 2,
            "total_tokens": 12,
        },
        "known_request_durations_s": [1.0, 2.0, 3.0, 4.0, 10.0],
        "known_request_seconds_sum": 20.0,
        "known_request_duration_count": 5,
        "administrative_unknown_requests": 1,
        "retry_backoff_seconds_sum": 2.0,
        "result": None,
        "n_valid": 0,
        "parse_failures": 3,
        "repairs": 1,
    }


class ResourceTests(unittest.TestCase):
    def test_unknown_and_null_are_retained(self):
        g = Group()
        g.add(row())
        z = g.finish()
        self.assertEqual(z["known_request_duration_count"], 5)
        self.assertEqual(z["counts"]["nulls"], 1)
        self.assertEqual(z["per_planned_prediction"]["observed_tokens"], 48)
        self.assertEqual(z["per_planned_prediction"]["attempt_slots"], 6)

    def test_unknown_zero_is_not_a_known_duration(self):
        r = row()
        r["known_request_durations_s"].append(0.0)
        with self.assertRaises(ValueError):
            Group().add(r)

    def test_repair_is_not_added_twice(self):
        r = row()
        r["observed_usage"]["total_tokens"] += 12
        with self.assertRaises(ValueError):
            Group().add(r)
        r = row()
        r["accounting"]["post_attempts"] += 1
        with self.assertRaises(ValueError):
            accounting_check(r["accounting"])

    def test_negative_or_nonfinite_duration_rejected(self):
        for bad in [-1, float("nan"), float("inf")]:
            r = row()
            r["known_request_durations_s"][0] = bad
            with self.assertRaises(ValueError):
                Group().add(r)

    def test_repeated_predictions_accumulate_not_amortize(self):
        g = Group()
        g.add(row())
        g.add(deepcopy(row()))
        z = g.finish()
        self.assertEqual(z["observed_usage"]["total_tokens"], 96)
        self.assertEqual(z["per_planned_prediction"]["denominator"], 2)
        self.assertEqual(z["per_planned_prediction"]["observed_tokens"], 48)

    def test_timezone_and_interval_guard(self):
        self.assertEqual(
            seconds("2026-10-12T12:00:00+02:00", "2026-10-12T11:00:00+00:00"), 3600
        )
        with self.assertRaises(ValueError):
            seconds("2026-10-12T12:00:00", "2026-10-12T13:00:00")
        with self.assertRaises(ValueError):
            seconds("2026-10-12T12:00:00+00:00", "2026-10-12T11:00:00+00:00")

    def test_unknown_shutdown_window_stays_unallocated(self):
        t = lambda h: f"2026-10-12T{h:02}:00:00+00:00"
        r = attempt_partition(
            [
                {"start": t(1), "model_check_start": t(2)},
                {"start": t(5), "model_check_start": t(6), "end": t(8)},
            ],
            t(8),
        )
        self.assertEqual(r["seconds_by_kind"]["unallocated_interrupted_window"], 10800)
        self.assertNotIn("explicit_between_attempt_gap", r["seconds_by_kind"])

    def test_explicit_gap_is_not_request_time(self):
        t = lambda h: f"2026-10-12T{h:02}:00:00+00:00"
        r = attempt_partition(
            [
                {"start": t(1), "model_check_start": t(2), "end": t(3)},
                {"start": t(5), "model_check_start": t(6), "end": t(8)},
            ],
            t(8),
        )
        self.assertEqual(r["seconds_by_kind"]["explicit_between_attempt_gap"], 7200)
        self.assertEqual(sum(r["seconds_by_kind"].values()), 25200)

    def test_quantile_definition(self):
        self.assertEqual(quantile([4, 1, 3, 2], 0.5), 2.5)
        self.assertEqual(quantile([1], 0.99), 1)


if __name__ == "__main__":
    unittest.main()

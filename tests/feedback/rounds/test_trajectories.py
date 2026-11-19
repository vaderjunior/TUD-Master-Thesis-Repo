import unittest
import numpy as np
from analysis.feedback.rounds import trajectories_metrics as k


class TrajectoryBoundaries(unittest.TestCase):
    def test_stable_fix(self):
        p = k.path_profile([1, 0, 0, 0, 0])
        self.assertEqual(p["initial_fix_state"], "retained_all_later")
        self.assertEqual(p["fixes"], [1])

    def test_reversed_fix(self):
        self.assertEqual(
            k.path_profile([1, 0, 1, 0, 1])["initial_fix_state"], "lost_by_r4"
        )

    def test_recovered_fix(self):
        self.assertEqual(
            k.path_profile([1, 0, 1, 1, 0])["initial_fix_state"], "recovered_by_r4"
        )

    def test_last_round_censoring(self):
        self.assertEqual(
            k.event_lifetime([1, 1, 1, 1, 0], 4, "fix"),
            dict(state="uninterrupted", future_rounds=0, censored_at_horizon=True),
        )

    def test_later_reversal_lifetime(self):
        self.assertEqual(
            k.event_lifetime([0, 1, 0, 1, 0], 2, "fix")["state"], "returned_by_r4"
        )

    def test_events_are_not_unique_inputs(self):
        self.assertEqual(k.path_profile([1, 0, 1, 0, 1])["fixes"], [1, 3])
        self.assertEqual(k.path_profile([1, 0, 1, 0, 1])["new_errors"], [2, 4])

    def test_missing_not_error(self):
        self.assertEqual(k.path_profile([1, 0, -1, 0, 0])["status"], "missing")
        self.assertEqual(k.transition(-1, 0), "missing")
        with self.assertRaises(ValueError):
            k.event_lifetime([1, 0, -1, 0, 0], 1, "fix")

    def test_transition_net_identity(self):
        s = k.transition_summary([[1, 0, 1], [0, 0, 1]], [[0, 1, 0], [1, 0, 1]])
        self.assertEqual(
            s["new_errors"] - s["old_errors"],
            s["counts"]["new_error"] - s["counts"]["fix"],
        )
        self.assertEqual(s["inputs_with_both"], 1)

    def test_equal_average_can_hide_opposing_changes(self):
        s = k.transition_summary([[1, 0, 0]], [[0, 1, 0]])
        self.assertEqual((s["equal_mean_inputs"], s["inputs_with_both"]), (1, 1))

    def test_final_origins_partition(self):
        paths = {
            "always_wrong_since_r1": [0, 1, 1, 1, 1],
            "recurrent_error": [0, 1, 0, 1, 1],
            "new_error_since_r1": [1, 0, 0, 1, 1],
            "always_correct_since_r1": [1, 0, 0, 0, 0],
            "retained_later_fix": [1, 1, 1, 0, 0],
            "recovered_after_reversal": [1, 1, 0, 1, 0],
            "recovered_new_error": [0, 0, 1, 1, 0],
        }
        for name, p in paths.items():
            self.assertEqual(k.path_profile(p)["final_origin"], name)

    def test_mean_advantage_is_not_strict_fix(self):
        e = np.zeros((4, 9), dtype=int)
        e[:, 0] = 1
        p = k.input_profile(e, np.ones((4, 3), dtype=int))
        self.assertFalse(p["durable_initial_fix"])
        self.assertEqual(p["contemporaneous"]["pattern"], "BBBB")

    def test_strict_reversal_recovers(self):
        e = np.zeros((4, 9), dtype=int)
        e[1, 4] = 1
        p = k.input_profile(e, np.ones((4, 3), dtype=int))
        self.assertTrue(p["initial_fix_reversal"] and p["initial_fix_recovered"])
        self.assertFalse(p["initial_fix_lost_at_r4"])

    def test_late_new_error_needs_followup(self):
        e = np.zeros((4, 9), dtype=int)
        e[3, :] = 1
        self.assertFalse(
            k.input_profile(e, np.zeros((4, 3), dtype=int))["sustained_new_error"]
        )
        e[2, :] = 1
        self.assertTrue(
            k.input_profile(e, np.zeros((4, 3), dtype=int))["sustained_new_error"]
        )

    def test_changing_reference_changes_relative_path(self):
        e = np.zeros((4, 9), dtype=int)
        r = np.ones((4, 3), dtype=int)
        r[2, :] = 0
        p = k.input_profile(e, r)
        self.assertEqual(p["contemporaneous"]["pattern"], "BB=B")
        self.assertEqual(p["initial_reference"]["pattern"], "BBBB")

    def test_alignment_range_and_partition(self):
        e = np.array([[[0, 1, 1], [0, 1, 1], [0, 1, 1], [0, 1, 1]]])
        r = np.ones((1, 3), dtype=int)
        rows = k.alignment_counts(r, e)
        self.assertEqual(len(rows), 216)
        self.assertEqual(
            (
                min(x["retained_all_later"] for x in rows),
                max(x["retained_all_later"] for x in rows),
            ),
            (0, 1),
        )
        self.assertTrue(
            all(
                x["initial_fixes"]
                == x["retained_all_later"] + x["lost_by_r4"] + x["recovered_by_r4"]
                for x in rows
            )
        )

    def test_no_case_replacement_for_missing(self):
        self.assertEqual(k.candidate_cases({"status": "missing"}, True), [])

    def test_hash_case_order_deterministic(self):
        self.assertEqual(
            k.case_rank("en_test", "durable_initial_fix", "a"),
            k.case_rank("en_test", "durable_initial_fix", "a"),
        )
        self.assertNotEqual(
            k.case_rank("en_test", "durable_initial_fix", "a"),
            k.case_rank("en_test", "durable_initial_fix", "b"),
        )


if __name__ == "__main__":
    unittest.main()

import copy
import unittest
from unittest.mock import patch
from analysis.common import execution_history as h


class HistoryTests(unittest.TestCase):

    def setUp(self):
        self.jobs = [{"key": str(i)} for i in range(2700)]
        self.summary = {
            "submitted": 2692,
            "saved_items": 2700,
            "unsubmitted": 0,
            "errors": [],
            "stopped": False,
        }
        self.mock = patch.object(h.b, "identity", lambda x: x)
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def test_clean_partition_accepted(self):
        h.validate_partition(self.jobs, self.jobs[:8], self.summary)

    def test_resubmitted_saved_or_omitted_new_jobs_rejected(self):
        for value in [2700, 2691]:
            s = {**self.summary, "submitted": value}
            with self.assertRaises(h.a.IntegrityError):
                h.validate_partition(self.jobs, self.jobs[:8], s)

    def test_wrong_preserved_set_or_duplicate_roster_rejected(self):
        with self.assertRaises(h.a.IntegrityError):
            h.validate_partition(self.jobs, self.jobs[1:9], self.summary)
        jobs = copy.deepcopy(self.jobs)
        jobs[-1] = jobs[0]
        with self.assertRaises(h.a.IntegrityError):
            h.validate_partition(jobs, jobs[:8], self.summary)

    def test_incomplete_or_stopped_summary_rejected(self):
        for key, value in [
            ("saved_items", 2699),
            ("unsubmitted", 1),
            ("stopped", True),
            ("errors", ["error"]),
        ]:
            with self.assertRaises(h.a.IntegrityError):
                h.validate_partition(
                    self.jobs, self.jobs[:8], {**self.summary, key: value}
                )

    def test_completion_epoch_cannot_reclassify_old_as_new(self):
        early = "2026-09-19T07:21:00+00:00"
        resume = "2026-09-19T07:24:37+00:00"
        late = "2026-09-19T07:25:00+00:00"
        h.validate_epoch(True, early, resume)
        h.validate_epoch(False, late, resume)
        for preserved, stamp in [(False, early), (True, late)]:
            with self.assertRaises(h.a.IntegrityError):
                h.validate_epoch(preserved, stamp, resume)


if __name__ == "__main__":
    unittest.main()

"""Tests for sampled banks storage."""

import struct
import unittest
from analysis.knowledge.sampled_banks import storage as s


class StorageAmendmentTests(unittest.TestCase):

    def test_expected_empty_header(self):
        self.assertEqual(
            s.compact_header(struct.pack(s.HEADER_FORMAT, *s.EMPTY_HEADER))["persisted_elements"], 0
        )

    def test_populated_header_cannot_use_exception(self):
        for index, value in ((3, 1), (2, 200), (7, 0), (0, 2)):
            fields = list(s.EMPTY_HEADER)
            fields[index] = value
            with self.assertRaises(RuntimeError):
                s.compact_header(struct.pack(s.HEADER_FORMAT, *fields))

    def test_truncated_header_refused(self):
        with self.assertRaises(RuntimeError):
            s.compact_header(b"bad")

    def test_only_capacity_files_may_differ(self):
        s.changed_file_policy(["chroma.sqlite3", str(s.Path("id") / "length.bin")], "id", 340)
        for name in ("header.bin", "link_lists.bin", "index_metadata.pickle"):
            with self.assertRaises(RuntimeError):
                s.changed_file_policy([str(s.Path("id") / name)], "id", 340)

    def test_larger_pool_cannot_use_exception(self):
        with self.assertRaises(RuntimeError):
            s.changed_file_policy(["chroma.sqlite3"], "id", 2040)

    def test_sqlite_lock_addition_only(self):
        s.lock_table_policy([(1, 1)], [(1, 1), (2, 1)])
        for wrong in ([], [(1, 0)], [(2, 1)], [(1, 1), (2, 1), (3, 1)]):
            with self.assertRaises(RuntimeError):
                s.lock_table_policy([(1, 1)], wrong)


if __name__ == "__main__":
    unittest.main(verbosity=2)

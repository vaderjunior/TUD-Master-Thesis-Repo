import io
import socket
import unittest
import numpy as np
from analysis.classification.combined import prepare as a


class Boundaries(unittest.TestCase):
    def test_language_prefix_is_explicit(self):
        self.assertEqual(a.key(dict(id="x", lang="en")), "en/x")
        self.assertNotEqual(
            a.key(dict(id="x", lang="en")), a.key(dict(id="x", lang="de"))
        )
        with self.assertRaises(a.g.IntegrityError):
            a.key(dict(id="x"))

    def test_canonical_alias_collision_rejected(self):
        with self.assertRaises(a.g.IntegrityError):
            a.index([dict(id="x", lang="en"), dict(id="en/x", lang="en")])

    def test_missing_not_empty(self):
        self.assertIsNone(a.labels(dict(gate=True, target_groups=None), "target_group"))
        self.assertEqual(
            a.labels(dict(gate=True, target_groups=[]), "target_group"), set()
        )

    def test_gold_gate_masks_fine_head(self):
        self.assertIsNone(a.labels(dict(gate=False, hate_types=["irony"]), "hate_type"))
        self.assertIsNone(a.labels(dict(gate=None, hate_types=["irony"]), "hate_type"))
        self.assertEqual(
            a.labels(dict(gate=True, hate_types=["irony"]), "hate_type"), {"irony"}
        )

    def test_legal_is_not_hate_gated(self):
        self.assertEqual(a.labels(dict(gate=False, legal=[]), "legal"), set())
        self.assertEqual(
            a.labels(dict(gate=None, legal=["propaganda"]), "legal"), {"propaganda"}
        )

    def test_boolean_gate_not_truthiness(self):
        for invalid in (1, "true", []):
            with self.assertRaises(a.g.IntegrityError):
                a.labels(dict(gate=invalid), "hate")
        self.assertEqual(a.labels(dict(gate=False), "hate"), {"not_hate"})

    def test_array_gold_and_duplicate_labels(self):
        self.assertEqual(
            a.labels(
                dict(gate=np.bool_(True), hate_types=np.array(["irony", "irony"])),
                "hate_type",
            ),
            {"irony"},
        )

    def test_string_is_not_multilabel_list(self):
        with self.assertRaises(a.g.IntegrityError):
            a.labels(dict(gate=True, target_groups="race"), "target_group")

    def test_text_normalization_keeps_language_identity_separate(self):
        self.assertEqual(a.normalized_sha(" Ａ\tB "), a.normalized_sha("a b"))
        self.assertNotEqual(a.normalized_sha("a b"), a.normalized_sha("a 0"))

    def test_support_is_fixed_before_validity(self):
        rr = [
            dict(
                id=str(i),
                lang="en",
                gate=True,
                hate_types=["irony"] if i < 10 else ["explicit"],
            )
            for i in range(19)
        ]
        ep = dict(
            dimension="hate_type",
            all_labels=["explicit", "irony"],
            macro_labels=["irony"],
            gold_support={"explicit": 9, "irony": 10},
            scorable_ids=[str(i) for i in range(19)],
        )
        v = a.registry("fixture", rr, ep)
        self.assertEqual(v["macro_labels"], ["irony"])
        changed = dict(ep, macro_labels=["explicit", "irony"])
        with self.assertRaises(a.g.IntegrityError):
            a.registry("fixture", rr, changed)

    def test_changed_annotation_order_is_not_id_alignment(self):
        rr = [dict(id=str(i), lang="en", gate=bool(i % 2)) for i in range(20)]
        ep = dict(
            dimension="hate",
            all_labels=["not_hate", "hate"],
            macro_labels=["not_hate", "hate"],
            gold_support={"not_hate": 10, "hate": 10},
            scorable_ids=[str(i) for i in reversed(range(20))],
        )
        with self.assertRaises(a.g.IntegrityError):
            a.registry("fixture", rr, ep)

    def test_common_nine_differs_from_pair_six(self):
        valid = np.ones((9, 4), dtype=bool)
        valid[8, 1] = False
        valid[0, 2] = False
        self.assertEqual(valid.all(0).tolist(), [True, False, False, True])
        self.assertEqual(valid[:6].all(0).tolist(), [True, True, False, True])

    def test_row_and_column_permutations_must_both_align(self):
        gold = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.uint8)
        permuted = gold[[1, 0]][:, [2, 0, 1]]
        self.assertFalse(np.array_equal(gold, permuted))
        self.assertTrue(np.array_equal(gold, permuted[[1, 0]][:, [1, 2, 0]]))

    def test_network_blocked(self):
        with a.g.offline():
            with self.assertRaises(a.g.IntegrityError):
                socket.create_connection(("localhost", 9))


if __name__ == "__main__":
    out = io.StringIO()
    result = unittest.TextTestRunner(stream=out, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(Boundaries)
    )
    print(out.getvalue())
    a.require(result.wasSuccessful(), "Boundary fixtures failed")
    a.save(
        a.BASE / "tests.json",
        dict(
            status="scientific_boundary_fixtures_passed",
            tests=result.testsRun,
            failures=len(result.failures),
            errors=len(result.errors),
            code_sha256=a.g.sha(__file__),
            output=out.getvalue(),
        ),
    )

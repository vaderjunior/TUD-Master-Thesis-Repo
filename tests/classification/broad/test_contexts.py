import unittest
import numpy as np
from analysis.classification.broad import contexts as x


class ContextTests(unittest.TestCase):
    def test_unknown_is_not_empty(self):
        a = x.annotations(
            [
                {"meta": {}},
                {"meta": {"target_groups": None}},
                {"meta": {"target_groups": []}},
            ]
        )
        self.assertEqual(a["target_group"]["known"], 1)
        self.assertEqual(a["target_group"]["empty"], 1)

    def test_gate_false_is_annotated(self):
        a = x.annotations(
            [{"meta": {"gate": False}}, {"meta": {"gate": True}}, {"meta": {}}]
        )
        self.assertEqual((a["hate"], a["nonhate"], a["gate_unknown"]), (1, 1, 1))

    def test_legal_only_not_negative_gate(self):
        a = x.annotations([{"meta": {"legal": ["propaganda"]}}])
        self.assertEqual(a["gate_unknown"], 1)
        self.assertEqual(a["nonhate"], 0)

    def test_empty_legal_is_known(self):
        a = x.annotations([{"meta": {"legal": []}}])
        self.assertEqual((a["legal"]["known"], a["legal"]["positive"]), (1, 0))

    def test_auxiliary_is_not_legal_gold(self):
        a = x.annotations([{"meta": {"astra_aux_stgb": "185"}}])
        self.assertEqual(a["auxiliary_flags"], 1)
        self.assertEqual(a["legal"]["known"], 0)

    def test_illustration_not_annotation(self):
        a = x.annotations(
            [
                {
                    "meta": {
                        "illustrative_only": True,
                        "gate": False,
                        "legal": ["propaganda"],
                    }
                }
            ]
        )
        self.assertEqual(
            (a["regular"], a["illustrative"], a["legal"]["known"]), (0, 1, 0)
        )

    def test_labels_count_slots_not_unique_sets(self):
        a = x.annotations(
            [
                {"meta": {"hate_types": ["irony", "grievance"]}},
                {"meta": {"hate_types": ["irony"]}},
            ]
        )
        self.assertEqual(a["hate_type"]["labels"], {"irony": 2, "grievance": 1})

    def test_severity_string_single_label(self):
        a = x.annotations([{"meta": {"severity": "low"}}])
        self.assertEqual(a["severity"]["labels"], {"low": 1})

    def test_complete_block_requires_actual_annotations(self):
        a = x.annotations(
            [
                {
                    "meta": {
                        "hate_types": [],
                        "target_groups": [],
                        "severity": "low",
                        "legal": [],
                    }
                },
                {
                    "meta": {
                        "hate_types": [],
                        "target_groups": [],
                        "severity": None,
                        "legal": [],
                    }
                },
            ]
        )
        self.assertEqual(a["all_four_fine_known"], 1)

    def test_normalization_preserves_content(self):
        self.assertEqual(x.normalized(" Ａ\t B "), "a b")
        self.assertNotEqual(x.normalized("not hate"), x.normalized("hate"))

    def test_token_similarity_is_not_order(self):
        self.assertEqual(
            x.jaccard(x.tokens("I reject hate"), x.tokens("hate reject I")), 1.0
        )

    def test_empty_tokens(self):
        self.assertEqual(x.jaccard(set(), set()), 0.0)

    def test_overlap_separates_order(self):
        a = dict(
            ids={k: ["a", "b"] for k in ("definitions", "guidelines", "examples")},
            rendered={k: "x" for k in ("definitions", "guidelines", "examples")},
            messages_sha256="a",
        )
        b = dict(
            ids={k: ["b", "a"] for k in ("definitions", "guidelines", "examples")},
            rendered={k: "y" for k in ("definitions", "guidelines", "examples")},
            messages_sha256="b",
        )
        d = x.overlap(a, b)
        self.assertTrue(d["examples"]["same_set"])
        self.assertFalse(d["examples"]["same_order"])
        self.assertEqual(d["examples"]["same_positions"], 0)

    def test_same_render_different_ids(self):
        a = dict(
            ids={k: ["a"] for k in ("definitions", "guidelines", "examples")},
            rendered={k: "x" for k in ("definitions", "guidelines", "examples")},
            messages_sha256="x",
        )
        b = dict(
            ids={k: ["b"] for k in ("definitions", "guidelines", "examples")},
            rendered=a["rendered"],
            messages_sha256="x",
        )
        self.assertTrue(x.overlap(a, b)["same_delivered_context"])
        self.assertEqual(x.overlap(a, b)["examples"]["shared"], 0)

    def test_stable_fix(self):
        self.assertEqual(
            x.binary_state(True, np.array([1, 1, 1, 0, 0, 0, 0, 1, 0], bool)),
            "stable_fix",
        )

    def test_stable_regression(self):
        self.assertEqual(
            x.binary_state(False, np.array([1, 1, 1, 0, 0, 0, 0, 1, 0], bool)),
            "stable_regression",
        )

    def test_stable_correct(self):
        self.assertEqual(x.binary_state(True, np.ones(9, bool)), "stable_correct")

    def test_stable_wrong(self):
        self.assertEqual(x.binary_state(False, np.ones(9, bool)), "stable_wrong")

    def test_variable_not_fix(self):
        self.assertEqual(
            x.binary_state(True, np.array([1, 0, 1, 0, 0, 0, 0, 0, 0], bool)),
            "variable",
        )

    def test_no_data_not_zero(self):
        self.assertIsNone(x.mean([]))
        self.assertIsNone(x.stats([])["mean"])

    def test_renderer_unknown_gate_omitted(self):
        self.assertNotIn("hate=", x.reader.prompt.corrected_gold({"legal": []}))

    def test_renderer_empty_target_shown(self):
        self.assertIn(
            "target_group=[]", x.reader.prompt.corrected_gold({"target_groups": []})
        )

    def test_renderer_absent_target_omitted(self):
        self.assertNotIn("target_group", x.reader.prompt.corrected_gold({}))

    def test_renderer_auxiliary_qualified(self):
        self.assertIn(
            "illustrative, not legal gold",
            x.reader.prompt.corrected_gold({"astra_aux_stgb": "185"}),
        )

    def test_renderer_invalid_gate_rejected(self):
        with self.assertRaises(ValueError):
            x.reader.prompt.corrected_gold({"gate": 1})

    def test_cache_is_identical(self):
        ctx = dict(definitions=[], guidelines=[], example_groups=[])
        old = x.reader.prompt.render_messages("a test", "en", ctx, "corrected")
        x.prepare_renderer()
        self.assertEqual(
            old, x.reader.prompt.render_messages("a test", "en", ctx, "corrected")
        )


if __name__ == "__main__":
    with x.r.original.offline():
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ContextTests)
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        if not result.wasSuccessful():
            raise SystemExit(1)
        x.save(
            x.BASE / "tests.json",
            dict(
                status="passed",
                tests=result.testsRun,
                source_sha256=x.c.sha(__file__),
                new_calls=0,
            ),
        )

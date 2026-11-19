"""Tests for components prepare."""

from copy import deepcopy
import unittest
from analysis.knowledge.components import prepare as p


def fixture():

    def hit(identifier, text, meta):
        return {"id": identifier, "text": text, "meta": meta, "via": "rrf"}

    context = {
        "definitions": [
            hit("def-hate", "Definition one.", {"dimension": "hate", "label": None}),
            hit("def-irony", "Definition two.", {"dimension": "hate_type", "label": "irony"}),
        ],
        "guidelines": [hit("guide-1", "Rule one.", {}), hit("guide-2", "Rule two.", {})],
        "example_groups": [
            [
                "Labelled examples",
                [
                    hit(
                        "ex-" + str(i),
                        "example\n\n" + str(i),
                        {
                            "gate": None if i == 0 else i % 2 == 0,
                            "target_groups": None if i == 0 else [],
                            "hate_types": None,
                            "severity": None,
                            "legal": ["insult_defamation"] if i == 0 else None,
                            "astra_aux_stgb": "185" if i == 1 else None,
                        },
                    )
                    for i in range(5)
                ],
            ]
        ],
    }
    source = {
        "text": "Unmodified target\n## Labelled examples\nwith a heading.",
        "lang": "de",
        "context": context,
    }
    source["messages"] = p.prompt.render_messages(
        source["text"], source["lang"], context, "corrected"
    )
    return source


class ComponentPreparationTests(unittest.TestCase):

    def test_original_bridge_is_exact_and_all_factorial_instructions_match(self):
        source = fixture()
        before = deepcopy(source)
        blocks = p.blocks(source)
        for condition in p.CONDITIONS:
            messages, context = p.render(source, condition, blocks)
            self.assertEqual(messages[0], source["messages"][0])
            self.assertTrue(messages[1]["text"].endswith(blocks["target"]))
            if condition == p.BRIDGE:
                self.assertEqual(messages, source["messages"])
                self.assertEqual(context, source["context"])
            else:
                self.assertTrue(messages[1]["text"].startswith(p.COMMON_NOTE + "\n\n"))
                self.assertEqual(messages[1]["text"].count(p.COMMON_NOTE), 1)
                self.assertNotIn(p.template_parts()[0]["EXAMPLES_CAUTION"], messages[1]["text"])
        self.assertEqual(source, before)

    def test_each_bucket_preserved_or_removed_without_refill(self):
        source = fixture()
        for condition, (e, d, g) in p.ARMS.items():
            messages, ctx = p.render(source, condition)
            for field, flag in [("definitions", d), ("guidelines", g), ("example_groups", e)]:
                self.assertEqual(ctx[field], source["context"][field] if flag else [])
            for marker, flag in [("Definition one.", d), ("Rule one.", g), ("example 0", e)]:
                self.assertEqual(marker in messages[1]["text"], bool(flag))
        messages, _ = p.render(source, "none")
        self.assertEqual(messages[1]["text"], p.COMMON_NOTE + "\n\n" + p.blocks(source)["target"])

    def test_missing_gate_not_false_and_legal_auxiliary_preserved(self):
        message = p.render(fixture(), "e")[0][1]["text"]
        self.assertIn("labels: legal=[insult_defamation]", message)
        self.assertNotIn("labels: hate=false, legal=[insult_defamation]", message)
        self.assertIn(
            "Auxiliary DeTox paragraph flags (illustrative, not legal gold): §185", message
        )
        self.assertIn("target_group=[]", message)

    def test_corrupt_source_unsupported_groups_and_unknown_arm_rejected(self):
        altered = fixture()
        altered["messages"][1]["text"] += "changed"
        with self.assertRaises(p.a.IntegrityError):
            p.blocks(altered)
        altered = fixture()
        altered["context"]["example_groups"][0][1][0]["via"] = "static"
        with self.assertRaises(p.a.IntegrityError):
            p.blocks(altered)
        altered = fixture()
        altered["context"]["definitions"].pop()
        with self.assertRaises(p.a.IntegrityError):
            p.blocks(altered)
        with self.assertRaises(p.a.IntegrityError):
            p.render(fixture(), "unplanned")

    def test_factorial_weights_recover_known_effects_and_interactions(self):
        effects = p.contrasts()["descriptive_factorial"]
        self.assertEqual(len(effects), 7)
        for weights in effects.values():
            self.assertAlmostEqual(sum(weights.values()), 0)
        scores = {k: 0.1 + 0.2 * e + 0.3 * d + 0.4 * g for k, (e, d, g) in p.ARMS.items()}
        values = {
            name: sum((weights[k] * scores[k] for k in weights))
            for name, weights in effects.items()
        }
        for name, expected in [
            ("e", 0.2),
            ("d", 0.3),
            ("g", 0.4),
            ("e_x_d", 0),
            ("e_x_g", 0),
            ("d_x_g", 0),
            ("e_x_d_x_g", 0),
        ]:
            self.assertAlmostEqual(values[name], expected)
        scores = {k: e * d * g for k, (e, d, g) in p.ARMS.items()}
        self.assertEqual(sum((effects["e_x_d_x_g"][k] * scores[k] for k in scores)), 1)
        self.assertEqual(
            p.contrasts()["focal_leave_one_out"]["guidelines_given_examples_definitions"],
            {"edg": 1.0, "ed": -1.0},
        )

    def test_analysis_inherits_masks_but_keeps_scope_names_and_new_roles(self):
        old = p.a.read(p.SOURCE / "plan.json")
        new = p.analysis_plan(old)
        self.assertEqual(new["focal_family_size"], 3)
        self.assertEqual(new["bootstrap_seed"], 20260921)
        self.assertEqual(new["conditions"], list(p.CONDITIONS))
        for subset, row in old["endpoints_inherited_masks_and_labels"].items():
            for key in ("macro_labels", "all_labels", "gold_support", "scorable_ids", "mask"):
                self.assertEqual(new["endpoints"][subset][key], row[key])
        self.assertEqual(
            new["endpoints"]["en_dev_eval_sq3_types"]["source_scope"], "hate_type_primary"
        )
        self.assertEqual(
            new["endpoints"]["en_dev_eval_types"]["source_scope"], "hate_type_secondary"
        )
        self.assertEqual(new["endpoints"]["en_dev_eval_sq3_types"]["role"], "focal")


if __name__ == "__main__":
    unittest.main()

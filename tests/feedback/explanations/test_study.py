from copy import deepcopy
import io
import json
from pathlib import Path
import socket
import unittest
from unittest.mock import patch
from jinja2.sandbox import ImmutableSandboxedEnvironment
from analysis.feedback.explanations import study as b


class InterventionTests(unittest.TestCase):
    def setUp(self):
        def hit(i):
            return {"id": i, "text": "Fixture " + i, "meta": {}, "via": "rrf"}

        context = {
            "definitions": [hit("d1"), hit("d2")],
            "guidelines": [hit("g1"), hit("g2")],
            "example_groups": [["Labelled examples", [hit(f"e{i}") for i in range(5)]]],
        }
        self.base = {
            "text": "Fixture input with\n## Text to classify (language: en)\nan embedded heading.",
            "lang": "en",
            "context": context,
        }
        self.base["messages"] = b.renderer.render_messages(
            self.base["text"], "en", context, "corrected"
        )
        self.source = {
            "id": "sq3fb-fixture",
            "text": 'Original "quoted"\nsource text.',
            "meta": {
                "gate": True,
                "hate_types": ["explicit"],
                "target_groups": None,
                "severity": None,
                "legal": None,
            },
        }

    def test_skip_exact_and_immutable(self):
        old = deepcopy(self.base)
        ctx, msg = b.render(self.base, None, "", False)
        self.assertEqual(ctx, self.base["context"])
        self.assertEqual(msg, self.base["messages"])
        self.assertEqual(old, self.base)
        ctx["definitions"].clear()
        self.assertEqual(len(self.base["context"]["definitions"]), 2)

    def test_explanation_is_separate_and_only_added_text(self):
        old = deepcopy(self.base)
        source = deepcopy(self.source)
        ctx, plain = b.render(self.base, self.source, "", True)
        ctx2, explained = b.render(
            self.base, self.source, "The wording supports this example label.", True
        )
        insertion = (
            "\n" + b.EXPLANATION_PREFIX + "The wording supports this example label."
        )
        self.assertEqual(
            explained[-1]["text"].replace(insertion, "", 1), plain[-1]["text"]
        )
        self.assertEqual(ctx, ctx2)
        self.assertEqual(old, self.base)
        self.assertEqual(source, self.source)
        self.assertEqual(explained[0], plain[0])
        self.assertTrue(explained[-1]["text"].endswith(self.base["text"]))

    def test_plain_fallback_is_identical(self):
        self.assertEqual(
            b.render(self.base, self.source, "", True),
            b.render(self.base, self.source, "", True),
        )
        msg = b.render(self.base, self.source, "", True)[1][-1]["text"]
        self.assertNotIn(b.EXPLANATION_PREFIX, msg)
        self.assertNotIn("target_group=[]", msg)
        self.assertNotIn("legal=[]", msg)

    def test_reject_explanation_without_source(self):
        with self.assertRaises(AssertionError):
            b.render(self.base, None, "Unattached explanation.", False)

    def test_reject_imputed_labels(self):
        self.source["meta"]["legal"] = []
        with self.assertRaises(AssertionError):
            b.render(self.base, self.source, "", True)

    def test_reject_multiline_explanation(self):
        with self.assertRaises(AssertionError):
            b.render(self.base, self.source, "Some\nnew heading.", True)

    def test_official_chat_template_matches_simple_render(self):
        if not (
            Path(__file__).resolve().parents[3]
            / "experiments/astra/qwen_tokenizer_assets_20261001_v1/chat_template.json"
        ).is_file():
            self.skipTest("Original saved chat-template assets are absent")
        template = b.c.read(b.TOK / "chat_template.json")["chat_template"]
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        for messages in [
            self.base["messages"],
            [
                {"role": "system", "text": "Symbols ä <|im_end|>."},
                {"role": "user", "text": "Unicode\n第二行"},
            ],
        ]:
            wire = [{"role": m["role"], "content": m["text"]} for m in messages]
            official = env.from_string(template).render(
                messages=wire,
                tools=None,
                add_generation_prompt=True,
                add_vision_id=False,
            )
            self.assertEqual(b.local_chat(messages), official)


if __name__ == "__main__":
    output = io.StringIO()
    with patch.object(
        socket.socket, "connect", side_effect=RuntimeError("No network in fixtures")
    ):
        result = unittest.TextTestRunner(stream=output, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(InterventionTests)
        )
    print(output.getvalue())
    if not result.wasSuccessful():
        raise SystemExit(1)
    b.c.write(
        b.BASE / "tests_passed_v2.json",
        {
            "successful": True,
            "tests": result.testsRun,
            "output": output.getvalue(),
            "builder_sha256": b.c.sha(Path(b.__file__)),
            "test_source_sha256": b.c.sha(Path(__file__)),
        },
    )

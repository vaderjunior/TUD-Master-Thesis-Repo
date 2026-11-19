import unittest
from analysis.knowledge.banks import english_clean as c
from analysis.knowledge.banks.english_policy import closure


class PolicyTests(unittest.TestCase):

    def test_repost_and_links(self):
        self.assertEqual(
            c.wrapper_key("RT: A long headline about the world! via"),
            c.wrapper_key("A long headline about the world"),
        )
        self.assertEqual(
            c.wrapper_key("A long headline. URL"),
            c.wrapper_key("A long headline https://example.invalid/x"),
        )

    def test_meaning_distinctions(self):
        for a, b in [
            ("I like this group", "I do not like this group"),
            ("Kill ISIS now", "Kill Islam now"),
            ("1/30/17 white house briefing", "2/1/17 white house briefing"),
        ]:
            self.assertNotEqual(c.wrapper_key(a), c.wrapper_key(b))

    def test_handles_and_mentions(self):
        self.assertEqual(
            c.wrapper_key("@alice stop this now"), c.wrapper_key("<user> stop this now")
        )
        self.assertNotEqual(
            c.wrapper_key("@alice @bob stop this now"),
            c.wrapper_key("@alice stop this now"),
        )
        self.assertNotEqual(
            c.body_norm("email a@example.com please"),
            c.body_norm("email b@example.com please"),
        )

    def test_short_keys(self):
        for text in ("RT via URL", "hello", "read more", "rt:"):
            self.assertIsNone(c.wrapper_key(text))

    def test_exact_closure_not_fuzzy(self):
        rows = [
            {"id": "a", "text": "RT: This is the same headline via"},
            {"id": "b", "text": "This is the same headline"},
            {"id": "c", "text": "This is not the same headline"},
        ]
        excluded, links, n = closure(rows, {"a"})
        self.assertEqual(excluded, {"a", "b"})
        self.assertIn("b", links)
        self.assertEqual(n, 1)
        self.assertEqual(closure(rows, excluded)[1], {})


if __name__ == "__main__":
    unittest.main()

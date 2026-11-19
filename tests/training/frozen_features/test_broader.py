import unittest
from training.frozen_features import broader as b


class InputAlignment(unittest.TestCase):
    def test_exact_keys_and_texts_required(self):
        roster = [{"key": "en/a", "text": "hello"}]
        b.verify_origins(
            roster, [{"key": "en/a", "exact_text_sha256": b.text_sha("hello")}]
        )
        with self.assertRaises(RuntimeError):
            b.verify_origins(
                roster, [{"key": "en/b", "exact_text_sha256": b.text_sha("hello")}]
            )
        with self.assertRaises(RuntimeError):
            b.verify_origins(
                roster, [{"key": "en/a", "exact_text_sha256": b.text_sha("hello ")}]
            )

    def test_extra_or_duplicate_origins_rejected(self):
        r = {"key": "en/a", "text": "hi"}
        o = {"key": "en/a", "exact_text_sha256": b.text_sha("hi")}
        with self.assertRaises(RuntimeError):
            b.verify_origins([r], [o, o])
        with self.assertRaises(RuntimeError):
            b.verify_origins([r, r], [o, o])


if __name__ == "__main__":
    unittest.main()

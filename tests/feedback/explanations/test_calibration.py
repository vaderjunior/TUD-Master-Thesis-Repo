import io
import json
from pathlib import Path
import socket
import unittest
from unittest.mock import patch
import numpy as np
from analysis.feedback.explanations import calibration as b
from analysis.feedback.explanations.validate_calibration import original_bucket_class


class CalibrationTests(unittest.TestCase):
    def pair(self, cos, y, gate=True, identity="q"):
        return {
            "cosine": cos,
            "compatible": y,
            "train_gate": gate,
            "query_id": identity,
            "partition": "calibration",
        }

    def test_unknown_is_not_negative(self):
        self.assertIsNone(b.signature({"meta": {"gate": True, "hate_types": None}}))
        self.assertIsNone(b.signature({"meta": {"gate": True, "hate_types": []}}))
        self.assertEqual(
            b.signature({"meta": {"gate": False, "hate_types": None}}), "benign"
        )

    def test_inconsistent_benign_fails(self):
        with self.assertRaises(AssertionError):
            b.signature({"meta": {"gate": False, "hate_types": ["irony"]}})

    def test_precision_cost_can_skip_positive_pairs(self):
        rows = [self.pair(0.81, True), self.pair(0.41, True)] + [
            self.pair(0.6, False, False, str(i)) for i in range(5)
        ]
        cutoff, _ = b.choose(rows)
        self.assertEqual(cutoff, 81)
        m = b.measure(rows, cutoff)
        self.assertEqual((m["tp"], m["fp"], m["fn"]), (1, 0, 1))
        self.assertEqual(m["recall"], 0.5)

    def test_all_zero_prefers_never(self):
        rows = [self.pair(0.9, False, False)]
        cutoff, _ = b.choose(rows)
        self.assertEqual(cutoff, 200)
        self.assertIsNone(b.measure(rows, cutoff)["precision"])

    def test_review_rows_cannot_fit(self):
        r = self.pair(0.5, True)
        r["partition"] = "review"
        with self.assertRaises(AssertionError):
            b.choose([r])

    def test_inclusive_boundary(self):
        self.assertEqual(b.measure([self.pair(0.55, True)], 55)["tp"], 1)
        self.assertEqual(b.measure([self.pair(0.55, True)], 56)["fn"], 1)

    def test_transitive_protected_component(self):
        gs = b.components(
            4,
            [(0, 1), (1, 2)],
            {2: ["en/protected"]},
            ["benign"] * 4,
            [(str(i),) for i in range(4)],
        )
        self.assertEqual(gs[0]["members"], [0, 1, 2])
        self.assertEqual(gs[0]["excluded_reason"], "protected_dev_overlap")
        self.assertEqual(gs[1]["representative"], 3)

    def test_conflicting_annotations_and_collapse(self):
        gs = b.components(
            4,
            [(0, 1), (2, 3)],
            {},
            ["benign", "hate:irony", "benign", "benign"],
            [("b",), ("a",), ("d",), ("c",)],
        )
        self.assertEqual(gs[0]["excluded_reason"], "conflicting_train_annotations")
        self.assertEqual(gs[1]["representative"], 3)

    def test_normalization(self):
        self.assertEqual(b.normalize("  : : Ｔｅｓｔ\tTEXT  "), "test text")
        self.assertEqual(b.normalize("Straße"), "strasse")

    def test_manual_hybrid_matches_original_with_shuffled_ids(self):
        rs = [
            {
                "id": f"r{i:02}",
                "text": ("common " * (i % 3 + 1))
                + f"unique{i} "
                + ("alpha" if i % 2 else "beta"),
            }
            for i in range(16)
        ]
        rng = np.random.default_rng(90210)
        vectors = rng.normal(size=(16, 8))
        a = b.LocalBucket(rs[::-1], vectors[::-1])
        c = original_bucket_class()(rs, vectors)
        for text in ("alpha alpha unique1", "common", "missing", "unique15 beta"):
            q = rng.normal(size=8)
            r, cos, trace = a.retrieve(text, q)
            rr, tt = c.retrieve(text, q)
            self.assertEqual(r["id"], rr["id"])
            idx = c.ids.index(rr["id"])
            self.assertAlmostEqual(
                cos, float(c.vectors[idx] @ (q / np.linalg.norm(q))), places=13
            )
            for key in ("dense", "bm25", "rrf"):
                self.assertEqual(
                    [x["id"] for x in trace[key]], [x["id"] for x in tt[key]]
                )
                np.testing.assert_allclose(
                    [x["score"] for x in trace[key]],
                    [x["score"] for x in tt[key]],
                    atol=1e-12,
                    rtol=0,
                )

    def test_dense_ties_are_id_deterministic(self):
        rs = [{"id": "z", "text": "unique z"}, {"id": "a", "text": "unique a"}]
        v = np.array([[1.0, 0.0], [1.0, 0.0]])
        r, _, trace = b.LocalBucket(rs, v).retrieve("missing", np.array([1.0, 0.0]))
        self.assertEqual(r["id"], "a")
        self.assertEqual(trace["bm25"], [])


if __name__ == "__main__":
    output = io.StringIO()
    with patch.object(
        socket.socket, "connect", side_effect=RuntimeError("No test network")
    ):
        result = unittest.TextTestRunner(stream=output, verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(CalibrationTests)
        )
    print(output.getvalue())
    if not result.wasSuccessful():
        raise SystemExit(1)
    path = b.BASE / "tests_passed.json"
    with path.open("x", encoding="utf8", newline="\n") as f:
        json.dump(
            {
                "successful": True,
                "tests": result.testsRun,
                "output": output.getvalue(),
                "builder_sha256": b.sha(Path(b.__file__)),
                "test_source_sha256": b.sha(Path(__file__)),
            },
            f,
            indent=2,
        )
        f.write("\n")

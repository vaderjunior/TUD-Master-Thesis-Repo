"""Tests for common execution."""

import argparse
import ast
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from analysis.common import study as c
from analysis.common import execution as e
from analysis.common import result_reader as s
from analysis.common import checkpoints as chunks
import time

TAX = {
    "dimensions": {
        "hate_type": {"labels": ["a", "b"]},
        "target_group": {"labels": ["a", "b"]},
        "severity": {"labels": ["low", "medium", "high"]},
        "legal": {"labels": ["a", "b"]},
    }
}


def prediction(hate=True, labels=None, legal=None):
    return {
        "hate": hate,
        "hate_type": labels or [],
        "target_group": labels or [],
        "severity": "low" if hate else None,
        "legal": legal or [],
    }


def records():
    return [
        {
            "id": str(i),
            "gate": i < 10,
            "hate_types": ["a"] if i < 10 else None,
            "target_groups": ["a"] if i < 10 else None,
            "severity": "low" if i < 10 else None,
            "legal": ["a"] if i % 2 else [],
            "source": "x" if i < 10 else "y",
        }
        for i in range(20)
    ]


class ScienceTests(unittest.TestCase):

    def test_encoder_gate_assembly_and_frozen_threshold_boundary(self):
        from training.original import validate as enc

        heads = {
            ("full", "hate", 0): {"x": "false"},
            ("full", "hate_type", 0): {"x": ["a"]},
            ("full", "target_group", 0): {"x": ["b"]},
            ("full", "severity", 0): {"x": "high"},
            ("de", "hate", 0): {"x": "false"},
            ("de", "legal", 0): {"x": ["a"]},
        }
        self.assertEqual(enc.assemble_value("full", 0, "x", heads), prediction(False))
        self.assertEqual(enc.assemble_value("de", 0, "x", heads)["legal"], ["a"])
        self.assertEqual(
            enc.e.en.decisions(np.array([[-1e-06, 0.0, 1e-06]]), "multilabel").tolist(),
            [[0, 1, 1]],
        )

    def test_offline_chunk_interruption_resume_and_corruption(self):
        folder = c.BASE / "offline_tests" / f"chunks_{time.time_ns()}"
        rr = [{"key": str(i)} for i in range(7)]
        called = []

        def make(i, row):
            called.append(i)
            if i == 4:
                raise KeyboardInterrupt()
            return row | {"value": i}

        with self.assertRaises(KeyboardInterrupt):
            chunks.gather(folder, rr, make, {"test": True}, chunk_size=3)
        self.assertEqual(called, [0, 1, 2, 3, 4])
        called.clear()

        def good(i, row):
            called.append(i)
            return row | {"value": i}

        values, receipt = chunks.gather(folder, rr, good, {"test": True}, chunk_size=3)
        self.assertEqual(called, [3, 4, 5, 6])
        self.assertEqual(receipt["reused_rows"], 3)
        chunks.gather(
            folder,
            rr,
            lambda *_: self.fail("Completed chunks recomputed"),
            {"test": True},
            chunk_size=3,
        )
        path = folder / "chunk_000000.json"
        v = c.read(path)
        v["rows"][0]["value"] = 999
        path.write_bytes(c.encoded(v))
        with self.assertRaises(RuntimeError):
            chunks.gather(folder, rr, good, {"test": True}, chunk_size=3)

    def test_unavailable_model_fails_before_classification(self):
        b = e.queue("matched")
        body = b'{"data":[{"id":"some-other-model"}]}'
        fake = __import__("types").SimpleNamespace(
            status_code=200, content=body, json=lambda: __import__("json").loads(body)
        )
        client = b.a.RequestsTransport(
            "https://openwebui.srv.peasec.de/api/v1", "synthetic-no-secret"
        )
        with patch("requests.get", return_value=fake) as get:
            with self.assertRaises(b.a.IntegrityError):
                b.availability(
                    b.BASE / "offline_check" / f"unavailable_{time.time_ns()}", client
                )
            self.assertEqual(get.call_count, 1)

    def test_missing_vs_empty_and_legal_independent(self):
        rr = records()
        rr[0]["legal"] = None
        self.assertEqual(len(s.endpoint(rr, "legal", TAX)["scorable_ids"]), 19)
        self.assertEqual(len(s.endpoint(rr, "hate_type", TAX)["scorable_ids"]), 10)
        ep = s.endpoint(records(), "legal", TAX)
        p = {r["id"]: prediction(False, legal=r["legal"]) for r in records()}
        v = s.contributions(records(), p, ep)
        self.assertEqual(s.original.score(v)["macro_f1_valid_only"], 1.0)

    def test_predicted_gate_error_retained_and_known_hand_f1(self):
        rr = records()
        ep = s.endpoint(rr, "hate_type", TAX)
        p = {r["id"]: prediction(r["gate"], r["hate_types"]) for r in rr}
        p["0"] = prediction(False)
        v = s.contributions(rr, p, ep)
        self.assertAlmostEqual(s.original.score(v)["macro_f1_valid_only"], 18 / 19)
        self.assertEqual(v["counts"][:, 0, 2].sum(), 1)
        bad = deepcopy(p)
        bad["0"]["hate_type"] = ["a"]
        with self.assertRaises(RuntimeError):
            s.contributions(rr, bad, ep)

    def test_full_roster_macro_labels_do_not_follow_null_mask(self):
        rr = records()
        ep = s.endpoint(rr, "hate_type", TAX)
        self.assertEqual(ep["macro_labels"], ["a"])
        self.assertEqual(ep["absent_labels"], ["b"])
        p = {r["id"]: None for r in rr}
        v = s.contributions(rr, p, ep)
        self.assertEqual(v["macro_indices"], [0])
        sc = s.original.score(v)
        self.assertIsNone(sc["macro_f1_valid_only"])
        self.assertEqual(sc["invalid_inclusive_bounds"], [0.0, 1.0])
        result = s.contrast([v], [v], list(range(10)), draws=64, family=6)
        self.assertIsNone(result["mean_difference"])
        self.assertEqual(result["full_roster_failure_difference_bounds"], [-1.0, 1.0])

    def test_exact_pairing_and_null_bounds(self):
        rr = records()
        ep = s.endpoint(rr, "hate", TAX)
        good = {r["id"]: prediction(r["gate"], r["hate_types"]) for r in rr}
        left = s.contributions(rr, good, ep)
        bad = dict(good)
        bad["0"] = None
        right = s.contributions(rr, bad, ep)
        out = s.contrast(
            [left] * 3,
            [right] * 9,
            [str(i // 2) for i in range(20)],
            draws=128,
            family=6,
        )
        self.assertEqual(out["n_common_valid"], 19)
        self.assertEqual(out["excluded_ids"], ["0"])
        self.assertEqual(out["mean_difference"], 0.0)
        self.assertEqual(out["family_interval"], [0.0, 0.0])
        self.assertGreater(out["full_roster_failure_difference_bounds"][1], 0.0)
        changed = deepcopy(right)
        changed["ids"] = list(reversed(changed["ids"]))
        with self.assertRaises(RuntimeError):
            s.contrast([left], [changed], list(range(20)), draws=1)

    def test_fixed_variant_average_not_pooled_prediction_score(self):
        rr = records()
        ep = s.endpoint(rr, "hate", TAX)
        good = {r["id"]: prediction(r["gate"], r["hate_types"]) for r in rr}
        wrong = {r["id"]: prediction(not r["gate"], []) for r in rr}
        l = s.contributions(rr, good, ep)
        z = s.contributions(rr, wrong, ep)
        out = s.contrast(
            [l] * 3, [l] * 3 + [z] * 6, [str(i) for i in range(20)], draws=128, family=6
        )
        self.assertAlmostEqual(out["mean_difference"], 2 / 3)
        self.assertEqual(len(out["common_f1_by_setting"]), 12)

    def test_cluster_and_source_strata_keep_cross_source_pairs_together(self):
        rr = records()
        ep = s.endpoint(rr, "hate", TAX)
        out = {r["id"]: prediction(r["gate"], r["hate_types"]) for r in rr}
        v = s.contributions(rr, out, ep)
        clusters = [str(i % 10) for i in range(20)]
        sources = [r["source"] for r in rr]
        result = s.contrast(
            [v], [v], clusters, sources=sources, stratify=True, draws=129, family=4
        )
        self.assertEqual(result["n_clusters"], 10)
        self.assertEqual(result["cluster_strata"], {"x|y": 10})
        self.assertEqual(result["direct_cluster_checks"], 3)
        self.assertEqual(result["family_confidence"], 0.9875)
        self.assertEqual(result["ci95"], [0.0, 0.0])

    def test_no_supported_labels_is_explicit(self):
        rr = records()[:5]
        ep = s.endpoint(rr, "hate_type", TAX)
        self.assertEqual(ep["macro_labels"], [])
        p = {r["id"]: prediction(True, ["a"]) for r in rr}
        v = s.contributions(rr, p, ep)
        self.assertIsNone(
            s.contrast([v], [v], list(range(5)), draws=2)["mean_difference"]
        )

    def test_transport_incomplete_outputs_cannot_be_scored(self):
        rr = records()
        ep = s.endpoint(rr, "hate", TAX)
        with self.assertRaises(ValueError):
            s.contributions(rr, {}, ep)

    def test_all_scheduled_identity_counts_and_input_grouping(self):
        for study, (n, settings, predictions) in e.SIZES.items():
            cat = {
                (f"input/{i}", f"condition{j}"): dict(
                    messages=[
                        {"role": "system", "text": "x"},
                        {"role": "user", "text": "y"},
                    ],
                    schema_path="fixture",
                    schema_sha256="fixture",
                    population="a" if i % 2 else "b",
                )
                for i in range(n)
                for j in range(settings)
            }
            for row in cat.values():
                row["messages_sha256"] = c.signature(row["messages"])
            schedule = e.make_schedule(study, cat)
            result = e.check_schedule(study, schedule, cat)
            self.assertEqual(result["predictions"], predictions)
            self.assertEqual(result["ordinary_calls"], predictions * 3)
            broken = deepcopy(schedule)
            broken["stages"][0]["jobs"].append(broken["stages"][0]["jobs"][0])
            with self.assertRaises(RuntimeError):
                e.check_schedule(study, broken, cat)

    def test_readiness_rejected_before_any_live_client(self):
        with patch.object(
            e, "validate", side_effect=RuntimeError("Synthetic missing readiness")
        ), patch.object(
            e.core, "live_client", side_effect=AssertionError("No credentials")
        ) as client:
            with self.assertRaises(RuntimeError):
                e.run_all("matched")
            client.assert_not_called()


def inherited_suite(b):
    """Reuse actual previously reviewed behavioral tests in a private namespace."""
    path = c.ROOT / "tests/feedback/summary/test_balanced.py"
    tree = ast.parse(path.read_text(encoding="utf8"))
    skipped = {
        "test_all_eight_taxonomies_have_isolated_validation",
        "test_parallel_schemas_do_not_leak",
        "test_removed_label_repair_uses_removed_schema",
        "test_schema_change_cannot_reuse_saved_item",
        "test_full_science_schedule_exact_counts_and_shared_references",
        "test_readiness_gate_precedes_live_client_creation",
    }
    selected = []
    for node in tree.body:
        if isinstance(node, ast.If) or (
            isinstance(node, ast.FunctionDef) and node.name == "main"
        ):
            continue
        if isinstance(node, ast.ImportFrom) and any(
            (n.asname in ("b", "audit") for n in node.names)
        ):
            continue
        if isinstance(node, ast.ClassDef):
            node.body = [
                x
                for x in node.body
                if not isinstance(x, ast.FunctionDef) or x.name not in skipped
            ]
        selected.append(node)
    ns = {
        "b": b,
        "audit": b.audit,
        "__name__": __name__ + "_inherited",
        "__package__": __package__,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), ns)
    return (
        unittest.defaultTestLoader.loadTestsFromTestCase(ns["CombinedTests"]),
        ns["TEST_ROOT"],
    )


def main(study):
    b = e.queue(study)
    with c.offline():
        inherited, root = inherited_suite(b)
        suite = unittest.TestSuite(
            [inherited, unittest.defaultTestLoader.loadTestsFromTestCase(ScienceTests)]
        )
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    b.save(
        b.test_receipt(),
        {
            "passed": True,
            "tests_run": result.testsRun,
            "created_utc": c.utc(),
            "environment": e.core.environment(),
            "source_hashes": {n: c.sha(c.ROOT / n) for n in e.SOURCE_FILES},
            "preserved_fixtures": c.rel(root),
            "actual_Qwen_calls": 0,
            "synthetic_only": True,
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--study", choices=sorted(e.SIZES), default="matched")
    args = p.parse_args()
    main(args.study)

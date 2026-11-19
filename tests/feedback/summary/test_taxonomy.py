from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
import json
import random
import socket
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import yaml
from analysis.feedback.restoration import study as s
from analysis.common import prompt_archive as b
from src.hsrag import (
    parse as original_parse,
    vote as original_vote,
    schema as original_schema,
)

ASSETS = s.BASE / "development/test_assets"


def asset(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == data
    else:
        with path.open("xb") as f:
            f.write(data)


def answer(label="explicit", hate=True):
    return {
        "reasoning": "Fixture rationale.",
        "hate": hate,
        "target_group": [],
        "hate_type": [label] if label else [],
        "severity": None,
        "legal": [],
    }


def parsed(parser, raw):
    result, n, error = parser.parse_response(raw)
    return (
        None if result is None else result.model_dump(),
        n,
        error,
        False if result is None else result.gate_normalised,
    )


class TaxonomyTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        raw = (s.ROOT / "config/taxonomy.yaml").read_bytes()
        tax = yaml.safe_load(raw)
        asset(ASSETS / "full.yaml", raw)
        for label in s.LABELS:
            changed = deepcopy(tax)
            del changed["dimensions"]["hate_type"]["labels"][label]
            asset(
                ASSETS / f"{label}.yaml",
                yaml.safe_dump(changed, sort_keys=False, allow_unicode=True).encode(),
            )
        cls.bundles = {name: b.bundle(ASSETS / f"{name}.yaml") for name in s.POOLS}

    def test_full_schema_is_production_schema(self):
        self.assertEqual(
            self.bundles["full"].schema.Result.model_json_schema(),
            original_schema.Result.model_json_schema(),
        )

    def test_every_withdrawn_label_is_rejected_before_gate_normalization(self):
        for label in s.LABELS:
            for gate in (True, False):
                raw = json.dumps(answer(label, gate))
                result, _, error = self.bundles[label].parser.parse_response(raw)
                self.assertIsNone(result)
                self.assertIn("schema violation", error)
                self.assertIsNotNone(self.bundles["full"].parser.parse_response(raw)[0])

    def test_allowed_labels_survive_each_variant(self):
        for label in s.LABELS:
            other = [x for x in s.LABELS if x != label]
            a = answer()
            a["hate_type"] = other
            result, _, error = self.bundles[label].parser.parse_response(json.dumps(a))
            self.assertIsNone(error)
            self.assertEqual(result.hate_type, other)

    def test_full_parser_conformance_format_and_type_drift(self):
        raws = [
            "not json",
            "[]",
            "null",
            "{}",
            "<think>ignore {}</think>```json\n" + json.dumps(answer()) + "\n```",
            "before " + json.dumps(answer()) + " after",
        ]
        rng = random.Random(20261001)
        for _ in range(160):
            a = answer(rng.choice(s.LABELS))
            key = rng.choice(
                [
                    "hate",
                    "hate_type",
                    "severity",
                    "target_group",
                    "reasoning",
                    "legal",
                    "extra",
                ]
            )
            a[key] = rng.choice(
                [None, [], {}, "YES", False, 0, "unknown", ["unknown"], "Race", "null"]
            )
            raws.append(json.dumps(a))
        for raw in raws:
            self.assertEqual(
                parsed(self.bundles["full"].parser, raw), parsed(original_parse, raw)
            )

    def test_aliases_and_scalar_labels_respect_allowed_space(self):
        for label in s.LABELS:
            a = answer()
            a.pop("hate_type")
            a["HATE_TYPES"] = label.upper()
            a["hate"] = "yes"
            self.assertIsNone(
                self.bundles[label].parser.parse_response(json.dumps(a))[0]
            )
            full, n, error = self.bundles["full"].parser.parse_response(json.dumps(a))
            self.assertIsNone(error)
            self.assertGreaterEqual(n, 3)
            self.assertEqual(full.hate_type, [label])

    def test_gate_normalization_preserves_independent_legal_labels(self):
        for name, bundle in self.bundles.items():
            label = next((x for x in s.LABELS if x != name))
            a = answer(label, False)
            a["target_group"] = ["race"]
            a["severity"] = "high"
            a["legal"] = ["insult_defamation"]
            result, _, error = bundle.parser.parse_response(json.dumps(a))
            self.assertIsNone(error)
            self.assertTrue(result.gate_normalised)
            self.assertEqual(
                (result.hate_type, result.target_group, result.severity), ([], [], None)
            )
            self.assertEqual(result.legal, ["insult_defamation"])

    def test_repair_uses_variant_enum_error_and_stops_at_limit(self):
        for label in s.LABELS:
            bundle = self.bundles[label]
            calls = []
            bad = json.dumps(answer(label))
            allowed = next((x for x in s.LABELS if x != label))
            outputs = iter([bad, json.dumps(answer(allowed))])
            client = SimpleNamespace(
                complete=lambda messages: calls.append(deepcopy(messages))
                or next(outputs)
            )
            result = bundle.parser.run_once(
                client, [{"role": "user", "text": "fixture"}], max_repairs=2
            )
            self.assertTrue(result.ok)
            self.assertEqual(result.repairs, 1)
            self.assertEqual(result.result.hate_type, [allowed])
            self.assertEqual(calls[1][-2], {"role": "assistant", "text": bad})
            self.assertNotIn("'" + label + "'", calls[1][-1]["text"])
            for other in s.LABELS:
                if other != label:
                    self.assertIn(other, calls[1][-1]["text"])
            failures = []
            client = SimpleNamespace(
                complete=lambda messages: failures.append(messages) or bad
            )
            failed = bundle.parser.run_once(
                client, [{"role": "user", "text": "fixture"}], max_repairs=2
            )
            self.assertFalse(failed.ok)
            self.assertEqual(len(failures), 3)
            self.assertTrue(failed.parse_failure)

    def test_all_invalid_vote_stays_null(self):
        for bundle in self.bundles.values():
            voted = bundle.voter.aggregate(
                [bundle.parser.RunResult(result=None) for _ in range(3)]
            )
            self.assertIsNone(voted.result)
            self.assertEqual(voted.n_valid, 0)
            self.assertEqual(voted.n_runs, 3)

    def test_tie_severity_and_legal_vote_match_production(self):
        arrays = [
            [answer("explicit"), answer(None, False)],
            [answer("irony"), answer("irony"), answer("grievance")],
            [
                dict(answer("irony"), severity="high", legal=["insult_defamation"]),
                dict(answer("irony"), severity="low", legal=["insult_defamation"]),
                None,
            ],
        ]
        for raw in arrays:
            runs = [
                original_parse.RunResult(
                    result=None if a is None else original_schema.Result(**a)
                )
                for a in raw
            ]
            own = [
                self.bundles["full"].parser.RunResult(
                    result=(
                        None if a is None else self.bundles["full"].schema.Result(**a)
                    )
                )
                for a in raw
            ]
            left, right = (
                original_vote.aggregate(runs),
                self.bundles["full"].voter.aggregate(own),
            )
            self.assertEqual(left.result.model_dump(), right.result.model_dump())
            self.assertEqual(
                (left.n_valid, left.uncertain, left.agreement),
                (right.n_valid, right.uncertain, right.agreement),
            )
        self.assertEqual(right.result.severity, "low")
        self.assertEqual(right.result.legal, ["insult_defamation"])

    def test_voter_output_is_owned_by_its_variant(self):
        for label in s.LABELS:
            bundle = self.bundles[label]
            allowed = next((x for x in s.LABELS if x != label))
            result = bundle.voter.aggregate(
                [
                    bundle.parser.RunResult(
                        result=bundle.schema.Result(**answer(allowed))
                    )
                    for _ in range(3)
                ]
            )
            self.assertIs(type(result.result), bundle.schema.Result)
            self.assertNotIn(label, result.result.hate_type)

    def test_concurrent_bundles_do_not_leak_enums(self):

        def run(label):
            bundle = b.bundle(ASSETS / f"{label}.yaml")
            return (
                bundle is self.bundles[label],
                bundle.parser.parse_response(json.dumps(answer(label)))[0] is None,
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            self.assertTrue(all((a and c for a, c in pool.map(run, s.LABELS * 20))))

    def test_schema_only_changes_system_label_line(self):
        full = self.bundles["full"].render("fixture query", "en", s.EMPTY)
        self.assertEqual(
            full,
            s.renderer.render_messages("fixture query", "en", s.EMPTY, "corrected"),
        )
        for label in s.LABELS:
            removed = self.bundles[label].render("fixture query", "en", s.EMPTY)
            self.assertEqual(full[1], removed[1])
            a, before = (full[0]["text"].splitlines(), removed[0]["text"].splitlines())
            diffs = [(x, y) for x, y in zip(a, before) if x != y]
            self.assertEqual(len(a), len(before))
            self.assertEqual(len(diffs), 1)
            self.assertTrue(diffs[0][0].startswith("- hate_type (multilabel):"))

    def test_pre_search_filter_is_forwarded_and_leak_rejected(self):
        seen = []

        class Col:
            metadata = {}

            def query(self, **kwargs):
                seen.append(kwargs)
                return {"ids": [["keep"]]}

        wrapper = s.FilteredCollection(Col(), [{"id": "keep"}], True)
        self.assertEqual(
            wrapper.query(query_embeddings=[[1.0, 0.0]], n_results=1)["ids"], [["keep"]]
        )
        self.assertEqual(seen[0]["ids"], ["keep"])
        with self.assertRaisesRegex(AssertionError, "competing"):
            wrapper.query(ids=["bad"])
        wrapper.allowed.clear()
        with self.assertRaisesRegex(AssertionError, "Withdrawn"):
            wrapper.query()

    def test_full_index_query_is_unfiltered(self):
        seen = []
        col = SimpleNamespace(
            metadata={}, query=lambda **kw: seen.append(kw) or {"ids": [["keep"]]}
        )
        s.FilteredCollection(col, [{"id": "keep"}], False).query(n_results=1)
        self.assertNotIn("ids", seen[0])

    def test_clusters_keep_duplicate_and_train_touch_is_transitive(self):
        self.assertEqual(s.groups(5, [(0, 1), (1, 2), (3, 4)]), [[0, 1, 2], [3, 4]])
        touched = {2}
        self.assertEqual(
            [g for g in s.groups(5, [(0, 1), (1, 2), (3, 4)]) if not set(g) & touched],
            [[3, 4]],
        )

    def test_trace_does_not_change_original_ranking(self):
        records = []
        for kind, n in [("definition", 12), ("guideline", 12), ("example", 40)]:
            for i in range(n):
                records.append(
                    {
                        "id": kind + str(i),
                        "text": f"query fixture {i % 9}",
                        "kind": kind,
                        "lang": "en",
                        "source": "fixture",
                        "dimension": None,
                        "label": None,
                        "meta": (
                            {"gate": True, "hate_types": ["explicit"]}
                            if kind == "example"
                            else {}
                        ),
                    }
                )
        path = ASSETS / "records.jsonl"
        asset(path, "".join((json.dumps(r) + "\n" for r in records)).encode())
        rng = np.random.default_rng(10)
        vectors = rng.normal(size=(len(records), 8))
        vectors /= np.linalg.norm(vectors, axis=1)[:, None]
        query = rng.normal(size=8)
        query /= np.linalg.norm(query)

        class Col:
            metadata = {}

            def query(self, query_embeddings, n_results, where, include, ids=None):
                filters = where.get("$and", [where])
                allowed = None if ids is None else set(ids)
                ix = [
                    i
                    for i, r in enumerate(records)
                    if (allowed is None or r["id"] in allowed)
                    and all((all((r[k] == v for k, v in f.items())) for f in filters))
                ]
                ix = sorted(ix, key=lambda i: -float(vectors[i] @ query))[:n_results]
                return {
                    "ids": [[records[i]["id"] for i in ix]],
                    "documents": [[records[i]["text"] for i in ix]],
                    "metadatas": [[s.flat_metadata(records[i]) for i in ix]],
                    "distances": [[1 - float(vectors[i] @ query) for i in ix]],
                    "embeddings": (
                        [[vectors[i].tolist() for i in ix]]
                        if "embeddings" in include
                        else None
                    ),
                }

        cfg = {
            "strategy": "hybrid",
            "use_mmr": True,
            "mmr_pool_factor": 3,
            "mmr_lambda": 0.7,
            "rrf_k": 60,
            "k_definitions": 2,
            "k_guidelines": 2,
            "k_examples": 5,
            "k_examples_feedback": 0,
        }
        left = s.make_retriever(Col(), records, path, cfg, {"query": query}, False)
        right = s.make_retriever(Col(), records, path, cfg, {"query": query}, True)
        events = s.instrument(right)
        self.assertEqual(
            s.renderer.from_retrieval(left.retrieve("query")),
            s.renderer.from_retrieval(right.retrieve("query")),
        )
        self.assertEqual(len(events), 10)


def main():
    before = {
        name: s.c.sha(s.ROOT / name)
        for name in (
            "config/taxonomy.yaml",
            "src/hsrag/schema.py",
            "src/hsrag/parse.py",
            "src/hsrag/vote.py",
        )
    }
    with patch.object(
        socket.socket, "connect", side_effect=RuntimeError("Network forbidden in tests")
    ):
        result = unittest.TextTestRunner(verbosity=2).run(
            unittest.defaultTestLoader.loadTestsFromTestCase(TaxonomyTests)
        )
    assert before == {name: s.c.sha(s.ROOT / name) for name in before}
    report = {
        "successful": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "source_hashes": {
            s.c.rel(p): s.c.sha(p)
            for p in (
                s.ROOT / "tests/feedback/summary/test_taxonomy.py",
                s.ROOT / "analysis/feedback/restoration/study.py",
                s.ROOT / "analysis/common/prompt_archive.py",
            )
        },
        "active_bindings_verified": s.c.verify_active(),
        "shared_sources_unchanged": before,
        "Qwen_calls": 0,
    }
    name = (
        "tests_passed.json"
        if result.wasSuccessful()
        else "development/tests_failed_" + str(__import__("time").time_ns()) + ".json"
    )
    s.put(name, report)
    assert result.wasSuccessful(), report


if __name__ == "__main__":
    main()

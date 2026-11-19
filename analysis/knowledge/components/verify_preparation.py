"""Components verify preparation."""

import argparse
from collections import Counter, defaultdict
import io
import json
import socket
import time
import unittest
from unittest.mock import patch
from analysis.knowledge.components import prepare as p

a = p.a
OUT = p.ROOT / "Thesis_Log/astra_result/evidence/sq1_components_preparation_20260921_v1.json"


def tests():
    a.require(not p.PLAN.exists(), "Run development tests before freezing preparation")
    dest = p.BASE / "development/preparation_tests_passed.json"
    a.require(not dest.exists(), "Preserve successful test receipt")
    attempt = p.BASE / "development" / ("preparation_test_" + str(time.time_ns()))
    hashes = {name: a.digest(p.ROOT / name) for name in p.SOURCE_FILES}
    p.save(
        attempt / "source_snapshot.json",
        {
            name: (p.ROOT / name).read_text(encoding="utf8")
            for name in p.SOURCE_FILES
            if "components_" in name
        },
    )
    with patch.object(socket.socket, "connect", p.forbidden), patch.object(
        socket, "create_connection", p.forbidden
    ), patch.object(p.core, "live_client", p.forbidden):
        suite = unittest.defaultTestLoader.loadTestsFromName(
            "tests.knowledge.components.test_prepare"
        )
        stream = io.StringIO()
        tick = time.perf_counter()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    receipt = {
        "created_utc": a.utc(),
        "passed": result.wasSuccessful(),
        "tests_run": result.testsRun,
        "elapsed_s": time.perf_counter() - tick,
        "output": stream.getvalue(),
        "source_hashes": hashes,
        "environment": p.core.environment(),
        "actual_API_calls": 0,
        "attempt": p.rel(attempt),
    }
    p.save(attempt / "tests.json", receipt)
    print(stream.getvalue(), flush=True)
    a.require(result.wasSuccessful(), "Preparation tests failed; preserve attempt")
    for name, digest in hashes.items():
        a.require(a.digest(p.ROOT / name) == digest, "Source changed during tests")
    p.save(dest, receipt)


def independent_parts(original):
    """Render single buckets with the unchanged, previously audited renderer."""
    context = original["context"]
    text = original["text"]
    lang = original["lang"]
    empty = {"definitions": [], "guidelines": [], "example_groups": []}
    empty_messages = p.prompt.render_messages(text, lang, empty, "corrected")
    target = empty_messages[1]["text"]
    chunks = {}
    for name, field in [("d", "definitions"), ("g", "guidelines"), ("e", "example_groups")]:
        ctx = {**empty, field: context[field]}
        messages = p.prompt.render_messages(text, lang, ctx, "corrected")
        a.require(messages[0] == empty_messages[0], "System changed across contexts")
        user = messages[1]["text"]
        suffix = "\n\n" + target
        a.require(user.endswith(suffix), "Input rendering changed")
        block = user[: -len(suffix)]
        if name == "e":
            prefix = (
                "## Labelled examples\n"
                + p.prompt.module("corrected")._load_template()["EXAMPLES_CAUTION"]
                + "\n\n"
            )
            a.require(block.startswith(prefix), "Cannot isolate exact legacy caution")
            block = "## Labelled examples\n\n" + block[len(prefix) :]
        chunks[name] = block
    return (empty_messages[0], target, chunks)


def check():
    a.require(
        not OUT.exists() and (not (p.BASE / "preparation_checked.json").exists()),
        "Preserve completed preparation audit",
    )
    plan = p.validate()
    old, dev, sources, _ = p.sources()
    analysis = a.read(p.BASE / "analysis_plan.json")
    a.require(analysis == p.analysis_plan(old), "Analysis contract differs")
    raw = p.core.rows(p.PROMPTS)
    cat = {(r["key"], r["condition"]): r for r in raw}
    expected = {(row["key"], arm) for row in dev for arm in p.CONDITIONS}
    a.require(
        len(raw) == len(cat) == 11655 and set(cat) == expected,
        "Prompt roster missing/extra/duplicate",
    )
    records = {r["id"]: r for r in p.core.rows(p.CORE / "kb/full/records.jsonl")}
    a.require(len(records) == 71131, "Pool record count changed")
    counts = defaultdict(Counter)
    lengths = defaultdict(list)
    retained_slots = Counter()
    reference_matches = 0
    for number, row in enumerate(dev):
        original = sources[row["key"]]
        original_messages = p.prompt.render_messages(
            row["text"], row["lang"], original["context"], "corrected"
        )
        a.require(
            original_messages == original["messages"], "Saved full-RAG prompt no longer reproduces"
        )
        for kind, hits in [
            ("definition", original["context"]["definitions"]),
            ("guideline", original["context"]["guidelines"]),
            ("example", original["context"]["example_groups"][0][1]),
        ]:
            for hit in hits:
                record = records[hit["id"]]
                meta = {
                    **{k: record[k] for k in ("kind", "lang", "source", "dimension", "label")},
                    **record.get("meta", {}),
                }
                a.require(
                    record["kind"] == kind and record["text"] == hit["text"],
                    "TRAIN text/kind changed",
                )
                a.require(
                    {k: v for k, v in hit["meta"].items() if k != "_ranks"} == meta,
                    "TRAIN annotation changed",
                )
                if kind == "example":
                    a.require(record["lang"] == row["lang"], "Cross-language example")
                retained_slots[kind] += 1
        system, target, chunks = independent_parts(original)
        subset = row["scopes"][0]["subset"]
        for condition in p.CONDITIONS:
            delivered = cat[row["key"], condition]
            for name in ("text", "lang", "item_id", "scopes"):
                a.require(delivered[name] == row[name], "Delivered input attribution differs")
            a.require(
                delivered["source_entry_sha256"] == a.signature(original)
                and delivered["source_messages_sha256"] == original["messages_sha256"],
                "Wrong parent binding",
            )
            if condition == p.BRIDGE:
                expected_messages = original["messages"]
                expected_context = original["context"]
                a.require(
                    delivered["prompt_family"] == "preserved_reference_rag",
                    "Bridge marked as factorial",
                )
                reference_matches += 1
                bits = (1, 1, 1)
            else:
                bits = p.ARMS[condition]
                e, d, g = bits
                selected = [chunks[name] for name, on in [("d", d), ("g", g), ("e", e)] if on]
                expected_messages = [
                    system,
                    {"role": "user", "text": "\n\n".join([p.COMMON_NOTE, *selected, target])},
                ]
                expected_context = {
                    name: original["context"][name] if on else []
                    for name, on in [("definitions", d), ("guidelines", g), ("example_groups", e)]
                }
                a.require(
                    delivered["prompt_family"] == "astra_components_common_v1",
                    "Factorial family differs",
                )
            a.require(delivered["factor_bits_e_d_g"] == list(bits), "Factor flags differ")
            a.require(
                delivered["messages"] == expected_messages
                and delivered["context"] == expected_context,
                "Independent matched prompt/context differs",
            )
            a.require(
                delivered["retrieved"] == p.prompt.hit_ids(expected_context),
                "Delivered record order differs",
            )
            a.require(
                delivered["messages_sha256"] == a.signature(expected_messages)
                and delivered["context_sha256"] == a.signature(expected_context),
                "Delivered signature differs",
            )
            payload = a.body(delivered["messages"])
            a.require(
                payload["model"] == a.MODEL
                and payload["messages"][1]["content"] == expected_messages[1]["text"],
                "API translation differs",
            )
            counts[subset][condition] += 1
            lengths[condition].append(sum((len(m["text"]) for m in expected_messages)))
        if (number + 1) % 250 == 0:
            print(f"Independently checked {number + 1}/1295 matched input families", flush=True)
    for subset, count in p.SCOPE_COUNTS.items():
        a.require(
            dict(counts[subset]) == {condition: count for condition in p.CONDITIONS},
            "Endpoint condition coverage differs",
        )
    core_cat = {r["key"]: r for r in p.core.rows(p.CORE / "retrieval_v1/full/prompts.jsonl")}
    core_difference = [
        key for key, row in sources.items() if row["messages"] != core_cat[key]["messages"]
    ]
    a.require(core_difference == ["en/implicit_hate-4207"], "Unexpected older core prompt drift")
    p.validate()
    report = {
        "created_utc": a.utc(),
        "status": "all_component_prompts_checked_offline",
        "plan_sha256": a.digest(p.PLAN),
        "prompts_sha256": a.digest(p.PROMPTS),
        "checked_inputs": 1295,
        "checked_payloads": 11655,
        "exact_reference_matches": reference_matches,
        "common_instruction": p.COMMON_NOTE,
        "source_slots_checked": dict(retained_slots),
        "endpoint_condition_counts": {k: dict(v) for k, v in counts.items()},
        "prompt_characters": {
            k: {"min": min(v), "max": max(v), "mean": sum(v) / len(v)} for k, v in lengths.items()
        },
        "historical_core_hybrid_prompt_differences": core_difference,
        "control_reuse": "No old predictions reused; source full-RAG bridge is exact for all1295 current strategy inputs. Older core differs for one item, preserved.",
        "verified_hashes": {
            **plan["source_hashes"],
            **plan["input_hashes"],
            p.rel(p.PLAN): a.digest(p.PLAN),
            p.rel(p.PROMPTS): a.digest(p.PROMPTS),
            p.rel(p.BASE / "analysis_plan.json"): a.digest(p.BASE / "analysis_plan.json"),
        },
        "actual_API_calls": 0,
        "credentials_loaded": False,
        "retrieval_calls": 0,
        "gold_scoring": False,
        "next": "Freeze and independently check the separate staged API runner. No live command authorized by this receipt alone.",
    }
    a.save_new(OUT, report)
    p.save(
        p.BASE / "preparation_checked.json",
        {
            "status": report["status"],
            "plan_sha256": a.digest(p.PLAN),
            "evidence": p.rel(OUT),
            "evidence_sha256": a.digest(OUT),
            "prompts_sha256": a.digest(p.PROMPTS),
        },
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "checked_inputs",
                    "checked_payloads",
                    "exact_reference_matches",
                    "actual_API_calls",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tests", "check"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", p.forbidden), patch.object(
        socket, "create_connection", p.forbidden
    ), patch.object(p.core, "live_client", p.forbidden):
        tests() if args.mode == "tests" else check()

"""Sampled banks verify annotations."""

from src.hsrag.paths import PROJECT_ROOT
import hashlib
import json
from pathlib import Path
import re
import socket
import time

ROOT = PROJECT_ROOT
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
BASE = ROOT / "experiments/astra/sq1_annotation_coverage_20260926_v1"
FIELDS = ("gate", "target_groups", "hate_types", "severity", "legal")


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def rows(p):
    return [json.loads(line) for line in p.read_text(encoding="utf8").splitlines() if line.strip()]


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def no_network(*args, **kwargs):
    raise RuntimeError("Offline independent check")


def describe_independently(records, labels):
    summary = {
        "records_or_slots": 0,
        "unique_records": 0,
        "sources": {},
        "gate": dict.fromkeys(("true", "false", "unknown"), 0),
        "fields": {},
        "annotation_signatures_field_order": list(FIELDS),
        "annotation_signatures": {},
    }
    for field in FIELDS:
        summary["fields"][field] = {
            "states": dict.fromkeys(("unannotated", "annotated_empty", "annotated_value"), 0),
            "annotated": 0,
            "label_occurrences": dict.fromkeys(labels[field], 0),
            "core_head_eligible": 0,
            "annotated_by_gate": dict.fromkeys(("true", "false", "unknown"), 0),
        }
    ids = set()
    for record in records:
        ids.add(record["id"])
        summary["records_or_slots"] += 1
        source = record["source"]
        summary["sources"][source] = summary["sources"].get(source, 0) + 1
        meta = record["meta"]
        gate = meta.get("gate")
        assert gate is None or type(gate) is bool
        gate_key = "true" if gate is True else "false" if gate is False else "unknown"
        summary["gate"][gate_key] += 1
        mask = ""
        for field in FIELDS:
            value = meta.get(field)
            counts = summary["fields"][field]
            mask += "0" if value is None else "1"
            if value is None:
                counts["states"]["unannotated"] += 1
                continue
            counts["annotated"] += 1
            counts["annotated_by_gate"][gate_key] += 1
            if field in ("gate", "legal") or gate is True:
                counts["core_head_eligible"] += 1
            if type(value) is list:
                assert len(value) == len(set(value))
                counts["states"]["annotated_empty" if len(value) == 0 else "annotated_value"] += 1
                values = value
            else:
                counts["states"]["annotated_value"] += 1
                values = [str(value).lower()] if field == "gate" else [value]
            for v in values:
                assert v in counts["label_occurrences"]
                counts["label_occurrences"][v] += 1
        summary["annotation_signatures"][mask] = summary["annotation_signatures"].get(mask, 0) + 1
    summary["unique_records"] = len(ids)
    return summary


def independent_gold(meta):
    parts = []
    if meta.get("gate") is not None:
        assert type(meta["gate"]) is bool
        parts.append("hate=" + ("true" if meta["gate"] else "false"))
    for field, rendered in [("target_groups", "target_group"), ("hate_types", "hate_type")]:
        if meta.get(field) is not None:
            parts.append(rendered + "=[" + ", ".join(meta[field]) + "]")
    if meta.get("severity") is not None:
        parts.append("severity=" + meta["severity"])
    if meta.get("legal") is not None:
        parts.append("legal=[" + ", ".join(meta["legal"]) + "]")
    text = ", ".join(parts)
    if meta.get("astra_aux_stgb"):
        text += (
            "\n  Auxiliary DeTox paragraph flags (illustrative, not legal gold): §"
            + meta["astra_aux_stgb"]
        )
    return text


def main():
    start = time.perf_counter()
    socket.socket.connect = no_network
    socket.create_connection = no_network
    assert not (BASE / "independent_check.json").exists(), "Preserve prior verification"
    result = read(BASE / "results.json")
    plan = read(BASE / "analysis_plan.json")
    receipt = read(BASE / "complete.json")
    assert (
        digest(BASE / "results.json")
        == receipt["results_sha256"]
        == digest(ROOT / receipt["evidence"])
    )
    assert digest(BASE / "analysis_plan.json") == receipt["plan_sha256"] == result["plan_sha256"]
    assert (
        digest(BASE / "prompt_delivery.jsonl")
        == result["detail_sha256"]
        == receipt["details_sha256"]
    )
    assert result["input_hashes"] == plan["input_hashes"]
    for name, expected in result["input_hashes"].items():
        assert digest(ROOT / name) == expected, name
    dev = rows(CORE / "dev_inputs.jsonl")
    dev_by_key = {r["key"]: r for r in dev}
    details = rows(BASE / "prompt_delivery.jsonl")
    detail_map = {(r["pool"], r["key"]): r for r in details}
    assert len(details) == len(detail_map) == result["verified_prompts"] == 2590
    counts = {
        "inventory_summaries": 0,
        "delivery_summaries": 0,
        "source_summaries": 0,
        "prompt_details": 0,
        "rendered_example_labels": 0,
        "paired_field_summaries": 0,
        "ordered_abstract_list_checks": 0,
    }
    records_by_pool = {}
    prompts = {}
    for pool in ("small", "full"):
        rr = rows(CORE / f"kb/{pool}/records.jsonl")
        kb = {r["id"]: r for r in rr}
        records_by_pool[pool] = kb
        assert len(kb) == len(rr)
        for lang in ("en", "de"):
            examples = [r for r in rr if r["kind"] == "example" and r["lang"] == lang]
            actual = describe_independently(examples, plan["labels"])
            expected = result["pool_inventory"][pool][lang]
            assert actual == {k: v for k, v in expected.items() if k != "by_source"}
            counts["inventory_summaries"] += 1
            for source, sub in expected["by_source"].items():
                assert (
                    describe_independently(
                        [r for r in examples if r["source"] == source], plan["labels"]
                    )
                    == sub
                )
                counts["source_summaries"] += 1
        pp = rows(CORE / f"retrieval_v1/{pool}/prompts.jsonl")
        prompts[pool] = {p["key"]: p for p in pp}
        assert len(prompts[pool]) == len(pp) == len(dev) == 1295
        for key, p in prompts[pool].items():
            d = detail_map[pool, key]
            source = dev_by_key[key]
            selected = [kb[rid] for rid in p["retrieved"]["Labelled examples"]]
            assert len(selected) == len({r["id"] for r in selected}) == 5
            assert d["key"] == key and d["scopes"] == [s["subset"] for s in source["scopes"]]
            assert d["example_ids"] == [r["id"] for r in selected]
            assert d["sources"] == [r["source"] for r in selected]
            assert d["annotations"] == [{f: r["meta"].get(f) for f in FIELDS} for r in selected]
            s = describe_independently(selected, plan["labels"])
            assert d["annotated_slots"] == {f: s["fields"][f]["annotated"] for f in FIELDS}
            assert d["core_eligible_slots"] == {
                f: s["fields"][f]["core_head_eligible"] for f in FIELDS
            }
            assert d["gate_counts"] == s["gate"]
            for field in ("definitions", "guidelines"):
                assert d[field] == p["retrieved"][field]
                assert [r["id"] for r in p["context"][field]] == d[field]
                for hit in p["context"][field]:
                    assert hit["text"] == kb[hit["id"]]["text"]
                counts["ordered_abstract_list_checks"] += 1
            assert d["message_characters"] == sum((len(m["text"]) for m in p["messages"]))
            assert d["example_text_characters"] == sum(
                (len(" ".join(r["text"].split())) for r in selected)
            )
            serialized = (
                json.dumps(p["messages"], ensure_ascii=False, indent=2, sort_keys=True) + "\n"
            ).encode("utf8")
            assert (
                d["messages_sha256"]
                == p["messages_sha256"]
                == hashlib.sha256(serialized).hexdigest()
            )
            user = p["messages"][1]["text"]
            suffix = "## Text to classify (language: " + source["lang"] + ")\n" + source["text"]
            assert user.endswith(suffix)
            context_text = user[: -len(suffix)]
            assert "## Labelled examples\n" in context_text
            example_text = context_text.split("## Labelled examples\n", 1)[1]
            label_lines = re.findall("^  labels: (.*)$", example_text, flags=re.MULTILINE)
            assert label_lines == [independent_gold(r["meta"]).split("\n", 1)[0] for r in selected]
            for r in selected:
                expected_fragment = (
                    '- "'
                    + " ".join(r["text"].split())
                    + '"\n  labels: '
                    + independent_gold(r["meta"])
                )
                assert expected_fragment in example_text
                counts["rendered_example_labels"] += 1
            counts["prompt_details"] += 1
        for subset, (lang, field, n) in plan["scope"].items():
            keys = [r["key"] for r in dev if any((s["subset"] == subset for s in r["scopes"]))]
            assert len(keys) == n
            selected = [
                kb[rid]
                for key in keys
                for rid in prompts[pool][key]["retrieved"]["Labelled examples"]
            ]
            actual = describe_independently(selected, plan["labels"])
            expected = result["delivery"][pool][subset]
            assert actual == {
                k: v
                for k, v in expected.items()
                if k not in ("by_source", "items", "annotated_slots_per_input")
            }
            assert expected["items"] == n
            for f in FIELDS:
                hist = dict.fromkeys(map(str, range(6)), 0)
                for key in keys:
                    n_known = 0
                    for rid in prompts[pool][key]["retrieved"]["Labelled examples"]:
                        n_known += kb[rid]["meta"].get(f) is not None
                    hist[str(n_known)] += 1
                assert hist == expected["annotated_slots_per_input"][f]
            for src, sub in expected["by_source"].items():
                assert (
                    describe_independently(
                        [r for r in selected if r["source"] == src], plan["labels"]
                    )
                    == sub
                )
                counts["source_summaries"] += 1
            counts["delivery_summaries"] += 1
    assert all(
        (records_by_pool["full"].get(rid) == rec for rid, rec in records_by_pool["small"].items())
    )
    for subset, (_, _, n) in plan["scope"].items():
        keys = [r["key"] for r in dev if any((s["subset"] == subset for s in r["scopes"]))]
        expected = result["paired"][subset]
        for f in FIELDS:
            distribution = dict.fromkeys(map(str, range(-5, 6)), 0)
            for key in keys:
                difference = 0
                for pool, sign in [("small", -1), ("full", 1)]:
                    for rid in prompts[pool][key]["retrieved"]["Labelled examples"]:
                        if records_by_pool[pool][rid]["meta"].get(f) is not None:
                            difference += sign
                distribution[str(difference)] += 1
            actual = {
                "difference_distribution": distribution,
                "full_minus_small_annotated_slots": sum(
                    (int(k) * v for k, v in distribution.items())
                ),
                "inputs_fewer": sum((v for k, v in distribution.items() if int(k) < 0)),
                "inputs_more": sum((v for k, v in distribution.items() if int(k) > 0)),
                "inputs_equal": distribution["0"],
            }
            assert actual == expected[f]
            counts["paired_field_summaries"] += 1
        for abstract in ("definitions", "guidelines"):
            equal = sum(
                (
                    prompts["full"][key]["retrieved"][abstract]
                    == prompts["small"][key]["retrieved"][abstract]
                    for key in keys
                )
            )
            assert equal == expected["identical_ordered_" + abstract]
        shared = 0
        for key in keys:
            for rid in prompts["small"][key]["retrieved"]["Labelled examples"]:
                shared += rid in prompts["full"][key]["retrieved"]["Labelled examples"]
        assert shared == expected["shared_example_slots"]
    assert counts["rendered_example_labels"] == result["verified_example_slots"] == 12950
    verification = {
        "status": "annotation_census_independently_verified",
        "checks": counts,
        "verified_input_hashes": len(result["input_hashes"]),
        "results_sha256": digest(BASE / "results.json"),
        "plan_sha256": digest(BASE / "analysis_plan.json"),
        "details_sha256": digest(BASE / "prompt_delivery.jsonl"),
        "checker_sha256": digest(Path(__file__)),
        "elapsed_seconds": time.perf_counter() - start,
        "method": "Independent loops over raw records and first-pass IDs; exact saved label lines and full example fragments; no census function imports.",
        "API_calls": 0,
        "new_retrieval": False,
        "accuracy_scoring": False,
    }
    data = (
        json.dumps(verification, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    ).encode("utf8")
    evidence = (
        ROOT / "Thesis_Log/astra_result/evidence/sq1_annotation_coverage_check_20260926_v1.json"
    )
    for path in [BASE / "independent_check.json", evidence]:
        with path.open("xb") as f:
            f.write(data)
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()

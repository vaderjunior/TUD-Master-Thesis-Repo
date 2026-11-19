from src.hsrag.paths import PROJECT_ROOT
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import socket
from unittest.mock import patch

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/sq3_explanation_authoring_20261001_v1"
PACKET = ROOT / "experiments/astra/sq3_weekend_preparation_20261001_v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def lines(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf8").splitlines()
        if line.strip()
    ]


def words(text):
    return len(re.findall(r"\S+", text))


def check():
    assert not (BASE / "complete.json").exists(), "Preserve completed check."
    receipt = read(BASE / "cards_prepared.json")
    for name, expected in receipt["outputs"].items():
        assert sha(BASE / name) == expected
    inputs = read(BASE / "inputs.json")
    for name, expected in inputs.items():
        assert sha(ROOT / name) == expected
    packet = {
        r["source_id"]: r for r in lines(PACKET / "explanation_source_packet.jsonl")
    }
    cards = lines(BASE / "explanation_cards.jsonl")
    details = lines(BASE / "source_review_audit.jsonl")
    authored = read(BASE / "assistant_source_review_v1.json")
    templates = read(BASE / "generic_templates_v1.json")
    assert len(cards) == len(details) == len(packet) == len(authored) == 58
    assert (
        [r["source_id"] for r in cards]
        == [r["source_id"] for r in details]
        == list(packet)
    )
    assert len({r["source_id"] for r in cards}) == 58
    available = []
    span_count = 0
    reasons = Counter()
    for row, review in zip(cards, details):
        iid = row["source_id"]
        source = packet[iid]
        draft = authored[iid]
        assert (
            row["source_text_sha256"]
            == hashlib.sha256(source["text"].encode("utf8")).hexdigest()
        )
        assert (
            row["hate"] == source["hate"] is True
            and row["hate_types"] == source["hate_types"]
        )
        assert review["note"] == draft["note"] and review["reason"] == draft["reason"]
        assert review["status"] == draft["status"]
        reasons[review["reason"]] += 1
        assert row["explanation_available"] == (draft["status"] == "available")
        if row["explanation_available"]:
            assert row["specific_explanation"] == draft["explanation"]
            assert len(row["hate_types"]) == 1
            assert row["generic_explanation"] == templates[row["hate_types"][0]]
            assert [s["text"] for s in review["evidence_spans"]] == draft[
                "evidence_spans"
            ]
            assert len(review["evidence_spans"]) >= 2
            for span in review["evidence_spans"]:
                assert source["text"][span["start"] : span["end"]] == span["text"]
                span_count += 1
            a, b = words(row["specific_explanation"]), words(row["generic_explanation"])
            assert 30 <= a <= 60 and 30 <= b <= 60 and abs(a - b) <= 8
            assert [row["specific_words"], row["generic_words"]] == [a, b]
            available.append(row)
        else:
            assert row["specific_explanation"] == row["generic_explanation"] == ""
            assert row["specific_words"] == row["generic_words"] == 0
            assert review["evidence_spans"] == []
        assert row["specific_characters"] == len(row["specific_explanation"])
        assert row["generic_characters"] == len(row["generic_explanation"])
        assert set(row) == {
            "source_id",
            "source_text_sha256",
            "hate",
            "hate_types",
            "explanation_available",
            "specific_explanation",
            "generic_explanation",
            "specific_words",
            "generic_words",
            "specific_characters",
            "generic_characters",
        }
    summary = read(BASE / "authoring_summary.json")
    assert (
        summary["sources"] == 58
        and summary["explanations"] == len(available) == 20
        and summary["fallbacks"] == 38
    )
    assert summary["review_reasons"] == dict(reasons)
    for label, counts in summary["per_label"].items():
        assert counts["total_sources"] == sum(label in r["hate_types"] for r in cards)
        assert counts["explained_sources"] == sum(
            label in r["hate_types"] for r in available
        )
    original_pools = read(
        ROOT
        / "experiments/astra/sq3_full_kb_feedback_20260928_v1/update_payloads_v1/pools.json"
    )["pools"]
    original_pools = [
        p for p in original_pools if p["arm"] == "error" and p["size"] == 40
    ]
    assert len(original_pools) == 3 and {
        i for p in original_pools for i in p["ids"]
    } == set(packet)
    explained_ids = {r["source_id"] for r in available}
    for pool in original_pools:
        stats = summary["pools"][str(pool["seed"])]
        assert set(stats["explained_ids"]) == set(pool["ids"]) & explained_ids
        assert stats["explained_sources"] == len(stats["explained_ids"])
        assert stats["plain_fallback_sources"] == 40 - len(stats["explained_ids"])
        assert stats["explained_type_support"] == dict(
            Counter(t for i in stats["explained_ids"] for t in packet[i]["hate_types"])
        )
    for variant in ("specific", "generic"):
        values = [words(r[variant + "_explanation"]) for r in available]
        assert summary["word_ranges"][variant + "_words"] == [min(values), max(values)]
    assert summary["max_pair_word_difference"] == max(
        abs(words(r["specific_explanation"]) - words(r["generic_explanation"]))
        for r in available
    )
    active = read(
        ROOT
        / "experiments/astra/sq3_full_kb_relevance_20260930_v1/api_execution_v1/plan.json"
    )
    n = 0
    for group in ("input_hashes", "source_hashes", "generated_hashes"):
        for name, expected in active[group].items():
            assert sha(ROOT / name) == expected, name
            n += 1
    assert n == 79
    result = {
        "status": "source_authoring_and_mechanical_checks_complete",
        "rows": 58,
        "available": 20,
        "fallbacks": 38,
        "source_spans_verified": span_count,
        "matched_specific_generic_pairs": 20,
        "original_pools_unchanged": 3,
        "active_bindings_unchanged": 79,
        "input_bindings_verified": len(inputs),
        "actual_Qwen_API_calls": 0,
        "mechanical_check_independent_of_builder": True,
        "semantic_review": "same assistant; no independent human annotation",
        "explanation_accuracy_proven": False,
        "classifier_study_ready": False,
        "cards_prepared_sha256": sha(BASE / "cards_prepared.json"),
        "checker_sha256": sha(Path(__file__)),
        "original_pools_sha256": sha(
            ROOT
            / "experiments/astra/sq3_full_kb_feedback_20260928_v1/update_payloads_v1/pools.json"
        ),
    }
    with (BASE / "complete.json").open("x", encoding="utf8", newline="\n") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    with patch.object(
        socket, "create_connection", side_effect=RuntimeError("Offline card check")
    ), patch.object(
        socket.socket, "connect", side_effect=RuntimeError("Offline card check")
    ):
        check()

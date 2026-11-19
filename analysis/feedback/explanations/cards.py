from src.hsrag.paths import PROJECT_ROOT
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
from unittest.mock import patch

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/sq3_explanation_authoring_20261001_v1"
PACKET = ROOT / "experiments/astra/sq3_weekend_preparation_20261001_v2"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(name, value):
    with (BASE / name).open("x", encoding="utf8", newline="\n") as stream:
        json.dump(
            value, stream, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        stream.write("\n")


def active_check():
    path = (
        ROOT
        / "experiments/astra/sq3_full_kb_relevance_20260930_v1/api_execution_v1/plan.json"
    )
    assert (
        sha(path) == "d4d34f90adf144131e3e59a13cb9c6cd351b7eefc97a093ee71613ee75a0babb"
    )
    plan = read(path)
    count = 0
    for group in ("input_hashes", "source_hashes", "generated_hashes"):
        for name, expected in plan[group].items():
            assert sha(ROOT / name) == expected, name
            count += 1
    assert count == 79
    return count


def run():
    assert not (
        BASE / "cards_prepared.json"
    ).exists(), "Preserve existing card generation."
    plan = read(BASE / "plan.json")
    inputs = dict(plan["inputs"])
    for path in (
        BASE / "plan.json",
        BASE / "assistant_source_review_v1.json",
        BASE / "assistant_consistency_review_v1.md",
        Path(__file__).resolve(),
    ):
        inputs[path.relative_to(ROOT).as_posix()] = sha(path)
    for name, expected in inputs.items():
        assert sha(ROOT / name) == expected, name
    active_check()
    packet = [
        json.loads(s)
        for s in (PACKET / "explanation_source_packet.jsonl")
        .read_text(encoding="utf8")
        .splitlines()
    ]
    source = {r["source_id"]: r for r in packet}
    review = read(BASE / "assistant_source_review_v1.json")
    generic = read(BASE / "generic_templates_v1.json")
    definitions = read(PACKET / "explanation_label_definitions.json")
    assert set(generic) == set(definitions) and len(generic) == 7
    assert len(source) == len(packet) == len(review) == 58 and set(source) == set(
        review
    )
    memberships = read(PACKET / "source_memberships.json")
    assert len(memberships) == len({r["source_id"] for r in memberships}) == 58
    assert {r["source_id"] for r in memberships} == set(source)
    reasons = {
        "missing_context",
        "gate_or_target_unclear",
        "quotation_or_endorsement_unclear",
        "annotation_definition_tension",
    }
    cards, audit = [], []
    for row in packet:
        iid = row["source_id"]
        authored = review[iid]
        assert authored["note"].strip()
        if authored["status"] == "available":
            assert (
                authored["reason"] == "supported_visible_text"
                and len(row["hate_types"]) == 1
            )
            specific = authored["explanation"]
            general = generic[row["hate_types"][0]]
            evidence = []
            assert len(authored["evidence_spans"]) >= 2
            for quote in authored["evidence_spans"]:
                assert quote and quote in row["text"], (iid, quote)
                offset = row["text"].index(quote)
                evidence.append(
                    {"start": offset, "end": offset + len(quote), "text": quote}
                )
            nw, ng = len(specific.split()), len(general.split())
            assert 30 <= nw <= 60 and 30 <= ng <= 60 and abs(nw - ng) <= 8, (
                iid,
                nw,
                ng,
            )
        else:
            assert authored["status"] == "fallback" and authored["reason"] in reasons
            assert "explanation" not in authored and "evidence_spans" not in authored
            specific = general = ""
            evidence = []
            nw = ng = 0
        cards.append(
            {
                "source_id": iid,
                "source_text_sha256": row["provenance"]["text_sha256"],
                "hate": row["hate"],
                "hate_types": row["hate_types"],
                "explanation_available": authored["status"] == "available",
                "specific_explanation": specific,
                "generic_explanation": general,
                "specific_words": nw,
                "generic_words": ng,
                "specific_characters": len(specific),
                "generic_characters": len(general),
            }
        )
        audit.append(
            {
                "source_id": iid,
                "status": authored["status"],
                "reason": authored["reason"],
                "note": authored["note"],
                "evidence_spans": evidence,
            }
        )
    cardmap = {r["source_id"]: r for r in cards}
    pools = sorted({seed for r in memberships for seed in r["pool_seeds"]})
    assert len(pools) == 3
    pool_summary = {}
    for seed in pools:
        ids = [r["source_id"] for r in memberships if seed in r["pool_seeds"]]
        assert len(ids) == 40
        explained = [i for i in ids if cardmap[i]["explanation_available"]]
        pool_summary[str(seed)] = {
            "sources": 40,
            "explained_sources": len(explained),
            "plain_fallback_sources": 40 - len(explained),
            "explained_ids": explained,
            "explained_type_support": dict(
                Counter(t for i in explained for t in source[i]["hate_types"])
            ),
        }
    label_summary = {
        t: {
            "total_sources": sum(t in r["hate_types"] for r in cards),
            "explained_sources": sum(
                t in r["hate_types"] and r["explanation_available"] for r in cards
            ),
        }
        for t in definitions
    }
    available = [r for r in cards if r["explanation_available"]]
    report = {
        "status": "source_authoring_complete_execution_not_ready",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "sources": len(cards),
        "explanations": len(available),
        "fallbacks": len(cards) - len(available),
        "pool_occurrences": 120,
        "pools": pool_summary,
        "per_label": label_summary,
        "review_reasons": dict(Counter(r["reason"] for r in audit)),
        "word_ranges": {
            f"{v}_words": [
                min(r[f"{v}_words"] for r in available),
                max(r[f"{v}_words"] for r in available),
            ]
            for v in ("specific", "generic")
        },
        "max_pair_word_difference": max(
            abs(r["specific_words"] - r["generic_words"]) for r in available
        ),
        "query_level_coverage_computed": False,
        "Qwen_token_matching_verified": False,
        "label_error_rate_estimated": False,
        "independent_human_review": False,
        "partial_relevance_results_read": False,
        "actual_API_calls": 0,
        "CUDA": False,
        "remaining": [
            "TRAIN applicability calibration",
            "benign roster decision",
            "exact prompt rendering/token accounting",
            "final study/statistics freeze",
            "runner rehearsal",
            "taxonomy implementation",
        ],
    }
    for name, content in [
        ("explanation_cards.jsonl", cards),
        ("source_review_audit.jsonl", audit),
    ]:
        with (BASE / name).open("x", encoding="utf8", newline="\n") as stream:
            for row in content:
                stream.write(
                    json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    + "\n"
                )
    text = [
        "# Reviewed correction explanation cards",
        "",
        "Assistant-authored source explanations, with conservative plain fallbacks.",
        "Not independently human-validated; no model performance result or live readiness claim.",
        "See AUTHORING_PROTOCOL.md and assistant_consistency_review_v1.md for limits.",
        "",
    ]
    for row, detail in zip(cards, audit):
        source_row = source[row["source_id"]]
        text += [
            f"## {row['source_id']}",
            "",
            f"Types: {', '.join(row['hate_types'])}. Status: {detail['status']}.",
            "",
            "~~~text",
            source_row["text"],
            "~~~",
            "",
            f"Review: {detail['note']}",
            "",
        ]
        if row["explanation_available"]:
            text += [
                "Specific: " + row["specific_explanation"],
                "",
                "Generic: " + row["generic_explanation"],
                "",
                f"Words: {row['specific_words']} specific / {row['generic_words']} generic.",
                "",
            ]
        else:
            text += [
                f"Fallback reason: {detail['reason']}. Both added explanation strings are empty.",
                "",
            ]
    with (BASE / "EXPLANATION_CARDS.md").open(
        "x", encoding="utf8", newline="\n"
    ) as stream:
        stream.write("\n".join(text) + "\n")
    save("authoring_summary.json", report)
    for name, expected in inputs.items():
        assert sha(ROOT / name) == expected, name
    save("inputs.json", inputs)
    n = active_check()
    names = [
        "explanation_cards.jsonl",
        "source_review_audit.jsonl",
        "EXPLANATION_CARDS.md",
        "authoring_summary.json",
        "inputs.json",
    ]
    save(
        "cards_prepared.json",
        {
            "status": "source_cards_prepared_independent_code_check_pending",
            "active_bindings_unchanged": n,
            "outputs": {name: sha(BASE / name) for name in names},
        },
    )
    print(
        json.dumps(
            {
                k: report[k]
                for k in (
                    "status",
                    "sources",
                    "explanations",
                    "fallbacks",
                    "per_label",
                    "word_ranges",
                    "max_pair_word_difference",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    with patch.object(
        socket,
        "create_connection",
        side_effect=RuntimeError("Offline authoring assembly"),
    ), patch.object(
        socket.socket, "connect", side_effect=RuntimeError("Offline authoring assembly")
    ):
        run()

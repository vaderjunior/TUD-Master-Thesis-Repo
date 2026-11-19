from collections import Counter, defaultdict
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import socket
from statistics import mean
from analysis.knowledge.policy import setup as s
from analysis.common.scoring import truth as original_truth


def select_independently(records, target):
    groups = defaultdict(list)
    for r in records:
        if type(r.get("gate")) is bool:
            groups[r["source"], r["gate"]].append(r["key"])
    total = sum((len(v) for v in groups.values()))
    shares = {g: Fraction(len(v) * target, total) for g, v in groups.items()}
    counts = {g: value.numerator // value.denominator for g, value in shares.items()}
    for group in sorted(groups, key=lambda g: (-(shares[g] - counts[g]), g))[
        : target - sum(counts.values())
    ]:
        counts[group] += 1
    selected = []
    for group, keys in groups.items():
        ranked = sorted(
            keys,
            key=lambda k: hashlib.sha256(
                (s.SELECTION_SEED + "|" + k).encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(ranked[: counts[group]])
    return sorted(selected)


def policy_block(records, definitions):
    if definitions:
        lines = []
        for r in records:
            dim = r.get("dimension", r.get("meta", {}).get("dimension"))
            label = r.get("label", r.get("meta", {}).get("label"))
            lines.append(
                "- "
                + dim
                + (" / " + label if label else "")
                + ": "
                + " ".join(r["text"].split())
            )
        return "## Label definitions\n" + "\n".join(lines)
    return "## Annotation guidelines\n" + "\n".join(("- " + r["text"] for r in records))


def direct_messages(old, inventory):

    defs = policy_block(old["context"]["definitions"], True)
    guidelines = policy_block(old["context"]["guidelines"], False)
    prefix = defs + "\n\n" + guidelines + "\n\n"
    user = old["messages"][1]["text"]
    s.require(
        user.startswith(prefix), "Saved policy prefix differs from independent renderer"
    )
    suffix = user[len(prefix) :]
    all_defs = policy_block([r for r in inventory if r["kind"] == "definition"], True)
    all_guides = policy_block([r for r in inventory if r["kind"] == "guideline"], False)
    variants = [old["messages"]]
    for new_prefix in (
        all_defs + "\n\n" + guidelines + "\n\n",
        all_defs + "\n\n" + all_guides + "\n\n",
    ):
        variants.append(
            [old["messages"][0], {"role": "user", "text": new_prefix + suffix}]
        )
    return dict(zip(s.CONDITIONS, variants))


def check():
    protocol = s.read(s.BASE / "protocol.json")
    for name, expected in s.read(s.BASE / "frozen_source_bindings.json").items():
        s.require(s.sha(s.ROOT / name) == expected, "Source changed: " + name)
    for name, expected in protocol["artifacts"].items():
        s.require(s.sha(s.BASE / name) == expected, "Design artifact changed: " + name)
    endpoints = s.read(s.BASE / "endpoints.json")
    selected_rows = list(s.rows(s.BASE / "roster.jsonl"))
    selected = {r["key"] for r in selected_rows}
    gold = {r["key"]: r for r in s.rows(s.GOLD)}
    s.require(len(selected) == len(selected_rows) == 6906, "Input identity count")
    for r in selected_rows:
        s.require(
            {k: v for k, v in r.items() if k != "analysis_endpoints"} == gold[r["key"]],
            "Roster content changed",
        )
        for dimension in ("hate", "hate_type", "target_group", "severity", "legal"):
            s.require(
                s.truth(r, dimension) == original_truth(r, dimension),
                "Gold semantics differ",
            )
    for name, target, phase in [
        ("en_hate", 2400, "en_test"),
        ("de_hate", 1500, "de_test"),
    ]:
        s.require(
            select_independently(
                [r for r in gold.values() if r["phase"] == phase], target
            )
            == endpoints[name]["keys"],
            "Sample differs",
        )
    for name, ep in endpoints.items():
        if ep["dimension"] != "hate":
            expected = sorted(
                (
                    r["key"]
                    for r in gold.values()
                    if r["phase"] == ep["phase"]
                    and original_truth(r, ep["dimension"]) is not None
                )
            )
            s.require(expected == ep["keys"], "Fine annotation roster differs")
        support = Counter()
        for key in ep["keys"]:
            value = original_truth(gold[key], ep["dimension"])
            support.update(
                ["hate" if value else "not_hate"]
                if ep["dimension"] == "hate"
                else [value] if ep["dimension"] == "severity" else value
            )
        s.require(
            {label: support[label] for label in ep["all_labels"]} == ep["gold_support"],
            "Support differs",
        )
        s.require(
            ep["macro_labels"]
            == [label for label in ep["all_labels"] if support[label] >= 10],
            "Macro scope differs",
        )
    inventory = [r["record"] for r in s.read(s.BASE / "policy_inventory.json")]
    inventory_map = {r["id"]: r for r in inventory}
    frozen_policy = [
        r for r in s.rows(s.RECORDS) if r["kind"] in ("definition", "guideline")
    ]
    s.require(inventory == frozen_policy, "Inventory content/order changed")
    expected_tokens = {r["key"]: r for r in s.rows(s.BASE / "prompt_bindings.jsonl")}
    s.require(set(expected_tokens) == selected, "Prompt identity roster differs")
    count_tokens = s.counter()
    seen = set()
    examples = set()
    token_sums = Counter()
    message_bytes = Counter()
    policy_counts = defaultdict(Counter)
    legal_slots = defaultdict(Counter)
    order_changes = Counter()
    max_tokens = Counter()
    preview = []
    full_order = {r["id"]: i for i, r in enumerate(inventory)}
    for line, old in enumerate(s.rows(s.PROMPTS), 1):
        key = old["key"]
        if key not in selected:
            continue
        s.require(key not in seen, "Duplicate original prompt")
        seen.add(key)
        saved = expected_tokens[key]
        s.require(
            saved["source_prompt_line"] == line
            and saved["population"] == gold[key]["phase"],
            "Prompt provenance differs",
        )
        s.require(
            s.signature(old["messages"]) == saved["original_messages_sha256"],
            "Original messages changed",
        )
        s.require(
            s.signature(old["context"]) == saved["original_context_sha256"],
            "Original context changed",
        )
        ctx = old["context"]
        s.require(
            len(ctx["definitions"]) == len(ctx["guidelines"]) == 2,
            "Original policy count differs",
        )
        s.require(
            sum((len(rr) for _, rr in ctx["example_groups"])) == 5,
            "Original example count differs",
        )
        for _, rr in ctx["example_groups"]:
            examples.update((r["id"] for r in rr))
        for channel in ("definitions", "guidelines"):
            ids = [r["id"] for r in ctx[channel]]
            s.require(len(set(ids)) == 2, "Duplicate original policy record")
            order_changes[channel] += ids != sorted(ids, key=full_order.get)
            for r in ctx[channel]:
                original = inventory_map[r["id"]]
                s.require(r["text"] == original["text"], "Policy wording differs")
                for field in ("kind", "lang", "source", "dimension", "label"):
                    s.require(
                        r["meta"][field] == original[field], "Policy metadata differs"
                    )
                policy_counts[old["population"]][r["id"]] += 1
        legal_slots[old["population"]][
            sum((r["meta"]["dimension"] == "legal" for r in ctx["guidelines"]))
        ] += 1
        variants = direct_messages(old, inventory)
        for condition, messages in variants.items():
            s.require(
                s.signature(messages) == saved["message_hashes"][condition],
                "Independent prompt hash mismatch",
            )
            s.require(messages[0] == old["messages"][0], "System message changed")
            s.require(
                messages[1]["text"].endswith("\n" + gold[key]["text"]),
                "Query bytes changed",
            )
            n = count_tokens(messages)
            s.require(n == saved["counts"][condition], "Token count mismatch")
            s.require(n < 8192, "Initial overflow")
            token_sums[condition] += n
            max_tokens[condition] = max(max_tokens[condition], n)
            message_bytes[condition] += len(
                json.dumps(messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
            )
        if len(preview) < 3:
            preview.append(
                {
                    "key": key,
                    "independent_message_hashes": {
                        c: s.signature(v) for c, v in variants.items()
                    },
                }
            )
        if len(seen) % 1500 == 0:
            print("Independently checked %s/6906 prompts." % len(seen), flush=True)
    s.require(seen == selected, "Missing prompt")
    bank_examples = {
        r["id"]
        for r in s.rows(s.RECORDS)
        if r["kind"] == "example" and r["id"] in examples
    }
    s.require(examples == bank_examples, "Example outside frozen bank")
    fixtures = 0
    for row, dim, expected in [
        ({"gate": None}, "hate", None),
        ({"gate": False, "hate_types": ["explicit"]}, "hate_type", None),
        ({"gate": True, "target_groups": None}, "target_group", None),
        ({"gate": True, "target_groups": []}, "target_group", []),
        ({"gate": False, "legal": ["propaganda"]}, "legal", ["propaganda"]),
        ({"gate": False, "legal": []}, "legal", []),
    ]:
        s.require(
            s.truth(row, dim) == original_truth(row, dim) == expected,
            "Gold boundary fixture",
        )
        fixtures += 1
    toy = [
        {"key": "toy" + str(i), "source": "a" if i < 5 else "b", "gate": i % 2 == 0}
        for i in range(12)
    ]
    for target in range(1, 13):
        actual, _ = s.sampled_keys(toy, target)
        s.require(
            actual
            == select_independently(toy, target)
            == select_independently(list(reversed(toy)), target),
            "Sampling boundary fixture",
        )
        fixtures += 1
    changed = deepcopy(inventory[0])
    changed["text"] += " changed"
    s.require(
        s.signature(changed) != s.signature(inventory[0]),
        "Policy mutation not detected",
    )
    fixtures += 1
    try:
        socket.create_connection(("offline.invalid", 443))
    except RuntimeError as exc:
        s.require("forbidden" in str(exc), "Unexpected network guard")
        fixtures += 1
    else:
        raise RuntimeError("Network guard failed")
    budget = s.read(s.BASE / "budget.json")
    s.require(
        sum(token_sums.values()) * 9 == budget["initial_input_tokens"],
        "Token budget differs",
    )
    identities = {
        (key, c, r) for key in selected for c in s.CONDITIONS for r in (0, 1, 2)
    }
    s.require(
        len(identities) == budget["counts"]["predictions"] == len(selected) * 9,
        "Prediction identity count differs",
    )
    s.require(
        budget["counts"]["ordinary_calls"] == len(identities) * 3,
        "Ordinary count differs",
    )
    s.require(
        budget["logical_call_ceiling"] == len(identities) * 9
        and budget["physical_POST_ceiling"] == len(identities) * 45,
        "Retry ceilings differ",
    )
    s.save(
        "independent_checks.json",
        {
            "status": "PASS",
            "created_utc": __import__("datetime")
            .datetime.now(__import__("datetime").timezone.utc)
            .isoformat(),
            "input_prompts": len(seen),
            "independently_reconstructed_condition_prompts": len(seen) * 3,
            "prediction_identities": len(identities),
            "boundary_checks": fixtures,
            "policy_records": len(inventory),
            "unique_original_examples": len(examples),
            "token_sums_before_outer_repeats_and_votes": dict(token_sums),
            "maximum_initial_tokens": dict(max_tokens),
            "serialized_messages_utf8_bytes": dict(message_bytes),
            "original_policy_usage_by_population": dict(policy_counts),
            "original_legal_guideline_slots_by_population": dict(legal_slots),
            "original_relative_order_changed_in_complete_block": dict(order_changes),
            "sample_hash_checks": preview,
            "verifier_sha256": s.sha(Path(__file__)),
            "source_preservation": {
                name: s.sha(s.ROOT / name) == value
                for name, value in s.read(
                    s.BASE / "frozen_source_bindings.json"
                ).items()
            },
            "limitations": "Offline design validation. Does not certify a runnable queue, live context limit, future model revision or future transport outcomes.",
        },
    )
    print(
        "PASS: 20,718 independently reconstructed prompts; all tokens and samples match. Boundary checks: "
        + str(fixtures)
    )


if __name__ == "__main__":
    with s.offline():
        check()

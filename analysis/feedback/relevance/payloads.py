from copy import deepcopy
import hashlib
import json
import socket
from unittest.mock import patch
from analysis.feedback.relevance import plan as design
from analysis.feedback.source import run as prior
from analysis.feedback.source import design as original_design
from analysis.common import prompt as renderer

h = design.h
a = prior.a
ROOT = h.ROOT
STUDY = design.OUT
BASE = STUDY / "payloads_v1"
SCIENCE = STUDY / "scientific_plan_v1.json"


def independent_pick(ids, seed, query_id, purpose):
    values = []
    for iid in ids:
        message = "|".join([design.SELECTION_SEED, str(seed), query_id, purpose, iid])
        values.append((hashlib.sha256(message.encode("utf-8")).hexdigest(), iid))
    return sorted(values)[0][1]


def run():
    h.require(not BASE.exists(), "Preserve existing payload preparation")
    science = h.read(SCIENCE)
    frozen = h.read(STUDY / "design_complete.json")
    h.require(
        h.digest(SCIENCE) == frozen["scientific_plan_sha256"]
        and h.digest(STUDY / "choices.jsonl") == frozen["choices_sha256"],
        "Frozen choices changed",
    )
    for name, sha in science["inputs"].items():
        h.require(h.digest(ROOT / name) == sha, "Scientific input changed: " + name)
    sources = [
        __file__,
        prior.__file__,
        renderer.__file__,
        design.__file__,
        ROOT / "analysis/feedback/source/design.py",
        ROOT / "src/hsrag/prompt.py",
        ROOT / "prompts/classify_v1.txt",
        ROOT / "config/taxonomy.yaml",
    ]
    inputs = {
        **science["inputs"],
        h.rel(SCIENCE): h.digest(SCIENCE),
        h.rel(STUDY / "choices.jsonl"): h.digest(STUDY / "choices.jsonl"),
    }
    inputs.update({h.rel(p): h.digest(p) for p in sources})
    h.save(
        BASE / "plan.json",
        {
            "created_utc": h.now(),
            "inputs": inputs,
            "scientific_plan_sha256": h.digest(SCIENCE),
            "scope": "5600 prompts; no retrieval or classifier calls. Old base context and natural prompts must match exactly.",
            "actual_API_calls": 0,
        },
    )
    roster = {r["id"]: r for r in h.rows(h.OLD / "feasibility_v1/roster.jsonl")}
    bases = {
        r["item_id"]: r for r in h.rows(h.OLD / "feasibility_v1/baseline_prompts.jsonl")
    }
    added = {
        r["provenance"]["source_id"]: r
        for r in h.rows(h.OLD / "update_payloads_v1/added_records.jsonl")
    }
    pools = {
        p["condition"]: p
        for p in h.read(h.OLD / "update_payloads_v1/pools.json")["pools"]
    }
    choices = h.rows(STUDY / "choices.jsonl")
    transfer = set(science["rosters"]["evaluation"])
    natural = {}
    with (h.OLD / "update_payloads_v1/prompts.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if (
                row["condition"].startswith("error_")
                and row["condition"].endswith("_b04")
                and (row["item_id"] in transfer)
            ):
                natural[row["item_id"], row["condition"]] = row
    h.require(len(natural) == len(choices) == 1050, "Natural prompt matrix incomplete")
    payloads, inventory = ([], [])
    natural_matches = 0
    for choice in choices:
        iid = choice["query_id"]
        seed = choice["pool_seed"]
        pool = pools[choice["old_pool_condition"]]
        gold = set(roster[iid]["gold"]["hate_type"])
        full = [
            j for j in pool["ids"] if not gold - set(roster[j]["gold"]["hate_type"])
        ]
        if full:
            relevant = full
        else:
            counts = {
                j: sum((l in roster[j]["gold"]["hate_type"] for l in gold))
                for j in pool["ids"]
            }
            relevant = [j for j, n in counts.items() if n == max(counts.values())]
        disjoint = [
            j
            for j in pool["ids"]
            if not any((l in gold for l in roster[j]["gold"]["hate_type"]))
        ]
        h.require(
            independent_pick(relevant, seed, iid, "relevant")
            == choice["relevant_source_id"]
            and independent_pick(disjoint, seed, iid, "disjoint")
            == choice["disjoint_source_id"],
            "Independent oracle source differs",
        )
        base = bases[iid]
        h.require(
            base["text"] == roster[iid]["text"] and base["key"] == roster[iid]["key"],
            "Base text differs",
        )
        h.require(
            len(base["context"]["definitions"])
            == len(base["context"]["guidelines"])
            == 2
            and len(base["context"]["example_groups"]) == 1
            and (len(base["context"]["example_groups"][0][1]) == 5),
            "Base counts differ",
        )
        for policy in design.POLICIES:
            field = (
                "natural_source_id"
                if policy == "natural"
                else (
                    "relevant_source_id"
                    if policy.startswith("relevant")
                    else "disjoint_source_id"
                )
            )
            sid = choice[field]
            source = added[sid]
            h.require(
                sid in pool["ids"] and source["text"] == roster[sid]["text"],
                "Example source differs",
            )
            expected_meta = {
                "gate": True,
                "hate_types": roster[sid]["gold"]["hate_type"],
                "target_groups": None,
                "severity": None,
                "legal": None,
            }
            h.require(
                source["meta"] == expected_meta and sid not in transfer,
                "Example label/source eligibility differs",
            )
            text = (
                science["prompt"]["labels_only_placeholder"]
                if policy.endswith("_labels")
                else source["text"]
            )
            record = {
                "id": source["id"],
                "text": text,
                "meta": deepcopy(source["meta"]),
                "via": "rrf",
            }
            context = deepcopy(base["context"])
            context["example_groups"].append(["Additional labelled examples", [record]])
            messages = renderer.render_messages(
                base["text"], base["lang"], context, "corrected"
            )
            alternate_source = deepcopy(source)
            alternate_source["text"] = text
            alternate_context = original_design.append_feedback(
                base["context"], alternate_source
            )
            h.require(
                alternate_context == context, "Independent feedback append differs"
            )
            h.require(
                base["context"] == bases[iid]["context"]
                and context["definitions"] == base["context"]["definitions"]
                and (context["guidelines"] == base["context"]["guidelines"])
                and (
                    context["example_groups"][:-1] == base["context"]["example_groups"]
                ),
                "Ordinary context changed",
            )
            if policy == "natural":
                old = natural[iid, pool["condition"]]
                h.require(
                    messages == old["messages"] and context == old["context"],
                    "Natural prompt changed from completed experiment",
                )
                natural_matches += 1
            row = {
                "key": base["key"],
                "item_id": iid,
                "condition": f"{policy}_s{seed}",
                "policy": policy,
                "pool_seed": seed,
                "strategy": "hybrid",
                "subset": roster[iid]["role"],
                "text": base["text"],
                "lang": base["lang"],
                "source_id": sid,
                "source_gold_types": source["meta"]["hate_types"],
                "full_cover_available": choice["full_cover_available"],
                "context": context,
                "messages": messages,
                "context_sha256": a.signature(context),
                "messages_sha256": a.signature(messages),
                "scientific_plan_sha256": frozen["scientific_plan_sha256"],
            }
            a.body(messages)
            payloads.append(row)
    for iid in science["rosters"]["evaluation"]:
        base = bases[iid]
        rendered = renderer.render_messages(
            base["text"], base["lang"], base["context"], "corrected"
        )
        h.require(rendered == base["messages"], "Reference rerender differs")
        payloads.append(
            {
                "key": base["key"],
                "item_id": iid,
                "condition": "reference",
                "policy": "reference",
                "pool_seed": None,
                "strategy": "hybrid",
                "subset": roster[iid]["role"],
                "text": base["text"],
                "lang": base["lang"],
                "source_id": None,
                "source_gold_types": None,
                "full_cover_available": iid in science["rosters"]["primary_full_cover"],
                "context": base["context"],
                "messages": rendered,
                "context_sha256": a.signature(base["context"]),
                "messages_sha256": a.signature(rendered),
                "scientific_plan_sha256": frozen["scientific_plan_sha256"],
            }
        )
    expected = {
        (f"en/{i}", f"{p}_s{s}")
        for i in transfer
        for p in design.POLICIES
        for s in h.SEEDS
    } | {(f"en/{i}", "reference") for i in transfer}
    h.require(
        len(payloads) == 5600
        and {(r["key"], r["condition"]) for r in payloads} == expected,
        "Payload matrix differs",
    )
    payloads.sort(key=lambda r: (r["key"], r["condition"]))
    for row in payloads:
        inventory.append(
            {
                k: row[k]
                for k in [
                    "key",
                    "condition",
                    "policy",
                    "pool_seed",
                    "source_id",
                    "messages_sha256",
                    "context_sha256",
                ]
            }
            | {"entry_sha256": a.signature(row)}
        )
    h.save_rows(BASE / "prompts.jsonl", payloads)
    h.save_rows(BASE / "inventory.jsonl", inventory)
    h.require(
        all((h.digest(ROOT / n) == sha for n, sha in inputs.items())),
        "Input changed while rendering",
    )
    h.save(
        BASE / "complete.json",
        {
            "status": "all_relevance_payloads_verified",
            "created_utc": h.now(),
            "plan_sha256": h.digest(BASE / "plan.json"),
            "scientific_plan_sha256": h.digest(SCIENCE),
            "inputs": inputs,
            "outputs": {
                h.rel(p): h.digest(p)
                for p in [BASE / "prompts.jsonl", BASE / "inventory.jsonl"]
            },
            "entries": len(payloads),
            "natural_exact_matches": natural_matches,
            "reference_exact_matches": 350,
            "independent_choices_checked": len(choices),
            "unchanged_base_contexts": len(payloads),
            "actual_API_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                "entries": len(payloads),
                "natural_exact_matches": natural_matches,
                "reference_exact_matches": 350,
                "actual_API_calls": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    with patch.object(
        socket.socket,
        "connect",
        side_effect=RuntimeError("No network in prompt preparation"),
    ), patch.object(
        socket,
        "create_connection",
        side_effect=RuntimeError("No network in prompt preparation"),
    ):
        run()

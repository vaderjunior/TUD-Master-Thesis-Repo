import argparse
from collections import Counter
from copy import deepcopy
from analysis.feedback.reserved import common as c
from analysis.common import prompt as renderer
from analysis.common import feedback_inputs as tokenizer_util


def sources():
    return c.input_files() + [
        c.BASE / "STUDY_PROTOCOL.md",
        c.ROOT / "analysis/feedback/reserved/common.py",
        c.ROOT / "analysis/feedback/reserved/prepare.py",
    ]


def main():
    if (c.BASE / "prepared.json").exists():
        c.verify(c.read(c.BASE / "preparation_plan.json")["inputs"])
        c.verify(c.read(c.BASE / "prepared.json")["outputs"])
        print("Existing A15 prepared prompts verified; no rewrite.")
        return
    c.require(not (c.BASE / "execution/live").exists(), "Preserve live execution")
    c.verify_parent(c.A14)
    c.verify_parent(c.A12)
    inputs = c.bindings(sources())
    plan_path = c.BASE / "preparation_plan.json"
    if plan_path.exists():
        c.require(
            c.read(plan_path)["inputs"] == inputs,
            "Interrupted preparation inputs changed",
        )
    else:
        c.save(
            plan_path,
            {
                "created_utc": c.utc(),
                "inputs": inputs,
                "new_model_calls": 0,
                "parent_audits": "Consumed frozen prompt/choice artifacts and independent-check receipts rehashed; old indexes never opened.",
            },
        )
    roster, pools, records, references, natural, corrections, choices = c.load_sources()
    conditions = c.condition_list(pools)
    science = {
        "conditions": conditions,
        "inputs": c.N_INPUTS,
        "predictions": c.N_PREDICTIONS,
        "ordinary_calls": c.N_CALLS,
        "pool_seeds": c.SEEDS,
        "pool_budget": 40,
        "decoder_repeats": 3,
        "votes_per_prediction": 3,
        "preparation_plan_sha256": c.sha(plan_path),
        "protocol_sha256": c.sha(c.BASE / "STUDY_PROTOCOL.md"),
        "primary": {
            "treatment": "reserved5__error_balanced",
            "control": "reference5",
            "endpoints": ["FPR", "FNR"],
            "confidence": 0.975,
            "bootstrap_seeds": [2026101201, 2026101202],
        },
        "mechanism": {
            "treatment": "reserved5__error_balanced",
            "controls": [
                "natural5__error_balanced",
                "extended6__error_balanced",
                "masked5__error_balanced",
            ],
            "endpoints": ["FPR", "FNR"],
            "confidence": 1 - 0.05 / 6,
            "bootstrap_seeds": list(range(2026101210, 2026101216)),
        },
        "bootstrap_draws": 10000,
        "unit": "whole original text-overlap cluster, dependent pool/repeat outputs retained",
        "supporting_confidence": 0.95,
        "scope": "Exposed DEV, post-A14 prospective extension, simulated binary feedback",
        "mandatory_excluded_keys": c.read(
            c.A14 / "cold_replay_review_v1/sensitivity_plan.json"
        )["excluded_keys"],
    }
    c.require(
        len(conditions) == c.N_CONDITIONS
        and len(science["mandatory_excluded_keys"]) == 6,
        "Wrong A15 scope",
    )
    c.save(c.BASE / "scientific_plan_v1.json", science)
    c.jsonlines(c.BASE / "roster.jsonl", roster)
    c.jsonlines(c.BASE / "choices.jsonl", (choices[k] for k in sorted(choices)))
    science_sha = c.sha(c.BASE / "scientific_plan_v1.json")
    tok = tokenizer_util.tokenizer()
    token_counts = []
    counts = Counter()
    identity_counts = Counter()
    pool_by_name = {p["condition"]: p for p in pools}

    def produce():
        for number, r in enumerate(roster, 1):
            base = references[r["key"]]
            c.require(base["text"] == r["text"], "Base query text changed")
            for condition in conditions:
                sid = None
                cosine = None
                original_sha = None
                if condition in ("reference5", "reference4"):
                    mode = condition
                    context = c.intervene(base["context"], mode)
                else:
                    mode, pool = condition.split("__", 1)
                    chosen = corrections[r["key"], pool]
                    sid = chosen["source_id"]
                    choice = choices[r["key"], pool]
                    c.require(
                        sid == choice["source_id"] and sid in pool_by_name[pool]["ids"],
                        "Choice outside frozen pool",
                    )
                    source = records[sid]
                    hit = chosen["hit"]
                    cosine = choice["cosine"]
                    c.require(
                        hit["id"] == source["id"]
                        and hit["text"] == source["text"]
                        and (hit["meta"] == source["meta"]),
                        "Correction source content changed",
                    )
                    c.require(
                        sid != r["id"]
                        and " ".join(hit["text"].split())
                        != " ".join(r["text"].split()),
                        "Source leaks query",
                    )
                    if mode == "natural5":
                        context = deepcopy(natural[r["key"], pool]["context"])
                        original_sha = natural[r["key"], pool]["messages_sha256"]
                    else:
                        context = c.intervene(base["context"], mode, hit)
                messages = renderer.render_messages(
                    r["text"], "en", context, "corrected"
                )
                c.require(
                    context["definitions"] == base["context"]["definitions"]
                    and context["guidelines"] == base["context"]["guidelines"],
                    "Abstract knowledge differs",
                )
                c.require(messages[0] == base["messages"][0], "System prompt differs")
                if condition == "reference5":
                    original_sha = base["messages_sha256"]
                if original_sha:
                    c.require(
                        c.sig(messages) == original_sha,
                        "Frozen parent message not reproduced",
                    )
                n = tokenizer_util.token_count(tok, messages)
                token_counts.append(n)
                counts[mode] += 1
                identity_counts["same_as_reference"] += messages == base["messages"]
                yield {
                    "key": r["key"],
                    "item_id": r["id"],
                    "condition": condition,
                    "population": r["population"],
                    "lang": "en",
                    "text": r["text"],
                    "context": context,
                    "messages": messages,
                    "context_sha256": c.sig(context),
                    "messages_sha256": c.sig(messages),
                    "reference_input_tokens": n,
                    "source_id": sid,
                    "selected_correction_cosine": cosine,
                    "parent_messages_sha256": original_sha,
                    "scientific_plan_sha256": science_sha,
                }
            if number % 50 == 0 or number == len(roster):
                print(
                    f"A15 prepared {number}/{len(roster)} inputs; no model calls.",
                    flush=True,
                )

    c.jsonlines(c.BASE / "prompts.jsonl", produce())
    c.require(sum(counts.values()) == c.N_PROMPTS, "Wrong prompt count")
    c.save(
        c.BASE / "summary.json",
        {
            "status": "prepared_requires_independent_check",
            "created_utc": c.utc(),
            "inputs": len(roster),
            "conditions": len(conditions),
            "prompts": sum(counts.values()),
            "predictions": c.N_PREDICTIONS,
            "ordinary_calls": c.N_CALLS,
            "counts": dict(counts),
            "identity_counts": dict(identity_counts),
            "token_min": min(token_counts),
            "token_max": max(token_counts),
            "new_model_calls": 0,
            "new_embeddings": 0,
            "CUDA_used": False,
        },
    )
    c.verify(inputs)
    output_paths = [
        c.BASE / n
        for n in (
            "scientific_plan_v1.json",
            "roster.jsonl",
            "prompts.jsonl",
            "choices.jsonl",
            "summary.json",
        )
    ]
    c.save(
        c.BASE / "prepared.json",
        {
            "status": "prepared_requires_independent_check",
            "outputs": c.bindings(output_paths),
            "created_utc": c.utc(),
        },
    )
    print(
        "A15 preparation complete. Independent checking is required before execution.",
        flush=True,
    )


if __name__ == "__main__":
    with c.offline():
        main()

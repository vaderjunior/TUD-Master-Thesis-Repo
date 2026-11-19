from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
from pathlib import Path
import numpy as np
from analysis.common import replay as c
from analysis.common import prompt as prompt

BASE = c.BASE / "evaluation_preparation"
SEEDS = [202610061, 202610062, 202610063]


def signature(r):
    m = r["meta"]
    return (
        r["lang"],
        r["source"],
        m.get("gate"),
        *(
            len(m[k]) if m.get(k) is not None else -1
            for k in ["target_groups", "hate_types", "legal"]
        ),
        m.get("severity") is not None,
        bool(m.get("astra_aux_stgb")),
    )


def main():
    c.verify(c.read(BASE / "plan.json")["inputs"])
    c.verify(c.read(BASE / "complete.json")["outputs"])
    check = c.read(BASE / "verification.json")
    assert check["status"] == "independent_offline_preparation_pass"
    kb = {
        r["id"]: r
        for r in c.rows(c.CORE / "kb/full/records.jsonl")
        if r["kind"] == "example"
    }
    buckets = defaultdict(list)
    for rid, r in sorted(kb.items()):
        buckets[signature(r)].append(rid)
    input_paths = [
        Path(__file__).resolve(),
        BASE / "plan.json",
        BASE / "matched_prompts.jsonl",
        BASE / "verification.json",
        c.CORE / "kb/full/records.jsonl",
    ]
    c.save(
        BASE / "coverage_amendment.json",
        {
            "created_utc": c.utc(),
            "status": "prospective_before_any_new_generation",
            "inputs": {c.rel(p): c.sha(p) for p in input_paths},
            "reason": "The pre-outcome annotation check found all three uniform German fixed sets have zero legal gold; two English fixed sets have zero type annotations. Retain those arms as context but add controls for coverage before interpreting matching.",
            "added_conditions": [f"coverage_matched_{s}" for s in SEEDS],
            "selection": "For every query and each of its five retrieved slots, draw a different TRAIN example with the same language, source, hate-gate value, known/missing and cardinality of target/type/legal, severity availability and auxiliary legal-flag availability. Sample deterministically using the declared seed/query; exclude all original five examples and already drawn examples.",
            "fallback": "If no eligible distinct substitute exists, retain the original slot unchanged and record it. Do not drop/resample inputs or relax the signature after seeing outcomes.",
            "primary_amendment": "Replace the three retrieved-minus-uniform-fixed focal contrasts with retrieved-minus-mean-coverage-matched controls. Keep the three retrieved-minus-permutation focal contrasts: six questions, 99.1666667% family confidence. Uniform fixed, text-only, labels-only and bridge comparisons remain descriptive.",
            "scope": "Same 941 input roster, abstracts/instruction/query/schema unchanged. Three fixed selection seeds separate from three decoder repeats. No query gold is used in selection. No result-based design changes.",
            "remaining_limits": "Actual label identities, lexical content, text length and other information can still differ. This controls source/availability/cardinality/gate prior, not every possible confound. Annotation matching is a diagnostic and does not require query gold at deployment.",
            "execution": "Additive 13-setting design, 110097 ordinary calls before repairs/retries. No Qwen requests or launcher.",
        },
    )
    sources = [
        r
        for r in c.rows(BASE / "matched_prompts.jsonl")
        if r["condition"] == "retrieved_complete"
    ]
    output = []
    fallbacks = []
    stats = Counter()
    for s in sources:
        original = s["example_text_ids"]
        blocked = set(original)
        for seed in SEEDS:
            rng = np.random.default_rng(
                int(hashlib.sha256(f"{seed}|{s['key']}".encode()).hexdigest()[:16], 16)
            )
            chosen = []
            for slot, old in enumerate(original):
                bucket = buckets[signature(kb[old])]
                unavailable = blocked | set(chosen)
                available_count = len(bucket) - sum((r in unavailable for r in bucket))
                if not available_count:
                    rid = old
                    fallbacks.append(
                        {
                            "key": s["key"],
                            "seed": seed,
                            "slot": slot,
                            "id": old,
                            "bucket_size": len(bucket),
                        }
                    )
                else:
                    while True:
                        rid = bucket[int(rng.integers(len(bucket)))]
                        if rid not in unavailable:
                            break
                assert signature(kb[rid]) == signature(kb[old])
                chosen.append(rid)
            assert len(set(chosen)) == 5
            assert all(
                (rid == old or rid not in blocked for rid, old in zip(chosen, original))
            )
            lines = []
            for i, rid in enumerate(chosen):
                r = kb[rid]
                text = " ".join(r["text"].split())
                lines.append(
                    str(i + 1)
                    + '. "'
                    + text
                    + '"\n   labels: '
                    + prompt.corrected_gold(r["meta"])
                )
            row = deepcopy(s)
            row["condition"] = f"coverage_matched_{seed}"
            row["example_text_ids"] = chosen
            row["annotation_source_ids"] = list(chosen)
            row["coverage_reference_ids"] = original
            row["selection_seed"] = seed
            text = row["messages"][1]["text"]
            prefix, rest = text.split("## Example context\n", 1)
            _, target = rest.split("\n\n## Text to classify", 1)
            row["messages"][1]["text"] = (
                prefix
                + "## Example context\n"
                + "\n".join(lines)
                + "\n\n## Text to classify"
                + target
            )
            row["messages_sha256"] = c.signature(row["messages"])
            output.append(row)
            stats["unchanged_entire_prompt"] += int(row["messages"] == s["messages"])
            for a, b in zip(chosen, original):
                stats["unchanged_text_slot"] += int(
                    c.normalized(kb[a]["text"]) == c.normalized(kb[b]["text"])
                )
    assert len(output) == 2823
    c.save_rows(BASE / "coverage_matched_prompts.jsonl", output)
    c.save(
        BASE / "coverage_preparation.json",
        {
            "status": "prepared_no_generation",
            "prompts": len(output),
            "slots": len(output) * 5,
            "fallbacks": fallbacks,
            "fallback_count": len(fallbacks),
            "stats": dict(stats),
            "new_conditions": 3,
            "total_conditions": 13,
            "matched_ordinary_calls": 110097,
            "broad_plus_matched_ordinary_calls": 694026,
            "new_model_calls": 0,
            "outputs": {
                c.rel(BASE / p): c.sha(BASE / p)
                for p in ["coverage_amendment.json", "coverage_matched_prompts.jsonl"]
            },
        },
    )
    print(
        "Coverage controls prepared:",
        len(output),
        "prompts;",
        len(fallbacks),
        "unchanged fallback slots;",
        dict(stats),
    )


if __name__ == "__main__":
    with c.network_guard():
        main()

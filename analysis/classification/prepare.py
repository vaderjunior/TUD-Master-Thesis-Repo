from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
from analysis.common import replay as c
from analysis.common import prompt as prompt

BASE = c.BASE / "evaluation_preparation"
SEEDS = [202610061, 202610062, 202610063]
EPS = ["en_dev_eval_main", "en_dev_eval_sq3_types", "de_legal_dev_eval"]
CONDITIONS = [
    "retrieved_complete",
    *[f"fixed_{s}" for s in SEEDS],
    "retrieved_text_only",
    "retrieved_labels_only",
    "permuted_0",
    "permuted_1",
    "permuted_2",
    "original_bridge",
]
NOTE = "## How to use the supplied context\nUse supplied definitions and annotation guidelines when judging the input. Example context can contain texts, annotations, or both. Example selection varies. A missing annotation does not mean a negative label. Judge the input on its own terms; do not assign labels merely because an example has them."


def stable(s):
    return hashlib.sha256(s.encode("utf8")).hexdigest()


def make_clusters(rr):
    "Whole connected template/reference components; no independence of variants assumed."
    parent = {str(r["id"]): str(r["id"]) for r in rr}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        a, b = (find(a), find(b))
        if a != b:
            parent[max(a, b)] = min(a, b)

    templates = defaultdict(list)
    norms = defaultdict(list)
    missing = []
    for r in rr:
        raw = json.loads(r["raw"])
        rid = str(r["id"])
        templates[raw["templ_id"]].append(rid)
        norms[c.normalized(r["text"])].append(rid)
        ref = raw["ref_case_id"]
        if ref is not None:
            other = f"hatecheck-{ref}"
            if other in parent:
                union(rid, other)
            else:
                missing.append([rid, other])
    for group in list(templates.values()) + list(norms.values()):
        for x in group[1:]:
            union(group[0], x)
    return ({rid: find(rid) for rid in parent}, missing)


def scope_rows():
    dev = c.gold_rows("en_dev")
    targets = [r for r in dev if r["gate"] is True and r["target_groups"] is not None]
    severity = {r["id"] for r in dev if r["gate"] is True and r["severity"] is not None}
    assert (
        len(targets) == 2506
        and len(severity) == 1606
        and (severity <= {r["id"] for r in targets})
    )
    rr = []
    for r in targets:
        rr.append(
            {
                **r,
                "phase": "larger_dev",
                "source_file": "data/processed/en_dev.parquet",
                "endpoints": ["target_group"]
                + (["severity"] if r["id"] in severity else []),
            }
        )
    for name in ["en_test", "de_test", "de_legal_test", "en_hatecheck_eval"]:
        rows = c.gold_rows(name)
        clusters, missing = (
            make_clusters(rows) if name == "en_hatecheck_eval" else ({}, [])
        )
        for r in rows:
            endpoints = []
            if r.get("gate") is not None:
                endpoints.append("hate")
            if r.get("gate") is True:
                for field, dim in [
                    ("target_groups", "target_group"),
                    ("hate_types", "hate_type"),
                    ("severity", "severity"),
                ]:
                    if r.get(field) is not None:
                        endpoints.append(dim)
            if r.get("legal") is not None:
                endpoints.append("legal")
            rr.append(
                {
                    **r,
                    "phase": name,
                    "source_file": f"data/processed/{name}.parquet",
                    "endpoints": endpoints,
                    "functional_cluster": clusters.get(str(r["id"])),
                }
            )
        if missing:
            raise AssertionError(("Missing HateCheck reference", missing))
    for r in rr:
        r["key"] = r["lang"] + "/" + str(r["id"])
        r["text_cluster"] = stable(c.normalized(r["text"]))
        r["empty_input"] = not c.normalized(r["text"])
    assert len({r["key"] for r in rr}) == len(rr)
    return rr


def reference_blocks(source):
    assert (
        prompt.render_messages(
            source["text"], source["lang"], source["context"], "corrected"
        )
        == source["messages"]
    )
    ctx = source["context"]
    m = prompt.module("corrected")
    assert len(ctx["definitions"]) == len(ctx["guidelines"]) == 2
    assert len(ctx["example_groups"]) == 1 and len(ctx["example_groups"][0][1]) == 5
    d = "## Label definitions\n" + "\n".join(
        (
            "- "
            + str(h["meta"]["dimension"])
            + (" / " + str(h["meta"]["label"]) if h["meta"].get("label") else "")
            + ": "
            + m._flatten(h["text"])
            for h in ctx["definitions"]
        )
    )
    g = "## Annotation guidelines\n" + "\n".join(
        ("- " + h["text"] for h in ctx["guidelines"])
    )
    return (d, g, f"## Text to classify (language: {source['lang']})\n{source['text']}")


def main():
    prior = c.read(c.BASE / "documentation/data_provenance.json")
    source = c.CORE / "retrieval_v1/full/prompts.jsonl"
    paths = [
        Path(__file__).resolve(),
        Path(c.__file__).resolve(),
        Path(prompt.__file__).resolve(),
        source,
        c.CORE / "retrieval_v1/full/complete.json",
        c.CORE / "kb/full/records.jsonl",
        c.CORE / "kb/small/records.jsonl",
        c.CORE / "plan.json",
        c.CORE / "api_execution_v1/plan.json",
        c.BASE / "documentation/data_provenance.json",
        c.ROOT / "src/hsrag/prompt.py",
        c.ROOT / "prompts/classify_v1.txt",
        c.ROOT / "config/taxonomy.yaml",
    ]
    datasets = [
        "en_dev",
        "en_test",
        "de_test",
        "de_legal_test",
        "en_hatecheck_eval",
    ] + EPS
    paths += [c.ROOT / f"data/processed/{name}.parquet" for name in datasets]
    core = c.read(c.CORE / "plan.json")
    api = c.read(c.CORE / "api_execution_v1/plan.json")
    c.save(
        BASE / "plan.json",
        {
            "created_utc": c.utc(),
            "status": "offline_design_frozen_no_generation",
            "inputs": {c.rel(p): c.sha(p) for p in paths},
            "backend": {k: core[k] for k in ["model", "temperature", "votes_per_item"]},
            "api_contract_inherited": api["api_contract"],
            "broad_systems": [
                "rag_full",
                "rag_small",
                "zero_shot",
                "aligned_encoders_three_seeds",
            ],
            "broad_primary_family": [
                "larger_dev_targets_full_minus_small",
                "larger_dev_severity_full_minus_small",
                "en_test_hate_full_minus_small",
                "de_legal_test_full_minus_small",
            ],
            "broad_primary_confidence": 0.9875,
            "broad_repeats": 3,
            "inner_votes": 3,
            "broad_policy": "All eligible target/severity EN DEV plus complete EN TEST, DE TEST, legal TEST and English HateCheck. Fresh matched Qwen responses for all three prompting systems. Preserve exposed historical results as context, never merge old/fresh repeats.",
            "encoder_policy": "Use existing fitted seeds/checkpoints and frozen tokenization/default thresholds. Inference only on expanded rows after checksum/tokenizer preflight; no refitting or DEV threshold selection. Do not substitute old TEST predictions from different encoder versions.",
            "secondary": "All other system/endpoint contrasts, per-source/per-label outputs, functional groups, missed hate, false alarms, exact sets and costs are supporting. No universal win is inferred from multiple unadjusted tests.",
            "mask": "Pair-specific common-valid sets across all compared conditions/repeats; own-valid results, null IDs and full-roster bounds also required. Gold-hate-gated fine heads retain predicted-gate errors; legal independent. Macro labels frozen by complete-roster support>=10; separately list rare/absent labels.",
            "uncertainty": "10000 paired resamples of NFKC/casefold text clusters; HateCheck uses connected template/reference/text components with item sensitivity. Fixed fitted seeds, decoder repeats and KB pools, never treated as independent input replicates. Natural-data source-stratified sensitivity also required.",
            "matched": {
                "endpoints": EPS,
                "conditions": CONDITIONS,
                "seeds": SEEDS,
                "primary_family": "Six questions: retrieved_complete minus mean of three fixed sets, and retrieved_complete minus mean of three label permutations, separately for the three endpoints.",
                "family_confidence": 1 - 0.05 / 6,
                "bootstrap_draws": 10000,
                "secondary": "Text-only, labels-only and original-message bridge contrasts are descriptive, including validity/annotation-delivery/mutation-coverage analyses.",
                "static_selection": "Three deterministic uniform hash draws of five distinct examples from the same full TRAIN pool per language. Same set for every input in a language. No favorable resampling for label availability; endpoint-specific annotated-slot counts must accompany effects.",
                "permutation": "Three distinct deterministic derangements of the five complete rendered annotation blocks per input. Gate/type/target/severity/legal and missingness stay together. Identical label blocks and unchanged messages remain included and counted.",
                "limits": "Selection/composition, token length, label balance and order can still matter. This is not a pure semantic-retrieval or internal-reasoning identification experiment. Fixed-set and permutation seeds are not decoder repeats.",
            },
            "blocked_or_deferred": [
                "New Qwen requests until same pinned model is available",
                "Broad retrieval/vector preparation, checkpoint inference and execution-wrapper integration remain separate implementation work",
                "No selected SQ3 adaptation arm added until balanced-correction/human-dependent choices are resolved",
                "No German functional benchmark download or legal generalization from HateCheck",
            ],
            "exposure": "Previously inspected DEV/TEST/HateCheck; no untouched-test claim. All old near-duplicate audits retained, not rerun.",
        },
    )
    rr = scope_rows()
    c.save_rows(BASE / "broad_roster.jsonl", rr)
    summary = {}
    for phase in sorted({r["phase"] for r in rr}):
        rows = [r for r in rr if r["phase"] == phase]
        support = {}
        for dim in sorted({d for r in rows for d in r["endpoints"]}):
            field = c.original.GOLD_FIELDS[dim]
            eligible = [r for r in rows if dim in r["endpoints"]]
            count = Counter()
            for r in eligible:
                value = r[field]
                values = value if isinstance(value, list) else [value]
                count.update((str(v) for v in values))
            support[dim] = {
                "n": len(eligible),
                "positive_support": dict(count),
                "macro_labels": [l for l, n in sorted(count.items()) if n >= 10],
            }
        summary[phase] = {
            "inputs": len(rows),
            "empty_inputs": [r["key"] for r in rows if r["empty_input"]],
            "endpoints": support,
            "unique_text_clusters": len({r["text_cluster"] for r in rows}),
            "functional_clusters": (
                len({r["functional_cluster"] for r in rows})
                if phase == "en_hatecheck_eval"
                else None
            ),
            "sources": dict(Counter((r["source"] for r in rows))),
            "ordinary_Qwen_calls": len(rows) * 3 * 3 * 3,
        }
    sources = [
        r for r in c.rows(source) if any((s["subset"] in EPS for s in r["scopes"]))
    ]
    assert len(sources) == 941
    complete = c.read(c.CORE / "retrieval_v1/full/complete.json")
    assert complete["prompt_sha256"] == c.sha(source)
    records = [
        r for r in c.rows(c.CORE / "kb/full/records.jsonl") if r["kind"] == "example"
    ]
    static = {}
    for lang in ["en", "de"]:
        pool = [r for r in records if r["lang"] == lang]
        for seed in SEEDS:
            static[lang, seed] = sorted(
                pool, key=lambda r: stable(f"{seed}|{lang}|" + r["id"])
            )[:5]
    permutations = [
        p
        for p in itertools.permutations(range(5))
        if all((k != v for k, v in enumerate(p)))
    ]
    assert len(permutations) == 44
    frozen = []
    changes = []
    for s in sorted(sources, key=lambda r: r["key"]):
        d, g, target = reference_blocks(s)
        retrieved = s["context"]["example_groups"][0][1]
        derangements = sorted(
            permutations, key=lambda p: stable("20261006|" + s["key"] + "|" + str(p))
        )[:3]
        for cond in CONDITIONS:
            hits = (
                static[s["lang"], int(cond.split("_")[1])]
                if cond.startswith("fixed_")
                else retrieved
            )
            texts = [prompt.module("corrected")._flatten(r["text"]) for r in hits]
            labels = [prompt.corrected_gold(r["meta"]) for r in hits]
            sourceids = [r["id"] for r in hits]
            labelids = list(sourceids)
            perm = list(range(5))
            if cond.startswith("permuted_"):
                perm = list(derangements[int(cond[-1])])
                labels = [labels[k] for k in perm]
                labelids = [sourceids[k] for k in perm]
            lines = []
            for i in range(5):
                parts = []
                if cond != "retrieved_labels_only":
                    parts.append('"' + texts[i] + '"')
                if cond != "retrieved_text_only":
                    parts.append("labels: " + labels[i])
                lines.append(f"{i + 1}. " + "\n   ".join(parts))
            messages = (
                deepcopy(s["messages"])
                if cond == "original_bridge"
                else [
                    deepcopy(s["messages"][0]),
                    {
                        "role": "user",
                        "text": "\n\n".join(
                            [
                                NOTE,
                                d,
                                g,
                                "## Example context\n" + "\n".join(lines),
                                target,
                            ]
                        ),
                    },
                ]
            )
            changed = (
                sum(
                    (
                        a != b
                        for a, b in zip(
                            labels,
                            [prompt.corrected_gold(r["meta"]) for r in retrieved],
                        )
                    )
                )
                if cond.startswith("permuted_")
                else None
            )
            frozen.append(
                {
                    "key": s["key"],
                    "condition": cond,
                    "item_id": s["item_id"],
                    "lang": s["lang"],
                    "scopes": [x for x in s["scopes"] if x["subset"] in EPS],
                    "messages": messages,
                    "messages_sha256": c.signature(messages),
                    "example_text_ids": sourceids,
                    "annotation_source_ids": labelids,
                    "permutation": perm,
                    "changed_annotation_slots": changed,
                    "source_messages_sha256": s["messages_sha256"],
                    "source_context_sha256": s["context_sha256"],
                    "annotation_fields_by_slot": [
                        [
                            k
                            for k in [
                                "gate",
                                "target_groups",
                                "hate_types",
                                "severity",
                                "legal",
                            ]
                            if hits[j]["meta"].get(k) is not None
                        ]
                        for j in perm
                    ],
                }
            )
            if changed is not None:
                changes.append(
                    {"key": s["key"], "condition": cond, "changed_slots": changed}
                )
    assert len(frozen) == 9410
    c.save_rows(BASE / "matched_prompts.jsonl", frozen)
    c.save(
        BASE / "fixed_sets.json",
        {
            f"{lang}/{seed}": [r["id"] for r in hits]
            for (lang, seed), hits in static.items()
        },
    )
    result = {
        "status": "offline_rosters_and_matched_prompts_prepared_not_run",
        "plan_sha256": c.sha(BASE / "plan.json"),
        "broad_roster_rows": len(rr),
        "broad_scopes": summary,
        "broad_ordinary_calls": len(rr) * 27,
        "matched_inputs": 941,
        "matched_prompts": 9410,
        "matched_predictions": 28230,
        "matched_ordinary_calls": 84690,
        "matched_zero_slot_changes": sum((x["changed_slots"] == 0 for x in changes)),
        "matched_permutations": changes,
        "total_future_ordinary_calls": len(rr) * 27 + 84690,
        "repairs_and_retries": "Additional bounded work; not included in ordinary count",
        "new_model_calls": 0,
        "readiness": "Scientific design and rosters prepared. Matched messages materialized. Independent manipulation check required. No live launcher prepared; broad retrieval/inference preflight still needed.",
    }
    c.save(BASE / "preparation.json", result)
    c.save(
        BASE / "complete.json",
        {
            "status": "offline_preparation_complete_not_execution",
            "outputs": {
                c.rel(BASE / name): c.sha(BASE / name)
                for name in [
                    "plan.json",
                    "broad_roster.jsonl",
                    "matched_prompts.jsonl",
                    "fixed_sets.json",
                    "preparation.json",
                ]
            },
        },
    )
    print(
        "Offline preparation complete:",
        {
            k: result[k]
            for k in [
                "broad_roster_rows",
                "broad_ordinary_calls",
                "matched_prompts",
                "matched_ordinary_calls",
                "total_future_ordinary_calls",
                "matched_zero_slot_changes",
            ]
        },
    )
    print("Broader rosters:", {k: v["inputs"] for k, v in summary.items()})


if __name__ == "__main__":
    with c.network_guard():
        main()

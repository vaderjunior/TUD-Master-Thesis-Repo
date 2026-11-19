"""Components prepare."""

import argparse
from collections import Counter
from copy import deepcopy
from functools import lru_cache
import itertools
import json
import os
from pathlib import Path
import socket
import time
from unittest.mock import patch
from analysis.common import batch as core
from analysis.common import prompt as prompt

a = core.a
ROOT = a.ROOT
CORE = core.CORE
SOURCE = ROOT / "experiments/astra/sq1_full_kb_strategy_20260920_v1"
BASE = ROOT / "experiments/astra/sq1_full_kb_components_20260921_v1"
PLAN = BASE / "plan.json"
PROMPTS = BASE / "prompts.jsonl"
SOURCE_PLAN_SHA = "d22c3018572515d19943c598422ee91a8c1fe42af631d85efa17dcf7eacb6ad1"
SOURCE_PROMPTS_SHA = "691189131b734178a21b15294f764623b96212d6b5ac9d7f758402175775ad7d"
SOURCE_AUDIT_SHA = "25befd0770a23e031a6b1b816328973e49fe3aeed665e18b7817afb8fdac7023"
SOURCE_COMPLETION_SHA = "c80421c370fbdc2f2cfe7dd42e7ee2a4297408f9f691b0ae379327df8e770475"
SCOPE_COUNTS = {
    "en_dev_eval_main": 299,
    "en_dev_eval_targets": 56,
    "en_dev_eval_severity": 45,
    "en_dev_eval_sq3_types": 467,
    "en_dev_eval_types": 103,
    "de_dev_eval": 150,
    "de_legal_dev_eval": 175,
}
ENDPOINT_ORDER = (
    "en_dev_eval_sq3_types",
    "en_dev_eval_main",
    "en_dev_eval_targets",
    "en_dev_eval_severity",
    "de_dev_eval",
    "de_legal_dev_eval",
    "en_dev_eval_types",
)
ARMS = {
    "none": (0, 0, 0),
    "g": (0, 0, 1),
    "d": (0, 1, 0),
    "dg": (0, 1, 1),
    "e": (1, 0, 0),
    "eg": (1, 0, 1),
    "ed": (1, 1, 0),
    "edg": (1, 1, 1),
}
BRIDGE = "reference_rag"
CONDITIONS = tuple(ARMS) + (BRIDGE,)
COMMON_NOTE = "## How to use the supplied context\nUse any supplied label definitions and annotation guidelines when judging the input. Labelled examples, when supplied, were retrieved by similarity to the input. They may all be hateful even when the input is not. Judge the input on its own terms; do not assign labels merely because it resembles an example."
SOURCE_FILES = [
    "analysis/knowledge/components/prepare.py",
    "tests/knowledge/components/test_prepare.py",
    "analysis/knowledge/components/verify_preparation.py",
    "analysis/common/batch.py",
    "analysis/common/scoring.py",
    "analysis/common/prompt.py",
    "src/hsrag/prompt.py",
    "prompts/classify_v1.txt",
    "config/taxonomy.yaml",
    *a.DEPENDENCIES,
]


def forbidden(*args, **kwargs):
    raise AssertionError("Network and credential loading forbidden during component preparation")


def rel(path):
    return core.rel(path)


def save(path, value):
    target = a.storage_path(path).resolve()
    a.require(
        target.is_relative_to(a.storage_path(BASE).resolve()), "Write outside new component study"
    )
    a.save_new(target, value)


@lru_cache(maxsize=1)
def template_parts():
    module = prompt.module("corrected")
    template = module._load_template()
    return (template, template["SYSTEM"].format(label_space=module.render_label_space()))


def blocks(source):
    """Preserve each record's original rendering, order and corrected labels."""
    ctx = source["context"]
    a.require(
        len(ctx["definitions"]) == len(ctx["guidelines"]) == 2, "Full abstract budgets required"
    )
    a.require(
        len(ctx["example_groups"]) == 1
        and ctx["example_groups"][0][0] == "Labelled examples"
        and (len(ctx["example_groups"][0][1]) == 5),
        "Expected five regular examples",
    )
    module = prompt.module("corrected")
    d = "## Label definitions\n" + "\n".join(
        (
            "- "
            + str(hit["meta"].get("dimension"))
            + (" / " + str(hit["meta"]["label"]) if hit["meta"].get("label") else "")
            + ": "
            + module._flatten(hit["text"])
            for hit in ctx["definitions"]
        )
    )
    g = "## Annotation guidelines\n" + "\n".join(("- " + hit["text"] for hit in ctx["guidelines"]))
    hits = ctx["example_groups"][0][1]
    a.require(
        all((not h["meta"].get("illustrative_only") and h["via"] != "static" for h in hits)),
        "Unexpected illustrative/static example",
    )
    e_rows = "\n".join(
        (
            '- "' + module._flatten(h["text"]) + '"\n  labels: ' + prompt.corrected_gold(h["meta"])
            for h in hits
        )
    )
    e = "## Labelled examples\n\n" + e_rows
    target = f"## Text to classify (language: {source['lang']})\n{source['text']}"
    template, system = template_parts()
    legacy_e = "## Labelled examples\n" + template["EXAMPLES_CAUTION"] + "\n\n" + e_rows
    reconstructed = [
        {"role": "system", "text": system},
        {"role": "user", "text": "\n\n".join([d, g, legacy_e, target])},
    ]
    a.require(
        reconstructed == source["messages"], "Cannot reconstruct exact accepted full-RAG prompt"
    )
    return {"d": d, "g": g, "e": e, "target": target, "system": system}


def render(source, condition, rendered=None):
    a.require(condition in CONDITIONS, "Unknown component condition")
    if condition == BRIDGE:
        return (deepcopy(source["messages"]), deepcopy(source["context"]))
    e, d, g = ARMS[condition]
    chunk = rendered or blocks(source)
    context = {
        "definitions": deepcopy(source["context"]["definitions"]) if d else [],
        "guidelines": deepcopy(source["context"]["guidelines"]) if g else [],
        "example_groups": deepcopy(source["context"]["example_groups"]) if e else [],
    }
    parts = [COMMON_NOTE]
    for name, on in [("d", d), ("g", g), ("e", e)]:
        if on:
            parts.append(chunk[name])
    parts.append(chunk["target"])
    return (
        [{"role": "system", "text": chunk["system"]}, {"role": "user", "text": "\n\n".join(parts)}],
        context,
    )


def contrasts():

    def pair(left, right):
        return {left: 1.0, right: -1.0}

    focal = {
        "examples_given_definitions_guidelines": pair("edg", "dg"),
        "definitions_given_examples_guidelines": pair("edg", "eg"),
        "guidelines_given_examples_definitions": pair("edg", "ed"),
    }
    descriptive = {
        "examples_alone": pair("e", "none"),
        "definitions_alone": pair("d", "none"),
        "guidelines_alone": pair("g", "none"),
        "abstracts_alone": pair("dg", "none"),
        "all_context": pair("edg", "none"),
        "abstracts_given_examples": pair("edg", "e"),
        "common_wording_full_minus_reference": pair("edg", BRIDGE),
    }
    factorial = {}
    for size in (1, 2, 3):
        for chosen in itertools.combinations(range(3), size):
            name = "_x_".join(("edg"[index] for index in chosen))
            weights = {}
            for arm, bits in ARMS.items():
                sign = (-1) ** sum((1 - bits[index] for index in chosen))
                weights[arm] = sign / 2 ** (3 - size)
            factorial[name] = weights
    return {
        "focal_leave_one_out": focal,
        "descriptive_simple": descriptive,
        "descriptive_factorial": factorial,
    }


def analysis_plan(source_plan):
    endpoints = {}
    for subset, row in source_plan["endpoints_inherited_masks_and_labels"].items():
        endpoints[subset] = {
            **row,
            "inherited_core_role": row["role"],
            "role": (
                "focal"
                if subset == "en_dev_eval_sq3_types"
                else "sensitivity" if subset == "en_dev_eval_types" else "secondary"
            ),
            "source_scope": {
                "en_dev_eval_sq3_types": "hate_type_primary",
                "en_dev_eval_types": "hate_type_secondary",
            }.get(subset, row["dimension"]),
        }
    return {
        "endpoints": endpoints,
        "conditions": list(CONDITIONS),
        "repeats": [0, 1, 2],
        "focal_endpoint": "en_dev_eval_sq3_types",
        "focal_family_size": 3,
        "contrasts": contrasts(),
        "bootstrap_draws": 10000,
        "bootstrap_seed": 20260921,
        "focal_family_confidence": 1 - 0.05 / 3,
        "descriptive_confidence": 0.95,
        "validity": "One shared mask per endpoint across all 24 factorial condition/repeat settings; bridge contrast uses a separate shared six-setting edg/reference_rag mask. No reference-only failure can remove an item from the factorial analysis. Report own-valid scores, all exclusions, valid-vote counts and full-roster failure bounds for each arm.",
        "estimator": "Fixed full-roster labels/support>=10; strict gate for fine heads, legal independent of hate; None unannotated, [] observed empty. Mean of three run-level macro-F1 values, sample SD/range and every repeat; never pooled votes or pooled repeat items.",
        "bootstrap": "Paired item resampling within endpoint, all conditions/repeats use the same item draws within their declared mask. Keep inherited labels even when a bootstrap draw has zero support. Resample items, not repeats; interval is conditional on the three realised decoder repeats and fixed retrieved contexts. Independent frequency-weighted recomputation required.",
        "multiple_comparisons": "Three leave-one-component-out contrasts on EN467 form the focal family: 98.333...% Bonferroni intervals plus pointwise95%. The same contrasts on six other endpoints, seven simple contrasts and seven factorial effects/interactions are descriptive95%, no additional confirmatory claims.",
        "diagnostics": [
            "FP/FN for hate; per-label F1, predicted cardinality, exact match and Hamming where applicable; false-negative gate counts on fine heads.",
            "Delivered TRAIN annotation/source/language composition, record counts and prompt lengths; counts do not prove usefulness or adherence.",
            "Schema repairs, invalid votes/nulls, observed tokens, retries, response identity and saved latency. Retain unknown remote work separately.",
        ],
        "no_partial_scoring": True,
        "limits": "Predeclared extension on previously inspected DEV, not untouched confirmation. Removal also changes length and position; effects concern supplying these retrieved blocks under this common instruction, not semantic content alone, optimal allocation, universal usefulness or an oracle. Bridge wording includes placement changes. A bridge interval crossing zero is not proof of equivalence. No automatic prompt/config promotion.",
    }


def sources():
    acceptance = a.read(SOURCE / "retrieval_acceptance_v1.json")
    a.require(
        acceptance["status"] == "accepted_first_pass_with_replay_qualification",
        "Retrieval not accepted",
    )
    a.require(
        a.digest(SOURCE / "plan.json") == acceptance["plan_sha256"] == SOURCE_PLAN_SHA,
        "Source plan differs",
    )
    path = ROOT / acceptance["prompts"]
    a.require(
        a.digest(path) == acceptance["prompts_sha256"] == SOURCE_PROMPTS_SHA,
        "First-pass source messages differ",
    )
    a.require(
        a.digest(ROOT / acceptance["evidence"])
        == acceptance["evidence_sha256"]
        == SOURCE_AUDIT_SHA,
        "Source audit differs",
    )
    complete = a.read(SOURCE / "completion_v1/complete.json")
    a.require(
        a.digest(ROOT / complete["evidence"])
        == complete["evidence_sha256"]
        == SOURCE_COMPLETION_SHA,
        "Previous study completion differs",
    )
    source_plan = a.read(SOURCE / "plan.json")
    prior_api = a.read(SOURCE / "api_execution_v1/plan.json")
    paths = [
        path,
        SOURCE / "plan.json",
        SOURCE / "retrieval_acceptance_v1.json",
        SOURCE / "completion_v1/complete.json",
        ROOT / acceptance["evidence"],
        ROOT / complete["evidence"],
        SOURCE / "api_execution_v1/plan.json",
        CORE / "dev_inputs.jsonl",
        CORE / "kb/full/records.jsonl",
        CORE / "api_execution_v1/scoring_plan.json",
    ]
    for name in [
        "experiments/astra/full_kb_core_20260915_v1/dev_inputs.jsonl",
        "experiments/astra/full_kb_core_20260915_v1/kb/full/records.jsonl",
        "analysis/common/prompt.py",
        "src/hsrag/prompt.py",
        "prompts/classify_v1.txt",
        "config/taxonomy.yaml",
    ]:
        a.require(
            a.digest(ROOT / name) == source_plan["input_hashes"][name],
            "Audited source changed: " + name,
        )
    for subset in ENDPOINT_ORDER:
        gold = ROOT / f"data/processed/{subset}.parquet"
        a.require(a.digest(gold) == prior_api["input_hashes"][rel(gold)], "Gold source changed")
        paths.append(gold)
    raw = core.rows(path)
    a.require(len(raw) == 3885, "Source strategy entry count differs")
    selected = {row["key"]: row for row in raw if row["condition"] == "rag_hybrid"}
    a.require(len(selected) == 1295, "Missing/duplicate hybrid source")
    dev = core.rows(CORE / "dev_inputs.jsonl")
    a.require(
        len(dev) == len({r["key"] for r in dev}) == 1295
        and set(selected) == {r["key"] for r in dev},
        "DEV identity differs",
    )
    counts = Counter()
    for row in dev:
        original = selected[row["key"]]
        for field in ("key", "text", "lang", "item_id", "scopes"):
            a.require(original[field] == row[field], "Source input differs: " + field)
        a.require(
            original["plan_sha256"] == SOURCE_PLAN_SHA
            and original["strategy"] == "hybrid"
            and (original["pool"] == "full"),
            "Source attribution differs",
        )
        a.require(
            a.signature(original["messages"]) == original["messages_sha256"]
            and a.signature(original["context"]) == original["context_sha256"],
            "Source signature differs",
        )
        a.require(
            prompt.hit_ids(original["context"]) == original["retrieved"],
            "Source record identities differ",
        )
        a.require(len(row["scopes"]) == 1, "Multiple/unknown input scopes")
        counts[row["scopes"][0]["subset"]] += 1
    a.require(dict(counts) == SCOPE_COUNTS, "Endpoint roster differs")
    return (source_plan, dev, selected, paths)


def entry(source, condition, rendered):
    messages, context = render(source, condition, rendered)
    a.body(messages)
    return {
        **{key: source[key] for key in ("key", "item_id", "lang", "text", "scopes")},
        "condition": condition,
        "strategy": "hybrid",
        "pool": "full",
        "factor_bits_e_d_g": list(ARMS[condition]) if condition in ARMS else [1, 1, 1],
        "prompt_family": (
            "astra_components_common_v1" if condition in ARMS else "preserved_reference_rag"
        ),
        "source_entry_sha256": a.signature(source),
        "source_messages_sha256": source["messages_sha256"],
        "context": context,
        "context_sha256": a.signature(context),
        "retrieved": prompt.hit_ids(context),
        "messages": messages,
        "messages_sha256": a.signature(messages),
    }


def write_prompts(entries):
    path = a.storage_path(PROMPTS)
    a.require(not path.exists(), "Preserve component prompts")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial_" + str(time.time_ns()))
    with tmp.open("x", encoding="utf8", newline="\n") as handle:
        for row in entries:
            handle.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )
        handle.flush()
        os.fsync(handle.fileno())
    os.link(tmp, path)
    tmp.unlink()


def prepare():
    a.require(
        not PLAN.exists() and (not PROMPTS.exists()), "Study already prepared; preserve artifacts"
    )
    test_path = BASE / "development/preparation_tests_passed.json"
    tests = a.read(test_path)
    a.require(
        tests["passed"] is True and tests["environment"] == core.environment(),
        "Passing preparation tests required",
    )
    for name, digest in tests["source_hashes"].items():
        a.require(a.digest(ROOT / name) == digest, "Preparation source changed after tests")
    old, dev, originals, paths = sources()
    save(BASE / "analysis_plan.json", analysis_plan(old))
    paths += [BASE / "PROTOCOL.md", test_path]
    hashes = {rel(path): a.digest(path) for path in paths}
    protocol = {
        "created_utc": a.utc(),
        "status": "protocol_frozen_before_new_model_outputs",
        "question": "What do naturally retrieved examples, definitions and guidelines contribute under matched instruction wording and fixed record selection?",
        "source_hashes": {name: a.digest(ROOT / name) for name in SOURCE_FILES},
        "input_hashes": hashes,
        "analysis_sha256": a.digest(BASE / "analysis_plan.json"),
        "environment": core.environment(),
        "arms_e_d_g": {k: list(v) for k, v in ARMS.items()},
        "bridge": BRIDGE,
        "common_instruction": COMMON_NOTE,
        "scope_counts": SCOPE_COUNTS,
        "budget": {
            "distinct_inputs": 1295,
            "conditions": 9,
            "factorial_conditions": 8,
            "prompt_payloads": 11655,
            "decoder_repeats": 3,
            "votes_per_prediction": 3,
            "fresh_predictions": 34965,
            "ordinary_responses": 104895,
            "max_logical_calls": 314685,
            "max_physical_POSTs": 1573425,
            "stages": 13,
            "maximum_predictions_per_stage": 2700,
        },
        "retrieval": "Use accepted first-pass hybrid contexts without reopening an index or retrieving again. Every retained bucket has identical records, metadata and order across arms/repeats. Remove only complete buckets. Budgets 2D/2G/5E or zero. No feedback, allocation change, label-aware selection, resampling or oracle.",
        "source_replay_qualification": "Original strategy replay changed the hybrid prompt for en/implicit_hate-4207. Retain accepted first-pass context for every arm; all focal EN467 contexts were replay-exact. No retry-to-match or replay substitution. Inference conditions on fixed contexts.",
        "reuse": "Reuse only audited source contexts, masks/labels, parser/voter/transport. All nine conditions receive fresh contemporaneously interleaved decoder repeats. Historical predictions are not extra repeats or controls.",
        "prompt_scope": "System, taxonomy/schema, target text and corrected TRAIN-label rendering unchanged. All eight factorial arms share COMMON_NOTE in the same first user-message position; the old examples caution is absent from them. Keep exact original full-RAG messages as ninth fresh bridge. This is a separate prompt-version intervention, not a silent change to main RAG.",
        "preservation": "New study files only. Old results/indexes/models/source, original logs and accepted strategy evidence untouched. No TEST/benchmarks, Priority3/adaptation or LaTeX. No API or credentials during offline preparation.",
        "next": "Render every payload, independent matched-context audit; separately freeze/test one-stage API runner before user execution.",
    }
    save(PLAN, protocol)

    def entries():
        for row in dev:
            original = originals[row["key"]]
            rendered = blocks(original)
            for condition in CONDITIONS:
                yield entry(original, condition, rendered)

    write_prompts(entries())
    for name, digest in hashes.items():
        a.require(a.digest(ROOT / name) == digest, "Input changed during rendering")
    save(
        BASE / "render_complete.json",
        {
            "created_utc": a.utc(),
            "status": "rendered_offline_review_pending",
            "plan_sha256": a.digest(PLAN),
            "prompts_sha256": a.digest(PROMPTS),
            "prompt_payloads": 11655,
            "actual_API_calls": 0,
            "retrieval_calls": 0,
        },
    )
    print(
        json.dumps(
            {
                "status": "rendered_offline_review_pending",
                "plan_sha256": a.digest(PLAN),
                "prompts_sha256": a.digest(PROMPTS),
                "budget": protocol["budget"],
            },
            indent=2,
        )
    )


def validate():
    plan = a.read(PLAN)
    a.require(plan["environment"] == core.environment(), "Preparation environment changed")
    for group in ("source_hashes", "input_hashes"):
        for name, digest in plan[group].items():
            a.require(a.digest(ROOT / name) == digest, "Frozen preparation changed: " + name)
    a.require(
        a.digest(BASE / "analysis_plan.json") == plan["analysis_sha256"], "Analysis plan changed"
    )
    receipt = a.read(BASE / "render_complete.json")
    a.require(
        receipt["plan_sha256"] == a.digest(PLAN) and receipt["prompts_sha256"] == a.digest(PROMPTS),
        "Rendered prompts changed",
    )
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "validate"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ), patch.object(core, "live_client", forbidden):
        if args.mode == "prepare":
            prepare()
        else:
            validate()
            print("Frozen component preparation verified; zero API calls.")

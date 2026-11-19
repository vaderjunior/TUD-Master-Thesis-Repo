import os

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_name] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import socket
import sys
import time
from unittest.mock import patch
import numpy as np
import pandas as pd
from tokenizers import Tokenizer
from analysis.feedback.explanations import calibration as c
from analysis.common import prompt as renderer

ROOT = c.ROOT
BASE = ROOT / "experiments/astra/sq3_full_kb_explanations_20261001_v1"
OLD = ROOT / "experiments/astra/sq3_full_kb_feedback_20260928_v1"
CORE = c.CORE
CARDS = ROOT / "experiments/astra/sq3_explanation_authoring_20261001_v1"
TOK = ROOT / "experiments/astra/qwen_tokenizer_assets_20261001_v1"
POLICIES = [
    "plain_always",
    "specific_always",
    "generic_always",
    "plain_selective",
    "specific_selective",
]
SEEDS = [202609281, 202609282, 202609283]
EXPLANATION_PREFIX = "  Explanation of this example's labels: "


def sig(obj):
    return c.digest(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )


def verify_prior():
    c.require(
        c.sha(c.BASE / "policy.json")
        == "b4ec3967d74e07139864ef42271cce38a46560a92ac8c4f015c685833021c7fb",
        "Changed cutoff",
    )
    for base, receipt in ((c.BASE, "prepared.json"), (CARDS, "cards_prepared.json")):
        done = c.read(base / "complete.json")
        key = "prepared_sha256" if base == c.BASE else "cards_prepared_sha256"
        c.require(c.sha(base / receipt) == done[key], "Prerequisite completion differs")
        for name, expected in c.read(base / receipt)["outputs"].items():
            c.require(
                c.sha(base / name) == expected, "Prerequisite output changed: " + name
            )
    token = c.read(TOK / "complete.json")
    for name, row in token["files"].items():
        c.require(c.sha(TOK / name) == row["sha256"], "Tokenizer changed")
    c.require(
        token["revision"] == "0cfaf48183f594c314753d30a4c4974bc75f3ccb",
        "Tokenizer revision differs",
    )
    c.require(
        c.sha(CORE / "retrieval_v1/full/prompts.jsonl")
        == c.read(CORE / "retrieval_v1/full/complete.json")["prompt_sha256"],
        "Base prompt origin mismatch",
    )
    return c.verify_active()


def render(base, source, explanation, deliver):

    context = deepcopy(base["context"])
    if not deliver:
        c.require(not explanation, "Cannot render explanation for a skipped source")
        return (context, deepcopy(base["messages"]))
    c.require(
        len(context["definitions"]) == len(context["guidelines"]) == 2,
        "Wrong base abstract budget",
    )
    c.require(
        len(context["example_groups"]) == 1
        and len(context["example_groups"][0][1]) == 5,
        "Wrong base examples",
    )
    c.require(
        source["meta"]["gate"] is True and source["meta"]["hate_types"],
        "Wrong added source",
    )
    c.require(
        all(
            (
                source["meta"].get(k) is None
                for k in ("target_groups", "severity", "legal")
            )
        ),
        "Imputed labels",
    )
    record = {
        "id": source["id"],
        "text": source["text"],
        "meta": deepcopy(source["meta"]),
        "via": "rrf",
    }
    context["example_groups"].append(["Additional labelled examples", [record]])
    messages = renderer.render_messages(
        base["text"], base["lang"], context, "corrected"
    )
    if explanation:
        c.require(
            explanation == " ".join(explanation.split()),
            "Explanations must be a single normalized line",
        )
        suffix = f"\n\n## Text to classify (language: {base['lang']})\n{base['text']}"
        c.require(messages[-1]["text"].endswith(suffix), "Input suffix changed")
        prefix = messages[-1]["text"][: -len(suffix)]
        c.require(
            prefix.endswith("  labels: " + renderer.corrected_gold(source["meta"])),
            "Explanation would attach to wrong labels",
        )
        messages[-1]["text"] = prefix + "\n" + EXPLANATION_PREFIX + explanation + suffix
    c.require(messages[0] == base["messages"][0], "System prompt changed")
    return (context, messages)


def local_chat(messages):
    c.require(
        [m["role"] for m in messages] == ["system", "user"],
        "Only original text-only pair supported",
    )
    return (
        "".join(
            (
                "<|im_start|>" + m["role"] + "\n" + m["text"] + "<|im_end|>\n"
                for m in messages
            )
        )
        + "<|im_start|>assistant\n"
    )


def freeze():
    c.require(
        not (BASE / "preparation_plan.json").exists(),
        "Preserve frozen preparation plan",
    )
    c.require(
        c.read(BASE / "tests_passed_v2.json")["successful"],
        "Intervention tests required",
    )
    verify_prior()
    paths = [
        BASE / "STUDY_PROTOCOL.md",
        BASE / "tests_passed_v2.json",
        CORE / "plan.json",
        CORE / "dev_inputs.jsonl",
        CORE / "kb/full/records.jsonl",
        CORE / "retrieval_v1/plan.json",
        CORE / "retrieval_v1/full_vectors.npy",
        CORE / "retrieval_v1/query_vectors.npy",
        CORE / "retrieval_v1/full/prompts.jsonl",
        CORE / "retrieval_v1/full/complete.json",
        CORE / "retrieval_v1/record_vector_sources.jsonl",
        CORE / "retrieval_v1/query_vector_sources.jsonl",
        ROOT / "data/processed/en_dev.parquet",
        OLD / "scientific_plan_v1.json",
        OLD / "feasibility_v1/roster.jsonl",
        OLD / "update_payloads_v1/pools.json",
        OLD / "update_payloads_v1/added_records.jsonl",
        OLD / "update_payloads_v1/prompts.jsonl",
        OLD / "update_payloads_v1/complete.json",
        OLD / "baseline_review_v1/selection.json",
        c.BASE / "plan.json",
        c.BASE / "retained_train.jsonl",
        c.BASE / "policy.json",
        c.BASE / "complete.json",
        CARDS / "explanation_cards.jsonl",
        CARDS / "cards_prepared.json",
        CARDS / "complete.json",
        TOK / "complete.json",
        c.ACTIVE,
        ROOT / "src/hsrag/prompt.py",
        ROOT / "prompts/classify_v1.txt",
        ROOT / "config/taxonomy.yaml",
        ROOT / "experiments/astra/english_kb_audit_20260914_v1/plan.json",
    ]
    paths += [TOK / name for name in c.read(TOK / "complete.json")["files"]]
    paths += [
        ROOT / name
        for name in [
            "analysis/feedback/explanations/study.py",
            "analysis/feedback/explanations/validate_study.py",
            "tests/feedback/explanations/test_study.py",
            "analysis/feedback/explanations/calibration.py",
            "analysis/feedback/explanations/validate_calibration.py",
            "analysis/feedback/source/retrieval.py",
            "analysis/common/prompt.py",
        ]
    ]
    hashes = {c.rel(p): c.sha(p) for p in paths}
    audit = c.read(ROOT / "experiments/astra/english_kb_audit_20260914_v1/plan.json")
    c.require(
        hashes["data/processed/en_dev.parquet"]
        == audit["input_hashes"]["data/processed/en_dev.parquet"],
        "DEV provenance changed",
    )
    c.write(
        BASE / "preparation_plan.json",
        {
            "status": "frozen_before_overlap_and_delivery_results",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": hashes,
            "primary_transfer_count": 350,
            "initial_benign_count": 208,
            "cutoff": 0.64,
            "near_threshold": 0.95,
            "borderline_margin": 0.0005,
            "policies": POLICIES,
            "pool_seeds": SEEDS,
            "input_token_guard": 8192,
            "network_calls_in_preparation": 0,
            "active_bindings_verified": 79,
        },
    )
    print(
        json.dumps(
            {"status": "preparation_rules_frozen", "input_bindings": len(hashes)}
        )
    )


def prepare():
    started = time.perf_counter()
    plan = c.read(BASE / "preparation_plan.json")
    c.require(
        not (BASE / "scope_inventory.json").exists(),
        "Preserve earlier preparation attempt",
    )
    for name, expected in plan["inputs"].items():
        c.require(c.sha(ROOT / name) == expected, "Frozen input changed: " + name)
    verify_prior()
    core = c.lines(CORE / "dev_inputs.jsonl")
    qmap = {r["item_id"]: i for i, r in enumerate(core)}
    records = c.lines(CORE / "kb/full/records.jsonl")
    recmap = {r["id"]: r for r in records}
    full_prompts = c.lines(CORE / "retrieval_v1/full/prompts.jsonl")
    bases = {r["item_id"]: r for r in full_prompts}
    qv = np.load(
        CORE / "retrieval_v1/query_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    fv = np.load(
        CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    c.require(
        qv.shape == (1295, 1024) and fv.shape == (71131, 1024), "Cache shape differs"
    )
    d = pd.read_parquet(ROOT / "data/processed/en_dev.parquet")
    dmap = {r["id"]: r for r in d.to_dict("records")}
    c.require(len(dmap) == len(d), "DEV IDs not unique")
    oldscience = c.read(OLD / "scientific_plan_v1.json")
    transfer = set(oldscience["rosters"]["transfer"])
    original = {r["id"]: r for r in c.lines(OLD / "feasibility_v1/roster.jsonl")}
    benign = {
        r["item_id"]
        for r in core
        if r["lang"] == "en" and dmap[r["item_id"]]["gate"] is False
    }
    c.require(
        len(transfer) == 350 and len(benign) == 208 and (not transfer & benign),
        "Initial roster changed",
    )
    ids = sorted(transfer | benign)
    all_roster = []
    for iid in ids:
        raw = dmap[iid]
        base = bases[iid]
        cr = core[qmap[iid]]
        c.require(
            raw["text"] == cr["text"] == base["text"] and base["key"] == cr["key"],
            "Input identity differs",
        )
        is_transfer = iid in transfer
        gold_types = list(raw["hate_types"]) if raw["hate_types"] is not None else None
        if is_transfer:
            c.require(
                raw["gate"] is True
                and set(gold_types) == set(original[iid]["gold"]["hate_type"]),
                "Transfer gold changed",
            )
        all_roster.append(
            {
                "id": iid,
                "key": cr["key"],
                "source": raw["source"],
                "lang": "en",
                "text": raw["text"],
                "text_sha256": c.digest(raw["text"]),
                "vector_row": qmap[iid],
                "cohort": "transfer_hateful" if is_transfer else "benign_extension",
                "gold": {"hate": bool(raw["gate"]), "hate_type": gold_types},
                "original_transfer_role": (
                    original[iid]["role"] if is_transfer else None
                ),
            }
        )
    c.write(
        BASE / "scope_inventory.json",
        {
            "initial_benign": len(benign),
            "transfer": len(transfer),
            "benign_source_counts": dict(
                Counter(
                    (
                        r["source"]
                        for r in all_roster
                        if r["cohort"] == "benign_extension"
                    )
                )
            ),
            "selection_uses_model_predictions": False,
            "initial_ids": ids,
        },
    )
    x = c.unit(qv[[qmap[iid] for iid in ids]])
    xn = [c.normalize(r["text"]) for r in all_roster]
    within = np.clip(x @ x.T, -1, 1)
    edges = []
    evidence = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            sim = float(within[i, j])
            exact = xn[i] == xn[j]
            edge = exact or sim >= 0.95
            if edge or sim >= 0.9495:
                evidence.append(
                    {
                        "kind": "evaluation",
                        "left": ids[i],
                        "right": ids[j],
                        "cosine": sim,
                        "exact": exact,
                        "edge": edge,
                        "borderline": abs(sim - 0.95) <= 0.0005,
                    }
                )
            if edge:
                edges.append((i, j))
    pools = sorted(
        [
            p
            for p in c.read(OLD / "update_payloads_v1/pools.json")["pools"]
            if p["arm"] == "error" and p["size"] == 40
        ],
        key=lambda p: p["seed"],
    )
    c.require([p["seed"] for p in pools] == SEEDS, "Pool selection changed")
    added = {
        r["provenance"]["source_id"]: r
        for r in c.lines(OLD / "update_payloads_v1/added_records.jsonl")
    }
    source_ids = sorted({iid for p in pools for iid in p["ids"]})
    cal = c.lines(c.BASE / "retained_train.jsonl")
    train = [(i, r) for i, r in enumerate(records) if r["kind"] == "example"]
    c.require(
        len(source_ids) == 58 and len(cal) == 2041 and (len(train) == 71091),
        "Protected population changed",
    )
    touched = defaultdict(list)
    nearest = {iid: {} for iid in ids}
    protected = [
        (
            "correction",
            source_ids,
            [added[i]["text"] for i in source_ids],
            qv[[qmap[i] for i in source_ids]],
        ),
        (
            "calibration",
            [r["id"] for r in cal],
            [r["text"] for r in cal],
            fv[[r["record_row"] for r in cal]],
        ),
    ]
    for start in range(0, len(train), 2048):
        chunk = train[start : start + 2048]
        protected.append(
            (
                "base_train",
                [r["id"] for _, r in chunk],
                [r["text"] for _, r in chunk],
                fv[[i for i, _ in chunk]],
            )
        )
    for kind, protected_ids, texts, vectors in protected:
        vec = c.unit(vectors)
        sims = np.clip(x @ vec.T, -1, 1)
        norm_index = defaultdict(list)
        for j, t in enumerate(texts):
            norm_index[c.normalize(t)].append(j)
        for i, iid in enumerate(ids):
            j = int(np.argmax(sims[i]))
            best = {"id": protected_ids[j], "cosine": float(sims[i, j])}
            previous = nearest[iid].get(kind)
            if previous is None or (-best["cosine"], best["id"]) < (
                -previous["cosine"],
                previous["id"],
            ):
                nearest[iid][kind] = best
            candidates = set(np.flatnonzero(sims[i] >= 0.9495).tolist()) | set(
                norm_index.get(xn[i], [])
            )
            for j in sorted(candidates):
                sim = float(sims[i, j])
                exact = xn[i] == c.normalize(texts[j])
                edge = exact or sim >= 0.95
                evidence.append(
                    {
                        "kind": kind,
                        "left": iid,
                        "right": protected_ids[j],
                        "cosine": sim,
                        "exact": exact,
                        "edge": edge,
                        "borderline": abs(sim - 0.95) <= 0.0005,
                    }
                )
                if edge:
                    touched[i].append(kind + ":" + protected_ids[j])
    comps = c.components(
        len(ids), edges, touched, ["evaluation"] * len(ids), [(iid,) for iid in ids]
    )
    excluded = {}
    clusters = {}
    cluster_rows = []
    for group in comps:
        members = [ids[i] for i in group["members"]]
        cluster_id = "sq3exp-" + c.digest("|".join(members))[:16]
        dirty = bool(group["protected_keys"])
        c.require(
            not (dirty and set(members) & transfer),
            "Original transfer overlap requires explicit review: " + ",".join(members),
        )
        for iid in members:
            clusters[iid] = cluster_id
            if dirty:
                excluded[iid] = group["protected_keys"]
        cluster_rows.append(
            {
                "cluster": cluster_id,
                "members": members,
                "protected_keys": group["protected_keys"],
                "mixed_hate_annotations": len(
                    {all_roster[i]["gold"]["hate"] for i in group["members"]}
                )
                > 1,
            }
        )
    roster = [
        {**r, "cluster": clusters[r["id"]]}
        for r in all_roster
        if r["id"] not in excluded
    ]
    c.require(
        len([r for r in roster if r["cohort"] == "transfer_hateful"]) == 350,
        "Main cohort changed",
    )
    c.require(
        any((r["cohort"] == "benign_extension" for r in roster)), "Benign cohort empty"
    )
    c.jsonlines(BASE / "overlap_evidence.jsonl", evidence)
    c.write(BASE / "overlap_nearest.json", nearest)
    c.write(BASE / "clusters.json", cluster_rows)
    c.write(BASE / "excluded_benign.json", excluded)
    c.jsonlines(BASE / "roster.jsonl", roster)
    for r in roster:
        base = bases[r["id"]]
        ctx = base["context"]
        c.require(
            len(ctx["definitions"]) == len(ctx["guidelines"]) == 2
            and len(ctx["example_groups"]) == 1
            and (len(ctx["example_groups"][0][1]) == 5),
            "Base budget differs",
        )
        for hit in (
            ctx["definitions"]
            + ctx["guidelines"]
            + [h for _, rr in ctx["example_groups"] for h in rr]
        ):
            rec = recmap[hit["id"]]
            c.require(hit["text"] == rec["text"], "Base source text changed")
            meta = {
                **{
                    k: rec.get(k)
                    for k in ("kind", "lang", "source", "dimension", "label")
                },
                **rec["meta"],
            }
            c.require(
                all((hit["meta"].get(k) == v for k, v in meta.items()))
                and set(hit["meta"]) <= set(meta) | {"_ranks"},
                "Base annotations changed",
            )
        rendered = renderer.render_messages(base["text"], "en", ctx, "corrected")
        c.require(
            rendered == base["messages"]
            and sig(rendered) == base["messages_sha256"]
            and (sig(ctx) == base["context_sha256"]),
            "Base prompt changed",
        )
    cards = {r["source_id"]: r for r in c.lines(CARDS / "explanation_cards.jsonl")}
    c.require(set(cards) == set(source_ids), "Authoring source set differs")
    buckets = {
        p["seed"]: c.LocalBucket(
            [added[i] for i in p["ids"]], qv[[qmap[i] for i in p["ids"]]]
        )
        for p in pools
    }
    choices = []
    for r in roster:
        for seed in SEEDS:
            source, cos, trace = buckets[seed].retrieve(r["text"], qv[r["vector_row"]])
            sid = source["provenance"]["source_id"]
            card = cards[sid]
            c.require(
                c.digest(source["text"]) == card["source_text_sha256"]
                and source["meta"]["hate_types"] == card["hate_types"],
                "Card/source mismatch",
            )
            choices.append(
                {
                    "query_id": r["id"],
                    "cohort": r["cohort"],
                    "pool_seed": seed,
                    "source_id": sid,
                    "cosine": cos,
                    "apply": cos >= 0.64,
                    "explanation_available": card["explanation_available"],
                    "retrieval": trace,
                }
            )
    c.jsonlines(BASE / "choices.jsonl", choices)
    sentinels = c.read(OLD / "baseline_review_v1/selection.json")["sentinel_ids"]
    scientific = {
        "status": "science_and_roster_frozen_execution_not_ready",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "preparation_plan_sha256": c.sha(BASE / "preparation_plan.json"),
        "protocol_sha256": c.sha(BASE / "STUDY_PROTOCOL.md"),
        "rosters": {
            "transfer": sorted(transfer),
            "benign": sorted(
                (r["id"] for r in roster if r["cohort"] == "benign_extension")
            ),
            "original_holdout": oldscience["rosters"]["holdout"],
            "historical_sentinels": sentinels,
        },
        "roster_sha256": c.sha(BASE / "roster.jsonl"),
        "cluster_sha256": c.sha(BASE / "clusters.json"),
        "choices_sha256": c.sha(BASE / "choices.jsonl"),
        "policy_sha256": c.sha(c.BASE / "policy.json"),
        "conditions": {
            "per_pool": POLICIES,
            "pool_seeds": SEEDS,
            "reference": "reference",
            "decoder_repeats": 3,
            "votes": 3,
        },
        "primary": {
            "explanation": "specific_always_minus_plain_always",
            "application": "plain_selective_minus_plain_always",
            "endpoint": "seven_type_macro_f1_original_350",
            "valid_mask": "per_contrast_all_18_outputs",
            "family_interval_level": 0.9875,
            "draws": 10000,
            "bootstrap_seeds": [20261001, 20261002],
        },
        "benign_role": "supporting_false_positive_diagnostic_mixed_source_exposed_DEV",
        "budget": {
            "inputs": len(roster),
            "prompt_identities": len(roster) * 16,
            "predictions": len(roster) * 48,
            "ordinary_calls": len(roster) * 144,
        },
        "actual_provider_calls": 0,
        "no_outcome_based_selection": True,
    }
    c.write(BASE / "scientific_plan_v1.json", scientific)
    print(
        json.dumps(
            {
                "status": "scope_and_choices_frozen",
                "retained": len(roster),
                "excluded_benign": len(excluded),
                "overlap_evidence_rows": len(evidence),
                "ordinary_calls": scientific["budget"]["ordinary_calls"],
            }
        ),
        flush=True,
    )
    tokenizer = Tokenizer.from_file(str(TOK / "tokenizer.json"))
    tokenizer.no_truncation()
    tokenizer.no_padding()
    original_plain = {}
    with (OLD / "update_payloads_v1/prompts.jsonl").open(encoding="utf8") as stream:
        for line in stream:
            old = json.loads(line)
            if (
                old["item_id"] in transfer
                and old["condition"].startswith("error_")
                and old["condition"].endswith("_b04")
            ):
                seed = int(old["condition"].split("_")[1][1:])
                original_plain[old["item_id"], seed] = old
    c.require(len(original_plain) == 1050, "Original natural matrix missing")
    rmap = {r["id"]: r for r in roster}
    choices_by_id = defaultdict(list)
    for ch in choices:
        choices_by_id[ch["query_id"]].append(ch)
    payloads = []
    inventory = []
    natural_matches = 0
    skip_matches = 0
    fallback_matches = 0
    token_cache = {}

    def save_payload(r, condition, policy, seed, ch, source, explanation, deliver):
        base = bases[r["id"]]
        context, messages = render(base, source, explanation, deliver)
        message_hash = sig(messages)
        if message_hash not in token_cache:
            token_cache[message_hash] = len(
                tokenizer.encode(local_chat(messages), add_special_tokens=False).ids
            )
        tokens = token_cache[message_hash]
        c.require(tokens <= 8192, "Input exceeds declared local token guard")
        row = {
            "key": r["key"],
            "item_id": r["id"],
            "cohort": r["cohort"],
            "condition": condition,
            "policy": policy,
            "pool_seed": seed,
            "strategy": "hybrid",
            "lang": "en",
            "text": r["text"],
            "natural_source_id": ch["source_id"] if ch else None,
            "source_id": ch["source_id"] if deliver else None,
            "apply": bool(deliver),
            "explanation_kind": (
                "specific"
                if explanation and policy.startswith("specific")
                else "generic" if explanation else None
            ),
            "explanation": explanation,
            "context": context,
            "messages": messages,
            "context_sha256": sig(context),
            "messages_sha256": message_hash,
            "reference_input_tokens": tokens,
            "scientific_plan_sha256": c.sha(BASE / "scientific_plan_v1.json"),
        }
        payloads.append(row)
        inventory.append(
            {
                k: row[k]
                for k in (
                    "key",
                    "condition",
                    "policy",
                    "pool_seed",
                    "cohort",
                    "source_id",
                    "apply",
                    "explanation_kind",
                    "messages_sha256",
                    "context_sha256",
                    "reference_input_tokens",
                )
            }
            | {"entry_sha256": sig(row)}
        )
        return row

    for r in roster:
        reference = save_payload(
            r, "reference", "reference", None, None, None, "", False
        )
        for ch in choices_by_id[r["id"]]:
            seed = ch["pool_seed"]
            source = added[ch["source_id"]]
            card = cards[ch["source_id"]]
            rendered = {}
            for policy in POLICIES:
                deliver = not policy.endswith("selective") or ch["apply"]
                explanation = ""
                if deliver and policy.startswith("specific"):
                    explanation = card["specific_explanation"]
                elif deliver and policy.startswith("generic"):
                    explanation = card["generic_explanation"]
                row = save_payload(
                    r,
                    f"{policy}_s{seed}",
                    policy,
                    seed,
                    ch,
                    source,
                    explanation,
                    deliver,
                )
                rendered[policy] = row
                if not deliver:
                    c.require(
                        row["messages"] == reference["messages"]
                        and row["context"] == reference["context"],
                        "Skip is not exact reference",
                    )
                    skip_matches += 1
                if (
                    deliver
                    and policy.startswith(("specific", "generic"))
                    and (not card["explanation_available"])
                ):
                    c.require(
                        row["messages"] == rendered["plain_always"]["messages"],
                        "Fallback not plain",
                    )
                    fallback_matches += 1
            if r["id"] in transfer:
                old = original_plain[r["id"], seed]
                plain = rendered["plain_always"]
                c.require(
                    plain["messages"] == old["messages"]
                    and plain["context"] == old["context"],
                    "Natural transfer prompt changed",
                )
                natural_matches += 1
    payloads.sort(key=lambda r: (r["key"], r["condition"]))
    inventory.sort(key=lambda r: (r["key"], r["condition"]))
    c.require(
        len(payloads) == len(roster) * 16
        and len({(r["key"], r["condition"]) for r in payloads}) == len(payloads),
        "Payload matrix differs",
    )
    c.jsonlines(BASE / "prompts.jsonl", payloads)
    c.jsonlines(BASE / "prompt_inventory.jsonl", inventory)
    coverage = {}
    for cohort in ("transfer_hateful", "benign_extension"):
        coverage[cohort] = {}
        for seed in [None] + SEEDS:
            rows = [
                r
                for r in choices
                if r["cohort"] == cohort and (seed is None or r["pool_seed"] == seed)
            ]
            coverage[cohort]["pooled" if seed is None else str(seed)] = {
                "input_pool_pairs": len(rows),
                "unique_inputs": len({r["query_id"] for r in rows}),
                "applied": sum((r["apply"] for r in rows)),
                "explanation_available": sum(
                    (r["explanation_available"] for r in rows)
                ),
                "specific_selective_delivered": sum(
                    (r["apply"] and r["explanation_available"] for r in rows)
                ),
            }
    token_stats = {}
    for cohort in ("transfer_hateful", "benign_extension"):
        token_stats[cohort] = {}
        for policy in ["reference"] + POLICIES:
            values = [
                p["reference_input_tokens"]
                for p in payloads
                if p["policy"] == policy and p["cohort"] == cohort
            ]
            token_stats[cohort][policy] = {
                "prompts": len(values),
                "min": min(values),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "max": max(values),
            }
    explanation_token_lengths = []
    for sid, card in sorted(cards.items()):
        if not card["explanation_available"]:
            continue
        sp = len(
            tokenizer.encode(card["specific_explanation"], add_special_tokens=False).ids
        )
        ge = len(
            tokenizer.encode(card["generic_explanation"], add_special_tokens=False).ids
        )
        explanation_token_lengths.append(
            {"source_id": sid, "specific": sp, "generic": ge, "difference": sp - ge}
        )
    unique_by_input = Counter(
        (
            len({p["messages_sha256"] for p in payloads if p["item_id"] == r["id"]})
            for r in roster
        )
    )
    summary = {
        "status": "prepared_independent_check_pending",
        "inputs": len(roster),
        "benign": len(roster) - 350,
        "excluded_benign": len(excluded),
        "overlap_evidence_rows": len(evidence),
        "overlap_borderline_rows": sum((r["borderline"] for r in evidence)),
        "evaluation_clusters": len(comps),
        "retained_clusters": len({r["cluster"] for r in roster}),
        "mixed_annotation_clusters": sum(
            (r["mixed_hate_annotations"] for r in cluster_rows)
        ),
        "coverage": coverage,
        "token_statistics": token_stats,
        "source_explanation_token_lengths": explanation_token_lengths,
        "prompt_identities": len(payloads),
        "unique_message_hashes": len(token_cache),
        "unique_prompt_count_per_input": dict(sorted(unique_by_input.items())),
        "reference_exact_matches": len(roster),
        "natural_transfer_exact_matches": natural_matches,
        "skip_exact_matches": skip_matches,
        "plain_fallback_exact_matches": fallback_matches,
        "ordinary_calls_planned": len(roster) * 144,
        "new_provider_calls": 0,
        "cuda_jobs": 0,
        "active_bindings_verified": verify_prior(),
        "seconds": time.perf_counter() - started,
        "server_tokenizer_equivalence": "Not independently verified; counts are for pinned local official reference.",
    }
    c.write(BASE / "summary.json", summary)
    outputs = {
        p.name: c.sha(p)
        for p in BASE.iterdir()
        if p.is_file() and p.name not in ("prepared.json", "failure.json")
    }
    c.write(
        BASE / "prepared.json",
        {
            "status": "independent_check_pending",
            "outputs": outputs,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "status",
                    "inputs",
                    "benign",
                    "prompt_identities",
                    "unique_message_hashes",
                    "ordinary_calls_planned",
                    "new_provider_calls",
                    "seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    c.require(
        len(sys.argv) == 2 and sys.argv[1] in ("--freeze", "--prepare"),
        "Use --freeze or --prepare",
    )
    with patch.object(
        socket.socket,
        "connect",
        side_effect=RuntimeError("No network in study preparation"),
    ):
        if sys.argv[1] == "--freeze":
            freeze()
        else:
            prepare()

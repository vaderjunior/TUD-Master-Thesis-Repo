from src.hsrag.paths import PROJECT_ROOT
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
import hashlib
import json
from pathlib import Path
import re
import socket
import time
import unicodedata
from unittest.mock import patch
import numpy as np
import pandas as pd
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer
from analysis.feedback.explanations.validate_calibration import original_bucket_class

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/sq3_full_kb_explanations_20261001_v1"
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
OLD = ROOT / "experiments/astra/sq3_full_kb_feedback_20260928_v1"
CAL = ROOT / "experiments/astra/sq3_selective_calibration_20261001_v1"
CARDS = ROOT / "experiments/astra/sq3_explanation_authoring_20261001_v1"
TOK = ROOT / "experiments/astra/qwen_tokenizer_assets_20261001_v1"
ACTIVE = (
    ROOT
    / "experiments/astra/sq3_full_kb_relevance_20260930_v1/api_execution_v1/plan.json"
)
SEEDS = [202609281, 202609282, 202609283]
POLICIES = [
    "plain_always",
    "specific_always",
    "generic_always",
    "plain_selective",
    "specific_selective",
]


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def jl(path):
    with path.open(encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(text):
    return hashlib.sha256(text.encode("utf8")).hexdigest()


def sig(obj):
    return digest(
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    )


def norm(text):
    return (
        re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold())
        .strip()
        .lstrip(": ")
    )


def unit(a):
    a = np.asarray(a, dtype=np.float64)
    ns = np.sqrt(np.sum(a * a, axis=1))
    assert np.isfinite(a).all() and np.all(np.abs(ns - 1) < 0.0002)
    return a / ns[:, None]


def equivalent_gold(left, right):
    if set(left) != set(right) or set(left) != {"hate", "hate_type"}:
        return False
    if type(left["hate"]) is not bool or type(right["hate"]) is not bool:
        return False
    if left["hate"] != right["hate"]:
        return False
    a, b = left["hate_type"], right["hate_type"]
    return (
        isinstance(a, list)
        and isinstance(b, list)
        and len(a) == len(set(a))
        and len(b) == len(set(b))
        and set(a) == set(b)
    )


def amendment_fixtures():
    truth = {"hate": True, "hate_type": ["irony", "inferiority"]}
    assert equivalent_gold(truth, {"hate": True, "hate_type": ["inferiority", "irony"]})
    assert not equivalent_gold(
        truth, {"hate": True, "hate_type": ["irony", "threatening"]}
    )
    assert not equivalent_gold(truth, {"hate": True, "hate_type": ["irony"]})
    assert not equivalent_gold(
        truth, {"hate": True, "hate_type": ["irony", "inferiority", "irony"]}
    )
    assert not equivalent_gold(
        truth, {"hate": False, "hate_type": ["irony", "inferiority"]}
    )
    assert not equivalent_gold(
        truth, {"hate": True, "hate_type": ["irony", "inferiority"], "extra": 1}
    )


def check():
    amendment_fixtures()
    amendment = read(BASE / "checker_amendment_v1.json")
    assert amendment["new_checker_sha256"] == sha(Path(__file__))
    assert amendment["prepared_sha256"] == sha(BASE / "prepared.json")
    start = time.perf_counter()
    assert not (BASE / "complete.json").exists(), "Preserve completed independent check"
    prepared = read(BASE / "prepared.json")
    plan = read(BASE / "preparation_plan.json")
    summary = read(BASE / "summary.json")
    for path, expected in prepared["outputs"].items():
        assert sha(BASE / path) == expected, path
    for path, expected in plan["inputs"].items():
        assert sha(ROOT / path) == expected, path
    for path, expected in read(CAL / "plan.json")["inputs"].items():
        assert sha(ROOT / path) == expected, path
    core = jl(CORE / "dev_inputs.jsonl")
    qmap = {r["item_id"]: i for i, r in enumerate(core)}
    records = jl(CORE / "kb/full/records.jsonl")
    rmap = {r["id"]: r for r in records}
    qv = np.load(
        CORE / "retrieval_v1/query_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    fv = np.load(
        CORE / "retrieval_v1/full_vectors.npy", mmap_mode="r", allow_pickle=False
    )
    assert qv.shape == (1295, 1024) and fv.shape == (71131, 1024)
    assert qv.dtype == fv.dtype == np.dtype("float32")
    for rows, src, key in (
        (records, jl(CORE / "retrieval_v1/record_vector_sources.jsonl"), "id"),
        (core, jl(CORE / "retrieval_v1/query_vector_sources.jsonl"), "key"),
    ):
        assert len(rows) == len(src)
        assert all(
            r[key] == s[key] and digest(r["text"]) == s["text_sha256"]
            for r, s in zip(rows, src)
        )
    d = pd.read_parquet(ROOT / "data/processed/en_dev.parquet").set_index("id")
    assert d.index.is_unique
    oldscience = read(OLD / "scientific_plan_v1.json")
    transfer = set(oldscience["rosters"]["transfer"])
    original = {r["id"]: r for r in jl(OLD / "feasibility_v1/roster.jsonl")}
    benign = {
        r["item_id"]
        for r in core
        if r["lang"] == "en" and d.at[r["item_id"], "gate"] == False
    }
    assert len(transfer) == 350 and len(benign) == 208
    ids = sorted(transfer | benign)
    idmap = {iid: i for i, iid in enumerate(ids)}
    inventory = read(BASE / "scope_inventory.json")
    assert (
        inventory["initial_ids"] == ids
        and inventory["initial_benign"] == 208
        and inventory["transfer"] == 350
    )
    assert inventory["benign_source_counts"] == dict(
        Counter(d.at[iid, "source"] for iid in benign)
    )
    x = unit(qv[[qmap[iid] for iid in ids]])
    normalized = [norm(d.at[iid, "text"]) for iid in ids]
    sim = np.clip(x @ x.T, -1, 1)
    graph = {iid: set() for iid in ids}
    expected_edges = {}
    touched = defaultdict(set)

    def store(kind, left, right, score, exact):
        edge = bool(exact or score >= 0.95)
        expected_edges[kind, left, right] = {
            "cosine": float(score),
            "exact": exact,
            "edge": edge,
            "borderline": abs(score - 0.95) <= 0.0005,
        }
        return edge

    for i, iid in enumerate(ids):
        for j in range(i + 1, len(ids)):
            exact = normalized[i] == normalized[j]
            score = sim[i, j]
            if exact or score >= 0.9495:
                if store("evaluation", iid, ids[j], score, exact):
                    graph[iid].add(ids[j])
                    graph[ids[j]].add(iid)
    pools = sorted(
        [
            p
            for p in read(OLD / "update_payloads_v1/pools.json")["pools"]
            if p["arm"] == "error" and p["size"] == 40
        ],
        key=lambda p: p["seed"],
    )
    assert [p["seed"] for p in pools] == SEEDS
    added = {
        r["provenance"]["source_id"]: r
        for r in jl(OLD / "update_payloads_v1/added_records.jsonl")
    }
    source_ids = sorted({iid for p in pools for iid in p["ids"]})
    cal = jl(CAL / "retained_train.jsonl")
    train = [(i, r) for i, r in enumerate(records) if r["kind"] == "example"]
    assert (len(source_ids), len(cal), len(train)) == (58, 2041, 71091)
    nearest = read(BASE / "overlap_nearest.json")
    maxima = {iid: {} for iid in ids}

    def screen(kind, rids, texts, vectors):
        scores = np.clip(x @ unit(vectors).T, -1, 1)
        lookup = defaultdict(set)
        for j, text in enumerate(texts):
            lookup[norm(text)].add(j)
        for i, iid in enumerate(ids):
            maxima[iid][kind] = max(maxima[iid].get(kind, -2), float(scores[i].max()))
            eligible = (
                set(np.flatnonzero(scores[i] >= 0.9495).tolist())
                | lookup[normalized[i]]
            )
            for j in eligible:
                if store(
                    kind, iid, rids[j], scores[i, j], normalized[i] == norm(texts[j])
                ):
                    touched[iid].add(kind + ":" + rids[j])

    screen(
        "correction",
        source_ids,
        [added[i]["text"] for i in source_ids],
        qv[[qmap[i] for i in source_ids]],
    )
    screen(
        "calibration",
        [r["id"] for r in cal],
        [r["text"] for r in cal],
        fv[[r["record_row"] for r in cal]],
    )

    for n in range(0, len(train), 1536):
        chunk = train[n : n + 1536]
        screen(
            "base_train",
            [r["id"] for _, r in chunk],
            [r["text"] for _, r in chunk],
            fv[[i for i, _ in chunk]],
        )
    observed = jl(BASE / "overlap_evidence.jsonl")
    assert len(observed) == len(expected_edges)
    for r in observed:
        expected = expected_edges.pop((r["kind"], r["left"], r["right"]))
        assert abs(r["cosine"] - expected["cosine"]) < 2e-12
        assert all(r[k] == expected[k] for k in ("exact", "edge", "borderline"))
    assert not expected_edges
    train_index = {r["id"]: i for i, r in train}
    cal_index = {r["id"]: r["record_row"] for r in cal}
    for iid in ids:
        for kind, item in nearest[iid].items():
            assert abs(item["cosine"] - maxima[iid][kind]) < 2e-12
            v = (
                qv[qmap[item["id"]]]
                if kind == "correction"
                else fv[
                    (
                        cal_index[item["id"]]
                        if kind == "calibration"
                        else train_index[item["id"]]
                    )
                ]
            )
            score = float(x[idmap[iid]] @ unit(np.array([v]))[0])
            assert abs(score - item["cosine"]) < 2e-12
    groups = []
    visited = set()
    excluded = {}
    cluster_map = {}
    for iid in ids:
        if iid in visited:
            continue
        todo = [iid]
        component = set()
        while todo:
            current = todo.pop()
            if current in component:
                continue
            component.add(current)
            todo.extend(graph[current] - component)
        visited |= component
        members = sorted(component)
        protected = sorted({key for i in members for key in touched[i]})
        assert not (protected and set(members) & transfer)
        cid = "sq3exp-" + digest("|".join(members))[:16]
        for i in members:
            cluster_map[i] = cid
            if protected:
                excluded[i] = protected
        groups.append(
            {
                "cluster": cid,
                "members": members,
                "protected_keys": protected,
                "mixed_hate_annotations": len({bool(d.at[i, "gate"]) for i in members})
                > 1,
            }
        )
    assert (
        read(BASE / "clusters.json") == groups
        and read(BASE / "excluded_benign.json") == excluded
    )
    roster = jl(BASE / "roster.jsonl")
    roster_map = {r["id"]: r for r in roster}
    assert [r["id"] for r in roster] == [i for i in ids if i not in excluded]
    assert set(transfer) <= set(roster_map)
    for r in roster:
        iid = r["id"]
        raw = d.loc[iid]
        assert (
            r["key"] == core[qmap[iid]]["key"]
            and r["text"] == raw["text"] == core[qmap[iid]]["text"]
        )
        assert r["source"] == raw["source"] and r["text_sha256"] == digest(raw["text"])
        assert r["cluster"] == cluster_map[iid] and r["vector_row"] == qmap[iid]
        assert r["gold"]["hate"] == bool(raw["gate"])
        ts = list(raw["hate_types"]) if raw["hate_types"] is not None else None
        assert r["gold"]["hate_type"] == ts
        assert r["cohort"] == (
            "transfer_hateful" if iid in transfer else "benign_extension"
        )
        if iid in transfer:
            assert (
                equivalent_gold(r["gold"], original[iid]["gold"])
                and r["original_transfer_role"] == original[iid]["role"]
            )
    bases = {r["item_id"]: r for r in jl(CORE / "retrieval_v1/full/prompts.jsonl")}
    cards = {r["source_id"]: r for r in jl(CARDS / "explanation_cards.jsonl")}
    assert set(cards) == set(source_ids)
    for sid, card in cards.items():
        assert card["source_text_sha256"] == digest(added[sid]["text"])
        assert (
            card["hate_types"] == added[sid]["meta"]["hate_types"]
            and card["hate"] is True
        )
        if not card["explanation_available"]:
            assert card["specific_explanation"] == card["generic_explanation"] == ""
    cls = original_bucket_class()
    buckets = {
        p["seed"]: cls([added[i] for i in p["ids"]], qv[[qmap[i] for i in p["ids"]]])
        for p in pools
    }
    choices = jl(BASE / "choices.jsonl")
    cm = {}
    maximum_error = 0.0
    for ch in choices:
        identity = ch["query_id"], ch["pool_seed"]
        assert identity not in cm
        cm[identity] = ch
        row = roster_map[ch["query_id"]]
        b = buckets[ch["pool_seed"]]
        source, trace = b.retrieve(row["text"], qv[row["vector_row"]])
        sid = source["provenance"]["source_id"]
        q = np.asarray(qv[row["vector_row"]], dtype=np.float64)
        q /= np.linalg.norm(q)
        cosine = float(b.vectors[b.ids.index(source["id"])] @ q)
        assert (
            ch["source_id"] == sid
            and abs(ch["cosine"] - cosine) < 2e-12
            and ch["apply"] == (cosine >= 0.64)
        )
        assert (
            ch["cohort"] == row["cohort"]
            and ch["explanation_available"] == cards[sid]["explanation_available"]
        )
        assert ch["retrieval"]["selected_id"] == trace["selected_id"]
        for key in ("dense", "bm25", "rrf"):
            assert [r["id"] for r in ch["retrieval"][key]] == [
                r["id"] for r in trace[key]
            ]
            for left, right in zip(ch["retrieval"][key], trace[key]):
                err = abs(left["score"] - right["score"])
                maximum_error = max(maximum_error, err)
                assert err < 1e-10
    assert set(cm) == {(r["id"], s) for r in roster for s in SEEDS}
    science = read(BASE / "scientific_plan_v1.json")
    assert science["preparation_plan_sha256"] == sha(BASE / "preparation_plan.json")
    assert science["protocol_sha256"] == sha(BASE / "STUDY_PROTOCOL.md") and science[
        "choices_sha256"
    ] == sha(BASE / "choices.jsonl")
    assert science["roster_sha256"] == sha(BASE / "roster.jsonl") and science[
        "cluster_sha256"
    ] == sha(BASE / "clusters.json")
    assert (
        science["policy_sha256"]
        == sha(CAL / "policy.json")
        == "b4ec3967d74e07139864ef42271cce38a46560a92ac8c4f015c685833021c7fb"
    )
    assert science["rosters"]["transfer"] == sorted(transfer) and science["rosters"][
        "benign"
    ] == sorted(benign - set(excluded))
    assert science["rosters"]["original_holdout"] == oldscience["rosters"]["holdout"]
    assert (
        science["rosters"]["historical_sentinels"]
        == read(OLD / "baseline_review_v1/selection.json")["sentinel_ids"]
    )
    assert science["budget"] == {
        "inputs": len(roster),
        "prompt_identities": len(roster) * 16,
        "predictions": len(roster) * 48,
        "ordinary_calls": len(roster) * 144,
    }
    template_parts = re.split(
        r"^=== (\w+) ===$",
        (ROOT / "prompts/classify_v1.txt").read_text("utf8"),
        flags=re.M,
    )
    tpl = {
        template_parts[i]: template_parts[i + 1].strip()
        for i in range(1, len(template_parts), 2)
    }
    tokenizer = Tokenizer.from_file(str(TOK / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    chat = ImmutableSandboxedEnvironment(
        trim_blocks=True, lstrip_blocks=True
    ).from_string(read(TOK / "chat_template.json")["chat_template"])
    oldplain = {}
    with (OLD / "update_payloads_v1/prompts.jsonl").open(encoding="utf8") as stream:
        for line in stream:
            row = json.loads(line)
            if (
                row["item_id"] in transfer
                and row["condition"].startswith("error_")
                and row["condition"].endswith("_b04")
            ):
                oldplain[row["item_id"], int(row["condition"].split("_")[1][1:])] = row
    assert len(oldplain) == 1050
    prompts = jl(BASE / "prompts.jsonl")
    pinventory = jl(BASE / "prompt_inventory.jsonl")
    seen = set()
    tokens_by_hash = {}
    natural = skips = fallbacks = references = 0
    for n, (p, inv) in enumerate(zip(prompts, pinventory), 1):
        iid = p["item_id"]
        seed = p["pool_seed"]
        policy = p["policy"]
        base = bases[iid]
        r = roster_map[iid]
        identity = p["key"], p["condition"]
        assert identity not in seen
        seen.add(identity)
        assert (
            p["key"] == r["key"]
            and p["text"] == r["text"] == base["text"]
            and p["cohort"] == r["cohort"]
        )
        assert (
            p["lang"] == "en"
            and p["strategy"] == "hybrid"
            and p["scientific_plan_sha256"] == sha(BASE / "scientific_plan_v1.json")
        )
        assert set(p) == {
            "key",
            "item_id",
            "cohort",
            "condition",
            "policy",
            "pool_seed",
            "strategy",
            "lang",
            "text",
            "natural_source_id",
            "source_id",
            "apply",
            "explanation_kind",
            "explanation",
            "context",
            "messages",
            "context_sha256",
            "messages_sha256",
            "reference_input_tokens",
            "scientific_plan_sha256",
        }
        expected_context = deepcopy(base["context"])
        expected_messages = deepcopy(base["messages"])
        if policy == "reference":
            assert (
                seed is None
                and p["condition"] == "reference"
                and p["natural_source_id"] is None
            )
            deliver = False
            explanation = ""
            kind = None
            references += 1
        else:
            assert policy in POLICIES and p["condition"] == f"{policy}_s{seed}"
            ch = cm[iid, seed]
            source = added[ch["source_id"]]
            card = cards[ch["source_id"]]
            assert p["natural_source_id"] == ch["source_id"]
            deliver = not policy.endswith("selective") or ch["apply"]
            explanation = ""
            kind = None
            if deliver:
                if policy.startswith("specific"):
                    explanation = card["specific_explanation"]
                    kind = "specific" if explanation else None
                elif policy.startswith("generic"):
                    explanation = card["generic_explanation"]
                    kind = "generic" if explanation else None
                assert source["meta"] == {
                    "gate": True,
                    "hate_types": card["hate_types"],
                    "target_groups": None,
                    "severity": None,
                    "legal": None,
                }
                rec = {
                    "id": source["id"],
                    "text": source["text"],
                    "meta": deepcopy(source["meta"]),
                    "via": "rrf",
                }
                expected_context["example_groups"].append(
                    ["Additional labelled examples", [rec]]
                )
                labels = "hate=true, hate_type=[" + ", ".join(card["hate_types"]) + "]"
                block = (
                    "\n\n## Additional labelled examples\n"
                    + tpl["EXAMPLES_CAUTION"]
                    + '\n\n- "'
                    + " ".join(source["text"].split())
                    + '"\n  labels: '
                    + labels
                )
                if explanation:
                    block += "\n  Explanation of this example's labels: " + explanation
                suffix = "\n\n## Text to classify (language: en)\n" + base["text"]
                assert expected_messages[-1]["text"].endswith(suffix)
                expected_messages[-1]["text"] = (
                    expected_messages[-1]["text"][: -len(suffix)] + block + suffix
                )
                if (
                    policy.startswith(("specific", "generic"))
                    and not card["explanation_available"]
                ):
                    fallbacks += 1
            else:
                skips += 1
            if policy == "plain_always" and iid in transfer:
                assert (
                    p["context"] == oldplain[iid, seed]["context"]
                    and p["messages"] == oldplain[iid, seed]["messages"]
                )
                natural += 1
        assert (
            p["apply"] == deliver
            and p["explanation"] == explanation
            and p["explanation_kind"] == kind
        )
        assert p["source_id"] == (cm[iid, seed]["source_id"] if deliver else None)
        assert p["context"] == expected_context and p["messages"] == expected_messages
        assert p["context_sha256"] == sig(expected_context) and p[
            "messages_sha256"
        ] == sig(expected_messages)
        for hit in (
            base["context"]["definitions"]
            + base["context"]["guidelines"]
            + [h for _, rows in base["context"]["example_groups"] for h in rows]
        ):
            rec = rmap[hit["id"]]
            assert hit["text"] == rec["text"]
            meta = {
                **{
                    k: rec.get(k)
                    for k in ("kind", "lang", "source", "dimension", "label")
                },
                **rec["meta"],
            }
            assert all(hit["meta"].get(k) == v for k, v in meta.items()) and set(
                hit["meta"]
            ) <= set(meta) | {"_ranks"}
        mh = p["messages_sha256"]
        if mh not in tokens_by_hash:
            wire = [
                {"role": m["role"], "content": m["text"]} for m in expected_messages
            ]
            rendered = chat.render(
                messages=wire,
                tools=None,
                add_generation_prompt=True,
                add_vision_id=False,
            )
            tokens_by_hash[mh] = len(
                tokenizer.encode(rendered, add_special_tokens=False).ids
            )
        assert p["reference_input_tokens"] == tokens_by_hash[mh] <= 8192
        assert inv["entry_sha256"] == sig(p)
        assert all(p[k] == v for k, v in inv.items() if k != "entry_sha256")
        if n % 2000 == 0:
            print("Verified prompt identities", n, "/", len(prompts), flush=True)
    expected_ids = {(r["key"], "reference") for r in roster} | {
        (r["key"], f"{p}_s{s}") for r in roster for p in POLICIES for s in SEEDS
    }
    assert (
        len(prompts) == len(pinventory) == len(expected_ids) == len(roster) * 16
        and seen == expected_ids
    )
    assert summary["prompt_identities"] == len(prompts) and summary[
        "unique_message_hashes"
    ] == len(tokens_by_hash)
    assert summary["reference_exact_matches"] == references == len(roster)
    assert summary["natural_transfer_exact_matches"] == natural == 1050
    assert (
        summary["skip_exact_matches"] == skips
        and summary["plain_fallback_exact_matches"] == fallbacks
    )
    assert (
        summary["inputs"] == len(roster)
        and summary["benign"] == len(roster) - 350
        and summary["excluded_benign"] == len(excluded)
    )
    assert summary["overlap_evidence_rows"] == len(observed) and summary[
        "overlap_borderline_rows"
    ] == sum(r["borderline"] for r in observed)
    assert summary["evaluation_clusters"] == len(groups) and summary[
        "retained_clusters"
    ] == len({r["cluster"] for r in roster})
    assert summary["mixed_annotation_clusters"] == sum(
        r["mixed_hate_annotations"] for r in groups
    )
    for cohort in ("transfer_hateful", "benign_extension"):
        for seed in [None] + SEEDS:
            rows = [
                r
                for r in choices
                if r["cohort"] == cohort and (seed is None or r["pool_seed"] == seed)
            ]
            expected = {
                "input_pool_pairs": len(rows),
                "unique_inputs": len({r["query_id"] for r in rows}),
                "applied": sum(r["apply"] for r in rows),
                "explanation_available": sum(r["explanation_available"] for r in rows),
                "specific_selective_delivered": sum(
                    r["apply"] and r["explanation_available"] for r in rows
                ),
            }
            assert (
                summary["coverage"][cohort]["pooled" if seed is None else str(seed)]
                == expected
            )
        for policy in ["reference"] + POLICIES:
            values = [
                p["reference_input_tokens"]
                for p in prompts
                if p["cohort"] == cohort and p["policy"] == policy
            ]
            assert summary["token_statistics"][cohort][policy] == {
                "prompts": len(values),
                "min": min(values),
                "median": float(np.median(values)),
                "mean": float(np.mean(values)),
                "max": max(values),
            }
    for row in summary["source_explanation_token_lengths"]:
        card = cards[row["source_id"]]
        sp = len(
            tokenizer.encode(card["specific_explanation"], add_special_tokens=False).ids
        )
        ge = len(
            tokenizer.encode(card["generic_explanation"], add_special_tokens=False).ids
        )
        assert row == {
            "source_id": card["source_id"],
            "specific": sp,
            "generic": ge,
            "difference": sp - ge,
        }
    assert len(summary["source_explanation_token_lengths"]) == 20
    unique = Counter(
        len({p["messages_sha256"] for p in prompts if p["item_id"] == r["id"]})
        for r in roster
    )
    assert summary["unique_prompt_count_per_input"] == {
        str(k): v for k, v in sorted(unique.items())
    }
    assert (
        summary["ordinary_calls_planned"] == len(roster) * 144
        and summary["new_provider_calls"] == summary["cuda_jobs"] == 0
    )
    active = read(ACTIVE)
    count = 0
    assert (
        sha(ACTIVE)
        == "d4d34f90adf144131e3e59a13cb9c6cd351b7eefc97a093ee71613ee75a0babb"
    )
    for field in ("input_hashes", "source_hashes", "generated_hashes"):
        for name, expected in active[field].items():
            assert sha(ROOT / name) == expected
            count += 1
    assert count == 79
    result = {
        "status": "joint_study_prompts_independently_verified_execution_not_ready",
        "prepared_sha256": sha(BASE / "prepared.json"),
        "checker_sha256": sha(Path(__file__)),
        "scientific_plan_sha256": sha(BASE / "scientific_plan_v1.json"),
        "prompts_sha256": sha(BASE / "prompts.jsonl"),
        "inputs": len(roster),
        "benign": len(roster) - 350,
        "retrieval_pairs": len(choices),
        "prompt_identities": len(prompts),
        "natural_transfer_exact_matches": natural,
        "reference_exact_matches": references,
        "skip_exact_matches": skips,
        "plain_fallback_exact_matches": fallbacks,
        "max_retrieval_score_error": maximum_error,
        "input_bindings_verified": len(plan["inputs"]),
        "active_bindings_verified": count,
        "new_provider_calls": 0,
        "seconds": time.perf_counter() - start,
        "checker_amendment_sha256": sha(BASE / "checker_amendment_v1.json"),
        "amendment_fixtures": 6,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    with (BASE / "complete.json").open("x", encoding="utf8", newline="\n") as f:
        json.dump(result, f, sort_keys=True, indent=2)
        f.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    with patch.object(
        socket.socket,
        "connect",
        side_effect=RuntimeError("No network in independent study check"),
    ):
        check()

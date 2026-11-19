from src.hsrag.paths import PROJECT_ROOT
import os

for name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "2"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import socket
import time
import unicodedata
from unittest.mock import patch
import numpy as np
import pandas as pd
import yaml
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer
from analysis.knowledge.strategies import verify as rank
from analysis.common import prompt as renderer
from analysis.common import prompt_archive as isolated

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/sq3_full_kb_taxonomy_20261001_v1"
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
RET = CORE / "retrieval_v1"
PACKET = ROOT / "experiments/astra/sq3_weekend_preparation_20261001_v2"
TOK = ROOT / "experiments/astra/qwen_tokenizer_assets_20261001_v1"
LABELS = [
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
]


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def rows(p):
    with p.open(encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sig(obj):
    return digest(
        json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
        + "\n"
    )


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split()).lstrip(": ")


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v, axis=1)
    assert np.isfinite(v).all() and np.all(np.abs(norm - 1) < 0.0002)
    return v / norm[:, None]


def check_files(base, hashes):
    for name, expected in hashes.items():
        assert sha(base / name) == expected, "Changed file: " + name


def active():
    path = (
        ROOT
        / "experiments/astra/sq3_full_kb_relevance_20260930_v1/api_execution_v1/plan.json"
    )
    assert (
        sha(path) == "d4d34f90adf144131e3e59a13cb9c6cd351b7eefc97a093ee71613ee75a0babb"
    )
    plan = read(path)
    n = 0
    for key in ("input_hashes", "source_hashes", "generated_hashes"):
        check_files(ROOT, plan[key])
        n += len(plan[key])
    assert n == 79
    return n


def check_scope(records, core, qraw, vraw):
    df = pd.read_parquet(ROOT / "data/processed/en_dev.parquet")
    gold = {r["id"]: r for r in df.to_dict("records")}
    old = rows(
        ROOT
        / "experiments/astra/sq3_full_kb_feedback_20260928_v1/feasibility_v1/roster.jsonl"
    )
    hateful = {r["id"] for r in old}
    benign = {
        r["item_id"]
        for r in core
        if r["lang"] == "en" and gold[r["item_id"]]["gate"] is False
    }
    assert len(hateful) == 467 and len(benign) == 208 and (not hateful & benign)
    ids = sorted(hateful | benign)
    qmap = {r["item_id"]: i for i, r in enumerate(core)}
    q = unit(qraw[[qmap[i] for i in ids]])
    names = [normalized(gold[i]["text"]) for i in ids]
    adjacency = {i: set() for i in range(len(ids))}
    expected_evidence = {}
    touched = set()
    sims = q @ q.T
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            same = names[i] == names[j]
            score = float(sims[i, j])
            edge = same or score >= 0.95
            if edge:
                adjacency[i].add(j)
                adjacency[j].add(i)
            if edge or abs(score - 0.95) <= 0.0005:
                expected_evidence[ids[i], ids[j], "evaluation"] = {
                    "cosine": score,
                    "normalized_equal": same,
                    "edge": edge,
                }
    train = [i for i, r in enumerate(records) if r["kind"] == "example"]
    normmap = defaultdict(list)
    assert len(train) == 71091
    for i in train:
        normmap[normalized(records[i]["text"])].append(records[i]["id"])
    for i, name in enumerate(names):
        for rid in normmap[name]:
            touched.add(i)
            expected_evidence[ids[i], rid, "train_exact"] = {"edge": True}
    nearest = np.full(len(ids), -np.inf)
    nearest_ids = [None] * len(ids)
    for start in range(0, len(train), 777):
        ix = train[start : start + 777]
        sim = q @ unit(vraw[ix]).T
        for i in range(len(ids)):
            k = int(np.argmax(sim[i]))
            score = float(sim[i, k])
            if score > nearest[i]:
                nearest[i], nearest_ids[i] = (score, records[ix[k]]["id"])
            for k in np.flatnonzero(sim[i] >= 0.9495):
                score = float(sim[i, k])
                edge = score >= 0.95
                if edge:
                    touched.add(i)
                expected_evidence[ids[i], records[ix[k]]["id"], "train_cosine"] = {
                    "cosine": score,
                    "edge": edge,
                }
    evidence = rows(BASE / "overlap_evidence.jsonl")
    assert len(evidence) == len(expected_evidence)
    for row in evidence:
        expected = expected_evidence.pop((row["left"], row["right"], row["kind"]))
        assert set(row) == {"left", "right", "kind"} | set(expected)
        for k, v in expected.items():
            assert abs(row[k] - v) < 1e-12 if k == "cosine" else row[k] == v
    assert not expected_evidence
    got_nearest = rows(BASE / "nearest_train.jsonl")
    assert [r["id"] for r in got_nearest] == ids
    for i, r in enumerate(got_nearest):
        assert (
            r["nearest_id"] == nearest_ids[i] and abs(r["cosine"] - nearest[i]) < 1e-12
        )
    components = []
    visited = set()
    for i in range(len(ids)):
        if i in visited:
            continue
        todo = [i]
        group = set()
        while todo:
            j = todo.pop()
            if j in group:
                continue
            group.add(j)
            todo.extend(adjacency[j] - group)
        visited |= group
        members = [ids[j] for j in sorted(group)]
        included = not bool(group & touched)
        components.append(
            {
                "cluster": "taxonomy-" + sig(members)[:16],
                "ids": members,
                "representative": min(
                    members,
                    key=lambda iid: (
                        digest("20261001-taxonomy-representative|" + iid),
                        iid,
                    ),
                ),
                "included": included,
                "reason": None if included else "TRAIN_overlap_in_component",
            }
        )
    assert components == read(BASE / "clusters.json")
    included = {iid for x in components if x["included"] for iid in x["ids"]}
    roster = rows(BASE / "roster.jsonl")
    assert [r["id"] for r in roster] == sorted(included)
    for r in roster:
        d = gold[r["id"]]
        assert (
            r["text"] == d["text"]
            and r["source"] == d["source"]
            and (r["lang"] == "en")
        )
        assert r["gold"]["hate"] is bool(d["gate"]) and r["key"] == "en/" + r["id"]
        if r["id"] in hateful:
            assert r["population"] == "hateful_types" and len(
                r["gold"]["hate_type"]
            ) == len(set(r["gold"]["hate_type"]))
            assert set(r["gold"]["hate_type"]) == set(d["hate_types"])
        else:
            assert r["population"] == "benign" and r["gold"]["hate_type"] is None
        assert r["cluster"] == next(
            (g["cluster"] for g in components if r["id"] in g["ids"])
        )
    assert set(read(BASE / "excluded_inputs.json")) == set(ids) - included
    return (roster, components)


def main():
    start = time.perf_counter()
    assert not (BASE / "complete.json").exists(), "Preserve earlier review"
    plan = read(BASE / "preparation_plan.json")
    check_files(ROOT, plan["inputs"])
    tests = read(BASE / "tests_passed.json")
    assert tests["successful"] and tests["tests_run"] == 16
    check_files(ROOT, tests["source_hashes"])
    prepared = read(BASE / "prepared.json")
    check_files(BASE, prepared["outputs"])
    check_files(BASE, read(BASE / "scope_complete.json")["outputs"])
    assert active() == 79
    assert prepared["preparation_plan_sha256"] == sha(BASE / "preparation_plan.json")
    records = rows(CORE / "kb/full/records.jsonl")
    record_map = {r["id"]: r for r in records}
    ix = {r["id"]: i for i, r in enumerate(records)}
    core = rows(CORE / "dev_inputs.jsonl")
    qraw = np.load(RET / "query_vectors.npy", mmap_mode="r", allow_pickle=False)
    vraw = np.load(RET / "full_vectors.npy", mmap_mode="r", allow_pickle=False)
    vectors = np.load(BASE / "indexed_vectors.npy", mmap_mode="r", allow_pickle=False)
    assert vectors.dtype == np.float32 and vectors.shape == vraw.shape == (71131, 1024)
    assert np.isfinite(vectors).all() and float(np.max(np.abs(vectors - vraw))) < 2e-07
    roster, components = check_scope(records, core, qraw, vraw)
    queries = {r["key"]: qraw[i] for i, r in enumerate(core)}
    tax = yaml.safe_load((ROOT / "config/taxonomy.yaml").read_text(encoding="utf8"))
    assert (BASE / "taxonomies/full.yaml").read_bytes() == (
        ROOT / "config/taxonomy.yaml"
    ).read_bytes()
    saved_contexts = {}
    totals = Counter()
    maximum_error = 0.0
    min_margin = 1.0
    replay_changes = {}
    raw_lines = (CORE / "kb/full/records.jsonl").read_bytes().splitlines(keepends=True)
    for pool in ["full"] + LABELS:
        if pool == "full":
            retained = records
        else:
            remove = {
                r["id"]
                for r in records
                if r["kind"] == "example"
                and pool in (r["meta"].get("hate_types") or [])
                or (
                    r["kind"] == "definition"
                    and r["dimension"] == "hate_type"
                    and (r["label"] == pool)
                )
            }
            manifest = read(PACKET / f"taxonomy_manifests/{pool}.json")
            assert remove == set(
                manifest["removed_example_ids"] + manifest["removed_definition_ids"]
            )
            retained = [r for r in records if r["id"] not in remove]
            assert (BASE / f"kb/{pool}/records.jsonl").read_bytes() == b"".join(
                (raw for raw, r in zip(raw_lines, records) if r["id"] not in remove)
            )
            expected = deepcopy(tax)
            del expected["dimensions"]["hate_type"]["labels"][pool]
            assert (
                yaml.safe_load(
                    (BASE / f"taxonomies/{pool}.yaml").read_text(encoding="utf8")
                )
                == expected
            )
        bundle = isolated.bundle(BASE / f"taxonomies/{pool}.yaml")
        assert bundle.schema.Result.model_json_schema() == read(
            BASE / f"taxonomies/{pool}.schema.json"
        )
        selected = vectors[[ix[r["id"]] for r in retained]]
        checker = rank.TraceAudit(retained, selected, queries)
        first = rows(BASE / f"retrieval/{pool}.jsonl")
        replay = rows(BASE / f"replay/{pool}.jsonl")
        assert (
            [r["key"] for r in first]
            == [r["key"] for r in roster]
            == [r["key"] for r in replay]
        )
        saved_contexts[pool] = {}
        differences = Counter()
        known = {r["id"]: r for r in retained}
        for row, entry, cold in zip(roster, first, replay):
            assert (
                entry["pool"] == pool
                and entry["text"] == row["text"]
                and (entry["lang"] == "en")
            )
            checker.entry(entry)
            context = rank.source_context(entry["final_hits"], known)
            assert sig(context) == sig(entry["context"]) == entry["context_sha256"]
            assert entry["retrieved"] == renderer.hit_ids(context)
            saved_contexts[pool][row["key"]] = context
            flags = cold["matches"]
            assert set(flags) == {"context", "retrieved", "final_hits", "trace"}
            assert all((type(value) is bool for value in flags.values()))
            if all(flags.values()):
                assert set(cold) == {"key", "matches"}
            else:
                actual = cold["actual_entry"]
                assert (
                    actual["key"] == row["key"]
                    and actual["text"] == row["text"]
                    and (actual["pool"] == pool)
                )
                checker.entry(actual)
                assert (
                    sig(rank.source_context(actual["final_hits"], known))
                    == sig(actual["context"])
                    == actual["context_sha256"]
                )
                assert flags == {k: actual[k] == entry[k] for k in flags}
                differences.update((k for k, v in flags.items() if not v))
        assert (
            dict(differences)
            == read(BASE / f"replay/{pool}.complete.json")["differences"]
        )
        replay_changes[pool] = dict(differences)
        totals.update(checker.counts)
        maximum_error = max(maximum_error, checker.max_dense_score_error)
        min_margin = min(min_margin, checker.min_mmr_margin)
        print(
            f"Independently checked taxonomy {pool}: {len(first)} retrievals and cold alternatives",
            flush=True,
        )
        del checker, selected, first, replay
        gc.collect()
    science = read(BASE / "scientific_plan_v1.json")
    prompts = rows(BASE / "prompts.jsonl")
    assert science["protocol_sha256"] == sha(BASE / "STUDY_PROTOCOL.md")
    assert (
        science["input_count"] == len(roster)
        and len(prompts) == science["prompts"] == len(roster) * 23
    )
    expected_conditions = [
        {"condition": "rag_full", "pool": "full", "taxonomy": "full"},
        {"condition": "zs_full", "pool": None, "taxonomy": "full"},
    ]
    for label in LABELS:
        expected_conditions.extend(
            [
                {"condition": "rag_removed_" + label, "pool": label, "taxonomy": label},
                {"condition": "rag_schema_" + label, "pool": label, "taxonomy": "full"},
                {"condition": "zs_removed_" + label, "pool": None, "taxonomy": label},
            ]
        )
    assert science["conditions"] == expected_conditions
    assert science["primary_mask_conditions"] == ["rag_full"] + [
        "rag_schema_" + l for l in LABELS
    ]
    assert (
        science["primary_interval"] == 0.9875
        and science["bootstrap_draws"] == 10000
        and (science["primary_bootstrap_seed"] == 2026100103)
    )
    render = renderer.module("corrected")
    tpl = render._load_template()
    taxonomies = {
        name: yaml.safe_load(
            (BASE / f"taxonomies/{name}.yaml").read_text(encoding="utf8")
        )
        for name in ["full"] + LABELS
    }
    environment = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
    chat = environment.from_string(
        (TOK / "chat_template.jinja").read_text(encoding="utf8")
    )
    tokenizer = Tokenizer.from_file(str(TOK / "tokenizer.json"))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    lookup = {}
    token_min = 10**9
    token_max = 0
    unique = set()
    for index, (row, condition) in enumerate(
        ((r, c) for r in roster for c in expected_conditions)
    ):
        entry = prompts[index]
        assert (
            entry["key"] == row["key"]
            and entry["item_id"] == row["id"]
            and (entry["text"] == row["text"])
            and (entry["lang"] == "en")
        )
        assert all((entry[k] == v for k, v in condition.items()))
        context = (
            saved_contexts[condition["pool"]][row["key"]]
            if condition["pool"]
            else {"definitions": [], "guidelines": [], "example_groups": []}
        )
        assert sig(entry["context"]) == sig(context) == entry["context_sha256"]
        messages = renderer.render_messages(row["text"], "en", context, "corrected")
        messages[0]["text"] = tpl["SYSTEM"].format(
            label_space=render.render_label_space(taxonomies[condition["taxonomy"]])
        )
        assert (
            messages == entry["messages"] and sig(messages) == entry["messages_sha256"]
        )
        assert entry["taxonomy_sha256"] == sha(
            BASE / f"taxonomies/{condition['taxonomy']}.yaml"
        )
        assert entry["scientific_plan_sha256"] == sha(BASE / "scientific_plan_v1.json")
        body = [{"role": m["role"], "content": m["text"]} for m in messages]
        rendered = chat.render(
            messages=body, tools=None, add_generation_prompt=True, add_vision_id=False
        )
        count = len(tokenizer.encode(rendered, add_special_tokens=False).ids)
        assert count == entry["input_tokens_local"] and count <= 8192
        token_min = min(token_min, count)
        token_max = max(token_max, count)
        unique.add(entry["messages_sha256"])
        lookup[row["key"], condition["condition"]] = entry
    delivery = defaultdict(Counter)
    historical = {r["key"]: r for r in rows(RET / "full/prompts.jsonl")}
    changes = []
    for row in roster:
        full = lookup[row["key"], "rag_full"]
        old = historical[row["key"]]
        if full["messages"] != old["messages"]:
            changes.append(
                {
                    "id": row["id"],
                    "old_messages_sha256": old["messages_sha256"],
                    "new_messages_sha256": full["messages_sha256"],
                    "old_retrieved": old["retrieved"],
                    "new_retrieved": renderer.hit_ids(full["context"]),
                }
            )
        for label in LABELS:
            removed = lookup[row["key"], "rag_removed_" + label]
            restored = lookup[row["key"], "rag_schema_" + label]
            assert (
                removed["context"] == restored["context"]
                and removed["messages"][1] == restored["messages"][1]
            )
            a, z = (
                removed["messages"][0]["text"].splitlines(),
                restored["messages"][0]["text"].splitlines(),
            )
            assert (
                len(a) == len(z)
                and sum((left != right for left, right in zip(a, z))) == 1
            )
            prefix = row["population"] + "_"
            delivery[label][prefix + "same_user_context_as_full"] += (
                removed["messages"][1] == full["messages"][1]
            )
            ctx = full["context"]
            delivery[label][prefix + "full_target_example_reached"] += any(
                (
                    label in (h["meta"].get("hate_types") or [])
                    for _, hh in ctx["example_groups"]
                    for h in hh
                )
            )
            delivery[label][prefix + "full_target_definition_reached"] += any(
                (h["id"] == "def-hate_type-" + label for h in ctx["definitions"])
            )
    assert changes == read(BASE / "historical_full_differences.json")
    summary = read(BASE / "summary.json")
    scope_summary = read(BASE / "scope_summary.json")
    assert summary["delivery"] == {k: dict(v) for k, v in delivery.items()}
    assert (
        summary["unique_messages"] == len(unique)
        and summary["local_input_tokens_min"] == token_min
        and (summary["local_input_tokens_max"] == token_max)
    )
    assert scope_summary["counts"] == dict(Counter((r["population"] for r in roster)))
    assert scope_summary["type_support"] == dict(
        Counter((l for r in roster for l in r["gold"]["hate_type"] or []))
    )
    assert scope_summary["benign_sources"] == dict(
        Counter((r["source"] for r in roster if r["population"] == "benign"))
    )
    assert scope_summary["clusters"] == sum((g["included"] for g in components))
    check_files(ROOT, plan["inputs"])
    assert active() == 79
    result = {
        "status": "taxonomy_prompts_independently_verified_no_classifier_calls",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "prepared_sha256": sha(BASE / "prepared.json"),
        "checker_sha256": sha(Path(__file__)),
        "input_bindings": len(plan["inputs"]),
        "active_bindings": 79,
        "inputs": len(roster),
        "clusters": sum((g["included"] for g in components)),
        "prompts_checked": len(prompts),
        "trace_checks_including_changed_replay": dict(totals),
        "dense_score_max_error": maximum_error,
        "minimum_mmr_margin": min_margin,
        "cold_replay_differences": replay_changes,
        "historical_full_message_differences": len(changes),
        "local_token_min": token_min,
        "local_token_max": token_max,
        "seconds": time.perf_counter() - start,
        "Qwen_calls": 0,
        "new_embeddings": 0,
        "CUDA": False,
        "limits": [
            "Logical eligible-record filtering, not a rebuilt ANN graph.",
            "ANN candidates audited, not exact-neighbor optimality.",
            "Cluster screen is a heuristic; historical DEV exposure remains.",
            "No classification effectiveness result yet; execution integration remains.",
        ],
    }
    with (BASE / "complete.json").open("x", encoding="utf8", newline="\n") as f:
        json.dump(
            result, f, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        f.write("\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    with patch.object(
        socket.socket,
        "connect",
        side_effect=RuntimeError("Network forbidden in offline checker"),
    ):
        main()

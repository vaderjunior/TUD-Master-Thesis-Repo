"""Guidelines inventory."""

import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
from unittest.mock import patch
import numpy as np
import pandas as pd
import yaml
from analysis.common import retrieval as r
from analysis.knowledge.banks import build as u

ROOT = r.ROOT
BASE = ROOT / "experiments/astra/sq1_full_kb_guideline_oracle_20260925_v1"
DEST = BASE / "input_inventory_v1"
EVIDENCE = (
    ROOT
    / "Thesis_Log/astra_result/evidence/sq1_guideline_input_inventory_20260925_v1.json"
)
DATA = ROOT / "data/processed/en_hatecheck_eval.parquet"
OLD = ROOT / "Thesis_Log/astra_result/evidence/guideline_oracle_20260909_v1.json"
AUDIT = ROOT / "experiments/astra/english_kb_audit_20260914_v1"
CLEAN = ROOT / "experiments/astra/english_kb_cleanup_20260914_v1"
CORE = r.CORE


def write_new(path, value):
    path = Path(path)
    r.require(path.resolve().is_relative_to(ROOT), "Write outside workspace")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as handle:
        json.dump(
            value, handle, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        handle.write("\n")


def forbidden(*args, **kwargs):
    raise AssertionError("Offline inventory; no network")


def main():
    r.require(
        not EVIDENCE.exists() and (not DEST.exists()),
        "Preserve completed/partial inventory",
    )
    bindings = {}

    def bind(path, expected=None):
        digest = r.sha(path)
        if expected is not None:
            r.require(digest == expected, "Source changed: " + r.rel(path))
        bindings[r.rel(path)] = digest
        return digest

    old = r.read(OLD)
    old_inputs = {x["path"]: x["sha256"] for x in old["inputs"]}
    bind(OLD)
    bind(DATA, old_inputs[r.rel(DATA)])
    runner = ROOT / "analysis/prototype/guideline_run.py"
    bind(runner, old_inputs[r.rel(runner)])
    matches = [
        node.value
        for node in ast.parse(runner.read_text(encoding="utf-8-sig")).body
        if isinstance(node, ast.Assign)
        and any((isinstance(t, ast.Name) and t.id == "MAPPING" for t in node.targets))
    ]
    r.require(len(matches) == 1, "Ambiguous historical mapping")
    mapping = ast.literal_eval(matches[0])
    data = pd.read_parquet(DATA).to_dict("records")
    selected = []
    blocks = Counter()
    template_functions = defaultdict(set)
    for row in data:
        meta = json.loads(row["raw"])
        blocks[meta["block"]] += 1
        if meta["block"] != "guideline_covered":
            continue
        gate = bool(row["gate"]) if isinstance(row["gate"], np.bool_) else row["gate"]
        r.require(
            gate is False
            and row["lang"] == "en"
            and (meta["functionality"] in mapping),
            "Unexpected eligible case",
        )
        r.require(
            type(meta["templ_id"]) is int
            and str(row["id"]) == "hatecheck-" + str(meta["case_id"]),
            "Bad identity/template",
        )
        selected.append(
            {
                "key": "en/" + row["id"],
                "item_id": row["id"],
                "lang": "en",
                "text": row["text"],
                "text_sha256": hashlib.sha256(row["text"].encode("utf8")).hexdigest(),
                "gate": False,
                "functionality": meta["functionality"],
                "templ_id": meta["templ_id"],
                "ref_case_id": meta["ref_case_id"],
            }
        )
        template_functions[meta["templ_id"]].add(meta["functionality"])
    oracle = next((x for x in old["pools"] if x["name"] == "oracle_only"))
    r.require(
        len(selected) == len({x["item_id"] for x in selected}) == 693
        and len(template_functions) == 302,
        "Roster size differs",
    )
    r.require(
        {x["item_id"] for x in selected} == set(oracle["item_ids"]),
        "Historical oracle roster changed",
    )
    groups = {
        fn: sorted(
            [x for x in selected if x["functionality"] == fn],
            key=lambda x: (x["templ_id"], int(x["item_id"].split("-")[1])),
        )
        for fn in sorted(mapping)
    }
    selected = [
        group[index]
        for index in range(max(map(len, groups.values())))
        for group in groups.values()
        if index < len(group)
    ]
    core = r.read(CORE / "plan.json")
    bind(CORE / "plan.json", r.CORE_SHA)
    full_path = CORE / "kb/full/records.jsonl"
    bind(full_path, core["generated_hashes"][r.rel(full_path)])
    records = r.rows(full_path)
    by_id = {x["id"]: x for x in records}
    r.require(len(records) == len(by_id) == 71131, "Full KB identity differs")
    full_en = [x for x in records if x["kind"] == "example" and x["lang"] == "en"]
    clean = r.read(CLEAN / "complete.json")
    bind(CLEAN / "complete.json")
    bind(CLEAN / "policy.json", clean["outputs"][r.rel(CLEAN / "policy.json")])
    clean_path = CLEAN / "full_english_records.jsonl"
    bind(clean_path, clean["outputs"][r.rel(clean_path)])
    r.require(
        {x["id"]: x for x in full_en} == {x["id"]: x for x in r.rows(clean_path)}
        and len(full_en) == 55753,
        "Frozen cleaned English pool differs",
    )
    rules = {}
    directions = yaml.safe_load(
        (ROOT / "config/decisive_guidelines.yaml").read_text(encoding="utf8")
    )["decisive"]
    legacy = {x["id"]: x for x in r.rows(ROOT / "kb/records.jsonl")}
    bind(ROOT / "kb/records.jsonl", old_inputs["kb/records.jsonl"])
    for fn, ids in mapping.items():
        r.require(len(ids) == len(set(ids)) == 3, "Repeated rule slots")
        for rid in ids:
            rec = by_id[rid]
            r.require(
                rec["kind"] == "guideline" and rec["lang"] == "en",
                "Missing/non-English mapped rule",
            )
            r.require(
                rec == legacy[rid], "Mapped guideline changed since original oracle"
            )
            rules[rid] = {
                "record": rec,
                "configured_decisive_direction": directions.get(rid),
                "configured_direction_present": rid in directions,
            }
    lookup = defaultdict(list)
    for rec in full_en:
        lookup[u.norm(rec["text"])].append(rec["id"])
    overlaps = [
        {"item_id": x["item_id"], "training_records": lookup[u.norm(x["text"])]}
        for x in selected
        if u.norm(x["text"]) in lookup
    ]
    r.require(not overlaps, "New exact-normalized overlap requires review")
    plan = r.read(AUDIT / "plan.json")
    done = r.read(AUDIT / "complete.json")
    plan_sha = bind(AUDIT / "plan.json", done["plan_sha256"])
    bind(AUDIT / "complete.json")
    bind(DATA, plan["input_hashes"][r.rel(DATA)])
    r.require(
        plan["model"] == r.MODEL
        and plan["embedding_cache"]["revision"] == r.REVISION
        and (plan["precision"] == "float32"),
        "Embedding recipe differs",
    )
    held_path = AUDIT / "heldout_texts.jsonl"
    bind(held_path, plan["generated_hashes"][r.rel(held_path)])
    held = r.rows(held_path)
    wanted = {x["text"] for x in selected}
    found = {}
    chunks = {}
    wanted_ids = defaultdict(set)
    for x in selected:
        wanted_ids[x["text"]].add(x["item_id"])
    for start in range(0, len(held), plan["cache_chunk"]):
        block = held[start : start + plan["cache_chunk"]]
        picks = [i for i, x in enumerate(block) if x["text"] in wanted]
        if not picks:
            continue
        identity = {
            "plan_sha256": plan_sha,
            "model": r.MODEL,
            "revision": r.REVISION,
            "precision": "float32",
            "texts_sha256": r.signature([x["text"] for x in block]),
            "rows": len(block),
        }
        path = (
            AUDIT
            / f"embedding_chunks/heldout_{start:06d}_{r.signature(identity)[:16]}.npz"
        )
        bind(path, done["embedding_hashes"][r.rel(path)])
        vectors = r.load_chunk(path, identity)
        chunks[r.rel(path)] = {"rows": len(block), "selected_rows": len(picks)}
        for i in picks:
            text = block[i]["text"]
            refs = {(v["file"], v["id"]) for v in block[i]["references"]}
            r.require(
                all(((r.rel(DATA), rid) in refs for rid in wanted_ids[text])),
                "Cached data-source identity differs",
            )
            r.require(text not in found, "Ambiguous exact-text cache entry")
            found[text] = (
                vectors[i].copy(),
                {"file": r.rel(path), "row": i, "source_kind": "heldout"},
            )
    r.require(
        set(found) == wanted, "Missing cached oracle query; no fresh encoding permitted"
    )
    vectors = np.asarray([found[x["text"]][0] for x in selected], dtype=np.float32)
    r.validate_vectors(vectors, len(selected))
    vector_plan = r.read(r.PLAN)
    bind(r.PLAN)
    full_vector_path = r.BASE / "full_vectors.npy"
    bind(full_vector_path, vector_plan["generated_hashes"][r.rel(full_vector_path)])
    full_vectors = np.load(full_vector_path, allow_pickle=False, mmap_mode="r")
    r.validate_vectors(full_vectors, len(records))
    prior_en_queries = {
        x["text"]: x for x in r.rows(CORE / "dev_inputs.jsonl") if x["lang"] == "en"
    }
    shared_dev = [x["item_id"] for x in selected if x["text"] in prior_en_queries]
    for path in (
        Path(__file__),
        Path(r.__file__),
        Path(u.__file__),
        ROOT / "config/decisive_guidelines.yaml",
        ROOT / "scripts/make_hatecheck_subset.py",
        ROOT / "src/hsrag/prompt.py",
        ROOT / "prompts/classify_v1.txt",
        ROOT / "config/taxonomy.yaml",
        ROOT
        / "Thesis_Log/astra_result/evidence/sq1_allocation_completion_20260925_v1.json",
        ROOT
        / "Thesis_Log/astra_result/evidence/sq1_components_completion_20260924_v1.json",
    ):
        bind(path)
    DEST.mkdir(parents=True)
    with (DEST / "query_vectors.npy").open("xb") as handle:
        np.save(handle, vectors, allow_pickle=False)
    with (DEST / "inputs.jsonl").open("x", encoding="utf8", newline="\n") as handle:
        for x in selected:
            handle.write(
                json.dumps(x, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
    write_new(
        DEST / "query_sources.json", {x["key"]: found[x["text"]][1] for x in selected}
    )
    write_new(
        DEST / "rule_mapping.json",
        {
            fn: {"matched": ids[0], "alternative": ids[1], "filler": ids[2]}
            for fn, ids in mapping.items()
        },
    )
    outputs = {
        r.rel(DEST / name): r.sha(DEST / name)
        for name in (
            "query_vectors.npy",
            "inputs.jsonl",
            "query_sources.json",
            "rule_mapping.json",
        )
    }
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "offline_roster_rules_and_query_cache_verified",
        "input_hashes": bindings,
        "generated_hashes": outputs,
        "original_oracle_evidence_sha256": r.sha(OLD),
        "source_dataset": r.rel(DATA),
        "dataset_rows": len(data),
        "blocks": dict(blocks),
        "selected_rows": len(selected),
        "selected_unique_texts": len(wanted),
        "selected_templates": len(template_functions),
        "by_functionality": {
            fn: {"rows": len(group), "templates": len({x["templ_id"] for x in group})}
            for fn, group in groups.items()
        },
        "templates_spanning_functionalities": {
            str(k): sorted(v) for k, v in template_functions.items() if len(v) > 1
        },
        "all_selected_gold_non_hate": True,
        "selected_exact_text_shared_with_core_dev": shared_dev,
        "full_pool_counts": dict(
            Counter((x["kind"] + "/" + x["lang"] for x in records))
        ),
        "exact_normalized_English_train_matches": overlaps,
        "cleaned_English_pool_exactly_matches_core": True,
        "rule_mapping": mapping,
        "rules": rules,
        "missing_decisive_direction_metadata": [
            rid for rid in rules if rid not in directions
        ],
        "verified_cache_chunks": chunks,
        "cached_query_shape": list(vectors.shape),
        "cached_full_vector_shape": list(full_vectors.shape),
        "new_embedding_calls": 0,
        "CUDA_used": False,
        "new_retrieval_calls": 0,
        "new_API_calls": 0,
        "new_gold_scoring": False,
        "limits": [
            "Existing, previously inspected functional-test stratum; no unseen-test claim or general prevalence estimate.",
            "All 693 selected texts are non-hateful: proposed experiment estimates false alarms, not hate recall or overall hate-detection quality.",
            "Original functionality mapping is inherited, not a new per-item proof that a rule applies or its alternative cannot apply.",
            "Selected oracle rule group-vs-individual has no explicit entry in decisive_guidelines.yaml; do not assert every mapped rule is certified there.",
            "No new semantic overlap scan. Frozen English cleanup includes HateCheck and retained-exception qualifications; exact-normalized check does not prove semantic independence.",
            "No index opened and no prompt delivery checked yet; this is input/cache preparation, not live-run readiness.",
        ],
        "next": "Freeze concrete protocol, then implement/test isolated CPU prompt preparation and replay; API preparation/execution requires a later checkpoint.",
    }
    write_new(EVIDENCE, report)
    write_new(
        DEST / "complete.json",
        {
            "status": report["status"],
            "evidence": r.rel(EVIDENCE),
            "evidence_sha256": r.sha(EVIDENCE),
        },
    )
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "dataset_rows",
                    "blocks",
                    "selected_rows",
                    "selected_unique_texts",
                    "selected_templates",
                    "by_functionality",
                    "templates_spanning_functionalities",
                    "missing_decisive_direction_metadata",
                    "cached_query_shape",
                    "cached_full_vector_shape",
                    "selected_exact_text_shared_with_core_dev",
                    "full_pool_counts",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    print(
        json.dumps(
            {
                "guideline_texts": {
                    rid: row["record"]["text"] for rid, row in rules.items()
                },
                "cache_chunks": len(chunks),
                "evidence_sha256": r.sha(EVIDENCE),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket, "create_connection", forbidden
    ):
        main()

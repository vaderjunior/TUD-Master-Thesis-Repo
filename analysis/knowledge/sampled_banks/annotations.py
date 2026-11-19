"""Sampled banks annotations."""

from src.hsrag.paths import PROJECT_ROOT
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import socket
import time
from analysis.common import prompt as renderer

ROOT = PROJECT_ROOT
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
BASE = ROOT / "experiments/astra/sq1_annotation_coverage_20260926_v1"
EVIDENCE = ROOT / "Thesis_Log/astra_result/evidence/sq1_annotation_coverage_20260926_v1.json"
FIELDS = ("gate", "target_groups", "hate_types", "severity", "legal")
LABELS = {
    "gate": ("false", "true"),
    "target_groups": (
        "race",
        "religion",
        "gender",
        "sexual_orientation",
        "disability",
        "national_origin",
        "other",
    ),
    "hate_types": (
        "explicit",
        "incitement",
        "inferiority",
        "irony",
        "stereotyping",
        "threatening",
        "grievance",
    ),
    "severity": ("low", "medium", "high"),
    "legal": ("propaganda", "incitement_public_peace", "insult_defamation"),
}
SCOPES = {
    "en_dev_eval_main": ("en", "gate", 299),
    "en_dev_eval_targets": ("en", "target_groups", 56),
    "en_dev_eval_sq3_types": ("en", "hate_types", 467),
    "en_dev_eval_types": ("en", "hate_types", 103),
    "en_dev_eval_severity": ("en", "severity", 45),
    "de_dev_eval": ("de", "gate", 150),
    "de_legal_dev_eval": ("de", "legal", 175),
}
PINNED_CORE_SHA = "5c610f4fe320b5e0f185315933d47b388ab645b20b52eed738541a1579184e4e"
PINNED_REVIEW_SHA = "48c6191f369b814f1a23b3bbf79d1b7af561e1ed603e7ba175b4dc938eda49d5"


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def block_network(*args, **kwargs):
    raise RuntimeError("Offline annotation audit: network access forbidden")


def rel(p):
    return Path(p).relative_to(ROOT).as_posix()


def encoded(obj):
    return (
        json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    ).encode("utf8")


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding="utf8"))


def rows(p):
    with Path(p).open(encoding="utf8") as f:
        return [json.loads(line) for line in f if line.strip()]


def save_new(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("xb") as f:
        f.write(encoded(obj))


def gate_name(meta):
    gate = meta.get("gate")
    require(gate is None or type(gate) is bool, "Invalid gate annotation")
    return "unknown" if gate is None else str(gate).lower()


def annotation_state(meta, field):
    value = meta.get(field)
    if value is None:
        return "unannotated"
    if field == "gate":
        require(type(value) is bool, "Invalid non-boolean gate")
    elif field in ("target_groups", "hate_types", "legal"):
        require(type(value) is list, "Invalid multilabel shape")
        require(all((type(v) is str and v in LABELS[field] for v in value)), "Invalid label")
        require(len(value) == len(set(value)), "Duplicate label")
        if not value:
            return "annotated_empty"
    else:
        require(type(value) is str and value in LABELS[field], "Invalid severity")
    return "annotated_value"


def occurrences(meta, field):
    state = annotation_state(meta, field)
    if state == "unannotated":
        return []
    value = meta[field]
    if field == "gate":
        return [str(value).lower()]
    return value if type(value) is list else [value]


def eligible(meta, field):
    """Same eligibility semantics as core head controls; legal is ungated."""
    return meta.get(field) is not None and (
        field not in ("target_groups", "hate_types", "severity") or meta.get("gate") is True
    )


def describe(records):
    require(bool(records), "Empty inventory")
    fields = {}
    for field in FIELDS:
        states = Counter((annotation_state(r["meta"], field) for r in records))
        labels = Counter((v for r in records for v in occurrences(r["meta"], field)))
        fields[field] = {
            "states": {s: states[s] for s in ("unannotated", "annotated_empty", "annotated_value")},
            "annotated": len(records) - states["unannotated"],
            "label_occurrences": {v: labels[v] for v in LABELS[field]},
            "core_head_eligible": sum((eligible(r["meta"], field) for r in records)),
            "annotated_by_gate": {
                g: sum(
                    (
                        r["meta"].get(field) is not None and gate_name(r["meta"]) == g
                        for r in records
                    )
                )
                for g in ("true", "false", "unknown")
            },
        }
    signatures = Counter(
        ("".join(("1" if r["meta"].get(f) is not None else "0" for f in FIELDS)) for r in records)
    )
    return {
        "records_or_slots": len(records),
        "unique_records": len({r["id"] for r in records}),
        "sources": dict(sorted(Counter((r["source"] for r in records)).items())),
        "gate": {
            g: sum((gate_name(r["meta"]) == g for r in records))
            for g in ("true", "false", "unknown")
        },
        "fields": fields,
        "annotation_signatures_field_order": list(FIELDS),
        "annotation_signatures": dict(sorted(signatures.items())),
    }


def validate_hit(hit, record):
    require(
        hit["id"] == record["id"] and hit["text"] == record["text"], "Hit identity/text changed"
    )
    expected = {
        **record["meta"],
        **{f: record[f] for f in ("lang", "source", "dimension", "label", "kind")},
    }
    actual = {k: v for k, v in hit["meta"].items() if k != "_ranks"}
    require(actual == expected, "Supplied metadata differs from frozen pool")
    require(hit["via"] != "static", "Unexpected static example in RAG prompt")


def main():
    started = time.perf_counter()
    socket.socket.connect = block_network
    socket.create_connection = block_network
    require(not (BASE / "analysis_plan.json").exists(), "Preserve completed or partial audit")
    require(digest(CORE / "plan.json") == PINNED_CORE_SHA, "Core plan changed")
    core = read(CORE / "plan.json")
    api = read(CORE / "api_execution_v1/plan.json")
    review = read(CORE / "retrieval_v1/review_complete_v2.json")
    require(
        review["evidence_sha256"] == PINNED_REVIEW_SHA == digest(ROOT / review["evidence"]),
        "Retrieval acceptance changed",
    )
    require(api["core_sha256"] == PINNED_CORE_SHA, "API/core binding mismatch")
    source_paths = [
        Path(__file__),
        ROOT / "tests/knowledge/sampled_banks/test_annotations.py",
        ROOT / "analysis/knowledge/sampled_banks/verify_annotations.py",
        ROOT / "analysis/common/prompt.py",
        ROOT / "src/hsrag/prompt.py",
        ROOT / "prompts/classify_v1.txt",
        ROOT / "config/taxonomy.yaml",
    ]
    paths = source_paths + [
        CORE / "plan.json",
        CORE / "api_execution_v1/plan.json",
        CORE / "retrieval_v1/plan.json",
        CORE / "retrieval_v1/review_complete_v2.json",
        ROOT / review["evidence"],
        CORE / "dev_inputs.jsonl",
    ]
    pinned = {
        **core["input_hashes"],
        **core["generated_hashes"],
        **read(CORE / "retrieval_v1/plan.json")["input_hashes"],
        **api["input_hashes"],
        **api["source_hashes"],
    }
    for p in source_paths[3:] + [CORE / "dev_inputs.jsonl"]:
        require(
            rel(p) in pinned and digest(p) == pinned[rel(p)],
            "Historical input/source changed: " + rel(p),
        )
    kb = {}
    prompts = {}
    for pool, total in [("small", 340), ("full", 71131)]:
        kp = CORE / f"kb/{pool}/records.jsonl"
        pp = CORE / f"retrieval_v1/{pool}/prompts.jsonl"
        cp = CORE / f"retrieval_v1/{pool}/complete.json"
        paths += [kp, pp, cp]
        require(digest(kp) == core["generated_hashes"][rel(kp)], "KB changed")
        require(
            digest(pp) == api["input_hashes"][rel(pp)] == review["prompt_hashes"][pool],
            "Prompts changed",
        )
        rr = rows(kp)
        pr = rows(pp)
        kb[pool] = {r["id"]: r for r in rr}
        prompts[pool] = {r["key"]: r for r in pr}
        require(len(rr) == len(kb[pool]) == total, "Duplicate or missing KB IDs")
        require(len(pr) == len(prompts[pool]) == 1295, "Duplicate or missing prompt IDs")
        require(read(cp)["prompt_sha256"] == digest(pp), "Prompt completion changed")
    require(
        all((kb["full"].get(k) == v for k, v in kb["small"].items())),
        "Small is not exact full subset",
    )
    for pool in kb:
        require(
            sum((r["kind"] != "example" for r in kb[pool].values())) == 40, "Wrong abstract count"
        )
        require(
            not any((r["meta"].get("illustrative_only") for r in kb[pool].values())),
            "Unexpected illustration",
        )
    dev = rows(CORE / "dev_inputs.jsonl")
    require(len(dev) == len({r["key"] for r in dev}) == 1295, "Unexpected DEV roster")
    require(
        all((set(r) == {"key", "item_id", "lang", "scopes", "text"} for r in dev)),
        "Unexpected gold field in inputs",
    )
    subsets = {
        s: [r["key"] for r in dev if any((sc["subset"] == s for sc in r["scopes"]))] for s in SCOPES
    }
    require(
        {s: len(keys) for s, keys in subsets.items()} == core["scope_counts"],
        "Scope counts changed",
    )
    require(
        {s: len(keys) for s, keys in subsets.items()} == {s: v[2] for s, v in SCOPES.items()},
        "Unexpected endpoint size",
    )
    hashes = {rel(p): digest(p) for p in paths}
    contract = {
        "status": "descriptive_offline_census_contract",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "question": "What annotation and source composition is available in each pool, and actually delivered in its five saved examples?",
        "unit": "Unique TRAIN records for pool inventory; repeated retrieved slots and paired distinct DEV inputs for delivery. No decoder repeat multiplication.",
        "annotation_policy": "Missing/None is unknown; [] is observed empty; false is observed non-hate. No inferred fine labels. Legal remains independent of gate.",
        "scope": SCOPES,
        "fields": FIELDS,
        "labels": LABELS,
        "input_hashes": hashes,
        "preexisting_results": "This census is post-hoc to previously inspected DEV results, not a confirmatory accuracy experiment. No evaluation gold or predictions read.",
        "inference": "Complete fixed-roster counts; no significance tests or causal interpretation. Pool size, source/class/annotation composition and example identity co-vary.",
        "next": "Use inventory to define separate controlled training-pool experiments; no condition is promoted and no live execution is authorized by this file.",
        "no_live_API": True,
        "no_retrieval": True,
        "no_CUDA": True,
    }
    save_new(BASE / "analysis_plan.json", contract)
    pool_stats = {}
    delivery = {}
    detail_rows = []
    selected_by_pool = {}
    for pool in ("small", "full"):
        rr = list(kb[pool].values())
        pool_stats[pool] = {}
        for lang in ("en", "de"):
            ex = [r for r in rr if r["kind"] == "example" and r["lang"] == lang]
            pool_stats[pool][lang] = describe(ex)
            pool_stats[pool][lang]["by_source"] = {
                source: describe([r for r in ex if r["source"] == source])
                for source in sorted({r["source"] for r in ex})
            }
        selected_by_pool[pool] = {}
        for row in dev:
            key = row["key"]
            p = prompts[pool][key]
            ctx = p["context"]
            require(
                p["condition"] == "rag_" + pool
                and p["item_id"] == row["item_id"]
                and (p["lang"] == row["lang"]),
                "Prompt identity mismatch",
            )
            require(
                len(ctx["definitions"]) == len(ctx["guidelines"]) == 2, "Unexpected abstract budget"
            )
            require(
                len(ctx["example_groups"]) == 1
                and ctx["example_groups"][0][0] == "Labelled examples",
                "Unexpected example grouping",
            )
            hits = ctx["example_groups"][0][1]
            require(
                len(hits) == len({h["id"] for h in hits}) == 5,
                "Unexpected example budget/duplicates",
            )
            selected = [kb[pool][h["id"]] for h in hits]
            require(
                all((r["kind"] == "example" and r["lang"] == row["lang"] for r in selected)),
                "Wrong example language/kind",
            )
            for h in ctx["definitions"] + ctx["guidelines"] + hits:
                validate_hit(h, kb[pool][h["id"]])
            require(
                renderer.hit_ids(ctx) == p["retrieved"], "Saved retrieved IDs disagree with context"
            )
            require(
                renderer.render_messages(row["text"], row["lang"], ctx, "corrected")
                == p["messages"],
                "Full prompt re-render mismatch",
            )
            require(
                hashlib.sha256(encoded(p["messages"])).hexdigest() == p["messages_sha256"],
                "Message signature mismatch",
            )
            selected_by_pool[pool][key] = selected
            detail_rows.append(
                {
                    "pool": pool,
                    "key": key,
                    "scopes": [s["subset"] for s in row["scopes"]],
                    "example_ids": [r["id"] for r in selected],
                    "sources": [r["source"] for r in selected],
                    "annotations": [{f: r["meta"].get(f) for f in FIELDS} for r in selected],
                    "annotated_slots": {
                        f: sum((r["meta"].get(f) is not None for r in selected)) for f in FIELDS
                    },
                    "core_eligible_slots": {
                        f: sum((eligible(r["meta"], f) for r in selected)) for f in FIELDS
                    },
                    "gate_counts": {
                        g: sum((gate_name(r["meta"]) == g for r in selected))
                        for g in ("true", "false", "unknown")
                    },
                    "definitions": p["retrieved"]["definitions"],
                    "guidelines": p["retrieved"]["guidelines"],
                    "message_characters": sum((len(m["text"]) for m in p["messages"])),
                    "example_text_characters": sum(
                        (len(" ".join(r["text"].split())) for r in selected)
                    ),
                    "messages_sha256": p["messages_sha256"],
                }
            )
        delivery[pool] = {}
        old = read(CORE / f"retrieval_v1/{pool}/complete.json")["summary"]
        for subset, keys in subsets.items():
            ex = [r for k in keys for r in selected_by_pool[pool][k]]
            summary = describe(ex)
            summary["items"] = len(keys)
            summary["annotated_slots_per_input"] = {
                f: {
                    str(n): sum(
                        (
                            sum((r["meta"].get(f) is not None for r in selected_by_pool[pool][k]))
                            == n
                            for k in keys
                        )
                    )
                    for n in range(6)
                }
                for f in FIELDS
            }
            summary["by_source"] = {
                source: describe([r for r in ex if r["source"] == source])
                for source in sorted({r["source"] for r in ex})
            }
            require(
                summary["records_or_slots"] == old[subset]["example_slots"],
                "Old slot summary mismatch",
            )
            for f in FIELDS[1:]:
                require(
                    summary["fields"][f]["annotated"] == old[subset][f + "_annotated"],
                    "Old annotation summary mismatch",
                )
            for g in ("True", "False", "None"):
                require(
                    summary["gate"][{"True": "true", "False": "false", "None": "unknown"}[g]]
                    == old[subset].get("gate_" + g, 0),
                    "Old gate summary mismatch",
                )
            delivery[pool][subset] = summary
    paired = {}
    for subset, keys in subsets.items():
        paired[subset] = {}
        for f in FIELDS:
            deltas = [
                sum((r["meta"].get(f) is not None for r in selected_by_pool["full"][k]))
                - sum((r["meta"].get(f) is not None for r in selected_by_pool["small"][k]))
                for k in keys
            ]
            paired[subset][f] = {
                "full_minus_small_annotated_slots": sum(deltas),
                "inputs_fewer": sum((x < 0 for x in deltas)),
                "inputs_equal": deltas.count(0),
                "inputs_more": sum((x > 0 for x in deltas)),
                "difference_distribution": {str(n): deltas.count(n) for n in range(-5, 6)},
            }
        paired[subset]["identical_ordered_definitions"] = sum(
            (
                prompts["small"][k]["retrieved"]["definitions"]
                == prompts["full"][k]["retrieved"]["definitions"]
                for k in keys
            )
        )
        paired[subset]["identical_ordered_guidelines"] = sum(
            (
                prompts["small"][k]["retrieved"]["guidelines"]
                == prompts["full"][k]["retrieved"]["guidelines"]
                for k in keys
            )
        )
        paired[subset]["shared_example_slots"] = sum(
            (
                len(
                    set(prompts["small"][k]["retrieved"]["Labelled examples"])
                    & set(prompts["full"][k]["retrieved"]["Labelled examples"])
                )
                for k in keys
            )
        )
    details_path = BASE / "prompt_delivery.jsonl"
    with details_path.open("x", encoding="utf8", newline="\n") as f:
        for row in detail_rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    require(
        delivery["small"]["en_dev_eval_targets"]["fields"]["target_groups"]["annotated"] == 171,
        "Target history mismatch",
    )
    require(
        delivery["full"]["en_dev_eval_targets"]["fields"]["target_groups"]["annotated"] == 113,
        "Target history mismatch",
    )
    require(
        delivery["small"]["en_dev_eval_severity"]["fields"]["severity"]["annotated"] == 120,
        "Severity history mismatch",
    )
    require(
        delivery["full"]["en_dev_eval_severity"]["fields"]["severity"]["annotated"] == 93,
        "Severity history mismatch",
    )
    result = {
        "status": "annotation_census_complete_independent_check_pending",
        "plan_sha256": digest(BASE / "analysis_plan.json"),
        "input_hashes": hashes,
        "pool_inventory": pool_stats,
        "delivery": delivery,
        "paired": paired,
        "verified_prompts": len(detail_rows),
        "verified_example_slots": len(detail_rows) * 5,
        "unique_DEV_inputs": len(dev),
        "detail_path": rel(details_path),
        "detail_sha256": digest(details_path),
        "all_historical_prompt_summary_counts_match": True,
        "interpretation": contract["inference"],
        "actual_API_calls": 0,
        "new_retrieval": False,
        "new_accuracy_scoring": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    require(
        all((digest(ROOT / name) == sha for name, sha in hashes.items())),
        "Input changed during census",
    )
    save_new(BASE / "results.json", result)
    save_new(EVIDENCE, result)
    save_new(
        BASE / "complete.json",
        {
            "status": result["status"],
            "plan_sha256": result["plan_sha256"],
            "results_sha256": digest(BASE / "results.json"),
            "evidence": rel(EVIDENCE),
            "evidence_sha256": digest(EVIDENCE),
            "details_sha256": digest(details_path),
        },
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "prompts": len(detail_rows),
                "slots": len(detail_rows) * 5,
                "pools": {
                    p: {
                        l: {
                            "records": s["records_or_slots"],
                            "gate": s["gate"],
                            "annotated": {f: v["annotated"] for f, v in s["fields"].items()},
                        }
                        for l, s in langs.items()
                    }
                    for p, langs in pool_stats.items()
                },
                "endpoint_annotated_small_full": {
                    s: [delivery[p][s]["fields"][head]["annotated"] for p in ("small", "full")]
                    for s, (_, head, _) in SCOPES.items()
                },
                "results_sha256": digest(BASE / "results.json"),
                "elapsed_seconds": result["elapsed_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

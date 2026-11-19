from collections import Counter, defaultdict
from datetime import datetime, timezone
from itertools import zip_longest
import json
from pathlib import Path
import socket
from unittest.mock import patch
from analysis.knowledge.sampled_banks import retrieval as p
from analysis.knowledge.sampled_banks import retrieval_amendment as amendment

r = p.r
BASE = p.BASE / "review_v1"
AUDIT_SHA = "b80da2cb0135a923da957afa2fdcf5e589b105ce3a5d070a34bb479f0ff1100d"
ASSEMBLED_SHA = "6d750f21b14202fa3aca8026806cc2c65a234585f6e0f9609d627265143f4c6b"
FIELDS = ("gate", "target_groups", "hate_types", "severity", "legal")


def lines(path):
    with path.open(encoding="utf8") as f:
        for line in f:
            r.require(line.strip(), "Unexpected blank JSONL row")
            yield json.loads(line)


def example_ids(entry):
    return [h["id"] for h in entry["context"]["example_groups"][0][1]]


def trace_difference(first, other):
    r.require(
        len(first["trace"]) == len(other["trace"]) == 10, "Unexpected trace shape"
    )
    changed = [
        i
        for i, (left, right) in enumerate(zip(first["trace"], other["trace"]))
        if r.js(left) != r.js(right)
    ]
    r.require(changed, "Changed entry without a changed trace")
    details = []
    for i in changed:
        left = first["trace"][i]
        right = other["trace"][i]
        r.require(left["operation"] == right["operation"], "Operation ordering changed")
        old = [hit["id"] for hit in left.get("hits", [])]
        new = [hit["id"] for hit in right.get("hits", [])]
        details.append(
            {
                "position": i,
                "operation": left["operation"],
                "first_ids": old,
                "replay_ids": new,
                "removed": [v for v in old if v not in new],
                "added": [v for v in new if v not in old],
            }
        )
    before, after = (example_ids(first), example_ids(other))
    return {
        "earliest_position": changed[0],
        "earliest_operation": first["trace"][changed[0]]["operation"],
        "events": details,
        "original_example_ids": before,
        "replay_example_ids": after,
        "removed_examples": [v for v in before if v not in after],
        "added_examples": [v for v in after if v not in before],
        "unchanged_example_count": len(set(before) & set(after)),
    }


def source_context_check(entry, records, ref, row):
    r.require(
        all((r.js(entry[k]) == r.js(v) for k, v in row.items())),
        "Input identity/text/scope mismatch",
    )
    context = entry["context"]
    for kind in ("definitions", "guidelines"):
        r.require(
            r.js(context[kind]) == r.js(ref["context"][kind])
            and len(context[kind]) == 2,
            "Abstract control changed",
        )
    r.require(
        len(context["example_groups"]) == 1
        and context["example_groups"][0][0] == "Labelled examples",
        "Example grouping changed",
    )
    hits = context["example_groups"][0][1]
    r.require(
        len(hits) == len({h["id"] for h in hits}) == 5, "Example count/duplicates"
    )
    for hit in hits:
        rec = records[hit["id"]]
        r.require(
            rec["kind"] == "example"
            and rec["lang"] == row["lang"]
            and (hit["text"] == rec["text"]),
            "Wrong example text/language",
        )
        expected_meta = {
            **{k: rec[k] for k in ("kind", "lang", "source", "dimension", "label")},
            **rec["meta"],
        }
        supplied_meta = {k: v for k, v in hit["meta"].items() if k != "_ranks"}
        r.require(
            r.js(supplied_meta) == r.js(expected_meta), "Example annotations changed"
        )
    r.require(
        r.signature(context) == entry["context_sha256"]
        and r.signature(entry["messages"]) == entry["messages_sha256"],
        "Payload signatures differ",
    )
    r.require(
        entry["messages"][-1]["text"].endswith(
            "## Text to classify (language: " + row["lang"] + ")\n" + row["text"]
        ),
        "Wrong classifier input",
    )


def main():
    r.require(
        not (BASE / "analysis.json").exists(),
        "Preserve completed review; do not overwrite",
    )
    plan, amend = amendment.validate()
    audit_path = p.BASE / "full_audit_v2.json"
    assembly_path = p.BASE / "assembly.json"
    payload_path = p.BASE / "prompts.jsonl"
    r.require(
        r.sha(audit_path) == AUDIT_SHA and r.sha(payload_path) == ASSEMBLED_SHA,
        "Completed run differs from inspected snapshot",
    )
    audit = r.read(audit_path)
    assembly = r.read(assembly_path)
    r.require(
        audit["status"] == "full_pool_retrieval_checked_Astra_review_pending_not_API"
        and audit["mode"] == "full"
        and (audit["plan_sha256"] == r.sha(p.PLAN))
        and (audit["audit_amendment_sha256"] == r.sha(amendment.AMENDMENT)),
        "Wrong audit identity",
    )
    expected_checks = {
        "bm25_query": 69930,
        "dense_query": 69930,
        "entries": 23310,
        "index_records": 74520,
        "indexes": 18,
        "inherited_reference_payloads": 2590,
        "mmr": 23310,
        "rendered_example_label_blocks": 116550,
        "rendered_prompts": 23310,
        "rrf": 69930,
    }
    r.require(
        audit["checks"] == expected_checks and assembly["entries"] == 14245,
        "Audit matrix incomplete",
    )
    hashes = {
        **audit["hashes"],
        **assembly["hashes"],
        r.rel(audit_path): AUDIT_SHA,
        r.rel(assembly_path): r.sha(assembly_path),
        r.rel(Path(__file__)): r.sha(Path(__file__)),
    }
    r.verify_hashes(hashes)
    r.require(
        not audit["actual_API_calls"]
        and (not audit["CUDA_used"])
        and (not audit["gold_scoring"]),
        "Unexpected model/scoring scope",
    )
    design = r.read(p.STUDY / "scientific_plan.json")
    dev = r.rows(p.CORE / "dev_inputs.jsonl")
    keys = [row["key"] for row in dev]
    refs = {
        condition: {row["key"]: row for row in lines(p.ROOT / path)}
        for condition, path in plan["reference_prompts"].items()
    }
    source_full = {rec["id"]: rec for rec in lines(p.CORE / "kb/full/records.jsonl")}
    source_small = {rec["id"]: rec for rec in lines(p.CORE / "kb/small/records.jsonl")}
    pairs = {}
    differences = []
    counters = Counter()
    by_pool = []
    worker_seconds = 0.0
    starts = []
    ends = []
    audit_pools = {row["condition"]: row for row in audit["by_pool"]}
    for job in plan["jobs"]:
        receipts = {}
        for kind in ("prepare", "replay"):
            done = p.receipt(job["pool_id"], kind)
            r.require(done is not None, "Missing complete worker")
            attempts = list(
                p.receipt_path(job["pool_id"], kind).parent.glob("attempt_*")
            )
            r.require(
                len(attempts) == 1
                and attempts[0].resolve() == (p.ROOT / done["attempt"]).resolve(),
                "Partial/additional attempt present",
            )
            r.require(
                done["keys"] == keys
                and done["entries"] == 1295
                and (done["condition"] == job["condition"]),
                "Worker roster differs",
            )
            source_kind = "smoke" if kind == "prepare" else "prepare"
            source_receipt = p.receipt_path(job["pool_id"], source_kind)
            r.require(
                done["source_receipt"] == r.rel(source_receipt)
                and done["source_receipt_sha256"] == r.sha(source_receipt),
                "Index-copy lineage changed",
            )
            r.require(
                not done["actual_API_calls"] and (not done["CUDA_used"]),
                "Unexpected worker model scope",
            )
            start = r.read(p.ROOT / done["attempt"] / "start.json")
            starts.append(datetime.fromisoformat(start["created_utc"]))
            ends.append(datetime.fromisoformat(done["sealed_utc"]))
            worker_seconds += done["child_elapsed_seconds"]
            receipts[kind] = done
        a, b = (receipts["prepare"], receipts["replay"])
        per = Counter()
        scope_changed = Counter()
        before = lines(p.ROOT / a["prompts"])
        after = lines(p.ROOT / b["output"])
        for i, (first, comparison) in enumerate(zip_longest(before, after)):
            r.require(
                first is not None
                and comparison is not None
                and (first["key"] == comparison["key"] == keys[i]),
                "Replay roster/order differs",
            )
            flags = comparison["matches"]
            r.require(
                set(flags) == set(p.COMPARE_FIELDS) | {"entry"}
                and all((type(v) is bool for v in flags.values())),
                "Bad replay flags",
            )
            if flags["entry"]:
                r.require(
                    all(flags.values()) and "actual_entry" not in comparison,
                    "Equal row hides an alternative",
                )
            else:
                alternative = comparison["actual_entry"]
                expected = {
                    field: r.js(first[field]) == r.js(alternative[field])
                    for field in p.COMPARE_FIELDS
                }
                expected["entry"] = r.js(first) == r.js(alternative)
                r.require(flags == expected, "Incorrect replay difference flags")
                r.require(
                    first["key"] == alternative["key"]
                    and first["condition"]
                    == alternative["condition"]
                    == job["condition"],
                    "Alternative identity changed",
                )
                detail = trace_difference(first, alternative)
                r.require(
                    detail["earliest_position"] == 6
                    and detail["earliest_operation"] == "dense_query",
                    "Unreviewed earliest change",
                )
                item = {
                    "key": first["key"],
                    "condition": job["condition"],
                    "scopes": [s["subset"] for s in first["scopes"]],
                    "message_changed": not flags["messages"],
                    "matches": flags,
                    **detail,
                    "original_messages_sha256": first["messages_sha256"],
                    "replay_messages_sha256": alternative["messages_sha256"],
                    "original_path": a["prompts"],
                    "replay_path": b["output"],
                }
                differences.append(item)
                if not flags["messages"]:
                    scope_changed.update(item["scopes"])
            per.update({k: int(not v) for k, v in flags.items()})
        r.require(
            i + 1 == 1295
            and dict(per)
            == b["replay_differences"]
            == audit_pools[job["condition"]]["replay_differences"],
            "Replay totals disagree",
        )
        counters.update(per)
        pairs[job["condition"]] = receipts
        by_pool.append(
            {
                "condition": job["condition"],
                "records": job["count"],
                "first_entries": a["entries"],
                "replay_entries": b["entries"],
                "differences": dict(per),
                "message_changes_by_scope": dict(scope_changed),
                "child_seconds": a["child_elapsed_seconds"]
                + b["child_elapsed_seconds"],
                "retrieval_seconds": a["retrieval_seconds"] + b["retrieval_seconds"],
            }
        )
    r.require(
        dict(counters) == audit["replay_differences"]
        and counters["messages"] == 13
        and (counters["entry"] == 62),
        "Unexpected reviewed difference pattern",
    )
    message_changes = [row for row in differences if row["message_changed"]]
    r.require(
        all(
            (
                1 <= len(x["added_examples"]) == len(x["removed_examples"]) <= 2
                for x in message_changes
            )
        ),
        "Unreviewed prompt alteration",
    )
    delivered = defaultdict(Counter)
    roster = Counter()
    identities = set()
    fingerprints = {}
    annotated_sources = defaultdict(set)
    inventory = []
    raw_abstract_changes = Counter()
    per_input_empty = defaultdict(Counter)
    assembled = iter(lines(payload_path))
    conditions = list(plan["reference_prompts"]) + [
        j["condition"] for j in plan["jobs"]
    ]
    for condition in conditions:
        if condition in refs:
            original = [refs[condition][row["key"]] for row in dev]
            known = source_small if condition == "curated_small" else source_full
        else:
            job = next((j for j in plan["jobs"] if j["condition"] == condition))
            known = {rec["id"]: rec for rec in lines(p.ROOT / job["records"])}
            original = lines(p.ROOT / pairs[condition]["prepare"]["prompts"])
        for i, previous in enumerate(original):
            entry = next(assembled)
            row = dev[i]
            r.require(
                entry["condition"] == condition
                and entry["key"] == previous["key"] == row["key"],
                "Assembly order/identity changed",
            )
            if condition in refs:
                for key in ("messages", "context", "retrieved", "messages_sha256"):
                    r.require(
                        r.js(entry[key]) == r.js(previous[key]),
                        "Reference payload changed: " + key,
                    )
            else:
                r.require(
                    r.js(entry) == r.js(previous),
                    "Assembly replaced original first-pass entry",
                )
                for bucket in ("definitions", "guidelines"):
                    raw_abstract_changes[bucket] += (
                        entry["raw_context"][bucket] != entry["context"][bucket]
                    )
            source_context_check(entry, known, refs["full_reference"][row["key"]], row)
            identity = (condition, entry["key"])
            r.require(identity not in identities, "Duplicate matrix entry")
            identities.add(identity)
            inventory.append(
                {
                    "condition": condition,
                    "key": entry["key"],
                    "messages_sha256": entry["messages_sha256"],
                    "context_sha256": entry["context_sha256"],
                    "entry_sha256": r.signature(entry),
                }
            )
            fingerprints[identity] = {
                "messages": entry["messages_sha256"],
                "ids": example_ids(entry),
            }
            hits = entry["context"]["example_groups"][0][1]
            for scope in row["scopes"]:
                subset = scope["subset"]
                tag = subset + "/" + condition
                bucket = delivered[tag]
                roster[tag] += 1
                bucket["inputs"] += 1
                bucket["example_slots"] += len(hits)
                target_field = design["evaluation"]["all_rosters"][subset]["field"]
                count = sum((hit["meta"].get(target_field) is not None for hit in hits))
                per_input_empty[tag][str(count)] += 1
                for hit in hits:
                    for field in FIELDS:
                        value = hit["meta"].get(field)
                        bucket[field + "_annotated"] += value is not None
                        bucket[field + "_empty_lists"] += type(value) is list and (
                            not value
                        )
                    gate = hit["meta"].get("gate")
                    bucket[
                        (
                            "gate_true"
                            if gate is True
                            else "gate_false" if gate is False else "gate_unknown"
                        )
                    ] += 1
                    annotated_sources[tag].add(hit["id"])
            r.require(entry["plan_sha256"] == r.sha(p.PLAN), "Wrong assembly plan")
        r.require(i + 1 == 1295, "Condition roster incomplete")
        print("Reviewed first-pass payloads: " + condition + " 1295/1295", flush=True)
    r.require(
        next(assembled, None) is None and len(identities) == len(inventory) == 14245,
        "Unexpected final matrix size",
    )
    for subset, definition in design["evaluation"]["all_rosters"].items():
        for condition in conditions:
            tag = subset + "/" + condition
            r.require(roster[tag] == definition["items"], "DEV endpoint count changed")
            if condition not in refs:
                independent_counts = audit["delivered_annotation_counts"][tag]
                r.require(
                    all(
                        (delivered[tag][k] == v for k, v in independent_counts.items())
                    ),
                    "Annotation audit mismatch",
                )
    overlap = []
    for condition in conditions:
        if condition in refs:
            continue
        for reference in refs:
            same_messages = 0
            same_order = 0
            shared = 0
            for row in dev:
                a = fingerprints[condition, row["key"]]
                b = fingerprints[reference, row["key"]]
                same_messages += a["messages"] == b["messages"]
                same_order += a["ids"] == b["ids"]
                shared += len(set(a["ids"]) & set(b["ids"]))
            overlap.append(
                {
                    "condition": condition,
                    "reference": reference,
                    "identical_messages": same_messages,
                    "identical_ordered_examples": same_order,
                    "total_shared_example_slots": shared,
                    "mean_shared_examples": shared / 1295,
                    "inputs": 1295,
                }
            )
    for row in audit["by_pool"]:
        for check in row["checks"]:
            db = check["database"]
            r.require(
                db["integrity"] == "ok" and db["actual_orphans"] == 0,
                "Unresolved database integrity finding",
            )
            expected = (
                "empty_graph_capacity_bytes_qualified_complete_WAL_verified"
                if "compact" in row["condition"]
                else "all_persisted_graph_files_exact"
            )
            r.require(
                db["persistence_review"]["policy"] == expected,
                "Unreviewed persistence policy",
            )
    audit_logs = sorted((p.BASE / "logs").glob("run_audit_v2_*.txt"))
    r.require(
        len(audit_logs) == 1
        and "full_pool_retrieval_checked_Astra_review_pending_not_API"
        in audit_logs[0].read_text(),
        "Missing full audit completion log",
    )
    hashes[r.rel(audit_logs[0])] = r.sha(audit_logs[0])
    p.save_rows(BASE / "replay_differences.jsonl", differences)
    p.save_rows(BASE / "prompt_inventory.jsonl", inventory)
    changes_by_scope = Counter(
        (subset for row in message_changes for subset in row["scopes"])
    )
    aggregate = {
        "changed_entries": len(differences),
        "changed_messages": len(message_changes),
        "trace_only_entries": len(differences) - len(message_changes),
        "sampled_prompts": 11655,
        "exact_message_match_percent": (11655 - len(message_changes)) / 11655 * 100,
        "distinct_changed_message_inputs": len({row["key"] for row in message_changes}),
        "changes_by_scope": dict(changes_by_scope),
        "changed_message_fraction_within_5000_pools": 13 / (1295 * 3),
        "original_first_pass_preserved_for_all_entries": True,
    }
    report = {
        "status": "full_pool_retrieval_reviewed_first_pass_accepted_for_API_preparation_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "full_audit_sha256": AUDIT_SHA,
        "plan_sha256": r.sha(p.PLAN),
        "audit_amendment_sha256": r.sha(amendment.AMENDMENT),
        "source_hashes_verified": len(audit["hashes"]),
        "completed_independent_audit_checks": audit["checks"],
        "review_checks": {
            "full_worker_receipts": 18,
            "complete_attempts_per_worker": 1,
            "streamed_first_pass_payloads": len(inventory),
            "exact_inherited_reference_payloads": 2590,
            "original_sampled_entries_retained": 11655,
            "changed_alternatives_inspected": len(differences),
            "conditions": len(conditions),
            "endpoints_per_condition": 7,
            "raw_abstract_context_differences": dict(raw_abstract_changes),
        },
        "replay": aggregate,
        "by_pool": by_pool,
        "delivered_annotation_counts": {k: dict(v) for k, v in delivered.items()},
        "task_annotation_count_per_input": {
            k: dict(v) for k, v in per_input_empty.items()
        },
        "unique_retrieved_examples": {k: len(v) for k, v in annotated_sources.items()},
        "overlap_with_references": overlap,
        "timing": {
            "worker_child_seconds_sum": worker_seconds,
            "first_worker_start": min(starts).isoformat(),
            "last_worker_seal": max(ends).isoformat(),
            "audit_completed": audit["created_utc"],
            "elapsed_first_worker_to_audit_seconds": (
                datetime.fromisoformat(audit["created_utc"]) - min(starts)
            ).total_seconds(),
        },
        "maximum_stored_vector_error": max(
            (
                c["index_vector_max_error"]
                for row in audit["by_pool"]
                for c in row["checks"]
            )
        ),
        "maximum_dense_score_error": max(
            (
                c["max_dense_score_error"]
                for row in audit["by_pool"]
                for c in row["checks"]
            )
        ),
        "minimum_MMR_margin": min(
            (c["minimum_MMR_margin"] for row in audit["by_pool"] for c in row["checks"])
        ),
        "decision": "Retain every original first-pass prompt, including all 13 changed-message cases. No drop, substitute, retry-to-match or retrieval tuning. Qualified replay acceptance only; separately freeze and test API/scoring/schedule before user model runs.",
        "limitations": [
            "Observed approximate candidates are not certified exact nearest neighbours; precise internal nondeterminism cause is not established.",
            "Identical replay flags are recorded by the frozen worker; unchanged second-pass payloads are not stored as an independent complete second file.",
            "Compact graph persistence exception remains narrow and documented; physical identity of populated graph files does not prove in-memory graph identity.",
            "Supplied TRAIN annotations count slots, not new labels or classifier accuracy. No annotation imputation.",
            "DEV was already inspected before this study design; fixed pool draws/decoder repeats are not independent new evaluation samples.",
        ],
        "actual_API_calls": 0,
        "CUDA_used": False,
        "gold_scoring": False,
        "retrieval_rerun": False,
        "hashes": {
            **hashes,
            r.rel(BASE / "replay_differences.jsonl"): r.sha(
                BASE / "replay_differences.jsonl"
            ),
            r.rel(BASE / "prompt_inventory.jsonl"): r.sha(
                BASE / "prompt_inventory.jsonl"
            ),
        },
    }
    r.verify_hashes(report["hashes"])
    p.save(BASE / "analysis.json", report)
    p.save(
        BASE / "retrieval_acceptance.json",
        {
            "status": "accepted_original_first_pass_prompts_with_documented_replay_variation",
            "analysis": r.rel(BASE / "analysis.json"),
            "analysis_sha256": r.sha(BASE / "analysis.json"),
            "prompts": r.rel(payload_path),
            "prompts_sha256": ASSEMBLED_SHA,
            "entries": 14245,
            "next": "Separate frozen four-stage API/scoring preparation. No Qwen runner ready yet.",
            "change_original_retrieval": False,
            "new_accuracy_result": False,
        },
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "replay": aggregate,
                "timing": report["timing"],
                "analysis_sha256": r.sha(BASE / "analysis.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", p.no_network), patch.object(
        socket, "create_connection", p.no_network
    ):
        main()

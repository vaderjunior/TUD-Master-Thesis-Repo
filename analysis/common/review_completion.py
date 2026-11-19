from src.hsrag.paths import PROJECT_ROOT
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

ROOT = PROJECT_ROOT
CORE = ROOT / "experiments/astra/full_kb_core_20260915_v1"
BASE = CORE / "api_execution_v1"
OUT = ROOT / "Thesis_Log/astra_result/evidence"
PUBLICATION = OUT / "core_sensitivity_publication_diagnostics_20260920_v1.json"
SUMMARY = OUT / "core_completion_summary_20260920_v1.json"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def storage(path):
    return Path("\\\\?\\" + str(Path(path).resolve()))


def read(path):
    return json.loads(storage(path).read_text(encoding="utf8"))


def rows(path):
    return [
        json.loads(line)
        for line in storage(path).read_text(encoding="utf8").splitlines()
        if line
    ]


def digest(path):
    return hashlib.sha256(storage(path).read_bytes()).hexdigest()


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def bind(path, hashes, expected=None):
    value = digest(path)
    require(expected is None or value == expected, "Hash mismatch: " + rel(path))
    hashes[rel(path)] = value
    return value


def save(path, value):
    with storage(path).open("x", encoding="utf8", newline="\n") as handle:
        json.dump(
            value, handle, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        handle.write("\n")


def publication():
    dest = CORE / "secondary_results_v4/en_dev_eval_types"
    report, plan, receipt = (
        read(dest / name) for name in ("results.json", "plan.json", "complete.json")
    )
    hashes = {}
    bind(dest / "complete.json", hashes)
    bind(dest / "results.json", hashes, receipt["results_sha256"])
    bind(dest / "plan.json", hashes, receipt["plan_sha256"])
    bind(ROOT / receipt["evidence"], hashes, receipt["evidence_sha256"])
    require(report["plan_sha256"] == receipt["plan_sha256"], "Plan binding")
    require(receipt["assembled"] == report["assembled"], "Publication manifest differs")
    for name, expected in report["verified_input_hashes"].items():
        bind(ROOT / name, hashes, expected)
    endpoint = report["endpoint"]
    ids = endpoint["scorable_ids"]
    require(
        len(ids)
        == len(set(ids))
        == endpoint["rows"]
        == endpoint["scorable_rows"]
        == 103,
        "Roster",
    )
    require(
        endpoint["role"] == "sensitivity"
        and endpoint["mask"] == "strict gold gate true and dimension annotated",
        "Scoring scope",
    )
    require(
        endpoint["macro_labels"] == endpoint["all_labels"]
        and min(endpoint["gold_support"].values()) >= 10,
        "Fixed label support",
    )
    dev = rows(CORE / "dev_inputs.jsonl")
    selected = [
        r for r in dev if any(s["subset"] == "en_dev_eval_types" for s in r["scopes"])
    ]
    require({r["item_id"] for r in selected} == set(ids), "Prepared roster")
    require(
        all(
            r["scopes"]
            == [{"subset": "en_dev_eval_types", "dimension": "hate_type_secondary"}]
            for r in selected
        ),
        "Exact scope tag",
    )
    main = read(CORE / "secondary_results_v4/en_dev_eval_sq3_types/results.json")
    require(
        set(ids).isdisjoint(main["endpoint"]["scorable_ids"]),
        "Type rosters overlap by ID",
    )
    cache, kinds, nulls, runs, total = {}, Counter(), [], {}, 0
    for name, metadata in report["assembled"].items():
        bind(ROOT / name, hashes, metadata["sha256"])
        exported = rows(ROOT / name)
        require(
            [r["id"] for r in exported] == ids and len(exported) == metadata["rows"],
            "Exported roster",
        )
        condition, repeat = exported[0]["condition"], exported[0]["replicate"]
        require((condition, repeat) not in runs, "Duplicate setting")
        counts = Counter()
        for row in exported:
            require(
                (row["condition"], row["replicate"]) == (condition, repeat),
                "Exported setting",
            )
            require(row["key"] == "en/" + row["id"], "Origin key")
            source = row["source"]
            if source not in cache:
                bind(ROOT / source, hashes, row["source_sha256"])
                cache[source] = (
                    read(ROOT / source)
                    if row["kind"] == "qwen_decoder_repeat"
                    else {r["key"]: r for r in rows(ROOT / source)}
                )
            require(hashes[source] == row["source_sha256"], "Origin binding")
            original = cache[source]
            if row["kind"] == "qwen_decoder_repeat":
                require(
                    report["verified_item_hashes"][source] == row["source_sha256"],
                    "Audited Qwen origin",
                )
                n_valid = original["n_valid"]
            else:
                require(
                    row["kind"] == "encoder_training_seed"
                    and plan["hashes"][source] == row["source_sha256"],
                    "Encoder origin",
                )
                original, n_valid = original[row["key"]], 1
            require(
                row["result"] == original["result"] and row["n_valid"] == n_valid,
                "Export/source mismatch",
            )
            kinds[row["kind"]] += 1
            result = row["result"]
            if result is None:
                nulls.append(
                    {"id": row["id"], "condition": condition, "repeat": repeat}
                )
                continue
            require(type(result["hate"]) is bool, "Strict gate")
            require(result["hate"] or not result["hate_type"], "Type gate suppression")
            counts["missed_hate"] += not result["hate"]
            counts["any_type_assigned"] += bool(result["hate_type"])
            counts["predicted_type_labels"] += len(result["hate_type"])
            counts["hate_true_but_no_type"] += (
                result["hate"] and not result["hate_type"]
            )
        expected = report["conditions"][condition]["runs"][repeat]
        require(
            expected["replicate"] == repeat
            and counts["missed_hate"]
            == expected["score"]["missed_hate_on_gold_positive_scorable_inputs"],
            "Gate diagnostic",
        )
        runs[condition, repeat] = {
            "replicate": repeat,
            "texts": len(exported),
            **dict(counts),
        }
        total += len(exported)
    require(
        not nulls and total == 3399 and len(runs) == 33, "Final sensitivity coverage"
    )
    require(
        dict(kinds) == {"encoder_training_seed": 618, "qwen_decoder_repeat": 2781},
        "Row kinds",
    )
    for contrast in report["contrasts"].values():
        require(
            contrast["common_valid_ids"] == ids and contrast["n_common_valid"] == 103,
            "Common mask",
        )
        require(
            contrast["family_ci"] is None
            and contrast["independent_frequency_weighted_bootstrap_verified"],
            "Sensitivity uncertainty",
        )
    diagnostics = {}
    for condition in sorted(report["conditions"]):
        three = [runs[condition, repeat] for repeat in range(3)]
        diagnostics[condition] = {
            "runs": three,
            "means": {
                key: statistics.mean(r[key] for r in three)
                for key in (
                    "missed_hate",
                    "any_type_assigned",
                    "predicted_type_labels",
                    "hate_true_but_no_type",
                )
            },
        }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete_sensitivity_publication_verified",
        "source_sha256": digest(__file__),
        "subset": endpoint["subset"],
        "results_sha256": receipt["results_sha256"],
        "plan_sha256": receipt["plan_sha256"],
        "files": len(runs),
        "rows": total,
        "row_kinds": dict(kinds),
        "nulls": nulls,
        "common_valid_texts": 103,
        "all_complete_ordered_rosters_and_origin_hash_bindings_match": True,
        "every_exported_prediction_rechecked_against_source": True,
        "all_six_contrasts_share_same_common_mask": True,
        "exact_prepared_scope_checked": "hate_type_secondary",
        "main_and_sensitivity_rosters_disjoint_by_id": True,
        "gate_and_label_output_diagnostics_on_common_mask": diagnostics,
        "verified_publication_hashes": hashes,
        "actual_API_calls": 0,
        "new_gold_scoring": False,
        "limits": [
            "Diagnostics use the frozen all-gold-hateful scorable mask; no gate false positives can be measured.",
            "Disjoint IDs do not establish absence of shared text or independent selection; these remain DEV sensitivity results.",
        ],
    }


def completion():
    hashes = {}
    schedule = read(BASE / "schedule.json")
    bind(BASE / "schedule.json", hashes)
    plan_hash = bind(BASE / "plan.json", hashes)
    counts, accounting, usage, observed_errors = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    stage_rows, identities, keys, receipts, nulls = [], set(), set(), set(), []
    per_subset, administrative_unknown = Counter(), 0
    for stage in schedule["stages"]:
        sid = stage["id"]
        version = {
            "stage_01_primary": 1,
            "stage_02_primary": 2,
            "stage_09_secondary": 4,
        }.get(sid, 3)
        receipt_path = BASE / "audits" / f"{sid}_v{version}" / "complete.json"
        receipt = read(receipt_path)
        bind(receipt_path, hashes)
        evidence = read(ROOT / receipt["evidence"])
        bind(ROOT / receipt["evidence"], hashes, receipt["evidence_sha256"])
        complete_path = BASE / "stages" / sid / "complete.json"
        bind(complete_path, hashes, receipt["stage_complete_sha256"])
        complete = read(complete_path)
        require(receipt["stage"] == evidence["stage"] == sid, "Stage identity")
        require(
            receipt["plan_sha256"]
            == evidence["plan_sha256"]
            == complete["plan_sha256"]
            == plan_hash,
            "Execution binding",
        )
        require(
            evidence["counts"]["items"]
            == stage["items"]
            == len(stage["jobs"])
            == len(complete["item_receipt_hashes"]),
            "Stage coverage",
        )
        require(
            evidence["accounting"]["ordinary_logical_calls"]
            == stage["ordinary_responses"],
            "Ordinary calls",
        )
        for key, value in complete["summary"]["completed_item_accounting"].items():
            require(evidence["accounting"][key] == value, "Stage accounting")
        require(
            not (receipts & complete["item_receipt_hashes"].keys()),
            "Repeated stage item",
        )
        receipts.update(complete["item_receipt_hashes"])
        for job in stage["jobs"]:
            ident = (job["subset"], job["key"], job["condition"], job["repeat"])
            require(ident not in identities, "Duplicate scheduled identity")
            identities.add(ident)
            keys.add(job["key"])
            per_subset[job["subset"]] += 1
        require(
            evidence["actual_audit_API_calls"]
            == evidence["completed_stage_restart_new_POSTs"]
            == 0,
            "Audit made live calls",
        )
        counts.update(evidence["counts"])
        accounting.update(evidence["accounting"])
        usage.update(evidence["observed_usage_sums"])
        for field, value in evidence["observed_usage_field_counts"].items():
            require(
                value == evidence["accounting"]["http_success_with_text"],
                "Observed usage coverage: " + field,
            )
        errors = dict(
            evidence.get(
                "transport_failure_classes",
                evidence.get(
                    "retry_trigger_classes_including_administrative_recovery", {}
                ),
            )
        )
        unknown = errors.pop("LocalProcessInterruptedOutcomeUnknown", 0)
        administrative_unknown += unknown
        observed_errors.update(errors)
        stage_nulls = evidence.get("null_prediction_details", [])
        require(
            len(stage_nulls) == evidence["counts"]["null_items"], "Null detail count"
        )
        nulls.extend({"stage": sid, **row} for row in stage_nulls)
        stage_rows.append(
            {
                "stage": sid,
                "predictions": stage["items"],
                "nulls": len(stage_nulls),
                "accounting": evidence["accounting"],
                "observed_usage": evidence["observed_usage_sums"],
                "evidence": receipt["evidence"],
            }
        )
    require(
        len(stage_rows) == 14
        and len(identities) == len(receipts) == counts["items"] == 32274
        and len(keys) == 1295,
        "Core coverage",
    )
    require(len(nulls) == counts["null_items"] == 19, "Total nulls")
    require(
        accounting["post_attempts"]
        == accounting["ordinary_logical_calls"]
        + accounting["repair_logical_calls"]
        + accounting["retry_attempts"],
        "Call arithmetic",
    )
    require(
        sum(observed_errors.values()) + administrative_unknown
        == accounting["retry_attempts"],
        "Retry arithmetic",
    )
    require(administrative_unknown == 8, "Recovery record")
    destinations = [
        "primary_results_v1",
        "secondary_results_v2/en_dev_eval_targets",
        "severity_results_v1",
        "secondary_results_v3/de_dev_eval",
        "secondary_results_v4/en_dev_eval_sq3_types",
        "secondary_results_v4/en_dev_eval_types",
    ]
    endpoints = {}
    for destination in destinations:
        dest = CORE / destination
        receipt, report = read(dest / "complete.json"), read(dest / "results.json")
        bind(dest / "complete.json", hashes)
        bind(dest / "results.json", hashes, receipt["results_sha256"])
        bind(dest / "plan.json", hashes, receipt["plan_sha256"])
        bind(ROOT / receipt["evidence"], hashes, receipt["evidence_sha256"])
        require(
            report["plan_sha256"] == receipt["plan_sha256"]
            and report["actual_API_calls"] == 0,
            "Score binding",
        )
        reports = (
            report["endpoints"]
            if "endpoints" in report
            else {report["endpoint"]["subset"]: report}
        )
        for name, endpoint_report in reports.items():
            endpoint = endpoint_report["endpoint"]
            conditions = endpoint_report["conditions"]
            null_count = sum(
                c["invalid_predictions"]
                for name, c in conditions.items()
                if not name.startswith("encoder_")
            )
            require(
                null_count == sum(x["job"]["subset"] == name for x in nulls),
                "Endpoint/audit null mismatch",
            )
            endpoints[name] = {
                "endpoint": {k: v for k, v in endpoint.items() if k != "scorable_ids"},
                "qwen_predictions": per_subset[name],
                "nulls": null_count,
                "conditions": {n: c["summary"] for n, c in conditions.items()},
                "fewshot_groups": {
                    n: g["across_set_means"]
                    for n, g in endpoint_report["fewshot_groups"].items()
                },
                "full_minus_small": {
                    k: v
                    for k, v in endpoint_report["contrasts"][
                        "rag_full_minus_rag_small"
                    ].items()
                    if k
                    not in (
                        "common_valid_ids",
                        "per_setting_f1",
                        "excluded_from_common_valid",
                    )
                },
                "source_results": rel(dest / "results.json"),
                "source_sha256": receipt["results_sha256"],
            }
    require(
        set(endpoints) == set(per_subset) and len(endpoints) == 7,
        "Missing scored endpoint",
    )
    require(
        sum(e["endpoint"]["rows"] for e in endpoints.values()) == len(keys),
        "Endpoint roster total",
    )
    for path, expected in hashes.items():
        require(digest(ROOT / path) == expected, "Input changed during review")
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "all_14_stages_audited_all_7_core_endpoints_scored",
        "source_sha256": digest(__file__),
        "verified_summary_input_hashes": hashes,
        "stages": stage_rows,
        "endpoints": endpoints,
        "unique_inputs": len(keys),
        "qwen_predictions": len(identities),
        "nonnull_predictions": counts["items"] - len(nulls),
        "null_predictions": len(nulls),
        "null_details": nulls,
        "counts": dict(counts),
        "accounting": dict(accounting),
        "observed_successful_response_usage": dict(usage),
        "observed_transport_failures": dict(observed_errors),
        "administrative_unknown_outcomes": administrative_unknown,
        "actual_API_calls": 0,
        "new_gold_scoring": False,
        "limits": [
            "Receipt-bound aggregation of previously audited stages/results, not a second raw-response replay of older stages.",
            "Attempt slots include eight stage-02 administrative unknown outcomes; request delivery, completion, usage and duration are unknown.",
            "Observed token usage excludes unknown remote outcomes of transport failures/interruption, is not complete billed cost, and excludes pilot/retrieval/training.",
            "Main versus sensitivity labels and primary family-adjusted versus descriptive pointwise intervals remain separate; no pooled overall F1.",
            "100 percent applies only to this frozen core programme, not all planned full-KB research or the thesis.",
        ],
    }


if __name__ == "__main__":
    require(
        not PUBLICATION.exists() and not SUMMARY.exists(),
        "Preserve previous or partial review outputs",
    )
    p = publication()
    c = completion()
    save(PUBLICATION, p)
    save(SUMMARY, c)
    print(
        json.dumps(
            {
                "publication_sha256": digest(PUBLICATION),
                "completion_sha256": digest(SUMMARY),
                "files": p["files"],
                "rows": p["rows"],
                "predictions": c["qwen_predictions"],
                "nonnull": c["nonnull_predictions"],
                "nulls": c["null_predictions"],
                "counts": c["counts"],
                "accounting": c["accounting"],
                "observed_usage": c["observed_successful_response_usage"],
                "transport": c["observed_transport_failures"],
                "administrative_unknown_outcomes": c["administrative_unknown_outcomes"],
            },
            indent=2,
        )
    )

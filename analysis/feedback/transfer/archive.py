from __future__ import annotations
from src.hsrag.paths import PROJECT_ROOT

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
from unittest.mock import patch

ROOT = PROJECT_ROOT
STUDY = ROOT / "experiments/astra/sq3_transfer_20261012_v1"
EXECUTION = STUDY / "execution"
LIVE = EXECUTION / "live/transfer"
OVERVIEW = (
    ROOT
    / "experiments/astra/sq3_transfer_review_plan_20261023_v1/completion_overview.json"
)
OUT = ROOT / "experiments/astra/sq3_transfer_part1_20261023_v1"
PLAN_SHA = "ca08393271cfe2461966376daebe45d5d156dcf0acc38f4ce0e9f480f8a2ed5d"
FIELDS = ("hate", "hate_type", "target_group", "severity", "legal")
GOLD_FIELDS = ("hate_types", "target_groups", "severity", "legal")
LABELS = {
    "hate_type": {
        "explicit",
        "incitement",
        "inferiority",
        "irony",
        "stereotyping",
        "threatening",
        "grievance",
    },
    "target_group": {
        "race",
        "religion",
        "gender",
        "sexual_orientation",
        "disability",
        "national_origin",
        "other",
    },
    "legal": {"propaganda", "incitement_public_peace", "insult_defamation"},
}
ADMIN = "LocalProcessInterruptedOutcomeUnknown"


def need(ok, message):
    if not ok:
        raise ValueError(message)


def physical(path):
    value = str(Path(str(path).removeprefix("\\\\?\\")).resolve())
    return Path("\\\\?\\" + value) if os.name == "nt" else Path(value)


def rel(path):
    return (
        Path(str(path).removeprefix("\\\\?\\")).resolve().relative_to(ROOT).as_posix()
    )


def inside(root, name):
    path = (root / name).resolve()
    need(path.is_relative_to(root.resolve()), "Evidence path left its root")
    return path


def encoded(value, *, pretty=False):
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            indent=2 if pretty else None,
        )
        + "\n"
    ).encode("utf8")


def sha(path):
    digest = hashlib.sha256()
    with physical(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(physical(path).read_bytes())


def read_bound(path, expected):
    raw = physical(path).read_bytes()
    need(
        hashlib.sha256(raw).hexdigest() == expected, "Changed bound file: " + str(path)
    )
    return json.loads(raw)


def rows(path):
    with physical(path).open("rb") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write(path, value):
    need(
        path.resolve().is_relative_to(OUT.resolve()),
        "Write outside the new analysis folder",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with physical(path).open("xb") as stream:
        stream.write(value if isinstance(value, bytes) else encoded(value, pretty=True))


def utc():
    return datetime.now(timezone.utc).isoformat()


def blocked(*args, **kwargs):
    raise AssertionError("Offline transfer analysis: network is disabled")


@contextmanager
def offline():
    with patch.object(socket.socket, "connect", blocked), patch.object(
        socket.socket, "connect_ex", blocked
    ), patch.object(socket, "create_connection", blocked):
        yield


def identity(job):
    return {
        "experiment_sha256": PLAN_SHA,
        "condition": job["study"] + "::" + job["condition"],
        "repeat": job["repeat"],
        "key": job["key"],
    }


def item_id(job):
    return hashlib.sha256(encoded(identity(job), pretty=True)).hexdigest()


def validate_prediction(value):
    if value is None:
        return
    need(type(value["hate"]) is bool, "Nonboolean prediction gate")
    for field, permitted in LABELS.items():
        labels = value[field]
        need(
            type(labels) is list and all(type(x) is str for x in labels),
            "Invalid label list",
        )
        need(
            len(set(labels)) == len(labels) and set(labels) <= permitted,
            "Unknown or duplicate label",
        )
    need(
        value["severity"] is None or value["severity"] in ("low", "medium", "high"),
        "Unknown severity",
    )
    if value["hate"] is False:
        need(
            value["hate_type"] == []
            and value["target_group"] == []
            and value["severity"] is None,
            "Saved false gate was not normalised",
        )


def aggregate_votes(predictions):
    valid = [p for p in predictions if p is not None]
    if not valid:
        return None, False
    count = len(valid)
    yes = sum(v["hate"] for v in valid)
    final = {"hate": 2 * yes > count}
    for field in LABELS:
        occurrences = Counter(label for v in valid for label in v[field])
        final[field] = sorted(
            label for label, n in occurrences.items() if 2 * n > count
        )
    ordered = sorted(
        ("low", "medium", "high").index(v["severity"])
        for v in valid
        if v["severity"] is not None
    )
    final["severity"] = (
        ("low", "medium", "high")[ordered[(len(ordered) - 1) // 2]] if ordered else None
    )
    if not final["hate"]:
        final.update(hate_type=[], target_group=[], severity=None)
    return final, 2 * yes == count


def compact(saved, job, receipt, receipt_sha, result_sha):
    need(saved["identity"] == identity(job), "Result identity differs")
    need(
        saved["execution_mode"] == "live"
        and saved["messages_sha256"] == job["messages_sha256"],
        "Wrong execution mode or prompt binding",
    )
    votes = saved["run_predictions"]
    need(saved["n_runs"] == 3 and len(votes) == 3, "Wrong number of votes")
    need(
        type(saved["n_valid"]) is int
        and saved["n_valid"] == sum(v is not None for v in votes),
        "Valid-vote count differs",
    )
    for value in votes + [saved["result"]]:
        validate_prediction(value)
    final, tie = aggregate_votes(votes)
    need(
        final
        == (
            {k: saved["result"][k] for k in FIELDS}
            if saved["result"] is not None
            else None
        ),
        "Saved vote aggregation differs",
    )
    need(
        type(saved["uncertain"]) is bool and tie == saved["uncertain"],
        "Saved gate-tie flag differs",
    )
    usage, ordinary, repair, models, finishes, classes = (Counter() for _ in range(6))
    durations, backoffs, unknown = [], [], 0
    for event in saved["events"]:
        for field in ("elapsed_s", "backoff_elapsed_s"):
            value = event[field]
            need(
                type(value) in (int, float) and math.isfinite(value) and value >= 0,
                "Invalid event duration",
            )
        if event["response_class"] == ADMIN:
            need(
                event["elapsed_s"] == 0 and event["action"] == "retry",
                "Invalid unknown interruption record",
            )
            unknown += 1
        else:
            durations.append(event["elapsed_s"])
        backoffs.append(event["backoff_elapsed_s"])
        classes[event["response_class"]] += 1
        if event["action"] == "ok":
            metadata = event["model_response_metadata"]
            u = metadata["usage"]
            names = ("prompt_tokens", "completion_tokens", "total_tokens")
            need(
                all(type(u.get(k)) is int and u[k] >= 0 for k in names)
                and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"],
                "Invalid observed usage",
            )
            used = {k: u[k] for k in names}
            usage.update(used)
            (ordinary if event["turn"] == 0 else repair).update(used)
            models[str(metadata["reported_model"])] += 1
            finishes[str(metadata["finish_reason"])] += 1
    accounting = saved["accounting"]
    need(
        all(type(v) is int and v >= 0 for v in accounting.values()),
        "Invalid accounting",
    )
    need(
        len(durations) + unknown == accounting["post_attempts"] == len(saved["events"]),
        "Event count differs",
    )
    need(
        sum(e["action"] == "ok" for e in saved["events"])
        == accounting["http_success_with_text"],
        "Successful-event total differs",
    )
    return {
        "key": job["key"],
        "population": job["population"],
        "condition": job["condition"],
        "repeat": job["repeat"],
        "result": final,
        "n_valid": saved["n_valid"],
        "uncertain": tie,
        "votes": [{k: v[k] for k in FIELDS} if v is not None else None for v in votes],
        "messages_sha256": job["messages_sha256"],
        "item_receipt": rel(LIVE / ("items/" + item_id(job) + "/complete.json")),
        "item_receipt_sha256": receipt_sha,
        "result_sha256": result_sha,
        "item_completed_utc": receipt["created_utc"],
        "accounting": accounting,
        "observed_usage": dict(usage),
        "ordinary_usage": dict(ordinary),
        "repair_usage": dict(repair),
        "reported_models": dict(models),
        "finish_reasons": dict(finishes),
        "response_classes": dict(classes),
        "known_request_durations_s": durations,
        "known_request_seconds_sum": math.fsum(durations),
        "known_request_duration_count": len(durations),
        "retry_backoff_seconds_sum": math.fsum(backoffs),
        "administrative_unknown_requests": unknown,
        "vote_latencies_s": saved["latencies"],
        **{
            k: saved[k]
            for k in ("repairs", "parse_failures", "normalisations", "gate_normalised")
        },
    }


def read_item(job, receipts):
    name = "items/" + item_id(job)
    receipt_name = name + "/complete.json"
    digest = receipts[receipt_name]
    receipt = read_bound(LIVE / receipt_name, digest)
    need(receipt["status"] == "complete", "Incomplete saved item")
    result_name = name + "/result.json"
    result_sha = receipt["hashes"][result_name]
    saved = read_bound(LIVE / result_name, result_sha)
    return compact(saved, job, receipt, digest, result_sha)


def record_counts(records):
    counts, accounting, usage = Counter(), Counter(), Counter()
    for row in records:
        counts["predictions"] += 1
        counts["nulls"] += row["result"] is None
        counts["valid_votes_" + str(row["n_valid"])] += 1
        counts["gate_ties"] += row["uncertain"]
        for name in ("repairs", "parse_failures", "normalisations", "gate_normalised"):
            counts[name] += row[name]
        accounting.update(row["accounting"])
        usage.update(row["observed_usage"])
    return counts, accounting, usage


def fixed_support(roster):
    support = {}
    for population in ("en_test", "en_hatecheck_eval"):
        support[population] = {}
        for field in GOLD_FIELDS:
            eligible = [
                r
                for r in roster
                if r["population"] == population
                and r.get(field) is not None
                and (field == "legal" or r["gate"] is True)
            ]
            positives = Counter()
            for r in eligible:
                value = r[field]
                positives.update(value if isinstance(value, list) else [value])
            support[population][field] = {
                "rows": len(eligible),
                "positive_support": dict(positives),
                "support_ge_10": sorted(k for k, v in positives.items() if v >= 10),
            }
    return support


def matrix_row(source, input_index, records, conditions, replay_keys):
    expected = {(c, repeat) for c in conditions for repeat in range(3)}
    by_slot = {(r["condition"], r["repeat"]): r for r in records}
    need(
        len(records) == len(by_slot) == 12 and set(by_slot) == expected,
        "Missing or duplicate analysis slot",
    )
    need(
        all(
            r["key"] == source["key"] and r["population"] == source["population"]
            for r in records
        ),
        "Prediction/roster join differs",
    )
    need(
        len({r["stage"] for r in records}) == 1, "Input split across execution batches"
    )
    valid = all(r["result"] is not None for r in records)
    strict = all(r["n_valid"] == 3 for r in records)
    replay = source["key"] in replay_keys
    need(not strict or valid, "Strict mask admits a null")

    return {
        "input_index": input_index,
        "key": source["key"],
        "population": source["population"],
        "source": source["source"],
        "source_file": source["source_file"],
        **{
            k: source[k]
            for k in (
                "inference_cluster",
                "functional_cluster",
                "text_cluster",
                "overlap_cluster",
            )
        },
        "gold_hate": source["gate"],
        **{"gold_" + k: source.get(k) for k in GOLD_FIELDS},
        "known_fine_gold": {k: source.get(k) is not None for k in GOLD_FIELDS},
        "eligible_fine_gold": {
            k: source.get(k) is not None and (k == "legal" or source["gate"])
            for k in GOLD_FIELDS
        },
        "stage": records[0]["stage"],
        "execution_period": records[0]["execution_period"],
        "masks": {
            "common_valid": valid,
            "strict_all_votes": strict,
            "replay_exclusion_member": replay,
            "common_valid_exclude_replay": valid and not replay,
            "strict_exclude_replay": strict and not replay,
        },
        "predictions": [
            [
                {k: by_slot[c, repeat][k] for k in ("result", "n_valid", "uncertain")}
                for repeat in range(3)
            ]
            for c in conditions
        ],
    }


def build():
    start = time.monotonic()
    need(
        not OUT.exists(),
        "Preserve existing Part 1 output; use a reviewed new version if needed",
    )
    overview = read(OVERVIEW)
    need(
        overview["status"] == "completion_overview_pass",
        "Initial completion check has not passed",
    )
    bindings = {rel(OVERVIEW): sha(OVERVIEW), rel(Path(__file__)): sha(Path(__file__))}
    bindings[rel(ROOT / "tests/feedback/transfer/test_archive.py")] = sha(
        ROOT / "tests/feedback/transfer/test_archive.py"
    )

    def bound(path, expected=None):
        expected = expected or overview["verified_metadata_hashes"][rel(path)]
        value = read_bound(path, expected)
        bindings[rel(path)] = expected
        return value

    plan = bound(EXECUTION / "plan.json", PLAN_SHA)
    done = bound(EXECUTION / "live/complete.json", overview["completion_sha256"])
    need(
        done["status"] == "all_studies_executed_and_audited"
        and done["mode"] == "live"
        and done["plan_sha256"] == PLAN_SHA
        and done["gold_scoring"] is False,
        "Wrong completed execution",
    )
    need(done["actual_finalization_API_calls"] == 0, "Unexpected finalization contract")
    science = bound(STUDY / "scientific_plan_v1.json")
    schedule = bound(EXECUTION / "schedule.json")
    roster_path = STUDY / "roster.jsonl"
    need(
        sha(roster_path) == plan["input_hashes"][rel(roster_path)],
        "Frozen roster changed",
    )
    bindings[rel(roster_path)] = plan["input_hashes"][rel(roster_path)]
    for path in (STUDY / "STUDY_PROTOCOL.md", ROOT / "config/taxonomy.yaml"):
        need(
            sha(path) == plan["input_hashes"][rel(path)], "Protocol or taxonomy changed"
        )
        bindings[rel(path)] = plan["input_hashes"][rel(path)]
    roster = list(rows(roster_path))
    positions = {r["key"]: i for i, r in enumerate(roster)}
    need(
        len(roster) == len(positions) == science["inputs"] == 15758,
        "Wrong roster identity/count",
    )
    need(all(type(r["gate"]) is bool for r in roster), "Nonboolean gold gate")
    conditions = science["conditions"]
    need(
        len(conditions) == len(set(conditions)) == 4
        and conditions[0] == "reference"
        and science["decoder_repeats"] == science["votes_per_prediction"] == 3,
        "Wrong matrix axes",
    )
    support = fixed_support(roster)
    need(support == science["fixed_support"], "Frozen known-label support differs")
    replay = set(science["mandatory_replay_keys"])
    need(len(replay) == 20 and replay <= set(positions), "Replay exclusion keys differ")
    write(
        OUT / "started.json",
        {
            "status": "assembling_already_audited_results",
            "started_utc": utc(),
            "input_bindings": bindings,
            "new_Qwen_calls": 0,
            "new_fits": 0,
            "raw_reparse": False,
            "gold_scoring": False,
        },
    )
    write(OUT / "roster.jsonl", physical(roster_path).read_bytes())
    artifacts = {rel(OUT / "roster.jsonl"): sha(OUT / "roster.jsonl")}
    seen, seen_inputs, batches = set(), set(), []
    totals = [Counter(), Counter(), Counter()]
    matrix = [None] * len(roster)
    stage_ids = [f"transfer_{i:02d}" for i in range(1, 23)]
    need([s["id"] for s in schedule["stages"]] == stage_ids, "Wrong batch order")
    with ThreadPoolExecutor(max_workers=8) as executor:
        for stage in schedule["stages"]:
            target = LIVE / "stages" / stage["id"]
            review = bound(
                target / "review.json",
                done["review_hashes"][rel(target / "review.json")],
            )
            need(
                review["status"] == "audited"
                and review["plan_sha256"] == PLAN_SHA
                and review["mode"] == "live"
                and review["study"] == "transfer"
                and review["stage"] == stage["id"],
                "Wrong batch review",
            )
            complete = bound(target / "complete.json", review["complete_sha256"])
            report = bound(
                inside(target, review["report"]), review["artifacts"][review["report"]]
            )
            need(
                report["stage_complete_sha256"] == review["complete_sha256"],
                "Audit/batch binding differs",
            )
            need(
                not complete["summary"]["stopped"]
                and not complete["summary"]["errors"],
                "Unfinished batch",
            )
            jobs = stage["jobs"]
            receipts = complete["item_receipt_hashes"]
            job_slots = [(j["key"], j["condition"], j["repeat"]) for j in jobs]
            need(
                len(set(job_slots)) == len(jobs) == stage["items"],
                "Duplicate/missing schedule job",
            )
            need(
                set(receipts)
                == {"items/" + item_id(j) + "/complete.json" for j in jobs},
                "Receipt matrix differs",
            )
            for job in jobs:
                need(
                    job["study"] == "transfer"
                    and job["key"] in positions
                    and job["condition"] in conditions
                    and type(job["repeat"]) is int
                    and job["repeat"] in range(3),
                    "Unplanned job",
                )
                need(
                    job["population"] == roster[positions[job["key"]]]["population"],
                    "Scheduled population differs",
                )
                need(
                    job["schema_path"] == "config/taxonomy.yaml"
                    and job["schema_sha256"] == bindings["config/taxonomy.yaml"],
                    "Scheduled taxonomy differs",
                )
            records = list(executor.map(lambda job: read_item(job, receipts), jobs))
            counts = record_counts(records)
            for got, field in zip(counts, ("counts", "accounting", "observed_usage")):
                need(
                    got == Counter(report[field]),
                    "Batch aggregate differs: " + stage["id"] + "/" + field,
                )
            grouped = defaultdict(list)
            period = (
                "original_8"
                if stage["id"] < "transfer_04"
                else (
                    "mixed_resume_8_128"
                    if stage["id"] == "transfer_04"
                    else "later_128"
                )
            )
            for row in records:
                slot = (row["key"], row["condition"], row["repeat"])
                need(slot not in seen, "Prediction repeated across batches")
                seen.add(slot)
                row.update(
                    stage=stage["id"],
                    execution_period=period,
                    input_index=positions[row["key"]],
                )
                grouped[row["key"]].append(row)
            need(set(grouped) == set(stage["input_keys"]), "Batch input roster differs")
            for key, input_records in grouped.items():
                need(key not in seen_inputs, "Text repeated across batches")
                seen_inputs.add(key)
                i = positions[key]
                matrix[i] = matrix_row(roster[i], i, input_records, conditions, replay)
            for total, values in zip(totals, counts):
                total.update(values)
            path = OUT / "assembly" / (stage["id"] + ".jsonl")
            write(path, b"".join(encoded(r) for r in records))
            artifacts[rel(path)] = sha(path)
            batches.append(
                {
                    "stage": stage["id"],
                    "inputs": len(grouped),
                    "counts": dict(counts[0]),
                    "accounting": dict(counts[1]),
                    "observed_usage": dict(counts[2]),
                    "execution_period": period,
                }
            )
            print(
                f"Part 1: {stage['id']} assembled, {len(seen):,}/189,096 predictions; elapsed {time.monotonic()-start:.0f}s",
                flush=True,
            )
    need(
        len(seen) == science["predictions"] == 189096 and seen_inputs == set(positions),
        "Incomplete final matrix",
    )
    for got, field in zip(totals, ("counts", "accounting", "observed_usage")):
        need(got == Counter(done[field]), "Final aggregate differs: " + field)
    group_masks = {}
    for population in ("en_test", "en_hatecheck_eval"):
        for gate in (False, True):
            eligible = [
                r
                for r in matrix
                if r["population"] == population and r["gold_hate"] == gate
            ]
            group_masks[population + ("_FNR" if gate else "_FPR")] = {
                "full_inputs": len(eligible),
                "full_inference_clusters": len(
                    {r["inference_cluster"] for r in eligible}
                ),
                **{
                    name: sum(r["masks"][name] for r in eligible)
                    for name in matrix[0]["masks"]
                },
            }
    missingness = []
    for population in ("en_test", "en_hatecheck_eval"):
        for ci, condition in enumerate(conditions):
            for repeat in range(3):
                rr = [r for r in matrix if r["population"] == population]
                nv = Counter(r["predictions"][ci][repeat]["n_valid"] for r in rr)
                missingness.append(
                    {
                        "population": population,
                        "condition": condition,
                        "repeat": repeat,
                        "slots": len(rr),
                        "valid_vote_counts": {str(i): nv[i] for i in range(4)},
                    }
                )
    analysis = {
        "status": "analysis_matrix_ready_unscored",
        "conditions": conditions,
        "repeats": [0, 1, 2],
        "shape": [15758, 4, 3],
        "fixed_support": support,
        "endpoint_masks": group_masks,
        "inputs_with_any_null": sum(not r["masks"]["common_valid"] for r in matrix),
        "inputs_with_any_under_three_votes": sum(
            not r["masks"]["strict_all_votes"] for r in matrix
        ),
        "missingness_by_population_condition_repeat": missingness,
        "batch_summaries": batches,
        "new_Qwen_calls": 0,
        "new_fits": 0,
        "gold_scoring": False,
        "execution_period_note": "Batch 4 is mixed, including possible partial-vote resumes; no individual call is reclassified from a completion timestamp.",
        "matrix_schema": "Roster order; four conditions in scientific_plan order, then decoder repeats 0/1/2. None is missing output/gold, not false or an empty set. Reference occupies three slots only.",
        "sensitivity_note": "Common/strict/replay masks are recorded. Cluster IDs support prespecified representative and cluster-bootstrap checks in subsequent parts; no result-selected exclusions.",
    }
    write(OUT / "matrix.jsonl", b"".join(encoded(r) for r in matrix))
    write(OUT / "analysis_ready.json", analysis)
    for name in ("matrix.jsonl", "analysis_ready.json", "started.json"):
        artifacts[rel(OUT / name)] = sha(OUT / name)
    for name, expected in bindings.items():
        need(
            sha(inside(ROOT, name)) == expected,
            "Consumed source changed during assembly: " + name,
        )
    write(
        OUT / "assembly_complete.json",
        {
            "status": "complete_from_audited_receipts_unscored",
            "completed_utc": utc(),
            "elapsed_seconds": time.monotonic() - start,
            "input_bindings": bindings,
            "artifacts": artifacts,
            "plan_sha256": PLAN_SHA,
            "counts": dict(totals[0]),
            "accounting": dict(totals[1]),
            "observed_usage": dict(totals[2]),
            "item_receipts_checked": 189096,
            "saved_results_checked": 189096,
            "saved_vote_aggregations_checked": 189096,
            "raw_reparse": False,
            "gold_scoring": False,
            "new_Qwen_calls": 0,
            "new_fits": 0,
            "verification_basis": "Reuse finished automatic raw audits; hash-check only consumed metadata, item receipts and result files. Independently reaggregate saved parsed votes. Independent matrix verification is separate.",
        },
    )
    print(
        "Part 1 assembly complete. Independent matrix check is next; no effectiveness scoring.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true", required=True)
    parser.parse_args()
    with offline():
        build()

from collections import Counter, defaultdict
import ctypes
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import time
import unittest

from analysis.feedback.reserved import review as io
from analysis.feedback.reserved import resources_metrics as core
from tests.feedback.reserved import test_resources as tests

ROOT = io.ROOT
OUT = ROOT / "experiments/astra/sq3_reserved_part8_results_20261020_v2"
PREP = ROOT / "experiments/astra/sq3_reserved_part8_preparation_20261020_v1"
ASSEMBLY = ROOT / "experiments/astra/sq3_reserved_audited_assembly_20261019_v1"
P4 = ROOT / "experiments/astra/sq3_reserved_part4_results_20261020_v1"
P7 = ROOT / "experiments/astra/sq3_reserved_part7_results_20261020_v1"
LIVE = io.EXECUTION / "live/reserved_slot"
LOG = ROOT / "Thesis_Log/astra_result"
need = core.need
BINDINGS = {}


def write(path, value):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside Part 8 results")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        value
        if isinstance(value, str)
        else json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(data)


def bind(path, expected=None):
    name, digest = io.rel(path), io.sha(path)
    need(expected is None or digest == expected, "Changed bound evidence: " + name)
    need(name not in BINDINGS or BINDINGS[name] == digest, "Concurrent change: " + name)
    BINDINGS[name] = digest
    return digest


def read(path, expected=None):
    bind(path, expected)
    return io.read(path)


def policy(condition):
    return (
        condition
        if "__" not in condition
        else condition.rsplit("_s", 1)[0].replace("__", "/")
    )


def mem():
    class M(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (k, ctypes.c_ulonglong)
            for k in (
                "ullTotalPhys",
                "ullAvailPhys",
                "ullTotalPageFile",
                "ullAvailPageFile",
                "ullTotalVirtual",
                "ullAvailVirtual",
                "ullAvailExtendedVirtual",
            )
        ]

    m = M()
    m.dwLength = ctypes.sizeof(m)
    need(
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m)),
        "Memory check failed",
    )
    return {
        "available_ram_bytes": m.ullAvailPhys,
        "disk_free_bytes": shutil.disk_usage(ROOT).free,
    }


def audit_history(done):
    stages, reports, statuses, models, finishes = (
        [],
        {},
        Counter(),
        Counter(),
        Counter(),
    )
    retry_records = []
    for i in range(1, 17):
        stage = f"reserved_slot_{i:02}"
        folder = LIVE / "stages" / stage
        rev = read(
            folder / "review.json",
            done["review_hashes"][io.rel(folder / "review.json")],
        )
        comp = read(folder / "complete.json", rev["complete_sha256"])
        report = read(folder / rev["report"], rev["artifacts"][rev["report"]])
        need(
            report["stage_complete_sha256"] == rev["complete_sha256"],
            "Stage/audit differs",
        )
        history = {}
        for name, sha in report["attempt_hashes"].items():
            history[name] = read(LIVE / name, sha)
        attempts, check_durations = [], []
        folders = sorted(
            {
                str(Path(n).parent).replace("\\", "/")
                for n in history
                if Path(n).parent.name.startswith("attempt_")
            }
        )
        for name in folders:
            start = history[name + "/start.json"]
            a = {"path": io.rel(LIVE / name), "start": start["started_utc"]}
            if name + "/inventory.json" in history:
                inv = history[name + "/inventory.json"]
                a.update(
                    completed_on_entry=inv["completed"], pending_on_entry=inv["pending"]
                )
            if name + "/model_check_intent.json" in history:
                check = history[name + "/model_check_intent.json"]
                response = history[name + "/model_check_response.json"]
                need(
                    response["status"] == 200
                    and check["action"] == "GET PEASEC /models",
                    "Unexpected availability-check state",
                )
                a["model_check_start"] = check["started_utc"]
                a["model_check_seconds"] = response["elapsed_s"]
                check_durations.append(response["elapsed_s"])
            ends = []
            for endfile in ("summary.json", "interrupted.json"):
                if name + "/" + endfile in history:
                    z = history[name + "/" + endfile]
                    ends.append(z["finished_utc"])
                    a["end_kind"] = endfile
                    if endfile == "summary.json":
                        a["stopped"] = z["stopped"]
                        a["saved_items"] = z["saved_items"]
            need(len(ends) <= 1, "Conflicting attempt endpoint markers")
            if ends:
                a["end"] = ends[0]
            attempts.append(a)
        need(
            len(attempts) == report["attempt_count"]
            and len(check_durations) == report["model_availability_checks"],
            "Attempt/GET count differs",
        )
        first = min(x["start"] for x in attempts)
        summary_end = comp["summary"]["finished_utc"]
        end = comp["completed_utc"]
        audited = report["created_utc"]
        partition = core.attempt_partition(attempts, summary_end)
        need(
            math.isclose(
                core.seconds(first, end),
                report["wall_seconds_including_restarts"],
                abs_tol=1e-6,
            ),
            "Original wall calculation differs",
        )
        stages.append(
            {
                "stage": stage,
                "first_start_utc": first,
                "final_summary_utc": summary_end,
                "stage_complete_utc": end,
                "audit_report_created_utc": audited,
                "attempts": attempts,
                "attempt_partition": partition,
                "stage_span_seconds_including_restarts": core.seconds(first, end),
                "work_window_seconds_including_restarts": core.seconds(
                    first, summary_end
                ),
                "completion_bookkeeping_window_seconds": core.seconds(summary_end, end),
                "completion_to_audit_report_window_seconds": core.seconds(end, audited),
                "availability_GETs": len(check_durations),
                "availability_GET_seconds_sum": math.fsum(check_durations),
                "original_vote_seconds_sum": report["vote_seconds_sum"],
            }
        )
        statuses.update(report["http_statuses"])
        models.update(report["reported_models"])
        finishes.update(report["finish_reasons"])
        for r in report["retry_events"]:
            retry_records.append({**r, "stage": stage})
        reports[stage] = report
    between = []
    for left, right in zip(stages, stages[1:]):
        between.append(
            {
                "after": left["stage"],
                "before": right["stage"],
                "start": left["audit_report_created_utc"],
                "end": right["first_start_utc"],
                "seconds": core.seconds(
                    left["audit_report_created_utc"], right["first_start_utc"]
                ),
            }
        )
    partition = Counter()
    for s in stages:
        partition.update(s["attempt_partition"]["seconds_by_kind"])
        partition["completion_bookkeeping_window"] += s[
            "completion_bookkeeping_window_seconds"
        ]
        partition["completion_to_audit_report_window"] += s[
            "completion_to_audit_report_window_seconds"
        ]
    partition["between_stage_window"] = math.fsum(x["seconds"] for x in between)
    envelope = core.seconds(
        stages[0]["first_start_utc"], stages[-1]["audit_report_created_utc"]
    )
    need(
        math.isclose(math.fsum(partition.values()), envelope, abs_tol=1e-5),
        "Full chronological partition differs",
    )
    need(
        sum(statuses.values()) == done["accounting"]["post_attempts"], "Statuses differ"
    )
    need(
        len(retry_records) == done["accounting"]["retry_attempts"],
        "Retry records differ",
    )
    retry_identity = [
        (
            r["stage"],
            r["job"]["key"],
            r["job"]["condition"],
            r["job"]["repeat"],
            r["vote"],
            r["turn"],
            r["attempt"],
        )
        for r in retry_records
    ]
    need(len(set(retry_identity)) == len(retry_identity), "Duplicate retry attempt")
    ready = read(
        ROOT
        / "experiments/astra/sq3_reserved_part1_integrity_20261019_v1/readiness/check.json"
    )
    meta_path = (
        ROOT
        / "experiments/astra/sq3_reserved_part1_integrity_20261019_v1/metadata_check.json"
    )
    metadata = read(meta_path, ready["sources"][io.rel(meta_path)])
    mtime = metadata["final_receipt_mtime_utc"]
    completion_proxy = {
        "final_receipt_mtime_utc_recorded_in_part1": mtime,
        "start_to_final_receipt_mtime_seconds": core.seconds(
            stages[0]["first_start_utc"], mtime
        ),
        "last_audit_report_to_final_receipt_mtime_seconds": core.seconds(
            stages[-1]["audit_report_created_utc"], mtime
        ),
        "qualification": "Historical filesystem timestamp captured in Part 1; not an internally "
        "logged finalization endpoint or pure finalizer CPU duration.",
    }
    return reports, {
        "stages": stages,
        "between_stages": between,
        "first_start_utc": stages[0]["first_start_utc"],
        "last_audit_report_created_utc": stages[-1]["audit_report_created_utc"],
        "envelope_seconds": envelope,
        "partition_seconds": dict(partition),
        "availability_GETs": sum(s["availability_GETs"] for s in stages),
        "availability_GET_seconds_sum": math.fsum(
            s["availability_GET_seconds_sum"] for s in stages
        ),
        "http_and_transport_statuses": dict(statuses),
        "reported_models": dict(models),
        "finish_reasons": dict(finishes),
        "retry_records": retry_records,
        "completion_proxy": completion_proxy,
        "qualification": "Windows include local work, waiting and possible interruptions. "
        "Recorded request durations overlap across eight workers. Unallocated shutdown windows "
        "are not measured downtime. Audit endpoints mark report creation, not pure audit CPU. "
        "No recorded final-queue UTC endpoint; no finalization duration inferred.",
    }


def recoveries(selected):
    output = []
    for name in ("recovery_20261014_v1", "recovery_20261016_v2"):
        manifest = read(io.EXECUTION / name / "manifest.json")
        records = []
        for c in manifest["calls"]:
            item = c["item"]
            row = selected[item]
            receipt_path = ROOT / row["item_receipt"]
            receipt = read(receipt_path, row["item_receipt_sha256"])
            saved = read(ROOT / item / "result.json", row["result_sha256"])
            need(
                receipt["hashes"][item.split("/reserved_slot/")[-1] + "/result.json"]
                == row["result_sha256"],
                "Recovery result receipt differs",
            )
            events = [
                e
                for e in saved["events"]
                if e["vote"] == c["vote"] and e["turn"] == c["turn"]
            ]
            if name.endswith("14_v1"):
                need(
                    [e["attempt"] for e in events] == [0, 1],
                    "Shutdown recovery differs",
                )
                need(
                    events[0]["response_class"] == io.ADMIN
                    and events[0]["elapsed_s"] == 0,
                    "Unknown shutdown sentinel changed",
                )
            else:
                need(
                    [e["attempt"] for e in events] == list(range(6)),
                    "Wi-Fi recovery differs",
                )
                need(
                    all(e["action"] == "retry" for e in events[:5]),
                    "Earlier failure missing",
                )
            need(events[-1]["action"] == "ok", "Recovery did not complete")

            timings = []
            for e in events:
                folder = (
                    ROOT
                    / item
                    / f"calls/vote_{c['vote']}_turn_{c['turn']}/attempt_{e['attempt']}"
                )
                local = folder.relative_to(LIVE).as_posix()
                intent = read(
                    folder / "intent.json", receipt["hashes"][local + "/intent.json"]
                )
                outcome = read(
                    folder / "outcome.json", receipt["hashes"][local + "/outcome.json"]
                )
                need(
                    outcome["elapsed_s"] == e["elapsed_s"], "Recovery duration differs"
                )
                timings.append(
                    {
                        "attempt": e["attempt"],
                        "started_utc": intent["started_utc"],
                        "finished_utc": outcome["finished_utc"],
                        "finished_utc_is_administrative_review_time": e[
                            "response_class"
                        ]
                        == io.ADMIN,
                        "known_elapsed_s": (
                            None if e["response_class"] == io.ADMIN else e["elapsed_s"]
                        ),
                    }
                )
            records.append(
                {
                    "item": item,
                    "key": row["key"],
                    "condition": row["condition"],
                    "repeat": row["repeat"],
                    "vote": c["vote"],
                    "turn": c["turn"],
                    "events": events,
                    "timing_markers": timings,
                }
            )
        output.append(
            {
                "incident": name,
                "records": records,
                "unknown_remote_delivery_usage_duration": name.endswith("14_v1"),
                "new_part8_calls": 0,
            }
        )
    return output


def inherited():
    base = ROOT / "experiments/astra/sq3_additions_results_20261006_v1"
    receipt = read(base / "benign_selection/complete.json")
    for field in ("inputs", "outputs"):
        for name, sha in receipt[field].items():

            if name.endswith(
                (
                    "review/predictions.jsonl",
                    "review/summary.json",
                    "benign_selection/summary.json",
                    "actual_pools.json",
                )
            ):
                bind(ROOT / name, sha)
    summary = read(base / "review/summary.json")
    audit = summary["stage_operational"]["benign_selection_01"]
    totals, keys, repeats = Counter(), set(), set()
    n = 0
    for row in io.rows(base / "review/predictions.jsonl"):
        if row["study"] != "benign_selection":
            continue
        n += 1
        keys.add(row["key"])
        repeats.add((row["key"], row["repeat"]))
        totals.update(row["costs"])
    need(n == len(repeats) == 3072 and len(keys) == 1024, "Selection matrix differs")
    for k, v in {**audit["accounting"], **audit["observed_usage"]}.items():
        need(totals[k] == v, "Screening costs disagree with stage audit")
    selected = read(base / "benign_selection/summary.json")
    pools = read(P4 / "source_pool_audit.json")
    need(
        selected["labels_inspected"] == pools["candidate_count"] == 1024
        and selected["distinct_insertable_sources"]
        == pools["unique_all_pool_sources"]
        == 413,
        "Screening/pool labels differ",
    )
    return {
        "screening": {
            "inputs": 1024,
            "predictions": n,
            "existing_binary_labels_inspected": 1024,
            "accounting": audit["accounting"],
            "observed_usage": audit["observed_usage"],
            "original_stage_wall_seconds_including_restarts": audit[
                "wall_seconds_including_restarts"
            ],
            "original_vote_seconds_sum": audit["vote_seconds_sum"],
            "first_use_programme": "SQ3 additions including benign selection",
            "included_in_current_run_totals": False,
            "additional_source_labels": 0,
            "additional_screening_calls": 0,
        },
        "pools": {
            "count": 18,
            "memberships": 720,
            "unique_source_texts": 413,
            "candidates_reviewed": 1024,
            "nonunanimous_previous_majorities": 75,
            "labels_per_correction": "Binary only; all fine fields remain unknown",
        },
        "human_review_separate": {
            "people": 2,
            "forms": 4,
            "texts": 24,
            "judgments_across_two_guides": 96,
            "reported_participant_minutes": 100,
            "used_as_feedback_in_current_run": False,
            "future_model_test_ordinary_calls": 648,
            "future_study_executed": False,
        },
        "limits": [
            "Existing dataset labels inspected, not 1,024 newly collected judgments.",
            "Forty sources per pool are not the complete label/selection budget.",
            "Overlapping pools and four delivery modes do not multiply annotation labor.",
            "KB preparation, cached retrieval, authoring, audits and curation are inherited; "
            "complete person-hours, hardware energy and whole-project costs were not timed.",
        ],
    }


def historical_ledger(total):
    oldfile = LOG / "astra_resource_evidence_20261012_v3.md"
    bind(oldfile)
    lines = oldfile.read_text(encoding="utf-8-sig").splitlines()
    programmes = []
    for line in lines:
        if not line.startswith("| ["):
            continue
        fields = [x.strip() for x in line.strip("|").split("|")]
        if len(fields) != 6:
            continue
        title, path = re.fullmatch(r"\[([^]]+)\]\(([^)]+)\)", fields[0]).groups()
        nums = [int(x.replace(",", "")) for x in fields[1:]]
        p = LOG / path
        bind(p)
        programmes.append(
            dict(
                zip(
                    (
                        "ordinary_slots",
                        "repair_slots",
                        "retry_or_recovered_failure_attempts",
                        "successful_responses",
                        "reported_tokens",
                    ),
                    nums,
                ),
                programme=title,
                source=io.rel(p),
                attempt_slots=nums[0] + nums[1] + nums[2],
            )
        )
    need(len(programmes) == 9, "Historical ledger does not contain nine programmes")
    previous = read(
        ROOT
        / "experiments/astra/offline_closure_20261006_v1/documentation/resources.json"
    )
    fields = (
        "ordinary_slots",
        "repair_slots",
        "retry_or_recovered_failure_attempts",
        "successful_responses",
        "reported_tokens",
        "attempt_slots",
    )
    for a, b in zip(programmes[:7], previous["selected_programmes"]):
        need(
            all(a[k] == b[k] for k in fields),
            "Historical seven-programme ledger differs",
        )
    for index, folder in (
        (7, "sq3_balanced_results_20261010_v1"),
        (8, "sq3_five_slot_results_20261012_v1"),
    ):
        s = read(ROOT / f"experiments/astra/{folder}/review/summary.json")
        operational = s["operational"]

        a = operational.get("accounting", operational)
        u = operational.get("observed_usage", operational)
        aliases = {
            "ordinary_slots": "ordinary_logical_calls",
            "repair_slots": "repair_logical_calls",
            "retry_or_recovered_failure_attempts": "retry_attempts",
            "successful_responses": "http_success_with_text",
            "attempt_slots": "post_attempts",
        }
        need(
            all(programmes[index][k] == a[v] for k, v in aliases.items())
            and programmes[index]["reported_tokens"] == u["total_tokens"],
            "Historical added programme differs",
        )
    historical = {k: sum(p[k] for p in programmes) for k in fields}
    need(
        historical["reported_tokens"] == 880940304
        and historical["successful_responses"] == 960894
        and historical["attempt_slots"] == 963424,
        "Nine-programme published total differs",
    )
    a = total["accounting"]
    programmes.append(
        {
            "programme": "SQ3 reserved correction delivery",
            "source": io.rel(OUT / "resource_groups.json"),
            "ordinary_slots": a["ordinary_logical_calls"],
            "repair_slots": a["repair_logical_calls"],
            "retry_or_recovered_failure_attempts": a["retry_attempts"],
            "successful_responses": a["http_success_with_text"],
            "attempt_slots": a["post_attempts"],
            "reported_tokens": total["observed_usage"]["total_tokens"],
        }
    )
    return {
        "programmes": programmes,
        "preserved_nine_programme_totals": historical,
        "ten_programme_totals": {k: sum(p[k] for p in programmes) for k in fields},
        "scope": "Partial ledger of ten nonoverlapping remote programmes, not the entire thesis. "
        "Saved-vote subsets and source screening already included; never add twice. "
        "Includes accounted unknown delivery slots; excludes unknown usage, omitted "
        "programmes, local fitting/embedding history, money and energy.",
    }


def main():
    began = time.monotonic()
    need(not OUT.exists(), "Preserve any earlier Part 8 result attempt")
    environment = mem()
    need(
        environment["available_ram_bytes"] >= 2 * 2**30
        and environment["disk_free_bytes"] >= 2 * 2**30,
        "Insufficient resources for bounded offline analysis",
    )
    suite = unittest.defaultTestLoader.loadTestsFromModule(tests)
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    need(result.wasSuccessful(), "Synthetic accounting checks failed")
    for p in (
        Path(__file__),
        Path(core.__file__),
        Path(tests.__file__),
        PREP / "scope.md",
        Path(io.__file__),
        LOG / "astra_sq3_human_feedback_preparation_20261015_v1.md",
        LOG / "astra_sq3_reserved_connection_recovery_20261016_v2.md",
    ):
        bind(p)
    done = read(io.EXECUTION / "live/complete.json")
    assembly = read(
        ASSEMBLY / "complete.json",
        "f6bbca18ca59dcfcf7a639399acd8367bcdbcc27cdb424143afe0254abaf7d03",
    )
    need(
        done["status"] == "all_studies_executed_and_audited"
        and assembly["status"]
        == "complete_from_automatically_audited_receipts_unscored",
        "Wrong completion provenance",
    )
    for name, sha in assembly["artifacts"].items():
        bind(ROOT / name, sha)
    science = read(io.STUDY / "scientific_plan_v1.json")
    roster_path = io.STUDY / "roster.jsonl"
    bind(roster_path)
    roster = {r["key"]: r for r in io.rows(roster_path)}
    for p in (
        P4 / "complete.json",
        P7 / "complete.json",
        P7 / "interpretation_check.json",
        LOG / "astra_sq3_reserved_part7_robustness_20261020_v1.md",
    ):
        bind(p)
    p7done = io.read(P7 / "complete.json")
    for name, sha in p7done.get("report_binding", {}).items():
        bind(ROOT / name, sha)
    write(
        OUT / "started.json",
        {
            "status": "part8_offline_resource_analysis_started",
            "created_utc": io.utc(),
            "environment": environment,
            "tests_passed": result.testsRun,
            "initial_bindings": BINDINGS.copy(),
            "new_model_calls": 0,
            "new_training_fits": 0,
            "full_raw_replay": False,
        },
    )
    reports, timeline = audit_history(done)
    write(OUT / "execution_timeline.json", timeline)
    recovery_items = set()
    for name in ("recovery_20261014_v1", "recovery_20261016_v2"):
        recovery_items.update(
            c["item"] for c in read(io.EXECUTION / name / "manifest.json")["calls"]
        )
    groups, selected, seen = defaultdict(core.Group), {}, set()
    for i in range(1, 17):
        stage = f"reserved_slot_{i:02}"
        for row in io.rows(ASSEMBLY / f"assembly/{stage}.jsonl"):
            ident = row["key"], row["condition"], row["repeat"]
            need(
                ident not in seen and row["stage"] == stage,
                "Duplicate or wrong batch record",
            )
            seen.add(ident)
            r = roster[row["key"]]
            src = r.get("data_source", r.get("source"))
            need(isinstance(src, str), "Missing source")
            pop = r["population"]
            condition = row["condition"]
            labels = [
                "all",
                "condition/" + condition,
                "batch/" + stage,
                "mode/" + condition.split("__")[0],
                "policy/" + policy(condition),
                "population/" + pop,
                "source/" + src,
                "source_policy/" + src + "/" + policy(condition),
                "repeat/" + str(row["repeat"]),
                "text/" + row["key"],
            ]
            for label in labels:
                groups[label].add(row)
            item = row["item_receipt"].removesuffix("/complete.json")
            if item in recovery_items:
                selected[item] = row
        z = groups["batch/" + stage].finish()
        need(
            z["accounting"] == reports[stage]["accounting"]
            and z["observed_usage"] == reports[stage]["observed_usage"],
            "Batch resource mismatch",
        )
        need(
            math.isclose(
                z["known_request_seconds_sum"] + z["retry_backoff_seconds_sum"],
                reports[stage]["vote_seconds_sum"],
                abs_tol=1e-6,
            ),
            "Observed request plus wait disagrees with original audit",
        )
        print(
            f"Part 8 resources: {stage}, {len(seen):,}/149,850 predictions", flush=True
        )
    expected = {
        (k, c, r) for k in roster for c in science["conditions"] for r in range(3)
    }
    need(
        seen == expected and len(selected) == 16,
        "Incomplete resource matrix/recovery subset",
    )
    values = {k: g.finish() for k, g in groups.items()}
    total = values["all"]
    need(
        total["accounting"] == done["accounting"] == assembly["accounting"]
        and total["observed_usage"]
        == done["observed_usage"]
        == assembly["observed_usage"],
        "Overall resource reconciliation differs",
    )
    need(total["counts"]["administrative_unknown_requests"] == 8, "Unknowns changed")
    write(
        OUT / "resource_groups.json",
        {k: v for k, v in values.items() if not k.startswith("text/")},
    )
    write(
        OUT / "text_resource_profiles.json",
        {k[5:]: v for k, v in values.items() if k.startswith("text/")},
    )
    write(OUT / "recovery_details.json", recoveries(selected))
    write(OUT / "inherited_work.json", inherited())
    write(OUT / "partial_programme_ledger.json", historical_ledger(total))

    promptfile = P4 / "prompt_metadata.jsonl"
    bind(promptfile)
    local = defaultdict(
        lambda: {
            "tokens": [],
            "delta": [],
            "hashes": set(),
            "same_reference": 0,
            "delivered": 0,
        }
    )
    prompt_seen = set()
    for row in io.rows(promptfile):
        key = row["key"], row["condition"]
        need(key not in prompt_seen, "Duplicate prompt")
        prompt_seen.add(key)
        for label in (
            "all",
            "condition/" + row["condition"],
            "mode/" + row["condition"].split("__")[0],
            "policy/" + policy(row["condition"]),
        ):
            z = local[label]
            z["tokens"].append(
                row["reference_input_tokens"] + row["token_difference_vs_reference"]
            )
            z["delta"].append(row["token_difference_vs_reference"])
            z["hashes"].add(row["messages_sha256"])
            z["same_reference"] += int(row["identical_to_reference"])
            z["delivered"] += bool(row["delivered_correction_ids"])
    need(len(prompt_seen) == 49950, "Wrong frozen prompt count")
    local_values = {}
    for label, z in local.items():
        n = len(z["tokens"])
        local_values[label] = {
            "prompts": n,
            "local_token_sum": sum(z["tokens"]),
            "local_token_mean": sum(z["tokens"]) / n,
            "local_token_min": min(z["tokens"]),
            "local_token_max": max(z["tokens"]),
            "delta_vs_reference_mean": sum(z["delta"]) / n,
            "same_reference_prompts": z["same_reference"],
            "delivered_correction_prompts": z["delivered"],
            "unique_message_hashes": len(z["hashes"]),
            "local_tokens_times_nine_ordinary_slots": 9 * sum(z["tokens"]),
            "observed_ordinary_prompt_tokens": values[label]["ordinary_usage"][
                "prompt_tokens"
            ],
            "observed_minus_local_times_nine": values[label]["ordinary_usage"][
                "prompt_tokens"
            ]
            - 9 * sum(z["tokens"]),
        }
    write(
        OUT / "local_prompt_costs.json",
        {
            "groups": local_values,
            "qualification": "Local frozen tokenizer estimates; not provider billing. Each frozen prompt "
            "has three decoder repetitions of three ordinary votes. Repairs are additional messages. "
            "Unique byte hashes are a descriptive catalogue, not permission to deduplicate planned samples.",
        },
    )
    summary = {
        "status": "part8_resource_statistics_complete",
        "counts": total["counts"],
        "accounting": total["accounting"],
        "observed_usage": total["observed_usage"],
        "ordinary_usage": total["ordinary_usage"],
        "repair_usage": total["repair_usage"],
        "known_request_seconds_sum": total["known_request_seconds_sum"],
        "retry_backoff_seconds_sum": total["retry_backoff_seconds_sum"],
        "known_request_duration_count": total["known_request_duration_count"],
        "elapsed_envelope_seconds": timeline["envelope_seconds"],
        "availability_GETs": timeline["availability_GETs"],
        "resource_groups": len(values) - len(roster),
        "text_profiles": len(roster),
        "top_repair_texts": sorted(
            [
                {
                    "key": k[5:],
                    "repairs": v["accounting"]["repair_logical_calls"],
                    "observed_tokens": v["observed_usage"]["total_tokens"],
                    "nulls": v["counts"]["nulls"],
                }
                for k, v in values.items()
                if k.startswith("text/")
            ],
            key=lambda z: (-z["repairs"], z["key"]),
        )[:10],
        "provenance": "Bound completed automatic audits and compact audited-receipt assembly; "
        "small recovery subset rechecked against original item receipts. No full raw replay.",
        "new_model_calls": 0,
        "new_embeddings": 0,
        "new_training_fits": 0,
    }
    write(OUT / "resource_summary.json", summary)
    io.verify(ROOT, BINDINGS)
    paths = [p for p in OUT.iterdir() if p.is_file()]
    write(
        OUT / "statistics_complete.json",
        {
            "status": "part8_resource_statistics_complete",
            "created_utc": io.utc(),
            "elapsed_local_analysis_seconds": time.monotonic() - began,
            "bindings": BINDINGS,
            "artifacts": {io.rel(p): io.sha(p) for p in paths},
            "tests_passed": result.testsRun,
            "new_model_calls": 0,
            "new_training_fits": 0,
            "review_pending": True,
        },
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    with io.offline():
        main()

from collections import Counter, defaultdict
from datetime import datetime
import math
from pathlib import Path
import statistics
from analysis.feedback.reserved import resources as a


def check():
    io, need = a.io, a.need
    completed = io.read(a.OUT / "statistics_complete.json")
    io.verify(a.ROOT, completed["bindings"])
    io.verify(a.ROOT, completed["artifacts"])
    groups = io.read(a.OUT / "resource_groups.json")
    totals, durations, waits = Counter(), [], []
    conditions = defaultdict(Counter)
    features, null_cost, reduced_cost, repeat_usage = (
        Counter(),
        Counter(),
        Counter(),
        Counter(),
    )
    seen = set()
    for path in sorted((a.ASSEMBLY / "assembly").glob("*.jsonl")):
        for r in io.rows(path):
            ident = r["key"], r["condition"], r["repeat"]
            need(ident not in seen, "Duplicate independent resource identity")
            seen.add(ident)
            flat = {
                **{"accounting/" + k: v for k, v in r["accounting"].items()},
                **{"usage/" + k: v for k, v in r["observed_usage"].items()},
                **{"ordinary/" + k: v for k, v in r["ordinary_usage"].items()},
                **{"repair/" + k: v for k, v in r["repair_usage"].items()},
                "predictions": 1,
                "nulls": int(r["result"] is None),
                "unknown": r["administrative_unknown_requests"],
            }
            totals.update(flat)
            conditions[r["condition"]].update(flat)
            durations.extend(r["known_request_durations_s"])
            waits.append(r["retry_backoff_seconds_sum"])
            features["repair_calls_" + str(r["repairs"])] += 1
            features["predictions_needing_repairs"] += r["repairs"] > 0
            features["terminal_invalid_votes"] += r["parse_failures"]
            if r["result"] is None:
                null_cost.update(r["observed_usage"])
                null_cost.update(r["accounting"])
            if r["n_valid"] in (1, 2):
                reduced_cost.update(r["observed_usage"])
                reduced_cost.update(r["accounting"])
            repeat_usage[r["repeat"]] += r["observed_usage"]["total_tokens"]
    need(len(seen) == 149850, "Independent matrix count differs")
    scalar_checks = 0
    for label, ct in [("all", totals)] + [
        ("condition/" + k, v) for k, v in conditions.items()
    ]:
        g = groups[label]
        for prefix, field in [
            ("accounting", "accounting"),
            ("usage", "observed_usage"),
            ("ordinary", "ordinary_usage"),
            ("repair", "repair_usage"),
        ]:
            for k, v in g[field].items():
                need(
                    ct[prefix + "/" + k] == v,
                    "Independent condition total differs: " + label + "/" + k,
                )
                scalar_checks += 1
        need(
            ct["predictions"] == g["counts"]["predictions"]
            and ct["nulls"] == g["counts"]["nulls"]
            and ct["unknown"] == g["counts"]["administrative_unknown_requests"],
            "Independent condition denominator differs",
        )
        scalar_checks += 3
    total = groups["all"]
    need(len(durations) == 464869 and totals["unknown"] == 8, "Duration slots differ")
    need(
        math.isclose(
            math.fsum(durations), total["known_request_seconds_sum"], abs_tol=1e-7
        )
        and math.isclose(
            math.fsum(waits), total["retry_backoff_seconds_sum"], abs_tol=1e-8
        ),
        "Independent request/wait duration differs",
    )
    quantiles = statistics.quantiles(durations, n=100, method="inclusive")
    for field, value in [
        ("p50", statistics.median(durations)),
        ("p95", quantiles[94]),
        ("p99", quantiles[98]),
    ]:
        need(
            math.isclose(value, total["known_request_seconds_" + field], abs_tol=1e-9),
            "Independent quantile differs",
        )

    for prefix, count in [
        ("condition/", 74),
        ("batch/", 16),
        ("mode/", 6),
        ("policy/", 26),
        ("population/", 2),
        ("source/", 3),
        ("source_policy/", 78),
        ("repeat/", 3),
    ]:
        cells = [v for k, v in groups.items() if k.startswith(prefix)]
        need(len(cells) == count, "Wrong resource group count: " + prefix)
        for field in ("accounting", "observed_usage", "ordinary_usage", "repair_usage"):
            summed = Counter()
            for z in cells:
                summed.update(z[field])
            need(
                summed == Counter(total[field]),
                "Non-partition/double-count: " + prefix + field,
            )
            scalar_checks += len(summed)
        need(
            sum(z["counts"]["predictions"] for z in cells) == 149850,
            "Partition denominators differ",
        )
    text = io.read(a.OUT / "text_resource_profiles.json")
    need(
        len(text) == 675
        and all(z["counts"]["predictions"] == 222 for z in text.values()),
        "Text denominator differs",
    )
    need(
        sum(z["observed_usage"]["total_tokens"] for z in text.values())
        == totals["usage/total_tokens"],
        "Text costs differ",
    )
    need(
        features["terminal_invalid_votes"]
        == 3 * total["counts"]["valid_votes_0"]
        + 2 * total["counts"]["valid_votes_1"]
        + total["counts"]["valid_votes_2"]
        == 2301,
        "Invalid vote arithmetic differs",
    )
    timeline = io.read(a.OUT / "execution_timeline.json")
    retry_classes, retry_turns = Counter(), Counter()
    for r in timeline["retry_records"]:
        retry_classes[r["class"]] += 1
        retry_turns["ordinary" if r["turn"] == 0 else "repair"] += 1
    need(
        sum(retry_classes.values()) == sum(retry_turns.values()) == 2060,
        "Retry classification differs",
    )
    for k, v in retry_classes.items():
        need(
            timeline["http_and_transport_statuses"]["transport_error:" + k] == v,
            "Retry/status mismatch",
        )

    dt = datetime.fromisoformat
    previous_audit = None
    whole_parts = []
    for s in timeline["stages"]:
        if previous_audit is not None:
            whole_parts.append(
                (dt(s["first_start_utc"]) - dt(previous_audit)).total_seconds()
            )
        pieces = s["attempt_partition"]["pieces"]
        cursor = s["first_start_utc"]
        for p in pieces:
            need(dt(p["start"]) == dt(cursor), "Gap or overlap inside time partition")
            actual = (dt(p["end"]) - dt(p["start"])).total_seconds()
            need(actual >= 0 and actual == p["seconds"], "Wrong elapsed interval")
            whole_parts.append(actual)
            cursor = p["end"]
        need(dt(cursor) == dt(s["final_summary_utc"]), "Incomplete work window")
        whole_parts.extend(
            [
                (dt(s["stage_complete_utc"]) - dt(cursor)).total_seconds(),
                (
                    dt(s["audit_report_created_utc"]) - dt(s["stage_complete_utc"])
                ).total_seconds(),
            ]
        )
        previous_audit = s["audit_report_created_utc"]
    elapsed = (dt(previous_audit) - dt(timeline["first_start_utc"])).total_seconds()
    need(
        math.isclose(math.fsum(whole_parts), elapsed, abs_tol=1e-6)
        and elapsed == timeline["envelope_seconds"],
        "Independent chronology differs",
    )
    proxy = timeline["completion_proxy"]
    need(
        math.isclose(
            elapsed + proxy["last_audit_report_to_final_receipt_mtime_seconds"],
            proxy["start_to_final_receipt_mtime_seconds"],
            abs_tol=1e-6,
        ),
        "Filesystem proxy interval differs",
    )
    rec = io.read(a.OUT / "recovery_details.json")
    need(
        len(rec) == 2 and all(len(z["records"]) == 8 for z in rec),
        "Recovery matrix differs",
    )
    incident_stats = []
    for z in rec:
        counts = Counter()
        for r in z["records"]:
            for e in r["events"]:
                counts[e["response_class"]] += 1
                counts["all_attempt_slots"] += 1
            need(r["events"][-1]["action"] == "ok", "Recovery failed")
        incident_stats.append({"incident": z["incident"], "counts": dict(counts)})
    need(
        incident_stats[0]["counts"]["all_attempt_slots"] == 16
        and incident_stats[1]["counts"]["all_attempt_slots"] == 48
        and incident_stats[1]["counts"]["success"] == 8,
        "Recovery attempts double-counted",
    )
    local = io.read(a.OUT / "local_prompt_costs.json")["groups"]
    need(
        local["all"]["prompts"] == 49950
        and local["all"]["observed_ordinary_prompt_tokens"]
        == total["ordinary_usage"]["prompt_tokens"],
        "Prompt coverage differs",
    )
    for condition in conditions:
        z = local["condition/" + condition]
        need(
            z["prompts"] == 675
            and groups["condition/" + condition]["counts"]["predictions"] == 2025,
            "Resource normalization denominator differs",
        )
        need(
            z["observed_minus_local_times_nine"]
            == groups["condition/" + condition]["ordinary_usage"]["prompt_tokens"]
            - 9 * z["local_token_sum"],
            "Local/provider distinction differs",
        )
    ledger = io.read(a.OUT / "partial_programme_ledger.json")
    for k, v in ledger["ten_programme_totals"].items():
        need(
            v == sum(p[k] for p in ledger["programmes"]), "Ten-programme total differs"
        )

    amend = io.read(a.PREP / "implementation_amendment_v2.json")
    old = a.ROOT / "analysis/feedback/reserved/resources.py"
    need(
        io.sha(old) == amend["old_script_sha256"]
        and io.sha(Path(a.__file__)) == amend["new_script_sha256"],
        "Implementation amendment changed",
    )
    first = a.ROOT / "experiments/astra/sq3_reserved_part8_results_20261020_v1"
    need(
        {p.name for p in first.iterdir()} == {"started.json"},
        "First partial attempt changed",
    )
    outputs = {
        "status": "part8_resources_independently_reconciled",
        "created_utc": io.utc(),
        "independent_integer_scalar_checks": scalar_checks,
        "all_predictions_reaggregated": len(seen),
        "conditions": 74,
        "resource_partitions": 8,
        "exact_partition_integer_totals": True,
        "request_duration_samples_checked": len(durations),
        "quantiles_checked_with_standard_library_statistics": 3,
        "chronology_contiguous_and_nonoverlapping": True,
        "retry_failure_classes": dict(retry_classes),
        "retry_failures_by_logical_turn": dict(retry_turns),
        "repair_and_validity_counts": dict(features),
        "null_prediction_costs": dict(null_cost),
        "reduced_valid_prediction_costs": dict(reduced_cost),
        "repeat_observed_usage": dict(repeat_usage),
        "recovery_incidents_checked": incident_stats,
        "unknown_duration_sentinels_excluded": 8,
        "partial_v1_preserved": True,
        "new_model_calls": 0,
        "new_training_fits": 0,
        "bindings": {
            io.rel(p): io.sha(p)
            for p in [
                Path(__file__),
                a.OUT / "statistics_complete.json",
                a.PREP / "implementation_amendment_v2.json",
                first / "started.json",
                old,
            ]
        },
    }
    io.verify(a.ROOT, completed["bindings"])
    io.verify(a.ROOT, completed["artifacts"])
    a.write(a.OUT / "independent_resource_check.json", outputs)
    print(
        "Independent resource reconciliation passed:",
        scalar_checks,
        "integer checks,",
        len(durations),
        "request durations; original evidence unchanged.",
        flush=True,
    )


if __name__ == "__main__":
    with a.io.offline():
        check()

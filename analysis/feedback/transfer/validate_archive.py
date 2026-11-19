from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import io
import json
from pathlib import Path
import unittest

from analysis.feedback.transfer import archive as a
from tests.feedback.transfer import test_archive as fixtures


def require(ok, message):
    if not ok:
        raise AssertionError(message)


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()


def runtime_estimate(overview):
    evidence = {}
    attempts = []
    for row in overview["higher_concurrency_attempts"]:
        if row["stage"] == "transfer_04":
            continue
        target = a.inside(a.ROOT, row["attempt"])
        parsed = []
        for name in ("start.json", "summary.json"):
            path = target / name
            expected = overview["verified_metadata_hashes"][a.rel(path)]
            parsed.append(a.read_bound(path, expected))
            evidence[a.rel(path)] = expected
        start, summary = parsed
        require(
            not summary["stopped"] and not summary["errors"],
            "Timing includes an unfinished attempt",
        )
        seconds = (
            datetime.fromisoformat(summary["finished_utc"])
            - datetime.fromisoformat(start["started_utc"])
        ).total_seconds()
        require(
            seconds > 0 and summary["submitted"] == summary["saved_items"],
            "Timing includes reused items",
        )
        attempts.append(
            {
                "stage": row["stage"],
                "predictions": summary["submitted"],
                "seconds": seconds,
                "started_utc": start["started_utc"],
                "finished_utc": summary["finished_utc"],
            }
        )
    require(len(attempts) == 18, "Wrong fully high-concurrency timing sample")
    predictions = sum(r["predictions"] for r in attempts)
    seconds = sum(r["seconds"] for r in attempts)
    last_audit = overview["batch_summaries"][-1]["audited_utc"]
    wall_to_audit = (
        datetime.fromisoformat(last_audit)
        - datetime.fromisoformat(attempts[0]["started_utc"])
    ).total_seconds()
    final_file = overview["completion_file_mtime_utc"]
    final_audit_seconds = (
        datetime.fromisoformat(final_file) - datetime.fromisoformat(last_audit)
    ).total_seconds()
    iterative_root = a.ROOT / "experiments/astra/sq3_iterative_20261012_v1"
    iterative_ready = a.read(iterative_root / "ready.json")
    iterative_plan = a.read_bound(
        iterative_root / "plan.json", iterative_ready["plan_sha256"]
    )
    protocol_path = iterative_root / "STUDY_PROTOCOL.md"
    require(
        a.sha(protocol_path) == iterative_plan["protocol_sha256"],
        "Repeated-round protocol changed",
    )
    capacity_report = (
        a.ROOT / "Thesis_Log/astra_result/astra_server_capacity_results_20261021_v1.md"
    )
    for path in (
        iterative_root / "plan.json",
        iterative_root / "ready.json",
        protocol_path,
        capacity_report,
    ):
        evidence[a.rel(path)] = a.sha(path)
    ordinary_remaining = iterative_plan["ordinary_calls"]
    require(
        ordinary_remaining == 777672
        and iterative_plan["predictions"] == ordinary_remaining // 3,
        "Remaining SQ3 call scope differs",
    )

    return {
        "status": "conditional_planning_estimate_not_a_launch",
        "input_bindings": evidence,
        "scope": "SQ3 repeated rounds only; excludes the two remaining SQ2 Qwen studies.",
        "ordinary_calls_remaining": ordinary_remaining,
        "complete_three_vote_predictions_remaining": ordinary_remaining // 3,
        "capacity_confirmation_ordinary_equivalent_calls_per_minute": 356.37,
        "capacity_confirmation_generation_hours": ordinary_remaining / 356.37 / 60,
        "later_transfer_batches": attempts,
        "later_transfer_predictions": predictions,
        "later_transfer_active_generation_hours": seconds / 3600,
        "later_transfer_ordinary_equivalent_calls_per_minute": predictions
        * 3
        / (seconds / 60),
        "later_transfer_batch_audit_inclusive_hours": wall_to_audit / 3600,
        "transfer_final_audit_lag_hours_filesystem_based": final_audit_seconds / 3600,
        "projected_generation_hours_at_sustained_transfer_rate": ordinary_remaining
        / (predictions * 3)
        * seconds
        / 3600,
        "projected_batch_audit_inclusive_hours": ordinary_remaining
        / (predictions * 3)
        * wall_to_audit
        / 3600,
        "projected_final_audit_hours_linear_file_count_scenario": final_audit_seconds
        / 3600
        * (ordinary_remaining / 3)
        / 189096,
        "planning_continuous_machine_days": [2, 3],
        "analysis_and_synthesis": "Several additional working days; roughly a week overall is a planning target after readiness, not a measured guarantee.",
        "conditions": [
            "128 must first be prepared and checked for the unstarted repeated-round launcher.",
            "Same university endpoint capacity, continuous laptop uptime, similar responses and disk audit speed.",
            "Sequential source screening/round audits, repairs, interruptions or other server users can add time.",
            "Includes one remaining Qwen study; up to 72 small CPU classifier fits and scientific review are separate.",
            "The 128 setting is chosen concurrency, not an administrator-enforced hard rate limit.",
        ],
        "new_Qwen_calls": 0,
    }


def main():
    output = a.OUT / "independent_matrix_check.json"
    require(not output.exists(), "Keep the existing verification evidence")
    completed = a.read(a.OUT / "assembly_complete.json")
    require(
        completed["status"] == "complete_from_audited_receipts_unscored",
        "Assembly is not complete",
    )
    for name, value in {
        **completed["input_bindings"],
        **completed["artifacts"],
    }.items():
        require(
            a.sha(a.inside(a.ROOT, name)) == value,
            "Changed analysis input/artifact: " + name,
        )
    log = io.StringIO()
    tests = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(fixtures)
    )
    require(tests.wasSuccessful(), "New-reader fixture failed")
    a.write(
        a.OUT / "focused_tests.json",
        {
            "tests_run": tests.testsRun,
            "passed": True,
            "output": log.getvalue(),
            "test_script_sha256": a.sha(Path(fixtures.__file__)),
            "reader_sha256": a.sha(Path(a.__file__)),
        },
    )
    science = a.read(a.STUDY / "scientific_plan_v1.json")
    roster = list(a.rows(a.STUDY / "roster.jsonl"))
    require(list(a.rows(a.OUT / "roster.jsonl")) == roster, "Copied roster differs")
    conditions = science["conditions"]
    ci = {c: i for i, c in enumerate(conditions)}
    position = {r["key"]: i for i, r in enumerate(roster)}
    slot_hashes = {}
    counts, accounting, usage = Counter(), Counter(), Counter()
    missing_groups = defaultdict(Counter)
    nulls_by_input, under3_by_input = Counter(), Counter()
    batch_by_input, period_by_input = {}, {}
    unknown = 0
    for batch in range(1, 23):
        stage = f"transfer_{batch:02d}"
        for row in a.rows(a.OUT / "assembly" / (stage + ".jsonl")):
            index = position[row["key"]]
            require(
                index == row["input_index"] and row["stage"] == stage,
                "Incorrect row/index/batch",
            )
            require(
                row["population"] == roster[index]["population"],
                "Incorrect row population",
            )
            slot = index, ci[row["condition"]], row["repeat"]
            require(
                slot not in slot_hashes and row["repeat"] in (0, 1, 2),
                "Duplicate/unplanned slot",
            )
            slot_hashes[slot] = digest(
                {k: row[k] for k in ("result", "n_valid", "uncertain")}
            )
            counts["predictions"] += 1
            counts["nulls"] += int(row["result"] is None)
            counts["valid_votes_" + str(row["n_valid"])] += 1
            counts["gate_ties"] += int(row["uncertain"])
            require(
                (row["result"] is None) == (row["n_valid"] == 0),
                "Null/valid-vote mismatch",
            )
            for name in (
                "gate_normalised",
                "normalisations",
                "repairs",
                "parse_failures",
            ):
                counts[name] += row[name]
            for name, value in row["accounting"].items():
                accounting[name] += value
            for name, value in row["observed_usage"].items():
                usage[name] += value
            for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
                require(
                    row["observed_usage"].get(name, 0)
                    == row["ordinary_usage"].get(name, 0)
                    + row["repair_usage"].get(name, 0),
                    "Ordinary/repair token partition differs",
                )
            missing_groups[row["population"], row["condition"], row["repeat"]][
                row["n_valid"]
            ] += 1
            if row["result"] is None:
                nulls_by_input[index] += 1
            if row["n_valid"] < 3:
                under3_by_input[index] += 1
            require(batch_by_input.get(index, stage) == stage, "Input crosses batches")
            batch_by_input[index] = stage
            period_by_input[index] = row["execution_period"]
            unknown += row["administrative_unknown_requests"]
    require(len(slot_hashes) == len(roster) * 12 == 189096, "Wrong final slot count")
    for name, value in [
        ("counts", counts),
        ("accounting", accounting),
        ("observed_usage", usage),
    ]:
        require(value == Counter(completed[name]), "Independent total differs: " + name)
    require(unknown == 0, "Unexpected administrative unknown duration")
    ready = a.read(a.OUT / "analysis_ready.json")
    replay = set(science["mandatory_replay_keys"])
    masks = defaultdict(Counter)
    clusters = defaultdict(set)
    fine = defaultdict(Counter)
    support_labels = defaultdict(Counter)
    source_counts, period_counts, null_gold_source = Counter(), Counter(), Counter()
    seen = set()
    for index, row in enumerate(a.rows(a.OUT / "matrix.jsonl")):
        r = roster[index]
        require(
            row["input_index"] == index
            and row["key"] == r["key"]
            and index not in seen,
            "Matrix order/identity differs",
        )
        seen.add(index)
        for name in (
            "population",
            "source",
            "source_file",
            "inference_cluster",
            "functional_cluster",
            "text_cluster",
            "overlap_cluster",
        ):
            require(row[name] == r[name], "Matrix metadata differs: " + name)
        require(
            type(row["gold_hate"]) is bool and row["gold_hate"] == r["gate"],
            "Gold gate differs",
        )
        require(
            row["stage"] == batch_by_input[index]
            and row["execution_period"] == period_by_input[index],
            "Batch provenance differs",
        )
        require(
            len(row["predictions"]) == 4
            and all(len(v) == 3 for v in row["predictions"]),
            "Wrong prediction axes",
        )
        all_valid = True
        all_three = True
        for condition in range(4):
            for repeat in range(3):
                p = row["predictions"][condition][repeat]
                require(
                    digest(p) == slot_hashes.pop((index, condition, repeat)),
                    "Matrix/prediction content differs",
                )
                all_valid = all_valid and p["result"] is not None
                all_three = all_three and p["n_valid"] == 3
        excluded = row["key"] in replay
        expected_masks = {
            "common_valid": all_valid,
            "strict_all_votes": all_three,
            "replay_exclusion_member": excluded,
            "common_valid_exclude_replay": all_valid and not excluded,
            "strict_exclude_replay": all_three and not excluded,
        }
        require(
            row["masks"] == expected_masks,
            "Mask differs from raw assembled availability",
        )
        require(
            (index in nulls_by_input) == (not all_valid)
            and (index in under3_by_input) == (not all_three),
            "Distinct affected-input counts differ",
        )
        endpoint = r["population"] + ("_FNR" if r["gate"] else "_FPR")
        masks[endpoint]["full_inputs"] += 1
        clusters[endpoint].add(r["inference_cluster"])
        for name, value in expected_masks.items():
            masks[endpoint][name] += int(value)
        for field in ("hate_types", "target_groups", "severity", "legal"):
            value = r.get(field)
            known = value is not None
            eligible = known and (field == "legal" or r["gate"])
            require(
                row["gold_" + field] == value
                and row["known_fine_gold"][field] == known
                and row["eligible_fine_gold"][field] == eligible,
                "Missing/known fine gold differs",
            )
            if eligible:
                key = (r["population"], field)
                fine[key]["full"] += 1
                fine[key]["common_valid"] += int(all_valid)
                fine[key]["strict_all_votes"] += int(all_three)
                support_labels[key].update(
                    value if isinstance(value, list) else [value]
                )
        source_counts[r["population"] + "/" + r["source"]] += 1
        period_counts[row["execution_period"]] += 1
        if not all_valid:
            null_gold_source[endpoint + "/" + r["source"]] += 1
    require(len(seen) == 15758 and not slot_hashes, "Incomplete matrix coverage")
    independent_masks = {
        key: dict(value, full_inference_clusters=len(clusters[key]))
        for key, value in masks.items()
    }
    require(
        independent_masks == ready["endpoint_masks"],
        "Endpoint denominator table differs",
    )
    require(
        len(nulls_by_input) == ready["inputs_with_any_null"]
        and len(under3_by_input) == ready["inputs_with_any_under_three_votes"],
        "Affected-input summary differs",
    )
    for row in ready["missingness_by_population_condition_repeat"]:
        values = missing_groups.pop(
            (row["population"], row["condition"], row["repeat"])
        )
        require(
            {str(i): values[i] for i in range(4)} == row["valid_vote_counts"]
            and sum(values.values()) == row["slots"],
            "Missingness group differs",
        )
    require(not missing_groups, "Omitted missingness group")
    for population, fields in science["fixed_support"].items():
        for field, expected in fields.items():
            values = support_labels[population, field]
            independently = {
                "rows": fine[population, field]["full"],
                "positive_support": dict(values),
                "support_ge_10": sorted(
                    label for label, n in values.items() if n >= 10
                ),
            }
            require(
                independently == expected, "Frozen support failed independent count"
            )
    overview = a.read(a.OVERVIEW)
    timing = runtime_estimate(overview)
    a.write(a.OUT / "remaining_sq3_timing.json", timing)
    a.write(
        output,
        {
            "status": "independent_matrix_check_pass",
            "checked_utc": a.utc(),
            "assembly_complete_sha256": a.sha(a.OUT / "assembly_complete.json"),
            "verifier_sha256": a.sha(Path(__file__)),
            "checked_predictions": 189096,
            "checked_inputs": 15758,
            "counts": dict(counts),
            "accounting": dict(accounting),
            "observed_usage": dict(usage),
            "endpoint_masks": independent_masks,
            "sources": dict(source_counts),
            "execution_period_inputs": dict(period_counts),
            "inputs_with_any_null": len(nulls_by_input),
            "null_slots_per_affected_input": dict(Counter(nulls_by_input.values())),
            "inputs_with_any_under_three_votes": len(under3_by_input),
            "null_affected_inputs_by_endpoint_source": dict(null_gold_source),
            "fine_support_masks": {p + "/" + f: dict(n) for (p, f), n in fine.items()},
            "fixture_tests_passed": tests.testsRun,
            "new_Qwen_calls": 0,
            "new_fits": 0,
            "raw_reparse": False,
            "gold_scoring": False,
            "scope": "Independent matrix-to-assembly joins, all identities/slots, roster/gold/cluster metadata, availability masks, fixed support, resource totals and current bindings. Original completed raw audits are reused.",
        },
    )
    artifacts = dict(completed["artifacts"])
    for name in (
        "assembly_complete.json",
        "focused_tests.json",
        "remaining_sq3_timing.json",
        "independent_matrix_check.json",
    ):
        artifacts[a.rel(a.OUT / name)] = a.sha(a.OUT / name)
    a.write(
        a.OUT / "complete.json",
        {
            "status": "transfer_part1_complete_ready_for_part2",
            "completed_utc": a.utc(),
            "artifacts": artifacts,
            "input_bindings": completed["input_bindings"],
            "verification_source": {a.rel(Path(__file__)): a.sha(Path(__file__))},
            "new_Qwen_calls": 0,
            "new_fits": 0,
            "raw_reparse": False,
            "gold_scoring": False,
            "next_step": "Part 2: the four frozen false-positive/false-negative endpoints; no new generation.",
        },
    )
    print(
        json.dumps(
            {
                "status": "Part 1 verified",
                "endpoint_masks": independent_masks,
                "null_affected_inputs": len(nulls_by_input),
                "strict_affected_inputs": len(under3_by_input),
                "timing_active_calls_per_min": timing[
                    "later_transfer_ordinary_equivalent_calls_per_minute"
                ],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    with a.offline():
        main()

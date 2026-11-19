from collections import Counter, defaultdict
from fractions import Fraction as F
import json
import math
import re
import numpy as np
from analysis.feedback.reserved import cross_checks as a

io, c, need = a.io, a.c, a.need
ROOT, OUT, PARTS = a.ROOT, a.OUT, a.PARTS


def category(gold, pred):
    extra, missing = pred & (127 ^ gold), gold & (127 ^ pred)
    return (
        "both_extra_and_missing"
        if extra and missing
        else "only_extra" if extra else "only_missing" if missing else "exact"
    )


def transition_phase(data, plan, specs):
    saved = io.read(PARTS[6] / "transition_summary.json")
    definitions = {
        s["id"].rsplit("/", 1)[0]: s
        for s in specs.values()
        if s["endpoint"] == "FPR" and s["family"] != "primary"
    }
    need(len(definitions) == 64, "Transition comparison coverage")
    inventory = {}
    for key, obj in saved.items():
        cohort, mask, rest = key.split("/", 2)
        sid, ep = rest.rsplit("/", 1)
        spec = {**definitions[sid], "endpoint": ep}
        point = data.rate(spec, excluded=a.exclusions(plan, cohort), mask=mask)
        need(point["common_ids"] == obj["common_ids"], key + " transition mask")
        ix = point["indices"]
        truth = ep == "FNR"
        pairs = [
            (t, controls[0] if len(controls) == 1 else controls[j])
            for j, t in enumerate(spec["treatment_conditions"])
            for controls in [spec["control_conditions"]]
        ]
        cnt = Counter()
        directions = Counter()
        for i in ix:
            text = Counter()
            for treatment, control in pairs:
                for rep in range(3):
                    ti, ci = data.columns[treatment, rep], data.columns[control, rep]
                    t, co = int(data.hate[i, ti]), int(data.hate[i, ci])
                    label = (
                        ("unchanged_correct" if t == truth else "regression")
                        if co == truth
                        else ("fix" if t == truth else "unchanged_error")
                    )
                    cnt[label] += 1
                    text[label] += 1
                    cnt["strict_valid_pairs"] += int(
                        data.valid[i, ti] == 3 and data.valid[i, ci] == 3
                    )
                    cnt["both_gates_positive"] += int(t == co == 1)
                    ts, cs = int(data.types[i, ti]), int(data.types[i, ci])
                    cnt["types_changed"] += int(ts != cs)
                    gold = int(data.gold_types[i])
                    if gold < 0:
                        cnt["unknown_type_pairs"] += 1
                        continue
                    cnt["known_type_pairs"] += 1
                    type_label = (
                        ("unchanged_correct" if ts == gold else "regression")
                        if cs == gold
                        else ("fix" if ts == gold else "unchanged_error")
                    )
                    cnt["type_" + type_label] += 1
                    if t == co == 1:
                        cnt["both_positive_type_" + type_label] += 1
                    cnt[
                        "category_" + category(gold, cs) + "->" + category(gold, ts)
                    ] += 1
                    cnt["correct_labels_added"] += (ts & gold & (127 ^ cs)).bit_count()
                    cnt["correct_labels_lost"] += (cs & gold & (127 ^ ts)).bit_count()
                    cnt["extra_labels_added"] += (
                        ts & (127 ^ gold) & (127 ^ cs)
                    ).bit_count()
                    cnt["extra_labels_removed"] += (
                        cs & (127 ^ gold) & (127 ^ ts)
                    ).bit_count()
            delta = text["regression"] - text["fix"]
            directions[
                (
                    "fewer_errors"
                    if delta < 0
                    else "more_errors" if delta > 0 else "same_average_errors"
                )
            ] += 1
            directions["any_binary_changes"] += int(
                text["fix"] + text["regression"] > 0
            )
            directions["equal_average_with_changes"] += int(
                delta == 0 and text["fix"] > 0
            )
            directions["offsetting_binary_changes"] += int(
                text["fix"] > 0 and text["regression"] > 0
            )
            directions["all_pairs_unchanged_error"] += int(
                text["unchanged_error"] == 3 * len(pairs)
            )
        n = len(ix) * 3 * len(pairs)
        cnt.update({"eligible_pairs": n, "valid": n, "unique_texts": len(ix)})
        for field, value in obj.items():
            if field in (
                "common_ids",
                "per_text_direction",
                "binary_error_difference",
                "type_exact_difference",
            ):
                continue
            a.close(cnt[field], value, key + "/" + field)
        for field in set(cnt) - set(obj):
            need(cnt[field] == 0, key + " unreported nonzero transition")
        for field in set(directions) | set(obj["per_text_direction"]):
            a.close(
                directions[field],
                obj["per_text_direction"].get(field, 0),
                key + "/text/" + field,
            )
        a.close(
            point["difference"],
            obj["binary_error_difference"],
            key + " rate/transition join",
        )
        type_delta = (
            F(cnt["type_fix"] - cnt["type_regression"], cnt["known_type_pairs"])
            if cnt["known_type_pairs"]
            else None
        )
        a.close(type_delta, obj["type_exact_difference"], key + " exact-set join")
        inventory[key] = dict(
            common_n=len(ix),
            dependent_pairs=n,
            fix=cnt["fix"],
            regression=cnt["regression"],
            difference_exact=str(point["difference"]),
            per_text_direction=dict(directions),
        )
    need(len(inventory) == 768, "Transition mask coverage")
    prior = io.read(PARTS[6] / "independent_readback.json")
    need(
        prior["pairs_checked"] == 384750
        and prior["text_records_checked"] == 43200
        and prior["context_rows_checked"] == 49950,
        "Prior exhaustive case ledger coverage",
    )
    a.write(
        OUT / "independent_transitions.json",
        dict(
            status="all_transition_summaries_rebuilt_and_joined_to_rates_and_exact_type_sets",
            created_utc=io.utc(),
            comparison_packages=64,
            mask_endpoint_summaries=768,
            summaries=inventory,
            original_pair_and_text_ledger_proof_reused={
                "pairs": 384750,
                "text_records": 43200,
                "context_rows": 49950,
            },
            bindings={
                io.rel(PARTS[6] / f): io.sha(PARTS[6] / f)
                for f in (
                    "transition_summary.json",
                    "independent_readback.json",
                    "context_check.json",
                    "case_selection.json",
                    "case_packets.json",
                    "cases_complete.json",
                )
                if (PARTS[6] / f).exists()
            },
            new_model_calls=0,
        ),
    )
    print(
        "All 768 transition summaries independently rebuilt and joined to rates and exact type sets.",
        flush=True,
    )


def resource_phase(data):
    groups = io.read(PARTS[8] / "resource_groups.json")
    counters = defaultdict(Counter)
    durations = []
    backoffs = []
    for path in sorted((a.ASSEMBLY / "assembly").glob("reserved_slot_*.jsonl")):
        for record in io.rows(path):
            cond = record["condition"]
            if cond.startswith("reference"):
                policy = "policy/" + cond
            else:
                mode, selection = cond.rsplit("_s", 1)[0].split("__")
                policy = "policy/" + mode + "/" + selection
            counts = {
                "predictions": 1,
                "nulls": int(record["result"] is None),
                "valid_votes_" + str(record["n_valid"]): 1,
                "repairs": record["repairs"],
                "parse_failures": record["parse_failures"],
                "administrative_unknown_requests": record[
                    "administrative_unknown_requests"
                ],
            }
            values = {"counts/" + k: v for k, v in counts.items()}
            for section in (
                "accounting",
                "observed_usage",
                "ordinary_usage",
                "repair_usage",
            ):
                values.update(
                    {section + "/" + k: v for k, v in record[section].items()}
                )
            for key in ("all", "condition/" + cond, policy):
                counters[key].update(values)
            durations.extend(record["known_request_durations_s"])
            backoffs.append(record["retry_backoff_seconds_sum"])
    actual_keys = {}
    for key, values in counters.items():

        lookup = (
            key
            if key in groups
            else key.replace("policy/reference", "condition/reference")
        )
        need(lookup in groups, "Missing resource group " + key)
        for field, value in values.items():
            section, name = field.split("/")
            need(value == groups[lookup][section].get(name, 0), key + "/" + field)
        actual_keys[key] = dict(values)
    all_group = groups["all"]
    need(
        len(durations) == 464869
        and counters["all"]["counts/administrative_unknown_requests"] == 8,
        "Unknown duration sentinel handling",
    )
    need(
        abs(math.fsum(durations) - all_group["known_request_seconds_sum"]) < 1e-6,
        "Duration sum mismatch",
    )
    need(
        abs(
            math.fsum(backoffs)
            - io.read(PARTS[8] / "resource_summary.json")["retry_backoff_seconds_sum"]
        )
        < 1e-7,
        "Backoff sum",
    )
    for q, field in ((0.5, "p50"), (0.95, "p95"), (0.99, "p99")):
        a.close(
            float(np.quantile(durations, q)),
            all_group["known_request_seconds_" + field],
            "Request quantile " + field,
        )
    local = io.read(PARTS[8] / "local_prompt_costs_v2.json")["groups"]
    local_verified = 0
    for key, obj in local.items():
        if key not in groups:
            continue
        need(
            obj["local_token_sum"] * 9
            == groups[key]["ordinary_usage"]["prompt_tokens"],
            "Corrected local/provider prompt usage mismatch " + key,
        )
        local_verified += 1
    need(local_verified == 107, "Local prompt group coverage")
    account = all_group["accounting"]
    need(
        account["ordinary_logical_calls"]
        + account["repair_logical_calls"]
        + account["retry_attempts"]
        == account["post_attempts"],
        "Ordinary/repair/retry conservation",
    )
    need(
        account["post_attempts"] == account["http_success_with_text"] + 2052 + 8,
        "Outcome conservation",
    )
    ledger = io.read(PARTS[8] / "partial_programme_ledger.json")
    a.write(
        OUT / "resource_crosscheck.json",
        dict(
            status="all_condition_policy_costs_and_corrected_prompt_groups_reconciled",
            created_utc=io.utc(),
            fresh_groups=len(counters),
            groups=actual_keys,
            local_prompt_groups=local_verified,
            request_durations_recounted=len(durations),
            known_request_seconds=math.fsum(durations),
            actual_successes=462817,
            measured_transport_exceptions=2052,
            unknown_remote_outcomes=8,
            total_accounted_attempt_slots=464877,
            reported_tokens=426017311,
            existing_original_prompt_check_reused=49950,
            original_resource_partition_and_timeline_proof_reused=True,
            partial_ledger_totals=ledger["ten_programme_totals"],
            bindings={
                io.rel(PARTS[8] / f): io.sha(PARTS[8] / f)
                for f in (
                    "resource_groups.json",
                    "local_prompt_costs_v2.json",
                    "prompt_token_check.json",
                    "independent_resource_check_v2.json",
                    "execution_timeline.json",
                    "partial_programme_ledger.json",
                )
            },
            new_model_calls=0,
        ),
    )
    print(
        f"Fresh resource joins passed: {len(counters)} overall/condition/policy groups, 107 prompt groups, all durations.",
        flush=True,
    )


def main():
    data, plan, specs, _ = a.load_data()
    transition_phase(data, plan, specs)
    resource_phase(data)


if __name__ == "__main__":
    with io.offline():
        main()

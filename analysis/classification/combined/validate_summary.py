import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
from analysis.classification.combined import summary as b

checks = 0
largest = 0.0


def equal(x, y, label):
    global checks, largest
    if isinstance(x, dict):
        assert isinstance(y, dict) and x.keys() == y.keys(), label + " keys"
        for k in x:
            equal(x[k], y[k], label + "/" + str(k))
    elif isinstance(x, list):
        assert isinstance(y, list) and len(x) == len(y), label + " length"
        for i, (u, v) in enumerate(zip(x, y)):
            equal(u, v, label + "/" + str(i))
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        assert (
            isinstance(y, (int, float)) and math.isfinite(x) and math.isfinite(y)
        ), label
        diff = abs(x - y)
        largest = max(largest, diff)
        assert diff <= max(2e-11, abs(x) * 2e-13), (label, x, y)
        checks += 1
    else:
        assert x == y, (label, x, y)


def sign(ci):
    if ci is None:
        return "unsupported"
    assert len(ci) == 2 and ci[0] <= ci[1]
    if ci[0] <= 0 <= ci[1]:
        return "crosses_or_touches_zero"
    return "above_zero" if ci[0] > 0 else "below_zero"


def run():
    contract = b.read(b.BASE / "analysis_contract.json")
    b.a.verify(contract["sources"])
    for path, v in contract["preimages"].items():
        assert b.a.g.sha(b.ROOT / v["copy"]) == v["sha256"]

    seals = [
        b.a.BASE / "part1_complete.json",
        b.P / "part2_complete.json",
        b.Q / "part3_complete.json",
        b.E / "sq2_gpu_synthesis_20261104_v1/part9_complete.json",
        b.E / "sq2_broad_synthesis_20261104_v1/part9_complete.json",
        b.M / "complete.json",
        b.E / "sq3_final_synthesis_20261027_v1/publication_complete.json",
    ]
    mutable = {
        "Thesis_Log/astra_session.md",
        "Thesis_Log/astra_review.md",
        "Thesis_Log/astra_master_research_journal.md",
        "Thesis_Log/astra_master_research_journal.html",
        "Thesis_Log/astra_writing_updates.json",
        "Thesis_Log/astra_writing_sync.md",
        "thesis_writing_handover.md",
    }
    bound = {}
    skipped = []
    for p in seals:
        s = b.read(p)
        for file, sha in s.get("outputs", s.get("artifacts", {})).items():
            if file in mutable:
                skipped.append(
                    dict(
                        seal=b.a.rel(p),
                        path=file,
                        reason="Mutable navigation/writer state, not immutable scientific evidence.",
                    )
                )
                continue
            assert file not in bound or bound[file] == sha, "Conflicting seals " + file
            bound[file] = sha
    amendment_path = (
        b.E / "sq3_final_synthesis_20261027_v1/status_readback_amendment_v2.json"
    )
    amendment = b.read(amendment_path)
    historical_path = (
        "Thesis_Log/astra_result/astra_remaining_work_before_writing_20261027_v102.md"
    )
    assert bound[historical_path] == amendment["before"][historical_path]
    assert (
        b.a.g.sha(b.ROOT / historical_path)
        == amendment["current_outputs"][historical_path]
    )
    bound[historical_path] = amendment["current_outputs"][historical_path]
    bound[b.a.rel(amendment_path)] = b.a.g.sha(amendment_path)
    b.a.verify(bound)
    ci = b.read(b.BASE / "confidence_inventory.json")
    si = b.read(b.BASE / "sensitivity_inventory.json")
    orig = b.read(b.P / "original_qwen_contrasts.json")
    for z, s in zip(ci["broad"], orig):
        equal(z["original"], s["original"], "broad-copy")
        assert z["relation"] == sign(z["interval"])
        if z["difference"] is not None:
            equal(
                z["difference"],
                s["original"]["left_common_mean"] - s["original"]["right_common_mean"],
                "broad-point",
            )
    raw = b.read(b.BR / "sensitivities.json")
    changes = []
    for z, s in zip(si["broad"], raw):
        equal(z["original"], s, "sensitivity-copy")
        intv = s["result"].get("family_interval") or s["result"].get("ci95")
        assert z["relation"] == sign(intv) and z["original_relation"] == sign(
            s["original_interval"]
        )
        assert z["classification_changed"] == (
            sign(intv) != sign(s["original_interval"])
        )
        if s["result"].get("mean_difference") is None:
            assert z["point_change"] is None and s["original_difference"] is None
        else:
            equal(
                z["point_change"],
                s["result"]["mean_difference"] - s["original_difference"],
                "point-change",
            )
        if z["classification_changed"]:
            changes.append(z)
    assert len(changes) == 3
    required = {
        "required_sampling",
        "core_exclusion",
        "replay_exclusion",
        "whitespace_exclusion",
    }
    assert not [z for z in changes if z["scenario"] in required]
    assert not [z for z in changes if z["right"] == "zero_shot"]
    assert len([z for z in ci["broad"] if z["role"] == "primary"]) == 4
    assert len([z for z in ci["matched"] if z["role"] == "primary"]) == 6
    assert len(ci["gpu"]) == 24
    matched = b.read(b.M / "metrics.json")["results"]
    for z in ci["matched"]:
        src = matched[z["endpoint"]]["contrasts"][z["control"]]
        equal(z["original"], src, "matched-copy")
        equal(
            z["difference"],
            src["left_common_mean"] - src["right_common_mean"],
            "matched-point",
        )
        assert z["relation"] == sign(z["interval"])
    for z in si["matched"]:
        equal(
            z["strict"],
            matched[z["endpoint"]]["robustness"][z["control"]].get("three_valid_votes"),
            "strict-copy",
        )
        if z["role"] == "primary":
            assert z["strict_relation"] == z["original_relation"]
    primary = b.read(b.GM / "primary_comparisons.json")["comparisons"]
    for z, s in zip(ci["gpu"], primary):
        equal({k: z[k] for k in s}, s, "gpu-copy")
        equal(z["difference"], statistics.mean(s["pool_differences"]), "equal-pool")
        equal(z["difference"], s["left"] - s["right"], "gpu-delta")
        assert z["relation"] == sign(s["family_interval"])
        assert s["excludes_zero"] == (
            sign(s["family_interval"]) != "crosses_or_touches_zero"
        )
    combinations = 0
    for z in si["gpu_finite"]:
        poolmeans = []
        for p in z["pools"]:
            dd = [u - v for u, v in zip(p["left_values"], p["right_values"])]
            equal(p["differences"], dd, "finite-differences")
            equal(p["mean"], statistics.mean(dd), "finite-poolmean")
            poolmeans.append(statistics.mean(dd))
        equal(z["difference"], statistics.mean(poolmeans), "finite-grandmean")
        if "one_seed_per_pool" not in z:
            assert z["left"] == "frozen_roberta" and all(
                p["n"] == 1 for p in z["pools"]
            )
        if z.get("one_seed_per_pool"):
            dd = []
            for row in z["one_seed_per_pool"]:
                val = statistics.mean(
                    p["differences"][p["seeds"].index(seed)]
                    for p, seed in zip(z["pools"], row["seeds"])
                )
                equal(row["difference"], val, "finite-combination")
                dd.append(val)
                combinations += 1
            stats = z["combination_statistics"]
            equal(stats["minimum"], min(dd), "finite-min")
            equal(stats["maximum"], max(dd), "finite-max")
            equal(stats["mean"], statistics.mean(dd), "finite-mean")
            equal(stats["positive"], sum(x > 0 for x in dd), "finite-positive")
    for z in si["gpu"]:
        assert z["relation"] == sign([z["low"], z["high"]])
    for z in ci["earlier"]:
        assert z["relation"] == sign(z["interval"])
    ledger = b.read(b.BASE / "partial_programme_ledger.json")
    old = b.read(b.M / "partial_programme_ledger.json")
    equal(ledger["programmes"][:-1], old["programmes"], "old14")
    assert len({z["programme"] for z in ledger["programmes"]}) == 15
    for key, total in ledger["totals"].items():
        equal(total, sum(row[key] for row in ledger["programmes"]), "sum15/" + key)
    row = ledger["programmes"][-1]
    equal(
        row["ordinary_slots"] + row["repair_slots"],
        row["successful_responses"],
        "broad-logical",
    )
    equal(
        row["successful_responses"] + row["retry_or_recovered_failure_attempts"],
        row["attempt_slots"],
        "broad-POST",
    )
    res = b.read(b.BASE / "resource_reconciliation.json")
    equal(res["remote_totals"], ledger["totals"], "resource-ledger")
    equal(
        res["historical_SQ3"],
        b.read(b.E / "sq3_final_synthesis_20261027_v1/resource_reconciliation.json"),
        "SQ3-preserved",
    )
    cond = res["broad_condition_costs"]
    assert len(cond) == 3
    equal(sum(v["predictions"] for v in cond), 194643, "predictions")
    equal(
        sum(v["metrics"]["total_tokens"]["sum"] for v in cond),
        row["reported_tokens"],
        "condition-tokens",
    )
    for v in cond:
        equal(
            v["metrics"]["total_tokens"]["sum"] / v["predictions"],
            v["metrics"]["total_tokens"]["mean"],
            "tokens-per-answer",
        )
        equal(
            v["predictions"], v["valid_predictions"] + v["null_predictions"], "validity"
        )
    fm = res["gpu_main"]["fit_totals"]
    for name, v in fm.items():
        equal(sum(v["states"].values()), v["slots"], "state-partition")
        equal(
            v["positive_updates"] + v["accepted_zero_rate_steps"],
            v["successful_steps"],
            "steps",
        )
        equal(v["successful_steps"] + v["skipped_steps"], v["planned_steps"], "planned")
        equal(
            v["loop_worker_seconds"] + v["outside_loop_worker_seconds"],
            v["total_worker_seconds"],
            "timers",
        )
        equal(v["timed_fits"] + v["untimed_slots"], v["slots"], "slots")
    tm = res["gpu_main"]["timeline"]
    equal(tm["active_seconds"] + tm["gap_seconds"], tm["calendar_seconds"], "walltime")
    subsets = res["gpu_main"]["executed_search_subsets"]
    equal(
        sum(
            subsets[k]["total_worker_seconds"]
            for k in ["one_rate_all_durations_losses", "extra_two_rates"]
        ),
        fm["search"]["total_worker_seconds"],
        "searchpartition",
    )
    store = res["gpu_main"]["storage"]
    equal(
        store["weight_bytes"] + store["nonweight_unique_logical_bytes"],
        store["unique_logical_bytes"],
        "logicalbytes",
    )
    equal(
        sum(v["payload_bytes"] for v in store["pools"]),
        store["total_payload_bytes"],
        "payload",
    )
    equal(
        store["weight_bytes"] / store["unique_logical_bytes"],
        store["weight_share_of_unique_logical"],
        "weightfraction",
    )
    equal(res["gpu_main"]["inference_head_rows"], 530 * 19082, "inferenceunits")

    views = b.read(b.BASE / "integrated_result_views.json")
    for ep, groups in views["performance"].items():
        src = b.read(b.P / "endpoints" / ep / "scores.json")["common_nine"]["groups"]
        for k, v in groups.items():
            equal(v, src[k], "performance-view")
    for ep, groups in views["quality"].items():
        src = b.read(b.Q / "endpoints" / (ep.replace("/", "__") + ".json"))[
            "common_nine"
        ]["groups"]
        for k, v in groups.items():
            equal(v, src[k], "quality-view")
    for ep, v in views["joint"].items():
        src = b.read(b.Q / "joint_quality.json")[ep]
        for k, z in v["groups"].items():
            equal(z, src["common_nine"]["groups"][k], "joint-view")

    fixtures = []
    for interval, want in [
        (None, "unsupported"),
        ([0, 1], "crosses_or_touches_zero"),
        ([-1, 0], "crosses_or_touches_zero"),
        ([-1, 1], "crosses_or_touches_zero"),
        ([1e-12, 1], "above_zero"),
        ([-1, -1e-12], "below_zero"),
        ([0, 0], "crosses_or_touches_zero"),
    ]:
        assert b.relation(interval) == want
        fixtures.append(dict(case=str(interval), expected=want))
    for invalid in [[2, 1], [float("nan"), 1], [0, float("inf")]]:
        try:
            b.relation(invalid)
        except Exception:
            fixtures.append(dict(case=str(invalid), expected="reject"))
        else:
            raise AssertionError("Bad interval accepted")
    assert [
        z
        for z in ci["earlier"]
        if z["study"] == "expanded_shorter"
        and z["relation"] != "crosses_or_touches_zero"
    ] == []
    assert [
        z
        for z in ci["broad"]
        if z["endpoint"] == "de_test/target_group" and z["difference"] is not None
    ] == []
    assert ledger["administrative_unknown_outcomes_in_inherited_reserved_row"] == 8
    b.a.verify(contract["sources"])
    b.save(
        "independent_verification.json",
        dict(
            status="passed",
            created_utc=b.a.utc(),
            numeric_values=checks,
            maximum_discrepancy=largest,
            sealed_immutable_outputs=len(bound),
            prior_seals=len(seals),
            mutable_navigation_excluded=skipped,
            historical_navigation_amendment=b.a.rel(amendment_path),
            finite_combinations_reconstructed=combinations,
            boundary_fixtures=fixtures,
            broad_sensitivities=len(si["broad"]),
            matched_contrasts=len(ci["matched"]),
            gpu_original_contrasts=len(ci["gpu"]),
            checked_required_scenarios=sorted(required),
            required_classification_changes=0,
            full_zero_sensitivity_classification_changes=0,
            original_point_and_interval_values_unchanged=True,
            prior_fourteen_remote_rows_unchanged=True,
            technical_note="The producer summary used three old exclusion aliases in its filter; this independent check uses all four actual scenario names. All required exclusion classifications remain unchanged, so no scientific value or conclusion changes.",
            new_draws=0,
            new_predictions=0,
            new_training=0,
        ),
    )
    print(
        json.dumps(
            dict(
                status="passed",
                numeric_values=checks,
                maximum_discrepancy=largest,
                sealed_outputs=len(bound),
                finite_combinations=combinations,
            )
        )
    )


if __name__ == "__main__":
    b.a.g.offline()
    run()

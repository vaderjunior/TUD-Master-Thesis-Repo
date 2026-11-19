from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import math
import numpy as np
from analysis.feedback.transfer import resources as p

a = p.a
OUT = p.OUT


def close(left, right, what, atol=1e-7):
    a.need(np.allclose(left, right, rtol=0, atol=atol), what)


def secs(start, end):
    delta = (
        datetime.fromisoformat(end) - datetime.fromisoformat(start)
    ).total_seconds()
    a.need(delta >= 0, "Chronological order")
    return delta


def quantile(array, prob):
    ordered = np.sort(array, axis=0)
    at = (len(ordered) - 1) * prob
    low = int(math.floor(at))
    return ordered[low] + (ordered[int(math.ceil(at))] - ordered[low]) * (at - low)


def main():
    a.need(
        (OUT / "calculated.json").exists() and not (OUT / "verified.json").exists(),
        "Preserve numerical proof",
    )
    assembled = a.read(OUT / "assembled.json")
    calculated = a.read(OUT / "calculated.json")
    contract = a.read(OUT / "analysis_contract.json")
    a.need(
        calculated["assembled_sha256"] == a.sha(OUT / "assembled.json"),
        "Current calculation assembly",
    )
    for receipt in (contract, assembled, calculated):
        for group in ("input_bindings", "analysis_sources", "sources", "artifacts"):
            for name, digest in receipt.get(group, {}).items():
                a.need(a.sha(a.ROOT / name) == digest, "Changed bound evidence " + name)
    groups = a.read(OUT / "resource_groups.json")
    allq = groups["all"]
    meta = list(a.rows(p.P3 / "metadata.jsonl"))
    matrix = list(a.rows(a.OUT / "matrix.jsonl"))
    N = len(meta)
    slots = N * 12
    accounting_names = sorted(allq["accounting"])
    class_names = sorted(allq["response_classes"])
    accounting = np.zeros((slots, len(accounting_names)), np.int64)
    tokens = np.zeros((slots, 4), np.int64)
    classes = np.zeros((slots, len(class_names)), np.int64)
    flags = np.full((slots, 4), -1, np.int64)
    wait = np.zeros(slots)
    time_sum = np.zeros(slots)
    duration = np.empty(608267)
    duration_slot = np.empty(608267, np.int32)
    seen = np.zeros(slots, bool)
    pos = 0
    with np.load(OUT / "arrays.npz", allow_pickle=False) as f:
        saved = f["resources"]
        local = f["local_prompt_tokens"]
        saved_valid = f["valid_votes"]
    workers = np.load(p.P7 / "workers.npz")["workers"].reshape(-1)
    offset = Counter()
    for file in sorted((a.OUT / "assembly").glob("transfer_*.jsonl")):
        for row in a.rows(file):
            i = row["input_index"]
            c = p.CONDITIONS.index(row["condition"])
            r = row["repeat"]
            j = i * 12 + c * 3 + r
            a.need(
                not seen[j]
                and row["key"] == meta[i]["key"]
                and row["stage"] == meta[i]["stage"],
                "Original resource identity",
            )
            seen[j] = True
            accounting[j] = [row["accounting"][key] for key in accounting_names]
            tokens[j] = [
                row["ordinary_usage"]["prompt_tokens"],
                row["ordinary_usage"]["completion_tokens"],
                row["repair_usage"].get("prompt_tokens", 0),
                row["repair_usage"].get("completion_tokens", 0),
            ]
            classes[j] = [row["response_classes"].get(key, 0) for key in class_names]
            flags[j] = [
                row["n_valid"],
                row["parse_failures"],
                int(row["uncertain"]),
                row["administrative_unknown_requests"],
            ]
            ds = row["known_request_durations_s"]
            a.need(
                len(ds)
                == row["known_request_duration_count"]
                == row["accounting"]["post_attempts"]
                and row["administrative_unknown_requests"] == 0,
                "Complete known attempt duration coverage",
            )
            duration[pos : pos + len(ds)] = ds
            duration_slot[pos : pos + len(ds)] = j
            pos += len(ds)
            time_sum[j] = math.fsum(ds)
            close(
                time_sum[j],
                row["known_request_seconds_sum"],
                "Original per-output duration sum",
                1e-9,
            )
            wait[j] = row["retry_backoff_seconds_sum"]
            expected = [
                sum(tokens[j]),
                sum(tokens[j, :2]),
                sum(tokens[j, 2:]),
                row["accounting"]["repair_logical_calls"],
                row["accounting"]["retry_attempts"],
                row["accounting"]["post_attempts"],
                time_sum[j],
                wait[j],
            ]
            a.need(
                np.array_equal(saved[i, c, r], expected)
                and saved_valid[i, c, r] == flags[j, 0],
                "Original complete resource vector",
            )
            offset[str(int(tokens[j, 0]) - 3 * int(local[i, c]))] += 1
        print("Independently reconstructed resources " + file.stem, flush=True)
    a.need(
        seen.all()
        and pos == 608267
        and np.isfinite(duration).all()
        and (duration >= 0).all(),
        "All original outputs and durations",
    )

    input_index = np.repeat(np.arange(N), 12)
    cond = np.tile(np.repeat(np.arange(4), 3), N)
    repeat = np.tile(np.arange(3), N * 4)
    ac = {key: accounting[:, i] for i, key in enumerate(accounting_names)}
    v = flags[:, 0]
    repairs = ac["repair_logical_calls"]
    extra = ac["retry_attempts"]
    masks = {"all": np.ones(slots, bool)}
    input_mask = lambda predicate: np.repeat(np.array([predicate(q) for q in meta]), 12)
    for pop in p.POPS:
        popmask = input_mask(lambda q: q["population"] == pop)
        masks["population/" + pop] = popmask
        for c, condition in enumerate(p.CONDITIONS):
            masks[pop + "/" + condition] = popmask & (cond == c)
            for r in range(3):
                masks[f"actual/{pop}/{condition}/repeat_{r+1}"] = (
                    popmask & (cond == c) & (repeat == r)
                )
        for w in (8, 128):
            masks[f"{pop}/workers_{w}"] = popmask & (workers == w)
        for n in range(4):
            masks[f"{pop}/valid_votes_{n}"] = popmask & (v == n)
    for c, condition in enumerate(p.CONDITIONS):
        masks["condition/" + condition] = cond == c
        for w in (8, 128):
            masks[f"{condition}/workers_{w}"] = (cond == c) & (workers == w)
    for src in sorted({q["source"] for q in meta}):
        sm = input_mask(lambda q: q["source"] == src)
        masks["source/" + src] = sm
        for gold in (0, 1):
            masks[f"source/{src}/gold_{gold}"] = sm & input_mask(
                lambda q: int(q["gold_hate"]) == gold
            )
    for fn in sorted({q["functionality"] for q in meta if q.get("functionality")}):
        masks["function/" + fn] = input_mask(lambda q: q.get("functionality") == fn)
    for stage in sorted({q["stage"] for q in meta}):
        masks["batch/" + stage] = input_mask(lambda q: q["stage"] == stage)
    for w in (8, 128):
        masks[f"workers/{w}"] = workers == w
    for n in range(4):
        masks[f"valid_votes/{n}"] = v == n
    for bit in (0, 1):
        masks[f"repair_used/{bit}"] = (repairs > 0) == bool(bit)
        masks[f"retry_used/{bit}"] = (extra > 0) == bool(bit)
        for prefix, field in [
            ("main_common", "common_valid"),
            ("strict_common", "strict_all_votes"),
        ]:
            masks[f"{prefix}/{bit}"] = np.repeat(
                np.array([q["masks"][field] for q in matrix]) == bool(bit), 12
            )
    masks = {name: value for name, value in masks.items() if value.any()}
    a.need(set(masks) == set(groups), "Complete resource group inventory")
    for name, mask in masks.items():
        q = groups[name]
        n = int(mask.sum())
        acs = {key: int(value[mask].sum()) for key, value in ac.items()}
        a.need(q["accounting"] == acs, "Group original integer accounting " + name)
        got = tokens[mask].sum(axis=0)
        expected_usage = {
            "ordinary_usage": got[:2],
            "repair_usage": got[2:],
            "observed_usage": got[:2] + got[2:],
        }
        for field, pair in expected_usage.items():
            a.need(
                q[field]
                == dict(
                    prompt_tokens=int(pair[0]),
                    completion_tokens=int(pair[1]),
                    total_tokens=int(pair.sum()),
                ),
                "Group exact usage " + name + "/" + field,
            )
        counts = dict(
            predictions=n,
            nulls=int(((v == 0) & mask).sum()),
            administrative_unknown_requests=0,
            parse_failures=int(flags[mask, 1].sum()),
            repairs=int(repairs[mask].sum()),
            repair_used=int(((repairs > 0) & mask).sum()),
            retry_used=int(((extra > 0) & mask).sum()),
            gate_ties=int(flags[mask, 2].sum()),
        )
        for vnum in range(4):
            count = int(((v == vnum) & mask).sum())
            if count:
                counts[f"valid_votes_{vnum}"] = count
        a.need(q["counts"] == counts, "Group null/repair/validity counts " + name)
        a.need(
            q["response_classes"]
            == {
                key: int(classes[mask, j].sum())
                for j, key in enumerate(class_names)
                if classes[mask, j].sum()
            },
            "Group response status " + name,
        )
        ds = duration[mask[duration_slot]]
        a.need(
            q["known_request_duration_count"] == len(ds), "Group duration denominator"
        )
        close(q["known_request_seconds_sum"], ds.sum(), "Group measured duration total")
        close(q["retry_backoff_seconds_sum"], wait[mask].sum(), "Group wait")
        for key, value in [
            ("mean", ds.mean()),
            ("min", ds.min()),
            ("max", ds.max()),
            ("p50", quantile(ds, 0.5)),
            ("p95", quantile(ds, 0.95)),
            ("p99", quantile(ds, 0.99)),
        ]:
            close(
                q["known_request_seconds_" + key], value, "Group duration " + key, 1e-8
            )
        e = q["per_planned_prediction"]
        a.need(e["denominator"] == n, "Cost denominator includes nulls")
        close(
            [
                e[key]
                for key in (
                    "observed_tokens",
                    "prompt_tokens",
                    "completion_tokens",
                    "repair_calls",
                    "attempt_slots",
                    "known_request_seconds",
                )
            ],
            [
                got.sum() / n,
                (got[0] + got[2]) / n,
                (got[1] + got[3]) / n,
                acs["repair_logical_calls"] / n,
                acs["post_attempts"] / n,
                ds.sum() / n,
            ],
            "Per-planned-output costs",
        )
        close(
            q["repair_calls_per_ordinary_call"],
            acs["repair_logical_calls"] / acs["ordinary_logical_calls"],
            "Repair denominator",
            1e-14,
        )
        close(
            q["extra_attempts_per_logical_call"],
            acs["retry_attempts"]
            / (acs["ordinary_logical_calls"] + acs["repair_logical_calls"]),
            "Retry denominator",
            1e-14,
        )

    failure = a.read(OUT / "failure_resources.json")
    details = list(p.previous.gzrows(OUT / "retry_details.jsonl.gz"))
    by_path = defaultdict(list)
    for q in details:
        by_path[q["result_path"]].append(q)
    a.need(
        set(by_path) == set(failure["original_result_bindings"])
        and len(details) == 4195,
        "Failure subset inventory",
    )
    fcounts = Counter()
    fdurations = defaultdict(list)
    fwait = defaultdict(list)
    for name, records in by_path.items():
        raw = a.read_bound(a.ROOT / name, failure["original_result_bindings"][name])
        events = [q for q in raw["events"] if q["action"] != "ok"]
        a.need(len(records) == len(events), "Retry original event count")
        for row, event in zip(records, events):
            for key in (
                "action",
                "attempt",
                "vote",
                "turn",
                "kind",
                "response_class",
                "elapsed_s",
                "backoff_elapsed_s",
            ):
                a.need(row[key] == event[key], "Exact original retry event")
            j = (
                row["input_index"] * 12
                + p.CONDITIONS.index(row["condition"]) * 3
                + row["repeat"]
            )
            a.need(
                row["effective_workers"] == workers[j]
                and row["result_sha256"] == failure["original_result_bindings"][name],
                "Failure identity",
            )
            fcounts[row["response_class"]] += 1
            for group in (
                "all",
                f"workers/{workers[j]}",
                "class/" + row["response_class"],
                "kind/" + row["kind"],
            ):
                fdurations[group].append(row["elapsed_s"])
                fwait[group].append(row["backoff_elapsed_s"])
    a.need(dict(fcounts) == failure["classes"], "Failure classes")
    for name, ds in fdurations.items():
        q = failure["groups"][name]
        a.need(q["attempts"] == len(ds), "Failure duration count")
        close(
            [
                q[key]
                for key in (
                    "request_seconds",
                    "recorded_backoff_seconds",
                    "duration_p50",
                    "duration_p95",
                    "duration_max",
                )
            ],
            [
                math.fsum(ds),
                math.fsum(fwait[name]),
                float(quantile(np.array(ds), 0.5)),
                float(quantile(np.array(ds), 0.95)),
                max(ds),
            ],
            "Failure duration attribution",
        )

    timeline = a.read(OUT / "timeline.json")
    windows = Counter()
    gets = []
    for row in timeline["stages"]:
        folder = a.LIVE / "stages" / row["stage"]
        review = a.read(folder / "review.json")
        comp = a.read(folder / "complete.json")
        report = a.read(folder / review["report"])
        a.need(
            row["generation_summary_utc"] == comp["summary"]["finished_utc"]
            and row["stage_completion_utc"] == comp["completed_utc"]
            and row["audit_report_utc"] == report["created_utc"],
            "Generation/completion/audit clocks",
        )
        for j, attempt in enumerate(row["attempts"]):
            folder = a.ROOT / attempt["path"]
            start = a.read(folder / "start.json")
            end = a.read(folder / attempt["end_kind"])
            check = a.read(folder / "model_check_intent.json")
            response = a.read(folder / "model_check_response.json")
            inventory = a.read(folder / "inventory.json")
            a.need(
                attempt["start"] == start["started_utc"]
                and attempt["end"] == end["finished_utc"]
                and attempt["model_check_start"] == check["started_utc"]
                and attempt["pending_on_entry"] == inventory["pending"]
                and attempt["completed_on_entry"] == inventory["completed"],
                "Original attempt clocks and inventory",
            )
            gets.append(response["elapsed_s"])
            windows["start_to_model_check"] += secs(
                attempt["start"], attempt["model_check_start"]
            )
            windows["model_check_to_recorded_end"] += secs(
                attempt["model_check_start"], attempt["end"]
            )
            close(
                attempt["recorded_window_seconds"],
                secs(attempt["start"], attempt["end"]),
                "Attempt span",
            )
            if j + 1 < len(row["attempts"]):
                windows["explicit_between_attempt_gap"] += secs(
                    attempt["end"], row["attempts"][j + 1]["start"]
                )
        close(
            row["generation_envelope_seconds"],
            secs(row["first_start_utc"], row["generation_summary_utc"]),
            "Stage generation span",
        )
        close(
            row["completion_bookkeeping_seconds"],
            secs(row["generation_summary_utc"], row["stage_completion_utc"]),
            "Bookkeeping",
        )
        close(
            row["audit_window_seconds"],
            secs(row["stage_completion_utc"], row["audit_report_utc"]),
            "Audit window",
        )
        windows["completion_bookkeeping"] += row["completion_bookkeeping_seconds"]
        windows["completion_to_audit"] += row["audit_window_seconds"]
    for left, right in zip(timeline["stages"], timeline["stages"][1:]):
        windows["between_batches"] += secs(
            left["audit_report_utc"], right["first_start_utc"]
        )
    windows["last_audit_to_final_file_proxy"] = secs(
        timeline["stages"][-1]["audit_report_utc"],
        a.read(a.OVERVIEW)["completion_file_mtime_utc"],
    )
    a.need(
        set(windows) == set(timeline["partition_seconds"]) and len(gets) == 24,
        "All chronology components",
    )
    for key, value in windows.items():
        close(
            value,
            timeline["partition_seconds"][key],
            "Chronology partition " + key,
            1e-6,
        )
    close(
        sum(windows.values()),
        timeline["start_to_final_file_proxy_seconds"],
        "Calendar envelope",
        1e-5,
    )
    close(sum(gets), timeline["availability_seconds"], "GET timing")
    for worker, q in timeline["throughput"].items():
        subset = [
            row
            for row in timeline["stages"]
            if len(row["attempts"]) == 1
            and row["attempts"][0]["effective_workers"] == int(worker)
        ]
        a.need(
            q["stages"] == [row["stage"] for row in subset]
            and q["predictions"] == sum(row["predictions"] for row in subset),
            "Homogeneous uninterrupted throughput subset",
        )
        close(
            q["predictions_per_hour_generation"],
            3600
            * q["predictions"]
            / sum(row["generation_envelope_seconds"] for row in subset),
            "Observed throughput",
        )
        close(
            q["predictions_per_hour_through_audit"],
            3600
            * q["predictions"]
            / sum(row["start_to_audit_seconds"] for row in subset),
            "Through-audit throughput",
        )

    local_expected = np.zeros_like(local)
    for q in a.rows(p.P5 / "query_contexts.jsonl"):
        local_expected[q["input_index"], 0] = q["reference_input_tokens"]
    for q in a.rows(p.P5 / "delivery.jsonl"):
        local_expected[q["input_index"], p.CONDITIONS.index(q["condition"])] = q[
            "feedback_input_tokens"
        ]
        a.need(
            q["reference_input_tokens"] == local_expected[q["input_index"], 0],
            "Part 5 complete local reference",
        )
    a.need(
        np.array_equal(local, local_expected)
        and dict(offset)
        == a.read(OUT / "local_prompt_costs.json")[
            "ordinary_api_prompt_tokens_minus_three_local_counts"
        ],
        "Local versus observed prompt units",
    )
    paired = a.read(OUT / "paired_costs.json")
    max_error = 0.0
    scalar_count = 0
    raw = saved.sum(axis=2)
    numerators = np.stack(
        (
            3 * raw[:, 0],
            raw[:, 1:].sum(axis=1),
            raw[:, 1:].sum(axis=1) - 3 * raw[:, 0],
            *[3 * (raw[:, c] - raw[:, 0]) for c in (1, 2, 3)],
        ),
        axis=2,
    )
    for name, q in paired.items():
        expected_idx = [
            i
            for i, m in enumerate(meta)
            if m["population"] == q["population"]
            and (q["cohort"] == "all_roster" or matrix[i]["masks"]["common_valid"])
        ]
        a.need(expected_idx == q["input_indices"], "Cost comparison cohort")
        idx = np.array(expected_idx)
        labels = sorted({meta[i]["inference_cluster"] for i in idx})
        mapping = {key: n for n, key in enumerate(labels)}
        codes = np.array([mapping[meta[i]["inference_cluster"]] for i in idx])
        sizes = np.bincount(codes)
        cluster = np.zeros((len(labels), 8, 6))
        np.add.at(cluster, codes, numerators[idx])
        rng = np.random.Generator(np.random.PCG64(q["seed"]))
        draws = np.load(a.ROOT / q["draws"])
        reconstructed = np.empty_like(draws)
        a.need(
            q["inputs"] == len(idx)
            and q["inference_clusters"] == len(labels)
            and q["seed"] == p.k.seed(name),
            "Bootstrap dimensions and fixed seed",
        )
        for start in range(0, 10000, 16):
            chosen = rng.integers(
                0, len(labels), size=(min(16, 10000 - start), len(labels))
            )
            multiplicity = np.array(
                [np.bincount(sample, minlength=len(labels)) for sample in chosen]
            )
            n = multiplicity @ sizes
            reconstructed[start : start + len(chosen)] = (
                multiplicity @ cluster.reshape(len(labels), 48)
            ).reshape(-1, 8, 6) / (9 * n[:, None, None])
        difference = float(np.abs(draws - reconstructed).max())
        max_error = max(max_error, difference)
        a.need(difference < 1e-8, "Independent complete resource bootstrap " + name)
        scalar_count += draws.size
        point = numerators[idx].sum(axis=0) / (9 * len(idx))
        lo = quantile(reconstructed, 0.025)
        hi = quantile(reconstructed, 0.975)
        for j, metric in enumerate(p.k.METRICS):
            for c, column in enumerate(p.k.COLUMNS):
                close(
                    q["metrics"][metric][column]["estimate"],
                    point[j, c],
                    "Exact pooled cost point",
                    1e-8,
                )
                close(
                    q["metrics"][metric][column]["interval95"],
                    [lo[j, c], hi[j, c]],
                    "Independent cost interval",
                    1e-8,
                )
        print("Independently regenerated all resource draws " + name, flush=True)
    a.need(scalar_count == 1920000, "All four bootstrap sets")
    clf = a.read(OUT / "classifier_resources.json")
    original = a.read(p.previous.REVIEW)
    for name, q in clf["fits"].items():
        old = original["jobs"][name]
        a.need(
            q["timing_seconds"] == old["timing_seconds"]
            and q["fit_rows"] == old["fit_rows"]
            and q["source_ids"] == old["source_ids"]
            and q["prediction_rows"] == 16433,
            "Original classifier scope and timings",
        )
    for name, q in clf["nested_subsets"].items():
        rows = [clf["fits"][n] for n in q["receipts"]]
        a.need(
            q["fits"] == len(rows)
            and q["model_file_bytes"] == sum(z["model_file_bytes"] for z in rows),
            "Nested fit denominator",
        )
        for key in (
            "feature_seconds",
            "fit_seconds",
            "inference_and_verification_seconds",
        ):
            close(
                q[key],
                math.fsum(z["timing_seconds"][key] for z in rows),
                "Classifier clock " + key,
                1e-9,
            )
        close(
            q["worker_monitored_seconds"],
            math.fsum(z["worker_monitored_seconds"] for z in rows),
            "Monitored worker total",
            1e-9,
        )
    a.need(
        len(clf["inherited_vector_files"]) == 3
        and sum(q["shape"][0] for q in clf["inherited_vector_files"]) == 91237,
        "Inherited representation coverage",
    )
    for row in clf["inherited_vector_files"]:
        path = a.ROOT / row["path"]
        arr = np.load(path, mmap_mode="r")
        a.need(
            row["file_bytes"] == path.stat().st_size
            and row["shape"] == list(arr.shape)
            and row["dtype"] == str(arr.dtype),
            "Cached array bytes/shape",
        )
        del arr
    scenarios = a.read(OUT / "practical_scenarios.json")
    rates = a.read(p.P8 / "rates.json")
    fpr = rates["all_permitted/cutoff_06/cohort/main/en_test_FPR"]
    fnr = rates["all_permitted/cutoff_06/cohort/main/en_test_FNR"]
    for name, q in scenarios["comparisons"].items():
        a1 = fpr["estimates"]
        a2 = fnr["estimates"]
        dfp = (
            a1["qwen_change"]
            if name.endswith("vs_reference")
            else a1["qwen_updated"] - a1["linear_updated"]
        )
        dfn = (
            a2["qwen_change"]
            if name.endswith("vs_reference")
            else a2["qwen_updated"] - a2["linear_updated"]
        )
        close(
            [q["delta_FPR"], q["delta_FNR"]],
            [dfp, dfn],
            "Scenario original points",
            1e-14,
        )
        close(
            q["equal_error_cost_break_even_prevalence"],
            dfp / (dfp - dfn),
            "Equal-error illustration",
            1e-14,
        )
        for row in q["rows"]:
            pv = row["hypothetical_hate_prevalence"]
            close(
                row["miss_to_false_alarm_break_even_cost"],
                (1 - pv) * dfp / (-pv * dfn),
                "Hypothetical trade-off ratio",
                1e-12,
            )
            for ratio, value in row["weighted_changes"].items():
                close(
                    value,
                    1000 * ((1 - pv) * dfp + pv * float(ratio) * dfn),
                    "Explicit scenario formula",
                    1e-10,
                )
    old = a.read(
        a.ROOT
        / "experiments/astra/sq3_human_feedback_results_20261020_v1/partial_programme_ledger.json"
    )
    addition = a.read(OUT / "resource_addition.json")
    a.need(
        addition["preserved_previous_programmes"] == old["programmes"]
        and not addition["global_ledger_changed"],
        "Global ledger unchanged",
    )
    for key, total in addition["prospective_twelve_programme_totals"].items():
        a.need(
            total == old["totals"][key] + addition["addition"][key],
            "Exactly one staged ledger addition",
        )
    a.need(
        addition["addition"]["reported_tokens"] == 552493301
        and addition["capacity_diagnostics_excluded"]
        and addition["inherited_screening_excluded_from_new_row"],
        "No inherited/diagnostic double counting",
    )
    summary = a.read(OUT / "resource_summary.json")
    a.need(
        summary["accounting"] == allq["accounting"]
        and summary["observed_usage"] == allq["observed_usage"],
        "Summary totals",
    )
    close(
        summary["failed_POST_seconds"] + summary["successful_POST_seconds"],
        allq["known_request_seconds_sum"],
        "Failure/success time partition",
    )
    a.need(
        all(
            summary[key] is None
            for key in ("remote_GPU_hours", "remote_energy", "money_cost")
        ),
        "Unknown resources remain unknown",
    )
    ledger_count = 0
    for row in p.previous.gzrows(OUT / "input_costs.jsonl.gz"):
        i = row["input_index"]
        a.need(
            i == ledger_count
            and row["key"] == meta[i]["key"]
            and np.array_equal(row["actual_resources"], saved[i])
            and np.array_equal(row["local_prompt_tokens"], local[i])
            and np.array_equal(row["n_valid"], saved_valid[i]),
            "Complete input cost ledger",
        )
        ledger_count += 1
    a.need(ledger_count == N, "Every input including null costs")
    p.save(
        "verified.json",
        dict(
            status="part9_original_resource_chronology_cost_bootstraps_classifier_and_scenarios_verified",
            checked_utc=a.utc(),
            script_sha256=a.sha(Path(__file__)),
            calculated_sha256=a.sha(OUT / "calculated.json"),
            assembled_sha256=a.sha(OUT / "assembled.json"),
            original_outputs=189096,
            original_POST_durations=608267,
            resource_groups=len(groups),
            retry_events=4195,
            targeted_retry_result_files=len(by_path),
            chronological_attempts=24,
            chronological_batches=22,
            availability_checks=24,
            original_static_fits=38,
            nested_part8_fits=8,
            nested_principal_updates=3,
            paired_bootstrap_sets=4,
            verified_bootstrap_scalars=scalar_count,
            max_bootstrap_discrepancy=max_error,
            input_cost_rows=ledger_count,
            scenarios_are_hypothetical_points=True,
            global_ledger_unchanged=True,
            original_evidence_unchanged=True,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
            new_primary_families=0,
        ),
    )
    print("Independent Part 9 numerical verification passed.", flush=True)


if __name__ == "__main__":
    with a.offline():
        main()

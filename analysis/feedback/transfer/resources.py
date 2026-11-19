import os

for name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[name] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from collections import Counter, defaultdict
from pathlib import Path
import gzip
import io
import json
import math
import sys
import time
import unittest
import numpy as np
from analysis.feedback.transfer import classifiers as previous
from analysis.feedback.transfer import resources_metrics as k
from analysis.feedback.reserved import resources_metrics as core

a = previous.a
ROOT = a.ROOT
OUT = ROOT / "experiments/astra/sq3_transfer_part9_20261023_v1"
P8 = previous.OUT
P7 = previous.P7
P5 = previous.P5
P3 = previous.P3
P2 = previous.P2
CONDITIONS = previous.CONDITIONS
POPS = previous.POPS
LINEAR = previous.LINEAR
DIAG = a.EXECUTION / "concurrency_20261021_v3/publication/diagnostic_resources.json"


def save(name, value):
    path = OUT / name
    a.need(path.resolve().is_relative_to(OUT), "Write outside Part 9")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(
            value
            if isinstance(value, bytes)
            else a.encoded(previous.native(value), pretty=True)
        )


def gzopen(name):
    return io.TextIOWrapper(
        gzip.GzipFile(filename=str(OUT / name), mode="xb", mtime=0, compresslevel=5),
        encoding="utf8",
        newline="\n",
    )


def emit(f, value):
    f.write(
        json.dumps(
            previous.native(value),
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
            separators=(",", ":"),
        )
        + "\n"
    )


def bind(path, bindings):
    bindings[a.rel(path)] = a.sha(path)
    return a.read(path)


def prepare():
    a.need(not OUT.exists(), "Preserve prior Part 9 work")
    last = a.read(P8 / "complete.json")
    check = a.read(P8 / "publication/final_check.json")
    a.need(
        check["complete_sha256"] == a.sha(P8 / "complete.json")
        and last["completed_transfer_parts"] == list(range(1, 9)),
        "Latest verified Part 8",
    )
    bindings = {
        **last["input_bindings"],
        **last["analysis_sources"],
        **last["artifacts"],
    }
    for path in (
        P8 / "complete.json",
        P8 / "publication/final_check.json",
        a.OVERVIEW,
        a.EXECUTION / "live/complete.json",
        DIAG,
        ROOT
        / "experiments/astra/sq3_reserved_part8_results_20261020_v2/inherited_work.json",
        ROOT
        / "experiments/astra/sq3_human_feedback_results_20261020_v1/partial_programme_ledger.json",
        ROOT / "analysis/feedback/reserved/resources_metrics.py",
    ):
        bindings[a.rel(path)] = a.sha(path)
    for name, digest in bindings.items():
        a.need(a.sha(ROOT / name) == digest, "Changed inherited evidence " + name)
    from tests.feedback.transfer import test_resources as tests

    log = io.StringIO()
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromModule(tests)
    )
    a.need(result.wasSuccessful(), "Resource fixtures " + log.getvalue())
    save(
        "analysis_contract.json",
        dict(
            created_utc=a.utc(),
            input_bindings=bindings,
            analysis_sources={
                a.rel(Path(module.__file__)): a.sha(Path(module.__file__))
                for module in (sys.modules[__name__], k, core, tests)
            },
            scope="Part 9 resource/practical review only; no original source, taxonomy, scoring, threshold or runtime change. All new work is offline. Existing numerical/raw audits remain authoritative; no full raw replay.",
            accounting="Use every one of 189096 original planned predictions, including nulls, under-three outputs and repairs. Ordinary logical calls, repair calls, transport extra attempts and availability GETs remain separate. Successful-response tokens exclude unknown remote work behind exceptions. Capacity diagnostic totals are separately attributed and excluded.",
            groups="All; populations; four conditions; each actual population/condition/repeat; sources/gold; all 29 functions; actual eight/128-worker identities; stages; vote-validity classes; repair/retry occurrence; included/excluded common/strict masks; full input cost ledger. Record actual request quantiles and summed durations, never label them wall/GPU-hours.",
            chronology="Bind all 24 attempts and 22 audited stages. Distinguish original generation summary, stage-completion receipt, audit report and historically captured final-file timestamp. Separate explicit interruptions/gaps; show homogeneous uninterrupted batch throughput observationally. No causal concurrency estimate or future throughput guarantee.",
            paired_costs=dict(
                role="Supporting after earlier outcomes; not a new confirmatory family, equivalence or deployment-cost estimate",
                cohorts=["all_roster", "main_common"],
                populations=POPS,
                metrics=k.METRICS,
                columns=k.COLUMNS,
                draws=10000,
                confidence=0.95,
                seed_rule="first 8 SHA256 bytes little-endian of A16-part9-resource-v1|family",
                unit="Whole original inference cluster, item weighted; all conditions/repeats and metric components kept together",
                null_costs="Retained in all-roster costs; common-mask results separately labelled",
            ),
            practical_scenarios="Descriptive English-only algebra: per-1000 conditional errors and explicitly hypothetical prevalence/error-cost grids. No measured deployment prevalence, policy selection, extrapolated guarantee or new utility significance test. Hold observed conditional rates fixed only for the illustration. HateCheck never supplies deployment prevalence.",
            classifier="Reconcile all 38 previously verified static fit receipts; identify the eight used here and principal three updates as nested subsets, never additional costs. Separate feature materialization, solver fitting, inference-plus-verification, worker monitored wall time and sampled memory. Cached vectors/prior search/annotation are inherited; binary head is not full multi-field Qwen output.",
            local_tokens="Use already verified complete local prompt counts from Part 5, separately from successful-response API usage. No retokenization, embedding or retrieval.",
            integration="Stage one exact-once transfer ledger addition for Part 10; leave global journal/resource/writer artifacts unchanged. Preserve deadlines, nine experiments (3 complete/6 open), sixteen inventory rows and later cumulative studies.",
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
            new_annotations=0,
            new_primary_families=0,
        ),
    )
    save(
        "focused_tests.json",
        dict(passed=True, tests_run=result.testsRun, log=log.getvalue()),
    )
    print("Part 9 scope and fourteen boundary fixtures saved.", flush=True)


def chronology(bindings):
    done = bind(a.EXECUTION / "live/complete.json", bindings)
    stages = []
    statuses = Counter()
    retry = []
    gets = []
    periods = Counter()
    for stage_id in range(1, 23):
        stage = f"transfer_{stage_id:02}"
        folder = a.LIVE / "stages" / stage
        review = bind(folder / "review.json", bindings)
        comp = bind(folder / "complete.json", bindings)
        report = bind(folder / review["report"], bindings)
        a.need(
            done["review_hashes"][a.rel(folder / "review.json")]
            == a.sha(folder / "review.json")
            and review["complete_sha256"] == a.sha(folder / "complete.json")
            and review["artifacts"][review["report"]]
            == a.sha(folder / review["report"]),
            "Original completed stage binding",
        )
        history = {}
        for name, digest in report["attempt_hashes"].items():
            path = a.LIVE / name
            a.need(a.sha(path) == digest, "Saved attempt history")
            history[name] = bind(path, bindings)
        attempts = []
        folders = sorted(
            {
                str(Path(n).parent).replace("\\", "/")
                for n in history
                if Path(n).parent.name.startswith("attempt_")
            }
        )
        for folder_name in folders:
            start = history[folder_name + "/start.json"]
            inv = history[folder_name + "/inventory.json"]
            intent = history[folder_name + "/model_check_intent.json"]
            response = history[folder_name + "/model_check_response.json"]
            a.need(
                response["status"] == 200 and intent["action"] == "GET PEASEC /models",
                "Recorded model availability",
            )
            endpoints = [
                name
                for name in ("summary.json", "interrupted.json")
                if folder_name + "/" + name in history
            ]
            a.need(len(endpoints) == 1, "Recorded attempt endpoint required")
            end = history[folder_name + "/" + endpoints[0]]
            workers = (
                128
                if stage_id >= 5 or stage_id == 4 and endpoints[0] == "summary.json"
                else 8
            )
            row = dict(
                path=a.rel(a.LIVE / folder_name),
                start=start["started_utc"],
                end=end["finished_utc"],
                end_kind=endpoints[0],
                model_check_start=intent["started_utc"],
                model_check_seconds=response["elapsed_s"],
                completed_on_entry=inv["completed"],
                pending_on_entry=inv["pending"],
                effective_workers=workers,
            )
            if endpoints[0] == "summary.json":
                row.update(
                    saved_items=end["saved_items"],
                    submitted=end["submitted"],
                    peak_submitted=end["peak_submitted"],
                    stopped=end["stopped"],
                )
                a.need(
                    end["peak_submitted"] == workers and not end["stopped"],
                    "Observed final workers",
                )
            row["recorded_window_seconds"] = core.seconds(row["start"], row["end"])
            periods[str(workers)] += row["recorded_window_seconds"]
            attempts.append(row)
            gets.append(
                dict(
                    stage=stage,
                    attempt=row["path"],
                    started_utc=intent["started_utc"],
                    seconds=response["elapsed_s"],
                    status=200,
                )
            )
        generation = comp["summary"]["finished_utc"]
        a.need(
            generation == attempts[-1]["end"]
            and len(attempts) == report["attempt_count"],
            "Original generation summary",
        )
        partition = core.attempt_partition(attempts, generation)
        first = attempts[0]["start"]
        completed = comp["completed_utc"]
        audited = report["created_utc"]
        a.need(
            math.isclose(
                core.seconds(first, completed),
                report["wall_seconds_including_restarts"],
                abs_tol=1e-6,
            ),
            "Original stage wall",
        )
        n = report["counts"]["predictions"]
        stages.append(
            dict(
                stage=stage,
                first_start_utc=first,
                generation_summary_utc=generation,
                stage_completion_utc=completed,
                audit_report_utc=audited,
                attempts=attempts,
                partition=partition,
                predictions=n,
                accounting=report["accounting"],
                observed_usage=report["observed_usage"],
                counts=report["counts"],
                generation_envelope_seconds=core.seconds(first, generation),
                completion_bookkeeping_seconds=core.seconds(generation, completed),
                audit_window_seconds=core.seconds(completed, audited),
                start_to_audit_seconds=core.seconds(first, audited),
                homogeneous_uninterrupted=len(attempts) == 1,
                effective_workers=(
                    attempts[0]["effective_workers"]
                    if len({x["effective_workers"] for x in attempts}) == 1
                    else None
                ),
            )
        )
        statuses.update(report["http_statuses"])
        retry.extend(dict(stage=stage, **q) for q in report["retry_events"])
    components = Counter()
    gaps = []
    for row in stages:
        components.update(row["partition"]["seconds_by_kind"])
        components["completion_bookkeeping"] += row["completion_bookkeeping_seconds"]
        components["completion_to_audit"] += row["audit_window_seconds"]
    for left, right in zip(stages, stages[1:]):
        sec = core.seconds(left["audit_report_utc"], right["first_start_utc"])
        gaps.append(dict(after=left["stage"], before=right["stage"], seconds=sec))
        components["between_batches"] += sec
    overview = a.read(a.OVERVIEW)
    mtime = overview["completion_file_mtime_utc"]
    components["last_audit_to_final_file_proxy"] = core.seconds(
        stages[-1]["audit_report_utc"], mtime
    )
    elapsed = core.seconds(stages[0]["first_start_utc"], mtime)
    a.need(
        math.isclose(sum(components.values()), elapsed, abs_tol=1e-5),
        "Whole chronology partition",
    )
    throughput = {}
    for workers in (8, 128):
        rows = [
            q
            for q in stages
            if q["homogeneous_uninterrupted"] and q["effective_workers"] == workers
        ]
        n = sum(q["predictions"] for q in rows)
        secs = sum(q["generation_envelope_seconds"] for q in rows)
        wall = sum(q["start_to_audit_seconds"] for q in rows)
        throughput[str(workers)] = dict(
            stages=[q["stage"] for q in rows],
            predictions=n,
            generation_window_seconds=secs,
            through_audit_seconds=wall,
            predictions_per_hour_generation=3600 * n / secs,
            predictions_per_hour_through_audit=3600 * n / wall,
            per_batch_generation_predictions_per_hour=[
                3600 * q["predictions"] / q["generation_envelope_seconds"] for q in rows
            ],
        )
    a.need(
        len(gets) == 24
        and sum(statuses.values()) == done["accounting"]["post_attempts"]
        and len(retry) == 4195,
        "GET/status/retry totals",
    )
    return dict(
        stages=stages,
        between_batches=gaps,
        availability_checks=gets,
        availability_seconds=sum(q["seconds"] for q in gets),
        http_statuses=statuses,
        retry_event_inventory=retry,
        recorded_attempt_windows_by_workers=periods,
        partition_seconds=components,
        start_to_last_generation_seconds=core.seconds(
            stages[0]["first_start_utc"], stages[-1]["generation_summary_utc"]
        ),
        start_to_last_audit_seconds=core.seconds(
            stages[0]["first_start_utc"], stages[-1]["audit_report_utc"]
        ),
        start_to_final_file_proxy_seconds=elapsed,
        final_file_proxy_utc=mtime,
        throughput=throughput,
        qualification="All windows include local preparation/waiting. Interrupted gaps have explicit endpoints; audit windows are not pure CPU timings. Final-file timestamp is the prior captured filesystem proxy, not an internal finalization clock. Homogeneous batches contain different inputs and service states; no causal concurrency multiplier.",
    )


def classifiers(bindings):
    review = bind(previous.REVIEW, bindings)
    used = a.read(P8 / "fit_bindings.json")
    records = {}
    monitors = {}
    for path in sorted((LINEAR / "resource_checks").glob("*/resources.json")):
        q = bind(path, bindings)
        a.need(
            review["bindings"][a.rel(path)] == a.sha(path), "Verified monitor binding"
        )
        cmd = q["command"]
        variant = cmd[cmd.index("--variant") + 1]
        ids = cmd[cmd.index("--ids") + 1]
        monitors[variant, ids] = q
    for name, q in review["jobs"].items():
        path = ROOT / name
        receipt = bind(path, bindings)
        at = path.parent / receipt["attempt"]
        attribution = bind(at / "attribution.json", bindings)
        a.need(
            review["bindings"][name] == a.sha(path)
            and receipt["outputs"][receipt["attempt"] + "/attribution.json"]
            == a.sha(at / "attribution.json"),
            "Verified classifier attribution",
        )
        a.need(
            all(attribution[key] == value for key, value in q["timing_seconds"].items())
            and q["seconds_this_attempt"] == receipt["seconds_this_attempt"],
            "Original classifier clocks",
        )
        monitor = monitors[q["variant"], ",".join(q["source_ids"])]
        aliases = [
            label for label, job in used["jobs"].items() if job["receipt"] == name
        ]
        model = at / "model.npz"
        prob = at / "probabilities.npy"
        a.need(
            a.sha(model) == receipt["outputs"][receipt["attempt"] + "/model.npz"],
            "Saved head hash",
        )
        records[name] = dict(
            variant=q["variant"],
            source_count=len(q["source_ids"]),
            source_ids=q["source_ids"],
            baseline=q["baseline_reproduced"],
            fit_rows=q["fit_rows"],
            prediction_rows=q["probability_rows"],
            timing_seconds=q["timing_seconds"],
            attempt_wall_seconds=q["seconds_this_attempt"],
            worker_monitored_seconds=monitor["seconds"],
            sampled_peak_private_GiB=monitor["sampled_peak_worker_tree_private_GiB"],
            sampled_min_available_GiB=monitor["sampled_min_available_GiB"],
            model_file_bytes=model.stat().st_size,
            probability_file_bytes=prob.stat().st_size,
            used_in_part8=bool(aliases),
            part8_aliases=aliases,
        )
    subsets = {
        "all_static": list(records),
        "part8_eight": [n for n, q in records.items() if q["used_in_part8"]],
        "principal_three_updates": [
            n
            for n, q in records.items()
            if q["used_in_part8"]
            and q["variant"] == "all_permitted"
            and not q["baseline"]
        ],
    }
    summary = {}
    for name, names in subsets.items():
        rows = [records[n] for n in names]
        summary[name] = dict(
            fits=len(rows),
            receipts=names,
            **{
                key: math.fsum(q["timing_seconds"][key] for q in rows)
                for key in (
                    "feature_seconds",
                    "fit_seconds",
                    "inference_and_verification_seconds",
                )
            },
            worker_monitored_seconds=math.fsum(
                q["worker_monitored_seconds"] for q in rows
            ),
            sampled_peak_private_GiB=max(q["sampled_peak_private_GiB"] for q in rows),
            model_file_bytes=sum(q["model_file_bytes"] for q in rows),
        )
    a.need(
        len(records) == 38
        and len(subsets["part8_eight"]) == 8
        and len(subsets["principal_three_updates"]) == 3,
        "Nested static fit accounting",
    )
    a.need(
        math.isclose(
            summary["all_static"]["fit_seconds"],
            review["resources"]["fit_seconds"],
            abs_tol=1e-9,
        ),
        "Historical fit total",
    )

    vectors = []
    for name, digest in review["bindings"].items():
        if name.endswith(
            ("/full_vectors.npy", "/query_vectors.npy", "/evaluation_vectors.npy")
        ):
            path = ROOT / name
            a.need(a.sha(path) == digest, "Inherited vector binding")
            bindings[name] = digest
            arr = np.load(path, mmap_mode="r", allow_pickle=False)
            vectors.append(
                dict(
                    path=name,
                    sha256=digest,
                    file_bytes=path.stat().st_size,
                    shape=list(arr.shape),
                    dtype=str(arr.dtype),
                    new_encoding=False,
                )
            )
            del arr
    return dict(
        fits=records,
        nested_subsets=summary,
        inherited_vector_files=vectors,
        new_fits=0,
        new_embeddings=0,
        cumulative_fits_pending=True,
        qualification="Feature seconds materialize cached vectors, not BGE inference. Inference-and-verification includes model reload, independent arithmetic and output persistence for 16433 texts, not a pure production latency benchmark. Memory is sampled, model bytes exclude BGE weights/vectors. Nested subsets are already inside the 38-fit block; no summing them as separate programmes.",
    )


def assemble():
    a.need(
        (OUT / "analysis_contract.json").exists()
        and not (OUT / "assembled.json").exists(),
        "New assembly only",
    )
    contract = a.read(OUT / "analysis_contract.json")
    bindings = dict(contract["input_bindings"])
    began = time.monotonic()
    for name, digest in contract["analysis_sources"].items():
        a.need(a.sha(ROOT / name) == digest, "Changed contract source")
    timeline = chronology(bindings)
    save("timeline.json", timeline)
    clf = classifiers(bindings)
    save("classifier_resources.json", clf)
    meta = list(a.rows(P3 / "metadata.jsonl"))
    matrix = list(a.rows(a.OUT / "matrix.jsonl"))
    N = len(meta)
    workers = np.load(P7 / "workers.npz")["workers"]
    x = np.full((N, 4, 3, 8), np.nan)
    valid = np.full((N, 4, 3), -1, dtype=np.int8)
    seen = set()
    groups = defaultdict(core.Group)
    groupclasses = defaultdict(Counter)
    failure_counts = Counter()
    failure_durations = defaultdict(list)
    failure_waits = defaultdict(list)
    failure_sources = {}
    api_offset = Counter()
    token = np.zeros((N, 4), dtype=np.int64)
    for row in a.rows(P5 / "query_contexts.jsonl"):
        token[row["input_index"], 0] = row["reference_input_tokens"]
    for row in a.rows(P5 / "delivery.jsonl"):
        c = CONDITIONS.index(row["condition"])
        i = row["input_index"]
        a.need(
            row["reference_input_tokens"] == token[i, 0]
            and row["feedback_input_tokens"] - token[i, 0] == row["token_difference"],
            "Complete local prompt counts",
        )
        token[i, c] = row["feedback_input_tokens"]
    a.need((token > 0).all(), "Every complete frozen prompt count")
    with gzopen("retry_details.jsonl.gz") as retries:
        for file in sorted((a.OUT / "assembly").glob("transfer_*.jsonl")):
            a.need(
                a.sha(file)
                == a.read(a.OUT / "assembly_complete.json")["artifacts"][a.rel(file)],
                "Bound assembled batch",
            )
            bindings[a.rel(file)] = a.sha(file)
            for row in a.rows(file):
                i = row["input_index"]
                c = CONDITIONS.index(row["condition"])
                r = row["repeat"]
                ident = (i, c, r)
                m = meta[i]
                a.need(
                    ident not in seen
                    and m["key"] == row["key"]
                    and m["population"] == row["population"],
                    "Unique resource slot",
                )
                seen.add(ident)
                ac = row["accounting"]
                u = row["observed_usage"]
                ordinary = row["ordinary_usage"]
                repair = row["repair_usage"]
                w = int(workers[i, c, r])
                valid[i, c, r] = row["n_valid"]
                api_offset[str(ordinary["prompt_tokens"] - 3 * int(token[i, c]))] += 1
                x[i, c, r] = [
                    u["total_tokens"],
                    ordinary["total_tokens"],
                    repair.get("total_tokens", 0),
                    ac["repair_logical_calls"],
                    ac["retry_attempts"],
                    ac["post_attempts"],
                    row["known_request_seconds_sum"],
                    row["retry_backoff_seconds_sum"],
                ]
                pop = m["population"]
                condition = row["condition"]
                src = m["source"]
                gold = int(m["gold_hate"])
                labels = [
                    "all",
                    "population/" + pop,
                    "condition/" + condition,
                    f"{pop}/{condition}",
                    f"actual/{pop}/{condition}/repeat_{r+1}",
                    "source/" + src,
                    f"source/{src}/gold_{gold}",
                    f"workers/{w}",
                    f"{pop}/workers_{w}",
                    f"{condition}/workers_{w}",
                    "batch/" + row["stage"],
                    f'valid_votes/{row["n_valid"]}',
                    f'{pop}/valid_votes_{row["n_valid"]}',
                    f'repair_used/{int(bool(row["repairs"]))}',
                    f'retry_used/{int(bool(ac["retry_attempts"]))}',
                    f'main_common/{int(matrix[i]["masks"]["common_valid"])}',
                    f'strict_common/{int(matrix[i]["masks"]["strict_all_votes"])}',
                ]
                if m.get("functionality"):
                    labels.append("function/" + m["functionality"])
                for name in labels:
                    groups[name].add(row)
                    groups[name].counts["repair_used"] += int(bool(row["repairs"]))
                    groups[name].counts["retry_used"] += int(bool(ac["retry_attempts"]))
                    groups[name].counts["gate_ties"] += int(row["uncertain"])
                    groupclasses[name].update(row["response_classes"])
                if ac["retry_attempts"]:
                    path = Path(row["item_receipt"]).with_name("result.json")
                    saved = a.read_bound(ROOT / path, row["result_sha256"])
                    failure_sources[path.as_posix()] = row["result_sha256"]
                    events = [
                        event for event in saved["events"] if event["action"] != "ok"
                    ]
                    a.need(len(events) == ac["retry_attempts"], "Retry event count")
                    for event in events:
                        record = dict(
                            input_index=i,
                            key=row["key"],
                            population=pop,
                            condition=condition,
                            repeat=r,
                            stage=row["stage"],
                            effective_workers=w,
                            result_path=path.as_posix(),
                            result_sha256=row["result_sha256"],
                            **{
                                name: event[name]
                                for name in (
                                    "action",
                                    "attempt",
                                    "vote",
                                    "turn",
                                    "kind",
                                    "response_class",
                                    "elapsed_s",
                                    "backoff_elapsed_s",
                                )
                            },
                        )
                        emit(retries, record)
                        a.need(
                            event["response_class"] != a.ADMIN
                            and event["action"] == "retry",
                            "Current run has no unknown interruption sentinel",
                        )
                        failure_counts[event["response_class"]] += 1
                        for name in (
                            "all",
                            "workers/" + str(w),
                            "class/" + event["response_class"],
                            "kind/" + event["kind"],
                        ):
                            failure_durations[name].append(event["elapsed_s"])
                            failure_waits[name].append(event["backoff_elapsed_s"])
            print(
                f"Resource census {file.stem}: {len(seen):,}/189,096 outputs",
                flush=True,
            )
    a.need(len(seen) == 189096 and np.isfinite(x).all(), "Complete resource matrix")
    values = {
        name: dict(**g.finish(), response_classes=dict(groupclasses[name]))
        for name, g in sorted(groups.items())
    }
    done = a.read(a.EXECUTION / "live/complete.json")
    total = values["all"]
    a.need(
        total["accounting"] == done["accounting"]
        and total["observed_usage"] == done["observed_usage"],
        "Original total reconciliation",
    )
    for stage in timeline["stages"]:
        q = values["batch/" + stage["stage"]]
        a.need(
            q["accounting"] == stage["accounting"]
            and q["observed_usage"] == stage["observed_usage"],
            "Every original batch total",
        )
    a.need(
        total["counts"]["administrative_unknown_requests"] == 0
        and sum(failure_counts.values()) == 4195,
        "Failure and duration coverage",
    )
    save("resource_groups.json", values)
    save(
        "failure_resources.json",
        dict(
            classes=failure_counts,
            groups={
                name: dict(
                    attempts=len(ds),
                    request_seconds=math.fsum(ds),
                    recorded_backoff_seconds=math.fsum(failure_waits[name]),
                    duration_p50=core.quantile(ds, 0.5),
                    duration_p95=core.quantile(ds, 0.95),
                    duration_max=max(ds),
                )
                for name, ds in failure_durations.items()
            },
            original_result_bindings=failure_sources,
            qualification="Observed client exceptions do not quantify remote generation/token work. Read only result records with retries, not a full raw audit. Backoff values are recorded event fields, not reconstructed timeout downtime.",
        ),
    )
    with (OUT / "arrays.npz").open("xb") as f:
        np.savez_compressed(
            f, resources=x, local_prompt_tokens=token, valid_votes=valid
        )
    save(
        "array_schema.json",
        dict(
            shape=list(x.shape),
            axes=[
                "original input_index",
                "reference then fixed pools 441/442/443",
                "three actual decoder repeats",
                "metric",
            ],
            metrics=k.METRICS,
            conditions=CONDITIONS,
            request_seconds="Sum of measured attempts for this voted output; not end-to-end latency or GPU time",
            all_null_costs_retained=True,
        ),
    )
    save(
        "local_prompt_costs.json",
        dict(
            complete_local_tokens_by_population_condition={
                pop
                + "/"
                + condition: dict(
                    prompts=int(sum(m["population"] == pop for m in meta)),
                    sum=int(token[[m["population"] == pop for m in meta], c].sum()),
                    min=int(token[[m["population"] == pop for m in meta], c].min()),
                    max=int(token[[m["population"] == pop for m in meta], c].max()),
                )
                for pop in POPS
                for c, condition in enumerate(CONDITIONS)
            },
            ordinary_api_prompt_tokens_minus_three_local_counts=api_offset,
            qualification="Complete frozen local input-token estimate from verified Part 5; each prompt reused across three actual repeats and three ordinary votes. Provider-reported prompt usage is a distinct measurement; no local count is added twice.",
        ),
    )
    with gzopen("input_costs.jsonl.gz") as f:
        for i, m in enumerate(meta):
            emit(
                f,
                dict(
                    input_index=i,
                    key=m["key"],
                    population=m["population"],
                    source=m["source"],
                    gold_hate=m["gold_hate"],
                    inference_cluster=m["inference_cluster"],
                    main_common=matrix[i]["masks"]["common_valid"],
                    strict_common=matrix[i]["masks"]["strict_all_votes"],
                    actual_resources=x[i],
                    local_prompt_tokens=token[i],
                    effective_workers=workers[i],
                    n_valid=valid[i],
                ),
            )
    diagnostic = bind(DIAG, bindings)
    save(
        "excluded_capacity_costs.json",
        dict(
            source=a.rel(DIAG),
            source_sha256=a.sha(DIAG),
            diagnostic_POSTs=diagnostic["diagnostic_POSTs"],
            models_GETs=diagnostic["models_GETs"],
            observed_tokens=diagnostic["observed_tokens"],
            timeouts_with_unknown_remote_work=diagnostic[
                "earlier_timeout_POSTs_with_unknown_remote_work"
            ],
            scientific_predictions=0,
            included_in_transfer_totals=False,
            qualification="Separate completed diagnostic scope recorded at the approved 128-worker seal; not a newly run probe or a token charge attributable solely to feedback. Earlier diagnostic checkpoints are nested, not added again.",
        ),
    )
    inherited = a.read(
        ROOT
        / "experiments/astra/sq3_reserved_part8_results_20261020_v2/inherited_work.json"
    )
    save(
        "inherited_costs.json",
        dict(
            screening=inherited["screening"],
            candidate_labels=1024,
            three_balanced_pools=dict(
                memberships=120, unique_sources=98, labels_each=40
            ),
            screening_already_in_additions=True,
            new_screening_calls=0,
            new_labels=0,
            new_human_judgments=0,
            human_review="Separate completed human study, not feedback in this transfer; its 96 judgments/100 participant-minutes are not charged again.",
            other_unmeasured=[
                "Initial BGE embedding/index creation",
                "Existing label creation and curation",
                "Prior classifier selection",
                "Initial broad retrieval and correction-prompt preparation",
                "Human research/review hours",
                "Remote GPU time/energy and unknown timeout work",
            ],
            qualification="Three retained forty-label pools do not equal forty labels consulted. Never sum inherited screening or nested classifier subsets into the new scientific POST row.",
        ),
    )
    files = [
        path
        for path in OUT.iterdir()
        if path.is_file()
        and path.name not in ("analysis_contract.json", "focused_tests.json")
    ]
    save(
        "assembled.json",
        dict(
            status="part9_complete_resource_census",
            created_utc=a.utc(),
            elapsed_local_seconds=time.monotonic() - began,
            input_bindings=bindings,
            artifacts={a.rel(path): a.sha(path) for path in files},
            predictions=189096,
            inputs=15758,
            groups=len(values),
            retry_result_files=len(failure_sources),
            retry_events=4195,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print("Resource census, chronology and saved-fit attribution complete.", flush=True)


if __name__ == "__main__":
    with a.offline():
        if sys.argv[1] == "prepare":
            prepare()
        elif sys.argv[1] == "assemble":
            assemble()
        else:
            raise ValueError("prepare or assemble")

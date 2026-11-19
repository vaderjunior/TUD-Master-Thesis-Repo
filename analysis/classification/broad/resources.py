import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
import numpy as np
from analysis.classification.broad import robustness as prior

reader = prior.reader
r = prior.r
c = prior.c
ROOT = prior.ROOT
BASE = ROOT / "experiments/astra/sq2_broad_resources_review_20261104_v1"
PRIOR = prior.BASE
CONDITIONS = prior.CONDITIONS
require = prior.require
bindings = prior.bindings
FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens")
EVENT_COLUMNS = (
    "record",
    "vote",
    "turn",
    "attempt",
    "ok",
    "seconds",
    "backoff",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "raw_chars",
    "raw_utf8_bytes",
)


def save(path, value):
    path = Path(path).resolve()
    require(path.is_relative_to(BASE.resolve()), "Output outside Part 8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        f.write(c.encoded(prior.safe(value)))


def dt(value):
    return datetime.fromisoformat(value)


def seconds(start, end):
    value = (dt(end) - dt(start)).total_seconds()
    require(value >= 0, "Reversed time interval")
    return value


def freeze():
    paths = [
        Path(__file__),
        ROOT / "Thesis_Log/astra_result/astra_sq2_broad_analysis_plan_20261104_v1.md",
        ROOT
        / "Thesis_Log/astra_result/astra_remaining_work_before_writing_20261104_v150.md",
        PRIOR / "part7_complete.json",
        PRIOR / "final_review.json",
        reader.BASE / "predictions.jsonl",
        reader.BASE / "item_bindings.json",
        reader.BASE / "reader_review.json",
        r.BASE / "live/complete.json",
        r.BASE / "storage_complete.json",
        r.BASE / "concurrency_20261029_v1/live_complete.json",
        r.PLAN,
        r.MANIFEST,
        ROOT / "analysis/common/api.py",
        ROOT / "analysis/classification/broad/execution.py",
        c.BASE / "retrieval_v3/complete.json",
        c.BASE / "retrieval_v3/plan.json",
        ROOT
        / "experiments/astra/sq2_broad_context_review_20261104_v1/part6_complete.json",
        ROOT
        / "experiments/astra/sq2_broad_context_review_20261104_v1/prompt_descriptors.jsonl",
        ROOT
        / "Thesis_Log/astra_result/astra_sq2_gpu_resources_practical_value_20261104_v1.md",
        ROOT / "experiments/astra/sq2_matched_review_20261030_v1/resources.json",
    ]
    save(
        BASE / "analysis_contract.json",
        dict(
            status="part8_scope_recorded_before_resource_extraction",
            created_utc=c.utc(),
            sources=bindings(paths),
            prior_knowledge="Parts 1–7 are complete. Discovery read the sealed whole-run totals and retrieval-pass summaries; no condition cost rankings or new quality scores were selected.",
            accounting="Consume every saved result once by the verified Part 1 cache order and result hash. Reconcile every event/turn/vote, ordinary/repair/POST/retry counts, reported token usage and recorded latency against all 52 original audit reports and final completion. Do not reread the raw request tree.",
            strata=[
                "whole run",
                "condition",
                "phase/condition",
                "phase/condition/repeat",
                "source/condition",
                "batch",
                "ordinary versus repair",
                "successful versus timeout attempts",
                "null versus partial versus complete outputs",
            ],
            time="Summed HTTP timers and backoff are overlapping client request time, not GPU compute or wall time. Preserve each batch start/summary/complete/audit timestamp, 128-worker provenance, all attempts and any restart gaps. Final completion mtime is only a proxy. Do not assign unexplained gaps to training, outage or labour.",
            output_sizes="Reported prompt/output tokens by successful event, actual saved generated text characters/UTF-8 bytes, and delivered rationale sizes. Repair tokens include their actual extended prompt. Missing token usage, failed attempts and cached-token details remain explicit. No inferred failed-response tokens.",
            preparation="Inspect bound retrieval passes and checkpoint-resume semantics, reused vectors, retained KB/index/vector/prompt file sizes without loading models or opening a retrieval database. Record historical/shared versus incremental boundaries and incomplete preparation timing.",
            quality="Reuse original Part 2 common-nine scores and fixed fields/masks for descriptive cost-quality alignment. No new F1, interval, best-repeat or one-call-system selection. Per-answer cost includes the whole structured response, not a separately billed head. Keep Part 4 exactness and Part 7 limitations.",
            comparisons="Full/small/zero have the same inputs and output contract, but their request timing can be confounded by shared load and schedule. Classifier/matched-study context retains distinct workloads/hardware/information budgets; no common monetary or energy ranking.",
            verification="Scientific accounting fixtures, independent aggregation from compact event/record arrays, original audit reconciliation, deterministic raw-record sample, derived-value/table/figure/link review, and immutable source hashes.",
            prohibited=[
                "network",
                "Qwen calls",
                "model loading",
                "inference",
                "training",
                "retrieval/index queries",
                "gold edits",
                "source mutation",
                "server actions",
                "deletion",
                "writer edits",
            ],
            deferred=[
                "Standalone broader synthesis Part 9",
                "Combined SQ2 synthesis and global resource/journal update",
            ],
        ),
    )


def extract_record(value, metadata, index):
    require(
        value["identity"]["key"] == metadata["key"]
        and value["identity"]["repeat"] == metadata["repeat"]
        and (value["identity"]["condition"] == "broad::" + metadata["condition"]),
        "Resource identity differs",
    )
    require(
        value["messages_sha256"] == metadata["messages_sha256"]
        and value["n_valid"] == metadata["n_valid"],
        "Resource/cache binding differs",
    )
    require(len(value["latencies"]) == len(value["raw_runs"]) == 3, "Vote slots differ")
    usage = Counter()
    repair_usage = Counter()
    ordinary_usage = Counter()
    classes = Counter()
    counts = Counter(
        {
            k: 0
            for k in (
                "ordinary_logical_calls",
                "repair_logical_calls",
                "post_attempts",
                "retry_attempts",
                "retryable_failures",
                "http_success_with_text",
                "reported_model_present",
            )
        }
    )
    observed = []
    requests = []
    offsets = Counter()
    last_attempt = {}
    seen = set()
    per_vote = np.zeros(3)
    success_keys = set()
    timings = Counter()
    for event in value["events"]:
        vote, turn, attempt = (event[k] for k in ("vote", "turn", "attempt"))
        require(
            type(vote) is int
            and 0 <= vote < 3
            and (type(turn) is int)
            and (0 <= turn < 3)
            and (type(attempt) is int)
            and (0 <= attempt < 5),
            "Event slots invalid",
        )
        ident = (vote, turn, attempt)
        require(ident not in seen, "Duplicate attempt")
        seen.add(ident)
        require(
            event["kind"] == ("repair" if turn else "ordinary"),
            "Call kind/turn differs",
        )
        require(
            attempt == last_attempt.get((vote, turn), -1) + 1,
            "Attempt sequence differs",
        )
        last_attempt[vote, turn] = attempt
        require((vote, turn) not in success_keys, "Attempt after success")
        elapsed, backoff = (event["elapsed_s"], event["backoff_elapsed_s"])
        require(
            all(
                (
                    type(v) in (float, int) and math.isfinite(v) and (v >= 0)
                    for v in (elapsed, backoff)
                )
            ),
            "Invalid time",
        )
        require(attempt > 0 or backoff == 0, "Initial attempt has backoff")
        ok = event["action"] == "ok"
        require(
            ok or event["action"] == "retry",
            "Unexpected terminal event in completed result",
        )
        counts.update(
            post_attempts=1,
            retry_attempts=int(attempt > 0),
            retryable_failures=int(not ok),
            http_success_with_text=int(ok),
        )
        classes[event["response_class"]] += 1
        group = "repair" if turn else "ordinary"
        timings[group + "_seconds"] += elapsed
        timings[group + "_backoff"] += backoff
        timings["success_seconds" if ok else "failed_seconds"] += elapsed
        per_vote[vote] += elapsed + backoff
        tokens = [0, 0, 0]
        chars = rawbytes = 0
        if ok:
            require(
                event["response_class"] == "success", "Successful event class differs"
            )
            mm = event["model_response_metadata"]
            require(
                mm["reported_model"]
                == value["requested_model"]
                == value["active_model"],
                "Reported model differs",
            )
            u = mm["usage"]
            require(
                isinstance(u, dict)
                and all((type(u.get(k)) is int and u[k] >= 0 for k in FIELDS)),
                "Missing or invalid recorded token usage",
            )
            require(
                u["prompt_tokens"] + u["completion_tokens"] == u["total_tokens"],
                "Token sum differs",
            )
            require(turn == offsets[vote], "Response/repair order differs")
            raw = value["raw_runs"][vote][turn]
            chars = len(raw)
            rawbytes = len(raw.encode("utf8"))
            offsets[vote] += 1
            tokens = [u[k] for k in FIELDS]
            usage.update(dict(zip(FIELDS, tokens)))
            (repair_usage if turn else ordinary_usage).update(dict(zip(FIELDS, tokens)))
            counts["reported_model_present"] += 1
            counts["repair_logical_calls" if turn else "ordinary_logical_calls"] += 1
            success_keys.add((vote, turn))
            observed.append(
                dict(
                    prompt_details=u.get("prompt_tokens_details"),
                    completion_details=u.get("completion_tokens_details"),
                    finish_reason=mm.get("finish_reason"),
                )
            )
        else:
            require(
                not event.get("model_response_metadata"),
                "Failed event unexpectedly contains response metadata",
            )
        requests.append(
            [
                index,
                vote,
                turn,
                attempt,
                int(ok),
                elapsed,
                backoff,
                *tokens,
                chars,
                rawbytes,
            ]
        )
    require(
        all((offsets[i] == len(chain) for i, chain in enumerate(value["raw_runs"]))),
        "Saved text/event count differs",
    )
    require(
        dict(counts) == value["accounting"], "Per-result request accounting differs"
    )
    require(
        counts["ordinary_logical_calls"] == 3
        and counts["repair_logical_calls"] == value["repairs"],
        "Logical calls differ",
    )
    require(
        np.allclose(per_vote, value["latencies"], rtol=0, atol=1e-08),
        "Recorded vote time differs",
    )
    item = dict(
        index=index,
        **{
            k: metadata[k]
            for k in (
                "key",
                "phase",
                "source",
                "condition",
                "repeat",
                "stage",
                "n_valid",
                "repairs",
                "parse_failures",
                "gate_normalised",
                "normalisations",
                "uncertain",
                "result_path",
            )
        },
        accounting=dict(counts),
        usage=dict(usage),
        repair_usage={k: repair_usage[k] for k in FIELDS},
        ordinary_usage=dict(ordinary_usage),
        response_classes=dict(classes),
        request_seconds=sum((e["elapsed_s"] for e in value["events"])),
        backoff_seconds=sum((e["backoff_elapsed_s"] for e in value["events"])),
        vote_seconds=list(per_vote),
        time_components=dict(timings),
        delivered_rationale_chars=(
            len(value["result"]["reasoning"]) if value["result"] is not None else None
        ),
        delivered_rationale_utf8_bytes=(
            len(value["result"]["reasoning"].encode("utf8"))
            if value["result"] is not None
            else None
        ),
        success_raw_chars=sum((v[-2] for v in requests)),
        success_raw_utf8_bytes=sum((v[-1] for v in requests))
    )
    return (item, requests, observed)


def extract():
    contract = c.read(BASE / "analysis_contract.json")
    r.verify(contract["sources"])
    require(
        c.read(BASE / "tests.json")["status"] == "passed", "Accounting tests required"
    )
    original = c.read(reader.BASE / "item_bindings.json")
    final = c.read(r.BASE / "live/complete.json")
    r.verify(final["review_hashes"])
    accumulated = defaultdict(
        lambda: dict(
            counts=Counter(), accounting=Counter(), usage=Counter(), vote_seconds=0.0
        )
    )
    event_rows = []
    metadata = Counter()
    record_bytes = 0
    start = time.perf_counter()
    with (BASE / "records.jsonl").open("x", encoding="utf8", newline="\n") as out:
        for index, row in enumerate(reader.rows(reader.BASE / "predictions.jsonl")):
            p = Path(row["result_path"])
            data = p.read_bytes()
            require(
                hashlib.sha256(data).hexdigest() == original[row["result_path"]],
                "Original result hash differs",
            )
            record_bytes += len(data)
            value = json.loads(data)
            item, events, reported = extract_record(value, row, index)
            item["result_sha256"] = original[row["result_path"]]
            out.write(
                json.dumps(item, sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            )
            event_rows.extend(events)
            metadata.update((json.dumps(v, sort_keys=True) for v in reported))
            group = accumulated[row["stage"]]
            group["counts"].update(
                dict(
                    predictions=1,
                    nulls=int(row["n_valid"] == 0),
                    gate_ties=int(row["uncertain"]),
                    **{"valid_votes_" + str(row["n_valid"]): 1},
                    **{
                        k: row[k]
                        for k in (
                            "repairs",
                            "parse_failures",
                            "gate_normalised",
                            "normalisations",
                        )
                    }
                )
            )
            group["accounting"].update(item["accounting"])
            group["usage"].update(item["usage"])
            group["vote_seconds"] += sum(item["vote_seconds"])
            if (index + 1) % 10000 == 0:
                print(
                    "Saved request accounting:",
                    index + 1,
                    "/194643;",
                    round(time.perf_counter() - start, 1),
                    "seconds",
                    flush=True,
                )
    require(index + 1 == 194643, "Prediction matrix incomplete")
    stages = []
    bound = {}
    totals = {k: Counter() for k in ("counts", "accounting", "usage")}
    runtime = c.read(r.BASE / "concurrency_20261029_v1/live_complete.json")
    r.verify(runtime["provenance_hashes"])
    provenance = {
        c.read(p)["original_attempt"]: c.read(p) for p in runtime["provenance_hashes"]
    }
    for sid in sorted(accumulated):
        root = r.BASE / "live/broad/stages" / sid
        review = c.read(root / "review.json")
        report = c.read(root / review["report"])
        complete = c.read(root / "complete.json")
        r.verify(
            {
                r.reference(root / k): v
                for k, v in review["artifacts"].items()
                if not k.endswith(("raw_hashes.json", "response_ids.json"))
            }
        )
        require(
            c.sha(root / "complete.json") == review["complete_sha256"],
            "Stage complete differs",
        )
        group = accumulated[sid]
        for key, field in [
            ("counts", "counts"),
            ("accounting", "accounting"),
            ("usage", "observed_usage"),
        ]:
            require(
                dict(group[key]) == report[field], "Stage " + key + " differs: " + sid
            )
            totals[key].update(group[key])
        require(
            abs(group["vote_seconds"] - report["vote_seconds_sum"]) < 1e-05,
            "Stage vote seconds differ",
        )
        attempts = []
        for folder in sorted(root.glob("attempt_*")):
            start_record = c.read(folder / "start.json")
            summary = c.read(folder / "summary.json")
            check = c.read(folder / "model_check_response.json")
            pv = provenance[r.reference(folder)]
            require(
                pv["effective_workers"] == 128
                and pv["original_start_sha256"] == c.sha(folder / "start.json"),
                "Worker provenance differs",
            )
            require(
                not summary["errors"]
                and (not summary["stopped"])
                and (summary["unsubmitted"] == 0),
                "Stopped attempt needs explicit review",
            )
            require(check["status"] == 200, "Availability check failed")
            attempts.append(
                dict(
                    path=r.reference(folder),
                    start=start_record["started_utc"],
                    end=summary["finished_utc"],
                    seconds=seconds(
                        start_record["started_utc"], summary["finished_utc"]
                    ),
                    peak_workers=summary["peak_submitted"],
                    workers=128,
                    saved_items=summary["saved_items"],
                    submitted=summary["submitted"],
                    model_check_seconds=check["elapsed_s"],
                    model_check_method="GET /models, no generated completion",
                )
            )
            for p in folder.iterdir():
                if p.is_file() and p.name in (
                    "start.json",
                    "summary.json",
                    "model_check_response.json",
                    "model_check_intent.json",
                ):
                    bound[r.reference(p)] = c.sha(p)
        require(len(attempts) == report["attempt_count"], "Attempt count differs")
        stages.append(
            dict(
                stage=sid,
                counts=dict(group["counts"]),
                accounting=dict(group["accounting"]),
                usage=dict(group["usage"]),
                vote_seconds=group["vote_seconds"],
                attempts=attempts,
                generation_complete=complete["completed_utc"],
                audit_complete=report["created_utc"],
                audit_report_wall_seconds=report["wall_seconds_including_restarts"],
                http_statuses=report["http_statuses"],
                actual_audit_API_calls=report["actual_audit_API_calls"],
            )
        )
        bound.update(
            bindings(
                [root / "review.json", root / "complete.json", root / review["report"]]
            )
        )
    require(len(stages) == 52, "Expected 52 batches")
    for key, field in [
        ("counts", "counts"),
        ("accounting", "accounting"),
        ("usage", "observed_usage"),
    ]:
        require(dict(totals[key]) == final[field], "Whole-study " + key + " differs")
    with (BASE / "events.npz").open("xb") as f:
        np.savez_compressed(
            f,
            values=np.asarray(event_rows, dtype=np.float64),
            columns=np.asarray(EVENT_COLUMNS),
        )
    save(
        BASE / "execution_inventory.json",
        dict(
            stages=stages,
            totals=totals,
            model_response_metadata=metadata,
            final_completion_mtime_utc=datetime.fromtimestamp(
                (r.BASE / "live/complete.json").stat().st_mtime, timezone.utc
            ).isoformat(),
            final_mtime_is_proxy=True,
            original_result_bytes_consumed=record_bytes,
            source_bindings=bound | runtime["provenance_hashes"],
        ),
    )
    save(
        BASE / "extraction_complete.json",
        dict(
            status="all_saved_result_request_resources_reconciled",
            created_utc=c.utc(),
            predictions=index + 1,
            events=len(event_rows),
            elapsed_reader_seconds=time.perf_counter() - start,
            sources=bindings(
                [
                    Path(__file__),
                    BASE / "analysis_contract.json",
                    BASE / "tests.json",
                    reader.BASE / "item_bindings.json",
                    reader.BASE / "predictions.jsonl",
                    r.BASE / "live/complete.json",
                    r.BASE / "concurrency_20261029_v1/live_complete.json",
                ]
            ),
            outputs=bindings(
                [
                    BASE / "records.jsonl",
                    BASE / "events.npz",
                    BASE / "execution_inventory.json",
                ]
            ),
            new_Qwen_calls=0,
        ),
    )
    print(
        "Complete resource extraction:",
        index + 1,
        "predictions;",
        len(event_rows),
        "transport attempts.",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["freeze", "extract"], required=True)
    args = parser.parse_args()
    with r.original.offline():
        freeze() if args.mode == "freeze" else extract()

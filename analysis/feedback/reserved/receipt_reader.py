import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import time

from analysis.feedback.reserved import review as io

ROOT = io.ROOT
OUT = ROOT / "experiments/astra/sq3_reserved_audited_assembly_20261019_v1"
READY = (
    ROOT
    / "experiments/astra/sq3_reserved_part1_integrity_20261019_v1/readiness/check.json"
)
LIVE = io.EXECUTION / "live/reserved_slot"
need = io.require


def write(path, value):
    need(path.resolve().is_relative_to(OUT.resolve()), "Write outside new assembly")
    path.parent.mkdir(parents=True, exist_ok=True)
    text = (
        value
        if isinstance(value, str)
        else json.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n"
    )
    with path.open("x", encoding="utf-8", newline="\n") as f:
        f.write(text)


def read_bound(path, digest):
    raw = io.physical(path).read_bytes()
    need(hashlib.sha256(raw).hexdigest() == digest, "Changed bound file: " + str(path))
    return json.loads(raw)


def item_id(job, plan_sha):
    ident = io.identity(job, plan_sha)
    return hashlib.sha256(
        (
            json.dumps(
                ident, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False
            )
            + "\n"
        ).encode("utf8")
    ).hexdigest()


def compact(saved, job, plan_sha, receipt_path, receipt_sha, result_sha):
    need(saved["identity"] == io.identity(job, plan_sha), "Result identity differs")
    need(
        saved["execution_mode"] == "live"
        and saved["messages_sha256"] == job["messages_sha256"],
        "Wrong mode or messages",
    )
    need(
        saved["n_runs"] == 3 and len(saved["run_predictions"]) == 3, "Wrong vote count"
    )
    need(
        type(saved["n_valid"]) is int
        and saved["n_valid"] == sum(v is not None for v in saved["run_predictions"]),
        "Valid vote count differs",
    )
    for vote in saved["run_predictions"]:
        if vote is not None:
            need(type(vote["hate"]) is bool, "Nonboolean saved vote")
    final, tie = io.independent_vote(saved["run_predictions"])
    need(
        final
        == (
            {k: saved["result"][k] for k in io.FIELDS}
            if saved["result"] is not None
            else None
        ),
        "Saved vote aggregation differs",
    )
    need(tie == saved["uncertain"], "Tie indicator differs")
    usage, ordinary, repair = Counter(), Counter(), Counter()
    durations, backoffs, unknown = [], [], 0
    for event in saved["events"]:
        need(
            type(event["elapsed_s"]) in (int, float)
            and math.isfinite(event["elapsed_s"])
            and event["elapsed_s"] >= 0,
            "Invalid duration",
        )
        need(
            type(event["backoff_elapsed_s"]) in (int, float)
            and math.isfinite(event["backoff_elapsed_s"])
            and event["backoff_elapsed_s"] >= 0,
            "Invalid backoff",
        )
        if event["response_class"] == io.ADMIN:
            need(
                event["elapsed_s"] == 0 and event["action"] == "retry",
                "Administrative unknown changed",
            )
            unknown += 1
        else:
            durations.append(event["elapsed_s"])
        backoffs.append(event["backoff_elapsed_s"])
        if event["action"] == "ok":
            u = event["model_response_metadata"]["usage"]
            need(
                all(
                    type(u.get(k)) is int and u[k] >= 0
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                )
                and u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"],
                "Bad event usage",
            )
            u = {
                k: u[k] for k in ("prompt_tokens", "completion_tokens", "total_tokens")
            }
            usage.update(u)
            (ordinary if event["turn"] == 0 else repair).update(u)
    accounting = saved["accounting"]
    need(
        len(durations) + unknown == accounting["post_attempts"] == len(saved["events"]),
        "Event/attempt totals differ",
    )
    need(
        sum(e["action"] == "ok" for e in saved["events"])
        == accounting["http_success_with_text"],
        "Successful event count differs",
    )
    return {
        "key": job["key"],
        "condition": job["condition"],
        "repeat": job["repeat"],
        "result": final,
        "n_valid": saved["n_valid"],
        "uncertain": tie,
        "messages_sha256": job["messages_sha256"],
        "item_receipt": receipt_path,
        "item_receipt_sha256": receipt_sha,
        "result_sha256": result_sha,
        "accounting": accounting,
        "observed_usage": dict(usage),
        "ordinary_usage": dict(ordinary),
        "repair_usage": dict(repair),
        "known_request_durations_s": durations,
        "known_request_seconds_sum": math.fsum(durations),
        "known_request_duration_count": len(durations),
        "retry_backoff_seconds_sum": math.fsum(backoffs),
        "administrative_unknown_requests": unknown,
        **{
            k: saved[k]
            for k in ("repairs", "parse_failures", "normalisations", "gate_normalised")
        },
    }


def read_item(job, expected_receipts, plan_sha, root=LIVE):
    name = "items/" + item_id(job, plan_sha)
    receipt_name = name + "/complete.json"
    receipt_sha = expected_receipts[receipt_name]
    receipt = read_bound(root / receipt_name, receipt_sha)
    need(receipt["status"] == "complete", "Incomplete item receipt")
    result_name = name + "/result.json"
    result_sha = receipt["hashes"][result_name]
    saved = read_bound(root / result_name, result_sha)
    return compact(
        saved, job, plan_sha, io.rel(root / receipt_name), receipt_sha, result_sha
    )


def build():
    started_at = time.monotonic()
    done = io.completion_gate()
    need(not OUT.exists(), "Preserve existing assembly; inspect before another version")
    ready = io.read(READY)
    need(
        ready["status"] == "part1_user_revised_readiness_scope_passed",
        "Focused readiness has not passed",
    )
    io.verify(ROOT, ready["sources"])
    plan = io.read(io.EXECUTION / "plan.json")
    io.verify(ROOT, plan["source_hashes"])
    io.verify(ROOT, plan["input_hashes"])
    schedule = io.read(io.EXECUTION / "schedule.json")
    science = io.read(io.STUDY / "scientific_plan_v1.json")
    roster = list(io.rows(io.STUDY / "roster.jsonl"))
    need(
        len(roster) == 675
        and len(science["conditions"]) == 74
        and science["decoder_repeats"] == 3,
        "Wrong scientific matrix",
    )
    expected = {
        (r["key"], c, rep)
        for r in roster
        for c in science["conditions"]
        for rep in range(3)
    }
    source_hashes = io.reviewer_sources()
    source_hashes[io.rel(Path(__file__))] = io.sha(Path(__file__))
    bindings = {
        **plan["input_hashes"],
        **plan["source_hashes"],
        **source_hashes,
        io.rel(READY): io.sha(READY),
        io.rel(io.EXECUTION / "live/complete.json"): io.sha(
            io.EXECUTION / "live/complete.json"
        ),
    }
    write(
        OUT / "started.json",
        {
            "status": "assembling_already_audited_results",
            "started_utc": io.utc(),
            "input_bindings": bindings,
            "verification_basis": "Automatic semantic audits + final integrity; freshly hash-check consumed item receipts/results and reaggregate saved votes.",
            "full_raw_reparse": False,
            "new_model_calls": 0,
            "gold_scoring": False,
        },
    )
    seen, batch_by_key, artifacts = set(), {}, {}
    total_counts, total_accounting, total_usage = Counter(), Counter(), Counter()
    total_unknown = 0

    with ThreadPoolExecutor(max_workers=8) as executor:
        for stage in schedule["stages"]:
            target = LIVE / "stages" / stage["id"]
            review = io.read(target / "review.json")
            need(
                io.sha(target / "review.json")
                == done["review_hashes"][io.rel(target / "review.json")],
                "Stage/final binding changed",
            )
            io.verify(target, review["artifacts"])
            complete = read_bound(target / "complete.json", review["complete_sha256"])
            report = io.read(target / review["report"])
            need(
                report["stage_complete_sha256"] == review["complete_sha256"],
                "Audit/stage binding changed",
            )
            need(
                not complete["summary"]["stopped"]
                and not complete["summary"]["errors"],
                "Incomplete stage",
            )
            receipts = complete["item_receipt_hashes"]
            need(
                set(receipts)
                == {
                    "items/" + item_id(j, io.PLAN_SHA) + "/complete.json"
                    for j in stage["jobs"]
                },
                "Stage matrix differs",
            )
            records = list(
                executor.map(
                    lambda job: read_item(job, receipts, io.PLAN_SHA), stage["jobs"]
                )
            )
            counts, accounting, usage = Counter(), Counter(), Counter()
            for row in records:
                key = (row["key"], row["condition"], row["repeat"])
                need(
                    key not in seen and key in expected,
                    "Duplicate/unplanned prediction",
                )
                seen.add(key)
                need(
                    batch_by_key.get(row["key"], stage["id"]) == stage["id"],
                    "Text split across batches",
                )
                batch_by_key[row["key"]] = stage["id"]
                row["stage"] = stage["id"]
                counts["predictions"] += 1
                counts["nulls"] += row["result"] is None
                counts["valid_votes_" + str(row["n_valid"])] += 1
                counts["gate_ties"] += row["uncertain"]
                for field in (
                    "repairs",
                    "parse_failures",
                    "normalisations",
                    "gate_normalised",
                ):
                    counts[field] += row[field]
                accounting.update(row["accounting"])
                usage.update(row["observed_usage"])
                total_unknown += row["administrative_unknown_requests"]
            need(
                counts == Counter(report["counts"])
                and accounting == Counter(report["accounting"])
                and usage == Counter(report["observed_usage"]),
                "Stage aggregate mismatch",
            )
            total_counts.update(counts)
            total_accounting.update(accounting)
            total_usage.update(usage)
            path = OUT / "assembly" / (stage["id"] + ".jsonl")
            write(
                path,
                "".join(
                    json.dumps(r, ensure_ascii=False, sort_keys=True, allow_nan=False)
                    + "\n"
                    for r in records
                ),
            )
            artifacts[io.rel(path)] = io.sha(path)
            print(
                f"Audited receipt reader: {stage['id']} complete; {len(seen)}/149850 saved predictions",
                flush=True,
            )
    need(
        seen == expected
        and total_counts == Counter(done["counts"])
        and total_accounting == Counter(done["accounting"])
        and total_usage == Counter(done["observed_usage"]),
        "Final assembly mismatch",
    )
    need(
        total_unknown == done["administrative_unknown_requests"] == 8,
        "Unknown request accounting differs",
    )

    first = list(io.rows(io.OUTPUT / "assembly/reserved_slot_01.jsonl"))
    new_first = list(io.rows(OUT / "assembly/reserved_slot_01.jsonl"))
    need(len(first) == len(new_first) == 9546, "First batch size differs")
    for old, new in zip(first, new_first):
        need(
            all(new[k] == v for k, v in old.items()),
            "Audited-reader output differs from completed independent first-batch replay",
        )
    write(OUT / "batches.json", batch_by_key)
    artifacts[io.rel(OUT / "batches.json")] = io.sha(OUT / "batches.json")
    io.verify(ROOT, bindings)
    write(
        OUT / "complete.json",
        {
            "status": "complete_from_automatically_audited_receipts_unscored",
            "completed_utc": io.utc(),
            "elapsed_seconds": time.monotonic() - started_at,
            "artifacts": artifacts,
            "input_bindings": bindings,
            "plan_sha256": io.PLAN_SHA,
            "scientific_plan_sha256": io.sha(io.STUDY / "scientific_plan_v1.json"),
            "counts": dict(total_counts),
            "accounting": dict(total_accounting),
            "observed_usage": dict(total_usage),
            "administrative_unknown_requests": total_unknown,
            "first_batch_matches_independent_raw_replay": 9546,
            "verification_basis": "Every item receipt and result hash verified against automatic audit chain; saved votes independently reaggregated. Raw parse/transport semantics rely on original completed audits.",
            "full_raw_reparse": False,
            "gold_scoring": False,
            "new_model_calls": 0,
        },
    )
    print("Audited-receipt assembly complete; no scoring and no new calls.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    with io.offline():
        if args.build:
            build()
        else:
            io.completion_gate()
            print(
                "Automatic completion gate passed; use --build for separate receipt assembly."
            )

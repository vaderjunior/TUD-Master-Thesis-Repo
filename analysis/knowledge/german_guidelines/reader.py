"""German guidelines reader."""

import os

os.environ.update(
    CUDA_VISIBLE_DEVICES="",
    OMP_NUM_THREADS="1",
    MKL_NUM_THREADS="1",
    OPENBLAS_NUM_THREADS="1",
    NUMEXPR_NUM_THREADS="1",
    PYTHONDONTWRITEBYTECODE="1",
)
import argparse
import base64
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import time
import unicodedata
from unittest.mock import patch
import yaml
from analysis.knowledge.german_guidelines import common as c
from analysis.common import schema as isolated
from analysis.common import result_reader as scoring

BASE = c.ROOT / "experiments/astra/german_guidelines_reader_review_20261112_v1"
LIVE = c.OUTPUT / "live"
ANALYSIS_PLAN = (
    c.ROOT / "Thesis_Log/astra_result/german_guidelines_analysis_plan_20261112_v1.md"
)
FIELDS = ("hate", "target_group", "hate_type", "severity", "legal")
LIST_FIELDS = ("target_group", "hate_type", "legal")
DECISION_FIELDS = set(FIELDS) | {"reasoning"}
VALUE_FIELDS = {
    "accounting",
    "active_model",
    "agreement",
    "events",
    "execution_mode",
    "gate_normalised",
    "identity",
    "latencies",
    "messages_sha256",
    "n_runs",
    "n_valid",
    "normalisations",
    "parse_failures",
    "raw_runs",
    "repairs",
    "requested_model",
    "result",
    "run_predictions",
    "uncertain",
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def save(path, value):
    require(
        c.normal(path).is_relative_to(BASE),
        "Derived write outside reader-review folder",
    )
    c.save(path, value)


def write_rows(path, values):
    require(
        c.normal(path).is_relative_to(BASE), "Derived rows outside reader-review folder"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as handle:
        for value in values:
            handle.write(
                json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            )


def checked_json(path, expected):
    data = c.a.storage_path(path).read_bytes()
    require(
        hashlib.sha256(data).hexdigest() == expected,
        "Changed audited artifact: " + c.ref(path),
    )
    return json.loads(data)


def timestamp(text):
    value = datetime.fromisoformat(text)
    require(value.tzinfo is not None, "Saved timestamp lacks timezone")
    return value


def identity(job, plan_sha):
    return {
        "experiment_sha256": plan_sha,
        "condition": job["condition"],
        "repeat": job["repeat"],
        "key": job["key"],
    }


def own_sources():
    return [
        Path(__file__),
        ANALYSIS_PLAN,
        c.ROOT / "tests/knowledge/german_guidelines/test_reader.py",
        c.ROOT / "analysis/knowledge/german_guidelines/validate_reader.py",
    ]


@contextmanager
def offline():
    with c.offline(), patch.object(
        c.a.RequestsTransport, "__call__", c.forbidden
    ), patch.object(scoring.original, "score", c.forbidden), patch.object(
        scoring.original, "macro", c.forbidden
    ), patch.object(
        scoring, "contrast", c.forbidden
    ):
        yield


def check_decision(value, taxonomy, duplicates=False):
    if value is None:
        return
    require(
        type(value) is dict and set(value) == DECISION_FIELDS,
        "Unexpected decision fields",
    )
    require(
        type(value["hate"]) is bool and type(value["reasoning"]) is str,
        "Malformed gate/rationale",
    )
    for field in LIST_FIELDS:
        labels = value[field]
        require(
            type(labels) is list and all((type(x) is str for x in labels)),
            "Malformed label list",
        )
        require(
            set(labels) <= set(taxonomy["dimensions"][field]["labels"]),
            "Unknown predicted label",
        )
        require(
            duplicates or len(labels) == len(set(labels)),
            "Duplicate delivered aggregate label",
        )
    require(
        value["severity"] is None
        or value["severity"] in taxonomy["dimensions"]["severity"]["labels"],
        "Unknown severity label",
    )
    if not value["hate"]:
        require(
            value["target_group"] == value["hate_type"] == []
            and value["severity"] is None,
            "Inconsistent delivered hate gate",
        )


def independent_vote(votes, severity_order, deduplicate=False):
    valid = [v for v in votes if v is not None]
    if not valid:
        return {"result": None, "n_valid": 0, "uncertain": False, "agreement": {}}
    n = len(valid)
    positives = sum((v["hate"] for v in valid))
    hate = positives * 2 > n
    agreement = {"hate": max(positives, n - positives) / n}
    result = {"hate": hate}
    for field in LIST_FIELDS:
        counts = Counter(
            (
                label
                for v in valid
                for label in (set(v[field]) if deduplicate else v[field])
            )
        )
        result[field] = sorted(
            (label for label, count in counts.items() if count * 2 > n)
        )
        agreement[field] = (
            min((count / n for count in counts.values())) if counts else 1.0
        )
    severities = sorted(
        (
            severity_order.index(v["severity"])
            for v in valid
            if v["severity"] is not None
        )
    )
    result["severity"] = (
        severity_order[severities[(len(severities) - 1) // 2]] if severities else None
    )
    agreement["severity"] = len(severities) / n if severities else 1.0
    if not hate:
        result.update(target_group=[], hate_type=[], severity=None)
    chosen = next((v for v in valid if v["hate"] == hate), valid[0])
    result["reasoning"] = chosen["reasoning"]
    agreement["_dropped"] = [
        field
        for field in LIST_FIELDS
        if any((v[field] for v in valid)) and (not result[field])
    ]
    return {
        "result": result,
        "n_valid": n,
        "uncertain": positives * 2 == n,
        "agreement": agreement,
    }


def check_value(value, taxonomy):
    require(set(value) == VALUE_FIELDS, "Saved result shape changed")
    require(
        value["execution_mode"] == "live"
        and value["active_model"] == value["requested_model"] == c.a.MODEL,
        "Mock result or different model in live reader",
    )
    votes, raws = (value["run_predictions"], value["raw_runs"])
    require(
        value["n_runs"] == len(votes) == len(raws) == len(value["latencies"]) == 3,
        "Expected exactly three inner vote slots",
    )
    require(
        all(
            (
                type(chain) is list
                and 1 <= len(chain) <= 3
                and all((type(t) is str for t in chain))
                for chain in raws
            )
        ),
        "Invalid saved repair chain",
    )
    for vote in votes:
        check_decision(vote, taxonomy, duplicates=True)
    check_decision(value["result"], taxonomy)
    order = list(taxonomy["dimensions"]["severity"]["labels"])
    expected = independent_vote(votes, order)
    for field in ("result", "n_valid", "uncertain", "agreement"):
        require(value[field] == expected[field], "Independent vote mismatch: " + field)
    require(
        value["parse_failures"] == 3 - value["n_valid"],
        "Invalid-vote accounting differs",
    )
    require(
        value["repairs"] == sum((len(chain) - 1 for chain in raws)),
        "Repair count differs",
    )
    for name in (
        "normalisations",
        "gate_normalised",
        "parse_failures",
        "repairs",
        "n_valid",
    ):
        require(
            type(value[name]) is int and value[name] >= 0, "Malformed count: " + name
        )
    require(value["gate_normalised"] <= value["n_valid"], "Too many normalized votes")
    require(
        all((c.a.finite_nonnegative(x) for x in value["latencies"])),
        "Malformed saved latency",
    )
    groups = defaultdict(list)
    for event in value["events"]:
        vote, turn, attempt = (event["vote"], event["turn"], event["attempt"])
        require(
            type(vote) is type(turn) is type(attempt) is int
            and 0 <= vote < 3
            and (0 <= turn < len(raws[vote]))
            and (0 <= attempt < 5),
            "Invalid call position",
        )
        require(
            event["kind"] == ("ordinary" if turn == 0 else "repair"),
            "Incorrect call kind",
        )
        require(
            c.a.finite_nonnegative(event["elapsed_s"])
            and c.a.finite_nonnegative(event["backoff_elapsed_s"]),
            "Invalid event time",
        )
        require(
            (
                event["backoff_elapsed_s"] == 0
                if attempt == 0
                else event["backoff_elapsed_s"]
                >= c.a.CONTRACT["backoff_seconds"][attempt - 1] - 0.1
            ),
            "Retry backoff outside contract",
        )
        groups[vote, turn].append(event)
    require(
        set(groups) == {(v, t) for v in range(3) for t in range(len(raws[v]))},
        "Missing logical call",
    )
    chronological = []
    for key in sorted(groups):
        events = groups[key]
        require(
            [e["attempt"] for e in events] == list(range(len(events))),
            "Gapped/repeated attempt sequence",
        )
        require(
            all(
                (
                    e["action"] == "retry" and e["model_response_metadata"] is None
                    for e in events[:-1]
                )
            )
            and events[-1]["action"] == "ok"
            and (events[-1]["response_class"] == "success"),
            "Invalid completed transport chain",
        )
        meta = events[-1]["model_response_metadata"]
        require(
            meta["reported_model"] in (c.a.MODEL, None)
            and meta["finish_reason"] in ("stop", None),
            "Unexpected reported model or finish reason",
        )
        chronological.extend(events)
    require(chronological == value["events"], "Saved event order differs")
    for vote in range(3):
        latency = sum(
            (
                e["elapsed_s"] + e["backoff_elapsed_s"]
                for e in value["events"]
                if e["vote"] == vote
            )
        )
        require(
            abs(latency - value["latencies"][vote]) <= 1e-09 * max(1, latency),
            "Vote latency accounting differs",
        )
    events = value["events"]
    accounting = {
        "ordinary_logical_calls": 3,
        "repair_logical_calls": value["repairs"],
        "post_attempts": len(events),
        "retry_attempts": sum((e["attempt"] > 0 for e in events)),
        "retryable_failures": sum((e["action"] == "retry" for e in events)),
        "reported_model_present": sum(
            (
                e["action"] == "ok"
                and e["model_response_metadata"]["reported_model"] is not None
                for e in events
            )
        ),
        "http_success_with_text": sum((e["action"] == "ok" for e in events)),
    }
    require(value["accounting"] == accounting, "Saved call accounting differs")
    duplicates = [
        {
            "vote": i,
            "field": field,
            "labels": {k: n for k, n in Counter(v[field]).items() if n > 1},
        }
        for i, v in enumerate(votes)
        if v is not None
        for field in LIST_FIELDS
        if len(v[field]) != len(set(v[field]))
    ]
    flags = []
    for name, active in [
        ("null", value["result"] is None),
        ("repair", value["repairs"] > 0),
        ("gate_normalization", value["gate_normalised"] > 0),
        ("tie", value["uncertain"]),
        ("duplicate_labels", bool(duplicates)),
        ("retry", accounting["retry_attempts"] > 0),
        (
            "legal_with_nonhate",
            value["result"] is not None
            and (not value["result"]["hate"])
            and bool(value["result"]["legal"]),
        ),
    ]:
        if active:
            flags.append(name)
    dedup = (
        independent_vote(votes, order, deduplicate=True)["result"]
        if duplicates
        else value["result"]
    )
    changed = [
        field
        for field in FIELDS
        if dedup is not None and dedup[field] != value["result"][field]
    ]
    return {
        "flags": flags,
        "duplicate_labels": duplicates,
        "duplicate_sensitive_fields": changed,
    }


def prompt_catalogue():
    cat = {}
    for row in c.rows(c.BASE / "prompts.jsonl"):
        key = (row["key"], row["condition"])
        require(
            key not in cat and c.signature(row["messages"]) == row["messages_sha256"],
            "Duplicate/changed prompt",
        )
        cat[key] = {
            k: row[k]
            for k in (
                "key",
                "population",
                "condition",
                "messages",
                "messages_sha256",
                "context_sha256",
                "reference_input_tokens",
                "original_messages_sha256",
            )
        }
        cat[key]["guideline_ids"] = [hit["id"] for hit in row["context"]["guidelines"]]
    require(len(cat) == 11031, "Prepared prompt count differs")
    return cat


def verify_schedule(schedule, cat, roster):
    expected = {
        (key, cond, repeat)
        for key in roster
        for cond in c.CONDITIONS
        for repeat in range(3)
    }
    seen, owners = (set(), {})
    for stage in schedule["stages"]:
        require(
            stage["items"] == len(stage["jobs"])
            and stage["ordinary_calls"] == 3 * stage["items"],
            "Batch budget differs",
        )
        require(
            set(stage["input_keys"]) == {j["key"] for j in stage["jobs"]},
            "Batch input roster differs",
        )
        for job in stage["jobs"]:
            ident = (job["key"], job["condition"], job["repeat"])
            require(
                ident in expected and ident not in seen,
                "Unexpected/duplicate prediction slot",
            )
            seen.add(ident)
            entry = cat[job["key"], job["condition"]]
            require(
                job["population"]
                == stage["population"]
                == roster[job["key"]]["phase"]
                == entry["population"]
                and job["messages_sha256"] == entry["messages_sha256"],
                "Schedule identity mismatch",
            )
            require(
                owners.setdefault(job["key"], stage["id"]) == stage["id"],
                "Input comparisons split across batches",
            )
    require(
        seen == expected and len(seen) == 33093 and (len(schedule["stages"]) == 27),
        "Incomplete frozen matrix",
    )


def check_batch(stage, expected_hash, plan_sha):
    directory = LIVE / "batches" / stage["id"]
    path = directory / "review.json"
    review = checked_json(path, expected_hash)
    require(
        review["status"] == "raw_responses_and_votes_verified"
        and review["mode"] == "live"
        and (review["plan_sha256"] == plan_sha)
        and (review["stage_sha256"] == c.signature(stage)),
        "Batch review identity differs",
    )
    expected_items = {c.signature(identity(job, plan_sha)) for job in stage["jobs"]}
    require(set(review["items"]) == expected_items, "Audited batch item roster differs")
    bindings = review["availability_hashes"]
    for name, digest in bindings.items():
        require(
            c.normal(name).is_relative_to(directory),
            "Batch receipt reference escapes its batch",
        )
        require(c.sha(name) == digest, "Changed batch execution/availability receipt")
    attempts = []
    for attempt in sorted(directory.glob("attempt_*")):
        start = c.read(attempt / "start.json")
        require(
            bindings.get(c.ref(attempt / "start.json"))
            == c.sha(attempt / "start.json"),
            "Unbound attempt start",
        )
        require(
            start["plan_sha256"] == plan_sha
            and start["stage_sha256"] == c.signature(stage)
            and (start["mode"] == "live")
            and (start["workers"] == 128),
            "Attempt contract differs",
        )
        summary = (
            c.read(attempt / "summary.json")
            if (attempt / "summary.json").exists()
            else None
        )
        stopped = (
            c.read(attempt / "stop.json") if (attempt / "stop.json").exists() else None
        )
        finish = (summary or stopped or {}).get("finished_utc")
        require(
            finish is not None and timestamp(start["started_utc"]) <= timestamp(finish),
            "Attempt has no valid final timestamp",
        )
        if summary:
            require(
                bindings.get(c.ref(attempt / "summary.json"))
                == c.sha(attempt / "summary.json"),
                "Unbound attempt summary",
            )
            require(
                0 <= summary["peak_submitted"] <= 128
                and summary["submitted"] <= start["pending_predictions"],
                "Worker/submission count differs",
            )
        ready = attempt / "availability_ready.json"
        if summary and summary["submitted"]:
            require(ready.exists(), "Submitted attempt lacks availability approval")
        model_seen = False
        if ready.exists():
            model_intent = c.read(attempt / "model_check_intent.json")
            response = c.read(attempt / "model_check_response.json")
            approval = c.read(ready)
            require(
                approval
                == {
                    "model": c.a.MODEL,
                    "response_sha256": c.sha(attempt / "model_check_response.json"),
                },
                "Availability approval changed",
            )
            raw = base64.b64decode(response["body_base64"], validate=True)
            require(
                hashlib.sha256(raw).hexdigest() == response["body_sha256"]
                and response["status"] == 200,
                "Model-list response differs",
            )
            model_seen = c.a.MODEL in {v["id"] for v in json.loads(raw)["data"]}
            require(
                model_seen and model_intent["model"] == c.a.MODEL,
                "Pinned model unavailable for recorded attempt",
            )
            require(
                timestamp(start["started_utc"])
                <= timestamp(model_intent["started_utc"])
                <= timestamp(finish),
                "Model check chronology differs",
            )
        actual_files = {c.ref(p) for p in attempt.iterdir() if p.is_file()}
        require(actual_files <= set(bindings), "Unreviewed execution-attempt artifact")
        attempts.append(
            {
                "attempt": c.ref(attempt),
                "start": start,
                "summary": summary,
                "stop": stopped,
                "finished_utc": finish,
                "pinned_model_available": model_seen,
            }
        )
    require(bool(attempts), "No execution attempt found")
    return (review, attempts)


def read_item(job, entry, hashes, plan_sha, backend):
    ident = identity(job, plan_sha)
    folder = LIVE / "items" / c.signature(ident)
    receipt = checked_json(folder / "complete.json", hashes["complete.json"])
    value = checked_json(folder / "result.json", hashes["result.json"])
    prefix = "items/" + c.signature(ident) + "/"
    require(
        receipt["status"] == "complete"
        and receipt["hashes"].get(prefix + "result.json") == hashes["result.json"],
        "Result not bound by item completion",
    )
    require(prefix + "job.json" in receipt["hashes"], "Item stamp missing from receipt")
    stamp = checked_json(folder / "job.json", receipt["hashes"][prefix + "job.json"])
    require(
        stamp
        == {
            "identity": ident,
            "messages_sha256": entry["messages_sha256"],
            "contract": c.a.CONTRACT,
            "implementation": backend.implementation(),
            "execution_mode": "live",
        },
        "Input/schema/source stamp differs",
    )
    require(receipt["job_stamp"] == c.signature(stamp), "Completion stamp differs")
    require(
        value["identity"] == ident
        and value["messages_sha256"] == entry["messages_sha256"],
        "Wrong prediction slot",
    )
    return (
        value,
        receipt,
        {
            "result_path": c.ref(folder / "result.json"),
            "result_sha256": hashes["result.json"],
            "receipt_path": c.ref(folder / "complete.json"),
            "receipt_sha256": hashes["complete.json"],
            "job_sha256": receipt["hashes"][prefix + "job.json"],
            "created_utc": receipt["created_utc"],
        },
    )


def replay_sample(job, value, cat, plan_sha, backend, attempts):
    folder = LIVE / "items" / c.signature(identity(job, plan_sha))
    audit = backend.audit_item(
        LIVE,
        identity(job, plan_sha),
        cat[job["key"], job["condition"]]["messages"],
        execution_mode="live",
    )
    parse_errors, normalisations, normalised_votes = ([], 0, 0)
    texts = 0
    for i, chain in enumerate(value["raw_runs"]):
        last = None
        for turn, raw in enumerate(chain):
            parsed, n, error = backend.bundle.parser.parse_response(raw)
            normalisations += n
            texts += 1
            require((parsed is None) == bool(error), "Parser error state differs")
            require(
                parsed is None or turn == len(chain) - 1,
                "Valid response followed by unnecessary repair",
            )
            if parsed is None:
                parse_errors.append({"vote": i, "turn": turn, "error": error})
            last = parsed
        require(
            (last.model_dump() if last is not None else None)
            == value["run_predictions"][i],
            "Raw-string parser replay differs",
        )
        normalised_votes += bool(last is not None and last.gate_normalised)
    require(
        normalisations == value["normalisations"]
        and normalised_votes == value["gate_normalised"],
        "Free-normalization accounting differs",
    )
    chronology = []
    previous = None
    for event in value["events"]:
        call = (
            folder
            / "calls"
            / f"vote_{event['vote']}_turn_{event['turn']}"
            / f"attempt_{event['attempt']}"
        )
        intent, outcome = (c.read(call / "intent.json"), c.read(call / "outcome.json"))
        start, finish = (
            timestamp(intent["started_utc"]),
            timestamp(outcome["finished_utc"]),
        )
        require(
            start <= finish and (previous is None or previous <= start),
            "Saved POST chronology differs",
        )
        owners = [
            r["attempt"]
            for r in attempts
            if timestamp(r["start"]["started_utc"])
            <= start
            <= finish
            <= timestamp(r["finished_utc"])
        ]
        require(len(owners) == 1, "Saved POST not inside exactly one recorded attempt")
        chronology.append(
            {
                "vote": event["vote"],
                "turn": event["turn"],
                "attempt": event["attempt"],
                "started_utc": intent["started_utc"],
                "finished_utc": outcome["finished_utc"],
                "batch_attempt": owners[0],
            }
        )
        previous = finish
    return {
        "identity": value["identity"],
        "raw_audit": audit,
        "raw_texts_reparsed": texts,
        "parse_errors": parse_errors,
        "chronology": chronology,
    }


def endpoint_setup(roster, frozen, validity):
    output = {}
    for phase in c.COUNTS:
        endpoint = frozen[phase]
        keys = endpoint["scorable_ids"]
        require(
            len(keys) == len(set(keys)) == c.COUNTS[phase]
            and set(keys) == {k for k, row in roster.items() if row["phase"] == phase},
            "Gold endpoint roster differs",
        )
        dimension = "legal" if "legal" in phase else "hate"
        require(endpoint["dimension"] == dimension, "Wrong scored dimension")
        support = Counter()
        labels = endpoint["all_labels"]
        values = []
        for key in keys:
            row = roster[key]
            value = row["gate"] if dimension == "hate" else row["legal"]
            require(
                type(value) is bool if dimension == "hate" else type(value) is list,
                "Missing/malformed gold annotation",
            )
            positive = (
                ["hate" if value else "not_hate"] if dimension == "hate" else value
            )
            require(set(positive) <= set(labels), "Gold label outside frozen universe")
            support.update(positive)
            values.append(value)
        fixed_labels = (
            labels
            if dimension == "hate"
            else [label for label in labels if support[label] >= 10]
        )
        require(
            endpoint["gold_support"] == {label: support[label] for label in labels}
            and endpoint["macro_labels"] == fixed_labels,
            "Full-roster label support differs",
        )
        cells = {}
        common = set(keys)
        for cond in c.CONDITIONS:
            for repeat in range(3):
                v = validity[phase, cond, repeat]
                require(
                    set(v) == set(keys),
                    "Missing or extra completed prediction for endpoint",
                )
                invalid = [key for key in keys if not v[key]]
                common.difference_update(invalid)
                cells[f"{cond}/{repeat}"] = {
                    "complete": len(keys),
                    "valid": len(keys) - len(invalid),
                    "invalid_keys": invalid,
                }
        memberships = defaultdict(set)
        for key in keys:
            memberships[roster[key]["text_cluster"]].add(roster[key]["source"])
        output[phase] = {
            **endpoint,
            "phase": phase,
            "role": "supporting DEV" if "dev" in phase else "main TEST supplement",
            "keys": keys,
            "gold": values,
            "sources": [roster[key]["source"] for key in keys],
            "clusters": [roster[key]["text_cluster"] for key in keys],
            "source_memberships_by_cluster": {
                k: sorted(v) for k, v in memberships.items()
            },
            "source_counts": dict(Counter((roster[key]["source"] for key in keys))),
            "n_clusters": len(memberships),
            "cells": cells,
            "common_valid_keys": [key for key in keys if key in common],
            "common_excluded_keys": [key for key in keys if key not in common],
            "observed_empty_gold_sets": (
                sum((v == [] for v in values)) if dimension == "legal" else None
            ),
            "fixed_macro_labels_after_exclusions": True,
            "paired_mask": "Intersection across all three conditions and all three outer repeats; null predictions stay null.",
        }
    return output


def prepare():
    c.validate()
    before = c.bindings(
        [
            c.PLAN,
            c.BASE / "ready.json",
            c.BASE / "schedule.json",
            c.BASE / "roster.jsonl",
            c.BASE / "endpoints.json",
            c.BASE / "prompts.jsonl",
            c.PROTOCOL,
            LIVE / "complete.json",
            c.OUTPUT / "storage_layout.json",
            *own_sources(),
        ]
    )
    save(
        BASE / "analysis_contract.json",
        {
            "status": "part1_reader_contract_before_result_matrix_consumption",
            "created_utc": c.a.utc(),
            "source_bindings": before,
            "matrix": {
                "inputs": 3677,
                "conditions": list(c.CONDITIONS),
                "outer_repeats": [0, 1, 2],
                "inner_votes": 3,
                "predictions": 33093,
                "batches": 27,
            },
            "receipt_policy": "Hash-check every consumed batch review, item receipt, result and job stamp; reuse existing raw audits.",
            "raw_sample": "All null results and all transport-retry items; first two in frozen schedule order per population/condition/n_valid; first two per population/condition for each repair, tie, duplicate, gate-normalization or legal-with-nonhate flag. Deduplicate sampled identities.",
            "votes": "Independently reconstruct every delivered decision, rationale origin and agreement diagnostic; preserve occurrence-counting duplicate-label semantics.",
            "missing": "Keep nulls; prepare one common-valid mask across all nine prediction slots per endpoint, plus own-valid masks.",
            "scope": "No F1, accuracy, false-positive/negative counts, outcome-driven case selection, bootstrap or ranking of conditions.",
            "protected": [
                "D: raw evidence",
                "frozen preparation and source code",
                "previous experiments",
                "LaTeX and presentation",
            ],
            "network": "Block sockets, HTTP, live transport and credential loading. Block scoring/contrast functions.",
        },
    )
    print("Reader contract frozen before consuming the result matrix.", flush=True)


def build():
    c.validate()
    contract = c.read(BASE / "analysis_contract.json")
    c.verify(contract["source_bindings"])
    run = BASE / "runs" / f"run_{time.time_ns()}"
    run.mkdir(parents=True)
    save(
        run / "start.json",
        {
            "created_utc": c.a.utc(),
            "contract_sha256": c.sha(BASE / "analysis_contract.json"),
        },
    )
    plan_sha = c.sha(c.PLAN)
    backend = isolated.adapter(c.ROOT / "config/taxonomy.yaml")
    taxonomy = yaml.safe_load(
        (c.ROOT / "config/taxonomy.yaml").read_text(encoding="utf8")
    )
    schedule = c.read(c.BASE / "schedule.json")
    cat = prompt_catalogue()
    roster_list = list(c.rows(c.BASE / "roster.jsonl"))
    roster = {row["key"]: row for row in roster_list}
    require(
        len(roster_list) == len(roster) == 3677
        and Counter((row["phase"] for row in roster_list)) == c.COUNTS,
        "Gold/input roster differs",
    )
    for row in roster_list:
        normalized = re.sub(
            "\\s+", " ", unicodedata.normalize("NFKC", row["text"]).casefold()
        ).strip()
        require(
            hashlib.sha256(normalized.encode()).hexdigest() == row["text_cluster"],
            "Normalized-text cluster changed",
        )
    verify_schedule(schedule, cat, roster)
    done = c.read(LIVE / "complete.json")
    require(
        done["status"] == "all_three_conditions_executed_and_audited"
        and done["mode"] == "live"
        and (done["plan_sha256"] == plan_sha),
        "Completed live matrix not established",
    )
    review_names = {
        c.ref(LIVE / "batches" / stage["id"] / "review.json")
        for stage in schedule["stages"]
    }
    require(
        set(done["review_hashes"]) == review_names, "Completion review roster differs"
    )
    expected_folders = {
        c.signature(identity(job, plan_sha))
        for stage in schedule["stages"]
        for job in stage["jobs"]
    }
    require(
        {p.name for p in (LIVE / "items").iterdir()} == expected_folders,
        "Unexpected/missing live item folder",
    )
    require(
        {p.name for p in (LIVE / "batches").iterdir()}
        == {stage["id"] for stage in schedule["stages"]},
        "Unexpected batch directory",
    )
    counts, by_cell, selections = (Counter(), defaultdict(Counter), Counter())
    validity, selected, nulls, duplicates = (defaultdict(dict), [], [], [])
    timelines, models, finishes, retry_classes = ([], Counter(), Counter(), Counter())
    consumed, rows_written = (0, 0)
    before = dict(contract["source_bindings"]) | done["review_hashes"]
    previous_end = None
    with (run / "predictions.jsonl").open("x", encoding="utf8", newline="\n") as cache:
        for stage in schedule["stages"]:
            review, attempts = check_batch(
                stage,
                done["review_hashes"][
                    c.ref(LIVE / "batches" / stage["id"] / "review.json")
                ],
                plan_sha,
            )
            before.update(review["availability_hashes"])
            first_start = min((timestamp(x["start"]["started_utc"]) for x in attempts))
            last_end = max((timestamp(x["finished_utc"]) for x in attempts))
            require(
                previous_end is None or first_start >= previous_end,
                "Execution batches overlap in recorded chronology",
            )
            previous_end = last_end
            timelines.append(
                {
                    "stage": stage["id"],
                    "population": stage["population"],
                    "attempts": attempts,
                }
            )
            stage_counts = Counter()
            for job in stage["jobs"]:
                ident = identity(job, plan_sha)
                entry = cat[job["key"], job["condition"]]
                value, receipt, evidence = read_item(
                    job, entry, review["items"][c.signature(ident)], plan_sha, backend
                )
                require(
                    first_start <= timestamp(receipt["created_utc"]) <= last_end,
                    "Item completion outside batch execution",
                )
                observation = check_value(value, taxonomy)
                cell = (job["population"], job["condition"], job["repeat"])
                require(job["key"] not in validity[cell], "Duplicate cached prediction")
                validity[cell][job["key"]] = value["result"] is not None
                cell_stats = by_cell["/".join(map(str, cell))]
                cell_stats["predictions"] += 1
                cell_stats["valid"] += value["result"] is not None
                cell_stats[f"valid_votes_{value['n_valid']}"] += 1
                cell_stats["invalid_votes"] += value["parse_failures"]
                cell_stats["repair_calls"] += value["repairs"]
                cell_stats["gate_normalized_votes"] += value["gate_normalised"]
                for flag in observation["flags"]:
                    cell_stats[flag + "_predictions"] += 1
                stage_counts["predictions"] += 1
                stage_counts["null_results"] += value["result"] is None
                stage_counts.update(value["accounting"])
                for event in value["events"]:
                    if event["action"] == "ok":
                        models[
                            str(event["model_response_metadata"]["reported_model"])
                        ] += 1
                        finishes[
                            str(event["model_response_metadata"]["finish_reason"])
                        ] += 1
                    else:
                        retry_classes[event["response_class"]] += 1
                output = {
                    **value,
                    "key": job["key"],
                    "condition": job["condition"],
                    "repeat": job["repeat"],
                    "phase": job["population"],
                    "source": roster[job["key"]]["source"],
                    "stage": stage["id"],
                    "evidence": evidence,
                    "reader_flags": observation,
                }
                cache.write(
                    json.dumps(
                        output, sort_keys=True, ensure_ascii=False, allow_nan=False
                    )
                    + "\n"
                )
                rows_written += 1
                reasons = []
                sample_groups = ["valid_votes_" + str(value["n_valid"])] + observation[
                    "flags"
                ]
                for name in sample_groups:
                    group = (job["population"], job["condition"], name)
                    if selections[group] < 2:
                        selections[group] += 1
                        reasons.append("first_two/" + name)
                if value["result"] is None:
                    reasons.append("all_null_outputs")
                    nulls.append(
                        {
                            "key": job["key"],
                            "phase": job["population"],
                            "condition": job["condition"],
                            "repeat": job["repeat"],
                            "evidence": evidence,
                        }
                    )
                if value["accounting"]["retry_attempts"]:
                    reasons.append("all_transport_retry_items")
                if observation["duplicate_labels"]:
                    duplicates.append(
                        {"identity": ident, "phase": job["population"], **observation}
                    )
                if reasons:
                    selected.append(
                        replay_sample(job, value, cat, plan_sha, backend, attempts)
                        | {
                            "phase": job["population"],
                            "selection_reasons": reasons,
                            "evidence": evidence,
                        }
                    )
                consumed += 3
            require(
                dict(stage_counts) == review["counts"],
                "Batch counts do not reproduce from saved predictions",
            )
            counts.update(stage_counts)
            print(
                f"{stage['id']}: reader verified {rows_written}/33093 results; {len(selected)} targeted raw checks so far.",
                flush=True,
            )
    require(
        rows_written == 33093 and dict(counts) == done["counts"],
        "Whole-run completion counts differ",
    )
    endpoints = endpoint_setup(roster, c.read(c.BASE / "endpoints.json"), validity)
    exclusions = c.read(c.BROAD / "replay_review_20261009_v1/sensitivity_plan.json")[
        "exclusion_roster"
    ]
    german_replay = [v["key"] for v in exclusions if v["key"] in roster]
    require(
        len(german_replay) == 7 and roster["de/gahd-f4c7eaf2"]["text"] == " ",
        "Declared sensitivity roster differs",
    )
    identical = [
        key
        for key in roster
        if cat[key, "mixed_2"]["messages"] == cat[key, "general_2"]["messages"]
    ]
    require(len(identical) == 194, "Previously identified equal-prompt group differs")
    setup = {
        "counts": c.COUNTS,
        "conditions": list(c.CONDITIONS),
        "same_prompt_keys": identical,
        "same_prompt_counts_by_population": dict(
            Counter((roster[key]["phase"] for key in identical))
        ),
        "reference_input_tokens_by_condition": {
            cond: {
                "min": min(
                    (
                        v["reference_input_tokens"]
                        for (k, cc), v in cat.items()
                        if cc == cond
                    )
                ),
                "max": max(
                    (
                        v["reference_input_tokens"]
                        for (k, cc), v in cat.items()
                        if cc == cond
                    )
                ),
            }
            for cond in c.CONDITIONS
        },
        "sensitivity_rosters": {
            "prior_ANN_replay": german_replay,
            "whitespace_only": ["de/gahd-f4c7eaf2"],
        },
        "main_mask_exclusions": {
            phase: value["common_excluded_keys"] for phase, value in endpoints.items()
        },
        "fine_output_limit": "Only hate and legal have the declared scored endpoints. No complete five-dimension German accuracy claim.",
        "agreement_fields": "Gate is vote share; multilabel is least-supported observed label occurrence share; severity is nonempty presence. They are not general agreement rates.",
    }
    comparisons = {
        "family_size": 4,
        "descriptive_confidence": 0.95,
        "family_confidence": 0.9875,
        "main": [
            {"phase": phase, "left": cond, "right": "mixed_2"}
            for phase in ("de_test", "de_legal_test")
            for cond in ("general_2", "general_2_legal_1")
        ],
        "supporting": [
            {"phase": phase, "left": "general_2_legal_1", "right": "general_2"}
            for phase in c.COUNTS
        ],
        "DEV_baseline_comparisons": [
            {"phase": phase, "left": cond, "right": "mixed_2"}
            for phase in ("de_dev_eval", "de_legal_dev_eval")
            for cond in ("general_2", "general_2_legal_1")
        ],
        "unit": "Input text/normalized-text cluster; fixed repeat scores are averaged, never treated as additional independent texts.",
        "bootstrap": {
            "draws": 10000,
            "paired": True,
            "source_stratified_whole_clusters": True,
            "fixed_full_roster_macro_labels": True,
        },
        "status": "Frozen comparison roles retained; no contrasts computed in Part 1.",
    }
    save(run / "endpoint_setup.json", endpoints)
    save(run / "prompt_setup.json", setup)
    save(run / "comparison_map.json", comparisons)
    save(run / "coverage.json", dict(by_cell))
    save(run / "null_outputs.json", nulls)
    save(run / "duplicate_labels.json", duplicates)
    save(run / "raw_spot_checks.json", selected)
    save(run / "execution_timeline.json", timelines)
    memberships = defaultdict(set)
    for row in roster_list:
        memberships[row["text_cluster"]].add(row["phase"])
    report = {
        "status": "complete_matrix_reader_and_comparison_setup_verified",
        "plan_sha256": plan_sha,
        "counts": dict(counts),
        "results_checked": rows_written,
        "item_artifacts_hash_checked": consumed,
        "batch_reviews_checked": 27,
        "execution_attempts": sum((len(t["attempts"]) for t in timelines)),
        "raw_items_spot_checked": len(selected),
        "raw_texts_reparsed_in_spot_checks": sum(
            (r["raw_texts_reparsed"] for r in selected)
        ),
        "independent_votes_reconstructed": rows_written,
        "model_reports": dict(models),
        "finish_reasons": dict(finishes),
        "retry_classes": dict(retry_classes),
        "null_predictions": len(nulls),
        "distinct_null_input_keys": len({x["key"] for x in nulls}),
        "common_valid_counts": {
            k: len(v["common_valid_keys"]) for k, v in endpoints.items()
        },
        "duplicate_label_predictions": len(duplicates),
        "duplicate_sensitive_predictions": sum(
            (bool(x["duplicate_sensitive_fields"]) for x in duplicates)
        ),
        "cross_population_normalized_clusters": {
            k: sorted(v) for k, v in memberships.items() if len(v) > 1
        },
        "actual_Qwen_calls": 0,
        "actual_model_list_GETs": 0,
        "performance_scores_computed": False,
        "scope": "Reused automatic full raw audits; verified consumed item artifacts and all votes; independently replayed targeted raw cases only.",
    }
    save(run / "reader_review.json", report)
    c.verify(before)
    c.validate()
    save(
        run / "source_and_preservation_check.json",
        {
            "status": "all_bound_sources_unchanged",
            "hashes": before,
            "D_writes": 0,
            "original_preparation_writes": 0,
            "training_calls": 0,
            "actual_Qwen_calls": 0,
            "latex_or_presentation_edits": 0,
            "scope": "Hashes cover completion, reviews, execution receipts, consumed result/job chains and preparation bindings; no full raw tree rescan.",
        },
    )
    outputs = c.bindings((p for p in run.iterdir() if p.is_file()))
    manifest = {
        "status": "reader_cache_ready_for_independent_check",
        "run": c.ref(run),
        "contract_sha256": c.sha(BASE / "analysis_contract.json"),
        "source_hashes": c.bindings(own_sources()),
        "outputs": outputs,
        "counts": dict(counts),
        "performance_scores_computed": False,
    }
    save(run / "manifest.json", manifest)
    print(json.dumps({"manifest": c.ref(run / "manifest.json"), **report}), flush=True)


def read_verified():
    done = c.read(BASE / "complete.json")
    require(
        done["status"] == "part1_complete_reader_ready",
        "Part 1 independent review incomplete",
    )
    c.verify(done["bindings"])
    manifest = c.read(c.ROOT / done["manifest"])
    c.verify(manifest["outputs"])
    return (manifest, c.ROOT / manifest["run"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "build", "check"))
    args = parser.parse_args()
    with offline():
        if args.mode == "prepare":
            prepare()
        elif args.mode == "build":
            build()
        else:
            manifest, path = read_verified()
            print(
                json.dumps(
                    {
                        "ready": True,
                        "predictions": manifest["counts"]["predictions"],
                        "run": c.ref(path),
                        "new_Qwen_calls": 0,
                    }
                )
            )

import argparse
import base64
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import socket
import time
import traceback
from unittest.mock import patch
from analysis.feedback.source import updates as b

a = b.a
ROOT = b.ROOT
u = b.preparation.review.u
old = b.preparation.review
BASE = b.STUDY / "updates_review_v1"
EVIDENCE = (
    ROOT
    / "Thesis_Log/astra_result/evidence/sq3_feedback_updates_review_20260930_v1.json"
)


def checked_merge(target, source):
    for name, sha in source.items():
        a.require(
            name not in target or target[name] == sha,
            "Conflicting file binding: " + name,
        )
        target[name] = sha


def error_family(message):
    if message.startswith("invalid JSON"):
        return "invalid_JSON"
    for field in ("hate_type", "target_group", "legal", "hate", "severity"):
        if "('" + field + "'," in message:
            return "invalid_" + field
    return "other"


def verify(bindings, inspect_responses=False):

    items = list(bindings.items())
    remote_ids = Counter()
    meta = Counter()
    usage = Counter()

    def one(pair):
        name, expected = pair
        raw = a.storage_path(ROOT / name).read_bytes()
        a.require(
            hashlib.sha256(raw).hexdigest() == expected, "Bound file changed: " + name
        )
        if inspect_responses and name.endswith("/outcome.json"):
            outcome = json.loads(raw)
            if outcome.get("body_base64") and outcome.get("status") == 200:
                data = json.loads(
                    base64.b64decode(outcome["body_base64"], validate=True)
                )
                a.require(
                    data.get("id") and data.get("model") == a.MODEL,
                    "Remote identity/model missing",
                )
                a.require(
                    data["choices"][0]["finish_reason"] == "stop", "Non-stop response"
                )
                tokens = data["usage"]
                a.require(
                    all(
                        type(tokens[k]) is int and tokens[k] >= 0
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                    )
                    and tokens["prompt_tokens"] + tokens["completion_tokens"]
                    == tokens["total_tokens"],
                    "Invalid usage",
                )
                return data["id"], {
                    k: tokens[k]
                    for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                }
        return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        for begin in range(0, len(items), 1000):
            for result in executor.map(one, items[begin : begin + 1000]):
                if result:
                    remote_ids[result[0]] += 1
                    usage.update(result[1])
                    meta["successful_raw_responses"] += 1
            end = min(begin + 1000, len(items))
            if inspect_responses and (end % 25000 == 0 or end == len(items)):
                print(
                    f"Verified {end}/{len(items)} frozen raw-output files.", flush=True
                )
    a.require(
        not remote_ids or max(remote_ids.values()) == 1,
        "A remote response ID was reused across settings/phases",
    )
    return {
        "files": len(items),
        "unique_remote_ids": len(remote_ids),
        "duplicates": 0,
        "metadata": dict(meta),
        "usage": dict(usage),
    }


def run():
    from src.hsrag.parse import parse_response, extract_json

    a.require(
        not (BASE / "complete.json").exists() and not EVIDENCE.exists(),
        "Preserve completed review",
    )
    attempt = BASE / "attempts" / str(time.time_ns())
    a.save_new(
        attempt / "start.json",
        {
            "created_utc": a.utc(),
            "source_sha256": a.digest(__file__),
            "actual_API_calls": 0,
        },
    )
    try:
        b.validate()
        plan_sha = a.digest(b.PLAN)
        schedule = a.read(b.BASE / "schedule.json")
        inputs = {
            b.rel(__file__): a.digest(__file__),
            b.rel(b.PLAN): plan_sha,
            b.rel(b.SCIENCE): a.digest(b.SCIENCE),
        }
        outputs = {}
        audits = []
        summaries = []
        aggregate = defaultdict(Counter)
        for stage in schedule["stages"]:
            stage_id = stage["id"]
            rp = b.BASE / "stage_reviews" / f"{stage_id}.json"
            receipt = a.read(rp)
            ep = ROOT / receipt["evidence"]
            evidence = a.read(ep)
            cp = b.BASE / "live/stages" / stage_id / "complete.json"
            complete = a.read(cp)
            a.require(
                receipt["status"] == "audited"
                and receipt["plan_sha256"] == plan_sha
                and a.digest(ep) == receipt["evidence_sha256"]
                and a.digest(cp) == receipt["complete_sha256"],
                "Audit/complete binding changed",
            )
            a.require(
                evidence["status"] == "completed_updates_stage_verified"
                and evidence["semantic_replay_items"] == stage["items"]
                and evidence["independent_field_vote_checks"] == stage["items"]
                and evidence["completed_stage_restart_new_POSTs"] == 0
                and evidence["actual_audit_API_calls"] == 0
                and evidence["actual_gold_scoring"] is False,
                "Audit scope differs",
            )
            a.require(
                not complete["summary"]["errors"]
                and not complete["summary"]["stopped"],
                "Incomplete/failed execution",
            )
            a.require(
                not evidence["duplicate_response_metadata_ids"],
                "Duplicate stage response ID",
            )
            checked_merge(inputs, evidence["verified_input_hashes"])
            for p in (rp, ep, cp):
                inputs[b.rel(p)] = a.digest(p)
            checked_merge(outputs, evidence["verified_output_hashes"])
            fields = (
                "stage",
                "start_utc",
                "completed_utc",
                "wall_seconds_including_final_save",
                "counts",
                "accounting",
                "http_statuses",
                "reported_models",
                "finish_reasons",
                "observed_usage_sums",
                "transport_failure_classes",
                "null_prediction_details",
                "retry_events",
                "latency",
                "first_response_validation_errors",
                "terminal_invalid_vote_errors",
                "validation_errors_all_response_turns",
            )
            summaries.append(
                {
                    **{k: evidence[k] for k in fields},
                    "evidence": b.rel(ep),
                    "evidence_sha256": a.digest(ep),
                    "input_hashes": len(evidence["verified_input_hashes"]),
                    "output_hashes": len(evidence["verified_output_hashes"]),
                }
            )
            for field in (
                "counts",
                "accounting",
                "observed_usage_sums",
                "transport_failure_classes",
                "http_statuses",
                "validation_errors_all_response_turns",
                "first_response_validation_errors",
                "terminal_invalid_vote_errors",
            ):
                aggregate[field].update(evidence[field])
            audits.append((stage, evidence, complete))
        baseline = a.read(old.EVIDENCE)
        checked_merge(inputs, baseline["verified_inputs"])
        checked_merge(inputs, baseline["outputs"])
        for p in (
            old.EVIDENCE,
            old.BASE / "complete.json",
            old.BASE / "selection.json",
            b.preparation.BASE / "complete.json",
            b.preparation.EVIDENCE,
            b.BASE / "offline_check_complete.json",
        ):
            inputs[b.rel(p)] = a.digest(p)
        baseline_sha = a.read(old.BASE / "complete.json")["evidence_sha256"]
        a.require(
            a.digest(old.EVIDENCE) == baseline_sha == b.BASELINE_REVIEW_SHA,
            "Baseline review changed",
        )
        baseline_raw = baseline["verified_live_output_hashes"]
        update_hashes = len(outputs)
        checked_merge(outputs, baseline_raw)
        plan = {
            "created_utc": a.utc(),
            "scope": "Independent completed-study integrity/raw-vote review, then exports; no outcome selection or scoring.",
            "inputs": inputs,
            "audit_evidence": [
                {"path": x["evidence"], "sha256": x["evidence_sha256"]}
                for x in summaries
            ],
            "raw_output_files_union": len(outputs),
            "update_raw_files_union": update_hashes,
            "raw_hash_source": "Stage audit evidence plus separately frozen baseline-review evidence",
            "script_sha256": a.digest(__file__),
            "actual_API_calls": 0,
        }
        a.save_new(attempt / "plan.json", plan)
        verify(inputs)
        checked = verify(outputs, True)
        expected_responses = (
            aggregate["accounting"]["http_success_with_text"]
            + baseline["operational"]["accounting"]["http_success_with_text"]
        )
        a.require(
            checked["unique_remote_ids"] == expected_responses,
            "Global response count differs",
        )
        expected_usage = Counter(aggregate["observed_usage_sums"])
        expected_usage.update(baseline["operational"]["observed_usage_sums"])
        a.require(
            checked["usage"] == dict(expected_usage),
            "Independent global raw-envelope usage differs",
        )
        science = a.read(b.SCIENCE)
        ids = set(science["rosters"]["evaluation_phase"])
        cat = b.frozen_catalog(ROOT / a.read(b.PLAN)["prompts"])
        inventory = {
            (x["key"], x["condition"]): x
            for x in b.core.rows(b.preparation.BASE / "prompt_inventory.jsonl")
        }
        expected_dirs = {
            a.signature(b.identity(j, plan_sha))
            for st in schedule["stages"]
            for j in st["jobs"]
        }
        a.require(
            {p.name for p in (b.BASE / "live/items").iterdir() if p.is_dir()}
            == expected_dirs,
            "Unexpected/missing live job",
        )
        assembled = defaultdict(dict)
        counts = Counter()
        accounting = Counter()
        nulls = []
        invalid = Counter()
        terminal = Counter()
        failed_details = []
        jobs_seen = set()
        for stage, evidence, complete in audits:
            local = Counter()
            for number, job in enumerate(stage["jobs"], 1):
                identity = b.identity(job, plan_sha)
                signature = a.signature(identity)
                a.require(signature not in jobs_seen, "Repeated job identity")
                jobs_seen.add(signature)
                folder = b.item_dir(b.BASE / "live", job, plan_sha)
                value = a.read(folder / "result.json")
                a.require(
                    value["identity"] == identity
                    and value["execution_mode"] == "live"
                    and value["active_model"] == a.MODEL,
                    "Result scope/identity differs",
                )
                a.require(
                    value["messages_sha256"]
                    == inventory[job["key"], job["condition"]]["messages_sha256"]
                    == cat[job["key"], job["condition"]]["messages_sha256"],
                    "Live prompt differs from independently prepared prompt",
                )
                a.require(
                    a.digest(folder / "complete.json")
                    == complete["item_receipt_hashes"][b.rel(folder / "complete.json")],
                    "Completion receipt changed",
                )
                votes = []
                for vote in range(3):
                    saved = a.read(folder / "votes" / f"{vote}.json")
                    parsed = []
                    for turn, raw in enumerate(saved["raw_outputs"]):
                        obj, _, err = parse_response(raw)
                        parsed.append(obj)
                        counts["raw_turns"] += 1
                        if obj is None:
                            try:
                                loose = json.loads(extract_json(raw))
                                bad = set(loose.get("hate_type", [])) - set(u.LABELS)
                            except (ValueError, TypeError, AttributeError):
                                bad = set()
                            invalid.update(bad)
                            if turn == len(saved["raw_outputs"]) - 1:
                                terminal.update(bad)
                                failed_details.append(
                                    {
                                        "job": job,
                                        "vote": vote,
                                        "error": err,
                                        "invalid_type_labels": sorted(bad),
                                        "saved_vote": b.rel(
                                            folder / "votes" / f"{vote}.json"
                                        ),
                                    }
                                )
                    final = parsed[-1].model_dump() if parsed[-1] is not None else None
                    a.require(
                        final == saved["result"] == value["run_predictions"][vote]
                        and all(p is None for p in parsed[:-1]),
                        "Raw votes/repair stopping differ",
                    )
                    if final is None:
                        a.require(
                            len(parsed) == 3,
                            "Terminal failure without exhausted bounded repair",
                        )
                    votes.append(final)
                voted, n, tie = u.vote(votes)
                a.require(
                    n == value["n_valid"] and tie == value["uncertain"],
                    "Independent valid-vote/tie differs",
                )
                a.require(
                    voted
                    == (
                        {k: value["result"][k] for k in u.FIELDS}
                        if value["result"] is not None
                        else None
                    ),
                    "Independent aggregate differs",
                )
                u.check_prediction(value["result"], u.LABELS)
                iid = job["key"].split("/", 1)[1]
                setting = (job["condition"], job["repeat"])
                a.require(
                    iid in ids and iid not in assembled[setting],
                    "Unexpected/duplicate roster ID",
                )
                row = {
                    "id": iid,
                    "key": job["key"],
                    "condition": job["condition"],
                    "repeat": job["repeat"],
                    "arm": job["arm"],
                    "seed": job["seed"],
                    "budget_level": job["budget_level"],
                    "pool_size": job["pool_size"],
                    "subset": job["subset"],
                    "result": value["result"],
                    "n_valid": n,
                    "source": b.rel(folder / "result.json"),
                    "source_sha256": a.digest(folder / "result.json"),
                    "messages_sha256": value["messages_sha256"],
                }
                assembled[setting][iid] = row
                counts["predictions"] += 1
                counts["valid_votes_" + str(n)] += 1
                local["items"] += 1
                counts["parse_failures"] += value["parse_failures"]
                counts["repairs"] += value["repairs"]
                accounting.update(value["accounting"])
                if voted is None:
                    nulls.append(job)
                    local["null_items"] += 1
                if number % 2000 == 0 or number == stage["items"]:
                    print(
                        f"Reconstructed {stage['id']}: {number}/{stage['items']} predictions.",
                        flush=True,
                    )
            a.require(
                local["items"] == evidence["counts"]["items"]
                and local["null_items"] == evidence["counts"]["null_items"],
                "Stage reconstruction total differs",
            )
        a.require(
            len(assembled) == 72 and all(set(run) == ids for run in assembled.values()),
            "Incomplete condition/repeat matrix",
        )
        a.require(
            counts["predictions"] == 33552
            and counts["raw_turns"] == aggregate["accounting"]["http_success_with_text"]
            and counts["parse_failures"] == aggregate["counts"]["parse_failures"]
            and counts["repairs"] == aggregate["counts"]["repairs"]
            and accounting == aggregate["accounting"],
            "Reconstruction/accounting differs",
        )
        a.require(
            len(nulls) == aggregate["counts"]["null_items"], "Whole-null total differs"
        )
        exports = {}
        for (condition, rep), run in sorted(assembled.items()):
            path = BASE / "assembled" / f"{condition}_rep{rep}.jsonl"
            old.write_rows(path, [run[i] for i in sorted(run)])
            exports[b.rel(path)] = {"rows": len(run), "sha256": a.digest(path)}
        errors = {name: dict(Counter({})) for name in ()}
        for key in (
            "validation_errors_all_response_turns",
            "first_response_validation_errors",
            "terminal_invalid_vote_errors",
        ):
            grouped = Counter()
            for message, n in aggregate[key].items():
                grouped[error_family(message)] += n
            errors[key] = dict(grouped)
        source_rows = {
            x["id"]: x
            for x in b.core.rows(
                b.preparation.review.b.preparation.OUT / "roster.jsonl"
            )
        }
        null_texts = [
            {
                "id": j["key"].split("/", 1)[1],
                "job": j,
                "source_gold": source_rows[j["key"].split("/", 1)[1]]["gold"],
                "text": source_rows[j["key"].split("/", 1)[1]]["text"],
            }
            for j in nulls
        ]
        stages_start = min(datetime.fromisoformat(x["start_utc"]) for x in summaries)
        last_audit = max(
            datetime.fromisoformat(a.read(ROOT / x["evidence"])["created_utc"])
            for x in summaries
        )
        verify(inputs)
        report = {
            "created_utc": a.utc(),
            "status": "all_feedback_updates_independently_reviewed",
            "plan_sha256": a.digest(attempt / "plan.json"),
            "attempt": b.rel(attempt),
            "inputs": inputs,
            "stages": summaries,
            "combined": {k: dict(v) for k, v in aggregate.items()},
            "independent": checked,
            "reconstruction": dict(counts),
            "assembled": exports,
            "update_raw_files_union": update_hashes,
            "baseline_raw_files": len(baseline_raw),
            "whole_study_unique_raw_files": len(outputs),
            "shared_remote_response_ids": 0,
            "nulls": nulls,
            "null_source_diagnostics": null_texts,
            "failed_vote_details": failed_details,
            "invalid_type_labels_all_turns": dict(invalid),
            "terminal_invalid_type_labels": dict(terminal),
            "error_families": errors,
            "update_valid_coverage": (33552 - len(nulls)) / 33552,
            "whole_study_counts": dict(
                Counter(aggregate["counts"])
                + Counter(baseline["operational"]["counts"])
            ),
            "whole_study_accounting": dict(
                Counter(accounting) + Counter(baseline["operational"]["accounting"])
            ),
            "whole_study_observed_usage": dict(expected_usage),
            "generation_wall_seconds_sum": sum(
                x["wall_seconds_including_final_save"] for x in summaries
            ),
            "combined_elapsed_seconds_including_audits": (
                last_audit - stages_start
            ).total_seconds(),
            "elapsed_scope": "From first update batch start to final automatic-audit publication. Includes gaps/audits; baseline reported separately.",
            "actual_API_calls": 0,
            "next": "Score the fixed full matrix under the existing scientific plan; do not rerun failed answers.",
            "limits": [
                "Remote work on connection/SSL failures is unknown; observed tokens exclude any such work.",
                "Invalid votes are preserved and valid surviving votes follow the original rules.",
                "No model result regenerated and no frozen pool/prompt/source changed.",
            ],
        }
        a.save_new(EVIDENCE, report)
        a.save_new(
            BASE / "complete.json",
            {
                "status": report["status"],
                "evidence": b.rel(EVIDENCE),
                "evidence_sha256": a.digest(EVIDENCE),
                "plan": b.rel(attempt / "plan.json"),
                "plan_sha256": a.digest(attempt / "plan.json"),
                "outputs": {p: v["sha256"] for p, v in exports.items()},
                "source_sha256": a.digest(__file__),
            },
        )
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "reconstruction": report["reconstruction"],
                    "accounting": dict(accounting),
                    "nulls": nulls,
                    "terminal_invalid_type_labels": dict(terminal),
                    "error_families": errors,
                    "generation_hours": report["generation_wall_seconds_sum"] / 3600,
                    "elapsed_hours_with_audits": report[
                        "combined_elapsed_seconds_including_audits"
                    ]
                    / 3600,
                    "evidence_sha256": a.digest(EVIDENCE),
                },
                indent=2,
            ),
            flush=True,
        )
    except Exception:
        a.save_new(
            attempt / "failure.json",
            {
                "created_utc": a.utc(),
                "traceback": traceback.format_exc(),
                "actual_API_calls": 0,
            },
        )
        raise


if __name__ == "__main__":
    with patch.object(socket.socket, "connect", u.forbid), patch.object(
        socket, "create_connection", u.forbid
    ), patch.object(b.core, "live_client", u.forbid):
        run()

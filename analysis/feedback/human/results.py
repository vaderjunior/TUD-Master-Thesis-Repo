from src.hsrag.paths import PROJECT_ROOT
import base64
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

ROOT = PROJECT_ROOT
BASE = ROOT / "experiments/astra/sq3_human_feedback_20261015_v1"
LIVE = BASE / "execution_v2/live"
OUT = ROOT / "experiments/astra/sq3_human_feedback_results_20261020_v1"
PRECHECK = ROOT / "experiments/astra/sq3_human_feedback_launch_check_20261020_v1"
LOG = ROOT / "Thesis_Log/astra_result"
ARMS = ("reference5", "dataset_feedback5", "human_feedback5")
LABELS = (
    "explicit",
    "incitement",
    "inferiority",
    "irony",
    "stereotyping",
    "threatening",
    "grievance",
)


def need(ok, message):
    if not ok:
        raise AssertionError(message)


def logical(p):
    return Path(str(p).removeprefix("\\\\?\\")).resolve()


def physical(p):
    return Path("\\\\?\\" + str(logical(p))) if os.name == "nt" else logical(p)


def rel(p):
    return logical(p).relative_to(ROOT).as_posix()


def read(p):
    return json.loads(physical(p).read_text(encoding="utf-8-sig"))


def rows(p):
    return [
        json.loads(s)
        for s in physical(p).read_text(encoding="utf-8-sig").splitlines()
        if s.strip()
    ]


def sha(p):
    h = hashlib.sha256()
    with physical(p).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def new(p, value):
    p = logical(p)
    need(p.is_relative_to(ROOT), "Write outside research workspace")
    p.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(value, bytes):
        value = (
            value
            if isinstance(value, str)
            else json.dumps(
                value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
            )
            + "\n"
        ).encode("utf-8")
    with physical(p).open("xb") as f:
        f.write(value)


def verify(mapping):
    for p, expected in mapping.items():
        need(sha(ROOT / p) == expected, "Changed evidence: " + p)


def inventory(root):
    return {
        rel(Path(folder) / name): sha(Path(folder) / name)
        for folder, _, names in os.walk(physical(root))
        for name in names
    }


def frozen_target(row, reference):

    if reference == "dataset":
        return row["gold"]["hate"], row["gold"]["hate_type"]
    answer = row["judgments"][reference + "_project"]
    gate = answer["binary_hate"]
    return gate, (
        None
        if gate is None
        else (
            []
            if not gate
            else answer["types"] if answer["type_state"] == "assigned" else None
        )
    )


def main():
    from analysis.feedback.human import feedback_api as api
    from analysis.feedback.human import feedback_common as c
    from analysis.feedback.human import feedback_score as scorer

    need(not OUT.exists(), "Preserve previous review attempt")
    with c.offline(), patch.object(api.template.core, "live_client", c.forbidden):
        prior = read(PRECHECK / "before.json")
        verify(prior["frozen_bindings"])
        verify(prior["receipt_bindings"])
        b = api.queue()
        b.validate(ready=True)
        done = read(LIVE / "complete.json")
        need(
            done["status"] == "all_studies_executed_and_audited"
            and done["mode"] == "live",
            "Completed automatic audit required",
        )
        original = inventory(LIVE)
        new(
            OUT / "before.json",
            {
                "created_utc": utc(),
                "live_files": original,
                "frozen_bindings": prior["frozen_bindings"],
                "receipt_bindings": prior["receipt_bindings"],
                "analysis_source_sha256": sha(__file__),
                "old_roadmap": {
                    "path": rel(
                        LOG / "astra_remaining_work_before_writing_20261020_v59.md"
                    ),
                    "sha256": sha(
                        LOG / "astra_remaining_work_before_writing_20261020_v59.md"
                    ),
                },
            },
        )
        print(
            f"Preserved hashes of {len(original)} live files; replaying all 216 predictions offline.",
            flush=True,
        )
        predictions, dependencies, completion = api.extract(b, LIVE, "live")
        roster = rows(BASE / "roster.jsonl")
        prompts = {(p["key"], p["condition"]): p for p in rows(BASE / "prompts.jsonl")}
        choices = {p["key"]: p for p in rows(BASE / "choices.jsonl")}
        report = scorer.analyze(roster, predictions)
        report["execution_accounting"] = {
            k: completion.get(k)
            for k in (
                "counts",
                "accounting",
                "observed_usage",
                "stages",
                "review_hashes",
            )
        }
        report["input_hashes"] = dependencies
        c.save(BASE / "results/descriptive_analysis.json", report)
        new(OUT / "frozen_descriptive_analysis.json", report)
        schedule = read(BASE / "execution_v2/schedule.json")
        by_key = {r["key"]: r for r in roster}
        items, calls, ids = [], [], []
        resource_groups = defaultdict(Counter)
        errors = Counter()
        audits = []
        for stage in schedule["stages"]:
            stage_root = LIVE / "human_feedback/stages" / stage["id"]
            review = read(stage_root / "review.json")
            audit = read(stage_root / review["report"])
            audits.append(audit)
            for job in stage["jobs"]:
                folder = logical(b.item_dir(LIVE / "human_feedback", job, sha(b.PLAN)))
                value = read(folder / "result.json")
                r = by_key[job["key"]]
                prompt = prompts[job["key"], job["condition"]]
                item = {
                    "alias": r["packet_id"],
                    "key": r["key"],
                    "condition": job["condition"],
                    "repeat": job["repeat"],
                    "source": r["is_correction_source"],
                    "transfer": r["transfer_eligible"],
                    "selected_correction": r["selected_correction"],
                    "stratum": r["stratum"],
                    "cluster": r["cluster"],
                    "result_path": rel(folder / "result.json"),
                    "result_sha256": sha(folder / "result.json"),
                    "prediction": value["result"],
                    "n_valid": value["n_valid"],
                    "votes": value["run_predictions"],
                    "raw_runs": value["raw_runs"],
                    "accounting": value["accounting"],
                    "repairs": value["repairs"],
                    "prompt_tokens_local": prompt["reference_input_tokens"],
                    "messages_sha256": value["messages_sha256"],
                }
                items.append(item)
                group = resource_groups[job["condition"]]
                group["predictions"] += 1
                group.update(value["accounting"])
                group["local_prompt_tokens_sum_once_per_prediction"] += prompt[
                    "reference_input_tokens"
                ]
                group["null_predictions"] += value["result"] is None
                for event in value["events"]:
                    location = (
                        folder
                        / "calls"
                        / f"vote_{event['vote']}_turn_{event['turn']}"
                        / f"attempt_{event['attempt']}"
                    )
                    intent, outcome = read(location / "intent.json"), read(
                        location / "outcome.json"
                    )
                    call = {
                        "alias": r["packet_id"],
                        "condition": job["condition"],
                        "repeat": job["repeat"],
                        "vote": event["vote"],
                        "turn": event["turn"],
                        "attempt": event["attempt"],
                        "kind": event["kind"],
                        "action": event["action"],
                        "started_utc": intent["started_utc"],
                        "finished_utc": outcome["finished_utc"],
                        "elapsed_s": outcome["elapsed_s"],
                        "backoff_elapsed_s": intent["backoff_elapsed_s"],
                        "outcome_path": rel(location / "outcome.json"),
                        "outcome_sha256": sha(location / "outcome.json"),
                        "status": outcome.get("status"),
                        "error_category": outcome.get("category"),
                    }
                    group["POST_seconds_sum"] += outcome["elapsed_s"]
                    group["retry_wait_seconds_sum"] += intent["backoff_elapsed_s"]
                    if outcome["kind"] == "transport_error":
                        errors[outcome["category"]] += 1
                    if event["action"] == "ok":
                        raw = json.loads(
                            base64.b64decode(outcome["body_base64"], validate=True)
                        )
                        call["usage"] = raw["usage"]
                        call["remote_id"] = raw["id"]
                        call["reported_model"] = raw["model"]
                        call["finish_reason"] = raw["choices"][0]["finish_reason"]
                        ids.append(raw["id"])
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens"):
                            group[k] += raw["usage"][k]
                            group[event["kind"] + "_" + k] += raw["usage"][k]
                    calls.append(call)
        totals = Counter()
        for stats in resource_groups.values():
            totals.update(stats)
        for k, v in completion["accounting"].items():
            need(totals[k] == v, "Raw accounting differs: " + k)
        for k, v in completion["observed_usage"].items():
            need(totals[k] == v, "Raw usage differs: " + k)
        need(
            len(ids) == len(set(ids)) == completion["distinct_remote_ids"],
            "Duplicate remote response",
        )
        need(
            len(items) == 216 and len(calls) == totals["post_attempts"],
            "Incomplete review matrix",
        )
        stage = schedule["stages"][0]
        stage_root = LIVE / "human_feedback/stages" / stage["id"]
        stage_complete = read(stage_root / "complete.json")
        starts = [
            read(p)["started_utc"] for p in stage_root.glob("attempt_*/start.json")
        ]
        start = min(starts)
        end = stage_complete["completed_utc"]
        wall = (
            datetime.fromisoformat(end) - datetime.fromisoformat(start)
        ).total_seconds()
        need(
            abs(wall - audits[0]["wall_seconds_including_restarts"]) < 1e-6,
            "Run wall clock differs",
        )
        model_gets = []
        for p in stage_root.glob("attempt_*/model_check_response.json"):
            v = read(p)
            model_gets.append(
                {"path": rel(p), "elapsed_s": v["elapsed_s"], "sha256": sha(p)}
            )
        resources = {
            "totals": dict(totals),
            "by_condition": dict(resource_groups),
            "transport_error_categories": dict(errors),
            "distinct_successful_response_ids": len(set(ids)),
            "started_utc": start,
            "execution_completed_utc": end,
            "execution_wall_seconds": wall,
            "automatic_audit_created_utc": audits[0]["created_utc"],
            "model_list_GETs": model_gets,
            "limits": [
                "Measured POST times overlap across eight workers; not wall time or GPU time.",
                "Usage from successful responses only; remote work behind transport errors unknown.",
                "Separate inherited 100 participant-minutes/96 judgments; no new annotation.",
                "No price, energy or whole-thesis resource claim.",
            ],
        }
        grouped = {
            (r["key"], arm): sorted(
                [p for p in items if p["key"] == r["key"] and p["condition"] == arm],
                key=lambda x: x["repeat"],
            )
            for r in roster
            for arm in ARMS
        }
        cases = []
        for r in roster:
            case = dict(r)
            case["choice"] = choices[r["key"]]
            case["targets"] = {
                ref: {
                    "gate": frozen_target(r, ref)[0],
                    "types": frozen_target(r, ref)[1],
                }
                for ref in ("A", "B", "dataset")
            }
            case["predictions"] = {arm: grouped[r["key"], arm] for arm in ARMS}
            case["contexts"] = {arm: prompts[r["key"], arm]["context"] for arm in ARMS}
            case["feedback_messages_identical"] = (
                prompts[r["key"], ARMS[1]]["messages"]
                == prompts[r["key"], ARMS[2]]["messages"]
            )
            cases.append(case)
        new(
            OUT / "predictions.json",
            sorted(
                items,
                key=lambda x: (x["alias"], ARMS.index(x["condition"]), x["repeat"]),
            ),
        )
        new(OUT / "all_cases.json", cases)
        new(OUT / "all_calls.json", calls)
        new(OUT / "resources.json", resources)
        verify(original)
        need(inventory(LIVE) == original, "Live file inventory changed")
        verify(prior["frozen_bindings"])
        verify(prior["receipt_bindings"])
        new(
            OUT / "raw_review.json",
            {
                "status": "all_216_raw_predictions_replayed_and_independently_aggregated",
                "created_utc": utc(),
                "preserved_live_files": len(original),
                "raw_predictions_checked": len(items),
                "valid_votes": sum(x["n_valid"] for x in items),
                "request_attempts": len(calls),
                "all_unique_response_ids": len(ids),
                "nulls": len(report["null_predictions"]),
                "frozen_scoring_used": rel(BASE / "results/descriptive_analysis.json"),
                "originals_unchanged": True,
                "new_model_calls": 0,
                "analysis_source_sha256": sha(__file__),
                "artifacts": {
                    rel(p): sha(p)
                    for p in [
                        OUT / "before.json",
                        OUT / "frozen_descriptive_analysis.json",
                        OUT / "predictions.json",
                        OUT / "all_cases.json",
                        OUT / "all_calls.json",
                        OUT / "resources.json",
                        BASE / "results/descriptive_analysis.json",
                    ]
                },
            },
        )
        print(
            json.dumps(
                {
                    "status": "raw_replay_and_frozen_scoring_complete",
                    "cohorts": report["cohort_sizes"],
                    "resources": resources,
                    "primary_complete": {
                        r: report["transfer"]["other_all"][r]["complete"]["arms"]
                        for r in ("A", "B", "dataset")
                    },
                    "source_uptake": report["source_uptake"],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()

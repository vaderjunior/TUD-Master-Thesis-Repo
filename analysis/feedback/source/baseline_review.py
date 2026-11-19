import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import socket
import time
from unittest.mock import patch
from analysis.feedback.source import run as b
from analysis.feedback.source import design as d

a = b.a
u = d.u
ROOT = b.ROOT
STUDY = b.STUDY
BASE = STUDY / "baseline_review_v1"
EVIDENCE = (
    ROOT
    / "Thesis_Log/astra_result/evidence/sq3_feedback_baseline_results_20260929_v1.json"
)
AUDIT_SHA = "feb96a3b5ff2cdaf296f67b796aa90360a32c76df16988d8b90d4f862d7972e7"


def verify_hashes(bindings):
    items = list(bindings.items())

    def one(item):
        name, expected = item
        return name, expected, a.digest(ROOT / name)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(items), 1024):
            for name, expected, actual in pool.map(one, items[start : start + 1024]):
                a.require(expected == actual, "Bound artifact changed: " + name)


def write_rows(path, rows):
    path = a.storage_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            )


def select(eligible, pool, gold, runs):
    selected = d.choose_updates(eligible, gold, runs)
    sentinel = sorted(
        i
        for i in pool
        if all(
            r[i]["result"] is not None and gold[i] <= set(r[i]["result"]["hate_type"])
            for r in runs
        )
    )
    selected["sentinel_ids"] = sentinel
    selected["selection_diagnostics"] = [
        {
            "id": i,
            "valid_repeats": sum(r[i]["result"] is not None for r in runs),
            "contains_repeats": sum(
                r[i]["result"] is not None
                and gold[i] <= set(r[i]["result"]["hate_type"])
                for r in runs
            ),
            "role": "candidate" if i in eligible else "pool",
        }
        for i in sorted(set(eligible) | set(pool))
    ]

    expected = {"error": [], "correct": [], "invalid": []}
    for i in sorted(eligible):
        values = [
            (
                None
                if r[i]["result"] is None
                else int(all(label in r[i]["result"]["hate_type"] for label in gold[i]))
            )
            for r in runs
        ]
        group = (
            "invalid"
            if None in values
            else "correct" if sorted(values)[1] == 1 else "error"
        )
        expected[group].append(i)
    a.require(
        selected["groups"] == expected, "Independent candidate classification differs"
    )
    for row in selected["sets"]:
        ranking = sorted(
            expected[row["arm"]],
            key=lambda i: (
                hashlib.sha256(
                    "/".join([str(row["seed"]), row["arm"], i]).encode("utf8")
                ).hexdigest(),
                i,
            ),
        )
        a.require(
            row["ids"] == ranking[: row["size"]], "Independent selected pool differs"
        )
    a.require(not set(sentinel) & set(eligible), "Sentinel/candidate overlap")
    return selected


def checks():
    ids = [f"e{i:02d}" for i in range(40)] + [f"c{i:02d}" for i in range(40)] + ["bad"]
    pool = ["stable", "unstable", "invalid"]
    gold = {i: {"explicit"} for i in ids + pool}
    runs = [
        {
            i: {
                "result": (
                    None
                    if i in ("bad", "invalid")
                    else {
                        "hate_type": [] if i.startswith("e") else ["explicit", "irony"]
                    }
                )
            }
            for i in ids + pool
        }
        for _ in range(3)
    ]
    runs[2]["unstable"]["result"]["hate_type"] = []
    result = select(ids, pool, gold, runs)
    a.require(
        result["k"] == 40
        and result["levels"] == [10, 20, 30, 40]
        and result["sentinel_ids"] == ["stable"],
        "Selection/sentinel fixture",
    )
    a.require(
        result["groups"]["invalid"] == ["bad"] and len(result["sets"]) == 24,
        "Invalid candidate fixture",
    )
    a.require(
        result
        == select(
            list(reversed(ids)), list(reversed(pool)), gold, list(reversed(runs))
        ),
        "Order invariance fixture",
    )
    pred = {
        "a": {"result": {"hate": False, "hate_type": []}},
        "b": {"result": {"hate": True, "hate_type": ["explicit", "irony"]}},
        "c": {"result": None},
    }
    gg = {"a": {"explicit"}, "b": {"explicit"}, "c": {"irony"}}
    metric = u.metric(list(gg), gg, pred)
    a.require(
        metric["valid"] == 2
        and metric["null_ids"] == ["c"]
        and metric["gate_false"] == 1,
        "Failure/gate accounting fixture",
    )
    a.require(
        abs(metric["macro_f1"] - (2 / 3) / 7) < 1e-12
        and metric["contains"] == 0.5
        and metric["exact"] == 0,
        "Fixed-label metric fixture",
    )
    a.require(
        metric["full_roster_failure_bounds"][0]
        <= metric["macro_f1"]
        <= metric["full_roster_failure_bounds"][1],
        "Bounds fixture",
    )
    return {
        "passed": True,
        "fixtures": 6,
        "scope": "Selection nulls/nesting/order, sentinel independence, fixed-label counts and failure bounds",
    }


def run():
    from src.hsrag.parse import parse_response, extract_json

    a.require(
        not (BASE / "plan.json").exists() and not EVIDENCE.exists(),
        "Preserve baseline review",
    )
    tested = checks()
    b.validate()
    plan_sha = a.digest(b.PLAN)
    review_path = b.BASE / "stage_reviews/stage_01.json"
    receipt = a.read(review_path)
    audit_path = ROOT / receipt["evidence"]
    audit = a.read(audit_path)
    a.require(
        a.digest(audit_path) == receipt["evidence_sha256"] == AUDIT_SHA,
        "Stage audit differs",
    )
    complete_path = b.BASE / "live/stages/stage_01/complete.json"
    complete = a.read(complete_path)
    a.require(
        a.digest(complete_path) == receipt["complete_sha256"]
        and receipt["plan_sha256"] == plan_sha
        and receipt["status"] == "audited"
        and audit["semantic_replay_items"] == 2433,
        "Completed baseline binding differs",
    )
    a.require(
        audit["completed_stage_restart_new_POSTs"] == 0
        and not audit["duplicate_response_metadata_ids"],
        "Audit restart/response identity check failed",
    )
    bindings = {
        **audit["verified_input_hashes"],
        b.rel(audit_path): AUDIT_SHA,
        b.rel(review_path): a.digest(review_path),
        b.rel(__file__): a.digest(__file__),
    }
    a.save_new(
        BASE / "plan.json",
        {
            "created_utc": a.utc(),
            "status": "offline_baseline_review_planned",
            "tests": tested,
            "inputs": bindings,
            "raw_outputs": audit["verified_output_hashes"],
            "analysis": {
                "sequence": "Verify raw evidence, reconstruct votes, freeze actual pools and sentinels using selection only, then score reference.",
                "reference_groups": [
                    "evaluation_phase",
                    "transfer",
                    "holdout",
                    "sentinel_selection_pool",
                    "eligible_candidates",
                    "sentinel",
                ],
                "historical_comparison": "Same350 transfer IDs from original full-core three repeats, descriptive consistency snapshot only; no significance/causal claim or design change.",
                "labels": u.LABELS,
                "nulls": "Keep completed failures, exclude any whole-null selection from candidate/sentinel selection, report scored masks and wrong/right bounds.",
                "source_protection": "No mutation of live predictions, old studies, active KB or global taxonomy; no new API calls.",
            },
        },
    )
    verify_hashes(bindings)
    verify_hashes(audit["verified_output_hashes"])
    print(
        f"Verified {len(bindings)} input and {len(audit['verified_output_hashes'])} live output hashes.",
        flush=True,
    )
    schedule = a.read(b.BASE / "schedule.json")
    jobs = schedule["stages"][0]["jobs"]
    cat = b.frozen_catalog(ROOT / a.read(b.PLAN)["prompts"])
    live = a.storage_path(b.BASE / "live")
    phase_runs = {p: [{} for _ in range(3)] for p in ("selection", "reference")}
    job_ids = defaultdict(set)
    remote_ids = defaultdict(set)
    ids_all = Counter()
    invalid_labels = Counter()
    terminal_invalid = Counter()
    nulls = []
    verified = 0
    raw_turns = 0
    phase_accounting = defaultdict(Counter)
    exported = defaultdict(list)
    for number, job in enumerate(jobs, 1):
        identity = b.identity(job, plan_sha)
        folder = b.item_dir(live, job, plan_sha)
        value = a.read(folder / "result.json")
        a.require(
            value["identity"] == identity
            and value["execution_mode"] == "live"
            and value["active_model"] == a.MODEL,
            "Prediction identity/model/mode differs",
        )
        a.require(
            value["messages_sha256"]
            == cat[job["key"], job["condition"]]["messages_sha256"],
            "Prediction prompt differs",
        )
        a.require(
            a.digest(folder / "complete.json")
            == complete["item_receipt_hashes"][b.rel(folder / "complete.json")],
            "Prediction completion differs",
        )
        iid = job["key"].split("/", 1)[1]
        phase = job["phase"]
        repeat = job["repeat"]
        a.require(
            phase == job["condition"].removesuffix("_baseline")
            and iid not in phase_runs[phase][repeat],
            "Duplicate phase prediction",
        )
        recreated = []
        for k in range(3):
            saved = a.read(folder / "votes" / f"{k}.json")
            parsed = []
            for turn, raw in enumerate(saved["raw_outputs"]):
                obj, _, err = parse_response(raw)
                parsed.append(obj)
                raw_turns += 1
                if obj is None:
                    try:
                        loose = json.loads(extract_json(raw))
                        bad = set(loose.get("hate_type", [])) - set(u.LABELS)
                    except (ValueError, TypeError, AttributeError):
                        bad = set()
                    invalid_labels.update(bad)
                    if turn == len(saved["raw_outputs"]) - 1:
                        terminal_invalid.update(bad)
            result = parsed[-1].model_dump() if parsed[-1] is not None else None
            a.require(
                result == saved["result"] == value["run_predictions"][k]
                and all(x is None for x in parsed[:-1]),
                "Saved vote/raw repair path differs",
            )
            recreated.append(result)
        voted, n, tie = u.vote(recreated)
        a.require(
            n == value["n_valid"] and tie == value["uncertain"],
            "Independent vote accounting differs",
        )
        if voted is None:
            a.require(value["result"] is None, "Independent null differs")
            nulls.append(job)
        else:
            a.require(
                voted == {k: value["result"][k] for k in u.FIELDS},
                "Independent field vote differs",
            )
        u.check_prediction(value["result"], u.LABELS)
        row = {
            "id": iid,
            "key": job["key"],
            "condition": job["condition"],
            "phase": phase,
            "replicate": repeat,
            "result": value["result"],
            "n_valid": value["n_valid"],
            "subset": job["subset"],
            "source": b.rel(folder / "result.json"),
            "source_sha256": a.digest(folder / "result.json"),
        }
        phase_runs[phase][repeat][iid] = row
        exported[phase, repeat].append(row)
        job_ids[phase].add(a.signature(identity))
        phase_accounting[phase].update(value["accounting"])
        for event in value["events"]:
            meta = event["model_response_metadata"]
            if event["action"] == "ok":
                response_id = meta.get("response_id")

                call = (
                    folder
                    / "calls"
                    / f"vote_{event['vote']}_turn_{event['turn']}"
                    / f"attempt_{event['attempt']}"
                    / "outcome.json"
                )
                outcome = a.read(call)
                raw = json.loads(__import__("base64").b64decode(outcome["body_base64"]))
                response_id = raw.get("id")
                a.require(bool(response_id), "Response ID absent in baseline")
                ids_all[response_id] += 1
                remote_ids[phase].add(response_id)
        verified += 1
        if number % 500 == 0 or number == len(jobs):
            print(
                f"Independently reconstructed {number}/{len(jobs)} predictions.",
                flush=True,
            )
    a.require(
        verified == 2433 and raw_turns == 7394 and len(nulls) == 1,
        "Stage totals differ",
    )
    a.require(
        max(ids_all.values()) == 1
        and len(ids_all) == 7394
        and not remote_ids["selection"] & remote_ids["reference"]
        and not job_ids["selection"] & job_ids["reference"],
        "Selection/reference reused responses",
    )
    science = a.read(b.SCIENCE)
    roles = science["rosters"]
    roster = b.core.rows(b.preparation.OUT / "roster.jsonl")
    byid = {r["id"]: r for r in roster}
    gold = {i: set(r["gold"]["hate_type"]) for i, r in byid.items()}
    for phase, which in (
        ("selection", "selection_phase"),
        ("reference", "evaluation_phase"),
    ):
        for rep, run in enumerate(phase_runs[phase]):
            a.require(set(run) == set(roles[which]), "Phase roster differs")
    selection = select(
        roles["eligible_candidates"],
        roles["sentinel_selection_pool"],
        gold,
        phase_runs["selection"],
    )
    selection.update(
        created_utc=a.utc(),
        scientific_plan_sha256=b.STUDY_PLAN_SHA,
        baseline_plan_sha256=plan_sha,
        baseline_audit_sha256=AUDIT_SHA,
        selection_input_hashes={
            r["source"]: r["source_sha256"]
            for rr in phase_runs["selection"]
            for r in rr.values()
        },
        reference_predictions_used_for_selection=False,
    )

    selection_path = BASE / "selection.json"
    a.save_new(selection_path, selection)
    sentinel_path = BASE / "sentinels.json"
    a.save_new(
        sentinel_path,
        {
            "created_utc": a.utc(),
            "selection_sha256": a.digest(selection_path),
            "ids": selection["sentinel_ids"],
            "definition": "Gold contained in all three valid selection outputs on original pool229; no target size.",
            "reference_scored_yet": False,
        },
    )
    print(
        json.dumps(
            {
                "selection_frozen": True,
                "groups": {k: len(v) for k, v in selection["groups"].items()},
                "k": selection["k"],
                "levels": selection["levels"],
                "sentinels": len(selection["sentinel_ids"]),
            },
            indent=2,
        ),
        flush=True,
    )
    assembled = {}
    for (phase, rep), rows in exported.items():
        path = BASE / "assembled" / f"{phase}_rep{rep}.jsonl"
        write_rows(path, sorted(rows, key=lambda r: r["id"]))
        assembled[b.rel(path)] = {"rows": len(rows), "sha256": a.digest(path)}
    reference = {}
    group_ids = {
        name: roles[name]
        for name in (
            "evaluation_phase",
            "transfer",
            "holdout",
            "sentinel_selection_pool",
            "eligible_candidates",
        )
    }
    group_ids["sentinel"] = selection["sentinel_ids"]
    for name, ids in group_ids.items():
        if not ids:
            reference[name] = {"n": 0, "runs": [], "macro_f1": None}
            continue
        scores = [u.metric(ids, gold, run) for run in phase_runs["reference"]]
        reference[name] = {
            "n": len(ids),
            "runs": scores,
            "macro_f1": u.describe([v["macro_f1"] for v in scores]),
            "contains": u.describe([v["contains"] for v in scores]),
            "gate_false": u.describe([v["gate_false"] for v in scores]),
        }
    historical = [
        {
            r["id"]: r
            for r in b.core.rows(
                b.preparation.SCORES / f"assembled/rag_full_rep{k}.jsonl"
            )
        }
        for k in range(3)
    ]
    common = sorted(
        i
        for i in roles["transfer"]
        if all(
            run[i]["result"] is not None
            for run in [*historical, *phase_runs["reference"]]
        )
    )
    historical_scores = [u.metric(common, gold, run) for run in historical]
    fresh_scores = [u.metric(common, gold, run) for run in phase_runs["reference"]]
    consistency = {
        "common_valid_ids": common,
        "historical_runs": historical_scores,
        "fresh_runs": fresh_scores,
        "historical_macro": u.describe([v["macro_f1"] for v in historical_scores]),
        "fresh_macro": u.describe([v["macro_f1"] for v in fresh_scores]),
        "interpretation": "Descriptive repeated-baseline snapshot on same IDs; no isolated KB-size or feedback effect, no equivalence/causal inference.",
    }
    composition = []
    for pool in selection["sets"]:
        composition.append(
            {
                **{k: v for k, v in pool.items() if k != "ids"},
                "label_support": {
                    l: sum(l in gold[i] for i in pool["ids"]) for l in u.LABELS
                },
            }
        )
    sentinel_flips = []
    for rep, run in enumerate(phase_runs["reference"]):
        missing = [i for i in selection["sentinel_ids"] if run[i]["result"] is None]
        flips = [
            i
            for i in selection["sentinel_ids"]
            if run[i]["result"] is not None
            and not gold[i] <= set(run[i]["result"]["hate_type"])
        ]
        valid = len(selection["sentinel_ids"]) - len(missing)
        sentinel_flips.append(
            {
                "repeat": rep,
                "n": len(selection["sentinel_ids"]),
                "valid": valid,
                "null_ids": missing,
                "flipped_ids": flips,
                "flip_count": len(flips),
                "flip_rate": len(flips) / valid if valid else None,
            }
        )
    outputs = {
        **{p: rec["sha256"] for p, rec in assembled.items()},
        b.rel(selection_path): a.digest(selection_path),
        b.rel(sentinel_path): a.digest(sentinel_path),
    }
    report = {
        "created_utc": a.utc(),
        "status": "baseline_review_complete_selection_frozen",
        "plan_sha256": a.digest(BASE / "plan.json"),
        "scientific_plan_sha256": b.STUDY_PLAN_SHA,
        "baseline_plan_sha256": plan_sha,
        "audit_sha256": AUDIT_SHA,
        "tests": tested,
        "verified_inputs": bindings,
        "verified_live_output_hashes": audit["verified_output_hashes"],
        "outputs": outputs,
        "assembled": assembled,
        "verified_predictions": verified,
        "reparsed_response_turns": raw_turns,
        "phase_job_counts": {p: len(v) for p, v in job_ids.items()},
        "phase_remote_response_ids": {p: len(v) for p, v in remote_ids.items()},
        "shared_phase_job_or_remote_ids": 0,
        "nulls": nulls,
        "terminal_invalid_hate_type_labels": dict(terminal_invalid),
        "all_invalid_hate_type_labels": dict(invalid_labels),
        "operational": {
            k: audit[k]
            for k in (
                "wall_seconds_including_final_save",
                "counts",
                "accounting",
                "observed_usage_sums",
                "reported_models",
                "finish_reasons",
            )
        },
        "phase_accounting": {p: dict(v) for p, v in phase_accounting.items()},
        "candidate_groups": {k: len(v) for k, v in selection["groups"].items()},
        "k": selection["k"],
        "levels": selection["levels"],
        "sentinel_count": len(selection["sentinel_ids"]),
        "sentinel_fresh_reference_flips": sentinel_flips,
        "reference": reference,
        "historical_full_core_consistency": consistency,
        "selected_pool_label_support": composition,
        "remaining_execution": {
            "predictions": 33552,
            "ordinary_responses": 100656,
            "linear_baseline_pace_hours": 33552
            / 2433
            * audit["wall_seconds_including_final_save"]
            / 3600,
            "qualification": "Extrapolation only; larger prompts, server variation, repairs and offline audits can increase elapsed time.",
        },
        "actual_API_calls": 0,
        "next": "Freeze actual correction records and all update retrieval/payloads, independently audit and replay, then prepare one resumable launcher for the remaining three batches.",
    }
    a.save_new(EVIDENCE, report)
    a.save_new(
        BASE / "complete.json",
        {
            "status": report["status"],
            "evidence": b.rel(EVIDENCE),
            "evidence_sha256": a.digest(EVIDENCE),
            "plan_sha256": a.digest(BASE / "plan.json"),
            "selection_sha256": a.digest(selection_path),
            "sentinels_sha256": a.digest(sentinel_path),
            "outputs": outputs,
        },
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "predictions": verified,
                "nulls": len(nulls),
                "candidate_groups": report["candidate_groups"],
                "k": selection["k"],
                "levels": selection["levels"],
                "sentinels": report["sentinel_count"],
                "reference_transfer_macro": reference["transfer"]["macro_f1"],
                "historical_transfer_macro": consistency["historical_macro"],
                "sentinel_flip_counts": [x["flip_count"] for x in sentinel_flips],
                "remaining_hours_at_baseline_pace": report["remaining_execution"][
                    "linear_baseline_pace_hours"
                ],
                "evidence_sha256": a.digest(EVIDENCE),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["tests", "run"])
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", u.forbid), patch.object(
        socket, "create_connection", u.forbid
    ), patch.object(b.core, "live_client", u.forbid):
        print(json.dumps(checks(), indent=2)) if args.mode == "tests" else run()

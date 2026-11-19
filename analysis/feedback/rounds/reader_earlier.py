from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
from types import FunctionType, SimpleNamespace

from analysis.feedback.transfer import archive as pure

ROOT = pure.ROOT
STUDY = ROOT / "experiments/astra/sq3_iterative_20261012_v1"
REVIEW = ROOT / "experiments/astra/sq3_iterative_review_20261026_v1"
OUT = REVIEW / "part2"
READER = OUT / "reader_v2"
STREAMS = [2026101201, 2026101202, 2026101203]
POLICIES = ["adaptive", "static", "random"]
CONDITIONS = ["reference"] + [f"{p}_s{s}" for s in STREAMS for p in POLICIES]
need, sha, read, rows, rel, physical = (
    pure.need,
    pure.sha,
    pure.read,
    pure.rows,
    pure.rel,
    pure.physical,
)


def write(path, value):
    need(
        path.resolve().is_relative_to(OUT.resolve()),
        "Write outside the new Part 2 folder",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with physical(path).open("xb") as stream:
        stream.write(
            value if isinstance(value, bytes) else pure.encoded(value, pretty=True)
        )


def write_lines(path, values):
    need(
        path.resolve().is_relative_to(OUT.resolve()),
        "Write outside the new Part 2 folder",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with physical(path).open("xb") as stream:
        for value in values:
            stream.write(pure.encoded(value))


def bound_helpers(plan_sha, live):

    def identity(job):
        return {
            "experiment_sha256": plan_sha,
            "condition": job["study"] + "::" + job["condition"],
            "repeat": job["repeat"],
            "key": job["key"],
        }

    def item_id(job):
        return hashlib.sha256(pure.encoded(identity(job), pretty=True)).hexdigest()

    namespace = dict(
        vars(pure),
        identity=identity,
        item_id=item_id,
        LIVE=live,
        validate_prediction=validate_prediction,
    )
    compact = FunctionType(pure.compact.__code__, namespace, "phase_compact")
    return SimpleNamespace(identity=identity, item_id=item_id, compact=compact)


def validate_prediction(value):

    if value is None:
        return
    need(type(value["hate"]) is bool, "Nonboolean prediction gate")
    for field, permitted in pure.LABELS.items():
        labels = value[field]
        need(
            type(labels) is list and all(type(x) is str for x in labels),
            "Invalid label list",
        )
        need(set(labels) <= permitted, "Unknown label: " + field)
    need(
        value["severity"] is None or value["severity"] in ("low", "medium", "high"),
        "Unknown severity",
    )
    if value["hate"] is False:
        need(
            value["hate_type"] == []
            and value["target_group"] == []
            and value["severity"] is None,
            "Saved false gate was not normalised",
        )


def phase_axes(name):
    if name == "source_initial":
        return {"role": "source", "round": 0, "stream": None}
    match = re.fullmatch(r"r([1-4])_evaluation", name)
    if match:
        return {"role": "evaluation", "round": int(match[1]), "stream": None}
    match = re.fullmatch(r"r([2-4])_source_s(202610120[1-3])", name)
    need(match is not None, "Unknown phase name")
    return {"role": "source", "round": int(match[1]), "stream": int(match[2])}


def read_item(job, receipts, live, helpers):
    name = "items/" + helpers.item_id(job)
    receipt_name = name + "/complete.json"
    receipt_sha = receipts[receipt_name]
    receipt = pure.read_bound(live / receipt_name, receipt_sha)
    need(receipt["status"] == "complete", "Incomplete item")
    result_name = name + "/result.json"
    result_sha = receipt["hashes"][result_name]
    saved = pure.read_bound(live / result_name, result_sha)
    return helpers.compact(saved, job, receipt, receipt_sha, result_sha)


def matrix_row(source, index, records, replay):
    expected = {(r, c, d) for r in range(1, 5) for c in CONDITIONS for d in range(3)}
    by_slot = {(v["round"], v["condition"], v["repeat"]): v for v in records}
    need(
        len(records) == len(by_slot) == 120 and set(by_slot) == expected,
        "Missing or duplicate four-round slot",
    )
    need(
        all(
            v["key"] == source["key"]
            and v["population"] == source["population"]
            and v["role"] == "evaluation"
            for v in records
        ),
        "Wrong roster/role join",
    )
    need(
        all(
            len({v["stage"] for v in records if v["round"] == r}) == 1
            for r in range(1, 5)
        ),
        "Input split across batches within a round",
    )
    prediction = [
        [
            [
                {k: by_slot[r, c, d][k] for k in ("result", "n_valid", "uncertain")}
                for d in range(3)
            ]
            for c in CONDITIONS
        ]
        for r in range(1, 5)
    ]
    valid = [
        all(v["result"] is not None for arm in rr for v in arm) for rr in prediction
    ]
    strict = [all(v["n_valid"] == 3 for arm in rr for v in arm) for rr in prediction]
    return {
        "input_index": index,
        "key": source["key"],
        "population": source["population"],
        "source": source["source"],
        "source_file": source["source_file"],
        **{
            k: source[k]
            for k in (
                "inference_cluster",
                "functional_cluster",
                "text_cluster",
                "overlap_cluster",
            )
        },
        "gold_hate": source["gate"],
        **{"gold_" + k: source.get(k) for k in pure.GOLD_FIELDS},
        "known_fine_gold": {k: source.get(k) is not None for k in pure.GOLD_FIELDS},
        "predictions": prediction,
        "masks": {
            "round_common": valid,
            "round_strict": strict,
            "all_round_common": all(valid),
            "all_round_strict": all(strict),
            "replay_exclusion_member": source["key"] in replay,
        },
    }


def build():
    started = time.monotonic()
    need(
        not (READER / "complete.json").exists(),
        "Reader already complete; use verify instead",
    )
    part1_done = read(REVIEW / "part1/complete.json")
    ready = pure.read_bound(
        REVIEW / "part1/readiness.json", part1_done["readiness_sha256"]
    )
    bindings = {}

    def bind(path, expected=None):
        expected = expected or ready["source_bindings"].get(rel(path)) or sha(path)
        value = pure.read_bound(path, expected)
        bindings[rel(path)] = expected
        return value

    for path in [
        REVIEW / "part1/complete.json",
        REVIEW / "part1/readiness.json",
        REVIEW / "review_plan.json",
    ]:
        bind(path)
    for path in [
        Path(__file__),
        Path(pure.__file__),
        ROOT / "tests/feedback/rounds/test_reader_earlier.py",
    ]:
        bindings[rel(path)] = sha(path)
    for name in ["src/hsrag/schema.py", "src/hsrag/vote.py", "config/taxonomy.yaml"]:
        path = ROOT / name
        need(sha(path) == ready["source_bindings"][name], "Frozen schema/voter changed")
        bindings[name] = sha(path)
    roster_path = STUDY / "evaluation.jsonl"
    need(
        sha(roster_path) == ready["source_bindings"][rel(roster_path)],
        "Evaluation roster changed",
    )
    bindings[rel(roster_path)] = sha(roster_path)
    roster = list(rows(roster_path))
    positions = {v["key"]: i for i, v in enumerate(roster)}
    need(
        len(roster) == len(positions) == 2077
        and all(type(v["gate"]) is bool for v in roster),
        "Wrong evaluation roster",
    )
    sensitivity = bind(
        ROOT / "experiments/astra/sq3_followups_20261012_v1/scope/sensitivity.json"
    )
    replay = set(sensitivity["replay_keys"]) & positions.keys()
    by_key = defaultdict(list)
    totals = [Counter(), Counter(), Counter()]
    stage_receipts = []
    duplicate_labels = []
    for phase in ready["phases"]:
        name = phase["phase"]
        axes = phase_axes(name)
        root = STUDY / "phases" / name
        execution = root / "execution"
        plan_sha = phase["phase_plan_sha256"]
        plan = bind(execution / "plan.json", plan_sha)
        science = bind(root / "scientific_plan_v1.json")
        schedule = bind(execution / "schedule.json")
        done = bind(execution / "live/complete.json")
        need(
            done["status"] == "all_studies_executed_and_audited"
            and done["plan_sha256"] == plan_sha,
            "Wrong phase completion",
        )
        phase_roster = root / "roster.jsonl"
        roster_sha = plan["input_hashes"][rel(phase_roster)]
        need(sha(phase_roster) == roster_sha, "Phase roster changed")
        bindings[rel(phase_roster)] = roster_sha
        phase_rows = {v["key"]: v for v in rows(phase_roster)}
        need(
            len(phase_rows) == science["inputs"] == phase["inputs"],
            "Wrong phase roster size",
        )
        if axes["role"] == "evaluation":
            need(
                list(phase_rows.values()) == roster
                and science["conditions"] == CONDITIONS,
                "Evaluation axes changed",
            )
        phase_totals = [Counter(), Counter(), Counter()]
        expected_seen = {
            (k, c, d)
            for k in phase_rows
            for c in science["conditions"]
            for d in range(3)
        }
        seen = set()
        for stage in schedule["stages"]:
            live = execution / "live" / stage["study"]
            target = live / "stages" / stage["id"]
            review = bind(
                target / "review.json",
                done["review_hashes"][rel(target / "review.json")],
            )
            complete = bind(target / "complete.json", review["complete_sha256"])
            report_path = target / review["report"]
            report = bind(report_path, review["artifacts"][review["report"]])
            need(
                review["status"] == "audited"
                and review["mode"] == "live"
                and review["plan_sha256"] == plan_sha,
                "Wrong stage review",
            )
            need(
                not complete["summary"]["stopped"]
                and not complete["summary"]["errors"],
                "Stopped batch",
            )
            helpers = bound_helpers(plan_sha, live)
            receipt_names = {
                "items/" + helpers.item_id(j) + "/complete.json" for j in stage["jobs"]
            }
            need(
                receipt_names == set(complete["item_receipt_hashes"]),
                "Scheduled item receipts differ",
            )
            output = READER / "stages" / name / stage["id"]
            local_inputs = {
                rel(p): sha(p)
                for p in [
                    execution / "plan.json",
                    target / "review.json",
                    target / "complete.json",
                    report_path,
                ]
            }
            if (output / "complete.json").exists():
                receipt = read(output / "complete.json")
                need(receipt["inputs"] == local_inputs, "Reader resume input changed")
                need(
                    sha(output / "predictions.jsonl") == receipt["predictions_sha256"],
                    "Reader output changed",
                )
                records = list(rows(output / "predictions.jsonl"))
            else:
                with ThreadPoolExecutor(max_workers=8) as executor:
                    records = list(
                        executor.map(
                            lambda j: read_item(
                                j, complete["item_receipt_hashes"], live, helpers
                            ),
                            stage["jobs"],
                        )
                    )
                for value in records:
                    value.update(
                        phase=name,
                        **axes,
                        stage=stage["id"],
                        phase_plan_sha256=plan_sha,
                        effective_worker_limit=128,
                    )
                write_lines(output / "predictions.jsonl", records)
                receipt = {
                    "status": "bound_saved_results_assembled",
                    "inputs": local_inputs,
                    "predictions_sha256": sha(output / "predictions.jsonl"),
                    "rows": len(records),
                    "reader_sha256": sha(Path(__file__)),
                    "pure_helpers_sha256": sha(Path(pure.__file__)),
                }
                write(output / "complete.json", receipt)
            need(
                receipt["reader_sha256"] == sha(Path(__file__))
                and receipt["pure_helpers_sha256"] == sha(Path(pure.__file__)),
                "Reader source changed during assembly",
            )
            counts, accounting, usage = pure.record_counts(records)
            for actual, field in zip(
                [counts, accounting, usage], ["counts", "accounting", "observed_usage"]
            ):
                need(
                    actual == Counter(report[field]),
                    f'{name}/{stage["id"]}: {field} differs from audit',
                )
            for job, value in zip(stage["jobs"], records):
                slot = (value["key"], value["condition"], value["repeat"])
                need(
                    slot not in seen and slot in expected_seen,
                    "Duplicate or unexpected phase slot",
                )
                need(
                    all(
                        value[k] == job[k]
                        for k in [
                            "key",
                            "condition",
                            "repeat",
                            "population",
                            "messages_sha256",
                        ]
                    ),
                    "Assembly job order differs",
                )
                seen.add(slot)
                for vote_index, vote in enumerate(value["votes"]):
                    if vote is not None:
                        for field in pure.LABELS:
                            if len(vote[field]) != len(set(vote[field])):
                                duplicate_labels.append(
                                    {
                                        "phase": name,
                                        "key": value["key"],
                                        "condition": value["condition"],
                                        "repeat": value["repeat"],
                                        "vote": vote_index,
                                        "field": field,
                                        "labels": vote[field],
                                        "item_receipt": value["item_receipt"],
                                        "result_sha256": value["result_sha256"],
                                    }
                                )
                if axes["role"] == "evaluation":
                    by_key[value["key"]].append(
                        {
                            k: value[k]
                            for k in [
                                "key",
                                "population",
                                "condition",
                                "repeat",
                                "round",
                                "role",
                                "stage",
                                "result",
                                "n_valid",
                                "uncertain",
                            ]
                        }
                    )
            for accumulator, count in zip(phase_totals, [counts, accounting, usage]):
                accumulator.update(count)
            stage_receipts.append(
                {
                    "phase": name,
                    "stage": stage["id"],
                    "role": axes["role"],
                    "receipt": rel(output / "complete.json"),
                    "receipt_sha256": sha(output / "complete.json"),
                    "predictions": rel(output / "predictions.jsonl"),
                    "counts": dict(counts),
                }
            )
            print(
                f'Reader {name}/{stage["id"]}: {len(records):,} bound predictions checked',
                flush=True,
            )
        need(seen == expected_seen, "Incomplete phase matrix")
        for actual, field in zip(
            phase_totals, ["counts", "accounting", "observed_usage"]
        ):
            need(actual == Counter(done[field]), f"{name}: phase {field} differs")
        for accumulator, count in zip(totals, phase_totals):
            accumulator.update(count)
    need(
        totals[0] == Counter(ready["counts"])
        and totals[1] == Counter(ready["accounting"]),
        "Global audit totals differ",
    )
    matrix = [
        matrix_row(source, i, by_key[source["key"]], replay)
        for i, source in enumerate(roster)
    ]
    write_lines(READER / "matrix.jsonl", matrix)
    write(READER / "roster.jsonl", physical(roster_path).read_bytes())
    write(READER / "duplicate_vote_labels.json", duplicate_labels)
    masks = {}
    for pop in ["en_test", "en_hatecheck_eval"]:
        masks[pop] = {}
        for gold in [False, True]:
            rr = [
                v for v in matrix if v["population"] == pop and v["gold_hate"] == gold
            ]
            masks[pop][str(gold)] = {
                "total": len(rr),
                "clusters": len({v["inference_cluster"] for v in rr}),
                "round_common": [
                    sum(v["masks"]["round_common"][r] for v in rr) for r in range(4)
                ],
                "round_strict": [
                    sum(v["masks"]["round_strict"][r] for v in rr) for r in range(4)
                ],
                "all_round_common": sum(v["masks"]["all_round_common"] for v in rr),
                "all_round_strict": sum(v["masks"]["all_round_strict"] for v in rr),
            }
    receipt = {
        "status": "all_four_rounds_reader_assembled",
        "created_utc": pure.utc(),
        "conditions": CONDITIONS,
        "streams": STREAMS,
        "rounds": [1, 2, 3, 4],
        "decoder_repeats": [0, 1, 2],
        "input_bindings": bindings,
        "stage_receipts": stage_receipts,
        "artifacts": {
            rel(p): sha(p)
            for p in [
                READER / "matrix.jsonl",
                READER / "roster.jsonl",
                READER / "duplicate_vote_labels.json",
            ]
        },
        "counts": dict(totals[0]),
        "accounting": dict(totals[1]),
        "observed_usage": dict(totals[2]),
        "mask_counts": masks,
        "replay_keys": sorted(replay),
        "fixed_support": pure.fixed_support(roster),
        "elapsed_seconds": time.monotonic() - started,
        "new_Qwen_calls": 0,
        "new_fits": 0,
        "raw_responses_reparsed": False,
        "scientific_effects_scored": False,
        "duplicate_vote_label_fields": len(duplicate_labels),
        "reader_adjustment": "Preserve duplicate list entries accepted by the frozen schema and their occurrence-count aggregation; no saved prediction changed. Failed v1 assembly remains preserved.",
        "reference_rule": "one condition x three decoder repeats per round; never replicated by source stream",
    }
    write(READER / "complete.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "counts": receipt["counts"],
                "mask_counts": masks,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["build"])
    parser.parse_args()
    with pure.offline():
        build()

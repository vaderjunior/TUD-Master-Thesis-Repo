from src.hsrag.paths import PROJECT_ROOT
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import socket
from unittest.mock import patch

ROOT = PROJECT_ROOT
STUDY = ROOT / "experiments/astra/sq3_iterative_20261012_v1"
OUT = ROOT / "experiments/astra/sq3_iterative_review_20261026_v1/part1"
SEEDS = (2026101201, 2026101202, 2026101203)
POLICIES = ("adaptive", "static", "random")
CONDITIONS = ["reference"] + [f"{p}_s{s}" for s in SEEDS for p in POLICIES]
BINDINGS = {}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def physical(path):
    value = str(Path(str(path).removeprefix("\\\\?\\")).resolve())
    return Path("\\\\?\\" + value) if os.name == "nt" else Path(value)


def rel(path):
    return (
        Path(str(path).removeprefix("\\\\?\\")).resolve().relative_to(ROOT).as_posix()
    )


def read(path):
    return json.loads(physical(path).read_bytes())


def sha(path):
    digest = hashlib.sha256()
    with physical(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bind(path, expected=None):
    name = rel(path)
    value = BINDINGS.get(name)
    if value is None:
        value = sha(path)
        BINDINGS[name] = value
    need(expected is None or value == expected, "Changed bound file: " + name)
    return value


def check(mapping):
    for name, expected in mapping.items():
        bind(ROOT / name, expected)


def rows(path):
    with physical(path).open("rb") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def write(name, value):
    path = OUT / name
    need(
        path.resolve().is_relative_to(OUT.resolve()), "Output outside new review folder"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as stream:
        json.dump(
            value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        stream.write("\n")


def forbidden(*args, **kwargs):
    raise AssertionError("Part 1 is offline: no sockets, model calls or fitting")


@contextmanager
def offline():
    with patch.object(socket.socket, "connect", forbidden), patch.object(
        socket.socket, "connect_ex", forbidden
    ), patch.object(socket, "create_connection", forbidden):
        yield


def completions():
    done = read(STUDY / "complete.json")
    bind(STUDY / "complete.json")
    plan = read(STUDY / "plan.json")
    bind(STUDY / "plan.json", done["plan_sha256"])
    need(
        done["status"] == "all_four_simulated_rounds_audited_scientific_review_pending"
        and done["evaluation_scoring_performed"] is False
        and done["real_expert_feedback"] is False,
        "Wrong final study state",
    )
    need(
        done["totals"] == {"ordinary_calls": 777672, "predictions": 259224},
        "Wrong scientific scope",
    )
    check(plan["source_hashes"])
    check(plan["outputs"])
    check(plan["inputs"])
    check(done["phase_receipts"])
    names = {
        "runtime": "concurrency_20261024_v1/live_runtime_complete.json",
        "outage": "recovery_20261024_v1/live_recovery_complete.json",
        "paths": "path_20261024_v2/live_path_runtime_complete.json",
        "ranking": "ranking_20261025_v3/live_ranking_runtime_complete.json",
    }
    receipts = {}
    manifests = {}
    statuses = {
        "runtime": "all_four_rounds_and_runtime_provenance_verified_scientific_review_pending",
        "outage": "all_four_rounds_and_reviewed_outage_audited_scientific_review_pending",
        "paths": "all_rounds_outage_recovery_and_path_continuation_verified_scientific_review_pending",
        "ranking": "all_rounds_and_checker_continuation_verified_scientific_review_pending",
    }
    for key, name in names.items():
        path = STUDY / "execution" / name
        receipt = read(path)
        bind(path)
        need(
            receipt["status"] == statuses[key]
            and receipt["original_complete_sha256"] == bind(STUDY / "complete.json"),
            "Completion chain differs",
        )
        manifest = path.parent / "manifest.json"
        bind(
            manifest,
            (
                receipt["manifest_sha256"]
                if key != "outage"
                else receipt["recovery"]["manifest_sha256"]
            ),
        )
        amendment = read(manifest)
        check(amendment["bindings"] if key == "runtime" else amendment["source_hashes"])
        receipts[key] = receipt
        manifests[key] = amendment
    runtime, outage, paths, ranking = (
        receipts[k] for k in ("runtime", "outage", "paths", "ranking")
    )
    need(
        runtime["phase_receipts"] == done["phase_receipts"]
        and runtime["scientific_scoring"] is False
        and runtime["cumulative_classifier_started"] is False
        and runtime["next_queue_started"] is False,
        "Unexpected later work",
    )
    need(
        paths["runtime_complete_sha256"] == bind(STUDY / "execution" / names["runtime"])
        and paths["outage_complete_sha256"]
        == bind(STUDY / "execution" / names["outage"])
        and ranking["path_complete_sha256"]
        == bind(STUDY / "execution" / names["paths"]),
        "Broken recovery chain",
    )
    need(
        manifests["runtime"]["parent_plan_sha256"] == bind(STUDY / "plan.json")
        and manifests["outage"]["parent_runtime_sha256"]
        == bind(STUDY / "execution/concurrency_20261024_v1/manifest.json")
        and manifests["paths"]["parent_outage_manifest_sha256"]
        == bind(STUDY / "execution/recovery_20261024_v1/manifest.json")
        and manifests["ranking"]["parent_path_manifest_sha256"]
        == bind(STUDY / "execution/path_20261024_v2/manifest.json"),
        "Amendment parent chain differs",
    )
    check(runtime["marker_hashes"])
    check(paths["source_extraction_markers"])
    check(ranking["phase_markers"])
    need(
        len(paths["source_extraction_markers"]) == 10
        and len(ranking["phase_markers"]) == 5,
        "Missing source/checker marker",
    )
    need(
        runtime["actual_extra_API_calls"]
        == outage["actual_finalization_API_calls"]
        == paths["actual_finalization_API_calls"]
        == ranking["actual_finalization_Qwen_calls"]
        == 0,
        "Finalization made extra calls",
    )
    return done, plan, receipts


def phases(done, runtime):
    expected = {
        "source_initial",
        *[f"r{r}_evaluation" for r in range(1, 5)],
        *[f"r{r}_source_s{s}" for r in range(2, 5) for s in SEEDS],
    }
    need(
        {Path(name).parents[2].name for name in done["phase_receipts"]} == expected,
        "Phase roster differs",
    )
    reference = list(rows(STUDY / "evaluation.jsonl"))
    keys = {r["key"] for r in reference}
    need(len(reference) == len(keys) == 2077, "Evaluation roster repeats")
    need(
        Counter(r["population"] for r in reference)
        == {"en_test": 1024, "en_hatecheck_eval": 1053},
        "Evaluation scope differs",
    )
    scope = STUDY.parent / "sq3_followups_20261012_v1/scope"
    sample = read(scope / "iteration_evaluation.json")
    bind(scope / "iteration_evaluation.json")
    need(
        set(sample["keys"]) == keys and sample["outcome_selected"] is False,
        "Outcome-selected evaluation roster",
    )
    check(read(scope / "complete.json")["outputs"])
    bind(scope / "complete.json")
    counts = Counter()
    accounting = Counter()
    details = []
    stage_count = 0
    for name in sorted(done["phase_receipts"]):
        completion = ROOT / name
        root = completion.parents[2]
        execution = root / "execution"
        phase = read(completion)
        plan = read(execution / "plan.json")
        plan_sha = bind(execution / "plan.json", phase["plan_sha256"])
        check(plan["source_hashes"])
        check(plan["input_hashes"])
        science = read(root / "scientific_plan_v1.json")
        roster = list(rows(root / "roster.jsonl"))
        schedule = read(execution / "schedule.json")
        bind(execution / "schedule.json")
        source = root.name == "source_initial" or "_source_" in root.name
        conditions = (
            ["reference" if root.name == "source_initial" else "adaptive"]
            if source
            else CONDITIONS
        )
        need(
            science["conditions"] == conditions and science["inputs"] == len(roster),
            "Condition or input matrix differs",
        )
        need(
            len(roster)
            == (1024 if root.name == "source_initial" else 256 if source else 2077),
            "Wrong phase size",
        )
        if not source:
            need(roster == reference, "Evaluation roster changed between rounds")
        expected_slots = {
            (r["key"], condition, rep)
            for r in roster
            for condition in conditions
            for rep in range(3)
        }
        actual = set()
        owners = {}
        stage_counts = Counter()
        stage_accounting = Counter()
        need(
            phase["status"] == "all_studies_executed_and_audited"
            and phase["mode"] == "live",
            "Unaudited phase",
        )
        check(phase["review_hashes"])
        for stage in schedule["stages"]:
            target = execution / "live" / stage["study"] / "stages" / stage["id"]
            rp = target / "review.json"
            review = read(rp)
            need(
                rel(rp) in phase["review_hashes"], "Review absent from phase completion"
            )
            need(
                review["status"] == "audited"
                and review["plan_sha256"] == plan_sha
                and review["stage"] == stage["id"]
                and review["mode"] == "live",
                "Wrong audit binding",
            )
            bind(target / "complete.json", review["complete_sha256"])
            for leaf, digest in review["artifacts"].items():
                bind(target / leaf, digest)
            report = read(target / review["report"])
            closed = read(target / "complete.json")
            need(
                report["status"] == "schema_specific_stage_audited"
                and report["plan_sha256"] == plan_sha,
                "Wrong audit report",
            )
            need(
                not closed["summary"]["stopped"] and not closed["summary"]["errors"],
                "Final attempt stopped",
            )
            n = len(stage["jobs"])
            need(
                n
                == stage["items"]
                == closed["summary"]["saved_items"]
                == report["counts"]["predictions"],
                "Stage count differs",
            )
            need(stage["ordinary_calls"] == 3 * n, "Ordinary-vote budget differs")
            for job in stage["jobs"]:
                slot = job["key"], job["condition"], job["repeat"]
                need(
                    slot not in actual and slot in expected_slots,
                    "Duplicate or unexpected prediction slot",
                )
                actual.add(slot)
                need(
                    owners.setdefault(job["key"], stage["id"]) == stage["id"],
                    "Paired input split across batches",
                )
            stage_counts.update(report["counts"])
            stage_accounting.update(report["accounting"])
            stage_count += 1
        need(
            actual == expected_slots
            and dict(stage_counts) == phase["counts"]
            and dict(stage_accounting) == phase["accounting"],
            "Phase matrix/count reconciliation failed",
        )
        counts.update(phase["counts"])
        accounting.update(phase["accounting"])
        details.append(
            dict(
                phase=root.name,
                source=source,
                inputs=len(roster),
                conditions=conditions,
                stages=len(schedule["stages"]),
                counts=phase["counts"],
                accounting=phase["accounting"],
                distinct_remote_ids=phase["distinct_remote_ids"],
                phase_plan_sha256=plan_sha,
                completion=rel(completion),
            )
        )
        print(
            f"Verified completed phase {root.name}: {len(actual):,} prediction slots",
            flush=True,
        )
    need(
        stage_count == 38
        and dict(counts) == runtime["counts"]
        and dict(accounting) == runtime["accounting"],
        "Global count reconciliation failed",
    )
    need(
        sum(p["distinct_remote_ids"] for p in details)
        == runtime["global_distinct_response_ids"]
        == accounting["http_success_with_text"],
        "Global distinct-response total differs",
    )
    return details, dict(counts), dict(accounting), reference


def selection_checks():
    from analysis.feedback.rounds import policy as policy

    candidates = list(
        rows(STUDY.parent / "sq3_additions_20261004_v1/benign_selection/roster.jsonl")
    )
    by = {r["id"]: r for r in candidates}
    blocks = read(STUDY / "source_blocks.json")["blocks"]
    cached = {}
    details = []
    for root in sorted((STUDY / "phases").iterdir()):
        if root.name != "source_initial" and "_source_" not in root.name:
            continue
        path = root / "execution/source_gates.json"
        receipt_path = root / "execution/source_gates_receipt.json"
        receipt = read(receipt_path)
        bind(receipt_path)
        bind(path, receipt["bindings"][rel(path)])
        cached[root.name] = read(path)
        marker = read(root / "execution/path_20261024_v2/extract.json")
        need(
            marker["source_gates_sha256"] == bind(path)
            and marker["source_gates_receipt_sha256"] == bind(receipt_path),
            "Source extraction proof differs",
        )
        need(
            receipt["gold_evaluation_read"] is False,
            "Source selection used evaluation gold",
        )
    baseline = cached["source_initial"]
    unique = set()
    for seed in SEEDS:
        need(
            blocks[str(seed)] == policy.partition(candidates, seed),
            "Frozen source partition differs",
        )
        previous = {p: [] for p in POLICIES}
        for round_no in range(1, 5):
            ids = blocks[str(seed)][round_no - 1]
            rr = [by[i] for i in ids]
            old = {i: baseline[i] for i in ids}
            current = old if round_no == 1 else cached[f"r{round_no}_source_s{seed}"]
            expected = policy.verified_selection(
                rr, old, current, previous, seed, round_no
            )
            path = STUDY / f"rounds/r{round_no}/selection_s{seed}.json"
            saved = read(path)
            bind(path)
            need(
                all(saved[k] == value for k, value in expected.items()),
                "Saved selection differs from both frozen implementations",
            )
            need(
                saved["evaluation_outcomes_read"] is False
                and saved["parent_plan_sha256"] == bind(STUDY / "plan.json"),
                "Selection dependency differs",
            )
            for name, h in saved["source_bindings"].items():

                if "/items/" not in name:
                    bind(ROOT / name, h)
            for values in saved["cumulative"].values():
                unique.add(tuple(values))
            details.append(
                dict(
                    seed=seed,
                    round=round_no,
                    per_class_budget=saved["per_class_budget"],
                    additions_per_policy={
                        p: len(saved["additions"][p]) for p in POLICIES
                    },
                    cumulative_per_policy={
                        p: len(saved["cumulative"][p]) for p in POLICIES
                    },
                    invalid_source_ids=len(saved["invalid_ids"]),
                    shortfall=saved["shortfall"],
                )
            )
            previous = saved["cumulative"]
    need(len(details) == 12, "Twelve stream-round selections required")
    return details, len(unique)


def main():
    need(not (OUT / "complete.json").exists(), "Preserve completed Part 1")
    done, plan, receipts = completions()
    details, counts, accounting, roster = phases(done, receipts["runtime"])
    selections, unique = selection_checks()
    bind(Path(__file__))
    populations = {
        pop: dict(
            inputs=sum(r["population"] == pop for r in roster),
            inference_clusters=len(
                {r["inference_cluster"] for r in roster if r["population"] == pop}
            ),
            hateful=sum(r["population"] == pop and r["gate"] is True for r in roster),
            benign=sum(r["population"] == pop and r["gate"] is False for r in roster),
        )
        for pop in ("en_test", "en_hatecheck_eval")
    }
    need(
        sum(p["counts"]["predictions"] for p in details if p["source"]) == 9984,
        "Screening scope differs",
    )
    result = dict(
        status="completion_and_metadata_readiness_verified_scientific_analysis_pending",
        created_utc=datetime.now(timezone.utc).isoformat(),
        phase_count=14,
        stage_count=38,
        scientific_prediction_slots=259224,
        evaluation_slots=249240,
        screening_slots=9984,
        counts=counts,
        accounting=accounting,
        populations=populations,
        phases=details,
        selections=selections,
        distinct_cumulative_id_sets=unique,
        final_completion_chain={
            key: receipt["status"] for key, receipt in receipts.items()
        },
        reused_existing_full_raw_audits=True,
        replayed_raw_responses=False,
        result_reader_assembled=False,
        evaluation_predictions_scored=False,
        global_response_uniqueness_reused_from_final_audit=True,
        actual_Qwen_calls=0,
        actual_model_list_GETs=0,
        new_fits=0,
        new_embeddings=0,
        source_bindings=dict(sorted(BINDINGS.items())),
    )
    for name, digest in BINDINGS.items():
        need(sha(ROOT / name) == digest, "File changed during review: " + name)
    write("readiness.json", result)
    write(
        "complete.json",
        dict(
            status="focused_part1_complete",
            readiness_sha256=sha(OUT / "readiness.json"),
            source_bindings_count=len(BINDINGS),
            actual_Qwen_calls=0,
            new_fits=0,
            scientific_effects_scored=False,
        ),
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "phase_count",
                    "stage_count",
                    "counts",
                    "populations",
                    "distinct_cumulative_id_sets",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    with offline():
        main()

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import time
from analysis.classification.matched import common as m

a, c, e, r = m.a, m.c, m.e, m.runtime


def assemble():
    with r.offline():
        r.validate("matched")
        b = r.bind_stage(
            e.queue("matched"), dict(path=r.rel(r.MANIFEST), sha256=c.sha(r.MANIFEST))
        )
        schedule = c.read(b.BASE / "schedule.json")
        done_path = b.BASE / "live/complete.json"
        runtime_path = b.BASE / "concurrency_20261029_v1/live_complete.json"
        done, runtime_done = a.read(done_path), a.read(runtime_path)
        a.require(
            done["status"] == "all_studies_executed_and_audited"
            and done["mode"] == "live",
            "Study not fully audited",
        )
        a.require(
            done["counts"]["predictions"] == 36699
            and done["plan_sha256"] == r.PLANS["matched"],
            "Completion count/plan differs",
        )
        a.require(
            runtime_done["status"]
            == "128_worker_execution_and_original_audits_complete"
            and runtime_done["original_completion_sha256"] == c.sha(done_path),
            "Runtime completion differs",
        )
        c.verify(runtime_done["provenance_hashes"])
        c.verify(done["review_hashes"])
        cat = r.load_catalog("matched")
        e.check_schedule("matched", schedule, cat)
        groups, _ = m.scoring.scientific_rosters("matched")
        gold_by_key = {row["id"]: row for values in groups.values() for row in values}
        a.require(len(gold_by_key) == 941, "Input roster overlaps")
        phase_by_key = {
            row["id"]: phase for phase, values in groups.items() for row in values
        }
        engine = b.isolated.adapter(c.ROOT / "config/taxonomy.yaml")
        immutable = dict(c.read(b.PLAN)["input_hashes"])
        immutable.update(c.read(b.PLAN)["source_hashes"])
        immutable.update(
            m.bindings(
                [
                    b.PLAN,
                    b.BASE / "ready.json",
                    done_path,
                    runtime_path,
                    r.MANIFEST,
                    m.PLAN,
                ]
            )
        )
        immutable.update(runtime_done["provenance_hashes"])
        immutable.update(done["review_hashes"])
        m.save(m.BASE / "preservation_before.json", immutable)
        counters, accounting, usage, errors = Counter(), Counter(), Counter(), Counter()
        item_hashes = {}
        stages = []
        all_ids = set()
        records = []
        nulls = []
        seen = set()
        start_time = time.perf_counter()
        for stage in schedule["stages"]:
            root = a.storage_path(b.BASE / "live/matched")
            target = root / "stages" / stage["id"]
            a.require(not list(target.glob("stop_*.json")), "Unreviewed stage stop")
            review = b.verify_review(stage, root, r.PLANS["matched"], "live")
            report = a.read(target / review["report"])
            complete = a.read(target / "complete.json")
            ids = a.read(target / review["response_ids"])
            a.require(
                len(ids) == len(set(ids)) and not all_ids.intersection(ids),
                "Duplicate remote response ID",
            )
            all_ids.update(ids)
            stage_counts, stage_accounts, stage_usage = Counter(), Counter(), Counter()
            attempts = []
            for attempt in sorted(target.glob("attempt_*")):
                start = a.read(attempt / "start.json")
                summary = a.read(attempt / "summary.json")
                a.require(
                    not summary["errors"] and not summary["stopped"],
                    "Failed attempt needs review",
                )
                attempts.append(
                    dict(
                        path=r.rel(attempt),
                        start=start["started_utc"],
                        end=summary["finished_utc"],
                        peak_workers=summary["peak_submitted"],
                        model_check=a.read(attempt / "model_check_response.json")[
                            "elapsed_s"
                        ],
                    )
                )
            for j in stage["jobs"]:
                identity = b.identity(j, r.PLANS["matched"])
                ident = j["key"], j["condition"], j["repeat"]
                a.require(ident not in seen, "Repeated analysis output")
                seen.add(ident)
                folder = b.item_dir(root, j, r.PLANS["matched"])
                receipt_path, result_path = (
                    folder / "complete.json",
                    folder / "result.json",
                )
                receipt_sha = c.sha(receipt_path)
                a.require(
                    complete["item_receipt_hashes"][
                        receipt_path.relative_to(root).as_posix()
                    ]
                    == receipt_sha,
                    "Item receipt changed",
                )
                receipt = a.read(receipt_path)
                result_sha = c.sha(result_path)
                a.require(
                    receipt["hashes"][result_path.relative_to(root).as_posix()]
                    == result_sha,
                    "Result changed after audit",
                )
                value = a.read(result_path)
                a.require(
                    value["identity"] == identity
                    and value["messages_sha256"] == j["messages_sha256"]
                    and value["execution_mode"] == "live",
                    "Scored identity/prompt/mode differs",
                )
                item_hashes[r.rel(receipt_path)] = receipt_sha
                item_hashes[r.rel(result_path)] = result_sha
                parsed_votes = []
                failure_errors = []
                a.require(
                    len(value["raw_runs"]) == len(value["run_predictions"]) == 3,
                    "Expected exactly three saved votes",
                )
                for raw_run, saved_vote in zip(
                    value["raw_runs"], value["run_predictions"]
                ):
                    last = None
                    for turn, raw in enumerate(raw_run):
                        parsed, normalizations, error = (
                            engine.bundle.parser.parse_response(raw)
                        )
                        if error:
                            failure_errors.append(error)
                        last = parsed.model_dump() if parsed is not None else None
                        a.require(
                            parsed is None or turn == len(raw_run) - 1,
                            "Response followed a valid vote",
                        )
                    a.require(
                        m.gold.decision(last) == m.gold.decision(saved_vote),
                        "Saved vote differs from text replay",
                    )
                    parsed_votes.append(m.gold.decision(last))
                a.require(
                    len(parsed_votes) == 3
                    and sum(v is not None for v in parsed_votes) == value["n_valid"],
                    "Vote roster/validity differs",
                )
                result = m.gold.decision(value["result"])
                a.require(
                    m.gold.independent_aggregate(parsed_votes) == result,
                    "Independent majority/gate vote differs",
                )
                resource = Counter()
                response_classes = Counter()
                repairs_usage = Counter()
                for event in value["events"]:
                    response_classes[event["response_class"]] += 1
                    if event["action"] == "ok":
                        u = event["model_response_metadata"]["usage"]
                        a.require(
                            u["prompt_tokens"] + u["completion_tokens"]
                            == u["total_tokens"],
                            "Invalid observed usage",
                        )
                        resource.update(
                            {
                                k: u[k]
                                for k in (
                                    "prompt_tokens",
                                    "completion_tokens",
                                    "total_tokens",
                                )
                            }
                        )
                        if event["turn"] > 0:
                            repairs_usage.update(
                                {
                                    k: u[k]
                                    for k in (
                                        "prompt_tokens",
                                        "completion_tokens",
                                        "total_tokens",
                                    )
                                }
                            )
                row = dict(
                    key=j["key"],
                    phase=phase_by_key[j["key"]],
                    source=gold_by_key[j["key"]]["source"],
                    condition=j["condition"],
                    repeat=j["repeat"],
                    result=result,
                    votes=parsed_votes,
                    n_valid=value["n_valid"],
                    uncertain=value["uncertain"],
                    repairs=value["repairs"],
                    parse_failures=value["parse_failures"],
                    gate_normalised=value["gate_normalised"],
                    normalisations=value["normalisations"],
                    messages_sha256=j["messages_sha256"],
                    accounting=value["accounting"],
                    usage=dict(resource),
                    repair_usage=dict(repairs_usage),
                    response_classes=dict(response_classes),
                    validation_errors=dict(Counter(failure_errors)),
                    request_seconds=sum(x["elapsed_s"] for x in value["events"]),
                    backoff_seconds=sum(
                        x["backoff_elapsed_s"] for x in value["events"]
                    ),
                    stage=j["study"] + "/" + stage["id"],
                    result_path=r.rel(result_path),
                    result_sha256=result_sha,
                )
                records.append(row)
                stage_counts.update(
                    predictions=1,
                    nulls=int(result is None),
                    gate_ties=int(value["uncertain"]),
                )
                stage_counts["valid_votes_" + str(value["n_valid"])] += 1
                stage_counts.update(
                    {
                        k: value[k]
                        for k in (
                            "repairs",
                            "parse_failures",
                            "gate_normalised",
                            "normalisations",
                        )
                    }
                )
                stage_accounts.update(value["accounting"])
                stage_usage.update(resource)
                errors.update(failure_errors)
                if result is None:
                    nulls.append(
                        {
                            k: row[k]
                            for k in (
                                "key",
                                "phase",
                                "condition",
                                "repeat",
                                "validation_errors",
                            )
                        }
                    )
                if len(records) % 2000 == 0:
                    print(
                        "Verified and parsed",
                        len(records),
                        "/ 36699 saved results",
                        flush=True,
                    )
            for name in report["counts"]:
                a.require(
                    stage_counts[name] == report["counts"][name],
                    "Stage count mismatch: " + name,
                )
            a.require(
                dict(stage_accounts) == report["accounting"]
                and dict(stage_usage) == report["observed_usage"],
                "Stage resource/accounting mismatch",
            )
            counters.update(stage_counts)
            accounting.update(stage_accounts)
            usage.update(stage_usage)
            immutable.update(
                m.bindings(
                    [
                        target / "complete.json",
                        target / "review.json",
                        target / review["report"],
                        target / review["response_ids"],
                        target / review["raw_hashes"],
                    ]
                )
            )
            stages.append(
                dict(
                    id=stage["id"],
                    population=stage["population"],
                    counts=report["counts"],
                    accounting=report["accounting"],
                    usage=report["observed_usage"],
                    attempts=attempts,
                    audit_utc=report["created_utc"],
                    generation_end=complete["completed_utc"],
                    vote_seconds=report["vote_seconds_sum"],
                    http_statuses=report["http_statuses"],
                    validation_errors=report["validation_errors"],
                )
            )
        a.require(
            len(seen) == len(records) == 36699 and len(all_ids) == 118018,
            "Incomplete matrix or remote IDs",
        )
        a.require(
            dict(counters) == done["counts"]
            and dict(accounting) == done["accounting"]
            and dict(usage) == done["observed_usage"],
            "Whole-study accounting differs",
        )
        a.require(
            sum(errors.values()) == counters["repairs"] + counters["parse_failures"],
            "Parse error accounting differs",
        )
        m.jsonlines(m.BASE / "predictions.jsonl", records)
        m.jsonlines(
            m.BASE / "gold_rosters.jsonl",
            [
                dict(row, phase=phase)
                for phase, values in groups.items()
                for row in values
            ],
        )
        m.save(m.BASE / "item_bindings.json", item_hashes)
        m.save(
            m.BASE / "execution_review.json",
            dict(
                counts=dict(counters),
                accounting=dict(accounting),
                observed_usage=dict(usage),
                stages=stages,
                null_outputs=nulls,
                validation_errors=dict(errors),
                final_file_mtime_utc=datetime.fromtimestamp(
                    done_path.stat().st_mtime, timezone.utc
                ).isoformat(),
                final_file_time_is_proxy=True,
                complete_response_ids=len(all_ids),
                original_audits_reused=True,
                new_full_raw_HTTP_replay=False,
                saved_generated_texts_reparsed=sum(
                    x["http_success_with_text"] for x in [accounting]
                ),
                elapsed_reader_seconds=time.perf_counter() - start_time,
            ),
        )
        m.save(m.BASE / "preservation_inputs.json", immutable)
        c.verify(immutable)
        m.save(
            m.BASE / "assembly.json",
            dict(
                status="complete_audited_receipt_reader_and_text_replay_pass",
                outputs=m.bindings(
                    [
                        m.BASE / p
                        for p in [
                            "predictions.jsonl",
                            "gold_rosters.jsonl",
                            "item_bindings.json",
                            "execution_review.json",
                            "preservation_inputs.json",
                        ]
                    ]
                ),
                sources=m.bindings(
                    [Path(__file__), Path(m.__file__), Path(m.scoring.__file__), m.PLAN]
                ),
                actual_Qwen_calls=0,
                server_accesses=0,
                training_fits=0,
            ),
        )
        print("Reader complete:", dict(counters), flush=True)


if __name__ == "__main__":
    assemble()

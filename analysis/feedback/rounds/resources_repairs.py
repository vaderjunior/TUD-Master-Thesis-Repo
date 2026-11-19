from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import math
from analysis.feedback.rounds import resources_inputs as a
from analysis.feedback.reserved import resources_metrics as core


def inspect(row):
    path = a.ROOT / Path(row["item_receipt"]).with_name("result.json")
    a.need(a.sha(path) == row["result_sha256"], "Targeted original result changed")
    q = a.read(path)
    a.need(
        q["n_valid"] == row["n_valid"]
        and q["repairs"] == row["repairs"]
        and q["accounting"] == row["accounting"],
        "Targeted result identity differs",
    )
    records = []
    failures = []
    for vote in range(3):
        events = [e for e in q["events"] if e["vote"] == vote and e["action"] == "ok"]
        turns = [e["turn"] for e in events]
        a.need(
            turns == list(range(len(turns))) and len(turns) <= 3,
            "Successful logical turns differ",
        )
        if len(turns) == 1:
            continue
        valid = q["run_predictions"][vote] is not None
        depth = len(turns) - 1
        category = "recovered_after_" + str(depth) if valid else "terminal_invalid"
        a.need(valid or depth == 2, "Invalid vote did not exhaust repairs")
        costs = []
        for event in events:
            usage = {
                k: event["model_response_metadata"]["usage"][k]
                for k in ["prompt_tokens", "completion_tokens", "total_tokens"]
            }
            core.usage_check(usage)
            costs.append(usage)
        records.append(
            dict(
                **{
                    k: row[k]
                    for k in [
                        "phase",
                        "key",
                        "role",
                        "population",
                        "round",
                        "policy",
                        "stream",
                        "condition",
                        "repeat",
                    ]
                },
                vote=vote,
                repair_depth=depth,
                category=category,
                final_schema_valid=valid,
                usage_by_turn=costs,
                result_path=a.rel(path),
                result_sha256=row["result_sha256"],
            )
        )
    for event in q["events"]:
        if event["action"] != "ok":
            failures.append(
                dict(
                    phase=row["phase"],
                    key=row["key"],
                    condition=row["condition"],
                    repeat=row["repeat"],
                    vote=event["vote"],
                    turn=event["turn"],
                    response_class=event["response_class"],
                    request_seconds=event["elapsed_s"],
                    backoff_seconds=event["backoff_elapsed_s"],
                    result_path=a.rel(path),
                    result_sha256=row["result_sha256"],
                )
            )
    a.need(
        sum(r["repair_depth"] for r in records) == row["repairs"]
        and len(failures) == row["accounting"]["retry_attempts"],
        "Targeted costs do not reconcile",
    )
    return records, failures


def main():
    a.need(not (a.OUT / "repair_depth.json").exists(), "Preserve repair-depth review")
    items = a.read(a.OUT / "targeted_result_inventory.json")
    groups = a.read(a.OUT / "resource_groups.json")
    a.write(
        a.OUT / "repair_contract.json",
        dict(
            created_utc=a.pure.utc(),
            source_sha256=a.sha(Path(__file__)),
            inputs=a.bindings(
                [a.OUT / "assembled.json", a.OUT / "targeted_result_inventory.json"]
            ),
            scope="Read only outputs with repairs or extra transport attempts. Report schema recovery and successful-response usage by turn; validity is not correctness. No reparse, request, imputed usage or changed prediction.",
            new_Qwen_calls=0,
        ),
    )
    summary = defaultdict(Counter)
    by_role = defaultdict(Counter)
    failures = []
    bindings = {}
    votes = 0
    outputs = 0
    with a.gzopen(a.OUT / "repair_vote_costs.jsonl.gz") as out, ThreadPoolExecutor(
        max_workers=6
    ) as executor:
        for i, (row, (records, failed)) in enumerate(
            zip(items, executor.map(inspect, items)), 1
        ):
            path = a.ROOT / Path(row["item_receipt"]).with_name("result.json")
            bindings[a.rel(path)] = row["result_sha256"]
            outputs += bool(row["repairs"])
            failures.extend(failed)
            for r in records:
                a.emit(out, r)
                votes += 1
                costs = r["usage_by_turn"]
                category = r["category"]
                for turn, cost in enumerate(costs):
                    summary["all_turn_" + str(turn)].update(calls=1, **cost)
                    summary[category + "/turn_" + str(turn)].update(calls=1, **cost)
                summary["category/" + category].update(
                    votes=1,
                    repair_calls=r["repair_depth"],
                    ordinary_tokens=costs[0]["total_tokens"],
                    repair_tokens=sum(c["total_tokens"] for c in costs[1:]),
                    total_tokens=sum(c["total_tokens"] for c in costs),
                )
                by_role[r["role"] + "/" + category].update(
                    votes=1,
                    repair_calls=r["repair_depth"],
                    repair_tokens=sum(c["total_tokens"] for c in costs[1:]),
                )
            if i % 1000 == 0:
                print(f"Targeted repair/failure results {i}/{len(items)}", flush=True)
    overall = groups["all"]
    first = summary["all_turn_1"]["calls"]
    second = summary["all_turn_2"]["calls"]
    one = summary["category/recovered_after_1"]["votes"]
    two = summary["category/recovered_after_2"]["votes"]
    bad = summary["category/terminal_invalid"]["votes"]
    a.need(
        outputs == overall["counts"]["repair_used"]
        and first + second == 50458
        and bad == 8220
        and first == one + second
        and second == two + bad
        and first == votes,
        "Repair transitions do not reconcile",
    )
    for field in ["prompt_tokens", "completion_tokens", "total_tokens"]:
        a.need(
            sum(summary["all_turn_" + str(t)][field] for t in [1, 2])
            == overall["repair_usage"][field],
            "Repair token total differs",
        )
    failuregroups = {}
    for cls in sorted({r["response_class"] for r in failures}):
        rs = [r for r in failures if r["response_class"] == cls]
        ds = [r["request_seconds"] for r in rs]
        failuregroups[cls] = dict(
            events=len(rs),
            request_seconds=math.fsum(ds),
            backoff_seconds=math.fsum(r["backoff_seconds"] for r in rs),
            p50_seconds=core.quantile(ds, 0.5),
            p95_seconds=core.quantile(ds, 0.95),
            max_seconds=max(ds),
        )
    a.need(
        len(failures) == overall["accounting"]["retry_attempts"] == 669,
        "Retry events differ",
    )
    a.write(
        a.OUT / "failure_costs.json",
        dict(
            groups=failuregroups,
            events=failures,
            qualification="Client-side recorded failures do not measure any remote generation/token work. These events are already included in total POST attempts and durations.",
        ),
    )
    a.write(
        a.OUT / "repair_depth.json",
        dict(
            status="part8_targeted_repair_and_failure_costs_complete",
            created_utc=a.pure.utc(),
            source_sha256=a.sha(Path(__file__)),
            original_result_bindings=bindings,
            outputs_repaired=outputs,
            targeted_results=len(items),
            repaired_votes=votes,
            first_repairs=first,
            second_repairs=second,
            recovered_after_first=one,
            recovered_after_second=two,
            terminal_invalid_votes=bad,
            groups=summary,
            role_groups=by_role,
            ledger=a.rel(a.OUT / "repair_vote_costs.jsonl.gz"),
            ledger_sha256=a.sha(a.OUT / "repair_vote_costs.jsonl.gz"),
            qualification="Recovery means the saved vote passes the frozen schema, not that its label or complete answer is correct. All costs are subsets of the existing study, not new experiments.",
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    a.write(
        a.OUT / "repairs_complete.json",
        dict(
            status="part8_targeted_costs_complete",
            created_utc=a.pure.utc(),
            inputs=a.bindings(
                [
                    a.OUT / "assembled.json",
                    a.OUT / "targeted_result_inventory.json",
                    a.OUT / "resource_groups.json",
                ]
            ),
            source_sha256=a.sha(Path(__file__)),
            artifacts=a.bindings(
                [
                    a.OUT / p
                    for p in [
                        "repair_contract.json",
                        "repair_vote_costs.jsonl.gz",
                        "repair_depth.json",
                        "failure_costs.json",
                    ]
                ]
            ),
            new_Qwen_calls=0,
        ),
    )
    print(
        f"Repairs reconciled: {votes} initially invalid votes, {one+two} recovered, {bad} terminal-invalid.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()

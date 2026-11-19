from collections import Counter, defaultdict
from pathlib import Path
import math
from analysis.feedback.transfer import resources as p

a = p.a
OUT = p.OUT


def main():
    a.need(
        (OUT / "assembled.json").exists() and not (OUT / "repair_depth.json").exists(),
        "New targeted repair review only",
    )
    p.save(
        "repair_depth_contract.json",
        dict(
            created_utc=a.utc(),
            script_sha256=a.sha(Path(__file__)),
            scope="Read only already audited outputs that used a repair; count actual first/second repair turns, successful-response usage and final schema validity. No parsing change, replay, request or counterfactual accuracy claim. Costs remain subsets of the existing run.",
            input_binding=dict(
                path=a.rel(OUT / "assembled.json"), sha256=a.sha(OUT / "assembled.json")
            ),
        ),
    )
    groups = defaultdict(Counter)
    bindings = {}
    seen = set()
    output_count = 0
    rows = 0
    with p.gzopen("repair_vote_costs.jsonl.gz") as out:
        for file in sorted((a.OUT / "assembly").glob("transfer_*.jsonl")):
            for row in a.rows(file):
                if not row["repairs"]:
                    continue
                path = a.ROOT / Path(row["item_receipt"]).with_name("result.json")
                q = a.read_bound(path, row["result_sha256"])
                bindings[a.rel(path)] = row["result_sha256"]
                a.need(
                    q["n_valid"] == row["n_valid"] and q["repairs"] == row["repairs"],
                    "Original repair output identity",
                )
                output_count += 1
                for vote in range(3):
                    events = [
                        e
                        for e in q["events"]
                        if e["vote"] == vote and e["action"] == "ok"
                    ]
                    turns = [e["turn"] for e in events]
                    a.need(
                        turns == list(range(len(turns))) and len(turns) <= 3,
                        "Bounded successful logical repair chain",
                    )
                    if len(turns) == 1:
                        continue
                    final_valid = q["run_predictions"][vote] is not None
                    depth = len(turns) - 1
                    category = (
                        "recovered_after_" + str(depth)
                        if final_valid
                        else "terminal_invalid"
                    )
                    a.need(
                        final_valid or depth == 2,
                        "Terminal invalid exhausts both repairs",
                    )
                    identity = (
                        row["input_index"],
                        row["condition"],
                        row["repeat"],
                        vote,
                    )
                    a.need(identity not in seen, "Unique repaired vote")
                    seen.add(identity)
                    costs = []
                    for event in events:
                        usage = event["model_response_metadata"]["usage"]
                        a.need(
                            usage["total_tokens"]
                            == usage["prompt_tokens"] + usage["completion_tokens"],
                            "Repair-depth observed usage",
                        )
                        costs.append(
                            {
                                key: usage[key]
                                for key in (
                                    "prompt_tokens",
                                    "completion_tokens",
                                    "total_tokens",
                                )
                            }
                        )
                        for name in (
                            "all_turn_" + str(event["turn"]),
                            category + "/turn_" + str(event["turn"]),
                        ):
                            groups[name].update(calls=1, **costs[-1])
                    record = dict(
                        input_index=row["input_index"],
                        key=row["key"],
                        population=row["population"],
                        condition=row["condition"],
                        repeat=row["repeat"],
                        vote=vote,
                        repair_depth=depth,
                        category=category,
                        final_schema_valid=final_valid,
                        usage_by_turn=costs,
                        result_path=a.rel(path),
                        result_sha256=row["result_sha256"],
                    )
                    p.emit(out, record)
                    rows += 1
                    groups["category/" + category].update(
                        votes=1,
                        repair_calls=depth,
                        ordinary_tokens=costs[0]["total_tokens"],
                        repair_tokens=sum(u["total_tokens"] for u in costs[1:]),
                        total_tokens=sum(u["total_tokens"] for u in costs),
                    )
            print("Repair-depth review " + file.stem, flush=True)
    existing = a.read(OUT / "resource_groups.json")["all"]
    a.need(
        output_count == existing["counts"]["repair_used"] == 14056,
        "Every repaired output",
    )
    a.need(
        groups["all_turn_1"]["calls"] + groups["all_turn_2"]["calls"]
        == existing["accounting"]["repair_logical_calls"],
        "Repair logical turns reconcile",
    )
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        a.need(
            groups["all_turn_1"][key] + groups["all_turn_2"][key]
            == existing["repair_usage"][key],
            "Every reported repair token reconciles",
        )
    a.need(
        groups["category/terminal_invalid"]["votes"]
        == existing["counts"]["parse_failures"]
        == 6803,
        "Terminal invalid votes match raw-verified Part 7",
    )

    actual = defaultdict(Counter)
    seen_again = set()
    for row in p.previous.gzrows(OUT / "repair_vote_costs.jsonl.gz"):
        ident = (row["input_index"], row["condition"], row["repeat"], row["vote"])
        a.need(ident not in seen_again, "Ledger unique vote")
        seen_again.add(ident)
        cat = row["category"]
        usage = row["usage_by_turn"]
        for t, u in enumerate(usage):
            for name in ("all_turn_" + str(t), cat + "/turn_" + str(t)):
                actual[name]["calls"] += 1
                for key, value in u.items():
                    actual[name][key] += value
        actual["category/" + cat]["votes"] += 1
        actual["category/" + cat]["repair_calls"] += len(usage) - 1
        actual["category/" + cat]["ordinary_tokens"] += usage[0]["total_tokens"]
        actual["category/" + cat]["repair_tokens"] += sum(
            u["total_tokens"] for u in usage[1:]
        )
        actual["category/" + cat]["total_tokens"] += sum(
            u["total_tokens"] for u in usage
        )
    a.need(
        actual == groups and seen_again == seen and rows == len(seen),
        "Independent repair ledger regrouping",
    )
    first = groups["all_turn_1"]["calls"]
    second = groups["all_turn_2"]["calls"]
    recovered_first = groups["category/recovered_after_1"]["votes"]
    recovered_second = groups["category/recovered_after_2"]["votes"]
    a.need(
        first == recovered_first + second and second == recovered_second + 6803,
        "Actual repair transitions",
    )
    p.save(
        "repair_depth.json",
        dict(
            status="targeted_original_repair_turn_costs_verified",
            created_utc=a.utc(),
            script_sha256=a.sha(Path(__file__)),
            contract_sha256=a.sha(OUT / "repair_depth_contract.json"),
            original_result_bindings=bindings,
            output_records=output_count,
            repaired_votes=rows,
            first_repairs=first,
            second_repairs=second,
            recovered_after_first=recovered_first,
            recovered_after_second=recovered_second,
            terminal_invalid_votes=6803,
            groups=groups,
            ledger=a.rel(OUT / "repair_vote_costs.jsonl.gz"),
            ledger_sha256=a.sha(OUT / "repair_vote_costs.jsonl.gz"),
            independent_ledger_regrouping=True,
            qualification="Recovered means the final saved vote passed the frozen schema. It does not mean a correct hate label, an improved complete answer, or that a new repair policy was evaluated. All ordinary tokens shown here belong to the repaired-vote subset; all repair tokens reconcile exactly to the full scientific run.",
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print("Targeted repair review and exact token reconciliation passed.", flush=True)


if __name__ == "__main__":
    with a.offline():
        main()

from collections import Counter
from analysis.common import feedback_inputs as c

SEEDS = [2026100441, 2026100442, 2026100443]


def classify_candidates(roster, predictions):

    expected = {(r["id"], rep) for r in roster for rep in range(3)}
    c.require(set(predictions) == expected, "Wrong source-selection response matrix")
    output = {}
    for r in roster:
        values = [predictions[r["id"], rep] for rep in range(3)]
        c.require(type(r["gold"]["hate"]) is bool, "Binary source gold required")
        if any(v is None for v in values):
            output[r["id"]] = {
                "status": "invalid",
                "hate": r["gold"]["hate"],
                "decisions": [None if v is None else v["hate"] for v in values],
            }
            continue
        c.require(
            all(type(v.get("hate")) is bool for v in values), "Invalid gate decision"
        )
        decisions = [v["hate"] for v in values]
        majority = sum(decisions) >= 2
        output[r["id"]] = {
            "status": "correct" if majority == r["gold"]["hate"] else "error",
            "hate": r["gold"]["hate"],
            "decisions": decisions,
            "majority_hate": majority,
        }
    return output


def select_pools(statuses):
    buckets = {
        (gate, status): sorted(
            i
            for i, r in statuses.items()
            if r["hate"] == gate and r["status"] == status
        )
        for gate in (False, True)
        for status in ("correct", "error")
    }
    capacity = min(map(len, buckets.values()))
    budget = next((n for n in (40, 20) if capacity >= n), None)
    if budget is None:
        return {
            "status": "insufficient_predeclared_candidate_supply",
            "capacity": capacity,
            "counts": {
                f"{gate}_{status}": len(v) for (gate, status), v in buckets.items()
            },
            "pools": [],
        }
    pools = []
    for seed in SEEDS:
        for selection in ("error", "correct", "random"):

            def ordered(gate):
                eligible = [
                    i
                    for i, r in statuses.items()
                    if r["hate"] == gate
                    and r["status"] != "invalid"
                    and (selection == "random" or r["status"] == selection)
                ]
                return sorted(
                    eligible,
                    key=lambda i: c.source_order(f"{seed}/{selection}/{gate}", i),
                )

            positive = ordered(True)[:budget]
            benign = ordered(False)[: budget // 2]
            c.require(
                len(positive) == budget and len(benign) == budget // 2,
                "Pool supply differs",
            )
            for polarity, ids in [
                ("positive", positive),
                ("balanced", positive[: budget // 2] + benign),
            ]:
                pools.append(
                    {
                        "condition": f"{selection}_{polarity}_s{seed}",
                        "selection": selection,
                        "polarity": polarity,
                        "seed": seed,
                        "budget": budget,
                        "ids": ids,
                        "actual_status_counts": dict(
                            Counter(statuses[i]["status"] for i in ids)
                        ),
                    }
                )
    return {
        "status": "pools_selected_by_frozen_rule",
        "capacity": capacity,
        "budget": budget,
        "counts": {f"{gate}_{status}": len(v) for (gate, status), v in buckets.items()},
        "pools": pools,
        "distinct_sources": len({i for p in pools for i in p["ids"]}),
        "seeds": SEEDS,
    }


def gate_only_record(source):
    c.require(
        type(source["gold"]["hate"]) is bool, "Known binary correction label required"
    )
    return {
        "id": "sq3benign-" + source["id"],
        "kind": "feedback",
        "lang": "en",
        "source": source["source"],
        "text": source["text"],
        "meta": {
            "gate": source["gold"]["hate"],
            "hate_types": None,
            "target_groups": None,
            "severity": None,
            "legal": None,
        },
        "provenance": {
            "source_id": source["id"],
            "annotation_scope": "binary gate only; all other heads unannotated",
        },
    }

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import ast
import hashlib
import json
import re
from analysis.feedback.rounds import robustness_inputs as a


def describe(v):
    error = str(v["error"])
    match = re.search(r" at (\(.*\))$", error)
    path = None
    if match:
        try:
            path = ast.literal_eval(match[1])
        except (ValueError, SyntaxError):
            pass
    raw = v["raw_outputs"]
    parsed = None
    strict = False
    scalar = None
    try:
        parsed = json.loads(raw[-1])
        strict = True
    except (ValueError, TypeError, IndexError):
        pass
    if strict and isinstance(path, tuple):
        try:
            scalar = parsed
            for key in path:
                scalar = scalar[key]
        except (KeyError, IndexError, TypeError):
            scalar = None
    if scalar is not None and (
        not isinstance(scalar, (str, int, float, bool)) or len(str(scalar)) > 120
    ):
        scalar = "<noncompact value>"
    return dict(
        error=error,
        category=(
            "schema_violation"
            if error.startswith("schema violation:")
            else "json_or_other_validation"
        ),
        field=str(path[0]) if isinstance(path, tuple) and path else "unresolved",
        path=list(path) if isinstance(path, tuple) else None,
        strict_json_last_output=strict,
        rejected_scalar=scalar,
        raw_turns=len(raw),
        distinct_texts=len(set(raw)),
        identical_repair_texts=len(set(raw)) == 1,
        raw_text_sha256=[hashlib.sha256(s.encode("utf8")).hexdigest() for s in raw],
    )


def main():
    proof = a.read(a.OUT / "independent_verification.json")
    a.need(
        proof["status"] == "part7_independent_verification_pass",
        "Verified robustness first",
    )
    a.write(
        a.OUT / "failure_review_contract.json",
        dict(
            status="invalid_vote_review_rules_recorded",
            created_utc=a.pure.utc(),
            timing="After main Part 7 robustness scoring; diagnostic explanation only.",
            scope="All 8220 terminal-invalid votes in all 6584 partial/null final outputs, including source screening. Read audit-bound item/result/vote records only. Reuse full HTTP audits.",
            rule="Describe stored final validation error and field, strict-JSON availability, compact rejected scalar, and exact raw-text repetition over repair turns. No repaired interpretation, taxonomy edit or recomputed scientific answer.",
            unknown="If strict JSON decoding fails, the offending scalar remains unknown. A stop finish reason is not proof of valid JSON or valid schema.",
            verification_sha256=a.sha(a.OUT / "independent_verification.json"),
            source_sha256=a.sha(Path(__file__)),
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    jobs = list(a.rows(a.OUT / "null_outputs.jsonl")) + list(
        a.rows(a.OUT / "partial_vote_outputs.jsonl")
    )
    a.need(len(jobs) == 6584, "Invalid-output inventory size")

    def inspect(row):
        receipt_path = a.ROOT / row["item_receipt"]
        a.need(
            a.sha(receipt_path) == row["item_receipt_sha256"], "Item receipt changed"
        )
        receipt = a.read(receipt_path)
        result_path = receipt_path.with_name("result.json")
        a.need(a.sha(result_path) == row["result_sha256"], "Result changed")
        result = a.read(result_path)
        output = []
        item = receipt_path.parent.name
        for vi, pred in enumerate(result["run_predictions"]):
            if pred is not None:
                continue
            path = receipt_path.parent / "votes" / f"{vi}.json"
            digest = receipt["hashes"][f"items/{item}/votes/{vi}.json"]
            a.need(a.sha(path) == digest, "Invalid-vote record changed")
            v = a.read(path)
            a.need(
                v["result"] is None and v["parse_failure"] is True,
                "Expected a terminal-invalid vote",
            )
            output.append(
                dict(
                    key=row["key"],
                    phase=row["phase"],
                    role=row["role"],
                    round=row["round"],
                    condition=row["condition"],
                    repeat=row["repeat"],
                    vote=vi,
                    null_aggregate=row["n_valid"] == 0,
                    n_valid=row["n_valid"],
                    item_receipt=row["item_receipt"],
                    item_receipt_sha256=row["item_receipt_sha256"],
                    result_sha256=row["result_sha256"],
                    original_vote=a.rel(path),
                    original_vote_sha256=digest,
                    repairs=v["repairs"],
                    **describe(v),
                )
            )
        a.need(len(output) == 3 - row["n_valid"], "Invalid-vote count per item differs")
        return output

    groups = defaultdict(
        lambda: dict(
            votes=0,
            inputs=set(),
            categories=Counter(),
            fields=Counter(),
            errors=Counter(),
            rejected_scalars=Counter(),
            repairs=Counter(),
            turns=Counter(),
            distinct_texts=Counter(),
            identical_repair_texts=0,
            strict_json=0,
        )
    )
    rows = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for j, batch in enumerate(pool.map(inspect, jobs)):
            for r in batch:
                rows.append(r)
                names = [
                    "all",
                    "role/" + r["role"],
                    "phase/" + r["phase"],
                    "aggregate/" + ("null" if r["null_aggregate"] else "partial"),
                ]
                for name in names:
                    g = groups[name]
                    g["votes"] += 1
                    g["inputs"].add(r["key"])
                    g["categories"][r["category"]] += 1
                    g["fields"][r["field"]] += 1
                    g["errors"][r["error"]] += 1
                    if r["rejected_scalar"] is not None:
                        g["rejected_scalars"][
                            json.dumps(
                                r["rejected_scalar"], sort_keys=True, ensure_ascii=False
                            )
                        ] += 1
                    for f, rf in [
                        ("repairs", "repairs"),
                        ("turns", "raw_turns"),
                        ("distinct_texts", "distinct_texts"),
                    ]:
                        g[f][str(r[rf])] += 1
                    g["identical_repair_texts"] += r["identical_repair_texts"]
                    g["strict_json"] += r["strict_json_last_output"]
            if (j + 1) % 1000 == 0:
                print(f"Bound invalid-output details: {j+1}/{len(jobs)}", flush=True)
    a.need(
        len(rows) == 8220
        and groups["aggregate/null"]["votes"] == 726
        and groups["role/evaluation"]["votes"] == 8177,
        "Invalid-vote totals differ",
    )
    for g in groups.values():
        g["inputs"] = len(g["inputs"])
    a.lines(a.OUT / "invalid_vote_details.jsonl", rows)
    a.write(a.OUT / "failure_review.json", groups)
    a.write(
        a.OUT / "failure_review_complete.json",
        dict(
            status="all_invalid_votes_inspected_pending_independent_check",
            created_utc=a.pure.utc(),
            source_sha256=a.sha(Path(__file__)),
            inputs=a.bindings(
                [
                    a.OUT / "independent_verification.json",
                    a.OUT / "null_outputs.jsonl",
                    a.OUT / "partial_vote_outputs.jsonl",
                ]
            ),
            artifacts=a.bindings(
                [
                    a.OUT / f
                    for f in [
                        "failure_review_contract.json",
                        "invalid_vote_details.jsonl",
                        "failure_review.json",
                    ]
                ]
            ),
            item_receipts=6584,
            invalid_votes=8220,
            null_aggregate_votes=726,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        "All terminal-invalid votes reviewed; all saved answers unchanged.", flush=True
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()

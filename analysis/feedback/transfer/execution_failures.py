from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import ast
import hashlib
import json
import re
import numpy as np
from analysis.feedback.transfer import execution as p

a = p.a
OUT = p.OUT


def details(vote):
    error = vote["error"]
    text = error if isinstance(error, str) else str(error)
    match = re.search(r" at (\(.*\))$", text)
    path = None
    if match:
        try:
            path = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError):
            pass
    field = str(path[0]) if isinstance(path, tuple) and path else "unresolved"
    category = (
        "schema_violation"
        if text.startswith("schema violation:")
        else "json_or_other_validation"
    )
    raw = vote["raw_outputs"]
    same = len(set(raw)) == 1
    last = raw[-1] if raw else ""
    parsed = None
    parsed_ok = False
    try:
        parsed = json.loads(last)
        parsed_ok = True
    except (ValueError, TypeError):
        pass
    rejected = None
    if parsed_ok and isinstance(path, tuple):
        try:
            rejected = parsed
            for key in path:
                rejected = rejected[key]
        except (KeyError, TypeError, IndexError):
            rejected = None

    if rejected is not None and (
        not isinstance(rejected, (str, int, float, bool)) or len(str(rejected)) > 120
    ):
        rejected = "<noncompact value>"
    return dict(
        error=text,
        error_category=category,
        error_field=field,
        error_path=list(path) if isinstance(path, tuple) else None,
        raw_turns=len(raw),
        distinct_raw_texts=len(set(raw)),
        all_repair_texts_identical=same,
        strict_json_last_output=parsed_ok,
        rejected_scalar=rejected,
        raw_text_sha256=[hashlib.sha256(s.encode("utf8")).hexdigest() for s in raw],
    )


def main():
    a.need(
        not (OUT / "failure_details_complete.json").exists()
        and not (OUT / "failure_detail_contract.json").exists(),
        "Preserve earlier invalid-vote review",
    )
    verified = a.read(OUT / "verified.json")
    a.need(
        verified["calculated_sha256"] == a.sha(OUT / "calculated.json"),
        "Verified main sensitivity analysis",
    )
    p.save(
        "failure_detail_contract.json",
        dict(
            created_utc=a.utc(),
            scope="All 6803 terminal-invalid votes, including all 828 votes behind the 276 null aggregates. Read bound vote receipts only, reuse original raw audit, no full replay.",
            analysis="Summarize stored final error strings, schema field paths, exact repetition of saved raw repair texts and compact offending scalar labels where the final raw text is strict JSON. This is a diagnostic, not an alternative parser or rescoring.",
            json_qualification="If strict json.loads cannot parse the last raw text, its rejected value stays unknown. Never infer a recovered valid prediction or edit the taxonomy from these diagnostics.",
            verified_sha256=a.sha(OUT / "verified.json"),
            script_sha256=a.sha(Path(__file__)),
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    with np.load(p.P6 / "arrays.npz", allow_pickle=False) as f:
        raw = f["votes"][..., 0]
    meta = list(a.rows(p.P4 / "metadata.jsonl"))
    jobs = [q for q in p.gzrows(OUT / "output_execution.jsonl.gz") if q["n_valid"] < 3]
    a.need(len(jobs) == 5188, "Under-three-output count")

    def read_item(row):
        receipt_path = a.ROOT / row["item_receipt"]
        receipt = a.read_bound(receipt_path, row["item_receipt_sha256"])
        i = row["input_index"]
        c = p.CONDITIONS.index(row["condition"])
        t = row["repeat"]
        out = []
        for v in range(3):
            if raw[i, c, t, v] >= 0:
                continue
            path = receipt_path.parent / "votes" / f"{v}.json"
            name = path.relative_to(a.LIVE).as_posix()
            digest = receipt["hashes"][name]
            saved = a.read_bound(path, digest)
            a.need(
                saved["result"] is None and saved["parse_failure"] is True,
                "Invalid-vote diagnostic mismatch",
            )
            out.append(
                dict(
                    input_index=i,
                    key=row["key"],
                    source=meta[i]["source"],
                    population=row["population"],
                    gold_hate=meta[i]["gold_hate"],
                    condition=row["condition"],
                    repeat=t,
                    vote=v,
                    effective_workers=row["effective_workers"],
                    null_aggregate=row["n_valid"] == 0,
                    original_vote=a.rel(path),
                    original_vote_sha256=digest,
                    item_receipt=row["item_receipt"],
                    item_receipt_sha256=row["item_receipt_sha256"],
                    repairs=saved["repairs"],
                    **details(saved),
                )
            )
        a.need(len(out) == 3 - row["n_valid"], "Invalid votes per item")
        return out

    groups = defaultdict(
        lambda: dict(
            invalid_votes=0,
            distinct_inputs=set(),
            error_categories=Counter(),
            error_fields=Counter(),
            error_messages=Counter(),
            rejected_scalars=Counter(),
            repair_counts=Counter(),
            raw_turn_counts=Counter(),
            distinct_raw_text_counts=Counter(),
            all_repair_texts_identical=0,
            strict_json_last_outputs=0,
        )
    )
    rows = 0
    with ThreadPoolExecutor(max_workers=8) as pool, p.gzopen(
        "invalid_vote_details.jsonl.gz"
    ) as f:
        for batch in pool.map(read_item, jobs):
            for row in batch:
                p.emit(f, row)
                rows += 1
                names = [
                    "all",
                    f"workers/{row['effective_workers']}",
                    f"source/{row['source']}",
                    f"population/{row['population']}",
                    f"gold/{int(row['gold_hate'])}",
                ]
                if row["null_aggregate"]:
                    names += [
                        "null_aggregates",
                        f"null_aggregates/workers/{row['effective_workers']}",
                    ]
                for name in names:
                    g = groups[name]
                    g["invalid_votes"] += 1
                    g["distinct_inputs"].add(row["key"])
                    g["error_categories"][row["error_category"]] += 1
                    g["error_fields"][row["error_field"]] += 1
                    g["error_messages"][row["error"]] += 1
                    if row["rejected_scalar"] is not None:
                        g["rejected_scalars"][
                            json.dumps(
                                row["rejected_scalar"],
                                ensure_ascii=False,
                                sort_keys=True,
                            )
                        ] += 1
                    g["repair_counts"][str(row["repairs"])] += 1
                    g["raw_turn_counts"][str(row["raw_turns"])] += 1
                    g["distinct_raw_text_counts"][str(row["distinct_raw_texts"])] += 1
                    g["all_repair_texts_identical"] += int(
                        row["all_repair_texts_identical"]
                    )
                    g["strict_json_last_outputs"] += int(row["strict_json_last_output"])
    a.need(
        rows == 6803 and groups["null_aggregates"]["invalid_votes"] == 828,
        "Complete invalid-vote inventory",
    )
    for g in groups.values():
        g["distinct_inputs"] = len(g["distinct_inputs"])
    p.save("failure_detail_summary.json", groups)
    p.save(
        "failure_details_complete.json",
        dict(
            status="terminal_invalid_vote_diagnostics_complete_independent_check_pending",
            created_utc=a.utc(),
            script_sha256=a.sha(Path(__file__)),
            main_verification_sha256=a.sha(OUT / "verified.json"),
            artifacts={
                a.rel(OUT / n): a.sha(OUT / n)
                for n in (
                    "failure_detail_contract.json",
                    "invalid_vote_details.jsonl.gz",
                    "failure_detail_summary.json",
                )
            },
            invalid_votes=rows,
            invalid_votes_in_null_aggregates=828,
            read_item_receipts=5188,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        "Reviewed all 6,803 bound terminal-invalid votes; original raw outputs and labels unchanged."
    )


if __name__ == "__main__":
    with a.offline():
        main()

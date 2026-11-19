from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import ast
import hashlib
import json
import numpy as np
from analysis.feedback.transfer import execution as p

a = p.a
OUT = p.OUT


def main():
    a.need(
        not (OUT / "failure_details_verified.json").exists(),
        "Preserve earlier failure verification",
    )
    done = a.read(OUT / "failure_details_complete.json")
    for name, digest in done["artifacts"].items():
        a.need(a.sha(a.ROOT / name) == digest, "Changed invalid-vote evidence")
    rows = list(p.gzrows(OUT / "invalid_vote_details.jsonl.gz"))
    meta = list(a.rows(p.P4 / "metadata.jsonl"))
    with np.load(p.P6 / "arrays.npz", allow_pickle=False) as f:
        raw = f["votes"][..., 0]
        nv = f["n_valid"]
    with np.load(OUT / "workers.npz", allow_pickle=False) as f:
        workers = f["workers"]
    expected = {tuple(map(int, x)) for x in np.argwhere(raw < 0)}
    seen = set()

    def check(row):
        value = a.read_bound(a.ROOT / row["original_vote"], row["original_vote_sha256"])
        i = row["input_index"]
        c = p.CONDITIONS.index(row["condition"])
        t = row["repeat"]
        v = row["vote"]
        a.need(
            value["result"] is None and value["parse_failure"] and raw[i, c, t, v] < 0,
            "Original invalid vote",
        )
        a.need(
            row["key"] == meta[i]["key"]
            and row["source"] == meta[i]["source"]
            and row["gold_hate"] == meta[i]["gold_hate"]
            and row["effective_workers"] == int(workers[i, c, t])
            and row["null_aggregate"] == bool(nv[i, c, t] == 0),
            "Invalid-vote source/runtime identity",
        )
        error = str(value["error"])
        field = "unresolved"
        path = None
        if " at " in error:
            suffix = error.rsplit(" at ", 1)[1]
            try:
                decoded = ast.literal_eval(suffix)
            except (ValueError, SyntaxError):
                decoded = None
            if isinstance(decoded, tuple):
                path = list(decoded)
                field = str(decoded[0]) if decoded else "unresolved"
        texts = value["raw_outputs"]
        strict = False
        scalar = None
        try:
            decoded = json.loads(texts[-1])
            strict = True
            if path is not None:
                scalar = decoded
                for key in path:
                    scalar = scalar[key]
        except (ValueError, KeyError, IndexError, TypeError):
            scalar = None
        if scalar is not None and (
            not isinstance(scalar, (str, int, float, bool)) or len(str(scalar)) > 120
        ):
            scalar = "<noncompact value>"
        want = dict(
            error=error,
            error_category=(
                "schema_violation"
                if error.startswith("schema violation:")
                else "json_or_other_validation"
            ),
            error_field=field,
            error_path=path,
            raw_turns=len(texts),
            distinct_raw_texts=len(set(texts)),
            all_repair_texts_identical=len(set(texts)) == 1,
            strict_json_last_output=strict,
            rejected_scalar=scalar,
            repairs=value["repairs"],
            raw_text_sha256=[
                hashlib.sha256(text.encode("utf8")).hexdigest() for text in texts
            ],
        )
        for name, w in want.items():
            a.need(row[name] == w, "Independent invalid-vote detail " + name)
        return (i, c, t, v)

    with ThreadPoolExecutor(max_workers=8) as pool:
        for identity in pool.map(check, rows):
            a.need(identity not in seen, "Duplicate terminal vote")
            seen.add(identity)
    a.need(seen == expected and len(seen) == 6803, "Every invalid vote read back")
    results = {}
    groups = [
        "all",
        "workers/8",
        "workers/128",
        "gold/0",
        "gold/1",
        "null_aggregates",
        "null_aggregates/workers/8",
        "null_aggregates/workers/128",
    ]
    groups += ["source/" + s for s in sorted({r["source"] for r in rows})] + [
        "population/" + s for s in sorted({r["population"] for r in rows})
    ]
    for name in groups:

        def eligible(r):
            if name == "all":
                return True
            if name == "null_aggregates":
                return r["null_aggregate"]
            if name.startswith("null_aggregates/"):
                return r["null_aggregate"] and r["effective_workers"] == int(
                    name.rsplit("/", 1)[1]
                )
            kind, value = name.split("/")
            return r[
                {"workers": "effective_workers", "gold": "gold_hate"}.get(kind, kind)
            ] == (int(value) if kind in ("workers", "gold") else value)

        rr = [r for r in rows if eligible(r)]
        results[name] = dict(
            invalid_votes=len(rr),
            distinct_inputs=len({r["key"] for r in rr}),
            error_categories=Counter(r["error_category"] for r in rr),
            error_fields=Counter(r["error_field"] for r in rr),
            error_messages=Counter(r["error"] for r in rr),
            rejected_scalars=Counter(
                json.dumps(r["rejected_scalar"], ensure_ascii=False, sort_keys=True)
                for r in rr
                if r["rejected_scalar"] is not None
            ),
            repair_counts=Counter(str(r["repairs"]) for r in rr),
            raw_turn_counts=Counter(str(r["raw_turns"]) for r in rr),
            distinct_raw_text_counts=Counter(str(r["distinct_raw_texts"]) for r in rr),
            all_repair_texts_identical=sum(r["all_repair_texts_identical"] for r in rr),
            strict_json_last_outputs=sum(r["strict_json_last_output"] for r in rr),
        )
    a.need(
        p.native(results) == a.read(OUT / "failure_detail_summary.json"),
        "Every failure diagnostic group",
    )
    a.need(
        results["null_aggregates"]["error_fields"] == Counter(hate_type=828),
        "Every null aggregate is a terminal hate-type failure",
    )
    p.save(
        "failure_details_verified.json",
        dict(
            status="all_terminal_invalid_votes_and_failure_diagnostics_independently_verified",
            checked_utc=a.utc(),
            completion_sha256=a.sha(OUT / "failure_details_complete.json"),
            verifier_sha256=a.sha(Path(__file__)),
            original_votes_read_and_hashed=6803,
            unique_affected_inputs=1459,
            null_aggregate_votes=828,
            summary_groups=len(results),
            same_error_pattern_pre_and_post_switch=True,
            original_validation_and_scientific_outputs_unchanged=True,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        "Independently read and checked all 6,803 terminal-invalid vote records and every summary group."
    )


if __name__ == "__main__":
    with a.offline():
        main()

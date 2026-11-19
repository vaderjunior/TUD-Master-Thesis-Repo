from pathlib import Path
import hashlib
import json
from analysis.feedback.rounds import subgroups_inputs as a


def sig(v):
    return hashlib.sha256(
        (
            json.dumps(v, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        ).encode()
    ).hexdigest()


def main():
    proof = a.read(a.OUT / "independent_verification.json")
    a.need(
        proof["status"] == "part5_independent_verification_pass",
        "Verify numerical results first",
    )
    selection = a.read(a.OUT / "case_selection.json")
    indices = {r["input_index"] for r in selection if r["status"] == "selected"}
    matrix = {
        r["input_index"]: r
        for r in a.rows(a.READER / "matrix.jsonl")
        if r["input_index"] in indices
    }
    meta = {
        r["input_index"]: r
        for r in a.rows(a.OUT / "metadata.jsonl")
        if r["input_index"] in indices
    }
    profiles = {
        r["input_index"]: r
        for r in a.rows(a.PART4 / "input_trajectories.jsonl")
        if r["input_index"] in indices
    }
    keys = {r["key"]: i for i, r in matrix.items()}
    roster = {
        r["key"]: r for r in a.rows(a.STUDY / "evaluation.jsonl") if r["key"] in keys
    }
    records = a.read(a.STUDY / "gate_only_records.json")
    allowed = a.read(a.PART3 / "assembled.json")["input_bindings"]
    promptbindings = {
        (r["phase"], r["key"], r["condition"]): r
        for r in a.rows(a.PART3 / "prompt_bindings.jsonl")
        if r["key"] in keys and r["phase"].endswith("_evaluation")
    }
    bindings = {}
    prompts = {}
    saved = {}
    for rnd in range(1, 5):
        path = a.STUDY / f"phases/r{rnd}_evaluation/prompts.jsonl"
        digest = a.sha(path)
        a.need(digest == allowed[a.rel(path)], "Original prompt file changed")
        bindings[a.rel(path)] = digest
        for p in a.rows(path):
            if p["key"] not in keys:
                continue
            bound = promptbindings[f"r{rnd}_evaluation", p["key"], p["condition"]]
            a.need(
                sig(p["messages"]) == p["messages_sha256"] == bound["messages_sha256"]
                and sig(p["context"]) == p["context_sha256"] == bound["context_sha256"],
                "Actual prompt/hash mismatch",
            )
            a.need(
                p["text"] == roster[p["key"]]["text"]
                and p["source_id"] == bound["source_id"],
                "Query or correction differs",
            )
            ident = (p["key"], rnd, p["condition"])
            a.need(ident not in prompts, "Duplicate prompt")
            prompts[ident] = p
    reader = a.read(a.READER / "complete.json")
    for stage in reader["stage_receipts"]:
        if stage["role"] != "evaluation":
            continue
        receipt_path = a.ROOT / stage["receipt"]
        receipt = a.read(receipt_path)
        a.need(a.sha(receipt_path) == stage["receipt_sha256"], "Stage receipt changed")
        path = a.ROOT / stage["predictions"]
        digest = a.sha(path)
        a.need(digest == receipt["predictions_sha256"], "Audited output changed")
        bindings[a.rel(path)] = digest
        bindings[a.rel(receipt_path)] = a.sha(receipt_path)
        for r in a.rows(path):
            if r["key"] not in keys:
                continue
            rnd = int(r["phase"][1])
            col = a.CONDITIONS.index(r["condition"])
            rep = r["repeat"]
            i = keys[r["key"]]
            expected = matrix[i]["predictions"][rnd - 1][col][rep]
            a.need(
                all(r[f] == expected[f] for f in ["result", "n_valid", "uncertain"]),
                "Case result differs from verified matrix",
            )
            a.need(
                r["messages_sha256"]
                == prompts[r["key"], rnd, r["condition"]]["messages_sha256"],
                "Executed message binding differs",
            )
            ident = (r["key"], rnd, r["condition"], rep)
            a.need(ident not in saved, "Duplicate output")
            saved[ident] = r
    a.need(
        len(prompts) == 40 * len(indices) and len(saved) == 120 * len(indices),
        "Complete case axes",
    )
    catalog = []
    for i in sorted(indices):
        key = matrix[i]["key"]
        frames = []
        for rnd in range(1, 5):
            frames.append(
                dict(
                    round=rnd,
                    conditions=[
                        dict(
                            condition=c,
                            prompt=prompts[key, rnd, c],
                            outputs=[saved[key, rnd, c, r] for r in range(3)],
                            correction_record=(
                                records[prompts[key, rnd, c]["source_id"]]
                                if prompts[key, rnd, c]["source_id"]
                                else None
                            ),
                        )
                        for c in a.CONDITIONS
                    ],
                )
            )
        case = dict(
            input_index=i,
            key=key,
            population=matrix[i]["population"],
            gold=roster[key],
            metadata=meta[i],
            trajectory=profiles[i],
            selection_slots=[s for s in selection if s["input_index"] == i],
            frames=frames,
        )
        path = a.OUT / "cases_v1" / f"input_{i:04d}.json"
        a.write(path, case)
        a.need(a.read(path) == a.native(case), "Saved case copy differs")
        catalog.append(
            dict(
                input_index=i,
                key=key,
                file=a.rel(path),
                sha256=a.sha(path),
                categories=[s["category"] for s in case["selection_slots"]],
            )
        )
    a.write(
        a.OUT / "cases_v1/complete.json",
        dict(
            status="part5_original_case_evidence_copied",
            created_utc=a.pure.utc(),
            input_bindings=bindings,
            catalog=catalog,
            case_slots=len(selection),
            empty_slots=[s for s in selection if s["status"] == "empty"],
            unique_inputs=len(indices),
            prompts=len(prompts),
            predictions=len(saved),
            source_sha256=a.sha(Path(__file__)),
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        f"Part 5 case evidence: {len(indices)} inputs, {len(prompts)} original prompts and {len(saved)} saved predictions.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()

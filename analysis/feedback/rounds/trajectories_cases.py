from collections import defaultdict
from pathlib import Path
from analysis.feedback.rounds import trajectories_inputs as a


def main():
    proof = a.read(a.OUT / "independent_verification.json")
    a.need(
        proof["status"] == "part4_independent_verification_pass",
        "Independent numeric proof required",
    )
    for name, digest in proof["input_bindings"].items():
        a.need(a.sha(a.ROOT / name) == digest, "Changed independently verified input")
    selection = a.read(a.OUT / "case_selection.json")
    indices = {r["input_index"] for r in selection if r["status"] == "selected"}
    matrices = {
        r["input_index"]: r
        for r in a.rows(a.READER / "matrix.jsonl")
        if r["input_index"] in indices
    }
    profiles = {
        r["input_index"]: r
        for r in a.rows(a.OUT / "input_trajectories.jsonl")
        if r["input_index"] in indices
    }
    keys = {r["key"]: i for i, r in matrices.items()}
    roster = {
        r["key"]: r for r in a.rows(a.STUDY / "evaluation.jsonl") if r["key"] in keys
    }
    records = a.read(a.STUDY / "gate_only_records.json")
    allowed = a.read(a.PART3 / "assembled.json")["input_bindings"]
    bindings = {}
    prompts = {}
    saved = {}
    slots = 0
    for rnd in range(1, 5):
        path = a.STUDY / f"phases/r{rnd}_evaluation/prompts.jsonl"
        name = a.rel(path)
        bindings[name] = a.sha(path)
        a.need(bindings[name] == allowed[name], "Original phase prompt changed")
        for prompt in a.rows(path):
            if prompt["key"] not in keys:
                continue
            key = (prompt["key"], rnd, prompt["condition"])
            a.need(key not in prompts, "Duplicate case prompt")
            prompts[key] = prompt
    reader = a.read(a.READER / "complete.json")
    for stage in reader["stage_receipts"]:
        if stage["role"] != "evaluation":
            continue
        receipt_path = a.ROOT / stage["receipt"]
        receipt = a.read(receipt_path)
        a.need(
            a.sha(receipt_path) == stage["receipt_sha256"],
            "Reader case receipt changed",
        )
        path = a.ROOT / stage["predictions"]
        digest = a.sha(path)
        a.need(digest == receipt["predictions_sha256"], "Case assembly source changed")
        bindings[a.rel(path)] = digest
        bindings[a.rel(receipt_path)] = a.sha(receipt_path)
        for row in a.rows(path):
            if row["key"] not in keys:
                continue
            rnd = int(row["phase"][1])
            col = a.CONDITIONS.index(row["condition"])
            rep = row["repeat"]
            i = keys[row["key"]]
            expected = matrices[i]["predictions"][rnd - 1][col][rep]
            a.need(
                row["result"] == expected["result"]
                and row["n_valid"] == expected["n_valid"]
                and row["uncertain"] == expected["uncertain"],
                "Case output differs from matrix",
            )
            a.need(
                row["messages_sha256"]
                == prompts[row["key"], rnd, row["condition"]]["messages_sha256"],
                "Case actual message binding differs",
            )
            ident = (row["key"], rnd, row["condition"], rep)
            a.need(ident not in saved, "Duplicate case output")
            saved[ident] = row
            slots += 1
    a.need(
        len(prompts) == len(indices) * 40 and slots == len(indices) * 120,
        "Complete case matrix",
    )
    catalog = []
    for i in sorted(indices):
        row = matrices[i]
        key = row["key"]
        frames = []
        for rnd in range(1, 5):
            conditions = []
            for condition in a.CONDITIONS:
                prompt = prompts[key, rnd, condition]
                sid = prompt["source_id"]
                conditions.append(
                    dict(
                        condition=condition,
                        prompt=prompt,
                        outputs=[saved[key, rnd, condition, r] for r in range(3)],
                        correction_record=records[sid] if sid else None,
                    )
                )
            frames.append(dict(round=rnd, conditions=conditions))
        slots_for_key = [r for r in selection if r["input_index"] == i]
        case = dict(
            input_index=i,
            key=key,
            population=row["population"],
            gold=roster[key],
            reader_metadata={k: v for k, v in row.items() if k != "predictions"},
            trajectory=profiles[i],
            selection_slots=slots_for_key,
            frames=frames,
            interpretation_scope="Recorded-label binary correctness and observed contexts. Outcomes selected these declared illustrative categories; no causal reasoning or complete-answer conclusion is inferred.",
        )
        path = a.OUT / "cases_v1" / f"input_{i:04d}.json"
        a.write(path, case)

        a.need(a.read(path) == a.native(case), "Saved case copy changed")
        catalog.append(
            dict(
                input_index=i,
                key=key,
                file=a.rel(path),
                sha256=a.sha(path),
                slots=[r["slot"] for r in slots_for_key],
            )
        )
    a.write(
        a.OUT / "cases_v1/complete.json",
        dict(
            status="original_prompt_and_audited_case_matrix_complete",
            created_utc=a.pure.utc(),
            input_bindings=bindings,
            catalog=catalog,
            case_slots=len(selection),
            selected_slots=sum(r["status"] == "selected" for r in selection),
            empty_slots=[r for r in selection if r["status"] == "empty"],
            unique_inputs=len(indices),
            prompts=len(prompts),
            prediction_slots=slots,
            source_sha256=a.sha(Path(__file__)),
            case_selection_independently_verified=True,
            new_Qwen_calls=0,
            new_fits=0,
            new_embeddings=0,
        ),
    )
    print(
        f"{len(indices)} complete cases extracted: {len(prompts)} original prompts, {slots} audited predictions; empty slots retained.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()

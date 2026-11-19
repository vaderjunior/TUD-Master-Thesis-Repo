import hashlib
import json
from pathlib import Path
from analysis.feedback.rounds import trajectories_inputs as a


def sig(v):
    return hashlib.sha256(
        (
            json.dumps(v, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False)
            + "\n"
        ).encode()
    ).hexdigest()


def main():
    receipt = a.read(a.OUT / "cases_v1/complete.json")
    rows = list(a.rows(a.READER / "matrix.jsonl"))
    for p, h in receipt["input_bindings"].items():
        a.need(a.sha(a.ROOT / p) == h, "Case original input changed")
    bindings = {
        (r["phase"], r["key"], r["condition"]): r
        for r in a.rows(a.PART3 / "prompt_bindings.jsonl")
        if r["phase"].endswith("_evaluation")
    }
    selections = a.read(a.OUT / "case_selection.json")
    prompt_count = prediction_count = 0
    for item in receipt["catalog"]:
        path = a.ROOT / item["file"]
        a.need(a.sha(path) == item["sha256"], "Case file changed")
        c = a.read(path)
        i = c["input_index"]
        original = rows[i]
        a.need(
            c["key"] == original["key"] and c["gold"]["gate"] == original["gold_hate"],
            "Case query/gold",
        )
        a.need(
            c["selection_slots"] == [s for s in selections if s["input_index"] == i],
            "Case selection category",
        )
        for frame in c["frames"]:
            rnd = frame["round"]
            a.need(
                [v["condition"] for v in frame["conditions"]] == a.CONDITIONS,
                "Complete case conditions",
            )
            for col, v in enumerate(frame["conditions"]):
                p = v["prompt"]
                condition = v["condition"]
                bound = bindings[f"r{rnd}_evaluation", c["key"], condition]
                a.need(
                    sig(p["messages"])
                    == p["messages_sha256"]
                    == bound["messages_sha256"]
                    and sig(p["context"])
                    == p["context_sha256"]
                    == bound["context_sha256"],
                    "Independent case message/content signature",
                )
                a.need(
                    p["text"] == c["gold"]["text"]
                    and p["source_id"] == bound["source_id"],
                    "Case prompt query/source",
                )
                prompt_count += 1
                for rep, out in enumerate(v["outputs"]):
                    target = original["predictions"][rnd - 1][col][rep]
                    a.need(
                        out["repeat"] == rep
                        and out["key"] == c["key"]
                        and out["condition"] == condition
                        and out["phase"] == f"r{rnd}_evaluation",
                        "Case prediction axes",
                    )
                    a.need(
                        out["result"] == target["result"]
                        and out["n_valid"] == target["n_valid"]
                        and out["uncertain"] == target["uncertain"],
                        "Case result equals independent matrix",
                    )
                    a.need(
                        out["messages_sha256"] == bound["messages_sha256"],
                        "Executed case hash",
                    )
                    prediction_count += 1
        for p, trajectory in c["trajectory"]["policies"].items():
            codes = []
            for frame in c["frames"]:
                values = []
                for stream in a.STREAMS:
                    condition = next(
                        v
                        for v in frame["conditions"]
                        if v["condition"] == f"{p}_s{stream}"
                    )
                    values.extend(
                        int(out["result"]["hate"] != c["gold"]["gate"])
                        for out in condition["outputs"]
                    )
                codes.append(values)
            a.need(
                codes == trajectory["errors_by_repeat"],
                "Case trajectory reconstructed from actual outputs",
            )
            for witness in trajectory["same_prompt_flip_witnesses"]:
                old, new = c["frames"][witness["round"] - 2 : witness["round"]]
                old = next(
                    v
                    for v in old["conditions"]
                    if v["condition"] == f'{p}_s{witness["stream"]}'
                )
                new = next(
                    v
                    for v in new["conditions"]
                    if v["condition"] == f'{p}_s{witness["stream"]}'
                )
                a.need(
                    old["prompt"]["messages"] == new["prompt"]["messages"],
                    "Same-prompt case witness differs",
                )
                a.need(
                    old["outputs"][witness["repeat"]]["result"]["hate"]
                    != new["outputs"][witness["repeat"]]["result"]["hate"],
                    "No actual gate flip in witness",
                )
    a.need(
        prompt_count == receipt["prompts"] == 440
        and prediction_count == receipt["prediction_slots"] == 1320,
        "Case coverage totals",
    )
    a.write(
        a.OUT / "case_independent_verification.json",
        dict(
            status="passed",
            created_utc=a.pure.utc(),
            unique_inputs=len(receipt["catalog"]),
            prompts=prompt_count,
            predictions=prediction_count,
            empty_slots=len(receipt["empty_slots"]),
            case_selection_slots=len(selections),
            inputs={
                a.rel(a.OUT / "cases_v1/complete.json"): a.sha(
                    a.OUT / "cases_v1/complete.json"
                )
            },
            source_sha256=a.sha(Path(__file__)),
            new_Qwen_calls=0,
        ),
    )
    print(
        "All 11 cases, 440 original prompts, 1320 saved predictions and identical-prompt flip witnesses verified.",
        flush=True,
    )


if __name__ == "__main__":
    with a.pure.offline():
        main()

from pathlib import Path
import re
from analysis.feedback.rounds import reader_earlier as base

ROOT, STUDY, REVIEW, OUT = base.ROOT, base.STUDY, base.REVIEW, base.OUT
READER = OUT / "reader_v3"
STREAMS, POLICIES, CONDITIONS = base.STREAMS, base.POLICIES, base.CONDITIONS
need, sha, read, rows, rel, physical, pure = (
    base.need,
    base.sha,
    base.read,
    base.rows,
    base.rel,
    base.physical,
    base.pure,
)
write, write_lines = base.write, base.write_lines


def explicit_axes(value):
    result = dict(value)
    result["source_phase_stream"] = result.pop("stream")
    if result["role"] == "evaluation":
        if result["condition"] == "reference":
            result.update(policy="reference", stream=None)
        else:
            match = re.fullmatch(
                r"(adaptive|static|random)_s(202610120[1-3])", result["condition"]
            )
            need(match is not None, "Invalid evaluation condition")
            result.update(policy=match[1], stream=int(match[2]))
        need(
            result["source_phase_stream"] is None,
            "Evaluation phase has an unexpected source stream",
        )
    else:
        need(
            result["condition"] in ["reference", "adaptive"], "Invalid source condition"
        )
        result.update(policy=result["condition"], stream=result["source_phase_stream"])
    return result


def build():
    previous = read(base.READER / "complete.json")
    need(
        previous["status"] == "all_four_rounds_reader_assembled",
        "Complete v2 assembly required",
    )
    for name, digest in {**previous["input_bindings"], **previous["artifacts"]}.items():
        need(sha(ROOT / name) == digest, "Changed v2 input/output")
    stages = []
    for stage in previous["stage_receipts"]:
        need(
            sha(ROOT / stage["receipt"]) == stage["receipt_sha256"],
            "Changed v2 stage receipt",
        )
        old_receipt = read(ROOT / stage["receipt"])
        old_predictions = ROOT / stage["predictions"]
        need(
            sha(old_predictions) == old_receipt["predictions_sha256"],
            "Changed v2 predictions",
        )
        target = READER / "stages" / stage["phase"] / stage["stage"]
        write_lines(
            target / "predictions.jsonl",
            (explicit_axes(v) for v in rows(old_predictions)),
        )
        receipt = dict(
            old_receipt,
            inputs={
                **old_receipt["inputs"],
                stage["predictions"]: sha(old_predictions),
                stage["receipt"]: sha(ROOT / stage["receipt"]),
            },
            reader_sha256=sha(Path(__file__)),
            predictions_sha256=sha(target / "predictions.jsonl"),
            change="Explicit policy/stream fields only; original stream retained as source_phase_stream",
        )
        write(target / "complete.json", receipt)
        stages.append(
            dict(
                stage,
                receipt=rel(target / "complete.json"),
                receipt_sha256=sha(target / "complete.json"),
                predictions=rel(target / "predictions.jsonl"),
            )
        )
        print(
            "Explicit axes checked: " + stage["phase"] + "/" + stage["stage"],
            flush=True,
        )
    for filename in ["matrix.jsonl", "roster.jsonl", "duplicate_vote_labels.json"]:
        write(READER / filename, physical(base.READER / filename).read_bytes())
    result = dict(
        previous,
        stage_receipts=stages,
        created_utc=pure.utc(),
        input_bindings={
            **previous["input_bindings"],
            rel(base.READER / "complete.json"): sha(base.READER / "complete.json"),
            rel(Path(__file__)): sha(Path(__file__)),
            "tests/feedback/rounds/test_reader.py": sha(
                ROOT / "tests/feedback/rounds/test_reader.py"
            ),
        },
        artifacts={
            rel(READER / f): sha(READER / f)
            for f in ["matrix.jsonl", "roster.jsonl", "duplicate_vote_labels.json"]
        },
        axis_adjustment="Explicit condition-level policy and stream; source-phase stream retained separately. Matrix and scientific values byte-identical to v2.",
    )
    write(READER / "complete.json", result)
    print(
        "Explicit multi-phase reader complete; independent verification is next.",
        flush=True,
    )


if __name__ == "__main__":
    with pure.offline():
        build()

from collections import defaultdict
import hashlib
from pathlib import Path
import numpy as np
from analysis.common import study as c
from training.original import inference as e


def assemble_value(arm, seed, key, heads):
    gate = heads[arm, "hate", seed][key]
    c.require(gate in ("false", "true"), "Unexpected encoded gate")
    value = {
        "hate": gate == "true",
        "hate_type": [],
        "target_group": [],
        "severity": None,
        "legal": [],
    }
    if arm == "de":
        value["legal"] = heads[arm, "legal", seed][key]
    elif value["hate"]:
        for head in ("hate_type", "target_group", "severity"):
            value[head] = heads[arm, head, seed][key]
    return value


def main():
    p = e.validate()
    done = c.read(e.BASE / "complete.json")
    c.require(
        done["status"] == "all_fixed_heads_complete"
        and done["plan_sha256"] == c.sha(e.PLAN),
        "Complete fixed-head matrix required",
    )
    c.verify(done["outputs"])
    roster = c.rows(c.PREP / "broad_roster.jsonl")
    heads = {}
    bindings = {
        c.rel(e.PLAN): c.sha(e.PLAN),
        c.rel(e.BASE / "complete.json"): c.sha(e.BASE / "complete.json"),
        **done["outputs"],
    }
    for job in p["jobs"]:
        root = e.BASE / "predictions" / job["job"]
        receipt = c.read(root / "complete.json")
        c.verify(receipt["outputs"])
        bindings.update(receipt["outputs"])
        rr = [r for r in roster if r["lang"] == job["meta"]["lang"]]
        saved = c.rows(root / "heads.jsonl")
        c.require(
            [r["key"] for r in saved] == [r["key"] for r in rr]
            and len(rr) == job["rows"],
            "Raw inference roster mismatch",
        )
        raw_chunks = []
        for path in sorted((root / "chunks").glob("chunk_*.json")):
            chunk = c.read(path)
            c.require(
                chunk["stamp"] == {"plan_sha256": c.sha(e.PLAN), "job": job["job"]}
                and c.signature(chunk["rows"]) == chunk["rows_sha256"],
                "Chunk binding differs",
            )
            raw_chunks += chunk["rows"]
        c.require(
            raw_chunks == saved,
            "Consolidated head predictions differ from durable chunks",
        )
        decisions = {}
        for row, source in zip(saved, rr):
            c.require(
                row["text_sha256"]
                == hashlib.sha256(source["text"].encode()).hexdigest()
                and row["job"] == job["job"]
                and (row["labels"] == job["meta"]["labels"])
                and (row["checkpoint_sha256"] == job["checkpoint_sha256"]),
                "Raw head attribution changed",
            )
            logits = np.asarray(row["logits"])
            c.require(
                logits.shape == (len(row["labels"]),) and np.isfinite(logits).all(),
                "Invalid saved logit shape/value",
            )
            expected = (
                (logits >= 0).astype(int).tolist()
                if job["meta"]["kind"] == "multilabel"
                else int(np.argmax(logits))
            )
            c.require(
                expected == row["decision"],
                "Saved decision differs from default threshold",
            )
            decisions[row["key"]] = (
                [lab for lab, bit in zip(row["labels"], expected) if bit]
                if isinstance(expected, list)
                else row["labels"][expected]
            )
        heads[job["arm"], job["head"], job["seed"]] = decisions
        print("Verified every raw head:", job["job"], flush=True)
    files = {}
    dimensions = {
        "encoder_full": ["hate", "target_group", "hate_type", "severity"],
        "encoder_kb": ["hate", "target_group", "hate_type", "severity"],
        "encoder_gbert": ["hate", "legal"],
    }
    for arm, lang, condition in [
        ("full", "en", "encoder_full"),
        ("kb", "en", "encoder_kb"),
        ("de", "de", "encoder_gbert"),
    ]:
        for seed in range(3):
            output = []
            for row in (r for r in roster if r["lang"] == lang):
                key = row["key"]
                value = assemble_value(arm, seed, key, heads)
                output.append(
                    {
                        "key": key,
                        "condition": condition,
                        "seed": seed,
                        "result": value,
                        "available_dimensions": dimensions[condition],
                    }
                )
            path = e.BASE / "assembled" / f"{condition}_seed{seed}.jsonl"
            c.save_rows(path, output)
            files[c.rel(path)] = c.sha(path)
    c.save(
        e.BASE / "assembled/complete.json",
        {
            "status": "all_logits_verified_and_gated_predictions_assembled",
            "plan_sha256": c.sha(e.PLAN),
            "inputs": bindings,
            "outputs": files,
            "available_dimensions": dimensions,
            "source_sha256": {
                c.rel(Path(__file__)): c.sha(Path(__file__)),
                c.rel(Path(e.__file__)): c.sha(Path(e.__file__)),
            },
            "new_training_steps": 0,
            "new_Qwen_calls": 0,
            "supporting_only": True,
        },
    )
    print(
        "Fixed encoder predictions assembled; ready for later paired supporting comparisons."
    )


if __name__ == "__main__":
    with c.offline():
        main()

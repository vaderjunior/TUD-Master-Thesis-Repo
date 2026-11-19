"""Policy verify."""

from collections import Counter
from copy import deepcopy
import hashlib
import json
import time
from analysis.knowledge.policy import common as c
from analysis.knowledge.policy import run as r


def definition_block(records):
    return "## Label definitions\n" + "\n".join(
        (
            "- "
            + v["meta"]["dimension"]
            + (" / " + v["meta"]["label"] if v["meta"].get("label") else "")
            + ": "
            + " ".join(v["text"].split())
            for v in records
        )
    )


def guideline_block(records):
    return "## Annotation guidelines\n" + "\n".join(("- " + v["text"] for v in records))


def check_schedule(schedule, index):
    lookup = {(v["key"], v["condition"]): v for v in index}
    keys = sorted(
        {v["key"] for v in index},
        key=lambda k: (hashlib.sha256((c.SEED + "|schedule|" + k).encode()).hexdigest(), k),
    )
    base = [(rep, condition) for rep in range(3) for condition in c.CONDITIONS]
    actual = []
    seen = set()
    global_index = 0
    for number, stage in enumerate(schedule["stages"]):
        expected_keys = keys[number * 150 : (number + 1) * 150]
        c.require(
            stage["id"] == f"batch_{number + 1:02d}" and stage["input_keys"] == expected_keys,
            "Batch input order differs",
        )
        c.require(
            stage["items"] == len(stage["jobs"]) == 9 * len(expected_keys)
            and stage["ordinary_calls"] == 3 * stage["items"],
            "Batch counts differ",
        )
        expected = []
        for key in expected_keys:
            offset = global_index % 9
            for repeat, condition in base[offset:] + base[:offset]:
                entry = lookup[key, condition]
                expected.append(
                    {
                        "key": key,
                        "condition": condition,
                        "repeat": repeat,
                        "population": entry["population"],
                        "messages_sha256": entry["messages_sha256"],
                    }
                )
            global_index += 1
        c.require(expected == stage["jobs"], "Condition/repeat interleaving differs")
        c.require(
            stage["population_counts"]
            == dict(Counter((lookup[k, c.CONDITIONS[0]]["population"] for k in expected_keys))),
            "Batch population identity differs",
        )
        for job in stage["jobs"]:
            identity = (job["key"], job["condition"], job["repeat"])
            c.require(identity not in seen, "Duplicate job")
            seen.add(identity)
        actual += stage["input_keys"]
    c.require(
        actual == keys
        and seen == {(k, cond, rep) for k in keys for cond in c.CONDITIONS for rep in range(3)},
        "Incomplete matrix",
    )
    c.require(len(seen) == 62154 and len(schedule["stages"]) == 47, "Frozen workload differs")
    return {
        "predictions": len(seen),
        "ordinary_calls": len(seen) * 3,
        "batches": len(schedule["stages"]),
        "last_batch_inputs": len(schedule["stages"][-1]["input_keys"]),
    }


def verify():
    c.part1_check()
    source_hashes = c.bindings(c.source_files())
    expected = {v["key"]: v for v in c.rows(c.SETUP / "prompt_bindings.jsonl")}
    inventory = [v["record"] for v in c.read(c.SETUP / "policy_inventory.json")]

    def as_hit(v):
        meta = {k: v[k] for k in ("kind", "lang", "source", "dimension", "label")}
        meta.update(v.get("meta", {}))
        return {"id": v["id"], "text": v["text"], "meta": meta, "via": "fixed_policy"}

    definitions = [as_hit(v) for v in inventory if v["kind"] == "definition"]
    guidelines = [as_hit(v) for v in inventory if v["kind"] == "guideline"]
    index = c.read(c.BASE / "prompt_index.json")
    c.require(
        len(index) == 20718 and len({(v["key"], v["condition"]) for v in index}) == 20718,
        "Prompt index repeats/misses entries",
    )
    by_key = {}
    for entry in index:
        by_key.setdefault(entry["key"], {})[entry["condition"]] = entry
    c.require(set(by_key) == set(expected), "Execution keys differ from Part 1")
    counter = r.RequestGuard(0, resource_check=lambda: {})
    seen = set()
    counts = Counter()
    totals = Counter()
    max_batch_bytes = 0
    with (c.BASE / "prompts.jsonl").open("rb") as stream:
        next_offset = 0
        for entry in index:
            c.require(
                entry["offset"] == next_offset and entry["length"] > 0,
                "Gapped or overlapping prompt index",
            )
            next_offset += entry["length"]
        c.require(
            next_offset == (c.BASE / "prompts.jsonl").stat().st_size,
            "Index does not cover exactly the prompt file",
        )
        for line_no, old in enumerate(c.rows(c.ORIGINAL_PROMPTS), 1):
            key = old["key"]
            if key not in expected:
                continue
            seen.add(key)
            binding = expected[key]
            c.require(line_no == binding["source_prompt_line"], "Source line differs")
            c.require(
                c.signature(old["messages"]) == binding["original_messages_sha256"]
                and c.signature(old["context"]) == binding["original_context_sha256"],
                "Original content differs",
            )
            original_prefix = (
                definition_block(old["context"]["definitions"])
                + "\n\n"
                + guideline_block(old["context"]["guidelines"])
                + "\n\n"
            )
            user = old["messages"][1]["text"]
            c.require(user.startswith(original_prefix), "Original prompt prefix differs")
            suffix = user[len(original_prefix) :]
            for condition in c.CONDITIONS:
                entry = by_key[key][condition]
                stream.seek(entry["offset"])
                raw = stream.read(entry["length"])
                c.require(
                    hashlib.sha256(raw).hexdigest() == entry["row_sha256"], "Indexed bytes differ"
                )
                row = json.loads(raw)
                ctx = deepcopy(old["context"])
                if condition != "retrieved_policy":
                    ctx["definitions"] = definitions
                if condition == "complete_policy":
                    ctx["guidelines"] = guidelines
                prefix = (
                    definition_block(ctx["definitions"])
                    + "\n\n"
                    + guideline_block(ctx["guidelines"])
                    + "\n\n"
                )
                wanted = [old["messages"][0], {"role": "user", "text": prefix + suffix}]
                c.require(
                    row["messages"] == wanted and row["context_sha256"] == c.signature(ctx),
                    "Prompt change outside planned policy blocks",
                )
                c.require(
                    row["messages_sha256"]
                    == entry["messages_sha256"]
                    == binding["message_hashes"][condition]
                    == c.signature(wanted),
                    "Message binding differs",
                )
                tokens = counter.count_tokens(c.a.body(wanted))
                c.require(
                    tokens
                    == row["reference_input_tokens"]
                    == entry["reference_input_tokens"]
                    == binding["counts"][condition],
                    "Token count differs",
                )
                c.require(
                    row["population"] == entry["population"] == old["population"]
                    and row["lang"] == old["lang"],
                    "Population/language differs",
                )
                c.require(
                    not {
                        "gate",
                        "legal",
                        "hate_types",
                        "target_groups",
                        "severity",
                        "analysis_endpoints",
                        "gold",
                    }
                    & set(row),
                    "Evaluation annotations in execution record",
                )
                totals[condition] += tokens
                counts[condition] += 1
            if len(seen) % 1500 == 0:
                print(f"Independent execution-matrix check: {len(seen)}/6906.", flush=True)
    c.require(
        seen == set(expected) and all((counts[k] == 6906 for k in c.CONDITIONS)), "Missing prompt"
    )
    budget = c.read(c.SETUP / "budget.json")
    c.require(
        sum(totals.values()) * 9 == budget["initial_input_tokens"], "Total token budget differs"
    )
    schedule = c.read(c.BASE / "schedule.json")
    checked = check_schedule(schedule, index)
    for stage in schedule["stages"]:
        cat = r.catalogue_for(stage, index)
        max_batch_bytes = max(
            max_batch_bytes,
            sum((len(json.dumps(v, ensure_ascii=False).encode()) for v in cat.values())),
        )
        c.require(len(cat) == 3 * len(stage["input_keys"]) <= 450, "Batch reader is not bounded")
    c.require(
        source_hashes == c.bindings(c.source_files()), "Implementation changed during verification"
    )
    report = {
        "status": "independent_full_matrix_verification_pass",
        "created_utc": c.a.utc(),
        "source_hashes": source_hashes,
        "inputs": 6906,
        "prompts": 20718,
        "schedule": checked,
        "reference_token_totals_before_repeats_votes": dict(totals),
        "maximum_batch_serialized_prompt_bytes": max_batch_bytes,
        "batch_reader_limit": 450,
        "actual_Qwen_calls": 0,
        "actual_model_list_GETs": 0,
        "credentials_read": False,
        "evaluation_gold_in_execution_records": False,
        "part1_complete_sha256": c.sha(c.SETUP / "part1_complete.json"),
    }
    path = c.BASE / "offline_checks" / f"matrix_{time.time_ns()}.json"
    c.save(path, report)
    print(
        json.dumps({"report": c.ref(path), "status": report["status"], "schedule": checked}),
        flush=True,
    )


if __name__ == "__main__":
    with c.offline():
        verify()

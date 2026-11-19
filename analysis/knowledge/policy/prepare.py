"""Policy prepare."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import os
import time
from analysis.knowledge.policy import common as c
from analysis.knowledge.policy import setup as setup


def schedule(index):
    keys = sorted(
        {v["key"] for v in index},
        key=lambda k: (
            hashlib.sha256((c.SEED + "|schedule|" + k).encode()).hexdigest(),
            k,
        ),
    )
    lookup = {(v["key"], v["condition"]): v for v in index}
    base = [(rep, cond) for rep in range(3) for cond in c.CONDITIONS]
    jobs = []
    stages = []
    for i, key in enumerate(keys):
        rotation = i % 9
        for rep, cond in base[rotation:] + base[:rotation]:
            entry = lookup[key, cond]
            jobs.append(
                {
                    "key": key,
                    "condition": cond,
                    "repeat": rep,
                    "population": entry["population"],
                    "messages_sha256": entry["messages_sha256"],
                }
            )
        if (i + 1) % 150 == 0 or i == len(keys) - 1:
            stage_keys = keys[len(stages) * 150 : i + 1]
            stages.append(
                {
                    "id": f"batch_{len(stages) + 1:02d}",
                    "input_keys": stage_keys,
                    "jobs": jobs,
                    "items": len(jobs),
                    "ordinary_calls": len(jobs) * 3,
                    "population_counts": dict(
                        Counter(
                            (
                                lookup[k, c.CONDITIONS[0]]["population"]
                                for k in stage_keys
                            )
                        )
                    ),
                }
            )
            jobs = []
    return {
        "ordering": "Part 1 hash order; repeat-major A/B/C base list rotated left by global input index mod 9.",
        "stages": stages,
    }


def prepare():
    c.part1_check()
    c.require(
        not c.BASE.exists(),
        "Preserve existing preparation folder; inspect it before preparing again",
    )
    c.BASE.mkdir()
    c.save(
        c.BASE / "preservation_before.json",
        {"part1_complete_sha256": c.sha(c.SETUP / "part1_complete.json")},
    )
    selected = {r["key"]: r for r in c.rows(c.SETUP / "prompt_bindings.jsonl")}
    c.require(len(selected) == 6906, "Selected roster differs")
    counts = Counter()
    index = []
    seen = set()
    token_counter = setup.counter()
    with (c.BASE / "prompts.jsonl").open("xb") as stream:
        for line_no, old in enumerate(c.rows(c.ORIGINAL_PROMPTS), 1):
            key = old["key"]
            if key not in selected:
                continue
            expected = selected[key]
            c.require(
                key not in seen and line_no == expected["source_prompt_line"],
                "Original prompt identity differs",
            )
            seen.add(key)
            c.require(
                c.signature(old["messages"]) == expected["original_messages_sha256"]
                and c.signature(old["context"]) == expected["original_context_sha256"],
                "Original prompt changed",
            )
            marker = "## Text to classify (language: " + old["lang"] + ")\n"
            c.require(
                old["messages"][1]["text"].count(marker) == 1, "Ambiguous query header"
            )
            query = old["messages"][1]["text"].split(marker, 1)[1]
            for condition, messages, context in setup.messages_for(old, query):
                message_sha = c.signature(messages)
                tokens = token_counter(messages)
                c.require(
                    message_sha == expected["message_hashes"][condition]
                    and tokens == expected["counts"][condition],
                    "Prepared prompt differs from verified Part 1",
                )
                c.require(0 < tokens <= 8192, "Initial prompt overflow")
                entry = {
                    "key": key,
                    "condition": condition,
                    "population": old["population"],
                    "lang": old["lang"],
                    "messages": messages,
                    "messages_sha256": message_sha,
                    "reference_input_tokens": tokens,
                    "context_sha256": c.signature(context),
                    "original_messages_sha256": expected["original_messages_sha256"],
                    "original_context_sha256": expected["original_context_sha256"],
                    "source_prompt_line": line_no,
                }
                raw = (
                    json.dumps(
                        entry, sort_keys=True, ensure_ascii=False, allow_nan=False
                    )
                    + "\n"
                ).encode("utf-8")
                offset = stream.tell()
                stream.write(raw)
                index.append(
                    {
                        k: entry[k]
                        for k in (
                            "key",
                            "condition",
                            "population",
                            "lang",
                            "messages_sha256",
                            "reference_input_tokens",
                        )
                    }
                    | {
                        "offset": offset,
                        "length": len(raw),
                        "row_sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
                counts[condition] += 1
            if len(seen) % 1500 == 0:
                print(f"Prepared {len(seen)}/6906 inputs offline.", flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    c.require(
        seen == set(selected) and all((counts[k] == 6906 for k in c.CONDITIONS)),
        "Prompt matrix incomplete",
    )
    c.save(c.BASE / "prompt_index.json", index)
    c.save(c.BASE / "schedule.json", schedule(index))
    c.save(
        c.BASE / "preparation_summary.json",
        {
            "created_utc": c.a.utc(),
            "inputs": len(seen),
            "prompts": len(index),
            "execution_prompt_bytes": (c.BASE / "prompts.jsonl").stat().st_size,
            "evaluation_gold_loaded": False,
            "new_retrieval": False,
            "actual_Qwen_calls": 0,
            "actual_model_list_GETs": 0,
            "part1_complete_sha256": c.sha(c.SETUP / "part1_complete.json"),
        },
    )
    c.part1_check()
    print(
        "Prompt-only matrix saved. Independent verification, mock execution and final freeze remain.",
        flush=True,
    )


def freeze(matrix_path, test_path):
    c.require(not c.PLAN.exists(), "Execution plan already frozen; do not overwrite it")
    c.part1_check()
    matrix, test = (c.read(matrix_path), c.read(test_path))
    c.require(
        matrix["status"] == "independent_full_matrix_verification_pass"
        and test["passed"],
        "Offline checks did not pass",
    )
    sources = c.bindings(c.source_files())
    c.require(
        matrix["source_hashes"] == sources == test["source_hashes"],
        "Checks do not bind the current implementation",
    )
    prepared = [
        c.BASE / name
        for name in (
            "prompts.jsonl",
            "prompt_index.json",
            "schedule.json",
            "preparation_summary.json",
        )
    ]
    inputs = {
        c.ref(c.SETUP / "part1_complete.json"): c.sha(c.SETUP / "part1_complete.json")
    }
    inputs.update(
        {
            c.ref(c.SETUP / name): value
            for name, value in c.read(c.SETUP / "part1_complete.json")[
                "artifacts"
            ].items()
        }
    )
    inputs.update(c.read(c.SETUP / "frozen_source_bindings.json"))
    c.save(
        c.PLAN,
        {
            "created_utc": c.a.utc(),
            "status": "frozen_offline_execution_plan",
            "conditions": list(c.CONDITIONS),
            "workers": c.WORKERS,
            "output_root": c.OUTPUT.as_posix(),
            "deadline": c.DEADLINE.isoformat(),
            "latest_new_start": c.LATEST_START.isoformat(),
            "environment": c.environment(),
            "api_contract": c.a.CONTRACT,
            "budget": c.COUNTS,
            "source_hashes": sources,
            "input_hashes": inputs,
            "prepared_hashes": c.bindings(prepared),
            "checked_by": c.bindings([matrix_path, test_path]),
            "no_accuracy_based_stopping": True,
            "resource_contract": {
                "initial_free_RAM_GiB": 2.0,
                "running_free_RAM_GiB": 0.75,
                "C_floor_GiB": 8.0,
                "D_floor_GiB": 8.0,
                "D_bytes_per_remaining_prediction": 2**20,
                "per_request_recheck_seconds": 30,
            },
            "request_guard": {
                "reference_input_cap": 8192,
                "repair_conversations_checked": True,
                "new_request_guard_before_intent": True,
                "no_truncation": True,
            },
            "scope": "One late complete-policy SQ1 comparison. No subsequent study, training or thesis edit.",
        },
    )
    print(
        "Execution plan frozen. Storage, PowerShell and final offline preflight must pass before ready.json.",
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("prepare", "freeze"))
    p.add_argument("--matrix")
    p.add_argument("--tests")
    args = p.parse_args()
    with c.offline():
        if args.mode == "prepare":
            prepare()
        else:
            c.require(
                args.matrix and args.tests, "Both offline check reports are required"
            )
            freeze(Path(args.matrix), Path(args.tests))

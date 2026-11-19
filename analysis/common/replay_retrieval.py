import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
from analysis.common import retrieval as r
from analysis.common import prompt as prompt

BASE = r.BASE / "replay_v1"
PLAN = BASE / "plan.json"
PROBE_KEY = "de/detox-1389930895067467778"


def comparison(old, new):
    changed = sorted(
        (
            k
            for k in set(old) | set(new)
            if k not in old or k not in new or old[k] != new[k]
        )
    )
    return {
        "same_entry": not changed,
        "same_messages": old["messages"] == new["messages"],
        "same_context": old["context"] == new["context"],
        "same_retrieved_ids": old["retrieved"] == new["retrieved"],
        "changed_fields": changed,
    }


def prepare():
    r.require(not PLAN.exists(), "Preserve frozen replay plan")
    r.validate()
    required = [
        Path(__file__),
        r.ROOT / "tests/common/test_retrieval_replay.py",
        r.ROOT / "scripts/replay_retrieval.ps1",
        r.PLAN,
        r.BASE / "full/complete.json",
        r.BASE / "full/prompts.jsonl",
        r.BASE / "small/cold_check.json",
        r.BASE / "logs/check_full_1789485784975188200.txt",
        r.BASE / "reopen_probe_1789486543389871100.json",
        r.ROOT / "analysis/common/retrieval_diagnosis.py",
    ]
    r.put(
        PLAN,
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "original_plan_sha256": r.sha(r.PLAN),
            "input_hashes": {r.rel(p): r.sha(p) for p in required},
            "passes": 2,
            "rows_per_full_pass": 1295,
            "scope": "Diagnostic only. Two fresh processes replay all full-pool DEV queries in original order; save per-item match flags and the full actual entry for every mismatch. Never replace the first-pass prompts, relax a comparison, rebuild the index, or call a classifier.",
            "reason": "Original cold replay stopped after 1000 logged matches at one German input; actual differing entry was not saved. A separate three-query diagnostic matched the original exactly. Cause and extent remain unresolved.",
            "index_policy": "Revalidate every stored record/annotation/vector before each pass. Save physical file hashes before and after each pass as observations, not as a reason to rebuild. Original ranking code and cached query vectors unchanged.",
        },
    )
    print(json.dumps({"prepared": r.rel(PLAN), "sha256": r.sha(PLAN)}, indent=2))


def validate():
    p = r.read(PLAN)
    r.require(p["original_plan_sha256"] == r.sha(r.PLAN), "Original plan changed")
    r.verify_hashes(p["input_hashes"])
    r.validate()
    return p


def index_hashes():
    return {
        r.rel(p): r.sha(p)
        for p in sorted((r.BASE / "full/chroma").rglob("*"))
        if p.is_file()
    }


def worker(mode, number):
    import chromadb

    r.require(
        mode == "probe" and number == 0 or (mode == "full" and number in (0, 1)),
        "Unexpected replay identifier",
    )
    validate()
    records, vectors, dev, q = r.data("full")
    stored = {x["key"]: x for x in r.rows(r.BASE / "full/prompts.jsonl")}
    encoder = r.CachedEncoder([x["text"] for x in dev], q)
    if mode == "probe":
        dev = [dev[0], next((x for x in dev if x["key"] == PROBE_KEY)), dev[-1]]
    target = BASE / f"{mode}_{number}"
    if (target / "complete.json").exists():
        done = r.read(target / "complete.json")
        r.require(done["plan_sha256"] == r.sha(PLAN), "Wrong replay receipt")
        r.require(
            done["mode"] == mode
            and done["pass"] == number
            and (done["totals"]["items"] == len(dev)),
            "Incomplete/wrong replay receipt",
        )
        r.verify_hashes(done["hashes"])
        print(f"{mode}/{number} already complete", flush=True)
        return
    attempt = target / f"attempt_{time.time_ns()}"
    attempt.mkdir(parents=True, exist_ok=False)
    before = index_hashes()
    col = chromadb.PersistentClient(path=str(r.BASE / "full/chroma")).get_collection(
        "hate_kb"
    )
    origin = r.read(r.BASE / "full/origin.json")
    r.require(
        all((col.metadata.get(k) == v for k, v in origin.items())),
        "Index origin changed",
    )
    maxdiff = r.verify_collection(col, records, vectors)
    retriever = r.make_retriever(
        col, r.CORE / "kb/full/records.jsonl", r.read(r.PLAN)["retrieval"], encoder
    )
    known = {x["id"]: x for x in records}
    totals = {
        "items": 0,
        "entry_differences": 0,
        "message_differences": 0,
        "context_differences": 0,
        "retrieved_id_differences": 0,
    }
    log = attempt / "comparisons.jsonl"
    started = time.perf_counter()
    with log.open("x", encoding="utf8", newline="\n") as f:
        for x in dev:
            ctx = prompt.from_retrieval(retriever.retrieve(x["text"], x["lang"]))
            r.validate_context(ctx, known, x["lang"])
            messages = prompt.render_messages(x["text"], x["lang"], ctx, "corrected")
            actual = {
                **x,
                "pool": "full",
                "condition": "rag_full",
                "context": ctx,
                "context_sha256": r.signature(ctx),
                "messages": messages,
                "messages_sha256": r.signature(messages),
                "retrieved": prompt.hit_ids(ctx),
                "plan_sha256": r.sha(r.PLAN),
            }
            check = comparison(stored[x["key"]], actual)
            record = {
                "key": x["key"],
                **check,
                "actual_messages_sha256": actual["messages_sha256"],
                "actual_context_sha256": actual["context_sha256"],
            }
            if not check["same_entry"]:
                record["actual_entry"] = actual
            f.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True, allow_nan=False)
                + "\n"
            )
            f.flush()
            totals["items"] += 1
            for field, name in [
                ("same_entry", "entry_differences"),
                ("same_messages", "message_differences"),
                ("same_context", "context_differences"),
                ("same_retrieved_ids", "retrieved_id_differences"),
            ]:
                totals[name] += not check[field]
            if totals["items"] % 100 == 0 or totals["items"] == len(dev):
                print(
                    f"{mode.upper()} {number}: {totals} ({time.perf_counter() - started:.0f}s)",
                    flush=True,
                )
    rr = r.rows(log)
    r.require(
        [x["key"] for x in rr] == [x["key"] for x in dev],
        "Incomplete diagnostic output",
    )
    receipt = {
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "status": "diagnostic_complete_not_clearance",
        "plan_sha256": r.sha(PLAN),
        "mode": mode,
        "pass": number,
        "totals": totals,
        "index_vector_max_abs_difference": maxdiff,
        "physical_index_before": before,
        "physical_index_after": index_hashes(),
        "hashes": {r.rel(log): r.sha(log)},
        "next": "Astra inspects all recorded differences and preserves the original failed cold check. A completed diagnostic is not automatic clearance.",
    }
    r.put(target / "complete.json", receipt)


def run(mode):
    with r.lock():
        validate()
        if mode == "full":
            ready = r.read(BASE / "probe_0/complete.json")
            r.require(
                ready["mode"] == "probe"
                and ready["totals"]["items"] == 3
                and (ready["plan_sha256"] == r.sha(PLAN)),
                "Short probe incomplete",
            )
            r.verify_hashes(ready["hashes"])
        for number in range(2 if mode == "full" else 1):
            subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "-B",
                    "-u",
                    "-m",
                    __spec__.name,
                    "worker",
                    "--kind",
                    mode,
                    "--number",
                    str(number),
                ],
                cwd=r.ROOT,
                check=True,
            )
        receipts = [
            r.read(BASE / f"{mode}_{i}/complete.json")
            for i in range(2 if mode == "full" else 1)
        ]
        print(
            json.dumps(
                {
                    "status": "diagnostics_finished",
                    "mode": mode,
                    "passes": [x["totals"] for x in receipts],
                },
                indent=2,
            ),
            flush=True,
        )
        if mode == "full":
            print(
                "BOTH COLD REPLAY PASSES COMPLETE. Tell Astra to inspect the saved comparisons; do not replace any prompts.",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", choices=["prepare", "validate", "probe", "run", "worker", "status"]
    )
    parser.add_argument("--kind", choices=["probe", "full"])
    parser.add_argument("--number", type=int, default=0)
    a = parser.parse_args()
    if a.mode == "prepare":
        prepare()
    elif a.mode == "validate":
        validate()
        print("Replay plan validated. No retrieval/model/API calls.")
    elif a.mode in ("probe", "run"):
        run("probe" if a.mode == "probe" else "full")
    elif a.mode == "worker":
        r.require(a.kind is not None, "Worker kind required")
        worker(a.kind, a.number)
    else:
        print(
            json.dumps(
                {
                    "prepared": PLAN.exists(),
                    "probe": (BASE / "probe_0/complete.json").exists(),
                    "full_passes": [
                        (BASE / f"full_{i}/complete.json").exists() for i in range(2)
                    ],
                }
            )
        )


if __name__ == "__main__":
    main()

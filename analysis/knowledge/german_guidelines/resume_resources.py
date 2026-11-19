from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
from analysis.knowledge.german_guidelines import resources as x

c = x.c


def scan(root):
    root = Path(root)
    io_root = c.a.storage_path(root)
    stack = [io_root]
    groups = defaultdict(lambda: Counter(files=0, bytes=0))
    types = defaultdict(lambda: Counter(files=0, bytes=0))
    records = []
    x.require(
        not io_root.is_symlink() and (not io_root.is_junction()), "Linked storage root"
    )
    while stack:
        path = stack.pop()
        with os.scandir(path) as entries:
            for entry in entries:
                x.require(
                    not entry.is_symlink() and (not Path(entry.path).is_junction()),
                    "Linked inventory entry",
                )
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    continue
                x.require(
                    entry.is_file(follow_symlinks=False), "Unexpected storage object"
                )
                st = entry.stat(follow_symlinks=False)
                rel = Path(entry.path).relative_to(io_root).as_posix()
                parts = rel.split("/")
                group = (
                    "/".join(parts[:2])
                    if parts[0] == "live" and len(parts) > 2
                    else parts[0] if len(parts) > 1 else "root_files"
                )
                groups[group].update(files=1, bytes=st.st_size)
                types[
                    (
                        entry.name
                        if entry.name
                        in (
                            "intent.json",
                            "outcome.json",
                            "result.json",
                            "complete.json",
                            "job.json",
                            "item.lock",
                        )
                        else Path(entry.name).suffix or "no_suffix"
                    )
                ].update(files=1, bytes=st.st_size)
                records.append((rel, st.st_size, st.st_mtime_ns))
                if len(records) % 100000 == 0:
                    print(
                        f"Storage metadata only: {len(records):,} files counted.",
                        flush=True,
                    )
    digest = hashlib.sha256()
    for rec in sorted(records):
        digest.update(
            (json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        )
    return {
        "root": str(root),
        "scanned_utc": c.a.utc(),
        "files": len(records),
        "logical_bytes": sum((v[1] for v in records)),
        "groups": dict(groups),
        "by_basename_or_suffix": dict(types),
        "path_size_mtime_metadata_sha256": digest.hexdigest(),
        "method": "Extended-path metadata traversal; no file contents read by inventory; logical bytes only, not disk blocks or a new content audit.",
        "final_completion_mtime_utc": (
            datetime.fromtimestamp(
                (io_root / "live/complete.json").stat().st_mtime, timezone.utc
            ).isoformat()
            if root == c.OUTPUT
            else None
        ),
    }


def tests():
    out = x.BASE / "tests" / f"long_paths_{time.time_ns()}"
    p = out / "fixture" / ("a" * 70) / ("b" * 70) / ("c" * 70) / "test.bin"
    io = c.a.storage_path(p)
    io.parent.mkdir(parents=True)
    with io.open("xb") as f:
        f.write(b"example")
    x.require(len(str(p)) > 260, "Long-path fixture is too short")
    z = scan(out / "fixture")
    x.require(
        z["files"] == 1 and z["logical_bytes"] == 7, "Extended-path file count failed"
    )
    x.require(
        z["groups"]["a" * 70] == {"files": 1, "bytes": 7},
        "Long-path relative grouping failed",
    )
    x.save(
        out / "report.json",
        {
            "passed": True,
            "checks": 3,
            "source": c.bindings([Path(__file__)]),
            "fixture_preserved": str(p),
            "actual_Qwen_calls": 0,
        },
    )
    print("Extended-path traversal tests passed; fixture preserved.", flush=True)


def resume(out):
    x.require(
        out.parent == x.BASE / "runs" and (not (out / "manifest.json").exists()),
        "Expected unfinished Part 7 analysis only",
    )
    c.verify(c.read(x.BASE / "analysis_contract.json")["bindings"])
    _, rob = x.r.verified()
    tests = sorted((x.BASE / "tests").glob("long_paths_*/report.json"))
    x.require(tests and c.read(tests[-1])["passed"], "Extended-path test missing")
    c.verify(c.read(tests[-1])["source"])
    prior = c.bindings((p for p in out.iterdir() if p.is_file()))
    x.save(
        out / "inventory_resume_scope.json",
        {
            "status": "resume_only_unfinished_inventory_and_synthesis_inputs",
            "created_utc": c.a.utc(),
            "reason": "Initial D: metadata count finished, but the following C: preparation test-folder traversal exceeded ordinary Windows path handling before either total was saved. Reuse all saved request/token/timing outputs; repeat metadata-only counting with extended paths.",
            "unchanged_completed_outputs": prior,
            "sources": c.bindings([Path(__file__), tests[-1]]),
            "actual_Qwen_calls": 0,
            "D_writes": 0,
        },
    )
    for name, root in [("D_study", c.OUTPUT), ("C_preparation", c.BASE)]:
        x.require(c.normal(root) in (c.OUTPUT, c.BASE), "Out-of-scope inventory")
        x.save(out / ("storage_" + name + ".json"), scan(root))
    x.save(
        out / "storage.json",
        {
            "D_study": c.read(out / "storage_D_study.json"),
            "C_preparation": c.read(out / "storage_C_preparation.json"),
            "excludes": "Other studies, shared caches, thesis, private backups and new Part 7 outputs; no allocated-block or peak-space estimate.",
        },
    )
    prior_scores = c.read(rob / "original_coverage_bounds_repeats.json")
    tradeoff = {}
    for left, right in x.r.PAIRS:
        aa = prior_scores["de_test"]["common"][left]["mean"]
        bb = prior_scores["de_test"]["common"][right]["mean"]
        fp = aa["fp"] - bb["fp"]
        fn = aa["fn"] - bb["fn"]
        tradeoff[left + "_minus_" + right] = {
            "mean_extra_FP": fp,
            "mean_extra_FN": fn,
            "equal_cost_error_change": fp + fn,
            "illustrative_missed_hate_to_false_alarm_break_even": (
                fp / -fn if fp > 0 and fn < 0 else None
            ),
            "scope": "Observed cohort, FP cost normalized to 1; no chosen utility, prevalence transfer, confidence interval or deployment recommendation.",
        }
    x.save(out / "practical_tradeoffs.json", tradeoff)
    x.save(
        out / "synthesis_scores.json",
        {
            ph: {
                k: v
                for k, v in prior_scores[ph].items()
                if k
                in (
                    "full_inputs",
                    "common_inputs",
                    "common",
                    "contrasts",
                    "full_roster_bounds",
                )
            }
            for ph in x.PHASES
        },
    )
    c.verify(prior)
    c.verify(c.read(x.BASE / "analysis_contract.json")["bindings"])
    x.save(
        out / "manifest.json",
        {
            "status": "part7_pending_independent_check",
            "created_utc": c.a.utc(),
            "run": c.ref(out),
            "outputs": c.bindings((p for p in out.rglob("*") if p.is_file())),
            "additional_sources": c.bindings([Path(__file__), tests[-1]]),
            "actual_Qwen_calls": 0,
            "D_writes": 0,
        },
    )
    print(
        json.dumps(
            {
                "run": c.ref(out),
                "status": "Completed with saved resource calculations unchanged; ready for independent check",
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["test", "resume"])
    parser.add_argument("run", nargs="?")
    args = parser.parse_args()
    with x.offline():
        tests() if args.mode == "test" else resume(c.ROOT / args.run)

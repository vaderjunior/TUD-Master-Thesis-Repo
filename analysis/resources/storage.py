from src.hsrag.paths import PROJECT_ROOT
from collections import defaultdict
import argparse
from datetime import datetime, timezone
import heapq
import json
import os
from pathlib import Path
import shutil
import time

ROOT = PROJECT_ROOT
OUT = ROOT / "experiments/astra/storage_review_20261011_v1"
NOW = time.time()
REPARSE = 1024


def scan(root):
    groups = defaultdict(
        lambda: dict(files=0, bytes=0, modified_24h_bytes=0, modified_48h_bytes=0)
    )
    suffixes = defaultdict(lambda: dict(files=0, bytes=0))
    days = defaultdict(lambda: dict(files=0, bytes=0))
    largest, recent, errors, skipped = ([], [], [], [])
    stack = [(root, ())]
    files = total = 0
    last = started = time.monotonic()
    while stack:
        directory, parts = stack.pop()
        try:
            with os.scandir(directory) as entries:
                for e in entries:
                    try:
                        st = e.stat(follow_symlinks=False)
                        if (
                            e.is_symlink()
                            or getattr(st, "st_file_attributes", 0) & REPARSE
                        ):
                            skipped.append(str(Path(e.path).relative_to(root)))
                            continue
                        if e.is_dir(follow_symlinks=False):
                            stack.append((Path(e.path), parts + (e.name,)))
                            continue
                        if not e.is_file(follow_symlinks=False):
                            continue
                        n, age = (st.st_size, NOW - st.st_mtime)
                        files += 1
                        total += n
                        names = ["(all)"] + [
                            "/".join(parts[:d])
                            for d in range(1, min(4, len(parts)) + 1)
                        ]
                        if not parts:
                            names.append("(root files)")
                        for name in names:
                            g = groups[name]
                            g["files"] += 1
                            g["bytes"] += n
                            if 0 <= age <= 86400:
                                g["modified_24h_bytes"] += n
                            if 0 <= age <= 172800:
                                g["modified_48h_bytes"] += n
                        suf = Path(e.name).suffix.lower() or "(none)"
                        suffixes[suf]["files"] += 1
                        suffixes[suf]["bytes"] += n
                        day = (
                            datetime.fromtimestamp(st.st_mtime, timezone.utc)
                            .date()
                            .isoformat()
                        )
                        days[day]["files"] += 1
                        days[day]["bytes"] += n
                        relative = "/".join((*parts, e.name))
                        row = (n, relative, st.st_mtime)
                        if len(largest) < 60:
                            heapq.heappush(largest, row)
                        elif n > largest[0][0]:
                            heapq.heapreplace(largest, row)
                        if n >= 16 * 2**20 and 0 <= age <= 172800:
                            recent.append(
                                dict(
                                    path=relative,
                                    bytes=n,
                                    modified_utc=datetime.fromtimestamp(
                                        st.st_mtime, timezone.utc
                                    ).isoformat(),
                                )
                            )
                    except OSError as error:
                        errors.append(
                            dict(path=str(e.path), error=type(error).__name__)
                        )
        except OSError as error:
            errors.append(dict(path=str(directory), error=type(error).__name__))
        if time.monotonic() - last > 12:
            print(
                f"{root.name}: metadata for {files:,} files, {total / 2 ** 30:.2f} GiB counted",
                flush=True,
            )
            last = time.monotonic()
    return dict(
        root=str(root),
        files=files,
        logical_bytes=total,
        seconds=time.monotonic() - started,
        by_directory=dict(sorted(groups.items())),
        by_suffix=dict(suffixes),
        by_mtime_day=dict(sorted(days.items())),
        largest=[
            dict(
                bytes=n,
                path=p,
                modified_utc=datetime.fromtimestamp(t, timezone.utc).isoformat(),
            )
            for n, p, t in sorted(largest, reverse=True)
        ],
        recent_large=sorted(recent, key=lambda r: r["bytes"], reverse=True),
        skipped_reparse=skipped,
        errors=errors,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--focused", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / ("focused_inventory.json" if args.focused else "inventory.json")
    if destination.exists():
        raise RuntimeError("Preserve existing inventory")
    roots = [
        ROOT,
        Path(os.environ.get("HF_HOME") or Path.home() / ".cache/huggingface"),
        Path(
            os.environ.get("PIP_CACHE_DIR")
            or (
                (
                    Path(
                        os.environ.get("LOCALAPPDATA") or Path.home() / "AppData/Local"
                    )
                    / "pip/Cache"
                )
                if os.name == "nt"
                else (
                    Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
                    / "pip"
                )
            )
        ),
        Path(os.environ.get("TORCH_HOME") or Path.home() / ".cache/torch"),
    ]
    if args.focused:
        roots = [
            ROOT / "models",
            ROOT / "experiments/astra/compact_roberta_tuning_20261010_v1",
            ROOT / "experiments/astra/matched_type_pools_20261011_v1",
            ROOT / ".venv",
            ROOT / ".git",
            ROOT / "backups",
            ROOT / "data",
            ROOT / "kb",
            *roots[1:],
        ]
    result = dict(
        created_utc=datetime.now(timezone.utc).isoformat(),
        roots=[],
        disks={},
        scope="Repository and named model/package caches; filesystem metadata only.",
        limits="Logical file sizes; hard links may be counted more than once. Reparse points skipped. Modification dates do not establish net disk growth or creation time. A running experiment can change after inventory.",
    )
    if args.focused:
        result["scope"] = (
            "Focused model, runtime, dataset, KB, Git, backup and named cache inventory after model offload; not all experiment API files."
        )
        result["earlier_broad_scan"] = (
            "Stopped deliberately after reporting 3,755,107 files / 47.01 GiB examined; it was incomplete, produced no total inventory, and changed no payloads. This focused scan avoids walking millions of request records again."
        )
    for root in roots:
        if root.exists():
            value = scan(root)
            result["roots"].append(value)
            print(
                json.dumps(
                    dict(
                        root=str(root),
                        files=value["files"],
                        GiB=value["logical_bytes"] / 2**30,
                        errors=len(value["errors"]),
                    )
                ),
                flush=True,
            )
    for drive in ("C:/", "D:/", "F:/"):
        try:
            disk = shutil.disk_usage(drive)
            result["disks"][drive] = dict(
                total=disk.total, used=disk.used, free=disk.free
            )
        except OSError as error:
            result["disks"][drive] = dict(error=type(error).__name__)
    destination.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf8"
    )
    first = result["roots"][0]
    rows = [
        dict(
            path=k,
            GiB=round(v["bytes"] / 2**30, 3),
            modified_48h_GiB=round(v["modified_48h_bytes"] / 2**30, 3),
            files=v["files"],
        )
        for k, v in first["by_directory"].items()
        if k.count("/") == 0
        or (k.startswith("experiments/astra/") and k.count("/") == 2)
    ]
    print(
        json.dumps(
            dict(
                report=str(destination),
                top_folders=sorted(rows, key=lambda v: v["GiB"], reverse=True)[:24],
                disks={
                    k: dict(v, free_GiB=round(v.get("free", 0) / 2**30, 2))
                    for k, v in result["disks"].items()
                },
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

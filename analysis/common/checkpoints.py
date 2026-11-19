"""Common checkpoints."""

from pathlib import Path
from analysis.common import study as c


def gather(folder, items, make, stamp, chunk_size=100, progress=None):
    folder = c.own(folder)
    folder.mkdir(parents=True, exist_ok=True)
    expected = {f"chunk_{i:06d}.json" for i in range(0, len(items), chunk_size)}
    c.require(
        {p.name for p in folder.iterdir()} <= expected,
        "Unknown or interrupted partial chunk; preserve for review",
    )
    result = []
    reused = 0
    for start in range(0, len(items), chunk_size):
        group = items[start : start + chunk_size]
        keys = [r["key"] for r in group]
        path = folder / f"chunk_{start:06d}.json"
        if path.exists():
            record = c.read(path)
            c.require(
                record["stamp"] == stamp
                and record["keys"] == keys
                and (record["rows_sha256"] == c.signature(record["rows"])),
                "Changed chunk or input identity",
            )
            c.require(
                [r["key"] for r in record["rows"]] == keys, "Chunk output omitted/reordered input"
            )
            values = record["rows"]
            reused += len(values)
        else:
            values = [make(i, row) for i, row in enumerate(group, start)]
            c.require([r["key"] for r in values] == keys, "Producer changed input identities")
            c.save(
                path,
                {"stamp": stamp, "keys": keys, "rows": values, "rows_sha256": c.signature(values)},
            )
        result.extend(values)
        if progress:
            progress(len(result), len(items), reused)
    return (
        result,
        {
            "items": len(result),
            "reused_rows": reused,
            "chunks": len(expected),
            "hashes": {c.rel(p): c.sha(p) for p in sorted(folder.iterdir())},
        },
    )

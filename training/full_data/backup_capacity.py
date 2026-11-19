from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid
from training.full_data.backup import volume_id
from training.full_data import common as c


def measure(first, second):
    volumes = [
        dict(
            directory=str(path.resolve()),
            volume_id=volume_id(path),
            free_bytes=shutil.disk_usage(path).free,
        )
        for path in (first, second)
    ]
    c.require(
        volumes[0]["volume_id"] != volumes[1]["volume_id"],
        "Choose two different backup volumes.",
    )
    return dict(checked_utc=datetime.now(timezone.utc).isoformat(), volumes=volumes)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    parser.add_argument("receipt", type=Path)
    args = parser.parse_args()
    c.require(
        not args.receipt.exists(),
        "Use a new receipt filename; preserve previous measurements.",
    )
    value = measure(args.first, args.second)
    c.save(args.receipt, value)
    for volume in value["volumes"]:
        print(volume["directory"], f"{volume['free_bytes']/1e9:.2f} GB free")
    print("Upload this receipt as /inputs/backup_capacity.json before the final fits.")

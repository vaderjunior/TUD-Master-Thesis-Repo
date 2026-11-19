import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import zipfile


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unpack(source, destination, expected):
    if digest(source) != expected:
        raise RuntimeError("ZIP checksum differs.")
    if destination.exists():
        raise RuntimeError(
            "Destination already exists. Keep it and inspect it; do not overwrite."
        )
    if destination.name != "sq2_full":
        raise RuntimeError("Use the prepared sq2_full destination.")
    destination = destination.resolve()
    with zipfile.ZipFile(source) as z:
        entries = z.infolist()
        names = [v.filename for v in entries]
        if len(names) != len(set(names)):
            raise RuntimeError("Duplicate ZIP member.")
        manifest = json.loads(z.read("manifest.json"))
        if set(names) != set(manifest["files"]) | {"manifest.json"}:
            raise RuntimeError("ZIP inventory differs.")
        for info in entries:
            name = PurePosixPath(info.filename)
            if (
                name.is_absolute()
                or any(x in ("", "..", ".") for x in info.filename.split("/"))
                or "\\" in info.filename
                or ":" in info.filename
                or info.is_dir()
                or (info.external_attr >> 16) & 0o170000 == 0o120000
            ):
                raise RuntimeError("Unsafe ZIP member.")
        if sum(v.file_size for v in entries) > 1_000_000_000:
            raise RuntimeError("Bundle exceeds its prepared bound.")
        destination.mkdir(parents=True)
        for info in entries:
            path = destination / info.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with z.open(info) as src, path.open("xb") as dst:
                while block := src.read(1024 * 1024):
                    dst.write(block)
                dst.flush()
                os.fsync(dst.fileno())
            if (
                info.filename != "manifest.json"
                and digest(path) != manifest["files"][info.filename]
            ):
                raise RuntimeError(
                    "Extracted file differs. Keep the partial directory."
                )
    print("Bundle extracted and verified. No training started.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("zip", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("sha256")
    args = parser.parse_args()
    unpack(args.zip, args.destination, args.sha256)

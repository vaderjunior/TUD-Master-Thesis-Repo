import base64
import ctypes
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request
import uuid
from urllib.parse import quote
from training.full_data import common as c
from training.full_data import plan as p
from training.full_data import runtime as r
from training.full_data.archive import verify_part
from training.full_data.archive import validate_copies

HOST = "https://ajay-thesis-files.srv.peasec.de"


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise RuntimeError(
            "Redirect refused; credentials stay on the file-service host."
        )


def volume_id(path):
    path = Path(path).resolve()
    c.require(path.is_dir(), "Create the local backup directory first.")
    if os.name == "nt":
        mount = ctypes.create_unicode_buffer(32768)
        guid = ctypes.create_unicode_buffer(32768)
        c.require(
            ctypes.windll.kernel32.GetVolumePathNameW(str(path), mount, len(mount)),
            "Cannot identify the backup volume.",
        )
        c.require(
            ctypes.windll.kernel32.GetVolumeNameForVolumeMountPointW(
                mount.value, guid, len(guid)
            ),
            "Cannot identify the backup volume GUID.",
        )
        return guid.value
    return str(path.stat().st_dev)


def fetch(opener, auth, url, start=0):
    c.require(
        url.startswith(HOST + "/dav/results/sq2_full_20261029_v1/archives/"),
        "Unexpected download URL.",
    )
    headers = {"Authorization": "Basic " + auth}
    if start:
        headers["Range"] = f"bytes={start}-"
    response = opener.open(urllib.request.Request(url, headers=headers), timeout=60)
    c.require(response.geturl() == url, "Unexpected final URL.")
    return response


def download(opener, auth, url, destination, expected):
    if destination.exists():
        c.require(
            destination.stat().st_size == expected["bytes"]
            and c.sha(destination) == expected["sha256"],
            "Existing local archive differs.",
        )
        return
    partial = destination.with_suffix(destination.suffix + ".partial")
    identity = partial.with_suffix(partial.suffix + ".json")
    c.save(identity, dict(url=url, bytes=expected["bytes"], sha256=expected["sha256"]))
    start = partial.stat().st_size if partial.exists() else 0
    c.require(start <= expected["bytes"], "Partial archive exceeds its declared size.")
    c.require(
        shutil.disk_usage(destination.parent).free
        >= expected["bytes"] - start + 12_000_000_000,
        "Local download drive needs more free space.",
    )
    if start < expected["bytes"]:
        with fetch(opener, auth, url, start) as response:
            if start:
                c.require(
                    response.status == 206
                    and response.headers.get("Content-Range")
                    == f"bytes {start}-{expected['bytes']-1}/{expected['bytes']}",
                    "Server did not accept an exact ranged resume; preserve the partial file.",
                )
            else:
                c.require(response.status == 200, "Unexpected download status.")
            length = response.headers.get("Content-Length")
            c.require(
                length is not None and int(length) == expected["bytes"] - start,
                "Unexpected download length.",
            )
            with partial.open("ab" if start else "xb") as stream:
                total = start
                while True:
                    block = response.read(
                        min(1024 * 1024, expected["bytes"] - total + 1)
                    )
                    if not block:
                        break
                    c.require(
                        total + len(block) <= expected["bytes"],
                        "Response exceeds declared archive size.",
                    )
                    stream.write(block)
                    total += len(block)
                stream.flush()
                os.fsync(stream.fileno())
    c.require(
        partial.stat().st_size == expected["bytes"]
        and c.sha(partial) == expected["sha256"],
        "Downloaded archive checksum differs; keep the partial evidence.",
    )
    c.commit_file(partial, destination)


def copy_file(source, destination):
    if destination.exists():
        c.require(c.sha(destination) == c.sha(source), "The second copy differs.")
        return
    c.require(
        shutil.disk_usage(destination.parent).free
        >= source.stat().st_size + 12_000_000_000,
        "The second backup drive needs more free space.",
    )
    temp = destination.with_name(destination.name + "." + uuid.uuid4().hex + ".copying")
    with source.open("rb") as src, temp.open("xb") as dst:
        shutil.copyfileobj(src, dst, 1024 * 1024)
        dst.flush()
        os.fsync(dst.fileno())
    c.require(
        c.sha(temp) == c.sha(source),
        "Second copy checksum differs; preserve the partial copy.",
    )
    c.commit_file(temp, destination)


def verify_copies(pool, manifest, first, second):
    value = c.read(manifest)
    copies = []
    c.require(
        value["pool"] == pool and value["status"] == "complete",
        "Wrong archive manifest.",
    )
    ids = [volume_id(first), volume_id(second)]
    c.require(
        ids[0] != ids[1],
        "Use two different storage volumes, not two folders on one drive.",
    )
    for directory, vid in zip((first, second), ids):
        parts = {}
        for name, record in value["parts"].items():
            path = r.inside(directory, name)
            verify_part(path, record)
            parts[name] = dict(bytes=record["bytes"], sha256=record["sha256"])
        copies.append(
            dict(
                volume_id=vid,
                directory=str(directory.resolve()),
                parts=parts,
                all_members_verified=True,
            )
        )
    result = dict(pool=pool, archive_manifest_sha256=c.sha(manifest), copies=copies)
    validate_copies(result, value, c.sha(manifest))
    return result


def run(pool, first, second):
    c.require(pool in p.NAMES, "Unknown pool.")

    c.require(
        volume_id(first) != volume_id(second),
        "Two different backup volumes are required.",
    )
    first = first / pool
    second = second / pool
    first.mkdir(exist_ok=True)
    second.mkdir(exist_ok=True)
    password = getpass.getpass("Password for ajay on the file service: ")
    auth = base64.b64encode(("ajay:" + password).encode("utf8")).decode("ascii")
    del password
    opener = urllib.request.build_opener(NoRedirect())
    base = (
        HOST
        + "/dav/results/sq2_full_20261029_v1/archives/"
        + quote(pool, safe="")
        + "/"
    )
    manifest = first / "complete.json"
    with fetch(opener, auth, base + "complete.json") as response:
        c.require(response.status == 200, "Archive manifest request failed.")
        raw = response.read(100_000_001)
        c.require(len(raw) <= 100_000_000, "Unexpectedly large manifest.")
    value = json.loads(raw)
    c.require(
        value["pool"] == pool and value["status"] == "complete",
        "Wrong archive manifest.",
    )
    c.publish(manifest, raw)
    c.publish(second / "complete.json", raw)
    for name, record in value["parts"].items():
        c.require(
            name == Path(name).name and name.endswith(".zip"), "Invalid archive name."
        )
        path = r.inside(first, name)
        download(opener, auth, base + quote(name, safe=""), path, record)
        verify_part(path, record)
        copy_file(path, r.inside(second, name))
        print("Downloaded and copied:", name, flush=True)
    del auth
    receipt = verify_copies(pool, manifest, first, second)
    c.save(first / "backup_receipt.json", receipt)
    c.save(second / "backup_receipt.json", receipt)
    print(
        "Two verified copies are ready. Upload only backup_receipt.json to /inputs/backup_receipts/"
        + pool
        + "/ ."
    )
    print("This helper does not delete or release server files.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("pool", choices=p.NAMES)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    run(args.pool, args.first, args.second)

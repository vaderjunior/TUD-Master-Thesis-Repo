from pathlib import Path
import uuid
import zipfile
from training.full_data import common as c
from training.full_data import plan as p
from training.full_data import runtime as r

PART_LIMIT = 4_000_000_000


def archive_root(pool):
    c.require(pool in p.NAMES, "Unknown pool.")
    return r.OUTPUT / "archives" / pool


def verify_part(path, record):
    c.require(
        path.stat().st_size == record["bytes"] and c.sha(path) == record["sha256"],
        "Archive checksum differs.",
    )
    import hashlib

    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        c.require(
            len(names) == len(set(names)) and set(names) == set(record["files"]),
            "Archive member roster differs.",
        )
        for name, spec in record["files"].items():
            r.inside(Path("/safe-root"), name)
            info = z.getinfo(name)
            c.require(
                not info.is_dir()
                and info.file_size == spec["bytes"]
                and info.compress_type == zipfile.ZIP_STORED,
                "Unexpected archive member.",
            )
            c.require(
                (info.external_attr >> 16) & 0o170000 != 0o120000, "Archived symlink."
            )
            h = hashlib.sha256()
            with z.open(name) as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    h.update(block)
            c.require(h.hexdigest() == spec["sha256"], "Archived member differs.")


def inventory(pool):
    audit = r.OUTPUT / "audits" / f"{pool}_complete.json"
    value = c.read(audit)
    c.require(
        value["status"] == "pass" and value["pool"] == pool,
        "The pool must pass its final audit first.",
    )
    r.verify_tree(r.OUTPUT, value["outputs"])
    selected = c.read(r.OUTPUT / "selections" / f"{pool}.json")
    r.verify_tree(r.OUTPUT, selected["search_receipts"])
    paths = [audit, r.OUTPUT / "selections" / f"{pool}.json"]
    for stem in ("jobs", "models", "inference", "predictions"):
        paths.extend(r.files(r.OUTPUT / stem / pool))
    return {
        path.relative_to(r.OUTPUT).as_posix(): dict(
            bytes=path.stat().st_size, sha256=c.sha(path)
        )
        for path in sorted(paths)
    }


def groups(items, limit=PART_LIMIT):
    output = []
    group = {}
    size = 0
    for name, info in sorted(items.items()):

        cost = info["bytes"] + 4096 + len(name.encode("utf8")) * 2
        c.require(cost < limit, "One file exceeds the archive part bound.")
        if group and size + cost > limit:
            output.append(group)
            group = {}
            size = 0
        group[name] = info
        size += cost
    if group:
        output.append(group)
    return output


def build(pool, session=None):
    root = archive_root(pool)
    if (root / "complete.json").exists():
        verify(pool)
        return
    entries = inventory(pool)
    c.save(
        root / "inventory.json",
        dict(pool=pool, plan_sha256=c.sha(p.PACKAGE / "plan.json"), files=entries),
    )
    records = {}
    for index, items in enumerate(groups(entries)):
        name = f"part_{index:03d}.zip"
        path = root / name
        receipt = root / (name + ".json")
        if receipt.exists():
            rec = c.read(receipt)
            c.require(rec["files"] == items, "Archive partition changed.")
            verify_part(path, rec)
        else:
            r.deadline_guard(session)
            r.storage(sum(x["bytes"] + 8192 for x in items.values()))
            if not path.exists():
                temp = root / (name + "." + uuid.uuid4().hex + ".writing")
                with zipfile.ZipFile(
                    temp, "x", compression=zipfile.ZIP_STORED, allowZip64=True
                ) as z:
                    for rel, spec in items.items():
                        r.deadline_guard(session)
                        src = r.inside(r.OUTPUT, rel)
                        c.require(
                            src.stat().st_size == spec["bytes"]
                            and c.sha(src) == spec["sha256"],
                            "Source changed before archive.",
                        )
                        z.write(src, rel)

                import os

                with temp.open("r+b") as synced:
                    os.fsync(synced.fileno())
                c.commit_file(temp, path)
            rec = dict(bytes=path.stat().st_size, sha256=c.sha(path), files=items)
            c.require(
                rec["bytes"] < PART_LIMIT, "Archive part exceeds its download bound."
            )
            verify_part(path, rec)
            c.save(receipt, rec)
        records[name] = rec
    c.save(
        root / "complete.json",
        dict(
            status="complete",
            pool=pool,
            plan_sha256=c.sha(p.PACKAGE / "plan.json"),
            inventory_sha256=c.sha(root / "inventory.json"),
            parts=records,
        ),
    )
    verify(pool)


def verify(pool):
    root = archive_root(pool)
    value = c.read(root / "complete.json")
    c.require(
        value["pool"] == pool
        and value["plan_sha256"] == c.sha(p.PACKAGE / "plan.json")
        and value["inventory_sha256"] == c.sha(root / "inventory.json"),
        "Archive identity changed.",
    )
    all_files = {}
    for name, rec in value["parts"].items():
        c.require(
            name == Path(name).name and name.endswith(".zip"), "Invalid part name."
        )
        c.require(not set(all_files) & set(rec["files"]), "Duplicate archived member.")
        all_files.update(rec["files"])
    c.require(
        all_files == c.read(root / "inventory.json")["files"],
        "Archive coverage differs.",
    )
    if (root / "released.json").exists():
        verify_release(pool)
    else:
        for name, rec in value["parts"].items():
            verify_part(root / name, rec)
    return value


def validate_copies(value, archive_value, manifest_sha):
    c.require(
        value["archive_manifest_sha256"] == manifest_sha
        and value["pool"] == archive_value["pool"],
        "Backup receipt belongs to another archive.",
    )
    copies = value["copies"]
    c.require(
        len(copies) == 2 and copies[0]["volume_id"] != copies[1]["volume_id"],
        "Two distinct storage volumes are required.",
    )
    expected = {
        name: dict(bytes=v["bytes"], sha256=v["sha256"])
        for name, v in archive_value["parts"].items()
    }
    for copy in copies:
        c.require(
            copy["parts"] == expected
            and copy["all_members_verified"] is True
            and copy["volume_id"]
            and copy["directory"],
            "Incomplete or changed backup receipt.",
        )


def verify_release(pool):
    root = archive_root(pool)
    release = c.read(root / "released.json")
    value = c.read(root / "complete.json")
    intent = c.read(root / "release_intent.json")
    c.require(
        release["intent_sha256"] == c.sha(root / "release_intent.json")
        and release["archive_manifest_sha256"] == c.sha(root / "complete.json"),
        "Release binding differs.",
    )
    validate_copies(
        c.read(root / "backup_receipt.json"), value, c.sha(root / "complete.json")
    )
    c.require(
        intent["backup_sha256"] == c.sha(root / "backup_receipt.json"),
        "Backup binding differs.",
    )
    for name in intent["remove"]:
        c.require(
            not r.inside(r.OUTPUT, name).exists(), "A released file has reappeared."
        )


def release(pool, receipt_path):

    root = archive_root(pool)
    value = c.read(root / "complete.json")
    backup = c.read(receipt_path)
    validate_copies(backup, value, c.sha(root / "complete.json"))
    if (root / "released.json").exists():
        verify_release(pool)
        return
    files = c.read(root / "inventory.json")["files"]
    remove = {}
    for name, spec in files.items():
        if name.startswith("models/" + pool + "/"):
            r.inside(r.OUTPUT / "models" / pool, name[len("models/" + pool + "/") :])
            remove[name] = spec["sha256"]
    for name, rec in value["parts"].items():
        remove[(root / name).relative_to(r.OUTPUT).as_posix()] = rec["sha256"]
    c.require(
        remove and any(x.startswith("models/") for x in remove),
        "No completed model files to release.",
    )
    if not (root / "release_intent.json").exists():
        verify(pool)
        for name, digest in remove.items():
            c.require(
                c.sha(r.inside(r.OUTPUT, name)) == digest, "Release source differs."
            )
        c.save(root / "backup_receipt.json", backup)
        c.save(
            root / "release_intent.json",
            dict(
                archive_manifest_sha256=c.sha(root / "complete.json"),
                backup_sha256=c.sha(root / "backup_receipt.json"),
                remove=remove,
            ),
        )
    intent = c.read(root / "release_intent.json")
    c.require(
        intent["remove"] == remove
        and intent["archive_manifest_sha256"] == c.sha(root / "complete.json")
        and intent["backup_sha256"] == c.sha(root / "backup_receipt.json")
        and c.read(root / "backup_receipt.json") == backup,
        "Changed release intent.",
    )

    for name, digest in remove.items():
        path = r.inside(r.OUTPUT, name)
        if path.exists():
            c.require(c.sha(path) == digest, "File changed during release.")
            path.unlink()
    c.save(
        root / "released.json",
        dict(
            archive_manifest_sha256=c.sha(root / "complete.json"),
            intent_sha256=c.sha(root / "release_intent.json"),
            released_files=len(remove),
        ),
    )
    verify_release(pool)


def pending():
    return [
        pool
        for pool in p.NAMES
        if (archive_root(pool) / "complete.json").exists()
        and not (archive_root(pool) / "released.json").exists()
    ]

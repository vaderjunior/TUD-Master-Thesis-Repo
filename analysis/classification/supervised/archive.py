from src.hsrag.paths import PROJECT_ROOT
from collections import OrderedDict
from contextlib import contextmanager
import ctypes
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import re
import socket
import stat
import unicodedata
from unittest.mock import patch
import zipfile

import numpy as np

ROOT = PROJECT_ROOT
PACKAGE = ROOT / "deploy/full_training"
LINEAR = ROOT / "experiments/astra/sq2_stage_b20_20261021_v1"
BACKUP = Path("E:/SQ2_Backup/sq2_full_20261029_v1")
REMAINING = ROOT / "artifacts/sq2_remaining_download_20261102_v1"
PLAN_SHA = "0e1c920713aac9681ca6381ff306d27538182148911f4eb5357aca76860abba5"
PACKAGE_SHA = "01720f9f51402da9f9dd4bc3b35d883f649a4926171e40ce2e92f28039e4414a"
LINEAR_PLAN_SHA = "8e8e0742781af7df37e2b9d784c97920cc2dd7bef1ed2366017507bb98a43217"
RECOVERY_SHA = "f85be2ff42751dbfa31646703a8c51aa406a011abaf140665f0c14cd5eb7c24b"
DOWNLOAD_PLAN_SHAS = (
    "254cf831d1151e44b70e43aa4831cf011c9a55899e32d454b88f2ca3ddc0874e",
    "6aa2bbaf0b1d3469b58656d70fcbbe774a22cc45e6c17cfed0b35edc3a10adc3",
    "28c09ef9462d29cc8cb50b43d3782ff24c7db78198a470b27c4334a678fb16b2",
)
INVOCATION = "7d732de36c42465981a0120b7502130d"
HEADS = ("hate", "hate_type", "target_group", "severity")
FAMILIES = ("fixed_unweighted", "fixed_balanced", "tuned")
SEEDS = (10, 11, 12, 13, 14)
POLICIES = ("default", "selected")
POOLS = tuple(
    f"uniform_{size}_s{seed}"
    for size in ("compact", "1000", "5000")
    for seed in (42, 43, 44)
)
PRIMARY = (
    "en_test/hate",
    "en_test/hate_type",
    "larger_dev/target_group",
    "larger_dev/severity",
)
FIELDS = dict(
    hate="gate",
    hate_type="hate_types",
    target_group="target_groups",
    severity="severity",
)
MAX_READ = 64 * 1024 * 1024


class IntegrityError(RuntimeError):
    pass


def require(ok, message):
    if not ok:
        raise IntegrityError(message)


def no_network(*args, **kwargs):
    raise IntegrityError("This reader is offline.")


@contextmanager
def offline():
    with patch.object(socket.socket, "connect", no_network), patch.object(
        socket.socket, "connect_ex", no_network
    ), patch.object(socket, "create_connection", no_network):
        yield


def encoded(value):
    return (
        json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n"
    ).encode("utf8")


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def signature(value):
    return digest(encoded(value))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "Duplicate JSON field: " + key)
        value[key] = item
    return value


def parsed(raw):
    return json.loads(raw, object_pairs_hook=unique_object)


def read(path):
    return parsed(Path(path).read_bytes())


def relative(name):
    require(
        isinstance(name, str)
        and name
        and "\\" not in name
        and ":" not in name
        and not PurePosixPath(name).is_absolute()
        and all(
            re.fullmatch(r"[A-Za-z0-9_.-]+", p) and p not in (".", "..")
            for p in name.split("/")
        ),
        "Unsafe relative file name: " + str(name),
    )
    return name


def ordinary_path(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        if item.exists():
            info = item.lstat()
            require(
                not stat.S_ISLNK(info.st_mode)
                and not getattr(info, "st_file_attributes", 0) & 0x400,
                "A source path is redirected: " + str(item),
            )
    return path


def volume_id(path):
    volume = ctypes.create_unicode_buffer(1024)
    require(
        ctypes.windll.kernel32.GetVolumeNameForVolumeMountPointW(
            Path(path).anchor, volume, 1024
        ),
        "Backup volume unavailable.",
    )
    return volume.value


def checked_file(path, expected, bindings=None):
    path = ordinary_path(path)
    require(path.is_file() and sha(path) == expected, "Changed source: " + str(path))
    if bindings is not None:
        bindings[str(path)] = expected
    return path


def arrays_from_bytes(raw):
    require(len(raw) <= MAX_READ, "Array payload too large for this reader.")
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            info = z.infolist()
            require(
                len(info) == len({x.filename for x in info})
                and sum(x.file_size for x in info) <= MAX_READ,
                "Duplicate/oversized array bundle.",
            )
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            result = {name: z[name] for name in z.files}
    except (ValueError, zipfile.BadZipFile) as exc:
        raise IntegrityError("Invalid array bundle.") from exc
    for value in result.values():
        require(
            value.dtype.kind in "biuf" and np.isfinite(value).all(),
            "Invalid array values.",
        )
        value.setflags(write=False)
    return result


def probabilities(head, logits):
    require(head in HEADS, "Unknown head.")
    z = np.asarray(logits, dtype=np.float64)
    require(
        z.ndim == 2 and np.isfinite(z).all(), "Nonfinite or wrong-dimensional logits."
    )
    if head in ("hate_type", "target_group"):
        return np.exp(-np.logaddexp(0.0, -z))
    value = np.exp(z - z.max(1, keepdims=True))
    return value / value.sum(1, keepdims=True)


def decode(probs, selected, policy):
    require(
        policy in POLICIES and set(probs) == set(HEADS) and set(selected) == set(HEADS),
        "Invalid decision scope.",
    )
    result = {}
    rows = len(probs["hate"])
    for head in HEADS:
        p = np.asarray(probs[head])
        require(
            p.ndim == 2
            and len(p) == rows
            and np.isfinite(p).all()
            and ((p >= 0) & (p <= 1)).all(),
            "Invalid probabilities.",
        )
        cutoff = (
            None
            if head == "severity"
            else 0.5 if policy == "default" else selected[head]["selected"]["threshold"]
        )
        if head != "severity":
            require(
                isinstance(cutoff, (int, float)) and 0 < cutoff < 1, "Invalid cutoff."
            )
        if head == "hate":
            require(p.shape[1] == 2, "Hate columns must be negative, positive.")
            gate = p[:, 1] > cutoff
            raw = np.column_stack((~gate, gate))
        elif head == "severity":
            raw = np.eye(p.shape[1], dtype=bool)[p.argmax(1)]
        else:
            raw = p >= cutoff
        gated = raw.copy()
        if head != "hate":
            gated[~gate] = False
        result[head] = gated.astype(np.uint8)
        result[head + "_ungated"] = raw.astype(np.uint8)
    return result


def validate_decisions(values, labels, rows):
    require(
        set(values) == {h + s for h in HEADS for s in ("", "_ungated")},
        "Missing/extra decision arrays.",
    )
    for head in HEADS:
        for suffix in ("", "_ungated"):
            v = values[head + suffix]
            require(
                v.dtype == np.uint8
                and v.shape == (rows, len(labels[head]))
                and ((v == 0) | (v == 1)).all(),
                "Invalid decision matrix: " + head + suffix,
            )
    require(
        np.array_equal(values["hate"], values["hate_ungated"])
        and (values["hate"].sum(1) == 1).all(),
        "Invalid hate decisions.",
    )
    gate = values["hate"][:, 1].astype(bool)
    require(
        (values["severity_ungated"].sum(1) == 1).all(),
        "Severity must use one argmax label.",
    )
    for head in HEADS[1:]:
        expected = values[head + "_ungated"].copy()
        expected[~gate] = 0
        require(np.array_equal(expected, values[head]), "Wrong hate gate: " + head)


def validate_inputs(inputs, expected_rows):
    require(
        len(inputs) == len({x["key"] for x in inputs}) == expected_rows,
        "Wrong input union or duplicate identity.",
    )
    require(
        all(
            isinstance(x["key"], str)
            and isinstance(x["text"], str)
            and digest(x["text"].encode("utf8")) == x["exact_text_sha256"]
            for x in inputs
        ),
        "Inference text identity differs.",
    )


def scope_positions(inputs, rows):
    require(
        len(rows)
        == len({x["key"] for x in rows})
        == len({x["source_id"] for x in rows})
        == len({x["position"] for x in rows}),
        "Duplicate scope identity/position.",
    )
    require(
        all(
            isinstance(x["source_id"], str)
            and type(x["position"]) is int
            and 0 <= x["position"] < len(inputs)
            and inputs[x["position"]]["key"] == x["key"]
            for x in rows
        ),
        "Scope position differs from inference order.",
    )
    return {x["source_id"]: x["position"] for x in rows}


class ArchiveStore:

    def __init__(self, directory, manifest_raw, expected_sha, pool):
        require(
            digest(manifest_raw) == expected_sha, "Archive manifest checksum differs."
        )
        self.directory = ordinary_path(directory)
        self.manifest = parsed(manifest_raw)
        m = self.manifest
        require(
            m["pool"] == pool and m["plan_sha256"] == PLAN_SHA,
            "Archive belongs to a different pool/plan.",
        )
        split = m.get("format") == "sq2-split-backup-v1"
        require(
            m["status"] == ("ready_for_download" if split else "complete"),
            "Archive is incomplete.",
        )
        if split:
            require(
                m["local_copy_count"] == 1
                and m["server_originals_retained"]
                and not m["server_deletion_authorized"],
                "Wrong split-backup scope.",
            )
        self.pool = pool
        self.entries = {}
        self.parts = {}
        self.opened = OrderedDict()
        self.read_bindings = {}
        self.bytes_read = 0
        for name, spec in m.get("weights", {}).items():
            relative(name)
            require(
                name.startswith(f"models/{pool}/")
                and name.endswith("/model.safetensors"),
                "Invalid direct weight.",
            )
            self.entries[name] = (None, spec)
        require(bool(m["parts"]), "Empty archive parts.")
        for part, record in m["parts"].items():
            require(
                relative(part) == Path(part).name and part.endswith(".zip"),
                "Invalid archive part name.",
            )
            self.parts[part] = record
            path = ordinary_path(self.directory / part)
            require(
                path.is_file() and path.stat().st_size == record["bytes"],
                "Missing or truncated archive part.",
            )
            for name, spec in record["files"].items():
                relative(name)
                require(name not in self.entries, "Duplicate archived file: " + name)
                if pool != "study_records":
                    require(
                        name
                        in (f"audits/{pool}_complete.json", f"selections/{pool}.json")
                        or any(
                            name.startswith(f"{stem}/{pool}/")
                            for stem in ("jobs", "models", "inference", "predictions")
                        ),
                        "Archive contains a different pool.",
                    )
                self.entries[name] = (part, spec)
        for _, spec in self.entries.values():
            require(
                type(spec["bytes"]) is int
                and spec["bytes"] >= 0
                and re.fullmatch("[0-9a-f]{64}", spec["sha256"]),
                "Invalid file identity.",
            )
        inventory = dict(
            pool=pool,
            plan_sha256=PLAN_SHA,
            files={k: v[1] for k, v in self.entries.items()},
        )
        require(
            signature(inventory) == m["inventory_sha256"],
            "Archive inventory coverage differs.",
        )
        self.stamps = {
            part: (self.directory / part).stat().st_mtime_ns for part in self.parts
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for z in self.opened.values():
            z.close()
        self.opened.clear()

    def expected(self, name):
        relative(name)
        require(name in self.entries, "File is absent from the frozen archive: " + name)
        return self.entries[name][1]["sha256"]

    def _zip(self, part):
        path = ordinary_path(self.directory / part)
        require(
            path.stat().st_size == self.parts[part]["bytes"]
            and path.stat().st_mtime_ns == self.stamps[part],
            "Archive changed while being read.",
        )
        if part not in self.opened:
            z = zipfile.ZipFile(path)
            try:
                info = z.infolist()
                require(
                    len(info) == len({i.filename for i in info})
                    and {i.filename for i in info} == set(self.parts[part]["files"]),
                    "ZIP member roster differs.",
                )
                for item in info:
                    require(
                        not item.is_dir()
                        and not item.flag_bits & 1
                        and item.file_size
                        == self.parts[part]["files"][item.filename]["bytes"],
                        "ZIP entry differs.",
                    )
            except BaseException:
                z.close()
                raise
            self.opened[part] = z
            if len(self.opened) > 2:
                _, old = self.opened.popitem(last=False)
                old.close()
        self.opened.move_to_end(part)
        return self.opened[part]

    def raw(self, name):
        self.expected(name)
        part, spec = self.entries[name]
        require(
            not name.endswith("/model.safetensors"),
            "The analysis reader does not load model weights.",
        )
        require(
            part is not None and spec["bytes"] <= MAX_READ,
            "Unsupported or oversized analysis file.",
        )
        raw = self._zip(part).read(name)
        require(
            len(raw) == spec["bytes"] and digest(raw) == spec["sha256"],
            "Saved member checksum differs: " + name,
        )
        self.read_bindings[name] = spec["sha256"]
        self.bytes_read += len(raw)
        return raw

    def json(self, name):
        return parsed(self.raw(name))

    def array(self, name):
        try:
            value = np.load(io.BytesIO(self.raw(name)), allow_pickle=False)
        except ValueError as exc:
            raise IntegrityError("Invalid saved array.") from exc
        require(
            isinstance(value, np.ndarray)
            and value.dtype.kind in "biuf"
            and np.isfinite(value).all(),
            "Invalid saved array.",
        )
        value.setflags(write=False)
        return value

    def npz(self, name):
        return arrays_from_bytes(self.raw(name))


class GpuStudy:
    def __init__(self):
        self.bindings = {}
        checked_file(PACKAGE / "plan.json", PLAN_SHA, self.bindings)
        checked_file(PACKAGE / "manifest.json", PACKAGE_SHA, self.bindings)
        self.package_manifest = read(PACKAGE / "manifest.json")
        self.plan = read(PACKAGE / "plan.json")
        self.inputs = self.package_json("data/inference.json")
        validate_inputs(self.inputs, 19082)
        self.input_sha = signature(self.inputs)
        self.configs = {x["id"]: x for x in self.package_json("data/configs.json")}
        require(
            len(self.configs) == 42
            and {x["learning_rate"] for x in self.configs.values()}
            == {5e-6, 2e-5, 5e-5},
            "Wrong full grid.",
        )
        checked_file(
            ROOT / "artifacts/sq2_recovery_20261030_v1/recovery_policy.json",
            RECOVERY_SHA,
            self.bindings,
        )
        self.archives = {}
        self.plans = []
        folders = [
            ROOT / f"artifacts/sq2_archive_download_20261101_v{v}" for v in (1, 2)
        ] + [REMAINING]
        for folder, expected in zip(folders, DOWNLOAD_PLAN_SHAS):
            checked_file(folder / "download_plan.json", expected, self.bindings)
            plan = read(folder / "download_plan.json")
            require(
                plan["training_plan_sha256"] == PLAN_SHA,
                "Download study identity differs.",
            )
            marker = read(folder / "download_complete.json")
            require(
                marker["download_plan_sha256"] == expected,
                "Wrong download completion plan.",
            )
            self.bindings[str(folder / "download_complete.json")] = sha(
                folder / "download_complete.json"
            )
            split = folder == REMAINING
            require(
                marker["status"]
                == (
                    "remaining_results_verified_on_E"
                    if split
                    else "two_pools_two_verified_volumes"
                ),
                "Download verification is incomplete.",
            )
            require(
                volume_id(BACKUP) == plan["destinations"][0]["volume_id"],
                "The E: backup drive changed.",
            )
            for pool in plan["pools"]:
                require(
                    pool not in self.archives, "A pool appears in two download batches."
                )
                spec = plan["manifests"][pool]
                checked_file(folder / spec["path"], spec["sha256"], self.bindings)
                local_manifest = (
                    BACKUP / pool / ("manifest.json" if split else "complete.json")
                )
                checked_file(local_manifest, spec["sha256"], self.bindings)
                m = read(local_manifest)
                checked_file(
                    BACKUP / pool / "inventory.json",
                    m["inventory_sha256"],
                    self.bindings,
                )
                receipt_name = (
                    "download_receipt.json" if split else "backup_receipt.json"
                )
                local_receipt = folder / "receipts" / pool / receipt_name
                receipt_sha = sha(local_receipt)
                checked_file(BACKUP / pool / receipt_name, receipt_sha, self.bindings)
                self.bindings[str(local_receipt)] = receipt_sha
                receipt = read(local_receipt)
                if split:
                    require(
                        marker["receipts"][pool] == receipt_sha
                        and receipt["status"] == "one_local_copy_verified"
                        and receipt["all_files_and_members_verified"]
                        and receipt["manifest_sha256"] == spec["sha256"]
                        and receipt["inventory_sha256"] == m["inventory_sha256"],
                        "Incomplete split receipt.",
                    )
                else:
                    require(
                        receipt["archive_manifest_sha256"] == spec["sha256"]
                        and any(
                            x["all_members_verified"]
                            and Path(x["directory"]) == BACKUP / pool
                            for x in receipt["copies"]
                        ),
                        "Missing verified E: archive receipt.",
                    )
                self.archives[pool] = (local_manifest, spec["sha256"])
            self.plans.append(plan)
        require(
            set(self.archives) == set(POOLS) | {"study_records"},
            "Incomplete nine-pool archive set.",
        )
        with self.store("study_records") as store:
            self.selection_lock = store.json("selection_lock.json")
            self.exports = store.json("final_export_plan.json")
            name = f"invocations/{INVOCATION}/fitting_complete_archiving_pending.json"
            raw = store.raw(name)
            require(
                digest(raw) == self.plans[-1]["completion_sha256"],
                "Fitting completion identity differs.",
            )
            self.completion = parsed(raw)
            require(
                self.completion["status"]
                == "fits_and_predictions_audited_archiving_pending"
                and self.completion["models_total"]
                == self.exports["models_total"]
                == 530
                and set(self.exports["models"]) == set(POOLS),
                "GPU work is not complete.",
            )
            require(
                self.selection_lock["plan_sha256"]
                == self.exports["plan_sha256"]
                == PLAN_SHA,
                "Wrong selection/export plan.",
            )

    def package_json(self, name):
        checked_file(
            PACKAGE / name, self.package_manifest["files"][name], self.bindings
        )
        return read(PACKAGE / name)

    def store(self, pool):
        require(pool in self.archives, "Unknown archive pool.")
        path, expected = self.archives[pool]
        return ArchiveStore(BACKUP / pool, path.read_bytes(), expected, pool)

    def pool(self, pool):
        return GpuPool(self, pool)


class GpuPool:
    def __init__(self, study, pool):
        require(pool in POOLS, "Unknown training pool.")
        self.study, self.pool = study, pool
        self.store = study.store(pool)
        self.data = study.package_json(f"data/train/{pool}.json")
        self.labels = {h: self.data["heads"][h]["labels"] for h in HEADS}
        name = f"selections/{pool}.json"
        self.selection = self.store.json(name)
        require(
            self.store.expected(name) == study.selection_lock["selections"][name]
            and self.selection["plan_sha256"] == PLAN_SHA
            and self.selection["pool"] == pool
            and not self.selection["evaluation_gold_used"],
            "Selection binding differs.",
        )
        self.families = self.selection["families"]
        require(
            set(self.families) == set(FAMILIES)
            and all(set(h) == set(HEADS) for h in self.families.values()),
            "Missing model family/head.",
        )
        self.audit = self.store.json(f"audits/{pool}_complete.json")
        require(
            self.audit["status"] == "pass"
            and self.audit["pool"] == pool
            and not self.audit["evaluation_gold_used"]
            and self.audit["selection_sha256"] == self.store.expected(name)
            and self.store.expected(f"audits/{pool}_complete.json")
            == study.completion["pool_audits"][pool],
            "Pool audit differs.",
        )
        for name, expected in {
            **self.selection["search_receipts"],
            **self.audit["outputs"],
        }.items():
            require(
                self.store.expected(name) == expected,
                "Audited output binding differs: " + name,
            )
        expected_search = {
            f"jobs/{pool}/{h}/{cfg}/s{s}_f{fold}/complete.json"
            for h in HEADS
            for cfg in study.configs
            for s in (0, 1, 2)
            for fold in range(5)
        }
        require(
            set(self.selection["search_receipts"]) == expected_search,
            "Search receipt roster differs from full grid.",
        )
        jobs = {self.job(h, f, s)["id"] for f in FAMILIES for h in HEADS for s in SEEDS}
        require(
            sorted(jobs) == sorted(study.exports["models"][pool])
            and len(jobs) == self.audit["final_models"],
            "Final export roster differs.",
        )
        expected_outputs = {
            f"{stem}/{j}/complete.json" for j in jobs for stem in ("jobs", "inference")
        } | {
            f"predictions/{pool}/{f}_{p}_s{s}.npz"
            for f in FAMILIES
            for p in POLICIES
            for s in SEEDS
        }
        require(
            set(self.audit["outputs"]) == expected_outputs
            and self.audit["text_predictions"] == len(study.inputs) * 30,
            "Final audit output coverage differs.",
        )
        self.jobs = sorted(jobs)
        self.probability_cache = {}
        self.job_records = {}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.probability_cache.clear()
        self.store.__exit__(*args)

    def job(self, head, family, seed):
        require(
            head in HEADS and family in FAMILIES and seed in SEEDS,
            "Invalid neural setting.",
        )
        config = self.families[family][head]["selected"]["config"]
        require(config in self.study.configs, "Unknown selected configuration.")
        if family != "tuned":
            require(
                config == self.study.plan["fixed_configs"][family],
                "Fixed control configuration changed.",
            )
        return dict(
            id=f"{self.pool}/{head}/{config}/s{seed}",
            pool=self.pool,
            head=head,
            config=config,
            seed=seed,
            fold=None,
            mode="final",
        )

    def final_probabilities(self, job):
        ident = job["id"]
        if ident in self.probability_cache:
            return self.probability_cache[ident]
        source = f"jobs/{ident}/complete.json"
        rec = self.store.json(source)
        identity = dict(
            job=job,
            plan_sha256=PLAN_SHA,
            package_sha256=PACKAGE_SHA,
            gpu_profile="final_1x80",
        )
        require(
            rec["status"] == "complete"
            and rec["identity"] == identity
            and rec["state"]
            in ("trained", "initialization_only", "amp_initialization_only"),
            "Wrong completed fit.",
        )
        attempt = relative(rec["attempt"])
        require(
            str(PurePosixPath(attempt).parent) == f"jobs/{ident}/attempts",
            "Wrong canonical attempt.",
        )
        if "verification_amendment_sha256" in rec:
            require(
                rec["verification_amendment_sha256"] == RECOVERY_SHA,
                "Wrong recovery amendment.",
            )
        if rec["state"] == "amp_initialization_only":
            require(
                rec.get("verification_amendment_sha256") == RECOVERY_SHA,
                "Unbound initialization-only fit.",
            )
        for name, expected in rec["products"].items():
            require(
                self.store.expected(attempt + "/" + relative(name)) == expected,
                "Canonical fit product differs.",
            )
        require(
            {"started.json", "training.json"} <= set(rec["products"]),
            "Incomplete final fit products.",
        )
        model = "models/" + ident + "/" + PurePosixPath(attempt).name
        require(
            rec["model_directory"] == "/results/sq2_full_20261029_v1/" + model
            and {"model.safetensors", "config.json", "study_meta.json"}
            <= set(rec["model_products"]),
            "Wrong model export path.",
        )
        for name, expected in rec["model_products"].items():
            require(
                self.store.expected(model + "/" + relative(name)) == expected,
                "Model product binding differs.",
            )
        meta = self.store.json(model + "/study_meta.json")
        require(
            meta
            == dict(
                job=job,
                labels=self.labels[job["head"]],
                kind=self.data["heads"][job["head"]]["kind"],
                config=self.study.configs[job["config"]],
                pilot_only=False,
            ),
            "Wrong export metadata.",
        )
        name = f"inference/{ident}/complete.json"
        inf = self.store.json(name)
        require(
            inf["binding"]
            == dict(
                identity=identity,
                training_receipt_sha256=self.store.expected(source),
                input_sha256=self.study.input_sha,
                labels=self.labels[job["head"]],
                rows=len(self.study.inputs),
            ),
            "Inference is not aligned with its fit and input order.",
        )
        expected_products = {"logits.npy"} | {
            f"chunks/{start:06d}_{min(start+512,len(self.study.inputs)):06d}.{ext}"
            for start in range(0, len(self.study.inputs), 512)
            for ext in ("npy", "json")
        }
        require(
            set(inf["products"]) == expected_products, "Inference chunk roster differs."
        )
        for name, expected in inf["products"].items():
            require(
                self.store.expected(f"inference/{ident}/" + relative(name)) == expected,
                "Inference product differs.",
            )
        z = self.store.array(f"inference/{ident}/logits.npy")
        require(
            z.dtype == np.float32
            and z.shape == (len(self.study.inputs), len(self.labels[job["head"]])),
            "Wrong logits shape/type.",
        )
        probs = probabilities(job["head"], z)
        probs.setflags(write=False)
        self.probability_cache[ident] = probs
        self.job_records[ident] = dict(
            receipt_sha256=self.store.expected(source),
            attempt=attempt,
            model_directory=model,
            state=rec["state"],
            inference_sha256=self.store.expected(f"inference/{ident}/complete.json"),
            logits_sha256=self.store.expected(f"inference/{ident}/logits.npy"),
            recovery_amendment=rec.get("verification_amendment_sha256"),
        )
        return probs

    def predictions(self, family, policy, seed):
        require(
            family in FAMILIES and policy in POLICIES and seed in SEEDS,
            "Invalid neural variant.",
        )
        probs = {h: self.final_probabilities(self.job(h, family, seed)) for h in HEADS}
        saved = self.store.npz(f"predictions/{self.pool}/{family}_{policy}_s{seed}.npz")
        validate_decisions(saved, self.labels, len(self.study.inputs))
        expected = decode(probs, self.families[family], policy)
        require(
            all(np.array_equal(saved[k], expected[k]) for k in expected),
            "Saved decisions differ from logits and matching gate.",
        )
        return saved


class LinearStudy:
    def __init__(self, study):
        self.study = study
        self.bindings = {}
        checked_file(LINEAR / "plan.json", LINEAR_PLAN_SHA, self.bindings)
        plan = read(LINEAR / "plan.json")
        self.plan = plan
        certificate_path = (
            ROOT
            / "experiments/astra/sq2_server_full_preparation_20261029_v1/linear_reuse_check.json"
        )
        preflight = read(certificate_path.parent / "complete.json")
        key = certificate_path.relative_to(ROOT).as_posix()
        checked_file(certificate_path, preflight["outputs"][key], self.bindings)
        certificate = read(certificate_path)
        require(
            certificate["status"] == "pass"
            and certificate["old_plan_sha256"] == LINEAR_PLAN_SHA
            and certificate["new_fits"] == 0,
            "Wrong baseline reuse certificate.",
        )
        checked_file(
            LINEAR / "linear/complete.json",
            certificate["linear_receipt_sha256"],
            self.bindings,
        )
        self.complete = read(LINEAR / "linear/complete.json")
        require(
            self.complete["status"] == "linear_fits_and_text_only_predictions_complete"
            and self.complete["plan_sha256"] == LINEAR_PLAN_SHA,
            "Baseline predictions incomplete.",
        )
        self.outputs = self.complete["outputs"]
        for name in ("inference.json", "scopes.json", "endpoints.json"):
            path = LINEAR / "data" / name
            checked_file(
                path, plan["prepared"][path.relative_to(ROOT).as_posix()], self.bindings
            )
        require(
            read(LINEAR / "data/inference.json") == study.inputs,
            "Baseline evaluation order differs.",
        )
        for pool in POOLS:
            path = LINEAR / "data/train" / f"{pool}.json"
            checked_file(
                path, plan["prepared"][path.relative_to(ROOT).as_posix()], self.bindings
            )
            require(
                read(path) == study.package_json(f"data/train/{pool}.json"),
                "Baseline training budget differs.",
            )
        self.cache = {}

    def load(self, name):
        relative(name)
        require(
            name in self.outputs, "Baseline file lacks a completion binding: " + name
        )
        checked_file(ROOT / name, self.outputs[name], self.bindings)
        return ROOT / name

    def setting(self, pool, method, fold):
        require(
            pool in POOLS
            and (
                method == "bge"
                and fold in (42, 43, 44)
                or method == "frozen_roberta"
                and fold == 42
            ),
            "Invalid baseline setting.",
        )
        key = (pool, method, fold)
        if key in self.cache:
            return self.cache[key]
        prefix = (LINEAR / "linear" / pool).relative_to(ROOT).as_posix()
        receipt = read(self.load(f"{prefix}/{method}_fold{fold}_predictions.json"))
        require(
            receipt["pool"] == pool
            and receipt["method"] == method
            and receipt["fold_assignment"] == fold
            and receipt["plan_sha256"] == LINEAR_PLAN_SHA
            and receipt["rows"] == len(self.study.inputs)
            and receipt["evaluation_gold_used"] is False,
            "Wrong baseline prediction identity.",
        )
        source = read(self.load(receipt["source_receipt"]))
        if method == "frozen_roberta":
            require(
                source["identity"]["pool"] == pool
                and source["identity"]["fold_assignment"] == 42
                and source["identity"]["plan_sha256"] == LINEAR_PLAN_SHA,
                "Wrong frozen-encoder fit.",
            )
            fit = read(self.load(source["fit_path"]))
        else:
            require(
                source["pool"] == pool
                and source["fold_assignment"] == fold
                and source["plan_sha256"] == LINEAR_PLAN_SHA,
                "Wrong reused BGE fit.",
            )
            fit_name = f"experiments/astra/compact_pools_20261008_v1/jobs/{pool}_fold{fold}/fit.json"
            require(fit_name in source["outputs"], "Historical BGE fit not bound.")
            fit = read(self.load(fit_name))
            require(
                fit["job"]["pool"] == pool and fit["job"]["seed"] == fold,
                "Historical BGE identity differs.",
            )
        heads = fit["heads"]
        data = self.study.package_json(f"data/train/{pool}.json")
        for h in HEADS:
            require(
                heads[h]["labels"] == data["heads"][h]["labels"]
                and heads[h]["training_ids"]
                == [r["id"] for r in data["heads"][h]["rows"]],
                "Baseline labels or TRAIN rows differ.",
            )
        name = (
            (LINEAR / "predictions" / pool / f"{method}_fold{fold}_probabilities.npz")
            .relative_to(ROOT)
            .as_posix()
        )
        require(
            receipt["outputs"][name] == self.outputs[name],
            "Baseline probabilities lack identity binding.",
        )
        raw = arrays_from_bytes(self.load(name).read_bytes())
        require(set(raw) == set(HEADS), "Missing/extra baseline probability heads.")
        for h in HEADS:
            require(
                raw[h].shape
                == (
                    len(self.study.inputs),
                    1 if h == "hate" else len(heads[h]["labels"]),
                )
                and ((raw[h] >= 0) & (raw[h] <= 1)).all(),
                "Baseline probabilities invalid.",
            )
        gate = raw["hate"][:, 0]
        probs = {**raw, "hate": np.column_stack((1 - gate, gate))}
        self.cache[key] = (heads, probs, receipt)
        return self.cache[key]

    def predictions(self, pool, method, fold, policy):
        require(policy in POLICIES, "Unknown baseline cutoff policy.")
        heads, probs, receipt = self.setting(pool, method, fold)
        name = (
            (LINEAR / "predictions" / pool / f"{method}_fold{fold}_{policy}.npz")
            .relative_to(ROOT)
            .as_posix()
        )
        require(
            receipt["outputs"][name] == self.outputs[name],
            "Baseline decision binding differs.",
        )
        saved = arrays_from_bytes(self.load(name).read_bytes())
        validate_decisions(
            saved, {h: heads[h]["labels"] for h in HEADS}, len(self.study.inputs)
        )
        expected = decode(probs, heads, policy)
        require(
            all(np.array_equal(saved[k], expected[k]) for k in expected),
            "Baseline decisions use the wrong gate/cutoff.",
        )
        return saved

    def release_pool(self):
        self.cache.clear()


def normalized(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text).casefold()).strip()


def clean(value):
    if value is None or isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def gold_labels(row, head):
    value, gate = clean(row.get(FIELDS[head])), clean(row.get("gate"))
    if value is None or head != "hate" and gate is not True:
        return None
    if head == "hate":
        require(type(value) is bool, "Nonboolean hate annotation.")
        return {"hate" if value else "not_hate"}
    if head == "severity":
        return {value}
    require(isinstance(value, (list, tuple)), "Invalid multilabel annotation.")
    return set(value)


def endpoint_rows(records, endpoint):
    head = endpoint["dimension"]
    kept, targets = [], []
    for row in records:
        labels = gold_labels(row, head)
        if labels is None:
            continue
        require(labels <= set(endpoint["all_labels"]), "Unknown gold label.")
        kept.append(row)
        targets.append([x in labels for x in endpoint["all_labels"]])
    truth = np.asarray(targets, dtype=bool).reshape(
        len(kept), len(endpoint["all_labels"])
    )
    require(
        [str(x["id"]) for x in kept] == endpoint["scorable_ids"],
        "Gold eligibility/order differs.",
    )
    require(
        truth.sum(0).tolist()
        == [endpoint["gold_support"][x] for x in endpoint["all_labels"]],
        "Frozen gold support differs.",
    )
    return kept, truth


class SharedContext:
    def __init__(self, study, linear):
        self.inputs = study.inputs
        self.scopes = read(LINEAR / "data/scopes.json")
        self.endpoints = read(LINEAR / "data/endpoints.json")
        self.labels = {
            h: study.package_json(f"data/train/{POOLS[0]}.json")["heads"][h]["labels"]
            for h in HEADS
        }
        self.positions = {}
        self.bindings = {}
        for scope, rows in self.scopes.items():
            self.positions[scope] = scope_positions(self.inputs, rows)
        require(
            {k: len(v) for k, v in self.scopes.items()} == dict(core=970, broad=18275),
            "Wrong evaluation scope.",
        )
        self.records = {}
        broad_path = (
            ROOT
            / "experiments/astra/offline_closure_20261006_v1/evaluation_preparation/broad_roster.jsonl"
        )
        broad_plan = read(
            ROOT / "experiments/astra/compact_broad_20261008_v1/plan.json"
        )
        core_plan = read(ROOT / "experiments/astra/compact_bge_20261008_v1/plan.json")
        require(
            self.endpoints["broad"] == broad_plan["endpoints"]
            and self.endpoints["core"] == core_plan["endpoints"],
            "Endpoint contract differs.",
        )
        checked_file(
            broad_path,
            broad_plan["inputs"][broad_path.relative_to(ROOT).as_posix()],
            self.bindings,
        )
        broad = [
            parsed(line)
            for line in broad_path.read_bytes().splitlines()
            if line.strip()
        ]
        for name in self.endpoints["broad"]:
            rr = [
                row | {"id": row["key"]}
                for row in broad
                if row["phase"] == name.split("/")[0]
            ]
            self.records["broad", name] = rr
        import pandas as pd

        for name in self.endpoints["core"]:
            path = ROOT / f"data/processed/{name}.parquet"
            checked_file(
                path,
                core_plan["inputs"][path.relative_to(ROOT).as_posix()],
                self.bindings,
            )
            self.records["core", name] = [
                {k: clean(v) for k, v in row.items()}
                for row in pd.read_parquet(path).to_dict("records")
            ]
        self.views = {}
        for (scope, name), records in self.records.items():
            endpoint = self.endpoints[scope][name]
            for row in records:
                source_id = str(row["id"])
                require(
                    source_id in self.positions[scope]
                    and self.inputs[self.positions[scope][source_id]]["text"]
                    == row["text"],
                    "Gold text is not the input that was classified.",
                )
            kept, truth = endpoint_rows(records, endpoint)
            labels = (
                ["not_hate", "hate"]
                if endpoint["dimension"] == "hate"
                else self.labels[endpoint["dimension"]]
            )
            require(
                set(endpoint["all_labels"]) <= set(labels)
                and endpoint["macro_labels"]
                == [
                    x
                    for x in endpoint["all_labels"]
                    if endpoint["gold_support"][x] >= 10
                ],
                "Evaluation label support/order differs.",
            )
            indices = [labels.index(x) for x in endpoint["all_labels"]]
            positions = [self.positions[scope][str(row["id"])] for row in kept]
            clusters = [
                (
                    row.get("functional_cluster")
                    if name.startswith("en_hatecheck_eval/")
                    else digest(normalized(row["text"]).encode("utf8"))
                )
                for row in kept
            ]
            require(all(clusters), "Missing evaluation cluster.")
            require(
                all(
                    "text_cluster" not in row
                    or row["text_cluster"]
                    == digest(normalized(row["text"]).encode("utf8"))
                    for row in kept
                ),
                "Normalized-text cluster identity differs.",
            )
            self.views[scope, name] = dict(
                endpoint=endpoint,
                records=kept,
                truth=truth,
                positions=positions,
                label_indices=indices,
                clusters=clusters,
            )

    def aligned(self, values, scope, endpoint_name, *, ungated=False):
        require((scope, endpoint_name) in self.views, "Unknown evaluation endpoint.")
        view = self.views[scope, endpoint_name]
        head = view["endpoint"]["dimension"]
        validate_decisions(values, self.labels, len(self.inputs))
        result = values[head + ("_ungated" if ungated else "")][view["positions"]][
            :, view["label_indices"]
        ]
        require(
            result.shape == view["truth"].shape, "Aligned prediction dimensions differ."
        )
        return result

    def summary(self):
        return dict(
            input_rows=len(self.inputs),
            scope_counts={k: len(v) for k, v in self.scopes.items()},
            scope_overlap=len(
                set(x["key"] for x in self.scopes["core"])
                & set(x["key"] for x in self.scopes["broad"])
            ),
            endpoints={
                scope
                + "/"
                + name: dict(
                    rows=len(v["records"]),
                    labels=v["endpoint"]["all_labels"],
                    macro_labels=v["endpoint"]["macro_labels"],
                    gold_support=v["endpoint"]["gold_support"],
                    clusters=len(set(v["clusters"])),
                    positions_sha256=signature(v["positions"]),
                    label_indices=v["label_indices"],
                    primary=(scope == "broad" and name in PRIMARY),
                )
                for (scope, name), v in self.views.items()
            },
        )

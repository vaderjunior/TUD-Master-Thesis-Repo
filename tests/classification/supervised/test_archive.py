import copy
import io
import json
from pathlib import Path
import socket
import tempfile
from types import SimpleNamespace
import unittest
import warnings
import zipfile

import numpy as np
from analysis.classification.supervised import archive as r


def npy(value):
    b = io.BytesIO()
    np.save(b, value, allow_pickle=False)
    return b.getvalue()


def bundle(value):
    b = io.BytesIO()
    np.savez(b, **value)
    return b.getvalue()


class MemoryStore:
    def __init__(self):
        self.files = {}

    def put(self, name, value):
        self.files[name] = value if isinstance(value, bytes) else r.encoded(value)

    def expected(self, name):
        r.require(name in self.files, "Missing synthetic product.")
        return r.digest(self.files[name])

    def json(self, name):
        return r.parsed(self.files[name])

    def array(self, name):
        return np.load(io.BytesIO(self.files[name]), allow_pickle=False)


def final_fixture():
    job = dict(
        id="p/hate/cfg/s10",
        pool="p",
        head="hate",
        config="cfg",
        seed=10,
        fold=None,
        mode="final",
    )
    identity = dict(
        job=job,
        plan_sha256=r.PLAN_SHA,
        package_sha256=r.PACKAGE_SHA,
        gpu_profile="final_1x80",
    )
    inputs = [
        dict(key=str(i), text=str(i), exact_text_sha256=r.digest(str(i).encode()))
        for i in range(2)
    ]
    cfg = dict(id="cfg", epochs=1, learning_rate=2e-5, loss="unweighted")
    store = MemoryStore()
    attempt = "jobs/p/hate/cfg/s10/attempts/accepted"
    model = "models/p/hate/cfg/s10/accepted"
    for name in ("started.json", "training.json"):
        store.put(attempt + "/" + name, {})
    store.put(
        model + "/study_meta.json",
        dict(
            job=job,
            labels=["false", "true"],
            kind="binary",
            config=cfg,
            pilot_only=False,
        ),
    )
    store.put(model + "/model.safetensors", b"weights must never be read")
    store.put(model + "/config.json", {})
    rec = dict(
        status="complete",
        identity=identity,
        state="trained",
        attempt=attempt,
        products={
            n: store.expected(attempt + "/" + n)
            for n in ("started.json", "training.json")
        },
        model_directory="/results/sq2_full_20261029_v1/" + model,
        model_products={
            n: store.expected(model + "/" + n)
            for n in ("study_meta.json", "model.safetensors", "config.json")
        },
    )
    source = "jobs/" + job["id"] + "/complete.json"
    store.put(source, rec)
    root = "inference/" + job["id"]
    for name, value in [
        ("logits.npy", npy(np.array([[1, 0], [0, 1]], dtype=np.float32))),
        ("chunks/000000_000002.npy", b"bound existing chunk"),
        ("chunks/000000_000002.json", b"{}"),
    ]:
        store.put(root + "/" + name, value)
    inf = dict(
        binding=dict(
            identity=identity,
            training_receipt_sha256=store.expected(source),
            input_sha256=r.signature(inputs),
            labels=["false", "true"],
            rows=2,
        ),
        products={
            n: store.expected(root + "/" + n)
            for n in (
                "logits.npy",
                "chunks/000000_000002.npy",
                "chunks/000000_000002.json",
            )
        },
    )
    store.put(root + "/complete.json", inf)
    pool = r.GpuPool.__new__(r.GpuPool)
    pool.store = store
    pool.probability_cache = {}
    pool.job_records = {}
    pool.labels = {"hate": ["false", "true"]}
    pool.data = {"heads": {"hate": {"kind": "binary"}}}
    pool.study = SimpleNamespace(
        inputs=inputs, input_sha=r.signature(inputs), configs={"cfg": cfg}
    )
    return pool, job, source, root, rec, inf


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        parent = r.ROOT / ".test-output/sq2_reader_tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.parent = parent
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()
        if self.parent.exists() and not any(self.parent.iterdir()):
            self.parent.rmdir()

    def fixture(self, *, split=False, payload=None, actual=None, duplicate=False):
        payload = payload or {"jobs/p/item.json": b'{"value":1}'}
        actual = payload if actual is None else actual
        part = "records_000.zip" if split else "part_000.zip"
        path = self.root / part
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as z:
                for n, v in actual.items():
                    z.writestr(n, v)
                if duplicate:
                    n = next(iter(actual))
                    z.writestr(n, actual[n])
        files = {n: dict(bytes=len(v), sha256=r.digest(v)) for n, v in payload.items()}
        weights = (
            {"models/p/h/c/s/a/model.safetensors": dict(bytes=1, sha256=r.digest(b"x"))}
            if split
            else {}
        )
        m = dict(
            pool="p",
            plan_sha256=r.PLAN_SHA,
            status="ready_for_download" if split else "complete",
            parts={
                part: dict(bytes=path.stat().st_size, sha256=r.sha(path), files=files)
            },
            inventory_sha256=r.signature(
                dict(pool="p", plan_sha256=r.PLAN_SHA, files=files | weights)
            ),
        )
        if split:
            m.update(
                format="sq2-split-backup-v1",
                weights=weights,
                local_copy_count=1,
                server_originals_retained=True,
                server_deletion_authorized=False,
            )
        return m

    def store(self, m):
        raw = r.encoded(m)
        return r.ArchiveStore(self.root, raw, r.digest(raw), "p")

    def test_old_archive(self):
        with self.store(self.fixture()) as s:
            self.assertEqual(s.json("jobs/p/item.json"), {"value": 1})

    def test_split_archive_and_weight_boundary(self):
        with self.store(self.fixture(split=True)) as s:
            self.assertEqual(s.json("jobs/p/item.json"), {"value": 1})
            with self.assertRaises(r.IntegrityError):
                s.raw("models/p/h/c/s/a/model.safetensors")

    def test_wrong_manifest_hash(self):
        raw = r.encoded(self.fixture())
        with self.assertRaises(r.IntegrityError):
            r.ArchiveStore(self.root, raw, "0" * 64, "p")

    def test_wrong_plan(self):
        m = self.fixture()
        m["plan_sha256"] = "0" * 64
        with self.assertRaises(r.IntegrityError):
            self.store(m)

    def test_missing_inventory_entry(self):
        m = self.fixture()
        m["parts"]["part_000.zip"]["files"].clear()
        with self.assertRaises(r.IntegrityError):
            self.store(m)

    def test_path_traversal(self):
        m = self.fixture(payload={"jobs/p/../../other.json": b"{}"})
        with self.assertRaises(r.IntegrityError):
            self.store(m)

    def test_wrong_pool(self):
        m = self.fixture(payload={"jobs/other/item.json": b"{}"})
        with self.assertRaises(r.IntegrityError):
            self.store(m)

    def test_duplicate_inventory(self):
        m = self.fixture()
        m["parts"]["part_001.zip"] = copy.deepcopy(m["parts"]["part_000.zip"])
        (self.root / "part_001.zip").write_bytes(
            (self.root / "part_000.zip").read_bytes()
        )
        with self.assertRaises(r.IntegrityError):
            self.store(m)

    def test_truncated_zip(self):
        m = self.fixture()
        (self.root / "part_000.zip").write_bytes(b"broken")
        with self.assertRaises(r.IntegrityError):
            self.store(m)

    def test_changed_member_same_length(self):
        m = self.fixture(
            payload={"jobs/p/item.json": b'{"value":1}'},
            actual={"jobs/p/item.json": b'{"value":2}'},
        )
        with self.store(m) as s:
            with self.assertRaises(r.IntegrityError):
                s.raw("jobs/p/item.json")

    def test_duplicate_zip_member(self):
        with self.store(self.fixture(duplicate=True)) as s:
            with self.assertRaises(r.IntegrityError):
                s.raw("jobs/p/item.json")

    def test_extra_zip_member(self):
        m = self.fixture(
            actual={"jobs/p/item.json": b'{"value":1}', "jobs/p/extra.json": b"{}"}
        )
        with self.store(m) as s:
            with self.assertRaises(r.IntegrityError):
                s.raw("jobs/p/item.json")

    def test_changed_open_archive(self):
        m = self.fixture()
        with self.store(m) as s:
            (self.root / "part_000.zip").write_bytes(b"changed")
            with self.assertRaises(r.IntegrityError):
                s.raw("jobs/p/item.json")

    def test_unlisted_file(self):
        with self.store(self.fixture()) as s:
            with self.assertRaises(r.IntegrityError):
                s.raw("jobs/p/missing.json")

    def test_incomplete_manifest(self):
        m = self.fixture()
        m["status"] = "started"
        with self.assertRaises(r.IntegrityError):
            self.store(m)

    def test_duplicate_json_field(self):
        with self.assertRaises(r.IntegrityError):
            r.parsed(b'{"x":1,"x":2}')


class DecisionsTests(unittest.TestCase):
    def setUp(self):
        self.probs = {h: np.full((2, 2), 0.5) for h in r.HEADS}
        self.heads = {
            h: {"selected": {"threshold": None if h == "severity" else 0.5}}
            for h in r.HEADS
        }
        self.labels = {h: ["a", "b"] for h in r.HEADS}

    def test_exact_cutoff_ties(self):
        value = r.decode(self.probs, self.heads, "selected")
        self.assertTrue((value["hate"][:, 0] == 1).all())
        self.assertTrue((value["hate_type_ungated"] == 1).all())
        self.assertTrue((value["hate_type"] == 0).all())
        self.assertTrue((value["severity_ungated"] == [[1, 0], [1, 0]]).all())

    def test_matching_selected_gate(self):
        self.probs["hate"][:] = [0.4, 0.6]
        self.heads["hate"]["selected"]["threshold"] = 0.7
        self.assertTrue(
            (r.decode(self.probs, self.heads, "selected")["hate_type"] == 0).all()
        )
        self.assertTrue(
            (r.decode(self.probs, self.heads, "default")["hate_type"] == 1).all()
        )

    def test_wrong_gate_rejected(self):
        v = r.decode(self.probs, self.heads, "default")
        v["target_group"][0, 0] = 1
        with self.assertRaises(r.IntegrityError):
            r.validate_decisions(v, self.labels, 2)

    def test_wrong_width_rejected(self):
        v = r.decode(self.probs, self.heads, "default")
        v["hate_type"] = v["hate_type"][:, :1]
        with self.assertRaises(r.IntegrityError):
            r.validate_decisions(v, self.labels, 2)

    def test_wrong_array_dtype_rejected(self):
        v = r.decode(self.probs, self.heads, "default")
        v["hate"] = v["hate"].astype(np.float32)
        with self.assertRaises(r.IntegrityError):
            r.validate_decisions(v, self.labels, 2)

    def test_nonfinite_logits(self):
        with self.assertRaises(r.IntegrityError):
            r.probabilities("hate", np.array([[np.nan, 0]]))

    def test_extreme_logits_finite(self):
        for h in r.HEADS:
            self.assertTrue(
                np.isfinite(r.probabilities(h, np.array([[-1000, 1000]]))).all()
            )

    def test_unsupported_policy(self):
        with self.assertRaises(r.IntegrityError):
            r.decode(self.probs, self.heads, "best_test")

    def test_pickle_array_rejected(self):
        with self.assertRaises(r.IntegrityError):
            r.arrays_from_bytes(bundle({"a": np.array([{}], dtype=object)}))

    def test_nonfinite_bundle_rejected(self):
        with self.assertRaises(r.IntegrityError):
            r.arrays_from_bytes(bundle({"a": np.array([np.inf])}))


class IdentityTests(unittest.TestCase):
    def test_good_canonical_fit(self):
        p, j, *_ = final_fixture()
        x = p.final_probabilities(j)
        self.assertEqual(x.shape, (2, 2))
        self.assertFalse(x.flags.writeable)
        self.assertEqual(
            p.job_records[j["id"]]["attempt"].rsplit("/", 1)[-1], "accepted"
        )

    def bad(self, key, value):
        p, j, source, root, rec, inf = final_fixture()
        rec[key] = value
        p.store.put(source, rec)
        with self.assertRaises(r.IntegrityError):
            p.final_probabilities(j)

    def test_stopped_attempt_not_used(self):
        self.bad("status", "stopped")

    def test_search_attempt_not_used(self):
        self.bad("attempt", "jobs/p/hate/cfg/s0_f0/attempts/other")

    def test_wrong_model_attempt(self):
        self.bad(
            "model_directory",
            "/results/sq2_full_20261029_v1/models/p/hate/cfg/s10/latest",
        )

    def test_wrong_amendment(self):
        self.bad("verification_amendment_sha256", "unreviewed")

    def test_unbound_amp_state(self):
        self.bad("state", "amp_initialization_only")

    def test_wrong_training_binding(self):
        p, j, s, root, rec, inf = final_fixture()
        inf["binding"]["training_receipt_sha256"] = "0" * 64
        p.store.put(root + "/complete.json", inf)
        with self.assertRaises(r.IntegrityError):
            p.final_probabilities(j)

    def test_wrong_input_order(self):
        p, j, *_ = final_fixture()
        p.study.input_sha = r.signature(list(reversed(p.study.inputs)))
        with self.assertRaises(r.IntegrityError):
            p.final_probabilities(j)

    def test_wrong_label_order(self):
        p, j, *_ = final_fixture()
        p.labels["hate"] = ["true", "false"]
        with self.assertRaises(r.IntegrityError):
            p.final_probabilities(j)

    def test_wrong_seed(self):
        p, j, *_ = final_fixture()
        j = dict(j, seed=11)
        with self.assertRaises(r.IntegrityError):
            p.final_probabilities(j)

    def test_missing_chunk_binding(self):
        p, j, s, root, rec, inf = final_fixture()
        del inf["products"]["chunks/000000_000002.json"]
        p.store.put(root + "/complete.json", inf)
        with self.assertRaises(r.IntegrityError):
            p.final_probabilities(j)

    def test_duplicate_input(self):
        p, *_ = final_fixture()
        with self.assertRaises(r.IntegrityError):
            r.validate_inputs([p.study.inputs[0]] * 2, 2)

    def test_changed_input_text(self):
        p, *_ = final_fixture()
        p.study.inputs[0]["text"] = "changed"
        with self.assertRaises(r.IntegrityError):
            r.validate_inputs(p.study.inputs, 2)

    def test_bad_scope_position(self):
        p, *_ = final_fixture()
        with self.assertRaises(r.IntegrityError):
            r.scope_positions(
                p.study.inputs, [{"key": "0", "source_id": "0", "position": 1}]
            )

    def test_network_blocked(self):
        with r.offline():
            with self.assertRaises(r.IntegrityError):
                socket.create_connection(("example.com", 443))


class GoldTests(unittest.TestCase):
    def test_missing_and_empty_remain_different(self):
        self.assertIsNone(
            r.gold_labels({"gate": True, "hate_types": None}, "hate_type")
        )
        self.assertEqual(
            r.gold_labels({"gate": True, "hate_types": []}, "hate_type"), set()
        )
        self.assertIsNone(
            r.gold_labels({"gate": False, "hate_types": ["a"]}, "hate_type")
        )
        self.assertIsNone(
            r.gold_labels({"gate": None, "hate_types": ["a"]}, "hate_type")
        )

    def test_unknown_annotation_label(self):
        ep = dict(
            dimension="hate_type",
            all_labels=["a"],
            scorable_ids=["1"],
            gold_support={"a": 0},
        )
        with self.assertRaises(r.IntegrityError):
            r.endpoint_rows([{"id": "1", "gate": True, "hate_types": ["x"]}], ep)

    def test_eligibility_order(self):
        ep = dict(
            dimension="hate",
            all_labels=["not_hate", "hate"],
            scorable_ids=["1", "2"],
            gold_support={"not_hate": 1, "hate": 1},
        )
        rows = [{"id": "2", "gate": False}, {"id": "1", "gate": True}]
        with self.assertRaises(r.IntegrityError):
            r.endpoint_rows(rows, ep)

    def test_nonboolean_gate_rejected(self):
        with self.assertRaises(r.IntegrityError):
            r.gold_labels({"gate": "false"}, "hate")


if __name__ == "__main__":
    unittest.main(verbosity=2)

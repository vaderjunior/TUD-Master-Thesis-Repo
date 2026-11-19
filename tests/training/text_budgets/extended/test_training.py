from contextlib import contextmanager
import copy
import io
from pathlib import Path
import socket
from types import SimpleNamespace
import uuid
import numpy as np
import pytest
from training.text_budgets.extended import common as b
from training.text_budgets.extended import train as train
from training.text_budgets.extended import audit as audit
from training.text_budgets.extended import features as features
from training.text_budgets.extended import linear as linear

c = b.c


ARCHIVE_BASE = b.BASE


def archived_data(pool):
    path = ARCHIVE_BASE / "data/train" / (pool + ".json")
    if not path.is_file():
        pytest.skip("Original text-budget TRAIN membership is absent")
    return c.read(path)


@pytest.fixture
def isolated(monkeypatch):
    from datetime import datetime, timezone

    class FixtureDate(datetime):
        @classmethod
        def now(cls, tz=None):
            value = cls(2026, 10, 25, tzinfo=timezone.utc)
            return (
                value.astimezone(tz) if tz is not None else value.replace(tzinfo=None)
            )

    monkeypatch.setattr(b, "datetime", FixtureDate)
    path = b.BASE / "development/fixtures" / uuid.uuid4().hex[:10]
    path.mkdir(parents=True)
    monkeypatch.setattr(b.resource, "resource_guard", lambda *a, **k: {"fixture": True})
    monkeypatch.setattr(b, "ac_guard", lambda: None)
    with b.isolated(base=path):
        yield path


def fixture_data():
    ids = [f"r{i}" for i in range(20)]
    folds = [i % 5 for i in range(20)]
    heads = {}
    for head in c.HEADS:
        labs = (
            ["not_hate", "hate"]
            if head == "hate"
            else ["l0", "l1", "l2"] if head == "severity" else ["l0", "l1"]
        )
        ix = range(20) if head == "hate" else range(10, 20)
        rows = []
        for i in ix:
            target = (
                int(i >= 10)
                if head == "hate"
                else i % 3 if head == "severity" else [int(i % 2 == 0), int(i % 2 == 1)]
            )
            rows.append(dict(id=ids[i], feature_row=i, cluster=f"g{i}", target=target))
        heads[head] = dict(
            labels=labs,
            rows=rows,
            kind="multilabel" if head in ("hate_type", "target_group") else "single",
        )
    return dict(
        gate_ids=ids,
        texts=[f"text{i}" for i in range(20)],
        folds={"42": folds},
        heads=heads,
    )


@pytest.mark.parametrize("grid,count,init,empty,steps", [(b.GRID, 1080, 0, 12, 792720)])
def test_frozen_manifest_independent_inventory(grid, count, init, empty, steps):
    path = ARCHIVE_BASE / "data" / f"jobs_{grid}.json"
    if not path.is_file():
        pytest.skip("Original text-budget job manifest is absent")
    rows = c.read(path)
    assert len(rows) == len({r["id"] for r in rows}) == count
    assert sum(r["state"] == "initialization_only" for r in rows) == init
    assert sum(r["state"] == "empty_held_fold" for r in rows) == empty
    assert sum(r["steps"] for r in rows) == steps
    assert set(r["pool"] for r in rows) == set(b.NAMES)


@pytest.mark.parametrize("pool", b.NAMES)
def test_fixed_group_folds_cover_all_real_annotations(pool):
    data = archived_data(pool)
    for head in c.HEADS:
        held = []
        for fold in range(5):
            job = b.job(pool, head, "l1_e20_u", 0, fold)
            spec, fit, val = b.split(data, job)
            assert not {v["cluster"] for v in fit} & {v["cluster"] for v in val}
            held.extend(v["id"] for v in val)
        assert sorted(held) == sorted(r["id"] for r in data["heads"][head]["rows"])
        assert len(held) == len(set(held))


def test_missing_annotation_is_not_a_negative():
    data = archived_data("uniform_compact_s42")
    assert len(data["heads"]["hate_type"]["rows"]) == 5
    assert len(data["gate_ids"]) == 147
    assert len({r["feature_row"] for r in data["heads"]["hate_type"]["rows"]}) == 5


def test_group_leak_is_rejected():
    data = fixture_data()
    data["heads"]["hate"]["rows"][1]["cluster"] = "g0"
    with pytest.raises(Exception, match="leakage"):
        b.split(data, b.job(b.NAMES[0], "hate", "l1_e20_u", 0, 0))


def test_oof_seed_alignment_and_duplicate_rejection():
    data = fixture_data()
    j = b.job(b.NAMES[0], "hate", "l1_e20_u", 2, 0)
    matrix = np.full((3, 20, 2), np.nan)
    z = np.array([[0, 1], [0, 2], [0, 3], [0, 4]], dtype=np.float32)
    audit.place_oof(matrix, data, j, z)
    assert np.isnan(matrix[:2]).all()
    assert np.allclose(matrix[2, [0, 5, 10, 15], 1], 1 / (1 + np.exp(-np.arange(1, 5))))
    with pytest.raises(Exception, match="twice"):
        audit.place_oof(matrix, data, j, z)


@pytest.mark.parametrize("kind", ["shape", "nonfinite", "seed"])
def test_invalid_oof_is_rejected(kind):
    data = fixture_data()
    j = b.job(b.NAMES[0], "hate", "l1_e20_u", 0, 0)
    z = np.zeros((4, 2), dtype=np.float32)
    if kind == "shape":
        z = z[:3]
    elif kind == "nonfinite":
        z[0, 0] = np.nan
    else:
        j["seed"] = 10
    with pytest.raises(Exception):
        audit.place_oof(np.full((3, 20, 2), np.nan), data, j, z)


def test_twenty_step_sparse_fit_and_empty_fold_are_distinct():
    data = archived_data(b.NAMES[0])
    found = set()
    with b.isolated():
        for fold in range(5):
            state, steps = b.expected_state(
                data, b.job(b.NAMES[0], "hate_type", "l1_e20_u", 0, fold)
            )
            found.add(state)
            assert steps == (0 if state == "empty_held_fold" else 20)
    assert found == {"trained", "empty_held_fold"}


def test_empty_fold_never_calls_trainer(isolated, monkeypatch):
    data = archived_data(b.NAMES[0])
    j = next(
        b.job(b.NAMES[0], "hate_type", "l1_e20_u", 0, f)
        for f in range(5)
        if not b.split(data, b.job(b.NAMES[0], "hate_type", "l1_e20_u", 0, f))[2]
    )
    monkeypatch.setattr(
        train,
        "train_attempt",
        lambda *a, **k: pytest.fail("Empty held fold fitted a model"),
    )
    assert train.execute(j, data, {}, "plan") == "empty_held_fold"
    r, path = b.receipt(j, "plan")
    assert r["state"] == "empty_held_fold"
    assert train.execute(j, data, {}, "plan") == "verified"
    assert not (path / "held_logits.npy").exists()


def test_interrupted_fit_uses_new_attempt_then_completed_skip(isolated, monkeypatch):
    data = fixture_data()
    j = b.job(b.NAMES[0], "hate", "l1_e20_u", 0, 0)
    calls = []

    def fake(path, *args, **kwargs):
        calls.append(path)
        b.save(path / "partial.json", dict(call=len(calls)))
        if len(calls) == 1:
            raise KeyboardInterrupt("fixture interruption")
        return {"model_directory": None}

    monkeypatch.setattr(train, "train_attempt", fake)
    with pytest.raises(KeyboardInterrupt):
        train.execute(j, data, {"base_cache": {"snapshot": "fixture"}}, "plan")
    assert (calls[0] / "stopped.json").exists() and not (
        c.job_dir(j) / "complete.json"
    ).exists()
    train.execute(j, data, {"base_cache": {"snapshot": "fixture"}}, "plan")
    assert (
        len(calls) == 2
        and calls[0] != calls[1]
        and (calls[0] / "partial.json").exists()
    )
    assert train.execute(j, data, {}, "plan") == "verified" and len(calls) == 2
    with pytest.raises(Exception):
        b.receipt(j, "wrong_plan")
    (calls[1] / "partial.json").write_bytes(b"changed fixture")
    with pytest.raises(Exception):
        b.receipt(j, "plan")


def fake_inference(monkeypatch, root):
    data = fixture_data()
    j = b.job(b.NAMES[0], "hate", "l1_e20_u", 10, None)
    b.save(c.job_dir(j) / "complete.json", {"fixture": True})
    source = {"model_directory": str(root / "model")}
    calls = []
    monkeypatch.setattr(
        train,
        "setup",
        lambda: SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
    )
    monkeypatch.setattr(b, "receipt", lambda *a, **k: (source, root / "training"))
    monkeypatch.setattr(train, "load", lambda *a, **k: (None, None, {}))
    monkeypatch.setattr(train, "tokens", lambda tok, texts: [int(x) for x in texts])

    def infer(model, rows, **kwargs):
        calls.append(list(rows))
        return np.column_stack((rows, np.negative(rows))).astype(np.float32)

    monkeypatch.setattr(train, "infer", infer)
    inputs = [dict(key=f"k{i}", text=str(i)) for i in range(513)]
    return data, j, inputs, calls


def test_interrupted_prediction_chunk_preserved_and_resume_skips(isolated, monkeypatch):
    data, j, inputs, calls = fake_inference(monkeypatch, isolated)
    stem = c.BASE / "inference" / j["id"] / "chunks/000000_000512"
    old = np.column_stack((np.arange(512), -np.arange(512))).astype(np.float32)
    old[0, 0] = 1e-5
    b.array(stem.with_suffix(".npy"), old)
    digest = c.sha(stem.with_suffix(".npy"))
    train.inference_job(j, data, "plan", inputs)
    assert c.sha(stem.with_suffix(".npy")) == digest and len(calls) == 2
    got = b.short.compare_logits(
        c.BASE / "inference" / j["id"] / "complete.json",
        j,
        "plan",
        c.sha(c.job_dir(j) / "complete.json"),
        inputs,
        data["heads"]["hate"]["labels"],
    )
    assert np.array_equal(got[:512], old)
    assert (
        train.inference_job(j, data, "plan", inputs) == "verified" and len(calls) == 2
    )


def test_orphan_prediction_disagreement_stops(isolated, monkeypatch):
    data, j, inputs, calls = fake_inference(monkeypatch, isolated)
    stem = c.BASE / "inference" / j["id"] / "chunks/000000_000512"
    b.array(stem.with_suffix(".npy"), np.zeros((512, 2), dtype=np.float32))
    digest = c.sha(stem.with_suffix(".npy"))
    with pytest.raises(Exception, match="Interrupted chunk differs"):
        train.inference_job(j, data, "plan", inputs)
    assert (
        c.sha(stem.with_suffix(".npy")) == digest
        and not stem.with_suffix(".json").exists()
    )


def test_prediction_wrong_input_and_raw_corruption_rejected(isolated, monkeypatch):
    data, j, inputs, calls = fake_inference(monkeypatch, isolated)
    train.inference_job(j, data, "plan", inputs)
    path = c.BASE / "inference" / j["id"] / "complete.json"
    args = [
        path,
        j,
        "plan",
        c.sha(c.job_dir(j) / "complete.json"),
        inputs,
        data["heads"]["hate"]["labels"],
    ]
    changed = copy.deepcopy(inputs)
    changed[0]["text"] = "changed"
    args[4] = changed
    with pytest.raises(Exception, match="input binding"):
        b.short.compare_logits(*args)
    args[4] = inputs
    (path.parent / "chunks/000512_000513.npy").write_bytes(b"broken fixture")
    with pytest.raises(Exception):
        b.short.compare_logits(*args)


@pytest.mark.parametrize("variant", ["wrong_plan", "wrong_keys", "bad_norm", "nan"])
def test_feature_chunk_identity_and_values(isolated, variant):
    stem = c.BASE / "chunk"
    identity = dict(plan_sha256="plan", keys=["a", "b"])
    value = np.zeros((2, 768), dtype=np.float32)
    value[:, 0] = 1
    features.commit_chunk(stem, identity, value)
    if variant.startswith("wrong"):
        other = copy.deepcopy(identity)
        if variant == "wrong_plan":
            other["plan_sha256"] = "different"
        else:
            other["keys"] = ["b", "a"]
        with pytest.raises(Exception):
            features.commit_chunk(stem, other)
    else:
        if variant == "nan":
            value[0, 0] = np.nan
        else:
            value[0, 0] = 2
        with pytest.raises(Exception):
            features.validate_values(value, 2)


def test_feature_orphan_is_compared_without_replacing(isolated):
    stem = c.BASE / "chunk"
    identity = dict(plan_sha256="p", keys=["a"])
    value = np.zeros((1, 768), dtype=np.float32)
    value[0, 0] = 1
    b.array(stem.with_suffix(".npy"), value)
    old = c.sha(stem.with_suffix(".npy"))
    other = value.copy()
    other[0, 1] = 1e-5
    assert np.array_equal(features.commit_chunk(stem, identity, other), value)
    assert c.sha(stem.with_suffix(".npy")) == old


def test_semantic_npz_resume_and_refusal_to_replace(isolated):
    path = c.BASE / "values.npz"
    values = {"a": np.arange(10)}
    b.arrays(path, values)
    digest = c.sha(path)
    b.arrays(path, values)
    assert c.sha(path) == digest
    with pytest.raises(Exception):
        b.arrays(path, {"a": np.arange(11)})
    with pytest.raises(Exception):
        b.arrays(path, {"b": np.arange(10)})


def test_gate_threshold_and_argmax_ties_are_explicit():
    heads = {
        h: dict(selected=dict(threshold=0.5 if h != "severity" else None))
        for h in c.HEADS
    }
    probs = dict(
        hate=np.array([[0.5, 0.5], [0.49, 0.51]]),
        hate_type=np.array([[0.5, 0.5], [0.5, 0.49]]),
        target_group=np.array([[1.0, 1.0], [0.5, 0.2]]),
        severity=np.array([[0.5, 0.5, 0], [0.5, 0.5, 0]]),
    )
    result = audit.decoded(probs, heads, "selected")
    assert result["hate"].tolist() == [[1, 0], [0, 1]]
    assert result["hate_type"].tolist() == [[0, 0], [1, 0]]
    assert result["severity"].tolist() == [[0, 0, 0], [1, 0, 0]]
    assert result["hate_type_ungated"].tolist() == [[1, 1], [1, 0]]


def test_selection_uses_each_seeds_own_gate(monkeypatch):
    data = fixture_data()
    configs = [
        v for v in b.configs() if v["id"] in ("l1_e20_u", "l1_e20_b", "l1_e20_u")
    ]
    monkeypatch.setattr(c, "configs", lambda: configs)
    oof = {}
    for head, spec in data["heads"].items():
        truth = c.truth(head, spec["rows"], spec["labels"]).astype(float)
        good = truth * 0.98 + 0.01
        prob = np.stack([good.copy() for _ in c.CV_SEEDS])
        if head == "hate":
            prob[0] = 1 - good
        oof[head] = {cfg["id"]: prob.copy() for cfg in configs}
    selected = c.select(data, oof)
    audit.reference.check_selection(data, oof, selected)
    for family, heads in selected.items():
        assert np.allclose(heads["hate_type"]["selected"]["seed_scores"], [0, 1, 1])
        if family == "tuned":
            assert heads["hate"]["selected"]["config"] == "l1_e20_u"


def test_linear_candidates_and_saved_coefficient_predictions():
    data = fixture_data()
    x = np.random.default_rng(27).normal(size=(20, 6))
    cache, counts = linear.linear.make_oof(data, x, 42)
    heads = linear.linear.fit_recipe(data, x, 42, cache)
    assert linear.linear.check_selection(data, 42, heads, cache) == 112
    assert linear.check_heads(data, heads, 6) == 4
    probs = linear.checked_probabilities(heads, x)
    assert probs["hate"].shape == (20, 1)
    heads["hate"]["training_ids"].reverse()
    with pytest.raises(Exception, match="TRAIN support/order"):
        linear.check_heads(data, heads, 6)


def test_fold_class_weights_do_not_see_held_labels():
    data = fixture_data()
    j = b.job(b.NAMES[0], "hate", "l1_e20_b", 0, 0)
    spec, fit, held = b.split(data, j)
    weights = c.class_weights("hate", [r["target"] for r in fit], 2, True)
    for r in held:
        r["target"] = 1 - r["target"]
    spec, fit2, held2 = b.split(data, j)
    assert weights == c.class_weights("hate", [r["target"] for r in fit2], 2, True)
    assert weights["counts"] == [8, 8]


def test_signed_zero_fingerprint_is_exact_numeric_equality():

    class Parameter:
        def __init__(self, value):
            self.value = value

        def detach(self):
            return self

        def cpu(self):
            return self

        def contiguous(self):
            return self

        def numpy(self):
            return self.value

    def model(x):
        return SimpleNamespace(named_parameters=lambda: [("weights", Parameter(x))])

    a = np.array([0.0, -0.0, 1.0], dtype=np.float32)
    z = np.array([-0.0, 0.0, 1.0], dtype=np.float32)
    before = train.parameter_fingerprint(model(a))
    after = train.parameter_fingerprint(model(z))
    assert (
        before["raw_sha256"] != after["raw_sha256"]
        and before["zero_canonical_sha256"] == after["zero_canonical_sha256"]
    )
    z[2] = np.nextafter(np.float32(1), np.float32(2))
    assert (
        train.parameter_fingerprint(model(z))["zero_canonical_sha256"]
        != before["zero_canonical_sha256"]
    )
    assert np.signbit(a[1]) and not np.signbit(a[0])


def stores_fixture():
    return dict(
        pools={p: "D:/Thesis_Models/astra_sq2_stage_b20_20261021_v1" for p in b.NAMES},
        drive_reserve_GiB={"C:\\": 50, "D:\\": 16},
        initial_drive_reserve_GiB={"C:\\": 50, "D:\\": 16},
    )


@pytest.mark.parametrize("drive", ["C:\\", "D:\\"])
def test_storage_includes_both_drives_tokenizer_and_reserves(
    isolated, monkeypatch, drive
):
    stores = stores_fixture()
    b.validate_stores(stores)
    free = {"C:\\": 50 * 2**30, "D:\\": 360 * b.EXPORT_BYTES_MAX + 16 * 2**30}
    monkeypatch.setattr(
        b.shutil, "disk_usage", lambda drive: SimpleNamespace(free=free[drive])
    )
    b.store_guard(stores, initial=True)
    free[drive] -= 1
    with pytest.raises(Exception, match="storage reserve"):
        b.store_guard(stores, initial=True)


def test_remaining_final_exports_reduce_reservation(isolated, monkeypatch):
    stores = stores_fixture()
    j = b.job(b.NAMES[0], "hate", "l1_e20_b", 10, None)
    free = {"C:\\": 50 * 2**30, "D:\\": b.EXPORT_BYTES_MAX + 16 * 2**30}
    monkeypatch.setattr(
        b.shutil, "disk_usage", lambda drive: SimpleNamespace(free=free[drive])
    )
    b.store_guard(stores, remaining=[j])
    b.save(c.job_dir(j) / "complete.json", {"fixture": "verified elsewhere"})
    free["D:\\"] -= b.EXPORT_BYTES_MAX
    b.store_guard(stores, remaining=[j])


def test_export_path_cannot_escape_and_does_not_create_files(isolated):
    j = b.job(b.NAMES[0], "hate", "l1_e20_b", 10, None)
    stores = stores_fixture()
    p = b.model_path(j, "a", stores)
    assert p.is_relative_to(Path(stores["pools"][j["pool"]]).resolve())
    bad = dict(j, id="../../../../../other")
    with pytest.raises(Exception, match="escapes"):
        b.model_path(bad, "a", stores)


def test_local_lock_blocks_second_owner(isolated):
    path = isolated / "lock"
    with b.resource.file_lock(path):
        with pytest.raises(RuntimeError, match="already active"):
            with b.resource.file_lock(path):
                pass


def test_offline_context_blocks_connections(isolated):
    with pytest.raises(RuntimeError, match="offline"):
        socket.create_connection(("invalid.example", 1))
    sock = socket.socket()
    try:
        with pytest.raises(RuntimeError, match="offline"):
            sock.connect(("127.0.0.1", 1))
    finally:
        sock.close()


def test_immutable_publication_and_workspace_containment(isolated):
    p = isolated / "proof.json"
    b.save(p, {"x": 1})
    before = p.read_bytes()
    b.save(p, {"x": 1})
    with pytest.raises(Exception):
        b.save(p, {"x": 2})
    assert p.read_bytes() == before
    with pytest.raises(Exception):
        b.save(isolated.parent / "outside.json", {"x": 1})


def test_missing_plan_does_not_start_any_work(monkeypatch):
    from training.text_budgets.extended import run as runner

    monkeypatch.setattr(b, "BASE", b.BASE / "development/nonexistent_plan_fixture")
    monkeypatch.setattr(
        train, "execute", lambda *a, **k: pytest.fail("Unsealed runner started a fit")
    )
    with pytest.raises(Exception, match="not sealed"):
        runner.run()


@pytest.mark.parametrize("grid", ["full", "one_rate", "balanced_only"])
def test_unapproved_grids_cannot_execute(grid):
    with pytest.raises(Exception, match="adopted"):
        b.configs(grid)


def mixed_families():
    fixed = {
        name: {
            h: {
                "selected": {
                    "config": "l1_e20_" + loss,
                    "threshold": 0.5 if h != "severity" else None,
                }
            }
            for h in c.HEADS
        }
        for name, loss in [("fixed_unweighted", "u"), ("fixed_balanced", "b")]
    }
    fixed["tuned"] = {
        h: copy.deepcopy(fixed["fixed_balanced" if i % 2 else "fixed_unweighted"][h])
        for i, h in enumerate(c.HEADS)
    }
    return fixed


def test_mixed_loss_selection_reuses_exactly_360_final_weights():
    selected = mixed_families()
    jobs = [j for pool in b.NAMES for j in b.final_jobs(pool, selected)]
    assert len(jobs) == len({j["id"] for j in jobs}) == 360
    assert {j["config"] for j in jobs} == {"l1_e20_u", "l1_e20_b"}
    assert {j["seed"] for j in jobs} == set(range(10, 15))
    assert (
        selected["tuned"]["hate"]["selected"]["config"]
        != selected["tuned"]["hate_type"]["selected"]["config"]
    )


def test_all_final_steps_and_fold_schedules_match_metadata():
    import math

    with b.isolated():
        jobs = [j for p in b.NAMES for j in b.final_jobs(p, mixed_families())]
        assert (
            sum(b.expected_state(archived_data(j["pool"]), j)[1] for j in jobs)
            == 330600
        )
        assert all(
            b.expected_state(archived_data(j["pool"]), j)[0] == "trained" for j in jobs
        )
        assert all(
            x["epochs"] == 20 and x["learning_rate"] == 2e-5 for x in b.configs()
        )


def test_pilot_and_scientific_output_roots_are_separate(isolated):
    j = b.job(b.NAMES[0], "hate", "l1_e20_u", 10, None)
    assert b.model_path(j, "technical").is_relative_to(isolated)
    assert c.job_dir(j).is_relative_to(isolated)
    assert isolated.is_relative_to(b.BASE / "development")


@pytest.mark.parametrize("kind", ["missing_C_reserve", "external_drive", "wrong_name"])
def test_unapproved_storage_cannot_be_sealed(kind):
    stores = stores_fixture()
    if kind == "missing_C_reserve":
        del stores["drive_reserve_GiB"]["C:\\"]
    elif kind == "external_drive":
        stores["pools"][b.NAMES[0]] = "F:/Thesis_Models/astra_sq2_stage_b20_20261021_v1"
    else:
        stores["pools"][b.NAMES[0]] = "D:/Thesis_Models/existing_study"
    with pytest.raises(Exception):
        b.validate_stores(stores)


def test_four_day_budget_stops_and_preserves_original_clock(isolated, monkeypatch):
    from datetime import datetime, timezone, timedelta

    class BeforeDeadline:
        @staticmethod
        def now(tz):
            return datetime(2026, 10, 21, 12, tzinfo=timezone.utc)

        @staticmethod
        def fromisoformat(s):
            return datetime.fromisoformat(s)

        def __new__(cls, *args, **kwargs):
            return datetime(*args, **kwargs)

    monkeypatch.setattr(b, "datetime", BeforeDeadline)
    monkeypatch.setattr(
        c, "utc", lambda: datetime(2026, 10, 21, 12, tzinfo=timezone.utc).isoformat()
    )
    b.begin_execution("p")
    path = c.BASE / "execution_budget.json"
    digest = c.sha(path)
    b.begin_execution("p")
    assert c.sha(path) == digest
    with pytest.raises(Exception, match="another plan"):
        b.begin_execution("wrong")

    class AfterBudget(BeforeDeadline):
        @staticmethod
        def now(tz):
            return datetime(2026, 10, 25, 12, tzinfo=timezone.utc)

    monkeypatch.setattr(b, "datetime", AfterBudget)
    with pytest.raises(Exception, match="four-day"):
        b.deadline_guard()
    assert c.sha(path) == digest


def test_queue_search_finish_and_completed_restart(isolated, monkeypatch):
    from contextlib import nullcontext
    from training.text_budgets.extended import run as runner
    from training.text_budgets.extended import prepare as prep

    plan = dict(grid=b.GRID, stores={"fixture": True})
    calls = []
    selected = mixed_families()
    monkeypatch.setattr(b, "BASE", isolated)
    monkeypatch.setattr(b, "locks", lambda: nullcontext())
    monkeypatch.setattr(b, "source_guard", lambda: {})
    monkeypatch.setattr(b, "deadline_guard", lambda: None)
    monkeypatch.setattr(b, "validate_stores", lambda *args: None)
    monkeypatch.setattr(b, "store_guard", lambda *args, **kwargs: None)
    monkeypatch.setattr(b, "data", lambda *args: fixture_data())
    monkeypatch.setattr(prep, "validate", lambda: plan)

    def search(pool, *args):
        return [b.job(pool, "hate", "l1_e20_u", 0, fold) for fold in (0, 1)]

    monkeypatch.setattr(b, "jobs", search)

    def stub(path):
        b.save(path, dict(outputs={}))
        return dict(outputs={})

    def fit(job, *args):
        path = c.job_dir(job) / "complete.json"
        if path.exists():
            assert c.read(path)["job"] == job
            return
        calls.append(("fit", job["mode"], job["id"]))
        b.save(path, dict(job=job))

    def choose(pool, grid, digest):
        assert all((c.job_dir(j) / "complete.json").exists() for j in search(pool))
        b.save(
            c.BASE / "selections" / f"{pool}.json",
            dict(plan_sha256=digest, families=selected),
        )
        return selected

    def predict(job, *args):
        assert (c.BASE / "all_selections.json").exists()
        calls.append(("predict", job["id"]))
        stub(c.BASE / "inference" / job["id"] / "complete.json")

    monkeypatch.setattr(runner.train, "execute", fit)
    monkeypatch.setattr(runner.train, "inference_job", predict)
    monkeypatch.setattr(
        runner.features, "run", lambda *args: stub(c.BASE / "features/complete.json")
    )
    monkeypatch.setattr(
        runner.linear, "run", lambda *args: stub(c.BASE / "linear/complete.json")
    )
    monkeypatch.setattr(runner.audit, "select", choose)
    monkeypatch.setattr(
        runner.audit,
        "final_check",
        lambda pool, *args: stub(c.BASE / "audits" / f"{pool}_final.json"),
    )
    monkeypatch.setattr(
        runner.audit,
        "assemble",
        lambda pool, *args: stub(c.BASE / "audits" / f"{pool}_assembly.json"),
    )
    b.save(c.BASE / "plan.json", plan)
    b.save(c.BASE / "data/inference.json", [dict(key="fixture", text="fixture")])
    runner.run("search")
    assert sum(x[0] == "fit" for x in calls) == 18 and not (c.BASE / "refits").exists()
    runner.run("finish")
    assert sum(x[0] == "fit" and x[1] == "final" for x in calls) == 360
    assert sum(x[0] == "predict" for x in calls) == 360
    assert c.read(c.BASE / "complete.json")["scientific_review_pending"]
    before = list(calls)
    runner.run()
    assert calls == before

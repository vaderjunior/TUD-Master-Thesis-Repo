import copy
import math
from pathlib import Path
import socket
from types import SimpleNamespace
import numpy as np
import pytest
from training.roberta.short import common as s
from training.roberta.short import audit as audit
from training.roberta.short import prepare as preparation
from training.roberta.short import run as run
from training.roberta.short import train as train

c = s.c
SOURCE_ROOT = c.ROOT


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    monkeypatch.setattr(s, "ROOT", tmp_path)
    monkeypatch.setattr(s, "OLD", tmp_path / "old")
    monkeypatch.setattr(s, "BASE", tmp_path / "new")
    for module in [audit, preparation, run]:
        monkeypatch.setattr(module, "BASE", s.BASE)
    with s.isolated():
        yield s.BASE


def toy():
    n = 20
    rows = [
        dict(
            id=str(i),
            feature_row=i,
            cluster=str(i),
            target=i % 2,
            original_partition="train",
        )
        for i in range(n)
    ]
    data = dict(
        folds={"42": [i % 5 for i in range(n)]},
        texts=[str(i) for i in range(n)],
        heads={},
    )
    for head in c.HEADS:
        rr = copy.deepcopy(rows)
        if head in ("hate_type", "target_group"):
            for row in rr:
                row["target"] = [1 - row["target"], row["target"]]
        data["heads"][head] = dict(
            labels=["a", "b"],
            rows=rr,
            kind="multilabel" if head in ("hate_type", "target_group") else "binary",
        )
    oof = {
        head: {
            cfg: np.tile(
                np.asarray(
                    [[0.9, 0.1] if i % 2 == 0 else [0.1, 0.9] for i in range(n)]
                ),
                (3, 1, 1),
            )
            for cfg in c.config_map()
        }
        for head in c.HEADS
    }
    return data, oof


def test_grid_is_full_cross_product_and_context_restores():
    original = c.configs
    with s.isolated():
        new = s.search_jobs()
        old = s.original_search_jobs()
        assert len(new) == len({j["id"] for j in new}) == 1800
        assert len(old) == len({j["id"] for j in old}) == 720
        assert not {j["id"] for j in new} & {j["id"] for j in old}
        assert len(c.configs()) == 42
        assert {x["epochs"] for x in c.configs()} == {1, 2, 5, 10, 15, 20, 60}
        for head in c.HEADS:
            for seed in c.CV_SEEDS:
                for fold in range(5):
                    assert (
                        len(
                            [
                                j
                                for j in new
                                if (j["head"], j["seed"], j["fold"])
                                == (head, seed, fold)
                            ]
                        )
                        == 30
                    )
        assert len(c.configs()) == 42
    assert c.configs is original and len(c.configs()) == 12


def test_all_1232_candidate_scores_fixed_controls_and_ties(isolated):
    data, oof = toy()
    selected = c.select(data, oof)
    assert audit.old.check_selection(data, oof, selected) == 1232
    with s.isolated(s.OLD, expanded=False):
        prior = c.select(
            data, {h: {cfg: oof[h][cfg] for cfg in c.config_map()} for h in c.HEADS}
        )
    for family in ["fixed_unweighted", "fixed_balanced"]:
        assert selected[family] == prior[family]
    assert all(selected["tuned"][h]["selected"]["config"] == "l0_e1_u" for h in c.HEADS)
    assert selected["tuned"]["hate"]["selected"]["threshold"] == 0.5
    assert len(s.all_jobs(selected)) == 60


def test_gate_seed_mapping_and_unsupported_labels(isolated):
    from sklearn.metrics import f1_score

    data, oof = toy()
    ix = [11, 3, 17, 5, 1, 9]
    for cfg in c.config_map():
        oof["hate"][cfg][1] = 1 - oof["hate"][cfg][1]
        oof["hate"][cfg][2, :, 1] = 0.6
        oof["hate"][cfg][2, :, 0] = 0.4
    for head in c.HEADS[1:]:
        data["heads"][head]["rows"] = [data["heads"][head]["rows"][i] for i in ix]

        if head == "severity":
            for i, row in enumerate(data["heads"][head]["rows"]):
                row["target"] = i % 2
        else:
            data["heads"][head]["labels"].append("unsupported")
            for i, row in enumerate(data["heads"][head]["rows"]):
                row["target"] = [i % 2, 1 - i % 2, 0]
        for cfg in c.config_map():
            p = oof[head][cfg][:, ix]
            if head != "severity":
                p = np.concatenate([p, np.ones((3, len(ix), 1)) * 0.99], axis=2)
            oof[head][cfg] = p
    chosen = c.select(data, oof)
    assert audit.old.check_selection(data, oof, chosen) == 1232
    item = chosen["tuned"]["hate_type"]["selected"]
    gate = chosen["tuned"]["hate"]["selected"]
    truth = np.asarray([r["target"] for r in data["heads"]["hate_type"]["rows"]])
    probs = oof["hate_type"][item["config"]]
    gates = oof["hate"][gate["config"]][:, :, 1] > gate["threshold"]
    expected = []
    for seed in range(3):
        pred = probs[seed] >= item["threshold"]
        pred[~gates[seed, ix]] = False
        expected.append(
            np.mean(
                [f1_score(truth[:, j], pred[:, j], zero_division=0) for j in [0, 1]]
            )
        )
    assert item["seed_scores"] == expected and item["mean_score"] == np.mean(expected)
    assert chosen["tuned"]["hate_type"]["selection_indices"] == [0, 1]
    assert len(data["heads"]["hate_type"]["labels"]) == 3
    oof["hate"]["l0_e1_u"][0, 0] = np.nan
    with pytest.raises(RuntimeError, match="OOF shape"):
        c.select(data, oof)


def test_no_network_and_immutable_arrays(isolated):
    with pytest.raises(RuntimeError, match="network disabled"):
        socket.create_connection(("example.invalid", 443))
    path = isolated / "pred.npz"
    s.arrays(path, {"x": np.arange(10)})
    before = path.read_bytes()
    s.arrays(path, {"x": np.arange(10)})
    assert path.read_bytes() == before
    with pytest.raises(RuntimeError, match="Existing arrays differ"):
        s.arrays(path, {"x": np.arange(11)})
    with pytest.raises(RuntimeError, match="outside isolated"):
        c.save(isolated.parent / "escape.json", {})


def simple_fit(attempt, job, *args, **kwargs):
    c.save(attempt / "training.json", dict(job=job, stub=True))


def test_fitting_interrupt_resume_and_corruption(isolated, monkeypatch):
    monkeypatch.setattr(
        train.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 2**30)
    )

    def stop(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(train, "train_attempt", stop)
    job = s.search_jobs()[0]
    plan = {"base_cache": {"snapshot": "mock"}}
    with pytest.raises(KeyboardInterrupt):
        train.execute(job, {}, plan, "p")
    stopped = list(c.job_dir(job).glob("attempts/*/stopped.json"))
    assert len(stopped) == 1
    old = stopped[0].read_bytes()
    calls = []

    def fit(*args, **kwargs):
        calls.append(1)
        simple_fit(*args, **kwargs)

    monkeypatch.setattr(train, "train_attempt", fit)
    assert train.execute(job, {}, plan, "p") == "new"
    assert train.execute(job, {}, plan, "p") == "verified" and calls == [1]
    assert (
        stopped[0].read_bytes() == old
        and len(list(c.job_dir(job).glob("attempts/*"))) == 2
    )
    with pytest.raises(RuntimeError, match="job/plan"):
        train.execute(job, {}, plan, "changed")
    receipt = c.read(c.job_dir(job) / "complete.json")
    path = c.BASE / receipt["attempt"] / "training.json"
    path.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="artifact differs"):
        train.execute(job, {}, plan, "p")


def test_archived_weight_exception_is_narrow_and_paths_bound(isolated):
    job = dict(
        id="hate/l1_e20_u/s10",
        mode="final",
        head="hate",
        config="l1_e20_u",
        seed=10,
        fold=None,
    )
    with s.isolated(s.OLD, expanded=False):
        attempt = c.job_dir(job) / "attempts/a"
        c.save(attempt / "training.json", {"fixture": True})
        rp = c.job_dir(job) / "complete.json"
        weight = attempt / "model/model.safetensors"
        val = dict(
            job=job,
            status="complete",
            plan_sha256=s.OLD_PLAN_SHA,
            attempt=attempt.relative_to(s.OLD).as_posix(),
            products={
                "training.json": c.sha(attempt / "training.json"),
                "model/model.safetensors": "weightdigest",
            },
        )
        c.save(rp, val)
    archive = {
        c.rel(weight): dict(
            sha256="weightdigest", receipt=c.rel(rp), receipt_sha256=c.sha(rp)
        )
    }
    _, _, missing = s.verify_old_training(job, archive)
    assert missing == [c.rel(weight)]
    with pytest.raises(RuntimeError, match="Unexpected missing"):
        s.verify_old_training(job)
    wrong = copy.deepcopy(archive)
    wrong[c.rel(weight)]["receipt_sha256"] = "bad"
    with pytest.raises(RuntimeError, match="recorded archive"):
        s.verify_old_training(job, wrong)
    changed = copy.deepcopy(val)
    changed["products"] = {
        "training.json": val["products"]["training.json"],
        "held_logits.npy": "bad",
    }
    rp.write_bytes(c.encoded(changed))
    with pytest.raises(RuntimeError, match="Unexpected missing"):
        s.verify_old_training(job, archive)
    changed["products"] = {"../escape.txt": "bad"}
    rp.write_bytes(c.encoded(changed))
    with pytest.raises(RuntimeError, match="escaped attempt"):
        s.verify_old_training(job, archive)
    changed["attempt"] = "../elsewhere"
    rp.write_bytes(c.encoded(changed))
    with pytest.raises(RuntimeError, match="escaped job"):
        s.verify_old_training(job, archive)


def inference_fixture(job, inputs, plan_sha, training_sha, full):
    folder = c.BASE / "inference" / job["id"]
    artifacts = {}
    for start in range(0, len(inputs), 512):
        stop = min(start + 512, len(inputs))
        stem = folder / "chunks" / f"{start:06d}_{stop:06d}"
        c.array(stem.with_suffix(".npy"), full[start:stop])
        c.save(
            stem.with_suffix(".json"),
            dict(
                identity=dict(
                    job=job,
                    plan_sha256=plan_sha,
                    training_receipt_sha256=training_sha,
                    start=start,
                    stop=stop,
                    input_sha256=c.signature(inputs[start:stop]),
                ),
                sha256=c.sha(stem.with_suffix(".npy")),
            ),
        )
        for suffix in [".npy", ".json"]:
            artifacts[c.rel(stem.with_suffix(suffix))] = c.sha(stem.with_suffix(suffix))
    c.array(folder / "logits.npy", full)
    artifacts[c.rel(folder / "logits.npy")] = c.sha(folder / "logits.npy")
    c.save(
        folder / "complete.json",
        dict(
            job=job,
            plan_sha256=plan_sha,
            training_receipt_sha256=training_sha,
            rows=len(inputs),
            artifacts=artifacts,
        ),
    )
    return folder / "complete.json"


def test_inference_input_roster_and_merged_data_are_verified(isolated):
    inputs = [dict(key=f"en/{i}", text=f"text{i}") for i in range(1025)]
    job = {"id": "h/c/s10"}
    full = np.arange(2050, dtype=np.float32).reshape(-1, 2)
    rp = inference_fixture(job, inputs, "p", "t", full)
    assert np.array_equal(s.compare_logits(rp, job, "p", "t", inputs, ["a", "b"]), full)
    altered = copy.deepcopy(inputs)
    altered[513]["text"] = "other text"
    with pytest.raises(RuntimeError, match="input binding"):
        s.compare_logits(rp, job, "p", "t", altered, ["a", "b"])
    rec = c.read(rp)
    saved = rp.read_bytes()
    rec["artifacts"].pop(c.rel(rp.parent / "logits.npy"))
    rp.write_bytes(c.encoded(rec))
    with pytest.raises(RuntimeError, match="roster incomplete"):
        s.compare_logits(rp, job, "p", "t", inputs, ["a", "b"])
    rp.write_bytes(saved)
    (rp.parent / "logits.npy").write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="Changed frozen input"):
        s.compare_logits(rp, job, "p", "t", inputs, ["a", "b"])


def test_real_inference_path_resumes_chunks_and_preserves_orphans(
    isolated, monkeypatch
):
    monkeypatch.setattr(
        train.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 2**30)
    )
    monkeypatch.setattr(train, "train_attempt", simple_fit)
    monkeypatch.setattr(
        train,
        "setup",
        lambda: SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None)),
    )
    monkeypatch.setattr(train, "load", lambda *a, **kw: (None, None, None))
    monkeypatch.setattr(train, "tokens", lambda _, texts: texts)
    calls = []
    interrupt = [True]

    def infer(_, texts):
        calls.append(len(texts))
        if len(calls) == 2 and interrupt[0]:
            raise KeyboardInterrupt()
        return np.asarray([[float(v), -float(v)] for v in texts], dtype=np.float32)

    monkeypatch.setattr(train, "infer", infer)
    job = dict(
        id="hate/l0_e1_u/s10",
        mode="final",
        head="hate",
        config="l0_e1_u",
        seed=10,
        fold=None,
    )
    data = {"heads": {"hate": {"labels": ["a", "b"]}}}
    inputs = [{"text": str(i), "key": str(i)} for i in range(1025)]
    train.execute(job, data, {"base_cache": {"snapshot": "unused"}}, "p")
    with pytest.raises(KeyboardInterrupt):
        train.inference_job(job, data, "p", inputs)
    folder = c.BASE / "inference" / job["id"]
    first = (folder / "chunks/000000_000512.npy").read_bytes()
    orphan = folder / "chunks/000512_001024.npy"
    c.array(orphan, np.asarray([[i, -i] for i in range(512, 1024)], dtype=np.float32))
    orphan_bytes = orphan.read_bytes()
    interrupt[0] = False
    assert train.inference_job(job, data, "p", inputs) == "new"
    assert calls == [512, 512, 512, 1] and orphan.read_bytes() == orphan_bytes
    assert (folder / "chunks/000000_000512.npy").read_bytes() == first
    assert train.inference_job(job, data, "p", inputs) == "verified" and len(calls) == 4
    assert s.compare_logits(
        folder / "complete.json",
        job,
        "p",
        c.sha(c.job_dir(job) / "complete.json"),
        inputs,
        ["a", "b"],
    ).shape == (1025, 2)


def test_trainer_derivative_has_only_documented_changes():
    import ast
    import copy

    def functions(relative):
        source = (SOURCE_ROOT / relative).read_text(encoding="utf8")
        return {
            n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
        }

    original = functions("training/roberta/train.py")
    short = functions("training/roberta/short/train.py")
    assert original.keys() == short.keys()

    class ShortSchedule(ast.NodeTransformer):
        changes = 0

        def visit_Assign(self, node):
            self.generic_visit(node)
            if any(
                isinstance(t, ast.Name) and t.id == "max_steps" for t in node.targets
            ):
                assert ast.dump(node.value) == ast.dump(
                    ast.parse("8 if pilot else total", mode="eval").body
                )
                node.value = ast.parse(
                    "min(8, total) if pilot else total", mode="eval"
                ).body
                self.changes += 1
            if any(isinstance(t, ast.Name) and t.id == "unit" for t in node.targets):
                self.changes += 1
                capture = ast.parse(
                    "c.array(attempt / 'epoch_logits' / f'{epoch:03d}.npy', z)"
                ).body[0]
                return [capture, node]
            return node

    transform = ShortSchedule()
    for name, node in original.items():
        expected = transform.visit(copy.deepcopy(node))
        assert ast.dump(expected, include_attributes=False) == ast.dump(
            short[name], include_attributes=False
        )
    assert transform.changes == 2


def fake_scientific_fit(attempt, job, data, cache, **kwargs):
    spec, fit, held = train.split(data, job)
    cfg = c.config_map()[job["config"]]
    total = math.ceil(len(fit) / 16) * cfg["epochs"]
    warm = math.ceil(0.1 * total)
    trace = [
        dict(
            lr=cfg["learning_rate"]
            * (i / warm if i < warm else (total - i) / max(1, total - warm)),
            skipped=False,
            scale_before=8,
            scale_after=8,
        )
        for i in range(total)
    ]
    curve = [dict(epoch=e) for e in range(1, cfg["epochs"] + 1)]
    if held:
        z = np.asarray(
            [[2.0, -2.0] if int(r["id"]) % 2 == 0 else [-2.0, 2.0] for r in held],
            dtype=np.float32,
        )
        y = np.asarray([r["target"] for r in held])
        loss = (
            np.mean(np.logaddexp(0, z) - y * z)
            if job["head"] in ("hate_type", "target_group")
            else np.mean(np.log(np.exp(z).sum(1)) - z[np.arange(len(y)), y])
        )
        for entry in curve:
            entry["held_unweighted_loss"] = float(loss)
            if c.BASE == s.BASE:
                c.array(attempt / "epoch_logits" / f"{entry['epoch']:03d}.npy", z)
        c.array(attempt / "held_logits.npy", z)
    if job["mode"] == "final":
        c.save(
            attempt / "model/study_meta.json",
            dict(
                job=job,
                labels=spec["labels"],
                kind=spec["kind"],
                config=cfg,
                pilot_only=False,
            ),
        )
        c.publish_bytes(attempt / "model/model.safetensors", b"fake-model-never-loaded")
    meta = dict(
        job=job,
        configuration=cfg,
        pilot_only=False,
        fit_ids=[r["id"] for r in fit],
        held_ids=[r["id"] for r in held],
        planned_steps=total,
        attempted_steps=total,
        successful_steps=total,
        skipped_steps=0,
        steps=trace,
        epochs=curve,
        changes={"encoder": 0.1, "classifier": 0.1},
        class_weights=c.class_weights(
            job["head"],
            [r["target"] for r in fit],
            len(spec["labels"]),
            cfg["loss"] == "balanced",
        ),
        export_check={"clean_reload": True, "max_abs_logit_difference": 0},
    )
    c.save(attempt / "training.json", meta)


def test_epoch_capture_is_independently_checked(isolated):
    data, _ = toy()
    job = s.search_jobs()[0]
    attempt = isolated / "epoch-fixture"
    fake_scientific_fit(attempt, job, data, "unused")
    audit.metadata(job, data, attempt, True)
    saved = c.read(attempt / "training.json")
    saved["epochs"][0]["held_unweighted_loss"] += 0.1
    (attempt / "training.json").write_bytes(c.encoded(saved))
    with pytest.raises(RuntimeError, match="loss/logits disagree"):
        audit.metadata(job, data, attempt, True)


def test_all_known_gpu_locks_and_resource_accounting(isolated):
    competing = [
        s.OLD / "launcher.lock",
        s.ROOT / "experiments/astra/matched_type_pools_20261011_v1/launcher.lock",
        s.ROOT / "experiments/astra/compact_roberta_probe_20261008_v1/inference.lock",
        s.ROOT
        / "experiments/astra/later_evaluation_20261008_v1/encoders/inference.lock",
    ]
    for path in competing:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"0")
    with s.locks():
        with pytest.raises(RuntimeError, match="already active"):
            with s.locks():
                pass
        for path in competing:
            with pytest.raises(RuntimeError, match="already active"):
                with s.resource.file_lock(path, existing=True):
                    pass
    assert all(p.read_bytes() == b"0" for p in competing)
    jobs = [{"id": str(i)} for i in range(20)]
    assert s.reserve_for_jobs(jobs) == 23.0
    for job in jobs:
        c.save(isolated / "refits" / job["id"] / "complete.json", {})
    assert s.reserve_for_jobs(jobs) == 15.0


def test_entire_union_search_reuse_routing_and_resume(isolated, monkeypatch):
    "Exercise the real 1800+720 selection, 40 reused+20 new finals and audits."
    original_sha, original_read = c.sha, c.read
    sha_cache, read_cache = {}, {}

    def version(path):
        path = Path(path)
        info = path.stat()
        return path, info.st_mtime_ns, info.st_size

    def cached_sha(path):
        key = version(path)
        if key not in sha_cache:
            sha_cache[key] = original_sha(path)
        return sha_cache[key]

    def cached_read(path):
        key = version(path)
        if key not in read_cache:
            read_cache[key] = original_read(path)
        return copy.deepcopy(read_cache[key])

    monkeypatch.setattr(c, "sha", cached_sha)
    monkeypatch.setattr(c, "read", cached_read)
    monkeypatch.setattr(c.os, "fsync", lambda _descriptor: None)
    data, _ = toy()
    inputs = [{"key": f"en/{i}", "text": str(i)} for i in range(8)]
    monkeypatch.setattr(
        train.shutil, "disk_usage", lambda _: SimpleNamespace(free=100 * 2**30)
    )
    monkeypatch.setattr(train, "setup", lambda: None)
    monkeypatch.setattr(s.resource, "resource_guard", lambda *a: {})
    monkeypatch.setattr(s.resource, "resources", lambda: {"mock": True})
    monkeypatch.setattr(s, "archive_map", lambda: {})
    fitted = []
    inferred = []

    def fit(*args, **kwargs):
        fitted.append((str(c.BASE), args[1]["id"]))
        return fake_scientific_fit(*args, **kwargs)

    monkeypatch.setattr(train, "train_attempt", fit)

    def infer(job, data, sha, inputs):
        inferred.append((str(c.BASE), job["id"]))
        raw = np.asarray(
            [[2.0, -2.0] if i % 2 == 0 else [-2.0, 2.0] for i in range(len(inputs))],
            dtype=np.float32,
        )
        inference_fixture(
            job, inputs, sha, c.sha(c.job_dir(job) / "complete.json"), raw
        )
        return "mock"

    monkeypatch.setattr(train, "inference_job", infer)
    plan = dict(base_cache={"snapshot": "unused"}, inputs={}, prepared={})
    with s.isolated(s.OLD, expanded=False):
        for job in s.original_search_jobs():
            train.execute(job, data, plan, s.OLD_PLAN_SHA)
        oof, _, _ = audit.old.load_oof(data, s.OLD_PLAN_SHA)
        prior = c.select(data, oof)
        c.save(s.OLD / "selection.json", {"families": prior})
        for job in c.final_jobs(prior):
            train.execute(job, data, plan, s.OLD_PLAN_SHA)
            infer(job, data, s.OLD_PLAN_SHA, inputs)
        audit.old.assemble(prior, data, s.OLD_PLAN_SHA, inputs)
    c.save(isolated / "plan.json", plan)
    c.save(isolated / "data/train.json", data)
    c.save(isolated / "data/inference.json", inputs)
    c.save(isolated / "data/scopes.json", {})
    monkeypatch.setattr(preparation, "validate", lambda: plan)
    run.run("search")
    assert not (isolated / "complete.json").exists()
    assert len([x for x in fitted if x[0] == str(s.BASE)]) == 1800
    selected = c.read(isolated / "selection.json")["families"]
    routes = audit.routes(selected)
    assert sum(v["origin"] == "new_fit" for v in routes) == 20
    assert sum(v["origin"] == "historical_saved_predictions" for v in routes) == 40
    assert {v["job"]["seed"] for v in routes} == set(c.FINAL_SEEDS)
    run.run("finish")
    before = (len(fitted), len(inferred))
    run.run()
    assert before == (len(fitted), len(inferred))
    result = c.read(isolated / "complete.json")
    assert result["new_final_models"] == 20 and result["reused_final_models"] == 40
    assert (
        result["new_search_fits"] == 1800
        and result["reused_search_fits"] == 720
        and result["complete_outputs"] == 240
    )
    assert len([x for x in fitted if x[0] == str(s.BASE)]) == 1820
    assert c.read(isolated / "prediction_audit.json")[
        "fixed_historical_predictions_reproduce"
    ]
    chunk = next((isolated / "inference").glob("*/*/*/chunks/*.npy"))
    chunk.write_bytes(b"broken")
    with pytest.raises(RuntimeError, match="Changed frozen input"):
        run.run()

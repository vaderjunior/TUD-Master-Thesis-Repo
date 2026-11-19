import copy
import math
import numpy as np
import pytest
from training.roberta import common as c
from training.roberta import audit as audit
from training.roberta import train as train


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "ROOT", tmp_path)
    monkeypatch.setattr(c, "BASE", tmp_path / "study")
    return c.BASE


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
    for h in c.HEADS:
        rr = copy.deepcopy(rows)
        if h in ("hate_type", "target_group"):
            for r in rr:
                r["target"] = [1 - r["target"], r["target"]]
        data["heads"][h] = dict(
            labels=["a", "b"],
            rows=rr,
            kind="multilabel" if h in ("hate_type", "target_group") else "binary",
        )
    oof = {
        h: {
            cfg: np.tile(
                np.asarray(
                    [[0.9, 0.1] if i % 2 == 0 else [0.1, 0.9] for i in range(n)]
                ),
                (3, 1, 1),
            )
            for cfg in c.config_map()
        }
        for h in c.HEADS
    }
    return data, oof


def test_schedule_includes_original_and_balanced_and_all_seeds():
    jobs = c.search_jobs()
    assert len(jobs) == len({j["id"] for j in jobs}) == 720
    assert c.config_map()["l1_e20_u"]["learning_rate"] == 2e-5
    assert c.config_map()["l1_e20_b"]["epochs"] == 20
    assert set(j["seed"] for j in jobs) == {0, 1, 2}
    assert set(c.FINAL_SEEDS).isdisjoint(c.CV_SEEDS)


def test_training_folds_exclude_same_groups_and_return_only_annotations():
    data, _ = toy()
    data["heads"]["hate_type"]["rows"] = data["heads"]["hate_type"]["rows"][:7]
    job = dict(head="hate_type", fold=0, mode="search")
    _, fit, held = train.split(data, job)
    assert len(fit) == 5 and len(held) == 2
    assert not {r["id"] for r in fit} & {r["id"] for r in held}


def test_group_leakage_fails_closed():
    data, _ = toy()
    data["heads"]["hate"]["rows"][1]["cluster"] = "0"
    with pytest.raises(RuntimeError, match="Held group"):
        train.split(data, dict(head="hate", fold=0, mode="search"))


def test_final_refit_uses_every_permitted_row():
    data, _ = toy()
    _, fit, held = train.split(data, dict(head="hate_type", fold=None, mode="final"))
    assert len(fit) == 20 and held == []


def test_threshold_edges_and_severity_ties():
    assert c.decisions("hate", np.array([[0.5, 0.5]]), 0.5).tolist() == [[1, 0]]
    assert c.decisions("hate_type", np.array([[0.5, 0.5]]), 0.5).tolist() == [[1, 1]]
    assert c.decisions("severity", np.array([[0.5, 0.5]]), None).tolist() == [[1, 0]]


def test_gate_clears_all_fine_outputs():
    p = np.array([[0.99, 0.98], [0.9, 0.1]])
    for h in ("hate_type", "target_group", "severity"):
        z = c.decisions(h, p, 0.5, np.array([False, True]))
        assert z[0].sum() == 0 and z[1].sum() > 0


def test_extreme_logits_are_finite_probabilities():
    for h in c.HEADS:
        p = c.probabilities(h, np.array([[1000.0, -1000.0], [-1000.0, 1000.0]]))
        assert np.isfinite(p).all() and np.allclose(p, [[1, 0], [0, 1]])


def test_nonfinite_logits_rejected():
    with pytest.raises(RuntimeError, match="Invalid logits"):
        c.probabilities("hate", [[float("nan"), 0]])


def test_balancing_uses_only_given_training_labels():
    w = c.class_weights("hate", [0, 0, 0, 1], 2, True)
    assert np.allclose(w["weights"], [2 / 3, 2])
    w2 = c.class_weights("hate_type", [[1, 0], [0, 0], [0, 0], [0, 0]], 2, True)
    assert np.allclose(w2["weights"], [[2 / 3, 2], [1, 1]])


def test_absent_classes_have_no_infinite_weight():
    for h, y, k in [
        ("hate", [0, 0], 2),
        ("severity", [0, 0], 3),
        ("hate_type", [[0, 1], [0, 1]], 2),
    ]:
        w = c.class_weights(h, y, k, True)
        assert np.isfinite(w["weights"]).all()


@pytest.mark.parametrize(
    "head,y,logits",
    [
        ("hate", [0, 0, 1], [[2.0, 0], [1, 0], [0, 2]]),
        ("severity", [0, 1, 2], [[1.0, 0, 0], [0, 2, 0], [0, 0, 1]]),
        ("hate_type", [[1, 0], [0, 1], [0, 0]], [[1.0, -1], [-2, 2], [0.2, -0.1]]),
    ],
)
@pytest.mark.parametrize("balanced", [False, True])
def test_losses_match_independent_numpy(head, y, logits, balanced):
    torch = pytest.importorskip(
        "torch", reason="Torch is needed for independent tensor loss checks"
    )

    z = np.asarray(logits, dtype=float)
    target = np.asarray(y)
    weights = c.class_weights(head, target, z.shape[1], balanced)
    got = float(train.loss_value(torch.tensor(z), torch.tensor(target), head, weights))
    w = np.asarray(weights["weights"])
    if head == "hate_type":
        terms = np.logaddexp(0, z) - target * z
        expected = np.mean(terms * np.where(target, w[:, 1], w[:, 0]))
    else:
        logsum = np.log(np.exp(z).sum(1))
        expected = np.mean((logsum - z[np.arange(len(y)), target]) * w[target])
    assert abs(got - expected) < 1e-6


def test_all_candidates_and_selection_independently_reproduce():
    data, oof = toy()
    s = c.select(data, oof)
    assert audit.check_selection(data, oof, s) == 392
    assert len(s) == 3 and set(s) == set(c.FAMILIES)


def test_selection_does_not_ensemble_search_seeds():
    data, oof = toy()
    for h in c.HEADS:
        for p in oof[h].values():
            p[1:] = 1 - p[1:]
    s = c.select(data, oof)
    for family in c.FAMILIES:
        cand = s[family]["hate"]["selected"]
        assert cand["mean_score"] == np.mean(cand["seed_scores"])
    assert audit.check_selection(data, oof, s) == 392


def test_gated_head_selection_uses_matching_rows():
    data, oof = toy()
    keep = [i for i in range(20) if i % 3 == 0]
    for h in c.HEADS[1:]:
        data["heads"][h]["rows"] = [data["heads"][h]["rows"][i] for i in keep]
        for cfg in c.config_map():
            oof[h][cfg] = oof[h][cfg][:, keep]
    s = c.select(data, oof)
    assert audit.check_selection(data, oof, s) == 392


def test_selection_rejects_missing_cv_prediction():
    data, oof = toy()
    oof["hate"]["l0_e20_u"][0, 0] = np.nan
    with pytest.raises(RuntimeError, match="OOF shape"):
        c.select(data, oof)


def test_tie_rule_prefers_less_training_then_lower_lr():
    items = [
        dict(config="l2_e60_b", threshold=0.5, mean_score=0.6),
        dict(config="l1_e20_u", threshold=0.4, mean_score=0.6),
        dict(config="l0_e20_u", threshold=0.5, mean_score=0.6),
    ]
    assert min(items, key=c.candidate_key)["config"] == "l0_e20_u"


def test_final_job_dedup_keeps_all_five_initializations():
    data, oof = toy()
    s = c.select(data, oof)
    jobs = c.final_jobs(s)
    assert len(jobs) == len({j["id"] for j in jobs}) and 40 <= len(jobs) <= 60
    assert set(j["seed"] for j in jobs) == set(c.FINAL_SEEDS)


def test_learning_rate_audit_handles_skipped_step():
    meta = dict(
        planned_steps=10,
        configuration=dict(learning_rate=0.01),
        successful_steps=2,
        steps=[
            dict(lr=0, skipped=False, scale_before=8, scale_after=8),
            dict(lr=0.01, skipped=True, scale_before=8, scale_after=4),
            dict(lr=0.01, skipped=False, scale_before=4, scale_after=4),
        ],
    )
    audit.require_learning_rate_trace(meta)
    meta["steps"][2]["lr"] = 0.009
    with pytest.raises(RuntimeError, match="schedule"):
        audit.require_learning_rate_trace(meta)


def test_immutable_publication_and_owner_guard(isolated):
    path = isolated / "value.json"
    c.save(path, {"x": 1})
    c.save(path, {"x": 1})
    with pytest.raises(RuntimeError, match="replace"):
        c.save(path, {"x": 2})
    with pytest.raises(RuntimeError, match="outside"):
        c.save(isolated.parent / "bad.json", {})


def fake_train(attempt, job, *args, **kwargs):
    c.save(attempt / "training.json", dict(job=job, stub=True))


def test_completed_job_resumes_without_refitting_and_detects_corruption(
    isolated, monkeypatch
):
    calls = []

    def work(*args, **kwargs):
        calls.append(1)
        fake_train(*args, **kwargs)

    monkeypatch.setattr(train, "train_attempt", work)
    monkeypatch.setattr(
        train.shutil, "disk_usage", lambda p: type("D", (), {"free": 100 * 2**30})()
    )
    job = c.search_jobs()[0]
    plan = dict(base_cache=dict(snapshot="unused"))
    assert train.execute(job, {}, plan, "p") == "new"
    assert train.execute(job, {}, plan, "p") == "verified" and len(calls) == 1
    receipt = c.read(c.job_dir(job) / "complete.json")
    p = c.BASE / receipt["attempt"] / "training.json"
    p.write_text("{}")
    with pytest.raises(RuntimeError, match="artifact differs"):
        train.execute(job, {}, plan, "p")
    assert len(calls) == 1


def test_interrupt_preserves_attempt_and_restarts_only_that_job(isolated, monkeypatch):
    monkeypatch.setattr(
        train.shutil, "disk_usage", lambda p: type("D", (), {"free": 100 * 2**30})()
    )

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(train, "train_attempt", interrupt)
    job = c.search_jobs()[0]
    plan = dict(base_cache=dict(snapshot="unused"))
    with pytest.raises(KeyboardInterrupt):
        train.execute(job, {}, plan, "p")
    failed = list((c.job_dir(job) / "attempts").glob("*/stopped.json"))
    assert len(failed) == 1 and not (c.job_dir(job) / "complete.json").exists()
    old = failed[0].read_bytes()
    monkeypatch.setattr(train, "train_attempt", fake_train)
    assert train.execute(job, {}, plan, "p") == "new"
    assert (
        failed[0].read_bytes() == old
        and len(list((c.job_dir(job) / "attempts").iterdir())) == 2
    )


def test_changed_plan_cannot_reuse_completed_job(isolated, monkeypatch):
    monkeypatch.setattr(train, "train_attempt", fake_train)
    monkeypatch.setattr(
        train.shutil, "disk_usage", lambda p: type("D", (), {"free": 100 * 2**30})()
    )
    job = c.search_jobs()[0]
    plan = dict(base_cache=dict(snapshot="unused"))
    train.execute(job, {}, plan, "first")
    with pytest.raises(RuntimeError, match="job/plan"):
        train.execute(job, {}, plan, "different")


def test_second_launcher_is_rejected(isolated):
    with c.launcher_lock():
        with pytest.raises(RuntimeError, match="Another"):
            with c.launcher_lock():
                pass


def test_entire_mock_sequence_restart_and_corrupted_inference(isolated, monkeypatch):
    "Exercise 720 tiny fake fits through the real selectors, audits and decoder."
    from training.roberta import run as run

    data, _ = toy()
    c.BASE.mkdir(parents=True)
    c.save(c.BASE / "plan.json", {"mock": True})
    plan_sha = c.sha(c.BASE / "plan.json")
    c.save(c.BASE / "data/train.json", data)
    inputs = [dict(key=f"en/{i}", text=f"text{i}") for i in range(1025)]
    c.save(c.BASE / "data/inference.json", inputs)
    plan = dict(jobs=c.search_jobs(), base_cache=dict(snapshot="mock"))
    monkeypatch.setattr(run.preparation, "validate", lambda: plan)
    monkeypatch.setattr(train, "setup", lambda: None)
    monkeypatch.setattr(
        run.shutil, "disk_usage", lambda p: type("D", (), {"free": 100 * 2**30})()
    )
    fitted = []
    inferred = []

    def fake_fit(attempt, job, data, cache, **kwargs):
        fitted.append(job["id"])
        spec, fit, held = train.split(data, job)
        cfg = c.config_map()[job["config"]]
        n = math.ceil(len(fit) / 16) * cfg["epochs"]
        warm = math.ceil(0.1 * n)
        trace = [
            dict(
                lr=cfg["learning_rate"]
                * (i / warm if i < warm else (n - i) / (n - warm)),
                skipped=False,
                scale_before=8,
                scale_after=8,
            )
            for i in range(n)
        ]
        info = dict(
            job=job,
            pilot_only=False,
            configuration=cfg,
            planned_steps=n,
            attempted_steps=n,
            successful_steps=n,
            skipped_steps=0,
            fit_ids=[r["id"] for r in fit],
            held_ids=[r["id"] for r in held],
            changes={"encoder": 0.1, "classifier": 0.1},
            steps=trace,
            class_weights=c.class_weights(
                job["head"],
                [r["target"] for r in fit],
                len(spec["labels"]),
                cfg["loss"] == "balanced",
            ),
            export_check={"clean_reload": True, "max_abs_logit_difference": 1e-6},
        )
        c.save(attempt / "training.json", info)
        if job["mode"] == "final":
            c.save(attempt / "model/config.json", {"mock": True})
            c.publish_bytes(
                attempt / "model/model.safetensors", b"mock-weights-never-loaded"
            )
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
        if held:
            z = np.array(
                [[2.0, -2.0] if int(r["id"]) % 2 == 0 else [-2.0, 2.0] for r in held],
                dtype=np.float32,
            )
            c.array(attempt / "held_logits.npy", z)

    def fake_infer(job, data, plan_sha, inputs):
        inferred.append(job["id"])
        directory = c.BASE / "inference" / job["id"]
        source = c.sha(c.job_dir(job) / "complete.json")
        artifacts = {}
        full = np.asarray(
            [[2.0, -2.0] if i % 2 == 0 else [-2.0, 2.0] for i in range(len(inputs))],
            dtype=np.float32,
        )
        for start in range(0, len(inputs), 512):
            stop = min(start + 512, len(inputs))
            chunk = directory / "chunks" / f"{start:06d}_{stop:06d}"
            c.array(chunk.with_suffix(".npy"), full[start:stop])
            c.save(
                chunk.with_suffix(".json"),
                dict(
                    identity=dict(
                        job=job,
                        plan_sha256=plan_sha,
                        training_receipt_sha256=source,
                        start=start,
                        stop=stop,
                        input_sha256=c.signature(inputs[start:stop]),
                    ),
                    sha256=c.sha(chunk.with_suffix(".npy")),
                ),
            )
            for suffix in (".json", ".npy"):
                p = chunk.with_suffix(suffix)
                artifacts[c.rel(p)] = c.sha(p)
        c.array(directory / "logits.npy", full)
        artifacts[c.rel(directory / "logits.npy")] = c.sha(directory / "logits.npy")
        c.save(
            directory / "complete.json",
            dict(
                job=job,
                plan_sha256=plan_sha,
                training_receipt_sha256=source,
                rows=len(inputs),
                artifacts=artifacts,
            ),
        )
        return "mock"

    monkeypatch.setattr(train, "train_attempt", fake_fit)
    monkeypatch.setattr(train, "inference_job", fake_infer)
    run.run()
    result = c.read(c.BASE / "complete.json")
    assert (
        result["search_fits"] == 720
        and result["final_fits"] == 60
        and len(fitted) == 780
    )
    assert len(inferred) == 60 and result["complete_outputs"] == 6 * 5 * 1025
    before = (len(fitted), len(inferred))
    run.run()
    assert before == (len(fitted), len(inferred))
    chunk = next((c.BASE / "inference").glob("*/*/*/chunks/*.npy"))
    chunk.write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="Changed frozen input"):
        run.run()

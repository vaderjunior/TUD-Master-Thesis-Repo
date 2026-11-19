import contextlib
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch
import uuid

import numpy as np
from training.matched_annotations import run as m

c = m.c


def fixture_vectors(texts):
    out = np.zeros((len(texts), 768), dtype=np.float32)
    for i, text in enumerate(texts):
        out[i, int(c.signature(text)[:8], 16) % 768] = 1
    return out


def feature_resume_check():
    base = m.BASE / "development" / ("feature_resume_" + uuid.uuid4().hex[:12])
    study, cache = base / "study", base / "cache"
    cache.mkdir(parents=True)
    rows = [
        dict(key=f"train/synthetic-{i}", text=f"Synthetic feature {i}")
        for i in range(132)
    ]
    (cache / "feature_inputs.json").write_bytes(c.encoded(rows[:2]))
    np.save(
        cache / "features_mean.npy",
        fixture_vectors([r["text"] for r in rows[:2]]),
        allow_pickle=False,
    )
    state = dict(calls=0, fail_infer=True, fail_receipt=False)
    real_save = m.save

    def infer(tok, model, texts, device):
        state["calls"] += 1
        if state["fail_infer"] and state["calls"] == 5:
            raise RuntimeError("synthetic feature interruption")
        return dict(mean=fixture_vectors(texts))

    def save(path, value):
        if state["fail_receipt"] and path == study / "features/chunks/000064.json":
            state["fail_receipt"] = False
            raise RuntimeError("synthetic interruption after array")
        real_save(path, value)

    fake_torch = SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda: None))

    def expect_failure(message):
        try:
            m.make_features({"synthetic": True})
        except RuntimeError as error:
            c.require(message in str(error), f"Unexpected synthetic failure: {error}")
        else:
            raise AssertionError("Synthetic failure did not occur: " + message)

    with (
        patch.object(m, "BASE", study),
        patch.object(m.feature, "BASE", cache),
        m.isolated(),
        patch.object(m.trainer, "setup", return_value=fake_torch),
        patch.object(m.feature, "load", return_value=(object(), object(), {})),
        patch.object(m.feature, "infer", side_effect=infer),
        patch.object(m, "save", side_effect=save),
        patch.dict(sys.modules, {"torch": fake_torch}),
        contextlib.redirect_stdout(io.StringIO()),
    ):
        m.save(study / "plan.json", {"synthetic": True})
        m.save(study / "data/feature_inputs.json", rows)
        expect_failure("synthetic feature interruption")
        first = study / "features/chunks/000000.npy"
        first_hash = c.sha(first)
        c.require(
            (first.with_suffix(".json")).exists(), "Completed feature chunk was lost"
        )
        state.update(fail_infer=False, fail_receipt=True)
        expect_failure("synthetic interruption after array")
        orphan = study / "features/chunks/000064.npy"
        c.require(
            orphan.exists() and not orphan.with_suffix(".json").exists(),
            "Orphan fixture missing",
        )
        m.make_features({"synthetic": True})
        c.require(c.sha(first) == first_hash, "Resumed first chunk changed")
        done = c.read(study / "features/complete.json")
        c.require(
            done["reused_old_rows"] == 2 and done["new_rows"] == 130,
            "Feature reuse count differs",
        )
        np.testing.assert_array_equal(
            np.load(study / "features/mean.npy"),
            fixture_vectors([r["text"] for r in rows]),
        )
        calls = state["calls"]
        m.make_features({"synthetic": True})
        c.require(calls == state["calls"], "Completed feature cache was recomputed")
        original = orphan.read_bytes()
        orphan.write_bytes(b"intentional synthetic corruption")
        expect_failure("Changed frozen input")
        orphan.write_bytes(original)

        second = base / "wrong_text_study"
        with patch.object(m, "BASE", second), m.isolated():
            m.save(second / "plan.json", {"synthetic": True})
            m.save(
                second / "data/feature_inputs.json", [dict(rows[0], text="Wrong text")]
            )
            expect_failure("Cached TRAIN feature text differs")
    return dict(
        status="pass",
        synthetic_only=True,
        cases=[
            "interruption preserves and reuses completed chunks",
            "orphan array is checked on resume",
            "completed output requires no new inference",
            "corruption rejected",
            "cached text mismatch rejected",
        ],
        fixture_root=c.rel(base),
        old_rows_reused=2,
        new_rows=130,
        chunks=3,
        CUDA=False,
        Qwen_calls=0,
    )


def main():
    c.require(
        not (m.BASE / "handoff_readiness.json").exists(),
        "Preserve previous handoff check",
    )
    source_hashes = m.hashes(m.source_files())
    extra = feature_resume_check()
    plan = m.validate()
    m.verify_draft_sources()
    pilot = c.read(m.BASE / "development/pilot_v1/complete.json")
    c.verify(pilot["outputs"])
    tests = c.read(m.test_receipt_path())
    replay = c.read(m.BASE / "development/bge_reference_replay.json")
    c.require(
        tests["passed"] and tests["tests"] == 17, "Expected sealed tests not passed"
    )
    c.require(
        replay["reference_reconstructions"] == 10 and replay["status"] == "pass",
        "BGE replay incomplete",
    )
    c.require(
        source_hashes == m.hashes(m.source_files()),
        "Sealed source changed during review",
    )
    for pool in m.POOL_NAMES:
        m.audit_data(m.data(pool))
    scientific = [
        "execution_started.json",
        "search",
        "refits",
        "features",
        "predictions",
        "complete.json",
    ]
    c.require(
        all(not (m.BASE / path).exists() for path in scientific),
        "Scientific run already started; handoff status must change",
    )
    free = shutil.disk_usage(m.ROOT).free
    c.require(free >= 85 * 2**30, "Less than launcher initial disk requirement")
    receipt = dict(
        status="ready_for_user_long_run",
        utc=c.utc(),
        plan_sha256=c.sha(m.BASE / "plan.json"),
        source_bindings_verified=len(plan["inputs"]),
        prepared_bindings_verified=len(plan["prepared"]),
        pretrained_cache_bindings_verified=len(plan["base_cache"]["files"]),
        tests=17,
        supplementary_feature_checks=extra,
        pilot_outputs_verified=len(pilot["outputs"]),
        bge_references_reproduced=10,
        bge_candidates_checked=sum(r["candidate_checks"] for r in replay["pools"]),
        free_GiB=free / 2**30,
        first_run_requires_GiB=85,
        max_checkpoint_estimate_GiB=70,
        scientific_fits_started=False,
        Qwen_calls=0,
        CUDA_this_check=False,
        checks_source=dict(path=c.rel(Path(__file__)), sha256=c.sha(Path(__file__))),
    )
    m.save(m.BASE / "handoff_readiness.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()

import json
import types
import uuid
from pathlib import Path
from unittest.mock import patch
import numpy as np
from training.roberta import common as c
from training.roberta import prepare as p
from training.roberta import train as t


def mock_case(base):
    inputs = [dict(key=f"en/{i}", text=str(i)) for i in range(1025)]
    job = dict(
        id="hate/l1_e20_u/s10",
        mode="final",
        head="hate",
        config="l1_e20_u",
        seed=10,
        fold=None,
    )
    data = dict(heads=dict(hate=dict(labels=["false", "true"], kind="binary")))
    attempt = c.job_dir(job) / "attempts/mock"
    c.save(attempt / "marker.json", {"mock_only": True})
    c.save(
        c.job_dir(job) / "complete.json",
        dict(
            status="complete",
            job=job,
            plan_sha256="mock",
            attempt=attempt.relative_to(base).as_posix(),
            products={"marker.json": c.sha(attempt / "marker.json")},
        ),
    )
    state = dict(loads=0, infers=0, interrupt=True)

    def load(*args, **kwargs):
        state["loads"] += 1
        return None, object(), {}

    def values(encoded):
        return np.asarray(
            [[int(s) % 3, -(int(s) % 3)] for s in encoded], dtype=np.float32
        )

    def infer(model, encoded, *args, **kwargs):
        state["infers"] += 1
        if state["interrupt"] and state["infers"] == 2:
            raise KeyboardInterrupt()
        return values(encoded)

    dummy = types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None))
    with patch.object(t, "setup", return_value=dummy), patch.object(
        t, "load", side_effect=load
    ), patch.object(t, "tokens", side_effect=lambda tok, text: text), patch.object(
        t, "infer", side_effect=infer
    ):
        try:
            t.inference_job(job, data, "mock", inputs)
        except KeyboardInterrupt:
            pass
        else:
            raise RuntimeError("Injected interrupt was not propagated")
        d = base / "inference" / job["id"]
        first = d / "chunks/000000_000512.npy"
        c.require(
            first.exists() and not (d / "complete.json").exists(),
            "Interrupted publication state differs",
        )
        first_sha = c.sha(first)
        state["interrupt"] = False
        c.require(
            t.inference_job(job, data, "mock", inputs) == "new", "Resume did not finish"
        )
        c.require(
            state["infers"] == 4 and c.sha(first) == first_sha,
            "Resume repeated or changed the verified first chunk",
        )
        counts = dict(state)
        c.require(
            t.inference_job(job, data, "mock", inputs) == "verified"
            and state == counts,
            "Completed inference was rerun",
        )

        first.write_bytes(b"deliberately-corrupted-mock-array")
        try:
            t.inference_job(job, data, "mock", inputs)
        except RuntimeError as exc:
            c.require(
                "Changed frozen input" in str(exc), "Unexpected corruption response"
            )
        else:
            raise RuntimeError("Corrupted chunk accepted")
    return dict(
        interruption_preserved_first_chunk=True,
        resume_inferred_only_remaining_chunks=True,
        completed_inference_reused=True,
        corrupt_completed_chunk_rejected=True,
    )


def orphan_case(base, *, agrees):
    inputs = [dict(key="en/0", text="0"), dict(key="en/1", text="1")]
    job = dict(
        id="hate/l1_e20_u/s10",
        mode="final",
        head="hate",
        config="l1_e20_u",
        seed=10,
        fold=None,
    )
    data = dict(heads=dict(hate=dict(labels=["false", "true"], kind="binary")))
    attempt = c.job_dir(job) / "attempts/mock"
    c.save(attempt / "marker.json", {"mock_only": True})
    c.save(
        c.job_dir(job) / "complete.json",
        dict(
            status="complete",
            job=job,
            plan_sha256="mock",
            attempt=attempt.relative_to(base).as_posix(),
            products={"marker.json": c.sha(attempt / "marker.json")},
        ),
    )
    original = np.array([[0, 0], [1, -1]], dtype=np.float32)
    d = base / "inference" / job["id"]
    path = d / "chunks/000000_000002.npy"
    c.array(path, original)
    before = c.sha(path)
    replacement = original.copy() if agrees else original + 1
    dummy = types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None))
    with patch.object(t, "setup", return_value=dummy), patch.object(
        t, "load", return_value=(None, object(), {})
    ), patch.object(t, "tokens", side_effect=lambda tok, text: text), patch.object(
        t, "infer", return_value=replacement
    ):
        if agrees:
            t.inference_job(job, data, "mock", inputs)
            c.require(
                (d / "complete.json").exists(), "Matching orphan did not complete"
            )
        else:
            try:
                t.inference_job(job, data, "mock", inputs)
            except RuntimeError as exc:
                c.require(
                    "Interrupted chunk differs" in str(exc),
                    "Wrong orphan mismatch response",
                )
            else:
                raise RuntimeError("Different orphan accepted")
            c.require(
                not path.with_suffix(".json").exists(), "Different orphan committed"
            )
    c.require(c.sha(path) == before, "Orphan was overwritten")
    return dict(agrees=agrees, passed=True, preserved_array=True)


def main():
    plan = p.validate()
    base = c.BASE
    out = base / "development/chunk_checks" / uuid.uuid4().hex[:12]
    with patch.object(c, "BASE", out / "interruption"):
        interruption = mock_case(c.BASE)
    orphan = []
    for match in (True, False):
        with patch.object(
            c, "BASE", out / ("matching_orphan" if match else "different_orphan")
        ):
            orphan.append(orphan_case(c.BASE, agrees=match))
    c.require(
        not (base / "search").exists()
        and not (base / "refits").exists()
        and not (base / "complete.json").exists(),
        "Scientific execution already exists; this handoff check expects preparation only",
    )
    check = dict(
        status="ready_for_user_run",
        plan_sha256=c.sha(base / "plan.json"),
        tests=c.read(base / "development/tests.json")["passed"],
        offline_fixture_count=29,
        pilot_summary=c.read(base / "development/pilot_complete.json")["summary"],
        pilot_independent=c.read(base / "development/pilot_independent_check.json"),
        inference_interruption=interruption,
        inference_orphans=orphan,
        scientific_search_jobs_started=0,
        final_models_fitted=0,
        Qwen_calls=0,
        source_sha256=c.sha(Path(__file__)),
        source=c.rel(Path(__file__)),
        artifact_hashes={
            c.rel(base / n): c.sha(base / n)
            for n in (
                "plan.json",
                "protocol.json",
                "development/tests.json",
                "development/pilot_complete.json",
                "development/pilot_independent_check.json",
            )
        },
    )
    c.save(base / "handoff_readiness.json", check)
    print(
        json.dumps(
            {
                k: check[k]
                for k in (
                    "status",
                    "offline_fixture_count",
                    "scientific_search_jobs_started",
                    "inference_interruption",
                    "inference_orphans",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

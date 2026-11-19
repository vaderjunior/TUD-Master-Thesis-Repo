from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import uuid

import numpy as np
import pytest

from analysis.feedback.rounds import resume as m

ORIGINAL_DIAGNOSIS = m.BASE / "diagnosis.json"


@pytest.fixture(autouse=True)
def offline():
    with m.previous.previous.inspection.offline(), m.original.threadpool_limits(
        limits=1
    ):
        yield


def assert_match(first, second):
    for channel in ("dense", "bm25", "rrf"):
        assert [r["id"] for r in first[channel]] == [r["id"] for r in second[channel]]
        assert all(
            abs(x["score"] - y["score"]) < 1e-10
            for x, y in zip(first[channel], second[channel])
        )
    assert first["selected_id"] == second["selected_id"]


def real_failure():
    if not ORIGINAL_DIAGNOSIS.is_file():
        pytest.skip("Original stopped-ranking diagnosis and vector inputs are absent")
    diag = m.c.read(ORIGINAL_DIAGNOSIS)
    root = m.c.ITERATIVE / "phases/r3_evaluation"
    science = m.c.read(root / "scientific_plan_v1.json")
    row = next(
        r for r in m.c.rows(root / "roster.jsonl") if r["key"] == diag["failed_key"]
    )
    indices = {
        r["id"]: i for i, r in enumerate(m.c.rows(m.c.CANDIDATES / "roster.jsonl"))
    }
    records = m.c.read(m.c.ITERATIVE / "gate_only_records.json")
    ids = science["pools"][diag["failed_condition"]]
    sv = np.load(m.c.CANDIDATES / "query_vectors.npy", mmap_mode="r")
    ev = np.load(m.original.SCOPE / "vectors.npy", mmap_mode="r")
    return (
        row,
        [records[i] for i in ids],
        sv[[indices[i] for i in ids]],
        ev[row["scope_node_index"]],
    )


def test_real_failure_is_reproduced_and_checker_alone_fixes_it():
    row, records, vectors, query = real_failure()
    primary = m.original.FeedbackBucket(records, vectors)
    old = m.original.LocalBucket(records, vectors)
    new = m.VerificationBucket(records, vectors)
    winner, trace = primary.retrieve(row["text"], query)
    old_winner, _, old_trace = old.retrieve(row["text"], query)
    new_winner, _, new_trace = new.retrieve(row["text"], query)
    assert old_winner == new_winner == winner
    assert old_trace != trace and new_trace == trace
    assert new.idf == primary.bm25.idf and new.idf != old.idf
    assert new.retrieve.__func__ is m.original.LocalBucket.retrieve


@pytest.mark.parametrize("n", [1, 2, 10, 20, 30, 40])
@pytest.mark.parametrize(
    "query_text", ["absent", "common common repeated", "w1 common"]
)
def test_pool_sizes_repeated_tokens_no_overlap_and_negative_idf(n, query_text):
    rng = np.random.default_rng(25 + n)
    records = [
        dict(
            id=f"fixture-{i:03}", text=f"common common w{i} repeated " + ("extra " * i)
        )
        for i in range(n)
    ]
    vectors = rng.normal(size=(n, 12))
    query = rng.normal(size=12)
    primary = m.original.FeedbackBucket(records, vectors)
    new = m.VerificationBucket(records, vectors)
    selected, trace = primary.retrieve(query_text, query)
    other, _, checked = new.retrieve(query_text, query)
    assert selected == other
    assert_match(trace, checked)
    assert new.idf == primary.bm25.idf


def test_true_dense_ties_and_input_permutation_keep_id_order():
    records = [dict(id=i, text="same common") for i in ["c", "a", "b"]]
    vectors = np.ones((3, 4))
    query = np.ones(4)
    first = m.VerificationBucket(records, vectors).retrieve("same", query)
    second = m.VerificationBucket(records[::-1], vectors[::-1]).retrieve("same", query)
    assert first == second and first[0]["id"] == "a"
    assert_match(
        m.original.FeedbackBucket(records, vectors).retrieve("same", query)[1], first[2]
    )


def test_builder_only_changes_the_checker_class():
    old = m.original.make_phase
    new = m.phase_builder()
    assert old.__code__ is new.__code__
    assert {
        k for k in new.__globals__ if new.__globals__[k] is not old.__globals__[k]
    } == {"LocalBucket"}
    assert new.__globals__["FeedbackBucket"] is m.original.FeedbackBucket
    assert new.__globals__["c"].require is m.c.require


def test_controller_keeps_both_earlier_recoveries():
    old = m.previous.bound_controller()
    with patch.object(m.previous, "bound_controller", lambda: old):
        new = m.bound_controller()
    assert old.__code__ is new.__code__ is m.original.run.__code__
    assert {
        k for k in new.__globals__ if new.__globals__[k] is not old.__globals__[k]
    } == {"make_phase"}
    assert new.__globals__["extract"] is m.previous.extract
    assert new.__globals__["q"].execute is m.previous.previous.execute


def minimal_phase(checker):
    row, records, vectors, query = real_failure()
    root = m.c.ITERATIVE / "offline_check" / ("rk" + uuid.uuid4().hex[:6])
    source_ids = [r["provenance"]["source_id"] for r in records]
    bases = {row["key"]: {"messages": ["base"], "context": {}}}
    adjusted = dict(row, scope_node_index=0)
    data = (
        None,
        {i: n for n, i in enumerate(source_ids)},
        None,
        bases,
        dict(zip(source_ids, records)),
        vectors,
        np.array([query]),
    )
    renderer = SimpleNamespace(render_messages=lambda *_: ["base"])
    fn = m.previous.bind(
        m.phase_builder(), LocalBucket=checker, validate=lambda: None, renderer=renderer
    )
    return fn, root, adjusted, source_ids, data


@pytest.mark.parametrize("fault", ["selection", "ranking", "score"])
def test_selection_rank_and_score_mismatches_still_stop(fault):
    class Corrupt(m.VerificationBucket):
        def retrieve(self, *args):
            selected, similarity, trace = super().retrieve(*args)
            if fault == "selection":
                selected = dict(selected, id="wrong")
            if fault == "ranking":
                trace["dense"][0], trace["dense"][1] = (
                    trace["dense"][1],
                    trace["dense"][0],
                )
            if fault == "score":
                trace["dense"][0]["score"] += 1e-6
            return selected, similarity, trace

    fn, root, row, ids, data = minimal_phase(Corrupt)
    with patch.object(m.original.tokens, "tokenizer", lambda: None):
        with pytest.raises(
            RuntimeError,
            match={
                "selection": "retrieval selection",
                "ranking": "ranking differs",
                "score": "scores differ",
            }[fault],
        ):
            fn(root, "probe", [row], {"test": ids}, {}, data)
    assert (
        not (root / "complete.json").exists() and not (root / "execution/live").exists()
    )


def test_old_checker_still_fails_at_the_real_prompt_builder_guard():
    fn, root, row, ids, data = minimal_phase(m.original.LocalBucket)
    with patch.object(m.original.tokens, "tokenizer", lambda: None):
        with pytest.raises(RuntimeError, match="Adaptive ranking differs"):
            fn(root, "probe", [row], {"test": ids}, {}, data)


def test_new_phase_with_live_files_requires_provenance_before_any_builder():
    root = m.c.ITERATIVE / "offline_check" / ("rv" + uuid.uuid4().hex[:6])
    with patch.object(m.c, "ITERATIVE", root):
        phase = root / "phases/r3_evaluation"
        (phase / "execution/live").mkdir(parents=True)
        with patch.object(
            m, "phase_builder", side_effect=AssertionError("Must not rebuild")
        ):
            with pytest.raises(FileNotFoundError):
                m.make_phase(phase, "evaluation", [], {}, {}, None)


def test_final_receipt_requires_all_five_phase_markers_and_detects_changes():
    root = m.c.ITERATIVE / "offline_check" / ("rf" + uuid.uuid4().hex[:6])
    manifest = root / "manifest.json"
    m.c.save(manifest, {"fixture": True})
    with patch.object(m.c, "ITERATIVE", root), patch.object(
        m, "BASE", root / "amendment"
    ), patch.object(m, "MANIFEST", manifest), patch.object(
        m.previous, "BASE", root / "previous"
    ), patch.object(
        m.previous, "final_check", lambda: None
    ):
        for name in (
            "plan.json",
            "complete.json",
            "previous/live_path_runtime_complete.json",
        ):
            m.c.save(root / name, {"fixture": True})
        for name in sorted(m.NEW_PHASES):
            phase = root / "phases" / name
            for leaf in (
                "scientific_plan_v1.json",
                "prompts.jsonl",
                "independent_verification.json",
                "complete.json",
            ):
                m.c.save(phase / leaf, {"fixture": True, "phase": name})
            m.record_phase(phase)
        m.final_check()
        receipt = m.BASE / "live_ranking_runtime_complete.json"
        before = m.c.sha(receipt)
        m.final_check()
        assert m.c.sha(receipt) == before
        assert len(m.c.read(receipt)["phase_markers"]) == 5
        (root / "phases/r4_evaluation/prompts.jsonl").write_text(
            '{"fixture_tamper":true}', encoding="utf8"
        )
        with pytest.raises(m.a.IntegrityError, match="provenance differs"):
            m.final_check()

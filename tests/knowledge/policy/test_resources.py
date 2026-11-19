"""Tests for policy resources."""

import math
import numpy as np
from analysis.knowledge.policy import resources as p


def test_resources_boundaries():
    checks = []
    for metadata, wanted in [
        (None, [-1, -1, -1]),
        ({}, [-1, -1, -1]),
        ({"usage": None}, [-1, -1, -1]),
        ({"usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}, [0, 0, 0]),
        (
            {"usage": {"prompt_tokens": 10, "completion_tokens": None, "total_tokens": 12}},
            [10, -1, 12],
        ),
    ]:
        assert p.usage(metadata) == wanted
        checks.append("usage_unknown_or_zero")
    for bad in [
        dict(prompt_tokens=-1),
        dict(prompt_tokens=True),
        dict(prompt_tokens=1.5),
        dict(prompt_tokens=2, completion_tokens=3, total_tokens=6),
    ]:
        try:
            p.usage({"usage": bad})
        except p.c.a.IntegrityError:
            checks.append("reject_bad_usage")
        else:
            raise AssertionError("Malformed usage accepted")
    out = p.span_totals([(10.0, 20.0), (25.0, 40.0)])
    assert (
        out["batch_attempt_s"] == 25
        and out["between_batch_s"] == 5
        and (out["first_to_last_attempt_s"] == 30)
    )
    checks.append("disjoint_timeline_partition")
    for bad in [[], [(3, 2)], [(0, 2), (1, 3)], [(0, float("nan"))]]:
        try:
            p.span_totals(bad)
        except p.c.a.IntegrityError:
            checks.append("reject_bad_wall_spans")
        else:
            raise AssertionError("Invalid timeline accepted")
    meta = [
        dict(
            index=0,
            key="a",
            condition="a",
            n_valid=3,
            repairs=1,
            usable_final=True,
            vote_request_chain_s=[8.0, 2.0, 2.0],
            prediction_request_chain_s=12.0,
        ),
        dict(
            index=1,
            key="b",
            condition="a",
            n_valid=0,
            repairs=0,
            usable_final=False,
            vote_request_chain_s=[2.0, 2.0, 2.0],
            prediction_request_chain_s=6.0,
        ),
    ]
    events = []
    for i in range(2):
        for vote in range(3):
            events.append([i, vote, 0, 0, 0, 1, 2.0, 0.0, 10.0, 2.0, 12.0])
    events.extend(
        [
            [0, 0, 1, 0, 1, 0, 1.0, 0.0, -1.0, -1.0, -1.0],
            [0, 0, 1, 1, 1, 1, 3.0, 2.0, 20.0, 4.0, 24.0],
        ]
    )
    data = np.array(events, float)
    r = p.group_stats(meta, data, [0, 1])
    assert (
        r["physical_POST_attempts"] == 8
        and r["successful_responses"] == 7
        and (r["ordinary_logical_calls"] == 6)
        and (r["repair_logical_calls"] == 1)
    )
    assert (
        r["retry_attempts"] == 1
        and r["failed_attempts"] == 1
        and (r["valid_final"] == 1)
        and (r["null_final"] == 1)
    )
    assert (
        r["summed_transport_s"] == 16
        and r["backoff_s"] == 2
        and (r["prediction_request_chain_s"]["sum"] == 18)
    )
    assert (
        r["usage"]["all_success"]["total_tokens"]["sum"] == 96
        and r["per_valid_voted_answer"]["total_tokens"] == 96
    )
    assert (
        r["per_voted_answer"]["total_tokens"] == 48
        and r["usage"]["repair"]["total_tokens"]["sum"] == 24
    )
    checks.extend(
        [
            "logical_vs_physical_calls",
            "failed_usage_not_zero_success",
            "transport_vs_backoff",
            "voted_vs_valid_denominator",
            "repair_usage_included",
        ]
    )
    changed = data.copy()
    changed[0, p.IX["prompt_tokens"]] = -1
    r = p.group_stats(meta, changed, [0, 1])
    assert (
        r["usage"]["all_success"]["prompt_tokens"]["unknown"] == 1
        and r["usage"]["all_success"]["prompt_tokens"]["sum"] == 70
    )
    checks.append("partial_unknown_not_imputed")
    r = p.group_stats(meta, data, [1])
    assert r["per_valid_voted_answer"] is None and r["physical_POST_attempts"] == 3
    checks.append("no_valid_answers_no_infinite_efficiency")
    x = p.stats([1, 2, 9])
    assert x["sum"] == 12 and x["median"] == 2 and (x["mean"] == 4)
    assert p.stats([])["mean"] is None
    checks.append("summary_and_empty_group")

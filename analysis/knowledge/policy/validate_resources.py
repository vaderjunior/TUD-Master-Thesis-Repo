from collections import Counter, defaultdict
from datetime import datetime, timezone
import base64
import hashlib
import json
import math
import numpy as np
from analysis.knowledge.policy import resources as p
from analysis.knowledge.policy import practical_comparisons as practical

c = p.c


def close(actual, wanted, label):
    assert math.isclose(float(actual), float(wanted), rel_tol=1e-11, abs_tol=1e-06), (
        label,
        actual,
        wanted,
    )


def percentile(values, q):
    values = sorted(values)
    position = q * (len(values) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    return values[lo] + (values[hi] - values[lo]) * (position - lo)


def check_stats(saved, values, label):
    assert saved["n"] == len(values), label
    close(saved["sum"], math.fsum(values), label + "/sum")
    if not values:
        assert saved["mean"] is None, label
        return
    for name, wanted in [
        ("mean", math.fsum(values) / len(values)),
        ("min", min(values)),
        ("max", max(values)),
        *[
            (name, percentile(values, q))
            for name, q in [("median", 0.5), ("p90", 0.9), ("p95", 0.95), ("p99", 0.99)]
        ],
    ]:
        close(saved[name], wanted, label + "/" + name)


def run():
    contract = c.read(p.BASE / "analysis_contract.json")
    c.verify(contract["inputs"])
    receipt = c.read(p.BASE / "extraction_complete.json")
    c.verify(receipt["output_bindings"])
    c.verify(c.read(p.BASE / "boundary_tests.json")["sources"])
    meta = list(c.rows(p.BASE / "prediction_resources.jsonl"))
    events = np.load(p.BASE / "event_resources.npz", allow_pickle=False)["events"]
    assert len(meta) == 62154 and len(events) == 193872
    assert all((r["index"] == i for i, r in enumerate(meta)))
    by_prediction = defaultdict(list)
    for row in events:
        by_prediction[int(row[0])].append(row.tolist())
    roster = {r["key"]: r for r in c.rows(c.SETUP / "roster.jsonl")}
    ep = c.read(c.SETUP / "endpoints.json")
    selected = {"all": list(range(len(meta)))}
    for cond in p.NAMES:
        selected["condition/" + cond] = [
            r["index"] for r in meta if r["condition"] == cond
        ]
    for lang in ("en", "de"):
        for cond in p.NAMES:
            selected["language/" + lang + "/" + cond] = [
                r["index"]
                for r in meta
                if r["condition"] == cond and roster[r["key"]]["lang"] == lang
            ]
    data = c.read(p.BASE / "resources.json")
    groups = dict(data["groups"])
    for name, e in ep.items():
        encoded = np.load(p.RUN / name / "encoded.npz", allow_pickle=False)
        keys = {k for k, v in zip(e["keys"], encoded["common"]) if v}
        for cond in p.NAMES:
            group = "endpoint/" + name + "/" + cond
            selected[group] = [
                r["index"] for r in meta if r["key"] in keys and r["condition"] == cond
            ]
            groups[group] = data["endpoint_common_costs"][name][cond]
    checked = []
    for name, indices in selected.items():
        saved = groups[name]
        records = [meta[i] for i in indices]
        rows = [v for i in indices for v in by_prediction[i]]
        successful = [v for v in rows if v[5] == 1]
        ordinary = [v for v in successful if v[4] == 0]
        repair = [v for v in successful if v[4] == 1]
        expected = dict(
            predictions=len(indices),
            unique_inputs=len({r["key"] for r in records}),
            valid_final=sum((r["usable_final"] for r in records)),
            null_final=sum((not r["usable_final"] for r in records)),
            valid_inner=sum((r["n_valid"] for r in records)),
            repaired_predictions=sum((r["repairs"] > 0 for r in records)),
            ordinary_logical_calls=len(ordinary),
            repair_logical_calls=len(repair),
            successful_responses=len(successful),
            physical_POST_attempts=len(rows),
            retry_attempts=sum((v[3] > 0 for v in rows)),
            failed_attempts=sum((v[5] == 0 for v in rows)),
        )
        assert all((saved[k] == v for k, v in expected.items())), (name, expected)
        close(
            saved["summed_transport_s"],
            math.fsum((v[6] for v in rows)),
            name + "/transport",
        )
        close(saved["backoff_s"], math.fsum((v[7] for v in rows)), name + "/backoff")
        close(
            saved["prediction_request_chain_s"]["sum"],
            math.fsum((v[6] + v[7] for v in rows)),
            name + "/chain",
        )
        for kind, values in [
            ("all_success", successful),
            ("ordinary", ordinary),
            ("repair", repair),
        ]:
            for column, label in [
                (8, "prompt_tokens"),
                (9, "completion_tokens"),
                (10, "total_tokens"),
            ]:
                known = [int(v[column]) for v in values if v[column] >= 0]
                target = saved["usage"][kind][label]
                assert target["known"] == len(known) and target["unknown"] == len(
                    values
                ) - len(known)
                check_stats(target, known, name + "/" + kind + "/" + label)
            assert all((int(v[8]) + int(v[9]) == int(v[10]) for v in values))
        for key, col in [
            ("prompt_tokens", 8),
            ("completion_tokens", 9),
            ("total_tokens", 10),
        ]:
            total = sum((int(v[col]) for v in successful))
            close(
                saved["per_voted_answer"][key],
                total / len(indices),
                name + "/per_voted",
            )
            close(
                saved["per_valid_voted_answer"][key],
                total / expected["valid_final"],
                name + "/per_valid",
            )
        close(
            saved["per_voted_answer"]["successful_requests"],
            len(successful) / len(indices),
            name + "/requests_per_answer",
        )
        for label, values in [
            (
                "prediction_request_chain_s",
                [r["prediction_request_chain_s"] for r in records],
            ),
            (
                "vote_request_chain_s",
                [v for r in records for v in r["vote_request_chain_s"]],
            ),
            ("successful_response_s", [v[6] for v in successful]),
            ("ordinary_success_s", [v[6] for v in ordinary]),
            ("repair_success_s", [v[6] for v in repair]),
            ("failed_attempt_s", [v[6] for v in rows if v[5] == 0]),
        ]:
            check_stats(saved[label], values, name + "/" + label)
        checked.append(name)
    assert len(checked) == 28
    for name, old in c.read(p.P5 / "process_counts.json").items():
        now = groups["condition/" + name]
        assert (
            now["null_final"] == old["missing"]
            and now["repair_logical_calls"] == old["repair_calls"]
        )
        assert now["repaired_predictions"] == old["repaired_predictions"]
    assert (
        groups["all"]["successful_responses"] == 193864
        and groups["all"]["repair_logical_calls"] == 7402
    )
    assert all((all((v == -1 for v in row[8:])) for row in events if row[5] == 0))
    samples = set((int(row[0]) for row in events if row[5] == 0))
    for cond in p.NAMES:
        for repeat in range(3):
            candidates = [
                r["index"]
                for r in meta
                if r["condition"] == cond and r["repeat"] == repeat
            ]
            samples.update((candidates[0], candidates[-1]))
        candidates = [r for r in meta if r["condition"] == cond]
        samples.add(
            max(candidates, key=lambda r: r["prediction_request_chain_s"])["index"]
        )
        samples.add(max(candidates, key=lambda r: r["repairs"])["index"])
    samples.add(next((r["index"] for r in meta if not r["usable_final"])))
    samples.add(max(meta, key=lambda r: max(r["ordinary_server_inputs"]))["index"])
    envelope_counts = Counter()
    sample_bindings = {}
    for i in sorted(samples):
        row = meta[i]
        folder = c.OUTPUT / "live/items" / row["stamp"]
        result = c.read(folder / "result.json")
        assert c.sha(folder / "result.json") == row["result_sha256"]
        vote_s = [0.0, 0.0, 0.0]
        assert len(result["events"]) == len(by_prediction[i])
        for event, compact in zip(result["events"], by_prediction[i]):
            path = (
                folder
                / "calls"
                / f"vote_{event['vote']}_turn_{event['turn']}"
                / f"attempt_{event['attempt']}"
            )
            intent = c.read(path / "intent.json")
            outcome = c.read(path / "outcome.json")
            assert c.a.signature(intent) == outcome["intent_sha256"]
            assert (
                intent["backoff_elapsed_s"] == compact[7]
                and outcome["elapsed_s"] == compact[6]
            )
            vote_s[event["vote"]] += outcome["elapsed_s"] + intent["backoff_elapsed_s"]
            if outcome["kind"] == "http_response":
                body = base64.b64decode(outcome["body_base64"], validate=True)
                assert hashlib.sha256(body).hexdigest() == outcome["body_sha256"]
                if 200 <= outcome["status"] < 300:
                    raw = json.loads(body)
                    usage = raw["usage"]
                    assert [
                        usage[k]
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                    ] == compact[8:]
                    assert raw["model"] == "Qwen/Qwen3-VL-32B-Instruct"
                    assert raw["choices"][0]["finish_reason"] == "stop"
                    envelope_counts["successful"] += 1
                else:
                    assert outcome["status"] == 502 and compact[5] == 0
                    envelope_counts["http_502"] += 1
            else:
                assert outcome["category"] == "ConnectTimeout" and compact[5] == 0
                envelope_counts["ConnectTimeout"] += 1
            sample_bindings.update(
                c.bindings([path / "intent.json", path / "outcome.json"])
            )
        for a, b in zip(vote_s, row["vote_request_chain_s"]):
            close(a, b, "raw vote duration")
        sample_bindings.update(c.bindings([folder / "result.json"]))
    assert envelope_counts["http_502"] == 3 and envelope_counts["ConnectTimeout"] == 5
    timing = c.read(p.BASE / "timing.json")
    c.verify(timing["bindings"])
    wall = timing["wall"]
    stages = timing["stages"]
    assert len(stages) == 47 and all((s["workers"] == 128 for s in stages))
    assert (
        all((s["peak_submitted"] == 128 for s in stages[:-1]))
        and stages[-1]["peak_submitted"] == 54
    )
    assert sum((s["predictions"] for s in stages)) == 62154
    assert sum((s["POST_attempts"] for s in stages)) == 193872
    assert sum((s["successful_responses"] for s in stages)) == 193864
    previous = None
    gap = 0.0
    active = 0.0
    for s in stages:
        start = datetime.fromisoformat(s["start_utc"])
        finish = datetime.fromisoformat(s["finish_utc"])
        seconds = (finish - start).total_seconds()
        close(seconds, s["attempt_s"], "raw stage wall")
        active += seconds
        if previous is not None:
            assert start >= previous
            gap += (start - previous).total_seconds()
        previous = finish
    begin = datetime.fromisoformat(wall["invocation_start_utc"])
    end = datetime.fromisoformat(wall["invocation_finish_utc"])
    close(wall["invocation_s"], (end - begin).total_seconds(), "invocation wall")
    close(wall["batch_attempt_s"], active, "batch wall")
    close(wall["between_batch_s"], gap, "gaps")
    close(wall["time_partition_check_s"], wall["invocation_s"], "complete partition")
    assert len(timing["context_metadata"]) == 47
    assert all(
        (
            x["effective_reported_limit"] == 75000
            and x["reference_input_guard"] == 8192
            and (not x["limit_verified_by_benchmark"])
            for x in timing["context_metadata"]
        )
    )
    close(
        wall["model_list_GET_s"],
        sum((s["model_list_GET_s"] for s in stages)),
        "recorded GET duration",
    )
    storage = c.read(p.BASE / "storage.json")
    for name in ("raw_D", "execution_preparation_C", "design_setup_C"):
        inv = storage[name]
        for group in ("groups", "by_basename_or_suffix"):
            assert sum((v["files"] for v in inv[group].values())) == inv["files"]
            assert (
                sum((v["bytes"] for v in inv[group].values())) == inv["logical_bytes"]
            )
    d = storage["raw_D"]
    assert d["groups"]["live/items"]["files"] == 62154 * 7 + 193872 * 2
    assert d["by_basename_or_suffix"]["intent.json"]["files"] == 193872 + 27
    assert d["by_basename_or_suffix"]["result.json"]["files"] == 62154 + 9
    mock = list((c.OUTPUT / "offline_checks").rglob("result.json"))
    assert len(mock) == 9
    assert (
        sum((r["result_bytes"] for r in meta)) + sum((x.stat().st_size for x in mock))
        == d["by_basename_or_suffix"]["result.json"]["bytes"]
    )
    quality = c.read(p.BASE / "quality_cost_matrix.json")["rows"]
    assert len(quality) == 18
    for r in quality:
        point = c.read(p.RUN / r["endpoint"] / "results.json")["point"]
        j = p.NAMES.index(r["condition"])
        for field in ("macro_f1", "micro_f1", "exact"):
            value = point.get(field, [None] * 3)[j]
            if value is None:
                assert r[field] is None
            else:
                close(r[field], value, "quality copy/" + field)
        assert (
            groups["endpoint/" + r["endpoint"] + "/" + r["condition"]]["predictions"]
            == 3 * r["n"]
        )
    for delta_fp, delta_fn in [
        (-3, -2),
        (3, 2),
        (0, 0),
        (-4, 2),
        (4, -2),
        (0, 2),
        (0, -2),
        (-2, 0),
        (2, 0),
    ]:
        item = practical.cost_direction(delta_fp, delta_fn)
        root = item["break_even_FN_over_FP"]
        if root is not None:
            close(delta_fp + root * delta_fn, 0, "illustrative cost boundary")
    for r in c.read(p.BASE / "illustrative_error_costs.json")["rows"]:
        if r["group"] == "whole":
            value = c.read(p.RUN / r["endpoint"] / "results.json")["point"][
                "confusion_mean"
            ]
        else:
            original = c.read(p.P4 / "groups_and_sources.json")[r["endpoint"]][
                "groups"
            ][r["group"]]["confusion_repeats"]
            value = {k: [sum(v) / 3 for v in original[k]] for k in ("fp", "fn")}
        left = p.NAMES.index(r["left"])
        right = p.NAMES.index(r["right"])
        close(r["delta_FP"], value["fp"][left] - value["fp"][right], "utility FP")
        close(r["delta_FN"], value["fn"][left] - value["fn"][right], "utility FN")
    no_repair_keys = {
        k for k in roster if all((r["repairs"] == 0 for r in meta_by_key(meta, k)))
    }
    diagnostics = {}
    for cond in p.NAMES:
        items = [r for r in meta if r["condition"] == cond]
        diagnostics[cond] = dict(
            no_repair_prediction=p.stats(
                [r["prediction_request_chain_s"] for r in items if r["repairs"] == 0]
            ),
            repaired_prediction=p.stats(
                [r["prediction_request_chain_s"] for r in items if r["repairs"] > 0]
            ),
            common_no_repair_all_nine=p.stats(
                [
                    r["prediction_request_chain_s"]
                    for r in items
                    if r["key"] in no_repair_keys
                ]
            ),
        )
    p.save(
        "timing_diagnostics.json",
        dict(
            common_no_repair_inputs=len(no_repair_keys),
            conditions=diagnostics,
            caution="Condition-specific repair splits are different populations. Common no-repair is an output-selected diagnostic under the same mixed queue, not a repairs-disabled experiment or deployment latency benchmark.",
        ),
    )
    p.save(
        "independent_checks.json",
        dict(
            passed=True,
            resource_groups_checked=len(checked),
            groups=checked,
            raw_prediction_samples=len(samples),
            raw_envelopes=dict(envelope_counts),
            sample_bindings=sample_bindings,
            timestamps_utc=dict(
                start=begin.astimezone(timezone.utc).isoformat(),
                finish=end.astimezone(timezone.utc).isoformat(),
            ),
            counts_verified=dict(
                predictions=len(meta),
                physical_attempts=len(events),
                successful_responses=193864,
                ordinary=186462,
                repairs=7402,
                unknown_failed_usage=8,
                batches=47,
                quality_rows=18,
            ),
            extra_cost_boundary_checks=9,
            storage_reconciliation="All category sums; live item file formula; nine local mock results and 27 mock requests kept separate.",
            no_network=True,
            scripts=c.bindings([__file__, p.__file__, practical.__file__]),
        ),
    )
    print(
        f"Independent verification passed: {len(checked)} resource groups, {len(samples)} raw prediction samples, all eight retries, 47 batches and 18 quality/cost rows.",
        flush=True,
    )


_key_index = None


def meta_by_key(meta, key):
    global _key_index
    if _key_index is None:
        _key_index = defaultdict(list)
        for row in meta:
            _key_index[row["key"]].append(row)
    return _key_index[key]


if __name__ == "__main__":
    with c.offline():
        run()

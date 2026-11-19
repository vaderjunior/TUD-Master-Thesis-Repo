"""German guidelines resources."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import time
import numpy as np
from analysis.knowledge.german_guidelines import robustness as r

c, h, l, reader = (r.c, r.h, r.l, r.reader)
offline, require = (r.offline, r.require)
BASE = c.ROOT / "experiments/astra/german_guidelines_synthesis_20261113_v1"
PHASES = ("de_dev_eval", "de_legal_dev_eval", "de_test", "de_legal_test")
CONDITIONS = r.CONDITIONS
EVENT_COLUMNS = (
    "prediction",
    "kind_repair",
    "success",
    "attempt",
    "elapsed_s",
    "backoff_s",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)
PRIOR_REPORTS = [
    c.ROOT / "Thesis_Log/astra_result" / name
    for name in (
        "german_guidelines_part1_reader_20261112_v1.md",
        "german_guidelines_part2_hate_20261112_v1.md",
        "german_guidelines_part3_legal_20261113_v1.md",
        "german_guidelines_part4_context_20261113_v1.md",
        "german_guidelines_part5_datasets_cases_20261113_v1.md",
        "german_guidelines_part6_robustness_20261113_v1.md",
        "astra_sq1_components_german_results_20260923_v1.md",
        "astra_sq1_guideline_final_results_20260926_v1.md",
        "astra_sq2_broad_final_synthesis_20261104_v2.md",
        "astra_sq2_combined_final_synthesis_20261104_v1.md",
    )
]
OLD_EVIDENCE = [
    c.ROOT / "Thesis_Log/astra_result/evidence" / name
    for name in (
        "sq1_components_de_dev_eval_results_20260923_v1.json",
        "sq1_components_de_legal_dev_eval_results_20260923_v1.json",
        "sq1_guideline_final_results_20260926_v1.json",
    )
]


def save(path, value):
    require(c.normal(path).is_relative_to(BASE), "Part 7 write outside new folder")
    c.save(path, h.safe(value))


def rowsave(path, values):
    require(c.normal(path).is_relative_to(BASE), "Part 7 rows outside new folder")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        for value in values:
            f.write(
                json.dumps(
                    h.safe(value), sort_keys=True, ensure_ascii=False, allow_nan=False
                )
                + "\n"
            )


def stats(values):
    a = np.asarray(values, dtype=float)
    require(a.ndim == 1 and np.isfinite(a).all(), "Nonfinite or nonvector statistics")
    if not len(a):
        return {
            "n": 0,
            "sum": 0.0,
            "mean": None,
            "min": None,
            "median": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return dict(
        n=len(a),
        sum=float(a.sum()),
        mean=float(a.mean()),
        min=float(a.min()),
        median=float(np.quantile(a, 0.5)),
        p90=float(np.quantile(a, 0.9)),
        p95=float(np.quantile(a, 0.95)),
        p99=float(np.quantile(a, 0.99)),
        max=float(a.max()),
    )


def usage(metadata):
    value = metadata.get("usage") if isinstance(metadata, dict) else None
    if value is None:
        return [-1, -1, -1]
    require(isinstance(value, dict), "Malformed saved usage")
    out = []
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = value.get(key)
        require(v is None or (type(v) is int and v >= 0), "Invalid usage counter")
        out.append(-1 if v is None else v)
    if all((x >= 0 for x in out)):
        require(out[0] + out[1] == out[2], "Server token arithmetic differs")
    return out


def wall_intervals(spans):
    spans = sorted(spans)
    require(
        spans
        and all((math.isfinite(x) and math.isfinite(y) and (x <= y) for x, y in spans)),
        "Invalid clock spans",
    )
    require(
        all((spans[i][1] <= spans[i + 1][0] for i in range(len(spans) - 1))),
        "Batch overlap unexpected",
    )
    active = sum((b - a for a, b in spans))
    window = spans[-1][1] - spans[0][0]
    return {
        "attempt_seconds": active,
        "first_to_last_attempt_seconds": window,
        "between_attempt_seconds": window - active,
        "gaps_seconds": [spans[i + 1][0] - spans[i][1] for i in range(len(spans) - 1)],
    }


def sources():
    return [
        Path(__file__),
        c.ROOT / "analysis/knowledge/german_guidelines/validate_resource_snapshot.py",
        c.ROOT / "tests/knowledge/german_guidelines/test_resources.py",
    ]


def prepare():
    _, rp = reader.read_verified()
    _, rob = r.verified()
    c.validate()
    tests = sorted((BASE / "tests").glob("*/report.json"))
    require(tests and c.read(tests[-1])["passed"], "Passing tests missing")
    require(
        c.read(tests[-1])["sources"] == c.bindings(sources()),
        "Code changed after tests",
    )
    bound = [
        c.PLAN,
        c.PROTOCOL,
        c.BASE / "prompts.jsonl",
        reader.BASE / "complete.json",
        rp / "manifest.json",
        r.BASE / "complete.json",
        r.BASE / "report/part6_complete.json",
        rob / "manifest.json",
        c.OUTPUT / "live/complete.json",
        c.ROOT / "analysis/common/api.py",
        c.ROOT / "analysis/knowledge/german_guidelines/run.py",
        *PRIOR_REPORTS,
        *OLD_EVIDENCE,
        *sources(),
        tests[-1],
    ]
    save(
        BASE / "analysis_contract.json",
        {
            "status": "part7_scope_before_new_resource_totals",
            "created_utc": c.a.utc(),
            "bindings": c.bindings(bound),
            "scope": "Consolidate verified effectiveness without rescoring, add descriptive runtime/token/storage accounting and bounded scientific synthesis. Writer handover is Part 8.",
            "request_units": "Ordinary and repair logical calls, successful responses and physical POST attempts remain distinct. Model-list GETs are separate. Failed-attempt token work is unknown, never imputed zero.",
            "latency": "Saved event elapsed seconds are client transport durations. Inner-vote latency sums attempts/backoff including repairs. Prediction request-chain sum excludes disk/parsing/queue overhead and is not wall completion latency. Concurrent sums are not wall or GPU time.",
            "wall": "Batch start/summary timestamps define measured attempt spans and between-attempt gaps. Gaps contain audit/check/other overhead, not pure audit time. Final completion has no embedded timestamp; its filesystem mtime is reported separately as a weaker approximate endpoint.",
            "tokens": "Successful server usage, ordinary/repair separately. Track missing fields and arithmetic. Compare paired initial prompt tokens for identical inputs, preserving all conditions/repeats. Reference tokenizer counts are not assumed identical to the server.",
            "storage": "Read-only metadata walk of only this completed D: study and its frozen C: preparation; logical file bytes, not allocated disk. No content rehash/full raw audit or deletion. Record path/size metadata digest and aggregate categories.",
            "practical": "Observed hate FP/FN costs may be explained with a labelled illustrative break-even ratio on this cohort, not an application-selected utility or deployment rule. No new statistical claims from resource tables.",
            "cross_study": "Use verified earlier German DEV component removal, English benign HateCheck guideline-delivery, and broader full/small/no-retrieval results with separate masks, wording, dates and roles. No pooling or revised main configuration.",
            "limits": "Late exposed data, one rule bank/model alias, language/content/length/position confounds, partial repairs, source dependence and Part 6 inference boundary remain explicit.",
            "actual_Qwen_calls": 0,
            "D_writes": 0,
        },
    )
    print("Part 7 scope frozen; offline resource calculation can proceed.", flush=True)


def group_stats(meta, events, indices):
    selected = np.isin(events[:, 0].astype(int), indices)
    ev = events[selected]
    ok = ev[:, 2] == 1
    ans = {
        "predictions": len(indices),
        "nulls": sum((meta[i]["n_valid"] == 0 for i in indices)),
        "repaired_predictions": sum((meta[i]["repairs"] > 0 for i in indices)),
        "ordinary_logical_calls": 3 * len(indices),
        "repair_logical_calls": sum((meta[i]["repairs"] for i in indices)),
        "physical_POST_attempts": len(ev),
        "successful_responses": int(ok.sum()),
        "failed_attempts": int((~ok).sum()),
        "retry_attempts": int((ev[:, 3] > 0).sum()),
        "backoff_seconds": float(ev[:, 5].sum()),
        "all_transport_seconds": float(ev[:, 4].sum()),
        "successful_latency_s": stats(ev[ok, 4]),
        "ordinary_latency_s": stats(ev[ok & (ev[:, 1] == 0), 4]),
        "repair_latency_s": stats(ev[ok & (ev[:, 1] == 1), 4]),
        "failed_attempt_latency_s": stats(ev[~ok, 4]),
        "prediction_request_chain_s": stats(
            [meta[i]["request_chain_s"] for i in indices]
        ),
        "vote_request_chain_s": stats(
            [t for i in indices for t in meta[i]["vote_chain_s"]]
        ),
        "usage": {},
    }
    for kind, mask in [
        ("all_success", ok),
        ("ordinary", ok & (ev[:, 1] == 0)),
        ("repair", ok & (ev[:, 1] == 1)),
    ]:
        u = ev[mask]
        ans["usage"][kind] = {"responses": len(u)}
        for field, col in [
            ("prompt_tokens", 6),
            ("completion_tokens", 7),
            ("total_tokens", 8),
        ]:
            known = u[:, col] >= 0
            ans["usage"][kind][field] = {
                "known_responses": int(known.sum()),
                "unknown_responses": int((~known).sum()),
                **stats(u[known, col]),
            }
    return ans


def inventory(root):
    root = c.normal(root)
    require(root in (c.OUTPUT, c.BASE), "Inventory outside the two study roots")
    require(not root.is_symlink() and (not root.is_junction()), "Linked storage root")
    stack = [root]
    groups = defaultdict(lambda: Counter(files=0, bytes=0))
    types = defaultdict(lambda: Counter(files=0, bytes=0))
    records = []
    while stack:
        path = stack.pop()
        with os.scandir(path) as entries:
            for entry in entries:
                require(
                    not entry.is_symlink() and (not Path(entry.path).is_junction()),
                    "Linked inventory entry",
                )
                if entry.is_dir(follow_symlinks=False):
                    stack.append(Path(entry.path))
                    continue
                require(
                    entry.is_file(follow_symlinks=False), "Unexpected storage object"
                )
                st = entry.stat(follow_symlinks=False)
                rel = Path(entry.path).relative_to(root).as_posix()
                parts = rel.split("/")
                group = (
                    "/".join(parts[:2])
                    if parts[0] == "live" and len(parts) > 2
                    else parts[0] if len(parts) > 1 else "root_files"
                )
                groups[group].update(files=1, bytes=st.st_size)
                types[
                    (
                        entry.name
                        if entry.name
                        in (
                            "intent.json",
                            "outcome.json",
                            "result.json",
                            "complete.json",
                            "job.json",
                            "item.lock",
                        )
                        else Path(entry.name).suffix or "no_suffix"
                    )
                ].update(files=1, bytes=st.st_size)
                records.append((rel, st.st_size, st.st_mtime_ns))
                if len(records) % 100000 == 0:
                    print(
                        f"Storage metadata only: {len(records):,} files counted.",
                        flush=True,
                    )
    digest = hashlib.sha256()
    for rec in sorted(records):
        digest.update(
            (json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        )
    return {
        "root": str(root),
        "scanned_utc": c.a.utc(),
        "files": len(records),
        "logical_bytes": sum((v[1] for v in records)),
        "groups": dict(groups),
        "by_basename_or_suffix": dict(types),
        "path_size_mtime_metadata_sha256": digest.hexdigest(),
        "method": "No file contents read by inventory; regular-file sizes only, not disk blocks or independently rehashed evidence.",
        "final_completion_mtime_utc": (
            datetime.fromtimestamp(
                (root / "live/complete.json").stat().st_mtime, timezone.utc
            ).isoformat()
            if root == c.OUTPUT
            else None
        ),
    }


def run():
    contract = c.read(BASE / "analysis_contract.json")
    c.verify(contract["bindings"])
    _, rp = reader.read_verified()
    _, rob = r.verified()
    out = BASE / "runs" / f"run_{time.time_ns()}"
    out.mkdir(parents=True)
    save(
        out / "start.json",
        {
            "created_utc": c.a.utc(),
            "contract_sha256": c.sha(BASE / "analysis_contract.json"),
        },
    )
    meta = []
    events = []
    ordinary_tokens = defaultdict(list)
    usage_details = Counter()
    retries = Counter()
    for row in c.rows(rp / "predictions.jsonl"):
        i = len(meta)
        evs = row["events"]
        lat = [0.0, 0.0, 0.0]
        local = Counter()
        for e in evs:
            require(
                e["kind"] in ("ordinary", "repair") and e["action"] in ("ok", "retry"),
                "Unexpected saved call status",
            )
            require(
                math.isfinite(e["elapsed_s"])
                and e["elapsed_s"] >= 0
                and math.isfinite(e["backoff_elapsed_s"])
                and (e["backoff_elapsed_s"] >= 0),
                "Invalid elapsed time",
            )
            u = (
                usage(e.get("model_response_metadata"))
                if e["action"] == "ok"
                else [-1, -1, -1]
            )
            events.append(
                [
                    i,
                    int(e["kind"] == "repair"),
                    int(e["action"] == "ok"),
                    e["attempt"],
                    e["elapsed_s"],
                    e["backoff_elapsed_s"],
                    *u,
                ]
            )
            lat[e["vote"]] += e["elapsed_s"] + e["backoff_elapsed_s"]
            local[e["action"]] += 1
            local[e["kind"] + "_" + e["action"]] += 1
            if e["action"] == "ok":
                value = e["model_response_metadata"].get("usage")
                for field in ("prompt_tokens_details", "completion_tokens_details"):
                    usage_details[
                        field
                        + (
                            "_present"
                            if value and value.get(field) is not None
                            else "_absent"
                        )
                    ] += 1
                if e["kind"] == "ordinary":
                    ordinary_tokens[row["key"], row["condition"]].append(u[0])
            else:
                retries[e["response_class"]] += 1
        require(
            np.allclose(lat, row["latencies"], rtol=0, atol=1e-07),
            "Vote-chain latency differs",
        )
        require(
            local["ordinary_ok"] == 3
            and local["repair_ok"] == row["repairs"]
            and (local["ok"] == row["accounting"]["http_success_with_text"])
            and (len(evs) == row["accounting"]["post_attempts"]),
            "Resource accounting differs",
        )
        meta.append(
            {
                "phase": row["phase"],
                "condition": row["condition"],
                "key": row["key"],
                "repeat": row["repeat"],
                "source": row["source"],
                "stage": row["stage"],
                "n_valid": row["n_valid"],
                "repairs": row["repairs"],
                "vote_chain_s": lat,
                "request_chain_s": sum(lat),
                "event_count": len(evs),
            }
        )
    a = np.asarray(events, dtype=float)
    rowsave(out / "prediction_resources.jsonl", meta)
    with (out / "event_resources.npz").open("xb") as f:
        np.savez_compressed(f, events=a)
    groups = {"all": group_stats(meta, a, list(range(len(meta))))}
    for co in CONDITIONS:
        groups["condition/" + co] = group_stats(
            meta, a, [i for i, v in enumerate(meta) if v["condition"] == co]
        )
    for ph in PHASES:
        for co in CONDITIONS:
            groups[ph + "/" + co] = group_stats(
                meta,
                a,
                [
                    i
                    for i, v in enumerate(meta)
                    if v["phase"] == ph and v["condition"] == co
                ],
            )
    save(
        out / "resources.json",
        {
            "groups": groups,
            "columns": EVENT_COLUMNS,
            "usage_details": dict(usage_details),
            "retry_classes": dict(retries),
            "usage_scope": "Successful text-bearing response metadata. Unknown usage for failed attempts is excluded from known totals and not assigned zero.",
        },
    )
    prompts = []
    pt = {}
    for row in c.rows(c.BASE / "prompts.jsonl"):
        k = (row["key"], row["condition"])
        values = ordinary_tokens.pop(k)
        require(len(values) == 9, "Initial prompt usage roster differs")
        known = [x for x in values if x >= 0]
        unique = sorted(set(known))
        record = {
            "key": row["key"],
            "phase": row["population"],
            "condition": row["condition"],
            "reference_tokens": row["reference_input_tokens"],
            "ordinary_server_prompt_values": unique,
            "ordinary_usage_known": len(known),
            "initial_server_prompt_tokens": (
                unique[0] if len(unique) == 1 and len(known) == 9 else None
            ),
        }
        prompts.append(record)
        pt[k] = record
    require(not ordinary_tokens, "Unmatched resource prompt")
    rowsave(out / "prompt_resources.jsonl", prompts)
    prompt_summary = {}
    paired = {}
    for ph in ("all", *PHASES):
        prompt_summary[ph] = {}
        paired[ph] = {}
        for co in CONDITIONS:
            records = [
                v
                for v in prompts
                if v["condition"] == co and (ph == "all" or v["phase"] == ph)
            ]
            known = [
                v for v in records if v["initial_server_prompt_tokens"] is not None
            ]
            prompt_summary[ph][co] = {
                "prompts": len(records),
                "stable_complete_server_usage": len(known),
                "reference_tokens": stats([v["reference_tokens"] for v in records]),
                "server_prompt_tokens": stats(
                    [v["initial_server_prompt_tokens"] for v in known]
                ),
                "server_minus_reference": stats(
                    [
                        v["initial_server_prompt_tokens"] - v["reference_tokens"]
                        for v in known
                    ]
                ),
            }
        for left, right in r.PAIRS:
            rr = [
                v
                for v in prompts
                if v["condition"] == left and (ph == "all" or v["phase"] == ph)
            ]
            vv = [
                (v, pt[v["key"], right])
                for v in rr
                if v["initial_server_prompt_tokens"] is not None
                and pt[v["key"], right]["initial_server_prompt_tokens"] is not None
            ]
            ds = [
                x["initial_server_prompt_tokens"] - y["initial_server_prompt_tokens"]
                for x, y in vv
            ]
            paired[ph][left + "_minus_" + right] = {
                "matched_inputs": len(vv),
                "token_difference": stats(ds),
                "relative_mean_difference": (
                    None
                    if not vv
                    else sum(ds)
                    / sum((y["initial_server_prompt_tokens"] for x, y in vv))
                ),
                "unchanged_inputs": sum((d == 0 for d in ds)),
                "shorter_inputs": sum((d < 0 for d in ds)),
                "longer_inputs": sum((d > 0 for d in ds)),
            }
    save(out / "prompt_tokens.json", {"conditions": prompt_summary, "paired": paired})
    timeline = c.read(rp / "execution_timeline.json")
    stages = []
    paths = []
    live = c.read(c.OUTPUT / "live/complete.json")
    require(live["counts"]["post_attempts"] == len(a), "Completion POST total differs")
    for stage in timeline:
        require(len(stage["attempts"]) == 1, "Unexpected repeated stage")
        at = stage["attempts"][0]
        start = reader.timestamp(at["start"]["started_utc"])
        end = reader.timestamp(at["finished_utc"])
        review_path = c.OUTPUT / "live/batches" / stage["stage"] / "review.json"
        review = reader.checked_json(
            review_path, live["review_hashes"][c.ref(review_path)]
        )
        response_path = Path(at["attempt"]) / "model_check_response.json"
        response = reader.checked_json(
            response_path, review["availability_hashes"][c.ref(response_path)]
        )
        paths.extend([review_path, response_path])
        inds = [i for i, v in enumerate(meta) if v["stage"] == stage["stage"]]
        sel = np.isin(a[:, 0].astype(int), inds)
        stages.append(
            {
                "stage": stage["stage"],
                "phase": stage["population"],
                "start_utc": start.isoformat(),
                "finish_utc": end.isoformat(),
                "seconds": (end - start).total_seconds(),
                "predictions": len(inds),
                "POST_attempts": int(sel.sum()),
                "successful_responses": int((a[sel, 2] == 1).sum()),
                "workers": at["start"]["workers"],
                "peak_submitted": at["summary"]["peak_submitted"],
                "model_list_GETs": 1,
                "model_list_elapsed_s": response["elapsed_s"],
                "request_seconds": float(a[sel, 4].sum()),
                "backoff_seconds": float(a[sel, 5].sum()),
                "stopped": at["summary"]["stopped"],
                "errors": at["summary"]["errors"],
                "saved_review_mtime_utc": datetime.fromtimestamp(
                    review_path.stat().st_mtime, timezone.utc
                ).isoformat(),
            }
        )
    spans = [
        (
            reader.timestamp(s["start_utc"]).timestamp(),
            reader.timestamp(s["finish_utc"]).timestamp(),
        )
        for s in stages
    ]
    wall = wall_intervals(spans)
    mtime = (c.OUTPUT / "live/complete.json").stat().st_mtime
    wall.update(
        first_start_utc=stages[0]["start_utc"],
        last_finish_utc=stages[-1]["finish_utc"],
        filesystem_completion_mtime_utc=datetime.fromtimestamp(
            mtime, timezone.utc
        ).isoformat(),
        filesystem_indicated_end_to_end_s=mtime - spans[0][0],
        filesystem_indicated_final_tail_s=mtime - spans[-1][1],
        model_list_GETs=len(stages),
        model_list_elapsed_s=sum((s["model_list_elapsed_s"] for s in stages)),
        POST_per_attempt_second=len(a) / wall["attempt_seconds"],
        predictions_per_attempt_second=len(meta) / wall["attempt_seconds"],
        successful_responses_per_attempt_second=int((a[:, 2] == 1).sum())
        / wall["attempt_seconds"],
        summed_transport_over_attempt_time=float(a[:, 4].sum())
        / wall["attempt_seconds"],
        caution="Batch intervals include model check and local work. Between-attempt gaps include audits and other checks. No audit-only timer or embedded final-completion timestamp. Transport overlap ratio is not GPU utilization.",
    )
    save(
        out / "timing.json",
        {"wall": wall, "stages": stages, "receipt_bindings": c.bindings(paths)},
    )
    print(
        "Request, token and timing ledgers computed; counting study storage metadata only.",
        flush=True,
    )
    save(
        out / "storage.json",
        {
            "D_study": inventory(c.OUTPUT),
            "C_preparation": inventory(c.BASE),
            "excludes": "Other studies, shared caches, thesis, private backups and new Part 7 outputs; no allocated-block or peak-space estimate.",
        },
    )
    prior = c.read(rob / "original_coverage_bounds_repeats.json")
    tradeoff = {}
    for left, right in r.PAIRS:
        aa = prior["de_test"]["common"][left]["mean"]
        bb = prior["de_test"]["common"][right]["mean"]
        fp = aa["fp"] - bb["fp"]
        fn = aa["fn"] - bb["fn"]
        tradeoff[left + "_minus_" + right] = {
            "mean_extra_FP": fp,
            "mean_extra_FN": fn,
            "equal_cost_error_change": fp + fn,
            "illustrative_missed_hate_to_false_alarm_break_even": (
                fp / -fn if fp > 0 and fn < 0 else None
            ),
            "scope": "Observed cohort, FP cost normalized to 1; no chosen utility, prevalence transfer, confidence interval or deployment recommendation.",
        }
    save(out / "practical_tradeoffs.json", tradeoff)
    save(
        out / "synthesis_scores.json",
        {
            ph: {
                k: v
                for k, v in prior[ph].items()
                if k
                in (
                    "full_inputs",
                    "common_inputs",
                    "common",
                    "contrasts",
                    "full_roster_bounds",
                )
            }
            for ph in PHASES
        },
    )
    c.verify(contract["bindings"])
    save(
        out / "manifest.json",
        {
            "status": "part7_pending_independent_check",
            "created_utc": c.a.utc(),
            "run": c.ref(out),
            "outputs": c.bindings((p for p in out.rglob("*") if p.is_file())),
            "actual_Qwen_calls": 0,
            "D_writes": 0,
        },
    )
    print(
        json.dumps(
            {
                "run": c.ref(out),
                "POSTs": len(a),
                "status": "Resource ledger ready for independent check",
            }
        ),
        flush=True,
    )


def verified():
    done = c.read(BASE / "complete.json")
    require(
        done["status"] == "part7_resources_verified",
        "Independent resource check missing",
    )
    c.verify(done["bindings"])
    m = c.read(c.ROOT / done["manifest"])
    c.verify(m["outputs"])
    return (m, c.ROOT / m["run"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["prepare", "run"])
    args = parser.parse_args()
    with offline():
        prepare() if args.mode == "prepare" else run()

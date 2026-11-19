"""Policy resources."""

import os

os.environ.update(
    PYTHONDONTWRITEBYTECODE="1",
    OPENBLAS_NUM_THREADS="2",
    MKL_NUM_THREADS="2",
    OMP_NUM_THREADS="2",
    CUDA_VISIBLE_DEVICES="",
)
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import json
import math
import time
import numpy as np
from analysis.knowledge.policy import common as c
from analysis.knowledge.policy import reader as reader
from analysis.knowledge.policy import metrics as metrics

BASE = c.ROOT / "experiments/astra/complete_policy_part6_20261115_v1"
P3 = reader.BASE
P4 = c.ROOT / "experiments/astra/complete_policy_part4_20261115_v1"
P5 = c.ROOT / "experiments/astra/complete_policy_part5_20261115_v1"
RUN = None
NAMES = c.CONDITIONS
COLS = (
    "prediction",
    "vote",
    "turn",
    "attempt",
    "repair",
    "success",
    "elapsed_s",
    "backoff_s",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
)
IX = {k: i for i, k in enumerate(COLS)}


def load_run():
    global RUN
    if RUN is None:
        RUN = c.ROOT / c.read(P3 / "analysis_complete.json")["run"]
    return RUN


def save(name, value):
    path = BASE / name
    c.require(c.normal(path).is_relative_to(BASE), "Write outside new Part 6 directory")
    c.save(path, metrics.serial(value))


def rowsave(name, values):
    path = BASE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf8", newline="\n") as f:
        for v in values:
            f.write(
                json.dumps(metrics.serial(v), sort_keys=True, ensure_ascii=False, allow_nan=False)
                + "\n"
            )


def npzsave(name, **values):
    path = BASE / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as f:
        np.savez_compressed(f, **values)


def stats(values):
    x = np.asarray(values, float)
    c.require(x.ndim == 1 and np.isfinite(x).all(), "Invalid numerical vector")
    if not len(x):
        return dict(
            n=0, sum=0.0, mean=None, min=None, median=None, p90=None, p95=None, p99=None, max=None
        )
    return dict(
        n=len(x),
        sum=float(x.sum()),
        mean=float(x.mean()),
        min=float(x.min()),
        median=float(np.median(x)),
        p90=float(np.quantile(x, 0.9)),
        p95=float(np.quantile(x, 0.95)),
        p99=float(np.quantile(x, 0.99)),
        max=float(x.max()),
    )


def usage(metadata):
    u = metadata.get("usage") if isinstance(metadata, dict) else None
    if u is None:
        return [-1, -1, -1]
    c.require(isinstance(u, dict), "Unexpected usage structure")
    values = []
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        v = u.get(name)
        c.require(v is None or (type(v) is int and v >= 0), "Invalid usage counter")
        values.append(-1 if v is None else v)
    if min(values) >= 0:
        c.require(values[0] + values[1] == values[2], "Usage arithmetic differs")
    return values


def elapsed(value):
    return datetime.fromisoformat(value).timestamp()


def span_totals(spans):
    spans = sorted(spans)
    c.require(
        spans and all((math.isfinite(a) and math.isfinite(b) and (a <= b) for a, b in spans)),
        "Invalid wall span",
    )
    c.require(
        all((b <= cc for (_, b), (cc, _) in zip(spans, spans[1:]))),
        "Unexpected overlapping batch spans",
    )
    active = sum((b - a for a, b in spans))
    window = spans[-1][1] - spans[0][0]
    return dict(
        batch_attempt_s=active,
        first_to_last_attempt_s=window,
        between_batch_s=window - active,
        gaps_s=[cc - b for (_, b), (cc, _) in zip(spans, spans[1:])],
    )


def prepare():
    if (BASE / "analysis_contract.json").exists():
        c.verify(c.read(BASE / "analysis_contract.json")["inputs"])
        return
    c.require(c.sha(c.PLAN) == reader.PLAN_SHA, "Experiment plan changed")
    bound = {}
    for folder, filename in (
        (P3, "part3_complete.json"),
        (P4, "part4_complete.json"),
        (P5, "part5_complete.json"),
    ):
        receipt = c.read(folder / filename)
        for field in ("artifacts", "scripts", "reports"):
            c.verify(receipt.get(field, {}))
            bound.update(receipt.get(field, {}))
        bound.update(c.bindings([folder / filename]))
    paths = [
        c.PLAN,
        c.BASE / "schedule.json",
        c.BASE / "prompts.jsonl",
        c.SETUP / "budget.json",
        c.SETUP / "roster.jsonl",
        c.SETUP / "endpoints.json",
        c.OUTPUT / "live/complete.json",
        c.OUTPUT / "live/started.json",
        c.ROOT / "analysis/common/api.py",
        c.ROOT / "analysis/knowledge/policy/run.py",
        c.ROOT / "src/hsrag/retrieve.py",
        c.ROOT / "Thesis_Log/astra_result/complete_policy_work_plan_20261114_v2.md",
        c.ROOT / "Thesis_Log/astra_result/complete_policy_part1_protocol_20261114_v1.md",
        *list((c.OUTPUT / "invocations").glob("*/*.json")),
    ]
    bound.update(c.bindings(paths))
    bound.update(c.read(P3 / "reader_complete.json")["reader_files"])
    save(
        "analysis_contract.json",
        dict(
            status="part6_scope_before_resource_totals",
            created_utc=c.a.utc(),
            inputs=bound,
            prior_exposure="Parts 3–5 performance, context and robustness findings are known. Resource accounting is descriptive; no new hypothesis family or model selection.",
            units="Separate ordinary/repair logical requests, physical POST attempts, successful HTTP responses, usable inner answers, voted final answers and unique texts. A production-style voted answer uses three inner requests; three outer repeats multiply the evaluation budget.",
            tokens="Count server-reported successful-response usage, including schema-invalid replies and repair prompts. Missing or failed-attempt usage stays unknown. Check initial server tokens against all frozen reference prompts. No cache discount or price assumed.",
            timing="Saved client event durations include transport/response handling but exclude guard waits, intent/output-file writes, parsing, voting and queue waits. Summed vote/request-chain times are not elapsed experiment duration or GPU time. Use batch start/summary, invocation start/finish and progress records for wall components; label combined audit/preparation gaps and final-tail overhead.",
            storage="Read-only metadata inventories of this D study and its C preparation/setup only. Count logical bytes, files and folders by category, with a metadata digest. Do not rehash the full raw output tree, delete files, infer allocated bytes/peak usage or include other studies.",
            practical="Relate measured resources to the fixed quality/robustness evidence, including valid-answer coverage, exactness and recall trade-offs. Same-query paired request-chain differences are descriptive under this mixed 128-worker queue. No estimate of full-condition standalone speed or real-time retrieval savings.",
            utility="Optional explicit false-negative/false-positive cost ratios on saved English/German hate cohorts are illustrations, not an adopted cost function, fitted threshold or deployment recommendation. No single utility over the five outputs.",
            extrapolation="Per-1000 quantities are arithmetic normalizations at the recorded three-inner-vote settings. No claim that a single-response system has the same quality, that throughput scales linearly or that model-token counts measure electricity/GPU utilization.",
            next="Final SQ1 integration remains Part 7; separate writer/defence handover remains Part 8.",
            no_network=True,
            no_latex_edits=True,
            thesis_before=reader.snapshot_thesis(),
        ),
    )
    with (BASE / "session_before.md").open("xb") as f:
        f.write((c.ROOT / "Thesis_Log/astra_session.md").read_bytes())
    print("Part 6 scope and preservation snapshot recorded.", flush=True)


def extract_one(pair):
    i, row = pair
    path = c.OUTPUT / "live/items" / row["stamp"] / "result.json"
    raw = c.a.storage_path(path).read_bytes()
    c.require(
        hashlib.sha256(raw).hexdigest() == row["result_sha256"], "Changed result: " + row["stamp"]
    )
    v = json.loads(raw)
    events = []
    details = Counter()
    retry = Counter()
    vote_s = [0.0, 0.0, 0.0]
    logical = set()
    ordinary_inputs = []
    server_models = Counter()
    finish = Counter()
    for e in v["events"]:
        ok = e["action"] == "ok"
        repair = e["kind"] == "repair"
        c.require(
            e["action"] in ("ok", "retry") and e["kind"] in ("ordinary", "repair"), "Unknown event"
        )
        c.require(
            all((math.isfinite(e[k]) and e[k] >= 0 for k in ("elapsed_s", "backoff_elapsed_s"))),
            "Bad duration",
        )
        u = usage(e["model_response_metadata"]) if ok else [-1, -1, -1]
        events.append(
            [
                i,
                e["vote"],
                e["turn"],
                e["attempt"],
                int(repair),
                int(ok),
                e["elapsed_s"],
                e["backoff_elapsed_s"],
                *u,
            ]
        )
        vote_s[e["vote"]] += e["elapsed_s"] + e["backoff_elapsed_s"]
        if ok:
            logical.add((e["vote"], e["turn"]))
            md = e["model_response_metadata"]
            server_models[str(md["reported_model"])] += 1
            finish[str(md["finish_reason"])] += 1
            uu = md.get("usage") or {}
            for field in ("prompt_tokens_details", "completion_tokens_details"):
                value = uu.get(field)
                details[
                    field + "/" + ("absent" if value is None else json.dumps(value, sort_keys=True))
                ] += 1
            if not repair:
                ordinary_inputs.append(u[0])
        else:
            retry[e["response_class"]] += 1
    c.require(
        np.allclose(vote_s, v["latencies"], rtol=0, atol=1e-07), "Vote/request time discrepancy"
    )
    c.require(
        len(ordinary_inputs) == 3
        and len(logical) == 3 + v["repairs"]
        and (len(events) == v["accounting"]["post_attempts"]),
        "Accounting discrepancy",
    )
    c.require(
        v["n_valid"] == row["n_valid"] and v["repairs"] == row["repairs"], "Reader metadata changed"
    )
    item = dict(
        index=i,
        key=row["key"],
        condition=row["condition"],
        repeat=row["repeat"],
        stage=row["stage"],
        stamp=row["stamp"],
        result_sha256=row["result_sha256"],
        result_bytes=len(raw),
        n_valid=v["n_valid"],
        repairs=v["repairs"],
        vote_request_chain_s=vote_s,
        prediction_request_chain_s=sum(vote_s),
        ordinary_server_inputs=ordinary_inputs,
        events=len(events),
        usable_final=v["result"] is not None,
        final_reasoning_characters=None if v["result"] is None else len(v["result"]["reasoning"]),
    )
    return (item, events, dict(details), dict(retry), dict(server_models), dict(finish))


def extract():
    c.verify(c.read(BASE / "analysis_contract.json")["inputs"])
    c.require(c.read(BASE / "boundary_tests.json")["passed"], "Boundary tests missing")
    rows = [
        r
        for path in c.read(P3 / "reader_complete.json")["reader_files"]
        for r in c.read(c.ROOT / path)
    ]
    events = []
    meta = []
    details = Counter()
    retries = Counter()
    models = Counter()
    finishes = Counter()
    with (BASE / "prediction_resources.jsonl").open(
        "x", encoding="utf8", newline="\n"
    ) as f, ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, len(rows), 256):
            batch = list(
                pool.map(
                    extract_one, [(i, rows[i]) for i in range(start, min(start + 256, len(rows)))]
                )
            )
            for item, ev, dt, rt, md, fn in batch:
                f.write(
                    json.dumps(item, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
                )
                meta.append(item)
                events.extend(ev)
                details.update(dt)
                retries.update(rt)
                models.update(md)
                finishes.update(fn)
            if start // 5000 != (start + len(batch)) // 5000 or len(meta) == len(rows):
                print(
                    f"Offline request ledger: {len(meta)}/{len(rows)} saved predictions", flush=True
                )
    a = np.asarray(events, float)
    npzsave("event_resources.npz", events=a)
    save(
        "extraction_complete.json",
        dict(
            predictions=len(meta),
            events=len(events),
            columns=COLS,
            usage_detail_values=details,
            retry_classes=retries,
            reported_models=models,
            finish_reasons=finishes,
            output_bindings=c.bindings(
                [BASE / "prediction_resources.jsonl", BASE / "event_resources.npz"]
            ),
            original_result_hash_source=c.ref(P3 / "reader_complete.json"),
            new_requests=0,
            local_read_workers=8,
        ),
    )


def group_stats(meta, events, indices):
    mask = np.isin(events[:, IX["prediction"]].astype(int), indices)
    ev = events[mask]
    ok = ev[:, IX["success"]] == 1
    repair = ev[:, IX["repair"]] == 1
    records = [meta[i] for i in indices]
    valid = sum((r["usable_final"] for r in records))
    out = dict(
        predictions=len(indices),
        unique_inputs=len({r["key"] for r in records}),
        valid_final=valid,
        null_final=len(indices) - valid,
        valid_inner=sum((r["n_valid"] for r in records)),
        repaired_predictions=sum((r["repairs"] > 0 for r in records)),
        ordinary_logical_calls=int((ok & ~repair).sum()),
        repair_logical_calls=int((ok & repair).sum()),
        successful_responses=int(ok.sum()),
        physical_POST_attempts=len(ev),
        retry_attempts=int((ev[:, IX["attempt"]] > 0).sum()),
        failed_attempts=int((~ok).sum()),
        summed_transport_s=float(ev[:, IX["elapsed_s"]].sum()),
        backoff_s=float(ev[:, IX["backoff_s"]].sum()),
        prediction_request_chain_s=stats([r["prediction_request_chain_s"] for r in records]),
        vote_request_chain_s=stats([x for r in records for x in r["vote_request_chain_s"]]),
        successful_response_s=stats(ev[ok, IX["elapsed_s"]]),
        ordinary_success_s=stats(ev[ok & ~repair, IX["elapsed_s"]]),
        repair_success_s=stats(ev[ok & repair, IX["elapsed_s"]]),
        failed_attempt_s=stats(ev[~ok, IX["elapsed_s"]]),
        usage={},
    )
    for kind, on in [("all_success", ok), ("ordinary", ok & ~repair), ("repair", ok & repair)]:
        sub = ev[on]
        values = {}
        for col in ("prompt_tokens", "completion_tokens", "total_tokens"):
            v = sub[:, IX[col]]
            known = v >= 0
            values[col] = dict(
                known=int(known.sum()), unknown=int((~known).sum()), **stats(v[known])
            )
        out["usage"][kind] = values
    token = out["usage"]["all_success"]
    out["per_voted_answer"] = dict(
        successful_requests=ok.sum() / len(indices),
        prompt_tokens=token["prompt_tokens"]["sum"] / len(indices),
        completion_tokens=token["completion_tokens"]["sum"] / len(indices),
        total_tokens=token["total_tokens"]["sum"] / len(indices),
    )
    out["per_valid_voted_answer"] = (
        None
        if valid == 0
        else dict(
            successful_requests=ok.sum() / valid,
            prompt_tokens=token["prompt_tokens"]["sum"] / valid,
            completion_tokens=token["completion_tokens"]["sum"] / valid,
            total_tokens=token["total_tokens"]["sum"] / valid,
        )
    )
    return out


def summarize():
    receipt = c.read(BASE / "extraction_complete.json")
    c.verify(receipt["output_bindings"])
    meta = list(c.rows(BASE / "prediction_resources.jsonl"))
    a = np.load(BASE / "event_resources.npz", allow_pickle=False)["events"]
    roster = {r["key"]: r for r in c.rows(c.SETUP / "roster.jsonl")}
    endpoints = c.read(c.SETUP / "endpoints.json")
    groups = {"all": group_stats(meta, a, list(range(len(meta))))}
    for cond in NAMES:
        groups["condition/" + cond] = group_stats(
            meta, a, [r["index"] for r in meta if r["condition"] == cond]
        )
    for lang in ("en", "de"):
        for cond in NAMES:
            groups["language/" + lang + "/" + cond] = group_stats(
                meta,
                a,
                [
                    r["index"]
                    for r in meta
                    if r["condition"] == cond and roster[r["key"]]["lang"] == lang
                ],
            )
    endpoint_costs = {}
    for endpoint, ep in endpoints.items():
        encoded = np.load(load_run() / endpoint / "encoded.npz", allow_pickle=False)
        keys = {k for k, on in zip(ep["keys"], encoded["common"]) if on}
        endpoint_costs[endpoint] = {
            cond: group_stats(
                meta, a, [r["index"] for r in meta if r["condition"] == cond and r["key"] in keys]
            )
            for cond in NAMES
        }
    save(
        "resources.json",
        dict(
            groups=groups,
            endpoint_common_costs=endpoint_costs,
            caution="All five fields are generated together. Endpoint cohorts overlap; do not sum their requests or costs. Failed-attempt usage is unknown, never assigned zero.",
        ),
    )
    contexts = {r["key"]: r for r in c.rows(P4 / "context_features.jsonl")}
    values = defaultdict(list)
    for row in meta:
        values[row["key"], row["condition"]].extend(row["ordinary_server_inputs"])
    records = []
    for key in sorted(contexts):
        for cond in NAMES:
            vals = values.pop((key, cond))
            known = [x for x in vals if x >= 0]
            unique = sorted(set(known))
            c.require(len(vals) == 9, "Wrong prompt-use count")
            reference = contexts[key]["settings"][cond]["tokens"]
            records.append(
                dict(
                    key=key,
                    condition=cond,
                    reference_tokens=reference,
                    server_values=unique,
                    known=len(known),
                    server_tokens=unique[0] if len(unique) == 1 and len(known) == 9 else None,
                    source=roster[key]["source"],
                    lang=roster[key]["lang"],
                )
            )
    c.require(not values, "Unmatched initial prompt usage")
    rowsave("prompt_resources.jsonl", records)
    summary = {
        cond: dict(
            reference=stats([r["reference_tokens"] for r in records if r["condition"] == cond]),
            server=stats(
                [
                    r["server_tokens"]
                    for r in records
                    if r["condition"] == cond and r["server_tokens"] is not None
                ]
            ),
            mismatch_keys=[
                r["key"]
                for r in records
                if r["condition"] == cond and r["server_tokens"] != r["reference_tokens"]
            ],
        )
        for cond in NAMES
    }
    save(
        "prompt_tokens.json",
        dict(
            summary=summary, stable_all_nine=sum((r["server_tokens"] is not None for r in records))
        ),
    )
    bykey = {(r["key"], r["condition"], r["repeat"]): r for r in meta}
    pair = []
    for left, right in ((1, 0), (2, 0), (2, 1)):
        for endpoint in ("all", *endpoints):
            if endpoint == "all":
                keys = list(roster)
            else:
                ar = np.load(load_run() / endpoint / "encoded.npz", allow_pickle=False)
                keys = [k for k, on in zip(endpoints[endpoint]["keys"], ar["common"]) if on]
            deltas = []
            for key in keys:
                a0 = np.mean(
                    [bykey[key, NAMES[left], rr]["prediction_request_chain_s"] for rr in range(3)]
                )
                b0 = np.mean(
                    [bykey[key, NAMES[right], rr]["prediction_request_chain_s"] for rr in range(3)]
                )
                deltas.append(a0 - b0)
            pair.append(
                dict(
                    endpoint=endpoint,
                    left=NAMES[left],
                    right=NAMES[right],
                    inputs=len(keys),
                    request_chain_difference_s=stats(deltas),
                    fraction_left_faster=float((np.array(deltas) < 0).mean()),
                    caution="Mean three recorded request-chain sums per input. Not standalone runtime, user-facing latency or a causal server speed benchmark.",
                )
            )
    save("paired_request_time.json", pair)
    print("Condition, language, matched-cohort and prompt resource totals saved.", flush=True)


def timing():
    process = c.read(P5 / "process_evidence.json")
    meta = list(c.rows(BASE / "prediction_resources.jsonl"))
    ev = np.load(BASE / "event_resources.npz")["events"]
    stages = []
    bindings = {}
    model_get = []
    context = []
    invocations = []
    for path in sorted((c.OUTPUT / "invocations").iterdir()):
        start = c.read(path / "start.json")
        finish = c.read(path / "finished.json")
        c.require(finish["status"] == "complete", "Incomplete invocation")
        c.require(
            finish["completion_sha256"] == c.sha(c.OUTPUT / "live/complete.json"),
            "Invocation completion mismatch",
        )
        progress = [
            dict(stage=s.stem.removesuffix("_progress"), **c.read(s))
            for s in sorted(path.glob("*_progress.json"))
        ]
        invocations.append(dict(path=c.ref(path), start=start, finish=finish, progress=progress))
        bindings.update(
            c.bindings([path / "start.json", path / "finished.json", *path.glob("*_progress.json")])
        )
    c.require(
        len(invocations) == 1 and invocations[0]["start"]["already_audited"] == 0,
        "Unexpected resume timeline needs explicit handling",
    )
    for stage in process["stages"]:
        name = stage["stage"]
        attempts = stage["attempts"]
        c.require(len(attempts) == 1, "Unexpected multiple attempts")
        attempt = attempts[0]
        begin = elapsed(attempt["start"]["started_utc"])
        end = elapsed(attempt["summary"]["finished_utc"])
        review = c.read(c.OUTPUT / "live/batches" / name / "review.json")
        starts = [c.ROOT / k for k in review["availability_hashes"] if k.endswith("/start.json")]
        c.require(len(starts) == 1, "Multiple stage starts")
        folder = starts[0].parent
        response = c.read(folder / "model_check_response.json")
        context.append(c.read(folder / "context_metadata.json"))
        model_get.append(response["elapsed_s"])
        bindings.update(review["availability_hashes"])
        inds = [r["index"] for r in meta if r["stage"] == name]
        sel = np.isin(ev[:, IX["prediction"]].astype(int), inds)
        stages.append(
            dict(
                stage=name,
                start_utc=attempt["start"]["started_utc"],
                finish_utc=attempt["summary"]["finished_utc"],
                attempt_s=end - begin,
                predictions=len(inds),
                POST_attempts=int(sel.sum()),
                successful_responses=int(ev[sel, IX["success"]].sum()),
                transport_s=float(ev[sel, IX["elapsed_s"]].sum()),
                backoff_s=float(ev[sel, IX["backoff_s"]].sum()),
                workers=attempt["start"]["workers"],
                peak_submitted=attempt["summary"]["peak_submitted"],
                model_list_GET_s=response["elapsed_s"],
                review_mtime_utc=datetime.fromtimestamp(
                    (folder.parent / "review.json").stat().st_mtime, timezone.utc
                ).isoformat(),
            )
        )
    spans = [(elapsed(s["start_utc"]), elapsed(s["finish_utc"])) for s in stages]
    wall = span_totals(spans)
    inv = invocations[0]
    begin = elapsed(inv["start"]["started_utc"])
    end = elapsed(inv["finish"]["finished_utc"])
    last_progress = inv["progress"][-1]["elapsed_this_invocation_s"]
    wall.update(
        invocation_start_utc=inv["start"]["started_utc"],
        invocation_finish_utc=inv["finish"]["finished_utc"],
        invocation_s=end - begin,
        initial_before_first_batch_s=spans[0][0] - begin,
        after_last_batch_s=end - spans[-1][1],
        progress_elapsed_at_last_batch_audit_s=last_progress,
        finalization_after_last_progress_approx_s=end - begin - last_progress,
        main_queue_to_all_batches_audited_approx_s=last_progress,
        model_list_GETs=len(stages),
        model_list_GET_s=sum(model_get),
        successful_responses_per_batch_attempt_second=float(ev[:, IX["success"]].sum())
        / wall["batch_attempt_s"],
        predictions_per_batch_attempt_second=len(meta) / wall["batch_attempt_s"],
        predictions_per_invocation_second=len(meta) / (end - begin),
        transport_seconds_per_batch_attempt_second=float(ev[:, IX["elapsed_s"]].sum())
        / wall["batch_attempt_s"],
        time_partition_check_s=wall["batch_attempt_s"]
        + wall["between_batch_s"]
        + (spans[0][0] - begin)
        + (end - spans[-1][1]),
        caution="Inter-batch and final-tail times include checks/audits/local work. No dedicated audit-only timer. Last-progress uses a monotonic origin just after invocation start, so its absolute-time split is approximate by the small origin offset. Transport overlap is not GPU utilization.",
    )
    c.require(
        abs(wall["time_partition_check_s"] - wall["invocation_s"]) < 1e-06,
        "Wall-time partition mismatch",
    )
    save(
        "timing.json",
        dict(
            wall=wall,
            stages=stages,
            invocations=invocations,
            context_metadata=context,
            bindings=bindings,
        ),
    )
    print("Recorded batch and invocation timing reconciled.", flush=True)


def inventory_root(root):
    root = c.normal(root)
    allowed = (c.OUTPUT, c.BASE, c.SETUP)
    c.require(root in allowed and root.is_dir(), "Inventory root not allowlisted")
    scanned = c.a.storage_path(root)
    stack = [str(scanned)]
    groups = defaultdict(Counter)
    types = defaultdict(Counter)
    directory_count = 0
    files = 0
    logical = 0
    h = hashlib.sha256()
    while stack:
        folder = stack.pop()
        directory_count += 1
        with os.scandir(folder) as it:
            entries = sorted(it, key=lambda x: x.name)
        children = []
        for entry in entries:
            c.require(
                not entry.is_symlink() and (not entry.is_junction()), "Linked entry in inventory"
            )
            if entry.is_dir(follow_symlinks=False):
                children.append(entry.path)
                continue
            c.require(entry.is_file(follow_symlinks=False), "Unexpected filesystem object")
            st = entry.stat(follow_symlinks=False)
            rel = os.path.relpath(entry.path, str(scanned)).replace("\\", "/")
            parts = rel.split("/")
            group = (
                "/".join(parts[:2])
                if parts[0] == "live" and len(parts) > 2
                else parts[0] if len(parts) > 1 else "root_files"
            )
            kind = (
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
            groups[group].update(files=1, bytes=st.st_size)
            types[kind].update(files=1, bytes=st.st_size)
            files += 1
            logical += st.st_size
            h.update(
                (
                    json.dumps(
                        [rel, st.st_size, st.st_mtime_ns], ensure_ascii=False, separators=(",", ":")
                    )
                    + "\n"
                ).encode("utf8")
            )
            if files % 100000 == 0:
                print(f"Storage metadata: {root.name}: {files:,} files", flush=True)
        stack.extend(reversed(children))
    return dict(
        root=root.as_posix(),
        scanned_utc=c.a.utc(),
        files=files,
        folders_including_root=directory_count,
        logical_bytes=logical,
        groups=groups,
        by_basename_or_suffix=types,
        path_size_mtime_sha256=h.hexdigest(),
        method="Sorted depth-first regular-file metadata census. No file-body reads by this inventory. Logical bytes only; not allocation, peak usage or a replacement integrity audit.",
    )


def storage():
    save(
        "storage.json",
        dict(
            raw_D=inventory_root(c.OUTPUT),
            execution_preparation_C=inventory_root(c.BASE),
            design_setup_C=inventory_root(c.SETUP),
            excludes="Other studies, shared tokenizer/model caches, both thesis copies, backups and this new Part 6 output directory. No files moved or deleted.",
        ),
    )
    print("Read-only storage inventories complete.", flush=True)


if __name__ == "__main__":
    a = argparse.ArgumentParser()
    a.add_argument("mode", choices=["prepare", "extract", "summarize", "timing", "storage"])
    arg = a.parse_args()
    with c.offline():
        globals()[arg.mode]()
